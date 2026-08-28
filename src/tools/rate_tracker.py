"""
tools/rate_tracker.py
Pelacak pemakaian rate limit Groq (per-menit DAN per-hari, rolling window)
untuk mencegah klik "Jalankan Diskusi Agent" pas sudah mepet/lewat limit.

PENTING - PERUBAHAN PENYIMPANAN (fix "reset ke 0 saat refresh browser"):
Versi sebelumnya menyimpan log di st.session_state - itu SEBABNYA angka
Groq kembali ke 0 setiap refresh browser penuh (session_state memang
direset tiap sesi baru), padahal limit Groq itu sebenarnya PER API KEY,
bukan per sesi browser - jadi seharusnya persist lintas refresh, bahkan
lintas user kalau semua orang pakai API key yang sama di 1 deployment,
sama seperti cara Tavily dibaca (get_usage_this_month() query akun Tavily
asli, bukan data lokal per-sesi).

Sekarang log disimpan ke FILE lokal (.groq_rate_log.jsonl di root project),
bukan session_state - supaya bertahan lintas refresh & lintas sesi.

FIX KEDUA (ditemukan setelah user ganti akun/API key tapi dashboard masih
tampil penuh): entri log SEBELUMNYA tidak menyimpan identitas API key sama
sekali - cuma [timestamp, jumlah_token]. Akibatnya kalau user ganti
GROQ_API_KEY ke akun lain, tracker tidak tahu itu akun berbeda dan tetap
menjumlahkan sisa entri lama (punya akun sebelumnya) ke dalam rolling
window 24 jam yang sama, seolah itu pemakaian akun yang sekarang aktif.
Sekarang tiap entri menyimpan FINGERPRINT api key (hash pendek, BUKAN key
mentahnya - supaya tidak nyimpen credential asli di file), dan semua fungsi
baca/hitung memfilter entri HANYA milik fingerprint key yang sedang aktif.
Entri milik key lain tetap disimpan di file (tidak dihapus - kalau user
balik pakai key lama, riwayatnya masih ada), tapi tidak ikut dihitung ke
kuota key yang sedang dipakai sekarang.

JUJUR SOAL AKURASI/KETERBATASAN:
- Angka token Groq yang dicatat adalah ANGKA ASLI dari response API
  (usage_metadata), BUKAN tebakan/estimasi.
- TPD dihitung sebagai ROLLING 24 JAM dari waktu lokal (bukan reset
  jam-tetap seperti kemungkinan dilakukan Groq di sisi server) - estimasi
  yang cukup untuk peringatan dini, bukan angka yang dijamin identik
  dengan dashboard resmi Groq.
- File ini di filesystem LOKAL proses yang sedang jalan - bertahan lintas
  refresh browser & lintas sesi selama proses/container yang sama masih
  hidup, TAPI akan reset kalau container di-redeploy/restart penuh (wajar
  untuk skala tool pengembangan/demo, bukan sistem produksi enterprise).
- Tracking sekarang per API-key-fingerprint, TAPI masih dalam file yang
  sama - kalau butuh isolasi lebih ketat (misal per file terpisah per key)
  atau persist lintas restart deploy, upgrade berikutnya adalah storage
  eksternal (database kecil/Redis), di luar cakupan perbaikan ini.
"""

import hashlib
import json
import os
import threading
import time

from ..config import GROQ_TPM_LIMIT, GROQ_TPD_LIMIT, TAVILY_MONTHLY_LIMIT

_WINDOW_DETIK = 60
_WINDOW_HARIAN_DETIK = 24 * 60 * 60  # 24 jam - dipakai untuk TPD

# File log persisten - taruh di root project (2 level di atas tools/), 1
# baris JSON per entri: [timestamp_epoch, jumlah_token, fingerprint_key].
# Append-only saat ditulis, tapi entri kadaluarsa (>24 jam, LINTAS SEMUA
# key) dibuang tiap kali ada pencatatan baru supaya file tidak numpuk tak
# terbatas.
_FILE_LOG_GROQ = os.path.join(os.path.dirname(__file__), "..", "..", ".groq_rate_log.jsonl")

# Lock antar-thread (tiap sesi Streamlit jalan di thread berbeda dalam 1
# proses) - mencegah tulis-bareng ke file yang sama saling menimpa/korup.
_LOCK = threading.Lock()


def _fingerprint_api_key(api_key: str) -> str:
    """Hash pendek dari API key - dipakai untuk MEMBEDAKAN akun tanpa
    menyimpan credential asli di file log (menghindari risiko kebocoran
    kalau file ini kebaca/ke-commit tidak sengaja)."""
    if not api_key:
        return "unknown"
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()[:16]


def _api_key_aktif() -> str:
    """Ambil API key yang sedang dipakai sekarang, dari secrets/env var yang
    sama seperti llm_client.py. Import lokal (bukan di top-level) untuk
    menghindari kemungkinan circular import antar module."""
    from ..llm_client import _get_secret
    return _get_secret("GROQ_API_KEY", "") or ""


def _baca_log_groq() -> list:
    """Baca SEMUA entri dari file log (lintas key), masing-masing berupa
    (timestamp, jumlah_token, fingerprint_key). Return list kosong kalau
    file belum ada atau rusak/tidak bisa dibaca (fail gracefully, jangan
    crash dashboard cuma gara-gara file log). Entri format LAMA (2 elemen,
    tanpa fingerprint) tetap dibaca dan diberi fingerprint "unknown" supaya
    tidak crash saat migrasi dari versi sebelumnya, tapi otomatis TIDAK akan
    cocok dengan fingerprint key aktif manapun (jadi tidak salah dihitung)."""
    if not os.path.exists(_FILE_LOG_GROQ):
        return []
    entri = []
    try:
        with open(_FILE_LOG_GROQ, "r", encoding="utf-8") as f:
            for baris in f:
                baris = baris.strip()
                if not baris:
                    continue
                try:
                    parsed = json.loads(baris)
                    if len(parsed) == 3:
                        t, n, fp = parsed
                    else:
                        t, n = parsed
                        fp = "unknown"  # entri lama dari sebelum fix ini
                    entri.append((float(t), int(n), str(fp)))
                except Exception:
                    continue  # baris korup/tidak valid - lewati, jangan gagal total
    except Exception:
        return []
    return entri


def _tulis_log_groq(entri: list):
    """Timpa seluruh isi file dengan entri yang sudah di-prune. Gagal tulis
    (misal masalah izin filesystem) diam saja - dashboard tetap jalan tanpa
    tracking sesaat, lebih baik daripada bikin seluruh app crash."""
    try:
        with open(_FILE_LOG_GROQ, "w", encoding="utf-8") as f:
            for t, n, fp in entri:
                f.write(json.dumps([t, n, fp]) + "\n")
    except Exception:
        pass


def catat_token_groq(jumlah_token: int):
    """Catat pemakaian token Groq (dari usage_metadata response asli, bukan
    tebakan) ke file persisten - bertahan lintas refresh browser. Setiap
    entri ditandai fingerprint API key yang aktif SAAT INI, supaya ganti
    akun tidak tercampur dengan pemakaian akun lain."""
    if not jumlah_token:
        return
    fp = _fingerprint_api_key(_api_key_aktif())
    with _LOCK:
        entri = _baca_log_groq()
        entri.append((time.time(), jumlah_token, fp))
        # Buang entri yang sudah lewat window TERBESAR (24 jam, dipakai TPD)
        # supaya file tidak numpuk tak terbatas, tapi data untuk TPM (60
        # detik) tetap tersedia karena masih dalam 24 jam terakhir. Ini
        # dilakukan LINTAS SEMUA key (bukan cuma key aktif), supaya riwayat
        # key lain yang masih dalam 24 jam tetap tersimpan (jaga-jaga kalau
        # user balik pakai key itu lagi).
        batas = time.time() - _WINDOW_HARIAN_DETIK
        entri = [(t, n, fp2) for (t, n, fp2) in entri if t >= batas]
        _tulis_log_groq(entri)


def token_groq_terpakai_60_detik() -> int:
    """Total token Groq yang tercatat dalam 60 detik terakhir (rolling window),
    HANYA untuk API key yang sedang aktif sekarang - dipakai untuk estimasi
    TPM (Token Per Minute)."""
    fp_aktif = _fingerprint_api_key(_api_key_aktif())
    entri = _baca_log_groq()
    batas = time.time() - _WINDOW_DETIK
    return sum(n for (t, n, fp) in entri if t >= batas and fp == fp_aktif)


def token_groq_terpakai_24_jam() -> int:
    """Total token Groq yang tercatat dalam 24 jam terakhir (rolling window),
    HANYA untuk API key yang sedang aktif sekarang - dipakai untuk estimasi
    TPD (Token Per Day)."""
    fp_aktif = _fingerprint_api_key(_api_key_aktif())
    entri = _baca_log_groq()
    batas = time.time() - _WINDOW_HARIAN_DETIK
    return sum(n for (t, n, fp) in entri if t >= batas and fp == fp_aktif)


def sisa_kuota_groq() -> int:
    """Sisa kuota TPM (per-menit), untuk API key yang aktif sekarang."""
    return max(0, GROQ_TPM_LIMIT - token_groq_terpakai_60_detik())


def sisa_kuota_groq_harian() -> int:
    """Sisa kuota TPD (per-hari, estimasi rolling 24 jam), untuk API key
    yang aktif sekarang."""
    return max(0, GROQ_TPD_LIMIT - token_groq_terpakai_24_jam())


def perkiraan_detik_sampai_reset_groq() -> float:
    """Perkiraan detik sampai entri TERTUA (milik key aktif) di window
    60-detik lepas (artinya sebagian kuota TPM mulai 'bebas' lagi). 0 kalau
    tidak ada entri dalam 60 detik terakhir yang perlu ditunggu.

    Filter SENGAJA dibatasi ke entri dalam window 60 detik saja (bukan ambil
    dari seluruh log 24 jam) - supaya tidak salah menghitung "detik sampai
    reset TPM" pakai entri berjam-jam lalu yang sudah tidak relevan untuk TPM.
    Juga difilter ke fingerprint key aktif saja, konsisten dengan fungsi lain."""
    fp_aktif = _fingerprint_api_key(_api_key_aktif())
    entri = _baca_log_groq()
    batas_tpm = time.time() - _WINDOW_DETIK
    entri_dalam_window_tpm = [(t, n, fp) for (t, n, fp) in entri if t >= batas_tpm and fp == fp_aktif]
    if not entri_dalam_window_tpm:
        return 0.0
    entri_tertua_waktu = min(t for (t, n, fp) in entri_dalam_window_tpm)
    sisa = _WINDOW_DETIK - (time.time() - entri_tertua_waktu)
    return max(0.0, sisa)


def groq_boleh_jalan(perkiraan_kebutuhan_token: int = 1500) -> bool:
    """Cek kasar: apakah cukup ruang di rolling window 60 detik (TPM) DAN
    rolling window 24 jam (TPD) untuk 1 kali call LLM lagi, untuk API key
    yang aktif sekarang.
    `perkiraan_kebutuhan_token` default 1500 - perkiraan kasar untuk 1
    giliran agent dengan konteks Tavily, BUKAN patokan presisi."""
    return (
        sisa_kuota_groq() >= perkiraan_kebutuhan_token
        and sisa_kuota_groq_harian() >= perkiraan_kebutuhan_token
    )


def tavily_boleh_jalan(pemakaian_bulan_ini: int) -> bool:
    return pemakaian_bulan_ini < TAVILY_MONTHLY_LIMIT