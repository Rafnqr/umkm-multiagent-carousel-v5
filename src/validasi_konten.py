"""
validasi_konten.py
Validasi PROGRAMATIK (bukan LLM lagi) untuk caption final LocalizationAgent.

KENAPA INI PERLU: instruksi di system prompt ("WAJIB Bahasa Indonesia",
"WAJIB narasi Hook-Value-CTA") membantu, tapi LLM tetap probabilistik - dari
log nyata, kadang hasilnya benar (Bahasa Indonesia + narasi mengalir), kadang
meleset (Bahasa Inggris, atau tetap berbentuk daftar spesifikasi baris-per-
baris). Fungsi di sini jadi lapisan pengaman KODE, bukan cuma andalkan LLM
"berjanji patuh" - dipanggil orchestrator.py setelah LocalizationAgent
selesai, SEBELUM caption ditampilkan ke user. Kalau gagal, orchestrator akan
minta LocalizationAgent.perbaiki_format() merevisi (lihat
config.LOCALIZATION_FORMAT_MAX_RETRY).

Heuristik di sini SENGAJA sederhana (bukan model bahasa lagi) - tujuannya
cuma jaring pengaman kasar untuk 2 pola kegagalan yang SUDAH TERBUKTI terjadi,
bukan validator sempurna untuk segala kasus.
"""

import re

_STOPWORD_ID = {
    "yang", "dan", "untuk", "dengan", "ini", "itu", "atau", "juga", "akan",
    "tidak", "ada", "kamu", "kami", "kita", "produk", "bisa", "dari", "ke",
    "di", "pada", "sudah", "saja", "coba", "sekarang", "karena", "jadi",
    "lebih", "masih", "cocok", "aman", "tanpa", "hari", "setiap",
}
_STOPWORD_EN = {
    "the", "and", "for", "with", "this", "that", "or", "also", "will",
    "not", "there", "you", "we", "product", "can", "from", "to", "in",
    "on", "already", "only", "try", "now", "because", "so", "more",
    "still", "suitable", "safe", "without", "day", "every", "is", "are",
    "your", "our", "it",
}


def validasi_bahasa_indonesia(teks: str) -> tuple:
    """Heuristik kasar: hitung kemunculan stopword umum Indonesia vs Inggris
    di badan caption (bukan hashtag). Return (ok: bool, alasan: str).

    Ini BUKAN language-detector akurat (tidak pakai library khusus supaya
    tidak nambah dependency), tapi cukup untuk menangkap kasus jelas seperti
    caption yang KESELURUHANNYA Bahasa Inggris (skor Inggris jauh lebih
    tinggi) - kasus yang sudah terbukti terjadi di log."""
    badan = re.split(r"\n?HASHTAG\s*:", teks, maxsplit=1)[0]
    kata = re.findall(r"[a-zA-Z]+", badan.lower())
    if len(kata) < 8:
        # Terlalu pendek untuk dinilai heuristik kata - jangan false-positive,
        # biarkan lolos (kasus kosong/pendek ditangani validator lain).
        return True, ""

    skor_id = sum(1 for k in kata if k in _STOPWORD_ID)
    skor_en = sum(1 for k in kata if k in _STOPWORD_EN)

    if skor_en > skor_id and skor_en >= 3:
        return False, (
            f"Terdeteksi kemungkinan besar caption berbahasa Inggris "
            f"(indikator kata umum Inggris={skor_en} vs Indonesia={skor_id}). "
            "Tulis ulang SELURUH caption dalam Bahasa Indonesia."
        )
    return True, ""


def validasi_struktur_hook_value_cta(teks: str) -> tuple:
    """Heuristik kasar untuk mendeteksi 2 pola gagal yang SUDAH TERBUKTI:
    (1) caption kosong/nyaris kosong, (2) caption berbentuk daftar
    spesifikasi baris-per-baris (bukan narasi Hook-Value-CTA yang mengalir).
    Return (ok: bool, alasan: str)."""
    badan = re.split(r"\n?HASHTAG\s*:", teks, maxsplit=1)[0].strip()

    if len(badan) < 20:
        return False, "Caption kosong atau terlalu pendek untuk dianggap valid."

    baris_isi = [b.strip() for b in badan.split("\n") if b.strip()]
    if not baris_isi:
        return False, "Caption tidak punya baris isi sama sekali."

    # Indikasi "spec-sheet list": banyak baris pendek (<=8 kata) yang
    # masing-masing berdiri sendiri sebagai kalimat lengkap (diawali huruf
    # kapital, diakhiri titik), TANPA ada paragraf yang menyatukan beberapa
    # kalimat jadi satu blok mengalir (dipisah baris kosong ganda "\n\n").
    jumlah_paragraf = len([p for p in badan.split("\n\n") if p.strip()])
    baris_pendek_mandiri = sum(
        1 for b in baris_isi
        if len(b.split()) <= 8 and b.endswith((".", "!", "?"))
    )
    rasio_baris_pendek = baris_pendek_mandiri / len(baris_isi)

    if jumlah_paragraf <= 1 and rasio_baris_pendek >= 0.6 and len(baris_isi) >= 3:
        return False, (
            "Caption terdeteksi berbentuk daftar spesifikasi baris-per-baris "
            "(tiap baris kalimat pendek berdiri sendiri, tidak ada paragraf "
            "yang mengalir/menyatu). Tulis ulang sebagai narasi Hook-Value-CTA: "
            "hook 2 baris, lalu value dalam paragraf-paragraf yang saling "
            "terhubung (dipisah baris kosong), bukan daftar terpisah."
        )

    return True, ""
