"""
visual_pipeline.py
Orkestrasi Visual Content Generator - CAROUSEL 3 slide (Cover/Isi/Penutup)
yang JALAN SEKALI setelah diskusi 5 agent selesai (caption final Hook-Value-
CTA sudah ada), BUKAN bagian dari loop diskusi utama.

Peran tiap slide:
  - Cover: representasi visual dari Hook - nama produk + angle penarik perhatian
  - Isi: mendukung bagian Value - fokus SATU manfaat produk saja
  - Penutup: mendukung CTA - ringkasan + ajakan

Foto produk ASLI (kalau user upload) TIDAK PERNAH diubah/di-crop - dia
ditempel sebagai "kartu" di atas background dekoratif AI-generate (lihat
image_compositor.render_kartu_dengan_latar_ai). AI TIDAK PERNAH menggambar
ulang produknya sendiri - HANYA latar belakang, yang diminta menyesuaikan
gaya/warna produk asli lewat deskripsi vision (lihat
_susun_prompt_latar_ai). Kalau TIDAK ada upload, semua 3 slide full
AI-generate dari deskripsi (dengan deskriptor gaya yang SAMA di ketiga
prompt supaya konsisten - Pollinations tidak punya "ingatan" antar
pemanggilan, jadi konsistensi HARUS dipaksa lewat teks prompt yang
eksplisit).

Rasio 4:5 (portrait) - standar feed/carousel modern, konten penting selalu
diposisikan di tengah (bukan mepet tepi) supaya aman dari crop thumbnail.

CATATAN PENTING - BACKGROUND HARUS BENAR-BENAR KOSONG DARI OBJEK:
karena foto produk ASLI ditempel di atas background AI, background ITU
SENDIRI tidak boleh menggambar objek/prop apapun (termasuk pedestal/podium/
silinder dekoratif) - kalau tidak, hasilnya tumpang tindih dengan foto asli
atau prop asing yang mengganggu komposisi. Ini ditangani 2 lapis:
1. Prompt background (_susun_prompt_latar_ai + _klausa_larangan_objek)
   eksplisit melarang kata bentuk objek/prop, bukan cuma "jangan gambar
   produk" secara umum (text-to-image model buruk menaati negasi halus).
2. Verifikasi ulang via vision (_cek_latar_bersih_dari_objek) SETELAH
   background di-generate, sebelum dipakai compositing - retry kalau masih
   terdeteksi ada objek/prop.
"""

import random
import re
import time

from .config import VISUAL_MAX_RETRY, VISUAL_RETRY_DELAY_DETIK, DESIGN_TEXT_MAX_RETRY
from .llm_client import call_llm
from .parsing_utils import extract_verdict, extract_field
from .tools.pollinations import generate_image
from .tools.vision_describe import describe_image
from .tools.image_compositor import render_desain, render_kartu_dengan_latar_ai, deteksi_warna_dominan_produk

_POLA_KARAKTER_AMAN = re.compile(r"[^a-zA-Z0-9\s.,!?%\-()&'\"×xX/:]")

# Pola label field lain - dipakai _potong_di_label_berikutnya untuk memotong
# nilai field yang "bocor" menangkap label field berikutnya (lihat bug nyata
# di docstring fungsi tsb).
_POLA_LABEL_FIELD_LAIN = re.compile(
    r"\s*(COVER_HEADLINE|COVER_SUBHEADLINE|ISI_HEADLINE|ISI_SUBHEADLINE|"
    r"KEUNGGULAN_?1|KEUNGGULAN_?2|KEUNGGULAN_?3|PENUTUP_HEADLINE|PENUTUP_CTA)\s*:",
    re.IGNORECASE,
)


def _bersihkan_teks_desain(teks: str) -> str:
    """Hapus emoji/simbol yang tidak didukung font Poppins - pengaman
    programatik, BUKAN cuma andalkan LLM patuh instruksi 'jangan pakai emoji'."""
    if not teks:
        return teks
    bersih = _POLA_KARAKTER_AMAN.sub("", teks)
    return re.sub(r"\s+", " ", bersih).strip()


def _potong_di_label_berikutnya(teks: str) -> str:
    """Pengaman - kadang LLM menulis semua field dalam 1 baris tanpa newline
    bersih (mis. 'ISI_SUBHEADLINE: teks A KEUNGGULAN1: teks B'), sehingga
    extract_field() ikut menangkap label field BERIKUTNYA sebagai bagian
    dari nilai field SEKARANG (bug nyata: subheadline Isi pernah ikut
    menyeret 'KEUNGGULAN1: ... KEUNGGULAN2: ...' mentah-mentah, tampil di
    slide render sebagai teks acak/random yang membingungkan). Potong teks
    begitu ketemu pola label field lain di dalamnya, ambil HANYA bagian
    sebelum itu."""
    if not teks:
        return teks
    match = _POLA_LABEL_FIELD_LAIN.search(teks)
    if match:
        return teks[:match.start()].strip()
    return teks


# Rasio 4:5 (portrait) - standar carousel/feed modern
UKURAN_CAROUSEL = (1024, 1280)

# Deskriptor gaya visual yang SAMA persis dipakai di SEMUA prompt AI (Cover,
# Isi, Penutup) - ini satu-satunya cara memaksa konsistensi visual antar 3
# pemanggilan Pollinations yang terpisah (tidak ada "memory" antar call).
_GAYA_KONSISTEN_CAROUSEL = (
    "minimalist modern flat design background, soft cohesive color palette, "
    "clean professional aesthetic, generous negative space, subject centered "
    "in frame (not near edges, to survive thumbnail cropping)"
)

_PERAN_CAROUSEL = ["Cover", "Isi", "Penutup"]


def _nama_warna_kasar(rgb: tuple) -> str:
    """Ubah RGB jadi nama warna kasar Bahasa Indonesia - dipakai di prompt
    background supaya LLM/Pollinations dapat instruksi warna yang natural
    dibaca (bukan cuma kode hex yang sering diabaikan text-to-image model)."""
    import colorsys
    r, g, b = [x / 255 for x in rgb]
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    if s < 0.15:
        return "netral putih/abu-abu terang" if v > 0.5 else "netral abu-abu gelap"
    h_deg = h * 360
    if h_deg < 15 or h_deg >= 345:
        return "merah"
    if h_deg < 45:
        return "oranye"
    if h_deg < 70:
        return "kuning"
    if h_deg < 170:
        return "hijau"
    if h_deg < 200:
        return "toska/teal"
    if h_deg < 250:
        return "biru"
    if h_deg < 290:
        return "ungu"
    return "pink/magenta"


def _klausa_larangan_objek(produk: dict) -> str:
    """Pengaman programatik - dipaksa ke SETIAP prompt background, TIDAK
    bergantung pada LLM/Pollinations patuh instruksi 'jangan gambar objek'.
    Diperluas lagi setelah kasus nyata: silinder DAN blok/kotak persegi
    panjang tetap muncul di slide Isi & Penutup meski sudah dilarang di
    versi sebelumnya - daftar larangan bentuk geometris diperbanyak,
    termasuk istilah "display table/stage" yang sering dipakai text-to-image
    model untuk komposisi product photography walau tidak diminta."""
    kategori = produk.get("kategori", "product")
    nama = produk.get("nama", "")
    sebutan = f"{kategori}" + (f" or {nama}" if nama else "")
    return (
        f"empty decorative background only, absolutely no {sebutan} depicted, "
        "no objects, no props, no items resembling any product, no pedestals, "
        "no podiums, no plinths, no risers, no stands, no platforms, no tables, "
        "no benches, no stages, no display surfaces, no geometric blocks, no "
        "cubes, no boxes, no cylinders, no tubes, no shapes of any kind sitting "
        "in the scene, no product photography staging or studio setup - "
        "just a flat plain surface, texture, gradient, light and color only, "
        "completely empty backdrop as if for flat-lay product photography "
        "where the product will be composited in separately, identical in "
        "spirit to a simple seamless studio paper backdrop with nothing "
        "placed on it"
    )


def _susun_prompt_latar_ai(
    produk: dict, peran: str, ide_visual: str, catatan_larangan: str = "",
    warna_dominan_produk: tuple = None, deskripsi_foto_asli: str = "",
) -> str:
    """Prompt Bahasa Inggris untuk background dekoratif (Cover/Isi/Penutup) -
    SELALU sertakan _GAYA_KONSISTEN_CAROUSEL supaya 3 slide terasa 1 set.

    FIX (permintaan user): hint per-peran SEBELUMNYA beda-beda gaya
    ("bold centerpiece" utk Cover, "warm inviting" utk Penutup dsb) - ini
    kemungkinan mendorong model membuat komposisi "product display" dengan
    alas/panggung dekoratif, padahal user cuma mau gradient/tekstur polos
    SAMA seperti hasil Cover yang sudah bagus. Sekarang SEMUA peran memakai
    instruksi gaya yang SAMA PERSIS (flat gradient/tekstur polos, tanpa
    nuansa "displaying" apapun) - beda hanya di WARNA/nuansa terang-gelap
    ringan, bukan di konsep komposisinya."""
    system_prompt = (
        "Kamu menerjemahkan ide konten marketing UMKM jadi PROMPT BAHASA "
        "INGGRIS untuk BACKGROUND DEKORATIF KOSONG (bukan produk itu sendiri "
        "- produk akan ditempel terpisah sebagai foto asli di atas latar "
        "ini). ATURAN KERAS: JANGAN sebutkan kata/frasa yang merujuk ke "
        "BENTUK objek produk atau alat peraga/prop dekoratif apapun - "
        "termasuk nama kategori produknya sendiri, wadah, dudukan, stand, "
        "pedestal, podium, silinder, meja, panggung, kotak, blok, kubus, "
        "tali, permukaan berbentuk objek, atau APAPUN yang terkesan seperti "
        "'menampilkan produk di atas sesuatu'. HANYA deskripsikan warna, "
        "tekstur, gradient, cahaya, material permukaan (kain, marmer, "
        "kertas, dsb) - persis seperti backdrop studio polos tanpa apapun "
        "diletakkan di atasnya, seolah kanvas kosong rata. "
        "JANGAN sertakan teks/tulisan (akan ditambahkan terpisah). Output "
        "HANYA prompt-nya, maksimal 2 kalimat."
    )
    # SEMUA peran pakai hint KONSEP yang sama (flat/gradient polos) - beda
    # HANYA di nuansa warna/mood ringan, BUKAN di jenis komposisinya, supaya
    # ketiga slide konsisten seperti hasil Cover yang sudah bagus.
    hint_peran = {
        "Cover": "clean bright airy mood, soft warm tone",
        "Isi": "calm neutral mood, gentle soft tone",
        "Penutup": "warm cozy mood, soft inviting tone",
    }[peran]
    larangan_block = f"\nHindari: {catatan_larangan}" if catatan_larangan else ""
    warna_block = ""
    if warna_dominan_produk:
        nama_warna = _nama_warna_kasar(warna_dominan_produk)
        warna_block = (
            f"\nWarna dominan kemasan/produk asli: {nama_warna} (RGB{warna_dominan_produk}) - "
            "background HARUS pakai palet warna yang SENADA/harmonis dengan warna ini "
            "(bisa senada langsung atau warna komplementer yang lembut), JANGAN warna "
            "yang kontras tajam/bentrok - tujuannya produk terlihat menyatu natural "
            "dengan latar, bukan seperti ditempel asal di atas warna acak."
        )
    deskripsi_block = (
        f"\nKesan warna/suasana dari foto produk asli (dari vision - AMBIL "
        f"KESAN WARNANYA SAJA, JANGAN sebut ulang bentuk objeknya sama sekali): "
        f"{deskripsi_foto_asli}\n"
        "Rancang latar dengan warna/suasana senada, TAPI TETAP flat gradient/"
        "tekstur polos saja - DILARANG KERAS menyebut, menggambarkan, atau "
        "menyiratkan bentuk objek/produk/prop/permukaan tempat produk berdiri "
        "apapun - fokus HANYA pada warna, tekstur, cahaya, dan material "
        "permukaan yang RATA/DATAR."
        if deskripsi_foto_asli else ""
    )
    user_prompt = (
        f"Kategori produk: {produk['kategori']}\n"
        f"Ide konten: {ide_visual}\n"
        f"Peran slide ini: {peran} ({hint_peran})"
        f"{warna_block}{deskripsi_block}"
        f"{larangan_block}\n\n"
        "Susun prompt background dekoratif KOSONG-nya (flat gradient/tekstur "
        "polos saja, tanpa objek/prop/panggung apapun)."
    )
    prompt_dasar = call_llm(system_prompt, user_prompt, temperature=0.6)
    return f"{prompt_dasar}, {_GAYA_KONSISTEN_CAROUSEL}, {_klausa_larangan_objek(produk)}"


def _susun_prompt_gambar_penuh(produk: dict, peran: str, ide_visual: str, catatan_larangan: str = "") -> str:
    """Prompt Bahasa Inggris untuk gambar PENUH (produk+scene, dipakai kalau
    TIDAK ada upload - full AI imagine, SELALU sertakan gaya konsisten."""
    system_prompt = (
        "Kamu menerjemahkan ide konten marketing UMKM jadi PROMPT BAHASA "
        "INGGRIS untuk foto produk bergaya (styled product photography). "
        "Produk jadi FOKUS UTAMA di tengah frame (bukan mepet tepi), beri "
        "ruang kosong di sekitarnya. JANGAN sertakan teks/tulisan (akan "
        "ditambahkan terpisah). Output HANYA prompt-nya, maksimal 3 kalimat."
    )
    hint_peran = {
        "Cover": "hero shot, bold and attention-grabbing composition",
        "Isi": "simple detail-focused shot highlighting one feature",
        "Penutup": "clean closing shot, inviting and warm",
    }[peran]
    larangan_block = f"\nHindari: {catatan_larangan}" if catatan_larangan else ""
    user_prompt = (
        f"Produk: {produk['nama']} ({produk['kategori']})\n"
        f"Deskripsi produk: {produk['deskripsi']}\n"
        f"Ide konten: {ide_visual}\n"
        f"Peran slide ini: {peran} ({hint_peran})"
        f"{larangan_block}\n\n"
        "Susun prompt gambarnya."
    )
    prompt_dasar = call_llm(system_prompt, user_prompt, temperature=0.6)
    return f"{prompt_dasar}, {_GAYA_KONSISTEN_CAROUSEL}, professional photography, no distorted body parts, no text, no watermark"


def _susun_desain_teks_carousel(produk: dict, caption_final: str, ide_visual: str, catatan_perbaikan: str = "") -> dict:
    """Design Agent - susun teks utk 3 slide (Cover/Isi/Penutup). WAJIB
    menyertakan nama produk & manfaat produk (sesuai permintaan), BUKAN
    menulis ulang caption panjang, dan BUKAN mengarang klaim baru.

    PENTING: hasil fungsi ini SEKARANG dicek ulang oleh ComplianceAgent di
    generate_carousel_content() (dulu TIDAK PERNAH dicek sama sekali - celah
    nyata yang menyebabkan klaim karangan seperti "Hapus Noda Jerawat" bisa
    lolos ke slide render tanpa terdeteksi apapun). catatan_perbaikan dipakai
    kalau recheck itu TOLAK dan perlu revisi terbatas."""
    system_prompt = (
        "PERAN: Kamu adalah Visual Design Copywriter untuk carousel marketing UMKM.\n\n"
        "TUJUAN: Menyusun teks PENDEK untuk 3 slide carousel, berdasarkan "
        "caption Hook-Value-CTA yang SUDAH disetujui Compliance - BUKAN "
        "menulis ulang caption panjang, dan BUKAN mengarang klaim baru.\n\n"
        "ATURAN PER SLIDE:\n"
        "- COVER_HEADLINE: representasi Hook - WAJIB sebut nama produk, "
        "SANGAT PENDEK (maks 6 kata), menarik perhatian\n"
        "- COVER_SUBHEADLINE: 1 kalimat pendek pendukung headline, maks 8 "
        "kata, atau N/A kalau headline sudah cukup kuat sendiri\n"
        "- ISI_HEADLINE: SATU manfaat produk paling utama (bukan daftar "
        "semua manfaat - pilih SATU yang paling kuat), maks 6 kata\n"
        "- ISI_SUBHEADLINE: 1 kalimat pendukung singkat manfaat itu, maks "
        "10 kata, atau N/A kalau headline sudah cukup\n"
        "- KEUNGGULAN_1/2/3: TIGA keunggulan produk BERBEDA, masing-masing "
        "SANGAT PENDEK (maks 4 kata, bukan kalimat lengkap - misal 'Tanpa "
        "Pengawet' bukan 'Produk ini tanpa pengawet'), ditampilkan sebagai "
        "3 poin terpisah mengelilingi foto produk\n"
        "- PENUTUP_HEADLINE: ringkasan singkat/ajakan, maks 5 kata\n"
        "- PENUTUP_CTA: teks tombol, SANGAT PENDEK (2-4 kata), SATU ajakan "
        "spesifik saja (samakan dengan CTA di caption, jangan campur "
        "beberapa ajakan)\n\n"
        "FORMAT OUTPUT WAJIB - SANGAT PENTING: setiap field HARUS di baris "
        "TERPISAH (satu field satu baris, diakhiri newline), JANGAN PERNAH "
        "menulis dua field atau lebih dalam satu baris yang sama walau "
        "kelihatan singkat - ini WAJIB supaya sistem bisa memisahkan tiap "
        "field dengan benar, kalau digabung dalam satu baris akan membuat "
        "nilai field jadi rusak/tercampur.\n\n"
        "BATASAN (WAJIB, ini SAMA KETATNYA dengan aturan ComplianceAgent -\n"
        "teks ini AKAN dicek ulang oleh Compliance sebelum dipakai):\n"
        "- DILARANG KERAS mengarang manfaat/klaim APAPUN yang tidak benar-benar "
        "ada di 'Caption Hook-Value-CTA yang sudah disetujui' ATAU deskripsi "
        "produk di bawah. Setiap kata benda klaim (misal 'menghilangkan X', "
        "'menyembuhkan Y', 'menghapus Z') WAJIB bisa kamu tunjuk balik ke "
        "kalimat SPESIFIK di caption/deskripsi - kalau tidak ketemu, JANGAN "
        "ditulis, pilih manfaat lain yang benar-benar disebutkan\n"
        "- KEUNGGULAN_1/2/3 WAJIB 3 manfaat BERBEDA (jangan mengulang manfaat "
        "yang sama dengan kata lain) - kalau caption/deskripsi cuma menyebut "
        "1-2 manfaat konkret, isi sisanya dengan atribut netral yang memang "
        "ada (misal ukuran kemasan, cara pakai) daripada mengarang manfaat baru\n"
        "- DILARANG KERAS membuat klaim medis/kosmetik yang lebih kuat dari "
        "aslinya (misal deskripsi cuma bilang 'membantu mengontrol minyak' "
        "JANGAN ditulis ulang jadi klaim yang terdengar seperti menyembuhkan "
        "kondisi kulit tertentu - itu klaim BARU yang tidak ada dasarnya)\n"
        "- JANGAN PAKAI EMOJI (font render tidak mendukung emoji, hasilnya "
        "kotak rusak)\n\n"
        "FORMAT OUTPUT (WAJIB, gunakan label ini persis, SATU FIELD SATU BARIS):\n"
        "COVER_HEADLINE: <teks>\n"
        "COVER_SUBHEADLINE: <teks atau N/A>\n"
        "ISI_HEADLINE: <teks>\n"
        "ISI_SUBHEADLINE: <teks atau N/A>\n"
        "KEUNGGULAN_1: <teks>\n"
        "KEUNGGULAN_2: <teks>\n"
        "KEUNGGULAN_3: <teks>\n"
        "PENUTUP_HEADLINE: <teks>\n"
        "PENUTUP_CTA: <teks tombol>"
    )
    catatan_block = (
        f"\n\nPERBAIKAN DIPERLUKAN - Compliance MENOLAK versi sebelumnya "
        f"dengan alasan: {catatan_perbaikan}\nRevisi HANYA bagian yang "
        f"bermasalah, JANGAN mengarang fakta baru untuk memperbaikinya - kalau "
        f"tidak bisa diperbaiki tanpa mengarang, ganti dengan manfaat lain "
        f"yang benar-benar ada di caption/deskripsi, atau kosongkan (N/A)."
        if catatan_perbaikan else ""
    )
    user_prompt = (
        f"Produk: {produk['nama']} ({produk['kategori']})\n"
        f"Deskripsi: {produk['deskripsi']}\n"
        f"Caption Hook-Value-CTA yang sudah disetujui:\n{caption_final}\n\n"
        f"Ide visual: {ide_visual}\n"
        f"Platform: {produk['platform']}"
        f"{catatan_block}\n\n"
        "Susun teks untuk 3 slide sesuai format output."
    )
    respon = call_llm(system_prompt, user_prompt, temperature=0.6)

    def _ambil(field, default=""):
        v = (extract_field(respon, field) or default).strip()
        v = _potong_di_label_berikutnya(v)
        return "" if v.upper() == "N/A" else _bersihkan_teks_desain(v)

    # NAMA_TOKO & NAMA_PRODUK SENGAJA tidak diminta dari LLM - diambil
    # langsung dari data produk (deterministik, tanpa risiko halusinasi nama).
    nama_toko = _bersihkan_teks_desain(produk.get("nama_usaha") or produk["nama"])
    nama_produk = _bersihkan_teks_desain(produk["nama"])

    return {
        "Cover": {
            "nama_toko": nama_toko,
            "headline": _ambil("COVER_HEADLINE", produk["nama"]),
            "subheadline": _ambil("COVER_SUBHEADLINE"),
            "nama_produk": nama_produk,
        },
        "Isi": {
            "nama_toko": nama_toko,
            "headline": _ambil("ISI_HEADLINE"),
            "subheadline": _ambil("ISI_SUBHEADLINE"),
            "keunggulan_1": _ambil("KEUNGGULAN_1"),
            "keunggulan_2": _ambil("KEUNGGULAN_2"),
            "keunggulan_3": _ambil("KEUNGGULAN_3"),
        },
        "Penutup": {
            "nama_toko": nama_toko,
            "headline": _ambil("PENUTUP_HEADLINE"),
            "cta": _ambil("PENUTUP_CTA", "Hubungi Kami"),
        },
    }


def _gabung_teks_desain(desain_teks: dict) -> str:
    """Satukan semua teks yang akan tampil di 3 slide jadi 1 string, supaya
    bisa dicek compliance SEKALIGUS (bukan per-field, biar konteksnya utuh).
    NAMA_TOKO/NAMA_PRODUK sengaja TIDAK diikutkan - itu data produk asli
    langsung dari form, bukan teks yang dikarang LLM, jadi tidak perlu cek
    compliance (bukan klaim)."""
    kunci_klaim = ["headline", "subheadline", "keunggulan_1", "keunggulan_2", "keunggulan_3", "cta"]
    bagian = []
    for peran in _PERAN_CAROUSEL:
        t = desain_teks[peran]
        potongan = " | ".join(t[k] for k in kunci_klaim if t.get(k))
        if potongan:
            bagian.append(f"[{peran}] {potongan}")
    return "\n".join(bagian)


def _teks_desain_generik_aman(produk: dict) -> dict:
    """Fallback kalau desain teks berulang kali gagal compliance recheck -
    HANYA nama produk/toko & CTA generik, tidak ada klaim apapun sama
    sekali, supaya carousel tetap bisa jalan tanpa risiko klaim karangan."""
    nama_toko = _bersihkan_teks_desain(produk.get("nama_usaha") or produk["nama"])
    nama_produk = _bersihkan_teks_desain(produk["nama"])
    return {
        "Cover": {"nama_toko": nama_toko, "headline": nama_produk, "subheadline": "", "nama_produk": nama_produk},
        "Isi": {"nama_toko": nama_toko, "headline": nama_produk, "subheadline": "",
                "keunggulan_1": "", "keunggulan_2": "", "keunggulan_3": ""},
        "Penutup": {"nama_toko": nama_toko, "headline": "Cek Selengkapnya", "cta": "Hubungi Kami"},
    }


def _verifikasi_compliance_desain_teks(
    produk: dict, caption_final: str, ide_visual: str, desain_teks: dict,
    compliance_agent, log_gabungan: list,
) -> dict:
    """PENTING - ini menutup celah nyata: sebelumnya teks yang dirender ke
    slide (headline/subheadline/CTA dari Design Agent) TIDAK PERNAH dicek
    ComplianceAgent sama sekali - hanya background/foto yang dicek lewat
    vision-describe. Klaim karangan seperti 'Hapus Noda Jerawat' padahal
    deskripsi produk tidak menyebutkan itu sama sekali bisa lolos ke slide
    render tanpa terdeteksi apapun. Sekarang teks desain di-recheck di sini,
    dengan retry terbatas (DESIGN_TEXT_MAX_RETRY), fallback ke teks generik
    tanpa klaim apapun kalau tetap gagal."""
    teks_sekarang = desain_teks
    for percobaan in range(DESIGN_TEXT_MAX_RETRY + 1):
        gabungan = _gabung_teks_desain(teks_sekarang)
        if not gabungan.strip():
            return teks_sekarang  # tidak ada teks berklaim sama sekali, aman

        hasil_compliance = compliance_agent.evaluasi_teks(produk, gabungan)
        verdict = extract_verdict(hasil_compliance)
        alasan = extract_field(hasil_compliance, "ALASAN")
        saran = extract_field(hasil_compliance, "SARAN_PERBAIKAN")
        log_gabungan.append({
            "percobaan": percobaan + 1,
            "status": f"CEK_TEKS_DESAIN_{verdict}",
            "prompt": f"(teks desain carousel)\n{gabungan}",
            "deskripsi_vision": "(bukan gambar - ini teks headline/subheadline/CTA slide)",
            "alasan_compliance": alasan,
        })

        if verdict == "AMAN":
            return teks_sekarang

        if percobaan < DESIGN_TEXT_MAX_RETRY:
            teks_sekarang = _susun_desain_teks_carousel(
                produk, caption_final, ide_visual, catatan_perbaikan=f"{alasan} {saran}".strip()
            )
        else:
            log_gabungan.append({
                "percobaan": percobaan + 1,
                "status": "FALLBACK_TEKS_DESAIN_GENERIK",
                "prompt": "(teks desain carousel)",
                "deskripsi_vision": "-",
                "alasan_compliance": (
                    "Teks desain berulang kali gagal compliance recheck - fallback ke "
                    "teks generik tanpa klaim (nama produk saja) demi keamanan."
                ),
            })
            return _teks_desain_generik_aman(produk)

    return teks_sekarang


def _cek_compliance_gambar(compliance_agent, produk, gambar_bytes, content_type):
    """Describe + cek compliance 1 gambar. Return (verdict, alasan, saran, deskripsi)."""
    deskripsi = describe_image(gambar_bytes, content_type, produk)
    if deskripsi:
        hasil_compliance = compliance_agent.evaluasi_teks(produk, deskripsi)
    else:
        hasil_compliance = (
            "AMAN\nALASAN: Vision describe gagal, tidak ada teks untuk "
            "diperiksa rule engine - ini BUKAN jaminan aman, cuma tidak ada "
            "temuan karena pemeriksaan tidak sempat jalan.\nSARAN_PERBAIKAN: N/A"
        )
    verdict = extract_verdict(hasil_compliance)
    alasan = extract_field(hasil_compliance, "ALASAN")
    saran = extract_field(hasil_compliance, "SARAN_PERBAIKAN")
    return verdict, alasan, saran, deskripsi


def _cek_latar_bersih_dari_objek(produk: dict, deskripsi_latar: str) -> bool:
    """Cek APAKAH background hasil AI generate masih menggambar bentuk objek
    ATAU PROP dekoratif (pedestal/podium/silinder/blok/meja dkk) - diperluas
    lagi setelah kasus nyata: silinder oranye DAN blok/kotak persegi panjang
    tetap muncul (lolos gate versi sebelumnya karena kata "blok"/"meja"/
    "panggung" belum ada di daftar kata kunci lama)."""
    if not deskripsi_latar:
        return True
    deskripsi_lower = deskripsi_latar.strip().lower()
    penanda_aman = [
        "elemen dekoratif", "hanya elemen dekoratif", "tanpa objek",
        "tidak ada produk", "tidak ada objek", "murni merupakan elemen",
    ]
    if any(p in deskripsi_lower for p in penanda_aman):
        return True
    kategori = (produk.get("kategori") or "").strip().lower()
    if kategori and kategori in deskripsi_lower:
        return False
    kata_kunci_objek = [
        "object", "product", "item", "pedestal", "podium", "platform",
        "stand", "riser", "plinth", "silinder", "cylinder", "tabung",
        "wadah", "alas", "dudukan", "prop",
        # tambahan - kata bentuk geometris/perabot yang lolos di kasus nyata
        "blok", "block", "kotak", "box", "kubus", "cube", "meja", "table",
        "bangku", "bench", "panggung", "stage", "display", "balok",
    ]
    if any(k in deskripsi_lower for k in kata_kunci_objek):
        return False
    return True


def generate_carousel_content(
    produk: dict,
    hasil_diskusi: dict,
    compliance_agent,
    foto_produk_upload: bytes = None,
    foto_produk_content_type: str = "image/jpeg",
) -> dict:
    """
    Generate carousel 3 slide (Cover/Isi/Penutup), rasio 4:5.

    - Ada foto upload: foto ASLI ditempel sebagai kartu di atas background
      dekoratif AI-generate (beda per slide, tapi diarahkan oleh deskripsi
      vision foto asli supaya gayanya nyambung) - produk TETAP 100% asli,
      TIDAK PERNAH digambar ulang oleh AI.
    - Tanpa upload: 3 slide full AI-generate dengan gaya konsisten.

    Return dict:
      - slide: list of {"peran", "gambar_bytes", "content_type", "status"}
      - desain_teks: dict per peran (untuk transparansi dashboard)
      - sumber_foto, log
    """
    ide_visual = (
        hasil_diskusi.get("saran_konten_alternatif", {}).get("ide_visual")
        or hasil_diskusi.get("caption_final", "")
    )
    desain_teks = _susun_desain_teks_carousel(
        produk, hasil_diskusi.get("caption_final", ""), ide_visual
    )
    log_gabungan = []
    slide_hasil = []

    # PENTING: verifikasi compliance untuk teks desain SEBELUM dipakai render
    # apapun - lihat docstring _verifikasi_compliance_desain_teks untuk celah
    # yang ini tutup (klaim karangan yang dulu lolos begitu saja).
    desain_teks = _verifikasi_compliance_desain_teks(
        produk, hasil_diskusi.get("caption_final", ""), ide_visual,
        desain_teks, compliance_agent, log_gabungan,
    )

    if foto_produk_upload:
        sumber_foto = "UPLOAD_USER"

        # Cek compliance foto asli SEKALI (bukan per-slide - fotonya sama,
        # TIDAK di-retry/regenerate karena ini aset asli usaha, ditempel apa
        # adanya). Deskripsi vision-nya JUGA dipakai lagi di bawah untuk
        # mengarahkan gaya latar AI (lihat _susun_prompt_latar_ai) - bukan
        # untuk menggambar ulang produknya.
        verdict_foto, alasan_foto, _, deskripsi_foto = _cek_compliance_gambar(
            compliance_agent, produk, foto_produk_upload, foto_produk_content_type
        )
        log_gabungan.append({
            "percobaan": 1, "status": verdict_foto,
            "prompt": "(foto diupload user)", "deskripsi_vision": deskripsi_foto or "(kosong)",
            "alasan_compliance": alasan_foto,
        })
        status_foto_dasar = "AMAN" if verdict_foto == "AMAN" else "TOLAK_FOTO_USER"

        # Warna dominan produk asli - dihitung SEKALI (foto sama dipakai di
        # semua slide), dipakai supaya background AI diminta senada dengan
        # warna produk, bukan warna kategori generik yang bisa bentrok/
        # terasa asal tempel. Kalau deteksi gagal, fallback None (prompt
        # tetap jalan tanpa hint warna, seperti sebelumnya).
        try:
            warna_dominan = deteksi_warna_dominan_produk(foto_produk_upload)
        except Exception:
            warna_dominan = None

        for peran in _PERAN_CAROUSEL:
            catatan_larangan = ""
            teks = desain_teks[peran]
            berhasil = False

            for percobaan in range(1, VISUAL_MAX_RETRY + 1):
                prompt_latar = _susun_prompt_latar_ai(
                    produk, peran, ide_visual, catatan_larangan,
                    warna_dominan_produk=warna_dominan,
                    deskripsi_foto_asli=deskripsi_foto,
                )
                latar_hasil = generate_image(prompt_latar, seed=random.randint(1, 999999),
                                              width=UKURAN_CAROUSEL[0], height=UKURAN_CAROUSEL[1])
                if latar_hasil is None:
                    alasan_gagal = getattr(generate_image, "alasan_gagal_terakhir", "") or "tidak diketahui"
                    log_gabungan.append({"percobaan": percobaan, "status": "GAGAL_GENERATE",
                                          "prompt": prompt_latar, "deskripsi_vision": "",
                                          "alasan_compliance": f"[{peran}] latar AI gagal digenerate ({alasan_gagal})"})
                    if percobaan < VISUAL_MAX_RETRY:
                        time.sleep(VISUAL_RETRY_DELAY_DETIK)
                    continue

                verdict_latar, alasan_latar, saran_latar, deskripsi_latar = _cek_compliance_gambar(
                    compliance_agent, produk, latar_hasil["bytes"], latar_hasil["content_type"]
                )

                # PENTING: cek TAMBAHAN ini yang menutup bug tumpang tindih/
                # prop asing - verdict_latar "AMAN" dari compliance_agent
                # cuma berarti kontennya tidak melanggar rule bisnis, BUKAN
                # jaminan backgroundnya kosong dari objek/prop. Makanya
                # dicek terpisah.
                latar_kosong = _cek_latar_bersih_dari_objek(produk, deskripsi_latar)

                if verdict_latar == "AMAN" and not latar_kosong:
                    log_gabungan.append({"percobaan": percobaan, "status": "TOLAK_LATAR_ADA_OBJEK",
                                          "prompt": f"[{peran}] {prompt_latar}",
                                          "deskripsi_vision": deskripsi_latar or "(kosong)",
                                          "alasan_compliance": (
                                              f"Background masih menggambar bentuk objek/prop terkait "
                                              f"{produk['kategori']} - harus benar-benar kosong (hanya "
                                              "tekstur/warna/cahaya) karena foto produk asli akan "
                                              "ditempel di atasnya."
                                          )})
                    catatan_larangan = (
                        f"Background masih menggambar bentuk {produk['kategori']} atau objek/prop "
                        "serupa (termasuk pedestal/podium/silinder dekoratif) - WAJIB benar-benar "
                        "kosong tanpa objek/prop apapun, hanya warna/tekstur/cahaya polos."
                    )
                    if percobaan < VISUAL_MAX_RETRY:
                        time.sleep(VISUAL_RETRY_DELAY_DETIK)
                    continue

                log_gabungan.append({"percobaan": percobaan, "status": verdict_latar,
                                      "prompt": f"[{peran}] {prompt_latar}",
                                      "deskripsi_vision": deskripsi_latar or "(kosong)",
                                      "alasan_compliance": alasan_latar})

                if verdict_latar == "AMAN":
                    gambar_final = render_kartu_dengan_latar_ai(
                        latar_hasil["bytes"], foto_produk_upload, UKURAN_CAROUSEL, produk["kategori"],
                        peran, teks,
                    )
                    slide_hasil.append({"peran": peran, "gambar_bytes": gambar_final,
                                         "content_type": "image/png", "status": status_foto_dasar})
                    berhasil = True
                    break
                catatan_larangan = saran_latar
                if percobaan < VISUAL_MAX_RETRY:
                    time.sleep(VISUAL_RETRY_DELAY_DETIK)

            if not berhasil:
                # Fallback: latar polos warna netral (tanpa AI) - foto produk tetap tampil
                from PIL import Image
                import io as _io
                kanvas_polos = Image.new("RGB", UKURAN_CAROUSEL, (245, 245, 245))
                buf = _io.BytesIO()
                kanvas_polos.save(buf, format="PNG")
                gambar_final = render_kartu_dengan_latar_ai(
                    buf.getvalue(), foto_produk_upload, UKURAN_CAROUSEL, produk["kategori"],
                    peran, teks,
                )
                slide_hasil.append({"peran": peran, "gambar_bytes": gambar_final,
                                     "content_type": "image/png", "status": "FALLBACK_LATAR_POLOS"})
    else:
        sumber_foto = "AI_GENERATED"

        for peran in _PERAN_CAROUSEL:
            catatan_larangan = ""
            teks = desain_teks[peran]
            berhasil = False

            for percobaan in range(1, VISUAL_MAX_RETRY + 1):
                prompt_gambar = _susun_prompt_gambar_penuh(produk, peran, ide_visual, catatan_larangan)
                hasil_gambar = generate_image(prompt_gambar, seed=random.randint(1, 999999),
                                               width=UKURAN_CAROUSEL[0], height=UKURAN_CAROUSEL[1])
                if hasil_gambar is None:
                    alasan_gagal = getattr(generate_image, "alasan_gagal_terakhir", "") or "tidak diketahui"
                    log_gabungan.append({"percobaan": percobaan, "status": "GAGAL_GENERATE",
                                          "prompt": prompt_gambar, "deskripsi_vision": "",
                                          "alasan_compliance": f"[{peran}] gagal digenerate ({alasan_gagal})"})
                    if percobaan < VISUAL_MAX_RETRY:
                        time.sleep(VISUAL_RETRY_DELAY_DETIK)
                    continue

                verdict, alasan, saran, deskripsi = _cek_compliance_gambar(
                    compliance_agent, produk, hasil_gambar["bytes"], hasil_gambar["content_type"]
                )
                log_gabungan.append({"percobaan": percobaan, "status": verdict,
                                      "prompt": f"[{peran}] {prompt_gambar}",
                                      "deskripsi_vision": deskripsi or "(kosong)", "alasan_compliance": alasan})

                if verdict == "AMAN":
                    gambar_final = render_desain(
                        hasil_gambar["bytes"], UKURAN_CAROUSEL, produk["kategori"], peran, teks,
                    )
                    slide_hasil.append({"peran": peran, "gambar_bytes": gambar_final,
                                         "content_type": "image/png", "status": "AMAN"})
                    berhasil = True
                    break
                catatan_larangan = saran
                if percobaan < VISUAL_MAX_RETRY:
                    time.sleep(VISUAL_RETRY_DELAY_DETIK)

            if not berhasil:
                slide_hasil.append({"peran": peran, "gambar_bytes": None,
                                     "content_type": None, "status": "GAGAL_TOTAL"})

    return {
        "slide": slide_hasil,
        "desain_teks": desain_teks,
        "sumber_foto": sumber_foto,
        "log": log_gabungan,
    }