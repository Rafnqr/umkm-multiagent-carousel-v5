"""
tools/image_compositor.py
Dua mode render:
1. render_desain() - foto (AI generate ATAU upload) jadi FULL background,
   dipakai kalau tidak ada foto asli untuk dijaga keasliannya (misal carousel
   yang full AI-generate tanpa upload, ATAU sekarang juga dipakai untuk foto
   upload yang sudah di-regenerate AI berdasarkan deskripsi foto asli - lihat
   visual_pipeline._susun_prompt_gambar_dari_foto).
2. render_kartu_dengan_latar_ai() - foto asli user ditempel DI ATAS
   background AI dekoratif, TANPA bingkai/kotak putih (background putih foto
   produk dihilangkan jadi transparan, lalu diberi bayangan lembut yang
   MENGIKUTI SILUET produk - bukan kotak persegi). DIPERTAHANKAN di file ini
   untuk kompatibilitas/kemungkinan dipakai ulang, TAPI per permintaan user
   sekarang TIDAK dipanggil lagi dari visual_pipeline.py untuk carousel
   Cover/Isi/Penutup jalur upload (diganti render_desain() dengan gambar
   yang di-regenerate AI supaya konsisten dgn foto asli tanpa menempel foto
   mentah).

DESAIN: foto produk (baik full background maupun kartu) SELALU contain-fit -
tidak pernah crop-to-fill yang bisa memotong bagian penting. Latar belakang
dekoratif (AI-generated) BOLEH di-crop-to-fill karena itu cuma dekorasi,
bukan aset asli yang harus dijaga utuh.

Palet warna HARDCODE per kategori (bukan ditebak LLM) - vibrant/warna-warni.
Font: Poppins (open-source, lisensi OFL) - dibundel di assets/fonts/.

CATATAN CTA PENUTUP: CTA sekarang digambar sebagai TOMBOL SOLID (lihat
_gambar_tombol_cta) memakai warna aksen kategori produk - bukan teks
mengambang dengan scrim transparan seperti elemen lain, supaya terlihat
jelas sebagai ajakan bertindak (tombol), bukan sekadar judul.

CATATAN LAYOUT ISI: posisi KEUNGGULAN_1/2/3 sengaja dibuat MEPET ke tepi
foto produk (bukan mepet ke tepi kanvas) - lihat _hitung_layout_slide,
supaya secara visual terlihat "menunjuk"/terkait langsung ke foto produk di
tengah, bukan terasa terpisah jauh di pinggir kanvas.
"""

import io
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

_DIR_FONT = os.path.join(os.path.dirname(__file__), "..", "..", "assets", "fonts")
_FONT_BOLD = os.path.join(_DIR_FONT, "Poppins-Bold.ttf")
_FONT_SEMIBOLD = os.path.join(_DIR_FONT, "Poppins-SemiBold.ttf")
_FONT_REGULAR = os.path.join(_DIR_FONT, "Poppins-Regular.ttf")

_PALET_PER_KATEGORI = {
    "skincare": {"utama": (255, 225, 214), "aksen": (214, 90, 90), "teks_aksen": (255, 255, 255)},
    "fashion": {"utama": (28, 28, 30), "aksen": (196, 164, 92), "teks_aksen": (28, 28, 30)},
    "makanan": {"utama": (255, 205, 60), "aksen": (214, 60, 40), "teks_aksen": (255, 255, 255)},
    "lainnya": {"utama": (207, 232, 232), "aksen": (36, 92, 122), "teks_aksen": (255, 255, 255)},
}


def _palet(kategori: str) -> dict:
    kategori_lower = kategori.strip().lower()
    kata_kunci = {
        "skincare": ["skincare", "skin care", "kosmetik", "kecantikan", "perawatan kulit", "perawatan wajah"],
        "fashion": ["fashion", "baju", "pakaian", "outfit", "busana"],
        "makanan": ["makanan", "kuliner", "snack", "camilan", "minuman", "food", "kopi", "kue"],
    }
    for kunci, daftar_kata in kata_kunci.items():
        if any(kata in kategori_lower for kata in daftar_kata):
            return _PALET_PER_KATEGORI[kunci]
    return _PALET_PER_KATEGORI["lainnya"]


def _bungkus_teks(draw, teks, font, lebar_maks_px):
    kata = teks.split()
    baris = []
    baris_sekarang = ""
    for kata_ke in kata:
        cobaan = (baris_sekarang + " " + kata_ke).strip()
        bbox = draw.textbbox((0, 0), cobaan, font=font)
        if bbox[2] - bbox[0] <= lebar_maks_px:
            baris_sekarang = cobaan
        else:
            if baris_sekarang:
                baris.append(baris_sekarang)
            baris_sekarang = kata_ke
    if baris_sekarang:
        baris.append(baris_sekarang)
    return baris


def deteksi_warna_dominan_produk(foto_bytes: bytes) -> tuple:
    """Ekstrak warna dominan (RGB) dari AREA PRODUK SAJA (background sudah
    dihilangkan pakai jalur yang sama seperti _potong_produk_transparan) -
    dipakai supaya prompt background/gambar AI bisa diminta 'senada/harmonis'
    dengan warna produk asli, bukan warna kategori generik yang bisa
    bentrok/tidak natural. Median dipakai (bukan mean) supaya tahan terhadap
    outlier seperti highlight terang/bayangan tepi produk."""
    foto_asli = Image.open(io.BytesIO(foto_bytes))
    sudah_ada_transparansi = False
    if foto_asli.mode in ("RGBA", "LA") or "transparency" in foto_asli.info:
        foto_rgba = foto_asli.convert("RGBA")
        alpha_asli = np.array(foto_rgba.split()[-1])
        if (alpha_asli < 250).sum() / alpha_asli.size > 0.01:
            transparan = foto_rgba
            sudah_ada_transparansi = True
    if not sudah_ada_transparansi:
        transparan = _hilangkan_background_putih(foto_asli.convert("RGB"))

    arr = np.array(transparan)
    mask_produk = arr[:, :, 3] > 200
    if mask_produk.sum() < 100:
        return (200, 200, 200)  # fallback netral kalau deteksi gagal total

    piksel_produk = arr[mask_produk][:, :3]
    return tuple(int(x) for x in np.median(piksel_produk, axis=0))


def _warna_teks_kontras(kanvas_rgb: Image.Image, kotak: tuple) -> tuple:
    """Pilih warna teks (gelap/terang) berdasarkan kecerahan RATA-RATA area
    background di bawah kotak teks - supaya tetap kebaca di background AI
    apa saja, bukan diasumsikan selalu putih seperti desain lama."""
    x0, y0, x1, y1 = [max(0, min(v, kanvas_rgb.width if i % 2 == 0 else kanvas_rgb.height))
                      for i, v in enumerate(kotak)]
    if x1 <= x0 or y1 <= y0:
        return (30, 30, 30)
    region = kanvas_rgb.crop((x0, y0, x1, y1)).convert("L")
    return (30, 30, 30) if np.array(region).mean() > 150 else (248, 248, 248)


def _tempel_scrim_lembut(lapisan_overlay: Image.Image, kotak: tuple, blur_radius: int = 4, opacity: int = 225) -> Image.Image:
    """Scrim putih di belakang 1 blok teks, supaya teks tetap kebaca jelas di
    atas foto background apapun warnanya.

    FIX (permintaan user - "hilangkan bayangan pada teks, biar lebih bersih"):
    versi lama pakai blur_radius=20 + opacity=150, yang secara visual jadi
    HALO/GLOW putih tebal menyebar jauh di luar teks (terlihat di
    Cover/Isi/Penutup - bukan efek bayangan tipis, tapi cahaya blur besar).
    Sekarang blur dikecilkan jauh (4px, cuma buat menghaluskan sudut pill)
    dan opacity dinaikkan supaya jadi backing SOLID bersih dengan tepi rapi,
    bukan glow yang menyebar."""
    x0, y0, x1, y1 = kotak
    scrim = Image.new("RGBA", lapisan_overlay.size, (0, 0, 0, 0))
    draw_scrim = ImageDraw.Draw(scrim)
    radius = max(4, int((y1 - y0) * 0.3))
    draw_scrim.rounded_rectangle([(x0, y0), (x1, y1)], radius=radius, fill=(255, 255, 255, opacity))
    if blur_radius > 0:
        scrim = scrim.filter(ImageFilter.GaussianBlur(blur_radius))
    return Image.alpha_composite(lapisan_overlay, scrim)


def _gambar_blok_teks(
    lapisan_overlay: Image.Image, kanvas_dasar_rgb: Image.Image, teks: str, font: ImageFont.FreeTypeFont,
    x: int, y: int, lebar_maks: int, align: str = "kiri", maks_baris: int = 3,
    pakai_scrim: bool = False, spasi_baris_persen: float = 0.25,
) -> tuple:
    """Gambar 1 blok teks (wrap otomatis) di posisi (x,y), dengan scrim
    lembut opsional di belakangnya untuk kontras. Return (lapisan_overlay
    baru, y_setelah_blok_ini) supaya blok berikutnya bisa disusun berurutan."""
    if not teks:
        return lapisan_overlay, y

    draw = ImageDraw.Draw(lapisan_overlay)
    baris_list = _bungkus_teks(draw, teks, font, lebar_maks)[:maks_baris]
    if not baris_list:
        return lapisan_overlay, y

    tinggi_baris = []
    lebar_baris = []
    for baris in baris_list:
        bbox = draw.textbbox((0, 0), baris, font=font)
        tinggi_baris.append(bbox[3] - bbox[1])
        lebar_baris.append(bbox[2] - bbox[0])
    spasi = int(max(tinggi_baris) * spasi_baris_persen)
    tinggi_total = sum(tinggi_baris) + spasi * max(0, len(baris_list) - 1)
    lebar_blok = max(lebar_baris)

    # Padding dikecilkan sedikit (0.4 -> 0.28) supaya pill/backing pas
    # mengikuti teks (rapi), bukan kotak besar dengan banyak ruang kosong
    # yang tampak seperti halo saat digabung dengan blur.
    padding_scrim = int(max(tinggi_baris) * 0.28)
    if align == "kanan":
        x0_blok = x - lebar_blok
    elif align == "tengah":
        x0_blok = x - lebar_blok // 2
    else:
        x0_blok = x

    if pakai_scrim:
        kotak_scrim = (x0_blok - padding_scrim, y - padding_scrim,
                       x0_blok + lebar_blok + padding_scrim, y + tinggi_total + padding_scrim)
        lapisan_overlay = _tempel_scrim_lembut(lapisan_overlay, kotak_scrim)
        warna = _warna_teks_kontras(kanvas_dasar_rgb, kotak_scrim)
    else:
        warna = _warna_teks_kontras(kanvas_dasar_rgb, (x0_blok, y, x0_blok + lebar_blok, y + tinggi_total))

    draw = ImageDraw.Draw(lapisan_overlay)
    y_kursor = y
    for i, baris in enumerate(baris_list):
        if align == "kanan":
            x_baris = x - lebar_baris[i]
        elif align == "tengah":
            x_baris = x - lebar_baris[i] // 2
        else:
            x_baris = x
        draw.text((x_baris, y_kursor), baris, font=font, fill=(*warna, 255))
        y_kursor += tinggi_baris[i] + spasi

    return lapisan_overlay, y_kursor


def _gambar_tombol_cta(
    lapisan_overlay: Image.Image, teks: str, font: ImageFont.FreeTypeFont,
    x_tengah: int, y: int, lebar_maks: int, warna_tombol: tuple, warna_teks: tuple,
) -> Image.Image:
    """Gambar CTA sebagai TOMBOL SOLID (bukan teks mengambang/scrim
    transparan seperti elemen lain) - background warna aksen kategori
    produk, supaya terlihat jelas sebagai ajakan bertindak (tombol nyata),
    bukan cuma judul yang tenggelam di atas foto. Selalu 1 baris (CTA
    memang didesain pendek, maks 2-4 kata)."""
    if not teks:
        return lapisan_overlay

    draw = ImageDraw.Draw(lapisan_overlay)
    bbox = draw.textbbox((0, 0), teks, font=font)
    lebar_teks, tinggi_teks = bbox[2] - bbox[0], bbox[3] - bbox[1]

    padding_x = int(tinggi_teks * 1.1)
    padding_y = int(tinggi_teks * 0.55)
    lebar_tombol = min(lebar_teks + padding_x * 2, lebar_maks)
    tinggi_tombol = tinggi_teks + padding_y * 2

    x0 = x_tengah - lebar_tombol // 2
    y0 = y
    x1 = x0 + lebar_tombol
    y1 = y0 + tinggi_tombol

    draw.rounded_rectangle([(x0, y0), (x1, y1)], radius=tinggi_tombol // 2, fill=(*warna_tombol, 255))
    x_teks = x_tengah - lebar_teks // 2
    y_teks = y0 + padding_y - bbox[1]
    draw.text((x_teks, y_teks), teks, font=font, fill=(*warna_teks, 255))
    return lapisan_overlay


def _hitung_layout_slide(peran: str, lebar: int, tinggi: int) -> dict:
    """Koordinat layout PERSIS mengikuti wireframe yang diberikan user - beda
    per peran slide (Cover/Isi/Penutup), BUKAN 1 layout generik untuk semua
    seperti desain lama (badge kiri/kanan-atas + tombol CTA)."""
    padding = int(lebar * 0.07)

    if peran == "Cover":
        foto_w, foto_h = int(lebar * 0.56), int(tinggi * 0.47)
        foto_x0 = (lebar - foto_w) // 2
        foto_y0 = int(tinggi * 0.31)
        return {
            "foto_box": (foto_x0, foto_y0, foto_x0 + foto_w, foto_y0 + foto_h),
            "nama_toko": (lebar - padding, padding, "kanan"),
            "headline": (padding, padding + int(tinggi * 0.03), "kiri", int(lebar * 0.6)),
            "nama_produk": ("tengah_bawah_foto", int(tinggi * 0.025)),
        }
    if peran == "Isi":
        foto_w, foto_h = int(lebar * 0.44), int(tinggi * 0.38)
        foto_x0 = (lebar - foto_w) // 2
        foto_y0 = int(tinggi * 0.34)
        # gap dipersempit (dari 0.025 ke 0.008) supaya teks KEUNGGULAN lebih
        # MEPET ke tepi foto produk (bukan mepet ke tepi kanvas seperti
        # sebelumnya) - permintaan user, biar terasa "menunjuk" langsung ke
        # foto produk di tengah.
        gap = int(lebar * 0.008)
        # y_headline: nama_toko (font_kecil, ~0.026*lebar) makan tinggi baris
        # sekitar ~0.036*lebar termasuk leading - headline HARUS mulai di
        # bawah itu + jeda visual tambahan, bukan cuma +0.02*tinggi yang
        # nyaris tidak memberi jarak sama sekali (menyebabkan mepet/tabrakan
        # terlihat di hasil render carousel).
        y_headline_isi = padding + int(lebar * 0.036) + int(tinggi * 0.03)
        return {
            "foto_box": (foto_x0, foto_y0, foto_x0 + foto_w, foto_y0 + foto_h),
            "nama_toko": (lebar - padding, padding, "kanan"),
            "headline": (lebar // 2, y_headline_isi, "tengah", int(lebar * 0.8)),
            # keunggulan_1 & keunggulan_3: rata KANAN, posisi x tepat di sisi
            # kiri foto dikurangi gap kecil - teks tumbuh ke arah kiri
            # (menjauhi foto) tapi TEPI KANAN teks tetap mepet ke foto.
            "keunggulan_1": (foto_x0 - gap, foto_y0 + int(foto_h * 0.05), "kanan", foto_x0 - gap - padding // 2),
            "keunggulan_3": (foto_x0 - gap, foto_y0 + foto_h - int(foto_h * 0.1), "kanan", foto_x0 - gap - padding // 2),
            # keunggulan_2: rata KIRI, posisi x tepat di sisi kanan foto
            # ditambah gap kecil - teks tumbuh ke arah kanan (menjauhi foto)
            # tapi TEPI KIRI teks tetap mepet ke foto.
            "keunggulan_2": (foto_x0 + foto_w + gap, foto_y0 + int(foto_h * 0.55), "kiri", lebar - padding - (foto_x0 + foto_w) - gap),
        }
    # Penutup
    foto_w, foto_h = int(lebar * 0.56), int(tinggi * 0.42)
    foto_x0 = (lebar - foto_w) // 2
    foto_y0 = int(tinggi * 0.32)
    return {
        "foto_box": (foto_x0, foto_y0, foto_x0 + foto_w, foto_y0 + foto_h),
        "nama_toko": (lebar - padding, padding, "kanan"),
        "headline": (lebar // 2, int(tinggi * 0.16), "tengah", int(lebar * 0.7)),
        "cta": ("tengah_bawah_foto", int(tinggi * 0.05)),
    }


def _gambar_layout_slide(kanvas_rgba: Image.Image, lebar: int, tinggi: int, peran: str, teks: dict, kategori: str = "") -> Image.Image:
    """Gambar SEMUA teks 1 slide sesuai layout wireframe (per-peran), ganti
    total sistem badge tunggal (_gambar_overlay_teks) yang lama.

    kategori: dipakai untuk menentukan warna tombol CTA di Penutup (lihat
    _gambar_tombol_cta) - kosong string aman (fallback ke palet "lainnya")."""
    layout = _hitung_layout_slide(peran, lebar, tinggi)
    kanvas_dasar_rgb = kanvas_rgba.convert("RGB")
    lapisan = Image.new("RGBA", kanvas_rgba.size, (0, 0, 0, 0))

    font_kecil = ImageFont.truetype(_FONT_REGULAR, int(lebar * 0.026))
    # Cover headline diperbesar lagi (0.072 -> 0.086) supaya hierarki visual
    # jelas - sebagai Hook utama, headline Cover HARUS terasa paling
    # padat/berat dibanding Isi/Penutup, bukan sekadar sedikit lebih besar.
    font_headline_besar = ImageFont.truetype(_FONT_BOLD, int(lebar * 0.086))
    font_headline_sedang = ImageFont.truetype(_FONT_BOLD, int(lebar * 0.055))
    font_sub = ImageFont.truetype(_FONT_REGULAR, int(lebar * 0.033))
    font_keunggulan = ImageFont.truetype(_FONT_SEMIBOLD, int(lebar * 0.026))
    font_nama_produk = ImageFont.truetype(_FONT_SEMIBOLD, int(lebar * 0.04))
    font_cta = ImageFont.truetype(_FONT_BOLD, int(lebar * 0.05))

    x, y, align = layout["nama_toko"]
    lapisan, _ = _gambar_blok_teks(lapisan, kanvas_dasar_rgb, teks.get("nama_toko", ""), font_kecil,
                                     x, y, int(lebar * 0.35), align=align, maks_baris=1, pakai_scrim=False)

    if peran == "Cover":
        x, y, align, lebar_maks = layout["headline"]
        lapisan, y_next = _gambar_blok_teks(lapisan, kanvas_dasar_rgb, teks.get("headline", ""),
                                              font_headline_besar, x, y, lebar_maks, align=align, maks_baris=2,
                                              spasi_baris_persen=0.12)

        if teks.get("nama_produk"):
            y_np = layout["foto_box"][3] + layout["nama_produk"][1]
            lapisan, _ = _gambar_blok_teks(lapisan, kanvas_dasar_rgb, teks["nama_produk"], font_nama_produk,
                                             lebar // 2, y_np, int(lebar * 0.7), align="tengah", maks_baris=1)

    elif peran == "Isi":
        x, y, align, lebar_maks = layout["headline"]
        lapisan, y_next = _gambar_blok_teks(lapisan, kanvas_dasar_rgb, teks.get("headline", ""),
                                              font_headline_sedang, x, y, lebar_maks, align=align, maks_baris=2)
        lapisan, _ = _gambar_blok_teks(lapisan, kanvas_dasar_rgb, teks.get("subheadline", ""),
                                         font_sub, x, y_next + int(tinggi * 0.008), lebar_maks, align=align, maks_baris=2)

        for kunci in ("keunggulan_1", "keunggulan_2", "keunggulan_3"):
            if teks.get(kunci) and kunci in layout:
                x, y, align, lebar_maks = layout[kunci]
                lapisan, _ = _gambar_blok_teks(lapisan, kanvas_dasar_rgb, teks[kunci], font_keunggulan,
                                                 x, y, max(60, lebar_maks), align=align, maks_baris=2)

    else:  # Penutup
        x, y, align, lebar_maks = layout["headline"]
        lapisan, _ = _gambar_blok_teks(lapisan, kanvas_dasar_rgb, teks.get("headline", ""),
                                         font_headline_sedang, x, y, lebar_maks, align=align, maks_baris=2)
        if teks.get("cta"):
            y_cta = layout["foto_box"][3] + layout["cta"][1]
            palet = _palet(kategori)
            lapisan = _gambar_tombol_cta(
                lapisan, teks["cta"], font_cta, lebar // 2, y_cta,
                int(lebar * 0.8), warna_tombol=palet["aksen"], warna_teks=palet["teks_aksen"],
            )

    return Image.alpha_composite(kanvas_rgba, lapisan)


def render_desain(
    background_bytes: bytes,
    ukuran: tuple,
    kategori: str,
    peran: str,
    teks: dict,
    zoom: float = 1.0,
    margin: float = 0.0,
) -> bytes:
    """Foto (AI generate ATAU upload) jadi FULL background, contain-fit
    (SELALU utuh, tidak pernah terpotong). teks: dict field per peran, lihat
    _hitung_layout_slide/_gambar_layout_slide untuk field yang dipakai."""
    lebar, tinggi = ukuran
    bg = Image.open(io.BytesIO(background_bytes)).convert("RGB")
    palet = _palet(kategori)
    warna_latar = palet["utama"]

    if zoom > 1.0:
        w, h = bg.size
        crop_w, crop_h = w / zoom, h / zoom
        cx, cy = w / 2, h / 2
        bg = bg.crop((int(cx - crop_w / 2), int(cy - crop_h / 2), int(cx + crop_w / 2), int(cy + crop_h / 2)))

    margin = max(0.0, min(margin, 0.4))
    area_lebar = int(lebar * (1 - margin))
    area_tinggi = int(tinggi * (1 - margin))

    rasio_target = area_lebar / area_tinggi
    rasio_asli = bg.width / bg.height
    if rasio_asli > rasio_target:
        baru_lebar = area_lebar
        baru_tinggi = max(1, int(baru_lebar / rasio_asli))
    else:
        baru_tinggi = area_tinggi
        baru_lebar = max(1, int(baru_tinggi * rasio_asli))
    bg_resized = bg.resize((baru_lebar, baru_tinggi))

    kanvas = Image.new("RGB", (lebar, tinggi), warna_latar).convert("RGBA")
    offset_x = (lebar - baru_lebar) // 2
    offset_y = (tinggi - baru_tinggi) // 2
    kanvas.paste(bg_resized, (offset_x, offset_y))

    hasil = _gambar_layout_slide(kanvas, lebar, tinggi, peran, teks, kategori=kategori).convert("RGB")
    buffer = io.BytesIO()
    hasil.save(buffer, format="PNG")
    return buffer.getvalue()


def _hilangkan_background_putih(foto_rgb: Image.Image, toleransi: int = 24) -> Image.Image:
    """Ubah background putih/nyaris-putih jadi transparan - HANYA piksel yang
    tersambung ke TEPI gambar (flood-fill dari 4 sudut), supaya area putih
    DI DALAM produk (misal logo/teks putih di kemasan) TIDAK ikut hilang,
    cuma background di sekelilingnya yang hilang. Pakai ImageDraw.floodfill
    (implementasi C bawaan Pillow, cepat) - BUKAN color-key global naif yang
    akan salah menghapus bagian putih di dalam desain kemasan.

    KETERBATASAN JUJUR: ini flood-fill berbasis warna, BUKAN AI background
    removal (rembg/U2Net dkk) - bekerja baik untuk foto produk studio dengan
    background putih/polos bersih (kasus paling umum UMKM, termasuk foto di
    contoh), tapi kurang rapi kalau background sumber ada gradasi/bayangan
    asli yang kuat atau warnanya tidak benar-benar putih/netral."""
    kerja = foto_rgb.copy()
    w, h = kerja.size
    sentinel = (1, 2, 3)  # warna penanda, nyaris mustahil ada natural di foto produk
    for sudut in [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)]:
        try:
            ImageDraw.floodfill(kerja, sudut, sentinel, thresh=toleransi)
        except Exception:
            pass

    arr = np.array(kerja)
    mask_background = np.all(arr == np.array(sentinel), axis=-1)
    alpha = np.where(mask_background, 0, 255).astype("uint8")
    alpha_img = Image.fromarray(alpha, mode="L").filter(ImageFilter.GaussianBlur(1.5))

    hasil = foto_rgb.convert("RGBA")
    hasil.putalpha(alpha_img)
    return hasil


def _potong_produk_transparan(foto_bytes: bytes, ukuran_area: tuple) -> Image.Image:
    """Pengganti _buat_kartu_produk lama (bingkai/kotak putih) - foto produk
    di-contain-fit LANGSUNG ke kanvas transparan seukuran area, TANPA
    bingkai/kotak putih apapun. Foto TIDAK PERNAH di-crop, cuma diperkecil
    muat ke area (sama seperti sebelumnya), bedanya background aslinya
    sekarang dihilangkan supaya menyatu natural dengan latar AI di
    belakangnya, bukan kelihatan seperti stiker kotak putih ditempel.

    PENTING (ditemukan saat testing pakai file PNG asli): kalau foto yang
    diupload SUDAH punya alpha channel transparan yang berarti (misal PNG
    hasil AI image generator/editor lain), pakai transparansi ASLI itu
    langsung - JANGAN di-flood-fill ulang, karena piksel transparan sering
    disimpan dengan RGB gelap/hitam (premultiplied alpha) yang akan salah
    dibaca sebagai "bukan putih" oleh flood-fill warna. Flood-fill HANYA
    dipakai untuk foto flat tanpa transparansi (JPG kamera HP di kertas/meja
    putih - kasus paling umum foto produk UMKM asli)."""
    aw, ah = ukuran_area
    foto_asli = Image.open(io.BytesIO(foto_bytes))

    sudah_ada_transparansi = False
    if foto_asli.mode in ("RGBA", "LA") or "transparency" in foto_asli.info:
        foto_rgba_asli = foto_asli.convert("RGBA")
        alpha_asli = np.array(foto_rgba_asli.split()[-1])
        # Threshold 1%: kalau cuma segelintir piksel yang < 250 (misal noise
        # kompresi), anggap itu BUKAN transparansi berarti - tetap flood-fill.
        if (alpha_asli < 250).sum() / alpha_asli.size > 0.01:
            foto_transparan_penuh = foto_rgba_asli
            sudah_ada_transparansi = True

    if not sudah_ada_transparansi:
        foto_transparan_penuh = _hilangkan_background_putih(foto_asli.convert("RGB"))

    rasio_target = aw / ah
    rasio_asli = foto_transparan_penuh.width / foto_transparan_penuh.height
    if rasio_asli > rasio_target:
        baru_w = aw
        baru_h = max(1, int(baru_w / rasio_asli))
    else:
        baru_h = ah
        baru_w = max(1, int(baru_h * rasio_asli))
    foto_resized = foto_transparan_penuh.resize((baru_w, baru_h), Image.LANCZOS)

    kanvas = Image.new("RGBA", (aw, ah), (0, 0, 0, 0))
    ox = (aw - baru_w) // 2
    oy = (ah - baru_h) // 2
    kanvas.paste(foto_resized, (ox, oy), foto_resized)
    return kanvas


def _buat_bayangan_siluet(
    ukuran_kanvas: tuple, produk_rgba: Image.Image, posisi_xy: tuple,
    offset: tuple = (0, 14), blur_radius: int = 16, opacity: int = 95,
) -> Image.Image:
    """Bayangan lembut yang bentuknya MENGIKUTI SILUET produk (dipetakan dari
    alpha channel produk_rgba, hasil _potong_produk_transparan) - BUKAN kotak
    rounded-rectangle seperti bayangan lama yang sudah dihapus. Return layer
    RGBA seukuran kanvas penuh, siap di-composite di background SEBELUM
    produk ditempel di atasnya."""
    lapisan = Image.new("RGBA", ukuran_kanvas, (0, 0, 0, 0))
    alpha_mask = produk_rgba.split()[-1]
    siluet = Image.new("RGBA", produk_rgba.size, (0, 0, 0, opacity))
    siluet.putalpha(alpha_mask)
    x, y = posisi_xy
    lapisan.paste(siluet, (x + offset[0], y + offset[1]), siluet)
    return lapisan.filter(ImageFilter.GaussianBlur(blur_radius))


def render_kartu_dengan_latar_ai(
    latar_ai_bytes: bytes,
    foto_produk_bytes: bytes,
    ukuran: tuple,
    kategori: str,
    peran: str,
    teks: dict,
) -> bytes:
    """
    Foto produk ASLI ditempel LANGSUNG di atas background dekoratif
    AI-generate - TANPA bingkai/kotak putih (background putih foto produk
    dihilangkan jadi transparan), dengan bayangan lembut yang MENGIKUTI
    SILUET produk (bukan kotak). Produk TETAP 100% asli - AI cuma bikin
    latar belakang, tidak pernah "mengarang" produknya.
    """
    lebar, tinggi = ukuran

    # Background AI dekoratif - BOLEH crop-to-fill (cuma dekorasi, bukan aset asli)
    latar = Image.open(io.BytesIO(latar_ai_bytes)).convert("RGB")
    rasio_target = lebar / tinggi
    rasio_asli = latar.width / latar.height
    if rasio_asli > rasio_target:
        baru_tinggi = tinggi
        baru_lebar = max(1, int(baru_tinggi * rasio_asli))
    else:
        baru_lebar = lebar
        baru_tinggi = max(1, int(baru_lebar / rasio_asli))
    latar_resized = latar.resize((baru_lebar, baru_tinggi))
    kiri = (baru_lebar - lebar) // 2
    atas = (baru_tinggi - tinggi) // 2
    latar_cropped = latar_resized.crop((kiri, atas, kiri + lebar, atas + tinggi))
    kanvas = latar_cropped.convert("RGBA")

    layout = _hitung_layout_slide(peran, lebar, tinggi)
    fx0, fy0, fx1, fy1 = layout["foto_box"]
    kw, kh = fx1 - fx0, fy1 - fy0
    produk = _potong_produk_transparan(foto_produk_bytes, (kw, kh))
    pos_x, pos_y = fx0, fy0

    offset_bayangan = (0, max(6, int(kh * 0.02)))
    blur_bayangan = max(8, int(min(kw, kh) * 0.025))
    bayangan = _buat_bayangan_siluet(
        (lebar, tinggi), produk, (pos_x, pos_y),
        offset=offset_bayangan, blur_radius=blur_bayangan, opacity=90,
    )
    kanvas = Image.alpha_composite(kanvas, bayangan)
    kanvas.paste(produk, (pos_x, pos_y), produk)

    hasil = _gambar_layout_slide(kanvas, lebar, tinggi, peran, teks, kategori=kategori).convert("RGB")
    buffer = io.BytesIO()
    hasil.save(buffer, format="PNG")
    return buffer.getvalue()