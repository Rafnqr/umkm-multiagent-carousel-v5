"""
rule_engine.py
Lapisan deteksi berbasis rule (regex/keyword) untuk Compliance Agent.
Tujuan: deteksi cepat & 100% konsisten untuk pola pelanggaran yang sudah jelas,
supaya LLM tidak perlu "menebak" tiap kali — LLM hanya menjelaskan ALASAN di
belakang hasil rule engine (P3: Compliance = Rule Engine + LLM, bukan LLM murni).
"""

import re

# Pola larangan berlaku SEMUA kategori (EPI + UU Perlindungan Konsumen)
POLA_UNIVERSAL = [
    (r"\b(100%|seratus persen)", "klaim_superlatif_tanpa_bukti",
     "Klaim '100%'/superlatif wajib dibuktikan otoritas resmi (EPI)"),
    (r"\b(paling|nomor satu|terbaik se-)\b", "klaim_superlatif_tanpa_bukti",
     "Klaim 'paling/nomor satu' wajib dibuktikan (EPI)"),
    # PENTING - dulu ini regex kata tunggal `\bmurni\b` yang match SEMUA
    # kemunculan kata "murni", termasuk saat kata itu dipakai sebagai
    # keterangan netral ("murni berupa elemen desain", "murni tanpa klaim") -
    # ini menyebabkan false-positive nyata: deskripsi vision-model yang
    # justru bilang "TIDAK ada klaim" ikut kena TOLAK gara-gara satu kata itu.
    # Sekarang HANYA match kalau "murni/asli" benar-benar menempel ke klaim
    # produk (bahan/kandungan/persentase), bukan kata itu berdiri sendiri.
    (r"\b(bahan (yang )?murni|murni alami|100%\s*murni|asli 100%?)\b", "klaim_superlatif_tanpa_bukti",
     "Klaim 'bahan murni/asli 100%' wajib dibuktikan otoritas resmi (EPI)"),
    (r"\bhalal\b", "klaim_halal_perlu_verifikasi",
     "Kata 'halal' hanya boleh dipakai jika produk bersertifikat resmi MUI (EPI)"),
]

# Pola larangan khusus kategori SKINCARE/KOSMETIK (BPOM 18/2024)
POLA_SKINCARE = [
    (r"\b(instan|semalam|dalam \d+ (hari|jam)|3 hari|seketika)\b", "klaim_waktu_instan",
     "Klaim hasil instan/waktu spesifik tanpa uji klinis terdaftar (BPOM 18/2024)"),
    (r"\b(menyembuhkan|mengobati|obat jerawat)\b", "klaim_medis",
     "Klaim menyembuhkan/mengobati masuk ranah obat, bukan kosmetik (BPOM 18/2024)"),
    (r"\b(ampuh|terbukti (secara )?ilmiah)\b", "klaim_superlatif_tanpa_bukti",
     "Klaim 'ampuh/terbukti ilmiah' butuh data uji terverifikasi (BPOM 18/2024)"),
    (r"\b(merapatkan|mengencangkan) (organ|payudara)\b", "klaim_tubuh_sensitif",
     "Klaim perubahan fungsi tubuh sensitif dilarang tanpa dasar hukum (BPOM 18/2024)"),
]

# Pola larangan yang relevan lintas kategori tapi sering muncul di kuliner/fashion
POLA_TAMBAHAN = [
    (r"\b(lebih baik dari|dibanding(kan)? (kompetitor|merek lain))\b", "merendahkan_kompetitor",
     "Iklan tidak boleh merendahkan produk pesaing (EPI)"),
    (r"\b(jangan sampai menyesal|bahaya jika tidak|rugi kalau tidak)\b", "memicu_ketakutan_berlebihan",
     "Iklan tidak boleh memicu rasa takut/cemas berlebihan (EPI)"),
]


def cek_rule_engine(teks: str, kategori: str) -> list:
    """
    Cek teks terhadap pola regex sesuai kategori produk.
    Return list of dict: {pola, kode_pelanggaran, alasan, kutipan_yang_match}
    """
    teks_lower = teks.lower()
    hasil = []

    pola_aktif = list(POLA_UNIVERSAL) + list(POLA_TAMBAHAN)
    # Kategori sekarang inputan bebas (bukan dropdown tetap), jadi cek pakai
    # "mengandung kata kunci" bukan exact match - supaya "Skincare", "skin care",
    # "kosmetik", "perawatan kulit" dll tetap kena pola BPOM skincare.
    kategori_lower = kategori.strip().lower()
    kata_kunci_skincare = ["skincare", "skin care", "kosmetik", "perawatan kulit", "perawatan wajah"]
    if any(kata in kategori_lower for kata in kata_kunci_skincare):
        pola_aktif += POLA_SKINCARE

    for pattern, kode, alasan in pola_aktif:
        match = re.search(pattern, teks_lower, re.IGNORECASE)
        if match:
            hasil.append({
                "kode_pelanggaran": kode,
                "alasan": alasan,
                "kutipan": match.group(0),
            })

    return hasil


def ringkas_hasil_rule_engine(hasil: list) -> str:
    """Format hasil rule engine jadi teks ringkas untuk dikirim ke LLM sebagai konteks."""
    if not hasil:
        return "Rule engine: tidak menemukan pola pelanggaran yang jelas."
    baris = ["Rule engine mendeteksi potensi pelanggaran berikut:"]
    for h in hasil:
        baris.append(f"- '{h['kutipan']}' -> {h['kode_pelanggaran']}: {h['alasan']}")
    return "\n".join(baris)
