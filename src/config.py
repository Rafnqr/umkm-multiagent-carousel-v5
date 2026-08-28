"""
config.py
Konstanta terpusat supaya parameter sistem gampang di-tuning tanpa bongkar kode.
"""

import os

MAX_ROUNDS = 5

# Ambang skor Effectiveness (skala 1-10) untuk overall score
SKOR_AMBANG_OVERALL = 6

# Kategori produk yang didukung (dipakai untuk filter regulasi & tren)
KATEGORI_PRODUK = ["skincare", "fashion", "makanan", "lainnya"]

# Bobot untuk menghitung Overall dari 3 dimensi scoring matrix
BOBOT_SCORING = {
    "engagement": 0.4,
    "brand_fit": 0.3,
    "compliance": 0.3,
}

# ===== Limit API eksternal (untuk dashboard rate tracker) =====
# Groq: limit TOKEN PER MENIT (TPM), reset per-menit (rolling window).
# Angka di bawah CONFIRMED dari console.groq.com/settings/limits per model
# saat ini (openai/gpt-oss-20b & qwen/qwen3.6-27b: RPM 30, RPD 1000, TPM 8000,
# TPD 200000) - beda dari fallback lama (6000) yang cuma tebakan. TETAP cek
# ulang di console kalau model berubah, karena angka ini bisa berbeda per
# model dan bisa berubah sewaktu-waktu di sisi Groq. Override via env var
# GROQ_TPM_LIMIT kalau kamu ganti model/limit berubah.
GROQ_TPM_LIMIT = int(os.environ.get("GROQ_TPM_LIMIT", "8000"))

# RPD (Request Per Day) = 1000 untuk openai/gpt-oss-20b & qwen-vision -
# BUKAN cuma soal token, tapi JUMLAH PANGGILAN. Satu sesi generate caption
# bisa pakai 15-20+ request (5 agent x beberapa putaran x recheck), jadi RPD
# 1000 = kira-kira 50-65x generate/hari max, terlepas dari TPD masih sisa
# atau tidak. Dashboard rate tracker sebaiknya juga hitung ini kalau relevan.
GROQ_RPD_LIMIT = int(os.environ.get("GROQ_RPD_LIMIT", "1000"))

# TPD (Tokens Per Day) - RESET HARIAN (bukan rolling 60 detik seperti TPM).
# 200000 untuk openai/gpt-oss-20b & qwen/qwen3.6-27b (lihat console.groq.com/settings/limits).
# Override via env var GROQ_TPD_LIMIT kalau model/limit berubah.
GROQ_TPD_LIMIT = int(os.environ.get("GROQ_TPD_LIMIT", "200000"))

# Tavily: limit CREDIT PER BULAN, reset bulanan (BUKAN per-menit).
# 1000 adalah kuota plan Researcher (free tier).
TAVILY_MONTHLY_LIMIT = 1000

# ===== Visual Content Generator =====
# Maksimal percobaan generate gambar (termasuk gagal generate ATAU ditolak
# compliance) sebelum fallback ke gambar generik tanpa klaim visual berisiko.
VISUAL_MAX_RETRY = 3

# Maksimal percobaan revisi LocalizationAgent kalau compliance recheck di
# caption final (Hook-Value-CTA) TOLAK - lebih dari ini, fallback ke draft
# pendek yang sudah terverifikasi aman (lihat orchestrator.py).
LOCALIZATION_MAX_RETRY = 2

# Maksimal percobaan revisi LocalizationAgent kalau caption LOLOS compliance
# tapi gagal validasi programatik (bukan Bahasa Indonesia, atau bukan
# struktur Hook-Value-CTA yang benar - lihat validasi_konten.py). Terpisah
# dari LOCALIZATION_MAX_RETRY karena ini bukan soal compliance, jadi tidak
# perlu lewat ComplianceAgent lagi.
LOCALIZATION_FORMAT_MAX_RETRY = 2

# Jeda (detik) antar percobaan generate gambar Pollinations dalam 1 slide -
# Pollinations gratis kadang mulai throttle setelah beberapa request beruntun
# dalam 1 sesi; jeda kecil ini mengurangi risiko itu tanpa bikin user nunggu lama.
VISUAL_RETRY_DELAY_DETIK = 2.0

# Maksimal percobaan revisi Design Agent (_susun_desain_teks_carousel) kalau
# teks slide (headline/subheadline/CTA) TERNYATA gagal compliance recheck -
# lihat visual_pipeline.py. Lebih dari ini, fallback ke teks generik
# (nama produk saja) yang pasti tidak mengandung klaim.
DESIGN_TEXT_MAX_RETRY = 2
