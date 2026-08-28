"""
tools/vision_describe.py
Deskripsikan isi gambar jadi teks, pakai model vision Groq.

CATATAN MODEL (per Juli 2026): llama-3.2-11b/90b-vision-preview sudah
dideprecate Groq sejak 04/2025, penggantinya llama-4-scout JUGA baru
dideprecate 17/07/2026. Satu-satunya model vision aktif sekarang adalah
qwen/qwen3.6-27b - itu yang dipakai di sini. KALAU model ini nanti juga
dideprecate, cek console.groq.com/docs/vision untuk model vision aktif
terbaru dan ganti GROQ_VISION_MODEL.

CATATAN REASONING: qwen3.6-27b adalah REASONING model, sama seperti
openai/gpt-oss-20b di llm_client.py - kalau max_tokens tidak dibatasi, dia
bisa menghabiskan ratusan-ribuan token untuk <think>...</think> SEBELUM
menulis jawaban 3-5 kalimat yang sebenarnya dibutuhkan. max_tokens dibatasi
eksplisit di sini, dan <think> block SELALU di-strip dari hasil akhir.

CATATAN TRACKING TOKEN (fix - dashboard kuota sebelumnya tidak akurat):
fungsi ini manggil ChatGroq LANGSUNG (bukan lewat call_llm() di
llm_client.py), jadi sebelumnya token yang kepakai di sini TIDAK PERNAH
tercatat ke rate_tracker - dashboard "Status Kuota Sistem" jadi meleset,
terlihat aman padahal vision (foto upload + 3x background per carousel)
sebenarnya sudah memakan kuota nyata. Sekarang catat_token_groq() dipanggil
di sini juga, persis sama seperti call_llm(), dari usage_metadata response
asli (bukan tebakan).

Deskripsi teks hasil fungsi ini lalu diperiksa lewat
ComplianceAgent.evaluasi_teks() yang SUDAH ADA (reuse pipeline compliance
yang sama dengan draft copy teks, BUKAN rule engine baru untuk gambar -
lihat visual_pipeline.py dan alasan desain di rencana-lanjutan-upgrade-visual.md).
"""

import base64
import os
import re
import time

import streamlit as st

from .rate_tracker import catat_token_groq

GROQ_VISION_MODEL = os.environ.get("GROQ_VISION_MODEL", "qwen/qwen3.6-27b")

# max_tokens vision: jawaban final cuma 3-5 kalimat (~150-250 token), tapi
# reasoning model butuh budget tambahan utk <think> block-nya sebelum sampai
# ke jawaban. 700 dipilih sebagai batas aman.
_VISION_MAX_TOKENS = 700

_RETRY_MAX = 2
_RETRY_JEDA_DETIK = 2.0


def _get_secret(key: str, default=None):
    try:
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    return os.environ.get(key, default)


def _strip_think_block(teks: str) -> str:
    """Hapus blok <think>...</think> - model reasoning mengembalikan
    chain-of-thought penuh yang TIDAK dimaksudkan jadi output final."""
    if not teks:
        return teks
    return re.sub(r"<think>.*?</think>", "", teks, flags=re.DOTALL).strip()


def _catat_token_response(response):
    """Catat token ASLI dari response vision (usage_metadata) ke rate
    tracker - PERSIS pola yang sama dengan call_llm() di llm_client.py,
    supaya dashboard kuota mencerminkan pemakaian vision juga, bukan cuma
    pemakaian teks diskusi agent. Defensif: kalau field usage_metadata tidak
    ada (versi langchain-groq beda), diam saja, jangan crash."""
    try:
        usage = getattr(response, "usage_metadata", None)
        if usage:
            catat_token_groq(usage.get("total_tokens", 0))
    except Exception:
        pass


def describe_image(image_bytes: bytes, content_type: str, produk: dict) -> str:
    """
    Return deskripsi teks isi gambar, FOKUS ke elemen yang relevan untuk cek
    compliance (before-after, efek dramatis, simbol otoritas medis, dll) -
    BUKAN deskripsi estetika umum ("gambar bagus", "warna cerah"). Return
    string kosong kalau gagal (fail gracefully, caller anggap tidak ada info
    tambahan - visual_pipeline.py tetap jalankan rule engine di teks kosong,
    yang otomatis lolos tanpa temuan apapun karena tidak ada teks utk dicek).

    Hasil SELALU sudah bersih dari <think> block sebelum dikembalikan ke
    caller. Token pemakaian SELALU dicatat ke rate_tracker (kalau response
    berhasil didapat), baik hasil akhirnya kosong maupun tidak.
    """
    api_key = _get_secret("GROQ_API_KEY")
    if not api_key:
        return ""

    try:
        from langchain_groq import ChatGroq
        from langchain.schema import HumanMessage

        mime = content_type if content_type.startswith("image/") else "image/jpeg"
        b64 = base64.b64encode(image_bytes).decode("utf-8")
        model = ChatGroq(
            api_key=api_key, model=GROQ_VISION_MODEL, temperature=0.2,
            max_tokens=_VISION_MAX_TOKENS,
        )

        instruksi = (
            "Deskripsikan isi gambar ini secara faktual dan objektif, dengan "
            "FOKUS KHUSUS pada elemen yang berpotensi jadi KLAIM PEMASARAN, "
            f"untuk produk kategori '{produk['kategori']}'. Sebutkan secara "
            "eksplisit KALAU BENAR-BENAR ADA dan JELAS TERLIHAT: (1) "
            "perbandingan before-after atau efek dramatis yang ditampilkan, "
            "(2) teks/angka/persentase yang tertulis di gambar, (3) simbol "
            "otoritas medis/ilmiah (jubah dokter, logo sertifikasi, dll) "
            "yang tidak semestinya ada, (4) kondisi kulit/tubuh yang "
            "ditampilkan secara ekstrem/tidak wajar. PENTING: kalau gambar "
            "ini cuma background/elemen dekoratif abstrak (gradasi warna, "
            "pola, tekstur) TANPA produk/orang/teks sama sekali, JANGAN "
            "memaksakan salah satu dari 4 poin di atas ada hanya supaya "
            "instruksi ini terjawab - nyatakan dengan jelas bahwa gambar "
            "hanya elemen dekoratif TANPA klaim pemasaran apapun. LANGSUNG "
            "JAWAB TANPA proses berpikir panjang/step-by-step - tulis "
            "langsung 3-5 kalimat kesimpulan faktual, jangan menilai "
            "bagus/jelek secara estetika."
        )

        message = HumanMessage(
            content=[
                {"type": "text", "text": instruksi},
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
            ]
        )

        for percobaan in range(_RETRY_MAX + 1):
            try:
                response = model.invoke([message])
                _catat_token_response(response)
                return _strip_think_block(response.content)
            except Exception as e:
                pesan_error = str(e).lower()
                kemungkinan_rate_limit = "429" in pesan_error or "rate limit" in pesan_error
                if kemungkinan_rate_limit and percobaan < _RETRY_MAX:
                    time.sleep(_RETRY_JEDA_DETIK * (percobaan + 1))
                    continue
                return ""
    except Exception:
        return ""