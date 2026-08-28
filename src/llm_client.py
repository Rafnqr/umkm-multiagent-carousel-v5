"""
llm_client.py
Wrapper LLM berbasis LANGCHAIN (memenuhi soal 3: framework open source).
Default: Groq (WAJIB untuk versi deploy — Ollama tidak bisa jalan di Streamlit
Community Cloud karena butuh server lokal sendiri).

Pilih backend lewat environment variable LLM_BACKEND: "groq" (default) atau "ollama".

Setup cepat:
- Groq (dipakai untuk versi deploy): daftar gratis di console.groq.com, ambil API
  key, lalu simpan sebagai secret GROQ_API_KEY (lihat README bagian Deployment).
- Ollama (opsional, hanya untuk pengembangan/test lokal): install dari ollama.com,
  jalankan `ollama pull llama3.1` lalu `ollama serve`.

CATATAN PASCA-UPGRADE TAVILY & MIGRASI MODEL (Juli 2026): prompt tiap agent
sekarang lebih panjang (ikut menyertakan hasil Tavily live search + transkrip
diskusi penuh tiap putaran). Default model juga sudah dipindah dari
llama-3.1-8b-instant (deprecated Groq, shutdown 16 Agustus 2026) ke
openai/gpt-oss-20b - model open-weight OpenAI yang di-hosting Groq, TETAP
pakai GROQ_API_KEY yang sama, bukan API OpenAI. PENTING: GROQ_TPM_LIMIT di
config.py (6000) itu angka lama khusus llama-3.1-8b-instant - WAJIB dicek
ulang di console.groq.com/settings/limits untuk openai/gpt-oss-20b dan
disesuaikan, supaya rate tracker dashboard akurat. RETRY_MAX_ATTEMPTS di
bawah menangani rate limit dengan retry otomatis + backoff sebagai pengaman
kalau limit di config.py belum sempat disesuaikan.
"""

import os
import re
import time
import streamlit as st

from .tools.rate_tracker import catat_token_groq

LLM_BACKEND = os.environ.get("LLM_BACKEND", "groq")  # "groq" (default) atau "ollama"

# Retry untuk rate limit (429) - khusus backend Groq
RETRY_MAX_ATTEMPTS = 5
RETRY_BACKOFF_AWAL_DETIK = 3  # naik 2x tiap percobaan (3, 6, 12, 24, 48 detik)

# Retry TERPISAH untuk jawaban KOSONG/kena potong (bukan rate limit) - lihat
# catatan reasoning model di _get_chat_model(). Beda mekanisme jadi beda
# counter: rate limit nunggu lama (bisa puluhan detik), jawaban kosong cukup
# dicoba ulang cepat (kemungkinan besar cuma variasi acak generation).
RETRY_MAX_ATTEMPTS_KOSONG = 3
RETRY_JEDA_KOSONG_DETIK = 1.5


def _get_secret(key: str, default: str = None):
    """Ambil config dari st.secrets (Streamlit Cloud) dulu, fallback ke env var (lokal)."""
    try:
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    return os.environ.get(key, default)


def _get_chat_model(temperature: float = 0.7, max_tokens: int = 2048):
    """Bangun chat model LangChain sesuai backend yang aktif.

    PENTING soal max_tokens: default model Groq sekarang (openai/gpt-oss-20b)
    adalah REASONING MODEL - dia menghabiskan sebagian token budget untuk
    "berpikir" (reasoning tokens) SEBELUM menulis jawaban akhir yang terlihat
    di `response.content`. Kalau max_tokens tidak di-set eksplisit (dan
    sebelumnya memang tidak di-set), ada risiko nyata reasoning menghabiskan
    seluruh budget sebelum sempat menulis jawaban - hasilnya jawaban jadi
    KOSONG TOTAL atau TERPOTONG di tengah kalimat. Ini pola bug yang benar-benar
    terjadi (caption Localization pernah kosong & pernah terpotong "...experience
    the"). max_tokens dinaikkan cukup besar di sini sebagai mitigasi; lihat
    juga call_llm() untuk deteksi+retry kalau tetap kejadian."""
    if LLM_BACKEND == "ollama":
        from langchain_community.chat_models import ChatOllama

        model = os.environ.get("OLLAMA_MODEL", "llama3.1")
        return ChatOllama(model=model, temperature=temperature, num_predict=max_tokens)

    # Default: Groq
    from langchain_groq import ChatGroq

    api_key = _get_secret("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY belum diset. Untuk lokal: export GROQ_API_KEY=xxx. "
            "Untuk Streamlit Cloud: tambahkan di menu App settings > Secrets."
        )
    model = os.environ.get("GROQ_MODEL", "openai/gpt-oss-20b")
    return ChatGroq(api_key=api_key, model=model, temperature=temperature, max_tokens=max_tokens)


def _ambil_saran_tunggu_dari_pesan_error(pesan_error: str) -> float:
    """Groq kasih saran tunggu di pesan error, misal 'Please try again in 2.29s'.
    Kalau ketemu, pakai itu (+buffer kecil); kalau tidak, return None supaya
    caller pakai backoff default."""
    match = re.search(r"try again in ([\d.]+)s", pesan_error)
    if match:
        return float(match.group(1)) + 0.5  # +buffer kecil biar aman
    return None


def call_llm(system_prompt: str, user_prompt: str, temperature: float = 0.7, max_tokens: int = 2048) -> str:
    """Panggil LLM lewat LangChain chat model. Return teks respons.

    Kalau backend Groq kena rate limit (429), retry otomatis dengan backoff
    (memakai waktu tunggu yang disarankan Groq di pesan error kalau ada),
    maksimal RETRY_MAX_ATTEMPTS kali. Error lain (bukan rate limit) langsung
    dilempar ulang, tidak di-retry - supaya bug asli tidak tertutup retry loop.

    TAMBAHAN: kalau jawaban yang balik KOSONG/whitespace saja (indikasi kuat
    reasoning model "gpt-oss" menghabiskan token budget sebelum sempat
    menulis jawaban - lihat catatan di _get_chat_model), retry ulang sampai
    RETRY_MAX_ATTEMPTS_KOSONG kali. Ini BUKAN ditelan diam-diam jadi string
    kosong seperti sebelumnya - kalau semua percobaan tetap kosong, lempar
    RuntimeError yang jelas, supaya kelihatan di log, bukan caption kosong
    yang membingungkan user.
    """
    from langchain.schema import SystemMessage, HumanMessage

    chat_model = _get_chat_model(temperature=temperature, max_tokens=max_tokens)
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]

    percobaan = 0
    percobaan_kosong = 0
    while True:
        try:
            response = chat_model.invoke(messages)

            if LLM_BACKEND != "ollama":
                # Catat token ASLI dari response (bukan tebakan) ke rate
                # tracker, supaya dashboard bisa tampilkan sisa kuota Groq
                # yang akurat. Defensif: kalau field usage_metadata tidak
                # ada (versi langchain-groq beda), diam saja, jangan crash.
                try:
                    usage = getattr(response, "usage_metadata", None)
                    if usage:
                        catat_token_groq(usage.get("total_tokens", 0))
                except Exception:
                    pass

            konten = (response.content or "").strip()
            if not konten:
                percobaan_kosong += 1
                finish_reason = None
                try:
                    finish_reason = (getattr(response, "response_metadata", None) or {}).get("finish_reason")
                except Exception:
                    pass
                if percobaan_kosong <= RETRY_MAX_ATTEMPTS_KOSONG:
                    time.sleep(RETRY_JEDA_KOSONG_DETIK)
                    continue
                raise RuntimeError(
                    f"LLM mengembalikan jawaban KOSONG setelah {percobaan_kosong} percobaan "
                    f"(finish_reason={finish_reason!r}). Kemungkinan besar reasoning model "
                    "menghabiskan token budget sebelum sempat menulis jawaban akhir - coba "
                    "naikkan parameter max_tokens di pemanggilan call_llm ini."
                )

            return konten
        except Exception as e:
            pesan = str(e)
            is_rate_limit = (
                "rate_limit_exceeded" in pesan
                or "429" in pesan
                or type(e).__name__ == "RateLimitError"
            )
            percobaan += 1

            if not is_rate_limit or percobaan >= RETRY_MAX_ATTEMPTS:
                # Bukan rate limit, ATAU sudah habis jatah retry -> lempar
                # ulang error asli, jangan ditelan diam-diam
                raise

            tunggu = _ambil_saran_tunggu_dari_pesan_error(pesan)
            if tunggu is None:
                tunggu = RETRY_BACKOFF_AWAL_DETIK * (2 ** (percobaan - 1))

            time.sleep(tunggu)
