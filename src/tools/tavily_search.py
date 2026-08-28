"""
tools/tavily_search.py
Wrapper Tavily Search API untuk Trend Agent, Effectiveness Agent, dan
tambahan konteks kasus terbaru untuk Compliance Agent.

KEPUTUSAN EKSPLISIT: TANPA CACHING - tiap panggilan selalu fresh search.
Konsekuensinya kuota gratis (1000 credit/bulan, plan Researcher) bisa lebih
cepat habis kalau sering re-test kategori yang sama saat development. Untuk
itu ada logging pemakaian sederhana di bawah (BUKAN cache hasil - cuma
penghitung), supaya kelihatan sebelum kuota benar-benar habis.
"""

import os
import json
from datetime import datetime, timezone

import streamlit as st

USAGE_LOG_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "_tavily_usage_log.json"
)

# Ambang peringatan (dari sisa 1000 credit/bulan plan Researcher) - dicek
# tiap panggilan, cuma untuk keperluan tampilan/log, tidak menghentikan apapun
AMBANG_PERINGATAN = [800, 900, 950]


def _get_secret(key: str, default=None):
    """Ambil config dari st.secrets (Streamlit Cloud) dulu, fallback ke env var (lokal)."""
    try:
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    return os.environ.get(key, default)


def _catat_pemakaian() -> int:
    """Tambah 1 hitungan pemakaian ke log lokal per-bulan. Ini BUKAN cache hasil
    pencarian - cuma penghitung jumlah request, supaya bisa dipantau di dashboard."""
    bulan_ini = datetime.now(timezone.utc).strftime("%Y-%m")
    log = {"bulan": bulan_ini, "jumlah_request": 0}

    try:
        if os.path.exists(USAGE_LOG_PATH):
            with open(USAGE_LOG_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            if data.get("bulan") == bulan_ini:
                log = data
    except Exception:
        pass

    log["jumlah_request"] += 1

    try:
        os.makedirs(os.path.dirname(USAGE_LOG_PATH), exist_ok=True)
        with open(USAGE_LOG_PATH, "w", encoding="utf-8") as f:
            json.dump(log, f)
    except Exception:
        pass  # gagal tulis log tidak boleh menghentikan pencarian

    return log["jumlah_request"]


def get_usage_this_month() -> int:
    """Baca jumlah pemakaian Tavily bulan ini (untuk ditampilkan di dashboard/README)."""
    bulan_ini = datetime.now(timezone.utc).strftime("%Y-%m")
    try:
        with open(USAGE_LOG_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if data.get("bulan") == bulan_ini:
            return data.get("jumlah_request", 0)
    except Exception:
        pass
    return 0


def tavily_search(query: str, max_results: int = 3) -> str:
    """
    Panggil Tavily Search API secara live, TANPA CACHE (selalu fresh search).

    Return teks konteks gabungan hasil pencarian (siap dipakai sebagai context
    LLM), atau string kosong kalau API key belum diset / request gagal - agar
    agent yang memanggil ini tetap bisa jalan (fail gracefully) alih-alih
    membuat seluruh sesi diskusi crash karena 1 API eksternal bermasalah.
    """
    api_key = _get_secret("TAVILY_API_KEY")
    if not api_key:
        # Sengaja tidak raise error di sini - Trend/Effectiveness/Compliance
        # Agent tetap harus bisa jalan (dengan konteks kosong) supaya orang
        # yang belum sempat daftar Tavily masih bisa lihat sistem berjalan,
        # meski hasilnya kurang optimal tanpa konteks live.
        return ""

    try:
        from tavily import TavilyClient

        client = TavilyClient(api_key=api_key)
        hasil = client.search(query=query, max_results=max_results, search_depth="basic")
    except Exception:
        return ""

    _catat_pemakaian()

    potongan = []
    for item in hasil.get("results", []):
        judul = item.get("title", "")
        konten = item.get("content", "")
        if konten:
            potongan.append(f"[{judul}]\n{konten}")

    return "\n---\n".join(potongan)
