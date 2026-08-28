"""
parsing_utils.py
Helper untuk mengekstrak field terstruktur (misal "CAPTION: ...", "SKOR: ...")
dari teks respons LLM. Dipakai karena semua agent sekarang punya Output Format
yang eksplisit (P2), supaya orchestrator/dashboard bisa ambil bagian yang
relevan tanpa harus parsing manual tiap tempat.
"""

import re


def extract_field(text: str, field_name: str) -> str:
    """
    Ambil isi setelah 'FIELD_NAME:' sampai baris field lain berikutnya atau akhir teks.
    Return string kosong kalau field tidak ditemukan (bukan None, supaya aman dipakai
    langsung tanpa cek None di banyak tempat).
    """
    pattern = rf"{field_name}\s*:\s*(.+?)(?=\n[A-Z_]{{2,}}\s*:|\Z)"
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else ""


def extract_number(text: str, field_name: str, default: float = 0.0) -> float:
    """Ambil angka pertama dari isi field tertentu, misal 'SKOR: 7/10' -> 7.0."""
    val = extract_field(text, field_name)
    if not val:
        return default
    match = re.search(r"(\d+(\.\d+)?)", val)
    return float(match.group(1)) if match else default


def extract_verdict(text: str) -> str:
    """Ambil AMAN/TOLAK dari awal teks compliance (case-insensitive)."""
    stripped = text.strip().upper()
    if stripped.startswith("AMAN"):
        return "AMAN"
    if stripped.startswith("TOLAK"):
        return "TOLAK"
    # fallback: cari di mana saja di teks, default TOLAK (fail-safe, bukan fail-open)
    if "AMAN" in stripped and "TOLAK" not in stripped:
        return "AMAN"
    return "TOLAK"
