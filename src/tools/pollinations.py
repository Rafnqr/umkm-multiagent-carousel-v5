"""
tools/pollinations.py
Wrapper generate gambar via Pollinations.ai - GRATIS, TANPA API KEY.

Catatan jujur (lihat rencana-lanjutan-upgrade-visual.md):
- Kualitas & konsistensi lebih rendah dibanding layanan berbayar, dan server
  kadang tidak stabil - ini trade-off yang disengaja demi tetap 100% gratis
  tanpa perlu daftar apapun.
- Pakai parameter `seed` kalau butuh hasil yang bisa direplikasi (misal untuk
  laporan/demo yang konsisten). Untuk retry (generate ulang karena ditolak
  compliance), SENGAJA pakai seed berbeda tiap percobaan supaya hasilnya
  benar-benar bervariasi, bukan gambar yang sama persis.

PENTING - log nyata (lihat log generator visual) menunjukkan pola: slide
pertama (Cover) berhasil, tapi slide berikutnya (Isi/Penutup) makin sering
GAGAL_GENERATE murni (bukan ditolak compliance) - konsisten dengan layanan
gratis ini mulai throttle setelah beberapa request beruntun dalam 1 sesi.
Dulu TIDAK ADA retry/backoff sama sekali di level ini (beda dengan
llm_client.call_llm yang punya retry 429) - sekarang ditambahkan retry
ringan KHUSUS untuk kasus yang kemungkinan besar throttling (status 429 atau
timeout), supaya generate_image tidak langsung menyerah di percobaan
pertama kalau memang cuma soal timing.
"""

import random
import time
import urllib.parse

import requests

BASE_URL = "https://image.pollinations.ai/prompt"
TIMEOUT_DETIK = 30

# Retry INTERNAL (di dalam 1 pemanggilan generate_image) khusus untuk
# indikasi throttling (429) atau timeout - TERPISAH dari retry di
# visual_pipeline.py (yang retry dengan PROMPT baru kalau kena TOLAK
# compliance). Retry di sini murni soal "server lagi sibuk", jadi pakai
# prompt/seed yang sama, cuma dicoba lagi setelah jeda singkat.
_RETRY_INTERNAL_MAX = 2
_RETRY_INTERNAL_JEDA_DETIK = 2.0


def generate_image(prompt: str, seed: int = None, width: int = 1024, height: int = 1024) -> dict:
    """
    Generate gambar dari prompt teks.
    Return {"bytes": <image bytes>, "content_type": <mime asli>} kalau
    berhasil, atau None kalau gagal (fail gracefully - caller yang
    menentukan fallback). Kalau gagal, alasan detailnya (status code/exception)
    ditulis ke `generate_image.alasan_gagal_terakhir` supaya bisa dilog
    caller tanpa mengubah signature return yang sudah dipakai di mana-mana.
    """
    if seed is None:
        seed = random.randint(1, 999999)

    prompt_encoded = urllib.parse.quote(prompt)
    url = f"{BASE_URL}/{prompt_encoded}"
    params = {
        "width": width,
        "height": height,
        "seed": seed,
        "nologo": "true",
        "model": "flux",
    }

    generate_image.alasan_gagal_terakhir = ""

    for percobaan_internal in range(_RETRY_INTERNAL_MAX + 1):
        try:
            resp = requests.get(url, params=params, timeout=TIMEOUT_DETIK)

            if resp.status_code == 429:
                generate_image.alasan_gagal_terakhir = (
                    f"HTTP 429 (rate limited) pada percobaan internal {percobaan_internal + 1}"
                )
                if percobaan_internal < _RETRY_INTERNAL_MAX:
                    time.sleep(_RETRY_INTERNAL_JEDA_DETIK * (percobaan_internal + 1))
                    continue
                return None

            resp.raise_for_status()
            content_type = resp.headers.get("content-type", "")
            if not content_type.startswith("image/"):
                # Server kadang balas HTML/error alih-alih gambar - anggap gagal
                generate_image.alasan_gagal_terakhir = (
                    f"Content-type bukan gambar: {content_type!r}"
                )
                return None
            return {"bytes": resp.content, "content_type": content_type}

        except requests.exceptions.Timeout:
            generate_image.alasan_gagal_terakhir = (
                f"Timeout setelah {TIMEOUT_DETIK}s pada percobaan internal {percobaan_internal + 1}"
            )
            if percobaan_internal < _RETRY_INTERNAL_MAX:
                time.sleep(_RETRY_INTERNAL_JEDA_DETIK * (percobaan_internal + 1))
                continue
            return None
        except Exception as e:
            generate_image.alasan_gagal_terakhir = f"{type(e).__name__}: {e}"
            return None

    return None


generate_image.alasan_gagal_terakhir = ""
