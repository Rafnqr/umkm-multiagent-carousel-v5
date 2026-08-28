# Sistem Multi-Agent Konten & Compliance UMKM

Proyek UAS ST167 — Proyek Data Mining. Sistem multi-agent LLM **kolaboratif**
(agent saling menanggapi, bukan pipeline satu arah) untuk menghasilkan copy
marketing yang persuasif, patuh regulasi lintas-sektor, dan efektif — untuk
UMKM dari bidang usaha apapun (skincare, fashion, kuliner, dll).

**Framework:** LangChain (soal 3). **Compliance:** Rule Engine (regex) + LLM,
bukan LLM murni — supaya hasil konsisten. **Alur:** kolaboratif, bukan pipeline
linier — Trend Agent aktif di tiap putaran, orchestrator mengambil keputusan
berbeda tergantung hasil Compliance & Effectiveness (lihat bagian "Cara Kerja").

**Sumber konteks per agent** (per-Juli 2026, upgrade dari versi RAG-penuh):

| Agent | Sumber konteks | Kenapa |
|---|---|---|
| Trend | Tavily live search | Butuh data real-time, RAG statis dulu "memaksakan" pola dari kategori lain untuk kategori "lainnya" |
| Effectiveness | Tavily live search (**PROXY**, bukan data engagement asli) | Data engagement asli butuh OAuth ke akun IG/TikTok UMKM, di luar scope API gratis |
| Compliance | RAG statis (regulasi inti) + Tavily (kasus terbaru, tambahan) | Teks hukum jarang berubah & butuh determinisme; Tavily cuma memperkaya penjelasan LLM |
| Localization | RAG statis | Referensi gaya bahasa, stabil, tidak butuh data real-time |

## Setup Lokal

### 1. Install dependency
```bash
pip install -r requirements.txt
```
> Windows: baris `pysqlite3-binary` otomatis dilewati (pakai environment
> marker), tidak perlu dihapus manual.

### 2. Siapkan API key Groq (wajib)
1. Daftar gratis di https://console.groq.com, buat API key
2. Copy `.streamlit/secrets.toml.example` jadi `.streamlit/secrets.toml`
3. Isi `GROQ_API_KEY` dengan key asli

### 2b. Siapkan API key Tavily (wajib untuk Trend & Effectiveness Agent)
1. Daftar gratis di https://tavily.com (plan Researcher: 1000 credit/bulan)
2. Isi `TAVILY_API_KEY` di `.streamlit/secrets.toml`
3. **Tanpa API key ini, Trend & Effectiveness Agent tetap jalan tapi tanpa
   konteks live** (hasil kurang optimal, bukan error) — lihat
   `src/tools/tavily_search.py`
4. **Tidak ada caching** — tiap panggilan agent selalu fresh search
   (keputusan sengaja, lihat bagian Keterbatasan). Pemakaian bulan ini bisa
   dipantau di sidebar dashboard.

### 3. Jalankan
```bash
python -m venv venv
source venv/Scripts/activate
pip install -r requirements.txt
streamlit run app.py
```
### 3. Coba
https://umkm-multiagent-carousel.streamlit.app/
### jika muncul error ini, pilih reboot
This app has gone over its resource limits.
It's using too much memory! A few possible solutions:
Upgrade the Streamlit version of this app to the latest release.
Follow these tips to make the app use less memory.
Reboot the app to clear the memory. The app might go over its resource limits again, though!
Fill out this form if you have a good-for-the-world use case and need more resources.
