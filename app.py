"""
app.py
Dashboard Streamlit - jalankan dengan: streamlit run app.py

Sistem Multi-Agent Konten & Compliance UMKM

Struktur tampilan (sesuai P8):
Pakai Caption Ini -> Skor -> Status -> Alasan Singkat -> Agent Discussion
(collapse) -> Konteks yang Dipakai Tiap Agent (collapse)
"""

import streamlit as st
from src.vectorstore import get_collections
from src.agents import build_agents
from src.orchestrator import run_discussion
from src.visual_pipeline import generate_carousel_content
from src.ragas_eval import hitung_faithfulness, hitung_answer_relevancy, hitung_context_precision_recall
from src.config import GROQ_TPM_LIMIT, GROQ_TPD_LIMIT, TAVILY_MONTHLY_LIMIT
from src.tools.tavily_search import get_usage_this_month
from src.tools.rate_tracker import (
    token_groq_terpakai_60_detik,
    groq_boleh_jalan,
    perkiraan_detik_sampai_reset_groq,
    tavily_boleh_jalan,
)
from src.tools.rate_tracker import (
    token_groq_terpakai_60_detik,
    token_groq_terpakai_24_jam,
    groq_boleh_jalan,
    perkiraan_detik_sampai_reset_groq,
    tavily_boleh_jalan,
)



st.set_page_config(page_title="Sistem Multi-Agent Konten & Compliance UMKM", layout="wide")

st.title("Sistem Multi-Agent Konten & Compliance UMKM")
st.caption(
    "Trend, Persuasion, Compliance, dan Effectiveness berdiskusi kolaboratif "
    "(bukan pipeline satu arah) sampai konsensus tercapai, lalu Localization "

    "menyesuaikan gaya bahasa final."
)

with st.sidebar:
    st.header("Input Produk & Usaha")
    nama_usaha = st.text_input("Nama usaha/brand", placeholder="Contoh: GlowNatura")
    kategori = st.text_input(
        "Kategori produk", placeholder="Contoh: skincare, kopi kekinian, kerajinan tangan, dll"
    )
    nama_produk = st.text_input("Nama produk", "Serum Wajah Vitamin C")
    deskripsi = st.text_area(
        "Deskripsi produk",
        "Serum dengan kandungan vitamin C untuk membantu kulit tampak lebih cerah.",
    )
    platform = st.selectbox("Target platform", ["Instagram", "Facebook", "TikTok"])
    tone = st.selectbox("Tone/gaya bahasa yang diinginkan",
                         ["Friendly/Hangat", "Playful/Gen Z", "Profesional/Formal", "Hangat-Tradisional"])

    st.divider()
    foto_produk_file = st.file_uploader(
        "Foto produk asli (opsional)",
        type=["jpg", "jpeg", "png"],
        help=(
            "Kalau diisi, foto ASLI ini yang dipakai untuk visual (bukan "
            "generate AI). Tetap dicek compliance sekali (tanpa regenerate, "
            "karena ini foto asli usaha). Kalau dikosongkan, sistem generate "
            "foto otomatis."
        ),
    )

    st.divider()
    st.caption("Status Kuota Sistem")

    # --- Mesin AI teks (limit per-menit, reset cepat) - nama vendor sengaja
    # tidak ditampilkan ke user demi kerahasiaan detail implementasi API ---
    terpakai_groq_harian = token_groq_terpakai_24_jam()
    groq_ok = groq_boleh_jalan()
    st.progress(min(terpakai_groq_harian / GROQ_TPD_LIMIT, 1.0))
    st.caption(
        f"Mesin AI Teks (harian): ~**{terpakai_groq_harian}/{GROQ_TPD_LIMIT}** "
        "unit/24 jam (estimasi rolling, bukan reset jam-tetap Groq)"
    )
    if not groq_ok:
        detik_tunggu = perkiraan_detik_sampai_reset_groq()
        st.warning(
            f"Kuota mesin AI teks lagi mepet/habis. Coba lagi dalam "
            f"~{detik_tunggu:.0f} detik."
        )

    # --- Riset tren real-time (limit bulanan, reset lambat) - TIDAK
    # memblokir tombol, karena kalau diblokir user harus nunggu sebulan
    # penuh. Sistem tetap jalan tanpa konteks live (fail gracefully).
    pemakaian_tavily = get_usage_this_month()
    tavily_ok = tavily_boleh_jalan(pemakaian_tavily)
    st.progress(min(pemakaian_tavily / TAVILY_MONTHLY_LIMIT, 1.0))
    st.caption(
        f"Riset Tren Real-time: **{pemakaian_tavily}/{TAVILY_MONTHLY_LIMIT}** unit bulan ini "
        "(reset bulanan, tanpa cache)"
    )
    if not tavily_ok:
        st.warning(
            "Kuota riset tren bulan ini habis. Trend & Effectiveness Agent tetap "
            "jalan tapi TANPA konteks live sampai reset bulan depan."
        )

    jalankan = st.button(
        "Jalankan Diskusi Agent",
        type="primary",
        disabled=not groq_ok,
        help=None if groq_ok else "Tunggu kuota mesin AI teks reset dulu (lihat status di atas).",
    )

if "collections" not in st.session_state:
    with st.spinner("Menyiapkan vector DB (sekali saja di awal sesi)..."):
        st.session_state.collections = get_collections()
        st.session_state.agents = build_agents(st.session_state.collections)

if jalankan:
    if not nama_usaha.strip():
        st.warning("Isi dulu nama usaha/brand di sidebar sebelum menjalankan diskusi.")
        st.stop()
    if not kategori.strip():
        st.warning("Isi dulu kategori produk di sidebar sebelum menjalankan diskusi.")
        st.stop()

    produk = {
        "nama": nama_produk,
        "deskripsi": deskripsi,
        "kategori": kategori,
        "nama_usaha": nama_usaha,
        "tone": tone,
        "platform": platform,
    }

    with st.spinner("Agent sedang berdiskusi (bisa beberapa putaran)..."):
        hasil = run_discussion(produk, st.session_state.agents)

    with st.spinner("Membuat carousel 3 slide (Cover/Isi/Penutup) + cek compliance..."):
        foto_bytes = foto_produk_file.read() if foto_produk_file else None
        foto_content_type = foto_produk_file.type if foto_produk_file else "image/jpeg"
        hasil_visual = generate_carousel_content(
            produk, hasil, st.session_state.agents["compliance"],
            foto_produk_upload=foto_bytes,
            foto_produk_content_type=foto_content_type,
        )

    st.session_state.hasil_terakhir = hasil
    st.session_state.hasil_visual_terakhir = hasil_visual
    st.session_state.produk_terakhir = produk

if "hasil_terakhir" in st.session_state:
    hasil = st.session_state.hasil_terakhir
    hasil_visual = st.session_state.get("hasil_visual_terakhir", {})
    produk = st.session_state.produk_terakhir
    m = hasil["metrik"]

    # ===== 1. PAKAI CAPTION INI =====
    st.subheader("Pakai Caption Ini")
    st.success(hasil["caption_final"])
    if hasil["hashtag_final"]:
        st.caption(hasil["hashtag_final"])
    status_recheck = m.get("compliance_recheck_status")
    if status_recheck == "FALLBACK_DRAFT_PENDEK":
        st.warning(
            "⚠️ Caption Hook-Value-CTA gagal lolos cek compliance ulang "
            "setelah beberapa revisi - sistem fallback ke draft pendek yang "
            "sudah pasti aman (bukan versi Hook-Value-CTA lengkap)."
        )

    # ===== 1b. CAROUSEL 3 SLIDE (Cover / Isi / Penutup) =====
    st.subheader("Carousel Visual (3 Slide)")
    sumber_foto = hasil_visual.get("sumber_foto")
    if sumber_foto == "UPLOAD_USER":
        st.caption("Foto produk ASLI kamu ditempel di atas background dekoratif AI-generate (produk tidak pernah diubah/di-crop).")
    else:
        st.caption("3 slide full AI-generate dengan gaya visual yang konsisten, berdasarkan deskripsi produk.")

    daftar_slide = hasil_visual.get("slide", [])
    kolom_slide = st.columns(len(daftar_slide)) if daftar_slide else []
    for idx, (kolom, slide) in enumerate(zip(kolom_slide, daftar_slide)):
        with kolom:
            st.markdown(f"**{slide['peran']}**")
            gambar_bytes = slide.get("gambar_bytes")
            status_v = slide.get("status")
            if gambar_bytes:
                st.image(gambar_bytes, use_column_width=True)
                if status_v == "AMAN":
                    st.caption("✅ Lolos compliance")
                elif status_v == "TOLAK_FOTO_USER":
                    st.warning("⚠️ Foto upload berpotensi melanggar - lihat detail di bawah.")
                elif status_v == "FALLBACK_LATAR_POLOS":
                    st.caption("⚠️ Background AI ditolak berulang - dipakai latar polos, produk tetap tampil.")
                st.download_button(
                    "⬇️ Download",
                    data=gambar_bytes,
                    file_name=f"slide_{slide['peran'].lower()}.png",
                    mime="image/png",
                    key=f"download_{idx}",
                )
            else:
                st.caption("❌ Gagal generate slide ini.")

    # ===== 2. SKOR =====
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Engagement", f"{m['skor_engagement_akhir']}/10")
    col2.metric("Brand Fit", f"{m['skor_brand_fit_akhir']}/10")
    col3.metric("Compliance", f"{m['skor_compliance_akhir']}/10")
    col4.metric("Overall", f"{m['skor_overall_akhir']}/10")

    # ===== 3. STATUS =====
    status_label = {
        "konsensus_tercapai": "✅ Konsensus tercapai",
        "batas_putaran_tercapai": "⚠️ Batas putaran tercapai (perlu revisi manual)",
    }.get(m["status_akhir"], m["status_akhir"])
    st.markdown(f"**Status:** {status_label}  |  **Jumlah putaran:** {m['putaran']}  |  "
                f"**Compliance ditolak:** {m['compliance_ditolak_berapa_kali']}x")

    # ===== 4. ALASAN SINGKAT (Explainability - P9) =====
    with st.expander("Kenapa caption ini yang dipilih? (Explainability)", expanded=True):
        st.markdown(hasil["explainability"])

    # ===== 5. AGENT DISCUSSION (Collapse) =====
    with st.expander("Lihat Transkrip Diskusi Agent Lengkap (klik untuk buka)"):
        for entry in hasil["transkrip"]:
            role = "user" if entry["agent"] in ("compliance", "orchestrator") else "assistant"
            with st.chat_message(role):
                st.markdown(f"**[Putaran {entry['putaran']}] {entry['agent'].upper()}**\n\n{entry['isi']}")

    # ===== 5a. PROSES VISUAL CONTENT GENERATOR (Collapse) =====
    with st.expander("Lihat Proses Pembuatan Konten Visual (klik untuk buka)"):
        sumber_label = {"UPLOAD_USER": "Foto upload kamu", "AI_GENERATED": "Foto AI-generated"}.get(
            hasil_visual.get("sumber_foto"), "-"
        )
        st.markdown(f"**Sumber foto:** {sumber_label}")
        desain_teks = hasil_visual.get("desain_teks")
        if desain_teks:
            for peran, teks in desain_teks.items():
                if peran == "Cover":
                    ringkasan = (
                        f"Nama Toko: \"{teks.get('nama_toko', '')}\" | "
                        f"Headline: \"{teks.get('headline', '')}\" | "
                        f"Subheadline: \"{teks.get('subheadline') or '(kosong)'}\" | "
                        f"Nama Produk: \"{teks.get('nama_produk', '')}\""
                    )
                elif peran == "Isi":
                    ringkasan = (
                        f"Nama Toko: \"{teks.get('nama_toko', '')}\" | "
                        f"Headline: \"{teks.get('headline', '')}\" | "
                        f"Subheadline: \"{teks.get('subheadline') or '(kosong)'}\" | "
                        f"Keunggulan: \"{teks.get('keunggulan_1', '')}\" · "
                        f"\"{teks.get('keunggulan_2', '')}\" · \"{teks.get('keunggulan_3', '')}\""
                    )
                else:  # Penutup
                    ringkasan = (
                        f"Nama Toko: \"{teks.get('nama_toko', '')}\" | "
                        f"Headline: \"{teks.get('headline', '')}\" | "
                        f"Tombol: \"{teks.get('cta') or '(kosong)'}\""
                    )
                st.markdown(f"**{peran}** — {ringkasan}")
        st.divider()
        log_visual = hasil_visual.get("log", [])
        if not log_visual:
            st.caption("Belum ada proses visual yang tercatat.")
        for entri in log_visual:
            st.markdown(f"**Percobaan {entri['percobaan']} — Status: {entri['status']}**")
            st.caption(f"Prompt gambar: {entri['prompt']}")
            if entri.get("deskripsi_vision"):
                st.caption(f"Hasil analisis visual: {entri['deskripsi_vision']}")
            st.caption(f"Alasan compliance: {entri['alasan_compliance']}")
            st.divider()

    # ===== 5b. SARAN KONTEN ALTERNATIF (VISUAL vs VIDEO) =====
    saran = hasil.get("saran_konten_alternatif", {})
    if saran.get("format_direkomendasikan") == "VIDEO" and saran.get("ide_video"):
        st.info(
            "**Catatan:** tren untuk kategori ini sebenarnya lebih optimal dalam "
            f"bentuk **video** ({saran.get('alasan_rekomendasi', '')}). Sistem ini "
            "tetap men-generate GAMBAR sebagai deliverable utama (lihat di bawah), "
            "tapi kalau usaha ini mau ikut tren secara penuh, pertimbangkan produksi "
            f"video terpisah dengan konsep: *{saran['ide_video']}*"
        )

    # ===== 6. KONTEKS YANG DIPAKAI TIAP AGENT =====
    SUMBER_KONTEKS = {
        "compliance": "Basis Pengetahuan Internal (dokumen regulasi)",
        "trend": "Riset Tren Real-time",
        "effectiveness": "Riset Tren Real-time (Proxy - bukan data engagement asli)",
        "localization": "Basis Pengetahuan Internal (gaya bahasa)",
    }
    with st.expander("Lihat Konteks yang Dipakai Tiap Agent (transparansi sumber)"):
        for agent_name, dokumen in hasil["dokumen_rag"].items():
            sumber = SUMBER_KONTEKS.get(agent_name, "-")
            st.markdown(f"**{agent_name.capitalize()}** — sumber: *{sumber}*")
            if dokumen:
                st.text(dokumen[:800] + ("..." if len(dokumen) > 800 else ""))
            else:
                st.caption("(kosong - kemungkinan API riset tren belum dikonfigurasi)")
            st.divider()

    # ===== 7. EVALUASI RAGAS (LIVE - Faithfulness & Answer Relevancy) =====
    st.subheader("Evaluasi Marketing & Sistem (RAGAS)")
    tab1, tab2 = st.tabs(["Evaluasi Sesi Ini (Live)", "Evaluasi Sistem (Batch)"])

    with tab1:
        if st.button("Hitung Faithfulness & Answer Relevancy untuk sesi ini"):
            with st.spinner("Menghitung metrik RAGAS live..."):
                konteks_gabungan = "\n".join(hasil["dokumen_rag"].values())
                faith = hitung_faithfulness(hasil["caption_final"], konteks_gabungan)
                relev = hitung_answer_relevancy(produk, hasil["caption_final"])
            c1, c2 = st.columns(2)
            c1.metric("Faithfulness", f"{faith['skor']:.2f}")
            c1.caption(faith["alasan"])
            c2.metric("Answer Relevancy", f"{relev['skor']:.2f}")
            c2.caption(relev["alasan"])

    with tab2:
        st.caption(
            "Context Precision & Context Recall diukur terhadap dataset kecil "
            "berlabel manual (bukan per-sesi), untuk menilai kualitas retriever "
            "Compliance Agent secara umum."
        )
        if st.button("Jalankan Evaluasi Sistem (Context Precision/Recall)"):
            with st.spinner("Menjalankan evaluasi terhadap dataset benchmark..."):
                hasil_eval = hitung_context_precision_recall(st.session_state.collections)
            c1, c2 = st.columns(2)
            c1.metric("Context Precision (rata-rata)", hasil_eval["context_precision_rata2"])
            c2.metric("Context Recall (rata-rata)", hasil_eval["context_recall_rata2"])
            st.dataframe(hasil_eval["detail_per_query"])
else:
    st.info("Isi form di sidebar lalu klik 'Jalankan Diskusi Agent' untuk mulai.")
