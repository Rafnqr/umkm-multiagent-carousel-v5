"""
orchestrator.py
Alur diskusi KOLABORATIF antar 5 agent (P1) - bukan pipeline satu arah.

Trend Agent ikut aktif di SETIAP putaran (bukan cuma sekali di awal), dan
orchestrator mengambil keputusan routing yang berbeda tergantung hasil (P7):
- Compliance TOLAK -> panggil ulang Trend (perlu sudut pandang baru)
- Compliance AMAN tapi skor efektivitas rendah -> cukup Persuasion revisi
  (skip Trend, hemat waktu & API call)
- Compliance AMAN dan skor efektivitas cukup -> konsensus tercapai

Setiap keputusan orchestrator dicatat di transkrip sebagai entry tersendiri,
supaya proses pengambilan keputusan terlihat jelas (bukan black box).
"""

from .parsing_utils import extract_verdict, extract_number, extract_field
from .config import MAX_ROUNDS, SKOR_AMBANG_OVERALL, LOCALIZATION_MAX_RETRY, LOCALIZATION_FORMAT_MAX_RETRY
from .vectorstore import retrieve_context
from .validasi_konten import validasi_bahasa_indonesia, validasi_struktur_hook_value_cta


def _log(transkrip: list, agent: str, isi: str, putaran: int, log_callback=None):
    entry = {"agent": agent, "isi": isi, "putaran": putaran}
    transkrip.append(entry)
    if log_callback:
        log_callback(entry)


def run_discussion(produk: dict, agents: dict, log_callback=None) -> dict:
    """
    Jalankan diskusi kolaboratif multi-agent untuk satu produk.
    Return dict berisi transkrip lengkap, caption final, metrik, dan explainability.
    """
    transkrip = []
    metrik = {
        "putaran": 0,
        "compliance_ditolak_berapa_kali": 0,
        "skor_engagement_akhir": 0,
        "skor_brand_fit_akhir": 0,
        "skor_compliance_akhir": 0,
        "skor_overall_akhir": 0,
        "status_akhir": "gagal",
    }

    include_trend = True  # putaran pertama selalu mulai dari Trend
    verdict = "TOLAK"
    overall = 0
    trend_output = ""
    effectiveness_output = ""
    compliance_output = ""

    for putaran in range(1, MAX_ROUNDS + 1):
        metrik["putaran"] = putaran

        # 1. Trend (kondisional - hanya kalau include_trend True)
        if include_trend:
            trend_output = agents["trend"].respond(produk, transkrip, putaran)
            _log(transkrip, "trend", trend_output, putaran, log_callback)

        # 2. Persuasion (selalu jalan tiap putaran)
        persuasion_output = agents["persuasion"].respond(produk, transkrip, putaran)
        _log(transkrip, "persuasion", persuasion_output, putaran, log_callback)

        # 3. Compliance (rule engine + LLM)
        compliance_output = agents["compliance"].respond(produk, transkrip, putaran)
        _log(transkrip, "compliance", compliance_output, putaran, log_callback)
        verdict = extract_verdict(compliance_output)
        if verdict == "TOLAK":
            metrik["compliance_ditolak_berapa_kali"] += 1

        # 4. Effectiveness (scoring matrix 3 dimensi)
        effectiveness_output = agents["effectiveness"].respond(produk, transkrip, putaran)
        _log(transkrip, "effectiveness", effectiveness_output, putaran, log_callback)
        overall = extract_number(effectiveness_output, "OVERALL")

        # 5. Keputusan orchestrator (P7) - dicatat sebagai entry tersendiri
        if verdict == "AMAN" and overall >= SKOR_AMBANG_OVERALL:
            keputusan = (
                f"KONSENSUS TERCAPAI. Compliance = AMAN, skor overall {overall} "
                f">= ambang {SKOR_AMBANG_OVERALL}. Lanjut ke Localization."
            )
            _log(transkrip, "orchestrator", keputusan, putaran, log_callback)
            metrik["status_akhir"] = "konsensus_tercapai"
            break
        elif verdict == "TOLAK":
            keputusan = (
                "Compliance MENOLAK draf. Keputusan: panggil ulang TREND AGENT "
                "di putaran berikutnya untuk mencari sudut pandang baru yang "
                "tetap menarik tapi menghindari elemen yang ditolak."
            )
            _log(transkrip, "orchestrator", keputusan, putaran, log_callback)
            include_trend = True
        else:  # AMAN tapi overall < ambang
            keputusan = (
                f"Compliance = AMAN, tapi skor overall {overall} masih di bawah "
                f"ambang {SKOR_AMBANG_OVERALL}. Keputusan: TIDAK perlu panggil "
                "ulang Trend Agent - cukup PERSUASION AGENT merevisi draf dengan "
                "angle yang sama untuk efisiensi."
            )
            _log(transkrip, "orchestrator", keputusan, putaran, log_callback)
            include_trend = False
    else:
        metrik["status_akhir"] = "batas_putaran_tercapai"

    # Ambil skor detail dari Effectiveness Agent putaran terakhir
    metrik["skor_engagement_akhir"] = extract_number(effectiveness_output, "ENGAGEMENT")
    metrik["skor_brand_fit_akhir"] = extract_number(effectiveness_output, "BRAND_FIT")
    metrik["skor_compliance_akhir"] = extract_number(effectiveness_output, "COMPLIANCE")
    metrik["skor_overall_akhir"] = overall

    # 6. Localization - jalan setelah diskusi utama selesai/dihentikan
    localization_output = agents["localization"].respond(produk, transkrip, metrik["putaran"])
    _log(transkrip, "localization", localization_output, metrik["putaran"], log_callback)

    caption_final = extract_field(localization_output, "CAPTION") or localization_output
    hashtag_final = extract_field(localization_output, "HASHTAG")

    # 6b. COMPLIANCE RECHECK di caption final (Hook-Value-CTA) - PENTING:
    # draft pendek dari Persuasion sudah dicek Compliance, TAPI caption final
    # yang direstrukturisasi jadi 150-200 kata BISA saja "membengkak" dengan
    # detail yang belum pernah dicek. Reuse pipeline compliance yang sama
    # (BUKAN rule engine baru), dengan retry terbatas ke Localization kalau
    # ternyata versi panjangnya melanggar.
    status_compliance_final = "AMAN"
    for percobaan_recheck in range(LOCALIZATION_MAX_RETRY + 1):
        hasil_compliance_final = agents["compliance"].evaluasi_teks(produk, caption_final)
        verdict_final = extract_verdict(hasil_compliance_final)
        _log(
            transkrip, "compliance",
            f"[RECHECK CAPTION FINAL - percobaan {percobaan_recheck + 1}]\n{hasil_compliance_final}",
            metrik["putaran"], log_callback,
        )
        if verdict_final == "AMAN":
            status_compliance_final = "AMAN"
            break

        status_compliance_final = "TOLAK"
        alasan_final = extract_field(hasil_compliance_final, "ALASAN")
        saran_final = extract_field(hasil_compliance_final, "SARAN_PERBAIKAN")

        if percobaan_recheck < LOCALIZATION_MAX_RETRY:
            localization_output = agents["localization"].revisi(
                produk, caption_final, alasan_final, saran_final
            )
            _log(transkrip, "localization", localization_output, metrik["putaran"], log_callback)
            caption_final = extract_field(localization_output, "CAPTION") or caption_final
            hashtag_final = extract_field(localization_output, "HASHTAG") or hashtag_final
        else:
            # Retry habis - fallback ke draft pendek yang SUDAH terverifikasi
            # aman di loop diskusi utama (persuasion_output putaran terakhir),
            # BUKAN caption Hook-Value-CTA yang berulang kali gagal recheck.
            caption_final = extract_field(persuasion_output, "DRAFT") or caption_final
            hashtag_final = ""
            status_compliance_final = "FALLBACK_DRAFT_PENDEK"

    metrik["compliance_recheck_status"] = status_compliance_final

    # 6c. VALIDASI FORMAT (bahasa & struktur) - PROGRAMATIK, bukan LLM lagi.
    # Instruksi "WAJIB Bahasa Indonesia" / "WAJIB narasi Hook-Value-CTA" di
    # system prompt LocalizationAgent membantu, tapi LLM tetap probabilistik -
    # log nyata pernah menunjukkan caption jadi Bahasa Inggris, atau tetap
    # berbentuk daftar spesifikasi baris-per-baris walau instruksinya sudah
    # ada. Ini jaring pengaman KODE terakhir sebelum caption tampil ke user.
    metrik["validasi_format_status"] = "OK"
    for percobaan_format in range(LOCALIZATION_FORMAT_MAX_RETRY + 1):
        ok_bahasa, alasan_bahasa = validasi_bahasa_indonesia(caption_final)
        ok_struktur, alasan_struktur = validasi_struktur_hook_value_cta(caption_final)
        if ok_bahasa and ok_struktur:
            break

        masalah = " ".join(filter(None, [
            "" if ok_bahasa else alasan_bahasa,
            "" if ok_struktur else alasan_struktur,
        ]))
        _log(
            transkrip, "orchestrator",
            f"[VALIDASI FORMAT - percobaan {percobaan_format + 1}] GAGAL: {masalah}",
            metrik["putaran"], log_callback,
        )

        if percobaan_format >= LOCALIZATION_FORMAT_MAX_RETRY:
            metrik["validasi_format_status"] = f"GAGAL_SETELAH_RETRY: {masalah}"
            break

        caption_sebelum_perbaikan = caption_final
        hashtag_sebelum_perbaikan = hashtag_final
        localization_output = agents["localization"].perbaiki_format(produk, caption_final, masalah)
        _log(transkrip, "localization", localization_output, metrik["putaran"], log_callback)
        caption_final = extract_field(localization_output, "CAPTION") or caption_final
        hashtag_final = extract_field(localization_output, "HASHTAG") or hashtag_final

        # Perbaikan format seharusnya tidak menambah fakta baru, tapi tetap
        # dicek ulang untuk jaga-jaga (bukan diasumsikan aman begitu saja) -
        # kalau ternyata versi baru malah TOLAK, jangan dipakai, kembalikan
        # ke versi sebelumnya dan hentikan percobaan format (caption yang
        # formatnya kurang ideal tapi pasti aman > caption baru yang berisiko).
        hasil_compliance_ulang = agents["compliance"].evaluasi_teks(produk, caption_final)
        _log(
            transkrip, "compliance",
            f"[RECHECK ULANG SETELAH PERBAIKAN FORMAT]\n{hasil_compliance_ulang}",
            metrik["putaran"], log_callback,
        )
        if extract_verdict(hasil_compliance_ulang) == "TOLAK":
            caption_final = caption_sebelum_perbaikan
            hashtag_final = hashtag_sebelum_perbaikan
            metrik["validasi_format_status"] = "DIBATALKAN_RISIKO_COMPLIANCE"
            break

    # 7. Explainability (P9) - ringkasan "caption ini dipilih karena..."
    explainability = _bangun_explainability(transkrip, metrik)

    # Saran konten alternatif (visual vs video) dari Trend Agent putaran
    # terakhir - VISUAL selalu jadi dasar draft (lihat PersuasionAgent),
    # VIDEO cuma referensi buat usaha kalau mau produksi terpisah.
    saran_konten_alternatif = {
        "format_direkomendasikan": extract_field(trend_output, "FORMAT_REKOMENDASI") or "VISUAL",
        "ide_visual": extract_field(trend_output, "TREND_IDEA_VISUAL"),
        "ide_video": extract_field(trend_output, "TREND_IDEA_VIDEO"),
        "alasan_rekomendasi": extract_field(trend_output, "ALASAN_REKOMENDASI"),
    }

    # Kumpulkan konteks yang dipakai tiap agent (untuk transparansi dashboard -
    # P8). Compliance & Localization masih RAG ChromaDB statis. Trend &
    # Effectiveness sekarang sumbernya riset tren real-time - diambil dari
    # last_context yang disimpan tiap agent di respond() terakhirnya, BUKAN
    # dari retrieve_context() (yang akan selalu kosong untuk keduanya karena
    # sudah tidak ada di CORPUS_MAP).
    collections = agents["trend"].collections
    dokumen_rag = {
        "compliance": retrieve_context(collections, "compliance", f"aturan klaim untuk kategori {produk['kategori']}"),
        "trend": agents["trend"].last_context,
        "effectiveness": agents["effectiveness"].last_context,
        "localization": retrieve_context(collections, "localization", f"gaya bahasa tone {produk.get('tone', '')}"),
    }

    return {
        "transkrip": transkrip,
        "caption_final": caption_final,
        "hashtag_final": hashtag_final,
        "metrik": metrik,
        "explainability": explainability,
        "dokumen_rag": dokumen_rag,
        "saran_konten_alternatif": saran_konten_alternatif,
    }


def _bangun_explainability(transkrip: list, metrik: dict) -> str:
    """Susun ringkasan singkat kenapa caption final dipilih, berdasarkan putaran terakhir."""
    putaran_akhir = metrik["putaran"]
    entries_akhir = [e for e in transkrip if e["putaran"] == putaran_akhir]

    baris = ["Caption ini dipilih karena:"]
    for entry in entries_akhir:
        agent = entry["agent"]
        isi = entry["isi"]
        if agent == "trend":
            ringkas = extract_field(isi, "TREND_IDEA_VISUAL") or isi[:100]
            baris.append(f"- Trend: mengusulkan angle visual '{ringkas}'")
        elif agent == "compliance":
            alasan = extract_field(isi, "ALASAN") or isi[:100]
            baris.append(f"- Compliance: {alasan}")
        elif agent == "effectiveness":
            alasan = extract_field(isi, "ALASAN") or isi[:100]
            overall = extract_number(isi, "OVERALL")
            baris.append(f"- Effectiveness: skor overall {overall}/10 - {alasan}")
        elif agent == "localization":
            baris.append("- Localization: gaya bahasa & hashtag disesuaikan dengan identitas usaha dan platform target")

    return "\n".join(baris)
