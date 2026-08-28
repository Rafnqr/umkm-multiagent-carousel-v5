"""
ragas_eval.py
Implementasi ringan 4 metrik gaya-RAGAS. CATATAN JUJUR: ini BUKAN memanggil
library `ragas` resmi (yang butuh setup dataset batch terpisah dan kurang cocok
untuk skoring live per-sesi di dashboard). Ini implementasi custom yang meniru
metodologi RAGAS:

- Faithfulness & Answer Relevancy: dihitung LIVE tiap sesi diskusi (LLM-as-judge),
  karena keduanya memang didesain untuk menilai satu output spesifik.
- Context Precision & Context Recall: dihitung sebagai "Evaluasi Sistem" terpisah
  (bukan per-sesi), diuji terhadap dataset kecil berlabel manual di
  ragas_eval_dataset.py, karena keduanya menilai KUALITAS RETRIEVER secara umum,
  bukan satu jawaban spesifik.

Sesuai rekomendasi mentor: dipisah jadi "Evaluasi Sistem" vs "Evaluasi Marketing".
"""

from .llm_client import call_llm
from .parsing_utils import extract_number, extract_field
from .vectorstore import retrieve_docs_list
from .ragas_eval_dataset import EVAL_SET

JUDGE_FAITHFULNESS_PROMPT = """PERAN: Kamu adalah evaluator independen (LLM-as-judge).

TUGAS: Nilai apakah SEMUA klaim di dalam CAPTION didukung oleh KONTEKS yang
diberikan (regulasi + data tren + data performa yang dipakai selama diskusi).
Skor 1.0 = semua klaim didukung penuh oleh konteks (tidak ada halusinasi).
Skor 0.0 = banyak klaim yang mengada-ada, tidak ada dasarnya di konteks.

FORMAT OUTPUT (WAJIB):
SKOR: <angka 0.0-1.0>
ALASAN: <penjelasan singkat, sebutkan klaim mana yang didukung/tidak>
"""

JUDGE_RELEVANCY_PROMPT = """PERAN: Kamu adalah evaluator independen (LLM-as-judge).

TUGAS: Nilai seberapa RELEVAN caption yang dihasilkan terhadap deskripsi produk
asli yang diberikan user. Skor 1.0 = sangat relevan & tepat sasaran. Skor 0.0 =
tidak nyambung dengan produk.

FORMAT OUTPUT (WAJIB):
SKOR: <angka 0.0-1.0>
ALASAN: <penjelasan singkat>
"""


def hitung_faithfulness(caption: str, konteks_dipakai: str) -> dict:
    """Live metric - dipanggil tiap sesi diskusi selesai."""
    user_prompt = f"KONTEKS:\n{konteks_dipakai}\n\nCAPTION:\n{caption}"
    response = call_llm(JUDGE_FAITHFULNESS_PROMPT, user_prompt, temperature=0.0)
    return {
        "skor": extract_number(response, "SKOR", default=0.5),
        "alasan": extract_field(response, "ALASAN"),
    }


def hitung_answer_relevancy(produk: dict, caption: str) -> dict:
    """Live metric - dipanggil tiap sesi diskusi selesai."""
    user_prompt = (
        f"Deskripsi produk asli: {produk['deskripsi']}\n\nCaption yang dihasilkan:\n{caption}"
    )
    response = call_llm(JUDGE_RELEVANCY_PROMPT, user_prompt, temperature=0.0)
    return {
        "skor": extract_number(response, "SKOR", default=0.5),
        "alasan": extract_field(response, "ALASAN"),
    }


def hitung_context_precision_recall(collections: dict) -> dict:
    """
    Evaluasi Sistem (bukan per-sesi) - jalankan dataset kecil berlabel manual
    terhadap retriever Compliance Agent, ukur precision & recall retrieval.
    """
    detail = []
    precision_scores = []
    recall_scores = []

    for item in EVAL_SET:
        chunks = retrieve_docs_list(collections, "compliance", item["query"], n_results=3)
        chunks_lower = [c.lower() for c in chunks]
        keywords = [k.lower() for k in item["ground_truth_keywords"]]

        # Precision: dari chunk yang di-retrieve, berapa % yang relevan (mengandung >=1 keyword)
        relevan_chunks = sum(1 for c in chunks_lower if any(k in c for k in keywords))
        precision = relevan_chunks / len(chunks_lower) if chunks_lower else 0.0

        # Recall: dari semua keyword ground truth, berapa % yang ketemu di gabungan chunk
        gabungan = " ".join(chunks_lower)
        keyword_ketemu = sum(1 for k in keywords if k in gabungan)
        recall = keyword_ketemu / len(keywords) if keywords else 0.0

        precision_scores.append(precision)
        recall_scores.append(recall)
        detail.append({
            "query": item["query"],
            "precision": round(precision, 2),
            "recall": round(recall, 2),
        })

    return {
        "context_precision_rata2": round(sum(precision_scores) / len(precision_scores), 2),
        "context_recall_rata2": round(sum(recall_scores) / len(recall_scores), 2),
        "detail_per_query": detail,
    }
