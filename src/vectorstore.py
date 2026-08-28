"""
vectorstore.py
Membangun & mengelola vector store terpisah untuk tiap agent (RAG per-domain),
dibangun DI ATAS LANGCHAIN (memenuhi soal 3: framework open source Langchain/Llama/AutoGen).

Tiap agent hanya "melihat" korpus yang relevan dengan perannya.
"""

import os

# --- Fix wajib untuk deploy di Streamlit Community Cloud ---
# Streamlit Cloud memakai versi sqlite3 bawaan sistem yang terlalu lama untuk
# ChromaDB. pysqlite3-binary menyediakan versi lebih baru; baris ini menggantikan
# modul sqlite3 bawaan dengan pysqlite3 SEBELUM chromadb di-import.
try:
    __import__("pysqlite3")
    import sys

    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass  # aman diabaikan kalau jalan di lokal (bukan Streamlit Cloud)

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
DB_DIR = os.path.join(os.path.dirname(__file__), "..", "chroma_db")

# Embedding model open source, berbasis ONNX (fastembed) - TIDAK butuh
# PyTorch/accelerate sama sekali, jadi bebas dari bug "meta tensor" yang sering
# muncul di kombinasi torch/transformers/accelerate tertentu. Model multilingual
# supaya cocok untuk teks Bahasa Indonesia. Nama model sudah dicek terdaftar
# resmi di fastembed.TextEmbedding.list_supported_models().
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Pemetaan agent -> file sumber data.
# CATATAN: "trend" & "effectiveness" SENGAJA tidak ada di sini lagi - kedua
# agent itu sudah pindah ke Tavily live search (lihat src/tools/tavily_search.py
# dan src/agents.py), bukan RAG ChromaDB. Compliance (regulasi inti, butuh
# determinisme) & Localization (gaya bahasa) tetap RAG statis.
CORPUS_MAP = {
    "compliance": "regulasi_periklanan.txt",
    "localization": "gaya_bahasa_referensi.txt",
}

_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)


def _get_embeddings():
    # FastEmbedEmbeddings pakai ONNX runtime, bukan PyTorch - jadi tidak ada
    # konsep device CPU/GPU yang perlu diatur manual seperti sebelumnya.
    return FastEmbedEmbeddings(model_name=EMBEDDING_MODEL)


def _hash_semua_corpus() -> str:
    """Hash gabungan semua file data - dipakai deteksi 'apakah data berubah sejak build terakhir'."""
    import hashlib
    gabungan = ""
    for filename in CORPUS_MAP.values():
        filepath = os.path.join(DATA_DIR, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            gabungan += f.read()
    return hashlib.md5(gabungan.encode("utf-8")).hexdigest()


def build_all_collections():
    """Bangun semua Chroma vector store (via LangChain) dari file di /data."""
    embeddings = _get_embeddings()
    stores = {}

    for agent_name, filename in CORPUS_MAP.items():
        filepath = os.path.join(DATA_DIR, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            raw_text = f.read()

        docs = _splitter.split_documents([Document(page_content=raw_text)])

        persist_path = os.path.join(DB_DIR, agent_name)
        store = Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
            persist_directory=persist_path,
            collection_name=agent_name,
        )
        stores[agent_name] = store

    # Simpan hash data source saat ini, supaya get_collections() tahu kapan harus rebuild
    os.makedirs(DB_DIR, exist_ok=True)
    with open(os.path.join(DB_DIR, "_corpus_hash.txt"), "w") as f:
        f.write(_hash_semua_corpus())

    return stores


def get_collections():
    """
    Ambil vector store yang sudah ada. REBUILD OTOMATIS jika:
    - Folder belum ada sama sekali, ATAU
    - Isi file di data/ berubah sejak build terakhir (dideteksi lewat hash)
    """
    hash_file = os.path.join(DB_DIR, "_corpus_hash.txt")
    hash_sekarang = _hash_semua_corpus()

    if not os.path.exists(hash_file):
        return build_all_collections()

    with open(hash_file, "r") as f:
        hash_tersimpan = f.read().strip()

    if hash_tersimpan != hash_sekarang:
        # Data source berubah sejak build terakhir -> hapus cache lama, build ulang
        import shutil
        if os.path.exists(DB_DIR):
            shutil.rmtree(DB_DIR)
        return build_all_collections()

    embeddings = _get_embeddings()
    stores = {}
    for agent_name in CORPUS_MAP:
        persist_path = os.path.join(DB_DIR, agent_name)
        if not os.path.exists(persist_path):
            return build_all_collections()
        stores[agent_name] = Chroma(
            persist_directory=persist_path,
            embedding_function=embeddings,
            collection_name=agent_name,
        )
    return stores


def retrieve_context(collections: dict, agent_name: str, query: str, n_results: int = 3) -> str:
    """Ambil potongan konteks paling relevan untuk satu agent (lewat retriever LangChain)."""
    if agent_name not in collections:
        return ""
    retriever = collections[agent_name].as_retriever(search_kwargs={"k": n_results})
    docs = retriever.invoke(query)
    return "\n---\n".join(d.page_content for d in docs)


def retrieve_docs_list(collections: dict, agent_name: str, query: str, n_results: int = 3) -> list:
    """Sama seperti retrieve_context, tapi return list per-chunk (untuk evaluasi RAGAS)."""
    if agent_name not in collections:
        return []
    retriever = collections[agent_name].as_retriever(search_kwargs={"k": n_results})
    docs = retriever.invoke(query)
    return [d.page_content for d in docs]


if __name__ == "__main__":
    print("Membangun semua vector store (LangChain + Chroma)...")
    build_all_collections()
    print("Selesai. Tersimpan di:", DB_DIR)
