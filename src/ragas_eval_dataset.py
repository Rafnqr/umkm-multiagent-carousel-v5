"""
ragas_eval_dataset.py
Dataset kecil berlabel manual untuk mengukur Context Precision & Context Recall
dari retriever Compliance Agent. Disusun manual dari data/regulasi_periklanan.txt
sebagai "ground truth" — bukan dari sistem (supaya evaluasi tetap independen/objektif).

Ini dipakai untuk "Evaluasi Sistem" (terpisah dari "Evaluasi Marketing" per sesi
diskusi produk), sesuai rekomendasi mentor.
"""

EVAL_SET = [
    {
        "query": "aturan klaim untuk kategori skincare",
        "ground_truth_keywords": ["objektif", "lengkap", "tidak menyesatkan", "klaim di luar fungsi produk"],
    },
    {
        "query": "aturan klaim waktu instan pada kosmetik",
        "ground_truth_keywords": ["glowing 3 hari", "klaim di luar fungsi produk", "uji klinis"],
    },
    {
        "query": "aturan kata halal dalam iklan makanan",
        "ground_truth_keywords": ["halal", "MUI", "sertifikat resmi"],
    },
    {
        "query": "aturan foto dan sudut kamera iklan fashion",
        "ground_truth_keywords": ["sudut pengambilan gambar", "mengeksploitasi", "tubuh"],
    },
    {
        "query": "aturan klaim superlatif seperti 100 persen atau paling bagus",
        "ground_truth_keywords": ["100%", "murni", "asli", "otentik", "dibuktikan"],
    },
    {
        "query": "dasar hukum umum informasi menyesatkan konsumen",
        "ground_truth_keywords": ["UU Perlindungan Konsumen", "menyesatkan", "sanksi administratif"],
    },
    {
        "query": "aturan membandingkan dengan produk kompetitor",
        "ground_truth_keywords": ["merendahkan", "pesaing", "langsung maupun tidak langsung"],
    },
    {
        "query": "penggunaan atribut profesi kesehatan dalam iklan kosmetik",
        "ground_truth_keywords": ["atribut profesi kesehatan", "dokter", "endorse"],
    },
]
