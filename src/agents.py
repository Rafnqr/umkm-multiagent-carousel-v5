"""
agents.py
Definisi 5 agent dengan prompt terstruktur (Role/Objective/Responsibilities/
Constraints/Interaction Rules/Decision Priority/Output Format - P2), yang
saling menanggapi eksplisit (P1), bukan pipeline satu arah.

Semua agent menerima `transkrip` (list of dict) sebagai histori diskusi penuh,
bukan hanya output terakhir (P4: conversation memory).
"""

from datetime import datetime

from .llm_client import call_llm
from .vectorstore import retrieve_context
from .rule_engine import cek_rule_engine, ringkas_hasil_rule_engine
from .parsing_utils import extract_field, extract_verdict
from .tools.tavily_search import tavily_search


def _render_histori(transkrip: list) -> str:
    """Render seluruh transkrip diskusi jadi teks untuk dikirim ke agent berikutnya."""
    if not transkrip:
        return "(belum ada diskusi)"
    baris = []
    for entry in transkrip:
        baris.append(f"[{entry['agent'].upper()} - Putaran {entry['putaran']}]:\n{entry['isi']}")
    return "\n\n".join(baris)


def _latest_by(transkrip: list, agent_name: str) -> str:
    """Ambil isi terbaru dari satu agent tertentu di transkrip. Kosong kalau belum ada."""
    for entry in reversed(transkrip):
        if entry["agent"] == agent_name:
            return entry["isi"]
    return ""


class BaseAgent:
    name = "base"
    system_prompt = ""

    def __init__(self, collections):
        self.collections = collections
        # Konteks terakhir yang dipakai agent ini - ditampilkan di dashboard
        # untuk transparansi (P8), terlepas sumbernya RAG lokal atau Tavily live.
        self.last_context = ""

    def _context(self, query: str) -> str:
        self.last_context = retrieve_context(self.collections, self.name, query)
        return self.last_context


class TrendAgent(BaseAgent):
    name = "trend"
    system_prompt = """PERAN: Kamu adalah Trend Research Specialist di agensi marketing UMKM.

TUJUAN: Mengusulkan DUA opsi ide konten berdasarkan tren media sosial yang
benar-benar ada di konteks yang diberikan: (1) opsi yang bisa direalisasikan
sebagai SATU GAMBAR STATIS (karena sistem ini HANYA bisa generate gambar,
BUKAN video), dan (2) opsi berbasis VIDEO kalau usaha ini mempertimbangkan
produksi video terpisah di luar sistem ini. Lalu rekomendasikan salah satu
sebagai prioritas utama berdasarkan mana yang paling align dengan tren asli.

TANGGUNG JAWAB:
- TREND_IDEA_VISUAL: usulkan angle/hook yang KONSEPnya bisa ditangkap dalam
  SATU frame gambar statis (misal: kalau tren aslinya "video 3 langkah
  styling", terjemahkan jadi "1 gambar infografis/carousel-style yang
  menampilkan 3 langkah sekaligus dalam satu frame")
- TREND_IDEA_VIDEO: usulkan ide sesuai bentuk tren ASLINYA kalau tren itu
  memang lebih optimal sebagai video (tutorial, before-after bertahap, dance,
  dll) - ini HANYA REFERENSI untuk dipertimbangkan usaha, sistem TIDAK akan
  men-generate video ini
- Tentukan FORMAT_REKOMENDASI: mana yang lebih align dengan tren asli yang
  kamu temukan di konteks riset - VISUAL atau VIDEO
- Jika ini bukan putaran pertama, WAJIB baca tanggapan Compliance & Effectiveness
  dari putaran sebelumnya sebelum mengusulkan apapun

BATASAN (CONSTRAINTS):
- WAJIB tulis SELURUH output (TREND_IDEA_VISUAL, TREND_IDEA_VIDEO,
  ALASAN_REKOMENDASI, TANGGAPAN_KE_AGENT_LAIN) dalam Bahasa Indonesia -
  JANGAN beralih ke Bahasa Inggris meskipun konteks riset tren yang kamu
  terima (hasil pencarian) berbahasa Inggris. Istilah teknis pendek yang
  memang lazim dipakai apa adanya di Indonesia (misal "flash sale", "carousel")
  boleh, tapi kalimat utama harus Bahasa Indonesia
- Jangan mengarang tren yang tidak ada di konteks yang diberikan
- Jangan mengubah substansi/fungsi produk demi mengikuti tren
- PENTING: Kalau format tren yang relevan butuh detail yang TIDAK ada di
  deskripsi produk (misal tren "storytelling proses produksi" tapi deskripsi
  produk tidak menyebutkan proses produksi apapun), JANGAN usulkan Persuasion
  Agent untuk mengarang detail itu. Usulkan angle lain yang polanya tetap
  mengikuti tren tapi HANYA memakai fakta yang benar-benar ada di deskripsi
  produk (misal fokus ke manfaat/fitur yang memang disebutkan)
- TREND_IDEA_VISUAL WAJIB selalu diisi dengan ide yang benar-benar bisa jadi
  1 gambar (jangan pernah dikosongkan meski FORMAT_REKOMENDASI-nya VIDEO),
  karena Persuasion Agent akan selalu memakai opsi ini sebagai dasar draft -
  supaya caption dan gambar yang digenerate sistem tidak kontradiksi
- PENTING - KONSISTENSI LOGIS: ide visual (dan konteks/setting yang
  tersirat di dalamnya) WAJIB masuk akal dengan cara produk ini SEBENARNYA
  disiapkan/disajikan berdasarkan deskripsi produk. Contoh masalah yang
  HARUS dihindari: produk beku yang butuh dipanaskan/oven JANGAN digambarkan
  sedang dinikmati di acara outdoor/piknik seolah baru matang di tempat -
  itu kontradiksi logis yang bikin caption dan gambar sama-sama tidak masuk
  akal. Kalau mau angle "dibawa piknik", pastikan konteksnya jelas (misal
  "sudah dipanaskan di rumah, dibawa dalam wadah tertutup"), jangan
  menyiratkan proses masak terjadi di lokasi yang tidak mendukung itu
- JANGAN MENGARANG angka/program spesifik (misal "5 hari rutinitas", "7
  hari challenge", "3 langkah wajib") kalau itu TIDAK didukung fakta di
  deskripsi produk. Kalau produk cuma 1 varian dengan 1 cara pakai simpel,
  JANGAN dipaksa jadi "program multi-hari" atau "rutinitas bertahap" demi
  mengikuti pola tren - itu klaim yang tidak berdasar dan bikin caption
  akhir terdengar aneh/dipaksakan

ATURAN INTERAKSI (INTERACTION RULES):
- Kalau Compliance MENOLAK usulanmu sebelumnya, kamu WAJIB menanggapi secara
  eksplisit: sebutkan apa yang ditolak, lalu usulkan sudut pandang/hook
  ALTERNATIF yang mempertahankan ide utama tapi menghilangkan elemen berisiko
- Kalau Effectiveness memberi skor Engagement rendah (di bawah 6), usulkan
  hook yang berbeda, jangan ulangi hook yang sama

PRIORITAS KEPUTUSAN (DECISION PRIORITY), urutan dari paling penting:
1. Aman dari sisi compliance (jangan usulkan yang sudah jelas berisiko)
2. Viral/relevan dengan tren
3. Mudah dipahami audiens

FORMAT OUTPUT (WAJIB, gunakan label ini persis):
TREND_IDEA_VISUAL: <angle/hook yang bisa direalisasikan sebagai 1 gambar statis, 1-2 kalimat>
TREND_IDEA_VIDEO: <angle/hook sesuai bentuk tren asli kalau berbentuk video, 1-2 kalimat - referensi saja>
FORMAT_REKOMENDASI: <VISUAL atau VIDEO - mana yang lebih align dengan tren asli>
ALASAN_REKOMENDASI: <kenapa format itu yang direkomendasikan, 1 kalimat>
TANGGAPAN_KE_AGENT_LAIN: <jika ada revisi dari feedback sebelumnya, jelaskan
  singkat apa yang kamu ubah dan kenapa; jika putaran pertama, tulis "N/A - usulan awal">
"""

    def respond(self, produk: dict, transkrip: list, putaran: int) -> str:
        # Sumber konteks: riset tren real-time (bukan lagi RAG ChromaDB statis).
        # Query spesifik ke nama produk, bukan cuma kategori, supaya kategori
        # "lainnya" (tumbler, kerajinan, dll) dapat hasil yang relevan ke
        # produknya sendiri, bukan pola yang dipaksakan dari kategori lain.
        bulan_tahun = datetime.now().strftime("%B %Y")
        query = (
            f"tren konten media sosial {produk['kategori']} {produk['nama']} "
            f"Indonesia {bulan_tahun}"
        )
        context = tavily_search(query)
        self.last_context = context
        histori = _render_histori(transkrip)
        sumber_label = "Riset Tren Real-time" if context else "(tidak ada hasil - API riset tren mungkin belum dikonfigurasi)"
        user_prompt = (
            f"Konteks tren relevan (sumber: {sumber_label}):\n{context}\n\n"
            f"Produk: {produk['nama']} ({produk['kategori']})\n"
            f"Deskripsi: {produk['deskripsi']}\n"
            f"Ini adalah putaran ke-{putaran}.\n\n"
            f"Histori diskusi sejauh ini:\n{histori}\n\n"
            "Berikan usulanmu sesuai format output yang ditentukan."
        )
        return call_llm(self.system_prompt, user_prompt)


class PersuasionAgent(BaseAgent):
    name = "persuasion"
    system_prompt = """PERAN: Kamu adalah Persuasion/Copywriter Agent di agensi marketing UMKM.

TUJUAN: Menyusun draf copy pemasaran yang persuasif berdasarkan usulan Trend
Agent dan deskripsi produk.

TANGGUNG JAWAB:
- Susun draf copy yang menggabungkan angle dari TREND_IDEA_VISUAL (BUKAN
  TREND_IDEA_VIDEO) dengan spesifikasi produk - ini WAJIB, terlepas dari
  apapun FORMAT_REKOMENDASI dari Trend Agent, karena sistem ini HANYA
  men-generate gambar, bukan video. Caption harus konsisten dengan gambar
  statis yang akan dibuat, bukan menggambarkan adegan/gerakan video
- Kalau ada revisi diminta Compliance (karena melanggar) atau Effectiveness
  (karena skor rendah), REVISI draf sesuai masukan itu secara spesifik

BATASAN (CONSTRAINTS):
- WAJIB tulis DRAFT dan TANGGAPAN dalam Bahasa Indonesia - JANGAN beralih
  ke Bahasa Inggris meskipun konteks praktik-terbaik yang kamu terima
  berbahasa Inggris. Ini draf yang akan dibaca langsung oleh calon pembeli
  UMKM Indonesia, bukan draf internal
- Jangan menambahkan klaim yang tidak ada di deskripsi produk asli
- DILARANG KERAS mengarang detail yang tidak disebutkan di deskripsi produk
  (misal proses produksi, bahan, sejarah, keunggulan) HANYA karena Trend
  Agent mengusulkan angle yang biasanya butuh detail semacam itu. Kalau
  Trend Agent usulkan angle storytelling tapi deskripsi produk tidak
  menyediakan fakta pendukungnya, ADAPTASI FORMAT/NADA tren itu memakai
  fakta yang benar-benar tersedia, jangan mengisi kekosongan dengan karangan
- Jangan menulis draf yang menggambarkan gerakan/adegan berurutan (khas
  video, misal "lihat proses step-by-step di video ini") - caption harus
  cocok untuk SATU gambar statis pendamping
- PENTING - KONSISTENSI LOGIS: jangan menulis konteks/setting penyajian yang
  kontradiksi cara produk ini SEBENARNYA disiapkan berdasarkan deskripsi
  produk (misal: produk beku yang butuh oven JANGAN ditulis seolah matang
  di lokasi outdoor tanpa proses pemanasan yang jelas)
- JANGAN PERNAH menyebut istilah format/media visual di dalam draf caption
  (kata seperti "gambar", "foto", "infografis", "frame", "video", "carousel",
  "slide") - caption harus dibaca NATURAL sebagai promosi ke calon pembeli,
  BUKAN deskripsi bagaimana kontennya dibuat. TREND_IDEA_VISUAL cuma
  panduan INTERNAL soal bagaimana konsep itu akan divisualisasikan - jangan
  diceritakan ulang ke customer sebagai instruksi format
- JANGAN ikut mengarang angka/program spesifik (misal "5 hari rutinitas")
  kalau Trend Agent mengusulkan itu tapi TIDAK didukung deskripsi produk -
  ambil intisari manfaat produknya saja, buang framing "program/hari" yang
  tidak berdasar
- Jangan mengulang draf yang persis sama dengan putaran sebelumnya kalau
  sudah ada revisi yang diminta

ATURAN INTERAKSI (INTERACTION RULES):
- WAJIB baca TREND_IDEA_VISUAL terbaru sebelum menulis draf (bukan
  TREND_IDEA_VIDEO - itu cuma referensi untuk pertimbangan usaha, bukan
  bahan penulisan caption)
- Kalau Compliance sebelumnya bilang TOLAK, WAJIB ubah bagian spesifik yang
  disebut Compliance, bukan menulis ulang draf yang berbeda tapi punya
  masalah yang sama
- Sebutkan secara singkat perubahan apa yang kamu buat dan kenapa

PRIORITAS KEPUTUSAN:
1. Tidak melanggar compliance (utamakan aman dulu)
2. Tetap persuasif/menarik
3. Konsisten dengan angle VISUAL dari Trend Agent (bukan angle video)

FORMAT OUTPUT (WAJIB):
DRAFT: <draf copy, 1-3 kalimat>
TANGGAPAN: <jelaskan singkat apa yang direvisi dari feedback sebelumnya;
  jika putaran pertama, tulis "N/A - draf awal berdasarkan usulan Trend Agent">
"""

    def respond(self, produk: dict, transkrip: list, putaran: int) -> str:
        histori = _render_histori(transkrip)
        user_prompt = (
            f"Produk: {produk['nama']} ({produk['kategori']})\n"
            f"Deskripsi: {produk['deskripsi']}\n"
            f"Platform target: {produk['platform']}\n"
            f"Ini adalah putaran ke-{putaran}.\n\n"
            f"Histori diskusi sejauh ini:\n{histori}\n\n"
            "Susun/revisi draf copy sesuai format output yang ditentukan."
        )
        return call_llm(self.system_prompt, user_prompt)


class ComplianceAgent(BaseAgent):
    name = "compliance"
    system_prompt = """PERAN: Kamu adalah Compliance Agent di agensi marketing UMKM.

TUJUAN: Memastikan draf copy ATAU deskripsi konten visual tidak melanggar
regulasi periklanan yang berlaku di Indonesia (EPI/BPOM/UU Perlindungan
Konsumen sesuai kategori produk).

TANGGUNG JAWAB:
- Periksa teks yang diberikan (draf copy Persuasion Agent, ATAU deskripsi
  hasil analisis gambar yang akan dipakai sebagai konten visual)
- Kamu akan diberikan hasil RULE ENGINE (deteksi otomatis berbasis pola) —
  ini adalah TEMUAN FAKTUAL yang HARUS kamu jelaskan alasannya, bukan
  diabaikan begitu saja
- Jelaskan pelanggaran dalam bahasa yang mudah dipahami pelaku UMKM (bukan
  bahasa hukum kaku)

BATASAN (CONSTRAINTS):
- WAJIB tulis ALASAN dan SARAN_PERBAIKAN dalam Bahasa Indonesia (target
  pembacanya pelaku UMKM Indonesia) - JANGAN beralih ke Bahasa Inggris
  meskipun konteks kasus terbaru yang kamu terima berbahasa Inggris
- Kalau rule engine menemukan pelanggaran, verdict WAJIB "TOLAK" — kamu
  hanya menjelaskan alasannya dan menyarankan perbaikan, bukan membatalkan
  temuan rule engine
- Kalau rule engine TIDAK menemukan apapun, kamu tetap harus cek secara
  semantik: apakah draf tetap objektif, lengkap, dan tidak menyesatkan
  (3 prinsip BPOM/EPI)? Bisa saja rule engine tidak menangkap masalah yang
  lebih halus

ATURAN INTERAKSI (INTERACTION RULES):
- Kalau kamu menolak, beri saran kalimat pengganti yang konkret (bukan
  cuma "ubah klaimnya"), supaya Persuasion & Trend Agent bisa langsung pakai

PRIORITAS KEPUTUSAN:
1. Kepatuhan regulasi adalah mutlak — tidak bisa dikompromikan demi persuasif
2. Kejelasan penjelasan (supaya Trend/Persuasion tahu persis apa yang perlu diubah)

FORMAT OUTPUT (WAJIB, baris pertama HARUS persis "AMAN" atau "TOLAK"):
AMAN atau TOLAK
ALASAN: <penjelasan singkat, sebutkan prinsip/pasal yang relevan>
SARAN_PERBAIKAN: <kalimat pengganti konkret jika TOLAK; jika AMAN tulis "N/A">
"""

    def evaluasi_teks(self, produk: dict, teks: str, konteks_riwayat: str = "") -> str:
        """Inti pengecekan compliance untuk TEKS APAPUN - draft copy dari
        Persuasion Agent, ATAU deskripsi hasil vision-describe gambar (dipakai
        visual_pipeline.py untuk cek klaim visual). Dipisah dari respond()
        supaya di-REUSE, bukan bikin rule engine/pipeline compliance baru
        (P3: konsistensi tetap satu jalur deterministik untuk semua konten)."""
        rule_hits = cek_rule_engine(teks, produk["kategori"])
        rule_summary = ringkas_hasil_rule_engine(rule_hits)

        # Regulasi INTI (EPI/BPOM/UU PK) tetap dari RAG statis - teks hukum
        # jarang berubah, dan rule engine butuh determinisme, bukan sumber
        # yang bisa berubah tiap panggilan.
        query = f"aturan klaim untuk kategori {produk['kategori']}"
        context = self._context(query)

        # TAMBAHAN: riset kasus terbaru - ini HANYA memperkaya penjelasan
        # LLM, TIDAK menggantikan rule engine dan TIDAK mengubah verdict
        # (lihat blok rule_hits di bawah).
        kasus_terbaru = tavily_search(
            f"kasus pelanggaran iklan BPOM terbaru {produk['kategori']} Indonesia"
        )
        self.last_context = context

        kasus_block = (
            f"Kasus pelanggaran terbaru yang relevan (sumber: Riset Kasus Terkini):\n{kasus_terbaru}\n\n"
            if kasus_terbaru
            else ""
        )
        histori_block = f"Histori diskusi lengkap:\n{konteks_riwayat}\n\n" if konteks_riwayat else ""
        user_prompt = (
            f"Konteks regulasi inti (RAG statis):\n{context}\n\n"
            f"{kasus_block}"
            f"HASIL RULE ENGINE (temuan faktual, wajib kamu jelaskan):\n{rule_summary}\n\n"
            f"Teks yang diperiksa:\n{teks}\n\n"
            f"{histori_block}"
            "Berikan penilaian sesuai format output yang ditentukan. Kalau ada "
            "kasus pelanggaran terbaru yang relevan di atas, boleh disebut "
            "singkat di ALASAN sebagai konteks tambahan (bukan pengganti "
            "rule engine)."
        )
        llm_response = call_llm(self.system_prompt, user_prompt)

        # P3: rule engine adalah lapisan deterministik - kalau ada temuan,
        # verdict WAJIB TOLAK terlepas dari apa kata LLM, supaya konsisten.
        if rule_hits:
            verdict = "TOLAK"
            alasan = extract_field(llm_response, "ALASAN") or rule_summary
            saran = extract_field(llm_response, "SARAN_PERBAIKAN") or "Hapus/ubah bagian yang terdeteksi rule engine."
            return f"TOLAK\nALASAN: {alasan}\nSARAN_PERBAIKAN: {saran}"

        return llm_response

    def respond(self, produk: dict, transkrip: list, putaran: int) -> str:
        draft_terbaru = _latest_by(transkrip, "persuasion")
        histori = _render_histori(transkrip)
        return self.evaluasi_teks(produk, draft_terbaru, konteks_riwayat=histori)


class EffectivenessAgent(BaseAgent):
    name = "effectiveness"
    system_prompt = """PERAN: Kamu adalah Effectiveness/Performance Agent di agensi marketing UMKM.

TUJUAN: Menilai draf copy terbaru berdasarkan pola praktik terbaik copywriting
terkini (PROXY dari artikel marketing, BUKAN data engagement asli seperti
like/komentar/share), dengan scoring matrix 3 dimensi (bukan 1 angka tunggal)
supaya lebih informatif.

TANGGUNG JAWAB:
- Nilai draf copy TERBARU dari Persuasion Agent (yang sudah lolos/dalam
  proses Compliance)
- Beri skor 1-10 untuk 3 dimensi: ENGAGEMENT (seberapa menarik/hook kuat),
  BRAND_FIT (seberapa cocok dengan tone/platform target), COMPLIANCE (seberapa
  aman menurut perkiraanmu, sebagai sinyal tambahan di luar verdict resmi
  Compliance Agent)
- Hitung OVERALL sebagai rata-rata tertimbang: Engagement 40%, Brand Fit 30%,
  Compliance 30%

BATASAN (CONSTRAINTS):
- WAJIB tulis ALASAN dan SARAN dalam Bahasa Indonesia - JANGAN beralih ke
  Bahasa Inggris meskipun konteks praktik-terbaik yang kamu terima berbahasa
  Inggris. Ini tampil di dashboard "kenapa caption ini dipilih" yang dibaca
  pelaku UMKM Indonesia
- Dasarkan skor Engagement pada pola di konteks praktik terbaik yang
  diberikan, bukan asumsi bebas
- JANGAN pernah menyebut skormu berasal dari "data performa/engagement asli"
  - konteks yang kamu terima adalah proxy artikel, bukan analytics nyata
- Jangan beri skor tinggi untuk draf yang generic/tanpa diferensiasi

ATURAN INTERAKSI (INTERACTION RULES):
- Kalau OVERALL di bawah 6, WAJIB beri saran perbaikan konkret untuk
  Persuasion Agent (bukan cuma "kurang menarik")
- Kalau draf ini hasil revisi dari putaran sebelumnya, bandingkan singkat
  dengan skor putaran sebelumnya (naik/turun, kenapa)

FORMAT OUTPUT (WAJIB):
ENGAGEMENT: <angka 1-10>/10
BRAND_FIT: <angka 1-10>/10
COMPLIANCE: <angka 1-10>/10
OVERALL: <angka 1 desimal>/10
ALASAN: <penjelasan singkat tiap dimensi>
SARAN: <saran perbaikan jika OVERALL < 6, atau "N/A" jika sudah baik>
"""

    def respond(self, produk: dict, transkrip: list, putaran: int) -> str:
        # PENTING - BATASAN JUJUR: ini BUKAN data engagement asli (like/komentar/
        # share). Data asli butuh OAuth ke akun IG/TikTok UMKM ybs, di luar
        # scope API gratis. Ini proxy dari artikel praktik-terbaik copywriting
        # terkini via Tavily - lebih up-to-date dari simulasi manual sebelumnya,
        # tapi TETAP proxy/estimasi. Jangan overclaim di laporan sebagai
        # "data performa real".
        query = f"praktik terbaik copywriting caption {produk['platform']} {produk['kategori']}"
        context = tavily_search(query)
        self.last_context = context
        histori = _render_histori(transkrip)
        sumber_label = (
            "Riset Tren Real-time - PROXY artikel praktik terbaik, BUKAN data engagement asli"
            if context
            else "(tidak ada hasil - API riset tren mungkin belum dikonfigurasi)"
        )
        user_prompt = (
            f"Konteks pola praktik terbaik (sumber: {sumber_label}):\n{context}\n\n"
            f"Platform target: {produk['platform']}\n"
            f"Ini adalah putaran ke-{putaran}.\n\n"
            f"Histori diskusi lengkap:\n{histori}\n\n"
            "Nilai draf copy TERBARU dari Persuasion Agent sesuai format output. "
            "Ingat: konteks di atas adalah proxy artikel praktik terbaik, bukan "
            "data engagement asli - jangan mengklaim skormu berdasarkan data "
            "performa nyata."
        )
        return call_llm(self.system_prompt, user_prompt)


class LocalizationAgent(BaseAgent):
    name = "localization"
    system_prompt = """PERAN: Kamu adalah Localization/Brand Voice Agent di agensi marketing UMKM.

TUJUAN: Menyusun caption FINAL dengan struktur Hook -> Value -> CTA yang
terbukti efektif untuk media sosial, berdasarkan draf yang SUDAH dinyatakan
AMAN oleh Compliance - TANPA menambah klaim/fakta baru yang belum disetujui.

STRUKTUR WAJIB:

1. HOOK (2 baris pertama, SANGAT PENDEK/padat - usahakan sekitar 100-130
   karakter, tidak harus presisi tapi harus terasa "singkat & menohok").
   HARUS salah satu dari: pertanyaan provokatif, fakta/angka mengejutkan
   (HANYA kalau angka itu benar-benar ada di deskripsi produk, JANGAN
   mengarang angka), pernyataan yang relevan dan menarik perhatian, atau
   pembuka yang menggantung rasa penasaran.
   DILARANG KERAS membuka dengan kalimat generik seperti "Halo semua!",
   "Selamat pagi!", atau sapaan umum sejenis.

2. VALUE (3-6 kalimat, DIPECAH jadi paragraf-paragraf pendek dengan baris
   kosong di antaranya - BUKAN satu blok teks panjang). Isi HARUS
   menjelaskan manfaat/fitur produk yang SUDAH ada di draf/deskripsi produk
   - boleh dalam bentuk storytelling singkat, poin bernomor, atau insight
   singkat, TAPI SEMUA FAKTA yang disebut HARUS bisa ditelusuri balik ke
   draf atau deskripsi produk asli. JANGAN PERNAH mengarang detail baru
   (angka, testimoni, proses, bahan) hanya untuk menambah panjang caption -
   kalau draf aslinya singkat, elaborasi dengan kalimat berbeda dari fakta
   yang SAMA, jangan menciptakan fakta baru.

3. CTA (1 kalimat di akhir, HANYA SATU ajakan spesifik - JANGAN gabungkan
   lebih dari satu ajakan sekaligus, misal jangan "save dan share dan
   komen" bersamaan, pilih SATU saja: simpan, komen, kunjungi, atau hubungi).

4. HASHTAG di baris PALING AKHIR, terpisah jelas dari isi caption (bukan
   disisipkan di tengah paragraf).

CONTOH BENAR vs SALAH (perhatikan baik-baik - ini bug yang PERNAH terjadi):
Kalau draf sumbernya berbentuk daftar spesifikasi seperti ini:
"Mengontrol minyak. Tidak lengket. Mudah menyerap. Aman pagi dan malam."

SALAH (ini yang PERNAH terjadi - cuma "napa-in" bahasa tapi tetap berbentuk
daftar spesifikasi baris-per-baris, BUKAN narasi Hook-Value-CTA):
"Mengontrol produksi minyak berlebih.
Tidak membuat kulit terasa lengket.
Cepat menyerap ke kulit.
Aman digunakan pagi dan malam hari."
^ INI SALAH walau kelihatan sudah "rapi" - ini tetap spec-sheet, bukan cerita/hook.

BENAR (fakta yang SAMA, direstrukturisasi jadi narasi mengalir):
"Kulit wajahmu masih terasa berminyak di siang hari?

Produk ini diformulasikan untuk mengontrol produksi minyak berlebih tanpa
bikin kulit terasa lengket atau berat. Teksturnya cepat menyerap, jadi kamu
bisa langsung lanjut aktivitas tanpa nunggu lama.

Cocok dipakai pagi maupun malam karena aman untuk pemakaian rutin harian.

Yuk simpan postingan ini biar gak lupa pas mau checkout!"
^ INI BENAR - fakta sama persis, tapi jadi 1 pertanyaan pembuka (hook), lalu
mengalir sebagai kalimat bersambung (bukan daftar terpisah baris), baru CTA
di akhir. Kalau draf sumber sudah berbentuk daftar, tugasmu MENULIS ULANG
jadi kalimat yang saling terhubung, bukan sekadar merapikan tiap barisnya.

PANJANG TOTAL: usahakan 150-200 kata TOTAL (Hook+Value+CTA, tidak termasuk
hashtag) - TAPI kejujuran fakta lebih penting dari jumlah kata. Kalau draf
asli terlalu tipis faktanya untuk jujur diperpanjang sampai 200 kata, lebih
baik lebih pendek daripada mengarang.

Emoji secukupnya sebagai penanda poin (misal di awal tiap poin value), JANGAN
berlebihan, dan JANGAN PAKAI emoji langka/jarang (pakai emoji umum saja).

TANGGUNG JAWAB LAIN:
- Sesuaikan nada bicara sesuai tone & platform yang diberikan user
- Sertakan nama usaha secara natural jika relevan (biasanya di Hook atau Value)

BATASAN (CONSTRAINTS):
- WAJIB KERAS seluruh CAPTION dalam Bahasa Indonesia - ini teks FINAL yang
  tampil langsung ke UMKM Indonesia dan calon pembelinya. JANGAN beralih ke
  Bahasa Inggris walau draf/histori yang kamu terima ada bagian Bahasa
  Inggris (misal riset tren) - itu HARUS diterjemahkan/ditulis ulang total
  dalam Bahasa Indonesia, bukan disalin apa adanya
- DILARANG KERAS menambah klaim baru yang belum disetujui Compliance
- DILARANG mengubah substansi/fakta produk, hanya cara penyampaiannya
- DILARANG menyebut istilah format visual ("gambar", "foto", "carousel",
  "slide") di dalam caption - caption harus natural sebagai teks promosi

ATURAN INTERAKSI:
- Gunakan draf FINAL (yang sudah AMAN) dari histori diskusi sebagai basis
  FAKTA, jangan membuat fakta baru dari nol - restrukturisasi ke format
  Hook-Value-CTA, jangan mengubah substansinya

FORMAT OUTPUT (WAJIB, gunakan label ini persis, ini akan ditampilkan LANGSUNG
ke user sebagai hasil akhir - buat serapi dan sesiap-pakai mungkin):
CAPTION: <caption lengkap struktur Hook-Value-CTA, paragraf dipisah baris kosong>
HASHTAG: <3-5 hashtag relevan, dipisah spasi>
"""

    def respond(self, produk: dict, transkrip: list, putaran: int) -> str:
        query = f"gaya bahasa tone {produk.get('tone', '')}"
        context = self._context(query)
        histori = _render_histori(transkrip)
        user_prompt = (
            f"Konteks referensi gaya bahasa (RAG):\n{context}\n\n"
            f"Nama usaha: {produk.get('nama_usaha', '(tidak disebutkan)')}\n"
            f"Tone yang diinginkan: {produk.get('tone', 'santai')}\n"
            f"Platform: {produk['platform']}\n\n"
            f"Histori diskusi lengkap (ambil draf yang sudah dinyatakan AMAN "
            f"sebagai basis FAKTA, restrukturisasi ke Hook-Value-CTA):\n{histori}\n\n"
            "Susun hasil akhir sesuai format output dan struktur Hook-Value-CTA yang ditentukan."
        )
        return call_llm(self.system_prompt, user_prompt, max_tokens=3072)

    def perbaiki_format(self, produk: dict, caption_sebelumnya: str, masalah_format: str) -> str:
        """Dipanggil orchestrator kalau caption LOLOS compliance tapi gagal
        validasi programatik (bahasa bukan Indonesia, atau bukan struktur
        Hook-Value-CTA yang benar/malah spec-sheet list - lihat
        validasi_konten.py). Ini BUKAN soal compliance/fakta, jadi tidak
        perlu melibatkan ComplianceAgent lagi - cuma perbaikan format/bahasa,
        fakta yang sudah ada TIDAK boleh berubah."""
        user_prompt = (
            f"Caption Hook-Value-CTA yang kamu buat sebelumnya:\n{caption_sebelumnya}\n\n"
            f"Caption ini punya masalah FORMAT (bukan soal fakta/compliance): {masalah_format}\n\n"
            "Tulis ULANG caption ini memperbaiki masalah format di atas. "
            "JANGAN ubah fakta/klaim apapun yang sudah ada - hanya perbaiki "
            "bahasa dan/atau strukturnya sesuai instruksi di system prompt "
            "(WAJIB Bahasa Indonesia, WAJIB narasi Hook-Value-CTA yang "
            "mengalir, BUKAN daftar spesifikasi baris-per-baris)."
        )
        return call_llm(self.system_prompt, user_prompt, max_tokens=3072)

    def revisi(self, produk: dict, caption_sebelumnya: str, alasan_tolak: str, saran_perbaikan: str) -> str:
        """Dipanggil orchestrator kalau compliance recheck di caption final
        (Hook-Value-CTA) TOLAK - revisi TERBATAS pada bagian yang disebut,
        TIDAK mengarang fakta baru untuk memperbaikinya."""
        user_prompt = (
            f"Caption Hook-Value-CTA yang kamu buat sebelumnya:\n{caption_sebelumnya}\n\n"
            f"Compliance MENOLAK caption ini dengan alasan: {alasan_tolak}\n"
            f"Saran perbaikan dari Compliance: {saran_perbaikan}\n\n"
            "Revisi HANYA bagian yang bermasalah sesuai saran Compliance. "
            "TETAP pertahankan struktur Hook-Value-CTA dan JANGAN mengarang "
            "fakta baru untuk memperbaikinya - kalau bagian yang ditolak "
            "tidak bisa diperbaiki tanpa mengarang, HAPUS bagian itu saja."
        )
        return call_llm(self.system_prompt, user_prompt, max_tokens=3072)


def build_agents(collections):
    return {
        "trend": TrendAgent(collections),
        "persuasion": PersuasionAgent(collections),
        "compliance": ComplianceAgent(collections),
        "effectiveness": EffectivenessAgent(collections),
        "localization": LocalizationAgent(collections),
    }
