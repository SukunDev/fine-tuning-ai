# PRD — RAG (Retrieval-Augmented Generation)

Project lanjutan, terpisah dari track fine-tuning di `PRD.md`. Tujuannya: chatbot yang bisa jawab pertanyaan berdasarkan dokumen spesifik (belum ditentukan domainnya), tanpa training/fine-tuning model apa pun.

**Kenapa tidak perlu training:** RAG bekerja dengan menyuntikkan potongan dokumen relevan (hasil pencarian similarity) ke dalam prompt sebelum model generate jawaban. Model generator dan model embedding dipakai apa adanya (pretrained), tinggal disambungkan lewat kode retrieval + prompt template. Fine-tuning baru relevan kalau nanti butuh model yang "lebih pintar ikut instruksi konteks" secara konsisten — bukan prasyarat untuk RAG jalan.

**Beda dengan `fine-tuning-ai` (track utama):** di sana model diubah bobotnya (LoRA) supaya menyerap gaya/domain tertentu secara permanen. Di sini model generator tetap pretrained murni; pengetahuan domain datang dari dokumen yang di-retrieve saat runtime, bukan dari bobot model.

---

## Milestone 0 — Environment & Setup

**Goal:** Dependency RAG terinstall, struktur folder siap.

**Deliverables:**
- [ ] `requirements-rag.txt` (terpisah dari `requirements.txt` fine-tuning): `sentence-transformers`, `faiss-cpu`, `transformers`, `torch`, `accelerate`
- [ ] Struktur folder:
  ```
  rag/
  ├── documents/       # dokumen sumber mentah (pdf/txt/md)
  ├── index/           # vector index (FAISS) hasil embedding, gitignored
  ├── ingest/          # chunking + build index
  ├── retrieve/         # query -> top-k chunk relevan
  ├── generate/          # prompt template + call generator model
  └── chat/               # CLI chat interface
  ```
- [ ] Folder `rag/index/` & `rag/documents/*` masuk `.gitignore` (dokumen & index bisa besar/berubah-ubah)

**Done when:** `python -c "import sentence_transformers, faiss, transformers; print('ok')"` jalan tanpa error.

---

## Milestone 1 — Pilih model embedding & model generator

**Goal:** Punya pasangan model embedding (retrieval) + model generator instruct (generation) yang gratis, muat di 6GB VRAM.

**Kandidat model embedding (multilingual, paham Indonesia):**
- `intfloat/multilingual-e5-small` (~118M) — kualitas bagus, ringan
- `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` — alternatif populer, cepat

**Kandidat model generator (instruct, bukan base LM):**
- `Qwen/Qwen2.5-1.5B-Instruct` — multilingual, ikut instruksi cukup baik, muat nyaman di 6GB (inference-only, fp16)
- `Qwen/Qwen2.5-0.5B-Instruct` — lebih ringan kalau 1.5B masih berat/lambat
- **Catatan:** base model fine-tuning (`cahya/gpt2-small-indonesian-522M`) sengaja **tidak** dipakai di sini — dia base LM (bukan instruct-tuned) dan context window-nya cuma 1024 token, kurang cocok untuk "baca konteks lalu jawab berdasarkan itu" (lihat diskusi groundedness sebelumnya).

**Deliverables:**
- [ ] Script kecil: load model embedding, coba encode beberapa kalimat, cek dimensi vektor
- [ ] Script kecil: load model generator, coba generate jawaban dari prompt manual (belum ada retrieval)
- [ ] Putuskan model final untuk masing-masing, dokumentasikan alasan di sini

**Done when:** Kedua model ke-load di GPU/CPU tanpa OOM, masing-masing menghasilkan output yang masuk akal secara terpisah.

---

## Milestone 2 — Ingest dokumen & bangun index

**Goal:** Dokumen sumber (format & domain ditentukan saat implementasi) berhasil di-chunk dan diubah jadi vector index yang bisa di-query.

**Deliverables:**
- [ ] `rag/ingest/chunk.py` — pecah dokumen jadi potongan (chunk) dengan overlap, ukuran chunk disesuaikan context window generator
- [ ] `rag/ingest/build_index.py` — encode semua chunk pakai model embedding (M1), simpan ke FAISS index + mapping chunk-id → teks asli
- [ ] Statistik: jumlah dokumen, jumlah chunk, rata-rata panjang chunk

**Done when:** `rag/index/` berisi index FAISS + metadata yang bisa di-load ulang tanpa re-embed dari nol.

---

## Milestone 3 — Retrieval pipeline

**Goal:** Dari pertanyaan user, dapat top-k chunk paling relevan.

**Deliverables:**
- [ ] `rag/retrieve/search.py` — encode query, similarity search ke FAISS index, return top-k chunk + score
- [ ] Threshold/filter skor minimum (biar tidak paksa retrieve chunk yang sebenarnya tidak relevan kalau index kosong/tidak nyambung)
- [ ] Beberapa contoh query manual buat sanity check hasil retrieval

**Done when:** Query manual menghasilkan chunk yang relevan secara isi (bukan cuma similarity tinggi tapi topiknya beda).

---

## Milestone 4 — Generation dengan konteks

**Goal:** Jawaban model benar-benar berdasarkan chunk yang di-retrieve, bukan ngarang dari pengetahuan internal.

**Deliverables:**
- [ ] `rag/generate/prompt.py` — template prompt: system instruction ("jawab hanya berdasarkan konteks berikut, kalau tidak ada jawab tidak tahu") + konteks (chunk hasil retrieval) + pertanyaan user
- [ ] `rag/generate/answer.py` — gabungkan retrieval (M3) + generation, hasilkan jawaban akhir
- [ ] Uji kasus: pertanyaan yang jawabannya ADA di dokumen vs pertanyaan yang jawabannya TIDAK ADA di dokumen (model harus jujur bilang tidak tahu, bukan ngarang)

**Done when:** Untuk pertanyaan di luar cakupan dokumen, model tidak berhalusinasi jawaban — mengaku tidak tahu atau minta klarifikasi.

---

## Milestone 5 — Evaluasi

**Goal:** Ada cara mengukur kualitas retrieval & groundedness jawaban, bukan cuma "kelihatan oke".

**Deliverables:**
- [ ] Set pertanyaan uji manual (~15-20) dengan jawaban referensi dari dokumen
- [ ] `rag/eval/evaluate.py` — cek apakah chunk yang di-retrieve memang mengandung jawaban referensi (retrieval accuracy), dan apakah jawaban akhir konsisten dengan chunk tsb (manual/eyeball check awal, bisa otomatis pakai LLM-judge belakangan)

**Done when:** Ada laporan sederhana: berapa % pertanyaan uji yang retrieval-nya tepat dan jawabannya grounded.

---

## Milestone 6 — Chat interface

**Goal:** Bisa tanya-jawab interaktif dengan dokumen lewat CLI.

**Deliverables:**
- [ ] `rag/chat/chat.py` — loop input pertanyaan → retrieve → generate → print jawaban (+ opsional tampilkan sumber chunk yang dipakai)
- [ ] README update: cara menambah dokumen baru & rebuild index

**Done when:** `python -m rag.chat.chat` bisa diajak tanya-jawab soal isi dokumen yang sudah di-ingest, dengan jawaban yang bisa ditelusuri sumbernya.

---

## Catatan

- Domain dokumen belum ditentukan — pilih saat mulai implementasi (bisa reuse `llm-iqmal/datasets/alodokter-OTC-drugs_scrapping.json` sebagai contoh cepat, atau dokumen lain).
- Tidak ada milestone fine-tuning di sini secara sengaja. Kalau nanti setelah M6 ternyata generator masih sering ngarang meski sudah dikasih konteks, opsi lanjutannya: prompt engineering lebih ketat dulu (paling murah), baru pertimbangkan fine-tuning generator khusus untuk "taat konteks" (jauh lebih mahal, butuh dataset RAG-style instruksi+konteks+jawaban).
- FAISS dipilih (bukan vector DB server seperti Qdrant/Milvus) supaya tetap jalan lokal tanpa service tambahan, konsisten dengan filosofi project ini (gratis, resource terbatas, minim dependency eksternal).
