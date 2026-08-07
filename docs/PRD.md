# PRD — fine-tuning-ai

Fine-tuning model bahasa pretrained gratis (HuggingFace) jadi chatbot bahasa Indonesia, pakai LoRA supaya muat di GPU 6GB (GTX 1660 Ti). Berbeda dari `llm-iqmal` (build GPT dari scratch) — di sini modal utamanya model yang sudah paham bahasa, tinggal disesuaikan ke gaya/domain lewat fine-tuning.

---

## Milestone 0 — Environment & Project Setup

**Goal:** Project siap dikerjakan, dependency terinstall, struktur folder bersih.

**Deliverables:**
- [x] `.venv` terbuat dengan `py -3 -m uv venv .venv`
- [x] `requirements.txt`: `torch` (CUDA build, samakan tag `cuXXX` dengan driver — cek `nvidia-smi`), `transformers`, `peft`, `accelerate`, `datasets`, `sentencepiece`
- [x] `config.py` dengan hyperparameter fine-tuning default
- [x] Folder struktur sesuai `CLAUDE.md`

**Done when:** `python -c "import torch, transformers, peft; print(torch.cuda.is_available())"` return `True` tanpa error.

---

## Milestone 1 — Pilih base model pretrained

**Goal:** Punya base model bahasa Indonesia gratis yang muat di-load + LoRA fine-tune di 6GB VRAM.

**Kandidat (dicek ketersediaan & lisensi sebelum dipakai):**
- `cahya/gpt2-small-indonesian-522M` — GPT2-small, ~124M param, pretrained corpus ID
- `indonesian-nlp/gpt2` — alternatif GPT2 ID
- Model multilingual instruct kecil (mis. keluarga Qwen/Gemma ukuran kecil) kalau butuh kualitas instruksi lebih baik — cek dulu ukurannya realistis untuk 6GB + LoRA

**Deliverables:**
- [x] Script/notebook kecil: load base model + tokenizer, coba generate teks Indonesia biasa (belum fine-tune) sebagai baseline — `eval/generate.py`
- [x] Catat jumlah parameter, ukuran file, & kebutuhan VRAM saat load (fp16)
- [x] Putuskan base model final, dokumentasikan alasan pemilihan di sini

**Catatan hasil baseline (`cahya/gpt2-small-indonesian-522M`):**
- Params: 124,439,808 (~124M — "522M" di nama repo adalah ukuran korpus training, bukan jumlah parameter)
- Bobot di HF hub: `pytorch_model.bin` ~510MB (fp32); di-load sebagai fp16 → VRAM saat load jauh di bawah 1GB, sangat aman untuk GPU 6GB + ruang LoRA fine-tune nanti
- Load sukses di GPU (`cuda`), generate teks Indonesia yang gramatikal & koheren pada baseline (belum fine-tune)

**Keputusan:** `cahya/gpt2-small-indonesian-522M` dipakai sebagai base model final untuk Milestone 2+. Alasan: arsitektur GPT2-small standar (kompatibel LoRA lewat target module `c_attn`), ukuran kecil jadi ruang VRAM besar tersisa untuk batch/LoRA di 6GB, dan sudah pretrained korpus Indonesia sehingga baseline generate sudah koheren tanpa fine-tune.

**Done when:** Base model ke-load di GPU, generate teks Indonesia yang grammatically masuk akal (baseline sebelum fine-tune). ✅

---

## Milestone 2 — Dataset instruksi/dialog

**Goal:** Punya dataset instruksi/dialog bahasa Indonesia, diformat sesuai template chat base model.

**Sumber kandidat (HuggingFace):**
- `cahya/alpaca-id-cleaned` — instruksi-jawaban umum
- `izzulgod/indonesian-conversation` — dialog multi-turn kurasi manual
- `FreedomIntelligence/sharegpt-indonesian` — percakapan multi-turn
- `MBZUAI/Bactrian-X` (split `id`) — Alpaca+Dolly translate
- Data lokal yang sudah ada di `llm-iqmal/datasets/alodokter-OTC-drugs_scrapping.json` (domain obat, format instruction/input/output) — bisa dipakai sebagai contoh domain-specific fine-tuning

**Deliverables:**
- [x] `scripts/download_dataset.py` — download salah satu/gabungan dataset di atas
- [x] `scripts/prepare_dataset.py` — format ke template chat base model (system/user/assistant atau instruction/response, sesuai model M1), split train/val
- [x] Statistik dataset: jumlah pasangan instruksi, rata-rata panjang

**Dataset dipakai:** `cahya/alpaca-id-cleaned` (51.590 baris, kolom `instruction`/`input`/`output`) — dipilih karena satu "keluarga" dengan base model M1 (`cahya/*`), format Alpaca standar, cukup besar untuk LoRA fine-tune.

**Template:** GPT2 base tidak punya chat template bawaan, jadi dipakai template Alpaca klasik (`### Instruksi:` / `### Input:` opsional / `### Jawaban:`), ditutup token `<|endoftext|>` supaya model belajar kapan berhenti generate.

**Statistik:**
- Total pasangan: 51.590 → train 49.011 / val 2.579 (95/5, seed 42)
- Rata-rata panjang: 167 token; maksimum 1.343 token (beberapa contoh lebih panjang dari `max_seq_length=512` di `config.py` — akan di-truncate saat training)

**Done when:** `data/processed/train.jsonl` & `val.jsonl` tersedia, tervalidasi format-nya cocok dengan chat template base model. ✅

---

## Milestone 3 — Fine-tuning pipeline (LoRA)

**Goal:** Fine-tuning jalan end-to-end pakai LoRA (`peft`), checkpoint adapter tersimpan.

**Deliverables:**
- [x] `config.py`: LoRA rank/alpha/dropout, target modules, learning rate, batch size, max steps
- [x] `training/finetune.py`:
  - Load base model (fp16) + tokenizer
  - Wrap dengan `peft.LoraConfig` + `get_peft_model`
  - Training loop pakai `transformers.Trainer` atau loop manual
  - Checkpoint adapter tersimpan tiap N step ke `adapters/`
- [x] Validasi VRAM tidak OOM di GTX 1660 Ti (6GB)
- [x] Test run singkat (beberapa ratus step) — loss turun, tidak error

**Hasil test run (300 step, r=8/alpha=16, batch efektif 16, ~35 menit):**
- Train loss: 4.65 → 3.41 (turun stabil, tidak error)
- Eval loss di val set: 3.34
- VRAM terpakai: ~5.8GB / 6GB — aman, tidak OOM, tapi mepet (headroom tipis kalau mau naikkan batch size)
- Adapter tersimpan di `adapters/best/` (`adapter_model.safetensors` ~1.2MB — cuma bobot LoRA, bukan base model, sesuai aturan project)
- Baru 0.098 epoch dari target 3 epoch penuh — ini masih validasi pipeline, belum training final (lihat Milestone 4 untuk cek kualitas, lanjut training penuh kalau hasil M4 menjanjikan)

**Done when:** Fine-tuning selesai jalan sampai `max_steps`, adapter tersimpan di `adapters/best/`. ✅

---

## Milestone 4 — Evaluasi

**Goal:** Bisa ukur apakah fine-tuning membuat model lebih baik dibanding baseline M1.

**Deliverables:**
- [x] `eval/generate.py` — generate dari base model vs base+adapter, bandingkan side-by-side (`--adapter <path>` opsional)
- [x] `eval/evaluate.py` — hitung loss/perplexity di val set
- [x] Beberapa contoh prompt manual buat sanity check kualitas jawaban

**Hasil kuantitatif (500 sampel val set):**

| | Val loss | Perplexity |
|---|---|---|
| Base model (baseline) | 4.67 | 106.74 |
| Base + LoRA adapter (300 step) | 3.36 | **28.89** |

**Hasil kualitatif** (prompt: `### Instruksi:\nSebutkan tiga manfaat olahraga bagi kesehatan\n\n### Jawaban:`):
- Base model: mengabaikan format instruksi sepenuhnya, generate teks tidak nyambung (soal stasiun MRT Singapura)
- Base+adapter: ikut format `### Instruksi/Jawaban`, jawaban nyambung ke topik olahraga & kesehatan, meski masih repetitif dan drift ke topik lain setelah ~2 kalimat (wajar — baru 300 step / 0.098 epoch, belum full training)

**Done when:** Ada bukti kualitatif/kuantitatif bahwa base+adapter lebih sesuai ke dataset fine-tuning dibanding base model polos. ✅

---

## Milestone 5 — Chatbot Interface

**Goal:** Bisa chat interaktif pakai base model + LoRA adapter.

**Deliverables:**
- [x] `chatbot/chat.py` — CLI: load base model + adapter, loop input → generate → print
- [x] Format prompt sesuai chat template base model (bukan format custom kayak `llm-iqmal`)
- [x] README update dengan cara pakai

**Catatan implementasi:**
- Base model (GPT2) tidak punya chat template/multi-turn memory bawaan, tiap giliran diproses independen pakai template Alpaca yang sama dengan training (`### Instruksi:` / `### Jawaban:`)
- `generate()` dikasih `repetition_penalty=1.3` + `no_repeat_ngram_size=3` — tanpa ini model (yang baru 300 step training) gampang masuk repetition loop (mis. `"Tulis:"` berulang puluhan kali)
- Output dipotong di stop marker (`### Instruksi`, `### Jawaban`, `<|endoftext|>`, dst.) supaya tidak ikut generate giliran halusinasi berikutnya
- Kualitas jawaban masih sering ngelantur topik setelah 1-2 kalimat — konsisten dengan catatan M3/M4 bahwa adapter saat ini baru validasi pipeline (300 step), belum training penuh

**Done when:** `python -m chatbot.chat --base-model <hf-model> --adapter adapters/best` bisa diajak ngobrol dengan gaya sesuai dataset fine-tuning. ✅

---

## Milestone 6 — (Opsional) Merge & Export

**Goal:** Adapter LoRA di-merge ke base model jadi satu model deploy-ready, kalau dibutuhkan untuk distribusi/inference tanpa `peft`.

**Deliverables:**
- [ ] Script merge adapter → base model
- [ ] Export ke format lebih ringkas (mis. GGUF via `llama.cpp` convert) kalau mau jalan di CPU/edge

**Done when:** Model hasil merge bisa di-load tanpa dependency `peft`, hasil generate konsisten dengan versi adapter.

---

## Catatan

- Beda filosofi dari `llm-iqmal`: di sini tujuan cepat dapat hasil yang "masuk akal", bukan belajar bikin arsitektur dari nol.
- Prioritas LoRA (bukan full fine-tune) karena VRAM terbatas (6GB) dan base model kemungkinan >100M parameter — full fine-tune butuh jauh lebih banyak VRAM untuk optimizer state.
- Kalau base model dari M1 ternyata masih kebesaran untuk 6GB meski pakai LoRA, turunkan ke model lebih kecil atau pakai quantization (4-bit/8-bit lewat `bitsandbytes` — cek dulu kompatibilitas Windows sebelum pilih ini).
