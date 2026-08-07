# PRD — fine-tuning-ai

Fine-tuning model bahasa pretrained gratis (HuggingFace) jadi chatbot bahasa Indonesia, pakai LoRA supaya muat di GPU 6GB (GTX 1660 Ti). Berbeda dari `llm-iqmal` (build GPT dari scratch) — di sini modal utamanya model yang sudah paham bahasa, tinggal disesuaikan ke gaya/domain lewat fine-tuning.

---

## Milestone 0 — Environment & Project Setup

**Goal:** Project siap dikerjakan, dependency terinstall, struktur folder bersih.

**Deliverables:**
- [ ] `.venv` terbuat dengan `py -3 -m uv venv .venv`
- [ ] `requirements.txt`: `torch` (CUDA build, samakan tag `cuXXX` dengan driver — cek `nvidia-smi`), `transformers`, `peft`, `accelerate`, `datasets`, `sentencepiece`
- [ ] `config.py` dengan hyperparameter fine-tuning default
- [ ] Folder struktur sesuai `CLAUDE.md`

**Done when:** `python -c "import torch, transformers, peft; print(torch.cuda.is_available())"` return `True` tanpa error.

---

## Milestone 1 — Pilih base model pretrained

**Goal:** Punya base model bahasa Indonesia gratis yang muat di-load + LoRA fine-tune di 6GB VRAM.

**Kandidat (dicek ketersediaan & lisensi sebelum dipakai):**
- `cahya/gpt2-small-indonesian-522M` — GPT2-small, ~124M param, pretrained corpus ID
- `indonesian-nlp/gpt2` — alternatif GPT2 ID
- Model multilingual instruct kecil (mis. keluarga Qwen/Gemma ukuran kecil) kalau butuh kualitas instruksi lebih baik — cek dulu ukurannya realistis untuk 6GB + LoRA

**Deliverables:**
- [ ] Script/notebook kecil: load base model + tokenizer, coba generate teks Indonesia biasa (belum fine-tune) sebagai baseline
- [ ] Catat jumlah parameter, ukuran file, & kebutuhan VRAM saat load (fp16)
- [ ] Putuskan base model final, dokumentasikan alasan pemilihan di sini

**Done when:** Base model ke-load di GPU, generate teks Indonesia yang grammatically masuk akal (baseline sebelum fine-tune).

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
- [ ] `scripts/download_dataset.py` — download salah satu/gabungan dataset di atas
- [ ] `scripts/prepare_dataset.py` — format ke template chat base model (system/user/assistant atau instruction/response, sesuai model M1), split train/val
- [ ] Statistik dataset: jumlah pasangan instruksi, rata-rata panjang

**Done when:** `data/processed/train.jsonl` & `val.jsonl` tersedia, tervalidasi format-nya cocok dengan chat template base model.

---

## Milestone 3 — Fine-tuning pipeline (LoRA)

**Goal:** Fine-tuning jalan end-to-end pakai LoRA (`peft`), checkpoint adapter tersimpan.

**Deliverables:**
- [ ] `config.py`: LoRA rank/alpha/dropout, target modules, learning rate, batch size, max steps
- [ ] `training/finetune.py`:
  - Load base model (fp16) + tokenizer
  - Wrap dengan `peft.LoraConfig` + `get_peft_model`
  - Training loop pakai `transformers.Trainer` atau loop manual
  - Checkpoint adapter tersimpan tiap N step ke `adapters/`
- [ ] Validasi VRAM tidak OOM di GTX 1660 Ti (6GB)
- [ ] Test run singkat (beberapa ratus step) — loss turun, tidak error

**Done when:** Fine-tuning selesai jalan sampai `max_steps`, adapter tersimpan di `adapters/best/`.

---

## Milestone 4 — Evaluasi

**Goal:** Bisa ukur apakah fine-tuning membuat model lebih baik dibanding baseline M1.

**Deliverables:**
- [ ] `eval/generate.py` — generate dari base model vs base+adapter, bandingkan side-by-side
- [ ] `eval/evaluate.py` — hitung loss/perplexity di val set
- [ ] Beberapa contoh prompt manual buat sanity check kualitas jawaban

**Done when:** Ada bukti kualitatif/kuantitatif bahwa base+adapter lebih sesuai ke dataset fine-tuning dibanding base model polos.

---

## Milestone 5 — Chatbot Interface

**Goal:** Bisa chat interaktif pakai base model + LoRA adapter.

**Deliverables:**
- [ ] `chatbot/chat.py` — CLI: load base model + adapter, loop input → generate → print
- [ ] Format prompt sesuai chat template base model (bukan format custom kayak `llm-iqmal`)
- [ ] README update dengan cara pakai

**Done when:** `python -m chatbot.chat --base-model <hf-model> --adapter adapters/best` bisa diajak ngobrol dengan gaya sesuai dataset fine-tuning.

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
