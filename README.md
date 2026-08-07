# fine-tuning-ai

Fine-tuning model bahasa pretrained (gratis, dari HuggingFace) jadi chatbot bahasa Indonesia, pakai LoRA supaya muat di GPU 6GB (GTX 1660 Ti). Lihat `docs/PRD.md` untuk detail milestone, dan `docs/PRD-RAG.md` untuk rencana project RAG lanjutan (tanpa training).

## Setup

```bash
# Init venv
py -3 -m uv venv .venv

# Install dependency ke .venv (pakai py launcher, bukan python venv langsung,
# supaya uv terdeteksi — lihat catatan di bawah)
py -3 -m uv pip install -r requirements.txt --index-strategy unsafe-best-match --python .venv/Scripts/python.exe
```

> **Catatan Windows/Git Bash:** `uv` kadang tidak mendeteksi `.venv` yang aktif lewat `source .venv/Scripts/activate` di Git Bash (env var `VIRTUAL_ENV` berformat POSIX path, tidak dikenali `uv.exe`). Paling aman selalu tambahkan `--python .venv/Scripts/python.exe` secara eksplisit.

## Alur kerja

```bash
# 1. Download & format dataset instruksi
PYTHONPATH=. .venv/Scripts/python.exe scripts/download_dataset.py
PYTHONPATH=. .venv/Scripts/python.exe scripts/prepare_dataset.py

# 2. Fine-tuning LoRA (default: test run 300 step, lihat config.py utk ubah ke full training)
PYTHONPATH=. .venv/Scripts/python.exe -m training.finetune

# Override parameter tanpa edit config.py:
PYTHONPATH=. .venv/Scripts/python.exe -m training.finetune --max-steps 1000 --output-dir adapters/eksperimen1

# 3. Evaluasi: bandingkan base model vs base+adapter
PYTHONPATH=. .venv/Scripts/python.exe eval/generate.py --adapter adapters/best --prompt "..."
PYTHONPATH=. .venv/Scripts/python.exe eval/evaluate.py --adapter adapters/best --limit 500

# 4. Chat interaktif
PYTHONPATH=. .venv/Scripts/python.exe -m chatbot.chat --adapter adapters/best
```

Semua perintah di atas dijalankan dari root project. `PYTHONPATH=.` diperlukan supaya `config.py` di root ke-import dari dalam `scripts/`, `training/`, `eval/`, `chatbot/`.

## Hasil fine-tuning saat ini

Base model: `cahya/gpt2-small-indonesian-522M` (~124M parameter). Adapter test run (300 step / 0.098 epoch) di `adapters/best/`:

| | Val loss | Perplexity |
|---|---|---|
| Base model | 4.67 | 106.74 |
| Base + adapter | 3.36 | 28.89 |

Adapter ini baru validasi pipeline, belum training penuh — hasil chat masih sering ngelantur topik setelah beberapa kalimat. Untuk training penuh, ubah `max_steps=-1` di `config.py` (jalan sampai `num_train_epochs` selesai).

## Struktur

Lihat `CLAUDE.md` untuk struktur folder & aturan pengembangan.

## Hardware

GPU: NVIDIA GTX 1660 Ti, 6GB VRAM. LoRA + fp16 wajib supaya muat. Test run 300 step memakai ~5.8GB/6GB VRAM — mepet, turunkan `per_device_train_batch_size` di `config.py` kalau OOM.
