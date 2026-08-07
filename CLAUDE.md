# fine-tuning-ai

Fine-tuning model bahasa pretrained (gratis, dari HuggingFace) untuk chatbot bahasa Indonesia — beda pendekatan dari [`llm-iqmal`](../llm-iqmal) yang build GPT dari scratch. Di sini fokusnya memanfaatkan model yang sudah paham bahasa Indonesia, lalu disesuaikan (fine-tune) ke gaya/domain tertentu dengan resource terbatas (GPU 6GB).

## Stack

- Python 3.14 (system)
- Package manager: `python -m uv` (bukan `uv` langsung, bukan `pip`)
- Framework: PyTorch + HuggingFace `transformers`, `peft` (LoRA), `accelerate`, `datasets`
- Virtualenv: `.venv/` di root project

## Setup commands

```bash
# Init venv
py -3 -m uv venv .venv

# Aktifkan (Git Bash/zsh di Windows)
source .venv/Scripts/activate

# Install deps
python -m uv pip install -r requirements.txt --index-strategy unsafe-best-match
```

## Struktur project

```
fine-tuning-ai/
├── data/
│   ├── raw/          # Dataset instruksi/dialog mentah
│   └── processed/    # Setelah dibentuk jadi format chat/instruksi
├── scripts/          # Download & preprocessing dataset
├── training/          # Fine-tuning loop (LoRA via peft)
├── adapters/           # LoRA adapter weights hasil training (bukan full model)
├── eval/                # Evaluasi & inferensi
├── chatbot/             # Chat interface CLI
├── docs/
│   └── PRD.md            # Product Requirements per milestone
├── requirements.txt
├── CLAUDE.md
└── README.md
```

## Aturan pengembangan

- Satu file = satu tanggung jawab, sama seperti `llm-iqmal`.
- Semua hyperparameter masuk ke `config.py` atau argparse — tidak boleh hardcode di dalam fungsi.
- Pakai LoRA (`peft`), bukan full fine-tuning — supaya muat di GPU 6GB dan checkpoint kecil (cuma adapter, bukan seluruh bobot model).
- Adapter disimpan terpisah dari base model (jangan merge kecuali untuk deployment akhir).
- Komentar dalam bahasa Indonesia, kode dalam bahasa Inggris.

## Hardware constraints

- GPU: NVIDIA GTX 1660 Ti, 6GB VRAM (Turing, tanpa Tensor Core — FP16 tetap didukung native di CUDA core)
- Training: LoRA + mixed precision (fp16) wajib supaya muat di 6GB, terutama untuk base model >100M parameter
- Batch size kecil + gradient accumulation kalau perlu

## Cara run

Lihat `docs/PRD.md` untuk urutan milestone. Ringkas:

```bash
# Siapkan dataset instruksi/dialog
python scripts/prepare_dataset.py --input data/raw/ --output data/processed/

# Fine-tuning (LoRA)
python -m training.finetune --config config.py

# Chat pakai adapter hasil fine-tuning
python -m chatbot.chat --base-model <nama-model-hf> --adapter adapters/best
```
