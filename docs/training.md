Karena saya di jawaban sampingan ini tidak punya akses tools (tidak bisa langsung menulis file), berikut isi yang bisa kamu simpan sendiri sebagai `docs/TRAINING.md` — atau minta agent utama menuliskannya begitu selesai dengan test run yang sedang jalan:

```markdown
# Cara Training Sendiri

## 1. Aktifkan environment

py -3 -m uv venv .venv # kalau .venv belum ada
python -m uv pip install -r requirements.txt --index-strategy unsafe-best-match

## 2. Siapkan dataset (kalau belum ada data/processed/train.jsonl & val.jsonl)

PYTHONPATH=. .venv/Scripts/python.exe scripts/download_dataset.py
PYTHONPATH=. .venv/Scripts/python.exe scripts/prepare_dataset.py

## 3. Jalankan fine-tuning LoRA

PYTHONPATH=. .venv/Scripts/python.exe -m training.finetune

# Override parameter tanpa edit config.py, contoh:

PYTHONPATH=. .venv/Scripts/python.exe -m training.finetune --max-steps 1000 --output-dir adapters/eksperimen1

## 4. Hyperparameter default ada di config.py

- LoraConfig: r=8, alpha=16, dropout=0.05, target_modules=["c_attn"]
- TrainingConfig: batch_size=2, grad_accum=8, lr=2e-4, max_steps=300 (default test run singkat)
- Ubah nilai di config.py kalau mau training penuh (misal max_steps=-1 + num_train_epochs=3)

## 5. Hasil

Adapter LoRA (bukan full model) tersimpan di --output-dir (default adapters/best/).
Base model TIDAK ikut ter-download ulang tiap training — dicache HuggingFace di ~/.cache/huggingface.

## Catatan VRAM (GTX 1660 Ti, 6GB)

- fp16 wajib aktif (sudah default di config.py)
- Kalau OOM: turunkan per_device_train_batch_size atau naikkan gradient_accumulation_steps
```

Catatan: agent utama masih menjalankan test run 300 step di background — kalau mau, minta dia langsung menuliskan file ini ke `docs/TRAINING.md` setelah training selesai supaya isinya bisa disesuaikan dengan hasil test run yang sebenarnya (misalnya berapa lama waktu per step, VRAM terpakai aktual).
