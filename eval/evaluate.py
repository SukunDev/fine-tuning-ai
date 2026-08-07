"""Hitung loss & perplexity di val set — base model vs base+adapter.

Dipakai buat bukti kuantitatif apakah fine-tuning LoRA memperbaiki model
dibanding baseline (Milestone 4).
"""

import argparse
import math

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling

from config import data_config, model_config


def parse_args():
    parser = argparse.ArgumentParser(description="Hitung loss/perplexity di val set")
    parser.add_argument("--base-model", default=model_config.base_model)
    parser.add_argument("--adapter", default=None, help="Path ke LoRA adapter (kosongkan untuk baseline)")
    parser.add_argument("--val-file", default=data_config.val_file)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None, help="Batasi jumlah contoh (buat cek cepat)")
    return parser.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ds = load_dataset("json", data_files={"validation": args.val_file})["validation"]
    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))

    def tokenize(examples):
        return tokenizer(examples["text"], truncation=True, max_length=model_config.max_seq_length)

    ds = ds.map(tokenize, batched=True, remove_columns=["text"])

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device)

    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter)

    model.eval()

    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    loader = DataLoader(ds, batch_size=args.batch_size, collate_fn=collator)

    total_loss = 0.0
    total_batches = 0
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            total_loss += out.loss.item()
            total_batches += 1

    avg_loss = total_loss / total_batches
    perplexity = math.exp(avg_loss)

    print(f"Model      : {args.base_model}")
    print(f"Adapter    : {args.adapter or '(tidak ada, baseline)'}")
    print(f"N contoh   : {len(ds)}")
    print(f"Val loss   : {avg_loss:.4f}")
    print(f"Perplexity : {perplexity:.2f}")


if __name__ == "__main__":
    main()
