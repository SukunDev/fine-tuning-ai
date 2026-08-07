"""Format dataset instruksi mentah ke template teks untuk causal LM (GPT2),
lalu split train/val.

Base model (cahya/gpt2-small-indonesian-522M) adalah GPT2 polos tanpa chat
template, jadi dipakai template Alpaca klasik: instruksi (+input opsional)
diikuti jawaban, ditutup token EOS supaya model belajar kapan berhenti.
"""

import argparse
import json
import os
import random

from transformers import AutoTokenizer

from config import model_config

TEMPLATE_WITH_INPUT = "### Instruksi:\n{instruction}\n\n### Input:\n{input}\n\n### Jawaban:\n{output}"
TEMPLATE_NO_INPUT = "### Instruksi:\n{instruction}\n\n### Jawaban:\n{output}"


def format_example(row: dict, eos_token: str) -> str:
    if row.get("input"):
        text = TEMPLATE_WITH_INPUT.format(**row)
    else:
        text = TEMPLATE_NO_INPUT.format(**row)
    return text + eos_token


def parse_args():
    parser = argparse.ArgumentParser(description="Format dataset mentah ke template chat + split train/val")
    parser.add_argument("--input", default="data/raw/alpaca-id-cleaned.jsonl")
    parser.add_argument("--output", default="data/processed/")
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()

    input_path = args.input
    if os.path.isdir(input_path):
        candidates = [f for f in os.listdir(input_path) if f.endswith(".jsonl")]
        if not candidates:
            raise FileNotFoundError(f"Tidak ada file .jsonl di {input_path}")
        input_path = os.path.join(input_path, candidates[0])

    tokenizer = AutoTokenizer.from_pretrained(model_config.base_model)
    eos_token = tokenizer.eos_token

    with open(input_path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]

    examples = [format_example(row, eos_token) for row in rows]

    random.seed(args.seed)
    random.shuffle(examples)
    n_val = max(1, int(len(examples) * args.val_ratio))
    val_examples, train_examples = examples[:n_val], examples[n_val:]

    os.makedirs(args.output, exist_ok=True)
    train_path = os.path.join(args.output, "train.jsonl")
    val_path = os.path.join(args.output, "val.jsonl")

    for path, split in [(train_path, train_examples), (val_path, val_examples)]:
        with open(path, "w", encoding="utf-8") as f:
            for text in split:
                f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")

    lengths_tokens = [len(tokenizer(text)["input_ids"]) for text in examples]
    print(f"Total pasangan  : {len(examples)}")
    print(f"Train / Val     : {len(train_examples)} / {len(val_examples)}")
    print(f"Rata-rata token : {sum(lengths_tokens) / len(lengths_tokens):.1f}")
    print(f"Token maks      : {max(lengths_tokens)}")
    print(f"Tersimpan ke    : {train_path}, {val_path}")


if __name__ == "__main__":
    main()
