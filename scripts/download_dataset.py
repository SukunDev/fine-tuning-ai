"""Download dataset instruksi bahasa Indonesia dari HuggingFace Hub.

Default: cahya/alpaca-id-cleaned (instruksi/input/output, format Alpaca,
cocok dipasangkan dengan base model cahya/gpt2-small-indonesian-522M).
"""

import argparse
import json
import os

from datasets import load_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Download dataset instruksi dari HuggingFace Hub")
    parser.add_argument("--dataset", default="cahya/alpaca-id-cleaned")
    parser.add_argument("--split", default="train")
    parser.add_argument("--output", default="data/raw/alpaca-id-cleaned.jsonl")
    return parser.parse_args()


def main():
    args = parse_args()

    ds = load_dataset(args.dataset, split=args.split)
    print(f"Dataset : {args.dataset} [{args.split}]")
    print(f"Rows    : {len(ds)}")
    print(f"Columns : {ds.column_names}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for row in ds:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Tersimpan ke {args.output}")


if __name__ == "__main__":
    main()
