"""Generate teks dari base model (opsional + LoRA adapter).

Dipakai untuk baseline check (Milestone 1, tanpa --adapter) dan nanti
untuk komparasi base vs base+adapter (Milestone 4, dengan --adapter).
"""

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import model_config


def load_model(base_model: str, adapter: str | None, device: str):
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device)

    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)

    model.eval()
    return tokenizer, model


def generate(tokenizer, model, prompt: str, max_new_tokens: int, device: str) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_p=0.9,
            temperature=0.8,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(output[0], skip_special_tokens=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate teks dari base model (+ opsional LoRA adapter)")
    parser.add_argument("--base-model", default=model_config.base_model)
    parser.add_argument("--adapter", default=None, help="Path ke LoRA adapter (kosongkan untuk baseline)")
    parser.add_argument("--prompt", default="Indonesia adalah negara yang")
    parser.add_argument("--max-new-tokens", type=int, default=60)
    return parser.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer, model = load_model(args.base_model, args.adapter, device)

    print(f"Model  : {args.base_model}")
    print(f"Adapter: {args.adapter or '(tidak ada, baseline)'}")
    print(f"Device : {device}")
    print(f"Params : {sum(p.numel() for p in model.parameters()):,}")
    print("-" * 60)
    print(f"Prompt : {args.prompt}")
    print("-" * 60)

    text = generate(tokenizer, model, args.prompt, args.max_new_tokens, device)
    print(text)


if __name__ == "__main__":
    main()
