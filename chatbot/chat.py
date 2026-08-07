"""CLI chat interaktif: base model + LoRA adapter.

Model base (GPT2) tidak punya memori multi-turn, jadi tiap giliran diproses
independen memakai template Alpaca yang sama seperti saat training.
"""

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import model_config

PROMPT_TEMPLATE = "### Instruksi:\n{instruction}\n\n### Jawaban:\n"
STOP_MARKERS = ["### Instruksi", "### Jawaban", "### Input", "## Jawaban", "<|endoftext|>"]


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


def generate_reply(tokenizer, model, instruction: str, max_new_tokens: int, device: str) -> str:
    prompt = PROMPT_TEMPLATE.format(instruction=instruction)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_p=0.9,
            temperature=0.8,
            repetition_penalty=1.3,
            no_repeat_ngram_size=3,
            pad_token_id=tokenizer.eos_token_id,
        )

    full_text = tokenizer.decode(output[0], skip_special_tokens=False)
    reply = full_text[len(prompt):]

    for marker in STOP_MARKERS:
        idx = reply.find(marker)
        if idx != -1:
            reply = reply[:idx]

    return reply.strip()


def parse_args():
    parser = argparse.ArgumentParser(description="Chat interaktif dengan base model + LoRA adapter")
    parser.add_argument("--base-model", default=model_config.base_model)
    parser.add_argument("--adapter", default="adapters/best", help="Path ke LoRA adapter (kosongkan untuk baseline)")
    parser.add_argument("--max-new-tokens", type=int, default=120)
    return parser.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading {args.base_model} (+ adapter: {args.adapter or 'tidak ada'}) di {device}...")
    tokenizer, model = load_model(args.base_model, args.adapter, device)
    print("Siap. Ketik pertanyaan/instruksi (ketik 'exit' atau 'quit' untuk keluar).\n")

    while True:
        try:
            instruction = input("Kamu: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not instruction:
            continue
        if instruction.lower() in {"exit", "quit"}:
            break

        reply = generate_reply(tokenizer, model, instruction, args.max_new_tokens, device)
        print(f"Bot : {reply}\n")


if __name__ == "__main__":
    main()
