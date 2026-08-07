"""Fine-tuning LoRA untuk base model causal LM (GPT2) di data instruksi Indonesia.

Load base model (fp16) -> bungkus peft.LoraConfig -> train pakai
transformers.Trainer -> simpan adapter (bukan full model) ke adapters/.
"""

import argparse
import time

import torch
from datasets import load_dataset
from peft import LoraConfig as PeftLoraConfig
from peft import get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from config import data_config, lora_config, model_config, training_config


def _fmt_duration(seconds: float) -> str:
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}h{m:02d}m{s:02d}s"


class ProgressCallback(TrainerCallback):
    """Log step/pct/loss/lr/elapsed/ETA ke stdout, gaya sama seperti llm-iqmal/training/train.py."""

    def on_train_begin(self, args, state, control, **kwargs):
        self.t_start = time.time()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return

        max_steps = state.max_steps
        step = state.global_step
        elapsed = time.time() - self.t_start

        if "eval_loss" in logs:
            print(
                f"{'=' * 60}\n"
                f"step {step:5d}/{max_steps} | eval_loss {logs['eval_loss']:.4f}\n"
                f"{'=' * 60}",
                flush=True,
            )
            return

        if "loss" not in logs:
            return

        pct = step / max_steps * 100 if max_steps else 0.0
        avg_step = elapsed / step if step else 0.0
        eta = avg_step * (max_steps - step) if max_steps else 0.0
        lr = logs.get("learning_rate", 0.0)
        print(
            f"step {step:5d}/{max_steps} ({pct:5.1f}%) | loss {logs['loss']:.4f} | lr {lr:.2e} | "
            f"elapsed {_fmt_duration(elapsed)} | ETA {_fmt_duration(eta)}",
            flush=True,
        )


def parse_args():
    parser = argparse.ArgumentParser(description="LoRA fine-tuning untuk base model Indonesia")
    parser.add_argument("--base-model", default=model_config.base_model)
    parser.add_argument("--train-file", default=data_config.train_file)
    parser.add_argument("--val-file", default=data_config.val_file)
    parser.add_argument("--output-dir", default=training_config.output_dir)
    parser.add_argument("--max-steps", type=int, default=training_config.max_steps)
    parser.add_argument("--num-train-epochs", type=int, default=training_config.num_train_epochs)
    return parser.parse_args()


def load_tokenized_dataset(tokenizer, train_file: str, val_file: str, max_seq_length: int):
    raw = load_dataset("json", data_files={"train": train_file, "validation": val_file})

    def tokenize(examples):
        return tokenizer(examples["text"], truncation=True, max_length=max_seq_length)

    return raw.map(tokenize, batched=True, remove_columns=["text"])


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Tokenisasi dataset...", flush=True)
    dataset = load_tokenized_dataset(tokenizer, args.train_file, args.val_file, model_config.max_seq_length)

    print("Load base model...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device)

    peft_config = PeftLoraConfig(
        r=lora_config.r,
        lora_alpha=lora_config.alpha,
        lora_dropout=lora_config.dropout,
        target_modules=lora_config.target_modules,
        bias=lora_config.bias,
        task_type=lora_config.task_type,
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=training_config.per_device_train_batch_size,
        gradient_accumulation_steps=training_config.gradient_accumulation_steps,
        learning_rate=training_config.learning_rate,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        save_steps=training_config.save_steps,
        logging_steps=training_config.logging_steps,
        fp16=training_config.fp16 and device == "cuda",
        warmup_ratio=training_config.warmup_ratio,
        seed=training_config.seed,
        eval_strategy="steps",
        eval_steps=training_config.save_steps,
        save_strategy="steps",
        save_total_limit=2,
        report_to="none",
        disable_tqdm=True,
    )

    print(f"Model         : {args.base_model}")
    print(f"Device        : {device}")
    print(f"Train / Val   : {len(dataset['train'])} / {len(dataset['validation'])}")
    print(f"Max steps     : {args.max_steps if args.max_steps > 0 else '(ikut num_train_epochs)'}")
    print(f"Batch efektif : {training_config.per_device_train_batch_size * training_config.gradient_accumulation_steps}")
    print("-" * 60)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        callbacks=[ProgressCallback()],
    )

    trainer.train()

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Adapter tersimpan ke {args.output_dir}")


if __name__ == "__main__":
    main()
