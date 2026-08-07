"""Konfigurasi default untuk fine-tuning LoRA.

Semua hyperparameter didefinisikan di sini (bukan hardcode di dalam fungsi),
supaya bisa diubah lewat argparse tanpa menyentuh kode training.
"""

from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    base_model: str = "cahya/gpt2-small-indonesian-522M"
    max_seq_length: int = 512


@dataclass
class LoraConfig:
    r: int = 8
    alpha: int = 16
    dropout: float = 0.05
    target_modules: list = field(default_factory=lambda: ["c_attn"])  # sesuaikan dengan arsitektur base model
    bias: str = "none"
    task_type: str = "CAUSAL_LM"


@dataclass
class TrainingConfig:
    output_dir: str = "adapters/best"
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    num_train_epochs: int = 3
    max_steps: int = -1  # -1 = ikuti num_train_epochs
    save_steps: int = 100
    logging_steps: int = 10
    fp16: bool = True
    warmup_ratio: float = 0.03
    seed: int = 42


@dataclass
class DataConfig:
    train_file: str = "data/processed/train.jsonl"
    val_file: str = "data/processed/val.jsonl"


model_config = ModelConfig()
lora_config = LoraConfig()
training_config = TrainingConfig()
data_config = DataConfig()
