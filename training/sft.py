"""
SFT training script for Fortnite style fine-tuning.

Usage:
    python training/sft.py --config configs/sft_fortnite.yaml

Fine-tunes SmolLM-360M-Instruct (which already has ChatML and instruction-following)
directly on Fortnite-dialect conversations.

Dataset format:
    {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, ...]}
"""

import argparse
import json
import yaml
from pathlib import Path
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer


ROOT = Path(__file__).resolve().parent.parent


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_jsonl_dataset(path: str) -> Dataset:
    records = []
    with open(path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return Dataset.from_list(records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to training config YAML")
    args = parser.parse_args()

    config = load_config(args.config)

    model_name = config["model_name"]
    print(f"Loading model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name)

    dataset = load_jsonl_dataset(ROOT / config["dataset_path"])
    print(f"Training on {len(dataset)} conversations")

    training_args = SFTConfig(
        output_dir=str(ROOT / config["output_dir"]),
        num_train_epochs=config["num_train_epochs"],
        per_device_train_batch_size=config["per_device_train_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        learning_rate=config["learning_rate"],
        warmup_ratio=config["warmup_ratio"],
        logging_steps=config["logging_steps"],
        save_steps=config["save_steps"],
        max_seq_length=config["max_seq_length"],
        bf16=True,
        report_to="wandb",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=dataset,
    )

    trainer.train()
    trainer.save_model()
    tokenizer.save_pretrained(str(ROOT / config["output_dir"]))


if __name__ == "__main__":
    main()
