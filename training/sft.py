"""
SFT training script for both stages.

Usage:
    python training/sft.py --config configs/sft_openassistant.yaml
    python training/sft.py --config configs/sft_fortnite.yaml

Both stages consume the same format:
    {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, ...]}

Applies the ChatML template used by SmolLM-360M-Instruct:
    <|im_start|>user
    ...<|im_end|>
    <|im_start|>assistant
    ...<|im_end|>
"""

import argparse
import json
import yaml
from pathlib import Path
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer


ROOT = Path(__file__).resolve().parent.parent

CHATML_TEMPLATE = (
    "{% for message in messages %}"
    "{{'<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n'}}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"
)


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

    # Ensure ChatML special tokens exist
    special_tokens = {"additional_special_tokens": ["<|im_start|>", "<|im_end|>"]}
    if tokenizer.pad_token is None:
        special_tokens["pad_token"] = "<|im_end|>"
    tokenizer.add_special_tokens(special_tokens)

    # Set the chat template
    tokenizer.chat_template = CHATML_TEMPLATE

    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.resize_token_embeddings(len(tokenizer))

    # Load dataset — either from HF or local JSONL
    if "dataset_path" in config:
        dataset = load_jsonl_dataset(ROOT / config["dataset_path"])
    else:
        from datasets import load_dataset
        dataset = load_dataset(config["dataset"], split="train")
        # If loading pre-processed JSONL from the processed dir
        processed_path = ROOT / "data" / "processed" / "openassistant_filtered.jsonl"
        if processed_path.exists():
            dataset = load_jsonl_dataset(str(processed_path))

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
