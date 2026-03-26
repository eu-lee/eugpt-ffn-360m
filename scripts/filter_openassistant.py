"""
Filter OpenAssistant dataset to ~50k high-quality conversations.
"""

import json
import yaml
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm


ROOT = Path(__file__).resolve().parent.parent


def load_config():
    with open(ROOT / "configs" / "sft_openassistant.yaml") as f:
        return yaml.safe_load(f)


def main():
    config = load_config()
    dataset = load_dataset(config["dataset"], split="train")
    print(f"Loaded {len(dataset)} examples from {config['dataset']}")

    # TODO: implement filtering logic (quality score, language, etc.)

    output_path = ROOT / "data" / "processed" / "openassistant_filtered.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Output will be saved to {output_path}")


if __name__ == "__main__":
    main()
