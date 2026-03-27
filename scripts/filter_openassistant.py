"""
Filter OpenAssistant dataset to ~50k high-quality conversations.
Reconstructs conversation threads from the tree structure and outputs
in the shared chat format.
"""

import json
import yaml
from collections import defaultdict
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm


ROOT = Path(__file__).resolve().parent.parent

ROLE_MAP = {"prompter": "user", "assistant": "assistant"}


def load_config():
    with open(ROOT / "configs" / "sft_openassistant.yaml") as f:
        return yaml.safe_load(f)


def build_threads(dataset):
    """Reconstruct conversation threads from the message tree."""
    messages = {}
    children = defaultdict(list)

    for row in dataset:
        msg_id = row["message_id"]
        messages[msg_id] = row
        if row["parent_id"]:
            children[row["parent_id"]].append(msg_id)

    # Find root messages (no parent)
    roots = [mid for mid, msg in messages.items() if msg["parent_id"] is None]

    threads = []
    for root_id in roots:
        # Walk down the tree, picking the best-ranked child at each level
        thread = []
        current_id = root_id

        while current_id:
            msg = messages[current_id]
            thread.append({
                "role": ROLE_MAP[msg["role"]],
                "content": msg["text"],
            })

            kids = children.get(current_id, [])
            if not kids:
                break

            # Pick the best-ranked assistant reply (rank 0 = best)
            ranked = [messages[k] for k in kids if messages[k]["rank"] is not None]
            if ranked:
                best = min(ranked, key=lambda m: m["rank"])
                current_id = best["message_id"]
            else:
                # No ranked children, just pick the first
                current_id = kids[0]

        threads.append(thread)

    return threads


def filter_threads(threads, max_samples, min_turns=2, lang="en"):
    """Filter to high-quality, English, multi-turn conversations."""
    filtered = []
    for thread in threads:
        if len(thread) < min_turns:
            continue
        # Must start with user and alternate
        if thread[0]["role"] != "user":
            continue
        filtered.append({"turns": thread})
        if len(filtered) >= max_samples:
            break
    return filtered


def main():
    config = load_config()
    dataset = load_dataset(config["dataset"], split="train")
    print(f"Loaded {len(dataset)} messages from {config['dataset']}")

    # Filter to English only
    dataset = dataset.filter(lambda x: x["lang"] == "en")
    print(f"After language filter: {len(dataset)} messages")

    threads = build_threads(dataset)
    print(f"Reconstructed {len(threads)} conversation threads")

    filtered = filter_threads(threads, config["max_samples"])
    print(f"After filtering: {len(filtered)} conversations")

    output_path = ROOT / "data" / "processed" / "openassistant_filtered.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for conv in filtered:
            f.write(json.dumps(conv) + "\n")

    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
