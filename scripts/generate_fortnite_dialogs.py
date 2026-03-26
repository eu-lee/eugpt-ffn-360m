"""
Convert DailyDialog conversations into Fortnite terminology using an LLM API.

Uses hand-written examples from data/few_shot_examples/fortnite_dialogs.jsonl
as few-shot prompts to guide the conversion.
"""

import json
import yaml
import anthropic
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm


ROOT = Path(__file__).resolve().parent.parent


def load_config():
    with open(ROOT / "configs" / "generation.yaml") as f:
        return yaml.safe_load(f)


def load_few_shot_examples(path: Path, n: int) -> list[dict]:
    examples = []
    if path.exists():
        with open(path) as f:
            for line in f:
                if line.strip():
                    examples.append(json.loads(line))
    return examples[:n]


def build_system_prompt(few_shot_examples: list[dict]) -> str:
    prompt = (
        "You are a translator that converts everyday conversations into "
        "Fortnite terminology and slang. Keep the same conversational structure "
        "and meaning, but replace everyday language with Fortnite-themed equivalents.\n\n"
        "Rules:\n"
        "- Replace everyday locations, objects, and actions with Fortnite equivalents\n"
        "- Keep the conversation natural and coherent\n"
        "- Maintain the same number of turns in the dialog\n"
        "- Output valid JSON with keys 'original' and 'fortnite'\n"
    )

    if few_shot_examples:
        prompt += "\nHere are some examples of the conversions:\n\n"
        for ex in few_shot_examples:
            prompt += f"Original:\n{json.dumps(ex['original'])}\n\n"
            prompt += f"Fortnite version:\n{json.dumps(ex['fortnite'])}\n\n"

    return prompt


def convert_dialog(client: anthropic.Anthropic, config: dict, system_prompt: str, dialog: list[str]) -> dict | None:
    try:
        response = client.messages.create(
            model=config["model"],
            max_tokens=config["max_tokens"],
            temperature=config["temperature"],
            system=system_prompt,
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"Convert this dialog into Fortnite terminology:\n\n"
                        f"{json.dumps(dialog)}\n\n"
                        f"Respond with only valid JSON."
                    ),
                }
            ],
        )
        result = json.loads(response.content[0].text)
        return result
    except (json.JSONDecodeError, Exception) as e:
        print(f"Error converting dialog: {e}")
        return None


def main():
    config = load_config()

    few_shot_examples = load_few_shot_examples(
        ROOT / config["few_shot_path"],
        config["num_few_shot"],
    )
    print(f"Loaded {len(few_shot_examples)} few-shot examples")

    system_prompt = build_system_prompt(few_shot_examples)

    dataset = load_dataset("daily_dialog", split=config["daily_dialog_split"])
    print(f"Loaded {len(dataset)} dialogs from DailyDialog")

    client = anthropic.Anthropic()

    output_path = ROOT / config["output_path"]
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume from existing output
    existing = 0
    if output_path.exists():
        with open(output_path) as f:
            existing = sum(1 for line in f if line.strip())
    print(f"Resuming from {existing} existing conversions")

    with open(output_path, "a") as f:
        for i, example in enumerate(tqdm(dataset, initial=existing)):
            if i < existing:
                continue

            dialog = example["dialog"]
            result = convert_dialog(client, config, system_prompt, dialog)

            if result:
                record = {
                    "original": dialog,
                    "fortnite": result.get("fortnite", result),
                    "daily_dialog_idx": i,
                }
                f.write(json.dumps(record) + "\n")
                f.flush()


if __name__ == "__main__":
    main()
