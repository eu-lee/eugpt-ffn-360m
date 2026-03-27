"""
Convert DailyDialog conversations into Fortnite terminology using an LLM API.

Uses hand-written examples from data/few_shot_examples/fortnite_dialogs.jsonl
as few-shot prompts to guide the conversion.

Output format matches the shared chat format:
{"turns": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, ...]}

User turns stay as normal English, assistant turns get Fortnite-ified.
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


def dialog_to_turns(dialog: list[str]) -> list[dict]:
    """Convert a flat DailyDialog list into alternating user/assistant turns."""
    turns = []
    for i, text in enumerate(dialog):
        role = "user" if i % 2 == 0 else "assistant"
        turns.append({"role": role, "content": text.strip()})
    return turns


def build_system_prompt(few_shot_examples: list[dict]) -> str:
    prompt = (
        "You are a translator that converts assistant responses in conversations "
        "into Fortnite terminology and slang. You will be given a conversation with "
        "alternating user and assistant turns.\n\n"
        "Rules:\n"
        "- ONLY rewrite the assistant turns into Fortnite speak\n"
        "- Keep user turns EXACTLY as they are\n"
        "- Preserve the same meaning and conversational flow\n"
        "- Replace everyday language in assistant turns with Fortnite-themed equivalents\n"
        "- Output valid JSON: a list of {\"role\": ..., \"content\": ...} objects\n"
    )

    if few_shot_examples:
        prompt += "\nHere are some examples:\n\n"
        for ex in few_shot_examples:
            original_turns = ex.get("turns", [])
            # Show the original assistant lines for context
            original_assistant = ex.get("original_assistant_lines", [])
            if original_assistant:
                prompt += "Original assistant lines:\n"
                for line in original_assistant:
                    prompt += f"  - {line}\n"
            prompt += f"\nConverted conversation:\n{json.dumps(ex['turns'], indent=2)}\n\n"

    return prompt


def convert_dialog(client: anthropic.Anthropic, config: dict, system_prompt: str, dialog: list[str]) -> list[dict] | None:
    turns = dialog_to_turns(dialog)

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
                        f"Convert the assistant turns in this conversation to Fortnite speak. "
                        f"Keep user turns unchanged.\n\n"
                        f"{json.dumps(turns, indent=2)}\n\n"
                        f"Respond with only the JSON list of turns."
                    ),
                }
            ],
        )
        result = json.loads(response.content[0].text)
        # Validate structure
        if not isinstance(result, list):
            return None
        if len(result) != len(turns):
            return None
        for orig, conv in zip(turns, result):
            if orig["role"] != conv.get("role"):
                return None
            # User turns should be unchanged
            if orig["role"] == "user" and orig["content"] != conv["content"]:
                return None
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
                record = {"turns": result}
                f.write(json.dumps(record) + "\n")
                f.flush()


if __name__ == "__main__":
    main()
