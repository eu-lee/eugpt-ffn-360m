"""
Convert DailyDialog conversations into Fortnite terminology using an LLM API.

Uses hand-written examples from data/few_shot_examples/fortnite_dialogs.jsonl
as few-shot prompts to guide the conversion.

Output format matches the shared chat format:
{"turns": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, ...]}

User turns stay as normal English, assistant turns get Fortnite-ified.
"""

import json
import re
import time
import yaml
import anthropic
from dotenv import load_dotenv
from pathlib import Path
from tqdm import tqdm


ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


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
        cleaned = re.sub(r'\s+([?.!,;:\'"])', r'\1', text.strip())
        turns.append({"role": role, "content": cleaned})
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
        "- Output valid JSON: a list of {\"role\": ..., \"content\": ...} objects\n\n"
        "Fortnite vocabulary to draw from (use sparingly and naturally, NOT in every sentence):\n"
        "- Combat/weapons: 200 pumped (hit for max damage), boxed like a fish (trapped in a box), "
        "clapped (killed/beaten), spaz (pump shotgun), aug (burst AR), deagle (hand cannon), "
        "aimbot (cheating/suspiciously good aim), no scope, one-shot (low health), "
        "elim (elimination), thirst (finish a downed player), W-key (push aggressively)\n"
        "- Items/resources: big pots (50 shield), minis (25 shield), mats (materials), "
        "V-Bucks (currency), 5-5-1 (mat count: wood-brick-metal in hundreds), "
        "shocks (shockwave grenades), zero point fish, launch pad, loadout\n"
        "- Gameplay: bot/botted (bad player/playing badly), cracked (very skilled or shields broken), "
        "dogwater (terrible), goated (extremely good), sweat (tryhard player), noob, "
        "skill issue (you're just bad), clutch (win against the odds), dub (a win), "
        "GGs (good game), off-spawn (right after landing), hot drop (busy landing spot)\n"
        "- Strategy: rotate/rotation (moving to next area), deadside (safer side of zone), "
        "storm circle/zone (the shrinking safe area), turtling (boxing up defensively), "
        "cranking 90s (build technique), box fight/build battle, high ground, "
        "heal-off, camper, push\n"
        "- Map/world: POI (point of interest), drop spot (where you land), lobby (waiting area/respawn), "
        "reboot van/card, spawn island, Battle Bus, Tilted Towers, the island, "
        "loot spawn, vault/vaulted (removed from game)\n"
        "- Meta: nerfed (made weaker), buffed (made stronger), OP (overpowered), "
        "OG (original/veteran), meta (current best strategy), SBMM, pubs (public lobbies), FNCS\n"
        "- Slang: no cap (truthfully), bet (agreed), say less (understood), finna (going to), "
        "bussin' (very good), lowkey/highkey, hits different, straight fire, mid (mediocre), "
        "dogwater (terrible), periodt (end of discussion), big yikes (embarrassing), "
        "shook (scared), let him cook (let them do their thing), understood the assignment, "
        "poggers (excited), sus (suspicious), yeet (throw)\n\n"
        "Style notes:\n"
        "- The goal is someone who TALKS like a Fortnite player, not a Fortnite dictionary\n"
        "- Most of the sentence should be normal English — just sprinkle in Fortnite terms where they fit naturally\n"
        "- Fortnite words can replace normal words even outside gaming context "
        "(e.g. 'rotate to the store across the street', 'that movie was dogwater', 'she clutched the exam')\n"
        "- Not every sentence needs a Fortnite term. Some responses can just have the vibe/attitude "
        "of a Fortnite player without any specific terminology\n"
        "- Aim for roughly 2-4 Fortnite terms per response, not one per phrase\n"
        "- The tone should feel like a teenager/young adult who plays a lot of Fortnite, "
        "not a Fortnite wiki article\n"
    )

    if few_shot_examples:
        prompt += "\nHere are some examples:\n\n"
        for ex in few_shot_examples:
            prompt += f"Converted conversation:\n{json.dumps(ex['messages'], indent=2)}\n\n"

    return prompt


def convert_dialog(client, config, system_prompt, dialog):
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
        raw_text = response.content[0].text.strip()

        # Strip markdown code fences if present
        if raw_text.startswith("```"):
            raw_text = raw_text.split("\n", 1)[1]  # remove ```json line
            raw_text = raw_text.rsplit("```", 1)[0].strip()

        result = json.loads(raw_text)
        # Validate structure
        if not isinstance(result, list):
            print(f"Validation fail: not a list, got {type(result).__name__}")
            return None
        if len(result) != len(turns):
            print(f"Validation fail: expected {len(turns)} turns, got {len(result)}")
            return None
        for orig, conv in zip(turns, result):
            if orig["role"] != conv.get("role"):
                print(f"Validation fail: role mismatch, expected {orig['role']}, got {conv.get('role')}")
                return None
        # Re-inject original user turns to ensure they're unchanged
        for i, orig in enumerate(turns):
            if orig["role"] == "user":
                result[i]["content"] = orig["content"]
        return result
    except anthropic.RateLimitError:
        print("Rate limited, waiting 60s...")
        time.sleep(60)
        return convert_dialog(client, config, system_prompt, dialog)
    except json.JSONDecodeError as e:
        print(f"JSON parse error: {e}\nRaw response: {raw_text[:200]}")
        return None
    except Exception as e:
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

    # Load DailyDialog from local raw files
    dialog_file = ROOT / "data" / "raw" / "ijcnlp_dailydialog" / config["daily_dialog_split"] / f"dialogues_{config['daily_dialog_split']}.txt"
    with open(dialog_file) as f:
        dataset = []
        for line in f:
            if line.strip():
                turns = [t.strip() for t in line.strip().split("__eou__") if t.strip()]
                dataset.append(turns)
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

    max_samples = config.get("max_samples", len(dataset))

    with open(output_path, "a") as f:
        for i, example in enumerate(tqdm(dataset, total=max_samples, initial=existing)):
            if i < existing:
                continue
            if i >= max_samples:
                break

            dialog = example
            if len(dialog) > 8:
                continue
            result = convert_dialog(client, config, system_prompt, dialog)

            if result:
                record = {"messages": result}
                f.write(json.dumps(record) + "\n")
                f.flush()

            # Stay under rate limit (5 req/min = 13s between requests)
            time.sleep(13)


if __name__ == "__main__":
    main()
