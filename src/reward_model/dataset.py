"""
src/reward_model/dataset.py

Creates training data for the reward model.
"""

import sys
import json
import random
from pathlib import Path

sys.path.insert(0, ".")

from src.ingestion.loader import load_documents
from src.ingestion.chunker import chunk_documents

PREFERENCES_PATH = "data/preferences/preference_pairs.json"
random.seed(42)


def degrade_answer(answer: str) -> str:
    strategies = [
        lambda a: a[:20] + "...",
        lambda a: "I don't have enough information to answer that.",
        lambda a: "Please contact our support team for assistance.",
        lambda a: "Thank you for reaching out to us today.",
        lambda a: a[-20:] if len(a) > 20 else "I cannot help with that.",
    ]
    return random.choice(strategies)(answer)


def create_preference_pairs(chunks, num_pairs: int = 500) -> list[dict]:
    pairs   = []
    sampled = random.sample(chunks, min(num_pairs, len(chunks)))

    for chunk in sampled:
        content  = chunk.page_content
        lines    = content.split("\n")
        question = None
        answer   = None

        for line in lines:
            if line.startswith("Customer Question:"):
                question = line.replace("Customer Question:", "").strip()
            if line.startswith("Support Response:"):
                answer = line.replace("Support Response:", "").strip()

        if not question or not answer:
            continue

        question = question.replace("{{", "[").replace("}}", "]")
        answer   = answer.replace("{{", "[").replace("}}", "]")

        if len(answer) < 30:
            continue

        bad_answer = degrade_answer(answer)

        pairs.append({
            "question":    question,
            "good_answer": answer,
            "bad_answer":  bad_answer,
            "good_label":  1,
            "bad_label":   0,
            "category":    chunk.metadata.get("category", "UNKNOWN"),
            "intent":      chunk.metadata.get("intent", "unknown"),
        })

    return pairs


def save_preference_pairs(pairs: list[dict]) -> None:
    Path("data/preferences").mkdir(parents=True, exist_ok=True)
    with open(PREFERENCES_PATH, "w") as f:
        json.dump(pairs, f, indent=2)
    print(f"Saved {len(pairs)} preference pairs to {PREFERENCES_PATH}")


def load_preference_pairs() -> list[dict]:
    if not Path(PREFERENCES_PATH).exists():
        raise FileNotFoundError(
            f"No preference pairs found at {PREFERENCES_PATH}. "
            "Run dataset.py first to generate them."
        )
    with open(PREFERENCES_PATH, "r") as f:
        pairs = json.load(f)
    print(f"Loaded {len(pairs)} preference pairs from {PREFERENCES_PATH}")
    return pairs


def build_training_records(pairs: list[dict]) -> list[dict]:
    records = []
    for pair in pairs:
        records.append({
            "text":  f"Question: {pair['question']} Answer: {pair['good_answer']}",
            "label": 1,
        })
        records.append({
            "text":  f"Question: {pair['question']} Answer: {pair['bad_answer']}",
            "label": 0,
        })
    random.shuffle(records)
    return records


if __name__ == "__main__":
    docs    = load_documents()
    chunks  = chunk_documents(docs)
    pairs   = create_preference_pairs(chunks, num_pairs=500)
    save_preference_pairs(pairs)
    records = build_training_records(pairs)
    print(f"Total training records: {len(records)}")