"""
src/reward_model/infer.py

Loads the trained reward model and scores any
(question, answer) pair from 0.0 to 1.0.
"""

import sys
sys.path.insert(0, ".")

import torch
import torch.nn.functional as F
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
)
from pathlib import Path

SAVED_MODEL_PATH = "src/reward_model/saved_model"
MAX_LENGTH       = 256

_tokenizer = None
_model     = None


def load_reward_model():
    global _tokenizer, _model

    if _tokenizer is not None and _model is not None:
        return _tokenizer, _model

    if not Path(SAVED_MODEL_PATH).exists():
        raise FileNotFoundError(
            f"No saved model found at {SAVED_MODEL_PATH}. "
            "Run src/reward_model/train.py first."
        )

    print(f"Loading reward model from: {SAVED_MODEL_PATH}")
    _tokenizer = DistilBertTokenizer.from_pretrained(SAVED_MODEL_PATH)
    _model     = DistilBertForSequenceClassification.from_pretrained(
        SAVED_MODEL_PATH
    )
    _model.eval()
    print("Reward model loaded.")
    return _tokenizer, _model


def reward_score(question: str, answer: str) -> float:
    tokenizer, model = load_reward_model()
    text    = f"Question: {question} Answer: {answer}"
    encoded = tokenizer(
        text,
        max_length=MAX_LENGTH,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        outputs = model(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
        )
    probs = F.softmax(outputs.logits, dim=-1)
    score = probs[0][1].item()
    return round(score, 4)


def score_candidates(question: str, candidates: list[str]) -> list[dict]:
    scored = []
    for i, answer in enumerate(candidates):
        score = reward_score(question, answer)
        scored.append({
            "answer": answer,
            "score":  score,
            "rank":   i + 1,
        })
    scored.sort(key=lambda x: x["score"], reverse=True)
    for i, item in enumerate(scored):
        item["rank"] = i + 1
    return scored


def get_best_answer(question: str, candidates: list[str]) -> str:
    scored = score_candidates(question, candidates)
    return scored[0]["answer"]


if __name__ == "__main__":
    from src.pipeline.generator import generate_candidates
    question   = "How do I cancel my order?"
    result     = generate_candidates(question)
    candidates = result["candidates"]
    scored     = score_candidates(question, candidates)
    print("\nCANDIDATES RANKED BY REWARD MODEL:")
    for item in scored:
        print(f"  Rank {item['rank']} -- Score: {item['score']}")
        print(f"  {item['answer'][:150]}...")