"""
tests/test_reward.py

Unit tests for the RLHF reward model.
Run with: pytest tests/test_reward.py -v
"""

import sys
import pytest
sys.path.insert(0, ".")


def test_reward_model_loads():
    from src.reward_model.infer import load_reward_model
    tokenizer, model = load_reward_model()
    assert tokenizer is not None
    assert model is not None
    print("Reward model loaded successfully")


def test_reward_score_returns_float():
    from src.reward_model.infer import reward_score
    score = reward_score(
        question="how do I cancel my order",
        answer="Go to My Orders and click Cancel within 1 hour of purchase."
    )
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0
    print(f"Reward score: {score}")


def test_good_answer_scores_higher_than_bad():
    from src.reward_model.infer import reward_score
    question    = "how do I cancel my order"
    good_answer = (
        "To cancel your order, go to My Orders, find the order "
        "you want to cancel and click the Cancel button within 1 hour of purchase."
    )
    bad_answer  = "I don't have enough information to answer that question."

    good_score = reward_score(question, good_answer)
    bad_score  = reward_score(question, bad_answer)

    print(f"Good answer score: {good_score}")
    print(f"Bad answer score:  {bad_score}")

    assert good_score > bad_score, (
        f"Expected good answer ({good_score}) to score higher "
        f"than bad answer ({bad_score})"
    )


def test_score_candidates_returns_ranked_list():
    from src.reward_model.infer import score_candidates
    question   = "how do I reset my password"
    candidates = [
        "Click Forgot Password on the login page and follow the email instructions.",
        "I cannot help with that.",
        "To reset your password go to the login page click Forgot Password enter your email and follow the reset link sent to you.",
    ]
    scored = score_candidates(question, candidates)
    assert len(scored) == 3
    assert scored[0]["score"] >= scored[1]["score"] >= scored[2]["score"]
    assert all("answer" in s for s in scored)
    assert all("score" in s for s in scored)
    assert all("rank" in s for s in scored)
    print(f"Rank 1 score: {scored[0]['score']}")
    print(f"Rank 2 score: {scored[1]['score']}")
    print(f"Rank 3 score: {scored[2]['score']}")


def test_get_best_answer_returns_string():
    from src.reward_model.infer import get_best_answer
    question   = "how do I track my delivery"
    candidates = [
        "Visit My Orders and click Track to see real time delivery updates.",
        "Please contact support.",
        "I am unable to help with that request.",
    ]
    best = get_best_answer(question, candidates)
    assert isinstance(best, str)
    assert len(best) > 10
    print(f"Best answer: {best[:80]}...")


def test_preference_dataset_loads():
    from src.reward_model.dataset import load_preference_pairs
    pairs = load_preference_pairs()
    assert len(pairs) >= 100, f"Expected at least 100 pairs, got {len(pairs)}"
    assert all("question" in p for p in pairs)
    assert all("good_answer" in p for p in pairs)
    assert all("bad_answer" in p for p in pairs)
    assert all("good_label" in p for p in pairs)
    print(f"Loaded {len(pairs)} preference pairs")


def test_training_records_are_balanced():
    from src.reward_model.dataset import load_preference_pairs, build_training_records
    pairs   = load_preference_pairs()
    records = build_training_records(pairs)
    positive = sum(1 for r in records if r["label"] == 1)
    negative = sum(1 for r in records if r["label"] == 0)
    assert positive == negative, (
        f"Expected balanced dataset, got {positive} positive and {negative} negative"
    )
    print(f"Balanced dataset: {positive} positive, {negative} negative")