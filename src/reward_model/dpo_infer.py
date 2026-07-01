"""
src/reward_model/dpo_infer.py

PURPOSE:
    Generates answers using the DPO fine-tuned Qwen2-0.5B model.
    
    This is the Policy Optimization step of RLHF:
        - distilBERT reward model = scores answers (already built)
        - Qwen2-0.5B DPO model   = generates better answers by default
    
    Together these complete the full RLHF pipeline:
        SFT (base model) → Reward Model → DPO → Better answers
    
USED BY:
    src/pipeline/generator.py
    app/streamlit_app.py
"""

import sys
sys.path.insert(0, ".")

import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


DPO_MODEL_PATH = "src/reward_model/dpo_model"
BASE_MODEL     = "Qwen/Qwen2-0.5B"

_tokenizer = None
_model     = None


def load_dpo_model():
    """
    Loads the DPO fine-tuned model.
    Uses module level cache so we only load once per session.
    
    The model is a LoRA adapter on top of Qwen2-0.5B.
    LoRA means we only trained 0.1% of parameters (540K out of 494M)
    which is why training was fast and the file is small.
    """
    global _tokenizer, _model

    if _tokenizer is not None and _model is not None:
        return _tokenizer, _model

    if not Path(DPO_MODEL_PATH).exists():
        raise FileNotFoundError(
            f"No DPO model found at {DPO_MODEL_PATH}. "
            "Run the DPO training notebook on Google Colab first."
        )

    print(f"Loading DPO model from: {DPO_MODEL_PATH}")

    _tokenizer = AutoTokenizer.from_pretrained(DPO_MODEL_PATH)
    _tokenizer.pad_token = _tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        dtype=torch.float32,
        device_map="cpu",
    )

    _model = PeftModel.from_pretrained(base_model, DPO_MODEL_PATH)
    _model.eval()

    print("DPO model loaded successfully.")
    return _tokenizer, _model


def generate_dpo_answer(question: str, context: str = "", max_new_tokens: int = 200) -> str:
    """
    Generates a single answer using the DPO fine-tuned model.
    
    The DPO training taught this model to prefer the same kinds
    of answers that humans rated as helpful -- so it generates
    better answers by default compared to the base model.
    
    Args:
        question       : the customer question
        context        : retrieved context from ChromaDB (optional)
        max_new_tokens : maximum length of generated answer
    
    Returns:
        generated answer as a string
    """
    tokenizer, model = load_dpo_model()

    if context:
        prompt = (
            f"You are a helpful ShopBot customer support agent.\n"
            f"Use the following context to answer the question.\n\n"
            f"Context:\n{context[:500]}\n\n"
            f"Customer question: {question}\n"
            f"Answer:"
        )
    else:
        prompt = (
            f"You are a helpful ShopBot customer support agent.\n"
            f"Customer question: {question}\n"
            f"Answer:"
        )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated = outputs[0][inputs["input_ids"].shape[1]:]
    answer    = tokenizer.decode(generated, skip_special_tokens=True)
    return answer.strip()


def generate_dpo_candidates(
    question: str,
    context: str = "",
    num_candidates: int = 3,
) -> list[str]:
    """
    Generates multiple candidate answers using the DPO model.
    These candidates are then scored by the reward model.
    
    This combines both components of our RLHF pipeline:
        - DPO model generates candidates (policy optimization)
        - Reward model scores and picks the best (reward signal)
    """
    candidates = []
    for i in range(num_candidates):
        print(f"    DPO candidate {i + 1}/{num_candidates}...")
        answer = generate_dpo_answer(question, context)
        candidates.append(answer)
    return candidates


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from src.retrieval.retriever import retrieve, format_context
    from src.reward_model.infer import score_candidates

    question = "How do I cancel my order?"
    print(f"Question: {question}")
    print("\nStep 1 -- Retrieving context...")
    docs    = retrieve(question, k=3)
    context = format_context(docs)

    print("\nStep 2 -- Generating candidates with DPO model...")
    candidates = generate_dpo_candidates(question, context, num_candidates=3)

    print("\nStep 3 -- Scoring with reward model...")
    scored = score_candidates(question, candidates)

    print("\nCANDIDATES RANKED BY REWARD MODEL:")
    for item in scored:
        print(f"\n  Rank {item['rank']} -- Score: {item['score']}")
        print(f"  {item['answer'][:200]}...")

    print(f"\nBEST ANSWER:")
    print(scored[0]["answer"])