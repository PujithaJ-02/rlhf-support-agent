"""
src/pipeline/generator.py

PURPOSE:
    Connects the RAG retriever to Llama 3.2 via Ollama.
    For any user question it:
        1. Retrieves relevant chunks from ChromaDB
        2. Builds a prompt with those chunks as context
        3. Calls Llama 3.2 three times to get 3 candidate answers
        4. Returns all 3 candidates for the reward model to score

WHY 3 CANDIDATES:
    The reward model needs options to compare.
    It scores each candidate and the best one gets shown to the user.
    This is the core of the RLHF selection loop.

USED BY:
    src/pipeline/pipeline.py (Phase 5)
    app/streamlit_app.py (Phase 6)
"""

import sys
sys.path.insert(0, ".")

from langchain_ollama import OllamaLLM
from src.retrieval.retriever import retrieve, format_context


LLM_MODEL       = "llama3.2"
NUM_CANDIDATES  = 3
TEMPERATURE     = 0.7


def get_llm(temperature: float = TEMPERATURE) -> OllamaLLM:
    """
    Loads Llama 3.2 via Ollama.
    Runs completely locally — no API key, no cost.

    temperature controls creativity:
        0.0 = always the same answer (deterministic)
        0.7 = some variety between candidates (what we want)
        1.0 = very random and creative
    """
    return OllamaLLM(
        model=LLM_MODEL,
        temperature=temperature,
    )


def build_prompt(question: str, context: str) -> str:
    """
    Builds the prompt we send to Llama 3.2.

    We inject the retrieved context so Llama answers
    from our knowledge base, not from its training data.
    This prevents hallucination.
    """
    cleaned_context = context.replace("{{Order Number}}", "[ORDER-ID]")
    cleaned_context = cleaned_context.replace("{{", "[")
    cleaned_context = cleaned_context.replace("}}", "]")

    return f"""You are a helpful customer support agent for ShopBot.
The context below contains real customer support Q&A pairs.
Use the Support Responses in the context to answer the customer question.
Be specific, friendly, and concise — 2 to 3 sentences maximum.
Do not say you lack information if the context contains Support Responses.

CONTEXT:
{cleaned_context}

CUSTOMER QUESTION:
{question}

YOUR ANSWER (use the Support Responses above as your guide):"""


def generate_candidates(
    question: str,
    k: int = 4,
    num_candidates: int = NUM_CANDIDATES,
) -> dict:
    """
    Main function — generates multiple candidate answers for a question.

    Args:
        question      : the user's question in plain English
        k             : number of chunks to retrieve from ChromaDB
        num_candidates: number of candidate answers to generate

    Returns:
        dict with:
            question   : original question
            context    : retrieved chunks as formatted string
            candidates : list of candidate answer strings
            sources    : list of source metadata dicts
    """
    print(f"\nGenerating {num_candidates} candidates for: {question}")

    print("  Step 1 — Retrieving relevant chunks...")
    docs    = retrieve(question, k=k)
    context = format_context(docs)

    sources = [
        {
            "category": doc.metadata.get("category", "UNKNOWN"),
            "intent":   doc.metadata.get("intent", "unknown"),
            "preview":  doc.page_content[:100],
        }
        for doc in docs
    ]

    prompt = build_prompt(question, context)

    print(f"  Step 2 — Calling Llama 3.2 for {num_candidates} candidates...")
    llm        = get_llm(temperature=TEMPERATURE)
    candidates = []

    for i in range(num_candidates):
        print(f"    Generating candidate {i + 1}/{num_candidates}...")
        answer = llm.invoke(prompt)
        candidates.append(answer.strip())

    print(f"  Done. {len(candidates)} candidates generated.")

    return {
        "question":   question,
        "context":    context,
        "candidates": candidates,
        "sources":    sources,
    }


if __name__ == "__main__":
    test_questions = [
        "How do I cancel my order?",
        "I want to get a refund for my purchase.",
    ]

    for question in test_questions:
        result = generate_candidates(question)

        print("\n" + "=" * 60)
        print(f"QUESTION: {result['question']}")
        print("=" * 60)

        print("\nSOURCES RETRIEVED:")
        for i, source in enumerate(result["sources"]):
            print(f"  {i+1}. [{source['category']}] {source['intent']}")

        print("\nCANDIDATE ANSWERS:")
        for i, candidate in enumerate(result["candidates"]):
            print(f"\n--- Candidate {i+1} ---")
            print(candidate)

        print("\n")