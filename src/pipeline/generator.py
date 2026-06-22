"""
src/pipeline/generator.py

Connects RAG retriever to Llama 3.2 via Ollama.
All Ollama connections are lazy — only made when needed.
"""

import sys
sys.path.insert(0, ".")

LLM_MODEL      = "llama3.2"
NUM_CANDIDATES = 3
TEMPERATURE    = 0.7


def get_llm(temperature: float = TEMPERATURE):
    from langchain_ollama import OllamaLLM
    return OllamaLLM(
        model=LLM_MODEL,
        temperature=temperature,
    )


def build_prompt(question: str, context: str) -> str:
    cleaned = context.replace("{{", "[").replace("}}", "]")
    return f"""You are a helpful customer support agent for ShopBot.
The context below contains real customer support Q&A pairs.
Use the Support Responses in the context to answer the customer question.
Be specific, friendly, and concise — 2 to 3 sentences maximum.
Do not say you lack information if the context contains Support Responses.

CONTEXT:
{cleaned}

CUSTOMER QUESTION:
{question}

YOUR ANSWER (use the Support Responses above as your guide):"""


def generate_candidates(
    question: str,
    k: int = 4,
    num_candidates: int = NUM_CANDIDATES,
) -> dict:
    from src.retrieval.retriever import retrieve, format_context

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