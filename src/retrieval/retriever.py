"""
src/retrieval/retriever.py

Clean search interface over ChromaDB.
Loads vector store lazily — only when first search is called.
"""

import sys
sys.path.insert(0, ".")

from langchain_core.documents import Document

_vectorstore = None


def get_vs():
    """Lazy loader — only connects to ChromaDB when first called."""
    global _vectorstore
    if _vectorstore is None:
        from src.retrieval.vectorstore import get_vectorstore
        _vectorstore = get_vectorstore()
    return _vectorstore


def retrieve(query: str, k: int = 4) -> list[Document]:
    """
    Searches ChromaDB for the most relevant chunks.
    """
    results = get_vs().similarity_search(query, k=k)
    return results


def retrieve_with_scores(query: str, k: int = 4) -> list[tuple[Document, float]]:
    """
    Same as retrieve() but also returns similarity scores.
    """
    results = get_vs().similarity_search_with_score(query, k=k)
    return results


def format_context(docs: list[Document]) -> str:
    """
    Formats retrieved chunks into a single context string
    ready to be injected into an LLM prompt.
    """
    context_parts = []

    for i, doc in enumerate(docs):
        part = (
            f"[Source {i + 1}]\n"
            f"Category: {doc.metadata.get('category', 'UNKNOWN')}\n"
            f"Intent: {doc.metadata.get('intent', 'unknown')}\n"
            f"{doc.page_content}"
        )
        context_parts.append(part)

    return "\n\n---\n\n".join(context_parts)


if __name__ == "__main__":
    test_queries = [
        "how do I track my delivery?",
        "my payment failed what do I do?",
        "how do I cancel my subscription?",
    ]

    for query in test_queries:
        print(f"\nQuery: {query}")
        print("-" * 40)
        docs    = retrieve(query, k=2)
        context = format_context(docs)
        print(context[:400])
        print("...")