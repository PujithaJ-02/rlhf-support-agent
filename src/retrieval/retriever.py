"""
src/retrieval/retriever.py

PURPOSE:
    Clean search interface over ChromaDB.
    This is the only file other modules should call
    when they need to search the knowledge base.

    Takes a user question and returns the most relevant
    Q&A chunks from the 26,872 vectors stored in ChromaDB.

USED BY:
    src/agent/agent.py
    src/pipeline/generator.py
"""

import sys
sys.path.insert(0, ".")

from langchain_core.documents import Document
from src.retrieval.vectorstore import get_vectorstore


def retrieve(query: str, k: int = 4) -> list[Document]:
    """
    Searches ChromaDB for the most relevant chunks.

    Args:
        query : the user question in plain English
        k     : number of chunks to return (default 4)

    Returns:
        list of Document objects, most relevant first
    """
    vectorstore = get_vectorstore()
    results     = vectorstore.similarity_search(query, k=k)
    return results


def retrieve_with_scores(query: str, k: int = 4) -> list[tuple[Document, float]]:
    """
    Same as retrieve() but also returns similarity scores.
    Score closer to 0 = more similar.
    Useful for debugging retrieval quality.
    """
    vectorstore = get_vectorstore()
    results     = vectorstore.similarity_search_with_score(query, k=k)
    return results


def format_context(docs: list[Document]) -> str:
    """
    Formats retrieved chunks into a single string
    ready to be injected into an LLM prompt.

    Args:
        docs : list of Document objects from retrieve()

    Returns:
        formatted string with all chunks joined
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