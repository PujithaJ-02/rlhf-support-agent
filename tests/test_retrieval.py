"""
tests/test_retrieval.py

Unit tests for the RAG retrieval pipeline.
Run with: pytest tests/test_retrieval.py -v
"""

import sys
import pytest
sys.path.insert(0, ".")


def test_load_documents():
    from src.ingestion.loader import load_documents
    docs = load_documents()
    assert len(docs) == 11, f"Expected 11 documents, got {len(docs)}"
    assert all(hasattr(d, "page_content") for d in docs)
    assert all(hasattr(d, "metadata") for d in docs)
    print(f"Loaded {len(docs)} documents")


def test_chunk_documents():
    from src.ingestion.loader import load_documents
    from src.ingestion.chunker import chunk_documents
    docs   = load_documents()
    chunks = chunk_documents(docs)
    assert len(chunks) > 20000, f"Expected 20000+ chunks, got {len(chunks)}"
    assert all(hasattr(c, "page_content") for c in chunks)
    assert all("category" in c.metadata for c in chunks)
    print(f"Created {len(chunks)} chunks")


def test_chunk_content_quality():
    from src.ingestion.loader import load_documents
    from src.ingestion.chunker import chunk_documents
    docs   = load_documents()
    chunks = chunk_documents(docs)
    for chunk in chunks[:10]:
        assert len(chunk.page_content) > 50, "Chunk too short"
        assert "Customer Question:" in chunk.page_content or \
               "Support Response:" in chunk.page_content, \
               "Chunk missing Q&A structure"
    print("Chunk quality check passed")


def test_vectorstore_loads():
    from src.retrieval.vectorstore import get_vectorstore
    vs = get_vectorstore()
    count = vs._collection.count()
    assert count > 20000, f"Expected 20000+ vectors, got {count}"
    print(f"Vector store loaded with {count} vectors")


def test_retrieval_returns_results():
    from src.retrieval.retriever import retrieve
    results = retrieve("how do I cancel my order", k=3)
    assert len(results) == 3, f"Expected 3 results, got {len(results)}"
    assert all(hasattr(r, "page_content") for r in results)
    print(f"Retrieved {len(results)} results")


def test_retrieval_relevance():
    from src.retrieval.retriever import retrieve
    results = retrieve("I want a refund", k=3)
    categories = [r.metadata.get("category") for r in results]
    assert "REFUND" in categories, \
        f"Expected REFUND in results, got {categories}"
    print(f"Relevance check passed: {categories}")


def test_format_context():
    from src.retrieval.retriever import retrieve, format_context
    docs    = retrieve("track my order", k=2)
    context = format_context(docs)
    assert "[Source 1]" in context
    assert "[Source 2]" in context
    assert len(context) > 100
    print("Context formatting check passed")