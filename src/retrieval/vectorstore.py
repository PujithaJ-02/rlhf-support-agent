"""
src/retrieval/vectorstore.py

PURPOSE:
    Takes all 26,872 Q&A chunks from chunker.py,
    converts each chunk into a vector using Ollama (free, local),
    and stores all vectors in ChromaDB on disk.

    Run this ONCE to build the database.
    Every run after that loads from disk — no re-embedding needed.

HOW IT WORKS:
    1. Each chunk is passed through nomic-embed-text (runs via Ollama)
    2. The model converts text into a vector (list of numbers)
    3. Similar texts get similar vectors
    4. ChromaDB stores and searches those vectors efficiently

USED BY:
    src/retrieval/retriever.py
"""

from pathlib import Path
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings


CHROMA_DIR  = "data/processed/chroma_db"
COLLECTION  = "support_docs"
EMBED_MODEL = "nomic-embed-text"


def get_embeddings() -> OllamaEmbeddings:
    """
    Loads the Ollama embedding model.
    Runs locally — no internet, no API key, no cost.
    Make sure Ollama app is running before calling this.
    """
    print(f"Loading embedding model: {EMBED_MODEL} via Ollama")
    return OllamaEmbeddings(model=EMBED_MODEL)


def build_vectorstore(chunks: list[Document]) -> Chroma:
    """
    Embeds all chunks and saves them to ChromaDB on disk.
    Processes in batches of 200 to show progress clearly.
    """
    print(f"Building vector store with {len(chunks)} chunks...")
    print(f"Saving to: {CHROMA_DIR}")
    print("This will take 5-10 minutes — Ollama runs on your CPU.")

    Path(CHROMA_DIR).mkdir(parents=True, exist_ok=True)

    embeddings    = get_embeddings()
    batch_size    = 200
    vectorstore   = None
    total_batches = (len(chunks) - 1) // batch_size + 1

    for i in range(0, len(chunks), batch_size):
        batch     = chunks[i : i + batch_size]
        batch_num = i // batch_size + 1

        print(f"  Batch {batch_num}/{total_batches} — {len(batch)} chunks...")

        if vectorstore is None:
            vectorstore = Chroma.from_documents(
                documents=batch,
                embedding=embeddings,
                persist_directory=CHROMA_DIR,
                collection_name=COLLECTION,
            )
        else:
            vectorstore.add_documents(batch)

    total = vectorstore._collection.count()
    print(f"\nVector store built successfully.")
    print(f"Total vectors stored: {total}")
    return vectorstore


def load_vectorstore() -> Chroma:
    """
    Loads existing ChromaDB from disk.
    Fast — no embedding, just loading.
    """
    print(f"Loading vector store from disk: {CHROMA_DIR}")
    embeddings  = get_embeddings()
    vectorstore = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
        collection_name=COLLECTION,
    )
    total = vectorstore._collection.count()
    print(f"Loaded. Total vectors: {total}")
    return vectorstore


def get_vectorstore(chunks: list[Document] = None) -> Chroma:
    """
    Smart loader — the only function other modules should call.

    - If ChromaDB already exists on disk  → loads it (fast)
    - If ChromaDB does not exist yet      → builds it from chunks (slow, once)
    """
    chroma_path = Path(CHROMA_DIR)

    if chroma_path.exists() and any(chroma_path.iterdir()):
        print("Found existing vector store — loading from disk.")
        return load_vectorstore()

    if chunks is None:
        raise ValueError(
            "No vector store found on disk. "
            "Pass chunks= argument to build one."
        )

    return build_vectorstore(chunks)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")

    from src.ingestion.loader import load_documents
    from src.ingestion.chunker import chunk_documents

    print("=" * 50)
    print("Step 1 — Loading documents")
    print("=" * 50)
    docs = load_documents()

    print("\n" + "=" * 50)
    print("Step 2 — Chunking into Q&A pairs")
    print("=" * 50)
    chunks = chunk_documents(docs)

    print("\n" + "=" * 50)
    print("Step 3 — Building vector store")
    print("Ollama is running locally — no API key needed")
    print("=" * 50)
    vs = get_vectorstore(chunks)

    print("\n" + "=" * 50)
    print("Step 4 — Testing retrieval")
    print("=" * 50)

    test_queries = [
        "how do I cancel my order?",
        "I want a refund",
        "I forgot my password",
    ]

    for query in test_queries:
        print(f"\nQuery: {query}")
        results = vs.similarity_search(query, k=2)
        for i, doc in enumerate(results):
            print(f"  Result {i + 1}:")
            print(f"    Category : {doc.metadata.get('category')}")
            print(f"    Intent   : {doc.metadata.get('intent')}")
            print(f"    Preview  : {doc.page_content[:120]}...")