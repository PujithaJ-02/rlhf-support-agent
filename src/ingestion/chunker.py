"""
src/ingestion/chunker.py

Splits documents into meaningful Q&A pair chunks.
Each chunk = one customer question + one support response.
"""

from langchain_core.documents import Document


def chunk_documents(
    documents: list[Document],
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[Document]:
    """
    Splits documents into Q&A pair chunks.
    Each chunk contains one complete question + answer pair.

    Args:
        documents    : list of LangChain Document objects from loader.py
        chunk_size   : ignored here, kept for API consistency
        chunk_overlap: ignored here, kept for API consistency

    Returns:
        list of Document chunks, one per Q&A pair
    """
    chunks = []

    for doc in documents:
        category = doc.metadata.get("category", "UNKNOWN")
        source   = doc.metadata.get("source", "")
        lines    = doc.page_content.split("\n")

        current_question = None
        current_response = None
        current_intent   = None

        for line in lines:
            line = line.strip()

            if line.startswith("Customer Question:"):
                if current_question and current_response:
                    content = (
                        f"Category: {category}\n"
                        f"Intent: {current_intent or 'unknown'}\n"
                        f"Customer Question: {current_question}\n"
                        f"Support Response: {current_response}"
                    )
                    chunks.append(Document(
                        page_content=content,
                        metadata={
                            "category": category,
                            "source":   source,
                            "intent":   current_intent or "unknown",
                        }
                    ))
                current_question = line.replace("Customer Question:", "").strip()
                current_response = None
                current_intent   = None

            elif line.startswith("Support Response:"):
                current_response = line.replace("Support Response:", "").strip()

            elif line.startswith("Intent:"):
                current_intent = line.replace("Intent:", "").strip()

        if current_question and current_response:
            content = (
                f"Category: {category}\n"
                f"Intent: {current_intent or 'unknown'}\n"
                f"Customer Question: {current_question}\n"
                f"Support Response: {current_response}"
            )
            chunks.append(Document(
                page_content=content,
                metadata={
                    "category": category,
                    "source":   source,
                    "intent":   current_intent or "unknown",
                }
            ))

    print(f"Original documents  : {len(documents)}")
    print(f"Total chunks created: {len(chunks)}")
    print(f"Avg chunk size      : {sum(len(c.page_content) for c in chunks) // max(len(chunks), 1)} chars")

    return chunks


def preview_chunks(chunks: list[Document], n: int = 3) -> None:
    """
    Prints n sample chunks so you can verify quality.
    """
    print(f"\n--- Sample Chunks (showing {n} of {len(chunks)}) ---")
    for i, chunk in enumerate(chunks[:n]):
        print(f"\nChunk {i + 1}:")
        print(f"  Category : {chunk.metadata.get('category')}")
        print(f"  Intent   : {chunk.metadata.get('intent')}")
        print(f"  Length   : {len(chunk.page_content)} chars")
        print(f"  Content  :\n{chunk.page_content[:250]}")
        print()


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from src.ingestion.loader import load_documents

    docs   = load_documents()
    chunks = chunk_documents(docs)
    preview_chunks(chunks)
