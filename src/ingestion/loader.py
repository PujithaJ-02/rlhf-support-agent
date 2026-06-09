"""
src/ingestion/loader.py

Loads raw support documents from data/raw/ and returns them
as LangChain Document objects ready for chunking.
"""

import os
import json
from pathlib import Path
from langchain_community.document_loaders import TextLoader  # noqa
from langchain_core.documents import Document


def load_documents(raw_dir: str = "data/raw") -> list[Document]:
    """
    Loads all .txt files from data/raw/ and returns a list
    of LangChain Document objects.

    Each Document has:
        - page_content : the text content
        - metadata     : filename, category, source
    """
    raw_path = Path(raw_dir)
    documents = []

    txt_files = list(raw_path.glob("*.txt"))

    if not txt_files:
        raise FileNotFoundError(f"No .txt files found in {raw_dir}")

    print(f"Found {len(txt_files)} files in {raw_dir}")

    for filepath in sorted(txt_files):
        category = filepath.stem.replace("_support", "").upper()

        loader = TextLoader(str(filepath), encoding="utf-8")
        docs = loader.load()

        for doc in docs:
            doc.metadata["category"] = category
            doc.metadata["source"]   = filepath.name
            doc.metadata["filepath"] = str(filepath)

        documents.extend(docs)
        print(f"  Loaded: {filepath.name} — {len(docs)} document(s)")

    print(f"\nTotal documents loaded: {len(documents)}")
    return documents


def load_metadata(raw_dir: str = "data/raw") -> dict:
    """
    Loads dataset_metadata.json for reference and logging.
    """
    metadata_path = Path(raw_dir) / "dataset_metadata.json"

    if not metadata_path.exists():
        return {}

    with open(metadata_path, "r") as f:
        return json.load(f)


if __name__ == "__main__":
    docs = load_documents()
    metadata = load_metadata()

    print("\n--- Metadata ---")
    print(f"Source     : {metadata.get('source')}")
    print(f"Total rows : {metadata.get('total_examples')}")
    print(f"Categories : {list(metadata.get('categories', {}).keys())}")

    print("\n--- Sample Document ---")
    print(f"Content preview : {docs[0].page_content[:200]}")
    print(f"Metadata        : {docs[0].metadata}")
