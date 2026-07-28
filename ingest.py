"""
PHASE 1 + 2: Load documents -> chunk -> embed -> store persistently.
Fully OFFLINE -- embeddings run locally, no API keys needed.

Run this once whenever you add/change documents in the data/ folder.
Usage: python ingest.py
"""

from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

DATA_DIR = Path("data")
VECTORSTORE_DIR = "vectorstore"

# Local embedding model -> runs on your machine, no API calls, no rate limits.
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def load_documents():
    """Load every PDF/TXT in data/ and tag each with source metadata."""
    docs = []
    for file_path in DATA_DIR.glob("*"):
        if file_path.suffix.lower() == ".pdf":
            loader = PyPDFLoader(str(file_path))
        elif file_path.suffix.lower() == ".txt":
            loader = TextLoader(str(file_path))
        else:
            continue

        loaded = loader.load()
        for doc in loaded:
            doc.metadata["source_file"] = file_path.name
        docs.extend(loaded)
        print(f"Loaded {len(loaded)} pages/sections from {file_path.name}")

    return docs


def chunk_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunks")
    return chunks


def build_vectorstore(chunks):
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=VECTORSTORE_DIR,
    )
    print(f"Vector store persisted to '{VECTORSTORE_DIR}/'")
    return vectorstore


if __name__ == "__main__":
    if not any(DATA_DIR.glob("*")):
        print(f"No files found in {DATA_DIR}/ -- add PDFs or TXT files there first.")
        exit(1)

    documents = load_documents()
    chunks = chunk_documents(documents)
    build_vectorstore(chunks)
    print("\nIngestion complete. Run query.py to ask questions.")
