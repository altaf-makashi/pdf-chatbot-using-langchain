"""
STEP 1: Load PDFs/TXT from data/ -> chunk -> embed (local) -> save to Chroma.
Run once, or again whenever your documents change:  python ingest.py
"""

import shutil
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

DATA_DIR = Path("data")
VECTORSTORE_DIR = "vectorstore"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def load_documents():
    docs = []
    for path in sorted(DATA_DIR.glob("*")):
        if path.suffix.lower() == ".pdf":
            loader = PyPDFLoader(str(path))
        elif path.suffix.lower() == ".txt":
            loader = TextLoader(str(path), encoding="utf-8")
        else:
            continue
        pages = loader.load()
        for p in pages:
            p.metadata["source_file"] = path.name
        docs.extend(pages)
        print(f"Loaded {len(pages)} pages/sections from {path.name}")
    return docs


def main():
    if not DATA_DIR.exists() or not any(DATA_DIR.glob("*")):
        print(f"Put your PDF/TXT files in the '{DATA_DIR}/' folder first.")
        raise SystemExit(1)

    docs = load_documents()
    if not docs:
        print("No readable PDF/TXT files found.")
        raise SystemExit(1)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunks")

    # Start clean so re-running never creates duplicate chunks
    shutil.rmtree(VECTORSTORE_DIR, ignore_errors=True)

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=VECTORSTORE_DIR,
    )
    print(f"Saved vector store to '{VECTORSTORE_DIR}/'. Next: python query.py")


if __name__ == "__main__":
    main()
