"""
PHASE 1 + 2: Ask questions against your ingested documents from the terminal.
Shows the answer AND which file/page it came from.

Fully OFFLINE -- no API keys, no rate limits, no credits, no internet needed
(after the one-time model download via Ollama).

Setup:
  1. Install Ollama: https://ollama.com/download
  2. Pull a model:   ollama pull llama3.1:8b
  3. pip install langchain-ollama

Usage: python query.py
"""

import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
# from langchain_community.chat_models import ChatOllama
from langchain_core.callbacks import StreamingStdOutCallbackHandler


EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # local, offline
LLM_MODEL = "llama3.2:1b"  # local, offline, served by Ollama
VECTORSTORE_DIR = "vectorstore"

PROMPT = ChatPromptTemplate.from_template(
    """You are a helpful assistant answering questions using ONLY the context below.
If the answer is not in the context, say "I don't know based on the provided documents."
Do not make up information.

Match the depth of your answer to what's being asked:
- If the question mentions marks/points (e.g. "for 10 marks"), give a detailed,
  well-structured answer with definition, key points/characteristics, and examples
- answer each question briefly with related context you find


Context:
{context}

Question: {question}

Answer:"""
)


def format_docs_with_sources(docs):
    """Turn retrieved chunks into a context string AND keep track of sources separately."""
    context_parts = []
    for i, doc in enumerate(docs, 1):
        context_parts.append(f"[{i}] {doc.page_content}")
    return "\n\n".join(context_parts)


def build_chain():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = Chroma(persist_directory=VECTORSTORE_DIR, embedding_function=embeddings)

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 6, "fetch_k": 15},
    )

    # ChatOllama talks to your local Ollama server (default: http://localhost:11434)
    llm = ChatOllama(
        model="llama3.2:1b",
        temprature=0.2,
        callbacks=[StreamingStdOutCallbackHandler()]
    )

    chain = (
        {"context": retriever | format_docs_with_sources, "question": RunnablePassthrough()}
        | PROMPT
        | llm
        | StrOutputParser()
    )
    return chain, retriever


def ask(question, chain, retriever):
    print("\n--- ANSWER ---\n")
    answer = chain.invoke(question)
    print("\n")
    sources = retriever.invoke(question)
    print("\n--- SOURCES ---")
    seen = set()
    for doc in sources:
        src = doc.metadata.get("source_file", "unknown")
        page = doc.metadata.get("page", "?")
        key = (src, page)
        if key not in seen:
            print(f"- {src} (page {page})")
            seen.add(key)


if __name__ == "__main__":
    if not os.path.exists(VECTORSTORE_DIR):
        print("No vector store found. Run 'python ingest.py' first.")
        exit(1)

    chain, retriever = build_chain()
    print("Document Q&A ready (fully offline via Ollama). Type 'exit' to quit.\n")

    while True:
        q = input("Ask a question: ").strip()
        if q.lower() in ("exit", "quit"):
            break
        if not q:
            continue
        ask(q, chain, retriever)
        print()
