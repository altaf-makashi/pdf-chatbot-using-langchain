"""
STEP 2: Ask questions about your documents from the terminal.
Usage: python query.py

Needs Ollama running with the model pulled:  ollama pull llama3.2:1b
"""

import os

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "llama3.2:1b"          # change to "llama3.2:3b" for better answers
VECTORSTORE_DIR = "vectorstore"

# Keep this exact sentence: the evaluation script looks for "I don't know".
REFUSAL = "I don't know based on the provided documents."

PROMPT = ChatPromptTemplate.from_template(
    """You are a helpful assistant. Answer the question using ONLY the context below.

Rules:
- If the answer is not clearly in the context, reply exactly:
  "I don't know based on the provided documents."
- Do not use outside knowledge, even for easy questions.
- Do not make up information.
- Keep answers short and clear. If the question mentions marks (e.g. "for 10 marks"),
  give a longer, structured answer with a definition, key points and examples.

Context:
{context}

Question: {question}

Answer:"""
)


def format_docs(docs):
    return "\n\n".join(f"[{i}] {d.page_content}" for i, d in enumerate(docs, 1))


def build_chain():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = Chroma(persist_directory=VECTORSTORE_DIR, embedding_function=embeddings)

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 6, "fetch_k": 15},
    )

    llm = ChatOllama(model=LLM_MODEL, temperature=0)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | PROMPT
        | llm
        | StrOutputParser()
    )
    return chain, retriever


def ask(question, chain, retriever):
    answer = chain.invoke(question)
    print("\n--- ANSWER ---\n")
    print(answer)

    print("\n--- SOURCES ---")
    seen = set()
    for doc in retriever.invoke(question):
        key = (doc.metadata.get("source_file", "unknown"), doc.metadata.get("page", "?"))
        if key not in seen:
            seen.add(key)
            # PyPDFLoader counts pages from 0, so +1 matches your PDF viewer
            page = key[1] + 1 if isinstance(key[1], int) else key[1]
            print(f"- {key[0]} (page {page} in a PDF viewer, index {key[1]})")


if __name__ == "__main__":
    if not os.path.exists(VECTORSTORE_DIR):
        print("No vector store found. Run 'python ingest.py' first.")
        raise SystemExit(1)

    chain, retriever = build_chain()
    print(f"Ready (offline, model: {LLM_MODEL}). Type 'exit' to quit.\n")

    while True:
        q = input("Ask a question: ").strip()
        if q.lower() in ("exit", "quit"):
            break
        if q:
            ask(q, chain, retriever)
            print()
