"""
Level 3 evaluation: automated RAG metrics using RAGAS.
Fully OFFLINE -- both the RAG bot and the RAGAS judge run via Ollama, locally.

Metrics computed:
  - faithfulness       -> does the answer stick to what's actually in the retrieved context?
  - answer_relevancy   -> does the answer actually address the question asked?
  - context_precision  -> are the retrieved chunks relevant to the question?
  - context_recall     -> did retrieval find everything needed to answer correctly?

Setup:
  1. Fill in ragas_questions.json with real questions + correct reference answers.
  2. Ollama running locally with a model pulled (ollama pull llama3.1:8b).
  3. Run: python ragas_evaluate.py
"""

import json

from ragas import evaluate, EvaluationDataset, SingleTurnSample
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

from query import build_chain, EMBEDDING_MODEL
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama

with open("ragas_questions.json") as f:
    test_cases = json.load(f)

print("Building RAG chain (offline, via Ollama)...")
chain, retriever = build_chain()

samples = []
for case in test_cases:
    question = case["question"]
    reference = case["ground_truth"]

    print(f"Running: {question}")
    answer = chain.invoke(question)
    retrieved_docs = retriever.invoke(question)
    contexts = [doc.page_content for doc in retrieved_docs]

    samples.append(
        SingleTurnSample(
            user_input=question,
            response=answer,
            retrieved_contexts=contexts,
            reference=reference,
        )
    )

dataset = EvaluationDataset(samples=samples)

# Judge also runs locally via Ollama -- no API, no quota, no cost
judge_llm = LangchainLLMWrapper(ChatOllama(model="qwen2:0.5b", temperature=0.0))
judge_embeddings = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL))

print("\nRunning RAGAS evaluation locally (this may take a while on CPU)...\n")

results = evaluate(
    dataset=dataset,
    metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    llm=judge_llm,
    embeddings=judge_embeddings,
)

print("=" * 50)
print("RAGAS RESULTS")
print("=" * 50)
df = results.to_pandas()
print(df[["user_input", "faithfulness", "answer_relevancy", "context_precision", "context_recall"]])

print("\nAverages:")
for metric in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
    print(f"  {metric}: {df[metric].mean():.2f}")

df.to_csv("ragas_results.csv", index=False)
print("\nFull results saved to ragas_results.csv")