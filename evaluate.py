"""
Evaluate your RAG pipeline against a set of test questions.

1. Edit eval_questions.json:
   - Add real questions you know the answer to from your PDF
   - Set "expected_page" to the page where that answer actually lives
   - Set "should_answer": false for questions the bot should refuse (out-of-scope)

2. Run: python evaluate.py

It checks TWO separate things:
  - Retrieval hit: did the expected page show up in the retrieved sources?
  - Refusal behavior: did the bot correctly say "I don't know" for out-of-scope questions?
"""

import json
from query import build_chain

with open("eval_questions.json") as f:
    test_cases = json.load(f)

chain, retriever = build_chain()

results = []

print(f"Running {len(test_cases)} test cases...\n")

for i, case in enumerate(test_cases, 1):
    question = case["question"]
    expected_page = case.get("expected_page")
    should_answer = case.get("should_answer", True)

    answer = chain.invoke(question)
    answer_lower = answer.lower()
    refused = "i don't know" in answer_lower or "don't know based on" in answer_lower

    retrieved_docs = retriever.invoke(question)
    retrieved_pages = [d.metadata.get("page") for d in retrieved_docs]

    if should_answer:
        # Check 1: did it actually attempt an answer (not refuse)?
        answered_correctly = not refused
        # Check 2: was the right page retrieved?
        retrieval_hit = expected_page in retrieved_pages if expected_page is not None else None
    else:
        # For questions that SHOULD be refused, "correct" means it DID refuse
        answered_correctly = refused
        retrieval_hit = None

    results.append({
        "question": question,
        "answer": answer,
        "refused": refused,
        "expected_to_refuse": not should_answer,
        "answered_correctly": answered_correctly,
        "retrieval_hit": retrieval_hit,
        "retrieved_pages": retrieved_pages,
    })

    status = "PASS" if answered_correctly and (retrieval_hit is not False) else "FAIL"
    print(f"[{i}] {status} -- \"{question}\"")
    print(f"    Answer: {answer[:100]}{'...' if len(answer) > 100 else ''}")
    if expected_page is not None:
        print(f"    Expected page {expected_page} in retrieved pages {retrieved_pages}: {retrieval_hit}")
    print()

# ---------- summary ----------
total = len(results)
passed = sum(1 for r in results if r["answered_correctly"] and r["retrieval_hit"] is not False)
retrieval_cases = [r for r in results if r["retrieval_hit"] is not None]
retrieval_hits = sum(1 for r in retrieval_cases if r["retrieval_hit"])

print("=" * 40)
print(f"Overall: {passed}/{total} passed")
if retrieval_cases:
    print(f"Retrieval accuracy: {retrieval_hits}/{len(retrieval_cases)} "
          f"({100 * retrieval_hits / len(retrieval_cases):.0f}%)")
