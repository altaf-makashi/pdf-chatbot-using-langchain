"""
STEP 3: Percentage-based evaluation.   Usage: python evaluate.py

Reads test_questions.json and reports:
  - Answer correctness : judge model says CORRECT vs your reference (in-scope only)
  - Refusal accuracy   : out-of-scope questions correctly refused
  - Retrieval hit rate : expected page found in retrieved chunks (only if expected_page is set)
  - Overall pass rate  : correct answers + correct refusals / all questions

Also saves evaluation_details.csv so you can read every answer yourself.

Needs:  ollama pull llama3.1   (the judge; do NOT use a 1B model as judge)
"""

import json

import pandas as pd
from langchain_ollama import ChatOllama

from query import build_chain

JUDGE_MODEL = "llama3.1"

judge = ChatOllama(model=JUDGE_MODEL, temperature=0)

JUDGE_PROMPT = """You are grading a chatbot answer against a reference answer.

Question: {q}
Reference answer: {ref}
Chatbot answer: {ans}

The chatbot answer is CORRECT only if its key facts (dates, names, places, numbers) agree with
the reference and it contains no contradicting facts. Extra detail is fine if it does not
contradict the reference. If it answers a different question, or says it does not know,
it is INCORRECT.

Reply with exactly one word: CORRECT or INCORRECT."""


def is_refusal(text):
    t = text.lower()
    return "i don't know" in t or "don't know based on" in t


def judge_correct(q, ref, ans):
    out = judge.invoke(JUDGE_PROMPT.format(q=q, ref=ref, ans=ans)).content.strip().upper()
    return out.startswith("CORRECT")


def pct(a, b):
    return f"{a}/{b} = {100 * a / b:.1f}%" if b else "n/a (no cases)"


with open("test_questions.json", encoding="utf-8") as f:
    cases = json.load(f)

chain, retriever = build_chain()
rows = []

for i, c in enumerate(cases, 1):
    q = c["question"]
    answer = chain.invoke(q)
    pages = [d.metadata.get("page") for d in retriever.invoke(q)]
    refused = is_refusal(answer)

    if c.get("should_answer", True):
        kind = "in-scope"
        passed = (not refused) and judge_correct(q, c["reference"], answer)
    else:
        kind = "out-of-scope"
        passed = refused

    hit = None
    if c.get("expected_page") is not None:
        hit = c["expected_page"] in pages

    rows.append({"type": kind, "question": q, "answer": answer, "passed": passed,
                 "retrieval_hit": hit, "retrieved_pages": pages})
    print(f"[{i}/{len(cases)}] {'PASS' if passed else 'FAIL'} ({kind}) {q}\n    -> {answer[:150]}")

df = pd.DataFrame(rows)
df.to_csv("evaluation_details.csv", index=False)

inscope = df[df.type == "in-scope"]
outscope = df[df.type == "out-of-scope"]
page_cases = df[df.retrieval_hit.notna()]

print("\n" + "=" * 50)
print("RESULTS")
print("=" * 50)
print(f"Answer correctness : {pct(int(inscope.passed.sum()), len(inscope))}")
print(f"Refusal accuracy   : {pct(int(outscope.passed.sum()), len(outscope))}")
print(f"Retrieval hit rate : {pct(int(page_cases.retrieval_hit.sum()), len(page_cases))}")
print(f"Overall pass rate  : {pct(int(df.passed.sum()), len(df))}")
print("\nDetails saved to evaluation_details.csv")
