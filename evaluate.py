"""
evaluate.py — CI-gated RAG evaluation pipeline
===============================================
Runs a fixed golden test suite against the retrieval + generation stack
and asserts that three core RAG metrics stay above defined thresholds.
This file is executed by .github/workflows/rag_eval.yml on every PR.

Metrics evaluated
-----------------
1. Citation rate      — fraction of retrieved passages cited in the answer
2. Context precision  — fraction of retrieved passages actually relevant to
                        the question (scored by LLM-as-judge)
3. Faithfulness       — whether the answer contradicts the source passages
                        (scored by LLM-as-judge)

Thresholds (edit METRIC_THRESHOLDS to tighten as the system matures)
----------------------------------------------------------------------
These gate merges: a PR that drops any metric below its threshold fails CI.
"""

import os
import re
import sys
import json
import textwrap
from dataclasses import dataclass, field

from groq import Groq

# ── Import core RAG logic from the app ──────────────────────────────────────
from app import NewsResearchTool

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
METRIC_THRESHOLDS = {
    "citation_rate":     0.50,   # ≥50 % of passages must be cited
    "context_precision": 0.60,   # ≥60 % of passages judged relevant
    "faithfulness":      0.45,   # ≥45 % of claims judged faithful to context
}

# Golden test cases: (question, article_urls, must_contain_keywords)
# URLs chosen to be publicly accessible without paywalls.
GOLDEN_TESTS = [
    {
        "question": "What is Python programming language used for?",
        "urls": [
            "https://en.wikipedia.org/wiki/Python_(programming_language)",
        ],
        "must_contain": ["programming", "language"],
    },
    {
        "question": "Who founded Wikipedia?",
        "urls": [
            "https://en.wikipedia.org/wiki/Wikipedia",
        ],
        "must_contain": ["Jimmy Wales", "Larry Sanger"],
    },
    {
        "question": "What is machine learning?",
        "urls": [
            "https://en.wikipedia.org/wiki/Machine_learning",
        ],
        "must_contain": ["algorithm", "data"],
    },
]


# ---------------------------------------------------------------------------
# LLM-as-judge helpers
# ---------------------------------------------------------------------------
def _judge_call(client: Groq, prompt: str) -> str:
    """Single LLM call for evaluation judgement. Uses a small fast model."""
    resp = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a strict evaluator. "
                    "Respond ONLY with valid JSON and nothing else."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        model="llama-3.3-70b-versatile",
        temperature=0.0,
        max_tokens=300,
    )
    return resp.choices[0].message.content.strip()


def score_context_precision(
    client: Groq, question: str, passages: list[str]
) -> float:
    """
    LLM-as-judge: for each passage, ask whether it is relevant to the question.
    Returns fraction of passages judged relevant.
    Context precision = relevant_passages / total_passages.
    """
    if not passages:
        return 0.0

    relevant = 0
    for i, passage in enumerate(passages):
        prompt = textwrap.dedent(f"""
            Question: {question}

            Passage [P{i+1}]: {passage[:800]}

            Is this passage relevant to answering the question?
            Respond with exactly: {{"relevant": true}} or {{"relevant": false}}
        """)
        try:
            raw = _judge_call(client, prompt)
            result = json.loads(raw)
            if result.get("relevant") is True:
                relevant += 1
        except (json.JSONDecodeError, KeyError):
            pass  # conservative: don't count malformed responses as relevant

    return relevant / len(passages)


def score_faithfulness(
    client: Groq, question: str, answer: str, passages: list[str]
) -> float:
    """
    LLM-as-judge: decompose the answer into claims, check each against context.
    Returns fraction of claims that are faithful (not contradicted by) the passages.
    """
    context = "\n\n".join(f"[P{i+1}] {p[:600]}" for i, p in enumerate(passages))

    prompt = textwrap.dedent(f"""
        Context passages:
        {context}

        Answer to evaluate:
        {answer[:1000]}

        Identify each factual claim in the answer and judge whether it is
        supported by (or at least not contradicted by) the context passages.

        Respond with exactly this JSON:
        {{
            "total_claims": <integer>,
            "faithful_claims": <integer>
        }}
    """)
    try:
        raw = _judge_call(client, prompt)
        result = json.loads(raw)
        total = result.get("total_claims", 0)
        faithful = result.get("faithful_claims", 0)
        return faithful / max(total, 1)
    except (json.JSONDecodeError, KeyError, ZeroDivisionError):
        return 0.0


# ---------------------------------------------------------------------------
# Per-test evaluation
# ---------------------------------------------------------------------------
@dataclass
class TestResult:
    question: str
    citation_rate: float
    context_precision: float
    faithfulness: float
    keyword_hit: bool
    passed: bool
    notes: list[str] = field(default_factory=list)


def run_single_test(
    tool: NewsResearchTool,
    groq_client: Groq,
    test: dict,
    top_k: int = 3,
) -> TestResult:
    notes = []

    # Scrape + index
    documents = tool.process_urls(test["urls"])
    if not documents:
        return TestResult(
            question=test["question"],
            citation_rate=0.0,
            context_precision=0.0,
            faithfulness=0.0,
            keyword_hit=False,
            passed=False,
            notes=["Scraping failed — no content extracted"],
        )

    faiss_index, bm25_index, all_chunks, chunk_sources = tool.build_chunk_store(
        documents, test["urls"]
    )

    # Hybrid retrieval + reranking
    retrieved_chunks, _ = tool.hybrid_retrieve(
        faiss_index, bm25_index, all_chunks, chunk_sources,
        test["question"], top_k=top_k
    )

    if not retrieved_chunks:
        return TestResult(
            question=test["question"],
            citation_rate=0.0,
            context_precision=0.0,
            faithfulness=0.0,
            keyword_hit=False,
            passed=False,
            notes=["Retrieval returned no chunks"],
        )

    # Generate answer
    answer, citation_rate = tool.generate_response(retrieved_chunks, test["question"])

    # LLM-as-judge scores
    context_precision = score_context_precision(
        groq_client, test["question"], retrieved_chunks
    )
    faithfulness = score_faithfulness(
        groq_client, test["question"], answer, retrieved_chunks
    )

    # Keyword hit check (basic answer quality sanity check)
    answer_lower = answer.lower()
    keyword_hit = all(kw.lower() in answer_lower for kw in test.get("must_contain", []))
    if not keyword_hit:
        notes.append(
            f"Missing expected keywords: {test.get('must_contain', [])}"
        )

    # Gate check
    passed = (
        citation_rate     >= METRIC_THRESHOLDS["citation_rate"]
        and context_precision >= METRIC_THRESHOLDS["context_precision"]
        and faithfulness      >= METRIC_THRESHOLDS["faithfulness"]
    )

    return TestResult(
        question=test["question"],
        citation_rate=citation_rate,
        context_precision=context_precision,
        faithfulness=faithfulness,
        keyword_hit=keyword_hit,
        passed=passed,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def main():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("ERROR: GROQ_API_KEY environment variable not set.")
        sys.exit(1)

    groq_client = Groq(api_key=api_key)
    tool = NewsResearchTool()
    tool.client = groq_client

    results: list[TestResult] = []
    print(f"\n{'='*60}")
    print("RAG Evaluation Pipeline")
    print(f"{'='*60}\n")

    for i, test in enumerate(GOLDEN_TESTS, 1):
        print(f"[{i}/{len(GOLDEN_TESTS)}] {test['question']}")
        result = run_single_test(tool, groq_client, test)
        results.append(result)

        status = "✅ PASS" if result.passed else "❌ FAIL"
        print(f"  {status}")
        print(f"  Citation rate     : {result.citation_rate:.2%}")
        print(f"  Context precision : {result.context_precision:.2%}")
        print(f"  Faithfulness      : {result.faithfulness:.2%}")
        print(f"  Keyword hit       : {result.keyword_hit}")
        for note in result.notes:
            print(f"  ⚠  {note}")
        print()

    # Aggregate
    n = len(results)
    avg_citation    = sum(r.citation_rate     for r in results) / n
    avg_precision   = sum(r.context_precision for r in results) / n
    avg_faithfulness = sum(r.faithfulness     for r in results) / n
    all_passed      = all(r.passed            for r in results)

    print(f"{'='*60}")
    print("Aggregate Results")
    print(f"{'='*60}")
    print(f"  Avg citation rate     : {avg_citation:.2%}  (threshold: {METRIC_THRESHOLDS['citation_rate']:.0%})")
    print(f"  Avg context precision : {avg_precision:.2%}  (threshold: {METRIC_THRESHOLDS['context_precision']:.0%})")
    print(f"  Avg faithfulness      : {avg_faithfulness:.2%}  (threshold: {METRIC_THRESHOLDS['faithfulness']:.0%})")
    print(f"\n  Overall: {'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")

    # Write machine-readable report for CI artifact upload
    report = {
        "thresholds": METRIC_THRESHOLDS,
        "averages": {
            "citation_rate": avg_citation,
            "context_precision": avg_precision,
            "faithfulness": avg_faithfulness,
        },
        "tests": [
            {
                "question": r.question,
                "citation_rate": r.citation_rate,
                "context_precision": r.context_precision,
                "faithfulness": r.faithfulness,
                "keyword_hit": r.keyword_hit,
                "passed": r.passed,
                "notes": r.notes,
            }
            for r in results
        ],
        "all_passed": all_passed,
    }
    with open("rag_eval_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print("\n  Report written to rag_eval_report.json")

    # Non-zero exit code fails the CI job
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
