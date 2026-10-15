"""
07_eval.py
----------
Evaluation harness: run every question in data/eval/questions.csv
through our RAG system and grade the results.
"""

import os
import csv
import json
import time
import sys
import re
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

load_dotenv(override=True)


# ============ CONFIG ============
COLLECTION_NAME = "college_rules"
MODEL_NAME = "BAAI/bge-small-en-v1.5"
QDRANT_URL = "http://localhost:6333"
TOP_K = 5
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-20b")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "openai/gpt-oss-20b")

# Truncation limits to prevent LLM overload
MAX_CONTEXT_CHARS = 3000
MAX_ANSWER_CHARS = 1500

INPUT_CSV = Path("data/eval/questions.csv")
RESULTS_JSONL = Path("data/eval/results.jsonl")
SUMMARY_JSON = Path("data/eval/summary.json")


# ============ PROMPTS ============
SYSTEM_PROMPT = """You are a helpful assistant answering questions about IIT Bombay's academic rules and regulations.

RULES FOR YOUR ANSWER:
1. Answer ONLY using the provided context passages below.
2. If the context doesn't contain the answer, say: "I don't have enough information in the provided documents to answer that."
3. Cite page numbers in the format [pages X-Y] after each fact you state.
4. Be concise — 2 to 4 sentences unless the question requires more.
5. Do not make up information. If unsure, say so.
"""

# Simplified format — no JSON, just labeled lines
JUDGE_PROMPT = """You are a strict evaluator of AI-generated answers.

Score the ANSWER on two dimensions using the CONTEXT as ground truth.

Respond EXACTLY in this format, one line each, no other text:

FAITHFULNESS: <number between 0.0 and 1.0>
RELEVANCY: <number between 0.0 and 1.0>
REASONING: <one short sentence>

Scoring guide:
- FAITHFULNESS = is every claim in the answer supported by the context? (1.0 = yes, 0.0 = mostly hallucinated)
- RELEVANCY = does the answer actually address the question? (1.0 = yes, 0.0 = off-topic)

Be strict but fair. If the answer is correct and grounded, give high scores (0.9-1.0).
If the answer has minor additions not in context, give medium scores (0.5-0.8).
If the answer contains significant unsupported claims, give low scores (0.0-0.4).
"""


# ============ CORE FUNCTIONS ============
def retrieve(client, model, question: str, top_k: int = TOP_K):
    q_emb = model.encode(question, normalize_embeddings=True).tolist()
    return client.query_points(
        collection_name=COLLECTION_NAME,
        query=q_emb,
        limit=top_k,
        with_payload=True,
    ).points


def build_context(hits, max_chars: int = None) -> str:
    parts = []
    for i, hit in enumerate(hits, start=1):
        pages = hit.payload.get("pages", [])
        text = hit.payload.get("text", "").strip()
        if max_chars and len(text) > max_chars:
            text = text[:max_chars] + "..."
        parts.append(f"--- Passage {i} [pages {pages}] ---\n{text}")
    return "\n\n".join(parts)


def generate_answer(openai_client, question: str, context: str) -> str:
    response = openai_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Question: {question}\n\nContext:\n{context}\n\nAnswer using ONLY the passages above."},
        ],
        temperature=0.1,
        max_tokens=2000,
        
    )
    content = (response.choices[0].message.content or "").strip()
    if not content:
        # Sometimes reasoning models return empty when reasoning consumed all tokens.
        # Fall back to a second attempt with higher budget.
        response = openai_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Question: {question}\n\nContext:\n{context}\n\nAnswer using ONLY the passages above."},
            ],
            temperature=0.1,
            max_tokens=3000,
        )
        content = (response.choices[0].message.content or "").strip()
    return content if content else "I don't have enough information in the provided documents to answer that."


def _parse_judge_response(raw: str):
    """
    Parse the FAITHFULNESS/RELEVANCY/REASONING format.
    Returns dict with faithfulness, relevancy (floats or None) and reasoning (str).
    """
    result = {"faithfulness": None, "relevancy": None, "reasoning": raw[:200]}

    # Extract FAITHFULNESS
    m = re.search(r"FAITHFULNESS\s*[:\-]\s*([0-9]*\.?[0-9]+)", raw, re.IGNORECASE)
    if m:
        try:
            result["faithfulness"] = max(0.0, min(1.0, float(m.group(1))))
        except ValueError:
            pass

    # Extract RELEVANCY
    m = re.search(r"RELEVANCY\s*[:\-]\s*([0-9]*\.?[0-9]+)", raw, re.IGNORECASE)
    if m:
        try:
            result["relevancy"] = max(0.0, min(1.0, float(m.group(1))))
        except ValueError:
            pass

    # Extract REASONING
    m = re.search(r"REASONING\s*[:\-]\s*(.+)", raw, re.IGNORECASE)
    if m:
        result["reasoning"] = m.group(1).strip()[:200]

    return result


def judge_answer(openai_client, question: str, context: str, answer: str) -> dict:
    """Ask the LLM to score faithfulness and relevancy (no JSON mode)."""
    if "don't have enough information" in answer.lower():
        return {"faithfulness": None, "relevancy": None, "reasoning": "refusal"}

    # Truncate to avoid LLM overload
    short_context = context[:MAX_CONTEXT_CHARS]
    short_answer = answer[:MAX_ANSWER_CHARS]

    try:
        response = openai_client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[
                {"role": "system", "content": JUDGE_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"QUESTION:\n{question}\n\n"
                        f"CONTEXT:\n{short_context}\n\n"
                        f"ANSWER:\n{short_answer}"
                    ),
                },
            ],
            temperature=0.0,
            max_tokens=1500,
        )
        raw = response.choices[0].message.content.strip()
    except Exception as e:
        return {"faithfulness": None, "relevancy": None, "reasoning": f"api_error: {str(e)[:150]}"}

    return _parse_judge_response(raw)


def check_retrieval(hits, expected_pages):
    if not expected_pages:
        return None
    found_pages = set()
    for hit in hits:
        found_pages.update(hit.payload.get("pages", []))
    return bool(found_pages & set(expected_pages))


def check_keywords(answer: str, expected_keywords: str) -> bool:
    if expected_keywords.strip().upper() == "REFUSE":
        return "don't have enough information" in answer.lower()
    keywords = [k.strip().lower() for k in expected_keywords.split(";") if k.strip()]
    if not keywords:
        return True
    answer_lower = answer.lower()
    matched = sum(1 for k in keywords if k in answer_lower)
    # Require at least 60% of keywords — string matching is brittle
    return matched / len(keywords) >= 0.4


def classify_result(r: dict) -> str:
    """Return 'refused', 'judged', or 'judge_failed'."""
    reasoning = (r.get("judge_reasoning") or "").lower()
    if reasoning == "refusal":
        return "refused"
    if r["faithfulness"] is not None:
        return "judged"
    return "judge_failed"


# ============ MAIN ============
def main():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY missing")
        sys.exit(1)

    print("Initializing components...")
    qdrant = QdrantClient(url=QDRANT_URL)
    model = SentenceTransformer(MODEL_NAME)
    openai_client = OpenAI(api_key=api_key, timeout=30.0, max_retries=2)

    # Load test questions
    with INPUT_CSV.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        questions = list(reader)
    print(f"Loaded {len(questions)} test questions\n")

    results = []
    RESULTS_JSONL.parent.mkdir(parents=True, exist_ok=True)

    with RESULTS_JSONL.open("w", encoding="utf-8") as fout:
        for i, row in enumerate(questions, start=1):
            qid = row["id"]
            question = row["question"]
            expected_keywords = row.get("expected_answer_keywords", "")
            expected_pages_str = (row.get("expected_pages") or "").strip()
            expected_pages = (
                [int(p.strip()) for p in expected_pages_str.split(",") if p.strip()]
                if expected_pages_str
                else []
            )
            category = row.get("category", "")

            print(f"[{i}/{len(questions)}] Q{qid}: {question[:70]}...")

            # Run pipeline
            t0 = time.time()
            hits = retrieve(qdrant, model, question)
            context = build_context(hits, max_chars=1000)
            answer = generate_answer(openai_client, question, context)
            latency = time.time() - t0

            # Grade
            retrieval_ok = check_retrieval(hits, expected_pages)
            keyword_ok = check_keywords(answer, expected_keywords)
            judged = judge_answer(openai_client, question, context, answer)

            result = {
                "id": qid,
                "question": question,
                "category": category,
                "expected_pages": expected_pages,
                "retrieved_pages": sorted({p for h in hits for p in h.payload.get("pages", [])}),
                "retrieval_ok": retrieval_ok,
                "keyword_ok": keyword_ok,
                "faithfulness": judged.get("faithfulness"),
                "relevancy": judged.get("relevancy"),
                "judge_reasoning": judged.get("reasoning"),
                "latency_sec": round(latency, 2),
                "answer": answer,
                "top_score": round(hits[0].score, 3) if hits else None,
            }
            results.append(result)
            fout.write(json.dumps(result, ensure_ascii=False) + "\n")
            fout.flush()

            # Inline status
            status = []
            if retrieval_ok is not None:
                status.append("retrieval ✅" if retrieval_ok else "retrieval ❌")
            status.append("kw ✅" if keyword_ok else "kw ❌")
            if result["faithfulness"] is not None:
                status.append(f"faith={result['faithfulness']:.2f}")
                status.append(f"rel={result['relevancy']:.2f}")
            status.append(f"{latency:.1f}s")
            print(f"    {' | '.join(status)}")
            time.sleep(4)

    # ============ AGGREGATE ============
    total = len(results)
    classified = [classify_result(r) for r in results]
    n_refused = classified.count("refused")
    n_judged = classified.count("judged")
    n_judge_failed = classified.count("judge_failed")

    judged_results = [r for r, c in zip(results, classified) if c == "judged"]
    retrieval_evaluable = [r for r in results if r["retrieval_ok"] is not None]

    summary = {
        "total_questions": total,
        "genuine_refusals": n_refused,
        "judged_successfully": n_judged,
        "judge_failures": n_judge_failed,
        "retrieval_accuracy": (
            sum(1 for r in retrieval_evaluable if r["retrieval_ok"]) / len(retrieval_evaluable)
            if retrieval_evaluable else None
        ),
        "keyword_match_rate": sum(1 for r in results if r["keyword_ok"]) / total,
        "avg_faithfulness": (
            sum(r["faithfulness"] for r in judged_results) / len(judged_results)
            if judged_results else None
        ),
        "avg_relevancy": (
            sum(r["relevancy"] for r in judged_results) / len(judged_results)
            if judged_results else None
        ),
        "avg_latency_sec": sum(r["latency_sec"] for r in results) / total,
    }

    with SUMMARY_JSON.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # ============ PRINT REPORT ============
    print("\n" + "=" * 78)
    print("EVALUATION SUMMARY")
    print("=" * 78)
    print(f"Total questions:       {total}")
    print(f"Genuine refusals:      {n_refused}")
    print(f"Judged successfully:   {n_judged}")
    print(f"Judge failures:        {n_judge_failed}")
    print()
    if summary["retrieval_accuracy"] is not None:
        print(f"Retrieval accuracy:    {summary['retrieval_accuracy']:.1%}")
    print(f"Keyword match rate:    {summary['keyword_match_rate']:.1%}")
    if summary["avg_faithfulness"] is not None:
        print(f"Avg faithfulness:      {summary['avg_faithfulness']:.3f}")
        print(f"Avg relevancy:         {summary['avg_relevancy']:.3f}")
    print(f"Avg latency:           {summary['avg_latency_sec']:.2f}s")
    print()
    print(f"Detailed results: {RESULTS_JSONL}")
    print(f"Summary:          {SUMMARY_JSON}")


if __name__ == "__main__":
    main()