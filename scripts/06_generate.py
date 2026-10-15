"""
06_generate.py
--------------
Full RAG pipeline:
1. Take a question
2. Embed it, search Qdrant for top-K relevant chunks
3. Send question + chunks to an LLM (Groq)
4. LLM writes an answer with citations

Usage:
    python scripts/06_generate.py "your question"
    python scripts/06_generate.py   # uses default questions
"""

import os
import sys
from dotenv import load_dotenv
from groq import Groq
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

# ============ CONFIG ============
COLLECTION_NAME = "college_rules"
MODEL_NAME = "BAAI/bge-small-en-v1.5"
QDRANT_URL = "http://localhost:6333"
TOP_K = 5
LLM_MODEL = "openai/gpt-oss-120b"

DEFAULT_QUESTIONS = [
    "What is the minimum CGPA required to graduate?",
    "What happens if I fail a course?",
    "Can I take a break from my studies?",
    "How do I calculate my CPI?",
]

SYSTEM_PROMPT = """You are a helpful assistant answering questions about IIT Bombay's academic rules and regulations.

RULES FOR YOUR ANSWER:
1. Answer ONLY using the provided context passages below.
2. If the context doesn't contain the answer, say: "I don't have enough information in the provided documents to answer that."
3. Cite page numbers in the format [pages X-Y] after each fact you state.
4. Be concise — 2 to 4 sentences unless the question requires more.
5. Do not make up information. If unsure, say so.
"""


# ============ CORE FUNCTIONS ============
def retrieve(client, model, question: str, top_k: int = TOP_K):
    """Embed the question, retrieve top-K chunks from Qdrant."""
    q_emb = model.encode(question, normalize_embeddings=True).tolist()
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=q_emb,
        limit=top_k,
        with_payload=True,
    )
    return results.points


def build_context(hits) -> str:
    """Format retrieved chunks into a single string for the LLM."""
    parts = []
    for i, hit in enumerate(hits, start=1):
        pages = hit.payload.get("pages", [])
        text = hit.payload.get("text", "").strip()
        parts.append(f"--- Passage {i} [pages {pages}] (relevance {hit.score:.2f}) ---\n{text}")
    return "\n\n".join(parts)


def generate_answer(groq_client, question: str, context: str) -> str:
    """Ask the LLM to answer using only the given context."""
    user_message = f"""Question: {question}

Context passages:
{context}

Answer using ONLY the passages above. Cite pages in [pages X-Y] format."""

    response = groq_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.1,  # low = focused, factual answers
        max_tokens=500,
    )
    return response.choices[0].message.content


# ============ MAIN ============
def main():
    # Load API key from .env
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("❌ GROQ_API_KEY not found in .env")
        sys.exit(1)

    print("Initializing components...")
    qdrant = QdrantClient(url=QDRANT_URL)
    model = SentenceTransformer(MODEL_NAME)
    groq_client = Groq(api_key=api_key)

    questions = [" ".join(sys.argv[1:])] if len(sys.argv) > 1 else DEFAULT_QUESTIONS

    for question in questions:
        print("\n" + "=" * 78)
        print(f"Q: {question}")
        print("=" * 78)

        # 1. Retrieve
        hits = retrieve(qdrant, model, question)

        # 2. Build context
        context = build_context(hits)

        # 3. Generate
        answer = generate_answer(groq_client, question, context)

        # 4. Print answer + sources
        print("\nANSWER:")
        print(answer)
        print("\nSOURCES:")
        for i, hit in enumerate(hits, start=1):
            pages = hit.payload.get("pages", [])
            print(f"  [{i}] pages={pages}  (relevance {hit.score:.3f})")


if __name__ == "__main__":
    main()