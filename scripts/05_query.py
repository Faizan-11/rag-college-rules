"""
05_query.py
-----------
Query the Qdrant collection with a natural language question.
Shows the top-K most semantically similar chunks.

Usage:
    python scripts/05_query.py "your question here"
    python scripts/05_query.py   # uses default questions
"""

import sys
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

# ============ CONFIG ============
COLLECTION_NAME = "college_rules"
MODEL_NAME = "BAAI/bge-small-en-v1.5"
QDRANT_URL = "http://localhost:6333"
TOP_K = 5

DEFAULT_QUESTIONS = [
    "What is the minimum CGPA required to graduate?",
    "How many credits do I need to complete my degree?",
    "What happens if I fail a course?",
    "Can I take a break from my studies?",
    "What is the weather in Tokyo?",  # control — should score low
]


# ============ MAIN ============
def search(client, model, question: str, top_k: int = TOP_K):
    """Embed the question, search Qdrant, return top-K results."""
    q_emb = model.encode(question, normalize_embeddings=True).tolist()

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=q_emb,
        limit=top_k,
        with_payload=True,
        with_vectors=False,
    )

    return results.points


def main():
    print(f"Connecting to Qdrant at {QDRANT_URL}...")
    client = QdrantClient(url=QDRANT_URL)

    print(f"Loading embedding model: {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)

    # Get question(s) from CLI args, or use defaults
    if len(sys.argv) > 1:
        questions = [" ".join(sys.argv[1:])]
    else:
        questions = DEFAULT_QUESTIONS

    for question in questions:
        print("\n" + "=" * 78)
        print(f"Q: {question}")
        print("=" * 78)

        results = search(client, model, question)

        for rank, hit in enumerate(results, start=1):
            score = hit.score
            pages = hit.payload.get("pages", [])
            text = hit.payload.get("text", "").strip().replace("\n", " ")

            # Show a preview of the chunk
            preview = text[:280] + ("..." if len(text) > 280 else "")

            print(f"\n  [{rank}] score={score:.3f}  pages={pages}")
            print(f"      {preview}")


if __name__ == "__main__":
    main()