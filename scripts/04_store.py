"""
04_store.py
-----------
Loads chunks + embeddings from JSONL into Qdrant, so we can query them fast.

Input:  data/chunks/chunks_with_embeddings.jsonl
Output: Qdrant collection "college_rules" (persisted in ./qdrant_storage/)
"""

import json
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# ============ CONFIG ============
INPUT_FILE = Path("data/chunks/chunks_with_embeddings.jsonl")
COLLECTION_NAME = "college_rules"
VECTOR_SIZE = 384           # BGE-small produces 384-dim vectors
QDRANT_URL = "http://localhost:6333"


# ============ MAIN ============
def main():
    print(f"Connecting to Qdrant at {QDRANT_URL}...")
    client = QdrantClient(url=QDRANT_URL)

    # Verify connection
    collections = client.get_collections().collections
    print(f"  Connected. Existing collections: {[c.name for c in collections]}\n")

    # -------- Create (or recreate) collection --------
    # If collection already exists, delete it. Why? So we can rerun this script
    # safely during development without worrying about duplicate points.
    if client.collection_exists(COLLECTION_NAME):
        print(f"Collection '{COLLECTION_NAME}' already exists. Recreating...")
        client.delete_collection(COLLECTION_NAME)

    print(f"Creating collection '{COLLECTION_NAME}'...")
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )
    print(f"  Collection created. Vector size: {VECTOR_SIZE}, distance: COSINE\n")

    # -------- Load chunks --------
    print(f"Reading chunks from {INPUT_FILE}...")
    chunks = []
    with INPUT_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    print(f"  Loaded {len(chunks)} chunks\n")

    # -------- Upload to Qdrant --------
    print(f"Uploading {len(chunks)} chunks to Qdrant...")
    points = []
    for chunk in chunks:
        points.append(
            PointStruct(
                id=chunk["chunk_id"],                       # unique ID
                vector=chunk["embedding"],                  # 384-dim vector
                payload={                                    # everything else
                    "text": chunk["text"],
                    "pages": chunk["pages"],
                    "token_count": chunk["token_count"],
                },
            )
        )

    client.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"  Uploaded {len(points)} points\n")

    # -------- Verify --------
    info = client.get_collection(COLLECTION_NAME)
    print(f"Collection '{COLLECTION_NAME}' now has {info.points_count} points.")
    print(f"Vector config: {info.config.params.vectors}")


if __name__ == "__main__":
    main()