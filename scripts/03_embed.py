"""
03_embed.py
-----------
Turns each text chunk into a vector (list of numbers) that captures its meaning.

Why: We can't search text by meaning directly. But we can search numbers.
Embeddings = the bridge from "text" to "searchable by meaning".

Input:  data/chunks/chunks.jsonl
Output: data/chunks/chunks_with_embeddings.jsonl
"""

import json
import time
from pathlib import Path
from sentence_transformers import SentenceTransformer

# ============ CONFIG ============
MODEL_NAME = "BAAI/bge-small-en-v1.5"   # 384-dim, fast, high quality
INPUT_FILE = Path("data/chunks/chunks.jsonl")
OUTPUT_FILE = Path("data/chunks/chunks_with_embeddings.jsonl")
BATCH_SIZE = 32


# ============ MAIN ============
def main():
    print(f"Loading embedding model: {MODEL_NAME}")
    print("(First run downloads ~130 MB. Subsequent runs are instant.)\n")
    model = SentenceTransformer(MODEL_NAME)

    print(f"Reading chunks from {INPUT_FILE}...")
    chunks = []
    with INPUT_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    print(f"  Loaded {len(chunks)} chunks\n")

    texts = [c["text"] for c in chunks]

    print(f"Embedding {len(texts)} chunks...")
    start = time.time()
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,  # so we can use dot product = cosine similarity
    )
    elapsed = time.time() - start

    print(f"\n  Done in {elapsed:.1f}s ({len(texts) / elapsed:.1f} chunks/sec)")
    print(f"  Embedding shape: {embeddings.shape}")
    print(f"  Vector dimension: {embeddings.shape[1]}\n")

    # Save: attach embedding to each chunk
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        for chunk, emb in zip(chunks, embeddings):
            chunk["embedding"] = emb.tolist()
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    print(f"Saved {len(chunks)} embedded chunks to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()