"""
02_chunk.py
-----------
Takes the extracted text and splits it into smaller chunks.

Why: We can't embed a whole document as one vector (meaning gets diluted).
We split into ~500 token chunks with overlap so nothing is lost at boundaries.

Input:  data/extracted/iitb_rules.txt
Output: data/chunks/chunks.jsonl   (one chunk per line, JSON format)
"""

import json
import re
from pathlib import Path
import tiktoken

# ============ CONFIG ============
CHUNK_SIZE = 500        # target tokens per chunk
CHUNK_OVERLAP = 50      # overlap between chunks (in tokens)
ENCODING_NAME = "cl100k_base"  # OpenAI's tokenizer

INPUT_FILE = Path("data/extracted/iitb_rules.txt")
OUTPUT_FILE = Path("data/chunks/chunks.jsonl")


# ============ HELPERS ============
def load_text(path: Path) -> str:
    """Read the extracted .txt file."""
    return path.read_text(encoding="utf-8")


def split_by_pages(text: str):
    """
    Our extractor inserted '===== Page N =====' markers.
    Split on those so we can track which page each chunk came from.
    """
    # Split on the page marker pattern
    parts = re.split(r"===== Page (\d+) =====", text)

    # parts looks like: ['', '1', 'text of page 1', '2', 'text of page 2', ...]
    # Pair them up
    pages = []
    for i in range(1, len(parts), 2):
        page_num = int(parts[i])
        page_text = parts[i + 1] if i + 1 < len(parts) else ""
        pages.append((page_num, page_text.strip()))

    return pages


def split_text_into_segments(text: str):
    """
    Break a page's text into paragraph-level segments.
    We try to keep chunks semantically coherent by starting with paragraphs.
    """
    # Split on double newlines (paragraph breaks) first
    paragraphs = re.split(r"\n\s*\n", text)
    return [p.strip() for p in paragraphs if p.strip()]


def chunk_by_tokens(pages, encoding):
    """
    Build chunks targeting CHUNK_SIZE tokens with CHUNK_OVERLAP overlap.
    Each chunk remembers which page(s) it came from (for citations).
    """
    chunks = []
    buffer_tokens = []
    buffer_pages = []  # track pages of tokens in the buffer

    def emit_chunk():
        """Save current buffer as a chunk and start a fresh buffer with overlap."""
        if not buffer_tokens:
            return

        text = encoding.decode(buffer_tokens)

        # Record pages covered by this chunk
        pages_in_chunk = sorted(set(buffer_pages))

        chunks.append({
            "text": text,
            "pages": pages_in_chunk,
            "token_count": len(buffer_tokens),
        })

        # Keep last CHUNK_OVERLAP tokens as start of next buffer
        if len(buffer_tokens) > CHUNK_OVERLAP:
            buffer_tokens[:] = buffer_tokens[-CHUNK_OVERLAP:]
            buffer_pages[:] = buffer_pages[-CHUNK_OVERLAP:]
        else:
            buffer_tokens.clear()
            buffer_pages.clear()

    for page_num, page_text in pages:
        segments = split_text_into_segments(page_text)

        for segment in segments:
            seg_tokens = encoding.encode(segment)
            seg_pages = [page_num] * len(seg_tokens)

            # Would adding this segment exceed the target size?
            if len(buffer_tokens) + len(seg_tokens) > CHUNK_SIZE and buffer_tokens:
                emit_chunk()

            buffer_tokens.extend(seg_tokens)
            buffer_pages.extend(seg_pages)

    # Don't forget the last partial chunk
    if buffer_tokens:
        text = encoding.decode(buffer_tokens)
        chunks.append({
            "text": text,
            "pages": sorted(set(buffer_pages)),
            "token_count": len(buffer_tokens),
        })

    return chunks


# ============ MAIN ============
def main():
    print("Loading text...")
    text = load_text(INPUT_FILE)
    print(f"  Total characters: {len(text):,}")

    pages = split_by_pages(text)
    print(f"  Pages detected: {len(pages)}")

    encoding = tiktoken.get_encoding(ENCODING_NAME)

    print(f"\nChunking with target size={CHUNK_SIZE} tokens, overlap={CHUNK_OVERLAP}...")
    chunks = chunk_by_tokens(pages, encoding)
    print(f"  Total chunks created: {len(chunks)}")

    # Stats
    sizes = [c["token_count"] for c in chunks]
    print(f"\nChunk size stats:")
    print(f"  Min:    {min(sizes)} tokens")
    print(f"  Max:    {max(sizes)} tokens")
    print(f"  Mean:   {sum(sizes) / len(sizes):.0f} tokens")
    print(f"  Median: {sorted(sizes)[len(sizes) // 2]} tokens")

    # Save as JSONL (one JSON object per line)
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        for i, chunk in enumerate(chunks):
            chunk["chunk_id"] = i
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    print(f"\nSaved {len(chunks)} chunks to {OUTPUT_FILE}")

    # Show a sample chunk so we can sanity check
    print("\n" + "=" * 60)
    print("SAMPLE CHUNK (chunk_id=0):")
    print("=" * 60)
    print(chunks[0]["text"][:600])
    print("...")
    print(f"\nPages referenced: {chunks[0]['pages']}")
    print(f"Token count: {chunks[0]['token_count']}")


if __name__ == "__main__":
    main()