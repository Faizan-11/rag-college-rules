"""
01_extract.py
-------------
Reads every PDF in data/raw/ and saves its text to data/extracted/.

Why: LLMs can't read PDFs directly. Step 1 of any RAG system is
turning PDFs into clean, plain text we can work with later.
"""

import pymupdf  # modern import name (older code uses `fitz`)
from pathlib import Path

# Where things live
RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/extracted")


def extract_text_from_pdf(pdf_path: Path) -> str:
    """
    Open a PDF and return all its text, page by page.
    We insert a marker line before each page so we can cite pages later.
    """
    doc = pymupdf.open(pdf_path)
    pages = []

    for page_num, page in enumerate(doc, start=1):
        text = page.get_text()
        pages.append(f"\n\n===== Page {page_num} =====\n\n{text}")

    return "".join(pages)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(RAW_DIR.glob("*.pdf"))
    print(f"Found {len(pdf_files)} PDF(s) in {RAW_DIR}")

    if not pdf_files:
        print("No PDFs found. Put your PDF in data/raw/ first.")
        return

    for pdf_path in pdf_files:
        print(f"\nProcessing: {pdf_path.name}")
        text = extract_text_from_pdf(pdf_path)

        out_path = OUT_DIR / f"{pdf_path.stem}.txt"
        out_path.write_text(text, encoding="utf-8")

        print(f"  → Saved {len(text):,} characters to {out_path}")


if __name__ == "__main__":
    main()