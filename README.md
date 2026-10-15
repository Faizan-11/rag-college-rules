# IITB Academic Rules Assistant

A **Retrieval-Augmented Generation (RAG)** system that answers questions about IIT Bombay's undergraduate academic rules, with citations to source pages.

**🔗 Live Demo:** [iitb-rag-demo.streamlit.app](https://iitb-rag-demo-aqedvxktwwyqwum34ufqbs.streamlit.app/)  
*First load may take 20–30 seconds while the free-tier app wakes up. Subsequent queries are instant.*

---

## What It Does

Ask a natural-language question like *"What happens if I fail a course?"* and the system:

1. Converts your question into a semantic embedding
2. Retrieves the 5 most relevant passages from the IITB rulebook
3. Uses GPT-4o-mini to generate a cited answer grounded *only* in those passages
4. Refuses to answer when the information isn't in the source document

**Try it yourself:** [Live demo link](https://iitb-rag-demo-aqedvxktwwyqwum34ufqbs.streamlit.app/)

---

## Architecture

```text
                    ┌─────────────────────────────┐
                    │   User Question             │
                    └──────────────┬──────────────┘
                                   ▼
                    ┌─────────────────────────────┐
                    │   BGE-small-en-v1.5         │
                    │   (Query Embedding)         │
                    └──────────────┬──────────────┘
                                   ▼
                    ┌─────────────────────────────┐
                    │   Qdrant Vector DB          │
                    │   (Semantic Retrieval)      │
                    │   → Top-5 Chunks            │
                    └──────────────┬──────────────┘
                                   ▼
                    ┌─────────────────────────────┐
                    │   GPT-4o-mini               │
                    │   (Grounded Answer +        │
                    │    Page Citations)          │
                    └──────────────┬──────────────┘
                                   ▼
                    ┌─────────────────────────────┐
                    │   Answer + Source Pages     │
                    └─────────────────────────────┘
```



### Pipeline Stages

| Stage | Script | What it does |
|---|---|---|
| 1 | `01_extract.py` | Extract text from source PDF (PyMuPDF) |
| 2 | `02_chunk.py` | Split into ~500-token chunks with overlap (tiktoken) |
| 3 | `03_embed.py` | Generate 384-dim embeddings (BAAI/bge-small-en-v1.5) |
| 4 | `04_store.py` | Index chunks in Qdrant vector DB |
| 5 | `05_query.py` | Semantic search test harness |
| 6 | `06_generate.py` | RAG pipeline: retrieve → generate with citations |
| 7 | `07_eval.py` | Evaluation harness with LLM-as-judge |

---

## Evaluation Results

Evaluated on a hand-labeled test set of **10 questions** drawn from the rulebook.

| Metric | Score |
|---|---|
| **Retrieval accuracy** | **100%** (10/10 found correct page ranges) |
| **Avg faithfulness** | **0.83** (answer grounded in source) |
| **Avg relevancy** | **0.93** (answer addresses the question) |
| **Avg latency** | **1.36s** per query (end-to-end) |
| **Genuine refusals** | 3/10 (correctly refused out-of-scope questions) |

**Baseline results:** `data/eval/summary_v1_baseline.json`

### Methodology

- **Retrieval accuracy:** Did any top-5 chunk come from the expected page range?
- **Faithfulness & Relevancy:** Scored by a second LLM (GPT-4o-mini as judge) on a 0–1 scale.
- **Refusal correctness:** Verified against questions whose answers are not in the source document.

---

## Known Limitations

Real engineering comes with honesty. Here's what this system *doesn't* do well yet:

1. **Q5 (re-examination) is a judge false negative.** The system produced a correct, cited answer from pages 29–30, but the automated judge scored it 0.00 due to judge context truncation. Fix: increase judge context window or use a stronger judge model.

2. **Q6 (CGPA for good standing) is a false refusal.** The answer is in the source (pages 20–21), but the retrieved chunks didn't surface it. Fix: query rewriting or hybrid (BM25 + vector) retrieval.

3. **Keyword matching is brittle.** The eval's string-match component scores 70% because answers often use synonyms (e.g., *re-register* vs. *repeat*). Fix: replace with semantic similarity on the answer.

4. **Tables lose structure on extraction.** Common PDF limitation. Numbers and cells are preserved as flattened text; deeper table-aware parsing would require `pdfplumber` or `unstructured`.

---

## Tech Stack

| Layer | Tool |
|---|---|
| **PDF extraction** | PyMuPDF |
| **Tokenization** | tiktoken |
| **Embeddings** | BAAI/bge-small-en-v1.5 (Sentence Transformers) |
| **Vector DB** | Qdrant (Docker) |
| **LLM** | OpenAI GPT-4o-mini |
| **Web UI** | Streamlit |
| **Evaluation** | Custom harness with LLM-as-judge |
| **Deployment** | Streamlit Community Cloud |
| **Language** | Python 3.11 |

---

## Running Locally

### Prerequisites
- Python 3.11
- Docker (for Qdrant)
- OpenAI API key

### Setup

```bash
git clone https://github.com/Faizan-11/rag-college-rules.git
cd rag-college-rules

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start Qdrant
docker run -d --name qdrant -p 6333:6333 -p 6334:6334 \
  -v "$(pwd)/qdrant_storage:/qdrant/storage" qdrant/qdrant:latest

# Set up secrets
echo "OPENAI_API_KEY=sk-your-key-here" > .env

# Run the Streamlit app
streamlit run app/streamlit_app.py
```

### Run the evaluation harness

```bash
python scripts/07_eval.py
```

---

## What I'd Improve Next

- **Hybrid retrieval** — combine BM25 keyword search with vector search to fix Q6
- **Query rewriting** — use an LLM to rephrase the user's question before retrieval
- **Cross-encoder reranking** — add a second-pass reranker (bge-reranker) to improve top-k precision
- **Semantic answer scoring** — replace keyword matching in the eval with cosine similarity
- **Scale test** — replicate the pipeline on a 500+ page document (e.g., Indian Income Tax Act)

---

## Project Structure

```
rag-college-rules/
├── app/
│   └── streamlit_app.py          # Deployed web UI
├── scripts/
│   ├── 01_extract.py             # PDF → text
│   ├── 02_chunk.py               # text → chunks
│   ├── 03_embed.py               # chunks → vectors
│   ├── 04_store.py               # vectors → Qdrant
│   ├── 05_query.py               # semantic search demo
│   ├── 06_generate.py            # RAG pipeline
│   └── 07_eval.py                # evaluation harness
├── data/
│   ├── raw/                      # source PDF
│   ├── chunks/                   # chunked + embedded data
│   └── eval/                     # test set + results
├── requirements.txt
└── .gitignore
```

---

## License

MIT — see [LICENSE](LICENSE).

---

*Built as a portfolio project demonstrating end-to-end RAG engineering: data pipeline → embeddings → vector search → LLM generation → evaluation → deployment.*