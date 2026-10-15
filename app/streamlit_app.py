"""
streamlit_app.py
----------------
A public web demo of our RAG system over IIT Bombay rules.

Run locally:
    streamlit run app/streamlit_app.py

Deploy to HuggingFace Spaces:
    Push to a Space repo — see README for instructions.
"""

import json
import os
from pathlib import Path

import numpy as np
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer

# ============ CONFIG ============
CHUNKS_FILE = Path(__file__).parent / "chunks.jsonl"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
LLM_MODEL = "gpt-4o-mini"
TOP_K = 5

SYSTEM_PROMPT = """You are a helpful assistant answering questions about IIT Bombay's academic rules and regulations.

RULES FOR YOUR ANSWER:
1. Answer ONLY using the provided context passages below.
2. If the context doesn't contain the answer, say: "I don't have enough information in the provided documents to answer that."
3. Cite page numbers in the format [pages X-Y] after each fact you state.
4. Be concise — 2 to 4 sentences unless the question requires more.
5. Do not make up information. If unsure, say so.
"""


# ============ LOAD RESOURCES (CACHED) ============
@st.cache_resource
def load_chunks_and_embeddings():
    """Load chunks + embeddings from disk. Cached so it only runs once per session."""
    chunks = []
    with CHUNKS_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    embeddings = np.array([c["embedding"] for c in chunks], dtype=np.float32)
    # Normalize so cosine similarity = dot product
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.clip(norms, 1e-9, None)
    return chunks, embeddings


@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(EMBED_MODEL)


@st.cache_resource
def load_openai_client():
    try:
        from dotenv import load_dotenv
        load_dotenv(override=True)
    except ImportError:
        pass
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("OPENAI_API_KEY not set. Add it to your environment or secrets.")
        st.stop()
    return OpenAI(api_key=api_key, timeout=30.0, max_retries=2)


# ============ RETRIEVAL & GENERATION ============
def retrieve_top_k(question: str, chunks, embeddings, model, k: int = TOP_K):
    q_emb = model.encode(question, normalize_embeddings=True).astype(np.float32)
    scores = embeddings @ q_emb
    top_idx = np.argsort(scores)[::-1][:k]
    results = []
    for i in top_idx:
        results.append({
            "score": float(scores[i]),
            "text": chunks[i]["text"],
            "pages": chunks[i]["pages"],
        })
    return results


def build_context(hits):
    parts = []
    for i, hit in enumerate(hits, start=1):
        parts.append(f"--- Passage {i} [pages {hit['pages']}] ---\n{hit['text']}")
    return "\n\n".join(parts)


def generate_answer(client, question: str, context: str) -> str:
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Question: {question}\n\nContext:\n{context}\n\nAnswer using ONLY the passages above."},
        ],
        temperature=0.1,
        max_tokens=500,
    )
    return response.choices[0].message.content


# ============ STREAMLIT UI ============
st.set_page_config(page_title="IITB Rules Assistant", page_icon="🎓", layout="centered")

st.title("🎓 IITB Academic Rules Assistant")
st.caption(
    "Ask questions about IIT Bombay's undergraduate academic rules. "
    "Answers cite the source pages from the official rulebook."
)

with st.expander("ℹ️ How this works"):
    st.markdown("""
    This is a **Retrieval-Augmented Generation (RAG)** system:
    1. Your question is converted into an embedding (a semantic fingerprint).
    2. The system finds the 5 most relevant passages from the rulebook.
    3. An LLM (GPT-4o-mini) reads those passages and writes an answer with citations.

    **Built with:** Sentence Transformers, Qdrant (dev), OpenAI GPT-4o-mini, Streamlit.

    **Tip:** Ask specific questions like *"What happens if I fail a course?"* — vague questions work less well.
    """)

# Load resources
with st.spinner("Loading models and document chunks..."):
    chunks, embeddings = load_chunks_and_embeddings()
    embed_model = load_embedding_model()
    openai_client = load_openai_client()

# Example questions
st.write("**Try one of these:**")
examples = [
    "What happens if I fail a course?",
    "Can I take a break from my studies?",
    "How do I calculate my CPI?",
    "What is the attendance requirement?",
    "What is a Minor degree?",
]
cols = st.columns(len(examples))
for i, example in enumerate(examples):
    if cols[i].button(example, key=f"ex_{i}", use_container_width=True):
        st.session_state["question"] = example

# Question input
question = st.text_input(
    "Ask a question:",
    value=st.session_state.get("question", ""),
    placeholder="e.g. How do I calculate my CPI?",
)

if st.button("Get Answer", type="primary") and question.strip():
    with st.spinner("Searching rules and generating answer..."):
        hits = retrieve_top_k(question, chunks, embeddings, embed_model)
        context = build_context(hits)
        answer = generate_answer(openai_client, question, context)

    st.markdown("### Answer")
    st.markdown(answer)

    with st.expander("📚 Source passages"):
        for i, hit in enumerate(hits, start=1):
            st.markdown(f"**Passage {i}** — pages {hit['pages']} (relevance {hit['score']:.3f})")
            st.text(hit["text"][:600] + ("..." if len(hit["text"]) > 600 else ""))
            st.divider()

st.divider()
st.caption("Built from IIT Bombay UG Rules & Regulations. Not an official IITB service.")