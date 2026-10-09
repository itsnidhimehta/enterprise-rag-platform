"""Streamlit UI for the Enterprise RAG Platform.

Run locally:  streamlit run app.py
"""

from __future__ import annotations

import hashlib
import logging
import os
import time
from pathlib import Path

import streamlit as st

from rag import config
from rag.chain import NOT_FOUND, answer, make_llm
from rag.ingest import IngestError, load_pdf, split_documents
from rag.vectorstore import FastEmbedder, build_index, retrieve

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("app")

SAMPLE_PDF = Path(__file__).parent / "sample_docs" / "northwind_home_loan_policy.pdf"

st.set_page_config(page_title="Enterprise RAG Platform", page_icon="📄", layout="wide")


# ---------- helpers ----------------------------------------------------------

def get_api_key() -> str | None:
    """Read the key from Streamlit secrets (cloud) or an environment variable (local/Docker)."""
    try:
        if "GROQ_API_KEY" in st.secrets:
            return st.secrets["GROQ_API_KEY"]
    except Exception:  # no secrets.toml file at all
        pass
    return os.getenv("GROQ_API_KEY")


@st.cache_resource(show_spinner="Loading embedding model (first run only)...")
def get_embedder() -> FastEmbedder:
    # cache_resource = load once per server, shared by every user.
    return FastEmbedder()


def index_files(files: list[tuple[str, bytes]]) -> None:
    """Ingest -> chunk -> embed -> FAISS, then keep the index in this user's session."""
    fingerprint = hashlib.sha256(b"".join(name.encode() + data for name, data in files)).hexdigest()
    if st.session_state.get("fingerprint") == fingerprint:
        return  # same files as before: reuse the index, don't re-embed

    start = time.perf_counter()
    pages = []
    for name, data in files:
        try:
            pages.extend(load_pdf(data, name))
        except IngestError as exc:
            st.sidebar.warning(str(exc))
    if not pages:
        return

    chunks = split_documents(pages)
    with st.spinner(f"Embedding {len(chunks)} chunks..."):
        st.session_state.index = build_index(chunks, get_embedder())
    st.session_state.fingerprint = fingerprint
    st.session_state.stats = {
        "files": sorted({p.metadata["source"] for p in pages}),
        "pages": len(pages),
        "chunks": len(chunks),
        "seconds": time.perf_counter() - start,
    }
    st.session_state.messages = []  # new documents -> fresh conversation


# ---------- session state ----------------------------------------------------

st.session_state.setdefault("messages", [])
st.session_state.setdefault("questions_asked", 0)

# ---------- sidebar: documents ----------------------------------------------

with st.sidebar:
    st.header("📚 Documents")
    uploads = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
    use_sample = st.toggle("Use the sample bank policy PDF", value=not uploads,
                           help="A fictional home-loan policy, so you can try the app without uploading anything.")

    files = [(f.name, f.getvalue()) for f in uploads or []]
    if use_sample and SAMPLE_PDF.exists():
        files.append((SAMPLE_PDF.name, SAMPLE_PDF.read_bytes()))

    if files:
        index_files(files)

    stats = st.session_state.get("stats")
    if stats and files:
        st.success(f"Indexed {stats['pages']} pages → {stats['chunks']} chunks in {stats['seconds']:.1f}s")
        for name in stats["files"]:
            st.caption(f"• {name}")

    top_k = st.slider("Chunks to retrieve (top-k)", 1, 8, config.TOP_K)
    st.divider()
    st.caption(f"LLM: `{config.LLM_MODEL}` via Groq  \nEmbeddings: `{config.EMBED_MODEL}`  \nVector store: FAISS")
    if st.button("Clear chat"):
        st.session_state.messages = []

# ---------- main: chat --------------------------------------------------------

st.title("📄 Enterprise RAG Platform")
st.caption("Ask questions about your PDFs. Every answer cites the exact page it came from.")

api_key = get_api_key()
if not api_key:
    st.error("GROQ_API_KEY is not set. Add it to `.streamlit/secrets.toml` or as an environment variable.")
    st.stop()

if not files or "index" not in st.session_state:
    st.info("Upload a PDF or switch on the sample document in the sidebar to start.")
    st.stop()


def show_sources(sources: list[dict]) -> None:
    with st.expander(f"Sources ({len(sources)} chunks retrieved)"):
        for i, s in enumerate(sources, start=1):
            st.markdown(f"**[{i}] {s['source']} — page {s['page']}** · relevance `{s['score']:.2f}`")
            st.caption(s["text"][:600] + ("..." if len(s["text"]) > 600 else ""))


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            show_sources(msg["sources"])
        if msg.get("timing"):
            st.caption(msg["timing"])

if not st.session_state.messages:
    st.markdown("**Try:** *What is the maximum loan-to-value ratio?* · *What documents do I need?* · "
                "*Can I prepay my loan?*")

question = st.chat_input("Ask a question about your documents")
if question:
    if st.session_state.questions_asked >= config.MAX_QUESTIONS_PER_SESSION:
        st.warning("Demo limit reached for this session. Refresh the page to start again.")
        st.stop()
    st.session_state.questions_asked += 1

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            t0 = time.perf_counter()
            hits = retrieve(st.session_state.index, question, k=top_k)
            t1 = time.perf_counter()
            with st.spinner("Thinking..."):
                reply = answer(make_llm(api_key), question, hits)
            t2 = time.perf_counter()
        except Exception:
            log.exception("Question failed")
            reply, hits, t0 = "Sorry, something went wrong talking to the LLM. Please try again.", [], None

        sources = [] if reply == NOT_FOUND else [
            {"source": d.metadata["source"], "page": d.metadata["page"], "score": s, "text": d.page_content}
            for d, s in hits
        ]
        timing = f"⏱ retrieval {(t1 - t0) * 1000:.0f} ms · generation {(t2 - t1):.1f} s" if t0 else ""
        log.info("q=%r best_score=%.2f %s", question, hits[0][1] if hits else 0, timing)

        st.markdown(reply)
        if sources:
            show_sources(sources)
        if timing:
            st.caption(timing)

    st.session_state.messages.append(
        {"role": "assistant", "content": reply, "sources": sources, "timing": timing}
    )
