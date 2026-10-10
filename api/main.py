import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from rag import config
from rag.chain import NOT_FOUND, answer, make_llm
from rag.ingest import load_pdf, split_documents
from rag.vectorstore import FastEmbedder, build_index, retrieve
from api.db import get_stats, init_db, log_query

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("api")

SAMPLE_PDF = Path(__file__).parent.parent / "sample_docs" / "northwind_home_loan_policy.pdf"
state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Runs ONCE when the server starts
    init_db()
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not set")
    pages = load_pdf(SAMPLE_PDF.read_bytes(), SAMPLE_PDF.name)
    chunks = split_documents(pages)
    state["index"] = build_index(chunks, FastEmbedder())
    state["llm"] = make_llm(api_key)
    log.info("Index ready: %d chunks", len(chunks))
    yield
    # Runs once when the server stops
    state.clear()


app = FastAPI(title="Enterprise RAG API", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    top_k: int = Field(default=config.TOP_K, ge=1, le=8)


class Source(BaseModel):
    source: str
    page: int
    score: float


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]
    latency_sec: float


@app.get("/health")
def health():
    return {"status": "ok", "index_ready": "index" in state}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    start = time.perf_counter()
    try:
        hits = retrieve(state["index"], req.question, k=req.top_k)
        reply = answer(state["llm"], req.question, hits)
    except Exception as exc:
        latency = round(time.perf_counter() - start, 2)
        log.exception("Question failed")
        log_query(req.question, str(exc), None, latency, "error")
        raise HTTPException(status_code=503, detail="LLM call failed, please try again")

    latency = round(time.perf_counter() - start, 2)
    best_score = round(hits[0][1], 2) if hits else 0.0
    status = "not_found" if reply == NOT_FOUND else "success"
    sources = [] if reply == NOT_FOUND else [
        Source(source=d.metadata["source"], page=d.metadata["page"], score=round(s, 2))
        for d, s in hits
    ]
    log.info("q=%r best_score=%.2f latency=%.2fs status=%s", req.question, best_score, latency, status)
    log_query(req.question, reply, best_score, latency, status)
    return AskResponse(answer=reply, sources=sources, latency_sec=latency)


@app.get("/stats")
def stats():
    return get_stats()