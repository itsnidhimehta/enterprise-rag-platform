"""All tunable settings in one place.

Every value can be overridden with an environment variable, so the same code
runs on a laptop, inside Docker, and on Streamlit Cloud without edits.
"""

import os


def _int(name: str, default: int) -> int:
    return int(os.getenv(name, default))


def _float(name: str, default: float) -> float:
    return float(os.getenv(name, default))


# --- LLM (Groq) ---------------------------------------------------------
# openai/gpt-oss-120b is a current Groq production model. Llama 3.3 70B was
# retired on Groq's free/dev tier in August 2026, so we don't use it here.
LLM_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
LLM_TEMPERATURE = _float("LLM_TEMPERATURE", 0.0)  # 0 = factual, repeatable answers

# --- Embeddings (runs locally, free, no API key) ------------------------
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
EMBED_CACHE_DIR = os.getenv("EMBED_CACHE_DIR", ".cache/fastembed")

# --- Chunking -----------------------------------------------------------
CHUNK_SIZE = _int("CHUNK_SIZE", 800)        # characters per chunk
CHUNK_OVERLAP = _int("CHUNK_OVERLAP", 150)  # shared characters between neighbours

# --- Retrieval ----------------------------------------------------------
TOP_K = _int("TOP_K", 4)                    # chunks sent to the LLM
MIN_RELEVANCE = _float("MIN_RELEVANCE", 0.30)  # below this we answer "not found"

# --- Guardrails (protect the free API key on a public demo) -------------
MAX_FILE_MB = _int("MAX_FILE_MB", 10)
MAX_PAGES = _int("MAX_PAGES", 150)
MAX_QUESTIONS_PER_SESSION = _int("MAX_QUESTIONS_PER_SESSION", 25)
