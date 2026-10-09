"""Step 2 of RAG: turn chunks into vectors and search them by meaning.

Analogy: every index card gets GPS coordinates in "meaning space". Cards about
similar ideas sit close together. To answer a question, we drop a pin for the
question and pick up the nearest cards.
"""

from __future__ import annotations

import logging

import faiss
import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from rag import config

log = logging.getLogger(__name__)


class FastEmbedder(Embeddings):
    """Small wrapper so the pipeline can use FastEmbed.

    FastEmbed runs the model on the CPU with ONNX - no GPU, no API key, no
    cost per call. The model (~130 MB) downloads once and is then cached.
    """

    def __init__(self, model_name: str = config.EMBED_MODEL, cache_dir: str = config.EMBED_CACHE_DIR):
        from fastembed import TextEmbedding  # imported here so tests don't need the model

        self.model = TextEmbedding(model_name=model_name, cache_dir=cache_dir)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [vector.tolist() for vector in self.model.embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        # BGE models are trained with a special prefix on search queries; query_embed adds it.
        return next(iter(self.model.query_embed(text))).tolist()


def _to_unit_vectors(vectors: list[list[float]]) -> np.ndarray:
    """Scale every vector to length 1, so inner product == cosine similarity."""
    array = np.asarray(vectors, dtype="float32")
    faiss.normalize_L2(array)
    return array


class VectorIndex:
    """FAISS index plus the chunks it points to.

    IndexFlatIP compares the question with every chunk exactly (no
    approximation). That is fast enough for thousands of chunks; for millions
    you would switch to an approximate index such as HNSW or IVF.
    """

    def __init__(self, chunks: list[Document], embeddings: Embeddings):
        if not chunks:
            raise ValueError("Cannot build an index with no chunks.")
        self.chunks = chunks
        self.embeddings = embeddings
        vectors = _to_unit_vectors(embeddings.embed_documents([c.page_content for c in chunks]))
        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)

    def __len__(self) -> int:
        return self.index.ntotal

    def search(self, question: str, k: int) -> list[tuple[Document, float]]:
        query = _to_unit_vectors([self.embeddings.embed_query(question)])
        scores, ids = self.index.search(query, min(k, len(self)))
        # Cosine similarity is -1..1; clip to 0..1 so it reads as a relevance score.
        return [(self.chunks[i], float(max(s, 0.0))) for s, i in zip(scores[0], ids[0]) if i != -1]


def build_index(chunks: list[Document], embeddings: Embeddings) -> VectorIndex:
    """Embed every chunk and store the vectors in an in-memory FAISS index."""
    index = VectorIndex(chunks, embeddings)
    log.info("Built FAISS index with %d vectors", len(index))
    return index


def retrieve(index: VectorIndex, question: str, k: int = config.TOP_K) -> list[tuple[Document, float]]:
    """Return the k most relevant chunks, best first, with a 0-1 relevance score."""
    return index.search(question, k)
