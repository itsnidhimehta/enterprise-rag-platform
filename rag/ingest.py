"""Step 1 of RAG: turn PDFs into small, labelled chunks of text.

Analogy: we tear a big book into index cards. Each card keeps a note of
which book and which page it came from, so answers can cite their source.
"""

from __future__ import annotations

import io
import logging

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from rag import config

log = logging.getLogger(__name__)


class IngestError(ValueError):
    """Raised when a file can't be used (empty, too big, scanned image, ...)."""


def load_pdf(data: bytes, filename: str) -> list[Document]:
    """Read a PDF and return one Document per page that has text."""
    size_mb = len(data) / (1024 * 1024)
    if size_mb > config.MAX_FILE_MB:
        raise IngestError(f"{filename} is {size_mb:.1f} MB; the limit is {config.MAX_FILE_MB} MB.")

    try:
        reader = PdfReader(io.BytesIO(data))
    except Exception as exc:  # pypdf raises several error types
        raise IngestError(f"{filename} could not be read as a PDF.") from exc

    if len(reader.pages) > config.MAX_PAGES:
        raise IngestError(f"{filename} has {len(reader.pages)} pages; the limit is {config.MAX_PAGES}.")

    pages = []
    for number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append(Document(page_content=text, metadata={"source": filename, "page": number}))

    if not pages:
        # Usually a scanned PDF: the pages are pictures, not text.
        raise IngestError(f"No text found in {filename}. It may be a scanned image (needs OCR).")

    log.info("Loaded %s: %d pages with text", filename, len(pages))
    return pages


def split_documents(
    pages: list[Document],
    chunk_size: int = config.CHUNK_SIZE,
    chunk_overlap: int = config.CHUNK_OVERLAP,
) -> list[Document]:
    """Split pages into overlapping chunks, keeping source + page on every chunk.

    The splitter tries paragraph breaks first, then lines, then sentences, then
    words - so chunks end at natural boundaries instead of mid-word. The overlap
    means a sentence that sits on a boundary appears in both neighbours.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(pages)
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
    log.info("Split %d pages into %d chunks", len(pages), len(chunks))
    return chunks
