import pytest
from langchain_core.documents import Document

from rag import config
from rag.ingest import IngestError, load_pdf, split_documents


def test_load_pdf_keeps_page_numbers(make_pdf):
    data = make_pdf(["First page about interest rates.", "Second page about prepayment."])
    pages = load_pdf(data, "policy.pdf")

    assert [p.metadata["page"] for p in pages] == [1, 2]
    assert all(p.metadata["source"] == "policy.pdf" for p in pages)
    assert "prepayment" in pages[1].page_content


def test_blank_pdf_is_rejected(make_pdf):
    with pytest.raises(IngestError, match="No text found"):
        load_pdf(make_pdf([""]), "scan.pdf")


def test_non_pdf_is_rejected():
    with pytest.raises(IngestError, match="could not be read"):
        load_pdf(b"this is not a pdf", "notes.pdf")


def test_oversized_file_is_rejected(monkeypatch):
    monkeypatch.setattr(config, "MAX_FILE_MB", 0)
    with pytest.raises(IngestError, match="limit"):
        load_pdf(b"x" * 2048, "big.pdf")


def test_split_keeps_metadata_and_overlaps():
    text = " ".join(f"word{i}" for i in range(400))
    chunks = split_documents([Document(page_content=text, metadata={"source": "a.pdf", "page": 7})],
                             chunk_size=200, chunk_overlap=50)

    assert len(chunks) > 1
    assert all(c.metadata["source"] == "a.pdf" and c.metadata["page"] == 7 for c in chunks)
    assert all(len(c.page_content) <= 200 for c in chunks)
    # overlap: the end of one chunk reappears at the start of the next
    last_word = chunks[0].page_content.split()[-1]
    assert last_word in chunks[1].page_content
