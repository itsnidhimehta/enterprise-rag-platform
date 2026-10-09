"""Retrieval + answer tests.

They use fake embeddings and a fake LLM, so they run in seconds with no
internet and no API key - which is what lets them run in CI on every push.
"""

from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models import FakeListChatModel

from rag.chain import NOT_FOUND, answer, format_context, is_answerable
from rag.vectorstore import build_index, retrieve

CHUNKS = [
    Document(page_content="The maximum LTV is 90% for small loans.", metadata={"source": "p.pdf", "page": 2}),
    Document(page_content="Prepayment has no penalty on floating rates.", metadata={"source": "p.pdf", "page": 4}),
    Document(page_content="Complaints are acknowledged in 2 working days.", metadata={"source": "p.pdf", "page": 5}),
]


def test_retrieve_ranks_matching_chunk_first():
    index = build_index(CHUNKS, DeterministicFakeEmbedding(size=64))
    hits = retrieve(index, CHUNKS[1].page_content, k=3)

    assert hits[0][0].metadata["page"] == 4
    scores = [score for _, score in hits]
    assert scores == sorted(scores, reverse=True)
    assert hits[0][1] > 0.99  # identical text -> near-perfect relevance


def test_format_context_numbers_and_labels_chunks():
    text = format_context([(CHUNKS[0], 0.9), (CHUNKS[1], 0.8)])
    assert text.startswith("[1] (p.pdf, page 2)")
    assert "[2] (p.pdf, page 4)" in text


def test_weak_match_skips_llm():
    class ExplodingLLM(FakeListChatModel):
        def invoke(self, *args, **kwargs):
            raise AssertionError("LLM must not be called for weak matches")

    llm = ExplodingLLM(responses=["unused"])
    assert answer(llm, "What is the weather?", [(CHUNKS[0], 0.05)]) == NOT_FOUND
    assert answer(llm, "Anything?", []) == NOT_FOUND


def test_strong_match_calls_llm():
    llm = FakeListChatModel(responses=["The maximum LTV is 90% [1]."])
    reply = answer(llm, "What is the max LTV?", [(CHUNKS[0], 0.85)])
    assert reply == "The maximum LTV is 90% [1]."


def test_is_answerable_threshold():
    assert is_answerable([(CHUNKS[0], 0.5)], min_relevance=0.3)
    assert not is_answerable([(CHUNKS[0], 0.2)], min_relevance=0.3)
