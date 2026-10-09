"""Smoke test of the real Streamlit UI, with the embedding model and LLM swapped for fakes."""

from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models import FakeListChatModel
from streamlit.testing.v1 import AppTest

import rag.chain
import rag.vectorstore


def test_app_answers_with_sources(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setattr(rag.vectorstore, "FastEmbedder", lambda: DeterministicFakeEmbedding(size=64))
    monkeypatch.setattr(rag.chain, "make_llm",
                        lambda api_key: FakeListChatModel(responses=["Up to 90% [1]."]))
    # Fake embeddings aren't semantic, so let every match through in this UI test.
    monkeypatch.setattr(rag.chain.config, "MIN_RELEVANCE", 0.0)

    at = AppTest.from_file("../app.py", default_timeout=60).run()
    assert not at.exception
    assert "Indexed" in at.sidebar.success[0].value  # sample PDF was loaded and indexed

    at.chat_input[0].set_value("What is the maximum LTV?").run()
    assert not at.exception
    assert at.chat_message[-1].markdown[0].value == "Up to 90% [1]."
    assert "Sources" in at.chat_message[-1].expander[0].label


def test_app_explains_missing_key(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    at = AppTest.from_file("../app.py", default_timeout=60).run()
    assert "GROQ_API_KEY is not set" in at.error[0].value
