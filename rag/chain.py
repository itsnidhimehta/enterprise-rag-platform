"""Step 3 of RAG: give the LLM the retrieved chunks and ask for a grounded answer.

Analogy: an open-book exam. The LLM may only use the index cards we hand it,
and must say which card each fact came from.
"""

from __future__ import annotations

import logging

from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from rag import config

log = logging.getLogger(__name__)

NOT_FOUND = "I couldn't find that in the uploaded documents."

SYSTEM_PROMPT = f"""You are a careful assistant that answers questions about the user's documents.

Rules:
1. Use ONLY the numbered context passages. Do not use outside knowledge.
2. After each fact, cite the passage it came from like [1] or [2].
3. If the passages don't contain the answer, reply exactly: "{NOT_FOUND}"
4. Be concise. Use short bullet points when listing several items.
"""


def format_context(hits: list[tuple[Document, float]]) -> str:
    """Number each chunk so the LLM can cite it, e.g. [1] (policy.pdf, page 3)."""
    blocks = []
    for number, (doc, _score) in enumerate(hits, start=1):
        label = f"{doc.metadata.get('source', 'document')}, page {doc.metadata.get('page', '?')}"
        blocks.append(f"[{number}] ({label})\n{doc.page_content}")
    return "\n\n".join(blocks)


def is_answerable(hits: list[tuple[Document, float]], min_relevance: float | None = None) -> bool:
    """Guardrail: if even the best chunk is a weak match, don't ask the LLM at all.

    This stops the model from inventing an answer, and saves an API call.
    """
    threshold = config.MIN_RELEVANCE if min_relevance is None else min_relevance
    return bool(hits) and hits[0][1] >= threshold


def answer(llm: BaseChatModel, question: str, hits: list[tuple[Document, float]]) -> str:
    if not is_answerable(hits):
        log.info("Best relevance below threshold; returning NOT_FOUND without calling the LLM")
        return NOT_FOUND

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"Context:\n{format_context(hits)}\n\nQuestion: {question}"),
    ]
    response = llm.invoke(messages)
    return str(response.content).strip()


def make_llm(api_key: str, model: str = config.LLM_MODEL) -> BaseChatModel:
    from langchain_groq import ChatGroq

    return ChatGroq(
        model=model,
        api_key=api_key,
        temperature=config.LLM_TEMPERATURE,
        reasoning_format="hidden",  # gpt-oss "thinks" first; show only the final answer
        max_retries=2,
        timeout=60,
    )
