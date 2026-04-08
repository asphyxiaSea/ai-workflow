from __future__ import annotations

from typing import Any, TypedDict
from typing_extensions import NotRequired

from langchain_core.documents import Document


class RagChatState(TypedDict):
    question: str
    collection_name: NotRequired[str]
    knowledge_domain: NotRequired[str]
    book_id: NotRequired[str]
    top_k: NotRequired[int]
    docs_with_scores: NotRequired[list[tuple[Document, float]]]
    answer: NotRequired[str]
    citations: NotRequired[list[dict[str, Any]]]
