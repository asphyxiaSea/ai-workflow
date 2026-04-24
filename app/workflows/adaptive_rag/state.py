from __future__ import annotations

from typing import Any, Annotated, Literal, TypedDict
from typing_extensions import NotRequired

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


RagRoute = Literal[
    "direct_answer",
    "fixed_rag",
    "agent_rag",
    "strict_insufficient",
]


class AdaptiveRagState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    collection_name: NotRequired[str]
    knowledge_domain: NotRequired[str]
    book_id: NotRequired[str]
    top_k: NotRequired[int]
    route: NotRequired[RagRoute]
    route_reason: NotRequired[str]
    rewritten_question: NotRequired[str]
    docs_with_scores: NotRequired[list[tuple[Document, float]]]
    citations: NotRequired[list[dict[str, Any]]]
    retrieval_count: NotRequired[int]
    answer: NotRequired[str]
    trace: NotRequired[dict[str, Any]]
