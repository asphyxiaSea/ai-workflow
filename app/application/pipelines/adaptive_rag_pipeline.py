from __future__ import annotations

from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage

from app.core.errors import InvalidRequestError
from app.workflows.adaptive_rag.graph import build_adaptive_rag_graph
from app.workflows.adaptive_rag.state import AdaptiveRagState


async def run_adaptive_rag_pipeline(
    *,
    messages: list[dict[str, Any]],
    thread_id: str,
    user_id: str,
    collection_name: str | None = None,
    knowledge_domain: str | None = None,
    book_id: str | None = None,
    top_k: int | None = None,
) -> dict[str, Any]:
    if not messages:
        raise InvalidRequestError(message="messages 不能为空")

    normalized_messages: list[BaseMessage] = [
        HumanMessage(content=str(message["content"]).strip())
        for message in messages
        if str(message["content"]).strip()
    ]
    if not normalized_messages:
        raise InvalidRequestError(message="messages 不能为空")

    graph = build_adaptive_rag_graph()

    state: AdaptiveRagState = {
        "messages": normalized_messages,
    }
    if collection_name is not None:
        state["collection_name"] = collection_name
    if knowledge_domain is not None:
        state["knowledge_domain"] = knowledge_domain
    if book_id is not None:
        state["book_id"] = book_id
    if top_k is not None:
        state["top_k"] = top_k

    result = await graph.ainvoke(
        state,
        config={
            "configurable": {
                "thread_id": thread_id.strip(),
                "user_id": user_id.strip(),
            }
        },
    )
    return {
        "answer": result.get("answer", ""),
        "citations": result.get("citations", []),
        "trace": result.get("trace", {}),
    }
