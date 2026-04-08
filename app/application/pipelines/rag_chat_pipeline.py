from __future__ import annotations

from typing import Any

from app.workflows.rag.graph import build_rag_chat_graph
from app.workflows.rag.state import RagChatState


async def run_rag_chat_pipeline(
    *,
    question: str,
    collection_name: str | None = None,
    knowledge_domain: str | None = None,
    book_id: str | None = None,
    top_k: int | None = None,
) -> dict[str, Any]:
    graph = build_rag_chat_graph()
    state: RagChatState = {
        "question": question,
    }
    if collection_name:
        state["collection_name"] = collection_name
    if knowledge_domain:
        state["knowledge_domain"] = knowledge_domain
    if book_id:
        state["book_id"] = book_id
    if top_k is not None:
        state["top_k"] = top_k

    result = await graph.ainvoke(state)
    return {
        "answer": result.get("answer", ""),
        "citations": result.get("citations", []),
    }
