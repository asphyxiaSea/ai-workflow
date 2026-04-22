from __future__ import annotations

from typing import Any

from app.workflows.adaptive_rag.graph import build_adaptive_rag_graph
from app.workflows.adaptive_rag.state import AdaptiveRagState


async def run_adaptive_rag_pipeline(
    *,
    question: str,
    collection_name: str | None = None,
    knowledge_domain: str | None = None,
    book_id: str | None = None,
    top_k: int | None = None,
) -> dict[str, Any]:
    graph = build_adaptive_rag_graph()

    state: AdaptiveRagState = {
        "question": question,
    }
    if collection_name is not None:
        state["collection_name"] = collection_name
    if knowledge_domain is not None:
        state["knowledge_domain"] = knowledge_domain
    if book_id is not None:
        state["book_id"] = book_id
    if top_k is not None:
        state["top_k"] = top_k

    result = await graph.ainvoke(state)
    return {
        "answer": result.get("answer", ""),
        "citations": result.get("citations", []),
        "trace": result.get("trace", {}),
    }
