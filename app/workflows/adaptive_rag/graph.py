from __future__ import annotations

from typing import Any

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from app.workflows.adaptive_rag.nodes import (
    agent_rag_node,
    direct_answer_node,
    fixed_rag_node,
    route_decision_node,
    route_selector,
    strict_insufficient_node,
)
from app.workflows.adaptive_rag.state import AdaptiveRagState


_adaptive_rag_graph: Any | None = None
_adaptive_rag_checkpointer = InMemorySaver()


def build_adaptive_rag_graph():
    global _adaptive_rag_graph
    if _adaptive_rag_graph is not None:
        return _adaptive_rag_graph

    graph_builder = StateGraph(AdaptiveRagState)
    graph_builder.add_node("route_decision", route_decision_node)
    graph_builder.add_node("direct_answer", direct_answer_node)
    graph_builder.add_node("fixed_rag", fixed_rag_node)
    graph_builder.add_node("agent_rag", agent_rag_node)
    graph_builder.add_node("strict_insufficient", strict_insufficient_node)

    graph_builder.add_edge(START, "route_decision")
    graph_builder.add_conditional_edges(
        "route_decision",
        route_selector,
        {
            "direct_answer": "direct_answer",
            "fixed_rag": "fixed_rag",
            "agent_rag": "agent_rag",
            "strict_insufficient": "strict_insufficient",
        },
    )

    graph_builder.add_edge("direct_answer", END)
    graph_builder.add_edge("fixed_rag", END)
    graph_builder.add_edge("agent_rag", END)
    graph_builder.add_edge("strict_insufficient", END)

    _adaptive_rag_graph = graph_builder.compile(checkpointer=_adaptive_rag_checkpointer)
    return _adaptive_rag_graph
