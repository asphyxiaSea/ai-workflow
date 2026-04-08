from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from app.workflows.rag.nodes import (
    generate_answer_node,
    retrieve_node,
)
from app.workflows.rag.state import RagChatState


def build_rag_chat_graph():
    graph_builder = StateGraph(RagChatState)
    graph_builder.add_node("retrieve", retrieve_node)
    graph_builder.add_node("generate_answer", generate_answer_node)

    graph_builder.add_edge(START, "retrieve")
    graph_builder.add_edge("retrieve", "generate_answer")
    graph_builder.add_edge("generate_answer", END)
    return graph_builder.compile()
