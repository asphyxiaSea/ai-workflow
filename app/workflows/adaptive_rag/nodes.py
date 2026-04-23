from __future__ import annotations

from typing import Any, Literal, cast

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain.tools import tool
from pydantic import BaseModel

from app.core.model_factory import get_chat_model
from app.core.errors import InvalidRequestError
from app.core.settings import (
    ADAPTIVE_RAG_DIRECT_PROMPT,
    ADAPTIVE_RAG_REWRITE_PROMPT,
    ADAPTIVE_RAG_ROUTER_PROMPT,
    RAG_CHROMA_COLLECTION,
    RAG_RETRIEVAL_TOP_K,
)
from app.infra.clients.chroma_client import build_citations, search_chroma
from app.workflows.adaptive_rag.state import AdaptiveRagState, RagRoute


class RouteDecision(BaseModel):
    route: Literal[
        "direct_answer",
        "fixed_rag",
        "agent_rag",
        "strict_insufficient",
    ]
    reason: str


class QueryRewrite(BaseModel):
    rewritten_query: str


def _build_trace(*, route: str, reason: str, retrieval_count: int) -> dict[str, Any]:
    return {
        "route": route,
        "reason": reason,
        "retrieval_count": retrieval_count,
    }


def _extract_route_reason(state: AdaptiveRagState) -> str:
    reason = str(state.get("route_reason") or "")
    return reason.strip() or "路由模型未提供原因"


def _extract_final_text(messages: list[BaseMessage]) -> str:
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            content = message.content
            if isinstance(content, str):
                text = content.strip()
                if text:
                    return text
            elif isinstance(content, list):
                parts: list[str] = []
                for item in content:
                    if isinstance(item, dict):
                        text = item.get("text")
                        if isinstance(text, str) and text.strip():
                            parts.append(text.strip())
                if parts:
                    return "\n".join(parts)
    return ""


def _build_context_blocks(docs_with_scores: list[tuple[Any, float]]) -> str:
    contexts: list[str] = []
    for idx, (doc, score) in enumerate(docs_with_scores, start=1):
        source = doc.metadata.get("source", "unknown")
        chunk_idx = doc.metadata.get("chunk_index", -1)
        contexts.append(
            f"[{idx}] source={source}, chunk={chunk_idx}, score={score:.4f}\n{doc.page_content}"
        )
    return "\n\n".join(contexts)


async def _answer_with_context(*, question: str, rewritten_question: str, contexts: str) -> str:
    model = get_chat_model()
    result = await model.ainvoke(
        [
            SystemMessage(
                content=(
                    "你是企业知识库问答助手。"
                    "只能根据给定上下文回答，禁止编造。"
                    "若证据不足，请明确说明“依据不足”。"
                    "答案请精炼，并在末尾给出引用编号，如 [1][2]。"
                )
            ),
            HumanMessage(
                content=(
                    f"原始问题：{question}\n"
                    f"改写检索问题：{rewritten_question}\n\n"
                    "可用上下文如下：\n"
                    f"{contexts}"
                )
            ),
        ]
    )
    return str(result.content)


def _retrieve(
    *,
    query: str,
    collection_name: str | None,
    knowledge_domain: str | None,
    book_id: str | None,
    top_k: int | None,
) -> tuple[list[tuple[Any, float]], list[dict[str, Any]]]:
    selected_collection = collection_name or RAG_CHROMA_COLLECTION
    selected_top_k = top_k or RAG_RETRIEVAL_TOP_K
    selected_knowledge_domain = (knowledge_domain or "").strip()
    selected_book_id = (book_id or "").strip()

    metadata_filter: dict[str, Any] | None = None
    filter_map: dict[str, Any] = {}
    if selected_knowledge_domain:
        filter_map["domain"] = selected_knowledge_domain
    if selected_book_id:
        filter_map["book_id"] = selected_book_id
    if filter_map:
        metadata_filter = filter_map

    docs_with_scores = search_chroma(
        query=query.strip(),
        top_k=selected_top_k,
        collection_name=selected_collection,
        metadata_filter=metadata_filter,
    )
    return docs_with_scores, build_citations(docs_with_scores)


async def route_decision_node(state: AdaptiveRagState) -> dict[str, Any]:
    model = get_chat_model().with_structured_output(RouteDecision)
    decision_raw = await model.ainvoke(
        [
            SystemMessage(content=ADAPTIVE_RAG_ROUTER_PROMPT),
            HumanMessage(content=f"用户问题：{state['question']}"),
        ]
    )

    if isinstance(decision_raw, RouteDecision):
        decision = decision_raw
    elif isinstance(decision_raw, BaseModel):
        decision = RouteDecision.model_validate(decision_raw.model_dump())
    elif isinstance(decision_raw, dict):
        decision = RouteDecision.model_validate(decision_raw)
    else:
        raise InvalidRequestError(message="LLM 路由结果格式非法", detail={"type": str(type(decision_raw))})

    route = str(decision.route)
    if route not in {
        "direct_answer",
        "fixed_rag",
        "agent_rag",
        "strict_insufficient",
    }:
        raise InvalidRequestError(message="LLM 路由结果非法", detail={"route": route})

    return {
        "route": route,
        "route_reason": decision.reason.strip() or "无",
    }


async def direct_answer_node(state: AdaptiveRagState) -> dict[str, Any]:
    model = get_chat_model()
    result = await model.ainvoke(
        [
            SystemMessage(content=ADAPTIVE_RAG_DIRECT_PROMPT),
            HumanMessage(content=state["question"]),
        ]
    )
    reason = _extract_route_reason(state)
    return {
        "answer": str(result.content),
        "citations": [],
        "retrieval_count": 0,
        "trace": _build_trace(route="direct_answer", reason=reason, retrieval_count=0),
    }


async def fixed_rag_node(state: AdaptiveRagState) -> dict[str, Any]:
    docs_with_scores, citations = _retrieve(
        query=state["question"],
        collection_name=state.get("collection_name"),
        knowledge_domain=state.get("knowledge_domain"),
        book_id=state.get("book_id"),
        top_k=state.get("top_k"),
    )

    if not docs_with_scores:
        answer = "没有检索到相关资料，当前无法基于知识库给出可靠答案。"
    else:
        contexts = _build_context_blocks(docs_with_scores)
        answer = await _answer_with_context(
            question=state["question"],
            rewritten_question=state["question"].strip(),
            contexts=contexts,
        )

    reason = _extract_route_reason(state)
    retrieval_count = 1
    return {
        "answer": answer,
        "citations": citations,
        "retrieval_count": retrieval_count,
        "trace": _build_trace(route="fixed_rag", reason=reason, retrieval_count=retrieval_count),
    }


async def _rewrite_query(question: str) -> str:
    model = get_chat_model().with_structured_output(QueryRewrite)
    rewrite_raw = await model.ainvoke(
        [
            SystemMessage(content=ADAPTIVE_RAG_REWRITE_PROMPT),
            HumanMessage(content=question),
        ]
    )

    if isinstance(rewrite_raw, QueryRewrite):
        rewrite_result = rewrite_raw
    elif isinstance(rewrite_raw, BaseModel):
        rewrite_result = QueryRewrite.model_validate(rewrite_raw.model_dump())
    elif isinstance(rewrite_raw, dict):
        rewrite_result = QueryRewrite.model_validate(cast(dict[str, Any], rewrite_raw))
    else:
        raise InvalidRequestError(message="LLM 改写结果格式非法", detail={"type": str(type(rewrite_raw))})

    rewritten = rewrite_result.rewritten_query.strip()
    if not rewritten:
        rewritten = question.strip()
    return rewritten


async def agent_rag_node(state: AdaptiveRagState) -> dict[str, Any]:
    reason = _extract_route_reason(state)

    runtime: dict[str, Any] = {
        "rewritten_question": state["question"].strip(),
        "docs_with_scores": [],
        "citations": [],
        "retrieval_count": 0,
    }

    @tool
    async def rewrite_query(query: str) -> str:
        """Rewrite user query into a concise retrieval-friendly query."""
        rewritten = await _rewrite_query(query)
        runtime["rewritten_question"] = rewritten
        return rewritten

    @tool
    def retrieve_context(query: str) -> str:
        """Retrieve relevant knowledge-base chunks for a query and return formatted context."""
        docs_with_scores, citations = _retrieve(
            query=query,
            collection_name=state.get("collection_name"),
            knowledge_domain=state.get("knowledge_domain"),
            book_id=state.get("book_id"),
            top_k=state.get("top_k"),
        )
        runtime["docs_with_scores"] = docs_with_scores
        runtime["citations"] = citations
        runtime["retrieval_count"] = int(runtime["retrieval_count"]) + 1
        if not docs_with_scores:
            return "NO_CONTEXT"
        return _build_context_blocks(docs_with_scores)

    model = get_chat_model()
    agent = create_agent(model=model, tools=[rewrite_query, retrieve_context])
    agent_result = await agent.ainvoke(
        {
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "你是企业知识库问答助手。"
                        "你可以使用两个工具：rewrite_query 与 retrieve_context。"
                        "如需知识库证据，必须先调用 retrieve_context 再回答。"
                        "若证据不足，请明确说明“依据不足”。"
                        "答案请精炼，并在末尾给出引用编号，如 [1][2]。"
                    ),
                },
                {"role": "user", "content": state["question"]},
            ]
        },
        config={"recursion_limit": 8},
    )

    final_messages = agent_result.get("messages", []) if isinstance(agent_result, dict) else []
    answer = _extract_final_text(final_messages)
    docs_with_scores = runtime["docs_with_scores"]
    citations = runtime["citations"]
    retrieval_count = int(runtime["retrieval_count"])
    rewritten_question = str(runtime["rewritten_question"])

    if not answer:
        if not docs_with_scores:
            answer = "没有检索到相关资料，当前无法基于知识库给出可靠答案。"
        else:
            contexts = _build_context_blocks(docs_with_scores)
            answer = await _answer_with_context(
                question=state["question"],
                rewritten_question=rewritten_question,
                contexts=contexts,
            )

    return {
        "rewritten_question": rewritten_question,
        "docs_with_scores": docs_with_scores,
        "citations": citations,
        "retrieval_count": retrieval_count,
        "answer": answer,
        "trace": _build_trace(
            route="agent_rag",
            reason=reason,
            retrieval_count=retrieval_count,
        ),
    }


async def strict_insufficient_node(state: AdaptiveRagState) -> dict[str, Any]:
    reason = _extract_route_reason(state)
    return {
        "answer": "依据不足，当前无法给出可靠答案。",
        "citations": [],
        "retrieval_count": 0,
        "trace": _build_trace(route="strict_insufficient", reason=reason, retrieval_count=0),
    }


def route_selector(state: AdaptiveRagState) -> RagRoute:
    route = state.get("route")
    if route in {
        "direct_answer",
        "fixed_rag",
        "agent_rag",
        "strict_insufficient",
    }:
        return route
    raise InvalidRequestError(message="缺少有效路由结果", detail={"route": route})
