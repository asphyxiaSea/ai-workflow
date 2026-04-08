from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.application.pipelines.rag_chat_pipeline import run_rag_chat_pipeline
from app.core.settings import RAG_DEFAULT_KNOWLEDGE_DOMAIN
from app.core.errors import ExternalServiceError, InvalidRequestError


router = APIRouter(tags=["rag"])


class RagChatRequest(BaseModel):
    question: str = Field(min_length=1)
    collection_name: str | None = None
    knowledge_domain: str = Field(default=RAG_DEFAULT_KNOWLEDGE_DOMAIN, min_length=1)
    book_id: str | None = None
    top_k: int | None = Field(default=None, ge=1, le=20)


@router.post("/rag/chat")
async def rag_chat(body: RagChatRequest) -> dict[str, Any]:
    try:
        result = await run_rag_chat_pipeline(
            question=body.question,
            collection_name=body.collection_name,
            knowledge_domain=body.knowledge_domain,
            book_id=body.book_id,
            top_k=body.top_k,
        )
        return result
    except InvalidRequestError:
        raise
    except Exception as exc:
        raise ExternalServiceError(message="RAG 问答失败", detail=str(exc)) from exc
