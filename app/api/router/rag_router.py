from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter
from pydantic import BaseModel, Field, model_validator

from app.application.task_dispatcher import TaskType, get_task_dispatcher_service
from app.core.settings import RAG_DEFAULT_KNOWLEDGE_DOMAIN
from app.core.errors import AppError, ExternalServiceError, InvalidRequestError


router = APIRouter(tags=["rag"])


class RagChatRequest(BaseModel):
    messages: list["RagMessage"] = Field(min_length=1)
    thread_id: str = Field(min_length=1)
    user_id: str = Field(min_length=1)
    collection_name: str | None = None
    knowledge_domain: str = Field(default=RAG_DEFAULT_KNOWLEDGE_DOMAIN, min_length=1)
    book_id: str | None = None
    top_k: int | None = Field(default=None, ge=1, le=20)

    @model_validator(mode="after")
    def validate_messages(self) -> "RagChatRequest":
        has_valid_message = any(str(message.content).strip() for message in self.messages)
        if not has_valid_message:
            raise ValueError("messages 中至少需要一条非空 user 消息")
        return self


class RagMessage(BaseModel):
    role: Literal["user"]
    content: str = Field(min_length=1)


@router.post("/rag/chat")
async def rag_chat(body: RagChatRequest) -> dict[str, Any]:
    try:
        dispatcher = get_task_dispatcher_service()
        task_id = await dispatcher.submit_task(
            task_type=TaskType.RAG_CHAT,
            payload={
                "messages": [message.model_dump() for message in body.messages],
                "thread_id": body.thread_id,
                "user_id": body.user_id,
                "collection_name": body.collection_name,
                "knowledge_domain": body.knowledge_domain,
                "book_id": body.book_id,
                "top_k": body.top_k,
            },
        )
        return {
            "task_id": task_id,
            "status": "PENDING",
        }
    except InvalidRequestError:
        raise
    except AppError:
        raise
    except Exception as exc:
        raise ExternalServiceError(message="RAG 任务提交失败", detail=str(exc)) from exc


@router.get("/rag/chat/tasks/{task_id}")
async def rag_chat_task_status(task_id: str) -> dict[str, Any]:
    dispatcher = get_task_dispatcher_service()
    return await dispatcher.get_task_snapshot(task_id)


@router.get("/rag/chat/tasks/{task_id}/result")
async def rag_chat_task_result(task_id: str) -> dict[str, Any]:
    dispatcher = get_task_dispatcher_service()
    task = await dispatcher.get_task_snapshot(task_id)
    status = task["status"]

    if status in ("PENDING", "RUNNING"):
        return {
            "task_id": task_id,
            "status": status,
            "message": "任务尚未完成",
        }

    if status == "FAILED":
        return {
            "task_id": task_id,
            "status": status,
            "error": task.get("error", "任务执行失败"),
        }

    result = task.get("result") or {}
    return {
        "task_id": task_id,
        "status": status,
        "answer": result.get("answer", ""),
        "citations": result.get("citations", []),
        "trace": result.get("trace", {}),
    }
