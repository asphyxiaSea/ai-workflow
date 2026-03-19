from __future__ import annotations

from typing import cast

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api.router.pdf_structured_router import router as pdf_structured_router
from app.core.errors import AppError


async def app_error_handler(request: Request, exc: Exception) -> JSONResponse:
    app_exc = cast(AppError, exc)
    return JSONResponse(
        status_code=app_exc.status_code,
        content={
            "error": {
                "code": app_exc.code,
                "message": app_exc.message,
                "detail": app_exc.detail,
            }
        },
    )


async def health() -> dict[str, str]:
    return {"status": "ok"}


def create_app() -> FastAPI:
    app = FastAPI(title="langchain app")
    app.add_exception_handler(AppError, app_error_handler)
    app.get("/health")(health)
    app.include_router(pdf_structured_router, prefix="/ai-agent")
    return app


app = create_app()