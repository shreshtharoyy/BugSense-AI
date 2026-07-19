from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.serving.retrieval_service import get_retrieval_service


class SearchRequest(BaseModel):
    bug_id: str | None = None
    query_text: str | None = None
    top_k: int = Field(default=5, ge=1, le=10)


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_retrieval_service()
    yield


app = FastAPI(
    title="BugSense AI API",
    version="0.1.0",
    description="Search similar historical bugs using the champion hybrid retriever.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict:
    service = get_retrieval_service()
    return {"status": "ok", **service.corpus_summary()}


@app.get("/api/bugs/{bug_id}")
def get_bug(bug_id: str) -> dict:
    service = get_retrieval_service()
    bug = service.get_bug(bug_id)
    if bug is None:
        raise HTTPException(status_code=404, detail=f"Bug #{bug_id} not found in index.")
    return bug


@app.post("/api/search")
def search(payload: SearchRequest) -> dict:
    service = get_retrieval_service()

    if payload.bug_id:
        try:
            return service.search_by_bug_id(payload.bug_id.strip(), top_k=payload.top_k)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    if payload.query_text:
        try:
            return service.search_by_text(payload.query_text, top_k=payload.top_k)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    raise HTTPException(
        status_code=400,
        detail="Provide either bug_id or query_text.",
    )
