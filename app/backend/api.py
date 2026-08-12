"""
FastAPI service for the RAG system.

Endpoints:
    GET  /             -> service info + which retrievers/generators are registered
    GET  /health        -> whether the configured indexes exist on disk
    POST /retrieve       -> retrieval only (debugging retrieval quality, no generation)
    POST /query           -> full pipeline: retrieve + generate + build a cited answer

Retriever/generator selection and defaults all come from `src.config.settings`,
and objects are built through `src.factory` -- the same module `run_rag.py`
uses. That means the CLI and the API can never silently diverge the way the
old `api.py` / `full_api.py` / `retriever_factory.py` did (broken imports,
mismatched constructor signatures, a `k=` vs `top_k=` mismatch).

Run:
    python -m uvicorn app.backend.api:app --reload
"""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.config import settings
from src.data_models.data_models import Citation, RetrievalResult
from src.factory import available_generators, available_retrievers, get_pipeline, get_retriever

app = FastAPI(title="Textmining RAG System", version="1.0.0")


class QueryRequest(BaseModel):
    query: str = Field(min_length=1)
    k: int = Field(default=settings.default_top_k, ge=1, le=50)
    retriever: str = Field(default=settings.default_retriever)
    generator: str = Field(default=settings.default_generator)


class RetrieveResponse(BaseModel):
    query: str
    retriever: str
    k: int
    results: list[RetrievalResult]


class QueryResponse(BaseModel):
    query: str
    retriever: str
    generator: str
    answer: str
    abstained: bool
    citations: list[Citation]
    retrieved_context: list[RetrievalResult]
    latency_ms: float
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


@app.get("/")
def root() -> dict:
    return {
        "message": "Textmining RAG System API",
        "available_retrievers": available_retrievers(),
        "available_generators": available_generators(),
        "defaults": {"retriever": settings.default_retriever, "generator": settings.default_generator},
    }


@app.get("/health")
def health() -> dict:
    """Reports whether the *configured default* retriever's index exists on
    disk. Doesn't load anything -- just checks the path, so it's safe to call
    frequently (e.g. from a container healthcheck)."""
    index_dir = Path(settings.index_dir_for(settings.default_retriever))
    return {
        "status": "ok" if index_dir.exists() else "missing_index",
        "default_retriever": settings.default_retriever,
        "index_dir": str(index_dir),
    }


@app.post("/retrieve", response_model=RetrieveResponse)
def retrieve(request: QueryRequest) -> RetrieveResponse:
    try:
        retriever = get_retriever(request.retriever)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Index for '{request.retriever}' not found on disk. Build it first "
            f"(see README) -- {e}",
        ) from e

    results = retriever.search(request.query, top_k=request.k)
    return RetrieveResponse(query=request.query, retriever=request.retriever, k=request.k, results=results)


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    try:
        pipeline = get_pipeline(retriever_name=request.retriever, generator_name=request.generator)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Index for '{request.retriever}' not found on disk. Build it first "
            f"(see README) -- {e}",
        ) from e

    answer, generation_result = pipeline.answer_with_debug(request.query, top_k=request.k)
    return QueryResponse(
        query=request.query,
        retriever=request.retriever,
        generator=request.generator,
        answer=answer.text,
        abstained=answer.abstained,
        citations=answer.citations,
        retrieved_context=generation_result.retrieval_results,
        latency_ms=generation_result.latency_ms,
        prompt_tokens=generation_result.prompt_tokens,
        completion_tokens=generation_result.completion_tokens,
    )
