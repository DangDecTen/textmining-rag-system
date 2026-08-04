"""FastAPI service for testing retrieval pipeline."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.retrieval.base import Retriever
from src.retrieval.retriever_factory import RetrieverFactory

DEFAULT_DENSE_INDEX_DIR = Path("data/index/dense")
DEFAULT_BM25_INDEX_DIR = Path("data/index/bm25")


class QueryRequest(BaseModel):
    query: str = Field(min_length=1)
    k: int = Field(default=5, ge=1, le=20)
    retriever: str = Field(default="hybrid")


class QueryResponse(BaseModel):
    query: str
    retriever: str
    k: int
    results: list[dict]


app = FastAPI(title="Textmining RAG System", version="0.1.0")


@lru_cache(maxsize=3)
def get_retriever(retriever_type: str = "hybrid") -> Retriever:
    return RetrieverFactory.create(retriever_type)


@app.get("/health")
def health() -> dict:
    has_dense = DEFAULT_DENSE_INDEX_DIR.exists()
    has_bm25 = DEFAULT_BM25_INDEX_DIR.exists()
    return {
        "status": "ok" if (has_dense or has_bm25) else "missing_index",
        "dense_index": has_dense,
        "bm25_index": has_bm25,
    }


def _format_result(r) -> dict:
    doc = r.document
    return {
        "doc_id": r.doc_id,
        "chunk_id": r.doc_id,
        "score": r.score,
        "text": doc.text if doc else "",
        "subject_id": doc.subject_id if doc else "",
        "name": doc.subject_name or (doc.subject_id if doc else ""),
        "source": doc.source if doc else "",
        "url": doc.url if doc else "",
    }


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    if not DEFAULT_DENSE_INDEX_DIR.exists() and not DEFAULT_BM25_INDEX_DIR.exists():
        raise HTTPException(status_code=503, detail="Index directories not found. Build indices first.")

    try:
        retriever = get_retriever(request.retriever)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    raw_results = retriever.search(request.query, top_k=request.k)
    formatted_results = [_format_result(r) for r in raw_results]

    return QueryResponse(
        query=request.query,
        retriever=request.retriever,
        k=request.k,
        results=formatted_results,
    )
