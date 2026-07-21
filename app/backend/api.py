"""FastAPI service for testing the dense retrieval pipeline."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.retrieval.retriever import DenseRetriever

DEFAULT_INDEX_DIR = Path("data/index/faiss_index")


class QueryRequest(BaseModel):
    query: str = Field(min_length=1)
    k: int = Field(default=5, ge=1, le=20)


class QueryResponse(BaseModel):
    query: str
    k: int
    results: list[dict]


app = FastAPI(title="Textmining RAG System", version="0.1.0")


@lru_cache(maxsize=1)
def get_retriever(index_dir: str = str(DEFAULT_INDEX_DIR)) -> DenseRetriever:
    return DenseRetriever(index_dir)


@app.get("/health")
def health() -> dict:
    index_dir = DEFAULT_INDEX_DIR
    return {
        "status": "ok" if index_dir.exists() else "missing_index",
        "index_dir": str(index_dir),
    }


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    if not DEFAULT_INDEX_DIR.exists():
        raise HTTPException(status_code=503, detail="FAISS index not found. Build it first.")

    retriever = get_retriever()
    results = retriever.search(request.query, k=request.k)
    return QueryResponse(query=request.query, k=request.k, results=results)
