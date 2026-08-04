from functools import lru_cache
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.generation.llama_generator import LlamaGenerator
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
    answer: str
    sources: list[dict]


class RetrieveResponse(BaseModel):
    query: str
    retriever: str
    k: int
    results: list[dict]


app = FastAPI(
    title="Text Mining RAG System",
    version="1.0.0"
)


@lru_cache(maxsize=10)
def get_retriever(retriever_type: str):
    return RetrieverFactory.create(retriever_type)


@lru_cache(maxsize=1)
def get_generator():
    return LlamaGenerator()


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


@app.get("/")
def root():
    return {"message": "API running"}


@app.get("/health")
def health():
    has_index = DEFAULT_DENSE_INDEX_DIR.exists() or DEFAULT_BM25_INDEX_DIR.exists()
    return {
        "status": "ok" if has_index else "missing_index"
    }


@app.post("/retrieve", response_model=RetrieveResponse)
def retrieve(request: QueryRequest):
    try:
        retriever = get_retriever(request.retriever)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    raw_results = retriever.search(request.query, top_k=request.k)
    formatted = [_format_result(r) for r in raw_results]

    return RetrieveResponse(
        query=request.query,
        retriever=request.retriever,
        k=request.k,
        results=formatted
    )


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    if not DEFAULT_DENSE_INDEX_DIR.exists() and not DEFAULT_BM25_INDEX_DIR.exists():
        raise HTTPException(
            status_code=503,
            detail="Index directories not found."
        )

    try:
        retriever = get_retriever(request.retriever)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    generator = get_generator()
    contexts = retriever.search(request.query, top_k=request.k)
    gen_result = generator.generate(request.query, contexts)

    answer_text = gen_result.answer if hasattr(gen_result, "answer") else gen_result.get("answer", "")
    formatted_sources = [_format_result(r) for r in contexts]

    return QueryResponse(
        query=request.query,
        retriever=request.retriever,
        answer=answer_text,
        sources=formatted_sources
    )