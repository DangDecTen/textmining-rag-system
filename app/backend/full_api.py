from functools import lru_cache
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from generation.generator import Generator
from src.retrieval.retriever_factory import RetrieverFactory

DEFAULT_INDEX_DIR = Path("data/index/faiss_index")

class QueryRequest(BaseModel):
    query: str = Field(min_length=1)
    k: int = Field(default=5, ge=1, le=20)
    retriever: str = Field(default="dense")

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
    return Generator()

@app.get("/")
def root():
    return {"message": "API running"}
    
@app.get("/health")
def health():
    return {
        "status":
            "ok"
            if DEFAULT_INDEX_DIR.exists()
            else "missing_index"
    }

@app.post("/retrieve", response_model=RetrieveResponse)
def retrieve(request: QueryRequest):
    retriever = get_retriever(request.retriever)
    results = retriever.search(request.query, request.k)
    return RetrieveResponse(
        query=request.query,
        retriever=request.retriever,
        k=request.k,
        results=results
    )

@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    if not DEFAULT_INDEX_DIR.exists():
        raise HTTPException(
            status_code=503,
            detail="Index not found"
        )

    retriever = get_retriever(request.retriever)
    generator = get_generator()
    contexts = retriever.search(request.query, request.k)
    answer = generator.generate(request.query, contexts)

    return QueryResponse(
        query=request.query,
        retriever=request.retriever,
        answer=answer,
        sources=contexts
    )