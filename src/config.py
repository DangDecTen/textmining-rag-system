"""
Single source of truth for paths, model names, and defaults.

Previously these were hardcoded in four different places (run_rag.py,
run_bm25.py, run_dense.py, retriever_factory.py, api.py, full_api.py) and
had already drifted (e.g. two different "default index dir" constants
pointing at two different paths). Change a value here once; every entry
point (CLI scripts, API, tests) picks it up.

All values are overridable via environment variables / a `.env` file, e.g.:

    BM25_INDEX_DIR=data/index/bm25_v2
    DEFAULT_RETRIEVER=dense
    DEFAULT_TOP_K=8
"""
from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- Data ---
    dataset: str = "data/benchmark/attackqa.parquet"
    processed_data_dir: str = "data/processed"
    corpus_path: str = "data/processed/corpus.jsonl"
    qa_path_template: str = "data/processed/qa_{split}.jsonl"

    # --- Indexes ---
    bm25_index_dir: str = "data/index/bm25_k1_b25"
    dense_index_dir: str = "data/index/dense_bge_small"

    # --- Retrieval ---
    default_retriever: str = "bm25"
    default_top_k: int = 10

    # BM25 hyperparameters
    bm25_method: str = "lucene"
    bm25_k1: float = 1.0
    bm25_b: float = 0.25

    # Dense retrieval
    dense_model_name: str = "BAAI/bge-small-en-v1.5"  # "BAAI/bge-base-en-v1.5"
    dense_batch_size: int = 64

    # Hybrid retrieval (RRF / weighted-sum fusion of bm25 + dense)
    hybrid_alpha: float = 0.5  # weight given to dense; (1 - alpha) to bm25
    hybrid_rrf_k: int = 10
    hybrid_use_rrf: bool = False

    # --- Generation ---
    default_generator: str = "llama"
    llama_model_name: str = "llama-3.3-70b-versatile"
    qwen_model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    max_context_tokens: int = 1500
    max_new_tokens: int = 128

    # --- Reranking ---
    rerank_enabled: bool = True
    default_reranker: str = "cross_encoder"
    cross_encoder_model_name: str = "BAAI/bge-reranker-base"
    cross_encoder_batch_size: int = 32
    # How many candidates the retriever is asked for before reranking cuts
    # down to the requested top_k. Wider than top_k on purpose -- reranking
    # only helps if it has more to choose from than the final count.
    rerank_candidate_k: int = 20

    def index_dir_for(self, retriever_name: str) -> str:
        return {"bm25": self.bm25_index_dir, "dense": self.dense_index_dir}[retriever_name.lower()]


settings = Settings()
