"""
Learned Sparse Retrieval (LSR) using BAAI/bge-m3.

Optimized Architecture:
- Two-stage pipeline:
  - Stage 1: Fast candidate generation (Hybrid / BM25 / Dense) to retrieve Top-N candidates (N=10-15).
  - Stage 2: BGE-M3 Learned Sparse lexical weight extraction & neural inner-product scoring.
- Optimization for CPU:
  - Multithreading: torch.set_num_threads(os.cpu_count())
  - Passage caching: in-memory cache for passage lexical weights (0ms for repeated passages)
  - Sequence truncation: max_length=192 for passages, max_length=64 for queries
  - Batch inference: batch_size=16
- Provides Explainable AI (XAI): inspect exact token contributions to retrieval score.
"""

from __future__ import annotations

import os
import torch
from typing import Any

from src.data_models.data_models import Document, RetrievalResult
from src.retrieval.base import Retriever
from src.retrieval.registry import register_retriever


class BGEM3SparseEngine:
    """Singleton wrapper for BGE-M3 model with in-memory caching and CPU optimizations."""
    _instance: BGEM3SparseEngine | None = None

    def __init__(self, model_name: str = "BAAI/bge-m3", device: str | None = None):
        from FlagEmbedding import BGEM3FlagModel

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model_name = model_name

        if device == "cpu":
            num_cores = os.cpu_count() or 4
            torch.set_num_threads(num_cores)

        self.model = BGEM3FlagModel(model_name, use_fp16=(device == "cuda"), device=device)
        self._doc_cache: dict[str, dict[str, float]] = {}

    @classmethod
    def get_instance(cls, model_name: str = "BAAI/bge-m3", device: str | None = None) -> BGEM3SparseEngine:
        if cls._instance is None:
            cls._instance = cls(model_name=model_name, device=device)
        return cls._instance

    def encode_query(self, query: str) -> dict[str, float]:
        """Extract neural lexical weights for a query with short max_length."""
        out = self.model.encode([query], return_dense=False, return_sparse=True, return_colbert_vecs=False, max_length=64)
        return out["lexical_weights"][0]

    def encode_passages(self, doc_ids: list[str], passages: list[str]) -> list[dict[str, float]]:
        """Extract neural lexical weights with in-memory caching and batching."""
        results: list[dict[str, float]] = [None] * len(passages)  # type: ignore
        to_encode_indices = []
        to_encode_texts = []

        for idx, (doc_id, text) in enumerate(zip(doc_ids, passages)):
            if doc_id in self._doc_cache:
                results[idx] = self._doc_cache[doc_id]
            else:
                to_encode_indices.append(idx)
                to_encode_texts.append(text)

        if to_encode_texts:
            out = self.model.encode(
                to_encode_texts,
                return_dense=False,
                return_sparse=True,
                return_colbert_vecs=False,
                batch_size=16,
                max_length=192,
            )
            encoded_weights = out["lexical_weights"]
            for orig_idx, weights in zip(to_encode_indices, encoded_weights):
                doc_id = doc_ids[orig_idx]
                self._doc_cache[doc_id] = weights
                results[orig_idx] = weights

        return results

    def compute_scores(self, q_weights: dict[str, float], doc_weights_list: list[dict[str, float]]) -> list[float]:
        """Compute sparse inner-product matching scores."""
        scores = []
        for d_w in doc_weights_list:
            s = self.model.compute_lexical_matching_score(q_weights, d_w)
            scores.append(float(s))
        return scores


@register_retriever("learned_sparse")
class LearnedSparseRetriever(Retriever):
    def __init__(
        self,
        base_retriever: Retriever,
        model_name: str = "BAAI/bge-m3",
        candidate_k: int = 15,
        device: str | None = None,
        corpus_lookup: dict[str, Document] | None = None,
    ):
        self.base_retriever = base_retriever
        self.candidate_k = candidate_k
        self.corpus_lookup = corpus_lookup or getattr(base_retriever, "corpus_lookup", {})
        self.engine = BGEM3SparseEngine.get_instance(model_name=model_name, device=device)

    def search(self, query: str, top_k: int = 10) -> list[RetrievalResult]:
        initial_k = max(top_k, self.candidate_k)
        candidates = self.base_retriever.search(query, top_k=initial_k)

        if not candidates:
            return []

        valid_doc_ids = []
        valid_candidates = []
        passages = []

        for cand in candidates:
            doc = cand.document or self.corpus_lookup.get(cand.doc_id)
            if doc and doc.text:
                valid_doc_ids.append(cand.doc_id)
                valid_candidates.append((cand.doc_id, doc))
                passages.append(doc.text)

        if not passages:
            return candidates[:top_k]

        # Extract sparse representations with caching
        q_weights = self.engine.encode_query(query)
        passages_weights = self.engine.encode_passages(valid_doc_ids, passages)

        # Compute neural lexical matching scores
        scores = self.engine.compute_scores(q_weights, passages_weights)

        scored_results = [
            RetrievalResult(doc_id=doc_id, score=score, document=doc)
            for (doc_id, doc), score in zip(valid_candidates, scores)
        ]

        scored_results.sort(key=lambda r: r.score, reverse=True)
        return scored_results[:top_k]

    def explain_query_matching(self, query: str, top_tokens: int = 8) -> dict[str, Any]:
        """Explain the learned token expansion and weights for a query."""
        q_weights = self.engine.encode_query(query)
        sorted_tokens = sorted(q_weights.items(), key=lambda x: x[1], reverse=True)[:top_tokens]
        return {
            "query": query,
            "num_active_tokens": len(q_weights),
            "top_expanded_terms": sorted_tokens,
        }
