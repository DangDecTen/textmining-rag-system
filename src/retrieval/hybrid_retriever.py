"""
Hybrid Retriever combining Dense (FAISS + BGE embeddings) and Lexical (BM25s) retrieval.
Uses Reciprocal Rank Fusion (RRF) by default to merge candidate rankings.
"""

from __future__ import annotations

from src.data_models.data_models import Document, RetrievalResult
from src.retrieval.base import Retriever
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.dense_retriever import DenseRetriever


class HybridRetriever(Retriever):
    def __init__(
        self,
        dense_retriever: DenseRetriever,
        bm25_retriever: BM25Retriever,
        alpha: float = 0.5,
        rrf_k: int = 60,
        use_rrf: bool = True,
    ):
        """
        alpha: weight given to dense retriever (1-alpha given to bm25). Default 0.5.
        rrf_k: constant k parameter in RRF formula score = alpha / (k + rank_dense) + (1-alpha) / (k + rank_bm25).
        use_rrf: if True uses RRF, otherwise uses min-max normalized weighted sum.
        """
        self.dense_retriever = dense_retriever
        self.bm25_retriever = bm25_retriever
        self.alpha = alpha
        self.rrf_k = rrf_k
        self.use_rrf = use_rrf
        self.corpus_lookup = dense_retriever.corpus_lookup

    def search(self, query: str, top_k: int = 10) -> list[RetrievalResult]:
        candidate_k = max(top_k * 3, 50)
        dense_results = self.dense_retriever.search(query, top_k=candidate_k)
        bm25_results = self.bm25_retriever.search(query, top_k=candidate_k)

        if self.use_rrf:
            return self._rrf_fusion(dense_results, bm25_results, top_k=top_k)
        else:
            return self._weighted_score_fusion(dense_results, bm25_results, top_k=top_k)

    def _rrf_fusion(
        self,
        dense_results: list[RetrievalResult],
        bm25_results: list[RetrievalResult],
        top_k: int,
    ) -> list[RetrievalResult]:
        rrf_scores: dict[str, float] = {}

        for rank, res in enumerate(dense_results, start=1):
            doc_id = res.doc_id
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (self.alpha / (self.rrf_k + rank))

        for rank, res in enumerate(bm25_results, start=1):
            doc_id = res.doc_id
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + ((1.0 - self.alpha) / (self.rrf_k + rank))

        sorted_doc_ids = sorted(rrf_scores.keys(), key=lambda d: rrf_scores[d], reverse=True)[:top_k]

        return [
            RetrievalResult(
                doc_id=doc_id,
                score=float(rrf_scores[doc_id]),
                document=self.corpus_lookup.get(doc_id),
            )
            for doc_id in sorted_doc_ids
        ]

    def _weighted_score_fusion(
        self,
        dense_results: list[RetrievalResult],
        bm25_results: list[RetrievalResult],
        top_k: int,
    ) -> list[RetrievalResult]:
        def normalize(results: list[RetrievalResult]) -> dict[str, float]:
            if not results:
                return {}
            scores = [r.score for r in results]
            min_s, max_s = min(scores), max(scores)
            if max_s == min_s:
                return {r.doc_id: 1.0 for r in results}
            return {r.doc_id: (r.score - min_s) / (max_s - min_s) for r in results}

        dense_norm = normalize(dense_results)
        bm25_norm = normalize(bm25_results)

        all_doc_ids = set(dense_norm.keys()) | set(bm25_norm.keys())
        combined_scores: dict[str, float] = {}
        for doc_id in all_doc_ids:
            s_dense = dense_norm.get(doc_id, 0.0)
            s_bm25 = bm25_norm.get(doc_id, 0.0)
            combined_scores[doc_id] = self.alpha * s_dense + (1.0 - self.alpha) * s_bm25

        sorted_doc_ids = sorted(combined_scores.keys(), key=lambda d: combined_scores[d], reverse=True)[:top_k]

        return [
            RetrievalResult(
                doc_id=doc_id,
                score=float(combined_scores[doc_id]),
                document=self.corpus_lookup.get(doc_id),
            )
            for doc_id in sorted_doc_ids
        ]
