"""
ColBERTv2 Late-Interaction Reranker.

Reranks candidate retrieval results using exact token-level MaxSim matching:
    Score(Q, D) = sum_{q in Q} max_{d in D} (E_q . E_d)
"""

from __future__ import annotations

from src.data_models.data_models import RetrievalResult
from src.indexing.colbert_engine import ColBERTv2Engine
from src.reranking.base import Reranker
from src.reranking.registry import register_reranker


@register_reranker("colbert")
class ColBERTReranker(Reranker):
    def __init__(
        self,
        model_name: str = "colbert-ir/colbertv2.0",
        device: str | None = None,
    ):
        self.engine = ColBERTv2Engine.get_instance(model_name=model_name, device=device)

    def rerank(
        self, query: str, results: list[RetrievalResult], top_k: int | None = None
    ) -> list[RetrievalResult]:
        if not results:
            return results

        scorable = [r for r in results if r.document is not None and r.document.text]
        if not scorable:
            return results[:top_k] if top_k is not None else results

        passages = [r.document.text for r in scorable]
        scores = self.engine.score_query_against_passages(query, passages)

        reranked = [
            RetrievalResult(doc_id=r.doc_id, score=score, document=r.document)
            for r, score in zip(scorable, scores)
        ]
        reranked.sort(key=lambda r: r.score, reverse=True)

        return reranked[:top_k] if top_k is not None else reranked
