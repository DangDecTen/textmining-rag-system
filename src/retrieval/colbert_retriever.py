"""
ColBERTv2 Late-Interaction Retriever.

Two-stage architecture:
- Stage 1: Fast candidate retrieval (Hybrid / Dense / BM25) to retrieve Top-N candidates (e.g. N=50).
- Stage 2: ColBERTv2 token-level Late-Interaction MaxSim scoring to re-rank candidates into Top-K.
"""

from __future__ import annotations

from src.data_models.data_models import Document, RetrievalResult
from src.indexing.colbert_engine import ColBERTv2Engine
from src.retrieval.base import Retriever
from src.retrieval.registry import register_retriever


@register_retriever("colbert")
class ColBERTRetriever(Retriever):
    def __init__(
        self,
        base_retriever: Retriever,
        model_name: str = "colbert-ir/colbertv2.0",
        candidate_k: int = 50,
        device: str | None = None,
        corpus_lookup: dict[str, Document] | None = None,
    ):
        self.base_retriever = base_retriever
        self.candidate_k = candidate_k
        self.corpus_lookup = corpus_lookup or getattr(base_retriever, "corpus_lookup", {})
        self.engine = ColBERTv2Engine.get_instance(model_name=model_name, device=device)

    def search(self, query: str, top_k: int = 10) -> list[RetrievalResult]:
        initial_k = max(top_k, self.candidate_k)
        candidates = self.base_retriever.search(query, top_k=initial_k)

        if not candidates:
            return []

        # Prepare passages
        valid_candidates = []
        passages = []

        for cand in candidates:
            doc = cand.document or self.corpus_lookup.get(cand.doc_id)
            if doc and doc.text:
                valid_candidates.append((cand.doc_id, doc))
                passages.append(doc.text)

        if not passages:
            return candidates[:top_k]

        # Compute MaxSim scores
        scores = self.engine.score_query_against_passages(query, passages)

        # Build scored results
        scored_results = [
            RetrievalResult(doc_id=doc_id, score=score, document=doc)
            for (doc_id, doc), score in zip(valid_candidates, scores)
        ]

        scored_results.sort(key=lambda r: r.score, reverse=True)
        return scored_results[:top_k]
