"""
Cross-Encoder Reranker built on `sentence_transformers.CrossEncoder`.

Unlike bm25/dense retrieval, which score query and document independently
(so a document's embedding can be precomputed once, offline), a
cross-encoder scores a query and a document *jointly* -- the pair is
concatenated and run through the model together. That's why this class
takes candidates already narrowed down by a retriever rather than
searching the whole corpus itself: it's far more accurate per pair, but
too slow to run over every document in the index.

Design decisions:
- **Model**: `BAAI/bge-reranker-base`, loaded via `sentence_transformers`'s
  `CrossEncoder` (consistent with how `DenseRetriever` uses
  `sentence-transformers` rather than raw `transformers` -- lower bug
  risk, and no new dependency since `sentence-transformers` is already a
  requirement).
- **Score semantics**: `CrossEncoder.predict()` returns raw relevance
  logits for `bge-reranker-base` (no sigmoid applied). We don't rescale
  them -- only their *order* matters here, since `rerank()` just re-sorts
  by score. Don't compare these scores against retriever scores (BM25 /
  cosine similarity); they're on an unrelated scale.
- **Batching**: all (query, doc_text) pairs for one `rerank()` call are
  scored via a single `predict()` call with an internal batch size, rather
  than one-by-one, since that's what actually uses the model efficiently
  (batched forward passes) -- scoring one pair at a time would work but
  defeats the point of batching on GPU/CPU.
"""

from __future__ import annotations

from src.data_models.data_models import RetrievalResult
from src.reranking.base import Reranker
from src.reranking.registry import register_reranker


@register_reranker("cross_encoder")
class CrossEncoderReranker(Reranker):
    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-base",
        batch_size: int = 32,
        device: str | None = None,
    ):
        # Imported lazily so importing this module (e.g. for registration
        # via factory.py's side-effect imports) doesn't force a
        # sentence-transformers/torch import -- and therefore a model
        # download attempt -- until a CrossEncoderReranker is actually
        # constructed.
        from sentence_transformers import CrossEncoder

        self.model_name = model_name
        self.batch_size = batch_size
        self.model = CrossEncoder(model_name, device=device)

    def rerank(self, query: str, results: list[RetrievalResult], top_k: int | None = None) -> list[RetrievalResult]:
        if not results:
            return results

        # Documents missing from corpus_lookup (e.g. a stale doc_id) have no
        # text to score -- skip them rather than crashing the whole rerank.
        scorable = [r for r in results if r.document is not None]
        if not scorable:
            return results[:top_k] if top_k is not None else results

        pairs = [(query, r.document.text) for r in scorable]
        scores = self.model.predict(pairs, batch_size=self.batch_size)

        reranked = [
            RetrievalResult(doc_id=r.doc_id, score=float(score), document=r.document)
            for r, score in zip(scorable, scores)
        ]
        reranked.sort(key=lambda r: r.score, reverse=True)

        return reranked[:top_k] if top_k is not None else reranked
