"""
Cross-Encoder Re-ranking Retriever.
Implements a 2-stage retrieval approach:
Stage 1: Base Retriever (Hybrid RRF / Dense) fetches Top-N candidates (e.g. N=50).
Stage 2: Cross-Encoder scores (query, passage) pairs using full joint self-attention,
         re-sorting candidates to produce Top-K results.
"""

from __future__ import annotations

import os
from typing import Sequence
from dotenv import load_dotenv

from src.data_models.data_models import Document, RetrievalResult
from src.retrieval.base import Retriever

load_dotenv()

if hf_token := os.getenv("HF_TOKEN"):
    os.environ["HF_TOKEN"] = hf_token


class CrossEncoderRetriever(Retriever):
    def __init__(
        self,
        base_retriever: Retriever,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        candidate_k: int = 50,
        device: str | None = None,
        corpus_lookup: dict[str, Document] | None = None,
    ):
        self.base_retriever = base_retriever
        self.model_name = model_name
        self.candidate_k = candidate_k
        self.corpus_lookup = corpus_lookup or getattr(base_retriever, "corpus_lookup", {})
        
        self.model = None
        self._model_name = model_name
        self._device = device

    def _get_model(self):
        if self.model is None:
            try:
                from sentence_transformers import CrossEncoder
                self.model = CrossEncoder(self._model_name, device=self._device)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load CrossEncoder model '{self._model_name}': {e}"
                )
        return self.model

    def search(self, query: str, top_k: int = 10) -> list[RetrievalResult]:
        # Stage 1: Candidate retrieval
        initial_k = max(top_k, self.candidate_k)
        candidates = self.base_retriever.search(query, top_k=initial_k)

        if not candidates:
            return []

        # Prepare (query, doc_text) pairs
        valid_candidates = []
        pairs = []

        for cand in candidates:
            doc = cand.document or self.corpus_lookup.get(cand.doc_id)
            if doc and doc.text:
                valid_candidates.append(cand)
                pairs.append((query, doc.text))

        if not pairs:
            return candidates[:top_k]

        # Stage 2: Cross-Encoder Scoring
        model = self._get_model()
        scores = model.predict(pairs, show_progress_bar=False, batch_size=32)

        # Re-sort candidates by cross-encoder score
        scored_candidates = []
        for cand, score in zip(valid_candidates, scores):
            doc = cand.document or self.corpus_lookup.get(cand.doc_id)
            scored_candidates.append(
                RetrievalResult(
                    doc_id=cand.doc_id,
                    score=float(score),
                    document=doc,
                )
            )

        scored_candidates.sort(key=lambda x: x.score, reverse=True)
        return scored_candidates[:top_k]
