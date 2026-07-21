"""
BM25 Retriever built on the `bm25s` library.
- Tokenize question using BM25 Index's Tokenizer
- Search for top-k
"""

from __future__ import annotations

from src.data_models.data_models import Document, RetrievalResult
from src.indexing.bm25_index import BM25Index
from src.retrieval.base import Retriever


class BM25Retriever(Retriever):
    def __init__(self, index: BM25Index, corpus_lookup: dict[str, Document]):
        """corpus_lookup: doc_id -> Document, typically loaded from corpus.jsonl."""
        self.index = index
        self.corpus_lookup = corpus_lookup

    def search(self, query: str, top_k: int = 10) -> list[RetrievalResult]:
        # update_vocab=False: query terms not seen in the corpus are simply
        # out-of-vocabulary and contribute no match -- correct IR behavior,
        # not a bug. Explicit here even though the Tokenizer's default is
        # already "only update vocab on first call" (which was the corpus).
        query_tokens = self.index.tokenizer.tokenize([query], update_vocab=False)

        results, scores = self.index.model.retrieve(
            query_tokens, corpus=self.index.doc_ids, k=top_k, return_as="tuple"
        )
        doc_ids_row, scores_row = results[0], scores[0]

        return [
            RetrievalResult(
                doc_id=doc_id['text'],
                score=float(score),
                document=self.corpus_lookup.get(doc_id['text']),
            )
            for doc_id, score in zip(doc_ids_row, scores_row)
        ]
