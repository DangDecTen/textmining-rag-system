from __future__ import annotations

from src.data_models.data_models import Document, RetrievalResult
from src.indexing.dense_index import DenseIndex
from src.retrieval.base import Retriever


class DenseRetriever(Retriever):
    def __init__(self, index: DenseIndex, corpus_lookup: dict[str, Document]):
        self.index = index
        self.corpus_lookup = corpus_lookup

    def search(self, query: str, top_k: int = 10) -> list[RetrievalResult]:
        query_embedding = self.index.embedder.embed_query(query)  # shape (1, dim)
        scores, indices = self.index.index.search(query_embedding, top_k)  # faiss: (D, I)

        results = []
        for row_idx, score in zip(indices[0], scores[0]):
            if row_idx == -1:  # faiss pads with -1 if fewer than top_k results exist
                continue
            doc_id = self.index.doc_ids[row_idx]
            results.append(
                RetrievalResult(doc_id=doc_id, score=float(score), document=self.corpus_lookup.get(doc_id))
            )
        return results
