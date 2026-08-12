from abc import ABC, abstractmethod

from src.data_models.data_models import RetrievalResult


class Reranker(ABC):

    @abstractmethod
    def rerank(self, query: str, results: list[RetrievalResult], top_k: int | None = None) -> list[RetrievalResult]:
        """Re-scores and re-sorts `results` for `query`, best-first.

        `top_k`: if given, truncate to this many after reranking (lets a
        caller pull a wide candidate set from retrieval, e.g. top_k=50, then
        rerank down to a narrow context, e.g. top_k=5). If None, returns all
        of `results` re-sorted.
        """
        pass
