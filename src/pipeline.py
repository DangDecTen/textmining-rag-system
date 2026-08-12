"""
Combines the components built in earlier stages into a single call:
Retriever.search() -> [Reranker.rerank()] -> Generator.generate()
"""
from __future__ import annotations

from src.config import settings
from src.data_models.data_models import Answer, GenerationResult
from src.generation.base import Generator
from src.generation.response_builder import ResponseBuilder
from src.reranking.base import Reranker
from src.retrieval.base import Retriever


class Pipeline:
    def __init__(self, retriever: Retriever, generator: Generator, reranker: Reranker | None = None):
        self.retriever = retriever
        self.generator = generator
        self.reranker = reranker
        self.response_builder = ResponseBuilder()

    def answer(self, question: str, top_k: int = 10) -> Answer:
        return self.answer_with_debug(question, top_k=top_k)[0]

    def answer_with_debug(self, question: str, top_k: int = 10) -> tuple[Answer, GenerationResult]:
        # If reranking, over-fetch from the retriever (rerank_candidate_k,
        # or top_k itself if that's already larger) so the reranker has a
        # meaningfully wider pool than top_k to pick from -- reranking a
        # list that's already been cut down to top_k just reorders it
        # within a set that may already be missing the best candidates.
        search_k = max(top_k, settings.rerank_candidate_k) if self.reranker else top_k
        contexts = self.retriever.search(question, top_k=search_k)

        if self.reranker:
            contexts = self.reranker.rerank(question, contexts, top_k=top_k)

        generation_result = self.generator.generate(question, contexts)
        answer = self.response_builder.build(generation_result)
        return answer, generation_result