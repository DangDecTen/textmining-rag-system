"""
Combines the components built in earlier stages into a single call:
Retriever.search() -> Generator.generate()
"""
from __future__ import annotations

from src.data_models.data_models import Answer, GenerationResult
from src.generation.base import Generator
from src.generation.response_builder import ResponseBuilder
from src.retrieval.base import Retriever


class Pipeline:
    def __init__(self, retriever: Retriever, generator: Generator):
        self.retriever = retriever
        self.generator = generator
        self.response_builder = ResponseBuilder()

    def answer(self, question: str, top_k: int = 10) -> Answer:
        return self.answer_with_debug(question, top_k=top_k)[0]

    def answer_with_debug(self, question: str, top_k: int = 10) -> tuple[Answer, GenerationResult]:
        contexts = self.retriever.search(question, top_k=top_k)
        generation_result = self.generator.generate(question, contexts)
        answer = self.response_builder.build(generation_result)
        return answer, generation_result