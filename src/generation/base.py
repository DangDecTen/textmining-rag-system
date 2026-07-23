from abc import ABC, abstractmethod
from src.data_models.data_models import RetrievalResult, GenerationResult


class Generator(ABC):
    @abstractmethod
    def generate(self, question: str, contexts: list[RetrievalResult]) -> GenerationResult:
        pass
