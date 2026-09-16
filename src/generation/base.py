from abc import ABC, abstractmethod
from src.data_models.data_models import RetrievalResult, GenerationResult
from src.generation.prompt import PromptMode


class Generator(ABC):
    @abstractmethod
    def generate(self, question: str, contexts: list[RetrievalResult], prompt_mode: PromptMode = "baseline") -> GenerationResult:
        pass
