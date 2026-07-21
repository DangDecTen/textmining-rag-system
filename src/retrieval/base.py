from abc import ABC
from abc import abstractmethod

class Retriever(ABC):

    @abstractmethod
    def search(self, query: str, top_k: int = 10) -> list:
        """Returns list[RetrievalResult], ranked best-first."""
        pass
    