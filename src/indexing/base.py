from abc import ABC
from abc import abstractmethod

class Index(ABC):

    @abstractmethod
    def build(self, chunks: list) -> None:
        """chunks: list[Document]"""
        pass
 
    @abstractmethod
    def save(self, path: str) -> None:
        pass
 
    @classmethod
    @abstractmethod
    def load(cls, path: str) -> "Index":
        pass
 