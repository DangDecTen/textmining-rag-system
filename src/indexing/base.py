from abc import ABC
from abc import abstractmethod

class Index(ABC):

    @abstractmethod
    def build(self, chunks):
        pass

    @abstractmethod
    def save(self, path):
        pass

    @classmethod
    @abstractmethod
    def load(cls, path):
        pass