from dataclasses import dataclass

from .chunk import Chunk

@dataclass(slots=True)
class RetrievedChunk:

    chunk: Chunk
    score: float
    rank: int
    retriever: str