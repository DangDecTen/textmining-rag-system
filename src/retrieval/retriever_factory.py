from pathlib import Path
from src.retrieval.retriever import DenseRetriever

DEFAULT_INDEX_DIR = Path("data/index/faiss_index")

class RetrieverFactory:
    @staticmethod
    def create(retriever_type: str):
        retriever_type = (retriever_type.lower())

        if retriever_type == "dense":
            return DenseRetriever(DEFAULT_INDEX_DIR)

        raise ValueError(
            f"Unknown retriever: "
            f"{retriever_type}"
        )