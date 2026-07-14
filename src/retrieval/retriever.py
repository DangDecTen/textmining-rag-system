"""Basic dense retriever over the FAISS ATT&CK index."""

from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np

from src.indexing.embeddings import EmbeddingBackend
from src.indexing.build_index import EMBED_MODEL_NAME, INDEX_FILE_NAME, METADATA_FILE_NAME


def load_metadata(metadata_path: str | Path) -> list[dict]:
    with open(metadata_path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


class DenseRetriever:
    def __init__(self, index_dir: str | Path, model_name: str = EMBED_MODEL_NAME):
        index_path = Path(index_dir) / INDEX_FILE_NAME
        metadata_path = Path(index_dir) / METADATA_FILE_NAME

        if not index_path.exists():
            raise FileNotFoundError(f"Missing FAISS index: {index_path}")
        if not metadata_path.exists():
            raise FileNotFoundError(f"Missing chunk metadata: {metadata_path}")

        self.index = faiss.read_index(str(index_path))
        self.metadata = load_metadata(metadata_path)
        self.model = EmbeddingBackend(model_name)

        if self.index.ntotal != len(self.metadata):
            raise ValueError(
                f"Index size ({self.index.ntotal}) does not match metadata rows ({len(self.metadata)})"
            )

    def search(self, query: str, k: int = 5) -> list[dict]:
        query_embedding = self.model.encode([query]).astype(np.float32)

        scores, indices = self.index.search(query_embedding, k)
        results = []
        for score, index in zip(scores[0], indices[0]):
            if index < 0:
                continue
            record = dict(self.metadata[index])
            record["score"] = float(score)
            results.append(record)
        return results


if __name__ == "__main__":
    retriever = DenseRetriever("data/index/faiss_index")
    for hit in retriever.search("mitigations for command and control", k=3):
        print(hit["score"], hit["chunk_id"], hit["name"])
