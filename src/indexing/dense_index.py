"""
Module Overview:
- Embed documents (e.g. "BAAI/bge-small-en-v1.5")
- Index with FAISS Exact search (IndexFlatIP, inner product on
  L2-normalized vectors = cosine similarity)
- Store model config and index in local

Components:
- A class Embedder for query and document embeddings
- A class DenseIndex for indexing
"""


from __future__ import annotations

import json
import numpy as np
from pathlib import Path
import faiss
from sentence_transformers import SentenceTransformer
from src.indexing.base import Index
import os
from dotenv import load_dotenv
load_dotenv()


if hf_token := os.getenv("HF_TOKEN"):
    os.environ["HF_TOKEN"] = hf_token

QUERY_INSTRUCTION_PREFIX = "Represent this sentence for searching relevant passages: "


class Embedder:
    """
    Query instruction prefix: applied EXPLICITLY here rather than relying on
    sentence-transformers' automatic prompt_name="query" mechanism, because that
    mechanism only fires if the model's bundled config_sentence_transformers.json
    defines a "query" prompt -- not guaranteed for every checkpoint/version. BAAI's
    own official usage example manually prepends the instruction to queries only
    (never to documents/passages), so we do the same explicitly to remove any
    ambiguity about whether it's actually being applied.

    Note: BAAI's bge-v1.5 model card states the instruction only gives a "slight"
    improvement for v1.5 models and is fine to skip "for convenience" -- so this
    is a minor lever, not a correctness-critical one, but it's free to get right.
    """

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", device: str | None = None):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device)

    def embed_documents(self, texts: list[str], batch_size: int = 64, show_progress_bar: bool = False) -> np.ndarray:
        """No instruction prefix -- documents/passages never get one (BGE convention)."""
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=show_progress_bar,
            convert_to_numpy=True,
        )
        return embeddings.astype("float32")

    def embed_query(self, text: str) -> np.ndarray:
        embedding = self.model.encode(
            [QUERY_INSTRUCTION_PREFIX + text],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return embedding.astype("float32")


class DenseIndex(Index):
    """
    Dense retrieval index: FAISS IndexFlatIP over bge-small-en-v1.5 embeddings.

    Exact search (IndexFlatIP, inner product on L2-normalized vectors = cosine
    similarity) -- appropriate at ~17.7k documents; no reason to reach for an
    approximate index (IVF/HNSW) at this scale (see project discussion).
    """

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", batch_size: int = 64):
        self.model_name = model_name
        self.batch_size = batch_size
        self.embedder = Embedder(model_name=model_name)
        self.index: faiss.IndexFlatIP | None = None
        self.doc_ids: list[str] = []

    def build(self, chunks: list) -> None:
        """chunks: list[Document]."""
        self.doc_ids = [c.doc_id for c in chunks]
        texts = [c.text for c in chunks]

        embeddings = self.embedder.embed_documents(texts, batch_size=self.batch_size, show_progress_bar=True)
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)

    def save(self, path: str) -> None:
        if self.index is None:
            raise RuntimeError("Cannot save an index that hasn't been built yet.")
        path_p = Path(path)
        path_p.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path_p / "index.faiss"))
        with open(path_p / "doc_ids.json", "w") as f:
            json.dump(self.doc_ids, f)
        with open(path_p / "config.json", "w") as f:
            json.dump({"model_name": self.model_name, "batch_size": self.batch_size}, f)

    @classmethod
    def load(cls, path: str) -> "DenseIndex":
        path_p = Path(path)
        with open(path_p / "config.json") as f:
            config = json.load(f)

        instance = cls(model_name=config["model_name"], batch_size=config.get("batch_size", 64))
        instance.index = faiss.read_index(str(path_p / "index.faiss"))
        with open(path_p / "doc_ids.json") as f:
            instance.doc_ids = json.load(f)
        return instance
