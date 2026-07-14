"""Embedding backends for dense retrieval.

The preferred backend is sentence-transformers when available. In lean test
environments where that dependency is missing, we fall back to a deterministic
hashed bag-of-words encoder so the FAISS pipeline still works end to end.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Iterable

import numpy as np

EMBED_DIMENSION = 384

try:
    from sentence_transformers import SentenceTransformer as _SentenceTransformer
except ImportError:  # pragma: no cover - exercised when optional dependency is absent
    _SentenceTransformer = None


_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_./:-]+")


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def _hash_token(token: str, dimension: int) -> int:
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % dimension


def _hash_encode(texts: Iterable[str], dimension: int = EMBED_DIMENSION) -> np.ndarray:
    matrix = np.zeros((len(texts), dimension), dtype=np.float32)
    for row, text in enumerate(texts):
        for token in _TOKEN_PATTERN.findall(text.lower()):
            matrix[row, _hash_token(token, dimension)] += 1.0
    return _normalize_rows(matrix)


@dataclass
class EmbeddingBackend:
    model_name: str
    dimension: int = EMBED_DIMENSION

    def __post_init__(self) -> None:
        self._model = _SentenceTransformer(self.model_name) if _SentenceTransformer is not None else None

    @property
    def using_sentence_transformers(self) -> bool:
        return self._model is not None

    def encode(self, texts: Iterable[str]) -> np.ndarray:
        texts = list(texts)
        if self._model is not None:
            embeddings = self._model.encode(
                texts,
                batch_size=32,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            return np.asarray(embeddings, dtype=np.float32)
        return _hash_encode(texts, self.dimension)
