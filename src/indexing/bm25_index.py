"""
BM25 index built on the `bm25s` library.
- Tokenize document
- Index scoring artifacts (sparse score matrix)
- Store the Index (with `doc_id`, not full `text`), and Tokenizer in local

Components:
- We use bm25s's `Tokenizer` class with OUR OWN splitter function
(`simple_tokenize`) rather than bm25s's built-in tokenizer.
- BM25Index with abstract class Index
"""

from __future__ import annotations
import re
from pathlib import Path
import bm25s
from bm25s.tokenization import Tokenizer
from src.indexing.base import Index


_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def simple_tokenize(text: str) -> list[str]:
    """
    Design: lowercase + regex word-splitting (`[a-z0-9]+`). No stopword removal,
    no stemming -- decided deliberately simple to start (see project discussion).
    Note that "T1055.001" splits into "t1055" and "001".
    """
    return _TOKEN_PATTERN.findall(text.lower())


class BM25Index(Index):
    def __init__(self, method: str = "lucene", k1: float = 1.0, b: float = 0.25):
        self.method = method
        self.k1 = k1
        self.b = b
        self.tokenizer = Tokenizer(stemmer=None, stopwords=[], splitter=simple_tokenize)
        self.model: bm25s.BM25 | None = None
        self.doc_ids: list[str] = []

    def build(self, chunks: list) -> None:
        """chunks: list[Document]. Document == chunk for stage01 (see ingestion notes)."""
        self.doc_ids = [c.doc_id for c in chunks]
        texts = [c.text for c in chunks]

        corpus_tokens = self.tokenizer.tokenize(texts)  # builds vocab on this first call
        self.model = bm25s.BM25(method=self.method, k1=self.k1, b=self.b)
        self.model.index(corpus_tokens)

    def save(self, path: str) -> None:
        if self.model is None:
            raise RuntimeError("Cannot save an index that hasn't been built yet.")
        path_p = Path(path)
        path_p.mkdir(parents=True, exist_ok=True)
        # corpus=self.doc_ids: bm25s returns these doc_ids directly on retrieve(),
        # not full Document objects -- that join happens in BM25Retriever.
        self.model.save(str(path_p), corpus=self.doc_ids)
        self.tokenizer.save_vocab(save_dir=str(path_p))
        self.tokenizer.save_stopwords(save_dir=str(path_p))

    @classmethod
    def load(cls, path: str) -> "BM25Index":
        path_p = Path(path)
        model = bm25s.BM25.load(str(path_p), load_corpus=True)

        instance = cls()
        instance.model = model
        instance.doc_ids = model.corpus  # the doc_id list we passed to save()

        instance.tokenizer = Tokenizer(stemmer=None, stopwords=[], splitter=simple_tokenize)
        instance.tokenizer.load_vocab(save_dir=str(path_p))
        instance.tokenizer.load_stopwords(save_dir=str(path_p))
        return instance
