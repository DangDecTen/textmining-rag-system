"""
Shared construction logic for retrievers, generators, and the full pipeline.

This is the ONE place that knows how to turn a name ("bm25", "dense",
"llama", "qwen") into a ready-to-use object. `run_rag.py` and
`app/backend/api.py` both call into this module instead of each
re-implementing "load the index, load the corpus, build the retriever" --
that duplication is exactly how the old `retriever_factory.py` drifted out
of sync with the real classes.

Importing this module has the side effect of importing every built-in
retriever/generator module, which is what triggers their
`@register_retriever` / `@register_generator` decorators. If you add a new
implementation, add its import here too (one line) -- see
src/retrieval/README.md and src/generation/README.md for the full
"how to add a new X" walkthrough.
"""
from __future__ import annotations

from functools import lru_cache

from src.config import settings
from src.data_models.io import load_corpus_lookup
from src.pipeline import Pipeline

# Side-effect imports: populate the registries.
import src.retrieval.bm25_retriever  # noqa: F401
import src.retrieval.dense_retriever  # noqa: F401
import src.generation.llama_generator  # noqa: F401
import src.generation.qwen_generator  # noqa: F401

from src.retrieval.registry import build_retriever, available_retrievers
from src.generation.registry import build_generator, available_generators
from src.retrieval.base import Retriever
from src.generation.base import Generator

__all__ = [
    "get_retriever",
    "get_generator",
    "get_pipeline",
    "available_retrievers",
    "available_generators",
]


@lru_cache(maxsize=1)
def _corpus_lookup():
    return load_corpus_lookup(settings.corpus_path)


@lru_cache(maxsize=8)
def get_retriever(name: str | None = None) -> Retriever:
    """Load (and cache) a retriever by name, e.g. 'bm25' or 'dense'.

    Loads the corresponding pre-built index from disk. Run
    `python -m src.run_bm25 --rebuild` / `python -m src.run_dense --rebuild`
    first if the index doesn't exist yet.
    """
    name = (name or settings.default_retriever).lower()
    corpus_lookup = _corpus_lookup()

    if name == "bm25":
        from src.indexing.bm25_index import BM25Index

        index = BM25Index.load(settings.bm25_index_dir)
    elif name == "dense":
        from src.indexing.dense_index import DenseIndex

        index = DenseIndex.load(settings.dense_index_dir)
    else:
        raise ValueError(f"Unknown retriever '{name}'. Available: {available_retrievers()}")

    return build_retriever(name, index=index, corpus_lookup=corpus_lookup)


@lru_cache(maxsize=8)
def get_generator(name: str | None = None) -> Generator:
    """Load (and cache) a generator by name, e.g. 'llama' or 'qwen'."""
    name = (name or settings.default_generator).lower()

    if name == "llama":
        return build_generator(
            name,
            model_name=settings.llama_model_name,
            max_context_tokens=settings.max_context_tokens,
            max_new_tokens=settings.max_new_tokens,
        )
    if name == "qwen":
        return build_generator(
            name,
            model_name=settings.qwen_model_name,
            max_context_tokens=settings.max_context_tokens,
            max_new_tokens=settings.max_new_tokens,
        )
    raise ValueError(f"Unknown generator '{name}'. Available: {available_generators()}")


def get_pipeline(retriever_name: str | None = None, generator_name: str | None = None) -> Pipeline:
    """Build a full Pipeline. This is what the CLI and API both call."""
    return Pipeline(retriever=get_retriever(retriever_name), generator=get_generator(generator_name))
