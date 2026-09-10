"""
Application-specific construction logic.

Unlike src.factory, this factory uses Groq-backed generators.
Retrievers and rerankers continue to come from the shared src.factory.
"""

from __future__ import annotations

from functools import lru_cache
from src.data_models.io import load_corpus_lookup
from src.config import settings
from src.factory import (
    get_reranker,
    get_retriever,
    available_retrievers,
)

from src.pipeline import Pipeline

from app.backend.app_config import (
    APP_DEFAULT_GENERATOR,
    get_app_generator_model,
)
from app.backend.groq_generator import GroqGenerator


@lru_cache(maxsize=8)
def get_app_generator(name: str, prompt_mode: str = "baseline") -> GroqGenerator:
    name = (name or APP_DEFAULT_GENERATOR).lower()

    return GroqGenerator(
        model_name=get_app_generator_model(name),
        prompt_mode=prompt_mode,
        max_context_tokens=settings.max_context_tokens,
        max_new_tokens=settings.max_new_tokens,
    )


def get_app_pipeline(
    retriever_name: str | None = None,
    generator_name: str | None = None,
    prompt_mode: str = "baseline",
) -> Pipeline:
    return Pipeline(
        retriever=get_retriever(retriever_name),
        generator=get_app_generator(generator_name, prompt_mode),
        reranker=get_reranker() if settings.rerank_enabled else None,
    )
