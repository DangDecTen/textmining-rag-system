"""
Registry for Reranker implementations. Mirrors src/retrieval/registry.py --
see that module's docstring for the rationale.

Usage:
    from src.reranking.registry import register_reranker
    from src.reranking.base import Reranker

    @register_reranker("my_new_reranker")
    class MyNewReranker(Reranker):
        ...

    from src.reranking.registry import build_reranker
    reranker = build_reranker("my_new_reranker", **kwargs)
"""
from __future__ import annotations

from src.reranking.base import Reranker

_REGISTRY: dict[str, type[Reranker]] = {}


def register_reranker(name: str):
    """Class decorator: registers a Reranker subclass under `name`."""

    def _decorator(cls: type[Reranker]) -> type[Reranker]:
        key = name.lower()
        if key in _REGISTRY and _REGISTRY[key] is not cls:
            raise ValueError(f"Reranker name '{key}' is already registered to {_REGISTRY[key]!r}")
        _REGISTRY[key] = cls
        return cls

    return _decorator


def build_reranker(name: str, **kwargs) -> Reranker:
    """Instantiate a registered Reranker by name, e.g. 'cross_encoder'."""
    key = name.lower()
    if key not in _REGISTRY:
        raise ValueError(f"Unknown reranker '{name}'. Available: {sorted(_REGISTRY)}")
    return _REGISTRY[key](**kwargs)


def available_rerankers() -> list[str]:
    return sorted(_REGISTRY)
