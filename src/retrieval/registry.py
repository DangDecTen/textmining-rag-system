"""
Registry for Retriever implementations.

Why this exists: previously, adding a new retriever meant hand-editing an
if/elif chain in a separate factory file, which drifted out of sync with the
actual classes (see git history / PR description). With a registry, a new
retriever is "self-installing": decorate the class, import the module once,
and it's available everywhere `build_retriever()` is used -- CLI scripts,
the API, tests -- with no other file to touch.

Usage:
    # in your retriever module
    from src.retrieval.registry import register_retriever
    from src.retrieval.base import Retriever

    @register_retriever("my_new_retriever")
    class MyNewRetriever(Retriever):
        ...

    # anywhere else
    from src.retrieval.registry import build_retriever
    retriever = build_retriever("my_new_retriever", **kwargs)
"""
from __future__ import annotations

from src.retrieval.base import Retriever

_REGISTRY: dict[str, type[Retriever]] = {}


def register_retriever(name: str):
    """Class decorator: registers a Retriever subclass under `name`."""

    def _decorator(cls: type[Retriever]) -> type[Retriever]:
        key = name.lower()
        if key in _REGISTRY and _REGISTRY[key] is not cls:
            raise ValueError(f"Retriever name '{key}' is already registered to {_REGISTRY[key]!r}")
        _REGISTRY[key] = cls
        return cls

    return _decorator


def build_retriever(name: str, **kwargs) -> Retriever:
    """Instantiate a registered Retriever by name, e.g. 'bm25' or 'dense'."""
    key = name.lower()
    if key not in _REGISTRY:
        raise ValueError(f"Unknown retriever '{name}'. Available: {sorted(_REGISTRY)}")
    return _REGISTRY[key](**kwargs)


def available_retrievers() -> list[str]:
    return sorted(_REGISTRY)
