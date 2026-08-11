"""
Registry for Generator implementations. Mirrors src/retrieval/registry.py --
see that module's docstring for the rationale.

Usage:
    from src.generation.registry import register_generator
    from src.generation.base import Generator

    @register_generator("my_new_generator")
    class MyNewGenerator(Generator):
        ...

    from src.generation.registry import build_generator
    generator = build_generator("my_new_generator", **kwargs)
"""
from __future__ import annotations

from src.generation.base import Generator

_REGISTRY: dict[str, type[Generator]] = {}


def register_generator(name: str):
    """Class decorator: registers a Generator subclass under `name`."""

    def _decorator(cls: type[Generator]) -> type[Generator]:
        key = name.lower()
        if key in _REGISTRY and _REGISTRY[key] is not cls:
            raise ValueError(f"Generator name '{key}' is already registered to {_REGISTRY[key]!r}")
        _REGISTRY[key] = cls
        return cls

    return _decorator


def build_generator(name: str, **kwargs) -> Generator:
    """Instantiate a registered Generator by name, e.g. 'llama' or 'qwen'."""
    key = name.lower()
    if key not in _REGISTRY:
        raise ValueError(f"Unknown generator '{name}'. Available: {sorted(_REGISTRY)}")
    return _REGISTRY[key](**kwargs)


def available_generators() -> list[str]:
    return sorted(_REGISTRY)
