"""Simple model registry so new architectures can be added without touching
training/eval scripts — register once, select by config `model.name`.
"""
from __future__ import annotations

from typing import Callable

_REGISTRY: dict[str, Callable] = {}


def register_model(name: str):
    def decorator(builder_fn: Callable):
        if name in _REGISTRY:
            raise ValueError(f"Model '{name}' already registered.")
        _REGISTRY[name] = builder_fn
        return builder_fn

    return decorator


def build_model(name: str, **kwargs):
    if name not in _REGISTRY:
        raise KeyError(f"Unknown model '{name}'. Registered models: {sorted(_REGISTRY.keys())}")
    return _REGISTRY[name](**kwargs)
