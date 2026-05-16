from __future__ import annotations

from collections.abc import Callable
from typing import Any

from semantic_resume_matcher.models.base import ResumeRequirementModel


ModelFactory = Callable[..., ResumeRequirementModel]
_REGISTRY: dict[str, ModelFactory] = {}


def register_model(name: str) -> Callable[[ModelFactory], ModelFactory]:
    def decorator(model_cls: ModelFactory) -> ModelFactory:
        if name in _REGISTRY:
            raise ValueError(f"Model '{name}' is already registered.")
        _REGISTRY[name] = model_cls
        return model_cls

    return decorator


def get_model_class(name: str) -> ModelFactory:
    try:
        return _REGISTRY[name]
    except KeyError as error:
        available = ", ".join(sorted(_REGISTRY)) or "none"
        raise KeyError(f"Unknown model '{name}'. Available models: {available}") from error


def build_model(name: str, *, vocab_size: int, pad_id: int, params: dict[str, Any]) -> ResumeRequirementModel:
    model_cls = get_model_class(name)
    return model_cls(vocab_size=vocab_size, pad_id=pad_id, **params)


def list_models() -> list[str]:
    return sorted(_REGISTRY)

