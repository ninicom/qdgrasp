"""Name-to-builder allowlists for declarative configuration.

YAML may only name entries that were registered in Python.  There is no
``eval``, no ``globals()`` lookup and no dynamic import driven by document
content, so a configuration file can never widen the executable surface.
"""

from __future__ import annotations

from typing import Callable, TypeVar


T = TypeVar("T")

_MODEL_BUILDERS: dict[str, Callable[..., object]] = {}
_DATASET_BUILDERS: dict[str, Callable[..., object]] = {}


class RegistryError(KeyError):
    """Raised when a configuration names a builder that is not registered."""


def _register(table: dict[str, Callable[..., object]], name: str, builder: Callable[..., T]) -> Callable[..., T]:
    if name in table and table[name] is not builder:
        raise RegistryError(f"builder '{name}' is already registered")
    table[name] = builder
    return builder


def register_model(name: str) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Register a model builder under an allowlisted ``name``."""

    def decorator(builder: Callable[..., T]) -> Callable[..., T]:
        return _register(_MODEL_BUILDERS, name, builder)

    return decorator


def register_dataset(name: str) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Register a dataset builder under an allowlisted ``name``."""

    def decorator(builder: Callable[..., T]) -> Callable[..., T]:
        return _register(_DATASET_BUILDERS, name, builder)

    return decorator


def _lookup(table: dict[str, Callable[..., object]], kind: str, name: str) -> Callable[..., object]:
    try:
        return table[name]
    except KeyError:
        known = ", ".join(sorted(table)) or "<none>"
        raise RegistryError(f"unknown {kind} type '{name}'; registered types: {known}") from None


def get_model_builder(name: str) -> Callable[..., object]:
    """Return the registered model builder or raise :class:`RegistryError`."""

    return _lookup(_MODEL_BUILDERS, "model", name)


def get_dataset_builder(name: str) -> Callable[..., object]:
    """Return the registered dataset builder or raise :class:`RegistryError`."""

    return _lookup(_DATASET_BUILDERS, "dataset", name)


def registered_models() -> tuple[str, ...]:
    """Sorted names of every registered model builder."""

    return tuple(sorted(_MODEL_BUILDERS))


def registered_datasets() -> tuple[str, ...]:
    """Sorted names of every registered dataset builder."""

    return tuple(sorted(_DATASET_BUILDERS))
