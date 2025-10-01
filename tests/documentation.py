"""Shared documentation helpers for architectural tests."""

from __future__ import annotations

from typing import Callable, TypeVar

F = TypeVar("F", bound=Callable[..., object])


def documents(note: str) -> Callable[[F], F]:
    """Annotate a test with the documentation note it enforces."""

    def decorator(func: F) -> F:
        func.__doc__ = note if func.__doc__ is None else f"{note}\n{func.__doc__}"
        return func

    return decorator


__all__ = ["documents"]

