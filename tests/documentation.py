"""Shared documentation helpers for architectural tests."""

from __future__ import annotations

from datetime import date
from functools import wraps
from typing import Callable, TypeVar, cast

F = TypeVar("F", bound=Callable[..., object])


def documents(note: str) -> Callable[[F], F]:
    """Annotate a test with the documentation note it enforces."""

    def decorator(func: F) -> F:
        func.__doc__ = note if func.__doc__ is None else f"{note}\n{func.__doc__}"
        return func

    return decorator


def valid_until(iso_date: str, *, reason: str) -> Callable[[F], F]:
    """Fail the decorated test once the provided date has passed.

    This helper acts as a temporal contract around architectural decisions: once
    the stated expiry date is exceeded the test will raise an assertion failure,
    signalling that the guarded assumption needs to be reviewed.
    """

    deadline = date.fromisoformat(iso_date)

    def decorator(func: F) -> F:
        note = f"Valid until {deadline.isoformat()} – {reason}"

        @wraps(func)
        def wrapper(*args: object, **kwargs: object):
            today = date.today()
            if today > deadline:
                raise AssertionError(
                    "Temporal contract expired: "
                    f"{func.__name__!r} requires review after {deadline.isoformat()} "
                    f"because {reason}."
                )
            return func(*args, **kwargs)

        wrapper.__doc__ = (
            note if func.__doc__ is None else f"{note}\n{func.__doc__}"
        )
        return cast(F, wrapper)

    return decorator


__all__ = ["documents", "valid_until"]

