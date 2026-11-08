"""Private helpers for consistent API deprecation warnings."""

from __future__ import annotations

import functools
import warnings
from collections.abc import Callable
from typing import ParamSpec, TypeVar, cast

P = ParamSpec("P")
R = TypeVar("R")
T = TypeVar("T")


def warn_deprecated(
    name: str,
    *,
    since: str,
    replacement: str,
    stacklevel: int = 2,
) -> None:
    """Emit a consistently worded deprecation warning.

    ``stacklevel`` counts frames from the public call site. The helper adds its
    own frame so warnings identify the caller rather than this module.
    """
    warnings.warn(
        f"{name} is deprecated since {since}; use {replacement} instead.",
        DeprecationWarning,
        stacklevel=stacklevel + 1,
    )


def deprecated_argument_alias(
    old_value: T | None,
    new_value: T | None,
    *,
    owner: str,
    old_name: str,
    new_name: str,
    since: str,
    stacklevel: int = 2,
) -> T | None:
    """Resolve a deprecated keyword alias while preserving its old value."""
    if old_value is not None:
        warn_deprecated(
            f"{owner}.{old_name}",
            since=since,
            replacement=f"{owner}.{new_name}",
            stacklevel=stacklevel + 1,
        )
    return new_value if new_value is not None else old_value


def deprecated_function(
    *,
    since: str,
    replacement: str,
    name: str | None = None,
    condition: Callable[..., bool] | None = None,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Mark function calls as deprecated, optionally under a call condition."""

    def decorate(function: Callable[P, R]) -> Callable[P, R]:
        warning_name = (
            name
            or f"{getattr(function, '__qualname__', type(function).__qualname__)}()"
        )

        @functools.wraps(function)
        def wrapped(*args: P.args, **kwargs: P.kwargs) -> R:
            if condition is None or condition(*args, **kwargs):
                warn_deprecated(
                    warning_name,
                    since=since,
                    replacement=replacement,
                    stacklevel=2,
                )
            return function(*args, **kwargs)

        return cast(Callable[P, R], wrapped)

    return decorate
