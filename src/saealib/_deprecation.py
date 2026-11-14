"""Private helpers for consistent API deprecation warnings."""

from __future__ import annotations

import functools
import os
import sys
import warnings
from collections.abc import Callable
from types import FrameType
from typing import ParamSpec, TypeVar, cast

P = ParamSpec("P")
R = TypeVar("R")
T = TypeVar("T")

_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__)) + os.sep


def _is_package_frame(frame: FrameType) -> bool:
    """Return whether a frame runs code that belongs to this package."""
    return frame.f_code.co_filename.startswith(_PACKAGE_DIR)


def _stacklevel_outside_package(default: int = 2) -> int:
    """Locate the first stack frame outside this package.

    Parameters
    ----------
    default
        Stack level returned when no outside frame is found or when frame
        introspection fails for any reason, including a refusal raised by an
        audit hook.

    Returns
    -------
    int
        ``stacklevel`` for :func:`warn_deprecated`, counted from that
        function's caller. Level 1 is the caller of :func:`warn_deprecated`,
        so a warning raised by a package helper is attributed to the code that
        invoked the helper.
    """
    level = 1
    try:
        frame: FrameType | None = sys._getframe(1)
        while frame is not None:
            if not _is_package_frame(frame):
                return level
            frame = frame.f_back
            level += 1
    except Exception:
        return default
    return default


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
