"""Resolved semantic values used by execution preparation."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

from saealib.defaults.keys import DefaultKey
from saealib.exceptions import ValidationError


class DefaultStrength(IntEnum):
    """Stored strength value retained by optimizer pickle records."""

    FALLBACK = 1
    RECOMMENDED = 2
    REQUIRED = 3


@dataclass(frozen=True)
class DefaultHint:
    """Stored semantic hint retained by optimizer pickle records."""

    key: DefaultKey
    value: Any
    strength: DefaultStrength
    source: str
    reason: str = ""

    def __post_init__(self) -> None:
        """Validate values when the record is constructed directly."""
        if not isinstance(self.key, DefaultKey):
            raise ValidationError("DefaultHint.key must be a DefaultKey")
        if not isinstance(self.strength, DefaultStrength):
            raise ValidationError("DefaultHint.strength must be a DefaultStrength")
        value_type = self.key.value_type
        if not isinstance(value_type, type):
            raise ValidationError(
                f"DefaultKey {self.key.name!r} must declare a type as value_type"
            )
        if isinstance(self.value, bool) and value_type is int:
            raise ValidationError(
                f"DefaultHint value for {self.key.name!r} must be an int, not bool"
            )
        if not isinstance(self.value, value_type):
            article = "an" if value_type.__name__[0] in "aeiou" else "a"
            raise ValidationError(
                f"DefaultHint value for {self.key.name!r} must be "
                f"{article} {value_type.__name__}, got {type(self.value).__name__}"
            )

    def __repr__(self) -> str:
        return (
            f"DefaultHint(key={self.key.name!r}, value={self.value!r}, "
            f"strength={self.strength.name}, source={self.source!r})"
        )


@dataclass(frozen=True)
class ResolvedDefault:
    """A semantic value selected for the current composition."""

    key: DefaultKey
    value: Any
    source: str
    reason: str


@dataclass(frozen=True)
class DefaultResolution:
    """Resolved semantic values and their selection reasons."""

    values: dict[DefaultKey, Any] = field(default_factory=dict)
    resolved: dict[DefaultKey, ResolvedDefault] = field(default_factory=dict)

    def get(self, key: DefaultKey, default: Any = None) -> Any:
        """Return the selected value for ``key`` or ``default``."""
        return self.values.get(key, default)
