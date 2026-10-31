from __future__ import annotations

from dataclasses import dataclass

from saealib.core.state.keys import StateKey
from saealib.exceptions import ValidationError

__all__ = ["StateContract"]


@dataclass(frozen=True, kw_only=True)
class StateContract:
    """Declare state keys read, written, or exported by a component."""

    reads: tuple[StateKey[object], ...] = ()
    writes: tuple[StateKey[object], ...] = ()
    exports: tuple[StateKey[object], ...] = ()
    reads_enumerable: bool = True
    writes_enumerable: bool = True

    def __post_init__(self) -> None:
        for field_name in ("reads", "writes", "exports"):
            keys = tuple(getattr(self, field_name))
            if any(not isinstance(key, StateKey) for key in keys):
                raise ValidationError(
                    f"StateContract {field_name} must contain StateKey values"
                )
            object.__setattr__(self, field_name, keys)
        if not isinstance(self.reads_enumerable, bool):
            raise ValidationError("StateContract reads_enumerable must be a boolean")
        if not isinstance(self.writes_enumerable, bool):
            raise ValidationError("StateContract writes_enumerable must be a boolean")
