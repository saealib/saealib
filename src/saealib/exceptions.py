"""Public exception hierarchy for saealib."""

from dataclasses import dataclass
from typing import Any

__all__ = [
    "CheckpointError",
    "ConfigurationError",
    "EvaluationFatalError",
    "EvaluationProtocolError",
    "EvaluationSubmissionError",
    "RejectedChange",
    "ReplacementRejectedError",
    "SaealibError",
    "ValidationError",
]


class SaealibError(Exception):
    """Base class for all saealib exceptions."""


class ValidationError(SaealibError, ValueError):
    """Raised when user-supplied arguments fail validation at a public boundary."""


class StalePlanError(ValidationError):
    """Raised when a compiled plan no longer matches its node contracts."""

    code = "stale_plan"


class ConfigurationError(SaealibError, ValueError):
    """Raised when an :class:`~saealib.Optimizer` is misconfigured at run time."""


@dataclass(frozen=True, kw_only=True)
class RejectedChange:
    """One slot a commit point put back to its published component.

    Attributes
    ----------
    slot : str
        Binding slot that held the rejected component.
    previous : object or None
        Component the slot held in the last published configuration.
    candidate : object or None
        Component the slot held when the commit point rejected the change.
    source : str or None
        Source of the rejected write, when the optimizer recorded one.
    """

    slot: str
    previous: object | None
    candidate: object | None
    source: str | None = None


class ReplacementRejectedError(ConfigurationError):
    """A commit point refused a replacement and restored the published one.

    The optimizer holds the configuration it last published, and the binding
    version and configuration counter keep the values they had when that
    configuration was published, so a resumed run publishes nothing for the
    rejected change.

    Attributes
    ----------
    changes : tuple[RejectedChange, ...]
        Every slot whose component differed from the published one when the
        rejection happened.
    diagnostics : tuple[Diagnostic, ...]
        Structured findings that caused the rejection, for example a missing
        population column or a component that cannot be checkpointed.
    version : int
        Binding version of the configuration the optimizer kept.
    source : str or None
        The controller that produced the rejected command, as its registration
        index and class name, or ``None`` when the change came from a setter or
        a direct assignment.
    """

    def __init__(
        self,
        changes: tuple[RejectedChange, ...],
        diagnostics: Any,
        *,
        version: int,
        source: str | None = None,
    ) -> None:
        self.changes = tuple(changes)
        self.diagnostics = tuple(diagnostics)
        self.version = version
        self.source = source
        message = "Component replacement rejected at the commit point"
        if source is not None:
            message += f" (requested by {source})"
        if self.diagnostics:
            message += ":\n" + "\n".join(
                f"  - {diagnostic}" for diagnostic in self.diagnostics
            )
        super().__init__(message)


class CheckpointError(ValidationError):
    """Raised when a checkpoint is invalid or cannot be migrated."""


class EvaluationProtocolError(SaealibError, ValueError):
    """Raised when an evaluation lifecycle message violates its contract."""


class EvaluationFatalError(EvaluationProtocolError, RuntimeError):
    """Raised when a post-effect lifecycle failure cannot be retried."""

    def __init__(self, message: str, state: Any) -> None:
        super().__init__(message)
        self.state = state


class EvaluationSubmissionError(EvaluationProtocolError):
    """Raised when submission leaves externally running work."""

    def __init__(self, message: str, state: Any) -> None:
        super().__init__(message)
        self.state = state
