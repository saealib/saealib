"""Event classes for the optimization lifecycle."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

if TYPE_CHECKING:
    from saealib.acquisition.base import AcquisitionResult
    from saealib.execution.evaluator import EvaluationStatus
    from saealib.population import Archive, Population
    from saealib.problem import Problem
    from saealib.surrogate.base import Surrogate


class _EventContext(Protocol):
    """Read-only context shared by lifecycle state and graph runtime events."""

    @property
    def archive(self) -> Any: ...

    @property
    def comparator(self) -> Any: ...

    @property
    def n_obj(self) -> int: ...

    @property
    def gen(self) -> int: ...

    @property
    def fe(self) -> int: ...

    @property
    def problem(self) -> Problem: ...

    @property
    def population(self) -> Population: ...


@dataclass
class Event:
    """
    Base class for all callback events.

    Attributes
    ----------
    ctx : _EventContext
        Read-only context for the lifecycle boundary that emitted the event.
    """

    ctx: _EventContext


# --- Optimizer.run events ---


@dataclass
class RunStartEvent(Event):
    """Fired once when the optimization run starts."""


@dataclass
class RunEndEvent(Event):
    """Fired once when the optimization run ends."""


@dataclass
class GenerationStartEvent(Event):
    """Fired at the beginning of each generation."""


@dataclass
class GenerationEndEvent(Event):
    """Fired at the end of each generation, before yielding the context."""


# --- Surrogate events ---


@dataclass
class SurrogateStartEvent(Event):
    """Fired before surrogate-based candidate scoring."""

    offspring: Population | None = None


@dataclass
class SurrogateEndEvent(Event):
    """Fired after surrogate-based candidate scoring."""

    offspring: Population | None = None


# --- Acquisition events ---


@dataclass
class AcquisitionStartEvent(Event):
    """Fired before acquisition scoring, inside ``AcquisitionStage``.

    Attributes
    ----------
    offspring : Population or None
        Candidates about to be scored. Mirrors ``SurrogateStartEvent``'s
        shape for consistency; carries candidate ids indirectly via
        ``offspring.get_array("id")`` when the offspring schema has one.
    """

    offspring: Population | None = None


@dataclass
class AcquisitionEndEvent(Event):
    """Fired after acquisition scoring succeeds, inside ``AcquisitionStage``.

    Not dispatched if ``AcquisitionFunction.evaluate()`` raises.

    Attributes
    ----------
    offspring : Population or None
        Candidates that were scored.
    result : AcquisitionResult or None
        The validated acquisition result.
    """

    offspring: Population | None = None
    result: AcquisitionResult | None = None


# --- Algorithm.ask events ---


@dataclass
class PostCrossoverEvent(Event):
    """
    Fired after crossover and repair.

    ``candidates`` is exposed for observation only; reassigning it does not
    change the pipeline's output (algorithms keep using their own local
    array reference after dispatch). To actually alter crossover output, use
    ``with_post(fn)`` on the ``Crossover`` instance instead.
    """

    candidates: np.ndarray | None = None


@dataclass
class PostMutationEvent(Event):
    """
    Fired after mutation and repair.

    ``candidates`` is exposed for observation only; reassigning it does not
    change the pipeline's output (algorithms keep using their own local
    array reference after dispatch). To actually alter mutation output, use
    ``with_post(fn)`` on the ``Mutation`` instance instead.
    """

    candidates: np.ndarray | None = None


@dataclass
class PostAskEvent(Event):
    """
    Fired after the full ask step (post crossover and mutation).

    ``candidates`` is exposed for observation only; reassigning it does not
    change the pipeline's output. Whether an in-place edit is picked up
    depends on the algorithm: some (e.g. ``PSO``) dispatch this event after
    the candidates have already been copied into the offspring population,
    so even in-place changes have no effect. Do not rely on either form of
    mutation here; to actually alter ask output, use ``with_post(fn)`` on
    the relevant ``Crossover``/``Mutation`` instance instead.
    """

    candidates: np.ndarray | None = None


# --- Model events ---


@dataclass
class PostSurrogateFitEvent(Event):
    """Fired after the surrogate model is fitted.

    Attributes
    ----------
    surrogate : Surrogate or None
        The surrogate model that was just fitted.
    train_x : np.ndarray or None
        Design variable matrix used for fitting, shape (n_train, dim).
    train_f : np.ndarray or None
        Objective value matrix used for fitting, shape (n_train, n_obj).
    """

    surrogate: Surrogate | None = None
    train_x: np.ndarray | None = None
    train_f: np.ndarray | None = None


@dataclass(frozen=True, kw_only=True)
class ConfigurationChange:
    """One binding slot a published configuration replaced.

    Attributes
    ----------
    slot : str
        Binding slot whose component changed.
    previous : object or None
        Component the slot held in the previously published version.
    current : object or None
        Component the slot holds in the published version.
    source : str or None
        Where the write came from, using the optimizer's origin vocabulary
        (for example ``"setter"`` or ``"direct assignment"``), or ``None``
        when no origin was recorded.
    """

    slot: str
    previous: object | None
    current: object | None
    source: str | None = None


@dataclass
class ConfigurationCommittedEvent(Event):
    """Fired once per published binding version, after publication.

    A commit point that leaves the configuration unchanged publishes nothing
    and fires nothing, so an ordinary run produces no event however many
    times it enters a commit point.

    The event is dispatched to the callback manager of the published
    configuration, which is the one in effect from this point: a handler that
    the replacement registered on its own manager observes the configuration
    that now runs, and a manager the replacement replaced does not receive the
    event.

    The event is an observer notification sent after the publication completed.
    A handler that raises propagates to the caller, but it cannot veto the
    publication: the new binding version, the published snapshot, the bound
    slots, and the run state's binding version all keep describing the new
    configuration.

    Supported: every publication, on synchronous and asynchronous structured
    flows and on sequential plans, with one entry per replaced binding slot.
    Extensible toward: parameter changes, which the event does not carry.
    Out of scope: a History channel for configuration changes.

    Attributes
    ----------
    version : int
        Binding version the publication created; equal to the run state's
        ``_binding_version`` after the publication.
    changes : tuple of ConfigurationChange
        Replaced binding slots in ``BINDING_SLOTS`` order.  Empty when the
        publication replaced no slot, for example when a part inside a
        published component changed in place.
    """

    version: int
    changes: tuple[ConfigurationChange, ...]


@dataclass
class PostEvaluationEvent(Event):
    """Fired after true evaluation of selected candidates.

    Attributes
    ----------
    offspring : Population or None
        The candidates that were evaluated with the true objective function.
        All individuals in this population have true objective values assigned.
    """

    offspring: Population | None = None
    request_id: np.int64 | None = None
    candidate_ids: np.ndarray | None = None
    status: EvaluationStatus | None = None


# --- Initializer events ---


@dataclass
class InitialEvaluationStartEvent(Event):
    """Fired after initial sampling, before initial archive evaluation.

    Attributes
    ----------
    candidates_x : np.ndarray or None
        Design vectors to be evaluated, shape (n_init_archive, dim).
    """

    candidates_x: np.ndarray | None = None


@dataclass
class InitialEvaluationEndEvent(Event):
    """Fired after initial archive evaluation, before archive sorting.

    Attributes
    ----------
    archive : Archive or None
        The archive populated with initial evaluation results.
    """

    archive: Archive | None = None
