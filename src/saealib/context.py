"""OptimizationState: immutable-style state passed through the optimization pipeline."""

from __future__ import annotations

import copy
import dataclasses
import functools
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast
from urllib.parse import quote

import numpy as np

from saealib.core.state import (
    EVALUATIONS_COUNT,
    EVALUATIONS_OWNERS,
    EVALUATIONS_PLAN,
    EVALUATIONS_PLAN_STATE,
    EVALUATIONS_PLAN_UPDATES,
    FEEDBACK_ACCUMULATOR,
    FEEDBACK_RESULT,
    PENDING_EVALUATIONS,
    PROPOSALS_ID_ALLOCATOR,
    PROPOSALS_OFFSPRING,
    RUNTIME_ASYNC_FATAL,
    RUNTIME_CANDIDATE_ID_ALLOCATOR,
    RUNTIME_DECISION_COUNT,
    RUNTIME_GENERATION,
    RUNTIME_REQUEST_ID_ALLOCATOR,
    RUNTIME_RNG,
    SURROGATES_PREDICTIONS,
    USER_DATA,
    StateKey,
    StatePatch,
    StateStore,
)
from saealib.exceptions import CheckpointError, ValidationError
from saealib.identity import IDAllocator
from saealib.space import BoundsService

if TYPE_CHECKING:
    from saealib.acquisition.base import AcquisitionResult
    from saealib.comparators import Comparator
    from saealib.execution.evaluator import (
        EvaluationHandle,
        EvaluationRequest,
        EvaluationUpdate,
        PendingEvaluation,
    )
    from saealib.execution.history import History
    from saealib.policies.evaluation import EvaluationPlan
    from saealib.policies.feedback import FeedbackResult
    from saealib.population import Archive, ParetoArchive, Population
    from saealib.problem import Problem
    from saealib.surrogate.prediction import SurrogatePrediction


CURRENT_CHECKPOINT_SCHEMA_VERSION = 2
SUPPORTED_CHECKPOINT_SCHEMA_VERSIONS = frozenset({1, 2})
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]*$")
_MISSING = object()

_STORE_FIELDS = {
    "rng": RUNTIME_RNG,
    "candidate_id_allocator": RUNTIME_CANDIDATE_ID_ALLOCATOR,
    "proposal_id_allocator": PROPOSALS_ID_ALLOCATOR,
    "request_id_allocator": RUNTIME_REQUEST_ID_ALLOCATOR,
    "fe": EVALUATIONS_COUNT,
    "gen": RUNTIME_GENERATION,
    "decision_count": RUNTIME_DECISION_COUNT,
    "evaluation_plan": EVALUATIONS_PLAN,
    "evaluation_plan_state": EVALUATIONS_PLAN_STATE,
    "evaluation_plan_updates": EVALUATIONS_PLAN_UPDATES,
    "evaluation_owners": EVALUATIONS_OWNERS,
    "pending_evaluations": PENDING_EVALUATIONS,
    "async_fatal": RUNTIME_ASYNC_FATAL,
    "data": USER_DATA,
    "predictions": SURROGATES_PREDICTIONS,
    "feedback_result": FEEDBACK_RESULT,
    "feedback_accumulator": FEEDBACK_ACCUMULATOR,
    "offspring": PROPOSALS_OFFSPRING,
}
_REPLACE_FAST_EXCLUDED = frozenset(
    {
        "problem",
        "populations",
        "archives",
        "evaluation_plan",
    }
)


@functools.lru_cache(maxsize=32)
def _dataclass_field_names(state_type: type[Any]) -> frozenset[str]:
    return frozenset(item.name for item in dataclasses.fields(state_type))


@functools.lru_cache(maxsize=1024)
def _collection_key(kind: str, name: str) -> StateKey[object]:
    return StateKey(namespace=kind, name=name, schema_version=1)


class _NamedCollection(dict[str, Any]):
    def __init__(self, kind: str, values: dict[str, Any]) -> None:
        self._kind = kind
        self._on_change: Any = None
        self._on_read: Any = None
        super().__init__()
        for name, value in values.items():
            self[name] = value

    def _bind(self, on_change: Any) -> None:
        self._on_change = on_change

    def _bind_read(self, on_read: Any) -> None:
        self._on_read = on_read

    def __getitem__(self, name: str) -> Any:
        value = dict.__getitem__(self, name)
        if self._on_read is not None:
            return self._on_read(name, value)
        return value

    def get(self, name: object, default: Any = None) -> Any:
        if not isinstance(name, str) or name not in self:
            return default
        return self[name]

    def _commit(self, values: dict[str, Any]) -> None:
        if self._on_change is not None:
            self._on_change(values)

    def _replace_local(self, values: dict[str, Any]) -> None:
        dict.clear(self)
        for name, value in values.items():
            dict.__setitem__(self, name, value)

    def __setitem__(self, name: str, value: Any) -> None:
        self._validate_entry(name, value)
        values = dict(self)
        values[name] = value
        self._commit(values)
        super().__setitem__(name, value)

    def _validate_entry(self, name: str, value: Any) -> None:
        _validate_collection_name(name)
        from saealib.population import Archive, ParetoArchive, Population

        if self._kind == "populations":
            valid = isinstance(value, Population)
        else:
            valid = (
                isinstance(value, (Archive, ParetoArchive))
                and (name != "main" or isinstance(value, Archive))
                and (name != "pareto" or isinstance(value, ParetoArchive))
            )
        if not valid:
            raise ValidationError(f"{self._kind}[{name!r}] has an invalid type")

    def __delitem__(self, name: str) -> None:
        _validate_collection_name(name)
        if (self._kind == "populations" and name == "main") or (
            self._kind == "archives" and name in {"main", "pareto"}
        ):
            raise ValidationError(f"cannot remove required {self._kind} entry {name!r}")
        values = dict(self)
        del values[name]
        self._commit(values)
        super().__delitem__(name)

    def clear(self) -> None:
        required = {"main"} if self._kind == "populations" else {"main", "pareto"}
        values = {name: value for name, value in self.items() if name in required}
        self._commit(values)
        self._replace_local(values)

    def pop(self, name: object, default: Any = _MISSING) -> Any:
        if not isinstance(name, str):
            raise ValidationError(f"invalid collection name: {name!r}")
        _validate_collection_name(name)
        required = {"main"} if self._kind == "populations" else {"main", "pareto"}
        if name in required:
            raise ValidationError(f"cannot remove required {self._kind} entry {name!r}")
        if name not in self:
            if default is _MISSING:
                raise KeyError(name)
            return default
        value = self[name]
        values = dict(self)
        del values[name]
        self._commit(values)
        super().__delitem__(name)
        return value

    def popitem(self) -> tuple[str, Any]:
        required = {"main"} if self._kind == "populations" else {"main", "pareto"}
        for name in reversed(list(self)):
            if name not in required:
                value = self[name]
                values = dict(self)
                del values[name]
                self._commit(values)
                super().__delitem__(name)
                return name, value
        raise ValidationError(f"cannot remove required {self._kind} entries")

    def setdefault(self, name: str, default: Any = None) -> Any:
        _validate_collection_name(name)
        if name in self:
            self._validate_entry(name, self[name])
            return self[name]
        self[name] = default
        return default

    def update(self, other: Any = (), /, **kwargs: Any) -> None:
        values = dict(other, **kwargs)
        for name, value in values.items():
            self._validate_entry(name, value)
        merged = dict(self)
        merged.update(values)
        self._commit(merged)
        for name, value in values.items():
            super().__setitem__(name, value)

    def __ior__(self, other: Any):
        self.update(other)
        return self

    def __reduce__(self):
        return (type(self), (self._kind, dict(self)))

    def __copy__(self):
        """Copy the mapping without invoking validation or callback setup."""
        result = dict.__new__(type(self))
        dict.update(result, self)
        result.__dict__ = self.__dict__.copy()
        return result


@functools.lru_cache(maxsize=1024)
def _validate_collection_name(name: str) -> None:
    if (
        not isinstance(name, str)
        or not name
        or name in {".", ".."}
        or name.startswith("_")
        or "//" in name
        or not _NAME_RE.fullmatch(name)
    ):
        raise ValidationError(f"invalid collection name: {name!r}")


def _encoded_name(name: str) -> str:
    return quote(name, safe="")


@dataclass(frozen=True)
class EvaluationPlanState:
    """Checkpointable lifecycle state for a multi-request evaluation plan."""

    submitted: tuple[int, ...] = ()
    completed: tuple[int, ...] = ()
    acknowledged: tuple[int, ...] = ()
    deferred: tuple[int, ...] = ()
    continuation: Any = None
    feedback: Any = None

    def __post_init__(self) -> None:
        """Validate request ID collections."""
        for name in ("submitted", "completed", "acknowledged", "deferred"):
            values = tuple(int(value) for value in getattr(self, name))
            if len(values) != len(set(values)):
                raise ValidationError(f"evaluation plan state {name} has duplicates")
            object.__setattr__(self, name, values)
        submitted = set(self.submitted)
        completed = set(self.completed)
        acknowledged = set(self.acknowledged)
        deferred = set(self.deferred)
        if not completed <= submitted:
            raise ValidationError("completed requests must be submitted")
        if not acknowledged <= completed:
            raise ValidationError("acknowledged requests must be completed")
        if deferred & (submitted | completed | acknowledged):
            raise ValidationError("deferred requests cannot be terminal or submitted")


def _validate_plan_state(
    plan: EvaluationPlan | None, plan_state: EvaluationPlanState | None
) -> None:
    if plan is None:
        return
    if plan_state is None:
        raise ValidationError(
            "evaluation plan state is required while a plan is active"
        )
    plan_ids = {int(request.request_id) for request in plan.requests}
    state_ids = (
        set(plan_state.submitted)
        | set(plan_state.completed)
        | set(plan_state.acknowledged)
        | set(plan_state.deferred)
    )
    if not state_ids <= plan_ids:
        raise ValidationError("evaluation plan state references an unknown request")


@dataclass(init=False)
class OptimizationState:
    """
    Optimization State class.

    Holds the state of the optimization process.  Passed through the pipeline
    as a value object; use :meth:`replace` to produce an updated copy rather
    than mutating fields directly.

    Controlled mutable exceptions (documented):

    - ``archive`` is append-only; copying on every evaluation would incur
      O(FE²) cost, so in-place appends are permitted.
    - ``rng`` advances its internal state as a controlled side effect.
    - ``candidate_id_allocator`` / ``proposal_id_allocator`` /
      ``request_id_allocator`` advance their
      internal counters as a controlled side effect, identically to ``rng``.

    Attributes
    ----------
    problem : Problem
        Problem instance.
    population : Population
        Population instance.  Its ``objective_source`` column (``uint8``,
        read with ``population.get_array("objective_source")``) records where
        each row's ``f`` came from: ``0`` true evaluation, ``1`` surrogate
        prediction, ``2`` imputed fallback value, ``255`` unknown.  Loading
        a checkpoint of schema version 1, which has no such column, marks a
        row true when its ``f`` equals the last true evaluation of its
        candidate ID in the History ``evaluation`` channel and unknown
        otherwise.
    archive : Archive
        Archive instance.  Append-only in-place mutation is permitted.
    pareto_archive : ParetoArchive
        Pareto archive instance.
    rng : np.random.Generator
        Random number generator.  Advances its state as a side effect.
    candidate_id_allocator : IDAllocator
        Allocates stable, unique int64 candidate IDs.  Advances its state as
        a side effect.
    request_id_allocator : IDAllocator
        Allocates stable, unique int64 evaluation request IDs.  Advances its
        state as a side effect.
    proposal_id_allocator : IDAllocator
        Allocates stable, unique int64 proposal IDs.  When omitted, it starts
        at the current request allocator value.
    history : History or None
        Optional execution history accumulator. Summary-channel rows are
        persisted by checkpoints.
    fe : int
        Number of function evaluations.
    gen : int
        Number of generations.
    decision_count : int
        Number of confirmed evaluation plans (one per infill decision,
        regardless of how many candidates that plan evaluates). Incremented
        once per genuinely new evaluation plan the runtime confirms --
        :class:`~saealib.stages.EvaluationPlanStage` for synchronous
        execution, :class:`~saealib.stages.AsyncEvaluationSubmitStage` for
        async/steady-state execution -- never on a plan continuation.
        Independent of ``gen``/``fe``.
    offspring : Population or None
        Candidate population produced by the current generation's ask step.
        Set by :class:`~saealib.stages.AskStage`; consumed and updated by
        downstream stages.
    evaluated_offspring : Population or None
        Sub-population that has received true objective values.
        Set by :class:`~saealib.stages.TrueEvaluationStage`; consumed by
        :class:`~saealib.stages.ArchiveUpdateStage`.
    scores : np.ndarray or None
        Acquisition scores for ``offspring``, shape ``(n_candidates,)``.
        Set by :class:`~saealib.stages.AcquisitionStage`.
    acquisition_result : AcquisitionResult or None
        Complete acquisition output, including artifacts needed by planners.
    predictions : SurrogatePrediction or None
        Batched surrogate prediction covering every row of ``offspring``.
        Set by :class:`~saealib.stages.SurrogatePredictStage`.
    pending_candidate_ids : np.ndarray
        Unique candidate IDs derived from ``pending_evaluations``.
    reserved_fe : int
        Number of candidates represented by ``pending_evaluations``.
    reserved_cost : float
        Total estimated cost represented by ``pending_evaluations``.
    async_fatal : dict[str, Any] or None
        Cross-node asynchronous evaluation failure signal.
    data : dict[str, Any]
        User-extensible key-value store.  Custom stages and callbacks may
        store arbitrary values here.  Use ``state.replace(data={**state.data,
        "key": value})`` — never mutate this dict in place.
    """

    problem: Problem

    populations: dict[str, Population]
    archives: dict[str, Archive | ParetoArchive]
    rng: np.random.Generator
    candidate_id_allocator: IDAllocator = field(default_factory=IDAllocator)
    proposal_id_allocator: IDAllocator = field(default_factory=IDAllocator)
    request_id_allocator: IDAllocator = field(default_factory=IDAllocator)
    history: History | None = None

    fe: int = 0
    gen: int = 0
    decision_count: int = 0

    # Pipeline stage data (typed)
    offspring: Population | None = None
    evaluated_offspring: Population | None = None
    scores: np.ndarray | None = None
    acquisition_result: AcquisitionResult | None = None
    predictions: SurrogatePrediction | None = None
    evaluation_request: EvaluationRequest | None = None
    evaluation_plan: EvaluationPlan | None = None
    evaluation_plan_state: EvaluationPlanState | None = None
    evaluation_updates: list[EvaluationUpdate] = field(default_factory=list)
    evaluation_plan_updates: dict[int, list[EvaluationUpdate]] = field(
        default_factory=dict
    )
    evaluation_update_new_ids: list[np.ndarray] = field(default_factory=list)
    evaluation_new_ids: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.int64)
    )
    evaluation_handles: dict[int, EvaluationHandle] = field(default_factory=dict)
    evaluation_owners: dict[int, Population] = field(default_factory=dict)
    pending_evaluations: dict[int, PendingEvaluation] = field(default_factory=dict)
    feedback_result: FeedbackResult | None = None
    feedback_accumulator: dict[str, Any] | None = None
    async_fatal: dict[str, Any] | None = None

    # User-extensible data
    data: dict[str, Any] = field(default_factory=dict)

    _custom_state: dict[StateKey, object] = field(
        default_factory=dict, repr=False, compare=False
    )

    def __getattr__(self, name: str) -> Any:
        """Read store-backed fields only when ordinary lookup misses."""
        if name == "populations":
            try:
                return object.__getattribute__(self, "_population_collection")
            except AttributeError as exc:
                raise AttributeError(name) from exc
        if name == "archives":
            try:
                return object.__getattribute__(self, "_archive_collection")
            except AttributeError as exc:
                raise AttributeError(name) from exc

        key = _STORE_FIELDS.get(name)
        if key is None:
            raise AttributeError(name)

        try:
            store = object.__getattribute__(self, "_store")
        except AttributeError:
            try:
                return object.__getattribute__(self, "_pending_" + name)
            except AttributeError as exc:
                raise AttributeError(name) from exc
        return store.get(key)

    def __setattr__(self, name: str, value: Any) -> None:
        """Patch store-backed fields instead of exposing store moves."""
        if name in {"populations", "archives"}:
            try:
                object.__getattribute__(self, "_store")
            except AttributeError:
                object.__setattr__(self, "_pending_" + name, value)
                return
            self._replace_collection(name, value)
            return
        key = _STORE_FIELDS.get(name)
        if key is not None:
            try:
                object.__getattribute__(self, "_store")
            except AttributeError:
                object.__setattr__(self, "_pending_" + name, value)
                return
            self._write_store(key, value)
            return
        object.__setattr__(self, name, value)

    def _write_store(self, key: StateKey, value: object) -> None:
        self._store = self._store.apply_patch(StatePatch(writes={key: value}))

    def _replace_collection(self, kind: str, values: dict[str, Any]) -> None:
        collection = _NamedCollection(kind, dict(values))
        required = {"main"} if kind == "populations" else {"main", "pareto"}
        if not required <= set(collection):
            raise ValidationError(f"{kind} must contain {sorted(required)!r}")
        writes = {
            _collection_key(kind, name): value for name, value in collection.items()
        }
        deletes = frozenset(
            key for key in self._store_keys(kind) if key.name not in collection
        )
        self._store = self._store.apply_patch(
            StatePatch(writes=writes, deletes=deletes)
        )
        collection._bind(lambda new: self._commit_collection(kind, new))
        collection._bind_read(
            lambda name, value: self._store.get(_collection_key(kind, name))
        )
        object.__setattr__(
            self,
            "_"
            + ("population" if kind == "populations" else "archive")
            + "_collection",
            collection,
        )

    def _store_keys(self, namespace: str) -> tuple[StateKey, ...]:
        return tuple(key for key in self._store._values if key.namespace == namespace)

    def _commit_collection(self, kind: str, values: dict[str, Any]) -> None:
        collection = _NamedCollection(kind, values)
        required = {"main"} if kind == "populations" else {"main", "pareto"}
        if not required <= set(collection):
            raise ValidationError(f"{kind} must contain {sorted(required)!r}")
        current = {key.name: key for key in self._store_keys(kind)}
        writes = {
            current.get(name, _collection_key(kind, name)): value
            for name, value in collection.items()
        }
        deletes = frozenset(
            key for name, key in current.items() if name not in collection
        )
        self._store = self._store.apply_patch(
            StatePatch(writes=writes, deletes=deletes)
        )

    def get_state(self, key: StateKey[Any]) -> Any:
        """Return a custom or built-in value held by this state's store.

        ``key`` is validated by :class:`StateStore`; user components may use
        ``namespace="user"`` keys without adding a core constant.
        """
        return self._store.get(key)

    def bind_compiled_services(self, services: dict[str, object]) -> None:
        """Bind compiler-resolved services for use outside the registry hot path.

        Bindings are deliberately transient runtime state: they are not
        dataclass fields and therefore are neither part of checkpoints nor the
        public state schema.  A service name may not be rebound to another
        object, since that would make a state use a stale compiled plan.
        """
        current = self.__dict__.setdefault("_compiled_services", {})
        for name, service in services.items():
            previous = current.get(name, _MISSING)
            if previous is not _MISSING and previous is not service:
                # ``replace(problem=...)`` may carry a binding from the old
                # problem.  A new problem starts a new binding generation.
                if self.__dict__.get("problem") is not self.__dict__.get(
                    "_compiled_problem"
                ):
                    current[name] = service
                    continue
                raise ValidationError(
                    f"compiled service {name!r} is bound to conflicting objects"
                )
        current.update(services)
        self.__dict__["_compiled_problem"] = self.__dict__.get("problem")

    def compiled_service(self, name: str) -> object:
        """Return a runtime-bound compiler service without registry fallback."""
        service = self.__dict__.get("_compiled_services", {}).get(name, _MISSING)
        if service is _MISSING:
            raise ValidationError(
                f"compiled service {name!r} is not bound to this OptimizationState"
            )
        return service

    def set_state(self, key: StateKey[Any], value: Any) -> None:
        """Replace one custom or built-in value through the state store."""
        if key.namespace in {"populations", "archives"}:
            expected_key = _collection_key(key.namespace, key.name)
            if key != expected_key:
                raise ValidationError(
                    f"collection key must be {expected_key!r}, got {key!r}"
                )
            collection = (
                self._population_collection
                if key.namespace == "populations"
                else self._archive_collection
            )
            if key.name not in collection:
                raise ValidationError(
                    f"cannot set unknown {key.namespace} entry {key.name!r}"
                )
            collection._validate_entry(key.name, value)
            self._write_store(key, value)
            dict.__setitem__(collection, key.name, value)
            return
        self._write_store(key, value)
        if key not in _STORE_FIELDS.values() and key.namespace not in {
            "populations",
            "archives",
        }:
            object.__setattr__(
                self, "_custom_state", {**self._custom_state, key: value}
            )

    def __init__(
        self,
        problem: Problem,
        population: Population | None = None,
        archive: Archive | None = None,
        pareto_archive: ParetoArchive | None = None,
        rng: np.random.Generator | None = None,
        *,
        populations: dict[str, Population] | None = None,
        archives: dict[str, Archive | ParetoArchive] | None = None,
        candidate_id_allocator: IDAllocator | None = None,
        proposal_id_allocator: IDAllocator | None = None,
        request_id_allocator: IDAllocator | None = None,
        history: History | None = None,
        fe: int = 0,
        gen: int = 0,
        decision_count: int = 0,
        offspring: Population | None = None,
        evaluated_offspring: Population | None = None,
        scores: np.ndarray | None = None,
        acquisition_result: AcquisitionResult | None = None,
        predictions: SurrogatePrediction | None = None,
        evaluation_request: EvaluationRequest | None = None,
        evaluation_plan: EvaluationPlan | None = None,
        evaluation_plan_state: EvaluationPlanState | None = None,
        evaluation_updates: list[EvaluationUpdate] | None = None,
        evaluation_plan_updates: dict[int, list[EvaluationUpdate]] | None = None,
        evaluation_update_new_ids: list[np.ndarray] | None = None,
        evaluation_new_ids: np.ndarray | None = None,
        evaluation_handles: dict[int, EvaluationHandle] | None = None,
        evaluation_owners: dict[int, Population] | None = None,
        pending_evaluations: dict[int, PendingEvaluation] | None = None,
        feedback_result: FeedbackResult | None = None,
        feedback_accumulator: dict[str, Any] | None = None,
        async_fatal: dict[str, Any] | None = None,
        data: dict[str, Any] | None = None,
        _custom_state: dict[StateKey, object] | None = None,
    ) -> None:
        if populations is None:
            if population is None:
                raise ValidationError("population or populations is required")
            populations = {"main": population}
        elif population is not None and populations.get("main") is not population:
            raise ValidationError("population must alias populations['main']")
        if archives is None:
            if archive is None or pareto_archive is None:
                raise ValidationError("archive and pareto_archive are required")
            archives = {"main": archive, "pareto": pareto_archive}
        else:
            if archive is not None and archives.get("main") is not archive:
                raise ValidationError("archive must alias archives['main']")
            if (
                pareto_archive is not None
                and archives.get("pareto") is not pareto_archive
            ):
                raise ValidationError("pareto_archive must alias archives['pareto']")
        object.__setattr__(
            self, "_population_collection", _NamedCollection("populations", populations)
        )
        object.__setattr__(
            self, "_archive_collection", _NamedCollection("archives", archives)
        )
        if "main" not in self.populations:
            raise ValidationError("populations must contain 'main'")
        if "main" not in self.archives or "pareto" not in self.archives:
            raise ValidationError("archives must contain 'main' and 'pareto'")
        self.problem = problem
        self.rng = rng if rng is not None else np.random.default_rng()
        self.candidate_id_allocator = (
            candidate_id_allocator
            if candidate_id_allocator is not None
            else IDAllocator()
        )
        self.request_id_allocator = (
            request_id_allocator if request_id_allocator is not None else IDAllocator()
        )
        self.proposal_id_allocator = (
            proposal_id_allocator
            if proposal_id_allocator is not None
            else IDAllocator(self.request_id_allocator.next_value)
        )
        self.history = history
        self.fe = fe
        self.gen = gen
        self.decision_count = decision_count
        self.offspring = offspring
        self.evaluated_offspring = evaluated_offspring
        self.scores = scores
        self.acquisition_result = acquisition_result
        self.predictions = predictions
        self.evaluation_request = evaluation_request
        self.evaluation_plan = evaluation_plan
        self.evaluation_plan_state = evaluation_plan_state
        _validate_plan_state(evaluation_plan, evaluation_plan_state)
        if evaluation_plan is not None and evaluation_plan_updates is not None:
            plan_ids = {int(request.request_id) for request in evaluation_plan.requests}
            if not set(map(int, evaluation_plan_updates)) <= plan_ids:
                raise ValidationError(
                    "evaluation plan updates reference an unknown request"
                )
        self.evaluation_updates = (
            [] if evaluation_updates is None else evaluation_updates
        )
        self.evaluation_plan_updates = (
            {} if evaluation_plan_updates is None else evaluation_plan_updates
        )
        self.evaluation_update_new_ids = (
            [] if evaluation_update_new_ids is None else evaluation_update_new_ids
        )
        self.evaluation_new_ids = (
            np.empty(0, dtype=np.int64)
            if evaluation_new_ids is None
            else evaluation_new_ids
        )
        self.evaluation_handles = (
            {} if evaluation_handles is None else evaluation_handles
        )
        self.evaluation_owners = {} if evaluation_owners is None else evaluation_owners
        self.pending_evaluations = (
            {} if pending_evaluations is None else pending_evaluations
        )
        self.feedback_result = feedback_result
        self.feedback_accumulator = feedback_accumulator
        self.async_fatal = async_fatal
        self.data = {} if data is None else data
        initial_store: dict[StateKey, object] = {
            _collection_key("populations", name): value
            for name, value in populations.items()
        }
        initial_store.update(
            {
                _collection_key("archives", name): value
                for name, value in archives.items()
            }
        )
        for name, key in _STORE_FIELDS.items():
            initial_store[key] = object.__getattribute__(self, "_pending_" + name)
        custom_state = {} if _custom_state is None else dict(_custom_state)
        initial_store.update(custom_state)
        object.__setattr__(self, "_store", StateStore(initial_store))
        object.__setattr__(self, "_custom_state", custom_state)
        # State construction is the formal setup seam used by direct
        # algorithm callers.  PipelineRuntime later verifies/rebinds this
        # object against the compiler's resolved reference.
        bounds_service = self.problem.space.services.get("BoundsService")
        object.__setattr__(
            self,
            "_compiled_services",
            {} if bounds_service is None else {"BoundsService": bounds_service},
        )
        object.__setattr__(self, "_compiled_problem", self.problem)
        self._population_collection._bind(
            lambda values: self._commit_collection("populations", values)
        )
        self._population_collection._bind_read(
            lambda name, value: self._store.get(_collection_key("populations", name))
        )
        self._archive_collection._bind(
            lambda values: self._commit_collection("archives", values)
        )
        self._archive_collection._bind_read(
            lambda name, value: self._store.get(_collection_key("archives", name))
        )
        for name in (*_STORE_FIELDS, "populations", "archives"):
            self.__dict__.pop("_pending_" + name, None)

    @property
    def population(self) -> Population:
        """Return the main population."""
        return self.get_population()

    @population.setter
    def population(self, value: Population) -> None:
        self.populations["main"] = value

    @property
    def archive(self) -> Archive:
        """Return the main archive."""
        return self.get_archive()

    @archive.setter
    def archive(self, value: Archive) -> None:
        self.archives["main"] = value

    @property
    def pareto_archive(self) -> ParetoArchive:
        """Return the Pareto archive."""
        return cast(Any, self.get_archive("pareto"))

    @pareto_archive.setter
    def pareto_archive(self, value: ParetoArchive) -> None:
        self.archives["pareto"] = value

    @property
    def pending_candidate_ids(self) -> np.ndarray:
        """Return unique candidate IDs reserved by pending evaluations."""
        values = [
            pending.request.candidate_ids
            for pending in self.pending_evaluations.values()
        ]
        return (
            np.unique(np.concatenate(values)).astype(np.int64, copy=False)
            if values
            else np.empty(0, dtype=np.int64)
        )

    @property
    def reserved_fe(self) -> int:
        """Return the number of candidates reserved by pending evaluations."""
        return sum(
            len(pending.request.candidate_ids)
            for pending in self.pending_evaluations.values()
        )

    @property
    def reserved_cost(self) -> float:
        """Return the total estimated cost reserved by pending evaluations."""
        from math import fsum

        return fsum(
            pending.reserved_cost for pending in self.pending_evaluations.values()
        )

    def __getstate__(self) -> dict[str, Any]:
        """Exclude runtime evaluation handles from serialized state."""
        state = {
            item.name: getattr(self, item.name)
            for item in dataclasses.fields(self)
            if item.init
        }
        state["evaluation_handles"] = {}
        state["evaluation_request"] = None
        state["evaluation_updates"] = []
        state["evaluation_update_new_ids"] = []
        state.pop("evaluation_new_ids", None)
        state.pop("evaluation_update_new_ids", None)
        return state

    def replace(self, **kwargs: Any) -> OptimizationState:
        """Return a new state with the given fields replaced.

        Parameters
        ----------
        **kwargs
            Field names and new values.

        Returns
        -------
        OptimizationState
        """
        legacy = {"population", "archive", "pareto_archive"}
        if any(key in kwargs for key in legacy):
            if "populations" in kwargs and "population" in kwargs:
                raise ValidationError(
                    "replace() cannot combine population and populations"
                )
            if "archives" in kwargs and legacy.intersection(kwargs) - {"population"}:
                raise ValidationError(
                    "replace() cannot combine archive aliases and archives"
                )
            populations = kwargs.pop("populations", self.populations)
            archives = kwargs.pop("archives", self.archives)
            if "population" in kwargs:
                populations = dict(self.populations)
                populations["main"] = kwargs.pop("population")
            if "archive" in kwargs or "pareto_archive" in kwargs:
                archives = dict(self.archives)
                if "archive" in kwargs:
                    archives["main"] = kwargs.pop("archive")
                if "pareto_archive" in kwargs:
                    archives["pareto"] = kwargs.pop("pareto_archive")
            kwargs["populations"] = populations
            kwargs["archives"] = archives

        field_names = _dataclass_field_names(type(self))
        unknown = set(kwargs) - field_names
        if unknown:
            name = sorted(unknown)[0]
            raise TypeError(
                "OptimizationState.__init__() got an unexpected keyword argument "
                f"{name!r}"
            )

        plan_fields = {"evaluation_plan_state", "evaluation_plan_updates"}
        # The common pipeline case replaces one store-backed value (usually
        # ``gen``) or an ordinary dataclass field.  The collection wrappers
        # still need to be distinct, but they do not need reconstruction when
        # no collection or plan definition changed.
        if all(name not in _REPLACE_FAST_EXCLUDED for name in kwargs):
            if plan_fields.intersection(kwargs):
                self._validate_plan_replacement(kwargs)
            return self._replace_lightweight(kwargs)

        populations = dict(kwargs.get("populations", self.populations))
        archives = dict(kwargs.get("archives", self.archives))
        population_collection = _NamedCollection("populations", populations)
        archive_collection = _NamedCollection("archives", archives)
        if "main" not in population_collection:
            raise ValidationError("populations must contain 'main'")
        if not {"main", "pareto"} <= set(archive_collection):
            raise ValidationError("archives must contain 'main' and 'pareto'")
        self._validate_plan_replacement(kwargs)

        writes: dict[StateKey, object] = {}
        deletes: set[StateKey] = set()
        current_values = self._store._values
        for kind, collection in (
            ("populations", population_collection),
            ("archives", archive_collection),
        ):
            current = {
                key.name: value
                for key, value in current_values.items()
                if key.namespace == kind
            }
            requested = kind in kwargs
            for name, value in collection.items():
                key = _collection_key(kind, name)
                if requested and current.get(name, _MISSING) is not value:
                    writes[key] = value
            if requested:
                deletes.update(
                    _collection_key(kind, name)
                    for name in current
                    if name not in collection
                )

        for name, value in kwargs.items():
            key = _STORE_FIELDS.get(name)
            if key is not None:
                writes[key] = value

        replacement_store = copy.copy(self._store)
        if writes or deletes:
            replacement_store = replacement_store.apply_patch(
                StatePatch(writes=writes, deletes=frozenset(deletes))
            )

        # Do not shallow-copy the OptimizationState itself: its reduction path
        # invokes __getstate__, which intentionally strips transient evaluation
        # fields for pickle.
        result = object.__new__(type(self))
        object.__setattr__(result, "__dict__", self.__dict__.copy())
        if "problem" in kwargs and kwargs["problem"] is not self.problem:
            # A state can be continued with a different problem (the dynamic
            # archive workflow does this).  The old plan bindings must not be
            # mistaken for bindings of the new problem's service objects.
            replacement_problem = kwargs["problem"]
            bounds_service = replacement_problem.space.services.get("BoundsService")
            object.__setattr__(
                result,
                "_compiled_services",
                {} if bounds_service is None else {"BoundsService": bounds_service},
            )
        for name, value in kwargs.items():
            if name not in _STORE_FIELDS and name not in {"populations", "archives"}:
                object.__setattr__(result, name, value)
        for name in _STORE_FIELDS:
            result.__dict__.pop(name, None)
        object.__setattr__(result, "_store", replacement_store)
        object.__setattr__(result, "_population_collection", population_collection)
        object.__setattr__(result, "_archive_collection", archive_collection)
        population_collection._bind(
            lambda values: result._commit_collection("populations", values)
        )
        population_collection._bind_read(
            lambda name, value: result._store.get(_collection_key("populations", name))
        )
        archive_collection._bind(
            lambda values: result._commit_collection("archives", values)
        )
        archive_collection._bind_read(
            lambda name, value: result._store.get(_collection_key("archives", name))
        )
        return result

    def _validate_plan_replacement(self, kwargs: dict[str, Any]) -> None:
        evaluation_plan = kwargs.get("evaluation_plan", self.evaluation_plan)
        evaluation_plan_state = kwargs.get(
            "evaluation_plan_state", self.evaluation_plan_state
        )
        _validate_plan_state(evaluation_plan, evaluation_plan_state)
        evaluation_plan_updates = kwargs.get(
            "evaluation_plan_updates", self.evaluation_plan_updates
        )
        if evaluation_plan is not None and evaluation_plan_updates is not None:
            plan_ids = {int(request.request_id) for request in evaluation_plan.requests}
            if not set(map(int, evaluation_plan_updates)) <= plan_ids:
                raise ValidationError(
                    "evaluation plan updates reference an unknown request"
                )

    def _replace_lightweight(self, kwargs: dict[str, Any]) -> OptimizationState:
        """Clone state and rebind collections without rebuilding them."""
        writes = {
            _STORE_FIELDS[name]: value
            for name, value in kwargs.items()
            if name in _STORE_FIELDS
        }
        replacement_store = copy.copy(self._store)
        if writes:
            replacement_store = replacement_store.apply_patch(StatePatch(writes=writes))

        result = object.__new__(type(self))
        object.__setattr__(result, "__dict__", self.__dict__.copy())
        for name, value in kwargs.items():
            if name not in _STORE_FIELDS:
                object.__setattr__(result, name, value)
        for name in _STORE_FIELDS:
            result.__dict__.pop(name, None)
        object.__setattr__(result, "_store", replacement_store)

        population_collection = copy.copy(self._population_collection)
        archive_collection = copy.copy(self._archive_collection)
        object.__setattr__(result, "_population_collection", population_collection)
        object.__setattr__(result, "_archive_collection", archive_collection)
        population_collection._bind(
            lambda values: result._commit_collection("populations", values)
        )
        population_collection._bind_read(
            lambda name, value: result._store.get(_collection_key("populations", name))
        )
        archive_collection._bind(
            lambda values: result._commit_collection("archives", values)
        )
        archive_collection._bind_read(
            lambda name, value: result._store.get(_collection_key("archives", name))
        )
        return result

    def add_population(self, name: str, population: Population) -> None:
        """Add a named population."""
        self.populations[name] = population

    def add_archive(self, name: str, archive: Archive) -> None:
        """Add a named archive."""
        self.archives[name] = archive

    def get_population(self, name: str = "main") -> Population:
        """Return a named population."""
        _validate_collection_name(name)
        return cast(Any, self._store.get(_collection_key("populations", name)))

    def get_archive(self, name: str = "main") -> Archive:
        """Return a named archive."""
        _validate_collection_name(name)
        return cast(Any, self._store.get(_collection_key("archives", name)))

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Rebuild through ``__init__`` so the state store is reconstructed."""
        self.__init__(**state)

    # ------------------------------------------------------------------
    # Convenience properties (delegate to problem)
    # ------------------------------------------------------------------

    @property
    def dim(self) -> int:
        """Return the dimension of the problem."""
        return self.problem.dim

    @property
    def n_obj(self) -> int:
        """Return the number of objectives."""
        return self.problem.n_obj

    @property
    def lb(self) -> np.ndarray:
        """Return the lower bounds of the problem via BoundsService."""
        bounds_srv = cast(BoundsService, self.compiled_service("BoundsService"))
        return bounds_srv.bounds[0]

    @property
    def ub(self) -> np.ndarray:
        """Return the upper bounds of the problem via BoundsService."""
        bounds_srv = cast(BoundsService, self.compiled_service("BoundsService"))
        return bounds_srv.bounds[1]

    @property
    def direction(self) -> np.ndarray:
        """Return the optimization direction of the problem."""
        return self.problem.direction

    @property
    def comparator(self) -> Comparator:
        """Return the comparator of the problem."""
        return self.problem.comparator

    def count_fe(self, count: int = 1) -> None:
        """
        Count function evaluations.

        Parameters
        ----------
        count : int
            Number of function evaluations to add.
        """
        self.fe += count

    def count_generation(self) -> None:
        """Count generations."""
        self.gen += 1

    # ------------------------------------------------------------------
    # Checkpoint: npz (best-effort reproducibility)
    # ------------------------------------------------------------------

    def _save_current(self, path: str | Path) -> None:
        """Save every live store entry in the current per-key envelope."""
        if any(
            not pending.checkpointable and pending.fatal_error is None
            for pending in self.pending_evaluations.values()
        ):
            raise ValidationError(
                "cannot checkpoint while synchronous evaluations are pending"
            )
        p = Path(path)
        if not p.suffix:
            p = p.with_suffix(".npz")
        arrays: dict[str, np.ndarray] = {}
        entries: list[dict[str, Any]] = []
        genome_codec = self.problem.space.services.get("GenomeCodec")
        for index, (key, value) in enumerate(self._store._values.items()):
            if key == EVALUATIONS_OWNERS:
                encoded = {
                    "codec": "owners",
                    "value": [
                        {
                            "request_id": int(request_id),
                            **(
                                {"alias": "offspring"}
                                if owner is self.offspring
                                else next(
                                    (
                                        {"alias": f"population:{name}"}
                                        for name, population in self.populations.items()
                                        if owner is population
                                    ),
                                    {
                                        "entry": _encode_state_value(
                                            StateKey(
                                                namespace="evaluations",
                                                name="owner",
                                                schema_version=1,
                                            ),
                                            owner,
                                            arrays,
                                            f"_entry_{index}__owner_{request_id}",
                                            genome_codec=genome_codec,
                                        )
                                    },
                                )
                            ),
                        }
                        for request_id, owner in value.items()
                    ],
                }
            else:
                encoded = _encode_state_value(
                    key, value, arrays, f"_entry_{index}", genome_codec=genome_codec
                )
            entries.append(
                {
                    "key": _state_key_to_json(key),
                    "target_schema_version": key.schema_version,
                    "value": encoded,
                }
            )
        if self.history is not None:
            channels = sorted(self.history.enabled)
            history_columns: dict[str, list[str]] = {}
            history_blocks: dict[str, list[str]] = {}
            for channel in channels:
                channel_columns = self.history.channel(channel)
                history_columns[channel] = list(channel_columns)
                for column, values in channel_columns.items():
                    arrays[f"_history__{channel}__{column}"] = values
                block_storage = self.history._block_storage(channel)
                history_blocks[channel] = list(block_storage)
                for column, (values, offsets) in block_storage.items():
                    arrays[f"_history__{channel}__{column}"] = values
                    arrays[f"_history__{channel}__{column}__offsets"] = offsets
            arrays["_history_meta"] = _json_array(
                {
                    "channels": channels,
                    "columns": history_columns,
                    "blocks": history_blocks,
                }
            )
        arrays["_checkpoint_schema_version"] = np.array(
            CURRENT_CHECKPOINT_SCHEMA_VERSION, dtype=np.int64
        )
        arrays["_state_entries"] = _json_array(entries)
        np.savez(p, **cast(Any, arrays))

    def save(self, path: str | Path) -> None:
        """Save a portable checkpoint using the current schema version.

        The current schema version is ``CURRENT_CHECKPOINT_SCHEMA_VERSION``;
        :meth:`~saealib.context.OptimizationState.load` also reads every version in
        ``SUPPORTED_CHECKPOINT_SCHEMA_VERSIONS``.
        """
        self._save_current(path)

    @classmethod
    def load(cls, path: str | Path, problem: Problem) -> OptimizationState:
        """
        Restore an OptimizationState from an npz checkpoint file.

        The returned state has ``data["resumed"] = True``.

        Parameters
        ----------
        path : str or Path
            Path to the npz file.  The ``.npz`` extension is added if absent.
        problem : Problem
            The problem instance to attach (must match the one used when saving).

        Returns
        -------
        OptimizationState
        """
        p = Path(path)
        if not p.suffix:
            p = p.with_suffix(".npz")

        try:
            data = np.load(p, allow_pickle=False)
            if "_checkpoint_schema_version" not in data.files:
                raise CheckpointError(
                    "Checkpoint is missing '_checkpoint_schema_version'"
                )
            schema_version = _scalar_int(data["_checkpoint_schema_version"])
            if schema_version not in SUPPORTED_CHECKPOINT_SCHEMA_VERSIONS:
                raise CheckpointError(
                    f"Unsupported checkpoint schema version {schema_version}; "
                    "supported versions are "
                    f"{sorted(SUPPORTED_CHECKPOINT_SCHEMA_VERSIONS)}"
                )
            return _load_checkpoint(cls, data, problem, schema_version)
        except CheckpointError:
            raise
        except (KeyError, TypeError, ValueError, json.JSONDecodeError, OSError) as exc:
            raise CheckpointError(f"Invalid checkpoint: {exc}") from exc


# Keep dataclass field metadata and generated replacement/pickle behavior while
# forcing store-backed fields through __getattr__. A class attribute left behind
# by a field default would satisfy ordinary lookup and silently shadow the store.
for _store_field_name in _STORE_FIELDS:
    if _store_field_name in vars(OptimizationState):
        delattr(OptimizationState, _store_field_name)


def _state_key_to_json(key: StateKey[Any]) -> dict[str, Any]:
    return {
        "namespace": key.namespace,
        "name": key.name,
        "schema_version": key.schema_version,
    }


def _state_key_from_json(value: Any) -> StateKey[Any]:
    if not isinstance(value, dict):
        raise CheckpointError("state entry key is malformed")
    try:
        return StateKey(
            namespace=value["namespace"],
            name=value["name"],
            schema_version=value["schema_version"],
        )
    except (KeyError, TypeError, ValidationError) as exc:
        raise CheckpointError("state entry key is malformed") from exc


def _encode_state_value(
    key: StateKey[Any],
    value: Any,
    arrays: dict[str, np.ndarray],
    prefix: str,
    *,
    genome_codec: Any = None,
) -> Any:
    from saealib.population import Archive, ParetoArchive, Population

    if isinstance(value, Population):
        kind = (
            "archive" if isinstance(value, (Archive, ParetoArchive)) else "population"
        )
        collection_label = f"{kind} {key.namespace}/{key.name}"
        descriptor = _collection_descriptor(kind, key.name, value)
        encoded_genomes = None
        if "x" not in value.schema or kind == "population":
            if genome_codec is None or not hasattr(genome_codec, "encode"):
                raise CheckpointError(
                    f"GenomeCodec is required to save {collection_label}"
                )
            try:
                payload = genome_codec.encode(value.genomes)
            except Exception as exc:
                raise CheckpointError(
                    f"GenomeCodec failed to encode {collection_label}: {exc}"
                ) from exc
            if not isinstance(payload, dict):
                raise CheckpointError(
                    f"GenomeCodec returned an invalid payload for {collection_label}"
                )
            payload_meta = dict(payload)
            genome_array = payload_meta.pop("array", None)
            encoded_genomes = {"payload": _json_safe(payload_meta)}
            if genome_array is not None:
                if not isinstance(genome_array, np.ndarray):
                    raise CheckpointError(
                        f"GenomeCodec payload for {collection_label} "
                        "has a non-NumPy array field"
                    )
                array_key = f"{prefix}__genomes"
                arrays[array_key] = np.array(genome_array, copy=True)
                encoded_genomes["array"] = array_key
        for attr_name, array in value._data.items():
            if kind == "population" and attr_name == "x":
                continue
            if array.dtype == object:
                raise CheckpointError(
                    f"object dtype is not checkpointable: {key.name!r}"
                )
            arrays[f"{prefix}__{_encoded_name(attr_name)}"] = np.array(
                array[: len(value)], copy=True
            )
        result = {"codec": "collection", "kind": kind, "descriptor": descriptor}
        if encoded_genomes is not None:
            result["genomes"] = encoded_genomes
        return result
    if key == RUNTIME_RNG:
        return {"codec": "json", "value": _json_safe(value.bit_generator.state)}
    if key in {
        RUNTIME_CANDIDATE_ID_ALLOCATOR,
        PROPOSALS_ID_ALLOCATOR,
        RUNTIME_REQUEST_ID_ALLOCATOR,
    }:
        return {"codec": "scalar", "value": value.next_value}
    if key == FEEDBACK_RESULT:
        return {
            "codec": "feedback",
            "value": None if value is None else _feedback_to_json(value),
        }
    if key == FEEDBACK_ACCUMULATOR:
        return {"codec": "feedback_accumulator", "value": value}
    if key == SURROGATES_PREDICTIONS:
        return {
            "codec": "prediction",
            "value": None if value is None else _prediction_to_json(value),
        }
    if (
        key.namespace == EVALUATIONS_PLAN.namespace
        and key.name == EVALUATIONS_PLAN.name
    ):
        return {
            "codec": "plan",
            "value": None if value is None else _plan_to_json(value),
        }
    if (
        key.namespace == EVALUATIONS_PLAN_STATE.namespace
        and key.name == EVALUATIONS_PLAN_STATE.name
    ):
        return {
            "codec": "plan_state",
            "value": None if value is None else _plan_state_to_json(value),
        }
    if (
        key.namespace == EVALUATIONS_PLAN_UPDATES.namespace
        and key.name == EVALUATIONS_PLAN_UPDATES.name
    ):
        return {
            "codec": "updates",
            "value": {
                str(k): [_update_to_json(item) for item in v] for k, v in value.items()
            },
        }
    if (
        key.namespace == PENDING_EVALUATIONS.namespace
        and key.name == PENDING_EVALUATIONS.name
    ):
        return {
            "codec": "pending",
            "value": {str(k): _pending_to_json(item) for k, item in value.items()},
        }
    if key == EVALUATIONS_OWNERS:
        return {
            "codec": "owners",
            "value": [
                {
                    "request_id": int(k),
                    "entry": _encode_state_value(
                        StateKey(
                            namespace="evaluations", name="owner", schema_version=1
                        ),
                        owner,
                        arrays,
                        f"{prefix}__owner_{k}",
                        genome_codec=genome_codec,
                    ),
                }
                for k, owner in value.items()
            ],
        }
    return {"codec": "json", "value": _json_safe(value)}


def _decode_state_value(
    key: StateKey[Any],
    encoded: Any,
    data: Any,
    prefix: str,
    *,
    genome_codec: Any = None,
    dense_numeric_view: Any = None,
    space: Any = None,
    objective_source_targets: list[Any] | None = None,
) -> Any:
    if not isinstance(encoded, dict) or not isinstance(encoded.get("codec"), str):
        raise CheckpointError(
            f"state entry {key.namespace}/{key.name} has malformed value"
        )
    codec = encoded["codec"]
    value = encoded.get("value")
    if codec == "collection":
        if not isinstance(encoded.get("descriptor"), dict):
            raise CheckpointError("state collection descriptor is malformed")
        genome_payload = encoded.get("genomes")
        return _restore_collection(
            encoded["kind"],
            encoded["descriptor"],
            data,
            prefix,
            genome_codec=genome_codec,
            dense_numeric_view=dense_numeric_view,
            genome_payload=genome_payload,
            space=space,
            objective_source_targets=objective_source_targets,
        )
    if codec == "json":
        return value
    if codec == "alias":
        return value
    if codec == "scalar":
        return _allocator_scalar(np.array(value), key.name)
    if codec == "feedback":
        return None if value is None else _feedback_from_json(value)
    if codec == "feedback_accumulator":
        from saealib.core.adapters.accumulator import _validate_state

        if value is not None:
            _validate_state(value)
        return value
    if codec == "prediction":
        return None if value is None else _prediction_from_json(value)
    if codec == "plan":
        return None if value is None else _plan_from_json(value)
    if codec == "plan_state":
        return None if value is None else _plan_state_from_json(value)
    if codec == "updates":
        if not isinstance(value, dict):
            raise CheckpointError("state evaluation plan updates are malformed")
        return {
            int(k): [_update_from_json(item) for item in v] for k, v in value.items()
        }
    if codec == "pending":
        if not isinstance(value, dict):
            raise CheckpointError("state pending evaluations are malformed")
        result = {}
        for request_id, item in value.items():
            restored = _pending_from_json(item)
            if not restored.checkpointable and restored.fatal_error is None:
                raise CheckpointError("non-checkpointable pending evaluation")
            result[int(request_id)] = restored
        return result
    if codec == "owners":
        if not isinstance(value, list):
            raise CheckpointError("state evaluation owners are malformed")
        result = {}
        for item in value:
            request_id = int(item["request_id"])
            if "alias" in item:
                result[request_id] = {"__state_alias__": item["alias"]}
            else:
                result[request_id] = _decode_state_value(
                    StateKey(namespace="evaluations", name="owner", schema_version=1),
                    item["entry"],
                    data,
                    f"{prefix}__owner_{request_id}",
                    genome_codec=genome_codec,
                    dense_numeric_view=dense_numeric_view,
                    space=space,
                    objective_source_targets=objective_source_targets,
                )
        return result
    raise CheckpointError(f"unknown state value codec {codec!r}")


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        if value.dtype == object:
            raise CheckpointError("object dtype is not checkpointable")
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise CheckpointError("checkpoint metadata keys must be strings")
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise CheckpointError(f"non-serializable checkpoint value: {type(value).__name__}")


def _json_array(value: Any) -> np.ndarray:
    try:
        encoded = json.dumps(_json_safe(value), allow_nan=True).encode()
    except (TypeError, ValueError) as exc:
        raise CheckpointError(f"value is not JSON serializable: {exc}") from exc
    return np.frombuffer(encoded, dtype=np.uint8)


def _read_json(data: Any, key: str) -> Any:
    if key not in data.files:
        raise CheckpointError(f"Checkpoint is missing {key!r}")
    try:
        return json.loads(bytes(data[key]).decode())
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise CheckpointError(f"Checkpoint key {key!r} is malformed") from exc


def _result_to_json(result: Any) -> dict[str, Any]:
    return {
        "f": _json_safe(result.f),
        "g": _json_safe(result.g),
        "cv": _json_safe(result.cv),
        "candidate_ids": _json_safe(result.candidate_ids),
        "cost": _json_safe(result.cost),
        "noise": _json_safe(result.noise),
        "outputs": _json_safe(result.outputs),
    }


def _result_from_json(value: Any) -> Any:
    from saealib.execution.evaluator import EvaluationResult

    if not isinstance(value, dict):
        raise CheckpointError("pending result is malformed")
    return EvaluationResult(
        np.asarray(value["f"], dtype=np.float64),
        np.asarray(value["g"], dtype=np.float64),
        np.asarray(value["cv"], dtype=np.float64),
        None
        if value.get("candidate_ids") is None
        else np.asarray(value["candidate_ids"], dtype=np.int64),
        None
        if value.get("cost") is None
        else np.asarray(value["cost"], dtype=np.float64),
        None
        if value.get("noise") is None
        else np.asarray(value["noise"], dtype=np.float64),
        {
            str(key): np.asarray(array, dtype=np.float64)
            for key, array in value.get("outputs", {}).items()
        },
    )


def _feedback_to_json(result: Any) -> dict[str, Any]:
    return {
        "candidate_ids": _json_safe(result.candidate_ids),
        "f": _json_safe(result.f),
        "g": _json_safe(result.g),
        "cv": _json_safe(result.cv),
        "evaluated_mask": _json_safe(result.evaluated_mask),
        "source": _json_safe(result.source),
        "artifacts": _json_safe(result.artifacts),
    }


def _feedback_from_json(value: Any) -> Any:
    from saealib.policies.feedback import FeedbackResult

    if not isinstance(value, dict):
        raise CheckpointError("feedback result is malformed")
    return FeedbackResult(
        np.asarray(value["candidate_ids"], dtype=np.int64),
        np.asarray(value["f"], dtype=np.float64),
        None if value.get("g") is None else np.asarray(value["g"], dtype=np.float64),
        None if value.get("cv") is None else np.asarray(value["cv"], dtype=np.float64),
        np.asarray(value["evaluated_mask"], dtype=bool),
        np.asarray(value["source"], dtype=np.uint8),
        {
            str(name): np.asarray(array)
            for name, array in value.get("artifacts", {}).items()
        },
    )


def _payload_to_json(payload: Any) -> dict[str, Any]:
    from saealib.population.genome import (
        DenseVectorBatch,
        ObjectBatch,
        PermutationBatch,
        VariableLengthBatch,
    )

    if isinstance(payload, DenseVectorBatch):
        return {"kind": "dense_vector", "items": _json_safe(payload.array)}
    if isinstance(payload, ObjectBatch):
        return {"kind": "object", "items": _json_safe(payload.items)}
    if isinstance(payload, PermutationBatch):
        return {
            "kind": "permutation",
            "length": payload.length,
            "items": _json_safe(payload.array),
        }
    if isinstance(payload, VariableLengthBatch):
        return {"kind": "sequence", "items": _json_safe(payload.sequences)}
    raise CheckpointError(
        "evaluation request payload is not JSON-checkpointable: "
        f"{type(payload).__name__}"
    )


def _payload_from_json(value: Any) -> Any:
    from saealib.population.genome import (
        DenseVectorBatch,
        ObjectBatch,
        PermutationBatch,
        VariableLengthBatch,
    )

    if not isinstance(value, dict) or not isinstance(value.get("kind"), str):
        raise CheckpointError("evaluation request payload is malformed")
    try:
        if value["kind"] == "dense_vector":
            return DenseVectorBatch(np.asarray(value["items"], dtype=np.float64))
        if value["kind"] == "object":
            return ObjectBatch(value["items"])
        if value["kind"] == "permutation":
            return PermutationBatch(value["items"], length=int(value["length"]))
        if value["kind"] == "sequence":
            return VariableLengthBatch(value["items"])
    except (KeyError, TypeError, ValueError, ValidationError) as exc:
        raise CheckpointError("evaluation request payload is malformed") from exc
    raise CheckpointError(f"unknown evaluation request payload kind {value['kind']!r}")


def _request_to_json(request: Any) -> dict[str, Any]:
    return {
        "request_id": int(request.request_id),
        "candidate_ids": _json_safe(request.candidate_ids),
        "payload": _payload_to_json(request.payload),
        "outputs": list(request.outputs),
        "metadata": _json_safe(dict(request.metadata)),
    }


def _request_from_json(value: Any) -> Any:
    from saealib.execution.evaluator import EvaluationRequest

    if not isinstance(value, dict):
        raise CheckpointError("evaluation request is malformed")
    try:
        payload = _payload_from_json(value["payload"])
        return EvaluationRequest(
            np.int64(value["request_id"]),
            np.asarray(value["candidate_ids"], dtype=np.int64),
            payload,
            tuple(value.get("outputs", ("f", "g", "cv"))),
            value.get("metadata", {}),
        )
    except (KeyError, TypeError, ValueError, ValidationError) as exc:
        raise CheckpointError("evaluation request is malformed") from exc


def _plan_to_json(plan: Any) -> dict[str, Any]:
    return {
        "requests": [_request_to_json(request) for request in plan.requests],
        "completion_rule": _json_safe(plan.completion_rule),
        "continuation": _json_safe(plan.continuation),
        "artifacts": _json_safe(dict(plan.artifacts)),
    }


def _plan_from_json(value: Any) -> Any:
    from saealib.policies.evaluation import EvaluationPlan

    if not isinstance(value, dict) or not isinstance(value.get("requests"), list):
        raise CheckpointError("evaluation plan is malformed")
    try:
        return EvaluationPlan(
            tuple(_request_from_json(item) for item in value["requests"]),
            value.get("completion_rule"),
            value.get("continuation"),
            value.get("artifacts", {}),
        )
    except (TypeError, ValueError, ValidationError) as exc:
        raise CheckpointError("evaluation plan is malformed") from exc


def _plan_state_to_json(value: EvaluationPlanState) -> dict[str, Any]:
    return {
        "submitted": list(value.submitted),
        "completed": list(value.completed),
        "acknowledged": list(value.acknowledged),
        "deferred": list(value.deferred),
        "continuation": _json_safe(value.continuation),
        "feedback": _json_safe(value.feedback),
    }


def _plan_state_from_json(value: Any) -> EvaluationPlanState:
    if not isinstance(value, dict):
        raise CheckpointError("evaluation plan state is malformed")
    try:
        return EvaluationPlanState(
            tuple(value.get("submitted", ())),
            tuple(value.get("completed", ())),
            tuple(value.get("acknowledged", ())),
            tuple(value.get("deferred", ())),
            value.get("continuation"),
            value.get("feedback"),
        )
    except (TypeError, ValueError, ValidationError) as exc:
        raise CheckpointError("evaluation plan state is malformed") from exc


def _update_to_json(update: Any) -> dict[str, Any]:
    return {
        "request_id": int(update.request_id),
        "status": update.status.name,
        "candidate_ids": _json_safe(update.candidate_ids),
        "result": None if update.result is None else _result_to_json(update.result),
        "error": None
        if update.error is None
        else {
            "error_type": update.error.error_type,
            "message": update.error.message,
            "details": _json_safe(dict(update.error.details)),
        },
        "sequence": int(update.sequence),
    }


def _update_from_json(value: Any) -> Any:
    from saealib.execution.evaluator import (
        EvaluationErrorInfo,
        EvaluationStatus,
        EvaluationUpdate,
    )

    if not isinstance(value, dict):
        raise CheckpointError("evaluation plan update is malformed")
    error = value.get("error")
    try:
        return EvaluationUpdate(
            np.int64(value["request_id"]),
            EvaluationStatus[value["status"]],
            np.asarray(value["candidate_ids"], dtype=np.int64),
            None if value.get("result") is None else _result_from_json(value["result"]),
            None
            if error is None
            else EvaluationErrorInfo(
                error["error_type"], error["message"], error.get("details", {})
            ),
            int(value.get("sequence", 0)),
        )
    except (KeyError, TypeError, ValueError, ValidationError) as exc:
        raise CheckpointError("evaluation plan update is malformed") from exc


def _prediction_to_json(prediction: Any) -> dict[str, Any]:
    return {
        "x": _json_safe(prediction.x),
        "label": _json_safe(prediction.label),
        "metadata": _json_safe(prediction.metadata),
        "channels": {
            name: {
                "value": _json_safe(channel.value),
                "std": _json_safe(channel.std),
                "covariance": _json_safe(channel.covariance),
                "samples": _json_safe(channel.samples),
                "metadata": _json_safe(channel.metadata),
            }
            for name, channel in prediction.channels.items()
        },
    }


def _prediction_from_json(value: Any) -> Any:
    from saealib.surrogate.prediction import PredictionChannel, SurrogatePrediction

    if not isinstance(value, dict) or not isinstance(value.get("channels"), dict):
        raise CheckpointError("prediction is malformed")
    channels = {}
    for name, channel in value["channels"].items():
        if not isinstance(channel, dict):
            raise CheckpointError("prediction channel is malformed")
        channels[name] = PredictionChannel(
            np.asarray(channel["value"], dtype=np.float64),
            None
            if channel.get("std") is None
            else np.asarray(channel["std"], dtype=np.float64),
            None
            if channel.get("covariance") is None
            else np.asarray(channel["covariance"], dtype=np.float64),
            None
            if channel.get("samples") is None
            else np.asarray(channel["samples"], dtype=np.float64),
            channel.get("metadata", {}),
        )
    return SurrogatePrediction(
        channels,
        None if value.get("x") is None else np.asarray(value["x"]),
        None if value.get("label") is None else np.asarray(value["label"]),
        value.get("metadata", {}),
    )


def _pending_to_json(pending: Any) -> dict[str, Any]:
    request = pending.request
    return {
        "request_id": int(request.request_id),
        "candidate_ids": _json_safe(request.candidate_ids),
        "payload": _payload_to_json(request.payload),
        "outputs": list(request.outputs),
        "metadata": _json_safe(dict(request.metadata)),
        "status": pending.status.name,
        "applied_candidate_ids": _json_safe(pending.applied_candidate_ids),
        "last_delivered_sequence": pending.last_delivered_sequence,
        "last_acknowledged_sequence": pending.last_acknowledged_sequence,
        "processing": {str(key): value for key, value in pending.processing.items()},
        "buffered_updates": [
            {
                "request_id": int(update.request_id),
                "status": update.status.name,
                "candidate_ids": _json_safe(update.candidate_ids),
                "result": None
                if update.result is None
                else _result_to_json(update.result),
                "error": None
                if update.error is None
                else {
                    "error_type": update.error.error_type,
                    "message": update.error.message,
                    "details": _json_safe(dict(update.error.details)),
                },
                "sequence": update.sequence,
            }
            for update in pending.buffered_updates
        ],
        "reserved_cost": pending.reserved_cost,
        "retry_count": pending.retry_count,
        "checkpointable": pending.checkpointable,
        "original_candidate_ids": _json_safe(pending.original_candidate_ids),
        "feedback_result": None
        if pending.feedback_result is None
        else _feedback_to_json(pending.feedback_result),
        "fatal_error": None
        if pending.fatal_error is None
        else {
            "error_type": pending.fatal_error.error_type,
            "message": pending.fatal_error.message,
            "details": _json_safe(dict(pending.fatal_error.details)),
        },
        "prediction": None
        if pending.prediction is None
        else _prediction_to_json(pending.prediction),
    }


def _pending_from_json(value: Any) -> Any:
    from saealib.execution.evaluator import (
        EvaluationErrorInfo,
        EvaluationRequest,
        EvaluationStatus,
        EvaluationUpdate,
        PendingEvaluation,
    )

    if not isinstance(value, dict):
        raise CheckpointError("pending evaluation is malformed")
    try:
        payload = (
            _payload_from_json(value["payload"])
            if "payload" in value
            else np.asarray(value["x"], dtype=np.float64)
        )
        request = EvaluationRequest(
            np.int64(value["request_id"]),
            np.asarray(value["candidate_ids"], dtype=np.int64),
            payload,
            tuple(value.get("outputs", ("f", "g", "cv"))),
            value.get("metadata", {}),
        )
        updates = []
        for item in value.get("buffered_updates", []):
            error = item.get("error")
            updates.append(
                EvaluationUpdate(
                    np.int64(item["request_id"]),
                    EvaluationStatus[item["status"]],
                    np.asarray(item["candidate_ids"], dtype=np.int64),
                    None
                    if item.get("result") is None
                    else _result_from_json(item["result"]),
                    None
                    if error is None
                    else EvaluationErrorInfo(
                        error["error_type"], error["message"], error.get("details", {})
                    ),
                    int(item["sequence"]),
                )
            )
        return PendingEvaluation(
            request,
            EvaluationStatus[value["status"]],
            np.asarray(value["applied_candidate_ids"], dtype=np.int64),
            int(value["last_delivered_sequence"]),
            int(value["last_acknowledged_sequence"]),
            {
                int(key): str(status)
                for key, status in value.get("processing", {}).items()
            },
            tuple(updates),
            float(value.get("reserved_cost", 0.0)),
            int(value.get("retry_count", 0)),
            bool(value.get("checkpointable", False)),
            None
            if value.get("original_candidate_ids") is None
            else np.asarray(value["original_candidate_ids"], dtype=np.int64),
            None
            if value.get("feedback_result") is None
            else _feedback_from_json(value["feedback_result"]),
            None
            if value.get("fatal_error") is None
            else EvaluationErrorInfo(
                value["fatal_error"]["error_type"],
                value["fatal_error"]["message"],
                value["fatal_error"].get("details", {}),
            ),
            None
            if value.get("prediction") is None
            else _prediction_from_json(value["prediction"]),
        )
    except (
        AttributeError,
        IndexError,
        KeyError,
        TypeError,
        ValueError,
        ValidationError,
    ) as exc:
        raise CheckpointError("pending evaluation is malformed") from exc


def _scalar_int(value: Any) -> int:
    array = np.asarray(value)
    if array.shape != () or array.dtype.kind not in "iu":
        raise CheckpointError("checkpoint scalar must be an integer")
    return int(array)


def _allocator_scalar(value: Any, name: str) -> int:
    result = _scalar_int(value)
    if result < 0 or result > np.iinfo(np.int64).max:
        raise CheckpointError(f"{name} is outside the int64 allocator range")
    return result


def _default_json(value: Any) -> Any:
    if isinstance(value, (float, np.floating)) and np.isnan(value):
        return "__nan__"
    return _json_safe(value)


def _default_value(value: Any) -> Any:
    if isinstance(value, str) and value == "__nan__":
        return np.nan
    return value


def _attribute_descriptors(collection: Any) -> list[dict[str, Any]]:
    result = []
    for name, attr in collection.schema.items():
        dtype = np.dtype(attr.dtype)
        if dtype == np.dtype(object):
            raise CheckpointError(f"object dtype is not checkpointable: {name!r}")
        result.append(
            {
                "name": name,
                "dtype": dtype.str,
                "shape": list(attr.shape),
                "default": _default_json(attr.default),
            }
        )
    return result


def _collection_descriptor(kind: str, name: str, collection: Any) -> dict[str, Any]:
    _validate_collection_name(name)
    descriptor: dict[str, Any] = {
        "name": name,
        "subtype": "Population" if kind == "population" else "Archive",
        "schema": _attribute_descriptors(collection),
        "size": len(collection),
        "capacity": collection._capacity,
    }
    if kind == "archive":
        from saealib.population import ParetoArchive

        if isinstance(collection, ParetoArchive):
            descriptor["subtype"] = "ParetoArchive"
            descriptor["direction"] = _json_safe(collection.direction)
            descriptor["eps_cv"] = _json_safe(collection.eps_cv)
        descriptor.update(
            {
                "duplicate_policy": getattr(
                    collection, "duplicate_policy", "keep_first"
                ),
                "key_attr": getattr(collection, "key_attr", "x"),
                "atol": getattr(collection, "atol", 0.0),
                "rtol": getattr(collection, "rtol", 0.0),
            }
        )
    return descriptor


def _attrs_from_descriptor(descriptor: dict[str, Any]) -> list[Any]:
    from saealib.population import PopulationAttribute

    schema = descriptor.get("schema")
    if not isinstance(schema, list):
        raise CheckpointError("collection schema must be a list")
    attrs = []
    seen: set[str] = set()
    for item in schema:
        if not isinstance(item, dict) or not isinstance(item.get("name"), str):
            raise CheckpointError("malformed collection attribute descriptor")
        name = item["name"]
        if name in seen:
            raise CheckpointError(f"duplicate attribute name: {name!r}")
        seen.add(name)
        try:
            dtype = np.dtype(item["dtype"])
            shape = tuple(item["shape"])
            if dtype == np.dtype(object) or any(
                not isinstance(axis, int) or axis < 0 for axis in shape
            ):
                raise ValueError
            attrs.append(
                PopulationAttribute(
                    name=name,
                    dtype=dtype,
                    shape=shape,
                    default=_default_value(item.get("default")),
                )
            )
        except (TypeError, ValueError) as exc:
            raise CheckpointError(f"malformed attribute {name!r}") from exc
    return attrs


def _restore_collection(
    kind: str,
    descriptor: dict[str, Any],
    data: Any,
    storage_prefix: str | None = None,
    *,
    genome_codec: Any = None,
    dense_numeric_view: Any = None,
    genome_payload: Any = None,
    space: Any = None,
    objective_source_targets: list[Any] | None = None,
) -> Any:
    """Rebuild one population or archive from its checkpoint descriptor.

    When ``objective_source_targets`` is a list, a collection without an
    ``objective_source`` column gets one filled with the unknown code and is
    appended to that list.
    """
    from saealib.population import Archive, ParetoArchive, Population
    from saealib.population.archive import _validate_observation_schema
    from saealib.population.population import (
        OBJECTIVE_SOURCE_COLUMN,
        UNKNOWN_SOURCE_CODE,
        PopulationAttribute,
    )

    name = descriptor.get("name")
    if not isinstance(name, str):
        raise CheckpointError("collection name is missing")
    _validate_collection_name(name)
    size = descriptor.get("size")
    capacity = descriptor.get("capacity")
    if (
        not isinstance(size, int)
        or not isinstance(capacity, int)
        or size < 0
        or capacity < size
    ):
        raise CheckpointError(f"invalid size/capacity for collection {name!r}")
    attrs = _attrs_from_descriptor(descriptor)
    stored_names = {attr.name for attr in attrs}
    add_objective_source = (
        objective_source_targets is not None
        and OBJECTIVE_SOURCE_COLUMN not in stored_names
    )
    if add_objective_source:
        attrs.append(
            PopulationAttribute(
                OBJECTIVE_SOURCE_COLUMN, np.uint8, (), default=UNKNOWN_SOURCE_CODE
            )
        )
    schema = {attr.name: attr for attr in attrs}
    for id_name in ("id", "request_id"):
        if id_name in schema and np.dtype(schema[id_name].dtype) != np.dtype(np.int64):
            raise CheckpointError(f"{id_name} column must use int64")
    collection_label = (
        f"{kind} {'populations' if kind == 'population' else 'archives'}/{name}"
    )
    genomes = None
    if genome_payload is not None:
        if genome_codec is None or not hasattr(genome_codec, "decode"):
            raise CheckpointError(f"GenomeCodec is required to load {collection_label}")
        if not isinstance(genome_payload, dict):
            raise CheckpointError("collection GenomeCodec payload is malformed")
        array_key = genome_payload.get("array")
        payload_meta = genome_payload.get("payload")
        if not isinstance(payload_meta, dict):
            raise CheckpointError(
                f"GenomeCodec payload for {collection_label} is malformed"
            )
        payload = dict(payload_meta)
        if array_key is not None:
            if not isinstance(array_key, str) or array_key not in data.files:
                raise CheckpointError(
                    f"GenomeCodec payload for {collection_label} is missing its array"
                )
            payload["array"] = np.asarray(data[array_key])
        try:
            genomes = genome_codec.decode(payload)
        except Exception as exc:
            raise CheckpointError(
                f"GenomeCodec failed to decode {collection_label}: {exc}"
            ) from exc
        if len(genomes) != size:
            raise CheckpointError(
                f"GenomeCodec decoded {collection_label} with invalid size"
            )

    subtype = descriptor.get("subtype")
    if kind == "population":
        if subtype != "Population":
            raise CheckpointError(f"unknown population subtype: {subtype!r}")
        collection = Population(
            attrs,
            capacity,
            genomes=None if genomes is None else genomes.take([]),
            dense_numeric_view=dense_numeric_view,
            space=space,
        )
    else:
        if subtype not in {"Archive", "ParetoArchive"}:
            raise CheckpointError(f"unknown archive subtype: {subtype!r}")
        params = {
            "key_attr": descriptor.get("key_attr", "x"),
            "atol": descriptor.get("atol", 0.0),
            "rtol": descriptor.get("rtol", 0.0),
            "duplicate_policy": descriptor.get("duplicate_policy", "keep_first"),
        }
        if subtype == "ParetoArchive" and params["duplicate_policy"] == "append":
            raise CheckpointError("ParetoArchive does not support append policy")
        if subtype == "Archive":
            _validate_observation_schema(attrs, params["duplicate_policy"])
        if subtype == "ParetoArchive":
            direction_value = descriptor.get("direction")
            archive_kwargs = (
                {
                    "genomes": genomes.take([]),
                    "space": space,
                }
                if genomes is not None
                else {}
            )
            collection = ParetoArchive(
                attrs,
                capacity,
                direction=(
                    None
                    if direction_value is None
                    else np.asarray(direction_value, dtype=np.float64)
                ),
                eps_cv=float(descriptor.get("eps_cv", 0.0)),
                **params,
                **archive_kwargs,
            )
        else:
            collection = Archive(
                attrs,
                capacity,
                **params,
                **(
                    {
                        "genomes": genomes.take([]),
                        "space": space,
                    }
                    if genomes is not None
                    else {}
                ),
            )
    encoded = _encoded_name(name)
    values: dict[str, np.ndarray] = {}
    if (
        kind in {"population", "archive"}
        and genome_payload is not None
        and (genome_codec is None or not hasattr(genome_codec, "decode"))
    ):
        raise CheckpointError(f"GenomeCodec is required to load {collection_label}")
    for attr in attrs:
        if kind == "population" and genome_payload is not None and attr.name == "x":
            continue
        if attr.name not in stored_names:
            continue
        if storage_prefix is None:
            key = f"{kind}__{encoded}__{_encoded_name(attr.name)}"
        else:
            key = f"{storage_prefix}__{_encoded_name(attr.name)}"
        if key not in data.files:
            raise CheckpointError(f"Checkpoint is missing array {key!r}")
        array = np.asarray(data[key])
        expected = (size, *attr.shape)
        if array.dtype != np.dtype(attr.dtype) or array.shape != expected:
            raise CheckpointError(f"array {key!r} has an invalid dtype or shape")
        values[attr.name] = np.array(array, copy=True)
    if getattr(collection, "duplicate_policy", None) == "append":
        if not {"id", "request_id"}.issubset(schema) or any(
            np.dtype(schema[name].dtype) != np.dtype(np.int64)
            for name in ("id", "request_id")
        ):
            raise CheckpointError(
                "append archives require int64 id and request_id columns"
            )
        if size and "id" in values:
            pairs = np.column_stack((values["request_id"], values["id"]))
            if len(np.unique(pairs, axis=0)) != len(pairs):
                raise CheckpointError("duplicate (request_id, candidate_id) pair")
    if (
        kind == "population"
        and genomes is not None
        and "x" in schema
        and hasattr(genomes, "array")
    ):
        values["x"] = np.asarray(genomes.array)
    if genomes is not None:
        values["genomes"] = genomes
    if size:
        collection._extend_internal(
            values,
            preserve_ids=True,
            allow_duplicate_ids=(
                getattr(collection, "duplicate_policy", None) == "append"
            ),
        )
    if add_objective_source and objective_source_targets is not None:
        objective_source_targets.append(collection)
    return collection


def _fill_objective_source_from_history(collections: list[Any], history: Any) -> None:
    """Mark rows whose ``f`` matches their last recorded true evaluation as true.

    Every other row keeps the unknown code, as do all rows when the History
    ``evaluation`` channel was not recorded.
    """
    from saealib.execution.history import _true_objectives_by_candidate
    from saealib.policies.feedback import LEGACY_SOURCE_CODES, TRUE
    from saealib.population.population import (
        OBJECTIVE_SOURCE_COLUMN,
        UNKNOWN_SOURCE_CODE,
    )

    truth = {} if history is None else _true_objectives_by_candidate(history)
    if not truth:
        return
    for collection in collections:
        if "id" not in collection.schema or "f" not in collection.schema:
            continue
        ids = collection.get_array("id")
        f = collection.get_array("f")
        codes = np.full(len(collection), UNKNOWN_SOURCE_CODE, dtype=np.uint8)
        for row, candidate_id in enumerate(ids.tolist()):
            value = truth.get(candidate_id)
            if value is not None and np.array_equal(value, f[row], equal_nan=True):
                codes[row] = LEGACY_SOURCE_CODES[TRUE]
        collection.update_array(OBJECTIVE_SOURCE_COLUMN, codes)


def _load_checkpoint(
    cls: type[OptimizationState],
    data: Any,
    problem: Problem,
    schema_version: int = CURRENT_CHECKPOINT_SCHEMA_VERSION,
) -> OptimizationState:
    """Load a checkpoint written with any supported schema version.

    Version 1 stores no ``objective_source`` column; it is added to every
    restored population and archive and filled from the History
    ``evaluation`` channel.
    """
    objective_source_targets: list[Any] | None = [] if schema_version == 1 else None
    genome_codec = problem.space.services.get("GenomeCodec")
    dense_numeric_view = problem.space.services.get("DenseNumericView")
    entries = _read_json(data, "_state_entries")
    if not isinstance(entries, list):
        raise CheckpointError("state entries are malformed")
    restored: dict[StateKey[Any], Any] = {}
    for index, item in enumerate(entries):
        if not isinstance(item, dict):
            raise CheckpointError("state entry is malformed")
        key = _state_key_from_json(item.get("key"))
        target = item.get("target_schema_version")
        if isinstance(target, bool) or not isinstance(target, int) or target < 1:
            raise CheckpointError(
                f"state entry {key.namespace}/{key.name} target version is malformed"
            )
        expected_target = target
        for expected_key in _STORE_FIELDS.values():
            if (
                expected_key.namespace == key.namespace
                and expected_key.name == key.name
            ):
                expected_target = max(expected_target, expected_key.schema_version)
                break
        target = max(target, expected_target)
        if key.schema_version != target:
            raise CheckpointError(
                f"State key {key.namespace}/{key.name} is at schema version "
                f"v{key.schema_version} but the checkpoint targets v{target}; "
                "this version provides no state migration."
            )
        value = _decode_state_value(
            key,
            item.get("value"),
            data,
            f"_entry_{index}",
            genome_codec=genome_codec,
            dense_numeric_view=dense_numeric_view,
            space=problem.space,
            objective_source_targets=objective_source_targets,
        )
        if key in restored:
            raise CheckpointError(f"duplicate state entry {key.namespace}/{key.name}")
        restored[key] = value

    populations = {
        key.name: value
        for key, value in restored.items()
        if key.namespace == "populations"
    }
    archives = {
        key.name: value
        for key, value in restored.items()
        if key.namespace == "archives"
    }
    if "main" not in populations or not {"main", "pareto"} <= set(archives):
        raise CheckpointError(
            "state entries are missing required main/pareto collections"
        )
    values = {
        name: restored[key] for name, key in _STORE_FIELDS.items() if key in restored
    }
    if "rng" not in values or not isinstance(values["rng"], dict):
        raise CheckpointError("runtime/rng state is missing or malformed")
    rng = np.random.default_rng()
    rng.bit_generator.state = values.pop("rng")
    candidate_allocator = IDAllocator(values.pop("candidate_id_allocator"))
    request_allocator = IDAllocator(values.pop("request_id_allocator"))
    proposal_allocator = IDAllocator(
        values.pop("proposal_id_allocator", request_allocator.next_value)
    )
    offspring = values.pop("offspring", None)
    owners = values.get("evaluation_owners", {})
    if isinstance(owners, dict):
        values["evaluation_owners"] = {
            request_id: (
                offspring
                if isinstance(owner, dict)
                and owner.get("__state_alias__") == "offspring"
                else next(
                    (
                        population
                        for name, population in populations.items()
                        if isinstance(owner, dict)
                        and owner.get("__state_alias__") == f"population:{name}"
                    ),
                    owner,
                )
            )
            for request_id, owner in owners.items()
        }
    custom = {
        key: value
        for key, value in restored.items()
        if key not in _STORE_FIELDS.values()
        and key.namespace not in {"populations", "archives"}
    }
    kwargs: dict[str, Any] = {
        "problem": problem,
        "populations": populations,
        "archives": archives,
        "rng": rng,
        "candidate_id_allocator": candidate_allocator,
        "proposal_id_allocator": proposal_allocator,
        "request_id_allocator": request_allocator,
        "offspring": offspring,
        "_custom_state": custom,
        "data": values.pop("data", {}),
    }
    history = None
    if "_history_meta" in data.files:
        from saealib.execution.history import (
            History,
            _fill_missing_evaluation_columns,
        )

        metadata = _read_json(data, "_history_meta")
        if not isinstance(metadata, dict):
            raise CheckpointError("checkpoint history metadata is malformed")
        channels = metadata.get("channels")
        columns_by_channel = metadata.get("columns")
        blocks_by_channel = metadata.get(
            "blocks",
            {channel: [] for channel in channels}
            if isinstance(channels, list)
            else None,
        )
        if (
            not isinstance(channels, list)
            or any(not isinstance(channel, str) for channel in channels)
            or len(set(channels)) != len(channels)
            or not isinstance(columns_by_channel, dict)
            or set(columns_by_channel) != set(channels)
            or not isinstance(blocks_by_channel, dict)
            or set(blocks_by_channel) != set(channels)
        ):
            raise CheckpointError("checkpoint history metadata is malformed")

        history = History(channels=channels)
        for channel in channels:
            column_names = columns_by_channel[channel]
            if (
                not isinstance(column_names, list)
                or any(not isinstance(column, str) for column in column_names)
                or len(set(column_names)) != len(column_names)
            ):
                raise CheckpointError("checkpoint history metadata is malformed")
            columns: dict[str, np.ndarray] = {}
            for column in column_names:
                array_key = f"_history__{channel}__{column}"
                if array_key not in data.files:
                    raise CheckpointError(f"Checkpoint is missing array {array_key!r}")
                columns[column] = np.asarray(data[array_key])
            history._restore_channel(channel, columns)
            block_names = blocks_by_channel[channel]
            if (
                not isinstance(block_names, list)
                or any(not isinstance(column, str) for column in block_names)
                or len(set(block_names)) != len(block_names)
            ):
                raise CheckpointError("checkpoint history metadata is malformed")
            block_columns: dict[str, tuple[np.ndarray, np.ndarray]] = {}
            for column in block_names:
                value_key = f"_history__{channel}__{column}"
                offsets_key = f"{value_key}__offsets"
                if value_key not in data.files:
                    raise CheckpointError(f"Checkpoint is missing array {value_key!r}")
                if offsets_key not in data.files:
                    raise CheckpointError(
                        f"Checkpoint is missing array {offsets_key!r}"
                    )
                block_columns[column] = (
                    np.asarray(data[value_key]),
                    np.asarray(data[offsets_key]),
                )
            history._restore_blocks(channel, block_columns)
            if channel == "evaluation":
                _fill_missing_evaluation_columns(
                    history, problem.n_obj, problem.n_constraints
                )
    kwargs["history"] = history
    if objective_source_targets:
        _fill_objective_source_from_history(objective_source_targets, history)
    kwargs.update(values)
    kwargs["fe"] = int(kwargs.get("fe", 0))
    kwargs["gen"] = int(kwargs.get("gen", 0))
    kwargs["decision_count"] = int(kwargs.get("decision_count", 0))
    state = cls(**kwargs)
    state.data = {**state.data, "resumed": True}
    return state
