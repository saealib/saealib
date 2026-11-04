"""Concrete Stage implementations for the optimization pipeline.

Each stage receives an OptimizationState, performs one well-defined operation,
and returns an updated state via ``state.replace()``.

Standard pipeline fields on OptimizationState
----------------------------------------------
``offspring``
    Current candidate population (Population), set by AskStage.
``scores``
    1-D acquisition score array (np.ndarray), set by AcquisitionStage.
``predictions``
    Batched SurrogatePrediction for ``offspring``, set by SurrogatePredictStage.
``evaluated_offspring``
    Sub-population with true objective values, set by TrueEvaluationStage.

Custom stages may store additional values in ``state.data`` (user-extensible
dict) via ``state.replace(data={**state.data, "key": value})``.
"""

from __future__ import annotations

import inspect
import sys
from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import replace
from math import fsum
from typing import TYPE_CHECKING, Any, Protocol, cast, runtime_checkable

import numpy as np

from saealib.acquisition.base import AcquisitionResult
from saealib.algorithms.base import ProposalRequest
from saealib.callback import (
    AcquisitionEndEvent,
    AcquisitionStartEvent,
    PostEvaluationEvent,
    PostSurrogateFitEvent,
    SurrogateEndEvent,
    SurrogateStartEvent,
)
from saealib.context import EvaluationPlanState, OptimizationState
from saealib.core.contracts import (
    ComponentContract,
    FeedbackBatch,
    PartSpec,
    StateContract,
)
from saealib.core.contracts.execution import ExecutionContract
from saealib.core.contracts.feedback import FeedbackChannel
from saealib.core.contracts.observation import SURROGATE, TRUE
from saealib.core.contracts.proposals import ProposalBatch
from saealib.core.state import (
    ACQUISITION_RESULT,
    ARCHIVES_MAIN,
    ARCHIVES_PARETO,
    EVALUATED_OFFSPRING,
    EVALUATION_HANDLES,
    EVALUATION_NEW_IDS,
    EVALUATION_REQUEST,
    EVALUATION_UPDATE_NEW_IDS,
    EVALUATION_UPDATES,
    EVALUATIONS_COUNT,
    EVALUATIONS_OWNERS,
    EVALUATIONS_PENDING,
    EVALUATIONS_PLAN,
    EVALUATIONS_PLAN_STATE,
    EVALUATIONS_PLAN_UPDATES,
    FEEDBACK_ACCUMULATOR,
    FEEDBACK_RESULT,
    POPULATIONS_MAIN,
    PROPOSALS_CURRENT,
    PROPOSALS_ID_ALLOCATOR,
    PROPOSALS_OFFSPRING,
    RUNTIME_ASYNC_FATAL,
    RUNTIME_CANDIDATE_ID_ALLOCATOR,
    RUNTIME_DECISION_COUNT,
    RUNTIME_GENERATION,
    RUNTIME_REQUEST_ID_ALLOCATOR,
    RUNTIME_RNG,
    SCORES,
    SURROGATES_PREDICTIONS,
    USER_DATA,
    StatePatch,
    StateStore,
    StateView,
)
from saealib.core.state.context import RuntimeContext
from saealib.exceptions import EvaluationProtocolError, ValidationError
from saealib.execution.evaluator import (
    EvaluationRequest,
    EvaluationResult,
    EvaluationStatus,
    Evaluator,
    PendingEvaluation,
)
from saealib.pipeline import _EXECUTION_BOUNDARIES, ComponentRef, Stage
from saealib.policies.evaluation import (
    EvaluateAll,
    EvaluationPlan,
    EvaluationPlanner,
    _aggregate_repeated_updates,
    _continue_fidelity_plan,
)
from saealib.policies.feedback import (
    FeedbackBuilder,
    MixedFeedback,
    TrueOnlyFeedback,
    _feedback_batch_from_result,
)
from saealib.space.space import encode_features

if TYPE_CHECKING:
    from saealib.acquisition.base import AcquisitionFunction
    from saealib.algorithms.base import Algorithm, FeedbackConsumer, Proposer
    from saealib.callback import CallbackManager, Event
    from saealib.context import OptimizationState
    from saealib.execution.evaluator import Evaluator
    from saealib.execution.initializer import Initializer
    from saealib.optimizer import ComponentProvider
    from saealib.problem import Problem
    from saealib.surrogate.manager import SurrogateManager


@runtime_checkable
class _AskCandidatePopulation(Protocol):
    """Population operations needed only by the built-in ask stage."""

    schema: Mapping[str, Any]

    def get_array(self, key: str) -> np.ndarray: ...

    def _assign_ids(self, indices: np.ndarray, ids: np.ndarray) -> None: ...


class _TransactionalExecutionContext(Protocol):
    """Legacy transaction surface consumed by the Stage compatibility adapter."""

    _store: Any

    def get_state(self, key: Any) -> object: ...

    def set_state(self, key: Any, value: object) -> None: ...

    def replace(self, **kwargs: Any) -> _TransactionalExecutionContext: ...


def _state_ctx(event: Event) -> Event:
    """Rewrite a callback event ctx to the run's own state view.

    Stage transactions limit the ctx a component reads to its declared
    reads; handlers observe the run itself and read the optimization state
    ctx (gen, fe, populations, archives) from the event.  Values the Stage
    has already written in its transaction stay visible: the ctx reflects
    the transaction's writes on top of the run state.
    """
    ctx = getattr(event, "ctx", None)
    if isinstance(ctx, OptimizationState):
        return event
    if isinstance(ctx, _StageStateProxy):
        return replace(event, ctx=ctx._materialize())
    state = getattr(ctx, "_state", None)
    if not isinstance(state, OptimizationState):
        state = getattr(getattr(ctx, "_context", None), "_state", None)
    if not isinstance(state, OptimizationState):
        return event
    return replace(event, ctx=state)


def _dispatch_callback(cbmanager: CallbackManager | None, event: Event) -> None:
    """Dispatch through ``cbmanager`` with the run's own state ctx."""
    if cbmanager is not None:
        cbmanager.dispatch(_state_ctx(event))


class _DispatchProxy:
    """Minimal component provider used to pass callbacks to ``Algorithm``."""

    def __init__(self, cbmanager: CallbackManager | None = None) -> None:
        self._cbmanager = cbmanager

    def dispatch(self, event: Event) -> None:
        _dispatch_callback(self._cbmanager, event)

    @property
    def algorithm(self) -> None:
        return None

    @property
    def strategy(self) -> None:
        return None

    @property
    def surrogate_manager(self) -> None:
        return None

    @property
    def evaluator(self) -> None:
        return None

    @property
    def termination(self) -> None:
        return None

    @property
    def cbmanager(self) -> CallbackManager | None:
        return self._cbmanager

    @property
    def seed(self) -> None:
        return None


def _plan_complete(state: OptimizationState) -> bool:
    plan = state.evaluation_plan
    if plan is None:
        return True
    plan_state = state.evaluation_plan_state
    if plan_state is None:
        return False
    terminal = set(plan_state.completed) | set(plan_state.acknowledged)
    return {int(request.request_id) for request in plan.requests} <= terminal


def _plan_incomplete(state: OptimizationState) -> bool:
    return state.evaluation_plan is not None and not _plan_complete(state)


def _apply_component_patch(state: OptimizationState, patch: StatePatch) -> None:
    if not isinstance(patch, StatePatch):
        raise ValidationError("feedback consumer must return a StatePatch")
    if patch.writes or patch.deletes:
        state._store = state._store.apply_patch(patch)


def deliver_feedback(
    consumer: Any,
    feedback: FeedbackBatch,
    state: OptimizationState,
    *,
    reads: Any = None,
    dispatch: Callable[[object], None] | None = None,
    offspring: Any = None,
) -> None:
    """Deliver one final feedback batch through the canonical tell boundary."""
    if reads is None:
        contract = getattr(consumer, "contract", None)
        reads = (
            contract().state if callable(contract) else (POPULATIONS_MAIN, RUNTIME_RNG)
        )
    # ``evaluated_offspring`` is normally the feedback boundary's selected
    # view of the proposal.  Algorithms that maintain row-wise state (such as
    # PSO's particle population) can explicitly retain the full proposal.
    if offspring is not None:
        tell_state = state.replace(
            offspring=offspring,
            evaluated_offspring=offspring,
        )
    elif state.evaluated_offspring is not None and not getattr(
        consumer, "tell_requires_full_proposal", False
    ):
        tell_state = state.replace(offspring=state.evaluated_offspring)
    else:
        tell_state = state
    state_view = tell_state._store.view(reads, context=tell_state, dispatch=dispatch)
    patch = consumer.tell(feedback, state_view)
    _apply_component_patch(state, patch)


def _sync_feedback_metadata(
    state: OptimizationState,
    channel: FeedbackChannel,
) -> tuple[int, bool]:
    """Derive sync delivery metadata from the existing evaluation lifecycle."""
    if channel == TRUE and state.evaluation_updates:
        update = state.evaluation_updates[-1]
        final = update.status in {
            EvaluationStatus.COMPLETED,
            EvaluationStatus.FAILED,
            EvaluationStatus.CANCELLED,
        }
        return int(update.sequence), final
    # A surrogate-only synchronous stage has exactly one delivery and no
    # evaluator-assigned sequence; zero is the first runtime sequence.
    return 0, True


def _component_or_reference(
    value: object,
) -> tuple[Any, ComponentRef | None]:
    """Split a constructor component argument into component or reference.

    A component explicitly passed to a Stage is a Stage-local binding and
    never enters the plan's shared slot table; a reference is resolved from
    the plan's binding snapshot.
    """
    if isinstance(value, ComponentRef):
        return None, value
    return value, None


def _unresolved_reference_roles(stage: object) -> tuple[str, ...]:
    """Return the reference roles the stage has not resolved into components."""
    return tuple(
        ref.role
        for name, ref in getattr(stage, "_refs", ())
        if getattr(stage, name, "sentinel") is None
    )


def _snapshot_cbmanager(bindings: Mapping[str, object]) -> Any:
    """Return the plan snapshot's callback manager for a reference stage."""
    return bindings.get("cbmanager")


def _resolve_reference(
    stage: Stage,
    name: str,
    bindings: Mapping[str, object],
    defaults: dict[tuple[str, str], object],
    *,
    fallback: Callable[[], object] | None = None,
) -> Any:
    """Resolve one declared ``(slot, role)`` reference of *stage*.

    Snapshot values take precedence; an empty slot falls back to a default
    component created once per plan so every reference to the same
    ``(slot, role)`` of one plan shares one instance.
    """
    ref = next((r for attr, r in getattr(stage, "_refs", ()) if attr == name), None)
    if ref is None:
        component = getattr(stage, name, None)
        if component is not None:
            return component
        raise ValidationError(
            f"{type(stage).__name__} holds neither a component nor a "
            f"(slot, role) reference for {name!r}; pass the component or a "
            "ComponentRef."
        )
    component = bindings.get(ref.slot)
    if component is not None:
        return component
    key = (ref.slot, ref.role)
    if key in defaults:
        return defaults[key]
    factory = getattr(type(stage), "_reference_default", None) or fallback
    if factory is None:
        raise ValidationError(
            f"{type(stage).__name__} needs a component for role "
            f"{ref.role!r} from slot {ref.slot!r}: bind the slot on the "
            "optimizer or pass the component explicitly to the stage."
        )
    defaults[key] = component = factory()
    return component


def _component_state_reads(component: object) -> Any:
    """Recompute the state reads a Stage opens on behalf of a component."""
    state_reads = getattr(component, "_state_reads", None)
    if state_reads is None:
        contract = getattr(component, "contract", None)
        state_reads = (
            contract().state if callable(contract) else (POPULATIONS_MAIN, RUNTIME_RNG)
        )
    return state_reads


def _stage_contract(
    *,
    reads: tuple[Any, ...] = (),
    writes: tuple[Any, ...] = (),
    exports: tuple[Any, ...] = (),
    components: tuple[tuple[str, Any], ...] = (),
    required_runtime_capabilities: tuple[str, ...] = (),
    offered_runtime_capabilities: tuple[str, ...] = (),
    reads_enumerable: bool = True,
    writes_enumerable: bool = True,
    unresolved: tuple[str, ...] = (),
) -> ComponentContract:
    """Build a Stage contract while keeping held contracts as named parts."""
    if unresolved:
        raise ValidationError(
            "stage holds unresolved (slot, role) reference(s) for "
            + ", ".join(sorted(set(unresolved)))
            + "; resolve it against a binding snapshot before lowering."
        )
    parts: list[PartSpec] = []
    for name, component in components:
        contract = getattr(component, "contract", None)
        if callable(contract):
            parts.append(PartSpec(name=name, contract=contract()))
    return ComponentContract(
        parts=tuple(parts),
        state=StateContract(
            reads=reads,
            writes=writes,
            exports=exports,
            reads_enumerable=reads_enumerable,
            writes_enumerable=writes_enumerable,
        ),
        execution=ExecutionContract(
            required_runtime_capabilities=required_runtime_capabilities,
            offered_runtime_capabilities=offered_runtime_capabilities,
        ),
    )


_STATE_FIELD_KEYS: dict[str, Any] = {
    "population": POPULATIONS_MAIN,
    "archive": ARCHIVES_MAIN,
    "pareto_archive": ARCHIVES_PARETO,
    "offspring": PROPOSALS_OFFSPRING,
    "evaluated_offspring": EVALUATED_OFFSPRING,
    "scores": SCORES,
    "acquisition_result": ACQUISITION_RESULT,
    "predictions": SURROGATES_PREDICTIONS,
    "evaluation_request": EVALUATION_REQUEST,
    "evaluation_plan": EVALUATIONS_PLAN,
    "evaluation_plan_state": EVALUATIONS_PLAN_STATE,
    "evaluation_updates": EVALUATION_UPDATES,
    "evaluation_plan_updates": EVALUATIONS_PLAN_UPDATES,
    "evaluation_update_new_ids": EVALUATION_UPDATE_NEW_IDS,
    "evaluation_new_ids": EVALUATION_NEW_IDS,
    "evaluation_handles": EVALUATION_HANDLES,
    "evaluation_owners": EVALUATIONS_OWNERS,
    "pending_evaluations": EVALUATIONS_PENDING,
    "feedback_result": FEEDBACK_RESULT,
    "feedback_accumulator": FEEDBACK_ACCUMULATOR,
    "rng": RUNTIME_RNG,
    "candidate_id_allocator": RUNTIME_CANDIDATE_ID_ALLOCATOR,
    "proposal_id_allocator": PROPOSALS_ID_ALLOCATOR,
    "request_id_allocator": RUNTIME_REQUEST_ID_ALLOCATOR,
    "fe": EVALUATIONS_COUNT,
    "gen": RUNTIME_GENERATION,
    "decision_count": RUNTIME_DECISION_COUNT,
    "async_fatal": RUNTIME_ASYNC_FATAL,
    "data": USER_DATA,
    "proposal_id": PROPOSALS_CURRENT,
}


class _StageTransactionStore:
    """Store facade for the Stage transaction proxy."""

    def __init__(self, owner: _StageStateProxy) -> None:
        self._owner = owner

    def view(self, reads: Any, *, context: object | None = None, dispatch=None):
        declared = tuple(reads.reads if hasattr(reads, "reads") else reads)
        values = {key: self._owner._value(key) for key in declared}
        store = StateStore(values)
        return store.view(
            declared,
            context=self._owner._context._for_stage_compatibility(),
            dispatch=dispatch,
        )

    def apply_patch(self, patch: StatePatch) -> _StageTransactionStore:
        if not isinstance(patch, StatePatch):
            raise ValidationError("apply_patch() expects a StatePatch")
        self._owner._apply(patch)
        return self


class _StageStateProxy:
    """OptimizationState-shaped transaction backed by a declared StateView."""

    def __init__(
        self,
        view: StateView,
        context: RuntimeContext,
        *,
        writes: Mapping[Any, object] | None = None,
        deletes: frozenset[Any] = frozenset(),
    ) -> None:
        self._view = view
        self._context = context
        self._writes = dict(writes or {})
        self._deletes = set(deletes)
        self._store = _StageTransactionStore(self)

    def _value(self, key: Any) -> object:
        if key in self._writes:
            return self._writes[key]
        if key in self._deletes:
            raise KeyError(key)
        state = object.__getattribute__(self._context, "_state")
        get_state = getattr(state, "get_state", None)
        if callable(get_state):
            try:
                return get_state(key)
            except KeyError:
                pass
        name = next(
            (name for name, candidate in _STATE_FIELD_KEYS.items() if candidate == key),
            None,
        )
        if name is not None:
            try:
                return object.__getattribute__(state, name)
            except AttributeError:
                pass
        try:
            present = self._view.contains(key)
        except KeyError:
            raise
        if present:
            return self._view.get(key)
        if name is None:
            raise KeyError(key)
        try:
            return getattr(self._context, name)
        except AttributeError as exc:
            raise KeyError(key) from exc

    def _apply(self, patch: StatePatch) -> None:
        self._writes.update(patch.writes)
        self._deletes.update(patch.deletes)
        for key in patch.writes:
            self._deletes.discard(key)

    def _patch(self) -> StatePatch:
        return StatePatch(writes=self._writes, deletes=frozenset(self._deletes))

    def _materialize(self) -> OptimizationState:
        """Return the run state with this transaction's writes applied.

        Callback handlers observe the run itself; the ctx they see reflects
        everything the Stage has written so far in its transaction.
        """
        state = object.__getattribute__(self._context, "_state")
        fields = {
            name: self._writes[key]
            for name, key in _STATE_FIELD_KEYS.items()
            if key in self._writes
        }
        return state.replace(**fields) if fields else state

    def get_state(self, key: Any) -> object:
        return self._value(key)

    def set_state(self, key: Any, value: object) -> None:
        self._apply(StatePatch(writes={key: value}))

    def replace(self, **kwargs: Any) -> _StageStateProxy:
        writes = dict(self._writes)
        deletes = frozenset(self._deletes)
        for name, value in kwargs.items():
            key = _STATE_FIELD_KEYS.get(name)
            if key is None:
                if name == "problem":
                    raise ValidationError("graph-native Stage cannot replace problem")
                raise TypeError(f"unknown state field {name!r}")
            writes[key] = value
            deletes = deletes - {key}
        return _StageStateProxy(
            self._view, self._context, writes=writes, deletes=deletes
        )

    def __getattr__(self, name: str) -> object:
        key = _STATE_FIELD_KEYS.get(name)
        if key is not None:
            try:
                return self._value(key)
            except KeyError:
                raise AttributeError(name)
        # Problem, dimensions, archives, and services are runtime capabilities,
        # not arbitrary state leakage.
        try:
            return getattr(self._context, name)
        except AttributeError as exc:
            raise AttributeError(name) from exc


async def _await_or_value(result: object) -> object:
    if inspect.isawaitable(result):
        return await result
    return result


class _TransactionalStageExecutor(Protocol):
    def execute(
        self, state: _TransactionalExecutionContext
    ) -> _TransactionalExecutionContext | StatePatch | None: ...


class StageStateViewAdapter:
    """Expose a Stage through the graph-native StateView contract."""

    _execution_mode = "graph-native"

    def __init__(
        self,
        stage: Stage,
        *,
        node_path: str | None = None,
        execution_boundary: str = "transaction",
    ) -> None:
        if not isinstance(stage, Stage):
            raise ValidationError("StageStateViewAdapter stage must be a Stage")
        if execution_boundary not in _EXECUTION_BOUNDARIES:
            raise ValidationError(
                f"{type(stage).__name__} execution_boundary must be one of "
                f"{_EXECUTION_BOUNDARIES}, got {execution_boundary!r}"
            )
        self.stage = stage
        self.execution_boundary = execution_boundary
        self.node_path = node_path or stage.name or type(stage).__name__
        self.name = stage.name
        self.label = stage.label
        self.notation = stage.notation
        direct = stage.contract()
        if not isinstance(direct, ComponentContract):
            raise ValidationError("Stage.contract() must return ComponentContract")
        from saealib.core.graph_builder import StageNodeAdapter

        held = StageNodeAdapter(stage).contract().state
        state = StateContract(
            reads=tuple(dict.fromkeys((*direct.state.reads, *held.reads))),
            writes=tuple(dict.fromkeys((*direct.state.writes, *held.writes))),
            exports=tuple(dict.fromkeys((*direct.state.exports, *held.exports))),
            reads_enumerable=direct.state.reads_enumerable and held.reads_enumerable,
            writes_enumerable=direct.state.writes_enumerable and held.writes_enumerable,
        )
        self._contract = replace(direct, state=state)

    def contract(self) -> ComponentContract:
        return self._contract

    @staticmethod
    def _to_patch(proxy: _StageStateProxy, result: object, stage: object) -> StatePatch:
        if isinstance(result, _StageStateProxy):
            return result._patch()
        if isinstance(result, StatePatch):
            patch = proxy._patch()
            return StatePatch(
                writes={**patch.writes, **result.writes},
                deletes=frozenset((*patch.deletes, *result.deletes)),
            )
        if result is proxy or result is None or result is stage:
            return proxy._patch()
        raise ValidationError(
            "graph-native Stage adapter requires the Stage to return its transaction"
        )

    def execute_state(self, state: Any) -> Any:
        """Run the Stage at the optimization-state boundary.

        The Stage sees the run's own state, and callback events carry that
        state ctx instead of the declared-reads transaction ctx.
        """
        return self.stage.execute(state)

    async def execute_async_state(self, state: Any, **kwargs: Any) -> Any:
        """Run the Stage's async seam at the optimization-state boundary."""
        method = getattr(self.stage, "execute_async", None)
        if not callable(method):
            return await _await_or_value(self.stage.execute(state))
        parameters = inspect.signature(method).parameters
        accepts_kwargs = any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in parameters.values()
        )
        if accepts_kwargs:
            return await _await_or_value(method(state, **kwargs))
        selected = {key: value for key, value in kwargs.items() if key in parameters}
        return await _await_or_value(method(state, **selected))

    def execute(self, state: StateView) -> StatePatch:
        context = state.context
        if not isinstance(context, RuntimeContext):
            raise ValidationError("StageStateViewAdapter requires RuntimeContext")
        proxy = _StageStateProxy(state, context)
        result = cast(_TransactionalStageExecutor, self.stage).execute(proxy)
        return self._to_patch(proxy, result, self.stage)

    async def execute_async(self, state: StateView, **kwargs: Any) -> StatePatch:
        context = state.context
        if not isinstance(context, RuntimeContext):
            raise ValidationError("StageStateViewAdapter requires RuntimeContext")
        proxy = _StageStateProxy(state, context)
        method = getattr(self.stage, "execute_async", None)
        if not callable(method):
            raise ValidationError(
                "StageStateViewAdapter stage does not provide execute_async"
            )
        parameters = inspect.signature(method).parameters
        accepts_kwargs = any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in parameters.values()
        )
        selected = (
            kwargs
            if accepts_kwargs
            else {key: value for key, value in kwargs.items() if key in parameters}
        )
        result = method(proxy, **selected)
        if inspect.isawaitable(result):
            result = await result
        return self._to_patch(proxy, result, self.stage)


def stage_component(
    stage: Stage, *, node_path: str | None = None
) -> StageStateViewAdapter:
    """Create a graph-native component from an existing Stage."""
    return StageStateViewAdapter(
        stage,
        node_path=node_path,
        execution_boundary=getattr(stage, "execution_boundary", "transaction"),
    )


wrap_stage = stage_component


class CountGenerationStage(Stage):
    """Increment the generation counter by one."""

    name = "count_generation"
    label = "Count generation"
    notation = r"$gen \leftarrow gen + 1$"

    def contract(self) -> ComponentContract:
        return _stage_contract(
            reads=(RUNTIME_GENERATION, EVALUATIONS_PENDING),
            writes=(RUNTIME_GENERATION,),
        )

    def execute(self, state: OptimizationState) -> OptimizationState:
        # Async steady-state refill calls strategy.step() while an earlier
        # generation is still open.  Those calls are insertions, not new
        # generations; Runner owns the generation boundary and its events.
        if state.pending_evaluations:
            return state
        return state.replace(gen=state.gen + 1)


class AskStage(Stage):
    """Generate offspring candidates via the algorithm's ask() method.

    Writes the offspring population to ``state.offspring``.

    Parameters
    ----------
    algorithm : Algorithm
        The evolutionary algorithm that generates candidates.
    n_offspring : int or None
        Number of offspring to request.  Passed directly to
        ``algorithm.ask()``.  ``None`` lets the algorithm decide.
    cbmanager : CallbackManager or None
        If provided, PostCrossoverEvent / PostMutationEvent / PostAskEvent
        are dispatched through this manager.
    """

    name = "ask"
    label = "Generate offspring"
    notation = r"$\mathcal{Q} \leftarrow \text{ask}(P, n)$"

    def contract(self) -> ComponentContract:
        return _stage_contract(
            # Proposer ctx reads beyond the algorithm's declared contract: the
            # proposal ctx exposes offspring and the proposal ID allocator to
            # components built on the ask boundary.
            reads=(
                PROPOSALS_OFFSPRING,
                PROPOSALS_ID_ALLOCATOR,
                EVALUATIONS_PLAN,
                EVALUATIONS_PLAN_STATE,
                RUNTIME_CANDIDATE_ID_ALLOCATOR,
            ),
            writes=(
                PROPOSALS_OFFSPRING,
                PROPOSALS_CURRENT,
                RUNTIME_CANDIDATE_ID_ALLOCATOR,
                EVALUATED_OFFSPRING,
            ),
            components=(("_algorithm", self._algorithm),),
            unresolved=_unresolved_reference_roles(self),
        )

    def __init__(
        self,
        algorithm: Algorithm | Proposer | ComponentRef,
        n_offspring: int | None = None,
        cbmanager: CallbackManager | None = None,
    ) -> None:
        super().__init__()

        component, reference = _component_or_reference(algorithm)
        if component is None and reference is None:
            raise ValidationError(
                "AskStage requires an algorithm or a (slot, role) reference"
            )
        if reference is not None and cbmanager is not None:
            raise ValidationError(
                "AskStage takes no cbmanager next to a reference; callbacks "
                "come from the plan's binding snapshot"
            )
        self._n_offspring = n_offspring
        self._proxy = _DispatchProxy(None if reference is not None else cbmanager)
        self._algorithm: Any = component
        self._refs = (("_algorithm", reference),) if reference is not None else ()
        if reference is not None:
            self._state_reads = (POPULATIONS_MAIN, RUNTIME_RNG)
        else:
            self._state_reads = _component_state_reads(component)

    def _resolve_stage(
        self,
        bindings: Mapping[str, object],
        defaults: dict[tuple[str, str], object],
    ) -> Stage:
        return AskStage(
            _resolve_reference(self, "_algorithm", bindings, defaults),
            n_offspring=self._n_offspring,
            cbmanager=_snapshot_cbmanager(bindings),
        )

    def to_pseudocode(self, *, expand: bool = False, indent: int = 0) -> str:
        r"""Expand into per-operator lines via ``Algorithm.ask_notation``."""
        prefix = "  " * indent
        ask_notation: list[str] | None = getattr(self._algorithm, "ask_notation", None)
        if expand and ask_notation:
            label = self.label or self.name
            lines = "\n".join(f"{prefix}  \\State {n}" for n in ask_notation)
            return f"{prefix}\\Comment{{{label}}}\n{lines}"
        return f"{prefix}\\State {self.notation}"

    def execute(self, state: OptimizationState) -> OptimizationState:
        if _plan_incomplete(state):
            return state
        state_view = state._store.view(
            self._state_reads,
            context=state,
            dispatch=cast(Callable[[object], None], self._proxy.dispatch),
        )
        proposal = self._algorithm.ask(
            ProposalRequest(n_offspring=self._n_offspring), state_view
        )
        if not isinstance(proposal, ProposalBatch):
            raise ValidationError("proposer ask() must return a ProposalBatch")
        candidates = proposal.candidates
        if (
            isinstance(candidates, _AskCandidatePopulation)
            and "id" in candidates.schema
        ):
            id_arr = candidates.get_array("id")
            unassigned = np.where(id_arr == -1)[0]
            if len(unassigned) > 0:
                new_ids = state.candidate_id_allocator.allocate(len(unassigned))
                candidates._assign_ids(unassigned, new_ids)
            assigned = candidates.get_array("id")
            real = assigned[assigned != -1]
            if len(real) != len(np.unique(real)):
                raise ValidationError(
                    "AskStage received offspring with duplicate candidate ids"
                )
        # A new proposal invalidates the previous evaluated view.  Keeping it
        # would make the canonical tell boundary bind stale rows to a later
        # surrogate-only or true-evaluation delivery.
        state = state.replace(offspring=candidates, evaluated_offspring=None)
        state.set_state(PROPOSALS_CURRENT, proposal.proposal_id)
        return state


class SurrogatePredictStage(Stage):
    """Predict offspring with the surrogate model.

    Reads ``state.offspring``, writes the batched prediction to
    ``state.predictions``.  Also assigns predicted objective values
    ``TellStage`` receives feedback from a separate policy.  Does not compute
    acquisition scores; pair with :class:`AcquisitionStage` for that.

    Parameters
    ----------
    surrogate_manager : SurrogateManager
        Manager that coordinates fit / predict.
    cbmanager : CallbackManager or None
        If provided, SurrogateStartEvent and SurrogateEndEvent are dispatched.
    refit : bool
        Passed directly to ``surrogate_manager.predict()``.
        Set to ``False`` inside inner loops where the surrogate was already
        fitted by an explicit ``SurrogateFitStage``.
    """

    name = "surrogate_predict"
    label = "Surrogate prediction"
    notation = r"$\hat{y} \leftarrow \text{predict}(\mathcal{Q}, \mathcal{A})$"

    def contract(self) -> ComponentContract:
        return _stage_contract(
            reads=(PROPOSALS_OFFSPRING, POPULATIONS_MAIN, ARCHIVES_MAIN),
            writes=(PROPOSALS_OFFSPRING, SURROGATES_PREDICTIONS),
            components=(("_sm", self._sm),),
            unresolved=_unresolved_reference_roles(self),
        )

    def __init__(
        self,
        surrogate_manager: SurrogateManager | ComponentRef,
        cbmanager: CallbackManager | None = None,
        *,
        refit: bool = True,
    ) -> None:
        super().__init__()
        component, reference = _component_or_reference(surrogate_manager)
        if component is None and reference is None:
            raise ValidationError(
                "SurrogatePredictStage requires a surrogate manager or a "
                "(slot, role) reference"
            )
        if reference is not None and cbmanager is not None:
            raise ValidationError(
                "SurrogatePredictStage takes no cbmanager next to a "
                "reference; callbacks come from the plan's binding snapshot"
            )
        self._sm: Any = component
        self._refs = (("_sm", reference),) if reference is not None else ()
        self._cbmanager = None if reference is not None else cbmanager
        self._refit = refit

    def _resolve_stage(
        self,
        bindings: Mapping[str, object],
        defaults: dict[tuple[str, str], object],
    ) -> Stage:
        return SurrogatePredictStage(
            _resolve_reference(self, "_sm", bindings, defaults),
            cbmanager=_snapshot_cbmanager(bindings),
            refit=self._refit,
        )

    def execute(self, state: OptimizationState) -> OptimizationState:
        candidates = state.offspring
        assert candidates is not None

        if self._cbmanager is not None:
            _dispatch_callback(
                self._cbmanager,
                SurrogateStartEvent(ctx=state, offspring=candidates),
            )

        prediction = self._sm.predict(
            encode_features(state.problem.space, candidates.genomes),
            state.archive,
            state,
            refit=self._refit,
        )
        if self._refit and self._cbmanager is not None:
            _dispatch_callback(
                self._cbmanager,
                PostSurrogateFitEvent(
                    ctx=state,
                    surrogate=getattr(self._sm, "surrogate", None),
                ),
            )
        if self._cbmanager is not None:
            _dispatch_callback(
                self._cbmanager, SurrogateEndEvent(ctx=state, offspring=candidates)
            )

        return state.replace(offspring=candidates, predictions=prediction)


class PendingEvaluationContextStage(Stage):
    """Expose asynchronous reservations to downstream components."""

    name = "pending_evaluation_context"
    label = "Pending evaluation context"
    notation = r"$C \leftarrow \text{pending}(C)$"

    def contract(self) -> ComponentContract:
        return _stage_contract()

    def __init__(self, scheduler: Any) -> None:
        super().__init__()
        self._scheduler = scheduler

    def execute(self, state: OptimizationState) -> OptimizationState:
        return state


class AcquisitionStage(Stage):
    """Score offspring via an independent AcquisitionFunction.

    Reads ``state.offspring``/``state.predictions``, writes the resulting
    score array to ``state.scores``.

    Caches ``acquisition.prepare()``'s result per ``(acquisition instance
    identity, generation, decision_count, archive.value_version,
    archive.structure_version)``. A stage instance running the same
    acquisition against an unchanged archive within one generation does not
    recompute the reference, while a mid-generation archive append
    (structure_version bump), value-only mutation (value_version bump), or a
    newly confirmed evaluation plan (``decision_count`` bump -- relevant to
    async steady-state refill, where ``gen`` can stay fixed across several
    confirmed plans) correctly invalidates it.

    Empty candidate input (``len(state.offspring) == 0``) skips straight to
    an empty ``AcquisitionResult`` without touching the cache or calling
    ``prepare()``/``evaluate()``, so it never advances RNG state.

    Parameters
    ----------
    acquisition : AcquisitionFunction
        Acquisition function that scores ``state.offspring`` against
        ``state.predictions``.
    cbmanager : CallbackManager or None
        If provided, AcquisitionStartEvent and AcquisitionEndEvent are
        dispatched.  AcquisitionEndEvent is not dispatched if
        ``acquisition.evaluate()`` raises.
    """

    name = "acquisition"
    label = "Acquisition scoring"
    notation = (
        r"$\mathbf{s} \leftarrow \text{acquire}(\mathcal{Q}, \hat{y}, \mathcal{A})$"
    )

    def contract(self) -> ComponentContract:
        return _stage_contract(
            reads=(
                PROPOSALS_OFFSPRING,
                SURROGATES_PREDICTIONS,
                ARCHIVES_MAIN,
                RUNTIME_GENERATION,
                RUNTIME_DECISION_COUNT,
                RUNTIME_RNG,
            ),
            writes=(SCORES, ACQUISITION_RESULT),
            components=(("_acquisition", self._acquisition),),
            unresolved=_unresolved_reference_roles(self),
        )

    def __init__(
        self,
        acquisition: AcquisitionFunction | ComponentRef,
        cbmanager: CallbackManager | None = None,
    ) -> None:
        super().__init__()
        component, reference = _component_or_reference(acquisition)
        if component is None and reference is None:
            raise ValidationError(
                "AcquisitionStage requires an acquisition function or a "
                "(slot, role) reference"
            )
        if reference is not None and cbmanager is not None:
            raise ValidationError(
                "AcquisitionStage takes no cbmanager next to a reference; "
                "callbacks come from the plan's binding snapshot"
            )
        self._acquisition: Any = component
        self._refs = (("_acquisition", reference),) if reference is not None else ()
        self._cbmanager = None if reference is not None else cbmanager
        # Single-entry cache: id(self._acquisition) is constant for the life
        # of this stage instance, so only (gen, decision_count,
        # value_version, structure_version) actually varies -- a growing
        # dict would retain every past generation's prepared reference for
        # as long as this stage instance is reused.
        self._prepared_cache_key: tuple[int, int, int, int, int] | None = None
        self._prepared_cache_value: object = None

    def _resolve_stage(
        self,
        bindings: Mapping[str, object],
        defaults: dict[tuple[str, str], object],
    ) -> Stage:
        return AcquisitionStage(
            _resolve_reference(self, "_acquisition", bindings, defaults),
            cbmanager=_snapshot_cbmanager(bindings),
        )

    def execute(self, state: OptimizationState) -> OptimizationState:
        candidates = state.offspring
        assert candidates is not None

        if self._cbmanager is not None:
            _dispatch_callback(
                self._cbmanager,
                AcquisitionStartEvent(ctx=state, offspring=candidates),
            )

        if len(candidates) == 0:
            result = AcquisitionResult(scores=np.empty(0, dtype=np.float64))
        else:
            archive = state.archive
            key = (
                id(self._acquisition),
                state.gen,
                state.decision_count,
                archive.value_version,
                archive.structure_version,
            )
            if key != self._prepared_cache_key:
                self._prepared_cache_value = self._acquisition.prepare(archive, state)
                self._prepared_cache_key = key
            prepared = self._prepared_cache_value

            raw = self._acquisition.evaluate(
                encode_features(state.problem.space, candidates.genomes),
                state.predictions,
                archive,
                state,
                prepared=prepared,
            )
            scores = (
                None
                if raw.scores is None
                else np.array(raw.scores, dtype=np.float64, copy=True)
            )
            if scores is not None and scores.shape != (len(candidates),):
                raise ValidationError(
                    f"{type(self._acquisition).__name__}.evaluate() returned "
                    f"scores with shape {scores.shape}, expected "
                    f"({len(candidates)},)."
                )
            result = AcquisitionResult(scores=scores, artifacts=deepcopy(raw.artifacts))

        if self._cbmanager is not None:
            _dispatch_callback(
                self._cbmanager,
                AcquisitionEndEvent(ctx=state, offspring=candidates, result=result),
            )

        # Keep the complete result alive until planning.  In particular,
        # joint acquisitions may put candidate ordering or covariance data in
        # artifacts; reconstructing AcquisitionResult from scores loses it.
        return state.replace(
            scores=result.scores,
            acquisition_result=AcquisitionResult(
                scores=result.scores,
                artifacts=deepcopy(result.artifacts),
            ),
        )


class SurrogateFitStage(Stage):
    """Pre-fit the surrogate on the current archive.

    Use this before a surrogate-only inner loop where the archive does not
    change between iterations.  Pass ``refit=False`` to the downstream
    :class:`SurrogatePredictStage` to skip redundant refitting.

    Parameters
    ----------
    surrogate_manager : SurrogateManager
        Manager to pre-fit.
    """

    name = "surrogate_fit"
    label = "Fit surrogate"
    notation = r"$\hat{f} \leftarrow \text{fit}(\mathcal{A})$"

    def contract(self) -> ComponentContract:
        return _stage_contract(
            reads=(POPULATIONS_MAIN, ARCHIVES_MAIN),
            writes=(),
            components=(("_sm", self._sm),),
            unresolved=_unresolved_reference_roles(self),
        )

    def __init__(
        self,
        surrogate_manager: SurrogateManager | ComponentRef,
        cbmanager: CallbackManager | None = None,
    ) -> None:
        super().__init__()
        component, reference = _component_or_reference(surrogate_manager)
        if component is None and reference is None:
            raise ValidationError(
                "SurrogateFitStage requires a surrogate manager or a "
                "(slot, role) reference"
            )
        if reference is not None and cbmanager is not None:
            raise ValidationError(
                "SurrogateFitStage takes no cbmanager next to a reference; "
                "callbacks come from the plan's binding snapshot"
            )
        self._sm: Any = component
        self._refs = (("_sm", reference),) if reference is not None else ()
        self._cbmanager = None if reference is not None else cbmanager

    def _resolve_stage(
        self,
        bindings: Mapping[str, object],
        defaults: dict[tuple[str, str], object],
    ) -> Stage:
        return SurrogateFitStage(
            _resolve_reference(self, "_sm", bindings, defaults),
            cbmanager=_snapshot_cbmanager(bindings),
        )

    def execute(self, state: OptimizationState) -> OptimizationState:
        self._sm.fit(state.archive, state)
        if self._cbmanager is not None:
            self._cbmanager.dispatch(
                PostSurrogateFitEvent(
                    ctx=state, surrogate=getattr(self._sm, "surrogate", None)
                )
            )
        return state


class TopKSelectionStage(Stage):
    """Select the top-k offspring by surrogate score.

    Reads ``state.scores`` and ``state.offspring``, replaces
    ``state.offspring`` with the top-k candidates sorted highest-score first.

    Parameters
    ----------
    k : int
        Number of candidates to keep.
    """

    name = "top_k_selection"
    label = "Top-k pre-selection"
    notation = r"$\mathcal{Q} \leftarrow \text{top-}k(\mathcal{Q}, \mathbf{s})$"

    def contract(self) -> ComponentContract:
        return _stage_contract(
            reads=(PROPOSALS_OFFSPRING, SCORES), writes=(PROPOSALS_OFFSPRING,)
        )

    def __init__(self, k: int) -> None:
        super().__init__()
        self._k = k

    def execute(self, state: OptimizationState) -> OptimizationState:
        assert state.offspring is not None
        assert state.scores is not None
        idx = np.argsort(-state.scores)
        selected = state.offspring.extract(idx[: self._k])
        return state.replace(offspring=selected)


class SortByScoreStage(Stage):
    """Sort all offspring by surrogate score descending, keeping every candidate.

    Unlike :class:`TopKSelectionStage`, no candidates are discarded.  Used in
    IB-style strategies where :class:`TellStage` receives *all* offspring sorted
    by score while only a top fraction receives true evaluation.

    Reads ``state.scores`` and ``state.offspring``, returns state with both
    arrays reordered by descending score.
    """

    name = "sort_by_score"
    label = "Sort offspring by score"
    notation = r"$\mathcal{Q} \leftarrow \text{sort\_desc}(\mathcal{Q},\,\mathbf{s})$"

    def contract(self) -> ComponentContract:
        return _stage_contract(
            reads=(PROPOSALS_OFFSPRING, SCORES),
            writes=(PROPOSALS_OFFSPRING, SCORES),
        )

    def execute(self, state: OptimizationState) -> OptimizationState:
        assert state.offspring is not None
        assert state.scores is not None
        idx = np.argsort(-state.scores)
        return state.replace(
            offspring=state.offspring.extract(idx),
            scores=state.scores[idx],
        )


class EvaluationPlanStage(Stage):
    """Create an owned request and pending record.

    Increments ``state.decision_count`` by exactly one whenever a genuinely
    new plan is confirmed (the branch that calls the planner and builds
    ``pending_evaluations``), never on a continuation of an already-pending
    multi-request plan. One decision covers the whole confirmed
    ``EvaluationPlan``, however many candidates its requests evaluate.
    ``decision_count`` counts a genuinely new evaluation plan confirmed by
    the runtime regardless of execution mode: this method increments it for
    synchronous execution; :class:`AsyncEvaluationSubmitStage` (which
    :meth:`execute_async` delegates to) increments it the same way for
    async/steady-state execution.
    """

    name = "evaluation_plan"
    label = "Plan evaluation"
    notation = r"$R \leftarrow \text{plan}(Q)$"
    async_protocol = "evaluation"
    async_protocol_role = "driver"

    def contract(self) -> ComponentContract:
        return _stage_contract(
            reads=(
                PROPOSALS_OFFSPRING,
                EVALUATIONS_PENDING,
                EVALUATION_REQUEST,
                EVALUATIONS_PLAN,
                EVALUATIONS_PLAN_STATE,
                EVALUATION_HANDLES,
                EVALUATIONS_OWNERS,
                ACQUISITION_RESULT,
                SCORES,
                SURROGATES_PREDICTIONS,
                RUNTIME_REQUEST_ID_ALLOCATOR,
                RUNTIME_DECISION_COUNT,
            ),
            writes=(
                EVALUATION_REQUEST,
                EVALUATIONS_PLAN,
                EVALUATIONS_PLAN_STATE,
                EVALUATIONS_PLAN_UPDATES,
                EVALUATIONS_PENDING,
                EVALUATION_UPDATES,
                EVALUATION_UPDATE_NEW_IDS,
                EVALUATION_NEW_IDS,
                EVALUATION_HANDLES,
                EVALUATIONS_OWNERS,
                RUNTIME_REQUEST_ID_ALLOCATOR,
                RUNTIME_DECISION_COUNT,
            ),
            components=(("_planner", self._planner),),
            reads_enumerable=not callable(self._n_eval),
            unresolved=_unresolved_reference_roles(self),
        )

    def __init__(
        self,
        planner: EvaluationPlanner | None | ComponentRef = None,
        n_eval=None,
    ) -> None:
        super().__init__()
        if planner is not None and n_eval is not None:
            raise ValidationError("provide planner or n_eval")
        _, reference = _component_or_reference(planner)
        self._planner: Any = (planner or EvaluateAll()) if reference is None else None
        self._n_eval = n_eval
        self._refs = (("_planner", reference),) if reference is not None else ()

    def _resolve_stage(
        self,
        bindings: Mapping[str, object],
        defaults: dict[tuple[str, str], object],
    ) -> Stage:
        return EvaluationPlanStage(
            _resolve_reference(self, "_planner", bindings, defaults),
            self._n_eval,
        )

    #: Created once per plan when the evaluation_planner slot is empty.
    _reference_default = staticmethod(EvaluateAll)

    def execute(self, state: OptimizationState) -> OptimizationState:
        if _plan_incomplete(state):
            plan = state.evaluation_plan
            if plan is None:
                raise EvaluationProtocolError("evaluation plan is missing")
            if state.evaluation_request is not None:
                return state
            plan_state = state.evaluation_plan_state
            terminal = set(plan_state.completed if plan_state else ()) | set(
                plan_state.acknowledged if plan_state else ()
            )
            next_request = next(
                request
                for request in plan.requests
                if int(request.request_id) not in terminal
            )
            return state.replace(evaluation_request=next_request)
        candidates = state.offspring
        if candidates is None:
            raise EvaluationProtocolError("evaluation candidates are missing")
        if self._n_eval is not None:
            n = len(candidates) if self._n_eval is None else self._n_eval
            n = n if isinstance(n, int) else n(state)
            if n < 0 or n > len(candidates):
                raise ValidationError("n_eval must be within the candidate batch")
            from saealib.policies.evaluation import TopKEvaluation

            acquisition = state.acquisition_result or AcquisitionResult(
                scores=state.scores
            )
            plan = TopKEvaluation(n).plan(candidates, acquisition, state)
        else:
            acquisition = state.acquisition_result
            if acquisition is None and state.scores is not None:
                acquisition = AcquisitionResult(scores=state.scores)
            plan = self._planner.plan(candidates, acquisition, state)
        if not isinstance(plan, EvaluationPlan):
            raise EvaluationProtocolError(
                "evaluation planner must return EvaluationPlan"
            )
        plan_ids = {int(item.request_id) for item in plan.requests}
        occupied_ids = (
            set(map(int, state.pending_evaluations))
            | set(map(int, state.evaluation_handles))
            | set(map(int, state.evaluation_owners))
        )
        if plan_ids & occupied_ids:
            raise EvaluationProtocolError(
                "evaluation plan request ID collides with existing work"
            )
        pending_map = dict(state.pending_evaluations)
        for planned in plan.requests:
            pending_map[int(planned.request_id)] = PendingEvaluation(
                planned,
                EvaluationStatus.PENDING,
                np.empty(0, dtype=np.int64),
                checkpointable=True,
            )
        request = plan.requests[0]
        return state.replace(
            evaluation_request=request,
            evaluation_plan=plan,
            evaluation_plan_state=EvaluationPlanState(
                deferred=tuple(int(item.request_id) for item in plan.requests)
            ),
            pending_evaluations=pending_map,
            evaluation_updates=[],
            evaluation_update_new_ids=[],
            evaluation_plan_updates={},
            evaluation_new_ids=np.empty(0, dtype=np.int64),
            decision_count=state.decision_count + 1,
        )

    def execute_async(
        self,
        state: OptimizationState,
        *,
        scheduler: Any,
        feedback_builder: FeedbackBuilder | None = None,
        algorithm: Any = None,
        callback_manager: Any = None,
        prefix=None,
        strategy: Any = None,
        allow_new_proposals: bool = True,
    ) -> OptimizationState:
        """Submit through the public async execution seam.

        Runtime supplies lifecycle services and a prefix callback; planning and
        refill decisions remain owned by this stage.
        """
        if scheduler is None:
            raise ValidationError("Async evaluation requires a scheduler")
        plan_state = state.evaluation_plan_state
        terminal = (
            state.evaluation_plan is not None
            and plan_state is not None
            and all(
                int(item.request_id)
                in set(plan_state.completed) | set(plan_state.acknowledged)
                for item in state.evaluation_plan.requests
            )
        )
        progressed = bool(
            plan_state is not None and (plan_state.completed or plan_state.acknowledged)
        )
        refill = bool(
            getattr(strategy, "supports_async_refill", False)
            and state.evaluation_plan is not None
            and not terminal
            and progressed
            and plan_state is not None
            and not plan_state.deferred
            and len(state.pending_evaluations) < scheduler.max_pending
            and allow_new_proposals
        )
        current = state
        if refill:
            current = current.replace(
                evaluation_plan=None,
                evaluation_plan_state=None,
                evaluation_plan_updates={},
            )
        # A deferred binding publication must not let the plan drain open a
        # new proposal round; remaining requests of the active plan still run.
        if not allow_new_proposals and (current.evaluation_plan is None or terminal):
            return current
        if prefix is not None and (
            current.evaluation_plan is None or terminal or refill
        ):
            current = prefix(current)
        return AsyncEvaluationSubmitStage(
            scheduler,
            self._planner,
            feedback_builder,
            algorithm,
            callback_manager,
        ).execute(current)

    def has_async_work(self, state: OptimizationState) -> bool:
        plan = state.evaluation_plan
        plan_state = state.evaluation_plan_state
        if plan is None or plan_state is None:
            return False
        terminal = set(plan_state.completed) | set(plan_state.acknowledged)
        return any(int(item.request_id) not in terminal for item in plan.requests)


class AsyncEvaluationSubmitStage(Stage):
    """Plan and submit one request to an asynchronous scheduler."""

    name = "async_evaluation_submit"
    label = "Submit asynchronous evaluation"

    def contract(self) -> ComponentContract:
        return _stage_contract(
            reads=(
                PROPOSALS_OFFSPRING,
                ACQUISITION_RESULT,
                SCORES,
                EVALUATIONS_PLAN,
                EVALUATIONS_PLAN_STATE,
                EVALUATIONS_PENDING,
                EVALUATION_HANDLES,
                RUNTIME_REQUEST_ID_ALLOCATOR,
                RUNTIME_DECISION_COUNT,
            ),
            writes=(
                EVALUATION_REQUEST,
                EVALUATIONS_PLAN,
                EVALUATIONS_PLAN_STATE,
                EVALUATIONS_PLAN_UPDATES,
                EVALUATIONS_PENDING,
                RUNTIME_REQUEST_ID_ALLOCATOR,
                RUNTIME_DECISION_COUNT,
            ),
            components=(
                ("_planner", self._planner),
                ("_scheduler", self._scheduler),
            ),
            required_runtime_capabilities=("partial_feedback",),
        )

    def __init__(
        self,
        scheduler: Any,
        planner: EvaluationPlanner,
        feedback_builder: FeedbackBuilder | None = None,
        algorithm: Any = None,
        callback_manager: Any = None,
    ) -> None:
        super().__init__()
        self._scheduler = scheduler
        self._planner = planner
        self._feedback_builder = feedback_builder
        self._algorithm = algorithm
        self._callback_manager = callback_manager

    def execute(self, state: OptimizationState) -> OptimizationState:
        candidates = state.offspring
        if candidates is None:
            raise EvaluationProtocolError("evaluation candidates are missing")
        self._scheduler.feedback_builder = self._feedback_builder
        acquisition = state.acquisition_result
        if acquisition is None and state.scores is not None:
            acquisition = AcquisitionResult(scores=state.scores)
        plan = state.evaluation_plan
        if plan is not None and state.evaluation_plan_state is not None:
            plan_state = state.evaluation_plan_state
            terminal = set(plan_state.completed) | set(plan_state.acknowledged)
            if all(int(item.request_id) in terminal for item in plan.requests):
                state = state.replace(
                    evaluation_plan=None,
                    evaluation_plan_state=None,
                    evaluation_plan_updates={},
                )
                plan = None
        if plan is None:
            plan = self._planner.plan(candidates, acquisition, state)
            state = state.replace(decision_count=state.decision_count + 1)
        if not isinstance(plan, EvaluationPlan):
            raise EvaluationProtocolError(
                "evaluation planner must return EvaluationPlan"
            )
        plan_state = state.evaluation_plan_state
        submitted_ids = set(plan_state.submitted if plan_state else ())
        completed_ids = set(plan_state.completed if plan_state else ())
        acknowledged_ids = set(plan_state.acknowledged if plan_state else ())
        active_pending = {
            request_id: pending
            for request_id, pending in state.pending_evaluations.items()
            if request_id in state.evaluation_handles
            or request_id not in {int(item.request_id) for item in plan.requests}
        }
        if len(active_pending) != len(state.pending_evaluations):
            state = state.replace(pending_evaluations=active_pending)
        remaining_requests = [
            request
            for request in plan.requests
            if int(request.request_id) not in submitted_ids
            and int(request.request_id) not in completed_ids
            and int(request.request_id) not in acknowledged_ids
        ]
        capacity = self._scheduler.max_pending - len(state.pending_evaluations)
        if (
            len(remaining_requests) == 1
            and capacity >= 1
            and len(remaining_requests[0].candidate_ids) > 1
            and not (
                isinstance(plan.continuation, Mapping)
                and plan.continuation.get("kind") == "fidelity_promotion"
            )
        ):
            request = remaining_requests[0]
            chunk_count = (
                len(request.candidate_ids)
                if capacity == 1
                else min(capacity, len(request.candidate_ids))
            )
            chunks = np.array_split(request.candidate_ids, chunk_count)
            requests = []
            total_cost = float(
                request.metadata.get("estimated_cost", len(request.candidate_ids))
            )
            allocated_cost = 0.0
            for index, ids in enumerate(chunks):
                rows = np.asarray(
                    [
                        int(np.flatnonzero(request.candidate_ids == value)[0])
                        for value in ids
                    ],
                    dtype=np.intp,
                )
                request_id = request.request_id
                if index:
                    request_id = np.int64(state.request_id_allocator.allocate(1)[0])
                chunk_cost = (
                    total_cost - fsum((allocated_cost,))
                    if index == len(chunks) - 1
                    else total_cost * len(ids) / len(request.candidate_ids)
                )
                allocated_cost = fsum((allocated_cost, chunk_cost))
                requests.append(
                    EvaluationRequest(
                        request_id,
                        ids,
                        request.payload.take(rows),
                        request.outputs,
                        {
                            **request.metadata,
                            "estimated_cost": float(chunk_cost),
                        },
                    )
                )
            plan_requests = list(plan.requests)
            target_index = next(
                index
                for index, planned in enumerate(plan_requests)
                if int(planned.request_id) == int(request.request_id)
            )
            plan_requests[target_index : target_index + 1] = requests
            plan = EvaluationPlan(
                tuple(plan_requests),
                completion_rule=plan.completion_rule,
                continuation=plan.continuation,
                artifacts=plan.artifacts,
            )
            remaining_requests = [
                planned
                for planned in plan.requests
                if int(planned.request_id) not in submitted_ids
                and int(planned.request_id) not in completed_ids
                and int(planned.request_id) not in acknowledged_ids
            ]
        requests = list(remaining_requests[:capacity])
        deferred = tuple(
            int(request.request_id) for request in remaining_requests[capacity:]
        )
        if not requests:
            return state.replace(
                evaluation_plan=plan,
                evaluation_plan_state=EvaluationPlanState(
                    submitted=tuple(sorted(submitted_ids)),
                    completed=tuple(sorted(completed_ids)),
                    acknowledged=tuple(sorted(acknowledged_ids)),
                    deferred=deferred,
                    continuation=plan.continuation,
                ),
            )
        submitted = self._scheduler.submit(state, requests)
        request = requests[0]
        return submitted.replace(
            evaluation_request=request,
            evaluation_plan=plan,
            evaluation_plan_state=EvaluationPlanState(
                submitted=tuple(
                    sorted(submitted_ids | {int(item.request_id) for item in requests})
                ),
                completed=tuple(sorted(completed_ids)),
                acknowledged=tuple(sorted(acknowledged_ids)),
                deferred=deferred,
                continuation=plan.continuation,
            ),
        )


class EvaluationSubmitStage(Stage):
    """Submit the planned request to an evaluator."""

    name = "evaluation_submit"
    label = "Submit evaluation"
    notation = r"$H \leftarrow \text{submit}(R)$"

    def contract(self) -> ComponentContract:
        return _stage_contract(
            reads=(
                EVALUATION_REQUEST,
                EVALUATIONS_PLAN,
                EVALUATIONS_PLAN_STATE,
                EVALUATIONS_PENDING,
                EVALUATION_HANDLES,
                EVALUATIONS_OWNERS,
            ),
            writes=(EVALUATIONS_PENDING, EVALUATION_HANDLES, EVALUATIONS_PLAN_STATE),
            components=(("_evaluator", self._evaluator),),
            unresolved=_unresolved_reference_roles(self),
        )

    def __init__(self, evaluator: Evaluator | ComponentRef) -> None:
        super().__init__()
        component, reference = _component_or_reference(evaluator)
        if component is None and reference is None:
            raise ValidationError(
                "EvaluationSubmitStage requires an evaluator or a "
                "(slot, role) reference"
            )
        self._evaluator: Any = component
        self._refs = (("_evaluator", reference),) if reference is not None else ()

    def _resolve_stage(
        self,
        bindings: Mapping[str, object],
        defaults: dict[tuple[str, str], object],
    ) -> Stage:
        return EvaluationSubmitStage(
            _resolve_reference(self, "_evaluator", bindings, defaults)
        )

    def execute_async(
        self,
        state: OptimizationState,
        *,
        scheduler: Any = None,
        **kwargs: Any,
    ) -> OptimizationState:
        """Submit planned requests through the scheduler seam.

        With a scheduler bound, submission honors the scheduler's capacity
        and routes through its evaluator; requests beyond capacity stay
        deferred until the plan driver resubmits them as capacity frees up.
        Without a scheduler the synchronous submission runs.
        """
        del kwargs
        if scheduler is None:
            return self.execute(state)
        plan = state.evaluation_plan
        if plan is None:
            return state
        plan_state = state.evaluation_plan_state
        submitted_ids = set(plan_state.submitted if plan_state else ())
        completed_ids = set(plan_state.completed if plan_state else ())
        acknowledged_ids = set(plan_state.acknowledged if plan_state else ())
        remaining = tuple(
            item
            for item in plan.requests
            if int(item.request_id) not in submitted_ids
            and int(item.request_id) not in completed_ids
            and int(item.request_id) not in acknowledged_ids
            and int(item.request_id) not in state.evaluation_handles
        )
        active_pending = {
            request_id: pending
            for request_id, pending in state.pending_evaluations.items()
            if request_id in state.evaluation_handles
            or request_id not in {int(item.request_id) for item in plan.requests}
        }
        if len(active_pending) != len(state.pending_evaluations):
            state = state.replace(pending_evaluations=active_pending)
        capacity = scheduler.max_pending - len(active_pending)
        if capacity <= 0 or not remaining:
            return state
        requests = list(remaining[:capacity])
        submitted = scheduler.submit(state, requests)
        return submitted.replace(
            evaluation_request=requests[0],
            evaluation_plan=plan,
            evaluation_plan_state=EvaluationPlanState(
                submitted=tuple(
                    sorted(submitted_ids | {int(item.request_id) for item in requests})
                ),
                completed=tuple(sorted(completed_ids)),
                acknowledged=tuple(sorted(acknowledged_ids)),
                deferred=tuple(
                    sorted(int(item.request_id) for item in remaining[capacity:])
                ),
                continuation=plan.continuation,
            ),
        )

    def execute(self, state: OptimizationState) -> OptimizationState:
        request = state.evaluation_request
        if request is None:
            raise EvaluationProtocolError("evaluation request is missing")
        plan = state.evaluation_plan
        requests = plan.requests if plan is not None else (request,)
        plan_state = state.evaluation_plan_state
        completed_ids = set(plan_state.completed if plan_state else ())
        acknowledged_ids = set(plan_state.acknowledged if plan_state else ())
        requests = tuple(
            item
            for item in requests
            if int(item.request_id) not in completed_ids
            and int(item.request_id) not in acknowledged_ids
            and int(item.request_id) not in state.evaluation_handles
        )
        if not requests:
            return state
        request_ids = [int(item.request_id) for item in requests]
        if len(request_ids) != len(set(request_ids)):
            raise EvaluationProtocolError(
                "evaluation submit contains duplicate request IDs"
            )
        plan_id_set = set(request_ids)
        pending_map = dict(state.pending_evaluations)
        for planned_id in plan_id_set:
            if planned_id not in state.evaluation_handles:
                pending_map.pop(planned_id, None)
        occupied_ids = (
            (set(map(int, pending_map)) - plan_id_set)
            | set(map(int, state.evaluation_handles))
            | set(map(int, state.evaluation_owners))
        )
        if occupied_ids.intersection(request_ids):
            raise EvaluationProtocolError(
                "evaluation submit request ID collides with existing work"
            )
        handles = dict(state.evaluation_handles)
        for planned in requests:
            handle = self._evaluator.submit(planned, state.problem)
            handles[int(planned.request_id)] = handle
            pending = pending_map.get(int(planned.request_id))
            if pending is None:
                pending = PendingEvaluation(
                    planned, EvaluationStatus.PENDING, np.empty(0, dtype=np.int64)
                )
            pending_map[int(planned.request_id)] = PendingEvaluation(
                planned,
                EvaluationStatus.PENDING,
                pending.applied_candidate_ids,
                checkpointable=True,
            )
        return state.replace(
            evaluation_handles=handles,
            pending_evaluations=pending_map,
            evaluation_plan_state=(
                None
                if plan_state is None
                else EvaluationPlanState(
                    submitted=tuple(
                        sorted(
                            set(plan_state.submitted)
                            | {int(item.request_id) for item in requests}
                        )
                    ),
                    completed=plan_state.completed,
                    acknowledged=plan_state.acknowledged,
                    deferred=tuple(
                        sorted(
                            int(item.request_id)
                            for item in (plan.requests if plan is not None else ())
                            if int(item.request_id)
                            not in (
                                set(plan_state.submitted)
                                | {int(item.request_id) for item in requests}
                                | set(plan_state.completed)
                                | set(plan_state.acknowledged)
                            )
                        )
                    ),
                    continuation=plan.continuation if plan is not None else None,
                )
            ),
        )


class EvaluationCollectStage(Stage):
    """Collect delivered updates for the active request."""

    name = "evaluation_collect"
    label = "Collect evaluation"
    notation = r"$U \leftarrow \text{collect}(H)$"

    def contract(self) -> ComponentContract:
        return _stage_contract(
            reads=(
                EVALUATIONS_PLAN,
                EVALUATIONS_PLAN_STATE,
                EVALUATIONS_PENDING,
                EVALUATION_HANDLES,
                EVALUATIONS_PLAN_UPDATES,
                EVALUATION_REQUEST,
            ),
            writes=(
                EVALUATION_UPDATES,
                EVALUATION_REQUEST,
                EVALUATIONS_PLAN,
                EVALUATIONS_PLAN_STATE,
                EVALUATIONS_PLAN_UPDATES,
                EVALUATIONS_PENDING,
                EVALUATION_UPDATE_NEW_IDS,
                EVALUATION_NEW_IDS,
            ),
            components=(("_evaluator", self._evaluator),),
            unresolved=_unresolved_reference_roles(self),
        )

    def __init__(self, evaluator: Evaluator | ComponentRef) -> None:
        super().__init__()
        component, reference = _component_or_reference(evaluator)
        if component is None and reference is None:
            raise ValidationError(
                "EvaluationCollectStage requires an evaluator or a "
                "(slot, role) reference"
            )
        self._evaluator: Any = component
        self._refs = (("_evaluator", reference),) if reference is not None else ()

    def _resolve_stage(
        self,
        bindings: Mapping[str, object],
        defaults: dict[tuple[str, str], object],
    ) -> Stage:
        return EvaluationCollectStage(
            _resolve_reference(self, "_evaluator", bindings, defaults)
        )

    def execute(self, state: OptimizationState) -> OptimizationState:
        request = state.evaluation_request
        if request is None or int(request.request_id) not in state.evaluation_handles:
            raise EvaluationProtocolError("evaluation handle is missing")
        plan = state.evaluation_plan
        requests = plan.requests if plan is not None else (request,)
        all_updates = {}
        current_updates = []
        current_was_cached = False
        pending_map = dict(state.pending_evaluations)
        for planned in requests:
            planned_id = int(planned.request_id)
            previous_updates = tuple(state.evaluation_plan_updates.get(planned_id, ()))
            planned_pending = pending_map[planned_id]
            if (
                _plan_incomplete(state)
                and planned_pending.last_delivered_sequence >= 0
                and planned_pending.status
                in {
                    EvaluationStatus.COMPLETED,
                    EvaluationStatus.FAILED,
                    EvaluationStatus.CANCELLED,
                }
            ):
                all_updates[planned_id] = list(previous_updates)
                if planned_id == int(request.request_id):
                    current_was_cached = True
                continue
            handle = state.evaluation_handles[int(planned.request_id)]
            delivered = self._evaluator.collect(handle)
            all_updates[planned_id] = list(previous_updates) + list(delivered)
            if delivered:
                # A graph-native stage sees a proxy with no history field key.
                history = getattr(state, "_state", state).history
                if history is not None:
                    for _delivered in delivered:
                        history._observe_evaluation(_delivered, 0)
                expected = planned_pending.last_delivered_sequence + 1
                for update in delivered:
                    if (
                        update.request_id != planned.request_id
                        or update.sequence != expected
                    ):
                        raise EvaluationProtocolError(
                            "evaluation updates must be ascending and contiguous"
                        )
                    expected += 1
                pending_map[planned_id] = replace(
                    planned_pending,
                    status=delivered[-1].status,
                    last_delivered_sequence=delivered[-1].sequence,
                    buffered_updates=planned_pending.buffered_updates
                    + tuple(delivered),
                )
            if planned_id == int(request.request_id):
                current_updates = all_updates[planned_id]
        updates = current_updates
        prior = state.evaluation_plan_state
        completed_set = set(prior.completed if prior is not None else ())
        completed_set.update(
            request_id
            for request_id, values in all_updates.items()
            if values
            and values[-1].status
            in {
                EvaluationStatus.COMPLETED,
                EvaluationStatus.FAILED,
                EvaluationStatus.CANCELLED,
            }
        )
        completed = tuple(sorted(completed_set))
        continuation_plan = None
        if isinstance(plan, EvaluationPlan) and set(
            int(item.request_id) for item in plan.requests
        ) <= set(completed):
            continuation_plan = _continue_fidelity_plan(plan, all_updates, state)
        if continuation_plan is not None:
            promoted_id = continuation_plan.artifacts.get("promoted_request_id")
            if promoted_id is None:
                raise EvaluationProtocolError(
                    "fidelity continuation is missing its request ID"
                )
            promoted = next(
                request
                for request in continuation_plan.requests
                if int(request.request_id) == int(promoted_id)
            )
            plan = continuation_plan
            updates = []
            return state.replace(
                evaluation_request=promoted,
                evaluation_plan=plan,
                evaluation_updates=[],
                evaluation_plan_updates=all_updates,
                evaluation_update_new_ids=[],
                evaluation_new_ids=np.empty(0, dtype=np.int64),
                evaluation_plan_state=EvaluationPlanState(
                    submitted=prior.submitted if prior is not None else (),
                    completed=completed,
                    acknowledged=prior.acknowledged if prior is not None else (),
                    deferred=(int(promoted.request_id),),
                    continuation=plan.continuation,
                    feedback=prior.feedback if prior is not None else None,
                ),
                pending_evaluations=pending_map,
            )
        if (
            isinstance(plan, EvaluationPlan)
            and len(plan.requests) > 1
            and all("replicate" in item.metadata for item in plan.requests)
            and set(int(item.request_id) for item in plan.requests) <= set(completed)
        ):
            final_update = all_updates[int(request.request_id)][-1]
            aggregate = _aggregate_repeated_updates(plan, all_updates, final_update)
            pending = pending_map[int(request.request_id)]
            updates = [
                replace(
                    aggregate,
                    sequence=pending.last_acknowledged_sequence + 1,
                )
            ]
        if isinstance(plan, EvaluationPlan) and len(plan.requests) > 1:
            plan_ids = {int(item.request_id) for item in plan.requests}
            if not plan_ids <= set(completed):
                updates = []
            elif current_was_cached and not updates:
                updates = list(all_updates[int(request.request_id)])
        return state.replace(
            evaluation_updates=updates,
            evaluation_plan_updates=all_updates,
            evaluation_update_new_ids=[],
            evaluation_plan_state=(
                None
                if prior is None
                else EvaluationPlanState(
                    submitted=prior.submitted,
                    completed=completed,
                    deferred=prior.deferred,
                    continuation=prior.continuation,
                    feedback=prior.feedback,
                )
            ),
            pending_evaluations={
                **pending_map,
            },
        )


class EvaluationApplyStage(Stage):
    """Validate and apply delivered result rows atomically."""

    name = "evaluation_apply"
    label = "Apply evaluation"
    notation = r"$Q \leftarrow \text{apply}(U)$"

    def contract(self) -> ComponentContract:
        return _stage_contract(
            reads=(
                PROPOSALS_OFFSPRING,
                EVALUATION_REQUEST,
                EVALUATION_UPDATES,
                EVALUATIONS_PLAN,
                EVALUATIONS_PLAN_STATE,
                EVALUATIONS_PLAN_UPDATES,
                EVALUATIONS_PENDING,
            ),
            writes=(
                PROPOSALS_OFFSPRING,
                EVALUATED_OFFSPRING,
                EVALUATION_NEW_IDS,
                EVALUATION_UPDATE_NEW_IDS,
                EVALUATIONS_PENDING,
            ),
        )

    def execute(self, state: OptimizationState) -> OptimizationState:
        if _plan_incomplete(state):
            return state
        candidates = state.offspring
        request = state.evaluation_request
        pending = (
            state.pending_evaluations.get(int(request.request_id)) if request else None
        )
        if candidates is None or request is None or pending is None:
            raise EvaluationProtocolError("evaluation state is incomplete")
        live_ids = (
            np.asarray(candidates.get_array("id"), dtype=np.int64)
            if "id" in candidates.schema
            else request.candidate_ids
        )
        request_rows = request.metadata.get("row_indices")
        request_set = set(map(int, request.candidate_ids))
        applied = set(map(int, pending.applied_candidate_ids))
        seen_result_ids: set[int] = set()
        new_ids: list[int] = []
        per_update_new_ids: list[np.ndarray] = []
        evaluated_indices: list[int] = []
        operations: list[tuple[np.ndarray, dict[str, np.ndarray]]] = []
        next_status = pending.status
        expected_sequence = pending.last_acknowledged_sequence + 1
        terminal = {
            EvaluationStatus.COMPLETED,
            EvaluationStatus.FAILED,
            EvaluationStatus.CANCELLED,
        }
        allowed = {
            EvaluationStatus.PENDING: {
                EvaluationStatus.RUNNING,
                EvaluationStatus.PARTIAL,
                EvaluationStatus.COMPLETED,
                EvaluationStatus.FAILED,
                EvaluationStatus.CANCELLED,
            },
            EvaluationStatus.RUNNING: {
                EvaluationStatus.PARTIAL,
                EvaluationStatus.COMPLETED,
                EvaluationStatus.FAILED,
                EvaluationStatus.CANCELLED,
            },
            EvaluationStatus.PARTIAL: {
                EvaluationStatus.PARTIAL,
                EvaluationStatus.COMPLETED,
                EvaluationStatus.FAILED,
                EvaluationStatus.CANCELLED,
            },
        }
        processed = False
        for update in state.evaluation_updates:
            if (
                update.request_id != request.request_id
                or update.sequence != expected_sequence
            ):
                raise EvaluationProtocolError(
                    "evaluation update sequence or request mismatch"
                )
            expected_sequence += 1
            if (
                pending.last_acknowledged_sequence >= 0 and pending.status in terminal
            ) or (processed and next_status in terminal):
                raise EvaluationProtocolError("terminal evaluation cannot transition")
            prior_status = (
                EvaluationStatus.PENDING
                if not processed and pending.last_acknowledged_sequence < 0
                else next_status
            )
            if (
                update.status
                in (
                    EvaluationStatus.PENDING,
                    EvaluationStatus.RUNNING,
                )
                and update.result is not None
            ):
                raise EvaluationProtocolError(
                    "pending or running updates cannot carry results"
                )
            if update.status not in allowed.get(prior_status, set()):
                raise EvaluationProtocolError("illegal evaluation status transition")
            if update.status is EvaluationStatus.PARTIAL and (
                update.result is None or len(update.candidate_ids) == 0
            ):
                raise EvaluationProtocolError(
                    "partial updates require a non-empty result"
                )
            if (
                update.status in (EvaluationStatus.FAILED, EvaluationStatus.CANCELLED)
                and update.error is None
            ):
                raise EvaluationProtocolError(
                    "failed or cancelled updates require structured error info"
                )
            update_set = set(map(int, update.candidate_ids))
            if not update_set <= request_set or len(update_set) != len(
                update.candidate_ids
            ):
                raise EvaluationProtocolError(
                    "evaluation update candidate membership is invalid"
                )
            if update_set & applied or update_set & seen_result_ids:
                raise EvaluationProtocolError("evaluation result was duplicated")
            update_new_ids: list[int] = []
            if update.result is not None:
                result_ids = update.result.candidate_ids
                if result_ids is None or not np.array_equal(
                    result_ids, update.candidate_ids
                ):
                    raise EvaluationProtocolError(
                        "result candidate_ids do not match update"
                    )
                if "id" not in candidates.schema and request_rows is not None:
                    row_map = {
                        int(candidate_id): int(row)
                        for candidate_id, row in zip(
                            request.candidate_ids, request_rows, strict=True
                        )
                    }
                    rows_found = [
                        np.array([row_map[int(candidate_id)]])
                        if int(candidate_id) in row_map
                        else np.empty(0, dtype=np.intp)
                        for candidate_id in update.candidate_ids
                    ]
                else:
                    rows_found = [
                        np.flatnonzero(live_ids == candidate_id)
                        for candidate_id in update.candidate_ids
                    ]
                if any(len(row) != 1 for row in rows_found):
                    raise EvaluationProtocolError(
                        "evaluation candidate is not in the population"
                    )
                rows = [int(row[0]) for row in rows_found]
                values = {
                    "f": update.result.f,
                    "g": update.result.g,
                    "cv": update.result.cv,
                }
                operations.append((np.asarray(rows, dtype=np.intp), values))
                update_new_ids.extend(map(int, update.candidate_ids))
                new_ids.extend(update_new_ids)
                seen_result_ids.update(update_new_ids)
                evaluated_indices.extend(rows)
            per_update_new_ids.append(np.asarray(update_new_ids, dtype=np.int64))
            next_status = update.status
            processed = True
        if (
            state.evaluation_updates
            and state.evaluation_updates[-1].status is EvaluationStatus.COMPLETED
        ):
            accounted = applied | set(new_ids)
            if accounted != request_set:
                raise EvaluationProtocolError("completed update does not cover request")
        extra_ids: list[int] = []
        repeated_plan = state.evaluation_plan is not None and all(
            "replicate" in item.metadata for item in state.evaluation_plan.requests
        )
        if state.evaluation_plan is not None and not repeated_plan:
            extra_indices: list[int] = []
            for planned in state.evaluation_plan.requests:
                if int(planned.request_id) == int(request.request_id):
                    continue
                for update in state.evaluation_plan_updates.get(
                    int(planned.request_id), ()
                ):
                    if update.result is None:
                        continue
                    rows = [
                        int(np.flatnonzero(live_ids == candidate_id)[0])
                        for candidate_id in update.candidate_ids
                    ]
                    candidates.update_rows(
                        np.asarray(rows, dtype=np.intp),
                        {
                            "f": update.result.f,
                            "g": update.result.g,
                            "cv": update.result.cv,
                        },
                    )
                    extra_indices.extend(rows)
                    extra_ids.extend(map(int, update.candidate_ids))
            evaluated_indices.extend(extra_indices)
        for rows, values in operations:
            candidates.update_rows(rows, values)
        evaluated = candidates.extract(np.asarray(evaluated_indices, dtype=np.intp))
        updated_pending = PendingEvaluation(
            request,
            next_status,
            np.asarray(sorted(applied | set(new_ids)), dtype=np.int64),
            pending.last_delivered_sequence,
            pending.last_acknowledged_sequence,
        )
        return state.replace(
            offspring=candidates,
            evaluated_offspring=evaluated,
            evaluation_new_ids=np.asarray(new_ids + extra_ids, dtype=np.int64),
            evaluation_update_new_ids=per_update_new_ids,
            pending_evaluations={
                **state.pending_evaluations,
                int(request.request_id): updated_pending,
            },
        )


class EvaluationAcknowledgeStage(Stage):
    """Acknowledge committed updates and account for their evaluations."""

    name = "evaluation_acknowledge"
    label = "Acknowledge evaluation"
    notation = r"$H \leftarrow \text{ack}(U)$"
    async_protocol = "evaluation"
    async_protocol_role = "end"

    def contract(self) -> ComponentContract:
        return _stage_contract(
            reads=(
                PROPOSALS_OFFSPRING,
                EVALUATION_REQUEST,
                EVALUATIONS_PLAN,
                EVALUATIONS_PLAN_STATE,
                EVALUATIONS_PENDING,
                EVALUATION_HANDLES,
                EVALUATION_UPDATES,
                EVALUATION_UPDATE_NEW_IDS,
                EVALUATIONS_PLAN_UPDATES,
                EVALUATIONS_COUNT,
            ),
            writes=(
                EVALUATIONS_PLAN,
                EVALUATIONS_PLAN_STATE,
                EVALUATIONS_PLAN_UPDATES,
                EVALUATIONS_PENDING,
                EVALUATION_HANDLES,
                EVALUATIONS_COUNT,
            ),
            components=(("_evaluator", self._evaluator),),
            unresolved=_unresolved_reference_roles(self),
        )

    def __init__(
        self,
        evaluator: Evaluator | ComponentRef,
        cbmanager: CallbackManager | None = None,
    ) -> None:
        super().__init__()
        component, reference = _component_or_reference(evaluator)
        if component is None and reference is None:
            raise ValidationError(
                "EvaluationAcknowledgeStage requires an evaluator or a "
                "(slot, role) reference"
            )
        if reference is not None and cbmanager is not None:
            raise ValidationError(
                "EvaluationAcknowledgeStage takes no cbmanager next to a "
                "reference; callbacks come from the plan's binding snapshot"
            )
        self._evaluator: Any = component
        self._refs = (("_evaluator", reference),) if reference is not None else ()
        self._cbmanager = None if reference is not None else cbmanager

    def _resolve_stage(
        self,
        bindings: Mapping[str, object],
        defaults: dict[tuple[str, str], object],
    ) -> Stage:
        return EvaluationAcknowledgeStage(
            _resolve_reference(self, "_evaluator", bindings, defaults),
            cbmanager=_snapshot_cbmanager(bindings),
        )

    def execute(self, state: OptimizationState) -> OptimizationState:
        if _plan_incomplete(state):
            return state
        request = state.evaluation_request
        if request is None:
            raise EvaluationProtocolError("evaluation request is missing")
        handle = state.evaluation_handles[int(request.request_id)]
        pending = state.pending_evaluations[int(request.request_id)]
        plan = state.evaluation_plan
        current = state
        pending_map = dict(state.pending_evaluations)
        handles = dict(state.evaluation_handles)
        pending_ids = set(map(int, pending.applied_candidate_ids))
        new_by_update = state.evaluation_update_new_ids
        if len(new_by_update) != len(state.evaluation_updates):
            raise EvaluationProtocolError("evaluation update bookkeeping is missing")
        raw_updates = state.evaluation_updates
        if isinstance(plan, EvaluationPlan) and len(plan.requests) > 1:
            raw_updates = tuple(
                state.evaluation_plan_updates.get(
                    int(request.request_id), state.evaluation_updates
                )
            )
        for update in raw_updates:
            self._evaluator.acknowledge(handle, update.sequence)
        for update, update_new_ids in zip(
            state.evaluation_updates, new_by_update, strict=True
        ):
            pending_ids.update(map(int, update_new_ids))
            updated_pending = PendingEvaluation(
                request,
                update.status,
                np.asarray(sorted(pending_ids), dtype=np.int64),
                update.sequence,
                update.sequence,
            )
            terminal = update.status in (
                EvaluationStatus.COMPLETED,
                EvaluationStatus.FAILED,
                EvaluationStatus.CANCELLED,
            )
            if terminal:
                pending_map.pop(int(request.request_id), None)
                handles.pop(int(request.request_id), None)
            else:
                pending_map[int(request.request_id)] = updated_pending
            current = current.replace(
                fe=current.fe + len(update_new_ids),
                pending_evaluations=pending_map.copy(),
                evaluation_handles=handles.copy(),
            )
            if len(update_new_ids) and self._cbmanager is not None:
                offspring = current.offspring
                if offspring is None:
                    raise EvaluationProtocolError("offspring is missing for callback")
                if "id" in offspring.schema:
                    rows = [
                        int(
                            np.flatnonzero(offspring.get_array("id") == candidate_id)[0]
                        )
                        for candidate_id in update_new_ids
                    ]
                else:
                    request_rows = request.metadata.get("row_indices")
                    if request_rows is None:
                        rows = [int(candidate_id) for candidate_id in update_new_ids]
                    else:
                        row_map = {
                            int(candidate_id): int(row)
                            for candidate_id, row in zip(
                                request.candidate_ids, request_rows, strict=True
                            )
                        }
                        rows = [
                            row_map[int(candidate_id)]
                            for candidate_id in update_new_ids
                        ]
                _dispatch_callback(
                    self._cbmanager,
                    PostEvaluationEvent(
                        ctx=current,
                        offspring=offspring.extract(rows),
                        request_id=request.request_id,
                        candidate_ids=update_new_ids,
                        status=update.status,
                    ),
                )
        if isinstance(plan, EvaluationPlan) and len(plan.requests) > 1:
            all_updates = state.evaluation_plan_updates
            for planned in plan.requests:
                planned_id = int(planned.request_id)
                if planned_id == int(request.request_id):
                    continue
                extra_handle = handles.get(planned_id)
                if extra_handle is None:
                    pending_map.pop(planned_id, None)
                    continue
                extra_pending = pending_map.get(planned_id)
                for update in all_updates.get(planned_id, ()):
                    self._evaluator.acknowledge(extra_handle, update.sequence)
                    if update.result is not None:
                        current = current.replace(
                            fe=current.fe + len(update.candidate_ids)
                        )
                if extra_pending is not None:
                    pending_map.pop(planned_id, None)
                    handles.pop(planned_id, None)
            current = current.replace(
                pending_evaluations=pending_map.copy(),
                evaluation_handles=handles.copy(),
            )
        return current.replace(
            evaluation_plan=None,
            evaluation_plan_state=None,
            evaluation_plan_updates={},
        )


class TrueEvaluationStage(Stage):
    """Evaluate offspring with the true objective function.

    Reads ``state.offspring``, evaluates all candidates, updates their
    ``f / g / cv`` attributes in-place, increments ``state.fe``, and writes
    the evaluated sub-population to ``state.evaluated_offspring``.

    Parameters
    ----------
    evaluator : Evaluator
        Evaluator that calls the true objective function.
    cbmanager : CallbackManager or None
        If provided, PostEvaluationEvent is dispatched after evaluation.
    n_eval : int, callable, or None
        Number of candidates to evaluate from the head of the offspring
        population.  If callable, it receives the current
        :class:`~saealib.context.OptimizationState` and must return an int
        (e.g. ``lambda s: max(1, int(ratio * len(s.offspring)))``).
        ``None`` means evaluate all.
    """

    name = "true_evaluation"
    label = "True objective evaluation"
    notation = r"$\mathcal{Q}_{eval} \leftarrow \text{eval}(\mathcal{Q})$"

    def contract(self) -> ComponentContract:
        return _stage_contract(
            reads=(PROPOSALS_OFFSPRING,),
            writes=(PROPOSALS_OFFSPRING, EVALUATED_OFFSPRING, EVALUATIONS_COUNT),
            components=(("_evaluator", self._evaluator),),
            unresolved=_unresolved_reference_roles(self),
        )

    def __init__(
        self,
        evaluator: Evaluator | ComponentRef,
        cbmanager: CallbackManager | None = None,
        n_eval: int | Callable[[OptimizationState], int] | None = None,
    ) -> None:
        super().__init__()
        component, reference = _component_or_reference(evaluator)
        if component is None and reference is None:
            raise ValidationError(
                "TrueEvaluationStage requires an evaluator or a (slot, role) reference"
            )
        if reference is not None and cbmanager is not None:
            raise ValidationError(
                "TrueEvaluationStage takes no cbmanager next to a reference; "
                "callbacks come from the plan's binding snapshot"
            )
        self._evaluator: Any = component
        self._refs = (("_evaluator", reference),) if reference is not None else ()
        self._cbmanager = None if reference is not None else cbmanager
        self._n_eval = n_eval

    def _resolve_stage(
        self,
        bindings: Mapping[str, object],
        defaults: dict[tuple[str, str], object],
    ) -> Stage:
        return TrueEvaluationStage(
            _resolve_reference(self, "_evaluator", bindings, defaults),
            cbmanager=_snapshot_cbmanager(bindings),
            n_eval=self._n_eval,
        )

    def execute(self, state: OptimizationState) -> OptimizationState:
        candidates = state.offspring
        assert candidates is not None
        if self._n_eval is None:
            n = len(candidates)
        elif isinstance(self._n_eval, int):
            n = self._n_eval
        else:
            n = self._n_eval(state)
        n = min(n, len(candidates))

        result = self._evaluator.evaluate_batch(
            candidates.genomes.take(np.arange(n)), state.problem
        )
        candidates.update_rows(
            np.arange(n), {"f": result.f, "g": result.g, "cv": result.cv}
        )

        evaluated = candidates.extract(list(range(n)))
        counted = state.replace(fe=state.fe + n)

        if self._cbmanager is not None:
            _dispatch_callback(
                self._cbmanager, PostEvaluationEvent(ctx=state, offspring=evaluated)
            )

        return counted.replace(
            offspring=candidates,
            evaluated_offspring=evaluated,
        )


class ArchiveUpdateStage(Stage):
    """Append evaluated offspring to archive and Pareto archive.

    Reads ``state.evaluated_offspring`` and appends each individual to
    ``state.archive`` and ``state.pareto_archive`` (both are controlled
    mutable exceptions — append-only in-place updates).
    """

    name = "archive_update"
    label = "Archive update"
    notation = r"$\mathcal{A} \leftarrow \mathcal{A} \cup \mathcal{Q}_{eval}$"

    def contract(self) -> ComponentContract:
        return _stage_contract(
            reads=(
                EVALUATED_OFFSPRING,
                ARCHIVES_MAIN,
                ARCHIVES_PARETO,
                EVALUATIONS_PLAN,
                EVALUATIONS_PLAN_STATE,
            ),
            writes=(ARCHIVES_MAIN, ARCHIVES_PARETO, EVALUATED_OFFSPRING),
        )

    def execute(self, state: OptimizationState) -> OptimizationState:
        if _plan_incomplete(state):
            return state
        evaluated = state.evaluated_offspring
        assert evaluated is not None
        has_id = "id" in evaluated.schema
        for i in range(len(evaluated)):
            ind = evaluated[i]
            entry = {
                "genome": ind.genome,
                "f": ind.f,
                "g": ind.g,
                "cv": float(ind.cv),
            }
            if "x" in evaluated.schema:
                entry["x"] = ind.x
            if has_id:
                entry["id"] = int(ind.id)
            state.archive.add(entry)
            if not has_id or int(ind.id) not in set(
                map(int, state.pareto_archive.get_array("id"))
            ):
                state.pareto_archive.add(entry)
        if has_id and len(evaluated):
            _, first = np.unique(evaluated.get_array("id"), return_index=True)
            evaluated_for_feedback = evaluated.extract(np.sort(first))
            return state.replace(evaluated_offspring=evaluated_for_feedback)
        return state


class FeedbackStage(Stage):
    """Copy evaluated objective channels into the offspring batch."""

    name = "feedback"
    label = "Apply feedback"
    notation = r"$\mathcal{Q} \leftarrow \mathrm{feedback}(\mathcal{Q})$"

    def contract(self) -> ComponentContract:
        return _stage_contract(
            reads=(
                PROPOSALS_OFFSPRING,
                EVALUATED_OFFSPRING,
                EVALUATION_NEW_IDS,
                SURROGATES_PREDICTIONS,
                EVALUATIONS_PLAN,
                EVALUATIONS_PLAN_STATE,
            ),
            writes=(PROPOSALS_OFFSPRING, FEEDBACK_RESULT),
            components=(("_builder", self._builder),),
            unresolved=_unresolved_reference_roles(self),
        )

    def __init__(
        self,
        builder: FeedbackBuilder | None | ComponentRef = None,
    ) -> None:
        super().__init__()
        _, reference = _component_or_reference(builder)
        self._builder: Any = (
            (builder or TrueOnlyFeedback()) if reference is None else None
        )
        self._refs = (("_builder", reference),) if reference is not None else ()

    #: Created once per plan when the feedback_builder slot is empty.
    _reference_default = staticmethod(TrueOnlyFeedback)

    def _resolve_stage(
        self,
        bindings: Mapping[str, object],
        defaults: dict[tuple[str, str], object],
    ) -> Stage:
        return FeedbackStage(_resolve_reference(self, "_builder", bindings, defaults))

    def execute(self, state: OptimizationState) -> OptimizationState:
        if _plan_incomplete(state):
            return state
        offspring = state.offspring
        evaluated = state.evaluated_offspring
        if offspring is None:
            return state
        evaluation = None
        if evaluated is not None and len(evaluated):
            if "id" in evaluated.schema:
                ids = np.array(evaluated.get_array("id"), dtype=np.int64, copy=True)
            else:
                ids = np.array(state.evaluation_new_ids, dtype=np.int64, copy=True)
            evaluation = EvaluationResult(
                np.array(evaluated.f, dtype=np.float64, copy=True),
                np.array(evaluated.g, dtype=np.float64, copy=True),
                np.array(evaluated.cv, dtype=np.float64, copy=True),
                candidate_ids=ids,
            )
        result = self._builder.build(
            offspring, state.predictions, evaluation, state.evaluation_new_ids, state
        )
        if len(result.candidate_ids) == 0:
            return state.replace(feedback_result=result)
        if "id" not in offspring.schema:
            rows = [int(candidate_id) for candidate_id in result.candidate_ids]
        else:
            ids = offspring.get_array("id")
            rows = []
            for candidate_id in result.candidate_ids:
                matches = np.flatnonzero(ids == candidate_id)
                if len(matches) != 1:
                    raise ValidationError(
                        f"feedback candidate ID {candidate_id} is not in offspring"
                    )
                rows.append(int(matches[0]))
        values = {"f": result.f}
        if result.g is not None and "g" in offspring.schema:
            values["g"] = result.g
        if result.cv is not None and "cv" in offspring.schema:
            values["cv"] = result.cv
        offspring.update_rows(np.asarray(rows, dtype=np.intp), values)
        return state.replace(offspring=offspring, feedback_result=result)


class TellStage(Stage):
    """Update the population via the algorithm's tell() method.

    Reads ``state.offspring`` (the full candidate population, including
    both surrogate-scored and true-evaluated individuals, as the algorithm
    expects) and calls ``algorithm.tell()``.

    Parameters
    ----------
    algorithm : Algorithm
        The evolutionary algorithm that updates the population.
    """

    name = "tell"
    label = "Update population"
    notation = r"$P \leftarrow \text{tell}(P, \mathcal{Q})$"

    def contract(self) -> ComponentContract:
        return _stage_contract(
            reads=(
                PROPOSALS_OFFSPRING,
                PROPOSALS_CURRENT,
                FEEDBACK_RESULT,
                EVALUATED_OFFSPRING,
                EVALUATIONS_PLAN,
                EVALUATIONS_PLAN_STATE,
                EVALUATION_UPDATES,
            ),
            writes=(),
            components=(("_algorithm", self._algorithm),),
            unresolved=_unresolved_reference_roles(self),
        )

    def __init__(
        self,
        algorithm: Algorithm | Proposer | FeedbackConsumer | ComponentRef,
        *,
        channel: FeedbackChannel = TRUE,
    ) -> None:
        super().__init__()

        component, reference = _component_or_reference(algorithm)
        if component is None and reference is None:
            raise ValidationError(
                "TellStage requires an algorithm or a (slot, role) reference"
            )
        self._proxy = _DispatchProxy()
        self._channel = channel
        self._algorithm: Any = component
        self._refs = (("_algorithm", reference),) if reference is not None else ()
        if reference is not None:
            self._state_reads = (POPULATIONS_MAIN, RUNTIME_RNG)
        else:
            self._state_reads = _component_state_reads(component)

    def _resolve_stage(
        self,
        bindings: Mapping[str, object],
        defaults: dict[tuple[str, str], object],
    ) -> Stage:
        return TellStage(
            _resolve_reference(self, "_algorithm", bindings, defaults),
            channel=self._channel,
        )

    def to_pseudocode(self, *, expand: bool = False, indent: int = 0) -> str:
        r"""Expand into per-step lines via ``Algorithm.tell_notation``."""
        prefix = "  " * indent
        tell_notation: list[str] | None = getattr(
            self._algorithm, "tell_notation", None
        )
        if expand and tell_notation:
            label = self.label or self.name
            lines = "\n".join(f"{prefix}  \\State {n}" for n in tell_notation)
            return f"{prefix}\\Comment{{{label}}}\n{lines}"
        return f"{prefix}\\State {self.notation}"

    def execute(self, state: OptimizationState) -> OptimizationState:
        if _plan_incomplete(state):
            return state
        result = state.feedback_result
        if state.offspring is None or result is None or len(result.candidate_ids) == 0:
            return state
        try:
            proposal_id = int(state.get_state(PROPOSALS_CURRENT))
        except KeyError:
            raise ValidationError("feedback proposal ID is missing")
        sequence, final = _sync_feedback_metadata(state, self._channel)
        feedback = _feedback_batch_from_result(
            result,
            proposal_id=proposal_id,
            channel=self._channel,
            final=final,
            sequence=sequence,
        )
        deliver_feedback(
            self._algorithm,
            feedback,
            state,
            reads=self._state_reads,
            dispatch=cast(Callable[[object], None], self._proxy.dispatch),
        )
        return state


class SurrogateOnlyLoopStage(Stage):
    """Run *gen_ctrl* surrogate-only generations before real evaluation.

    Fits the surrogate model once on the current archive, then repeats
    ``gen_ctrl`` times: CountGeneration → Ask → SurrogatePredict(refit=False)
    → Acquisition → Tell.  If *gen_ctrl* is 0 this stage is a no-op.

    Used by :class:`~saealib.strategies.gb.GenerationBasedStrategy` to
    execute inner surrogate-driven generations before a single true-evaluation
    generation.

    Parameters
    ----------
    algorithm : Algorithm or ComponentRef
        Evolutionary algorithm for ask/tell, or the ``(slot, role)``
        reference resolved from the plan's binding snapshot.
    surrogate_manager : SurrogateManager or ComponentRef
        Manager used for fitting and prediction.
    gen_ctrl : int
        Number of surrogate-only generations.
    cbmanager : CallbackManager or None
        Forwarded to inner stages for event dispatching.  Must stay ``None``
        for a reference-holding loop; callbacks then come from the plan's
        binding snapshot.
    acquisition : AcquisitionFunction or ComponentRef
        Acquisition function used to score offspring inside the inner loop.
        Keyword-only so existing positional
        ``SurrogateOnlyLoopStage(algorithm, surrogate_manager, gen_ctrl,
        cbmanager)`` calls stay valid.
    feedback_builder : FeedbackBuilder, ComponentRef, or None
        Inner-loop feedback builder; ``None`` keeps the built-in
        surrogate-channel default.
    """

    name = "surrogate_only_loop"
    label = "Surrogate-only generations"
    notation = (
        r"$\text{for}\;i=1\dots gen\_ctrl$: "
        r"$P \leftarrow \mathrm{tell}(P,\,"
        r"\mathrm{acquire}(\mathrm{predict}(\mathrm{ask}(P))))$"
    )

    def contract(self) -> ComponentContract:
        # The nested stages are separate graph units.  This declaration covers
        # only the loop's direct fit/read access and avoids double counting.
        return _stage_contract(
            reads=(ARCHIVES_MAIN,),
            components=(("_sm", self._sm),),
            unresolved=_unresolved_reference_roles(self),
        )

    def __init__(
        self,
        algorithm: Algorithm | ComponentRef,
        surrogate_manager: SurrogateManager | ComponentRef,
        gen_ctrl: int,
        cbmanager: CallbackManager | None = None,
        *,
        acquisition: AcquisitionFunction | ComponentRef,
        feedback_builder: FeedbackBuilder | None | ComponentRef = None,
    ) -> None:
        super().__init__()
        self._gen_ctrl = gen_ctrl
        # Reference-tracking attributes stay unset until resolution; only
        # the surrogate manager enters the contract as a named part.
        self._sm: Any = None
        self._algorithm: Any = None
        self._acquisition: Any = None
        self._builder: Any = None
        references: list[tuple[str, ComponentRef]] = []
        if isinstance(surrogate_manager, ComponentRef):
            references.append(("_sm", surrogate_manager))
        elif surrogate_manager is not None:
            self._sm = surrogate_manager
        for name, value in (
            ("_algorithm", algorithm),
            ("_acquisition", acquisition),
            ("_builder", feedback_builder),
        ):
            if isinstance(value, ComponentRef):
                references.append((name, value))
            elif value is not None:
                setattr(self, name, value)
        self._refs = tuple(references)
        referenced_names = {name for name, _ in references}
        for name in ("_sm", "_algorithm", "_acquisition"):
            if getattr(self, name) is None and name not in referenced_names:
                raise ValidationError(
                    f"SurrogateOnlyLoopStage requires {name[1:]} as a component or a "
                    "(slot, role) reference"
                )
        if cbmanager is not None and self._refs:
            raise ValidationError(
                "SurrogateOnlyLoopStage takes no cbmanager next to a "
                "reference; callbacks come from the plan's binding snapshot"
            )
        self._cbmanager = None if self._refs else cbmanager
        if gen_ctrl > 0 and not self._refs:
            self.stages = self._build_inner_stages(
                algorithm, surrogate_manager, acquisition, feedback_builder, cbmanager
            )
        else:
            self.stages = []

    @staticmethod
    def _build_inner_stages(
        algorithm: Any,
        surrogate_manager: Any,
        acquisition: Any,
        feedback_builder: FeedbackBuilder | None | ComponentRef,
        cbmanager: CallbackManager | None,
    ) -> list[Stage]:
        return [
            CountGenerationStage(),
            AskStage(algorithm, cbmanager=cbmanager),
            SurrogatePredictStage(surrogate_manager, cbmanager=cbmanager, refit=False),
            AcquisitionStage(acquisition, cbmanager=cbmanager),
            FeedbackStage(feedback_builder or MixedFeedback()),
            TellStage(algorithm, channel=SURROGATE),
        ]

    def _resolve_stage(
        self,
        bindings: Mapping[str, object],
        defaults: dict[tuple[str, str], object],
    ) -> Stage:
        referenced_names = {name for name, _ in self._refs}
        builder = (
            _resolve_reference(
                self, "_builder", bindings, defaults, fallback=MixedFeedback
            )
            if "_builder" in referenced_names
            else self._builder
        )
        return SurrogateOnlyLoopStage(
            _resolve_reference(self, "_algorithm", bindings, defaults),
            _resolve_reference(self, "_sm", bindings, defaults),
            self._gen_ctrl,
            cbmanager=(
                _snapshot_cbmanager(bindings) if referenced_names else self._cbmanager
            ),
            acquisition=_resolve_reference(self, "_acquisition", bindings, defaults),
            feedback_builder=builder,
        )

    def to_pseudocode(self, *, expand: bool = False, indent: int = 0) -> str:
        r"""Render as a ``\For`` loop block when *expand* is True."""
        prefix = "  " * indent
        if expand and self.stages:
            inner_lines = "\n".join(
                s.to_pseudocode(expand=True, indent=indent + 1) for s in self.stages
            )
            return (
                f"{prefix}\\For{{$i = 1, \\ldots, gen\\_ctrl$}}\n"
                f"{inner_lines}\n"
                f"{prefix}\\EndFor"
            )
        return f"{prefix}\\State {self.notation}"

    def execute(self, state: OptimizationState) -> OptimizationState:
        if self._gen_ctrl > 0:
            self._sm.fit(state.archive, state)
            if self._cbmanager is not None:
                _dispatch_callback(
                    self._cbmanager,
                    PostSurrogateFitEvent(
                        ctx=state, surrogate=getattr(self._sm, "surrogate", None)
                    ),
                )
            for _ in range(self._gen_ctrl):
                for stage in self.stages:
                    state = stage.execute(state)
        return state


class InitializationStage(Stage):
    """Wrap an :class:`~saealib.execution.initializer.Initializer` as a Stage.

    Delegates to ``initializer.initialize(provider, problem)`` and returns the
    resulting :class:`~saealib.context.OptimizationState`.  The *state*
    argument passed to :meth:`execute` is **ignored** — initialization always
    produces a fresh state from scratch.

    This stage is intended for use at the head of a user-defined Pipeline when
    the initializer itself should participate in the pipeline abstraction (e.g.
    to build custom init-then-optimize flows or to inspect / swap the
    initialization step via ``Pipeline["initialization"]``).

    Parameters
    ----------
    initializer : Initializer
        The concrete initializer (e.g.
        :class:`~saealib.execution.initializer.LHSInitializer`).
    provider : ComponentProvider
        Component provider forwarded to ``Initializer.initialize()``.
    problem : Problem
        The optimization problem.
    """

    name = "initialization"
    label = "Initialize population"
    notation = r"$\mathcal{A}_0,\,P_0 \leftarrow \mathrm{init}(n_{\mathrm{init}})$"

    def contract(self) -> ComponentContract:
        return _stage_contract(
            writes=(),
            components=(("_initializer", self._initializer),),
        )

    def __init__(
        self,
        initializer: Initializer,
        provider: ComponentProvider,
        problem: Problem,
    ) -> None:
        super().__init__()
        self._initializer = initializer
        self._provider = provider
        self._problem = problem

    def execute(self, state: OptimizationState) -> OptimizationState:
        return self._initializer.initialize(self._provider, self._problem)


def discover_builtin_stages() -> tuple[type[Stage], ...]:
    """Return the operational Stage classes shipped with SAEALib.

    This is the single production inventory used by tooling that needs to
    inspect all built-in stages.  It intentionally excludes ``Stage`` and
    ``Pipeline`` themselves, as well as user-defined subclasses.
    """
    return tuple(
        cls
        for cls in vars(sys.modules[__name__]).values()
        if isinstance(cls, type)
        and cls is not Stage
        and issubclass(cls, Stage)
        and cls.__module__ == __name__
        and "contract" in cls.__dict__
    )


class _ContractProbe:
    """Side-effect-free held component used for contract introspection."""

    def contract(self) -> ComponentContract:
        return ComponentContract()

    def ask(self, request: object, state: object) -> object:
        del request, state
        return None

    def tell(self, feedback: object, state: object) -> object:
        del feedback, state
        return None


def _builtin_stage_instances_for_contracts() -> tuple[Stage, ...]:
    """Build minimal instances for tooling that calls each ``contract()``."""
    instances: list[Stage] = []
    for stage_type in discover_builtin_stages():
        positional: list[object] = []
        keyword: dict[str, object] = {}
        for parameter in inspect.signature(stage_type).parameters.values():
            if parameter.default is not inspect.Parameter.empty:
                continue
            value: object = _ContractProbe()
            if parameter.name == "k":
                value = 1
            elif parameter.name == "gen_ctrl":
                value = 0
            if parameter.kind is inspect.Parameter.POSITIONAL_ONLY:
                positional.append(value)
            else:
                keyword[parameter.name] = value
        constructor = cast(Callable[..., Stage], stage_type)
        instances.append(constructor(*positional, **keyword))
    return tuple(instances)
