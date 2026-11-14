"""Runtime implementations and the optimizer environment compatibility bridge."""

from __future__ import annotations

import asyncio
import contextlib
import copy
import inspect
import logging
import time
from collections.abc import Awaitable, Callable, Iterable, Mapping
from dataclasses import dataclass, replace
from types import SimpleNamespace
from typing import Any, Protocol, cast

import numpy as np

from saealib._deprecation import deprecated_function
from saealib.callback import (
    Event,
    GenerationEndEvent,
    GenerationStartEvent,
    RunEndEvent,
    RunStartEvent,
)
from saealib.comparators import NSGA3Comparator
from saealib.context import OptimizationState
from saealib.core.compiler.compiler import ExecutablePlan
from saealib.core.compiler.diagnostics import (
    ContractPath,
    Diagnostic,
    Severity,
)
from saealib.core.compiler.graph import ComponentNode
from saealib.core.compiler.regions import (
    BranchRegion,
    Condition,
    LoopRegion,
    RegionNode,
    RepeatRegion,
    SequenceRegion,
)
from saealib.core.compiler.structured import StructuredGraph
from saealib.core.contracts.contract import ComponentContract
from saealib.core.contracts.execution import RuntimeCapability
from saealib.core.contracts.vocabulary import validate_name
from saealib.core.runtime import (
    ExecutionRuntime,
    NodeResult,
    NodeStatus,
    PollResult,
    RegionFrame,
    RequestRecompile,
    RequestTermination,
    RuntimeCommand,
    RuntimeSession,
    RuntimeStep,
    SequentialPlan,
    StructuredPlan,
    validate_plan_contracts,
)
from saealib.core.state import (
    EVALUATION_HANDLES,
    EVALUATION_NEW_IDS,
    EVALUATION_REQUEST,
    EVALUATION_UPDATE_NEW_IDS,
    EVALUATION_UPDATES,
    EVALUATIONS_PLAN_UPDATES,
    OPTIMIZATION_STATE_INITIAL_KEYS,
    RUNTIME_REGION_COUNTS,
    RUNTIME_STEP,
    RuntimeContext,
    StateKey,
    StateView,
)
from saealib.core.state.patch import StatePatch
from saealib.core.state.readonly import _unwrap_readonly
from saealib.exceptions import (
    ConfigurationError,
    EvaluationFatalError,
    RejectedChange,
    ReplacementRejectedError,
    ValidationError,
)
from saealib.execution.bindings import (
    _ComponentOrigin,
    _ComponentResolution,
    _feature_encoder_signature,
    _resolve_explicit_feature_encoder,
    resolve_components,
)
from saealib.execution.history import (
    _ensure_region_history,
    record_decision,
    record_evaluations,
    record_generation,
)
from saealib.execution.prepare import BINDING_SLOTS, _prepare_plan
from saealib.identity import PopulationAttribute
from saealib.termination import Termination
from saealib.transition import (
    AlignColumns,
    Chain,
    StagedColumns,
    Transition,
    TransitionContext,
    _iter_applied,
)

logger = logging.getLogger(__name__)

__all__ = [
    "AsyncPipelineRuntime",
    "PipelineRuntime",
    "RuntimeFactory",
    "RuntimeRegistration",
    "RuntimeRegistry",
    "create_runtime",
    "default_runtime_registry",
    "execute_strategy_step",
    "resolve_plan",
]


def _iteration_region_ids(graph: StructuredGraph) -> tuple[str, ...]:
    region_ids: list[str] = []
    seen: set[str] = set()

    def visit(current: StructuredGraph) -> None:
        for operation in current.operations:
            if not isinstance(operation, RegionNode):
                continue
            region = operation.region
            if (
                isinstance(region, (RepeatRegion, LoopRegion))
                and region.qualified_id not in seen
            ):
                region_ids.append(region.qualified_id)
                seen.add(region.qualified_id)
            body = region.body
            if isinstance(body, StructuredGraph):
                visit(body)
            otherwise = getattr(region, "otherwise", None)
            if isinstance(otherwise, StructuredGraph):
                visit(otherwise)

    visit(graph)
    return tuple(region_ids)


def _increment_step(state: OptimizationState) -> None:
    try:
        store = object.__getattribute__(state, "_store")
    except AttributeError:
        return
    if store.contains(RUNTIME_STEP):
        state.step += 1


def _has_pending_evaluations(state: OptimizationState) -> bool:
    try:
        return bool(state.pending_evaluations)
    except (AttributeError, KeyError):
        return False


def _has_state_key(state: OptimizationState, key: StateKey) -> bool:
    try:
        store = object.__getattribute__(state, "_store")
    except AttributeError:
        return False
    return store.contains(key)


def _increment_region_count(state: OptimizationState, region_id: str) -> None:
    if not _has_state_key(state, RUNTIME_REGION_COUNTS):
        return
    counts = dict(state.region_counts)
    counts[region_id] = counts.get(region_id, 0) + 1
    state.region_counts = counts


class LifecycleEnvironment(Protocol):
    """Provider boundary needed by lifecycle-only runtime paths."""

    def is_terminated(self, state: OptimizationState) -> bool: ...

    def dispatch(self, event: Event) -> None: ...

    def finish_generation(self, state: OptimizationState) -> None: ...

    def fatal(self, state: OptimizationState) -> None: ...


class SequentialExecutionEnvironment(LifecycleEnvironment, Protocol):
    """Lifecycle provider with the sequential execution capability."""

    def execute(
        self, plan: SequentialPlan, state: OptimizationState
    ) -> OptimizationState: ...


class AsyncExecutionEnvironment(LifecycleEnvironment, Protocol):
    """Lifecycle provider with asynchronous execution capabilities."""

    def execute_async(
        self, plan: SequentialPlan, state: OptimizationState
    ) -> OptimizationState: ...

    def reattach(self, state: OptimizationState) -> OptimizationState: ...

    def poll(self, state: OptimizationState) -> PollResult: ...

    def can_refill(self, state: OptimizationState) -> bool: ...


class RecompileProvider(Protocol):
    """Provider capability for rebuilding a runtime plan."""

    def recompile(
        self, plan: SequentialPlan | StructuredPlan
    ) -> SequentialPlan | StructuredPlan: ...


@dataclass(frozen=True)
class _ExecutionOutcome:
    state: OptimizationState
    node_results: tuple[NodeResult, ...] = ()
    executed_node_ids: tuple[str, ...] = ()
    refused_commands: tuple[RuntimeCommand, ...] = ()
    plan_completed: bool = False
    terminated: bool = False
    frames: tuple[RegionFrame, ...] = ()

    @property
    def recompile_requested(self) -> bool:
        return any(
            isinstance(command, RequestRecompile)
            and command not in self.refused_commands
            for result in self.node_results
            for command in result.commands
        )


class _ExecutionHaltError(Exception):
    def __init__(self, outcome: _ExecutionOutcome) -> None:
        self.outcome = outcome


def _record_history_observation(state: OptimizationState) -> None:
    """Record all history channels at one existing runtime observation point."""
    record_decision(state)
    record_evaluations(state)


@dataclass(frozen=True)
class _LeafExecution:
    raw: object


_TRANSIENT_STATE_FIELDS = (
    ("evaluation_request", EVALUATION_REQUEST),
    ("evaluation_updates", EVALUATION_UPDATES),
    ("evaluation_update_new_ids", EVALUATION_UPDATE_NEW_IDS),
    ("evaluation_new_ids", EVALUATION_NEW_IDS),
    ("evaluation_handles", EVALUATION_HANDLES),
)


def _sync_transient_fields_from_store(state: OptimizationState) -> None:
    for name, key in _TRANSIENT_STATE_FIELDS:
        try:
            value = state.get_state(key)
        except KeyError:
            continue
        object.__setattr__(state, name, value)


def _transient_field_for_key(key: StateKey) -> str | None:
    return next(
        (name for name, candidate in _TRANSIENT_STATE_FIELDS if candidate == key),
        None,
    )


def _empty_transient_field(name: str) -> object:
    if name == "evaluation_request":
        return None
    if name == "evaluation_new_ids":
        return np.empty(0, dtype=np.int64)
    if name in {"evaluation_updates", "evaluation_update_new_ids"}:
        return []
    return {}


def _apply_structured_patch(state: OptimizationState, patch: StatePatch) -> None:
    transient_writes: dict[str, object] = {}
    transient_deletes: set[str] = set()
    store_writes: dict[StateKey, object] = {}
    store_deletes: set[StateKey] = set()
    for key, value in patch.writes.items():
        name = _transient_field_for_key(key)
        if name is None:
            store_writes[key] = value
        else:
            transient_writes[name] = _unwrap_readonly(value)
            store_deletes.add(key)
    for key in patch.deletes:
        name = _transient_field_for_key(key)
        if name is None:
            store_deletes.add(key)
        elif key not in patch.writes:
            transient_deletes.add(name)
            store_deletes.add(key)
    if store_writes or store_deletes:
        state._store = state._store.apply_patch(
            StatePatch(writes=store_writes, deletes=frozenset(store_deletes))
        )
    for name, value in transient_writes.items():
        object.__setattr__(state, name, value)
    for name in transient_deletes:
        object.__setattr__(state, name, _empty_transient_field(name))


def _environment_execute(
    environment: SequentialExecutionEnvironment,
    plan: SequentialPlan,
    state: OptimizationState,
) -> _ExecutionOutcome:
    logger.debug("Runtime synchronous execution start")
    try:
        execute_step = getattr(environment, "execute_step", None)
        if callable(execute_step):
            outcome = execute_step(plan, state)
            if isinstance(outcome, _ExecutionOutcome):
                return outcome
            if isinstance(outcome, RuntimeStep):
                return _ExecutionOutcome(
                    state=outcome.state,
                    node_results=outcome.node_results,
                    executed_node_ids=outcome.executed_node_ids,
                    refused_commands=outcome.refused_commands,
                    terminated=outcome.finished,
                )
            if isinstance(outcome, OptimizationState):
                return _ExecutionOutcome(state=outcome)
            raise ValidationError(
                "runtime environment execute_step returned an invalid value"
            )
        return _ExecutionOutcome(state=environment.execute(plan, state))
    finally:
        logger.debug("Runtime synchronous execution end")


def _environment_execute_async(
    environment: AsyncExecutionEnvironment,
    plan: SequentialPlan,
    state: OptimizationState,
) -> _ExecutionOutcome:
    logger.debug("Runtime asynchronous execution start")
    try:
        execute_step = getattr(environment, "execute_async_step", None)
        if callable(execute_step):
            outcome = execute_step(plan, state)
            if isinstance(outcome, _ExecutionOutcome):
                return outcome
            if isinstance(outcome, RuntimeStep):
                return _ExecutionOutcome(
                    state=outcome.state,
                    node_results=outcome.node_results,
                    executed_node_ids=outcome.executed_node_ids,
                    refused_commands=outcome.refused_commands,
                    terminated=outcome.finished,
                )
            if isinstance(outcome, OptimizationState):
                return _ExecutionOutcome(state=outcome)
            raise ValidationError(
                "runtime environment execute_async_step returned an invalid value"
            )
        return _ExecutionOutcome(state=environment.execute_async(plan, state))
    finally:
        logger.debug("Runtime asynchronous execution end")


def _environment_recompile(
    environment: RecompileProvider, plan: SequentialPlan | StructuredPlan
) -> SequentialPlan | StructuredPlan:
    recompile = getattr(environment, "recompile", None)
    if not callable(recompile):
        raise ValidationError(
            "runtime cannot satisfy RequestRecompile: no recompile provider"
        )
    logger.debug("Runtime recompile request accepted: %s", type(plan).__name__)
    rebuilt = recompile(plan)
    if not isinstance(rebuilt, (SequentialPlan, StructuredPlan)):
        raise ValidationError("runtime recompile provider returned an invalid plan")
    if isinstance(plan, StructuredPlan) is not isinstance(rebuilt, StructuredPlan):
        raise ValidationError("runtime recompile provider changed plan structure")
    logger.debug("Runtime recompile result: %s", type(rebuilt).__name__)
    return rebuilt


def _structured_recompile_accepted(
    metadata: _ExecutionOutcome,
) -> bool:
    return (
        metadata.recompile_requested
        and metadata.plan_completed
        and metadata.frames == ()
        and not metadata.state.pending_evaluations
        and not any(
            result.status in {NodeStatus.BLOCKED, NodeStatus.RUNNING, NodeStatus.FAILED}
            for result in metadata.node_results
        )
    )


def _refuse_structured_recompile(
    metadata: _ExecutionOutcome,
) -> _ExecutionOutcome:
    requests = tuple(
        command
        for result in metadata.node_results
        for command in result.commands
        if isinstance(command, RequestRecompile)
        and command not in metadata.refused_commands
    )
    if not requests:
        return metadata
    return replace(
        metadata,
        refused_commands=(*metadata.refused_commands, *requests),
    )


class _BoundStateView:
    """StateView-compatible alias projection for a bound component contract."""

    def __init__(self, view: StateView, aliases: dict[StateKey, StateKey]) -> None:
        self._view = view
        self._aliases = aliases

    def _resolve(self, key: StateKey) -> StateKey:
        return self._aliases.get(key, key)

    def get(self, key: StateKey) -> object:
        resolved = self._resolve(key)
        if self._view.contains(resolved):
            return self._view.get(resolved)
        name = _transient_field_for_key(resolved)
        if name is not None:
            return getattr(self._view.context, name)
        return self._view.get(resolved)

    def contains(self, key: StateKey) -> bool:
        resolved = self._resolve(key)
        if self._view.contains(resolved):
            return True
        return _transient_field_for_key(resolved) is not None

    @property
    def context(self) -> object:
        return self._view.context

    def dispatch(self, event: object) -> None:
        self._view.dispatch(event)


def _resolve_state_key(key: StateKey, bindings: tuple[StateKey, ...]) -> StateKey:
    candidates = [item for item in bindings if item.namespace == key.namespace]
    exact = [item for item in candidates if item.name == key.name]
    if len(exact) == 1:
        return exact[0]
    if len(candidates) == 1:
        return candidates[0]
    return key


def _node_state_aliases(
    plan: SequentialPlan | StructuredPlan, node: ComponentNode
) -> dict[StateKey, StateKey]:
    bindings = tuple(
        binding.state_key
        for binding in plan.plan.graph.state_bindings
        if binding.node.component_id == node.component_id
    )
    contract = node.contract.state
    aliases: dict[StateKey, StateKey] = {}
    for key in (*contract.reads, *contract.writes, *contract.exports):
        resolved = _resolve_state_key(key, bindings)
        if resolved != key:
            aliases[key] = resolved
    return aliases


def _bound_patch(
    plan: SequentialPlan | StructuredPlan, node: ComponentNode, patch: StatePatch
) -> StatePatch:
    aliases = _node_state_aliases(plan, node)
    if not node.contract.state.writes_enumerable:
        # The node declares its writes non-enumerable (an opaque Stage), so it
        # may write any key; only the declared aliases are translated and no
        # allowed-set restriction is applied.
        writes = {aliases.get(key, key): value for key, value in patch.writes.items()}
        deletes = frozenset(aliases.get(key, key) for key in patch.deletes)
        return StatePatch(writes=writes, deletes=deletes)
    allowed = {aliases.get(key, key) for key in node.contract.state.writes}
    writes = {}
    for key, value in patch.writes.items():
        resolved = aliases.get(key, key)
        if resolved not in allowed:
            raise ValidationError(
                f"Graph node {node.component_id!r} attempted an undeclared "
                f"state write: {key!r}"
            )
        writes[resolved] = value
    deletes = set()
    for key in patch.deletes:
        resolved = aliases.get(key, key)
        if resolved not in allowed:
            raise ValidationError(
                f"Graph node {node.component_id!r} attempted an undeclared "
                f"state delete: {key!r}"
            )
        deletes.add(resolved)
    return StatePatch(writes=writes, deletes=frozenset(deletes))


def _evaluate_condition(
    condition: Condition,
    state: OptimizationState,
    *,
    dispatch: Callable[[Event], None] | None,
) -> bool:
    contract = condition.contract()
    view = state._store.view(
        contract.reads,
        context=RuntimeContext(
            state, reads=contract.reads, dispatch=dispatch, resolved_services={}
        ),
        dispatch=dispatch,
    )
    return bool(condition.evaluate(view))


def _async_stage_driver(component: object) -> Callable[..., object] | None:
    stage = getattr(component, "stage", None)
    if stage is None or not callable(getattr(stage, "execute_async", None)):
        return None
    target = getattr(component, "execute_async", None)
    return target if callable(target) else None


def _async_protocol_marker(component: object) -> tuple[str, str] | None:
    stage = getattr(component, "stage", None)
    protocol = getattr(component, "async_protocol", None)
    role = getattr(component, "async_protocol_role", None)
    if stage is not None:
        protocol = getattr(stage, "async_protocol", protocol)
        role = getattr(stage, "async_protocol_role", role)
    if not isinstance(protocol, str) or not isinstance(role, str):
        return None
    return protocol, role


def _async_leaf_driver(component: object) -> Callable[..., object] | None:
    if getattr(component, "stage", None) is not None:
        return _async_stage_driver(component)
    target = getattr(component, "execute_async", None)
    return target if callable(target) else None


def _async_driver_required(node: ComponentNode) -> bool:
    return bool(
        getattr(node.component, "requires_async_execution", False)
        or getattr(node.component, "async_execution_required", False)
        or "partial_feedback" in node.contract.execution.required_runtime_capabilities
    )


def _structured_async_plan_complete(state: OptimizationState) -> bool:
    plan = getattr(state, "evaluation_plan", None)
    plan_state = getattr(state, "evaluation_plan_state", None)
    if plan is None or plan_state is None:
        return False
    if getattr(state, "pending_evaluations", {}) or getattr(
        state, "evaluation_handles", {}
    ):
        return False
    request_ids = {int(request.request_id) for request in plan.requests}
    terminal = set(plan_state.completed) | set(plan_state.acknowledged)
    return request_ids <= terminal


def _structured_async_has_unacknowledged_deliveries(state: OptimizationState) -> bool:
    """Delivered updates whose acknowledge step has not run yet."""
    plan = getattr(state, "evaluation_plan", None)
    plan_state = getattr(state, "evaluation_plan_state", None)
    if plan is None or plan_state is None:
        return False
    return bool(set(plan_state.completed) - set(plan_state.acknowledged))


def _structured_async_waiting(node: ComponentNode, state: OptimizationState) -> bool:
    return bool(
        EVALUATION_UPDATES in node.contract.state.writes
        and EVALUATIONS_PLAN_UPDATES in node.contract.state.reads
        and (
            getattr(state, "pending_evaluations", {})
            or _structured_async_plan_complete(state)
        )
    )


def _rewind_to_async_driver(
    frames: tuple[RegionFrame, ...],
) -> tuple[RegionFrame, ...]:
    if not frames:
        return frames
    frame = frames[-1]
    driver_index = None
    for index in range(frame.operation_index - 1, -1, -1):
        if _is_async_driver(frame.graph.operations[index]):
            driver_index = index
            break
    if driver_index is None:
        if len(frames) == 1 and frame.operation_index == 0:
            return _frames_at_nested_async_driver(frame.graph) or frames
        return frames
    return (*frames[:-1], replace(frame, operation_index=driver_index))


def _async_resume_frames(
    frames: tuple[RegionFrame, ...],
    state: OptimizationState,
    *,
    plan_graph: StructuredGraph | None = None,
    strategy: object | None = None,
    for_checkpoint: bool = False,
) -> tuple[RegionFrame, ...] | None:
    """Return the exact frames used to resume an asynchronous wait point."""
    if not frames:
        return None
    if for_checkpoint and not getattr(state, "pending_evaluations", {}):
        return None
    frame = frames[-1]
    at_driver = frame.operation_index < len(
        frame.graph.operations
    ) and _is_async_driver(frame.graph.operations[frame.operation_index])
    rewound = _rewind_to_async_driver(frames)
    if not at_driver and rewound == frames:
        return None
    source_graph = plan_graph or frames[0].graph
    if (
        for_checkpoint
        and _is_generation_based_strategy(strategy)
        and _contains_repeat_region(source_graph)
    ):
        # The active Repeat iteration is not represented by the frames rebuilt
        # at load time, even when their static driver positions happen to match.
        return None
    expected = _frames_at_nested_async_driver(source_graph)
    if expected is None or rewound != expected:
        return None
    return expected


def _is_generation_based_strategy(strategy: object) -> bool:
    """Match GenerationBasedStrategy subclasses without importing at module load."""
    from saealib.strategies import GenerationBasedStrategy

    return isinstance(strategy, GenerationBasedStrategy)


def _contains_repeat_region(graph: StructuredGraph) -> bool:
    for operation in graph.operations:
        if not isinstance(operation, RegionNode):
            continue
        if isinstance(operation.region, RepeatRegion):
            return True
        body = operation.region.body
        if isinstance(body, StructuredGraph) and _contains_repeat_region(body):
            return True
    return False


def _is_async_driver(operation: object) -> bool:
    return (
        isinstance(operation, ComponentNode)
        and _async_leaf_driver(operation.component) is not None
        and (marker := _async_protocol_marker(operation.component)) is not None
        and marker[1] == "driver"
    )


def _frames_at_nested_async_driver(
    graph: StructuredGraph,
) -> tuple[RegionFrame, ...] | None:
    """Frames the load path rebuilds to reach an asynchronous driver.

    A run resumed from a checkpoint starts without frames; when its active
    plan still has requests to submit, execution continues at the driver that
    opened the plan instead of starting a new step.
    """
    for index, operation in enumerate(graph.operations):
        if _is_async_driver(operation):
            return (RegionFrame(graph=graph, operation_index=index),)
    for index, operation in enumerate(graph.operations):
        if not isinstance(operation, RegionNode) or not isinstance(
            operation.region, SequenceRegion
        ):
            continue
        body = operation.region.body
        if not isinstance(body, StructuredGraph):
            continue
        root = RegionFrame(graph=graph, operation_index=index)
        for driver_index, child in enumerate(body.operations):
            if _is_async_driver(child):
                return (
                    root,
                    RegionFrame(
                        graph=body,
                        region_id=operation.region.qualified_id,
                        operation_index=driver_index,
                    ),
                )
        nested = _frames_at_nested_async_driver(body)
        if nested is not None:
            return (
                root,
                replace(nested[0], region_id=operation.region.qualified_id),
                *nested[1:],
            )
    return None


def _invoke_async_leaf(
    target: Callable[..., object],
    view: _BoundStateView,
    kwargs: dict[str, object],
) -> object:
    try:
        parameters = inspect.signature(target).parameters
    except (TypeError, ValueError) as exc:
        raise ValidationError(
            "async leaf execution target must be inspectable"
        ) from exc
    if any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    ):
        accepted = kwargs
    else:
        accepted = {name: value for name, value in kwargs.items() if name in parameters}
    result = target(view, **accepted)
    if not inspect.isawaitable(result):
        return result
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(_await_async_result(result))
    raise ValidationError(
        "structured async runtime cannot await a leaf from a running event loop"
    )


async def _await_async_result(result: Awaitable[object]) -> object:
    return await result


def _state_boundary_target(
    component: object, name: str
) -> Callable[..., OptimizationState] | None:
    """Resolve a state-boundary runner through schema-freshened wrappers."""
    target = component
    for _ in range(4):
        candidate = inspect.getattr_static(target, name, None)
        if candidate is not None:
            return cast(
                Callable[[OptimizationState], OptimizationState],
                getattr(target, name),
            )
        target = getattr(target, "_component", None)
        if target is None:
            return None
    return None


def _structured_async_leaf_executor(
    async_kwargs: dict[str, object],
) -> Callable[
    [StructuredPlan, ComponentNode, OptimizationState, _BoundStateView], _LeafExecution
]:
    def execute(
        plan: StructuredPlan,
        node: ComponentNode,
        state: OptimizationState,
        view: _BoundStateView,
    ) -> _LeafExecution:
        if getattr(
            node.component, "execution_boundary", "transaction"
        ) == "state" and not _async_driver_required(node):
            seam = _state_boundary_target(node.component, "execute_async_state")
            if seam is not None:
                return _LeafExecution(
                    _invoke_async_leaf(
                        seam,
                        cast(_BoundStateView, state),
                        async_kwargs,
                    )
                )
            run_state = _state_boundary_target(node.component, "execute_state")
            if run_state is None:
                raise ValidationError(
                    f"state-boundary node {node.component_id!r} lost its runner"
                )
            return _LeafExecution(run_state(state))
        target = _async_leaf_driver(node.component)
        if target is None:
            if _async_driver_required(node):
                raise ValidationError(
                    f"Structured async node {node.component_id!r} requires an "
                    "async execution driver"
                )
            return _LeafExecution(plan._execute_targets[node.component_id](view))
        return _LeafExecution(_invoke_async_leaf(target, view, async_kwargs))

    return execute


def _async_protocol_end_index(
    graph: StructuredGraph, operation_index: int
) -> int | None:
    protocol: str | None = None
    for index in range(operation_index, -1, -1):
        operation = graph.operations[index]
        if not isinstance(operation, ComponentNode):
            continue
        marker = _async_protocol_marker(operation.component)
        if marker is not None and marker[1] == "driver":
            protocol = marker[0]
            break
    if protocol is None:
        return None
    for index in range(operation_index, len(graph.operations)):
        operation = graph.operations[index]
        if not isinstance(operation, ComponentNode):
            continue
        marker = _async_protocol_marker(operation.component)
        if marker == (protocol, "end"):
            return index
    return None


def _execute_structured(
    plan: StructuredPlan,
    state: OptimizationState,
    frames: tuple[RegionFrame, ...] = (),
    *,
    dispatch: Callable[[Event], None] | None = None,
    observe: Callable[[OptimizationState], None] | None = None,
    leaf_executor: Callable[
        [StructuredPlan, ComponentNode, OptimizationState, _BoundStateView],
        _LeafExecution,
    ]
    | None = None,
    async_control: bool = False,
    resume_async_driver: bool = False,
) -> _ExecutionOutcome:
    """Execute structured operations until a node yields or the graph ends."""
    _sync_transient_fields_from_store(state)
    current = state
    stack = list(frames or (RegionFrame(graph=plan.graph),))
    checkpoint_owner = getattr(dispatch, "__self__", None)
    if async_control and resume_async_driver:
        strategy = getattr(
            getattr(checkpoint_owner, "optimizer", None), "strategy", None
        )
        reconstructed_frames = _frames_at_nested_async_driver(plan.graph)
        resumed_frames = _async_resume_frames(
            reconstructed_frames or tuple(stack),
            current,
            plan_graph=plan.graph,
            strategy=strategy,
        )
        if resumed_frames is not None:
            stack = list(resumed_frames)
    results: list[NodeResult] = []
    executed: list[str] = []
    refused: list[RuntimeCommand] = []
    debug_enabled = logger.isEnabledFor(logging.DEBUG)
    mark_checkpoint = getattr(checkpoint_owner, "_mark_checkpoint_state", None)

    def finish_child() -> bool:
        """Advance an enclosing region after its body reaches its end."""
        nonlocal stack
        if len(stack) == 1:
            return True
        child = stack[-1]
        parent = stack[-2]
        operation = parent.graph.operations[parent.operation_index]
        if not isinstance(operation, RegionNode):
            raise ValidationError("StructuredPlan frame does not point to a region")
        region = operation.region
        if isinstance(region, RepeatRegion):
            if child.region_id is not None:
                _increment_region_count(current, child.region_id)
            count = child.count if child.count is not None else 0
            if child.iteration + 1 < count:
                stack[-1] = replace(
                    child, operation_index=0, iteration=child.iteration + 1
                )
                return False
        elif isinstance(region, LoopRegion):
            if child.region_id is not None:
                _increment_region_count(current, child.region_id)
            if not _evaluate_condition(region.condition, current, dispatch=dispatch):
                stack[-1] = replace(
                    child, operation_index=0, iteration=child.iteration + 1
                )
                return False
        stack.pop()
        stack[-1] = replace(parent, operation_index=parent.operation_index + 1)
        return False

    while True:
        if callable(mark_checkpoint):
            mark_checkpoint(current, phase="running", frames=tuple(stack[1:]))
        frame = stack[-1]
        if frame.operation_index >= len(frame.graph.operations):
            if finish_child():
                return _ExecutionOutcome(
                    state=current,
                    node_results=tuple(results),
                    executed_node_ids=tuple(executed),
                    refused_commands=tuple(refused),
                    plan_completed=True,
                    frames=(),
                )
            continue

        operation = frame.graph.operations[frame.operation_index]
        if isinstance(operation, RegionNode):
            region = operation.region
            body = region.body
            if not isinstance(body, StructuredGraph):
                raise ValidationError(
                    "StructuredPlan region bodies must be StructuredGraph values"
                )
            if isinstance(region, RepeatRegion):
                count = region.count
                if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                    raise ValidationError("Repeat count must be a non-negative integer")
                if count == 0:
                    stack[-1] = replace(
                        frame, operation_index=frame.operation_index + 1
                    )
                    continue
                stack.append(
                    RegionFrame(graph=body, region_id=region.qualified_id, count=count)
                )
                continue
            if isinstance(region, LoopRegion):
                if _evaluate_condition(region.condition, current, dispatch=dispatch):
                    stack[-1] = replace(
                        frame, operation_index=frame.operation_index + 1
                    )
                    continue
                stack.append(RegionFrame(graph=body, region_id=region.qualified_id))
                continue
            if isinstance(region, BranchRegion):
                selected = region.body
                branch = "then"
                if not _evaluate_condition(
                    region.condition, current, dispatch=dispatch
                ):
                    selected = region.otherwise
                    branch = "else"
                if not isinstance(selected, StructuredGraph):
                    stack[-1] = replace(
                        frame, operation_index=frame.operation_index + 1
                    )
                    continue
                stack.append(
                    RegionFrame(
                        graph=selected, region_id=region.qualified_id, branch=branch
                    )
                )
                continue
            stack.append(RegionFrame(graph=body, region_id=region.qualified_id))
            continue

        node = operation
        if async_control and _structured_async_waiting(node, current):
            if _structured_async_plan_complete(current):
                end_index = _async_protocol_end_index(
                    frame.graph, frame.operation_index
                )
                if end_index is not None:
                    stack[-1] = replace(frame, operation_index=end_index + 1)
                    continue
            executed.append(node.component_id)
            if debug_enabled:
                logger.debug("Runtime node start: %s", node.component_id)
            result = NodeResult(
                patch=StatePatch(writes={}),
                status=NodeStatus.BLOCKED,
            )
            results.append(result)
            if debug_enabled:
                logger.debug(
                    "Runtime node end: %s status=%s", node.component_id, result.status
                )
            return _ExecutionOutcome(
                state=current,
                node_results=tuple(results),
                executed_node_ids=tuple(executed),
                refused_commands=tuple(refused),
                frames=tuple(stack),
            )
        executed.append(node.component_id)
        if debug_enabled:
            logger.debug("Runtime node start: %s", node.component_id)
        if (
            not async_control
            and getattr(node.component, "execution_boundary", "transaction") == "state"
        ):
            run_state = _state_boundary_target(node.component, "execute_state")
            if run_state is None:
                raise ValidationError(
                    f"state-boundary node {node.component_id!r} lost its runner"
                )
            next_state = run_state(current)
            current = next_state
            if observe is not None:
                observe(current)
            results.append(NodeResult(patch=StatePatch(writes={})))
            stack[-1] = replace(frame, operation_index=frame.operation_index + 1)
            continue
        aliases = _node_state_aliases(plan, node)
        view = current._store.view(
            tuple({aliases.get(key, key) for key in node.contract.state.reads}),
            context=RuntimeContext(
                current,
                reads=node.contract.state.reads,
                dispatch=dispatch,
                resolved_services=node.resolved_services,
            ),
            dispatch=dispatch,
        )
        bound_view = _BoundStateView(view, aliases)
        leaf = (
            leaf_executor(plan, node, current, bound_view)
            if leaf_executor is not None
            else _LeafExecution(plan._execute_targets[node.component_id](bound_view))
        )
        raw = leaf.raw
        if isinstance(raw, OptimizationState):
            # A declared state-boundary leaf returned the run's own state.
            current = raw
            if observe is not None:
                observe(current)
            results.append(NodeResult(patch=StatePatch(writes={})))
            stack[-1] = replace(frame, operation_index=frame.operation_index + 1)
            continue
        if isinstance(raw, StatePatch):
            result = NodeResult(patch=raw)
        elif isinstance(raw, NodeResult):
            result = raw
        else:
            raise ValidationError(
                f"Graph node {node.component_id!r} must return NodeResult or StatePatch"
            )
        patch = _bound_patch(plan, node, result.patch)
        if result.status is not NodeStatus.FAILED and (patch.writes or patch.deletes):
            _apply_structured_patch(current, patch)
            if observe is not None:
                observe(current)
        if dispatch is not None:
            for event in result.events:
                dispatch(event)
        results.append(result)
        recompile_commands: list[RequestRecompile] = []
        for command in result.commands:
            if isinstance(command, RequestRecompile):
                recompile_commands.append(command)
                if debug_enabled:
                    logger.debug(
                        "Runtime recompile request: node=%s reason=%s",
                        node.component_id,
                        command.reason,
                    )
        recompile_requested = bool(recompile_commands)
        if debug_enabled:
            logger.debug(
                "Runtime node end: %s status=%s", node.component_id, result.status
            )
        for command in result.commands:
            if not isinstance(command, (RequestTermination, RequestRecompile)):
                refused.append(command)
        if result.status is NodeStatus.COMPLETED:
            stack[-1] = replace(frame, operation_index=frame.operation_index + 1)
            root_complete = len(stack) == 1 and stack[-1].operation_index >= len(
                stack[-1].graph.operations
            )
            if recompile_requested and root_complete:
                return _ExecutionOutcome(
                    state=current,
                    node_results=tuple(results),
                    executed_node_ids=tuple(executed),
                    refused_commands=tuple(refused),
                    plan_completed=True,
                    frames=(),
                )
            if recompile_requested:
                refused.extend(recompile_commands)
        elif recompile_requested:
            refused.extend(recompile_commands)
        if result.status is not NodeStatus.COMPLETED or any(
            isinstance(command, RequestTermination) for command in result.commands
        ):
            return _ExecutionOutcome(
                state=current,
                node_results=tuple(results),
                executed_node_ids=tuple(executed),
                refused_commands=tuple(refused),
                plan_completed=False,
                terminated=any(
                    isinstance(command, RequestTermination)
                    for command in result.commands
                ),
                frames=tuple(stack),
            )


def _execute_sequential_plan(
    plan: SequentialPlan,
    state: OptimizationState,
    *,
    dispatch: Callable[[Event], None] | None = None,
) -> OptimizationState:
    return _execute_with_metadata(
        plan, state, dispatch=dispatch, observe=_record_history_observation
    ).state


def _execute_with_metadata(
    plan: SequentialPlan,
    state: OptimizationState,
    *,
    dispatch: Callable[[Event], None] | None = None,
    observe: Callable[[OptimizationState], None] | None = None,
) -> _ExecutionOutcome:
    from saealib.core.graph_builder import StageNodeAdapter
    from saealib.pipeline import Stage

    results: list[NodeResult] = []
    executed: list[str] = []
    refused: list[RuntimeCommand] = []
    current = state
    debug_enabled = logger.isEnabledFor(logging.DEBUG)
    for node, execute in zip(plan.execution_nodes, plan._execute_targets):
        executed.append(node.component_id)
        stage = getattr(node.component, "stage", None)
        if isinstance(node.component, StageNodeAdapter) or isinstance(stage, Stage):
            if debug_enabled:
                logger.debug("Runtime stage start: %s", node.component_id)
            next_state = execute(current)
            if not isinstance(next_state, OptimizationState):
                raise ValidationError(
                    f"Stage node {node.component_id!r} did not return an "
                    "OptimizationState"
                )
            current = next_state
            if observe is not None:
                observe(current)
            results.append(NodeResult(patch=StatePatch(writes={})))
            if debug_enabled:
                logger.debug(
                    "Runtime stage end: %s status=%s",
                    node.component_id,
                    NodeStatus.COMPLETED,
                )
            continue
        if debug_enabled:
            logger.debug("Runtime node start: %s", node.component_id)
        aliases = _node_state_aliases(plan, node)
        view = current._store.view(
            tuple({aliases.get(key, key) for key in node.contract.state.reads}),
            context=RuntimeContext(
                current,
                reads=node.contract.state.reads,
                dispatch=dispatch,
                resolved_services=node.resolved_services,
            ),
            dispatch=dispatch,
        )
        raw = execute(_BoundStateView(view, aliases))
        if isinstance(raw, StatePatch):
            result = NodeResult(patch=raw)
        elif isinstance(raw, NodeResult):
            result = raw
        else:
            raise ValidationError(
                f"Graph node {node.component_id!r} must return NodeResult or StatePatch"
            )
        patch = _bound_patch(plan, node, result.patch)
        if result.status is not NodeStatus.FAILED and (patch.writes or patch.deletes):
            current._store = current._store.apply_patch(patch)
            if observe is not None:
                observe(current)
        if dispatch is not None:
            for event in result.events:
                dispatch(event)
        results.append(result)
        if debug_enabled:
            for command in result.commands:
                if isinstance(command, RequestRecompile):
                    logger.debug(
                        "Runtime recompile request: node=%s reason=%s",
                        node.component_id,
                        command.reason,
                    )
            logger.debug(
                "Runtime node end: %s status=%s", node.component_id, result.status
            )
        for command in result.commands:
            if isinstance(command, (RequestTermination, RequestRecompile)):
                continue
            refused.append(command)
        if result.status in {
            NodeStatus.FAILED,
            NodeStatus.BLOCKED,
            NodeStatus.RUNNING,
        }:
            break
        if any(isinstance(command, RequestTermination) for command in result.commands):
            break
    return _ExecutionOutcome(
        state=current,
        node_results=tuple(results),
        executed_node_ids=tuple(executed),
        refused_commands=tuple(refused),
        plan_completed=not results
        or all(
            result.status is NodeStatus.COMPLETED
            and not any(
                isinstance(command, RequestTermination) for command in result.commands
            )
            for result in results
        ),
        terminated=any(
            isinstance(command, RequestTermination)
            for result in results
            for command in result.commands
        ),
    )


def _algorithm_runtime_capabilities(
    optimizer: object,
) -> frozenset[RuntimeCapability]:
    algorithm = getattr(optimizer, "algorithm", None)
    contract_method = getattr(algorithm, "contract", None)
    contract = contract_method() if callable(contract_method) else None
    execution = getattr(contract, "execution", None)
    required = getattr(execution, "required_runtime_capabilities", ())
    if "partial_feedback" in required:
        return frozenset({"partial_feedback"})
    return frozenset()


_BINDING_SLOTS = BINDING_SLOTS
_UNSET = object()


class _SnapshotProviderView:
    """Provider view over one candidate binding set.

    Slots are served from the pending snapshot values, so plan preparation
    never reads the live optimizer, even mid-run.  Non-slot attributes fall
    through to the optimizer (problem, space, runtime flags).
    """

    _optimizer: Any
    _bindings: dict[str, object]
    _component_resolution: _ComponentResolution | None
    _stage_selections: list[tuple[str, str, object, str]]

    def __init__(
        self,
        optimizer: Any,
        bindings: dict[str, object],
        resolution: _ComponentResolution | None = None,
    ) -> None:
        object.__setattr__(self, "_optimizer", optimizer)
        object.__setattr__(self, "_bindings", bindings)
        object.__setattr__(self, "_component_resolution", resolution)
        object.__setattr__(self, "_stage_selections", [])

    def binding_values(self) -> dict[str, object]:
        """Candidate snapshot values for declaration resolution."""
        return dict(self._bindings)

    def __getattr__(self, name: str) -> object:
        if name in {"_optimizer", "_bindings"} or (
            name.startswith("__") and name.endswith("__")
        ):
            raise AttributeError(name)
        bindings = object.__getattribute__(self, "_bindings")
        if name in _BINDING_SLOTS:
            return bindings.get(name)
        optimizer = object.__getattribute__(self, "_optimizer")
        return getattr(optimizer, name)

    def __setattr__(self, name: str, value: object) -> None:
        optimizer = object.__getattribute__(self, "_optimizer")
        setattr(optimizer, name, value)


def _contract_records(
    bindings: Mapping[str, object],
) -> tuple[tuple[str, ComponentContract], ...]:
    """Return the contract of every binding that declares a readable one.

    A slot whose ``contract`` attribute cannot be looked up, is missing, is not
    callable, or raises is left out: a commit point compares the contracts it
    can read and keeps running otherwise.
    """
    records: list[tuple[str, ComponentContract]] = []
    for name, component in bindings.items():
        try:
            contract_method = getattr(component, "contract", None)
            if not callable(contract_method):
                continue
            contract = contract_method()
        except Exception:
            continue
        if isinstance(contract, ComponentContract):
            records.append((name, contract))
    return tuple(records)


def _plan_declared_state_keys(
    plan: SequentialPlan | StructuredPlan,
) -> tuple[frozenset[StateKey], frozenset[StateKey]] | None:
    """Return the state keys a plan's components declare they read and write.

    ``None`` when a component declares its effects non-enumerably: the caller
    then cannot verify the keys and must not invalidate any value.
    """
    reads: set[StateKey] = set()
    writes: set[StateKey] = set()
    try:
        components = tuple(_iter_plan_components(plan))
    except Exception:
        return None
    for node in components:
        try:
            state = node.contract.state
        except Exception:
            return None
        if not state.reads_enumerable or not state.writes_enumerable:
            return None
        reads.update(state.reads)
        writes.update(state.writes)
    return frozenset(reads), frozenset(writes)


@dataclass(frozen=True)
class _BindingSnapshot:
    """One published binding set: component identities plus a version.

    Holds strong references to every bound component so identity comparison
    at later commit points cannot be confused by object id reuse.  The stored
    preset object is compared by identity too: component resolution reads it
    lazily during publication, so a preset reaches a commit point without
    being a bound slot.  ``contracts`` records the contract of every slot
    that declares one, so a slot whose nested part changed in place is
    distinguishable from an unchanged one even though it holds the same
    object.
    """

    version: int
    values: dict[str, object]
    provenance: dict[str, _ComponentOrigin]
    feature_encoder_selection: tuple[str, str, str] | None
    strategy_values: tuple[tuple[str, str], ...]
    configuration_version: int
    preset: object = None
    contracts: tuple[tuple[str, ComponentContract], ...] = ()
    state_reads: frozenset[StateKey] | None = None
    state_writes: frozenset[StateKey] | None = None

    def unchanged_by(
        self,
        values: dict[str, object],
        strategy_values: tuple[tuple[str, str], ...],
        feature_encoder_selection: tuple[str, str, str] | None,
        preset: object,
    ) -> bool:
        """Return True when the candidate bindings resolve to this snapshot."""
        if set(values) != set(self.values):
            return False
        if any(values[name] is not bound for name, bound in self.values.items()):
            return False
        return (
            feature_encoder_selection == self.feature_encoder_selection
            and strategy_values == self.strategy_values
            and preset is getattr(self, "preset", None)
        )

    def contracts_match(
        self, contracts: tuple[tuple[str, ComponentContract], ...]
    ) -> bool:
        """Return True when every readable candidate contract is the published one.

        A slot whose contract cannot be read right now is left out of the
        comparison rather than counted as a change.
        """
        published = dict(self.contracts)
        return all(published.get(slot) == contract for slot, contract in contracts)

    def __getstate__(self) -> dict[str, object]:
        """Drop the contract records, which pickle cannot carry."""
        state = self.__dict__.copy()
        state["contracts"] = ()
        return state

    def __setstate__(self, state: dict[str, object]) -> None:
        """Rebuild the contract records from the components this snapshot holds."""
        self.__dict__.update(state)
        self.__dict__["contracts"] = _contract_records(dict(self.values))


def _executable_uses_service(executable: object, service_name: str) -> bool:
    graph = getattr(executable, "graph", None)
    return any(
        service_name in node.resolved_services for node in getattr(graph, "nodes", ())
    )


@dataclass(frozen=True)
class _PublishedPlan:
    plan: SequentialPlan | StructuredPlan
    executable: ExecutablePlan | None
    snapshot: _BindingSnapshot


def _restore_attribute(target: object, name: str, value: object) -> None:
    """Put one attribute back to a captured value, removing it when it was absent."""
    if value is _UNSET:
        if hasattr(target, name):
            object.__delattr__(target, name)
        return
    object.__setattr__(target, name, value)


@dataclass(frozen=True)
class _CommitMemento:
    """Every term one commit point may change, captured before it changes one.

    The captured values are the restore targets, not a log of writes:
    ``attributes`` holds the component the optimizer must hold again after a
    failure, by identity, for every slot the published snapshot recorded, and
    the published snapshot supplies the preset, the configuration counter, the
    origin and materialized-component contents, the feature-encoder default,
    and the feedback-builder flag.  The remaining terms are the live runtime
    values the commit replaces: the resolution results, the published binding
    record and plan, the state's binding version, store, custom state,
    comparator, compiled services, and history, the region history the
    candidate plan would add, the scheduler attributes publication writes, and
    the staged column writes.
    """

    attributes: Mapping[str, object]
    preset: object
    configuration_version: object
    origins: Mapping[str, object]
    live_origins: Mapping[str, object]
    materialized: Mapping[str, object]
    feature_encoder_default: object
    comparator_default: object
    feedback_builder_explicit: object
    resolution: object
    executable_plan: object
    default_resolution: object
    prepared_diagnostic_cache: object
    last_published_bindings: object
    published: object
    snapshot_values: Mapping[str, object]
    binding_version: object
    store: object
    custom_state: object
    comparator: object
    compiled_services: object
    history: object
    region_ids: tuple[str, ...]
    history_columns: Mapping[str, frozenset[str]]
    scheduler: Mapping[str, object]
    staged: _StagedCollections
    candidate_resolution: object
    candidate_diagnostics: tuple[Diagnostic, ...]
    publication_deferred: bool
    pending_recompile: bool


def _iter_plan_components(
    plan: SequentialPlan | StructuredPlan,
) -> Iterable[ComponentNode]:
    if isinstance(plan, StructuredPlan):
        for operation in plan.graph.operations:
            if isinstance(operation, ComponentNode):
                yield operation
            elif isinstance(operation, RegionNode):
                body = operation.region.body
                if isinstance(body, StructuredGraph):
                    yield from _iter_plan_components_from_graph(body)
    else:
        yield from plan.execution_nodes


def _iter_plan_components_from_graph(
    graph: StructuredGraph,
) -> Iterable[ComponentNode]:
    for operation in graph.operations:
        if isinstance(operation, ComponentNode):
            yield operation
        elif isinstance(operation, RegionNode):
            body = operation.region.body
            if isinstance(body, StructuredGraph):
                yield from _iter_plan_components_from_graph(body)


def _structured_true_feedback_builder(plan: object) -> object | None:
    """Return the builder of the Stage that feeds a plan's true evaluations back.

    A structured plan builds true feedback in the ``FeedbackStage`` of the
    sequence that plans the evaluations; asynchronous delivery uses that same
    builder.  ``None`` when the plan has no such unique Stage.
    """
    from saealib.stages import EvaluationPlanStage, FeedbackStage

    if not isinstance(plan, StructuredPlan):
        return None
    builders: list[object] = []

    def visit(graph: StructuredGraph) -> None:
        stages = tuple(
            getattr(operation.component, "stage", None)
            for operation in graph.operations
            if isinstance(operation, ComponentNode)
        )
        if any(isinstance(stage, EvaluationPlanStage) for stage in stages):
            builders.extend(
                stage._builder for stage in stages if isinstance(stage, FeedbackStage)
            )
        for region_node in graph.region_nodes:
            for body in (
                region_node.region.body,
                getattr(region_node.region, "otherwise", None),
            ):
                if isinstance(body, StructuredGraph):
                    visit(body)

    visit(plan.graph)
    return builders[0] if len(builders) == 1 else None


def _plan_has_unfinished_async_work(
    plan: SequentialPlan | StructuredPlan, state: OptimizationState
) -> bool:
    for node in _iter_plan_components(plan):
        capability = getattr(node.component, "has_async_work", None)
        if callable(capability) and capability(state):
            return True
    return False


def _replacement_blocked(
    scheduler: object | None,
    plan: SequentialPlan | StructuredPlan | None,
    state: OptimizationState | None,
) -> bool:
    if scheduler is None or state is None:
        return False
    if state.pending_evaluations or state.evaluation_handles:
        return True
    return plan is not None and _plan_has_unfinished_async_work(plan, state)


def _state_column_names(state: OptimizationState) -> frozenset[str]:
    """Return the columns every collection of a running state carries."""
    present: frozenset[str] | None = None
    for name in ("population", "archive", "pareto_archive"):
        collection = getattr(state, name, None)
        schema = getattr(collection, "schema", None)
        if schema is None:
            continue
        names = frozenset(schema)
        present = names if present is None else present & names
    return present if present is not None else frozenset()


def _required_column_names(component: object, problem: object) -> tuple[str, ...]:
    """Return the population column names one component declares, once."""
    getter = getattr(component, "get_required_attrs", None)
    if not callable(getter):
        return ()
    return tuple(attribute.name for attribute in getter(problem))


def _required_attributes(
    component: object, problem: object
) -> tuple[PopulationAttribute, ...]:
    """Return the population attributes one component declares."""
    getter = getattr(component, "get_required_attrs", None)
    if not callable(getter):
        return ()
    return tuple(getter(problem))


def _specification_matches(
    present: PopulationAttribute, required: PopulationAttribute
) -> bool:
    """Return whether a present column attribute satisfies a requirement."""
    try:
        same_dtype = bool(np.dtype(present.dtype) == np.dtype(required.dtype))
    except TypeError:
        return False
    return same_dtype and tuple(present.shape) == tuple(required.shape)


@dataclass
class _StagedCollections:
    """Column writes a transition staged for the three state collections."""

    population: StagedColumns | None = None
    archive: StagedColumns | None = None
    pareto_archive: StagedColumns | None = None

    @classmethod
    def of(cls, state: OptimizationState | None) -> _StagedCollections:
        if state is None:
            return cls()

        def staged(name: str) -> StagedColumns | None:
            collection = getattr(state, name, None)
            return None if collection is None else StagedColumns(collection)

        return cls(
            population=staged("population"),
            archive=staged("archive"),
            pareto_archive=staged("pareto_archive"),
        )

    def items(self) -> tuple[tuple[str, StagedColumns], ...]:
        """Return the collection names that have a staged view, in order."""
        return tuple(
            (name, staged)
            for name in ("population", "archive", "pareto_archive")
            if (staged := getattr(self, name)) is not None
        )

    def targets(self) -> frozenset[str]:
        """Return the names of the collections a transition may write."""
        return frozenset(name for name, _ in self.items())

    def apply(self) -> None:
        """Apply every staged operation to its collection, atomically.

        Every operation of every collection is validated before the first
        mutation, and a collection that already changed is restored when a
        later one still fails.
        """
        for _, staged in self.items():
            staged.validate()
        applied: list[StagedColumns] = []
        try:
            for _, staged in self.items():
                staged.apply()
                applied.append(staged)
        except BaseException:
            for staged in reversed(applied):
                staged.rollback()
            raise

    def restore(self) -> None:
        """Undo applied column operations and drop the staged ones."""
        for _, staged in reversed(self.items()):
            staged.rollback()
            staged.discard()

    def discard(self) -> None:
        """Drop every staged operation without touching a collection."""
        for _, staged in self.items():
            staged.discard()


def _incoming_bindings(
    optimizer: object,
    resolution: _ComponentResolution | None,
    bindings: Mapping[str, object],
) -> dict[str, object]:
    """Return the components of the version being committed, by slot."""
    if resolution is not None:
        return dict(resolution.bindings)
    values = dict(bindings)
    for slot in BINDING_SLOTS:
        values.setdefault(slot, getattr(optimizer, slot, None))
    return values


def _replaced_slots(
    published: Mapping[str, object], incoming: Mapping[str, object]
) -> tuple[str, ...]:
    """Return the slots whose component differs from the published one."""
    return tuple(
        slot for slot in BINDING_SLOTS if published.get(slot) is not incoming.get(slot)
    )


def _transition_entries(
    records: Mapping[str, object],
    replaced: tuple[str, ...],
    incoming: Mapping[str, object],
    problem: object,
) -> tuple[tuple[str, Transition], ...]:
    """Return the transitions of the replaced slots, in slot order.

    A slot without a matching record uses the default column alignment; a
    recorded transition replaces that default for its slot.
    """
    entries: list[tuple[str, Transition]] = []
    for slot in replaced:
        component = incoming.get(slot)
        record = records.get(slot)
        transition = getattr(record, "transition", None)
        if not isinstance(transition, Transition) or (
            getattr(record, "component", None) is not component
        ):
            transition = AlignColumns(problem)  # type: ignore[arg-type]
        entries.append((slot, transition))
    return tuple(entries)


def _combined_transition(
    entries: tuple[tuple[str, Transition], ...],
) -> Transition | None:
    """Return one transition whose declaration covers the whole commit."""
    if not entries:
        return None
    if len(entries) == 1:
        return entries[0][1]
    return Chain(*(transition for _, transition in entries))


def _check_transition(
    transition: Transition | None,
    new_bindings: Mapping[str, object],
    *,
    present: frozenset[str],
    problem: object,
) -> tuple[frozenset[str] | None, tuple[Diagnostic, ...]]:
    """Check a transition's column declaration against the incoming bindings.

    Returns the declared column set (``None`` when the transition declares
    nothing) and the diagnostics the declaration produces: an error per slot
    whose required columns the transition neither finds in the state nor
    declares, and one warning when a transition declares nothing and the check
    therefore cannot be verified.
    """
    declared = (
        frozenset()
        if transition is None
        else transition._declared_columns(
            new_bindings,
            problem,  # type: ignore[arg-type]
        )
    )
    undeclared: list[tuple[str, tuple[str, ...]]] = []
    unverifiable: list[tuple[str, tuple[str, ...]]] = []
    for slot in BINDING_SLOTS:
        component = new_bindings.get(slot)
        if component is None:
            continue
        missing = tuple(
            name
            for name in _required_column_names(component, problem)
            if name not in present
        )
        if not missing:
            continue
        if declared is None:
            unverifiable.append((slot, missing))
            continue
        withheld = tuple(name for name in missing if name not in declared)
        if withheld:
            undeclared.append((slot, withheld))
    diagnostics: list[Diagnostic] = []
    for slot, withheld in undeclared:
        component = new_bindings.get(slot)
        diagnostics.append(_missing_column_diagnostic(slot, component, withheld))
    if unverifiable:
        diagnostics.append(
            Diagnostic(
                severity=Severity.WARNING,
                code="transition_columns_undeclared",
                message=(
                    "Transition declares no produced columns, so the columns "
                    "its replacement requires cannot be verified: "
                    + ", ".join(
                        f"{slot} ({', '.join(names)})" for slot, names in unverifiable
                    )
                    + "."
                ),
                path=ContractPath(components=tuple(slot for slot, _ in unverifiable)),
                resolutions=(
                    "Declare the columns with Transition.produced_columns().",
                    "Add the columns before the replacement.",
                ),
            )
        )
    return declared, tuple(diagnostics)


def _missing_column_diagnostic(
    slot: str, component: object, missing: tuple[str, ...]
) -> Diagnostic:
    """Return the rejection diagnostic for columns a slot still requires."""
    return Diagnostic(
        severity=Severity.ERROR,
        code="missing_population_column",
        message=(
            f"{type(component).__name__} requires population columns "
            "that are missing: " + ", ".join(sorted(missing)) + "."
        ),
        path=ContractPath(components=(slot,)),
        resolutions=(
            "Add the columns with Population.add_columns() before the replacement.",
            "Provide a transition that produces the columns.",
        ),
    )


def _non_checkpointable_components(
    plan: ExecutablePlan,
) -> tuple[tuple[str, str], ...]:
    """Return ``(component id, type name)`` for non-checkpointable components.

    A part that cannot be exported is reported under its parent's id with the
    part name appended.
    """
    rejected: list[tuple[str, str]] = []
    for component_id, contract in dict(plan.contract_snapshots).items():
        component = plan.graph.node_by_id(component_id).component
        held = getattr(component, "component", component)
        if not contract.assumptions["state.checkpointable"]:
            rejected.append((component_id, type(held).__name__))
            continue
        for part in contract.parts:
            if part.contract.assumptions["state.checkpointable"]:
                continue
            part_component = getattr(getattr(component, "stage", None), part.name, None)
            rejected.append(
                (f"{component_id}__{part.name}", type(part_component).__name__)
            )
    return tuple(rejected)


def _checkpointable_diagnostic(component_id: str, type_name: str) -> Diagnostic:
    return Diagnostic(
        severity=Severity.ERROR,
        code="state_not_checkpointable",
        message=(
            "Portable checkpointing requires state.checkpointable=True; "
            f"{type_name} declares state.checkpointable=False."
        ),
        path=ContractPath(components=(component_id,)),
        resolutions=(
            "Write pickle checkpoints or omit the checkpoint path for this run.",
            "Make the component's state exportable and declare "
            "state.checkpointable=True.",
        ),
    )


class _OptimizerEnvironment:
    def __init__(self, optimizer: Any, plan: SequentialPlan | StructuredPlan) -> None:
        self.optimizer = optimizer
        self._publication_deferred = False
        self._pending_recompile = False
        self._last_state: OptimizationState | None = None
        self._entry_checked = False
        self._state_keys_known: bool | None = None
        self._checkpoint_phase = "running"
        self._checkpoint_run_end_callback = False
        self._checkpoint_reject_run_end_callback = False
        self._checkpoint_frames: tuple[RegionFrame, ...] = ()
        self._recorded_bindings = getattr(optimizer, "_last_published_bindings", None)
        resolution = getattr(optimizer, "_component_resolution", None)
        values = (
            dict(resolution.bindings)
            if isinstance(resolution, _ComponentResolution)
            else self._bindings()
        )
        provenance = (
            dict(resolution.origins)
            if isinstance(resolution, _ComponentResolution)
            else {}
        )
        executable = getattr(optimizer, "_executable_plan", None)
        feature_encoder_selection = None
        if (
            isinstance(resolution, _ComponentResolution)
            and _executable_uses_service(executable, "FeatureEncoder")
            and resolution.feature_encoder is not None
        ):
            values["feature_encoder"] = resolution.feature_encoder
            if resolution.feature_encoder_origin is not None:
                origin = resolution.feature_encoder_origin
                provenance["feature_encoder"] = origin
                feature_encoder_selection = (
                    origin.source,
                    origin.reason,
                    origin.signature,
                )
        self._published = _PublishedPlan(
            plan=plan,
            executable=getattr(optimizer, "_executable_plan", None),
            snapshot=_BindingSnapshot(
                version=0,
                values=values,
                provenance=provenance,
                feature_encoder_selection=feature_encoder_selection,
                strategy_values=self._strategy_values(),
                configuration_version=getattr(optimizer, "_configuration_version", 0),
                preset=getattr(optimizer, "_preset", None),
                contracts=_contract_records(values),
            ),
        )
        self._candidate_resolution: _ComponentResolution | None = None
        self._candidate_diagnostics: tuple[Diagnostic, ...] = ()
        self.capabilities = _algorithm_runtime_capabilities(optimizer)
        self._publish_scheduler(plan.plan)

    @property
    def plan(self) -> SequentialPlan | StructuredPlan:
        """The plan published with the active component bindings."""
        return self._published.plan

    @property
    def _active_snapshot(self) -> _BindingSnapshot:
        return self._published.snapshot

    @property
    def active_version(self) -> int:
        """Version of the binding snapshot that the active plan belongs to."""
        return self._active_snapshot.version

    def binding_values(self) -> dict[str, object]:
        """Active snapshot bindings for components resolved by the runtime."""
        return dict(self._active_snapshot.values)

    def defer_recompile(self) -> None:
        """Remember a recompile request until a safe execution boundary."""
        self._pending_recompile = True
        self._publication_deferred = True

    def _remember_state(self, state: OptimizationState) -> None:
        if self._state_keys_known is None:
            resumed = state.__dict__.get("_binding_version") is not None
            self._state_keys_known = not (
                resumed and not isinstance(self._recorded_bindings, _BindingSnapshot)
            )
        self._last_state = state
        self._mark_checkpoint_state(state)

    def _mark_checkpoint_state(
        self,
        state: OptimizationState,
        *,
        phase: str | None = None,
        frames: tuple[RegionFrame, ...] | None = None,
    ) -> None:
        if phase is not None:
            self._checkpoint_phase = phase
        if frames is not None:
            self._checkpoint_frames = tuple(frames)
        state.__dict__["_checkpoint_guard"] = self
        state.__dict__["_binding_version"] = self.active_version

    def execute(
        self, plan: SequentialPlan, state: OptimizationState
    ) -> OptimizationState:
        self._refresh_plan_if_needed(state)
        if not isinstance(self.plan, SequentialPlan):
            raise ValidationError(
                "sequential environment cannot execute a structured plan"
            )
        return self.execute_step(plan, state).state

    def execute_step(
        self, plan: SequentialPlan, state: OptimizationState
    ) -> _ExecutionOutcome:
        self._last_state = state
        self._mark_checkpoint_state(state, phase="running")
        self._refresh_plan_if_needed(state)
        if not isinstance(self.plan, SequentialPlan):
            raise ValidationError(
                "sequential environment cannot execute a structured plan"
            )
        outcome = _execute_with_metadata(
            self.plan,
            state,
            dispatch=self.dispatch,
            observe=_record_history_observation,
        )
        self._last_state = outcome.state
        return outcome

    def _bindings(self) -> dict[str, object]:
        bindings = {
            name: getattr(self.optimizer, name, None) for name in _BINDING_SLOTS
        }
        executable = getattr(
            getattr(self, "_published", None), "executable", None
        ) or getattr(self.optimizer, "_executable_plan", None)
        if _executable_uses_service(executable, "FeatureEncoder"):
            problem = getattr(self.optimizer, "problem", None)
            space = getattr(problem, "space", None)
            services = getattr(space, "services", None)
            getter = getattr(services, "get", None)
            explicit = getter("FeatureEncoder") if callable(getter) else None
            cached = getattr(self.optimizer, "_feature_encoder_default", None)
            if explicit is not None:
                bindings["feature_encoder"] = explicit
            elif isinstance(cached, tuple) and len(cached) == 3 and cached[0] is space:
                bindings["feature_encoder"] = cached[1]
        return bindings

    def _feature_encoder_selection(self) -> tuple[str, str, str] | None:
        executable = getattr(
            getattr(self, "_published", None), "executable", None
        ) or getattr(self.optimizer, "_executable_plan", None)
        if not _executable_uses_service(executable, "FeatureEncoder"):
            return None
        problem = getattr(self.optimizer, "problem", None)
        space = getattr(problem, "space", None)
        explicit, origin = _resolve_explicit_feature_encoder(self.optimizer, space)
        if explicit is not None:
            assert origin is not None
            return (
                origin.source,
                origin.reason,
                origin.signature,
            )
        cached = getattr(self.optimizer, "_feature_encoder_default", None)
        if (
            isinstance(cached, tuple)
            and len(cached) == 3
            and cached[0] is space
            and cached[1] is not None
        ):
            encoder, reason = cached[1], cached[2]
            return (
                "framework default",
                reason,
                f"{id(space)}:{_feature_encoder_signature(encoder)}",
            )
        return None

    def _strategy_values(self) -> tuple[tuple[str, str], ...]:
        strategy = getattr(self.optimizer, "strategy", None)
        if strategy is None:
            return ()
        return tuple(
            (name, repr(value))
            for name, value in vars(strategy).items()
            if name != "pipeline"
        )

    def _refresh_plan_if_needed(self, state: OptimizationState | None = None) -> None:
        """Commit point: publish a new binding version once, or defer it.

        A replacement detected between commit points is applied only here;
        during one flow execution the active version stays untouched.  A
        published slot whose contract differs from the one recorded at its
        publication counts as replaced too, so a part swapped inside a
        component whose slot still holds the same object takes this path.
        Slots that declare no ``contract()`` (``comparator``, ``cbmanager``,
        ``feature_encoder``) are never compared, and a contract holds no
        parameter values, so writing one leaves the version and the plan
        alone.  A rejection restores slots, not parts: an in-place change
        stays installed and a later commit point rejects it again until the
        part is put back.  Reading every contract costs roughly a millisecond
        per commit point for a GA configuration and half of that for a PSO
        one, measured against a cheap objective; next to an expensive
        evaluation that is negligible.
        """
        self._check_entry_state()
        bindings = self._bindings()
        strategy_values = self._strategy_values()
        unchanged = (
            not self._pending_recompile
            and self._active_snapshot.unchanged_by(
                bindings,
                strategy_values,
                self._feature_encoder_selection(),
                getattr(self.optimizer, "_preset", None),
            )
            and self._active_snapshot.contracts_match(_contract_records(bindings))
        )
        if unchanged:
            # A swap that was deferred while its evaluations ran may resolve
            # back to the active bindings; a matching commit point then ends
            # the deferral, otherwise pending-collection would stall the run.
            self._publication_deferred = False
            self._discard_stale_replacements()
            return
        if self._replacement_blocked(state):
            self._publication_deferred = True
            return
        self._publication_deferred = False
        self._pending_recompile = False
        self._publish(bindings, strategy_values)

    def _check_entry_state(self) -> None:
        """Repair the columns an entry starts with, once per environment.

        The entry has already compiled the plan and validated the same
        bindings, so this adds missing declared columns only: it publishes
        nothing, raises no event, and changes no version.  A resumed state can
        name bindings other than the ones it was written under, which the next
        replacement still aligns with its own publication.
        """
        if self._entry_checked:
            return
        self._entry_checked = True
        state = self._last_state
        problem = getattr(self.optimizer, "problem", None)
        if state is None or problem is None:
            return
        declared: dict[str, PopulationAttribute] = {}
        for slot in BINDING_SLOTS:
            component = self._active_snapshot.values.get(slot)
            if component is None:
                continue
            for attribute in _required_attributes(component, problem):
                declared.setdefault(attribute.name, attribute)
        if not declared:
            return
        for name in ("population", "archive", "pareto_archive"):
            collection = getattr(state, name, None)
            schema = getattr(collection, "schema", None)
            if collection is None or schema is None:
                continue
            additions = [
                attribute
                for column, attribute in declared.items()
                if column not in schema and (column != "x" or "x" in schema)
            ]
            if additions:
                collection.add_columns(additions)

    def _publication_state_keys(
        self,
    ) -> tuple[frozenset[StateKey] | None, frozenset[StateKey] | None]:
        """Return the outgoing plan's declared state keys, or ``None``."""
        if not self._state_keys_known:
            return None, None
        snapshot = self._active_snapshot
        if snapshot.state_reads is not None and snapshot.state_writes is not None:
            return snapshot.state_reads, snapshot.state_writes
        return _plan_declared_state_keys(self._published.plan) or (None, None)

    def _stale_state_keys(
        self,
        outgoing_writes: frozenset[StateKey] | None,
        incoming: tuple[frozenset[StateKey], frozenset[StateKey]] | None,
    ) -> tuple[StateKey, ...]:
        """Return the named state values a replacement invalidates."""
        if outgoing_writes is None or incoming is None:
            if outgoing_writes is not None or incoming is not None:
                logger.warning(
                    "Named state keys cannot be enumerated for this replacement, "
                    "so no named state value is invalidated."
                )
            return ()
        incoming_reads, incoming_writes = incoming
        stale = outgoing_writes & incoming_reads - incoming_writes
        return tuple(sorted(stale, key=lambda key: (key.namespace, key.name)))

    def _replacement_blocked(self, state: OptimizationState | None) -> bool:
        """Outstanding asynchronous work under the active version."""
        scheduler = getattr(self.optimizer, "async_evaluation_scheduler", None)
        return _replacement_blocked(scheduler, self.plan, state)

    def _adopt_published_bindings(self, state: OptimizationState) -> None:
        """Start a resumed run from the bindings the optimizer last published.

        A state written under the recorded binding version belongs to those
        bindings, whatever the live slots say; a state under any other version
        starts from the live slots.  The record then follows this environment,
        so an entry that publishes nothing still leaves the bindings it
        resumed with.
        """
        record = self._recorded_bindings
        if (
            isinstance(record, _BindingSnapshot)
            and state.__dict__.get("_binding_version") == record.version
        ):
            self._published = _PublishedPlan(
                plan=self._published.plan,
                executable=self._published.executable,
                snapshot=record,
            )
        self._remember_published_bindings()

    def _remember_published_bindings(self) -> None:
        object.__setattr__(
            self.optimizer, "_last_published_bindings", self._published.snapshot
        )

    def _publish(
        self,
        bindings: dict[str, object],
        strategy_values: tuple[tuple[str, str], ...],
    ) -> None:
        """Prepare a candidate binding set and publish it, or reject it.

        ``_capture_commit_memento`` records the terms this commit may change
        before the first mutation, and every failure after it -- in
        preparation, in a transition, in post-validation, in the staged apply,
        or inside publication -- goes through :meth:`_reject_commit`, which
        restores those terms.  Region history for the candidate plan is created
        only after the transitions and post-validation accepted it, the staged
        column writes are applied only after post-validation, and the child
        seed a randomized comparator needs is drawn only once publication
        succeeded, so a rejected candidate consumes no seed of the run's
        generator.
        """
        self._candidate_resolution = None
        self._candidate_diagnostics = ()
        records = self._consume_replacements()
        staged = _StagedCollections.of(self._last_state)
        memento = self._capture_commit_memento(bindings, staged)
        try:
            executable, rebuilt = self._compile_current_plan(bindings)
            resolution = self._candidate_resolution
            self._gate(executable, resolution)
            incoming = _incoming_bindings(self.optimizer, resolution, bindings)
            replaced = _replaced_slots(memento.snapshot_values, incoming)
            entries = _transition_entries(
                records,
                replaced,
                incoming,
                getattr(self.optimizer, "problem", None),
            )
            transition = _combined_transition(entries)
            outgoing_writes = self._publication_state_keys()[1]
            incoming_state_keys = _plan_declared_state_keys(rebuilt)
            stale_state_keys = self._stale_state_keys(
                outgoing_writes, incoming_state_keys
            )
            self._check_candidate_state(executable, resolution, transition, incoming)
        except BaseException as error:
            self._reject_commit(error, memento)
        try:
            self._run_transitions(staged, entries, incoming, replaced)
        except BaseException as error:
            self._reject_commit(error, memento, report=False)
        try:
            self._check_staged_changes(staged, incoming)
            self._prepare_region_history(rebuilt)
            staged.apply()
            incoming_reads, incoming_writes = (
                incoming_state_keys if incoming_state_keys is not None else (None, None)
            )
            self._publish_prepared(
                executable,
                rebuilt,
                resolution,
                bindings,
                strategy_values,
                state_reads=incoming_reads,
                state_writes=incoming_writes,
            )
            if self._last_state is not None:
                for key in stale_state_keys:
                    self._last_state.set_state(key, None)
        except BaseException as error:
            self._reject_commit(error, memento)
        self._assign_published_comparator_rng()

    def _publish_prepared(
        self,
        executable: ExecutablePlan,
        rebuilt: SequentialPlan | StructuredPlan,
        resolution: _ComponentResolution | None,
        bindings: dict[str, object],
        strategy_values: tuple[tuple[str, str], ...],
        *,
        state_reads: frozenset[StateKey] | None = None,
        state_writes: frozenset[StateKey] | None = None,
    ) -> None:
        consumes_feature_encoder = _executable_uses_service(
            executable, "FeatureEncoder"
        )
        if resolution is not None:
            publish_resolution = getattr(
                self.optimizer, "_publish_component_resolution", None
            )
            if callable(publish_resolution):
                publish_resolution(
                    resolution,
                    feature_encoder_consumed=consumes_feature_encoder,
                )
            values = dict(resolution.bindings)
            provenance = dict(resolution.origins)
        else:
            values = dict(bindings)
            provenance = dict(self._active_snapshot.provenance)
        if (
            resolution is not None
            and consumes_feature_encoder
            and resolution.feature_encoder is not None
        ):
            values["feature_encoder"] = resolution.feature_encoder
            if resolution.feature_encoder_origin is not None:
                provenance["feature_encoder"] = resolution.feature_encoder_origin
        elif not consumes_feature_encoder:
            values.pop("feature_encoder", None)
            provenance.pop("feature_encoder", None)
        if hasattr(self.optimizer, "_executable_plan"):
            self.optimizer._executable_plan = executable
        version = self.active_version + 1
        self._published = _PublishedPlan(
            plan=rebuilt,
            executable=executable,
            snapshot=_BindingSnapshot(
                version=version,
                values=values,
                provenance=provenance,
                feature_encoder_selection=(
                    None
                    if not consumes_feature_encoder
                    or resolution is None
                    or resolution.feature_encoder_origin is None
                    else (
                        resolution.feature_encoder_origin.source,
                        resolution.feature_encoder_origin.reason,
                        resolution.feature_encoder_origin.signature,
                    )
                ),
                strategy_values=strategy_values,
                configuration_version=getattr(
                    self.optimizer, "_configuration_version", 0
                ),
                preset=getattr(self.optimizer, "_preset", None),
                contracts=_contract_records(values),
                state_reads=state_reads,
                state_writes=state_writes,
            ),
        )
        self._remember_published_bindings()
        if self._last_state is not None:
            current_services = self._last_state.__dict__.get("_compiled_services")
            if consumes_feature_encoder and resolution is not None:
                if resolution.feature_encoder is not None:
                    if not isinstance(current_services, dict):
                        current_services = {}
                        self._last_state.__dict__["_compiled_services"] = (
                            current_services
                        )
                    current_services["FeatureEncoder"] = resolution.feature_encoder
            elif isinstance(current_services, dict):
                current_services.pop("FeatureEncoder", None)
            self._last_state.__dict__["_binding_version"] = version
        self._candidate_resolution = None
        self._publish_scheduler(executable)
        if self._last_state is not None:
            comparator = cast(Any, self._active_snapshot.values.get("comparator"))
            if comparator is not None and self._last_state.comparator is not comparator:
                self._last_state.comparator = comparator
                prepare_state_comparator(self._last_state)

    def _check_candidate_state(
        self,
        executable: ExecutablePlan,
        resolution: _ComponentResolution | None,
        transition: Transition | None,
        incoming: Mapping[str, object],
    ) -> None:
        """Run the checks that need the running state or the run's format."""
        self._check_candidate_columns(incoming, transition)
        self._check_candidate_checkpointable(executable)
        if self._candidate_diagnostics:
            raise ConfigurationError(
                "Component replacement rejected at the commit point:\n"
                + "\n".join(f"  - {item}" for item in self._candidate_diagnostics)
            )

    def _check_candidate_columns(
        self, incoming: Mapping[str, object], transition: Transition | None
    ) -> None:
        """Reject candidates whose declared columns the running state lacks.

        Columns the transition declares it produces are left to
        post-validation, which checks the schema the staged writes would leave
        behind.  A transition that declares nothing cannot be verified, so a
        missing column only warns there.  The check needs a state to compare
        against, so a commit point without one (a recompile from outside a run)
        skips it.
        """
        state = self._last_state
        problem = getattr(self.optimizer, "problem", None)
        if state is None or problem is None:
            return
        _, diagnostics = _check_transition(
            transition,
            incoming,
            present=_state_column_names(state),
            problem=problem,
        )
        for diagnostic in diagnostics:
            if diagnostic.severity is Severity.ERROR:
                self._candidate_diagnostics += (diagnostic,)
            else:
                logger.warning("%s", diagnostic.message)

    def _run_transitions(
        self,
        staged: _StagedCollections,
        entries: tuple[tuple[str, Transition], ...],
        incoming: Mapping[str, object],
        replaced: tuple[str, ...],
    ) -> None:
        """Run the recorded transitions of this commit point, in slot order."""
        if not entries:
            return
        state = self._last_state
        tctx = TransitionContext(
            state=state,
            problem=getattr(self.optimizer, "problem", None),
            rng=None if state is None else state.rng,
            outgoing=dict(self._active_snapshot.values),
            incoming=dict(incoming),
            replaced_slots=replaced,
            write_targets=staged.targets(),
            population=staged.population,
            archive=staged.archive,
            pareto_archive=staged.pareto_archive,
        )
        for slot, transition in entries:
            for requested in _iter_applied(transition, tctx):
                raise ConfigurationError(
                    f"Transition {type(transition).__name__} of slot {slot!r} "
                    f"yielded {requested!r}; evaluation requested by a "
                    "transition is not available at the commit point."
                )

    def _assign_published_comparator_rng(self) -> None:
        """Draw the child seed of a randomized comparator once it is published."""
        state = self._last_state
        if state is not None:
            assign_comparator_rng(state)

    def _prepare_region_history(self, rebuilt: SequentialPlan | StructuredPlan) -> None:
        """Create the region history an accepted candidate plan needs."""
        state = self._last_state
        if state is None or not _has_state_key(state, RUNTIME_REGION_COUNTS):
            return
        region_ids = (
            _iteration_region_ids(rebuilt.graph)
            if isinstance(rebuilt, StructuredPlan)
            else ()
        )
        _ensure_region_history(state, region_ids)

    def _check_staged_changes(
        self, staged: _StagedCollections, incoming: Mapping[str, object]
    ) -> None:
        """Reject a commit whose staged writes leave a required column out.

        Runs after the transitions and before the staged writes are applied, so
        a rejected commit leaves every collection unchanged.  Columns a
        transition adds are checked for dtype and shape as well.
        """
        problem = getattr(self.optimizer, "problem", None)
        if self._last_state is None or problem is None:
            return
        missing: dict[str, list[str]] = {}
        mismatched: dict[tuple[str, str], Diagnostic] = {}
        for _, collection in staged.items():
            schema = collection.planned_schema()
            added = collection.added_columns()
            for slot in BINDING_SLOTS:
                component = incoming.get(slot)
                if component is None:
                    continue
                for attribute in _required_attributes(component, problem):
                    present = schema.get(attribute.name)
                    if present is None:
                        names = missing.setdefault(slot, [])
                        if attribute.name not in names:
                            names.append(attribute.name)
                        continue
                    key = (slot, attribute.name)
                    if attribute.name not in added or key in mismatched:
                        continue
                    if _specification_matches(present, attribute):
                        continue
                    mismatched[key] = Diagnostic(
                        severity=Severity.ERROR,
                        code="population_column_specification_mismatch",
                        message=(
                            f"{type(component).__name__} requires column "
                            f"{attribute.name!r} with dtype "
                            f"{np.dtype(attribute.dtype)} and shape "
                            f"{tuple(attribute.shape)}, but the produced column "
                            f"has dtype {np.dtype(present.dtype)} and shape "
                            f"{tuple(present.shape)}."
                        ),
                        path=ContractPath(components=(slot,)),
                        resolutions=(
                            "Produce the column with the declared dtype and shape.",
                        ),
                    )
        for slot, names in missing.items():
            self._candidate_diagnostics += (
                _missing_column_diagnostic(slot, incoming.get(slot), tuple(names)),
            )
        self._candidate_diagnostics += tuple(mismatched.values())
        if self._candidate_diagnostics:
            raise ConfigurationError(
                "Component replacement rejected at the commit point:\n"
                + "\n".join(f"  - {item}" for item in self._candidate_diagnostics)
            )

    def _check_candidate_checkpointable(self, executable: ExecutablePlan) -> None:
        """Reject candidates whose state a portable checkpoint cannot carry."""
        if not getattr(self.optimizer, "_portable_checkpointing", False):
            return
        rejected = _non_checkpointable_components(executable)
        if not rejected:
            return
        self._candidate_diagnostics += tuple(
            _checkpointable_diagnostic(component_id, type_name)
            for component_id, type_name in rejected
        )
        validator = getattr(self.optimizer, "_validate_checkpointable", None)
        if callable(validator):
            validator(plan=executable)

    def _rejected_changes(
        self, published: Mapping[str, object], origins: Mapping[str, object]
    ) -> tuple[RejectedChange, ...]:
        """Return every slot whose component differs from the published one."""
        current = self._bindings()
        names = list(_BINDING_SLOTS)
        names.extend(sorted((set(published) | set(current)) - set(_BINDING_SLOTS)))
        changes: list[RejectedChange] = []
        for name in names:
            previous = published.get(name)
            candidate = current.get(name)
            if previous is candidate:
                continue
            origin = origins.get(name)
            changes.append(
                RejectedChange(
                    slot=name,
                    previous=previous,
                    candidate=candidate,
                    source=getattr(origin, "source", None),
                )
            )
        return tuple(changes)

    def _consume_replacements(self) -> dict[str, object]:
        """Take the replacements recorded since the last commit point."""
        consume = getattr(self.optimizer, "_consume_replacements", None)
        if not callable(consume):
            return {}
        return dict(consume())

    def _discard_stale_replacements(self) -> None:
        """Drop recorded replacements that describe the published components."""
        if getattr(self.optimizer, "_replacement_records", None):
            self._consume_replacements()

    def _capture_commit_memento(
        self, bindings: Mapping[str, object], staged: _StagedCollections
    ) -> _CommitMemento:
        """Return every term this commit point may change, before it changes one."""
        optimizer = self.optimizer
        state = self._last_state
        snapshot = self._active_snapshot
        history = None if state is None else getattr(state, "history", None)
        attributes: dict[str, object] = dict(snapshot.values)
        for name in bindings:
            if name not in attributes:
                attributes[name] = getattr(optimizer, name, _UNSET)
        compiled_services: object = _UNSET
        if state is not None:
            compiled_services = state.__dict__.get("_compiled_services", _UNSET)
            if isinstance(compiled_services, dict):
                compiled_services = dict(compiled_services)
        return _CommitMemento(
            attributes=attributes,
            preset=snapshot.preset,
            configuration_version=snapshot.configuration_version,
            origins=self._published_origins(),
            live_origins=dict(getattr(optimizer, "_component_origins", None) or {}),
            materialized=self._published_materialized_components(),
            feature_encoder_default=self._published_feature_encoder_default(),
            comparator_default=self._published_comparator_default(),
            feedback_builder_explicit=self._published_feedback_builder_explicit(),
            resolution=getattr(optimizer, "_component_resolution", _UNSET),
            executable_plan=getattr(optimizer, "_executable_plan", _UNSET),
            default_resolution=getattr(optimizer, "_default_resolution", _UNSET),
            prepared_diagnostic_cache=getattr(
                optimizer, "_prepared_diagnostic_cache", _UNSET
            ),
            last_published_bindings=getattr(
                optimizer, "_last_published_bindings", _UNSET
            ),
            published=self._published,
            snapshot_values=dict(snapshot.values),
            binding_version=(
                _UNSET
                if state is None
                else state.__dict__.get("_binding_version", _UNSET)
            ),
            store=(
                _UNSET
                if state is None
                else copy.copy(state.__dict__.get("_store", _UNSET))
            ),
            custom_state=(
                _UNSET if state is None else state.__dict__.get("_custom_state", _UNSET)
            ),
            comparator=_UNSET if state is None else state.comparator,
            compiled_services=compiled_services,
            history=history,
            region_ids=(
                () if history is None else tuple(getattr(history, "_region_ids", ()))
            ),
            history_columns=(
                {}
                if history is None
                else {
                    name: frozenset(columns)
                    for name, columns in history._columns.items()
                }
            ),
            scheduler=self._scheduler_state(),
            staged=staged,
            candidate_resolution=self._candidate_resolution,
            candidate_diagnostics=self._candidate_diagnostics,
            publication_deferred=self._publication_deferred,
            pending_recompile=self._pending_recompile,
        )

    def _restore_commit_memento(self, memento: _CommitMemento) -> None:
        """Put back every term a failed commit point changed."""
        optimizer = self.optimizer
        state = self._last_state
        memento.staged.restore()
        for name, value in memento.attributes.items():
            if getattr(optimizer, name, _UNSET) is not value:
                _restore_attribute(optimizer, name, value)
        for name, value in (
            ("_preset", memento.preset),
            ("_configuration_version", memento.configuration_version),
            ("_component_origins", dict(memento.origins)),
            ("_materialized_components", dict(memento.materialized)),
            ("_feature_encoder_default", memento.feature_encoder_default),
            ("_comparator_default", memento.comparator_default),
            ("feedback_builder_explicit", memento.feedback_builder_explicit),
            ("_component_resolution", memento.resolution),
            ("_executable_plan", memento.executable_plan),
            ("_default_resolution", memento.default_resolution),
            ("_prepared_diagnostic_cache", memento.prepared_diagnostic_cache),
            ("_last_published_bindings", memento.last_published_bindings),
        ):
            _restore_attribute(optimizer, name, value)
        self._published = cast(_PublishedPlan, memento.published)
        if state is not None:
            if memento.binding_version is _UNSET:
                state.__dict__.pop("_binding_version", None)
            else:
                state.__dict__["_binding_version"] = memento.binding_version
            if memento.store is not _UNSET:
                object.__setattr__(state, "_store", copy.copy(memento.store))
            if memento.custom_state is _UNSET:
                state.__dict__.pop("_custom_state", None)
            else:
                object.__setattr__(state, "_custom_state", memento.custom_state)
            if memento.comparator is not _UNSET:
                state.comparator = cast(Any, memento.comparator)
            if memento.compiled_services is _UNSET:
                state.__dict__.pop("_compiled_services", None)
            elif isinstance(memento.compiled_services, dict):
                object.__setattr__(
                    state, "_compiled_services", dict(memento.compiled_services)
                )
            self._restore_region_history(memento)
        self._restore_scheduler_state(memento.scheduler)
        self._candidate_resolution = cast(
            "_ComponentResolution | None", memento.candidate_resolution
        )
        self._candidate_diagnostics = memento.candidate_diagnostics
        self._publication_deferred = memento.publication_deferred
        self._pending_recompile = memento.pending_recompile

    def _restore_region_history(self, memento: _CommitMemento) -> None:
        """Remove the region history the candidate plan added."""
        state = self._last_state
        if state is None:
            return
        history = cast(Any, memento.history)
        if history is not None and state.history is not history:
            state.history = history
        if history is None:
            return
        history._region_ids = memento.region_ids
        for name, saved in memento.history_columns.items():
            stored = history._columns.get(name)
            if not isinstance(stored, dict):
                continue
            for column in [column for column in stored if column not in saved]:
                del stored[column]

    def _restore_scheduler_state(self, scheduler_state: Mapping[str, object]) -> None:
        """Put the scheduler attributes publication writes back."""
        scheduler = getattr(self.optimizer, "async_evaluation_scheduler", None)
        if scheduler is None:
            return
        for name, value in scheduler_state.items():
            with contextlib.suppress(AttributeError):
                object.__setattr__(scheduler, name, value)

    def _scheduler_state(self) -> dict[str, object]:
        """Return the scheduler terms publication serves from the snapshot."""
        scheduler = getattr(self.optimizer, "async_evaluation_scheduler", None)
        if scheduler is None:
            return {}
        return {
            name: getattr(scheduler, name, None)
            for name in (
                "algorithm",
                "callback_manager",
                "feedback_builder",
                "_feedback_accumulator",
                "_feedback_accumulator_state",
            )
        }

    def _reject_commit(
        self,
        error: BaseException,
        memento: _CommitMemento,
        *,
        report: bool = True,
    ) -> None:
        """Restore the commit memento, then report the rejection.

        Every failure after the memento was taken restores the same terms,
        including one raised inside publication, so the optimizer, the run
        state, the collections, the region history, and the scheduler describe
        the last published configuration again.  A plain
        :class:`ConfigurationError` becomes a :class:`ReplacementRejectedError`
        with the rejected slots and the diagnostics that caused it; any other
        exception, and any failure reported with ``report=False``, keeps its
        type and message.
        """
        changes = self._rejected_changes(memento.snapshot_values, memento.live_origins)
        diagnostics = self._candidate_diagnostics
        self._restore_commit_memento(memento)
        if report and type(error) is ConfigurationError:
            raise ReplacementRejectedError(
                changes, diagnostics, version=self.active_version
            ) from error
        raise error

    def _published_feature_encoder_default(
        self,
    ) -> tuple[object, object, str] | None:
        """Return the cached default encoder the published snapshot selected.

        A framework-default encoder is chosen once and cached on the optimizer;
        the selection a commit point compares is read back from that cache when
        no encoder is registered on the search space.  Rebuilding the cache
        from the published snapshot keeps the selection equal to the one the
        snapshot recorded.
        """
        selection = self._active_snapshot.feature_encoder_selection
        encoder = self._active_snapshot.values.get("feature_encoder")
        space = getattr(getattr(self.optimizer, "problem", None), "space", None)
        if selection is None or selection[0] != "framework default":
            return None
        if encoder is None or space is None:
            return None
        return (space, encoder, selection[1])

    def _published_origins(self) -> dict[str, _ComponentOrigin]:
        """Return the component origins the published snapshot recorded.

        Setters and direct assignments rewrite the optimizer's origin entries
        in place before a commit point runs, so the dictionary captured at the
        start of preparation already describes the rejected components.  The
        snapshot's provenance holds the origins the publication wrote, and the
        publication keeps a ``feature_encoder`` entry only for a persisted
        encoder.
        """
        return {
            name: origin
            for name, origin in self._active_snapshot.provenance.items()
            if name != "feature_encoder" or origin.source == "persisted"
        }

    def _published_materialized_components(self) -> dict[str, object]:
        """Return the plan-owned components the published snapshot recorded."""
        return {
            name: origin.value
            for name, origin in self._active_snapshot.provenance.items()
            if name != "feature_encoder"
            and origin.source not in {"setter", "direct assignment"}
        }

    def _published_comparator_default(self) -> object:
        """Return the comparator default the published selection implies."""
        origin = self._active_snapshot.provenance.get("comparator")
        if origin is None or origin.source != "framework default":
            return None
        return (
            getattr(self.optimizer, "problem", None),
            self._active_snapshot.values.get("comparator"),
        )

    def _published_feedback_builder_explicit(self) -> object:
        """Return whether the published feedback builder came from an explicit slot."""
        origin = self._active_snapshot.provenance.get("feedback_builder")
        return getattr(origin, "source", None) in {
            "setter",
            "direct assignment",
            "user preset",
        }

    def _publish_scheduler(self, executable: ExecutablePlan) -> None:
        """Serve the scheduler from the active snapshot, never live reads."""
        scheduler = getattr(self.optimizer, "async_evaluation_scheduler", None)
        if scheduler is None:
            return
        if any(
            getattr(insertion, "adapter_name", None) == "feedback_accumulator"
            for insertion in executable.inserted_adapters
        ):
            scheduler.enable_feedback_accumulator()
        values = self._active_snapshot.values
        updated = {
            "algorithm": values.get("algorithm"),
            "callback_manager": values.get("cbmanager"),
        }
        if getattr(scheduler, "_feedback_accumulator", None) is not None:
            feedback_builder = _structured_true_feedback_builder(self.plan)
            if feedback_builder is None:
                feedback_builder = values.get("feedback_builder")
            if feedback_builder is None:
                feedback_builder = getattr(
                    values.get("strategy"), "feedback_builder", None
                )
            updated["feedback_builder"] = feedback_builder
        for name, value in updated.items():
            with contextlib.suppress(AttributeError):
                setattr(scheduler, name, value)

    def _gate(
        self,
        executable: ExecutablePlan,
        resolution: _ComponentResolution | None = None,
    ) -> None:
        """Run the initial-run validation gate on a candidate plan.

        Raises before publication so the previous plan and active version
        stay usable when a replacement does not compile.
        """
        optimizer = self.optimizer
        validate = getattr(optimizer, "validate", None)
        if not callable(validate):
            return
        has_plan = hasattr(optimizer, "_executable_plan")
        previous = getattr(optimizer, "_executable_plan", None)
        saved_components = {}
        saved_resolution = getattr(optimizer, "_component_resolution", None)
        saved_defaults = getattr(optimizer, "_default_resolution", None)
        saved_origins = getattr(optimizer, "_component_origins", None)
        saved_materialized = getattr(optimizer, "_materialized_components", None)
        saved_feedback_explicit = getattr(optimizer, "feedback_builder_explicit", False)
        has_diagnostic_cache = hasattr(optimizer, "_prepared_diagnostic_cache")
        saved_diagnostic_cache = getattr(optimizer, "_prepared_diagnostic_cache", None)
        if resolution is not None and hasattr(
            optimizer, "_publish_component_resolution"
        ):
            saved_components = {
                name: getattr(optimizer, name, None) for name in resolution.components
            }
            for name, value in resolution.components.items():
                object.__setattr__(optimizer, name, value)
            object.__setattr__(optimizer, "_component_resolution", resolution)
            object.__setattr__(optimizer, "_default_resolution", resolution.defaults)
        if has_plan:
            optimizer._executable_plan = executable
        try:
            signature_builder = getattr(optimizer, "_diagnostic_plan_signature", None)
            if resolution is not None and callable(signature_builder):
                optimizer._prepared_diagnostic_cache = (
                    signature_builder(resolution, require_initializer=False),
                    executable.diagnostics,
                )
            issues = validate(require_initializer=False)
        finally:
            if has_plan:
                optimizer._executable_plan = previous
            if has_diagnostic_cache:
                optimizer._prepared_diagnostic_cache = saved_diagnostic_cache
            if resolution is not None and saved_components:
                for name, value in saved_components.items():
                    object.__setattr__(optimizer, name, value)
                object.__setattr__(optimizer, "_component_resolution", saved_resolution)
                object.__setattr__(optimizer, "_default_resolution", saved_defaults)
                object.__setattr__(optimizer, "_component_origins", saved_origins)
                object.__setattr__(
                    optimizer, "_materialized_components", saved_materialized
                )
                object.__setattr__(
                    optimizer, "feedback_builder_explicit", saved_feedback_explicit
                )
        if issues:
            reported = tuple(
                item
                for item in getattr(optimizer, "_last_contract_diagnostics", ())
                if item.severity is Severity.ERROR
            )
            self._candidate_diagnostics = self._candidate_diagnostics + reported
            raise ConfigurationError(
                "Component replacement rejected at the commit point:\n"
                + "\n".join(f"  - {message}" for message in issues)
            )

    def recompile(
        self, plan: SequentialPlan | StructuredPlan
    ) -> SequentialPlan | StructuredPlan:
        if self._replacement_blocked(self._last_state):
            self._pending_recompile = True
            self._publication_deferred = True
            return self.plan
        self._publish(self._bindings(), self._strategy_values())
        self._pending_recompile = False
        return self.plan

    def _compile_current_plan(
        self, bindings: dict[str, object]
    ) -> tuple[ExecutablePlan, SequentialPlan | StructuredPlan]:
        has_resolver = callable(getattr(self.optimizer, "resolve_defaults", None))
        resolution = (
            resolve_components(self.optimizer, configured=bindings)
            if has_resolver
            else None
        )
        strategy = (
            resolution.components.get("strategy")
            if resolution is not None
            else getattr(self.optimizer, "strategy", None)
        )
        build_graph = getattr(strategy, "build_graph", None)
        problem = getattr(self.optimizer, "problem", None)
        if not callable(build_graph) or problem is None:
            raise ValidationError(
                "runtime plan refresh requires strategy.build_graph and problem"
            )
        provider = _SnapshotProviderView(
            self.optimizer,
            dict(resolution.bindings) if resolution is not None else bindings,
            resolution,
        )
        executable = _prepare_plan(
            strategy,
            provider,
            problem=problem,
            offered_runtime_capabilities=default_runtime_registry.offered_capabilities(
                provider
            ),
            initial_state_keys=OPTIMIZATION_STATE_INITIAL_KEYS,
            _verification_enabled=getattr(
                self.optimizer, "_verification_enabled", True
            ),
            _component_resolution=resolution,
            _require_initializer=False,
        )
        validate_plan_contracts(executable)
        rebuilt: SequentialPlan | StructuredPlan
        if isinstance(executable.graph, StructuredGraph):
            rebuilt = StructuredPlan.from_executable_plan(executable)
        else:
            rebuilt = SequentialPlan.from_executable_plan(executable)
        self._candidate_resolution = getattr(
            provider, "_component_resolution", resolution
        )
        return executable, rebuilt

    def execute_async(
        self, plan: SequentialPlan, state: OptimizationState
    ) -> OptimizationState:
        return self.execute_async_step(plan, state).state

    def execute_async_step(
        self, plan: SequentialPlan, state: OptimizationState
    ) -> _ExecutionOutcome:
        """Execute the canonical graph through the async runtime dispatcher.

        The graph contains the synchronous evaluation contract for both
        modes.  Async runtime owns submission and scheduler feedback delivery;
        the graph's synchronous submit/collect/apply/feedback/tell tail is not
        executed, preventing a second tell for the same proposal.
        """
        self._last_state = state
        self._mark_checkpoint_state(state, phase="running")
        scheduler = getattr(self.optimizer, "async_evaluation_scheduler", None)
        if scheduler is None:
            raise ValidationError("Async runtime requires an evaluation scheduler")
        self._refresh_plan_if_needed(state)
        if not isinstance(self.plan, SequentialPlan):
            raise ValidationError("async environment cannot execute a structured plan")
        sequential_plan = self.plan
        nodes = sequential_plan.execution_nodes
        async_index = next(
            (
                index
                for index, node in enumerate(nodes)
                if callable(getattr(node.component, "execute_async", None))
            ),
            None,
        )
        if async_index is None:
            # Custom plans without the evaluation protocol still run normally.
            outcome = _execute_with_metadata(
                sequential_plan,
                state,
                dispatch=self.dispatch,
                observe=_record_history_observation,
            )
            self._last_state = outcome.state
            return outcome
        current = state
        prefix_outcome: _ExecutionOutcome | None = None

        def prefix(value: OptimizationState) -> OptimizationState:
            nonlocal prefix_outcome
            prefix_plan = SequentialPlan(
                plan=sequential_plan.plan,
                nodes=sequential_plan.nodes,
                execution_nodes=sequential_plan.execution_nodes[:async_index],
            )
            prefix_outcome = _execute_with_metadata(
                prefix_plan,
                value,
                dispatch=self.dispatch,
                observe=_record_history_observation,
            )
            if prefix_outcome.terminated or any(
                result.status is not NodeStatus.COMPLETED
                for result in prefix_outcome.node_results
            ):
                raise _ExecutionHaltError(prefix_outcome)
            return prefix_outcome.state

        seam = getattr(nodes[async_index].component, "execute_async")
        bindings = self._active_snapshot.values
        builder = bindings.get("feedback_builder")
        if builder is None:
            builder = getattr(bindings.get("strategy"), "feedback_builder", None)
        kwargs = {
            "scheduler": scheduler,
            "feedback_builder": builder,
            "algorithm": bindings.get("algorithm"),
            "callback_manager": bindings.get("cbmanager"),
            "prefix": prefix,
            "strategy": bindings.get("strategy"),
            "allow_new_proposals": not self._publication_deferred,
        }
        parameters = inspect.signature(seam).parameters
        component = nodes[async_index].component
        kind = "stage" if getattr(component, "stage", None) is not None else "node"
        logger.debug("Runtime %s start: %s", kind, nodes[async_index].component_id)
        try:
            result = seam(
                current,
                **{key: value for key, value in kwargs.items() if key in parameters},
            )
        except _ExecutionHaltError as halt:
            logger.debug(
                "Runtime %s end: %s status=halted",
                kind,
                nodes[async_index].component_id,
            )
            self._last_state = halt.outcome.state
            return halt.outcome
        logger.debug(
            "Runtime %s end: %s status=completed",
            kind,
            nodes[async_index].component_id,
        )
        _record_history_observation(result)
        if prefix_outcome is not None:
            outcome = _ExecutionOutcome(
                state=result,
                node_results=prefix_outcome.node_results,
                executed_node_ids=prefix_outcome.executed_node_ids,
                refused_commands=prefix_outcome.refused_commands,
            )
        else:
            outcome = _ExecutionOutcome(state=result)
        self._last_state = outcome.state
        return outcome

    def reattach(self, state: OptimizationState) -> OptimizationState:
        scheduler = getattr(self.optimizer, "async_evaluation_scheduler", None)
        if scheduler is None:
            raise ValidationError("Async runtime requires an evaluation scheduler")
        return scheduler.reattach(state)

    def poll(self, state: OptimizationState) -> PollResult:
        scheduler = getattr(self.optimizer, "async_evaluation_scheduler", None)
        if scheduler is None:
            raise ValidationError("Async runtime requires an evaluation scheduler")
        return scheduler.poll_result(state, wait=False)

    def can_refill(self, state: OptimizationState) -> bool:
        scheduler = getattr(self.optimizer, "async_evaluation_scheduler", None)
        if scheduler is None:
            raise ValidationError("Async runtime requires an evaluation scheduler")
        return len(state.pending_evaluations) < scheduler.max_pending

    def is_terminated(self, state: OptimizationState) -> bool:
        return self.optimizer.termination.is_terminated(state)

    def dispatch(self, event: Event) -> None:
        boundary = isinstance(event, GenerationEndEvent) and not self._checkpoint_frames
        run_end = isinstance(event, RunEndEvent)
        strategy = getattr(self.optimizer, "strategy", None)
        plan_state = getattr(event.ctx, "evaluation_plan_state", None)
        deferred = bool(plan_state is not None and plan_state.deferred)
        run_end_callback = (
            run_end
            and bool(self._checkpoint_frames)
            and getattr(self.optimizer, "async_evaluation_scheduler", None) is not None
            and _is_generation_based_strategy(strategy)
            and deferred
        )
        self._checkpoint_run_end_callback = (
            run_end
            and bool(self._checkpoint_frames)
            and getattr(self.optimizer, "async_evaluation_scheduler", None) is not None
            and _is_generation_based_strategy(strategy)
            and not deferred
        )
        self._checkpoint_reject_run_end_callback = run_end_callback
        if isinstance(event.ctx, OptimizationState):
            self._mark_checkpoint_state(
                event.ctx,
                phase=(
                    "run-end-callback"
                    if run_end_callback
                    else (
                        "post-run"
                        if run_end
                        else ("boundary" if boundary else "callback")
                    )
                ),
            )
        try:
            self.optimizer.dispatch(event)
        finally:
            self._checkpoint_run_end_callback = False
            self._checkpoint_reject_run_end_callback = False

    def validate_checkpoint(self, state: OptimizationState) -> None:
        from saealib.context import (
            _validate_pending_reattachable,
        )
        from saealib.exceptions import CheckpointError

        if self._checkpoint_reject_run_end_callback:
            raise CheckpointError(
                "cannot save from this asynchronous RunEndEvent callback while "
                "the evaluation plan still has requests waiting to be submitted; "
                "save at a GenerationEndEvent instead, or save the state "
                "returned by run()"
            )
        plan_graph = self.plan.graph if isinstance(self.plan, StructuredPlan) else None
        strategy = getattr(self.optimizer, "strategy", None)
        resumable_frames = (
            _async_resume_frames(
                self._checkpoint_frames,
                state,
                plan_graph=plan_graph,
                strategy=strategy,
                for_checkpoint=True,
            )
            if self._checkpoint_frames
            else None
        )
        if (
            self._checkpoint_frames
            and self._checkpoint_phase != "post-run"
            and resumable_frames is None
        ):
            raise CheckpointError("checkpoint is not supported inside an open region")
        resumable_async = resumable_frames is not None
        if (
            self._checkpoint_phase not in {"boundary", "post-run"}
            and not resumable_async
        ):
            raise CheckpointError(
                "checkpoint is supported only at a step boundary or run end"
            )
        bindings = self._bindings()
        if self._pending_recompile or not self._active_snapshot.unchanged_by(
            bindings,
            self._strategy_values(),
            self._feature_encoder_selection(),
            getattr(self.optimizer, "_preset", None),
        ):
            raise CheckpointError(
                "checkpoint cannot be saved because the optimizer configuration "
                "changed after this state was produced"
            )
        scheduler = getattr(self.optimizer, "async_evaluation_scheduler", None)
        if scheduler is not None:
            _validate_pending_reattachable(state, scheduler.evaluator)

    def finish_generation(self, state: OptimizationState) -> None:
        _increment_step(state)
        handler = state.problem.handler
        handler.on_generation_end(state.gen, state.population)
        state.comparator.eps_cv = state.problem.handler.feasibility_threshold
        manager = getattr(self.optimizer, "surrogate_manager", None)
        if manager is not None:
            manager.on_generation_end(state.gen, state.archive, state)
        record_generation(state)
        self.dispatch(GenerationEndEvent(ctx=state))

    def fatal(self, state: OptimizationState) -> None:
        if state.async_fatal:
            raise EvaluationFatalError(
                str(state.async_fatal.get("reason", "async fatal")), state
            )


def _has_async_evaluation_scheduler(*args: Any, **kwargs: Any) -> bool:
    provider = kwargs.get("provider")
    if provider is None and len(args) > 2:
        provider = args[2]
    return getattr(provider, "async_evaluation_scheduler", None) is not None


@deprecated_function(
    since="0.2.0",
    replacement="Optimizer.iterate() or Optimizer.run()",
    name="execute_strategy_step() with asynchronous evaluation",
    condition=_has_async_evaluation_scheduler,
)
def execute_strategy_step(
    strategy: object, state: OptimizationState, provider: object
) -> OptimizationState:
    """Compatibility step facade with async lifecycle owned by Runtime.

    This supports callers that still invoke ``strategy.step`` directly. When
    requests are in flight, the call blocks until they finish and returns
    without running the step; the next call uses the current components. A
    scheduler without a timeout can make this call wait indefinitely. The
    helper raises ``ConfigurationError`` when plan verification reports an
    ERROR diagnostic, and ``ValidationError`` when the strategy graph cannot
    be prepared or uses an unsupported asynchronous region.
    """
    return _execute_strategy_step(strategy, state, provider)


def _execute_strategy_step(
    strategy: object, state: OptimizationState, provider: object
) -> OptimizationState:
    scheduler = getattr(provider, "async_evaluation_scheduler", None)
    if scheduler is not None and state.pending_evaluations:
        if set(state.evaluation_handles) != set(state.pending_evaluations):
            state = scheduler.reattach(state)
        return scheduler.poll(state, wait=True)
    build_graph = cast(Any, getattr(strategy, "build_graph", None))
    if not callable(build_graph):
        raise ValidationError("strategy requires a callable build_graph")
    resolver = getattr(provider, "resolve_defaults", None)
    if callable(resolver):
        resolver()
    component_resolution = getattr(provider, "_component_resolution", None)

    def live_context() -> tuple[Any, frozenset[str] | None]:
        return state.problem, default_runtime_registry.offered_capabilities(provider)

    plan = _prepare_plan(
        strategy,
        provider,
        problem=None,
        offered_runtime_capabilities=None,
        _context_factory=live_context,
        initial_state_keys=OPTIMIZATION_STATE_INITIAL_KEYS,
        _verification_enabled=getattr(provider, "_verification_enabled", True),
        _component_resolution=(
            component_resolution
            if isinstance(component_resolution, _ComponentResolution)
            else None
        ),
        _require_initializer=False,
    )
    resolution = getattr(provider, "_component_resolution", component_resolution)
    publish_resolution = getattr(provider, "_publish_component_resolution", None)
    if isinstance(resolution, _ComponentResolution) and callable(publish_resolution):
        publish_resolution(
            resolution,
            feature_encoder_consumed=_executable_uses_service(plan, "FeatureEncoder"),
        )
    if plan.diagnostics:
        errors = [item for item in plan.diagnostics if item.severity is Severity.ERROR]
        if errors:
            raise ConfigurationError(
                "strategy step rejected by compiler diagnostics: "
                + "; ".join(item.code for item in errors)
            )
    from saealib.strategies.base import TemplateStrategy, build_pipeline_from_graph

    graph = plan.graph
    if isinstance(graph, StructuredGraph):
        if not isinstance(strategy, TemplateStrategy) or strategy.pipeline is None:
            build_pipeline = getattr(strategy, "build_pipeline", None)
            if not callable(build_pipeline):
                raise ValidationError(
                    "structured strategy graph requires build_pipeline(provider)"
                )
            setattr(strategy, "pipeline", build_pipeline(provider))
    else:
        setattr(strategy, "pipeline", build_pipeline_from_graph(graph))
    if scheduler is None:
        runtime = PipelineRuntime()
        session = runtime.initialize(plan, state)
        return runtime.advance(session).state
    if isinstance(graph, StructuredGraph) and graph.region_nodes:
        raise ValidationError(
            "async strategy steps require a sequential graph execution seam"
        )
    structured = isinstance(graph, StructuredGraph)
    optimizer = SimpleNamespace(
        strategy=strategy,
        problem=state.problem,
        algorithm=getattr(provider, "algorithm", None),
        evaluator=getattr(provider, "evaluator", None),
        feedback_builder=getattr(provider, "feedback_builder", None),
        cbmanager=getattr(provider, "cbmanager", None),
        async_evaluation_scheduler=scheduler,
        dispatch=getattr(provider, "dispatch", None) or (lambda event: None),
        termination=getattr(provider, "termination", None)
        or Termination(lambda ctx: False),
    )
    if structured:
        # Graph-native region-less plans reuse the asynchronous structured
        # runtime branch; the strategy-step seam remains reserved for
        # sequential plans.
        structured_plan = StructuredPlan.from_executable_plan(plan)
        runtime = AsyncPipelineRuntime(
            _OptimizerEnvironment(optimizer, structured_plan)
        )
        session = runtime.initialize(plan, state)
        return runtime.advance(session).state
    sequential = SequentialPlan.from_executable_plan(plan)
    environment = _OptimizerEnvironment(optimizer, sequential)
    return environment.execute_async(sequential, state)


def prepare_state_comparator(state: OptimizationState) -> None:
    """Bind the run state's comparator to the search criteria of this run."""
    comparator = state.comparator
    if getattr(comparator, "direction", None) is None:
        comparator.direction = state.problem.direction
    comparator.eps_cv = state.problem.handler.feasibility_threshold


def assign_comparator_rng(state: OptimizationState) -> None:
    """Give a randomized comparator its niche tie-breaking generator.

    Consumes one child seed of the run's generator.  A commit point calls this
    only after its new configuration is published, and a rejected replacement
    never reaches it, so the parent seed sequence is untouched by a rejected
    candidate and the next accepted randomized comparator draws the same child
    a run without the rejected attempt would.
    """
    comparator = state.comparator
    if isinstance(comparator, NSGA3Comparator) and comparator._rng is None:
        comparator.rng = state.rng.spawn(1)[0]


class PipelineRuntime:
    """Drive synchronous lifecycle and expose one observable step per generation."""

    capabilities = frozenset()

    def __init__(self, environment: LifecycleEnvironment | None = None) -> None:
        self.environment = environment

    def initialize(
        self, plan: ExecutablePlan, state: OptimizationState
    ) -> RuntimeSession:
        """Create a lifecycle session and dispatch the run-start boundary."""
        if not isinstance(plan, ExecutablePlan):
            raise ValidationError(
                "PipelineRuntime.initialize requires an ExecutablePlan"
            )
        if not isinstance(state, OptimizationState):
            raise ValidationError(
                "PipelineRuntime.initialize requires an OptimizationState"
            )
        validate_plan_contracts(plan)
        structured_graph = isinstance(plan.graph, StructuredGraph)
        services: dict[str, object] = {}
        feature_encoders: dict[str, object] = {}
        for node in plan.graph.nodes:
            required = {
                requirement.name
                for port in node.contract.ports.values()
                for spec in (port.inputs + port.outputs)
                for requirement in spec.required_services
            }
            if not required <= set(node.resolved_services):
                missing = ", ".join(sorted(required - set(node.resolved_services)))
                raise ValidationError(
                    f"compiled plan node {node.component_id!r} is missing resolved "
                    f"services: {missing}"
                )
            for name, service in node.resolved_services.items():
                if name == "FeatureEncoder":
                    existing_encoder = feature_encoders.get(name)
                    if existing_encoder is not None and existing_encoder is not service:
                        raise ValidationError(
                            "compiled plan resolves FeatureEncoder to conflicting "
                            "objects"
                        )
                    feature_encoders[name] = service
                if not structured_graph:
                    previous = services.get(name)
                    if previous is not None and previous is not service:
                        raise ValidationError(
                            f"compiled plan resolves service {name!r} to "
                            "conflicting objects"
                        )
                    services[name] = service
        structured = (
            StructuredPlan.from_executable_plan(plan) if structured_graph else None
        )
        services.update(feature_encoders)
        current_services = state.__dict__.get("_compiled_services", {})
        if "FeatureEncoder" in services and "FeatureEncoder" in current_services:
            current_services["FeatureEncoder"] = services.pop("FeatureEncoder")
        else:
            current_services.pop("FeatureEncoder", None)
        if services:
            state.bind_compiled_services(services)
        selected_plan = structured or SequentialPlan.from_executable_plan(plan)
        region_ids = (
            _iteration_region_ids(selected_plan.graph)
            if isinstance(selected_plan, StructuredPlan)
            else ()
        )
        if _has_state_key(state, RUNTIME_REGION_COUNTS):
            _ensure_region_history(state, region_ids)
        capabilities = self.capabilities | frozenset(
            getattr(self.environment, "capabilities", ())
        )
        if not selected_plan.accepts(capabilities):
            missing = selected_plan.required_runtime_capabilities - capabilities
            names = ", ".join(sorted(missing))
            raise ValidationError(
                f"PipelineRuntime lacks required capabilities: {names}"
            )
        if self.environment is None:
            return RuntimeSession(
                plan=selected_plan, state=state, observable=True, generation_open=False
            )
        adopt = getattr(self.environment, "_adopt_published_bindings", None)
        if callable(adopt):
            adopt(state)
        self._prepare_state(state)
        remember_state = getattr(self.environment, "_remember_state", None)
        if callable(remember_state):
            remember_state(state)
        self.environment.dispatch(RunStartEvent(ctx=state))
        mark = getattr(self.environment, "_mark_checkpoint_state", None)
        if callable(mark):
            mark(state, phase="boundary", frames=())
        return RuntimeSession(
            plan=selected_plan,
            state=state,
            observable=True,
            generation_open=bool(state.pending_evaluations),
        )

    def advance(self, session: RuntimeSession) -> RuntimeStep:
        """Advance lifecycle state by one runtime-owned step."""
        if not isinstance(session, RuntimeSession):
            raise ValidationError("PipelineRuntime.advance requires a RuntimeSession")
        if self.environment is not None:
            mark = getattr(self.environment, "_mark_checkpoint_state", None)
            if callable(mark):
                mark(session.state, phase="running", frames=session.frames)
        if isinstance(session.plan, StructuredPlan):
            if self.environment is None:
                metadata = _execute_structured(
                    session.plan,
                    session.state,
                    session.frames,
                    observe=_record_history_observation,
                )
                if metadata.recompile_requested:
                    raise ValidationError(
                        "PipelineRuntime cannot satisfy RequestRecompile "
                        "without a recompile provider"
                    )
                if any(
                    result.status is NodeStatus.FAILED
                    for result in metadata.node_results
                ):
                    raise ValidationError(
                        "structured runtime node reported FAILED status"
                    )
                if metadata.plan_completed and not _has_pending_evaluations(
                    metadata.state
                ):
                    _increment_step(metadata.state)
                run_finished = metadata.plan_completed or metadata.terminated
                next_session = RuntimeSession(
                    plan=session.plan,
                    state=metadata.state,
                    step_index=session.step_index + 1,
                    observable=True,
                    finished=run_finished,
                    frames=metadata.frames,
                )
                return RuntimeStep(
                    state=metadata.state,
                    node_results=metadata.node_results,
                    executed_node_ids=metadata.executed_node_ids,
                    observable=True,
                    finished=run_finished,
                    refused_commands=metadata.refused_commands,
                    session=next_session,
                )

            env = self.environment
            env.fatal(session.state)
            refresh = getattr(env, "_refresh_plan_if_needed", None)
            if callable(refresh) and not session.frames:
                refresh(session.state)
            plan = getattr(env, "plan", session.plan)
            if not isinstance(plan, StructuredPlan):
                raise ValidationError(
                    "structured runtime environment returned a non-structured plan"
                )
            state = session.state
            if env.is_terminated(state):
                env.dispatch(RunEndEvent(ctx=state))
                return self._step(
                    session,
                    state,
                    session.generation_open,
                    finished=True,
                    plan=plan,
                )
            generation_open = session.generation_open
            if not generation_open:
                env.dispatch(GenerationStartEvent(ctx=state))
                generation_open = True
            metadata = _execute_structured(
                plan,
                state,
                session.frames,
                dispatch=env.dispatch,
                observe=_record_history_observation,
            )
            if metadata.recompile_requested:
                if not _structured_recompile_accepted(metadata):
                    defer = getattr(env, "defer_recompile", None)
                    if callable(defer):
                        defer()
                    metadata = _refuse_structured_recompile(metadata)
                else:
                    env.finish_generation(metadata.state)
                    plan = _environment_recompile(cast(RecompileProvider, env), plan)
                    return self._step(
                        session,
                        metadata.state,
                        False,
                        observable=True,
                        metadata=metadata,
                        plan=plan,
                        frames=(),
                    )
            failed = next(
                (
                    result
                    for result in metadata.node_results
                    if result.status is NodeStatus.FAILED
                ),
                None,
            )
            if failed is not None:
                raise ValidationError("structured runtime node reported FAILED status")
            if metadata.terminated:
                env.dispatch(RunEndEvent(ctx=metadata.state))
                return self._step(
                    session,
                    metadata.state,
                    generation_open,
                    finished=True,
                    metadata=metadata,
                    plan=plan,
                )
            if metadata.plan_completed:
                if metadata.state.pending_evaluations:
                    return self._step(
                        session,
                        metadata.state,
                        generation_open,
                        metadata=metadata,
                        plan=plan,
                    )
                env.finish_generation(metadata.state)
                return self._step(
                    session,
                    metadata.state,
                    False,
                    observable=True,
                    finished=False,
                    metadata=metadata,
                    plan=plan,
                )
            if any(
                result.status in {NodeStatus.BLOCKED, NodeStatus.RUNNING}
                for result in metadata.node_results
            ):
                return self._step(
                    session,
                    metadata.state,
                    generation_open,
                    metadata=metadata,
                    plan=plan,
                )
            if metadata.state.pending_evaluations:
                return self._step(
                    session,
                    metadata.state,
                    generation_open,
                    metadata=metadata,
                    plan=plan,
                )
            env.finish_generation(metadata.state)
            return self._step(
                session,
                metadata.state,
                False,
                observable=True,
                metadata=metadata,
                plan=plan,
            )
        if not isinstance(session.plan, SequentialPlan):
            raise ValidationError(
                "PipelineRuntime.advance requires a SequentialPlan session"
            )
        if self.environment is None:
            metadata = _execute_with_metadata(
                session.plan,
                session.state,
                observe=_record_history_observation,
            )
            if metadata.recompile_requested:
                raise ValidationError(
                    "PipelineRuntime cannot satisfy RequestRecompile without "
                    "a recompile provider"
                )
            state = metadata.state
            node_results = metadata.node_results
            if metadata.plan_completed and not _has_pending_evaluations(state):
                _increment_step(state)
            finished = metadata.terminated
            next_session = RuntimeSession(
                plan=session.plan,
                state=state,
                step_index=session.step_index + 1,
                observable=True,
                finished=finished,
            )
            return RuntimeStep(
                state=state,
                node_results=node_results,
                executed_node_ids=metadata.executed_node_ids,
                observable=True,
                finished=finished,
                refused_commands=metadata.refused_commands,
                session=next_session,
            )

        env = cast(SequentialExecutionEnvironment, self.environment)
        env.fatal(session.state)
        state = session.state
        metadata = _ExecutionOutcome(state=state)
        plan = session.plan
        generation_open = session.generation_open
        if state.pending_evaluations:
            metadata = _environment_execute(env, plan, state)
            state = metadata.state
            if metadata.recompile_requested:
                plan = _environment_recompile(cast(RecompileProvider, env), plan)
                return self._step(
                    session, state, generation_open, metadata=metadata, plan=plan
                )
            if metadata.terminated:
                env.dispatch(RunEndEvent(ctx=state))
                return self._step(
                    session,
                    state,
                    generation_open,
                    finished=True,
                    metadata=metadata,
                    plan=plan,
                )
            if state.pending_evaluations:
                return self._step(
                    session, state, generation_open, metadata=metadata, plan=plan
                )
            if generation_open:
                env.finish_generation(state)
                return self._step(
                    session,
                    state,
                    False,
                    observable=True,
                    metadata=metadata,
                    plan=plan,
                )
        if env.is_terminated(state):
            env.dispatch(RunEndEvent(ctx=state))
            return self._step(
                session,
                state,
                generation_open,
                finished=True,
                metadata=metadata,
                plan=plan,
            )
        env.dispatch(GenerationStartEvent(ctx=state))
        generation_open = True
        metadata = _environment_execute(env, plan, state)
        state = metadata.state
        if metadata.recompile_requested:
            plan = _environment_recompile(cast(RecompileProvider, env), plan)
            return self._step(
                session, state, generation_open, metadata=metadata, plan=plan
            )
        if metadata.terminated:
            env.dispatch(RunEndEvent(ctx=state))
            return self._step(
                session,
                state,
                generation_open,
                finished=True,
                metadata=metadata,
                plan=plan,
            )
        if not state.pending_evaluations:
            env.finish_generation(state)
            generation_open = False
            return self._step(
                session,
                state,
                generation_open,
                observable=True,
                metadata=metadata,
                plan=plan,
            )
        return self._step(
            session,
            state,
            generation_open,
            observable=False,
            metadata=metadata,
            plan=plan,
        )

    def _step(
        self,
        session: RuntimeSession,
        state: OptimizationState,
        generation_open: bool,
        *,
        observable: bool = False,
        finished: bool | None = None,
        metadata: _ExecutionOutcome | None = None,
        plan: SequentialPlan | StructuredPlan | None = None,
        frames: tuple[RegionFrame, ...] | None = None,
    ) -> RuntimeStep:
        remember_state = getattr(self.environment, "_remember_state", None)
        if callable(remember_state):
            remember_state(state)
        step_finished = (
            (metadata.terminated if metadata is not None else False)
            if finished is None
            else finished
        )
        next_frames = (
            metadata.frames
            if metadata is not None
            else (frames if frames is not None else session.frames)
        )
        mark_checkpoint = getattr(self.environment, "_mark_checkpoint_state", None)
        if callable(mark_checkpoint):
            phase = "post-run" if step_finished else "running"
            if observable and getattr(self.environment, "_checkpoint_phase", None) in {
                "boundary",
                "post-run",
            }:
                phase = "boundary"
            mark_checkpoint(state, phase=phase, frames=next_frames)
        next_session = RuntimeSession(
            plan=plan or session.plan,
            state=state,
            finished=step_finished,
            observable=observable,
            step_index=session.step_index + 1,
            generation_open=generation_open,
            frames=next_frames,
        )
        return RuntimeStep(
            state=state,
            node_results=metadata.node_results if metadata else (),
            executed_node_ids=metadata.executed_node_ids if metadata else (),
            refused_commands=metadata.refused_commands if metadata else (),
            observable=observable,
            finished=step_finished,
            session=next_session,
        )

    @staticmethod
    def _execute_plan(
        plan: SequentialPlan, state: OptimizationState
    ) -> OptimizationState:
        return _execute_sequential_plan(plan, state)

    @staticmethod
    def _prepare_state(state: OptimizationState) -> None:
        prepare_state_comparator(state)
        assign_comparator_rng(state)


class AsyncPipelineRuntime(PipelineRuntime):
    """Drive structured or sequential asynchronous evaluation lifecycle."""

    capabilities = frozenset({"partial_feedback"})

    def initialize(
        self, plan: ExecutablePlan, state: OptimizationState
    ) -> RuntimeSession:
        """Create a session for a compiled graph plan."""
        return super().initialize(plan, state)

    @staticmethod
    def _async_leaf_kwargs(environment: object) -> dict[str, object]:
        optimizer = getattr(environment, "optimizer", None)
        scheduler = getattr(optimizer, "async_evaluation_scheduler", None)
        if scheduler is None:
            return {}
        snapshot_values = getattr(environment, "binding_values", None)
        if callable(snapshot_values):
            bindings = snapshot_values()
            builder = _structured_true_feedback_builder(
                getattr(environment, "plan", None)
            )
            if builder is None:
                builder = bindings.get("feedback_builder")
            if builder is None:
                builder = getattr(bindings.get("strategy"), "feedback_builder", None)
            return {
                "scheduler": scheduler,
                "feedback_builder": builder,
                "algorithm": bindings.get("algorithm"),
                "callback_manager": bindings.get("cbmanager"),
                "strategy": bindings.get("strategy"),
                "allow_new_proposals": not getattr(
                    environment, "_publication_deferred", False
                ),
            }
        builder = getattr(optimizer, "feedback_builder", None)
        if builder is None:
            strategy = getattr(optimizer, "strategy", None)
            builder = getattr(strategy, "feedback_builder", None)
        return {
            "scheduler": scheduler,
            "feedback_builder": builder,
            "algorithm": getattr(optimizer, "algorithm", None),
            "callback_manager": getattr(optimizer, "cbmanager", None),
            "strategy": getattr(optimizer, "strategy", None),
        }

    def _advance_structured_async(self, session: RuntimeSession) -> RuntimeStep:
        structured_plan = cast(StructuredPlan, session.plan)
        if self.environment is None:
            metadata = _execute_structured(
                structured_plan,
                session.state,
                session.frames,
                observe=_record_history_observation,
                leaf_executor=_structured_async_leaf_executor({}),
                async_control=True,
            )
            if metadata.recompile_requested:
                raise ValidationError(
                    "AsyncPipelineRuntime cannot satisfy RequestRecompile "
                    "without a recompile provider"
                )
            if any(
                result.status is NodeStatus.FAILED for result in metadata.node_results
            ):
                raise ValidationError("structured runtime node reported FAILED status")
            if metadata.plan_completed and not _has_pending_evaluations(metadata.state):
                _increment_step(metadata.state)
            run_finished = metadata.plan_completed or metadata.terminated
            next_session = RuntimeSession(
                plan=session.plan,
                state=metadata.state,
                step_index=session.step_index + 1,
                observable=True,
                finished=run_finished,
                frames=metadata.frames,
            )
            return RuntimeStep(
                state=metadata.state,
                node_results=metadata.node_results,
                executed_node_ids=metadata.executed_node_ids,
                observable=True,
                finished=run_finished,
                refused_commands=metadata.refused_commands,
                session=next_session,
            )

        environment = self.environment
        environment.fatal(session.state)
        state = session.state
        _sync_transient_fields_from_store(state)
        refresh = getattr(environment, "_refresh_plan_if_needed", None)
        if callable(refresh) and not session.frames:
            refresh(state)
        plan = getattr(environment, "plan", session.plan)
        if not isinstance(plan, StructuredPlan):
            raise ValidationError(
                "structured async runtime environment returned a non-structured plan"
            )
        generation_open = session.generation_open
        frames = session.frames
        resume_async_driver = False
        if state.pending_evaluations:
            if set(state.pending_evaluations) != set(state.evaluation_handles):
                reattach = getattr(environment, "reattach", None)
                if callable(reattach):
                    state = reattach(state)
            poll = getattr(environment, "poll", None)
            result = PollResult(state=state, progressed=False)
            if callable(poll):
                result = poll(state)
                state = result.state
                _record_history_observation(state)
                resume_async_driver = not state.pending_evaluations and not (
                    _structured_async_plan_complete(state)
                )
            if state.pending_evaluations:
                if environment.is_terminated(state):
                    if not result.progressed:
                        time.sleep(0.001)
                    return self._step(
                        session,
                        state,
                        generation_open,
                        plan=plan,
                    )
                bindings = getattr(environment, "binding_values", None)
                bindings = bindings() if callable(bindings) else {}
                strategy = bindings.get("strategy")
                can_refill = getattr(environment, "can_refill", None)
                if (
                    not getattr(strategy, "supports_async_refill", False)
                    or not callable(can_refill)
                    or not can_refill(state)
                    or plan.graph.region_nodes
                ):
                    if not result.progressed:
                        time.sleep(0.001)
                    return self._step(
                        session,
                        state,
                        generation_open,
                        plan=plan,
                    )
                plan_state = state.evaluation_plan_state
                plan_ids = (
                    {int(item.request_id) for item in state.evaluation_plan.requests}
                    if state.evaluation_plan is not None
                    else set()
                )
                terminal_ids = (
                    set(plan_state.completed) | set(plan_state.acknowledged)
                    if plan_state is not None
                    else set()
                )
                active_plan_complete = state.evaluation_plan is None or (
                    plan_state is not None and plan_ids <= terminal_ids
                )
                active_plan_progressed = bool(
                    plan_state is not None
                    and (plan_state.completed or plan_state.acknowledged)
                )
                has_deferred_requests = bool(
                    plan_state is not None and plan_state.deferred
                )
                if self._publication_deferred(environment):
                    if has_deferred_requests:
                        resume_async_driver = True
                    else:
                        if not result.progressed:
                            time.sleep(0.001)
                        return self._step(
                            session,
                            state,
                            generation_open,
                            plan=plan,
                        )
                elif active_plan_complete or (
                    active_plan_progressed and not has_deferred_requests
                ):
                    state = state.replace(
                        evaluation_plan=None,
                        evaluation_plan_state=None,
                        evaluation_plan_updates={},
                    )
                    frames = ()
                elif has_deferred_requests:
                    resume_async_driver = True
                else:
                    if not result.progressed:
                        time.sleep(0.001)
                    return self._step(
                        session,
                        state,
                        generation_open,
                        plan=plan,
                    )
        if environment.is_terminated(state):
            # Region-bound flows end the run here even while the last delivery
            # or a blocked frame still owes work.
            if plan.graph.region_nodes:
                environment.dispatch(RunEndEvent(ctx=state))
                return self._step(
                    session,
                    state,
                    generation_open,
                    finished=True,
                    plan=plan,
                )
            if (
                not session.frames
                and not _structured_async_has_unacknowledged_deliveries(state)
            ):
                # Region-less flows close the drained generation boundary
                # before the run end so the final tell and acknowledge belong
                # to this run.
                if generation_open:
                    environment.finish_generation(state)
                else:
                    environment.dispatch(RunEndEvent(ctx=state))
                    return self._step(
                        session,
                        state,
                        generation_open,
                        finished=True,
                        plan=plan,
                    )
                environment.dispatch(RunEndEvent(ctx=state))
                return self._step(
                    session,
                    state,
                    False,
                    observable=True,
                    finished=True,
                    plan=plan,
                )
        if not generation_open:
            environment.dispatch(GenerationStartEvent(ctx=state))
            generation_open = True
        metadata = _execute_structured(
            plan,
            state,
            frames,
            dispatch=environment.dispatch,
            observe=_record_history_observation,
            leaf_executor=_structured_async_leaf_executor(
                self._async_leaf_kwargs(environment)
            ),
            async_control=True,
            resume_async_driver=resume_async_driver,
        )
        if metadata.recompile_requested:
            if not _structured_recompile_accepted(metadata):
                defer = getattr(environment, "defer_recompile", None)
                if callable(defer):
                    defer()
                metadata = _refuse_structured_recompile(metadata)
            else:
                environment.finish_generation(metadata.state)
                plan = _environment_recompile(
                    cast(RecompileProvider, environment), plan
                )
                return self._step(
                    session,
                    metadata.state,
                    False,
                    observable=True,
                    metadata=metadata,
                    plan=plan,
                    frames=(),
                )
        if any(result.status is NodeStatus.FAILED for result in metadata.node_results):
            raise ValidationError("structured runtime node reported FAILED status")
        if metadata.terminated:
            environment.dispatch(RunEndEvent(ctx=metadata.state))
            return self._step(
                session,
                metadata.state,
                generation_open,
                finished=True,
                metadata=metadata,
                plan=plan,
            )
        if any(
            result.status in {NodeStatus.BLOCKED, NodeStatus.RUNNING}
            for result in metadata.node_results
        ):
            return self._step(
                session,
                metadata.state,
                generation_open,
                metadata=metadata,
                plan=plan,
            )
        if metadata.plan_completed:
            if metadata.state.pending_evaluations:
                return self._step(
                    session,
                    metadata.state,
                    generation_open,
                    metadata=metadata,
                    plan=plan,
                )
            environment.finish_generation(metadata.state)
            return self._step(
                session,
                metadata.state,
                False,
                observable=True,
                finished=False,
                metadata=metadata,
                plan=plan,
            )
        return self._step(
            session,
            metadata.state,
            generation_open,
            metadata=metadata,
            plan=plan,
        )

    @staticmethod
    def _has_unfinished_async_work(
        plan: SequentialPlan, state: OptimizationState
    ) -> bool:
        return _plan_has_unfinished_async_work(plan, state)

    @staticmethod
    def _publication_deferred(environment: object) -> bool:
        return getattr(environment, "_publication_deferred", False) is True

    def advance(self, session: RuntimeSession) -> RuntimeStep:
        """Poll, refill, drain, and expose asynchronous generation boundaries."""
        if not isinstance(session, RuntimeSession):
            raise ValidationError(
                "AsyncPipelineRuntime.advance requires a RuntimeSession"
            )
        if self.environment is not None:
            mark = getattr(self.environment, "_mark_checkpoint_state", None)
            if callable(mark):
                mark(session.state, phase="running", frames=session.frames)
        if isinstance(session.plan, StructuredPlan):
            return self._advance_structured_async(session)
        if not isinstance(session.plan, SequentialPlan):
            raise ValidationError(
                "AsyncPipelineRuntime.advance requires a SequentialPlan session"
            )
        if self.environment is None:
            raise ValidationError("AsyncPipelineRuntime requires a runtime environment")
        env = cast(AsyncExecutionEnvironment, self.environment)
        env.fatal(session.state)
        state = session.state
        plan = session.plan
        generation_open = session.generation_open
        metadata = _ExecutionOutcome(state=state)
        commit = getattr(env, "_refresh_plan_if_needed", None)
        if callable(commit):
            commit(state)
            plan = getattr(env, "plan", plan)

        if state.pending_evaluations:
            if set(state.pending_evaluations) != set(state.evaluation_handles):
                state = env.reattach(state)
            result = env.poll(state)
            state = result.state
            _record_history_observation(state)
            if state.pending_evaluations:
                if env.is_terminated(state):
                    if not result.progressed:
                        time.sleep(0.001)
                    return self._step(session, state, generation_open)
                if not env.can_refill(state) or self._publication_deferred(env):
                    return self._step(session, state, generation_open)
                metadata = _environment_execute_async(env, plan, state)
                state = metadata.state
                if metadata.recompile_requested:
                    plan = _environment_recompile(cast(RecompileProvider, env), plan)
                    return self._step(
                        session,
                        state,
                        generation_open,
                        metadata=metadata,
                        plan=plan,
                    )
                if metadata.terminated:
                    env.dispatch(RunEndEvent(ctx=state))
                    return self._step(
                        session,
                        state,
                        generation_open,
                        finished=True,
                        metadata=metadata,
                        plan=plan,
                    )
                if state.pending_evaluations:
                    if not result.progressed:
                        time.sleep(0.001)
                    return self._step(
                        session, state, generation_open, metadata=metadata
                    )

            # Drain capacity-split plans before closing the generation.
            if self._has_unfinished_async_work(plan, state):
                metadata = _environment_execute_async(env, plan, state)
                state = metadata.state
                if metadata.recompile_requested:
                    plan = _environment_recompile(cast(RecompileProvider, env), plan)
                    return self._step(
                        session,
                        state,
                        generation_open,
                        metadata=metadata,
                        plan=plan,
                    )
                if metadata.terminated:
                    env.dispatch(RunEndEvent(ctx=state))
                    return self._step(
                        session,
                        state,
                        generation_open,
                        finished=True,
                        metadata=metadata,
                        plan=plan,
                    )
                if state.pending_evaluations:
                    return self._step(
                        session, state, generation_open, metadata=metadata
                    )

            if generation_open:
                env.finish_generation(state)
                return self._step(
                    session,
                    state,
                    False,
                    observable=True,
                    metadata=metadata,
                    plan=plan,
                )

        elif self._has_unfinished_async_work(plan, state):
            # Checkpoints may retain deferred requests without active handles.
            metadata = _environment_execute_async(env, plan, state)
            state = metadata.state
            if metadata.recompile_requested:
                plan = _environment_recompile(cast(RecompileProvider, env), plan)
                return self._step(
                    session,
                    state,
                    generation_open,
                    metadata=metadata,
                    plan=plan,
                )
            if metadata.terminated:
                env.dispatch(RunEndEvent(ctx=state))
                return self._step(
                    session,
                    state,
                    generation_open,
                    finished=True,
                    metadata=metadata,
                    plan=plan,
                )
            if state.pending_evaluations:
                return self._step(
                    session, state, generation_open, metadata=metadata, plan=plan
                )

        if env.is_terminated(state):
            env.dispatch(RunEndEvent(ctx=state))
            return self._step(session, state, generation_open, finished=True, plan=plan)
        env.dispatch(GenerationStartEvent(ctx=state))
        metadata = _environment_execute_async(env, plan, state)
        state = metadata.state
        if metadata.recompile_requested:
            plan = _environment_recompile(cast(RecompileProvider, env), plan)
            return self._step(
                session,
                state,
                generation_open,
                metadata=metadata,
                plan=plan,
            )
        if metadata.terminated:
            env.dispatch(RunEndEvent(ctx=state))
            return self._step(
                session,
                state,
                generation_open,
                finished=True,
                metadata=metadata,
                plan=plan,
            )
        generation_open = True
        if not state.pending_evaluations:
            env.finish_generation(state)
            generation_open = False
            return self._step(
                session,
                state,
                generation_open,
                observable=True,
                metadata=metadata,
                plan=plan,
            )
        return self._step(session, state, generation_open, metadata=metadata, plan=plan)


class RuntimeFactory(Protocol):
    """Factory protocol for runtimes selected by a provider registry."""

    def __call__(self, optimizer: object, plan: ExecutablePlan) -> ExecutionRuntime:
        """Create a runtime for an already resolved plan."""
        ...


@dataclass(frozen=True, kw_only=True)
class RuntimeRegistration:
    """One explicit runtime selection rule."""

    name: str
    matches: Callable[[object], bool]
    factory: RuntimeFactory
    offered_runtime_capabilities: frozenset[RuntimeCapability] = frozenset()
    capability_provider: Callable[[object], Iterable[RuntimeCapability]] | None = None

    def __post_init__(self) -> None:
        if self.capability_provider is not None and not callable(
            self.capability_provider
        ):
            raise ValidationError("capability_provider must be callable")
        capabilities = frozenset(self.offered_runtime_capabilities)
        for capability in capabilities:
            validate_name(capability)
        object.__setattr__(
            self,
            "offered_runtime_capabilities",
            capabilities,
        )

    def offered_capabilities(self, optimizer: object) -> frozenset[RuntimeCapability]:
        """Resolve this provider's static or optimizer-dependent offer."""
        values = (
            self.capability_provider(optimizer)
            if self.capability_provider is not None
            else self.offered_runtime_capabilities
        )
        capabilities = frozenset(values)
        for capability in capabilities:
            validate_name(capability)
        return capabilities


class RuntimeRegistry:
    """Runtime selection errors on conflict; compiler rules do not select a runtime."""

    def __init__(self, registrations: tuple[RuntimeRegistration, ...] = ()) -> None:
        self._registrations: dict[str, RuntimeRegistration] = {}
        for registration in registrations:
            self.register(registration)

    def register(self, registration: RuntimeRegistration) -> None:
        """Add a uniquely named runtime provider."""
        if registration.name in self._registrations:
            raise ValidationError(
                f"runtime registration {registration.name!r} already exists"
            )
        if not callable(registration.matches) or not callable(registration.factory):
            raise ValidationError(
                "runtime registration requires callable matches and factory"
            )
        self._registrations[registration.name] = registration

    def unregister(self, name: str) -> None:
        """Remove the runtime provider registered under ``name``."""
        if name not in self._registrations:
            raise ValidationError(f"runtime registration {name!r} is not registered")
        del self._registrations[name]

    def replace(self, name: str, registration: RuntimeRegistration) -> None:
        """Replace the uniquely named runtime provider under ``name``."""
        if name not in self._registrations:
            raise ValidationError(f"runtime registration {name!r} is not registered")
        if not callable(registration.matches) or not callable(registration.factory):
            raise ValidationError(
                "runtime registration requires callable matches and factory"
            )
        if registration.name != name and registration.name in self._registrations:
            raise ValidationError(
                f"runtime registration {registration.name!r} already exists"
            )
        del self._registrations[name]
        self._registrations[registration.name] = registration

    def _resolve(self, optimizer: object) -> RuntimeRegistration:
        matches = tuple(
            registration
            for registration in self._registrations.values()
            if registration.matches(optimizer)
        )
        registered_names = ", ".join(repr(name) for name in self._registrations)
        if not matches:
            names = registered_names or "(none)"
            raise ConfigurationError(
                "no registered runtime matches the optimizer; "
                f"registered runtime names: {names}"
            )
        if len(matches) > 1:
            names = ", ".join(repr(registration.name) for registration in matches)
            raise ConfigurationError(
                "multiple registered runtimes match the optimizer; "
                f"conflicting registration names: {names}"
            )
        return matches[0]

    def create(self, optimizer: object, plan: ExecutablePlan) -> ExecutionRuntime:
        """Select and instantiate the unique matching provider."""
        return self._resolve(optimizer).factory(optimizer, plan)

    def offered_capabilities(self, optimizer: object) -> frozenset[RuntimeCapability]:
        """Return the effective offer from the selected runtime provider."""
        return self._resolve(optimizer).offered_capabilities(optimizer)

    def registrations(self) -> tuple[RuntimeRegistration, ...]:
        """Return registered providers in registration order."""
        return tuple(self._registrations.values())


def _sync_runtime_factory(optimizer: object, plan: ExecutablePlan) -> ExecutionRuntime:
    runtime_plan: SequentialPlan | StructuredPlan
    if isinstance(plan.graph, StructuredGraph):
        runtime_plan = StructuredPlan.from_executable_plan(plan)
    else:
        runtime_plan = SequentialPlan.from_executable_plan(plan)
    return PipelineRuntime(_OptimizerEnvironment(optimizer, runtime_plan))


def _async_runtime_factory(optimizer: object, plan: ExecutablePlan) -> ExecutionRuntime:
    runtime_plan: SequentialPlan | StructuredPlan
    if isinstance(plan.graph, StructuredGraph):
        runtime_plan = StructuredPlan.from_executable_plan(plan)
    else:
        runtime_plan = SequentialPlan.from_executable_plan(plan)
    return AsyncPipelineRuntime(_OptimizerEnvironment(optimizer, runtime_plan))


def _sync_runtime_capability_provider(
    optimizer: object,
) -> frozenset[RuntimeCapability]:
    return _algorithm_runtime_capabilities(optimizer)


default_runtime_registry = RuntimeRegistry(
    (
        RuntimeRegistration(
            name="async",
            matches=lambda optimizer: (
                getattr(optimizer, "async_evaluation_scheduler", None) is not None
            ),
            factory=_async_runtime_factory,
            offered_runtime_capabilities=AsyncPipelineRuntime.capabilities,
        ),
        RuntimeRegistration(
            name="sync",
            matches=lambda optimizer: (
                getattr(optimizer, "async_evaluation_scheduler", None) is None
            ),
            factory=_sync_runtime_factory,
            offered_runtime_capabilities=PipelineRuntime.capabilities,
            capability_provider=_sync_runtime_capability_provider,
        ),
    )
)


def create_runtime(optimizer: object) -> ExecutionRuntime:
    """Create a runtime through the replaceable default provider registry."""
    plan = resolve_plan(optimizer)
    return default_runtime_registry.create(optimizer, plan)


def resolve_plan(optimizer: object) -> ExecutablePlan:
    """Return the optimizer plan, compiling only for bare providers."""
    plan = getattr(optimizer, "executable_plan", None)
    if isinstance(plan, ExecutablePlan):
        return plan
    resolver = getattr(optimizer, "resolve_defaults", None)
    if callable(resolver):
        resolver()
    strategy = getattr(optimizer, "strategy", None)
    build_graph = getattr(strategy, "build_graph", None)
    if not callable(build_graph):
        raise ValidationError("default runtime requires an executable plan")
    component_resolution = getattr(optimizer, "_component_resolution", None)
    plan = _prepare_plan(
        strategy,
        optimizer,
        problem=None,
        offered_runtime_capabilities=None,
        _context_factory=lambda: (
            getattr(optimizer, "problem", None),
            default_runtime_registry.offered_capabilities(optimizer),
        ),
        initial_state_keys=OPTIMIZATION_STATE_INITIAL_KEYS,
        _verification_enabled=getattr(optimizer, "_verification_enabled", True),
        _component_resolution=(
            component_resolution
            if isinstance(component_resolution, _ComponentResolution)
            else None
        ),
    )
    resolution = getattr(optimizer, "_component_resolution", component_resolution)
    publish_resolution = getattr(optimizer, "_publish_component_resolution", None)
    if isinstance(resolution, _ComponentResolution) and callable(publish_resolution):
        publish_resolution(
            resolution,
            feature_encoder_consumed=_executable_uses_service(plan, "FeatureEncoder"),
        )
    return plan
