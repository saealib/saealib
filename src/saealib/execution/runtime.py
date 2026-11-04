"""Runtime implementations and the optimizer environment compatibility bridge."""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import logging
import time
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass, replace
from types import SimpleNamespace
from typing import Any, Protocol, cast

import numpy as np

from saealib.callback import (
    Event,
    GenerationEndEvent,
    GenerationStartEvent,
    RunEndEvent,
    RunStartEvent,
)
from saealib.comparators import NSGA3Comparator
from saealib.context import OptimizationState
from saealib.core.compiler.compiler import CompileContext, Compiler, ExecutablePlan
from saealib.core.compiler.graph import ComponentNode
from saealib.core.compiler.regions import (
    BranchRegion,
    Condition,
    LoopRegion,
    RegionNode,
    RepeatRegion,
)
from saealib.core.compiler.structured import StructuredGraph
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
    RuntimeContext,
    StateKey,
    StateView,
)
from saealib.core.state.patch import StatePatch
from saealib.core.state.readonly import _unwrap_readonly
from saealib.exceptions import ConfigurationError, EvaluationFatalError, ValidationError
from saealib.execution.history import (
    record_decision,
    record_evaluations,
    record_generation,
)
from saealib.execution.prepare import BINDING_SLOTS

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


# Compatibility aliases.  The capability protocols above are the canonical
# contracts; these names remain available for callers that imported them.
RuntimeEnvironment = SequentialExecutionEnvironment
AsyncRuntimeEnvironment = AsyncExecutionEnvironment


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
        operation = frame.graph.operations[index]
        if (
            isinstance(operation, ComponentNode)
            and _async_leaf_driver(operation.component) is not None
            and (marker := _async_protocol_marker(operation.component)) is not None
            and marker[1] == "driver"
        ):
            driver_index = index
            break
    if driver_index is None:
        return frames
    return (*frames[:-1], replace(frame, operation_index=driver_index))


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
    if async_control and resume_async_driver:
        stack = list(_rewind_to_async_driver(tuple(stack)))
    results: list[NodeResult] = []
    executed: list[str] = []
    refused: list[RuntimeCommand] = []
    debug_enabled = logger.isEnabledFor(logging.DEBUG)

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
            count = child.count if child.count is not None else 0
            if child.iteration + 1 < count:
                stack[-1] = replace(
                    child, operation_index=0, iteration=child.iteration + 1
                )
                return False
        elif isinstance(region, LoopRegion):
            if not _evaluate_condition(region.condition, current, dispatch=dispatch):
                stack[-1] = replace(
                    child, operation_index=0, iteration=child.iteration + 1
                )
                return False
        stack.pop()
        stack[-1] = replace(parent, operation_index=parent.operation_index + 1)
        return False

    while True:
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


class _SnapshotProviderView:
    """Provider view over one candidate binding set.

    Slots are served from the pending snapshot values, so plan preparation
    never reads the live optimizer, even mid-run.  Non-slot attributes fall
    through to the optimizer (problem, space, runtime flags).
    """

    def __init__(self, optimizer: Any, bindings: dict[str, object]) -> None:
        self._optimizer = optimizer
        self._bindings = bindings

    def binding_values(self) -> dict[str, object]:
        """Candidate snapshot values for declaration resolution."""
        return dict(self._bindings)

    def __getattr__(self, name: str) -> object:
        if name in _BINDING_SLOTS:
            return self._bindings.get(name)
        return getattr(self._optimizer, name)


@dataclass(frozen=True)
class _BindingSnapshot:
    """One published binding set: component identities plus a version.

    Holds strong references to every bound component so identity comparison
    at later commit points cannot be confused by object id reuse.
    """

    version: int
    values: dict[str, object]
    strategy_values: tuple[tuple[str, str], ...]

    def unchanged_by(
        self,
        values: dict[str, object],
        strategy_values: tuple[tuple[str, str], ...],
    ) -> bool:
        """Return True when the candidate bindings resolve to this snapshot."""
        if set(values) != set(self.values):
            return False
        if any(values[name] is not bound for name, bound in self.values.items()):
            return False
        return strategy_values == self.strategy_values


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


def _plan_has_unfinished_async_work(
    plan: SequentialPlan | StructuredPlan, state: OptimizationState
) -> bool:
    for node in _iter_plan_components(plan):
        capability = getattr(node.component, "has_async_work", None)
        if callable(capability) and capability(state):
            return True
    return False


class _OptimizerEnvironment:
    def __init__(self, optimizer: Any, plan: SequentialPlan | StructuredPlan) -> None:
        self.optimizer = optimizer
        self.plan = plan
        self._version = 0
        self._publication_deferred = False
        self._active_snapshot = _BindingSnapshot(
            version=self._version,
            values=self._bindings(),
            strategy_values=self._strategy_values(),
        )
        self.capabilities = _algorithm_runtime_capabilities(optimizer)
        self._publish_scheduler()
        scheduler = getattr(optimizer, "async_evaluation_scheduler", None)
        if scheduler is not None and any(
            getattr(insertion, "adapter_name", None) == "feedback_accumulator"
            for insertion in plan.plan.inserted_adapters
        ):
            # The graph owns delivery; the seam only enables scheduler state.
            scheduler.enable_feedback_accumulator()

    @property
    def active_version(self) -> int:
        """Version of the binding snapshot that the active plan belongs to."""
        return self._active_snapshot.version

    def binding_values(self) -> dict[str, object]:
        """Active snapshot bindings for components resolved by the runtime."""
        return dict(self._active_snapshot.values)

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
        self._refresh_plan_if_needed(state)
        if not isinstance(self.plan, SequentialPlan):
            raise ValidationError(
                "sequential environment cannot execute a structured plan"
            )
        return _execute_with_metadata(
            self.plan,
            state,
            dispatch=self.dispatch,
            observe=_record_history_observation,
        )

    def _bindings(self) -> dict[str, object]:
        return {name: getattr(self.optimizer, name, None) for name in _BINDING_SLOTS}

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
        during one flow execution the active version stays untouched.
        """
        bindings = self._bindings()
        strategy_values = self._strategy_values()
        if self._active_snapshot.unchanged_by(bindings, strategy_values):
            # A swap that was deferred while its evaluations ran may resolve
            # back to the active bindings; a matching commit point then ends
            # the deferral, otherwise pending-collection would stall the run.
            self._publication_deferred = False
            return
        if self._replacement_blocked(state):
            self._publication_deferred = True
            return
        self._publication_deferred = False
        self._publish(bindings, strategy_values)

    def _replacement_blocked(self, state: OptimizationState | None) -> bool:
        """Outstanding asynchronous work under the active version."""
        scheduler = getattr(self.optimizer, "async_evaluation_scheduler", None)
        if scheduler is None or state is None:
            return False
        if state.pending_evaluations or state.evaluation_handles:
            return True
        return _plan_has_unfinished_async_work(self.plan, state)

    def _publish(
        self,
        bindings: dict[str, object],
        strategy_values: tuple[tuple[str, str], ...],
    ) -> None:
        executable, rebuilt = self._compile_current_plan(bindings)
        self._gate(executable)
        self.plan = rebuilt
        if hasattr(self.optimizer, "_executable_plan"):
            self.optimizer._executable_plan = executable
        self._version += 1
        self._active_snapshot = _BindingSnapshot(
            version=self._version,
            values=bindings,
            strategy_values=strategy_values,
        )
        self._publish_scheduler()

    def _publish_scheduler(self) -> None:
        """Serve the scheduler from the active snapshot, never live reads.

        The feedback builder is served by the submission that opens each
        request; requests restored from a checkpoint commit with the builder
        the scheduler already holds.
        """
        scheduler = getattr(self.optimizer, "async_evaluation_scheduler", None)
        if scheduler is None:
            return
        values = self._active_snapshot.values
        updated = {
            "algorithm": values.get("algorithm"),
            "callback_manager": values.get("cbmanager"),
        }
        for name, value in updated.items():
            with contextlib.suppress(AttributeError):
                setattr(scheduler, name, value)

    def _gate(self, executable: ExecutablePlan) -> None:
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
        if has_plan:
            optimizer._executable_plan = executable
        try:
            issues = validate(require_initializer=False)
        finally:
            if has_plan:
                optimizer._executable_plan = previous
        if issues:
            raise ConfigurationError(
                "Component replacement rejected at the commit point:\n"
                + "\n".join(f"  - {message}" for message in issues)
            )

    def recompile(
        self, plan: SequentialPlan | StructuredPlan
    ) -> SequentialPlan | StructuredPlan:
        self._publish(self._bindings(), self._strategy_values())
        return self.plan

    def _compile_current_plan(
        self, bindings: dict[str, object]
    ) -> tuple[ExecutablePlan, SequentialPlan | StructuredPlan]:
        strategy = getattr(self.optimizer, "strategy", None)
        build_graph = getattr(strategy, "build_graph", None)
        problem = getattr(self.optimizer, "problem", None)
        if not callable(build_graph) or problem is None:
            raise ValidationError(
                "runtime plan refresh requires strategy.build_graph and problem"
            )
        graph = build_graph(_SnapshotProviderView(self.optimizer, bindings))
        executable = Compiler().compile(
            graph,
            CompileContext(
                space=problem.space,
                problem=problem,
                offered_runtime_capabilities=default_runtime_registry.offered_capabilities(
                    self.optimizer
                ),
                initial_state_keys=OPTIMIZATION_STATE_INITIAL_KEYS,
            ),
        )
        validate_plan_contracts(executable)
        rebuilt: SequentialPlan | StructuredPlan
        if isinstance(graph, StructuredGraph):
            rebuilt = StructuredPlan.from_executable_plan(executable)
        else:
            rebuilt = SequentialPlan.from_executable_plan(executable)
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
            return _execute_with_metadata(
                sequential_plan,
                state,
                dispatch=self.dispatch,
                observe=_record_history_observation,
            )
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
            return halt.outcome
        logger.debug(
            "Runtime %s end: %s status=completed",
            kind,
            nodes[async_index].component_id,
        )
        _record_history_observation(result)
        if prefix_outcome is not None:
            return _ExecutionOutcome(
                state=result,
                node_results=prefix_outcome.node_results,
                executed_node_ids=prefix_outcome.executed_node_ids,
                refused_commands=prefix_outcome.refused_commands,
            )
        return _ExecutionOutcome(
            state=result,
        )

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
        self.optimizer.dispatch(event)

    def finish_generation(self, state: OptimizationState) -> None:
        handler = state.problem.handler
        handler.on_generation_end(state.gen, state.population)
        threshold = state.problem.handler.feasibility_threshold
        state.comparator.eps_cv = threshold
        state.pareto_archive.eps_cv = threshold
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


def execute_strategy_step(
    strategy: object, state: OptimizationState, provider: object
) -> OptimizationState:
    """Compatibility step facade with async lifecycle owned by Runtime.

    This supports callers that still invoke ``strategy.step`` directly.  The
    strategy contributes only its canonical pipeline; scheduler polling,
    capacity checks, and async submission remain runtime responsibilities.
    """
    scheduler = getattr(provider, "async_evaluation_scheduler", None)
    build_graph = cast(Any, getattr(strategy, "build_graph", None))
    if not callable(build_graph):
        raise ValidationError("strategy requires a callable build_graph")
    graph = build_graph(provider)
    plan = Compiler().compile(
        graph,
        CompileContext(
            space=state.problem.space,
            problem=state.problem,
            offered_runtime_capabilities=(
                frozenset({"partial_feedback"})
                if scheduler is not None
                else frozenset()
            ),
            initial_state_keys=OPTIMIZATION_STATE_INITIAL_KEYS,
        ),
    )
    from saealib.strategies.base import TemplateStrategy, build_pipeline_from_graph

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
    if isinstance(graph, StructuredGraph):
        raise ValidationError(
            "async strategy steps require a sequential graph execution seam"
        )
    if state.pending_evaluations:
        state = scheduler.poll(state, wait=False)
        if (
            not state.pending_evaluations
            or len(state.pending_evaluations) >= scheduler.max_pending
        ):
            return state
    optimizer = SimpleNamespace(
        strategy=strategy,
        problem=state.problem,
        algorithm=getattr(provider, "algorithm", None),
        evaluator=getattr(provider, "evaluator", None),
        feedback_builder=getattr(provider, "feedback_builder", None),
        cbmanager=getattr(provider, "cbmanager", None),
        async_evaluation_scheduler=scheduler,
    )
    environment = _OptimizerEnvironment(
        optimizer, SequentialPlan.from_executable_plan(plan)
    )
    sequential = SequentialPlan.from_executable_plan(plan)
    environment.recompile(sequential)
    return environment.execute_async(sequential, state)


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
            if not structured_graph:
                for name, service in node.resolved_services.items():
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
        if structured is None:
            state.bind_compiled_services(services)
        selected_plan = structured or SequentialPlan.from_executable_plan(plan)
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
        self._prepare_state(state)
        self.environment.dispatch(RunStartEvent(ctx=state))
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
        threshold = state.problem.handler.feasibility_threshold
        state.comparator.eps_cv = threshold
        state.pareto_archive.eps_cv = threshold
        if (
            isinstance(state.comparator, NSGA3Comparator)
            and state.comparator._rng is None
        ):
            state.comparator.rng = state.rng.spawn(1)[0]


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
        refresh = getattr(environment, "_refresh_plan_if_needed", None)
        if callable(refresh) and not session.frames:
            refresh(session.state)
        plan = getattr(environment, "plan", session.plan)
        if not isinstance(plan, StructuredPlan):
            raise ValidationError(
                "structured async runtime environment returned a non-structured plan"
            )
        state = session.state
        _sync_transient_fields_from_store(state)
        generation_open = session.generation_open
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
                if not result.progressed:
                    time.sleep(0.001)
                return self._step(
                    session,
                    state,
                    generation_open,
                    plan=plan,
                )
        if environment.is_terminated(state):
            environment.dispatch(RunEndEvent(ctx=state))
            return self._step(
                session,
                state,
                generation_open,
                finished=True,
                plan=plan,
            )
        if not generation_open:
            environment.dispatch(GenerationStartEvent(ctx=state))
            generation_open = True
        metadata = _execute_structured(
            plan,
            state,
            session.frames,
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
    strategy = getattr(optimizer, "strategy", None)
    build_graph = getattr(strategy, "build_graph", None)
    if not callable(build_graph):
        raise ValidationError("default runtime requires an executable plan")
    graph = build_graph(optimizer)
    problem = getattr(optimizer, "problem", None)
    context = None
    if problem is not None:
        context = CompileContext(
            space=problem.space,
            problem=problem,
            offered_runtime_capabilities=default_runtime_registry.offered_capabilities(
                optimizer
            ),
            initial_state_keys=OPTIMIZATION_STATE_INITIAL_KEYS,
        )
    return Compiler().compile(graph, context)
