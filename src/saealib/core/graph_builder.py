"""Build component graphs from stage-based compatibility pipelines.

The graph builder is deliberately a bridge.  ``Stage`` remains the execution
object used by the existing optimizer, while ``StageNodeAdapter`` gives the
compiler vocabulary a component-shaped view of the components a stage holds.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from types import SimpleNamespace
from typing import Any, cast

from saealib.core.compiler.contract_diagnostics import check_component_contract
from saealib.core.compiler.graph import (
    ComponentGraph,
    ComponentNode,
    ControlEdge,
    DataEdge,
    NodeRef,
    StateBinding,
)
from saealib.core.compiler.structured import StructuredGraph
from saealib.core.contracts import (
    AssumptionSet,
    ComponentContract,
    DataSpec,
    ExecutionContract,
    LifecycleContract,
    PortContract,
    PortDirection,
    StateContract,
)
from saealib.core.contracts.flow import (
    CallInput,
    CallOutput,
    ComponentCall,
    FlowEdge,
    StageFlowSummary,
    StateValue,
    Transform,
    _apply_transforms,
    _transform_input_data,
)
from saealib.core.contracts.ports import PortSpec
from saealib.core.state import StateKey
from saealib.exceptions import ValidationError
from saealib.pipeline import ComponentRef, Pipeline, Stage

__all__ = [
    "NodeAdapterSpec",
    "StageContractNodeAdapter",
    "StageNodeAdapter",
    "StagePartNodeAdapter",
    "build_component_graph",
    "build_decomposed_component_graph",
    "build_decomposed_component_graph_from_specs",
    "build_decomposed_component_graph_from_stages",
    "decompose_structured_graph",
]

_MISSING = object()


@dataclass(frozen=True)
class _HeldComponent:
    path: tuple[str, ...]
    installed_name: str | None
    component: object
    contract: ComponentContract


def _component_contract(value: object) -> ComponentContract | None:
    if isinstance(value, Stage):
        return None
    method = getattr(value, "analysis_contract", None)
    if not callable(method):
        method = getattr(value, "contract", None)
    if not callable(method):
        return None
    contract = method()
    if not isinstance(contract, ComponentContract):
        raise ValidationError(
            f"{type(value).__name__}.contract() must return ComponentContract"
        )
    return contract


def _path_piece(value: object) -> str:
    if isinstance(value, str):
        return value.lstrip("_") or "component"
    return str(value)


def _held_components(stage: Stage) -> tuple[_HeldComponent, ...]:
    """Collect contracts held by *stage*, including nested stage components.

    The walk is intentionally structural rather than based on concrete Stage
    classes.  This keeps custom stages usable and covers the nested component
    collections used by composite surrogate managers.  A Stage's own
    ``contract()`` is never consulted; only its held components contribute.
    """
    found: list[_HeldComponent] = []
    seen: set[int] = set()

    def visit(
        value: object,
        path: tuple[str, ...],
        installed_name: str | None = None,
    ) -> None:
        value_id = id(value)
        if value_id in seen:
            return
        if isinstance(value, ComponentRef):
            raise ValidationError(
                f"{type(value).__name__}({value.role!r}) at {'/'.join(path)!r} is "
                "an unresolved (slot, role) reference; resolve the declaration "
                "against a binding snapshot before building the graph."
            )
        if isinstance(value, Stage):
            seen.add(value_id)
            for index, child in enumerate(value.stages):
                visit(child, (*path, f"stage_{index}"), installed_name)
            attributes = getattr(value, "__dict__", {})
            if isinstance(attributes, Mapping):
                for name, child in attributes.items():
                    if name != "stages":
                        visit(child, (*path, _path_piece(name)), installed_name)
            return
        if isinstance(value, Mapping):
            seen.add(value_id)
            for key, child in value.items():
                visit(
                    child,
                    (*path, _path_piece(key)),
                    _path_piece(key),
                )
            return
        if isinstance(value, (list, tuple, set, frozenset)):
            seen.add(value_id)
            for index, child in enumerate(value):
                visit(child, (*path, f"item_{index}"), installed_name)
            return

        contract = _component_contract(value)
        if contract is not None:
            found.append(
                _HeldComponent(
                    path=path,
                    installed_name=installed_name,
                    component=value,
                    contract=contract,
                )
            )

        # Components such as CompositeSurrogateManager keep named child
        # managers in ordinary attributes.  Walk those attributes as well so
        # each installed surrogate can receive its own graph binding.
        seen.add(value_id)
        attributes = getattr(value, "__dict__", {})
        if isinstance(attributes, Mapping):
            for name, child in attributes.items():
                visit(child, (*path, _path_piece(name)), installed_name)

    visit(stage, (stage.name or type(stage).__name__,))
    return tuple(found)


def _append_unique(values: list[Any], value: Any) -> None:
    if value not in values:
        values.append(value)


def _qualified_role(path: tuple[str, ...], role: str, index: int) -> str:
    owner = _path_piece(path[-1] if path else f"component_{index}")
    return f"{owner}:{role}"


def _compose_contracts(held: Sequence[_HeldComponent]) -> ComponentContract:
    """Compose the contracts of the components held by a Stage.

    State, lifecycle, execution, and assumptions are unions.  Ports retain
    their original role when possible; a colliding role is qualified by the
    held-component path.  Held components are not declared as parts because
    they are a collection inside the Stage, not named adapter attrs.
    """
    ports: dict[str, PortContract] = {}
    reads: list[StateKey[object]] = []
    writes: list[StateKey[object]] = []
    exports: list[StateKey[object]] = []
    reads_enumerable = True
    writes_enumerable = True
    events = []
    feedback = None
    capabilities: list[str] = []
    offered_capabilities: list[str] = []
    assumptions: dict[str, bool] = {}

    for index, item in enumerate(held, start=1):
        contract = item.contract
        for role, port in contract.ports.items():
            role_name = role
            if role_name in ports:
                if ports[role_name] == port:
                    continue
                role_name = _qualified_role(item.path, role, index)
                while role_name in ports:
                    role_name = _qualified_role(
                        (*item.path, f"held_{index}"), role, index
                    )
            ports[role_name] = port
        for key in contract.state.reads:
            _append_unique(reads, key)
        for key in contract.state.writes:
            _append_unique(writes, key)
        for key in contract.state.exports:
            _append_unique(exports, key)
        reads_enumerable = reads_enumerable and contract.state.reads_enumerable
        writes_enumerable = writes_enumerable and contract.state.writes_enumerable
        for event in contract.lifecycle.events:
            _append_unique(events, event)
        if contract.lifecycle.feedback is not None:
            if feedback is None:
                feedback = contract.lifecycle.feedback
            elif feedback != contract.lifecycle.feedback:
                # A Stage adapter has no lifecycle policy of its own.  A
                # conflicting pair cannot be represented by LifecycleContract;
                # retain the structural union and leave the later lifecycle
                # rule to inspect the underlying parts.
                feedback = None
        for capability in contract.execution.required_runtime_capabilities:
            _append_unique(capabilities, capability)
        for capability in contract.execution.offered_runtime_capabilities:
            _append_unique(offered_capabilities, capability)
        for name, value in contract.assumptions.items():
            if name not in assumptions:
                assumptions[name] = value
            elif assumptions[name] != value:
                # The conservative value keeps a composed contract from
                # claiming an assumption that one of its parts rejects.
                assumptions[name] = False

    return ComponentContract(
        ports=ports,
        lifecycle=LifecycleContract(events=tuple(events), feedback=feedback),
        state=StateContract(
            reads=tuple(reads),
            writes=tuple(writes),
            exports=tuple(exports),
            reads_enumerable=reads_enumerable,
            writes_enumerable=writes_enumerable,
        ),
        execution=ExecutionContract(
            required_runtime_capabilities=tuple(capabilities),
            offered_runtime_capabilities=tuple(offered_capabilities),
        ),
        assumptions=AssumptionSet(assumptions),
    )


def _surrogate_key_name(node_id: str, item: _HeldComponent) -> str:
    node_id = node_id.replace(".", "__")
    installed_name = item.installed_name
    if installed_name is None:
        ignored = {
            "manager",
            "managers",
            "sm",
            "surrogate",
            "surrogate_manager",
        }
        for piece in reversed(item.path[1:]):
            if piece in ignored or piece.startswith("stage_"):
                continue
            installed_name = piece
            break
    if installed_name is None:
        return node_id
    return f"{node_id}:{installed_name}"


class StageNodeAdapter:
    """Represent a ``Stage`` as a graph component.

    The adapter delegates execution to the unchanged Stage and exposes only a
    composed contract for the components held by that Stage.  It intentionally
    does not declare the Stage's direct ``OptimizationState`` accesses; those
    declarations belong to the Stage contract layer.
    """

    _execution_mode = "optimization_state"

    def __init__(self, stage: Stage, *, node_path: str | None = None) -> None:
        if not isinstance(stage, Stage):
            raise ValidationError("StageNodeAdapter stage must be a Stage")
        self.stage = stage
        self._execute_target = getattr(stage, "execute", None)
        if not callable(self._execute_target):
            raise ValidationError("StageNodeAdapter stage must be executable")
        self.node_path = node_path or stage.name or type(stage).__name__
        self._held = _held_components(stage)
        self._contract = _compose_contracts(self._held)

    def __getattr__(self, name: str) -> object:
        return getattr(self.stage, name)

    def execute(self, state: Any) -> Any:
        """Delegate execution to the existing Stage implementation."""
        return self.stage.execute(state)

    def contract(self) -> ComponentContract:
        """Compose the held components' contracts for this adapter."""
        return self._contract

    def state_bindings(self, node: NodeRef | str) -> tuple[StateBinding, ...]:
        """Return node-qualified bindings for exported surrogate state."""
        node_ref = NodeRef.from_value(node)
        bindings: list[StateBinding] = []
        seen: set[StateKey[object]] = set()
        for item in self._held:
            for key in item.contract.state.exports:
                if key.namespace != "surrogates":
                    continue
                qualified = StateKey[object](
                    namespace=key.namespace,
                    name=_surrogate_key_name(node_ref.component_id, item),
                    schema_version=key.schema_version,
                )
                if qualified in seen:
                    continue
                seen.add(qualified)
                bindings.append(StateBinding(node=node_ref, state_key=qualified))
        return tuple(bindings)


class StageContractNodeAdapter(StageNodeAdapter):
    """Executable Stage node carrying the Stage's direct contract."""

    def __init__(self, stage: Stage, *, node_path: str | None = None) -> None:
        if not isinstance(stage, Stage):
            raise ValidationError("StageNodeAdapter stage must be a Stage")
        self.stage = stage
        self._execute_target = getattr(stage, "execute", None)
        if not callable(self._execute_target):
            raise ValidationError("StageNodeAdapter stage must be executable")
        self.node_path = node_path or stage.name or type(stage).__name__
        self._held = ()
        contract = stage.contract()
        if not isinstance(contract, ComponentContract):
            raise ValidationError("Stage.contract() must return ComponentContract")
        self._contract = contract

    def contract(self) -> ComponentContract:
        """Return the Stage's declared contract."""
        return self._contract


class StagePartNodeAdapter:
    """Expose one Stage-held component as a verification-only node.

    Parts deliberately do not implement ``execute``. The Stage remains
    the executable owner; these nodes provide contract and data-dependency
    identity without executing a component a second time.
    """

    def __init__(self, component: object, contract: ComponentContract) -> None:
        self.component = component
        self._contract = contract

    def __getattr__(self, name: str) -> object:
        if name == "execute":
            raise AttributeError(name)
        return getattr(self.component, name)

    def contract(self) -> ComponentContract:
        """Return the held part's captured contract."""
        return self._contract


@dataclass(frozen=True, kw_only=True)
class NodeAdapterSpec:
    """Describe one executable graph node and its Stage adapter.

    The canonical builder consumes these already-materialized node specs.  A
    Stage is only an implementation detail of the compatibility adapter; it
    is not an input to the canonical graph construction API.
    """

    component_id: str
    adapter: StageContractNodeAdapter

    def __post_init__(self) -> None:
        if not isinstance(self.adapter, StageContractNodeAdapter):
            raise ValidationError(
                "NodeAdapterSpec adapter must be a StageContractNodeAdapter"
            )


def _unique_node_ids(stages: Sequence[Stage]) -> tuple[str, ...]:
    counts: dict[str, int] = {}
    used: set[str] = set()
    ids: list[str] = []
    for index, stage in enumerate(stages):
        base = stage.name or f"stage_{index}"
        count = counts.get(base, 1)
        candidate = base if count == 1 else f"{base}_{count}"
        while candidate in used:
            count += 1
            candidate = f"{base}_{count}"
        counts[base] = count + 1
        used.add(candidate)
        ids.append(candidate)
    return tuple(ids)


def _pipeline_stages(pipeline: Pipeline) -> tuple[Stage, ...]:
    """Validate the Pipeline-to-Stage compatibility bridge boundary."""
    stages = tuple(pipeline.stages)
    if any(not isinstance(stage, Stage) for stage in stages):
        raise ValidationError("Pipeline stages must contain Stage values")
    return tuple(stage for stage in stages if isinstance(stage, Stage))


def build_component_graph(pipeline: Pipeline) -> ComponentGraph:
    """Build a component graph from the flow declarations in *pipeline*."""
    if not isinstance(pipeline, Pipeline):
        raise ValidationError("build_component_graph requires a Pipeline")
    return build_decomposed_component_graph_from_stages(_pipeline_stages(pipeline))


@dataclass(frozen=True)
class _StageDecomposition:
    """Component-call nodes and edges for an ordered Stage sequence."""

    part_nodes: tuple[Mapping[str, tuple[str, ComponentNode]], ...]
    control_edges: tuple[ControlEdge, ...]
    data_edges: tuple[DataEdge, ...]
    state_bindings: tuple[StateBinding, ...]
    extra_nodes: tuple[ComponentNode, ...] = ()

    @property
    def nodes(self) -> tuple[ComponentNode, ...]:
        """Every declared component-call and external state node."""
        return (
            tuple(
                node for parts in self.part_nodes for _part_id, node in parts.values()
            )
            + self.extra_nodes
        )


def _unique_id(base: str, used: set[str]) -> str:
    candidate = base
    occurrence = 2
    while candidate in used:
        candidate = f"{base}_{occurrence}"
        occurrence += 1
    used.add(candidate)
    return candidate


def _state_source_id(key: StateKey[object], target_id: str) -> str:
    namespace = key.namespace.encode("utf-8").hex()
    name = key.name.encode("utf-8").hex()
    target = target_id.encode("utf-8").hex()
    return f"__state_{namespace}_{name}_{key.schema_version}_{target}"


def _state_source_node(
    key: StateKey[object],
    target_id: str,
    target_port: PortSpec,
    transforms=(),
    *,
    used_ids: set[str] | None = None,
) -> ComponentNode:
    source_base = _state_source_id(key, target_id)
    source_id = (
        _unique_id(source_base, used_ids) if used_ids is not None else source_base
    )
    source_contract = ComponentContract(
        ports={
            "state": PortContract(
                outputs=(
                    PortSpec(
                        name="value",
                        direction=PortDirection.OUTPUT,
                        data=_transform_input_data(target_port.data, tuple(transforms)),
                        cardinality=target_port.cardinality,
                    ),
                )
            )
        }
    )
    component = SimpleNamespace(
        state_key=key,
        target_id=target_id,
        contract=lambda: source_contract,
    )
    return ComponentNode(
        component_id=source_id,
        component=component,
        role="state",
    )


def _state_flow_node(
    key: StateKey[object],
    node_id: str,
    data: DataSpec,
    *,
    used_ids: set[str],
) -> ComponentNode:
    component_id = _unique_id(_state_source_id(key, node_id) + "__flow", used_ids)
    contract = ComponentContract(
        ports={
            "state": PortContract(
                inputs=(
                    PortSpec(
                        name="value",
                        direction=PortDirection.INPUT,
                        data=data,
                        cardinality="MANY",
                    ),
                ),
                outputs=(
                    PortSpec(
                        name="value",
                        direction=PortDirection.OUTPUT,
                        data=data,
                        cardinality="MANY",
                    ),
                ),
            )
        }
    )
    component = SimpleNamespace(
        state_key=key, flow_origin=node_id, contract=lambda: contract
    )
    return ComponentNode(component_id=component_id, component=component, role="state")


def _mark_verification_only(node: ComponentNode) -> None:
    """Mark a builder-created call or state node as non-executable."""
    object.__setattr__(node, "_verification_only", True)


def _call_port(
    node: ComponentNode, call: ComponentCall, port_name: str, direction: PortDirection
) -> PortSpec:
    role_contract = node.contract.ports.get(call.role)
    if role_contract is None:
        flow_contract = getattr(node.component, "flow_contract", None)
        if callable(flow_contract):
            role_contract = flow_contract().ports.get(call.role)
    ports = (
        ()
        if role_contract is None
        else (
            role_contract.outputs
            if direction is PortDirection.OUTPUT
            else role_contract.inputs
        )
    )
    matches = tuple(port for port in ports if port.name == port_name)
    if len(matches) != 1:
        raise ValidationError(
            f"Stage flow call {call.name!r} does not declare its "
            f"{direction.value} port {port_name!r} for role {call.role!r}"
        )
    return matches[0]


def _summary_call_contract(
    call: ComponentCall, summary: StageFlowSummary
) -> ComponentContract:
    """Describe only the directional ports named by a call summary."""
    unknown = DataSpec(kind="UnverifiedFlowValue")
    input_data: dict[str, DataSpec] = {}
    output_data: dict[str, DataSpec] = {}
    for edge in summary.edges:
        if isinstance(edge.target, CallInput) and edge.target.call == call:
            data = unknown
            if edge.transforms:
                first_input = _transform_input_data(unknown, edge.transforms)
                data, error = _apply_transforms(first_input, edge.transforms)
                if error is not None or data is None:
                    data = unknown
            input_data.setdefault(edge.target.port, data)
        if isinstance(edge.source, CallOutput) and edge.source.call == call:
            data = (
                _transform_input_data(unknown, edge.transforms)
                if edge.transforms
                else unknown
            )
            output_data.setdefault(edge.source.port, data)
    input_names = set(input_data)
    output_names = set(output_data)
    return ComponentContract(
        ports={
            call.role: PortContract(
                inputs=tuple(
                    PortSpec(
                        name=name,
                        direction=PortDirection.INPUT,
                        data=input_data[name],
                        cardinality="MANY",
                    )
                    for name in sorted(input_names)
                ),
                outputs=tuple(
                    PortSpec(
                        name=name,
                        direction=PortDirection.OUTPUT,
                        data=output_data[name],
                        cardinality="MANY",
                    )
                    for name in sorted(output_names)
                ),
            )
        }
    )


def _stage_flow_calls(
    stage_id: str,
    stage: Stage,
    used_ids: set[str],
) -> dict[str, tuple[str, ComponentNode]]:
    summary = stage.flow_summary()
    if summary is None:
        calls: dict[str, tuple[str, ComponentNode]] = {}
        stage_contract = stage.contract()
        if not isinstance(stage_contract, ComponentContract):
            raise ValidationError("Stage.contract() must return ComponentContract")
        declared_parts = stage_contract.parts
        held = list(_held_components(stage))
        parts: list[tuple[str, object, ComponentContract]] = []
        if declared_parts:
            for part in declared_parts:
                component = getattr(stage, part.name, _MISSING)
                if component is not _MISSING:
                    actual_contract = _component_contract(component)
                    if actual_contract != part.contract:
                        raise ValidationError(
                            f"{type(stage).__name__} component {part.name!r} "
                            "does not match its declared contract"
                        )
                    held = [item for item in held if item.component is not component]
                else:
                    candidates = [
                        item
                        for item in held
                        if item.contract == part.contract
                        and _path_piece(item.path[-1]) == _path_piece(part.name)
                    ]
                    if not candidates:
                        candidates = [
                            item for item in held if item.contract == part.contract
                        ]
                    if not candidates:
                        component = _MISSING
                    else:
                        matched = candidates[0]
                        component = matched.component
                        held.remove(matched)
                parts.append((part.name, component, part.contract))
        else:
            parts.extend(
                (item.path[-1], item.component, item.contract) for item in held
            )
        for name, component, contract in parts:
            if component is _MISSING:
                raise ValidationError(
                    f"{type(stage).__name__} declares missing Stage component {name!r}"
                )
            suffix = _path_piece(name)
            component_id = _unique_id(f"{stage_id}__{suffix}", used_ids)
            calls[name] = (
                component_id,
                ComponentNode(
                    component_id=component_id,
                    component=StagePartNodeAdapter(component, contract),
                    role=None,
                ),
            )
        return calls
    if not isinstance(summary, StageFlowSummary):
        raise ValidationError(
            "Stage.flow_summary() must return StageFlowSummary or None"
        )

    stage_contract = stage.contract()
    parts_by_name = {part.name: part for part in stage_contract.parts}
    calls: dict[str, tuple[str, ComponentNode]] = {}
    for call in summary.calls:
        component = getattr(stage, call.attribute, _MISSING)
        if component is _MISSING:
            raise ValidationError(
                f"{type(stage).__name__} flow call {call.name!r} refers to "
                f"missing attribute {call.attribute!r}"
            )
        contract = _component_contract(component)
        declared_contract = contract
        node_component = component
        if contract is None:
            contract = _summary_call_contract(call, summary)
            node_component = SimpleNamespace(
                original_component=component,
                contract=lambda contract=contract: contract,
            )
        part = parts_by_name.get(call.attribute)
        if declared_contract is not None and call.role not in declared_contract.ports:
            nested_part_mismatch = part is not None and any(
                diagnostic.code == "part_contract_mismatch"
                for diagnostic in check_component_contract(
                    declared_contract, component=component
                )
            )
            if not nested_part_mismatch:
                raise ValidationError(
                    f"{type(stage).__name__} flow call {call.name!r} uses unknown "
                    f"role {call.role!r}"
                )
            flow_contract = _summary_call_contract(call, summary)
            node_component = SimpleNamespace(
                original_component=component,
                contract=lambda contract=declared_contract: contract,
                flow_contract=lambda contract=flow_contract: contract,
            )
        if declared_contract is not None and parts_by_name and part is None:
            raise ValidationError(
                f"{type(stage).__name__} flow call {call.name!r} is not a "
                "declared Stage component"
            )
        if part is not None and part.contract != contract:
            raise ValidationError(
                f"{type(stage).__name__} flow call {call.name!r} does not "
                "match the component contract held by the Stage"
            )
        component_id = _unique_id(f"{stage_id}__{call.name}", used_ids)
        calls[call.name] = (
            component_id,
            ComponentNode(
                component_id=component_id,
                component=node_component,
                role=call.role,
            ),
        )

    missing_parts = set(parts_by_name) - {call.attribute for call in summary.calls}
    if missing_parts:
        raise ValidationError(
            f"{type(stage).__name__} flow summary omits declared components: "
            + ", ".join(sorted(missing_parts))
        )

    for edge in summary.edges:
        if isinstance(edge.source, CallOutput):
            source = calls.get(edge.source.call.name)
            if source is None:
                raise ValidationError("Stage flow edge refers to an unavailable call")
            _call_port(
                source[1], edge.source.call, edge.source.port, PortDirection.OUTPUT
            )
        if isinstance(edge.target, CallInput):
            target = calls.get(edge.target.call.name)
            if target is None:
                raise ValidationError("Stage flow edge refers to an unavailable call")
            _call_port(
                target[1], edge.target.call, edge.target.port, PortDirection.INPUT
            )
    return calls


def _validate_state_edge(stage: Stage, edge: FlowEdge) -> None:
    state = stage.contract().state
    if isinstance(edge.source, StateValue) and edge.source.key not in state.reads:
        raise ValidationError(
            f"{type(stage).__name__} flow reads undeclared state key "
            f"{edge.source.key.namespace}:{edge.source.key.name}"
        )
    if isinstance(edge.target, StateValue) and edge.target.key not in state.writes:
        raise ValidationError(
            f"{type(stage).__name__} flow writes undeclared state key "
            f"{edge.target.key.namespace}:{edge.target.key.name}"
        )


def _linear_state_edges(
    stage_ids: Sequence[str],
    stages: Sequence[Stage],
    calls_by_stage: Sequence[Mapping[str, tuple[str, ComponentNode]]],
    extra_nodes: dict[str, ComponentNode],
    state_flow_inputs: set[tuple[str, str, str]],
) -> list[DataEdge]:
    producers: dict[
        StateKey[object],
        tuple[
            NodeRef | None,
            StateKey[object] | None,
            str,
            tuple[Transform, ...],
        ]
        | None,
    ] = {}
    source_nodes: dict[tuple[StateKey[object], str], ComponentNode] = {}
    for node in extra_nodes.values():
        state_key = getattr(node.component, "state_key", None)
        target_id = getattr(node.component, "target_id", None)
        if isinstance(state_key, StateKey) and isinstance(target_id, str):
            source_nodes[(state_key, target_id)] = node
    edges: list[DataEdge] = []
    for stage_id, stage, calls in zip(stage_ids, stages, calls_by_stage):
        summary = stage.flow_summary()
        if summary is None:
            state = stage.contract().state
            unknown_keys = (
                set(producers) if not state.writes_enumerable else set(state.writes)
            )
            for key in unknown_keys:
                producers[key] = None
            continue
        for item in summary.edges:
            if isinstance(item.source, StateValue) and isinstance(
                item.target, CallInput
            ):
                if (
                    stage_id,
                    item.target.call.name,
                    item.target.port,
                ) in state_flow_inputs:
                    continue
                target_id, target_node = calls[item.target.call.name]
                target_port = _call_port(
                    target_node,
                    item.target.call,
                    item.target.port,
                    PortDirection.INPUT,
                )
                producer = producers.get(item.source.key)
                if producer is None:
                    producer = (None, item.source.key, "value", ())
                source_ref, state_key, source_port, prior_transforms = producer
                transforms = (*prior_transforms, *item.transforms)
                if source_ref is None:
                    source_key = state_key or item.source.key
                    source_node = source_nodes.get((source_key, target_id))
                    if source_node is None:
                        used_ids = {
                            *stage_ids,
                            *(
                                component_id
                                for calls in calls_by_stage
                                for component_id, _node in calls.values()
                            ),
                            *extra_nodes,
                        }
                        source_node = _state_source_node(
                            source_key,
                            target_id,
                            target_port,
                            transforms,
                            used_ids=used_ids,
                        )
                        source_nodes[(source_key, target_id)] = source_node
                        extra_nodes[source_node.component_id] = source_node
                    source_ref = NodeRef(
                        component_id=source_node.component_id,
                        role="state",
                    )
                edges.append(
                    DataEdge(
                        source=source_ref,
                        target=NodeRef(
                            component_id=target_id,
                            role=item.target.call.role,
                        ),
                        source_port=source_port,
                        target_port=target_port.name,
                        transforms=transforms,
                    )
                )
        for item in summary.edges:
            if isinstance(item.source, StateValue) and isinstance(
                item.target, StateValue
            ):
                prior = producers.get(item.source.key)
                if prior is None:
                    prior = (None, item.source.key, "value", ())
                source, state_key, port, transforms = prior
                producers[item.target.key] = (
                    source,
                    state_key or item.source.key,
                    port,
                    (*transforms, *item.transforms),
                )
            elif isinstance(item.source, CallOutput) and isinstance(
                item.target, StateValue
            ):
                call_id, _node = calls[item.source.call.name]
                producers[item.target.key] = (
                    NodeRef(component_id=call_id, role=item.source.call.role),
                    None,
                    item.source.port,
                    item.transforms,
                )
        explicit_writes = {
            item.target.key
            for item in summary.edges
            if isinstance(item.target, StateValue)
        }
        for key in set(stage.contract().state.writes) - explicit_writes:
            producers[key] = (None, key, "value", ())
    return edges


def _decompose_stages(
    stage_ids: Sequence[str],
    stages: Sequence[Stage],
    stage_nodes: Sequence[ComponentNode],
    *,
    resolve_state: bool = True,
    used_ids: set[str] | None = None,
) -> _StageDecomposition:
    """Build verification nodes from Stage summaries without changing execution."""
    used = set(stage_ids) if used_ids is None else used_ids
    calls_by_stage = tuple(
        _stage_flow_calls(stage_id, stage, used)
        for stage_id, stage in zip(stage_ids, stages)
    )
    part_nodes_by_stage = [dict(calls) for calls in calls_by_stage]
    control_edges: list[ControlEdge] = []
    for index, stage_id in enumerate(stage_ids):
        call_ids = tuple(
            component_id for component_id, _node in calls_by_stage[index].values()
        )
        if not call_ids:
            if index + 1 < len(stage_ids):
                control_edges.append(
                    ControlEdge(
                        source=NodeRef(component_id=stage_id),
                        target=NodeRef(component_id=stage_ids[index + 1]),
                    )
                )
            continue
        control_edges.extend(
            ControlEdge(
                source=NodeRef(component_id=stage_id),
                target=NodeRef(component_id=call_id),
            )
            for call_id in call_ids
        )
        if index + 1 < len(stage_ids):
            control_edges.extend(
                ControlEdge(
                    source=NodeRef(component_id=call_id),
                    target=NodeRef(component_id=stage_ids[index + 1]),
                )
                for call_id in call_ids
            )

    extra_nodes_by_id: dict[str, ComponentNode] = {}
    for stage_id, stage, calls in zip(stage_ids, stages, calls_by_stage):
        summary = stage.flow_summary()
        if summary is not None:
            for edge in summary.edges:
                _validate_state_edge(stage, edge)
                if (
                    resolve_state
                    and isinstance(edge.source, StateValue)
                    and isinstance(edge.target, CallInput)
                ):
                    target_id, target_node = calls[edge.target.call.name]
                    target_port = _call_port(
                        target_node,
                        edge.target.call,
                        edge.target.port,
                        PortDirection.INPUT,
                    )
                    source_node = _state_source_node(
                        edge.source.key,
                        target_id,
                        target_port,
                        edge.transforms,
                        used_ids=used,
                    )
                    extra_nodes_by_id[source_node.component_id] = source_node

    data_edges: list[DataEdge] = []
    state_flow_nodes: dict[StateKey[object], ComponentNode] = {}
    state_flow_inputs: set[tuple[str, str, str]] = set()
    for stage_id, stage, calls in zip(stage_ids, stages, calls_by_stage):
        summary = stage.flow_summary()
        if summary is None:
            if resolve_state:
                state = stage.contract().state
                invalidated = (
                    tuple(state_flow_nodes)
                    if not state.writes_enumerable
                    else state.writes
                )
                for key in invalidated:
                    state_flow_nodes.pop(key, None)
            continue
        for edge in summary.edges:
            if isinstance(edge.source, CallOutput) and isinstance(
                edge.target, CallInput
            ):
                source_id, _source_node = calls[edge.source.call.name]
                target_id, _target_node = calls[edge.target.call.name]
                data_edges.append(
                    DataEdge(
                        source=NodeRef(
                            component_id=source_id,
                            role=edge.source.call.role,
                        ),
                        target=NodeRef(
                            component_id=target_id,
                            role=edge.target.call.role,
                        ),
                        source_port=edge.source.port,
                        target_port=edge.target.port,
                        transforms=edge.transforms,
                    )
                )
            elif (
                resolve_state
                and isinstance(edge.source, CallOutput)
                and isinstance(edge.target, StateValue)
            ):
                source_id, source_node = calls[edge.source.call.name]
                source_port = _call_port(
                    source_node,
                    edge.source.call,
                    edge.source.port,
                    PortDirection.OUTPUT,
                )
                data, error = _apply_transforms(source_port.data, edge.transforms)
                if error is not None or data is None:
                    data = source_port.data
                sink = _state_flow_node(
                    edge.target.key,
                    f"{stage_id}__{edge.source.call.name}",
                    data,
                    used_ids=used,
                )
                state_flow_nodes[edge.target.key] = sink
                extra_nodes_by_id[sink.component_id] = sink
                data_edges.append(
                    DataEdge(
                        source=NodeRef(
                            component_id=source_id,
                            role=edge.source.call.role,
                        ),
                        target=NodeRef(component_id=sink.component_id, role="state"),
                        source_port=edge.source.port,
                        target_port="value",
                        transforms=edge.transforms,
                    )
                )
            elif (
                resolve_state
                and isinstance(edge.source, StateValue)
                and isinstance(edge.target, StateValue)
            ):
                source = state_flow_nodes.get(edge.source.key)
                if source is None:
                    continue
                source_data = source.contract.ports["state"].outputs[0].data
                data, error = _apply_transforms(source_data, edge.transforms)
                if error is not None or data is None:
                    data = source_data
                target = _state_flow_node(
                    edge.target.key,
                    f"{stage_id}__state",
                    data,
                    used_ids=used,
                )
                state_flow_nodes[edge.target.key] = target
                extra_nodes_by_id[target.component_id] = target
                data_edges.append(
                    DataEdge(
                        source=NodeRef(component_id=source.component_id, role="state"),
                        target=NodeRef(component_id=target.component_id, role="state"),
                        source_port="value",
                        target_port="value",
                        transforms=edge.transforms,
                    )
                )
            elif (
                resolve_state
                and isinstance(edge.source, StateValue)
                and isinstance(edge.target, CallInput)
            ):
                source = state_flow_nodes.get(edge.source.key)
                if source is None:
                    continue
                target_id, target_node = calls[edge.target.call.name]
                target_port = _call_port(
                    target_node,
                    edge.target.call,
                    edge.target.port,
                    PortDirection.INPUT,
                )
                source_port = source.contract.ports["state"].outputs[0]
                if target_port.name != edge.target.port:
                    raise ValidationError("Stage flow input does not match its port")
                data_edges.append(
                    DataEdge(
                        source=NodeRef(component_id=source.component_id, role="state"),
                        target=NodeRef(
                            component_id=target_id, role=edge.target.call.role
                        ),
                        source_port=source_port.name,
                        target_port=edge.target.port,
                        transforms=edge.transforms,
                    )
                )
                state_flow_inputs.add(
                    (stage_id, edge.target.call.name, edge.target.port)
                )

        if resolve_state:
            explicit_writes = {
                edge.target.key
                for edge in summary.edges
                if isinstance(edge.target, StateValue)
            }
            for key in set(stage.contract().state.writes) - explicit_writes:
                state_flow_nodes.pop(key, None)

    if resolve_state:
        data_edges.extend(
            _linear_state_edges(
                stage_ids,
                stages,
                calls_by_stage,
                extra_nodes_by_id,
                state_flow_inputs,
            )
        )

    used_state_nodes = {
        reference.component_id
        for edge in data_edges
        for reference in (edge.source, edge.target)
        if reference.component_id in extra_nodes_by_id
    }
    extra_nodes_by_id = {
        component_id: node
        for component_id, node in extra_nodes_by_id.items()
        if component_id in used_state_nodes
    }

    bindings: list[StateBinding] = []
    for index, calls in enumerate(calls_by_stage):
        for call_name, (call_id, node) in calls.items():
            for key in node.contract.state.exports:
                if key.namespace != "surrogates":
                    continue
                qualified = StateKey[object](
                    namespace=key.namespace,
                    name=f"{stage_ids[index].replace('.', '__')}:{call_name}",
                    schema_version=key.schema_version,
                )
                bindings.append(
                    StateBinding(
                        node=NodeRef(component_id=call_id, role=node.role),
                        state_key=qualified,
                    )
                )

    return _StageDecomposition(
        part_nodes=tuple(part_nodes_by_stage),
        control_edges=tuple(control_edges),
        data_edges=tuple(data_edges),
        state_bindings=tuple(bindings),
        extra_nodes=tuple(extra_nodes_by_id.values()),
    )


def build_decomposed_component_graph_from_specs(
    specs: Sequence[NodeAdapterSpec],
) -> ComponentGraph:
    """Build a graph from ordered Stage adapters.

    Stage nodes own execution. Declared call and state nodes provide contract
    checks and data-dependency verification without running the held objects.
    """
    specs = tuple(specs)
    if any(not isinstance(spec, NodeAdapterSpec) for spec in specs):
        raise ValidationError("graph specs must contain NodeAdapterSpec values")
    stage_ids = tuple(spec.component_id for spec in specs)
    if len(set(stage_ids)) != len(stage_ids):
        raise ValidationError("graph specs must have unique component_id values")
    stage_adapters = tuple(spec.adapter for spec in specs)
    stage_nodes = tuple(
        ComponentNode(component_id=stage_id, component=adapter)
        for stage_id, adapter in zip(stage_ids, stage_adapters)
    )
    decomposition = _decompose_stages(
        stage_ids,
        tuple(adapter.stage for adapter in stage_adapters),
        stage_nodes,
    )
    for node in decomposition.nodes:
        _mark_verification_only(node)
    return ComponentGraph(
        nodes=(*stage_nodes, *decomposition.nodes),
        data_edges=decomposition.data_edges,
        control_edges=decomposition.control_edges,
        state_bindings=decomposition.state_bindings,
        entry_points=(
            *((NodeRef(component_id=stage_ids[0]),) if stage_ids else ()),
            *(
                NodeRef(component_id=node.component_id, role="state")
                for node in decomposition.extra_nodes
            ),
        ),
    )


def decompose_structured_graph(graph: StructuredGraph) -> StructuredGraph:
    """Add Stage call and state verification nodes to a lowered graph.

    Custom strategies must call this function after ``lower_pipeline`` when
    they want Stage flow edges included in compiler verification.
    """
    sequences: list[tuple[ComponentNode, ...]] = []
    used_ids: set[str] = set()

    def collect(current: StructuredGraph) -> None:
        used_ids.update(node.component_id for node in current.nodes)
        leaves = tuple(
            operation
            for operation in current.operations
            if isinstance(operation, ComponentNode)
            and isinstance(getattr(operation.component, "stage", None), Stage)
        )
        if leaves:
            sequences.append(leaves)
        for region_node in current.region_nodes:
            region = region_node.region
            for body in (region.body, getattr(region, "otherwise", None)):
                if isinstance(body, StructuredGraph):
                    collect(body)

    collect(graph)
    part_nodes: list[ComponentNode] = []
    data_edges: list[DataEdge] = []
    bindings: list[StateBinding] = []
    control_edges: list[ControlEdge] = []
    flow_call_ids: dict[str, dict[str, str]] = {}
    for leaves in sequences:
        stages = tuple(cast(Stage, getattr(node.component, "stage")) for node in leaves)
        decomposition = _decompose_stages(
            tuple(node.component_id for node in leaves),
            stages,
            leaves,
            resolve_state=False,
            used_ids=used_ids,
        )
        part_nodes.extend(decomposition.nodes)
        data_edges.extend(decomposition.data_edges)
        bindings.extend(decomposition.state_bindings)
        control_edges.extend(decomposition.control_edges)
        for node, calls in zip(leaves, decomposition.part_nodes):
            flow_call_ids[node.component_id] = {
                name: component_id for name, (component_id, _call_node) in calls.items()
            }
    if not part_nodes:
        return graph
    control_edges = list(dict.fromkeys((*graph.control_edges, *control_edges)))
    unique_nodes: dict[str, ComponentNode] = {
        node.component_id: node for node in graph.nodes
    }
    unique_nodes.update((node.component_id, node) for node in part_nodes)
    for node in part_nodes:
        _mark_verification_only(node)
    metadata = dict(graph.metadata)
    metadata["stage_flow_calls"] = flow_call_ids
    existing_verification_nodes = metadata.get("verification_only_nodes", ())
    if not isinstance(existing_verification_nodes, (tuple, list)):
        existing_verification_nodes = ()
    metadata["verification_only_nodes"] = tuple(
        dict.fromkeys(
            (
                *existing_verification_nodes,
                *(node.component_id for node in part_nodes),
            )
        )
    )
    return replace(
        graph,
        nodes=tuple(unique_nodes.values()),
        data_edges=(*graph.data_edges, *data_edges),
        control_edges=tuple(control_edges),
        state_bindings=tuple(dict.fromkeys((*graph.state_bindings, *bindings))),
        entry_points=tuple(
            dict.fromkeys(
                (
                    *graph.entry_points,
                    *(
                        NodeRef(component_id=node.component_id, role="state")
                        for node in part_nodes
                        if getattr(node.component, "state_key", None) is not None
                    ),
                )
            )
        ),
        operations=graph.operations,
        metadata=metadata,
    )


def build_decomposed_component_graph_from_stages(
    stages: Sequence[Stage],
) -> ComponentGraph:
    """Compatibility bridge from an ordered Stage sequence to node specs."""
    stages = tuple(stages)
    if any(not isinstance(stage, Stage) for stage in stages):
        raise ValidationError("graph stages must contain Stage values")
    stage_ids = _unique_node_ids(stages)
    specs = tuple(
        NodeAdapterSpec(
            component_id=stage_id,
            adapter=StageContractNodeAdapter(stage, node_path=stage_id),
        )
        for stage_id, stage in zip(stage_ids, stages)
    )
    return build_decomposed_component_graph_from_specs(specs)


def build_decomposed_component_graph(
    value: Pipeline | Sequence[NodeAdapterSpec],
) -> ComponentGraph:
    """Build a graph from node specs, retaining Pipeline compatibility.

    New callers should pass ``NodeAdapterSpec`` values.  ``Pipeline`` is
    accepted solely for the stage compatibility facade.
    """
    if isinstance(value, Pipeline):
        return build_decomposed_component_graph_from_stages(_pipeline_stages(value))
    if not isinstance(value, Sequence):
        raise ValidationError(
            "build_decomposed_component_graph requires node specs or a Pipeline"
        )
    return build_decomposed_component_graph_from_specs(value)
