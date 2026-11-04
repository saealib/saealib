"""Build component graphs from stage-based compatibility pipelines.

The graph builder is deliberately a bridge.  ``Stage`` remains the execution
object used by the existing optimizer, while ``StageNodeAdapter`` gives the
compiler vocabulary a component-shaped view of the components a stage holds.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from itertools import pairwise
from typing import Any, cast

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
    ExecutionContract,
    LifecycleContract,
    PortContract,
    PortDirection,
    StateContract,
)
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

_MAX_CACHED_EXECUTION_UNWRAPS = 8
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


def cached_execution_target(component: object) -> object:
    """Return the cached target for a Stage adapter hidden by freshening.

    Schema freshening is deliberately transparent at the component boundary,
    but its ``execute`` fallback would otherwise call the adapter wrapper on
    every execution.  Only the known freshening wrapper is traversed; custom
    components retain normal dynamic ``execute`` dispatch.  The bound is a
    guard against malformed wrapper cycles or unexpectedly deep nesting.
    """
    current = component
    for _ in range(_MAX_CACHED_EXECUTION_UNWRAPS):
        if isinstance(current, StageNodeAdapter):
            return current._execute_target
        if not getattr(current, "_saealib_schema_freshened", False):
            break
        nested = getattr(current, "_component", _MISSING)
        if nested is _MISSING:
            break
        current = nested
    else:
        raise ValidationError(
            "Stage execution target wrapper nesting exceeds the supported bound"
        )
    return getattr(component, "execute", None)


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
        self._held = _held_components(stage)
        self._contract = stage.contract()

    def contract(self) -> ComponentContract:
        """Return the direct Stage contract, excluding held parts."""
        return self._contract


class StagePartNodeAdapter:
    """Expose one declared Stage part as an independently discoverable node.

    Parts deliberately do not implement ``execute``. The Stage remains
    the executable owner until the runtime migration unit; these nodes provide
    the contract and data-dependency identity without executing a component a
    second time.
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


_DATA_PORTS: tuple[tuple[str, str, str, str, str, str], ...] = (
    ("ask", "proposer", "genomes", "surrogate_predict", "predictor", "candidates"),
    (
        "ask",
        "proposer",
        "genomes",
        "evaluation_plan",
        "evaluation_planner",
        "candidates",
    ),
    (
        "ask",
        "proposer",
        "genomes",
        "async_evaluation_submit",
        "evaluation_planner",
        "candidates",
    ),
    (
        "surrogate_predict",
        "predictor",
        "prediction",
        "acquisition",
        "acquisition",
        "prediction",
    ),
    (
        "acquisition",
        "acquisition",
        "scores",
        "evaluation_plan",
        "evaluation_planner",
        "acquisition",
    ),
    (
        "acquisition",
        "acquisition",
        "scores",
        "async_evaluation_submit",
        "evaluation_planner",
        "acquisition",
    ),
    (
        "feedback",
        "feedback_builder",
        "feedback",
        "tell",
        "feedback_consumer",
        "feedback",
    ),
)


def _has_port(
    contract: ComponentContract,
    role: str,
    name: str,
    direction: PortDirection,
) -> bool:
    port_contract = contract.ports.get(role)
    if port_contract is None:
        return False
    matches = tuple(
        port
        for port in (*port_contract.inputs, *port_contract.outputs)
        if port.name == name and port.direction is direction
    )
    return len(matches) == 1


def _unique_node_ids(stages: Sequence[Stage]) -> tuple[str, ...]:
    counts: dict[str, int] = {}
    ids: list[str] = []
    for index, stage in enumerate(stages):
        base = stage.name or f"stage_{index}"
        count = counts.get(base, 0) + 1
        counts[base] = count
        ids.append(base if count == 1 else f"{base}_{count}")
    return tuple(ids)


def _pipeline_stages(pipeline: Pipeline) -> tuple[Stage, ...]:
    """Validate the Pipeline-to-Stage compatibility bridge boundary."""
    stages = tuple(pipeline.stages)
    if any(not isinstance(stage, Stage) for stage in stages):
        raise ValidationError("Pipeline stages must contain Stage values")
    return tuple(stage for stage in stages if isinstance(stage, Stage))


def build_component_graph(pipeline: Pipeline) -> ComponentGraph:
    """Build a graph with the same top-level stages as *pipeline*.

    Adjacent stages always receive a control edge. Data edges are emitted only
    when both endpoints expose the declared role, name, and direction.
    """
    if not isinstance(pipeline, Pipeline):
        raise ValidationError("build_component_graph requires a Pipeline")
    stages = _pipeline_stages(pipeline)
    node_ids = _unique_node_ids(stages)
    adapters = tuple(
        StageNodeAdapter(stage, node_path=node_id)
        for stage, node_id in zip(stages, node_ids)
    )
    nodes = tuple(
        ComponentNode(component_id=node_id, component=adapter)
        for node_id, adapter in zip(node_ids, adapters)
    )
    control_edges = tuple(
        ControlEdge(
            source=NodeRef(component_id=node_ids[index]),
            target=NodeRef(component_id=node_ids[index + 1]),
        )
        for index in range(len(node_ids) - 1)
    )

    data_edges: list[DataEdge] = []
    for (
        source_name,
        source_role,
        source_port,
        target_name,
        target_role,
        target_port,
    ) in _DATA_PORTS:
        source_indices = [
            index for index, stage in enumerate(stages) if stage.name == source_name
        ]
        target_indices = [
            index for index, stage in enumerate(stages) if stage.name == target_name
        ]
        for source_index in source_indices:
            for target_index in target_indices:
                if source_index < target_index:
                    source_node = nodes[source_index]
                    target_node = nodes[target_index]
                    if not (
                        _has_port(
                            source_node.contract,
                            source_role,
                            source_port,
                            PortDirection.OUTPUT,
                        )
                        and _has_port(
                            target_node.contract,
                            target_role,
                            target_port,
                            PortDirection.INPUT,
                        )
                    ):
                        continue
                    data_edges.append(
                        DataEdge(
                            source=NodeRef(
                                component_id=node_ids[source_index], role=source_role
                            ),
                            target=NodeRef(
                                component_id=node_ids[target_index], role=target_role
                            ),
                            source_port=source_port,
                            target_port=target_port,
                        )
                    )

    bindings: list[StateBinding] = []
    for node_id, adapter in zip(node_ids, adapters):
        bindings.extend(adapter.state_bindings(node_id))

    return ComponentGraph(
        nodes=nodes,
        data_edges=tuple(data_edges),
        control_edges=control_edges,
        state_bindings=tuple(bindings),
        entry_points=(NodeRef(component_id=node_ids[0]),) if node_ids else (),
    )


def _declared_part_component(stage: Stage, name: str) -> object | None:
    for candidate in (f"_{name}", name):
        if hasattr(stage, candidate):
            return getattr(stage, candidate)
    return None


def _decomposed_role_node(
    stage_node: str,
    role: str,
    part_nodes: Mapping[str, tuple[str, ComponentNode]],
    stage: ComponentNode,
    direction: PortDirection,
    port: str,
) -> NodeRef | None:
    declared_matches: list[NodeRef] = []
    held_matches: list[NodeRef] = []
    for part_name, (part_node, node) in part_nodes.items():
        if _has_port(node.contract, role, port, direction):
            target = (
                declared_matches if not part_name.startswith("held_") else held_matches
            )
            target.append(NodeRef(component_id=part_node, role=role))
    if declared_matches:
        return declared_matches[0] if len(declared_matches) == 1 else None
    if held_matches:
        return held_matches[0] if len(held_matches) == 1 else None
    if _has_port(stage.contract, role, port, direction):
        return NodeRef(component_id=stage_node, role=role)
    return None


@dataclass(frozen=True)
class _StageDecomposition:
    """Part nodes, part edges, and part bindings of an ordered Stage sequence.

    ``part_nodes[i]`` maps part names of the ``i``-th Stage to their
    ``(node id, node)`` pairs in declaration order.  ``control_edges`` chain
    each Stage through its own parts to the next Stage; ``data_edges`` connect
    the declared ports of parts held by an earlier Stage to a later one.
    """

    part_nodes: tuple[Mapping[str, tuple[str, ComponentNode]], ...]
    control_edges: tuple[ControlEdge, ...]
    data_edges: tuple[DataEdge, ...]
    state_bindings: tuple[StateBinding, ...]

    @property
    def nodes(self) -> tuple[ComponentNode, ...]:
        """Every part node, Stage by Stage."""
        return tuple(
            node for parts in self.part_nodes for _part_id, node in parts.values()
        )


def _decompose_stages(
    stage_ids: Sequence[str],
    stages: Sequence[Stage],
    stage_nodes: Sequence[ComponentNode],
    held_by_stage: Sequence[tuple[_HeldComponent, ...]],
) -> _StageDecomposition:
    """Expand ordered Stage nodes into ``(component, role)`` part nodes.

    Each Stage stays the sole executable node of its parts.  Every declared
    part gets its own contract-bearing node; a held component that the
    Stage contract does not declare is exposed as a ``held_N`` part so its
    ports stay visible to the compiler.
    """
    part_nodes_by_stage: list[dict[str, tuple[str, ComponentNode]]] = []
    for stage_id, stage, stage_node, held in zip(
        stage_ids, stages, stage_nodes, held_by_stage
    ):
        declared = stage_node.contract
        stage_parts: dict[str, tuple[str, ComponentNode]] = {}
        declared_components: set[int] = set()
        for part in declared.parts:
            component = _declared_part_component(stage, part.name)
            if component is None:
                if part.optional:
                    continue
                raise ValidationError(
                    f"{type(stage).__name__} declares missing part {part.name!r}"
                )
            part_id = f"{stage_id}__{part.name}"
            part_node = ComponentNode(
                component_id=part_id,
                component=StagePartNodeAdapter(component, part.contract),
            )
            stage_parts[part.name] = (part_id, part_node)
            declared_components.add(id(component))
        for item_index, item in enumerate(held, start=1):
            if id(item.component) in declared_components:
                continue
            part_name = f"held_{item_index}"
            while part_name in stage_parts:
                part_name = f"{part_name}_1"
            part_id = f"{stage_id}__{part_name}"
            part_node = ComponentNode(
                component_id=part_id,
                component=StagePartNodeAdapter(item.component, item.contract),
            )
            stage_parts[part_name] = (part_id, part_node)
        part_nodes_by_stage.append(stage_parts)

    control_edges: list[ControlEdge] = []
    for index, stage_id in enumerate(stage_ids):
        owned = tuple(part_nodes_by_stage[index].values())
        chain = [stage_id, *(part_id for part_id, _ in owned)]
        if index + 1 < len(stage_ids):
            chain.append(stage_ids[index + 1])
        control_edges.extend(
            ControlEdge(
                source=NodeRef(component_id=source),
                target=NodeRef(component_id=target),
            )
            for source, target in pairwise(chain)
        )

    data_edges: list[DataEdge] = []
    for (
        source_name,
        source_role,
        source_port,
        target_name,
        target_role,
        target_port,
    ) in _DATA_PORTS:
        for source_index, source_stage in enumerate(stages):
            if source_stage.name != source_name:
                continue
            for target_index, target_stage in enumerate(stages):
                if target_stage.name != target_name or source_index >= target_index:
                    continue
                source_ref = _decomposed_role_node(
                    stage_ids[source_index],
                    source_role,
                    part_nodes_by_stage[source_index],
                    stage_nodes[source_index],
                    PortDirection.OUTPUT,
                    source_port,
                )
                target_ref = _decomposed_role_node(
                    stage_ids[target_index],
                    target_role,
                    part_nodes_by_stage[target_index],
                    stage_nodes[target_index],
                    PortDirection.INPUT,
                    target_port,
                )
                if source_ref is not None and target_ref is not None:
                    data_edges.append(
                        DataEdge(
                            source=source_ref,
                            target=target_ref,
                            source_port=source_port,
                            target_port=target_port,
                        )
                    )

    bindings: list[StateBinding] = []
    seen_bindings: set[tuple[str, StateKey[object]]] = set()
    for index, held in enumerate(held_by_stage):
        for part_id, part_node in part_nodes_by_stage[index].values():
            part_adapter = cast(StagePartNodeAdapter, part_node.component)
            component = part_adapter.component
            for item in held:
                if item.component is not component:
                    continue
                for key in item.contract.state.exports:
                    if key.namespace == "surrogates":
                        qualified = StateKey[object](
                            namespace=key.namespace,
                            name=_surrogate_key_name(stage_ids[index], item),
                            schema_version=key.schema_version,
                        )
                        marker = (part_id, qualified)
                        if marker in seen_bindings:
                            continue
                        seen_bindings.add(marker)
                        bindings.append(
                            StateBinding(
                                node=NodeRef(component_id=part_id),
                                state_key=qualified,
                            )
                        )

    return _StageDecomposition(
        part_nodes=tuple(part_nodes_by_stage),
        control_edges=tuple(control_edges),
        data_edges=tuple(data_edges),
        state_bindings=tuple(bindings),
    )


def build_decomposed_component_graph_from_specs(
    specs: Sequence[NodeAdapterSpec],
) -> ComponentGraph:
    """Build the canonical graph from ordered component/adapter specs.

    ``build_component_graph`` remains the Stage bridge.  This function is an
    opt-in decomposed path: each Stage is retained as the sole executable node,
    while every declared part gets its own contract-bearing node.  Part nodes are
    connected into the Stage control chain so ordering remains unique; data
    edges target the part that owns the declared port and never encode control
    side effects.
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
        tuple(adapter._held for adapter in stage_adapters),
    )
    return ComponentGraph(
        nodes=(*stage_nodes, *decomposition.nodes),
        data_edges=decomposition.data_edges,
        control_edges=decomposition.control_edges,
        state_bindings=decomposition.state_bindings,
        entry_points=(NodeRef(component_id=stage_ids[0]),) if stage_ids else (),
    )


def decompose_structured_graph(graph: StructuredGraph) -> StructuredGraph:
    """Add the ``(component, role)`` part nodes of a lowered structured graph.

    Every Stage the graph runs is expanded as
    :func:`build_decomposed_component_graph_from_specs` expands a Stage
    sequence: its parts are chained between the Stage and the Stage's control
    successors, and data edges join the declared ports of Stages that run in
    the same sequence.  Operations are left unchanged, so the Stages stay the
    executable nodes and the part nodes are verification-only.

    Data edges are not added between Stages in different sequences, such as
    a Stage inside a region and a Stage in the enclosing sequence, so the
    port contracts between them are not checked.
    """
    sequences: list[tuple[ComponentNode, ...]] = []

    def collect(current: StructuredGraph) -> None:
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
    chains: dict[str, tuple[str, ...]] = {}
    for leaves in sequences:
        stages = tuple(cast(Stage, getattr(node.component, "stage")) for node in leaves)
        decomposition = _decompose_stages(
            tuple(node.component_id for node in leaves),
            stages,
            leaves,
            tuple(_held_components(stage) for stage in stages),
        )
        part_nodes.extend(decomposition.nodes)
        data_edges.extend(decomposition.data_edges)
        bindings.extend(decomposition.state_bindings)
        for node, parts in zip(leaves, decomposition.part_nodes):
            if parts:
                chains[node.component_id] = tuple(
                    part_id for part_id, _ in parts.values()
                )
    if not part_nodes:
        return graph
    control_edges: list[ControlEdge] = []
    for edge in graph.control_edges:
        chain = chains.get(edge.source.component_id)
        if chain is None:
            control_edges.append(edge)
        else:
            control_edges.append(replace(edge, source=NodeRef(component_id=chain[-1])))
    for stage_id, chain in chains.items():
        control_edges.extend(
            ControlEdge(
                source=NodeRef(component_id=source),
                target=NodeRef(component_id=target),
            )
            for source, target in pairwise((stage_id, *chain))
        )
    return replace(
        graph,
        nodes=(*graph.nodes, *part_nodes),
        data_edges=(*graph.data_edges, *data_edges),
        control_edges=tuple(control_edges),
        state_bindings=tuple(dict.fromkeys((*graph.state_bindings, *bindings))),
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
