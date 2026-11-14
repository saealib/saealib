"""Static verification of graph state effects."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from saealib.core.compiler.diagnostics import ContractPath, Diagnostic, Severity
from saealib.core.compiler.graph import ComponentGraph, ComponentNode, NodeRef
from saealib.core.compiler.regions import (
    BranchRegion,
    LoopRegion,
    RegionNode,
    RepeatRegion,
    StructuredRegion,
)
from saealib.core.compiler.structured import StructuredGraph
from saealib.core.state.keys import StateKey
from saealib.core.state_inference import InferenceStatus
from saealib.exceptions import ValidationError

if TYPE_CHECKING:
    from saealib.core.compiler.compiler import RuleContext, VerificationResult

__all__ = [
    "NodeStateEffects",
    "StateEffectFinding",
    "StateEffectRule",
    "compute_state_effect_findings",
    "declared_state_effects",
    "resolved_inferred_state_effects",
]


def _path(node: NodeRef) -> ContractPath:
    return ContractPath(components=(node.component_id,), role=node.role)


def _node_ref(node: ComponentNode) -> NodeRef:
    return NodeRef(component_id=node.component_id, role=node.role)


def _canonical_ref(graph: ComponentGraph, reference: NodeRef) -> NodeRef:
    try:
        node = graph.node_by_id(reference.component_id)
    except KeyError:
        return NodeRef(component_id=reference.component_id)
    return _node_ref(node)


def _bindings(
    graph: ComponentGraph, reference: NodeRef
) -> tuple[StateKey[object], ...]:
    return tuple(
        binding.state_key
        for binding in graph.state_bindings
        if _canonical_ref(graph, binding.node) == reference
    )


def _reachable(graph: ComponentGraph) -> dict[NodeRef, set[NodeRef]]:
    successors: dict[NodeRef, set[NodeRef]] = {
        _node_ref(node): set() for node in graph.nodes
    }
    for edge in (*graph.data_edges, *graph.control_edges):
        source = _canonical_ref(graph, edge.source)
        target = _canonical_ref(graph, edge.target)
        if source in successors and target in successors:
            successors[source].add(target)
    for source in tuple(successors):
        pending = list(successors[source])
        while pending:
            target = pending.pop()
            for successor in successors.get(target, ()):
                if successor not in successors[source]:
                    successors[source].add(successor)
                    pending.append(successor)
    return successors


def _resolve_keys(
    bindings: tuple[StateKey[object], ...],
    keys: tuple[StateKey[object], ...],
) -> set[StateKey[object]]:
    resolved: set[StateKey[object]] = set()
    for key in keys:
        candidates = [
            binding for binding in bindings if binding.namespace == key.namespace
        ]
        exact = [binding for binding in candidates if binding.name == key.name]
        if len(exact) == 1:
            resolved.add(exact[0])
        elif len(candidates) == 1:
            resolved.add(candidates[0])
        else:
            resolved.add(key)
    return resolved


def _unknown_stage_flow(node: ComponentNode) -> bool:
    stage = getattr(node.component, "stage", None)
    flow = getattr(stage, "flow_summary", None)
    return stage is not None and callable(flow) and flow() is None


def _strictly_before(ref: NodeRef, others: set[NodeRef], order: dict) -> bool:
    """Return whether ``ref`` runs before every node in ``others``."""
    return all(
        other in order.get(ref, set()) and ref not in order.get(other, set())
        for other in others
    )


@dataclass(frozen=True, kw_only=True)
class NodeStateEffects:
    """Resolved state effects used by one verification run."""

    reads: frozenset[StateKey[object]] = frozenset()
    writes: frozenset[StateKey[object]] = frozenset()
    exports: frozenset[StateKey[object]] = frozenset()
    reads_enumerable: bool = True
    writes_enumerable: bool = True


@dataclass(frozen=True, kw_only=True)
class StateEffectFinding:
    """One ordering conflict, independent of its diagnostic severity."""

    node: NodeRef
    key: StateKey[object]
    kind: str
    related: tuple[NodeRef, ...] = ()


def declared_state_effects(
    graph: ComponentGraph,
) -> dict[NodeRef, NodeStateEffects]:
    """Return the contract-declared effects for every node."""
    effects: dict[NodeRef, NodeStateEffects] = {}
    for node in graph.nodes:
        ref = _node_ref(node)
        bindings = _bindings(graph, ref)
        state = node.analysis_contract.state
        effects[ref] = NodeStateEffects(
            reads=frozenset(_resolve_keys(bindings, state.reads)),
            writes=frozenset(_resolve_keys(bindings, state.writes)),
            exports=frozenset(_resolve_keys(bindings, state.exports)),
            reads_enumerable=state.reads_enumerable,
            writes_enumerable=state.writes_enumerable,
        )
    return effects


def resolved_inferred_state_effects(
    graph: ComponentGraph,
    declared: dict[NodeRef, NodeStateEffects],
) -> dict[NodeRef, NodeStateEffects]:
    """Return declared effects with inferred effects substituted per node."""
    effects = dict(declared)
    for node in graph.nodes:
        inference = node._inferred_effects
        ref = _node_ref(node)
        if inference.status is InferenceStatus.INFERRED:
            bindings = _bindings(graph, ref)
            assert inference.effects is not None
            effects[ref] = NodeStateEffects(
                reads=frozenset(_resolve_keys(bindings, inference.effects.reads)),
                writes=frozenset(_resolve_keys(bindings, inference.effects.writes)),
                exports=declared[ref].exports,
                reads_enumerable=True,
                writes_enumerable=True,
            )
        elif inference.status is InferenceStatus.UNKNOWN:
            effects[ref] = NodeStateEffects(
                reads=frozenset(),
                writes=frozenset(),
                exports=declared[ref].exports,
                reads_enumerable=False,
                writes_enumerable=False,
            )
    return effects


def compute_state_effect_findings(
    graph: ComponentGraph,
    initial: frozenset[StateKey[object]],
    effects: dict[NodeRef, NodeStateEffects],
) -> tuple[StateEffectFinding, ...]:
    """Return ordering conflicts for one set of per-node state effects."""
    if isinstance(graph, StructuredGraph):
        return _structured_findings(graph, initial, effects)
    return _sequential_findings(graph, initial, effects)


def _sequential_findings(
    graph: ComponentGraph,
    initial: frozenset[StateKey[object]],
    effects: dict[NodeRef, NodeStateEffects],
) -> tuple[StateEffectFinding, ...]:
    order = _reachable(graph)
    refs = tuple(_node_ref(node) for node in graph.nodes)
    reads = {ref: effects[ref].reads for ref in refs}
    writes = {ref: effects[ref].writes for ref in refs}
    non_enumerable_reads = {ref for ref in refs if not effects[ref].reads_enumerable}
    non_enumerable_writes = {ref for ref in refs if not effects[ref].writes_enumerable}
    initial_keys = set(initial)
    findings: list[StateEffectFinding] = []

    def verifiable(ref: NodeRef) -> bool:
        unverifiable = {other for other in non_enumerable_writes if other != ref}
        return _strictly_before(ref, unverifiable, order)

    for ref in refs:
        if ref in non_enumerable_reads:
            continue
        for key in reads[ref]:
            writers = [other for other in refs if key in writes[other] and other != ref]
            if key in initial_keys:
                continue
            if key in writes[ref] and not writers:
                if verifiable(ref):
                    findings.append(
                        StateEffectFinding(
                            node=ref, key=key, kind="uninitialized_state_write"
                        )
                    )
                continue
            if any(ref in order.get(other, set()) for other in writers):
                continue
            if writers and all(other in order.get(ref, set()) for other in writers):
                kind = "unreachable_state_read"
                if not verifiable(ref):
                    continue
            else:
                kind = (
                    "concurrent_state_read_write"
                    if writers
                    else "unreachable_state_read"
                )
                if not writers and not verifiable(ref):
                    continue
            findings.append(
                StateEffectFinding(node=ref, key=key, kind=kind, related=tuple(writers))
            )
    for index, left in enumerate(refs):
        for right in refs[index + 1 :]:
            for key in writes[left] & writes[right]:
                if right not in order.get(left, set()) and left not in order.get(
                    right, set()
                ):
                    findings.append(
                        StateEffectFinding(
                            node=left,
                            key=key,
                            kind="concurrent_state_write",
                            related=(right,),
                        )
                    )
    return tuple(findings)


def _structured_findings(
    graph: StructuredGraph,
    initial: frozenset[StateKey[object]],
    effects: dict[NodeRef, NodeStateEffects],
) -> tuple[StateEffectFinding, ...]:
    initial_keys = set(initial)
    nodes_by_id = {node.component_id: node for node in graph.nodes}
    held_nodes_by_stage: dict[str, tuple[ComponentNode, ...]] = {}
    stage_flow_calls = graph.metadata.get("stage_flow_calls", {})
    if isinstance(stage_flow_calls, dict):
        for stage_id, calls in stage_flow_calls.items():
            stage_node = nodes_by_id.get(stage_id)
            if stage_node is None:
                continue
            stage = getattr(stage_node.component, "stage", None)
            if (
                stage is None
                or not callable(getattr(stage, "flow_summary", None))
                or stage.flow_summary() is not None
                or not isinstance(calls, dict)
            ):
                continue
            held_nodes_by_stage[stage_id] = tuple(
                nodes_by_id[component_id]
                for component_id in calls.values()
                if component_id in nodes_by_id
            )

    universe = set(initial_keys)
    for node in graph.nodes:
        effect = effects[_node_ref(node)]
        universe.update(effect.reads)
        universe.update(effect.writes)
        universe.update(effect.exports)

    def visit_regions(current: StructuredGraph) -> None:
        for operation in current.operations:
            if not isinstance(operation, RegionNode):
                continue
            region = operation.region
            if isinstance(region, (LoopRegion, BranchRegion)):
                universe.update(region.condition.contract().reads)
            universe.update(region.effect.reads)
            universe.update(region.effect.writes)
            visit_regions(_structured_body(region))
            otherwise = getattr(region, "otherwise", None)
            if isinstance(otherwise, StructuredGraph):
                visit_regions(otherwise)

    visit_regions(graph)
    non_enumerable_reads = {ref for ref in effects if not effects[ref].reads_enumerable}
    non_enumerable_writes = {
        ref for ref in effects if not effects[ref].writes_enumerable
    }
    findings: list[StateEffectFinding] = []

    def add(kind: str, node: NodeRef, key: StateKey[object]) -> None:
        findings.append(StateEffectFinding(node=node, key=key, kind=kind))

    def check_reads(node: NodeRef, available: set[StateKey[object]]) -> None:
        effect = effects[node]
        for key in sorted(effect.reads, key=str):
            if key not in available:
                add(
                    "uninitialized_state_write"
                    if key in effect.writes
                    else "unreachable_state_read",
                    node,
                    key,
                )

    def apply_node_effects(node: NodeRef, available: set[StateKey[object]]) -> None:
        if node not in non_enumerable_reads:
            check_reads(node, available)
        if node in non_enumerable_writes:
            available.update(universe)
        else:
            available.update(effects[node].writes)

    def check_condition(
        region: LoopRegion | BranchRegion, available: set[StateKey[object]]
    ) -> None:
        for key in sorted(region.condition.contract().reads, key=str):
            if key not in available:
                add(
                    "unreachable_state_read",
                    NodeRef(component_id=region.qualified_id),
                    key,
                )

    def sequence(
        current: StructuredGraph, available: set[StateKey[object]]
    ) -> set[StateKey[object]]:
        current_available = set(available)
        for operation in current.operations:
            if isinstance(operation, ComponentNode):
                ref = _node_ref(operation)
                apply_node_effects(ref, current_available)
                for held_node in held_nodes_by_stage.get(operation.component_id, ()):
                    apply_node_effects(_node_ref(held_node), current_available)
                continue
            region = operation.region
            body = _structured_body(region)
            if isinstance(region, RepeatRegion):
                if not isinstance(region.count, int):
                    sequence(body, current_available)
                    continue
                if region.count > 0:
                    body_available = sequence(body, current_available)
                    current_available.update(body_available)
                    current_available.update(region.effect.writes)
                continue
            if isinstance(region, LoopRegion):
                check_condition(region, current_available)
                sequence(body, current_available)
                continue
            if isinstance(region, BranchRegion):
                check_condition(region, current_available)
                then_available = sequence(body, current_available)
                otherwise = region.otherwise
                else_available = (
                    sequence(otherwise, current_available)
                    if isinstance(otherwise, StructuredGraph)
                    else set(current_available)
                )
                current_available.update(then_available & else_available)
                continue
            current_available.update(sequence(body, current_available))
        return current_available

    sequence(graph, initial_keys)
    return tuple(findings)


class StateEffectRule:
    """Verify initialization and ordering of declared state effects."""

    namespace = "core"
    name = "state_effect"
    phase: Literal["verification"] = "verification"

    def apply(self, context: RuleContext) -> VerificationResult:
        """Collect state initialization and ordering diagnostics."""
        from saealib.core.compiler.compiler import VerificationResult

        graph = context.graph
        effects = declared_state_effects(graph)
        findings = compute_state_effect_findings(
            graph, context.compile_context.initial_state_keys, effects
        )
        diagnostics: list[Diagnostic] = []
        for node in graph.nodes:
            ref = _node_ref(node)
            effect = effects[ref]
            if (
                effect.reads_enumerable and effect.writes_enumerable
            ) or _unknown_stage_flow(node):
                continue
            diagnostics.append(
                Diagnostic(
                    severity=Severity.WARNING,
                    code="opaque_state_effects",
                    message=(
                        f"Node {_path(ref)} declares its state effects "
                        "non-enumerable, so its state reads and writes "
                        "cannot be verified."
                    ),
                    path=_path(ref),
                    resolutions=(
                        "Declare the state effects in the component's "
                        "contract, or accept that they are not verified.",
                    ),
                )
            )
        for finding in findings:
            diagnostics.append(
                Diagnostic(
                    severity=Severity.ERROR,
                    code=finding.kind,
                    message=(
                        f"State key {finding.key.namespace}:{finding.key.name}:"
                        f"{finding.key.schema_version} at {_path(finding.node)} "
                        "has an invalid effect ordering."
                    ),
                    path=_path(finding.node),
                    related=tuple(_path(item) for item in finding.related),
                    resolutions=(
                        "Add an initial state key or an ordering edge, and keep "
                        "the state declaration node-qualified.",
                    ),
                )
            )
        return VerificationResult(diagnostics=tuple(diagnostics))


def _structured_body(region: StructuredRegion) -> StructuredGraph:
    body = region.body
    if not isinstance(body, StructuredGraph):
        raise ValidationError(
            f"Structured region {region.qualified_id!r} body must be lowered"
        )
    return body
