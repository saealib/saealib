"""Static verification of graph state effects."""

from __future__ import annotations

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
from saealib.exceptions import ValidationError

if TYPE_CHECKING:
    from saealib.core.compiler.compiler import RuleContext, VerificationResult

__all__ = ["StateEffectRule"]


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


def _reads_opaque(node: ComponentNode) -> bool:
    return not node.contract.state.reads_enumerable


def _writes_opaque(node: ComponentNode) -> bool:
    return not node.contract.state.writes_enumerable


def _strictly_before(ref: NodeRef, others: set[NodeRef], order: dict) -> bool:
    """Return whether ``ref`` runs before every node in ``others``."""
    return all(
        other in order.get(ref, set()) and ref not in order.get(other, set())
        for other in others
    )


class StateEffectRule:
    """Verify initialization and ordering of declared state effects."""

    namespace = "core"
    name = "state_effect"
    phase: Literal["verification"] = "verification"

    def apply(self, context: RuleContext) -> VerificationResult:
        """Collect state initialization and ordering diagnostics."""
        from saealib.core.compiler.compiler import VerificationResult

        graph = context.graph
        if isinstance(graph, StructuredGraph):
            return VerificationResult(
                diagnostics=_structured_diagnostics(graph, context)
            )
        order = _reachable(graph)
        refs = tuple(_node_ref(node) for node in graph.nodes)
        initial = set(context.compile_context.initial_state_keys)
        reads: dict[NodeRef, set[StateKey[object]]] = {}
        writes: dict[NodeRef, set[StateKey[object]]] = {}
        opaque_refs = {
            ref
            for node, ref in zip(graph.nodes, refs)
            if _reads_opaque(node) or _writes_opaque(node)
        }
        non_enumerable_reads = {
            ref for node, ref in zip(graph.nodes, refs) if _reads_opaque(node)
        }
        non_enumerable_writes = {
            ref for node, ref in zip(graph.nodes, refs) if _writes_opaque(node)
        }
        for node in graph.nodes:
            ref = _node_ref(node)
            bindings = tuple(
                binding.state_key
                for binding in graph.state_bindings
                if _canonical_ref(graph, binding.node) == ref
            )
            reads[ref] = _resolve_keys(bindings, node.contract.state.reads)
            writes[ref] = _resolve_keys(bindings, node.contract.state.writes)

        findings: list[Diagnostic] = []
        for node, ref in zip(graph.nodes, refs):
            if ref in opaque_refs:
                findings.append(
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

        def add(code: str, node: NodeRef, key: StateKey[object], related=()) -> None:
            findings.append(
                Diagnostic(
                    severity=Severity.ERROR,
                    code=code,
                    message=(
                        f"State key {key.namespace}:{key.name}:{key.schema_version} "
                        f"at {_path(node)} has an invalid effect ordering."
                    ),
                    path=_path(node),
                    related=tuple(_path(item) for item in related),
                    resolutions=(
                        "Add an initial state key or an ordering edge, and keep "
                        "the state declaration node-qualified.",
                    ),
                )
            )

        def verifiable(ref: NodeRef) -> bool:
            # A non-enumerable writer that may run before ``ref`` could
            # initialize any key, so initialization findings stop being
            # definite for ``ref``.
            unverifiable = {other for other in non_enumerable_writes if other != ref}
            return _strictly_before(ref, unverifiable, order)

        for ref in refs:
            if ref in non_enumerable_reads:
                continue
            for key in reads[ref]:
                writers = [
                    other for other in refs if key in writes[other] and other != ref
                ]
                if key in initial:
                    continue
                if key in writes[ref] and not writers:
                    if verifiable(ref):
                        add("uninitialized_state_write", ref, key)
                    continue
                if any(ref in order.get(other, set()) for other in writers):
                    continue
                if writers and all(other in order.get(ref, set()) for other in writers):
                    code = "unreachable_state_read"
                    if not verifiable(ref):
                        continue
                else:
                    code = (
                        "concurrent_state_read_write"
                        if writers
                        else "unreachable_state_read"
                    )
                    if not writers and not verifiable(ref):
                        continue
                add(code, ref, key, writers)
        for index, left in enumerate(refs):
            for right in refs[index + 1 :]:
                for key in writes[left] & writes[right]:
                    # A non-enumerable writer may also write keys outside its
                    # declaration, but keys it does declare are conclusive
                    # evidence for this check.
                    if right not in order.get(left, set()) and left not in order.get(
                        right, set()
                    ):
                        add("concurrent_state_write", left, key, (right,))
        return VerificationResult(diagnostics=tuple(findings))


def _structured_diagnostics(
    graph: StructuredGraph, context: RuleContext
) -> tuple[Diagnostic, ...]:
    """Check structured operations in execution order."""
    initial = set(context.compile_context.initial_state_keys)
    bindings_by_node: dict[NodeRef, tuple[StateKey[object], ...]] = {}
    for node in graph.nodes:
        ref = _node_ref(node)
        bindings_by_node[ref] = tuple(
            binding.state_key
            for binding in graph.state_bindings
            if _canonical_ref(graph, binding.node) == ref
        )

    universe = set(initial)
    for node in graph.nodes:
        bindings = bindings_by_node[_node_ref(node)]
        state = node.contract.state
        universe.update(_resolve_keys(bindings, state.reads))
        universe.update(_resolve_keys(bindings, state.writes))
        universe.update(_resolve_keys(bindings, state.exports))

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
    opaque_refs = {
        _node_ref(node)
        for node in graph.nodes
        if _reads_opaque(node) or _writes_opaque(node)
    }
    non_enumerable_reads = {
        _node_ref(node) for node in graph.nodes if _reads_opaque(node)
    }
    non_enumerable_writes = {
        _node_ref(node) for node in graph.nodes if _writes_opaque(node)
    }
    reads: dict[NodeRef, set[StateKey[object]]] = {}
    writes: dict[NodeRef, set[StateKey[object]]] = {}
    for node in graph.nodes:
        ref = _node_ref(node)
        bindings = bindings_by_node[ref]
        state = node.contract.state
        reads[ref] = _resolve_keys(bindings, state.reads)
        writes[ref] = _resolve_keys(bindings, state.writes)

    findings: list[Diagnostic] = []
    for node in graph.nodes:
        ref = _node_ref(node)
        if ref not in opaque_refs:
            continue
        findings.append(
            Diagnostic(
                severity=Severity.WARNING,
                code="opaque_state_effects",
                message=(
                    f"Node {_path(ref)} declares its state effects "
                    "non-enumerable, so its state reads and writes cannot "
                    "be verified."
                ),
                path=_path(ref),
                resolutions=(
                    "Declare the state effects in the component's contract, "
                    "or accept that they are not verified.",
                ),
            )
        )

    def add(code: str, node: NodeRef, key: StateKey[object]) -> None:
        findings.append(
            Diagnostic(
                severity=Severity.ERROR,
                code=code,
                message=(
                    f"State key {key.namespace}:{key.name}:{key.schema_version} "
                    f"at {_path(node)} has an invalid effect ordering."
                ),
                path=_path(node),
                resolutions=(
                    "Add an initial state key or an ordering edge, and keep "
                    "the state declaration node-qualified.",
                ),
            )
        )

    def check_reads(node: NodeRef, available: set[StateKey[object]]) -> None:
        for key in sorted(reads.get(node, ()), key=str):
            if key not in available:
                add(
                    "uninitialized_state_write"
                    if key in writes.get(node, ())
                    else "unreachable_state_read",
                    node,
                    key,
                )

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
                if ref not in non_enumerable_reads:
                    check_reads(ref, current_available)
                if ref in non_enumerable_writes:
                    # Unknown writes may touch any key, so every key becomes
                    # potentially available after the node runs.
                    current_available.update(universe)
                else:
                    current_available.update(writes[ref])
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

    sequence(graph, initial)
    return tuple(findings)


def _structured_body(region: StructuredRegion) -> StructuredGraph:
    body = region.body
    if not isinstance(body, StructuredGraph):
        raise ValidationError(
            f"Structured region {region.qualified_id!r} body must be lowered"
        )
    return body
