"""Resolution of port dataflow in structured graphs."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace

from saealib.core.compiler.diagnostics import (
    ContractPath,
    Diagnostic,
    Severity,
)
from saealib.core.compiler.graph import ComponentNode, DataEdge, NodeRef
from saealib.core.compiler.regions import (
    BranchRegion,
    LoopRegion,
    RegionNode,
    RepeatRegion,
)
from saealib.core.compiler.structured import StructuredGraph
from saealib.core.contracts.ports import (
    PortDirection,
    PortSpec,
    check_port_compatibility,
    definite_port_incompatibility,
)

__all__ = ["StructuredDataflowRule"]


def _edge_key(edge: DataEdge) -> str:
    def ref_key(reference: NodeRef) -> str:
        role = f"[{reference.role}]" if reference.role is not None else ""
        return f"{reference.component_id}{role}"

    return (
        f"{ref_key(edge.source)}.{edge.source_port}->"
        f"{ref_key(edge.target)}.{edge.target_port}"
    )


def _port_specs(node, direction: PortDirection) -> tuple[tuple[str, PortSpec], ...]:
    result = [
        (role, port)
        for role, contract in sorted(node.contract.ports.items())
        for port in (
            contract.outputs if direction is PortDirection.OUTPUT else contract.inputs
        )
    ]
    return tuple(sorted(result, key=lambda item: (item[0], item[1].name)))


def _target_is_connected(
    edge: DataEdge, target_id: str, target_role: str, target_port: str
) -> bool:
    return (
        edge.target.component_id == target_id
        and edge.target_port == target_port
        and (edge.target.role == target_role or edge.target.role is None)
    )


def _unregistered_vocabulary_names(
    nodes_by_id: dict[str, ComponentNode],
    source_ids: Iterable[str],
    target_port: PortSpec,
) -> tuple[str, ...]:
    """Collect unregistered kinds/cardinalities seen while rejecting sources.

    A source already ruled out by a mismatch between registered values can
    never become a producer, so its rejection stays definite and a missing
    producer remains a defect rather than unverifiable vocabulary.
    """
    unregistered: list[str] = []
    for source_id in sorted(source_ids):
        source = nodes_by_id[source_id]
        for _, source_port in _port_specs(source, PortDirection.OUTPUT):
            compatibility = check_port_compatibility(source_port, target_port)
            if compatibility.compatible or definite_port_incompatibility(compatibility):
                continue
            unregistered.extend(
                name
                for name in (*compatibility.unknown_kinds,)
                if name not in unregistered
            )
            unregistered.extend(
                name
                for name in (*compatibility.unknown_cardinalities,)
                if name not in unregistered
            )
    return tuple(unregistered)


class StructuredDataflowRule:
    """Infer compatible data edges between control-ordered component ports."""

    namespace = "core"
    name = "structured_dataflow"
    phase = "resolution"

    def apply(self, context):
        """Add uniquely resolved data edges and report unresolved inputs."""
        from saealib.core.compiler.compiler import ResolutionResult

        graph = context.graph
        if not isinstance(graph, StructuredGraph):
            return ResolutionResult(graph=graph)

        added: list[DataEdge] = []
        findings: list[Diagnostic] = []
        nodes_by_id = {node.component_id: node for node in graph.nodes}

        def compatible_edges(
            source_ids: Iterable[str], target, target_role: str, target_port: PortSpec
        ) -> list[DataEdge]:
            edges: list[DataEdge] = []
            for source in sorted(
                (nodes_by_id[node_id] for node_id in source_ids),
                key=lambda node: node.component_id,
            ):
                for source_role, source_port in _port_specs(
                    source, PortDirection.OUTPUT
                ):
                    if check_port_compatibility(source_port, target_port).compatible:
                        edges.append(
                            DataEdge(
                                source=NodeRef(
                                    component_id=source.component_id,
                                    role=source_role,
                                ),
                                target=NodeRef(
                                    component_id=target.component_id,
                                    role=target_role,
                                ),
                                source_port=source_port.name,
                                target_port=target_port.name,
                            )
                        )
            return sorted(edges, key=_edge_key)

        def report_unresolved(path: ContractPath) -> None:
            findings.append(
                Diagnostic(
                    severity=Severity.ERROR,
                    code="unresolved_input",
                    message=(f"Input {path} has no compatible upstream producer."),
                    path=path,
                    resolutions=(
                        "Provide one control-ordered compatible output port.",
                    ),
                )
            )

        def report_cannot_verify(
            path: ContractPath,
            unregistered: tuple[str, ...],
        ) -> None:
            findings.append(
                Diagnostic(
                    severity=Severity.WARNING,
                    code="cannot_verify_data_kind",
                    message=(
                        f"Input {path} cannot be verified: data kinds or "
                        "cardinalities are not registered: "
                        f"{', '.join(unregistered)}."
                    ),
                    path=path,
                    resolutions=(
                        "Register the data kinds and cardinalities in their "
                        "vocabularies, or use registered names in the port "
                        "contracts.",
                    ),
                )
            )

        def report_ambiguous(path: ContractPath, edges: list[DataEdge]) -> None:
            findings.append(
                Diagnostic(
                    severity=Severity.ERROR,
                    code="ambiguous_input",
                    message=(
                        f"Input {path} has multiple compatible upstream producers."
                    ),
                    path=path,
                    related=tuple(
                        ContractPath(
                            components=(edge.source.component_id,),
                            role=edge.source.role,
                            port=edge.source_port,
                        )
                        for edge in edges
                    ),
                    resolutions=(
                        "Add an explicit target-port data edge or remove "
                        "the competing producer.",
                    ),
                )
            )

        def resolve_target(
            target, available: set[str], paths: tuple[frozenset[str], ...]
        ) -> None:
            connected = graph.data_edges
            for target_role, target_port in _port_specs(target, PortDirection.INPUT):
                if any(
                    _target_is_connected(
                        edge, target.component_id, target_role, target_port.name
                    )
                    for edge in connected
                ):
                    continue
                candidates = compatible_edges(
                    available, target, target_role, target_port
                )
                path = ContractPath(
                    components=(target.component_id,),
                    role=target_role,
                    port=target_port.name,
                )
                if len(candidates) == 1:
                    edge = candidates[0]
                    if edge not in (*connected, *added):
                        added.append(edge)
                        context.claim("data_edge", _edge_key(edge))
                elif len(candidates) > 1:
                    report_ambiguous(path, candidates)
                else:
                    # No definite producer: accept one compatible producer per
                    # branch path (each leaf of ``paths`` holds the producers
                    # added along one execution path).
                    leaf_edges = [
                        compatible_edges(leaf, target, target_role, target_port)
                        for leaf in paths
                    ]
                    if any(not leaf for leaf in leaf_edges):
                        if not target_port.optional:
                            unregistered = _unregistered_vocabulary_names(
                                nodes_by_id, available, target_port
                            )
                            if unregistered:
                                report_cannot_verify(path, unregistered)
                            else:
                                report_unresolved(path)
                    elif any(len(leaf) > 1 for leaf in leaf_edges):
                        report_ambiguous(
                            path,
                            [
                                edge
                                for leaf in leaf_edges
                                if len(leaf) > 1
                                for edge in leaf
                            ],
                        )
                    else:
                        for edge in leaf_edges:
                            if edge[0] not in (*connected, *added):
                                added.append(edge[0])
                                context.claim("data_edge", _edge_key(edge[0]))

        def sequence(
            current: StructuredGraph,
            available: set[str],
            paths: tuple[frozenset[str], ...],
        ) -> tuple[set[str], tuple[frozenset[str], ...]]:
            current_available = set(available)
            current_paths = paths
            for operation in current.operations:
                if not isinstance(operation, RegionNode):
                    resolve_target(operation, current_available, current_paths)
                    current_available.add(operation.component_id)
                    current_paths = tuple(
                        leaf | {operation.component_id} for leaf in current_paths
                    )
                    continue
                region = operation.region
                body = region.body
                if not isinstance(body, StructuredGraph):
                    continue
                if isinstance(region, RepeatRegion):
                    body_available, _ = sequence(
                        body, current_available, (frozenset(),)
                    )
                    if isinstance(region.count, int) and region.count > 0:
                        current_available.update(body_available)
                    current_paths = (frozenset(),)
                    continue
                if isinstance(region, LoopRegion):
                    sequence(body, current_available, (frozenset(),))
                    current_paths = (frozenset(),)
                    continue
                if isinstance(region, BranchRegion):
                    then_available, then_paths = sequence(
                        body, current_available, current_paths
                    )
                    otherwise = getattr(region, "otherwise", None)
                    if isinstance(otherwise, StructuredGraph):
                        else_available, else_paths = sequence(
                            otherwise, current_available, current_paths
                        )
                    else:
                        else_available = set(current_available)
                        else_paths = current_paths
                    current_available = then_available & else_available
                    current_paths = (*then_paths, *else_paths)
                    continue
                body_available, body_paths = sequence(
                    body, current_available, current_paths
                )
                current_available.update(body_available)
                current_paths = body_paths
            return current_available, current_paths

        sequence(graph, set(), (frozenset(),))
        if not added:
            return ResolutionResult(graph=graph, diagnostics=tuple(findings))
        return ResolutionResult(
            graph=replace(
                graph,
                data_edges=(*graph.data_edges, *sorted(added, key=_edge_key)),
            ),
            diagnostics=tuple(findings),
        )
