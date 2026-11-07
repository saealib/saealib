"""Resolution of port dataflow in structured graphs."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, replace

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
from saealib.core.contracts.flow import (
    CallInput,
    CallOutput,
    StateValue,
    Transform,
    _apply_transforms,
)
from saealib.core.contracts.ports import (
    PortDirection,
    PortSpec,
    check_port_compatibility,
    definite_port_incompatibility,
)
from saealib.core.state.keys import StateKey

__all__ = ["StructuredDataflowRule"]


@dataclass(frozen=True)
class _StateProducer:
    source: NodeRef | None
    state_key: StateKey[object] | None
    port: str
    transforms: tuple[Transform, ...] = ()
    available: bool = False


def _merge_state_paths(
    alternatives: tuple[dict[StateKey[object], tuple[_StateProducer, ...] | None], ...],
) -> dict[StateKey[object], tuple[_StateProducer, ...] | None]:
    """Join state producers by key, retaining unknown and external sources."""
    joined: dict[StateKey[object], tuple[_StateProducer, ...] | None] = {}
    for key in set().union(*(set(path) for path in alternatives)):
        producers: list[_StateProducer] = []
        unknown = False
        for path in alternatives:
            if key not in path:
                producers.append(
                    _StateProducer(source=None, state_key=key, port="value")
                )
                continue
            current = path[key]
            if current is None:
                unknown = True
            else:
                producers.extend(current)
        if unknown:
            joined[key] = None
        elif producers:
            joined[key] = tuple(dict.fromkeys(producers))
    return joined


def _edge_key(edge: DataEdge) -> str:
    def ref_key(reference: NodeRef) -> str:
        role = f"[{reference.role}]" if reference.role is not None else ""
        return f"{reference.component_id}{role}"

    base = (
        f"{ref_key(edge.source)}.{edge.source_port}->"
        f"{ref_key(edge.target)}.{edge.target_port}"
    )
    return base + "|" + ",".join(item.name for item in edge.transforms)


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
        findings: list[Diagnostic] = []
        for node in graph.nodes:
            stage = getattr(node.component, "stage", None)
            if stage is None or callable(getattr(stage, "flow_summary", None)) is False:
                continue
            if stage.flow_summary() is not None:
                continue
            findings.append(
                Diagnostic(
                    severity=Severity.WARNING,
                    code="cannot_verify_stage_flow",
                    message=f"Stage flow for {node.component_id} cannot be verified.",
                    path=ContractPath(components=(node.component_id,)),
                    resolutions=(
                        "Declare the Stage component calls and data connections.",
                    ),
                )
            )

        if not isinstance(graph, StructuredGraph):
            return ResolutionResult(graph=graph, diagnostics=tuple(findings))

        added: list[DataEdge] = []
        nodes_by_id = {node.component_id: node for node in graph.nodes}
        original_node_ids = set(nodes_by_id)
        extra_nodes: dict[str, ComponentNode] = {}
        flow_call_ids = graph.metadata.get("stage_flow_calls", {})
        declared_call_ids = {
            component_id
            for calls in flow_call_ids.values()
            if isinstance(calls, dict)
            for component_id in calls.values()
        }
        current_stage_id = [""]
        used_ids = set(nodes_by_id)
        state_output_nodes: dict[
            tuple[StateKey[object], str, str, tuple[Transform, ...]], ComponentNode
        ] = {}
        state_transform_nodes: dict[tuple[StateKey[object], str], ComponentNode] = {}

        for edge in graph.data_edges:
            target_node = nodes_by_id.get(edge.target.component_id)
            state_key = getattr(
                getattr(target_node, "component", None), "state_key", None
            )
            has_state_input = (
                target_node is not None
                and "state" in target_node.contract.ports
                and bool(target_node.contract.ports["state"].inputs)
            )
            if (
                edge.target.role == "state"
                and isinstance(state_key, StateKey)
                and has_state_input
                and edge.source.component_id in declared_call_ids
            ):
                state_output_nodes[
                    (
                        state_key,
                        edge.source.component_id,
                        edge.source_port,
                        edge.transforms,
                    )
                ] = target_node
        for node in graph.nodes:
            state_key = getattr(node.component, "state_key", None)
            flow_origin = getattr(node.component, "flow_origin", None)
            state_contract = node.contract.ports.get("state")
            if (
                isinstance(state_key, StateKey)
                and isinstance(flow_origin, str)
                and state_contract is not None
                and state_contract.inputs
                and state_contract.outputs
            ):
                state_transform_nodes[(state_key, flow_origin)] = node

        def compatible_edges(
            source_ids: Iterable[str], target, target_role: str, target_port: PortSpec
        ) -> list[DataEdge]:
            edges: list[DataEdge] = []
            for source in sorted(
                (nodes_by_id[node_id] for node_id in source_ids),
                key=lambda node: node.component_id,
            ):
                if source.component_id in declared_call_ids:
                    continue
                if getattr(source.component, "stage", None) is not None:
                    continue
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
                                transforms=(),
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
                    severity=Severity.WARNING,
                    code="ambiguous_input",
                    message=(
                        f"Input {path} has multiple compatible upstream "
                        "producers; none is selected automatically, so the "
                        "connection cannot be verified."
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
            if (
                getattr(target.component, "stage", None) is not None
                or target.component_id in declared_call_ids
            ):
                return
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

        initial_state_keys = set(context.compile_context.initial_state_keys)
        state_keys: set[StateKey[object]] = set(initial_state_keys)
        for node in graph.nodes:
            stage = getattr(node.component, "stage", None)
            flow_summary = getattr(stage, "flow_summary", None)
            summary = flow_summary() if callable(flow_summary) else None
            if summary is None:
                continue
            for edge in summary.edges:
                if isinstance(edge.source, StateValue):
                    state_keys.add(edge.source.key)
                if isinstance(edge.target, StateValue):
                    state_keys.add(edge.target.key)

        def call_id(stage_id: str, name: str) -> str | None:
            stage_calls = flow_call_ids.get(stage_id, {})
            return stage_calls.get(name) if isinstance(stage_calls, dict) else None

        def external_producer(
            key: StateKey[object],
            target_id: str,
            target_role: str,
            target_port: PortSpec,
            transforms: tuple[Transform, ...],
            *,
            allow_non_initial: bool = False,
        ) -> _StateProducer | None:
            if key not in initial_state_keys and not allow_non_initial:
                return None
            from saealib.core.graph_builder import _state_source_node

            source_node = next(
                (
                    node
                    for node in (*graph.nodes, *extra_nodes.values())
                    if getattr(node.component, "state_key", None) == key
                    and getattr(node.component, "target_id", None) == target_id
                ),
                None,
            )
            if source_node is None:
                used_ids = set(nodes_by_id) | set(extra_nodes)
                source_node = _state_source_node(
                    key,
                    target_id,
                    target_port,
                    transforms,
                    used_ids=used_ids,
                )
                extra_nodes[source_node.component_id] = source_node
                nodes_by_id[source_node.component_id] = source_node
                context.claim("node", source_node.component_id)
            return _StateProducer(
                source=NodeRef(
                    component_id=source_node.component_id,
                    role="state",
                ),
                state_key=None,
                port="value",
            )

        def target_port_for(
            call: CallInput,
        ) -> tuple[str, ComponentNode, PortSpec] | None:
            component_id = call_id(current_stage_id[0], call.call.name)
            if component_id is None:
                return None
            node = nodes_by_id.get(component_id)
            if node is None:
                return None
            role_contract = node.contract.ports.get(call.call.role)
            if role_contract is None:
                return None
            matches = tuple(
                port for port in role_contract.inputs if port.name == call.port
            )
            if len(matches) != 1:
                return None
            return component_id, node, matches[0]

        def output_port_for(reference: NodeRef, port_name: str) -> PortSpec | None:
            node = nodes_by_id.get(reference.component_id)
            if node is None or reference.role is None:
                return None
            role_contract = node.contract.ports.get(reference.role)
            if role_contract is None:
                return None
            return next(
                (port for port in role_contract.outputs if port.name == port_name),
                None,
            )

        def add_edge(edge: DataEdge) -> None:
            if edge in (*graph.data_edges, *added):
                return
            added.append(edge)
            context.claim("data_edge", _edge_key(edge))

        def state_flow_for_output(
            key: StateKey[object], producer: _StateProducer
        ) -> _StateProducer:
            source = producer.source
            if source is None or source.component_id not in declared_call_ids:
                return producer
            cache_key = (key, source.component_id, producer.port, producer.transforms)
            state_node = state_output_nodes.get(cache_key)
            if state_node is None:
                source_port = output_port_for(source, producer.port)
                source_node = nodes_by_id.get(source.component_id)
                if source_port is None or source_node is None:
                    return producer
                data, error = _apply_transforms(source_port.data, producer.transforms)
                if error is not None or data is None:
                    data = source_port.data
                from saealib.core.graph_builder import _state_flow_node

                state_node = _state_flow_node(
                    key, source.component_id, data, used_ids=used_ids
                )
                nodes_by_id[state_node.component_id] = state_node
                extra_nodes[state_node.component_id] = state_node
                context.claim("node", state_node.component_id)
                state_output_nodes[cache_key] = state_node
                add_edge(
                    DataEdge(
                        source=source,
                        target=NodeRef(
                            component_id=state_node.component_id, role="state"
                        ),
                        source_port=producer.port,
                        target_port="value",
                        transforms=producer.transforms,
                    )
                )
            return _StateProducer(
                source=NodeRef(component_id=state_node.component_id, role="state"),
                state_key=None,
                port="value",
                available=producer.available,
            )

        def state_flow_for_transform(
            stage_id: str,
            target_key: StateKey[object],
            source_key: StateKey[object],
            producer: _StateProducer,
            transforms: tuple[Transform, ...],
            ordinal: int,
        ) -> _StateProducer:
            producer = state_flow_for_output(source_key, producer)
            source = producer.source
            if source is None:
                return _StateProducer(
                    source=None,
                    state_key=producer.state_key or source_key,
                    port=producer.port,
                    transforms=(*producer.transforms, *transforms),
                    available=producer.available,
                )
            source_node = nodes_by_id.get(source.component_id)
            if source_node is None:
                return producer
            source_port = output_port_for(source, producer.port)
            if source_port is None:
                return producer
            data, error = _apply_transforms(source_port.data, transforms)
            if error is not None or data is None:
                data = source_port.data

            node_name = f"{stage_id}__state"
            if ordinal:
                node_name = f"{node_name}_{ordinal + 1}"
            cache_key = (target_key, node_name)
            state_node = state_transform_nodes.get(cache_key)
            if state_node is None:
                from saealib.core.graph_builder import (
                    _state_flow_node,
                )

                state_node = _state_flow_node(
                    target_key, node_name, data, used_ids=used_ids
                )
                nodes_by_id[state_node.component_id] = state_node
                extra_nodes[state_node.component_id] = state_node
                context.claim("node", state_node.component_id)
                state_transform_nodes[cache_key] = state_node
            add_edge(
                DataEdge(
                    source=source,
                    target=NodeRef(component_id=state_node.component_id, role="state"),
                    source_port=producer.port,
                    target_port="value",
                    transforms=transforms,
                )
            )
            return _StateProducer(
                source=NodeRef(component_id=state_node.component_id, role="state"),
                state_key=None,
                port="value",
                available=producer.available,
            )

        def append_state_edges(
            stage_id: str,
            summary,
            paths: tuple[
                dict[StateKey[object], tuple[_StateProducer, ...] | None], ...
            ],
        ) -> None:
            current_stage_id[0] = stage_id
            for edge in summary.edges:
                if not isinstance(edge.source, StateValue) or not isinstance(
                    edge.target, CallInput
                ):
                    continue
                target = target_port_for(edge.target)
                if target is None:
                    continue
                target_id, _target_node, target_port = target
                for path in paths:
                    if edge.source.key not in path:
                        producers = (
                            external_producer(
                                edge.source.key,
                                target_id,
                                edge.target.call.role,
                                target_port,
                                edge.transforms,
                            ),
                        )
                    else:
                        known = path[edge.source.key]
                        producers = (
                            (
                                _StateProducer(
                                    source=None,
                                    state_key=edge.source.key,
                                    port="value",
                                    available=True,
                                ),
                            )
                            if known is None
                            else known
                        )
                    for producer in producers:
                        if producer is None:
                            continue
                        transforms = (*producer.transforms, *edge.transforms)
                        source = producer.source
                        if source is None:
                            external = external_producer(
                                producer.state_key or edge.source.key,
                                target_id,
                                edge.target.call.role,
                                target_port,
                                transforms,
                                allow_non_initial=producer.available,
                            )
                            if external is None:
                                continue
                            source = external.source
                            assert source is not None
                        elif source.component_id in declared_call_ids:
                            producer = state_flow_for_output(
                                producer.state_key or edge.source.key,
                                _StateProducer(
                                    source=source,
                                    state_key=producer.state_key,
                                    port=producer.port,
                                    transforms=producer.transforms,
                                    available=producer.available,
                                ),
                            )
                            source = producer.source
                            assert source is not None
                            transforms = edge.transforms
                        else:
                            transforms = (*producer.transforms, *edge.transforms)
                        data_edge = DataEdge(
                            source=source,
                            target=NodeRef(
                                component_id=target_id,
                                role=edge.target.call.role,
                            ),
                            source_port=producer.port,
                            target_port=edge.target.port,
                            transforms=transforms,
                        )
                        adapter_route_exists = any(
                            first.source == data_edge.source
                            and first.source_port == data_edge.source_port
                            and (adapter := nodes_by_id.get(first.target.component_id))
                            is not None
                            and getattr(adapter.component, "insertion", None)
                            is not None
                            and any(
                                second.source.component_id == adapter.component_id
                                and second.target == data_edge.target
                                and second.target_port == data_edge.target_port
                                for second in graph.data_edges
                            )
                            for first in graph.data_edges
                        )
                        if (
                            data_edge not in (*graph.data_edges, *added)
                            and not adapter_route_exists
                        ):
                            add_edge(data_edge)

        def update_state_paths(
            stage_id: str,
            summary,
            paths: tuple[
                dict[StateKey[object], tuple[_StateProducer, ...] | None], ...
            ],
        ) -> tuple[dict[StateKey[object], tuple[_StateProducer, ...] | None], ...]:
            updated: list[
                dict[StateKey[object], tuple[_StateProducer, ...] | None]
            ] = []
            for path in paths:
                current = dict(path)
                for ordinal, edge in enumerate(summary.edges):
                    if isinstance(edge.source, StateValue) and isinstance(
                        edge.target, StateValue
                    ):
                        if edge.source.key in current:
                            prior = current[edge.source.key]
                            current[edge.target.key] = (
                                (
                                    state_flow_for_transform(
                                        stage_id,
                                        edge.target.key,
                                        edge.source.key,
                                        _StateProducer(
                                            source=None,
                                            state_key=edge.source.key,
                                            port="value",
                                            available=True,
                                        ),
                                        edge.transforms,
                                        ordinal,
                                    ),
                                )
                                if prior is None
                                else tuple(
                                    state_flow_for_transform(
                                        stage_id,
                                        edge.target.key,
                                        edge.source.key,
                                        item,
                                        edge.transforms,
                                        ordinal,
                                    )
                                    for item in prior
                                )
                            )
                        elif edge.source.key in initial_state_keys:
                            current[edge.target.key] = (
                                state_flow_for_transform(
                                    stage_id,
                                    edge.target.key,
                                    edge.source.key,
                                    _StateProducer(
                                        source=None,
                                        state_key=edge.source.key,
                                        port="value",
                                        available=True,
                                    ),
                                    edge.transforms,
                                    ordinal,
                                ),
                            )
                        else:
                            current[edge.target.key] = None
                    elif isinstance(edge.source, CallOutput) and isinstance(
                        edge.target, StateValue
                    ):
                        producer_id = call_id(stage_id, edge.source.call.name)
                        if producer_id is None:
                            continue
                        producer = _StateProducer(
                            source=NodeRef(
                                component_id=producer_id,
                                role=edge.source.call.role,
                            ),
                            state_key=None,
                            port=edge.source.port,
                            transforms=edge.transforms,
                        )
                        current[edge.target.key] = (
                            state_flow_for_output(edge.target.key, producer),
                        )
                updated.append(current)
            return tuple(updated)

        def stage_flow(
            node: ComponentNode,
            paths: tuple[
                dict[StateKey[object], tuple[_StateProducer, ...] | None], ...
            ],
        ) -> tuple[dict[StateKey[object], tuple[_StateProducer, ...] | None], ...]:
            stage = getattr(node.component, "stage", None)
            if stage is None or not callable(getattr(stage, "flow_summary", None)):
                return paths
            summary = stage.flow_summary()
            if summary is None:
                writes = node.analysis_contract.state.writes
                unknown_keys = (
                    state_keys
                    if not node.analysis_contract.state.writes_enumerable
                    else writes
                )
                return tuple(
                    {**path, **{key: None for key in unknown_keys}} for path in paths
                )
            append_state_edges(node.component_id, summary, paths)
            updated = update_state_paths(node.component_id, summary, paths)
            explicit_writes = {
                edge.target.key
                for edge in summary.edges
                if isinstance(edge.target, StateValue)
            }
            stage_writes = set(node.analysis_contract.state.writes) - explicit_writes
            if not stage_writes:
                return updated
            return tuple(
                {
                    **path,
                    **{
                        key: (
                            _StateProducer(
                                source=None,
                                state_key=key,
                                port="value",
                                available=True,
                            ),
                        )
                        for key in stage_writes
                    },
                }
                for path in updated
            )

        def loop_writers(
            current: StructuredGraph,
            incoming: dict[StateKey[object], tuple[_StateProducer, ...] | None]
            | None = None,
        ) -> dict[StateKey[object], tuple[_StateProducer, ...] | None]:
            writers = {} if incoming is None else dict(incoming)

            def merge(
                alternatives: tuple[
                    dict[StateKey[object], tuple[_StateProducer, ...] | None], ...
                ],
            ) -> dict[StateKey[object], tuple[_StateProducer, ...] | None]:
                return _merge_state_paths(alternatives)

            for operation in current.operations:
                if isinstance(operation, RegionNode):
                    region = operation.region
                    if isinstance(region, RepeatRegion) and region.count == 0:
                        continue
                    body = region.body
                    body_writers = (
                        loop_writers(body, writers)
                        if isinstance(body, StructuredGraph)
                        else dict(writers)
                    )
                    otherwise = getattr(region, "otherwise", None)
                    if isinstance(region, BranchRegion):
                        other_writers = (
                            loop_writers(otherwise, writers)
                            if isinstance(otherwise, StructuredGraph)
                            else dict(writers)
                        )
                        writers = merge((body_writers, other_writers))
                    elif isinstance(region, LoopRegion) or (
                        isinstance(region, RepeatRegion)
                        and not isinstance(region.count, int)
                    ):
                        writers = merge((dict(writers), body_writers))
                    else:
                        writers = body_writers
                    continue
                stage = getattr(operation.component, "stage", None)
                if stage is None or not callable(getattr(stage, "flow_summary", None)):
                    continue
                summary = stage.flow_summary()
                if summary is None:
                    state = operation.analysis_contract.state
                    if not state.writes_enumerable:
                        for key in state_keys:
                            writers[key] = None
                    else:
                        for key in state.writes:
                            writers[key] = None
                    continue
                for ordinal, edge in enumerate(summary.edges):
                    if isinstance(edge.source, StateValue) and isinstance(
                        edge.target, StateValue
                    ):
                        prior = writers.get(edge.source.key, ())
                        producers = (
                            (
                                _StateProducer(
                                    source=None,
                                    state_key=edge.source.key,
                                    port="value",
                                    transforms=(),
                                    available=True,
                                ),
                            )
                            if prior is None
                            else prior
                            or (
                                _StateProducer(
                                    source=None,
                                    state_key=edge.source.key,
                                    port="value",
                                    transforms=(),
                                    available=True,
                                ),
                            )
                        )
                        writers[edge.target.key] = tuple(
                            state_flow_for_transform(
                                operation.component_id,
                                edge.target.key,
                                edge.source.key,
                                item,
                                edge.transforms,
                                ordinal,
                            )
                            for item in producers
                        )
                    elif isinstance(edge.source, CallOutput) and isinstance(
                        edge.target, StateValue
                    ):
                        producer_id = call_id(
                            operation.component_id, edge.source.call.name
                        )
                        if producer_id is not None:
                            writers[edge.target.key] = (
                                state_flow_for_output(
                                    edge.target.key,
                                    _StateProducer(
                                        source=NodeRef(
                                            component_id=producer_id,
                                            role=edge.source.call.role,
                                        ),
                                        state_key=None,
                                        port=edge.source.port,
                                        transforms=edge.transforms,
                                    ),
                                ),
                            )
                explicit_writes = {
                    edge.target.key
                    for edge in summary.edges
                    if isinstance(edge.target, StateValue)
                }
                for key in (
                    set(operation.analysis_contract.state.writes) - explicit_writes
                ):
                    writers[key] = (
                        _StateProducer(
                            source=None,
                            state_key=key,
                            port="value",
                            available=True,
                        ),
                    )
            return writers

        def sequence(
            current: StructuredGraph,
            available: set[str],
            paths: tuple[frozenset[str], ...],
            state_paths: tuple[
                dict[StateKey[object], tuple[_StateProducer, ...] | None], ...
            ],
        ) -> tuple[
            set[str],
            tuple[frozenset[str], ...],
            tuple[dict[StateKey[object], tuple[_StateProducer, ...] | None], ...],
        ]:
            current_available = set(available)
            current_paths = paths
            current_state_paths = state_paths
            for operation in current.operations:
                if not isinstance(operation, RegionNode):
                    stage = getattr(operation.component, "stage", None)
                    if stage is not None and callable(
                        getattr(stage, "flow_summary", None)
                    ):
                        current_state_paths = stage_flow(operation, current_state_paths)
                        continue
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
                    body_available, _, body_state_paths = sequence(
                        body,
                        current_available,
                        (frozenset(),),
                        current_state_paths,
                    )
                    if region.count != 0:
                        current_available.update(body_available)
                        current_state_paths = body_state_paths
                    current_paths = (frozenset(),)
                    continue
                if isinstance(region, LoopRegion):
                    back_writers = loop_writers(body)
                    loop_inputs: list[
                        dict[StateKey[object], tuple[_StateProducer, ...] | None]
                    ] = []
                    for path in current_state_paths:
                        loop_path = dict(path)
                        for key, writers in back_writers.items():
                            current_value = loop_path.get(key, ())
                            if writers is None:
                                loop_path[key] = None
                                continue
                            if current_value is None:
                                continue
                            entry_values = current_value
                            if key not in loop_path:
                                entry_values = (
                                    _StateProducer(
                                        source=None,
                                        state_key=key,
                                        port="value",
                                    ),
                                )
                            loop_path[key] = (*entry_values, *writers)
                        loop_inputs.append(loop_path)
                    _, _, body_state_paths = sequence(
                        body,
                        current_available,
                        (frozenset(),),
                        (_merge_state_paths(tuple(loop_inputs)),),
                    )
                    current_state_paths = (
                        _merge_state_paths((*current_state_paths, *body_state_paths)),
                    )
                    current_paths = (frozenset(),)
                    continue
                if isinstance(region, BranchRegion):
                    then_available, then_paths, then_state_paths = sequence(
                        body, current_available, current_paths, current_state_paths
                    )
                    otherwise = getattr(region, "otherwise", None)
                    if isinstance(otherwise, StructuredGraph):
                        else_available, else_paths, else_state_paths = sequence(
                            otherwise,
                            current_available,
                            current_paths,
                            current_state_paths,
                        )
                    else:
                        else_available = set(current_available)
                        else_paths = current_paths
                        else_state_paths = current_state_paths
                    current_available = then_available & else_available
                    current_paths = tuple(dict.fromkeys((*then_paths, *else_paths)))
                    current_state_paths = (
                        _merge_state_paths((*then_state_paths, *else_state_paths)),
                    )
                    continue
                body_available, body_paths, body_state_paths = sequence(
                    body, current_available, current_paths, current_state_paths
                )
                current_available.update(body_available)
                current_paths = body_paths
                current_state_paths = body_state_paths
            return current_available, current_paths, current_state_paths

        sequence(graph, set(), (frozenset(),), ({},))
        all_edges = (*graph.data_edges, *sorted(added, key=_edge_key))
        state_source_ids = {
            node.component_id
            for node in graph.nodes
            if isinstance(getattr(node.component, "state_key", None), StateKey)
            and isinstance(getattr(node.component, "target_id", None), str)
        }
        all_state_source_ids = state_source_ids | set(extra_nodes)
        used_state_source_ids = {
            reference.component_id
            for edge in all_edges
            for reference in (edge.source, edge.target)
            if reference.component_id in all_state_source_ids
        }
        nodes = tuple(
            node
            for node in graph.nodes
            if node.component_id not in state_source_ids
            or node.component_id in used_state_source_ids
        )
        nodes = (
            *nodes,
            *(
                extra_nodes[item]
                for item in sorted(used_state_source_ids)
                if item not in original_node_ids
            ),
        )
        retained_node_ids = {node.component_id for node in nodes}
        for node in graph.nodes:
            if node.component_id not in retained_node_ids:
                context.claim("node", node.component_id)
        state_entries = tuple(
            NodeRef(component_id=component_id, role="state")
            for component_id in sorted(used_state_source_ids)
        )
        entry_points = tuple(
            entry
            for entry in graph.entry_points
            if entry.component_id not in state_source_ids
            or entry.component_id in used_state_source_ids
        )
        entry_points = tuple(dict.fromkeys((*entry_points, *state_entries)))
        old_entry_points = set(graph.entry_points)
        for entry in entry_points:
            if entry not in old_entry_points:
                suffix = f"[{entry.role}]" if entry.role is not None else ""
                context.claim("entry_point", f"{entry.component_id}{suffix}")
        if not added and nodes == graph.nodes and entry_points == graph.entry_points:
            return ResolutionResult(graph=graph, diagnostics=tuple(findings))
        return ResolutionResult(
            graph=replace(
                graph,
                nodes=nodes,
                data_edges=all_edges,
                entry_points=entry_points,
            ),
            diagnostics=tuple(findings),
        )
