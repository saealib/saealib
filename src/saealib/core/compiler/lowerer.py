"""Lower structured pipeline values into an immutable :class:`StructuredGraph`."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

from saealib.core.compiler.graph import (
    ComponentNode,
    ControlEdge,
    NodeRef,
    StateBinding,
)
from saealib.core.compiler.regions import (
    BranchRegion,
    LoopRegion,
    RegionNode,
    RepeatRegion,
    SequenceRegion,
    StructuredRegion,
    _is_branch,
    _is_component,
    _is_loop,
    _is_pipeline_like,
    _is_repeat,
    _is_stage,
    _structural_name,
    compose_effects,
)
from saealib.core.compiler.structured import StructuredGraph
from saealib.core.contracts.contract import ComponentContract
from saealib.core.contracts.state import StateContract
from saealib.exceptions import ValidationError

__all__ = ["lower_pipeline", "lower_structured"]


def _pipeline_items(value: object) -> tuple[object, ...] | None:
    if isinstance(value, (list, tuple)):
        return tuple(value)
    if _is_pipeline_like(value):
        return tuple(value.stages)
    return None


def _pipeline_name(value: object) -> str:
    return _structural_name(value)


def _body_items(value: object) -> tuple[object, ...] | None:
    items = _pipeline_items(value)
    if items is not None:
        return items
    if isinstance(value, StructuredGraph):
        return None
    return (value,)


def _region_body(value: object) -> StructuredGraph | tuple[object, ...]:
    if isinstance(value, StructuredGraph):
        return value
    items = _pipeline_items(value)
    return items if items is not None else (value,)


def _entry_nodes(graph: StructuredGraph) -> tuple[ComponentNode, ...]:
    """Collect the component nodes declared as the graph's entry points."""
    ids = {ref.component_id for ref in graph.entry_points}
    entries = tuple(node for node in graph.nodes if node.component_id in ids)
    if entries or not graph.nodes:
        return entries
    return (graph.nodes[0],)


def _lower_sequence(items: tuple[object, ...], namespace: str) -> StructuredGraph:
    """Lower a sequence, discarding its exit nodes."""
    return _lower_sequence_with_exits(items, namespace)[0]


def _lower_sequence_with_exits(
    items: tuple[object, ...], namespace: str
) -> tuple[StructuredGraph, tuple[ComponentNode, ...]]:
    nodes: list[ComponentNode] = []
    edges: list[ControlEdge] = []
    region_nodes: list[RegionNode] = []
    operations: list[ComponentNode | RegionNode] = []
    effects: list[StateContract] = []
    bindings: list[StateBinding] = []
    previous: tuple[ComponentNode, ...] = ()
    first_entries: tuple[ComponentNode, ...] = ()
    used_names: set[str] = set()
    for index, item in enumerate(items):
        entries: tuple[ComponentNode, ...] = ()
        exits: tuple[ComponentNode, ...] = ()
        item_name = _structural_name(item)
        if isinstance(item, StructuredRegion):
            item_name = item.region_id
        local_id = (
            item_name if isinstance(item_name, str) and item_name else f"node{index}"
        )
        base_id = local_id
        occurrence = 2
        while local_id in used_names:
            local_id = f"{base_id}_{occurrence}"
            occurrence += 1
        used_names.add(local_id)
        child_namespace = f"{namespace}.{local_id}" if namespace else local_id
        nested_items = _pipeline_items(item)
        if _is_repeat(item):
            item = RepeatRegion(
                region_id=local_id,
                namespace=namespace,
                body=_region_body(item.body),
                count=item.count,
            )
        elif _is_loop(item):
            item = LoopRegion(
                region_id=local_id,
                namespace=namespace,
                body=_region_body(item.body),
                condition=item.condition,
            )
        elif _is_branch(item):
            item = BranchRegion(
                region_id=local_id,
                namespace=namespace,
                body=_region_body(item.then),
                condition=item.condition,
                otherwise=(None if item.else_ is None else _region_body(item.else_)),
            )
        if isinstance(item, (RepeatRegion, LoopRegion, BranchRegion)):
            nested_items = None
        if nested_items is not None:
            body, body_exits = _lower_sequence_with_exits(nested_items, child_namespace)
            region = SequenceRegion(
                region_id=local_id, namespace=namespace, body=body, effect=body.effect
            )
            region_nodes.append(
                RegionNode(region=region, metadata={"kind": "sequence"})
            )
            operations.append(region_nodes[-1])
            nodes.extend(body.nodes)
            edges.extend(body.control_edges)
            bindings.extend(body.state_bindings)
            effects.append(body.effect)
            if body.nodes:
                entries = _entry_nodes(body)
                exits = body_exits or (body.nodes[-1],)
        elif isinstance(item, StructuredRegion):
            body_exits: tuple[ComponentNode, ...] = ()
            otherwise_exits: tuple[ComponentNode, ...] = ()
            body_items = _body_items(item.body)
            if body_items is None:
                if isinstance(item.body, StructuredGraph):
                    body = item.body
                else:
                    raise ValidationError(
                        "Structured region body must be a sequence or StructuredGraph"
                    )
            else:
                body, body_exits = _lower_sequence_with_exits(
                    body_items, child_namespace
                )
            otherwise = getattr(item, "otherwise", None)
            otherwise_graph: StructuredGraph | None = None
            if otherwise is not None:
                otherwise_items = _body_items(otherwise)
                if otherwise_items is None:
                    if not isinstance(otherwise, StructuredGraph):
                        raise ValidationError(
                            "Structured region alternate body must be a sequence "
                            "or StructuredGraph"
                        )
                    otherwise_graph = otherwise
                else:
                    otherwise_graph, otherwise_exits = _lower_sequence_with_exits(
                        otherwise_items, f"{child_namespace}.else"
                    )
            condition = getattr(item, "condition", None)
            condition_contract = (
                condition.contract() if condition is not None else StateContract()
            )
            lowered = item.with_body(
                body,
                effect=compose_effects((body.effect, item.effect, condition_contract)),
            )
            if otherwise_graph is not None:
                lowered = replace(
                    lowered,
                    effect=compose_effects((lowered.effect, otherwise_graph.effect)),
                )
                lowered = replace(lowered, otherwise=otherwise_graph)
            region_nodes.append(
                RegionNode(
                    region=lowered,
                    metadata={
                        "kind": type(item).__name__,
                        "qualified_id": lowered.qualified_id,
                    },
                )
            )
            operations.append(region_nodes[-1])
            nodes.extend(body.nodes)
            edges.extend(body.control_edges)
            bindings.extend(body.state_bindings)
            if otherwise_graph is not None:
                nodes.extend(otherwise_graph.nodes)
                edges.extend(otherwise_graph.control_edges)
                bindings.extend(otherwise_graph.state_bindings)
            effects.append(lowered.effect)
            if otherwise_graph is not None:
                effects.append(otherwise_graph.effect)
            entries = tuple(
                node
                for graph in (body, otherwise_graph)
                if graph is not None and graph.nodes
                for node in _entry_nodes(graph)
            )
            exits = tuple(
                node
                for graph, graph_exits in (
                    (body, body_exits),
                    (otherwise_graph, otherwise_exits),
                )
                if graph is not None and graph.nodes
                for node in (graph_exits or (graph.nodes[-1],))
            )
        else:
            if _is_stage(item):
                raise ValidationError(
                    "Structured lowering does not accept bare Stage "
                    f"{type(item).__name__!r}; wrap it with stage_component(...) "
                    "or use the sequential runtime"
                )
            if not _is_component(item):
                raise ValidationError(
                    "Structured lowering requires components with contract(); "
                    "pipeline values must contain components with contract()"
                )
            contract_method = getattr(item, "analysis_contract", None)
            if not callable(contract_method):
                contract_method = item.contract
            contract = contract_method()
            if not isinstance(contract, ComponentContract):
                raise ValidationError(
                    "Component contract() must return ComponentContract"
                )
            node = ComponentNode(component_id=child_namespace, component=item)
            nodes.append(node)
            operations.append(node)
            bindings.extend(
                StateBinding(node=NodeRef(component_id=child_namespace), state_key=key)
                for key in dict.fromkeys(
                    (
                        *contract.state.reads,
                        *contract.state.writes,
                        *contract.state.exports,
                    )
                )
            )
            effects.append(contract.state)
            entries = (node,)
            exits = (node,)
        if previous and entries:
            edges.extend(
                ControlEdge(
                    source=NodeRef(component_id=source.component_id),
                    target=NodeRef(component_id=target.component_id),
                )
                for source in previous
                for target in entries
                if source.component_id != target.component_id
            )
        if entries:
            previous = exits
            first_entries = first_entries or entries
    entry = tuple(NodeRef(component_id=node.component_id) for node in first_entries)
    return StructuredGraph(
        nodes=tuple(nodes),
        control_edges=tuple(edges),
        state_bindings=tuple(bindings),
        entry_points=entry,
        region_nodes=tuple(region_nodes),
        operations=tuple(operations),
        effect=compose_effects(effects),
    ), previous


def lower_structured(
    value: object | Sequence[object], *, namespace: str | None = None
) -> StructuredGraph:
    """Recursively lower a Pipeline-like sequence without runtime execution.

    ``namespace`` overrides the node-ID prefix the value's own name would
    produce: ``None`` keeps the display name, ``""`` lowers nested nodes
    under their bare local IDs without a qualifier.
    """
    items = _pipeline_items(value)
    if items is None:
        raise ValidationError("lower_structured requires a Pipeline or sequence")
    return _lower_sequence(
        items, _pipeline_name(value) if namespace is None else namespace
    )


lower_pipeline = lower_structured
