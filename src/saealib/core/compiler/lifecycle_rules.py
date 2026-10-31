"""Feedback lifecycle verification and compile-time accumulation insertion."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Literal

from saealib.core.compiler.adapters import (
    DEFAULT_ADAPTER_REGISTRY,
    Adapter,
    AdapterComponent,
    AdapterInsertion,
    AdapterMatchContext,
    _feedback_accumulator_match,
)
from saealib.core.compiler.diagnostics import ContractPath, Diagnostic, Severity
from saealib.core.compiler.graph import ComponentGraph, ComponentNode, DataEdge, NodeRef
from saealib.core.contracts import ComponentContract, PortDirection, PortSpec
from saealib.core.contracts.feedback import (
    BY_PROPOSAL,
    COMPLETE_BATCH,
    IN_ORDER,
    OUT_OF_ORDER_ALLOWED,
    PARTIAL_ALLOWED,
    REPEATED_ALLOWED,
    SINGLE,
)
from saealib.core.contracts.observation import TRUE

if TYPE_CHECKING:
    from saealib.core.compiler.compiler import RuleContext, VerificationResult

__all__ = ["FeedbackAccumulatorRule", "LifecycleCompatibilityRule"]


_PARTIAL_FEEDBACK = "partial_feedback"


def _node_path(
    node: ComponentNode,
    *,
    role: str | None = None,
    port: str | None = None,
) -> ContractPath:
    return ContractPath(components=(node.component_id,), role=role, port=port)


def _edge_key(edge: DataEdge) -> str:
    def reference(value: NodeRef) -> str:
        return value.component_id + (f"[{value.role}]" if value.role else "")

    return (
        f"{reference(edge.source)}.{edge.source_port}->"
        f"{reference(edge.target)}.{edge.target_port}"
    )


def _edge_token(edge: DataEdge) -> str:
    def token(value: str | None) -> str:
        return value or "default"

    return "_".join(
        (
            edge.source.component_id,
            token(edge.source.role),
            edge.source_port,
            edge.target.component_id,
            token(edge.target.role),
            edge.target_port,
        )
    )


def _port(
    node: ComponentNode,
    reference: NodeRef,
    name: str,
    direction: PortDirection,
) -> PortSpec | None:
    contracts = (
        ((reference.role, node.contract.ports.get(reference.role)),)
        if reference.role is not None
        else tuple(sorted(node.contract.ports.items()))
    )
    found = [
        port
        for _, contract in contracts
        if contract is not None
        for port in (
            contract.outputs if direction is PortDirection.OUTPUT else contract.inputs
        )
        if port.name == name
    ]
    return found[0] if len(found) == 1 else None


def _feedback_consumer(node: ComponentNode):
    return getattr(node.contract.lifecycle, "feedback", None)


def _partial_feedback_offered(context: RuleContext, graph: ComponentGraph) -> bool:
    offered = context.compile_context.offered_runtime_capabilities
    return _PARTIAL_FEEDBACK in (offered if offered is not None else frozenset())


def _inserted_adapter_name(node: ComponentNode) -> str | None:
    insertion = getattr(node.component, "insertion", None)
    return getattr(insertion, "adapter_name", None)


def _is_accumulator_node(node: ComponentNode) -> bool:
    return _inserted_adapter_name(node) == "feedback_accumulator"


def _feedback_ports(
    contract: ComponentContract,
) -> tuple[tuple[str, PortSpec], tuple[str, PortSpec]] | None:
    """Find one FeedbackBatch output and one feedback-consumer input.

    A StageNodeAdapter may compose the two ports under arbitrary role names,
    so the rule chooses by lifecycle role and registered data vocabulary.  The
    consumer's representation is intentionally opaque to the core compiler.
    """
    outputs = tuple(
        (role, port)
        for role, role_contract in sorted(contract.ports.items())
        for port in role_contract.outputs
        if port.data.kind == "FeedbackBatch"
    )
    inputs = tuple(
        (role, port)
        for role, role_contract in sorted(contract.ports.items())
        for port in role_contract.inputs
    )
    preferred_outputs = tuple(
        item
        for item in outputs
        if item[0] == "feedback_builder" and item[1].name == "feedback"
    )
    preferred_inputs = tuple(
        item
        for item in inputs
        if item[0] == "feedback_consumer" and item[1].name == "feedback"
    )
    if len(preferred_outputs) == 1 and len(preferred_inputs) == 1:
        return preferred_outputs[0], preferred_inputs[0]
    if len(outputs) != 1 or len(inputs) != 1:
        return None
    return outputs[0], inputs[0]


def _edge_targets_feedback_consumer(edge: DataEdge, node: ComponentNode) -> bool:
    ports = _feedback_ports(node.contract)
    if ports is None:
        # A standalone consumer node may expose only its one feedback input;
        # its lifecycle declaration is enough to identify that connection.
        return _feedback_consumer(node) is not None
    (_, _), (role, port) = ports
    return edge.target.role == role and edge.target_port == port.name


def _adapter_registration(
    context: RuleContext,
    graph: ComponentGraph,
    source_node: ComponentNode,
    target_node: ComponentNode,
    source_port: PortSpec,
    target_port: PortSpec,
) -> Adapter | None:
    registry = getattr(context.compile_context, "adapter_registry", None)
    if registry is None:
        registry = DEFAULT_ADAPTER_REGISTRY
    registrations = tuple(
        adapter
        for adapter in registry.registrations()
        if isinstance(adapter, Adapter) and adapter.name == "feedback_accumulator"
    )
    if len(registrations) != 1:
        return None
    adapter = registrations[0]
    if source_port.data.kind != adapter.source.kind:
        return None
    match = AdapterMatchContext(
        source_node=source_node,
        target_node=target_node,
        source_port=source_port,
        target_port=target_port,
        compile_context=context.compile_context,
        graph=graph,
    )
    if not _feedback_accumulator_match(match):
        return None
    # The accumulator preserves the consumer's declared data shape.  The
    # target is copied here so the framework never needs to name a profile's
    # representation in its default adapter registry.
    return replace(adapter, target=target_port.data)


def _accumulator_inserted_between(graph: ComponentGraph, edge: DataEdge) -> bool:
    try:
        source = graph.node_by_id(edge.source.component_id)
    except KeyError:
        return False
    return _is_accumulator_node(source)


class FeedbackAccumulatorRule:
    """Insert the compile-time accumulator for partial feedback delivery."""

    namespace = "core"
    name = "feedback_accumulator"
    phase: Literal["resolution"] = "resolution"

    def apply(self, context: RuleContext):
        """Rewrite external and Stage-internal feedback connections."""
        from saealib.core.compiler.compiler import ResolutionResult

        graph = context.graph
        if not _partial_feedback_offered(context, graph):
            return ResolutionResult(graph=graph, claims=context.claims)

        candidates: list[DataEdge] = list(graph.data_edges)

        nodes = list(graph.nodes)
        rewritten_edges: list[DataEdge] = []
        claims = set(context.claims)
        inserted: dict[str, ComponentNode] = {}
        for edge in candidates:
            try:
                source = graph.node_by_id(edge.source.component_id)
                target = graph.node_by_id(edge.target.component_id)
            except KeyError:
                if edge in graph.data_edges:
                    rewritten_edges.append(edge)
                continue
            if _accumulator_inserted_between(graph, edge):
                if edge in graph.data_edges:
                    rewritten_edges.append(edge)
                continue
            source_port = _port(
                source, edge.source, edge.source_port, PortDirection.OUTPUT
            )
            target_port = _port(
                target, edge.target, edge.target_port, PortDirection.INPUT
            )
            feedback = _feedback_consumer(target)
            if (
                source_port is None
                or target_port is None
                or feedback is None
                or feedback.completion != COMPLETE_BATCH
                or not _edge_targets_feedback_consumer(edge, target)
            ):
                if edge in graph.data_edges:
                    rewritten_edges.append(edge)
                continue
            adapter = _adapter_registration(
                context,
                graph,
                source,
                target,
                source_port,
                target_port,
            )
            if adapter is None:
                if edge in graph.data_edges:
                    rewritten_edges.append(edge)
                continue
            source_path = _node_path(
                source,
                role=edge.source.role,
                port=edge.source_port,
            )
            target_path = _node_path(
                target,
                role=edge.target.role,
                port=edge.target_port,
            )
            synthetic_id = f"__adapter_feedback_accumulator_{_edge_token(edge)}"
            insertion = AdapterInsertion(
                adapter_name=adapter.name,
                source_path=source_path,
                target_path=target_path,
            )
            inserted[synthetic_id] = ComponentNode(
                component_id=synthetic_id,
                component=AdapterComponent(
                    adapter=adapter,
                    insertion=insertion,
                    input_cardinality=source_port.cardinality,
                    output_cardinality=target_port.cardinality,
                ),
            )
            replacement = (
                DataEdge(
                    source=edge.source,
                    target=NodeRef(component_id=synthetic_id, role="predictor"),
                    source_port=edge.source_port,
                    target_port="input",
                ),
                DataEdge(
                    source=NodeRef(component_id=synthetic_id, role="predictor"),
                    target=edge.target,
                    source_port="output",
                    target_port=edge.target_port,
                ),
            )
            if edge in graph.data_edges:
                # The original edge is replaced; the virtual Stage-internal
                # edge has no original slot and is simply materialized.
                context.claim("data_edge", _edge_key(edge))
            for replacement_edge in replacement:
                context.claim("data_edge", _edge_key(replacement_edge))
            context.claim("node", synthetic_id)
            claims.update(context.claims)
            rewritten_edges.extend(replacement)

        # Keep original edge order and de-duplicate only the replacement edges
        # that can be encountered through multiple composed Stage ports.
        unique_edges: list[DataEdge] = []
        seen_edges: set[DataEdge] = set()
        for edge in rewritten_edges:
            if edge in seen_edges:
                continue
            seen_edges.add(edge)
            unique_edges.append(edge)
        if inserted:
            nodes.extend(inserted[key] for key in sorted(inserted))
        return ResolutionResult(
            graph=replace(graph, nodes=tuple(nodes), data_edges=tuple(unique_edges)),
            claims=frozenset(claims),
        )


@dataclass(frozen=True)
class _DeliveryProperties:
    channels: frozenset[str]
    sources: frozenset[str]
    completion: str
    ordering: str
    multiplicity: str
    grouping: str


def _runtime_delivery_properties(partial: bool) -> _DeliveryProperties:
    return _DeliveryProperties(
        channels=frozenset({"true"}),
        sources=frozenset({TRUE}),
        completion=PARTIAL_ALLOWED if partial else COMPLETE_BATCH,
        ordering=IN_ORDER,
        multiplicity=REPEATED_ALLOWED if partial else SINGLE,
        grouping=BY_PROPOSAL,
    )


def _lifecycle_mismatches(
    producer: _DeliveryProperties,
    consumer,
) -> tuple[str, ...]:
    mismatches: list[str] = []
    if not producer.channels <= consumer.accepted_channels:
        mismatches.append("accepted_channels")
    if not producer.sources <= consumer.accepted_sources:
        mismatches.append("accepted_sources")
    if producer.completion == PARTIAL_ALLOWED and consumer.completion == COMPLETE_BATCH:
        mismatches.append("completion")
    if producer.ordering == OUT_OF_ORDER_ALLOWED and consumer.ordering == IN_ORDER:
        mismatches.append("ordering")
    if producer.multiplicity == REPEATED_ALLOWED and consumer.multiplicity == SINGLE:
        mismatches.append("multiplicity")
    if producer.grouping != consumer.grouping:
        mismatches.append("grouping")
    return tuple(mismatches)


def _lifecycle_diagnostic(
    *,
    path: ContractPath,
    related: ContractPath,
    axes: tuple[str, ...],
) -> Diagnostic:
    rendered = ", ".join(axes)
    return Diagnostic(
        severity=Severity.ERROR,
        code="incompatible_feedback_lifecycle",
        message=(
            f"Feedback delivery at {related} is incompatible with the consumer "
            f"at {path} on: {rendered}."
        ),
        path=path,
        related=(related,),
        resolutions=(
            "Insert a lossless FeedbackAccumulator for partial delivery, or "
            "change the consumer FeedbackContract to declare the delivered policy.",
        ),
    )


class LifecycleCompatibilityRule:
    """Verify feedback delivery properties against consumer declarations."""

    namespace = "core"
    name = "lifecycle_compatibility"
    phase: Literal["verification"] = "verification"

    def apply(self, context: RuleContext) -> VerificationResult:
        """Compare every feedback connection after resolution."""
        from saealib.core.compiler.compiler import VerificationResult

        graph = context.graph
        partial = _partial_feedback_offered(context, graph)
        producer = _runtime_delivery_properties(partial)
        findings: list[Diagnostic] = []
        reported_targets: set[str] = set()
        for edge in sorted(context.graph.data_edges, key=_edge_key):
            try:
                target = graph.node_by_id(edge.target.component_id)
                source = graph.node_by_id(edge.source.component_id)
            except KeyError:
                continue
            consumer = _feedback_consumer(target)
            if consumer is None or not _edge_targets_feedback_consumer(edge, target):
                continue
            target_path = _node_path(
                target,
                role=edge.target.role,
                port=edge.target_port,
            )
            source_path = _node_path(
                source,
                role=edge.source.role,
                port=edge.source_port,
            )
            if _is_accumulator_node(source):
                reported_targets.add(target.component_id)
                continue
            axes = _lifecycle_mismatches(producer, consumer)
            if not axes:
                continue
            findings.append(
                _lifecycle_diagnostic(
                    path=target_path,
                    related=source_path,
                    axes=axes,
                )
            )
            reported_targets.add(target.component_id)

        # A StageNodeAdapter can contain both scheduler and consumer without a
        # pre-existing feedback edge.  FeedbackAccumulatorRule materializes the
        # internal edge; this fallback keeps a missing insertion diagnosable.
        if partial:
            for node in graph.nodes:
                if node.component_id in reported_targets:
                    continue
                consumer = _feedback_consumer(node)
                ports = _feedback_ports(node.contract)
                if (
                    consumer is None
                    or consumer.completion != COMPLETE_BATCH
                    or ports is None
                ):
                    continue
                (output_role, output_port), (input_role, input_port) = ports
                findings.append(
                    _lifecycle_diagnostic(
                        path=_node_path(
                            node,
                            role=input_role,
                            port=input_port.name,
                        ),
                        related=_node_path(
                            node,
                            role=output_role,
                            port=output_port.name,
                        ),
                        axes=("completion",),
                    )
                )
        return VerificationResult(diagnostics=tuple(findings))
