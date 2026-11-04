"""Lossless data adapters used during compiler resolution."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from saealib.core.compiler.diagnostics import ContractPath, Diagnostic, Severity
from saealib.core.compiler.graph import ComponentGraph, ComponentNode, DataEdge, NodeRef
from saealib.core.contracts import (
    DATA_SPEC_KINDS,
    MANY,
    ComponentContract,
    DataSpec,
    PortContract,
    PortDirection,
    PortSpec,
    is_data_spec_compatible,
)
from saealib.core.contracts.feedback import (
    COMPLETE_BATCH,
)
from saealib.core.contracts.schema import unify_data_specs
from saealib.core.contracts.vocabulary import (
    Vocabulary,
    VocabularyDescriptor,
    validate_name,
)
from saealib.exceptions import ConfigurationError, ValidationError

if TYPE_CHECKING:
    from saealib.core.compiler.compiler import RuleContext

Matcher = Callable[["AdapterMatchContext"], bool]

_ADAPTER_CATEGORY_DEFINITIONS = (
    ("identity", True, "An identity conversion that changes nothing."),
    (
        "lossless_view",
        True,
        "A view that changes access without changing values or meaning.",
    ),
    (
        "batch_buffering",
        True,
        "Buffering that is invisible to a timing-insensitive consumer.",
    ),
    (
        "immutable_clone",
        True,
        "An ownership-preserving immutable clone.",
    ),
    (
        "partial_feedback_accumulation",
        True,
        "Feedback accumulation for a complete-batch consumer.",
    ),
    ("text_embedding", False, "A text embedding requiring explicit user choice."),
    (
        "graph_embedding",
        False,
        "A graph embedding requiring explicit user choice.",
    ),
    (
        "ordinal_encoding",
        False,
        "An ordinal encoding requiring explicit user choice.",
    ),
    (
        "one_hot_encoding",
        False,
        "A one-hot encoding requiring explicit user choice.",
    ),
    ("latent_mapping", False, "A latent mapping requiring explicit user choice."),
    (
        "approximate_distance",
        False,
        "An approximate distance conversion requiring explicit user choice.",
    ),
    ("feature_encoder", False, "A feature encoder requiring explicit user choice."),
)
ADAPTER_CATEGORIES: Vocabulary[VocabularyDescriptor] = Vocabulary()
for _category, _, _description in _ADAPTER_CATEGORY_DEFINITIONS:
    ADAPTER_CATEGORIES.register(
        _category,
        VocabularyDescriptor(name=_category, description=_description),
    )
_AUTO_INSERTABLE_CATEGORIES = frozenset(
    category
    for category, auto_insertable, _ in _ADAPTER_CATEGORY_DEFINITIONS
    if auto_insertable
)


@dataclass(frozen=True, kw_only=True)
class Adapter:
    """One named conversion between two nominal data specifications."""

    name: str
    source: DataSpec
    target: DataSpec
    lossless: bool
    auto_insertable: bool
    namespace: str = "core"
    category: str = "lossless_view"
    matcher: Matcher | None = None

    def __post_init__(self) -> None:
        validate_name(self.name)
        validate_name(self.namespace)
        validate_name(self.category)
        if ADAPTER_CATEGORIES.get(self.category) is None:
            raise ValidationError(f"Unknown adapter category: {self.category!r}")
        if not isinstance(self.source, DataSpec) or not isinstance(
            self.target, DataSpec
        ):
            raise ValidationError("Adapter source and target must be DataSpec values")
        if not isinstance(self.lossless, bool) or not isinstance(
            self.auto_insertable, bool
        ):
            raise ValidationError(
                "Adapter lossless and auto_insertable must be booleans"
            )
        if self.matcher is not None and not callable(self.matcher):
            raise ValidationError("Adapter matcher must be callable")

    @property
    def eligible_for_automatic_insertion(self) -> bool:
        """Return whether this adapter may be selected by the compiler."""
        return (
            self.lossless
            and self.auto_insertable
            and self.category in _AUTO_INSERTABLE_CATEGORIES
        )


@dataclass(frozen=True, kw_only=True)
class AdapterMatchContext:
    """Immutable, deliberately small context exposed to adapter matchers."""

    source_node: ComponentNode
    target_node: ComponentNode
    source_port: PortSpec
    target_port: PortSpec
    compile_context: object
    graph: ComponentGraph | None = None


class AdapterRegistry:
    """Registry of adapters with deterministic, name-based candidate ordering."""

    def __init__(self, adapters: Iterable[Adapter] = ()) -> None:
        self._adapters: dict[str, Adapter] = {}
        for adapter in adapters:
            self.register(adapter)

    def register(self, adapter: Adapter) -> Adapter:
        """Register an adapter under its unique name."""
        if not isinstance(adapter, Adapter):
            raise ValidationError("AdapterRegistry accepts Adapter values")
        if adapter.name in self._adapters:
            raise ConfigurationError(f"Adapter is already registered: {adapter.name!r}")
        self._adapters[adapter.name] = adapter
        return adapter

    def registrations(self) -> tuple[Adapter, ...]:
        """Return registrations in insertion order."""
        return tuple(self._adapters.values())

    def adapters(self) -> tuple[Adapter, ...]:
        """Return the registered adapters."""
        return self.registrations()

    def candidates(
        self,
        producer: DataSpec,
        consumer: DataSpec,
        *,
        compile_context: object | None = None,
        source_node: ComponentNode | None = None,
        target_node: ComponentNode | None = None,
        source_port: PortSpec | None = None,
        target_port: PortSpec | None = None,
        graph: ComponentGraph | None = None,
    ) -> tuple[Adapter, ...]:
        """Return deterministic adapters compatible with both endpoints."""
        enabled = getattr(compile_context, "enabled_rule_namespaces", frozenset())
        allowed = {"core", *enabled}
        result: list[Adapter] = []
        for adapter in sorted(self._adapters.values(), key=lambda item: item.name):
            if (
                adapter.namespace not in allowed
                or not adapter.eligible_for_automatic_insertion
            ):
                continue
            if not is_data_spec_compatible(
                producer, adapter.source, registry=DATA_SPEC_KINDS
            ):
                continue
            if not is_data_spec_compatible(
                adapter.target, consumer, registry=DATA_SPEC_KINDS
            ):
                continue
            if not unify_data_specs(producer, adapter.source).unified:
                continue
            if not unify_data_specs(adapter.target, consumer).unified:
                continue
            if adapter.matcher is not None:
                if (
                    source_node is None
                    or target_node is None
                    or source_port is None
                    or target_port is None
                ):
                    continue
                context = AdapterMatchContext(
                    source_node=source_node,
                    target_node=target_node,
                    source_port=source_port,
                    target_port=target_port,
                    compile_context=compile_context,
                    graph=graph,
                )
                if not adapter.matcher(context):
                    continue
            result.append(adapter)
        return tuple(result)


@dataclass(frozen=True, kw_only=True)
class AdapterComponent:
    """Synthetic component representing one inserted adapter."""

    adapter: Adapter
    insertion: AdapterInsertion | None = None
    input_cardinality: str = MANY
    output_cardinality: str = MANY

    def _build_contract(self) -> ComponentContract:
        return ComponentContract(
            ports={
                "predictor": PortContract(
                    inputs=(
                        PortSpec(
                            name="input",
                            direction=PortDirection.INPUT,
                            data=self.adapter.source,
                            cardinality=self.input_cardinality,
                        ),
                    ),
                    outputs=(
                        PortSpec(
                            name="output",
                            direction=PortDirection.OUTPUT,
                            data=self.adapter.target,
                            cardinality=self.output_cardinality,
                        ),
                    ),
                )
            }
        )

    def contract(self) -> ComponentContract:
        """Return the synthetic component's input and output contract."""
        return self._build_contract()


@dataclass(frozen=True, kw_only=True)
class AdapterInsertion:
    """Description of an adapter inserted between two component paths."""

    adapter_name: str
    source_path: ContractPath
    target_path: ContractPath

    def __str__(self) -> str:
        return f"{self.adapter_name}: {self.source_path} -> {self.target_path}"

    def __repr__(self) -> str:
        return f"AdapterInsertion({self})"


def _port(
    node: ComponentNode, reference: NodeRef, name: str, direction: PortDirection
) -> PortSpec | None:
    role = reference.role
    contracts = (
        ((role, node.contract.ports.get(role)),)
        if role
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


def _partial_feedback_is_offered(
    compile_context: object,
    *,
    graph: ComponentGraph | None = None,
    nodes: tuple[ComponentNode | None, ...] = (),
) -> bool:
    """Whether feedback can reach a consumer in partial deliveries.

    A structured graph with regions runs each evaluation plan inside its
    region and delivers the plan's feedback once the plan completes, so only
    region-less graphs see the partial delivery a runtime offers.
    """
    from saealib.core.compiler.structured import StructuredGraph

    offered = getattr(compile_context, "offered_runtime_capabilities", None)
    if "partial_feedback" not in (offered if offered is not None else frozenset()):
        return False
    return not (isinstance(graph, StructuredGraph) and graph.region_nodes)


def _target_requires_complete_batch(node: ComponentNode) -> bool:
    feedback = getattr(node.contract.lifecycle, "feedback", None)
    return (
        feedback is not None and getattr(feedback, "completion", None) == COMPLETE_BATCH
    )


def _partial_complete_feedback_pair(match: AdapterMatchContext) -> bool:
    return _target_requires_complete_batch(
        match.target_node
    ) and _partial_feedback_is_offered(
        match.compile_context,
        graph=match.graph,
        nodes=(match.source_node, match.target_node),
    )


def _feedback_accumulator_match(match: AdapterMatchContext) -> bool:
    return _partial_complete_feedback_pair(match)


DEFAULT_ADAPTER_REGISTRY = AdapterRegistry(
    (
        Adapter(
            name="feedback_accumulator",
            source=DataSpec(kind="FeedbackBatch"),
            target=DataSpec(kind="FeedbackBatch"),
            lossless=True,
            auto_insertable=True,
            category="partial_feedback_accumulation",
            matcher=_feedback_accumulator_match,
        ),
    )
)


class LosslessAdapterRule:
    """Insert one unambiguous lossless adapter for incompatible data edges."""

    namespace = "core"
    name = "lossless_adapter"
    phase = "resolution"

    def __init__(self, registry: AdapterRegistry | None = None) -> None:
        self.registry = DEFAULT_ADAPTER_REGISTRY if registry is None else registry

    def apply(self, context: RuleContext):
        """Rewrite each edge with exactly one unambiguous automatic adapter."""
        from dataclasses import replace

        from saealib.core.compiler.compiler import ResolutionResult
        from saealib.core.contracts.ports import check_port_compatibility

        graph = context.graph
        registry = getattr(context.compile_context, "adapter_registry", None)
        if registry is None:
            registry = self.registry
        nodes = list(graph.nodes)
        edges: list[DataEdge] = []
        findings: list[Diagnostic] = []
        inserted: dict[str, ComponentNode] = {}
        for edge in graph.data_edges:
            source = graph.node_by_id(edge.source.component_id)
            target = graph.node_by_id(edge.target.component_id)
            source_port = _port(
                source, edge.source, edge.source_port, PortDirection.OUTPUT
            )
            target_port = _port(
                target, edge.target, edge.target_port, PortDirection.INPUT
            )
            if (
                source_port is None
                or target_port is None
                or check_port_compatibility(source_port, target_port).compatible
            ):
                edges.append(edge)
                continue
            if (
                source_port.data.kind == "FeedbackBatch"
                and _partial_complete_feedback_pair(
                    AdapterMatchContext(
                        source_node=source,
                        target_node=target,
                        source_port=source_port,
                        target_port=target_port,
                        compile_context=context.compile_context,
                        graph=graph,
                    )
                )
            ):
                # FeedbackAccumulatorRule owns this lifecycle rewrite. Leaving
                # the edge untouched here prevents two rules from claiming the
                # same lifecycle connection.
                edges.append(edge)
                continue
            candidates = registry.candidates(
                source_port.data,
                target_port.data,
                compile_context=context.compile_context,
                source_node=source,
                target_node=target,
                source_port=source_port,
                target_port=target_port,
                graph=graph,
            )
            source_path = ContractPath(
                components=(source.component_id,),
                role=edge.source.role,
                port=edge.source_port,
            )
            target_path = ContractPath(
                components=(target.component_id,),
                role=edge.target.role,
                port=edge.target_port,
            )
            if len(candidates) > 1:
                findings.append(
                    Diagnostic(
                        severity=Severity.WARNING,
                        code="ambiguous_adapter",
                        message=(
                            f"Connection {source_path} -> {target_path} has multiple "
                            "automatic adapters: "
                            f"{', '.join(item.name for item in candidates)}; none is "
                            "inserted, so the connection cannot be verified."
                        ),
                        path=source_path,
                        related=(target_path,),
                        resolutions=(
                            "Name one adapter explicitly or remove the ambiguity.",
                        ),
                    )
                )
                edges.append(edge)
                continue
            if not candidates:
                edges.append(edge)
                continue
            adapter = candidates[0]
            edge_token = _edge_token(edge)
            synthetic_id = f"__adapter_{adapter.name}_{edge_token}"
            insertion = AdapterInsertion(
                adapter_name=adapter.name,
                source_path=source_path,
                target_path=target_path,
            )
            component = ComponentNode(
                component_id=synthetic_id,
                component=AdapterComponent(
                    adapter=adapter,
                    insertion=insertion,
                    input_cardinality=source_port.cardinality,
                    output_cardinality=target_port.cardinality,
                ),
            )
            inserted[synthetic_id] = component
            edges.extend(
                (
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
            )
            context.claim("data_edge", _edge_key(edge))
            context.claim("node", synthetic_id)
            context.claim("data_edge", _edge_key(edges[-2]))
            context.claim("data_edge", _edge_key(edges[-1]))
        if inserted:
            nodes.extend(inserted[key] for key in sorted(inserted))
        return ResolutionResult(
            graph=replace(graph, nodes=tuple(nodes), data_edges=tuple(edges)),
            claims=context.claims,
            diagnostics=tuple(findings),
        )


def _edge_key(edge: DataEdge) -> str:
    def ref(value: NodeRef) -> str:
        return value.component_id + (f"[{value.role}]" if value.role else "")

    return (
        f"{ref(edge.source)}.{edge.source_port}->{ref(edge.target)}.{edge.target_port}"
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


__all__ = [
    "ADAPTER_CATEGORIES",
    "DEFAULT_ADAPTER_REGISTRY",
    "Adapter",
    "AdapterComponent",
    "AdapterInsertion",
    "AdapterMatchContext",
    "AdapterRegistry",
    "LosslessAdapterRule",
]
