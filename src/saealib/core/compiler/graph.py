"""The graph vocabulary used by the compiler boundary."""

from __future__ import annotations

from collections.abc import Mapping
from copy import copy
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal, TypeAlias

from saealib.core.compiler.diagnostics import (
    ContractPath,
    Diagnostic,
    DiagnosticBag,
    Severity,
)
from saealib.core.contracts.contract import ComponentContract
from saealib.core.contracts.flow import Transform
from saealib.core.contracts.roles import RoleName
from saealib.core.contracts.vocabulary import validate_name
from saealib.core.state.keys import StateKey
from saealib.exceptions import ValidationError

if TYPE_CHECKING:
    from saealib.core.compiler.compiler import RuleContext, VerificationResult

__all__ = [
    "ComponentBindings",
    "ComponentGraph",
    "ComponentId",
    "ComponentNode",
    "ControlEdge",
    "DataEdge",
    "GraphTemplate",
    "IdentityRule",
    "NodeRef",
    "ReachabilityRule",
    "StateBinding",
]

ComponentId: TypeAlias = str


def _name(value: str, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValidationError(f"{label} must be a non-empty string")
    return validate_name(value)


def _component_name(value: str, label: str) -> str:
    """Validate plain or dot-qualified component identifiers."""
    if not isinstance(value, str) or not value:
        raise ValidationError(f"{label} must be a non-empty string")
    parts = value.split(".")
    if any(not part for part in parts):
        raise ValidationError(f"{label} must contain non-empty identifiers")
    for part in parts:
        validate_name(part)
    return value


@dataclass(frozen=True, kw_only=True)
class NodeRef:
    """Identify a graph node, optionally in one of its component roles."""

    component_id: ComponentId
    role: RoleName | None = None

    def __post_init__(self) -> None:
        _component_name(self.component_id, "NodeRef component_id")
        if self.role is not None:
            _name(self.role, "NodeRef role")

    @classmethod
    def from_value(cls, value: object) -> NodeRef:
        """Normalize the convenient endpoint forms accepted by the API."""
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            return cls(component_id=value)
        if isinstance(value, tuple) and len(value) in (1, 2):
            return cls(
                component_id=value[0], role=value[1] if len(value) == 2 else None
            )
        raise ValidationError(
            "Node endpoints must be NodeRef, string, or (component_id, role)"
        )


@dataclass(frozen=True, kw_only=True)
class ComponentNode:
    """Hold a component instance and its compilation-local contract snapshot."""

    component_id: ComponentId
    component: object
    role: RoleName | None = None
    resolved_services: Mapping[str, object] = field(default_factory=dict)
    _contract_snapshot: ComponentContract | None = field(
        init=False, default=None, repr=False, compare=False
    )
    _analysis_contract_snapshot: ComponentContract | None = field(
        init=False, default=None, repr=False, compare=False
    )
    _verification_only: bool = field(
        init=False, default=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        _component_name(self.component_id, "ComponentNode component_id")
        if self.role is not None:
            _name(self.role, "ComponentNode role")
        resolved_services = dict(self.resolved_services)
        for service_name in resolved_services:
            _name(service_name, "ComponentNode resolved service name")
        object.__setattr__(
            self,
            "resolved_services",
            MappingProxyType(resolved_services),
        )
        contract_method = getattr(self.component, "contract", None)
        if not callable(contract_method):
            raise ValidationError("ComponentNode component must provide contract()")

    @property
    def contract(self) -> ComponentContract:
        """Return the cached execution contract, taking a snapshot on demand.

        Compilation calls :meth:`with_contract_snapshot` before any compiler rule
        can read this property.  The fallback keeps the graph vocabulary usable
        by callers that inspect a node outside compilation, without doing work in
        ``ComponentNode`` construction.
        """
        contract = self._contract_snapshot
        if contract is None:
            contract = self._read_contract()
            object.__setattr__(self, "_contract_snapshot", contract)
        return contract

    @property
    def analysis_contract(self) -> ComponentContract:
        """Return the contract used to verify the component's graph behavior."""
        contract = self._analysis_contract_snapshot
        if contract is None:
            contract = self._read_analysis_contract()
            object.__setattr__(self, "_analysis_contract_snapshot", contract)
        return contract

    def _read_contract(self) -> ComponentContract:
        contract_method = getattr(self.component, "runtime_contract", None)
        if not callable(contract_method):
            contract_method = getattr(self.component, "contract", None)
        if not callable(contract_method):
            raise ValidationError("ComponentNode component must provide contract()")
        contract = contract_method()
        if not isinstance(contract, ComponentContract):
            raise ValidationError(
                "ComponentNode contract() must return ComponentContract"
            )
        return contract

    def _read_analysis_contract(
        self, *, fallback: ComponentContract | None = None
    ) -> ComponentContract:
        contract_method = getattr(self.component, "analysis_contract", None)
        if callable(contract_method):
            contract = contract_method()
        elif fallback is not None:
            contract = fallback
        else:
            contract = self._read_contract()
        if not isinstance(contract, ComponentContract):
            raise ValidationError(
                "ComponentNode analysis_contract() must return ComponentContract"
            )
        return contract

    def with_contract_snapshot(self, *, refresh: bool = False) -> ComponentNode:
        """Return a node with one immutable compilation snapshot.

        ``refresh`` is used only for the first pass of a new compilation.  A
        subsequent resolution pass preserves the snapshot already attached to
        the node, including when resolution only changes service bindings.
        """
        if (
            not refresh
            and self._contract_snapshot is not None
            and self._analysis_contract_snapshot is not None
        ):
            return self
        result = copy(self)
        contract = self._read_contract()
        object.__setattr__(result, "_contract_snapshot", contract)
        object.__setattr__(
            result,
            "_analysis_contract_snapshot",
            self._read_analysis_contract(fallback=contract),
        )
        return result

    def with_resolved_services(
        self, resolved_services: Mapping[str, object]
    ) -> ComponentNode:
        """Return a service-enriched node without re-reading its contract.

        Resolution rules may add service bindings later, but rebuilding the
        dataclass with ``dataclasses.replace`` must not invoke ``contract()``.
        """
        values = dict(resolved_services)
        for service_name in values:
            _name(service_name, "ComponentNode resolved service name")
        result = copy(self)
        object.__setattr__(result, "resolved_services", MappingProxyType(values))
        return result


@dataclass(frozen=True, kw_only=True)
class DataEdge:
    """Connect a source port to a target port."""

    source: NodeRef
    target: NodeRef
    source_port: str
    target_port: str
    transforms: tuple[Transform, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "source", NodeRef.from_value(self.source))
        object.__setattr__(self, "target", NodeRef.from_value(self.target))
        object.__setattr__(self, "source_port", _name(self.source_port, "source_port"))
        object.__setattr__(self, "target_port", _name(self.target_port, "target_port"))
        transforms = tuple(self.transforms)
        if any(not isinstance(item, Transform) for item in transforms):
            raise ValidationError("DataEdge transforms must contain Transform values")
        object.__setattr__(self, "transforms", transforms)


@dataclass(frozen=True, kw_only=True)
class ControlEdge:
    """Express control dependency between two graph nodes."""

    source: NodeRef
    target: NodeRef

    def __post_init__(self) -> None:
        object.__setattr__(self, "source", NodeRef.from_value(self.source))
        object.__setattr__(self, "target", NodeRef.from_value(self.target))


@dataclass(frozen=True, kw_only=True)
class StateBinding:
    """Bind a component node to an actual state key."""

    node: NodeRef
    state_key: StateKey[object]

    def __post_init__(self) -> None:
        object.__setattr__(self, "node", NodeRef.from_value(self.node))
        if not isinstance(self.state_key, StateKey):
            raise ValidationError("StateBinding state_key must be a StateKey")


@dataclass(frozen=True, kw_only=True)
class ComponentGraph:
    """A self-contained graph and its declared entry points."""

    nodes: tuple[ComponentNode, ...]
    data_edges: tuple[DataEdge, ...] = ()
    control_edges: tuple[ControlEdge, ...] = ()
    state_bindings: tuple[StateBinding, ...] = ()
    entry_points: tuple[NodeRef, ...] = ()

    def __post_init__(self) -> None:
        nodes = tuple(self.nodes)
        if any(not isinstance(node, ComponentNode) for node in nodes):
            raise ValidationError("ComponentGraph nodes must be ComponentNode values")
        object.__setattr__(self, "nodes", nodes)
        for name, expected in (
            ("data_edges", DataEdge),
            ("control_edges", ControlEdge),
            ("state_bindings", StateBinding),
        ):
            values = tuple(getattr(self, name))
            if any(not isinstance(value, expected) for value in values):
                raise ValidationError(f"{name} must contain {expected.__name__} values")
            object.__setattr__(self, name, values)
        entries = tuple(NodeRef.from_value(entry) for entry in self.entry_points)
        object.__setattr__(self, "entry_points", entries)

    def node_by_id(self, component_id: ComponentId) -> ComponentNode:
        """Return the first node with the requested component ID."""
        for node in self.nodes:
            if node.component_id == component_id:
                return node
        raise KeyError(component_id)

    def well_formedness(self) -> DiagnosticBag:
        """Collect structural and graph-level diagnostics without raising."""
        findings = DiagnosticBag()
        ids = tuple(node.component_id for node in self.nodes)
        known = set(ids)
        for edge in (*self.data_edges, *self.control_edges):
            for endpoint in (edge.source, edge.target):
                if endpoint.component_id not in known:
                    findings.append(
                        _diagnostic(
                            "invalid_graph_edge",
                            endpoint.component_id,
                            f"Edge endpoint {endpoint.component_id!r} is not a graph "
                            "node.",
                            "Add the referenced node or remove the edge.",
                        )
                    )
        if not self.entry_points:
            findings.append(
                _diagnostic(
                    "invalid_entry_point",
                    "graph",
                    "The graph has no entry point.",
                    "Declare at least one entry point.",
                )
            )
        for entry in self.entry_points:
            if entry.component_id not in known:
                findings.append(
                    _diagnostic(
                        "invalid_entry_point",
                        entry.component_id,
                        f"Entry point {entry.component_id!r} is not a graph node.",
                        "Declare an existing node as the entry point.",
                    )
                )
        for binding in self.state_bindings:
            if binding.node.component_id not in known:
                key = binding.state_key
                findings.append(
                    _diagnostic(
                        "invalid_state_binding",
                        binding.node.component_id,
                        f"State binding for {key.namespace}:{key.name}:"
                        f"{key.schema_version} references "
                        f"{binding.node.component_id!r}, which is not a graph node.",
                        "Add the referenced node or remove the state binding.",
                    )
                )
        return findings


def _diagnostic(
    code: str, component_id: str, message: str, resolution: str
) -> Diagnostic:
    return Diagnostic(
        severity=Severity.ERROR,
        code=code,
        message=message,
        path=ContractPath(components=(component_id,)),
        resolutions=(resolution,),
    )


@dataclass(frozen=True, kw_only=True)
class ComponentBindings:
    """The minimal immutable component mapping supplied to a template."""

    components: Mapping[ComponentId, object]

    def __post_init__(self) -> None:
        values = dict(self.components)
        for component_id in values:
            _name(component_id, "ComponentBindings component id")
        object.__setattr__(self, "components", MappingProxyType(values))


class GraphTemplate:
    """Provide the graph-building seam without defining compilation policy."""

    def build_graph(self, bindings: ComponentBindings) -> ComponentGraph:
        """Build a component graph from bound components."""
        raise NotImplementedError


class IdentityRule:
    """Check graph-level component identity without inspecting concrete types."""

    namespace = "core"
    name = "identity"
    phase: Literal["verification"] = "verification"

    def apply(self, context: RuleContext) -> VerificationResult:
        """Apply through the compiler rule protocol."""
        from saealib.core.compiler.compiler import VerificationResult

        return VerificationResult(
            diagnostics=tuple(_identity_diagnostics(context.graph))
        )


def _identity_diagnostics(graph: ComponentGraph) -> DiagnosticBag:
    bag = DiagnosticBag()
    ids = tuple(node.component_id for node in graph.nodes)
    for component_id in dict.fromkeys(ids):
        if ids.count(component_id) > 1:
            bag.append(
                _diagnostic(
                    "duplicate_component_id",
                    component_id,
                    f"ComponentId {component_id!r} is duplicated.",
                    "Use one node per ComponentId.",
                )
            )
    return bag


class ReachabilityRule:
    """Check reachability over both data and control edges."""

    namespace = "core"
    name = "reachability"
    phase: Literal["verification"] = "verification"

    def apply(self, context: RuleContext) -> VerificationResult:
        """Apply through the compiler rule protocol."""
        from saealib.core.compiler.compiler import VerificationResult

        return VerificationResult(
            diagnostics=tuple(_reachability_diagnostics(context.graph))
        )


def _reachability_diagnostics(graph: ComponentGraph) -> DiagnosticBag:
    bag = DiagnosticBag()
    known = {node.component_id for node in graph.nodes}
    reachable = {
        entry.component_id
        for entry in graph.entry_points
        if entry.component_id in known
    }
    edges = (*graph.data_edges, *graph.control_edges)
    changed = True
    while changed:
        changed = False
        for edge in edges:
            if (
                edge.source.component_id in reachable
                and edge.target.component_id in known
                and edge.target.component_id not in reachable
            ):
                reachable.add(edge.target.component_id)
                changed = True
    for node in graph.nodes:
        if node.component_id not in reachable:
            bag.append(
                _diagnostic(
                    "unreachable_node",
                    node.component_id,
                    f"Node {node.component_id!r} is unreachable from the entry points.",
                    "Connect the node to an entry point or remove it.",
                )
            )
    return bag
