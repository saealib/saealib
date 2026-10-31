"""Schema-variable freshening and graph-wide schema resolution."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from saealib.core.compiler.diagnostics import (
    ContractPath,
    Diagnostic,
    Severity,
    near_miss_suggestions,
)
from saealib.core.compiler.graph import ComponentNode, DataEdge, NodeRef
from saealib.core.contracts import (
    SCHEMA_VARIABLES,
    ComponentContract,
    DataSpec,
    PartSpec,
    Product,
    SchemaBinding,
    Var,
)
from saealib.core.contracts.ports import PortContract, PortSpec
from saealib.core.contracts.schema import (
    SchemaConstraint,
    Substitution,
    unify_data_specs,
)

if TYPE_CHECKING:
    from saealib.core.compiler.compiler import RuleContext


def _fresh_binding(binding: SchemaBinding, prefix: str) -> SchemaBinding:
    if isinstance(binding, Var):
        safe_prefix = prefix.replace(".", "_")
        return Var(name=f"{safe_prefix}__{binding.name}")
    if isinstance(binding, Product):
        return Product(
            elements=tuple(_fresh_binding(item, prefix) for item in binding.elements)
        )
    return binding


def _fresh_spec(spec: DataSpec, prefix: str) -> DataSpec:
    return DataSpec(
        kind=spec.kind,
        bindings={
            name: _fresh_binding(value, prefix) for name, value in spec.bindings.items()
        },
    )


def _fresh_contract(contract: ComponentContract, prefix: str) -> ComponentContract:
    ports: dict[str, PortContract] = {}
    for role, role_contract in contract.ports.items():
        ports[role] = PortContract(
            inputs=tuple(
                replace(port, data=_fresh_spec(port.data, prefix))
                for port in role_contract.inputs
            ),
            outputs=tuple(
                replace(port, data=_fresh_spec(port.data, prefix))
                for port in role_contract.outputs
            ),
        )
    parts = tuple(
        PartSpec(
            name=part.name,
            contract=_fresh_contract(part.contract, prefix),
            optional=part.optional,
        )
        for part in contract.parts
    )
    return replace(contract, ports=ports, parts=parts)


class _FreshenedComponent:
    _saealib_schema_freshened = True

    def __init__(self, component: object, contract: ComponentContract) -> None:
        self._component = component
        self._contract = contract

    def contract(self) -> ComponentContract:
        return self._contract

    def __getattr__(self, name: str) -> object:
        return getattr(self._component, name)


def _contains_var(binding: object) -> bool:
    if isinstance(binding, Var):
        return True
    if isinstance(binding, Product):
        return any(_contains_var(item) for item in binding.elements)
    return False


def _contract_contains_var(contract: ComponentContract) -> bool:
    return any(
        _contains_var(port.data.bindings.get(name))
        for role in contract.ports.values()
        for port in (*role.inputs, *role.outputs)
        for name in port.data.bindings
    ) or any(_contract_contains_var(part.contract) for part in contract.parts)


def _declared_service_names(contract: ComponentContract) -> set[str]:
    declared: set[str] = set()

    def collect(contract: ComponentContract) -> None:
        declared.update(requirement.name for requirement in contract.required_services)
        for role in contract.ports.values():
            for port in (*role.inputs, *role.outputs):
                declared.update(
                    requirement.name for requirement in port.required_services
                )
        for part in contract.parts:
            collect(part.contract)

    collect(contract)
    return declared


def _awaits_service_resolution(node: ComponentNode, resolvable: frozenset[str]) -> bool:
    """Whether a declared service can still be resolved for this node.

    Such a node is rewritten by service resolution in the same iteration, so
    freshening it here would double-claim it; it is freshened once services
    settle on a later pass.
    """
    pending = _declared_service_names(node.contract) - set(node.resolved_services)
    return bool(pending & resolvable)


def _binding_has_unresolved_var(
    binding: SchemaBinding, substitution: Substitution
) -> bool:
    resolved = substitution.resolve(binding)
    if isinstance(resolved, Var):
        return True
    if isinstance(resolved, Product):
        return any(
            _binding_has_unresolved_var(element, substitution)
            for element in resolved.elements
        )
    return False


def _port(
    node: ComponentNode, reference: NodeRef, name: str, output: bool
) -> PortSpec | None:
    contracts = (
        ((reference.role, node.contract.ports.get(reference.role)),)
        if reference.role
        else tuple(sorted(node.contract.ports.items()))
    )
    ports = [
        port
        for _, contract in contracts
        if contract is not None
        for port in (contract.outputs if output else contract.inputs)
        if port.name == name
    ]
    return ports[0] if len(ports) == 1 else None


class SchemaBindingRule:
    """Freshen variables per node and unify every data edge deterministically."""

    namespace = "core"
    name = "schema_binding"
    phase = "resolution"

    def apply(self, context: RuleContext):
        """Freshen contracts once, then resolve graph-wide schema constraints."""
        from saealib.core.compiler.compiler import (
            ResolutionResult,
            _resolvable_service_names,
        )

        graph = context.graph
        # ServiceResolutionRule owns the same node claim.  A node whose
        # declared services still resolve is rewritten by it in this same
        # iteration, so it is freshened only after services settle.
        resolvable = _resolvable_service_names(context.compile_context)
        awaiting = frozenset(
            node.component_id
            for node in graph.nodes
            if _awaits_service_resolution(node, resolvable)
        )
        nodes = tuple(
            node if node.component_id in awaiting else _freshen_node(node)
            for node in graph.nodes
        )
        current = replace(graph, nodes=nodes)
        claims = frozenset(
            context.claim("node", node.component_id)
            for old, node in zip(graph.nodes, current.nodes)
            if old != node
        )
        if awaiting:
            # An awaiting node is not freshened yet, so its unqualified
            # variables would merge across independent nodes in the shared
            # substitution below. Defer graph-wide unification to the next
            # iteration, after service resolution has settled.
            return ResolutionResult(graph=current, claims=claims, diagnostics=())
        substitution = Substitution()
        findings: list[Diagnostic] = []
        deferred: list[tuple[ContractPath, SchemaConstraint, SchemaBinding]] = []
        for edge in sorted(current.data_edges, key=_edge_key):
            source = current.node_by_id(edge.source.component_id)
            target = current.node_by_id(edge.target.component_id)
            source_port = _port(source, edge.source, edge.source_port, True)
            target_port = _port(target, edge.target, edge.target_port, False)
            if source_port is None or target_port is None:
                continue
            result = unify_data_specs(
                source_port.data, target_port.data, substitution=substitution
            )
            substitution = result.substitution
            path = ContractPath(
                components=(edge.source.component_id,),
                role=edge.source.role,
                port=edge.source_port,
            )
            for variable in result.unknown_variables:
                suggestions = near_miss_suggestions(variable, SCHEMA_VARIABLES)
                suggestion = (
                    f" Registered names similar to {variable!r}: "
                    f"{', '.join(suggestions)}."
                    if suggestions
                    else ""
                )
                findings.append(
                    Diagnostic(
                        severity=Severity.WARNING,
                        code="unknown_schema_variable",
                        message=(
                            f"Schema variable {variable!r} is not registered at "
                            f"{path}, so this binding cannot be verified."
                            f"{suggestion}"
                        ),
                        path=path,
                        resolutions=(
                            "Use a registered schema variable or a fixed binding.",
                        ),
                    )
                )
            for constraint in result.conflicts:
                findings.append(
                    Diagnostic(
                        severity=Severity.ERROR,
                        code="incompatible_port",
                        message=f"Schema binding conflict at {path}: {constraint}.",
                        path=path,
                        resolutions=(
                            "Align the producer and consumer schema bindings.",
                        ),
                    )
                )
            for constraint in result.deferred:
                producer_binding = source_port.data.bindings.get(constraint.variable)
                if producer_binding is not None:
                    deferred.append((path, constraint, producer_binding))
        for path, constraint, producer_binding in deferred:
            if _binding_has_unresolved_var(producer_binding, substitution):
                findings.append(
                    Diagnostic(
                        severity=Severity.WARNING,
                        code="schema_variable_unbound",
                        message=(
                            f"Schema variable remains deferred at {path}: "
                            f"{constraint}; the connection cannot be fully "
                            "verified."
                        ),
                        path=path,
                        resolutions=(
                            "Bind the schema variable before compilation completes.",
                        ),
                    )
                )
        return ResolutionResult(
            graph=current, claims=claims, diagnostics=tuple(findings)
        )


def _freshen_node(node: ComponentNode) -> ComponentNode:
    if getattr(node.component, "_saealib_schema_freshened", False):
        return node
    if not _contract_contains_var(node.contract):
        return node
    return ComponentNode(
        component_id=node.component_id,
        role=node.role,
        component=_FreshenedComponent(
            node.component, _fresh_contract(node.contract, node.component_id)
        ),
        resolved_services=node.resolved_services,
    )


def _edge_key(edge: DataEdge) -> str:
    def ref(value: NodeRef) -> str:
        return value.component_id + (f"[{value.role}]" if value.role else "")

    return (
        f"{ref(edge.source)}.{edge.source_port}->{ref(edge.target)}.{edge.target_port}"
    )


__all__ = ["SchemaBindingRule"]
