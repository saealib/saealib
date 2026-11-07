"""Resolve schema variables on declared graph connections."""

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
    Product,
    SchemaBinding,
    Var,
)
from saealib.core.contracts.flow import _apply_transforms
from saealib.core.contracts.ports import PortSpec
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


def _contains_var(binding: object) -> bool:
    if isinstance(binding, Var):
        return True
    if isinstance(binding, Product):
        return any(_contains_var(item) for item in binding.elements)
    return False


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
    """Freshen edge-local variable names and unify declared data edges."""

    namespace = "core"
    name = "schema_binding"
    phase = "resolution"

    def apply(self, context: RuleContext):
        """Resolve schema constraints for each declared data connection."""
        from saealib.core.compiler.compiler import (
            ResolutionResult,
            _resolvable_service_names,
        )

        graph = context.graph
        # ServiceResolutionRule owns the same node claim. Defer schema
        # unification until declared service bindings settle.
        resolvable = _resolvable_service_names(context.compile_context)
        awaiting = frozenset(
            node.component_id
            for node in graph.nodes
            if _awaits_service_resolution(node, resolvable)
        )
        if awaiting:
            return ResolutionResult(graph=graph, diagnostics=())
        current = graph
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
            if edge.transforms:
                transformed, error = _apply_transforms(
                    source_port.data, edge.transforms
                )
                if error is not None or transformed is None:
                    continue
                source_port = replace(source_port, data=transformed)
            source_port = replace(
                source_port,
                data=_fresh_spec(source_port.data, edge.source.component_id),
            )
            target_port = replace(
                target_port,
                data=_fresh_spec(target_port.data, edge.target.component_id),
            )
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
        return ResolutionResult(graph=current, diagnostics=tuple(findings))


def _edge_key(edge: DataEdge) -> str:
    def ref(value: NodeRef) -> str:
        return value.component_id + (f"[{value.role}]" if value.role else "")

    base = (
        f"{ref(edge.source)}.{edge.source_port}->{ref(edge.target)}.{edge.target_port}"
    )
    return base + "|" + ",".join(item.name for item in edge.transforms)


__all__ = ["SchemaBindingRule"]
