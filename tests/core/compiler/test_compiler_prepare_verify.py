"""Compile composition over the prepare/verify split."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

import numpy as np
import pytest

from saealib.core.compiler import (
    CompileContext,
    Compiler,
    ComponentGraph,
    ComponentNode,
    NodeRef,
    ResolutionResult,
    RuleContext,
    RuleRegistry,
    VerificationResult,
)
from saealib.core.compiler.compiler import (
    ExecutablePlan,
    _diagnostic_sort_key,
    _verification_registrations,
)
from saealib.core.compiler.diagnostics import ContractPath, Diagnostic, Severity
from saealib.core.compiler.graph import ControlEdge
from saealib.core.contracts import ComponentContract
from saealib.core.state import OPTIMIZATION_STATE_INITIAL_KEYS
from saealib.exceptions import ValidationError
from saealib.execution.runtime import default_runtime_registry
from saealib.optimizer import Optimizer
from saealib.problem import Problem
from saealib.strategies.direct import DirectStrategy, SteadyStateStrategy
from saealib.strategies.gb import GenerationBasedStrategy
from saealib.strategies.ib import IndividualBasedStrategy
from saealib.strategies.ps import PreSelectionStrategy
from saealib.surrogate.rbf import RBFSurrogate
from saealib.surrogate.rbf_kernels import GaussianKernel


class _Component:
    def contract(self) -> ComponentContract:
        return ComponentContract()


class _IdentityFeatureEncoder:
    def encode(self, genomes):
        return genomes.array


def _problem() -> Problem:
    return Problem(
        func=lambda x: float(np.sum(np.asarray(x) ** 2)),
        dim=2,
        n_obj=1,
        direction=np.array([-1.0]),
        lb=[-1.0, -1.0],
        ub=[1.0, 1.0],
    )


def _strategy_inputs(key: str):
    problem = _problem()
    problem.space.services.register("FeatureEncoder", _IdentityFeatureEncoder())
    optimizer = Optimizer(problem)
    if key == "direct":
        optimizer.set_strategy(DirectStrategy(n_offspring=4))
    elif key == "steady_state":
        optimizer.set_strategy(SteadyStateStrategy())
    elif key == "generation_based":
        optimizer.set_strategy(GenerationBasedStrategy(gen_ctrl=2))
        optimizer.set_surrogate(RBFSurrogate(GaussianKernel()))
    elif key == "individual_based":
        optimizer.set_strategy(IndividualBasedStrategy(evaluation_ratio=0.5))
        optimizer.set_surrogate(RBFSurrogate(GaussianKernel()))
    else:
        optimizer.set_strategy(PreSelectionStrategy(n_candidates=8, n_select=2))
        optimizer.set_surrogate(RBFSurrogate(GaussianKernel()))
    optimizer.resolve_defaults()
    graph = optimizer.strategy.build_graph(optimizer)
    context = CompileContext(
        space=problem.space,
        problem=problem,
        offered_runtime_capabilities=default_runtime_registry.offered_capabilities(
            optimizer
        ),
        initial_state_keys=OPTIMIZATION_STATE_INITIAL_KEYS,
    )
    return graph, context


def _simple_graph() -> ComponentGraph:
    return ComponentGraph(
        nodes=(ComponentNode(component_id="start", component=_Component()),),
        entry_points=(NodeRef(component_id="start"),),
    )


def _graph_with_wellformedness_error() -> ComponentGraph:
    return ComponentGraph(
        nodes=(ComponentNode(component_id="a", component=_Component()),),
        entry_points=(NodeRef(component_id="a"),),
        control_edges=(
            ControlEdge(
                source=NodeRef(component_id="a"),
                target=NodeRef(component_id="ghost"),
            ),
        ),
    )


def _graphs_equivalent(left: ComponentGraph, right: ComponentGraph) -> bool:
    """Compare graphs by shape and contracts, not by component identity.

    Per-preparation component wrappers (freshened or inserted by resolution
    rules) are new instances on every preparation run, so node identity
    cannot be compared across runs even for equal graphs.
    """
    return (
        left.data_edges == right.data_edges
        and left.control_edges == right.control_edges
        and left.state_bindings == right.state_bindings
        and left.entry_points == right.entry_points
        and len(left.nodes) == len(right.nodes)
        and all(
            a.component_id == b.component_id
            and a.role == b.role
            and a.contract == b.contract
            and dict(a.resolved_services) == dict(b.resolved_services)
            for a, b in zip(left.nodes, right.nodes)
        )
    )


def _composed_plan(
    compiler: Compiler, graph: ComponentGraph, context: CompileContext
) -> ExecutablePlan:
    prepared = compiler._prepare(graph, context)
    if prepared.wellformedness_failed:
        return ExecutablePlan(
            graph=prepared.graph,
            diagnostics=tuple(
                sorted(dict.fromkeys(prepared.diagnostics), key=_diagnostic_sort_key)
            ),
            required_runtime_capabilities=frozenset(
                capability
                for node in prepared.graph.nodes
                for capability in node.contract.execution.required_runtime_capabilities
            ),
            active_rule_namespaces=frozenset(),
            active_rule_names=(),
            contract_snapshots=tuple(
                (node.component_id, node.contract) for node in prepared.graph.nodes
            ),
        )
    diagnostics = compiler._verify(prepared)
    verification = _verification_registrations(
        compiler.registry.registrations(), prepared.graph, prepared.context
    )
    active = tuple(
        sorted(
            (*prepared.resolution, *verification),
            key=lambda item: (item.namespace, item.name, item.phase),
        )
    )
    return ExecutablePlan(
        graph=prepared.graph,
        diagnostics=tuple(sorted(dict.fromkeys(diagnostics), key=_diagnostic_sort_key)),
        required_runtime_capabilities=frozenset(
            capability
            for node in prepared.graph.nodes
            for capability in node.contract.execution.required_runtime_capabilities
        ),
        active_rule_namespaces=frozenset(item.namespace for item in active),
        active_rule_names=tuple(f"{item.namespace}:{item.name}" for item in active),
        inserted_adapters=tuple(
            sorted(
                (
                    insertion
                    for node in prepared.graph.nodes
                    if (insertion := getattr(node.component, "insertion", None))
                    is not None
                ),
                key=str,
            )
        ),
        contract_snapshots=tuple(
            (node.component_id, node.contract) for node in prepared.graph.nodes
        ),
    )


@pytest.mark.parametrize(
    "key",
    ["direct", "steady_state", "generation_based", "individual_based", "pre_selection"],
)
def test_prepare_composition_representative_strategies(key: str) -> None:
    compiler = Compiler()
    graph, context = _strategy_inputs(key)

    compiled = compiler.compile(graph, context)
    composed = _composed_plan(compiler, graph, context)

    assert compiled.diagnostics == composed.diagnostics
    assert (
        compiled.required_runtime_capabilities == composed.required_runtime_capabilities
    )
    assert compiled.active_rule_namespaces == composed.active_rule_namespaces
    assert compiled.active_rule_names == composed.active_rule_names
    assert compiled.inserted_adapters == composed.inserted_adapters
    assert compiled.contract_snapshots == composed.contract_snapshots
    assert compiled.describe() == composed.describe()
    assert _graphs_equivalent(compiled.graph, composed.graph)
    assert not [
        diagnostic
        for diagnostic in compiled.diagnostics
        if diagnostic.severity.value == "error"
    ]


def test_prepare_composition_wellformedness_error() -> None:
    compiler = Compiler()
    graph = _graph_with_wellformedness_error()
    context = CompileContext()

    compiled = compiler.compile(graph, context)
    prepared = compiler._prepare(graph, context)
    composed = _composed_plan(compiler, graph, context)

    assert prepared.wellformedness_failed
    assert compiled == composed
    assert compiled.active_rule_namespaces == frozenset()
    assert compiled.active_rule_names == ()


def test_skipping_verification_keeps_the_prepared_plan_unchanged() -> None:
    graph, context = _strategy_inputs("direct")
    graph = replace(
        graph,
        nodes=(
            *graph.nodes,
            ComponentNode(component_id="orphan", component=_Component()),
        ),
    )
    compiler = Compiler()

    verified = compiler.compile(graph, context)
    unverified = compiler.compile(graph, replace(context, _verification_enabled=False))

    assert any(
        item.code == "unreachable_node" and item.severity is Severity.ERROR
        for item in verified.diagnostics
    )
    assert unverified.diagnostics == ()
    assert _graphs_equivalent(verified.graph, unverified.graph)
    assert verified.inserted_adapters == unverified.inserted_adapters
    assert verified.contract_snapshots == unverified.contract_snapshots
    for verified_node, unverified_node in zip(
        verified.graph.nodes, unverified.graph.nodes
    ):
        assert verified_node.resolved_services.keys() == (
            unverified_node.resolved_services.keys()
        )
        assert all(
            verified_node.resolved_services[name]
            is unverified_node.resolved_services[name]
            for name in verified_node.resolved_services
        )
        verified_stage = getattr(verified_node.component, "stage", None)
        unverified_stage = getattr(unverified_node.component, "stage", None)
        if verified_stage is not None and unverified_stage is not None:
            for name in ("_algorithm", "_surrogate_manager", "_acquisition"):
                if hasattr(verified_stage, name):
                    assert getattr(verified_stage, name) is getattr(
                        unverified_stage, name
                    )
    assert tuple(verified.active_rule_names) != tuple(unverified.active_rule_names)
    assert not any("reachability" in name for name in unverified.active_rule_names)


def test_verification_setting_does_not_affect_context_repr_or_equality() -> None:
    enabled = CompileContext()
    disabled = replace(enabled, _verification_enabled=False)

    assert enabled == disabled
    assert "_verification_enabled" not in repr(enabled)
    assert "_verification_enabled" not in repr(disabled)


def _graph_fingerprint(graph: ComponentGraph) -> tuple[object, ...]:
    return tuple(
        (
            node.component_id,
            node.role,
            node.contract,
            tuple(node.contract.ports),
        )
        for node in graph.nodes
    ) + tuple(
        (
            edge.source.component_id,
            edge.source.role,
            edge.target.component_id,
            edge.target.role,
        )
        for edge in (*graph.data_edges, *graph.control_edges)
    )


@dataclass
class _FindingVerificationRule:
    name: str = "findings"
    namespace: str = "test"
    phase: Literal["verification"] = "verification"

    def apply(self, context: RuleContext) -> VerificationResult:
        return VerificationResult(
            diagnostics=(
                Diagnostic(
                    severity=Severity.WARNING,
                    code="unknown_data_spec",
                    message="finding from findings",
                    path=ContractPath(components=("graph",)),
                    resolutions=("Inspect the test rule.",),
                ),
            )
        )


def test_verify_is_idempotent_and_does_not_mutate_the_prepared_graph() -> None:
    compiler = Compiler(RuleRegistry([_FindingVerificationRule()]))
    graph, context = _strategy_inputs("pre_selection")
    context = replace(context, enabled_rule_namespaces=frozenset({"test"}))

    prepared = compiler._prepare(graph, context)
    fingerprint_before = _graph_fingerprint(prepared.graph)
    prepared_diagnostics = list(prepared.diagnostics)

    first = compiler._verify(prepared)
    second = compiler._verify(prepared)

    assert len(first) > len(prepared_diagnostics)
    assert list(first) == list(second)
    assert list(prepared.diagnostics) == prepared_diagnostics
    assert _graph_fingerprint(prepared.graph) == fingerprint_before


@dataclass
class _RewritingVerificationRule:
    name: str = "rewriter"
    namespace: str = "test"
    phase: Literal["verification"] = "verification"

    def apply(self, context: RuleContext) -> ResolutionResult:
        node = replace(context.graph.nodes[0], role="rewritten")
        return ResolutionResult(
            graph=replace(context.graph, nodes=(node,)),
            claims=frozenset({context.claim("node", "start")}),
        )


def test_verification_rule_returning_rewrites_is_rejected() -> None:
    compiler = Compiler(RuleRegistry([_RewritingVerificationRule()]))
    graph = _simple_graph()
    context = CompileContext(enabled_rule_namespaces=frozenset({"test"}))

    with pytest.raises(ValidationError):
        compiler.compile(graph, context)

    prepared = compiler._prepare(graph, context)
    with pytest.raises(ValidationError):
        compiler._verify(prepared)


def test_verify_of_a_malformed_graph_keeps_only_the_preparation_diagnostics() -> None:
    compiler = Compiler()
    prepared = compiler._prepare(_graph_with_wellformedness_error(), CompileContext())

    verified = compiler._verify(prepared)

    assert prepared.wellformedness_failed
    assert list(verified) == list(prepared.diagnostics)
    assert any(item.severity is Severity.ERROR for item in verified)
