"""Constraint locks for built-in contract overrides.

Every built-in ``contract()`` override must add information its family base
does not express.  The representation-fixing operators are the load-bearing
example: their ``Fixed`` representation bindings are what lets the compiler
reject incompatible pairings, so the tests lock them at the contract level
and through the compiler's schema-conflict path.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from saealib import (
    CrossoverCategorical,
    CrossoverIntegerSBX,
    CrossoverSBX,
    MutationCategorical,
    MutationGaussian,
    MutationIntegerUniform,
    MutationPolynomial,
    MutationUniform,
)
from saealib.core.compiler import (
    CompileContext,
    Compiler,
    ContractPath,
    DiagnosticBag,
    RuleContext,
    Severity,
)
from saealib.core.compiler.graph import ComponentGraph, ComponentNode, DataEdge, NodeRef
from saealib.core.compiler.schema_rules import SchemaBindingRule
from saealib.core.contracts import (
    ComponentContract,
    DataSpec,
    Fixed,
    PortContract,
    PortDirection,
    PortSpec,
    Var,
)
from saealib.operators.crossover import Crossover
from saealib.operators.mutation import Mutation
from saealib.operators.representation import OrderCrossover, SwapMutation

_REAL_MUTATIONS = (
    MutationUniform(),
    MutationPolynomial(eta=20.0),
    MutationGaussian(sigma=1.0),
)


@pytest.mark.parametrize("operator", _REAL_MUTATIONS, ids=lambda op: type(op).__name__)
def test_real_mutations_fix_representation_real_on_input_and_output(operator) -> None:
    role = operator.contract().ports["mutation"]
    assert role.inputs[0].data.bindings["representation"] == Fixed(value="real")
    assert role.outputs[0].data.bindings["representation"] == Fixed(value="real")


def test_discrete_representation_operators_fix_their_representation() -> None:
    integer = MutationIntegerUniform().contract().ports["mutation"]
    assert integer.inputs[0].data.bindings["representation"] == Fixed(value="integer")
    assert integer.outputs[0].data.bindings["representation"] == Fixed(value="integer")
    categorical = MutationCategorical().contract().ports["mutation"]
    assert categorical.inputs[0].data.bindings["representation"] == Fixed(
        value="categorical"
    )
    assert categorical.outputs[0].data.bindings["representation"] == Fixed(
        value="categorical"
    )
    permutation = SwapMutation().contract().ports["mutation"]
    for port in (*permutation.inputs, *permutation.outputs):
        assert port.data.bindings["representation"] == Fixed(value="permutation")
    order = OrderCrossover().contract().ports["crossover"]
    for port in (*order.inputs, *order.outputs):
        assert port.data.bindings["representation"] == Fixed(value="permutation")


def test_base_contracts_leave_the_representation_free() -> None:
    mutation = Mutation.contract(MutationUniform())
    binding = mutation.ports["mutation"].inputs[0].data.bindings["representation"]
    assert isinstance(binding, Var)
    crossover = Crossover.contract(CrossoverSBX(0.9, 15.0))
    binding = crossover.ports["crossover"].inputs[0].data.bindings["representation"]
    assert isinstance(binding, Var)


@dataclass(frozen=True)
class _PortEndpoint:
    data: DataSpec
    direction: PortDirection

    def contract(self) -> ComponentContract:
        port = PortSpec(
            name="value",
            direction=self.direction,
            data=self.data,
            cardinality="many",
        )
        role = (
            PortContract(outputs=(port,))
            if self.direction is PortDirection.OUTPUT
            else PortContract(inputs=(port,))
        )
        return ComponentContract(ports={"role": role})


def _edge_graph(source: DataSpec, target: DataSpec) -> ComponentGraph:
    return ComponentGraph(
        nodes=(
            ComponentNode(
                component_id="producer",
                component=_PortEndpoint(source, PortDirection.OUTPUT),
            ),
            ComponentNode(
                component_id="consumer",
                component=_PortEndpoint(target, PortDirection.INPUT),
            ),
        ),
        data_edges=(
            DataEdge(
                source=NodeRef(component_id="producer", role="role"),
                target=NodeRef(component_id="consumer", role="role"),
                source_port="value",
                target_port="value",
            ),
        ),
        entry_points=(NodeRef(component_id="producer", role="role"),),
    )


def test_real_mutation_port_rejects_permutation_representation_pairing() -> None:
    real = MutationUniform().contract().ports["mutation"]
    permutation = SwapMutation().contract().ports["mutation"]
    graph = _edge_graph(permutation.outputs[0].data, real.inputs[0].data)
    result = SchemaBindingRule().apply(
        RuleContext(
            graph=graph, compile_context=CompileContext(), diagnostics=DiagnosticBag()
        )
    )
    conflicts = [
        diagnostic
        for diagnostic in result.diagnostics
        if diagnostic.severity is Severity.ERROR
        and diagnostic.path.components[-1] == "producer"
    ]
    assert conflicts, "a real-fixed consumer must reject a permutation producer"
    assert all("representation" in diagnostic.message for diagnostic in conflicts)


def test_compiler_reports_the_representation_conflict_during_compilation() -> None:
    real = MutationUniform().contract().ports["mutation"]
    permutation = SwapMutation().contract().ports["mutation"]
    plan = Compiler().compile(
        _edge_graph(permutation.outputs[0].data, real.inputs[0].data)
    )
    findings = [
        diagnostic
        for diagnostic in plan.diagnostics
        if diagnostic.severity is Severity.ERROR
    ]
    assert findings, "the compiler must reject the incompatible pairing"
    assert all(
        "representation" in diagnostic.message
        or diagnostic.path.components[-1] == "producer"
        for diagnostic in findings
    )


_BUILTIN_CONTRACT_OVERRIDES: tuple[tuple[type, type], ...] = (
    (MutationUniform, Mutation),
    (MutationPolynomial, Mutation),
    (MutationGaussian, Mutation),
    (MutationIntegerUniform, Mutation),
    (MutationCategorical, Mutation),
    (CrossoverSBX, Crossover),
    (CrossoverIntegerSBX, Crossover),
    (CrossoverCategorical, Crossover),
    (SwapMutation, Mutation),
    (OrderCrossover, Crossover),
)


def _construct(operator_class: type):
    construction: dict[type, tuple[tuple, dict]] = {
        MutationUniform: ((), {}),
        MutationPolynomial: ((), {"eta": 20.0}),
        MutationGaussian: ((), {"sigma": 1.0}),
        MutationIntegerUniform: ((), {}),
        MutationCategorical: ((), {}),
        SwapMutation: ((), {}),
        OrderCrossover: ((), {}),
        CrossoverSBX: ((0.9, 15.0), {}),
        CrossoverIntegerSBX: ((0.9, 15.0), {}),
        CrossoverCategorical: ((0.9,), {}),
    }
    positional, keywords = construction[operator_class]
    return operator_class(*positional, **keywords)


@pytest.mark.parametrize(
    ("operator_class", "family"),
    _BUILTIN_CONTRACT_OVERRIDES,
    ids=lambda value: getattr(value, "__name__", str(value)),
)
def test_builtin_contract_overrides_add_information(operator_class, family) -> None:
    instance = _construct(operator_class)
    own = operator_class.contract(instance)
    base = family.contract(instance)
    assert own != base, (
        f"{operator_class.__name__}.contract() equals its {family.__name__} "
        "family contract; either remove the override or declare the "
        "information the base is missing."
    )
    diagnostic = check_contract_path(own)
    assert diagnostic is None


def check_contract_path(contract: ComponentContract):
    from saealib.core.compiler.contract_diagnostics import check_component_contract

    findings = check_component_contract(contract)
    return (findings.has_errors and findings) or None


def test_contract_paths_of_locked_overrides_are_clean() -> None:
    from saealib.core.compiler.contract_diagnostics import check_component_contract

    for operator_class, _family in _BUILTIN_CONTRACT_OVERRIDES:
        instance = _construct(operator_class)
        findings = check_component_contract(
            operator_class.contract(instance),  # ty: ignore[unresolved-attribute]
            path=ContractPath(components=(operator_class.__name__,)),
        )
        assert not findings.has_errors
