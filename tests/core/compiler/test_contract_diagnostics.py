from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest

from saealib.algorithms.ga import GA
from saealib.algorithms.pso import PSO
from saealib.core.compiler import (
    ContractPath,
    DiagnosticBag,
    Severity,
    check_component_contract,
)
from saealib.core.compiler.contract_diagnostics import _finding
from saealib.core.contracts import (
    AssumptionSet,
    ComponentContract,
    DataSpec,
    PartSpec,
    PortContract,
    PortDirection,
    PortSpec,
    ServiceRequirement,
    StateContract,
    Var,
)
from saealib.core.state import StateKey
from saealib.operators.crossover import CrossoverSBX
from saealib.operators.mutation import MutationUniform
from saealib.operators.selection import SequentialSelection, TruncationSelection
from saealib.optimizer import Optimizer
from saealib.problem import Problem


def _problem() -> Problem:
    return Problem(
        lambda x: float(np.sum(x**2)),
        1,
        1,
        np.array([-1.0]),
        lb=[-1.0],
        ub=[1.0],
    )


class _Child:
    def contract(self) -> ComponentContract:
        return ComponentContract(
            assumptions=AssumptionSet({"evaluation.deterministic": False})
        )


class _Parent:
    def __init__(self) -> None:
        self.child = _Child()

    def contract(self) -> ComponentContract:
        return ComponentContract(
            parts=(PartSpec(name="child", contract=ComponentContract()),)
        )


class _ExplodingChildAttribute:
    @property
    def child(self) -> object:
        raise RuntimeError("attribute access failed")

    def contract(self) -> ComponentContract:
        return ComponentContract(
            parts=(PartSpec(name="child", contract=ComponentContract()),)
        )


class _ContractRaises:
    def contract(self) -> ComponentContract:
        raise RuntimeError("broken contract")


class _ContractWrongType:
    def contract(self) -> str:
        return "not a contract"


class _ContractNotCallable:
    contract = object()


def test_unknown_port_role_has_one_finding_at_role_path() -> None:
    contract = ComponentContract(
        ports={
            "missing_role": PortContract(
                inputs=(
                    PortSpec(
                        name="genomes",
                        direction=PortDirection.INPUT,
                        data=DataSpec(kind="GenomeBatch"),
                        cardinality="ONE",
                    ),
                )
            )
        }
    )

    findings = check_component_contract(
        contract, path=ContractPath(components=("algorithm",))
    )

    assert len(findings) == 1
    finding = next(iter(findings))
    assert finding.code == "unknown_role"
    assert finding.path == ContractPath(components=("algorithm",), role="missing_role")


def test_unknown_role_is_a_warning_that_cannot_be_verified() -> None:
    contract = ComponentContract(ports={"magic": PortContract()})

    findings = check_component_contract(contract)

    assert [finding.code for finding in findings] == ["unknown_role"]
    finding = next(iter(findings))
    assert finding.severity is Severity.WARNING
    assert "cannot be verified" in finding.message


def test_unknown_cardinality_warning_suggests_registered_near_misses() -> None:
    contract = ComponentContract(
        ports={
            "proposer": PortContract(
                inputs=(
                    PortSpec(
                        name="genomes",
                        direction=PortDirection.INPUT,
                        data=DataSpec(kind="GenomeBatch"),
                        cardinality="many",
                    ),
                )
            )
        }
    )

    findings = check_component_contract(contract)

    assert [finding.code for finding in findings] == ["unknown_cardinality"]
    finding = next(iter(findings))
    assert finding.severity is Severity.WARNING
    assert "MANY" in finding.message


def test_part_contract_mismatch_stays_error() -> None:
    findings = check_component_contract(
        _Parent().contract(),
        component=_Parent(),
        path=ContractPath(components=("root",)),
    )

    assert [finding.code for finding in findings] == ["part_contract_mismatch"]
    assert next(iter(findings)).severity is Severity.ERROR


def test_unregistered_diagnostic_code_falls_back_to_registered_finding() -> None:
    findings = DiagnosticBag()
    path = ContractPath(components=("algorithm",))

    _finding(findings, "missing_diagnostic_code", path, "original problem", "fix it")

    finding = next(iter(findings))
    assert finding.code == "unregistered_diagnostic_code"
    assert finding.path == path


def _port_contract(
    *, kind: str = "GenomeBatch", service: str | None = None
) -> ComponentContract:
    services = () if service is None else (ServiceRequirement(name=service),)
    return ComponentContract(
        ports={
            "proposer": PortContract(
                inputs=(
                    PortSpec(
                        name="genomes",
                        direction=PortDirection.INPUT,
                        data=DataSpec(kind=kind),
                        cardinality="ONE",
                        required_services=services,
                    ),
                )
            )
        }
    )


@pytest.mark.parametrize(
    ("contract", "code", "path"),
    (
        (
            _port_contract(kind="missing_kind"),
            "unknown_data_spec",
            ContractPath(components=("component",), role="proposer", port="genomes"),
        ),
        (
            _port_contract(service="MissingService"),
            "unknown_service",
            ContractPath(components=("component",), role="proposer", port="genomes"),
        ),
        (
            ComponentContract(
                state=StateContract(
                    reads=(
                        StateKey(
                            namespace="missing_namespace",
                            name="value",
                            schema_version=1,
                        ),
                    )
                )
            ),
            "unknown_state_namespace",
            ContractPath(components=("component",)),
        ),
    ),
    ids=("kind", "service", "state-namespace"),
)
def test_unknown_contract_references_have_exactly_one_finding(
    contract: ComponentContract, code: str, path: ContractPath
) -> None:
    findings = check_component_contract(contract)

    assert len(findings) == 1
    finding = next(iter(findings))
    assert finding.code == code
    assert finding.path == path


def test_binding_keys_are_checked_but_var_names_are_not() -> None:
    contract = ComponentContract(
        ports={
            "proposer": PortContract(
                inputs=(
                    PortSpec(
                        name="genomes",
                        direction=PortDirection.INPUT,
                        data=DataSpec(
                            kind="GenomeBatch",
                            bindings={"missing_variable": Var(name="local_only")},
                        ),
                        cardinality="ONE",
                    ),
                )
            )
        }
    )

    findings = check_component_contract(contract)

    assert [finding.code for finding in findings] == ["unknown_schema_variable"]


def test_part_contract_mismatch_is_reported_recursively() -> None:
    findings = check_component_contract(
        _Parent().contract(),
        component=_Parent(),
        path=ContractPath(components=("root",)),
    )

    assert [finding.code for finding in findings] == ["part_contract_mismatch"]
    assert next(iter(findings)).path == ContractPath(components=("root", "child"))


def test_part_attribute_exception_is_reported_without_escaping() -> None:
    component = _ExplodingChildAttribute()

    findings = check_component_contract(
        component.contract(),
        component=component,
        path=ContractPath(components=("root",)),
    )

    assert [finding.code for finding in findings] == ["part_contract_mismatch"]
    assert next(iter(findings)).path == ContractPath(components=("root", "child"))


@pytest.mark.parametrize(
    "component",
    (_ContractRaises(), _ContractWrongType(), _ContractNotCallable()),
    ids=("raises", "wrong-type", "non-callable"),
)
def test_optimizer_reports_unavailable_contracts(component: object) -> None:
    optimizer = Optimizer(_problem()).set_algorithm(cast(Any, component))

    findings = optimizer.contract_diagnostics()

    assert len(findings) == 1
    finding = next(iter(findings))
    assert finding.code == "contract_unavailable"
    assert finding.path == ContractPath(components=("algorithm",))


def test_optimizer_accessor_does_not_fill_defaults_or_mutate_configuration() -> None:
    optimizer = Optimizer(_problem())
    evaluator = optimizer.evaluator

    findings = optimizer.contract_diagnostics()

    assert len(findings) == 0
    assert optimizer.evaluator is evaluator
    assert not hasattr(optimizer, "algorithm")
    assert not hasattr(optimizer, "strategy")


def _ga() -> GA:
    return GA(
        crossover=CrossoverSBX(prob=0.9, eta=15.0),
        mutation=MutationUniform(prob_var=0.3),
        parent_selection=SequentialSelection(),
        survivor_selection=TruncationSelection(),
    )


@pytest.mark.parametrize(
    "optimizer_factory",
    (
        lambda: Optimizer(_problem()),
        lambda: Optimizer(_problem()).set_algorithm(_ga()),
        lambda: Optimizer(_problem()).set_algorithm(PSO()),
    ),
    ids=("bundled-default", "configured-ga", "configured-pso"),
)
def test_default_and_representative_optimizers_have_no_contract_diagnostics(
    optimizer_factory,
) -> None:
    optimizer = optimizer_factory()
    optimizer.resolve_defaults()

    assert len(optimizer.contract_diagnostics()) == 0
