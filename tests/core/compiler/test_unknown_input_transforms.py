from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from saealib.core.compiler import (
    Compiler,
    ComponentGraph,
    ComponentNode,
    DataEdge,
    NodeRef,
    Severity,
)
from saealib.core.contracts import (
    CallInput,
    CallOutput,
    ComponentCall,
    ComponentContract,
    DataSpec,
    FlowEdge,
    PortContract,
    PortDirection,
    PortSpec,
    StageFlowSummary,
    StateContract,
    StateValue,
)
from saealib.core.contracts.flow import Transform
from saealib.core.state import SCORES
from saealib.optimizer import Optimizer
from saealib.pipeline import Pipeline, Stage
from saealib.problem import Problem
from saealib.strategies import TemplateStrategy


@dataclass
class _Endpoint:
    kind: str
    output: bool

    def contract(self) -> ComponentContract:
        port = PortSpec(
            name="value",
            direction=PortDirection.OUTPUT if self.output else PortDirection.INPUT,
            data=DataSpec(kind=self.kind),
            cardinality="MANY",
        )
        role = (
            PortContract(outputs=(port,))
            if self.output
            else PortContract(inputs=(port,))
        )
        return ComponentContract(ports={"main": role})


def _graph(source_kind: str, target_kind: str, names: tuple[str, ...]):
    return ComponentGraph(
        nodes=(
            ComponentNode(
                component_id="producer",
                component=_Endpoint(source_kind, output=True),
            ),
            ComponentNode(
                component_id="consumer",
                component=_Endpoint(target_kind, output=False),
            ),
        ),
        data_edges=(
            DataEdge(
                source=NodeRef(component_id="producer", role="main"),
                target=NodeRef(component_id="consumer", role="main"),
                source_port="value",
                target_port="value",
                transforms=tuple(Transform(name=name) for name in names),
            ),
        ),
        entry_points=(NodeRef(component_id="producer", role="main"),),
    )


@pytest.mark.parametrize(
    ("names", "cause"),
    (
        (("nonexistent_transform",), "unknown transform"),
        (("encode_features", "extract_genomes"), "requires CandidateSelection"),
    ),
)
def test_unknown_input_still_rejects_provable_transform_defects(
    names: tuple[str, ...], cause: str
) -> None:
    plan = Compiler().compile(_graph("UnverifiedFlowValue", "GenomeBatch", names))

    findings = [
        item for item in plan.diagnostics if item.code == "invalid_flow_transform"
    ]

    assert len(findings) == 1
    assert findings[0].severity is Severity.ERROR
    assert findings[0].path is not None
    assert findings[0].path.components == ("producer",)
    assert cause in findings[0].message


def test_unknown_input_checks_for_unregistered_transforms_later_in_chain() -> None:
    plan = Compiler().compile(
        _graph(
            "UnverifiedFlowValue",
            "ObservationBatch",
            ("extract_evaluation_observations", "nonexistent_transform"),
        )
    )

    findings = [
        item for item in plan.diagnostics if item.code == "invalid_flow_transform"
    ]

    assert len(findings) == 1
    assert findings[0].severity is Severity.ERROR
    assert "unknown transform 'nonexistent_transform'" in findings[0].message


def test_single_known_transform_keeps_unknown_input_unverified() -> None:
    plan = Compiler().compile(
        _graph(
            "UnverifiedFlowValue",
            "GenomeBatch",
            ("extract_population_genomes",),
        )
    )

    assert not [item for item in plan.diagnostics if item.severity is Severity.ERROR]
    assert any(
        item.code == "cannot_verify_data_kind" and item.severity is Severity.WARNING
        for item in plan.diagnostics
    )


@pytest.mark.parametrize(
    ("target_kind", "has_port_error"),
    (("GenomeBatch", True), ("FeatureBatch", False)),
)
def test_unknown_input_checks_known_transform_output_against_target(
    target_kind: str, has_port_error: bool
) -> None:
    plan = Compiler().compile(
        _graph("UnverifiedFlowValue", target_kind, ("encode_features",))
    )

    errors = [item for item in plan.diagnostics if item.severity is Severity.ERROR]
    port_errors = [item for item in errors if item.code == "incompatible_port"]
    assert len(port_errors) == int(has_port_error)
    if has_port_error:
        assert port_errors[0].path is not None
        assert port_errors[0].path.components == ("producer",)
        assert port_errors[0].related[0].components == ("consumer",)
        assert "data kinds are incompatible" in port_errors[0].message
    assert any(
        item.code == "cannot_verify_data_kind" and item.severity is Severity.WARNING
        for item in plan.diagnostics
    )


def test_unknown_input_keeps_warning_for_valid_transform_chain() -> None:
    plan = Compiler().compile(
        _graph(
            "UnverifiedFlowValue",
            "Population",
            ("extract_evaluation_observations", "apply_observations"),
        )
    )

    assert not [item for item in plan.diagnostics if item.severity is Severity.ERROR]
    assert any(
        item.code == "cannot_verify_data_kind" and item.severity is Severity.WARNING
        for item in plan.diagnostics
    )


def test_compatible_known_transform_chain_remains_valid() -> None:
    plan = Compiler().compile(
        _graph(
            "Population",
            "Population",
            ("extract_evaluation_observations", "apply_observations"),
        )
    )

    assert not [item for item in plan.diagnostics if item.severity is Severity.ERROR]


class _IndependentWriter:
    def produce(self):
        return None


class _WriterStage(Stage):
    def __init__(self) -> None:
        super().__init__(name="write")
        self.writer = _IndependentWriter()

    def contract(self) -> ComponentContract:
        return ComponentContract(state=StateContract(writes=(SCORES,)))

    def flow_summary(self) -> StageFlowSummary:
        writer = ComponentCall(name="writer", role="proposer", attribute="writer")
        return StageFlowSummary(
            calls=(writer,),
            edges=(
                FlowEdge(
                    source=CallOutput(call=writer, port="value"),
                    target=StateValue(key=SCORES),
                ),
            ),
        )

    def execute(self, state):
        return state


class _ReaderStage(Stage):
    def __init__(
        self, names: tuple[str, ...], target_kind: str = "GenomeBatch"
    ) -> None:
        super().__init__(name="read")
        self.consumer = _Endpoint(target_kind, output=False)
        self.names = names

    def contract(self) -> ComponentContract:
        return ComponentContract(state=StateContract(reads=(SCORES,)))

    def flow_summary(self) -> StageFlowSummary:
        consumer = ComponentCall(name="consumer", role="main", attribute="consumer")
        return StageFlowSummary(
            calls=(consumer,),
            edges=(
                FlowEdge(
                    source=StateValue(key=SCORES),
                    target=CallInput(call=consumer, port="value"),
                    transforms=tuple(Transform(name=name) for name in self.names),
                ),
            ),
        )

    def execute(self, state):
        return state


class _UnknownWriterStrategy(TemplateStrategy):
    requires_surrogate = False

    def __init__(
        self, names: tuple[str, ...], target_kind: str = "GenomeBatch"
    ) -> None:
        super().__init__(Pipeline([_WriterStage(), _ReaderStage(names, target_kind)]))


@pytest.mark.parametrize(
    "names",
    (
        ("nonexistent_transform",),
        ("encode_features", "extract_genomes"),
    ),
)
def test_optimizer_validation_reports_transform_defects_from_unknown_writer(
    names: tuple[str, ...],
) -> None:
    problem = Problem(
        lambda x: float(np.sum(np.asarray(x) ** 2)),
        dim=2,
        n_obj=1,
        direction=np.array([-1.0]),
        lb=[-1.0, -1.0],
        ub=[1.0, 1.0],
    )
    optimizer = Optimizer(problem).set_strategy(_UnknownWriterStrategy(names))

    errors = optimizer.validate()

    assert any("invalid_flow_transform" in error for error in errors)
    assert any(
        item.code == "invalid_flow_transform" and item.severity is Severity.ERROR
        for item in optimizer.last_contract_diagnostics
    )


@pytest.mark.parametrize(
    ("target_kind", "has_port_error"),
    (("GenomeBatch", True), ("FeatureBatch", False)),
)
def test_optimizer_validates_known_transform_output_from_unknown_writer(
    target_kind: str, has_port_error: bool
) -> None:
    evaluations: list[np.ndarray] = []

    def objective(x: np.ndarray) -> float:
        evaluations.append(x.copy())
        return float(np.sum(np.asarray(x) ** 2))

    problem = Problem(
        objective,
        dim=2,
        n_obj=1,
        direction=np.array([-1.0]),
        lb=[-1.0, -1.0],
        ub=[1.0, 1.0],
    )
    optimizer = Optimizer(problem).set_strategy(
        _UnknownWriterStrategy(("encode_features",), target_kind)
    )

    errors = optimizer.validate()
    diagnostics = optimizer.last_contract_diagnostics
    port_errors = [
        item
        for item in diagnostics
        if item.code == "incompatible_port" and item.severity is Severity.ERROR
    ]

    assert len(port_errors) == int(has_port_error)
    assert bool(errors) is has_port_error
    assert not evaluations
    assert any(
        item.code == "cannot_verify_data_kind" and item.severity is Severity.WARNING
        for item in diagnostics
    )
