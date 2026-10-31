"""Tests for persistence and runtime compatibility rules."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pytest

from saealib.algorithms.pymoo_algorithm import PymooAlgorithm
from saealib.context import OptimizationState
from saealib.core.compiler import CompileContext, Compiler, RuleContext, Severity
from saealib.core.compiler.diagnostics import Diagnostic, DiagnosticBag
from saealib.core.compiler.graph import ComponentGraph, ComponentNode, NodeRef
from saealib.core.compiler.persistence_runtime_rules import (
    PersistenceRule,
    RuntimeCompatibilityRule,
)
from saealib.core.contracts import (
    ComponentContract,
    ExecutionContract,
    Fixed,
    ParameterSpec,
    RepresentationSpec,
    StateContract,
)
from saealib.core.state import POPULATIONS_MAIN
from saealib.exceptions import CheckpointError
from saealib.execution.evaluator import SerialEvaluator
from saealib.execution.runtime import default_runtime_registry
from saealib.execution.scheduler import AsyncEvaluationScheduler
from saealib.optimizer import Optimizer
from saealib.population import Archive, ParetoArchive, Population, PopulationAttribute
from saealib.population.genome import DenseVectorBatch
from saealib.problem import Problem
from saealib.space import ObjectSpace, VectorSpace


@dataclass
class _Component:
    _contract: ComponentContract

    def contract(self) -> ComponentContract:
        return self._contract


class _SchedulerHolder:
    """Hold a scheduler without making it a runtime capability provider."""

    def __init__(self, contract: ComponentContract) -> None:
        self.scheduler = AsyncEvaluationScheduler(SerialEvaluator())
        self._contract = contract

    def contract(self) -> ComponentContract:
        return self._contract


def _graph(contract: ComponentContract, component_id: str = "node") -> ComponentGraph:
    return ComponentGraph(
        nodes=(
            ComponentNode(
                component_id=component_id,
                component=_Component(contract),
            ),
        ),
        entry_points=(NodeRef(component_id=component_id),),
    )


def _diagnostics(
    rule: PersistenceRule | RuntimeCompatibilityRule,
    graph: ComponentGraph,
    compile_context: CompileContext,
) -> tuple[Diagnostic, ...]:
    result = rule.apply(
        RuleContext(
            graph=graph,
            compile_context=compile_context,
            diagnostics=DiagnosticBag(),
        )
    )
    return result.diagnostics


def _object_space() -> ObjectSpace:
    return ObjectSpace(
        RepresentationSpec(
            kind="vector",
            parameters=(ParameterSpec(name="dim", value=Fixed(value=1)),),
        )
    )


def test_portable_population_export_without_codec_is_rejected_at_compile_time() -> None:
    graph = _graph(
        ComponentContract(state=StateContract(exports=(POPULATIONS_MAIN,))),
        component_id="population_owner",
    )

    findings = _diagnostics(
        PersistenceRule(),
        graph,
        CompileContext(space=_object_space(), portability_required=True),
    )

    assert [finding.code for finding in findings] == ["missing_genome_codec"]
    finding = findings[0]
    assert str(finding.path) == "population_owner"
    assert "GenomeCodec" in finding.message
    assert "population state main" in finding.message
    assert finding.resolutions


def test_the_same_configuration_really_fails_during_portable_save(tmp_path) -> None:
    problem = Problem(
        lambda x: np.array([x[0]]),
        1,
        1,
        np.array([-1.0]),
        lb=[0.0],
        ub=[1.0],
    )
    space = problem.space
    dense_view = cast(Any, space.services.get("DenseNumericView"))
    attrs = [
        PopulationAttribute(name="x", dtype=np.float64, shape=(1,), default=np.nan)
    ]
    population = Population(
        attrs,
        init_capacity=1,
        genomes=DenseVectorBatch(np.array([[0.0]])),
        dense_numeric_view=dense_view,
    )
    archive = Archive(
        attrs,
        init_capacity=1,
        space=space,
        dense_numeric_view=dense_view,
    )
    pareto_archive = ParetoArchive(
        attrs,
        init_capacity=1,
        dense_numeric_view=dense_view,
    )
    space.services._services.pop("GenomeCodec")
    state = OptimizationState(
        problem=problem,
        population=population,
        archive=archive,
        pareto_archive=pareto_archive,
    )

    with pytest.raises(
        CheckpointError,
        match=r"GenomeCodec is required to save population populations/main",
    ):
        state.save(tmp_path / "checkpoint.npz")


def test_portable_population_export_with_codec_has_no_persistence_diagnostic() -> None:
    graph = _graph(ComponentContract(state=StateContract(exports=(POPULATIONS_MAIN,))))

    findings = _diagnostics(
        PersistenceRule(),
        graph,
        CompileContext(
            space=VectorSpace(dim=1, lb=[0.0], ub=[1.0]),
            portability_required=True,
        ),
    )

    assert findings == ()


def test_portable_population_export_without_space_cannot_be_verified() -> None:
    graph = _graph(
        ComponentContract(state=StateContract(exports=(POPULATIONS_MAIN,))),
        component_id="population_owner",
    )

    findings = _diagnostics(
        PersistenceRule(),
        graph,
        CompileContext(portability_required=True),
    )

    assert [finding.code for finding in findings] == ["missing_genome_codec"]
    assert findings[0].severity is Severity.WARNING
    assert "cannot be verified" in findings[0].message


def test_runtime_capability_requires_set_containment() -> None:
    graph = _graph(
        ComponentContract(
            execution=ExecutionContract(
                required_runtime_capabilities=("partial_feedback",)
            )
        )
    )

    findings = _diagnostics(RuntimeCompatibilityRule(), graph, CompileContext())

    assert [finding.code for finding in findings] == ["missing_runtime_capability"]
    assert findings[0].severity is Severity.WARNING
    assert "partial_feedback" in findings[0].message
    assert "cannot be verified" in findings[0].message


def test_explicit_empty_runtime_offer_makes_missing_capability_an_error() -> None:
    graph = _graph(
        ComponentContract(
            execution=ExecutionContract(
                required_runtime_capabilities=("partial_feedback",)
            )
        )
    )

    findings = _diagnostics(
        RuntimeCompatibilityRule(),
        graph,
        CompileContext(offered_runtime_capabilities=frozenset()),
    )

    assert [finding.code for finding in findings] == ["missing_runtime_capability"]
    assert findings[0].severity is Severity.ERROR
    assert "offers [none]" in findings[0].message


def test_scheduler_contract_does_not_declare_partial_feedback_offer() -> None:
    contract = ComponentContract(
        execution=ExecutionContract(required_runtime_capabilities=("partial_feedback",))
    )
    graph = ComponentGraph(
        nodes=(
            ComponentNode(
                component_id="runtime_user",
                component=_SchedulerHolder(contract),
            ),
        ),
        entry_points=(NodeRef(component_id="runtime_user"),),
    )

    findings = _diagnostics(RuntimeCompatibilityRule(), graph, CompileContext())

    assert [finding.code for finding in findings] == ["missing_runtime_capability"]
    assert all(finding.severity is Severity.WARNING for finding in findings)


def test_explicit_runtime_offer_satisfies_requirement() -> None:
    graph = _graph(
        ComponentContract(
            execution=ExecutionContract(
                required_runtime_capabilities=("partial_feedback",)
            )
        )
    )

    findings = _diagnostics(
        RuntimeCompatibilityRule(),
        graph,
        CompileContext(offered_runtime_capabilities=frozenset({"partial_feedback"})),
    )

    assert findings == ()


def test_k7_rule_order_is_stable_when_shuffled() -> None:
    graph = _graph(
        ComponentContract(
            state=StateContract(exports=(POPULATIONS_MAIN,)),
            execution=ExecutionContract(
                required_runtime_capabilities=("partial_feedback",)
            ),
        )
    )
    compile_context = CompileContext(
        space=_object_space(),
        portability_required=True,
    )
    rules = (PersistenceRule(), RuntimeCompatibilityRule())
    expected: tuple[tuple[str, str, str], ...] | None = None
    for seed in range(8):
        shuffled = list(rules)
        random.Random(seed).shuffle(shuffled)
        findings = tuple(
            finding
            for rule in shuffled
            for finding in _diagnostics(rule, graph, compile_context)
        )
        signature = tuple(
            (finding.code, str(finding.path), finding.message)
            for finding in sorted(
                findings, key=lambda item: (item.code, str(item.path))
            )
        )
        if expected is None:
            expected = signature
        else:
            assert signature == expected


def _problem() -> Problem:
    return Problem(
        lambda x: np.array([float(np.sum(np.asarray(x) ** 2))]),
        2,
        1,
        np.array([-1.0]),
        lb=[-1.0, -1.0],
        ub=[1.0, 1.0],
    )


def _resolved_optimizer(*, kind: str) -> Optimizer:
    optimizer = Optimizer(_problem())
    if kind == "surrogate":
        from saealib.surrogate.rbf import RBFSurrogate
        from saealib.surrogate.rbf_kernels import GaussianKernel

        optimizer.set_surrogate(RBFSurrogate(kernel=GaussianKernel()))
    elif kind == "async":
        optimizer.set_async_evaluation_scheduler(
            AsyncEvaluationScheduler(SerialEvaluator())
        )
    elif kind == "pymoo":
        from pymoo.algorithms.soo.nonconvex.ga import GA as PymooGA  # noqa: N811

        optimizer.set_algorithm(
            PymooAlgorithm(PymooGA(pop_size=4), allow_partial_tell=True)
        )
        optimizer.set_async_evaluation_scheduler(
            AsyncEvaluationScheduler(SerialEvaluator())
        )
    optimizer._resolve_defaults()
    return optimizer


@pytest.mark.parametrize("kind", ("default", "surrogate", "async", "island", "pymoo"))
def test_actual_strategy_graphs_have_no_k7_errors(kind: str) -> None:
    if kind == "island":
        optimizers = (
            _resolved_optimizer(kind="default"),
            _resolved_optimizer(kind="default"),
        )
    else:
        optimizers = (_resolved_optimizer(kind=kind),)

    for optimizer in optimizers:
        graph = cast(Any, optimizer.strategy).build_graph(optimizer)
        plan = Compiler().compile(
            graph,
            CompileContext(
                space=optimizer.problem.space,
                problem=optimizer.problem,
                offered_runtime_capabilities=default_runtime_registry.offered_capabilities(
                    optimizer
                ),
            ),
        )
        assert graph.nodes
        own_errors = [
            diagnostic
            for diagnostic in plan.diagnostics
            if diagnostic.code in {"missing_genome_codec", "missing_runtime_capability"}
            and diagnostic.severity.value == "error"
        ]
        assert own_errors == []
