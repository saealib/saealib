"""Tests for validation-time contract diagnostics."""

from __future__ import annotations

import numpy as np
import pytest
from pymoo.algorithms.soo.nonconvex.ga import GA as PymooGA  # noqa: N811

from saealib.algorithms.pymoo_algorithm import PymooAlgorithm
from saealib.core.compiler import (
    Compiler,
    ContractPath,
    Diagnostic,
    DiagnosticBag,
    Severity,
)
from saealib.core.compiler.diagnostics import DIAGNOSTIC_CODES
from saealib.execution.evaluator import SerialEvaluator
from saealib.execution.scheduler import AsyncEvaluationScheduler
from saealib.islands import IslandModel
from saealib.optimizer import Optimizer
from saealib.problem import Problem
from saealib.strategies.direct import DirectStrategy
from saealib.surrogate.rbf import RBFSurrogate
from saealib.surrogate.rbf_kernels import GaussianKernel
from saealib.termination import Termination, max_fe


def _problem() -> Problem:
    return Problem(
        func=lambda x: np.array([float(np.sum(np.asarray(x) ** 2))]),
        dim=2,
        n_obj=1,
        direction=np.array([-1.0]),
        lb=[-1.0, -1.0],
        ub=[1.0, 1.0],
    )


def _default_optimizer() -> Optimizer:
    optimizer = Optimizer(_problem())
    optimizer.resolve_defaults()
    return optimizer


def _diagnostic(severity: Severity, message: str) -> Diagnostic:
    return Diagnostic(
        severity=severity,
        code="contract_unavailable",
        message=message,
        path=ContractPath(components=("algorithm",)),
    )


def test_error_contract_diagnostic_fails_validate(monkeypatch) -> None:
    optimizer = _default_optimizer()
    finding = _diagnostic(Severity.ERROR, "fatal contract finding")
    monkeypatch.setattr(
        optimizer,
        "contract_diagnostics",
        lambda: DiagnosticBag((finding,)),
    )

    issues = optimizer.validate()

    assert str(finding) in issues
    assert optimizer.last_contract_diagnostics == (finding,)


def test_warning_and_info_are_visible_without_failing_validate(monkeypatch) -> None:
    optimizer = _default_optimizer()
    warning = _diagnostic(Severity.WARNING, "advisory contract finding")
    info = _diagnostic(Severity.INFO, "informational contract finding")
    monkeypatch.setattr(
        optimizer,
        "contract_diagnostics",
        lambda: DiagnosticBag((warning, info)),
    )

    assert optimizer.validate() == []
    assert optimizer.last_contract_diagnostics == (warning, info)


def test_pymoo_complete_batch_with_async_runtime_is_diagnosed() -> None:
    optimizer = Optimizer(_problem()).set_algorithm(
        PymooAlgorithm(PymooGA(pop_size=4), allow_partial_tell=False)
    )
    optimizer.set_async_evaluation_scheduler(
        AsyncEvaluationScheduler(SerialEvaluator())
    )
    optimizer.resolve_defaults()

    diagnostics = tuple(optimizer.contract_diagnostics())

    assert [diagnostic.code for diagnostic in diagnostics] == [
        "pymoo_partial_feedback_unsupported"
    ]
    assert diagnostics[0].severity is Severity.ERROR
    assert diagnostics[0].path == ContractPath(components=("algorithm",))
    assert diagnostics[0].related == (
        ContractPath(components=("async_evaluation_scheduler",)),
    )
    assert any(
        "pymoo_partial_feedback_unsupported" in issue for issue in optimizer.validate()
    )


def test_pymoo_partial_tell_opt_in_matches_async_runtime() -> None:
    optimizer = Optimizer(_problem()).set_algorithm(
        PymooAlgorithm(PymooGA(pop_size=4), allow_partial_tell=True)
    )
    optimizer.set_async_evaluation_scheduler(
        AsyncEvaluationScheduler(SerialEvaluator())
    )
    optimizer.resolve_defaults()

    assert tuple(optimizer.contract_diagnostics()) == ()


@pytest.mark.parametrize("kind", ("default", "surrogate", "async"))
def test_supported_component_configurations_have_no_diagnostics(kind: str) -> None:
    optimizer = Optimizer(_problem())
    if kind == "surrogate":
        optimizer.set_surrogate(RBFSurrogate(kernel=GaussianKernel()))
    elif kind == "async":
        optimizer.set_async_evaluation_scheduler(
            AsyncEvaluationScheduler(SerialEvaluator())
        )
    optimizer.resolve_defaults()

    assert tuple(optimizer.contract_diagnostics()) == ()


def test_island_component_configurations_have_no_diagnostics() -> None:
    model = IslandModel(
        (
            Optimizer(_problem()).set_strategy(DirectStrategy(n_offspring=4)),
            Optimizer(_problem()).set_strategy(DirectStrategy(n_offspring=4)),
        ),
        migration_interval=0,
    )

    for optimizer in model.optimizers:
        optimizer.resolve_defaults()
        assert tuple(optimizer.contract_diagnostics()) == ()


def test_pymoo_diagnostic_code_is_persistently_registered() -> None:
    assert DIAGNOSTIC_CODES.get("pymoo_partial_feedback_unsupported") is not None


def test_contract_diagnostics_are_not_called_per_generation(monkeypatch) -> None:
    optimizer = _default_optimizer()
    optimizer.set_termination(Termination(max_fe(1)))
    calls = 0
    real_contract_diagnostics = optimizer.contract_diagnostics

    def count_calls():
        nonlocal calls
        calls += 1
        return real_contract_diagnostics()

    monkeypatch.setattr(optimizer, "contract_diagnostics", count_calls)

    optimizer.run()

    assert calls == 1


def test_optimizer_compiles_once_per_run_and_retains_plan(monkeypatch) -> None:
    optimizer = _default_optimizer()
    optimizer.set_termination(Termination(max_fe(1)))
    calls = 0
    original_compile = Compiler.compile

    def count_compile(self, graph, context=None):
        nonlocal calls
        calls += 1
        return original_compile(self, graph, context)

    monkeypatch.setattr(Compiler, "compile", count_compile)

    optimizer.run()

    assert calls == 1
    assert optimizer.executable_plan is not None
    assert optimizer.describe() == optimizer.executable_plan.describe()
