from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from saealib.comparators import ParetoComparator, SingleObjectiveComparator
from saealib.core.compiler import ContractPath, Severity
from saealib.core.compiler.contract_diagnostics import (
    _optimizer_configuration_diagnostics,
)
from saealib.exceptions import ConfigurationError
from saealib.execution.bindings import BINDING_SLOTS, resolve_components
from saealib.execution.evaluator import (
    EvaluationHandle,
    EvaluationRequest,
    EvaluationResult,
    Evaluator,
    SerialEvaluator,
)
from saealib.execution.initializer import LHSInitializer
from saealib.optimizer import Optimizer
from saealib.policies.evaluation import EvaluateAll
from saealib.policies.feedback import MixedFeedback
from saealib.problem import Problem
from saealib.strategies.base import OptimizationStrategy
from saealib.termination import Termination, max_fe


def _problem(n_obj: int = 1, func=None) -> Problem:
    return Problem(
        func=func or (lambda _: np.zeros(n_obj)),
        dim=2,
        n_obj=n_obj,
        direction=np.array([-1.0] * n_obj),
        lb=[-1.0, -1.0],
        ub=[1.0, 1.0],
    )


class _Strategy:
    requires_surrogate = False


class _Directional:
    def __init__(self, direction: object) -> None:
        self.direction = direction


def _bindings() -> dict[str, object]:
    return {
        "comparator": SingleObjectiveComparator(direction=-1.0),
        "algorithm": object(),
        "strategy": _Strategy(),
        "initializer": object(),
        "termination": object(),
        "surrogate_manager": object(),
        "acquisition": _Directional(np.array([-1.0])),
        "evaluation_planner": None,
        "feedback_builder": None,
        "evaluator": SerialEvaluator(),
        "cbmanager": object(),
    }


def _diagnostics(
    bindings: dict[str, object] | None = None,
    *,
    provenance: dict[str, tuple[str, str]] | None = None,
    problem: Problem | None = None,
    evaluator: object | None = None,
):
    resolved = _bindings() if bindings is None else bindings
    return _optimizer_configuration_diagnostics(
        resolved,
        provenance or {},
        problem or _problem(),
        evaluator or resolved.get("evaluator", SerialEvaluator()),
    )


@pytest.mark.parametrize(
    "slot", ("algorithm", "strategy", "initializer", "termination")
)
def test_required_component_defect_has_stable_error_location_and_cause(
    slot: str,
) -> None:
    bindings = _bindings()
    bindings[slot] = None

    findings = [
        item
        for item in _diagnostics(bindings)
        if item.code == "missing_required_component"
    ]

    assert len(findings) == 1
    finding = findings[0]
    assert finding.severity is Severity.ERROR
    assert finding.path == ContractPath(components=("optimizer", slot))
    assert f"{slot} is not set after component resolution" in finding.message


def test_prepared_requirements_accept_present_components() -> None:
    findings = _diagnostics()
    defect_codes = {
        "missing_required_component",
        "missing_surrogate_manager",
        "missing_acquisition",
        "direction_length_mismatch",
        "inconsistent_evaluator_lifecycle",
    }
    assert not [item for item in findings if item.code in defect_codes]


def test_missing_surrogate_and_acquisition_report_strategy_provenance() -> None:
    bindings = _bindings()
    bindings["strategy"] = type("SurrogateStrategy", (), {"requires_surrogate": True})()
    bindings["surrogate_manager"] = None
    bindings["acquisition"] = None
    provenance = {"strategy": ("user preset", "Selected from the user preset.")}

    findings = _diagnostics(bindings, provenance=provenance)

    by_code = {item.code: item for item in findings}
    assert by_code["missing_surrogate_manager"].severity is Severity.ERROR
    assert by_code["missing_surrogate_manager"].path == ContractPath(
        components=("optimizer", "surrogate_manager")
    )
    assert by_code["missing_acquisition"].severity is Severity.ERROR
    assert by_code["missing_acquisition"].path == ContractPath(
        components=("optimizer", "acquisition")
    )
    for code in ("missing_surrogate_manager", "missing_acquisition"):
        assert "user preset" in by_code[code].message
        assert "Selected from the user preset" in by_code[code].message


def test_unreadable_strategy_requirement_has_one_warning_with_cause() -> None:
    class UnreadableStrategy:
        @property
        def requires_surrogate(self):
            raise RuntimeError("requirement lookup failed")

    bindings = _bindings()
    bindings["strategy"] = UnreadableStrategy()

    findings = [
        item
        for item in _diagnostics(bindings)
        if item.code == "cannot_verify_component_requirement"
    ]

    assert len(findings) == 1
    assert findings[0].severity is Severity.WARNING
    assert findings[0].path == ContractPath(components=("optimizer", "strategy"))
    assert "RuntimeError: requirement lookup failed" in findings[0].message


def test_planner_and_feedback_protocols_are_checked_without_subclass_requirements() -> (
    None
):
    class IndependentPlanner:
        def plan(self, *args, **kwargs):
            pass

    class IndependentFeedback:
        def build(self, *args, **kwargs):
            pass

    bindings = _bindings()
    bindings["evaluation_planner"] = IndependentPlanner()
    bindings["feedback_builder"] = IndependentFeedback()

    findings = _diagnostics(bindings)

    assert not [item for item in findings if item.severity is Severity.ERROR]
    uncertain = [
        item for item in findings if item.code == "cannot_verify_component_requirement"
    ]
    assert {item.path for item in uncertain} == {
        ContractPath(components=("optimizer", "evaluation_planner")),
        ContractPath(components=("optimizer", "feedback_builder")),
    }
    assert all(item.severity is Severity.WARNING for item in uncertain)


def test_independent_planner_and_feedback_warnings_do_not_block_execution() -> None:
    class IndependentPlanner:
        def __init__(self) -> None:
            self._implementation = EvaluateAll()

        def plan(self, *args, **kwargs):
            return self._implementation.plan(*args, **kwargs)

    class IndependentFeedback:
        def __init__(self) -> None:
            self._implementation = MixedFeedback()

        def build(self, *args, **kwargs):
            return self._implementation.build(*args, **kwargs)

    optimizer = Optimizer(_problem(), seed=23)
    setattr(optimizer, "evaluation_planner", IndependentPlanner())
    setattr(optimizer, "feedback_builder", IndependentFeedback())

    assert optimizer.validate() == []
    warnings = [
        diagnostic
        for diagnostic in optimizer.last_contract_diagnostics
        if diagnostic.code == "cannot_verify_component_requirement"
    ]
    assert {diagnostic.path.components[-1] for diagnostic in warnings} == {
        "evaluation_planner",
        "feedback_builder",
    }

    state = optimizer.run()

    assert state.fe > 0


@pytest.mark.parametrize(
    ("slot", "method"),
    (("evaluation_planner", "plan"), ("feedback_builder", "build")),
)
def test_invalid_planner_or_feedback_operation_is_an_error(slot: str, method: str):
    bindings = _bindings()
    bindings[slot] = object()

    findings = [
        item
        for item in _diagnostics(bindings)
        if item.code == "component_protocol_mismatch"
    ]

    assert len(findings) == 1
    assert findings[0].severity is Severity.ERROR
    assert findings[0].path == ContractPath(components=("optimizer", slot))
    assert f"callable {method}()" in findings[0].message


def test_direction_length_defects_identify_comparator_and_selected_acquisition() -> (
    None
):
    problem = _problem()
    bindings = _bindings()
    bindings["comparator"] = ParetoComparator(direction=np.array([1.0, 1.0]))
    bindings["acquisition"] = _Directional(np.array([1.0, 1.0]))
    provenance = {
        "comparator": ("setter", "Configured with set_comparator()."),
        "acquisition": ("setter", "Configured with set_acquisition()."),
    }

    findings = [
        item
        for item in _diagnostics(bindings, provenance=provenance, problem=problem)
        if item.code == "direction_length_mismatch"
    ]

    assert len(findings) == 2
    assert {item.severity for item in findings} == {Severity.ERROR}
    comparator = next(
        item for item in findings if item.path.components == ("optimizer", "comparator")
    )
    acquisition = next(
        item
        for item in findings
        if item.path.components == ("optimizer", "acquisition")
    )
    assert comparator.path == ContractPath(components=("optimizer", "comparator"))
    assert acquisition.path == ContractPath(
        components=("optimizer", "acquisition"), role="acquisition"
    )
    assert "direction length (2)" in comparator.message
    assert "direction length (2)" in acquisition.message
    assert "setter" in acquisition.message
    assert "Configured with set_acquisition()" in acquisition.message


def test_unmeasurable_direction_length_is_a_warning() -> None:
    class UnknownLength:
        def __len__(self):
            raise TypeError("length unavailable")

    bindings = _bindings()
    bindings["acquisition"] = _Directional(UnknownLength())

    findings = [
        item
        for item in _diagnostics(bindings)
        if item.code == "cannot_verify_direction_length"
    ]

    assert len(findings) == 1
    assert findings[0].severity is Severity.WARNING
    assert findings[0].path.components == ("optimizer", "acquisition")
    assert "cannot be determined" in findings[0].message


class _PartialEvaluator(Evaluator):
    def evaluate_batch(self, x, problem) -> EvaluationResult:
        raise NotImplementedError

    def submit(self, request: EvaluationRequest, problem: Problem) -> EvaluationHandle:
        return super().submit(request, problem)


class _CompleteLifecycleEvaluator(_PartialEvaluator):
    def collect(self, handle: EvaluationHandle, *, wait: bool = True):
        return super().collect(handle, wait=wait)

    def acknowledge(self, handle: EvaluationHandle, sequence: int) -> None:
        super().acknowledge(handle, sequence)


class _InheritedLifecycleEvaluator(_CompleteLifecycleEvaluator):
    pass


class _IndependentEvaluator:
    def evaluate_batch(self, x, problem):
        raise NotImplementedError

    def submit(self, request, problem):
        raise NotImplementedError

    def collect(self, handle, *, wait=True):
        raise NotImplementedError

    def acknowledge(self, handle, sequence):
        raise NotImplementedError


def test_evaluator_lifecycle_must_be_consistent_and_inherited_lifecycle_is_valid() -> (
    None
):
    partial = [
        item
        for item in _diagnostics(evaluator=_PartialEvaluator())
        if item.code == "inconsistent_evaluator_lifecycle"
    ]

    assert len(partial) == 1
    assert partial[0].severity is Severity.ERROR
    assert partial[0].path == ContractPath(components=("optimizer", "evaluator"))
    assert "submit(), collect(), and acknowledge()" in partial[0].message
    assert not [
        item
        for item in _diagnostics(evaluator=_InheritedLifecycleEvaluator())
        if item.code == "inconsistent_evaluator_lifecycle"
    ]

    unknown = [
        item
        for item in _diagnostics(evaluator=_IndependentEvaluator())
        if item.code == "cannot_verify_component_requirement"
    ]
    assert len(unknown) == 1
    assert unknown[0].severity is Severity.WARNING
    assert unknown[0].path == ContractPath(components=("optimizer", "evaluator"))


def test_set_evaluator_keeps_early_partial_lifecycle_validation() -> None:
    with pytest.raises(ConfigurationError, match="overridden together"):
        Optimizer(_problem()).set_evaluator(_PartialEvaluator())


def test_incomplete_resolution_reports_required_components_and_skips_initializer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    optimizer = Optimizer(_problem())
    resolution = resolve_components(optimizer)
    components = dict(resolution.components)
    for slot in ("algorithm", "strategy", "initializer", "termination"):
        components[slot] = None
    incomplete = replace(
        resolution,
        components=components,
        bindings={slot: components.get(slot) for slot in BINDING_SLOTS},
        origins={},
    )
    monkeypatch.setattr("saealib.optimizer.resolve_components", lambda _: incomplete)

    errors = optimizer.validate()
    findings = optimizer.last_contract_diagnostics

    assert isinstance(errors, list)
    assert [
        item.path.components[-1]
        for item in findings
        if item.code == "missing_required_component"
    ] == ["algorithm", "strategy", "initializer", "termination"]
    assert [str(item) for item in findings if item.severity is Severity.ERROR] == errors
    assert len({(item.code, item.path) for item in findings}) == len(findings)

    resume_errors = optimizer.validate(require_initializer=False)
    assert not any("optimizer.initializer" in error for error in resume_errors)
    assert not any(
        item.code == "missing_required_component"
        and item.path.components == ("optimizer", "initializer")
        for item in optimizer.last_contract_diagnostics
    )


def test_strategy_graph_failure_is_a_diagnostic_with_selection_provenance() -> None:
    class BrokenStrategy(OptimizationStrategy):
        def build_graph(self, provider):
            raise RuntimeError("planner binding is unavailable")

    optimizer = Optimizer(_problem()).set_strategy(BrokenStrategy())

    issues = optimizer.validate()
    findings = [
        item
        for item in optimizer.last_contract_diagnostics
        if item.code == "strategy_graph_unavailable"
    ]

    assert len(findings) == 1
    assert findings[0].severity is Severity.ERROR
    assert findings[0].path == ContractPath(components=("optimizer", "strategy"))
    assert "planner binding is unavailable" in findings[0].message
    assert "setter" in findings[0].message
    assert "Configured with set_strategy()" in findings[0].message
    assert any("strategy_graph_unavailable" in issue for issue in issues)


def test_configuration_error_precedes_initial_objective_evaluation() -> None:
    calls: list[object] = []

    def objective(x):
        calls.append(x)
        return float(np.sum(np.asarray(x) ** 2))

    optimizer = Optimizer(_problem(func=objective)).set_comparator(
        ParetoComparator(direction=np.array([1.0, 1.0]))
    )

    with pytest.raises(ConfigurationError, match="direction_length_mismatch"):
        optimizer.run()

    assert calls == []


@pytest.mark.parametrize("entrypoint", ("run_from", "iterate_from"))
def test_resume_does_not_call_initializer(
    monkeypatch: pytest.MonkeyPatch, entrypoint: str
) -> None:
    optimizer = Optimizer(_problem(), seed=19)
    optimizer.resolve_defaults()
    initializer = optimizer.initializer
    assert initializer is not None
    state = initializer.initialize(optimizer, optimizer.problem)
    optimizer.set_termination(Termination(max_fe(state.fe)))
    optimizer.initializer = None

    def fail_if_called(*args, **kwargs):
        raise AssertionError("resume must not initialize the run state")

    monkeypatch.setattr(LHSInitializer, "initialize", fail_if_called)

    if entrypoint == "run_from":
        resumed = optimizer.run_from(state)
    else:
        resumed_states = optimizer.iterate_from(state)
        resumed = next(resumed_states)
        resumed_states.close()

    assert resumed.fe == state.fe
    assert not any(
        item.code == "missing_required_component"
        and item.path.components == ("optimizer", "initializer")
        for item in optimizer.last_contract_diagnostics
    )
