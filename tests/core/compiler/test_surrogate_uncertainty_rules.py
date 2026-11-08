from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest

from saealib import (
    PSO,
    IndividualBasedStrategy,
    LHSInitializer,
    PairwiseComparisonSet,
    PairwiseSurrogateManager,
    SklearnRFCClassificationSurrogate,
    Termination,
    max_fe,
)
from saealib.acquisition import (
    CompositeAcquisition,
    ExpectedImprovement,
    MeanPrediction,
)
from saealib.core.compiler import ContractPath, Severity
from saealib.exceptions import ConfigurationError
from saealib.execution.runtime import RuntimeRegistration, default_runtime_registry
from saealib.optimizer import Optimizer
from saealib.pipeline import Pipeline
from saealib.problem import Problem
from saealib.stages import AcquisitionStage, SurrogatePredictStage
from saealib.strategies import (
    DirectStrategy,
    GenerationBasedStrategy,
    PreSelectionStrategy,
    SteadyStateStrategy,
    TemplateStrategy,
)
from saealib.surrogate import (
    CompositeSurrogateManager,
    GlobalSurrogateManager,
    RBFSurrogate,
    SklearnGPRSurrogate,
    Surrogate,
    SurrogatePrediction,
    product_combine,
)
from saealib.surrogate.manager import SurrogateManager
from saealib.surrogate.rbf_kernels import GaussianKernel


def _problem() -> Problem:
    return Problem(
        func=lambda x: float(np.sum(np.asarray(x) ** 2)),
        dim=2,
        n_obj=1,
        direction=np.array([-1.0]),
        lb=[-5.0, -5.0],
        ub=[5.0, 5.0],
    )


def _optimizer(manager, acquisition, strategy=None) -> Optimizer:
    optimizer = (
        Optimizer(_problem())
        .set_strategy(strategy or IndividualBasedStrategy())
        .set_surrogate_manager(manager)
        .set_acquisition(acquisition)
    )
    optimizer.resolve_defaults()
    return optimizer


def _plan_diagnostics(optimizer: Optimizer, code: str):
    optimizer._compile_plan()
    plan = optimizer.executable_plan
    assert plan is not None
    return tuple(item for item in plan.diagnostics if item.code == code)


def _rbf_manager() -> GlobalSurrogateManager:
    return GlobalSurrogateManager(RBFSurrogate(GaussianKernel()))


def _gpr_manager() -> GlobalSurrogateManager:
    return GlobalSurrogateManager(SklearnGPRSurrogate())


def _uncertainty_diagnostics(optimizer: Optimizer, code: str):
    return tuple(
        item for item in optimizer.last_contract_diagnostics if item.code == code
    )


def test_uncertainty_mismatch_is_attached_to_acquisition_prediction_input() -> None:
    mismatch = _plan_diagnostics(
        _optimizer(_rbf_manager(), ExpectedImprovement()),
        "surrogate_uncertainty_mismatch",
    )
    assert len(mismatch) == 1
    assert mismatch[0].severity is Severity.ERROR
    assert mismatch[0].path == ContractPath(
        components=("acquisition__acquisition",),
        role="acquisition",
        port="prediction",
    )
    assert mismatch[0].related == (
        ContractPath(
            components=("surrogate_predict__predictor",),
            role="predictor",
            port="prediction",
        ),
    )

    assert not _plan_diagnostics(
        _optimizer(_gpr_manager(), ExpectedImprovement()),
        "surrogate_uncertainty_mismatch",
    )
    assert not _plan_diagnostics(
        _optimizer(_rbf_manager(), MeanPrediction()),
        "surrogate_uncertainty_mismatch",
    )


def test_composite_channels_are_checked_by_matching_name_only() -> None:
    manager = CompositeSurrogateManager(
        {"objective": _rbf_manager(), "secondary": _gpr_manager()}
    )
    acquisition = CompositeAcquisition(
        {"objective": ExpectedImprovement(), "secondary": MeanPrediction()},
        product_combine,
    )
    mismatch = _plan_diagnostics(
        _optimizer(manager, acquisition), "surrogate_uncertainty_mismatch"
    )
    assert len(mismatch) == 1
    assert "channel 'objective'" in mismatch[0].message

    manager = CompositeSurrogateManager(
        {"objective": _rbf_manager(), "secondary": _gpr_manager()}
    )
    acquisition = CompositeAcquisition(
        {"objective": MeanPrediction(), "secondary": ExpectedImprovement()},
        product_combine,
    )
    assert not _plan_diagnostics(
        _optimizer(manager, acquisition), "surrogate_uncertainty_mismatch"
    )


def test_missing_composite_prediction_channel_is_an_error() -> None:
    manager = CompositeSurrogateManager({"objective": _gpr_manager()})
    acquisition = CompositeAcquisition(
        {"secondary": ExpectedImprovement()}, product_combine
    )
    diagnostics = _plan_diagnostics(
        _optimizer(manager, acquisition), "missing_acquisition_channel"
    )
    assert len(diagnostics) == 1
    assert diagnostics[0].severity is Severity.ERROR
    assert "'secondary'" in diagnostics[0].message
    assert diagnostics[0].path.port == "prediction"


def test_nested_composite_managers_are_checked_recursively() -> None:
    manager = CompositeSurrogateManager(
        {"objective": CompositeSurrogateManager({"objective": _rbf_manager()})}
    )
    acquisition = CompositeAcquisition(
        {"objective": ExpectedImprovement()}, product_combine
    )
    diagnostics = _plan_diagnostics(
        _optimizer(manager, acquisition), "surrogate_uncertainty_mismatch"
    )
    assert len(diagnostics) == 1
    assert diagnostics[0].severity is Severity.ERROR


class _UndeclaredSurrogate:
    def __init__(self) -> None:
        self._contract = RBFSurrogate(GaussianKernel()).contract()

    def contract(self):
        return self._contract

    def fit(self, train_x: np.ndarray, train_y: np.ndarray) -> None:
        pass

    def predict(self, test_x: np.ndarray) -> SurrogatePrediction:
        return SurrogatePrediction.objective(np.zeros((len(test_x), 1)))


class _InheritedFalseSurrogate(Surrogate):
    def fit(self, train_x: np.ndarray, train_y: np.ndarray) -> None:
        pass

    def predict(self, test_x: np.ndarray) -> SurrogatePrediction:
        return SurrogatePrediction.objective(np.zeros((len(test_x), 1)))


class _UndeclaredManager(SurrogateManager):
    def predict(
        self,
        candidates_x: np.ndarray,
        archive,
        ctx=None,
        *,
        refit: bool = True,
    ) -> SurrogatePrediction:
        return SurrogatePrediction.objective(np.zeros((len(candidates_x), 1)))


def test_undeclared_custom_surrogate_and_manager_emit_warnings() -> None:
    for manager in (
        GlobalSurrogateManager(cast(Surrogate, _UndeclaredSurrogate())),
        _UndeclaredManager(),
    ):
        diagnostics = _plan_diagnostics(
            _optimizer(manager, ExpectedImprovement()),
            "cannot_verify_surrogate_uncertainty",
        )
        assert len(diagnostics) == 1
        assert diagnostics[0].severity is Severity.WARNING


def test_custom_manager_with_undeclared_channel_names_emits_warning() -> None:
    manager = _UndeclaredManager()
    acquisition = CompositeAcquisition(
        {"secondary": ExpectedImprovement()}, product_combine
    )
    diagnostics = _plan_diagnostics(
        _optimizer(manager, acquisition), "cannot_verify_acquisition_channel"
    )
    assert len(diagnostics) == 1
    assert diagnostics[0].severity is Severity.WARNING


def test_surrogate_base_default_is_a_false_declaration() -> None:
    diagnostics = _plan_diagnostics(
        _optimizer(
            GlobalSurrogateManager(_InheritedFalseSurrogate()),
            ExpectedImprovement(),
        ),
        "surrogate_uncertainty_mismatch",
    )
    assert len(diagnostics) == 1
    assert diagnostics[0].severity is Severity.ERROR


@pytest.mark.parametrize(
    "strategy",
    [
        IndividualBasedStrategy,
        lambda: PreSelectionStrategy(n_candidates=4, n_select=1),
        lambda: GenerationBasedStrategy(gen_ctrl=2),
    ],
    ids=("individual", "pre-selection", "generation-based"),
)
def test_validate_uses_the_compiled_uncertainty_diagnostic_before_compilation(
    strategy,
) -> None:
    optimizer = _optimizer(_rbf_manager(), ExpectedImprovement(), strategy())
    issues = optimizer.validate()
    assert optimizer.executable_plan is None
    assert any("surrogate_uncertainty_mismatch" in issue for issue in issues)
    before_compile = _uncertainty_diagnostics(
        optimizer, "surrogate_uncertainty_mismatch"
    )
    assert len(before_compile) == 1

    optimizer._compile_plan()
    assert optimizer.executable_plan is not None
    compiled = tuple(
        item
        for item in optimizer.executable_plan.diagnostics
        if item.code == "surrogate_uncertainty_mismatch"
    )
    assert compiled == before_compile
    optimizer.validate()
    assert (
        _uncertainty_diagnostics(optimizer, "surrogate_uncertainty_mismatch")
        == compiled
    )


def test_validation_builds_strategy_before_runtime_capability_lookup() -> None:
    events: list[str] = []

    class PreparedStrategy(IndividualBasedStrategy):
        def build_graph(self, provider):
            events.append("build_graph")
            return super().build_graph(provider)

    def offered(provider):
        events.append("offered_capabilities")
        tuple(provider.strategy.pipeline.steps)
        return frozenset()

    original = next(
        registration
        for registration in default_runtime_registry.registrations()
        if registration.name == "sync"
    )
    default_runtime_registry.replace(
        "sync",
        RuntimeRegistration(
            name="sync",
            matches=original.matches,
            factory=original.factory,
            capability_provider=offered,
        ),
    )
    try:
        strategy = PreparedStrategy()
        optimizer = _optimizer(_rbf_manager(), ExpectedImprovement(), strategy)
        optimizer.validate()
    finally:
        default_runtime_registry.replace("sync", original)

    assert events == ["build_graph", "offered_capabilities"]
    assert strategy.pipeline is None
    assert _uncertainty_diagnostics(optimizer, "surrogate_uncertainty_mismatch")


def test_validate_rechecks_uncertainty_after_plan_configuration_changes() -> None:
    rejected = _optimizer(_rbf_manager(), ExpectedImprovement()).set_termination(
        Termination(max_fe(30))
    )
    with pytest.raises(ConfigurationError):
        rejected.run()
    rejected.set_acquisition(MeanPrediction())
    assert not any(
        "surrogate_uncertainty_mismatch" in issue for issue in rejected.validate()
    )
    rejected.run()

    accepted = _optimizer(_gpr_manager(), ExpectedImprovement()).set_termination(
        Termination(max_fe(30))
    )
    accepted.run()
    accepted.set_surrogate_manager(_rbf_manager())
    assert any(
        "surrogate_uncertainty_mismatch" in issue for issue in accepted.validate()
    )


@pytest.mark.parametrize("strategy", [DirectStrategy, SteadyStateStrategy])
def test_strategies_without_acquisition_have_no_uncertainty_diagnostic(
    strategy,
) -> None:
    optimizer = _optimizer(_rbf_manager(), ExpectedImprovement(), strategy=strategy())
    optimizer.validate()
    assert not _uncertainty_diagnostics(optimizer, "surrogate_uncertainty_mismatch")
    optimizer._compile_plan()
    assert not _plan_diagnostics(optimizer, "surrogate_uncertainty_mismatch")


@pytest.mark.parametrize("stage_component", ["acquisition", "manager"])
def test_stage_local_components_are_checked_before_compilation(stage_component) -> None:
    from saealib.strategies.ib import build_ib_template

    declaration = build_ib_template()
    steps = []
    for stage in declaration.steps:
        if stage_component == "acquisition" and isinstance(stage, AcquisitionStage):
            replacement = AcquisitionStage(ExpectedImprovement())
            replacement.execution_boundary = stage.execution_boundary
            stage = replacement
        elif stage_component == "manager" and isinstance(stage, SurrogatePredictStage):
            replacement = SurrogatePredictStage(_rbf_manager())
            replacement.execution_boundary = stage.execution_boundary
            stage = replacement
        steps.append(stage)
    strategy = TemplateStrategy(Pipeline(name=declaration.name, steps=steps))
    manager = _rbf_manager() if stage_component == "acquisition" else _gpr_manager()
    acquisition = (
        MeanPrediction() if stage_component == "acquisition" else ExpectedImprovement()
    )
    optimizer = _optimizer(manager, acquisition, strategy)

    optimizer.validate()
    before = _uncertainty_diagnostics(optimizer, "surrogate_uncertainty_mismatch")
    assert len(before) == 1
    assert before[0].severity is Severity.ERROR
    assert optimizer.executable_plan is None
    optimizer._compile_plan()
    assert optimizer.executable_plan is not None
    compiled = tuple(
        item
        for item in optimizer.executable_plan.diagnostics
        if item.code == "surrogate_uncertainty_mismatch"
    )
    assert compiled == before


class _GraphBuildFailureStrategy(IndividualBasedStrategy):
    def build_graph(self, provider):
        raise ValueError("graph unavailable")


def test_validate_skips_uncertainty_check_when_graph_build_fails() -> None:
    optimizer = _optimizer(
        _rbf_manager(), ExpectedImprovement(), _GraphBuildFailureStrategy()
    )
    optimizer.validate()
    assert not _uncertainty_diagnostics(optimizer, "surrogate_uncertainty_mismatch")


@pytest.mark.parametrize("acquisition", [ExpectedImprovement, MeanPrediction])
def test_nested_composite_without_objective_channel_is_an_error(acquisition) -> None:
    manager = CompositeSurrogateManager(
        {"a": CompositeSurrogateManager({"x": _gpr_manager()})}
    )
    combined = CompositeAcquisition({"a": acquisition()}, product_combine)
    optimizer = _optimizer(manager, combined)

    issues = optimizer.validate()
    assert any("missing_acquisition_channel" in issue for issue in issues)
    before = _uncertainty_diagnostics(optimizer, "missing_acquisition_channel")
    assert len(before) == 1
    assert before[0].severity is Severity.ERROR
    compiled = _plan_diagnostics(optimizer, "missing_acquisition_channel")
    assert compiled == before


def test_pairwise_win_rate_channel_runs_with_composite_acquisition() -> None:
    from examples._support import reference_problem

    manager = PairwiseSurrogateManager(
        SklearnRFCClassificationSurrogate(n_estimators=8, random_state=37),
        training_set=PairwiseComparisonSet(),
    )
    acquisition = CompositeAcquisition({"win_rate": MeanPrediction()}, product_combine)
    optimizer = (
        Optimizer(reference_problem(), seed=37)
        .set_initializer(LHSInitializer(6, 4, 37))
        .set_algorithm(PSO())
        .set_surrogate_manager(manager)
        .set_acquisition(acquisition)
        .set_strategy(IndividualBasedStrategy(0.5))
        .set_termination(Termination(max_fe(6)))
    )

    optimizer.resolve_defaults()
    optimizer.validate()
    channel_diagnostics = _uncertainty_diagnostics(
        optimizer, "cannot_verify_acquisition_channel"
    )
    assert len(channel_diagnostics) == 1
    assert channel_diagnostics[0].severity is Severity.WARNING
    optimizer.run()


def test_acquisition_direction_length_validation_is_preserved() -> None:
    optimizer = _optimizer(_gpr_manager(), ExpectedImprovement())
    cast(Any, optimizer.acquisition).direction = np.array([-1.0, -1.0])
    issues = optimizer.validate()
    assert any("ExpectedImprovement direction length" in issue for issue in issues)
