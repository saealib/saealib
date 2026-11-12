from __future__ import annotations

import sys
import warnings

import numpy as np
import pytest

from saealib import (
    GA,
    CrossoverBLXAlpha,
    LHSInitializer,
    MutationUniform,
    Optimizer,
    SequentialSelection,
    Termination,
    TruncationSelection,
)
from saealib._deprecation import (
    deprecated_argument_alias,
    deprecated_function,
    warn_deprecated,
)
from saealib.acquisition import MeanPrediction
from saealib.core.compiler.contract_diagnostics import (
    _optimizer_configuration_diagnostics,
)
from saealib.core.compiler.lowerer import lower_pipeline
from saealib.core.compiler.strategy_requirements import resolve_requires_surrogate
from saealib.exceptions import ValidationError
from saealib.execution import (
    AsyncEvaluationScheduler,
    AsyncEvaluator,
    SerialEvaluator,
)
from saealib.execution.runtime import execute_strategy_step
from saealib.identity import PopulationAttribute
from saealib.pipeline import Pipeline
from saealib.population import Archive, ParetoArchive, Population
from saealib.problem import Problem
from saealib.stages import (
    ArchiveUpdateStage,
    AskStage,
    CountGenerationStage,
    EvaluationAcknowledgeStage,
    EvaluationApplyStage,
    EvaluationCollectStage,
    EvaluationPlanStage,
    EvaluationSubmitStage,
    FeedbackStage,
    TellStage,
    stage_component,
)
from saealib.strategies import (
    DirectStrategy,
    GenerationBasedStrategy,
    IndividualBasedStrategy,
    PreSelectionStrategy,
    SteadyStateStrategy,
)
from saealib.strategies.base import OptimizationStrategy
from saealib.surrogate import LocalSurrogateManager, RBFSurrogate
from saealib.surrogate.rbf_kernels import GaussianKernel


def _problem() -> Problem:
    return Problem(
        func=lambda x: np.array([float(np.sum(np.asarray(x) ** 2))]),
        dim=2,
        n_obj=1,
        direction=np.array([-1.0]),
        lb=[-5.0, -5.0],
        ub=[5.0, 5.0],
    )


def _algorithm() -> GA:
    return GA(
        crossover=CrossoverBLXAlpha(prob=0.9, alpha=0.4),
        mutation=MutationUniform(prob_var=0.1),
        parent_selection=SequentialSelection(),
        survivor_selection=TruncationSelection(),
    )


def _optimizer(strategy, *, asynchronous: bool) -> Optimizer:
    optimizer = (
        Optimizer(_problem(), seed=19)
        .set_initializer(LHSInitializer(6, 4, 19))
        .set_strategy(strategy)
        .set_algorithm(_algorithm())
        .set_termination(Termination(lambda state: state.gen >= 1))
    )
    if resolve_requires_surrogate(strategy).effective:
        optimizer.set_surrogate_manager(
            LocalSurrogateManager(RBFSurrogate(GaussianKernel()))
        ).set_acquisition(MeanPrediction())
    if asynchronous:
        evaluator = AsyncEvaluator(SerialEvaluator())
        optimizer.set_evaluator(evaluator)
        optimizer.set_async_evaluation_scheduler(
            AsyncEvaluationScheduler(evaluator, max_pending=2)
        )
    else:
        optimizer.set_evaluator(SerialEvaluator())
    optimizer.resolve_defaults()
    return optimizer


def _strategies():
    return (
        DirectStrategy(),
        GenerationBasedStrategy(gen_ctrl=1),
        IndividualBasedStrategy(),
        PreSelectionStrategy(n_candidates=8, n_select=2),
        SteadyStateStrategy(),
    )


def test_warn_deprecated_includes_release_and_replacement() -> None:
    with pytest.warns(
        DeprecationWarning,
        match="probe is deprecated since 0.2.0; use replacement_api instead",
    ):
        warn_deprecated(
            "probe",
            since="0.2.0",
            replacement="replacement_api",
            stacklevel=1,
        )


def test_deprecated_function_can_warn_conditionally() -> None:
    @deprecated_function(
        since="0.2.0",
        replacement="replacement_api",
        condition=lambda enabled: enabled,
    )
    def function(enabled: bool) -> bool:
        return enabled

    with pytest.warns(
        DeprecationWarning,
        match=r"deprecated since 0\.2\.0; use replacement_api instead",
    ):
        assert function(True)
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        assert not function(False)


def test_deprecated_argument_alias_preserves_precedence() -> None:
    with pytest.warns(
        DeprecationWarning,
        match=r"deprecated since 0\.2\.0; use Probe\.new_name instead",
    ):
        assert (
            deprecated_argument_alias(
                "alias",
                "canonical",
                owner="Probe",
                old_name="old_name",
                new_name="new_name",
                since="0.2.0",
                stacklevel=1,
            )
            == "canonical"
        )


class _DenseView:
    def get_view(self, genomes):
        return genomes.array


def _archive_attrs() -> list[PopulationAttribute]:
    return [
        PopulationAttribute("x", np.float64, (2,)),
        PopulationAttribute("f", np.float64, (1,)),
        PopulationAttribute("g", np.float64, (0,)),
        PopulationAttribute("cv", np.float64, ()),
    ]


def test_population_dense_view_warns_and_keeps_alias() -> None:
    view = _DenseView()
    with pytest.warns(
        DeprecationWarning,
        match=(
            "Population.dense_view is deprecated since 0.2.0; "
            "use Population.dense_numeric_view instead"
        ),
    ) as caught:
        population = Population(_archive_attrs(), dense_view=view)
    assert len(caught) == 1
    assert caught[0].filename == __file__
    assert population._dense_numeric_view is view


@pytest.mark.parametrize(
    ("archive_type", "warning_name"),
    [(Archive, "Archive"), (ParetoArchive, "ParetoArchive")],
)
def test_archive_dense_view_warns_and_keeps_alias(archive_type, warning_name) -> None:
    view = _DenseView()
    with pytest.warns(
        DeprecationWarning,
        match=(
            f"{warning_name}.dense_view is deprecated since 0.2.0; "
            f"use {warning_name}.dense_numeric_view instead"
        ),
    ) as caught:
        if archive_type is ParetoArchive:
            archive = archive_type(
                _archive_attrs(), dense_view=view, direction=np.array([-1.0])
            )
        else:
            archive = archive_type(_archive_attrs(), dense_view=view)
    assert len(caught) == 1
    assert caught[0].filename == __file__
    assert archive._dense_numeric_view is view


@pytest.mark.parametrize("path", ["direct", "facade"])
@pytest.mark.parametrize(
    "strategy", _strategies(), ids=lambda strategy: type(strategy).__name__
)
def test_async_strategy_step_warns_at_user_callsite(strategy, path) -> None:
    optimizer = _optimizer(strategy, asynchronous=True)
    initializer = optimizer.initializer
    assert initializer is not None
    state = initializer.initialize(optimizer, optimizer.problem)

    with pytest.warns(
        DeprecationWarning,
        match=(
            r"execute_strategy_step\(\) with asynchronous evaluation is deprecated "
            r"since 0.2.0; use Optimizer.iterate\(\) or Optimizer.run\(\) instead"
        ),
    ) as caught:
        if isinstance(strategy, GenerationBasedStrategy):
            with pytest.raises(
                ValidationError,
                match="async strategy steps require a sequential graph execution seam",
            ):
                if path == "direct":
                    call_line = sys._getframe().f_lineno + 1
                    execute_strategy_step(strategy, state, optimizer)
                else:
                    call_line = sys._getframe().f_lineno + 1
                    strategy.step(state, optimizer)
        elif path == "direct":
            call_line = sys._getframe().f_lineno + 1
            result = execute_strategy_step(strategy, state, optimizer)
            assert (result.gen, result.fe) == (1, 6)
        else:
            call_line = sys._getframe().f_lineno + 1
            result = strategy.step(state, optimizer)
            assert (result.gen, result.fe) == (1, 6)
    assert len(caught) == 1
    assert caught[0].filename == __file__
    assert caught[0].lineno == call_line


@pytest.mark.parametrize("method", ["iterate", "run"])
def test_optimizer_async_execution_does_not_emit_deprecation_warning(method) -> None:
    optimizer = _optimizer(DirectStrategy(), asynchronous=True)
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        if method == "iterate":
            assert list(optimizer.iterate())
        else:
            assert optimizer.run().gen == 1


def test_synchronous_strategy_step_does_not_emit_deprecation_warning() -> None:
    strategy = DirectStrategy()
    optimizer = _optimizer(strategy, asynchronous=False)
    initializer = optimizer.initializer
    assert initializer is not None
    state = initializer.initialize(optimizer, optimizer.problem)
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        result = strategy.step(state, optimizer)
    assert result.gen == 1


class _GraphOnlyStrategy(OptimizationStrategy):
    """Custom strategy that lowers its own component graph."""

    requires_surrogate = False

    def build_graph(self, provider):
        return lower_pipeline(
            Pipeline(
                steps=[
                    stage_component(CountGenerationStage()),
                    stage_component(AskStage(provider.algorithm, n_offspring=2)),
                    stage_component(EvaluationPlanStage(None)),
                    stage_component(EvaluationSubmitStage(provider.evaluator)),
                    stage_component(EvaluationCollectStage(provider.evaluator)),
                    stage_component(EvaluationApplyStage()),
                    stage_component(ArchiveUpdateStage()),
                    stage_component(FeedbackStage(provider.feedback_builder)),
                    stage_component(TellStage(provider.algorithm)),
                    stage_component(EvaluationAcknowledgeStage(provider.evaluator)),
                ]
            )
        )


def _graph_only_optimizer(strategy) -> Optimizer:
    return (
        Optimizer(_problem(), seed=19)
        .set_initializer(LHSInitializer(6, 4, 19))
        .set_algorithm(_algorithm())
        .set_strategy(strategy)
        .set_termination(Termination(lambda state: state.gen >= 1))
        .set_evaluator(SerialEvaluator())
    )


def test_build_graph_override_warns_once_per_strategy_class() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        _graph_only_optimizer(_GraphOnlyStrategy()).run()
        _graph_only_optimizer(_GraphOnlyStrategy()).run()

    override_warnings = [
        item for item in caught if "build_graph(provider)" in str(item.message)
    ]
    assert len(override_warnings) == 1
    assert issubclass(override_warnings[0].category, DeprecationWarning)
    assert "TemplateStrategy" in str(override_warnings[0].message)


def test_plain_subclass_of_a_builtin_strategy_does_not_warn_build_graph() -> None:
    class UserSteadyStateStrategy(SteadyStateStrategy):
        pass

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        state = _graph_only_optimizer(UserSteadyStateStrategy()).run()

    assert state.gen == 1


def test_build_graph_override_strategy_keeps_its_surrogate_requirement() -> None:
    class SurrogateGraphOnlyStrategy(_GraphOnlyStrategy):
        requires_surrogate = True

    strategy = SurrogateGraphOnlyStrategy()
    findings = _optimizer_configuration_diagnostics(
        {
            "algorithm": object(),
            "strategy": strategy,
            "initializer": object(),
            "termination": object(),
            "surrogate_manager": None,
            "acquisition": None,
            "evaluation_planner": None,
            "feedback_builder": None,
            "evaluator": SerialEvaluator(),
            "cbmanager": object(),
        },
        {},
        _problem(),
        SerialEvaluator(),
    )

    assert any(finding.code == "missing_surrogate_manager" for finding in findings)
