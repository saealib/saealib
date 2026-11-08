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
from saealib.exceptions import ValidationError
from saealib.execution import (
    AsyncEvaluationScheduler,
    AsyncEvaluator,
    SerialEvaluator,
)
from saealib.execution.runtime import execute_strategy_step
from saealib.identity import PopulationAttribute
from saealib.population import Archive, ParetoArchive, Population
from saealib.problem import Problem
from saealib.strategies import (
    DirectStrategy,
    GenerationBasedStrategy,
    IndividualBasedStrategy,
    PreSelectionStrategy,
    SteadyStateStrategy,
)
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
    if strategy.requires_surrogate:
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
