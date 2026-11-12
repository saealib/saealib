from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest
from _evaluation_helpers import _ControlledEvaluator

from saealib import (
    GA,
    CrossoverBLXAlpha,
    GenerationBasedStrategy,
    IndividualBasedStrategy,
    LHSInitializer,
    MutationUniform,
    Optimizer,
    PreSelectionStrategy,
    SequentialSelection,
    SteadyStateStrategy,
    Termination,
    TruncationSelection,
    max_steps,
)
from saealib.acquisition import MeanPrediction as MeanPrediction
from saealib.comparators import SingleObjectiveComparator
from saealib.context import OptimizationState
from saealib.core.contracts import StateContract
from saealib.core.state import RUNTIME_GENERATION
from saealib.execution import AsyncEvaluationScheduler
from saealib.pipeline import Loop, Pipeline, Repeat
from saealib.problem import Problem
from saealib.stages import CountGenerationStage
from saealib.strategies.base import TemplateStrategy
from saealib.strategies.direct import DirectStrategy
from saealib.surrogate.manager import SurrogateManager
from saealib.surrogate.prediction import SurrogatePrediction
from saealib.termination import max_gen


def _sphere(x: np.ndarray) -> np.ndarray:
    return np.array([np.sum(x**2)])


class _ConstantSurrogateManager(SurrogateManager):
    def predict(self, candidates_x, archive, *, refit=True, ctx):
        return SurrogatePrediction.objective(value=np.ones((len(candidates_x), 1)))


class _AtGeneration:
    def __init__(self, target: int) -> None:
        self.target = target

    def contract(self) -> StateContract:
        return StateContract(reads=(RUNTIME_GENERATION,))

    def evaluate(self, view) -> bool:
        return int(view.get(RUNTIME_GENERATION)) >= self.target


class _DerivedDirectStrategy(DirectStrategy):
    pass


def _make_problem() -> Problem:
    return Problem(
        func=_sphere,
        dim=2,
        n_obj=1,
        direction=np.array([-1.0]),
        lb=[-2.0, -2.0],
        ub=[2.0, 2.0],
        comparator=SingleObjectiveComparator(),
    )


def _make_optimizer(strategy, *, asynchronous: bool, step_limit: int = 2) -> Optimizer:
    optimizer = (
        Optimizer(_make_problem(), seed=7)
        .set_initializer(LHSInitializer(4, 4))
        .set_algorithm(
            GA(
                crossover=CrossoverBLXAlpha(prob=0.9, alpha=0.5),
                mutation=MutationUniform(prob_var=0.1),
                parent_selection=SequentialSelection(),
                survivor_selection=TruncationSelection(),
            )
        )
        .set_strategy(strategy)
        .set_termination(Termination(max_steps(step_limit)))
    )
    if isinstance(
        strategy,
        (GenerationBasedStrategy, IndividualBasedStrategy, PreSelectionStrategy),
    ):
        optimizer.set_surrogate_manager(_ConstantSurrogateManager()).set_acquisition(
            MeanPrediction()
        )
    if asynchronous:
        evaluator = _ControlledEvaluator()
        evaluator.release_all = True
        optimizer.set_evaluator(evaluator)
        optimizer.set_async_evaluation_scheduler(
            AsyncEvaluationScheduler(evaluator, max_pending=2)
        )
    return optimizer


@pytest.mark.parametrize(
    "strategy_factory",
    [
        DirectStrategy,
        _DerivedDirectStrategy,
        lambda: GenerationBasedStrategy(gen_ctrl=2),
        lambda: IndividualBasedStrategy(evaluation_ratio=0.5),
        lambda: PreSelectionStrategy(n_candidates=4, n_select=2),
        SteadyStateStrategy,
    ],
    ids=[
        "direct",
        "derived_direct",
        "generation_based",
        "individual_based",
        "preselection",
        "steady_state",
    ],
)
@pytest.mark.parametrize("asynchronous", [False, True])
def test_step_is_zero_for_initial_state_then_counts_completed_flows(
    strategy_factory, asynchronous
) -> None:
    optimizer = _make_optimizer(strategy_factory(), asynchronous=asynchronous)

    steps = [state.step for state in optimizer.iterate()]

    assert steps == [0, 1, 2]


@pytest.mark.parametrize("asynchronous", [False, True])
def test_region_history_schema_updates_after_strategy_replacement(asynchronous) -> None:
    optimizer = _make_optimizer(DirectStrategy(), asynchronous=asynchronous)
    optimizer.set_surrogate_manager(_ConstantSurrogateManager()).set_acquisition(
        MeanPrediction()
    )
    iterator = optimizer.iterate()

    assert next(iterator).step == 0
    assert next(iterator).step == 1

    optimizer.set_strategy(GenerationBasedStrategy(gen_ctrl=2))
    result = next(iterator)

    assert result.step == 2
    assert result.gen == 4
    assert result.region_counts["generation_based.surrogate_generations"] == 2
    assert result.history is not None
    column = "region_generation_based.surrogate_generations_iterations"
    summary = result.history.channel("summary")
    assert column in summary
    assert summary[column][-1] == 2
    assert np.all(summary[column][:-1] == 0)


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("checkpoint_format", ["portable", "pickle"])
def test_region_history_schema_resumes_after_strategy_replacement(
    tmp_path, checkpoint_format, asynchronous
) -> None:
    optimizer = _make_optimizer(
        GenerationBasedStrategy(gen_ctrl=2),
        asynchronous=asynchronous,
        step_limit=3,
    )
    iterator = optimizer.iterate()

    assert next(iterator).step == 0
    first = next(iterator)
    assert first.step == 1
    region_count = first.region_counts["generation_based.surrogate_generations"]

    optimizer.set_strategy(DirectStrategy())
    boundary = next(iterator)
    assert boundary.step == 2
    assert boundary.region_counts["generation_based.surrogate_generations"] == (
        region_count
    )
    assert boundary.history is not None
    column = "region_generation_based.surrogate_generations_iterations"
    boundary_values = boundary.history.channel("summary")[column].copy()

    checkpoint = tmp_path / f"replacement-{checkpoint_format}-{asynchronous}"
    if checkpoint_format == "portable":
        boundary.save(checkpoint)
        restored = type(boundary).load(checkpoint.with_suffix(".npz"), _make_problem())
        resumed_optimizer = optimizer
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            optimizer.save_pickle(boundary, checkpoint)
            resumed_optimizer, restored = Optimizer.load_pickle(
                checkpoint.with_suffix(".pkl")
            )

    resumed = resumed_optimizer.run_from(restored)

    assert resumed.step == 3
    assert (
        resumed.region_counts["generation_based.surrogate_generations"] == region_count
    )
    assert resumed.history is not None
    summary = resumed.history.channel("summary")
    assert np.array_equal(summary[column][:-1], boundary_values)
    assert summary[column][-1] == region_count


def test_named_regions_record_cumulative_completed_iterations() -> None:
    inner_repeat = Repeat(Pipeline([CountGenerationStage()]), 3, name="inner")
    outer_repeat = Repeat(Pipeline([inner_repeat]), 2, name="outer")
    pipeline = Pipeline(
        name="region_flow",
        steps=[
            Repeat(Pipeline([CountGenerationStage()]), 0, name="empty"),
            outer_repeat,
            Loop(
                Pipeline([CountGenerationStage()]),
                until=_AtGeneration(8),
                name="looping",
            ),
        ],
    )
    optimizer = (
        Optimizer(_make_problem(), seed=11)
        .set_initializer(LHSInitializer(4, 4))
        .set_algorithm(
            GA(
                crossover=CrossoverBLXAlpha(prob=0.9, alpha=0.5),
                mutation=MutationUniform(prob_var=0.1),
                parent_selection=SequentialSelection(),
                survivor_selection=TruncationSelection(),
            )
        )
        .set_strategy(TemplateStrategy(pipeline))
        .set_termination(Termination(max_steps(1)))
    )

    result = optimizer.run()

    assert result.history is not None
    summary = result.history.channel("summary")
    counts = {
        key.removeprefix("region_").removesuffix("_iterations"): values
        for key, values in summary.items()
        if key.startswith("region_") and key.endswith("_iterations")
    }
    assert result.gen == 8
    assert (
        next(values[-1] for key, values in counts.items() if key.endswith(".empty"))
        == 0
    )
    assert (
        next(values[-1] for key, values in counts.items() if key.endswith(".outer"))
        == 2
    )
    assert (
        next(
            values[-1] for key, values in counts.items() if key.endswith(".outer.inner")
        )
        == 6
    )
    assert (
        next(values[-1] for key, values in counts.items() if key.endswith(".looping"))
        == 2
    )
    assert all(values[0] == 0 for values in counts.values())


def test_max_steps_composes_with_generation_termination() -> None:
    optimizer = (
        Optimizer(_make_problem(), seed=9)
        .set_initializer(LHSInitializer(4, 4))
        .set_algorithm(
            GA(
                crossover=CrossoverBLXAlpha(prob=0.9, alpha=0.5),
                mutation=MutationUniform(prob_var=0.1),
                parent_selection=SequentialSelection(),
                survivor_selection=TruncationSelection(),
            )
        )
        .set_strategy(DirectStrategy())
        .set_termination(Termination(max_steps(5) | max_gen(1)))
    )

    steps = [state.step for state in optimizer.iterate()]

    assert steps == [0, 1]


def test_checkpoint_without_flow_counts_loads_and_resumes() -> None:
    problem = _make_problem()
    checkpoint = Path(__file__).parent / "fixtures" / "checkpoint_v010.npz"

    restored = OptimizationState.load(checkpoint, problem)

    assert restored.step == 0
    assert restored.region_counts == {}
    resumed = (
        Optimizer(problem, seed=23)
        .set_initializer(LHSInitializer(4, 4))
        .set_algorithm(
            GA(
                crossover=CrossoverBLXAlpha(prob=0.9, alpha=0.5),
                mutation=MutationUniform(prob_var=0.1),
                parent_selection=SequentialSelection(),
                survivor_selection=TruncationSelection(),
            )
        )
        .set_strategy(DirectStrategy())
        .set_termination(Termination(max_steps(1)))
        .run_from(restored)
    )

    assert resumed.step == 1


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("checkpoint_format", ["portable", "pickle"])
def test_step_and_region_counts_continue_after_resume(
    tmp_path, checkpoint_format, asynchronous
) -> None:
    problem = _make_problem()
    optimizer = (
        Optimizer(problem, seed=13)
        .set_initializer(LHSInitializer(4, 4))
        .set_algorithm(
            GA(
                crossover=CrossoverBLXAlpha(prob=0.9, alpha=0.5),
                mutation=MutationUniform(prob_var=0.1),
                parent_selection=SequentialSelection(),
                survivor_selection=TruncationSelection(),
            )
        )
        .set_surrogate_manager(_ConstantSurrogateManager())
        .set_acquisition(MeanPrediction())
        .set_strategy(GenerationBasedStrategy(gen_ctrl=2))
        .set_termination(Termination(max_steps(2)))
    )
    if asynchronous:
        evaluator = _ControlledEvaluator()
        evaluator.release_all = True
        optimizer.set_evaluator(evaluator).set_async_evaluation_scheduler(
            AsyncEvaluationScheduler(evaluator, max_pending=2)
        )
    iterator = optimizer.iterate()
    next(iterator)
    boundary = next(iterator)
    path = tmp_path / f"state-{checkpoint_format}-{asynchronous}"
    if checkpoint_format == "portable":
        boundary.save(path)
        restored = type(boundary).load(path.with_suffix(".npz"), problem)
        resumed_optimizer = optimizer
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            optimizer.save_pickle(boundary, path)
            resumed_optimizer, restored = Optimizer.load_pickle(
                path.with_suffix(".pkl")
            )

    assert restored.step == 1
    assert restored.region_counts
    resumed = resumed_optimizer.run_from(restored)

    assert list(resumed.region_counts.values()) == [4]
    assert resumed.history is not None
    region_column = next(
        key for key in resumed.history.channel("summary") if key.endswith("_iterations")
    )
    assert resumed.history.channel("summary")[region_column][-1] == 4
