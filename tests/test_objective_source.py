"""The ``objective_source`` population column records true vs predicted rows."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest

from saealib import (
    GA,
    PSO,
    CrossoverBLXAlpha,
    LHSInitializer,
    MutationUniform,
    Optimizer,
    Termination,
    TruncationSelection,
)
from saealib.acquisition import MeanPrediction
from saealib.execution import SerialEvaluator
from saealib.execution.history import History
from saealib.operators.selection import SequentialSelection
from saealib.population import Population
from saealib.problem import Problem
from saealib.stages import TrueEvaluationStage
from saealib.strategies import (
    DirectStrategy,
    GenerationBasedStrategy,
    IndividualBasedStrategy,
)
from saealib.surrogate.base import Surrogate, SurrogatePrediction
from saealib.surrogate.manager import LocalSurrogateManager

TRUE = 0
SURROGATE = 1


def _problem() -> Problem:
    return Problem(
        func=lambda x: np.array([float(np.sum(np.asarray(x) ** 2))]),
        dim=2,
        n_obj=1,
        direction=np.array([-1.0]),
        lb=[-1.0, -1.0],
        ub=[1.0, 1.0],
    )


def _ga(crossover: Any = None) -> GA:
    return GA(
        crossover=crossover or CrossoverBLXAlpha(prob=0.9, alpha=0.4),
        mutation=MutationUniform(prob_var=0.1),
        parent_selection=SequentialSelection(),
        survivor_selection=TruncationSelection(),
    )


def _optimizer(
    strategy: Any,
    *,
    algorithm: Any = None,
    n_archive: int = 12,
    n_population: int = 8,
    seed: int = 7,
) -> Optimizer:
    return (
        Optimizer(_problem(), seed=seed)
        .set_initializer(LHSInitializer(n_archive, n_population))
        .set_algorithm(algorithm or _ga())
        .set_termination(Termination(lambda state: False))
        .set_history(["summary", "evaluation"])
        .set_strategy(strategy)
    )


class _OptimisticSurrogate(Surrogate):
    """Predicts an objective value below every true value of ``_problem``."""

    def fit(self, train_x: Any, train_y: Any) -> None:
        pass

    def predict(self, test_x: Any) -> SurrogatePrediction:
        n = len(np.atleast_2d(np.asarray(test_x)))
        return SurrogatePrediction.objective(value=np.full((n, 1), -1.0))


def _true_values_by_id(history: History) -> dict[int, np.ndarray]:
    """Return the last true objective value recorded for each candidate ID."""
    status = np.asarray(history.channel("evaluation")["status_code"])
    candidate_blocks = history.blocks("evaluation", "candidate_ids")
    f_blocks = history.blocks("evaluation", "f")
    values: dict[int, np.ndarray] = {}
    for row, code in enumerate(status):
        if int(code) not in (2, 3):
            continue
        ids = np.asarray(candidate_blocks[row]).reshape(-1)
        f = np.asarray(f_blocks[row]).reshape(len(ids), -1)
        for position, candidate_id in enumerate(ids):
            values[int(candidate_id)] = f[position]
    return values


def _expected_sources(population: Population, history: History) -> np.ndarray:
    truth = _true_values_by_id(history)
    ids = population.get_array("id")
    f = population.get_array("f")
    expected = np.full(len(population), SURROGATE, dtype=np.uint8)
    for row, candidate_id in enumerate(ids):
        value = truth.get(int(candidate_id))
        if value is not None and np.array_equal(value, f[row]):
            expected[row] = TRUE
    return expected


def _sources(population: Population) -> np.ndarray:
    return population.get_array("objective_source")


def _steps(optimizer: Optimizer, n: int) -> Iterator[Any]:
    """Yield the live state after each of ``n`` steps; check it before advancing."""
    iterator = optimizer.iterate()
    for _ in range(n):
        yield next(iterator)


def test_initial_population_and_archive_rows_are_true() -> None:
    state = next(_steps(_optimizer(DirectStrategy()), 1))

    assert state.population.schema["objective_source"].dtype == np.uint8
    assert np.all(_sources(state.population) == TRUE)
    assert np.all(_sources(state.archive) == TRUE)


def test_direct_strategy_keeps_every_row_true() -> None:
    for state in _steps(_optimizer(DirectStrategy()), 5):
        assert np.all(_sources(state.population) == TRUE)
        assert np.all(_sources(state.archive) == TRUE)


def test_generation_based_marks_rows_from_surrogate_feedback_as_predicted() -> None:
    seen: set[int] = set()
    for state in _steps(_optimizer(GenerationBasedStrategy(gen_ctrl=4)), 4):
        sources = _sources(state.population)
        np.testing.assert_array_equal(
            sources, _expected_sources(state.population, state.history)
        )
        seen.update(int(value) for value in sources)
        assert np.all(_sources(state.archive) == TRUE)
    assert seen == {TRUE, SURROGATE}


def test_generation_based_marks_duplicate_x_rows_dropped_by_the_archive_as_true() -> (
    None
):
    def duplicate_first_child(offspring, parents, rng, ctx=None):
        offspring = np.array(offspring, copy=True)
        offspring[1] = offspring[0]
        return offspring

    crossover = CrossoverBLXAlpha(prob=0.9, alpha=0.4).with_post(duplicate_first_child)
    optimizer = _optimizer(
        GenerationBasedStrategy(gen_ctrl=2), algorithm=_ga(crossover)
    ).set_initializer(LHSInitializer(12, 8))
    dropped_true_rows = 0
    for state in _steps(optimizer, 6):
        population = state.population
        sources = _sources(population)
        np.testing.assert_array_equal(
            sources, _expected_sources(population, state.history)
        )
        archive_ids = set(state.archive.get_array("id").tolist())
        dropped_true_rows += sum(
            1
            for candidate_id, source in zip(
                population.get_array("id").tolist(), sources, strict=True
            )
            if source == TRUE and candidate_id not in archive_ids
        )
    assert dropped_true_rows > 0


@pytest.mark.parametrize("surrogate", [None, _OptimisticSurrogate])
def test_individual_based_pso_keeps_unevaluated_particles_marked_predicted(
    surrogate: Any,
) -> None:
    optimizer = _optimizer(
        IndividualBasedStrategy(evaluation_ratio=0.25), algorithm=PSO()
    )
    if surrogate is not None:
        optimizer.set_surrogate_manager(LocalSurrogateManager(surrogate()))
        optimizer.set_acquisition(MeanPrediction())

    seen: set[int] = set()
    for state in _steps(optimizer, 4):
        sources = _sources(state.population)
        np.testing.assert_array_equal(
            sources, _expected_sources(state.population, state.history)
        )
        seen.update(int(value) for value in sources)
    assert seen == {TRUE, SURROGATE}


def test_true_evaluation_of_a_predicted_row_marks_it_true() -> None:
    steps = _steps(_optimizer(GenerationBasedStrategy(gen_ctrl=4)), 2)
    next(steps)
    state = next(steps)
    predicted = np.flatnonzero(_sources(state.population) == SURROGATE)
    assert len(predicted)
    offspring = state.population.extract(predicted)
    predicted_f = np.array(offspring.get_array("f"), copy=True)

    evaluated = TrueEvaluationStage(SerialEvaluator()).execute(
        state.replace(offspring=offspring)
    )

    reevaluated = evaluated.offspring
    assert reevaluated is not None
    assert np.all(_sources(reevaluated) == TRUE)
    assert not np.array_equal(reevaluated.get_array("f"), predicted_f)


def test_generation_based_with_pymoo_algorithm_keeps_predicted_survivors_marked() -> (
    None
):
    pymoo_ga = pytest.importorskip("pymoo.algorithms.soo.nonconvex.ga")
    from saealib.algorithms.pymoo_algorithm import PymooAlgorithm

    optimizer = _optimizer(
        GenerationBasedStrategy(gen_ctrl=3),
        algorithm=PymooAlgorithm(pymoo_ga.GA(pop_size=8)),
    )

    seen: set[int] = set()
    for state in _steps(optimizer, 6):
        sources = _sources(state.population)
        np.testing.assert_array_equal(
            sources, _expected_sources(state.population, state.history)
        )
        seen.update(int(value) for value in sources)
    assert seen == {TRUE, SURROGATE}
