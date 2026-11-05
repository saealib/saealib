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
from saealib.context import EvaluationPlanState
from saealib.execution import SerialEvaluator
from saealib.execution.evaluator import (
    EvaluationRequest,
    EvaluationResult,
    EvaluationStatus,
    EvaluationUpdate,
    PendingEvaluation,
)
from saealib.execution.history import History
from saealib.execution.initializer import (
    GenomeInitializer,
    RandomInitializer,
    SobolInitializer,
)
from saealib.execution.scheduler import AsyncEvaluationScheduler
from saealib.operators.selection import SequentialSelection
from saealib.policies.evaluation import EvaluationPlan
from saealib.policies.feedback import FeedbackBuilder, FeedbackResult
from saealib.population import Archive, Population
from saealib.population.population import OBJECTIVE_SOURCE_COLUMN
from saealib.problem import Problem
from saealib.stages import ArchiveUpdateStage, EvaluationApplyStage, TrueEvaluationStage
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
    initializer: Any = None,
    n_archive: int = 12,
    n_population: int = 8,
    seed: int = 7,
) -> Optimizer:
    return (
        Optimizer(_problem(), seed=seed)
        .set_initializer(initializer or LHSInitializer(n_archive, n_population, seed))
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


def _population_without_objective_source(
    source: Population, candidate_id: int, x: np.ndarray, f: np.ndarray
) -> Population:
    attrs = [attr for attr in source.attrs if attr.name != "objective_source"]
    population = Population(
        attrs,
        1,
        dense_numeric_view=source._dense_numeric_view,
        space=source.space,
    )
    values = {
        name: source.get_array(name)[:1]
        for name in ("g", "cv")
        if name in source.schema
    }
    values.update(
        {
            "x": np.asarray(x, dtype=np.float64),
            "f": np.asarray(f, dtype=np.float64),
            "id": np.array([candidate_id], dtype=np.int64),
        }
    )
    population._extend_internal(values, preserve_ids=True)
    return population


def _steps(optimizer: Optimizer, n: int) -> Iterator[Any]:
    """Yield the live state after each of ``n`` steps; check it before advancing."""
    iterator = optimizer.iterate()
    for _ in range(n):
        yield next(iterator)


@pytest.mark.parametrize(
    "initializer",
    [
        GenomeInitializer(12, 8, seed=7),
        LHSInitializer(12, 8, seed=7),
        RandomInitializer(12, 8, seed=7),
        SobolInitializer(12, 8, seed=7),
    ],
    ids=["genome", "lhs", "random", "sobol"],
)
def test_initial_population_and_archive_rows_are_true(initializer: Any) -> None:
    state = next(_steps(_optimizer(DirectStrategy(), initializer=initializer), 1))

    assert state.population.schema["objective_source"].dtype == np.uint8
    assert np.all(_sources(state.population) == TRUE)
    assert np.all(_sources(state.archive) == TRUE)


def test_archive_update_marks_rows_true_when_offspring_has_no_source_column() -> None:
    state = next(_steps(_optimizer(DirectStrategy()), 1))
    x = np.array([[0.1234567, -0.7654321]], dtype=np.float64)
    f = np.array([[np.sum(x**2)]], dtype=np.float64)

    class AlgorithmWithoutObjectiveSource:
        def ask(self, population: Population) -> Population:
            return _population_without_objective_source(population, 9999, x, f)

    offspring = AlgorithmWithoutObjectiveSource().ask(state.population)
    state = state.replace(
        evaluated_offspring=offspring,
        evaluation_plan=None,
        evaluation_plan_state=None,
    )

    updated = ArchiveUpdateStage().execute(state)

    row = int(np.flatnonzero(updated.archive.get_array("id") == 9999)[0])
    assert updated.archive.get_array("objective_source")[row] == TRUE


def test_archive_update_marks_pareto_rows_true_when_only_it_has_source_column() -> None:
    state = next(_steps(_optimizer(DirectStrategy()), 1))
    x = np.array([[0.0, 0.0]], dtype=np.float64)
    f = np.array([[0.0]], dtype=np.float64)
    offspring = _population_without_objective_source(state.population, 9999, x, f)
    archive = Archive(
        [attr for attr in state.archive.attrs if attr.name != OBJECTIVE_SOURCE_COLUMN],
        dense_numeric_view=state.archive._dense_numeric_view,
        space=state.archive.space,
    )
    state = state.replace(
        evaluated_offspring=offspring,
        archive=archive,
        evaluation_plan=None,
        evaluation_plan_state=None,
    )

    updated = ArchiveUpdateStage().execute(state)

    assert OBJECTIVE_SOURCE_COLUMN not in updated.archive.schema
    assert OBJECTIVE_SOURCE_COLUMN in updated.pareto_archive.schema
    row = int(np.flatnonzero(updated.pareto_archive.get_array("id") == 9999)[0])
    assert updated.pareto_archive.get_array(OBJECTIVE_SOURCE_COLUMN)[row] == TRUE


def test_async_archive_apply_marks_rows_true_when_owner_has_no_source_column() -> None:
    state = next(_steps(_optimizer(DirectStrategy()), 1))
    x = np.array([[0.1234567, -0.7654321]], dtype=np.float64)
    f = np.array([[np.sum(x**2)]], dtype=np.float64)
    owner = _population_without_objective_source(state.population, 9999, x, f)
    request_id = np.int64(400)
    request = EvaluationRequest(request_id, np.array([9999]), x)
    result = EvaluationResult(
        f,
        np.empty((1, 0), dtype=np.float64),
        np.zeros(1, dtype=np.float64),
        candidate_ids=np.array([9999], dtype=np.int64),
    )
    update = EvaluationUpdate(
        request_id, EvaluationStatus.COMPLETED, np.array([9999]), result
    )
    state = state.replace(
        offspring=owner,
        evaluation_owners={request_id: owner},
        pending_evaluations={
            request_id: PendingEvaluation(
                request, EvaluationStatus.PENDING, np.empty(0, dtype=np.int64)
            )
        },
    )
    scheduler = AsyncEvaluationScheduler(SerialEvaluator())
    scheduler._apply_population(state, update)

    scheduler._apply_archive(state, update)

    row = int(np.flatnonzero(state.archive.get_array("id") == 9999)[0])
    assert state.archive.get_array("objective_source")[row] == TRUE


def test_async_feedback_writes_the_feedback_result_source() -> None:
    class NonzeroSourceFeedback(FeedbackBuilder):
        def build(self, candidates, prediction, evaluation, evaluated_indices, ctx):
            del candidates, prediction, evaluated_indices, ctx
            assert evaluation is not None
            assert evaluation.candidate_ids is not None
            size = len(evaluation.candidate_ids)
            return FeedbackResult(
                evaluation.candidate_ids,
                evaluation.f,
                evaluation.g,
                evaluation.cv,
                np.ones(size, dtype=bool),
                np.full(size, 2, dtype=np.uint8),
            )

    state = next(_steps(_optimizer(DirectStrategy()), 1))
    owner = state.population.extract(np.array([0]))
    candidate_ids = owner.get_array("id")
    request_id = np.int64(401)
    request = EvaluationRequest(request_id, candidate_ids, owner.get_array("x"))
    pending = PendingEvaluation(
        request, EvaluationStatus.PENDING, np.empty(0, dtype=np.int64)
    )
    result = EvaluationResult(
        owner.get_array("f"),
        owner.get_array("g"),
        owner.get_array("cv"),
        candidate_ids=candidate_ids,
    )
    update = EvaluationUpdate(
        request_id, EvaluationStatus.COMPLETED, candidate_ids, result
    )
    state = state.replace(
        offspring=owner,
        evaluation_owners={request_id: owner},
        pending_evaluations={request_id: pending},
    )
    scheduler = AsyncEvaluationScheduler(
        SerialEvaluator(), feedback_builder=NonzeroSourceFeedback()
    )

    updated = scheduler._apply_feedback(state, update)

    assert updated.feedback_result is not None
    np.testing.assert_array_equal(updated.feedback_result.source, [2])
    np.testing.assert_array_equal(_sources(owner), [2])


def test_evaluation_apply_marks_other_requests_in_the_plan_true() -> None:
    state = next(_steps(_optimizer(DirectStrategy()), 1))
    offspring = state.population.extract(np.array([0, 1]))
    ids = offspring.get_array("id")
    offspring.update_array(OBJECTIVE_SOURCE_COLUMN, np.ones(2, dtype=np.uint8))
    requests = tuple(
        EvaluationRequest(
            np.int64(1000 + index),
            np.array([candidate_id]),
            offspring.x[index : index + 1],
        )
        for index, candidate_id in enumerate(ids)
    )
    updates = []
    for index, request in enumerate(requests):
        candidate_id = request.candidate_ids
        result = EvaluationResult(
            np.array([[float(index + 1)]], dtype=np.float64),
            np.empty((1, 0), dtype=np.float64),
            np.zeros(1, dtype=np.float64),
            candidate_ids=candidate_id,
        )
        updates.append(
            EvaluationUpdate(
                request.request_id,
                EvaluationStatus.COMPLETED,
                candidate_id,
                result,
            )
        )
    first, second = requests
    state = state.replace(
        offspring=offspring,
        evaluation_request=first,
        evaluation_plan=EvaluationPlan(requests),
        evaluation_plan_state=EvaluationPlanState(
            submitted=tuple(int(request.request_id) for request in requests),
            completed=tuple(int(request.request_id) for request in requests),
        ),
        evaluation_updates=[updates[0]],
        evaluation_plan_updates={int(second.request_id): [updates[1]]},
        pending_evaluations={
            int(first.request_id): PendingEvaluation(
                first, EvaluationStatus.PENDING, np.empty(0, dtype=np.int64)
            )
        },
    )

    updated = EvaluationApplyStage().execute(state)

    assert updated.offspring is not None
    np.testing.assert_array_equal(_sources(updated.offspring), [TRUE, TRUE])


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
