"""Tests for execution history recording."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from saealib import (
    GA,
    CrossoverBLXAlpha,
    GenerationEndEvent,
    GenomeInitializer,
    LHSInitializer,
    MutationUniform,
    Optimizer,
    Problem,
    RandomInitializer,
    RepeatedEvaluation,
    SequentialSelection,
    SobolInitializer,
    Termination,
    TruncationSelection,
    max_fe,
    max_gen,
    maximize,
    minimize,
)
from saealib.acquisition import AcquisitionFunction, AcquisitionResult
from saealib.context import EvaluationPlanState, OptimizationState
from saealib.core.contracts.representation import RepresentationSpec
from saealib.exceptions import ValidationError
from saealib.execution import (
    AsyncEvaluationScheduler,
    EvaluationErrorInfo,
    EvaluationHandle,
    EvaluationRequest,
    EvaluationResult,
    EvaluationStatus,
    EvaluationUpdate,
    Evaluator,
    SerialEvaluator,
)
from saealib.execution.history import (
    History,
    record_decision,
    record_evaluations,
    record_generation,
)
from saealib.policies.evaluation import EvaluateAll
from saealib.population import Archive, ParetoArchive, Population, PopulationAttribute
from saealib.problem import (
    ConstraintHandler,
    EpsilonConstraintHandler,
    InequalityConstraint,
    linear_epsilon_schedule,
)
from saealib.result import Result
from saealib.space import ObjectSpace
from saealib.stages import AsyncEvaluationSubmitStage, EvaluationPlanStage
from saealib.strategies import (
    DirectStrategy,
    GenerationBasedStrategy,
    IndividualBasedStrategy,
)
from saealib.surrogate import PredictionChannel, SurrogatePrediction


def _problem(n_obj: int = 1) -> Problem:
    if n_obj == 1:

        def func(x: np.ndarray) -> np.ndarray:
            return np.array([np.sum(np.asarray(x) ** 2)])
    else:

        def func(x: np.ndarray) -> np.ndarray:
            values = np.asarray(x)
            return np.array([np.sum(values**2), np.sum((values - 0.5) ** 2)])

    return Problem(
        func=func,
        dim=2,
        n_obj=n_obj,
        direction=np.full(n_obj, -1.0),
        lb=[-1.0, -1.0],
        ub=[1.0, 1.0],
    )


def _observe(
    state: OptimizationState, update: EvaluationUpdate, attempt: int = 0
) -> None:
    """Mirror a stage/scheduler sink write against ``state.history``."""
    history = state.history
    assert history is not None
    history._observe_evaluation(update, attempt)


def _run_optimizer(problem: Problem, channels: list[str]) -> OptimizationState:
    return (
        Optimizer(problem, seed=7)
        .set_initializer(LHSInitializer(10, 3))
        .set_termination(Termination(max_fe(13)))
        .set_history(channels)
        .run()
    )


def _run_generation_based_optimizer(
    problem: Problem, channels: list[str]
) -> OptimizationState:
    return (
        Optimizer(problem, seed=7)
        .set_initializer(LHSInitializer(10, 3))
        .set_strategy(GenerationBasedStrategy(gen_ctrl=1))
        .set_termination(Termination(max_fe(13)))
        .set_history(channels)
        .run()
    )


class _NoScoresAcquisition(AcquisitionFunction):
    def evaluate(
        self,
        candidates_x: np.ndarray,
        prediction: SurrogatePrediction | None,
        archive: Archive,
        ctx: OptimizationState | None = None,
        *,
        prepared: Any = None,
    ) -> AcquisitionResult:
        return AcquisitionResult(scores=None)


class _CostEvaluator(Evaluator):
    """Return a distinct cost for every batch so recorded sums are observable."""

    def __init__(self) -> None:
        self.returned_costs: list[np.ndarray] = []

    def evaluate_batch(self, x: Any, problem: Problem) -> EvaluationResult:
        size = len(x)
        cost = np.full(size, len(self.returned_costs) + 1, dtype=np.float64)
        self.returned_costs.append(cost.copy())
        return EvaluationResult(
            f=np.zeros((size, problem.n_obj), dtype=np.float64),
            g=np.empty((size, problem.n_constraints), dtype=np.float64),
            cv=np.zeros(size, dtype=np.float64),
            cost=cost,
        )


class _FailingEvaluator(Evaluator):
    """Fail every submitted batch for the failed-update history test."""

    def __init__(self) -> None:
        self._request_candidate_ids: dict[int, np.ndarray] = {}

    def evaluate_batch(self, x: Any, problem: Problem) -> EvaluationResult:
        raise RuntimeError("expected test evaluation failure")

    def submit(self, request: Any, problem: Problem) -> Any:
        self._request_candidate_ids[int(request.request_id)] = np.array(
            request.candidate_ids, dtype=np.int64, copy=True
        )
        return super().submit(request, problem)

    def collect(self, handle: Any, *, wait: bool = True) -> list[EvaluationUpdate]:
        updates = super().collect(handle, wait=wait)
        return [
            EvaluationUpdate(
                request_id=update.request_id,
                status=update.status,
                candidate_ids=self._request_candidate_ids[int(update.request_id)],
                result=update.result,
                error=update.error,
                sequence=update.sequence,
            )
            for update in updates
        ]


_ALL_HISTORY_CHANNELS = [
    "summary",
    "front",
    "population",
    "surrogate_accuracy",
    "decision_candidates",
    "evaluation",
]


def _run_history_path(path: str, channels: list[str]) -> OptimizationState:
    optimizer = (
        Optimizer(_problem(2), seed=7)
        .set_initializer(LHSInitializer(10, 3))
        .set_termination(Termination(max_fe(13)))
        .set_history(channels)
    )
    if path == "structured":
        optimizer.set_strategy(GenerationBasedStrategy(gen_ctrl=1))
    if path == "async":
        evaluator = SerialEvaluator()
        optimizer.set_evaluator(evaluator).set_async_evaluation_scheduler(
            AsyncEvaluationScheduler(evaluator, max_pending=1)
        )
    return optimizer.run()


def _expected_surrogate_accuracy_pairs(
    state: OptimizationState,
) -> dict[tuple[int, int], tuple[np.ndarray, np.ndarray]]:
    """Join decision predictions and evaluation truth independently by ID."""
    assert state.history is not None
    predictions_by_candidate: dict[int, np.ndarray] = {}
    for candidate_ids, selected, prediction_mean in zip(
        state.history.blocks("decision_candidates", "candidate_ids"),
        state.history.blocks("decision_candidates", "selected"),
        state.history.blocks("decision_candidates", "prediction_mean"),
        strict=True,
    ):
        for candidate_id, is_selected, predicted in zip(
            candidate_ids.reshape(-1),
            selected.reshape(-1),
            prediction_mean,
            strict=True,
        ):
            if bool(is_selected):
                predictions_by_candidate[int(candidate_id)] = np.array(
                    predicted, dtype=np.float64, copy=True
                )

    evaluation = state.history.channel("evaluation")
    expected: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
    for request_id, sequence, candidate_ids, true_values in zip(
        evaluation["request_id"],
        evaluation["sequence"],
        state.history.blocks("evaluation", "candidate_ids"),
        state.history.blocks("evaluation", "f"),
        strict=True,
    ):
        if int(request_id) == -1:
            continue
        predicted_rows: list[np.ndarray] = []
        true_rows: list[np.ndarray] = []
        for candidate_id, true_value in zip(
            candidate_ids.reshape(-1), true_values, strict=True
        ):
            predicted = predictions_by_candidate.get(int(candidate_id))
            if predicted is None:
                continue
            predicted_rows.append(predicted)
            true_rows.append(np.array(true_value, dtype=np.float64, copy=True))

        shape = (len(predicted_rows), state.problem.n_obj)
        expected[(int(request_id), int(sequence))] = (
            np.asarray(predicted_rows, dtype=np.float64).reshape(shape),
            np.asarray(true_rows, dtype=np.float64).reshape(shape),
        )
    return expected


def _rewrite_npz(src: Path, dst: Path, **overrides: np.ndarray | None) -> None:
    """Copy an npz checkpoint while replacing or removing selected keys."""
    data = np.load(src, allow_pickle=False)
    rebuilt = dict(data.items())
    for key, value in overrides.items():
        if value is None:
            rebuilt.pop(key, None)
        else:
            rebuilt[key] = value
    np.savez(dst, **rebuilt)


def _empty_state(n_obj: int = 1) -> OptimizationState:
    problem = _problem(n_obj)
    attrs = [
        PopulationAttribute("x", np.float64, (2,), np.nan),
        PopulationAttribute("f", np.float64, (n_obj,), np.nan),
        PopulationAttribute("g", np.float64, (0,), 0.0),
        PopulationAttribute("cv", np.float64, (), 0.0),
        PopulationAttribute("id", np.int64, (), -1),
    ]
    return OptimizationState(
        problem=problem,
        population=Population(attrs, 1),
        archive=Archive(attrs, 1),
        pareto_archive=ParetoArchive(attrs, 1, direction=problem.direction),
        rng=np.random.default_rng(0),
        history=History(),
    )


def test_summary_records_initial_state_and_one_row_per_generation() -> None:
    result = minimize(_problem(), max_fe=13, pop_size=3, seed=7, verbose=False)
    assert result.history is not None
    summary = result.history.channel("summary")

    np.testing.assert_array_equal(summary["gen"][0], 0)
    np.testing.assert_array_equal(summary["fe"][0], 10)
    assert np.all(np.diff(summary["fe"]) >= 0)
    assert len(summary["gen"]) == result.gen + 1


@pytest.mark.parametrize("path", ["sync", "async"], ids=["sync", "async"])
def test_evaluation_history_records_all_post_initial_candidates(path: str) -> None:
    state = _run_history_path(path, ["evaluation"])

    assert state.history is not None
    candidate_id_blocks = state.history.blocks("evaluation", "candidate_ids")
    recorded_candidates = sum(block.shape[0] for block in candidate_id_blocks)

    assert recorded_candidates == state.fe
    if path == "async":
        assert recorded_candidates > 0


def test_evaluation_history_deduplicates_request_and_sequence() -> None:
    candidate_ids = np.array([101], dtype=np.int64)
    update = EvaluationUpdate(
        request_id=np.int64(17),
        status=EvaluationStatus.COMPLETED,
        candidate_ids=candidate_ids,
        result=EvaluationResult(
            f=np.array([[1.0]], dtype=np.float64),
            g=np.empty((1, 0), dtype=np.float64),
            cv=np.array([0.0], dtype=np.float64),
            candidate_ids=candidate_ids,
        ),
        sequence=3,
    )
    state = _empty_state().replace(history=History(["evaluation"]))
    _observe(state, update, 0)

    record_evaluations(state)
    record_evaluations(state)

    assert state.history is not None
    channel = state.history.channel("evaluation")
    assert len(channel["request_id"]) == 1
    assert list(zip(channel["request_id"], channel["sequence"], strict=True)) == [
        (17, 3)
    ]


def test_resumed_in_flight_evaluation_has_unknown_origin(tmp_path: Path) -> None:
    state = _empty_state().replace(
        history=History(["evaluation"]), decision_count=7, gen=3
    )
    state.population._extend_internal(
        {
            "x": np.zeros((1, 2)),
            "f": np.zeros((1, 1)),
            "g": np.zeros((1, 0)),
            "cv": np.zeros(1),
            "id": np.array([301], dtype=np.int64),
        },
        preserve_ids=True,
    )
    plan = EvaluateAll().plan(state.population, None, state)
    request = plan.requests[0]
    checkpoint_state = state.replace(
        evaluation_plan=plan,
        evaluation_plan_state=EvaluationPlanState(submitted=(int(request.request_id),)),
        evaluation_plan_updates={},
    )
    path = tmp_path / "in_flight_evaluation.npz"
    checkpoint_state.save(path)

    resumed = OptimizationState.load(path, state.problem)
    assert resumed.data["resumed"] is True
    candidate_ids = np.array(request.candidate_ids, dtype=np.int64, copy=True)
    update = EvaluationUpdate(
        request_id=request.request_id,
        status=EvaluationStatus.COMPLETED,
        candidate_ids=candidate_ids,
        result=EvaluationResult(
            f=np.zeros((1, 1), dtype=np.float64),
            g=np.empty((1, 0), dtype=np.float64),
            cv=np.zeros(1, dtype=np.float64),
            candidate_ids=candidate_ids,
        ),
        sequence=0,
    )
    _observe(resumed, update, 0)
    record_evaluations(resumed)

    assert resumed.history is not None
    channel = resumed.history.channel("evaluation")
    np.testing.assert_array_equal(channel["origin_decision_count"], [-1])


@pytest.mark.parametrize("path", ["sync", "async"], ids=["sync", "async"])
def test_evaluation_history_records_evaluator_costs(path: str) -> None:
    evaluator = _CostEvaluator()
    optimizer = (
        Optimizer(_problem(), seed=7)
        .set_initializer(LHSInitializer(10, 3))
        .set_evaluator(evaluator)
        .set_termination(Termination(max_fe(13)))
        .set_history(["evaluation"])
    )
    if path == "async":
        optimizer.set_async_evaluation_scheduler(
            AsyncEvaluationScheduler(evaluator, max_pending=1)
        )

    state = optimizer.run()
    assert state.history is not None
    recorded_cost = sum(
        float(np.sum(block)) for block in state.history.blocks("evaluation", "cost")
    )
    evaluator_cost = sum(float(np.sum(batch)) for batch in evaluator.returned_costs)

    assert evaluator.returned_costs
    assert recorded_cost == pytest.approx(evaluator_cost)
    assert np.all(
        np.isfinite(
            np.concatenate(
                [block.ravel() for block in state.history.blocks("evaluation", "cost")]
            )
        )
    )


@pytest.mark.parametrize(
    "initializer",
    [
        LHSInitializer(10, 3),
        GenomeInitializer(10, 3),
        RandomInitializer(10, 3),
        SobolInitializer(8, 3),
    ],
    ids=["lhs", "genome", "random", "sobol"],
)
def test_initializer_records_single_initial_evaluation_row(initializer) -> None:
    n_init = initializer.n_init_archive
    state = (
        Optimizer(_problem(), seed=7)
        .set_initializer(initializer)
        .set_termination(Termination(max_fe(n_init + 3)))
        .set_history(["evaluation"])
        .run()
    )
    assert state.history is not None
    channel = state.history.channel("evaluation")
    initial_rows = np.flatnonzero(channel["request_id"] == -1)
    assert len(initial_rows) == 1
    assert channel["fe_before"][initial_rows[0]] == 0
    assert channel["status_code"][initial_rows[0]] == 3
    assert channel["size"][initial_rows[0]] == n_init


def test_initial_evaluation_row_survives_checkpoint_roundtrip(tmp_path) -> None:
    state = (
        Optimizer(_problem(), seed=7)
        .set_initializer(LHSInitializer(10, 3))
        .set_termination(Termination(max_fe(13)))
        .set_history(["evaluation"])
        .run()
    )
    assert state.history is not None
    path = tmp_path / "initial_eval.npz"
    state.save(path)
    restored = OptimizationState.load(path, state.problem)
    assert restored.history is not None

    expected = state.history.channel("evaluation")
    actual = restored.history.channel("evaluation")
    assert list(actual) == list(expected)
    for name in expected:
        np.testing.assert_array_equal(actual[name], expected[name])

    initial_rows = np.flatnonzero(actual["request_id"] == -1)
    assert len(initial_rows) == 1
    restored_blocks = restored.history.blocks("evaluation", "f")
    np.testing.assert_array_equal(
        restored_blocks[initial_rows[0]],
        state.history.blocks("evaluation", "f")[initial_rows[0]],
    )
    assert restored_blocks[initial_rows[0]].shape[0] == 10


@pytest.mark.parametrize("path", ["sync", "async"], ids=["sync", "async"])
def test_evaluation_history_origin_decision_matches_selected_candidates(
    path: str,
) -> None:
    state = _run_history_path(path, ["decision_candidates", "evaluation"])

    assert state.history is not None
    decision_channel = state.history.channel("decision_candidates")
    origin_by_candidate: dict[int, int] = {}
    for decision_count, candidate_ids, selected in zip(
        decision_channel["decision_count"],
        state.history.blocks("decision_candidates", "candidate_ids"),
        state.history.blocks("decision_candidates", "selected"),
        strict=True,
    ):
        for candidate_id, is_selected in zip(candidate_ids[:, 0], selected[:, 0]):
            if is_selected:
                origin_by_candidate[int(candidate_id)] = int(decision_count)

    evaluation = state.history.channel("evaluation")
    for request_id, origin_decision_count, candidate_ids in zip(
        evaluation["request_id"],
        evaluation["origin_decision_count"],
        state.history.blocks("evaluation", "candidate_ids"),
        strict=True,
    ):
        if int(request_id) == -1:
            continue
        for candidate_id in candidate_ids[:, 0]:
            assert origin_by_candidate[int(candidate_id)] == int(origin_decision_count)


def test_failed_evaluator_keeps_nan_evaluation_record() -> None:
    state = _empty_state().replace(history=History(["evaluation"]))
    request = EvaluationRequest(
        request_id=np.int64(23),
        candidate_ids=np.array([201, 202], dtype=np.int64),
        payload=np.zeros((2, 2), dtype=np.float64),
    )
    evaluator = _FailingEvaluator()
    handle = evaluator.submit(request, state.problem)
    updates = evaluator.collect(handle)

    assert len(updates) == 1
    assert updates[0].status is EvaluationStatus.FAILED
    _observe(state, updates[0], 0)
    record_evaluations(state)

    assert state.history is not None
    channel = state.history.channel("evaluation")
    assert channel["status_code"].tolist() == [4]
    assert channel["size"].tolist() == [2]
    for column in ("f", "cv", "cost"):
        blocks = state.history.blocks("evaluation", column)
        assert len(blocks) == 1
        assert blocks[0].shape[0] == 2
        assert np.all(np.isnan(blocks[0]))


def test_evaluation_history_checkpoint_roundtrip_preserves_all_columns(
    tmp_path: Path,
) -> None:
    state = _run_history_path("sync", ["evaluation"])
    path = tmp_path / "evaluation.npz"
    state.save(path)
    restored = OptimizationState.load(path, state.problem)

    assert state.history is not None
    assert restored.history is not None
    expected_channel = state.history.channel("evaluation")
    actual_channel = restored.history.channel("evaluation")
    assert list(actual_channel) == list(expected_channel)
    for name, expected in expected_channel.items():
        actual = actual_channel[name]
        np.testing.assert_array_equal(actual, expected)
        assert actual.dtype in {np.dtype(np.int64), np.dtype(np.float64)}
        assert actual.dtype == expected.dtype

    for column in ("candidate_ids", "f", "cv", "cost"):
        expected_blocks = state.history.blocks("evaluation", column)
        actual_blocks = restored.history.blocks("evaluation", column)
        assert len(actual_blocks) == len(expected_blocks)
        for actual, expected in zip(actual_blocks, expected_blocks, strict=True):
            np.testing.assert_array_equal(actual, expected)
            assert actual.dtype in {np.dtype(np.int64), np.dtype(np.float64)}
            assert actual.dtype == expected.dtype


@pytest.mark.parametrize("path", ["sync", "async"], ids=["sync", "async"])
def test_surrogate_accuracy_matches_independent_evaluation_join(path: str) -> None:
    state = _run_history_path(
        path, ["decision_candidates", "evaluation", "surrogate_accuracy"]
    )

    assert state.history is not None
    expected = _expected_surrogate_accuracy_pairs(state)
    evaluation = state.history.channel("evaluation")
    accuracy = state.history.channel("surrogate_accuracy")
    evaluation_keys = [
        (int(request_id), int(sequence))
        for request_id, sequence in zip(
            evaluation["request_id"], evaluation["sequence"], strict=True
        )
        if int(request_id) != -1
    ]
    accuracy_keys = [
        (int(request_id), int(sequence))
        for request_id, sequence in zip(
            accuracy["request_id"], accuracy["sequence"], strict=True
        )
    ]

    assert len(evaluation_keys) == len(set(evaluation_keys))
    assert len(accuracy_keys) == len(set(accuracy_keys))
    assert set(accuracy_keys) == set(evaluation_keys)
    assert set(expected) == set(evaluation_keys)

    for column in (
        "request_id",
        "sequence",
        "gen",
        "fe_before",
        "fe_after",
        "decision_count",
        "size",
    ):
        assert accuracy[column].dtype == np.dtype(np.int64)
    for column in ("predicted", "true"):
        for block in state.history.blocks("surrogate_accuracy", column):
            assert block.dtype == np.dtype(np.float64)

    actual_predicted = dict(
        zip(
            accuracy_keys,
            state.history.blocks("surrogate_accuracy", "predicted"),
            strict=True,
        )
    )
    actual_true = dict(
        zip(
            accuracy_keys,
            state.history.blocks("surrogate_accuracy", "true"),
            strict=True,
        )
    )
    expected_pair_count = sum(len(predicted) for predicted, _ in expected.values())
    actual_pair_count = int(np.sum(accuracy["size"]))
    assert actual_pair_count == expected_pair_count
    assert actual_pair_count > 0
    if path == "async":
        assert actual_pair_count != 0

    for key, (predicted, true) in expected.items():
        np.testing.assert_array_equal(actual_predicted[key], predicted)
        np.testing.assert_array_equal(actual_true[key], true)

    assert state.history._surrogate_predictions == {}


@pytest.mark.parametrize(
    "path", ["sync", "async", "structured"], ids=["sync", "async", "structured"]
)
def test_all_history_channels_preserve_trajectory(path: str) -> None:
    recorded = _run_history_path(path, _ALL_HISTORY_CHANNELS)
    disabled = _run_history_path(path, [])

    assert len(recorded.archive) == len(disabled.archive)
    for column in ("x", "f", "g", "cv", "id"):
        np.testing.assert_array_equal(
            recorded.archive.get_array(column), disabled.archive.get_array(column)
        )
    assert recorded.history is not None
    assert recorded.history._surrogate_predictions == {}


def test_history_does_not_change_the_optimization_trajectory() -> None:
    recorded = _run_optimizer(_problem(), ["summary"])
    disabled = _run_optimizer(_problem(), [])

    np.testing.assert_array_equal(recorded.archive.x, disabled.archive.x)
    np.testing.assert_array_equal(recorded.archive.f, disabled.archive.f)


def test_front_history_does_not_change_the_optimization_trajectory() -> None:
    recorded = _run_optimizer(_problem(2), ["summary", "front"])
    disabled = _run_optimizer(_problem(2), [])

    np.testing.assert_array_equal(recorded.archive.x, disabled.archive.x)
    np.testing.assert_array_equal(recorded.archive.f, disabled.archive.f)


def test_history_views_are_read_only_and_column_dtypes_are_fixed() -> None:
    history = History()
    history.append(
        "summary",
        gen=0,
        fe=4,
        decision_count=1,
        front_size=2,
        f_min_0=1.0,
        f_max_0=2.0,
    )
    history.append(
        "summary",
        gen=1,
        fe=5,
        decision_count=2,
        front_size=2,
        f_min_0=0.5,
        f_max_0=1.5,
    )
    summary = history.channel("summary")

    for name, values in summary.items():
        assert not values.flags.writeable
        expected = (
            np.int64
            if name in {"gen", "fe", "decision_count", "front_size"}
            else np.float64
        )
        assert values.dtype == expected
    with pytest.raises(ValidationError):
        history.append("summary", gen=2)


def test_replace_shares_history_instance() -> None:
    state = _run_optimizer(_problem(), ["summary"])
    assert state.replace(gen=state.gen + 1).history is state.history


def test_empty_archive_and_population_record_nan_summary_values() -> None:
    state = _empty_state()
    record_generation(state)
    assert state.history is not None
    summary = state.history.channel("summary")

    assert summary["front_size"][0] == 0
    assert np.isnan(summary["f_min_0"][0])
    assert np.isnan(summary["f_max_0"][0])
    assert np.isnan(summary["best_f"][0])
    assert np.isnan(summary["feasible_ratio"][0])
    assert np.isnan(summary["min_cv"][0])


def test_feasible_ratio_uses_constraint_handler_threshold() -> None:
    state = _empty_state()
    state.population.extend(
        {
            "x": np.zeros((4, 2)),
            "f": np.zeros((4, 1)),
            "g": np.zeros((4, 0)),
            "cv": np.array([0.0, 5e-7, 1e-7, 1.0]),
            "id": np.array([-1, -1, -1, -1]),
        }
    )
    record_generation(state)
    assert state.history is not None
    summary = state.history.channel("summary")
    threshold = state.problem.handler.feasibility_threshold
    cv = np.array([0.0, 5e-7, 1e-7, 1.0])
    expected = float(np.count_nonzero(cv <= threshold) / len(cv))
    assert summary["feasible_ratio"][0] == pytest.approx(expected)
    assert summary["feasible_ratio"][0] == pytest.approx(0.75)


def test_epsilon_summary_channels_keep_reporting_and_diagnostics_distinct() -> None:
    n_gen = 8

    def make_optimizer(handler: ConstraintHandler) -> Optimizer:
        problem = Problem(
            func=lambda x: float(x[0]),
            dim=1,
            n_obj=1,
            direction=np.array([-1.0]),
            lb=[0.0],
            ub=[1.0],
            eps_cv=1e-6,
            constraints=[InequalityConstraint(lambda x: 0.5 - x[0])],
            handler=handler,
        )
        return (
            Optimizer(problem, seed=0)
            .set_initializer(
                LHSInitializer(n_init_archive=2, n_init_population=2, seed=0)
            )
            .set_algorithm(
                GA(
                    crossover=CrossoverBLXAlpha(prob=0.9, alpha=0.4),
                    mutation=MutationUniform(prob_var=0.1),
                    parent_selection=SequentialSelection(),
                    survivor_selection=TruncationSelection(),
                )
            )
            .set_strategy(IndividualBasedStrategy(evaluation_ratio=1.0))
            .set_termination(Termination(max_gen(n_gen)))
            .set_history(["summary"])
        )

    calibration = make_optimizer(
        EpsilonConstraintHandler(linear_epsilon_schedule(eps0=0.5, n_gen=n_gen))
    )
    calibration_state = calibration.run()
    measured_cv_max = float(np.max(calibration_state.archive.get_array("cv")))
    assert 0.0 < measured_cv_max <= 0.25
    eps0 = measured_cv_max

    problem = Problem(
        func=lambda x: float(x[0]),
        dim=1,
        n_obj=1,
        direction=np.array([-1.0]),
        lb=[0.0],
        ub=[1.0],
        eps_cv=1e-6,
        constraints=[InequalityConstraint(lambda x: 0.5 - x[0])],
        handler=EpsilonConstraintHandler(
            linear_epsilon_schedule(eps0=eps0, n_gen=n_gen)
        ),
    )
    optimizer = make_optimizer(problem.handler)
    archive_snapshots: dict[int, tuple[np.ndarray, np.ndarray, float]] = {}
    population_snapshots: dict[int, np.ndarray] = {}

    def capture_archive(event: GenerationEndEvent) -> None:
        archive_snapshots[event.ctx.gen] = (
            event.ctx.archive.get_array("f").copy(),
            event.ctx.archive.get_array("cv").copy(),
            float(event.ctx.problem.handler.feasibility_threshold),
        )
        population_snapshots[event.ctx.gen] = event.ctx.population.cv.copy()

    optimizer.cbmanager.register(GenerationEndEvent, capture_archive)
    state = optimizer.run()

    assert state.history is not None
    summary = state.history.channel("summary")
    best = summary["best_f"]
    feasible_ratio = summary["feasible_ratio"]
    min_cv = summary["min_cv"]

    score = best * problem.direction[0]
    assert np.all(np.diff(score) >= 0.0)
    assert np.ptp(best) > 0.0
    assert any(
        np.any(
            (cv > problem.eps_cv)
            & (cv <= threshold)
            & (f[:, 0] < best[int(np.flatnonzero(summary["gen"] == gen)[0])])
        )
        for gen, (f, cv, threshold) in archive_snapshots.items()
    )
    assert feasible_ratio[0] > feasible_ratio[-1]
    assert set(archive_snapshots) == set(summary["gen"][1:])
    for gen, cv in population_snapshots.items():
        row = int(np.flatnonzero(summary["gen"] == gen)[0])
        assert min_cv[row] == pytest.approx(float(np.min(cv)))


def test_multiobjective_summary_has_each_objective_column() -> None:
    result = minimize(_problem(2), max_fe=13, pop_size=3, seed=7, verbose=False)
    assert result.history is not None
    summary = result.history.channel("summary")

    for index in range(2):
        assert f"f_min_{index}" in summary
        assert f"f_max_{index}" in summary
    assert np.all(np.isnan(summary["best_f"]))


@pytest.mark.parametrize("run", [minimize, maximize], ids=["minimize", "maximize"])
def test_singleobjective_summary_best_f_matches_result(run) -> None:
    result = run(_problem(), max_fe=13, pop_size=3, seed=7, verbose=False)
    assert result.history is not None
    best_f = result.history.channel("summary")["best_f"]

    assert best_f.dtype == np.float64
    assert best_f[-1] == pytest.approx(float(result.f[0]))


def test_summary_best_f_uses_archive_selection_for_equal_infeasible_cv() -> None:
    state = _empty_state()
    state.archive.extend(
        {
            "x": np.zeros((2, 2)),
            "f": np.array([[1.0], [3.0]]),
            "g": np.zeros((2, 0)),
            "cv": np.array([1.0, 1.0]),
            "id": np.array([-1, -1]),
        }
    )
    state.pareto_archive.extend(
        {
            "x": np.zeros((1, 2)),
            "f": np.array([[2.0]]),
            "g": np.zeros((1, 0)),
            "cv": np.array([1.0]),
            "id": np.array([-1]),
        }
    )

    record_generation(state)

    assert state.history is not None
    summary = state.history.channel("summary")
    result = Result.from_state(state)
    assert summary["best_f"][0] == pytest.approx(float(result.f[0]))
    assert summary["best_f"][0] not in {
        summary["f_min_0"][0],
        summary["f_max_0"][0],
    }


def test_set_history_rejects_unknown_channel() -> None:
    with pytest.raises(ValidationError):
        Optimizer(_problem()).set_history(["unknown"])


def test_checkpoint_roundtrip_restores_summary_history(tmp_path: Path) -> None:
    state = _run_optimizer(_problem(), ["summary"])
    path = tmp_path / "history.npz"
    state.save(path)

    restored = OptimizationState.load(path, state.problem)
    assert state.history is not None
    assert restored.history is not None
    assert restored.history.enabled == state.history.enabled

    expected = state.history.channel("summary")
    actual = restored.history.channel("summary")
    assert list(actual) == list(expected)
    for name, values in expected.items():
        np.testing.assert_array_equal(actual[name], values)
        expected_dtype = (
            np.int64
            if name in {"gen", "fe", "decision_count", "front_size"}
            else np.float64
        )
        assert actual[name].dtype == expected_dtype

    raw = np.load(path, allow_pickle=False)
    metadata = json.loads(bytes(raw["_history_meta"]).decode())
    assert metadata == {
        "channels": ["summary"],
        "columns": {"summary": list(expected)},
        "blocks": {"summary": []},
    }


def test_current_summary_checkpoint_roundtrip_retains_best_f(tmp_path: Path) -> None:
    state = _run_optimizer(_problem(), ["summary"])
    assert state.history is not None
    path = tmp_path / "roundtrip.npz"
    state.save(path)
    restored = OptimizationState.load(path, state.problem)

    assert restored.history is not None
    np.testing.assert_array_equal(
        restored.history.channel("summary")["best_f"],
        state.history.channel("summary")["best_f"],
    )


def _resume_optimizer(
    state: OptimizationState, channels: list[str], max_fe_limit: int = 16
) -> Optimizer:
    optimizer = (
        Optimizer(state.problem, seed=7)
        .set_initializer(LHSInitializer(10, 3))
        .set_termination(Termination(max_fe(max_fe_limit)))
        .set_history(channels)
    )
    optimizer.resolve_defaults()
    return optimizer


def test_resume_preserves_and_appends_history_records() -> None:
    state = _run_optimizer(_problem(), ["summary", "front"])
    assert state.history is not None
    summary_before = {
        name: np.array(values, copy=True)
        for name, values in state.history.channel("summary").items()
    }
    front_before = [
        np.array(block, copy=True) for block in state.history.blocks("front", "f")
    ]
    summary_rows_before = len(summary_before["gen"])

    resumed = _resume_optimizer(state, ["summary", "front"]).run_from(state)

    assert resumed.history is not None
    summary_after = resumed.history.channel("summary")
    assert len(summary_after["gen"]) > summary_rows_before
    for name, values in summary_before.items():
        np.testing.assert_array_equal(summary_after[name][:summary_rows_before], values)
    front_after = resumed.history.blocks("front", "f")
    assert len(front_after) > len(front_before)
    for before, after in zip(front_before, front_after, strict=False):
        np.testing.assert_array_equal(after, before)


def test_resume_rejects_different_history_channels() -> None:
    state = _run_optimizer(_problem(), ["summary", "front"])

    with pytest.raises(ValidationError) as exc_info:
        _resume_optimizer(state, ["summary"]).run_from(state)

    message = str(exc_info.value)
    assert "requested channels=['summary']" in message
    assert "checkpoint channels=['front', 'summary']" in message
    assert "set_history()" in message
    assert "new run" in message


def test_resume_history_channels_ignore_order_and_duplicates() -> None:
    state = _run_optimizer(_problem(), ["summary", "front"])

    resumed = _resume_optimizer(state, ["front", "summary", "summary"]).run_from(state)

    assert resumed.history is not None
    assert resumed.history.enabled == frozenset({"summary", "front"})


def test_resume_without_explicit_history_adopts_checkpoint_channels() -> None:
    state = _run_optimizer(_problem(), ["summary", "front"])
    optimizer = (
        Optimizer(state.problem, seed=7)
        .set_initializer(LHSInitializer(10, 3))
        .set_termination(Termination(max_fe(16)))
    )
    optimizer.resolve_defaults()
    resumed = optimizer.run_from(state)
    assert resumed.history is not None
    assert resumed.history.enabled == frozenset({"summary", "front"})


def test_resume_without_explicit_history_keeps_disabled_history() -> None:
    state = _run_optimizer(_problem(), [])
    optimizer = (
        Optimizer(state.problem, seed=7)
        .set_initializer(LHSInitializer(10, 3))
        .set_termination(Termination(max_fe(16)))
    )
    optimizer.resolve_defaults()
    resumed = optimizer.run_from(state)
    assert resumed.history is not None
    assert resumed.history.enabled == frozenset()


def test_front_records_ragged_snapshots_with_summary_rows() -> None:
    state = _run_optimizer(_problem(2), ["summary", "front"])
    assert state.history is not None
    summary = state.history.channel("summary")
    front = state.history.channel("front")
    blocks = state.history.blocks("front", "f")

    assert len(front["gen"]) == len(summary["gen"]) == len(blocks)
    for index, block in enumerate(blocks):
        assert block.shape == (summary["front_size"][index], 2)
        assert not block.flags.writeable


def test_empty_front_is_recorded_with_objective_width() -> None:
    state = _empty_state(2)
    state.history = History(["summary", "front"])
    record_generation(state)

    assert state.history is not None
    blocks = state.history.blocks("front", "f")
    assert len(blocks) == 1
    assert blocks[0].shape == (0, 2)


def test_front_block_is_a_snapshot() -> None:
    state = _empty_state(2)
    state.history = History(["front"])
    state.pareto_archive.add(
        x=np.array([0.0, 0.0]),
        f=np.array([1.0, 2.0]),
        g=np.empty(0),
        cv=0.0,
        id=1,
    )
    record_generation(state)
    first = state.history.blocks("front", "f")[0].copy()

    state.pareto_archive.add(
        x=np.array([1.0, 1.0]),
        f=np.array([0.5, 2.5]),
        g=np.empty(0),
        cv=0.0,
        id=2,
    )
    record_generation(state)

    np.testing.assert_array_equal(state.history.blocks("front", "f")[0], first)


def test_typed_blocks_preserve_dtype_and_checkpoint_roundtrip(
    tmp_path: Path,
) -> None:
    history = History(["front"])
    history.append_block(
        "front",
        {
            "float_values": np.array([[1.5]], dtype=np.float32),
            "integer_values": np.array([[2]], dtype=np.int32),
            "boolean_values": np.array([[True]], dtype=bool),
        },
    )
    history.append_block(
        "front",
        {
            "float_values": np.array([[2.5]], dtype=np.float64),
            "integer_values": np.array([[7]], dtype=np.int64),
            "boolean_values": np.array([[False]], dtype=bool),
        },
    )

    assert history.blocks("front", "float_values")[0].dtype == np.float64
    assert history.blocks("front", "integer_values")[0].dtype == np.int64
    assert history.blocks("front", "boolean_values")[0].dtype == bool

    state = _empty_state().replace(history=history)
    path = tmp_path / "typed_blocks.npz"
    state.save(path)
    restored = OptimizationState.load(path, state.problem)

    assert state.history is not None
    assert restored.history is not None
    for column in ("float_values", "integer_values", "boolean_values"):
        expected = state.history.blocks("front", column)
        actual = restored.history.blocks("front", column)
        for expected_block, actual_block in zip(expected, actual):
            np.testing.assert_array_equal(actual_block, expected_block)
            assert actual_block.dtype == expected_block.dtype


def test_int64_block_above_float_precision_roundtrips_exactly(tmp_path: Path) -> None:
    value = 2**53 + 1
    history = History(["front"])
    history.append_block("front", {"f": np.array([[value]], dtype=np.int64)})
    state = _empty_state().replace(history=history)
    path = tmp_path / "int64_block.npz"
    state.save(path)
    restored = OptimizationState.load(path, state.problem)

    assert restored.history is not None
    block = restored.history.blocks("front", "f")[0]
    assert block.dtype == np.int64
    assert int(block[0, 0]) == value


def test_block_dtype_categories_are_fixed_and_object_arrays_rejected() -> None:
    history = History(["front"])
    history.append_block("front", {"values": np.array([[1]], dtype=np.int32)})
    history.append_block("front", {"values": np.array([[2]], dtype=np.uint64)})

    assert all(block.dtype == np.int64 for block in history.blocks("front", "values"))
    with pytest.raises(ValidationError):
        history.append_block("front", {"values": np.array([[1.0]], dtype=np.float64)})

    object_history = History(["front"])
    with pytest.raises(ValidationError):
        object_history.append_block("front", {"values": np.array([[1]], dtype=object)})


def test_population_records_ragged_snapshots_with_summary_rows() -> None:
    state = _run_optimizer(_problem(2), ["summary", "population"])
    assert state.history is not None
    summary = state.history.channel("summary")
    population = state.history.channel("population")
    f_blocks = state.history.blocks("population", "f")
    x_blocks = state.history.blocks("population", "x")

    assert len(population["gen"]) == len(summary["gen"])
    assert len(f_blocks) == len(x_blocks) == len(population["gen"])
    assert population["size"].dtype == np.int64
    for index, size in enumerate(population["size"]):
        assert f_blocks[index].shape == (size, 2)
        assert x_blocks[index].shape == (size, 2)


def test_population_without_dense_view_records_only_objectives() -> None:
    space = ObjectSpace(RepresentationSpec(kind="permutation"))
    problem = Problem(
        func=lambda _: np.array([0.0]),
        dim=2,
        n_obj=1,
        direction=np.array([-1.0]),
        space=space,
    )
    state = _empty_state().replace(problem=problem, history=History(["population"]))
    state.population.extend(
        {
            "x": np.array([[0.0, 1.0]]),
            "f": np.array([[2.0]]),
            "g": np.zeros((1, 0)),
            "cv": np.zeros(1),
        }
    )

    record_generation(state)

    assert state.history is not None
    assert state.history.blocks("population", "f")[0].shape == (1, 1)
    with pytest.raises(ValidationError):
        state.history.blocks("population", "x")


def test_population_block_is_a_snapshot() -> None:
    state = _empty_state()
    state.history = History(["population"])
    state.population.extend(
        {
            "x": np.array([[0.0, 1.0]]),
            "f": np.array([[2.0]]),
            "g": np.zeros((1, 0)),
            "cv": np.zeros(1),
        }
    )
    record_generation(state)
    assert state.history is not None
    first_f = state.history.blocks("population", "f")[0].copy()
    first_x = state.history.blocks("population", "x")[0].copy()

    state.population.update_rows(np.array([0]), {"f": np.array([[8.0]])})
    state.population.update_array("x", np.array([[1.0, 2.0]]))

    np.testing.assert_array_equal(state.history.blocks("population", "f")[0], first_f)
    np.testing.assert_array_equal(state.history.blocks("population", "x")[0], first_x)


def test_surrogate_accuracy_pairs_by_id_and_skips_unknown_ids() -> None:
    state = _empty_state()
    state.population._extend_internal(
        {
            "x": np.zeros((3, 2)),
            "f": np.zeros((3, 1)),
            "g": np.zeros((3, 0)),
            "cv": np.zeros(3),
            "id": np.array([10, 11, 12]),
        },
        preserve_ids=True,
    )
    offspring = state.population.extract([0, 1, 2])
    prediction_values = np.array([[1.0], [2.0], [3.0]])
    plan = EvaluateAll().plan(offspring.extract([0, 1]), None, state)
    state = state.replace(
        history=History(["surrogate_accuracy"]),
        offspring=offspring,
        evaluation_plan=plan,
        evaluation_plan_state=EvaluationPlanState(
            deferred=tuple(int(request.request_id) for request in plan.requests)
        ),
        predictions=SurrogatePrediction(
            {"objective": PredictionChannel(prediction_values)}
        ),
    )

    record_decision(state)
    candidate_ids = np.array([12, 10, 11], dtype=np.int64)
    update = EvaluationUpdate(
        request_id=plan.requests[0].request_id,
        status=EvaluationStatus.COMPLETED,
        candidate_ids=candidate_ids,
        result=EvaluationResult(
            f=np.array([[30.0], [10.0], [20.0]]),
            g=np.empty((3, 0)),
            cv=np.zeros(3),
            candidate_ids=candidate_ids,
        ),
    )
    _observe(state, update, 0)
    record_evaluations(state)
    record_evaluations(state)

    assert state.history is not None
    accuracy = state.history.channel("surrogate_accuracy")
    predicted = state.history.blocks("surrogate_accuracy", "predicted")
    true = state.history.blocks("surrogate_accuracy", "true")
    np.testing.assert_array_equal(predicted[0], [[1.0], [2.0]])
    np.testing.assert_array_equal(true[0], [[10.0], [20.0]])
    np.testing.assert_array_equal(accuracy["size"], [2])
    assert state.history._surrogate_predictions == {}


def test_repeated_evaluation_records_surrogate_accuracy_for_all_replicates() -> None:
    state = (
        Optimizer(_problem(), seed=7)
        .set_initializer(LHSInitializer(10, 3))
        .set_termination(Termination(max_fe(30)))
        .set_evaluation_planner(RepeatedEvaluation(2))
        .set_history(["summary", "surrogate_accuracy"])
        .run()
    )
    assert state.history is not None
    accuracy = state.history.channel("surrogate_accuracy")
    sizes_by_decision: dict[int, list[int]] = {}
    for decision_count, size in zip(
        accuracy["decision_count"], accuracy["size"], strict=True
    ):
        sizes_by_decision.setdefault(int(decision_count), []).append(int(size))

    assert sizes_by_decision
    for sizes in sizes_by_decision.values():
        assert len(sizes) == 2
        assert sizes[0] == sizes[1] > 0


def test_surrogate_accuracy_without_surrogate_records_empty_blocks() -> None:
    state = (
        Optimizer(_problem(), seed=7)
        .set_initializer(LHSInitializer(10, 3))
        .set_strategy(DirectStrategy())
        .set_termination(Termination(max_fe(13)))
        .set_history(["surrogate_accuracy"])
        .run()
    )
    assert state.history is not None
    accuracy = state.history.channel("surrogate_accuracy")
    predicted = state.history.blocks("surrogate_accuracy", "predicted")
    true = state.history.blocks("surrogate_accuracy", "true")

    assert len(predicted) == len(true) == len(accuracy["size"])
    assert len(predicted) > 0
    assert np.all(accuracy["size"] == 0)
    assert all(block.shape == (0, 1) for block in predicted)
    assert all(block.shape == (0, 1) for block in true)
    assert all(block.dtype == np.dtype(np.float64) for block in predicted + true)
    assert state.history._surrogate_predictions == {}


def test_record_generation_does_not_record_surrogate_accuracy() -> None:
    state = _empty_state().replace(history=History(["surrogate_accuracy"]))

    record_generation(state)

    assert state.history is not None
    assert state.history.channel("surrogate_accuracy") == {}
    with pytest.raises(ValidationError):
        state.history.blocks("surrogate_accuracy", "predicted")
    with pytest.raises(ValidationError):
        state.history.blocks("surrogate_accuracy", "true")


def test_population_and_surrogate_accuracy_checkpoint_roundtrip(
    tmp_path: Path,
) -> None:
    state = _run_optimizer(
        _problem(2), ["summary", "front", "population", "surrogate_accuracy"]
    )
    path = tmp_path / "population_accuracy.npz"
    state.save(path)
    restored = OptimizationState.load(path, state.problem)

    assert state.history is not None
    assert restored.history is not None
    for channel, columns in {
        "population": ("f", "x"),
        "surrogate_accuracy": ("predicted", "true"),
    }.items():
        expected_channel = state.history.channel(channel)
        actual_channel = restored.history.channel(channel)
        assert list(actual_channel) == list(expected_channel)
        for name, expected_values in expected_channel.items():
            actual_values = actual_channel[name]
            np.testing.assert_array_equal(actual_values, expected_values)
            assert actual_values.dtype == expected_values.dtype

        for column in columns:
            expected = state.history.blocks(channel, column)
            actual = restored.history.blocks(channel, column)
            assert len(actual) == len(expected)
            for expected_block, actual_block in zip(expected, actual, strict=True):
                np.testing.assert_array_equal(actual_block, expected_block)
                assert actual_block.dtype == expected_block.dtype


def test_evaluation_history_records_distinct_status_for_same_request_sequence() -> None:
    candidate_ids = np.array([101], dtype=np.int64)

    def make_update(status: EvaluationStatus) -> EvaluationUpdate:
        return EvaluationUpdate(
            request_id=np.int64(17),
            status=status,
            candidate_ids=candidate_ids,
            result=(
                None
                if status is EvaluationStatus.FAILED
                else EvaluationResult(
                    f=np.array([[1.0]], dtype=np.float64),
                    g=np.empty((1, 0), dtype=np.float64),
                    cv=np.array([0.0], dtype=np.float64),
                    candidate_ids=candidate_ids,
                )
            ),
            sequence=0,
        )

    state = _empty_state().replace(history=History(["evaluation"]))
    _observe(state, make_update(EvaluationStatus.FAILED), 0)
    _observe(state, make_update(EvaluationStatus.COMPLETED), 0)
    record_evaluations(state)

    assert state.history is not None
    channel = state.history.channel("evaluation")
    assert len(channel["request_id"]) == 2
    recorded = set(
        zip(
            channel["request_id"],
            channel["sequence"],
            channel["status_code"],
            strict=True,
        )
    )
    assert recorded == {(17, 0, 4), (17, 0, 3)}


def test_surrogate_accuracy_records_distinct_status_for_same_request_sequence() -> None:
    state = _empty_state()
    state.population._extend_internal(
        {
            "x": np.zeros((3, 2)),
            "f": np.zeros((3, 1)),
            "g": np.zeros((3, 0)),
            "cv": np.zeros(3),
            "id": np.array([10, 11, 12]),
        },
        preserve_ids=True,
    )
    offspring = state.population.extract([0, 1, 2])
    prediction_values = np.array([[1.0], [2.0], [3.0]])
    plan = EvaluateAll().plan(offspring.extract([0, 1]), None, state)
    request_id = int(plan.requests[0].request_id)
    state = state.replace(
        history=History(["surrogate_accuracy"]),
        offspring=offspring,
        evaluation_plan=plan,
        evaluation_plan_state=EvaluationPlanState(deferred=(request_id,)),
        predictions=SurrogatePrediction(
            {"objective": PredictionChannel(prediction_values)}
        ),
    )
    record_decision(state)

    candidate_ids = np.array([12, 10, 11], dtype=np.int64)

    def make_update(status: EvaluationStatus) -> EvaluationUpdate:
        return EvaluationUpdate(
            request_id=np.int64(request_id),
            status=status,
            candidate_ids=candidate_ids,
            result=(
                None
                if status is EvaluationStatus.FAILED
                else EvaluationResult(
                    f=np.array([[30.0], [10.0], [20.0]]),
                    g=np.empty((3, 0)),
                    cv=np.zeros(3),
                    candidate_ids=candidate_ids,
                )
            ),
            sequence=0,
        )

    _observe(state, make_update(EvaluationStatus.FAILED), 0)
    record_evaluations(state)
    _observe(state, make_update(EvaluationStatus.COMPLETED), 0)
    record_evaluations(state)

    assert state.history is not None
    accuracy = state.history.channel("surrogate_accuracy")
    assert len(accuracy["request_id"]) == 2
    recorded = set(
        zip(
            accuracy["request_id"],
            accuracy["sequence"],
            accuracy["status_code"],
            strict=True,
        )
    )
    assert recorded == {(request_id, 0, 4), (request_id, 0, 3)}


def test_surrogate_accuracy_status_code_roundtrip(tmp_path: Path) -> None:
    state = _run_optimizer(
        _problem(2), ["summary", "front", "population", "surrogate_accuracy"]
    )
    path = tmp_path / "accuracy.npz"
    state.save(path)
    restored = OptimizationState.load(path, state.problem)
    assert state.history is not None
    assert restored.history is not None
    original = state.history.channel("surrogate_accuracy")
    roundtripped = restored.history.channel("surrogate_accuracy")
    assert "status_code" in roundtripped
    np.testing.assert_array_equal(roundtripped["status_code"], original["status_code"])


def test_surrogate_accuracy_dedup_survives_checkpoint_without_status_code(
    tmp_path: Path,
) -> None:
    state = _run_optimizer(
        _problem(2), ["summary", "front", "population", "surrogate_accuracy"]
    )
    path = tmp_path / "accuracy.npz"
    state.save(path)
    restored = OptimizationState.load(path, state.problem)
    assert restored.history is not None
    # Simulate an older checkpoint whose surrogate_accuracy channel lacks status_code.
    restored.history._columns["surrogate_accuracy"].pop("status_code", None)
    record_evaluations(
        restored.replace(evaluation_updates=[], evaluation_plan_updates={})
    )
    keys = restored.history._surrogate_accuracy_keys
    assert keys
    assert all(k[2] == -1 for k in keys)


class _PartialRetryEvaluator(Evaluator):
    """Deliver a PARTIAL then fail on attempt 0, complete on retry.

    Mirrors the evaluator used by the async workflow integration tests so the
    PARTIAL-on-first-attempt behavior can be exercised through a full run.
    """

    def __init__(self) -> None:
        self.attempts = 0
        self.acks: list[tuple[int, int]] = []
        self.collected: list[tuple[int, list[int]]] = []

    def evaluate_batch(self, x: Any, problem: Problem) -> EvaluationResult:
        return SerialEvaluator().evaluate_batch(x, problem)

    def submit(self, request: Any, problem: Problem) -> EvaluationHandle:
        handle = EvaluationHandle(
            request.request_id,
            EvaluationStatus.PENDING,
            backend_token=(request, problem, self.attempts),
        )
        self.attempts += 1
        return handle

    def collect(
        self, handle: EvaluationHandle, *, wait: bool = True
    ) -> list[EvaluationUpdate]:
        request, problem, attempt = handle.backend_token
        self.collected.append((attempt, list(request.candidate_ids)))
        if handle._acknowledged_sequence >= 0:
            return []
        if attempt == 0:
            first = SerialEvaluator().evaluate_batch(request.x[:1], problem)
            first.candidate_ids = request.candidate_ids[:1]
            first.__post_init__()
            handle._delivered_sequence = 1
            return [
                EvaluationUpdate(
                    request.request_id,
                    EvaluationStatus.PARTIAL,
                    request.candidate_ids[:1],
                    first,
                    sequence=0,
                ),
                EvaluationUpdate(
                    request.request_id,
                    EvaluationStatus.FAILED,
                    np.empty(0, dtype=np.int64),
                    error=EvaluationErrorInfo("backend", "partial failure"),
                    sequence=1,
                ),
            ]
        result = SerialEvaluator().evaluate_batch(request.x, problem)
        result.candidate_ids = request.candidate_ids
        result.__post_init__()
        handle._delivered_sequence = 0
        return [
            EvaluationUpdate(
                request.request_id,
                EvaluationStatus.COMPLETED,
                request.candidate_ids,
                result,
                sequence=0,
            )
        ]

    def acknowledge(self, handle: EvaluationHandle, sequence: int) -> None:
        self.acks.append((handle.backend_token[2], sequence))
        handle._acknowledged_sequence = sequence


class _OrderedPartialEvaluator(_PartialRetryEvaluator):
    """Deliver partial and final updates in one request with ordered sequences."""

    def collect(
        self, handle: EvaluationHandle, *, wait: bool = True
    ) -> list[EvaluationUpdate]:
        request, problem, attempt = handle.backend_token
        if attempt != 0 or handle._acknowledged_sequence >= 0:
            return super().collect(handle, wait=wait)
        first = SerialEvaluator().evaluate_batch(request.x[:1], problem)
        first.candidate_ids = request.candidate_ids[:1]
        first.__post_init__()
        final = SerialEvaluator().evaluate_batch(request.x[1:2], problem)
        final.candidate_ids = request.candidate_ids[1:2]
        final.__post_init__()
        handle._delivered_sequence = 1
        return [
            EvaluationUpdate(
                request.request_id,
                EvaluationStatus.PARTIAL,
                request.candidate_ids[:1],
                first,
                sequence=0,
            ),
            EvaluationUpdate(
                request.request_id,
                EvaluationStatus.COMPLETED,
                request.candidate_ids[1:2],
                final,
                sequence=1,
            ),
        ]


def test_ordered_partial_records_candidate_zero_true_evaluation() -> None:
    evaluator = _OrderedPartialEvaluator()
    optimizer = (
        Optimizer(_problem(), seed=7)
        .set_initializer(LHSInitializer(6, 2))
        .set_evaluator(evaluator)
        .set_async_evaluation_scheduler(
            AsyncEvaluationScheduler(evaluator, max_pending=1)
        )
        .set_termination(Termination(max_fe(12)))
        .set_history(["evaluation", "surrogate_accuracy"])
    )
    state = optimizer.run()
    assert state.history is not None
    channel = state.history.channel("evaluation")
    partial_mask = channel["status_code"] == 2
    assert partial_mask.any()
    f_blocks = state.history.blocks("evaluation", "f")
    saw_non_nan = False
    for idx in range(len(channel["request_id"])):
        if channel["status_code"][idx] == 2:
            # The PARTIAL carries the true evaluation of candidate 0 (first row).
            assert not np.isnan(f_blocks[idx][0, 0])
            saw_non_nan = True
    assert saw_non_nan


def test_partial_retry_records_partial_and_completed_rows() -> None:
    evaluator = _PartialRetryEvaluator()
    problem = _problem()
    request = EvaluationRequest(
        np.int64(0),
        np.array([10, 11], dtype=np.int64),
        np.array([[0.2, 0.1], [0.1, 0.2]], dtype=np.float64),
    )
    handle = evaluator.submit(request, problem)
    first_updates = list(evaluator.collect(handle, wait=True))
    handle2 = evaluator.submit(request, problem)
    second_updates = list(evaluator.collect(handle2, wait=True))
    updates = first_updates + second_updates
    state = _empty_state().replace(history=History(["evaluation"]))
    for _u in updates:
        _observe(state, _u, 0)
    record_evaluations(state)
    assert state.history is not None
    channel = state.history.channel("evaluation")
    recorded = set(
        zip(
            channel["request_id"],
            channel["sequence"],
            channel["status_code"],
            strict=True,
        )
    )
    assert (0, 0, 2) in recorded and (0, 0, 3) in recorded


def test_partial_update_without_result_is_not_recorded() -> None:
    candidate_ids = np.array([101], dtype=np.int64)
    update = EvaluationUpdate(
        request_id=np.int64(17),
        status=EvaluationStatus.PARTIAL,
        candidate_ids=candidate_ids,
        result=None,
        sequence=0,
    )
    state = _empty_state().replace(history=History(["evaluation"]))
    _observe(state, update, 0)
    record_evaluations(state)
    assert state.history is not None
    channel = state.history.channel("evaluation")
    assert len(channel.get("request_id", [])) == 0


def test_partial_row_survives_checkpoint_roundtrip(tmp_path: Path) -> None:
    evaluator = _PartialRetryEvaluator()
    optimizer = (
        Optimizer(_problem(), seed=7)
        .set_initializer(LHSInitializer(6, 2))
        .set_evaluator(evaluator)
        .set_async_evaluation_scheduler(
            AsyncEvaluationScheduler(evaluator, max_pending=1, retry_limit=1)
        )
        .set_termination(Termination(max_fe(12)))
        .set_history(["evaluation", "surrogate_accuracy"])
    )
    state = optimizer.run()
    assert state.history is not None
    channel = state.history.channel("evaluation")
    assert int((channel["status_code"] == 2).sum()) > 0
    path = tmp_path / "partial.npz"
    state.save(path)
    restored = OptimizationState.load(path, _problem())
    assert restored.history is not None
    restored_channel = restored.history.channel("evaluation")
    assert int((restored_channel["status_code"] == 2).sum()) == int(
        (channel["status_code"] == 2).sum()
    )


def test_all_history_channels_do_not_change_the_optimization_trajectory() -> None:
    recorded = _run_optimizer(
        _problem(2), ["summary", "front", "population", "surrogate_accuracy"]
    )
    disabled = _run_optimizer(_problem(2), [])

    np.testing.assert_array_equal(recorded.archive.x, disabled.archive.x)
    np.testing.assert_array_equal(recorded.archive.f, disabled.archive.f)


def test_disabled_front_has_no_block_column() -> None:
    history = History()
    with pytest.raises(ValidationError):
        history.blocks("front", "f")


def test_front_checkpoint_roundtrip(tmp_path: Path) -> None:
    state = _run_optimizer(_problem(2), ["summary", "front"])
    assert state.history is not None
    path = tmp_path / "front.npz"
    state.save(path)

    restored = OptimizationState.load(path, state.problem)
    assert restored.history is not None
    expected = state.history.blocks("front", "f")
    actual = restored.history.blocks("front", "f")
    assert len(actual) == len(expected)
    for expected_block, actual_block in zip(expected, actual):
        np.testing.assert_array_equal(actual_block, expected_block)


def test_checkpoint_resume_continues_summary_history(tmp_path: Path) -> None:
    state = _run_optimizer(_problem(), ["summary"])
    assert state.history is not None
    before = {
        name: values.copy() for name, values in state.history.channel("summary").items()
    }
    path = tmp_path / "history.npz"
    state.save(path)
    restored = OptimizationState.load(path, state.problem)

    optimizer = (
        Optimizer(state.problem, seed=7)
        .set_initializer(LHSInitializer(10, 3))
        .set_termination(Termination(max_gen(state.gen + 2)))
        .set_history(["summary"])
    )
    optimizer.resolve_defaults()
    resumed = optimizer.run_from(restored)

    assert resumed.history is not None
    summary = resumed.history.channel("summary")
    assert len(summary["gen"]) > len(before["gen"])
    np.testing.assert_array_equal(summary["gen"][: len(before["gen"])], before["gen"])
    assert np.unique(summary["gen"]).size == len(summary["gen"])


def test_checkpoint_without_history_arrays_loads_history_none(tmp_path: Path) -> None:
    state = _run_optimizer(_problem(), ["summary"])
    source = tmp_path / "history.npz"
    stripped = tmp_path / "history_without_arrays.npz"
    state.save(source)

    raw = dict(np.load(source, allow_pickle=False).items())
    history_keys = [
        key for key in raw if key == "_history_meta" or key.startswith("_history__")
    ]
    _rewrite_npz(source, stripped, **{key: None for key in history_keys})

    restored = OptimizationState.load(stripped, state.problem)
    assert restored.history is None


def test_checkpoint_without_history_rows_starts_recording_on_resume(
    tmp_path: Path,
) -> None:
    optimizer = (
        Optimizer(_problem(), seed=7)
        .set_initializer(LHSInitializer(10, 3))
        .set_termination(Termination(max_gen(2)))
        .set_history(["summary"])
    )
    state = optimizer.run()
    source = tmp_path / "history.npz"
    stripped = tmp_path / "history_without_arrays.npz"
    state.save(source)

    raw = dict(np.load(source, allow_pickle=False).items())
    for key in list(raw):
        if key == "_history_meta" or key.startswith("_history__"):
            del raw[key]
    np.savez(stripped, **raw)

    restored = OptimizationState.load(stripped, state.problem)
    assert restored.history is None
    resume_optimizer = (
        Optimizer(state.problem, seed=7)
        .set_initializer(LHSInitializer(10, 3))
        .set_termination(Termination(max_gen(4)))
        .set_history(["summary"])
    )
    resume_optimizer.resolve_defaults()
    resumed = resume_optimizer.run_from(restored)

    assert resumed.history is not None
    generations = resumed.history.channel("summary")["gen"]
    assert len(generations) > 0
    assert np.all(generations > state.gen)
    assert np.unique(generations).size == len(generations)


def test_checkpoint_roundtrip_preserves_disabled_history(tmp_path: Path) -> None:
    state = _run_optimizer(_problem(), [])
    path = tmp_path / "history_disabled.npz"
    state.save(path)

    restored = OptimizationState.load(path, state.problem)
    assert restored.history is not None
    assert restored.history.enabled == frozenset()


def test_async_runtime_records_one_summary_row_per_generation() -> None:
    evaluator = SerialEvaluator()
    optimizer = (
        Optimizer(_problem(), seed=7)
        .set_initializer(LHSInitializer(10, 3))
        .set_evaluator(evaluator)
        .set_async_evaluation_scheduler(
            AsyncEvaluationScheduler(evaluator, max_pending=1)
        )
        .set_termination(Termination(max_gen(2)))
    )
    state = optimizer.run()
    assert state.history is not None
    summary = state.history.channel("summary")
    assert len(summary["gen"]) == state.gen + 1


def test_async_runtime_records_one_decision_row_per_decision() -> None:
    evaluator = SerialEvaluator()
    optimizer = (
        Optimizer(_problem(), seed=7)
        .set_initializer(LHSInitializer(10, 3))
        .set_evaluator(evaluator)
        .set_async_evaluation_scheduler(
            AsyncEvaluationScheduler(evaluator, max_pending=1)
        )
        .set_termination(Termination(max_fe(13)))
        .set_history(["decision_candidates"])
    )
    state = optimizer.run()

    assert state.history is not None
    channel = state.history.channel("decision_candidates")
    assert len(channel["decision_count"]) == state.decision_count


def test_decision_candidate_history_has_one_record_per_decision() -> None:
    state = _run_optimizer(_problem(), ["decision_candidates"])

    assert state.history is not None
    channel = state.history.channel("decision_candidates")
    assert len(channel["decision_count"]) == state.decision_count
    assert len(state.history.blocks("decision_candidates", "candidate_ids")) == (
        state.decision_count
    )
    np.testing.assert_array_equal(
        channel["decision_count"], np.arange(1, state.decision_count + 1)
    )


def test_decision_candidates_are_recorded_for_a_replaced_plan_stage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class CustomPlanStage(EvaluationPlanStage):
        name = "custom_plan"

    monkeypatch.setattr(
        "saealib.strategies.direct.EvaluationPlanStage", CustomPlanStage
    )
    state = (
        Optimizer(_problem(), seed=7)
        .set_strategy(DirectStrategy())
        .set_initializer(LHSInitializer(10, 3))
        .set_termination(Termination(max_fe(13)))
        .set_history(["decision_candidates"])
        .run()
    )

    assert state.history is not None
    assert len(state.history.channel("decision_candidates")["decision_count"]) == (
        state.decision_count
    )


def test_record_decision_skips_missing_inputs_and_repeated_states() -> None:
    state = _empty_state().replace(
        history=History(["decision_candidates"]), decision_count=1
    )

    record_decision(state)
    assert state.history is not None
    assert state.history.channel("decision_candidates") == {}


def test_decision_candidate_selected_matches_independently_observed_plans(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_execute = EvaluationPlanStage.execute
    observed_ids: list[set[int]] = []

    def observe_plan(stage: EvaluationPlanStage, state: OptimizationState):
        previous_count = state.decision_count
        result = original_execute(stage, state)
        if result.decision_count == previous_count + 1:
            assert result.evaluation_plan is not None
            observed_ids.append(
                {
                    int(candidate_id)
                    for request in result.evaluation_plan.requests
                    for candidate_id in request.candidate_ids
                }
            )
        return result

    monkeypatch.setattr(EvaluationPlanStage, "execute", observe_plan)
    state = _run_optimizer(_problem(), ["decision_candidates"])

    assert state.history is not None
    blocks = state.history.blocks("decision_candidates", "selected")
    candidate_id_blocks = state.history.blocks("decision_candidates", "candidate_ids")
    assert len(observed_ids) == len(blocks) == state.decision_count
    for selected, candidate_ids, planned_ids in zip(
        blocks, candidate_id_blocks, observed_ids
    ):
        expected = np.array(
            [
                [int(candidate_id) in planned_ids]
                for candidate_id in candidate_ids[:, 0]
            ],
            dtype=bool,
        )
        np.testing.assert_array_equal(selected, expected)


def test_decision_candidate_acquisition_scores_match_state_scores(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_execute = EvaluationPlanStage.execute
    observed_scores: list[np.ndarray] = []

    def observe_scores(stage: EvaluationPlanStage, state: OptimizationState):
        previous_count = state.decision_count
        scores = None if state.scores is None else np.array(state.scores, copy=True)
        result = original_execute(stage, state)
        if result.decision_count == previous_count + 1:
            assert scores is not None
            observed_scores.append(np.array(scores, dtype=np.float64, copy=True))
        return result

    monkeypatch.setattr(EvaluationPlanStage, "execute", observe_scores)
    state = _run_optimizer(_problem(), ["decision_candidates"])

    assert state.history is not None
    blocks = state.history.blocks("decision_candidates", "acquisition_scores")
    assert len(observed_scores) == len(blocks) == state.decision_count
    for block, scores in zip(blocks, observed_scores):
        np.testing.assert_array_equal(block[:, 0], scores)


def test_decision_candidate_none_acquisition_scores_are_nan_with_column() -> None:
    state = (
        Optimizer(_problem(), seed=7)
        .set_initializer(LHSInitializer(10, 3))
        .set_acquisition(_NoScoresAcquisition())
        .set_evaluation_planner(EvaluateAll())
        .set_termination(Termination(max_fe(13)))
        .set_history(["decision_candidates"])
        .run()
    )

    assert state.history is not None
    blocks = state.history.blocks("decision_candidates", "acquisition_scores")
    assert len(blocks) == state.decision_count
    assert blocks
    assert all(np.all(np.isnan(block)) for block in blocks)


def test_decision_candidate_block_dtypes_and_missing_values() -> None:
    value = 2**53 + 1
    state = _empty_state().replace(
        history=History(["decision_candidates"]), decision_count=1
    )
    state.population._extend_internal(
        {
            "x": np.zeros((1, 2)),
            "f": np.zeros((1, 1)),
            "g": np.zeros((1, 0)),
            "cv": np.zeros(1),
            "id": np.array([value], dtype=np.int64),
        },
        preserve_ids=True,
    )
    plan = EvaluateAll().plan(state.population, None, state)
    record_state = state.replace(
        offspring=state.population,
        evaluation_plan=plan,
        evaluation_plan_state=EvaluationPlanState(
            deferred=tuple(int(item.request_id) for item in plan.requests)
        ),
    )
    record_decision(record_state)
    record_decision(record_state)

    assert state.history is not None
    candidate_ids = state.history.blocks("decision_candidates", "candidate_ids")[0]
    selected = state.history.blocks("decision_candidates", "selected")[0]
    assert candidate_ids.dtype == np.int64
    np.testing.assert_array_equal(candidate_ids[:, 0], [value])
    assert selected.dtype == bool
    np.testing.assert_array_equal(selected, [[True]])
    for column in (
        "acquisition_scores",
        "prediction_mean",
        "prediction_std",
    ):
        block = state.history.blocks("decision_candidates", column)[0]
        assert block.dtype == np.float64
        assert np.all(np.isnan(block))


def test_decision_candidate_default_rbf_prediction_std_is_nan_with_column() -> None:
    state = _run_optimizer(_problem(), ["decision_candidates"])

    assert state.history is not None
    blocks = state.history.blocks("decision_candidates", "prediction_std")
    assert len(blocks) == state.decision_count
    assert blocks
    assert all(np.all(np.isnan(block)) for block in blocks)


def test_decision_candidates_without_dense_view_omit_only_candidates_block() -> None:
    space = ObjectSpace(RepresentationSpec(kind="permutation"))
    problem = Problem(
        func=lambda _: np.array([0.0]),
        dim=2,
        n_obj=1,
        direction=np.array([-1.0]),
        space=space,
    )
    state = _empty_state().replace(
        problem=problem,
        history=History(["decision_candidates"]),
    )
    state.population._extend_internal(
        {
            "x": np.array([[0.0, 1.0]]),
            "f": np.array([[2.0]]),
            "g": np.zeros((1, 0)),
            "cv": np.zeros(1),
            "id": np.array([10], dtype=np.int64),
        },
        preserve_ids=True,
    )
    plan = EvaluateAll().plan(state.population, None, state)
    record_decision(
        state.replace(
            decision_count=1,
            offspring=state.population,
            evaluation_plan=plan,
            evaluation_plan_state=EvaluationPlanState(
                deferred=tuple(int(item.request_id) for item in plan.requests)
            ),
        )
    )

    assert state.history is not None
    for column in (
        "candidate_ids",
        "selected",
        "acquisition_scores",
        "prediction_mean",
        "prediction_std",
    ):
        blocks = state.history.blocks("decision_candidates", column)
        assert len(blocks) == 1
        assert blocks[0].shape[0] == 1
    with pytest.raises(ValidationError):
        state.history.blocks("decision_candidates", "candidates")


def test_async_decision_candidates_record_multiple_decisions_in_one_generation() -> (
    None
):
    state = _empty_state().replace(history=History(["decision_candidates"]), gen=5)
    evaluator = SerialEvaluator()
    scheduler = AsyncEvaluationScheduler(evaluator, max_pending=1)
    stage = AsyncEvaluationSubmitStage(scheduler, EvaluateAll())

    def make_candidates(candidate_id: int) -> Population:
        candidates = state.population.empty_like(capacity=1)
        candidates._extend_internal(
            {
                "x": np.array([[float(candidate_id), 0.0]]),
                "f": np.array([[np.nan]]),
                "g": np.empty((1, 0)),
                "cv": np.zeros(1),
                "id": np.array([candidate_id], dtype=np.int64),
            },
            preserve_ids=True,
        )
        return candidates

    first = make_candidates(100)
    state = state.replace(offspring=first)
    state = stage.execute(state)
    record_decision(state)
    state = scheduler.poll(state, wait=True)
    assert not state.pending_evaluations

    second = make_candidates(101)
    state = state.replace(
        offspring=second,
        evaluation_request=None,
        evaluation_plan=None,
        evaluation_plan_state=None,
        evaluation_plan_updates={},
        evaluation_handles={},
        evaluation_owners={},
        pending_evaluations={},
    )
    state = stage.execute(state)
    record_decision(state)
    state = scheduler.poll(state, wait=True)

    assert state.history is not None
    channel = state.history.channel("decision_candidates")
    assert state.decision_count == 2
    assert len(channel["decision_count"]) == state.decision_count
    assert np.unique(channel["gen"]).size == 1


def test_decision_candidates_checkpoint_roundtrip_preserves_all_blocks(
    tmp_path: Path,
) -> None:
    state = _run_optimizer(_problem(2), ["decision_candidates"])
    path = tmp_path / "decision_candidates.npz"
    state.save(path)
    restored = OptimizationState.load(path, state.problem)

    assert state.history is not None
    assert restored.history is not None
    expected_channel = state.history.channel("decision_candidates")
    actual_channel = restored.history.channel("decision_candidates")
    assert list(actual_channel) == list(expected_channel)
    for name, values in expected_channel.items():
        np.testing.assert_array_equal(actual_channel[name], values)

    for column in (
        "candidate_ids",
        "selected",
        "acquisition_scores",
        "prediction_mean",
        "prediction_std",
        "candidates",
    ):
        expected = state.history.blocks("decision_candidates", column)
        actual = restored.history.blocks("decision_candidates", column)
        assert len(actual) == len(expected)
        for expected_block, actual_block in zip(expected, actual):
            np.testing.assert_array_equal(actual_block, expected_block)


def test_all_history_channels_with_decisions_do_not_change_trajectory() -> None:
    recorded = _run_optimizer(
        _problem(2),
        [
            "summary",
            "front",
            "population",
            "surrogate_accuracy",
            "decision_candidates",
        ],
    )
    disabled = _run_optimizer(_problem(2), [])

    np.testing.assert_array_equal(recorded.archive.x, disabled.archive.x)
    np.testing.assert_array_equal(recorded.archive.f, disabled.archive.f)


def test_structured_generation_decision_candidates_match_decision_count() -> None:
    state = _run_generation_based_optimizer(_problem(), ["decision_candidates"])

    assert state.history is not None
    channel = state.history.channel("decision_candidates")
    assert len(channel["decision_count"]) == state.decision_count
    assert len(state.history.blocks("decision_candidates", "candidate_ids")) == (
        state.decision_count
    )


def test_structured_generation_decision_history_does_not_change_trajectory() -> None:
    recorded = _run_generation_based_optimizer(
        _problem(2),
        [
            "summary",
            "front",
            "population",
            "surrogate_accuracy",
            "decision_candidates",
        ],
    )
    disabled = _run_generation_based_optimizer(_problem(2), [])

    np.testing.assert_array_equal(recorded.archive.x, disabled.archive.x)
    np.testing.assert_array_equal(recorded.archive.f, disabled.archive.f)


def test_replicate_records_both_raw_values_not_aggregate() -> None:
    calls = {"i": 0}

    def func(x: np.ndarray) -> np.ndarray:
        i = calls["i"]
        calls["i"] += 1
        base = float(np.sum(np.asarray(x) ** 2))
        noise = (i % 2) * 4.0
        return np.array([base + noise])

    problem = Problem(
        func=func,
        dim=2,
        n_obj=1,
        direction=np.array([-1.0]),
        lb=[-1.0, -1.0],
        ub=[1.0, 1.0],
    )
    optimizer = (
        Optimizer(problem, seed=7)
        .set_initializer(LHSInitializer(10, 1))
        .set_evaluation_planner(RepeatedEvaluation(2))
        .set_termination(Termination(max_fe(14)))
        .set_history(["evaluation"])
    )
    state = optimizer.run()
    assert state.history is not None
    ch = state.history.channel("evaluation")
    f_blocks = state.history.blocks("evaluation", "f")
    full_indices = [
        i for i in range(len(ch["request_id"])) if int(ch["request_id"][i]) != -1
    ]
    decision_counts = [int(ch["decision_count"][i]) for i in full_indices]
    per_decision: dict[int, int] = {}
    for d in decision_counts:
        per_decision[d] = per_decision.get(d, 0) + 1
    assert per_decision
    assert all(c == 2 for c in per_decision.values())

    decision_rows: dict[int, list[int]] = {}
    for idx, d in zip(full_indices, decision_counts):
        decision_rows.setdefault(d, []).append(idx)
    for idxs in decision_rows.values():
        assert len(idxs) == 2
        f0 = f_blocks[idxs[0]].reshape(-1)
        f1 = f_blocks[idxs[1]].reshape(-1)
        assert np.allclose(np.abs(f0 - f1), 4.0)


def test_retry_records_partial_attempts_as_distinct_rows() -> None:
    # A retry reuses the request ID and restarts the update sequence at 0, so
    # only the attempt separates the two PARTIALs.
    update_a0 = EvaluationUpdate(
        np.int64(0),
        EvaluationStatus.PARTIAL,
        np.array([10], dtype=np.int64),
        EvaluationResult(
            f=np.array([[1.0]]),
            g=np.empty((1, 0), dtype=np.float64),
            cv=np.zeros(1),
            candidate_ids=np.array([10], dtype=np.int64),
        ),
        sequence=0,
    )
    update_a1 = EvaluationUpdate(
        np.int64(0),
        EvaluationStatus.PARTIAL,
        np.array([11], dtype=np.int64),
        EvaluationResult(
            f=np.array([[2.0]]),
            g=np.empty((1, 0), dtype=np.float64),
            cv=np.zeros(1),
            candidate_ids=np.array([11], dtype=np.int64),
        ),
        sequence=0,
    )
    state = _empty_state().replace(history=History(["evaluation"]))
    _observe(state, update_a0, 0)
    _observe(state, update_a1, 1)
    record_evaluations(state)
    assert state.history is not None
    channel = state.history.channel("evaluation")
    partials = {
        (int(r), int(s), int(a))
        for r, s, st, a in zip(
            channel["request_id"],
            channel["sequence"],
            channel["status_code"],
            channel["attempt"],
        )
        if int(st) == 2
    }
    assert (0, 0, 0) in partials
    assert (0, 0, 1) in partials


def test_async_records_resultless_failed_terminal_updates() -> None:
    class _AsyncFailEval(Evaluator):
        def evaluate_batch(self, x: Any, problem: Problem) -> EvaluationResult:
            return EvaluationResult(
                f=np.array([[1.0] for _ in x]),
                g=np.empty((len(x), problem.n_constraints)),
                cv=np.zeros(len(x)),
                candidate_ids=np.arange(len(x)),
            )

        def submit(self, request: Any, problem: Problem) -> Any:
            return EvaluationHandle(
                request.request_id,
                EvaluationStatus.PENDING,
                backend_token=(request, problem),
            )

        def collect(self, handle: Any, *, wait: bool = True) -> list[EvaluationUpdate]:
            request, _problem = handle.backend_token
            if handle._acknowledged_sequence >= 0:
                return []
            handle._delivered_sequence = 0
            return [
                EvaluationUpdate(
                    request.request_id,
                    EvaluationStatus.FAILED,
                    np.empty(0, dtype=np.int64),
                    error=EvaluationErrorInfo("e", "boom"),
                    sequence=0,
                )
            ]

        def acknowledge(self, handle: Any, sequence: int) -> None:
            handle._acknowledged_sequence = sequence

    evaluator = _AsyncFailEval()
    optimizer = (
        Optimizer(_problem(), seed=7)
        .set_initializer(LHSInitializer(10, 3))
        .set_evaluator(evaluator)
        .set_async_evaluation_scheduler(
            AsyncEvaluationScheduler(evaluator, max_pending=1)
        )
        .set_termination(Termination(max_gen(3)))
        .set_history(["evaluation"])
    )
    state = optimizer.run()
    assert state.history is not None
    ch = state.history.channel("evaluation")
    assert any(int(st) == 4 for st in ch["status_code"])


@pytest.mark.parametrize("path", ["sync", "async", "structured"])
def test_evaluation_fe_before_after_match_state_fe(path: str) -> None:
    state = _run_history_path(path, ["evaluation"])
    assert state.history is not None
    ch = state.history.channel("evaluation")
    fe_before = ch["fe_before"]
    fe_after = ch["fe_after"]
    assert int(fe_before[0]) == 0
    assert int(fe_after[0]) == 10
    for i in range(1, len(ch["request_id"])):
        assert int(fe_before[i]) == int(fe_after[i - 1])
    assert int(fe_after[-1]) == int(state.fe)


def test_evaluation_history_checkpoint_roundtrip_preserves_attempt_and_fe(
    tmp_path: Path,
) -> None:
    state = _run_history_path("sync", ["evaluation"])
    path = tmp_path / "evaluation.npz"
    state.save(path)
    restored = OptimizationState.load(path, state.problem)
    assert state.history is not None
    assert restored.history is not None
    expected = state.history.channel("evaluation")
    actual = restored.history.channel("evaluation")
    for column in ("attempt", "fe_before", "fe_after"):
        np.testing.assert_array_equal(actual[column], expected[column])
        assert actual[column].dtype == np.dtype(np.int64)


def test_evaluation_history_resume_preserves_fe_counter(tmp_path: Path) -> None:
    optimizer = (
        Optimizer(_problem(), seed=7)
        .set_initializer(LHSInitializer(10, 3))
        .set_termination(Termination(max_fe(16)))
        .set_history(["evaluation"])
    )
    state = optimizer.run()
    path = tmp_path / "ckpt.npz"
    state.save(path)
    restored = OptimizationState.load(path, state.problem)
    assert restored.history is not None
    prev_max_fe_after = int(restored.history.channel("evaluation")["fe_after"].max())
    # run_from validates against the configured optimizer, so the resume must
    # reuse it and change only the budget.
    final = optimizer.set_termination(Termination(max_fe(20))).run_from(restored)
    assert final.history is not None
    ch = final.history.channel("evaluation")
    new_rows = [
        i
        for i in range(len(ch["request_id"]))
        if int(ch["fe_after"][i]) > prev_max_fe_after
    ]
    assert new_rows
    assert int(ch["fe_before"][new_rows[0]]) == prev_max_fe_after


def test_minimize_records_specified_history_channels() -> None:
    result = minimize(
        _problem(),
        max_fe=13,
        pop_size=3,
        seed=7,
        history_channels=["evaluation", "summary"],
        verbose=False,
    )
    assert result.history is not None
    assert "evaluation" in result.history.enabled
    assert "summary" in result.history.enabled
    assert result.history.channel("evaluation")["request_id"].shape[0] >= 1


def test_scheduler_retry_stamps_the_attempt_that_produced_each_update() -> None:
    evaluator = _PartialRetryEvaluator()
    state = _empty_state()
    state.population._extend_internal(
        {
            "id": np.array([10, 11], dtype=np.int64),
            "x": np.array([[0.2, 0.1], [0.1, 0.2]], dtype=np.float64),
            "f": np.full((2, 1), np.nan),
            "g": np.empty((2, 0)),
            "cv": np.zeros(2),
        },
        preserve_ids=True,
    )
    state = state.replace(history=History(["evaluation"]), offspring=state.population)
    request = EvaluationRequest(
        np.int64(0),
        np.array([10, 11], dtype=np.int64),
        np.array([[0.2, 0.1], [0.1, 0.2]], dtype=np.float64),
    )
    scheduler = AsyncEvaluationScheduler(evaluator, retry_limit=1)
    state = scheduler.poll(scheduler.submit(state, [request]), wait=True)
    record_evaluations(state)

    assert evaluator.attempts == 2
    assert state.history is not None
    channel = state.history.channel("evaluation")
    recorded = set(
        zip(
            channel["sequence"],
            channel["status_code"],
            channel["attempt"],
            strict=True,
        )
    )
    assert (0, 2, 0) in recorded
    assert (1, 4, 0) in recorded
    assert (0, 3, 1) in recorded


def test_retry_keeps_surrogate_prediction_for_the_completed_retry() -> None:
    state = _empty_state()
    state.population._extend_internal(
        {
            "id": np.array([10, 11], dtype=np.int64),
            "x": np.array([[0.2, 0.1], [0.1, 0.2]], dtype=np.float64),
            "f": np.full((2, 1), np.nan),
            "g": np.empty((2, 0)),
            "cv": np.zeros(2),
        },
        preserve_ids=True,
    )
    offspring = state.population.extract([0, 1])
    plan = EvaluateAll().plan(offspring, None, state)
    state = state.replace(
        history=History(["evaluation", "surrogate_accuracy"]),
        offspring=offspring,
        evaluation_plan=plan,
        evaluation_plan_state=EvaluationPlanState(
            deferred=tuple(int(r.request_id) for r in plan.requests)
        ),
        predictions=SurrogatePrediction(
            {"objective": PredictionChannel(np.array([[1.0], [2.0]]))}
        ),
    )
    evaluator = _PartialRetryEvaluator()
    scheduler = AsyncEvaluationScheduler(evaluator, retry_limit=1)
    state = scheduler.submit(state, list(plan.requests))
    record_evaluations(state)
    state = scheduler.poll(state, wait=True)
    record_evaluations(state)

    assert state.history is not None
    accuracy = state.history.channel("surrogate_accuracy")
    np.testing.assert_array_equal(accuracy["size"], [1, 0, 1])
    predicted = state.history.blocks("surrogate_accuracy", "predicted")
    np.testing.assert_array_equal(predicted[0], [[1.0]])
    np.testing.assert_array_equal(predicted[2], [[2.0]])
    assert state.history._surrogate_predictions == {}


def test_initial_evaluation_fe_counter_for_surrogate_accuracy_only() -> None:
    state = _run_optimizer(_problem(), ["surrogate_accuracy"])
    assert state.history is not None
    accuracy = state.history.channel("surrogate_accuracy")
    assert len(accuracy["fe_before"]) > 0
    np.testing.assert_array_equal(accuracy["fe_before"][0], 10)
