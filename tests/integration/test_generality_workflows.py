import importlib.util
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np

from saealib import (
    PSO,
    BatchExpectedImprovement,
    EvaluationRequest,
    EvaluationResult,
    Evaluator,
    GlobalSurrogateManager,
    LHSInitializer,
    Optimizer,
    PredictionChannel,
    PreSelectionStrategy,
    Problem,
    ReplicateSummary,
    SteadyStateStrategy,
    SurrogatePrediction,
    Termination,
    aggregate_replicates,
    max_fe,
)
from saealib.context import OptimizationState
from saealib.core.contracts import FeedbackBatch
from saealib.core.state import StatePatch, StateView
from saealib.surrogate import ArchiveObjectiveSet


def reference_problem(dim=2, shift=0.0):
    def evaluate(x):
        return np.array([np.sum((x - shift) ** 2)], dtype=np.float64)

    return Problem(
        func=evaluate,
        dim=dim,
        n_obj=1,
        direction=np.array([-1.0]),
        lb=[-1.0] * dim,
        ub=[1.0] * dim,
    )


def _load_example(name):
    path = Path(__file__).parents[2] / "examples" / name
    import sys

    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_qei_is_joint_and_uses_candidate_covariance():
    x = np.array([[0.0], [1.0]], dtype=np.float64)
    prediction = SurrogatePrediction(
        channels={
            "objective": PredictionChannel(
                value=np.array([[0.2], [0.2]]),
                std=np.array([[1.0], [1.0]]),
                covariance=np.array([[1.0, 0.95], [0.95, 1.0]]),
            )
        }
    )
    archive = SimpleNamespace(f=np.array([[0.0]], dtype=np.float64))
    result = BatchExpectedImprovement(n_draws=2048).evaluate(
        x, prediction, archive, ctx=None
    )
    assert result.artifacts["joint"] is True
    assert result.artifacts["order"].shape == (2,)
    assert np.array_equal(result.artifacts["order"], np.array([1, 0]))


def test_qei_direction_none_and_minimize_direction_are_equivalent():
    prediction = SurrogatePrediction(
        channels={
            "objective": PredictionChannel(
                value=np.array([[0.2], [0.4]]),
                std=np.ones((2, 1)),
                covariance=np.eye(2),
            )
        }
    )
    archive = SimpleNamespace(f=np.array([[0.0]], dtype=np.float64))
    none = BatchExpectedImprovement(n_draws=32).evaluate(
        np.zeros((2, 1)), prediction, archive, ctx=None
    )
    minimum = BatchExpectedImprovement(n_draws=32, direction=np.array([-1.0])).evaluate(
        np.zeros((2, 1)), prediction, archive, ctx=None
    )
    assert np.array_equal(none.scores, minimum.scores)


def test_qei_example_surrogate_exposes_correlated_posterior():
    surrogate = _load_example("generality_qei.py").CorrelatedQuadraticSurrogate(0.8)
    prediction = surrogate.predict(np.array([[0.0], [1.0]], dtype=np.float64))
    assert prediction.channels["objective"].covariance[0, 1] > 0


def test_qei_correlation_changes_known_selection_order():
    archive = SimpleNamespace(f=np.array([[0.0]], dtype=np.float64))
    values = np.zeros((2, 1), dtype=np.float64)
    std = np.ones((2, 1), dtype=np.float64)

    def evaluate(covariance):
        prediction = SurrogatePrediction(
            channels={
                "objective": PredictionChannel(
                    value=values, std=std, covariance=covariance
                )
            }
        )
        return BatchExpectedImprovement(n_draws=8192).evaluate(
            np.zeros((2, 1)),
            prediction,
            archive,
            ctx=cast(OptimizationState, SimpleNamespace(rng=np.random.default_rng(7))),
        )

    diagonal = evaluate(np.eye(2))
    correlated = evaluate(np.full((2, 2), 0.99) + np.diag([0.01, 0.01]))
    assert correlated.artifacts["qei"][1] < diagonal.artifacts["qei"][1]
    assert not np.allclose(diagonal.artifacts["qei"], correlated.artifacts["qei"])


def test_qei_optimizer_path_preserves_joint_covariance():
    class RecordingQEI(BatchExpectedImprovement):
        def __init__(self):
            super().__init__(n_draws=64)
            self.covariances = []

        def evaluate(self, candidates_x, prediction, archive, *, prepared=None, ctx):
            self.covariances.append(
                np.array(prediction.channels["objective"].covariance, copy=True)
            )
            return super().evaluate(
                candidates_x, prediction, archive, prepared=prepared, ctx=ctx
            )

    acquisition = RecordingQEI()
    optimizer = (
        Optimizer(reference_problem(), seed=41)
        .set_initializer(LHSInitializer(2, 2, 41))
        .set_algorithm(PSO())
        .set_surrogate_manager(
            GlobalSurrogateManager(
                _load_example("generality_qei.py").CorrelatedQuadraticSurrogate(0.8),
                ArchiveObjectiveSet(),
            )
        )
        .set_acquisition(acquisition)
        .set_strategy(PreSelectionStrategy(4, 2))
        .set_termination(Termination(max_fe(6)))
    )
    optimizer.run()
    assert any(
        covariance.shape[0] > 1
        and np.any(np.abs(covariance - np.diag(np.diag(covariance))) > 0)
        for covariance in acquisition.covariances
    )


def test_repeated_observations_preserve_ids_and_aggregate_noise():
    ids = np.array([10, 11], dtype=np.int64)
    observations = np.array(
        [[[1.0], [2.0]], [[1.2], [1.8]], [[0.8], [2.2]]], dtype=np.float64
    )
    summary = aggregate_replicates(ids, observations)
    assert isinstance(summary, ReplicateSummary)
    assert np.array_equal(summary.candidate_ids, ids)
    assert np.array_equal(summary.count, np.array([3, 3]))
    assert np.allclose(summary.mean[:, 0], [1.0, 2.0])
    assert np.all(summary.std > 0)


def test_repeated_policy_creates_distinct_requests_for_same_candidates():
    from saealib import RepeatedEvaluation

    class Allocator:
        def __init__(self):
            self.next = 10

        def allocate(self, count):
            values = np.arange(self.next, self.next + count, dtype=np.int64)
            self.next += count
            return values

    class Candidates(SimpleNamespace):
        def __len__(self):
            return len(self.x)

    candidates = Candidates(x=np.array([[0.1], [0.2]], dtype=np.float64), schema={})
    requests = RepeatedEvaluation(3).plan_replicates(
        candidates, SimpleNamespace(request_id_allocator=Allocator())
    )
    assert [request.metadata["replicate"] for request in requests] == [0, 1, 2]
    assert len({int(request.request_id) for request in requests}) == 3
    assert all(np.array_equal(request.candidate_ids, [0, 1]) for request in requests)


def test_async_example_contract_has_independent_candidate_futures():
    from saealib import AsyncEvaluator, Problem

    class Delayed(Evaluator):
        def evaluate_batch(self, x, problem):
            time.sleep(0.02 if x[0, 0] < 0.5 else 0.001)
            return EvaluationResult(
                f=np.array([[x[0, 0]]], dtype=np.float64),
                g=np.empty((1, 0), dtype=np.float64),
                cv=np.zeros(1, dtype=np.float64),
            )

    problem = Problem(
        lambda row: np.array([row[0]]),
        dim=1,
        n_obj=1,
        direction=np.array([-1.0]),
        lb=[0.0],
        ub=[1.0],
    )
    evaluator = AsyncEvaluator(Delayed(), max_workers=2)
    slow = EvaluationRequest(np.int64(1), np.array([10]), np.array([[0.1]]))
    fast = EvaluationRequest(np.int64(2), np.array([11]), np.array([[0.9]]))
    slow_handle = evaluator.submit(slow, problem)
    fast_handle = evaluator.submit(fast, problem)
    fast_update = evaluator.collect(fast_handle, wait=True)[0]
    slow_update = evaluator.collect(slow_handle, wait=True)[0]
    assert fast_update.candidate_ids.tolist() == [11]
    assert slow_update.candidate_ids.tolist() == [10]


def test_async_optimizer_uses_single_candidate_refill_and_drains():
    from saealib import AsyncEvaluationScheduler, AsyncEvaluator, SerialEvaluator

    class DelayedSerial(SerialEvaluator):
        def __init__(self):
            super().__init__()
            self.requests = []
            self.completions = []
            self.active = 0
            self.peak_active = 0
            self.lock = threading.Lock()
            self.first_started = threading.Event()
            self.refill_started = threading.Event()
            self.release_first = threading.Event()
            self.events = []

        def evaluate_request(self, request, problem):
            with self.lock:
                self.requests.append(request)
                self.active += 1
                self.peak_active = max(self.peak_active, self.active)
                single_count = sum(
                    len(item.candidate_ids) == 1 for item in self.requests
                )
                if len(request.candidate_ids) == 1:
                    self.events.append(f"start-{single_count}")
            if len(request.candidate_ids) == 1:
                if single_count == 1:
                    self.first_started.set()
                    assert self.release_first.wait(5)
                elif single_count == 2:
                    assert self.first_started.wait(5)
                else:
                    self.refill_started.set()
                    self.release_first.set()
            result = super().evaluate_request(request, problem)
            with self.lock:
                self.completions.append(request.candidate_ids.copy())
                if len(request.candidate_ids) == 1:
                    self.events.append(f"complete-{single_count}")
                self.active -= 1
            return result

    class RecordingPSO(PSO):
        def __init__(self):
            super().__init__()
            self.told_ids = []

        def tell(self, feedback: FeedbackBatch, state: StateView) -> StatePatch:
            self.told_ids.extend(state.context.offspring.get_array("id").tolist())
            return super().tell(feedback, state)

    recorder = DelayedSerial()
    evaluator = AsyncEvaluator(recorder, max_workers=2)
    scheduler = AsyncEvaluationScheduler(evaluator, max_pending=2, max_reserved_fe=6)
    algorithm = RecordingPSO()
    optimizer = (
        Optimizer(reference_problem(), seed=53)
        .set_initializer(LHSInitializer(2, 2, 53))
        .set_algorithm(algorithm)
        .set_strategy(SteadyStateStrategy())
        .set_evaluator(evaluator)
        .set_async_evaluation_scheduler(scheduler)
        .set_termination(
            Termination(lambda state: state.fe + len(state.pending_evaluations) >= 6)
        )
    )
    state = optimizer.run()
    while state.pending_evaluations:
        state = scheduler.poll(state, wait=True)
    assert all(len(request.candidate_ids) == 1 for request in recorder.requests[1:])
    assert len(recorder.requests) <= 6
    assert recorder.peak_active == 2
    assert recorder.refill_started.is_set()
    assert recorder.events.index("complete-2") < recorder.events.index("start-3")
    submitted = [int(request.candidate_ids[0]) for request in recorder.requests[1:]]
    assert len(submitted) >= 2
    archive_ids = set(state.archive.get_array("id"))
    completion_ids = set(np.concatenate(recorder.completions))
    initial_ids = archive_ids - completion_ids
    assert len(initial_ids) == 2
    assert completion_ids == set(algorithm.told_ids)
    assert archive_ids - initial_ids <= set(algorithm.told_ids)
    assert state.fe <= 6
    assert len(state.pending_evaluations) == 0
    assert len(state.evaluation_handles) == 0
    assert len(state.evaluation_owners) == 0


def test_examples_use_their_actual_public_components():
    examples = {
        "generality_sync_ib.py": "archive",
        "generality_ego.py": "archive",
        "generality_qei.py": "archive",
        "generality_async.py": "archive",
        "generality_pairwise.py": "archive",
        "generality_ehvi.py": "archive",
    }
    for name, attribute in examples.items():
        result = _load_example(name).main()
        assert len(getattr(result, attribute)) > 0
    async_state = _load_example("generality_async.py").main()
    assert async_state.fe == 6
    assert len(async_state.pending_evaluations) == 0
    assert len(async_state.evaluation_handles) == 0
    assert len(async_state.evaluation_owners) == 0
    assert len(async_state.archive) == len(
        np.unique(async_state.archive.get_array("id"))
    )

    islands = _load_example("generality_islands.py").main()
    assert len(islands["islands"]) == 2
    assert islands["events"]
    assert any(
        np.array_equal(source_row, target_row)
        for source_row in islands["islands"][0]
        for target_row in islands["islands"][1]
    )
    dynamic = _load_example("generality_dynamic_archive.py").main()
    assert len(dynamic["snapshots"]) == 2
    assert dynamic["selected_name"] == "env_10"
    assert dynamic["selected_name"] != "env_0"
    assert dynamic["selected"] is dynamic["state"].archives[dynamic["selected_name"]]
    ehvi_state = _load_example("generality_ehvi.py").main()
    assert ehvi_state.problem.n_obj == 2
    pareto_f = ehvi_state.pareto_archive.get_array("f")
    assert len(pareto_f) >= 4
    distinct_pareto_f = np.unique(pareto_f, axis=0)
    assert len(distinct_pareto_f) >= 2
    noisy = _load_example("generality_noisy.py").main()
    assert noisy["state"].fe == 8
    assert len(noisy["requests"]) == 3
    assert {int(request.metadata["replicate"]) for request in noisy["requests"]} == {
        0,
        1,
        2,
    }
    fidelity = _load_example("generality_multifidelity.py").main()
    assert fidelity["state"].fe >= 4
    assert [request.metadata["fidelity"] for request in fidelity["requests"]] == [
        0,
        1,
    ]
    assert fidelity["requests"][1].metadata["promotion_of"] == int(
        fidelity["requests"][0].request_id
    )
    assert set(fidelity["requests"][1].candidate_ids) <= set(
        fidelity["requests"][0].candidate_ids
    )


def test_ehvi_example_invokes_ehvi_acquisition(monkeypatch):
    example = _load_example("generality_ehvi.py")
    calls = []
    base_ehvi = example.EHVIAcquisition

    class RecordingEHVI(base_ehvi):
        def evaluate(self, candidates_x, prediction, archive, *, prepared=None, ctx):
            result = super().evaluate(
                candidates_x,
                prediction,
                archive,
                prepared=prepared,
                ctx=ctx,
            )
            calls.append((prediction.value.shape, result.scores.copy()))
            return result

    monkeypatch.setattr(example, "EHVIAcquisition", RecordingEHVI)
    state = example.main()

    assert state.problem.n_obj == 2
    assert calls
    assert all(shape[1] == 2 for shape, _scores in calls)
    assert all(np.all(np.isfinite(scores)) for _shape, scores in calls)
