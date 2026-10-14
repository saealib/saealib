import time
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from saealib.callback import CallbackManager, GenerationEndEvent, PostEvaluationEvent
from saealib.context import EvaluationPlanState, OptimizationState
from saealib.core.contracts import (
    ComponentContract,
    FeedbackBatch,
    FeedbackContract,
    FeedbackRequirement,
    LifecycleContract,
    ProposalBatch,
    ProposalRelations,
)
from saealib.core.contracts.feedback import IN_ORDER, PARTIAL_ALLOWED, REPEATED_ALLOWED
from saealib.core.state import PROPOSALS_CURRENT, StatePatch, StateView
from saealib.exceptions import (
    CheckpointError,
    EvaluationFatalError,
    EvaluationProtocolError,
    EvaluationSubmissionError,
    ValidationError,
)
from saealib.execution import (
    AsyncEvaluationScheduler,
    AsyncEvaluator,
    EvaluationErrorInfo,
    EvaluationHandle,
    EvaluationRequest,
    EvaluationResult,
    EvaluationStatus,
    EvaluationUpdate,
    Evaluator,
    PendingEvaluation,
    SerialEvaluator,
)
from saealib.execution.runner import Runner
from saealib.policies.evaluation import (
    EvaluateAll,
    FidelityPromotion,
    RepeatedEvaluation,
)
from saealib.policies.feedback import TrueOnlyFeedback
from saealib.population import Archive, ParetoArchive, Population, PopulationAttribute
from saealib.problem import Problem
from saealib.stages import (
    ArchiveUpdateStage,
    AsyncEvaluationSubmitStage,
    EvaluationAcknowledgeStage,
    EvaluationApplyStage,
    EvaluationCollectStage,
    EvaluationPlanStage,
    EvaluationSubmitStage,
    FeedbackStage,
    TellStage,
)
from saealib.strategies import DirectStrategy
from saealib.surrogate.prediction import PredictionChannel, SurrogatePrediction

ATTRS = [
    PopulationAttribute("id", np.int64, (), -1),
    PopulationAttribute("x", np.float64, (1,)),
    PopulationAttribute("f", np.float64, (1,)),
    PopulationAttribute("g", np.float64, (0,)),
    PopulationAttribute("cv", np.float64, (), 0.0),
]


def _proposal_for_state(state: StateView) -> ProposalBatch:
    """Return the current test population through the canonical ask boundary."""
    context = state.context
    candidates = context.offspring
    if candidates is None:
        raise AssertionError("test proposal state has no offspring")
    return ProposalBatch.from_allocator(
        context.proposal_id_allocator,
        candidates=candidates,
        relations=ProposalRelations(row_count=len(candidates)),
        requirements=FeedbackRequirement(quantities=()),
    )


class SlowEvaluator(Evaluator):
    def evaluate_batch(self, x, problem):
        time.sleep(float(x[0, 0]))
        return SerialEvaluator().evaluate_batch(x, problem)


class FailingEvaluator(Evaluator):
    def evaluate_batch(self, x, problem):
        raise RuntimeError("evaluation failed")


class BareEvaluator(Evaluator):
    def evaluate_batch(self, x, problem):
        return SerialEvaluator().evaluate_batch(x, problem)


class FailSecondEvaluator(BareEvaluator):
    def __init__(self):
        self.calls = 0

    def submit(self, request, problem):
        self.calls += 1
        if self.calls == 2:
            raise RuntimeError("second submission failed")
        return EvaluationHandle(request.request_id, EvaluationStatus.PENDING)


class ReattachEvaluator(Evaluator):
    def __init__(self):
        self.acknowledged = []

    def evaluate_batch(self, x, problem):
        return SerialEvaluator().evaluate_batch(x, problem)

    def submit(self, request, problem):
        return EvaluationHandle(
            request.request_id,
            EvaluationStatus.PENDING,
            backend_token=(request, problem),
        )

    def collect(self, handle, *, wait=True):
        if handle._acknowledged_sequence >= 0:
            return []
        request, problem = handle.backend_token
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

    def acknowledge(self, handle, sequence):
        self.acknowledged.append((int(handle.request_id), sequence))
        handle._acknowledged_sequence = sequence

    def can_reattach(self, pending):
        return True

    def reattach(self, pending, problem):
        return EvaluationHandle(
            pending.request.request_id,
            EvaluationStatus.PENDING,
            backend_token=(pending.request, problem),
        )


class ControlledReplicateEvaluator(ReattachEvaluator):
    def __init__(self):
        super().__init__()
        self.release_id = 0
        self.submitted_ids = []

    def submit(self, request, problem):
        self.submitted_ids.append(int(request.request_id))
        return super().submit(request, problem)

    def supports_batch_rollback(self):
        return True

    def collect(self, handle, *, wait=True):
        if int(handle.request_id) != self.release_id:
            return []
        return super().collect(handle, wait=wait)


class ControlledFidelityEvaluator(ReattachEvaluator):
    def __init__(self):
        super().__init__()
        self.released = set()
        self.submitted = []

    def submit(self, request, problem):
        self.submitted.append(request)
        return super().submit(request, problem)

    def collect(self, handle, *, wait=True):
        if int(handle.request_id) not in self.released:
            return []
        return super().collect(handle, wait=wait)


class FidelityValueEvaluator(ReattachEvaluator):
    def collect(self, handle, *, wait=True):
        return _add_fidelity_value(super().collect(handle, wait=wait), handle)


class ControlledFidelityValueEvaluator(ControlledFidelityEvaluator):
    def collect(self, handle, *, wait=True):
        return _add_fidelity_value(super().collect(handle, wait=wait), handle)


def _add_fidelity_value(updates, handle):
    if not updates:
        return updates
    request, _ = handle.backend_token
    fidelity = float(request.metadata.get("fidelity", 0))
    if fidelity == 0:
        return updates
    update = updates[0]
    assert update.result is not None
    result = EvaluationResult(
        update.result.f + fidelity * 10.0,
        update.result.g,
        update.result.cv,
        candidate_ids=update.candidate_ids,
    )
    return [
        EvaluationUpdate(
            update.request_id,
            update.status,
            update.candidate_ids,
            result,
            update.error,
            update.sequence,
        )
    ]


class PartialRetryEvaluator(Evaluator):
    def __init__(self):
        self.attempts = 0
        self.acks = []
        self.collected = []

    def evaluate_batch(self, x, problem):
        return SerialEvaluator().evaluate_batch(x, problem)

    def submit(self, request, problem):
        handle = EvaluationHandle(
            request.request_id,
            EvaluationStatus.PENDING,
            backend_token=(request, problem, self.attempts),
        )
        self.attempts += 1
        return handle

    def collect(self, handle, *, wait=True):
        request, problem, attempt = handle.backend_token
        self.collected.append((attempt, request.candidate_ids.tolist()))
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

    def acknowledge(self, handle, sequence):
        self.acks.append((handle.backend_token[2], sequence))
        handle._acknowledged_sequence = sequence


class OrderedPartialEvaluator(PartialRetryEvaluator):
    """Deliver partial and final updates in one request with ordered sequences."""

    def collect(self, handle, *, wait=True):
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


def make_state():
    problem = Problem(
        func=lambda x: np.array([x[0]]),
        dim=1,
        n_obj=1,
        direction=np.array([-1.0]),
        lb=[0.0],
        ub=[1.0],
    )
    population = Population(ATTRS, 2)
    population._extend_internal(
        {
            "id": np.array([10, 11], dtype=np.int64),
            "x": np.array([[0.2], [0.1]], dtype=np.float64),
            "f": np.full((2, 1), np.nan),
            "g": np.empty((2, 0)),
            "cv": np.zeros(2),
        },
        preserve_ids=True,
    )
    state = OptimizationState(
        problem=problem,
        population=population,
        archive=Archive(ATTRS, 2),
        pareto_archive=ParetoArchive(ATTRS, 2, direction=np.array([-1.0])),
        rng=np.random.default_rng(0),
        offspring=population,
    )
    state.set_state(PROPOSALS_CURRENT, 0)
    return state


def requests():
    return [
        EvaluationRequest(
            np.int64(0), np.array([10], dtype=np.int64), np.array([[0.2]])
        ),
        EvaluationRequest(
            np.int64(1), np.array([11], dtype=np.int64), np.array([[0.1]])
        ),
    ]


def test_async_out_of_order_and_nonblocking_poll():
    state = make_state()
    scheduler = AsyncEvaluationScheduler(
        AsyncEvaluator(SlowEvaluator(), max_workers=2), max_pending=2
    )
    state = scheduler.submit(state, requests())
    pending_result = scheduler.poll_result(state, wait=False)
    assert pending_result.state is state
    assert not pending_result.progressed
    completed_result = scheduler.poll_result(state, wait=True)
    assert completed_result.progressed
    state = completed_result.state
    assert state.pending_evaluations == {}
    np.testing.assert_array_equal(state.archive.id, [11, 10])
    assert state.fe == 2


def test_submit_prediction_is_submit_time_snapshot():
    state = make_state()
    prediction = SurrogatePrediction.objective(
        value=np.array([[1.0], [2.0]]),
        std=np.ones((2, 1)),
        x=np.array([[0.2], [0.1]]),
    )
    state = state.replace(predictions=prediction)
    scheduler = AsyncEvaluationScheduler(
        AsyncEvaluator(SlowEvaluator(), max_workers=2), max_pending=2
    )
    state = scheduler.submit(state, requests())
    submitted = [pending.prediction for pending in state.pending_evaluations.values()]
    assert all(
        submitted_prediction is not prediction for submitted_prediction in submitted
    )
    prediction.channels["objective"].value[:] += 100.0
    prediction.channels["objective"].std[:] += 100.0
    if prediction.x is not None:
        prediction.x[:] += 100.0
    for submitted_prediction in submitted:
        np.testing.assert_array_equal(submitted_prediction.value, [[1.0], [2.0]])
        np.testing.assert_array_equal(submitted_prediction.std, [[1.0], [1.0]])
        np.testing.assert_array_equal(submitted_prediction.x, [[0.2], [0.1]])


def test_repeated_plan_waits_for_all_requests_before_tell():
    class Algorithm:
        def __init__(self):
            self.told = []

        def contract(self):
            return ComponentContract(
                lifecycle=LifecycleContract(
                    feedback=FeedbackContract(accepted_channels=frozenset({"true"}))
                )
            )

        def tell(self, feedback: FeedbackBatch, view: StateView) -> StatePatch:
            del feedback
            self.told.extend(view.context.offspring.get_array("id").tolist())
            return StatePatch(writes={})

    state = make_state()
    evaluator = ControlledReplicateEvaluator()
    algorithm = Algorithm()
    scheduler = AsyncEvaluationScheduler(
        evaluator,
        max_pending=2,
        feedback_builder=TrueOnlyFeedback(),
        algorithm=algorithm,
    )
    plan = RepeatedEvaluation(2).plan(state.offspring, None, state)
    state = state.replace(
        evaluation_plan=plan,
        evaluation_plan_state=EvaluationPlanState(
            submitted=tuple(int(request.request_id) for request in plan.requests),
            deferred=(),
        ),
    )
    state = scheduler.submit(state, plan.requests)
    state = scheduler.poll(state, wait=False)
    assert state.pending_evaluations.keys() == {1}
    assert state.evaluation_plan is plan
    assert state.evaluation_plan_state is not None
    assert state.evaluation_plan_state.completed == (0,)
    assert algorithm.told == []

    evaluator.release_id = 1
    state = scheduler.poll(state, wait=True)
    assert state.pending_evaluations == {}
    assert algorithm.told == [10, 11]
    assert state.evaluation_plan_state is not None
    assert state.evaluation_plan_state.completed == (0, 1)
    assert state.fe == 4


def test_async_callback_runs_after_tell():
    order = []

    class Algorithm:
        def tell(self, feedback: FeedbackBatch, view: StateView) -> StatePatch:
            del feedback, view
            order.append("tell")
            return StatePatch(writes={})

    class Callback:
        def dispatch(self, event):
            order.append("event")

    state = make_state()
    scheduler = AsyncEvaluationScheduler(
        ReattachEvaluator(),
        algorithm=Algorithm(),
        feedback_builder=TrueOnlyFeedback(),
        callback_manager=Callback(),
    )
    state = scheduler.submit(state, [requests()[0]])
    scheduler.poll(state, wait=True)

    assert order == ["tell", "event"]


def test_repeated_sync_plan_uses_the_same_terminal_lifecycle():
    class Algorithm:
        def __init__(self):
            self.tell_count = 0

        def tell(self, feedback: FeedbackBatch, view: StateView) -> StatePatch:
            del feedback, view
            self.tell_count += 1
            return StatePatch(writes={})

    state = make_state()
    algorithm = Algorithm()
    evaluator = ReattachEvaluator()
    state = EvaluationPlanStage(RepeatedEvaluation(2)).execute(state)
    state = EvaluationSubmitStage(evaluator).execute(state)
    state = EvaluationCollectStage(evaluator).execute(state)
    state = EvaluationApplyStage().execute(state)
    state = ArchiveUpdateStage().execute(state)
    state = FeedbackStage(TrueOnlyFeedback()).execute(state)
    state = TellStage(cast(Any, algorithm)).execute(state)
    state = EvaluationAcknowledgeStage(evaluator).execute(state)

    assert algorithm.tell_count == 1
    assert state.fe == 4
    assert state.evaluation_plan is None


def test_sync_fidelity_promotion_delays_feedback_until_final_request():
    class Algorithm:
        def __init__(self):
            self.told = []

        def contract(self):
            return ComponentContract(
                lifecycle=LifecycleContract(
                    feedback=FeedbackContract(accepted_channels=frozenset({"true"}))
                )
            )

        def tell(self, feedback: FeedbackBatch, view: StateView) -> StatePatch:
            del feedback
            self.told.append(view.context.offspring.get_array("id").tolist())
            return StatePatch(writes={})

    state = make_state()
    evaluator = FidelityValueEvaluator()
    algorithm = Algorithm()
    planner = FidelityPromotion(0, 1, promotion_count=1)
    state = EvaluationPlanStage(planner).execute(state)
    low = state.evaluation_request
    assert low is not None
    state = EvaluationSubmitStage(evaluator).execute(state)
    state = EvaluationCollectStage(evaluator).execute(state)
    assert state.evaluation_updates == []
    assert algorithm.told == []

    state = EvaluationPlanStage(planner).execute(state)
    assert state.evaluation_plan is not None
    high = state.evaluation_plan.requests[1]
    assert int(high.request_id) != int(low.request_id)
    np.testing.assert_array_equal(high.candidate_ids, [11])
    state = EvaluationSubmitStage(evaluator).execute(state)
    state = EvaluationCollectStage(evaluator).execute(state)
    assert state.evaluation_plan_state is not None
    assert len(state.evaluation_updates) == 1
    assert algorithm.told == []

    for stage in (
        EvaluationApplyStage(),
        ArchiveUpdateStage(),
        FeedbackStage(TrueOnlyFeedback()),
        TellStage(cast(Any, algorithm)),
        EvaluationAcknowledgeStage(evaluator),
    ):
        state = stage.execute(state)

    assert len(algorithm.told) == 1
    assert state.evaluation_plan is None
    assert state.pending_evaluations == {}
    assert state.evaluation_handles == {}
    assert sorted(evaluator.acknowledged) == [(0, 0), (1, 0)]
    assert state.offspring is not None
    np.testing.assert_allclose(state.offspring.get_array("f"), [[0.2], [10.1]])
    high_row = int(np.flatnonzero(state.archive.id == 11)[0])
    np.testing.assert_allclose(state.archive.f[high_row], [10.1])


def test_async_fidelity_promotion_uses_plan_continuation_and_checkpoint(tmp_path):
    order = []

    class Algorithm:
        def tell(self, feedback: FeedbackBatch, view: StateView) -> StatePatch:
            del feedback
            order.append(("tell", view.context.offspring.get_array("id").tolist()))
            return StatePatch(writes={})

    class Callback:
        def dispatch(self, event):
            order.append(("event", event.candidate_ids.tolist()))

    state = make_state()
    evaluator = ControlledFidelityValueEvaluator()
    algorithm = Algorithm()
    planner = FidelityPromotion(0, 1, promotion_count=1)
    scheduler = AsyncEvaluationScheduler(
        evaluator,
        max_pending=1,
        feedback_builder=TrueOnlyFeedback(),
        algorithm=algorithm,
        callback_manager=Callback(),
    )
    state = AsyncEvaluationSubmitStage(
        scheduler,
        planner,
        feedback_builder=TrueOnlyFeedback(),
        algorithm=algorithm,
        callback_manager=Callback(),
    ).execute(state)
    low = evaluator.submitted[0]
    evaluator.released.add(int(low.request_id))
    state = scheduler.poll(state, wait=True)

    assert order == []
    assert state.evaluation_plan is not None
    assert state.evaluation_plan_state is not None
    assert state.evaluation_plan_state.completed == (int(low.request_id),)
    assert not state.pending_evaluations

    state = AsyncEvaluationSubmitStage(
        scheduler,
        planner,
        feedback_builder=TrueOnlyFeedback(),
        algorithm=algorithm,
        callback_manager=Callback(),
    ).execute(state)
    high = evaluator.submitted[1]
    assert int(high.request_id) != int(low.request_id)
    assert high.metadata["promotion_of"] == int(low.request_id)
    np.testing.assert_array_equal(high.candidate_ids, [11])
    assert state.evaluation_plan is not None
    assert [int(request.request_id) for request in state.evaluation_plan.requests] == [
        int(low.request_id),
        int(high.request_id),
    ]
    assert state.evaluation_plan_state is not None
    assert set(state.evaluation_plan_state.submitted) == {
        int(low.request_id),
        int(high.request_id),
    }

    checkpoint = tmp_path / "fidelity.npz"
    state.save(checkpoint)
    state = OptimizationState.load(checkpoint, state.problem)
    state = scheduler.reattach(state)
    assert [int(request.request_id) for request in evaluator.submitted] == [
        int(low.request_id),
        int(high.request_id),
    ]
    evaluator.released.add(int(high.request_id))
    state = scheduler.poll(state, wait=True)

    assert order[0][0] == "tell"
    assert order[1][0] == "event"
    assert len(order) == 2
    assert state.evaluation_plan_state is not None
    assert set(state.evaluation_plan_state.completed) == {
        int(low.request_id),
        int(high.request_id),
    }
    assert state.offspring is not None
    np.testing.assert_allclose(state.offspring.get_array("f"), [[0.2], [10.1]])
    high_row = int(np.flatnonzero(state.archive.id == 11)[0])
    np.testing.assert_allclose(state.archive.f[high_row], [10.1])
    assert sorted(evaluator.acknowledged) == [
        (int(low.request_id), 0),
        (int(high.request_id), 0),
    ]


def test_async_chunk_ids_are_owned_by_the_plan_until_all_complete(tmp_path):
    state = make_state()
    population = state.offspring.empty_like(capacity=4)
    population._extend_internal(
        {
            "id": np.arange(10, 14, dtype=np.int64),
            "x": np.arange(4, dtype=np.float64).reshape(-1, 1),
            "f": np.full((4, 1), np.nan),
            "g": np.empty((4, 0)),
            "cv": np.zeros(4),
        },
        preserve_ids=True,
    )
    state = state.replace(population=population, offspring=population)
    evaluator = ControlledReplicateEvaluator()
    scheduler = AsyncEvaluationScheduler(evaluator, max_pending=2)
    state = AsyncEvaluationSubmitStage(scheduler, EvaluateAll()).execute(state)

    assert state.evaluation_plan is not None
    plan_ids = tuple(
        int(request.request_id) for request in state.evaluation_plan.requests
    )
    assert len(plan_ids) == 2
    assert state.evaluation_plan_state is not None
    assert state.evaluation_plan_state.submitted == plan_ids
    assert state.evaluation_plan_state.deferred == ()

    state = scheduler.poll(state, wait=False)
    assert state.evaluation_plan is not None
    assert state.evaluation_plan_state is not None
    assert state.evaluation_plan_state.completed == (0,)
    assert tuple(state.pending_evaluations) == (1,)

    checkpoint = tmp_path / "chunks.npz"
    state.save(checkpoint)
    state = OptimizationState.load(checkpoint, state.problem)
    assert state.evaluation_plan is not None
    assert state.evaluation_plan_state is not None
    assert state.evaluation_plan_state.completed == (0,)
    state = scheduler.reattach(state)
    evaluator.release_id = 1
    state = scheduler.poll(state, wait=True)
    assert state.evaluation_plan is not None
    assert state.evaluation_plan_state is not None
    assert state.evaluation_plan_state.completed == (0, 1)


def test_async_new_plan_increments_decision_count():
    state = make_state()
    assert state.decision_count == 0
    evaluator = ControlledReplicateEvaluator()
    scheduler = AsyncEvaluationScheduler(evaluator, max_pending=2)
    state = AsyncEvaluationSubmitStage(scheduler, EvaluateAll()).execute(state)
    assert state.decision_count == 1


def test_async_repeated_execute_on_same_plan_does_not_increment_decision_count():
    state = make_state()
    evaluator = ControlledReplicateEvaluator()
    scheduler = AsyncEvaluationScheduler(evaluator, max_pending=1)
    stage = AsyncEvaluationSubmitStage(scheduler, RepeatedEvaluation(2))

    state = stage.execute(state)
    assert state.decision_count == 1
    state = scheduler.poll(state, wait=True)
    assert state.evaluation_plan is not None
    assert state.evaluation_plan_state is not None
    terminal = set(state.evaluation_plan_state.completed) | set(
        state.evaluation_plan_state.acknowledged
    )
    assert not all(
        int(request.request_id) in terminal
        for request in state.evaluation_plan.requests
    )

    state = stage.execute(state)
    assert state.decision_count == 1


def test_deferred_replicate_split_preserves_completed_plan_history(tmp_path):
    state = make_state()
    evaluator = ControlledReplicateEvaluator()
    scheduler = AsyncEvaluationScheduler(evaluator, max_pending=2)
    state = AsyncEvaluationSubmitStage(scheduler, RepeatedEvaluation(3)).execute(state)
    assert state.evaluation_plan is not None
    assert [int(request.request_id) for request in state.evaluation_plan.requests] == [
        0,
        1,
        2,
    ]

    state = scheduler.poll(state, wait=False)
    evaluator.release_id = 1
    state = scheduler.poll(state, wait=True)
    assert state.evaluation_plan is not None
    assert state.evaluation_plan_state is not None
    assert state.evaluation_plan_state.completed == (0, 1)
    assert set(state.evaluation_plan_updates) == {0, 1}

    checkpoint = tmp_path / "deferred-replicate.npz"
    state.save(checkpoint)
    state = OptimizationState.load(checkpoint, state.problem)
    state = AsyncEvaluationSubmitStage(scheduler, RepeatedEvaluation(3)).execute(state)
    assert state.evaluation_plan is not None
    plan_ids = [int(request.request_id) for request in state.evaluation_plan.requests]
    assert plan_ids[:2] == [0, 1]
    assert len(plan_ids) == 4
    assert set(state.evaluation_plan_updates) == {0, 1}
    assert state.evaluation_plan_state is not None
    assert set(state.evaluation_plan_state.completed) == {0, 1}
    assert set(state.evaluation_plan_state.submitted) == {0, 1, 2, 3}

    assert sorted(state.pending_evaluations) == plan_ids[2:]
    assert evaluator.submitted_ids == [0, 1, 2, 3]
    remaining_ids = sorted(state.pending_evaluations)
    evaluator.release_id = int(remaining_ids[0])
    state = scheduler.poll(state, wait=False)
    assert sorted(state.pending_evaluations) == remaining_ids[1:]
    evaluator.release_id = int(remaining_ids[1])
    state = scheduler.poll(state, wait=True)
    assert state.evaluation_plan is not None
    assert state.evaluation_plan_state is not None
    assert set(state.evaluation_plan_state.completed) == {0, 1, 2, 3}
    assert set(state.evaluation_plan_updates) == {0, 1, 2, 3}


def test_completed_futures_commit_by_completion_time():
    state = make_state()
    evaluator = AsyncEvaluator(SlowEvaluator(), max_workers=2)
    scheduler = AsyncEvaluationScheduler(evaluator, max_pending=2)
    try:
        state = scheduler.submit(state, requests())
        time.sleep(0.25)
        state = scheduler.poll(state, wait=False)
        np.testing.assert_array_equal(state.archive.id, [11, 10])
    finally:
        evaluator.close()


def test_async_capacity_and_reserved_budget():
    state = make_state()
    evaluator = AsyncEvaluator(SlowEvaluator(), max_workers=2)
    scheduler = AsyncEvaluationScheduler(evaluator, max_pending=1, max_reserved_fe=1)
    state = scheduler.submit(state, [requests()[0]])
    try:
        scheduler.submit(state, [requests()[1]])
    except Exception as exc:
        assert "capacity" in str(exc)
    else:
        raise AssertionError("capacity was not enforced")
    evaluator.close()


def test_batch_submit_rolls_back_started_handles_after_midway_failure():
    class MidwayFailureEvaluator(ReattachEvaluator):
        def __init__(self):
            super().__init__()
            self.submit_calls = 0
            self.cancelled = []

        def supports_batch_rollback(self):
            return True

        def submit(self, request, problem):
            self.submit_calls += 1
            if self.submit_calls == 2:
                raise RuntimeError("second submission failed")
            return super().submit(request, problem)

        def cancel(self, handle):
            self.cancelled.append(int(handle.request_id))
            return True

    state = make_state()
    evaluator = MidwayFailureEvaluator()
    scheduler = AsyncEvaluationScheduler(evaluator, max_pending=2)
    with pytest.raises(RuntimeError, match="second submission failed"):
        scheduler.submit(state, requests())
    assert evaluator.cancelled == [0]
    assert state.pending_evaluations == {}
    assert state.evaluation_handles == {}


def test_batch_submit_reports_started_handles_when_cleanup_fails():
    class UncleanableMidwayFailureEvaluator(ReattachEvaluator):
        def __init__(self):
            super().__init__()
            self.submit_calls = 0
            self.cancelled = []
            self.detached = []

        def supports_batch_rollback(self):
            return True

        def submit(self, request, problem):
            self.submit_calls += 1
            if self.submit_calls == 2:
                raise RuntimeError("second submission failed")
            return super().submit(request, problem)

        def cancel(self, handle):
            self.cancelled.append(int(handle.request_id))
            return False

        def detach(self, handle):
            self.detached.append(int(handle.request_id))
            return False

    state = make_state()
    evaluator = UncleanableMidwayFailureEvaluator()
    scheduler = AsyncEvaluationScheduler(evaluator, max_pending=2)
    with pytest.raises(EvaluationSubmissionError, match="cannot be cleaned up") as exc:
        scheduler.submit(state, requests())
    assert evaluator.cancelled == [0]
    assert evaluator.detached == [0]
    assert set(exc.value.state.pending_evaluations) == {0}
    assert set(exc.value.state.evaluation_handles) == {0}


def test_cancel_succeeds_and_removes_pending_request():
    class CancellingEvaluator(ReattachEvaluator):
        def cancel(self, handle):
            return True

    state = make_state()
    evaluator = CancellingEvaluator()
    scheduler = AsyncEvaluationScheduler(evaluator)
    state = scheduler.submit(state, [requests()[0]])
    state = scheduler.cancel(state, 0)
    assert state.pending_evaluations == {}
    assert state.evaluation_handles == {}
    assert state.fe == 0


def test_cancel_rejects_backend_failure_and_unregistered_request():
    class RejectCancelEvaluator(ReattachEvaluator):
        def cancel(self, handle):
            return False

    state = make_state()
    scheduler = AsyncEvaluationScheduler(RejectCancelEvaluator())
    state = scheduler.submit(state, [requests()[0]])
    with pytest.raises(EvaluationProtocolError, match="cannot be cancelled"):
        scheduler.cancel(state, 0)
    assert set(state.pending_evaluations) == {0}

    with pytest.raises(EvaluationProtocolError, match="not pending"):
        scheduler.cancel(state, 99)


def test_timeout_cancels_backend_work():
    class CancellingEvaluator(ReattachEvaluator):
        def __init__(self):
            super().__init__()
            self.cancelled = []

        def cancel(self, handle):
            self.cancelled.append(int(handle.request_id))
            return True

    state = make_state()
    evaluator = CancellingEvaluator()
    scheduler = AsyncEvaluationScheduler(evaluator, timeout=0)
    state = scheduler.submit(state, [requests()[0]])
    state = scheduler.poll(state, wait=False)
    assert evaluator.cancelled == [0]
    assert state.pending_evaluations == {}
    assert state.evaluation_handles == {}
    assert state.fe == 0


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_pending": 0}, "max_pending"),
        ({"max_reserved_fe": -1}, "max_reserved_fe"),
        ({"max_reserved_cost": -1}, "max_reserved_cost"),
    ],
)
def test_scheduler_rejects_invalid_capacity_limits(kwargs, message):
    with pytest.raises(ValidationError, match=message):
        AsyncEvaluationScheduler(ReattachEvaluator(), **kwargs)


@pytest.mark.parametrize("estimated_cost", [-1.0, np.nan, np.inf])
def test_submit_rejects_invalid_estimated_cost(estimated_cost):
    request = EvaluationRequest(
        np.int64(0),
        np.array([10], dtype=np.int64),
        np.array([[0.2]]),
        metadata={"estimated_cost": estimated_cost},
    )
    with pytest.raises(ValidationError, match="estimated_cost"):
        AsyncEvaluationScheduler(ReattachEvaluator()).submit(make_state(), [request])


def test_submit_rejects_reserved_cost_overflow():
    request = EvaluationRequest(
        np.int64(0),
        np.array([10], dtype=np.int64),
        np.array([[0.2]]),
        metadata={"estimated_cost": 1.0},
    )
    scheduler = AsyncEvaluationScheduler(ReattachEvaluator(), max_reserved_cost=0.5)
    with pytest.raises(EvaluationProtocolError, match="reserved cost"):
        scheduler.submit(make_state(), [request])


def test_partial_failure_is_terminal_when_retry_limit_is_exhausted():
    state = make_state()
    request = EvaluationRequest(
        np.int64(0),
        np.array([10, 11], dtype=np.int64),
        np.array([[0.2], [0.1]], dtype=np.float64),
    )
    evaluator = PartialRetryEvaluator()
    scheduler = AsyncEvaluationScheduler(evaluator, retry_limit=0)
    state = scheduler.submit(state, [request])
    state = scheduler.poll(state, wait=True)
    assert evaluator.attempts == 1
    assert evaluator.collected == [(0, [10, 11])]
    assert state.pending_evaluations == {}
    assert state.fe == 1
    np.testing.assert_array_equal(state.archive.id, [10])


def test_request_id_collision_is_rejected_before_state_changes():
    state = make_state()
    evaluator = ReattachEvaluator()
    scheduler = AsyncEvaluationScheduler(evaluator)
    state = scheduler.submit(state, [requests()[0]])
    before_pending = dict(state.pending_evaluations)
    before_handles = dict(state.evaluation_handles)
    with pytest.raises(EvaluationProtocolError, match="already pending"):
        scheduler.submit(state, [requests()[0]])
    assert state.pending_evaluations == before_pending
    assert state.evaluation_handles == before_handles


def test_async_checkpoint_requires_reattach(tmp_path):
    state = make_state()
    evaluator = AsyncEvaluator(SlowEvaluator())
    scheduler = AsyncEvaluationScheduler(evaluator)
    state = scheduler.submit(state, [requests()[0]])
    try:
        try:
            scheduler.checkpoint(state, str(tmp_path / "pending.npz"))
        except CheckpointError as exc:
            assert "reattach" in str(exc)
        else:
            raise AssertionError("unreattachable checkpoint was accepted")
        try:
            state.save(tmp_path / "direct.npz")
        except Exception as exc:
            assert "synchronous" in str(exc)
        else:
            raise AssertionError("direct checkpoint accepted pending work")
    finally:
        evaluator.close()


def test_reattachable_pending_checkpoint_resumes_once(tmp_path):
    state = make_state()
    evaluator = ReattachEvaluator()
    scheduler = AsyncEvaluationScheduler(evaluator)
    state = scheduler.submit(state, [requests()[0]])
    path = tmp_path / "pending.npz"
    state.save(path)
    restored = OptimizationState.load(path, state.problem)
    assert restored.pending_evaluations[0].request.candidate_ids.tolist() == [10]
    restored = scheduler.reattach(restored)
    restored = scheduler.poll(restored, wait=True)
    assert restored.pending_evaluations == {}
    assert restored.fe == 1
    assert evaluator.acknowledged == [(0, 0)]
    assert restored.offspring is not None
    np.testing.assert_allclose(restored.offspring.f[0], [0.2])
    np.testing.assert_array_equal(restored.archive.id, [10])


def test_timeout_detaches_running_future_without_applying_result():
    state = make_state()
    evaluator = AsyncEvaluator(SlowEvaluator())
    scheduler = AsyncEvaluationScheduler(evaluator, timeout=0.001)
    try:
        state = scheduler.submit(state, [requests()[0]])
        time.sleep(0.01)
        state = scheduler.poll(state, wait=False)
        assert state.pending_evaluations == {}
        assert state.fe == 0
        assert len(state.archive) == 0
    finally:
        evaluator.close()


def test_async_evaluation_exception_becomes_failed_update():
    state = make_state()
    evaluator = AsyncEvaluator(FailingEvaluator())
    try:
        scheduler = AsyncEvaluationScheduler(evaluator)
        state = scheduler.submit(state, [requests()[0]])
        state = scheduler.poll(state, wait=True)
        assert state.pending_evaluations == {}
        assert state.fe == 0
    finally:
        evaluator.close()


def test_timeout_requires_termination_capability():
    try:
        AsyncEvaluationScheduler(BareEvaluator(), timeout=0.01)
    except Exception as exc:
        assert "timeout" in str(exc)
    else:
        raise AssertionError("unsupported timeout was accepted")


def test_batch_submit_requires_rollback_capability_before_side_effects():
    evaluator = FailSecondEvaluator()
    scheduler = AsyncEvaluationScheduler(evaluator, max_pending=2)
    state = make_state()
    try:
        scheduler.submit(state, requests())
    except Exception as exc:
        assert "rollback" in str(exc)
        assert evaluator.calls == 0
    else:
        raise AssertionError("non-rollback batch submission was accepted")


def test_direct_strategy_uses_scheduler_for_submit_and_poll():
    class Algorithm:
        def __init__(self):
            self.tell_ids = []

        def ask(self, request, view):
            del request
            return _proposal_for_state(view)

        def tell(self, feedback: FeedbackBatch, view: StateView) -> StatePatch:
            del feedback
            self.tell_ids.extend(view.context.offspring.get_array("id").tolist())
            return StatePatch(writes={})

    state = make_state()
    algorithm = Algorithm()
    evaluator = AsyncEvaluator(SlowEvaluator(), max_workers=2)
    scheduler = AsyncEvaluationScheduler(evaluator, max_pending=2)
    provider = SimpleNamespace(
        algorithm=algorithm,
        evaluator=evaluator,
        evaluation_planner=None,
        feedback_builder=None,
        async_evaluation_scheduler=scheduler,
        cbmanager=None,
    )
    try:
        strategy = DirectStrategy()
        pending = strategy.step(state, cast(Any, provider))
        assert len(pending.pending_evaluations) == 2
        while pending.pending_evaluations:
            time.sleep(0.3)
            pending = strategy.step(pending, cast(Any, provider))
        assert sorted(algorithm.tell_ids) == [10, 11]
        assert pending.fe == 2
    finally:
        evaluator.close()


def test_partial_failure_retries_only_unapplied_candidates():
    class Algorithm:
        def __init__(self):
            self.told = []

        def contract(self):
            return ComponentContract(
                lifecycle=LifecycleContract(
                    feedback=FeedbackContract(accepted_channels=frozenset({"true"}))
                )
            )

        def tell(self, feedback: FeedbackBatch, view: StateView) -> StatePatch:
            del feedback
            self.told.extend(view.context.offspring.get_array("id").tolist())
            return StatePatch(writes={})

    state = make_state()
    request = EvaluationRequest(
        np.int64(0),
        np.array([10, 11], dtype=np.int64),
        np.array([[0.2], [0.1]], dtype=np.float64),
    )
    evaluator = PartialRetryEvaluator()
    algorithm = Algorithm()
    scheduler = AsyncEvaluationScheduler(
        evaluator,
        retry_limit=1,
        feedback_builder=TrueOnlyFeedback(),
        algorithm=algorithm,
    )
    scheduler.enable_feedback_accumulator()
    state = scheduler.submit(state, [request])
    state = scheduler.poll(state, wait=True)
    assert state.pending_evaluations == {}
    assert evaluator.attempts == 2
    assert sorted(algorithm.told) == [10, 11]
    assert evaluator.collected == [(0, [10, 11]), (1, [11])]
    assert state.fe == 2
    np.testing.assert_array_equal(np.sort(state.archive.id), [10, 11])


class _RecordingConsumer:
    """Complete-batch consumer used by the accumulator retry regression."""

    def __init__(self):
        self.feedback: list[FeedbackBatch] = []

    def contract(self):
        return ComponentContract(
            lifecycle=LifecycleContract(
                feedback=FeedbackContract(accepted_channels=frozenset({"true"}))
            )
        )

    def tell(self, feedback: FeedbackBatch, state: StateView) -> StatePatch:
        self.feedback.append(feedback)
        return StatePatch(writes={})


class _PartialRecordingConsumer:
    """Partial/repeated consumer for the real scheduler delivery path."""

    def __init__(self):
        self.feedback: list[FeedbackBatch] = []

    def contract(self):
        return ComponentContract(
            lifecycle=LifecycleContract(
                feedback=FeedbackContract(
                    accepted_channels=frozenset({"true"}),
                    completion=PARTIAL_ALLOWED,
                    multiplicity=REPEATED_ALLOWED,
                )
            )
        )

    def tell(self, feedback: FeedbackBatch, state: StateView) -> StatePatch:
        self.feedback.append(feedback)
        return StatePatch(writes={})


def test_partial_repeated_feedback_is_delivered_directly_by_async_scheduler():
    state = make_state()
    state.set_state(PROPOSALS_CURRENT, 703)
    consumer = _PartialRecordingConsumer()
    scheduler = AsyncEvaluationScheduler(
        PartialRetryEvaluator(),
        retry_limit=1,
        feedback_builder=TrueOnlyFeedback(),
        algorithm=consumer,
    )
    contract = consumer.contract().lifecycle.feedback
    assert contract is not None
    assert contract.completion == PARTIAL_ALLOWED
    assert contract.ordering == IN_ORDER
    assert contract.multiplicity == REPEATED_ALLOWED

    state = scheduler.poll(
        scheduler.submit(
            state,
            [
                EvaluationRequest(
                    np.int64(0),
                    np.array([10, 11], dtype=np.int64),
                    np.array([[0.2], [0.1]], dtype=np.float64),
                )
            ],
        ),
        wait=True,
    )

    assert scheduler._feedback_accumulator is None
    assert state.pending_evaluations == {}
    assert [
        (batch.proposal_id, batch.final, batch.sequence) for batch in consumer.feedback
    ] == [(703, False, 0), (703, True, 1)]
    observed = [
        set(
            np.asarray(
                batch.observations.records.column("subject_payload"), dtype=np.int64
            ).reshape(-1)
        )
        for batch in consumer.feedback
    ]
    assert observed == [{10}, {11}]


def test_ordered_partial_feedback_preserves_sequence_and_final_boundary():
    state = make_state()
    state.set_state(PROPOSALS_CURRENT, 704)
    consumer = _PartialRecordingConsumer()
    scheduler = AsyncEvaluationScheduler(
        OrderedPartialEvaluator(),
        feedback_builder=TrueOnlyFeedback(),
        algorithm=consumer,
    )

    state = scheduler.poll(
        scheduler.submit(
            state,
            [
                EvaluationRequest(
                    np.int64(0),
                    np.array([10, 11], dtype=np.int64),
                    np.array([[0.2], [0.1]], dtype=np.float64),
                )
            ],
        ),
        wait=True,
    )

    assert scheduler._feedback_accumulator is None
    assert state.pending_evaluations == {}
    assert [
        (batch.proposal_id, batch.final, batch.sequence) for batch in consumer.feedback
    ] == [(704, False, 0), (704, True, 1)]
    observed = [
        set(
            np.asarray(
                batch.observations.records.column("subject_payload"), dtype=np.int64
            ).reshape(-1)
        )
        for batch in consumer.feedback
    ]
    assert observed == [{10}, {11}]


def test_accumulator_partial_retry_tells_once_with_both_proposal_ids():
    state = make_state()
    state.set_state(PROPOSALS_CURRENT, 701)
    consumer = _RecordingConsumer()
    scheduler = AsyncEvaluationScheduler(
        PartialRetryEvaluator(),
        retry_limit=1,
        feedback_builder=TrueOnlyFeedback(),
        algorithm=consumer,
    )
    scheduler.enable_feedback_accumulator()
    request = EvaluationRequest(
        np.int64(0),
        np.array([10, 11], dtype=np.int64),
        np.array([[0.2], [0.1]], dtype=np.float64),
    )

    state = scheduler.poll(scheduler.submit(state, [request]), wait=True)

    assert state.pending_evaluations == {}
    assert len(consumer.feedback) == 1
    final = consumer.feedback[0]
    assert final.final is True
    assert final.proposal_id == 701
    subjects = final.observations.records.column("subject_payload")
    assert set(np.asarray(subjects, dtype=np.int64).reshape(-1)) == {10, 11}


def test_accumulator_exhausted_partial_failure_does_not_tell_incomplete_batch():
    state = make_state()
    state.set_state(PROPOSALS_CURRENT, 702)
    consumer = _RecordingConsumer()
    scheduler = AsyncEvaluationScheduler(
        PartialRetryEvaluator(),
        retry_limit=0,
        feedback_builder=TrueOnlyFeedback(),
        algorithm=consumer,
    )
    scheduler.enable_feedback_accumulator()
    request = EvaluationRequest(
        np.int64(0),
        np.array([10, 11], dtype=np.int64),
        np.array([[0.2], [0.1]], dtype=np.float64),
    )

    state = scheduler.poll(scheduler.submit(state, [request]), wait=True)

    assert state.pending_evaluations == {}
    assert len(consumer.feedback) == 0


def test_partial_retry_with_callback_keeps_applied_ids():
    class Algorithm:
        def __init__(self):
            self.told = []

        def contract(self):
            return ComponentContract(
                lifecycle=LifecycleContract(
                    feedback=FeedbackContract(accepted_channels=frozenset({"true"}))
                )
            )

        def tell(self, feedback: FeedbackBatch, view: StateView) -> StatePatch:
            del feedback
            self.told.extend(view.context.offspring.get_array("id").tolist())
            return StatePatch(writes={})

    events = []
    callback = CallbackManager()
    callback.register(PostEvaluationEvent, lambda event: events.append(event))
    state = make_state()
    evaluator = PartialRetryEvaluator()
    algorithm = Algorithm()
    scheduler = AsyncEvaluationScheduler(
        evaluator,
        retry_limit=1,
        feedback_builder=TrueOnlyFeedback(),
        algorithm=algorithm,
        callback_manager=callback,
    )
    scheduler.enable_feedback_accumulator()
    state = scheduler.submit(
        state,
        [
            EvaluationRequest(
                np.int64(0),
                np.array([10, 11], dtype=np.int64),
                np.array([[0.2], [0.1]], dtype=np.float64),
            )
        ],
    )
    state = scheduler.poll(state, wait=True)
    assert evaluator.collected == [(0, [10, 11]), (1, [11])]
    assert state.fe == 2
    assert sorted(algorithm.told) == [10, 11]
    assert [int(event.candidate_ids[0]) for event in events] == [10, 11]


def test_runner_drains_after_generation_termination():
    class Algorithm:
        def ask(self, request, view):
            del request
            return _proposal_for_state(view)

        def tell(self, feedback: FeedbackBatch, view: StateView) -> StatePatch:
            del feedback, view
            return StatePatch(writes={})

    state = make_state()
    evaluator = AsyncEvaluator(SlowEvaluator(), max_workers=2)
    scheduler = AsyncEvaluationScheduler(evaluator, max_pending=1)
    cbmanager = CallbackManager()
    events = []
    cbmanager.register(GenerationEndEvent, lambda event: events.append(event))
    optimizer = SimpleNamespace(
        strategy=DirectStrategy(),
        evaluator=evaluator,
        evaluation_planner=None,
        feedback_builder=None,
        feedback_builder_explicit=False,
        async_evaluation_scheduler=scheduler,
        algorithm=Algorithm(),
        cbmanager=cbmanager,
        surrogate_manager=None,
        termination=SimpleNamespace(is_terminated=lambda ctx: ctx.gen >= 1),
        dispatch=cbmanager.dispatch,
        problem=state.problem,
    )
    try:
        result = list(Runner(cast(Any, optimizer)).iterate_from(state))
        final = result[-1]
        assert final.fe == 2
        assert final.pending_evaluations == {}
        assert len(events) == 1
    finally:
        evaluator.close()


def test_runner_does_not_refill_after_termination_threshold():
    class CountingEvaluator(AsyncEvaluator):
        def __init__(self):
            super().__init__(SlowEvaluator(), max_workers=2)
            self.submit_count = 0

        def submit(self, request, problem):
            self.submit_count += 1
            return super().submit(request, problem)

    class Algorithm:
        def __init__(self):
            self.asks = 0

        def ask(self, request, view):
            del request
            self.asks += 1
            return _proposal_for_state(view)

        def tell(self, feedback: FeedbackBatch, view: StateView) -> StatePatch:
            del feedback, view
            return StatePatch(writes={})

    state = make_state()
    evaluator = CountingEvaluator()
    scheduler = AsyncEvaluationScheduler(evaluator, max_pending=2)
    state = scheduler.submit(state, requests())
    algorithm = Algorithm()
    optimizer = SimpleNamespace(
        strategy=DirectStrategy(),
        evaluator=evaluator,
        evaluation_planner=None,
        feedback_builder=None,
        async_evaluation_scheduler=scheduler,
        algorithm=algorithm,
        cbmanager=CallbackManager(),
        surrogate_manager=None,
        termination=SimpleNamespace(is_terminated=lambda ctx: ctx.fe >= 1),
        dispatch=lambda event: None,
        problem=state.problem,
    )
    try:
        final = Runner(cast(Any, optimizer)).run_from(state)
        assert final.fe == 2
        assert evaluator.submit_count == 2
        assert algorithm.asks == 0
    finally:
        evaluator.close()


def test_runner_reattaches_loaded_pending_state(tmp_path):
    class Algorithm:
        def ask(self, request, view):
            del request
            return _proposal_for_state(view)

        def tell(self, feedback: FeedbackBatch, view: StateView) -> StatePatch:
            del feedback, view
            return StatePatch(writes={})

    state = make_state()
    evaluator = ReattachEvaluator()
    scheduler = AsyncEvaluationScheduler(evaluator)
    state = scheduler.submit(state, [requests()[0]])
    path = tmp_path / "runner.npz"
    state.save(path)
    restored = OptimizationState.load(path, state.problem)
    optimizer = SimpleNamespace(
        strategy=DirectStrategy(),
        evaluator=evaluator,
        evaluation_planner=None,
        feedback_builder=None,
        feedback_builder_explicit=False,
        async_evaluation_scheduler=scheduler,
        algorithm=Algorithm(),
        cbmanager=CallbackManager(),
        surrogate_manager=None,
        termination=SimpleNamespace(is_terminated=lambda ctx: ctx.fe >= 1),
        dispatch=lambda event: None,
        problem=state.problem,
    )
    result = list(Runner(cast(Any, optimizer)).iterate_from(restored))
    assert result[-1].fe == 1
    assert result[-1].pending_evaluations == {}


def test_checkpoint_replays_buffered_update_without_backend_redelivery(tmp_path):
    state = make_state()
    result = SerialEvaluator().evaluate_batch(
        np.array([[0.2]], dtype=np.float64), state.problem
    )
    result.candidate_ids = np.array([10], dtype=np.int64)
    result.__post_init__()
    request = requests()[0]
    update = EvaluationUpdate(
        request.request_id,
        EvaluationStatus.COMPLETED,
        request.candidate_ids,
        result,
        sequence=0,
    )
    pending = PendingEvaluation(
        request,
        EvaluationStatus.COMPLETED,
        np.empty(0, dtype=np.int64),
        0,
        -1,
        {0: "received"},
        (update,),
        checkpointable=True,
    )
    state = state.replace(pending_evaluations={0: pending})
    path = tmp_path / "buffered.npz"
    state.save(path)
    restored = OptimizationState.load(path, state.problem)
    scheduler = AsyncEvaluationScheduler(ReattachEvaluator())
    restored = scheduler.reattach(restored)
    assert restored.pending_evaluations == {}
    assert restored.fe == 1
    np.testing.assert_array_equal(restored.archive.id, [10])


def test_callback_failure_is_fatal_without_losing_pending_or_fe():
    class FailingCallback:
        def __init__(self):
            self.calls = 0

        def dispatch(self, event):
            self.calls += 1
            raise RuntimeError("callback failed")

    state = make_state()
    evaluator = ReattachEvaluator()
    callback = FailingCallback()
    scheduler = AsyncEvaluationScheduler(evaluator, callback_manager=callback)
    state = scheduler.submit(state, [requests()[0]])
    try:
        scheduler.poll(state, wait=True)
    except RuntimeError as exc:
        assert "callback" in str(exc)
    else:
        raise AssertionError("callback failure was not fatal")
    assert state.pending_evaluations
    assert state.fe == 0
    assert len(state.archive) == 1
    assert callback.calls == 1


def test_callback_completed_replay_only_cleans_pending(tmp_path):
    state = make_state()
    evaluator = ReattachEvaluator()
    scheduler = AsyncEvaluationScheduler(evaluator)
    state = scheduler.submit(state, [requests()[0]])
    update = evaluator.collect(state.evaluation_handles[0])[0]
    pending = state.pending_evaluations[0]
    pending = PendingEvaluation(
        pending.request,
        EvaluationStatus.COMPLETED,
        np.array([10], dtype=np.int64),
        0,
        0,
        {0: "callback-completed"},
        (update,),
        pending.reserved_cost,
        0,
        True,
        pending.original_candidate_ids,
        None,
        None,
        pending.prediction,
    )
    state = state.replace(pending_evaluations={0: pending})
    path = tmp_path / "callback-completed.npz"
    state.save(path)
    restored = OptimizationState.load(path, state.problem)
    restored = AsyncEvaluationScheduler(ReattachEvaluator()).reattach(restored)
    assert restored.pending_evaluations == {}
    assert restored.fe == 0
    assert len(restored.archive) == 0


def test_partial_callback_checkpoint_reattaches_and_finishes(tmp_path):
    class TwoStageEvaluator(Evaluator):
        def __init__(self):
            self.attempts = 0
            self.reattach_ack = []
            self.submitted_ids = []

        def evaluate_batch(self, x, problem):
            return SerialEvaluator().evaluate_batch(x, problem)

        def submit(self, request, problem):
            self.submitted_ids.append(request.candidate_ids.tolist())
            handle = EvaluationHandle(
                request.request_id,
                EvaluationStatus.PENDING,
                backend_token=(request, problem, self.attempts, -1),
            )
            self.attempts += 1
            return handle

        def collect(self, handle, *, wait=True):
            request, problem, attempt, acknowledged = handle.backend_token
            if handle._acknowledged_sequence >= 0 and acknowledged < 0:
                return []
            if attempt == 0 and acknowledged < 0:
                result = SerialEvaluator().evaluate_batch(request.x[:1], problem)
                result.candidate_ids = request.candidate_ids[:1]
                result.__post_init__()
                handle._delivered_sequence = 0
                return [
                    EvaluationUpdate(
                        request.request_id,
                        EvaluationStatus.PARTIAL,
                        request.candidate_ids[:1],
                        result,
                        sequence=0,
                    )
                ]
            if attempt == 0 and acknowledged >= 0:
                handle._delivered_sequence = 1
                return [
                    EvaluationUpdate(
                        request.request_id,
                        EvaluationStatus.FAILED,
                        np.empty(0, dtype=np.int64),
                        error=EvaluationErrorInfo("backend", "retry"),
                        sequence=1,
                    )
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

        def acknowledge(self, handle, sequence):
            handle._acknowledged_sequence = sequence

        def can_reattach(self, pending):
            return True

        def reattach(self, pending, problem):
            request = pending.request
            handle = EvaluationHandle(
                request.request_id,
                EvaluationStatus.PENDING,
                backend_token=(
                    request,
                    problem,
                    pending.retry_count,
                    pending.last_acknowledged_sequence,
                ),
            )
            handle._acknowledged_sequence = pending.last_acknowledged_sequence
            self.reattach_ack.append(pending.last_acknowledged_sequence)
            return handle

    class Algorithm:
        def __init__(self):
            self.told = []

        def contract(self):
            return ComponentContract(
                lifecycle=LifecycleContract(
                    feedback=FeedbackContract(accepted_channels=frozenset({"true"}))
                )
            )

        def tell(self, feedback: FeedbackBatch, view: StateView) -> StatePatch:
            del feedback
            self.told.extend(view.context.offspring.get_array("id").tolist())
            return StatePatch(writes={})

    callback = CallbackManager()
    callback_ids = []

    def collect_callback_ids(event: PostEvaluationEvent) -> None:
        assert event.candidate_ids is not None
        callback_ids.extend(event.candidate_ids.tolist())

    callback.register(PostEvaluationEvent, collect_callback_ids)
    algorithm = Algorithm()
    evaluator = TwoStageEvaluator()
    scheduler = AsyncEvaluationScheduler(
        evaluator,
        retry_limit=1,
        feedback_builder=TrueOnlyFeedback(),
        algorithm=algorithm,
        callback_manager=callback,
    )
    scheduler.enable_feedback_accumulator()
    request = EvaluationRequest(
        np.int64(0),
        np.array([10, 11], dtype=np.int64),
        np.array([[0.2], [0.1]]),
    )
    state = scheduler.submit(make_state(), [request])
    state = scheduler.poll(state, wait=False)
    assert state.pending_evaluations[0].last_acknowledged_sequence == 0
    assert state.pending_evaluations[0].status is EvaluationStatus.PARTIAL
    assert state.pending_evaluations[0].processing[0] == "callback-completed"
    path = tmp_path / "partial-callback.npz"
    scheduler.checkpoint(state, path)
    restored = OptimizationState.load(path, state.problem)
    assert restored.pending_evaluations[0].status is EvaluationStatus.PARTIAL
    assert restored.pending_evaluations[0].processing[0] == "callback-completed"
    resumed_scheduler = AsyncEvaluationScheduler(
        evaluator,
        retry_limit=1,
        feedback_builder=TrueOnlyFeedback(),
        algorithm=algorithm,
        callback_manager=callback,
    )
    resumed_scheduler.enable_feedback_accumulator()
    resumed = resumed_scheduler.reattach(restored)
    resumed = resumed_scheduler.poll(resumed, wait=True)
    assert resumed.pending_evaluations == {}
    assert resumed.evaluation_handles == {}
    assert resumed.evaluation_owners == {}
    assert evaluator.reattach_ack == [0]
    assert evaluator.submitted_ids == [[10, 11], [11]]
    assert callback_ids == [10, 11]
    assert sorted(algorithm.told) == [10, 11]
    assert sorted(resumed.archive.id.tolist()) == [10, 11]
    assert resumed.fe == 2


def test_accumulator_checkpoint_reattach_rebuilds_partial_without_duplicate_tell(
    tmp_path,
):
    class PartialCheckpointEvaluator(Evaluator):
        def evaluate_batch(self, x, problem):
            return SerialEvaluator().evaluate_batch(x, problem)

        def submit(self, request, problem):
            return EvaluationHandle(
                request.request_id,
                EvaluationStatus.PENDING,
                backend_token=(request, problem, "initial"),
            )

        def collect(self, handle, *, wait=True):
            request, problem, phase = handle.backend_token
            if handle._acknowledged_sequence >= 0 and phase == "initial":
                return []
            if phase == "initial":
                result = SerialEvaluator().evaluate_batch(request.x[:1], problem)
                result.candidate_ids = request.candidate_ids[:1]
                result.__post_init__()
                handle._delivered_sequence = 0
                return [
                    EvaluationUpdate(
                        request.request_id,
                        EvaluationStatus.PARTIAL,
                        request.candidate_ids[:1],
                        result,
                        sequence=0,
                    )
                ]
            result = SerialEvaluator().evaluate_batch(request.x[1:], problem)
            result.candidate_ids = request.candidate_ids[1:]
            result.__post_init__()
            handle._delivered_sequence = 1
            return [
                EvaluationUpdate(
                    request.request_id,
                    EvaluationStatus.COMPLETED,
                    request.candidate_ids[1:],
                    result,
                    sequence=1,
                )
            ]

        def acknowledge(self, handle, sequence):
            handle._acknowledged_sequence = sequence

        def can_reattach(self, pending):
            return True

        def reattach(self, pending, problem):
            return EvaluationHandle(
                pending.request.request_id,
                EvaluationStatus.PENDING,
                backend_token=(pending.request, problem, "restored"),
            )

    state = make_state()
    state.set_state(PROPOSALS_CURRENT, 703)
    consumer = _RecordingConsumer()
    evaluator = PartialCheckpointEvaluator()
    scheduler = AsyncEvaluationScheduler(
        evaluator,
        feedback_builder=TrueOnlyFeedback(),
        algorithm=consumer,
    )
    scheduler.enable_feedback_accumulator()
    request = EvaluationRequest(
        np.int64(0),
        np.array([10, 11], dtype=np.int64),
        np.array([[0.2], [0.1]], dtype=np.float64),
    )
    state = scheduler.poll(scheduler.submit(state, [request]), wait=False)
    assert state.pending_evaluations[0].processing[0] == "committed"
    assert consumer.feedback == []

    path = tmp_path / "accumulator-partial.npz"
    scheduler.checkpoint(state, path)
    restored = OptimizationState.load(path, state.problem)
    resumed_scheduler = AsyncEvaluationScheduler(
        evaluator,
        feedback_builder=TrueOnlyFeedback(),
        algorithm=consumer,
    )
    resumed_scheduler.enable_feedback_accumulator()
    restored = resumed_scheduler.reattach(restored)
    restored = resumed_scheduler.poll(restored, wait=True)

    assert restored.pending_evaluations == {}
    assert len(consumer.feedback) == 1
    assert consumer.feedback[0].final is True
    subjects = consumer.feedback[0].observations.records.column("subject_payload")
    assert set(np.asarray(subjects, dtype=np.int64).reshape(-1)) == {10, 11}
    assert resumed_scheduler._feedback_accumulator is not None
    assert resumed_scheduler._feedback_accumulator.ready_count == 0


def test_fatal_tombstone_roundtrip_raises_typed_error(tmp_path):
    class FailingAlgorithm:
        def tell(self, feedback: FeedbackBatch, view: StateView) -> StatePatch:
            del feedback, view
            raise RuntimeError("tell failed")

    state = make_state()
    evaluator = AsyncEvaluator(SlowEvaluator())
    scheduler = AsyncEvaluationScheduler(
        evaluator,
        algorithm=FailingAlgorithm(),
        feedback_builder=TrueOnlyFeedback(),
    )
    state = scheduler.submit(state, [requests()[0]])
    try:
        scheduler.poll(state, wait=True)
    except EvaluationFatalError as exc:
        fatal_state = exc.state
    else:
        raise AssertionError("tell failure was not fatal")
    finally:
        evaluator.close()
    path = tmp_path / "fatal.npz"
    fatal_state.save(path)
    restored = OptimizationState.load(path, state.problem)
    with pytest.raises(EvaluationFatalError) as caught:
        AsyncEvaluationScheduler(AsyncEvaluator(SlowEvaluator())).reattach(restored)
    assert caught.value.state is restored


def test_scheduler_fatal_state_retains_the_keyed_state_reference():
    class FailingAlgorithm:
        def tell(self, feedback: FeedbackBatch, view: StateView) -> StatePatch:
            del feedback, view
            raise RuntimeError("tell failed")

    state = make_state()
    evaluator = ReattachEvaluator()
    scheduler = AsyncEvaluationScheduler(
        evaluator,
        algorithm=FailingAlgorithm(),
        feedback_builder=TrueOnlyFeedback(),
    )
    state = scheduler.submit(state, [requests()[0]])
    with pytest.raises(EvaluationFatalError) as caught:
        scheduler.poll(state, wait=True)

    retained = scheduler._fatal_states[id(state)]
    assert retained[0] is state
    assert retained[1] is caught.value.state


def test_tell_failure_cannot_be_retried_after_tell_started():
    class FailingAlgorithm:
        def __init__(self):
            self.calls = 0

        def tell(self, feedback: FeedbackBatch, view: StateView) -> StatePatch:
            del feedback, view
            self.calls += 1
            raise RuntimeError("tell failed")

    state = make_state()
    evaluator = ReattachEvaluator()
    algorithm = FailingAlgorithm()
    scheduler = AsyncEvaluationScheduler(
        evaluator, algorithm=algorithm, feedback_builder=TrueOnlyFeedback()
    )
    state = scheduler.submit(state, [requests()[0]])
    try:
        scheduler.poll(state, wait=True)
    except RuntimeError as exc:
        assert "tell" in str(exc)
    else:
        raise AssertionError("tell failure was not fatal")
    try:
        scheduler.poll(state, wait=True)
    except Exception as exc:
        assert "fatal" in str(exc) or "retried" in str(exc)
    else:
        raise AssertionError("tell failure was retried")
    assert algorithm.calls == 1


def test_async_archive_updates_main_and_pareto_once():
    state = make_state()
    scheduler = AsyncEvaluationScheduler(ReattachEvaluator())
    request = EvaluationRequest(
        np.int64(0), np.array([10, 11], dtype=np.int64), state.offspring.x.copy()
    )
    state = scheduler.submit(state, [request])
    state = scheduler.poll(state, wait=True)
    assert len(state.archive) == 2
    assert len(state.pareto_archive) == 1
    assert state.archive.id.tolist() == [10, 11]
    assert state.pareto_archive.id.tolist() == [11]


def test_archive_re_evaluation_uses_latest_observation():
    class ReevaluateEvaluator(ReattachEvaluator):
        def collect(self, handle, *, wait=True):
            if handle._acknowledged_sequence >= 0:
                return []
            request, problem = handle.backend_token
            base = SerialEvaluator().evaluate_batch(request.x, problem)
            result = EvaluationResult(
                np.full_like(base.f, float(request.metadata["value"])),
                base.g,
                base.cv,
                request.candidate_ids,
            )
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

    archive = Archive(ATTRS, 2, duplicate_policy="replace")
    state = make_state().replace(archive=archive)
    evaluator = ReevaluateEvaluator()
    scheduler = AsyncEvaluationScheduler(evaluator)
    first = EvaluationRequest(
        np.int64(0),
        np.array([10], dtype=np.int64),
        np.array([[0.2]]),
        metadata={"value": 0.2},
    )
    second = EvaluationRequest(
        np.int64(1),
        np.array([10], dtype=np.int64),
        np.array([[0.2]]),
        metadata={"value": 0.8},
    )
    state = scheduler.poll(scheduler.submit(state, [first]), wait=True)
    state = scheduler.poll(scheduler.submit(state, [second]), wait=True)
    assert len(state.archive) == 1
    np.testing.assert_allclose(state.archive.f[0], [0.8])
    assert len(state.pareto_archive) == 1
    np.testing.assert_allclose(state.pareto_archive.f[0], [0.8])


def test_append_archive_keeps_distinct_request_observations():
    attrs = [*ATTRS, PopulationAttribute("request_id", np.int64, (), -1)]
    state = make_state().replace(archive=Archive(attrs, 2, duplicate_policy="append"))
    evaluator = ReattachEvaluator()
    scheduler = AsyncEvaluationScheduler(evaluator)
    for request_id in (0, 1):
        request = EvaluationRequest(
            np.int64(request_id),
            np.array([10], dtype=np.int64),
            np.array([[0.2]]),
        )
        state = scheduler.poll(scheduler.submit(state, [request]), wait=True)
    assert len(state.archive) == 2
    assert sorted(
        zip(state.archive.request_id.tolist(), state.archive.id.tolist())
    ) == [(0, 10), (1, 10)]


def test_chunk_cost_budget_uses_fsum_boundary():
    population = Population(ATTRS, 8)
    population._extend_internal(
        {
            "id": np.arange(8, dtype=np.int64),
            "x": np.arange(8, dtype=np.float64).reshape(-1, 1),
            "f": np.full((8, 1), np.nan),
            "g": np.empty((8, 0)),
            "cv": np.zeros(8),
        },
        preserve_ids=True,
    )
    state = make_state().replace(population=population, offspring=population)
    scheduler = AsyncEvaluationScheduler(
        ReattachEvaluator(), max_pending=8, max_reserved_cost=0.1
    )
    for index in range(8):
        request = EvaluationRequest(
            np.int64(index),
            np.array([index], dtype=np.int64),
            population.x[index : index + 1],
            metadata={"estimated_cost": 0.1 / 8},
        )
        state = scheduler.submit(state, [request])
    assert scheduler.reserved_cost(state) <= 0.1


def test_pending_prediction_snapshot_survives_wave_overwrite(tmp_path):
    first = SurrogatePrediction(
        {"objective": PredictionChannel(np.array([[1.0]]))},
        x=np.array([[0.2]]),
    )
    second = SurrogatePrediction(
        {"objective": PredictionChannel(np.array([[9.0]]))},
        x=np.array([[0.2]]),
    )
    state = make_state().replace(predictions=first)
    scheduler = AsyncEvaluationScheduler(ReattachEvaluator())
    state = scheduler.submit(state, [requests()[0]])
    state = state.replace(predictions=second)
    assert state.pending_evaluations[0].prediction.value[0, 0] == 1.0
    path = tmp_path / "prediction-owner.npz"
    state.save(path)
    restored = OptimizationState.load(path, state.problem)
    assert restored.pending_evaluations[0].prediction.value[0, 0] == 1.0


def test_batch_submit_requires_declared_rollback_capability():
    class FalseRollbackEvaluator(BareEvaluator):
        def cancel(self, handle):
            return False

        def detach(self, handle):
            return False

    state = make_state()
    scheduler = AsyncEvaluationScheduler(FalseRollbackEvaluator(), max_pending=2)
    with pytest.raises(EvaluationProtocolError, match="rollback"):
        scheduler.submit(state, requests())


def test_timeout_without_runtime_termination_keeps_fatal_tombstone():
    class UnstoppableEvaluator(ReattachEvaluator):
        def cancel(self, handle):
            return False

        def detach(self, handle):
            return False

    state = make_state()
    scheduler = AsyncEvaluationScheduler(UnstoppableEvaluator(), timeout=0)
    state = scheduler.submit(state, [requests()[0]])
    state = scheduler.poll(state, wait=False)
    assert state.pending_evaluations[0].fatal_error is not None
    assert scheduler.pending_candidate_ids(state).tolist() == [10]
    assert state.async_fatal is not None
    assert state.async_fatal["request_id"] == 0
