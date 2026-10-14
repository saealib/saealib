import numpy as np

from saealib.context import OptimizationState
from saealib.core.contracts import (
    ComponentContract,
    FeedbackBatch,
    FeedbackContract,
    LifecycleContract,
)
from saealib.core.state import (
    FEEDBACK_ACCUMULATOR,
    PROPOSALS_CURRENT,
    StatePatch,
    StateView,
)
from saealib.execution import (
    AsyncEvaluationScheduler,
    EvaluationHandle,
    EvaluationRequest,
    EvaluationStatus,
    EvaluationUpdate,
    Evaluator,
    PendingEvaluation,
    SerialEvaluator,
)
from saealib.policies.feedback import TrueOnlyFeedback
from saealib.population import Archive, ParetoArchive, Population, PopulationAttribute
from saealib.problem import Problem

ATTRS = [
    PopulationAttribute("id", np.int64, (), -1),
    PopulationAttribute("x", np.float64, (1,)),
    PopulationAttribute("f", np.float64, (1,)),
    PopulationAttribute("g", np.float64, (0,)),
    PopulationAttribute("cv", np.float64, (), 0.0),
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
    return state


class AccumulatorAlgorithm:
    def __init__(self):
        self.delivered = {}
        self._contract = ComponentContract(
            lifecycle=LifecycleContract(
                feedback=FeedbackContract(accepted_channels=frozenset({"true"}))
            )
        )

    def contract(self) -> ComponentContract:
        return self._contract

    def tell(self, feedback: FeedbackBatch, view: StateView) -> StatePatch:
        self.delivered[feedback.proposal_id] = feedback
        return StatePatch(writes={})


class ResumeEvaluator(Evaluator):
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
        return []

    def acknowledge(self, handle, sequence):
        self.acknowledged.append((int(handle.request_id), int(sequence)))
        handle._acknowledged_sequence = int(sequence)

    def supports_batch_rollback(self):
        return True

    def can_reattach(self, pending):
        return True

    def reattach(self, pending, problem):
        return EvaluationHandle(
            pending.request.request_id,
            EvaluationStatus.PENDING,
            backend_token=(pending.request, problem),
        )


def test_resume_without_accumulator_snapshot_restores_pending_feedback(tmp_path):
    state = make_state()
    state.set_state(PROPOSALS_CURRENT, 0)
    request = EvaluationRequest(
        np.int64(0), np.array([10], dtype=np.int64), np.array([[0.2]])
    )
    result = SerialEvaluator().evaluate_batch(
        np.array([[0.2]], dtype=np.float64), state.problem
    )
    result.candidate_ids = request.candidate_ids
    result.__post_init__()
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

    checkpoint = tmp_path / "stashed.npz"
    state.save(checkpoint)
    restored = OptimizationState.load(checkpoint, state.problem)
    assert restored.pending_evaluations[0].checkpointable
    buffered = restored.pending_evaluations[0].buffered_updates
    assert len(buffered) == 1
    assert buffered[0].status is EvaluationStatus.COMPLETED
    assert buffered[0].result is not None
    np.testing.assert_array_equal(buffered[0].candidate_ids, [10])
    np.testing.assert_allclose(buffered[0].result.f, [[0.2]])

    algorithm = AccumulatorAlgorithm()
    evaluator = ResumeEvaluator()
    scheduler = AsyncEvaluationScheduler(
        evaluator,
        max_pending=2,
        feedback_builder=TrueOnlyFeedback(),
        algorithm=algorithm,
    )
    scheduler.enable_feedback_accumulator()
    restored = scheduler.reattach(restored)

    accumulator_state = restored.get_state(FEEDBACK_ACCUMULATOR)
    assert accumulator_state is not None
    assert accumulator_state["codec"] == "feedback_accumulator"
    assert accumulator_state["buffers"] == []
    assert accumulator_state["ready"] == []
    assert restored.pending_evaluations == {}
    assert restored.fe == 1
    delivered = algorithm.delivered[0]
    assert delivered.final
    np.testing.assert_array_equal(delivered.observations.candidate_ids, [10])
    np.testing.assert_allclose(delivered.observations.f, [[0.2]])
    assert evaluator.acknowledged == [(0, 0)]
