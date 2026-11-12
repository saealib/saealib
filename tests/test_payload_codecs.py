from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from saealib.context import (
    EvaluationPlanState,
    OptimizationState,
    _pending_to_json,
    _request_from_json,
    _request_to_json,
)
from saealib.exceptions import ValidationError
from saealib.execution.evaluator import (
    EvaluationErrorInfo,
    EvaluationHandle,
    EvaluationRequest,
    EvaluationResult,
    EvaluationStatus,
    EvaluationUpdate,
    Evaluator,
    PendingEvaluation,
)
from saealib.execution.scheduler import AsyncEvaluationScheduler
from saealib.identity import IDAllocator
from saealib.population import (
    Archive,
    DenseVectorBatch,
    ObjectBatch,
    ParetoArchive,
    Population,
    PopulationAttribute,
)
from saealib.problem import Problem


def _problem() -> Problem:
    return Problem(
        func=lambda x: np.array([np.sum(x**2)]),
        dim=2,
        n_obj=1,
        direction=np.array([-1.0]),
        lb=[-1.0, -1.0],
        ub=[1.0, 1.0],
    )


def _state() -> OptimizationState:
    attrs = [
        PopulationAttribute("x", np.float64, (2,), default=np.nan),
        PopulationAttribute("f", np.float64, (1,), default=np.nan),
        PopulationAttribute("g", np.float64, (0,), default=0.0),
        PopulationAttribute("cv", np.float64, (), default=0.0),
        PopulationAttribute("id", np.int64, (), default=-1),
        PopulationAttribute("request_id", np.int64, (), default=-1),
    ]
    population = Population(attrs, 4)
    population._extend_internal(
        {
            "x": np.array([[0.1, 0.2], [0.3, 0.4]]),
            "f": np.array([[1.0], [2.0]]),
            "g": np.empty((2, 0)),
            "cv": np.zeros(2),
            "id": np.array([7, 8], dtype=np.int64),
            "request_id": np.array([17, 18], dtype=np.int64),
        },
        preserve_ids=True,
    )
    archive = Archive(attrs, 4, key_attr="x", duplicate_policy="replace")
    archive._extend_internal(population, preserve_ids=True)
    pareto = ParetoArchive(attrs, 4, direction=np.array([-1.0]), eps_cv=0.25)
    pareto._extend_internal(population, preserve_ids=True)
    return OptimizationState(
        problem=_problem(),
        populations={"main": population},
        archives={"main": archive, "pareto": pareto},
        rng=np.random.default_rng(3),
        candidate_id_allocator=IDAllocator(20),
        request_id_allocator=IDAllocator(30),
        fe=2,
        gen=4,
    )


def _object_request(request_id: int = 1) -> EvaluationRequest:
    return EvaluationRequest(
        np.int64(request_id),
        np.array([7, 8], dtype=np.int64),
        ObjectBatch(["left", "right"]),
        metadata={"fidelity": 1},
    )


def test_request_normalizes_legacy_dense_x_and_exposes_readonly_view() -> None:
    request = EvaluationRequest(
        np.int64(1), np.array([10, 11]), np.array([[1.0, 2.0], [3.0, 4.0]])
    )

    assert isinstance(request.payload, DenseVectorBatch)
    np.testing.assert_array_equal(request.x, [[1.0, 2.0], [3.0, 4.0]])
    assert request.x.dtype == np.float64
    assert not request.x.flags.writeable


def test_object_payload_has_no_dense_compatibility_view() -> None:
    request = EvaluationRequest(np.int64(1), np.array([10]), ObjectBatch([{"x": 1}]))

    with pytest.raises(ValidationError, match=r"DenseNumericView.*DenseVectorBatch"):
        _ = request.x


def test_request_rejects_non_genome_payload_and_mismatched_ids() -> None:
    with pytest.raises(ValidationError, match="GenomeBatch"):
        EvaluationRequest(np.int64(1), np.array([10]), cast(Any, object()))
    with pytest.raises(ValidationError, match="match payload"):
        EvaluationRequest(np.int64(1), np.array([10]), ObjectBatch([1, 2]))


def test_request_codecs_write_and_read_payload() -> None:
    object_request = EvaluationRequest(
        np.int64(1), np.array([10]), ObjectBatch([{"label": "a"}])
    )
    encoded = _request_to_json(object_request)
    assert "payload" in encoded and "x" not in encoded
    restored = _request_from_json(encoded)
    assert isinstance(restored.payload, ObjectBatch)
    assert restored.payload.items == ({"label": "a"},)


def test_pending_codec_supports_object_payload_without_x() -> None:
    request = EvaluationRequest(np.int64(1), np.array([10]), ObjectBatch(["a"]))
    pending = PendingEvaluation(
        request,
        EvaluationStatus.PENDING,
        np.array([], dtype=np.int64),
        checkpointable=True,
    )

    encoded = _pending_to_json(pending)
    assert encoded["payload"]["kind"] == "object"
    assert "x" not in encoded


def test_non_dense_payload_survives_chunk_replicate_fidelity_and_retry() -> None:
    from saealib.policies.evaluation import (
        EvaluationPlan,
        RepeatedEvaluation,
        _continue_fidelity_plan,
    )
    from saealib.stages import AsyncEvaluationSubmitStage

    request = _object_request()

    class RecordingScheduler:
        max_pending = 2

        def __init__(self):
            self.requests = ()
            self.feedback_builder = None
            self.algorithm = None
            self.callback_manager = None

        def submit(self, state, requests):
            self.requests = tuple(requests)
            return state

    scheduler = RecordingScheduler()
    state = _state()
    state = state.replace(
        offspring=state.population,
        evaluation_plan=EvaluationPlan((request,)),
        evaluation_plan_state=EvaluationPlanState(deferred=(1,)),
    )
    submitted = AsyncEvaluationSubmitStage(
        scheduler, planner=RepeatedEvaluation(1)
    ).execute(state)
    assert [item.payload.items for item in scheduler.requests] == [
        ("left",),
        ("right",),
    ]
    assert submitted.evaluation_plan is not None

    class ObjectEvaluator(Evaluator):
        def __init__(self):
            self.attempts = 0
            self.submitted = []
            self.collected = []

        def evaluate_batch(self, x, problem):
            raise AssertionError("opaque payload reached the dense evaluator boundary")

        def submit(self, request, problem):
            attempt = self.attempts
            self.attempts += 1
            self.submitted.append((attempt, request.payload.items))
            return EvaluationHandle(
                request.request_id,
                EvaluationStatus.PENDING,
                backend_token=(request, attempt),
            )

        def collect(self, handle, *, wait=True):
            request, attempt = handle.backend_token
            self.collected.append((attempt, request.payload.items))
            if handle._acknowledged_sequence >= 0:
                return []
            if attempt == 0:
                partial = EvaluationResult(
                    np.ones((1, 1)),
                    np.empty((1, 0)),
                    np.zeros(1),
                    candidate_ids=request.candidate_ids[:1],
                )
                handle._delivered_sequence = 1
                return [
                    EvaluationUpdate(
                        request.request_id,
                        EvaluationStatus.PARTIAL,
                        request.candidate_ids[:1],
                        partial,
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
            result = EvaluationResult(
                np.ones((1, 1)),
                np.empty((1, 0)),
                np.zeros(1),
                candidate_ids=request.candidate_ids,
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

        def acknowledge(self, handle, sequence):
            handle._acknowledged_sequence = sequence

    object_state = _state()
    object_state = object_state.replace(offspring=object_state.population)
    evaluator = ObjectEvaluator()
    retry_state = object_state
    retry_scheduler = AsyncEvaluationScheduler(evaluator, retry_limit=1)
    retry_state = retry_scheduler.submit(retry_state, [request])
    retry_state = retry_scheduler.poll(retry_state, wait=True)
    assert evaluator.submitted == [(0, ("left", "right")), (1, ("right",))]
    assert evaluator.collected == evaluator.submitted
    assert retry_state.pending_evaluations == {}

    class ObjectRepeated(RepeatedEvaluation):
        def plan_replicates(self, candidates, ctx):
            return tuple(
                EvaluationRequest(
                    np.int64(index + 1),
                    request.candidate_ids,
                    request.payload,
                    metadata={"replicate": index},
                )
                for index in range(2)
            )

    repeated_plan = ObjectRepeated(2).plan(
        None, None, ctx=cast(OptimizationState, SimpleNamespace())
    )
    assert [item.payload.items for item in repeated_plan.requests] == [
        ("left", "right"),
        ("left", "right"),
    ]

    low_update = EvaluationUpdate(
        request.request_id,
        EvaluationStatus.COMPLETED,
        request.candidate_ids,
        EvaluationResult(
            np.array([[2.0], [1.0]]),
            np.empty((2, 0)),
            np.zeros(2),
            candidate_ids=request.candidate_ids,
        ),
        sequence=0,
    )
    context = SimpleNamespace(
        problem=SimpleNamespace(direction=np.array([-1.0])),
        request_id_allocator=IDAllocator(10),
    )
    promoted = _continue_fidelity_plan(
        EvaluationPlan(
            (request,),
            continuation={
                "kind": "fidelity_promotion",
                "next_fidelity": 2,
                "promotion_count": 1,
            },
        ),
        {int(request.request_id): [low_update]},
        cast(Any, context),
    )
    assert promoted is not None
    assert cast(ObjectBatch, promoted.requests[1].payload).items == ("right",)
