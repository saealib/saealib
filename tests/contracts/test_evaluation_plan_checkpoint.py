import json
from typing import cast

import numpy as np
import pytest
from test_named_state_checkpoint import _problem, _state

from saealib.acquisition import AcquisitionFunction, AcquisitionResult
from saealib.context import EvaluationPlanState
from saealib.exceptions import CheckpointError, ValidationError
from saealib.execution.evaluator import EvaluationRequest
from saealib.policies.evaluation import EvaluationPlan, EvaluationPlanner
from saealib.stages import AcquisitionStage, EvaluationPlanStage


def test_multi_request_plan_round_trip(tmp_path):
    state = _state()
    requests = tuple(
        EvaluationRequest(
            np.int64(index),
            np.array([7, 8], dtype=np.int64),
            np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float64),
            metadata={"replicate": index, "plan_id": 1},
        )
        for index in (1, 2, 3)
    )
    state = state.replace(
        evaluation_plan=EvaluationPlan(
            requests,
            completion_rule="all_requests_completed",
            artifacts={
                "replicate_ids": np.array([1, 2, 3], dtype=np.int64),
                "summary": {"count": 3, "complete": True},
            },
        ),
        evaluation_plan_state=EvaluationPlanState(
            submitted=(1, 2), deferred=(3,), continuation=False, feedback=True
        ),
    )
    path = tmp_path / "plan.npz"
    state.save(path)
    restored = type(state).load(path, _problem())
    assert restored.evaluation_plan is not None
    assert [int(item.request_id) for item in restored.evaluation_plan.requests] == [
        1,
        2,
        3,
    ]
    assert restored.evaluation_plan.requests[1].metadata["replicate"] == 2
    artifacts = restored.evaluation_plan.artifacts
    assert artifacts["summary"] == {
        "count": len(restored.evaluation_plan.requests),
        "complete": True,
    }
    np.testing.assert_array_equal(
        artifacts["replicate_ids"],
        [int(request.request_id) for request in restored.evaluation_plan.requests],
    )
    assert restored.evaluation_plan_state.deferred == (3,)
    assert restored.evaluation_plan_state.feedback is True


class _ArtifactAcquisition(AcquisitionFunction):
    def evaluate(self, candidates_x, prediction, archive, ctx=None, *, prepared=None):
        return AcquisitionResult(
            scores=np.array([0.25, 0.75], dtype=np.float64),
            artifacts={"candidate_order": candidates_x[:, 0].copy()},
        )


class _RecordingPlanner:
    def __init__(self):
        self.received = None

    def plan(self, candidates, acquisition, ctx):
        self.received = acquisition
        assert acquisition is not None
        return EvaluationPlan(
            (
                EvaluationRequest(
                    np.int64(1), candidates.id.copy(), candidates.x.copy()
                ),
            ),
            artifacts={
                "acquisition_artifact": acquisition.artifacts["candidate_order"]
            },
        )


class _IdentityFeatureEncoder:
    def encode(self, genomes):
        return genomes.array


def test_acquisition_result_artifacts_reach_evaluation_planner():
    state = _state()
    state.problem.space.services.register("FeatureEncoder", _IdentityFeatureEncoder())
    state = state.replace(offspring=state.population)
    scored = AcquisitionStage(_ArtifactAcquisition(), cbmanager=None).execute(state)

    acquisition_result = scored.acquisition_result
    offspring = scored.offspring
    assert acquisition_result is not None
    assert offspring is not None
    scores = acquisition_result.scores
    assert scores is not None
    assert scores.shape == (len(offspring),)
    np.testing.assert_array_equal(
        acquisition_result.artifacts["candidate_order"],
        offspring.x[:, 0],
    )

    planner = _RecordingPlanner()
    planned = EvaluationPlanStage(planner=cast(EvaluationPlanner, planner)).execute(
        scored
    )

    assert planner.received is acquisition_result
    np.testing.assert_array_equal(
        planner.received.artifacts["candidate_order"],
        acquisition_result.artifacts["candidate_order"],
    )
    assert (
        planned.evaluation_plan is not None
        and planned.evaluation_plan.artifacts["acquisition_artifact"]
        is acquisition_result.artifacts["candidate_order"]
    )


def test_plan_state_rejects_invalid_relationships_and_update_ids():
    with pytest.raises(ValidationError, match="completed requests"):
        EvaluationPlanState(completed=(1,))
    with pytest.raises(ValidationError, match="acknowledged requests"):
        EvaluationPlanState(acknowledged=(1,))
    state = _state()
    request = EvaluationRequest(
        np.int64(1), np.array([7, 8], dtype=np.int64), state.population.x.copy()
    )
    with pytest.raises(ValidationError, match="unknown request"):
        state.replace(
            evaluation_plan=EvaluationPlan((request,)),
            evaluation_plan_state=EvaluationPlanState(deferred=(1,)),
            evaluation_plan_updates={2: []},
        )


def _planned_state():
    state = _state()
    requests = tuple(
        EvaluationRequest(
            np.int64(request_id),
            np.array([7], dtype=np.int64),
            np.array([[0.1]], dtype=np.float64),
        )
        for request_id in (1, 2)
    )
    return state.replace(
        evaluation_plan=EvaluationPlan(requests),
        evaluation_plan_state=EvaluationPlanState(deferred=(1, 2)),
    )


def test_plan_state_replace_uses_lightweight_clone_and_retains_validation():
    state = _planned_state()
    updated = state.replace(
        evaluation_plan_state=state.evaluation_plan_state,
        evaluation_plan_updates=state.evaluation_plan_updates,
        gen=state.gen + 1,
    )

    assert updated is not state
    assert updated.evaluation_plan is state.evaluation_plan
    assert updated.populations is not state.populations
    with pytest.raises(ValidationError, match="unknown request"):
        state.replace(evaluation_plan_updates={99: []})


def _rewrite_state_entry(path, destination, name, value):
    payload = dict(np.load(path, allow_pickle=False).items())
    entries = json.loads(bytes(payload["_state_entries"]).decode())
    for item in entries:
        key = item["key"]
        if key["namespace"] == "evaluations" and key["name"] == name:
            item["value"]["value"] = value
            break
    else:
        raise AssertionError(f"evaluations/{name} entry was not found")
    payload["_state_entries"] = np.frombuffer(
        json.dumps(entries).encode(), dtype=np.uint8
    )
    np.savez(destination, **payload)


def test_checkpoint_rejects_malformed_evaluation_plan_payload(tmp_path):
    state = _planned_state()
    source = tmp_path / "valid.npz"
    state.save(source)
    bad = tmp_path / "bad-plan.npz"
    _rewrite_state_entry(source, bad, "plan", {"requests": [{}]})

    with pytest.raises(CheckpointError, match="evaluation plan is malformed"):
        type(state).load(bad, _problem())


def test_checkpoint_rejects_plan_state_referencing_unknown_request(tmp_path):
    state = _planned_state()
    source = tmp_path / "valid.npz"
    state.save(source)
    bad = tmp_path / "bad-state.npz"
    _rewrite_state_entry(source, bad, "plan_state", {"deferred": [99]})

    with pytest.raises(CheckpointError, match="unknown request"):
        type(state).load(bad, _problem())


def test_checkpoint_rejects_updates_for_unknown_request(tmp_path):
    state = _planned_state()
    source = tmp_path / "valid.npz"
    state.save(source)
    bad = tmp_path / "bad-updates.npz"
    _rewrite_state_entry(source, bad, "plan_updates", {"99": []})

    with pytest.raises(CheckpointError, match="updates reference an unknown request"):
        type(state).load(bad, _problem())
