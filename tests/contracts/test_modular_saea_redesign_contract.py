from types import SimpleNamespace
from typing import ClassVar

import numpy as np
import pytest

from saealib import (
    EvaluateAll,
    EvaluationPlan,
    FidelityPromotion,
    RepeatedEvaluation,
    TopKEvaluation,
)
from saealib.exceptions import ValidationError
from saealib.execution.evaluator import EvaluationRequest


class _Allocator:
    def __init__(self):
        self.next = 0

    def allocate(self, count):
        result = np.arange(self.next, self.next + count, dtype=np.int64)
        self.next += count
        return result


class _Candidates:
    schema: ClassVar[dict] = {}

    def __init__(self):
        self.x = np.arange(6, dtype=np.float64).reshape(3, 2)

    def __len__(self):
        return len(self.x)


def _candidates():
    return _Candidates()


def _ctx():
    return SimpleNamespace(
        candidate_id_allocator=_Allocator(), request_id_allocator=_Allocator()
    )


def test_builtin_planners_return_plans_and_repeated_requests():
    candidates = _candidates()
    ctx = _ctx()
    all_plan = EvaluateAll().plan(candidates, None, ctx=ctx)
    assert isinstance(all_plan, EvaluationPlan)
    assert len(all_plan.requests) == 1
    top_plan = TopKEvaluation(2).plan(
        candidates, SimpleNamespace(scores=np.array([1.0, 3.0, 2.0])), ctx=ctx
    )
    assert len(top_plan.requests[0].candidate_ids) == 2
    repeated = RepeatedEvaluation(3).plan(candidates, None, ctx=ctx)
    assert isinstance(repeated, EvaluationPlan)
    assert len(repeated.requests) == 3
    assert len({int(r.request_id) for r in repeated.requests}) == 3
    assert all(np.array_equal(r.candidate_ids, [0, 1, 2]) for r in repeated.requests)


def test_evaluation_policy_is_removed():
    with pytest.raises(ImportError):
        exec("from saealib import EvaluationPolicy", {})


def test_evaluation_plan_rejects_duplicate_request_ids():
    request = EvaluationRequest(np.int64(7), np.array([0]), np.array([[0.0]]))
    with pytest.raises(ValidationError, match="duplicate request IDs"):
        EvaluationPlan((request, request))


def test_fidelity_promotion_plan_declares_standard_continuation():
    planner = FidelityPromotion(fidelity=0, next_fidelity=1)
    ctx = _ctx()
    plan = planner.plan(_candidates(), None, ctx=ctx)

    assert len(plan.requests) == 1
    assert plan.completion_rule == "fidelity_promotion"
    assert plan.continuation["next_fidelity"] == 1
