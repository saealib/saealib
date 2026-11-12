from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from saealib.acquisition import AcquisitionResult
from saealib.acquisition.mean import MeanPrediction
from saealib.context import OptimizationState
from saealib.exceptions import ValidationError
from saealib.execution.evaluator import EvaluationResult
from saealib.policies import (
    ComparatorWorstFallback,
    EvaluateAll,
    EvaluationPlan,
    FidelityEvaluation,
    FidelityPromotion,
    MixedFeedback,
    NoFeedback,
    PredictedFeedback,
    RatioEvaluation,
    RepeatedEvaluation,
    TopKEvaluation,
    TrueOnlyFeedback,
    aggregate_replicates,
    select_ratio,
    select_top_k,
)
from saealib.population import Archive, ParetoArchive, Population, PopulationAttribute
from saealib.problem import Problem
from saealib.strategies.direct import DirectStrategy
from saealib.strategies.gb import GenerationBasedStrategy
from saealib.strategies.ib import IndividualBasedStrategy
from saealib.strategies.ps import PreSelectionStrategy
from saealib.surrogate import PredictionChannel, SurrogatePrediction
from saealib.surrogate.rbf import RBFSurrogate
from saealib.surrogate.rbf_kernels import GaussianKernel


def _state() -> OptimizationState:
    problem = Problem(
        func=lambda x: np.array([x[0]]),
        dim=1,
        n_obj=1,
        direction=np.array([-1.0]),
        lb=[-1.0],
        ub=[1.0],
    )
    attrs = [
        PopulationAttribute("id", np.int64, (), default=-1),
        PopulationAttribute("x", np.float64, (1,)),
        PopulationAttribute("f", np.float64, (1,)),
        PopulationAttribute("g", np.float64, (0,)),
        PopulationAttribute("cv", np.float64, ()),
    ]
    pop = Population(attrs, 2)
    pop._extend_internal(
        {
            "id": np.array([10, 11], dtype=np.int64),
            "x": np.array([[0.1], [0.2]], dtype=np.float64),
            "f": np.full((2, 1), np.nan),
            "g": np.empty((2, 0), dtype=np.float64),
            "cv": np.zeros(2),
        },
        preserve_ids=True,
    )
    return OptimizationState(
        problem=problem,
        population=pop,
        archive=Archive(attrs, 2),
        pareto_archive=ParetoArchive(attrs, 2, direction=np.array([-1.0])),
        rng=np.random.default_rng(0),
        offspring=pop,
    )


def test_selection_kernels_are_stable_and_non_mutating():
    scores = np.array([2.0, 2.0, 1.0], dtype=np.float64)
    original = scores.copy()
    np.testing.assert_array_equal(select_top_k(scores, 2), [0, 1])
    np.testing.assert_array_equal(select_ratio(scores, 2 / 3), [0, 1])
    np.testing.assert_array_equal(scores, original)


def test_selection_boundaries_and_empty():
    scores = np.empty(0, dtype=np.float64)
    np.testing.assert_array_equal(select_top_k(scores, 0), np.empty(0, dtype=np.intp))
    np.testing.assert_array_equal(select_ratio(scores, 0.1), np.empty(0, dtype=np.intp))
    with pytest.raises(ValidationError):
        select_top_k(np.ones(2), 3)
    with pytest.raises(ValidationError):
        select_ratio(np.ones(2), 1.1)
    with pytest.raises(ValidationError):
        select_top_k(np.array([1.0, np.nan]), 1)
    plan = TopKEvaluation(1, sanitize_nonfinite=True).plan(
        _state().offspring,
        AcquisitionResult(np.array([1.0, np.nan], dtype=np.float64)),
        ctx=_state(),
    )
    np.testing.assert_array_equal(plan.requests[0].candidate_ids, [10])


def test_rbf_singular_solve_nan_flows_through_sanitized_evaluation_planning():
    """RBFSurrogate's singular-solve NaN is a recoverable failure, not an
    abort: it reaches RatioEvaluation(sanitize_nonfinite=True) as a
    non-finite acquisition score and gets planned around rather than
    raising."""
    train_x = np.array([[-1.0], [0.0], [0.0], [1.0]])
    train_y = np.array([1.0, 2.0, 2.0, 3.0])
    surrogate = RBFSurrogate(kernel=GaussianKernel(), solver="solve")
    surrogate.fit(train_x, train_y)

    state = _state()
    assert state.offspring is not None
    prediction = surrogate.predict(state.offspring.x)
    assert np.all(np.isnan(prediction.value))

    scores = MeanPrediction(direction=np.array([-1.0])).score(prediction)
    assert np.all(np.isnan(scores))

    plan = RatioEvaluation(0.5, sanitize_nonfinite=True).plan(
        state.offspring, AcquisitionResult(scores), ctx=state
    )

    assert len(plan.requests[0].candidate_ids) >= 1


def test_plan_and_score_validation_covers_invalid_policy_inputs():
    state = _state()
    request = EvaluateAll().plan(state.offspring, None, ctx=state).requests[0]
    with pytest.raises(ValidationError, match="contain a request"):
        EvaluationPlan(())
    with pytest.raises(ValidationError, match="duplicate request IDs"):
        EvaluationPlan((request, request))
    with pytest.raises(ValidationError, match="acquisition score array"):
        TopKEvaluation(1).plan(state.offspring, None, ctx=state)
    with pytest.raises(ValidationError, match="float64"):
        TopKEvaluation(1).plan(
            state.offspring,
            AcquisitionResult(np.ones(2, dtype=np.float32)),
            ctx=state,
        )
    with pytest.raises(ValidationError, match="integer"):
        select_top_k(np.ones(2, dtype=np.float64), True)
    with pytest.raises(ValidationError, match="within"):
        select_top_k(np.ones(2, dtype=np.float64), -1)
    with pytest.raises(ValidationError, match="float64"):
        select_ratio(np.ones((2, 1), dtype=np.float64), 0.5)
    ratio = RatioEvaluation(0.5).plan(
        state.offspring,
        AcquisitionResult(np.array([1.0, 0.0], dtype=np.float64)),
        ctx=state,
    )
    assert len(ratio.requests[0].candidate_ids) == 1


def test_repeated_and_fidelity_planners_validate_and_annotate_requests():
    state = _state()
    with pytest.raises(ValidationError, match="positive integer"):
        RepeatedEvaluation(0)
    with pytest.raises(ValidationError, match="positive integer"):
        RepeatedEvaluation(True)
    repeated = RepeatedEvaluation(2).plan(state.offspring, None, ctx=state)
    assert len(repeated.requests) == 2
    assert [request.metadata["replicate"] for request in repeated.requests] == [0, 1]
    assert all(request.metadata["plan_id"] == 0 for request in repeated.requests)

    with pytest.raises(ValidationError, match="non-negative integer"):
        FidelityEvaluation(-1)
    fidelity = FidelityEvaluation(3).plan(state.offspring, None, ctx=state)
    assert fidelity.requests[0].metadata["fidelity"] == 3
    with pytest.raises(ValidationError, match="next_fidelity"):
        FidelityPromotion(1, 1)
    with pytest.raises(ValidationError, match="promotion_count"):
        FidelityPromotion(0, 1, promotion_count=0)
    with pytest.raises(ValidationError, match="promotion_fraction"):
        FidelityPromotion(0, 1, promotion_fraction=0.0)
    promotion = FidelityPromotion(0, 2, promotion_count=1).plan(
        state.offspring, None, ctx=state
    )
    assert promotion.continuation["kind"] == "fidelity_promotion"


def test_replicate_aggregation_validates_shape_and_owns_summary():
    with pytest.raises(ValidationError, match="observations"):
        aggregate_replicates(np.array([1, 2]), np.ones((2, 1)))
    summary = aggregate_replicates(
        np.array([1, 2], dtype=np.int64),
        np.array(
            [
                [[1.0], [3.0]],
                [[3.0], [5.0]],
            ],
            dtype=np.float64,
        ),
    )
    np.testing.assert_allclose(summary.mean, [[2.0], [4.0]])
    np.testing.assert_array_equal(summary.count, [2, 2])


def test_evaluation_policies_allocate_requests():
    state = _state()
    acquisition = AcquisitionResult(np.array([1.0, 0.0], dtype=np.float64))
    request = (
        TopKEvaluation(1).plan(state.offspring, acquisition, ctx=state).requests[0]
    )
    assert request.request_id == 0
    np.testing.assert_array_equal(request.candidate_ids, [10])
    assert not request.x.flags.writeable
    request_all = EvaluateAll().plan(state.offspring, None, ctx=state).requests[0]
    assert request_all.request_id == 1
    np.testing.assert_array_equal(request_all.candidate_ids, [10, 11])


def test_feedback_sources_and_objective_channel_requirement():
    state = _state()
    evaluation = EvaluationResult(
        np.array([[4.0]], dtype=np.float64),
        np.empty((1, 0), dtype=np.float64),
        np.zeros(1, dtype=np.float64),
        candidate_ids=np.array([10], dtype=np.int64),
    )
    prediction = SurrogatePrediction(
        {"objective": PredictionChannel(np.array([[5.0], [6.0]]))}
    )
    mixed = MixedFeedback().build(
        state.offspring, prediction, evaluation, np.array([10]), ctx=state
    )
    np.testing.assert_array_equal(mixed.candidate_ids, [10, 11])
    np.testing.assert_array_equal(mixed.f, [[4.0], [6.0]])
    np.testing.assert_array_equal(mixed.source, [0, 1])
    true_only = TrueOnlyFeedback().build(
        state.offspring, prediction, evaluation, np.array([10]), ctx=state
    )
    np.testing.assert_array_equal(true_only.candidate_ids, [10])
    assert (
        len(
            NoFeedback().build(state.offspring, prediction, evaluation, [], ctx=state).f
        )
        == 0
    )
    with pytest.raises(ValidationError):
        PredictedFeedback().build(
            state.offspring,
            SurrogatePrediction({"win_rate": PredictionChannel(np.ones((2, 1)))}),
            None,
            [],
            ctx=state,
        )
    assert (
        len(TrueOnlyFeedback().build(state.offspring, None, None, [], ctx=state).f) == 0
    )
    with pytest.raises(ValidationError, match="prediction rows"):
        PredictedFeedback().build(
            state.offspring,
            SurrogatePrediction({"objective": PredictionChannel(np.ones((1, 1)))}),
            None,
            [],
            ctx=state,
        )
    with pytest.raises(ValidationError, match="objective shape"):
        MixedFeedback().build(
            state.offspring,
            SurrogatePrediction({"objective": PredictionChannel(np.ones((1, 2)))}),
            None,
            [],
            ctx=state,
        )


def test_mixed_missing_rows_are_available_to_explicit_fallback():
    state = _state()
    state.population.update_rows(
        np.array([0, 1]), {"f": np.array([[1.0], [3.0]], dtype=np.float64)}
    )
    evaluation = EvaluationResult(
        np.array([[4.0]], dtype=np.float64),
        np.empty((1, 0), dtype=np.float64),
        np.zeros(1, dtype=np.float64),
        candidate_ids=np.array([10], dtype=np.int64),
    )
    result = ComparatorWorstFallback().build(
        state.offspring,
        SurrogatePrediction({"win_rate": PredictionChannel(np.ones((2, 1)))}),
        evaluation,
        np.array([10]),
        ctx=state,
    )
    np.testing.assert_array_equal(result.f, [[4.0], [3.0]])
    np.testing.assert_array_equal(result.source, [0, 2])


def test_feedback_result_rejects_misaligned_and_object_arrays():
    with pytest.raises(ValidationError):
        from saealib.policies import FeedbackResult

        FeedbackResult(
            np.array([1, 2], dtype=np.int64),
            np.ones((2, 1), dtype=np.float64),
            None,
            None,
            np.ones(2, dtype=bool),
            np.ones(2, dtype=np.uint8),
            {"artifact": np.ones(1, dtype=np.float64)},
        )
    with pytest.raises(ValidationError):
        from saealib.policies import FeedbackResult

        FeedbackResult(
            np.array([1, 1], dtype=np.int64),
            np.ones((2, 1), dtype=np.float64),
            None,
            None,
            np.ones(2, dtype=bool),
            np.ones(2, dtype=np.uint8),
        )
    with pytest.raises(ValidationError):
        from saealib.policies import FeedbackResult

        FeedbackResult(
            np.array([1], dtype=np.int64),
            np.array([["x"]], dtype=object),
            None,
            None,
            np.ones(1, dtype=bool),
            np.ones(1, dtype=np.uint8),
        )


def test_builtin_strategy_policy_compositions():
    assert isinstance(DirectStrategy().feedback_builder, TrueOnlyFeedback)
    assert isinstance(IndividualBasedStrategy().feedback_builder, MixedFeedback)
    assert isinstance(PreSelectionStrategy(8, 2).feedback_builder, TrueOnlyFeedback)
    assert isinstance(GenerationBasedStrategy(1).feedback_builder, MixedFeedback)
    assert isinstance(
        GenerationBasedStrategy(1).true_feedback_builder, TrueOnlyFeedback
    )


def test_generation_based_policies_distinguish_bundled_and_explicit():
    provider = SimpleNamespace(
        algorithm=object(),
        surrogate_manager=object(),
        acquisition=object(),
        evaluator=object(),
        cbmanager=None,
        evaluation_planner=None,
        feedback_builder=ComparatorWorstFallback(),
        feedback_builder_explicit=False,
    )
    strategy = GenerationBasedStrategy(1)
    pipeline = cast(Any, strategy).build_pipeline(cast(Any, provider))
    surrogate_generations = pipeline.find("surrogate_generations", recursive=True)
    true_generation = pipeline.find("true_generation", recursive=True)
    inner_feedback = surrogate_generations.body.stages[4]._builder
    outer_feedback = true_generation.stages[7]._builder
    assert isinstance(inner_feedback, PredictedFeedback)
    assert isinstance(outer_feedback, TrueOnlyFeedback)

    provider.feedback_builder = NoFeedback()
    provider.feedback_builder_explicit = True
    pipeline = cast(Any, strategy).build_pipeline(cast(Any, provider))
    assert isinstance(
        pipeline.find("true_generation", recursive=True).stages[7]._builder,
        NoFeedback,
    )
