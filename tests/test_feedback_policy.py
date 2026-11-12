import numpy as np

from saealib.core.contracts.observation import (
    HUMAN,
    IMPUTED,
    OK,
    SIMULATOR,
    SURROGATE,
    TRUE,
)
from saealib.core.contracts.observations import (
    ObservationBatch,
    ObservationRecord,
    ObservationRecords,
    ObservationSchema,
)
from saealib.execution.evaluator import EvaluationResult
from saealib.policies.feedback import (
    MISSING_VALUE_FALLBACK_POLICY,
    ComparatorWorstFallback,
    MixedFeedback,
    PredictedFeedback,
    SelectionPolicy,
    TrueOnlyFeedback,
)
from saealib.surrogate import PredictionChannel, SurrogatePrediction


def _state():
    from saealib.context import OptimizationState
    from saealib.population import (
        Archive,
        ParetoArchive,
        Population,
        PopulationAttribute,
    )
    from saealib.problem import Problem

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
        PopulationAttribute("cv", np.float64, (), default=0.0),
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


def _batch(records):
    return ObservationBatch(
        schema=ObservationSchema(objective_count=1),
        records=ObservationRecords.from_records(records),
    )


def _record(
    value,
    source=SURROGATE,
    *,
    candidate=10,
    quantity=("objective", 0),
    status=OK,
):
    return ObservationRecord(
        subject=("candidate", np.array([candidate], dtype=np.int64)),
        quantity=quantity,
        value=float(value),
        status=status,
        source=source,
        fidelity=1,
        provenance={"sequence": 1},
    )


def test_j6_default_source_priority_preserves_measured_evidence_order():
    assert SelectionPolicy().source_priority == (
        TRUE,
        HUMAN,
        SIMULATOR,
        SURROGATE,
        IMPUTED,
    )


def test_j6_total_tie_break_uses_batch_index_ascending():
    result = MixedFeedback().build(
        _state().offspring,
        None,
        _batch([_record(7.0), _record(8.0)]),
        [],
        ctx=_state(),
    )
    np.testing.assert_array_equal(result.f, [[7.0], [np.nan]])


def test_j6_true_source_precedes_surrogate():
    state = _state()
    evaluation = EvaluationResult(
        np.array([[4.0]]), np.empty((1, 0)), np.zeros(1), candidate_ids=np.array([10])
    )
    result = MixedFeedback().build(
        state.offspring,
        SurrogatePrediction({"objective": PredictionChannel(np.array([[5.0], [6.0]]))}),
        evaluation,
        [],
        ctx=state,
    )
    np.testing.assert_array_equal(result.f, [[4.0], [6.0]])
    np.testing.assert_array_equal(result.source, [0, 1])


def test_j6_materialize_preserves_unsorted_candidate_order():
    state = _state()
    candidates = state.offspring.extract(np.array([1, 0], dtype=np.intp))
    evaluation = EvaluationResult(
        np.array([[4.0]]), np.empty((1, 0)), np.zeros(1), candidate_ids=np.array([10])
    )
    result = MixedFeedback().build(
        candidates,
        SurrogatePrediction({"objective": PredictionChannel(np.array([[6.0], [5.0]]))}),
        evaluation,
        [],
        ctx=state,
    )
    np.testing.assert_array_equal(result.candidate_ids, [11, 10])
    np.testing.assert_array_equal(result.f, [[6.0], [4.0]])


def test_j6_comparator_fallback_is_imputed():
    state = _state()
    state.population.update_rows(np.array([0, 1]), {"f": np.array([[1.0], [3.0]])})
    result = ComparatorWorstFallback().build(
        state.offspring,
        SurrogatePrediction(
            {"objective": PredictionChannel(np.array([[1.0], [np.nan]]))}
        ),
        None,
        [],
        ctx=state,
    )
    np.testing.assert_array_equal(result.source, [1, 2])
    assert MISSING_VALUE_FALLBACK_POLICY.source == "imputed"


def test_j6_materializes_feedback_values():
    state = _state()
    values = np.array([[4.0], [6.0]])
    prediction = SurrogatePrediction({"objective": PredictionChannel(values)})
    result = MixedFeedback().build(state.offspring, prediction, None, [], ctx=state)
    values[0, 0] = 99.0
    np.testing.assert_array_equal(result.f, [[4.0], [6.0]])


def test_j6_materializes_f_g_cv_from_observation_batch():
    state = _state()
    records = ObservationRecords.from_records(
        [
            _record(4.0, source="true", candidate=10),
            _record(0.5, source="true", candidate=10, quantity=("constraint", 0)),
            _record(0.0, source="true", candidate=10, quantity=("cv", 0)),
            _record(6.0, source="true", candidate=11),
            _record(0.25, source="true", candidate=11, quantity=("constraint", 0)),
            _record(0.0, source="true", candidate=11, quantity=("cv", 0)),
        ]
    )
    batch = ObservationBatch(
        schema=ObservationSchema(
            objective_count=1,
            constraint_count=1,
            quantities={"cv": (0,)},
        ),
        records=records,
    )
    result = MixedFeedback().build(state.offspring, None, batch, [], ctx=state)
    np.testing.assert_array_equal(result.f, [[4.0], [6.0]])
    np.testing.assert_array_equal(result.g, [[0.5], [0.25]])
    np.testing.assert_array_equal(result.cv, [0.0, 0.0])
    records.columns["value"].flags.writeable = True
    records.columns["value"][0] = 99.0
    np.testing.assert_array_equal(result.f, [[4.0], [6.0]])


def test_j6_existing_builders_keep_visible_behavior():
    state = _state()
    evaluation = EvaluationResult(
        np.array([[4.0]]), np.empty((1, 0)), np.zeros(1), candidate_ids=np.array([10])
    )
    result = TrueOnlyFeedback().build(state.offspring, None, evaluation, [], ctx=state)
    np.testing.assert_array_equal(result.candidate_ids, [10])
    np.testing.assert_array_equal(result.source, [0])
    prediction = SurrogatePrediction(
        {"objective": PredictionChannel(np.array([[5.0], [6.0]]))}
    )
    predicted = PredictedFeedback().build(
        state.offspring, prediction, None, [], ctx=state
    )
    np.testing.assert_array_equal(predicted.f, [[5.0], [6.0]])
    np.testing.assert_array_equal(predicted.source, [1, 1])
    mixed = MixedFeedback().build(
        state.offspring, prediction, evaluation, [], ctx=state
    )
    np.testing.assert_array_equal(mixed.f, [[4.0], [6.0]])
    np.testing.assert_array_equal(mixed.source, [0, 1])
