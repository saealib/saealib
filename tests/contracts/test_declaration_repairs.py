"""Runtime-observed declaration repairs for built-in policies.

Each repaired declaration is verified against an attributed runtime record
(test-only instrumentation) or a narrow probe that can attribute the effect,
not against agreement between handwritten declarations and static inference.
"""

from __future__ import annotations

import numpy as np
import pytest
from instrumented_store import (
    Recorder,
    attach_instrumentation,
    instrumentation_scope,
    instrumented,
    instrumented_component,
)

from saealib import (
    GA,
    Archive,
    CrossoverBLXAlpha,
    DirectStrategy,
    EvaluateAll,
    LHSInitializer,
    MutationUniform,
    OptimizationState,
    Optimizer,
    ParetoArchive,
    Population,
    PopulationAttribute,
    Problem,
    SequentialSelection,
    SingleObjectiveComparator,
    Termination,
    TruncationSelection,
    max_gen,
)
from saealib.core.state import (
    POPULATIONS_MAIN,
    RUNTIME_CANDIDATE_ID_ALLOCATOR,
    RUNTIME_REQUEST_ID_ALLOCATOR,
)
from saealib.execution import AsyncEvaluationScheduler
from saealib.execution.evaluator import AsyncEvaluator, SerialEvaluator
from saealib.policies.evaluation import _RequestPlanner
from saealib.policies.feedback import ComparatorWorstFallback, MixedFeedback
from saealib.surrogate.training_set import (
    LevelBasedSet,
    PairwiseComparisonSet,
    ReferencePointComparisonSet,
    TopKBipartitionSet,
)


def _problem() -> Problem:
    return Problem(
        func=lambda x: np.array([np.sum(np.asarray(x) ** 2)]),
        dim=2,
        n_obj=1,
        direction=np.array([-1.0]),
        lb=[-2.0] * 2,
        ub=[2.0] * 2,
        comparator=SingleObjectiveComparator(),
    )


def _attrs() -> list[PopulationAttribute]:
    return [
        PopulationAttribute("x", np.float64, (2,)),
        PopulationAttribute("f", np.float64, (1,)),
        PopulationAttribute("g", np.float64, (0,)),
        PopulationAttribute("cv", np.float64, (), default=0.0),
        PopulationAttribute("id", np.int64, (), default=-1),
    ]


def _initial_state() -> OptimizationState:
    problem = _problem()
    population = Population(_attrs(), init_capacity=4)
    population._extend_internal(
        {
            "x": np.array([[0.1, 0.2], [0.3, 0.4]]),
            "f": np.array([[1.0], [2.0]]),
            "g": np.zeros((2, 0)),
            "cv": np.zeros(2),
            "id": np.array([1, 2]),
        },
        preserve_ids=True,
    )
    archive = Archive(_attrs(), init_capacity=4)
    archive._extend_internal(
        {
            "x": np.array([[0.1, 0.2], [0.3, 0.4]]),
            "f": np.array([[1.0], [2.0]]),
            "g": np.zeros((2, 0)),
            "cv": np.zeros(2),
            "id": np.array([1, 2]),
        },
        preserve_ids=True,
    )
    pareto = ParetoArchive(_attrs(), init_capacity=2, direction=problem.direction)
    return OptimizationState(
        problem=problem,
        populations={"main": population},
        archives={"main": archive, "pareto": pareto},
        rng=np.random.default_rng(0),
    )


def _unassigned_candidates(state: OptimizationState) -> Population:
    candidates = state.population.empty_like(capacity=2)
    candidates._extend_internal(
        {
            "x": np.array([[0.5, 0.5], [0.6, 0.6]]),
            "f": np.array([[np.nan], [np.nan]]),
            "g": np.zeros((2, 0)),
            "cv": np.zeros(2),
            "id": np.array([-1, -1]),
        },
        preserve_ids=True,
    )
    return candidates


def _incomplete_candidates(state: OptimizationState) -> Population:
    candidates = state.population.empty_like(capacity=2)
    candidates._extend_internal(
        {
            "x": np.array([[0.5, 0.5], [0.6, 0.6]]),
            "f": np.array([[np.nan], [np.nan]]),
            "g": np.zeros((2, 0)),
            "cv": np.zeros(2),
            "id": np.array([10, 11]),
        },
        preserve_ids=True,
    )
    return candidates


def _optimizer(planner, *, asynchronous: bool) -> Optimizer:
    optimizer = (
        Optimizer(_problem(), seed=7)
        .set_initializer(LHSInitializer(4, 4, seed=7))
        .set_algorithm(
            GA(
                crossover=CrossoverBLXAlpha(0.9, 0.5),
                mutation=MutationUniform(0.1),
                parent_selection=SequentialSelection(),
                survivor_selection=TruncationSelection(),
            )
        )
        .set_strategy(DirectStrategy())
        .set_evaluation_planner(planner)
        .set_termination(Termination(max_gen(2)))
    )
    if asynchronous:
        evaluator = AsyncEvaluator(SerialEvaluator(), max_workers=1)
        optimizer.set_evaluator(evaluator)
        optimizer.set_async_evaluation_scheduler(
            AsyncEvaluationScheduler(evaluator, max_pending=2)
        )
    return optimizer


@pytest.mark.parametrize("asynchronous", [False, True])
def test_request_planner_declares_its_allocator_state_use(asynchronous) -> None:
    planner = EvaluateAll()
    recorder = Recorder()
    optimizer = _optimizer(planner, asynchronous=asynchronous)
    planner.plan = instrumented("planner.plan", recorder)(planner.plan)  # ty: ignore[invalid-assignment]
    with instrumentation_scope(recorder):
        optimizer.run()

    observed = recorder.keys("planner.plan")
    assert RUNTIME_REQUEST_ID_ALLOCATOR in observed
    declared = planner.contract().state
    assert RUNTIME_REQUEST_ID_ALLOCATOR in set(declared.reads)
    assert RUNTIME_REQUEST_ID_ALLOCATOR in set(declared.writes)
    assert RUNTIME_CANDIDATE_ID_ALLOCATOR in set(declared.reads)
    assert RUNTIME_CANDIDATE_ID_ALLOCATOR in set(declared.writes)


def test_concrete_planners_inherit_the_request_planner_contract() -> None:
    from saealib.policies.evaluation import (
        EvaluateAll as _EvaluateAll,
    )
    from saealib.policies.evaluation import (
        FidelityEvaluation,
        FidelityPromotion,
        RatioEvaluation,
        RepeatedEvaluation,
        TopKEvaluation,
    )

    assert "contract" not in vars(_EvaluateAll)
    assert "contract" not in vars(TopKEvaluation)
    assert "contract" not in vars(RatioEvaluation)
    assert "contract" not in vars(RepeatedEvaluation)
    assert "contract" not in vars(FidelityEvaluation)
    assert "contract" not in vars(FidelityPromotion)
    assert isinstance(_EvaluateAll(), _RequestPlanner)


def test_request_planner_missing_id_branch_is_observed_and_declared() -> None:
    planner = EvaluateAll()
    recorder = Recorder()
    state = _initial_state()
    candidates = _unassigned_candidates(state)
    attach_instrumentation(state, recorder)
    with instrumented_component("planner.plan", recorder):
        plan = planner.plan(candidates, None, ctx=state)

    assigned = np.asarray(plan.requests[0].candidate_ids)
    assert len(assigned) == 2
    assert np.all(assigned >= 0)
    assert np.all(np.asarray(candidates.get_array("id")) >= 0)
    observed = recorder.keys("planner.plan")
    assert RUNTIME_CANDIDATE_ID_ALLOCATOR in observed
    declared = planner.contract().state
    assert RUNTIME_CANDIDATE_ID_ALLOCATOR in set(declared.reads) | set(declared.writes)


def test_planner_id_write_targets_the_input_candidates() -> None:
    """The missing-ID branch mutates the planner's candidates argument.

    The candidates object aliases a state value in a real run, so the effect
    is an in-place write to that state value; no contract field distinguishes
    mutating an input object from writing the declared state keys.  The probe
    records the effect and the representational gap instead of inventing a
    field.
    """
    planner = EvaluateAll()
    state = _initial_state()
    candidates = _unassigned_candidates(state)
    before = candidates.get_array("id").copy()

    planner.plan(candidates, None, ctx=state)

    assert np.all(np.asarray(before) < 0)
    assert np.all(np.asarray(candidates.get_array("id")) >= 0)


def test_comparator_worst_fallback_population_read_is_observed_and_declared() -> None:
    builder = ComparatorWorstFallback(MixedFeedback())
    recorder = Recorder()
    state = _initial_state()
    attach_instrumentation(state, recorder)
    candidates = _incomplete_candidates(state)
    with instrumented_component("builder.build", recorder):
        result = builder.build(
            candidates, None, None, np.array([], dtype=np.int64), ctx=state
        )

    assert np.all(np.isfinite(np.asarray(result.f)))
    assert POPULATIONS_MAIN in recorder.keys("builder.build", "read")
    assert POPULATIONS_MAIN in set(builder.contract().state.reads)


def test_comparator_worst_fallback_declares_no_write() -> None:
    builder = ComparatorWorstFallback(MixedFeedback())
    assert set(builder.contract().state.writes) == set()


@pytest.mark.parametrize(
    "set_class",
    [
        TopKBipartitionSet,
        LevelBasedSet,
        PairwiseComparisonSet,
        ReferencePointComparisonSet,
    ],
    ids=lambda value: value.__name__,
)
def test_comparator_using_training_sets_require_comparison_service(set_class) -> None:
    instance = (
        set_class(n_pairs=3) if set_class is PairwiseComparisonSet else set_class()
    )
    requirements = {
        requirement.name for requirement in instance.contract().required_services
    }
    assert "ComparisonService" in requirements


def test_sets_without_comparator_use_require_no_comparison_service() -> None:
    from saealib.surrogate.training_set import ArchiveObjectiveSet, KNNObjectiveSet

    for set_class in (ArchiveObjectiveSet, KNNObjectiveSet):
        requirements = {
            requirement.name for requirement in set_class().contract().required_services
        }
        assert "ComparisonService" not in requirements


def test_comparison_service_resolves_to_the_problem_comparator() -> None:
    state = _initial_state()
    set_ = LevelBasedSet()
    data = set_.build(state.archive, state.population, ctx=state)

    assert data.train_y.shape[0] == len(state.archive)
    comparator = state.problem.comparator
    sorted_idx = comparator.sort_population(state.archive)
    assert sorted_idx.shape == (len(state.archive),)


def test_default_optimizers_gain_no_service_errors_from_training_sets() -> None:
    optimizer = Optimizer(_problem())
    optimizer.resolve_defaults()
    errors = [
        diagnostic
        for diagnostic in optimizer.contract_diagnostics()
        if diagnostic.severity.value == "error"
    ]
    assert errors == []


def _observe_declaration(name, state, call):
    recorder = Recorder()
    attach_instrumentation(state, recorder)
    with instrumented_component(name, recorder):
        call()
    recorder.flush()
    return recorder.keys(name, "read"), recorder.keys(name, "write")


def _prediction(x):
    from saealib.surrogate import PredictionChannel, SurrogatePrediction

    return SurrogatePrediction(
        {
            "objective": PredictionChannel(
                value=np.array([[1.0], [2.0]]),
                std=np.full((2, 1), 0.5),
                covariance=np.eye(2) * 0.25,
            )
        },
        x=np.asarray(x),
    )


@pytest.mark.parametrize(
    "factory",
    [
        lambda: __import__(
            "saealib.acquisition.ei", fromlist=["ExpectedImprovement"]
        ).ExpectedImprovement(),
        lambda: __import__(
            "saealib.acquisition.lcb", fromlist=["LowerConfidenceBound"]
        ).LowerConfidenceBound(),
        lambda: __import__(
            "saealib.acquisition.mean", fromlist=["CORSDistance"]
        ).CORSDistance(),
        lambda: __import__(
            "saealib.acquisition.mean", fromlist=["MeanPrediction"]
        ).MeanPrediction(),
        lambda: __import__(
            "saealib.acquisition.pof", fromlist=["ProbabilityOfFeasibility"]
        ).ProbabilityOfFeasibility(),
        lambda: __import__(
            "saealib.acquisition.pof", fromlist=["ProductOfFeasibility"]
        ).ProductOfFeasibility(),
        lambda: __import__(
            "saealib.acquisition.smsego", fromlist=["SMSEGOAcquisition"]
        ).SMSEGOAcquisition(),
        lambda: __import__(
            "saealib.acquisition.uncertainty", fromlist=["MaxUncertainty"]
        ).MaxUncertainty(),
        lambda: __import__(
            "saealib.acquisition.archive_based", fromlist=["NoveltyAcquisition"]
        ).NoveltyAcquisition(),
    ],
    ids=[
        "expected-improvement",
        "lower-confidence-bound",
        "cors-distance",
        "mean-prediction",
        "probability-of-feasibility",
        "product-of-feasibility",
        "smsego",
        "max-uncertainty",
        "novelty",
    ],
)
def test_acquisition_declarations_match_runtime_observation(factory) -> None:
    acquisition = factory()
    state = _initial_state()
    archive = state.archive
    candidates = np.asarray(state.population.get_array("x"))
    prediction = _prediction(candidates)
    reads, writes = _observe_declaration(
        "acquisition",
        state,
        lambda: acquisition.evaluate(
            candidates,
            prediction,
            archive,
            prepared=acquisition.prepare(archive, ctx=state),
            ctx=state,
        ),
    )
    declared = acquisition.contract().state
    assert reads <= set(declared.reads)
    assert writes <= set(declared.writes)


def test_cors_distance_declaration_covers_decision_count() -> None:
    from saealib.acquisition.mean import CORSDistance
    from saealib.core.state import RUNTIME_DECISION_COUNT, RUNTIME_RNG

    acquisition = CORSDistance()
    state = _initial_state()
    archive = state.archive
    candidates = np.asarray(state.population.get_array("x"))
    prediction = _prediction(candidates)
    reads, _ = _observe_declaration(
        "acquisition",
        state,
        lambda: acquisition.evaluate(
            candidates,
            prediction,
            archive,
            prepared=acquisition.prepare(archive, ctx=state),
            ctx=state,
        ),
    )
    assert RUNTIME_RNG in reads
    assert RUNTIME_DECISION_COUNT in reads
    assert reads <= set(acquisition.contract().state.reads)


def test_beta_schedule_declaration_covers_decision_count() -> None:
    from saealib.acquisition.lcb import LowerConfidenceBound
    from saealib.core.state import RUNTIME_DECISION_COUNT, RUNTIME_RNG

    acquisition = LowerConfidenceBound(beta_schedule=lambda t: 1.0)
    state = _initial_state()
    archive = state.archive
    candidates = np.asarray(state.population.get_array("x"))
    prediction = _prediction(candidates)
    reads, _ = _observe_declaration(
        "acquisition",
        state,
        lambda: acquisition.evaluate(
            candidates,
            prediction,
            archive,
            prepared=acquisition.prepare(archive, ctx=state),
            ctx=state,
        ),
    )
    assert RUNTIME_RNG in reads
    assert RUNTIME_DECISION_COUNT in reads
    assert reads <= set(acquisition.contract().state.reads)


def test_composite_acquisition_declaration_covers_rng() -> None:
    from saealib.acquisition.base import CompositeAcquisition
    from saealib.acquisition.batch import BatchExpectedImprovement
    from saealib.core.state import RUNTIME_RNG

    acquisition = CompositeAcquisition(
        {"objective": BatchExpectedImprovement(n_draws=16)},
        lambda arrays: arrays[0],
    )
    state = _initial_state()
    archive = state.archive
    candidates = np.asarray(state.population.get_array("x"))
    prediction = _prediction(candidates)
    reads, writes = _observe_declaration(
        "acquisition",
        state,
        lambda: acquisition.evaluate(
            candidates,
            prediction,
            archive,
            prepared=acquisition.prepare(archive, ctx=state),
            ctx=state,
        ),
    )
    declared = acquisition.contract().state
    assert RUNTIME_RNG in reads
    assert RUNTIME_RNG in writes
    assert reads <= set(declared.reads)
    assert writes <= set(declared.writes)


def test_composite_surrogate_manager_declaration_covers_child_rng() -> None:
    from saealib.core.state import RUNTIME_RNG
    from saealib.surrogate.base import ComparisonSurrogate
    from saealib.surrogate.manager import (
        CompositeSurrogateManager,
        PairwiseSurrogateManager,
    )

    class _PairwiseSurrogate(ComparisonSurrogate):
        def fit(self, train_x, train_y) -> None:
            pass

        def predict_proba(self, test_x):
            from saealib.surrogate import PredictionChannel, SurrogatePrediction

            return SurrogatePrediction(
                {"win_rate": PredictionChannel(value=np.zeros((len(test_x), 1)))}
            )

    manager = CompositeSurrogateManager(
        {
            "objective": PairwiseSurrogateManager(
                _PairwiseSurrogate(),
                PairwiseComparisonSet(n_pairs=1),
            )
        }
    )
    state = _initial_state()
    state.archive._extend_internal(
        {
            "id": np.array([3]),
            "x": np.array([[0.5, 0.5]]),
            "f": np.array([[3.0]]),
            "g": np.empty((1, 0)),
            "cv": np.zeros(1),
        },
        preserve_ids=True,
    )
    archive = state.archive
    reads, writes = _observe_declaration(
        "manager", state, lambda: manager.fit(archive, ctx=state)
    )
    declared = manager.contract().state
    assert RUNTIME_RNG in reads
    assert RUNTIME_RNG in writes
    assert reads <= set(declared.reads)
    assert writes <= set(declared.writes)


def test_feedback_declarations_match_runtime_observation() -> None:
    from saealib.execution.evaluator import EvaluationResult
    from saealib.policies.feedback import TrueOnlyFeedback

    state = _initial_state()
    candidates = _incomplete_candidates(state)
    prediction = _prediction(candidates.get_array("x"))
    evaluation = EvaluationResult(
        np.array([[1.0], [2.0]]),
        np.empty((2, 0)),
        np.zeros(2),
        candidate_ids=np.array([10, 11]),
    )
    for builder in (TrueOnlyFeedback(), MixedFeedback()):
        reads, writes = _observe_declaration(
            "builder",
            state,
            lambda: builder.build(candidates, prediction, evaluation, [], ctx=state),
        )
        declared = builder.contract().state
        assert reads <= set(declared.reads)
        assert writes <= set(declared.writes)
        assert not declared.reads
        assert not declared.writes
