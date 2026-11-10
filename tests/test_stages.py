"""Tests for concrete Stage execute() logic in saealib.stages."""

from typing import Any, cast

import numpy as np
import pytest

from saealib import (
    GA,
    CrossoverBLXAlpha,
    MutationUniform,
    SequentialSelection,
    TruncationSelection,
)
from saealib.acquisition import MeanPrediction
from saealib.callback import CallbackManager, PostEvaluationEvent
from saealib.comparators import SingleObjectiveComparator
from saealib.context import EvaluationPlanState, OptimizationState
from saealib.exceptions import EvaluationProtocolError
from saealib.execution.evaluator import (
    EvaluationRequest,
    EvaluationResult,
    Evaluator,
    SerialEvaluator,
)
from saealib.policies.evaluation import EvaluationPlan
from saealib.population import Archive, ParetoArchive, Population, PopulationAttribute
from saealib.problem import Problem
from saealib.stages import (
    AcquisitionStage,
    AskStage,
    EvaluationPlanStage,
    SurrogateFitStage,
    SurrogatePredictStage,
    TrueEvaluationStage,
)
from saealib.surrogate.prediction import SurrogatePrediction

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

DIM = 4
N_POP = 6
N_OBJ = 1

_ATTRS = [
    PopulationAttribute(name="x", dtype=np.float64, shape=(DIM,)),
    PopulationAttribute(name="f", dtype=np.float64, shape=(N_OBJ,)),
    PopulationAttribute(name="g", dtype=np.float64, shape=(0,)),
    PopulationAttribute(name="cv", dtype=np.float64, shape=()),
]


def _make_problem() -> Problem:
    problem = Problem(
        func=lambda x: np.array([np.sum(x**2)]),
        dim=DIM,
        n_obj=N_OBJ,
        direction=np.array([-1.0]),
        lb=[-5.0] * DIM,
        ub=[5.0] * DIM,
        comparator=SingleObjectiveComparator(),
    )
    problem.space.services.register("FeatureEncoder", _IdentityFeatureEncoder())
    return problem


class _IdentityFeatureEncoder:
    def encode(self, genomes):
        return genomes.array


def _make_state() -> OptimizationState:
    problem = _make_problem()
    rng = np.random.default_rng(0)
    xs = rng.uniform(-3.0, 3.0, size=(N_POP, DIM))
    fs = np.array([[np.sum(x**2)] for x in xs])

    pop = Population(_ATTRS, init_capacity=N_POP + 5)
    pop.extend({"x": xs, "f": fs, "g": np.zeros((N_POP, 0)), "cv": np.zeros(N_POP)})

    arc = Archive(_ATTRS, init_capacity=N_POP + 5)
    arc.extend({"x": xs, "f": fs, "g": np.zeros((N_POP, 0)), "cv": np.zeros(N_POP)})

    pareto_arc = ParetoArchive(
        _ATTRS, init_capacity=N_POP + 5, direction=np.array([-1.0])
    )
    return OptimizationState(
        problem=problem,
        population=pop,
        archive=arc,
        pareto_archive=pareto_arc,
        rng=np.random.default_rng(1),
    )


def _make_ga() -> GA:
    return GA(
        crossover=CrossoverBLXAlpha(prob=0.9, alpha=0.4),
        mutation=MutationUniform(prob_var=0.1),
        parent_selection=SequentialSelection(),
        survivor_selection=TruncationSelection(),
    )


class _StubEvaluator(Evaluator):
    """Evaluator returning known, distinguishable f/g/cv values (no computation)."""

    def __init__(self, f: np.ndarray, g: np.ndarray, cv: np.ndarray) -> None:
        self._f = f
        self._g = g
        self._cv = cv

    def evaluate_batch(self, x, problem):
        n = len(x)
        return EvaluationResult(f=self._f[:n], g=self._g[:n], cv=self._cv[:n])


class _MockSurrogateManager:
    def fit(self, archive, ctx=None):
        pass

    def predict(self, candidates_x, archive, ctx=None, *, refit=True):
        n = len(candidates_x)
        return SurrogatePrediction.objective(value=np.ones((n, 1)))


class _Planner:
    def __init__(self, plan):
        self.plan_result = plan

    def plan(self, candidates, acquisition, ctx):
        return self.plan_result


def _state_with_plannable_offspring() -> OptimizationState:
    return AskStage(_make_ga()).execute(_make_state())


def _plan_with_request(request_id: int = 0) -> EvaluationPlan:
    return EvaluationPlan(
        (
            EvaluationRequest(
                np.int64(request_id),
                np.array([0], dtype=np.int64),
                np.zeros((1, DIM), dtype=np.float64),
            ),
        )
    )


class TestEvaluationPlanStage:
    def test_none_planner_uses_default_evaluate_all(self):
        state = _state_with_plannable_offspring()
        planned = EvaluationPlanStage(planner=None).execute(state)

        assert planned.evaluation_plan is not None
        assert len(planned.evaluation_plan.requests) == 1
        assert state.offspring is not None
        assert len(planned.evaluation_plan.requests[0].candidate_ids) == len(
            state.offspring
        )
        assert planned.evaluation_request is planned.evaluation_plan.requests[0]

    def test_rejects_planner_returning_non_plan(self):
        state = _state_with_plannable_offspring()
        with pytest.raises(EvaluationProtocolError, match="must return EvaluationPlan"):
            EvaluationPlanStage(cast(Any, _Planner(object()))).execute(state)

    def test_rejects_request_id_collision_with_existing_handle(self):
        state = _state_with_plannable_offspring().replace(
            evaluation_handles={0: object()}
        )
        with pytest.raises(EvaluationProtocolError, match="request ID collides"):
            EvaluationPlanStage(cast(Any, _Planner(_plan_with_request()))).execute(
                state
            )

    def test_new_plan_confirmation_increments_decision_count(self):
        state = _state_with_plannable_offspring()
        assert state.decision_count == 0
        planned = EvaluationPlanStage(planner=None).execute(state)
        assert planned.decision_count == 1

    def test_multi_request_plan_still_counts_as_one_decision(self):
        state = _state_with_plannable_offspring()
        plan = EvaluationPlan(
            (
                EvaluationRequest(
                    np.int64(0),
                    np.array([0], dtype=np.int64),
                    np.zeros((1, DIM), dtype=np.float64),
                ),
                EvaluationRequest(
                    np.int64(1),
                    np.array([1], dtype=np.int64),
                    np.zeros((1, DIM), dtype=np.float64),
                ),
            )
        )
        planned = EvaluationPlanStage(cast(Any, _Planner(plan))).execute(state)
        assert planned.evaluation_plan is not None
        assert len(planned.evaluation_plan.requests) == 2
        assert planned.decision_count == 1

    def test_plan_continuation_does_not_increment_decision_count(self):
        state = _state_with_plannable_offspring()
        plan = EvaluationPlan(
            (
                EvaluationRequest(
                    np.int64(0),
                    np.array([0], dtype=np.int64),
                    np.zeros((1, DIM), dtype=np.float64),
                ),
                EvaluationRequest(
                    np.int64(1),
                    np.array([1], dtype=np.int64),
                    np.zeros((1, DIM), dtype=np.float64),
                ),
            )
        )
        state = state.replace(
            evaluation_plan=plan,
            evaluation_plan_state=EvaluationPlanState(
                submitted=(0,), completed=(0,), deferred=(1,)
            ),
            evaluation_request=None,
            decision_count=5,
        )
        continued = EvaluationPlanStage(planner=None).execute(state)
        assert continued.evaluation_request is plan.requests[1]
        assert continued.decision_count == 5

    def test_plan_continuation_passthrough_does_not_increment_decision_count(self):
        state = _state_with_plannable_offspring()
        plan = _plan_with_request()
        state = state.replace(
            evaluation_plan=plan,
            evaluation_plan_state=EvaluationPlanState(deferred=(0,)),
            evaluation_request=plan.requests[0],
            decision_count=5,
        )
        unchanged = EvaluationPlanStage(planner=None).execute(state)
        assert unchanged.decision_count == 5


# ---------------------------------------------------------------------------
# AskStage — _DispatchProxy.dispatch when cbmanager is None
# ---------------------------------------------------------------------------


class TestAskStageNoCbmanager:
    """AskStage with cbmanager=None exercises _DispatchProxy.dispatch(None path)."""

    def test_execute_sets_offspring(self):
        state = _make_state()
        stage = AskStage(_make_ga(), cbmanager=None)
        new_state = stage.execute(state)
        assert new_state.offspring is not None

    def test_execute_offspring_count_matches_population(self):
        state = _make_state()
        stage = AskStage(_make_ga(), cbmanager=None)
        new_state = stage.execute(state)
        assert new_state.offspring is not None
        assert len(new_state.offspring) == len(state.population)


# ---------------------------------------------------------------------------
# SurrogateFitStage
# ---------------------------------------------------------------------------


class TestSurrogateFitStage:
    def test_execute_calls_fit(self):
        fit_called = [False]

        class _TrackedSM(_MockSurrogateManager):
            def fit(self, archive, ctx=None):
                fit_called[0] = True

        state = _make_state()
        SurrogateFitStage(_TrackedSM()).execute(state)  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
        assert fit_called[0]

    def test_execute_returns_same_state_object(self):
        state = _make_state()
        result = SurrogateFitStage(_MockSurrogateManager()).execute(state)  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
        assert result is state


# ---------------------------------------------------------------------------
# SurrogatePredictStage / AcquisitionStage — cbmanager=None branches
# ---------------------------------------------------------------------------


class TestSurrogatePredictAcquisitionStageNoCbmanager:
    """cbmanager=None skips SurrogateStartEvent / SurrogateEndEvent /
    AcquisitionStartEvent / AcquisitionEndEvent dispatch."""

    def test_execute_sets_scores_and_predictions(self):
        state = _make_state()
        state = AskStage(_make_ga(), cbmanager=None).execute(state)

        sm = _MockSurrogateManager()
        acquisition = MeanPrediction()
        state = SurrogatePredictStage(sm, cbmanager=None).execute(state)  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
        assert state.predictions is not None

        stage = AcquisitionStage(acquisition, cbmanager=None)
        new_state = stage.execute(state)
        assert new_state.scores is not None

    def test_scores_length_matches_offspring(self):
        state = _make_state()
        state = AskStage(_make_ga(), cbmanager=None).execute(state)
        assert state.offspring is not None
        n = len(state.offspring)

        sm = _MockSurrogateManager()
        acquisition = MeanPrediction()
        state = SurrogatePredictStage(sm, cbmanager=None).execute(state)  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
        stage = AcquisitionStage(acquisition, cbmanager=None)
        new_state = stage.execute(state)
        assert new_state.scores is not None
        assert len(new_state.scores) == n


# ---------------------------------------------------------------------------
# AcquisitionStage — empty candidates and prepare() caching
# ---------------------------------------------------------------------------


class _SpyAcquisition:
    """Records prepare()/evaluate() call counts and rng draws."""

    direction_sensitive = False

    def __init__(self):
        self.prepare_calls = 0
        self.evaluate_calls = 0

    def prepare(self, archive, ctx=None):
        self.prepare_calls += 1
        if ctx is not None:
            ctx.rng.random()  # consume rng so empty-input non-consumption is checkable
        return "prepared"

    def evaluate(self, candidates_x, prediction, archive, ctx=None, *, prepared=None):
        self.evaluate_calls += 1
        assert prepared == "prepared"
        n = len(candidates_x)
        from saealib.acquisition import AcquisitionResult

        return AcquisitionResult(scores=np.ones(n))


class TestAcquisitionStageEmptyAndCaching:
    def test_empty_offspring_skips_prepare_and_evaluate(self):
        state = _make_state()
        state = AskStage(_make_ga(), cbmanager=None).execute(state)
        assert state.offspring is not None
        state = state.replace(
            offspring=state.offspring.extract(np.array([], dtype=int))
        )

        acq = _SpyAcquisition()
        stage = AcquisitionStage(acq)  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
        new_state = stage.execute(state)

        assert acq.prepare_calls == 0
        assert acq.evaluate_calls == 0
        assert new_state.scores is not None
        assert len(new_state.scores) == 0

    def test_empty_offspring_does_not_advance_rng(self):
        state = _make_state()
        state = AskStage(_make_ga(), cbmanager=None).execute(state)
        assert state.offspring is not None
        state = state.replace(
            offspring=state.offspring.extract(np.array([], dtype=int))
        )
        rng_state_before = state.rng.bit_generator.state

        acq = _SpyAcquisition()
        stage = AcquisitionStage(acq)  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
        stage.execute(state)

        assert state.rng.bit_generator.state == rng_state_before

    def test_prepare_cached_within_same_generation_and_archive_version(self):
        state = _make_state()
        state = AskStage(_make_ga(), cbmanager=None).execute(state)

        acq = _SpyAcquisition()
        stage = AcquisitionStage(acq)  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
        stage.execute(state)
        stage.execute(state)  # same gen, same archive versions

        assert acq.prepare_calls == 1
        assert acq.evaluate_calls == 2

    def test_prepare_recomputed_on_generation_change(self):
        state = _make_state()
        state = AskStage(_make_ga(), cbmanager=None).execute(state)

        acq = _SpyAcquisition()
        stage = AcquisitionStage(acq)  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
        stage.execute(state)
        stage.execute(state.replace(gen=state.gen + 1))

        assert acq.prepare_calls == 2

    def test_prepare_recomputed_on_archive_structure_change(self):
        state = _make_state()
        state = AskStage(_make_ga(), cbmanager=None).execute(state)

        acq = _SpyAcquisition()
        stage = AcquisitionStage(acq)  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
        stage.execute(state)
        state.archive.add(x=np.zeros(DIM), f=np.zeros(N_OBJ), g=np.zeros(0), cv=0.0)
        stage.execute(state)

        assert acq.prepare_calls == 2


# ---------------------------------------------------------------------------
# TrueEvaluationStage — n_eval as int and cbmanager=None
# ---------------------------------------------------------------------------


class TestTrueEvaluationStage:
    def _state_with_offspring(self):
        state = _make_state()
        return AskStage(_make_ga(), cbmanager=None).execute(state)

    def test_n_eval_as_int_limits_evaluation(self):
        state = self._state_with_offspring()
        n_eval = 2
        stage = TrueEvaluationStage(SerialEvaluator(), n_eval=n_eval)
        new_state = stage.execute(state)
        assert new_state.fe == n_eval
        assert new_state.evaluated_offspring is not None
        assert len(new_state.evaluated_offspring) == n_eval

    def test_n_eval_as_int_without_cbmanager(self):
        state = self._state_with_offspring()
        stage = TrueEvaluationStage(SerialEvaluator(), cbmanager=None, n_eval=2)
        new_state = stage.execute(state)
        assert new_state.fe == 2

    def test_n_eval_as_int_capped_by_offspring_size(self):
        state = self._state_with_offspring()
        assert state.offspring is not None
        n_offspring = len(state.offspring)
        stage = TrueEvaluationStage(SerialEvaluator(), n_eval=n_offspring + 100)
        new_state = stage.execute(state)
        assert new_state.fe == n_offspring

    def test_bulk_write_matches_evaluator_result(self):
        """f/g/cv written by the bulk assignment exactly match the evaluator output."""
        state = self._state_with_offspring()
        assert state.offspring is not None
        n_offspring = len(state.offspring)
        n_eval = n_offspring - 1

        f = np.arange(n_offspring, dtype=float).reshape(n_offspring, N_OBJ)
        g = np.zeros((n_offspring, 0))
        cv = np.arange(100.0, 100.0 + n_offspring)

        stage = TrueEvaluationStage(_StubEvaluator(f, g, cv), n_eval=n_eval)
        new_state = stage.execute(state)

        assert new_state.evaluated_offspring is not None
        np.testing.assert_array_equal(new_state.evaluated_offspring.f, f[:n_eval])
        np.testing.assert_array_equal(new_state.evaluated_offspring.g, g[:n_eval])
        np.testing.assert_array_equal(new_state.evaluated_offspring.cv, cv[:n_eval])

        assert new_state.offspring is not None
        np.testing.assert_array_equal(new_state.offspring.f[:n_eval], f[:n_eval])
        np.testing.assert_array_equal(new_state.offspring.g[:n_eval], g[:n_eval])
        np.testing.assert_array_equal(new_state.offspring.cv[:n_eval], cv[:n_eval])

    def test_candidates_beyond_n_are_untouched(self):
        """Only the first n candidates are written; the rest keep their prior values."""
        state = self._state_with_offspring()
        assert state.offspring is not None
        n_offspring = len(state.offspring)
        n_eval = n_offspring - 1

        before_f = np.array(state.offspring.f, copy=True)
        before_g = np.array(state.offspring.g, copy=True)
        before_cv = np.array(state.offspring.cv, copy=True)

        f = np.arange(n_offspring, dtype=float).reshape(n_offspring, N_OBJ)
        g = np.zeros((n_offspring, 0))
        cv = np.arange(100.0, 100.0 + n_offspring)

        stage = TrueEvaluationStage(_StubEvaluator(f, g, cv), n_eval=n_eval)
        new_state = stage.execute(state)

        assert new_state.offspring is not None
        np.testing.assert_array_equal(new_state.offspring.f[n_eval:], before_f[n_eval:])
        np.testing.assert_array_equal(new_state.offspring.g[n_eval:], before_g[n_eval:])
        np.testing.assert_array_equal(
            new_state.offspring.cv[n_eval:], before_cv[n_eval:]
        )

    def test_bulk_write_aligns_multi_column_arrays(self):
        """With n_obj > 1 / n_constraints > 1, columns must not be transposed/mixed."""
        n = 4
        attrs_mo = [
            PopulationAttribute(name="x", dtype=np.float64, shape=(DIM,)),
            PopulationAttribute(name="f", dtype=np.float64, shape=(2,)),
            PopulationAttribute(name="g", dtype=np.float64, shape=(3,)),
            PopulationAttribute(name="cv", dtype=np.float64, shape=()),
        ]
        pop = Population(attrs_mo, init_capacity=n + 2)
        pop.extend({"x": np.zeros((n, DIM))})  # f/g/cv fall back to the NaN default

        state = self._state_with_offspring()
        state = state.replace(offspring=pop)

        # _StubEvaluator ignores `problem` entirely, so the state's n_obj=1
        # problem (from _make_state) is harmless here.
        f = np.arange(n * 2, dtype=float).reshape(n, 2)
        g = np.arange(100.0, 100.0 + n * 3).reshape(n, 3)
        cv = np.arange(200.0, 200.0 + n)

        stage = TrueEvaluationStage(_StubEvaluator(f, g, cv))
        new_state = stage.execute(state)

        assert new_state.evaluated_offspring is not None
        np.testing.assert_array_equal(new_state.evaluated_offspring.f, f)
        np.testing.assert_array_equal(new_state.evaluated_offspring.g, g)
        np.testing.assert_array_equal(new_state.evaluated_offspring.cv, cv)

    def test_zero_constraints_does_not_crash_g_assignment(self):
        """g has shape (n, 0) when the problem defines no constraints."""
        state = self._state_with_offspring()
        assert state.offspring is not None
        n_offspring = len(state.offspring)

        f = np.zeros((n_offspring, N_OBJ))
        g = np.zeros((n_offspring, 0))
        cv = np.zeros(n_offspring)

        stage = TrueEvaluationStage(_StubEvaluator(f, g, cv))
        new_state = stage.execute(state)
        assert new_state.evaluated_offspring is not None
        assert new_state.evaluated_offspring.g.shape == (n_offspring, 0)

    def test_value_version_bumped_exactly_once(self):
        """A single mod_value() call still signals the population as changed."""
        state = self._state_with_offspring()
        assert state.offspring is not None
        v0 = state.offspring._value_version

        stage = TrueEvaluationStage(SerialEvaluator(), n_eval=2)
        new_state = stage.execute(state)

        assert new_state.offspring is not None
        assert new_state.offspring._value_version == v0 + 1

    def test_post_evaluation_event_receives_written_values(self):
        """PostEvaluationEvent.offspring carries the bulk-written f/g/cv values."""
        state = self._state_with_offspring()
        assert state.offspring is not None
        n_offspring = len(state.offspring)
        n_eval = n_offspring - 1

        f = np.arange(n_offspring, dtype=float).reshape(n_offspring, N_OBJ)
        g = np.zeros((n_offspring, 0))
        cv = np.arange(100.0, 100.0 + n_offspring)

        received = []
        cbmanager = CallbackManager()
        cbmanager.register(
            PostEvaluationEvent, lambda event: received.append(event.offspring)
        )

        stage = TrueEvaluationStage(
            _StubEvaluator(f, g, cv), cbmanager=cbmanager, n_eval=n_eval
        )
        stage.execute(state)

        assert len(received) == 1
        np.testing.assert_array_equal(received[0].f, f[:n_eval])
        np.testing.assert_array_equal(received[0].cv, cv[:n_eval])
