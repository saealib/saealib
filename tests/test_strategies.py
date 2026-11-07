"""
Tests for optimization strategies.

Tests cover:
- GenerationBasedStrategy: generation count, fe count, archive growth
- PreSelectionStrategy: fe count equals n_select, archive growth, n_select cap
"""

import operator
from typing import Any, cast

import numpy as np

from saealib import (
    GA,
    CrossoverBLXAlpha,
    LHSInitializer,
    MutationUniform,
    Optimizer,
    SequentialSelection,
    TruncationSelection,
)
from saealib.acquisition import AcquisitionResult, MeanPrediction
from saealib.acquisition.archive_based import (
    InverseDensityAcquisition,
    NoveltyAcquisition,
)
from saealib.acquisition.base import _UNSET, AcquisitionFunction
from saealib.callback import AcquisitionEndEvent, CallbackManager
from saealib.comparators import SingleObjectiveComparator
from saealib.context import OptimizationState
from saealib.execution.evaluator import SerialEvaluator
from saealib.pipeline import Stage
from saealib.policies import EvaluationPlanner, FeedbackBuilder
from saealib.population import Archive, ParetoArchive, Population, PopulationAttribute
from saealib.problem import Problem
from saealib.stages import stage_component
from saealib.strategies.gb import GenerationBasedStrategy
from saealib.strategies.ib import IndividualBasedStrategy
from saealib.strategies.ps import PreSelectionStrategy
from saealib.surrogate.manager import (
    GlobalSurrogateManager,
    LocalSurrogateManager,
    SurrogateManager,
)
from saealib.surrogate.prediction import SurrogatePrediction
from saealib.surrogate.rbf import RBFSurrogate
from saealib.surrogate.rbf_kernels import GaussianKernel
from saealib.termination import Termination, max_gen

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

DIM = 6
N_OBJ = 1
N_POP = 10

_ATTRS = [
    PopulationAttribute(name="x", dtype=np.float64, shape=(DIM,)),
    PopulationAttribute(name="f", dtype=np.float64, shape=(N_OBJ,)),
    PopulationAttribute(name="g", dtype=np.float64, shape=(0,)),
    PopulationAttribute(name="cv", dtype=np.float64, shape=()),
]


def _make_problem() -> Problem:
    return Problem(
        func=lambda x: np.array([np.sum(x**2)]),
        dim=DIM,
        n_obj=N_OBJ,
        direction=np.array([-1.0]),
        lb=[-5.0] * DIM,
        ub=[5.0] * DIM,
        comparator=SingleObjectiveComparator(),
    )


def _make_ctx(n_pop: int = N_POP, rng_seed: int = 0) -> OptimizationState:
    problem = _make_problem()
    rng = np.random.default_rng(rng_seed)

    pop = Population(_ATTRS, init_capacity=n_pop + 5)
    xs = rng.uniform(-3.0, 3.0, size=(n_pop, DIM))
    fs = np.array([[np.sum(x**2)] for x in xs])
    pop.extend({"x": xs, "f": fs, "g": np.zeros((n_pop, 0)), "cv": np.zeros(n_pop)})

    arc = Archive(_ATTRS, init_capacity=n_pop + 5)
    arc.extend({"x": xs, "f": fs, "g": np.zeros((n_pop, 0)), "cv": np.zeros(n_pop)})

    pareto_arc = ParetoArchive(
        _ATTRS, init_capacity=n_pop + 5, direction=np.array([-1.0])
    )
    return OptimizationState(
        problem=problem,
        population=pop,
        archive=arc,
        pareto_archive=pareto_arc,
        rng=np.random.default_rng(rng_seed + 1),
    )


def _make_ga() -> GA:
    return GA(
        crossover=CrossoverBLXAlpha(prob=0.9, alpha=0.4),
        mutation=MutationUniform(prob_var=0.1),
        parent_selection=SequentialSelection(),
        survivor_selection=TruncationSelection(),
    )


class _LinspaceAcquisition(AcquisitionFunction):
    """Descending linspace scores for the strategy selection test."""

    def evaluate(self, candidates_x, prediction, archive, ctx=None, *, prepared=_UNSET):
        n = len(candidates_x)
        return AcquisitionResult(scores=np.linspace(1.0, 0.0, n))


class _MockSurrogateManager:
    """Returns constant predictions."""

    def fit(self, archive, ctx=None):
        pass

    def predict(self, candidates_x, archive, ctx=None, *, refit=True):
        n = len(candidates_x)
        return SurrogatePrediction.objective(value=np.ones((n, 1)))


class _MockProvider:
    seed: int | None = None
    strategy = IndividualBasedStrategy(evaluation_ratio=0.1)
    termination: Termination = Termination(max_gen(100_000))
    evaluation_planner: EvaluationPlanner | None = None
    feedback_builder: FeedbackBuilder | None = None

    def __init__(self, algorithm, surrogate_manager, acquisition=None):
        self.algorithm = algorithm
        self.surrogate_manager = surrogate_manager
        self.acquisition = acquisition or _LinspaceAcquisition()
        self.evaluator = SerialEvaluator()
        self.cbmanager = CallbackManager()

    def dispatch(self, event):
        pass


# ---------------------------------------------------------------------------
# GenerationBasedStrategy
# ---------------------------------------------------------------------------


class TestGenerationBasedStrategy:
    def _setup(self, gen_ctrl: int = 3):
        ctx = _make_ctx()
        provider = _MockProvider(_make_ga(), _MockSurrogateManager())
        strategy = GenerationBasedStrategy(gen_ctrl=gen_ctrl)
        return ctx, provider, strategy

    def test_gen_ctrl_stored(self):
        strategy = GenerationBasedStrategy(gen_ctrl=5)
        assert strategy.gen_ctrl == 5

    def test_generation_count_per_step(self):
        gen_ctrl = 3
        ctx, provider, strategy = self._setup(gen_ctrl=gen_ctrl)
        ctx = strategy.step(ctx, provider)
        assert ctx.gen == gen_ctrl + 1

    def test_fe_count_is_offspring_count(self):
        ctx, provider, strategy = self._setup(gen_ctrl=2)
        ctx = strategy.step(ctx, provider)
        # Only the final real-evaluation generation counts fe
        n_offspring = len(ctx.population)
        assert ctx.fe == n_offspring

    def test_archive_grows_by_offspring_count(self):
        ctx, provider, strategy = self._setup(gen_ctrl=2)
        before = len(ctx.archive)
        ctx = strategy.step(ctx, provider)
        n_offspring = len(ctx.population)
        assert len(ctx.archive) == before + n_offspring

    def test_surrogate_only_gens_do_not_increment_fe(self):
        gen_ctrl = 4
        ctx, provider, strategy = self._setup(gen_ctrl=gen_ctrl)
        ctx = strategy.step(ctx, provider)
        # fe should only reflect the single real-evaluation generation
        assert ctx.fe < len(ctx.population) * (gen_ctrl + 1)

    def test_gen_ctrl_zero_runs_one_real_generation(self):
        ctx, provider, strategy = self._setup(gen_ctrl=0)
        ctx = strategy.step(ctx, provider)
        assert ctx.gen == 1
        assert ctx.fe == len(ctx.population)

    def test_gen_ctrl_zero_skips_surrogate_fit(self):
        fit_count = [0]
        ctx = _make_ctx()
        surrogate = RBFSurrogate(GaussianKernel()).with_post_fit(
            lambda tx, ty, c: operator.setitem(fit_count, 0, fit_count[0] + 1)
        )
        provider = _MockProvider(
            _make_ga(),
            GlobalSurrogateManager(surrogate),
            MeanPrediction(),
        )
        GenerationBasedStrategy(gen_ctrl=0).step(ctx, cast(Any, provider))
        assert fit_count[0] == 0

    def test_pipeline_rebuilt_on_every_step(self):
        ctx, provider, strategy = self._setup(gen_ctrl=3)
        ctx = strategy.step(ctx, provider)
        pipeline_ref = strategy.pipeline
        strategy.step(ctx, cast(Any, provider))
        assert strategy.pipeline is not pipeline_ref

    def test_surrogate_fit_called_once_per_step(self):
        """fit() must be called exactly once per step(), not once per surrogate gen."""
        fit_count = [0]

        ctx = _make_ctx()
        surrogate = RBFSurrogate(GaussianKernel()).with_post_fit(
            lambda tx, ty, c: operator.setitem(fit_count, 0, fit_count[0] + 1)
        )
        manager = GlobalSurrogateManager(surrogate)
        provider = _MockProvider(_make_ga(), manager, MeanPrediction())
        strategy = GenerationBasedStrategy(gen_ctrl=3)

        strategy.step(ctx, cast(Any, provider))

        assert fit_count[0] == 1

    def test_local_surrogate_refit_false_ignored(self):
        """LocalSurrogateManager always fits per candidate; refit=False is a no-op."""
        fit_count = [0]

        ctx = _make_ctx()
        surrogate = RBFSurrogate(GaussianKernel()).with_post_fit(
            lambda tx, ty, c: operator.setitem(fit_count, 0, fit_count[0] + 1)
        )
        manager = LocalSurrogateManager(surrogate)
        provider = _MockProvider(_make_ga(), manager, MeanPrediction())
        strategy = GenerationBasedStrategy(gen_ctrl=1)

        strategy.step(ctx, cast(Any, provider))

        # LocalSurrogateManager always fits per candidate; refit=False is ignored
        n_offspring = len(ctx.population)
        assert fit_count[0] == n_offspring


# ---------------------------------------------------------------------------
# PreSelectionStrategy
# ---------------------------------------------------------------------------


class TestPreSelectionStrategy:
    def _setup(self, n_candidates: int = 20, n_select: int = 5):
        ctx = _make_ctx()
        provider = _MockProvider(_make_ga(), _MockSurrogateManager())
        strategy = PreSelectionStrategy(n_candidates=n_candidates, n_select=n_select)
        return ctx, provider, strategy

    def test_parameters_stored(self):
        strategy = PreSelectionStrategy(n_candidates=20, n_select=5)
        assert strategy.n_candidates == 20
        assert strategy.n_select == 5

    def test_tell_receives_selected_candidates_only(self):
        ctx, provider, strategy = self._setup(n_candidates=20, n_select=5)
        told = []
        tell = provider.algorithm.tell

        def recording_tell(feedback, state):
            told.append(state.context.offspring.x.copy())
            return tell(feedback, state)

        provider.algorithm.tell = recording_tell
        ctx = strategy.step(ctx, provider)
        assert len(told) == 1
        assert len(told[0]) == 5
        np.testing.assert_array_equal(told[0], ctx.evaluated_offspring.x)
        np.testing.assert_array_equal(
            ctx.evaluation_new_ids, ctx.feedback_result.candidate_ids
        )

    def test_fe_equals_n_select(self):
        ctx, provider, strategy = self._setup(n_candidates=20, n_select=5)
        ctx = strategy.step(ctx, provider)
        assert ctx.fe == 5

    def test_archive_grows_by_n_select(self):
        ctx, provider, strategy = self._setup(n_candidates=20, n_select=5)
        # prob_var=1.0: with the shared _make_ga()'s prob_var=0.1, the batched
        # mutation path changes RNG consumption enough for this
        # seed to leave one candidate's x exactly matching an archived
        # point), the archive's exact-duplicate check (atol=rtol=0)
        # sometimes reuses an existing index instead of growing, making
        # growth < n_select. Mutating every dimension via a continuous
        # Uniform(lb, ub) draw makes a coincidental exact match effectively
        # impossible, keeping this count deterministic.
        provider.algorithm.mutation = MutationUniform(prob=1.0, prob_var=1.0)
        before = len(ctx.archive)
        ctx = strategy.step(ctx, provider)
        assert len(ctx.archive) == before + 5

    def test_generation_count_incremented_once(self):
        ctx, provider, strategy = self._setup()
        ctx = strategy.step(ctx, provider)
        assert ctx.gen == 1

    def test_n_select_capped_by_n_candidates(self):
        # When n_select > n_candidates, evaluate all n_candidates
        n_candidates = 6
        ctx, provider, strategy = self._setup(
            n_candidates=n_candidates, n_select=n_candidates + 10
        )
        ctx = strategy.step(ctx, provider)
        assert ctx.fe == n_candidates

    def test_fe_not_equal_n_candidates(self):
        # Surrogate screening saves real evaluations: fe << n_candidates
        ctx, provider, strategy = self._setup(n_candidates=30, n_select=3)
        ctx = strategy.step(ctx, provider)
        assert ctx.fe == 3
        assert ctx.fe < 30


# ---------------------------------------------------------------------------
# IndividualBasedStrategy/PreSelectionStrategy + archive-based acquisitions
# ---------------------------------------------------------------------------


def _make_novelty_manager() -> tuple[GlobalSurrogateManager, AcquisitionFunction]:
    return (
        GlobalSurrogateManager(RBFSurrogate(GaussianKernel())),
        NoveltyAcquisition(k=3),
    )


def _make_density_manager() -> tuple[GlobalSurrogateManager, AcquisitionFunction]:
    return (
        GlobalSurrogateManager(RBFSurrogate(GaussianKernel())),
        InverseDensityAcquisition(eps=1.0),
    )


class TestIndividualBasedStrategyWithNoveltyAcquisition:
    """End-to-end: IndividualBasedStrategy scored by NoveltyAcquisition."""

    def _setup(self, evaluation_ratio: float = 0.5):
        ctx = _make_ctx()
        manager, acquisition = _make_novelty_manager()
        provider = _MockProvider(_make_ga(), manager, acquisition)
        strategy = IndividualBasedStrategy(evaluation_ratio=evaluation_ratio)
        return ctx, provider, strategy

    def test_step_runs_without_error(self):
        ctx, provider, strategy = self._setup()
        strategy.step(ctx, provider)

    def test_acquisition_scores_are_novelty_based(self):
        """Scores actually come from NoveltyAcquisition, not a constant fallback.

        Captures AcquisitionEndEvent.result.scores via a callback: k-NN
        distances over a random archive/offspring pair are (near-)certainly
        non-constant and always finite, unlike e.g. a manager that always
        returns the same score regardless of acquisition content.
        """
        ctx, provider, strategy = self._setup()
        captured: list[np.ndarray] = []
        provider.cbmanager.register(
            AcquisitionEndEvent,
            lambda event: captured.append(event.result.scores),
        )
        strategy.step(ctx, provider)
        assert len(captured) == 1
        scores = captured[0]
        assert np.all(np.isfinite(scores))
        assert np.ptp(scores) > 0

    def test_fe_equals_evaluation_ratio_times_offspring(self):
        evaluation_ratio = 0.5
        ctx, provider, strategy = self._setup(evaluation_ratio=evaluation_ratio)
        ctx = strategy.step(ctx, provider)
        n_offspring = len(ctx.population)
        expected_fe = max(1, int(evaluation_ratio * n_offspring))
        assert ctx.fe == expected_fe

    def test_archive_grows_by_n_eval(self):
        evaluation_ratio = 0.5
        ctx, provider, strategy = self._setup(evaluation_ratio=evaluation_ratio)
        # prob_var=1.0: see test_archive_grows_by_n_select's comment above
        # -- same fix, same reasoning (this class also uses the shared
        # _make_ga() via _setup()).
        provider.algorithm.mutation = MutationUniform(prob=1.0, prob_var=1.0)
        before = len(ctx.archive)
        ctx = strategy.step(ctx, provider)
        n_eval = max(1, int(evaluation_ratio * len(ctx.population)))
        assert len(ctx.archive) == before + n_eval

    def test_generation_count_incremented(self):
        ctx, provider, strategy = self._setup()
        ctx = strategy.step(ctx, provider)
        assert ctx.gen == 1


class TestPreSelectionStrategyWithInverseDensityAcquisition:
    """End-to-end: PreSelectionStrategy scored by InverseDensityAcquisition."""

    def _setup(self, n_candidates: int = 20, n_select: int = 5):
        ctx = _make_ctx()
        manager, acquisition = _make_density_manager()
        provider = _MockProvider(_make_ga(), manager, acquisition)
        strategy = PreSelectionStrategy(n_candidates=n_candidates, n_select=n_select)
        return ctx, provider, strategy

    def test_step_runs_without_error(self):
        ctx, provider, strategy = self._setup()
        ctx = strategy.step(ctx, provider)

    def test_acquisition_scores_are_density_based(self):
        """Scores actually come from InverseDensityAcquisition, not a constant fallback.

        Same reasoning as the Novelty test above, mirrored for the
        eps-neighborhood density criterion.
        """
        ctx, provider, strategy = self._setup()
        captured: list[np.ndarray] = []
        provider.cbmanager.register(
            AcquisitionEndEvent,
            lambda event: captured.append(event.result.scores),
        )
        strategy.step(ctx, provider)
        assert len(captured) == 1
        scores = captured[0]
        assert np.all(np.isfinite(scores))
        assert np.ptp(scores) > 0

    def test_fe_equals_n_select(self):
        ctx, provider, strategy = self._setup(n_candidates=20, n_select=5)
        ctx = strategy.step(ctx, provider)
        assert ctx.fe == 5

    def test_archive_grows_after_step(self):
        """Archive grows by at most n_select (duplicates may be rejected)."""
        n_select = 5
        ctx, provider, strategy = self._setup(n_candidates=20, n_select=n_select)
        before = len(ctx.archive)
        ctx = strategy.step(ctx, provider)
        assert before < len(ctx.archive) <= before + n_select

    def test_generation_count_incremented(self):
        ctx, provider, strategy = self._setup()
        ctx = strategy.step(ctx, provider)
        assert ctx.gen == 1


# ---------------------------------------------------------------------------
# Strategy.pipeline caching
# ---------------------------------------------------------------------------


from saealib.pipeline import Pipeline  # noqa: E402


class TestStrategyPipelineAttribute:
    """Strategy.pipeline is None before first step, Pipeline after."""

    def _providers_and_strategies(self):
        ga = _make_ga()
        sm = _MockSurrogateManager()
        provider = _MockProvider(ga, sm)
        return provider, {
            "ps": PreSelectionStrategy(n_candidates=20, n_select=5),
            "ib": IndividualBasedStrategy(evaluation_ratio=0.5),
            "gb": GenerationBasedStrategy(gen_ctrl=2),
        }

    def test_ps_pipeline_none_before_step(self):
        _, strategies = self._providers_and_strategies()
        assert strategies["ps"].pipeline is None

    def test_ib_pipeline_none_before_step(self):
        _, strategies = self._providers_and_strategies()
        assert strategies["ib"].pipeline is None

    def test_gb_pipeline_none_before_step(self):
        _, strategies = self._providers_and_strategies()
        assert strategies["gb"].pipeline is None

    def test_ps_pipeline_is_pipeline_after_step(self):
        provider, strategies = self._providers_and_strategies()
        ctx = _make_ctx()
        strategies["ps"].step(ctx, provider)
        assert isinstance(strategies["ps"].pipeline, Pipeline)

    def test_ib_pipeline_is_pipeline_after_step(self):
        provider, strategies = self._providers_and_strategies()
        ctx = _make_ctx()
        strategies["ib"].step(ctx, provider)
        assert isinstance(strategies["ib"].pipeline, Pipeline)

    def test_gb_pipeline_is_pipeline_after_step(self):
        provider, strategies = self._providers_and_strategies()
        ctx = _make_ctx()
        strategies["gb"].step(ctx, provider)
        assert isinstance(strategies["gb"].pipeline, Pipeline)

    def test_ps_pipeline_rebuilt_across_steps(self):
        provider, strategies = self._providers_and_strategies()
        ctx = _make_ctx()
        s = strategies["ps"]
        s.step(ctx, provider)
        first = s.pipeline
        s.step(ctx, provider)
        assert s.pipeline is not first


# ---------------------------------------------------------------------------
# Regression tests for #196: rebuilding the pipeline every step means
# component/parameter changes made mid-``iterate()`` take effect immediately.
# ---------------------------------------------------------------------------


class _SpyManager(SurrogateManager):
    """Surrogate manager that counts predict() calls."""

    def __init__(self):
        self.call_count = 0

    def predict(self, candidates_x, archive, ctx=None, *, refit=True):
        self.call_count += 1
        n = len(candidates_x)
        return SurrogatePrediction.objective(value=np.ones((n, 1)))


class TestPipelineRebuildOnRuntimeReassignment:
    """Regression tests for #196: pipeline must rebuild on every step()."""

    def _make_optimizer(self, strategy, manager) -> Optimizer:
        return (
            Optimizer(_make_problem())
            .set_initializer(
                LHSInitializer(n_init_archive=N_POP, n_init_population=N_POP, seed=0)
            )
            .set_algorithm(_make_ga())
            .set_strategy(strategy)
            .set_surrogate_manager(manager)
            .set_acquisition(_LinspaceAcquisition())
            .set_termination(Termination(max_gen(100)))
        )

    def test_reassigned_surrogate_manager_used_next_generation(self):
        old_manager = _SpyManager()
        new_manager = _SpyManager()
        opt = self._make_optimizer(
            IndividualBasedStrategy(evaluation_ratio=0.5), old_manager
        )

        gen = opt.iterate()
        next(gen)  # initial ctx, before any strategy.step() call
        next(gen)  # generation 1: uses old_manager
        assert old_manager.call_count == 1
        assert new_manager.call_count == 0

        opt.set_surrogate_manager(new_manager)
        next(gen)  # generation 2: pipeline is rebuilt, must use new_manager

        assert new_manager.call_count == 1
        assert old_manager.call_count == 1

    def test_user_stage_without_contract_override_runs_to_completion(self):
        class NoopCountStage(Stage):
            """Replaces count_generation with only execute() implemented."""

            name = "count_generation"

            def execute(self, state):
                return state.replace(gen=state.gen + 1)

        class UserCountStrategy(GenerationBasedStrategy):
            def build_pipeline(self, provider):
                pipeline = super().build_pipeline(provider)
                true_generation = cast(Any, pipeline.find("true_generation"))
                true_generation.replace(
                    "count_generation", stage_component(NoopCountStage())
                )
                return pipeline

        opt = self._make_optimizer(UserCountStrategy(gen_ctrl=1), _SpyManager())

        gen = opt.iterate()
        next(gen)  # initial ctx
        ctx = next(gen)  # one full generation step with the user stage

        assert ctx.gen == 2
        plan = opt.executable_plan
        assert plan is not None
        diagnostics = plan.diagnostics
        warnings = [
            diagnostic
            for diagnostic in diagnostics
            if diagnostic.code == "cannot_verify_stage_flow"
        ]
        assert len(warnings) == 1
        assert warnings[0].severity.name == "WARNING"
        assert warnings[0].code == "cannot_verify_stage_flow"
        assert warnings[0].path is not None
        assert warnings[0].path.components[-1].endswith("count_generation")
        assert not [
            diagnostic
            for diagnostic in diagnostics
            if diagnostic.severity.name == "ERROR"
        ]
        unknown_stage_ids = {
            node.component_id
            for node in plan.graph.nodes
            if isinstance(getattr(node.component, "stage", None), NoopCountStage)
        }
        assert len(unknown_stage_ids) == 1
        assert not any(
            edge.source.component_id in unknown_stage_ids
            or edge.target.component_id in unknown_stage_ids
            for edge in plan.graph.data_edges
        )
        assert warnings[0].severity.name == "WARNING"
        assert not [
            diagnostic
            for diagnostic in diagnostics
            if diagnostic.severity.name == "ERROR"
        ]

    def test_gen_ctrl_change_alters_surrogate_only_iteration_count(self):
        manager = _SpyManager()
        strategy = GenerationBasedStrategy(gen_ctrl=1)
        opt = self._make_optimizer(strategy, manager)

        gen = opt.iterate()
        next(gen)  # initial ctx
        next(gen)  # generation step with gen_ctrl=1
        assert manager.call_count == 1

        strategy.gen_ctrl = 3
        next(gen)  # generation step with gen_ctrl=3

        assert manager.call_count == 1 + 3

    def test_evaluation_ratio_change_alters_true_eval_count(self):
        manager = _SpyManager()
        strategy = IndividualBasedStrategy(evaluation_ratio=0.2)
        opt = self._make_optimizer(strategy, manager)

        gen = opt.iterate()
        ctx = next(gen)  # initial ctx, before any strategy.step() call
        fe_before = ctx.fe
        ctx = next(gen)  # generation step with evaluation_ratio=0.2
        n_offspring = len(ctx.population)
        fe_first_step = ctx.fe - fe_before
        assert fe_first_step == max(1, int(0.2 * n_offspring))

        strategy.evaluation_ratio = 0.6
        fe_before = ctx.fe
        ctx = next(gen)  # generation step with evaluation_ratio=0.6

        fe_second_step = ctx.fe - fe_before
        assert fe_second_step == max(1, int(0.6 * n_offspring))
