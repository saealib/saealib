"""Tests for Optimizer._resolve_defaults(): the 4-tier default resolution.

Precedence: components already set via set_*() > preset selected by the
algorithm's registered name > preset selected by Problem shape > universal
fallback preset. initializer/termination are computed from problem.dim
directly (not part of the presets file).
"""

import numpy as np
import pytest

from saealib.algorithms.pso import PSO
from saealib.execution import AsyncEvaluationScheduler, SerialEvaluator
from saealib.optimizer import Optimizer
from saealib.policies import (
    ComparatorWorstFallback,
    EvaluateAll,
    NoFeedback,
    RatioEvaluation,
)
from saealib.problem import Problem
from saealib.strategies.direct import DirectStrategy
from saealib.strategies.ps import PreSelectionStrategy


def _problem(n_obj: int = 1, dim: int = 2) -> Problem:
    return Problem(
        func=lambda x: np.sum(x) * np.ones(n_obj),
        dim=dim,
        n_obj=n_obj,
        direction=np.array([-1.0] * n_obj),
        lb=[-5.0] * dim,
        ub=[5.0] * dim,
    )


class TestSetComponentsAreNeverOverwritten:
    def test_set_algorithm_is_preserved(self):
        opt = Optimizer(_problem())
        pso = PSO()
        opt.set_algorithm(pso)
        opt._resolve_defaults()
        assert opt.algorithm is pso

    def test_set_algorithm_after_async_scheduler_waits_for_publication(self):
        from saealib.core.runtime import SequentialPlan
        from saealib.execution.runtime import _OptimizerEnvironment

        opt = Optimizer(_problem())
        scheduler = AsyncEvaluationScheduler(SerialEvaluator())

        opt.set_async_evaluation_scheduler(scheduler)
        algorithm = PSO()
        opt.set_algorithm(algorithm)
        assert scheduler.algorithm is None

        opt.set_strategy(DirectStrategy())
        plan = opt._compile_plan()
        assert plan is not None
        _OptimizerEnvironment(opt, SequentialPlan.from_executable_plan(plan))

        assert scheduler.algorithm is algorithm

    def test_set_strategy_is_preserved(self):
        from saealib.strategies.gb import GenerationBasedStrategy

        opt = Optimizer(_problem())
        gb = GenerationBasedStrategy(gen_ctrl=3)
        opt.set_strategy(gb)
        opt._resolve_defaults()
        assert opt.strategy is gb

    def test_set_surrogate_manager_is_preserved(self):
        from saealib.surrogate.manager import GlobalSurrogateManager
        from saealib.surrogate.rbf import RBFSurrogate
        from saealib.surrogate.rbf_kernels import GaussianKernel

        opt = Optimizer(_problem())
        manager = GlobalSurrogateManager(RBFSurrogate(GaussianKernel()))
        opt.set_surrogate_manager(manager)
        opt._resolve_defaults()
        assert opt.surrogate_manager is manager


class TestPolicyResolution:
    def test_defaults_include_explicit_fallback(self):
        opt = Optimizer(_problem())
        opt._resolve_defaults()
        assert isinstance(opt.evaluation_planner, RatioEvaluation)
        assert opt.evaluation_planner.sanitize_nonfinite is True
        assert isinstance(opt.feedback_builder, ComparatorWorstFallback)

    def test_explicit_strategy_does_not_receive_bundled_policies(self):
        opt = Optimizer(_problem()).set_strategy(
            PreSelectionStrategy(n_candidates=8, n_select=2)
        )
        opt._resolve_defaults()
        assert opt.evaluation_planner is None
        assert opt.feedback_builder is None

    def test_explicit_components_win_over_bundled_defaults(self):
        evaluation = EvaluateAll()
        feedback = NoFeedback()
        opt = (
            Optimizer(_problem())
            .set_strategy(PreSelectionStrategy(n_candidates=8, n_select=2))
            .set_evaluation_planner(evaluation)
            .set_feedback_builder(feedback)
        )
        opt._resolve_defaults()
        assert opt.evaluation_planner is evaluation
        assert opt.feedback_builder is feedback


class TestPresetPrecedence:
    """Uses a synthetic 3-tier defaults file so each precedence level yields
    a distinguishable result, independent of the real presets.yaml content."""

    @pytest.fixture
    def fake_defaults(self, monkeypatch):
        import saealib.defaults as defaults_module

        fake = {
            "presets": {
                "alg_preset": {
                    "strategy": {
                        "type": "GenerationBasedStrategy",
                        "params": {"gen_ctrl": 7},
                    }
                },
                "shape_preset": {
                    "strategy": {
                        "type": "IndividualBasedStrategy",
                        "params": {"evaluation_ratio": 0.2},
                    }
                },
                "fallback_preset": {
                    "strategy": {
                        "type": "IndividualBasedStrategy",
                        "params": {"evaluation_ratio": 0.99},
                    }
                },
            },
            "by_algorithm": {"PSO": "alg_preset"},
            "by_problem_shape": [{"when": {"n_obj": 1}, "preset": "shape_preset"}],
            "fallback": "fallback_preset",
        }
        monkeypatch.setattr(defaults_module, "load_defaults", lambda: fake)
        return fake

    def test_algorithm_set_selects_preset_by_algorithm_name(self, fake_defaults):
        from saealib.strategies.gb import GenerationBasedStrategy

        opt = Optimizer(_problem(n_obj=1))
        opt.set_algorithm(PSO())
        opt._resolve_defaults()

        assert isinstance(opt.strategy, GenerationBasedStrategy)
        assert opt.strategy.gen_ctrl == 7

    def test_no_algorithm_falls_back_to_problem_shape_rule(self, fake_defaults):
        from saealib.strategies.ib import IndividualBasedStrategy

        opt = Optimizer(_problem(n_obj=1))
        opt._resolve_defaults()

        assert isinstance(opt.strategy, IndividualBasedStrategy)
        assert opt.strategy.evaluation_ratio == 0.2

    def test_no_match_uses_universal_fallback(self, fake_defaults):
        from saealib.strategies.ib import IndividualBasedStrategy

        opt = Optimizer(_problem(n_obj=3))  # no by_algorithm/by_problem_shape match
        opt._resolve_defaults()

        assert isinstance(opt.strategy, IndividualBasedStrategy)
        assert opt.strategy.evaluation_ratio == 0.99


class TestSurrogateManagerInjection:
    """surrogate/acquisition are problem-dependent and injected at resolve
    time; they are intentionally absent from presets.yaml (Unit A)."""

    def test_injects_problem_dim_and_direction(self):
        from saealib.acquisition.mean import MeanPrediction
        from saealib.surrogate.manager import LocalSurrogateManager
        from saealib.surrogate.rbf import RBFSurrogate

        problem = _problem(n_obj=1, dim=3)
        opt = Optimizer(problem)
        opt._resolve_defaults()

        manager = opt.surrogate_manager
        assert isinstance(manager, LocalSurrogateManager)
        assert isinstance(manager.surrogate, RBFSurrogate)
        assert isinstance(opt.acquisition, MeanPrediction)
        np.testing.assert_array_equal(opt.acquisition.direction, problem.direction)

    def test_training_set_uses_preset_n_neighbors(self):
        from saealib.surrogate.manager import LocalSurrogateManager
        from saealib.surrogate.training_set import KNNObjectiveSet

        opt = Optimizer(_problem())
        opt._resolve_defaults()

        manager = opt.surrogate_manager
        assert isinstance(manager, LocalSurrogateManager)
        assert isinstance(manager.training_set, KNNObjectiveSet)
        assert manager.training_set.n_neighbors == 50


class TestInitializerAndTerminationDefaults:
    def test_initializer_scales_with_problem_dim(self):
        from saealib.execution.initializer import LHSInitializer

        problem = _problem(dim=4)
        opt = Optimizer(problem)
        opt._resolve_defaults()

        assert isinstance(opt.initializer, LHSInitializer)
        assert opt.initializer.n_init_archive == 5 * 4
        assert opt.initializer.n_init_population == 4 * 4

    def test_initializer_uses_optimizer_seed(self):
        from saealib.execution.initializer import LHSInitializer

        opt = Optimizer(_problem(), seed=42)
        opt._resolve_defaults()
        initializer = opt.initializer
        assert isinstance(initializer, LHSInitializer)
        assert initializer.seed == 42

    def test_termination_is_set(self):
        opt = Optimizer(_problem())
        opt._resolve_defaults()
        assert opt.termination is not None

    def test_resolve_defaults_is_idempotent(self):
        opt = Optimizer(_problem())
        opt._resolve_defaults()
        algorithm, strategy, initializer = (
            opt.algorithm,
            opt.strategy,
            opt.initializer,
        )
        opt._resolve_defaults()
        assert opt.algorithm is algorithm
        assert opt.strategy is strategy
        assert opt.initializer is initializer
