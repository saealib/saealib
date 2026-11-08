"""Tests for Optimizer.set_preset()/save_preset(): user-defined presets.

A user preset fills components not already configured via set_*(), just like
the bundled presets file, but is supplied by the user (path, dict, or YAML
text) and is stripped of problem-owned parameters (dim, direction) so it can
be reused across problems of different dimensionality.
"""

from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
import yaml

from saealib.acquisition.mean import MeanPrediction
from saealib.algorithms.ga import GA
from saealib.context import OptimizationState
from saealib.defaults.loader import dump_preset, load_preset
from saealib.exceptions import ValidationError
from saealib.operators.crossover import CrossoverSBX
from saealib.operators.selection import SequentialSelection, TruncationSelection
from saealib.optimizer import Optimizer
from saealib.problem import Problem
from saealib.strategies.gb import GenerationBasedStrategy
from saealib.strategies.ib import IndividualBasedStrategy
from saealib.surrogate.manager import LocalSurrogateManager
from saealib.surrogate.rbf import RBFSurrogate
from saealib.surrogate.rbf_kernels import RBFKernel
from saealib.termination import Termination, max_fe


def _make_ctx(fe: int) -> OptimizationState:
    return cast(OptimizationState, SimpleNamespace(fe=fe, gen=0, archive=None))


def _problem(n_obj: int = 1, dim: int = 2) -> Problem:
    return Problem(
        func=lambda x: np.sum(x**2) * np.ones(n_obj),
        dim=dim,
        n_obj=n_obj,
        direction=np.array([-1.0] * n_obj),
        lb=[-5.0] * dim,
        ub=[5.0] * dim,
    )


def _no_dim_or_direction(obj) -> bool:
    """Recursively check that no dict key ``dim``/``direction`` is present."""
    if isinstance(obj, dict):
        if "dim" in obj or "direction" in obj:
            return False
        return all(_no_dim_or_direction(v) for v in obj.values())
    if isinstance(obj, list):
        return all(_no_dim_or_direction(v) for v in obj)
    return True


def _mutation_uniform_ga() -> GA:
    from saealib.operators.mutation import MutationUniform

    return GA(
        crossover=CrossoverSBX(prob=0.9, eta=15.0),
        mutation=MutationUniform(prob_var=0.3),
        parent_selection=SequentialSelection(),
        survivor_selection=TruncationSelection(),
    )


class TestSavePresetStripsProblemParams:
    def test_saved_preset_has_no_dim_or_direction(self, tmp_path):
        opt = Optimizer(_problem(dim=2))
        opt.resolve_defaults()

        path = opt.save_preset(tmp_path / "p.yaml")

        assert path.exists()
        preset = yaml.safe_load(path.read_text("utf-8"))
        assert _no_dim_or_direction(preset)
        assert preset["evaluation_planner"]["type"] == "RatioEvaluation"
        assert preset["evaluation_planner"]["params"]["sanitize_nonfinite"] is True
        assert preset["feedback_builder"]["type"] == "ComparatorWorstFallback"
        assert preset["feedback_builder"]["params"]["inner"]["type"] == "MixedFeedback"


class TestRoundTripAcrossProblems:
    def test_save_and_reload_across_different_dim(self, tmp_path):
        problem2 = _problem(dim=2)
        opt = Optimizer(problem2)
        opt.set_algorithm(_mutation_uniform_ga())
        opt.set_strategy(IndividualBasedStrategy(evaluation_ratio=0.3))
        opt.set_termination(Termination(max_fe(123)))
        opt.resolve_defaults()

        path = opt.save_preset(tmp_path / "p.yaml")

        problem5 = _problem(dim=5)
        opt5 = Optimizer(problem5).set_preset(path)
        opt5.resolve_defaults()

        assert isinstance(opt5.strategy, IndividualBasedStrategy)
        assert opt5.strategy.evaluation_ratio == 0.3

        assert isinstance(opt5.algorithm, GA)
        assert isinstance(opt5.algorithm.crossover, CrossoverSBX)
        assert opt5.algorithm.crossover.prob == 0.9
        assert opt5.algorithm.crossover.eta == 15.0

        manager = opt5.surrogate_manager
        assert isinstance(manager, LocalSurrogateManager)
        assert isinstance(manager.surrogate, RBFSurrogate)
        assert isinstance(opt5.acquisition, MeanPrediction)
        np.testing.assert_array_equal(opt5.acquisition.direction, problem5.direction)

        termination = opt5.termination
        assert termination is not None
        assert termination.is_terminated(_make_ctx(fe=123)) is True
        assert termination.is_terminated(_make_ctx(fe=122)) is False


class TestPresetPrecedence:
    def test_set_strategy_wins_over_preset_strategy(self):
        opt = Optimizer(_problem())
        gb = GenerationBasedStrategy(gen_ctrl=3)
        opt.set_strategy(gb)
        opt.set_preset(
            {
                "strategy": {
                    "type": "IndividualBasedStrategy",
                    "params": {"evaluation_ratio": 0.5},
                }
            }
        )
        opt.resolve_defaults()

        assert opt.strategy is gb

    def test_preset_wins_over_bundled_default(self):
        opt = Optimizer(_problem())
        opt.set_preset(
            {"strategy": {"type": "GenerationBasedStrategy", "params": {"gen_ctrl": 9}}}
        )
        opt.resolve_defaults()

        assert isinstance(opt.strategy, GenerationBasedStrategy)
        assert opt.strategy.gen_ctrl == 9


class TestSavePresetErrors:
    def test_unserializable_termination_raises_with_component_name(self):
        opt = Optimizer(_problem())
        opt.set_termination(Termination(lambda ctx: True))

        with pytest.raises(ValidationError, match="termination"):
            opt.save_preset("unused.yaml")

    def test_no_components_configured_raises(self, tmp_path):
        opt = Optimizer(_problem())

        with pytest.raises(ValidationError):
            opt.save_preset(tmp_path / "p.yaml")


class TestLoadPresetValidation:
    def test_unknown_key_raises(self):
        with pytest.raises(ValidationError):
            Optimizer(_problem()).set_preset({"algorithim": {}})

    def test_wrong_schema_version_raises(self):
        with pytest.raises(ValidationError):
            Optimizer(_problem()).set_preset({"schema_version": 2})

    def test_dict_preset_is_accepted(self):
        opt = Optimizer(_problem()).set_preset(
            {"strategy": {"type": "IndividualBasedStrategy", "params": {}}}
        )
        assert opt._preset is not None


class TestPresetSmokeRun:
    def test_run_completes_with_preset_strategy(self):
        opt = Optimizer(_problem(dim=2))
        opt.set_preset(
            {
                "strategy": {
                    "type": "IndividualBasedStrategy",
                    "params": {"evaluation_ratio": 0.2},
                }
            }
        )
        opt.set_termination(Termination(max_fe(50)))
        ctx = opt.run()

        assert ctx.fe > 0


class TestPresetEndToEndRun:
    """Regression for the string-kernel bug: saved presets must run, not just
    round-trip type/dim, since ``kernel`` is an ``RBFKernel`` object
    serialized via ``to_spec()``/``build()`` alongside the rest of the
    surrogate manager."""

    def test_save_reload_across_dims_and_run(self, tmp_path):
        problem2 = _problem(dim=2)
        opt = Optimizer(problem2)
        opt.set_termination(Termination(max_fe(60)))
        opt.resolve_defaults()

        path = opt.save_preset(tmp_path / "p.yaml")

        problem5 = _problem(dim=5)
        opt5 = Optimizer(problem5, seed=1).set_preset(path)
        ctx = opt5.run()

        assert ctx.fe >= 60
        manager = opt5.surrogate_manager
        assert isinstance(manager, LocalSurrogateManager)
        assert isinstance(manager.surrogate, RBFSurrogate)
        assert manager.surrogate.n_features_in_ == 5
        assert isinstance(manager.surrogate.kernel, RBFKernel)


def test_legacy_feedback_key_is_rejected() -> None:
    with pytest.raises(ValidationError, match="Unknown preset key"):
        load_preset({"feedback_policy": {"type": "NoFeedback"}})


def test_dump_preset_writes_only_canonical_keys(tmp_path) -> None:
    path = dump_preset(
        {
            "evaluation_planner": {"type": "EvaluateAll"},
            "feedback_builder": {"type": "NoFeedback"},
        },
        tmp_path / "canonical",
    )
    loaded = yaml.safe_load(path.read_text())
    assert "evaluation_planner" in loaded
    assert "feedback_builder" in loaded
    assert "evaluation_policy" not in loaded
    assert "feedback_policy" not in loaded
