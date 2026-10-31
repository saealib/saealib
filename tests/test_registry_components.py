"""Integration tests: built-in components self-register and Registry.build
reproduces objects equivalent to the hardcoded resolvers in saealib.api."""

from typing import Any, cast

import numpy as np
import pytest

from saealib.acquisition import MeanPrediction
from saealib.algorithms.ga import GA
from saealib.algorithms.pso import PSO
from saealib.operators.crossover import CrossoverBLXAlpha
from saealib.operators.mutation import MutationUniform
from saealib.operators.selection import SequentialSelection, TruncationSelection
from saealib.registry import build, get
from saealib.strategies.gb import GenerationBasedStrategy
from saealib.strategies.ib import IndividualBasedStrategy
from saealib.strategies.ps import PreSelectionStrategy
from saealib.surrogate.manager import LocalSurrogateManager
from saealib.surrogate.rbf import RBFSurrogate
from saealib.surrogate.rbf_kernels import GaussianKernel
from saealib.surrogate.training_set import KNNObjectiveSet


class TestBuiltinComponentsRegistered:
    @pytest.mark.parametrize(
        ("name", "cls"),
        [
            ("GA", GA),
            ("PSO", PSO),
            ("RBFSurrogate", RBFSurrogate),
            ("LocalSurrogateManager", LocalSurrogateManager),
            ("KNNObjectiveSet", KNNObjectiveSet),
            ("IndividualBasedStrategy", IndividualBasedStrategy),
            ("GenerationBasedStrategy", GenerationBasedStrategy),
            ("PreSelectionStrategy", PreSelectionStrategy),
            ("CrossoverBLXAlpha", CrossoverBLXAlpha),
            ("MutationUniform", MutationUniform),
            ("SequentialSelection", SequentialSelection),
            ("TruncationSelection", TruncationSelection),
        ],
    )
    def test_registered_under_class_name(self, name, cls):
        assert get(name) is cls


class TestBuildMatchesApiResolvers:
    """Each spec below mirrors a hardcoded resolver in saealib/api.py."""

    def test_ga_default_spec_matches_resolve_algorithm(self):
        spec = {
            "type": "GA",
            "params": {
                "crossover": {
                    "type": "CrossoverBLXAlpha",
                    "params": {"prob": 0.7, "alpha": 0.4},
                },
                "mutation": {
                    "type": "MutationUniform",
                    "params": {"prob_var": 0.3},
                },
                "parent_selection": {"type": "SequentialSelection"},
                "survivor_selection": {"type": "TruncationSelection"},
            },
        }
        ga = build(spec)

        assert isinstance(ga, GA)
        assert isinstance(ga.crossover, CrossoverBLXAlpha)
        assert ga.crossover.prob == 0.7
        assert ga.crossover.alpha == 0.4
        assert isinstance(ga.mutation, MutationUniform)
        assert ga.mutation.prob_var == 0.3
        assert isinstance(ga.parent_selection, SequentialSelection)
        assert isinstance(ga.survivor_selection, TruncationSelection)

    def test_rbf_surrogate_manager_spec_matches_resolve_surrogate(self):
        direction = np.array([-1.0])
        spec = {
            "type": "LocalSurrogateManager",
            "params": {
                "surrogate": {
                    "type": "RBFSurrogate",
                    "params": {"kernel": GaussianKernel()},
                },
                "acquisition": {
                    "type": "MeanPrediction",
                    "params": {"direction": direction},
                },
                "training_set": {
                    "type": "KNNObjectiveSet",
                    "params": {"n_neighbors": 50},
                },
            },
        }
        manager_spec = {
            **spec,
            "params": {
                key: value
                for key, value in spec["params"].items()
                if key != "acquisition"
            },
        }
        manager = build(manager_spec)

        assert isinstance(manager, LocalSurrogateManager)
        assert isinstance(manager.surrogate, RBFSurrogate)
        acquisition = MeanPrediction(direction=direction)
        assert acquisition.direction is direction
        assert isinstance(manager.training_set, KNNObjectiveSet)
        assert manager.training_set.n_neighbors == 50

    @pytest.mark.parametrize(
        ("spec", "cls", "attrs"),
        [
            (
                {
                    "type": "IndividualBasedStrategy",
                    "params": {"evaluation_ratio": 0.1},
                },
                IndividualBasedStrategy,
                {"evaluation_ratio": 0.1},
            ),
            (
                {"type": "GenerationBasedStrategy", "params": {"gen_ctrl": 5}},
                GenerationBasedStrategy,
                {"gen_ctrl": 5},
            ),
            (
                {
                    "type": "PreSelectionStrategy",
                    "params": {"n_candidates": 40, "n_select": 4},
                },
                PreSelectionStrategy,
                {"n_candidates": 40, "n_select": 4},
            ),
        ],
    )
    def test_strategy_spec_matches_resolve_strategy(self, spec, cls, attrs):
        strategy = build(spec)
        assert isinstance(strategy, cls)
        for attr, expected in attrs.items():
            assert getattr(strategy, attr) == expected


def _normalize(value: Any) -> Any:
    """Normalize tuples/arrays to lists and sklearn objects to param dicts."""
    if isinstance(value, tuple | list | np.ndarray):
        return [_normalize(v) for v in value]
    if isinstance(value, dict):
        return {k: _normalize(v) for k, v in value.items()}
    if hasattr(value, "get_params"):
        return {k: _normalize(v) for k, v in value.get_params().items()}
    return value


class TestSklearnSurrogateRoundTrip:
    def test_gpr_round_trip_preserves_estimator_params(self):
        pytest.importorskip("sklearn")
        from sklearn.gaussian_process.kernels import Matern

        from saealib.registry import to_spec
        from saealib.surrogate.sklearn_surrogate import SklearnGPRSurrogate

        original = SklearnGPRSurrogate(kernel=Matern(nu=2.5), normalize_y=True)
        rebuilt = build(to_spec(original))

        assert isinstance(rebuilt, SklearnGPRSurrogate)
        original_estimator = cast(Any, original.estimator)
        rebuilt_estimator = rebuilt.estimator
        original_params = original_estimator.get_params()
        rebuilt_params = rebuilt_estimator.get_params()
        assert set(rebuilt_params) == set(original_params)
        assert all(
            _normalize(rebuilt_params[key]) == _normalize(original_params[key])
            for key in original_params
        )
        assert isinstance(rebuilt_estimator.kernel, Matern)
        assert rebuilt_estimator.kernel.nu == 2.5
        assert rebuilt_estimator.normalize_y is True


class TestNSGA3ComparatorRoundTrip:
    def test_nsga3_round_trip_keeps_seed_and_dominator(self):
        from saealib.comparators.comparators import NSGA3Comparator
        from saealib.comparators.dominance import ParetoDominator
        from saealib.registry import to_spec

        reference_points = np.eye(3)
        original = NSGA3Comparator(reference_points, seed=3)
        rebuilt = build(to_spec(original))

        assert isinstance(rebuilt, NSGA3Comparator)
        np.testing.assert_array_equal(rebuilt.reference_points, reference_points)
        assert rebuilt.seed == 3
        assert isinstance(rebuilt.dominator, ParetoDominator)
        assert rebuilt.eps_cv == original.eps_cv
        assert rebuilt.eps_obj == original.eps_obj
