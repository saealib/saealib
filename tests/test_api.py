"""Unit tests for saealib.api: minimize/maximize preset support and _UnsetType."""

import copy
from typing import Any

import numpy as np
import pytest

import saealib.api as api
from saealib import maximize, minimize
from saealib.acquisition.winrate import WinRateAcquisition
from saealib.algorithms.pso import PSO
from saealib.api import _UNSET, _UnsetType
from saealib.exceptions import ValidationError
from saealib.problem import Problem
from saealib.strategies.gb import GenerationBasedStrategy
from saealib.strategies.ps import PreSelectionStrategy
from saealib.surrogate.manager import (
    LocalSurrogateManager,
    PairwiseSurrogateManager,
)
from saealib.surrogate.rbf import RBFSurrogate
from saealib.surrogate.rbf_kernels import GaussianKernel
from saealib.surrogate.sklearn_surrogate import SklearnRFCClassificationSurrogate


class _ResolvedConfigurationError(Exception):
    pass


def _component_descriptor(component):
    if component is None:
        return None
    descriptor = {"type": type(component).__name__}
    for name in (
        "prob",
        "alpha",
        "prob_var",
        "w",
        "c1",
        "c2",
        "v_max",
        "randomize_ties",
        "n_neighbors",
        "direction",
        "evaluation_ratio",
        "gen_ctrl",
        "n_candidates",
        "n_select",
        "ratio",
        "sanitize_nonfinite",
    ):
        if hasattr(component, name):
            value = getattr(component, name)
            descriptor[name] = tuple(value) if isinstance(value, np.ndarray) else value
    if hasattr(component, "inner"):
        descriptor["inner"] = _component_descriptor(component.inner)
    if hasattr(component, "kernel"):
        descriptor["kernel"] = _component_descriptor(component.kernel)
    if hasattr(component, "conditions"):
        descriptor["conditions"] = tuple(
            condition._registry_spec for condition in component.conditions
        )
    return descriptor


def _resolution_descriptor(optimizer):
    algorithm = _component_descriptor(optimizer.algorithm)
    for name in (
        "crossover",
        "mutation",
        "parent_selection",
        "survivor_selection",
    ):
        algorithm[name] = _component_descriptor(
            getattr(optimizer.algorithm, name, None)
        )
    manager = _component_descriptor(optimizer.surrogate_manager)
    if manager is not None:
        manager["training_set"] = _component_descriptor(
            optimizer.surrogate_manager.training_set
        )
        manager["surrogate"] = _component_descriptor(
            optimizer.surrogate_manager.surrogate
        )
    return {
        "algorithm": algorithm,
        "surrogate_manager": manager,
        "acquisition": _component_descriptor(optimizer.acquisition),
        "strategy": _component_descriptor(optimizer.strategy),
        "evaluation_planner": _component_descriptor(optimizer.evaluation_planner),
        "feedback_builder": _component_descriptor(optimizer.feedback_builder),
        "initializer": {
            "n_init_archive": optimizer.initializer.n_init_archive,
            "n_init_population": optimizer.initializer.n_init_population,
            "seed": optimizer.initializer.seed,
        },
        "termination": {"type": type(optimizer.termination).__name__},
    }


class TestUnsetType:
    def test_repr_is_unset(self):
        assert repr(_UNSET) == "UNSET"

    def test_instance_of_dedicated_type(self):
        assert isinstance(_UNSET, _UnsetType)


def test_problem_rejects_direction_argument_with_actionable_message():
    problem = Problem(
        func=lambda x: float(np.sum(x**2)),
        dim=1,
        n_obj=1,
        direction=np.array([-1.0]),
        lb=[-1.0],
        ub=[1.0],
    )

    with pytest.raises(ValidationError, match="direction cannot be passed"):
        api._ensure_problem(problem, None, None, None, 1, ["maximize"], -1.0)


class TestResolvedConfiguration:
    @pytest.mark.parametrize(
        "case_id",
        [
            pytest.param("defaults", id="defaults"),
            pytest.param("ga_rbf_ib", id="ga_rbf_ib"),
            pytest.param("pso_surrogate_instance", id="pso_surrogate_instance"),
            pytest.param(
                "pso_instance_manager_instance", id="pso_instance_manager_instance"
            ),
            pytest.param("pairwise_manager", id="pairwise_manager"),
        ],
    )
    def test_api_resolution_snapshot(self, monkeypatch, case_id):
        def stub_run(self, *args, **kwargs):
            self.resolve_defaults()
            raise _ResolvedConfigurationError(_resolution_descriptor(self))

        monkeypatch.setattr(api.Optimizer, "run", stub_run)

        ga = {
            "type": "GA",
            "crossover": {"type": "CrossoverBLXAlpha", "prob": 0.7, "alpha": 0.4},
            "mutation": {"type": "MutationUniform", "prob": 1.0, "prob_var": 0.3},
            "parent_selection": {"type": "SequentialSelection"},
            "survivor_selection": {
                "type": "TruncationSelection",
                "randomize_ties": False,
            },
        }
        pso = {
            "type": "PSO",
            "w": 0.7,
            "c1": 1.5,
            "c2": 1.5,
            "v_max": None,
            "crossover": None,
            "mutation": None,
            "parent_selection": None,
            "survivor_selection": None,
        }
        local_rbf = {
            "type": "LocalSurrogateManager",
            "training_set": {"type": "KNNObjectiveSet", "n_neighbors": 50},
            "surrogate": {
                "type": "RBFSurrogate",
                "alpha": 1e-8,
                "kernel": {"type": "GaussianKernel"},
            },
        }
        pairwise = {
            "type": "PairwiseSurrogateManager",
            "training_set": {"type": "PairwiseComparisonSet"},
            "surrogate": {"type": "SklearnRFCClassificationSurrogate"},
        }
        common = {
            "initializer": {
                "n_init_archive": 10,
                "n_init_population": 6,
                "seed": 7,
            },
            "termination": {"type": "Termination"},
        }
        cases: dict[str, tuple[Any, dict[str, Any], dict[str, Any]]] = {
            "defaults": (
                minimize,
                {},
                {
                    "algorithm": ga,
                    "surrogate_manager": local_rbf,
                    "acquisition": {"type": "MeanPrediction", "direction": (-1.0,)},
                    "strategy": {
                        "type": "IndividualBasedStrategy",
                        "evaluation_ratio": 0.1,
                    },
                    "evaluation_planner": {
                        "type": "RatioEvaluation",
                        "ratio": 0.1,
                        "sanitize_nonfinite": True,
                    },
                    "feedback_builder": {
                        "type": "ComparatorWorstFallback",
                        "inner": {"type": "MixedFeedback"},
                    },
                    **common,
                },
            ),
            "ga_rbf_ib": (
                minimize,
                {"algorithm": "GA", "surrogate": "rbf", "strategy": "ib"},
                {
                    "algorithm": ga,
                    "surrogate_manager": local_rbf,
                    "acquisition": {"type": "MeanPrediction", "direction": (-1.0,)},
                    "strategy": {
                        "type": "IndividualBasedStrategy",
                        "evaluation_ratio": 0.1,
                    },
                    "evaluation_planner": {
                        "type": "RatioEvaluation",
                        "ratio": 0.1,
                        "sanitize_nonfinite": True,
                    },
                    "feedback_builder": {"type": "MixedFeedback"},
                    **common,
                },
            ),
            "pso_surrogate_instance": (
                maximize,
                {
                    "algorithm": "PSO",
                    "surrogate": RBFSurrogate(GaussianKernel()),
                    "strategy": "gb",
                },
                {
                    "algorithm": pso,
                    "surrogate_manager": local_rbf,
                    "acquisition": {"type": "MeanPrediction", "direction": (1.0,)},
                    "strategy": {"type": "GenerationBasedStrategy", "gen_ctrl": 5},
                    "evaluation_planner": {"type": "EvaluateAll"},
                    "feedback_builder": {"type": "TrueOnlyFeedback"},
                    **common,
                },
            ),
            "pso_instance_manager_instance": (
                minimize,
                {
                    "algorithm": PSO(),
                    "surrogate": LocalSurrogateManager(RBFSurrogate(GaussianKernel())),
                    "strategy": "ps",
                },
                {
                    "algorithm": pso,
                    "surrogate_manager": local_rbf,
                    "acquisition": {"type": "MeanPrediction", "direction": (-1.0,)},
                    "strategy": {
                        "type": "PreSelectionStrategy",
                        "n_candidates": 6,
                        "n_select": 1,
                    },
                    "evaluation_planner": {
                        "type": "TopKEvaluation",
                        "sanitize_nonfinite": True,
                    },
                    "feedback_builder": {"type": "TrueOnlyFeedback"},
                    **common,
                },
            ),
            "pairwise_manager": (
                minimize,
                {
                    "algorithm": "GA",
                    "surrogate": PairwiseSurrogateManager(
                        SklearnRFCClassificationSurrogate()
                    ),
                    "strategy": "ib",
                },
                {
                    "algorithm": ga,
                    "surrogate_manager": pairwise,
                    "acquisition": {"type": "WinRateAcquisition"},
                    "strategy": {
                        "type": "IndividualBasedStrategy",
                        "evaluation_ratio": 0.1,
                    },
                    "evaluation_planner": {
                        "type": "RatioEvaluation",
                        "ratio": 0.1,
                        "sanitize_nonfinite": True,
                    },
                    "feedback_builder": {"type": "MixedFeedback"},
                    **common,
                },
            ),
        }

        runner, components, expected = cases[case_id]
        with pytest.raises(_ResolvedConfigurationError) as caught:
            runner(
                lambda x: np.sum(x**2),
                dim=2,
                lb=[-1, -1],
                ub=[1, 1],
                pop_size=6,
                max_fe=30,
                seed=7,
                verbose=False,
                **components,
            )
        assert caught.value.args[0] == expected

    def test_api_and_optimizer_bundle_inject_problem_params(self, monkeypatch):
        import saealib.defaults as defaults_module
        import saealib.registry as registry
        from saealib.algorithms.base import Algorithm
        from saealib.population import Archive, Population
        from saealib.registry import register, to_spec
        from saealib.strategies.base import OptimizationStrategy

        previous = {
            name: registry._REGISTRY[name] for name in ("GA", "GenerationBasedStrategy")
        }
        captured: dict[str, Any] = {}
        try:

            @register("GA")
            class _DimensionAwareAlgorithm(Algorithm):
                def __init__(self, dim: int, direction: np.ndarray) -> None:
                    self.dim = dim
                    self.direction = direction

                def get_required_attrs(self, problem):
                    return []

                @property
                def population_class(self):
                    return Population

                @property
                def archive_class(self):
                    return Archive

                def ask(self, request, state):
                    raise NotImplementedError

                def tell(self, feedback, state):
                    raise NotImplementedError

            @register("GenerationBasedStrategy")
            class _DimensionAwareStrategy(OptimizationStrategy):
                def __init__(self, dim: int, direction: np.ndarray) -> None:
                    self.dim = dim
                    self.direction = direction

            bundled_preset = {
                "algorithm": {"type": "GA", "params": {}},
                "strategy": {
                    "type": "GenerationBasedStrategy",
                    "params": {},
                },
            }
            defaults = {
                "presets": {"injection": bundled_preset},
                "by_algorithm": {"GA": "injection"},
                "by_problem_shape": [],
                "fallback": "injection",
            }
            monkeypatch.setattr(defaults_module, "load_defaults", lambda: defaults)

            def stub_run(self, *args, **kwargs):
                captured["optimizer"] = self
                raise _ResolvedConfigurationError

            monkeypatch.setattr(api.Optimizer, "run", stub_run)
            with pytest.raises(_ResolvedConfigurationError):
                minimize(
                    lambda x: np.sum(x),
                    algorithm="GA",
                    dim=3,
                    lb=[-1, -1, -1],
                    ub=[1, 1, 1],
                    strategy="gb",
                    verbose=False,
                )

            api_optimizer = captured["optimizer"]
            problem = api_optimizer.problem
            user_optimizer = api.Optimizer(problem).set_preset(
                copy.deepcopy(bundled_preset)
            )
            user_optimizer.resolve_defaults()
            bundled_optimizer = api.Optimizer(problem)
            bundled_optimizer.resolve_defaults()

            for name in ("algorithm", "strategy"):
                api_spec = to_spec(getattr(api_optimizer, name))
                user_spec = to_spec(getattr(user_optimizer, name))
                bundled_spec = to_spec(getattr(bundled_optimizer, name))
                assert api_spec == user_spec == bundled_spec
                assert api_spec["params"]["dim"] == problem.dim
                np.testing.assert_array_equal(
                    api_spec["params"]["direction"], problem.direction
                )
        finally:
            registry._REGISTRY.update(previous)


class TestMinimizePreset:
    def test_completes_with_preset_dict(self):
        preset = {
            "strategy": {
                "type": "GenerationBasedStrategy",
                "params": {"gen_ctrl": 2},
            },
        }
        result = minimize(
            lambda x: np.sum(x**2),
            dim=2,
            lb=[-1, -1],
            ub=[1, 1],
            algorithm="GA",
            surrogate="rbf",
            preset=preset,
            max_fe=30,
            pop_size=6,
            seed=0,
            verbose=False,
        )
        assert isinstance(result.x, np.ndarray)
        assert result.fe > 0

    def test_preset_strategy_is_actually_used(self, monkeypatch):
        calls = {"gb": 0, "ps": 0}
        orig_gb_graph = GenerationBasedStrategy.build_graph
        orig_ps_graph = PreSelectionStrategy.build_graph

        def gb_graph(self, provider):
            calls["gb"] += 1
            return orig_gb_graph(self, provider)

        def ps_graph(self, provider):
            calls["ps"] += 1
            return orig_ps_graph(self, provider)

        # Observe the selected strategy at the graph/plan boundary, rather
        # than strategy.step(), which sync runtime execution intentionally
        # does not call.
        monkeypatch.setattr(GenerationBasedStrategy, "build_graph", gb_graph)
        monkeypatch.setattr(PreSelectionStrategy, "build_graph", ps_graph)

        preset = {
            "strategy": {
                "type": "GenerationBasedStrategy",
                "params": {"gen_ctrl": 2},
            },
        }
        minimize(
            lambda x: np.sum(x**2),
            dim=2,
            lb=[-1, -1],
            ub=[1, 1],
            algorithm="GA",
            surrogate="rbf",
            preset=preset,
            max_fe=30,
            pop_size=6,
            seed=0,
            verbose=False,
        )
        assert calls["gb"] > 0
        assert calls["ps"] == 0

    def test_explicit_strategy_overrides_preset(self, monkeypatch):
        calls = {"gb": 0, "ps": 0}
        orig_gb_graph = GenerationBasedStrategy.build_graph
        orig_ps_graph = PreSelectionStrategy.build_graph

        def gb_graph(self, provider):
            calls["gb"] += 1
            return orig_gb_graph(self, provider)

        def ps_graph(self, provider):
            calls["ps"] += 1
            return orig_ps_graph(self, provider)

        monkeypatch.setattr(GenerationBasedStrategy, "build_graph", gb_graph)
        monkeypatch.setattr(PreSelectionStrategy, "build_graph", ps_graph)

        # preset requests PreSelectionStrategy, but the explicit strategy="gb"
        # argument must take precedence.
        preset = {
            "strategy": {
                "type": "PreSelectionStrategy",
                "params": {"n_candidates": 6, "n_select": 2},
            },
        }
        minimize(
            lambda x: np.sum(x**2),
            dim=2,
            lb=[-1, -1],
            ub=[1, 1],
            algorithm="GA",
            surrogate="rbf",
            strategy="gb",
            preset=preset,
            max_fe=30,
            pop_size=6,
            seed=0,
            verbose=False,
        )
        assert calls["gb"] > 0
        assert calls["ps"] == 0


class TestMaximizePreset:
    def test_completes_with_preset_dict(self):
        preset = {
            "strategy": {
                "type": "GenerationBasedStrategy",
                "params": {"gen_ctrl": 2},
            },
        }
        result = maximize(
            lambda x: -np.sum(x**2),
            dim=2,
            lb=[-1, -1],
            ub=[1, 1],
            algorithm="GA",
            surrogate="rbf",
            preset=preset,
            max_fe=30,
            pop_size=6,
            seed=0,
            verbose=False,
        )
        assert isinstance(result.x, np.ndarray)
        assert result.fe > 0


class TestPopSizeExceedsDefaultArchive:
    def test_minimize_accepts_pop_size_larger_than_default_archive(self):
        dim = 2
        result = minimize(
            lambda x: np.sum(x**2),
            dim=dim,
            lb=[-1] * dim,
            ub=[1] * dim,
            pop_size=5 * dim + 1,
            max_fe=30,
            seed=0,
            verbose=False,
        )
        assert isinstance(result.x, np.ndarray)
        assert result.fe > 0


class TestSurrogateResolution:
    def test_pairwise_subclass_instance_keeps_auto_acquisition(self, monkeypatch):
        class CustomPairwiseManager(PairwiseSurrogateManager):
            pass

        captured: dict[str, Any] = {}

        def stub_run(self, *args, **kwargs):
            captured["acquisition"] = self.acquisition
            raise _ResolvedConfigurationError

        def fail_set_acquisition(self, acquisition):
            raise AssertionError("set_acquisition should not override the manager")

        surrogate = CustomPairwiseManager(SklearnRFCClassificationSurrogate())
        problem = api.Problem(
            func=lambda x: np.sum(x**2),
            dim=2,
            n_obj=1,
            direction=np.array([-1.0]),
            lb=[-1, -1],
            ub=[1, 1],
        )
        _, direct_acquisition = api._resolve_surrogate(surrogate, problem, {})
        assert isinstance(direct_acquisition, WinRateAcquisition)

        monkeypatch.setattr(api.Optimizer, "run", stub_run)
        monkeypatch.setattr(api.Optimizer, "set_acquisition", fail_set_acquisition)

        with pytest.raises(_ResolvedConfigurationError):
            minimize(
                lambda x: np.sum(x**2),
                dim=2,
                lb=[-1, -1],
                ub=[1, 1],
                algorithm="GA",
                surrogate=surrogate,
                strategy="ib",
                max_fe=30,
                pop_size=6,
                seed=7,
                verbose=False,
            )

        assert isinstance(captured["acquisition"], WinRateAcquisition)

    def test_pairwise_subclass_spec_uses_win_rate_acquisition(self):
        import saealib.registry as registry
        from saealib.registry import register

        missing = object()
        previous = registry._REGISTRY.get("CustomPairwiseManager", missing)
        try:

            @register()
            class CustomPairwiseManager(PairwiseSurrogateManager):
                pass

            manager, acquisition = api.Optimizer._build_components_from_spec_static(
                {
                    "type": "CustomPairwiseManager",
                    "params": {
                        "surrogate": SklearnRFCClassificationSurrogate(),
                    },
                },
                2,
                np.array([-1.0]),
            )

            assert isinstance(manager, CustomPairwiseManager)
            assert isinstance(acquisition, WinRateAcquisition)
        finally:
            if previous is missing:
                registry._REGISTRY.pop("CustomPairwiseManager", None)
            else:
                registry._REGISTRY["CustomPairwiseManager"] = previous

    def test_factory_registry_name_uses_mean_prediction_acquisition(self):
        import saealib.registry as registry
        from saealib.registry import register

        missing = object()
        previous = registry._REGISTRY.get("FactoryMadeManager", missing)
        try:

            @register("FactoryMadeManager")
            def factory_made_manager(**kw):
                return LocalSurrogateManager(**kw)

            direction = np.array([-1.0])
            spec = api._default_acquisition_spec("FactoryMadeManager", direction)

            assert spec["type"] == "MeanPrediction"
            np.testing.assert_array_equal(spec["params"]["direction"], direction)
        finally:
            if previous is missing:
                registry._REGISTRY.pop("FactoryMadeManager", None)
            else:
                registry._REGISTRY["FactoryMadeManager"] = previous

    def test_api_path_preserves_user_surrogate_identity(self, monkeypatch):
        captured: dict[str, Any] = {}

        def stub_run(self, *args, **kwargs):
            captured["manager"] = self.surrogate_manager
            raise _ResolvedConfigurationError

        monkeypatch.setattr(api.Optimizer, "run", stub_run)
        surrogate = RBFSurrogate(GaussianKernel())

        with pytest.raises(_ResolvedConfigurationError):
            minimize(
                lambda x: np.sum(x**2),
                dim=2,
                lb=[-1, -1],
                ub=[1, 1],
                algorithm="GA",
                surrogate=surrogate,
                strategy="ib",
                max_fe=30,
                pop_size=6,
                seed=7,
                verbose=False,
            )

        assert captured["manager"].surrogate is surrogate

    def test_same_spec_builds_do_not_share_surrogate_instances(self):
        from saealib.defaults import load_defaults

        defaults = load_defaults()
        default_spec = defaults["presets"]["ga_rbf_ib"]["surrogate_manager"]
        default_spec_snapshot = copy.deepcopy(default_spec)
        spec = copy.deepcopy(default_spec)
        spec.setdefault("params", {})["surrogate"] = RBFSurrogate(GaussianKernel())

        manager1, _ = api.Optimizer._build_components_from_spec_static(
            spec, 2, np.array([-1.0])
        )
        manager2, _ = api.Optimizer._build_components_from_spec_static(
            spec, 2, np.array([-1.0])
        )

        assert isinstance(manager1, LocalSurrogateManager)
        assert isinstance(manager2, LocalSurrogateManager)
        assert manager1.surrogate is not manager2.surrogate
        assert default_spec == default_spec_snapshot
