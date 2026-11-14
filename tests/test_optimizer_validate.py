"""Tests for Optimizer.validate()."""

from __future__ import annotations

from typing import Any, cast
from unittest.mock import MagicMock

import numpy as np
import pytest

from saealib.acquisition.ei import ExpectedImprovement
from saealib.acquisition.mean import MeanPrediction
from saealib.acquisition.pof import ProbabilityOfFeasibility, ProductOfFeasibility
from saealib.acquisition.uncertainty import MaxUncertainty
from saealib.algorithms import PSO
from saealib.comparators import ParetoComparator
from saealib.core.contracts import ComponentContract
from saealib.execution.initializer import LHSInitializer
from saealib.optimizer import Optimizer
from saealib.problem import Problem
from saealib.strategies.base import OptimizationStrategy
from saealib.strategies.direct import DirectStrategy
from saealib.strategies.ib import IndividualBasedStrategy
from saealib.surrogate.manager import GlobalSurrogateManager
from saealib.surrogate.rbf import RBFSurrogate
from saealib.surrogate.rbf_kernels import GaussianKernel
from saealib.surrogate.sklearn_surrogate import SklearnGPRSurrogate
from saealib.termination import Termination, max_gen

DIM = 2
N_OBJ = 1


def _make_problem(n_obj: int = N_OBJ) -> Problem:
    return Problem(
        func=lambda _: np.zeros(n_obj),
        dim=DIM,
        n_obj=n_obj,
        direction=np.array([-1.0] * n_obj),
        lb=[-5.0] * DIM,
        ub=[5.0] * DIM,
    )


def _stub_strategy(requires_surrogate: bool = False) -> OptimizationStrategy:
    m = MagicMock(spec=OptimizationStrategy)
    m.requires_surrogate = requires_surrogate
    m.contract.return_value = ComponentContract()
    return m


def _contract_mock() -> MagicMock:
    component = MagicMock()
    component.contract.return_value = ComponentContract()
    return component


def _fully_configured() -> Optimizer:
    opt = Optimizer(_make_problem())
    opt.resolve_defaults()
    return opt


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_fully_configured_returns_empty():
    assert _fully_configured().validate() == []


# ---------------------------------------------------------------------------
# Missing required components
# ---------------------------------------------------------------------------


def test_missing_algorithm_is_resolved_before_validation():
    opt = _fully_configured()
    del opt.algorithm
    assert not any("missing_required_component" in m for m in opt.validate())


def test_missing_strategy_is_resolved_before_validation():
    opt = _fully_configured()
    del opt.strategy
    assert not any("missing_required_component" in m for m in opt.validate())


def test_missing_initializer_is_resolved_before_validation():
    opt = _fully_configured()
    opt.initializer = None
    assert not any("missing_required_component" in m for m in opt.validate())


def test_missing_termination_is_resolved_before_validation():
    opt = _fully_configured()
    del opt.termination
    assert not any("missing_required_component" in m for m in opt.validate())


def test_all_required_components_are_available_after_default_preparation():
    opt = Optimizer(_make_problem())
    assert opt.validate() == []
    assert not hasattr(opt, "algorithm")
    assert not hasattr(opt, "strategy")


# ---------------------------------------------------------------------------
# Strategy x Surrogate
# ---------------------------------------------------------------------------


def test_strategy_surrogate_requirement_uses_prepared_manager():
    opt = _fully_configured()
    opt.set_strategy(IndividualBasedStrategy())
    assert not any(
        "missing_surrogate_manager" in item.code for item in opt.contract_diagnostics()
    )


def test_strategy_requires_surrogate_present():
    opt = _fully_configured()
    opt.set_strategy(IndividualBasedStrategy())
    opt.set_surrogate_manager(
        GlobalSurrogateManager(
            RBFSurrogate(kernel=GaussianKernel()),
            cast(Any, MeanPrediction()),
        )
    )
    assert not any("surrogate_manager" in m for m in opt.validate())


# ---------------------------------------------------------------------------
# Comparator x n_obj
# ---------------------------------------------------------------------------


def test_comparator_weight_mismatch():
    opt = _fully_configured()
    opt.set_comparator(ParetoComparator(direction=np.array([1.0, 1.0])))
    assert any("direction" in m for m in opt.validate())


# ---------------------------------------------------------------------------
# Acquisition x Surrogate uncertainty
# ---------------------------------------------------------------------------


def test_acquisition_uncertainty_mismatch():
    opt = _fully_configured()
    opt.set_strategy(IndividualBasedStrategy())
    opt.set_algorithm(PSO())
    opt.set_surrogate_manager(
        GlobalSurrogateManager(
            RBFSurrogate(kernel=GaussianKernel()),
        )
    )
    opt.set_acquisition(ExpectedImprovement())
    assert any("uncertainty" in m for m in opt.validate())


def test_mean_prediction_with_rbf_ok():
    opt = _fully_configured()
    opt.set_surrogate_manager(
        GlobalSurrogateManager(
            RBFSurrogate(kernel=GaussianKernel()),
        )
    )
    opt.set_acquisition(MeanPrediction())
    assert opt.validate() == []


# ---------------------------------------------------------------------------
# run() / iterate() raise on misconfiguration that defaults cannot resolve
# ---------------------------------------------------------------------------


def test_run_raises_on_unresolvable_misconfiguration():
    opt = _fully_configured()
    opt.set_comparator(ParetoComparator(direction=np.array([1.0, 1.0])))
    with pytest.raises(ValueError, match="Optimizer misconfigured"):
        opt.run()


def test_iterate_raises_on_unresolvable_misconfiguration():
    opt = _fully_configured()
    opt.set_comparator(ParetoComparator(direction=np.array([1.0, 1.0])))
    with pytest.raises(ValueError, match="Optimizer misconfigured"):
        opt.iterate()


# ---------------------------------------------------------------------------
# run() / iterate() auto-resolve unset components instead of raising
# ---------------------------------------------------------------------------


def test_run_succeeds_with_no_components_set():
    opt = Optimizer(_make_problem())
    ctx = opt.run()
    assert ctx.fe > 0


def test_iterate_succeeds_with_no_components_set():
    opt = Optimizer(_make_problem())
    ctx = next(opt.iterate())
    assert ctx is not None


def test_defaults_resolution_exposes_components_for_validation_and_pipeline():
    opt = Optimizer(_make_problem())

    assert opt.validate() == []
    assert not hasattr(opt, "algorithm")
    assert not hasattr(opt, "strategy")

    opt.resolve_defaults()
    algorithm, strategy = opt.algorithm, opt.strategy

    assert algorithm is not None
    assert strategy is not None
    assert not any(" is not set" in issue for issue in opt.validate())
    assert strategy.build_pipeline(opt) is not None

    opt.resolve_defaults()
    assert opt.algorithm is algorithm
    assert opt.strategy is strategy


# ---------------------------------------------------------------------------
# Acquisition direction injection
# ---------------------------------------------------------------------------


def test_acquisition_direction_length_mismatch():
    opt = _fully_configured()
    opt.set_surrogate_manager(
        GlobalSurrogateManager(
            RBFSurrogate(kernel=GaussianKernel()),
        )
    )
    opt.set_acquisition(ExpectedImprovement(direction=np.array([1.0, 1.0])))
    assert any("direction" in m for m in opt.validate())


def test_inject_acquisition_directions_sets_from_problem():
    problem = _make_problem(n_obj=2)
    opt = Optimizer(problem)
    opt.set_algorithm(MagicMock())
    opt.set_strategy(_stub_strategy())
    opt.initializer = MagicMock()
    opt.set_termination(MagicMock())
    acq = MeanPrediction()
    opt.set_surrogate_manager(
        GlobalSurrogateManager(RBFSurrogate(kernel=GaussianKernel()))
    )
    opt.set_acquisition(acq)
    opt._inject_acquisition_directions()
    np.testing.assert_array_equal(acq.direction, problem.direction)


def test_inject_acquisition_directions_idempotent():
    opt = _fully_configured()
    acq = MeanPrediction()
    opt.set_surrogate_manager(
        GlobalSurrogateManager(RBFSurrogate(kernel=GaussianKernel()))
    )
    opt.set_acquisition(acq)
    opt._inject_acquisition_directions()
    first = acq.direction
    opt._inject_acquisition_directions()
    assert acq.direction is first


def test_inject_acquisition_directions_preserves_explicit_direction():
    opt = _fully_configured()
    explicit = np.array([1.0])
    acq = MeanPrediction(direction=explicit)
    opt.set_surrogate_manager(
        GlobalSurrogateManager(RBFSurrogate(kernel=GaussianKernel()))
    )
    opt.set_acquisition(acq)
    opt._inject_acquisition_directions()
    assert acq.direction is explicit


def test_inject_acquisition_directions_skips_direction_insensitive():
    """PoF/ProductOfFeasibility/MaxUncertainty opt out and never get a direction."""
    opt = _fully_configured()
    for acq in (
        ProbabilityOfFeasibility(),
        ProductOfFeasibility(),
        MaxUncertainty(),
    ):
        opt.set_surrogate_manager(
            GlobalSurrogateManager(RBFSurrogate(kernel=GaussianKernel()))
        )
        opt.set_acquisition(acq)
        opt._inject_acquisition_directions()
        # These acquisitions never declare a `direction` field at all; opting
        # out via direction_sensitive=False means injection must not add one.
        assert getattr(acq, "direction", None) is None


def test_iterate_injects_acquisition_direction_end_to_end():
    """End-to-end: iterate() auto-injects problem.direction into an unset
    acquisition."""
    dim = 2
    problem = Problem(
        func=lambda x: np.array([-np.sum(x**2)]),
        dim=dim,
        n_obj=1,
        direction=np.array([1.0]),  # maximize
        lb=[-5.0] * dim,
        ub=[5.0] * dim,
    )
    acq = ExpectedImprovement()
    opt = (
        Optimizer(problem, seed=0)
        .set_initializer(LHSInitializer(n_init_archive=10, n_init_population=8, seed=0))
        .set_strategy(IndividualBasedStrategy(evaluation_ratio=0.5))
        .set_surrogate_manager(GlobalSurrogateManager(SklearnGPRSurrogate()))
        .set_acquisition(acq)
    )
    gen = opt.iterate()
    next(gen)

    np.testing.assert_array_equal(acq.direction, problem.direction)

    # Idempotent: re-injecting does not change an already-set direction.
    opt._inject_acquisition_directions()
    np.testing.assert_array_equal(acq.direction, problem.direction)


def test_validate_reuses_prepared_diagnostics_and_reprepares_after_a_replacement(
    monkeypatch,
) -> None:
    from saealib import optimizer as optimizer_module

    optimizer = (
        Optimizer(_make_problem(), seed=7)
        .set_initializer(LHSInitializer(6, 4, 7))
        .set_algorithm(PSO())
        .set_strategy(DirectStrategy())
        .set_termination(Termination(max_gen(3)))
    )
    original = optimizer_module._prepare_plan
    preparations: list[int] = []

    def counted(*args: Any, **kwargs: Any) -> Any:
        preparations.append(len(preparations))
        return original(*args, **kwargs)

    monkeypatch.setattr(optimizer_module, "_prepare_plan", counted)

    optimizer._compile_plan()
    compiled = len(preparations)
    optimizer.set_algorithm(optimizer.algorithm)
    optimizer.validate()
    assert len(preparations) == compiled

    optimizer._compile_plan()
    compiled = len(preparations)
    optimizer.set_algorithm(PSO())
    optimizer.validate()
    assert len(preparations) == compiled + 1
