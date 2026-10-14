"""
Tests for the PSO algorithm.

Tests cover:
- get_required_attrs: attribute names, shapes, defaults
- ask: output size, bounds, velocity storage, v_max clamping
- pbest initialization: NaN → current x/f, already-set → unchanged
- velocity update: pure inertia (w=1, c1=0, c2=0) and zero cognitive term
- tell / pbest update: better replaces, worse keeps, NaN skipped
- _select_leader: returns the best pbest position
"""

from typing import cast

import numpy as np
import pytest

from saealib.algorithms.base import ProposalRequest
from saealib.algorithms.pso import PSO
from saealib.comparators import SingleObjectiveComparator
from saealib.context import OptimizationState
from saealib.core.contracts import FeedbackBatch
from saealib.core.state import StateView
from saealib.exceptions import ConfigurationError
from saealib.population import Archive, ParetoArchive, Population, PopulationAttribute
from saealib.problem import Problem

# ---------------------------------------------------------------------------
# Shared constants and fixtures
# ---------------------------------------------------------------------------

DIM = 5
N_POP = 8

_PSO_ATTRS = [
    PopulationAttribute(name="x", dtype=np.float64, shape=(DIM,)),
    PopulationAttribute(name="f", dtype=np.float64, shape=(1,)),
    PopulationAttribute(name="g", dtype=np.float64, shape=(0,)),
    PopulationAttribute(name="cv", dtype=np.float64, shape=()),
    PopulationAttribute(name="velocity", dtype=np.float64, shape=(DIM,), default=0.0),
    PopulationAttribute(name="pbest_x", dtype=np.float64, shape=(DIM,), default=np.nan),
    PopulationAttribute(name="pbest_f", dtype=np.float64, shape=(1,), default=np.nan),
    PopulationAttribute(name="pbest_cv", dtype=np.float64, shape=(), default=np.nan),
]


class _DummyProvider:
    def dispatch(self, event):
        pass


def _view(ctx: OptimizationState, algorithm: PSO) -> StateView:
    return ctx._store.view(
        algorithm.contract().state,
        context=ctx,
        dispatch=lambda event: None,
    )


def _ask(algorithm: PSO, ctx: OptimizationState):
    return algorithm.ask(ProposalRequest(), _view(ctx, algorithm)).candidates


def _tell(algorithm: PSO, ctx: OptimizationState, offspring: Population):
    ctx.offspring = offspring
    return algorithm.tell(
        cast(FeedbackBatch, object()),
        _view(ctx, algorithm),
    )


class _CountingContext:
    def __init__(self, ctx: OptimizationState) -> None:
        self._ctx = ctx
        self.population_lookups = 0

    @property
    def population(self):
        self.population_lookups += 1
        return self._ctx.population

    def __getattr__(self, name):
        return getattr(self._ctx, name)


def _make_problem() -> Problem:
    # direction=-1.0 in SingleObjectiveComparator: sort_population puts lowest f first
    return Problem(
        func=lambda x: np.array([np.sum(x**2)]),
        dim=DIM,
        n_obj=1,
        direction=np.array([-1.0]),
        lb=[-5.0] * DIM,
        ub=[5.0] * DIM,
        comparator=SingleObjectiveComparator(direction=-1.0),
    )


def _make_pso_ctx(
    rng_seed: int = 0,
    init_pbest: bool = False,
    init_velocity: bool = False,
) -> OptimizationState:
    """
    Build an OptimizationState with a PSO-capable population.

    Parameters
    ----------
    init_pbest : bool
        If True, pbest_x/f/cv are set to the initial x/f/0; otherwise NaN.
    init_velocity : bool
        If True, velocity is randomised; otherwise zero.
    """
    problem = _make_problem()
    rng = np.random.default_rng(rng_seed)

    xs = rng.uniform(-3.0, 3.0, size=(N_POP, DIM))
    fs = np.array([[np.sum(x**2)] for x in xs])
    vs = (
        rng.uniform(-1.0, 1.0, size=(N_POP, DIM))
        if init_velocity
        else np.zeros((N_POP, DIM))
    )

    if init_pbest:
        pbest_x = xs.copy()
        pbest_f = fs.copy()
        pbest_cv = np.zeros(N_POP)
    else:
        pbest_x = np.full((N_POP, DIM), np.nan)
        pbest_f = np.full((N_POP, 1), np.nan)
        pbest_cv = np.full(N_POP, np.nan)

    pop = Population(_PSO_ATTRS, init_capacity=N_POP + 5)
    pop.extend(
        {
            "x": xs,
            "f": fs,
            "g": np.zeros((N_POP, 0)),
            "cv": np.zeros(N_POP),
            "velocity": vs,
            "pbest_x": pbest_x,
            "pbest_f": pbest_f,
            "pbest_cv": pbest_cv,
        }
    )
    arc = Archive(_PSO_ATTRS, init_capacity=5)
    pareto_arc = ParetoArchive(_PSO_ATTRS, init_capacity=5, direction=np.array([-1.0]))
    return OptimizationState(
        problem=problem,
        population=pop,
        archive=arc,
        pareto_archive=pareto_arc,
        rng=np.random.default_rng(rng_seed),
    )


def _make_offspring(
    ctx: OptimizationState,
    f_new: np.ndarray,
    pbest_f: np.ndarray,
    pbest_x: np.ndarray | None = None,
    pbest_cv: np.ndarray | None = None,
) -> Population:
    """Build a simulated evaluated offspring Population for tell() tests."""
    if pbest_x is None:
        pbest_x = ctx.population.get_array("pbest_x").copy()
    if pbest_cv is None:
        pbest_cv = np.zeros(N_POP)
    offspring = ctx.population.empty_like(capacity=N_POP)
    offspring.extend(
        {
            "x": np.zeros((N_POP, DIM)),
            "f": f_new,
            "g": np.zeros((N_POP, 0)),
            "cv": np.zeros(N_POP),
            "velocity": np.zeros((N_POP, DIM)),
            "pbest_x": pbest_x,
            "pbest_f": pbest_f,
            "pbest_cv": pbest_cv,
        }
    )
    return offspring


# ---------------------------------------------------------------------------
# TestPSORequiredAttrs
# ---------------------------------------------------------------------------


class TestPSORequiredAttrs:
    def test_returns_four_attrs(self):
        pso = PSO()
        attrs = pso.get_required_attrs(_make_problem())
        assert len(attrs) == 4

    def test_attr_names(self):
        pso = PSO()
        attrs = pso.get_required_attrs(_make_problem())
        names = {a.name for a in attrs}
        assert names == {"velocity", "pbest_x", "pbest_f", "pbest_cv"}

    def test_velocity_shape_and_default(self):
        pso = PSO()
        attrs = pso.get_required_attrs(_make_problem())
        vel = next(a for a in attrs if a.name == "velocity")
        assert vel.shape == (DIM,)
        assert vel.default == pytest.approx(0.0)

    def test_pbest_f_shape_matches_n_obj(self):
        pso = PSO()
        attrs = pso.get_required_attrs(_make_problem())
        pbf = next(a for a in attrs if a.name == "pbest_f")
        assert pbf.shape == (1,)

    def test_pbest_x_default_is_nan(self):
        pso = PSO()
        attrs = pso.get_required_attrs(_make_problem())
        pbx = next(a for a in attrs if a.name == "pbest_x")
        assert np.isnan(pbx.default)


# ---------------------------------------------------------------------------
# TestPSOAsk
# ---------------------------------------------------------------------------


class TestPSOAsk:
    def test_output_size(self):
        pso = PSO()
        ctx = _make_pso_ctx()
        cand = _ask(pso, ctx)
        assert len(cand) == N_POP

    def test_candidates_within_bounds(self):
        pso = PSO()
        ctx = _make_pso_ctx()
        cand = _ask(pso, ctx)
        x = cand.get_array("x")
        lb = np.array(ctx.problem.lb)
        ub = np.array(ctx.problem.ub)
        assert np.all(x >= lb - 1e-9)
        assert np.all(x <= ub + 1e-9)

    def test_velocity_stored_in_candidates(self):
        pso = PSO()
        ctx = _make_pso_ctx()
        cand = _ask(pso, ctx)
        assert cand.get_array("velocity").shape == (N_POP, DIM)

    def test_vmax_clamps_velocity(self):
        v_max = 0.5
        pso = PSO(v_max=v_max)
        ctx = _make_pso_ctx(init_velocity=True)
        cand = _ask(pso, ctx)
        v = cand.get_array("velocity")
        assert np.all(np.abs(v) <= v_max + 1e-9)

    def test_repair_clips_to_bounds(self):
        # Large velocity: handler.repair (default = clip) keeps positions in bounds.
        pso = PSO(w=100.0, c1=0.0, c2=0.0)
        ctx = _make_pso_ctx(init_pbest=True, init_velocity=True)
        cand = _ask(pso, ctx)
        x = cand.get_array("x")
        lb = np.array(ctx.problem.lb)
        ub = np.array(ctx.problem.ub)
        assert np.all(x >= lb) and np.all(x <= ub)


# ---------------------------------------------------------------------------
# TestPSOPbestInit
# ---------------------------------------------------------------------------


class TestPSOPbestInit:
    def test_pbest_initialized_when_nan(self):
        pso = PSO()
        ctx = _make_pso_ctx(init_pbest=False)
        cand = _ask(pso, ctx)
        assert not np.any(np.isnan(cand.get_array("pbest_f")))

    def test_pbest_initialized_from_current_f(self):
        pso = PSO()
        ctx = _make_pso_ctx(init_pbest=False)
        f_before = ctx.population.get_array("f").copy()
        cand = _ask(pso, ctx)
        np.testing.assert_array_equal(cand.get_array("pbest_f"), f_before)

    def test_pbest_initialized_from_current_x(self):
        pso = PSO()
        ctx = _make_pso_ctx(init_pbest=False)
        x_before = ctx.population.get_array("x").copy()
        cand = _ask(pso, ctx)
        np.testing.assert_array_equal(cand.get_array("pbest_x"), x_before)

    def test_already_initialized_pbest_not_overwritten(self):
        pso = PSO()
        ctx = _make_pso_ctx(init_pbest=True)
        pbest_f_before = ctx.population.get_array("pbest_f").copy()
        cand = _ask(pso, ctx)
        np.testing.assert_array_equal(cand.get_array("pbest_f"), pbest_f_before)


# ---------------------------------------------------------------------------
# TestPSOVelocityUpdate
# ---------------------------------------------------------------------------


class TestPSOVelocityUpdate:
    def test_pure_inertia(self):
        # w=1, c1=0, c2=0 → v_new = v_old
        pso = PSO(w=1.0, c1=0.0, c2=0.0)
        ctx = _make_pso_ctx(init_pbest=True, init_velocity=True)
        v_before = ctx.population.get_array("velocity").copy()
        cand = _ask(pso, ctx)
        np.testing.assert_allclose(cand.get_array("velocity"), v_before)

    def test_zero_velocity_when_at_pbest_no_social(self):
        # w=0, c1=1, c2=0, x == pbest_x → cognitive term is 0 → v_new = 0
        pso = PSO(w=0.0, c1=1.0, c2=0.0)
        ctx = _make_pso_ctx(init_pbest=True)
        # Force x == pbest_x so (pbest_x - x) = 0
        x = ctx.population.get_array("x")
        ctx.population.update_array("pbest_x", x)
        cand = _ask(pso, ctx)
        np.testing.assert_allclose(cand.get_array("velocity"), 0.0, atol=1e-12)


# ---------------------------------------------------------------------------
# TestPSOPbestUpdate
# ---------------------------------------------------------------------------


class TestPSOPbestUpdate:
    def test_better_solution_updates_pbest(self):
        pso = PSO()
        ctx = _make_pso_ctx(init_pbest=True)
        old_pbest_f = ctx.population.get_array("pbest_f").copy()
        # f=0 is strictly better than any positive pbest_f
        f_new = np.zeros((N_POP, 1))
        offspring = _make_offspring(ctx, f_new=f_new, pbest_f=old_pbest_f)
        _tell(pso, ctx, offspring)
        np.testing.assert_allclose(ctx.population.get_array("pbest_f"), 0.0)

    def test_worse_solution_does_not_update_pbest(self):
        pso = PSO()
        ctx = _make_pso_ctx(init_pbest=True)
        old_pbest_f = ctx.population.get_array("pbest_f").copy()
        # f=1e9 is worse than any reasonable pbest_f
        f_new = np.full((N_POP, 1), 1e9)
        offspring = _make_offspring(ctx, f_new=f_new, pbest_f=old_pbest_f)
        _tell(pso, ctx, offspring)
        np.testing.assert_array_equal(ctx.population.get_array("pbest_f"), old_pbest_f)

    def test_nan_f_new_does_not_update_pbest(self):
        pso = PSO()
        ctx = _make_pso_ctx(init_pbest=True)
        old_pbest_f = ctx.population.get_array("pbest_f").copy()
        f_new = np.full((N_POP, 1), np.nan)
        offspring = _make_offspring(ctx, f_new=f_new, pbest_f=old_pbest_f)
        _tell(pso, ctx, offspring)
        np.testing.assert_array_equal(ctx.population.get_array("pbest_f"), old_pbest_f)

    def test_population_x_updated_after_tell(self):
        pso = PSO()
        ctx = _make_pso_ctx(init_pbest=True)
        population = ctx.population
        x_new = np.ones((N_POP, DIM)) * 0.5
        f_new = np.full((N_POP, 1), 1e9)  # worse, so pbest stays
        offspring = _make_offspring(
            ctx, f_new=f_new, pbest_f=ctx.population.get_array("pbest_f").copy()
        )
        offspring.update_array("x", x_new)
        counting_ctx = _CountingContext(ctx)
        _tell(pso, cast(OptimizationState, counting_ctx), offspring)
        assert counting_ctx.population_lookups == 1
        assert ctx.population is population
        np.testing.assert_array_equal(ctx.population.get_array("x"), x_new)


# ---------------------------------------------------------------------------
# TestPSOLeaderSelection
# ---------------------------------------------------------------------------


class TestPSOTellGuards:
    def test_missing_offspring_raises_configuration_error(self):
        pso = PSO()
        ctx = _make_pso_ctx(init_pbest=True)
        with pytest.raises(ConfigurationError, match="offspring population"):
            pso.tell(cast(FeedbackBatch, object()), _view(ctx, pso))


class TestPSOLeaderSelection:
    def test_leader_is_best_pbest(self):
        pso = PSO()
        ctx = _make_pso_ctx(init_pbest=True)
        pbest_x = ctx.population.get_array("pbest_x").copy()
        pbest_f = ctx.population.get_array("pbest_f").copy()
        pbest_cv = ctx.population.get_array("pbest_cv").copy()

        # Make particle 3 clearly the best (f=0, lower is better for SOC)
        best_idx = 3
        pbest_f[:] = 100.0
        pbest_f[best_idx] = 0.0
        pbest_x[best_idx] = np.zeros(DIM)

        leader = pso._select_leader(ctx, pbest_x, pbest_f, pbest_cv)
        np.testing.assert_array_equal(leader, pbest_x[best_idx])

    def test_leader_shape(self):
        pso = PSO()
        ctx = _make_pso_ctx(init_pbest=True)
        pbest_x = ctx.population.get_array("pbest_x").copy()
        pbest_f = ctx.population.get_array("pbest_f").copy()
        pbest_cv = ctx.population.get_array("pbest_cv").copy()
        leader = pso._select_leader(ctx, pbest_x, pbest_f, pbest_cv)
        assert leader.shape == (DIM,)


# ---------------------------------------------------------------------------
# PSO class-level properties
# ---------------------------------------------------------------------------


class TestPSOProperties:
    def test_population_class(self):
        assert PSO().population_class is Population

    def test_archive_class(self):
        assert PSO().archive_class is Archive

    def test_ask_notation_is_list_of_strings(self):
        n = PSO().ask_notation
        assert isinstance(n, list)
        assert len(n) == 2
        assert all(isinstance(s, str) for s in n)

    def test_tell_notation_is_list_of_strings(self):
        n = PSO().tell_notation
        assert isinstance(n, list)
        assert len(n) == 2
        assert all(isinstance(s, str) for s in n)
