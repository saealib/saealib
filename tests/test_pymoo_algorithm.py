"""Tests for PymooAlgorithm against the real pymoo library."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest
from _algorithm_boundary import ask as algorithm_ask
from _algorithm_boundary import tell as algorithm_tell
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.soo.nonconvex.ga import GA as PymooGA  # noqa: N811

from saealib import minimize
from saealib.algorithms.pymoo_algorithm import PymooAlgorithm
from saealib.comparators import NSGA2Comparator, SingleObjectiveComparator
from saealib.context import OptimizationState
from saealib.exceptions import ConfigurationError
from saealib.population import Archive, ParetoArchive, Population, PopulationAttribute
from saealib.problem import Problem
from saealib.problem.constraint import InequalityConstraint
from saealib.strategies.direct import DirectStrategy
from saealib.strategies.ps import PreSelectionStrategy

DIM = 5
N_POP = 10


class _DummyProvider:
    def dispatch(self, event):
        pass


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


def _make_problem(
    n_obj: int = 1, direction: float = -1.0, constrained: bool = False
) -> Problem:
    if n_obj == 1:
        comparator = SingleObjectiveComparator(direction=direction)

        def func(x: np.ndarray) -> np.ndarray:
            return np.array([np.sum(x**2)])
    else:
        comparator = NSGA2Comparator(direction=np.full(n_obj, direction))

        def func(x: np.ndarray) -> np.ndarray:
            f1 = x[0]
            g = 1.0 + 9.0 * np.sum(x[1:]) / (len(x) - 1)
            f2 = g * (1.0 - np.sqrt(max(f1 / g, 0.0)))
            return np.array([f1, f2])

    constraints = (
        [InequalityConstraint(lambda x: x[0] - 3.0, threshold=0.0)]
        if constrained
        else None
    )

    return Problem(
        func=func,
        dim=DIM,
        n_obj=n_obj,
        direction=np.full(n_obj, direction),
        lb=[0.0] * DIM if n_obj > 1 else [-5.0] * DIM,
        ub=[1.0] * DIM if n_obj > 1 else [5.0] * DIM,
        comparator=comparator,
        constraints=constraints,
    )


def _make_ctx(
    algo: PymooAlgorithm,
    problem: Problem,
    n_pop: int = N_POP,
    rng_seed: int = 0,
    with_ids: bool = False,
    with_objective_source: bool = False,
) -> OptimizationState:
    rng = np.random.default_rng(rng_seed)
    attrs = [
        PopulationAttribute("x", float, (problem.dim,), default=np.nan),
        PopulationAttribute("f", float, (problem.n_obj,), default=np.nan),
        PopulationAttribute("g", float, (problem.n_constraints,), default=0.0),
        PopulationAttribute("cv", float, (), default=0.0),
        *algo.get_required_attrs(problem),
    ]
    if with_ids:
        attrs.insert(4, PopulationAttribute("id", np.int64, (), default=-1))
    if with_objective_source:
        attrs.append(PopulationAttribute("objective_source", np.uint8, (), default=0))
    pop = Population(attrs, init_capacity=n_pop + 5)
    xs = rng.uniform(problem.lb, problem.ub, size=(n_pop, problem.dim))
    fs = np.array([problem.func(x) for x in xs])
    gs = np.zeros((n_pop, problem.n_constraints))
    cvs = np.zeros(n_pop)
    data = {"x": xs, "f": fs, "g": gs, "cv": cvs}
    if with_ids:
        pop._extend_internal(
            {**data, "id": np.arange(n_pop, dtype=np.int64)}, preserve_ids=True
        )
    else:
        pop.extend(data)
    arc = Archive(attrs, init_capacity=n_pop + 5)
    pareto_arc = ParetoArchive(
        attrs, init_capacity=n_pop + 5, direction=problem.direction
    )
    return OptimizationState(
        problem=problem,
        population=pop,
        archive=arc,
        pareto_archive=pareto_arc,
        rng=np.random.default_rng(rng_seed),
    )


# ---------------------------------------------------------------------------
# ask / tell round trip (whitebox)
# ---------------------------------------------------------------------------


class TestPymooAlgorithmAskTell:
    def test_ask_produces_popsize_candidates(self):
        problem = _make_problem()
        algo = PymooAlgorithm(PymooGA(pop_size=N_POP))
        ctx = _make_ctx(algo, problem)
        cand = algorithm_ask(algo, ctx, _DummyProvider())
        assert len(cand) == N_POP
        assert np.all(cand.x >= problem.lb) and np.all(cand.x <= problem.ub)
        np.testing.assert_array_equal(
            np.sort(cand.get_array("pymoo_idx")), np.arange(N_POP)
        )

    def test_ask_and_sync_resolve_population_once_each(self):
        problem = _make_problem()
        algo = PymooAlgorithm(PymooGA(pop_size=N_POP))
        ctx = _make_ctx(algo, problem)
        population = ctx.population
        algorithm_ask(algo, ctx, _DummyProvider())

        counting_ctx = _CountingContext(ctx)
        algorithm_ask(algo, cast(OptimizationState, counting_ctx), _DummyProvider())
        assert counting_ctx.population_lookups == 1
        algo._sync_population(cast(OptimizationState, counting_ctx))
        assert counting_ctx.population_lookups == 2
        assert ctx.population is population

    def test_ask_tell_round_trip_updates_population(self):
        problem = _make_problem()
        algo = PymooAlgorithm(PymooGA(pop_size=N_POP))
        ctx = _make_ctx(algo, problem)
        cand = algorithm_ask(algo, ctx, _DummyProvider())
        cand.update_array("f", np.array([problem.func(x) for x in cand.x]))
        algorithm_tell(algo, ctx, cand, _DummyProvider())
        assert len(ctx.population) == N_POP
        assert np.isfinite(ctx.population.get_array("f")).all()

    def test_tell_with_reordered_offspring_still_aligns(self):
        """Simulates SortByScoreStage: offspring reordered before tell().

        The pymoo_idx column must recover the correct X<->F mapping even
        after a permutation, otherwise pymoo's internal survival compares
        the wrong F against the wrong X.
        """
        problem = _make_problem()
        algo = PymooAlgorithm(PymooGA(pop_size=N_POP))
        ctx = _make_ctx(algo, problem)
        cand = algorithm_ask(algo, ctx, _DummyProvider())

        f_correct = np.array([problem.func(x) for x in cand.x])
        cand.update_array("f", f_correct)

        perm = np.random.default_rng(1).permutation(N_POP)
        reordered = cand.extract(perm)
        # sanity: the permutation actually changed the order
        assert not np.array_equal(
            reordered.get_array("pymoo_idx"), cand.get_array("pymoo_idx")
        )

        algorithm_tell(algo, ctx, reordered, _DummyProvider())

        # every surviving x must match its own true f, not a mismatched one
        for x, f in zip(ctx.population.x, ctx.population.f, strict=True):
            np.testing.assert_allclose(f, problem.func(x), atol=1e-8)

    def test_maximize_sign_conversion(self):
        """F must be negated for saealib direction=+1 (maximize) and passed
        through unmodified for direction=-1 (minimize)."""
        for direction, expect_negated in [(-1.0, False), (1.0, True)]:
            problem = _make_problem(direction=direction)
            algo = PymooAlgorithm(PymooGA(pop_size=N_POP))
            ctx = _make_ctx(algo, problem)
            cand = algorithm_ask(algo, ctx, _DummyProvider())
            f = np.array([problem.func(x) for x in cand.x])
            cand.update_array("f", f)
            algorithm_tell(algo, ctx, cand, _DummyProvider())

            alg_pop = algo.pymoo_algorithm.pop
            assert alg_pop is not None
            pymoo_f = np.asarray(alg_pop.get("F"), dtype=float)
            saealib_f = ctx.population.get_array("f")
            # pymoo always stores minimization-form F; saealib_f is the
            # direction-aware value, so they should differ in sign iff
            # direction == +1.
            if expect_negated:
                np.testing.assert_allclose(pymoo_f, -saealib_f, atol=1e-8)
            else:
                np.testing.assert_allclose(pymoo_f, saealib_f, atol=1e-8)

    def test_multi_objective_ask_tell(self):
        problem = _make_problem(n_obj=2)
        algo = PymooAlgorithm(NSGA2(pop_size=20))
        ctx = _make_ctx(algo, problem, n_pop=20)
        cand = algorithm_ask(algo, ctx, _DummyProvider())
        assert len(cand) == 20
        f = np.array([problem.func(x) for x in cand.x])
        cand.update_array("f", f)
        algorithm_tell(algo, ctx, cand, _DummyProvider())
        assert len(ctx.population) == 20
        assert ctx.population.get_array("f").shape == (20, 2)

    def test_constrained_g_round_trips(self):
        problem = _make_problem(constrained=True)
        algo = PymooAlgorithm(PymooGA(pop_size=N_POP))
        ctx = _make_ctx(algo, problem)
        cand = algorithm_ask(algo, ctx, _DummyProvider())
        g = np.array([problem.evaluate_constraints(x)[0] for x in cand.x])
        f = np.array([problem.func(x) for x in cand.x])
        cand.update_array("f", f)
        cand.update_array("g", g)
        algorithm_tell(algo, ctx, cand, _DummyProvider())
        assert ctx.population.get_array("g").shape == (N_POP, 1)
        assert np.isfinite(ctx.population.get_array("cv")).all()

    def test_popsize_mismatch_raises_configuration_error(self):
        problem = _make_problem()
        algo = PymooAlgorithm(PymooGA(pop_size=N_POP + 3))
        ctx = _make_ctx(algo, problem, n_pop=N_POP)
        with pytest.raises(ConfigurationError):
            algorithm_ask(algo, ctx, _DummyProvider())

    def test_partial_tell_default_raises(self):
        problem = _make_problem()
        algo = PymooAlgorithm(PymooGA(pop_size=N_POP))
        ctx = _make_ctx(algo, problem)
        cand = algorithm_ask(algo, ctx, _DummyProvider())
        cand.update_array("f", np.array([problem.func(x) for x in cand.x]))
        truncated = cand.extract(np.arange(N_POP - 2))
        with pytest.raises(ConfigurationError):
            algorithm_tell(algo, ctx, truncated, _DummyProvider())

    def test_partial_tell_opt_in_runs(self):
        problem = _make_problem()
        algo = PymooAlgorithm(PymooGA(pop_size=N_POP), allow_partial_tell=True)
        ctx = _make_ctx(algo, problem)
        cand = algorithm_ask(algo, ctx, _DummyProvider())
        cand.update_array("f", np.array([problem.func(x) for x in cand.x]))
        truncated = cand.extract(np.arange(N_POP - 2))
        algorithm_tell(algo, ctx, truncated, _DummyProvider())  # should not raise
        assert len(ctx.population) == N_POP


# ---------------------------------------------------------------------------
# _sync_population candidate-ID continuity
# ---------------------------------------------------------------------------


class TestPymooSyncPopulationIdContinuity:
    def test_resyncing_unchanged_pop_preserves_ids(self):
        problem = _make_problem()
        algo = PymooAlgorithm(PymooGA(pop_size=N_POP))
        attrs = [
            PopulationAttribute("x", float, (problem.dim,), default=np.nan),
            PopulationAttribute("f", float, (problem.n_obj,), default=np.nan),
            PopulationAttribute("g", float, (problem.n_constraints,), default=0.0),
            PopulationAttribute("cv", float, (), default=0.0),
            PopulationAttribute("id", np.int64, (), default=-1),
            *algo.get_required_attrs(problem),
        ]
        rng = np.random.default_rng(0)
        xs = rng.uniform(problem.lb, problem.ub, size=(N_POP, problem.dim))
        fs = np.array([problem.func(x) for x in xs])
        pop = Population(attrs, init_capacity=N_POP + 5)
        pop._extend_internal(
            {
                "x": xs,
                "f": fs,
                "g": np.zeros((N_POP, problem.n_constraints)),
                "cv": np.zeros(N_POP),
                "id": np.arange(N_POP, dtype=np.int64),
            },
            preserve_ids=True,
        )
        arc = Archive(attrs, init_capacity=N_POP + 5)
        pareto_arc = ParetoArchive(
            attrs, init_capacity=N_POP + 5, direction=problem.direction
        )
        ctx = OptimizationState(
            problem=problem,
            population=pop,
            archive=arc,
            pareto_archive=pareto_arc,
            rng=np.random.default_rng(0),
        )
        ctx.candidate_id_allocator.allocate(N_POP)  # keep ahead of seeded ids

        cand = algorithm_ask(algo, ctx, _DummyProvider())
        cand.update_array("f", np.array([problem.func(x) for x in cand.x]))
        algorithm_tell(algo, ctx, cand, _DummyProvider())

        ids_after_first_sync = ctx.population.get_array("id").copy()
        x_after_first_sync = ctx.population.get_array("x").copy()
        assert np.all(ids_after_first_sync != -1)
        assert len(np.unique(ids_after_first_sync)) == N_POP

        # Re-sync with no intervening ask()/tell(): the wrapped algorithm's
        # .pop is unchanged, so every row's x is identical -- ids must be
        # preserved exactly, not re-minted.
        algo._sync_population(ctx)
        np.testing.assert_array_equal(ctx.population.get_array("x"), x_after_first_sync)
        np.testing.assert_array_equal(
            ctx.population.get_array("id"), ids_after_first_sync
        )

    def test_duplicate_x_rows_keep_explicit_pymoo_ids(self):
        problem = _make_problem()
        algo = PymooAlgorithm(PymooGA(pop_size=N_POP))
        ctx = _make_ctx(algo, problem, with_ids=True)
        ctx.population.update_array("x", np.zeros_like(ctx.population.x))
        ctx.candidate_id_allocator.allocate(N_POP)

        cand = algorithm_ask(algo, ctx, _DummyProvider())
        cand.update_array("f", np.array([problem.func(x) for x in cand.x]))
        algorithm_tell(algo, ctx, cand, _DummyProvider())

        ids = ctx.population.get_array("id")
        xs = ctx.population.get_array("x")
        assert len(np.unique(ids)) == len(ids)
        for row in range(len(xs)):
            matches = np.flatnonzero(np.all(xs == xs[row], axis=1))
            if len(matches) > 1:
                assert len(np.unique(ids[matches])) == len(matches)

    @pytest.mark.parametrize("offspring_objective", [0.0, 1_000.0])
    def test_survivor_ids_match_pymoo_provenance_for_both_sources(
        self, offspring_objective
    ):
        problem = _make_problem()
        algo = PymooAlgorithm(NSGA2(pop_size=N_POP))
        ctx = _make_ctx(algo, problem, with_ids=True)
        ctx.candidate_id_allocator.allocate(N_POP)
        parent_ids = ctx.population.get_array("id").copy()
        cand = algorithm_ask(algo, ctx, _DummyProvider())
        offspring_ids = cand.get_array("id").copy()
        cand.update_array(
            "f", np.full((len(cand), 1), offspring_objective, dtype=np.float64)
        )

        algorithm_tell(algo, ctx, cand, _DummyProvider())

        pymoo_ids = np.asarray(
            cast(Any, algo.pymoo_algorithm.pop).get("saealib_candidate_id"),
            dtype=np.int64,
        )
        np.testing.assert_array_equal(ctx.population.get_array("id"), pymoo_ids)
        assert set(pymoo_ids) <= set(parent_ids) | set(offspring_ids)
        expected_source = offspring_ids if offspring_objective == 0.0 else parent_ids
        assert set(pymoo_ids) == set(expected_source)

    @pytest.mark.parametrize("offspring_objective", [0.0, 1_000.0])
    def test_sync_carries_objective_source_of_each_survivor(self, offspring_objective):
        problem = _make_problem()
        algo = PymooAlgorithm(NSGA2(pop_size=N_POP))
        ctx = _make_ctx(algo, problem, with_ids=True, with_objective_source=True)
        ctx.candidate_id_allocator.allocate(N_POP)
        parent_sources = (np.arange(N_POP) % 2).astype(np.uint8)
        ctx.population.update_array("objective_source", parent_sources)
        cand = algorithm_ask(algo, ctx, _DummyProvider())
        cand.update_array(
            "f", np.full((len(cand), 1), offspring_objective, dtype=np.float64)
        )
        cand.update_array("objective_source", np.full(len(cand), 2, dtype=np.uint8))
        known = dict(
            zip(
                ctx.population.get_array("id").tolist(),
                parent_sources.tolist(),
                strict=True,
            )
        )
        known.update((int(i), 2) for i in cand.get_array("id"))

        algorithm_tell(algo, ctx, cand, _DummyProvider())

        expected = [known[int(i)] for i in ctx.population.get_array("id")]
        np.testing.assert_array_equal(
            ctx.population.get_array("objective_source"), expected
        )

    def test_partial_tell_scatter_preserves_noncontiguous_provenance(self):
        problem = _make_problem()
        algo = PymooAlgorithm(NSGA2(pop_size=N_POP), allow_partial_tell=True)
        ctx = _make_ctx(algo, problem, with_ids=True)
        ctx.candidate_id_allocator.allocate(N_POP)
        cand = algorithm_ask(algo, ctx, _DummyProvider())
        infill_ids = cand.get_array("id").copy()
        selected_idx = np.array([1, 3, 6, 9], dtype=np.intp)
        selected = cand.extract(selected_idx)
        selected.update_array("f", np.full((len(selected), 1), 0.0, dtype=np.float64))

        algorithm_tell(algo, ctx, selected, _DummyProvider())

        assert algo._infills is not None
        updated_ids = np.asarray(
            algo._pymoo_candidate_ids(algo._infills), dtype=np.int64
        )
        np.testing.assert_array_equal(
            updated_ids[selected_idx], selected.get_array("id")
        )
        unselected_idx = np.setdiff1d(np.arange(N_POP), selected_idx)
        np.testing.assert_array_equal(
            updated_ids[unselected_idx], infill_ids[unselected_idx]
        )

    def test_missing_survivor_provenance_is_rejected(self):
        problem = _make_problem()
        algo = PymooAlgorithm(NSGA2(pop_size=N_POP))
        ctx = _make_ctx(algo, problem, with_ids=True)
        ctx.candidate_id_allocator.allocate(N_POP)
        cand = algorithm_ask(algo, ctx, _DummyProvider())
        cand.update_array("f", np.zeros((len(cand), 1), dtype=np.float64))
        assert algo._infills is not None
        algo._infills.set(
            algo._candidate_id_attr, np.full(len(cand), -1, dtype=np.int64)
        )

        with pytest.raises(ConfigurationError, match="provenance"):
            algorithm_tell(algo, ctx, cand, _DummyProvider())


# ---------------------------------------------------------------------------
# End-to-end via minimize()
# ---------------------------------------------------------------------------


class TestPymooAlgorithmEndToEnd:
    def test_single_objective_ga_direct_strategy(self):
        result = minimize(
            lambda x: np.sum(x**2),
            dim=DIM,
            lb=[-5.0] * DIM,
            ub=[5.0] * DIM,
            algorithm=PymooAlgorithm(PymooGA(pop_size=N_POP)),
            surrogate="rbf",
            strategy=DirectStrategy(),
            max_fe=100,
            pop_size=N_POP,
            seed=0,
            verbose=False,
        )
        assert result.fe > 0
        assert np.isfinite(result.f).all()

    def test_multi_objective_nsga2(self):
        def zdt1(x):
            f1 = x[0]
            g = 1.0 + 9.0 * np.sum(x[1:]) / (len(x) - 1)
            f2 = g * (1.0 - np.sqrt(f1 / g))
            return np.array([f1, f2])

        result = minimize(
            zdt1,
            dim=10,
            lb=[0.0] * 10,
            ub=[1.0] * 10,
            n_obj=2,
            algorithm=PymooAlgorithm(NSGA2(pop_size=20)),
            surrogate="rbf",
            strategy=DirectStrategy(),
            max_fe=200,
            pop_size=20,
            seed=0,
            verbose=False,
        )
        assert result.fe > 0
        assert result.x is not None
        assert result.x.shape[-1] == 10

    def test_preselection_partial_tell_uses_policy_feedback(self):
        result = minimize(
            lambda x: np.sum(x**2),
            dim=DIM,
            lb=[-5.0] * DIM,
            ub=[5.0] * DIM,
            algorithm=PymooAlgorithm(PymooGA(pop_size=N_POP), allow_partial_tell=True),
            surrogate="rbf",
            strategy=PreSelectionStrategy(n_candidates=N_POP, n_select=N_POP - 3),
            max_fe=100,
            pop_size=N_POP,
            seed=0,
            verbose=False,
        )
        assert result.fe > 0
