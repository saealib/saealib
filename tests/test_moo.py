"""
Tests for multi-objective optimization support.

Tests cover:
- non_dominated_sort: basic ranking, NaN handling, all-non-dominated, all-dominated
- crowding_distance: boundary=inf, interior calculation, 2-point and 1-point edge cases
- crowding_distance_all_fronts: correct per-front assignment
- WeightedSumComparator: sort_population, compare_population, constraint handling
- NSGA2Comparator: sort_population, compare_population, cache, infeasible handling
- Problem: auto-selection of comparator (n_obj=1 / n_obj>1), custom injection
- MOO integration: 2-objective optimization with IndividualBasedStrategy
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest

from saealib import (
    GA,
    CrossoverBLXAlpha,
    DirectStrategy,
    GaussianKernel,
    IndividualBasedStrategy,
    LHSInitializer,
    MutationUniform,
    NonDominatedSorter,
    Optimizer,
    Problem,
    RBFSurrogate,
    SequentialSelection,
    Termination,
    TournamentSelection,
    TruncationSelection,
    WeightedSumComparator,
    crowding_distance,
    crowding_distance_all_fronts,
    max_fe,
    max_gen,
    non_dominated_sort,
    spea2_fitness,
)
from saealib.comparators import (
    Comparator,
    Dominator,
    EpsilonDominanceComparator,
    EpsilonDominator,
    HypervolumeComparator,
    NSGA2Comparator,
    NSGA3Comparator,
    ParetoComparator,
    ParetoDominator,
    RNSGA2Comparator,
    SingleObjectiveComparator,
    SPEA2Comparator,
)
from saealib.comparators.comparators import _normalize_objectives
from saealib.population import Population, PopulationAttribute
from saealib.utils.indicators import _non_dominated, hypervolume_contributions

logging.getLogger("saealib.surrogate.rbf").setLevel(logging.CRITICAL)

_default_dominator = ParetoDominator()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_pop(f_values: np.ndarray, cv_values: np.ndarray | None = None) -> Population:
    """
    Create a Population with 'f' and 'cv' attributes from arrays.

    Parameters
    ----------
    f_values : np.ndarray
        Objective matrix, shape (n, n_obj).
    cv_values : np.ndarray or None
        Constraint violations, shape (n,). Defaults to zeros.
    """
    n, n_obj = f_values.shape
    if cv_values is None:
        cv_values = np.zeros(n)
    attrs = [
        PopulationAttribute(name="f", dtype=np.float64, shape=(n_obj,)),
        PopulationAttribute(name="cv", dtype=np.float64, shape=(), default=0.0),
    ]
    pop = Population(attrs, init_capacity=n + 1)
    for i in range(n):
        pop.append(f=f_values[i], cv=float(cv_values[i]))
    return pop


def _make_spea2_pop(
    f_values: np.ndarray, comparator: SPEA2Comparator, problem: Problem
) -> Population:
    n, n_obj = f_values.shape
    attrs = [
        PopulationAttribute("f", np.float64, (n_obj,)),
        PopulationAttribute("cv", np.float64, (), default=0.0),
        *comparator.get_required_attrs(problem),
    ]
    population = Population(attrs, init_capacity=n + 1)
    for row in f_values:
        population.append(f=row, cv=0.0)
    return population


def _nsga3_tie_population_f() -> np.ndarray:
    """Create a five-point front with three exact duplicates in one niche.

    All points lie on the segment f1+f2=1, so no point dominates another and
    the whole set forms a single Pareto front. The 3 duplicates at (0.5, 0.5)
    guarantee ``_niche_count_select``'s ``rng.choice(candidates)`` branch is
    hit with more than one candidate at least once, exercising the tie-break.
    """
    return np.array(
        [
            [0.0, 1.0],
            [1.0, 0.0],
            [0.5, 0.5],
            [0.5, 0.5],
            [0.5, 0.5],
        ]
    )


def _nsga3_first_ctx(seed: int, reference_points: np.ndarray):
    """Build an ``Optimizer(seed=seed)`` wired with an unseeded NSGA3Comparator.

    Returns the context yielded immediately after ``RunStartEvent``, which is
    already past the point where ``Runner`` injects a spawned ``rng`` into an
    unseeded ``NSGA3Comparator``, so no generation needs to run to observe
    reproducibility.
    """
    dim = 2
    problem = Problem(
        func=lambda x: np.array([np.sum(x**2), np.sum((x - 2.0) ** 2)]),
        dim=dim,
        n_obj=2,
        direction=np.array([-1.0, -1.0]),
        lb=[0.0] * dim,
        ub=[2.0] * dim,
        comparator=NSGA3Comparator(reference_points, seed=None),
    )
    initializer = LHSInitializer(n_init_archive=10, n_init_population=8, seed=seed)
    algorithm = GA(
        crossover=CrossoverBLXAlpha(prob=0.7, alpha=0.4),
        mutation=MutationUniform(prob_var=0.3),
        parent_selection=SequentialSelection(),
        survivor_selection=TruncationSelection(),
    )
    opt = (
        Optimizer(problem, seed=seed)
        .set_initializer(initializer)
        .set_algorithm(algorithm)
        .set_termination(Termination(max_gen(0)))
        .set_strategy(DirectStrategy())
    )
    return next(opt.iterate())


# ===========================================================================
# non_dominated_sort Tests
# ===========================================================================
class TestNonDominatedSort:
    """Tests for the non_dominated_sort function."""

    def test_basic_two_fronts(self) -> None:
        """[0,0] dominates [1,1]; they form two separate fronts."""
        f = np.array([[0.0, 0.0], [1.0, 1.0]])
        ranks, fronts = non_dominated_sort(f)
        assert ranks[0] == 0
        assert ranks[1] == 1
        assert 0 in fronts[0]
        assert 1 in fronts[1]

    def test_all_non_dominated(self) -> None:
        """Points on the Pareto front are all in front 0."""
        # f1 trade-off with f2: each point differs on one objective
        f = np.array([[0.0, 3.0], [1.0, 2.0], [2.0, 1.0], [3.0, 0.0]])
        ranks, fronts = non_dominated_sort(f)
        assert np.all(ranks == 0)
        assert len(fronts) == 1
        assert sorted(fronts[0]) == [0, 1, 2, 3]

    def test_all_dominated_chain(self) -> None:
        """0 dominates 1 dominates 2: three separate fronts."""
        f = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        ranks, _fronts = non_dominated_sort(f)
        assert ranks[0] == 0
        assert ranks[1] == 1
        assert ranks[2] == 2

    def test_ranks_shape(self) -> None:
        rng = np.random.default_rng(0)
        f = rng.random((10, 2))
        ranks, _fronts = non_dominated_sort(f)
        assert ranks.shape == (10,)

    def test_nan_row_last_front(self) -> None:
        """NaN rows are placed in a sentinel front after all valid individuals."""
        f = np.array([[0.0, 0.0], [np.nan, np.nan], [1.0, 1.0]])
        ranks, _fronts = non_dominated_sort(f)
        assert ranks[0] == 0  # best valid point
        assert ranks[2] == 1  # dominated valid point
        # NaN point rank > all valid ranks
        assert ranks[1] > ranks[2]

    def test_single_point(self) -> None:
        f = np.array([[1.0, 2.0]])
        ranks, fronts = non_dominated_sort(f)
        assert ranks[0] == 0
        assert len(fronts) == 1

    def test_three_objectives(self) -> None:
        """Works for n_obj > 2."""
        f = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        ranks, _fronts = non_dominated_sort(f)
        assert ranks[0] == 0
        assert ranks[1] == 1

    # -----------------------------------------------------------------------
    # New tests for vectorised implementation (#89)
    # -----------------------------------------------------------------------

    @pytest.mark.parametrize("n,m", [(1, 1), (2, 1), (50, 2), (50, 5)])
    def test_equivalence_random(self, n: int, m: int) -> None:
        """Vectorised result matches a brute-force Pareto reference."""
        rng = np.random.default_rng(seed=n * 100 + m)
        f = rng.random((n, m))

        ranks, fronts = non_dominated_sort(f)

        # Brute-force reference using the public dominator.
        dom_count = np.zeros(n, int)
        dom_set: list[list[int]] = [[] for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i != j and _default_dominator.dominates(f[i], f[j]):
                    dom_set[i].append(j)
                    dom_count[j] += 1
        ref_ranks = np.full(n, -1, int)
        ref_fronts: list[list[int]] = [[]]
        for i in range(n):
            if dom_count[i] == 0:
                ref_ranks[i] = 0
                ref_fronts[0].append(i)
        k = 0
        while ref_fronts[k]:
            nxt: list[int] = []
            for i in ref_fronts[k]:
                for j in dom_set[i]:
                    dom_count[j] -= 1
                    if dom_count[j] == 0:
                        ref_ranks[j] = k + 1
                        nxt.append(j)
            ref_fronts.append(nxt)
            k += 1
        ref_fronts.pop()

        np.testing.assert_array_equal(ranks, ref_ranks)
        assert len(fronts) == len(ref_fronts)
        for f_new, f_ref in zip(fronts, ref_fronts):
            assert set(f_new) == set(f_ref)

    def test_direction_maximize(self) -> None:
        """With direction=+1 (maximize), [3,3] dominates [1,1]."""
        f = np.array([[3.0, 3.0], [1.0, 1.0]])
        direction = np.array([1.0, 1.0])
        ranks, fronts = non_dominated_sort(f, direction=direction)
        assert ranks[0] == 0
        assert ranks[1] == 1
        assert 0 in fronts[0]
        assert 1 in fronts[1]

    def test_all_nan_input(self) -> None:
        """When every row is NaN, all individuals become sentinel fronts."""
        f = np.array([[np.nan, np.nan], [np.nan, np.nan]])
        ranks, fronts = non_dominated_sort(f)
        # No valid front; both rows become individual sentinel fronts at rank 0.
        assert ranks[0] == 0
        assert ranks[1] == 0
        assert len(fronts) == 2
        assert fronts[0] == [0]
        assert fronts[1] == [1]

    def test_mixed_nan_multiple_nan_rows(self) -> None:
        """Each NaN row gets its own sentinel front; valid ranks are lower."""
        f = np.array(
            [
                [0.0, 0.0],
                [np.nan, np.nan],
                [1.0, 1.0],
                [np.nan, 2.0],
            ]
        )
        ranks, fronts = non_dominated_sort(f)
        # Valid individuals: 0 (front 0) and 2 (front 1)
        assert ranks[0] == 0
        assert ranks[2] == 1
        # Both NaN rows get rank == number of real fronts (2)
        assert ranks[1] == 2
        assert ranks[3] == 2
        # Each NaN row is its own front
        nan_fronts = fronts[2:]
        nan_indices = {f[0] for f in nan_fronts}
        assert nan_indices == {1, 3}
        for nf in nan_fronts:
            assert len(nf) == 1

    def test_all_equal_rows_single_front(self) -> None:
        """Identical rows are mutually non-dominating → all in front 0."""
        f = np.array([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]])
        ranks, fronts = non_dominated_sort(f)
        assert np.all(ranks == 0)
        assert len(fronts) == 1
        assert sorted(fronts[0]) == [0, 1, 2]


# ===========================================================================
# _non_dominated Tests (indicators.py, vectorised kernel parity)
# ===========================================================================
class TestNonDominatedIndicator:
    """Tests for _non_dominated (used internally by hypervolume)."""

    def test_parity_with_brute_force(self) -> None:
        """Vectorised _non_dominated returns same rows as brute-force reference."""
        rng = np.random.default_rng(42)
        f = rng.random((20, 3))

        result = _non_dominated(f)

        # Brute-force reference (minimisation)
        n = len(f)
        dominated = np.zeros(n, dtype=bool)
        for i in range(n):
            for j in range(n):
                if i != j and np.all(f[j] <= f[i]) and np.any(f[j] < f[i]):
                    dominated[i] = True
                    break
        expected = f[~dominated]

        # Compare as sets of tuples (row order may differ)
        assert {tuple(r) for r in result} == {tuple(r) for r in expected}

    def test_single_row_returned_unchanged(self) -> None:
        f = np.array([[0.5, 0.5]])
        result = _non_dominated(f)
        np.testing.assert_array_equal(result, f)

    def test_all_non_dominated(self) -> None:
        """Points on the Pareto front are all returned."""
        f = np.array([[0.0, 1.0], [1.0, 0.0]])
        result = _non_dominated(f)
        assert len(result) == 2

    def test_one_dominated(self) -> None:
        """Only the dominating point is returned."""
        f = np.array([[0.0, 0.0], [1.0, 1.0]])
        result = _non_dominated(f)
        assert len(result) == 1
        np.testing.assert_array_equal(result[0], [0.0, 0.0])


# ===========================================================================
# crowding_distance Tests
# ===========================================================================
class TestCrowdingDistance:
    """Tests for the crowding_distance function."""

    def test_boundary_points_are_infinite(self) -> None:
        """The two boundary solutions on each objective receive infinite distance."""
        f = np.array([[0.0, 3.0], [1.0, 2.0], [2.0, 1.0], [3.0, 0.0]])
        cd = crowding_distance(f)
        assert np.isinf(cd[0])
        assert np.isinf(cd[3])

    def test_two_points_both_infinite(self) -> None:
        """With only 2 points, both are boundary points → infinite distance."""
        f = np.array([[0.0, 1.0], [1.0, 0.0]])
        cd = crowding_distance(f)
        assert np.all(np.isinf(cd))

    def test_one_point_infinite(self) -> None:
        """Single-point front gets infinite crowding distance."""
        f = np.array([[1.0, 2.0]])
        cd = crowding_distance(f)
        assert np.isinf(cd[0])

    def test_interior_finite(self) -> None:
        """Interior points (not boundary) have finite crowding distance."""
        f = np.array([[0.0, 4.0], [1.0, 3.0], [2.0, 2.0], [3.0, 1.0], [4.0, 0.0]])
        cd = crowding_distance(f)
        # interior points: indices 1, 2, 3
        assert np.all(np.isfinite(cd[1:4]))

    def test_output_shape(self) -> None:
        rng = np.random.default_rng(0)
        f = rng.random((6, 2))
        cd = crowding_distance(f)
        assert cd.shape == (6,)

    def test_uniform_spacing_symmetric(self) -> None:
        """Uniformly spaced points have equal interior crowding distances."""
        # 5 points uniformly spaced on the line f1+f2=4, step 1
        f = np.array([[0.0, 4.0], [1.0, 3.0], [2.0, 2.0], [3.0, 1.0], [4.0, 0.0]])
        cd = crowding_distance(f)
        # all interior points should have the same distance
        interior = cd[1:4]
        assert np.allclose(interior, interior[0])


# ===========================================================================
# crowding_distance_all_fronts Tests
# ===========================================================================
class TestCrowdingDistanceAllFronts:
    """Tests for the crowding_distance_all_fronts function."""

    def test_output_shape(self) -> None:
        f = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        _, fronts = non_dominated_sort(f)
        cd = crowding_distance_all_fronts(f, fronts)
        assert cd.shape == (3,)

    def test_values_match_per_front(self) -> None:
        """Each front's CD should match crowding_distance computed on that front."""
        f = np.array([[0.0, 3.0], [1.5, 1.5], [3.0, 0.0], [2.0, 2.0]])
        _, fronts = non_dominated_sort(f)
        cd_all = crowding_distance_all_fronts(f, fronts)
        for front in fronts:
            if not front:
                continue
            idx = np.array(front)
            cd_front = crowding_distance(f[idx])
            np.testing.assert_array_almost_equal(cd_all[idx], cd_front)

    def test_empty_front_skipped(self) -> None:
        """An empty front list entry should not cause errors."""
        f = np.array([[0.0, 1.0], [1.0, 0.0]])
        fronts = [[0, 1], []]
        cd = crowding_distance_all_fronts(f, fronts)
        assert cd.shape == (2,)


# ===========================================================================
# WeightedSumComparator Tests
# ===========================================================================
class TestWeightedSumComparator:
    """Tests for WeightedSumComparator."""

    def test_sort_population_ascending_weighted_sum(self) -> None:
        """Sort by f @ weights descending (higher weighted sum = better)."""
        # weights = [-1, -1] → score = -(f1+f2); lower sum → higher score
        f = np.array([[1.0, 1.0], [0.5, 0.5], [2.0, 2.0]])
        pop = _make_pop(f)
        comp = WeightedSumComparator(direction=np.array([-1.0, -1.0]))
        order = comp.sort_population(pop)
        # sorted by weighted sum descending: idx=1 (sum=1), idx=0 (sum=2), idx=2 (sum=4)
        assert order[0] == 1
        assert order[2] == 2

    def test_sort_population_with_infeasible(self) -> None:
        """Infeasible (cv > eps) individuals come after all feasible ones."""
        f = np.array([[1.0, 1.0], [0.1, 0.1], [0.5, 0.5]])
        cv = np.array([0.0, 0.5, 0.0])  # idx=1 is infeasible
        pop = _make_pop(f, cv)
        comp = WeightedSumComparator(direction=np.array([-1.0, -1.0]))
        order = comp.sort_population(pop)
        # idx=1 (infeasible) should appear last
        assert order[-1] == 1

    def test_compare_population_a_better(self) -> None:
        f = np.array([[0.5, 0.5], [1.0, 1.0]])
        pop = _make_pop(f)
        comp = WeightedSumComparator(direction=np.array([-1.0, -1.0]))
        result = comp.compare_population(pop, 0, 1)
        assert result == -1  # a has lower sum → better

    def test_compare_population_b_better(self) -> None:
        f = np.array([[2.0, 2.0], [1.0, 1.0]])
        pop = _make_pop(f)
        comp = WeightedSumComparator(direction=np.array([-1.0, -1.0]))
        result = comp.compare_population(pop, 0, 1)
        assert result == 1  # b has lower sum → better

    def test_compare_population_equal(self) -> None:
        f = np.array([[1.0, 1.0], [1.0, 1.0]])
        pop = _make_pop(f)
        comp = WeightedSumComparator(direction=np.array([-1.0, -1.0]))
        result = comp.compare_population(pop, 0, 1)
        assert result == 0

    def test_compare_infeasible_vs_feasible(self) -> None:
        """Feasible solution is always preferred over infeasible."""
        f = np.array([[100.0, 100.0], [0.0, 0.0]])
        cv = np.array([0.5, 0.0])  # idx=0 infeasible, idx=1 feasible
        pop = _make_pop(f, cv)
        comp = WeightedSumComparator(direction=np.array([-1.0, -1.0]))
        result = comp.compare_population(pop, 0, 1)
        assert result == 1  # b (feasible) is better

    def test_single_objective_equivalent(self) -> None:
        """WeightedSumComparator with n_obj=1 behaves like SingleObjectiveComparator."""
        f = np.array([[2.0], [1.0], [3.0]])
        pop = _make_pop(f)
        comp = WeightedSumComparator(direction=np.array([-1.0]))
        order = comp.sort_population(pop)
        # ascending order of f: idx=1(1.0), idx=0(2.0), idx=2(3.0)
        assert order[0] == 1
        assert order[-1] == 2


# ===========================================================================
# SingleObjectiveComparator Tests
# ===========================================================================
class TestSingleObjectiveComparator:
    """Test direction-aware comparison and sorting."""

    def test_maximize_compare_and_sort_agree(self) -> None:
        """Comparison and sorting agree when maximizing."""
        f = np.array([[1.0], [3.0], [2.0]])
        pop = _make_pop(f)
        comp = SingleObjectiveComparator(direction=1.0)
        order = comp.sort_population(pop)
        assert order.tolist() == [1, 2, 0]  # descending: 3, 2, 1
        assert comp.compare(f[1], 0.0, f[0], 0.0) == -1  # 3.0 better than 1.0
        assert comp.compare(f[0], 0.0, f[1], 0.0) == 1
        assert comp.compare(f[1], 0.0, f[2], 0.0) == -1  # 3.0 better than 2.0
        assert comp.compare(f[0], 0.0, f[0], 0.0) == 0

    def test_minimize_regression(self) -> None:
        """direction=-1 behavior is unchanged: lower fitness is better."""
        comp = SingleObjectiveComparator(direction=-1.0)
        assert comp.compare(np.array([1.0]), 0.0, np.array([2.0]), 0.0) == -1
        assert comp.compare(np.array([2.0]), 0.0, np.array([1.0]), 0.0) == 1
        assert comp.compare(np.array([1.0]), 0.0, np.array([1.0]), 0.0) == 0

    def test_direction_none_defaults_to_minimize(self) -> None:
        """A standalone comparator with direction=None behaves as minimize."""
        comp = SingleObjectiveComparator()
        assert comp.direction is None
        assert comp.compare(np.array([1.0]), 0.0, np.array([2.0]), 0.0) == -1
        assert comp.compare(np.array([2.0]), 0.0, np.array([1.0]), 0.0) == 1

        f = np.array([[3.0], [1.0], [2.0]])
        pop = _make_pop(f)
        order = comp.sort_population(pop)
        assert order.tolist() == [1, 2, 0]  # ascending: 1, 2, 3


# ===========================================================================
# ParetoComparator Tests
# ===========================================================================
class TestParetoComparator:
    """Tests for ParetoComparator (non-dominated sorting, no crowding distance)."""

    def test_is_subclass_of_comparator(self) -> None:
        from saealib import Comparator

        assert issubclass(ParetoComparator, Comparator)

    def test_nsga2_is_subclass_of_pareto(self) -> None:
        assert issubclass(NSGA2Comparator, ParetoComparator)

    def test_sort_population_first_front_first(self) -> None:
        """Non-dominated solutions (front 0) appear before dominated ones."""
        f = np.array([[0.0, 3.0], [3.0, 0.0], [2.0, 2.0]])
        pop = _make_pop(f)
        comp = ParetoComparator()
        order = comp.sort_population(pop)
        assert set(order[:2]) == {0, 1}
        assert order[2] == 2

    def test_sort_population_infeasible_last(self) -> None:
        """Infeasible individuals appear after all feasible ones."""
        f = np.array([[10.0, 10.0], [0.0, 0.0], [1.0, 1.0]])
        cv = np.array([1.0, 0.0, 0.0])
        pop = _make_pop(f, cv)
        comp = ParetoComparator()
        order = comp.sort_population(pop)
        assert order[-1] == 0

    def test_sort_population_infeasible_by_ascending_cv(self) -> None:
        """Multiple infeasible individuals are sorted by ascending cv."""
        f = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
        cv = np.array([2.0, 0.5, 1.0])
        pop = _make_pop(f, cv)
        comp = ParetoComparator()
        order = comp.sort_population(pop)
        assert list(order) == [1, 2, 0]

    def test_sort_population_not_cached(self) -> None:
        """ParetoComparator does not write to the pareto_sort cache."""
        f = np.array([[0.0, 1.0], [1.0, 0.0]])
        pop = _make_pop(f)
        comp = ParetoComparator()
        comp.sort_population(pop)
        assert pop.get_cache("pareto_sort") is None

    def test_sort_population_nan_last_among_feasible(self) -> None:
        """NaN objective rows are placed last within the feasible block."""
        f = np.array([[0.0, 0.0], [np.nan, np.nan], [1.0, 1.0]])
        pop = _make_pop(f)
        comp = ParetoComparator()
        order = comp.sort_population(pop)
        assert order[0] == 0
        assert order[-1] == 1

    def test_sort_population_output_is_int_array(self) -> None:
        f = np.array([[0.0, 1.0], [1.0, 0.0], [0.5, 0.5]])
        pop = _make_pop(f)
        comp = ParetoComparator()
        order = comp.sort_population(pop)
        assert np.issubdtype(order.dtype, np.integer)

    def test_compare_population_a_dominates_b(self) -> None:
        f = np.array([[0.0, 0.0], [1.0, 1.0]])
        pop = _make_pop(f)
        comp = ParetoComparator()
        assert comp.compare_population(pop, 0, 1) == -1

    def test_compare_population_b_dominates_a(self) -> None:
        f = np.array([[1.0, 1.0], [0.0, 0.0]])
        pop = _make_pop(f)
        comp = ParetoComparator()
        assert comp.compare_population(pop, 0, 1) == 1

    def test_compare_population_non_dominated(self) -> None:
        f = np.array([[0.0, 1.0], [1.0, 0.0]])
        pop = _make_pop(f)
        comp = ParetoComparator()
        assert comp.compare_population(pop, 0, 1) == 0

    def test_compare_infeasible_vs_feasible(self) -> None:
        """Feasible always beats infeasible regardless of objectives."""
        f = np.array([[0.0, 0.0], [100.0, 100.0]])
        cv = np.array([1.0, 0.0])
        pop = _make_pop(f, cv)
        comp = ParetoComparator()
        assert comp.compare_population(pop, 0, 1) == 1

    def test_compare_both_infeasible_lower_cv_wins(self) -> None:
        f = np.array([[0.0, 0.0], [0.0, 0.0]])
        cv = np.array([0.5, 2.0])
        pop = _make_pop(f, cv)
        comp = ParetoComparator()
        assert comp.compare_population(pop, 0, 1) == -1

    def test_weights_direction_affects_ranking(self) -> None:
        """direction sign determines optimization direction."""
        f = np.array([[3.0, 3.0], [1.0, 1.0]])
        pop = _make_pop(f)
        comp_min = ParetoComparator(direction=np.array([-1.0, -1.0]))
        comp_max = ParetoComparator(direction=np.array([1.0, 1.0]))
        assert comp_min.sort_population(pop)[0] == 1  # [1,1] wins under minimization
        assert comp_max.sort_population(pop)[0] == 0  # [3,3] wins under maximization

    def test_direction_interface_stored_as_attribute(self) -> None:
        """direction= kwarg is stored as comparator.direction attribute."""
        d = np.array([-1.0, 1.0])
        comp = ParetoComparator(direction=d)
        np.testing.assert_array_equal(comp.direction, d)

    def test_no_direction_defaults_to_none(self) -> None:
        """Without direction, comparator.direction is None (all-minimize)."""
        comp = ParetoComparator()
        assert comp.direction is None


# ===========================================================================
# NSGA2Comparator Tests
# ===========================================================================
class TestNSGA2Comparator:
    """Tests for NSGA2Comparator."""

    def test_sort_population_first_front_first(self) -> None:
        """Non-dominated solutions (front 0) appear before dominated ones."""
        # f[0] and f[1] are non-dominated; f[2] is dominated by both
        f = np.array([[0.0, 3.0], [3.0, 0.0], [2.0, 2.0]])
        pop = _make_pop(f)
        comp = NSGA2Comparator()
        order = comp.sort_population(pop)
        # idx 0 and 1 must appear before idx 2
        assert set(order[:2]) == {0, 1}
        assert order[2] == 2

    def test_sort_population_infeasible_last(self) -> None:
        """Infeasible individuals appear after all feasible ones."""
        f = np.array([[10.0, 10.0], [0.0, 0.0], [1.0, 1.0]])
        cv = np.array([1.0, 0.0, 0.0])  # idx=0 infeasible
        pop = _make_pop(f, cv)
        comp = NSGA2Comparator()
        order = comp.sort_population(pop)
        assert order[-1] == 0

    def test_sort_population_result_cached(self) -> None:
        """Second call returns the cached result without recomputing."""
        f = np.array([[0.0, 1.0], [1.0, 0.0], [0.5, 0.5]])
        pop = _make_pop(f)
        comp = NSGA2Comparator()
        order1 = comp.sort_population(pop)
        order2 = comp.sort_population(pop)
        assert order1 is order2  # same object from cache

    def test_sort_population_cache_invalidated_on_mutation(self) -> None:
        """Cache is invalidated when the population is modified."""
        f = np.array([[0.0, 1.0], [1.0, 0.0]])
        pop = _make_pop(f)
        comp = NSGA2Comparator()
        order1 = comp.sort_population(pop)
        # mutate population
        pop.f = np.array([[0.5, 0.5], [0.5, 0.5]])
        order2 = comp.sort_population(pop)
        # objects should not be the same (cache was invalidated)
        assert order1 is not order2

    def test_sort_population_cache_invalidated_on_dominator_replace(self) -> None:
        """Replacing dominator mid-run causes ranks to be recomputed."""
        f = np.array([[0.0, 1.0], [1.0, 0.0], [0.5, 0.5]])
        pop = _make_pop(f)
        comp = NSGA2Comparator()
        order1 = comp.sort_population(pop)
        comp._dominator = EpsilonDominator(eps=10.0)  # coarse boxes → all same front
        order2 = comp.sort_population(pop)
        assert order1 is not order2

    def test_sort_population_cache_invalidated_on_direction_change(self) -> None:
        """Changing direction mid-run causes ranks to be recomputed."""
        f = np.array([[0.0, 1.0], [1.0, 0.0], [0.5, 0.5]])
        pop = _make_pop(f)
        comp = NSGA2Comparator(direction=np.array([1.0, 1.0]))
        order1 = comp.sort_population(pop)
        comp.direction = np.array([-1.0, -1.0])
        order2 = comp.sort_population(pop)
        assert order1 is not order2

    def test_compare_population_a_dominates_b(self) -> None:
        f = np.array([[0.0, 0.0], [1.0, 1.0]])
        pop = _make_pop(f)
        comp = NSGA2Comparator()
        result = comp.compare_population(pop, 0, 1)
        assert result == -1  # a dominates b

    def test_compare_population_b_dominates_a(self) -> None:
        f = np.array([[1.0, 1.0], [0.0, 0.0]])
        pop = _make_pop(f)
        comp = NSGA2Comparator()
        result = comp.compare_population(pop, 0, 1)
        assert result == 1  # b dominates a

    def test_compare_population_non_dominated(self) -> None:
        f = np.array([[0.0, 1.0], [1.0, 0.0]])
        pop = _make_pop(f)
        comp = NSGA2Comparator()
        result = comp.compare_population(pop, 0, 1)
        assert result == 0  # mutually non-dominated

    def test_compare_infeasible_vs_feasible(self) -> None:
        f = np.array([[0.0, 0.0], [0.0, 0.0]])
        cv = np.array([1.0, 0.0])  # a infeasible, b feasible
        pop = _make_pop(f, cv)
        comp = NSGA2Comparator()
        result = comp.compare_population(pop, 0, 1)
        assert result == 1  # b (feasible) is better

    def test_output_is_int_array(self) -> None:
        f = np.array([[0.0, 1.0], [1.0, 0.0], [0.5, 0.5]])
        pop = _make_pop(f)
        comp = NSGA2Comparator()
        order = comp.sort_population(pop)
        assert order.dtype == np.intp or np.issubdtype(order.dtype, np.integer)

    def test_weights_direction_affects_sorting(self) -> None:
        """direction sign determines optimization direction."""
        f = np.array([[3.0, 3.0], [1.0, 1.0]])
        comp_min = NSGA2Comparator(direction=np.array([-1.0, -1.0]))
        comp_max = NSGA2Comparator(direction=np.array([1.0, 1.0]))
        assert (
            comp_min.sort_population(_make_pop(f))[0] == 1
        )  # [1,1] wins under minimization
        assert (
            comp_max.sort_population(_make_pop(f))[0] == 0
        )  # [3,3] wins under maximization

    def test_compare_population_same_front_higher_cd_wins(self) -> None:
        # 3 points on the same front; boundary points (idx 0, 1) get cd=inf,
        # interior point (idx 2) gets finite cd → boundary beats interior.
        f = np.array([[0.0, 1.0], [1.0, 0.0], [0.5, 0.5]])
        pop = _make_pop(f)
        comp = NSGA2Comparator()
        assert comp.compare_population(pop, 0, 2) == -1  # boundary beats interior
        assert comp.compare_population(pop, 1, 2) == -1
        assert comp.compare_population(pop, 2, 0) == 1

    def test_compare_population_lower_rank_beats_higher_cd(self) -> None:
        # idx=2 is on rank 1; idx=0 and idx=1 are on rank 0 regardless of cd.
        f = np.array([[0.0, 0.0], [0.1, 0.1], [5.0, 5.0]])
        pop = _make_pop(f)
        comp = NSGA2Comparator()
        assert comp.compare_population(pop, 0, 2) == -1
        assert comp.compare_population(pop, 1, 2) == -1

    def test_compare_population_non_dominated_two_points_tie(self) -> None:
        # Two-point front: both are boundary points with cd=inf → tie.
        f = np.array([[0.0, 1.0], [1.0, 0.0]])
        pop = _make_pop(f)
        comp = NSGA2Comparator()
        assert comp.compare_population(pop, 0, 1) == 0


# ===========================================================================
# Comparator Default Ranking Hooks Tests
# ===========================================================================
class TestComparatorDefaultRankingHooks:
    @pytest.mark.parametrize(
        ("comparator_factory", "population_factory"),
        [
            pytest.param(
                lambda: SingleObjectiveComparator(),
                lambda: _make_pop(np.array([[3.0], [1.0], [2.0]])),
                id="single-objective",
            ),
            pytest.param(
                lambda: WeightedSumComparator(direction=np.array([-1.0, -1.0])),
                lambda: _make_pop(np.array([[0.0, 1.0], [1.0, 0.0], [0.5, 0.5]])),
                id="weighted-sum",
            ),
            pytest.param(
                lambda: ParetoComparator(),
                lambda: _make_pop(np.array([[0.0, 1.0], [1.0, 0.0], [0.5, 0.5]])),
                id="pareto",
            ),
            pytest.param(
                lambda: NSGA2Comparator(),
                lambda: _make_pop(np.array([[0.0, 1.0], [1.0, 0.0], [0.5, 0.5]])),
                id="nsga2",
            ),
            pytest.param(
                lambda: HypervolumeComparator(),
                lambda: _make_pop(np.array([[0.0, 1.0], [1.0, 0.0], [0.5, 0.5]])),
                id="hypervolume",
            ),
            pytest.param(
                lambda: EpsilonDominanceComparator(eps=0.5),
                lambda: _make_pop(np.array([[0.0, 1.0], [1.0, 0.0], [0.5, 0.5]])),
                id="epsilon-dominance",
            ),
            pytest.param(
                lambda: NSGA3Comparator(
                    np.array([[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]]), seed=0
                ),
                lambda: _make_pop(np.array([[0.0, 1.0], [1.0, 0.0], [0.5, 0.5]])),
                id="nsga3",
            ),
            pytest.param(
                lambda: RNSGA2Comparator(
                    np.array([[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]])
                ),
                lambda: _make_pop(np.array([[0.0, 1.0], [1.0, 0.0], [0.5, 0.5]])),
                id="rnsga2",
            ),
        ],
    )
    def test_default_ranking_hooks_are_noop_for_all_comparators(
        self,
        comparator_factory: Callable[[], Comparator],
        population_factory: Callable[[], Population],
    ) -> None:
        comparator = comparator_factory()
        population = population_factory()
        problem = Problem(
            func=lambda x: np.array([np.sum(x), np.sum(x)]),
            dim=2,
            n_obj=2,
            direction=np.array([-1.0, -1.0]),
            lb=[0.0, 0.0],
            ub=[1.0, 1.0],
        )
        f_before = population.get_array("f").copy()
        cv_before = population.get_array("cv").copy()

        assert comparator.get_required_attrs(problem) == []
        assert comparator.prepare_population(population) is None
        np.testing.assert_array_equal(population.get_array("f"), f_before)
        np.testing.assert_array_equal(population.get_array("cv"), cv_before)

        ranked = comparator.rank_population(population)
        sorted_direct = comparator.sort_population(population)
        assert np.array_equal(ranked, sorted_direct)


# ===========================================================================
# Problem Comparator Auto-selection Tests
# ===========================================================================
class TestProblemComparatorAutoSelection:
    """Tests for Problem's automatic comparator selection."""

    def test_single_objective_uses_single_objective_comparator(self) -> None:
        p = Problem(
            func=lambda x: np.sum(x**2),
            dim=2,
            n_obj=1,
            direction=np.array([-1.0]),
            lb=[-5.0, -5.0],
            ub=[5.0, 5.0],
        )
        assert isinstance(p.comparator, SingleObjectiveComparator)

    def test_multi_objective_uses_pareto_comparator(self) -> None:
        p = Problem(
            func=lambda x: np.array([np.sum(x**2), np.sum((x - 1) ** 2)]),
            dim=2,
            n_obj=2,
            direction=np.array([-1.0, -1.0]),
            lb=[-5.0, -5.0],
            ub=[5.0, 5.0],
        )
        assert isinstance(p.comparator, NSGA2Comparator)

    def test_custom_comparator_overrides_auto_selection(self) -> None:
        custom = WeightedSumComparator(direction=np.array([-1.0, -1.0]))
        p = Problem(
            func=lambda x: np.array([np.sum(x**2), np.sum((x - 1) ** 2)]),
            dim=2,
            n_obj=2,
            direction=np.array([-1.0, -1.0]),
            lb=[-5.0, -5.0],
            ub=[5.0, 5.0],
            comparator=custom,
        )
        assert p.comparator is custom

    def test_custom_comparator_with_none_direction_gets_injected(self) -> None:
        """A comparator constructed with direction=None inherits Problem's direction."""
        comp = SingleObjectiveComparator()
        assert comp.direction is None
        p = Problem(
            func=lambda x: np.array([np.sum(x**2)]),
            dim=2,
            n_obj=1,
            direction=np.array([-1.0]),
            lb=[-5.0, -5.0],
            ub=[5.0, 5.0],
            comparator=comp,
        )
        assert p.comparator is comp
        direction = p.comparator.direction
        assert direction is not None
        assert direction[0] == -1.0

    def test_problem_evaluate_returns_n_obj_values(self) -> None:
        p = Problem(
            func=lambda x: np.array([np.sum(x**2), np.sum((x - 1) ** 2)]),
            dim=2,
            n_obj=2,
            direction=np.array([-1.0, -1.0]),
            lb=[-5.0, -5.0],
            ub=[5.0, 5.0],
        )
        result = p.evaluate(np.array([0.5, 0.5]))
        assert result.shape == (2,)


# ===========================================================================
# MOO Integration Test
# ===========================================================================
class TestMOOIntegration:
    """2-objective integration test using IndividualBasedStrategy."""

    def test_biobj_optimization_converges(self) -> None:
        """
        2-objective optimization on a simple bi-sphere problem.

        f1(x) = sum(x^2)      (minimize, weight=-1)
        f2(x) = sum((x-2)^2)  (minimize, weight=-1)

        Pareto front: x_i in [0, 2] for each dimension.
        """
        dim = 3
        seed = 42

        def bisphere(x: np.ndarray) -> np.ndarray:
            return np.array([np.sum(x**2), np.sum((x - 2.0) ** 2)])

        problem = Problem(
            func=bisphere,
            dim=dim,
            n_obj=2,
            direction=np.array([-1.0, -1.0]),
            lb=[0.0] * dim,
            ub=[2.0] * dim,
        )
        initializer = LHSInitializer(
            n_init_archive=5 * dim,
            n_init_population=4 * dim,
            seed=seed,
        )
        algorithm = GA(
            crossover=CrossoverBLXAlpha(prob=0.7, alpha=0.4),
            mutation=MutationUniform(prob_var=0.3),
            parent_selection=SequentialSelection(),
            survivor_selection=TruncationSelection(),
        )
        termination = Termination(max_fe(80 * dim))
        surrogate = RBFSurrogate(GaussianKernel())
        strategy = IndividualBasedStrategy(evaluation_ratio=0.1)

        opt = (
            Optimizer(problem)
            .set_initializer(initializer)
            .set_algorithm(algorithm)
            .set_termination(termination)
            .set_surrogate(surrogate, n_neighbors=15)
            .set_strategy(strategy)
        )
        ctx = opt.run()

        assert ctx is not None

        # Archive has solutions
        archive_f = ctx.archive.get("f")
        assert archive_f is not None
        assert len(archive_f) > 0
        assert archive_f.shape[1] == 2

        # All objective values are finite
        assert np.all(np.isfinite(archive_f))

        # At least some non-dominated solutions exist (front 0 is non-empty)
        _ranks, fronts = non_dominated_sort(archive_f)
        assert len(fronts[0]) >= 1

        # Best f1 and f2 are reasonable (within the feasible region bounds)
        best_f1 = archive_f[:, 0].min()
        best_f2 = archive_f[:, 1].min()
        assert best_f1 >= 0.0
        assert best_f2 >= 0.0


# ===========================================================================
# NonDominatedSorter injection tests (#89)
# ===========================================================================
class TestNonDominatedSorterInjection:
    """Tests for the NonDominatedSorter injection seam in Pareto-based comparators."""

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------
    def _make_spy_sorter(self) -> tuple[list[tuple], NonDominatedSorter]:
        """Return (call_log, spy_sorter).

        The spy delegates to non_dominated_sort and records every call as
        (f_arg, direction_arg) in call_log.
        """
        call_log: list[tuple] = []

        def spy(
            f: np.ndarray,
            direction: np.ndarray | None = None,
            *,
            dominator=None,
        ) -> tuple[np.ndarray, list[list[int]]]:
            call_log.append((f, direction))
            return non_dominated_sort(f, direction)

        return call_log, spy

    # -----------------------------------------------------------------------
    # 1. Spy: verifies the injected sorter is called with the right arguments
    # -----------------------------------------------------------------------
    def test_pareto_comparator_calls_injected_sorter(self) -> None:
        """ParetoComparator.sort_population invokes the injected sorter."""
        call_log, spy = self._make_spy_sorter()
        weights = np.array([-1.0, -1.0])
        comp = ParetoComparator(direction=weights, sorter=spy)

        f = np.array([[0.0, 1.0], [1.0, 0.0], [0.5, 0.5]])
        pop = _make_pop(f)
        comp.sort_population(pop)

        assert len(call_log) == 1
        f_passed, dir_passed = call_log[0]
        # The comparator passes the feasible-subset objective matrix.
        assert f_passed.shape == (3, 2)
        # direction is passed directly as given
        np.testing.assert_array_equal(dir_passed, weights)

    def test_nsga2_comparator_calls_injected_sorter(self) -> None:
        """NSGA2Comparator.sort_population invokes the injected sorter."""
        call_log, spy = self._make_spy_sorter()
        weights = np.array([-1.0, -1.0])
        comp = NSGA2Comparator(direction=weights, sorter=spy)

        f = np.array([[0.0, 1.0], [1.0, 0.0], [0.5, 0.5]])
        pop = _make_pop(f)
        comp.sort_population(pop)

        assert len(call_log) == 1
        f_passed, dir_passed = call_log[0]
        assert f_passed.shape == (3, 2)
        np.testing.assert_array_equal(dir_passed, weights)

    def test_spy_receives_feasible_subset_only(self) -> None:
        """The sorter only receives the feasible-individual rows."""
        call_log, spy = self._make_spy_sorter()
        comp = ParetoComparator(sorter=spy)

        # idx=2 is infeasible
        f = np.array([[0.0, 1.0], [1.0, 0.0], [99.0, 99.0]])
        cv = np.array([0.0, 0.0, 5.0])
        pop = _make_pop(f, cv)
        comp.sort_population(pop)

        assert len(call_log) == 1
        f_passed, _ = call_log[0]
        # Only the 2 feasible rows should be passed to the sorter.
        assert f_passed.shape == (2, 2)

    def test_nsga2_spy_receives_feasible_subset_only(self) -> None:
        """NSGA2Comparator passes only feasible rows to the sorter."""
        call_log, spy = self._make_spy_sorter()
        comp = NSGA2Comparator(sorter=spy)

        f = np.array([[0.0, 1.0], [1.0, 0.0], [99.0, 99.0]])
        cv = np.array([0.0, 0.0, 5.0])
        pop = _make_pop(f, cv)
        comp.sort_population(pop)

        assert len(call_log) == 1
        f_passed, _ = call_log[0]
        assert f_passed.shape == (2, 2)

    # -----------------------------------------------------------------------
    # 2. Alternate sorter: honors the ranks/fronts returned by the injected sorter
    # -----------------------------------------------------------------------
    def test_pareto_comparator_honors_injected_ranks(self) -> None:
        """ParetoComparator.sort_population respects ranks from the injected sorter.

        We inject a sorter that inverts the natural ranking so that the
        *worst* objective point is declared front-0.
        """
        f = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        # Natural order: idx 0 (front-0) → idx 1 (front-1) → idx 2 (front-2).
        # Inverted sorter: rank 0 → idx 2, rank 1 → idx 1, rank 2 → idx 0.

        def inverted_sorter(
            f: np.ndarray,
            direction: np.ndarray | None = None,
            *,
            dominator=None,
        ) -> tuple[np.ndarray, list[list[int]]]:
            n = len(f)
            # Assign ranks in reverse order.
            ranks = np.arange(n - 1, -1, -1, dtype=int)
            fronts = [[i] for i in range(n - 1, -1, -1)]
            return ranks, fronts

        comp = ParetoComparator(sorter=inverted_sorter)
        pop = _make_pop(f)
        order = comp.sort_population(pop)

        # Inverted: global index 2 should be first (rank 0 for its local index 2).
        assert order[0] == 2

    def test_nsga2_comparator_honors_injected_ranks(self) -> None:
        """NSGA2Comparator.sort_population respects ranks from the injected sorter."""
        f = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])

        def inverted_sorter(
            f: np.ndarray,
            direction: np.ndarray | None = None,
            *,
            dominator=None,
        ) -> tuple[np.ndarray, list[list[int]]]:
            n = len(f)
            ranks = np.arange(n - 1, -1, -1, dtype=int)
            fronts = [[i] for i in range(n - 1, -1, -1)]
            return ranks, fronts

        comp = NSGA2Comparator(sorter=inverted_sorter)
        pop = _make_pop(f)
        order = comp.sort_population(pop)

        # Inverted: global index 2 should appear first.
        assert order[0] == 2

    # -----------------------------------------------------------------------
    # 3. Default behavior and public API
    # -----------------------------------------------------------------------
    def test_default_sorter_is_non_dominated_sort(self) -> None:
        """When sorter is not specified, non_dominated_sort is used (default)."""
        comp = ParetoComparator()
        assert comp.sorter is non_dominated_sort

    def test_nsga2_default_sorter_is_non_dominated_sort(self) -> None:
        """NSGA2Comparator default sorter is non_dominated_sort."""
        comp = NSGA2Comparator()
        assert comp.sorter is non_dominated_sort

    def test_non_dominated_sorter_importable_from_saealib(self) -> None:
        """NonDominatedSorter is importable from the top-level saealib package."""
        # Import is done at module level; this test confirms it succeeds.
        assert NonDominatedSorter is not None

    def test_non_dominated_sort_satisfies_protocol(self) -> None:
        """non_dominated_sort can be passed wherever NonDominatedSorter is expected."""
        # Passing the free function as the sorter kwarg should not raise.
        comp = ParetoComparator(sorter=non_dominated_sort)
        f = np.array([[0.0, 1.0], [1.0, 0.0]])
        pop = _make_pop(f)
        order = comp.sort_population(pop)
        assert len(order) == 2

    # -----------------------------------------------------------------------
    # Cache behavior must remain intact for NSGA2Comparator
    # -----------------------------------------------------------------------
    def test_nsga2_injected_sorter_cached_result(self) -> None:
        """Cache still works when a custom sorter is injected."""
        call_log, spy = self._make_spy_sorter()
        comp = NSGA2Comparator(sorter=spy)

        f = np.array([[0.0, 1.0], [1.0, 0.0]])
        pop = _make_pop(f)

        order1 = comp.sort_population(pop)
        order2 = comp.sort_population(pop)

        # Second call should use cache; spy called only once.
        assert len(call_log) == 1
        assert order1 is order2


# ===========================================================================
# Dominator abstraction tests (#89)
# ===========================================================================
class TestParetoDominator:
    """Tests for ParetoDominator and the Dominator ABC seam."""

    # -----------------------------------------------------------------------
    # 1. dominates() behavior
    # -----------------------------------------------------------------------
    def test_dominates_parity_random_pairs(self) -> None:
        """ParetoDominator.dominates matches the Pareto definition."""
        rng = np.random.default_rng(0)
        dom = ParetoDominator()
        for _ in range(200):
            fa = rng.random(3)
            fb = rng.random(3)
            expected = np.all(fa <= fb) and np.any(fa < fb)
            assert dom.dominates(fa, fb) == expected, (fa, fb)

    def test_dominates_nan_in_fa_returns_false(self) -> None:
        """NaN in fa never dominates."""
        dom = ParetoDominator()
        fa = np.array([np.nan, 0.0])
        fb = np.array([1.0, 1.0])
        assert not dom.dominates(fa, fb)

    def test_dominates_nan_in_fb_is_infinitely_bad(self) -> None:
        """NaN in fb is replaced with +inf before the batch comparison."""
        dom = ParetoDominator()
        fa = np.array([0.0, 0.0])
        fb = np.array([np.nan, 1.0])
        assert dom.dominates(fa, fb)

    def test_dominates_direction_maximize(self) -> None:
        """With direction=+1, larger values are better."""
        dom = ParetoDominator()
        direction = np.array([1.0, 1.0])
        fa = np.array([3.0, 3.0])
        fb = np.array([1.0, 1.0])
        assert dom.dominates(fa, fb, direction)
        assert not dom.dominates(fb, fa, direction)

    def test_dominates_direction_minimize(self) -> None:
        """With direction=-1 (explicit minimize), lower values are better."""
        dom = ParetoDominator()
        direction = np.array([-1.0, -1.0])
        fa = np.array([1.0, 1.0])
        fb = np.array([3.0, 3.0])
        assert dom.dominates(fa, fb, direction)

    # -----------------------------------------------------------------------
    # 2. dominance_matrix() parity with brute-force and scalar↔batched
    # -----------------------------------------------------------------------
    @pytest.mark.parametrize("n,m", [(3, 2), (10, 3), (20, 2)])
    def test_dominance_matrix_brute_force_parity(self, n: int, m: int) -> None:
        """dominance_matrix matches a brute-force reference built with dominates."""
        rng = np.random.default_rng(seed=n * 10 + m)
        f = rng.random((n, m))
        dom = ParetoDominator()
        mat = dom.dominance_matrix(f)
        for i in range(n):
            for j in range(n):
                expected = dom.dominates(f[i], f[j])
                assert mat[i, j] == expected, f"Mismatch at ({i},{j})"

    def test_dominance_matrix_scalar_batched_consistency(self) -> None:
        """Every (i,j) entry of dominance_matrix equals dominates(f[i], f[j])."""
        rng = np.random.default_rng(42)
        f = rng.random((8, 2))
        dom = ParetoDominator()
        mat = dom.dominance_matrix(f)
        n = len(f)
        for i in range(n):
            for j in range(n):
                assert mat[i, j] == dom.dominates(f[i], f[j])

    # -----------------------------------------------------------------------
    # 3. Custom Dominator injection: compare() and sort_population() both honor it
    # -----------------------------------------------------------------------
    def test_custom_dominator_affects_compare(self) -> None:
        """Injecting a custom Dominator changes compare() behaviour."""

        class AllDominatesDominator(Dominator):
            """Toy: every solution dominates every other (a > b always)."""

            def dominance_matrix(
                self,
                f: np.ndarray,
                direction: np.ndarray | None = None,
            ) -> np.ndarray:
                n = len(f)
                # All off-diagonal True → every pair mutually dominated.
                return np.ones((n, n), dtype=bool) & ~np.eye(n, dtype=bool)

            def dominates_many(self, fa, f_matrix, direction=None):
                dominates = np.ones(len(f_matrix), dtype=bool)
                return dominates, dominates.copy()

        comp = ParetoComparator(dominator=AllDominatesDominator())
        # Under AllDominatesDominator, fa dominates fb → compare returns -1.
        fa = np.array([5.0, 5.0])
        fb = np.array([0.0, 0.0])
        # Both would dominate each other; first check wins → -1.
        result = comp.compare(fa, 0.0, fb, 0.0)
        assert result == -1

    def test_custom_dominator_affects_sort_population(self) -> None:
        """Injecting a custom Dominator changes sort_population() output."""

        class ReverseParetoDominator(Dominator):
            """Toy: transpose the normal Pareto matrix (worse = better)."""

            def dominance_matrix(
                self,
                f: np.ndarray,
                direction: np.ndarray | None = None,
            ) -> np.ndarray:
                # Standard Pareto matrix, then transpose so dominated → dominator.
                _dom = ParetoDominator()
                return _dom.dominance_matrix(f, direction).T

            def dominates_many(self, fa, f_matrix, direction=None):
                fa_dominates, dominates_fa = ParetoDominator().dominates_many(
                    fa, f_matrix, direction
                )
                return dominates_fa, fa_dominates

        f = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        pop = _make_pop(f)
        comp = ParetoComparator(dominator=ReverseParetoDominator())
        order = comp.sort_population(pop)
        # Under reversed dominance, [2,2] becomes "best" (front-0).
        assert order[0] == 2

    def test_custom_dominator_nsga2_sort_population(self) -> None:
        """Injecting a custom Dominator into NSGA2Comparator also takes effect."""

        class ReverseParetoDominator(Dominator):
            def dominance_matrix(
                self,
                f: np.ndarray,
                direction: np.ndarray | None = None,
            ) -> np.ndarray:
                return ParetoDominator().dominance_matrix(f, direction).T

            def dominates_many(self, fa, f_matrix, direction=None):
                fa_dominates, dominates_fa = ParetoDominator().dominates_many(
                    fa, f_matrix, direction
                )
                return dominates_fa, fa_dominates

        f = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        pop = _make_pop(f)
        comp = NSGA2Comparator(dominator=ReverseParetoDominator())
        order = comp.sort_population(pop)
        assert order[0] == 2

    # -----------------------------------------------------------------------
    # 4. non_dominated_sort(..., dominator=ParetoDominator()) == default call
    # -----------------------------------------------------------------------
    @pytest.mark.parametrize("n,m", [(1, 2), (5, 2), (20, 3)])
    def test_non_dominated_sort_explicit_pareto_dominator_equals_default(
        self, n: int, m: int
    ) -> None:
        """dominator=ParetoDominator() yields identical result to the default."""
        rng = np.random.default_rng(seed=n * 7 + m)
        f = rng.random((n, m))
        ranks_default, fronts_default = non_dominated_sort(f)
        ranks_explicit, fronts_explicit = non_dominated_sort(
            f, dominator=ParetoDominator()
        )
        np.testing.assert_array_equal(ranks_default, ranks_explicit)
        assert len(fronts_default) == len(fronts_explicit)
        for fd, fe in zip(fronts_default, fronts_explicit):
            assert set(fd) == set(fe)

    def test_non_dominated_sort_explicit_dominator_with_direction(self) -> None:
        """dominator kwarg also works correctly when direction is passed."""
        f = np.array([[3.0, 3.0], [1.0, 1.0]])
        direction = np.array([1.0, 1.0])  # maximize
        ranks_default, _ = non_dominated_sort(f, direction=direction)
        ranks_explicit, _ = non_dominated_sort(
            f, direction=direction, dominator=ParetoDominator()
        )
        np.testing.assert_array_equal(ranks_default, ranks_explicit)

    # -----------------------------------------------------------------------
    # 5. Public API: Dominator and ParetoDominator importable from saealib
    # -----------------------------------------------------------------------
    def test_dominator_importable_from_saealib(self) -> None:
        from saealib import Dominator as ImportedDominator
        from saealib import ParetoDominator as ImportedParetoDominator

        assert issubclass(ParetoDominator, Dominator)
        assert issubclass(ImportedParetoDominator, ImportedDominator)

    def test_pareto_comparator_dominator_property(self) -> None:
        """ParetoComparator exposes a .dominator property."""
        comp = ParetoComparator()
        assert isinstance(comp.dominator, ParetoDominator)

    def test_nsga2_comparator_dominator_property(self) -> None:
        """NSGA2Comparator exposes a .dominator property (inherited)."""
        comp = NSGA2Comparator()
        assert isinstance(comp.dominator, ParetoDominator)

    def test_custom_dominator_stored_on_property(self) -> None:
        """Injected dominator is retrievable via .dominator property."""

        class MyDominator(Dominator):
            def dominance_matrix(self, f, direction=None):
                return np.zeros((len(f), len(f)), dtype=bool)

            def dominates_many(self, fa, f_matrix, direction=None):
                dominates = np.zeros(len(f_matrix), dtype=bool)
                return dominates, dominates.copy()

        d = MyDominator()
        comp = ParetoComparator(dominator=d)
        assert comp.dominator is d


# ===========================================================================
# EpsilonDominator Tests (#74)
# ===========================================================================
class TestEpsilonDominator:
    """Tests for EpsilonDominator (ε-box dominance, Laumanns et al. 2002)."""

    # -----------------------------------------------------------------------
    # 1. Same ε-box → mutually non-dominating
    # -----------------------------------------------------------------------
    def test_same_box_mutual_non_domination(self) -> None:
        """Two points in the same ε-box are mutually non-dominating."""
        # eps=1.0: both (0.1, 0.2) and (0.9, 0.8) fall in box (0, 0).
        dom = EpsilonDominator(eps=1.0)
        f = np.array([[0.1, 0.2], [0.9, 0.8]])
        mat = dom.dominance_matrix(f)
        assert not mat[0, 1]
        assert not mat[1, 0]

    # -----------------------------------------------------------------------
    # 2. Strictly better box → dominance
    # -----------------------------------------------------------------------
    def test_better_box_dominates(self) -> None:
        """A point in a strictly better box dominates one in a worse box."""
        # eps=1.0: (0.5, 0.5) → box (0, 0); (1.5, 1.5) → box (1, 1).
        dom = EpsilonDominator(eps=1.0)
        f = np.array([[0.5, 0.5], [1.5, 1.5]])
        mat = dom.dominance_matrix(f)
        assert mat[0, 1]  # (0,0) box dominates (1,1) box
        assert not mat[1, 0]

    # -----------------------------------------------------------------------
    # 3. Tiny eps recovers ordinary Pareto dominance
    # -----------------------------------------------------------------------
    def test_tiny_eps_recovers_pareto(self) -> None:
        """Additive mode with tiny eps recovers ParetoDominator exactly."""
        rng = np.random.default_rng(7)
        f = rng.random((12, 3))
        eps = 1e-9  # boxes effectively as small as floating-point resolution
        dom_eps = EpsilonDominator(eps=eps)
        dom_pareto = ParetoDominator()
        mat_eps = dom_eps.dominance_matrix(f)
        mat_pareto = dom_pareto.dominance_matrix(f)
        np.testing.assert_array_equal(mat_eps, mat_pareto)

    # -----------------------------------------------------------------------
    # 4. direction for a maximize objective
    # -----------------------------------------------------------------------
    def test_direction_maximize(self) -> None:
        """With direction=+1 (maximize), a higher-box point dominates."""
        # Under minimization, (0.5, 0.5) box (0,0) would dominate (1.5, 1.5) box (1,1).
        # Under maximization, the opposite holds.
        dom = EpsilonDominator(eps=1.0)
        f = np.array([[0.5, 0.5], [1.5, 1.5]])
        direction = np.array([1.0, 1.0])  # maximize both objectives
        mat = dom.dominance_matrix(f, direction=direction)
        # (1.5, 1.5) is in the larger box → better under maximization.
        assert mat[1, 0]
        assert not mat[0, 1]

    # -----------------------------------------------------------------------
    # 5. Per-objective eps array (different box widths per objective)
    # -----------------------------------------------------------------------
    def test_per_objective_eps_array(self) -> None:
        """Different eps per objective correctly places points in distinct boxes."""
        # eps = [2.0, 0.5]:
        #   f[0] = (0.5, 0.1) → box (0, 0)
        #   f[1] = (1.5, 0.6) → box (0, 1)  [same box on obj-0, different on obj-1]
        # → non-dominated (different on obj-1, same on obj-0)
        dom = EpsilonDominator(eps=np.array([2.0, 0.5]))
        f = np.array([[0.5, 0.1], [1.5, 0.6]])
        mat = dom.dominance_matrix(f)
        # obj-0 box: both 0; obj-1 box: 0 vs 1 → f[0] dominates f[1]
        assert mat[0, 1]
        assert not mat[1, 0]

    # -----------------------------------------------------------------------
    # 6. Multiplicative mode: basic dominance case
    # -----------------------------------------------------------------------
    def test_multiplicative_mode_basic(self) -> None:
        """Multiplicative mode: lower-box point dominates in a clear case."""
        # eps=0.5: log(1+0.5)=log(1.5) ≈ 0.405
        # f[0]=(1.1, 1.1): box = floor(log(1.1)/log(1.5)) = floor(0.228) = 0
        # f[1]=(2.5, 2.5): box = floor(log(2.5)/log(1.5)) = floor(2.27)  = 2
        dom = EpsilonDominator(eps=0.5, mode="multiplicative")
        f = np.array([[1.1, 1.1], [2.5, 2.5]])
        mat = dom.dominance_matrix(f)
        assert mat[0, 1]  # box (0,0) dominates box (2,2)
        assert not mat[1, 0]

    # -----------------------------------------------------------------------
    # 7. Multiplicative mode raises on non-positive values
    # -----------------------------------------------------------------------
    def test_multiplicative_raises_on_nonpositive(self) -> None:
        """Multiplicative mode raises ValueError when f contains non-positive values."""
        dom = EpsilonDominator(eps=0.5, mode="multiplicative")
        f_zero = np.array([[0.0, 1.0], [1.0, 2.0]])
        with pytest.raises(ValueError, match="strictly positive"):
            dom.dominance_matrix(f_zero)

        f_neg = np.array([[-1.0, 1.0], [1.0, 2.0]])
        with pytest.raises(ValueError, match="strictly positive"):
            dom.dominance_matrix(f_neg)

    # -----------------------------------------------------------------------
    # 8. Non-positive / zero eps raises ValueError
    # -----------------------------------------------------------------------
    def test_zero_eps_raises(self) -> None:
        """eps=0 raises ValueError at construction."""
        with pytest.raises(ValueError, match="strictly positive"):
            EpsilonDominator(eps=0.0)

    def test_negative_eps_raises(self) -> None:
        """Negative eps raises ValueError at construction."""
        with pytest.raises(ValueError, match="strictly positive"):
            EpsilonDominator(eps=-0.1)

    def test_partial_nonpositive_eps_array_raises(self) -> None:
        """An eps array with any non-positive element raises ValueError."""
        with pytest.raises(ValueError, match="strictly positive"):
            EpsilonDominator(eps=np.array([1.0, 0.0]))

    # -----------------------------------------------------------------------
    # 9. dominates() (scalar path) agrees with dominance_matrix()
    # -----------------------------------------------------------------------
    def test_dominates_agrees_with_matrix(self) -> None:
        """dominates() on a 2-row stack matches dominance_matrix()[0,1]."""
        dom = EpsilonDominator(eps=1.0)
        # (0.5, 0.5) box (0,0) and (1.5, 1.5) box (1,1)
        fa = np.array([0.5, 0.5])
        fb = np.array([1.5, 1.5])
        f = np.stack([fa, fb])
        mat = dom.dominance_matrix(f)
        assert dom.dominates(fa, fb) == bool(mat[0, 1])
        assert dom.dominates(fb, fa) == bool(mat[1, 0])

    def test_dominates_same_box_returns_false(self) -> None:
        """dominates() returns False when both points are in the same ε-box."""
        dom = EpsilonDominator(eps=1.0)
        fa = np.array([0.1, 0.2])
        fb = np.array([0.9, 0.8])
        assert not dom.dominates(fa, fb)
        assert not dom.dominates(fb, fa)


# ===========================================================================
# Dominator.dominates_many Tests (#224 archive fast-path acceleration hook)
# ===========================================================================
# Fixed (not hash()-based) per-case seed offsets: Python's hash() on str is
# salted per-process (PYTHONHASHSEED), which would make a failing seed
# non-reproducible across runs -- exactly the wrong property for a
# differential test.
_DIRECTION_KIND_OFFSET = {
    "none": 0,
    "all_minimize": 11,
    "all_maximize": 23,
    "mixed": 37,
}


class TestDominatesMany:
    """Tests for the required one-vs-many ``Dominator.dominates_many`` primitive."""

    # -----------------------------------------------------------------------
    # 1. Dominator.dominates_many is required
    # -----------------------------------------------------------------------
    def test_dominates_many_is_required(self) -> None:
        """A subclass implementing only dominance_matrix remains abstract."""

        class MinimalDominator(Dominator):
            def dominance_matrix(self, f, direction=None):
                return ParetoDominator().dominance_matrix(f, direction)

        with pytest.raises(TypeError, match="dominates_many"):
            MinimalDominator()

    # -----------------------------------------------------------------------
    # 2. ParetoDominator.dominates_many agrees exactly with dominates()
    # -----------------------------------------------------------------------
    @pytest.mark.parametrize("n,m", [(1, 2), (5, 3), (20, 4), (30, 1)])
    @pytest.mark.parametrize(
        "direction_kind", ["none", "all_minimize", "all_maximize", "mixed"]
    )
    def test_pareto_dominates_many_matches_dominates(
        self, n: int, m: int, direction_kind: str
    ) -> None:
        """dominates_many()[0][i]/[1][i] match dominates(fa, f_matrix[i]) both ways."""
        rng = np.random.default_rng(
            seed=n * 100 + m + _DIRECTION_KIND_OFFSET[direction_kind]
        )
        dom = ParetoDominator()
        fa = rng.random(m)
        f_matrix = rng.random((n, m))
        if direction_kind == "none":
            direction = None
        elif direction_kind == "all_minimize":
            direction = -np.ones(m)
        elif direction_kind == "all_maximize":
            direction = np.ones(m)
        else:
            direction = np.array([1.0 if i % 2 == 0 else -1.0 for i in range(m)])

        fa_dom, dom_fa = dom.dominates_many(fa, f_matrix, direction)
        assert fa_dom.shape == (n,)
        assert dom_fa.shape == (n,)
        for i in range(n):
            assert fa_dom[i] == dom.dominates(fa, f_matrix[i], direction), (i,)
            assert dom_fa[i] == dom.dominates(f_matrix[i], fa, direction), (i,)

    def test_pareto_dominates_many_ties(self) -> None:
        """Identical rows and partial-objective ties are handled correctly."""
        dom = ParetoDominator()
        fa = np.array([1.0, 2.0, 3.0])
        f_matrix = np.array(
            [
                [1.0, 2.0, 3.0],  # identical to fa -> neither dominates
                [1.0, 2.0, 2.0],  # tie on 0,1, better on 2 -> row dominates fa
                [1.0, 2.0, 4.0],  # tie on 0,1, worse on 2 -> fa dominates row
            ]
        )
        fa_dom, dom_fa = dom.dominates_many(fa, f_matrix)
        for i in range(len(f_matrix)):
            assert fa_dom[i] == dom.dominates(fa, f_matrix[i])
            assert dom_fa[i] == dom.dominates(f_matrix[i], fa)
        assert not fa_dom[0]
        assert not dom_fa[0]
        assert dom_fa[1]
        assert not fa_dom[1]
        assert fa_dom[2]
        assert not dom_fa[2]

    # -----------------------------------------------------------------------
    # 3. EpsilonDominator.dominates_many agrees exactly with dominates()
    # -----------------------------------------------------------------------
    @pytest.mark.parametrize("n,m", [(1, 2), (5, 3), (20, 2)])
    @pytest.mark.parametrize(
        "direction_kind", ["none", "all_minimize", "all_maximize", "mixed"]
    )
    def test_epsilon_dominates_many_matches_dominates(
        self, n: int, m: int, direction_kind: str
    ) -> None:
        """EpsilonDominator.dominates_many matches dominates() row-by-row."""
        rng = np.random.default_rng(
            seed=n * 50 + m + _DIRECTION_KIND_OFFSET[direction_kind]
        )
        dom = EpsilonDominator(eps=0.3)
        fa = rng.random(m)
        f_matrix = rng.random((n, m))
        if direction_kind == "none":
            direction = None
        elif direction_kind == "all_minimize":
            direction = -np.ones(m)
        elif direction_kind == "all_maximize":
            direction = np.ones(m)
        else:
            direction = np.array([1.0 if i % 2 == 0 else -1.0 for i in range(m)])

        fa_dom, dom_fa = dom.dominates_many(fa, f_matrix, direction)
        for i in range(n):
            assert fa_dom[i] == dom.dominates(fa, f_matrix[i], direction), (i,)
            assert dom_fa[i] == dom.dominates(f_matrix[i], fa, direction), (i,)

    def test_epsilon_dominates_many_same_box_ties(self) -> None:
        """Points in the same eps-box neither dominate under dominates_many."""
        dom = EpsilonDominator(eps=1.0)
        fa = np.array([0.1, 0.2])
        f_matrix = np.array([[0.9, 0.8], [1.5, 1.5]])
        fa_dom, dom_fa = dom.dominates_many(fa, f_matrix)
        assert not fa_dom[0]
        assert not dom_fa[0]
        assert fa_dom[1]
        assert not dom_fa[1]


# ===========================================================================
# EpsilonDominanceComparator Tests
# ===========================================================================
class TestEpsilonDominanceComparator:
    """Tests for EpsilonDominanceComparator (ε-box dominance, Laumanns 2002)."""

    # -----------------------------------------------------------------------
    # 1. Import from top-level package
    # -----------------------------------------------------------------------
    def test_import_from_saealib(self) -> None:
        """EpsilonDominanceComparator can be imported from saealib directly."""
        import saealib

        assert saealib.EpsilonDominanceComparator is EpsilonDominanceComparator

    # -----------------------------------------------------------------------
    # 2. eps / mode properties reflect constructor args
    # -----------------------------------------------------------------------
    def test_eps_property_scalar(self) -> None:
        comp = EpsilonDominanceComparator(eps=0.5)
        assert comp.eps == 0.5

    def test_mode_property_default(self) -> None:
        comp = EpsilonDominanceComparator(eps=0.5)
        assert comp.mode == "additive"

    def test_mode_property_multiplicative(self) -> None:
        comp = EpsilonDominanceComparator(eps=0.5, mode="multiplicative")
        assert comp.mode == "multiplicative"

    def test_eps_property_array(self) -> None:
        eps_arr = np.array([0.1, 0.2])
        comp = EpsilonDominanceComparator(eps=eps_arr)
        np.testing.assert_array_equal(comp.eps, eps_arr)

    # -----------------------------------------------------------------------
    # 3. Tiny eps behaves like standard Pareto (compare matches ParetoComparator)
    # -----------------------------------------------------------------------
    def test_tiny_eps_matches_pareto_comparator(self) -> None:
        """With eps→0, compare_population agrees with ParetoComparator."""
        f = np.array([[0.0, 1.0], [1.0, 0.0], [0.5, 0.5]])
        pop = _make_pop(f)
        pareto = ParetoComparator()
        eps_comp = EpsilonDominanceComparator(eps=1e-9)
        for i in range(len(f)):
            for j in range(len(f)):
                if i == j:
                    continue
                expected = pareto.compare_population(pop, i, j)
                actual = eps_comp.compare_population(pop, i, j)
                assert actual == expected, f"Mismatch at ({i}, {j})"

    # -----------------------------------------------------------------------
    # 4. Same ε-box → compare returns 0 where plain Pareto would return ±1
    # -----------------------------------------------------------------------
    def test_same_box_compare_returns_zero(self) -> None:
        """Two points in the same ε-box are non-dominated (compare == 0)."""
        # eps=1.0: (0.1, 0.2) and (0.9, 0.8) both fall in box (0, 0).
        # Under plain Pareto, (0.1, 0.2) dominates (0.9, 0.8).
        f = np.array([[0.1, 0.2], [0.9, 0.8]])
        pop = _make_pop(f)

        pareto = ParetoComparator()
        assert pareto.compare_population(pop, 0, 1) == -1  # plain Pareto: 0 dominates 1

        eps_comp = EpsilonDominanceComparator(eps=1.0)
        assert eps_comp.compare_population(pop, 0, 1) == 0  # same box → non-dominated
        assert eps_comp.compare_population(pop, 1, 0) == 0

    # -----------------------------------------------------------------------
    # 5. sort_population produces valid permutation; infeasible ranked last
    # -----------------------------------------------------------------------
    def test_sort_population_valid_permutation(self) -> None:
        """sort_population returns all indices exactly once."""
        f = np.array([[0.0, 3.0], [3.0, 0.0], [2.0, 2.0]])
        pop = _make_pop(f)
        comp = EpsilonDominanceComparator(eps=0.5)
        order = comp.sort_population(pop)
        assert sorted(order) == [0, 1, 2]

    def test_sort_population_output_is_int_array(self) -> None:
        f = np.array([[0.0, 1.0], [1.0, 0.0], [0.5, 0.5]])
        pop = _make_pop(f)
        comp = EpsilonDominanceComparator(eps=0.5)
        order = comp.sort_population(pop)
        assert np.issubdtype(order.dtype, np.integer)

    def test_sort_population_infeasible_last(self) -> None:
        """Infeasible individuals appear after all feasible ones."""
        f = np.array([[10.0, 10.0], [0.0, 0.0], [1.0, 1.0]])
        cv = np.array([1.0, 0.0, 0.0])
        pop = _make_pop(f, cv)
        comp = EpsilonDominanceComparator(eps=0.5)
        order = comp.sort_population(pop)
        assert order[-1] == 0  # infeasible individual is last

    def test_sort_population_infeasible_by_ascending_cv(self) -> None:
        """Multiple infeasible individuals are sorted by ascending cv."""
        f = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
        cv = np.array([2.0, 0.5, 1.0])
        pop = _make_pop(f, cv)
        comp = EpsilonDominanceComparator(eps=0.5)
        order = comp.sort_population(pop)
        assert list(order) == [1, 2, 0]

    # -----------------------------------------------------------------------
    # 6. Multiplicative mode works on strictly-positive objectives
    # -----------------------------------------------------------------------
    def test_multiplicative_mode_compare(self) -> None:
        """Multiplicative mode: point in lower box dominates point in higher box."""
        # eps=0.5, log(1.5)≈0.405
        # f[0]=(1.1,1.1): box=floor(log(1.1)/log(1.5))=0 each
        # f[1]=(2.5,2.5): box=floor(log(2.5)/log(1.5))=2 each
        # → f[0] dominates f[1]
        f = np.array([[1.1, 1.1], [2.5, 2.5]])
        pop = _make_pop(f)
        comp = EpsilonDominanceComparator(eps=0.5, mode="multiplicative")
        assert comp.compare_population(pop, 0, 1) == -1
        assert comp.compare_population(pop, 1, 0) == 1

    def test_multiplicative_mode_same_box_non_dominated(self) -> None:
        """Multiplicative mode: two points in the same ε-box are non-dominated."""
        # eps=1.0: log(2)≈0.693; f[0]=(1.1,1.1),f[1]=(1.9,1.9) → both box 0
        f = np.array([[1.1, 1.1], [1.9, 1.9]])
        pop = _make_pop(f)
        comp = EpsilonDominanceComparator(eps=1.0, mode="multiplicative")
        assert comp.compare_population(pop, 0, 1) == 0
        assert comp.compare_population(pop, 1, 0) == 0

    # -----------------------------------------------------------------------
    # 7. Subclass relationship
    # -----------------------------------------------------------------------
    def test_is_subclass_of_pareto_comparator(self) -> None:
        assert issubclass(EpsilonDominanceComparator, ParetoComparator)


# ===========================================================================
# spea2_fitness Tests (#74)
# ===========================================================================
class TestSpea2Fitness:
    """Tests for spea2_fitness (Zitzler et al., 2001, SPEA2)."""

    # -----------------------------------------------------------------------
    # 1. Non-dominated set: R==0 for all, F==D in (0, 0.5]
    # -----------------------------------------------------------------------
    def test_non_dominated_set_r_zero(self) -> None:
        """Mutually non-dominating points all have R(i)==0 and F(i) in (0, 0.5]."""
        # Points on the Pareto front of a 2-obj minimization problem
        f = np.array([[0.0, 3.0], [1.0, 2.0], [2.0, 1.0], [3.0, 0.0]])
        fitness = spea2_fitness(f)
        # R == 0 for all (no one is dominated)
        # Verify indirectly: F must equal D, and D in (0, 0.5]
        assert fitness.shape == (4,)
        assert np.all(fitness > 0.0)
        assert np.all(fitness <= 0.5)

    # -----------------------------------------------------------------------
    # 2. Dominance chain: R ordering and non-dominated R==0
    # -----------------------------------------------------------------------
    def test_dominance_chain_r_ordering(self) -> None:
        """In a clear dominance chain A→B→C, R(A) < R(B) < R(C) and R(A)==0."""
        # A dominates B dominates C (2-obj minimization)
        f = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        fitness = spea2_fitness(f)

        # Reconstruct R manually to verify ordering
        # S: A dominates B and C → S(A)=2; B dominates C → S(B)=1; S(C)=0
        # R: R(A)=0 (no one dominates A); R(B)=S(A)=2; R(C)=S(A)+S(B)=3
        # So fitness order: A < B < C (lower R = better)
        assert fitness[0] < fitness[1]  # A better than B
        assert fitness[1] < fitness[2]  # B better than C

        # Non-dominated point (A) has R==0, so F(A) = D(A) <= 0.5 < 1
        assert fitness[0] < 1.0

    # -----------------------------------------------------------------------
    # 3. direction for maximization flips non-dominated set
    # -----------------------------------------------------------------------
    def test_direction_maximization_flips_nondominated(self) -> None:
        """With direction=+1 (maximize), the 'best' and 'worst' points swap."""
        # Under minimization: [0,0] dominates [1,1]
        # Under maximization: [1,1] dominates [0,0]
        f = np.array([[0.0, 0.0], [1.0, 1.0]])
        direction = np.array([1.0, 1.0])  # maximize both

        fitness_max = spea2_fitness(f, direction=direction)
        # Under maximization, f[1]=[1,1] is non-dominated → R==0 → F < 1
        # f[0]=[0,0] is dominated → R > 0 → F >= R > 0
        assert fitness_max[1] < fitness_max[0]
        # Non-dominated point has R==0 → F <= 0.5
        assert fitness_max[1] <= 0.5

        # Under minimization (default), f[0] is non-dominated
        fitness_min = spea2_fitness(f)
        assert fitness_min[0] < fitness_min[1]
        assert fitness_min[0] <= 0.5

    # -----------------------------------------------------------------------
    # 4. Output shape and N==1 edge case
    # -----------------------------------------------------------------------
    def test_output_shape(self) -> None:
        """Returns shape (N,) for a general input."""
        rng = np.random.default_rng(0)
        f = rng.random((8, 3))
        fitness = spea2_fitness(f)
        assert fitness.shape == (8,)

    def test_single_point_r_zero_finite(self) -> None:
        """N==1 returns a single finite value with R==0 (F = D <= 0.5)."""
        f = np.array([[1.0, 2.0]])
        fitness = spea2_fitness(f)
        assert fitness.shape == (1,)
        assert np.isfinite(fitness[0])
        # Single point is trivially non-dominated → R==0 → F <= 0.5
        assert fitness[0] <= 0.5

    def test_empty_input_returns_empty(self) -> None:
        """N==0 returns an empty array of shape (0,)."""
        f = np.empty((0, 2))
        fitness = spea2_fitness(f)
        assert fitness.shape == (0,)

    # -----------------------------------------------------------------------
    # 5. Custom dominator gives same result as default
    # -----------------------------------------------------------------------
    def test_custom_pareto_dominator_matches_default(self) -> None:
        """Passing dominator=ParetoDominator() gives identical result to None."""
        rng = np.random.default_rng(42)
        f = rng.random((10, 2))
        fitness_default = spea2_fitness(f)
        fitness_explicit = spea2_fitness(f, dominator=ParetoDominator())
        np.testing.assert_array_almost_equal(fitness_default, fitness_explicit)

    # -----------------------------------------------------------------------
    # 6. Lower-is-better orientation: strongly dominated point has largest F
    # -----------------------------------------------------------------------
    def test_strongly_dominated_has_largest_fitness(self) -> None:
        """A point dominated by every other point has the largest SPEA2 fitness."""
        # f[3] = [10, 10] is dominated by all others in minimization
        f = np.array([[0.0, 3.0], [1.5, 1.5], [3.0, 0.0], [10.0, 10.0]])
        fitness = spea2_fitness(f)
        # The worst point must have the maximum fitness (lower = better)
        assert fitness[3] == fitness.max()


# ===========================================================================
# SPEA2Comparator Tests (#74)
# ===========================================================================
class TestSPEA2Comparator:
    """Tests for SPEA2Comparator (Zitzler et al., 2001)."""

    # -----------------------------------------------------------------------
    # 1. Persistent fitness declaration
    # -----------------------------------------------------------------------
    def test_get_required_attrs_declares_spea2_fitness(self) -> None:
        comp = SPEA2Comparator()
        problem = Problem(
            func=lambda x: np.array([np.sum(x), np.sum(x)]),
            dim=2,
            n_obj=2,
            direction=np.array([-1.0, -1.0]),
            lb=[0.0, 0.0],
            ub=[1.0, 1.0],
        )

        attrs = comp.get_required_attrs(problem)

        assert len(attrs) == 1
        attr = attrs[0]
        assert attr.name == "spea2_fitness"
        assert np.dtype(attr.dtype) == np.dtype(np.float64)
        assert attr.shape == ()
        assert np.isnan(attr.default)

    def test_fitness_persists_from_merged_pool_to_mating_population(self) -> None:
        comp = SPEA2Comparator()
        problem = Problem(
            func=lambda x: np.array([np.sum(x), np.sum(x)]),
            dim=2,
            n_obj=2,
            direction=np.array([-1.0, -1.0]),
            lb=[0.0, 0.0],
            ub=[1.0, 1.0],
        )
        f = np.array(
            [
                [0.6, 0.3],
                [0.3, 0.7],
                [0.0, 1.0],
                [0.1, 0.9],
                [0.2, 0.8],
                [0.4, 0.6],
                [0.5, 0.5],
                [0.7, 0.2],
            ]
        )
        pool = _make_spea2_pop(f, comp, problem)
        comp.prepare_population(pool)
        smaller = pool.extract([0, 1])

        assert comp.compare_population(smaller, 0, 1) == -1

        recomputed = comp._compute_fitness(smaller)
        np.testing.assert_allclose(recomputed[0], recomputed[1])

    def test_prepare_population_recomputes_stale_fitness(self) -> None:
        comp = SPEA2Comparator()
        problem = Problem(
            func=lambda x: np.array([np.sum(x), np.sum(x)]),
            dim=2,
            n_obj=2,
            direction=np.array([-1.0, -1.0]),
            lb=[0.0, 0.0],
            ub=[1.0, 1.0],
        )
        f = np.array([[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]])
        population = _make_spea2_pop(f, comp, problem)
        population.update_array("spea2_fitness", np.full(len(population), 999.0))
        expected = comp._compute_fitness(population)

        comp.prepare_population(population)

        np.testing.assert_allclose(population.get_array("spea2_fitness"), expected)

    def test_truncation_selection_recomputes_stale_pool_fitness(self) -> None:
        comp = SPEA2Comparator(direction=np.array([-1.0, -1.0]))
        problem = Problem(
            func=lambda x: np.array([np.sum(x), np.sum(x)]),
            dim=2,
            n_obj=2,
            direction=np.array([-1.0, -1.0]),
            lb=[0.0, 0.0],
            ub=[1.0, 1.0],
        )
        f = np.array(
            [
                [0.0, 1.0],
                [0.5, 0.5],
                [1.0, 0.0],
                [0.2, 0.8],
                [0.8, 0.2],
                [2.0, 2.0],
            ]
        )
        pool = _make_spea2_pop(f, comp, problem)
        pool.update_array("spea2_fitness", np.full(len(pool), 999.0))
        expected_fitness = comp._compute_fitness(pool)
        expected_pool = pool.extract(np.arange(len(pool)))
        expected_pool.update_array("spea2_fitness", expected_fitness)
        expected = comp.sort_population(expected_pool)[:3]

        selection_ctx = SimpleNamespace(comparator=comp, rng=np.random.default_rng(0))
        selected = TruncationSelection().select(selection_ctx, pool, 3)

        np.testing.assert_array_equal(selected, expected)
        np.testing.assert_allclose(pool.get_array("spea2_fitness"), expected_fitness)

    def test_truncation_selection_works_on_schema_less_population(self) -> None:
        comp = SPEA2Comparator(direction=np.array([-1.0, -1.0]))
        f = np.array([[0.6, 0.3], [0.3, 0.7], [0.0, 1.0], [0.1, 0.9], [0.2, 0.8]])
        pool = _make_pop(f)
        assert "spea2_fitness" not in pool.schema
        expected_fitness = comp._compute_fitness(pool)
        expected_pool = pool.extract(np.arange(len(pool)))
        expected = comp.sort_population(expected_pool)[:3]

        selection_ctx = SimpleNamespace(comparator=comp, rng=np.random.default_rng(0))
        selected = TruncationSelection().select(selection_ctx, pool, 3)

        np.testing.assert_array_equal(selected, expected)
        np.testing.assert_allclose(comp._compute_fitness(pool), expected_fitness)

    def test_rank_population_reflects_direction_change_on_schema_less_population(
        self,
    ) -> None:
        f = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]])
        pool = _make_pop(f)
        comp = SPEA2Comparator(direction=np.array([-1.0, -1.0]))

        order1 = comp.rank_population(pool)
        comp.direction = np.array([1.0, 1.0])
        order2 = comp.rank_population(pool)

        np.testing.assert_array_equal(order1, order2[::-1])

    def test_rank_population_works_with_unhashable_dominator(self) -> None:
        @dataclass
        class _UnhashableDominator(Dominator):
            epsilon: float

            def dominance_matrix(
                self, f: np.ndarray, direction: np.ndarray | None = None
            ) -> np.ndarray:
                return ParetoDominator().dominance_matrix(f, direction)

            def dominates_many(
                self,
                fa: np.ndarray,
                f_matrix: np.ndarray,
                direction: np.ndarray | None = None,
            ) -> tuple[np.ndarray, np.ndarray]:
                return ParetoDominator().dominates_many(fa, f_matrix, direction)

        with pytest.raises(TypeError):
            hash(_UnhashableDominator(epsilon=0.1))

        f = np.array([[0.6, 0.3], [0.3, 0.7], [0.0, 1.0], [0.1, 0.9], [0.2, 0.8]])
        pool = _make_pop(f)
        comp = SPEA2Comparator(
            direction=np.array([-1.0, -1.0]),
            dominator=_UnhashableDominator(epsilon=0.1),
        )

        order = comp.rank_population(pool)

        assert len(order) == len(pool)

    # -----------------------------------------------------------------------
    # 2. Class marker
    # -----------------------------------------------------------------------
    def test_is_population_relative_marker(self) -> None:
        """SPEA2Comparator.is_population_relative is True."""
        from saealib.comparators import SPEA2Comparator

        assert SPEA2Comparator.is_population_relative is True

    # -----------------------------------------------------------------------
    # 2. compare() raises NotImplementedError
    # -----------------------------------------------------------------------
    def test_compare_raises(self) -> None:
        """compare() raises NotImplementedError with a guiding message."""
        from saealib.comparators import SPEA2Comparator

        comp = SPEA2Comparator()
        with pytest.raises(NotImplementedError, match="population-relative"):
            comp.compare(
                np.array([0.0, 0.0]),
                0.0,
                np.array([1.0, 1.0]),
                0.0,
            )

    # -----------------------------------------------------------------------
    # 3. sort_population: dominance chain
    # -----------------------------------------------------------------------
    def test_sort_population_dominance_chain(self) -> None:
        """A dominates B dominates C → A first, then B, then C."""
        from saealib.comparators import SPEA2Comparator

        # 2-obj minimization: A=[0,0] dom B=[1,1] dom C=[2,2]
        f = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        pop = _make_pop(f)
        comp = SPEA2Comparator()
        order = comp.sort_population(pop)
        # Non-dominated A must be first; dominated C must be last
        assert order[0] == 0
        assert order[1] == 1
        assert order[2] == 2

    # -----------------------------------------------------------------------
    # 4. sort_population: feasibility — infeasible placed last
    # -----------------------------------------------------------------------
    def test_sort_population_infeasible_last(self) -> None:
        """Infeasible individuals appear after all feasible ones."""
        from saealib.comparators import SPEA2Comparator

        f = np.array([[10.0, 10.0], [0.0, 0.0], [1.0, 1.0]])
        cv = np.array([1.0, 0.0, 0.0])  # idx=0 infeasible
        pop = _make_pop(f, cv)
        comp = SPEA2Comparator()
        order = comp.sort_population(pop)
        assert order[-1] == 0  # infeasible is last

    def test_sort_population_infeasible_by_ascending_cv(self) -> None:
        """Multiple infeasible individuals are sorted by ascending cv."""
        from saealib.comparators import SPEA2Comparator

        f = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
        cv = np.array([2.0, 0.5, 1.0])  # all infeasible
        pop = _make_pop(f, cv)
        comp = SPEA2Comparator()
        order = comp.sort_population(pop)
        # Sorted ascending by cv: idx 1 (0.5), idx 2 (1.0), idx 0 (2.0)
        assert list(order) == [1, 2, 0]

    def test_sort_population_feasible_before_infeasible(self) -> None:
        """Feasible individuals always appear before infeasible ones."""
        from saealib.comparators import SPEA2Comparator

        f = np.array([[5.0, 5.0], [0.0, 0.0], [3.0, 3.0]])
        cv = np.array([0.0, 1.0, 0.0])  # idx=1 infeasible
        pop = _make_pop(f, cv)
        comp = SPEA2Comparator()
        order = comp.sort_population(pop)
        # idx=1 must be last; idx=0 and idx=2 are feasible and come first
        assert order[-1] == 1
        assert set(order[:2]) == {0, 2}

    # -----------------------------------------------------------------------
    # 5. compare_population: pairwise comparisons
    # -----------------------------------------------------------------------
    def test_compare_population_dominant_wins(self) -> None:
        """A dominating point has lower SPEA2 fitness → compare returns -1."""
        from saealib.comparators import SPEA2Comparator

        f = np.array([[0.0, 0.0], [1.0, 1.0]])  # idx 0 dominates idx 1
        pop = _make_pop(f)
        comp = SPEA2Comparator()
        assert comp.compare_population(pop, 0, 1) == -1
        assert comp.compare_population(pop, 1, 0) == 1

    def test_compare_population_feasible_beats_infeasible(self) -> None:
        """Feasible always beats infeasible regardless of objectives."""
        from saealib.comparators import SPEA2Comparator

        f = np.array([[100.0, 100.0], [0.0, 0.0]])
        # idx=0 feasible (bad obj), idx=1 infeasible (good obj)
        cv = np.array([0.0, 1.0])
        pop = _make_pop(f, cv)
        comp = SPEA2Comparator()
        assert comp.compare_population(pop, 0, 1) == -1  # feasible wins
        assert comp.compare_population(pop, 1, 0) == 1

    def test_compare_population_both_infeasible_lower_cv_wins(self) -> None:
        """Both infeasible: lower constraint violation wins."""
        from saealib.comparators import SPEA2Comparator

        f = np.array([[0.0, 0.0], [0.0, 0.0]])
        cv = np.array([0.5, 2.0])
        pop = _make_pop(f, cv)
        comp = SPEA2Comparator()
        assert comp.compare_population(pop, 0, 1) == -1
        assert comp.compare_population(pop, 1, 0) == 1

    # -----------------------------------------------------------------------
    # 6. Tournament-selection safety (compare_population loop)
    # -----------------------------------------------------------------------
    def test_compare_population_no_exception_all_pairs(self) -> None:
        """compare_population works for all index pairs without raising."""
        from saealib.comparators import SPEA2Comparator

        rng = np.random.default_rng(0)
        n = 10
        f = rng.random((n, 2))
        cv = np.zeros(n)
        cv[[2, 5]] = rng.uniform(0.1, 1.0, 2)  # make two individuals infeasible
        pop = _make_pop(f, cv)
        comp = SPEA2Comparator()
        for i in range(n):
            for j in range(n):
                result = comp.compare_population(pop, i, j)
                assert result in (-1, 0, 1)

    # -----------------------------------------------------------------------
    # 7. Import from top-level saealib package
    # -----------------------------------------------------------------------
    def test_import_from_saealib(self) -> None:
        """SPEA2Comparator can be imported from saealib directly."""
        from saealib import SPEA2Comparator

        assert SPEA2Comparator is not None

    # -----------------------------------------------------------------------
    # 8. dominator property and weights direction
    # -----------------------------------------------------------------------
    def test_dominator_property_default(self) -> None:
        """Default dominator is ParetoDominator."""
        from saealib.comparators import SPEA2Comparator

        comp = SPEA2Comparator()
        assert isinstance(comp.dominator, ParetoDominator)

    def test_weights_direction_maximize(self) -> None:
        """direction with +1 interpret objectives as maximize."""
        from saealib.comparators import SPEA2Comparator

        # f[0]=[3,3] is best under maximization; f[1]=[1,1] is dominated
        f = np.array([[3.0, 3.0], [1.0, 1.0]])
        pop = _make_pop(f)
        comp = SPEA2Comparator(direction=np.array([1.0, 1.0]))
        order = comp.sort_population(pop)
        assert order[0] == 0  # [3,3] should be first under maximize

    def test_direction_attribute_stored(self) -> None:
        """direction= kwarg is accessible as comparator.direction."""
        from saealib.comparators import SPEA2Comparator

        d = np.array([1.0, -1.0])
        comp = SPEA2Comparator(direction=d)
        np.testing.assert_array_equal(comp.direction, d)

    def test_is_subclass_of_comparator(self) -> None:
        """SPEA2Comparator is a direct subclass of Comparator (not ParetoComparator)."""
        from saealib.comparators import Comparator, SPEA2Comparator

        assert issubclass(SPEA2Comparator, Comparator)
        assert not issubclass(SPEA2Comparator, ParetoComparator)

    # -----------------------------------------------------------------------
    # 9. Truncation retains boundary/extreme solutions
    # -----------------------------------------------------------------------
    def test_truncation_retains_boundary_solutions(self) -> None:
        """Iterative nearest-neighbour truncation keeps both extreme points.

        This 6-point non-dominated front (all ``F(i) < 1``) is constructed
        so that a one-shot sort on the single-``k`` density ``D(i)`` discards
        the true extreme point at idx 5 (smallest obj-1) in favour of an
        interior point at idx 1, when truncated to the top 4. Zitzler et al.
        (2001) Section 3.2's
        iterative removal -- which recomputes neighbour distances after
        every step and, per the tie-break rule, prefers to remove crowded
        interior points before boundary ones -- keeps both extremes instead.
        """
        from saealib.comparators import SPEA2Comparator

        f = np.array(
            [
                [0.98, 6.85],  # idx 0: extreme (smallest obj 0)
                [2.64, 4.29],
                [2.88, 3.86],
                [6.07, 1.56],
                [6.51, 0.34],
                [7.41, 0.20],  # idx 5: extreme (smallest obj 1)
            ]
        )
        pop = _make_pop(f)
        comp = SPEA2Comparator()
        order = comp.sort_population(pop)
        kept = set(order[:4].tolist())
        assert {0, 5} <= kept


# ===========================================================================
# SPEA2 lifecycle Tests
# ===========================================================================
class TestSPEA2Lifecycle:
    def test_environmental_selection_refreshes_fitness_with_independent_sizes(
        self,
    ) -> None:
        direction = np.array([-1.0, -1.0])
        problem = Problem(
            func=lambda x: np.array([np.sum(x**2), np.sum((x - 1.0) ** 2)]),
            dim=2,
            n_obj=2,
            direction=direction,
            lb=[-1.0, -1.0],
            ub=[1.0, 1.0],
            comparator=SPEA2Comparator(direction=direction),
        )
        n_bar = 4
        n_offspring = 3
        optimizer = (
            Optimizer(problem, seed=0)
            .set_initializer(
                LHSInitializer(n_init_archive=6, n_init_population=n_bar, seed=0)
            )
            .set_algorithm(
                GA(
                    crossover=CrossoverBLXAlpha(prob=0.9, alpha=0.4),
                    mutation=MutationUniform(prob_var=0.1),
                    parent_selection=TournamentSelection(2),
                    survivor_selection=TruncationSelection(),
                )
            )
            .set_strategy(DirectStrategy(n_offspring=n_offspring))
            .set_termination(Termination(max_gen(3)))
        )

        states = list(optimizer.iterate())

        assert states[-1].gen >= 2
        assert len(states) >= 3
        for ctx in states:
            assert "spea2_fitness" in ctx.population.schema
            assert np.all(np.isfinite(ctx.population.get_array("spea2_fitness")))
            assert len(ctx.population) == n_bar


# ===========================================================================
# hypervolume_contributions Tests (#74, Beume et al. 2007)
# ===========================================================================
class TestHypervolumeContributions:
    """Tests for hypervolume_contributions (exclusive HV contribution).

    Reference: Beume, N., Naujoks, B., & Emmerich, M. (2007).
    SMS-EMOA: Multiobjective selection based on dominated hypervolume.
    European Journal of Operational Research, 181(3), 1653-1669.
    """

    # -----------------------------------------------------------------------
    # 1. Known 2D case with explicit reference point
    # -----------------------------------------------------------------------
    def test_two_point_front_known_values(self) -> None:
        """2-point 2D minimization front: verify contributions analytically.

        f = [[0, 2], [1, 0]], reference = [2, 3].

        Leave-one-out hypervolumes (minimization convention):
          HV(S, ref=[2,3])          = 4   (total dominated area)
          HV({[1,0]}, ref=[2,3])    = (2-1)*(3-0) = 3
          HV({[0,2]}, ref=[2,3])    = (2-0)*(3-2) = 2
          contrib[0] = 4 - 3 = 1
          contrib[1] = 4 - 2 = 2

        The exclusive contributions do NOT generally sum to the total HV
        (overlap between exclusive regions is counted only once in HV but
        twice in the individual HV calls).  The sum equals 3 here, not 4.
        """
        f = np.array([[0.0, 2.0], [1.0, 0.0]])
        ref = np.array([2.0, 3.0])

        from saealib.utils.indicators import hypervolume

        contrib = hypervolume_contributions(f, reference_point=ref)
        total = hypervolume(f, ref)

        assert contrib.shape == (2,)
        assert np.all(contrib > 0)
        np.testing.assert_allclose(total, 4.0, rtol=1e-10)

        # Verify exact leave-one-out values.
        np.testing.assert_allclose(contrib[0], 1.0, rtol=1e-10)
        np.testing.assert_allclose(contrib[1], 2.0, rtol=1e-10)

        # Point 1 (1,0) has a larger exclusive contribution than point 0 (0,2).
        assert contrib[1] > contrib[0]

    def test_three_point_front_contributions_positive(self) -> None:
        """3 mutually non-dominated 2D points: all contributions are positive."""
        f = np.array([[0.0, 3.0], [1.5, 1.5], [3.0, 0.0]])
        ref = np.array([4.0, 4.0])
        contrib = hypervolume_contributions(f, reference_point=ref)
        assert contrib.shape == (3,)
        assert np.all(contrib > 0)

    # -----------------------------------------------------------------------
    # 2. Dominated point gets contribution 0
    # -----------------------------------------------------------------------
    def test_dominated_point_zero_contribution(self) -> None:
        """A dominated point contributes 0 to the hypervolume.

        f[0]=[0,2] and f[1]=[2,0] are mutually non-dominated.
        f[2]=[3,3] is dominated by both → its contribution must be 0.
        """
        f = np.array([[0.0, 2.0], [2.0, 0.0], [3.0, 3.0]])
        ref = np.array([4.0, 4.0])
        contrib = hypervolume_contributions(f, reference_point=ref)
        # f[2] is dominated by f[0] and f[1] → contribution must be 0.
        assert contrib[2] == 0.0
        # Non-dominated points should have positive contributions.
        assert contrib[0] > 0.0
        assert contrib[1] > 0.0

    # -----------------------------------------------------------------------
    # 3. direction for maximization
    # -----------------------------------------------------------------------
    def test_direction_maximization_equivalent_to_negation(self) -> None:
        """Maximization via direction is equivalent to negating the objectives."""
        f = np.array([[3.0, 0.0], [0.0, 3.0], [1.0, 1.0]])
        ref_max = np.array([-1.0, -1.0])  # reference in maximize space
        direction = np.array([1.0, 1.0])

        # Contributions with direction=maximize
        contrib_dir = hypervolume_contributions(
            f, reference_point=ref_max, direction=direction
        )

        # Equivalent: negate f and reference_point, minimize
        f_neg = -f
        ref_neg = -ref_max  # = [1,1]
        contrib_neg = hypervolume_contributions(f_neg, reference_point=ref_neg)

        np.testing.assert_allclose(contrib_dir, contrib_neg, rtol=1e-10)

    # -----------------------------------------------------------------------
    # 4. Auto reference point (reference_point=None)
    # -----------------------------------------------------------------------
    def test_auto_reference_point_all_nonneg(self) -> None:
        """Auto reference point gives non-negative contributions to all points."""
        f = np.array([[0.0, 3.0], [1.5, 1.5], [3.0, 0.0]])
        contrib = hypervolume_contributions(f)
        assert contrib.shape == (3,)
        assert np.all(contrib >= 0.0)

    def test_auto_reference_point_nondominated_positive(self) -> None:
        """Non-dominated points get positive contribution with auto reference."""
        f = np.array([[0.0, 2.0], [1.0, 0.0], [3.0, 3.0]])
        # f[2] is dominated by both f[0] and f[1].
        contrib = hypervolume_contributions(f)
        assert contrib[0] > 0.0
        assert contrib[1] > 0.0

    def test_auto_reference_degenerate_single_axis(self) -> None:
        """Auto reference handles a degenerate axis (all points equal on one obj)."""
        # All points have f[:, 1] == 1.0 → span[1] == 0.
        f = np.array([[0.0, 1.0], [1.0, 1.0]])
        contrib = hypervolume_contributions(f)
        assert contrib.shape == (2,)
        assert np.all(contrib >= 0.0)

    # -----------------------------------------------------------------------
    # 5. Shape and edge cases
    # -----------------------------------------------------------------------
    def test_output_shape(self) -> None:
        """Returns shape (N,) for general inputs."""
        rng = np.random.default_rng(0)
        f = rng.random((7, 2))
        contrib = hypervolume_contributions(f)
        assert contrib.shape == (7,)

    def test_n_equals_zero_returns_empty(self) -> None:
        """Empty input returns empty array."""
        f = np.empty((0, 2))
        contrib = hypervolume_contributions(f)
        assert contrib.shape == (0,)

    def test_n_equals_one_returns_total_hv(self) -> None:
        """N==1: removing the only point leaves HV=0 → contribution equals total HV."""
        f = np.array([[1.0, 2.0]])
        ref = np.array([3.0, 4.0])

        from saealib.utils.indicators import hypervolume

        contrib = hypervolume_contributions(f, reference_point=ref)
        assert contrib.shape == (1,)
        assert contrib[0] > 0.0
        np.testing.assert_allclose(contrib[0], hypervolume(f, ref), rtol=1e-10)

    def test_1d_input_reshaped(self) -> None:
        """1-D input is treated as a single-point (1, n_obj) matrix."""
        f = np.array([1.0, 2.0])
        ref = np.array([3.0, 4.0])
        contrib = hypervolume_contributions(f, reference_point=ref)
        assert contrib.shape == (1,)
        assert contrib[0] > 0.0

    def test_all_values_nonneg(self) -> None:
        """All returned contributions are >= 0 (no negative floating noise)."""
        rng = np.random.default_rng(99)
        f = rng.random((12, 3))
        contrib = hypervolume_contributions(f)
        assert np.all(contrib >= 0.0)

    # -----------------------------------------------------------------------
    # 6. Import from both saealib and saealib.utils
    # -----------------------------------------------------------------------
    def test_import_from_saealib(self) -> None:
        """hypervolume_contributions is importable from top-level saealib."""
        from saealib import hypervolume_contributions as hvc

        assert hvc is not None

    def test_import_from_saealib_utils(self) -> None:
        """hypervolume_contributions is importable from saealib.utils."""
        from saealib.utils import hypervolume_contributions as hvc

        assert hvc is not None


# ===========================================================================
# HypervolumeComparator Tests
# ===========================================================================
class TestHypervolumeComparator:
    """Tests for HypervolumeComparator (Beume et al., 2007)."""

    # -----------------------------------------------------------------------
    # 1. Class marker
    # -----------------------------------------------------------------------
    def test_is_population_relative_marker(self) -> None:
        """HypervolumeComparator.is_population_relative is True."""
        from saealib.comparators import HypervolumeComparator

        assert HypervolumeComparator.is_population_relative is True

    # -----------------------------------------------------------------------
    # 2. compare() raises NotImplementedError
    # -----------------------------------------------------------------------
    def test_compare_raises(self) -> None:
        """compare() raises NotImplementedError with a guiding message."""
        from saealib.comparators import HypervolumeComparator

        comp = HypervolumeComparator()
        with pytest.raises(NotImplementedError, match="population-relative"):
            comp.compare(
                np.array([0.0, 0.0]),
                0.0,
                np.array([1.0, 1.0]),
                0.0,
            )

    # -----------------------------------------------------------------------
    # 3. sort_population: dominated point sorts after first-front points;
    #    within a front the highest-HV-contribution point comes first
    # -----------------------------------------------------------------------
    def test_sort_population_front_rank_ordering(self) -> None:
        """Dominated point (front 1) sorts after the non-dominated point."""
        from saealib.comparators import HypervolumeComparator

        # idx 0 = [0,0] is non-dominated (Pareto front 0);
        # idx 1 = [2,1] and idx 2 = [1,2] are in front 1 (both dominated by idx 0).
        f = np.array([[0.0, 0.0], [2.0, 1.0], [1.0, 2.0]])
        pop = _make_pop(f)
        comp = HypervolumeComparator()
        order = comp.sort_population(pop)
        # idx 0 (front 0) must come first
        assert order[0] == 0
        # idx 1 and 2 (front 1) occupy the last two positions
        assert set(order[1:]) == {1, 2}

    def test_last_front_only_leaves_better_fronts_in_input_order(self) -> None:
        f = np.array([[0.0, 3.0], [1.0, 1.0], [3.0, 0.0], [2.0, 4.0], [4.0, 2.0]])

        generic = HypervolumeComparator().sort_population(_make_pop(f))
        paper = HypervolumeComparator(last_front_only=True).sort_population(
            _make_pop(f)
        )

        assert generic[:3].tolist() == [1, 0, 2]
        assert paper[:3].tolist() == [0, 1, 2]
        assert set(generic[3:].tolist()) == set(paper[3:].tolist()) == {3, 4}

    def test_last_front_only_still_ranks_within_the_last_front(self) -> None:
        f = np.array([[0.0, 0.0], [5.0, 1.0], [1.0, 5.0], [3.0, 3.0]])

        order = HypervolumeComparator(last_front_only=True).sort_population(
            _make_pop(f)
        )

        assert order[0] == 0
        assert order[1:].tolist() == [3, 1, 2]

    def test_sort_population_hv_contribution_within_front(self) -> None:
        """Within front 0, the point with smallest HV contribution sorts last."""
        from saealib.comparators import HypervolumeComparator

        # Three non-dominated points (minimize).
        # [0,4] and [4,0] are extreme; [3.9, 0.1] is very close to [4,0]
        # and therefore has a small exclusive HV contribution.
        # Verified: with the default reference point (worst + 1.0 = [5,5],
        # Beume et al. 2007), hypervolume_contributions gives approx
        # [3.9, 0.39, 0.1] → idx 2 ([4,0]) has the smallest contribution and
        # must sort last.
        f = np.array([[0.0, 4.0], [3.9, 0.1], [4.0, 0.0]])
        pop = _make_pop(f)
        comp = HypervolumeComparator()
        order = comp.sort_population(pop)
        # idx 2 has the smallest HV contribution and must be last
        assert order[-1] == 2

    # -----------------------------------------------------------------------
    # 4. Feasibility: infeasible individuals placed last, ordered by cv
    # -----------------------------------------------------------------------
    def test_sort_population_infeasible_last(self) -> None:
        """Infeasible individuals appear after all feasible ones."""
        from saealib.comparators import HypervolumeComparator

        f = np.array([[10.0, 10.0], [0.0, 0.0], [1.0, 1.0]])
        cv = np.array([1.0, 0.0, 0.0])  # idx=0 infeasible
        pop = _make_pop(f, cv)
        comp = HypervolumeComparator()
        order = comp.sort_population(pop)
        assert order[-1] == 0

    def test_sort_population_infeasible_by_ascending_cv(self) -> None:
        """Multiple infeasible individuals are sorted by ascending cv."""
        from saealib.comparators import HypervolumeComparator

        f = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
        cv = np.array([2.0, 0.5, 1.0])  # all infeasible
        pop = _make_pop(f, cv)
        comp = HypervolumeComparator()
        order = comp.sort_population(pop)
        # Sorted ascending by cv: idx 1 (0.5), idx 2 (1.0), idx 0 (2.0)
        assert list(order) == [1, 2, 0]

    # -----------------------------------------------------------------------
    # 5. compare_population: pairwise comparisons
    # -----------------------------------------------------------------------
    def test_compare_population_first_front_beats_second_front(self) -> None:
        """A first-front point compares < 0 against a second-front point."""
        from saealib.comparators import HypervolumeComparator

        # idx 0 is non-dominated; idx 1 is dominated
        f = np.array([[0.0, 0.0], [1.0, 1.0]])
        pop = _make_pop(f)
        comp = HypervolumeComparator()
        assert comp.compare_population(pop, 0, 1) == -1
        assert comp.compare_population(pop, 1, 0) == 1

    def test_compare_population_same_front_higher_contrib_wins(self) -> None:
        """Within the same front, the higher-contribution point wins."""
        from saealib.comparators import HypervolumeComparator

        # Three non-dominated points; idx 0 [0,4] has the highest HV contribution
        # (~3.9); idx 2 [4,0] has the smallest (~0.1) because [3.9,0.1] is nearby.
        # Verified: with the default reference point (worst + 1.0 = [5,5]),
        # hypervolume_contributions ≈ [3.9, 0.39, 0.1]
        f = np.array([[0.0, 4.0], [3.9, 0.1], [4.0, 0.0]])
        pop = _make_pop(f)
        comp = HypervolumeComparator()
        # idx 0 (highest contrib) beats idx 2 (lowest contrib)
        assert comp.compare_population(pop, 0, 2) == -1
        assert comp.compare_population(pop, 2, 0) == 1

    def test_compare_population_feasible_beats_infeasible(self) -> None:
        """Feasible always beats infeasible regardless of objectives."""
        from saealib.comparators import HypervolumeComparator

        f = np.array([[100.0, 100.0], [0.0, 0.0]])
        cv = np.array([0.0, 1.0])  # idx=0 feasible (bad obj), idx=1 infeasible
        pop = _make_pop(f, cv)
        comp = HypervolumeComparator()
        assert comp.compare_population(pop, 0, 1) == -1
        assert comp.compare_population(pop, 1, 0) == 1

    def test_compare_population_both_infeasible_lower_cv_wins(self) -> None:
        """Both infeasible: lower constraint violation wins."""
        from saealib.comparators import HypervolumeComparator

        f = np.array([[0.0, 0.0], [0.0, 0.0]])
        cv = np.array([0.5, 2.0])
        pop = _make_pop(f, cv)
        comp = HypervolumeComparator()
        assert comp.compare_population(pop, 0, 1) == -1
        assert comp.compare_population(pop, 1, 0) == 1

    # -----------------------------------------------------------------------
    # 6. Tournament-safety: compare_population works for all index pairs
    # -----------------------------------------------------------------------
    def test_compare_population_no_exception_all_pairs(self) -> None:
        """compare_population works for all index pairs without raising."""
        from saealib.comparators import HypervolumeComparator

        rng = np.random.default_rng(42)
        n = 10
        f = rng.random((n, 2))
        cv = np.zeros(n)
        cv[[2, 5]] = rng.uniform(0.1, 1.0, 2)  # two infeasible individuals
        pop = _make_pop(f, cv)
        comp = HypervolumeComparator()
        for i in range(n):
            for j in range(n):
                result = comp.compare_population(pop, i, j)
                assert result in (-1, 0, 1)

    # -----------------------------------------------------------------------
    # 7. Custom reference_point is honored
    # -----------------------------------------------------------------------
    def test_custom_reference_point_honored(self) -> None:
        """A custom reference_point is stored and used during ranking."""
        from saealib.comparators import HypervolumeComparator

        f = np.array([[0.0, 3.0], [1.5, 1.5], [3.0, 0.0]])
        ref = np.array([5.0, 5.0])
        pop = _make_pop(f)
        comp = HypervolumeComparator(reference_point=ref)
        # reference_point property should be set
        np.testing.assert_array_equal(comp.reference_point, ref)
        # sort_population must run without error and return a valid permutation
        order = comp.sort_population(pop)
        assert set(order) == {0, 1, 2}
        assert order.dtype == int or np.issubdtype(order.dtype, np.integer)

    def test_custom_reference_point_affects_ranking(self) -> None:
        """Custom reference point runs without error and produces valid ranking."""
        from saealib.comparators import HypervolumeComparator

        # Two non-dominated points: [0.0, 2.0] and [2.0, 0.0]
        # With ref=[3.0, 3.0] both extreme points have equal-ish contributions;
        # we just verify the ranking runs and produces valid output.
        f = np.array([[0.0, 2.0], [2.0, 0.0], [1.5, 1.5]])
        ref = np.array([4.0, 4.0])
        pop = _make_pop(f)
        comp = HypervolumeComparator(reference_point=ref)
        order = comp.sort_population(pop)
        # The dominated middle point should still be last
        assert order[-1] == 2

    # -----------------------------------------------------------------------
    # 8. Import from saealib top-level package
    # -----------------------------------------------------------------------
    def test_import_from_saealib(self) -> None:
        """HypervolumeComparator can be imported from saealib directly."""
        from saealib import HypervolumeComparator

        assert HypervolumeComparator is not None

    # -----------------------------------------------------------------------
    # 9. Default reference point follows Beume et al. (2007): worst + 1.0 (#214)
    # -----------------------------------------------------------------------
    def test_default_reference_point_is_worst_plus_one(self) -> None:
        """With reference_point=None, contributions match an explicit worst+1.0 ref.

        Beume et al. (2007) Section 2.1.3 defines the default reference
        point as the vector of the currently worst objective values
        increased by 1.0, not the old margin-scaled relative padding.
        """
        from saealib.comparators import HypervolumeComparator
        from saealib.utils.indicators import hypervolume_contributions

        f = np.array([[0.0, 4.0], [3.9, 0.1], [4.0, 0.0]])
        pop = _make_pop(f)
        comp = HypervolumeComparator()
        _rank_all, contrib_all = comp._keys(pop)

        expected = hypervolume_contributions(f, reference_point=np.array([5.0, 5.0]))
        np.testing.assert_allclose(contrib_all, expected)

    def test_margin_is_inert_for_default_reference_point(self) -> None:
        """margin= no longer affects the default (reference_point=None) ranking."""
        from saealib.comparators import HypervolumeComparator

        f = np.array([[0.0, 4.0], [3.9, 0.1], [4.0, 0.0]])
        pop_a = _make_pop(f)
        pop_b = _make_pop(f)
        comp_default = HypervolumeComparator()
        comp_custom_margin = HypervolumeComparator(margin=0.9)
        np.testing.assert_array_equal(
            comp_default.sort_population(pop_a),
            comp_custom_margin.sort_population(pop_b),
        )


# ===========================================================================
# NSGA3Comparator Tests
# ===========================================================================
class TestNSGA3Comparator:
    """Tests for NSGA3Comparator (NSGA-III niche preservation)."""

    def test_is_subclass_of_pareto_comparator(self) -> None:
        assert issubclass(NSGA3Comparator, ParetoComparator)

    def test_sort_population_returns_int_array(self) -> None:
        ref = np.array([[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]])
        f = np.array([[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]])
        pop = _make_pop(f)
        comp = NSGA3Comparator(ref, seed=0)
        order = comp.sort_population(pop)
        assert np.issubdtype(order.dtype, np.integer)

    def test_sort_population_contains_all_indices(self) -> None:
        ref = np.array([[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]])
        f = np.array([[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]])
        pop = _make_pop(f)
        comp = NSGA3Comparator(ref, seed=0)
        order = comp.sort_population(pop)
        assert set(order.tolist()) == {0, 1, 2}

    def test_front_0_before_front_1(self) -> None:
        """Non-dominated solutions ranked before dominated ones."""
        ref = np.array([[0.0, 1.0], [1.0, 0.0]])
        f = np.array([[0.0, 1.0], [1.0, 0.0], [2.0, 2.0]])
        pop = _make_pop(f)
        comp = NSGA3Comparator(ref, seed=0)
        order = comp.sort_population(pop)
        assert set(order[:2].tolist()) == {0, 1}
        assert order[2] == 2

    def test_infeasible_last(self) -> None:
        ref = np.array([[0.0, 1.0], [1.0, 0.0]])
        f = np.array([[10.0, 10.0], [0.0, 1.0], [1.0, 0.0]])
        cv = np.array([1.0, 0.0, 0.0])
        pop = _make_pop(f, cv)
        comp = NSGA3Comparator(ref, seed=0)
        order = comp.sort_population(pop)
        assert order[-1] == 0

    def test_sort_population_cached(self) -> None:
        ref = np.array([[0.0, 1.0], [1.0, 0.0]])
        f = np.array([[0.0, 1.0], [1.0, 0.0]])
        pop = _make_pop(f)
        comp = NSGA3Comparator(ref, seed=0)
        order1 = comp.sort_population(pop)
        order2 = comp.sort_population(pop)
        np.testing.assert_array_equal(order1, order2)

    def test_reference_points_property(self) -> None:
        ref = np.array([[0.0, 1.0], [1.0, 0.0]])
        comp = NSGA3Comparator(ref)
        np.testing.assert_array_equal(comp.reference_points, ref)

    def test_three_objectives_uniform_refs(self) -> None:
        """Works with 3-objective uniform reference points from Das-Dennis method."""
        from saealib.utils.weight_vectors import uniform_weight_vectors

        ref = uniform_weight_vectors(3, 4)
        f = np.random.default_rng(42).random((10, 3))
        pop = _make_pop(f)
        comp = NSGA3Comparator(ref, seed=0)
        order = comp.sort_population(pop)
        assert len(order) == 10
        assert set(order.tolist()) == set(range(10))

    def test_direction_maximize(self) -> None:
        """direction=[1,1] selects higher values first."""
        ref = np.array([[0.5, 0.5]])
        f = np.array([[3.0, 3.0], [1.0, 1.0]])
        pop = _make_pop(f)
        comp = NSGA3Comparator(ref, direction=np.array([1.0, 1.0]), seed=0)
        order = comp.sort_population(pop)
        assert order[0] == 0  # [3,3] dominates under maximization

    def test_direction_maximize_matches_negated_minimize(self) -> None:
        """Maximization ordering matches negated minimization ordering.

        Sorting a maximize (direction=+1) population must give the same
        ordering as sorting the negated population under minimize
        (direction=-1), since both represent the same preference over the
        same underlying points.
        """
        ref = np.array([[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]])
        f = np.array([[1.0, 4.0], [4.0, 1.0], [2.0, 2.0], [3.0, 3.0]])

        pop_max = _make_pop(f)
        comp_max = NSGA3Comparator(ref, direction=np.array([1.0, 1.0]), seed=0)
        order_max = comp_max.sort_population(pop_max)

        pop_min = _make_pop(-f)
        comp_min = NSGA3Comparator(ref, direction=np.array([-1.0, -1.0]), seed=0)
        order_min = comp_min.sort_population(pop_min)

        np.testing.assert_array_equal(order_max, order_min)

    # -----------------------------------------------------------------------
    # Normalization must use S_t, not R_t
    # -----------------------------------------------------------------------
    def test_normalize_objectives_front_vs_combined_population_differ(self) -> None:
        """_normalize_objectives(S_t) and _normalize_objectives(R_t) can diverge.

        Deb & Jain (2014) Algorithm 2 normalizes over ``S_t`` (the fronts
        accumulated so far), not the full combined population ``R_t``. For
        a non-degenerate front, a dominated point can never win the ASF
        argmin used to pick extreme points nor lower any per-axis ideal
        value (any point outside ``S_t`` is transitively dominated by a
        member of ``S_t``, and ASF is monotone in every objective) -- so in
        general, restricting normalization to ``S_t`` has no observable
        effect on ``ideal``/``intercepts``. The one path where it does
        differ is the singular-matrix fallback (``intercepts =
        f_trans.max(axis=0)``): this front is small enough (3 points, 3
        objectives) to trigger that fallback on its own, while the larger
        combined array does not, so the two intercepts differ sharply.
        """
        front0 = np.array(
            [
                [0.81, 0.52, 0.29],
                [0.05, 0.38, 0.41],
                [0.05, 0.05, 1.00],
            ]
        )
        r_t = np.vstack(
            [
                front0,
                [
                    [0.81, 49.23, 0.29],  # front 1: dominated by row 0
                    [42.28, 20.06, 0.41],  # front 1: dominated by row 1
                    [33.91, 3.18, 1.00],  # front 1: dominated by row 2
                ],
            ]
        )

        _, ideal_s_t, intercepts_s_t = _normalize_objectives(front0, None)
        _, ideal_r_t, intercepts_r_t = _normalize_objectives(r_t, None)

        np.testing.assert_array_equal(ideal_s_t, ideal_r_t)
        assert not np.allclose(intercepts_s_t, intercepts_r_t)

    def test_normalization_uses_s_t_not_r_t(self) -> None:
        """sort_population still produces a valid, front-respecting ranking
        when normalization is recomputed per front over S_t (see the
        _normalize_objectives-level property test above for the mechanism)."""
        ref = np.array(
            [
                [0.71, 0.00, 0.50],
                [0.44, 0.20, 0.32],
                [0.81, 0.32, 0.15],
            ]
        )
        f = np.array(
            [
                [0.81, 0.52, 0.29],
                [0.05, 0.38, 0.41],
                [0.05, 0.05, 1.00],
                [0.81, 49.23, 0.29],  # front 1: dominated by row 0
                [42.28, 20.06, 0.41],  # front 1: dominated by row 1
                [33.91, 3.18, 1.00],  # front 1: dominated by row 2
            ]
        )
        pop = _make_pop(f)
        comp = NSGA3Comparator(ref, seed=0)
        order = comp.sort_population(pop)
        assert set(order[:3].tolist()) == {0, 1, 2}
        assert set(order[3:].tolist()) == {3, 4, 5}

    # -----------------------------------------------------------------------
    # seed=None reproducibility under a master Optimizer seed
    # -----------------------------------------------------------------------
    def test_seed_none_does_not_eagerly_create_rng(self) -> None:
        """seed=None must not eagerly call default_rng() at construction time."""
        ref = np.array([[0.0, 1.0], [1.0, 0.0]])
        comp = NSGA3Comparator(ref)
        assert comp._rng is None

    def test_seed_none_rng_property_lazily_created_and_cached(self) -> None:
        """The `rng` property lazily creates a Generator and caches it."""
        ref = np.array([[0.0, 1.0], [1.0, 0.0]])
        comp = NSGA3Comparator(ref)
        rng = comp.rng
        assert isinstance(rng, np.random.Generator)
        assert comp.rng is rng

    def test_seed_none_standalone_sort_population_still_works(self) -> None:
        """A standalone comparator (seed=None, never attached to an Optimizer)
        can still call sort_population without error."""
        ref = np.array([[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]])
        f = np.array([[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]])
        pop = _make_pop(f)
        comp = NSGA3Comparator(ref)
        order = comp.sort_population(pop)
        assert set(order.tolist()) == {0, 1, 2}

    def test_explicit_seed_reproducible_regression(self) -> None:
        """An explicit comparator seed gives repeatable ordering independently
        of optimizer-owned RNG state."""
        ref = np.array([[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]])
        f = _nsga3_tie_population_f()
        comp_a = NSGA3Comparator(ref, seed=42)
        comp_b = NSGA3Comparator(ref, seed=42)
        order_a = comp_a.sort_population(_make_pop(f))
        order_b = comp_b.sort_population(_make_pop(f))
        np.testing.assert_array_equal(order_a, order_b)

    def test_seed_none_reproducible_under_master_optimizer_seed(self) -> None:
        """Two Optimizer(seed=42) runs must reproduce NSGA3Comparator ordering.

        An unseeded NSGA3Comparator receives a spawned ``ctx.rng`` from
        Runner at run start, so it reproduces under a fixed master seed like
        every other RNG-consuming component.
        Uses a tie-break-sensitive population (duplicate objective values in
        the same niche) to actually exercise ``rng.choice`` tie-breaking.
        """
        ref = np.array([[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]])
        f = _nsga3_tie_population_f()

        ctx1 = _nsga3_first_ctx(42, ref)
        ctx2 = _nsga3_first_ctx(42, ref)

        pre_state = ctx1.comparator.rng.bit_generator.state["state"]["state"]
        order1 = ctx1.comparator.sort_population(_make_pop(f))
        post_state = ctx1.comparator.rng.bit_generator.state["state"]["state"]
        # The tie-break population must actually consume randomness.
        assert pre_state != post_state

        order2 = ctx2.comparator.sort_population(_make_pop(f))
        np.testing.assert_array_equal(order1, order2)


# ===========================================================================
# _normalize_objectives Tests
# ===========================================================================
class TestNormalizeObjectives:
    def test_direction_minimize_orientation(self) -> None:
        """direction=-1: f_norm must reflect raw f geometry, not -f.

        With 3 axis-aligned extreme points the individual with f=[1,0,0]
        must end up with f_norm close to [1,0,0], associating with ref [1,0,0].
        Before the Bug1 fix, f_signed=-f reversed this mapping.
        """
        f = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        direction = np.array([-1.0, -1.0, -1.0])
        f_norm, _, _ = _normalize_objectives(f, direction)
        # f_norm[0] should be extreme along axis 0
        assert f_norm[0, 0] > f_norm[0, 1]
        assert f_norm[0, 0] > f_norm[0, 2]

    def test_non_identity_intercepts(self) -> None:
        """Bug2: intercepts = 1/plane_coefficients, not plane_coefficients.

        For extreme matrix [[0,1,1],[1,0,1],[1,1,0]] the hyperplane
        coefficients are [0.5,0.5,0.5], so the true intercepts are [2,2,2].
        """
        # Pareto front: three axis-aligned points on a non-identity simplex
        # extreme rows of f_trans will be [0,1,1],[1,0,1],[1,1,0]
        f = np.array(
            [
                [2.0, 0.0, 0.0],  # extreme in obj 0
                [0.0, 2.0, 0.0],  # extreme in obj 1
                [0.0, 0.0, 2.0],  # extreme in obj 2
                [0.5, 0.5, 0.5],  # interior
            ]
        )
        # direction=None → f_pos = f (all-minimize)
        f_norm, _ideal, intercepts = _normalize_objectives(f, None)
        # ideal = [0,0,0]; extreme = [[2,0,0],[0,2,0],[0,0,2]] = 2*I
        # plane = linalg.solve(2I, ones) = [0.5,0.5,0.5]
        # intercepts (Bug2 fix) = 1/[0.5,...] = [2,2,2]
        np.testing.assert_allclose(intercepts, [2.0, 2.0, 2.0], atol=1e-9)
        # f_norm[0] should be [1,0,0] (extreme point / intercept = [2,0,0]/2)
        np.testing.assert_allclose(f_norm[0], [1.0, 0.0, 0.0], atol=1e-9)


# ===========================================================================
# RNSGA2Comparator Tests
# ===========================================================================
class TestRNSGA2Comparator:
    """Tests for RNSGA2Comparator (R-NSGA-II preference-based ordering)."""

    def test_is_subclass_of_pareto_comparator(self) -> None:
        assert issubclass(RNSGA2Comparator, ParetoComparator)

    def test_sort_population_returns_int_array(self) -> None:
        ref = np.array([[0.0, 1.0], [1.0, 0.0]])
        f = np.array([[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]])
        pop = _make_pop(f)
        comp = RNSGA2Comparator(ref)
        order = comp.sort_population(pop)
        assert np.issubdtype(order.dtype, np.integer)

    def test_sort_population_contains_all_indices(self) -> None:
        ref = np.array([[0.0, 1.0], [1.0, 0.0]])
        f = np.array([[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]])
        pop = _make_pop(f)
        comp = RNSGA2Comparator(ref)
        order = comp.sort_population(pop)
        assert set(order.tolist()) == {0, 1, 2}

    def test_front_0_before_front_1(self) -> None:
        """Non-dominated solutions ranked before dominated ones."""
        ref = np.array([[0.0, 1.0], [1.0, 0.0]])
        f = np.array([[0.0, 1.0], [1.0, 0.0], [2.0, 2.0]])
        pop = _make_pop(f)
        comp = RNSGA2Comparator(ref)
        order = comp.sort_population(pop)
        assert set(order[:2].tolist()) == {0, 1}
        assert order[2] == 2

    def test_infeasible_last(self) -> None:
        ref = np.array([[0.0, 1.0], [1.0, 0.0]])
        f = np.array([[10.0, 10.0], [0.0, 1.0], [1.0, 0.0]])
        cv = np.array([1.0, 0.0, 0.0])
        pop = _make_pop(f, cv)
        comp = RNSGA2Comparator(ref)
        order = comp.sort_population(pop)
        assert order[-1] == 0

    def test_solution_near_reference_point_ranked_first(self) -> None:
        """Within the same front, solution closest to a reference point ranks first."""
        ref = np.array([[0.0, 1.0]])
        # idx 0 is near ref [0,1], idx 1 is far
        f = np.array([[0.1, 0.9], [0.9, 0.1]])  # single front (non-dominated)
        pop = _make_pop(f)
        comp = RNSGA2Comparator(ref)
        order = comp.sort_population(pop)
        assert order[0] == 0

    def test_sort_population_cached(self) -> None:
        ref = np.array([[0.0, 1.0], [1.0, 0.0]])
        f = np.array([[0.0, 1.0], [1.0, 0.0]])
        pop = _make_pop(f)
        comp = RNSGA2Comparator(ref)
        order1 = comp.sort_population(pop)
        order2 = comp.sort_population(pop)
        np.testing.assert_array_equal(order1, order2)

    def test_reference_points_property(self) -> None:
        ref = np.array([[0.0, 1.0], [1.0, 0.0]])
        comp = RNSGA2Comparator(ref)
        np.testing.assert_array_equal(comp.reference_points, ref)

    def test_epsilon_property(self) -> None:
        ref = np.array([[0.5, 0.5]])
        comp = RNSGA2Comparator(ref, epsilon=0.05)
        assert comp.epsilon == 0.05

    def test_direction_maximize(self) -> None:
        """direction=[1,1] selects higher values first."""
        ref = np.array([[0.5, 0.5]])
        f = np.array([[3.0, 3.0], [1.0, 1.0]])
        pop = _make_pop(f)
        comp = RNSGA2Comparator(ref, direction=np.array([1.0, 1.0]))
        order = comp.sort_population(pop)
        assert order[0] == 0  # [3,3] dominates under maximization

    def test_direction_minimize_ref_association(self) -> None:
        """direction=-1: reference point must associate with the correct individual.

        With ref=[0.1, 0.9] (small f1, large f2) the individual [0.1, 0.9]
        should rank before [0.9, 0.1].  Before the Bug1 fix, f*direction=-f
        reversed the normalization and the wrong individual ranked first.
        """
        ref = np.array([[0.1, 0.9]])
        f = np.array([[0.1, 0.9], [0.9, 0.1]])
        pop = _make_pop(f)
        comp = RNSGA2Comparator(ref, direction=np.array([-1.0, -1.0]))
        order = comp.sort_population(pop)
        assert order[0] == 0  # [0.1, 0.9] is closer to ref [0.1, 0.9]

    # compare_population tests

    def test_compare_population_lower_rank_wins(self) -> None:
        """Lower Pareto rank beats higher rank regardless of ref-point distance."""
        ref = np.array([[0.5, 0.5]])
        # idx 0 dominates idx 1 (lower rank); idx 1 might be closer to ref
        f = np.array([[0.0, 0.0], [5.0, 5.0]])
        pop = _make_pop(f)
        comp = RNSGA2Comparator(ref)
        assert comp.compare_population(pop, 0, 1) == -1
        assert comp.compare_population(pop, 1, 0) == 1

    def test_compare_population_same_front_closer_ref_wins(self) -> None:
        """Within the same front, the individual closer to a reference point wins."""
        ref = np.array([[0.0, 1.0]])
        # Both non-dominated; idx 0 is near ref [0,1], idx 1 is far
        f = np.array([[0.1, 0.9], [0.9, 0.1]])
        pop = _make_pop(f)
        comp = RNSGA2Comparator(ref)
        assert comp.compare_population(pop, 0, 1) == -1  # idx 0 closer to ref
        assert comp.compare_population(pop, 1, 0) == 1

    def test_compare_population_feasible_beats_infeasible(self) -> None:
        """A feasible individual always beats an infeasible one."""
        ref = np.array([[0.5, 0.5]])
        f = np.array([[0.0, 0.0], [0.0, 0.0]])
        cv = np.array([1.0, 0.0])  # idx 0 infeasible, idx 1 feasible
        pop = _make_pop(f, cv)
        comp = RNSGA2Comparator(ref)
        assert comp.compare_population(pop, 0, 1) == 1  # infeasible loses
        assert comp.compare_population(pop, 1, 0) == -1  # feasible wins

    def test_compare_population_both_infeasible_lower_cv_wins(self) -> None:
        """Both infeasible: lower constraint violation wins."""
        ref = np.array([[0.5, 0.5]])
        f = np.array([[0.0, 0.0], [0.0, 0.0]])
        cv = np.array([0.5, 2.0])
        pop = _make_pop(f, cv)
        comp = RNSGA2Comparator(ref)
        assert comp.compare_population(pop, 0, 1) == -1
        assert comp.compare_population(pop, 1, 0) == 1

    def test_compare_population_tie(self) -> None:
        """Identical individuals on the same front at the same distance tie."""
        ref = np.array([[0.5, 0.5]])
        f = np.array([[0.5, 0.5], [0.5, 0.5]])
        pop = _make_pop(f)
        comp = RNSGA2Comparator(ref)
        assert comp.compare_population(pop, 0, 1) == 0

    def test_compare_population_three_fronts(self) -> None:
        """Three-level domination: rank 0 < rank 1 < rank 2."""
        ref = np.array([[0.5, 0.5]])
        # idx 0 dominates idx 1 dominates idx 2
        f = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        pop = _make_pop(f)
        comp = RNSGA2Comparator(ref)
        assert comp.compare_population(pop, 0, 1) == -1
        assert comp.compare_population(pop, 1, 2) == -1
        assert comp.compare_population(pop, 0, 2) == -1

    # -----------------------------------------------------------------------
    # Steps 1-2 match the paper's per-reference-point rank minimum
    # -----------------------------------------------------------------------
    def test_step1_2_min_rank_not_niche_count(self) -> None:
        """Ordering follows Deb & Sundar (2006) Steps 1-2, not niche_count.

        All 4 points form a single non-dominated front. A plain sort by each
        individual's distance to its single nearest reference point gives
        idx order ``[2, 3, 1, 0]``. The paper's actual procedure -- rank each
        individual against *every* reference point independently (Step 1)
        and take the minimum rank across reference points (Step 2) -- gives
        idx 1 a better (lower) min-rank than idx 3, producing ``[2, 1, 3,
        0]`` instead. Values verified by direct computation of both
        mechanisms.
        """
        ref = np.array([[0.73, 0.11], [0.39, 0.52]])
        f = np.array(
            [
                [0.09, 0.48],
                [0.24, 0.43],
                [0.58, 0.16],
                [0.80, 0.09],
            ]
        )
        pop = _make_pop(f)
        comp = RNSGA2Comparator(ref)
        order = comp.sort_population(pop)
        np.testing.assert_array_equal(order, [2, 1, 3, 0])


# ===========================================================================
# Comparator cache criterion keys
# ===========================================================================
class TestComparatorCacheCriterionKeys:
    """Population cache entries are keyed by the full ordering criterion."""

    def test_nsga3_does_not_reuse_nsga2_cached_order(self) -> None:
        """NSGA3Comparator sharing sorter+dominator with NSGA2 misses its cache."""
        f = np.array([[0.2, 0.8], [0.8, 0.2], [0.6, 0.6], [0.15, 0.85]])
        refs = np.array([[1.0, 0.0], [0.0, 1.0]])
        pop = _make_pop(f)
        dominator = ParetoDominator()
        n2 = NSGA2Comparator(dominator=dominator)
        n3 = NSGA3Comparator(refs, dominator=dominator, seed=0)
        order_n2 = n2.sort_population(pop)
        order_n3 = n3.sort_population(pop)
        assert order_n3 is not order_n2
        cold = NSGA3Comparator(refs, dominator=dominator, seed=0).sort_population(
            _make_pop(f)
        )
        assert np.array_equal(order_n3, cold)

    def test_rnsga2_does_not_reuse_nsga2_cached_order(self) -> None:
        """RNSGA2Comparator sharing sorter+dominator with NSGA2 misses its cache."""
        f = np.array([[0.1, 0.9], [0.9, 0.1], [0.5, 0.5], [0.11, 0.89]])
        ref = np.array([[0.0, 1.0], [1.0, 0.0]])
        pop = _make_pop(f)
        dominator = ParetoDominator()
        n2 = NSGA2Comparator(dominator=dominator)
        r2 = RNSGA2Comparator(ref, 0.5, dominator=dominator)
        order_n2 = n2.sort_population(pop)
        order_r2 = r2.sort_population(pop)
        assert order_r2 is not order_n2
        cold = RNSGA2Comparator(ref, 0.5, dominator=dominator).sort_population(
            _make_pop(f)
        )
        assert np.array_equal(order_r2, cold)

    def test_nsga2_cache_invalidated_on_eps_cv_change(self) -> None:
        """Changing eps_cv in place on an unchanged population recomputes."""
        f = np.array([[0.0, 0.0], [1.0, 3.0], [3.0, 1.0]])
        cv = np.array([0.5, 0.0, 0.0])
        pop = _make_pop(f, cv)
        comp = NSGA2Comparator()
        order1 = comp.sort_population(pop)
        assert list(order1) == [1, 2, 0]
        comp.eps_cv = 1.0
        order2 = comp.sort_population(pop)
        assert order1 is not order2
        assert list(order2) == [0, 1, 2]

    def test_equivalent_criteria_share_cache_entry(self) -> None:
        """Two comparators with identical criteria reuse one cache entry."""
        f = np.array([[0.0, 1.0], [1.0, 0.0], [0.5, 0.5]])
        pop = _make_pop(f)
        dominator = ParetoDominator()
        c1 = NSGA2Comparator(dominator=dominator)
        c2 = NSGA2Comparator(dominator=dominator)
        order1 = c1.sort_population(pop)
        order2 = c2.sort_population(pop)
        assert order2 is order1

    def test_spea2_fallback_not_shared_across_eps_cv(self) -> None:
        """A second SPEA2Comparator must not read another criterion's fallback."""
        f = np.array([[0.0, 0.0], [1.0, 1.0]])
        cv = np.array([0.5, 0.0])
        pop = _make_pop(f, cv)
        c1 = SPEA2Comparator()
        c1.prepare_population(pop)
        assert list(c1.sort_population(pop)) == [1, 0]

        c2 = SPEA2Comparator(eps_cv=1.0)
        order2 = c2.sort_population(pop)
        cold = SPEA2Comparator(eps_cv=1.0)
        cold.prepare_population(_make_pop(f, cv))
        assert list(order2) == [0, 1]
        assert np.array_equal(order2, cold.sort_population(pop))

    def test_spea2_fallback_invalidated_on_eps_cv_change(self) -> None:
        """Changing eps_cv in place recomputes the fallback fitness ordering."""
        f = np.array([[0.0, 0.0], [1.0, 1.0]])
        cv = np.array([0.5, 0.0])
        pop = _make_pop(f, cv)
        comp = SPEA2Comparator()
        order1 = comp.sort_population(pop)
        assert list(order1) == [1, 0]
        comp.eps_cv = 1.0
        order2 = comp.sort_population(pop)
        assert list(order2) == [0, 1]

    def test_hv_keys_not_shared_across_reference_points(self) -> None:
        """HypervolumeComparators differing in reference point miss hv_keys."""
        f = np.array([[0.0, 1.0], [1.0, 0.0], [0.4, 0.4]])
        pop = _make_pop(f)
        c1 = HypervolumeComparator()
        c2 = HypervolumeComparator(reference_point=np.array([0.5, 0.5]))
        keys1 = c1._keys(pop)
        keys2 = c2._keys(pop)
        assert keys2 is not keys1

    def test_hv_keys_not_shared_across_last_front_only(self) -> None:
        """HypervolumeComparators differing in last_front_only miss hv_keys."""
        f = np.array([[0.0, 1.0], [1.0, 0.0], [0.4, 0.4]])
        pop = _make_pop(f)
        c1 = HypervolumeComparator()
        c2 = HypervolumeComparator(last_front_only=True)
        assert c2._keys(pop) is not c1._keys(pop)

    def test_nsga3_cache_invalidated_on_reference_point_change(self) -> None:
        """NSGA3Comparators differing in reference points miss the sort cache."""
        f = np.array([[0.2, 0.8], [0.8, 0.2], [0.6, 0.6], [0.15, 0.85]])
        pop = _make_pop(f)
        dominator = ParetoDominator()
        c1 = NSGA3Comparator(
            np.array([[1.0, 0.0], [0.0, 1.0]]), dominator=dominator, seed=0
        )
        c2 = NSGA3Comparator(
            np.array([[0.0, 1.0], [1.0, 0.0]]), dominator=dominator, seed=0
        )
        order1 = c1.sort_population(pop)
        order2 = c2.sort_population(pop)
        assert order2 is not order1
        cold = NSGA3Comparator(
            np.array([[0.0, 1.0], [1.0, 0.0]]), dominator=dominator, seed=0
        ).sort_population(_make_pop(f))
        assert np.array_equal(order2, cold)

    def test_nsga3_cache_not_shared_across_instances_different_seeds(self) -> None:
        """NSGA3Comparators differing in seed miss the sort cache."""
        f = _nsga3_tie_population_f()
        refs = np.array([[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]])
        dominator = ParetoDominator()
        pop = _make_pop(f)
        c0 = NSGA3Comparator(refs, dominator=dominator, seed=0)
        c1 = NSGA3Comparator(refs, dominator=dominator, seed=1)
        order0 = c0.sort_population(pop)
        order1 = c1.sort_population(pop)
        assert order1 is not order0
        cold = NSGA3Comparator(
            refs, dominator=ParetoDominator(), seed=1
        ).sort_population(_make_pop(f))
        assert not np.array_equal(order1, order0)
        assert np.array_equal(order1, cold)

    def test_nsga3_cache_not_shared_across_same_seed_advanced_rng(self) -> None:
        """Same-seed instances with different RNG progress miss the sort cache."""
        f = _nsga3_tie_population_f()
        refs = np.array([[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]])
        dominator = ParetoDominator()
        advanced = NSGA3Comparator(refs, dominator=dominator, seed=42)
        independent = NSGA3Comparator(refs, dominator=dominator, seed=42)
        # Consume advanced's RNG on another population before the target one.
        advanced.sort_population(_make_pop(f))
        pop = _make_pop(f)
        advanced_order = advanced.sort_population(pop)
        independent_order = independent.sort_population(pop)
        assert independent_order is not advanced_order
        cold = NSGA3Comparator(
            refs, dominator=ParetoDominator(), seed=42
        ).sort_population(_make_pop(f))
        assert not np.array_equal(independent_order, advanced_order)
        assert np.array_equal(independent_order, cold)

    def test_nsga3_same_instance_reuses_cached_order(self) -> None:
        """Re-sorting an unchanged population returns the cached order."""
        f = _nsga3_tie_population_f()
        refs = np.array([[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]])
        comparator = NSGA3Comparator(refs, seed=42)
        pop = _make_pop(f)
        order1 = comparator.sort_population(pop)
        assert comparator.sort_population(pop) is order1

    def test_spea2_fallback_not_shared_across_dominators(self) -> None:
        """SPEA2Comparators differing in dominator miss the fallback cache."""
        f = np.array([[0.0, 1.0], [0.2, 0.8], [1.0, 0.0], [0.55, 0.55]])
        pop = _make_pop(f)
        order_a = SPEA2Comparator(dominator=EpsilonDominator(0.1)).sort_population(pop)
        order_b = SPEA2Comparator(dominator=EpsilonDominator(1.0)).sort_population(pop)
        cold = SPEA2Comparator(dominator=EpsilonDominator(1.0)).sort_population(
            _make_pop(f)
        )
        assert order_b is not order_a
        assert np.array_equal(order_b, cold)

    def test_nsga2_cache_invalidated_on_dominator_eps_change(self) -> None:
        """In-place eps change of a built-in dominator recomputes the order."""
        f = np.array([[0.0, 1.0], [0.2, 0.8], [1.0, 0.0], [0.55, 0.55]])
        pop = _make_pop(f)
        dominator = EpsilonDominator(0.1)
        comparator = NSGA2Comparator(dominator=dominator)
        order_a = comparator.sort_population(pop)
        dominator.eps[...] = 1.0
        order_b = comparator.sort_population(pop)
        cold = NSGA2Comparator(dominator=EpsilonDominator(1.0)).sort_population(
            _make_pop(f)
        )
        assert order_b is not order_a
        assert np.array_equal(order_b, cold)

    def test_nsga3_cache_invalidated_on_rng_generator_replacement(self) -> None:
        """Replacing the public rng generator recomputes the sort order."""
        f = _nsga3_tie_population_f()
        refs = np.array([[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]])
        comparator = NSGA3Comparator(refs, seed=0)
        pop = _make_pop(f)
        order_a = comparator.sort_population(pop)
        comparator.rng = np.random.default_rng(987654321)
        order_b = comparator.sort_population(pop)
        cold = NSGA3Comparator(refs, seed=987654321).sort_population(_make_pop(f))
        assert order_b is not order_a
        assert np.array_equal(order_b, cold)
