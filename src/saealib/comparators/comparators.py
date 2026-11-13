"""
Comparators module.

This module defines comparator classes and Pareto-related utility functions
for ranking and comparing solutions in single- and multi-objective optimization.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

from saealib.comparators.dominance import (
    Dominator,
    EpsilonDominator,
    ParetoDominator,
)
from saealib.comparators.nds import (
    NonDominatedSorter,
    crowding_distance,  # noqa: F401
    crowding_distance_all_fronts,
    dda_non_dominated_sort,  # noqa: F401
    non_dominated_sort,
    spea2_fitness,
    spea2_truncation_order,
)
from saealib.identity import PopulationAttribute


def _direction_key(direction: np.ndarray | None) -> tuple[float, ...] | None:
    return tuple(direction.tolist()) if direction is not None else None


def _array_key(arr: np.ndarray | None) -> bytes | None:
    return None if arr is None else np.ascontiguousarray(arr).tobytes()


if TYPE_CHECKING:
    from saealib.population import Population
    from saealib.problem import Problem


class Comparator(ABC):
    """
    Base class for comparator.

    Attributes
    ----------
    weights : np.ndarray
        Weights for objectives. shape = (n_obj, )
    eps_cv : float
        Epsilon for constraint violation feasibility threshold. Under
        ``Optimizer`` execution this value is overwritten every generation
        from ``problem.handler.feasibility_threshold``; the constructor
        argument only matters for standalone (non-``Optimizer``) use.
    eps_obj : float
        Epsilon for objective value equality comparison.
    direction : np.ndarray or None
        Per-objective optimization directions (+1 = maximize, -1 = minimize).
        None means all objectives are minimized.
    aligns_population_to_reference_points : bool
        Whether the default population size is aligned to this comparator's
        ``reference_points`` as ``4 * ceil(n_ref / 4)``. The base class
        defaults to False; set it to True on a comparator that needs the default
        population to follow its reference-point layout.
    """

    aligns_population_to_reference_points: bool = False

    @abstractmethod
    def __init__(
        self,
        weights: np.ndarray,
        eps_cv: float,
        eps_obj: float,
        direction: np.ndarray | None = None,
    ):
        self.weights = weights
        self.eps_cv = eps_cv
        self.eps_obj = eps_obj
        self.direction = direction

    @abstractmethod
    def sort_population(self, population: Population) -> np.ndarray:
        """
        Sort population based on their attributes.

        Parameters
        ----------
        population : Population
            The population to sort.

        Returns
        -------
        np.ndarray
            Sorted population indices.
        """
        pass

    @abstractmethod
    def compare_population(self, population: Population, idx_a: int, idx_b: int) -> int:
        """
        Compare individual a and b in population.

        Parameters
        ----------
        population : Population
            The population to compare.
        idx_a : int
            Index of the first individual.
        idx_b : int
            Index of the second individual.

        Returns
        -------
        int
            Comparison result.
            -1 if the individual a is better than the individual b.
            1 if the individual a is worse than the individual b.
            0 if the individual a is equal to the individual b.
        """
        pass

    @abstractmethod
    def compare(self, fa: np.ndarray, cv_a: float, fb: np.ndarray, cv_b: float) -> int:
        """
        Compare two solutions directly from raw values.

        This is the low-overhead variant of ``compare_population`` that avoids
        constructing or indexing into a Population object.

        Parameters
        ----------
        fa : np.ndarray
            Objective values of solution a. shape = (n_obj,)
        cv_a : float
            Constraint violation of solution a.
        fb : np.ndarray
            Objective values of solution b. shape = (n_obj,)
        cv_b : float
            Constraint violation of solution b.

        Returns
        -------
        int
            -1 if a is better than b, 1 if b is better than a, 0 if equal.
        """
        pass

    def _criterion_key(self) -> tuple:
        """Hashable key identifying everything affecting this comparator's ordering.

        Used as part of population cache keys so that entries written by one
        comparator are never reused by another whose ordering criterion
        differs. Overrides must extend ``super()._criterion_key()`` with any
        additional parameters that change the ordering result; arrays enter
        the key in value-based hashable form.
        """
        return (
            type(self).__name__,
            _direction_key(self.direction),
            float(self.eps_cv),
            float(self.eps_obj),
        )

    def get_required_attrs(self, problem: Problem) -> list[PopulationAttribute]:
        """Return population attributes this comparator needs to persist.

        Population-relative ranking state (e.g. SPEA2 fitness) that must
        survive across selections belongs here as a named, typed attribute
        rather than in the population's transient cache. Default: no
        additional attributes.

        Parameters
        ----------
        problem : Problem
            The optimization problem.

        Returns
        -------
        list[PopulationAttribute]
            Empty by default.
        """
        return []

    def prepare_population(self, population: Population) -> None:
        """Freshly recompute and materialize population-relative ranking state.

        Default is a no-op — comparators whose ranking depends only on
        per-individual values already stored in ``f``/``cv`` do not need this.
        Comparators that declare attributes via :meth:`get_required_attrs`
        should override this to recompute and write those attributes for the
        given population, unconditionally (not reusing any prior cached value).

        Parameters
        ----------
        population : Population
            The population to prepare ranking state for.
        """
        return None

    def rank_population(self, population: Population) -> np.ndarray:
        """Prepare population-relative ranking state, then sort.

        Equivalent to calling :meth:`prepare_population` followed by
        :meth:`sort_population`. Callers performing environmental selection on
        a freshly assembled pool (e.g. survivor selection merging population
        and offspring) should call this instead of :meth:`sort_population`
        directly, so any population-relative state declared via
        :meth:`get_required_attrs` is recomputed for the new set before
        ranking.

        Parameters
        ----------
        population : Population
            The population to rank.

        Returns
        -------
        np.ndarray
            Sorted population indices, as returned by
            :meth:`sort_population`.
        """
        self.prepare_population(population)
        return self.sort_population(population)


class SingleObjectiveComparator(Comparator):
    """Comparator for single-objective optimization."""

    def __init__(
        self,
        direction: float | None = None,
        *,
        eps_cv: float = 1e-6,
        eps_obj: float = 1e-6,
    ):
        direction_arr = np.array([direction]) if direction is not None else None
        weights = direction_arr if direction_arr is not None else np.empty(0)
        super().__init__(weights, eps_cv, eps_obj, direction=direction_arr)

    def sort_population(self, population: Population) -> np.ndarray:
        """
        Sort population based on their fitness and constraint violations.

        Parameters
        ----------
        population : Population
            The population to sort.

        Returns
        -------
        np.ndarray
            Sorted population indices.
        """
        f = population.get_array("f")
        cv = population.get_array("cv")
        return self._sort(f, cv)

    def compare_population(self, population: Population, idx_a: int, idx_b: int) -> int:
        """Compare individuals a and b; returns -1/0/1."""
        f = population.get_array("f")
        cv = population.get_array("cv")
        return self.compare(f[idx_a], cv[idx_a], f[idx_b], cv[idx_b])

    def compare(self, fa: np.ndarray, cv_a: float, fb: np.ndarray, cv_b: float) -> int:
        """Compare two solutions directly without a Population object."""
        return self._compare(fa, cv_a, fb, cv_b)

    def _compare(
        self, fitness_a: np.ndarray, cv_a: float, fitness_b: np.ndarray, cv_b: float
    ) -> int:
        """Constraint-domination comparison; -1=a better, 1=b better, 0=equal."""
        if cv_a > self.eps_cv and cv_b > self.eps_cv:
            if cv_a < cv_b:
                return -1
            elif cv_a > cv_b:
                return 1
            else:
                return 0
        elif cv_a > self.eps_cv and cv_b <= self.eps_cv:
            return 1
        elif cv_a <= self.eps_cv and cv_b > self.eps_cv:
            return -1
        else:
            d = self.direction[0] if self.direction is not None else -1.0
            sa = fitness_a[0] * d
            sb = fitness_b[0] * d
            if sa > sb + self.eps_obj:
                return -1
            elif sa < sb - self.eps_obj:
                return 1
            else:
                return 0

    def _sort(self, fitness: np.ndarray, cv: np.ndarray) -> np.ndarray:
        """
        Sort solutions based on their fitness and constraint violations.

        Parameters
        ----------
        fitness : np.ndarray
            Objective values of solutions. shape = (n_individuals, n_obj)
        cv : np.ndarray
            Constraint violations of solutions. shape = (n_individuals, )

        Returns
        -------
        np.ndarray
            Sorted indices of the solutions.
        """
        d = self.direction[0] if self.direction is not None else -1.0
        cv_key = np.where(cv > self.eps_cv, cv, 0)
        obj_key = fitness.flatten() * d
        return np.lexsort((-obj_key, cv_key))


class WeightedSumComparator(Comparator):
    """
    Comparator for multi-objective optimization via weighted scalarization.

    Aggregates multiple objectives into a single scalar using a weighted
    dot product: score = f @ weights. The sign convention for weights
    follows the same pattern as SingleObjectiveComparator: use negative
    weights for minimization (e.g., weights=np.array([-1.0, -1.0])).

    Also supports single-objective problems (n_obj == 1).

    Parameters
    ----------
    direction : np.ndarray
        Per-objective optimization directions (±1). shape = (n_obj,)
    """

    def __init__(
        self,
        direction: np.ndarray | None = None,
        *,
        eps_cv: float = 1e-6,
        eps_obj: float = 1e-6,
    ):
        if direction is None:
            raise TypeError(
                "WeightedSumComparator() missing required argument: 'direction'"
            )
        _dir = np.asarray(direction, dtype=float)
        super().__init__(_dir, eps_cv, eps_obj, direction=_dir)

    def sort_population(self, population: Population) -> np.ndarray:
        """Sort population by weighted sum of objectives, with feasibility first."""
        f = population.get_array("f")  # (n_ind, n_obj)
        cv = population.get_array("cv")  # (n_ind,)
        scalar = f @ self.direction  # (n_ind,) weighted sum per individual
        cv_key = np.where(cv > self.eps_cv, cv, 0)
        return np.lexsort((-scalar, cv_key))

    def compare_population(self, population: Population, idx_a: int, idx_b: int) -> int:
        """Compare two individuals by weighted sum; -1=a better, 1=b better, 0=equal."""
        f = population.get_array("f")
        cv = population.get_array("cv")
        return self.compare(f[idx_a], float(cv[idx_a]), f[idx_b], float(cv[idx_b]))

    def compare(self, fa: np.ndarray, cv_a: float, fb: np.ndarray, cv_b: float) -> int:
        """Compare two solutions directly without a Population object."""
        return self._compare(fa, cv_a, fb, cv_b)

    def _compare(self, fa: np.ndarray, cv_a: float, fb: np.ndarray, cv_b: float) -> int:
        if cv_a > self.eps_cv and cv_b > self.eps_cv:
            if cv_a < cv_b:
                return -1
            elif cv_a > cv_b:
                return 1
            else:
                return 0
        elif cv_a > self.eps_cv:
            return 1
        elif cv_b > self.eps_cv:
            return -1
        assert self.direction is not None
        sa = float(np.dot(fa, self.direction))
        sb = float(np.dot(fb, self.direction))
        if sa > sb + self.eps_obj:
            return -1
        elif sa < sb - self.eps_obj:
            return 1
        return 0


class ParetoComparator(Comparator):
    """
    Comparator for multi-objective optimization via non-dominated sorting only.

    Implements rank-only Pareto ordering:

    - sort_population: Pareto front rank order (no crowding distance tiebreaking).
      Infeasible individuals (cv > eps) are always ranked after feasible ones,
      ordered by ascending constraint violation.
    - compare_population: Pareto dominance (-1 = a dominates b, 1 = b
      dominates a, 0 = non-dominated). Infeasibility is handled with the
      standard constraint-domination rule.

    NaN objective values are treated as non-dominating and placed last within
    the feasible block (consistent with non_dominated_sort).

    This class is a concrete base for Pareto-based comparators. NSGA2Comparator
    inherits from it and overrides sort_population to add crowding-distance-based
    secondary ordering within each front.

    Parameters
    ----------
    direction : np.ndarray or None
        Per-objective optimization directions (+1 = maximize, -1 = minimize).
        None means all objectives are minimized (standard Pareto dominance).
    """

    def __init__(
        self,
        direction: np.ndarray | None = None,
        *,
        eps_cv: float = 1e-6,
        eps_obj: float = 1e-6,
        sorter: NonDominatedSorter = non_dominated_sort,
        dominator: Dominator | None = None,
    ):
        direction_arr = (
            np.asarray(direction, dtype=float) if direction is not None else None
        )
        super().__init__(np.empty(0), eps_cv, eps_obj, direction=direction_arr)
        self._sorter = sorter
        # Use None-sentinel to avoid a shared mutable default.
        self._dominator: Dominator = (
            dominator if dominator is not None else ParetoDominator()
        )

    def _criterion_key(self) -> tuple:
        return (
            *super()._criterion_key(),
            self._sorter,
            self._dominator,
            self._dominator._criterion_key(),
        )

    @property
    def sorter(self) -> NonDominatedSorter:
        """The non-dominated sorting callable used by this comparator."""
        return self._sorter

    @property
    def dominator(self) -> Dominator:
        """The dominance predicate used by this comparator."""
        return self._dominator

    def sort_population(self, population: Population) -> np.ndarray:
        """Sort by Pareto front rank; infeasible individuals come last."""
        f = population.get_array("f")
        cv = population.get_array("cv")
        feasible = np.where(cv <= self.eps_cv)[0]
        infeasible = np.where(cv > self.eps_cv)[0]

        sorted_feasible = np.empty(0, int)
        if len(feasible):
            ranks, _ = self._sorter(
                f[feasible], direction=self.direction, dominator=self._dominator
            )
            order = np.argsort(ranks, kind="stable")
            sorted_feasible = feasible[order]

        sorted_infeasible = np.empty(0, int)
        if len(infeasible):
            sorted_infeasible = infeasible[np.argsort(cv[infeasible])]

        return np.concatenate([sorted_feasible, sorted_infeasible]).astype(int)

    def compare_population(self, population: Population, idx_a: int, idx_b: int) -> int:
        """Compare via Pareto dominance; -1=a dominates, 1=b dominates, 0=equal."""
        f = population.get_array("f")
        cv = population.get_array("cv")
        return self.compare(f[idx_a], float(cv[idx_a]), f[idx_b], float(cv[idx_b]))

    def compare(self, fa: np.ndarray, cv_a: float, fb: np.ndarray, cv_b: float) -> int:
        """Compare two solutions directly without a Population object."""
        if cv_a > self.eps_cv and cv_b > self.eps_cv:
            if cv_a < cv_b:
                return -1
            elif cv_a > cv_b:
                return 1
            return 0
        elif cv_a > self.eps_cv:
            return 1
        elif cv_b > self.eps_cv:
            return -1

        if self._dominator.dominates(fa, fb, self.direction):
            return -1
        if self._dominator.dominates(fb, fa, self.direction):
            return 1
        return 0


class NSGA2Comparator(ParetoComparator):
    """
    Comparator for multi-objective optimization via NSGA-II style ranking.

    Extends ParetoComparator with crowding-distance-based secondary ordering:

    - sort_population: non-dominated sorting + crowding distance
    - compare_population: overrides ParetoComparator with Deb et al. (2002)'s
      crowded-comparison operator (prec_n): lower rank wins, ties broken by
      higher crowding distance -- see :meth:`compare_population`

    Infeasible individuals (cv > eps) are always ranked after feasible
    ones, ordered by ascending constraint violation.

    The sort result is cached in the Population cache and automatically
    invalidated when the population is modified.

    Parameters
    ----------
    direction : np.ndarray or None
        Per-objective optimization directions (+1 = maximize, -1 = minimize).
        None means all objectives are minimized.
    """

    def __init__(
        self,
        direction: np.ndarray | None = None,
        *,
        eps_cv: float = 1e-6,
        eps_obj: float = 1e-6,
        sorter: NonDominatedSorter = non_dominated_sort,
        dominator: Dominator | None = None,
    ):
        super().__init__(
            direction,
            eps_cv=eps_cv,
            eps_obj=eps_obj,
            sorter=sorter,
            dominator=dominator,
        )

    def sort_population(self, population: Population) -> np.ndarray:
        """Sort by Pareto front rank then crowding distance (NSGA-II style)."""
        _cache_key = ("pareto_sort", self._criterion_key())
        cached = population.get_cache(_cache_key)
        if cached is not None:
            return cached

        f = population.get_array("f")  # (n, n_obj)
        cv = population.get_array("cv")  # (n,)
        feasible = np.where(cv <= self.eps_cv)[0]
        infeasible = np.where(cv > self.eps_cv)[0]

        rank_full = np.full(len(population), np.inf)
        cd_full = np.zeros(len(population))

        sorted_feasible = np.empty(0, int)
        if len(feasible):
            ranks, fronts = self._sorter(
                f[feasible], direction=self.direction, dominator=self._dominator
            )
            cd = crowding_distance_all_fronts(f[feasible], fronts)
            order = np.lexsort((-cd, ranks))
            sorted_feasible = feasible[order]
            rank_full[feasible] = ranks
            cd_full[feasible] = cd

        sorted_infeasible = np.empty(0, int)
        if len(infeasible):
            sorted_infeasible = infeasible[np.argsort(cv[infeasible])]

        result = np.concatenate([sorted_feasible, sorted_infeasible]).astype(int)
        population.set_cache(_cache_key, result)
        _rc_key = ("nsga2_rank_cd", self._criterion_key())
        population.set_cache(_rc_key, (rank_full, cd_full))
        return result

    def compare_population(self, population: Population, idx_a: int, idx_b: int) -> int:
        """Compare via the NSGA-II crowded comparison operator.

        Implements the partial order defined by Deb et al. (2002) (Section
        III-B.2, "Crowded-Comparison Operator", operator prec_n): prefer
        the lower (better) nondomination rank; if ranks are equal, prefer
        the individual with the larger crowding distance.

        References
        ----------
        :cite:`deb2002nsga2`: Deb, K., Pratap, A., Agarwal, S., &
        Meyarivan, T. (2002). A fast and elitist multiobjective genetic
        algorithm: NSGA-II. *IEEE Transactions on Evolutionary
        Computation*, 6(2), 182-197.
        """
        cv = population.get_array("cv")
        cv_a = float(cv[idx_a])
        cv_b = float(cv[idx_b])

        if cv_a > self.eps_cv and cv_b > self.eps_cv:
            if cv_a < cv_b:
                return -1
            if cv_a > cv_b:
                return 1
            return 0
        if cv_a > self.eps_cv:
            return 1
        if cv_b > self.eps_cv:
            return -1

        # Both feasible: rank first, then crowding distance (higher = better).
        self.sort_population(population)
        _rc_key = ("nsga2_rank_cd", self._criterion_key())
        cached = population.get_cache(_rc_key)
        assert cached is not None
        rank_full, cd_full = cached
        rank_a, rank_b = rank_full[idx_a], rank_full[idx_b]
        if rank_a < rank_b:
            return -1
        if rank_a > rank_b:
            return 1
        cd_a, cd_b = cd_full[idx_a], cd_full[idx_b]
        if cd_a > cd_b:
            return -1
        if cd_a < cd_b:
            return 1
        return 0


class SPEA2Comparator(Comparator):
    """
    Comparator implementing SPEA2 fitness-based ranking (Zitzler et al., 2001).

    SPEA2 fitness ``F(i) = R(i) + D(i)`` is computed over the **entire feasible
    set** and used as the ranking criterion — lower is better.  Because the
    fitness depends on the whole population, pairwise comparison of two isolated
    points is undefined.

    Ordering rules:

    - **Feasible block**: the non-dominated subset (``F(i) < 1``) is ordered
      by the iterative nearest-neighbour truncation procedure -- see
      :func:`spea2_truncation_order` --
      instead of a one-shot sort, so that boundary/extreme solutions are
      preferentially retained under truncation.  The dominated subset
      (``F(i) >= 1``) follows, sorted by ascending SPEA2 fitness (lower =
      better), matching the paper's "fill with the best dominated
      individuals" branch.  See :func:`spea2_fitness` for the ``k = √N``
      density reduction note.
    - **Infeasible block**: always placed after feasible individuals, ordered by
      ascending constraint violation (Deb 2000 feasibility rule).

    .. note::
        ``compare()`` raises :exc:`NotImplementedError` because SPEA2 fitness is
        population-relative.  Components that require pairwise comparison (e.g.
        PSO pbest update, ``PairwiseComparisonSet``) should use a
        :class:`ParetoComparator` instead.  Tournament selection should call
        ``compare_population()``, which IS defined and safe to use.

    Parameters
    ----------
    direction : np.ndarray or None
        Per-objective optimization directions (+1 = maximize, -1 = minimize).
        None defaults to minimization for all objectives.
    eps_cv : float
        Feasibility threshold for constraint violation.
    eps_obj : float
        Epsilon for objective-value equality (stored for interface compatibility).
    dominator : Dominator or None
        Dominance predicate.  ``None`` defaults to :class:`ParetoDominator`.

    References
    ----------
    :cite:`zitzler2001spea2`: Zitzler, E., Laumanns, M., & Thiele, L. (2001).
    SPEA2: Improving the Strength Pareto Evolutionary Algorithm. TIK-Report
    103, ETH Zurich.

    :cite:`deb2000feasibility`: Deb, K. (2000). An efficient constraint
    handling method for genetic algorithms. Computer Methods in Applied
    Mechanics and Engineering, 186(2-4), 311-338.
    """

    is_population_relative: bool = True
    """Marker indicating that ``compare()`` is unavailable for this comparator."""

    def __init__(
        self,
        direction: np.ndarray | None = None,
        *,
        eps_cv: float = 1e-6,
        eps_obj: float = 1e-6,
        dominator: Dominator | None = None,
    ):
        direction_arr = (
            np.asarray(direction, dtype=float) if direction is not None else None
        )
        super().__init__(np.empty(0), eps_cv, eps_obj, direction=direction_arr)
        self._dominator: Dominator = (
            dominator if dominator is not None else ParetoDominator()
        )

    @property
    def dominator(self) -> Dominator:
        """The dominance predicate used by this comparator."""
        return self._dominator

    def _criterion_key(self) -> tuple:
        return (
            *super()._criterion_key(),
            self._dominator,
            self._dominator._criterion_key(),
        )

    def get_required_attrs(self, problem: Problem) -> list[PopulationAttribute]:
        """Declare persistent SPEA2 fitness storage."""
        return [PopulationAttribute("spea2_fitness", np.float64, (), default=np.nan)]

    def _compute_fitness(self, population: Population) -> np.ndarray:
        """Compute SPEA2 fitness for the current population."""
        f_arr = population.get_array("f")
        cv_arr = population.get_array("cv")
        n = len(f_arr)
        fitness_all = np.full(n, np.inf)
        feasible = np.where(cv_arr <= self.eps_cv)[0]
        if len(feasible):
            f_feasible = spea2_fitness(
                f_arr[feasible], direction=self.direction, dominator=self._dominator
            )
            nan_mask = np.isnan(f_arr[feasible]).any(axis=1)
            f_feasible = np.where(nan_mask, np.inf, f_feasible)
            fitness_all[feasible] = f_feasible
        return fitness_all

    def _fallback_fitness_key(self) -> tuple | None:
        key = ("spea2_fitness_fallback", self._criterion_key())
        try:
            hash(key)
        except TypeError:
            # An unhashable custom dominator cannot be keyed: recompute fresh.
            return None
        return key

    def prepare_population(self, population: Population) -> None:
        """Recompute and store SPEA2 fitness for the current population."""
        fitness = self._compute_fitness(population)
        if "spea2_fitness" in population.schema:
            population.update_array("spea2_fitness", fitness)
        else:
            key = self._fallback_fitness_key()
            if key is not None:
                population.set_cache(key, fitness)

    def _assigned_fitness(self, population: Population) -> np.ndarray | None:
        """Return persisted fitness when every active row has a value."""
        if "spea2_fitness" not in population.schema:
            return None
        values = population.get_array("spea2_fitness")
        if np.isnan(values).any():
            return None
        return values

    def _fitness(self, population: Population) -> np.ndarray:
        """Return persisted SPEA2 fitness or compute it through the fallback."""
        assigned = self._assigned_fitness(population)
        if assigned is not None:
            return assigned
        key = self._fallback_fitness_key()
        if key is not None:
            cached = population.get_cache(key)
            if cached is not None:
                return cached
        fitness_all = self._compute_fitness(population)
        if key is not None:
            population.set_cache(key, fitness_all)
        return fitness_all

    def sort_population(self, population: Population) -> np.ndarray:
        """
        Sort feasible individuals via SPEA2 environmental selection.

        Within the feasible block, the non-dominated subset (``F(i) < 1``)
        is ordered by :func:`spea2_truncation_order` (Zitzler et al. 2001),
        and the dominated subset (``F(i) >= 1``) follows,
        ordered by ascending SPEA2 fitness. Infeasible individuals come
        last, ordered by ascending constraint violation.

        Parameters
        ----------
        population : Population
            The population to sort.

        Returns
        -------
        np.ndarray
            Sorted population indices (int).
        """
        f_arr = population.get_array("f")
        cv_arr = population.get_array("cv")
        fitness_all = self._fitness(population)

        feasible = np.where(cv_arr <= self.eps_cv)[0]
        infeasible = np.where(cv_arr > self.eps_cv)[0]

        sorted_feasible: np.ndarray = np.empty(0, int)
        if len(feasible):
            fitness_feasible = fitness_all[feasible]
            nondominated_local = np.where(fitness_feasible < 1.0)[0]
            dominated_local = np.where(fitness_feasible >= 1.0)[0]

            truncation_order = np.empty(0, int)
            if len(nondominated_local):
                truncation_order = nondominated_local[
                    spea2_truncation_order(f_arr[feasible][nondominated_local])
                ]

            dominated_order = dominated_local[
                np.argsort(fitness_feasible[dominated_local], kind="stable")
            ]

            order = np.concatenate([truncation_order, dominated_order]).astype(int)
            sorted_feasible = feasible[order]

        sorted_infeasible: np.ndarray = np.empty(0, int)
        if len(infeasible):
            inf_order = np.argsort(cv_arr[infeasible], kind="stable")
            sorted_infeasible = infeasible[inf_order]

        return np.concatenate([sorted_feasible, sorted_infeasible]).astype(int)

    def compare_population(self, population: Population, idx_a: int, idx_b: int) -> int:
        """
        Compare two individuals using constraint-domination then SPEA2 fitness.

        Returns ``-1`` if ``a`` is better, ``1`` if ``b`` is better, ``0`` if
        equal.  The feasibility rule (Deb 2000) is applied first: feasible
        individuals always beat infeasible ones.  Among both-infeasible pairs,
        lower constraint violation wins.  Among both-feasible pairs, lower SPEA2
        fitness wins.

        Parameters
        ----------
        population : Population
            The population containing both individuals.
        idx_a : int
            Index of the first individual.
        idx_b : int
            Index of the second individual.

        Returns
        -------
        int
            ``-1``, ``0``, or ``1``.
        """
        cv_arr = population.get_array("cv")
        cv_a = float(cv_arr[idx_a])
        cv_b = float(cv_arr[idx_b])

        # Both infeasible: lower cv wins
        if cv_a > self.eps_cv and cv_b > self.eps_cv:
            if cv_a < cv_b:
                return -1
            elif cv_a > cv_b:
                return 1
            return 0

        # One infeasible: feasible wins
        if cv_a > self.eps_cv:
            return 1
        if cv_b > self.eps_cv:
            return -1

        # Both feasible: lower SPEA2 fitness wins
        fitness_all = self._fitness(population)
        fa = fitness_all[idx_a]
        fb = fitness_all[idx_b]
        if fa < fb:
            return -1
        elif fa > fb:
            return 1
        return 0

    def compare(
        self,
        fa: np.ndarray,
        cv_a: float,
        fb: np.ndarray,
        cv_b: float,
    ) -> int:
        """Raise NotImplementedError — SPEA2 fitness is population-relative."""
        raise NotImplementedError(
            "SPEA2Comparator.compare() is undefined: SPEA2 fitness is "
            "population-relative and cannot be computed from two isolated "
            "points. Use compare_population() / sort_population(), or supply "
            "a ParetoComparator for components that require pairwise "
            "compare() (e.g. PSO pbest update, PairwiseComparisonSet)."
        )


def _hv_default_reference_point(
    f: np.ndarray, direction: np.ndarray | None
) -> np.ndarray:
    """Beume et al. (2007) default hypervolume reference point.

    the boundary-solution handling described in the cited work: the reference point is
    the vector of the currently worst objective values increased by 1.0,
    recomputed every generation. For a maximized objective, "worst"
    (the direction to move *away* from) is the minimum observed value, and
    the offset is subtracted instead of added.

    Parameters
    ----------
    f : np.ndarray
        Objective matrix to derive the worst values from. shape: (n, n_obj).
    direction : np.ndarray or None
        Per-objective optimization direction: ``+1`` maximize, ``-1``
        minimize. ``None`` defaults to minimization for all objectives.

    Returns
    -------
    np.ndarray
        Reference point in the original objective space. shape: (n_obj,).
    """
    n_obj = f.shape[1]
    dir_arr = direction if direction is not None else np.full(n_obj, -1.0)
    worst = np.where(dir_arr == 1, f.min(axis=0), f.max(axis=0))
    offset = np.where(dir_arr == 1, -1.0, 1.0)
    return worst + offset


class HypervolumeComparator(ParetoComparator):
    """Comparator using front rank and exclusive HV contribution (SMS-EMOA style).

    Ordering rules:

    - **Feasibility first** (Deb 2000): infeasible individuals (cv > eps_cv)
      are placed after all feasible ones, ordered by ascending constraint
      violation.
    - **Primary key**: non-dominated front rank (ascending, lower = better).
    - **Secondary key** (within a front): exclusive hypervolume contribution
      (descending, higher = better).

    .. note::
        **Generalization from SMS-EMOA.** The original SMS-EMOA
        (Beume et al., 2007) computes HV contributions only on the *last*
        (worst) front to determine the single removal candidate at each
        generation.  By default this comparator applies HV-contribution
        ordering *within every front* to produce a full ranking over the
        entire population, a deliberate generalization that enables use in
        standard survivor-selection and tournament-selection contexts.  Set
        ``last_front_only=True`` for the paper's scope instead: contributions
        are then computed on the last front alone and better fronts are
        ordered by rank only.

    .. warning::
        Computing hypervolume is exponential in the number of objectives, and
        :func:`~saealib.utils.indicators.hypervolume_contributions` performs
        O(N) hypervolume evaluations per front (leave-one-out).  For large
        populations or many objectives this becomes expensive.

    .. note::
        ``compare()`` raises :exc:`NotImplementedError` because the
        hypervolume contribution of a point depends on the other points in
        the population (it is population-relative).  Components that require
        pairwise comparison (e.g. PSO pbest update,
        ``PairwiseComparisonSet``) should use a :class:`ParetoComparator`
        instead.  Tournament selection should call ``compare_population()``,
        which IS defined and safe to use.

    .. note::
        **Deferred gap.** Beume et al. (2007) explicitly special-cases
        two-objective problems: the reference point is skipped entirely and
        the two extremal (best-per-objective) solutions are unconditionally
        kept.  This comparator always computes a reference point and HV
        contributions for every point, including the two-objective case; the
        paper's special case is not implemented here.

    Parameters
    ----------
    direction : np.ndarray or None
        Per-objective optimization directions (+1 = maximize, -1 = minimize).
        None defaults to minimization for all objectives.
    eps_cv : float
        Feasibility threshold for constraint violation.
    eps_obj : float
        Epsilon for objective-value equality (stored for interface
        compatibility).
    reference_point : np.ndarray or None
        Reference point in the *original* objective space, shape
        ``(n_obj,)``.  If ``None`` (default), it is recomputed fresh on
        every call as the front's worst objective values plus ``1.0``
        (Beume et al. 2007, boundary-solution handling) -- see
        :func:`_hv_default_reference_point`.
    margin : float
        Unused by the default (``reference_point=None``) computation, which
        now follows the paper's fixed ``+1.0`` offset instead of a
        fractional one.  Retained only so existing constructor calls do not
        break; kept as a stored, inert attribute (see :attr:`margin`).
        Pass an explicit ``reference_point`` if a margin-scaled reference
        point is needed.
    last_front_only : bool
        Restrict HV-contribution ordering to the last (worst) front, as in
        Beume et al. (2007).  Individuals in better fronts then share one
        contribution value and keep their relative order.  Defaults to
        ``False``, which ranks within every front.  Setting it also removes
        the per-front hypervolume cost for all but the last front.
    sorter : NonDominatedSorter
        Non-dominated sorting callable.
    dominator : Dominator or None
        Dominance predicate.  ``None`` defaults to :class:`ParetoDominator`.

    References
    ----------
    :cite:`beume2007smsemoa`: Beume, N., Naujoks, B., & Emmerich, M. (2007).
    SMS-EMOA: Multiobjective selection based on dominated hypervolume.
    *European Journal of Operational Research*, 181(3), 1653-1669.

    :cite:`deb2000feasibility`: Deb, K. (2000). An efficient constraint
    handling method for genetic algorithms. *Computer Methods in Applied
    Mechanics and Engineering*, 186(2-4), 311-338.
    """

    is_population_relative: bool = True
    """Marker indicating that ``compare()`` is unavailable for this comparator."""

    def __init__(
        self,
        direction: np.ndarray | None = None,
        *,
        eps_cv: float = 1e-6,
        eps_obj: float = 1e-6,
        reference_point: np.ndarray | None = None,
        margin: float = 0.1,
        last_front_only: bool = False,
        sorter: NonDominatedSorter = non_dominated_sort,
        dominator: Dominator | None = None,
    ):
        super().__init__(
            direction,
            eps_cv=eps_cv,
            eps_obj=eps_obj,
            sorter=sorter,
            dominator=dominator,
        )
        self._reference_point = (
            None
            if reference_point is None
            else np.asarray(reference_point, dtype=float)
        )
        self._margin = float(margin)
        self._last_front_only = bool(last_front_only)

    @property
    def reference_point(self) -> np.ndarray | None:
        """Reference point used for hypervolume computation, or None for auto."""
        return self._reference_point

    @property
    def last_front_only(self) -> bool:
        """Whether HV contributions are restricted to the last front."""
        return self._last_front_only

    @property
    def margin(self) -> float:
        """Unused by the default reference-point computation; see class docstring."""
        return self._margin

    def _keys(self, population: Population) -> tuple[np.ndarray, np.ndarray]:
        """Return per-individual ``(rank_all, contrib_all)`` arrays, cached.

        Arrays have length N (population size).  Infeasible individuals
        receive ``+inf`` rank and ``-inf`` contribution so they sort last.
        Results are cached under ``("hv_keys", criterion_key)``.
        """
        cached = population.get_cache(("hv_keys", self._criterion_key()))
        if cached is not None:
            return cached  # type: ignore[return-value]  # cached value is Any; runtime type matches return annotation

        # Lazy import avoids a circular dependency at module load time.
        from saealib.utils.indicators import hypervolume_contributions

        f_arr = population.get_array("f")
        cv_arr = population.get_array("cv")
        n = len(f_arr)

        rank_all = np.full(n, np.inf)  # infeasible → +inf (worst rank)
        contrib_all = np.full(n, -np.inf)  # infeasible → -inf (worst contrib)

        feasible = np.where(cv_arr <= self.eps_cv)[0]
        if len(feasible):
            ranks, fronts = self._sorter(
                f_arr[feasible],
                direction=self.direction,
                dominator=self._dominator,
            )
            rank_all[feasible] = ranks

            if self._last_front_only:
                contrib_all[feasible] = 0.0
                fronts = fronts[-1:]

            for front in fronts:  # front = local indices into feasible subset
                local = np.asarray(front, dtype=int)
                gidx = feasible[local]
                front_f = f_arr[gidx]
                ref_point = (
                    self._reference_point
                    if self._reference_point is not None
                    else _hv_default_reference_point(front_f, self.direction)
                )
                contribs = hypervolume_contributions(
                    front_f,
                    reference_point=ref_point,
                    direction=self.direction,
                )
                contrib_all[gidx] = contribs

        keys = (rank_all, contrib_all)
        population.set_cache(("hv_keys", self._criterion_key()), keys)
        return keys

    def _criterion_key(self) -> tuple:
        return (
            *super()._criterion_key(),
            _array_key(self._reference_point),
            self._last_front_only,
        )

    def sort_population(self, population: Population) -> np.ndarray:
        """
        Sort by front rank then HV contribution; infeasible individuals last.

        Within the feasible block the primary sort key is non-dominated front
        rank (ascending) and the secondary key is exclusive hypervolume
        contribution (descending).  Infeasible individuals follow, sorted by
        ascending constraint violation.

        Parameters
        ----------
        population : Population
            The population to sort.

        Returns
        -------
        np.ndarray
            Sorted population indices (int).
        """
        cv_arr = population.get_array("cv")
        rank_all, contrib_all = self._keys(population)

        feasible = np.where(cv_arr <= self.eps_cv)[0]
        infeasible = np.where(cv_arr > self.eps_cv)[0]

        sorted_feasible: np.ndarray = np.empty(0, int)
        if len(feasible):
            # lexsort: last key is primary -> rank ascending, then -contrib ascending
            order = np.lexsort((-contrib_all[feasible], rank_all[feasible]))
            sorted_feasible = feasible[order]

        sorted_infeasible: np.ndarray = np.empty(0, int)
        if len(infeasible):
            sorted_infeasible = infeasible[np.argsort(cv_arr[infeasible])]

        return np.concatenate([sorted_feasible, sorted_infeasible]).astype(int)

    def compare_population(self, population: Population, idx_a: int, idx_b: int) -> int:
        """
        Compare two individuals using constraint-domination then HV ranking.

        Returns ``-1`` if ``a`` is better, ``1`` if ``b`` is better, ``0``
        if equal.  The feasibility rule (Deb 2000) is applied first.  Among
        both-feasible pairs, lower front rank wins; within the same front,
        higher HV contribution wins.

        Parameters
        ----------
        population : Population
            The population containing both individuals.
        idx_a : int
            Index of the first individual.
        idx_b : int
            Index of the second individual.

        Returns
        -------
        int
            ``-1``, ``0``, or ``1``.
        """
        cv_arr = population.get_array("cv")
        cv_a = float(cv_arr[idx_a])
        cv_b = float(cv_arr[idx_b])

        # Both infeasible: lower cv wins
        if cv_a > self.eps_cv and cv_b > self.eps_cv:
            if cv_a < cv_b:
                return -1
            elif cv_a > cv_b:
                return 1
            return 0

        # One infeasible: feasible wins
        if cv_a > self.eps_cv:
            return 1
        if cv_b > self.eps_cv:
            return -1

        # Both feasible: lower rank wins; tie-break on higher HV contribution
        rank_all, contrib_all = self._keys(population)
        ra = rank_all[idx_a]
        rb = rank_all[idx_b]
        if ra < rb:
            return -1
        elif ra > rb:
            return 1

        ca = contrib_all[idx_a]
        cb = contrib_all[idx_b]
        if ca > cb:
            return -1
        elif ca < cb:
            return 1
        return 0

    def compare(
        self,
        fa: np.ndarray,
        cv_a: float,
        fb: np.ndarray,
        cv_b: float,
    ) -> int:
        """Raise NotImplementedError — HV contribution is population-relative."""
        raise NotImplementedError(
            "HypervolumeComparator.compare() is undefined: the exclusive "
            "hypervolume contribution of a point is population-relative and "
            "cannot be computed from two isolated points.  Use "
            "compare_population() / sort_population(), or supply a "
            "ParetoComparator for components that require pairwise "
            "compare() (e.g. PSO pbest update, PairwiseComparisonSet)."
        )


class EpsilonDominanceComparator(ParetoComparator):
    """
    Comparator using ε-box dominance instead of standard Pareto dominance.

    Wraps :class:`EpsilonDominator` and injects it into the
    :class:`ParetoComparator` dominance seam via the ``dominator=`` argument.
    Front ranking, infeasibility handling, and constraint-domination logic are
    all inherited unchanged from :class:`ParetoComparator`.

    The ε-dominance relation is defined in:

        Laumanns, M., Thiele, L., Deb, K., & Zitzler, E. (2002).
        Combining convergence and diversity in evolutionary multiobjective
        optimization. *Evolutionary Computation*, 10(3), 263-282.

    Each objective axis is divided into ε-boxes of width ``eps``.  Two
    solutions that fall in the **same** ε-box are mutually non-dominating;
    a solution whose box index is strictly better in *every* objective
    dominates the other.  As ``eps → 0`` the relation recovers ordinary
    Pareto dominance.

    .. note::
        ε-box **representative selection** (the archive rule that keeps one
        solution per box — cf. Deb, Mohan & Mishra (2005), ε-MOEA) is **not**
        handled here.  That is an archive-truncation responsibility and will
        be added separately.

    Parameters
    ----------
    eps : float or np.ndarray
        Box size(s).  A scalar broadcasts to all objectives; an array of
        shape ``(n_obj,)`` sets per-objective widths.  All values must be
        strictly positive (> 0).
    mode : {"additive", "multiplicative"}
        Quantization mode passed to :class:`EpsilonDominator`.
        ``"additive"`` (default): box index = ``floor(f_i / eps_i)``.
        ``"multiplicative"``: box index = ``floor(log f_i / log(1 + eps_i))``;
        requires strictly positive objective values.
    direction : np.ndarray or None
        See :class:`ParetoComparator`.
    eps_cv : float
        See :class:`ParetoComparator`.
    eps_obj : float
        See :class:`ParetoComparator`.
    sorter : NonDominatedSorter
        See :class:`ParetoComparator`.
    """

    def __init__(
        self,
        eps: float | np.ndarray,
        mode: str = "additive",
        direction: np.ndarray | None = None,
        *,
        eps_cv: float = 1e-6,
        eps_obj: float = 1e-6,
        sorter: NonDominatedSorter = non_dominated_sort,
    ):
        super().__init__(
            direction,
            eps_cv=eps_cv,
            eps_obj=eps_obj,
            sorter=sorter,
            dominator=EpsilonDominator(eps, mode),
        )

    @property
    def eps(self) -> float | np.ndarray:
        """Box size(s) used by the underlying EpsilonDominator."""
        return self._dominator.eps  # type: ignore  # eps defined in concrete subclass; not visible through abstract base

    @property
    def mode(self) -> str:
        """Quantization mode of the underlying EpsilonDominator."""
        return self._dominator.mode  # type: ignore  # mode defined in concrete subclass; not visible through abstract base


def _normalize_objectives(
    f: np.ndarray,
    direction: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalize objectives using NSGA-III ideal point + hyperplane intercepts.

    Deb & Jain (2014).

    Parameters
    ----------
    f : np.ndarray
        Shape ``(n, n_obj)``.
    direction : np.ndarray or None
        ``+1`` maximize, ``-1`` minimize. ``None`` means all-minimize.
        Maximize objectives are internally converted to minimize-space
        (negated) before computing the ideal point and intercepts.

    Returns
    -------
    f_norm : np.ndarray
        Shape ``(n, n_obj)``. Normalized objectives.
    ideal : np.ndarray
        Shape ``(n_obj,)``. Ideal point used for translation.
    intercepts : np.ndarray
        Shape ``(n_obj,)``. Hyperplane intercepts used for scaling.
    """
    f = f * (-direction) if direction is not None else f
    ideal = f.min(axis=0)
    f_trans = f - ideal

    n_obj = f_trans.shape[1]
    eps_asf = 1e-6
    extreme = np.empty((n_obj, n_obj))
    for j in range(n_obj):
        w = np.full(n_obj, eps_asf)
        w[j] = 1.0
        asf = np.max(f_trans / w, axis=1)
        extreme[j] = f_trans[np.argmin(asf)]

    try:
        plane = np.linalg.solve(extreme, np.ones(n_obj))
        if np.any(plane <= 0):
            raise np.linalg.LinAlgError
        intercepts = 1.0 / plane
    except np.linalg.LinAlgError:
        intercepts = f_trans.max(axis=0).astype(float)
        intercepts[intercepts <= 0] = 1.0

    f_norm = f_trans / intercepts
    return f_norm, ideal, intercepts


def _associate_to_reference_points(
    f_norm: np.ndarray,
    reference_points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Associate each individual with the closest reference line.

    Deb & Jain (2014). Uses perpendicular distance from
    ``f_norm[i]`` to the ray from the origin through ``reference_points[r]``.

    Parameters
    ----------
    f_norm : np.ndarray
        Shape ``(n, n_obj)``.
    reference_points : np.ndarray
        Shape ``(n_ref, n_obj)``.

    Returns
    -------
    assoc_idx : np.ndarray
        Shape ``(n,)`` int. Index of the nearest reference point for each individual.
    dist : np.ndarray
        Shape ``(n,)`` float. Perpendicular distance to that reference line.
    """
    ref_norms_sq = np.sum(reference_points**2, axis=1)  # (n_ref,)
    ref_norms_sq = np.maximum(ref_norms_sq, 1e-30)

    # scalar projection coefficients: (n, n_ref)
    scalars = f_norm @ reference_points.T / ref_norms_sq  # (n, n_ref)
    # projected vectors: (n, n_ref, n_obj)
    proj = scalars[:, :, None] * reference_points[None, :, :]
    # perpendicular distances: (n, n_ref)
    diff = f_norm[:, None, :] - proj
    d_perp = np.linalg.norm(diff, axis=2)

    assoc_idx = np.argmin(d_perp, axis=1)
    dist = d_perp[np.arange(len(f_norm)), assoc_idx]
    return assoc_idx, dist


def _niche_count_select(
    front_local: np.ndarray,
    assoc: np.ndarray,
    dist: np.ndarray,
    niche_count: np.ndarray,
    n_needed: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Order individuals in a front using NSGA-III niche-preservation.

    Deb & Jain (2014). Selects ``n_needed`` individuals from
    ``front_local`` in preference order, updating ``niche_count`` in-place.

    Parameters
    ----------
    front_local : np.ndarray
        Local indices (into feasible sub-array) of individuals in this front.
    assoc : np.ndarray
        Shape ``(n_feasible,)``. Reference-point association for each feasible
        individual.
    dist : np.ndarray
        Shape ``(n_feasible,)``. Perpendicular distance to associated reference line.
    niche_count : np.ndarray
        Shape ``(n_ref,)``. Accumulated niche counts; updated in-place.
    n_needed : int
        Number of individuals to select (≤ len(front_local)).
    rng : np.random.Generator
        Random number generator for tie-breaking.

    Returns
    -------
    np.ndarray
        Shape ``(n_needed,)``. Selected indices from ``front_local`` in priority order.
    """
    pool = list(front_local)
    selected = []
    for _ in range(n_needed):
        # reference points that still have candidates in the pool
        pool_refs = {int(assoc[i]) for i in pool}
        min_nc = min(niche_count[r] for r in pool_refs)
        min_refs = [r for r in pool_refs if niche_count[r] == min_nc]
        chosen_ref = int(rng.choice(min_refs))

        candidates = [i for i in pool if assoc[i] == chosen_ref]
        if niche_count[chosen_ref] == 0:
            i_star = candidates[int(np.argmin(dist[candidates]))]
        else:
            i_star = int(rng.choice(candidates))

        selected.append(i_star)
        pool.remove(i_star)
        niche_count[chosen_ref] += 1

    return np.array(selected, dtype=int)


class NSGA3Comparator(ParetoComparator):
    """
    Comparator for many-objective optimization via NSGA-III style ranking.

    Extends ParetoComparator with reference-point-based niche preservation:

    - sort_population: non-dominated sorting + normalization + niche preservation
    - compare_population: inherited from ParetoComparator (Pareto dominance)

    The NSGA-III selection mechanism (Deb & Jain 2014):

    1. Non-dominated sorting (same fronts as NSGA-II).
    2. Normalize objectives over ``S_t`` (the fronts accumulated so far, not
       the full combined population ``R_t``): translate by the ideal point of
       ``S_t`` then scale by hyperplane intercepts computed from extreme
       points (ASF minimizers) of ``S_t``.
    3. Associate each individual with the nearest reference line (perpendicular
       distance from the individual to the ray from origin through each
       reference point).
    4. Order each front using niche-count-based preference: reference points
       with fewer associated individuals from earlier fronts are preferred;
       ties broken randomly.

    The total ordering returned by sort_population processes all fronts in order,
    accumulating niche counts across fronts so that earlier fronts' niche counts
    propagate to later ones—matching the selection pressure NSGA-III would apply
    when truncating to a target population size. Because sort_population ranks
    every front (not only the one boundary front ``F_l`` selected in the
    paper), each front is in turn treated as if it were that boundary front:
    normalization (step 2) is recomputed over ``S_t`` = the union of fronts
    processed up to and including the current one.

    Parameters
    ----------
    reference_points : np.ndarray
        Shape ``(n_ref, n_obj)``. Reference points on the unit simplex.
        Typically generated by
        :func:`~saealib.utils.weight_vectors.uniform_weight_vectors`.
    direction : np.ndarray or None
        Per-objective optimization directions (``+1`` maximize, ``-1`` minimize).
        ``None`` means all objectives are minimized.
    eps_cv : float
        Constraint-violation tolerance.
    eps_obj : float
        Objective tolerance (passed to base class).
    sorter : NonDominatedSorter
        Non-dominated sorting callable.
    dominator : Dominator or None
        Dominance predicate.
    seed : int or None
        Seed for the random number generator used in niche tie-breaking.

    References
    ----------
    :cite:`deb2014nsga3`: Deb, K., & Jain, H. (2014). An Evolutionary
    Many-Objective Optimization Algorithm Using Reference-Point-Based
    Nondominated Sorting Approach, Part I. IEEE Transactions on Evolutionary
    Computation, 18(4), 577-601.
    """

    aligns_population_to_reference_points = True

    def __init__(
        self,
        reference_points: np.ndarray,
        direction: np.ndarray | None = None,
        *,
        eps_cv: float = 1e-6,
        eps_obj: float = 1e-6,
        sorter: NonDominatedSorter = non_dominated_sort,
        dominator: Dominator | None = None,
        seed: int | None = None,
    ) -> None:
        super().__init__(
            direction,
            eps_cv=eps_cv,
            eps_obj=eps_obj,
            sorter=sorter,
            dominator=dominator,
        )
        self._reference_points = np.asarray(reference_points, dtype=float)
        self.seed = seed
        self._rng = np.random.default_rng(seed) if seed is not None else None
        self._cache_token = object()

    @property
    def reference_points(self) -> np.ndarray:
        """Reference points used for niche preservation."""
        return self._reference_points

    @property
    def rng(self) -> np.random.Generator:
        """
        Random number generator used for niche tie-breaking.

        Lazily created on first access if no seed was supplied at
        construction time, so a run-time owner (e.g.
        :class:`~saealib.execution.runner.Runner`) can inject a generator
        spawned from the master ``ctx.rng`` before this property is ever
        read.

        This generator is owned privately by the comparator once injected or
        lazily created: it advances independently of ``ctx.rng`` and is not
        part of :meth:`~saealib.context.OptimizationState.save`'s serialized
        state, so checkpoint resume re-spawns a fresh one rather than
        continuing this generator's own draw sequence.
        """
        if self._rng is None:
            self._rng = np.random.default_rng()
        return self._rng

    @rng.setter
    def rng(self, value: np.random.Generator) -> None:
        if value is self._rng:
            return
        self._rng = value
        self._cache_token = object()

    def sort_population(self, population: Population) -> np.ndarray:
        """Sort by Pareto front rank then NSGA-III niche preservation."""
        _cache_key = ("pareto_sort", self._criterion_key())
        cached = population.get_cache(_cache_key)
        if cached is not None:
            return cached

        f = population.get_array("f")
        cv = population.get_array("cv")
        feasible = np.where(cv <= self.eps_cv)[0]
        infeasible = np.where(cv > self.eps_cv)[0]

        sorted_feasible: list[int] = []
        if len(feasible):
            f_feasible = f[feasible]
            n_feasible = len(feasible)
            _ranks, fronts = self._sorter(
                f_feasible, direction=self.direction, dominator=self._dominator
            )
            assoc_full = np.full(n_feasible, -1, dtype=int)
            dist_full = np.full(n_feasible, np.inf)
            niche_count = np.zeros(len(self._reference_points), dtype=int)

            # Deb & Jain (2014) normalizes over S_t = F_1 union
            # ... union F_l, the fronts accumulated up to and including the
            # last one needed to reach the target population size, not over
            # the full combined population R_t. This class already
            # generalizes NSGA-III's single-boundary-front selection to
            # every front (see class docstring), so each front is, in turn,
            # treated as if it were that boundary front l: S_t is the union
            # of fronts processed so far, recomputed fresh at every step.
            s_t_local: list[int] = []
            for front_list in fronts:
                front_local = np.array(front_list, dtype=int)
                s_t_local.extend(front_local.tolist())
                s_idx = np.array(s_t_local, dtype=int)

                f_norm, _, _ = _normalize_objectives(f_feasible[s_idx], self.direction)
                assoc_s, dist_s = _associate_to_reference_points(
                    f_norm, self._reference_points
                )
                assoc_full[s_idx] = assoc_s
                dist_full[s_idx] = dist_s

                ordered = _niche_count_select(
                    front_local,
                    assoc_full,
                    dist_full,
                    niche_count,
                    len(front_local),
                    self.rng,
                )
                sorted_feasible.extend(feasible[ordered].tolist())

        sorted_infeasible: np.ndarray = np.empty(0, int)
        if len(infeasible):
            sorted_infeasible = infeasible[np.argsort(cv[infeasible])]

        result = np.concatenate(
            [np.array(sorted_feasible, dtype=int), sorted_infeasible]
        ).astype(int)
        population.set_cache(_cache_key, result)
        return result

    def _criterion_key(self) -> tuple:
        return (
            *super()._criterion_key(),
            self._cache_token,
            _array_key(self._reference_points),
        )


class RNSGA2Comparator(ParetoComparator):
    """
    Comparator for multi-objective optimization via R-NSGA-II style ranking.

    Extends ParetoComparator with reference-point-based secondary ordering:

    - sort_population: non-dominated sorting + reference-point proximity ordering
    - compare_population: overrides ParetoComparator with rank + reference-point
      distance: lower rank wins, ties broken by smaller distance to the
      nearest reference point -- see :meth:`compare_population`

    The R-NSGA-II selection mechanism (Deb & Sundar 2006):

    1. Non-dominated sorting (same fronts as NSGA-II).
    2. Normalize objectives by range (min-max per objective).
    3. Step 1: for each reference point, compute the normalized Euclidean
       distance of every individual in the front to that point and rank them
       ascending (closest = rank 1).
    4. Step 2: each individual's crowding-distance-equivalent is the minimum
       of its per-reference-point ranks from Step 1 (smaller = preferred).
    5. Step 3 (ε-clearing): solutions within ``epsilon`` of a higher-ranked
       solution sharing the same nearest reference point are moved to the
       back of their reference-point group.

    Parameters
    ----------
    reference_points : np.ndarray
        Shape ``(n_ref, n_obj)``. User-supplied reference points (aspiration
        points) in the objective space. Unlike NSGA-III, these need not lie on
        the unit simplex.
    epsilon : float
        ε-clearing radius in normalized objective space. Solutions within
        ``epsilon`` of a better-ranked solution sharing the same reference
        point are deprioritized.
    direction : np.ndarray or None
        Per-objective optimization directions (``+1`` maximize, ``-1`` minimize).
        ``None`` means all objectives are minimized.
    eps_cv : float
        Constraint-violation tolerance.
    eps_obj : float
        Objective tolerance (passed to base class).
    sorter : NonDominatedSorter
        Non-dominated sorting callable.
    dominator : Dominator or None
        Dominance predicate.

    References
    ----------
    :cite:`deb2006rnsga2`: Deb, K., & Sundar, J. (2006). Reference Point
    Based Multi-Objective Optimization Using Evolutionary Algorithms.
    Proceedings of GECCO 2006, 635-642.
    """

    def __init__(
        self,
        reference_points: np.ndarray,
        epsilon: float = 0.001,
        direction: np.ndarray | None = None,
        *,
        eps_cv: float = 1e-6,
        eps_obj: float = 1e-6,
        sorter: NonDominatedSorter = non_dominated_sort,
        dominator: Dominator | None = None,
    ) -> None:
        super().__init__(
            direction,
            eps_cv=eps_cv,
            eps_obj=eps_obj,
            sorter=sorter,
            dominator=dominator,
        )
        self._reference_points = np.asarray(reference_points, dtype=float)
        self._epsilon = float(epsilon)

    @property
    def reference_points(self) -> np.ndarray:
        """Reference points used for preference-based ordering."""
        return self._reference_points

    @property
    def epsilon(self) -> float:
        """ε-clearing radius in normalized objective space."""
        return self._epsilon

    def sort_population(self, population: Population) -> np.ndarray:
        """Sort by Pareto front rank then reference-point proximity (R-NSGA-II)."""
        _cache_key = ("pareto_sort", self._criterion_key())
        cached = population.get_cache(_cache_key)
        if cached is not None:
            return cached

        f = population.get_array("f")
        cv = population.get_array("cv")
        feasible = np.where(cv <= self.eps_cv)[0]
        infeasible = np.where(cv > self.eps_cv)[0]

        rank_full = np.full(len(population), np.inf)
        dist_min_full = np.full(len(population), np.inf)

        sorted_feasible: list[int] = []
        if len(feasible):
            _ranks, fronts = self._sorter(
                f[feasible], direction=self.direction, dominator=self._dominator
            )

            f_feasible = f[feasible]
            f_min = f_feasible.min(axis=0)
            f_max = f_feasible.max(axis=0)
            f_norm = (f_feasible - f_min) / np.maximum(f_max - f_min, 1e-12)

            # Euclidean distance to each reference point
            diff = f_norm[:, None, :] - self._reference_points[None, :, :]
            dist_matrix = np.linalg.norm(diff, axis=2)  # (n_feasible, n_ref)
            assoc_idx = np.argmin(dist_matrix, axis=1)  # (n_feasible,)
            dist_min = dist_matrix[np.arange(len(feasible)), assoc_idx]

            rank_full[feasible] = _ranks
            dist_min_full[feasible] = dist_min

            for front_list in fronts:
                front_local = np.array(front_list, dtype=int)
                ordered = self._order_front(
                    front_local, dist_matrix, assoc_idx, dist_min
                )
                sorted_feasible.extend(feasible[ordered].tolist())

        sorted_infeasible: np.ndarray = np.empty(0, int)
        if len(infeasible):
            sorted_infeasible = infeasible[np.argsort(cv[infeasible])]

        result = np.concatenate(
            [np.array(sorted_feasible, dtype=int), sorted_infeasible]
        ).astype(int)
        population.set_cache(_cache_key, result)
        _rd_key = ("rnsga2_rank_dist", self._criterion_key())
        population.set_cache(_rd_key, (rank_full, dist_min_full))
        return result

    def compare_population(self, population: Population, idx_a: int, idx_b: int) -> int:
        """Compare via R-NSGA-II operator: rank first, then reference-point distance.

        Lower rank wins; within the same rank, closer to the nearest reference
        point wins (smaller distance is better, opposite of NSGA-II crowding
        distance). Infeasibility is handled with the standard constraint-
        domination rule (Deb 2000).
        """
        cv = population.get_array("cv")
        cv_a = float(cv[idx_a])
        cv_b = float(cv[idx_b])

        if cv_a > self.eps_cv and cv_b > self.eps_cv:
            if cv_a < cv_b:
                return -1
            if cv_a > cv_b:
                return 1
            return 0
        if cv_a > self.eps_cv:
            return 1
        if cv_b > self.eps_cv:
            return -1

        # Both feasible: rank first, then distance to nearest reference point
        # (smaller distance = better, i.e. closer to user preference).
        self.sort_population(population)
        _rd_key = ("rnsga2_rank_dist", self._criterion_key())
        cached = population.get_cache(_rd_key)
        assert cached is not None
        rank_full, dist_min_full = cached
        rank_a, rank_b = rank_full[idx_a], rank_full[idx_b]
        if rank_a < rank_b:
            return -1
        if rank_a > rank_b:
            return 1
        dist_a, dist_b = dist_min_full[idx_a], dist_min_full[idx_b]
        if dist_a < dist_b:
            return -1
        if dist_a > dist_b:
            return 1
        return 0

    def _order_front(
        self,
        front_local: np.ndarray,
        dist_matrix: np.ndarray,
        assoc_idx: np.ndarray,
        dist_min: np.ndarray,
    ) -> np.ndarray:
        """Order a single front via the R-NSGA-II procedure (Deb & Sundar 2006).

        Step 1: for each reference point, rank the front's individuals in
        ascending order of normalized Euclidean distance to that point
        (closest = rank 1). Step 2: each individual's crowding-distance-
        equivalent is the minimum of its per-reference-point ranks from
        Step 1 (smaller = preferred); ties are broken by the individual's
        raw distance to its nearest reference point for determinism, since
        the paper does not specify a tie-break. Step 3 (ε-clearing):
        solutions within
        ``epsilon`` of a higher-ranked solution sharing the same nearest
        reference point are moved to the back of their reference-point
        group.
        """
        front_dist = dist_matrix[front_local]  # (n_front, n_ref)
        # Step 1: per-reference-point rank (1 = closest), vectorized via a
        # double argsort; Step 2: minimum rank across reference points.
        ranks_per_ref = (
            np.argsort(
                np.argsort(front_dist, axis=0, kind="stable"), axis=0, kind="stable"
            )
            + 1
        )
        min_rank = ranks_per_ref.min(axis=1)

        base_order = np.lexsort((dist_min[front_local], min_rank))  # ascending both
        ordered = front_local[base_order]

        # ε-clearing: within each reference-point group, move solutions
        # that are within epsilon of a higher-ranked solution to the back.
        selected: list[int] = []
        cleared: list[int] = []
        # track the lowest dist already accepted per reference point
        best_dist: dict[int, float] = {}

        for idx in ordered:
            ref = int(assoc_idx[idx])
            d_val = float(dist_min[idx])
            if ref in best_dist and abs(d_val - best_dist[ref]) < self._epsilon:
                cleared.append(idx)
            else:
                selected.append(idx)
                if ref not in best_dist:
                    best_dist[ref] = d_val

        return np.array(selected + cleared, dtype=int)

    def _criterion_key(self) -> tuple:
        return (
            *super()._criterion_key(),
            _array_key(self._reference_points),
            float(self._epsilon),
        )


def default_comparator(
    n_obj: int,
    direction: np.ndarray | None,
    *,
    eps_cv: float = 1e-6,
    eps_obj: float = 1e-6,
) -> Comparator:
    """
    Return the comparator selected for a problem with this objective count.

    Parameters
    ----------
    n_obj : int
        Number of objectives of the problem.
    direction : np.ndarray or None
        Per-objective optimization directions. shape = (n_obj, )
    eps_cv : float, optional
        Constraint-violation tolerance of the returned comparator.
    eps_obj : float, optional
        Objective-value tolerance of the returned comparator.

    Returns
    -------
    Comparator
        ``SingleObjectiveComparator`` when ``n_obj == 1``, otherwise
        ``NSGA2Comparator``.
    """
    if n_obj == 1:
        return SingleObjectiveComparator(
            direction=None if direction is None else direction[0],
            eps_cv=eps_cv,
            eps_obj=eps_obj,
        )
    return NSGA2Comparator(direction=direction, eps_cv=eps_cv, eps_obj=eps_obj)
