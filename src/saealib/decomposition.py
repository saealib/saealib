"""
Decomposition strategies and DecompositionComparator for MOEA/D-style optimization.

This module provides:

- :class:`Decomposition` — ABC for scalarization functions.
- :class:`WeightedSumDecomposition` — linear weighted sum.
- :class:`TchebycheffDecomposition` — Tchebycheff (Chebyshev) scalarization.
- :class:`PBIDecomposition` — Penalty-Based Boundary Intersection.
- :class:`DecompositionComparator` — :class:`~saealib.comparators.Comparator`
  that ranks solutions by their decomposition score.

References
----------
.. [1] Zhang, Q., & Li, H. (2007). MOEA/D: A multiobjective evolutionary
   algorithm based on decomposition. *IEEE Transactions on Evolutionary
   Computation*, 11(6), 712-731.
.. [2] Das, I., & Dennis, J. E. (1998). Normal-boundary intersection.
   *SIAM Journal on Optimization*, 8(3), 631-657.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

from saealib.comparators import Comparator
from saealib.core.contracts import (
    MANY,
    ComponentContract,
    DataSpec,
    PortContract,
    PortDirection,
    PortSpec,
    Var,
)

if TYPE_CHECKING:
    from saealib.population import Population


# ---------------------------------------------------------------------------
# Decomposition ABC
# ---------------------------------------------------------------------------


class Decomposition(ABC):
    """
    Abstract base for scalarization (decomposition) functions.

    A :class:`Decomposition` converts a batch of objective vectors into a
    single scalar score per vector.  Smaller scores are always better
    (minimization convention).  The caller is responsible for applying any
    direction transform before passing ``f`` (i.e., ``f`` should already be
    in a minimization frame where smaller = better for every objective).

    Parameters
    ----------
    f : np.ndarray
        Objective matrix, shape ``(N, n_obj)``. All objectives are assumed
        to be in minimization direction (direction transform applied by
        the caller).
    weights : np.ndarray
        Non-negative weight vector, shape ``(n_obj,)``.
    ideal_point : np.ndarray
        Reference minimum per objective, shape ``(n_obj,)``. For methods that
        do not use the ideal point (e.g. :class:`WeightedSumDecomposition`),
        this argument is accepted but ignored.

    Returns
    -------
    np.ndarray
        Scalar scores, shape ``(N,)``. Lower is better.
    """

    def contract(self) -> ComponentContract:
        """Return the decomposition contract."""
        objective_schema = Var(name="O")
        return ComponentContract(
            ports={
                "decomposition": PortContract(
                    inputs=(
                        PortSpec(
                            name="objectives",
                            direction=PortDirection.INPUT,
                            data=DataSpec(
                                kind="FeatureBatch",
                                bindings={"objective_schema": objective_schema},
                            ),
                            cardinality=MANY,
                        ),
                    ),
                    outputs=(
                        PortSpec(
                            name="scores",
                            direction=PortDirection.OUTPUT,
                            data=DataSpec(kind="RowPredicate"),
                            cardinality=MANY,
                        ),
                    ),
                ),
            },
        )

    @abstractmethod
    def aggregate(
        self,
        f: np.ndarray,
        weights: np.ndarray,
        ideal_point: np.ndarray,
    ) -> np.ndarray:
        """
        Scalarize a batch of objective vectors.

        Parameters
        ----------
        f : np.ndarray
            Objective matrix, shape ``(N, n_obj)``.  Must be in the
            minimization frame (all objectives: smaller = better).
        weights : np.ndarray
            Non-negative weight vector, shape ``(n_obj,)``.
        ideal_point : np.ndarray
            Reference minimum per objective, shape ``(n_obj,)``.

        Returns
        -------
        np.ndarray
            Scalar scores, shape ``(N,)``.  Lower is better.
        """


# ---------------------------------------------------------------------------
# Concrete decompositions
# ---------------------------------------------------------------------------


class WeightedSumDecomposition(Decomposition):
    """
    Linear weighted-sum scalarization.

    ``score_i = w · f_i``

    The ideal point is not used.  This is the simplest decomposition but
    cannot reach non-convex parts of the Pareto front — use
    :class:`TchebycheffDecomposition` or :class:`PBIDecomposition` for
    non-convex problems.

    References
    ----------
    :cite:`zhang2007moead`: Zhang, Q., & Li, H. (2007). MOEA/D: A
    multiobjective evolutionary algorithm based on decomposition. *IEEE
    Transactions on Evolutionary Computation*, 11(6), 712-731.
    """

    def aggregate(
        self,
        f: np.ndarray,
        weights: np.ndarray,
        ideal_point: np.ndarray,
    ) -> np.ndarray:
        """
        Compute the weighted dot product ``f @ weights``.

        Parameters
        ----------
        f : np.ndarray
            Objective matrix, shape ``(N, n_obj)``.
        weights : np.ndarray
            Non-negative weight vector, shape ``(n_obj,)``.
        ideal_point : np.ndarray
            Ignored.

        Returns
        -------
        np.ndarray
            Scores, shape ``(N,)``.
        """
        return np.asarray(f, dtype=float) @ np.asarray(weights, dtype=float)


class TchebycheffDecomposition(Decomposition):
    """
    Tchebycheff (Chebyshev) scalarization.

    ``score_i = max_{j} { w_j * |f_{ij} - z_j*| }``

    where ``z*`` is the ideal point.  Unlike the weighted sum, this
    scalarization can reach any point on the Pareto front, including parts
    of non-convex fronts, by varying the weight vector.

    Zero weights are replaced by ``1e-6`` to avoid degenerate sub-problems
    (a convention from Zhang & Li 2007, Appendix A).

    References
    ----------
    :cite:`zhang2007moead`: Zhang, Q., & Li, H. (2007). MOEA/D: A
    multiobjective evolutionary algorithm based on decomposition. *IEEE
    Transactions on Evolutionary Computation*, 11(6), 712-731.
    """

    _EPS_WEIGHT: float = 1e-6

    def aggregate(
        self,
        f: np.ndarray,
        weights: np.ndarray,
        ideal_point: np.ndarray,
    ) -> np.ndarray:
        """
        Compute the Tchebycheff scalar score.

        Parameters
        ----------
        f : np.ndarray
            Objective matrix, shape ``(N, n_obj)``.
        weights : np.ndarray
            Non-negative weight vector, shape ``(n_obj,)``. Zero entries are
            internally replaced by ``1e-6``.
        ideal_point : np.ndarray
            Ideal (reference minimum) point, shape ``(n_obj,)``.

        Returns
        -------
        np.ndarray
            Scores, shape ``(N,)``.  Lower is better.
        """
        f = np.asarray(f, dtype=float)
        w = np.asarray(weights, dtype=float)
        z = np.asarray(ideal_point, dtype=float)
        w = np.where(w == 0.0, self._EPS_WEIGHT, w)
        # (N, n_obj) element-wise, then max over objectives
        return (w * np.abs(f - z)).max(axis=1)


class PBIDecomposition(Decomposition):
    """
    Penalty-Based Boundary Intersection (PBI) scalarization.

    ``score_i = d1_i + theta * d2_i``

    where:

    - ``d1_i = |(f_i - z*) · (w / ‖w‖)|`` — distance along the weight vector.
    - ``d2_i = ‖(f_i - z*) - d1_i * (w / ‖w‖)‖`` — perpendicular distance.

    ``theta`` (default ``5.0``) penalizes deviation from the weight vector
    direction, controlling the trade-off between convergence and diversity.

    Parameters
    ----------
    theta : float
        Penalty weight for the perpendicular component.  The original
        MOEA/D paper (Zhang & Li, 2007) uses ``theta = 5``.

    References
    ----------
    :cite:`zhang2007moead`: Zhang, Q., & Li, H. (2007). MOEA/D: A
    multiobjective evolutionary algorithm based on decomposition. *IEEE
    Transactions on Evolutionary Computation*, 11(6), 712-731.
    """

    def __init__(self, theta: float = 5.0) -> None:
        self._theta = float(theta)

    @property
    def theta(self) -> float:
        """Penalty coefficient for the perpendicular distance component."""
        return self._theta

    def aggregate(
        self,
        f: np.ndarray,
        weights: np.ndarray,
        ideal_point: np.ndarray,
    ) -> np.ndarray:
        """
        Compute the PBI scalar score.

        Parameters
        ----------
        f : np.ndarray
            Objective matrix, shape ``(N, n_obj)``.
        weights : np.ndarray
            Non-negative weight vector, shape ``(n_obj,)``.
        ideal_point : np.ndarray
            Ideal (reference minimum) point, shape ``(n_obj,)``.

        Returns
        -------
        np.ndarray
            Scores, shape ``(N,)``.  Lower is better.
        """
        f = np.asarray(f, dtype=float)
        w = np.asarray(weights, dtype=float)
        z = np.asarray(ideal_point, dtype=float)

        w_norm = np.linalg.norm(w)
        w_hat = np.ones_like(w) / np.sqrt(len(w)) if w_norm < 1e-12 else w / w_norm

        diff = f - z  # (N, n_obj)
        # Scalar projection onto w_hat: (N,)
        d1 = np.abs(diff @ w_hat)
        # Perpendicular component: (N, n_obj), then its norm: (N,)
        proj = d1[:, None] * w_hat  # (N, n_obj)
        d2 = np.linalg.norm(diff - proj, axis=1)  # (N,)

        return d1 + self._theta * d2


# ---------------------------------------------------------------------------
# DecompositionComparator
# ---------------------------------------------------------------------------


class DecompositionComparator(Comparator):
    """
    Comparator that ranks solutions by a scalarization (decomposition) score.

    Implements MOEA/D-style decomposition: each sub-problem is defined by a
    weight vector ``weights`` and an ideal point ``ideal_point``.  The
    ordering follows the library convention (feasibility first, then score).

    **Ordering rules**:

    1. Infeasible individuals (``cv > eps_cv``) are always placed after
       feasible ones, sorted by ascending constraint violation (Deb 2000).
    2. Among feasible individuals, the individual with the *lower* aggregate
       score (as returned by ``decomposition.aggregate``) is ranked first.
    3. Two individuals with scores within ``eps_obj`` are considered equal.

    **Direction handling**: ``weights`` contains non-negative magnitudes only.
    Sign conventions (minimize vs. maximize) are expressed via ``direction``.
    Before calling ``decomposition.aggregate``, objectives are transformed as
    ``f_min = f * (-direction)`` so that all objectives are in the
    minimization frame.

    **Ideal-point tracking**: when ``ideal_point=None``, ``sort_population``
    computes the per-objective minimum of the feasible set at each call
    (dynamic tracking).  For ``compare``, the pair-wise minimum is used as a
    local approximation — consistent but not globally optimal.

    Parameters
    ----------
    decomposition : Decomposition
        Scalarization function (e.g. :class:`TchebycheffDecomposition`).
    weights : np.ndarray
        Non-negative weight vector, shape ``(n_obj,)``.
    ideal_point : np.ndarray or None
        Reference minimum per objective, shape ``(n_obj,)``.  ``None`` (the
        default) activates dynamic computation from the population.
    direction : np.ndarray or None
        Per-objective optimization direction: ``+1`` = maximize,
        ``-1`` = minimize.  ``None`` means minimize all objectives.
    eps_cv : float
        Feasibility threshold for constraint violation.
    eps_obj : float
        Score tolerance: differences smaller than this are treated as equal.

    References
    ----------
    :cite:`zhang2007moead`: Zhang, Q., & Li, H. (2007). MOEA/D: A
    multiobjective evolutionary algorithm based on decomposition. *IEEE
    Transactions on Evolutionary Computation*, 11(6), 712-731.

    :cite:`deb2000feasibility`: Deb, K. (2000). An efficient constraint
    handling method for genetic algorithms. *Computer Methods in Applied
    Mechanics and Engineering*, 186(2-4), 311-338.
    """

    def __init__(
        self,
        decomposition: Decomposition,
        weights: np.ndarray,
        ideal_point: np.ndarray | None = None,
        direction: np.ndarray | None = None,
        *,
        eps_cv: float = 0.0,
        eps_obj: float = 1e-6,
    ) -> None:
        weights_arr = np.asarray(weights, dtype=float)
        direction_arr = (
            np.asarray(direction, dtype=float) if direction is not None else None
        )
        super().__init__(weights_arr, eps_cv, eps_obj, direction=direction_arr)
        self._decomposition = decomposition
        self._ideal_point: np.ndarray | None = (
            None if ideal_point is None else np.asarray(ideal_point, dtype=float)
        )

    @property
    def decomposition(self) -> Decomposition:
        """The scalarization function used by this comparator."""
        return self._decomposition

    @property
    def ideal_point(self) -> np.ndarray | None:
        """Fixed ideal point, or None for dynamic computation."""
        return self._ideal_point

    def _to_min_frame(self, f: np.ndarray) -> np.ndarray:
        if self.direction is None:
            return f
        return f * (-self.direction)

    def _scores(self, population: Population) -> np.ndarray:
        """Return per-individual aggregate scores (length N), with caching.

        Infeasible individuals receive ``+inf`` so they always sort last.
        """
        cached = population.get_cache("decomp_scores")
        if cached is not None:
            return cached  # type: ignore[return-value]  # cached value is Any; runtime type matches return annotation

        f_arr = population.get_array("f")
        cv_arr = population.get_array("cv")
        n = len(f_arr)

        scores = np.full(n, np.inf)
        feasible = np.where(cv_arr <= self.eps_cv)[0]
        if len(feasible):
            f_min = self._to_min_frame(f_arr[feasible])
            z = (
                self._ideal_point
                if self._ideal_point is not None
                else f_min.min(axis=0)
            )
            scores[feasible] = self._decomposition.aggregate(f_min, self.weights, z)

        population.set_cache("decomp_scores", scores)
        return scores

    def sort_population(self, population: Population) -> np.ndarray:
        """
        Sort by aggregate score; infeasible individuals come last.

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
        scores = self._scores(population)

        feasible = np.where(cv_arr <= self.eps_cv)[0]
        infeasible = np.where(cv_arr > self.eps_cv)[0]

        sorted_feasible: np.ndarray = np.empty(0, int)
        if len(feasible):
            order = np.argsort(scores[feasible], kind="stable")
            sorted_feasible = feasible[order]

        sorted_infeasible: np.ndarray = np.empty(0, int)
        if len(infeasible):
            sorted_infeasible = infeasible[np.argsort(cv_arr[infeasible])]

        return np.concatenate([sorted_feasible, sorted_infeasible]).astype(int)

    def compare_population(self, population: Population, idx_a: int, idx_b: int) -> int:
        """
        Compare two individuals using feasibility rule then aggregate score.

        Returns ``-1`` if ``a`` is better, ``1`` if ``b`` is better, ``0``
        if equal (within ``eps_obj``).

        Parameters
        ----------
        population : Population
        idx_a : int
        idx_b : int

        Returns
        -------
        int
        """
        cv_arr = population.get_array("cv")
        cv_a = float(cv_arr[idx_a])
        cv_b = float(cv_arr[idx_b])

        if cv_a > self.eps_cv and cv_b > self.eps_cv:
            if cv_a < cv_b:
                return -1
            elif cv_a > cv_b:
                return 1
            return 0
        if cv_a > self.eps_cv:
            return 1
        if cv_b > self.eps_cv:
            return -1

        scores = self._scores(population)
        sa = scores[idx_a]
        sb = scores[idx_b]
        if sa < sb - self.eps_obj:
            return -1
        elif sa > sb + self.eps_obj:
            return 1
        return 0

    def compare(self, fa: np.ndarray, cv_a: float, fb: np.ndarray, cv_b: float) -> int:
        """
        Compare two solutions directly without a Population object.

        When ``ideal_point`` was not provided at construction time, the
        pair-wise minimum ``min(fa_min, fb_min)`` is used as a local
        approximation of the ideal point.  This is consistent for pairwise
        comparison but may differ from ``sort_population`` results (which use
        the population-wide minimum).

        Parameters
        ----------
        fa, fb : np.ndarray
            Objective vectors, shape ``(n_obj,)``.
        cv_a, cv_b : float
            Constraint violations.

        Returns
        -------
        int
            ``-1`` if ``a`` is better, ``1`` if ``b`` is better, ``0`` if equal.
        """
        if cv_a > self.eps_cv and cv_b > self.eps_cv:
            if cv_a < cv_b:
                return -1
            elif cv_a > cv_b:
                return 1
            return 0
        if cv_a > self.eps_cv:
            return 1
        if cv_b > self.eps_cv:
            return -1

        fa_min = self._to_min_frame(np.asarray(fa, dtype=float))
        fb_min = self._to_min_frame(np.asarray(fb, dtype=float))

        z = (
            self._ideal_point
            if self._ideal_point is not None
            else np.minimum(fa_min, fb_min)
        )

        f_batch = np.stack([fa_min, fb_min])
        scores = self._decomposition.aggregate(f_batch, self.weights, z)
        sa, sb = float(scores[0]), float(scores[1])

        if sa < sb - self.eps_obj:
            return -1
        elif sa > sb + self.eps_obj:
            return 1
        return 0
