"""Adapter exposing an already-constructed pymoo Problem as saealib's Problem."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from saealib.exceptions import ValidationError
from saealib.problem.constraint import EqualityConstraint, InequalityConstraint
from saealib.problem.problem import Problem

if TYPE_CHECKING:
    from pymoo.core.problem import Problem as PymooCoreProblem


class _EvalCache:
    """One-slot memo so F/G/H for a given x cost exactly one pymoo evaluation.

    saealib's ``SerialEvaluator`` calls ``evaluate_constraints(x)`` and then
    ``evaluate(x, g)`` back-to-back for the same ``x``. Without this cache, an
    n-constraint problem would trigger ``n + 1`` pymoo evaluations per
    candidate instead of one.
    """

    def __init__(self, pymoo_problem: PymooCoreProblem, n_ieq: int, n_eq: int) -> None:
        self._problem = pymoo_problem
        self._n_ieq = n_ieq
        self._n_eq = n_eq
        self._key: bytes | None = None
        self._f = np.empty(0, dtype=float)
        self._g = np.empty(0, dtype=float)
        self._h = np.empty(0, dtype=float)

    def get(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return cached ``(F, G, H)`` for ``x``, recomputing on a cache miss."""
        key = x.tobytes()
        if key != self._key:
            out: dict[str, Any] = self._problem.evaluate(
                x[np.newaxis, :], return_as_dictionary=True
            )
            self._f = np.asarray(out["F"][0], dtype=float)
            self._g = (
                np.asarray(out["G"][0], dtype=float)
                if self._n_ieq > 0
                else np.empty(0, dtype=float)
            )
            self._h = (
                np.asarray(out["H"][0], dtype=float)
                if self._n_eq > 0
                else np.empty(0, dtype=float)
            )
            self._key = key
        return self._f, self._g, self._h


class PymooProblem(Problem):
    """
    Adapter wrapping a pymoo Problem instance as a saealib ``Problem``.

    Lets researchers who already have a pymoo ``Problem`` definition (test
    suites, real-world models) reuse it unchanged inside saealib.

    Parameters
    ----------
    pymoo_problem : pymoo.core.problem.Problem
        An already-constructed pymoo problem instance with finite ``xl``/``xu``.
    eq_tolerance : float, optional
        Feasibility tolerance forwarded to each :class:`EqualityConstraint`
        built from the pymoo problem's ``H`` (equality constraint) output.
        Default: 1e-4, the equality tolerance pymoo uses.
    **problem_kwargs
        Forwarded to :class:`~saealib.problem.problem.Problem` (e.g.
        ``comparator``, ``handler``, ``eps_cv``, ``eps_obj``).

    Raises
    ------
    ValidationError
        If the pymoo problem has no finite ``xl``/``xu`` bounds.

    Notes
    -----
    Unlike opfunu (scalar, single-objective), pymoo problems are
    batch-evaluated (``_evaluate(X, out)``) and may declare inequality
    (``G``) and equality (``H``) constraints. :meth:`evaluate_batch` exploits
    this by calling the wrapped pymoo problem's own vectorized ``evaluate``
    once for the whole batch; callers going through :meth:`evaluate` /
    :meth:`evaluate_constraints` one row at a time instead fall back to the
    one-slot ``_EvalCache``, which still collapses the per-row
    ``evaluate_constraints()`` + ``evaluate()`` pair into a single pymoo call.

    ``direction`` is always all ``-1``: pymoo problems are unconditionally
    minimization. Wrap in :func:`~saealib.maximize` only if the pymoo
    problem itself already negates what you want to maximize.

    ``G`` maps to :class:`InequalityConstraint` (``threshold=0.0``, no sign
    flip — both conventions agree on "feasible when <= 0"); ``H`` maps to
    :class:`EqualityConstraint` without modification.
    """

    def __init__(
        self,
        pymoo_problem: PymooCoreProblem,
        *,
        eq_tolerance: float = 1e-4,
        **problem_kwargs: Any,
    ) -> None:
        if pymoo_problem.xl is None or pymoo_problem.xu is None:
            raise ValidationError(
                "PymooProblem requires the wrapped pymoo problem to define "
                "finite xl/xu bounds; got xl="
                f"{pymoo_problem.xl!r}, xu={pymoo_problem.xu!r}."
            )

        n_ieq = pymoo_problem.n_ieq_constr
        n_eq = pymoo_problem.n_eq_constr
        cache = _EvalCache(pymoo_problem, n_ieq, n_eq)

        def func(x: np.ndarray) -> np.ndarray:
            return cache.get(np.asarray(x, dtype=float))[0]

        def make_ieq_func(i: int) -> Any:
            return lambda x: cache.get(np.asarray(x, dtype=float))[1][i]

        def make_eq_func(i: int) -> Any:
            return lambda x: cache.get(np.asarray(x, dtype=float))[2][i]

        constraints = [
            InequalityConstraint(make_ieq_func(i), threshold=0.0) for i in range(n_ieq)
        ] + [
            EqualityConstraint(make_eq_func(i), tolerance=eq_tolerance)
            for i in range(n_eq)
        ]

        xl = np.broadcast_to(
            np.asarray(pymoo_problem.xl, dtype=float), (pymoo_problem.n_var,)
        )
        xu = np.broadcast_to(
            np.asarray(pymoo_problem.xu, dtype=float), (pymoo_problem.n_var,)
        )

        super().__init__(
            func=func,
            dim=pymoo_problem.n_var,
            n_obj=pymoo_problem.n_obj,
            direction=np.full(pymoo_problem.n_obj, -1.0),
            lb=list(xl),
            ub=list(xu),
            constraints=constraints,
            **problem_kwargs,
        )
        self.pymoo_problem = pymoo_problem

    def evaluate_batch(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
        """
        Evaluate a batch via the wrapped pymoo problem's own vectorized ``evaluate``.

        Parameters
        ----------
        x : np.ndarray
            Batch of candidates, shape ``(n, dim)``.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            ``(f_batch, g_batch)`` with shapes ``(n, n_obj)`` and
            ``(n, n_constraints)``; column order matches ``self.constraints``
            (inequality columns from ``G`` first, then equality columns from
            ``H``).
        """
        out: dict[str, Any] = self.pymoo_problem.evaluate(x, return_as_dictionary=True)
        f_batch = np.asarray(out["F"], dtype=float)
        n = x.shape[0]
        g_parts = []
        if self.pymoo_problem.n_ieq_constr > 0:
            g_parts.append(np.asarray(out["G"], dtype=float))
        if self.pymoo_problem.n_eq_constr > 0:
            g_parts.append(np.asarray(out["H"], dtype=float))
        g_batch = np.hstack(g_parts) if g_parts else np.empty((n, 0), dtype=float)
        return f_batch, g_batch
