"""Problem class for optimization."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from saealib.execution.evaluator import EvaluationAdapter
    from saealib.space import SearchSpace

from saealib.comparators import (
    Comparator,
    NSGA2Comparator,
    SingleObjectiveComparator,
)
from saealib.exceptions import ValidationError
from saealib.problem.constraint import (
    ConstraintHandler,
    InequalityConstraint,
    StaticToleranceHandler,
)
from saealib.variables import (
    CategoricalVariable,
    ContinuousVariable,
    IntegerVariable,
    Variable,
)


class Problem:
    """
    Definition of optimization problem.

    Attributes
    ----------
    dim : int
        Dimension of the design variables.
    n_obj : int
        Number of objectives.
    direction : np.ndarray
        Optimization direction per objective. shape = (n_obj, )
        Each element must be +1 (maximize) or -1 (minimize).
    lb : np.ndarray
        Lower bounds for design variables. shape = (dim, )
    ub : np.ndarray
        Upper bounds for design variables. shape = (dim, )
    variables : list[Variable]
        Per-dimension variable definitions.
    comparator : Comparator
        Comparator instance to compare solutions.
    eps_cv : float
        Epsilon for constraint violation feasibility threshold. Only used to
        seed the default ``handler``/``comparator`` at construction time;
        mutating this attribute afterwards has no runtime effect. The actual
        running threshold is ``handler.feasibility_threshold``, which the
        ``Runner`` syncs into ``comparator``/``pareto_archive`` every
        generation. Explicit handlers may therefore use a running threshold
        intentionally different from this value. This value remains the
        feasibility basis for ``Result`` and summary ``best_f``; an explicit
        handler with a different running threshold can therefore make reported
        solutions disagree with the comparator's criterion.
    eps_obj : float
        Epsilon for objective value equality comparison.
    func : callable -> float
        Objective function to evaluate solutions.
    constraints : list[InequalityConstraint]
        List of inequality constraint definitions.
    handler : ConstraintHandler
        Constraint-handling strategy used to aggregate violations and augment
        objectives.
    """

    def __init__(
        self,
        func: Callable[..., Any],
        dim: int | None,
        n_obj: int,
        direction: np.ndarray,
        lb: npt.ArrayLike | None = None,
        ub: npt.ArrayLike | None = None,
        comparator: Comparator | None = None,
        constraints: list[InequalityConstraint] | None = None,
        *,
        eps_cv: float = 1e-6,
        eps_obj: float = 1e-6,
        handler: ConstraintHandler | None = None,
        variables: list[Variable] | None = None,
        space: SearchSpace | None = None,
        evaluation_adapter: EvaluationAdapter | None = None,
    ):
        """
        Initialize Problem instance.

        Parameters
        ----------
        func : callable -> float
            Objective function to evaluate solutions.
        dim : int or None
            Dimension of the design variables.
        n_obj : int
            Number of objectives.
        direction : np.ndarray
            Optimization direction per objective. shape = (n_obj, )
            Each element must be +1 (maximize) or -1 (minimize).
        lb : array_like, optional
            Lower bounds for design variables. A scalar is broadcast to all
            dimensions; otherwise, the array must have length dim.
            Required when *variables* is not provided.
        ub : array_like, optional
            Upper bounds for design variables. A scalar is broadcast to all
            dimensions; otherwise, the array must have length dim.
            Required when *variables* is not provided.
        comparator : Comparator, optional
            Comparator instance to use. If None, auto-selected based on n_obj:
            n_obj == 1 -> SingleObjectiveComparator,
            n_obj >  1 -> NSGA2Comparator.
        constraints : list[InequalityConstraint], optional
            List of inequality constraint definitions. Default: empty list.
        eps_cv : float, optional
            Epsilon for constraint violation feasibility threshold. Default: 1e-6.
        eps_obj : float, optional
            Epsilon for objective value equality comparison. Default: 1e-6.
        handler : ConstraintHandler, optional
            Constraint-handling strategy. If None, a StaticToleranceHandler
            (sum-of-violations, fixed eps_cv) is used, reproducing the default
            behavior.
        variables : list[Variable], optional
            Per-dimension variable definitions.  When provided, *lb* and *ub*
            are derived from the variable bounds.  ``len(variables)`` must equal
            *dim*.
        space : SearchSpace, optional
            Search space to use for non-vector representations.  When omitted,
            a :class:`VectorSpace` is constructed from ``variables`` or bounds.
        evaluation_adapter : EvaluationAdapter, optional
            Adapter that transforms genome batches into evaluation payloads.
        """
        direction = np.asarray(direction, dtype=float)
        if not np.all(np.abs(direction) == 1):
            raise ValidationError("direction elements must be +1 or -1")

        if dim is None:
            dim = getattr(space, "dim", None)
        if dim is None or not isinstance(dim, int) or dim < 0:
            raise ValidationError("dim must be provided or exposed by space")

        if space is not None and variables is None and lb is None and ub is None:
            self.variables = []
            raw_lb = None
            raw_ub = None
        elif variables is not None:
            if len(variables) != dim:
                raise ValidationError(
                    f"len(variables)={len(variables)} does not match dim={dim}"
                )
            self.variables: list[Variable] = list(variables)
            raw_lb = np.array([v.lb for v in self.variables], dtype=float)
            raw_ub = np.array([v.ub for v in self.variables], dtype=float)
        else:
            if lb is None or ub is None:
                raise ValidationError(
                    "lb and ub are required when variables is not provided"
                )
            raw_lb = np.asarray(lb, dtype=float)
            raw_ub = np.asarray(ub, dtype=float)
            if raw_lb.ndim == 0:
                raw_lb = np.full(dim, float(raw_lb), dtype=float)
            elif raw_lb.ndim != 1 or raw_lb.size != dim:
                raise ValidationError(
                    f"lb has shape={raw_lb.shape} and length={raw_lb.size}; "
                    f"expected a scalar or a 1-D array with length dim={dim}"
                )
            if raw_ub.ndim == 0:
                raw_ub = np.full(dim, float(raw_ub), dtype=float)
            elif raw_ub.ndim != 1 or raw_ub.size != dim:
                raise ValidationError(
                    f"ub has shape={raw_ub.shape} and length={raw_ub.size}; "
                    f"expected a scalar or a 1-D array with length dim={dim}"
                )
            self.variables = [
                ContinuousVariable(float(raw_lb[i]), float(raw_ub[i]))
                for i in range(dim)
            ]

        if space is None:
            from saealib.space import VectorSpace

            assert raw_lb is not None and raw_ub is not None
            self._space = VectorSpace(dim=dim, lb=raw_lb, ub=raw_ub)
        else:
            self._space = space

        # Cache type masks (computed once).
        self._integer_mask = np.array(
            [isinstance(v, IntegerVariable) for v in self.variables], dtype=bool
        )
        self._categorical_mask = np.array(
            [isinstance(v, CategoricalVariable) for v in self.variables], dtype=bool
        )
        if not self.variables:
            self._integer_mask = np.zeros(dim, dtype=bool)
            self._categorical_mask = np.zeros(dim, dtype=bool)
        self._continuous_mask = ~(self._integer_mask | self._categorical_mask)
        self._n_categories = np.array(
            [
                v.n_categories if isinstance(v, CategoricalVariable) else 0
                for v in self.variables
            ],
            dtype=int,
        )

        self.n_obj = n_obj
        self.direction = direction
        self.eps_cv = eps_cv
        self.eps_obj = eps_obj
        self.func = func
        self.constraints = constraints if constraints is not None else []
        self.evaluation_adapter = evaluation_adapter
        self.handler = (
            handler if handler is not None else StaticToleranceHandler(eps_cv=eps_cv)
        )

        if comparator is not None:
            if comparator.direction is None:
                comparator.direction = direction
            self.comparator = comparator
        elif n_obj == 1:
            self.comparator = SingleObjectiveComparator(
                direction=direction[0], eps_cv=eps_cv, eps_obj=eps_obj
            )
        else:
            self.comparator = NSGA2Comparator(
                direction=direction, eps_cv=eps_cv, eps_obj=eps_obj
            )

    @property
    def space(self) -> SearchSpace:
        """Return the SearchSpace owned by this problem."""
        return self._space

    @property
    def dim(self) -> int:
        """Dimension of design variables (derived from space)."""
        return cast(int, getattr(self._space, "dim"))

    @property
    def lb(self) -> np.ndarray:
        """Lower bounds of design variables (derived from space)."""
        if not hasattr(self._space, "lb"):
            raise ValidationError("lb is unavailable for this non-vector space")
        return np.asarray(self._space.lb)

    @property
    def ub(self) -> np.ndarray:
        """Upper bounds of design variables (derived from space)."""
        if not hasattr(self._space, "ub"):
            raise ValidationError("ub is unavailable for this non-vector space")
        return np.asarray(self._space.ub)

    @property
    def n_constraints(self) -> int:
        """Number of constraint functions."""
        return len(self.constraints)

    @property
    def continuous_mask(self) -> np.ndarray:
        """Boolean mask of continuous dimensions. shape = (dim,)."""
        return self._continuous_mask

    @property
    def integer_mask(self) -> np.ndarray:
        """Boolean mask of integer dimensions. shape = (dim,)."""
        return self._integer_mask

    @property
    def categorical_mask(self) -> np.ndarray:
        """Boolean mask of categorical dimensions. shape = (dim,)."""
        return self._categorical_mask

    @property
    def n_categories(self) -> np.ndarray:
        """Category counts per dimension (0 for non-categorical). shape = (dim,)."""
        return self._n_categories

    def repair(self, x: np.ndarray) -> np.ndarray:
        """Project *x* onto valid variable domains.

        Rounds integer dimensions to the nearest integer and clips to bounds.
        Rounds categorical dimensions to the nearest valid index.
        Clips continuous dimensions to ``[lb, ub]``.

        Parameters
        ----------
        x : np.ndarray
            Design variable array. shape = ``(dim,)`` or ``(n, dim)``.

        Returns
        -------
        np.ndarray
            Repaired array, same shape as *x*.
        """
        if not self.variables:
            return x
        x = np.asarray(x, dtype=float)
        scalar = x.ndim == 1
        if scalar:
            x = x[np.newaxis, :]
        result = x.copy()
        for i, v in enumerate(self.variables):
            result[:, i] = v.repair(result[:, i])
        return result[0] if scalar else result

    def evaluate_constraints(self, x: Any) -> tuple[np.ndarray, float]:
        """
        Evaluate all constraint functions at x.

        Parameters
        ----------
        x : np.ndarray
            The solution to evaluate. shape = (dim, )

        Returns
        -------
        g : np.ndarray
            Raw constraint values. shape = (n_constraints, )
            Empty array when no constraints are defined.
        cv : float
            Aggregate constraint violation as computed by ``handler.compute_cv``.
            0.0 when no constraints are defined.
        """
        if not self.constraints:
            return np.empty(0, dtype=float), 0.0
        g = np.empty(len(self.constraints), dtype=float)
        for i, c in enumerate(self.constraints):
            g[i] = c.evaluate(x)
        cv = self.handler.compute_cv(self.constraints, x, g)
        return g, float(cv)

    def evaluate_batch(self, x: Any) -> tuple[np.ndarray, np.ndarray] | None:
        """
        Evaluate the objective and constraint functions for a batch of solutions.

        The default implementation returns ``None``, meaning batch evaluation is
        not supported by this ``Problem`` -- callers should fall back to the
        row-by-row path (:meth:`evaluate` / :meth:`evaluate_constraints` applied
        one row at a time). Subclasses (or ``Problem`` instances constructed
        around a batch-capable ``func``, e.g. a vectorized function, a GPU
        kernel, or an external library such as pymoo that can score many
        designs in one call) may override this method to evaluate all rows of
        *x* in a single call.

        The returned values are **raw**: unlike :meth:`evaluate`, the objective
        batch is *not* passed through ``handler.augment_objective``, and unlike
        :meth:`evaluate_constraints`, the constraint batch is *not* reduced via
        ``handler.compute_cv``. Callers are expected to apply those
        transformations themselves afterward, per row.

        Parameters
        ----------
        x : np.ndarray
            Batch of solutions to evaluate. shape = (n, dim)

        Returns
        -------
        tuple[np.ndarray, np.ndarray] or None
            ``(f_batch, g_batch)`` if batch evaluation is supported, else
            ``None``.
            ``f_batch`` : raw objective values. shape = (n, n_obj)
            ``g_batch`` : raw constraint values, in the same column order as
            ``self.constraints``. shape = (n, n_constraints); (n, 0) when the
            problem has no constraints.
            Row order matches the order of rows in *x*.
        """
        return None

    def evaluate(self, x: Any, g: np.ndarray | None = None) -> np.ndarray:
        """
        Evaluate the objective function at given solution x.

        After computing the raw objective, ``handler.augment_objective`` is
        applied so that penalty-based or augmented-Lagrangian handlers can
        transform the objective using constraint information. The default
        ``StaticToleranceHandler`` leaves the objective unchanged.

        Parameters
        ----------
        x : np.ndarray
            The solution to evaluate.
        g : np.ndarray, optional
            Pre-computed raw constraint values g(x), shape = (n_constraints, ).
            When None, constraints are evaluated internally if any are defined.
            Pass this to avoid re-evaluating constraints when ``g`` is already
            available (e.g. from :meth:`evaluate_constraints`).

        Returns
        -------
        np.ndarray
            The objective value(s) at solution x. shape = (n_obj, )
        """
        result = self.func(x)
        f = np.atleast_1d(np.asarray(result, dtype=float))
        if g is None:
            g, _ = self.evaluate_constraints(x)
        return self.handler.augment_objective(f, self.constraints, x, g)
