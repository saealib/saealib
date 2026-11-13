"""Constraint classes and handlers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from saealib.population import Population


class InequalityConstraint:
    """
    A single inequality constraint: g(x) <= threshold.

    Parameters
    ----------
    func : callable
        Constraint function g(x). Should return a scalar float.
    threshold : float, optional
        Right-hand side of the constraint g(x) <= threshold. Default: 0.0.
    """

    def __init__(self, func: Callable[..., Any], threshold: float = 0.0):
        self.func = func
        self.threshold = threshold

    def evaluate(self, x: np.ndarray) -> float:
        """Return the raw constraint value g(x)."""
        return float(self.func(x))

    def violation_from_value(self, g: float) -> float:
        """
        Return the violation for an already-evaluated raw value g(x).

        This is the single source of truth for the per-constraint violation
        formula: :meth:`violation`, :meth:`evaluate_with_violation`, and
        :class:`StaticToleranceHandler` all delegate to it. Subclasses (e.g.
        :class:`EqualityConstraint`) override this one method to change how a
        raw value maps to a violation, without re-evaluating ``func``.

        Parameters
        ----------
        g : float
            Raw constraint value g(x).

        Returns
        -------
        float
            Constraint violation: max(0, g - threshold).
        """
        return max(0.0, g - self.threshold)

    def violation(self, x: np.ndarray) -> float:
        """Return constraint violation: max(0, g(x) - threshold)."""
        return self.violation_from_value(self.evaluate(x))

    def evaluate_with_violation(self, x: np.ndarray) -> tuple[float, float]:
        """
        Return the raw value and violation with a single function evaluation.

        Returns
        -------
        g : float
            Raw constraint value g(x).
        cv : float
            Constraint violation: max(0, g(x) - threshold).
        """
        g = self.evaluate(x)
        return g, self.violation_from_value(g)

    def gradient(self, x: np.ndarray) -> np.ndarray | None:
        """
        Return the gradient of g(x) with respect to x, if available.

        Override this in a subclass to enable gradient-based constraint
        handlers (e.g. gradient-based repair). The default returns ``None``,
        signalling that no analytical Jacobian is provided.

        Parameters
        ----------
        x : np.ndarray
            The solution at which to evaluate the gradient. shape = (dim, )

        Returns
        -------
        np.ndarray or None
            Gradient vector with shape = (dim, ), or ``None`` when no
            gradient is available.
        """
        return None


class EqualityConstraint(InequalityConstraint):
    """
    A single equality constraint: h(x) = 0, satisfied within a tolerance.

    The violation is ``max(0, |h(x)| - tolerance)``, reusing the inequality
    violation path so that mixed inequality/equality problems aggregate through
    the same :class:`ConstraintHandler`. Only :meth:`violation_from_value` is
    overridden; :meth:`violation` and :meth:`evaluate_with_violation` inherit
    the equality behavior automatically.

    Parameters
    ----------
    func : callable
        Constraint function h(x). Should return a scalar float.
    tolerance : float, optional
        Feasibility tolerance for ``|h(x)| <= tolerance``. Default: 1e-4,
        the equality tolerance used by CEC 2006 and pymoo. Set
        ``tolerance=0.0`` to treat any non-zero residual as a violation, for
        example when a handler such as ``EpsilonConstraintHandler`` supplies
        the search-time threshold.

    Notes
    -----
    Provide an analytical Jacobian by overriding :meth:`gradient` to return
    ``∇h(x)``; this enables gradient-based constraint handlers (e.g. repair).
    """

    def __init__(self, func: Callable[..., Any], tolerance: float = 1e-4):
        super().__init__(func, threshold=0.0)
        self.tolerance = tolerance

    def violation_from_value(self, g: float) -> float:
        """Return equality violation: max(0, abs(h(x)) - tolerance)."""
        return max(0.0, abs(g) - self.tolerance)


def sum_violation(
    constraints: list[InequalityConstraint],
    x: np.ndarray,
    g: np.ndarray,
) -> float:
    """
    Aggregate raw constraint values by summing the per-constraint violations.

    This is the default constraint-violation aggregation and the aggregation
    :class:`StaticToleranceHandler` uses: each constraint maps its own raw
    value through :meth:`InequalityConstraint.violation_from_value`, so
    inequality (``max(0, g - threshold)``) and equality
    (``max(0, |h| - tolerance)``) constraints combine freely. The result is
    zero exactly when every per-constraint violation is zero.

    Pass it to ``Problem(cv_aggregator=...)`` to select it explicitly;
    ``Problem`` uses it through the default handler otherwise.

    Parameters
    ----------
    constraints : list[InequalityConstraint]
        The problem's constraints, aligned with ``g``.
    x : np.ndarray
        The evaluated design vector. shape = (dim, )
    g : np.ndarray
        Raw constraint values g(x). shape = (n_constraints, )

    Returns
    -------
    float
        Sum of the per-constraint violations; ``0.0`` means feasible.
    """
    cv = 0.0
    for gi, c in zip(g, constraints):
        cv += c.violation_from_value(float(gi))
    return cv


def _feasible_mask(
    constraints: list[InequalityConstraint],
    g: np.ndarray,
) -> np.ndarray:
    """
    Mark the rows whose every per-constraint violation is zero.

    Feasibility is decided from the constraint definitions alone: a row is
    feasible when :meth:`InequalityConstraint.violation_from_value` returns
    zero for every constraint after its own tolerance. The decision does not
    depend on how violations are aggregated into ``cv`` and does not use any
    search-time epsilon relaxation.

    Parameters
    ----------
    constraints : list[InequalityConstraint]
        The problem's constraints, aligned with the columns of ``g``.
    g : np.ndarray
        Raw constraint values per row. shape = (n_rows, n_constraints)

    Returns
    -------
    np.ndarray
        Boolean mask of feasible rows. shape = (n_rows, )
    """
    values = np.asarray(g, dtype=float)
    if values.ndim == 1:
        values = values[np.newaxis, :]
    feasible = np.ones(values.shape[0], dtype=bool)
    for column, constraint in enumerate(constraints):
        if column >= values.shape[1]:
            break
        for row in range(values.shape[0]):
            if not feasible[row]:
                continue
            if constraint.violation_from_value(float(values[row, column])) != 0.0:
                feasible[row] = False
    return feasible


def _feasible_rows(
    collection: Any,
    constraints: list[InequalityConstraint],
) -> np.ndarray:
    """
    Mark the feasible rows of a population or archive.

    Uses :func:`_feasible_mask` over the stored raw constraint values. A
    collection without a ``g`` column falls back to ``cv <= 0``, which the
    default aggregation makes equivalent.

    Parameters
    ----------
    collection : Population or Archive
        Rows to classify. The ``g`` column is read when present.
    constraints : list[InequalityConstraint]
        The problem's constraints, aligned with the ``g`` columns.

    Returns
    -------
    np.ndarray
        Boolean mask of feasible rows. shape = (len(collection), )
    """
    schema = getattr(collection, "schema", {})
    if "g" in schema:
        return _feasible_mask(constraints, collection.get_array("g"))
    cv = np.asarray(collection.get_array("cv"), dtype=float)
    return cv <= 0.0


class ConstraintHandler(ABC):
    """
    Pluggable constraint-processing strategy.

    A ``ConstraintHandler`` exposes the constraint-handling lifecycle as a set
    of overridable hooks, decoupling the per-constraint violation formula and
    the aggregation method from the core :class:`Problem`. This lets research
    code swap in alternative strategies (e.g. ε-constraint, penalty functions,
    gradient-based repair, augmented Lagrangian) without forking core classes.

    Lifecycle::

        Ask            -> [repair(x, constraints, lb, ub)]
                       -> evaluate f, g
                       -> [compute_cv(constraints, x, g)]        -> cv
                       -> [augment_objective(f, constraints, x, g)] -> f'
        Tell           -> Comparator(f', cv) with eps_cv = feasibility_threshold
                       (a search-time relaxation; reported feasibility uses
                       the per-constraint violations)
        Generation end -> [on_generation_end(gen, population)]

    Only :meth:`compute_cv` is abstract; the remaining hooks have sensible
    defaults so that subclasses implement just what they need.
    """

    def repair(
        self,
        x: np.ndarray,
        constraints: list[InequalityConstraint],
        lb: np.ndarray,
        ub: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        """
        Repair a design vector before evaluation.

        Called by the algorithm after each variation operator (crossover /
        mutation) and before objective evaluation. The default clips ``x`` to
        ``[lb, ub]``, reproducing the behaviour of ``repair_clipping``.

        Subclasses override this to add domain-constraint repair (e.g. a
        Newton step toward an equality-constraint manifold) on top of—or
        instead of—bounds clipping.

        Parameters
        ----------
        x : np.ndarray
            The design vector to repair. shape = (dim, )
        constraints : list[InequalityConstraint]
            The problem's domain constraints.
        lb : np.ndarray
            Lower bounds. shape = (dim, )
        ub : np.ndarray
            Upper bounds. shape = (dim, )
        **kwargs
            Reserved for future use (e.g. parent solution for reflection
            repair).

        Returns
        -------
        np.ndarray
            The repaired design vector. shape = (dim, )
        """
        return np.clip(x, lb, ub)

    @abstractmethod
    def compute_cv(
        self,
        constraints: list[InequalityConstraint],
        x: np.ndarray,
        g: np.ndarray,
    ) -> float:
        """
        Aggregate raw constraint values into a scalar constraint violation.

        Parameters
        ----------
        constraints : list[InequalityConstraint]
            The problem's inequality constraints, aligned with ``g``.
        x : np.ndarray
            The evaluated design vector. shape = (dim, )
        g : np.ndarray
            Raw constraint values g(x). shape = (n_constraints, )

        Returns
        -------
        float
            Aggregate constraint violation (``cv``); ``0.0`` means feasible.
        """
        ...

    def augment_objective(
        self,
        f: np.ndarray,
        constraints: list[InequalityConstraint],
        x: np.ndarray,
        g: np.ndarray,
    ) -> np.ndarray:
        """
        Transform objective values using constraint information.

        Used by penalty-based or augmented-Lagrangian strategies. The default
        is the identity (objectives are returned unchanged).

        Parameters
        ----------
        f : np.ndarray
            Raw objective values. shape = (n_obj, )
        constraints : list[InequalityConstraint]
            The problem's inequality constraints, aligned with ``g``.
        x : np.ndarray
            The evaluated design vector. shape = (dim, )
        g : np.ndarray
            Raw constraint values g(x). shape = (n_constraints, )

        Returns
        -------
        np.ndarray
            The (possibly) augmented objective values. shape = (n_obj, )
        """
        return f

    @property
    def feasibility_threshold(self) -> float:
        """Search-time threshold the comparator copies onto its ``eps_cv``."""
        return 0.0

    def on_generation_end(self, gen: int, population: Population) -> None:
        """
        Run end-of-generation bookkeeping.

        Used by adaptive strategies (e.g. ε-level control) that update their
        internal state between generations. The default is a no-op.

        Parameters
        ----------
        gen : int
            The generation index that just finished.
        population : Population
            The current population.
        """


class StaticToleranceHandler(ConstraintHandler):
    """
    Default constraint handler reproducing the static-tolerance behavior.

    The constraint violation is the sum of per-constraint violations
    ``sum(c_i.violation_from_value(g_i))`` and the feasibility threshold is a
    fixed ``eps_cv``. Each constraint maps its own raw value to a violation,
    so inequality (``max(0, g - threshold)``) and equality
    (``max(0, |h| - tolerance)``) constraints can be mixed freely. Objectives
    are not augmented.

    Parameters
    ----------
    eps_cv : float, optional
        Search-time feasibility relaxation returned by
        :attr:`feasibility_threshold` and copied onto the comparator. The
        default of 0.0 treats a solution as feasible only when
        :func:`sum_violation` is zero, which matches the per-constraint
        feasibility decision.
    """

    def __init__(self, eps_cv: float = 0.0):
        self._eps_cv = eps_cv

    def compute_cv(
        self,
        constraints: list[InequalityConstraint],
        x: np.ndarray,
        g: np.ndarray,
    ) -> float:
        """Return :func:`sum_violation` of the raw constraint values."""
        return sum_violation(constraints, x, g)

    @property
    def feasibility_threshold(self) -> float:
        """Fixed search-time feasibility relaxation ``eps_cv``."""
        return self._eps_cv


class EpsilonConstraintHandler(ConstraintHandler):
    """
    Dynamic ε-constraint handler.

    The feasibility threshold ε starts at ``schedule(0)`` and is updated
    to ``schedule(gen)`` at the end of each generation via
    :meth:`on_generation_end`. As ε decreases toward 0 the population is
    gradually driven into the feasible region.

    Each call to :meth:`on_generation_end` sets
    ``comparator.eps_cv = feasibility_threshold`` via the runner, so the
    comparator's feasibility criterion tightens in sync with ε.

    Parameters
    ----------
    schedule : callable
        Maps generation index (int) → current ε value (float).
        ``schedule(0)`` is the initial ε; it should decrease toward 0
        over generations.

    Notes
    -----
    :meth:`compute_cv` aggregates violations as:

    - *Equality* constraints: ``|h(x)|``  (raw absolute value; the
      ``EqualityConstraint.tolerance`` field is intentionally bypassed so
      that ε controls the feasibility threshold globally).
    - *Inequality* constraints: ``max(0, g(x) - threshold) ** 2``  (squared
      positive part; the constraint's ``threshold`` still applies).

    This follows the constraint-violation formulation summarized by
    Mezura-Montes & Coello Coello (2011): inequality violations are squared,
    equality violations are the unsquared absolute value. The cited survey states
    the formulation for constraint values that have been normalised beforehand;
    this handler aggregates the raw ``g``/``h`` values without that
    normalisation, so constraints on
    very different scales will not contribute evenly to ``cv``.

    The ε-constrained method itself originates from Takahama & Sakai
    (2006). The aggregation described above is the one given by the
    Mezura-Montes & Coello Coello (2011) survey, which attributes the method
    to them; it is not Takahama & Sakai's own, which raises the inequality
    and equality terms to one shared exponent rather than squaring only the
    inequalities.

    References
    ----------
    :cite:`mezuramontes2011epsilon`: Mezura-Montes, E., & Coello Coello,
    C. A. (2011). Constraint-handling in nature-inspired numerical
    optimization: Past, present and future. *Swarm and Evolutionary
    Computation*, 1(4), 173-194.
    """

    def __init__(self, schedule: Callable[[int], float]):
        self._schedule = schedule
        self._eps: float = float(schedule(0))

    def compute_cv(
        self,
        constraints: list[InequalityConstraint],
        x: np.ndarray,
        g: np.ndarray,
    ) -> float:
        """Return ``|h|`` for equality and ``max(0, g-threshold)**2`` for inequality."""
        cv = 0.0
        for gi, c in zip(g, constraints):
            if isinstance(c, EqualityConstraint):
                cv += abs(float(gi))
            else:
                cv += max(0.0, float(gi) - c.threshold) ** 2
        return cv

    @property
    def feasibility_threshold(self) -> float:
        """Current ε; the comparator treats ``cv <= ε`` as feasible."""
        return self._eps

    def on_generation_end(self, gen: int, population: Population) -> None:
        """Update internal ε to ``schedule(gen)``."""
        self._eps = float(self._schedule(gen))


def _numerical_gradient(
    c: InequalityConstraint, x: np.ndarray, g0: float
) -> np.ndarray:
    """
    Forward-difference approximation of ``c``'s gradient at ``x``.

    Step size ``sqrt(eps_machine) * max(1, |x_i|)`` follows the standard
    forward-difference rule of thumb (balances truncation vs. rounding
    error) scaled to the magnitude of each coordinate, since the paper only
    specifies "a small positive scalar" without a concrete value.
    """
    step_scale = np.sqrt(np.finfo(float).eps)
    grad = np.empty_like(x)
    for i in range(x.shape[0]):
        step = step_scale * max(1.0, abs(x[i]))
        x_perturbed = x.copy()
        x_perturbed[i] += step
        grad[i] = (c.evaluate(x_perturbed) - g0) / step
    return grad


class GradientRepairHandler(ConstraintHandler):
    """
    Gradient-based constraint repair handler.

    Implements the repair procedure of Chootinan & Chen (2006): each call
    to :meth:`repair` runs up to ``max_iter`` outer iterations. In each
    iteration, the set of currently-violated constraints is determined
    (equality: ``|h(x)| > tolerance``; inequality: ``g(x) > threshold``),
    their raw values and gradients are stacked into ``ΔV`` and the Jacobian
    ``J``, and a single simultaneous update is applied::

        Δx = pinv(J) @ ΔV
        x <- x + Δx

    using the Moore-Penrose pseudoinverse, since ``J`` is generally
    not square. Satisfied constraints contribute no row to ``J``/``ΔV`` for
    that iteration, per the paper's remark that "there is no
    need to adjust ... non-violated constraints." Iteration stops early once
    no constraint is violated, or once ``max(|Δx|) < epsilon``.

    ``ΔV`` follows the cited repair procedure. For an inequality constraint
    (single upper bound ``threshold`` only, per
    :class:`InequalityConstraint`), ``ΔV_i =
    threshold - g(x)``. For an :class:`EqualityConstraint` (``h(x) = 0`` by
    construction, i.e. the target value is already embedded as 0), ``ΔV_i =
    -h(x)``; both are the paper's "target minus current value." The paper's
    worked example uses ``ΔV = c - h(x) = 1``,
    which is the sign that the printed equality row does not
    literally show but the numeric example requires).

    When :meth:`~InequalityConstraint.gradient` returns ``None`` for a
    violated constraint, :func:`_numerical_gradient` supplies a
    forward-difference approximation instead of skipping it — this
    is what lets the handler now repair both equality and inequality
    constraints regardless of whether an analytical Jacobian is available.

    After all iterations the result is clipped to ``[lb, ub]`` so bounds
    feasibility is always guaranteed.

    Parameters
    ----------
    max_iter : int, optional
        Maximum number of outer iterations per call. Default: 1.
    epsilon : float, optional
        Convergence threshold: iteration stops early once
        ``max(abs(Δx)) < epsilon``. Default: 1e-6.

    References
    ----------
    :cite:`chootinan2006gradientrepair`: Chootinan, P., & Chen, A. (2006).
    Constraint handling in genetic algorithms using a gradient-based repair
    method. *Computers & Operations Research*, 33(8), 2263-2281.
    """

    def __init__(self, max_iter: int = 1, epsilon: float = 1e-6):
        self.max_iter = max_iter
        self.epsilon = epsilon

    def repair(
        self,
        x: np.ndarray,
        constraints: list[InequalityConstraint],
        lb: np.ndarray,
        ub: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        """Apply simultaneous pseudoinverse updates toward feasibility, then clip."""
        x = x.copy()
        for _ in range(self.max_iter):
            rows = []
            delta_v = []
            for c in constraints:
                value = c.evaluate(x)
                if isinstance(c, EqualityConstraint):
                    if abs(value) <= c.tolerance:
                        continue
                    dv = -value
                else:
                    if value <= c.threshold:
                        continue
                    dv = c.threshold - value
                grad = c.gradient(x)
                if grad is None:
                    grad = _numerical_gradient(c, x, value)
                rows.append(grad)
                delta_v.append(dv)

            if not rows:
                break

            jacobian = np.stack(rows)
            dx = np.linalg.pinv(jacobian) @ np.array(delta_v)
            x = x + dx
            if np.max(np.abs(dx)) < self.epsilon:
                break
        return np.clip(x, lb, ub)

    def compute_cv(
        self,
        constraints: list[InequalityConstraint],
        x: np.ndarray,
        g: np.ndarray,
    ) -> float:
        """Return :func:`sum_violation` of the raw constraint values."""
        return sum_violation(constraints, x, g)


def linear_epsilon_schedule(eps0: float, n_gen: int) -> Callable[[int], float]:
    """
    Return a linear ε schedule: ``eps0`` at gen 0, ``0.0`` at gen ``n_gen``.

    Parameters
    ----------
    eps0 : float
        Initial ε value (gen 0).
    n_gen : int
        Generation at which ε reaches 0.

    Returns
    -------
    callable
        ``schedule(gen) -> float``
    """

    def _schedule(gen: int) -> float:
        return max(0.0, eps0 * (1.0 - gen / n_gen))

    return _schedule


def exponential_epsilon_schedule(eps0: float, decay: float) -> Callable[[int], float]:
    """
    Return an exponential ε schedule: ``eps0 * decay**gen``.

    Parameters
    ----------
    eps0 : float
        Initial ε value (gen 0).
    decay : float
        Multiplicative decay factor per generation (0 < decay < 1).

    Returns
    -------
    callable
        ``schedule(gen) -> float``
    """

    def _schedule(gen: int) -> float:
        return eps0 * (decay**gen)

    return _schedule
