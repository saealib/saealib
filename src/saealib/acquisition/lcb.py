"""Lower Confidence Bound (LCB) acquisition function module."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import numpy as np

from saealib.acquisition.base import PointwiseAcquisition, direction_to_minimize_sign
from saealib.acquisition.kernels import lower_confidence_bound_kernel
from saealib.core.contracts import ComponentContract, StateContract
from saealib.core.state import RUNTIME_DECISION_COUNT, RUNTIME_RNG
from saealib.exceptions import ValidationError
from saealib.surrogate.prediction import SurrogatePrediction

if TYPE_CHECKING:
    from saealib.context import OptimizationState
    from saealib.population import Archive


class LowerConfidenceBound(PointwiseAcquisition):
    """
    Lower Confidence Bound (LCB) acquisition function.

    LCB trades off exploitation (low predicted mean) and exploration
    (high predicted uncertainty):

        LCB(x) = mu(x) - kappa * sigma(x)

    For minimization, lower LCB values are better. This class returns
    the negated LCB so that higher scores indicate more promising candidates,
    consistent with the convention used by other acquisition functions.

    Requires a surrogate that provides uncertainty estimates (std).

    Parameters
    ----------
    kappa : float
        Exploration-exploitation trade-off parameter used when
        ``beta_schedule`` is ``None``. Higher values encourage more
        exploration. Default: 2.0.
    obj_idx : int
        Index of the objective to optimize. Used for multi-objective
        problems where LCB is applied to a single objective. Default: 0.
    beta_schedule : callable or None
        Optional ``beta_t`` schedule ``schedule(t) -> float`` (``kappa``
        corresponds to ``sqrt(beta_t)`` in the cited bound). When set,
        ``t = ctx.decision_count + 1`` -- the number of genuinely new
        evaluation plans the runtime has confirmed so far (synchronous or
        async), plus one for the decision about to be made -- and
        ``kappa_t = sqrt(schedule(t))`` replaces the fixed ``kappa``. This
        makes ``t`` one full round in the cited paper's sense only under
        synchronous, single-point-per-decision execution; see
        :func:`gp_ucb_beta_schedule`. ``None`` (default) keeps ``kappa``
        fixed.
    direction : np.ndarray or None
        Per-objective optimization direction (+1 = maximize, -1 = minimize).
        shape: (n_obj,). The predicted mean is converted to minimize-space
        via ``direction_to_minimize_sign`` before the (minimize-only) LCB
        formula above runs. ``None`` (default) means already-minimize; when
        unset, it is auto-injected from ``problem.direction`` at run start.

    References
    ----------
    :cite:`srinivas2012gpucb`: Srinivas, N., Krause, A., Kakade, S. M., &
    Seeger, M. W. (2012). Information-theoretic regret bounds for Gaussian
    process optimization in the bandit setting. *IEEE Transactions on
    Information Theory*, 58(5), 3250-3265.
    """

    requires_uncertainty: bool = True

    def __init__(
        self,
        kappa: float = 2.0,
        obj_idx: int = 0,
        beta_schedule: Callable[[int], float] | None = None,
        direction: np.ndarray | None = None,
    ):
        self.kappa = kappa
        self.obj_idx = obj_idx
        self.beta_schedule = beta_schedule
        self.direction = direction

    def contract(self) -> ComponentContract:
        """Return the acquisition contract with the random generator it reads."""
        return replace(
            super().contract(),
            state=StateContract(reads=(RUNTIME_DECISION_COUNT, RUNTIME_RNG)),
        )

    def compute_reference(
        self, archive: Archive, rng: np.random.Generator | None = None
    ) -> Any:
        """LowerConfidenceBound uses no reference value; always returns None."""
        return None

    def prepare(self, archive: Archive, *, ctx: OptimizationState | None) -> Any:
        """
        Resolve the ``beta_schedule``-driven ``kappa`` for this decision, if set.

        Overrides :meth:`PointwiseAcquisition.prepare` (rather than only
        ``compute_reference()``) because resolving ``beta_schedule`` needs
        ``ctx.decision_count``, which ``compute_reference()`` has no access
        to. When ``beta_schedule`` is ``None``, delegates to
        ``compute_reference()`` unchanged.

        Returns
        -------
        Any
            ``None`` when ``beta_schedule`` is unset; otherwise the resolved
            ``kappa_t`` float, passed to :meth:`score` as ``reference``.

        Raises
        ------
        ValidationError
            If ``beta_schedule`` is set but ``ctx`` is ``None`` (``t`` has no
            source). Call via ``evaluate()`` with a real ``ctx``, not
            ``score()`` directly.
            If ``beta_schedule`` returns a non-finite or negative value.
        """
        if self.beta_schedule is None:
            return self.compute_reference(
                archive, rng=ctx.rng if ctx is not None else None
            )
        if ctx is None:
            raise ValidationError(
                "LowerConfidenceBound.beta_schedule requires ctx.decision_count; "
                "call evaluate() with a real OptimizationState, not score() "
                "directly with no ctx."
            )
        t = ctx.decision_count + 1
        beta = self.beta_schedule(t)
        if not math.isfinite(beta) or beta < 0:
            raise ValidationError(
                f"LowerConfidenceBound.beta_schedule({t}) returned {beta!r}; "
                "expected a finite, non-negative beta_t."
            )
        return math.sqrt(beta)

    def score(
        self,
        prediction: SurrogatePrediction,
        reference: Any = None,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """
        Compute negated LCB scores (higher is better).

        Parameters
        ----------
        prediction : SurrogatePrediction
            Surrogate predictions. Must have std (has_uncertainty == True).
        reference : Any
            The ``kappa_t`` resolved by :meth:`prepare` when ``beta_schedule``
            is set (``None`` otherwise). Passing ``None`` here while
            ``beta_schedule`` is set is a caller error (see raises below);
            it never silently falls back to the fixed ``kappa``.

        Returns
        -------
        np.ndarray
            Negated LCB scores. shape: (n_samples,)

        Raises
        ------
        TypeError
            If prediction does not contain uncertainty estimates.
        ValidationError
            If ``beta_schedule`` is set but ``reference`` is ``None`` (i.e.
            ``score()`` was called directly, bypassing ``prepare()``).
        """
        if not prediction.has_uncertainty:
            raise TypeError(
                "LowerConfidenceBound requires a surrogate with uncertainty "
                "estimates (prediction.std must not be None)."
            )
        assert prediction.std is not None
        if self.beta_schedule is not None:
            if reference is None:
                raise ValidationError(
                    "LowerConfidenceBound.beta_schedule is set but score() was "
                    "called without a prepare()-resolved reference; call "
                    "evaluate() with a real ctx, not score() directly."
                )
            kappa = reference
        else:
            kappa = self.kappa
        s = direction_to_minimize_sign(self.direction)
        s_idx = s[self.obj_idx] if isinstance(s, np.ndarray) else s
        mu = prediction.value[:, self.obj_idx] * s_idx  # (n_samples,)
        sigma = prediction.std[:, self.obj_idx]  # (n_samples,)
        return -lower_confidence_bound_kernel(mu, sigma, kappa)


@dataclass(frozen=True)
class _GPUCBBetaSchedule:
    """Picklable ``t -> beta_t`` callable for the finite-domain GP-UCB bound."""

    domain_size: int
    delta: float = 0.1

    def __post_init__(self) -> None:
        if not isinstance(self.domain_size, int) or isinstance(self.domain_size, bool):
            raise ValidationError("gp_ucb_beta_schedule: domain_size must be an int")
        if self.domain_size <= 0:
            raise ValidationError("gp_ucb_beta_schedule: domain_size must be positive")
        if not (0.0 < self.delta < 1.0):
            raise ValidationError(
                "gp_ucb_beta_schedule: delta must satisfy 0 < delta < 1"
            )

    def __call__(self, t: int) -> float:
        if not isinstance(t, int) or isinstance(t, bool):
            raise ValidationError("gp_ucb_beta_schedule: t must be an int")
        if t < 1:
            raise ValidationError("gp_ucb_beta_schedule: t must be >= 1")
        return 2.0 * math.log(self.domain_size * t**2 * math.pi**2 / (6.0 * self.delta))


def gp_ucb_beta_schedule(
    domain_size: int, delta: float = 0.1
) -> Callable[[int], float]:
    """
    Return the finite-domain GP-UCB ``beta_t`` schedule for use with ``beta_schedule``.

    Implements the finite-domain schedule
    ``beta_t = 2 log(domain_size * t^2 * pi^2 / (6 * delta))``
    :cite:`srinivas2012gpucb`, where ``domain_size`` is the cardinality
    ``|D|`` of a finite decision set. This helper does not implement the
    cited paper's separate continuous-domain schedule (a different
    formula, for a compact domain). Passing a ``domain_size`` for a
    continuous search space only substitutes a number into this
    finite-domain formula; it does not by itself grant the finite-domain
    theorem's regret guarantee -- see the "GP-UCB" algorithm page in the
    docs for the additional conditions that requires.

    Parameters
    ----------
    domain_size : int
        Cardinality ``|D|`` of the finite decision set (the cited bound's
        ``|D|``). Must be an int and positive.
    delta : float
        Confidence parameter. Must satisfy ``0 < delta < 1``. Default: 0.1.

    Returns
    -------
    callable
        A picklable ``schedule(t) -> float``, increasing in the round index
        ``t`` (``t`` must be an int starting at 1; raises
        :class:`~saealib.exceptions.ValidationError` for a non-integer ``t``
        or ``t < 1``).

    Raises
    ------
    ValidationError
        If ``domain_size`` is not an int or positive, or ``delta`` is not in
        ``(0, 1)``.
    """
    return _GPUCBBetaSchedule(domain_size, delta)
