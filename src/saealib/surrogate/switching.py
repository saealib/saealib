"""
Accuracy-based module and parameter switching helpers for the ``iterate()`` loop.

All switchers inherit :class:`AccuracyBasedSurrogateSwitcher` and expose a
single :meth:`~AccuracyBasedSurrogateSwitcher.switch` method intended to be
called once per step inside an ``iterate()`` loop::

    switcher = ManagerSwitcher(primary, fallback)
    for ctx in optimizer.iterate():
        optimizer.set_surrogate_manager(
            switcher.switch(optimizer.surrogate_manager.last_accuracy)
        )
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Generic, TypeVar

from saealib.core.contracts import ComponentContract, PartSpec
from saealib.exceptions import ValidationError

if TYPE_CHECKING:
    from saealib.strategies.base import OptimizationStrategy
    from saealib.surrogate.accuracy import SurrogateAccuracy
    from saealib.surrogate.manager import SurrogateManager

T = TypeVar("T")


class AccuracyBasedSurrogateSwitcher(ABC, Generic[T]):
    """Abstract base for accuracy-driven module and parameter switching.

    Implementations return a replacement surrogate manager, optimization
    strategy, or numeric parameter based on the most recent accuracy snapshot.
    Call :meth:`switch` inside an ``iterate()`` loop and apply the result::

        switcher = StrategySwitcher(ps_strategy, ib_strategy)
        for ctx in optimizer.iterate():
            optimizer.set_strategy(
                switcher.switch(optimizer.surrogate_manager.last_accuracy)
        )
    """

    @abstractmethod
    def switch(self, accuracy: SurrogateAccuracy | None) -> T:
        """Return the replacement value for the next step.

        Parameters
        ----------
        accuracy : SurrogateAccuracy or None
            Most recent accuracy snapshot from
            ``surrogate_manager.last_accuracy``. ``None`` on the first step
            or when no accuracy evaluator is configured.

        Returns
        -------
        T
            Replacement module or parameter value.
        """
        ...


class ManagerSwitcher(AccuracyBasedSurrogateSwitcher["SurrogateManager"]):
    """Switch between two surrogate managers based on an accuracy threshold.

    Returns ``primary`` when ``metric >= threshold``, ``fallback`` otherwise,
    including when accuracy is unavailable or the metric is non-finite.

    Parameters
    ----------
    primary : SurrogateManager
        Manager used when accuracy is satisfactory.
    fallback : SurrogateManager
        Manager used when accuracy is poor or not yet available.
    metric : str
        Metric key in :class:`~saealib.surrogate.accuracy.SurrogateAccuracy`.
        Default: ``"spearman"``.
    threshold : float
        Minimum acceptable metric value. Default: ``0.5``.
    """

    def __init__(
        self,
        primary: SurrogateManager,
        fallback: SurrogateManager,
        *,
        metric: str = "spearman",
        threshold: float = 0.5,
    ) -> None:
        self.primary = primary
        self.fallback = fallback
        self.metric = metric
        self.threshold = threshold

    def contract(self) -> ComponentContract:
        """Return the manager-switcher contract."""
        return ComponentContract(
            parts=(
                PartSpec(name="primary", contract=self.primary.contract()),
                PartSpec(name="fallback", contract=self.fallback.contract()),
            ),
        )

    def switch(self, accuracy: SurrogateAccuracy | None) -> SurrogateManager:
        """Return primary or fallback based on the metric threshold."""
        if accuracy is None:
            return self.fallback
        v = accuracy.get(self.metric)
        if math.isfinite(v) and v >= self.threshold:
            return self.primary
        return self.fallback


class StrategySwitcher(AccuracyBasedSurrogateSwitcher["OptimizationStrategy"]):
    """Switch between optimization strategies based on an accuracy threshold.

    Generalizes a threshold-based switch between two strategies to an
    arbitrary accuracy metric and pair of strategies: use ``primary``
    (e.g. a surrogate-heavy strategy such as PS) once the metric surpasses
    the threshold, otherwise fall back to ``fallback`` (e.g. a more robust
    strategy such as IB or GB).

    Passing ``mid`` and ``mid_threshold`` opts into a three-way split
    that mirrors Hanawa et al. (2025)'s IB/GB/PS recommendation instead of
    a two-way one: ``fallback`` below ``threshold``, ``mid`` between
    ``threshold`` and ``mid_threshold``, ``primary`` at or above
    ``mid_threshold``. Both must be supplied together — see Parameters.

    Parameters
    ----------
    primary : OptimizationStrategy
        Strategy used when accuracy is satisfactory (e.g. PS).
    fallback : OptimizationStrategy
        Strategy used when accuracy is poor (e.g. IB or GB).
    metric : str
        Metric key. Default: ``"spearman"``.
    threshold : float
        Minimum acceptable metric value to leave ``fallback``.
        Default: ``0.56``.
    mid : OptimizationStrategy or None
        Optional third strategy (e.g. GB) used when accuracy is between
        ``threshold`` and ``mid_threshold``. Enables three-way mode when
        provided together with ``mid_threshold``. Default: ``None``
        (two-way mode, ``switch()`` behaves exactly as without this
        parameter).
    mid_threshold : float or None
        Minimum metric value to move from ``mid`` to ``primary`` in
        three-way mode. Must be supplied together with ``mid`` — there is
        no default, since the paper's PS-viable band is narrow and
        accuracy-metric-dependent. Default: ``None``.

    Raises
    ------
    ValueError
        If exactly one of ``mid`` / ``mid_threshold`` is provided.

    Notes
    -----
    The default threshold of 0.56 is taken from Hanawa et al. (2025),
    which recommends IB when Spearman rho is 0.56 or below, GB when rho
    is between 0.57 and 0.99, and PS only at an exact rho of 1.0 — a
    *three*-way split, not a two-way split at 0.56. By default (``mid``
    unset), this ``StrategySwitcher`` collapses that finding to a
    two-strategy approximation, so 0.56 is best used as an IB/GB boundary
    rather than as a PS-switching threshold — callers pairing
    ``primary=PS`` with the default two-way mode should be aware of the
    mismatch. To reproduce the paper's three-way recommendation exactly,
    pass ``mid`` (GB) and ``mid_threshold=1.0`` alongside
    ``primary`` (PS), ``fallback`` (IB), and ``threshold=0.57``::

        switcher = StrategySwitcher(
            primary=ps_strategy,
            fallback=ib_strategy,
            mid=gb_strategy,
            threshold=0.57,
            mid_threshold=1.0,
        )
        for ctx in optimizer.iterate():
            optimizer.set_strategy(
                switcher.switch(optimizer.surrogate_manager.last_accuracy)
            )

    References
    ----------
    :cite:`hanawa2025switching`: Hanawa, Y., Harada, T., & Miura, Y.
    (2025). Impact of surrogate model accuracy on performance and model
    management strategy in surrogate-assisted evolutionary algorithms.
    *Array*, 27, 100461.
    """

    def __init__(
        self,
        primary: OptimizationStrategy,
        fallback: OptimizationStrategy,
        *,
        metric: str = "spearman",
        threshold: float = 0.56,
        mid: OptimizationStrategy | None = None,
        mid_threshold: float | None = None,
    ) -> None:
        if (mid is None) != (mid_threshold is None):
            raise ValidationError("mid and mid_threshold must be provided together")
        self.primary = primary
        self.fallback = fallback
        self.metric = metric
        self.threshold = threshold
        self.mid = mid
        self.mid_threshold = mid_threshold

    def contract(self) -> ComponentContract:
        """Return the strategy-switcher contract."""
        parts = [
            PartSpec(name="primary", contract=self.primary.contract()),
            PartSpec(name="fallback", contract=self.fallback.contract()),
        ]
        if self.mid is not None:
            parts.append(
                PartSpec(name="mid", contract=self.mid.contract(), optional=True)
            )
        return ComponentContract(parts=tuple(parts))

    def switch(self, accuracy: SurrogateAccuracy | None) -> OptimizationStrategy:
        """Return primary, mid (if configured), or fallback based on the metric."""
        if accuracy is None:
            return self.fallback
        v = accuracy.get(self.metric)
        if not math.isfinite(v):
            return self.fallback
        if self.mid is None:
            return self.primary if v >= self.threshold else self.fallback
        assert self.mid_threshold is not None  # enforced together in __init__
        if v < self.threshold:
            return self.fallback
        if v < self.mid_threshold:
            return self.mid
        return self.primary


class GenCtrlSwitcher(AccuracyBasedSurrogateSwitcher[int]):
    r"""Adjust gen_ctrl via exponential smoothing (Repický et al., 2017).

    The surrogate error is estimated from an accuracy metric, exponentially
    smoothed, then linearly mapped to gen_ctrl:

    .. math::

        \varepsilon_t = (1-r_u)\varepsilon_{t-1} + r_u\varepsilon_{\text{new}}

        g_m = \operatorname{round}((1-\varepsilon_t) \cdot g_{m,\max})

    For Spearman rho and R2, ``epsilon = 0.5 * (1 - value)`` maps the
    metric to ``[0, 1]``.  For RMSE the value is used directly (assumed
    to be in ``[0, 1]``).  When accuracy is unavailable the smoothed state
    is unchanged and the current estimate is returned.

    Parameters
    ----------
    gm_max : int
        Upper bound for gen_ctrl. Default: ``5``.
    gm_min : int
        Lower bound for gen_ctrl. Default: ``0``.
    update_rate : float
        Smoothing rate ``r_u`` in (0, 1]. Default: ``0.5``.
    metric : str
        Metric key. Default: ``"spearman"``.
    initial_error : float
        Initial smoothed error before the first observation.
        Default: ``0.5`` (neutral).

    Attributes
    ----------
    smoothed_error : float
        Current exponentially smoothed error estimate. Inspect to monitor
        the internal state during an ``iterate()`` run.
    """

    def __init__(
        self,
        *,
        gm_max: int = 5,
        gm_min: int = 0,
        update_rate: float = 0.5,
        metric: str = "spearman",
        initial_error: float = 0.5,
    ) -> None:
        if not (0 < update_rate <= 1.0):
            raise ValidationError("update_rate must be in (0, 1]")
        if gm_min < 0:
            raise ValidationError("gm_min must be >= 0")
        if gm_max < gm_min:
            raise ValidationError("gm_max must be >= gm_min")
        self.gm_max = gm_max
        self.gm_min = gm_min
        self.update_rate = update_rate
        self.metric = metric
        self.smoothed_error: float = float(initial_error)

    def switch(self, accuracy: SurrogateAccuracy | None) -> int:
        """Update smoothed error and return new gen_ctrl."""
        if accuracy is not None:
            v = accuracy.get(self.metric)
            if math.isfinite(v):
                eps = self._to_error(v)
                self.smoothed_error = (
                    1.0 - self.update_rate
                ) * self.smoothed_error + self.update_rate * eps
        gm = round((1.0 - self.smoothed_error) * self.gm_max)
        return max(self.gm_min, min(self.gm_max, gm))

    def _to_error(self, value: float) -> float:
        if self.metric in ("spearman", "r2"):
            return 0.5 * (1.0 - float(value))
        return float(max(0.0, min(1.0, value)))
