"""
Acquisition function base module.

This module defines the abstract base class for acquisition (infill criterion)
functions used in surrogate-assisted optimization.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

import numpy as np

from saealib.core.contracts import (
    MANY,
    ComponentContract,
    DataSpec,
    PortContract,
    PortDirection,
    PortSpec,
    ServiceRequirement,
)
from saealib.exceptions import ValidationError

if TYPE_CHECKING:
    from saealib.context import OptimizationState
    from saealib.population import Archive
    from saealib.surrogate.prediction import SurrogatePrediction


def direction_to_minimize_sign(direction: np.ndarray | None) -> np.ndarray | float:
    """Return the multiplicative sign converting objectives to minimize-space.

    Every direction-sensitive acquisition function's formulas are written
    assuming minimization. Multiplying a raw-objective-space quantity
    (``archive.f``, ``prediction.value``, a user-supplied ``reference``) by
    this sign converts it into minimize-space before the formula runs.
    Uncertainty magnitudes (``prediction.std``, ``sigma``) must never be
    multiplied by this sign.

    Parameters
    ----------
    direction : np.ndarray or None
        Per-objective optimization direction (+1 = maximize, -1 = minimize).
        shape: (n_obj,). ``None`` means already-minimize.

    Returns
    -------
    np.ndarray or float
        ``-direction`` if *direction* is given, else the scalar ``1.0``.
        Both broadcast correctly against ``(n_obj,)`` or ``(n, n_obj)`` arrays.
    """
    return -direction if direction is not None else 1.0


# Sentinel distinct from `None`, since a `prepare()`/`compute_reference()`
# implementation may legitimately return `None` as its prepared reference.
_UNSET = object()


@dataclass
class AcquisitionResult:
    """
    Result of :meth:`AcquisitionFunction.evaluate`.

    Attributes
    ----------
    scores : np.ndarray or None
        Acquisition scores. shape: (n_candidates,), dtype float64. Higher
        values are better.
    artifacts : dict[str, Any]
        Additional per-call information (e.g. joint or set-valued
        acquisition output) not captured by ``scores``.
    """

    scores: np.ndarray | None = None
    artifacts: dict[str, Any] = field(default_factory=dict)


class AcquisitionFunction(ABC):
    """
    Abstract base class for acquisition functions (infill criteria).

    An acquisition function converts a SurrogatePrediction (and/or the
    Archive) into a scalar score per candidate, which is used to rank
    candidates for true evaluation. It is completely decoupled from
    Surrogate/SurrogateManager: it knows nothing about how predictions are
    generated.
    """

    # Optimizer.validate() cross-checks this with surrogate.provides_uncertainty.
    requires_uncertainty: bool = False

    # Optimizer._inject_acquisition_directions() only auto-injects
    # problem.direction into acquisition functions that opt in via this flag.
    direction_sensitive: bool = True

    def contract(self) -> ComponentContract:
        """Return the acquisition contract."""
        return ComponentContract(
            required_services=(ServiceRequirement(name="FeatureEncoder"),),
            ports={
                "acquisition": PortContract(
                    inputs=(
                        PortSpec(
                            name="prediction",
                            direction=PortDirection.INPUT,
                            data=DataSpec(kind="SurrogatePrediction"),
                            cardinality=MANY,
                            optional=True,
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

    def prepare(self, archive: Archive, ctx: OptimizationState | None = None) -> Any:
        """
        Compute a generation-scoped reference value shared across ``evaluate()`` calls.

        Concrete hook returning ``None`` by default. :class:`PointwiseAcquisition`
        overrides it with the ``compute_reference()`` behavior; archive-only
        acquisitions need not override it.

        Parameters
        ----------
        archive : Archive
            Archive of evaluated solutions.
        ctx : OptimizationState or None, optional
            Optimization state, used here only for ``ctx.rng``.

        Returns
        -------
        Any
            A prepared reference value, passed to ``evaluate(..., prepared=...)``.
        """
        return None

    @abstractmethod
    def evaluate(
        self,
        candidates_x: np.ndarray,
        prediction: SurrogatePrediction | None,
        archive: Archive,
        ctx: OptimizationState | None = None,
        *,
        prepared: Any = _UNSET,
    ) -> AcquisitionResult:
        """
        Compute acquisition scores for a set of candidates.

        Parameters
        ----------
        candidates_x : np.ndarray
            Candidate design vectors. shape: (n_candidates, dim).
        prediction : SurrogatePrediction or None
            Surrogate predictions for ``candidates_x``, or ``None`` for an
            archive-only acquisition that needs no prediction.
        archive : Archive
            Archive of evaluated solutions.
        ctx : OptimizationState or None, optional
            Optimization state, used here only for ``ctx.rng``.
        prepared : Any, optional
            A previously computed reference value, as returned by
            ``prepare()``. The default sentinel ``_UNSET`` means "compute
            it now via ``prepare(archive, ctx)``"; an explicit ``None`` is a
            valid prepared value in its own right and skips ``prepare()``.

        Returns
        -------
        AcquisitionResult
            ``scores`` is an owned ``(n_candidates,)`` float64 array; higher
            is better.
        """
        ...


class PointwiseAcquisition(AcquisitionFunction):
    """
    Base for acquisitions using the existing ``compute_reference()``/``score()`` split.

    Concrete subclasses implement ``compute_reference()`` (once per prepared
    reference) and ``score()`` (per candidate batch, given that reference).
    """

    def contract(self) -> ComponentContract:
        """Return the pointwise acquisition contract."""
        contract = super().contract()
        role = contract.ports["acquisition"]
        return replace(
            contract,
            ports={
                **contract.ports,
                "acquisition": replace(
                    role,
                    inputs=tuple(replace(port, optional=False) for port in role.inputs),
                ),
            },
        )

    @abstractmethod
    def compute_reference(
        self,
        archive: Archive,
        rng: np.random.Generator | None = None,
    ) -> Any:
        """
        Compute the reference value required by this acquisition function.

        Each acquisition function derives its appropriate reference from the
        archive. If ``self.reference`` is set (injected externally at
        construction or later), implementations should return it instead of
        computing from the archive, allowing users to supply domain
        knowledge or a fixed reference point.

        Parameters
        ----------
        archive : Archive
            Archive of evaluated solutions.
        rng : np.random.Generator or None, optional
            Random number generator from ``ctx.rng``.  Implementations that
            require randomness should prefer this over a stored ``_rng`` so
            that all randomness flows through the single master RNG.

        Returns
        -------
        Any
            Reference value passed to ``score``. Return ``None`` if this
            acquisition function does not use a reference.
        """
        ...

    @abstractmethod
    def score(
        self,
        prediction: SurrogatePrediction,
        reference: Any,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """
        Compute acquisition scores for a set of candidates.

        Parameters
        ----------
        prediction : SurrogatePrediction
            Predictions from a surrogate model.
        reference : Any
            Reference value produced by ``compute_reference``.
        rng : np.random.Generator or None, optional
            Random number generator from ``ctx.rng``.

        Returns
        -------
        np.ndarray
            Acquisition scores. shape: (n_samples,)
            Higher scores indicate more promising candidates.
        """
        ...

    def prepare(self, archive: Archive, ctx: OptimizationState | None = None) -> Any:
        """Delegate to ``compute_reference(archive, rng=ctx.rng if ctx else None)``."""
        rng = ctx.rng if ctx is not None else None
        return self.compute_reference(archive, rng=rng)

    def evaluate(
        self,
        candidates_x: np.ndarray,
        prediction: SurrogatePrediction | None,
        archive: Archive,
        ctx: OptimizationState | None = None,
        *,
        prepared: Any = _UNSET,
    ) -> AcquisitionResult:
        """
        Reject ``prediction is None``, resolve the reference, then ``score()``.

        See :meth:`AcquisitionFunction.evaluate` for the shared parameter
        contract, including the ``_UNSET`` vs. explicit ``prepared=None``
        distinction.
        """
        if len(candidates_x) == 0:
            return AcquisitionResult(scores=np.empty(0, dtype=np.float64))
        if prediction is None:
            raise TypeError(f"{type(self).__name__} requires a prediction, got None")
        reference = self.prepare(archive, ctx) if prepared is _UNSET else prepared
        rng = ctx.rng if ctx is not None else None
        scores = self.score(prediction, reference, rng=rng)
        scores = np.array(scores, dtype=np.float64, order="C", copy=True)
        if scores.shape != (len(candidates_x),):
            raise ValidationError(
                f"{type(self).__name__}.score() returned shape {scores.shape}, "
                f"expected ({len(candidates_x)},)"
            )
        return AcquisitionResult(scores=scores)


class CompositeAcquisition(AcquisitionFunction):
    """
    Combine per-channel acquisition scores from a multi-channel prediction.

    Pairs with :class:`~saealib.surrogate.manager.CompositeSurrogateManager`
    The manager only composes predictions (one named ``PredictionChannel`` per
    sub-manager); this acquisition composes the
    *scores*. It never stores or invokes a ``SurrogateManager`` -- it only
    projects each configured channel out of an already-produced
    ``SurrogatePrediction`` (via :meth:`SurrogatePrediction.select_channel`)
    and hands it to that channel's child ``AcquisitionFunction``.

    Parameters
    ----------
    acquisitions : dict[str, AcquisitionFunction]
        Maps a prediction channel name to the acquisition function that
        scores it. Must be non-empty. Insertion order is the deterministic
        order of the score arrays passed to ``combine_fn``.
    combine_fn : callable(list[np.ndarray]) -> np.ndarray
        Accepts a list of score arrays (each shape ``(n_candidates,)``, in
        ``acquisitions`` insertion order) and returns a single combined score
        array of the same shape. Use
        :func:`~saealib.surrogate.manager.product_combine` for element-wise
        product (e.g. EI x PoF) or
        :func:`~saealib.surrogate.manager.rank_weighted_combine` for a
        rank-normalised weighted average.
    """

    def __init__(
        self,
        acquisitions: dict[str, AcquisitionFunction],
        combine_fn: Callable[[list[np.ndarray]], np.ndarray],
    ):
        if not acquisitions:
            raise ValidationError(
                "CompositeAcquisition requires at least one acquisition."
            )
        self.acquisitions = acquisitions
        self.combine_fn = combine_fn

    def prepare(self, archive: Archive, ctx: OptimizationState | None = None) -> Any:
        """Prepare every child acquisition; returns ``{name: child_prepared}``."""
        return {
            name: acq.prepare(archive, ctx) for name, acq in self.acquisitions.items()
        }

    def evaluate(
        self,
        candidates_x: np.ndarray,
        prediction: SurrogatePrediction | None,
        archive: Archive,
        ctx: OptimizationState | None = None,
        *,
        prepared: Any = _UNSET,
    ) -> AcquisitionResult:
        """
        Evaluate every configured child acquisition, then ``combine_fn`` the scores.

        See :meth:`AcquisitionFunction.evaluate` for the shared parameter
        contract.

        Raises
        ------
        TypeError
            If ``prediction is None``.
        ValidationError
            If a configured channel is missing from ``prediction.channels``,
            a child acquisition returns no scores, a child's or the combined
            result's score shape is not ``(len(candidates_x),)``.
        """
        if len(candidates_x) == 0:
            return AcquisitionResult(scores=np.empty(0, dtype=np.float64))
        if prediction is None:
            raise TypeError(f"{type(self).__name__} requires a prediction, got None")
        n = len(candidates_x)
        prepared_map = self.prepare(archive, ctx) if prepared is _UNSET else prepared
        score_list: list[np.ndarray] = []
        for name, acq in self.acquisitions.items():
            if name not in prediction.channels:
                raise ValidationError(
                    f"CompositeAcquisition: missing configured channel {name!r}"
                )
            projected = prediction.select_channel(name)
            child_prepared = (
                prepared_map[name] if isinstance(prepared_map, dict) else _UNSET
            )
            result = acq.evaluate(
                candidates_x, projected, archive, ctx, prepared=child_prepared
            )
            if result.scores is None:
                raise ValidationError(
                    f"CompositeAcquisition: acquisition {name!r} returned no scores"
                )
            if result.scores.shape != (n,):
                raise ValidationError(
                    f"CompositeAcquisition: acquisition {name!r} returned scores of "
                    f"shape {result.scores.shape}, expected ({n},)"
                )
            score_list.append(result.scores)
        combined = np.array(self.combine_fn(score_list), dtype=np.float64, copy=True)
        if combined.shape != (n,):
            raise ValidationError(
                f"CompositeAcquisition: combine_fn returned shape {combined.shape}, "
                f"expected ({n},)"
            )
        return AcquisitionResult(scores=combined)
