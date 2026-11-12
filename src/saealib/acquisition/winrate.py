"""WinRateAcquisition acquisition function module."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any

import numpy as np

from saealib.acquisition.base import (
    _UNSET,
    AcquisitionFunction,
    AcquisitionResult,
)
from saealib.core.contracts import ComponentContract
from saealib.registry import register

if TYPE_CHECKING:
    from saealib.context import OptimizationState
    from saealib.population import Archive
    from saealib.surrogate.prediction import SurrogatePrediction


@register()
class WinRateAcquisition(AcquisitionFunction):
    """
    Read the aggregated pairwise win rate from a ``"win_rate"`` channel.

    Pairs with
    :class:`~saealib.surrogate.manager.PairwiseSurrogateManager`, whose
    :meth:`~saealib.surrogate.manager.PairwiseSurrogateManager.predict`
    performs the full reference-sampling + pair-construction +
    ``predict_proba()`` + per-candidate win-rate-aggregation sequence and
    returns it as a ``"win_rate"`` prediction channel. This acquisition does
    not itself compute anything -- the win rate is a model prediction, not a
    post-hoc acquisition score over an independent prediction: the
    ``(candidate, reference)`` pairs it would need to
    predict on are not known until the manager's own reference-sampling
    logic runs, and ``AcquisitionFunction`` deliberately has no
    ``Surrogate``/``SurrogateManager`` access to construct and predict on
    them itself.
    """

    # Win rate is not an objective-space quantity; it has no direction.
    direction_sensitive: bool = False

    def contract(self) -> ComponentContract:
        """Return the win-rate acquisition contract."""
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

    def evaluate(
        self,
        candidates_x: np.ndarray,
        prediction: SurrogatePrediction | None,
        archive: Archive,
        *,
        prepared: Any = _UNSET,
        ctx: OptimizationState | None,
    ) -> AcquisitionResult:
        """
        Read the ``"win_rate"`` channel's value as the score.

        Parameters
        ----------
        prediction : SurrogatePrediction or None
            Must carry a ``"win_rate"`` channel, e.g. as returned by
            ``PairwiseSurrogateManager.predict()``.

        Returns
        -------
        AcquisitionResult
            ``scores`` is ``prediction.channels["win_rate"].value[:, 0]``.
            Higher is better.

        Raises
        ------
        TypeError
            If ``prediction is None``.
        """
        if prediction is None:
            raise TypeError(f"{type(self).__name__} requires a prediction, got None")
        return AcquisitionResult(
            scores=np.array(
                prediction.channels["win_rate"].value[:, 0],
                dtype=np.float64,
                copy=True,
            )
        )
