"""Joint batch improvement acquisitions."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np

from saealib.acquisition.base import (
    AcquisitionFunction,
    AcquisitionResult,
    direction_to_minimize_sign,
)
from saealib.core.contracts import ComponentContract, StateContract
from saealib.core.state import RUNTIME_RNG
from saealib.exceptions import ValidationError
from saealib.registry import register

if TYPE_CHECKING:
    from saealib.context import OptimizationState


def _normal_draws(prediction, ctx, n_draws: int) -> np.ndarray:
    channel = prediction.channels["objective"]
    mean = channel.value
    if mean.shape[1] != 1:
        raise ValidationError("joint expected improvement requires one objective")
    if channel.std is None:
        raise ValidationError("joint improvement requires uncertainty")
    std = channel.std[:, 0]
    covariance = channel.covariance
    if covariance is None:
        covariance = np.diag(std * std)
    covariance = np.asarray(covariance, dtype=np.float64)
    if covariance.shape != (len(mean), len(mean)):
        raise ValidationError("joint covariance must have shape (n, n)")
    rng = ctx.rng if ctx is not None else np.random.default_rng(0)
    return rng.multivariate_normal(mean[:, 0], covariance, size=n_draws)


def _qei_order(values: np.ndarray, best: float) -> tuple[np.ndarray, np.ndarray]:
    selected: list[int] = []
    marginal = np.zeros(values.shape[1], dtype=np.float64)
    current = np.zeros(values.shape[0], dtype=np.float64)
    for _ in range(values.shape[1]):
        utilities = np.empty(values.shape[1], dtype=np.float64)
        for index in range(values.shape[1]):
            if index in selected:
                utilities[index] = -np.inf
                continue
            next_value = (
                np.minimum(current, values[:, index]) if selected else values[:, index]
            )
            utilities[index] = np.maximum(best - next_value, 0.0).mean()
        index = int(np.argmax(utilities))
        selected.append(index)
        next_value = (
            np.minimum(current, values[:, index])
            if len(selected) > 1
            else values[:, index]
        )
        utility = np.maximum(best - next_value, 0.0).mean()
        marginal[index] = utility - (
            np.maximum(best - current, 0.0).mean() if len(selected) > 1 else 0.0
        )
        current = next_value
    return np.asarray(selected, dtype=np.intp), marginal


@register()
class BatchExpectedImprovement(AcquisitionFunction):
    """Monte Carlo qEI with candidate-joint covariance and greedy set utility."""

    requires_uncertainty = True

    def __init__(self, n_draws: int = 4096, direction: np.ndarray | None = None):
        if n_draws < 1:
            raise ValidationError("n_draws must be positive")
        self.n_draws = n_draws
        self.direction = direction

    def contract(self) -> ComponentContract:
        """Return the batch expected-improvement contract."""
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
            state=StateContract(reads=(RUNTIME_RNG,), writes=(RUNTIME_RNG,)),
        )

    def evaluate(
        self,
        candidates_x,
        prediction,
        archive,
        *,
        prepared=None,
        ctx: OptimizationState | None,
    ):
        """Return greedy qEI marginal scores from joint normal draws."""
        if prediction is None:
            raise ValidationError("qEI requires a prediction")
        draws = _normal_draws(prediction, ctx, self.n_draws)
        direction = float(
            np.asarray(direction_to_minimize_sign(self.direction)).reshape(-1)[0]
        )
        best = float(np.min(archive.f[:, 0] * direction))
        order, marginal = _qei_order(draws * direction, best)
        scores = np.full(len(order), -np.inf, dtype=np.float64)
        scores[order] = np.arange(len(order), 0, -1, dtype=np.float64)
        return AcquisitionResult(
            scores=scores,
            artifacts={"joint": True, "order": order, "qei": marginal},
        )
