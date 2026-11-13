"""MaxUncertainty acquisition function module."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any

import numpy as np

from saealib.acquisition.base import PointwiseAcquisition
from saealib.core.contracts import ComponentContract, StateContract
from saealib.core.state import RUNTIME_RNG
from saealib.surrogate.prediction import SurrogatePrediction

if TYPE_CHECKING:
    from saealib.population import Archive


class MaxUncertainty(PointwiseAcquisition):
    """
    Acquisition function that maximizes predictive uncertainty (exploration).

    Selects candidates where the surrogate model is least confident.
    Requires a surrogate that provides uncertainty estimates (std).
    This is the "uncertainty sampling" strategy from the active-learning
    literature, applied here as a pure-exploration acquisition function.

    For multi-objective problems, aggregates uncertainty across objectives
    using a weighted sum.

    Using sigma(x) alone as the criterion corresponds to the alpha -> infinity
    limit of the merit function f_M(x) = t_hat(x) - alpha * sigma(x) in
    Büche, Schraudolph & Koumoutsakos (2005). That paper's GPOP
    procedure never actually takes this limit: it optimizes f_M in parallel
    for the four fixed values alpha = 0, 1, 2, 4 in that procedure,
    evaluating all four resulting points rather than using sigma alone.

    Parameters
    ----------
    weights : np.ndarray or None
        Weights for aggregating uncertainty across objectives.
        shape: (n_obj,). If None, uses the mean across objectives.

    References
    ----------
    :cite:`lewis1994uncertaintysampling`: Lewis, D. D., & Gale, W. A.
    (1994). A sequential algorithm for training text classifiers. In
    *SIGIR '94*, 3-12.

    :cite:`settles2009activelearning`: Settles, B. (2009). Active
    learning literature survey. *Computer Sciences Technical Report
    1648*, University of Wisconsin-Madison.

    :cite:`buche2005gpes`: Büche, D., Schraudolph, N. N., & Koumoutsakos,
    P. (2005). Accelerating evolutionary algorithms with Gaussian process
    fitness function models. *IEEE Transactions on Systems, Man, and
    Cybernetics-Part C*, 35(2), 183-194.
    """

    requires_uncertainty: bool = True
    # Uncertainty magnitude has no notion of objective direction.
    direction_sensitive: bool = False

    def __init__(self, weights: np.ndarray | None = None, reference: Any = None):
        self.weights = weights
        self.reference = reference

    def contract(self) -> ComponentContract:
        """Return the acquisition contract with the random generator it reads."""
        return replace(
            super().contract(),
            state=StateContract(reads=(RUNTIME_RNG,)),
        )

    def compute_reference(
        self, archive: Archive, rng: np.random.Generator | None = None
    ) -> Any:
        """Return fixed reference if set, otherwise None."""
        return self.reference

    def score(
        self,
        prediction: SurrogatePrediction,
        reference: Any = None,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """
        Compute scores based on predictive standard deviation.

        Parameters
        ----------
        prediction : SurrogatePrediction
            Surrogate predictions. Must have std (has_uncertainty == True).
        reference : Any
            Not used. Accepted for interface compatibility.

        Returns
        -------
        np.ndarray
            Scores. shape: (n_samples,)

        Raises
        ------
        TypeError
            If prediction does not contain uncertainty estimates.
        """
        if not prediction.has_uncertainty:
            raise TypeError(
                "MaxUncertainty requires a surrogate with uncertainty estimates "
                "(prediction.std must not be None)."
            )
        std = prediction.std  # (n_samples, n_obj)
        assert std is not None
        if self.weights is not None:
            return std @ np.asarray(self.weights)
        return std.mean(axis=1)  # mean uncertainty across objectives
