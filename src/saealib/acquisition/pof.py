"""Probability of Feasibility (PoF) acquisition function module."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any

import numpy as np

from saealib.acquisition.base import PointwiseAcquisition
from saealib.acquisition.kernels import (
    probability_of_feasibility_kernel,
    product_of_feasibility_kernel,
)
from saealib.core.contracts import ComponentContract, StateContract
from saealib.core.state import RUNTIME_RNG
from saealib.surrogate.prediction import SurrogatePrediction

if TYPE_CHECKING:
    from saealib.population import Archive


class ProbabilityOfFeasibility(PointwiseAcquisition):
    """
    Probability of Feasibility (PoF) acquisition function.

    Estimates the probability that a candidate satisfies a constraint
    g(x) <= 0, using a surrogate model that predicts the constraint value.

    PoF(x) = Phi((0 - mu(x)) / sigma(x))

    Typically used in combination with another acquisition function
    (e.g., EI * PoF) to handle black-box constraints.

    Requires a surrogate that provides uncertainty estimates (std).

    Parameters
    ----------
    obj_idx : int
        Index of the predicted constraint to evaluate. Default: 0.

    References
    ----------
    :cite:`schonlau1997pof`: Schonlau, M. (1997). Computer Experiments
    and Global Optimization (PhD thesis, University of Waterloo).
    (Earliest formulation.)

    :cite:`gelbart2014pof`: Gelbart, M. A., Snoek, J., & Adams, R. P.
    (2014). Bayesian optimization with unknown constraints. *Proceedings of
    the 30th Conference on Uncertainty in Artificial Intelligence (UAI)*.
    (Constraint-handling with independent surrogates.)
    """

    requires_uncertainty: bool = True
    # Feasibility probability has no notion of objective direction.
    direction_sensitive: bool = False

    def __init__(self, obj_idx: int = 0, reference: Any = None):
        self.obj_idx = obj_idx
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
        Compute Probability of Feasibility scores.

        Parameters
        ----------
        prediction : SurrogatePrediction
            Surrogate predictions of constraint values.
            Must have std (has_uncertainty == True).
        reference : Any
            Not used. Accepted for interface compatibility.

        Returns
        -------
        np.ndarray
            PoF scores in [0, 1]. shape: (n_samples,)
            Higher scores indicate a higher probability of feasibility.

        Raises
        ------
        TypeError
            If prediction does not contain uncertainty estimates.
        """
        if not prediction.has_uncertainty:
            raise TypeError(
                "ProbabilityOfFeasibility requires a surrogate with uncertainty "
                "estimates (prediction.std must not be None)."
            )
        assert prediction.std is not None
        mu = prediction.value[:, self.obj_idx]  # (n_samples,)
        sigma = prediction.std[:, self.obj_idx]  # (n_samples,)
        return probability_of_feasibility_kernel(mu, sigma)


class ProductOfFeasibility(PointwiseAcquisition):
    r"""Joint probability of feasibility across all constraint columns.

    Computes the product of per-constraint feasibility probabilities:

    .. math::

        \text{PoF}_{\text{joint}}(x) = \prod_{i=1}^{m}
        \Phi\!\left(\frac{0 - \mu_i(x)}{\sigma_i(x)}\right)

    where :math:`\mu_i` and :math:`\sigma_i` are the surrogate's predicted
    mean and standard deviation for the *i*-th constraint column.

    This is the standard acquisition function for constrained Bayesian
    optimisation with independent constraint surrogates
    (:cite:`letham2019constraintbo`; Eriksson & Poloczek, 2021). Use it with
    a surrogate trained on
    ``ConstraintObjectiveSet`` (which returns ``archive.g`` as ``train_y``),
    typically via ``PerObjectiveSurrogate([GP(), ...] * n_constraints)``.

    Combine this acquisition with an objective acquisition through
    ``CompositeAcquisition`` and configure the result on the optimizer.

    Requires a surrogate that provides uncertainty estimates (std).

    Notes
    -----
    Assumes constraints are independent. When constraints are strongly
    correlated, a multi-output GP with joint predictions is more accurate
    but is not required by this acquisition function.

    References
    ----------
    :cite:`letham2019constraintbo`: Letham, B., Karrer, B., Ottoni, G., &
    Bakshy, E. (2019). Constrained Bayesian optimization with noisy
    experiments. *Bayesian Analysis*, 14(2), 495-519.

    Conceptual origin also draws on Eriksson & Poloczek (2021), SCBO.
    """

    requires_uncertainty: bool = True
    # Feasibility probability has no notion of objective direction.
    direction_sensitive: bool = False

    def contract(self) -> ComponentContract:
        """Return the acquisition contract with the random generator it reads."""
        return replace(
            super().contract(),
            state=StateContract(reads=(RUNTIME_RNG,)),
        )

    def compute_reference(
        self, archive: Archive, rng: np.random.Generator | None = None
    ) -> Any:
        """Return None (no reference needed)."""
        return None

    def score(
        self,
        prediction: SurrogatePrediction,
        reference: Any = None,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """
        Compute joint probability of feasibility scores.

        Parameters
        ----------
        prediction : SurrogatePrediction
            Surrogate predictions of constraint values.
            ``prediction.value`` shape: ``(n_samples, n_constraints)``.
            Must have std (``has_uncertainty == True``).
        reference : Any
            Not used. Accepted for interface compatibility.

        Returns
        -------
        np.ndarray
            Joint PoF scores in [0, 1]. shape: ``(n_samples,)``.
            Score is 1 when all constraints are clearly satisfied,
            0 when any constraint is clearly violated.

        Raises
        ------
        TypeError
            If prediction does not contain uncertainty estimates.
        """
        if not prediction.has_uncertainty:
            raise TypeError(
                "ProductOfFeasibility requires a surrogate with uncertainty "
                "estimates (prediction.std must not be None)."
            )
        assert prediction.std is not None
        mu = prediction.value  # (n_samples, n_constraints)
        return product_of_feasibility_kernel(mu, prediction.std)
