"""Expected Improvement (EI) acquisition function module."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any

import numpy as np

from saealib.acquisition.base import PointwiseAcquisition, direction_to_minimize_sign
from saealib.acquisition.kernels import expected_improvement_kernel
from saealib.core.contracts import ComponentContract, StateContract
from saealib.core.state import RUNTIME_RNG
from saealib.surrogate.prediction import SurrogatePrediction

if TYPE_CHECKING:
    from saealib.population import Archive


class ExpectedImprovement(PointwiseAcquisition):
    """
    Expected Improvement (EI) acquisition function.

    EI balances exploration and exploitation by computing the expected
    amount of improvement over the current best observed value.

    For single-objective minimization::

        EI(x) = E[max(f_best - f(x), 0)]
               = (f_best - mu) * Phi(Z) + sigma * phi(Z)
        where Z = (f_best - mu) / sigma

    Requires a surrogate that provides uncertainty estimates (std).

    Parameters
    ----------
    xi : float
        Exploration-exploitation trade-off parameter. Higher values
        encourage more exploration. Default: 0.01.
    obj_idx : int
        Index of the objective to optimize. Used for multi-objective
        problems where EI is applied to a single objective. Default: 0.
    direction : np.ndarray or None
        Per-objective optimization direction (+1 = maximize, -1 = minimize).
        shape: (n_obj,). ``archive.f``, a user-supplied ``reference``, and
        the predicted mean are converted to minimize-space via
        ``direction_to_minimize_sign`` before the (minimize-only) EI formula
        above runs. ``None`` (default) means already-minimize; when unset,
        it is auto-injected from ``problem.direction`` at run start.

    References
    ----------
    :cite:`jones1998ego`: Jones, D. R., Schonlau, M., & Welch, W. J. (1998).
    Efficient global optimization of expensive black-box functions. *Journal
    of Global Optimization*, 13(4), 455-492.
    """

    requires_uncertainty: bool = True

    def __init__(
        self,
        xi: float = 0.01,
        obj_idx: int = 0,
        reference: Any = None,
        direction: np.ndarray | None = None,
    ):
        self.xi = xi
        self.obj_idx = obj_idx
        self.reference = reference
        self.direction = direction

    def contract(self) -> ComponentContract:
        """Return the acquisition contract with the random generator it reads."""
        return replace(
            super().contract(),
            state=StateContract(reads=(RUNTIME_RNG,)),
        )

    def compute_reference(
        self, archive: Archive, rng: np.random.Generator | None = None
    ) -> np.ndarray:
        """Return fixed reference if set, otherwise component-wise best from archive."""
        if self.reference is not None:
            return np.asarray(self.reference, dtype=float) * direction_to_minimize_sign(
                self.direction
            )
        s = direction_to_minimize_sign(self.direction)
        return (archive.f * s).min(axis=0)

    def score(
        self,
        prediction: SurrogatePrediction,
        reference: Any,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """
        Compute Expected Improvement scores.

        Parameters
        ----------
        prediction : SurrogatePrediction
            Surrogate predictions. Must have std (has_uncertainty == True).
        reference : Any
            Current best objective value. Scalar or ndarray of shape (n_obj,).
            The objective at index obj_idx is used.

        Returns
        -------
        np.ndarray
            EI scores. shape: (n_samples,). Higher is better.

        Raises
        ------
        TypeError
            If prediction does not contain uncertainty estimates.
        """
        if not prediction.has_uncertainty:
            raise TypeError(
                "ExpectedImprovement requires a surrogate with uncertainty "
                "estimates (prediction.std must not be None)."
            )
        assert prediction.std is not None
        s = direction_to_minimize_sign(self.direction)
        s_idx = s[self.obj_idx] if isinstance(s, np.ndarray) else s
        mu = prediction.value[:, self.obj_idx] * s_idx  # (n_samples,)
        sigma = prediction.std[:, self.obj_idx]  # (n_samples,)

        ref = np.asarray(reference, dtype=float)
        f_best = float(ref.flat[self.obj_idx]) if ref.ndim > 0 else float(ref)

        return expected_improvement_kernel(mu, sigma, f_best, self.xi)
