"""MeanPrediction acquisition function module."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.spatial import distance

from saealib.acquisition.base import PointwiseAcquisition
from saealib.exceptions import ValidationError
from saealib.registry import register
from saealib.surrogate.prediction import SurrogatePrediction

if TYPE_CHECKING:
    from saealib.context import OptimizationState
    from saealib.population import Archive


@register()
class MeanPrediction(PointwiseAcquisition):
    """
    Acquisition function based on predicted mean value (exploitation).

    For single-objective problems, returns the predicted mean directly.
    For multi-objective problems, returns a weighted scalarization of the
    predicted mean.

    A higher score indicates a more promising candidate.
    The sign convention follows the weight: use a negative weight for
    minimization (e.g., weights=np.array([-1.0])) so that lower objective
    values yield higher scores.

    Parameters
    ----------
    weights : np.ndarray or None
        Weights for magnitude-aware scalarization of multi-objective predictions.
        shape: (n_obj,). If None and direction is also None, uses the first
        objective only.
    direction : np.ndarray or None
        Per-objective optimization directions (+1 = maximize, -1 = minimize).
        Used for direction-only scalarization when magnitude does not matter.
        Takes precedence over weights when both are provided. When unset,
        it is auto-injected from ``problem.direction`` at run start.
    """

    def __init__(
        self,
        weights: np.ndarray | None = None,
        reference: Any = None,
        direction: np.ndarray | None = None,
    ):
        self.weights = weights
        self.direction = direction
        self.reference = reference

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
        Compute scores based on predicted mean.

        Parameters
        ----------
        prediction : SurrogatePrediction
            Surrogate predictions. prediction.value shape: (n_samples, n_obj)
        reference : Any
            Not used. Accepted for interface compatibility.

        Returns
        -------
        np.ndarray
            Scores. shape: (n_samples,)
        """
        m = prediction.value  # (n_samples, n_obj)
        if self.direction is not None:
            return m @ np.asarray(self.direction)  # (n_samples,)
        if self.weights is not None:
            return m @ np.asarray(self.weights)  # (n_samples,)
        return m[:, 0]  # single-objective default


@dataclass(frozen=True)
class CORSReference:
    """Prepared archive reference and beta for one CORS decision."""

    evaluated_x: np.ndarray
    beta: float


@register()
class CORSDistance(PointwiseAcquisition):
    """
    CORS distance-constrained predicted-mean acquisition function.

    Implements the auxiliary problem (CORS-AP) of the Constrained
    Optimization using Response Surfaces (CORS) method: the predicted-mean
    score used to rank candidates for the next costly evaluation is
    overridden to the worst possible value (``-np.inf``) for any candidate
    that violates the distance constraint::

        ||x - x_j|| >= beta_i * Delta_i,  j = 1, ..., k + i - 1

    i.e. whose minimum distance to every previously evaluated point falls
    below the current iteration's threshold ``beta_i * Delta_i``. ``beta_i``
    cycles through ``search_pattern``: the parameters are set by performing
    cycles of ``N + 1`` iterations, with ``beta_i = beta_{i + N + 1}`` for
    all ``i >= 1`` and ``1 >= beta_1 >= ... >= beta_{N+1} = 0`` in the cited work.
    saealib accepts any non-empty finite sequence of beta values in ``[0, 1]``;
    the paper's ordering and terminal-zero conditions are not required.
    ``prepare()`` resolves the entry of ``search_pattern`` selected by
    ``ctx.decision_count``; the first decision uses ``search_pattern[0]``.
    The decision count is supplied by the runtime so repeated preparation
    caused by archive changes cannot skip a CORS phase. ``compute_reference()``
    remains available for context-free direct use and returns the first phase.

    Regis & Shoemaker (2005) define ``Delta_i`` as the maximin distance
    over the feasible domain D from the previously evaluated points::

        Delta_i = max_{x_tilde in D} min_j ||x_tilde - x_j||

    The exact domain-wide maximum is not available to an acquisition function,
    so when ``delta`` is ``None`` this implementation approximates it with
    the candidate pool in ``prediction.x`` as the cover points:
    ``max_c min_j ||candidate_c - evaluated_j||``. This approximation can
    underestimate ``Delta_i`` when a later candidate pool is concentrated in
    a promising region and therefore covers less of D than the full domain.
    Since ``beta <= 1`` guarantees that a maximin candidate in a non-empty
    pool is executable (equality is allowed), the approximation cannot silently
    leave no executable candidate and force an arbitrary choice.
    Passing a numeric ``delta`` opts into the legacy fixed distance scale.

    Because neither ``prediction.value`` nor the archive-derived
    ``reference`` carries candidate coordinates, this class reads the
    candidate design vectors from ``prediction.x``; the surrogate's
    ``predict()`` must populate that field.

    Parameters
    ----------
    delta : float or None, optional
        Fixed distance scale for the current threshold ``beta_i * delta``.
        When ``None`` (the default), compute the paper's ``Delta_i`` at each
        score call from the candidate pool as an approximation. A numeric
        value preserves the legacy fixed-scale behavior.
    search_pattern : Sequence[float]
        The beta cycling sequence. Any non-empty sequence of finite values in
        ``[0, 1]`` is accepted. Defaults to the paper's own SP1 =
        ``<0.95, 0.25, 0.05, 0.03, 0>``; reported as the
        stronger of the two search patterns the paper tested on the
        Dixon-Szego benchmark suite.
    weights : np.ndarray or None
        Weights for magnitude-aware scalarization of multi-objective
        predicted means. shape: (n_obj,). See :class:`MeanPrediction`.
    direction : np.ndarray or None
        Per-objective optimization direction (+1 = maximize, -1 =
        minimize). See :class:`MeanPrediction`. When unset, it is
        auto-injected from ``problem.direction`` at run start.

    References
    ----------
    :cite:`regis2005cors`: Regis, R. G., & Shoemaker, C. A. (2005).
    Constrained global optimization of expensive black box functions using
    radial basis functions. *Journal of Global Optimization*, 31(1),
    153-171.
    """

    def __init__(
        self,
        delta: float | None = None,
        search_pattern: Sequence[float] = (0.95, 0.25, 0.05, 0.03, 0.0),
        weights: np.ndarray | None = None,
        direction: np.ndarray | None = None,
    ):
        if delta is not None:
            try:
                valid_delta = bool(np.isfinite(delta)) and bool(delta > 0)
            except (TypeError, ValueError):
                valid_delta = False
            if not valid_delta:
                raise ValidationError(
                    "CORSDistance.delta must be finite and greater than 0"
                )
        self.delta = delta
        try:
            pattern = tuple(float(beta) for beta in search_pattern)
        except (TypeError, ValueError) as exc:
            raise ValidationError(
                "CORSDistance.search_pattern must be a sequence of finite beta values"
            ) from exc
        if not pattern:
            raise ValidationError("CORSDistance.search_pattern must not be empty")
        if any(not np.isfinite(beta) or not 0.0 <= beta <= 1.0 for beta in pattern):
            raise ValidationError(
                "CORSDistance.search_pattern beta values must be finite and "
                "between 0 and 1"
            )
        self.search_pattern = pattern
        self.weights = weights
        self.direction = direction

    def compute_reference(
        self, archive: Archive, rng: np.random.Generator | None = None
    ) -> CORSReference:
        """Return the archive reference using the first beta phase.

        The runtime calls :meth:`prepare` with an ``OptimizationState`` so
        that the beta phase is derived from ``ctx.decision_count``. Direct
        calls to this context-free hook use the first phase and do not retain
        mutable cycle state.
        """
        return CORSReference(
            evaluated_x=archive.x,
            beta=self.search_pattern[0],
        )

    def prepare(
        self, archive: Archive, ctx: OptimizationState | None = None
    ) -> CORSReference:
        """Resolve the CORS reference for the current runtime decision.

        ``ctx.decision_count`` counts confirmed evaluation plans, so it is
        unchanged by repeated preparation within a decision and advances by
        one for each new synchronous or asynchronous decision.

        Raises
        ------
        ValidationError
            If no runtime context is supplied. The phase has no reliable
            source without ``ctx.decision_count``; use ``compute_reference``
            for context-free direct access to the first phase.
        """
        if ctx is None:
            raise ValidationError(
                "CORSDistance requires ctx.decision_count; call evaluate() "
                "with a real OptimizationState, not prepare() directly "
                "without ctx."
            )
        beta = self.search_pattern[ctx.decision_count % len(self.search_pattern)]
        return CORSReference(evaluated_x=archive.x, beta=beta)

    def score(
        self,
        prediction: SurrogatePrediction,
        reference: CORSReference,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """
        Compute predicted-mean scores, excluding distance-constraint violators.

        Parameters
        ----------
        prediction : SurrogatePrediction
            Surrogate predictions. prediction.value shape: (n_samples, n_obj).
            prediction.x must hold the candidate design vectors,
            shape: (n_samples, n_features), aligned row-for-row with
            prediction.value.
        reference : CORSReference
            The reference returned by :meth:`prepare` or
            :meth:`compute_reference`, including the current beta constraint.

        Returns
        -------
        np.ndarray
            Scores. shape: (n_samples,). Candidates whose minimum distance
            to every evaluated point in ``reference`` is below the current
            ``beta_i * Delta_i`` threshold receive ``-np.inf``. With
            ``delta=None``, ``Delta_i`` is the candidate-pool maximin
            distance; with an explicit ``delta``, it is that fixed value.

        Raises
        ------
        ValueError
            If ``prediction.x`` is missing or its row count does not
            match ``prediction.value``.
        ValidationError
            If ``reference`` is not a ``CORSReference``.
        """
        if not isinstance(reference, CORSReference):
            raise ValidationError(
                "CORSDistance.score() requires a CORSReference from "
                "prepare() or compute_reference()."
            )
        evaluated_x = reference.evaluated_x
        beta = reference.beta
        m = prediction.value  # (n_samples, n_obj)
        if self.direction is not None:
            base = m @ np.asarray(self.direction)  # (n_samples,)
        elif self.weights is not None:
            base = m @ np.asarray(self.weights)  # (n_samples,)
        else:
            base = m[:, 0]  # single-objective default
        scores = np.array(base, dtype=float, copy=True)  # never mutate prediction.value

        evaluated_x = (
            np.asarray(evaluated_x, dtype=float)
            if evaluated_x is not None
            else np.empty((0, 0))
        )
        if beta <= 0 or evaluated_x.shape[0] == 0:
            # beta_i == 0 makes the distance constraint trivially satisfied by
            # every point; an empty archive likewise leaves no constraint to check.
            return scores

        candidate_x = self._candidate_x(prediction)
        if candidate_x.shape[0] == 0:
            return scores

        min_dist = distance.cdist(candidate_x, evaluated_x).min(axis=1)
        delta_i = min_dist.max() if self.delta is None else self.delta
        threshold = beta * delta_i
        if threshold <= 0:
            return scores
        scores[min_dist < threshold] = -np.inf
        return scores

    @staticmethod
    def _candidate_x(prediction: SurrogatePrediction) -> np.ndarray:
        x = prediction.x
        if x is None:
            raise ValidationError(
                "CORSDistance requires prediction.x (the candidate design "
                "vectors) to evaluate the distance constraint; the "
                "surrogate's predict() must populate it."
            )
        x = np.atleast_2d(np.asarray(x, dtype=float))
        if x.shape[0] != prediction.value.shape[0]:
            raise ValidationError(
                "prediction.x must have one row per candidate in "
                f"prediction.value (got {x.shape[0]} rows vs "
                f"{prediction.value.shape[0]})."
            )
        return x
