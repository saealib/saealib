"""Acquisition functions that evaluate candidate geometry against the archive.

These criteria do not use surrogate predictions; they operate directly on
candidate and archive design points.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.spatial.distance import cdist

from saealib.acquisition.base import _UNSET, AcquisitionFunction, AcquisitionResult
from saealib.core.contracts import ComponentContract

if TYPE_CHECKING:
    from saealib.context import OptimizationState
    from saealib.population import Archive
    from saealib.surrogate.prediction import SurrogatePrediction


class NoveltyAcquisition(AcquisitionFunction):
    """
    Score = mean k-nearest-neighbor distance to archive. Larger = more novel.

    ``prediction`` is ignored -- this is an archive-only criterion.

    Parameters
    ----------
    k : int
        Number of nearest neighbors to average. Clamped to archive size.
    """

    # Pure x-space distance; no objective-space semantics to auto-inject.
    direction_sensitive: bool = False

    def __init__(self, k: int = 1):
        self.k = k

    def contract(self) -> ComponentContract:
        """Return the acquisition contract; the archive is an argument."""
        return super().contract()

    def evaluate(
        self,
        candidates_x: np.ndarray,
        prediction: SurrogatePrediction | None,
        archive: Archive,
        *,
        prepared: Any = _UNSET,
        ctx: OptimizationState | None,
    ) -> AcquisitionResult:
        """Compute mean k-NN distance from each candidate to the archive."""
        if len(archive) == 0:
            scores = np.ones(len(candidates_x))
        else:
            dists = cdist(candidates_x, archive.x)
            k = min(self.k, dists.shape[1])
            scores = np.sort(dists, axis=1)[:, :k].mean(axis=1)
        return AcquisitionResult(scores=np.asarray(scores, dtype=np.float64))


class InverseDensityAcquisition(AcquisitionFunction):
    """
    Score = inverse eps-NN density. Prefer candidates in sparse regions.

    ``prediction`` is ignored -- this is an archive-only criterion.

    Parameters
    ----------
    eps : float
        Radius for counting archive neighbors.
    """

    direction_sensitive: bool = False

    def __init__(self, eps: float = 1.0):
        self.eps = eps

    def evaluate(
        self,
        candidates_x: np.ndarray,
        prediction: SurrogatePrediction | None,
        archive: Archive,
        *,
        prepared: Any = _UNSET,
        ctx: OptimizationState | None,
    ) -> AcquisitionResult:
        """Compute inverse eps-NN density for each candidate."""
        if len(archive) == 0:
            scores = np.ones(len(candidates_x))
        else:
            dists = cdist(candidates_x, archive.x)
            counts = (dists < self.eps).sum(axis=1)
            density = counts / len(archive)
            scores = 1.0 / (density + 1e-9)
        return AcquisitionResult(scores=np.asarray(scores, dtype=np.float64))


class MaximinDistanceAcquisition(AcquisitionFunction):
    """Score = min distance to other candidates + min distance to archive.

    Promotes diversity among candidates. ``prediction`` is ignored -- this is
    an archive-only criterion.

    Note: scores all candidates jointly. At small population sizes, score
    variance among candidates will be low -- this is expected behavior.
    """

    direction_sensitive: bool = False

    def evaluate(
        self,
        candidates_x: np.ndarray,
        prediction: SurrogatePrediction | None,
        archive: Archive,
        *,
        prepared: Any = _UNSET,
        ctx: OptimizationState | None,
    ) -> AcquisitionResult:
        """Compute min intra-candidate distance plus min archive distance."""
        n = len(candidates_x)
        if n == 1:
            scores = np.ones(1)
        else:
            intra = cdist(candidates_x, candidates_x)
            np.fill_diagonal(intra, np.inf)
            archive_min = (
                cdist(candidates_x, archive.x).min(axis=1)
                if len(archive) > 0
                else np.ones(n)
            )
            scores = intra.min(axis=1) + archive_min
        return AcquisitionResult(scores=np.asarray(scores, dtype=np.float64))
