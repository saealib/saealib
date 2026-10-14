"""Multi-objective quality indicators."""

from __future__ import annotations

import numpy as np

from saealib.comparators import ParetoDominator
from saealib.exceptions import ValidationError


def _objective_array(values: np.ndarray, n_obj: int | None = None) -> np.ndarray:
    """Convert objective values to a two-dimensional floating-point array."""
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        if values.size == 0 and n_obj is not None:
            values = values.reshape(0, n_obj)
        else:
            values = values.reshape((0, 0) if values.size == 0 else (1, -1))
    if values.ndim != 2:
        raise ValidationError(
            "objective values must be a one- or two-dimensional array"
        )
    if n_obj is not None and values.shape[1] != n_obj:
        raise ValidationError(
            "objective arrays must have the same number of objectives"
        )
    return values


def _pairwise_distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return Euclidean distances between every row of ``a`` and ``b``."""
    return np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)


def _pairwise_manhattan_distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return Manhattan distances between every row of ``a`` and ``b``."""
    return np.abs(a[:, None, :] - b[None, :, :]).sum(axis=2)


def _pairwise_plus_distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return minimization positive-part distances between rows of ``a`` and ``b``."""
    return np.linalg.norm(np.maximum(a[:, None, :] - b[None, :, :], 0.0), axis=2)


def gd(f: np.ndarray, reference_front: np.ndarray) -> float:
    """Compute the generational distance (GD), using minimization objectives.

    GD is the mean distance from each solution to its nearest reference-front
    point.  An empty obtained front returns ``np.nan``; an empty reference
    front raises :class:`~saealib.exceptions.ValidationError`.  Singleton and
    duplicate points are handled by the same nearest-neighbor formula.

    Parameters
    ----------
    f : np.ndarray
        Obtained objective matrix, shape (n, n_obj).
    reference_front : np.ndarray
        Reference-front objective matrix, shape (n_ref, n_obj).

    Returns
    -------
    float
        Mean nearest-reference Euclidean distance.

    References
    ----------
    Li, B., Li, J., Tang, K., & Yao, X. (2015).
        Many-objective evolutionary algorithms: A survey.
        *ACM Computing Surveys*, 48(1), Article 13.
        https://doi.org/10.1145/2792984
    """
    f, reference_front = _validated_pair(f, reference_front)
    if len(f) == 0:
        return np.nan
    return float(np.min(_pairwise_distances(f, reference_front), axis=1).mean())


def igd(f: np.ndarray, reference_front: np.ndarray) -> float:
    """Compute inverted generational distance (IGD) for minimization objectives.

    IGD is the arithmetic mean of the Euclidean distances from each
    reference-front point to its nearest solution. An empty obtained front
    returns ``np.nan``; an empty reference front raises
    :class:`~saealib.exceptions.ValidationError`. Singleton and duplicate
    points use the same nearest-neighbor formula.

    Parameters
    ----------
    f : np.ndarray
        Obtained objective matrix, shape (n, n_obj).
    reference_front : np.ndarray
        Reference-front objective matrix, shape (n_ref, n_obj).

    Returns
    -------
    float
        Mean nearest-solution Euclidean distance.

    References
    ----------
    Bosman, P. A. N., & Thierens, D. (2003).
        The balance between proximity and diversity in multi-objective
        evolutionary algorithms. *IEEE Transactions on Evolutionary
        Computation*, 7(2).
    """
    f, reference_front = _validated_pair(f, reference_front)
    if len(f) == 0:
        return np.nan
    return float(np.min(_pairwise_distances(reference_front, f), axis=1).mean())


def gd_plus(f: np.ndarray, reference_front: np.ndarray) -> float:
    """Compute generational distance plus (GD+) for minimization objectives.

    GD+ is the arithmetic mean of the nearest positive-part distances from
    each solution to the reference front. Only objective-wise deterioration of
    a solution relative to a reference point contributes. An empty obtained
    front returns ``np.nan``; an empty reference front raises
    :class:`~saealib.exceptions.ValidationError`.

    Parameters
    ----------
    f : np.ndarray
        Obtained objective matrix, shape (n, n_obj).
    reference_front : np.ndarray
        Reference-front objective matrix, shape (n_ref, n_obj).

    Returns
    -------
    float
        Mean nearest positive-part distance.

    References
    ----------
    Ishibuchi, H., Masuda, H., Tanigaki, Y., & Nojima, Y. (2015).
        Modified distance calculation in generational distance and inverted
        generational distance. *Evolutionary Multi-Criterion Optimization*.
        Springer.
    """
    f, reference_front = _validated_pair(f, reference_front)
    if len(f) == 0:
        return np.nan
    return float(np.min(_pairwise_plus_distances(f, reference_front), axis=1).mean())


def igd_plus(f: np.ndarray, reference_front: np.ndarray) -> float:
    """Compute inverted generational distance plus (IGD+) for minimization objectives.

    IGD+ is the arithmetic mean, over reference-front points, of the nearest
    positive-part distance from a solution. Only objective-wise deterioration
    contributes. An empty obtained front returns ``np.nan``; an empty
    reference front raises :class:`~saealib.exceptions.ValidationError`.

    Parameters
    ----------
    f : np.ndarray
        Obtained objective matrix, shape (n, n_obj).
    reference_front : np.ndarray
        Reference-front objective matrix, shape (n_ref, n_obj).

    Returns
    -------
    float
        Mean nearest positive-part distance.

    References
    ----------
    Ishibuchi, H., Masuda, H., Tanigaki, Y., & Nojima, Y. (2015).
        Modified distance calculation in generational distance and inverted
        generational distance. *Evolutionary Multi-Criterion Optimization*.
        Springer.
    """
    f, reference_front = _validated_pair(f, reference_front)
    if len(f) == 0:
        return np.nan
    return float(np.min(_pairwise_plus_distances(f, reference_front), axis=0).mean())


def spacing(f: np.ndarray, *, squared: bool = False) -> float:
    """Compute the spacing indicator for an obtained front.

    The indicator summarizes how evenly a front is spread, through each
    point's nearest other-point Manhattan distance. The default returns the
    sample standard deviation of those distances, matching the form jMetal
    and pymoo report. ``squared=True`` returns the sample variance instead,
    which is the quantity Schott defines, and is the square of the default.
    Empty and singleton sets return ``np.nan`` because both forms are
    undefined there. Duplicate-only sets with at least two points return
    ``0.0``; duplicates contribute zero nearest-neighbor distances.

    Parameters
    ----------
    f : np.ndarray
        Obtained objective matrix, shape (n, n_obj).
    squared : bool, optional
        Return Schott's variance rather than its square root.

    Returns
    -------
    float
        Sample variance of the nearest-neighbor Manhattan distances when
        *squared* is set, otherwise their sample standard deviation.

    References
    ----------
    Schott, J. R. (1995).
        *Fault Tolerant Design using Single and Multicriteria Genetic
        Algorithm Optimization*. Master's thesis, Massachusetts Institute of
        Technology.
    """
    f = _objective_array(f)
    if len(f) < 2:
        return np.nan
    distances = _pairwise_manhattan_distances(f, f)
    np.fill_diagonal(distances, np.inf)
    nearest = np.min(distances, axis=1)
    variance = float(np.sum((nearest - nearest.mean()) ** 2) / (len(f) - 1))
    return variance if squared else float(np.sqrt(variance))


def spread(f: np.ndarray, reference_front: np.ndarray) -> float:
    """Compute Zhou et al.'s generalized spread indicator.

    ``f`` and ``reference_front`` use minimization objectives.  Follows
    Equation (12) of Zhou et al. (2006): each reference-front point is
    measured against its nearest obtained solution, and the indicator reports
    the deviation of those distances from their mean.  The per-point distance
    ``d(X, S)`` allows a zero self-distance when ``X`` also belongs to ``S``
    (the ``Y != X`` exclusion is not applied across sets), so that a well
    distributed front covering the extremes yields ``0.0``.  This differs from
    jMetal-style implementations, which measure nearest-neighbor distances
    within the obtained front.  An empty obtained front returns ``np.nan``;
    an empty reference front raises
    :class:`~saealib.exceptions.ValidationError`. Singleton and duplicate-only
    obtained sets are evaluated by the generalized formula.

    Parameters
    ----------
    f : np.ndarray
        Obtained objective matrix, shape (n, n_obj).
    reference_front : np.ndarray
        Reference-front objective matrix, shape (n_ref, n_obj).

    Returns
    -------
    float
        Generalized spread value.

    References
    ----------
    Zhou, A., Jin, Y., Zhang, Q., Sendhoff, B., & Tsang, E. P. K. (2006).
        Combining model-based and genetics-based offspring generation for
        multi-objective optimization using a convergence criterion. *IEEE
        Congress on Evolutionary Computation*.

    Deb, K., Pratap, A., Agarwal, S., & Meyarivan, T. (2002).
        A fast and elitist multiobjective genetic algorithm: NSGA-II. *IEEE
        Transactions on Evolutionary Computation*, 6(2).
        https://doi.org/10.1109/4235.996017
    """
    f, reference_front = _validated_pair(f, reference_front)
    if len(f) == 0:
        return np.nan

    nearest = np.min(_pairwise_distances(reference_front, f), axis=1)
    mean_distance = nearest.mean()
    extreme_points = np.vstack(
        [
            reference_front[np.argmax(reference_front[:, j])]
            for j in range(reference_front.shape[1])
        ]
    )
    extremes = np.unique(extreme_points, axis=0)
    boundary = np.min(_pairwise_distances(extremes, f), axis=1).sum()
    denominator = boundary + len(reference_front) * mean_distance
    if denominator == 0.0:
        return 0.0
    return float((boundary + np.sum(np.abs(nearest - mean_distance))) / denominator)


def _validated_pair(
    f: np.ndarray, reference_front: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Normalize and validate a solution/reference-front pair."""
    f_raw = np.asarray(f, dtype=float)
    reference_raw = np.asarray(reference_front, dtype=float)
    if f_raw.ndim == 1 and f_raw.size:
        f_raw = f_raw.reshape(1, -1)
    if reference_raw.ndim == 1 and reference_raw.size:
        reference_raw = reference_raw.reshape(1, -1)
    if f_raw.ndim == 1 and f_raw.size == 0 and reference_raw.ndim == 2:
        f_raw = f_raw.reshape(0, reference_raw.shape[1])
    if reference_raw.ndim == 1 and reference_raw.size == 0 and f_raw.ndim == 2:
        reference_raw = reference_raw.reshape(0, f_raw.shape[1])
    reference_front = _objective_array(reference_raw)
    if len(reference_front) == 0:
        raise ValidationError("reference_front must not be empty")
    f = _objective_array(f_raw, reference_front.shape[1])
    return f, reference_front


def hypervolume(f: np.ndarray, reference_point: np.ndarray) -> float:
    """
    Compute hypervolume indicator (minimization convention).

    The hypervolume (S-metric) indicator was introduced by Zitzler &
    Thiele (1998) as a way to measure the quality of a Pareto front
    approximation without a priori knowledge of the true front.

    Uses a recursive slicing algorithm. Complexity: O(n^(m-1) * n log n),
    where n is the number of points and m is the number of objectives.

    Parameters
    ----------
    f : np.ndarray
        Objective matrix, shape (n, n_obj).
    reference_point : np.ndarray
        Reference (nadir) point, shape (n_obj,). Each component must be
        strictly greater than the corresponding component of at least one
        point in f for a non-zero hypervolume to be returned.

    Returns
    -------
    float
        Hypervolume value.
    """
    f = np.asarray(f, dtype=float)
    reference_point = np.asarray(reference_point, dtype=float)

    if f.ndim == 1:
        f = f.reshape(1, -1)

    # Keep only points that are strictly dominated by the reference point
    mask = np.all(f < reference_point, axis=1)
    f = f[mask]

    if len(f) == 0:
        return 0.0

    return _hv(f, reference_point)


def hypervolume_contributions(
    f: np.ndarray,
    reference_point: np.ndarray | None = None,
    direction: np.ndarray | None = None,
    margin: float = 0.1,
) -> np.ndarray:
    r"""
    Compute exclusive hypervolume contributions for each point.

    The exclusive contribution of point ``i`` is defined as::

        Delta_HV(i) = HV(S) - HV(S \ {i})

    i.e. the volume exclusively dominated by point ``i`` relative to the
    reference point.  Higher is better, consistent with the library-wide
    convention.  Dominated points and points not strictly better than the
    reference point contribute 0.

    This function reuses :func:`hypervolume` and therefore inherits the
    minimisation convention internally.  Use the ``direction`` parameter to
    handle maximisation objectives.

    .. warning::
        Computing hypervolume is exponential in the number of objectives, and
        the leave-one-out loop adds a factor of N.  For many objectives or
        large N this becomes prohibitively expensive.  The hypervolume kernel
        this function calls already processes objectives rather than points,
        the slicing structure of While, Hingston, Barone & Huband (2006), so
        the remaining cost is the repetition itself: a faster contribution
        algorithm has to avoid recomputing the whole volume once per point.
        That work is not to be confused with the WFG benchmark-problem paper
        by Huband et al. (2006), a different paper despite the overlapping
        author list.

    Parameters
    ----------
    f : np.ndarray
        Objective matrix, shape ``(N, n_obj)``.
    reference_point : np.ndarray or None, optional
        Reference point in the *original* objective space, shape
        ``(n_obj,)``.  If ``None``, it is auto-computed from the data with
        a fractional padding controlled by ``margin``.
    direction : np.ndarray or None, optional
        Per-objective optimisation direction: ``+1`` to maximise, ``-1``
        to minimise.  ``None`` defaults to all-minimise (equivalent to
        ``np.full(n_obj, -1.0)``).  Same convention as
        :func:`saealib.comparators.spea2_fitness`.
    margin : float, optional
        Fractional padding used when auto-computing the reference point.
        For each objective axis the auto reference is placed at
        ``nadir + margin * span``.  Ignored when ``reference_point`` is
        provided.

    Returns
    -------
    np.ndarray
        Exclusive hypervolume contribution per point, shape ``(N,)``.
        All values are ``>= 0`` (dominated / out-of-reference points get 0).

    References
    ----------
    Beume, N., Naujoks, B., & Emmerich, M. (2007).
        SMS-EMOA: Multiobjective selection based on dominated hypervolume.
        *European Journal of Operational Research*, 181(3), 1653-1669.
        https://doi.org/10.1016/j.ejor.2006.08.008

    Zitzler, E., & Thiele, L. (1998).
        Multiobjective optimization using evolutionary algorithms - a
        comparative case study.  *PPSN V*, LNCS 1498, pp. 292-301.
    """
    f = np.asarray(f, dtype=float)
    if f.ndim == 1:
        f = f.reshape(1, -1)

    n, n_obj = f.shape
    if n == 0:
        return np.empty(0)

    # Build per-objective sign vector: minimise → s=+1, maximise → s=-1.
    # Multiplying f by s converts every objective to a minimisation objective.
    s = np.ones(n_obj)
    if direction is not None:
        direction = np.asarray(direction, dtype=float)
        s[direction == 1] = -1.0

    g = f * s  # transformed objective matrix (always minimise)

    # Determine reference point in g-space.
    if reference_point is not None:
        ref_g = np.asarray(reference_point, dtype=float) * s
    else:
        nadir = g.max(axis=0)
        ideal = g.min(axis=0)
        span = nadir - ideal
        ref_g = nadir.copy()
        for j in range(n_obj):
            if span[j] == 0.0:
                # Degenerate axis: use absolute pad so ref is strictly > nadir.
                ref_g[j] = nadir[j] + max(margin, 1e-12)
            else:
                ref_g[j] = nadir[j] + margin * span[j]

    total = hypervolume(g, ref_g)

    contrib = np.empty(n)
    for i in range(n):
        g_without = np.delete(g, i, axis=0)
        hv_without = hypervolume(g_without, ref_g)
        contrib[i] = total - hv_without

    # Clip floating-point noise to zero.
    return np.maximum(contrib, 0.0)


def _hv(f: np.ndarray, ref: np.ndarray) -> float:
    n, m = f.shape

    if m == 1:
        return float(ref[0] - np.min(f[:, 0]))

    # Sort ascending by last objective
    order = np.argsort(f[:, -1], kind="stable")
    f_s = f[order]

    hv = 0.0
    for i in range(n):
        upper = f_s[i + 1, -1] if i + 1 < n else ref[-1]
        height = upper - f_s[i, -1]
        if height <= 0.0:
            continue
        proj = _non_dominated(f_s[: i + 1, :-1])
        hv += height * _hv(proj, ref[:-1])

    return hv


def _non_dominated(f: np.ndarray) -> np.ndarray:
    """Return non-dominated subset (minimization).

    Parameters
    ----------
    f : np.ndarray
        Objective matrix, shape (n, n_obj).  Must contain no NaN values.

    Returns
    -------
    np.ndarray
        Rows of ``f`` that are not dominated by any other row.
    """
    n = len(f)
    if n <= 1:
        return f
    # dom[i, j] = True iff row i dominates row j; a row is dominated when any
    # column of dom is True for it (i.e. dom.any(axis=0)[j] is True).
    dom = ParetoDominator().dominance_matrix(f)
    dominated = dom.any(axis=0)
    return f[~dominated]
