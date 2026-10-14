"""Uniform weight-vector generation for multi-objective decomposition."""

from __future__ import annotations

import math

import numpy as np

from saealib.exceptions import ValidationError


def uniform_weight_vectors(n_obj: int, n_divisions: int) -> np.ndarray:
    """
    Generate weight vectors uniformly distributed on the unit simplex.

    Uses the simplex lattice design described in Das & Dennis (1998), widely
    applied in MOEA/D (Zhang & Li, 2007) and NSGA-III (Deb & Jain, 2014).

    Each weight vector ``w`` satisfies ``sum(w) == 1`` and ``w_i >= 0``.
    The granularity is ``1 / n_divisions``; each component takes a value in
    ``{0, 1/H, 2/H, ..., 1}`` where ``H = n_divisions``.

    The number of vectors returned is ``math.comb(n_obj + H - 1, H)``.

    Parameters
    ----------
    n_obj : int
        Number of objectives. Must be >= 2.
    n_divisions : int
        Number of divisions ``H`` along each axis. Must be >= 1. Larger values
        produce more weight vectors and finer coverage.

    Returns
    -------
    np.ndarray
        Weight matrix, shape ``(N, n_obj)``. Each row sums to 1.0 and all
        entries are non-negative.

    Raises
    ------
    ValueError
        If ``n_obj < 2`` or ``n_divisions < 1``.

    References
    ----------
    :cite:`zhang2007moead`: Zhang, Q., & Li, H. (2007). MOEA/D: A
    multiobjective evolutionary algorithm based on decomposition. *IEEE
    Transactions on Evolutionary Computation*, 11(6), 712-731. (Appendix A
    describes the simplex lattice design originally due to Das & Dennis,
    1998.)

    Examples
    --------
    >>> uniform_weight_vectors(2, 4)
    array([[0.  , 1.  ],
           [0.25, 0.75],
           [0.5 , 0.5 ],
           [0.75, 0.25],
           [1.  , 0.  ]])
    """
    if n_obj < 2:
        raise ValidationError(f"n_obj must be >= 2; got {n_obj}")
    if n_divisions < 1:
        raise ValidationError(f"n_divisions must be >= 1; got {n_divisions}")

    n_vectors = math.comb(n_obj + n_divisions - 1, n_divisions)
    rows: list[list[int]] = []
    _fill(n_obj, n_divisions, n_divisions, [], rows)
    assert len(rows) == n_vectors  # sanity check
    return np.array(rows, dtype=float) / n_divisions


def _fill(
    n_obj: int,
    remaining: int,
    n_divisions: int,
    current: list[int],
    rows: list[list[int]],
) -> None:
    """Enumerate integer partitions of *n_divisions* into *n_obj* parts."""
    if len(current) == n_obj - 1:
        rows.append([*current, remaining])
        return
    for k in range(remaining + 1):
        _fill(n_obj, remaining - k, n_divisions, [*current, k], rows)
