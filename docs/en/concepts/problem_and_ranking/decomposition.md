---
primary_layer: layer2
related_layers: [layer3]
page_type: concept
---

# Decomposition

`DecompositionComparator` delegates aggregating multiple objectives into a single scalar value to `Decomposition`, a swappable component.
Approaches like MOEA/D {cite}`zhang2007moead`, which guide the search using a per-subproblem weight vector, are realized by combining `Decomposition` with `DecompositionComparator`.

## Decomposition's role

`Decomposition` requires only one method, `aggregate(f, weights, ideal_point) -> np.ndarray`, to be implemented.
`f` is a matrix of objective values with shape `(N, n_obj)`, `weights` is a non-negative weight vector with shape `(n_obj,)`, and `ideal_point` is the ideal point with shape `(n_obj,)`; it returns a scalar score with shape `(N,)`.
Scores are unified under the convention that lower is better.

`f` is passed already in a minimization frame, with the caller's `DecompositionComparator` having already applied the `direction` conversion.
When implementing a custom `Decomposition`, you don't need to apply the `direction` conversion again inside `aggregate`.

## Built-in Decompositions

All of these are based on the MOEA/D paper {cite}`zhang2007moead`.

| Class | Parameters | Characteristics |
|---|---|---|
| `WeightedSumDecomposition` | None | Linear weighted sum, `score = f @ weights`. Simplest, but cannot reach parts of a non-convex front |
| `TchebycheffDecomposition` | None | Chebyshev distance, `score = max_j(w_j * \|f_ij - z_j*\|)`. Can also reach non-convex fronts |
| `PBIDecomposition` | `theta=5.0` | `d1 + theta * d2` (projection distance along the weight-vector direction + penalty for orthogonal distance) |

`TchebycheffDecomposition` internally substitutes `1e-6` for weights that are exactly zero, since using zero directly would degenerate ({cite}`zhang2007moead` Appendix A's convention).

`PBIDecomposition`'s `theta` controls the trade-off between convergence (the projection distance `d1` along the weight-vector direction) and diversity (the penalty on the orthogonal distance `d2`).
A larger value applies a stronger penalty to solutions that deviate from the weight vector's direction.
`theta=5.0` is the default in {cite}`zhang2007moead`.

## DecompositionComparator

`DecompositionComparator(decomposition, weights, ideal_point=None, direction=None, *, eps_cv=0.0, eps_obj=1e-6)` implements MOEA/D-style ranking as a subclass of [Comparator](comparators.md).

The ordering rule prioritizes feasibility (feasibility-first {cite}`deb2000feasibility`), then sorts by ascending aggregated score.

If `ideal_point` is omitted (`None`), `sort_population` dynamically computes the ideal point from the population's feasible individuals.
`compare`, which compares only two points, on the other hand, uses the minimum of those two points' objective values as a local approximation of the ideal point.
These two computation methods may not fully agree.

`weights` carries only non-negative magnitude, with the sign (minimize/maximize) expressed by `direction` — a role split corresponding to the `direction`/`weight` role split described in [Problem](problem.md).

## Implementing a custom Decomposition

If you need a custom scalarization function, subclass `Decomposition` and implement only `aggregate()`.
The following example is a simple implementation that aggregates via a weighted product (assuming all objectives are positive).

```python
import numpy as np
from saealib import Decomposition


class WeightedProductDecomposition(Decomposition):
    """Aggregation via weighted product (a simple example assuming all objectives are positive)."""

    def aggregate(self, f, weights, ideal_point):
        f = np.asarray(f, dtype=float)
        w = np.asarray(weights, dtype=float)
        return np.prod(np.abs(f - ideal_point + 1e-6) ** w, axis=1)
```

Since `f` is passed already converted to a minimization frame, this implementation doesn't need to be aware of `direction`.

## Related components

- [Comparator](comparators.md): The base class `DecompositionComparator` inherits from
- [Problem](problem.md): The `direction`/`weights` role split

## References

- {py:class}`saealib.Decomposition`
- {py:class}`saealib.WeightedSumDecomposition`
- {py:class}`saealib.TchebycheffDecomposition`
- {py:class}`saealib.PBIDecomposition`
- {py:class}`saealib.DecompositionComparator`
