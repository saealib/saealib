---
primary_layer: layer3
page_type: concept
---

# ConstraintHandler

The basic way to define constraints (`InequalityConstraint`/`EqualityConstraint`), how to specify `Problem`'s `handler` argument, and how to choose among the three built-in handlers are covered in [Constrained Optimization](../../tutorials/constraints.md).
This page focuses on writing a custom `ConstraintHandler`, building on that foundation.

## ConstraintHandler's role

`ConstraintHandler` exposes its handling of constraints as a set of swappable lifecycle hooks.

```
Ask            -> [repair(x, constraints, lb, ub)]
               -> evaluate f, g
               -> [compute_cv(constraints, x, g)]        -> cv
               -> [augment_objective(f, constraints, x, g)] -> f'
Tell           -> Comparator(f', cv) with eps_cv = feasibility_threshold
Generation end -> [on_generation_end(gen, population)]
```

Only `compute_cv` is abstract; every other hook has a default implementation.
A custom `ConstraintHandler` only needs to override the hooks it needs.

**`repair(x, constraints, lb, ub, **kwargs)`**: Called after crossover and mutation, before evaluation.
The default is `np.clip(x, lb, ub)` — simply clipping to the bounds.

**`compute_cv(constraints, x, g)`** (abstract): Aggregates the set of constraints into a single `cv` value.

**`augment_objective(f, constraints, x, g)`**: Transforms the objective value using constraint information.
The default is the identity function (no transformation) — this is where you'd override to implement a penalty-function method or the augmented Lagrangian method.
None of the three built-in handlers override this hook, so it's currently left open as a future extension point.

**`feasibility_threshold`** (property): Defaults to `0.0`.
This value is a search-time relaxation: the runtime copies it onto the comparator's `eps_cv` every generation during an `Optimizer` run, and it never decides reported feasibility.

**`on_generation_end(gen, population)`**: Called at the end of a generation.
The default is a no-op; used by handlers with internal state that updates an ε value per generation, such as `EpsilonConstraintHandler`.

## Constraint violation and feasibility

Feasibility is decided from the constraint definitions, not from the aggregate `cv`: a solution is feasible when `violation_from_value` returns zero for every constraint after its own tolerance, that is `g(x) <= threshold` for an inequality and `|h(x)| <= tolerance` for an equality.
`Result`, the history's `best_f` and `feasible_ratio`, the `FeasibilityClassificationSet` training set, and the Pareto archive all use this decision, so relaxing a comparator during search never changes what is reported as feasible.
With `GenerationBasedStrategy` and `gen_ctrl > 0`, the final population may contain predicted rows with placeholder `g=0`, depending on the algorithm's `tell()` implementation; for example, GA can retain them.
The `feasible_ratio` in History's `summary` classifies each generation's `state.population` against the constraint definitions, so these rows may be counted as feasible.
True evaluation values remain in the archive.
To classify rows yourself, apply `violation_from_value` to each row's `g` column: a row is feasible when every result is zero.

The aggregate `cv` has one job: ordering infeasible solutions against each other.
Its contract is `cv == 0` if and only if every per-constraint violation is zero, plus monotonicity.
The default aggregation is `sum_violation`, the sum of the per-constraint violations, which is also what `StaticToleranceHandler` computes.
`Problem(cv_aggregator=...)` replaces it with any callable `cv_aggregator(constraints, x, g) -> float`, including a plain function; without `cv_aggregator`, a handler's `compute_cv` supplies `cv`.

`EpsilonConstraintHandler` is a search-side exception to the aggregation contract: its formula (`|h(x)|` for equalities, `max(0, g(x) - threshold) ** 2` for inequalities) reaches zero only when an equality residual is exactly zero, which is stricter than the constraint tolerance.
That formula is kept for the ε-constrained method, and its relaxation reaches the comparator through `feasibility_threshold`; reported feasibility always comes from the constraint definitions above.

## Hooks overridden by the built-in handlers

| Class | Hooks overridden |
|---|---|
| `StaticToleranceHandler` | `compute_cv` / `feasibility_threshold` |
| `EpsilonConstraintHandler` | `compute_cv` / `feasibility_threshold` / `on_generation_end` |
| `GradientRepairHandler` | `repair` / `compute_cv` |

`EpsilonConstraintHandler` {cite}`mezuramontes2011epsilon` updates ε per generation via a function `schedule(gen) -> float`.
Built-in schedule-generating functions `linear_epsilon_schedule(eps0, n_gen)`/`exponential_epsilon_schedule(eps0, decay)` are also exposed.

`GradientRepairHandler` {cite}`chootinan2006gradientrepair` overrides `repair()` with a simultaneous Moore-Penrose pseudoinverse update over the currently-violated constraints (equality and inequality alike).
Constraints whose `gradient()` returns `None` fall back to a forward-difference numerical approximation rather than being skipped.

## Extension points on InequalityConstraint/EqualityConstraint

`InequalityConstraint` itself also has two extension points.

**`gradient(x)`**: Returns `None` by default.
Override it to return an analytical gradient vector; otherwise `GradientRepairHandler` falls back to a numerical approximation.

**`violation_from_value(g)`**: Defines the conversion from the raw constraint value `g(x)` to a violation amount.
The default is `max(0, g - threshold)`.
`EqualityConstraint` overrides only this method, defining its own conversion, `max(0, |h(x)| - tolerance)`.

## Implementing a custom ConstraintHandler

If you need a custom constraint-handling strategy, subclass `ConstraintHandler` and implement only `compute_cv()`.
The following example is a penalty-function method that adds the constraint violation to the objective value.

```python
from saealib import ConstraintHandler


class PenaltyHandler(ConstraintHandler):
    """Adds the violation amount to the objective value as a penalty."""

    def __init__(self, penalty_coeff: float = 1e3):
        self.penalty_coeff = penalty_coeff

    def compute_cv(self, constraints, x, g):
        return sum(max(0.0, gi) for gi in g)

    def augment_objective(self, f, constraints, x, g):
        cv = self.compute_cv(constraints, x, g)
        return f + self.penalty_coeff * cv
```

While overriding `augment_objective` to add the penalty to the objective value, you can also implement `compute_cv` to always return `0` as the constraint violation, giving a configuration that handles infeasibility purely through the penalty-function method (treating every solution as feasible).

## Related components

- [Constrained Optimization](../../tutorials/constraints.md): Defining constraints and basic usage of the built-in handlers
- [Problem](problem.md): How to pass the `handler` and `cv_aggregator` arguments
- [Comparator](comparators.md): Where `feasibility_threshold` is synchronized to, as `eps_cv`

## References

- {py:class}`saealib.ConstraintHandler`
- {py:class}`saealib.StaticToleranceHandler`
- {py:class}`saealib.EpsilonConstraintHandler`
- {py:class}`saealib.GradientRepairHandler`
- {py:class}`saealib.InequalityConstraint`
- {py:class}`saealib.EqualityConstraint`
- {py:func}`saealib.problem.constraint.sum_violation`
- {py:func}`saealib.linear_epsilon_schedule`
- {py:func}`saealib.exponential_epsilon_schedule`
