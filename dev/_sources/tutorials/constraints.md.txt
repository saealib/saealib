---
primary_layer: layer1
related_layers: [layer2, layer3]
page_type: tutorial
---

# Constrained Optimization

Solve a problem with constraints on the design variables using `saealib`.

With `minimize()`, you can pass a string or instance for the algorithm, surrogate, and evaluation strategy.
The examples on this page add only problem-specific settings and use defaults for everything else.

:::{admonition} What you'll be able to do
:class: tip

By the end of this page, you'll be able to define constraints on a `Problem`, check feasibility, and choose a `ConstraintHandler`.
:::

For a simple objective-function optimization, use [Single-objective optimization](single_objective.md). To implement custom constraint handling, see [Custom components](custom_components.md).
This page covers defining constraints on a `Problem`, checking feasibility, and choosing a `ConstraintHandler`.

## Set up the problem

Assume a problem that, in addition to the objective function, has an inequality constraint `g(x) <= 0` that the solution must satisfy.

```python
import numpy as np


def expensive_func(x):
    return np.sum(x**2)


def g1(x):
    # require the sum of the design variables to be at least 1
    return 1.0 - np.sum(x)


DIM = 5
LB = [-5.0] * DIM
UB = [5.0] * DIM
```

Only solutions satisfying `g1(x) <= 0` are feasible.

## Define constraints

A constraint is defined with `InequalityConstraint(func, threshold=0.0)` and passed to `Problem`'s `constraints` argument.

```python
from saealib import InequalityConstraint, Problem, minimize

constraint = InequalityConstraint(g1, threshold=0.0)

problem = Problem(
    func=expensive_func,
    dim=DIM,
    n_obj=1,
    direction=np.array([-1.0]),
    lb=LB,
    ub=UB,
    constraints=[constraint],
)

result = minimize(problem, max_fe=1000, seed=0)
print(result.x, result.f)
print(constraint.violation(result.x))  # 0.0 means the constraint is satisfied
```

To impose a constraint of the form `g(x) >= threshold`, pass `func` with its sign flipped.

For an equality constraint `h(x) = 0`, use `EqualityConstraint(func, tolerance=1e-4)` instead of combining two sign-flipped `InequalityConstraint`s; 1e-4 is the default equality tolerance, matching CEC 2006 and pymoo.

```python
from saealib import EqualityConstraint


def h1(x):
    # require the sum of the design variables to be exactly 1
    return np.sum(x) - 1.0


equality = EqualityConstraint(h1, tolerance=1e-4)
```

`EqualityConstraint` treats solutions satisfying `|h(x)| <= tolerance` as feasible.

## Check feasibility

A solution is feasible when every constraint is satisfied after its own tolerance: `g(x) <= threshold` for an inequality and `|h(x)| <= tolerance` for an equality.

```python
g = result.ctx.archive.get_array("g")
feasible = np.array(
    [
        all(
            c.violation_from_value(float(value)) == 0.0
            for c, value in zip(problem.constraints, row)
        )
        for row in g
    ]
)
print(f"feasible: {feasible.sum()} / {len(feasible)}")
```

The same decision drives `result.x`/`result.f`, the history's `best_f` and `feasible_ratio`, training-set labels, and the Pareto archive, so it does not depend on how violations are aggregated into `cv` or on any ε relaxation used during search.
The default criterion of `result.best()` and `result.front()` uses that decision as well: feasible solutions come first and are then filtered by Pareto dominance.
When no solution is feasible, the default criterion uses the first row with the minimum `cv` for a single-objective problem, and `result.front()` for a multi-objective problem returns the non-dominated solutions among the rows with the minimum `cv`; a supplied criterion ranks every row with the minimum `cv`.

## Customize constraint handling with ConstraintHandler

`ConstraintHandler` supplies `cv` when `Problem(cv_aggregator=...)` is not set, and holds the search-side relaxation the comparator uses.

If `Problem`'s `handler` argument is omitted, `StaticToleranceHandler` is used; it aggregates `cv` as the sum of the per-constraint violations, the same as `sum_violation`.

To change only the aggregation, pass `Problem(cv_aggregator=...)` instead of writing a handler. The aggregation contract is `cv == 0` if and only if every violation is zero, and its only role is ordering infeasible solutions; feasibility itself always comes from the constraints.

```python
from saealib.problem import sum_violation


def squared_inequality(constraints, x, g):
    return sum(
        max(0.0, float(gi) - c.threshold) ** 2 for gi, c in zip(g, constraints)
    )


problem = Problem(
    func=expensive_func,
    dim=DIM,
    n_obj=1,
    direction=np.array([-1.0]),
    constraints=constraints,
    cv_aggregator=squared_inequality,
)
```

| Class | Behavior |
|--------|------|
| `StaticToleranceHandler` | Sums the per-constraint violations and keeps a fixed search-time relaxation (default) |
| `EpsilonConstraintHandler` | Shrinks the search-time relaxation toward 0 over generations, gradually driving solutions into the feasible region |
| `GradientRepairHandler` | Repairs infeasible solutions using the constraint gradient |

`EpsilonConstraintHandler` takes a function that receives the generation number and returns the tolerance.
Its relaxation applies to the comparator only; reported feasibility keeps using the constraint tolerances.

```python
from saealib import EpsilonConstraintHandler


def schedule(gen):
    return max(0.0, 1.0 - gen * 0.05)


problem = Problem(
    func=expensive_func,
    dim=DIM,
    n_obj=1,
    direction=np.array([-1.0]),
    lb=LB,
    ub=UB,
    constraints=[constraint],
    handler=EpsilonConstraintHandler(schedule),
)
result = minimize(problem, max_fe=1000, seed=0)
```

## Related concepts and reference

- {py:class}`saealib.InequalityConstraint` / {py:class}`saealib.EqualityConstraint`
- {py:class}`saealib.Problem`
- {py:class}`saealib.ConstraintHandler` / {py:class}`saealib.StaticToleranceHandler` / {py:class}`saealib.EpsilonConstraintHandler` / {py:class}`saealib.GradientRepairHandler`
- {py:func}`saealib.minimize`
