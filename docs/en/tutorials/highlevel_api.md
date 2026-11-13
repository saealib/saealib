---
primary_layer: layer1
related_layers: []
page_type: tutorial
---

# Run an optimization with the high-level API

The prerequisite is an objective function and bounds for the design variables.
:::{admonition} What you'll be able to do
:class: tip

By the end of this page, you'll be able to run minimization or maximization and retrieve solutions and run information from `Result`.
:::

This page runs a basic problem with the high-level API; to choose components individually, see [Swap built-in components](component_swap.md).
For per-generation state access or finer configuration, continue to the [low-level API](lowlevel_api.md).

## Set up the problem

The high-level API accepts `dim`, `lb`, and `ub` together with the objective function.
The objective function receives an array of design variables and returns a scalar or an array with one value per objective.

```python
import numpy as np
from saealib import minimize


def objective(x):
    return np.sum(x**2)


result = minimize(
    objective,
    dim=3,
    lb=[-5.0, -5.0, -5.0],
    ub=[5.0, 5.0, 5.0],
    max_fe=100,
    seed=0,
    verbose=False,
)
```

If you have already constructed a `Problem`, pass that instance as the first argument.
See the [Problem concept](../concepts/problem_and_ranking/problem.md) for objective directions, variables, and constraints.
The [Canonical Imports](../api/imports.md) page summarizes import choices.

## Choose minimize or maximize

When all objectives have the same direction, the function name expresses the intent.
By default, `minimize` minimizes every objective and `maximize` maximizes every objective.

| Requirement | Call | Additional specification |
|---|---|---|
| Minimize all objectives | `minimize(...)` | None |
| Maximize all objectives | `maximize(...)` | None |
| Directions already defined by `Problem` | `minimize(problem)` or `maximize(problem)` | Uses the directions from `Problem` |
| Mix directions across objectives | `minimize(..., direction=[...])` | Specify `minimize` or `maximize` for each objective |

See [Multi-objective optimization](multi_objective.md) for multiple objectives and directions, [Constrained optimization](constraints.md) for constraints, and [Mixed-variable optimization](mixed_variable.md) for mixed variables.

## Control the run

`max_fe` is the upper bound on true objective evaluations.
When omitted, the default is `200 * dim`.
`pop_size` sets the population size, `seed` sets the initialization seed, and `verbose=False` suppresses generation logs.

To choose built-in Algorithm, Surrogate, and Strategy components, pass strings or instances to `algorithm`, `surrogate`, and `strategy`.
The [Swap built-in components](component_swap.md) page covers the selection and swapping procedure.

The `strategy` argument has two valid modes.
When you explicitly provide a strategy, whether as the string shorthand `"ib"`, `"gb"`, or `"ps"`, or as a strategy instance, the bundled preset's `evaluation_planner` and `feedback_builder` are not applied; the strategy's own default policies are used.
When you omit `strategy`, the bundled preset supplies the standard configuration, including `RatioEvaluation` and `ComparatorWorstFallback(MixedFeedback)`.
Both behaviors are correct, and the choice is yours based on the configuration you want to use.

## Read Result

`minimize()` and `maximize()` return a `Result`.
For a single objective, `result.x` holds the best design variables and `result.f` holds the corresponding objective value as a one-element array.
`result.fe` is the number of true evaluations, and `result.gen` is the number of completed generations.
For multiple objectives, `x` and `f` are matrices of the non-dominated solutions under the default criterion, recomputed from the truly evaluated rows of the main archive.

```python
print(result.x)
print(result.f)
print(result.fe, result.gen)

archive = result.ctx.archive
print(archive.get_array("x"))
```

`result.best()` returns one solution and `result.front()` returns the non-dominated set.
Both accept a `Comparator` as `criterion` and use the default criterion when it is omitted: feasibility first, decided by the per-constraint violations (an inequality is satisfied at `0.0` and an equality at its `tolerance`), then Pareto dominance among the feasible solutions.
When no solution is feasible, the default criterion uses the first row with the minimum `cv` for a single-objective problem and the non-dominated solutions among the minimum-`cv` rows for a multi-objective problem; a supplied criterion ranks every row with the minimum `cv`.

```python
from saealib.comparators import ParetoComparator

best_x, best_f = result.best()
front_x, front_f = result.front()
custom_x, custom_f = result.best(
    ParetoComparator(direction=result.problem.direction)
)
```

`result.ctx` is the complete `OptimizationState`.
For per-generation state access, intermediate stopping, or component assembly, continue to the [low-level API](lowlevel_api.md).

## Related concepts and reference

- [Problem](../concepts/problem_and_ranking/problem.md)
- [High-level API reference](../api/highlevel.md)
