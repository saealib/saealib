---
primary_layer: layer1
related_layers: [layer2]
page_type: tutorial
---

# Multi-Objective Optimization

Solve multi-objective optimization problems with trade-offs between objectives, using `saealib`.

`minimize()` uses default components based on the number of objectives.
Use `Optimizer.set_*()` to configure the algorithm, surrogate model, or evaluation strategy individually.

This page covers what's specific to problems with two or more objectives: choosing a comparator and extracting the Pareto front.

:::{admonition} What you'll be able to do
:class: tip

By the end of this page, you'll be able to choose a Comparator for a multi-objective problem and extract the resulting Pareto front.
:::

For the basic single-objective procedure, use [Single-objective optimization](single_objective.md); if you need a custom Comparator, see [Custom components](custom_components.md).
This page runs a multi-objective `Problem`, selects a Comparator, and extracts the Pareto front.

## Set up the problem

When multiple objective functions are in a trade-off relationship, there exist solutions where improving one worsens another.

Under this relationship, the set of solutions not dominated by any other solution in every objective is called the **Pareto front**.

Here, as an example, we minimize the ZDT1 function built into `saealib`.

```python
from saealib.benchmarks import zdt1

problem = zdt1(n_var=10)
```

`zdt1` is a `Problem` instance returning a two-objective benchmark problem with a convex Pareto front.

## Run minimize

Passing a `Problem` instance directly carries over the number of objectives from it.

```python
from saealib import minimize

result = minimize(problem, max_fe=2000, seed=0)

print(result.x.shape)  # (n_pareto, dim)
print(result.f.shape)  # (n_pareto, n_obj)
```

Where `result.x`/`result.f` were a single point in the single-objective case, in the multi-objective case they become multiple solutions forming the Pareto front.

## Configure an Optimizer

With `Optimizer(problem)`, you can configure each component independently.
`set_*()` returns the same `Optimizer` for chaining, and `run()` or `iterate()` executes the optimization.

```python
from saealib import Optimizer, Termination, max_fe

optimizer = Optimizer(problem, seed=0).set_termination(Termination(max_fe(2000)))
ctx = optimizer.run()
pareto_f = ctx.pareto_archive.get_array("f")
```

Unconfigured components resolve to their defaults.
Set multi-objective ranking by choosing one of the following Comparators.

## Choose a comparator

In multi-objective optimization, `Comparator` decides the relative superiority between candidate solutions.

If no comparator is set, one is auto-selected based on the number of objectives (`SingleObjectiveComparator` when `n_obj == 1`, `NSGA2Comparator` when `n_obj > 1`).

| Class | Behavior |
|--------|------|
| `NSGA2Comparator` | Diversity maintenance via non-dominated sorting and crowding distance (default) |
| `SPEA2Comparator` | Fitness based on strength of dominance and neighborhood density |
| `HypervolumeComparator` | Superiority judged by hypervolume contribution |
| `EpsilonDominanceComparator` | Superiority judged by epsilon-dominance |
| `NSGA3Comparator` | Diversity maintenance via reference points. Requires `reference_points` |
| `RNSGA2Comparator` | Concentrates solutions near specified reference points. Requires `reference_points` |

Set the comparator on the `Optimizer` with `Optimizer.set_comparator()`.
An explicitly set comparator takes precedence over the default, and the comparator is recorded on the run state as `ctx.comparator`.

```python
from saealib import Optimizer, Termination, max_fe
from saealib.comparators import HypervolumeComparator

optimizer = (
    Optimizer(problem, seed=0)
    .set_comparator(HypervolumeComparator())
    .set_termination(Termination(max_fe(2000)))
)
ctx = optimizer.run()
```

`Problem(comparator=...)` still works but is deprecated; it emits a `DeprecationWarning` and the value is adopted as the optimizer's explicit comparator.

## Extract the Pareto front

After running, `result.ctx.pareto_archive` holds the final Pareto front.

```python
pareto_x = result.ctx.pareto_archive.get_array("x")
pareto_f = result.ctx.pareto_archive.get_array("f")
```

To compute the Pareto front from an arbitrary array of objective values, you can use `non_dominated_sort` directly.

```python
from saealib.comparators import non_dominated_sort

archive_f = result.ctx.archive.get_array("f")
ranks, fronts = non_dominated_sort(archive_f, direction=problem.direction)
front0_f = archive_f[fronts[0]]  # first non-dominated front
```

## Related concepts and reference

- {py:func}`saealib.minimize`
- {py:class}`saealib.Problem`
- {py:class}`saealib.NSGA2Comparator` / {py:class}`saealib.SPEA2Comparator` / {py:class}`saealib.HypervolumeComparator` / {py:class}`saealib.EpsilonDominanceComparator` / {py:class}`saealib.NSGA3Comparator` / {py:class}`saealib.RNSGA2Comparator`
- {py:func}`saealib.non_dominated_sort`
