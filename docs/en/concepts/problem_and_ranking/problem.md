---
primary_layer: layer1
related_layers: [layer2, layer3]
page_type: concept
---

# Problem

`Problem` brings the objective function, objective directions, constraints, comparison method, and SearchSpace together.
`Optimizer` and each component use this `Problem` to refer to the same optimization target.

For a vector representation, `VectorSpace` is created from `lb`, `ub`, or `variables`.
For a non-vector representation, pass a `SearchSpace` to `space`.
The design of `SearchSpace` and Genomes is covered in [SearchSpace](search_space.md).

## Problem's role

The `Problem` constructor accepts the following arguments.

```python
Problem(
    func, dim, n_obj, direction, lb=None, ub=None,
    comparator=None, constraints=None, *,
    eps_cv=1e-6, eps_obj=None, handler=None, variables=None,
    space=None, evaluation_adapter=None,
)
```

**`func`**: The objective function used by the evaluator.
For a vector problem it normally receives an array of design variables; a non-vector problem may receive the payload produced by its evaluation adapter.
**`dim`**: The number of design-variable dimensions.
It can be omitted when `space` provides `dim`.
**`n_obj`**: The number of objective functions.

**`direction`**: An array of shape `(n_obj,)` giving the optimization direction per objective; each element must be `+1` (maximize) or `-1` (minimize).
Passing any other value raises an exception at construction time.

**`lb`**/**`ub`**: The lower and upper bounds for a vector design space.
They are required when neither `variables` nor `space` supplies the representation.
When `variables` is provided, the bounds are derived from the variables' ranges.

**`comparator`**: Deprecated.
Configure the [Comparator](comparators.md) that compares solutions with `Optimizer.set_comparator()`, which keeps it on the optimizer rather than on the problem.
A value passed here emits a `DeprecationWarning` and is adopted as an explicit comparator of the `Optimizer` created for this problem; a comparator set with `set_comparator()` takes precedence over it.
Without an explicit comparator, `Optimizer` selects one from the number of objectives (`SingleObjectiveComparator` when `n_obj == 1`, `NSGA2Comparator` when `n_obj > 1`) and records that choice as a framework default.
`Problem` keeps its own `comparator` attribute as the default derived from this problem; it is what a standalone `OptimizationState.load()` uses, and it is not the object a run compares with.
A comparator with an unset (`None`) `direction` receives this problem's `direction` when it enters the optimizer slot.

**`constraints`**: A list of inequality constraints (`InequalityConstraint`).
How to define them is covered in [Constrained Optimization](../../tutorials/constraints.md).

**`eps_cv`**/**`eps_obj`**: The tolerance for feasibility judgment and the tolerance for objective-value equality judgment, respectively.
`eps_cv` is only carried over to the default `handler`/`comparator` at constructor time — directly rewriting `problem.eps_cv` after construction has no effect on runtime behavior.
The threshold actually used is `handler.feasibility_threshold`, which is synchronized to `state.comparator`/`pareto_archive` every generation during an `Optimizer` run.
`eps_obj` is deprecated: it only seeds the default comparator, and a value passed here emits a `DeprecationWarning`. Pass the tolerance to the comparator you configure with `Optimizer.set_comparator()` instead.

**`handler`**: The [ConstraintHandler](constraints.md) responsible for aggregating constraint violations and correcting the objective function.
If omitted, `StaticToleranceHandler(eps_cv=eps_cv)` is used.

**`variables`**: Specifies each design variable's type as a list of `Variable`.
This can be omitted for problems with only continuous variables, in which case every dimension is treated as `ContinuousVariable`.
If you want to mix integer and categorical variables, including `IntegerVariable`/`CategoricalVariable` here makes [Crossover](../search_algorithms/crossover.md)/[Mutation](../search_algorithms/mutation.md) automatically assign a different operator per variable type.

**`space`**: A `SearchSpace` that provides Genome representation and services such as sampling, validation, comparison, and distance calculation.
When `space` is passed and `variables`, `lb`, and `ub` are omitted, `dim` is obtained from the SearchSpace.
A non-vector SearchSpace may not have `lb` and `ub`.

**`evaluation_adapter`**: An `EvaluationAdapter` that converts a `GenomeBatch` into the input accepted by the objective function.
Set it explicitly on `Problem` when the representation and objective-input formats do not match.

```{note}
Older tutorials have examples using a `weight=` argument, but the current `Problem` has no such argument.
Passing `weight=` raises `TypeError`.
```

## What Problem holds

`direction` is a `±1` array, unified across all of saealib, representing sign only.
By contrast, the `weights` received by `WeightedSumComparator` or `DecompositionComparator` are non-negative weights for aggregating multiple objectives into a scalar value — a separate concept, independent of direction.

Under this role split, `weights` cannot express the importance (scaling) of an objective itself.
If you want to adjust the magnitude of an objective value, scale it inside `func`.
This is organized into two axes: `direction` expresses sign only, and `weights` expresses only the aggregation weighting.

## Invariants

`direction` must be `+1` or `-1` for each objective, and `dim`, variables, and the SearchSpace representation must agree. If they do not, construction-time validation or candidate processing during evaluation fails.

## Extending Problem

### Implementing a custom Variable

`Variable` (an ABC) requires only two properties, `lb`/`ub`, and one method, `repair(x)`.
The built-in `ContinuousVariable`/`IntegerVariable`/`CategoricalVariable` are all thin implementations that simply project a value onto their own domain; if you need a variable type beyond these (a periodic variable, a log-scale variable, etc.), subclass `Variable` directly.

The following example is a variable that wraps around at the boundary instead of clamping the value.
It can be used for a design variable, like an angle, where you want a value exceeding the upper bound to be treated as continuing from the lower bound.

```python
import numpy as np
from saealib import Variable


class PeriodicVariable(Variable):
    def __init__(self, lb: float, ub: float):
        self._lb = float(lb)
        self._ub = float(ub)

    @property
    def lb(self) -> float:
        return self._lb

    @property
    def ub(self) -> float:
        return self._ub

    def repair(self, x):
        span = self._ub - self._lb
        return self._lb + np.mod(np.asarray(x, dtype=float) - self._lb, span)
```

Where `ContinuousVariable.repair()` keeps out-of-range values pinned to the boundary via `np.clip`, this implementation wraps out-of-range values back from the opposite boundary via `np.mod`.
Note that the value `Variable` represents is a value in the "encoded float space" handled on the `Population` array.
For a variable type that needs a correspondence between category values and an internal index, like `CategoricalVariable`, `repair()` should stay confined to index space, with the actual conversion to category values handled by a separate method.

## External library adapters

`PymooProblem(pymoo_problem, *, eq_tolerance=1e-6, **problem_kwargs)` wraps an already-constructed [pymoo](https://pymoo.org/) problem instance (a built-in benchmark, or existing research code) as a `Problem`, forwarding `problem_kwargs` (`comparator`, `handler`, `eps_cv`, `eps_obj`) to `Problem.__init__`; the forwarded `comparator` and `eps_obj` keep the deprecation described above.
`direction` is always all `-1`, since pymoo problems are unconditionally minimization; pymoo's inequality constraints (`G`) map to `InequalityConstraint` and equality constraints (`H`) to `EqualityConstraint`, both verbatim (no sign flip).

pymoo problems batch-evaluate (`_evaluate(X, out)`), and `PymooProblem.evaluate_batch` exploits this directly by calling the wrapped problem's own `evaluate(X, return_as_dictionary=True)` once for the whole batch; `SerialEvaluator` uses this automatically whenever it's available.
For callers that evaluate row-by-row instead (calling `Problem.evaluate`/`evaluate_constraints` directly rather than going through `SerialEvaluator`), a one-slot cache keyed on `x` still keeps this to exactly one pymoo call per candidate — `evaluate_constraints()` and `evaluate()` are called back-to-back for the same `x`, and would otherwise cost one pymoo call per constraint plus one for the objective.

See [Installation](../../getting_started/installation.md) for the `pymoo` extra.

## Related components

- [Comparator](comparators.md): Swaps out how solutions are compared, via `Optimizer.set_comparator()`
- [ConstraintHandler](constraints.md): Swaps out how constraint violations are handled, via the `handler` argument
- [Crossover](../search_algorithms/crossover.md) / [Mutation](../search_algorithms/mutation.md): Operators applied per variable type as defined in `variables`
- [Constrained Optimization](../../tutorials/constraints.md): How to define constraints and choose a built-in `ConstraintHandler`

## References

- {py:class}`saealib.Problem`
- {py:class}`saealib.PymooProblem`
- {py:class}`saealib.Variable`
- {py:class}`saealib.ContinuousVariable`
- {py:class}`saealib.IntegerVariable`
- {py:class}`saealib.CategoricalVariable`
