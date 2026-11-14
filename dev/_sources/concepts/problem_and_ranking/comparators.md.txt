---
primary_layer: layer2
related_layers: [layer3]
page_type: concept
---

# Comparator

`Optimizer` delegates deciding the relative superiority of solutions to `Comparator`, a swappable top-level component.
Pass it with `Optimizer.set_comparator()`.

```python
import numpy as np
from saealib import Optimizer, Problem
from saealib.comparators import HypervolumeComparator

problem = Problem(
    lambda x: np.array([x[0], 1.0 - x[0]]),
    dim=1,
    n_obj=2,
    direction=np.array([-1.0, -1.0]),
    lb=[0.0],
    ub=[1.0],
)
optimizer = Optimizer(problem, seed=0).set_comparator(HypervolumeComparator())
ctx = optimizer.run()
print(ctx.comparator is optimizer.comparator)  # True
```

The comparator is an optimizer binding slot, not part of the problem.
When no comparator is set, `Optimizer` selects one from the number of objectives (`SingleObjectiveComparator` when `n_obj == 1`, `NSGA2Comparator` when `n_obj > 1`) and records that origin as a framework default; a comparator set explicitly always wins.
The comparator receives the problem's `direction` when it enters the slot if its own `direction` is unset.
A comparator whose direction, per-objective weights, or reference points do not match the problem's objective count is reported by `Optimizer.validate()` or `contract_diagnostics()` with diagnostic codes `direction_length_mismatch`, `objective_count_mismatch`, and `reference_points_dimension_mismatch`.
During a run every component reads the active comparator through `state.comparator`, so the epsilon threshold a constraint handler writes each generation reaches the same object the search orders solutions with.

`Problem(comparator=...)` and `Problem(eps_obj=...)` are deprecated.
A comparator passed to `Problem` emits a `DeprecationWarning` and is adopted as an explicit optimizer comparator; `eps_obj` only seeds the default comparator's tolerance.
A `Problem` still exposes the default comparator it derives from its objective count, and `OptimizationState` uses it when a state is created or loaded without an optimizer; resume a run with `Optimizer.run_from()` or `Optimizer.iterate_from()` to keep using that optimizer's comparator.

Islands each keep their own comparator: the default is created per `Optimizer`, so two islands do not share one. A comparator instance passed to `set_comparator()` on several islands stays shared deliberately, and because the runtime writes the handler threshold into that instance, sharing one across islands with different handlers is not recommended.

## Comparator's role

`Comparator` has four abstract methods, including `__init__` itself.

- **`__init__(weights, eps_cv, eps_obj, direction=None)`**: Holds `weights`/`eps_cv`/`eps_obj`/`direction`
- **`sort_population(population) -> np.ndarray`**: Returns an index array ordering the entire population from best to worst
- **`compare_population(population, idx_a, idx_b) -> int`**: Compares two individuals within a population (`-1`: a is better, `1`: b is better, `0`: equal)
- **`compare(fa, cv_a, fb, cv_b) -> int`**: A lightweight version that compares two points directly from their objective values and constraint violation, without going through `Population`

Having `__init__` be an abstract method is different from other components.
Subclasses implementing a custom `Comparator` must always call `super().__init__(weights, eps_cv, eps_obj, direction=...)`.

It also provides three concrete methods for population-relative ranking state that must persist across selections rather than being derived fresh from `f`/`cv` each time.
`get_required_attrs` and `prepare_population` have an empty/no-op default; `rank_population` composes the other two and is not itself a no-op:

- **`get_required_attrs(problem) -> list[PopulationAttribute]`**: Declares the `PopulationAttribute`s the comparator needs `Population`/`Archive` to carry. Empty by default
- **`prepare_population(population) -> None`**: Freshly recomputes and writes those attributes for the given population. No-op by default
- **`rank_population(population) -> np.ndarray`**: `prepare_population` followed by `sort_population`. Environmental selection on a freshly merged pool should call this instead of `sort_population` directly, so persisted state is recomputed for the new set before ranking

A `Comparator` that can derive its full ranking from the population currently being ranked, without needing to carry state across population extraction, needs none of the three — `rank_population` then behaves identically to `sort_population`.
This covers every built-in comparator except `SPEA2Comparator`: even `NSGA2Comparator`'s crowding distance, which depends on the whole population, is recomputed fresh from whatever `f`/`cv` values the given `Population` currently holds.
`SPEA2Comparator` is the built-in example where ranking state must survive environmental selection, because mating selection on the next generation's (smaller) population must reuse fitness computed on the earlier merged pool, not recompute it from scratch.

## Built-in Comparators

| Class | When to use |
|---|---|
| `SingleObjectiveComparator` | Single-objective problems |
| `WeightedSumComparator` | Scalarization via weighted linear combination. Works for both single- and multi-objective |
| `ParetoComparator` | Ranks by dominance relation only (no secondary metric such as crowding) |
| `NSGA2Comparator` | Pareto rank + crowding distance {cite}`deb2002nsga2` |
| `SPEA2Comparator` | SPEA2 fitness, which depends on the entire population {cite}`zitzler2001spea2` |
| `HypervolumeComparator` | Front rank + exclusive HV contribution (SMS-EMOA style) {cite}`beume2007smsemoa` |
| `EpsilonDominanceComparator` | Ranking via ε-dominance {cite}`laumanns2002epsilon` |
| `NSGA3Comparator` | Niche preservation via reference points {cite}`deb2014nsga3` |
| `RNSGA2Comparator` | Preference guidance via user-specified reference points {cite}`deb2006rnsga2` |

`SingleObjectiveComparator(direction=None, *, eps_cv=0.0, eps_obj=1e-6)` allows `direction` to be omitted, in which case it's treated as minimization.

`WeightedSumComparator(direction, *, eps_cv=0.0, eps_obj=1e-6)` requires `direction`; omitting it raises `TypeError`.
This class is the sole exception where the passed `direction` is used directly as the scalarization weights (`score = f @ direction`).
Not just the sign but also the magnitude functions as a weight, which is a treatment specific to this class, different from the general role split described in [Problem](problem.md) where "`direction` is sign only, and weight magnitude is a separate concept."

All built-in comparators default `eps_cv` to `0.0`, the same default as `Problem` and `ParetoArchive`.
Thus `cv == 0` marks a feasible solution.
During an `Optimizer` run, `handler.feasibility_threshold` is copied onto `eps_cv` at preparation and at the end of every generation.
`EpsilonConstraintHandler` uses this mechanism to relax the search without changing reported feasibility.
A comparator's constructor `eps_cv` value is overwritten at preparation and at the end of every generation, so it is not used for comparisons during a run.

`ParetoComparator(direction=None, *, eps_cv=0.0, eps_obj=1e-6, sorter=non_dominated_sort, dominator=None)` ranks the population by dominance relation alone.
It's the common base of `NSGA2Comparator`/`HypervolumeComparator`/`NSGA3Comparator`/`RNSGA2Comparator`/`EpsilonDominanceComparator`, and can also be used standalone as a concrete class.
The `dominator` argument injects a [Dominator](dominance.md) and the `sorter` argument injects a [NonDominatedSorter](nondominated_sorting.md) — independent swap points from each other.

`NSGA2Comparator` adds secondary ranking by crowding distance on top of `ParetoComparator`.
The sort result is stored in `Population`'s cache (`get_cache`/`set_cache`) and reused within the generation until the population changes.

### Population-relative Comparators

`SPEA2Comparator` and `HypervolumeComparator` both raise `NotImplementedError` when `compare()` is called.
This is because both SPEA2's fitness and exclusive HV contribution are metrics that depend on the entire population, and cannot be computed from just two points.
This is not a bug but an intentional design, marked by the class attribute `is_population_relative=True`.

These Comparators can't be used in situations that need a two-point comparison (`compare()`) alone, such as PSO's pbest update or `PairwiseComparisonSet`.
Use `ParetoComparator` instead in such situations.
`compare_population()` (comparison via population indices) is defined on both classes, so things like tournament selection work as-is.

`SPEA2Comparator` additionally uses the `get_required_attrs`/`prepare_population` pair described above: `get_required_attrs` declares a persistent `spea2_fitness` attribute, and `prepare_population` recomputes $S(i)$/$R(i)$/$D(i)$/$F(i)$ over the population it's given and writes them there.
Environmental selection (`TruncationSelection` calling `rank_population`) recomputes fitness fresh over the merged population-and-offspring pool; the individuals that survive truncation carry their fitness values with them, so `compare_population()` calls during the next generation's mating selection reuse the persisted values instead of recomputing over a different-sized set.

`HypervolumeComparator`'s HV computation performs O(N) leave-one-out evaluations per front.
For problems with many objectives, this computational cost grows large.

```{note}
Separate from `HypervolumeComparator`'s internal implementation, there is a public function `saealib.hypervolume(f, reference_point)`.
This can be used standalone as a performance metric for evaluating results after optimization, and is unrelated to `HypervolumeComparator`.
See [Utils](../../api/utils.md) for details.
```

### Comparators using reference points

`NSGA3Comparator(reference_points, direction=None, *, ...)` requires `reference_points` (`shape (n_ref, n_obj)`, points on the unit simplex).
Normally, you pass points generated uniformly via `saealib.utils.weight_vectors.uniform_weight_vectors(n_obj, n_divisions)`.
The `rng` property is lazily generated; during an `Optimizer` run, `Runner` injects a random number generator spawned from `ctx.rng`.
This internal rng is not included when a checkpoint is saved, and is re-spawned on resume.

Unlike `NSGA3Comparator`, `RNSGA2Comparator(reference_points, epsilon=0.001, direction=None, *, ...)` doesn't require the reference points to lie on the unit simplex — you can specify the objective values you actually want (an aspiration point) directly.
`epsilon` is the radius of the ε-clearing that thins out solutions close to the same reference point.

`EpsilonDominanceComparator(eps, mode="additive", direction=None, *, ...)` is a thin wrapper that simply swaps `ParetoComparator`'s `dominator` for an [EpsilonDominator](dominance.md).

[DecompositionComparator](decomposition.md) is a Comparator that ranks using MOEA/D-style scalarization.
See that page for details.

## Implementing a custom Comparator

If you need a custom ranking scheme, subclass `Comparator` and implement all four abstract methods.
`__init__` must always call `super().__init__(weights, eps_cv, eps_obj, direction=...)`.

```python
import numpy as np
from saealib import Comparator


class RandomComparator(Comparator):
    """A simple example that always considers only feasibility."""

    def __init__(self, direction=None, *, eps_cv=0.0, eps_obj=1e-6):
        super().__init__(np.empty(0), eps_cv, eps_obj, direction=direction)

    def sort_population(self, population):
        cv = population.get_array("cv")
        return np.argsort(cv)

    def compare_population(self, population, idx_a, idx_b):
        cv = population.get_array("cv")
        return self.compare(None, cv[idx_a], None, cv[idx_b])

    def compare(self, fa, cv_a, fb, cv_b):
        if cv_a > self.eps_cv and cv_b <= self.eps_cv:
            return 1
        if cv_b > self.eps_cv and cv_a <= self.eps_cv:
            return -1
        return 0
```

Set the class attribute `is_population_relative = True` and have `compare()` raise a `NotImplementedError` explaining why when the metric's pairwise ordering genuinely cannot be defined from two isolated objective/CV tuples — as with SPEA2 fitness or exclusive hypervolume contribution — not merely because the metric happens to depend on the whole population (`NSGA2Comparator`'s crowding distance also does, yet its `compare()` is well-defined via plain Pareto dominance).
Separately, if that metric also needs to survive being carried into a smaller population — as SPEA2 fitness must survive truncation to remain valid for the next generation's mating selection — override `get_required_attrs` to declare where it's stored and `prepare_population` to recompute it fresh; without that, a later read sees a value computed for a different-sized population.

## Related components

- [Dominator](dominance.md): The `dominator` argument of the `ParetoComparator` family
- [NonDominatedSorter](nondominated_sorting.md): The `sorter` argument of the `ParetoComparator` family
- [Decomposition](decomposition.md): The scalarization functions used by `DecompositionComparator`
- [Problem](problem.md): The default comparator a `Problem` exposes, and the deprecated `comparator` argument
- [ParentSelection](../search_algorithms/parent_selection.md) / [SurvivorSelection](../search_algorithms/survivor_selection.md): Operators that use `Comparator` to select individuals
- [Population](../observation_and_state/population.md): The `PopulationAttribute` schema that `get_required_attrs` extends

## References

- {py:class}`saealib.Comparator`
- {py:class}`saealib.SingleObjectiveComparator`
- {py:class}`saealib.WeightedSumComparator`
- {py:class}`saealib.ParetoComparator`
- {py:class}`saealib.NSGA2Comparator`
- {py:class}`saealib.SPEA2Comparator`
- {py:class}`saealib.HypervolumeComparator`
- {py:class}`saealib.EpsilonDominanceComparator`
- {py:class}`saealib.NSGA3Comparator`
- {py:class}`saealib.RNSGA2Comparator`
- {py:func}`saealib.hypervolume`
- {py:func}`saealib.comparators.default_comparator`
