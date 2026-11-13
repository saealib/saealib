---
primary_layer: layer2
page_type: concept
---

# Population

`Population` is a domain data structure for individuals and evaluation results.
On the legacy vector path, it is a structured array with `x`, `f`, `g`, and `cv` columns.
On the current graph-native path, `GenomeBatch` and `SearchSpace` define individual representations, and `Population.x` is a compatibility view available only when possible.

`Initializer` generates Genomes from the Problem's SearchSpace and registers them with their candidate IDs in state and the Archive.
Therefore, not every Problem can be converted into a floating-point `x` array.

## Population's role

`Population` holds Genomes, objective values, constraint values, constraint violations, and Algorithm-specific auxiliary attributes.
A Genome can be represented by a `GenomeBatch` implementation such as `DenseVectorBatch`, `ObjectBatch`, or `PermutationBatch`.
Use `x` or a dense numeric view only when the values can be safely obtained as numeric vectors.

`Population.genomes` is a read-only view borrowed from the live Population's genome storage, not an immutable snapshot.
Subsequent Population operations such as `update_array` may update that storage in place, so the contents observed through an existing `genomes` view can change.

Use `Population` as an instance in ordinary cases, and subclass it only when custom individual management is needed.

Define the attribute schema as a list of `PopulationAttribute(name, dtype, shape, default)` values.
On the legacy vector path, add `x`, `f`, `g`, `cv`, and the auxiliary attributes returned by `Algorithm.get_required_attrs(problem)` to this schema.
`get_required_attrs` returns `[]` by default — an algorithm declares attributes only when it stores algorithm-specific state in the population.
On the Genome-native path, the GenomeBatch and SearchSpace contracts own the representation structure; individuals are not represented by a column schema alone.

## What Population holds

| Method | Role |
|---|---|
| `get_array(key)` | Gets an attribute's raw array as a write-protected view |
| `update_array(key, value)` | Updates an attribute array in bulk |
| `get(key, default=None)` | A safe getter that returns `default` if the attribute doesn't exist |
| `append(element=None, **kwargs)` | Adds a single individual |
| `extend(other)` | Adds a batch of individuals from another `Population` or a dict |
| `extract(indices)` | Extracts a subset as a new `Population`, via an index array or slice |
| `truncate(new_size)` / `delete(index)` / `clear()` | Resizing and deletion |
| `reorder(order)` / `argsort(name, reverse=False)` | Reordering |
| `empty_like(capacity=None)` | Creates an empty `Population` with the same schema |
| `set_cache(key, value)` / `get_cache(key)` | Holds a computed result as a cache valid until the Population changes |
| `pop[i]` / `pop[a:b]` | Returns an `Individual` for a single int, or a `Population` subset for a slice |
| `len(pop)` | The number of individuals |

The cache from `set_cache`/`get_cache` is automatically invalidated whenever an operation that changes the population — `append`/`delete`/`update_array`, etc. — is called.
[NSGA2Comparator](../problem_and_ranking/comparators.md) uses this mechanism when reusing its front-splitting and crowding-distance computation results within a generation.

```python
import numpy as np
from saealib import Population, PopulationAttribute

attrs = [
    PopulationAttribute("x", np.float64, (2,)),
    PopulationAttribute("f", np.float64, (1,)),
    PopulationAttribute("cv", np.float64, ()),
]
pop = Population(attrs, init_capacity=4)
pop.append(x=np.array([0.1, 0.2]), f=np.array([1.0]), cv=0.0)
pop.append(x=np.array([0.3, 0.4]), f=np.array([2.0]), cv=0.0)

pop.x  # design-variable array with shape (2, 2)
pop[0]  # an Individual view of the first individual
pop[0:1]  # a Population containing only the first individual
```

### Individual

`Individual` is a lightweight view of a single individual, obtained via `pop[i]`.
It doesn't duplicate the actual data — it holds only a reference to the source `Population` and its own index.

You can read and write values via `get_readonly_value(key)`/`update_value(key, value)`, or equivalently via attribute access such as `ind.x`/`ind.f = ...`.
Using a stale `Individual` after the source `Population`'s structure (number of individuals or ordering) has changed raises an exception, as an invalid reference.

## Invariants

Individual-attribute arrays share the same individual count and schema, and mutating operations invalidate the cache. Updating them with mismatched row counts or schemas breaks the correspondence between individuals and evaluation results.

## Archive

`Archive` is a concrete class mixing `ArchiveMixin` into `Population`, used to accumulate evaluated solutions without duplicates.

`add(element, **kwargs)` takes almost the same arguments as `append`, but differs in that it ignores duplicate solutions.
The attribute used for duplicate detection is specified via the `key_attr` argument (default `"x"`), with `atol`/`rtol` adjusting the tolerance.
`get_knn(x, k)` provides nearest-neighbor search via a kd-tree (lazily built on first call), used by [LocalSurrogateManager](../surrogate_modeling/surrogate_manager.md)'s default `training_set` when gathering per-candidate local training data.

```python
from saealib import Archive

arc = Archive(attrs, init_capacity=4, key_attr="x")
arc.add(x=np.array([0.1, 0.2]), f=np.array([1.0]), cv=0.0)
arc.add(
    x=np.array([0.1, 0.2]), f=np.array([1.0]), cv=0.0
)  # the duplicate solution is ignored
idx, dist = arc.get_knn(np.array([0.1, 0.2]), k=1)
```

## ParetoArchive

`ParetoArchive` is a concrete class mixing `ParetoMixin` into `Population`, continuously maintaining a non-dominated solution set.

Every time a new solution is added, existing solutions it dominates are removed, and if the new solution is itself dominated by an existing one, the new solution is discarded.
Dominance is judged using a feasibility-first scheme.
A feasible solution always dominates an infeasible one; [Dominator](../problem_and_ranking/dominance.md)'s `dominates` is used only when both are feasible.
Feasibility comes from the problem's constraint definitions when the archive holds them in its `constraints` argument and the rows carry a `g` column: a row is feasible when every per-constraint violation is zero after its own tolerance, the same decision `Result` and the history use.
`Optimizer` creates the archive through the algorithm with the problem's constraints, and a checkpoint restore passes them again.

The `dominator` argument lets you swap out the definition of the dominance relation.
Without constraints, or without a `g` column, feasibility falls back to `cv <= eps_cv`, whose default is `0.0` and matches the constraint-based decision for an aggregation that reaches zero only for a feasible solution.
A search-time ε relaxation stays on the comparator and is not written to the archive.

## Extending Population

`ArchiveMixin`/`ParetoMixin` are designed on the assumption that they're mixed into a subclass of `Population` via multiple inheritance.
If you need custom population-management logic, you can define a new class combining these mixins (`class MyArchive(ArchiveMixin, Population): ...`).
You can also override [Algorithm](../search_algorithms/algorithm.md)'s `population_class`/`archive_class` to swap the Population/Archive that `Initializer` generates for your own custom subclass. Both default to the standard `Population`/`Archive`, so the override is optional.

## Related components

- [Initializer](../execution_and_evaluation/initialization.md): Constructs `Population`/`Archive`/`ParetoArchive` at the start of a run
- [OptimizationState](optimization_state.md): The state object holding the constructed `Population`/`Archive`/`ParetoArchive`
- [Algorithm](../search_algorithms/algorithm.md): Swaps the concrete classes via `population_class`/`archive_class`
- [Comparator](../problem_and_ranking/comparators.md): Reuses sort results via `set_cache`/`get_cache`
- [Dominance](../problem_and_ranking/dominance.md): The `Dominator` `ParetoArchive` uses to judge non-dominated solutions
- [SurrogateManager](../surrogate_modeling/surrogate_manager.md): Gathering local training data via `Archive.get_knn`

## References

- {py:class}`saealib.Population`
- {py:class}`saealib.Individual`
- {py:class}`saealib.PopulationAttribute`
- {py:class}`saealib.Archive`
- {py:class}`saealib.ArchiveMixin`
- {py:class}`saealib.ParetoArchive`
- {py:class}`saealib.ParetoMixin`
