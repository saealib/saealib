---
primary_layer: layer2
related_layers: [layer3]
page_type: concept
---

# Initializer

At the start of a run, `Optimizer` delegates construction of the initial `Population`, `Archive`, `ParetoArchive`, and `OptimizationState` to `Initializer`.
To change the sampling method, swap the Initializer rather than the entire Optimizer.

On the current standard path, `GenomeInitializer` samples Genomes from `Problem.space`.
`LHSInitializer`, `RandomInitializer`, and `SobolInitializer` remain as compatibility Initializers that use vector `x` arrays.

## Initializer's role

The implementation boundary for `Initializer` is `initialize(provider, problem) -> OptimizationState`.
`provider` gives access to already constructed Components such as the Algorithm and Evaluator.
`problem` is the target [Problem](../problem_and_ranking/problem.md).

## Built-in Initializers

| Class | Sampling method |
|---|---|
| `GenomeInitializer` | Generate Genomes with `Problem.space.sample()` |
| `LHSInitializer` | `scipy.stats.qmc.LatinHypercube` |
| `RandomInitializer` | `rng.uniform` |
| `SobolInitializer` | `scipy.stats.qmc.Sobol(scramble=True)` |

`GenomeInitializer` registers the Genomes provided by SearchSpace directly in the Archive.
The other three share the constructor `(n_init_archive, n_init_population, seed=None)` and generate initial points in vector form.
All four evaluate the initial Archive, rank it with the Comparator, and pass the top candidates to the initial Population.

```
Sample (n_init_archive points)
  -> Evaluate via provider.evaluator.evaluate_batch
  -> add to archive / pareto_archive
  -> Rank via problem.comparator.rank_population
  -> Feed the top n_init_population into population
```

Because this ranks a freshly assembled Archive rather than reusing state from a prior selection, it calls `rank_population` rather than `sort_population` directly — see [Comparator](../problem_and_ranking/comparators.md) for the distinction.
For a Comparator like `SPEA2Comparator`, this is also what first populates its persisted ranking state (`spea2_fitness`) on the initial Population.

The three classes' implementations are nearly identical except for the one line doing the sampling — this is a deliberate choice for simplicity, a design that doesn't over-abstract the shared processing.

## Base-class helper methods

The `Initializer` base provides two helper methods reusable in custom implementations.

**`_create_attrs(problem, provider)`**: Builds the `PopulationAttribute` values for `Population` and `Archive`.
The legacy vector path uses `x`, `f`, `g`, `cv`, and `id`, then merges in auxiliary attributes required by both the Algorithm (`Algorithm.get_required_attrs`) and the Comparator (`Comparator.get_required_attrs`) — this is how, for example, `SPEA2Comparator`'s `spea2_fitness` attribute ends up in the schema.
`GenomeInitializer` manages Genomes in a dedicated column, so its standard attributes do not include `x`.
The `objective_source` column records the source of each row's objective values; its codes are documented in the `OptimizationState` docstring.

**`_create_context(problem, archive, pareto_archive, population, rng)`**: Builds the `OptimizationState`.
When the Comparator is an `NSGA3Comparator` without an internal random generator, this sets `rng.spawn(1)[0]`.

## Implementing a custom Initializer

To add custom vector sampling, subclass `Initializer` and implement `initialize()`.
For a new Genome representation, implement an Initializer that uses `Problem.space.sample()` and `Problem.space.validate()`.

Genome-native initialization keeps the following order.

1. Create containers with the Algorithm's `population_class`, `archive_class`, and `create_pareto_archive()`; the first two default to the standard `Population`/`Archive`.
2. Build the `OptimizationState`.
3. Generate Genomes with `problem.space.sample(n, rng)` and validate them with `problem.space.validate()`.
4. Evaluate them with `provider.evaluator.evaluate_batch(genomes, problem)`.
5. Register candidate IDs, Genomes, and evaluation results in the Archive and ParetoArchive.
6. Update the evaluation count and pass Comparator-sorted candidates to the Population.

The following skeleton shows the input and output boundary used by a Genome-native Initializer.

```python
from saealib import GenomeInitializer


class CustomGenomeInitializer(GenomeInitializer):
    def initialize(self, provider, problem):
        # Replace problem.space.sample() with custom sampling.
        # Validate each generated Genome with space.validate() before passing it to the Evaluator.
        ...
```

The start and end of the initial evaluation can be observed through `CallbackManager`.
Because the evaluation Request must preserve candidate IDs and Genomes, do not manage Genomes by implicitly converting them to an `x` array.

Swap the Initializer with `Optimizer.set_initializer(initializer)`.

## Related components

- [OptimizationState](../observation_and_state/optimization_state.md): The state object `initialize()` ultimately returns
- [Population](../observation_and_state/population.md): The `Population`/`Archive`/`ParetoArchive` being constructed
- [Evaluator](evaluation.md): Used to evaluate the initial samples
- [CallbackManager](../observation_and_state/callbacks.md): Observing `InitialEvaluationStartEvent`/`InitialEvaluationEndEvent`
- [Comparator](../problem_and_ranking/comparators.md): `rank_population`, used to rank the initial Archive

## References

- {py:class}`saealib.Initializer`
- {py:class}`saealib.LHSInitializer`
- {py:class}`saealib.RandomInitializer`
- {py:class}`saealib.SobolInitializer`
