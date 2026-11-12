---
primary_layer: layer3
page_type: concept
---

# TrainingSet

`GlobalSurrogateManager`/`LocalSurrogateManager`/`PairwiseSurrogateManager` delegate building the training data to `TrainingSet`, a swappable component.
Pass it via the `training_set` argument.

## TrainingSet's role

`TrainingSet` requires only one method, `build(archive, population, candidate_x=None, *, ctx) -> TrainingData`, to be implemented.
`candidate_x` is an argument `LocalSurrogateManager` passes as the k-NN query center per candidate; `GlobalSurrogateManager` calls it with `None`.

When `ctx` provides no prepared `FeatureEncoder` and the space registers none, `ArchiveObjectiveSet` uses the same supported default encoder as optimizer preparation and emits a Python `UserWarning`; a prepared or explicitly registered encoder keeps priority without the default warning.

`TrainingData` is a dataclass holding `train_x` (normally shape `(n_train, dim)`; only `PairwiseComparisonSet` uses `(n_train, 2*dim)`) and `train_y` (`(n_train, n_obj)` for regression, `(n_train,)` for classification or ranking).

The 8 built-in classes are organized along two orthogonal axes.

- **Data-source axis**: Where training data is drawn from — archive / population / k nearest neighbors / pairs / reference point
- **Labeling axis**: How values are assigned — raw objective value (regression) / binary classification / multi-level ranking / pairwise comparison

## Correspondence with literature patterns

| Pattern | Source | Corresponding class |
|---|---|---|
| P1 CA-LLSO | {cite}`wei2021callso` | `LevelBasedSet` |
| P2 CPS-MOEA | {cite}`zhang2018cpsmoea` | `TopKBipartitionSet` |
| P3 Pairwise SAEA | {cite}`hao2024pairwise` | `PairwiseComparisonSet` |
| P4 SAPSO pbest | (general) | `ReferencePointComparisonSet` |
| P5 CSEA / pre-selection | (general) | `KNNObjectiveSet`, `ArchiveObjectiveSet` |
| P6 Constraint BO | {cite}`regis2005cors,letham2019constraintbo` | `ConstraintObjectiveSet`, `KNNConstraintObjectiveSet` |

## Built-in TrainingSets

| Class | Parameters | Description |
|---|---|---|
| `ArchiveObjectiveSet` | None | Uses the entire archive with raw objective values. `GlobalSurrogateManager`'s default |
| `KNNObjectiveSet` | `n_neighbors=50` | The k nearest archive points to `candidate_x`. `LocalSurrogateManager`'s default |
| `ConstraintObjectiveSet` | None | Uses the entire archive with raw constraint values `g` |
| `KNNConstraintObjectiveSet` | `n_neighbors=50` | The k-NN version of `ConstraintObjectiveSet` |
| `FeasibilityClassificationSet` | `source="archive"` | Binary classification labels via `cv <= eps_cv` |
| `TopKBipartitionSet` | `source="archive", top_ratio=0.5` | After sorting, labels the top `floor(n * top_ratio)` as label=1 and the rest as label=0 |
| `LevelBasedSet` | `source="population", n_levels=5` | After sorting, assigns multi-level labels across `n_levels` equally divided groups |
| `PairwiseComparisonSet` | `source="archive", n_pairs=None, rng=None` | Pairs up two points and labels the win/loss of their comparison |
| `ReferencePointComparisonSet` | `ref_source="population_best"` | A binary label for whether an archive point dominates a reference point |

`ConstraintObjectiveSet`/`KNNConstraintObjectiveSet` raise `ValueError` if the problem has no constraints (`archive.g` has 0 columns).

The `eps_cv` used for `FeasibilityClassificationSet`'s feasibility judgment is obtained from `ctx.problem.eps_cv`, falling back to `1e-6` when `ctx=None`.

Classes with a `source` argument (`FeasibilityClassificationSet`/`TopKBipartitionSet`/`LevelBasedSet`/`PairwiseComparisonSet`) all share the behavior of raising `ValueError` if `source="population"` is specified while `population=None`.

`PairwiseComparisonSet` builds an array of shape `(n_pairs, 2*dim)` by concatenating `train_x = [x_a, x_b]` for each pair `(a, b)`, labeling it `1` if `comparator.compare(f_a, cv_a, f_b, cv_b) <= 0` (a beats or ties b), and `0` otherwise.
If `n_pairs=None`, every pair, `n*(n-1)/2` of them, is used.

```{warning}
`PairwiseComparisonSet`'s `train_x` has the special shape `(n_pairs, 2*dim)`, incompatible in shape with a standard regression surrogate such as `RBFSurrogate`.
It needs to be paired with a dedicated pairwise-comparison surrogate from the [ComparisonSurrogate](surrogate.md) family.
```

Unlike `PairwiseComparisonSet`, `ReferencePointComparisonSet`'s `train_x` is only `(n_archive, dim)`, so it's compatible with `GlobalSurrogateManager`/`LocalSurrogateManager`.

## Implementing a custom TrainingSet

If you need a custom training-data extraction scheme, subclass `TrainingSet` and implement only `build()`.
The following example is an implementation that uses only the most recently added `k` entries as training data.

```python
from saealib import OptimizationState, TrainingData, TrainingSet


class RecentKSet(TrainingSet):
    """Uses only the most recently added k entries as training data."""

    def __init__(self, k: int = 20):
        self.k = k

    def build(
        self,
        archive,
        population,
        candidate_x=None,
        *,
        ctx: OptimizationState | None,
    ):
        x = archive.get_array("x")[-self.k :]
        y = archive.get_array("f")[-self.k :]
        return TrainingData(train_x=x, train_y=y)
```

## Related components

- [SurrogateManager](surrogate_manager.md): The managers with a `training_set` argument
- [Surrogate](surrogate.md): Where `TrainingData` is passed. `PairwiseComparisonSet` needs to be paired with the `ComparisonSurrogate` family
- [Comparator](../problem_and_ranking/comparators.md): The sorting and comparison used by `TopKBipartitionSet`/`LevelBasedSet`/`PairwiseComparisonSet`/`ReferencePointComparisonSet`

## References

- {py:class}`saealib.TrainingSet`
- {py:class}`saealib.TrainingData`
- {py:class}`saealib.ArchiveObjectiveSet`
- {py:class}`saealib.KNNObjectiveSet`
- {py:class}`saealib.ConstraintObjectiveSet`
- {py:class}`saealib.KNNConstraintObjectiveSet`
- {py:class}`saealib.FeasibilityClassificationSet`
- {py:class}`saealib.TopKBipartitionSet`
- {py:class}`saealib.LevelBasedSet`
- {py:class}`saealib.PairwiseComparisonSet`
- {py:class}`saealib.ReferencePointComparisonSet`
