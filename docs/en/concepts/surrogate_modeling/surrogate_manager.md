---
primary_layer: layer3
page_type: concept
---

# SurrogateManager

Where [Surrogate](surrogate.md) handles only fit/predict, `SurrogateManager` coordinates model fitting and batched prediction. An acquisition configured on the optimizer scores the returned predictions.

`Optimizer.set_surrogate_manager()` is a separate top-level swap point from `Optimizer.set_surrogate()` (a simplified version that wraps a [Surrogate](surrogate.md) in `LocalSurrogateManager`).

## SurrogateManager's role

`SurrogateManager`'s abstract method is `predict()`; fitting and generation hooks have default implementations.

**`predict(candidates_x, archive, *, refit=True, ctx) -> SurrogatePrediction`** (abstract): Predicts the candidates.
With `refit=True` (the default), the surrogate is retrained before prediction.

**`fit(archive, *, ctx) -> None`**: A no-op by default.
A pre-fit hook meant to be called once before a series of `predict(..., refit=False)` calls, used when the archive does not change.

**`last_accuracy: SurrogateAccuracy | None`** (class attribute): The accuracy metric computed by the most recent `fit`.
Covered in detail in [Surrogate accuracy evaluation and dynamic switching](surrogate_switching.md).

**`on_generation_end(gen, archive, ctx=None)`** / **`with_on_generation_end(fn)`**: An end-of-generation hook.
Also extensible via the same copy-and-chain approach.

The values received and returned at the two boundaries are:

| Boundary | Receives | Returns | Main users |
|---|---|---|---|
| Stage compatibility boundary | `archive`, `candidates_x: np.ndarray`, `ctx`, `refit` | `SurrogatePrediction`, fit completion state | compatibility Stages, sequential compatibility runtime |
| graph-native boundary | `GenomeBatch`, `FeatureEncoder`, `FeatureBatch`, declared services | Features the Surrogate can fit and predict, plus predictions | Components, Compiler, structured runtime |

The current vector-form Manager API on the Stage compatibility path uses `fit(archive, *, ctx)` and `predict(candidates_x: np.ndarray, archive, *, refit=True, ctx) -> SurrogatePrediction`.
The current API also includes constructors such as `GlobalSurrogateManager(surrogate, training_set=None, accuracy_evaluator=None)`, `LocalSurrogateManager(surrogate, training_set=None, accuracy_evaluator=None)`, and `PairwiseSurrogateManager(surrogate, training_set=None, n_ref=10)`.
In the structured runtime, the Surrogate input boundary is `GenomeBatch → FeatureEncoder → FeatureBatch → Surrogate`.
`FeatureEncoder` performs semantic conversion and determines the features the surrogate can learn. It differs from a space capability such as `SamplingService`.
Execution preparation selects an encoder only when the prepared strategy consumes it. `VectorSpace` defaults to identity encoding and reports INFO with the selection source and reason; `PermutationSpace` and `SequenceSpace` retain their built-in encodings and report a non-blocking WARNING recommending an explicit choice because encoding affects what the surrogate learns.
An encoder explicitly provided through `SearchSpace.services` takes priority and does not produce a default-selection diagnostic. `ObjectSpace` and custom spaces need to provide an encoder when the selected strategy consumes one.
Inspect the structured diagnostics with `optimizer.contract_diagnostics()`, or read `optimizer.last_contract_diagnostics` after `optimizer.validate()`. `Optimizer.validate()` returns ERROR findings only; INFO and WARNING remain available in the diagnostic stream.

## Built-in SurrogateManagers

| Class | Approach |
|---|---|
| `GlobalSurrogateManager` | Fits globally once over the entire archive and predicts every candidate at once |
| `LocalSurrogateManager` | Fits locally per candidate, using its k nearest neighbors |
| `CompositeSurrogateManager` | Combines named prediction channels |
| `PairwiseSurrogateManager` | Predicts win rates with a pairwise-comparison surrogate |

`GlobalSurrogateManager(surrogate, training_set=None, accuracy_evaluator=None)` uses `ArchiveObjectiveSet()` when `training_set` is omitted.

`LocalSurrogateManager(surrogate, training_set=None, accuracy_evaluator=None)` uses `KNNObjectiveSet(n_neighbors=50)` when `training_set` is omitted.
`n_neighbors` isn't a constructor argument of `LocalSurrogateManager` itself — it's a parameter of the default `training_set`.
Because it reuses and refits the same `surrogate` instance across candidates, it isn't thread-safe.

`CompositeSurrogateManager(managers)` calls each named manager's `predict()` and returns a multi-channel `SurrogatePrediction`.
Use `CompositeAcquisition` to evaluate one acquisition per channel and combine the resulting score arrays with `product_combine` or `rank_weighted_combine(weights=None)`.

`PairwiseSurrogateManager(surrogate, training_set=None, n_ref=10)` uses `PairwiseComparisonSet()` when `training_set` is omitted.

See [TrainingSet](training_set.md) and [Surrogate accuracy evaluation and dynamic switching](surrogate_switching.md), respectively, for details on each manager's `training_set`/`accuracy_evaluator` arguments.

### Archive-based acquisitions

These acquisitions score candidate geometry directly against the archive and do not require a surrogate prediction.

| Class | Parameters | Meaning of the score |
|---|---|---|
| `NoveltyAcquisition` | `k=1` | Higher when the average distance to the k nearest archive points is larger |
| `InverseDensityAcquisition` | `eps=1.0` | The reciprocal of the ε-neighborhood density |
| `MaximinDistanceAcquisition` | None | The minimum distance among candidates plus the minimum distance to the archive |

## Extending SurrogateManager and archive-based acquisitions

If you need a custom prediction scheme, subclass `SurrogateManager` and implement `predict()`.
For a custom archive-based criterion, subclass `AcquisitionFunction` and implement its evaluation contract.

```python
import numpy as np
from saealib import AcquisitionFunction, AcquisitionResult, OptimizationState


class ConstantAcquisition(AcquisitionFunction):
    """A minimal acquisition that assigns every candidate the same score."""

    def evaluate(
        self,
        candidates_x,
        prediction,
        archive,
        *,
        prepared=None,
        ctx: OptimizationState | None,
    ):
        return AcquisitionResult(scores=np.ones(len(candidates_x)))
```

## Related components

- [Surrogate](surrogate.md): The fit/predict implementation `SurrogateManager` coordinates
- [TrainingSet](training_set.md): Used by each `SurrogateManager` to extract training data
- [AcquisitionFunction](acquisition_functions.md): Scores the predictions in `AcquisitionStage`
- [Surrogate accuracy evaluation and dynamic switching](surrogate_switching.md): Details of `accuracy_evaluator`/`last_accuracy`
- [strategies](../execution_and_evaluation/strategies.md): Assemble `SurrogatePredictStage` and `AcquisitionStage`

## References

- {py:class}`saealib.SurrogateManager`
- {py:class}`saealib.GlobalSurrogateManager`
- {py:class}`saealib.LocalSurrogateManager`
- {py:class}`saealib.CompositeSurrogateManager`
- {py:class}`saealib.PairwiseSurrogateManager`
- {py:func}`saealib.product_combine`
- {py:func}`saealib.rank_weighted_combine`
- {py:class}`saealib.NoveltyAcquisition`
- {py:class}`saealib.InverseDensityAcquisition`
- {py:class}`saealib.MaximinDistanceAcquisition`
