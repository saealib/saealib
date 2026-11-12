---
primary_layer: layer3
page_type: concept
---

# AcquisitionFunction

`SurrogatePredictStage` passes a [Surrogate](surrogate.md)'s predictions to `AcquisitionStage`, which delegates scalar scoring to `AcquisitionFunction`.
`AcquisitionFunction` receives candidates, predictions, the archive, context, and prepared data, but it does not know the Surrogate's internal implementation or the algorithm that produced the predictions.

## AcquisitionFunction's role

`AcquisitionFunction` requires `evaluate(candidates_x, prediction, archive, *, prepared=..., ctx) -> AcquisitionResult`.
For pointwise criteria, the `PointwiseAcquisition` convenience base class splits this into two methods.

**`compute_reference(archive, rng=None) -> Any`**: On `PointwiseAcquisition`, computes a reference value used for scoring (such as the current best value) from the archive.
An acquisition function that doesn't use a reference value may return `None`.

**`score(prediction, reference, rng=None) -> np.ndarray`**: On `PointwiseAcquisition`, computes a score from `SurrogatePrediction` and the reference value.
As with saealib's overall convention, a higher score is better.

`compute_reference()` receives no `ctx`, so a subclass that needs runtime state (such as `LowerConfidenceBound`'s optional `beta_schedule`, which reads `ctx.decision_count`) overrides `prepare(archive, *, ctx)` directly instead; whatever it returns still flows into `score()` as `reference`.

The class attribute `requires_uncertainty: bool` indicates whether this acquisition function needs `SurrogatePrediction.std` (uncertainty).
`direction_sensitive: bool` (default `True`) indicates whether `Optimizer` automatically injects `problem.direction` into this acquisition function at the start of a run.
Acquisition functions with no notion of objective direction, such as feasibility probability, set `direction_sensitive = False` to disable this auto-injection.

## Built-in AcquisitionFunctions

| Class | Characteristics | `requires_uncertainty` |
|---|---|---|
| `MeanPrediction` | The simplest acquisition function, using only the predictive mean | `False` |
| `ExpectedImprovement` | Expected Improvement (EI) {cite}`jones1998ego` | `True` |
| `BatchExpectedImprovement` | Joint EI with batch diversity for qEI-style selection | `True` |
| `LowerConfidenceBound` | Lower Confidence Bound (LCB) {cite}`srinivas2012gpucb` | `True` |
| `MaxUncertainty` | The larger the predictive uncertainty, the better (exploration-leaning) | `True` |
| `EHVIAcquisition` | Expected Hypervolume Improvement {cite}`emmerich2006ehvi,hupkens2015ehvi,daulton2020ehvi`. For multi-objective use | `True` |
| `SMSEGOAcquisition` | An SMS-EMOA-style hypervolume indicator (SMS-EGO, proposed by Ponweiser et al., 2008). For multi-objective use | `True` |
| `ParEGOAcquisition` | ParEGO via random scalarization {cite}`knowles2006parego,chugh2020scalarizing`. For multi-objective use | `True` |
| `ProbabilityOfFeasibility` | Feasibility probability for a single constraint {cite}`schonlau1997pof,gelbart2014pof` | `True` |
| `ProductOfFeasibility` | Product of feasibility probabilities across multiple constraints {cite}`gelbart2014pof` | `True` |
| `NoveltyAcquisition` | Mean distance to the nearest archive points | `False` |
| `InverseDensityAcquisition` | Inverse archive neighborhood density | `False` |
| `MaximinDistanceAcquisition` | Candidate and archive diversity criterion | `False` |

The uncertainty-based acquisitions in the table have `requires_uncertainty=True`.
Archive-based acquisitions (`NoveltyAcquisition`, `InverseDensityAcquisition`, and `MaximinDistanceAcquisition`) do not require predictive uncertainty.
To use an uncertainty-based acquisition function, the `Surrogate` it's paired with must return `std` (`provides_uncertainty=True`). Joint batch acquisitions additionally require a candidate covariance or another explicit joint posterior representation; see the examples for a deterministic implementation of that contract.
Among the built-in surrogates, only `SklearnGPRSurrogate` satisfies this.
See the uncertainty-support table in [Surrogate](surrogate.md) for details.

`MaxUncertainty`/`ProbabilityOfFeasibility`/`ProductOfFeasibility` have `direction_sensitive = False`.
This is because the magnitude of uncertainty or a feasibility probability has no notion of maximizing vs. minimizing direction.

`ProbabilityOfFeasibility`/`ProductOfFeasibility` are used paired with a classification or regression surrogate that predicts a constraint value `g`.
The typical usage is to extract training data with [TrainingSet](training_set.md)'s `ConstraintObjectiveSet`, and combine it with the objective-side acquisition function (e.g. EI) via `CompositeAcquisition`'s `product_combine`.

```python
ei_manager = GlobalSurrogateManager(gp_surrogate, ArchiveObjectiveSet())
pof_manager = GlobalSurrogateManager(
    PerObjectiveSurrogate([gp_g1, gp_g2]),
    ConstraintObjectiveSet(),
)
optimizer.set_surrogate_manager(
    CompositeSurrogateManager({"objective": ei_manager, "feasibility": pof_manager})
)
optimizer.set_acquisition(
    CompositeAcquisition(
        {"objective": ExpectedImprovement(), "feasibility": ProductOfFeasibility()},
        product_combine,
    )
)
```

## The meaning of the weights/direction arguments

Some acquisition functions, such as `MeanPrediction`, can scalarize multi-objective predictions via the `weights` argument.
Use a negative weight for objectives you want to minimize, e.g. `weights=np.array([-1.0])`.

Specifying the `direction` argument produces a sign-only scalarization with no magnitude, and takes priority over `weights`.
If `direction` isn't specified explicitly, `problem.direction` is injected automatically at the start of a run (only for acquisition functions where `direction_sensitive` is `True`).

## Implementing a custom AcquisitionFunction

If you need a custom pointwise scoring scheme, subclass `PointwiseAcquisition` and implement `compute_reference()`/`score()`.
The following example is a simple acquisition function that gives a higher score to candidates whose predictive mean falls further below a threshold.

```python
from saealib import PointwiseAcquisition


class ThresholdAcquisition(PointwiseAcquisition):
    """Gives a higher score to candidates whose predictive mean falls further below a threshold (assumes minimization)."""

    def __init__(self, threshold: float = 0.0):
        self.threshold = threshold

    def compute_reference(self, archive, rng=None):
        return None

    def score(self, prediction, reference=None, rng=None):
        m = prediction.value[:, 0]
        return self.threshold - m
```

Combine it by passing it to `Optimizer.set_acquisition()`.

```{note}
`Optimizer.validate()` checks uncertainty support per prediction channel. If an acquisition requires uncertainty and its matching surrogate does not provide it, validation reports an ERROR diagnostic.
```

## Related components

- [Surrogate](surrogate.md): The source of `SurrogatePrediction`. See here also for the uncertainty-support table
- [SurrogateManager](surrogate_manager.md): Provides predictions to `AcquisitionFunction`
- [TrainingSet](training_set.md): Constraint data extraction paired with `ProbabilityOfFeasibility`/`ProductOfFeasibility`

## References

- {py:class}`saealib.AcquisitionFunction`
- {py:class}`saealib.MeanPrediction`
- {py:class}`saealib.ExpectedImprovement`
- {py:class}`saealib.BatchExpectedImprovement`
- {py:class}`saealib.LowerConfidenceBound`
- {py:class}`saealib.MaxUncertainty`
- {py:class}`saealib.EHVIAcquisition`
- {py:class}`saealib.SMSEGOAcquisition`
- {py:class}`saealib.ParEGOAcquisition`
- {py:class}`saealib.ProbabilityOfFeasibility`
- {py:class}`saealib.ProductOfFeasibility`
- {py:func}`saealib.gp_ucb_beta_schedule`
