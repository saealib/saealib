---
primary_layer: layer3
related_layers: [layer2]
page_type: concept
---

# OptimizationStrategy

`saealib` delegates the selection of candidates sent for expensive true evaluation to `OptimizationStrategy`.
The Strategy describes candidate generation, prediction, evaluation planning, Feedback, and Population updates as one graph.

## OptimizationStrategy's role

A strategy describes candidate generation, prediction, evaluation planning, Feedback, and Population updates as one declared flow.
The canonical way to write one is `TemplateStrategy` with a `Pipeline` declaration whose Stages reference components by `(slot, role)`; before execution, Optimizer lowers this declaration to a `ComponentGraph` and passes it to the Compiler to create an `ExecutablePlan`.

`build_graph(provider) -> ComponentGraph` remains as the lowering boundary of the declaration, and overriding it in a custom strategy is deprecated: it emits a `DeprecationWarning`, and the replacement is a `TemplateStrategy` with a declared Pipeline.
A strategy whose flow cannot be inspected this way keeps declaring `requires_surrogate: bool` by hand.

`requires_surrogate` is the plain attribute holding the requirement a strategy declares for a `SurrogateManager`.
`OptimizationStrategy` defines its `False` default, `TemplateStrategy` leaves it `None` so that the declaration decides, `DirectStrategy` and `SteadyStateStrategy` declare `False`, `IndividualBasedStrategy` and `PreSelectionStrategy` declare `True`, and `GenerationBasedStrategy` declares nothing, so its attribute is `None` whether or not its flow has surrogate-only generations.
Because the attribute is plain, a class attribute, an attribute inherited from a mixin, an instance assignment, and a value restored from a pickle or a spec all count as the strategy's declaration.
The attribute is not the effective requirement: for a `TemplateStrategy` that keeps the standard lowering, the configuration diagnostics derive the requirement from the declared flow, and it is `True` when any Stage, also inside nested Repeat, Loop, or Branch regions, refers to the surrogate manager slot through a `ComponentRef` or holds a surrogate manager directly.
That derived value decides the `missing_surrogate_manager` and `missing_acquisition` errors, so code that reads `strategy.requires_surrogate` as a boolean cannot rely on it; use `Optimizer.validate()` or `Optimizer.contract_diagnostics()` and its `missing_surrogate_manager` finding to check a configuration.
A declared value that contradicts the derived requirement is reported as a `contradicting_surrogate_requirement` WARNING diagnostic and does not stop the run, and a strategy with a custom `build_graph` keeps using its declared attribute.

## Built-in shorthand and explicit policy components

The built-in constructors are shorthand for a declaration combined with default policy components: `IndividualBasedStrategy(evaluation_ratio)` bundles `RatioEvaluation(evaluation_ratio, sanitize_nonfinite=True)`, `PreSelectionStrategy(n_candidates, n_select)` bundles `TopKEvaluation(min(n_select, n_candidates), sanitize_nonfinite=True)`, and each supplies a default feedback builder.
Both forms produce the same plan and the same results for a fixed seed.

A component set explicitly with `set_evaluation_planner`, `set_feedback_builder`, and similar setters takes precedence over the shorthand's default, and provenance records it as explicitly set.
The number of candidates (`n_candidates`, `n_offspring`) and the number of inner generations (`gen_ctrl`) shape the declaration itself, so they stay constructor arguments and do not move into policy components; constructor arguments are also what the spec round trip and pickle restoration use.

An in-place change of a shorthand parameter, such as `strategy.evaluation_ratio = 0.5`, takes effect at the next commit point: the next time the strategy resolves its declaration for a plan.
It never affects a component set explicitly; that component keeps its own configuration.

## Built-in Strategies

| Class | Parameters | Approach |
|---|---|---|
| `DirectStrategy` | None | Uses no surrogate; truly evaluates every generated candidate |
| `SteadyStateStrategy` | None | Generates one candidate per step and supports asynchronous refill |
| `IndividualBasedStrategy` | `evaluation_ratio: float = 0.1` | Scores every candidate with the surrogate, and truly evaluates only the top `evaluation_ratio` fraction |
| `PreSelectionStrategy` | `n_candidates: int, n_select: int` (both required) | Generates `n_candidates`, scores them, and truly evaluates only the top `n_select` |
| `GenerationBasedStrategy` | `gen_ctrl: int` (required) | Advances `gen_ctrl` generations using only the surrogate, then truly evaluates a single generation |

`IndividualBasedStrategy` selects by fraction of individuals, while `PreSelectionStrategy` selects by count of individuals.
`GenerationBasedStrategy` switches between surrogate and true evaluation by generation, not by individual.
`DirectStrategy` is the comparison baseline that uses no surrogate at all.

For steady-state execution, combine `DirectStrategy` with an
`AsyncEvaluationScheduler`. The scheduler fills available worker slots, polls without
blocking, and commits each completed update in the lifecycle order.

### Each Strategy's pipeline structure

| Class | Pipeline |
|---|---|
| `DirectStrategy` | CountGeneration → Ask → TrueEvaluation → ArchiveUpdate → Tell |
| `IndividualBasedStrategy` | CountGeneration → Ask → SurrogatePredictStage → AcquisitionStage → SortByScore → TrueEvaluation (ratio-based) → ArchiveUpdate → Tell |
| `PreSelectionStrategy` | CountGeneration → Ask (n_candidates) → SurrogatePredictStage → AcquisitionStage → TopKSelection(k=n_select) → TrueEvaluation → ArchiveUpdate → Tell |
| `GenerationBasedStrategy` | SurrogateOnlyLoop (gen_ctrl times) → CountGeneration → Ask → TrueEvaluation → ArchiveUpdate → Tell |

Each Strategy's components include the `EvaluationPlan`, asynchronous evaluation submission and collection, Feedback, and Tell.
A Stage is a compatibility execution unit for part of that work, not the same execution contract as the entire structured Pipeline.
See [Stage](../observation_and_state/stage.md) and [Framework](../../framework/index.md) for details.

## Which Strategy to choose

For problems where evaluation cost is extremely high, `IndividualBasedStrategy` or `PreSelectionStrategy`, which use the surrogate to filter out most candidates, are effective.
Early in the search, when the surrogate's reliability is still low, or when you don't want to pay the surrogate's own training cost frequently, `GenerationBasedStrategy`, which advances multiple generations at once using only the surrogate, is suited.
For problems where the surrogate's approximation error itself is unacceptable, or evaluation cost is sufficiently low, relying solely on true evaluation via `DirectStrategy` is reasonable.

## Behavior of runtime swapping

When you change configuration such as `provider.algorithm` or `provider.surrogate_manager` at a step boundary in `iterate()` or `run()`, the execution environment detects the change, recompiles the plan, and applies it from the next generation.
This procedure works on both the Stage compatibility path and the graph-native path.
When a Component requests recompilation, the plan node returns `RuntimeCommand` or `RequestRecompile` through `NodeResult.commands`.
The request is accepted when the Runtime environment provides `recompile()`, and recompilation happens between steps. Rejection is also a normal result.

## Implementing a custom Strategy

Declare the flow with `TemplateStrategy` and a `Pipeline` whose Stages reference components by `(slot, role)`.
Components come from the Optimizer slots at preparation time, and the strategy can bundle defaults for the slots it owns.

```python
from saealib.pipeline import ComponentRef, Pipeline
from saealib.strategies import TemplateStrategy
from saealib.stages import AskStage, CountGenerationStage, EvaluationPlanStage


class CustomStrategy(TemplateStrategy):
    """Skeleton for a declared candidate-selection flow."""

    def __init__(self, n_offspring):
        self.n_offspring = n_offspring
        super().__init__(
            Pipeline(
                steps=[
                    CountGenerationStage(),
                    AskStage(ComponentRef("proposer"), n_offspring=n_offspring),
                    EvaluationPlanStage(ComponentRef("evaluation_planner")),
                ]
            )
        )
```

Add evaluation submission, Feedback, and Tell Stages according to the composition, and declare a surrogate reference (`ComponentRef("predictor")`) only when the flow needs one; the surrogate requirement is derived from the declaration.

When adjusting only part of an existing Strategy, changing the compatibility Pipeline with [Pipeline.replace/find](../extension_guidelines.md) is preferable to creating a new Strategy.

## Related components

- [Stage](../observation_and_state/stage.md): The contract of each individual pipeline stage a Strategy combines
- [SurrogateManager](../surrogate_modeling/surrogate_manager.md): The prediction mechanism used by strategies with `requires_surrogate=True`
- [Extension guidelines](../extension_guidelines.md): Rearranging stages via `Pipeline.replace`/`find`
- [Components overview](../index.md): The diagram of the overall pipeline structure

## References

- {py:class}`saealib.OptimizationStrategy`
- {py:class}`saealib.IndividualBasedStrategy`
- {py:class}`saealib.GenerationBasedStrategy`
- {py:class}`saealib.PreSelectionStrategy`
- {py:class}`saealib.DirectStrategy`
- {py:class}`saealib.AsyncEvaluationScheduler`
