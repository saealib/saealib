---
primary_layer: cross
related_layers: [layer3, layer4]
page_type: guide
---

# Extension guidelines

This page covers lightweight swaps of existing components and the construction of compatibility Pipelines.
Extensions fall into swapping existing components and extending the framework.

- To add a component within an existing contract, see [Implement a custom Component](../tutorials/custom_components.md).
- To add a contract, SearchSpace, Graph, or Runtime, see [Extend the framework](../framework/extensions.md).

Import-path rules are collected in [Canonical Imports](../api/imports.md).

## Choose an extension point

Choose the shallowest boundary that owns the behavior you want to change.
Move into the Framework only when a new execution contract is required; implement swaps expressible by an existing contract through the tutorial procedure.

| What you want to change | Main boundary | Concept | API reference |
|---|---|---|---|
| Change candidate generation or individual updates | `Algorithm` | [Algorithm](search_algorithms/algorithm.md) | [Algorithms](../api/algorithms.md) |
| Change crossover, mutation, or selection | Operator | [Crossover](search_algorithms/crossover.md), [Mutation](search_algorithms/mutation.md) | [Operators](../api/operators.md) |
| Change a prediction model or training data | `Surrogate`, `SurrogateManager` | [Surrogate](surrogate_modeling/surrogate.md) | [Surrogate models](../api/surrogate.md) |
| Change candidate selection or evaluation order | `OptimizationStrategy`, `Stage` | [Strategy](execution_and_evaluation/strategies.md) | [Strategies](../api/strategies.md) |
| Connect an external library | Explicit Adapter | [Evaluation](execution_and_evaluation/evaluation.md) | [Evaluation](../api/evaluation.md) |
| Add candidate representations or variable constraints | `SearchSpace` | [SearchSpace](problem_and_ranking/search_space.md) | [Space](../api/space.md) |
| Declare inputs, outputs, state, or lifecycle | `ComponentContract` | [Contract](../framework/contract.md) | [Core reference](../api/core.md) |
| Change connections or compatibility checks | `ComponentGraph`, Compiler rule | [Graph](../framework/graph.md), [Compiler](../framework/compiler.md) | [Core reference](../api/core.md) |
| Change execution, resumption, or asynchronous-wait semantics | `ExecutionRuntime` | [Runtime](../framework/runtime.md) | [Execution reference](../api/execution.md) |

Obtain framework contracts from `saealib.core`; do not treat implementation modules as stable extension APIs.

`Algorithm`, `OptimizationStrategy`, `Surrogate`, `AcquisitionFunction`, and `SurrogateManager` each have a swappable contract.
To implement custom search or prediction, implement the relevant domain component and register it through `Optimizer.set_*()`.
To change only part of an existing component, use the following lightweight extension mechanisms.

## Context-dependent hooks

The context-dependent public hooks share one argument rule: a trailing, keyword-only, required `ctx` annotated `OptimizationState | None`.
`ParentSelection.select`, `SurvivorSelection.select`, `EvaluationPlanner.plan`, `FeedbackBuilder.build`, `TrainingSet.build`, `AcquisitionFunction.prepare`/`evaluate`, and `SurrogateManager.fit`/`predict` all follow it.
Calls that pass `ctx` positionally or omit it raise `TypeError`; pass it by keyword instead.
Custom overrides declare `ctx` the same way.

```python
from saealib import OptimizationState, ParentSelection


selection.select(
    population,
    n_pair=4,
    n_parents=2,
    rng=rng,
    ctx=state,
)


class MySelection(ParentSelection):
    def select(
        self,
        population,
        n_pair,
        n_parents,
        rng=np.random.default_rng(),
        *,
        ctx: OptimizationState | None,
    ):
        ...
```

Selection needs comparison and randomness, so every concrete built-in selection raises `ValidationError` for `ctx=None`.
Standalone calls made outside a run pass `ctx=None` explicitly; the other families keep their own required-context rejections.
Hooks that stay pure computation — `Surrogate.fit`/`predict`, crossover/mutation bodies, `Comparator`, `Decomposition`, `Variable` — take no `ctx`.

```{note}
Passing a whole `OptimizationState` into a component is the full-context escape hatch.
A custom component may also ignore `ctx` entirely and still work with a minimal declaration.
```

## with_post / with_post_fit

[Crossover](search_algorithms/crossover.md)/[Mutation](search_algorithms/mutation.md)'s `with_post(fn)` and [Surrogate](surrogate_modeling/surrogate.md)'s `with_post_fit(fn)` add post-processing to an existing instance without subclassing.
They don't modify the original instance — they return a copy with the hook added.

Typical uses are adding a repair function to `Crossover`/`Mutation`, or post-fit processing on `Surrogate`.
Each component page's "Extension hooks" section has concrete examples per component.

## Pipeline and Stage

`Stage` and `Pipeline` have two boundaries: a compatibility execution path and a structured DSL.
Stage is a compatibility execution unit that receives `OptimizationState` and returns updated state.
A structured Pipeline instead holds graph-native components, nested Pipelines, `Repeat`, `Loop`, and `Branch`; it does not execute state directly.

When placing a Stage in a structured Pipeline, make the compatibility boundary explicit with `stage_component(stage)`.
Optimizer lowers this DSL to a Graph internally and converts it into a plan executable by the structured Runtime.
Ordinary users do not need to call the Compiler directly.

`pipeline.replace("name", entry)` replaces a structural entry, and `pipeline.find("name", recursive=False)` searches for an entry by name.

```python
from saealib import Pipeline, Stage
from saealib.stages import stage_component
from saealib.stages import CountGenerationStage


class DoubleCountStage(Stage):
    name = "count_generation"

    def execute(self, state):
        return state.replace(gen=state.gen + 2)


pipeline = Pipeline(
    steps=[
        stage_component(CountGenerationStage()),
        # Add a graph-native component or another Stage adapter next.
    ]
)
pipeline.replace("count_generation", stage_component(DoubleCountStage()))
```

[Stage](observation_and_state/stage.md) explains the Stage contract, built-in Stages, and compatibility custom Stages.
This page explains which boundary to choose when replacing an existing configuration.

## CallbackManager

[CallbackManager](observation_and_state/callbacks.md) is an observation mechanism that calls handlers when Events fire.
Event fields such as `candidates` are for observation; reassigning them does not change Pipeline inputs.
Choose `with_post()` / `with_post_fit()` for data-flow post-processing, `Stage` for replacing an execution unit, and `Runtime` for changing execution structure.

Changing `optimizer.set_*()` or component attributes at an `iterate()` / `run()` step boundary makes the execution environment detect the change, recompile the plan, and apply it from the next generation.
This procedure works on both the Stage compatibility path and the graph-native path.
For a path where a Component requests recompilation, see [OptimizationStrategy](execution_and_evaluation/strategies.md), "Behavior of runtime swapping."
Callbacks are observation-only; handlers cannot return a `RuntimeCommand`.
Changes made by calling `optimizer.set_*()` in a Callback closure are applied through the path above, but configuration changes are documented as an `iterate()` procedure.

## Registry

`saealib.registry` is a mechanism for constructing an actual instance from a name (string) or a spec (`{"type": "Name", "params": {...}}`).
Where `with_post`/`Pipeline`-`Stage`/`CallbackManager` are mechanisms for "changing behavior at runtime," Registry serves a different purpose: "assembling components from strings or a config file."
Use it in situations that don't directly import classes, such as config-driven construction via preset YAML (`Optimizer.set_preset()`) or checkpoint resumption.

**`register(name=None)`** (decorator): Registers a class or function with the registry.
Adding `@register()` to a custom `Algorithm`/`Surrogate` subclass, etc., lets it be referenced by a short name just like a built-in component.

```python
from saealib import register
from saealib.surrogate import Surrogate


@register()
class MyCustomSurrogate(Surrogate):
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    ...
```

`get`/`build`/`to_spec` are not exposed from saealib's top level; import them directly from `saealib.registry`.

**`get(name)`**: Resolves a registered name, or, if unregistered, a dotted path in `"module.submodule.ClassName"` form.

**`build(spec)`**: Recursively builds a spec into an actual instance.
If a value inside the spec is itself a nested spec, it is built recursively.
The form `{"callable": "dotted.path"}` resolves a function or built-in function itself (without calling it).

**`to_spec(obj)`**: The inverse of `build()`.
It reflects the constructor signature, reads same-named attributes, and recursively serializes them into a spec.
Classes with a `_registry_spec` attribute (such as `TerminationCondition`) don't use this generic reflection — they return that attribute directly.
This is the path `Optimizer.save_preset()` uses.

```python
from saealib.registry import build, get, to_spec

obj = build({"type": "MyCustomSurrogate", "params": {"alpha": 2.0}})
get("MyCustomSurrogate")  # -> the MyCustomSurrogate class
to_spec(obj)  # -> {"type": "MyCustomSurrogate", "params": {"alpha": 2.0}}
```

Several component pages have a note saying "class X is not `@register()`ed" — this only matters if you resolve classes by name via the Registry.

## Which mechanism to use

| What you want to do | Mechanism to use |
|---|---|
| Just add post-processing to an existing operator or surrogate | `with_post` / `with_post_fit` |
| Change the order of stages itself | `Pipeline` / `Stage` |
| External observation, logging, or conditional decisions | `CallbackManager` |
| Assemble from a config file or preset | `Registry` |

## Related components

- [Crossover](search_algorithms/crossover.md) / [Mutation](search_algorithms/mutation.md) / [Surrogate](surrogate_modeling/surrogate.md): Components with `with_post`-style hooks
- [Stage](observation_and_state/stage.md): The contract of each stage that `Pipeline` combines
- [CallbackManager](observation_and_state/callbacks.md): The event list and observation mechanism
- [strategies](execution_and_evaluation/strategies.md): When the pipeline is rebuilt

## References

- {py:func}`saealib.register`
