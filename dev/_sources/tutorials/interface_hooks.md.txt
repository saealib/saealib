---
primary_layer: cross
related_layers: []
page_type: guide
---

# Choose a Hook, Stage, or Callback

The prerequisite is wanting to add observation or post-processing to an existing execution path.
:::{admonition} What you'll be able to do
:class: tip

By the end of this page, you'll be able to distinguish Hooks that change data, Stages that replace execution units, and Callbacks that observe events.
:::

See the [optimization components](../concepts/index.md) index; for a custom contract, continue to [Custom components](custom_components.md).

## Compare the boundaries

| Mechanism | Purpose | Execution point | Can change state? | Suitable uses |
|---|---|---|---|---|
| `with_post` | Post-process an existing Operator's output | After Crossover or Mutation | Can change candidates through its return value | Repair, rounding, and bounding |
| `with_post_fit` | Process after a Surrogate fit | After `fit()` | Can update the model or external state in the Hook | Fit records and model post-processing |
| `Stage` | Implement or replace a compatibility execution unit | At a Stage position in the Pipeline | Can update state by returning an `OptimizationState` | Add generation processing and swap a Stage |
| `CallbackManager` | Observe events | When a registered Event fires | `event.ctx` is read-only; do not use it to replace Pipeline input | Logging, history, and condition checks |

## Change data with a Hook

To change an Operator's candidate array, use `with_post()` rather than `CallbackManager`.
`with_post()` returns a copy with the Hook added without modifying the original instance.

```python
import numpy as np
from saealib.operators import MutationUniform


def snap_to_grid(offspring, parents, rng, ctx):
    return np.round(offspring * 10.0) / 10.0


mutation = MutationUniform(0.3).with_post(snap_to_grid)
```

For post-processing after a Surrogate fit, pass `fn(train_x, train_y, ctx) -> None` to `with_post_fit()`.
See the [Surrogate concept](../concepts/surrogate_modeling/surrogate.md) for the contract used by this Hook.

The context-dependent component hooks themselves — `ParentSelection.select`, `SurvivorSelection.select`, `EvaluationPlanner.plan`, `FeedbackBuilder.build`, `TrainingSet.build`, `AcquisitionFunction.prepare`/`evaluate`, and `SurrogateManager.fit`/`predict` — share one argument rule: a trailing, keyword-only, required `ctx` annotated `OptimizationState | None`.
See [Extension guidelines](../concepts/extension_guidelines.md) for the rule and how custom implementations declare it.

## Replace an execution unit with a Stage

`Stage` is the compatibility boundary `execute(state) -> state`.
Return a new State with `state.replace()` when updating state.
It is not the boundary for implementing a structured Framework Component or Compiler directly; for that, see [Framework extensions](../framework/extensions.md).

```python
from saealib import Stage


class LogGenerationStage(Stage):
    name = "log_generation"

    def execute(self, state):
        print(state.gen)
        return state
```

See the [Stage concept](../concepts/observation_and_state/stage.md) for Pipeline composition and the Stage contract.

## Observe with a Callback

`CallbackManager.register()` registers a function that receives an Event type and `event`.
The `event.ctx` read by the handler is the public read boundary for generation and evaluation counts.

```python
import numpy as np
from saealib import GenerationEndEvent
from saealib import Optimizer, Problem


def objective(x):
    return np.sum(x**2)


problem = Problem(
    objective,
    dim=3,
    n_obj=1,
    direction=np.array([-1.0]),
    lb=[-5.0] * 3,
    ub=[5.0] * 3,
)
optimizer = Optimizer(problem, seed=0)

history = []


def record(event):
    history.append((event.ctx.gen, event.ctx.fe))


optimizer.cbmanager.register(GenerationEndEvent, record)
```

Replacing candidate arrays or rewriting state is outside a Callback's responsibility.
Use Callbacks for event observation, logging, and recording history.

Use [Custom components](custom_components.md) when you need to implement a component contract, and see [Framework extensions](../framework/extensions.md) when changing the contract or Runtime itself.

## Related concepts and reference

- [Callback concept](../concepts/observation_and_state/callbacks.md)
- [Extension guidelines](../concepts/extension_guidelines.md)
- [Stage reference](../api/stages.md)
- [Callback reference](../api/callbacks.md)
