---
primary_layer: layer2
related_layers: [layer3]
page_type: concept
---

# CallbackManager

`Optimizer` delegates notifications about execution progress to `CallbackManager`.
CallbackManager supports observation such as logging, collecting convergence history, and making conditional decisions.
Events observe execution boundaries; changing an Event field does not replace a Pipeline input.

## CallbackManager's role

`CallbackManager` keeps a list of handlers per event type.

| Method | Description |
|---|---|
| `register(event_type, func)` | Registers `func(event)` to be called every time an event of `event_type` fires |
| `dispatch(event)` | Calls every registered handler, in registration order |
| `unregister(event_type, func)` | Removes a registered handler |
| `replace(event_type, old, new)` | Replaces a registered handler with a different one |

An Event's `ctx` is an `_EventContext` Protocol for reading values such as `archive`, `comparator`, `n_obj`, `gen`, and `fe`.
In ordinary sequential execution its concrete value is `OptimizationState`, but the Event's public contract is limited to those read capabilities.
Treat `ctx` as read-only; use an official boundary such as `with_post`, Component, or Stage to change values.

## List of available events

| Event | Fired when | Main fields |
|---|---|---|
| `RunStartEvent` | Once, at the start of a run | — |
| `RunEndEvent` | Once, at the end of a run | — |
| `GenerationStartEvent` | At the start of each generation | — |
| `GenerationEndEvent` | At the end of each generation (before the state is yielded) | — |
| `SurrogateStartEvent` / `SurrogateEndEvent` | Before/after surrogate-based scoring | `offspring` |
| `AcquisitionStartEvent` / `AcquisitionEndEvent` | Before/after acquisition scoring | `offspring` at start; `offspring` and `result` at end |
| `PostCrossoverEvent` | After crossover and repair | `candidates` |
| `PostMutationEvent` | After mutation and repair | `candidates` |
| `PostAskEvent` | After all of `ask()` (crossover and mutation) | `candidates` |
| `PostSurrogateFitEvent` | After the surrogate is fitted | `surrogate`; `train_x` and `train_f` are optional |
| `PostEvaluationEvent` | Once per evaluation plan, after true evaluation of the chosen candidates; evaluations a transition requests at a commit point do not fire it (use `PostEvaluationRequestEvent`) | `offspring` |
| `InitialEvaluationStartEvent` | After initial sampling, before initial evaluation | `candidates_x` |
| `InitialEvaluationEndEvent` | After initial evaluation, before the archive is sorted | `archive` |

`PostSurrogateFitEvent` is fired by the built-in Stage after the Surrogate is fit.
Events fired by built-in Stages may not set `train_x` and `train_f`.
`PostEvaluationEvent` may include `request_id`, `candidate_ids`, and `status` in addition to the evaluated `offspring`.
It is a plan-level event, so a plan with several evaluation requests fires it once, and an evaluation that a transition requests at a component replacement's commit point fires `PostEvaluationRequestEvent` instead (once per request).

## Default logging output

`Optimizer` automatically registers `logging_generation` on `GenerationStartEvent` at construction time.
Configuring the standard library's `logging` module makes per-generation progress (evaluation count, best objective value, or for multi-objective, the size and range of the first front) appear in the log.

## Registering custom handlers and recording convergence history

Register any handler with `cbmanager.register(EventType, handler)`.
To record convergence history, register a handler holding a list accumulated via closure.

```python
import numpy as np
from saealib import GenerationEndEvent, Optimizer, Problem, Termination, max_fe

problem = Problem(
    lambda x: np.array([x[0], 1.0 - x[0]]),
    dim=2,
    n_obj=2,
    direction=np.array([-1.0, -1.0]),
    lb=[0.0, 0.0],
    ub=[1.0, 1.0],
)
optimizer = Optimizer(problem, seed=0).set_termination(Termination(max_fe(20)))

history = []


def record_best(event):
    f = event.ctx.archive.get_array("f")
    history.append(float(f[:, 0].min()))


optimizer.cbmanager.register(GenerationEndEvent, record_best)
ctx = optimizer.run()
print(history)
```

## Tracking hypervolume

`logging_generation_hv(reference_point)` returns a handler that logs the first front's hypervolume relative to the specified reference point, every generation.

```python
import numpy as np
from saealib import (
    GenerationStartEvent,
    Optimizer,
    Problem,
    Termination,
    logging_generation_hv,
    max_fe,
)

problem = Problem(
    lambda x: np.array([x[0], 1.0 - x[0]]),
    dim=1,
    n_obj=2,
    direction=np.array([-1.0, -1.0]),
    lb=[0.0],
    ub=[1.0],
)
optimizer = Optimizer(problem, seed=0).set_termination(Termination(max_fe(20)))

optimizer.cbmanager.register(
    GenerationStartEvent,
    logging_generation_hv(reference_point=np.array([1.1, 1.1])),
)
```

## Replacing the default handler

You can remove the auto-registered `logging_generation` with `unregister(event_type, func)`, or replace it with a different handler via `replace(event_type, old, new)`.

```python
import numpy as np
from saealib import (
    GenerationStartEvent,
    Optimizer,
    Problem,
    Termination,
    logging_generation,
    max_fe,
)

problem = Problem(
    lambda x: np.array([x[0], 1.0 - x[0]]),
    dim=1,
    n_obj=2,
    direction=np.array([-1.0, -1.0]),
    lb=[0.0],
    ub=[1.0],
)
optimizer = Optimizer(problem, seed=0).set_termination(Termination(max_fe(20)))

optimizer.cbmanager.unregister(GenerationStartEvent, logging_generation)
```

## The candidates field is for observation only

The `candidates` field on `PostCrossoverEvent`/`PostMutationEvent`/`PostAskEvent` is for observation; reassigning it inside a handler (`event.candidates = new_array`) has no effect on the pipeline's output.
`GA` keeps using its own local array reference after the event fires, so an in-place change (`event.candidates[:] = ...`) does reach `GA`.
`PSO`, on the other hand, fires this event after `Population.extend()` has already finished copying the candidates, so even an in-place change is too late and has no effect at all.

If you want to actually swap out the candidate array, use `with_post(fn)` on [Crossover](../search_algorithms/crossover.md)/[Mutation](../search_algorithms/mutation.md) instead of `CallbackManager`.
Think of `CallbackManager` as a mechanism for observation (logging, recording, conditional branching decisions) by design, not a means of rewriting pipeline data.

## Switching configuration at runtime

`Event` is for observation, and directly changing `Optimizer` internals from a Callback closure is not the standard procedure.
Use `with_post()` or `with_post_fit()` to change data flow, and switch execution configuration at a step boundary in `iterate()` or `run()`.

When `optimizer.set_*()` or a component attribute changes at a step boundary, the execution environment detects the change, recompiles the plan, and applies it from the next generation.
This procedure works on both the Stage compatibility path and the graph-native path.
For a Component-requested recompilation path, see [OptimizationStrategy](../execution_and_evaluation/strategies.md)'s "Behavior of runtime swapping".
A Callback handler cannot return a `RuntimeCommand`.
A change made by calling `optimizer.set_*()` in a Callback closure is applied through the path above, but configuration changes are documented as a procedure on the `iterate()` side.
When calling a setter inside a callback, register a handler closure that captures the `Optimizer`.

## When to use CallbackManager vs. iterate()

| Aspect | CallbackManager | `iterate()` |
|---|---|---|
| Granularity of invocation | When a specific event occurs | Per generation (a `for` loop on the host side) |
| Main use | Logging, observation, conditional side effects | Intervening in the loop structure itself, e.g. switching components based on surrogate accuracy |
| Relationship to `run()` | Works with either `run()` or `iterate()` | Used in place of `run()` |

The switcher classes in [Surrogate accuracy evaluation and dynamic switching](../surrogate_modeling/surrogate_switching.md) are meant to be used inside an `iterate()` loop.

## CheckpointCallback

`CheckpointCallback` is an example Callback implementation.
The npz format uses `OptimizationState.save()`, while the pickle format uses `Optimizer.save_pickle()`.
Because the saved values and resumption conditions differ by format, see [Checkpointing](../../tutorials/checkpoint.md) for details.
Use this Callback as a reference for registering a custom Callback.

## Related components

- [Extension guidelines](../extension_guidelines.md): When to use `with_post`-style hooks instead
- [Crossover](../search_algorithms/crossover.md) / [Mutation](../search_algorithms/mutation.md): The actual means of swapping the candidate array
- [strategies](../execution_and_evaluation/strategies.md): When a runtime component swap takes effect
- [Surrogate accuracy evaluation and dynamic switching](../surrogate_modeling/surrogate_switching.md): Dynamic switching inside an `iterate()` loop
- [Checkpointing](../../tutorials/checkpoint.md): How to use `CheckpointCallback`

## References

- {py:class}`saealib.CallbackManager`
- {py:class}`saealib.RunStartEvent`
- {py:class}`saealib.RunEndEvent`
- {py:class}`saealib.GenerationStartEvent`
- {py:class}`saealib.GenerationEndEvent`
- {py:class}`saealib.AcquisitionStartEvent`
- {py:class}`saealib.AcquisitionEndEvent`
- {py:class}`saealib.PostSurrogateFitEvent`
- {py:class}`saealib.PostEvaluationEvent`
- {py:class}`saealib.InitialEvaluationStartEvent`
- {py:class}`saealib.InitialEvaluationEndEvent`
- {py:func}`saealib.logging_generation`
- {py:func}`saealib.logging_generation_hv`
- {py:class}`saealib.CheckpointCallback`
