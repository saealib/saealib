---
primary_layer: layer2
related_layers: [layer3]
page_type: concept
---

# Termination

`Optimizer` delegates the decision of when to stop optimizing to `Termination`, a swappable top-level component.
Pass it via `Optimizer.set_termination(termination)`.
On the Stage compatibility path (the synchronous Pipeline), `is_terminated(ctx) -> bool` is called at the end of each generation.

## Termination's role

`Termination(*conditions)` takes one or more conditions, and stops as soon as any single one becomes true (logical OR).
Passing multiple conditions to the constructor is equivalent to `any_of`, described below.

```python
from saealib import Termination, max_fe, max_gen

termination = Termination(max_fe(2000), max_gen(100))
```

Class methods are also available.

| Class method | Meaning |
|---|---|
| `Termination.any_of(*conditions)` | Stops when any one is met (synonymous with `Termination(*conditions)`) |
| `Termination.all_of(*conditions)` | Doesn't stop until every condition is met |
| `Termination.not_(condition)` | Doesn't stop as long as the condition is not met |

## Built-in termination conditions

Each condition passed to `Termination` is a thin wrapper called `TerminationCondition`. Every built-in factory function returns a `TerminationCondition`.

| Function | Termination condition |
|---|---|
| `max_fe(value)` | The evaluation count reaches `value` (`ctx.fe >= value`) |
| `max_gen(value)` | The generation count reaches `value` (`ctx.gen >= value`) |
| `max_steps(value)` | The number of completed flow executions reaches `value` (`ctx.step >= value`) |
| `f_target(value)` | The archive's best objective value reaches `value`. For single-objective use; automatically determines minimize/maximize from `ctx.direction`. Doesn't trigger while the archive is empty |
| `stalled(window, tol=1e-8)` | Stops if no improvement exceeding `tol` occurs for `window` consecutive generations |

The `TerminationCondition` returned by `stalled` is a stateful condition that holds "the best score so far" internally via closure.
It's meant to be used as one instance per `run` — reusing it across multiple `run`s leaves the previous state behind.

## Extending Termination

`TerminationCondition` isn't an abstract base class — it's a composition wrapper that accepts any callable and provides operator overloads for `|` (OR), `&` (AND), and `~` (NOT).
A plain function returning `OptimizationState -> bool` is automatically converted into a `TerminationCondition` wherever it's used, so you don't need to subclass a base class to add a custom termination condition.

```python
from saealib import Termination, max_fe


def my_condition(ctx):
    f = ctx.archive.get("f")
    return f is not None and len(f) > 0 and f.min() < 1e-6


termination = Termination(max_fe(2000), my_condition)
```

Only wrap it explicitly with `TerminationCondition(func, name=..., doc=...)` if you want to give it an explicit name or description.

Multiple conditions can be combined declaratively via operators.

```python
from saealib import max_fe, max_gen, stalled

both = max_fe(2000) & max_gen(100)  # continues until both are met
either = max_gen(100) | max_fe(2000)  # stops on whichever comes first
not_stalled = ~stalled(20)  # while not stalled
```

```{note}
The `spec` argument on `TerminationCondition`'s constructor is an internal field used when the [Registry](../extension_guidelines.md) serializes it to a preset, and normal user code doesn't need to be aware of it.
```

## Related components

- [OptimizationState](../observation_and_state/optimization_state.md): The `ctx` each termination condition receives
- [Extension guidelines](../extension_guidelines.md): Config-driven construction via `Registry`

## References

- {py:class}`saealib.Termination`
- {py:class}`saealib.TerminationCondition`
- {py:func}`saealib.max_fe`
- {py:func}`saealib.max_gen`
- {py:func}`saealib.f_target`
- {py:func}`saealib.stalled`
- {py:func}`saealib.max_steps`
