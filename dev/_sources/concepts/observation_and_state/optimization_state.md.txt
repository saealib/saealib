---
primary_layer: layer4
related_layers: [layer3]
page_type: concept
---

# OptimizationState

`OptimizationState` is the value that carries an `Optimizer` run's execution context and its checkpoint.
On the Stage compatibility path, a `Stage` receives this state directly and hands the copy it updated with `replace()` to the next `Stage`.

In the structured runtime, a graph-native component never receives the whole `OptimizationState`.
A component receives a `StateView` limited to the `StateKey`s it declared, and returns its changes as a `StatePatch`.
The runtime applies that patch to the `StateStore`, producing the next execution boundary.

## OptimizationState's role

`OptimizationState` gathers three things: the values the Stage compatibility path runs on, the results a user inspects, and the values needed to resume.
It is not where component contracts, graph structure, or compiler rules live.
Ownership of state belongs to `Optimizer` and the state itself on the Stage compatibility path, and to `StateStore` on the graph-native path.

`OptimizationState` produces a new value with `replace()`; `StateStore` applies a `StatePatch` to produce a new generation.
If a component were allowed to mutate `StateStore` or `OptimizationState` directly, neither the compiler nor the runtime could trace writes made outside the contract, or detect a resume point that no longer matches.

## The two state boundaries

`OptimizationState` and `StateView` represent different execution boundaries.

| Boundary | Receives | Returns | Main users |
|---|---|---|---|
| Stage compatibility boundary | `OptimizationState` | `OptimizationState` | existing Stages, sequential compatibility runtime |
| graph-native boundary | a declared `StateView` | `StatePatch` or `NodeResult` | components, compiler, structured runtime |

Conflating the two makes it look as if a graph-native component could read and write arbitrary state.
In a structured pipeline, `ComponentContract` declares which `StateKey`s are read and which are written, and the runtime hands the component only that range.

## What OptimizationState holds

`OptimizationState` holds the values you need to inspect a result or resume a run.

| Value | Contents |
|---|---|
| `problem` | the `Problem` being optimized |
| `population` | compatibility shortcut to the current `Population` |
| `archive` | compatibility shortcut to the `Archive` of evaluated solutions |
| `pareto_archive` | compatibility shortcut to the non-dominated solution set |
| `rng` | the random number generator |
| `fe` | the number of true evaluations |
| `gen` | the generation count |
| `step` | The number of completed flow executions (count, initial value: 0) |
| `decision_count` | The number of confirmed evaluation plans (infill decisions, initial value: 0) |
| `region_counts` | Cumulative iteration counts for named Repeat and Loop regions, keyed by qualified identifier (initial value: `{}`) |
| `comparator` | The `Comparator` resolved for the Optimizer |
| `data` | extra data for Stage compatibility extensions |

`population`, `archive`, and `pareto_archive` refer to named collections internally.
Do not design a new graph-native component to put arbitrary values into `data` — declare a `StateKey` in its `StateContract` instead.

## StateStore, StateView, and StatePatch

`StateStore` is the state store that maps a typed `StateKey` to a value.
The `StateView` handed to a component exposes only the read keys its `ComponentContract` declared.

When a component changes state, it returns a `StatePatch` rather than mutating the store.
The runtime applies the patch to the current store to produce the next state.
This is what prevents a component from incidentally reading or writing state it never declared.

The compiler checks that a `StateContract`'s reads, writes, and exports line up with its `StateBinding`.
The runtime restricts a `StateView` to the declared read keys, and applies a `StatePatch`'s writes to the store's typed keys and generation.
Patches that conflict on the same key, or that delete a key which does not exist, are handled according to the runtime's diagnostics.

```python
from saealib.core import StatePatch, StateView
from saealib.core import RUNTIME_GENERATION


def execute(view: StateView) -> StatePatch:
    current = view.get(RUNTIME_GENERATION)
    return StatePatch(writes={RUNTIME_GENERATION: current + 1})
```

Real `StateKey`s are the typed keys provided by `saealib.core.state`, not strings.
The code above is simplified to show only the boundary: read from the view, return a patch.

## Updating on the Stage compatibility path

A custom `Stage` receives an `OptimizationState` and returns the updated state via `state.replace(...)`.

```python
from saealib import Stage


class LogGenerationStage(Stage):
    name = "log_generation"

    def execute(self, state):
        print(state.gen)
        return state
```

To carry an extra value on the Stage compatibility path, copy `state.data` before passing it to `replace()`.
To carry the same value in a graph-native component, use a `StateKey` and a `StatePatch`.

## Updating and checkpointing

`replace(**kwargs) -> OptimizationState` is the method for updating values on the Stage compatibility path.
`archive` is mutated as it accumulates evaluation results, and `rng` advances its internal state every time it draws a random number.
These are explicit exceptions to the design in which every other value goes through `replace()`.

`save(path)` and `load(path, problem)` handle the npz checkpoint of an `OptimizationState`.
For automatic checkpointing, see [Checkpointing](../../tutorials/checkpoint.md).
The pickle format additionally requires `Optimizer`-side state, so which save path applies is decided by the combination of `CheckpointCallback` and `Optimizer`.

## Choosing a state boundary

- Implementing an existing Stage: use `OptimizationState` and `replace()`.
- Implementing a new graph-native component: use `ComponentContract`, `StateView`, and `StatePatch`.
- Inspecting the result of a run: use the `OptimizationState` returned by `Optimizer.run()` or `Optimizer.iterate()`.
- Resuming a run: use `saealib.context.OptimizationState.load()` and `Optimizer.run_from()`.

## Typical failures

Calling `replace()` on the wrong target on the Stage compatibility path leaves the next Stage without the value it expects.
Reading, writing, or exporting a key absent from the `StateContract` on the graph-native path becomes a diagnostic in the compiler's state-effect validation.
Applying a checkpoint to a different `Problem`, or to an incompatible plan, leaves the saved state and the execution boundary out of step.

## Related components

- [Stage](stage.md): the compatibility execution unit that receives `OptimizationState` directly
- [Framework](../../framework/index.md): how component, contract, graph, and compiler relate
- [Runtime](../../framework/runtime.md): plan, runtime, and the boundary at which a `StatePatch` is applied
- [Population](population.md): the `Population`, `Archive`, and `ParetoArchive` data structures
- [Checkpointing](../../tutorials/checkpoint.md): saving and resuming a checkpoint

## References

- {py:class}`saealib.context.OptimizationState`
- {py:class}`saealib.core.StateStore`
- {py:class}`saealib.core.StateView`
- {py:class}`saealib.core.StatePatch`
- [Framework ComponentContract](../../framework/contract.md): the contract, including state declarations
- [Framework Compiler](../../framework/compiler.md): state-effect validation and diagnostics
