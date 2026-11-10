---
primary_layer: layer4
related_layers: [layer3]
page_type: concept
---

# Runtime

The runtime receives a compiled plan and state, then advances execution to an observable state boundary.
The compiler that builds a plan and the runtime that advances it have separate responsibilities.

## ExecutionRuntime's role

`ExecutionRuntime` connects an `ExecutablePlan` to an execution position and state, and orders `StatePatch` values, events, and waits.
Contract compatibility and service resolution belong to the Compiler; the Runtime does not repair an unverified Graph.
`RuntimeSession` owns the current state and execution position, while `StateStore` owns persistent values.

## ExecutablePlan and ExecutionRuntime

`ExecutablePlan` is a verified representation of a `ComponentGraph` that carries required runtime capabilities and diagnostics.
`ExecutionRuntime` is the protocol for initializing a plan, advancing state by one step, and handling requests such as completion or recompilation.
The runtime applies `StatePatch` values, events, and commands from node results in order.

## RuntimeSession and RuntimeStep

`RuntimeSession` is a resumable session containing the plan, current state, execution position, and completion status.
`RuntimeStep` is the result of one advance operation and returns the next session, state, and observability.
This separation lets checkpoints and asynchronous-evaluation waits use the same execution model.

Only a boundary where StatePatches have been applied, events delivered, and commands processed is exposed as the next step.

| Value | Information held |
|---|---|
| `RuntimeSession` | `ExecutablePlan`, current `OptimizationState`, execution position, completion status, and the structured-region frame |
| `RuntimeStep` | Updated state, executed nodes, `NodeResult`, wait status, and the next session |

`ExecutionRuntime.initialize(plan, state)` creates a session, and `advance(session)` returns one `RuntimeStep`.
Applications use the step's `finished`, `observable`, and `session` fields to choose the next operation or resume position.

## PipelineRuntime and asynchronous waiting

`PipelineRuntime` executes an ordinary pipeline in order, while `AsyncPipelineRuntime` handles wait states for submitting and collecting asynchronous evaluations.
Because a submitted asynchronous request may not complete immediately, the runtime retains its state and returns to the next poll.
When evaluation completes, it applies the resulting update as a patch before resuming later stages.

`RuntimeSession` retains this wait position, and `RuntimeStep` reports whether progress occurred and returns the next session.
`AsyncEvaluator` and the scheduler implement asynchronous evaluation; `AsyncPipelineRuntime` connects them to the plan's execution boundary.

### Runner

`Runner` obtains a plan and initial state from `Optimizer`, iterates the runtime, and exposes only state boundaries as a thin internal implementation.
Treat it as an internal component for driving the execution runtime, not as a stable configuration API for applications.
For ordinary use, call `minimize()`, `maximize()`, `Optimizer.run()`, or `Optimizer.iterate()`.

## Construction time and use time

`ExecutionRuntime.initialize(plan, state)` creates a session, and `advance(session)` returns one `RuntimeStep`.
`RuntimeSession` is an immutable execution snapshot; the Runtime makes the resume position explicit by returning the next session.

## Invariants and diagnostics

The Runtime verifies that the environment provides the capabilities required by the plan, that `NodeResult` StatePatches stay within declared state effects, and that a waiting request is not applied twice.
ERROR-level plan diagnostics, undeclared state writes, invalid execution positions, and uncollectable asynchronous requests stop execution and are reported as Diagnostics or failure results. INFO and WARNING plan diagnostics do not stop execution.
Missing runtime capabilities become the Compiler diagnostic `missing_runtime_capability`.

## Extension points

When adding an execution capability or wait state, define the `execution` contract in `ComponentContract` together with the `RuntimeSession` transition.
When integrating an existing Stage, connect its `OptimizationState` boundary through an adapter and keep it distinct from a graph-native Component's `StateView` boundary.

## Related pages

See [Compiler](compiler.md) for assembling an execution plan and [OptimizationState](../concepts/observation_and_state/optimization_state.md) for state ownership and updates.
See [Framework extensions](extensions.md) for extension procedures and the [API reference](../api/index.md) for public import paths for concrete Runtime types.

## References

- {py:class}`saealib.core.ExecutionRuntime`
- {py:class}`saealib.core.ExecutablePlan`
- {py:class}`saealib.core.StatePatch`
