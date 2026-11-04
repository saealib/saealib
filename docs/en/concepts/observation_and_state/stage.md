---
primary_layer: layer2
related_layers: [layer3]
page_type: concept
---

# Stage

Built-in Strategies execute generation processing in units called `Stage`.
Stage is the sequential compatibility surface that supports existing Algorithms and custom Stages.

`Pipeline` is a structured DSL.
It describes graph-native components, nested Pipelines, `Repeat`, `Loop`, and `Branch`, but does not execute `OptimizationState` directly.
Optimizer delegates conversion of a Pipeline into an executable plan to the Compiler internally.
Ordinary users do not need to call the Compiler directly.

This page explains the Stage compatibility contract, built-in Stages, and how to implement a custom Stage.
See [Framework](../../framework/index.md) for implementing a graph-native component.

## Stage's role

The boundary implemented by `Stage` is `execute(state: OptimizationState) -> OptimizationState`.
Stage receives an `OptimizationState`, performs one operation, and returns the updated State.
This boundary differs from a graph-native component's `execute(StateView) -> StatePatch`.

There are three class attributes.

- **`name`**: A machine-readable identifier used for lookup via `pipeline["name"]`
- **`label`**: A human-readable description
- **`notation`**: The LaTeX notation used by `to_pseudocode()`

## Pipeline

`Pipeline` is a structural sequence.  Its entries can be graph-native
components, nested `Pipeline` values, or structured control values.  Lowering
preserves nested names as graph regions and keeps repeat/loop progress
resumable at runtime.

| Operation | Description |
|---|---|
| `pipeline["name"]` | Looks up a structural entry by `name` |
| `pipeline.replace(name, entry)` | Replaces a top-level structural entry |
| `pipeline.find(name, *, recursive=False)` | With `recursive=True`, searches nested pipelines and control bodies |

## Built-in Stages

The following table is generated from each operational Stage's production
`contract()` (`reads`, `writes`, and `exports`). It is refreshed with:
`python scripts/generate_stage_docs.py`.

<!-- BEGIN GENERATED STAGE CONTRACTS: do not edit -->

| Class | Name | Label | Notation | Reads | Writes | Exports |
|---|---|---|---|---|---|---|
| CountGenerationStage | count_generation | Count generation | $gen \leftarrow gen + 1$ | `runtime.generation`, `evaluations.pending` | `runtime.generation` | — |
| AskStage | ask | Generate offspring | $\mathcal{Q} \leftarrow \text{ask}(P, n)$ | `proposals.offspring`, `proposals.proposal_id_allocator`, `evaluations.plan`, `evaluations.plan_state`, `runtime.candidate_id_allocator` | `proposals.offspring`, `proposals.current`, `runtime.candidate_id_allocator`, `evaluations.evaluated_offspring` | — |
| SurrogatePredictStage | surrogate_predict | Surrogate prediction | $\hat{y} \leftarrow \text{predict}(\mathcal{Q}, \mathcal{A})$ | `proposals.offspring`, `populations.main`, `archives.main` | `proposals.offspring`, `surrogates.predictions` | — |
| PendingEvaluationContextStage | pending_evaluation_context | Pending evaluation context | $C \leftarrow \text{pending}(C)$ | — | — | — |
| AcquisitionStage | acquisition | Acquisition scoring | $\mathbf{s} \leftarrow \text{acquire}(\mathcal{Q}, \hat{y}, \mathcal{A})$ | `proposals.offspring`, `surrogates.predictions`, `archives.main`, `runtime.generation`, `runtime.decision_count`, `runtime.rng` | `evaluations.scores`, `evaluations.acquisition_result` | — |
| SurrogateFitStage | surrogate_fit | Fit surrogate | $\hat{f} \leftarrow \text{fit}(\mathcal{A})$ | `populations.main`, `archives.main` | — | — |
| TopKSelectionStage | top_k_selection | Top-k pre-selection | $\mathcal{Q} \leftarrow \text{top-}k(\mathcal{Q}, \mathbf{s})$ | `proposals.offspring`, `evaluations.scores` | `proposals.offspring` | — |
| SortByScoreStage | sort_by_score | Sort offspring by score | $\mathcal{Q} \leftarrow \text{sort\_desc}(\mathcal{Q},\,\mathbf{s})$ | `proposals.offspring`, `evaluations.scores` | `proposals.offspring`, `evaluations.scores` | — |
| EvaluationPlanStage | evaluation_plan | Plan evaluation | $R \leftarrow \text{plan}(Q)$ | `proposals.offspring`, `evaluations.pending`, `evaluations.request`, `evaluations.plan`, `evaluations.plan_state`, `evaluations.handles`, `evaluations.owners`, `evaluations.acquisition_result`, `evaluations.scores`, `surrogates.predictions`, `runtime.request_id_allocator`, `runtime.decision_count` | `evaluations.request`, `evaluations.plan`, `evaluations.plan_state`, `evaluations.plan_updates`, `evaluations.pending`, `evaluations.updates`, `evaluations.update_new_ids`, `evaluations.new_ids`, `evaluations.handles`, `evaluations.owners`, `runtime.request_id_allocator`, `runtime.decision_count` | — |
| AsyncEvaluationSubmitStage | async_evaluation_submit | Submit asynchronous evaluation |  | `proposals.offspring`, `evaluations.acquisition_result`, `evaluations.scores`, `evaluations.plan`, `evaluations.plan_state`, `evaluations.pending`, `evaluations.handles`, `runtime.request_id_allocator`, `runtime.decision_count` | `evaluations.request`, `evaluations.plan`, `evaluations.plan_state`, `evaluations.plan_updates`, `evaluations.pending`, `runtime.request_id_allocator`, `runtime.decision_count` | — |
| EvaluationSubmitStage | evaluation_submit | Submit evaluation | $H \leftarrow \text{submit}(R)$ | `evaluations.request`, `evaluations.plan`, `evaluations.plan_state`, `evaluations.pending`, `evaluations.handles`, `evaluations.owners` | `evaluations.pending`, `evaluations.handles`, `evaluations.plan_state` | — |
| EvaluationCollectStage | evaluation_collect | Collect evaluation | $U \leftarrow \text{collect}(H)$ | `evaluations.plan`, `evaluations.plan_state`, `evaluations.pending`, `evaluations.handles`, `evaluations.plan_updates`, `evaluations.request` | `evaluations.updates`, `evaluations.request`, `evaluations.plan`, `evaluations.plan_state`, `evaluations.plan_updates`, `evaluations.pending`, `evaluations.update_new_ids`, `evaluations.new_ids` | — |
| EvaluationApplyStage | evaluation_apply | Apply evaluation | $Q \leftarrow \text{apply}(U)$ | `proposals.offspring`, `evaluations.request`, `evaluations.updates`, `evaluations.plan`, `evaluations.plan_state`, `evaluations.plan_updates`, `evaluations.pending` | `proposals.offspring`, `evaluations.evaluated_offspring`, `evaluations.new_ids`, `evaluations.update_new_ids`, `evaluations.pending` | — |
| EvaluationAcknowledgeStage | evaluation_acknowledge | Acknowledge evaluation | $H \leftarrow \text{ack}(U)$ | `proposals.offspring`, `evaluations.request`, `evaluations.plan`, `evaluations.plan_state`, `evaluations.pending`, `evaluations.handles`, `evaluations.updates`, `evaluations.update_new_ids`, `evaluations.plan_updates`, `evaluations.count` | `evaluations.plan`, `evaluations.plan_state`, `evaluations.plan_updates`, `evaluations.pending`, `evaluations.handles`, `evaluations.count` | — |
| TrueEvaluationStage | true_evaluation | True objective evaluation | $\mathcal{Q}_{eval} \leftarrow \text{eval}(\mathcal{Q})$ | `proposals.offspring` | `proposals.offspring`, `evaluations.evaluated_offspring`, `evaluations.count` | — |
| ArchiveUpdateStage | archive_update | Archive update | $\mathcal{A} \leftarrow \mathcal{A} \cup \mathcal{Q}_{eval}$ | `evaluations.evaluated_offspring`, `archives.main`, `archives.pareto`, `evaluations.plan`, `evaluations.plan_state` | `archives.main`, `archives.pareto`, `evaluations.evaluated_offspring` | — |
| FeedbackStage | feedback | Apply feedback | $\mathcal{Q} \leftarrow \mathrm{feedback}(\mathcal{Q})$ | `proposals.offspring`, `evaluations.evaluated_offspring`, `evaluations.new_ids`, `surrogates.predictions`, `evaluations.plan`, `evaluations.plan_state` | `proposals.offspring`, `feedback.result` | — |
| TellStage | tell | Update population | $P \leftarrow \text{tell}(P, \mathcal{Q})$ | `proposals.offspring`, `proposals.current`, `feedback.result`, `evaluations.evaluated_offspring`, `evaluations.plan`, `evaluations.plan_state`, `evaluations.updates` | — | — |
| SurrogateOnlyLoopStage | surrogate_only_loop | Surrogate-only generations | $\text{for}\;i=1\dots gen\_ctrl$: $P \leftarrow \mathrm{tell}(P,\,\mathrm{acquire}(\mathrm{predict}(\mathrm{ask}(P))))$ | `archives.main` | — | — |
| InitializationStage | initialization | Initialize population | $\mathcal{A}_0,\,P_0 \leftarrow \mathrm{init}(n_{\mathrm{init}})$ | — | — | — |
<!-- END GENERATED STAGE CONTRACTS -->

`TellStage` declares `writes=()` because its state contract describes the
Stage boundary. When a legacy algorithm is used, the legacy adapter can still
mutate the algorithm's population during execution; that runtime behavior is
not added to the Stage contract.

The contract table includes all operational stages, including stages used by
asynchronous evaluation and nested surrogate-only loops.

`AskStage` calls `algorithm.ask()`, writes to `state.offspring`, and fires PostCrossover/PostMutation/PostAskEvent via `cbmanager`.

`SurrogatePredictStage` obtains prediction data; `AcquisitionStage` computes scores and preserves the complete `AcquisitionResult` for planning.

`SurrogateFitStage` is used to pre-fit the surrogate once, ahead of an inner loop where the archive doesn't change.
It is used before a surrogate prediction stage when an inner loop can reuse a fit.

`SurrogateOnlyLoopStage` remains available to stage-based compatibility
strategies.  `GenerationBasedStrategy` represents the same control flow with
the structured `Repeat` region instead.

```{note}
The `state` argument passed to `InitializationStage`'s `execute()` is ignored — it always builds a fresh state from initialization.
Used at the head of a user-defined pipeline when you want initialization itself to be treated as part of the pipeline.
```

See [strategies](../execution_and_evaluation/strategies.md) and the pipeline diagram in [Components overview](../index.md) for how the built-in strategies combine these stages and structured regions.
This page covers the contract of each Stage in isolation.

## Implementing a custom Stage

To add a custom Stage on the sequential compatibility path, subclass `Stage` and implement `execute()`.
Return a new State with `state.replace(...)`; a Stage should not manipulate Compiler `StatePatch` values directly.

```python
from saealib import Stage


class LogGenerationStage(Stage):
    """A custom stage that just prints the generation number to stdout."""

    name = "log_generation"
    label = "Log generation number"

    def execute(self, state):
        print(f"generation {state.gen}")
        return state
```

To pass an extra value through a compatibility Stage, store it in `OptimizationState.data`.
When adding a value, pass a new dictionary such as `state.replace(data={**state.data, "key": value})`.
For the same operation in a graph-native component, use a declared `StateKey` and `StatePatch`.

## `to_pseudocode`

`to_pseudocode(expand=False, indent=0)` is a mechanism that outputs each Stage's `notation` as pseudocode for a paper (LaTeX algorithmic notation).
`AskStage`/`TellStage`/`SurrogateOnlyLoopStage` have custom implementations that expand `Algorithm.ask_notation`/`tell_notation` when `expand=True`.

## Related components

- [Extension guidelines](../extension_guidelines.md): How to rearrange stages via `pipeline.replace`/`find`
- [strategies](../execution_and_evaluation/strategies.md): How the built-in Strategies combine Stages
- [OptimizationState](optimization_state.md): The state object `execute()` reads and writes
- [Components overview](../index.md): The diagram of the overall pipeline structure

## References

- {py:class}`saealib.Stage`
- {py:class}`saealib.Pipeline`
- {py:class}`saealib.CountGenerationStage`
- {py:class}`saealib.AskStage`
- {py:class}`saealib.SurrogatePredictStage`
- {py:class}`saealib.SurrogateFitStage`
- {py:class}`saealib.ArchiveUpdateStage`
- {py:class}`saealib.TellStage`
- {py:class}`saealib.SurrogateOnlyLoopStage`
- {py:class}`saealib.InitializationStage`
