---
primary_layer: layer3
related_layers: [layer2, layer4]
page_type: guide
---

# Add a custom component to an existing contract

This page assumes that you are implementing a replacement that fits an existing component contract.
:::{admonition} What you'll be able to do
:class: tip

By the end of this page, you'll be able to choose an extension point, verify its minimal contract, register the component, and test it.
:::

See [Extension guidelines](../concepts/extension_guidelines.md) for the extension-point categories. If you need to change a contract or the execution framework itself, continue to [Framework extensions](../framework/extensions.md).

The import policy is collected in [Canonical Imports](../api/imports.md).

## Choose an extension point

If changing a built-in component's settings is enough, use [Swapping built-in components](component_swap.md) or [Hooks](interface_hooks.md).
When you need new search, prediction, evaluation, or generation logic, choose the component with the corresponding contract.

| Change | Extension point | Minimal implementation boundary |
|---|---|---|
| Generate candidates and consume evaluation results | `Algorithm` | `ask()` and `tell()`, plus the required attributes and contract |
| Prediction model | `Surrogate` | `fit()` and `predict()` |
| Turn predictions into candidate scores | `AcquisitionFunction` | `evaluate()`, or the corresponding Pointwise `compute_reference()` and `score()` |
| Extend the evaluation backend on the graph-native path | `EvaluationAdapter` / `Evaluator` | `EvaluationRequestBatch` kind on `EvaluationPlanner`'s output port → candidate selection → `GenomeBatch` → `Evaluator` (`Problem.evaluation_adapter` set: `EvaluationAdapter.transform()` return value; unset with a `DenseVectorBatch`: `DenseVectorBatch.array` payload) → `Problem.evaluate_batch()` → `ObservationBatch` |
| Swap a synchronous Evaluator on the Stage compatibility path | `Evaluator` | `evaluate_batch()` and `EvaluationResult` |
| Generation execution policy | `Strategy` | `TemplateStrategy` with a declared `Pipeline` |
| Compatibility execution unit | `Stage` | `execute(state) -> state` |

Check the exact required methods for each contract in the [Algorithm reference](../api/algorithms.md), [Surrogate reference](../api/surrogate.md), [Acquisition reference](../api/acquisition.md), and [Stage reference](../api/stages.md).
Similar names do not imply identical contracts, so a method from another component cannot be reused without checking the boundary.

## Check the minimal contract

Before implementing the component, inspect the abstract methods on the existing abstract base class, `contract()`, and the attributes read at runtime.
`Algorithm.ask()` and `Algorithm.tell()` read a `StateView` and return a `ProposalBatch` or `StatePatch`.
`Surrogate` receives arrays and returns a `SurrogatePrediction`.
`Stage` receives an `OptimizationState` and returns the updated state.

## Extend a component on the graph-native path

Extend graph-native components such as `Algorithm` at the `ask(...) -> ProposalBatch` and `tell(FeedbackBatch, StateView) -> StatePatch` boundaries.
To add evaluation, declare `EvaluationRequestBatch` as the data kind on the `EvaluationPlanner` output port and pass the `GenomeBatch` selected by candidate selection to the Evaluator. When `Problem.evaluation_adapter` is set, pass the return value of `EvaluationAdapter.transform()` as the payload to `Problem.evaluate_batch()`; when it is unset and the input is a `DenseVectorBatch`, pass `DenseVectorBatch.array` as the payload.

## Extend a synchronous Evaluator on the Stage compatibility path

This example replaces only the evaluation step with an `Evaluator` on the Stage compatibility path.
The implementation boundary is `evaluate_batch(x, problem) -> EvaluationResult`; `EvaluationResult` accepts one- or two-dimensional `f`, `g`, and `cv` arrays.

```python
import numpy as np
from saealib import EvaluationResult, Evaluator


class BatchSphereEvaluator(Evaluator):
    def evaluate_batch(self, x, problem):
        x = np.atleast_2d(np.asarray(x, dtype=float))
        values = np.sum(x**2, axis=1)
        return EvaluationResult(
            f=values[:, None],
            g=np.empty((len(x), 0)),
            cv=np.zeros(len(x)),
        )
```

When you pass a custom Evaluator to `Optimizer.set_evaluator()`, the built-in synchronous evaluation adapter calls it.
`EvaluationResult` validates the number of rows and the array shapes.

## Register or compose the component

For direct use, pass an instance to the public `Optimizer.set_evaluator(evaluator)` method.
Add `@register()` and register the class with the Registry only when a configuration file or preset constructs it by name.

```python
import numpy as np
from saealib import Optimizer, Problem, register


@register()
class NamedBatchSphereEvaluator(BatchSphereEvaluator):
    pass


problem = Problem(
    lambda x: np.sum(x**2),
    dim=3,
    n_obj=1,
    direction=np.array([-1.0]),
    lb=[-5.0] * 3,
    ub=[5.0] * 3,
)
optimizer = Optimizer(problem).set_evaluator(NamedBatchSphereEvaluator())
```

Because the `Problem` objective is not always replaced by this Evaluator, validate a configuration that actually calls `set_evaluator()`.

## Add a minimal test

Start with a small unit test that fixes the shapes and values returned by `evaluate_batch()`.
Then run `Optimizer(problem).set_evaluator(...)` with a short `max_fe` and check contract validation and the execution path.
Listing abstract base classes and changing the execution framework are outside this guide; see the relevant API reference or framework-extension documentation.

(port-operators-to-native-saealib-code)=
## Port external operators to native saealib components

Wrapping an external operator with a [pymoo adapter](external_libraries.md) keeps a runtime dependency on that library.
The pymoo adapters already preserve population-level vectorization in `GA`'s default batch mode, calling the wrapped operator once for the whole batch or gated subset, so porting is not normally needed to remove per-item calling overhead.
Port an operator when you want to remove the runtime dependency, or when its logic should be directly auditable and citable as native saealib code for a software paper.
The core logic usually stays the same; the main changes are adapting the array shape where necessary and using saealib's RNG.

The one detail that is easy to get wrong when porting is *where* the individual-level probability gate lives.
saealib's `GA` checks `Crossover.prob` itself and passes only gated parent groups to `crossover_batch()`, but `Mutation.mutate_batch()` must draw one `self.prob` gate per row and leave ungated rows unchanged.
Both pymoo's built-in mutations and DEAP's `toolbox.mutate()` (gated externally by `algorithms.varAnd`'s `mutpb`) leave this gate outside the operator, so a ported mutation usually needs a gate array that the original code did not have.

### Port from pymoo

A custom pymoo `Mutation` typically vectorizes over the whole batch `X` (shape `(n_individuals, dim)`):

```python
import numpy as np
from pymoo.core.mutation import Mutation as PymooMutationBase


class MyPymooMutation(PymooMutationBase):
    def __init__(self, sigma=1.0):
        super().__init__()
        self.sigma = sigma

    def _do(self, problem, X, random_state=None, **kwargs):
        rng = random_state if random_state is not None else np.random.default_rng()
        Xp = X.copy()
        mask = rng.random(Xp.shape) < 0.5
        Xp[mask] += rng.normal(0, self.sigma, size=Xp.shape)[mask]
        return Xp
```

Ported to a native saealib `Mutation`, the batch axis stays in place.
The `_do()` body maps almost directly to `mutate_batch()`; the main addition is saealib's per-row `prob` gate:

```python
import numpy as np
from saealib.operators import Mutation


class MyMutation(Mutation):
    def __init__(self, prob=1.0, *, sigma=1.0, prob_var=0.5):
        super().__init__()
        self.prob = prob
        self.sigma = sigma
        self.prob_var = prob_var

    def mutate_batch(self, candidates_batch, mutate_range, rng=np.random.default_rng()):
        candidates_batch = np.asarray(candidates_batch, dtype=float)
        n, dim = candidates_batch.shape
        gate = rng.random(n) < self.prob
        result = candidates_batch.copy()
        if not np.any(gate):
            return result

        selected = result[gate]
        mask = rng.random((len(selected), dim)) < self.prob_var
        noise = rng.normal(0, self.sigma, size=selected.shape)
        selected[mask] += noise[mask]
        result[gate] = selected
        return result
```

### Port from DEAP

DEAP operators work on one individual or parent pair at a time, so a native saealib port must add and vectorize over a leading batch axis.
A custom crossover:

```python
import random


def my_cx(ind1, ind2, swap_rate=0.5):
    for i in range(len(ind1)):
        if random.random() < swap_rate:
            ind1[i], ind2[i] = ind2[i], ind1[i]
    return ind1, ind2
```

becomes a native `Crossover` by vectorizing over both the parent pairs and dimensions.
It also replaces `random.random()` with the `rng` saealib passes in, dropping DEAP's implicit global RNG state in favor of saealib's per-run `np.random.Generator` (kept reproducible via `minimize(..., seed=...)`):

```python
import numpy as np
from saealib.operators import Crossover


class MyCrossover(Crossover):
    def __init__(self, prob, swap_rate=0.5):
        super().__init__()
        self.prob = prob
        self.swap_rate = swap_rate

    def crossover_batch(self, parents_batch, bounds=None, rng=np.random.default_rng()):
        n_pair, _, dim = parents_batch.shape
        p1, p2 = parents_batch[:, 0, :], parents_batch[:, 1, :]
        mask = rng.random((n_pair, dim)) < self.swap_rate
        c1 = np.where(mask, p2, p1)
        c2 = np.where(mask, p1, p2)
        return np.stack((c1, c2), axis=1)
```

`toolbox.mate` is gated externally by `varAnd`'s `cxpb`, matching saealib's `GA`, so — unlike mutation — no extra gate is needed here.

### Port an Algorithm and Problem

Porting a whole search algorithm (survival selection, archive management, index-coupled state such as DE) is not a mechanical rewrite in the same way: it means reimplementing `Algorithm.ask()`/`tell()` from scratch. For pymoo specifically, the engine-mode `PymooAlgorithm` adapter described in [Integrating External Libraries](external_libraries.md) covers this case without a rewrite; for other libraries, see [Algorithm](../concepts/search_algorithms/algorithm.md) for what a native `Algorithm` subclass needs to implement.

## Separate component responsibilities

When extending an existing execution contract, define the component's responsibility and state boundary first.

| Component | Main responsibility | State boundary |
|---|---|---|
| `Algorithm` | Propose candidates and consume feedback | `ProposalBatch`, `FeedbackBatch`, `StatePatch` |
| Operator | Crossover, mutation, parent selection, and survivor selection | Candidates and RNG passed by the Algorithm |
| `Surrogate` | Fit and predict with the prediction model | Training data from the Archive and predictions |
| `OptimizationStrategy` | Evaluation plan, candidate selection, and feedback flow | `TemplateStrategy` with a declared `Pipeline` |
| `Stage` | One operation on the existing sequential path | `OptimizationState` and `replace()` |

Check the contract and built-in implementation on the relevant concept page before implementing each component.
`Algorithm.ask()` and `Algorithm.tell()`, the strategy's declared `Pipeline`, and `Stage.execute()` are different boundaries.

## Plan implementation and Stage migration

1. Check the data formats returned by `Problem` and `SearchSpace`.
2. Obtain dependent components from the public namespaces.
3. Implement the component's contract and state access in isolation.
4. Assemble it with the `Optimizer` replacement API.
5. Check feedback candidate IDs, evaluation provenance, and checkpoint behavior.

When connecting an existing Stage to a structured pipeline, make `stage_component(stage)` explicit.
For a new graph-native component, do not reuse the Stage's `OptimizationState` boundary; implement `ComponentContract`, `StateView`, and `StatePatch`.

For details, see [Algorithm](../concepts/search_algorithms/algorithm.md), [OptimizationStrategy](../concepts/execution_and_evaluation/strategies.md), and [Stage](../concepts/observation_and_state/stage.md).

## Related concepts and reference

- [Evaluator concept](../concepts/execution_and_evaluation/evaluation.md)
- [Extension guidelines](../concepts/extension_guidelines.md)
- [Evaluation reference](../api/evaluation.md)
- [Core reference](../api/core.md)
- {py:class}`saealib.Crossover`
- {py:class}`saealib.Mutation`
