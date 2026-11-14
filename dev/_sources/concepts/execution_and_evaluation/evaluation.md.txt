---
primary_layer: layer2
related_layers: [layer3, layer4]
page_type: concept
---

# Evaluator

`OptimizationStrategy` creates an `EvaluationPlan` and delegates evaluation to the Evaluator.

## Evaluator's role

The two synchronous and asynchronous evaluation boundaries receive and return the following values.

| Boundary | Receives | Returns | Main users |
|---|---|---|---|
| graph-native boundary | `GenomeBatch` selected by the evaluation planner's `EvaluationRequestBatch`, candidate IDs, proposal relations, and metadata | `ObservationBatch`, including partial, out-of-order, and repeated observations | Evaluator, asynchronous Scheduler, and structured runtime |
| Stage compatibility boundary | `x`, `Problem`, and `EvaluationRequest` | `EvaluationResult` (`f`, `g`, `cv`, and optional `candidate_ids`) | Compatibility Stages and sequential compatibility runtime |

Requests passed to the Evaluator or asynchronous Scheduler identify candidate IDs and the items to evaluate.
The Evaluator and Runtime provider can switch between synchronous and asynchronous evaluation without changing the Pipeline's candidate-generation contract.
`EvaluationRequestBatch` is the `DataSpec` kind declared on the evaluation planner's output port.
The Evaluator receives the selected `GenomeBatch`. When `Problem.evaluation_adapter` is set, it passes the return value of `EvaluationAdapter.transform()` to `Problem.evaluate_batch()`; when it is unset and the input is a `DenseVectorBatch`, `SerialEvaluator` uses `DenseVectorBatch.array` as the payload.
The Evaluator returns an `ObservationBatch` and matches observations by candidate ID, proposal relation, sequence, status, source, and completion semantics rather than by row position.

On the Stage compatibility path, use `Evaluator.evaluate_batch(...) -> EvaluationResult`.
`EvaluationRequest` holds a Request ID, candidate IDs, a GenomeBatch, and metadata.
Only at the compatibility boundary, `EvaluationResult` must satisfy the shape and row-count rules for `f`, `g`, and `cv`; when `candidate_ids` is provided, it must be unique and match the result row count.
When `Evaluator.submit` returns `candidate_ids`, they must exactly match the request's candidate IDs.
This does not require an external `ObservationBatch` to use row order.

`EvaluationResult` is a dataclass holding evaluation-result arrays and candidate IDs.

- **`f`**: Objective values, with shape `(n, n_obj)`.
- **`g`**: Raw constraint values, with shape `(n, n_constraints)`, or `(n, 0)` when there are no constraints.
- **`cv`**: Aggregated constraint violation for each candidate, with shape `(n,)`, or all `0` when there are no constraints.
- **`candidate_ids`**: The candidate ID corresponding to each result row.

## Built-in Evaluators

| Class | Parameters | Characteristics |
|---|---|---|
| `SerialEvaluator` | None | Evaluates candidates one at a time, sequentially. The default |
| `JoblibEvaluator` | `n_jobs=-1, backend="loky", **joblib_kwargs` | Evaluates candidates in parallel via `joblib.Parallel` |
| `ThreadPoolEvaluator` | Executor options | Uses a thread pool for synchronous batch evaluation |
| `AsyncEvaluator` | Adapter and scheduler options | Submits requests and exposes nonblocking lifecycle operations |

Using `JoblibEvaluator` requires the `parallel` extra (`pip install saealib[parallel]`); if not installed, it raises `ImportError` at construction time.
Besides `backend`'s default `"loky"` (a process pool serializing with cloudpickle), you can switch to third-party backends like `"dask"`/`"ray"` with a single parameter change (the corresponding package and cluster are required separately).
In configurations using multiple `JoblibEvaluator`s at once, such as an island model, CPU cores can end up over-reserved.
Either limit each island's `n_jobs` to `1` and control overall concurrency via the parallelism across islands, or use `joblib.parallel_backend` as a context manager to limit the number of inner workers.
For independent completion times, combine `AsyncEvaluator`, `SteadyStateStrategy`, `AsyncEvaluationScheduler`, and `Optimizer.set_async_evaluation_scheduler()`.
Each candidate is managed as a pending Request, while the Scheduler manages capacity, budget, and Request state.
`collect(wait=False)` reports progress without waiting for completion, and the Runtime orders updates to the Population and Archive.

Swap the synchronous Evaluator with `Optimizer.set_evaluator(evaluator)`.

`EvaluationRequest.metadata` is consumed by evaluators that expose explicit
execution planners. `RepeatedEvaluation` returns one `EvaluationPlan` request
per replicate with stable candidate IDs. The scheduler owns capacity, budget,
and request lifecycle.

## Implementing a custom Evaluator

The following example is a Stage-compatible synchronous Evaluator that implements `evaluate_batch()`.
For non-vector Genomes, define the payload passed to the evaluation function with `EvaluationAdapter`.

```python
import numpy as np
from saealib import EvaluationResult, Evaluator


class ReversedOrderEvaluator(Evaluator):
    """An Evaluator that evaluates candidates in reverse order, from the end."""

    def evaluate_batch(self, x, problem):
        x = np.atleast_2d(np.asarray(x, dtype=float))
        n = len(x)
        f = np.empty((n, problem.n_obj), dtype=float)
        g = np.empty((n, problem.n_constraints), dtype=float)
        cv = np.zeros(n, dtype=float)
        for i in reversed(range(n)):
            g_i, cv_i = problem.evaluate_constraints(x[i])
            f[i] = problem.evaluate(x[i], g_i)
            g[i] = g_i
            cv[i] = cv_i
        return EvaluationResult(f=f, g=g, cv=cv)
```

Keep the order that calls `problem.evaluate_constraints(xi)` before `problem.evaluate(xi, g_i)`.
[ConstraintHandler](../problem_and_ranking/constraints.md) uses constraint values to correct the objective, so reversing this order applies the correction incorrectly.

## Related components

- [Problem](../problem_and_ranking/problem.md): Where the objective and constraint functions evaluated by `evaluate_batch` are defined
- [ConstraintHandler](../problem_and_ranking/constraints.md): Aggregates constraint violation and corrects the objective value inside `problem.evaluate`
- [Initializer](initialization.md): Uses `Evaluator` to evaluate the initial population
- [strategies](strategies.md): Uses `Evaluator` to evaluate candidates each generation
- [Feedback](../observation_and_state/feedback.md): The Feedback structure that passes evaluation results to the Algorithm

## References

- {py:class}`saealib.Evaluator`
- {py:class}`saealib.SerialEvaluator`
- {py:class}`saealib.JoblibEvaluator`
- {py:class}`saealib.AsyncEvaluator`
- {py:class}`saealib.AsyncEvaluationScheduler`
- {py:class}`saealib.EvaluationResult`
