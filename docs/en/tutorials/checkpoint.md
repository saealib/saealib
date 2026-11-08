---
primary_layer: layer2
page_type: guide
---

# Reproducibility and Checkpointing

Makes long-running optimizations reproducible and lets you resume them partway through.

Checkpointing uses the low-level `Optimizer` API.
Construct `Optimizer(problem)`, configure components by chaining `set_*()`, and run it with `run()`.
Resume from a saved state with `run_from()`.

:::{admonition} What you'll be able to do
:class: tip

By the end of this page, you'll be able to ensure reproducibility with a random seed, save and load checkpoints, and resume an optimization.
:::

For a single run with the high-level API, use the [high-level API](highlevel_api.md); for finer control of run state, see the [low-level API](lowlevel_api.md).
This page shows how to save a reproducible low-level run and resume it partway through.

## Reproduce runs with a random seed

Passing the same `seed` to `Optimizer(problem, seed=...)` initializes all random-number-using processes in the same sequence, producing identical results.

```python
import numpy as np
from saealib import (
    Problem,
    Optimizer,
    GA,
    CrossoverBLXAlpha,
    MutationUniform,
    SequentialSelection,
    TruncationSelection,
    RBFSurrogate,
    GaussianKernel,
    LocalSurrogateManager,
    MeanPrediction,
    IndividualBasedStrategy,
    LHSInitializer,
    Termination,
    max_fe,
)


def expensive_func(x):
    return np.sum(x**2)


DIM = 10
SEED = 0

problem = Problem(
    func=expensive_func,
    dim=DIM,
    n_obj=1,
    direction=np.array([-1.0]),
    lb=[-5.0] * DIM,
    ub=[5.0] * DIM,
)


def build_optimizer(max_fe_value):
    return (
        Optimizer(problem, seed=SEED)
        .set_initializer(
            LHSInitializer(n_init_archive=5 * DIM, n_init_population=4 * DIM, seed=SEED)
        )
        .set_algorithm(
            GA(
                crossover=CrossoverBLXAlpha(0.7, 0.4),
                mutation=MutationUniform(0.3),
                parent_selection=SequentialSelection(),
                survivor_selection=TruncationSelection(),
            )
        )
        .set_surrogate_manager(
            LocalSurrogateManager(RBFSurrogate(kernel=GaussianKernel()))
        )
        .set_acquisition(MeanPrediction())
        .set_strategy(IndividualBasedStrategy(evaluation_ratio=0.1))
        .set_termination(Termination(max_fe(max_fe_value)))
    )


ctx1 = build_optimizer(300).run()
ctx2 = build_optimizer(300).run()

print(np.allclose(ctx1.archive.get_array("f"), ctx2.archive.get_array("f")))  # True
```

`build_optimizer` is used in the following sections too, to rebuild an `Optimizer` with the same component configuration.

To preserve reproducibility, recreate the SurrogateManager and AcquisitionFunction with the same configuration.
The current API does not support passing an AcquisitionFunction as an argument to `LocalSurrogateManager`.

## Save and resume a checkpoint

The `ctx` returned by `run()` can be saved to a single npz file with `ctx.save(path)`.

```python
ctx = build_optimizer(200).run()
ctx.save("checkpoint.npz")
```

Save at a step boundary: a state yielded by `iterate()`, a `GenerationEndEvent` or `RunEndEvent` callback (including `CheckpointCallback`), or after `run()` returns. A run may end inside an open region; saving after it returns is supported when the state can be resumed. With asynchronous evaluation, a `GenerationBasedStrategy` run can reach its termination condition while true evaluations are still in flight inside a generation. The state returned by `run()` in that situation may not resume to the same result. Save at `GenerationEndEvent`, or save from a `RunEndEvent` callback instead. The runtime also accepts a suspended point while waiting for asynchronous evaluations when it can continue from that point. Saving from another callback or while a step is executing inside a `Repeat` or other region raises `CheckpointError`. A `RunEndEvent` callback can save when no planned requests are still waiting to be submitted; a plan with requests still waiting to be submitted is rejected. In that case, save at `GenerationEndEvent`, or save the state returned by `run()` if the run ended at a generation boundary. Pending asynchronous requests are supported only when every request can be reattached and the feedback accumulator matches those requests. A request whose callback or feedback update has started, or a pending component replacement made with `set_*()` or attribute assignment, also rejects the save. Pickle saving reports the evaluator type when that evaluator cannot be serialized; portable checkpoints do not contain the evaluator.

A saved checkpoint can be loaded with `OptimizationState.load(path, problem)`, and passing it to `Optimizer.run_from(ctx)` resumes from where it left off.

```python
from saealib import OptimizationState

loaded_ctx = OptimizationState.load("checkpoint.npz", problem)

resumed_ctx = build_optimizer(600).run_from(loaded_ctx)
print(resumed_ctx.fe)  # includes the evaluations from before saving
print(resumed_ctx.data["resumed"])  # True
```

`ctx.data["resumed"]` is a flag set to `True` only on a context resumed via `run_from()`.

From a callback such as `RunStartEvent`, it can be accessed as `event.ctx.data["resumed"]`.

## Save checkpoints automatically

Passing `checkpoint_path` to `run()`/`iterate()` saves automatically every `checkpoint_interval` generations.

`checkpoint_path` is treated as a directory rather than a single file, and per-generation snapshots are created, named `checkpoint_{gen:06d}.npz`.

```python
ctx = build_optimizer(300).run(checkpoint_path="checkpoints", checkpoint_interval=5)
```

To resume, load the most recent snapshot in the directory.

```python
from pathlib import Path

latest = sorted(Path("checkpoints").glob("checkpoint_*.npz"))[-1]
loaded_ctx = OptimizationState.load(latest, problem)
```

If you don't want to leave snapshots behind after a successful run, specify `checkpoint_delete_on_success=True` (the directory itself is kept; only the files inside it are deleted).

```python
ctx = build_optimizer(300).run(
    checkpoint_path="checkpoints",
    checkpoint_interval=5,
    checkpoint_delete_on_success=True,
)
```

## Save in pickle format

npz saves only `ctx`, but pickle format can save the entire `Optimizer`, including the fitted surrogate parameters.

```python
optimizer = build_optimizer(200)
ctx = optimizer.run()
optimizer.save_pickle(ctx, "checkpoint.pkl")

loaded_optimizer, loaded_ctx = Optimizer.load_pickle("checkpoint.pkl")
```

A `UserWarning` about the Python or library version may appear at runtime.

An `Optimizer` containing objects that the standard `pickle` cannot serialize — such as a lambda used in `Termination` — cannot be pickle-saved.

## Use CheckpointCallback directly

The `checkpoint_path` argument of `run()` simply registers a `CheckpointCallback` internally.

To wire up the same behavior explicitly, register a `CheckpointCallback` on `cbmanager`.

```python
from saealib import CheckpointCallback

optimizer = build_optimizer(300)
callback = CheckpointCallback("checkpoints", interval=5, optimizer=optimizer)
callback.register(optimizer.cbmanager)

ctx = optimizer.run()
```

The `optimizer` argument is required when specifying `format="pickle"` or `format="both"`.

## Related concepts and reference

- {py:class}`saealib.Optimizer`
- {py:class}`saealib.CheckpointCallback`
- {py:func}`saealib.minimize`
