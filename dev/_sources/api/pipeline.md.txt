---
primary_layer: layer2
related_layers: [layer3, layer4]
page_type: reference
---

# Pipeline

`Pipeline` is a DSL for describing components and structured control regions.
It does not execute `OptimizationState` directly; Optimizer uses it as input when building an execution plan.

```python
from saealib import Branch, Loop, Pipeline, Repeat

pipeline = Pipeline(
    name="generation",
    steps=[
        Repeat(
            Pipeline(steps=[ask, acquire, tell], name="surrogate_generation"),
            count=10,
            name="surrogate_generations",
        ),
        Loop(evaluation, until=budget_reached, name="evaluation_loop"),
        Branch(route, then=fast_path, else_=safe_path, name="route"),
    ],
)
```

This example only shows how to build a Pipeline.
In ordinary use, `Optimizer` compiles the Graph and Pipeline produced by a Strategy, so users do not need to call the Compiler directly.

Nested Pipelines and control values are kept as named structured regions.
`Repeat` represents a fixed number of repetitions, while `Loop` and `Branch` evaluate conditions through declared state contracts.
Because regions are not reduced to ordinary graph cycles, the Runtime can preserve state effects and resume frames.

Components placed in a structured Pipeline provide the graph-native `contract()` and `execute(StateView)` boundaries.
The legacy `Stage` execution surface remains as a connection point for the compatibility Graph Builder.
When placing an existing Stage in a structured Pipeline, wrap it explicitly with `stage_component(stage)`.

During compilation, required input ports are matched to compatible outputs upstream in control order.
A uniquely matched connection becomes a `DataEdge`; no match or multiple candidates produces a Compiler diagnostic.
Use an explicit Graph when an ambiguous connection must be resolved.

`stage_component(stage)` is a migration adapter.
State writes through the transaction proxy become `StatePatch` values, but the adapter retains ownership of mutable objects exposed as Services or Context capabilities.
In a new graph-native component, keep persistent mutable state behind declared `StateKey` values and patches.

```{eval-rst}
.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   saealib.Stage
   saealib.Pipeline
   saealib.Repeat
   saealib.Loop
   saealib.Branch
   saealib.pipeline.ComponentRef
```
