---
primary_layer: layer2
related_layers: [layer3]
page_type: reference
---

# Optimization Strategies

Built-in Strategies compose candidate generation, evaluation planning, Feedback, and Population updates.
The canonical extension point for a new Strategy is `TemplateStrategy` with a declared `Pipeline`; overriding `build_graph(provider)` is deprecated and emits a `DeprecationWarning`.

## Base

```{eval-rst}
.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   saealib.OptimizationStrategy
   saealib.strategies.TemplateStrategy
```

## Implementations

```{eval-rst}
.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   saealib.GenerationBasedStrategy
   saealib.IndividualBasedStrategy
   saealib.PreSelectionStrategy
   saealib.DirectStrategy
   saealib.SteadyStateStrategy
```

## Island execution

```{eval-rst}
.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   saealib.IslandModel
```
