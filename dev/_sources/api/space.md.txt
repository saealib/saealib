---
primary_layer: layer4
related_layers: [layer3]
page_type: reference
---

# SearchSpace API

The SearchSpace API is the public facade for candidate representations and space-specific services.
Use names from `saealib.space` rather than the space implementation modules.

## Spaces, services, and adapters

### Search spaces

Search spaces provide `GenomeBatch` representations.

```{eval-rst}
.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   saealib.space.SearchSpace
   saealib.space.ServiceRegistry
   saealib.space.ValidationResult
   saealib.core.contracts.RepresentationSpec
   saealib.space.VectorSpace
   saealib.space.ObjectSpace
   saealib.space.SequenceSpace
   saealib.space.PermutationSpace
```

### SearchSpace services

SearchSpace services such as sampling and validation are capabilities provided by a space.

```{eval-rst}
.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   saealib.space.GenomeCodec
   saealib.space.SamplingService
   saealib.space.ValidationService
```

### Adapters

`EvaluationAdapter` is an adapter at the evaluation boundary that converts a `GenomeBatch` into an `EvaluationPayload`.
`FeatureEncoder` is an adapter subtype exposed as `saealib.space.FeatureEncoder`. It converts a `GenomeBatch` into a `FeatureBatch` and determines which features a surrogate can learn. It differs in kind from space capabilities such as `SamplingService`.
When a prepared strategy consumes `FeatureEncoder`, execution preparation selects a default if `SearchSpace.services` has no explicit encoder. `VectorSpace` uses identity encoding and reports INFO with its source and reason; `PermutationSpace` and `SequenceSpace` keep their built-in encodings and report a non-blocking WARNING that encoding affects surrogate learning and an explicit choice is recommended.
An explicitly provided encoder always takes priority and does not produce a default-selection diagnostic. `ObjectSpace` and custom spaces receive no inferred encoder; provide one when the selected strategy consumes it.
Direct calls to `encode_features(space, genomes)` use the same built-in defaults when no encoder is registered and emit a Python `UserWarning`. An explicit encoder is used without a warning, and a custom space without a supported default still raises `ValidationError`; the helper does not register its fallback in the space.
Inspect choices with `optimizer.contract_diagnostics()`, or use `optimizer.last_contract_diagnostics` after `optimizer.validate()`. `Optimizer.validate()` returns ERROR findings only, while the structured stream also includes INFO and WARNING.

```{eval-rst}
.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   saealib.space.FeatureEncoder
```
