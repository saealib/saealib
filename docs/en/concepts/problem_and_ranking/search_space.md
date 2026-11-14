---
primary_layer: layer4
related_layers: [layer3]
page_type: concept
---

# SearchSpace

`SearchSpace` defines candidate representation and the services that operate on that representation.
Algorithms and surrogates use the `GenomeBatch` and services provided by the space instead of changing its internal representation directly.

## SearchSpace's role

`SearchSpace` provides `representation`, `services`, `sample()`, and `validate()`.
`GenomeBatch` stores candidates row-wise; it does not directly own candidate IDs or objective values.
`validate()` returns per-row validity and batch-wide errors in a `ValidationResult`.

`ServiceRegistry` registers the services provided by a space by name.
The space that needs them exposes sampling, validation, cloning, equivalence, distance, genome serialization, and other capabilities as SearchSpace services.
`EvaluationAdapter` is an Adapter at the evaluation boundary that converts a `GenomeBatch` into evaluation request data.
`FeatureEncoder` is a `Protocol` defined in `saealib.space.services` and exposed as `saealib.space.FeatureEncoder`.
`FeatureEncoder` is registered and resolved as a service in `SearchSpace.services`, and determines the features the surrogate can learn.
`FeatureEncoder` is not an Adapter instance.
The public import is `from saealib.space import FeatureEncoder`.

## What SearchSpace holds

Built-in spaces are divided by the constraints on their representations:

- **`VectorSpace`**: Handles fixed-width dense numeric vectors.
- **`ObjectSpace`**: The minimal space for holding arbitrary object representations.
- **`SequenceSpace`**: Handles sequences whose length can vary.
- **`PermutationSpace`**: Handles representations that satisfy permutation constraints.

Each space satisfies the common `SearchSpace` contract, but its service set and `RepresentationSpec` differ.
For example, `ObjectSpace` does not register default services, so the configuration must provide the services required by the algorithm separately.

## Invariants

`RepresentationSpec` describes parameter types, shapes, and representation kinds.
This description says how candidates are stored; it is not itself an implementation of sampling or feature encoding.

Services are the executable boundaries that fill this gap.
When a prepared strategy consumes `FeatureEncoder`, execution preparation selects an encoder after checking for an explicit setting.
An encoder explicitly registered in `SearchSpace.services` always takes precedence over the default.
`VectorSpace` uses identity encoding and reports INFO with the selection source and reason; `PermutationSpace` and `SequenceSpace` use their built-in default encodings and report a non-blocking WARNING recommending explicit selection because encoding shapes what the surrogate learns.
This WARNING does not prevent execution.
`ObjectSpace` and custom spaces do not receive an inferred encoder, so they must provide one when the selected strategy consumes it.
Direct calls to `encode_features()` use the same defaults for built-in spaces and emit a Python `UserWarning` when they do; custom spaces without a supported default still require an explicit encoder.
Use `optimizer.contract_diagnostics()` to inspect the structured stream, or inspect `optimizer.last_contract_diagnostics` after `optimizer.validate()`. `Optimizer.validate()` returns ERROR findings only; INFO and WARNING selections remain in the diagnostic stream.

## Extending SearchSpace

`SearchSpace` owns candidate representation, candidate-generation services, and representation validity.
It does not own candidate objective values, observation provenance, component state, or Graph execution order.
The space owns the representation rules and services for `GenomeBatch` values, while `Population` owns the live genome storage.
`Population.genomes` exposes that storage as a read-only view, not an immutable snapshot; subsequent Population updates may change its contents.
It does not implicitly add candidate IDs or Feedback results.

When adding a new candidate representation, define its `RepresentationSpec`, sampling, validation, and required services within one `SearchSpace` boundary.
Do not move `RepresentationSpec` into `ComponentContract`; declare connections to the port's `DataSpec` and required services in the Compiler.

## Common failures

The `RepresentationSpec` and service set are fixed when the space is constructed, and the Compiler checks them against the component's `ServiceRequirement`.
The `GenomeBatch` returned by `sample()` conforms to the space's `RepresentationSpec`, and `validate()` reports mismatches in a `ValidationResult` instead of silently repairing them.
`SearchSpace` owns service implementation state, while components request and use registered services.
Mismatched kinds or shapes, missing required services, and invalid `GenomeBatch` values become Diagnostics from the Compiler or `ValidationResult`.

See [Framework extensions](../../framework/extensions.md) for extending candidate representations, [Declarative elements (Specs)](../../framework/specs.md) for port and `DataSpec` boundaries, and [Feedback](../observation_and_state/feedback.md) for observations associated with candidates.
See the [API reference](../../api/index.md) for public type import paths.

## Related components

- [Problem](problem.md): Combines the objective, constraints, optimization directions, and SearchSpace.
- [Population](../observation_and_state/population.md): Handles `GenomeBatch` values and individual evaluation results.
- [SurrogateManager](../surrogate_modeling/surrogate_manager.md): Requires the `FeatureEncoder` service.

## References

- {py:class}`saealib.space.SearchSpace`
- {py:class}`saealib.space.VectorSpace`
- {py:class}`saealib.space.ObjectSpace`
- {py:class}`saealib.space.SequenceSpace`
- {py:class}`saealib.space.PermutationSpace`
- {py:class}`saealib.space.FeatureEncoder`
