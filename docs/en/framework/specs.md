---
primary_layer: layer4
related_layers: [layer3]
page_type: concept
---

# Specs

Contract and SearchSpace boundaries pass connection conditions and representations as declarative values rather than processing them directly.
This page calls these declarative values specs.
Specs are immutable values, not Component subclasses, that the Compiler uses to determine connections and requirements.

## Specs' role

Specs own the boundary that passes connection conditions, representations, and required services to the Compiler rather than performing the processing itself.
They do not hold Component execution state; `ComponentContract` and `SearchSpace` own the declarative values.

## PortSpec and PortContract

`PortSpec` is one port declaration containing a port name, direction, `DataSpec`, cardinality, required services, and optionality.
`PortContract` is the set of input and output ports belonging to one role.
Port names must be unique within each direction.

The Compiler uses `ONE`, `MANY`, and `OPTIONAL` cardinalities to determine whether a provided value satisfies the consumer's requirement.
A connection is incompatible when its direction, registered data kind, schema binding, or cardinality does not match.

## DataSpec and ServiceRequirement

`DataSpec` represents a registered nominal data kind and schema binding.
Fixed values, variables, containment conditions, and product bindings can unify the same schema variable across ports.
DataSpec kind compatibility, schema unification, and service resolution are separate checks.

`ServiceRequirement` declares a named service required by a port or Component.
SearchSpace services such as `SamplingService` and `ValidationService` are matched as space capabilities.
`EvaluationAdapter` is an adapter at the evaluation boundary.
`FeatureEncoder` is a `Protocol` defined in `saealib.space.services` and exposed as `saealib.space.FeatureEncoder`.
`FeatureEncoder` is registered and resolved as a service in `SearchSpace.services`.
`FeatureEncoder` converts a `GenomeBatch` to a `FeatureBatch` and determines the features a surrogate model can learn.
`FeatureEncoder` is not an Adapter instance.
During execution preparation, a strategy that consumes `FeatureEncoder` receives a visible default when its space has no explicit encoder. `VectorSpace` uses identity encoding and reports the choice, source, and reason at INFO; `PermutationSpace` and `SequenceSpace` retain their built-in encodings and report a non-blocking WARNING recommending an explicit choice because encoding affects what the surrogate learns.
An encoder explicitly provided through `SearchSpace.services` takes priority over these defaults and is used without a default-selection diagnostic. `ObjectSpace` and custom spaces receive no inferred encoder; a strategy that consumes one must have the space provide it.
Inspect the structured stream with `optimizer.contract_diagnostics()`, or read `optimizer.last_contract_diagnostics` after `optimizer.validate()`. `Optimizer.validate()` returns ERROR findings only, so the structured stream is where INFO and WARNING selections are visible.

## RepresentationSpec boundary

`RepresentationSpec` is the candidate-representation specification on the SearchSpace side.
It describes candidate type, shape, and representation, but is not part of a ComponentContract's ports, state, or execution.
`FeatureEncoder` is registered and resolved as a SearchSpace service, and `RepresentationSpec` is the candidate-representation specification held by the space.

| Declarative value | Owner | Boundary using it |
|---|---|---|
| `PortSpec`, `PortContract` | `ComponentContract` | Graph connections and port compatibility |
| `DataSpec` | Port or contract | Data kind and schema binding unification |
| `ServiceRequirement` | Component or port | Resolving services provided by SearchSpace and Problem |
| `RepresentationSpec` | `SearchSpace` | Candidate representation and representation-service consistency |

## Construction time and use time

`PortSpec`, `DataSpec`, and `ServiceRequirement` are created when a ComponentContract is built and used by the Compiler when connecting the graph.
The contract or SearchSpace owns these values; the Runtime does not rewrite them during execution.

## Invariants and diagnostics

An unregistered kind, undefined cardinality, mismatched schema binding, or missing required service becomes a Diagnostic.
The implementation reports these with codes such as `unknown_data_spec`, `unknown_cardinality`, `unknown_schema_variable`, and `unresolved_service`.

## Extension points

When adding a data kind or service, state its registration and compatibility rules explicitly and do not connect it through an implicit type conversion.

## Related pages

See the [API reference](../api/index.md) for concrete port-declaration types and public import paths.
See [SearchSpace](../concepts/problem_and_ranking/search_space.md) for extending candidate representations and [Framework extensions](extensions.md) for extending Compiler rules.

## References

- {py:class}`saealib.core.PortSpec`
- {py:class}`saealib.core.PortContract`
- {py:class}`saealib.core.DataSpec`
