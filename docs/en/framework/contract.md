---
primary_layer: layer4
related_layers: [layer3]
page_type: concept
---

# ComponentContract

The capabilities a Component requires and the boundaries it provides are collected in one declarative value, `ComponentContract`.
The relationship with a Component is inclusion of the contract returned by the Component, not inheritance.

## ComponentContract's role

`ComponentContract` owns the capabilities a Component requires and the boundaries it provides as one declarative value.
It does not represent a Component inheritance hierarchy; it is the boundary Graph and Compiler use to verify connection and execution conditions.

## Structural elements

| Field | Declares |
|---|---|
| `ports` | A `PortContract` for each role |
| `required_services` | `ServiceRequirement` values required by the whole Component |
| `parts` | `PartSpec` values for held components |
| `lifecycle` | The boundary for `events` and Feedback |
| `state` | `reads`, `writes`, `exports`, and `reads_enumerable` |
| `execution` | `required_runtime_capabilities` and `offered_runtime_capabilities` |
| `assumptions` | Assumptions handled by the Compiler |

The role of `ports` is a named set for connecting the same Component by role in a Graph.
`parts` declares inclusion of child contracts in the parent contract; it does not duplicate child Component implementations.

## Information held by each contract

| Contract | Information held | Main checks |
|---|---|---|
| `PortContract` | Input and output `PortSpec` values | Port name, direction, DataSpec, and cardinality |
| `StateContract` | `StateKey` values read, written, and exposed | Undeclared state access and state effects |
| `LifecycleContract` | Consumed events and FeedbackContract | Event and Feedback compatibility |
| `ExecutionContract` | Required or offered Runtime capabilities | Missing environment capabilities |
| `AssumptionSet` | Assumptions made by the Component | Assumption registration and defaults |

## Construction time and use time

The Component creates its contract with `contract()`, and ComponentNode holds a snapshot for the compilation unit.
The Compiler uses contracts to verify port compatibility, service resolution, state effects, lifecycle, and Runtime capabilities.
The Runtime creates StateViews according to the contract boundaries in the verified ExecutablePlan and applies resulting StatePatches.

## Invariants and diagnostics

Contracts are treated as frozen data classes and are not changed during Compiler verification.
`ports`, `required_services`, and `parts` contain only their corresponding declarative values, and role and component names are verified as identifiers.
State `reads`, `writes`, and `exports` use typed StateKeys, and the Runtime does not expose undeclared keys to a Component.
If a required execution capability is absent from the capabilities offered by CompileContext, the Compiler cannot mark the ExecutablePlan executable.
An uncallable contract or incorrect return type becomes `contract_unavailable`; a held component whose contract differs from its declaration becomes `part_contract_mismatch`.
A component attribute that holds an undeclared child component becomes a non-blocking `undeclared_part` WARNING naming the parent and the attribute; it stays in the structured diagnostic stream and does not block execution.
Inconsistent target ports, required services, state effects, lifecycle, or Runtime capabilities are also reported as Compiler Diagnostics.
An invalid contract type raises `ValidationError` when the contract is created.

## State effects of an undeclared component

A component whose `contract()` implementation is its family default with an empty state does not describe its state effects.
For such a component the Compiler reads the component source and infers state reads and writes for diagnostics only.
State exports are not inferred, because the context has no export operation; exports come from a declaration.
The detection uses the identity of the resolved `contract` implementation: a hand-written declaration, an inherited built-in declaration, an override that only calls `super()`, and the real state declared by the `Algorithm` family default all count as declarations and are never replaced.

Inferred effects are kept in a separate internal record that only one verification rule reads.
They do not change the executable contract, the analysis contract, plan signatures, binding versions, runtime write-gate decisions, or checkpoints.
A conflict found from inferred effects is the non-blocking `inferred_state_effect_conflict` WARNING.
Inference only adds findings: existing findings, including the `opaque_state_effects` WARNING of a Stage, stay as they are.

When any effect cannot be inferred, a family-default component is treated as non-enumerable for verification; a non-Stage family default receives the `opaque_state_effects` WARNING with a message saying its effects could not be inferred.
The analyzer parses the analyzed function's source, resolves names through class and module dictionaries, never evaluates source expressions, never calls the methods it analyzes, and never raises; any failure leaves the component non-enumerable.
A function is analyzed only when every syntax node in it belongs to a small allowlist of modeled forms; nested definitions, lambdas, classes, comprehensions, `global`/`nonlocal`, `match`, assignment expressions, `except*`, augmented assignment, `del`, f-strings, subscripts, starred expressions, and rebinding or shadowing of the context are unknown.
A context value is followed only where it is the receiver of a modeled context attribute or method, a direct positional or keyword argument of a call whose target is analyzed or of a `typing.cast` call, the single name target of an alias assignment, or the returned value; an occurrence anywhere else, such as in a conditional or boolean expression, a container literal, or an iterated expression, makes the whole function unknown.
A name that the function binds itself, as a parameter, an assignment target, a `for`, `with`, or `except` target, or an import alias, is never resolved to a module or class value; using such a name as a state key or a call target makes the function unknown.
Unknown call targets, `getattr` on the context, code defined by `exec`, `functools.partial` attributes, a custom metaclass, and source that cannot be read also make the component non-enumerable.
A method that receives the context is counted whether or not the flow calls it, and a branch that does not run is counted.
A user class whose `__module__` starts with `saealib.` is not distinguished from a framework class.
Effects that pass through explicit arguments rather than the context, such as a mutation of candidates passed in or a read of an RNG passed in, are not represented; declare those effects in `contract()`.
The inference result is cached per class, so a class is analyzed once per process.
Built-in family-default classes that can be graph nodes and whose inference is unknown declare their effects in `contract()`, each verified against runtime observation; classes that cannot be graph nodes keep their inherited declarations.

## Minimal example

```python
from saealib.core import ComponentContract, StateContract


def contract() -> ComponentContract:
    return ComponentContract(state=StateContract())
```

## Extension points

The public path for Compiler rules may vary by release; see the [API reference](../api/index.md) for concrete imports.

## Related pages

See [Specs](specs.md) for port declarations and [Graph](graph.md) for placing them in a Graph.

## References

- {py:class}`saealib.core.ComponentContract`
- {py:class}`saealib.core.PortContract`
- {py:class}`saealib.core.StateContract`
