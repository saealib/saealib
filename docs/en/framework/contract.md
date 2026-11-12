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
