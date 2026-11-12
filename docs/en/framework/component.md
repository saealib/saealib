---
primary_layer: layer4
related_layers: [layer3]
page_type: concept
---

# Component

Components that execute optimization work keep execution behavior separate from the static contract.
The `Component` Protocol defines only the boundary at which a component returns a `ComponentContract`.

## Component's role

Component owns the boundary between execution behavior and its static contract.
The `Component` Protocol requires only returning a `ComponentContract` at compile time; it does not handle port compatibility or StateStore updates.

## Component Protocol and `contract()`

`Component` provides `contract() -> ComponentContract`.
`contract()` returns a pure contract snapshot read at compile time; it is not a place to return mutable execution state.

The `Component` Protocol itself does not define `execute()`.
A graph-native execution component may provide `execute(StateView)` as a separate execution boundary used by the Runtime.

ComponentContract is neither a Component base class nor a derived type.
When a Component holds other components, it includes their contracts in the parent contract as `PartSpec` values.

```python
from saealib.core import Component, ComponentContract


class Normalize(Component):
    def contract(self) -> ComponentContract:
        return ComponentContract()
```

This example shows only the minimal Protocol boundary.
Declare actual inputs, outputs, state, and services in [ComponentContract](contract.md) and [Specs](specs.md).

## PartSpec inclusion

`PartSpec` declares the named contract of another Component held by a Component through a constructor or similar mechanism.
The parent Component includes the child contract in `parts` rather than inheriting from the child Component.
`optional=True` means that the component may be omitted from the configuration.

This inclusion lets the Compiler verify the parent ports and state effects together with the held component's contract in the same plan.
The Component owns component instances at runtime, while `ComponentContract` preserves contract immutability.

Declare every held child component.
When a component attribute holds a child component that is not declared in `parts`, contract checking reports a non-blocking `undeclared_part` WARNING naming the parent type and the attribute.
The warning never registers the child, changes `parts`, or stops execution; it only points at the missing declaration.
Attributes that hold ordinary data — including collaborators that are not components — produce no warning.

## Construction time and use time

Component instances are placed in `ComponentNode` when the Graph is built.
The Compiler reads `contract()` at the start of compilation and uses that snapshot for Resolution and Verification in the same compilation.
The Runtime does not reinterpret contracts; it uses the ports, services, and state boundaries specified by the ExecutablePlan.

## Invariants and diagnostics

Component does not directly modify an arbitrary StateStore; it reads a StateView of declared StateKeys and returns a StatePatch or NodeResult.
Component does not decide port compatibility; it delegates that decision to the Compiler after connection.
If `contract()` does not return a ComponentContract, Graph construction or Compiler snapshotting reports `contract_unavailable`.

## Extension points

The minimal extension for a new Component is to implement `contract()` and the execution boundary used by the Runtime, then return the required contract.

## Related pages

Undeclared state access and inconsistencies between components are verified by [Contract](contract.md), [Graph](graph.md), and [Compiler](compiler.md).
See [Framework extensions](extensions.md) for extension procedures and the [API reference](../api/index.md) for public import paths.

## References

- {py:class}`saealib.core.Component`
- {py:class}`saealib.core.ComponentContract`
