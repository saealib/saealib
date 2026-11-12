---
primary_layer: layer4
related_layers: [layer3]
page_type: concept
---

# Compiler

After resolving and verifying a `ComponentGraph`, the Compiler returns an `ExecutablePlan` separate from execution.
The Compiler does not run the Runtime; it records the capabilities and diagnostics the Runtime needs in the plan.

## Compiler's role

The Compiler owns the boundary that resolves and verifies a ComponentGraph and returns an ExecutablePlan separate from execution.
It does not run the Runtime and records the capabilities and diagnostics the Runtime needs in the plan.

## Compilation, verification, and resolution

Compilation proceeds through contract snapshotting, Resolution, and Verification in that order.
`ResolutionRule` returns candidates for resolving services, adapters, and schema bindings as proposals for claimed Graph locations.
`VerificationRule` observes the resolved Graph and returns Diagnostics without changing it.
The shared `CompilationRule` boundary contains a namespace, name, phase, and `apply(RuleContext)`.

Resolution repeats until proposals converge; conflicts at the same Graph location and undeclared changes become error Diagnostics.
Verification checks Graph well-formedness, port compatibility, services, data flow, state effects, lifecycle, and Runtime capabilities.

## Executable plan

`ExecutablePlan` is an immutable value holding the resolved Graph, Diagnostics, required Runtime capabilities, enabled rules, inserted adapters, and contract snapshots.
Because the plan owns neither an execution position nor current state, the same plan can be passed to RuntimeSession to separate execution from resumption.
When Diagnostics contain errors, whether the plan is accepted as executable input follows the caller's checks and the Runtime contract.

### From verification to execution

```{mermaid}
flowchart LR
    C[Component] --> N[ComponentNode]
    N --> G[ComponentGraph]
    G --> K[Compiler]
    K --> P[ExecutablePlan]
    P --> R[ExecutionRuntime]
```

Component through Graph is configuration, the Compiler performs resolution and verification, and everything after ExecutablePlan belongs to the Runtime.
The Runtime applies StatePatches, events, and commands from plan node results and passes the verified state boundary to the next step.

## Construction time and use time

After the Graph is built, the Compiler reads contract snapshots and generates an ExecutablePlan through Resolution and Verification.
The Runtime uses the generated ExecutablePlan and does not repeat Compiler resolution or verification during execution.

## Invariants and diagnostics

`Diagnostic` is a verification result with severity, code, message, contract path, and resolution hint.
The Compiler does not represent failures only as arbitrary exceptions; it collects multiple issues in a DiagnosticBag for plan decisions.
Typical failures include unknown node endpoints, unconnected required ports, mismatched DataSpecs, missing required services, undeclared state access, missing Runtime capabilities, and components that hold undeclared child components (`undeclared_part` WARNING).
The implementation reports them with codes such as `invalid_graph_edge`, `incompatible_port`, `unresolved_service`, and `unknown_runtime_capability`.
Compiler rules do not change undeclared Graph locations or arbitrarily change a StructuredRegion execution tree during resolution.
Violations become Diagnostics such as `unclaimed_rewrite`, `conflicting_rewrite`, and `structured_execution_mutation`.
Adapters must state the meaning and losslessness of their conversion and must not bypass port-compatibility checks.
Unresolvable or ambiguous adapters become Diagnostics such as `ambiguous_adapter`.

## Extension points

See the release-specific [API reference](../api/index.md) for Compiler rule types and import paths.
See [Framework extensions](extensions.md) for extension guidance.

## Related pages

See [Contract](contract.md) and [Specs](specs.md) for declaring components, and [Runtime](runtime.md) for using an execution plan.

## References

- {py:class}`saealib.core.CompilationRule`
- {py:class}`saealib.core.ExecutablePlan`
- {py:class}`saealib.core.ComponentGraph`
