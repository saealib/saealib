"""Report state effect conflicts found by source inference as warnings."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from saealib.core.compiler.diagnostics import ContractPath, Diagnostic, Severity
from saealib.core.compiler.graph import NodeRef
from saealib.core.compiler.state_effect_rules import (
    StateEffectFinding,
    compute_state_effect_findings,
    declared_state_effects,
    resolved_inferred_state_effects,
)
from saealib.core.state.keys import StateKey
from saealib.core.state_inference import InferenceStatus

if TYPE_CHECKING:
    from saealib.core.compiler.compiler import RuleContext, VerificationResult

__all__ = ["InferredStateEffectRule"]

CONFLICT_CODE = "inferred_state_effect_conflict"
UNKNOWN_RESOLUTION = (
    "Declare the state effects in the component's contract, or accept that they "
    "are not verified."
)
_CONFLICT_RESOLUTION = (
    "Declare the state effects in the component's contract so the compiler can "
    "verify their ordering."
)
_KIND_MESSAGES = {
    "uninitialized_state_write": (
        "Inferred uninitialized state write {key} at {path} has no preceding writer."
    ),
    "unreachable_state_read": (
        "Inferred unreachable state read {key} at {path} has no reachable writer."
    ),
    "concurrent_state_read_write": (
        "Inferred concurrent state read/write {key} at {path} is not ordered with "
        "its writer."
    ),
    "concurrent_state_write": (
        "Inferred concurrent state write {key} at {path} is not ordered with the "
        "writer at {related}."
    ),
}


def _path(node: NodeRef) -> ContractPath:
    return ContractPath(components=(node.component_id,), role=node.role)


def _key_name(key: StateKey[object]) -> str:
    return f"{key.namespace}:{key.name}:{key.schema_version}"


def _warning(finding: StateEffectFinding) -> Diagnostic:
    template = _KIND_MESSAGES.get(finding.kind)
    if template is None:
        template = "Inferred {kind} {key} at {path}."
    return Diagnostic(
        severity=Severity.WARNING,
        code=CONFLICT_CODE,
        message=template.format(
            kind=finding.kind,
            key=_key_name(finding.key),
            path=_path(finding.node),
            related=", ".join(str(_path(item)) for item in finding.related),
        ),
        path=_path(finding.node),
        related=tuple(_path(item) for item in finding.related),
        resolutions=(_CONFLICT_RESOLUTION,),
    )


class InferredStateEffectRule:
    """Warn about the conflicts a hand-written declaration would report."""

    namespace = "core"
    name = "inferred_state_effect"
    phase: Literal["verification"] = "verification"

    def apply(self, context: RuleContext) -> VerificationResult:
        """Report unknown inference and conflicts that inferred effects add."""
        from saealib.core.compiler.compiler import VerificationResult

        graph = context.graph
        if not any(
            node._inferred_effects.status is not InferenceStatus.NOT_APPLICABLE
            for node in graph.nodes
        ):
            return VerificationResult(diagnostics=())
        initial = context.compile_context.initial_state_keys
        declared = declared_state_effects(graph)
        diagnostics: list[Diagnostic] = []
        for node in graph.nodes:
            if node._inferred_effects.status is not InferenceStatus.UNKNOWN:
                continue
            ref = NodeRef(component_id=node.component_id, role=node.role)
            base_effect = declared[ref]
            if not base_effect.reads_enumerable or not base_effect.writes_enumerable:
                continue
            diagnostics.append(
                Diagnostic(
                    severity=Severity.WARNING,
                    code="opaque_state_effects",
                    message=(
                        f"The state effects of {_path(ref)} could not be inferred, "
                        "so the component is treated as non-enumerable for "
                        "verification."
                    ),
                    path=_path(ref),
                    resolutions=(UNKNOWN_RESOLUTION,),
                )
            )
        base = set(compute_state_effect_findings(graph, initial, declared))
        extended = compute_state_effect_findings(
            graph, initial, resolved_inferred_state_effects(graph, declared)
        )
        diagnostics.extend(
            _warning(finding) for finding in extended if finding not in base
        )
        return VerificationResult(diagnostics=tuple(diagnostics))
