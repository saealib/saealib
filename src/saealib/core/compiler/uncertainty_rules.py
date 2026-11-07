"""Compiler checks for uncertainty required by acquisition functions."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Literal

from saealib.core.compiler.diagnostics import ContractPath, Diagnostic, Severity
from saealib.core.compiler.graph import ComponentGraph, ComponentNode, DataEdge
from saealib.core.state import SURROGATES_PREDICTIONS

if TYPE_CHECKING:
    from saealib.core.compiler.compiler import RuleContext, VerificationResult


_UNCERTAINTY_MISMATCH = "surrogate_uncertainty_mismatch"
_MISSING_ACQUISITION_CHANNEL = "missing_acquisition_channel"
_CHANNEL_UNVERIFIED = "cannot_verify_acquisition_channel"
_UNCERTAINTY_UNVERIFIED = "cannot_verify_surrogate_uncertainty"


def surrogate_uncertainty_diagnostics(
    manager: object,
    acquisition: object,
    *,
    path: ContractPath,
    related: ContractPath | None = None,
) -> tuple[Diagnostic, ...]:
    """Check each acquisition channel against its matching manager output."""
    channels = _acquisition_channels(acquisition)
    manager_channels = _manager_channels(manager)
    channel_names_known = _manager_channel_names_known(manager)
    findings: list[Diagnostic] = []
    for name, channel_acquisition in channels.items():
        if name not in manager_channels:
            code = (
                _MISSING_ACQUISITION_CHANNEL
                if channel_names_known
                else _CHANNEL_UNVERIFIED
            )
            severity = Severity.ERROR if channel_names_known else Severity.WARNING
            findings.append(
                Diagnostic(
                    severity=severity,
                    code=code,
                    message=(
                        f"{type(acquisition).__name__} requires prediction channel "
                        f"{name!r}, but {type(manager).__name__} does not provide it."
                        if channel_names_known
                        else f"Prediction channel {name!r} required by "
                        f"{type(acquisition).__name__} cannot be checked because "
                        f"{type(manager).__name__} does not declare its channel names."
                    ),
                    path=path,
                    related=() if related is None else (related,),
                    resolutions=(
                        (
                            f"Provide a surrogate-manager channel named {name!r}, "
                            "or configure an acquisition channel that exists."
                            if channel_names_known
                            else "Declare the manager's prediction channel names."
                        ),
                    ),
                )
            )
            continue
        provider = manager_channels[name]
        missing_nested_objective = _missing_nested_objective(provider)
        if missing_nested_objective is not None:
            findings.append(
                Diagnostic(
                    severity=Severity.ERROR,
                    code=_MISSING_ACQUISITION_CHANNEL,
                    message=(
                        f"{type(acquisition).__name__} requires prediction channel "
                        f"{name!r}, but nested "
                        f"{type(missing_nested_objective).__name__} does not provide "
                        "the required 'objective' channel."
                    ),
                    path=path,
                    related=() if related is None else (related,),
                    resolutions=(
                        "Provide an 'objective' channel from each nested composite "
                        "manager, or remove the nested manager.",
                    ),
                )
            )
            continue
        if _attribute(channel_acquisition, "requires_uncertainty") is not True:
            continue
        support, provider_name = _uncertainty_support(provider)
        if support is True:
            continue
        if support is False:
            severity = Severity.ERROR
            code = _UNCERTAINTY_MISMATCH
            message = (
                f"{type(channel_acquisition).__name__} requires uncertainty "
                f"estimates, but {provider_name} does not provide them "
                f"for prediction channel {name!r}."
            )
            resolution = (
                "Use a surrogate that provides uncertainty estimates, or choose "
                "an acquisition function that does not require them.",
            )
        else:
            severity = Severity.WARNING
            code = _UNCERTAINTY_UNVERIFIED
            message = (
                f"Uncertainty support for prediction channel {name!r} cannot be "
                f"determined from {provider_name}."
            )
            resolution = (
                "Declare whether the surrogate or manager provides uncertainty "
                "estimates.",
            )
        findings.append(
            Diagnostic(
                severity=severity,
                code=code,
                message=message,
                path=path,
                related=() if related is None else (related,),
                resolutions=resolution,
            )
        )
    return tuple(findings)


def _acquisition_channels(acquisition: object) -> Mapping[str, object]:
    acquisitions = _attribute(acquisition, "acquisitions")
    if isinstance(acquisitions, Mapping):
        return acquisitions
    return {"objective": acquisition}


def _manager_channels(manager: object) -> Mapping[str, object]:
    managers = _attribute(manager, "managers")
    if isinstance(managers, Mapping):
        return managers
    return {"objective": manager}


def _manager_channel_names_known(manager: object) -> bool:
    managers = _attribute(manager, "managers")
    return isinstance(managers, Mapping)


def _missing_nested_objective(manager: object) -> object | None:
    managers = _attribute(manager, "managers")
    if not isinstance(managers, Mapping):
        return None
    objective = managers.get("objective", _MISSING)
    if objective is _MISSING:
        return manager
    return _missing_nested_objective(objective)


def _uncertainty_support(provider: object) -> tuple[bool | None, str]:
    declared = _boolean_declaration(provider, "provides_uncertainty")
    if declared is not None:
        return declared, type(provider).__name__

    surrogate = _attribute(provider, "surrogate")
    if surrogate is not _MISSING:
        declared = _boolean_declaration(surrogate, "provides_uncertainty")
        if declared is not None:
            return declared, type(surrogate).__name__
        return None, type(surrogate).__name__

    managers = _attribute(provider, "managers")
    if isinstance(managers, Mapping):
        primary = managers.get("objective", _MISSING)
        if primary is not _MISSING:
            return _uncertainty_support(primary)

    return None, type(provider).__name__


_MISSING = object()


def _attribute(value: object, name: str) -> object:
    try:
        return getattr(value, name)
    except Exception:
        return _MISSING


def _boolean_declaration(value: object, name: str) -> bool | None:
    declaration = _attribute(value, name)
    return declaration if isinstance(declaration, bool) else None


class SurrogateUncertaintyRule:
    """Check acquisition inputs against their prediction-producing managers."""

    namespace = "core"
    name = "surrogate_uncertainty"
    phase: Literal["verification"] = "verification"

    def apply(self, context: RuleContext) -> VerificationResult:
        """Diagnose uncertainty and channel mismatches on prediction edges."""
        from saealib.core.compiler.compiler import VerificationResult

        graph = context.graph
        nodes = {node.component_id: node for node in graph.nodes}
        findings: list[Diagnostic] = []
        for edge in graph.data_edges:
            target = nodes.get(edge.target.component_id)
            if target is None or not _is_acquisition_prediction_edge(edge, target):
                continue
            source_state = nodes.get(edge.source.component_id)
            if source_state is None or not _is_prediction_state(source_state):
                continue
            manager_node = _prediction_manager(graph, source_state)
            if manager_node is None:
                continue
            findings.extend(
                surrogate_uncertainty_diagnostics(
                    manager_node.component,
                    target.component,
                    path=_node_path(target, edge),
                    related=_manager_path(graph, source_state, manager_node),
                )
            )
        return VerificationResult(diagnostics=tuple(findings))


def _is_acquisition_prediction_edge(edge: DataEdge, target: ComponentNode) -> bool:
    return (
        edge.target.role == "acquisition"
        and edge.target_port == "prediction"
        and "acquisition" in target.contract.ports
    )


def _is_prediction_state(node: ComponentNode) -> bool:
    return getattr(node.component, "state_key", None) == SURROGATES_PREDICTIONS


def _prediction_manager(
    graph: ComponentGraph, state_node: ComponentNode
) -> ComponentNode | None:
    flow_origin = getattr(state_node.component, "flow_origin", None)
    if not isinstance(flow_origin, str):
        return None
    try:
        return graph.node_by_id(flow_origin)
    except KeyError:
        return None


def _node_path(node: ComponentNode, edge: DataEdge) -> ContractPath:
    return ContractPath(
        components=(node.component_id,),
        role=edge.target.role,
        port=edge.target_port,
    )


def _manager_path(
    graph: ComponentGraph, state_node: ComponentNode, manager_node: ComponentNode
) -> ContractPath:
    for edge in graph.data_edges:
        if (
            edge.target.component_id == state_node.component_id
            and edge.target_port == "value"
            and edge.source.component_id == manager_node.component_id
            and edge.source_port == "prediction"
        ):
            return ContractPath(
                components=(manager_node.component_id,),
                role=edge.source.role,
                port=edge.source_port,
            )
    return ContractPath(
        components=(manager_node.component_id,), role="predictor", port="prediction"
    )
