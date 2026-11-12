"""Execution preparation for declared Stages.

Resolving a declaration's ``(slot, role)`` references is execution
preparation on the optimizer side: it reads one binding snapshot and
produces a resolved structure of Stage instances owned by the plan being
prepared.  The declaration itself is never mutated, so the same declaration
can be prepared for two Optimizers, or rebuilt for the same Optimizer,
without one plan seeing the other's components.  A component explicitly
passed to a Stage stays local to that Stage; only slot bindings land in the
plan's shared snapshot, and default components are created once per plan.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import copy
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from saealib.core.compiler import (
    CompileContext,
    Compiler,
    ComponentGraph,
    ContractPath,
    Diagnostic,
    ExecutablePlan,
    Severity,
)
from saealib.core.compiler.contract_diagnostics import (
    _optimizer_configuration_diagnostics,
)
from saealib.core.state import SCORES, SURROGATES_PREDICTIONS
from saealib.exceptions import ValidationError
from saealib.execution.bindings import (
    BINDING_SLOTS,
    _ComponentResolution,
    _resolve_feature_encoder,
)
from saealib.pipeline import Branch, Loop, Pipeline, PipelineEntry, Repeat, Stage

if TYPE_CHECKING:
    from saealib.core.state import StateKey
    from saealib.problem import Problem

__all__ = [
    "BINDING_SLOTS",
    "declared_components",
    "resolve_declaration",
]


class _InvalidStrategyGraphError(ValidationError):
    pass


class _StageDefaults(dict):
    def __init__(self, selections: list[tuple[str, str, object, str]] | None) -> None:
        super().__init__()
        self.selections = selections

    def record_fallback(
        self, slot: str, role: str, component: object, reason: str
    ) -> None:
        if self.selections is not None:
            self.selections.append((slot, role, component, reason))


class _PreparedProvider:
    _provider: Any
    _resolution: _ComponentResolution
    _component_resolution: _ComponentResolution
    _stage_selections: list[tuple[str, str, object, str]]

    def __init__(
        self,
        provider: Any,
        resolution: _ComponentResolution,
        selections: list[tuple[str, str, object, str]],
    ) -> None:
        object.__setattr__(self, "_provider", provider)
        object.__setattr__(self, "_resolution", resolution)
        object.__setattr__(self, "_component_resolution", resolution)
        object.__setattr__(self, "_stage_selections", selections)

    def binding_values(self) -> dict[str, object]:
        return dict(self._resolution.bindings)

    def __getattr__(self, name: str) -> object:
        if name in BINDING_SLOTS:
            return self._resolution.bindings.get(name)
        return getattr(self._provider, name)

    def __setattr__(self, name: str, value: object) -> None:
        if name in BINDING_SLOTS:
            setattr(self._provider, name, value)
        else:
            setattr(self._provider, name, value)


class _PreparedServices:
    def __init__(self, services: object, feature_encoder: object) -> None:
        self._services = services
        self._feature_encoder = feature_encoder

    def get(self, name: str) -> object | None:
        if name == "FeatureEncoder":
            return self._feature_encoder
        getter = getattr(self._services, "get", None)
        return getter(name) if callable(getter) else None

    def __getattr__(self, name: str) -> object:
        return getattr(self._services, name)


class _PreparedSearchSpace:
    def __init__(self, space: object, feature_encoder: object) -> None:
        self._space = space
        self.services = _PreparedServices(
            getattr(space, "services", None), feature_encoder
        )

    def __getattr__(self, name: str) -> object:
        return getattr(self._space, name)


def _requires_input(component: object, input_name: str) -> bool:
    contract_method = getattr(component, "contract", None)
    if not callable(contract_method):
        return False
    try:
        contract = contract_method()
    except Exception:
        return False
    return any(
        port.name == input_name and not port.optional
        for role in contract.ports.values()
        for port in role.inputs
    )


def _component_node_ids(graph: ComponentGraph, component: object) -> tuple[str, ...]:
    selected = []
    for node in graph.nodes:
        if node.component is component:
            selected.append(node.component_id)
        stage = getattr(node.component, "stage", None)
        attributes = getattr(stage, "__dict__", {})
        if (
            stage is not None
            and isinstance(attributes, Mapping)
            and any(value is component for value in attributes.values())
        ):
            selected.append(node.component_id)
    return tuple(dict.fromkeys(selected))


def _graph_uses_component(graph: ComponentGraph, component: object) -> bool:
    return bool(_component_node_ids(graph, component))


def _graph_precedes(
    graph: ComponentGraph, source_id: str, target_ids: tuple[str, ...]
) -> bool | None:
    successors: dict[str, set[str]] = {}
    for edge in (*graph.control_edges, *graph.data_edges):
        successors.setdefault(edge.source.component_id, set()).add(
            edge.target.component_id
        )

    def reachable(source: str, target: str) -> bool:
        pending = [source]
        visited = set()
        while pending:
            current = pending.pop()
            if current == target:
                return True
            if current in visited:
                continue
            visited.add(current)
            pending.extend(successors.get(current, ()))
        return False

    if any(reachable(source_id, target) for target in target_ids):
        return True
    if any(reachable(target, source_id) for target in target_ids):
        return False
    return None


def _state_producer_status(
    graph: ComponentGraph,
    state_key: StateKey[object],
    consumer: object,
) -> bool | None:
    targets = _component_node_ids(graph, consumer)
    if not targets:
        return None
    uncertain = False
    for node in graph.nodes:
        relation = _graph_precedes(graph, node.component_id, targets)
        if state_key in node.analysis_contract.state.writes:
            if relation is True:
                return True
            if relation is None:
                uncertain = True
        elif (
            not node.analysis_contract.state.writes_enumerable and relation is not False
        ):
            uncertain = True
    return None if uncertain else False


def _prepare_plan(
    strategy: Any,
    provider: Any,
    *,
    problem: Problem | None,
    offered_runtime_capabilities: frozenset[str] | None,
    initial_state_keys: frozenset[StateKey[object]],
    _context_factory: Callable[[], tuple[Problem | None, frozenset[str] | None]]
    | None = None,
    _verification_enabled: bool = True,
    _component_resolution: _ComponentResolution | None = None,
    _require_initializer: bool = True,
) -> ExecutablePlan:
    """Build and compile a strategy graph with one consistent context."""
    stage_selections: list[tuple[str, str, object, str]] = []
    graph, compile_context = _prepare_graph_and_context(
        strategy,
        provider,
        problem=problem,
        offered_runtime_capabilities=offered_runtime_capabilities,
        initial_state_keys=initial_state_keys,
        _context_factory=_context_factory,
        _verification_enabled=_verification_enabled,
        _component_resolution=_component_resolution,
        _stage_selections=stage_selections,
    )
    consumes_feature_encoder = _graph_requires_service(graph, "FeatureEncoder")
    effective_resolution = _component_resolution
    if effective_resolution is not None:
        if consumes_feature_encoder:
            feature_encoder, feature_encoder_origin = _resolve_feature_encoder(
                provider, compile_context.space
            )
            effective_resolution = replace(
                effective_resolution,
                feature_encoder=feature_encoder,
                feature_encoder_origin=feature_encoder_origin,
                feature_encoder_space=compile_context.space,
            )
        elif (
            effective_resolution.feature_encoder_origin is not None
            and effective_resolution.feature_encoder_origin.source
            == "framework default"
        ):
            effective_resolution = replace(
                effective_resolution,
                feature_encoder=None,
                feature_encoder_origin=None,
            )
    else:
        feature_encoder = None
    feature_encoder = (
        None if effective_resolution is None else effective_resolution.feature_encoder
    )
    feature_encoder_origin = (
        None
        if effective_resolution is None
        else effective_resolution.feature_encoder_origin
    )
    if consumes_feature_encoder and feature_encoder is not None:
        compile_context = replace(
            compile_context,
            space=_PreparedSearchSpace(compile_context.space, feature_encoder),
        )
    executable = Compiler().compile(
        graph,
        compile_context,
    )
    if effective_resolution is None:
        return executable
    object.__setattr__(provider, "_component_resolution", effective_resolution)
    diagnostics = list(executable.diagnostics)
    diagnostics.extend(
        _optimizer_configuration_diagnostics(
            effective_resolution.components,
            {
                slot: (origin.source, origin.reason)
                for slot, origin in effective_resolution.origins.items()
            },
            compile_context.problem,
            effective_resolution.components.get(
                "evaluator", getattr(provider, "evaluator", None)
            ),
            require_initializer=_require_initializer,
        )
    )
    if consumes_feature_encoder and feature_encoder_origin is not None:
        if feature_encoder_origin.source == "framework default":
            space_kind = getattr(
                getattr(compile_context.space, "representation", None), "kind", ""
            )
            if space_kind == "vector":
                message = (
                    "FeatureEncoder selected identity encoding from framework "
                    f"default. Source: {feature_encoder_origin.source}. "
                    f"Reason: {feature_encoder_origin.reason}"
                )
                severity = Severity.INFO
            else:
                message = (
                    "A default encoder was selected. Encoding affects what the "
                    "surrogate learns; an explicit choice is recommended. "
                    f"Selected {type(feature_encoder).__name__} from "
                    f"{feature_encoder_origin.source}: "
                    f"{feature_encoder_origin.reason}"
                )
                severity = Severity.WARNING
            diagnostics.append(
                Diagnostic(
                    severity=severity,
                    code="feature_encoder_default_selected",
                    message=message,
                    path=ContractPath(components=("optimizer", "feature_encoder")),
                )
            )
        else:
            try:
                has_encoder = callable(getattr(feature_encoder, "encode", None))
            except Exception:
                has_encoder = False
            if not has_encoder:
                diagnostics.append(
                    Diagnostic(
                        severity=Severity.ERROR,
                        code="component_protocol_mismatch",
                        message=(
                            "The explicitly configured FeatureEncoder does not "
                            "provide a callable encode() method."
                        ),
                        path=ContractPath(components=("optimizer", "feature_encoder")),
                        resolutions=(
                            "Provide an explicit SearchSpace FeatureEncoder with "
                            "a callable encode() method.",
                        ),
                    )
                )
    required_inputs = (
        (
            "evaluation_planner",
            "acquisition",
            SCORES,
            "acquisition scores",
            "score-producing component",
            "Use a strategy that produces acquisition scores or select a planner "
            "that does not require them.",
        ),
        (
            "feedback_builder",
            "prediction",
            SURROGATES_PREDICTIONS,
            "objective predictions",
            "prediction-producing component",
            "Use a strategy that produces objective predictions or select a "
            "feedback builder that does not require them.",
        ),
    )
    for slot, input_name, state_key, required, producer, resolution in required_inputs:
        component = effective_resolution.bindings.get(slot)
        if (
            component is None
            or not _requires_input(component, input_name)
            or not _graph_uses_component(graph, component)
            or _state_producer_status(graph, state_key, component) is not False
        ):
            continue
        origin = effective_resolution.origins.get(slot)
        selected = (
            f"selected {type(component).__name__} from {origin.source}: {origin.reason}"
            if origin is not None
            else f"selected {type(component).__name__}"
        )
        diagnostics.append(
            Diagnostic(
                severity=Severity.ERROR,
                code="unresolved_input",
                message=(
                    f"{slot} {selected} requires {required}, but the prepared "
                    f"graph has no {producer}."
                ),
                path=ContractPath(
                    components=("optimizer", slot),
                    port=input_name,
                ),
                resolutions=(resolution,),
            )
        )
    selection_resolutions = tuple(
        f"{slot} selected {type(origin.value).__name__} from "
        f"{origin.source}: {origin.reason}"
        for slot, origin in effective_resolution.origins.items()
        if slot in {"algorithm", "strategy"}
    )
    diagnosed = tuple(
        replace(
            diagnostic,
            message=(
                f"{diagnostic.message} Prepared selections: "
                f"{'; '.join(selection_resolutions)}."
            ),
            resolutions=(*diagnostic.resolutions, *selection_resolutions),
        )
        if diagnostic.code in {"missing_runtime_capability", "unresolved_input"}
        else diagnostic
        for diagnostic in diagnostics
    )
    stage_diagnostics = tuple(
        Diagnostic(
            severity=Severity.INFO,
            code="component_binding_selected",
            message=(
                f"{slot}[{role}] selected {type(component).__name__} from "
                f"Stage fallback: {reason}"
            ),
            path=ContractPath(components=("optimizer", slot), role=role),
        )
        for slot, role, component, reason in stage_selections
    )
    return replace(
        executable,
        diagnostics=tuple(dict.fromkeys((*diagnosed, *stage_diagnostics))),
    )


def _prepare_graph_and_context(
    strategy: Any,
    provider: Any,
    *,
    problem: Problem | None,
    offered_runtime_capabilities: frozenset[str] | None,
    initial_state_keys: frozenset[StateKey[object]],
    _context_factory: Callable[[], tuple[Problem | None, frozenset[str] | None]]
    | None = None,
    _verification_enabled: bool = True,
    _component_resolution: _ComponentResolution | None = None,
    _stage_selections: list[tuple[str, str, object, str]] | None = None,
) -> tuple[ComponentGraph, CompileContext]:
    """Build the graph before resolving runtime capabilities and context."""
    if _component_resolution is not None:
        provider = _PreparedProvider(
            provider,
            _component_resolution,
            [] if _stage_selections is None else _stage_selections,
        )
    graph = _build_strategy_graph(strategy, provider)
    if _context_factory is not None:
        problem, offered_runtime_capabilities = _context_factory()
    return graph, CompileContext(
        space=getattr(problem, "space", None),
        problem=problem,
        offered_runtime_capabilities=offered_runtime_capabilities,
        initial_state_keys=initial_state_keys,
        _verification_enabled=_verification_enabled,
    )


def _build_strategy_graph(strategy: Any, provider: Any) -> ComponentGraph:
    """Build the strategy graph using the execution preparation path."""
    from saealib.strategies.base import warn_deprecated_build_graph_override

    build_graph = getattr(strategy, "build_graph", None)
    if not callable(build_graph):
        raise _InvalidStrategyGraphError("strategy requires a callable build_graph")
    warn_deprecated_build_graph_override(strategy)
    graph = build_graph(provider)
    if not isinstance(graph, ComponentGraph):
        raise _InvalidStrategyGraphError(
            "strategy.build_graph must return a ComponentGraph"
        )
    return graph


def _contract_requires_service(contract: object, service_name: str) -> bool:
    if any(
        getattr(requirement, "name", None) == service_name
        for requirement in getattr(contract, "required_services", ())
    ):
        return True
    for port_contract in getattr(contract, "ports", {}).values():
        for port in (*port_contract.inputs, *port_contract.outputs):
            if any(
                requirement.name == service_name
                for requirement in port.required_services
            ):
                return True
    return any(
        _contract_requires_service(part.contract, service_name)
        for part in getattr(contract, "parts", ())
    )


def _graph_requires_service(graph: ComponentGraph, service_name: str) -> bool:
    return any(
        _contract_requires_service(node.contract, service_name) for node in graph.nodes
    )


def resolve_declaration(
    entry: PipelineEntry, bindings: Mapping[str, object]
) -> PipelineEntry:
    """Resolve every ``(slot, role)`` reference in *entry* from one snapshot.

    Returns a plan-owned resolved structure; the declaration *entry* and the
    Stage objects inside it are never mutated.
    """
    return _resolve_declaration(entry, bindings, None)


def _resolve_declaration(
    entry: PipelineEntry,
    bindings: Mapping[str, object],
    selections: list[tuple[str, str, object, str]] | None,
) -> PipelineEntry:
    return _resolve_entry(entry, bindings, _StageDefaults(selections))


def declared_components(entry: Any) -> Any:
    """Map a resolved declaration onto the component vocabulary.

    Leaf stages become graph-native StageStateViewAdapter components;
    Pipeline and structural control values are rebuilt with the same names
    so lowered node IDs are identical to the declaration's.
    """
    from saealib.stages import stage_component

    if isinstance(entry, Pipeline):
        return Pipeline(
            [declared_components(step) for step in entry.steps],
            name=entry.name,
            label=entry.label,
            notation=entry.notation,
        )
    if isinstance(entry, Repeat):
        return Repeat(declared_components(entry.body), entry.count, name=entry.name)
    if isinstance(entry, Loop):
        return Loop(declared_components(entry.body), until=entry.until, name=entry.name)
    if isinstance(entry, Branch):
        return Branch(
            entry.condition,
            then=declared_components(entry.then),
            else_=(
                declared_components(entry.else_) if entry.else_ is not None else None
            ),
            name=entry.name,
        )
    if isinstance(entry, Stage):
        return stage_component(entry)
    return entry


def _resolve_entry(
    entry: Any,
    bindings: Mapping[str, object],
    defaults: dict[tuple[str, str], object],
) -> Any:
    if isinstance(entry, Pipeline):
        return Pipeline(
            [_resolve_entry(step, bindings, defaults) for step in entry.steps],
            name=entry.name,
            label=entry.label,
            notation=entry.notation,
        )
    if isinstance(entry, Repeat):
        return Repeat(
            _resolve_entry(entry.body, bindings, defaults),
            entry.count,
            name=entry.name,
        )
    if isinstance(entry, Loop):
        return Loop(
            _resolve_entry(entry.body, bindings, defaults),
            until=entry.until,
            name=entry.name,
        )
    if isinstance(entry, Branch):
        return Branch(
            entry.condition,
            then=_resolve_entry(entry.then, bindings, defaults),
            else_=(
                _resolve_entry(entry.else_, bindings, defaults)
                if entry.else_ is not None
                else None
            ),
            name=entry.name,
        )
    if isinstance(entry, Stage):
        return resolve_stage(entry, bindings, defaults)
    return entry


def _copy_own_value(value: Any) -> Any:
    """Copy a declared Stage's own containers, keeping members by identity.

    List, dict, and set containers are rebuilt so two plans never share a
    Stage's mutable state.  Non-container members (declared child Stages,
    bound components, plain values) keep their identity: children are
    resolved recursively, and binding identity must not change.
    """
    if isinstance(value, list):
        return [_copy_own_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _copy_own_value(item) for key, item in value.items()}
    if isinstance(value, set):
        return set(value)
    return value


def _copy_declared_state(stage: Stage) -> Stage:
    """Copy a declared Stage without sharing its own mutable containers."""
    resolved = copy(stage)
    resolved.__dict__ = {
        name: _copy_own_value(value) for name, value in stage.__dict__.items()
    }
    return resolved


def resolve_stage(
    stage: Stage,
    bindings: Mapping[str, object],
    defaults: dict[tuple[str, str], object],
) -> Stage:
    """Return the plan-owned resolved instance of one declaration Stage.

    A Stage holding references resolves through its own ``_resolve_stage``,
    and the result keeps the declared instance's type and its instance
    ``name``/``label``/``notation`` so subclasses of the built-in Stages and
    renamed declarations survive resolution: only the component bindings and
    everything derived from them (reference table, read sets, dispatch
    proxies, caches) come from the freshly built Stage.  Declared child
    Stages are kept and resolved recursively, and the Stage's own mutable
    containers are copied so two plans never share state; bound components
    keep their identity.  Any other Stage is copied the same way so the
    declaration's instance is never shared between plans.  Composite
    ``stages`` are resolved recursively.
    """
    if getattr(stage, "_refs", ()):
        fresh = stage._resolve_stage(bindings, defaults)
        resolved = _copy_declared_state(stage)
        resolved.__dict__.update(
            {
                name: value
                for name, value in fresh.__dict__.items()
                if name != "stages" or not stage.stages
            }
        )
    else:
        resolved = _copy_declared_state(stage)
    boundary = getattr(stage, "execution_boundary", "transaction")
    if boundary != "transaction":
        resolved.execution_boundary = boundary
    if resolved is not stage and resolved.stages:
        resolved.stages = [
            resolve_stage(child, bindings, defaults) for child in resolved.stages
        ]
    return resolved
