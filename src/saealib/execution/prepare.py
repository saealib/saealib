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
from typing import TYPE_CHECKING, Any

from saealib.core.compiler import (
    CompileContext,
    Compiler,
    ComponentGraph,
    ExecutablePlan,
)
from saealib.exceptions import ValidationError
from saealib.pipeline import Branch, Loop, Pipeline, PipelineEntry, Repeat, Stage

if TYPE_CHECKING:
    from saealib.core.state import StateKey
    from saealib.problem import Problem

__all__ = [
    "BINDING_SLOTS",
    "declared_components",
    "resolve_declaration",
]

#: The Optimizer slots a binding snapshot carries.  Declaration references
#: resolve against these; ``cbmanager`` is served as a runtime service.
BINDING_SLOTS = (
    "algorithm",
    "strategy",
    "surrogate_manager",
    "acquisition",
    "evaluation_planner",
    "feedback_builder",
    "evaluator",
    "cbmanager",
)


class _InvalidStrategyGraphError(ValidationError):
    pass


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
) -> ExecutablePlan:
    """Build and compile a strategy graph with one consistent context."""
    graph, compile_context = _prepare_graph_and_context(
        strategy,
        provider,
        problem=problem,
        offered_runtime_capabilities=offered_runtime_capabilities,
        initial_state_keys=initial_state_keys,
        _context_factory=_context_factory,
        _verification_enabled=_verification_enabled,
    )
    return Compiler().compile(
        graph,
        compile_context,
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
) -> tuple[ComponentGraph, CompileContext]:
    """Build the graph before resolving runtime capabilities and context."""
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
    build_graph = getattr(strategy, "build_graph", None)
    if not callable(build_graph):
        raise _InvalidStrategyGraphError("strategy requires a callable build_graph")
    graph = build_graph(provider)
    if not isinstance(graph, ComponentGraph):
        raise _InvalidStrategyGraphError(
            "strategy.build_graph must return a ComponentGraph"
        )
    return graph


def resolve_declaration(
    entry: PipelineEntry, bindings: Mapping[str, object]
) -> PipelineEntry:
    """Resolve every ``(slot, role)`` reference in *entry* from one snapshot.

    Returns a plan-owned resolved structure; the declaration *entry* and the
    Stage objects inside it are never mutated.
    """
    return _resolve_entry(entry, bindings, {})


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
