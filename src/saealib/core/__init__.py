"""Public facade for the framework's core vocabulary.

The implementation modules are loaded on first attribute access.  Keeping the
facade lazy avoids making the core package import order part of the API.
"""

from __future__ import annotations

from importlib import import_module
from typing import Final

_LAZY_EXPORTS: Final[dict[str, str]] = {
    "Component": "saealib.core.component",
    "ComponentContract": "saealib.core.contracts.contract",
    "PartSpec": "saealib.core.contracts.contract",
    "DataSpec": "saealib.core.contracts.data",
    "PortContract": "saealib.core.contracts.ports",
    "PortSpec": "saealib.core.contracts.ports",
    "StateContract": "saealib.core.contracts.state",
    "ExecutionContract": "saealib.core.contracts.execution",
    "LifecycleContract": "saealib.core.contracts.lifecycle",
    "AssumptionSet": "saealib.core.contracts.assumptions",
    "ComponentGraph": "saealib.core.compiler.graph",
    "ComponentNode": "saealib.core.compiler.graph",
    "StructuredGraph": "saealib.core.compiler.structured",
    "StructuredRegion": "saealib.core.compiler.regions",
    "SequenceRegion": "saealib.core.compiler.regions",
    "RepeatRegion": "saealib.core.compiler.regions",
    "LoopRegion": "saealib.core.compiler.regions",
    "BranchRegion": "saealib.core.compiler.regions",
    "RegionNode": "saealib.core.compiler.regions",
    "RegionEffect": "saealib.core.compiler.regions",
    "Condition": "saealib.core.compiler.regions",
    "lower_structured": "saealib.core.compiler.lowerer",
    "lower_pipeline": "saealib.core.compiler.lowerer",
    "GraphTemplate": "saealib.core.compiler.graph",
    "CompilationRule": "saealib.core.compiler.compiler",
    "ExecutablePlan": "saealib.core.compiler.compiler",
    "StateStore": "saealib.core.state.store",
    "StateView": "saealib.core.state.store",
    "RuntimeContext": "saealib.core.state.context",
    "ExecutionContext": "saealib.core.state.context",
    "StatePatch": "saealib.core.state.patch",
    "RUNTIME_BASE_COLUMNS": "saealib.core.state.keys",
    "RUNTIME_DECISION_COUNT": "saealib.core.state.keys",
    "RUNTIME_GENERATION": "saealib.core.state.keys",
    "RUNTIME_STEP": "saealib.core.state.keys",
    "RUNTIME_REGION_COUNTS": "saealib.core.state.keys",
    "ExecutionRuntime": "saealib.core.runtime",
    "RegionFrame": "saealib.core.runtime",
    "StructuredPlan": "saealib.core.runtime",
}

__all__ = list(_LAZY_EXPORTS)


def __getattr__(name: str) -> object:
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(__all__) | set(_LAZY_EXPORTS))
