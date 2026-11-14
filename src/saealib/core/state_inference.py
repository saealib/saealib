"""Infer the state effects of components that keep their family default contract."""

from __future__ import annotations

import abc
import ast
import builtins
import enum
import inspect
import sys
import textwrap
import threading
import types
import typing
import weakref
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from types import FunctionType, ModuleType
from typing import cast

from saealib.core.contracts import ComponentContract, StateContract
from saealib.core.state.context import _CONTEXT_STATE_KEYS
from saealib.core.state.keys import (
    PROPOSALS_ID_ALLOCATOR,
    RUNTIME_CANDIDATE_ID_ALLOCATOR,
    RUNTIME_DECISION_COUNT,
    RUNTIME_REGION_COUNTS,
    RUNTIME_REQUEST_ID_ALLOCATOR,
    RUNTIME_RNG,
    RUNTIME_STEP,
    SURROGATES_PREDICTIONS,
    StateKey,
)
from saealib.core.state.patch import StatePatch

__all__ = [
    "NOT_APPLICABLE",
    "InferenceStatus",
    "StateInference",
    "clear_state_effect_cache",
    "infer_state_effects",
]

_UNRESOLVED = object()


def _static_mro(cls: type) -> tuple[type, ...]:
    return cast(tuple[type, ...], type.__getattribute__(cls, "__mro__"))


def _class_dict(cls: type) -> Mapping[str, object]:
    return cast(Mapping[str, object], type.__getattribute__(cls, "__dict__"))


_CONTEXT_PARAMETER_NAMES = frozenset(
    {"ctx", "state", "context", "state_view", "view", "tell_state"}
)
_CONTEXT_TYPE_NAMES = (
    "OptimizationState",
    "StateView",
    "ExecutionContext",
    "RuntimeContext",
)
_ALIAS_FUNCTIONS = frozenset({"algorithm_context", "cast"})
_ALIAS_METHODS = frozenset({"replace", "_for_stage_compatibility"})
_NON_STATE_ATTRIBUTES = frozenset(
    {
        "problem",
        "dim",
        "n_obj",
        "lb",
        "ub",
        "direction",
        "comparator",
        "dispatch",
        "context",
        "compiled_service",
        "get",
        "get_state",
        "set_state",
        "contains",
        "replace",
        "_for_stage_compatibility",
    }
)
_CONTEXT_METHOD_ATTRIBUTES = frozenset(
    {
        "get",
        "get_state",
        "set_state",
        "contains",
        "replace",
        "_for_stage_compatibility",
        "compiled_service",
    }
)
_CONTEXT_DATA_ATTRIBUTES = _NON_STATE_ATTRIBUTES - _CONTEXT_METHOD_ATTRIBUTES
_SKIP_METHODS = frozenset(
    {
        "contract",
        "analysis_contract",
        "runtime_contract",
        "to_pseudocode",
        "to_spec",
        "to_text",
        "_registry_spec",
        "__repr__",
        "__init__",
        "__new__",
        "state_export",
        "state_import",
        "_resolve_stage",
    }
)
_MUTATOR_NAMES = frozenset(
    {
        "add",
        "append",
        "clear",
        "delete",
        "extend",
        "pop",
        "reorder",
        "truncate",
        "update_array",
        "update_rows",
        "update_value",
        "mod_structure",
        "mod_value",
        "_extend_internal",
        "_append_internal",
        "_replace_from_population",
        "_replace_genome_rows",
        "_assign_ids",
        "_resize",
        "_get_mutable_array",
        "_reorder_genomes",
    }
)
_STATEFUL_KEYS = frozenset(
    {
        RUNTIME_RNG,
        PROPOSALS_ID_ALLOCATOR,
        RUNTIME_CANDIDATE_ID_ALLOCATOR,
        RUNTIME_REQUEST_ID_ALLOCATOR,
    }
)
_SIGNAL_KEYS = frozenset(
    {
        StateKey(namespace="runtime", name="async_fatal", schema_version=1),
    }
)

_FIELD_KEYS: dict[str, tuple[StateKey[object], ...]] = {
    name: value if isinstance(value, tuple) else (value,)
    for name, value in _CONTEXT_STATE_KEYS.items()
}
_FIELD_KEYS.update(
    {
        "predictions": (SURROGATES_PREDICTIONS,),
        "decision_count": (RUNTIME_DECISION_COUNT,),
        "step": (RUNTIME_STEP,),
        "region_counts": (RUNTIME_REGION_COUNTS,),
    }
)

_ALLOWED_VALUE_TYPES = (
    StateKey,
    str,
    int,
    float,
    bool,
    bytes,
    ModuleType,
    type,
    FunctionType,
    enum.Enum,
)

_OPERATOR_NODES = (
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.MatMult,
    ast.Div,
    ast.FloorDiv,
    ast.Mod,
    ast.Pow,
    ast.LShift,
    ast.RShift,
    ast.BitOr,
    ast.BitXor,
    ast.BitAnd,
    ast.UAdd,
    ast.USub,
    ast.Not,
    ast.Invert,
    ast.And,
    ast.Or,
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
    ast.Is,
    ast.IsNot,
    ast.In,
    ast.NotIn,
)
_ALLOWED_AST_NODES: frozenset[type[ast.AST]] = frozenset(
    {
        ast.arguments,
        ast.arg,
        ast.alias,
        ast.keyword,
        ast.Return,
        ast.Assign,
        ast.AnnAssign,
        ast.Expr,
        ast.Pass,
        ast.If,
        ast.For,
        ast.While,
        ast.Break,
        ast.Continue,
        ast.With,
        ast.withitem,
        ast.Try,
        ast.ExceptHandler,
        ast.Raise,
        ast.Assert,
        ast.Import,
        ast.ImportFrom,
        ast.Call,
        ast.Attribute,
        ast.Name,
        ast.Constant,
        ast.Tuple,
        ast.List,
        ast.Set,
        ast.Dict,
        ast.IfExp,
        ast.BoolOp,
        ast.UnaryOp,
        ast.BinOp,
        ast.Compare,
        ast.Load,
        ast.Store,
        ast.Del,
        *_OPERATOR_NODES,
    }
)


class InferenceStatus(enum.Enum):
    """Outcome of the family-default inference decision."""

    NOT_APPLICABLE = "not_applicable"
    INFERRED = "inferred"
    UNKNOWN = "unknown"


@dataclass(frozen=True, kw_only=True)
class StateInference:
    """Inference outcome and, when complete, the inferred effects."""

    status: InferenceStatus
    effects: StateContract | None = None


NOT_APPLICABLE = StateInference(status=InferenceStatus.NOT_APPLICABLE)
UNKNOWN = StateInference(status=InferenceStatus.UNKNOWN)
_CACHE: weakref.WeakKeyDictionary[type, StateInference] = weakref.WeakKeyDictionary()
_CACHE_LOCK = threading.RLock()


def clear_state_effect_cache() -> None:
    """Drop the per-class inference cache."""
    with _CACHE_LOCK:
        _CACHE.clear()


@dataclass
class _Facts:
    reads: set[StateKey[object]] = field(default_factory=set)
    writes: set[StateKey[object]] = field(default_factory=set)
    unknown: list[str] = field(default_factory=list)

    def merge(self, other: _Facts) -> None:
        self.reads |= other.reads
        self.writes |= other.writes
        for item in other.unknown:
            if item not in self.unknown:
                self.unknown.append(item)


def _class_module(cls: type) -> str:
    for base in _static_mro(cls):
        module = _class_dict(base).get("__module__")
        if type(module) is str:
            return module
    return ""


def _static_class_attr(cls: type, name: str) -> object:
    for base in _static_mro(cls):
        values = _class_dict(base)
        if name in values:
            return values[name]
    return _UNRESOLVED


def _static_method(cls: type, name: str) -> object:
    raw = _static_class_attr(cls, name)
    if raw is _UNRESOLVED:
        return _UNRESOLVED
    if type(raw) in (staticmethod, classmethod):
        return cast(staticmethod | classmethod, raw).__func__
    return raw


def _static_type_attribute(cls: type, name: str) -> object:
    return _static_class_attr(cls, name)


def _has_descriptor(value: object) -> bool:
    if not _has_supported_metaclass(value):
        return True
    return _static_type_attribute(type(value), "__get__") is not _UNRESOLVED


def _has_custom_instance_dict(value: object) -> bool:
    for base in _static_mro(type(value)):
        descriptor = _class_dict(base).get("__dict__")
        if (
            descriptor is not None
            and type(descriptor) is not types.GetSetDescriptorType
        ):
            return True
    return False


def _instance_dict(value: object) -> dict[str, object]:
    try:
        attributes = object.__getattribute__(value, "__dict__")
    except Exception:
        return {}
    return attributes if type(attributes) is dict else {}


def _is_allowed_value(value: object) -> bool:
    if value is None:
        return True
    kind = type(value)
    if kind in (
        StateKey,
        str,
        int,
        float,
        bool,
        bytes,
        ModuleType,
        FunctionType,
    ):
        return True
    if _is_class_object(value):
        return True
    return issubclass(kind, enum.Enum)


def _is_plain_function(function: object) -> bool:
    if type(function) is not FunctionType:
        return False
    if "__wrapped__" in function.__dict__:
        return False
    if not function.__name__.isidentifier():
        return False
    if (
        inspect.iscoroutinefunction(function)
        or inspect.isasyncgenfunction(function)
        or inspect.isgeneratorfunction(function)
    ):
        return False
    node = _source_tree(function)
    return node is not None and not node.decorator_list


def _static_attribute(base: object, name: str) -> object:
    if _is_class_object(base) and not _has_supported_metaclass(base):
        return _UNRESOLVED
    if type(base) is ModuleType:
        values = vars(base)
        if name not in values:
            return _UNRESOLVED
        value = values[name]
    elif _is_class_object(base):
        value = _static_class_attr(cast(type, base), name)
        if value is _UNRESOLVED:
            return _UNRESOLVED
    else:
        return _UNRESOLVED
    if type(value) is StateKey:
        return value
    if _is_class_object(value) and not _has_supported_metaclass(value):
        return _UNRESOLVED
    if _has_descriptor(value):
        return _UNRESOLVED
    return value if _is_allowed_value(value) else _UNRESOLVED


def _is_saealib_function(value: object, module: str, name: str) -> bool:
    return (
        type(value) is FunctionType
        and value.__module__ == module
        and value.__name__ == name
    )


def _source_tree(
    function: FunctionType,
) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    try:
        source_lines, _ = inspect.getsourcelines(function)
        tree = ast.parse(textwrap.dedent("".join(source_lines)))
    except (OSError, TypeError, SyntaxError, IndentationError, ValueError):
        return None
    node = tree.body[0]
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return None
    return node


def _unsupported_syntax(
    body: ast.FunctionDef | ast.AsyncFunctionDef,
) -> str | None:
    for node in ast.walk(body):
        if node is body:
            continue
        if type(node) not in _ALLOWED_AST_NODES:
            return type(node).__name__
    return None


def _target_names(node: ast.expr) -> set[str]:
    if isinstance(node, ast.Name):
        return {node.id}
    if isinstance(node, (ast.Tuple, ast.List)):
        names: set[str] = set()
        for item in node.elts:
            names.update(_target_names(item))
        return names
    return set()


def _parameter_names(body: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    arguments = body.args
    names = {
        argument.arg
        for argument in (*arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs)
    }
    for extra in (arguments.vararg, arguments.kwarg):
        if extra is not None:
            names.add(extra.arg)
    return names


def _assignment_names(body: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(body):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                names.update(_target_names(target))
        elif isinstance(node, ast.NamedExpr):
            names.update(_target_names(node.target))
        elif isinstance(
            node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)
        ):
            for generator in node.generators:
                names.update(_target_names(generator.target))
    return names


def _non_assignment_bindings(body: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(body):
        if isinstance(node, (ast.For, ast.AsyncFor)):
            names.update(_target_names(node.target))
        elif isinstance(node, (ast.With, ast.AsyncWith)):
            for item in node.items:
                if item.optional_vars is not None:
                    names.update(_target_names(item.optional_vars))
        elif isinstance(node, ast.ExceptHandler) and node.name:
            names.add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                names.add(alias.asname or alias.name.split(".")[0])
    return names


def _bound_names(body: ast.FunctionDef | ast.AsyncFunctionDef) -> frozenset[str]:
    return frozenset(
        _parameter_names(body)
        | _assignment_names(body)
        | _non_assignment_bindings(body)
    )


@dataclass
class _Scope:
    """Names and state values visible inside one analyzed function body."""

    context_names: set[str]
    objects: dict[str, set[StateKey[object]]]
    bound: frozenset[str]
    imports: dict[str, object]
    globals: dict[str, object]


def _method_owner(cls: type, name: str, function: FunctionType) -> type | None:
    for base in _static_mro(cls):
        candidate = _class_dict(base).get(name)
        if candidate is function:
            return base
        if (
            type(candidate) in (staticmethod, classmethod)
            and cast(staticmethod | classmethod, candidate).__func__ is function
        ):
            return base
    return None


def _implementation_owner(cls: type, implementation: object) -> type | None:
    for owner in _static_mro(cls):
        if _class_dict(owner).get("contract") is implementation:
            return owner
    return None


def _has_instance_contract(component: object) -> bool:
    attributes = _instance_dict(component)
    return any(
        name in attributes
        for name in ("contract", "analysis_contract", "runtime_contract")
    )


def _has_instance_method_override(component: object) -> bool:
    attributes = _instance_dict(component)
    method_names: set[str] = set()
    for base in _static_mro(type(component)):
        for name, raw in _class_dict(base).items():
            if type(raw) in (staticmethod, classmethod, FunctionType):
                method_names.add(name)
    return any(
        name in method_names and callable(value) for name, value in attributes.items()
    )


def _has_dynamic_attribute_hook(component: object) -> bool:
    for base in _static_mro(type(component)):
        if base is object:
            continue
        values = _class_dict(base)
        if "__getattr__" in values or "__getattribute__" in values:
            return True
    return False


def _is_class_object(value: object) -> bool:
    return issubclass(type(value), type)


def _has_supported_metaclass(value: object) -> bool:
    if _is_class_object(value):
        return type(value) in (type, abc.ABCMeta)
    return type(type(value)) in (type, abc.ABCMeta)


def _is_component(value: object) -> bool:
    contract = _static_method(type(value), "contract")
    return contract is not _UNRESOLVED and callable(contract)


def _unwrap_stage(component: object) -> object:
    """Return the Stage behind a framework graph wrapper, when there is one."""
    stage = _instance_dict(component).get("stage")
    if stage is None or stage is component:
        return component
    if not _class_module(type(component)).startswith("saealib."):
        return component
    return stage if _is_component(stage) else component


def _resolved_contract_implementation(component: object) -> object:
    for name in ("analysis_contract", "runtime_contract", "contract"):
        candidate = _static_class_attr(type(component), name)
        if candidate is not _UNRESOLVED:
            return candidate
    return _UNRESOLVED


def _is_family_default(component: object, implementation: object) -> bool:
    owner = _implementation_owner(type(component), implementation)
    if (
        owner is None
        or not _class_module(owner).startswith("saealib.")
        or not _class_dict(owner).get("__abstractmethods__")
    ):
        return False
    if (
        type(implementation) is not FunctionType
        or not implementation.__module__.startswith("saealib.")
        or not _is_plain_function(implementation)
    ):
        return False
    try:
        contract = implementation(component)
    except Exception:
        return False
    if type(contract) is not ComponentContract:
        return False
    state = contract.state
    return not state.reads and not state.writes and not state.exports


def _state_contract(facts: _Facts) -> StateContract:
    def ordered(keys: Iterable[StateKey[object]]) -> tuple[StateKey[object], ...]:
        return tuple(
            sorted(
                set(keys),
                key=lambda key: (key.namespace, key.name, key.schema_version),
            )
        )

    return StateContract(reads=ordered(facts.reads), writes=ordered(facts.writes))


def _has_static_instance_dict(value: object) -> bool:
    return any("__dict__" in _class_dict(base) for base in _static_mro(type(value)))


def _class_declaration_status(value: object) -> bool | None:
    if not _is_component(value):
        return True
    implementation = _resolved_contract_implementation(value)
    if implementation is _UNRESOLVED:
        return True
    return not _is_family_default(value, implementation)


def _declaration_established(component: object) -> bool | None:
    if _has_custom_instance_dict(component):
        return _class_declaration_status(component)
    attributes = _instance_dict(component)
    if any(
        name in attributes
        for name in ("contract", "analysis_contract", "runtime_contract")
    ):
        return True
    target = _unwrap_stage(component)
    if target is not component:
        if _has_custom_instance_dict(target):
            return _class_declaration_status(target)
        attributes = _instance_dict(target)
        if any(
            name in attributes
            for name in ("contract", "analysis_contract", "runtime_contract")
        ):
            return True
    return _class_declaration_status(target)


def infer_state_effects(component: object) -> StateInference:
    """Return the inference outcome for a component without a declaration.

    ``NOT_APPLICABLE`` means that a declaration is established: the component
    keeps its own contract.  ``INFERRED`` carries the completed effects.
    ``UNKNOWN`` means that the component is undeclared but its source or
    runtime shape cannot be analyzed completely; verification treats it as
    non-enumerable.
    """
    try:
        declared = _declaration_established(component)
    except Exception:
        return UNKNOWN
    if declared is True:
        return NOT_APPLICABLE
    if declared is None:
        return UNKNOWN
    try:
        if not _has_supported_metaclass(component) or _has_custom_instance_dict(
            component
        ):
            return UNKNOWN
        target = _unwrap_stage(component)
        if not _has_supported_metaclass(target) or _has_custom_instance_dict(target):
            return UNKNOWN
        if not _has_static_instance_dict(target) or "__slots__" in _class_dict(
            type(target)
        ):
            return UNKNOWN
        if _has_instance_contract(component) or _has_instance_contract(target):
            return UNKNOWN
        if _has_instance_method_override(target) or _has_dynamic_attribute_hook(target):
            return UNKNOWN
        cls = type(target)
        with _CACHE_LOCK:
            if cls in _CACHE:
                return _CACHE[cls]
            facts = _Analyzer(cls).analyze_class()
            result = (
                UNKNOWN
                if facts.unknown
                else StateInference(
                    status=InferenceStatus.INFERRED,
                    effects=_state_contract(facts),
                )
            )
            _CACHE[cls] = result
            return result
    except Exception:
        return UNKNOWN


class _Analyzer:
    def __init__(self, cls: type) -> None:
        self.cls = cls
        self._facts_by_function: dict[
            tuple[object, frozenset[str], type | None], _Facts
        ] = {}
        self._in_progress: set[tuple[object, frozenset[str], type | None]] = set()

    def analyze_class(self) -> _Facts:
        facts = _Facts()
        method_names: list[str] = []
        seen: set[str] = set()
        for owner in _static_mro(self.cls):
            if owner is not self.cls and _class_module(owner) in (
                "builtins",
                "abc",
                "typing",
                "typing_extensions",
            ):
                continue
            for name, raw in _class_dict(owner).items():
                if name in _SKIP_METHODS or name in seen:
                    continue
                function = (
                    cast(staticmethod | classmethod, raw).__func__
                    if type(raw) in (staticmethod, classmethod)
                    else raw
                )
                if type(function) is not FunctionType:
                    continue
                seen.add(name)
                method_names.append(name)
        for name in method_names:
            raw = _static_class_attr(self.cls, name)
            if raw is _UNRESOLVED:
                continue
            if type(raw) in (staticmethod, classmethod):
                function = cast(staticmethod | classmethod, raw).__func__
                if type(function) is not FunctionType:
                    facts.unknown.append(f"unanalyzable method {name!r}")
                elif self._context_parameters(function):
                    facts.unknown.append(f"unmodeled method binding {name!r}")
                continue
            if type(raw) is not FunctionType:
                facts.unknown.append(f"unanalyzable method {name!r}")
                continue
            if not _is_plain_function(raw):
                facts.unknown.append(f"unanalyzable method {name!r}")
                continue
            owner = _method_owner(self.cls, name, raw)
            parameters = self._context_parameters(raw)
            if not parameters:
                base_parameters = self._context_parameters_from_base(
                    owner or self.cls, name, raw
                )
                if base_parameters is None:
                    facts.unknown.append(
                        f"cannot determine the context parameter of {name!r}"
                    )
                    continue
                parameters = base_parameters
            if parameters:
                facts.merge(
                    self._analyze_function(
                        raw,
                        frozenset(parameters),
                        owner=owner,
                    )
                )
        return facts

    def _context_parameters(self, function: FunctionType) -> set[str]:
        parameters: set[str] = set()
        try:
            signature = inspect.signature(function)
        except (TypeError, ValueError):
            return parameters
        for parameter in signature.parameters.values():
            if parameter.name in ("self", "cls"):
                continue
            if parameter.name in _CONTEXT_PARAMETER_NAMES:
                parameters.add(parameter.name)
                continue
            annotation = parameter.annotation
            if annotation is inspect.Parameter.empty:
                continue
            text = annotation if isinstance(annotation, str) else str(annotation)
            if any(name in text for name in _CONTEXT_TYPE_NAMES):
                parameters.add(parameter.name)
        return parameters

    def _context_parameters_from_base(
        self,
        owner: type,
        name: str,
        function: FunctionType,
    ) -> set[str] | None:
        try:
            parameters = [
                parameter
                for parameter in inspect.signature(function).parameters.values()
                if parameter.name not in ("self", "cls")
            ]
        except (TypeError, ValueError):
            return None
        for base in _static_mro(owner)[1:]:
            raw = _class_dict(base).get(name, _UNRESOLVED)
            if raw is _UNRESOLVED:
                continue
            candidate = (
                cast(staticmethod | classmethod, raw).__func__
                if type(raw) in (staticmethod, classmethod)
                else raw
            )
            if candidate is function:
                continue
            if type(candidate) is not FunctionType:
                return None
            try:
                base_parameters = [
                    parameter
                    for parameter in inspect.signature(candidate).parameters.values()
                    if parameter.name not in ("self", "cls")
                ]
            except (TypeError, ValueError):
                return None
            if not any(
                parameter.name in _CONTEXT_PARAMETER_NAMES
                or any(
                    marker in str(parameter.annotation)
                    for marker in _CONTEXT_TYPE_NAMES
                )
                for parameter in base_parameters
            ):
                continue
            if type(raw) in (staticmethod, classmethod):
                return None
            if type(raw) is not FunctionType or not _is_plain_function(raw):
                return None
            found: set[str] = set()
            for index, parameter in enumerate(base_parameters):
                if parameter.name not in _CONTEXT_PARAMETER_NAMES and not any(
                    marker in str(parameter.annotation)
                    for marker in _CONTEXT_TYPE_NAMES
                ):
                    continue
                if index >= len(parameters):
                    continue
                override = parameters[index]
                if override.name == parameter.name or override.kind is parameter.kind:
                    found.add(override.name)
            if found:
                return found
        return set()

    def _analyze_function(
        self,
        function: FunctionType,
        context_parameters: frozenset[str],
        *,
        owner: type | None,
    ) -> _Facts:
        key = (function.__code__, context_parameters, owner)
        if key in self._in_progress:
            facts = _Facts()
            facts.unknown.append("recursive call")
            return facts
        cached = self._facts_by_function.get(key)
        if cached is not None:
            return cached
        facts = _Facts()
        self._facts_by_function[key] = facts
        self._in_progress.add(key)
        try:
            self._visit_function(function, context_parameters, owner, facts)
        except Exception:
            facts.unknown.append("analysis failed")
        finally:
            self._in_progress.discard(key)
        return facts

    def _visit_function(
        self,
        function: FunctionType,
        context_parameters: frozenset[str],
        owner: type | None,
        facts: _Facts,
    ) -> None:
        body = _source_tree(function)
        if body is None:
            facts.unknown.append("source unavailable")
            return
        if body.decorator_list:
            facts.unknown.append("decorated function")
            return
        if isinstance(body, ast.AsyncFunctionDef):
            facts.unknown.append("async function")
            return
        unsupported = _unsupported_syntax(body)
        if unsupported is not None:
            facts.unknown.append(f"unsupported syntax {unsupported}")
            return
        scope = _Scope(
            context_names=set(context_parameters),
            objects={},
            bound=_bound_names(body),
            imports={},
            globals=dict(function.__globals__),
        )
        self._add_local_imports(body, scope)
        escape = self._collect_aliases(body, scope)
        if escape is not None:
            facts.unknown.append(escape)
            return
        if not self._aliases_are_stable(body, scope):
            facts.unknown.append("rebinds a context name")
            return
        consumed_calls: set[int] = set()
        for node in ast.walk(body):
            if isinstance(node, ast.Attribute) and self._is_context(node.value, scope):
                self._visit_context_attribute(node, facts)
            if isinstance(node, ast.Call) and self._visit_call(
                node, scope, facts, owner
            ):
                consumed_calls.add(id(node))
        unconsumed = self._unconsumed_context_use(body, scope, consumed_calls)
        if unconsumed is not None:
            facts.unknown.append(unconsumed)

    def _collect_aliases(
        self, body: ast.FunctionDef | ast.AsyncFunctionDef, scope: _Scope
    ) -> str | None:
        assignments = [
            node
            for node in ast.walk(body)
            if isinstance(node, (ast.Assign, ast.AnnAssign))
        ]
        changed = True
        while changed:
            changed = False
            for node in assignments:
                targets = (
                    node.targets if isinstance(node, ast.Assign) else [node.target]
                )
                value = node.value
                if value is None:
                    continue
                if self._is_context(value, scope):
                    if len(targets) != 1 or not isinstance(targets[0], ast.Name):
                        return "assigns the context to a target other than one name"
                    target_name = targets[0].id
                    if target_name not in scope.context_names:
                        scope.context_names.add(target_name)
                        changed = True
                    continue
                if isinstance(value, (ast.List, ast.Tuple, ast.Set, ast.Dict)):
                    holders = (
                        [*value.keys, *value.values]
                        if isinstance(value, ast.Dict)
                        else list(value.elts)
                    )
                    if any(
                        holder is not None and self._object_keys(holder, scope)
                        for holder in holders
                    ):
                        return "stores a state value in a container"
                    continue
                keys = self._object_keys(value, scope)
                if not keys:
                    continue
                if len(targets) != 1 or not isinstance(targets[0], ast.Name):
                    return "assigns a state value to a target other than one name"
                target_name = targets[0].id
                if not keys <= scope.objects.get(target_name, set()):
                    scope.objects.setdefault(target_name, set()).update(keys)
                    changed = True
        return None

    def _aliases_are_stable(
        self, body: ast.FunctionDef | ast.AsyncFunctionDef, scope: _Scope
    ) -> bool:
        for node in ast.walk(body):
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                continue
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if not any(
                _target_names(target) & scope.context_names for target in targets
            ):
                continue
            value = node.value
            if value is None or not self._is_context(value, scope):
                return False
        return not _non_assignment_bindings(body) & scope.context_names

    def _unconsumed_context_use(
        self,
        body: ast.FunctionDef | ast.AsyncFunctionDef,
        scope: _Scope,
        consumed_calls: set[int],
    ) -> str | None:
        parents: dict[ast.AST, ast.AST] = {}
        for node in ast.walk(body):
            for child in ast.iter_child_nodes(node):
                parents[child] = node
        for node in ast.walk(body):
            if not isinstance(node, ast.expr) or not self._is_context(node, scope):
                continue
            if isinstance(node, ast.Name) and not isinstance(node.ctx, ast.Load):
                continue
            parent = parents.get(node)
            if isinstance(parent, ast.expr) and self._is_context(parent, scope):
                continue
            if self._context_use_is_modeled(node, parent, parents, consumed_calls):
                continue
            position = type(parent).__name__ if parent is not None else "nowhere"
            return f"uses the context in {position}"
        return None

    def _context_use_is_modeled(
        self,
        node: ast.expr,
        parent: ast.AST | None,
        parents: dict[ast.AST, ast.AST],
        consumed_calls: set[int],
    ) -> bool:
        if isinstance(parent, ast.Attribute) and parent.value is node:
            if parent.attr in _FIELD_KEYS or parent.attr in _CONTEXT_DATA_ATTRIBUTES:
                return True
            if parent.attr in _CONTEXT_METHOD_ATTRIBUTES:
                holder = parents.get(parent)
                return isinstance(holder, ast.Call) and holder.func is parent
            return False
        if isinstance(parent, ast.keyword) and parent.value is node:
            holder = parents.get(parent)
            return isinstance(holder, ast.Call) and id(holder) in consumed_calls
        if isinstance(parent, ast.Call):
            return parent.func is not node and id(parent) in consumed_calls
        if isinstance(parent, (ast.Assign, ast.AnnAssign)) and parent.value is node:
            targets = (
                parent.targets if isinstance(parent, ast.Assign) else [parent.target]
            )
            return len(targets) == 1 and isinstance(targets[0], ast.Name)
        return isinstance(parent, ast.Return) and parent.value is node

    def _value(self, name: str, scope: _Scope) -> object:
        if name in scope.imports:
            return scope.imports[name]
        if name in scope.bound:
            return _UNRESOLVED
        value = scope.globals.get(name, _UNRESOLVED)
        if value is not _UNRESOLVED:
            return value
        return vars(builtins).get(name, _UNRESOLVED)

    def _is_context(self, node: ast.expr, scope: _Scope) -> bool:
        if isinstance(node, ast.Name):
            return node.id in scope.context_names
        if isinstance(node, ast.Attribute) and node.attr == "context":
            return self._is_context(node.value, scope)
        if isinstance(node, ast.Call):
            function = node.func
            if self._is_alias_function(function, scope):
                return any(
                    self._is_context(argument, scope)
                    for argument in (
                        *node.args,
                        *(keyword.value for keyword in node.keywords),
                    )
                )
            if isinstance(function, ast.Attribute) and function.attr in _ALIAS_METHODS:
                return self._is_context(function.value, scope)
        return False

    def _is_alias_function(self, function: ast.expr, scope: _Scope) -> bool:
        if isinstance(function, ast.Name):
            value = self._value(function.id, scope)
            if function.id == "cast":
                return value is typing.cast
            if function.id == "algorithm_context":
                return _is_saealib_function(
                    value, "saealib.algorithms.base", "algorithm_context"
                )
            return False
        if isinstance(function, ast.Attribute) and function.attr in _ALIAS_FUNCTIONS:
            if not isinstance(function.value, ast.Name):
                return False
            base = self._value(function.value.id, scope)
            if base is _UNRESOLVED:
                return False
            if function.attr == "cast":
                return base is typing
            value = _static_attribute(base, "algorithm_context")
            return _is_saealib_function(
                value, "saealib.algorithms.base", "algorithm_context"
            )
        return False

    def _add_local_imports(
        self, body: ast.FunctionDef | ast.AsyncFunctionDef, scope: _Scope
    ) -> None:
        for node in ast.walk(body):
            if isinstance(node, ast.ImportFrom):
                module = sys.modules.get(node.module) if node.module else None
                if module is None:
                    continue
                values = vars(module)
                for alias in node.names:
                    value = values.get(alias.name, _UNRESOLVED)
                    if value is not _UNRESOLVED:
                        scope.imports[alias.asname or alias.name] = value
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    module = sys.modules.get(alias.name)
                    if module is not None:
                        scope.imports[alias.asname or alias.name.split(".")[0]] = module

    def _visit_context_attribute(self, node: ast.Attribute, facts: _Facts) -> None:
        if isinstance(node.ctx, ast.Store):
            keys = _FIELD_KEYS.get(node.attr)
            if keys is None:
                facts.unknown.append(f"assigns context attribute {node.attr!r}")
            else:
                facts.writes.update(keys)
            return
        keys = _FIELD_KEYS.get(node.attr)
        if keys is not None:
            facts.reads.update(keys)
            facts.writes.update(key for key in keys if key in _STATEFUL_KEYS)
            return
        if node.attr in _NON_STATE_ATTRIBUTES:
            return
        facts.unknown.append(f"reads unrecognized context attribute {node.attr!r}")

    def _visit_call(
        self,
        node: ast.Call,
        scope: _Scope,
        facts: _Facts,
        owner: type | None,
    ) -> bool:
        function = node.func
        context_arguments = any(
            self._is_context(argument, scope)
            for argument in (
                *node.args,
                *(keyword.value for keyword in node.keywords),
            )
        )
        if isinstance(function, ast.Attribute) and self._is_context(
            function.value, scope
        ):
            if function.attr == "compiled_service":
                return not context_arguments
            first = node.args[0] if node.args else None
            if function.attr in ("get", "get_state", "contains"):
                key = self._resolve_key(first, scope) if first is not None else None
                if key is None:
                    facts.unknown.append(f"unresolved {function.attr}() key")
                else:
                    facts.reads.add(key)
                return not context_arguments
            if function.attr == "set_state":
                key = self._resolve_key(first, scope) if first is not None else None
                if key is None:
                    facts.unknown.append("unresolved set_state() key")
                else:
                    facts.writes.add(key)
                return not context_arguments
            if function.attr == "replace":
                if node.args:
                    facts.unknown.append("positional replace() arguments")
                for keyword in node.keywords:
                    if keyword.arg is None:
                        facts.unknown.append("unpacked replace() arguments")
                    elif keyword.arg in _FIELD_KEYS:
                        facts.writes.update(_FIELD_KEYS[keyword.arg])
                    else:
                        facts.unknown.append(
                            f"replace({keyword.arg}=...) is not a state field"
                        )
                return not context_arguments

        if isinstance(function, ast.Attribute) and function.attr in _MUTATOR_NAMES:
            keys = self._object_keys(function.value, scope)
            if keys:
                facts.writes.update(keys)

        if isinstance(function, ast.Name) and function.id == StatePatch.__name__:
            value = self._value(function.id, scope)
            if value is StatePatch:
                self._visit_state_patch(node, facts, scope)
            else:
                facts.unknown.append("unresolved StatePatch reference")

        if (
            isinstance(function, ast.Name)
            and self._value(function.id, scope) is builtins.getattr
            and node.args
            and self._is_context(node.args[0], scope)
        ):
            facts.unknown.append("getattr on the context")
            return False

        if not context_arguments:
            return True
        if self._is_alias_function(function, scope):
            return True
        target = self._resolve_callee(function, scope, owner)
        if isinstance(target, tuple) and target[0] == "bound":
            target = target[1]
            offset = 1
        else:
            offset = 0
        if not _is_plain_function(target):
            facts.unknown.append(f"passes the context to {ast.unparse(function)}")
            return False
        target_function = cast(FunctionType, target)
        try:
            parameters = list(inspect.signature(target_function).parameters.values())
        except (TypeError, ValueError):
            facts.unknown.append(f"cannot inspect {ast.unparse(function)}")
            return False
        if any(
            parameter.kind is inspect.Parameter.VAR_POSITIONAL
            for parameter in parameters
        ):
            facts.unknown.append(
                f"passes the context through *args of {ast.unparse(function)}"
            )
            return False
        parameter_names = {parameter.name for parameter in parameters}
        next_parameters: set[str] = set()
        for position, argument in enumerate(node.args):
            if not self._is_context(argument, scope):
                continue
            index = position + offset
            if index >= len(parameters):
                facts.unknown.append(f"passes the context to {ast.unparse(function)}")
                return False
            next_parameters.add(parameters[index].name)
        for keyword in node.keywords:
            if not self._is_context(keyword.value, scope):
                continue
            if keyword.arg is None or keyword.arg not in parameter_names:
                facts.unknown.append(f"passes the context to {ast.unparse(function)}")
                return False
            next_parameters.add(keyword.arg)
        if not next_parameters:
            return True
        facts.merge(
            self._analyze_function(
                target_function,
                frozenset(next_parameters),
                owner=self._target_owner(target_function, owner),
            )
        )
        return True

    def _visit_state_patch(self, node: ast.Call, facts: _Facts, scope: _Scope) -> None:
        if node.args:
            facts.unknown.append("positional StatePatch arguments")
        for keyword in node.keywords:
            if keyword.arg is None:
                facts.unknown.append("unpacked StatePatch arguments")
                continue
            if keyword.arg not in ("writes", "deletes"):
                facts.unknown.append(f"StatePatch({keyword.arg}=...)")
                continue
            value = keyword.value
            if isinstance(value, ast.Dict):
                for key_node in value.keys:
                    if key_node is None:
                        facts.unknown.append("unpacked StatePatch keys")
                        continue
                    key = self._resolve_key(key_node, scope)
                    if key is None:
                        facts.unknown.append("unresolved StatePatch key")
                    else:
                        facts.writes.add(key)
            elif isinstance(value, (ast.Set, ast.Tuple, ast.List)):
                for item in value.elts:
                    key = self._resolve_key(item, scope)
                    if key is None:
                        facts.unknown.append("unresolved StatePatch key")
                    else:
                        facts.writes.add(key)
            else:
                facts.unknown.append("non-literal StatePatch keys")

    def _target_owner(self, target: FunctionType, owner: type | None) -> type | None:
        for candidate in _static_mro(self.cls):
            if _class_dict(candidate).get(target.__name__) is target:
                return candidate
        return owner

    def _resolve_callee(
        self,
        function: ast.expr,
        scope: _Scope,
        owner: type | None,
    ) -> object:
        if isinstance(function, ast.Name):
            value = self._value(function.id, scope)
            if value is _UNRESOLVED:
                return None
            return value if _is_plain_function(value) else None
        if not isinstance(function, ast.Attribute):
            return None
        base = function.value
        if isinstance(base, ast.Name) and base.id in ("self", "cls"):
            raw = _static_class_attr(self.cls, function.attr)
            if raw is _UNRESOLVED or type(raw) in (staticmethod, classmethod):
                return None
            return ("bound", raw) if _is_plain_function(raw) else None
        if (
            isinstance(base, ast.Call)
            and isinstance(base.func, ast.Name)
            and self._value(base.func.id, scope) is builtins.super
        ):
            mro = _static_mro(self.cls)
            start = mro.index(owner) + 1 if owner in mro else 0
            for candidate in mro[start:]:
                raw = _class_dict(candidate).get(function.attr)
                if raw is None:
                    continue
                if type(raw) in (staticmethod, classmethod):
                    return None
                return ("bound", raw) if _is_plain_function(raw) else None
            return None
        return None

    def _object_keys(
        self,
        node: ast.expr,
        scope: _Scope,
    ) -> set[StateKey[object]]:
        if isinstance(node, ast.Name):
            return set(scope.objects.get(node.id, set()))
        if isinstance(node, ast.Attribute) and self._is_context(node.value, scope):
            return set(_FIELD_KEYS.get(node.attr, ()))
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in ("get", "get_state")
            and self._is_context(node.func.value, scope)
            and node.args
        ):
            key = self._resolve_key(node.args[0], scope)
            return {key} if key is not None else set()
        return set()

    def _resolve_key(self, node: ast.expr, scope: _Scope) -> StateKey[object] | None:
        value = self._resolve_expression(node, scope)
        return cast(StateKey[object], value) if type(value) is StateKey else None

    def _resolve_expression(self, node: ast.expr, scope: _Scope) -> object:
        if isinstance(node, ast.Constant):
            return node.value if _is_allowed_value(node.value) else _UNRESOLVED
        if isinstance(node, ast.Name):
            value = self._value(node.id, scope)
            return value if _is_allowed_value(value) else _UNRESOLVED
        if isinstance(node, ast.Attribute):
            base = self._resolve_expression(node.value, scope)
            if base is _UNRESOLVED:
                return _UNRESOLVED
            return _static_attribute(base, node.attr)
        return _UNRESOLVED
