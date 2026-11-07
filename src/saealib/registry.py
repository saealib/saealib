"""Registry: name -> class resolution and recursive construction from specs.

A "spec" is one of:

- a live instance, returned unchanged
- a string ``"Name"``, resolved via the registry (or as a dotted import path
  ``"module.submodule.ClassName"`` if not registered) and instantiated with
  no arguments
- a mapping ``{"type": "Name", "params": {...}}``, resolved the same way and
  instantiated with ``params``. If ``params`` is a dict, it is passed as
  keyword arguments (``cls(**params)``); if it is a list, as positional
  arguments (``cls(*params)``), for constructors taking ``*args``. Values
  that are themselves specs are built recursively first.
- a mapping ``{"callable": "module.submodule.function_name"}``, resolved via
  the registry (or dotted import path) to the function/builtin itself,
  without calling it.

``to_spec()`` is the reverse operation: it serializes a live instance back
into a spec by reflecting its constructor signature and reading same-named
attributes. Classes that inherit ``object.__init__`` have no constructor
parameters and serialize with empty ``params``. A ``*args`` parameter is
read from the same-named attribute; a ``**kwargs`` parameter's same-named
mapping attribute is merged into ``params`` key by key. A class opts out of
this generic reflection by exposing a ``_registry_spec`` attribute (a spec,
or ``None`` if it cannot be serialized) which ``to_spec()`` uses directly
instead.

Custom components are registered the same way as builtins, which lets them be
referenced by a short name from preset YAML files just like ``GA`` or
``RBFSurrogate``:

>>> from saealib import register
>>> from saealib.surrogate.base import Surrogate
>>>
>>> @register()
... class MyCustomSurrogate(Surrogate):
...     ...

Once registered, ``"MyCustomSurrogate"`` (or ``{"type": "MyCustomSurrogate",
"params": {...}}``) resolves via :func:`get`/:func:`build` exactly like a
bundled component, including inside a preset passed to
:meth:`~saealib.optimizer.Optimizer.set_preset`.
"""

from __future__ import annotations

import importlib
import inspect
from collections.abc import Callable, Mapping
from typing import Any, Protocol, TypeVar

import numpy as np

from saealib.exceptions import ValidationError

_REGISTRY: dict[str, Any] = {}


class _Named(Protocol):
    __name__: str


T = TypeVar("T", bound=_Named)


def register(name: str | None = None) -> Callable[[T], T]:
    """Register a class or function under ``name`` (default ``cls.__name__``).

    Use this to make a custom ``Algorithm``/``Surrogate``/etc. subclass
    resolvable by name from a preset YAML file, the same way bundled
    components are:

    >>> from saealib import register
    >>> from saealib.surrogate.base import Surrogate
    >>>
    >>> @register()
    ... class MyCustomSurrogate(Surrogate):
    ...     ...

    ``MyCustomSurrogate`` can now be referenced as ``"MyCustomSurrogate"`` or
    ``{"type": "MyCustomSurrogate", "params": {...}}`` anywhere a spec is
    accepted, e.g. in a preset consumed by
    :meth:`~saealib.optimizer.Optimizer.set_preset`.
    """

    def _decorator(cls: T) -> T:
        _REGISTRY[name or cls.__name__] = cls
        return cls

    return _decorator


def get(name: str) -> Any:
    """Resolve ``name`` to a class/function via the registry, or a dotted path."""
    if name in _REGISTRY:
        return _REGISTRY[name]
    if "." in name:
        module_path, _, cls_name = name.rpartition(".")
        try:
            module = importlib.import_module(module_path)
            return getattr(module, cls_name)
        except (ImportError, AttributeError) as e:
            raise ValidationError(f"Cannot resolve {name!r}: {e}") from e
    raise ValidationError(
        f"Unknown name: {name!r}. Registered names: {sorted(_REGISTRY)}"
    )


def _is_spec(value: Any) -> bool:
    return isinstance(value, dict) and "type" in value


def _is_callable_ref(value: Any) -> bool:
    return isinstance(value, dict) and "callable" in value and "type" not in value


def _build_value(value: Any) -> Any:
    if _is_callable_ref(value) or _is_spec(value):
        return build(value)
    if isinstance(value, list):
        return [_build_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_build_value(item) for item in value)
    if isinstance(value, dict):
        return {key: _build_value(item) for key, item in value.items()}
    return value


def _restore_metadata(result: Any, metadata: Any, type_name: str) -> Any:
    if metadata is None:
        return result
    for name, value in metadata.items():
        try:
            setattr(result, name, value)
        except Exception as exc:
            raise ValidationError(
                f"Cannot restore {name!r} metadata for {type_name!r}: {exc}"
            ) from exc
    return result


def build(spec: Any) -> Any:
    """Recursively construct an object from a spec; non-spec values pass through.

    A ``{"callable": "dotted.path"}`` node resolves to the referenced
    function/builtin itself, without calling it.
    """
    if _is_callable_ref(spec):
        return get(spec["callable"])
    if isinstance(spec, str):
        return get(spec)()
    if _is_spec(spec):
        cls = get(spec["type"])
        params = spec.get("params", {})
        metadata = spec.get("_registry_stage_metadata")
        if isinstance(params, list):
            args = [_build_value(v) for v in params]
            try:
                result = cls(*args)
            except TypeError as e:
                raise ValidationError(f"Cannot construct {spec['type']!r}: {e}") from e
            return _restore_metadata(result, metadata, spec["type"])
        built_params = {k: _build_value(v) for k, v in params.items()}
        try:
            result = cls(**built_params)
        except TypeError as e:
            raise ValidationError(f"Cannot construct {spec['type']!r}: {e}") from e
        return _restore_metadata(result, metadata, spec["type"])
    if isinstance(spec, (list, tuple, dict)):
        return _build_value(spec)
    return spec


def _dotted_path(obj: Any) -> str:
    return f"{obj.__module__}.{obj.__qualname__}"


def _find_registered_name(cls: Any) -> str | None:
    for name, registered_cls in _REGISTRY.items():
        if registered_cls is cls:
            return name
    return None


def _path_text(path: tuple[object, ...], root: str = "value") -> str:
    return root + "".join(f"[{part!r}]" for part in path)


def _validate_importable_type(
    cls: type, path: tuple[object, ...], root: str = "value"
) -> None:
    if _find_registered_name(cls) is not None:
        return
    type_path = _dotted_path(cls)
    try:
        importable = get(type_path) is cls
    except ValidationError:
        importable = False
    if not importable:
        raise ValidationError(
            f"Cannot serialize {_path_text(path, root)}: "
            f"type {type_path!r} is not importable"
        )


_MISSING = object()
_OMIT = object()

_BUILTIN_STAGE_FIELDS = {
    "AskStage": {
        "algorithm": "_algorithm",
        "n_offspring": "_n_offspring",
        "cbmanager": ("_proxy", "_cbmanager"),
    },
    "SurrogatePredictStage": {
        "surrogate_manager": "_sm",
        "refit": "_refit",
        "cbmanager": "_cbmanager",
    },
    "AcquisitionStage": {
        "acquisition": "_acquisition",
        "cbmanager": "_cbmanager",
    },
    "SurrogateFitStage": {
        "surrogate_manager": "_sm",
        "cbmanager": "_cbmanager",
    },
    "TopKSelectionStage": {"k": "_k"},
    "EvaluationPlanStage": {"planner": "_planner", "n_eval": "_n_eval"},
    "PendingEvaluationContextStage": {"scheduler": "_scheduler"},
    "AsyncEvaluationSubmitStage": {
        "scheduler": "_scheduler",
        "planner": "_planner",
        "feedback_builder": "_feedback_builder",
        "algorithm": "_algorithm",
        "callback_manager": "_callback_manager",
    },
    "EvaluationSubmitStage": {"evaluator": "_evaluator"},
    "EvaluationCollectStage": {"evaluator": "_evaluator"},
    "EvaluationAcknowledgeStage": {
        "evaluator": "_evaluator",
        "cbmanager": "_cbmanager",
    },
    "TrueEvaluationStage": {
        "evaluator": "_evaluator",
        "n_eval": "_n_eval",
        "cbmanager": "_cbmanager",
    },
    "FeedbackStage": {"builder": "_builder"},
    "TellStage": {"algorithm": "_algorithm", "channel": "_channel"},
    "SurrogateOnlyLoopStage": {
        "algorithm": "_algorithm",
        "surrogate_manager": "_sm",
        "gen_ctrl": "_gen_ctrl",
        "acquisition": "_acquisition",
        "feedback_builder": "_builder",
        "cbmanager": "_cbmanager",
    },
    "InitializationStage": {
        "initializer": "_initializer",
        "provider": "_provider",
        "problem": "_problem",
    },
}


def _builtin_stage_fields(stage: Any) -> dict[str, Any]:
    for base in type(stage).__mro__:
        if base.__module__ == "saealib.stages" and type(
            stage
        ).__init__ is base.__dict__.get("__init__"):
            return _BUILTIN_STAGE_FIELDS.get(base.__name__, {})
    return {}


def _constructor_value(
    obj: Any,
    name: str,
    parameter_names: set[str],
    stage_fields: Mapping[str, Any],
    source_declaration: bool,
    *,
    allow_private: bool,
) -> Any:
    from saealib.pipeline import Branch, Loop, Pipeline, Repeat, Stage

    if isinstance(obj, Pipeline):
        if name == "steps" and "stages" in parameter_names:
            return _OMIT
        structural = {
            "stages": obj.stages,
            "steps": obj.steps,
            "name": obj.name,
            "label": obj.label,
            "notation": obj.notation,
        }
        if name in structural:
            return structural[name]
    elif isinstance(obj, Repeat):
        structural = {
            "body": obj.body,
            "count": obj.count,
            "name": obj.name,
            "label": obj.label,
            "notation": obj.notation,
        }
        if name in structural:
            return structural[name]
    elif isinstance(obj, Loop):
        structural = {
            "body": obj.body,
            "until": obj.until,
            "name": obj.name,
            "label": obj.label,
            "notation": obj.notation,
        }
        if name in structural:
            return structural[name]
    elif isinstance(obj, Branch):
        structural = {
            "condition": obj.condition,
            "then": obj.then,
            "else_": obj.else_,
            "name": obj.name,
            "label": obj.label,
            "notation": obj.notation,
        }
        if name in structural:
            return structural[name]

    if source_declaration and name == "pipeline":
        return obj.declaration

    alias = stage_fields.get(name)
    if alias is not None:
        instance_values = getattr(obj, "__dict__", {})
        refs = dict(instance_values.get("_refs", ()))
        if alias in refs and refs[alias] is not None:
            return refs[alias]
        if isinstance(alias, tuple):
            parent = instance_values.get(alias[0], _MISSING)
            value = getattr(parent, alias[1], _MISSING)
        else:
            value = instance_values.get(alias, _MISSING)
            if value is _MISSING:
                value = getattr(obj, alias, _MISSING)
        if value is not _MISSING:
            return value

    try:
        return getattr(obj, name)
    except AttributeError:
        pass
    if allow_private:
        instance_values = getattr(obj, "__dict__", {})
        private_name = f"_{name}"
        refs = dict(instance_values.get("_refs", ()))
        if (
            instance_values.get(private_name, _MISSING) is None
            and refs.get(private_name) is not None
        ):
            return _MISSING
        if isinstance(obj, Stage) and not stage_fields:
            parent_fields = {
                private_name
                for base in type(obj).__mro__
                if base.__module__ == "saealib.stages"
                for alias in _BUILTIN_STAGE_FIELDS.get(base.__name__, {}).values()
                for private_name in (alias if isinstance(alias, tuple) else (alias,))
            }
            if private_name in parent_fields:
                return _MISSING
        return instance_values.get(private_name, _MISSING)
    return _MISSING


def _identity_default(obj: Any, field: str) -> Any:
    from saealib.pipeline import Branch, Loop, Pipeline, Repeat

    if isinstance(obj, Pipeline):
        return ""
    if isinstance(obj, Repeat | Loop | Branch):
        if field == "name":
            return type(obj).__name__.lower()
        if field == "label":
            if isinstance(obj, Repeat):
                return f"Repeat {obj.count} times"
            if isinstance(obj, Loop):
                return "Loop"
            return "Branch"
        return ""
    return getattr(type(obj), field, "")


def _identity_metadata(obj: Any, parameter_names: set[str]) -> dict[str, Any]:
    metadata = {}
    for field in ("name", "label", "notation"):
        if field in parameter_names:
            continue
        descriptor = inspect.getattr_static(type(obj), field, _MISSING)
        if isinstance(descriptor, property) and descriptor.fset is None:
            continue
        try:
            value = getattr(obj, field)
        except AttributeError:
            continue
        if value != _identity_default(obj, field):
            metadata[field] = value
    return metadata


def _constructor_spec(
    obj: Any,
    path: tuple[object, ...],
    active: set[int],
    seen: dict[int, tuple[Any, str, bool]],
    root: str,
    *,
    strict: bool,
    source_declaration: bool = False,
) -> dict[str, Any]:
    from saealib.pipeline import Branch, Loop, Pipeline, Repeat, Stage
    from saealib.stages import EvaluationPlanStage

    cls = type(obj)
    type_name = _find_registered_name(cls) or _dotted_path(cls)
    if cls.__init__ is object.__init__:
        return {"type": type_name, "params": {}}

    parameters = [
        parameter
        for name, parameter in inspect.signature(cls.__init__).parameters.items()
        if name != "self"
    ]
    parameter_names = {
        p.name for p in parameters if p.kind not in {p.VAR_KEYWORD, p.VAR_POSITIONAL}
    }
    stage_fields = _builtin_stage_fields(obj) if isinstance(obj, Stage) else {}
    allow_private = strict or isinstance(obj, Stage)

    def unavailable(name: str, required: bool) -> None:
        if strict:
            raise ValidationError(
                f"Cannot serialize {_path_text(path, root)}[{name!r}]: "
                "constructor value is unavailable"
            )
        raise ValidationError(
            f"Cannot serialize {_path_text(path, root)} ({type_name!r}): "
            f"no attribute {name!r} matching "
            f"{'required' if required else 'optional'} constructor parameter."
        )

    variable_positional = next(
        (p for p in parameters if p.kind is p.VAR_POSITIONAL), None
    )
    spec: dict[str, Any]
    emitted_parameter_names: set[str] = set()
    if variable_positional is not None:
        parameter = variable_positional
        identity_fields = (
            {"name", "label", "notation"}
            if isinstance(obj, (Stage, Pipeline, Repeat, Loop, Branch))
            else set()
        )
        for other in parameters:
            if other is parameter or other.kind is other.VAR_KEYWORD:
                continue
            if (
                other.kind is other.KEYWORD_ONLY
                and other.name in identity_fields
                and other.default is not other.empty
            ):
                continue
            other_value = _constructor_value(
                obj,
                other.name,
                parameter_names,
                stage_fields,
                source_declaration,
                allow_private=allow_private,
            )
            if other_value is _OMIT:
                continue
            if (
                other_value is _MISSING
                and not strict
                and other.default is not other.empty
            ):
                continue
            raise ValidationError(
                f"Cannot serialize {_path_text(path, root)}[{other.name!r}]: "
                "constructor parameter cannot be represented alongside "
                f"*{parameter.name}; define _registry_spec."
            )
        value = _constructor_value(
            obj,
            parameter.name,
            parameter_names,
            stage_fields,
            source_declaration,
            allow_private=allow_private,
        )
        if value is _MISSING or value is _OMIT:
            unavailable(parameter.name, True)
        variable_keyword = next(
            (p for p in parameters if p.kind is p.VAR_KEYWORD), None
        )
        if variable_keyword is not None:
            mapping = _constructor_value(
                obj,
                variable_keyword.name,
                parameter_names,
                stage_fields,
                source_declaration,
                allow_private=allow_private,
            )
            if strict and (mapping is _MISSING or not isinstance(mapping, Mapping)):
                unavailable(variable_keyword.name, True)
            if isinstance(mapping, Mapping) and mapping:
                raise ValidationError(
                    f"Cannot serialize "
                    f"{_path_text(path, root)} ({type_name!r})"
                    f"[{variable_keyword.name!r}]: "
                    f"*{parameter.name} cannot represent the "
                    f"{len(mapping)} keyword setting(s) held in "
                    f"**{variable_keyword.name}; define _registry_spec."
                )
        spec = {
            "type": type_name,
            "params": [
                _to_spec(item, (*path, parameter.name, index), active, seen, root)
                for index, item in enumerate(value)
            ],
        }
    else:
        params: dict[str, Any] = {}
        for parameter in parameters:
            name = parameter.name
            value = _constructor_value(
                obj,
                name,
                parameter_names,
                stage_fields,
                source_declaration,
                allow_private=allow_private,
            )
            if value is _OMIT:
                continue
            if parameter.kind is parameter.VAR_KEYWORD:
                if value is _MISSING:
                    unavailable(name, True)
                if not isinstance(value, Mapping):
                    if strict:
                        unavailable(name, True)
                    raise ValidationError(
                        f"Cannot serialize {_path_text(path, root)} ({type_name!r}): "
                        f"attribute {name!r} matching **{name} constructor "
                        "parameter is not a mapping."
                    )
                for key, item in value.items():
                    if key in params:
                        raise ValidationError(
                            f"Cannot serialize {_path_text(path, root)}[{key!r}]: "
                            f"**{name} key collides with a constructor parameter"
                        )
                    params[key] = _to_spec(item, (*path, key), active, seen, root)
                continue
            if (
                name == "planner"
                and isinstance(obj, Stage)
                and type(obj).__init__ is EvaluationPlanStage.__init__
                and getattr(obj, "_n_eval", None) is not None
            ):
                continue
            if value is _MISSING:
                if not strict and parameter.default is not parameter.empty:
                    continue
                unavailable(name, parameter.default is parameter.empty)
            if (
                name == "name"
                and isinstance(obj, (Repeat, Loop, Branch))
                and value == type(obj).__name__.lower()
                and parameter.default == ""
            ):
                continue
            value_root = (
                "declaration" if source_declaration and name == "pipeline" else root
            )
            condition_slot = (isinstance(obj, Branch) and name == "condition") or (
                isinstance(obj, Loop) and name in {"until", "condition"}
            )
            params[name] = _to_spec(
                value,
                (*path, name),
                active,
                seen,
                value_root,
                structural_condition=condition_slot,
            )
        emitted_parameter_names = set(params)
        spec = {"type": type_name, "params": params}

    metadata = (
        _identity_metadata(obj, emitted_parameter_names)
        if isinstance(obj, (Stage, Pipeline, Repeat, Loop, Branch))
        else {}
    )
    if metadata:
        spec["_registry_stage_metadata"] = metadata
    return spec


def _to_spec(
    obj: Any,
    path: tuple[object, ...],
    active: set[int],
    seen: dict[int, tuple[Any, str, bool]],
    root: str = "value",
    *,
    structural_condition: bool = False,
) -> Any:
    """Serialize one value while walking a declaration or component graph.

    Primitives, ``None``, lists/tuples, and dicts pass through structurally
    (numpy arrays become lists). Functions/builtins become a
    ``{"callable": "dotted.path"}`` marker. A ``TemplateStrategy`` instance
    serializes its source declaration when its constructor is inherited from
    ``TemplateStrategy``. Structural pipelines and controls serialize their
    declaration fields; Stages use mapped constructor fields where available
    and store overridden identity attributes separately.
    Other objects are reflected via their constructor signature. A ``*args``
    parameter becomes a params list, while a ``**kwargs`` parameter's
    same-named mapping is merged into params. Cycles and shared object
    instances raise :exc:`ValidationError`; ``active`` tracks cycles and
    ``seen`` retains strong references throughout traversal. ``root`` names
    the source of the reported path.
    """
    if obj is None or isinstance(obj, bool | int | float | str):
        return obj
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, list | tuple | dict):
        identity = id(obj)
        if identity in active:
            raise ValidationError(
                f"Cannot serialize {_path_text(path, root)}: reference cycle detected"
            )
        active.add(identity)
        try:
            if isinstance(obj, dict):
                return {
                    k: _to_spec(v, (*path, k), active, seen, root)
                    for k, v in obj.items()
                }
            return [
                _to_spec(v, (*path, index), active, seen, root)
                for index, v in enumerate(obj)
            ]
        finally:
            active.remove(identity)
    if inspect.isfunction(obj) or inspect.isbuiltin(obj):
        callable_path = _dotted_path(obj)
        try:
            resolved = get(callable_path)
        except ValidationError:
            resolved = None
        if resolved is not obj:
            raise ValidationError(
                f"Cannot serialize {_path_text(path, root)} ({obj!r}) as "
                f"{callable_path!r}: lambdas and nested "
                "functions cannot be serialized; use a module-level function."
            )
        return {"callable": callable_path}

    from saealib.pipeline import Stage

    if isinstance(obj, Stage):
        parameters = inspect.signature(type(obj).__init__).parameters
        callback_names = [
            name for name in ("cbmanager", "callback_manager") if name in parameters
        ]
        fields = _builtin_stage_fields(obj)
        for name in callback_names:
            manager = _constructor_value(
                obj,
                name,
                set(parameters),
                fields,
                False,
                allow_private=True,
            )
            if manager is not _MISSING and manager is not None:
                raise ValidationError(
                    f"Cannot serialize {_path_text(path, root)}[{name!r}]: "
                    "explicit callback manager is not serializable"
                )

    if hasattr(obj, "_registry_spec"):
        identity = id(obj)
        if identity in seen:
            separate = structural_condition or seen[identity][2]
            raise ValidationError(
                f"Cannot serialize {_path_text(path, root)}: shares an object with "
                f"{seen[identity][1]}; "
                + (
                    "use separate instances."
                    if separate
                    else "use a ComponentRef to the same slot."
                )
            )
        seen[identity] = (obj, _path_text(path, root), structural_condition)
        spec = obj._registry_spec
        if spec is None:
            raise ValidationError(
                f"Cannot serialize {_path_text(path, root)}: {type(obj).__name__} "
                "instance has no recorded spec and "
                "cannot be serialized."
            )
        return spec

    from saealib.pipeline import Branch, ComponentRef, Loop, Pipeline, Repeat
    from saealib.strategies.base import TemplateStrategy

    if isinstance(obj, ComponentRef):
        _validate_importable_type(type(obj), path, root)
        return {
            "type": _find_registered_name(type(obj)) or _dotted_path(type(obj)),
            "params": {"role": obj.role, "slot": obj.slot},
        }

    source_declaration = (
        isinstance(obj, TemplateStrategy)
        and type(obj).__init__ is TemplateStrategy.__init__
    )
    if isinstance(obj, TemplateStrategy):
        parameters = inspect.signature(type(obj).__init__).parameters
        if not source_declaration and "pipeline" in parameters:
            raise ValidationError(
                f"Cannot serialize {_path_text(path, root)}: "
                f"{type(obj).__name__} overrides __init__ with a pipeline "
                "parameter, so its declaration transformation cannot be "
                "reconstructed; define _registry_spec."
            )

    _validate_importable_type(type(obj), path, root)

    identity = id(obj)
    if identity in active:
        raise ValidationError(
            f"Cannot serialize {_path_text(path, root)}: reference cycle detected"
        )
    if identity in seen:
        separate = (
            isinstance(obj, (Stage, Pipeline, Branch, Loop, Repeat))
            or structural_condition
            or seen[identity][2]
        )
        raise ValidationError(
            f"Cannot serialize {_path_text(path, root)}: shares an object with "
            f"{seen[identity][1]}; "
            + (
                "use separate instances."
                if separate
                else "use a ComponentRef to the same slot."
            )
        )
    active.add(identity)
    seen[identity] = (obj, _path_text(path, root), structural_condition)
    try:
        if isinstance(obj, TemplateStrategy):
            return _constructor_spec(
                obj,
                path,
                active,
                seen,
                root,
                strict=True,
                source_declaration=source_declaration,
            )
        if isinstance(obj, (Pipeline, Repeat, Loop, Branch, Stage)):
            return _constructor_spec(obj, path, active, seen, root, strict=True)
        return _to_spec_reflected(obj, path, active, seen, root)
    finally:
        active.remove(identity)


def _to_spec_reflected(
    obj: Any,
    path: tuple[object, ...],
    active: set[int],
    seen: dict[int, tuple[Any, str, bool]],
    root: str = "value",
) -> Any:
    return _constructor_spec(obj, path, active, seen, root, strict=False)


def to_spec(obj: Any) -> Any:
    """Serialize a component or declaration into a registry spec.

    Values are represented with nested type/parameter mappings. A
    ``TemplateStrategy`` instance serializes its source declaration when its
    constructor is inherited from ``TemplateStrategy``. Built-in strategy
    specs retain their constructor parameters. Pipelines use ``stages`` as
    their sequence parameter, and Stage identity overrides are stored separately from
    constructor parameters. Callable references and structural values are
    preserved when they can be rebuilt from an importable path.

    Parameters
    ----------
    obj : Any
        A component, strategy declaration, callable, or supported container.

    Returns
    -------
    Any
        A registry spec or a structurally serialized value.

    Raises
    ------
    ValidationError
        If a value cannot be reconstructed or the declaration graph contains
        a reference cycle or shared component instance.

    Notes
    -----
    Reflection follows the actual constructor signature. ``*args`` are
    represented as a list and ``**kwargs`` are merged into the parameter
    mapping. With ``*args``, other constructor parameters are supported only
    as keyword-only identity fields; otherwise, define ``_registry_spec``.
    ``_registry_spec`` overrides reflection. A template strategy
    whose constructor inherits :class:`TemplateStrategy` preserves its source
    declaration. A subclass that overrides ``__init__`` and accepts
    ``pipeline`` must define ``_registry_spec`` because the transformation of
    that argument cannot be inferred. Overridden constructors must store their
    arguments unchanged in same-named attributes, or define ``_registry_spec``.
    Non-importable local classes and shared instances are rejected with a
    location.
    """
    from saealib.strategies.base import TemplateStrategy

    root = "declaration" if isinstance(obj, TemplateStrategy) else "value"
    return _to_spec(obj, (), set(), {}, root)


def _strip_params(spec: Any, *names: str) -> Any:
    """Return a copy of ``spec`` with the given param names removed recursively.

    Non-spec values (including lists of specs, encountered while descending
    into ``params``) pass through structurally. ``spec`` itself is not
    mutated.
    """
    if isinstance(spec, list):
        return [_strip_params(v, *names) for v in spec]
    if not _is_spec(spec):
        return spec
    params = spec.get("params", {})
    if isinstance(params, list):
        new_params: Any = [_strip_params(v, *names) for v in params]
    else:
        new_params = {
            k: _strip_params(v, *names) for k, v in params.items() if k not in names
        }
    return {**spec, "params": new_params}


def _inject_params(spec: Any, **overrides: Any) -> Any:
    """Return a copy of ``spec`` with ``overrides`` injected recursively.

    For each ``{"type", "params": dict}`` node, an override is injected only
    if the target's signature accepts it as a parameter name and it is not
    already present in ``params``. ``spec`` itself is not mutated.
    """
    if isinstance(spec, list):
        return [_inject_params(spec_item, **overrides) for spec_item in spec]
    if not _is_spec(spec):
        return spec
    params = spec.get("params", {})
    if isinstance(params, list):
        new_params: Any = [_inject_params(v, **overrides) for v in params]
        return {**spec, "params": new_params}

    target = get(spec["type"])
    signature = inspect.signature(
        target.__init__ if inspect.isclass(target) else target
    )
    accepted = signature.parameters
    new_params = {k: _inject_params(v, **overrides) for k, v in params.items()}
    for name, value in overrides.items():
        if name in accepted and name not in new_params:
            new_params[name] = value
    return {**spec, "params": new_params}
