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
        if isinstance(params, list):
            args = [
                build(v) if _is_spec(v) or _is_callable_ref(v) else v for v in params
            ]
            try:
                return cls(*args)
            except TypeError as e:
                raise ValidationError(f"Cannot construct {spec['type']!r}: {e}") from e
        built_params = {
            k: build(v) if _is_spec(v) or _is_callable_ref(v) else v
            for k, v in params.items()
        }
        try:
            return cls(**built_params)
        except TypeError as e:
            raise ValidationError(f"Cannot construct {spec['type']!r}: {e}") from e
    return spec


def _dotted_path(obj: Any) -> str:
    return f"{obj.__module__}.{obj.__qualname__}"


def _find_registered_name(cls: Any) -> str | None:
    for name, registered_cls in _REGISTRY.items():
        if registered_cls is cls:
            return name
    return None


def to_spec(obj: Any) -> Any:
    """Recursively serialize ``obj`` into a spec (the reverse of ``build()``).

    Primitives, ``None``, lists/tuples, and dicts pass through structurally
    (numpy arrays become lists). Functions/builtins become a
    ``{"callable": "dotted.path"}`` marker. Other objects are reflected via
    their constructor signature: for each ``__init__`` parameter, the
    same-named attribute on ``obj`` is read and serialized recursively. A
    single ``*args`` (``VAR_POSITIONAL``) parameter is serialized as a
    params *list* instead of a dict, so it round-trips through ``build()``'s
    positional-call path. A ``**kwargs`` (``VAR_KEYWORD``) parameter's
    same-named mapping attribute is serialized item by item and merged into
    ``params``. The params-list form cannot carry keyword settings, so a
    class holding a non-empty same-named mapping attribute for its
    ``**kwargs`` parameter alongside ``*args`` cannot be serialized and
    raises :exc:`ValidationError` instead of dropping them.
    """
    if obj is None or isinstance(obj, bool | int | float | str):
        return obj
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, list | tuple):
        return [to_spec(v) for v in obj]
    if isinstance(obj, dict):
        return {k: to_spec(v) for k, v in obj.items()}
    if inspect.isfunction(obj) or inspect.isbuiltin(obj):
        path = _dotted_path(obj)
        try:
            resolved = get(path)
        except ValidationError:
            resolved = None
        if resolved is not obj:
            raise ValidationError(
                f"Cannot serialize {obj!r} as {path!r}: lambdas and nested "
                "functions cannot be serialized; use a module-level function."
            )
        return {"callable": path}

    if hasattr(obj, "_registry_spec"):
        spec = obj._registry_spec
        if spec is None:
            raise ValidationError(
                f"{type(obj).__name__} instance has no recorded spec and "
                "cannot be serialized."
            )
        return spec

    cls = type(obj)
    type_name = _find_registered_name(cls) or _dotted_path(cls)
    if cls.__init__ is object.__init__:
        return {"type": type_name, "params": {}}
    parameters = [
        p
        for name, p in inspect.signature(cls.__init__).parameters.items()
        if name != "self"
    ]

    var_positional = [p for p in parameters if p.kind is p.VAR_POSITIONAL]
    if var_positional:
        (param,) = var_positional
        if not hasattr(obj, param.name):
            raise ValidationError(
                f"Cannot serialize {type_name!r}: no attribute {param.name!r} "
                f"matching *{param.name} constructor parameter."
            )
        var_keyword = next((p for p in parameters if p.kind is p.VAR_KEYWORD), None)
        if var_keyword is not None and hasattr(obj, var_keyword.name):
            mapping = getattr(obj, var_keyword.name)
            if isinstance(mapping, Mapping) and mapping:
                raise ValidationError(
                    f"Cannot serialize {type_name!r}: *{param.name} form cannot "
                    f"represent the {len(mapping)} keyword setting(s) held in "
                    f"the attribute {var_keyword.name!r} matching "
                    f"**{var_keyword.name} constructor parameter."
                )
        return {
            "type": type_name,
            "params": [to_spec(v) for v in getattr(obj, param.name)],
        }

    params: dict[str, Any] = {}
    for param in parameters:
        if param.kind is param.VAR_KEYWORD:
            if not hasattr(obj, param.name):
                raise ValidationError(
                    f"Cannot serialize {type_name!r}: no attribute {param.name!r} "
                    f"matching **{param.name} constructor parameter."
                )
            mapping = getattr(obj, param.name)
            if not isinstance(mapping, Mapping):
                raise ValidationError(
                    f"Cannot serialize {type_name!r}: attribute {param.name!r} "
                    f"matching **{param.name} constructor parameter is not a "
                    "mapping."
                )
            for key, value in mapping.items():
                if key in params:
                    raise ValidationError(
                        f"Cannot serialize {type_name!r}: **{param.name} key "
                        f"{key!r} collides with a same-named constructor "
                        "parameter."
                    )
                params[key] = to_spec(value)
            continue
        if not hasattr(obj, param.name):
            if param.default is not param.empty:
                continue
            raise ValidationError(
                f"Cannot serialize {type_name!r}: no attribute {param.name!r} "
                "matching required constructor parameter."
            )
        params[param.name] = to_spec(getattr(obj, param.name))
    return {"type": type_name, "params": params}


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
