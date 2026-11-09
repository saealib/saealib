"""Composition-dependent component and value selection."""

from __future__ import annotations

from collections.abc import Mapping
from copy import copy, deepcopy
from dataclasses import dataclass
from math import ceil
from typing import Any, cast

from saealib.defaults.keys import INITIAL_ARCHIVE_SIZE, MAX_EVALUATIONS, POPULATION_SIZE
from saealib.defaults.model import DefaultResolution, ResolvedDefault
from saealib.exceptions import ConfigurationError

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

_COMPONENT_SLOTS = (*BINDING_SLOTS, "initializer", "termination")


@dataclass(frozen=True)
class _ComponentOrigin:
    value: object
    source: str
    reason: str
    signature: str


@dataclass(frozen=True)
class _ComponentResolution:
    components: Mapping[str, object]
    bindings: Mapping[str, object]
    origins: Mapping[str, _ComponentOrigin]
    defaults: DefaultResolution
    bundled_preset: str


@dataclass(frozen=True)
class _Candidate:
    value: object
    source: str
    reason: str


def _signature(value: object) -> str:
    from saealib.registry import to_spec

    try:
        return repr(to_spec(value))
    except Exception:
        state = getattr(value, "__dict__", None)
        return repr((type(value), state))


def _clone_default(value: object) -> object:
    try:
        return deepcopy(value)
    except Exception:
        try:
            return copy(value)
        except Exception as error:
            raise ConfigurationError(
                f"Cannot create a plan-owned {type(value).__name__} default"
            ) from error


def _strategy_default(strategy: object, value: object) -> object:
    """Keep strategy-owned component objects supplied on the instance."""
    instance_values = getattr(strategy, "__dict__", {})
    if isinstance(instance_values, dict) and any(
        component is value for component in instance_values.values()
    ):
        return value
    return _clone_default(value)


def _stored_origin(
    optimizer: object, slot: str, value: object
) -> _ComponentOrigin | None:
    origins = getattr(optimizer, "_component_origins", {})
    origin = origins.get(slot) if isinstance(origins, dict) else None
    return (
        origin
        if isinstance(origin, _ComponentOrigin) and origin.value is value
        else None
    )


def _assigned_candidate(
    optimizer: object, slot: str, value: object
) -> tuple[_Candidate | None, _ComponentOrigin | None]:
    if value is None:
        return None, None
    origin = _stored_origin(optimizer, slot, value)
    if origin is None:
        return _Candidate(
            value, "direct assignment", "Assigned directly to the optimizer."
        ), None
    if origin.source in {"setter", "direct assignment"}:
        return _Candidate(value, origin.source, origin.reason), origin
    if origin.source == "framework default":
        return None, origin
    return None, origin


def _build_spec(
    optimizer: object, slot: str, spec: object
) -> tuple[object, object | None]:
    from saealib.registry import _inject_params, build

    problem = getattr(optimizer, "problem")
    dim = problem.dim
    direction = problem.direction
    if slot == "surrogate_manager":
        manager, acquisition = cast(Any, optimizer)._build_components_from_spec(spec)
        return manager, acquisition
    if slot in {"algorithm", "strategy", "termination"}:
        spec = _inject_params(spec, dim=dim, direction=direction)
    return build(spec), None


def _preset_acquisition_spec(spec: object) -> object | None:
    """Return the acquisition a preset manager specification declares itself."""
    if not isinstance(spec, Mapping):
        return None
    params = spec.get("params")
    if not isinstance(params, Mapping):
        return None
    return params.get("acquisition")


def _preset_candidates(
    optimizer: object,
    preset: Mapping[str, object] | None,
    source: str,
    label: str,
    slots: set[str] | None = None,
) -> dict[str, _Candidate]:
    if preset is None:
        return {}
    result: dict[str, _Candidate] = {}
    for slot in (
        "algorithm",
        "surrogate_manager",
        "strategy",
        "evaluation_planner",
        "feedback_builder",
        "termination",
    ):
        if slot not in preset or (slots is not None and slot not in slots):
            continue
        value, acquisition = _build_spec(optimizer, slot, preset[slot])
        result[slot] = _Candidate(
            value, source, f"Selected from {label} for the {slot} slot."
        )
        if slot == "surrogate_manager" and acquisition is not None:
            result["acquisition"] = _Candidate(
                acquisition,
                source,
                f"Selected with the surrogate_manager specification in {label}.",
            )
    return result


def _resolve_defaults(optimizer: object) -> DefaultResolution:
    problem = getattr(optimizer, "problem")
    dim = problem.dim
    comparator = getattr(problem, "comparator", None)
    reference_points = getattr(comparator, "reference_points", None)
    aligns_to_reference_points = getattr(
        comparator, "aligns_population_to_reference_points", False
    )
    if reference_points is None or not aligns_to_reference_points:
        population = 4 * dim
        population_reason = f"Four individuals per dimension: 4 * {dim} = {population}."
    else:
        n_reference_points = len(reference_points)
        population = 4 * ceil(n_reference_points / 4)
        population_reason = (
            "Population aligned to the available reference points: "
            f"4 * ceil({n_reference_points} / 4) = {population}."
        )
    archive = 5 * dim
    archive_reason = f"Five archive entries per dimension: 5 * {dim} = {archive}."
    if archive < population:
        archive = population
        archive_reason = f"Raised to match the resolved population size: {population}."
    evaluations = 200 * dim
    return DefaultResolution(
        values={
            POPULATION_SIZE: population,
            INITIAL_ARCHIVE_SIZE: archive,
            MAX_EVALUATIONS: evaluations,
        },
        resolved={
            POPULATION_SIZE: ResolvedDefault(
                POPULATION_SIZE,
                population,
                "composition rule",
                population_reason,
            ),
            INITIAL_ARCHIVE_SIZE: ResolvedDefault(
                INITIAL_ARCHIVE_SIZE,
                archive,
                "composition rule",
                archive_reason,
            ),
            MAX_EVALUATIONS: ResolvedDefault(
                MAX_EVALUATIONS,
                evaluations,
                "composition rule",
                f"Two hundred evaluations per dimension: 200 * {dim} = {evaluations}.",
            ),
        },
    )


def resolve_components(
    optimizer: object,
    configured: Mapping[str, object] | None = None,
) -> _ComponentResolution:
    """Resolve one optimizer composition without changing its configuration."""
    raw = {slot: getattr(optimizer, slot, None) for slot in _COMPONENT_SLOTS}
    if configured is not None:
        raw.update(
            {slot: configured[slot] for slot in BINDING_SLOTS if slot in configured}
        )
    problem = getattr(optimizer, "problem")

    explicit: dict[str, _Candidate] = {}
    materialized: dict[str, _ComponentOrigin | None] = {}
    framework: dict[str, _Candidate] = {}
    for slot in _COMPONENT_SLOTS:
        candidate, origin = _assigned_candidate(optimizer, slot, raw.get(slot))
        if candidate is not None:
            explicit[slot] = candidate
        elif origin is not None and origin.source == "framework default":
            framework[slot] = _Candidate(
                origin.value,
                "strategy default",
                origin.reason,
            )
        else:
            materialized[slot] = origin

    user_preset = getattr(optimizer, "_preset", None)
    user_slots = {
        slot
        for slot in _COMPONENT_SLOTS
        if user_preset is not None and slot in user_preset
    }
    user_slots.difference_update(explicit)
    persisted_candidates = {
        slot: _Candidate(raw[slot], "persisted", origin.reason)
        for slot, origin in materialized.items()
        if slot in raw and origin is not None and origin.source == "persisted"
    }
    preset_manager_spec = (
        user_preset.get("surrogate_manager")
        if isinstance(user_preset, Mapping)
        else None
    )
    preset_manager_displaced = (
        "surrogate_manager" in explicit or "surrogate_manager" in persisted_candidates
    )
    # A preset surrogate manager supplies the acquisition it is paired with only
    # while that manager is still selected from the preset.  A manager selected by
    # a higher-priority source keeps the acquisition the preset declares itself,
    # but not the acquisition derived from the unselected manager.
    if preset_manager_displaced:
        if (
            preset_manager_spec is not None
            and "acquisition" not in explicit
            and _preset_acquisition_spec(preset_manager_spec) is not None
        ):
            user_slots.add("surrogate_manager")
        else:
            user_slots.discard("surrogate_manager")
    user_candidates = _preset_candidates(
        optimizer,
        user_preset,
        "user preset",
        "the user preset",
        user_slots,
    )

    algorithm = explicit.get("algorithm")
    if algorithm is None:
        algorithm = persisted_candidates.get("algorithm")
    if algorithm is None:
        algorithm = user_candidates.get("algorithm")
    if algorithm is None and materialized.get("algorithm") is not None:
        algorithm = _Candidate(raw.get("algorithm"), "bundled preset", "")

    from saealib.defaults import load_defaults
    from saealib.defaults.loader import select_preset_name

    defaults = load_defaults()
    algorithm_name = None if algorithm is None else type(algorithm.value).__name__
    bundled_name = select_preset_name(
        defaults, getattr(optimizer, "problem"), algorithm_name
    )
    bundled_preset = defaults["presets"][bundled_name]
    selected_strategy = (
        explicit.get("strategy")
        or persisted_candidates.get("strategy")
        or user_candidates.get("strategy")
    )
    bundled_strategy_selected = (
        selected_strategy is None and "strategy" in bundled_preset
    )
    bundled_slots = {
        slot
        for slot in bundled_preset
        if slot in _COMPONENT_SLOTS
        and slot not in explicit
        and slot not in persisted_candidates
        and slot not in user_candidates
        and (
            slot not in {"evaluation_planner", "feedback_builder"}
            or bundled_strategy_selected
        )
    }
    bundled_candidates = _preset_candidates(
        optimizer,
        bundled_preset,
        "bundled preset",
        f"the {bundled_name} bundled preset",
        bundled_slots,
    )

    strategy_candidate = selected_strategy or bundled_candidates.get("strategy")
    strategy = strategy_candidate.value if strategy_candidate is not None else None
    strategy_defaults: Mapping[str, object] = {}
    default_components = getattr(strategy, "_default_components", None)
    if callable(default_components):
        try:
            strategy_defaults = default_components()
        except Exception as error:
            raise ConfigurationError(
                f"{type(strategy).__name__}._default_components() failed: "
                f"{type(error).__name__}: {error}"
            ) from error
    persisted_strategy_defaults: dict[str, _Candidate] = {}
    if "strategy" in persisted_candidates:
        for slot in ("evaluation_planner", "feedback_builder"):
            value = strategy_defaults.get(slot)
            if raw.get(slot) is None and value is not None:
                persisted_strategy_defaults[slot] = _Candidate(
                    _strategy_default(strategy, value),
                    "strategy default",
                    f"Provided by {type(strategy).__name__} for the {slot} slot.",
                )

    candidates: dict[str, _Candidate] = {}
    for slot in _COMPONENT_SLOTS:
        if slot in explicit:
            candidates[slot] = explicit[slot]
        elif slot in persisted_candidates:
            candidates[slot] = persisted_candidates[slot]
        elif slot in user_candidates:
            candidates[slot] = user_candidates[slot]
        elif slot in persisted_strategy_defaults:
            candidates[slot] = persisted_strategy_defaults[slot]
        elif slot in bundled_candidates:
            candidates[slot] = bundled_candidates[slot]
        elif slot in framework:
            candidates[slot] = framework[slot]
        elif slot in strategy_defaults and strategy_defaults[slot] is not None:
            candidates[slot] = _Candidate(
                _strategy_default(strategy, strategy_defaults[slot]),
                "strategy default",
                f"Provided by {type(strategy).__name__} for the {slot} slot.",
            )

    if (
        "acquisition" not in candidates
        and candidates.get("surrogate_manager") is not None
    ):
        manager = candidates["surrogate_manager"].value
        from saealib.optimizer import _default_acquisition_spec
        from saealib.registry import build

        candidates["acquisition"] = _Candidate(
            build(_default_acquisition_spec(type(manager), problem.direction)),
            "strategy default",
            "Selected from the configured surrogate manager's prediction mode.",
        )

    from saealib.callback import CallbackManager
    from saealib.execution.evaluator import SerialEvaluator
    from saealib.execution.initializer import LHSInitializer
    from saealib.termination import Termination, max_fe

    candidates.setdefault(
        "evaluator",
        _Candidate(
            SerialEvaluator(), "bundled preset", "Uses serial objective evaluation."
        ),
    )
    if raw.get("cbmanager") is None and "cbmanager" not in candidates:
        candidates["cbmanager"] = _Candidate(
            CallbackManager(), "bundled preset", "Uses the optimizer callback manager."
        )
    elif "cbmanager" not in candidates and raw.get("cbmanager") is not None:
        origin = materialized.get("cbmanager")
        if origin is not None:
            candidates["cbmanager"] = _Candidate(
                raw["cbmanager"], origin.source, origin.reason
            )

    defaults_resolution = _resolve_defaults(optimizer)
    if "initializer" not in explicit:
        if "initializer" in user_candidates:
            candidates["initializer"] = user_candidates["initializer"]
        elif "initializer" in bundled_candidates:
            candidates["initializer"] = bundled_candidates["initializer"]
        elif (
            raw.get("initializer") is None
            or materialized.get("initializer") is not None
        ):
            candidates["initializer"] = _Candidate(
                LHSInitializer(
                    n_init_archive=defaults_resolution.get(INITIAL_ARCHIVE_SIZE),
                    n_init_population=defaults_resolution.get(POPULATION_SIZE),
                    seed=getattr(optimizer, "seed", None),
                ),
                "composition rule",
                "Uses the resolved population and archive sizes.",
            )

    if "termination" not in explicit and "termination" not in candidates:
        if "termination" in user_candidates:
            candidates["termination"] = user_candidates["termination"]
        elif "termination" in bundled_candidates:
            candidates["termination"] = bundled_candidates["termination"]
        elif (
            raw.get("termination") is None
            or materialized.get("termination") is not None
        ):
            candidates["termination"] = _Candidate(
                Termination(max_fe(defaults_resolution.get(MAX_EVALUATIONS))),
                "composition rule",
                "Uses the resolved maximum evaluation count.",
            )

    origins: dict[str, _ComponentOrigin] = {}
    resolved_values: dict[str, object] = {}
    reuse_origins = getattr(optimizer, "_component_origins", {})
    for slot in _COMPONENT_SLOTS:
        candidate = candidates.get(slot)
        if candidate is None:
            resolved_values[slot] = raw.get(slot)
            old = reuse_origins.get(slot) if isinstance(reuse_origins, dict) else None
            if (
                isinstance(old, _ComponentOrigin)
                and old.source == "persisted"
                and raw.get(slot) is not None
            ):
                origins[slot] = old
            continue
        signature = _signature(candidate.value)
        old = reuse_origins.get(slot) if isinstance(reuse_origins, dict) else None
        current = raw.get(slot)
        if (
            isinstance(old, _ComponentOrigin)
            and old.source == "persisted"
            and old.value is current
        ):
            value = current
            reason = old.reason
            source = "persisted"
            signature = old.signature
        elif (
            isinstance(old, _ComponentOrigin)
            and old.value is current
            and old.signature == signature
            and old.source == candidate.source
        ):
            value = current
            reason = old.reason
            source = candidate.source
        else:
            value = candidate.value
            reason = candidate.reason
            source = candidate.source
        resolved_values[slot] = value
        origins[slot] = _ComponentOrigin(value, source, reason, signature)

    for slot in ("initializer", "termination"):
        candidate = candidates.get(slot)
        if candidate is None:
            resolved_values[slot] = raw.get(slot)
            old = reuse_origins.get(slot) if isinstance(reuse_origins, dict) else None
            if (
                isinstance(old, _ComponentOrigin)
                and old.source == "persisted"
                and raw.get(slot) is not None
            ):
                origins[slot] = old
            continue
        signature = _signature(candidate.value)
        old = reuse_origins.get(slot) if isinstance(reuse_origins, dict) else None
        current = raw.get(slot)
        if (
            isinstance(old, _ComponentOrigin)
            and old.source == "persisted"
            and old.value is current
        ):
            value = current
            reason = old.reason
            source = "persisted"
            signature = old.signature
        elif (
            isinstance(old, _ComponentOrigin)
            and old.value is current
            and old.signature == signature
            and old.source == candidate.source
        ):
            value = current
            reason = old.reason
            source = candidate.source
        else:
            value = candidate.value
            reason = candidate.reason
            source = candidate.source
        resolved_values[slot] = value
        origins[slot] = _ComponentOrigin(value, source, reason, signature)

    defaults_resolution = DefaultResolution(
        values=defaults_resolution.values,
        resolved=defaults_resolution.resolved,
    )
    return _ComponentResolution(
        components=resolved_values,
        bindings={slot: resolved_values.get(slot) for slot in BINDING_SLOTS},
        origins=origins,
        defaults=defaults_resolution,
        bundled_preset=bundled_name,
    )
