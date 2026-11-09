"""Internal consistency diagnostics for component contracts."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sized
from typing import cast

from saealib.core.compiler.diagnostics import (
    DIAGNOSTIC_CODES,
    ContractPath,
    Diagnostic,
    DiagnosticBag,
    Severity,
    near_miss_suggestions,
)
from saealib.core.contracts import (
    ASSUMPTION_KEYS,
    CARDINALITIES,
    COMPLETE_BATCH,
    DATA_SPEC_KINDS,
    ROLES,
    RUNTIME_CAPABILITIES,
    SCHEMA_VARIABLES,
    SERVICE_VOCABULARY,
    ComponentContract,
)
from saealib.core.contracts.vocabulary import Vocabulary
from saealib.core.state import STATE_NAMESPACES

__all__ = [
    "check_component_contract",
    "check_pymoo_feedback_compatibility",
]


_MISSING = object()


def _selection_note(
    slot: str, component: object, provenance: Mapping[str, tuple[str, str]]
) -> str:
    selected = provenance.get(slot)
    if selected is None:
        return ""
    source, reason = selected
    return f" Selected {type(component).__name__} from {source}: {reason}"


def _optimizer_configuration_diagnostics(
    bindings: Mapping[str, object],
    provenance: Mapping[str, tuple[str, str]],
    problem: object,
    evaluator: object,
    *,
    require_initializer: bool = True,
) -> tuple[Diagnostic, ...]:
    """Check optimizer component requirements using resolved bindings."""
    findings: list[Diagnostic] = []
    for slot in ("algorithm", "strategy", "initializer", "termination"):
        if slot == "initializer" and not require_initializer:
            continue
        if bindings.get(slot) is None:
            findings.append(
                Diagnostic(
                    severity=Severity.ERROR,
                    code="missing_required_component",
                    message=(
                        f"{slot} is not set after component resolution; configure "
                        f"{slot} or select a composition that provides it."
                    ),
                    path=ContractPath(components=("optimizer", slot)),
                    resolutions=(f"Configure {slot} before execution.",),
                )
            )

    strategy = bindings.get("strategy")
    if strategy is not None:
        requirement_read_failed = False
        try:
            requires_surrogate = getattr(strategy, "requires_surrogate", _MISSING)
        except Exception as error:
            requirement_read_failed = True
            requires_surrogate = _MISSING
            findings.append(
                Diagnostic(
                    severity=Severity.WARNING,
                    code="cannot_verify_component_requirement",
                    message=(
                        f"The surrogate requirement for {type(strategy).__name__} "
                        f"cannot be read: {type(error).__name__}: {error}."
                        f"{_selection_note('strategy', strategy, provenance)}"
                    ),
                    path=ContractPath(components=("optimizer", "strategy")),
                    resolutions=(
                        "Make the strategy's surrogate requirement readable.",
                    ),
                )
            )
        if requires_surrogate is _MISSING and not requirement_read_failed:
            findings.append(
                Diagnostic(
                    severity=Severity.WARNING,
                    code="cannot_verify_component_requirement",
                    message=(
                        f"The surrogate requirement for {type(strategy).__name__} "
                        "cannot be verified because the strategy does not declare "
                        "requires_surrogate."
                        f"{_selection_note('strategy', strategy, provenance)}"
                    ),
                    path=ContractPath(components=("optimizer", "strategy")),
                    resolutions=(
                        "Declare the requirement when the strategy depends on a "
                        "surrogate manager.",
                    ),
                )
            )
        if requires_surrogate is True:
            if bindings.get("surrogate_manager") is None:
                findings.append(
                    Diagnostic(
                        severity=Severity.ERROR,
                        code="missing_surrogate_manager",
                        message=(
                            f"{type(strategy).__name__} requires a "
                            "surrogate_manager, but no manager is available after "
                            "component resolution."
                            f"{_selection_note('strategy', strategy, provenance)}"
                        ),
                        path=ContractPath(
                            components=("optimizer", "surrogate_manager")
                        ),
                        resolutions=(
                            "Configure a surrogate manager or choose a strategy "
                            "that does not require one.",
                        ),
                    )
                )
            if bindings.get("acquisition") is None:
                findings.append(
                    Diagnostic(
                        severity=Severity.ERROR,
                        code="missing_acquisition",
                        message=(
                            f"{type(strategy).__name__} requires an acquisition, "
                            "but no provider-owned acquisition is available after "
                            "component resolution."
                            f"{_selection_note('strategy', strategy, provenance)}"
                        ),
                        path=ContractPath(components=("optimizer", "acquisition")),
                        resolutions=(
                            "Configure an acquisition component for the selected "
                            "surrogate manager.",
                        ),
                    )
                )

    for slot, method_name in (
        ("evaluation_planner", "plan"),
        ("feedback_builder", "build"),
    ):
        component = bindings.get(slot)
        if component is None:
            continue
        try:
            method = getattr(component, method_name, None)
        except Exception as error:
            message = (
                f"{slot}.{method_name} could not be read from "
                f"{type(component).__name__}: {type(error).__name__}: {error}."
            )
        else:
            if callable(method):
                if not _declares_attribute(component, "contract"):
                    findings.append(
                        Diagnostic(
                            severity=Severity.WARNING,
                            code="cannot_verify_component_requirement",
                            message=(
                                f"The requirements of {type(component).__name__} "
                                f"in {slot} cannot be verified because it does "
                                "not declare a contract."
                                f"{_selection_note(slot, component, provenance)}"
                            ),
                            path=ContractPath(components=("optimizer", slot)),
                            resolutions=(
                                "Declare component input requirements when they "
                                "must be checked against the prepared flow.",
                            ),
                        )
                    )
                continue
            message = (
                f"{slot} must provide a callable {method_name}() operation; "
                f"{type(component).__name__} does not."
            )
        findings.append(
            Diagnostic(
                severity=Severity.ERROR,
                code="component_protocol_mismatch",
                message=message + _selection_note(slot, component, provenance),
                path=ContractPath(components=("optimizer", slot)),
                resolutions=(f"Provide a callable {method_name}() operation.",),
            )
        )

    comparator = _safe_attribute(problem, "comparator")
    if comparator is not _MISSING:
        findings.extend(
            _direction_length_diagnostics(
                comparator,
                slot="comparator",
                n_obj=_safe_attribute(problem, "n_obj"),
                provenance={},
            )
        )

    acquisition = bindings.get("acquisition")
    for channel, component in _acquisition_components(acquisition):
        findings.extend(
            _direction_length_diagnostics(
                component,
                slot="acquisition",
                n_obj=_safe_attribute(problem, "n_obj"),
                provenance=provenance,
                channel=channel,
            )
        )

    lifecycle_status = _evaluator_lifecycle_status(evaluator)
    if lifecycle_status is True:
        findings.append(
            Diagnostic(
                severity=Severity.ERROR,
                code="inconsistent_evaluator_lifecycle",
                message=(
                    "submit(), collect(), and acknowledge() must be overridden "
                    "together."
                    f"{_selection_note('evaluator', evaluator, provenance)}"
                ),
                path=ContractPath(components=("optimizer", "evaluator")),
                resolutions=(
                    "Override submit(), collect(), and acknowledge() together, "
                    "or inherit all three methods.",
                ),
            )
        )
    elif lifecycle_status is None:
        findings.append(
            Diagnostic(
                severity=Severity.WARNING,
                code="cannot_verify_component_requirement",
                message=(
                    f"The lifecycle consistency of {type(evaluator).__name__} "
                    "cannot be verified because it does not provide a lifecycle "
                    "verification method."
                    f"{_selection_note('evaluator', evaluator, provenance)}"
                ),
                path=ContractPath(components=("optimizer", "evaluator")),
                resolutions=(
                    "Ensure submit(), collect(), and acknowledge() implement one "
                    "consistent lifecycle.",
                ),
            )
        )
    return tuple(findings)


def _safe_attribute(value: object, name: str) -> object:
    try:
        return getattr(value, name)
    except Exception:
        return _MISSING


def _declares_attribute(value: object, name: str) -> bool:
    if name in getattr(value, "__dict__", {}):
        return True
    return any(name in base.__dict__ for base in type(value).__mro__)


def _acquisition_components(
    acquisition: object | None,
) -> tuple[tuple[str | None, object], ...]:
    if acquisition is None:
        return ()
    found: list[tuple[str | None, object]] = []
    visited: set[int] = set()

    def visit(component: object, channel: str | None) -> None:
        if id(component) in visited:
            return
        visited.add(id(component))
        children = _safe_attribute(component, "acquisitions")
        if isinstance(children, Mapping):
            for name, child in children.items():
                visit(child, str(name) if channel is None else f"{channel}.{name}")
        else:
            found.append((channel, component))

    visit(acquisition, None)
    return tuple(found)


def _direction_length_diagnostics(
    component: object,
    *,
    slot: str,
    n_obj: object,
    provenance: Mapping[str, tuple[str, str]],
    channel: str | None = None,
) -> tuple[Diagnostic, ...]:
    direction = _safe_attribute(component, "direction")
    if direction is _MISSING or direction is None:
        return ()
    path = (
        ContractPath(components=("problem", "comparator"))
        if slot == "comparator"
        else ContractPath(
            components=("optimizer", "acquisition"),
            role="acquisition",
            port=channel,
        )
    )
    if not isinstance(direction, Sized):
        return (
            Diagnostic(
                severity=Severity.WARNING,
                code="cannot_verify_direction_length",
                message=(
                    f"The direction length for {type(component).__name__} cannot "
                    f"be determined.{_selection_note(slot, component, provenance)}"
                ),
                path=path,
                resolutions=("Provide a direction value with a known length.",),
            ),
        )
    try:
        length = len(direction)
    except Exception:
        return (
            Diagnostic(
                severity=Severity.WARNING,
                code="cannot_verify_direction_length",
                message=(
                    f"The direction length for {type(component).__name__} cannot "
                    f"be determined.{_selection_note(slot, component, provenance)}"
                ),
                path=path,
                resolutions=("Provide a direction value with a known length.",),
            ),
        )
    if length == 0 or not isinstance(n_obj, int) or length == n_obj:
        return ()
    return (
        Diagnostic(
            severity=Severity.ERROR,
            code="direction_length_mismatch",
            message=(
                f"{type(component).__name__} direction length ({length}) does not "
                f"match problem.n_obj ({n_obj})."
                f"{_selection_note(slot, component, provenance)}"
            ),
            path=path,
            resolutions=(
                f"Provide a direction vector with {n_obj} entries, or leave the "
                "direction unset for automatic injection.",
            ),
        ),
    )


def _evaluator_lifecycle_status(evaluator: object) -> bool | None:
    checker = _safe_attribute(evaluator, "has_partial_lifecycle_override")
    if callable(checker):
        try:
            return bool(cast(Callable[[], object], checker)())
        except Exception:
            return None
    methods = ("submit", "collect", "acknowledge")
    implementations = tuple(getattr(type(evaluator), name, None) for name in methods)
    if any(not callable(getattr(evaluator, name, None)) for name in methods):
        return True
    checker_owner = next(
        (
            base
            for base in type(evaluator).__mro__
            if "has_partial_lifecycle_override" in base.__dict__
        ),
        None,
    )
    if checker_owner is None:
        return None
    overridden = tuple(
        implementation is not getattr(checker_owner, name, None)
        for name, implementation in zip(methods, implementations, strict=True)
    )
    return any(overridden) and not all(overridden)


def _strategy_graph_diagnostic(
    strategy: object,
    provenance: Mapping[str, tuple[str, str]],
    error: BaseException | None = None,
) -> Diagnostic:
    """Describe a strategy whose graph could not be prepared."""
    detail = (
        "does not provide a callable build_graph() operation"
        if error is None
        else f"failed to build its graph: {type(error).__name__}: {error}"
    )
    return Diagnostic(
        severity=Severity.ERROR,
        code="strategy_graph_unavailable",
        message=(
            f"{type(strategy).__name__} could not be prepared: {detail}."
            f"{_selection_note('strategy', strategy, provenance)}"
        ),
        path=ContractPath(components=("optimizer", "strategy")),
        resolutions=(
            "Make build_graph() return a ComponentGraph and ensure its declared "
            "contracts can be prepared.",
        ),
    )


def check_component_contract(
    contract: ComponentContract,
    *,
    component: object | None = None,
    path: ContractPath | None = None,
) -> DiagnosticBag:
    """Return diagnostics for one contract and its held parts.

    Vocabulary checks inspect only the supplied contract.  Passing ``component``
    additionally enables the live-object checks required for recursive parts.
    All ordinary component failures are converted to diagnostics.
    """
    root = path or ContractPath(components=("component",))
    diagnostics = DiagnosticBag()
    _check_vocabularies(contract, root, diagnostics)
    if component is not None:
        _check_parts(contract, component, root, diagnostics)
    return diagnostics


def check_pymoo_feedback_compatibility(
    consumer_contract: ComponentContract,
    *,
    consumer_path: ContractPath,
    runtime_path: ContractPath,
) -> DiagnosticBag:
    """Diagnose a complete-batch consumer paired with a partial runtime.

    The caller is responsible for restricting this check to the Pymoo
    ``allow_partial_tell=False`` and asynchronous-runtime combination.
    """
    diagnostics = DiagnosticBag()
    feedback = consumer_contract.lifecycle.feedback
    if feedback is None or feedback.completion != COMPLETE_BATCH:
        return diagnostics
    diagnostics.append(
        Diagnostic(
            severity=Severity.ERROR,
            code="pymoo_partial_feedback_unsupported",
            message=(
                f"Feedback consumer at {consumer_path} requires a complete "
                f"batch, but the partial runtime at {runtime_path} may deliver "
                "partial feedback."
            ),
            path=consumer_path,
            related=(runtime_path,),
            resolutions=(
                "Configure the feedback consumer to accept partial feedback, "
                "or disable partial delivery in the runtime.",
            ),
        )
    )
    return diagnostics


def _finding(
    diagnostics: DiagnosticBag,
    code: str,
    path: ContractPath,
    message: str,
    resolution: str,
    *,
    severity: Severity = Severity.ERROR,
) -> None:
    if DIAGNOSTIC_CODES.get(code) is None:
        message = (
            f"Attempted to emit unregistered diagnostic code {code!r} at {path}. "
            f"Original finding: {message}"
        )
        resolution = (
            f"Register diagnostic code {code!r}, or use a registered diagnostic code. "
            f"Original resolution: {resolution}"
        )
        code = "unregistered_diagnostic_code"
        severity = Severity.ERROR
    diagnostics.append(
        Diagnostic(
            severity=severity,
            code=code,
            message=message,
            path=path,
            resolutions=(resolution,),
        )
    )


def _suggestion_note(name: str, vocabulary: Vocabulary) -> str:
    suggestions = near_miss_suggestions(name, vocabulary)
    if not suggestions:
        return ""
    return f" Registered names similar to {name!r}: {', '.join(suggestions)}."


def _check_vocabularies(
    contract: ComponentContract, path: ContractPath, diagnostics: DiagnosticBag
) -> None:
    for role, role_contract in contract.ports.items():
        role_path = ContractPath(components=path.components, role=role)
        if ROLES.get(role) is None:
            _finding(
                diagnostics,
                "unknown_role",
                role_path,
                (
                    f"Role {role!r} is not registered at {role_path}, so ports "
                    f"declared with it cannot be verified."
                    f"{_suggestion_note(role, ROLES)}"
                ),
                f"Register {role!r} in ROLES, or use a registered role name.",
                severity=Severity.WARNING,
            )
        for port in (*role_contract.inputs, *role_contract.outputs):
            port_path = ContractPath(
                components=path.components, role=role, port=port.name
            )
            if DATA_SPEC_KINDS.get(port.data.kind) is None:
                _finding(
                    diagnostics,
                    "unknown_data_spec",
                    port_path,
                    (
                        f"Data specification kind {port.data.kind!r} is not "
                        f"registered at {port_path}, so this port cannot be "
                        f"verified against connections."
                        f"{_suggestion_note(port.data.kind, DATA_SPEC_KINDS)}"
                    ),
                    (
                        f"Register {port.data.kind!r} in DATA_SPEC_KINDS, or use "
                        "a registered data kind."
                    ),
                    severity=Severity.WARNING,
                )
            if CARDINALITIES.get(port.cardinality) is None:
                _finding(
                    diagnostics,
                    "unknown_cardinality",
                    port_path,
                    (
                        f"Cardinality {port.cardinality!r} is not registered at "
                        f"{port_path}, so this port cannot be verified against "
                        f"connections."
                        f"{_suggestion_note(port.cardinality, CARDINALITIES)}"
                    ),
                    (
                        f"Register {port.cardinality!r} in CARDINALITIES, or use "
                        "a registered cardinality."
                    ),
                    severity=Severity.WARNING,
                )
            for variable in port.data.bindings:
                if SCHEMA_VARIABLES.get(variable) is None:
                    _finding(
                        diagnostics,
                        "unknown_schema_variable",
                        port_path,
                        (
                            f"Schema variable {variable!r} is not registered at "
                            f"{port_path}, so bindings using it cannot be verified."
                            f"{_suggestion_note(variable, SCHEMA_VARIABLES)}"
                        ),
                        (
                            f"Register {variable!r} in SCHEMA_VARIABLES, or use "
                            "a registered schema variable."
                        ),
                        severity=Severity.WARNING,
                    )
            for service in port.required_services:
                if SERVICE_VOCABULARY.get(service.name) is None:
                    _finding(
                        diagnostics,
                        "unknown_service",
                        port_path,
                        (
                            f"Service {service.name!r} is not registered at "
                            f"{port_path}, so this requirement cannot be verified."
                            f"{_suggestion_note(service.name, SERVICE_VOCABULARY)}"
                        ),
                        (
                            f"Register {service.name!r} in SERVICE_VOCABULARY, or "
                            "use a registered service name."
                        ),
                        severity=Severity.WARNING,
                    )

    for state_keys in (
        contract.state.reads,
        contract.state.writes,
        contract.state.exports,
    ):
        for key in state_keys:
            if STATE_NAMESPACES.get(key.namespace) is None:
                _finding(
                    diagnostics,
                    "unknown_state_namespace",
                    path,
                    (
                        f"State namespace {key.namespace!r} is not registered at "
                        f"{path}, so this state declaration cannot be verified."
                        f"{_suggestion_note(key.namespace, STATE_NAMESPACES)}"
                    ),
                    (
                        f"Register {key.namespace!r} in STATE_NAMESPACES, or use "
                        "a registered namespace."
                    ),
                    severity=Severity.WARNING,
                )
    for capability in (
        *contract.execution.required_runtime_capabilities,
        *contract.execution.offered_runtime_capabilities,
    ):
        if RUNTIME_CAPABILITIES.get(capability) is None:
            _finding(
                diagnostics,
                "unknown_runtime_capability",
                path,
                (
                    f"Runtime capability {capability!r} is not registered at "
                    f"{path}, so this requirement cannot be verified."
                    f"{_suggestion_note(capability, RUNTIME_CAPABILITIES)}"
                ),
                (
                    f"Register {capability!r} in RUNTIME_CAPABILITIES, or use "
                    "a registered capability."
                ),
                severity=Severity.WARNING,
            )
    for assumption in contract.assumptions:
        if ASSUMPTION_KEYS.get(assumption) is None:
            _finding(
                diagnostics,
                "unknown_assumption_key",
                path,
                (
                    f"Assumption key {assumption!r} is not registered at {path}, "
                    f"so this assumption cannot be verified."
                    f"{_suggestion_note(assumption, ASSUMPTION_KEYS)}"
                ),
                (
                    f"Register {assumption!r} in ASSUMPTION_KEYS, or use a "
                    "registered assumption key."
                ),
                severity=Severity.WARNING,
            )


def _check_parts(
    contract: ComponentContract,
    component: object,
    path: ContractPath,
    diagnostics: DiagnosticBag,
) -> None:
    for part in contract.parts:
        part_path = ContractPath(components=(*path.components, part.name))
        try:
            child = getattr(component, part.name)
        except AttributeError:
            _finding(
                diagnostics,
                "part_contract_mismatch",
                part_path,
                (
                    f"Part {part.name!r} is missing on parent component type "
                    f"{type(component).__name__} at {part_path}."
                ),
                (
                    f"Add a {part.name!r} attribute holding the declared child "
                    f"component on {type(component).__name__}."
                ),
            )
            continue
        except Exception as error:
            _finding(
                diagnostics,
                "part_contract_mismatch",
                part_path,
                (
                    f"Reading part {part.name!r} on parent type "
                    f"{type(component).__name__} raised {type(error).__name__}: "
                    f"{error}."
                ),
                (
                    f"Make attribute {part.name!r} readable and return the held "
                    "child component."
                ),
            )
            continue

        try:
            contract_method = getattr(child, "contract", None)
        except Exception as error:
            _finding(
                diagnostics,
                "part_contract_mismatch",
                part_path,
                (
                    f"Reading contract() for part {part.name!r} on child type "
                    f"{type(child).__name__} raised {type(error).__name__}: "
                    f"{error}."
                ),
                (
                    f"Make {type(child).__name__}.contract readable and callable "
                    "so it returns a ComponentContract."
                ),
            )
            continue
        if not callable(contract_method):
            _finding(
                diagnostics,
                "part_contract_mismatch",
                part_path,
                (
                    f"Part {part.name!r} resolved to child type "
                    f"{type(child).__name__}, which has no callable contract() "
                    f"at {part_path}."
                ),
                (
                    f"Give child attribute {part.name!r} a callable contract() "
                    "returning its ComponentContract."
                ),
            )
            continue
        try:
            child_contract = contract_method()
        except Exception as error:
            _finding(
                diagnostics,
                "part_contract_mismatch",
                part_path,
                (
                    f"contract() for part {part.name!r} on child type "
                    f"{type(child).__name__} raised {type(error).__name__}: "
                    f"{error}."
                ),
                (
                    f"Make {type(child).__name__}.contract() return the declared "
                    "ComponentContract without raising."
                ),
            )
            continue
        if child_contract != part.contract:
            _finding(
                diagnostics,
                "part_contract_mismatch",
                part_path,
                (
                    f"Declared contract for part {part.name!r} on parent type "
                    f"{type(component).__name__} differs from child type "
                    f"{type(child).__name__} contract() at {part_path}."
                ),
                (
                    f"Make {type(child).__name__}.contract() equal the "
                    f"PartSpec.contract declared by {type(component).__name__}."
                ),
            )
            continue
        _check_vocabularies(child_contract, part_path, diagnostics)
        _check_parts(child_contract, child, part_path, diagnostics)
