"""Internal consistency diagnostics for component contracts."""

from __future__ import annotations

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

__all__ = ["check_component_contract", "check_pymoo_feedback_compatibility"]


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
