from __future__ import annotations

import difflib
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from enum import Enum

from saealib.core.contracts.vocabulary import Vocabulary, VocabularyDescriptor
from saealib.exceptions import ValidationError

__all__ = [
    "DIAGNOSTIC_CODES",
    "ContractPath",
    "Diagnostic",
    "DiagnosticBag",
    "DiagnosticCodeVocabulary",
    "Severity",
    "near_miss_suggestions",
]


class Severity(str, Enum):
    """Severity levels available to compiler diagnostics."""

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


def near_miss_suggestions(
    name: str, vocabulary: Vocabulary, *, count: int = 3
) -> tuple[str, ...]:
    """Return registered names closest to an unregistered ``name``."""
    folded = name.casefold()
    scored = [
        (
            difflib.SequenceMatcher(None, folded, str(candidate).casefold()).ratio(),
            str(candidate),
        )
        for candidate in vocabulary.names()
    ]
    scored.sort(key=lambda item: (-item[0], item[1]))
    matches = [candidate for ratio, candidate in scored if ratio >= 0.6]
    return tuple(matches[:count])


@dataclass(frozen=True, kw_only=True)
class ContractPath:
    """An ordered component path with an optional role and port."""

    components: tuple[str, ...]
    role: str | None = None
    port: str | None = None

    def __post_init__(self) -> None:
        components = tuple(self.components)
        if not components or any(
            not isinstance(component, str) or not component for component in components
        ):
            raise ValidationError("A contract path must contain a component id")
        if self.role is not None and (not isinstance(self.role, str) or not self.role):
            raise ValidationError("A contract path role must be a non-empty string")
        if self.port is not None and (not isinstance(self.port, str) or not self.port):
            raise ValidationError("A contract path port must be a non-empty string")
        object.__setattr__(self, "components", components)

    def __str__(self) -> str:
        rendered = ".".join(self.components)
        if self.role is not None:
            rendered += f"[{self.role}]"
        if self.port is not None:
            rendered += f".{self.port}"
        return rendered


DiagnosticCodeVocabulary = Vocabulary[VocabularyDescriptor]
DIAGNOSTIC_CODES = DiagnosticCodeVocabulary()
DIAGNOSTIC_CODES.register(
    "unknown_data_spec",
    VocabularyDescriptor(
        name="unknown_data_spec",
        description="A data specification kind has not been registered.",
    ),
)
for _name, _description in (
    (
        "component_binding_selected",
        "A component or semantic value was selected during execution preparation.",
    ),
    (
        "feature_encoder_default_selected",
        "A default FeatureEncoder was selected during execution preparation.",
    ),
    ("uninitialized_state_write", "A state write has no initialized predecessor."),
    ("concurrent_state_write", "Two state writes are not ordered."),
    ("unreachable_state_read", "A state read has no reachable initialized writer."),
    ("concurrent_state_read_write", "A state read and write are not ordered."),
    ("unknown_role", "A component contract role has not been registered."),
    ("unknown_schema_variable", "A data binding key has not been registered."),
    ("unknown_cardinality", "A port cardinality has not been registered."),
    ("unknown_service", "A port service requirement has not been registered."),
    (
        "unresolved_service",
        "A declared port service is registered but is not provided by "
        "the bound context.",
    ),
    ("unknown_state_namespace", "A state-key namespace has not been registered."),
    (
        "unknown_runtime_capability",
        "A required runtime capability has not been registered.",
    ),
    ("unknown_assumption_key", "An assumption key has not been registered."),
    (
        "contract_unavailable",
        "A component contract exists but cannot be called or has the wrong type.",
    ),
    (
        "part_contract_mismatch",
        "A declared part is missing, has no callable contract, or disagrees "
        "with its held component.",
    ),
    (
        "undeclared_part",
        "A component attribute holds a child component that is not declared "
        "in the contract's parts.",
    ),
    (
        "duplicate_component_id",
        "A component id occurs more than once in a component graph.",
    ),
    (
        "invalid_graph_edge",
        "A graph edge refers to a node that is not present in the graph.",
    ),
    (
        "invalid_entry_point",
        "A graph entry point is empty or refers to a missing node.",
    ),
    (
        "invalid_state_binding",
        "A state binding refers to a node that is not present in the graph.",
    ),
    (
        "unreachable_node",
        "A graph node cannot be reached from any declared entry point.",
    ),
    ("conflicting_rewrite", "Two resolution rules claimed the same graph location."),
    (
        "unstable_compilation",
        "Resolution did not reach a fixed point within the compilation limit.",
    ),
    (
        "unclaimed_rewrite",
        "A resolution rule changed a graph location without claiming it.",
    ),
    (
        "structured_execution_mutation",
        "A resolution rule changed the lowered structured execution tree.",
    ),
    (
        "incompatible_port",
        "A data connection does not satisfy its producer and consumer port contracts.",
    ),
    (
        "unknown_port",
        "A data connection names a port that is not declared at its graph endpoint.",
    ),
    (
        "ambiguous_port",
        "A data connection names a port shared by multiple roles without "
        "selecting one.",
    ),
    ("unresolved_input", "A required input has no compatible upstream producer."),
    (
        "missing_required_component",
        "A required optimizer component is not available after resolution.",
    ),
    (
        "component_protocol_mismatch",
        "A configured component does not provide a required callable operation.",
    ),
    (
        "cannot_verify_component_requirement",
        "A component requirement cannot be determined from the available declarations.",
    ),
    (
        "missing_surrogate_manager",
        "The selected strategy requires a surrogate manager that is not available.",
    ),
    (
        "contradicting_surrogate_requirement",
        "A template strategy's explicit surrogate requirement contradicts its "
        "declared flow, so the flow-derived value is used.",
    ),
    (
        "missing_acquisition",
        "The selected strategy requires an acquisition component that is not "
        "available.",
    ),
    (
        "direction_length_mismatch",
        "An objective direction vector does not match the problem objective count.",
    ),
    (
        "objective_count_mismatch",
        "A comparator's per-objective vector does not match the problem "
        "objective count.",
    ),
    (
        "reference_points_dimension_mismatch",
        "A comparator's reference points do not match the problem objective count.",
    ),
    (
        "cannot_verify_direction_length",
        "An objective direction length cannot be determined.",
    ),
    (
        "inconsistent_evaluator_lifecycle",
        "Evaluator submit, collect, and acknowledge methods are not consistently "
        "implemented.",
    ),
    (
        "strategy_graph_unavailable",
        "The selected strategy could not provide a graph for preparation.",
    ),
    ("ambiguous_input", "An input has multiple compatible upstream producers."),
    (
        "invalid_flow_transform",
        "A declared edge transform is unknown or does not accept the preceding "
        "data kind.",
    ),
    (
        "cannot_verify_stage_flow",
        "A Stage has no flow summary, so its component connections cannot be verified.",
    ),
):
    DIAGNOSTIC_CODES.register(
        _name,
        VocabularyDescriptor(name=_name, description=_description),
    )
DIAGNOSTIC_CODES.register(
    "schema_variable_unbound",
    VocabularyDescriptor(
        name="schema_variable_unbound",
        description="A schema variable could not be bound during compilation.",
    ),
)
DIAGNOSTIC_CODES.register(
    "incompatible_representation",
    VocabularyDescriptor(
        name="incompatible_representation",
        description=(
            "A component requires a representation kind the provided "
            "space does not offer. The diagnostic names both the "
            "provided and the required kind."
        ),
    ),
)
DIAGNOSTIC_CODES.register(
    "pymoo_partial_feedback_unsupported",
    VocabularyDescriptor(
        name="pymoo_partial_feedback_unsupported",
        description=(
            "A Pymoo feedback consumer requires complete batches but the "
            "runtime may deliver partial feedback."
        ),
    ),
)
DIAGNOSTIC_CODES.register(
    "unregistered_diagnostic_code",
    VocabularyDescriptor(
        name="unregistered_diagnostic_code",
        description="A diagnostic was emitted with an unregistered code.",
    ),
)
DIAGNOSTIC_CODES.register(
    "ambiguous_adapter",
    VocabularyDescriptor(
        name="ambiguous_adapter",
        description=(
            "More than one lossless adapter can be inserted for one connection."
        ),
    ),
)
DIAGNOSTIC_CODES.register(
    "cannot_verify_data_kind",
    VocabularyDescriptor(
        name="cannot_verify_data_kind",
        description=(
            "A connection names an unregistered data kind or cardinality, so "
            "its compatibility cannot be verified."
        ),
    ),
)
DIAGNOSTIC_CODES.register(
    "surrogate_uncertainty_mismatch",
    VocabularyDescriptor(
        name="surrogate_uncertainty_mismatch",
        description=(
            "An acquisition requires uncertainty estimates that its prediction "
            "source does not provide."
        ),
    ),
)
DIAGNOSTIC_CODES.register(
    "missing_acquisition_channel",
    VocabularyDescriptor(
        name="missing_acquisition_channel",
        description="An acquisition channel has no matching prediction channel.",
    ),
)
DIAGNOSTIC_CODES.register(
    "cannot_verify_acquisition_channel",
    VocabularyDescriptor(
        name="cannot_verify_acquisition_channel",
        description="The manager's prediction channel names cannot be determined.",
    ),
)
DIAGNOSTIC_CODES.register(
    "cannot_verify_surrogate_uncertainty",
    VocabularyDescriptor(
        name="cannot_verify_surrogate_uncertainty",
        description=(
            "Uncertainty support for a prediction channel cannot be determined."
        ),
    ),
)
DIAGNOSTIC_CODES.register(
    "cannot_verify_port",
    VocabularyDescriptor(
        name="cannot_verify_port",
        description=(
            "A connection names a port that is not declared at its graph "
            "endpoint, so the connection cannot be verified."
        ),
    ),
)
DIAGNOSTIC_CODES.register(
    "missing_population_column",
    VocabularyDescriptor(
        name="missing_population_column",
        description=(
            "A component requires population columns that the running state "
            "does not carry."
        ),
    ),
)
DIAGNOSTIC_CODES.register(
    "pristine_copy_unavailable",
    VocabularyDescriptor(
        name="pristine_copy_unavailable",
        description=(
            "A component that keeps construction-time state was re-inserted "
            "after its construction-time copy could not be made."
        ),
    ),
)
DIAGNOSTIC_CODES.register(
    "pymoo_pop_size_mismatch",
    VocabularyDescriptor(
        name="pymoo_pop_size_mismatch",
        description=(
            "A wrapped pymoo algorithm's initial population size does not "
            "match the running population size."
        ),
    ),
)
DIAGNOSTIC_CODES.register(
    "pymoo_pop_size_unverified",
    VocabularyDescriptor(
        name="pymoo_pop_size_unverified",
        description=(
            "A wrapped pymoo algorithm's initial population size cannot be "
            "determined before it runs."
        ),
    ),
)
DIAGNOSTIC_CODES.register(
    "state_not_checkpointable",
    VocabularyDescriptor(
        name="state_not_checkpointable",
        description=(
            "A component declares that its state cannot be exported to a "
            "portable checkpoint."
        ),
    ),
)
DIAGNOSTIC_CODES.register(
    "constraint_handler_threshold_moves",
    VocabularyDescriptor(
        name="constraint_handler_threshold_moves",
        description=(
            "A constraint handler updates its own feasibility threshold during a "
            "run; the search-side relaxation reaches the comparator through "
            "EpsilonSchedule."
        ),
    ),
)
DIAGNOSTIC_CODES.register(
    "opaque_state_effects",
    VocabularyDescriptor(
        name="opaque_state_effects",
        description=(
            "A node declares its state effects non-enumerable, so its state "
            "reads and writes cannot be verified."
        ),
    ),
)


@dataclass(frozen=True, kw_only=True)
class Diagnostic:
    """A structured compiler finding attached to a contract path."""

    severity: Severity
    code: str
    message: str
    path: ContractPath
    related: tuple[ContractPath, ...] = ()
    resolutions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.path, ContractPath):
            raise ValidationError("Diagnostic paths must be ContractPath values")
        related = tuple(self.related)
        if any(not isinstance(path, ContractPath) for path in related):
            raise ValidationError(
                "Related diagnostic paths must be ContractPath values"
            )
        object.__setattr__(self, "related", related)
        object.__setattr__(self, "resolutions", tuple(self.resolutions))

    def __str__(self) -> str:
        return f"{self.severity.value} [{self.code}] at {self.path}: {self.message}"


class DiagnosticBag:
    """An append-only collection of diagnostics."""

    def __init__(self, diagnostics: Iterable[Diagnostic] = ()) -> None:
        self._diagnostics: list[Diagnostic] = list(diagnostics)

    def append(self, diagnostic: Diagnostic) -> None:
        """Append one diagnostic."""
        self._diagnostics.append(diagnostic)

    def extend(self, diagnostics: Iterable[Diagnostic]) -> None:
        """Append diagnostics from an iterable."""
        self._diagnostics.extend(diagnostics)

    @property
    def has_errors(self) -> bool:
        """Return whether an error diagnostic is present."""
        return any(
            diagnostic.severity is Severity.ERROR for diagnostic in self._diagnostics
        )

    def __iter__(self) -> Iterator[Diagnostic]:
        return iter(self._diagnostics)

    def __len__(self) -> int:
        return len(self._diagnostics)


DIAGNOSTIC_CODES.register(
    "missing_genome_codec",
    VocabularyDescriptor(
        name="missing_genome_codec",
        description=(
            "Portable population state requires a GenomeCodec offered by the "
            "configured search space."
        ),
    ),
)
DIAGNOSTIC_CODES.register(
    "missing_runtime_capability",
    VocabularyDescriptor(
        name="missing_runtime_capability",
        description=(
            "A graph node requires a runtime capability absent from the "
            "configured runtime offer."
        ),
    ),
)
DIAGNOSTIC_CODES.register(
    "incompatible_feedback_lifecycle",
    VocabularyDescriptor(
        name="incompatible_feedback_lifecycle",
        description=(
            "A feedback producer's delivery lifecycle does not satisfy the "
            "consumer's declared FeedbackContract."
        ),
    ),
)
