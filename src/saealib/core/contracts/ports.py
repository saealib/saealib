from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal, TypeAlias

from saealib.core.contracts.data import (
    DATA_SPEC_KINDS,
    DataSpec,
    DataSpecKind,
    is_data_spec_compatible,
)
from saealib.core.contracts.schema import (
    Substitution,
    UnificationResult,
    unify_data_specs,
)
from saealib.core.contracts.vocabulary import (
    Vocabulary,
    VocabularyDescriptor,
    validate_name,
)
from saealib.exceptions import ValidationError

__all__ = [
    "CARDINALITIES",
    "MANY",
    "ONE",
    "OPTIONAL",
    "SERVICE_VOCABULARY",
    "Cardinality",
    "PortCompatibility",
    "PortContract",
    "PortDirection",
    "PortSpec",
    "ServiceDescriptor",
    "ServiceProvider",
    "ServiceRequirement",
    "cardinality_satisfies",
    "check_port_compatibility",
    "definite_port_incompatibility",
    "validate_port_contract_directions",
]

Cardinality: TypeAlias = str
ONE: Cardinality = "ONE"
MANY: Cardinality = "MANY"
OPTIONAL: Cardinality = "OPTIONAL"


class PortDirection(str, Enum):
    """The direction in which a port carries data."""

    INPUT = "input"
    OUTPUT = "output"


ServiceProvider: TypeAlias = Literal["space", "problem"]


@dataclass(frozen=True, kw_only=True)
class ServiceDescriptor(VocabularyDescriptor):
    """Metadata for a named service and the object that provides it."""

    provider: ServiceProvider

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.provider not in {"space", "problem"}:
            raise ValidationError(
                "Service providers must be either 'space' or 'problem'"
            )


CARDINALITIES: Vocabulary[VocabularyDescriptor] = Vocabulary()
for _name, _description in (
    (ONE, "Exactly one item per activation."),
    (MANY, "A finite batch known in full at activation."),
    (OPTIONAL, "Zero or one item."),
):
    CARDINALITIES.register(
        _name,
        VocabularyDescriptor(name=_name, description=_description),
    )


SERVICE_VOCABULARY: Vocabulary[ServiceDescriptor] = Vocabulary()
for _name, _description in (
    (
        "SamplingService",
        "draw genomes (initialization, restart, random immigrants)",
    ),
    ("ValidationService", "is this genome well-formed for this space"),
    ("CloneService", "produce an independent copy"),
    (
        "FingerprintService",
        "an exact, canonical, hashable identity for a genome",
    ),
    ("EquivalenceService", "configurable approximate duplicate matching"),
    ("GenomeCodec", "genome ↔ persistable primitives"),
    (
        "DenseNumericView",
        "zero-copy dense numeric access, where it exists",
    ),
    (
        "FeatureEncoder",
        "explicit genome-to-feature conversion for surrogate consumers",
    ),
    ("BoundsService", "per-variable bounds, where they exist"),
    ("DistanceService", "distance between genomes"),
    ("ComparisonService", "order candidates by their objective values"),
):
    SERVICE_VOCABULARY.register(
        _name,
        ServiceDescriptor(
            name=_name,
            description=_description,
            provider="problem" if _name == "ComparisonService" else "space",
        ),
    )


def cardinality_satisfies(provided: Cardinality, required: Cardinality) -> bool:
    """Return whether a produced cardinality satisfies a consumer."""
    if provided not in CARDINALITIES or required not in CARDINALITIES:
        return False
    if provided == required:
        return True
    return provided == ONE and required in {MANY, OPTIONAL}


@dataclass(frozen=True, kw_only=True)
class ServiceRequirement:
    """A named service required by one port."""

    name: str

    def __post_init__(self) -> None:
        validate_name(self.name)


@dataclass(frozen=True, kw_only=True)
class PortSpec:
    """A named, directional data contract.

    ``cardinality`` and ``optional`` are independent axes despite the shared
    vocabulary. ``cardinality`` describes how many items arrive per
    activation once the port is connected (membership in ``CARDINALITIES``,
    including ``"OPTIONAL"``, is checked by the compiler's contract
    diagnostics, not here). ``optional`` describes whether the port may have
    no upstream producer at all (checked by the dataflow compilation rule).
    Any combination of the two is valid, e.g. a port with
    ``cardinality=ONE, optional=True`` requires exactly one item whenever it
    is connected, but the connection itself is not required.
    """

    name: str
    direction: PortDirection
    data: DataSpec
    cardinality: Cardinality
    required_services: tuple[ServiceRequirement, ...] = ()
    optional: bool = False

    def __post_init__(self) -> None:
        validate_name(self.name)
        if not isinstance(self.direction, PortDirection):
            raise ValidationError("Port direction must be a PortDirection")
        if not isinstance(self.data, DataSpec):
            raise ValidationError("Port data must be a DataSpec")
        if not isinstance(self.cardinality, str) or not self.cardinality:
            raise ValidationError("Port cardinality must be a non-empty name")
        validate_name(self.cardinality)
        required_services = tuple(self.required_services)
        if any(
            not isinstance(service, ServiceRequirement) for service in required_services
        ):
            raise ValidationError(
                "Port required_services must contain ServiceRequirement values"
            )
        object.__setattr__(self, "required_services", required_services)


@dataclass(frozen=True, kw_only=True)
class PortContract:
    """Input and output ports for one component role."""

    inputs: tuple[PortSpec, ...] = ()
    outputs: tuple[PortSpec, ...] = ()

    def __post_init__(self) -> None:
        inputs = tuple(self.inputs)
        outputs = tuple(self.outputs)
        for ports in (inputs, outputs):
            if any(not isinstance(port, PortSpec) for port in ports):
                raise ValidationError("Port tuples must contain PortSpec values")
            names = [port.name for port in ports]
            if len(names) != len(set(names)):
                raise ValidationError("Port names must be unique per direction")
        object.__setattr__(self, "inputs", inputs)
        object.__setattr__(self, "outputs", outputs)


def validate_port_contract_directions(contract: PortContract) -> None:
    """Validate both direction-specific tuples in a port contract."""
    for ports, expected in (
        (contract.inputs, PortDirection.INPUT),
        (contract.outputs, PortDirection.OUTPUT),
    ):
        for port in ports:
            if port.direction is not expected:
                raise ValidationError(
                    f"Port {port.name!r} is not an {expected.value} port"
                )


@dataclass(frozen=True, kw_only=True)
class PortCompatibility:
    """Result of the port checks implemented in this contract layer."""

    kind_ok: bool
    cardinality_ok: bool
    direction_ok: bool
    schema: UnificationResult
    unknown_kinds: tuple[str, ...] = ()
    unknown_cardinalities: tuple[str, ...] = ()
    unchecked: tuple[str, ...] = ()

    @property
    def schema_ok(self) -> bool:
        """Return whether schema bindings unified without unknown variables."""
        return self.schema.unified

    @property
    def compatible(self) -> bool:
        """Return whether every performed compatibility check passed."""
        return (
            self.kind_ok
            and self.cardinality_ok
            and self.direction_ok
            and self.schema_ok
        )


def definite_port_incompatibility(compatibility: PortCompatibility) -> bool:
    """Check whether a registered-value check already failed.

    ``unknown_kinds`` and ``unknown_cardinalities`` make their own axis
    unverifiable, but a disagreement between registered values on another
    axis is evidence of a defect regardless of unregistered names.
    """
    return (
        (not compatibility.kind_ok and not compatibility.unknown_kinds)
        or (
            not compatibility.cardinality_ok and not compatibility.unknown_cardinalities
        )
        or not compatibility.direction_ok
        or bool(compatibility.schema.conflicts)
    )


def check_port_compatibility(
    producer: PortSpec,
    consumer: PortSpec,
    *,
    data_registry: Vocabulary[DataSpecKind] | None = None,
    substitution: Substitution | None = None,
) -> PortCompatibility:
    """Check port kinds, schema bindings, and cardinalities."""
    registry = DATA_SPEC_KINDS if data_registry is None else data_registry
    kind_names = (producer.data.kind, consumer.data.kind)
    unknown_kinds = tuple(
        dict.fromkeys(kind for kind in kind_names if registry.get(kind) is None)
    )
    cardinality_names = (producer.cardinality, consumer.cardinality)
    unknown_cardinalities = tuple(
        dict.fromkeys(
            cardinality
            for cardinality in cardinality_names
            if cardinality not in CARDINALITIES
        )
    )
    schema = unify_data_specs(
        producer.data,
        consumer.data,
        substitution=substitution,
    )
    return PortCompatibility(
        kind_ok=not unknown_kinds
        and is_data_spec_compatible(
            producer.data,
            consumer.data,
            registry=registry,
        ),
        cardinality_ok=not unknown_cardinalities
        and cardinality_satisfies(producer.cardinality, consumer.cardinality),
        direction_ok=(
            producer.direction is PortDirection.OUTPUT
            and consumer.direction is PortDirection.INPUT
        ),
        schema=schema,
        unknown_kinds=unknown_kinds,
        unknown_cardinalities=unknown_cardinalities,
        unchecked=("services",),
    )
