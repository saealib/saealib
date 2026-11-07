"""Declarations for the values passed through a Stage."""

from __future__ import annotations

from dataclasses import dataclass

from saealib.core.contracts.data import DataSpec, is_data_spec_compatible
from saealib.core.state.keys import StateKey
from saealib.exceptions import ValidationError

__all__ = [
    "CallInput",
    "CallOutput",
    "ComponentCall",
    "FlowEdge",
    "StageFlowSummary",
    "StateValue",
    "Transform",
]


@dataclass(frozen=True, kw_only=True)
class ComponentCall:
    """Identify one component invocation made by a Stage."""

    name: str
    role: str
    attribute: str

    def __post_init__(self) -> None:
        if any(
            not isinstance(value, str) or not value
            for value in (self.name, self.role, self.attribute)
        ):
            raise ValidationError("ComponentCall fields must be non-empty strings")


@dataclass(frozen=True, kw_only=True)
class CallInput:
    """Identify one input port of a declared component invocation."""

    call: ComponentCall
    port: str

    def __post_init__(self) -> None:
        if not isinstance(self.call, ComponentCall):
            raise ValidationError("CallInput call must be a ComponentCall")
        if not isinstance(self.port, str) or not self.port:
            raise ValidationError("CallInput port must be a non-empty string")


@dataclass(frozen=True, kw_only=True)
class CallOutput:
    """Identify one output port of a declared component invocation."""

    call: ComponentCall
    port: str

    def __post_init__(self) -> None:
        if not isinstance(self.call, ComponentCall):
            raise ValidationError("CallOutput call must be a ComponentCall")
        if not isinstance(self.port, str) or not self.port:
            raise ValidationError("CallOutput port must be a non-empty string")


@dataclass(frozen=True, kw_only=True)
class StateValue:
    """Identify a value read from or written to optimization state."""

    key: StateKey[object]

    def __post_init__(self) -> None:
        if not isinstance(self.key, StateKey):
            raise ValidationError("StateValue key must be a StateKey")


@dataclass(frozen=True, kw_only=True)
class Transform:
    """Name one compiler-recognized conversion in a data connection.

    Accepted names are ``encode_features``, ``extract_population_genomes``,
    ``apply_observations``, ``extract_evaluation_observations``,
    ``select_requested_candidates``, and ``extract_genomes``.
    """

    name: str

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValidationError("Transform name must be a non-empty string")


@dataclass(frozen=True, kw_only=True)
class FlowEdge:
    """Declare a value connection between component calls and Stage state."""

    source: CallOutput | StateValue
    target: CallInput | StateValue
    transforms: tuple[Transform, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.source, (CallOutput, StateValue)):
            raise ValidationError("FlowEdge source must be CallOutput or StateValue")
        if not isinstance(self.target, (CallInput, StateValue)):
            raise ValidationError("FlowEdge target must be CallInput or StateValue")
        transforms = tuple(self.transforms)
        if any(not isinstance(item, Transform) for item in transforms):
            raise ValidationError("FlowEdge transforms must contain Transform values")
        object.__setattr__(self, "transforms", transforms)


@dataclass(frozen=True, kw_only=True)
class StageFlowSummary:
    """Describe a Stage's component calls and declared value connections."""

    calls: tuple[ComponentCall, ...] = ()
    edges: tuple[FlowEdge, ...] = ()

    def __post_init__(self) -> None:
        calls = tuple(self.calls)
        edges = tuple(self.edges)
        if any(not isinstance(call, ComponentCall) for call in calls):
            raise ValidationError(
                "StageFlowSummary calls must contain ComponentCall values"
            )
        if any(not isinstance(edge, FlowEdge) for edge in edges):
            raise ValidationError("StageFlowSummary edges must contain FlowEdge values")
        names = [call.name for call in calls]
        if len(names) != len(set(names)):
            raise ValidationError("StageFlowSummary call names must be unique")
        calls_by_name = {call.name: call for call in calls}
        for edge in edges:
            for endpoint in (edge.source, edge.target):
                if (
                    isinstance(endpoint, (CallInput, CallOutput))
                    and calls_by_name.get(endpoint.call.name) != endpoint.call
                ):
                    raise ValidationError(
                        "StageFlowSummary edge refers to an undeclared call"
                    )
        object.__setattr__(self, "calls", calls)
        object.__setattr__(self, "edges", edges)


_TRANSFORM_TRANSITIONS: dict[str, tuple[DataSpec, DataSpec, frozenset[str]]] = {
    "encode_features": (
        DataSpec(kind="Population"),
        DataSpec(kind="FeatureBatch"),
        frozenset(),
    ),
    "extract_population_genomes": (
        DataSpec(kind="Population"),
        DataSpec(kind="GenomeBatch"),
        frozenset({"representation"}),
    ),
    "apply_observations": (
        DataSpec(kind="ObservationBatch"),
        DataSpec(kind="Population"),
        frozenset(),
    ),
    "extract_evaluation_observations": (
        DataSpec(kind="Population"),
        DataSpec(kind="ObservationBatch"),
        frozenset(),
    ),
    "select_requested_candidates": (
        DataSpec(kind="EvaluationRequestBatch"),
        DataSpec(kind="CandidateSelection"),
        frozenset({"representation"}),
    ),
    "extract_genomes": (
        DataSpec(kind="CandidateSelection"),
        DataSpec(kind="GenomeBatch"),
        frozenset({"representation"}),
    ),
}


def _apply_transforms(
    data: DataSpec, transforms: tuple[Transform, ...]
) -> tuple[DataSpec | None, str | None]:
    """Return the known output type, or an explanation for an invalid chain."""
    current = data
    for transform in transforms:
        transition = _TRANSFORM_TRANSITIONS.get(transform.name)
        if transition is None:
            return None, f"unknown transform {transform.name!r}"
        required, produced, preserved_bindings = transition
        if not is_data_spec_compatible(current, required):
            return (
                None,
                f"transform {transform.name!r} requires {required.kind}, "
                f"received {current.kind}",
            )
        current = DataSpec(
            kind=produced.kind,
            bindings={
                name: value
                for name, value in current.bindings.items()
                if name in preserved_bindings
            },
        )
    return current, None


def _transform_input_data(
    target_data: DataSpec, transforms: tuple[Transform, ...]
) -> DataSpec:
    """Return the declared input kind for a known transform chain."""
    if not transforms:
        return target_data
    inferred = target_data
    for transform in reversed(transforms):
        transition = _TRANSFORM_TRANSITIONS.get(transform.name)
        if transition is None:
            continue
        required, _produced, preserved_bindings = transition
        inferred = DataSpec(
            kind=required.kind,
            bindings={
                name: value
                for name, value in inferred.bindings.items()
                if name in preserved_bindings
            },
        )
    return inferred
