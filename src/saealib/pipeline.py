"""Stage compatibility surface and the structural pipeline DSL."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeAlias, TypeGuard

from saealib.core.compiler.regions import (
    Condition,
    _is_component,
    _is_condition,
    _register_structural_types,
    _structural_name,
    _structural_stages,
)
from saealib.core.component import Component
from saealib.core.contracts import ComponentContract, StateContract
from saealib.core.contracts.roles import ROLES
from saealib.exceptions import ValidationError

if TYPE_CHECKING:
    from saealib.context import OptimizationState

__all__ = [
    "Branch",
    "ComponentRef",
    "Condition",
    "Loop",
    "Pipeline",
    "PipelineEntry",
    "Repeat",
    "Stage",
]

#: The Optimizer slot each stage-consumed role is served from when a
#: declaration does not name one explicitly.  Slots not listed here have no
#: well-known binding and must be named on the reference.
_ROLE_WELL_KNOWN_SLOT: Mapping[str, str] = {
    "proposer": "algorithm",
    "feedback_consumer": "algorithm",
    "fitter": "surrogate_manager",
    "predictor": "surrogate_manager",
    "acquisition": "acquisition",
    "evaluation_planner": "evaluation_planner",
    "feedback_builder": "feedback_builder",
    "evaluator": "evaluator",
}


@dataclass(frozen=True, slots=True)
class ComponentRef:
    """
    A declared Stage's ``(slot, role)`` reference to a component.

    Declaration Stages hold this instead of a component so that the same
    declaration can be prepared for many plans.  The reference is resolved
    from the binding snapshot of the plan that executes the declaration,
    producing a resolved Stage owned by that plan; the declaration itself is
    never mutated.

    Parameters
    ----------
    role : str
        Registered ``ROLES`` name the referenced component plays.
    slot : str, optional
        Optimizer slot the component is read from.  Defaults to the role's
        well-known slot (for example ``"algorithm"`` for ``"proposer"``).

    Raises
    ------
    ValidationError
        If *role* is not registered, or *slot* is omitted for a role without
        a well-known slot.

    Notes
    -----
    References are resolved in a Stage's own attributes and in the child
    Stages listed in its ``stages``.  A Stage held in any other attribute
    keeps its references unresolved, and preparing the plan fails.
    """

    role: str
    slot: str = ""

    def __post_init__(self) -> None:
        if ROLES.get(self.role) is None:
            raise ValidationError(
                f"{self.role!r} is not a registered role; use one of the "
                "names declared in saealib ROLES."
            )
        if not self.slot:
            slot = _ROLE_WELL_KNOWN_SLOT.get(self.role)
            if slot is None:
                raise ValidationError(
                    f"role {self.role!r} has no well-known slot; record the "
                    "reference with an explicit slot."
                )
            object.__setattr__(self, "slot", slot)


class Stage(ABC):
    """
    Abstract base class for a compatibility execution step.

    A Stage receives an :class:`~saealib.context.OptimizationState`, performs
    one well-defined operation, and returns a (possibly new) state.  Structured
    graph execution does not accept this boundary.

    Attributes
    ----------
    name : str
        Short machine-readable identifier used for lookup via
        ``pipeline["name"]``.
    label : str
        Human-readable description of what this stage does.
    notation : str
        LaTeX math expression used by :meth:`to_pseudocode`.
    stages : list[Stage]
        Sub-pipeline for composite stages.  Empty list for leaf stages.
    """

    name: str = ""
    label: str = ""
    notation: str = ""
    stages: list[Stage]
    _execution_mode = "optimization_state"
    _saealib_stage_boundary = True

    #: ``(attribute-name, reference)`` entries a declaration Stage fills when
    #: its references are resolved from a binding snapshot.
    _refs: tuple[tuple[str, ComponentRef], ...] = ()

    def __init__(
        self,
        *,
        name: str = "",
        label: str = "",
        notation: str = "",
    ) -> None:
        """
        Initialize base Stage attributes.

        Parameters
        ----------
        name : str, optional
            Override the class-level ``name``.
        label : str, optional
            Override the class-level ``label``.
        notation : str, optional
            Override the class-level ``notation``.
        """
        if name:
            self.name = name
        if label:
            self.label = label
        if notation:
            self.notation = notation
        self.stages: list[Stage] = []

    @abstractmethod
    def execute(self, state: OptimizationState) -> OptimizationState:
        """
        Execute this stage and return the updated state.

        Parameters
        ----------
        state : OptimizationState
            Current optimization state.

        Returns
        -------
        OptimizationState
            Updated state.
        """

    def to_pseudocode(self, *, expand: bool = False, indent: int = 0) -> str:
        r"""
        Render this stage as a LaTeX algorithmic line.

        Parameters
        ----------
        expand : bool
            If True and this stage has sub-stages, recursively expand them.
        indent : int
            Current indentation level (number of ``\State`` indent steps).

        Returns
        -------
        str
            LaTeX algorithmic fragment.
        """
        prefix = "  " * indent
        if expand and self.stages:
            inner = "\n".join(
                s.to_pseudocode(expand=True, indent=indent + 1) for s in self.stages
            )
            label = self.label or self.name or type(self).__name__
            return f"{prefix}\\Comment{{{label}}}\n{inner}"
        notation = self.notation or self.label or self.name or type(self).__name__
        return f"{prefix}\\State {notation}"

    def to_text(self, *, expand: bool = False, indent: int = 0) -> str:
        """Render this stage as a plain-text algorithmic line."""
        prefix = "  " * indent
        if expand and self.stages:
            inner = "\n".join(
                s.to_text(expand=True, indent=indent + 1) for s in self.stages
            )
            label = self.label or self.name or type(self).__name__
            return f"{prefix}# {label}\n{inner}"
        notation = self.notation or self.label or self.name or type(self).__name__
        return f"{prefix}{notation}"

    def contract(self) -> ComponentContract:
        """Return this stage's structural and direct state-access contract."""
        # A Stage that does not override contract() keeps its state effects
        # private to execute(), so neither direction is enumerable and the
        # compiler can only warn that the effects cannot be verified.
        return ComponentContract(
            state=StateContract(reads_enumerable=False, writes_enumerable=False)
        )

    def _resolve_stage(
        self, bindings: Mapping[str, object], defaults: dict[tuple[str, str], object]
    ) -> Stage:
        """Resolve declaration references from one binding snapshot.

        Declaration Stages override this; a Stage subclass that records
        references without supporting resolution fails loudly instead of
        keeping an unresolved reference in a plan.
        """
        if self._refs:
            raise ValidationError(
                f"{type(self).__name__} records (slot, role) references but "
                "does not support resolution against a binding snapshot."
            )
        return self


def _is_pipeline_entry(value: object) -> TypeGuard[PipelineEntry]:
    return isinstance(value, (Pipeline, Repeat, Loop, Branch, Stage)) or _is_component(
        value
    )


def _find_recursive(stages: Sequence[object], name: str) -> PipelineEntry | None:
    for stage in stages:
        if not _is_pipeline_entry(stage):
            continue
        if _structural_name(stage) == name:
            return stage
        children = _structural_stages(stage)
        if children is not None:
            result = _find_recursive(children, name)
            if result is not None:
                return result
    return None


def _validate_dsl_condition(condition: object) -> None:
    if not _is_condition(condition):
        if not callable(getattr(condition, "contract", None)):
            raise ValidationError("DSL condition must provide contract()")
        raise ValidationError("DSL condition must provide evaluate(context)")
    if not isinstance(condition.contract(), StateContract):
        raise ValidationError("DSL condition contract() must return StateContract")


class _ControlValue:
    """Common identity and pseudocode surface for structural DSL values."""

    name: str = ""
    label: str = ""
    notation: str = ""

    def __init__(self, *, name: str = "", label: str = "", notation: str = "") -> None:
        self.name = name or type(self).__name__.lower()
        self.label = label
        self.notation = notation

    def to_pseudocode(self, *, expand: bool = False, indent: int = 0) -> str:
        """Render this repeat as a lightweight pseudocode line."""
        prefix = "  " * indent
        return f"{prefix}\\State {self.notation or self.label or self.name}"

    def to_text(self, *, expand: bool = False, indent: int = 0) -> str:
        """Render this control value as a plain-text line."""
        prefix = "  " * indent
        return f"{prefix}{self.notation or self.label or self.name}"


class Repeat(_ControlValue):
    """Repeat a structural body a fixed number of times."""

    _structured_kind = "repeat"

    def __init__(self, body: PipelineEntry, count: int, *, name: str = "") -> None:
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValidationError("Repeat count must be a non-negative integer")
        super().__init__(name=name, label=f"Repeat {count} times")
        self.body = body
        self.count = count

    def to_pseudocode(self, *, expand: bool = False, indent: int = 0) -> str:
        """Render this repeat as a lightweight pseudocode line."""
        prefix = "  " * indent
        return f"{prefix}\\State \\Repeat{{{self.count}}}"

    def to_text(self, *, expand: bool = False, indent: int = 0) -> str:
        """Render this repeat as a single plain-text line."""
        prefix = "  " * indent
        return f"{prefix}repeat {self.count} times:"


class Loop(_ControlValue):
    """Repeat a structural body while a condition remains active."""

    _structured_kind = "loop"

    def __init__(
        self, body: PipelineEntry, *, until: Condition, name: str = ""
    ) -> None:
        _validate_dsl_condition(until)
        super().__init__(name=name, label="Loop")
        self.body = body
        self.until = until
        self.condition = until

    def to_pseudocode(self, *, expand: bool = False, indent: int = 0) -> str:
        """Render this loop as a lightweight pseudocode line."""
        prefix = "  " * indent
        return f"{prefix}\\State \\Loop{{{self.name}}}"

    def to_text(self, *, expand: bool = False, indent: int = 0) -> str:
        """Render this loop as a single plain-text line."""
        prefix = "  " * indent
        return f"{prefix}loop until {self.name}:"


class Branch(_ControlValue):
    """Select one of two structural bodies using a condition."""

    _structured_kind = "branch"

    def __init__(
        self,
        condition: Condition,
        *,
        then: PipelineEntry,
        else_: PipelineEntry | None = None,
        name: str = "",
    ) -> None:
        _validate_dsl_condition(condition)
        super().__init__(name=name, label="Branch")
        self.condition = condition
        self.then = then
        self.else_ = else_

    def to_pseudocode(self, *, expand: bool = False, indent: int = 0) -> str:
        """Render this branch as a lightweight pseudocode line."""
        prefix = "  " * indent
        return f"{prefix}\\State \\If{{{self.name}}}"

    def to_text(self, *, expand: bool = False, indent: int = 0) -> str:
        """Render this branch as a single plain-text line."""
        prefix = "  " * indent
        return f"{prefix}if {self.name}:"


class Pipeline:
    """
    An ordered structural DSL sequence lowered to a semantic graph.

    A Pipeline has no state execution path. Its entries may be graph-native
    components, nested pipelines, or structured control values such as
    :class:`Repeat`, :class:`Loop`, and :class:`Branch`.

    Parameters
    ----------
    stages : sequence of PipelineEntry, optional
        Ordered structural entries. ``steps`` is the keyword spelling for the
        same value.
    name : str, optional
        Machine-readable identifier for this pipeline.
    label : str, optional
        Human-readable description.
    notation : str, optional
        LaTeX notation for pseudocode generation.
    """

    def __init__(
        self,
        stages: Sequence[PipelineEntry] | None = None,
        name: str = "",
        label: str = "",
        notation: str = "",
        *,
        steps: Sequence[PipelineEntry] | None = None,
    ) -> None:
        if stages is not None and steps is not None:
            raise TypeError("Provide either positional stages or steps=, not both")
        self.name = name
        self.label = label
        self.notation = notation
        self.steps: list[PipelineEntry] = list(
            steps if steps is not None else (stages or ())
        )
        self.stages = self.steps
        self._validate()

    def _validate(self) -> None:
        for stage in self.steps:
            if not _is_pipeline_entry(stage):
                raise TypeError(
                    f"{stage!r} is not a Stage instance or graph component; "
                    "all elements of a Pipeline must be structural values"
                )

    def __getitem__(self, name: str) -> PipelineEntry:
        """Look up a stage by its ``name`` attribute.

        Parameters
        ----------
        name : str
            The ``name`` of the stage to find.

        Returns
        -------
        structural value

        Raises
        ------
        KeyError
            If no stage with the given name exists.
        """
        for stage in self.stages:
            if _structural_name(stage) == name:
                return stage
        raise KeyError(name)

    def replace(self, name: str, stage: PipelineEntry) -> None:
        """Replace the named entry in the top-level structural sequence.

        Parameters
        ----------
        name : str
            The ``name`` of the stage to replace.
        stage : PipelineEntry
            Replacement structural value.

        Raises
        ------
        KeyError
            If no stage with the given name exists.
        TypeError
            If *stage* is not a structural value.
        """
        if not _is_pipeline_entry(stage):
            raise TypeError(
                f"{stage!r} is not a Stage instance or graph component; "
                "replacement must be a structural value"
            )
        for i, s in enumerate(self.stages):
            if _structural_name(s) == name:
                self.stages[i] = stage
                return
        raise KeyError(name)

    def find(self, name: str, *, recursive: bool = False) -> PipelineEntry:
        """Look up a named structural value, optionally recursively.

        Parameters
        ----------
        name : str
            The ``name`` of the stage to find.
        recursive : bool, optional
            If ``True``, descend into nested pipelines and control bodies.
            Defaults to ``False``.

        Returns
        -------
        structural value

        Raises
        ------
        KeyError
            If no stage with the given name exists.
        """
        if not recursive:
            return self[name]
        result = _find_recursive(self.stages, name)
        if result is not None:
            return result
        raise KeyError(name)

    def __len__(self) -> int:
        return len(self.stages)

    def __iter__(self) -> Iterator[PipelineEntry]:
        return iter(self.stages)

    def __repr__(self) -> str:
        names = ", ".join(type(s).__name__ for s in self.stages)
        if self.name:
            return f"Pipeline(name={self.name!r}, stages=[{names}])"
        return f"Pipeline(stages=[{names}])"

    def to_pseudocode(self, *, expand: bool = False, indent: int = 0) -> str:
        """Render this pipeline as a LaTeX algorithmic block."""
        if expand and self.stages:
            prefix = "  " * indent
            label = self.label or self.name or "Pipeline"
            inner = "\n".join(
                _to_pseudocode(s, expand=True, indent=indent + 1) for s in self.stages
            )
            return f"{prefix}\\Comment{{{label}}}\n{inner}"
        prefix = "  " * indent
        notation = self.notation or self.label or self.name or "Pipeline"
        return f"{prefix}\\State {notation}"

    def to_text(self, *, expand: bool = False, indent: int = 0) -> str:
        """Render this pipeline as a plain-text algorithmic block."""
        if expand and self.stages:
            prefix = "  " * indent
            label = self.label or self.name or "Pipeline"
            inner = "\n".join(
                _to_text(s, expand=True, indent=indent + 1) for s in self.stages
            )
            return f"{prefix}# {label}\n{inner}"
        prefix = "  " * indent
        notation = self.notation or self.label or self.name or "Pipeline"
        return f"{prefix}{notation}"


def _to_pseudocode(value: PipelineEntry, *, expand: bool, indent: int) -> str:
    renderer = getattr(value, "to_pseudocode", None)
    if callable(renderer):
        return renderer(expand=expand, indent=indent)
    prefix = "  " * indent
    name = _structural_name(value, type(value).__name__)
    return f"{prefix}\\State {name}"


def _to_text(value: PipelineEntry, *, expand: bool, indent: int) -> str:
    renderer = getattr(value, "to_text", None)
    if callable(renderer):
        return renderer(expand=expand, indent=indent)
    prefix = "  " * indent
    name = _structural_name(value, type(value).__name__)
    return f"{prefix}{name}"


_register_structural_types(Repeat, Loop, Branch, Stage)


PipelineEntry: TypeAlias = Component | Stage | Pipeline | Repeat | Loop | Branch
