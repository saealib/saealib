"""Changes applied at a component replacement's commit point.

``Optimizer.set_*(..., transition=...)`` attaches a :class:`Transition` to a
replacement.  The transition runs at the commit point of that replacement,
after the candidate configuration passes static validation and before the new
binding version is published, so a transition that raises leaves the published
configuration, the binding version, and the state's collections unchanged.

Capabilities:

- Supported: attaching a transition to a component setter, composing
  transitions with :class:`Chain`, and staging column additions and drops for
  the population, archive, and Pareto archive of the running state.
- Extensible toward: evaluation requests from a transition.  ``apply`` is a
  generator; a yielded value is rejected at the commit point with
  :class:`~saealib.exceptions.ConfigurationError` because evaluation requested
  by a transition is not available.
- Out of scope: replacing components or writing component parameters from a
  transition, and declaring properties other than population columns.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final

import numpy as np

from saealib.exceptions import ValidationError
from saealib.identity import PopulationAttribute
from saealib.population import Population
from saealib.population.population import _same_column_spec

if TYPE_CHECKING:
    from saealib.context import OptimizationState
    from saealib.problem import Problem

__all__ = ["Chain", "Transition", "TransitionContext", "chain"]

_ADD: Final = "add"
_DROP: Final = "drop"
_OBJECT_DTYPE: Final = np.dtype(object)
_SNAPSHOT_FIELDS: Final = (
    "_data",
    "_schema",
    "_structure_version",
    "_value_version",
    "_cache",
    "_dense_genomes_view_cache",
)


def _check_staged_spec(
    spec: PopulationAttribute, existing: PopulationAttribute | None
) -> None:
    """Raise unless one staged column can be added with *spec*.

    Applies the storage rules of ``Population.add_columns()`` without
    creating a column, so a staged operation that the collection would refuse
    is found before the first mutation.
    """
    name = spec.name
    try:
        dtype = np.dtype(spec.dtype)
    except TypeError as exc:
        raise ValidationError(
            f"add_columns() requires a NumPy dtype for {name!r}"
        ) from exc
    if dtype == _OBJECT_DTYPE:
        raise ValidationError(f"add_columns() rejects object-dtype column {name!r}")
    if not isinstance(spec.shape, tuple) or any(
        isinstance(size, bool) or not isinstance(size, (int, np.integer)) or size < 0
        for size in spec.shape
    ):
        raise ValidationError(
            f"add_columns() requires a tuple of non-negative column shape sizes "
            f"for {name!r}"
        )
    if existing is not None and not _same_column_spec(existing, spec):
        raise ValidationError(
            "Conflicting PopulationAttribute definitions for "
            f"{name!r}: {existing} vs {spec}"
        )


class StagedColumns:
    """Column writes a transition records for one state collection.

    The operations are recorded, not applied: the commit driver applies them
    after post-validation and before publication, so a rejected replacement
    leaves the collection untouched.  Applying is atomic: every operation is
    validated before the first mutation, and the stored columns are restored
    when one of them still fails.  The collection keeps its object identity;
    only its schema changes when the operations are applied.

    Parameters
    ----------
    collection : Population
        The population, archive, or Pareto archive the operations target.
    """

    def __init__(self, collection: Population) -> None:
        self._collection = collection
        self._operations: list[tuple[str, tuple[Any, ...]]] = []
        self._applied_snapshot: list[tuple[str, Any]] | None = None

    @property
    def schema(self) -> Mapping[str, PopulationAttribute]:
        """Return the collection's current schema, before staged writes."""
        return self._collection.schema

    def add_columns(self, specs: Iterable[PopulationAttribute]) -> tuple[str, ...]:
        """Stage column additions filled with each specification's default.

        Parameters
        ----------
        specs : iterable of PopulationAttribute
            Column definitions to add, in order.

        Returns
        -------
        tuple of str
            Names of the staged columns, in the given order.

        Raises
        ------
        ValidationError
            If an item is not a ``PopulationAttribute`` or a name is empty or
            repeated in this call.
        """
        staged: list[PopulationAttribute] = []
        names: set[str] = set()
        for spec in specs:
            if not isinstance(spec, PopulationAttribute):
                raise ValidationError(
                    "add_columns() expects PopulationAttribute definitions"
                )
            if not isinstance(spec.name, str) or not spec.name:
                raise ValidationError("a staged column requires a name")
            if spec.name in names:
                raise ValidationError(f"staged column {spec.name!r} is repeated")
            names.add(spec.name)
            staged.append(spec)
        self._operations.append((_ADD, tuple(staged)))
        return tuple(spec.name for spec in staged)

    def drop_columns(self, names: Iterable[str]) -> tuple[str, ...]:
        """Stage column removals.

        Parameters
        ----------
        names : iterable of str
            Column names to remove, in order.

        Returns
        -------
        tuple of str
            Names of the staged removals, in the given order.

        Raises
        ------
        ValidationError
            If a name is not a non-empty string or is repeated in this call.
        """
        staged: list[str] = []
        seen: set[str] = set()
        for name in names:
            if not isinstance(name, str) or not name:
                raise ValidationError("drop_columns() requires a column name")
            if name in seen:
                raise ValidationError(f"staged column {name!r} is repeated")
            seen.add(name)
            staged.append(name)
        self._operations.append((_DROP, tuple(staged)))
        return tuple(staged)

    def added_columns(self) -> frozenset[str]:
        """Return the names of the columns the staged operations add."""
        return frozenset(
            spec.name
            for kind, payload in self._operations
            if kind == _ADD
            for spec in payload
        )

    def planned_schema(self) -> dict[str, PopulationAttribute]:
        """Return the schema the staged operations would leave behind."""
        schema = dict(self._collection.schema)
        for kind, payload in self._operations:
            if kind == _ADD:
                for spec in payload:
                    schema[spec.name] = spec
            else:
                for name in payload:
                    schema.pop(name, None)
        return schema

    def validate(self) -> None:
        """Raise unless every staged operation can be applied to the collection.

        Each operation is checked against the schema the earlier ones would
        leave behind, so a deterministic rejection happens before any of them
        touches the collection.

        Raises
        ------
        ValidationError
            If a staged column cannot be added with its specification, a
            staged drop names an unknown or reserved column, or a later
            operation would act on a column an earlier one removes.
        """
        schema = dict(self._collection.schema)
        reserved = self._collection._reserved_columns()
        for kind, payload in self._operations:
            if kind == _ADD:
                for spec in payload:
                    _check_staged_spec(spec, schema.get(spec.name))
                    schema[spec.name] = spec
                continue
            unknown = [name for name in payload if name not in schema]
            if unknown:
                raise ValidationError(
                    "Cannot drop unknown column(s): " + ", ".join(unknown)
                )
            rejected = sorted(name for name in payload if name in reserved)
            if rejected:
                raise ValidationError(
                    "Cannot drop reserved column(s): " + ", ".join(rejected)
                )
            for name in payload:
                schema.pop(name, None)

    def apply(self) -> None:
        """Apply the staged operations to the collection, atomically.

        Validates every operation before the first mutation and restores the
        stored columns when applying one of them still fails, so a failed
        apply leaves the collection's columns, values, and versions as they
        were.  A successful apply keeps the pre-apply snapshot for
        :meth:`rollback`.
        """
        self.validate()
        snapshot = self._snapshot()
        try:
            for kind, payload in self._operations:
                if kind == _ADD:
                    self._collection.add_columns(payload)
                else:
                    self._collection.drop_columns(payload)
        except BaseException:
            self._restore(snapshot)
            raise
        self._applied_snapshot = snapshot
        self._operations = []

    def rollback(self) -> None:
        """Restore the collection to its state before the last successful apply."""
        snapshot = self._applied_snapshot
        if snapshot is None:
            return
        self._applied_snapshot = None
        self._restore(snapshot)

    def _snapshot(self) -> list[tuple[str, Any]]:
        """Return the collection storage the staged operations may change."""
        snapshot: list[tuple[str, Any]] = []
        for name in _SNAPSHOT_FIELDS:
            value = self._collection.__dict__.get(name)
            snapshot.append((name, dict(value) if isinstance(value, dict) else value))
        return snapshot

    def _restore(self, snapshot: list[tuple[str, Any]]) -> None:
        """Put the collection storage back to a snapshot."""
        for name, value in snapshot:
            object.__setattr__(self._collection, name, value)

    def discard(self) -> None:
        """Drop the staged operations without touching the collection."""
        self._operations = []

    def __repr__(self) -> str:
        return f"StagedColumns({type(self._collection).__name__})"


@dataclass(frozen=True, kw_only=True)
class TransitionContext:
    """Inputs and staged writes of one transition run.

    Attributes
    ----------
    state : OptimizationState or None
        The running state the replacement happens in.  ``None`` at a commit
        point outside a run, for example a recompile before the first step.
    problem : Problem or None
        The problem the run optimizes.
    rng : numpy.random.Generator or None
        The run's random generator; ``None`` when there is no run state.
    outgoing : mapping of str to object
        Components of the published binding version, by slot.
    incoming : mapping of str to object
        Components of the version being committed, by slot.
    replaced_slots : tuple of str
        Slots whose component changes at this commit point, in binding order.
    write_targets : frozenset of str
        Names of the collections this transition may stage writes for.
    population, archive, pareto_archive : StagedColumns or None
        Staged column writes of the three state collections; ``None`` when the
        collection or the run state is absent.
    """

    state: OptimizationState | None
    problem: Problem | None
    rng: np.random.Generator | None
    outgoing: Mapping[str, object]
    incoming: Mapping[str, object]
    replaced_slots: tuple[str, ...]
    write_targets: frozenset[str]
    population: StagedColumns | None = None
    archive: StagedColumns | None = None
    pareto_archive: StagedColumns | None = None


class Transition:
    """A change applied at a component replacement's commit point.

    ``apply`` is a generator: yield nothing to change the run, and record
    collection writes on the context.  A yielded value requests evaluation,
    which is not available at a commit point and is rejected with
    :class:`~saealib.exceptions.ConfigurationError`.

    A transition runs once per replacement, after the candidate configuration
    passes static validation and before its binding version is published.  A
    transition that raises stops the run at that commit point; the published
    configuration and the binding version keep their values, and issuing the
    same replacement again runs the transition again.

    Examples
    --------
    >>> from saealib.identity import PopulationAttribute
    >>> class AddScore(Transition):
    ...     def produced_columns(self, new_bindings):
    ...         return frozenset({"score"})
    ...     def apply(self, tctx):
    ...         spec = PopulationAttribute("score", np.float64, (), np.nan)
    ...         for staged in (tctx.population, tctx.archive):
    ...             if staged is not None:
    ...                 staged.add_columns([spec])
    ...         yield from ()
    """

    def apply(self, tctx: TransitionContext) -> Iterator[object]:
        """Perform the change, recording staged writes on *tctx*.

        Parameters
        ----------
        tctx : TransitionContext
            Inputs of this commit point and the staged writes to record.

        Returns
        -------
        Iterator of object
            Requests for the runtime.  Yielding a value raises
            ``ConfigurationError`` at the commit point.

        Notes
        -----
        The base implementation does nothing.  Implementations are generators
        so that later evaluation requests can be yielded from the same method.
        """
        return iter(())

    def produced_columns(
        self, new_bindings: Mapping[str, object]
    ) -> frozenset[str] | None:
        """Return the population columns this transition produces.

        Parameters
        ----------
        new_bindings : mapping of str to object
            Components of the version being committed, by slot.

        Returns
        -------
        frozenset of str or None
            Columns the transition adds, or ``None`` when it declares nothing.
            A transition that declares nothing cannot be verified, so the
            commit point reports a warning and relies on post-validation.
        """
        return None

    @classmethod
    def from_function(
        cls,
        function: Callable[[TransitionContext], Iterator[object] | None],
    ) -> Transition:
        """Wrap a function as a transition that declares no columns.

        Parameters
        ----------
        function : callable
            Called with the context, returning an iterator of runtime requests
            or ``None``.

        Returns
        -------
        Transition
            A transition whose ``apply`` calls *function*.

        Raises
        ------
        ValidationError
            If *function* is not callable.
        """
        if not callable(function):
            raise ValidationError("a transition function must be callable")
        return _FunctionTransition(function)


class _FunctionTransition(Transition):
    def __init__(
        self, function: Callable[[TransitionContext], Iterator[object] | None]
    ) -> None:
        self._function = function

    def apply(self, tctx: TransitionContext) -> Iterator[object]:
        steps = self._function(tctx)
        if steps is None:
            return
        yield from steps

    def __repr__(self) -> str:
        name = getattr(self._function, "__name__", type(self._function).__name__)
        return f"Transition.from_function({name})"


class Chain(Transition):
    """Several transitions applied in order at one commit point.

    Parameters
    ----------
    parts : Transition
        Transitions to apply, in the given order.

    Raises
    ------
    ValidationError
        If a part is not a :class:`Transition`.
    """

    def __init__(self, *parts: Transition) -> None:
        checked = tuple(parts)
        for part in checked:
            if not isinstance(part, Transition):
                raise ValidationError("Chain parts must be Transition instances")
        self._parts = checked

    @property
    def parts(self) -> tuple[Transition, ...]:
        """Return the chained transitions, in application order."""
        return self._parts

    def apply(self, tctx: TransitionContext) -> Iterator[object]:
        """Apply every part in order, forwarding their runtime requests."""
        for part in self._parts:
            yield from _iter_applied(part, tctx)

    def produced_columns(
        self, new_bindings: Mapping[str, object]
    ) -> frozenset[str] | None:
        """Return the union of the parts' columns, or ``None`` if any declares none."""
        declared: set[str] = set()
        for part in self._parts:
            columns = part.produced_columns(new_bindings)
            if columns is None:
                return None
            declared.update(columns)
        return frozenset(declared)

    def __repr__(self) -> str:
        return f"Chain({', '.join(repr(part) for part in self._parts)})"


def chain(*parts: Transition) -> Chain:
    """Return a transition that applies *parts* in order.

    Parameters
    ----------
    parts : Transition
        Transitions to apply, in the given order.

    Returns
    -------
    Chain
        The composed transition.
    """
    return Chain(*parts)


def _iter_applied(transition: Transition, tctx: TransitionContext) -> Iterator[object]:
    """Return what *transition* asks the runtime to do, if anything.

    Parameters
    ----------
    transition : Transition
        The transition to run.
    tctx : TransitionContext
        The context of the commit point.

    Returns
    -------
    Iterator of object
        Values yielded by the transition's ``apply``.

    Raises
    ------
    ValidationError
        If ``apply`` returns neither an iterator nor ``None``.
    """
    steps = transition.apply(tctx)
    if steps is None:
        return
    try:
        iterator = iter(steps)
    except TypeError as error:
        raise ValidationError(
            "Transition.apply must return an iterator or None"
        ) from error
    yield from iterator
