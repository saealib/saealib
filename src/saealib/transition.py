"""Changes applied at a component replacement's commit point.

``Optimizer.set_*(..., transition=...)`` attaches a :class:`Transition` to a
replacement.  The transition runs at the commit point of that replacement,
after the candidate configuration passes static validation and before the new
binding version is published, so a transition that raises leaves the published
configuration, the binding version, and its staged column writes unapplied.
The true evaluations a transition confirmed before it raised stay confirmed,
with their rows, archive entries, evaluation count, and History rows.

Capabilities:

- Supported: attaching a transition to a component setter, composing
  transitions with :class:`Chain`, staging column additions and drops for the
  population, archive, and Pareto archive of the running state, aligning the
  columns of those collections with the components bound after a replacement,
  and requesting true evaluations of population rows with
  :class:`EvaluateRequest` at the commit point of a synchronous or
  asynchronous structured flow.
- Extensible toward: evaluating rows of the archive alone, evaluation requests
  from a sequential plan's transition (a commit point that runs outside a
  structured flow has no evaluation path), and helper transitions for common
  cases such as re-evaluating predicted rows.
- Out of scope: replacing components or writing component parameters from a
  transition, declaring properties other than population columns, and telling
  the algorithm about the evaluations a transition requests.

Named state is not invalidated across processes: a state resumed from a
checkpoint in a new process does not record the flow that produced it, so the
first replacement of that process clears nothing.  A flow with a Stage that
declares no enumerable contract cannot be verified, so its declared reads and
writes are unknown and no named state value is invalidated for that
replacement; a warning reports it.
"""

from __future__ import annotations

import logging
import pickle
from collections.abc import Callable, Generator, Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Final

import numpy as np

from saealib.core.state import RUNTIME_BASE_COLUMNS
from saealib.exceptions import ConfigurationError, ValidationError
from saealib.identity import PopulationAttribute
from saealib.population import Population
from saealib.population.population import _same_column_spec

if TYPE_CHECKING:
    from saealib.context import OptimizationState
    from saealib.problem import Problem

__all__ = [
    "AlignColumns",
    "Chain",
    "EvaluateRequest",
    "EvaluationOutcome",
    "Transition",
    "TransitionContext",
    "chain",
]

logger = logging.getLogger(__name__)

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


class _DeclaredColumns:
    """Column declarations of the components the caller names."""

    @classmethod
    def of(
        cls, bindings: Mapping[str, object], problem: Problem | None
    ) -> dict[str, PopulationAttribute]:
        if problem is None:
            return {}
        from saealib.execution.prepare import BINDING_SLOTS

        declared: dict[str, PopulationAttribute] = {}
        for slot in BINDING_SLOTS:
            component = bindings.get(slot)
            getter = getattr(component, "get_required_attrs", None)
            if not callable(getter):
                continue
            for attribute in getter(problem):
                existing = declared.get(attribute.name)
                if existing is None:
                    declared[attribute.name] = attribute
                    continue
                if _same_attribute(existing, attribute):
                    continue
                raise ConfigurationError(
                    f"Conflicting PopulationAttribute definitions for "
                    f"{attribute.name!r}: {existing} vs {attribute}"
                )
        return declared


def _same_attribute(left: PopulationAttribute, right: PopulationAttribute) -> bool:
    try:
        same_dtype = bool(np.dtype(left.dtype) == np.dtype(right.dtype))
    except TypeError:
        return False
    try:
        same_default = bool(
            np.array_equal(
                np.asarray(left.default),
                np.asarray(right.default),
                equal_nan=True,
            )
        )
    except TypeError:
        same_default = left.default == right.default
    return same_dtype and tuple(left.shape) == tuple(right.shape) and same_default


@dataclass(frozen=True)
class EvaluateRequest:
    """A transition's request to evaluate population rows with the true objective.

    The commit point evaluates the rows of ``state.population`` whose candidate
    IDs this request names, confirms the results, and sends the outcome back
    into the transition's generator.  Rows whose objective values already come
    from a true evaluation are left out, so issuing the same replacement again
    evaluates no row twice and a request with no rows left evaluates nothing.

    Supported: population rows of a synchronous or asynchronous structured
    flow, with the results confirmed at the commit point and never rolled back.
    An asynchronous flow splits a request that exceeds the scheduler's free
    capacity into capacity-sized parts and submits them in batches of the
    scheduler's free capacity, polling after each batch, so six rows at
    ``max_pending=3`` become two requests of three rows; the flow's own
    evaluation requests keep their own split shape.  A synchronous flow
    evaluates the whole request with one ``evaluator.evaluate_batch()`` call.
    Extensible toward: rows that live in other collections.  Out of scope:
    telling the algorithm about the requested evaluations, and adding the
    request to the plan, whose requests and signatures stay unchanged.

    Attributes
    ----------
    candidate_ids : numpy.ndarray
        Candidate IDs of the rows to evaluate, shape ``(n,)``; they must be
        unique and present in ``state.population``.
    metadata : mapping of str to object
        Extra metadata carried with every evaluation request the commit point
        derives from this one.  The commit point adds its own entries to it.
    """

    candidate_ids: np.ndarray
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        ids = np.array(self.candidate_ids, dtype=np.int64, copy=True)
        if ids.ndim != 1:
            raise ValidationError("EvaluateRequest candidate_ids must be 1-D")
        if len(np.unique(ids)) != len(ids):
            raise ValidationError("EvaluateRequest candidate_ids must be unique")
        metadata = dict(self.metadata)
        for key in metadata:
            if not isinstance(key, str):
                raise ValidationError("EvaluateRequest metadata keys must be strings")
        pickle.dumps(metadata)
        ids.flags.writeable = False
        object.__setattr__(self, "candidate_ids", ids)
        object.__setattr__(self, "metadata", metadata)


@dataclass(frozen=True)
class EvaluationOutcome:
    """What the commit point evaluated for one :class:`EvaluateRequest`.

    Attributes
    ----------
    candidate_ids : numpy.ndarray
        Candidate IDs that were truly evaluated, in request order, shape
        ``(n,)``; empty when the request had nothing left to evaluate.
    f, g, cv : numpy.ndarray
        True objective values, shape ``(n, n_obj)``, raw constraint values,
        shape ``(n, n_constraints)``, and constraint violations, shape ``(n,)``
        of the evaluated rows.
    """

    candidate_ids: np.ndarray
    f: np.ndarray
    g: np.ndarray
    cv: np.ndarray


class Transition:
    """A change applied at a component replacement's commit point.

    ``apply`` is a generator: yield nothing to change the run, record
    collection writes on the context, and yield an :class:`EvaluateRequest` to
    evaluate population rows at the commit point.  The runtime evaluates the
    request and sends an :class:`EvaluationOutcome` back into the generator, so
    the transition can continue with the true values.  Any other yielded value
    is rejected with :class:`~saealib.exceptions.ConfigurationError`.

    A transition runs once per replacement, after the candidate configuration
    passes static validation and before its binding version is published.  A
    transition that raises stops the run at that commit point; the published
    configuration and the binding version keep their values, and issuing the
    same replacement again runs the transition again.

    Examples
    --------
    >>> import numpy as np
    >>> from saealib.identity import PopulationAttribute
    >>> class Reevaluate(Transition):
    ...     def apply(self, tctx):
    ...         ids = np.asarray(tctx.state.population.get_array("id"), dtype=np.int64)
    ...         yield EvaluateRequest(ids)
    ...         for staged in (tctx.population, tctx.archive):
    ...             if staged is not None:
    ...                 staged.add_columns(
    ...                     [PopulationAttribute("score", np.float64, (), np.nan)]
    ...                 )
    """

    def apply(self, tctx: TransitionContext) -> Iterator[object]:
        """Perform the change, recording staged writes on *tctx*.

        Parameters
        ----------
        tctx : TransitionContext
            Inputs of this commit point and the staged writes to record.

        Yields
        ------
        EvaluateRequest
            Rows to evaluate with the true objective before the transition
            continues.  The runtime sends an :class:`EvaluationOutcome` back
            into the generator.

        Returns
        -------
        Iterator of EvaluateRequest
            The requests the transition yields, in order; yielding any other
            value raises ``ConfigurationError`` at the commit point.

        Notes
        -----
        The base implementation does nothing.  The results of a request are
        confirmed before the outcome reaches the transition and stay confirmed
        if the transition or a later part of the commit fails, while column
        changes stay staged.
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

    def _declared_columns(
        self,
        new_bindings: Mapping[str, object],
        problem: Problem | None,
    ) -> frozenset[str] | None:
        """Return the declaration the commit point verifies against *problem*."""
        del problem
        return self.produced_columns(new_bindings)

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
        """Return the union of the parts' columns, ignoring parts that declare none."""
        declared: set[str] = set()
        for part in self._parts:
            columns = part.produced_columns(new_bindings)
            if columns is None:
                continue
            declared.update(columns)
        return frozenset(declared) if declared else None

    def _declared_columns(
        self,
        new_bindings: Mapping[str, object],
        problem: Problem | None,
    ) -> frozenset[str] | None:
        declared: set[str] = set()
        for part in self._parts:
            columns = part._declared_columns(new_bindings, problem)
            if columns is None:
                continue
            declared.update(columns)
        return frozenset(declared) if declared else None

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


class AlignColumns(Transition):
    """Align the state collections with the components bound after a replacement.

    Adds every column the incoming components declare and the collection
    lacks, filled with the declared default.  Drops a column only when an
    outgoing component declared it, no remaining component declares it, and
    the initializer did not create it.  The ``x``
    column is always kept and is never added to a collection that has none.
    A state without a base-column record is not changed by a drop, because the
    columns the initializer created are unknown; a warning reports that limit.

    Parameters
    ----------
    problem : Problem or None, optional
        The problem whose dimensions the components' declarations depend on.
        The commit point supplies it when it uses this transition by default.

    Notes
    -----
    Columns are staged through the context's collections, so a rejected
    replacement leaves every collection unchanged.  Only the population,
    archive, and Pareto archive are aligned; offspring collections are not
    touched.
    """

    def __init__(self, problem: Problem | None = None) -> None:
        self._problem = problem

    def produced_columns(
        self, new_bindings: Mapping[str, object]
    ) -> frozenset[str] | None:
        """Return the incoming components' declared column names."""
        return self._declared_columns(new_bindings, self._problem)

    def _declared_columns(
        self,
        new_bindings: Mapping[str, object],
        problem: Problem | None,
    ) -> frozenset[str] | None:
        return frozenset(_DeclaredColumns.of(new_bindings, problem))

    def apply(self, tctx: TransitionContext) -> Iterator[object]:
        """Stage the column additions and drops of this commit point."""
        state = tctx.state
        problem = tctx.problem
        if state is None or problem is None:
            return
        incoming = _DeclaredColumns.of(tctx.incoming, problem)
        outgoing = _DeclaredColumns.of(tctx.outgoing, problem)
        recorded = state.get_state(RUNTIME_BASE_COLUMNS)
        base_columns = (
            frozenset(recorded)
            if isinstance(recorded, (list, tuple, set, frozenset))
            and all(isinstance(name, str) for name in recorded)
            else None
        )
        for staged in (tctx.population, tctx.archive, tctx.pareto_archive):
            if staged is None:
                continue
            schema = staged.planned_schema()
            additions = [
                attribute
                for name, attribute in incoming.items()
                if name not in schema and (name != "x" or "x" in schema)
            ]
            if additions:
                staged.add_columns(additions)
            if base_columns is None:
                if any(name not in incoming for name in outgoing):
                    logger.warning(
                        "Column alignment cannot drop columns because the state "
                        "has no record of the columns the initializer created."
                    )
                continue
            removals = [
                name
                for name in outgoing
                if name not in incoming
                and name not in base_columns
                and name != "x"
                and name in schema
            ]
            if removals:
                staged.drop_columns(removals)
        yield from ()


def _iter_applied(
    transition: Transition, tctx: TransitionContext
) -> Generator[object, object, None]:
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
    sender = getattr(iterator, "send", None)
    try:
        value = next(iterator)
        while True:
            sent = yield value
            value = sender(sent) if sender is not None else next(iterator)
    except StopIteration:
        return
