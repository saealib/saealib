"""Confirming the true evaluations a transition asks for at a commit point.

A transition may yield an :class:`~saealib.transition.EvaluateRequest` while it
runs.  The commit point evaluates the requested population rows with the
evaluator (synchronous flow) or with the scheduler (asynchronous flow), and
confirms each result at once: the evaluated rows receive the true objective
values, the rows join the archive and the Pareto archive, the evaluation count
advances, and the History records the request.  The algorithm is not told about
these evaluations, and the confirmed results stay if the transition or a later
part of the commit point fails; only the transition's staged column writes are
discarded.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from saealib.context import (
    _begin_evaluation_report,
    _count_evaluated,
    _EvaluationReport,
)
from saealib.exceptions import EvaluationProtocolError
from saealib.execution.evaluator import (
    EvaluationRequest,
    EvaluationStatus,
    EvaluationUpdate,
    request_origin_kind,
    split_request,
)
from saealib.policies.feedback import (
    OBJECTIVE_SOURCE_CODES,
    TRUE,
    _set_objectives_true,
)
from saealib.population.population import OBJECTIVE_SOURCE_COLUMN

if TYPE_CHECKING:
    from collections.abc import Callable

    from saealib.context import OptimizationState
    from saealib.execution.evaluator import EvaluationResult
    from saealib.population.archive import Archive, ParetoArchive
    from saealib.population.population import Population


def rows_for(owner: Population, candidate_ids: np.ndarray) -> np.ndarray:
    """Return the owner rows that hold *candidate_ids*, in the given order.

    Parameters
    ----------
    owner : Population
        Population that holds the candidate rows.
    candidate_ids : numpy.ndarray
        Candidate IDs to locate.

    Returns
    -------
    numpy.ndarray
        Row indices, shape ``(len(candidate_ids),)``.

    Raises
    ------
    EvaluationProtocolError
        If *owner* has no ``id`` column or a candidate does not have exactly
        one row.
    """
    if "id" not in owner.schema:
        raise EvaluationProtocolError("evaluation owner requires candidate IDs")
    source = np.asarray(owner.get_array("id"), dtype=np.int64)
    rows = [np.flatnonzero(source == value) for value in candidate_ids]
    if any(len(row) != 1 for row in rows):
        raise EvaluationProtocolError("evaluated candidate is not in the owner")
    return np.asarray([int(row[0]) for row in rows], dtype=np.intp)


def apply_true_rows(
    owner: Population, candidate_ids: np.ndarray, result: EvaluationResult
) -> None:
    """Write the true objective values of *result* onto the owner rows.

    Rows whose owner has no objective column keep it missing, and the
    ``objective_source`` column marks every written row as truly evaluated.

    Parameters
    ----------
    owner : Population
        Population that holds the candidate rows.
    candidate_ids : numpy.ndarray
        Candidate IDs of the evaluated rows.
    result : EvaluationResult
        True objective values, raw constraint values, and violations.
    """
    rows = rows_for(owner, candidate_ids)
    values = {
        name: value
        for name, value in {
            "f": result.f,
            "g": result.g,
            "cv": result.cv,
        }.items()
        if name in owner.schema
    }
    _set_objectives_true(owner, values, len(rows))
    owner.update_rows(rows, values)


def append_evaluated_rows(
    archive: Archive,
    pareto_archive: Archive | ParetoArchive,
    owner: Population,
    request_id: int,
    candidate_ids: np.ndarray,
) -> None:
    """Append the owner rows of *candidate_ids* to the two archives.

    The archive's duplicate policy decides whether an already present row is
    kept, replaced, or appended again.  A failure restores every archive before
    it propagates, so a partly appended commit leaves no archive changed.

    Parameters
    ----------
    archive : Archive
        Main archive to append to.
    pareto_archive : Archive
        Pareto archive to append to; it keeps rows that duplicate the main
        archive under the ``keep_first`` policy.
    owner : Population
        Population that holds the evaluated rows.
    request_id : int
        Evaluation request that produced the rows.
    candidate_ids : numpy.ndarray
        Candidate IDs of the evaluated rows.
    """
    rows = rows_for(owner, candidate_ids)
    archives = (archive, pareto_archive)

    def row_genomes() -> Any:
        """Return the owner's genomes when every row carries one."""
        genomes = owner.genomes
        return genomes if len(genomes) == len(owner) else None

    def values_for(archive: Archive | ParetoArchive) -> list[dict[str, Any]]:
        genomes = None
        if "x" not in archive.schema or "x" not in owner.schema:
            genomes = row_genomes()
        values_list = []
        for row, candidate_id in zip(rows, candidate_ids, strict=True):
            values = {
                name: owner.get_array(name)[row]
                for name in archive.schema
                if name in owner.schema
            }
            if (
                OBJECTIVE_SOURCE_COLUMN in archive.schema
                and OBJECTIVE_SOURCE_COLUMN not in owner.schema
            ):
                values[OBJECTIVE_SOURCE_COLUMN] = np.uint8(OBJECTIVE_SOURCE_CODES[TRUE])
            if "id" in archive.schema:
                values["id"] = np.int64(candidate_id)
            if "request_id" in archive.schema:
                values["request_id"] = np.int64(request_id)
            if genomes is not None:
                values["genome"] = genomes.take([row])
            values_list.append(values)
        return values_list

    snapshots = [
        (
            target,
            {name: np.array(array, copy=True) for name, array in target._data.items()},
            target._size,
            target._structure_version,
            target._value_version,
            dict(target._cache),
            getattr(target, "_kdtree", None),
        )
        for target in archives
    ]
    try:
        for target in archives:
            for values in values_for(target):
                if "id" in values and len(target):
                    ids = target.get_array("id")[: len(target)]
                    if np.any(ids == values["id"]):
                        policy = getattr(target, "duplicate_policy", "keep_first")
                        if policy == "keep_first" and target is not pareto_archive:
                            continue
                        if policy == "append":
                            request_ids = target.get_array("request_id")[: len(target)]
                            if np.any(
                                (ids == values["id"])
                                & (request_ids == values.get("request_id"))
                            ):
                                continue
                        else:
                            target.delete(np.flatnonzero(ids == values["id"]))
                target.add(values)
    except Exception:
        restore_archive_snapshots(snapshots)
        raise


def restore_archive_snapshots(snapshots: list[tuple[Any, ...]]) -> None:
    """Restore the archive fields captured before an append.

    Parameters
    ----------
    snapshots : list of tuple
        Captured rows and structure fields, as taken before the append.
    """
    for (
        archive,
        snapshot,
        size,
        structure_version,
        value_version,
        cache,
        kdtree,
    ) in snapshots:
        archive._data = snapshot
        archive._size = size
        archive._structure_version = structure_version
        archive._value_version = value_version
        archive._cache = cache
        if hasattr(archive, "_kdtree"):
            archive._kdtree = kdtree
        invalidate = getattr(archive, "_invalidate_service_indexes", None)
        if callable(invalidate):
            invalidate()


def confirm_true_evaluation(
    state: OptimizationState,
    request: EvaluationRequest,
    result: EvaluationResult,
    *,
    owner: Population,
    attempt: int = 0,
    dispatch: Callable[[Any], None] | None = None,
) -> OptimizationState:
    """Confirm one completed true-evaluation request at a commit point.

    Confirms the request in an order that keeps the accounting consistent:
    validate the result, append the History row (a failure here changes
    nothing), write the owner rows and the archives and advance the evaluation
    count (these steps cannot fail), then dispatch the request-level event.  A
    raising observer therefore leaves the confirmed results, the evaluation
    count, and the History row in place, while nothing is told to the algorithm
    and nothing here is reverted by a failing commit point.

    Parameters
    ----------
    state : OptimizationState
        State the commit point works on.
    request : EvaluationRequest
        Request that produced *result*.
    result : EvaluationResult
        True objective values of the request's candidates.
    owner : Population
        Population that holds the requested candidate rows.
    attempt : int, optional
        Retry attempt the History records for the request.
    dispatch : callable or None, optional
        Receives the request-level evaluation event.  The counting function
        keeps the confirmed results and dispatches the event, so a raising
        observer keeps the confirmed results, the count, and the History row.

    Returns
    -------
    OptimizationState
        The state with the evaluation count advanced.
    """
    if result.candidate_ids is None:
        result.candidate_ids = request.candidate_ids
    if not np.array_equal(result.candidate_ids, request.candidate_ids):
        raise EvaluationProtocolError("result candidate IDs do not match the request")
    update = EvaluationUpdate(
        request.request_id,
        EvaluationStatus.COMPLETED,
        request.candidate_ids,
        result,
    )
    origin_kind = request_origin_kind(request)
    report = _EvaluationReport(
        request_id=int(request.request_id),
        candidate_ids=request.candidate_ids,
        status=EvaluationStatus.COMPLETED,
        origin_kind=origin_kind,
        dispatch=dispatch,
    )
    history = state.history
    if history is not None:
        history._observe_evaluation(update, attempt, origin_kind=origin_kind)
    _begin_evaluation_report(state, report, flush=True)
    apply_true_rows(owner, request.candidate_ids, result)
    append_evaluated_rows(
        state.archive,
        state.pareto_archive,
        owner,
        int(request.request_id),
        request.candidate_ids,
    )
    return _count_evaluated(state, request.candidate_ids, report=report)


def split_for_capacity(
    request: EvaluationRequest,
    capacity: int,
    *,
    allocate_request_id: Any,
) -> tuple[EvaluationRequest, ...]:
    """Split *request* into parts of at most *capacity* candidates.

    Parameters
    ----------
    request : EvaluationRequest
        Request to split.
    capacity : int
        Largest number of candidates one part may carry.
    allocate_request_id : callable
        Allocates the request ID of every part after the first.

    Returns
    -------
    tuple of EvaluationRequest
        The request unchanged when it fits, otherwise its parts.
    """
    if capacity < 1:
        raise EvaluationProtocolError("evaluation capacity is exhausted")
    count = max(1, -(-len(request.candidate_ids) // capacity))
    return split_request(request, count, allocate_request_id=allocate_request_id)
