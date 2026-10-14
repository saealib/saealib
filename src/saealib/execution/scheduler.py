"""Asynchronous evaluation scheduling."""

from __future__ import annotations

import copy
import logging
import time
from collections.abc import Iterable
from math import fsum
from typing import TYPE_CHECKING, Any

import numpy as np

from saealib.callback import PostEvaluationEvent
from saealib.context import EvaluationPlanState
from saealib.core.adapters import FeedbackAccumulator
from saealib.core.contracts import (
    ComponentContract,
    ExecutionContract,
    FeedbackBatch,
    FeedbackRequirement,
    PartSpec,
    ProposalBatch,
    ProposalRelations,
    QuantityRef,
    QuantityRequirement,
    StateContract,
)
from saealib.core.runtime import PollResult
from saealib.core.state import (
    ARCHIVES_MAIN,
    ARCHIVES_PARETO,
    EVALUATIONS_COUNT,
    EVALUATIONS_OWNERS,
    EVALUATIONS_PLAN,
    EVALUATIONS_PLAN_STATE,
    EVALUATIONS_PLAN_UPDATES,
    FEEDBACK_ACCUMULATOR,
    PENDING_EVALUATIONS,
    POPULATIONS_MAIN,
    PROPOSALS_CURRENT,
    RUNTIME_ASYNC_FATAL,
    RUNTIME_RNG,
)
from saealib.exceptions import (
    CheckpointError,
    EvaluationFatalError,
    EvaluationProtocolError,
    EvaluationSubmissionError,
    ValidationError,
)
from saealib.execution.evaluator import (
    EvaluationErrorInfo,
    EvaluationRequest,
    EvaluationStatus,
    EvaluationUpdate,
    Evaluator,
    PendingEvaluation,
)
from saealib.policies.evaluation import (
    _aggregate_repeated_updates,
    _combine_plan_updates,
    _continue_fidelity_plan,
)
from saealib.policies.feedback import (
    FeedbackBuilder,
    FeedbackResult,
    _feedback_batch_from_result,
)
from saealib.stages import deliver_feedback

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from saealib.context import OptimizationState


class AsyncEvaluationScheduler:
    """Coordinate asynchronous requests and serialized state mutation."""

    def contract(self) -> ComponentContract:
        """Return the scheduler contract and its evaluator part."""
        reads = (
            PENDING_EVALUATIONS,
            EVALUATIONS_OWNERS,
            EVALUATIONS_PLAN,
            EVALUATIONS_PLAN_STATE,
            EVALUATIONS_PLAN_UPDATES,
            EVALUATIONS_COUNT,
            ARCHIVES_MAIN,
            ARCHIVES_PARETO,
            PROPOSALS_CURRENT,
        )
        writes = (
            PENDING_EVALUATIONS,
            EVALUATIONS_OWNERS,
            EVALUATIONS_PLAN,
            EVALUATIONS_PLAN_STATE,
            EVALUATIONS_PLAN_UPDATES,
            EVALUATIONS_COUNT,
            ARCHIVES_MAIN,
            ARCHIVES_PARETO,
            RUNTIME_ASYNC_FATAL,
        )
        if self.algorithm is not None:
            reads += (POPULATIONS_MAIN, RUNTIME_RNG)
            writes += (POPULATIONS_MAIN, RUNTIME_RNG)
        return ComponentContract(
            parts=(PartSpec(name="evaluator", contract=self.evaluator.contract()),),
            state=StateContract(
                reads=reads,
                writes=writes,
            ),
            execution=ExecutionContract(),
        )

    def __init__(
        self,
        evaluator: Evaluator,
        *,
        max_pending: int = 1,
        max_reserved_fe: int | None = None,
        max_reserved_cost: float | None = None,
        timeout: float | None = None,
        retry_limit: int = 0,
        feedback_builder: FeedbackBuilder | None = None,
        algorithm: Any = None,
        callback_manager: Any = None,
    ) -> None:
        if max_pending < 1 or retry_limit < 0:
            raise ValidationError("max_pending and retry_limit are invalid")
        if timeout is not None and timeout < 0:
            raise ValidationError("timeout must be non-negative")
        if max_reserved_fe is not None and max_reserved_fe < 0:
            raise ValidationError("max_reserved_fe must be non-negative")
        if max_reserved_cost is not None and max_reserved_cost < 0:
            raise ValidationError("max_reserved_cost must be non-negative")
        if timeout is not None and (
            type(evaluator).cancel is Evaluator.cancel
            and type(evaluator).detach is Evaluator.detach
        ):
            raise ValidationError(
                "timeout requires an evaluator with cancellation or detach support"
            )
        self.evaluator = evaluator
        self.max_pending = max_pending
        self.max_reserved_fe = max_reserved_fe
        self.max_reserved_cost = max_reserved_cost
        self.timeout = timeout
        self.retry_limit = retry_limit
        self.feedback_builder = feedback_builder
        self.algorithm = algorithm
        self.callback_manager = callback_manager
        self._fatal_states: dict[int, tuple[OptimizationState, OptimizationState]] = {}
        self._feedback_accumulator: FeedbackAccumulator | None = None
        self._feedback_sequences: dict[int, int] = {}
        self._feedback_proposal_candidates: dict[int, frozenset[int]] = {}
        self._feedback_proposal_owners: dict[int, Any] = {}

    def enable_feedback_accumulator(self) -> None:
        """Enable delivery through a compiler-inserted accumulator."""
        consumer = self._feedback_consumer()
        contract_factory = getattr(consumer, "contract", None)
        contract = (
            contract_factory().lifecycle.feedback
            if callable(contract_factory)
            else None
        )
        if contract is None:
            return
        self._feedback_accumulator = FeedbackAccumulator(contract)

    def _sync_feedback_accumulator(self, state: OptimizationState) -> None:
        if self._feedback_accumulator is not None:
            state.set_state(FEEDBACK_ACCUMULATOR, self._feedback_accumulator.to_state())

    def pending_candidate_ids(self, state: OptimizationState) -> np.ndarray:
        """Return candidate IDs reserved by pending requests."""
        return state.pending_candidate_ids

    def reserved_fe(self, state: OptimizationState) -> int:
        """Return reserved candidate count."""
        return state.reserved_fe

    def reserved_cost(self, state: OptimizationState) -> float:
        """Return reserved estimated cost."""
        return state.reserved_cost

    def submit(
        self, state: OptimizationState, requests: Iterable[EvaluationRequest]
    ) -> OptimizationState:
        """Submit disjoint requests within capacity and budget limits."""
        requests = tuple(requests)
        if len(requests) > 1 and not self.evaluator.supports_batch_rollback():
            raise EvaluationProtocolError(
                "batch submission requires rollback-capable evaluator"
            )
        if state.offspring is None:
            raise EvaluationProtocolError("asynchronous submission requires offspring")
        existing = set(map(int, self.pending_candidate_ids(state)))
        plan_ids = {request.metadata.get("plan_id") for request in requests}
        allow_replicates = (
            bool(requests)
            and len(plan_ids) == 1
            and None not in plan_ids
            and all("replicate" in request.metadata for request in requests)
        )
        batch_candidates: set[int] = set()
        reserved_fe = self.reserved_fe(state)
        reserved_cost = self.reserved_cost(state)
        seen_requests: set[int] = set()
        plans: list[tuple[EvaluationRequest, float]] = []
        occupied_requests = (
            set(map(int, state.pending_evaluations))
            | set(map(int, state.evaluation_handles))
            | set(map(int, state.evaluation_owners))
        )
        for request in requests:
            request_id = int(request.request_id)
            if request_id in seen_requests or request_id in occupied_requests:
                raise EvaluationProtocolError(
                    f"request {request_id} is already pending"
                )
            seen_requests.add(request_id)
            occupied_requests.add(request_id)
        if len(state.pending_evaluations) + len(requests) > self.max_pending:
            raise EvaluationProtocolError("worker capacity would be exceeded")
        for request in requests:
            request_id = int(request.request_id)
            candidate_ids = set(map(int, request.candidate_ids))
            pending_same_plan = any(
                pending.request.metadata.get("plan_id")
                == request.metadata.get("plan_id")
                and "replicate" in pending.request.metadata
                and "replicate" in request.metadata
                for pending in state.pending_evaluations.values()
            )
            if existing.intersection(candidate_ids) and not (
                pending_same_plan and request.metadata.get("plan_id") in plan_ids
            ):
                raise EvaluationProtocolError(
                    "a candidate is pending in another request"
                )
            if not allow_replicates and batch_candidates.intersection(candidate_ids):
                raise EvaluationProtocolError(
                    "a candidate is present in more than one request"
                )
            estimated_cost = float(
                request.metadata.get("estimated_cost", len(candidate_ids))
            )
            if estimated_cost < 0 or not np.isfinite(estimated_cost):
                raise ValidationError("estimated_cost must be finite and non-negative")
            if (
                self.max_reserved_fe is not None
                and reserved_fe + len(candidate_ids) > self.max_reserved_fe
            ):
                raise EvaluationProtocolError(
                    "reserved evaluation budget would be exceeded"
                )
            if (
                self.max_reserved_cost is not None
                and fsum((reserved_cost, estimated_cost)) > self.max_reserved_cost
            ):
                raise EvaluationProtocolError("reserved cost budget would be exceeded")
            plans.append((request, estimated_cost))
            batch_candidates.update(candidate_ids)
            if not allow_replicates:
                existing.update(candidate_ids)
            reserved_fe += len(candidate_ids)
            reserved_cost += estimated_cost
        handles = dict(state.evaluation_handles)
        owners = dict(state.evaluation_owners)
        pending_map = dict(state.pending_evaluations)
        started: list[tuple[int, Any]] = []
        try:
            for request, estimated_cost in plans:
                request_id = int(request.request_id)
                try:
                    proposal_id = int(state.get_state(PROPOSALS_CURRENT))
                except KeyError:
                    proposal_id = None
                if proposal_id is not None and "proposal_id" not in request.metadata:
                    request = EvaluationRequest(
                        request.request_id,
                        request.candidate_ids,
                        request.payload,
                        request.outputs,
                        {**request.metadata, "proposal_id": proposal_id},
                    )
                handle = self.evaluator.submit(request, state.problem)
                started.append((request_id, handle))
                logger.debug(
                    "Evaluation submitted: request_id=%s candidates=%s",
                    request_id,
                    len(request.candidate_ids),
                )
                prediction = copy.deepcopy(state.predictions)
                pending = PendingEvaluation(
                    request,
                    EvaluationStatus.PENDING,
                    np.empty(0, dtype=np.int64),
                    reserved_cost=estimated_cost,
                    prediction=prediction,
                )
                checkpointable = self.evaluator.can_reattach(pending)
                pending = PendingEvaluation(
                    request,
                    EvaluationStatus.PENDING,
                    np.empty(0, dtype=np.int64),
                    reserved_cost=estimated_cost,
                    checkpointable=checkpointable,
                    prediction=prediction,
                )
                handles[request_id] = handle
                pending_map[request_id] = pending
                owners[request_id] = state.offspring
        except Exception as exc:
            cleanup_failed = self._cleanup_started_handles(started)
            if cleanup_failed:
                partial = state.replace(
                    evaluation_handles=handles,
                    evaluation_owners=owners,
                    pending_evaluations=pending_map,
                )
                raise EvaluationSubmissionError(
                    "submission failed and started handles cannot be cleaned up: "
                    f"{cleanup_failed}",
                    partial,
                ) from exc
            raise
        if not requests:
            return state
        return state.replace(
            evaluation_handles=handles,
            evaluation_owners=owners,
            pending_evaluations=pending_map,
        )

    def _cleanup_started_handles(self, started: list[tuple[int, Any]]) -> list[int]:
        cleanup_failed = []
        for request_id, handle in started:
            if not self.evaluator.cancel(handle) and not self.evaluator.detach(handle):
                cleanup_failed.append(request_id)
        return cleanup_failed

    def cancel(self, state: OptimizationState, request_id: int) -> OptimizationState:
        """Cancel a pending request and commit its terminal update."""
        pending = state.pending_evaluations.get(request_id)
        handle = state.evaluation_handles.get(request_id)
        if pending is None or handle is None:
            raise EvaluationProtocolError(f"request {request_id} is not pending")
        if not self.evaluator.cancel(handle):
            raise EvaluationProtocolError(f"request {request_id} cannot be cancelled")
        update = EvaluationUpdate(
            np.int64(request_id),
            EvaluationStatus.CANCELLED,
            np.empty(0, dtype=np.int64),
            error=EvaluationErrorInfo("Cancelled", "evaluation cancelled"),
            sequence=pending.last_delivered_sequence + 1,
        )
        handle._delivered_sequence = update.sequence
        current = self._commit_update(state, pending, handle, update)
        logger.warning("Evaluation cancelled: request_id=%s", request_id)
        return current

    def poll(
        self, state: OptimizationState, *, wait: bool = False
    ) -> OptimizationState:
        """Collect completed work and commit updates serially."""
        return self.poll_result(state, wait=wait).state

    def poll_result(
        self, state: OptimizationState, *, wait: bool = False
    ) -> PollResult:
        """Collect completed work and report whether this call made progress."""
        if not state.pending_evaluations:
            return PollResult(state=state, progressed=False)
        for pending in state.pending_evaluations.values():
            if pending.fatal_error is not None:
                raise EvaluationFatalError(
                    pending.fatal_error.message,
                    state,
                )
        fatal = self._fatal_states.get(id(state))
        if fatal is not None:
            _, fatal_state = fatal
            raise EvaluationFatalError(
                "async update has a persistent fatal state", fatal_state
            )
        current = state
        progressed = False
        while current.pending_evaluations:
            progress = False
            ready: list[tuple[float, int, Any, EvaluationUpdate]] = []
            for request_id, handle in tuple(current.evaluation_handles.items()):
                pending = current.pending_evaluations.get(request_id)
                if pending is None:
                    continue
                timed_out = self.timeout is not None and (
                    time.monotonic() - handle._submitted_at >= self.timeout
                )
                if timed_out:
                    cancelled = self.evaluator.cancel(handle)
                    detached = False if cancelled else self.evaluator.detach(handle)
                    recovery = (
                        "cancel" if cancelled else "detach" if detached else "none"
                    )
                    logger.warning(
                        "Evaluation timed out: request_id=%s recovery=%s",
                        request_id,
                        recovery,
                    )
                    if not cancelled and not detached:
                        error = EvaluationErrorInfo(
                            "Timeout",
                            "evaluation timed out and backend cannot terminate",
                        )
                        pending_map = dict(current.pending_evaluations)
                        pending_map[request_id] = PendingEvaluation(
                            pending.request,
                            EvaluationStatus.CANCELLED,
                            pending.applied_candidate_ids,
                            pending.last_delivered_sequence,
                            pending.last_acknowledged_sequence,
                            {**pending.processing, -1: "timeout-fatal"},
                            pending.buffered_updates,
                            pending.reserved_cost,
                            pending.retry_count,
                            True,
                            pending.original_candidate_ids,
                            pending.feedback_result,
                            error,
                            pending.prediction,
                        )
                        async_fatal = {
                            "request_id": request_id,
                            "reason": error.message,
                        }
                        current = current.replace(
                            async_fatal=async_fatal,
                            pending_evaluations=pending_map,
                        )
                        progress = True
                        progressed = True
                        continue
                    update = EvaluationUpdate(
                        np.int64(request_id),
                        EvaluationStatus.CANCELLED,
                        np.empty(0, dtype=np.int64),
                        error=EvaluationErrorInfo("Timeout", "evaluation timed out"),
                        sequence=pending.last_delivered_sequence + 1,
                    )
                    handle._delivered_sequence = update.sequence
                    ready.append(
                        (
                            getattr(handle, "_completed_at", None) or time.monotonic(),
                            request_id,
                            handle,
                            update,
                        )
                    )
                    continue
                updates = self.evaluator.collect(handle, wait=False)
                for update in updates:
                    logger.debug(
                        "Evaluation collected: request_id=%s status=%s sequence=%s",
                        request_id,
                        update.status.value,
                        update.sequence,
                    )
                    ready.append(
                        (
                            getattr(handle, "_completed_at", None) or time.monotonic(),
                            request_id,
                            handle,
                            update,
                        )
                    )
            for _, request_id, handle, update in sorted(
                ready, key=lambda item: (item[0], item[1], item[3].sequence)
            ):
                pending = current.pending_evaluations.get(request_id)
                if pending is None:
                    continue
                try:
                    current = self._commit_update(current, pending, handle, update)
                except EvaluationFatalError as exc:
                    self._fatal_states[id(state)] = (state, exc.state)
                    raise
                progress = True
                progressed = True
            if not wait or not current.pending_evaluations:
                break
            if not progress:
                time.sleep(0.001)
        return PollResult(state=current, progressed=progressed)

    def checkpoint(self, state: OptimizationState, path: str) -> None:
        """Save pending state only when the evaluator can reattach it."""
        if any(
            pending.fatal_error is None and not self.evaluator.can_reattach(pending)
            for pending in state.pending_evaluations.values()
        ):
            raise CheckpointError("evaluator cannot reattach asynchronous pending work")
        if self._feedback_accumulator is not None:
            state = state.replace(
                feedback_accumulator=self._feedback_accumulator.to_state()
            )
        state.save(path)

    def reattach(self, state: OptimizationState) -> OptimizationState:
        """Rebuild runtime handles for checkpointed requests."""
        handles = dict(state.evaluation_handles)
        for request_id, pending in state.pending_evaluations.items():
            if pending.fatal_error is not None:
                raise EvaluationFatalError(pending.fatal_error.message, state)
            if (
                pending.status
                in {
                    EvaluationStatus.COMPLETED,
                    EvaluationStatus.FAILED,
                    EvaluationStatus.CANCELLED,
                }
                and "callback-completed" in pending.processing.values()
            ):
                continue
            if not self.evaluator.can_reattach(pending):
                raise CheckpointError(f"evaluator cannot reattach request {request_id}")
            handles[request_id] = self.evaluator.reattach(pending, state.problem)
        current = state.replace(evaluation_handles=handles)
        if self._feedback_accumulator is not None:
            has_partial_feedback = any(
                update.result is not None and update.status is EvaluationStatus.PARTIAL
                for pending in current.pending_evaluations.values()
                for update in pending.buffered_updates
            )
            try:
                snapshot = current.get_state(FEEDBACK_ACCUMULATOR)
            except KeyError:
                snapshot = None
            if snapshot is not None:
                self._feedback_accumulator.restore_state(snapshot)
                if has_partial_feedback and (
                    self._feedback_accumulator.buffered_proposal_count == 0
                    and self._feedback_accumulator.ready_count == 0
                ):
                    raise CheckpointError(
                        "checkpoint has partial feedback updates but no "
                        "FeedbackAccumulator buffer"
                    )
            else:
                # Nothing was buffered, so rebuild from the pending requests.
                if has_partial_feedback:
                    raise CheckpointError(
                        "checkpoint is missing FeedbackAccumulator state for "
                        "partial feedback"
                    )
                for pending in current.pending_evaluations.values():
                    self._restore_accumulated_feedback(current, pending)
        for request_id, pending in tuple(current.pending_evaluations.items()):
            handle = current.evaluation_handles.get(request_id)
            for update in pending.buffered_updates:
                if pending.processing.get(update.sequence) == "committed":
                    continue
                current = self._commit_update(current, pending, handle, update)
                pending = current.pending_evaluations.get(request_id)
                if pending is None:
                    break
        return current

    def _commit_update(
        self,
        state: OptimizationState,
        pending: PendingEvaluation,
        handle: Any,
        update: EvaluationUpdate,
    ) -> OptimizationState:
        request_id = int(update.request_id)
        current_pending = state.pending_evaluations.get(request_id)
        if current_pending is None:
            return state
        original_ids = current_pending.original_candidate_ids
        if original_ids is None:
            raise EvaluationProtocolError("pending original candidate IDs are missing")
        processing = dict(current_pending.processing)
        key = int(update.sequence)
        if processing.get(key) == "committed":
            return state
        if processing.get(key) == "callback-completed":
            previous = next(
                (
                    item
                    for item in current_pending.buffered_updates
                    if item.sequence == key
                ),
                None,
            )
            if previous is None or not self._same_update(previous, update):
                raise EvaluationProtocolError(
                    "redelivered callback update payload differs"
                )
            if update.status not in {
                EvaluationStatus.COMPLETED,
                EvaluationStatus.FAILED,
                EvaluationStatus.CANCELLED,
            }:
                return state
            pending_map = dict(state.pending_evaluations)
            pending_map.pop(request_id, None)
            handles = dict(state.evaluation_handles)
            handles.pop(request_id, None)
            owners = dict(state.evaluation_owners)
            owners.pop(request_id, None)
            return state.replace(
                pending_evaluations=pending_map,
                evaluation_handles=handles,
                evaluation_owners=owners,
            )
        if processing.get(key) in {"tell-started", "callback-started"}:
            if current_pending.fatal_error is not None:
                raise EvaluationFatalError(current_pending.fatal_error.message, state)
            raise EvaluationProtocolError(
                "update has a fatal post-effect failure and cannot be retried"
            )
        replay = key in processing
        if not replay and update.sequence != (
            current_pending.last_delivered_sequence + 1
        ):
            raise EvaluationProtocolError("async update sequence is not contiguous")
        if replay:
            previous = next(
                (
                    item
                    for item in current_pending.buffered_updates
                    if item.sequence == update.sequence
                ),
                None,
            )
            if previous is None or not self._same_update(previous, update):
                raise EvaluationProtocolError("redelivered update payload differs")
        else:
            self._validate_update(current_pending, update)
        if (
            not replay
            and current_pending.retry_count > 0
            and update.status is EvaluationStatus.COMPLETED
        ):
            logger.warning(
                "Evaluation retry completed: request_id=%s attempt=%s",
                request_id,
                current_pending.retry_count,
            )
        buffered = list(current_pending.buffered_updates)
        if not any(
            item.request_id == update.request_id and item.sequence == update.sequence
            for item in buffered
        ):
            buffered.append(update)

        pending_record = current_pending

        def progress(
            current: OptimizationState,
            stage: str,
            applied: np.ndarray | None = None,
            fatal_error: EvaluationErrorInfo | None = None,
            acknowledged_sequence: int | None = None,
        ) -> OptimizationState:
            nonlocal current_pending, pending_record
            processing[key] = stage
            pending_record = PendingEvaluation(
                pending_record.request,
                update.status,
                pending_record.applied_candidate_ids if applied is None else applied,
                update.sequence,
                pending_record.last_acknowledged_sequence
                if acknowledged_sequence is None
                else acknowledged_sequence,
                processing.copy(),
                tuple(buffered),
                pending_record.reserved_cost,
                pending_record.retry_count,
                pending_record.checkpointable,
                original_ids,
                current.feedback_result,
                fatal_error,
                pending_record.prediction,
            )
            current_pending = pending_record
            return current.replace(
                pending_evaluations={
                    **current.pending_evaluations,
                    request_id: pending_record,
                }
            )

        if not replay:
            current = state.replace(
                pending_evaluations={
                    **state.pending_evaluations,
                    request_id: PendingEvaluation(
                        current_pending.request,
                        update.status,
                        current_pending.applied_candidate_ids,
                        update.sequence,
                        current_pending.last_acknowledged_sequence,
                        {**processing, key: "received"},
                        tuple(buffered),
                        current_pending.reserved_cost,
                        current_pending.retry_count,
                        current_pending.checkpointable,
                        original_ids,
                        current_pending.feedback_result,
                        current_pending.fatal_error,
                        current_pending.prediction,
                    ),
                }
            )
        else:
            current = state
        current_plan = current.evaluation_plan
        current_plan_ids = (
            {int(item.request_id) for item in current_plan.requests}
            if current_plan is not None
            else set()
        )
        if not replay and update.result is not None and request_id in current_plan_ids:
            plan_updates = {
                request_id: list(updates)
                for request_id, updates in current.evaluation_plan_updates.items()
            }
            plan_updates.setdefault(request_id, []).append(update)
            current = current.replace(evaluation_plan_updates=plan_updates)
        current_pending = current.pending_evaluations[request_id]
        if not replay:
            history = current.history
            if history is not None:
                history._observe_evaluation(update, current_pending.retry_count)
        if current_pending.feedback_result is not None:
            current = current.replace(feedback_result=current_pending.feedback_result)
        current_plan = current.evaluation_plan
        current_plan_ids = (
            {int(item.request_id) for item in current_plan.requests}
            if current_plan is not None
            else set()
        )
        active_plan_request = request_id in current_plan_ids
        if (
            active_plan_request
            and current_plan is not None
            and update.status
            in {
                EvaluationStatus.COMPLETED,
                EvaluationStatus.FAILED,
                EvaluationStatus.CANCELLED,
            }
        ):
            plan_state = current.evaluation_plan_state
            completed = set(plan_state.completed if plan_state else ())
            acknowledged = set(plan_state.acknowledged if plan_state else ())
            completed.add(request_id)
            plan_ids = {int(item.request_id) for item in current_plan.requests}
            if plan_ids <= completed | acknowledged:
                continuation_plan = _continue_fidelity_plan(
                    current_plan, current.evaluation_plan_updates, current
                )
                if continuation_plan is not None:
                    promoted_id = continuation_plan.artifacts.get("promoted_request_id")
                    if promoted_id is None:
                        raise EvaluationProtocolError(
                            "fidelity continuation is missing its request ID"
                        )
                    current = current.replace(
                        evaluation_plan=continuation_plan,
                        evaluation_plan_state=EvaluationPlanState(
                            submitted=plan_state.submitted if plan_state else (),
                            completed=tuple(sorted(completed)),
                            acknowledged=tuple(sorted(acknowledged)),
                            deferred=(int(promoted_id),),
                            continuation=continuation_plan.continuation,
                            feedback=plan_state.feedback if plan_state else None,
                        ),
                    )
                    current_plan = continuation_plan
                    current_plan_ids = {
                        int(item.request_id) for item in continuation_plan.requests
                    }
        repeated_plan = active_plan_request and self._is_repeated_plan(current_plan)
        plan_effect_update = update
        if (
            active_plan_request
            and current_plan is not None
            and len(current_plan.requests) > 1
        ):
            plan_state = current.evaluation_plan_state
            completed = set(plan_state.completed if plan_state else ())
            acknowledged = set(plan_state.acknowledged if plan_state else ())
            if update.status in {
                EvaluationStatus.COMPLETED,
                EvaluationStatus.FAILED,
                EvaluationStatus.CANCELLED,
            }:
                completed.add(request_id)
            if current_plan_ids <= completed | acknowledged:
                plan_effect_update = (
                    self._aggregate_plan_update(current, update)
                    if repeated_plan
                    else _combine_plan_updates(
                        current_plan, current.evaluation_plan_updates, update
                    )
                )
            else:
                plan_effect_update = None
        # Accumulator delivery preserves proposal completion across refills.
        accumulator_update = (
            update if self._feedback_accumulator is not None else plan_effect_update
        )
        population_update = update
        if (
            self._feedback_accumulator is None
            and active_plan_request
            and current_plan is not None
            and len(current_plan.requests) > 1
        ):
            population_update = plan_effect_update
        if population_update is None:
            has_rows = False
        else:
            has_rows = (
                population_update.result is not None
                and len(population_update.candidate_ids) > 0
            )
        ranks = {
            "received": 0,
            "population-applied": 1,
            "archived": 2,
            "feedback-applied": 3,
            "told": 4,
            "fe-applied": 5,
            "acknowledged": 6,
            "committed": 7,
            "callback-completed": 8,
        }
        stage_rank = ranks.get(processing.get(key, "received"), 0)
        if population_update is not None and stage_rank < 1 and has_rows:
            current = self._apply_population(current, population_update)
            current = progress(current, "population-applied")
            stage_rank = 1
        if accumulator_update is not None and stage_rank < 2 and has_rows:
            current = self._apply_archive(current, accumulator_update)
            current = progress(current, "archived")
            stage_rank = 2
        if accumulator_update is not None and stage_rank < 3 and has_rows:
            current = self._apply_feedback(current, accumulator_update)
            current = progress(current, "feedback-applied")
            stage_rank = 3
        if accumulator_update is not None and stage_rank < 4 and has_rows:
            current = progress(
                current,
                "tell-started",
                fatal_error=EvaluationErrorInfo(
                    "TellFatal", "algorithm tell started and cannot be retried"
                ),
            )
            try:
                current = self._apply_feedback_delivery(current, accumulator_update)
            except Exception as exc:
                raise EvaluationFatalError(
                    "algorithm tell failed after side effects; update is fatal",
                    current,
                ) from exc
            current = progress(current, "told")
            stage_rank = 4
        if stage_rank < 5:
            if update.result is not None:
                current = current.replace(fe=current.fe + len(update.candidate_ids))
            current = progress(current, "fe-applied")
        if stage_rank < 6:
            self.evaluator.acknowledge(handle, update.sequence)
            current = progress(
                current,
                "acknowledged",
                acknowledged_sequence=update.sequence,
            )
        applied = np.unique(
            np.concatenate(
                [current_pending.applied_candidate_ids, update.candidate_ids]
            )
        )
        terminal = update.status in {
            EvaluationStatus.COMPLETED,
            EvaluationStatus.FAILED,
            EvaluationStatus.CANCELLED,
        }
        remaining = np.setdiff1d(original_ids, applied)
        retried = False
        if (
            update.status is EvaluationStatus.FAILED
            and remaining.size
            and current_pending.retry_count < self.retry_limit
        ):
            retried = True
            reopen_history = current.history
            if reopen_history is not None:
                reopen_history._reopen_request(request_id)
            rows = np.asarray(
                [
                    int(
                        np.flatnonzero(current_pending.request.candidate_ids == value)[
                            0
                        ]
                    )
                    for value in remaining
                ],
                dtype=np.intp,
            )
            request = EvaluationRequest(
                current_pending.request.request_id,
                remaining,
                current_pending.request.payload.take(rows),
                current_pending.request.outputs,
                current_pending.request.metadata,
            )
            # Allocate from the original reservation; retry reservations are
            # per candidate, and a scalar estimate cannot be candidate-specific.
            unit_cost = fsum((current_pending.reserved_cost,)) / len(
                current_pending.request.candidate_ids
            )
            retry_cost = unit_cost * len(remaining)
            other_reserved_cost = fsum(
                pending.reserved_cost
                for rid, pending in current.pending_evaluations.items()
                if rid != request_id
            )
            if (
                self.max_reserved_cost is not None
                and fsum((other_reserved_cost, retry_cost)) > self.max_reserved_cost
            ):
                raise EvaluationProtocolError("retry would exceed reserved cost budget")
            retry_count = current_pending.retry_count + 1
            new_handle = self.evaluator.submit(request, state.problem)
            logger.warning(
                "Evaluation retry submitted: request_id=%s attempt=%s candidates=%s",
                request_id,
                retry_count,
                len(remaining),
            )
            retry_pending = PendingEvaluation(
                request,
                EvaluationStatus.PENDING,
                applied,
                reserved_cost=retry_cost,
                retry_count=retry_count,
                checkpointable=self.evaluator.can_reattach(
                    PendingEvaluation(
                        request,
                        EvaluationStatus.PENDING,
                        applied,
                        checkpointable=False,
                    )
                ),
                original_candidate_ids=current_pending.original_candidate_ids,
                prediction=current_pending.prediction,
            )
            current = current.replace(
                pending_evaluations={
                    **current.pending_evaluations,
                    request_id: retry_pending,
                },
                evaluation_handles={
                    **current.evaluation_handles,
                    request_id: new_handle,
                },
                evaluation_owners=current.evaluation_owners,
            )
        elif terminal:
            committed_pending = PendingEvaluation(
                current_pending.request,
                update.status,
                applied,
                update.sequence,
                update.sequence,
                {**processing, key: "committed"},
                tuple(buffered),
                current_pending.reserved_cost,
                current_pending.retry_count,
                current_pending.checkpointable,
                current_pending.original_candidate_ids,
                current_pending.feedback_result,
                None,
                current_pending.prediction,
            )
            current = current.replace(
                pending_evaluations={
                    **current.pending_evaluations,
                    request_id: committed_pending,
                }
            )
        else:
            current_pending = PendingEvaluation(
                current_pending.request,
                update.status,
                applied,
                update.sequence,
                update.sequence,
                {**processing, key: "committed"},
                tuple(buffered),
                current_pending.reserved_cost,
                current_pending.retry_count,
                current_pending.checkpointable,
                original_ids,
                current_pending.feedback_result,
                None,
                current_pending.prediction,
            )
            current = current.replace(
                pending_evaluations={
                    **current.pending_evaluations,
                    request_id: current_pending,
                }
            )
        if (
            self._feedback_accumulator is not None
            and terminal
            and not retried
            and update.result is None
        ):
            proposal_id = self._proposal_id(current, request_id)
            self._feedback_accumulator.discard(proposal_id)
            self._feedback_proposal_owners.pop(proposal_id, None)
            self._sync_feedback_accumulator(current)
        callback_update = (
            update if self._feedback_accumulator is not None else plan_effect_update
        )
        if (
            callback_update is not None
            and len(callback_update.candidate_ids)
            and self.callback_manager is not None
        ):
            current = progress(
                current,
                "callback-started",
                applied=applied,
                fatal_error=EvaluationErrorInfo(
                    "CallbackFatal", "callback started and cannot be retried"
                ),
            )
            offspring = None
            owner = self._owner(current, request_id)
            offspring = owner.extract(
                self._rows(current, callback_update.candidate_ids, request_id)
            )
            try:
                self.callback_manager.dispatch(
                    PostEvaluationEvent(
                        ctx=current,
                        offspring=offspring,
                        candidate_ids=callback_update.candidate_ids,
                        request_id=callback_update.request_id,
                        status=callback_update.status,
                    )
                )
            except Exception as exc:
                raise EvaluationFatalError(
                    "post-evaluation callback failed; update is fatal", current
                ) from exc
            current = progress(current, "callback-completed", applied=applied)
        if terminal and not retried:
            pending_map = dict(current.pending_evaluations)
            pending_map.pop(request_id, None)
            handles = dict(current.evaluation_handles)
            handles.pop(request_id, None)
            owners = dict(current.evaluation_owners)
            owners.pop(request_id, None)
            current = current.replace(
                pending_evaluations=pending_map,
                evaluation_handles=handles,
                evaluation_owners=owners,
            )
            plan_state = current.evaluation_plan_state
            if active_plan_request and plan_state is not None:
                completed = tuple(sorted(set(plan_state.completed) | {request_id}))
                acknowledged = tuple(
                    sorted(set(plan_state.acknowledged) | {request_id})
                )
                deferred = tuple(
                    item for item in plan_state.deferred if item != request_id
                )
                current = current.replace(
                    evaluation_plan_state=EvaluationPlanState(
                        submitted=tuple(
                            sorted(set(plan_state.submitted) | {request_id})
                        ),
                        completed=completed,
                        acknowledged=acknowledged,
                        deferred=deferred,
                        continuation=plan_state.continuation,
                        feedback=plan_state.feedback,
                    )
                )
        return current

    @staticmethod
    def _is_repeated_plan(plan: Any) -> bool:
        return bool(
            plan is not None
            and len(plan.requests) > 1
            and all("replicate" in request.metadata for request in plan.requests)
        )

    def _aggregate_plan_update(
        self, state: OptimizationState, update: EvaluationUpdate
    ) -> EvaluationUpdate | None:
        plan = state.evaluation_plan
        if plan is None:
            return update
        return _aggregate_repeated_updates(plan, state.evaluation_plan_updates, update)

    def _validate_update(
        self, pending: PendingEvaluation, update: EvaluationUpdate
    ) -> None:
        if int(update.request_id) != int(pending.request.request_id):
            raise EvaluationProtocolError(
                "update request_id does not match pending work"
            )
        requested = set(map(int, pending.request.candidate_ids))
        incoming = set(map(int, update.candidate_ids))
        applied = set(map(int, pending.applied_candidate_ids))
        if not incoming.issubset(requested) or incoming.intersection(applied):
            raise EvaluationProtocolError("async result candidate IDs are invalid")
        if update.result is not None and (
            update.result.candidate_ids is None
            or not np.array_equal(update.result.candidate_ids, update.candidate_ids)
        ):
            raise EvaluationProtocolError("result candidate IDs do not match update")
        if update.status in {EvaluationStatus.PENDING, EvaluationStatus.RUNNING} and (
            update.result is not None or len(update.candidate_ids)
        ):
            raise EvaluationProtocolError(
                "non-terminal running updates cannot carry results"
            )
        if update.status is EvaluationStatus.PARTIAL and (
            not len(update.candidate_ids) or update.result is None
        ):
            raise EvaluationProtocolError("partial updates require result rows")
        if update.status is EvaluationStatus.COMPLETED and (
            len(update.candidate_ids) > 0 and update.result is None
        ):
            raise EvaluationProtocolError("completed result rows require a result")
        if (
            update.status
            in {
                EvaluationStatus.FAILED,
                EvaluationStatus.CANCELLED,
            }
            and update.error is None
        ):
            raise EvaluationProtocolError("terminal failure requires error details")
        original_ids = pending.original_candidate_ids
        if original_ids is None:
            raise EvaluationProtocolError("pending original candidate IDs are missing")
        original = set(map(int, original_ids))
        if update.status is EvaluationStatus.COMPLETED and (
            incoming | applied != original
        ):
            raise EvaluationProtocolError(
                "completed update does not account for all requested candidates"
            )
        if pending.status in {
            EvaluationStatus.COMPLETED,
            EvaluationStatus.FAILED,
            EvaluationStatus.CANCELLED,
        }:
            raise EvaluationProtocolError("terminal evaluation cannot transition")
        allowed = {
            EvaluationStatus.PENDING: {
                EvaluationStatus.RUNNING,
                EvaluationStatus.PARTIAL,
                EvaluationStatus.COMPLETED,
                EvaluationStatus.FAILED,
                EvaluationStatus.CANCELLED,
            },
            EvaluationStatus.RUNNING: {
                EvaluationStatus.PARTIAL,
                EvaluationStatus.COMPLETED,
                EvaluationStatus.FAILED,
                EvaluationStatus.CANCELLED,
            },
            EvaluationStatus.PARTIAL: {
                EvaluationStatus.PARTIAL,
                EvaluationStatus.COMPLETED,
                EvaluationStatus.FAILED,
                EvaluationStatus.CANCELLED,
            },
        }
        if update.status not in allowed.get(pending.status, set()):
            raise EvaluationProtocolError("invalid evaluation status transition")

    @staticmethod
    def _same_update(left: EvaluationUpdate, right: EvaluationUpdate) -> bool:
        if (
            left.request_id != right.request_id
            or left.status is not right.status
            or left.sequence != right.sequence
            or not np.array_equal(left.candidate_ids, right.candidate_ids)
        ):
            return False
        if (left.error is None) != (right.error is None):
            return False
        if (
            left.error is not None
            and right.error is not None
            and (
                left.error.error_type != right.error.error_type
                or left.error.message != right.error.message
                or dict(left.error.details) != dict(right.error.details)
            )
        ):
            return False
        if (left.result is None) != (right.result is None):
            return False
        if left.result is None or right.result is None:
            return True
        return (
            np.array_equal(left.result.f, right.result.f)
            and np.array_equal(left.result.g, right.result.g)
            and np.array_equal(left.result.cv, right.result.cv)
            and AsyncEvaluationScheduler._optional_array_equal(
                left.result.candidate_ids, right.result.candidate_ids
            )
            and AsyncEvaluationScheduler._optional_array_equal(
                left.result.cost, right.result.cost
            )
            and AsyncEvaluationScheduler._optional_array_equal(
                left.result.noise, right.result.noise
            )
            and left.result.outputs.keys() == right.result.outputs.keys()
            and all(
                np.array_equal(left.result.outputs[name], right.result.outputs[name])
                for name in left.result.outputs
            )
        )

    @staticmethod
    def _optional_array_equal(left: Any, right: Any) -> bool:
        if left is None or right is None:
            return left is None and right is None
        return bool(np.array_equal(left, right))

    def _owner(self, state: OptimizationState, request_id: int) -> Any:
        owner = state.evaluation_owners.get(request_id, state.offspring)
        if owner is None:
            raise EvaluationProtocolError("async evaluation owner is missing")
        return owner

    def _proposal_id(self, state: OptimizationState, request_id: int) -> int:
        pending = state.pending_evaluations.get(int(request_id))
        if pending is not None:
            value = pending.request.metadata.get("proposal_id")
            if value is not None:
                return int(value)
        try:
            return int(state.get_state(PROPOSALS_CURRENT))
        except KeyError as exc:
            raise ValidationError("feedback proposal ID is missing") from exc

    def _capture_proposal_owner(
        self,
        state: OptimizationState,
        proposal_id: int,
        candidate_ids: frozenset[int],
    ) -> Any:
        existing = self._feedback_proposal_owners.get(proposal_id)
        if existing is not None:
            return existing

        candidates = tuple(int(value) for value in candidate_ids)
        owners = tuple(state.evaluation_owners.values())
        current = state.offspring
        if current is not None:
            owners = (current, *owners)
        for owner in owners:
            if "id" not in owner.schema:
                continue
            ids = np.asarray(owner.get_array("id"), dtype=np.int64)
            rows = np.flatnonzero(np.isin(ids, candidates))
            if len(rows) == len(candidates):
                captured = owner.extract(rows)
                self._feedback_proposal_owners[proposal_id] = captured
                return captured
        raise EvaluationProtocolError(
            f"async proposal {proposal_id} owner is missing candidate rows"
        )

    def _rows(
        self, state: OptimizationState, ids: np.ndarray, request_id: int
    ) -> np.ndarray:
        owner = self._owner(state, request_id)
        if "id" not in owner.schema:
            raise EvaluationProtocolError("async updates require offspring IDs")
        source = owner.get_array("id")
        rows = [np.flatnonzero(source == value) for value in ids]
        if any(len(row) != 1 for row in rows):
            raise EvaluationProtocolError("async result candidate is not in offspring")
        return np.asarray([int(row[0]) for row in rows], dtype=np.intp)

    def _apply_population(
        self, state: OptimizationState, update: EvaluationUpdate
    ) -> OptimizationState:
        if update.result is None:
            return state
        owner = self._owner(state, int(update.request_id))
        rows = self._rows(state, update.candidate_ids, int(update.request_id))
        owner.update_rows(
            rows,
            {
                name: value
                for name, value in {
                    "f": update.result.f,
                    "g": update.result.g,
                    "cv": update.result.cv,
                }.items()
                if name in owner.schema
            },
        )
        return state

    def _apply_archive(
        self, state: OptimizationState, update: EvaluationUpdate
    ) -> OptimizationState:
        if update.result is None:
            return state
        owner = self._owner(state, int(update.request_id))
        rows = self._rows(state, update.candidate_ids, int(update.request_id))
        archives = (state.archive, state.pareto_archive)

        def values_for(archive: Any) -> list[dict[str, Any]]:
            values_list = []
            for row, candidate_id in zip(rows, update.candidate_ids, strict=True):
                values = {
                    name: owner.get_array(name)[row]
                    for name in archive.schema
                    if name in owner.schema
                }
                if "id" in archive.schema:
                    values["id"] = np.int64(candidate_id)
                if "request_id" in archive.schema:
                    values["request_id"] = np.int64(update.request_id)
                values_list.append(values)
            return values_list

        snapshots = []
        for archive in archives:
            snapshots.append(
                (
                    archive,
                    {
                        name: np.array(array, copy=True)
                        for name, array in archive._data.items()
                    },
                    archive._size,
                    archive._structure_version,
                    archive._value_version,
                    dict(archive._cache),
                    getattr(archive, "_kdtree", None),
                )
            )
        try:
            for archive in archives:
                for values in values_for(archive):
                    if "id" in values and len(archive):
                        ids = archive.get_array("id")[: len(archive)]
                        if np.any(ids == values["id"]):
                            policy = getattr(archive, "duplicate_policy", "keep_first")
                            if (
                                policy == "keep_first"
                                and archive is not state.pareto_archive
                            ):
                                continue
                            if policy == "append":
                                request_ids = archive.get_array("request_id")[
                                    : len(archive)
                                ]
                                if np.any(
                                    (ids == values["id"])
                                    & (request_ids == values.get("request_id"))
                                ):
                                    continue
                            else:
                                archive.delete(np.flatnonzero(ids == values["id"]))
                    archive.add(values)
        except Exception:
            self._restore_archive_snapshots(snapshots)
            raise
        return state

    def _restore_archive_snapshots(self, snapshots: list[tuple[Any, ...]]) -> None:
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
            # Rebuild opaque service indexes from restored rows after rollback.
            invalidate = getattr(archive, "_invalidate_service_indexes", None)
            if callable(invalidate):
                invalidate()

    def _apply_feedback(
        self, state: OptimizationState, update: EvaluationUpdate
    ) -> OptimizationState:
        if self.feedback_builder is None or update.result is None:
            return state
        evaluation = update.result
        owner = self._owner(state, int(update.request_id))
        result: FeedbackResult = self.feedback_builder.build(
            owner,
            state.pending_evaluations[int(update.request_id)].prediction
            or state.predictions,
            evaluation,
            update.candidate_ids,
            state,
        )
        if len(result.candidate_ids):
            rows = self._rows(state, result.candidate_ids, int(update.request_id))
            values = {"f": result.f}
            if result.g is not None and "g" in owner.schema:
                values["g"] = result.g
            if result.cv is not None and "cv" in owner.schema:
                values["cv"] = result.cv
            owner.update_rows(rows, values)
        return state.replace(feedback_result=result)

    def _feedback_consumer(self) -> Any:
        return self.algorithm

    def _apply_feedback_delivery(
        self,
        state: OptimizationState,
        update: EvaluationUpdate,
    ) -> OptimizationState:
        if (
            self.algorithm is None
            or state.feedback_result is None
            or len(state.feedback_result.candidate_ids) == 0
        ):
            return state
        proposal_id = self._proposal_id(state, int(update.request_id))
        sequence = self._next_feedback_sequence(proposal_id, int(update.sequence))
        feedback = _feedback_batch_from_result(
            state.feedback_result,
            proposal_id=proposal_id,
            channel="true",
            final=update.status
            in {
                EvaluationStatus.COMPLETED,
                EvaluationStatus.FAILED,
                EvaluationStatus.CANCELLED,
            },
            sequence=sequence,
        )
        if self._feedback_accumulator is not None:
            use_accumulator = self._deliver_accumulated_feedback(
                state, update, feedback
            )
            if not use_accumulator:
                self._feedback_accumulator.discard(feedback.proposal_id)
                self._sync_feedback_accumulator(state)
                return state
            else:
                ready = self._feedback_accumulator.pop_ready()
                self._sync_feedback_accumulator(state)
                if ready is None:
                    return state
                feedback = ready
        consumer = self._feedback_consumer()
        tell_offspring = self._feedback_proposal_owners.get(feedback.proposal_id)
        if tell_offspring is None and self._feedback_accumulator is None:
            owner = self._owner(state, int(update.request_id))
            rows = self._rows(state, update.candidate_ids, int(update.request_id))
            tell_offspring = owner.extract(rows)
        deliver_feedback(
            consumer,
            feedback,
            state,
            dispatch=(
                self.callback_manager.dispatch
                if self.callback_manager is not None
                else None
            ),
            offspring=tell_offspring,
        )
        self._feedback_proposal_owners.pop(feedback.proposal_id, None)
        return state

    def _deliver_accumulated_feedback(
        self,
        state: OptimizationState,
        update: EvaluationUpdate,
        feedback: FeedbackBatch,
    ) -> bool:
        accumulator = self._feedback_accumulator
        if accumulator is None:
            return False
        proposal_ids = self._proposal_candidate_ids(state, update, feedback.proposal_id)
        if not proposal_ids:
            return False
        candidates = self._capture_proposal_owner(
            state,
            feedback.proposal_id,
            proposal_ids,
        )
        schema = feedback.observations.schema
        quantities = tuple(
            QuantityRequirement(
                quantity=QuantityRef(kind=kind, index=index),
                sources=accumulator.contract.accepted_sources,
            )
            for kind in schema.quantity_kinds
            for index in schema.indices(kind)
        )
        proposal = ProposalBatch(
            proposal_id=feedback.proposal_id,
            candidates=candidates,
            relations=ProposalRelations(row_count=len(candidates)),
            requirements=FeedbackRequirement(quantities=quantities),
        )
        accumulator.register(proposal)
        accumulator.add(
            FeedbackBatch(
                proposal_id=feedback.proposal_id,
                observations=feedback.observations,
                channel=feedback.channel,
                final=False,
                sequence=feedback.sequence,
            )
        )
        self._sync_feedback_accumulator(state)
        if update.status is EvaluationStatus.FAILED and self._retryable_remaining(
            state, update
        ):
            return True
        if update.status not in {
            EvaluationStatus.COMPLETED,
            EvaluationStatus.FAILED,
            EvaluationStatus.CANCELLED,
        }:
            return True
        request_id = int(update.request_id)
        other_ids: set[int] = set()
        for other_id, pending in state.pending_evaluations.items():
            if (
                int(other_id) != request_id
                and pending.original_candidate_ids is not None
            ):
                other_ids.update(map(int, pending.original_candidate_ids))
        # Include deferred requests in the completion boundary before materialization.
        plan = state.evaluation_plan
        try:
            current_proposal = int(state.get_state(PROPOSALS_CURRENT))
        except KeyError:
            current_proposal = None
        if plan is not None and current_proposal == feedback.proposal_id:
            plan_state = state.evaluation_plan_state
            terminal_plan_ids = set(plan_state.completed if plan_state else ()) | set(
                plan_state.acknowledged if plan_state else ()
            )
            for planned in plan.requests:
                planned_id = int(planned.request_id)
                if planned_id != request_id and planned_id not in terminal_plan_ids:
                    other_ids.update(map(int, planned.candidate_ids))
        if proposal_ids.intersection(other_ids):
            return True
        try:
            accumulator.finalize(feedback.proposal_id)
        except ValidationError:
            accumulator.discard(feedback.proposal_id)
            self._sync_feedback_accumulator(state)
            return False
        self._sync_feedback_accumulator(state)
        return True

    def _proposal_candidate_ids(
        self,
        state: OptimizationState,
        update: EvaluationUpdate,
        proposal_id: int,
    ) -> frozenset[int]:
        candidate_ids = set(self._feedback_proposal_candidates.get(proposal_id, ()))
        pending = state.pending_evaluations.get(int(update.request_id))
        if pending is not None and pending.original_candidate_ids is not None:
            candidate_ids.update(map(int, pending.original_candidate_ids))
        for other in state.pending_evaluations.values():
            if (
                other.request.metadata.get("proposal_id") == proposal_id
                and other.original_candidate_ids is not None
            ):
                candidate_ids.update(map(int, other.original_candidate_ids))
        plan = state.evaluation_plan
        try:
            current_proposal = int(state.get_state(PROPOSALS_CURRENT))
        except KeyError:
            current_proposal = None
        if plan is not None and current_proposal == proposal_id:
            for request in plan.requests:
                candidate_ids.update(map(int, request.candidate_ids))
        result = frozenset(candidate_ids)
        self._feedback_proposal_candidates[proposal_id] = result
        return result

    def _next_feedback_sequence(self, proposal_id: int, requested: int) -> int:
        persisted = (
            self._feedback_accumulator.last_sequence(proposal_id)
            if self._feedback_accumulator is not None
            else -1
        )
        sequence = max(
            requested,
            self._feedback_sequences.get(proposal_id, -1) + 1,
            persisted + 1,
        )
        self._feedback_sequences[proposal_id] = sequence
        return sequence

    def _retryable_remaining(
        self, state: OptimizationState, update: EvaluationUpdate
    ) -> bool:
        if update.status is not EvaluationStatus.FAILED:
            return False
        pending = state.pending_evaluations.get(int(update.request_id))
        if pending is None or pending.original_candidate_ids is None:
            return False
        applied = np.unique(
            np.concatenate([pending.applied_candidate_ids, update.candidate_ids])
        )
        remaining = np.setdiff1d(pending.original_candidate_ids, applied)
        return bool(remaining.size and pending.retry_count < self.retry_limit)

    def _restore_accumulated_feedback(
        self, state: OptimizationState, pending: PendingEvaluation
    ) -> None:
        if self._feedback_accumulator is None or self.feedback_builder is None:
            return
        for update in pending.buffered_updates:
            if update.result is None or update.status in {
                EvaluationStatus.COMPLETED,
                EvaluationStatus.FAILED,
                EvaluationStatus.CANCELLED,
            }:
                continue
            feedback_result = self.feedback_builder.build(
                self._owner(state, int(update.request_id)),
                pending.prediction or state.predictions,
                update.result,
                update.candidate_ids,
                state,
            )
            proposal_id = self._proposal_id(state, int(update.request_id))
            feedback = _feedback_batch_from_result(
                feedback_result,
                proposal_id=proposal_id,
                channel="true",
                final=False,
                sequence=int(update.sequence),
            )
            # The snapshot is authoritative; buffered updates are a transport log.
            if self._feedback_accumulator.has_delivery(proposal_id, feedback.sequence):
                continue
            self._deliver_accumulated_feedback(
                state,
                update,
                feedback,
            )
        self._sync_feedback_accumulator(state)
