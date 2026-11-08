from __future__ import annotations

from typing import Any

from saealib.execution import (
    EvaluationHandle,
    EvaluationRequest,
    EvaluationStatus,
    EvaluationUpdate,
    Evaluator,
    SerialEvaluator,
)
from saealib.problem import Problem


class _Evaluator(Evaluator):
    def __init__(self) -> None:
        self.delivered: dict[int, EvaluationUpdate] = {}
        self.submitted: list[EvaluationRequest] = []
        self._delivered: set[int] = set()
        self._seen_in_pass: set[int] = set()
        self._released_in_pass = False
        self.completion_order: list[int] = []

    def evaluate_batch(self, x, problem):
        return SerialEvaluator().evaluate_batch(x, problem)

    def submit(self, request: EvaluationRequest, problem: Problem):
        self.submitted.append(request)
        return EvaluationHandle(
            request.request_id,
            EvaluationStatus.PENDING,
            backend_token=(request, problem),
        )

    def collect(self, handle: EvaluationHandle, *, wait: bool = True):
        if handle._acknowledged_sequence >= 0:
            return []
        request, problem = handle.backend_token
        request_id = int(request.request_id)
        if request_id in self._seen_in_pass:
            self._seen_in_pass.clear()
            self._released_in_pass = False
        self._seen_in_pass.add(request_id)
        if self._released_in_pass:
            return []
        outstanding = [
            int(item.request_id)
            for item in self.submitted
            if int(item.request_id) not in self._delivered
        ]
        if not outstanding or request_id != min(outstanding):
            return []
        update = self.delivered.get(request_id)
        if update is None:
            result = SerialEvaluator().evaluate_request(request, problem)
            result.candidate_ids = request.candidate_ids
            result.__post_init__()
            update = EvaluationUpdate(
                request.request_id,
                EvaluationStatus.COMPLETED,
                request.candidate_ids,
                result,
                None,
                0,
            )
            self.delivered[request_id] = update
            self._delivered.add(request_id)
            self.completion_order.append(request_id)
        self._released_in_pass = True
        handle._delivered_sequence = update.sequence
        return [update]

    def acknowledge(self, handle: EvaluationHandle, sequence: int) -> None:
        handle._acknowledged_sequence = sequence
        self.delivered.pop(int(handle.request_id), None)

    def supports_batch_rollback(self) -> bool:
        return True

    def can_reattach(self, pending: Any) -> bool:
        return True

    def reattach(self, pending: Any, problem: Problem):
        if pending.request not in self.submitted:
            self.submitted.append(pending.request)
        return EvaluationHandle(
            pending.request.request_id,
            EvaluationStatus.PENDING,
            backend_token=(pending.request, problem),
        )


class _ControlledEvaluator(_Evaluator):
    def __init__(self) -> None:
        super().__init__()
        self.released_request_ids: set[int] = set()
        self.release_all = False

    def collect(self, handle: EvaluationHandle, *, wait: bool = True):
        if handle._acknowledged_sequence >= 0:
            return []
        request, problem = handle.backend_token
        request_id = int(request.request_id)
        if request_id in self._delivered or not (
            self.release_all or request_id in self.released_request_ids
        ):
            return []
        result = SerialEvaluator().evaluate_request(request, problem)
        result.candidate_ids = request.candidate_ids
        result.__post_init__()
        update = EvaluationUpdate(
            request.request_id,
            EvaluationStatus.COMPLETED,
            request.candidate_ids,
            result,
            None,
            0,
        )
        self.delivered[request_id] = update
        self._delivered.add(request_id)
        self.completion_order.append(request_id)
        handle._delivered_sequence = update.sequence
        return [update]
