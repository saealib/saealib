"""Failure matrix for the transition evaluation ledger.

Every failure kind and position must leave the confirmed evaluation ledger
(``state.fe``, the History rows and counters) at the pre-commit value plus the
rows truly confirmed before the failure, and everything outside the ledger at
the pre-commit value, so the same replacement can run again and evaluate only
the remaining rows.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from saealib import EvaluateRequest, PostEvaluationRequestEvent
from saealib.exceptions import EvaluationFatalError
from saealib.execution import AsyncEvaluationScheduler, SerialEvaluator
from saealib.execution.evaluator import (
    AsyncEvaluator,
    EvaluationErrorInfo,
    EvaluationStatus,
    EvaluationUpdate,
)
from saealib.strategies import DirectStrategy as _DirectStrategy
from saealib.transition import Transition, TransitionContext
from tests.integration.test_transition_evaluation import (
    _commit_point,
    _CountingEvaluator,
    _observable_vector,
    _optimizer,
    _Reevaluate,
    _runtime_vector,
)

_INITIAL_ROWS = 6
_INITIAL_FE = 6
_KINDS = [
    "evaluator",
    "terminal_failed",
    "terminal_cancelled",
    "history_runtime",
    "history_validation",
    "handler",
    "later_submit",
    "transition_body",
    "post_validation",
    "publication",
]
_POSITIONS = ["first", "after", "last"]


def _expected_confirmed(
    kind: str,
    target: int,
    *,
    asynchronous: bool,
    sizes: list[int],
    max_pending: int,
) -> int | None:
    """Return the rows the failure must leave confirmed, when deterministic."""
    if kind in {"post_validation", "publication"}:
        return sum(sizes)
    if kind == "transition_body":
        return sum(sizes[:target])
    if kind == "later_submit":
        return sum(sum(group) for group in _submits(sizes, max_pending)[: target - 1])
    if not asynchronous:
        if kind in {"evaluator", "history_runtime", "history_validation"}:
            return sum(sizes[: target - 1])
        if kind == "handler":
            return sum(sizes[:target])
        return None
    if kind in {"evaluator", "terminal_failed", "terminal_cancelled"}:
        batches = _submits(sizes, max_pending)
        index = (target - 1) // max_pending
        return sum(sum(group) for group in batches[: index + 1]) - sizes[target - 1]
    return None


class _ScriptedTransition(Transition):
    """Yield several evaluation requests and optionally raise after one."""

    def __init__(self, chunks: int, *, fail_after: int | None = None) -> None:
        self.chunks = chunks
        self.fail_after = fail_after
        self.outcomes: list[Any] = []

    def apply(self, tctx: TransitionContext) -> Any:
        assert tctx.state is not None
        population = tctx.state.population
        assert population is not None
        ids = np.asarray(population.get_array("id"), dtype=np.int64)
        for index, chunk in enumerate(np.array_split(ids, self.chunks)):
            self.outcomes.append((yield EvaluateRequest(chunk)))
            if self.fail_after == index:
                raise RuntimeError("transition body failed")


class _ScriptedEvaluator(_CountingEvaluator):
    """Fail the evaluation request at a chosen index."""

    def __init__(self) -> None:
        super().__init__()
        self.mode: str | None = None
        self.fail_index = 0
        self.requests = 0

    def evaluate_batch(self, x: Any, problem: Any) -> Any:
        self.requests += 1
        if self.mode == "error" and self.requests == self.fail_index:
            raise RuntimeError("objective failed")
        return super().evaluate_batch(x, problem)

    def evaluate_request(self, request: Any, problem: Any) -> Any:
        if self.mode == "error":
            self.requests += 1
            if self.requests == self.fail_index:
                raise RuntimeError("objective failed")
        return super().evaluate_request(request, problem)


class _ScriptedAsyncEvaluator(AsyncEvaluator):
    """Cancel the evaluation request at a chosen index."""

    def __init__(self, evaluator: Any) -> None:
        super().__init__(evaluator, max_workers=2)
        self.mode: str | None = None
        self.fail_index = 0
        self.collections = 0
        self.seen: set[int] = set()

    def collect(self, handle: Any, *, wait: bool = True) -> Any:
        request_id = int(handle.request_id)
        if self.mode == "cancelled" and request_id not in self.seen:
            self.seen.add(request_id)
            self.collections += 1
            if self.collections == self.fail_index:
                sequence = handle._delivered_sequence + 1
                handle._delivered_sequence = sequence
                return [
                    EvaluationUpdate(
                        handle.request_id,
                        EvaluationStatus.CANCELLED,
                        np.empty(0, dtype=np.int64),
                        error=EvaluationErrorInfo("Cancelled", "cancelled by test"),
                        sequence=sequence,
                    )
                ]
        return super().collect(handle, wait=wait)


def _chunk_sizes(count: int, capacity: int | None) -> list[int]:
    if capacity is None:
        return [2] * 3
    number = max(1, -(-count // capacity))
    return [len(chunk) for chunk in np.array_split(np.arange(count), number)]


def _submits(sizes: list[int], capacity: int) -> list[list[int]]:
    return [sizes[start : start + capacity] for start in range(0, len(sizes), capacity)]


def _build(asynchronous: bool, max_pending: int) -> tuple[Any, Any, Any, Any]:
    evaluator = _ScriptedEvaluator()
    if not asynchronous:
        optimizer = _optimizer(evaluator, strategy=_DirectStrategy())
        return evaluator, optimizer, None, None
    async_evaluator = _ScriptedAsyncEvaluator(evaluator)
    scheduler = AsyncEvaluationScheduler(async_evaluator, max_pending=max_pending)
    optimizer = _optimizer(async_evaluator, strategy=_DirectStrategy())
    optimizer.set_async_evaluation_scheduler(scheduler)
    return evaluator, optimizer, scheduler, async_evaluator


def _matrix(
    *,
    kind: str,
    position: str,
    asynchronous: bool,
    max_pending: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capacity = max_pending if asynchronous else None
    sizes = _chunk_sizes(_INITIAL_ROWS, capacity)
    target = {"first": 1, "after": 2, "last": len(sizes)}[position]
    if kind in {"post_validation", "publication"}:
        target = len(sizes)
    chunks = len(sizes) if (not asynchronous or kind == "transition_body") else 1

    # Skip combinations the path cannot express.
    if not asynchronous and kind in {
        "terminal_failed",
        "terminal_cancelled",
        "later_submit",
    }:
        pytest.skip("terminal updates and batch submits need the async path")
    if asynchronous and kind == "later_submit":
        submits = _submits(sizes, max_pending)
        if len(submits) < target:
            pytest.skip("not enough batches for this position")
    if kind in {
        "evaluator",
        "terminal_failed",
        "terminal_cancelled",
        "history_runtime",
        "history_validation",
        "handler",
        "transition_body",
    } and target > len(sizes):
        pytest.skip("not enough requests for this position")

    evaluator, optimizer, scheduler, async_evaluator = _build(asynchronous, max_pending)
    _runner, captured = _commit_point(optimizer)
    environment = captured["environment"]
    state = captured["session"].state
    before = _observable_vector(state, scheduler)
    history = state.history
    assert history is not None

    if kind in {"evaluator", "terminal_failed"}:
        evaluator.requests = 0
        evaluator.mode = "error"
        evaluator.fail_index = target
    elif kind == "terminal_cancelled":
        async_evaluator.mode = "cancelled"
        async_evaluator.fail_index = target
    elif kind in {"history_runtime", "history_validation"}:
        error_type = RuntimeError if kind == "history_runtime" else ValueError

        def fail_append(channel: str, *args: Any, **kwargs: Any) -> Any:
            if channel != "evaluation":
                return append(channel, *args, **kwargs)
            append_calls[0] += 1
            if append_calls[0] == target:
                raise error_type("history append failed")
            return append(channel, *args, **kwargs)

        append = history.append_block
        append_calls = [0]
        monkeypatch.setattr(history, "append_block", fail_append)
    elif kind == "handler":
        handler_calls = [0]

        def raise_from_handler(event: Any) -> None:
            handler_calls[0] += 1
            if handler_calls[0] == target:
                raise RuntimeError("observer failed")

        optimizer.cbmanager.register(PostEvaluationRequestEvent, raise_from_handler)
    elif kind == "later_submit":
        submit_calls = [0]
        batches = _submits(sizes, max_pending)
        fail_call = sum(len(group) for group in batches[: target - 1]) + 1
        submit = async_evaluator.submit

        def fail_submit(request: Any, problem: Any) -> Any:
            submit_calls[0] += 1
            if submit_calls[0] == fail_call:
                raise RuntimeError("later submit failed")
            return submit(request, problem)

        monkeypatch.setattr(async_evaluator, "submit", fail_submit)
    elif kind == "post_validation":

        def fail_validation(*args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("post-validation failed")

        monkeypatch.setattr(environment, "_check_staged_changes", fail_validation)
    elif kind == "publication":

        def fail_publication(*args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("publication failed")

        monkeypatch.setattr(environment, "_publish_prepared", fail_publication)

    fail_after = target - 1 if kind == "transition_body" else None
    transition = _ScriptedTransition(chunks, fail_after=fail_after)
    optimizer.set_strategy(_DirectStrategy(), transition=transition)
    with pytest.raises((RuntimeError, ValueError, EvaluationFatalError)):
        environment._refresh_plan_if_needed(state)

    if kind == "handler":
        optimizer.cbmanager.unregister(PostEvaluationRequestEvent, raise_from_handler)

    # Expected ledger: rows of the requests confirmed before the failure.
    expected = _expected_confirmed(
        kind, target, asynchronous=asynchronous, sizes=sizes, max_pending=max_pending
    )
    failed = environment._last_state
    assert failed is not None
    population = failed.population
    confirmed = int(np.sum(np.asarray(population.get_array("objective_source")) == 0))
    channel = failed.history.channel("evaluation")
    assert int(failed.fe) == _INITIAL_FE + confirmed
    assert int(np.sum(channel["size"])) == _INITIAL_FE + confirmed
    assert int(channel["fe_after"][-1]) == _INITIAL_FE + confirmed
    assert int(failed.history._fe_counter) == _INITIAL_FE + confirmed
    assert confirmed <= _INITIAL_ROWS
    if expected is not None:
        assert confirmed == expected
    assert _runtime_vector(_observable_vector(failed, scheduler)) == _runtime_vector(
        before
    )
    assert environment.active_version == 0

    # Recover and rerun the same replacement: only the remaining rows run.
    evaluator.mode = None
    if async_evaluator is not None:
        async_evaluator.mode = None
    monkeypatch.undo()
    rerun = _Reevaluate()
    optimizer.set_strategy(_DirectStrategy(), transition=rerun)
    resumed = environment._refresh_plan_if_needed(failed)

    assert resumed is not None
    assert environment.active_version == 1
    assert len(rerun.outcomes) == 1
    assert len(rerun.outcomes[0].candidate_ids) == _INITIAL_ROWS - confirmed
    channel = resumed.history.channel("evaluation")
    assert int(resumed.fe) == _INITIAL_ROWS + _INITIAL_FE
    assert int(np.sum(channel["size"])) == _INITIAL_ROWS + _INITIAL_FE
    assert int(channel["fe_after"][-1]) == _INITIAL_ROWS + _INITIAL_FE
    assert not resumed.pending_evaluations
    assert not resumed.evaluation_handles


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("max_pending", [1, 2, 3, 6])
@pytest.mark.parametrize("position", _POSITIONS)
@pytest.mark.parametrize("kind", _KINDS)
def test_a_failure_keeps_exactly_the_confirmed_evaluation_ledger(
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
    position: str,
    max_pending: int,
    asynchronous: bool,
) -> None:
    _matrix(
        kind=kind,
        position=position,
        asynchronous=asynchronous,
        max_pending=max_pending,
        monkeypatch=monkeypatch,
    )


class _StreamingEvaluator(SerialEvaluator):
    """Deliver one PARTIAL update, then the remaining rows as COMPLETED."""

    def __init__(self) -> None:
        super().__init__()
        self.complete = False
        self.requests: dict[int, tuple[Any, Any]] = {}
        self.steps: dict[int, int] = {}

    def supports_batch_rollback(self) -> bool:
        return True

    def submit(self, request: Any, problem: Any) -> Any:
        from saealib.execution.evaluator import EvaluationHandle

        self.requests[int(request.request_id)] = (request, problem)
        self.steps[int(request.request_id)] = 0
        return EvaluationHandle(
            request.request_id,
            EvaluationStatus.PENDING,
            _candidate_ids=request.candidate_ids,
        )

    def collect(self, handle: Any, *, wait: bool = True) -> Any:
        request, problem = self.requests[int(handle.request_id)]
        step = self.steps[int(handle.request_id)]
        if step > 1:
            return []
        if self.complete:
            rows = np.arange(len(request.candidate_ids))
            status = EvaluationStatus.COMPLETED
        elif step == 0:
            rows = np.array([0])
            status = EvaluationStatus.PARTIAL
        else:
            rows = np.arange(1, len(request.candidate_ids))
            status = EvaluationStatus.COMPLETED
        from saealib.execution.evaluator import EvaluationRequest

        part = EvaluationRequest(
            request.request_id,
            request.candidate_ids[rows],
            request.payload.take(rows),
            metadata=request.metadata,
        )
        result = SerialEvaluator().evaluate_request(part, problem)
        result.candidate_ids = part.candidate_ids
        result.__post_init__()
        self.steps[int(handle.request_id)] += 1
        handle._delivered_sequence = step
        return [
            EvaluationUpdate(
                request.request_id, status, part.candidate_ids, result, sequence=step
            )
        ]

    def acknowledge(self, handle: Any, sequence: int) -> None:
        super().acknowledge(handle, sequence)

    def cancel(self, handle: Any) -> bool:
        self.requests.pop(int(handle.request_id), None)
        return True

    def detach(self, handle: Any) -> bool:
        return self.cancel(handle)


def _ledger_setup(shape: str) -> tuple[Any, Any, Any, Any, Any]:
    if shape == "partial":
        evaluator = _StreamingEvaluator()
        async_evaluator = evaluator
    else:
        evaluator = _CountingEvaluator()
        async_evaluator = AsyncEvaluator(evaluator, max_workers=1)
    scheduler = AsyncEvaluationScheduler(async_evaluator, max_pending=6)
    optimizer = _optimizer(async_evaluator, strategy=_DirectStrategy())
    optimizer.set_async_evaluation_scheduler(scheduler)
    return evaluator, optimizer, scheduler, async_evaluator, None


@pytest.mark.parametrize("shape", ["completed", "partial"])
@pytest.mark.parametrize("j", [1, 2, 3])
def test_an_exception_after_the_jth_counting_call_keeps_the_ledger(
    monkeypatch: pytest.MonkeyPatch, shape: str, j: int
) -> None:
    import saealib.context as context_module

    evaluator, optimizer, scheduler, async_evaluator, _ = _ledger_setup(shape)
    _runner, captured = _commit_point(optimizer)
    environment = captured["environment"]
    state = captured["session"].state
    before = _observable_vector(state, scheduler)
    original = context_module._finish_evaluation_report
    calls = [0]

    def fail_after(state: Any, report: Any) -> Any:
        result = original(state, report)
        calls[0] += 1
        if calls[0] == j:
            raise RuntimeError("injected after counting call")
        return result

    monkeypatch.setattr(context_module, "_finish_evaluation_report", fail_after)
    optimizer.set_strategy(_DirectStrategy(), transition=_ScriptedTransition(1))
    error: Exception | None = None
    try:
        environment._refresh_plan_if_needed(state)
    except Exception as exc:
        error = exc
    monkeypatch.undo()
    if error is None:
        pytest.skip("fewer counting calls than the injection index")

    failed = environment._last_state
    assert failed is not None
    population = failed.population
    confirmed = int(np.sum(np.asarray(population.get_array("objective_source")) == 0))
    channel = failed.history.channel("evaluation")
    assert int(failed.fe) == _INITIAL_FE + confirmed
    assert int(np.sum(channel["size"])) == _INITIAL_FE + confirmed
    assert int(channel["fe_after"][-1]) == _INITIAL_FE + confirmed
    assert int(failed.history._fe_counter) == _INITIAL_FE + confirmed
    assert _runtime_vector(_observable_vector(failed, scheduler)) == _runtime_vector(
        before
    )
    assert environment.active_version == 0

    if shape == "partial":
        evaluator.complete = True
        async_evaluator.complete = True
    ran_before = (
        len(evaluator.batches)
        if hasattr(evaluator, "batches")
        else len(evaluator.requests)
    )
    rerun = _Reevaluate()
    optimizer.set_strategy(_DirectStrategy(), transition=rerun)
    resumed = environment._refresh_plan_if_needed(failed)

    assert resumed is not None
    assert environment.active_version == 1
    assert len(rerun.outcomes) == 1
    assert len(rerun.outcomes[0].candidate_ids) == _INITIAL_ROWS - confirmed
    channel = resumed.history.channel("evaluation")
    assert int(resumed.fe) == _INITIAL_ROWS + _INITIAL_FE
    assert int(np.sum(channel["size"])) == _INITIAL_ROWS + _INITIAL_FE
    assert int(channel["fe_after"][-1]) == _INITIAL_ROWS + _INITIAL_FE
    assert not resumed.pending_evaluations
    assert not resumed.evaluation_handles
    if confirmed == _INITIAL_ROWS:
        ran_after = (
            len(evaluator.batches)
            if hasattr(evaluator, "batches")
            else len(evaluator.requests)
        )
        assert ran_after == ran_before


def test_a_failure_after_the_rows_are_applied_and_before_acknowledge_keeps_the_ledger(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evaluator = _CountingEvaluator()
    async_evaluator = AsyncEvaluator(evaluator, max_workers=1)
    scheduler = AsyncEvaluationScheduler(async_evaluator, max_pending=6)
    optimizer = _optimizer(async_evaluator, strategy=_DirectStrategy())
    optimizer.set_async_evaluation_scheduler(scheduler)
    _runner, captured = _commit_point(optimizer)
    environment = captured["environment"]
    state = captured["session"].state
    before = _observable_vector(state, scheduler)

    def fail_acknowledge(handle: Any, sequence: int) -> None:
        raise RuntimeError("acknowledge failed")

    monkeypatch.setattr(async_evaluator, "acknowledge", fail_acknowledge)
    optimizer.set_strategy(_DirectStrategy(), transition=_ScriptedTransition(1))
    with pytest.raises(RuntimeError):
        environment._refresh_plan_if_needed(state)
    monkeypatch.undo()

    failed = environment._last_state
    assert failed is not None
    channel = failed.history.channel("evaluation")
    assert int(failed.fe) == _INITIAL_ROWS + _INITIAL_FE
    assert int(np.sum(channel["size"])) == _INITIAL_ROWS + _INITIAL_FE
    assert int(channel["fe_after"][-1]) == _INITIAL_ROWS + _INITIAL_FE
    assert int(failed.history._fe_counter) == _INITIAL_ROWS + _INITIAL_FE
    assert _runtime_vector(_observable_vector(failed, scheduler)) == _runtime_vector(
        before
    )
    assert environment.active_version == 0

    ran_before = len(evaluator.batches)
    rerun = _Reevaluate()
    optimizer.set_strategy(_DirectStrategy(), transition=rerun)
    resumed = environment._refresh_plan_if_needed(failed)

    assert resumed is not None
    assert environment.active_version == 1
    assert len(rerun.outcomes[0].candidate_ids) == 0
    assert len(evaluator.batches) == ran_before
    channel = resumed.history.channel("evaluation")
    assert int(resumed.fe) == _INITIAL_ROWS + _INITIAL_FE
    assert int(np.sum(channel["size"])) == _INITIAL_ROWS + _INITIAL_FE
