"""Replacement of optimizer components between commit points (issue #286).

Each slot the Optimizer exposes is bound to a component at a commit point:
a replacement observed there rebuilds the plan and publishes the new active
binding version together with it; nothing half-applies.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from saealib import (
    PSO,
    CrossoverBLXAlpha,
    LHSInitializer,
    MutationUniform,
    Optimizer,
    SequentialSelection,
    Termination,
    TruncationSelection,
)
from saealib.algorithms import GA
from saealib.callback import CallbackManager, GenerationEndEvent, PostAskEvent
from saealib.exceptions import ConfigurationError
from saealib.execution import (
    AsyncEvaluationScheduler,
    EvaluationHandle,
    EvaluationRequest,
    EvaluationStatus,
    EvaluationUpdate,
    Evaluator,
    SerialEvaluator,
)
from saealib.execution.runner import Runner
from saealib.execution.runtime import create_runtime, resolve_plan
from saealib.policies.feedback import TrueOnlyFeedback
from saealib.problem import Problem
from saealib.strategies import DirectStrategy, SteadyStateStrategy


def _problem(dim: int = 2) -> Problem:
    return Problem(
        func=lambda x: np.array([float(np.sum(np.asarray(x) ** 2))], dtype=np.float64),
        dim=dim,
        n_obj=1,
        direction=np.array([-1.0]),
        lb=[-5.0] * dim,
        ub=[5.0] * dim,
        constraints=[],
    )


class _RecordingPSO(PSO):
    def __init__(self) -> None:
        super().__init__()
        self.asks: list[int] = []
        self.tell_ids: list[tuple[int, ...]] = []
        self.ask_xs: list[tuple[tuple[float, ...], ...]] = []
        self.tell_xs: list[tuple[tuple[float, ...], ...]] = []

    def ask(self, request: Any, state: Any) -> Any:
        batch = super().ask(request, state)
        self.asks.append(len(batch.candidates))
        self.ask_xs.append(_as_x_rows(batch.candidates))
        return batch

    def tell(self, feedback: Any, state: Any) -> Any:
        population = state.context.offspring
        self.tell_ids.append(tuple(np.asarray(population.get_array("id")).tolist()))
        self.tell_xs.append(_as_x_rows(population))
        return super().tell(feedback, state)


class _RecordingGA(GA):
    def __init__(self) -> None:
        super().__init__(
            crossover=CrossoverBLXAlpha(prob=0.9, alpha=0.4),
            mutation=MutationUniform(prob_var=0.1),
            parent_selection=SequentialSelection(),
            survivor_selection=TruncationSelection(),
        )
        self.asks: list[int] = []
        self.tell_ids: list[tuple[int, ...]] = []
        self.ask_xs: list[tuple[tuple[float, ...], ...]] = []
        self.tell_xs: list[tuple[tuple[float, ...], ...]] = []

    def ask(self, request: Any, state: Any) -> Any:
        batch = super().ask(request, state)
        self.asks.append(len(batch.candidates))
        self.ask_xs.append(_as_x_rows(batch.candidates))
        return batch

    def tell(self, feedback: Any, state: Any) -> Any:
        population = state.context.offspring
        self.tell_ids.append(tuple(np.asarray(population.get_array("id")).tolist()))
        self.tell_xs.append(_as_x_rows(population))
        return super().tell(feedback, state)


def _as_x_rows(population: Any) -> tuple[tuple[float, ...], ...]:
    return tuple(
        tuple(float(value) for value in row)
        for row in np.asarray(population.get_array("x")).tolist()
    )


class _DeterministicEvaluator(Evaluator):
    """Evaluator with fully controlled, wall-clock-free completion.

    Within one poll pass only the rank-min outstanding request is released,
    so completion order follows the submission order regardless of timing.
    ``release_after_collects`` gates each handle by how many polls it must
    observe before it may deliver; ``max_collects`` is a hard loop cap used
    to fail fast instead of hanging when a drain rule does not terminate.
    """

    def __init__(
        self,
        order: str = "fifo",
        release_after_collects: int = 1,
        max_collects: int = 2000,
    ) -> None:
        self.order = order
        self.release_after_collects = release_after_collects
        self.max_collects = max_collects
        self.submitted: list[EvaluationRequest] = []
        self.collect_calls = 0
        self.handle_collects: dict[int, int] = {}
        self.acks: list[tuple[int, int]] = []
        self.reattached: list[int] = []
        self._handles: dict[int, EvaluationHandle] = {}
        self._delivered: set[int] = set()
        self.problem = _problem()
        self.on_ack: Any = None

    def evaluate_batch(self, x: Any, problem: Any) -> Any:
        return SerialEvaluator().evaluate_batch(x, problem)

    def submit(self, request: EvaluationRequest, problem: Any) -> EvaluationHandle:
        self.submitted.append(request)
        handle = EvaluationHandle(
            request.request_id,
            EvaluationStatus.PENDING,
            backend_token=(request, problem),
        )
        self._handles[int(request.request_id)] = handle
        return handle

    def collect(self, handle: EvaluationHandle, *, wait: bool = True) -> list:
        self.collect_calls += 1
        if self.collect_calls > self.max_collects:
            raise RuntimeError("deterministic evaluator collect cap exceeded")
        request, problem = handle.backend_token
        request_id = int(request.request_id)
        if handle._acknowledged_sequence >= 0 or request_id in self._delivered:
            return []
        self.handle_collects[request_id] = self.handle_collects.get(request_id, 0) + 1
        if self.handle_collects[request_id] < self.release_after_collects:
            return []
        outstanding = [
            int(item.request_id)
            for item in self.submitted
            if int(item.request_id) not in self._delivered
        ]
        if request_id not in outstanding:
            return []
        ranks = {
            int(item.request_id): self._rank(index)
            for index, item in enumerate(self.submitted)
        }
        next_release = min(outstanding, key=lambda rid: ranks[rid])
        if request_id != next_release:
            return []
        result = SerialEvaluator().evaluate_request(request, problem)
        result.candidate_ids = request.candidate_ids
        result.__post_init__()
        self._delivered.add(request_id)
        handle._delivered_sequence = 0
        return [
            EvaluationUpdate(
                request.request_id,
                EvaluationStatus.COMPLETED,
                request.candidate_ids,
                result,
                None,
                0,
            )
        ]

    def acknowledge(self, handle: EvaluationHandle, sequence: int) -> None:
        self.acks.append((int(handle.request_id), sequence))
        handle._acknowledged_sequence = sequence
        if self.on_ack is not None:
            self.on_ack(int(handle.request_id), sequence)

    def supports_batch_rollback(self) -> bool:
        return True

    def can_reattach(self, pending: Any) -> bool:
        return True

    def reattach(self, pending: Any, problem: Any) -> EvaluationHandle:
        if pending.request not in self.submitted:
            self.submitted.append(pending.request)
        self.reattached.append(int(pending.request.request_id))
        return EvaluationHandle(
            pending.request.request_id,
            EvaluationStatus.PENDING,
            backend_token=(pending.request, problem),
        )

    def _rank(self, submit_index: int) -> int:
        if self.order == "fifo":
            return submit_index
        return len(self.submitted) - 1 - submit_index


class _RecordingFeedback(TrueOnlyFeedback):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[tuple[int, ...]] = []

    def build(self, *args: Any, **kwargs: Any) -> Any:
        candidates = args[0]
        try:
            ids = tuple(int(value) for value in candidates.get_array("id"))
        except Exception:
            ids = ()
        self.calls.append(ids)
        return super().build(*args, **kwargs)


def _build_optimizer(algorithm: Any, seed: int = 7):
    return (
        Optimizer(_problem(), seed=seed)
        .set_initializer(LHSInitializer(6, 4, seed))
        .set_algorithm(algorithm)
        .set_strategy(DirectStrategy())
        .set_termination(Termination(lambda state: state.gen >= 10))
    )


def _capturing_runner(optimizer: Any) -> tuple[Runner, dict[str, Any]]:
    """Runner plus a captured handle to its runtime environment."""
    captured: dict[str, Any] = {}

    def factory(optimizer_obj: Any) -> Any:
        runtime = create_runtime(optimizer_obj)
        captured["environment"] = getattr(runtime, "environment", None)
        return runtime

    return Runner(optimizer, runtime_factory=factory), captured


# ---------------------------------------------------------------------------
# Synchronous commit points
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("use_setter", [True, False])
def test_direct_replacement_splits_asks_and_tells_by_generation(
    use_setter: bool,
) -> None:
    pso = _RecordingPSO()
    ga = _RecordingGA()
    optimizer = _build_optimizer(pso)
    swap_generations: list[int] = []

    def on_generation_end(event: Any) -> None:
        if event.ctx.gen == 5:
            swap_generations.append(event.ctx.gen)
            if use_setter:
                optimizer.set_algorithm(ga)
            else:
                optimizer.algorithm = ga

    cbm = CallbackManager()
    cbm.register(GenerationEndEvent, on_generation_end)
    optimizer.cbmanager = cbm
    state = optimizer.run()

    assert state.gen == 10
    assert swap_generations == [5]
    assert len(pso.asks) == 5
    assert len(pso.tell_ids) == 5
    assert len(ga.asks) == 5
    assert len(ga.tell_ids) == 5
    assert all(size == 4 for size in pso.asks + ga.asks)


def test_callback_replacement_applies_from_the_next_step() -> None:
    pso = _RecordingPSO()
    ga = _RecordingGA()
    optimizer = _build_optimizer(pso)
    swapped = []

    def on_ask(event: Any) -> None:
        # The step that requested this ask produced its candidates with the
        # old algorithm; the replacement waits for the next commit point.
        if len(pso.asks) >= 2 and not swapped:
            swapped.append(len(pso.asks))
            optimizer.set_algorithm(ga)

    cbm = CallbackManager()
    cbm.register(PostAskEvent, on_ask)
    optimizer.cbmanager = cbm
    state = optimizer.run()

    assert state.gen == 10
    assert swapped == [2]
    assert pso.asks == [4] * 3
    assert len(pso.tell_ids) == 3
    assert ga.asks == [4] * 7
    assert len(ga.tell_ids) == 7


def test_several_slots_replaced_between_yields_publish_together() -> None:
    pso = _RecordingPSO()
    ga = _RecordingGA()
    old_builder = _RecordingFeedback()
    new_builder = _RecordingFeedback()
    optimizer = _build_optimizer(pso)
    optimizer.set_feedback_builder(old_builder)

    runner, captured = _capturing_runner(optimizer)
    version_before: int | None = None
    for state in runner.iterate():
        if state.gen == 2 and version_before is None:
            version_before = captured["environment"].active_version
            optimizer.set_algorithm(ga)
            optimizer.set_feedback_builder(new_builder)

    assert version_before is not None
    assert captured["environment"].active_version == version_before + 1
    assert len(old_builder.calls) == 2
    assert len(new_builder.calls) == 8
    assert ga.asks == [4] * 8
    assert len(ga.tell_ids) == 8


def test_replacement_with_error_diagnostic_keeps_previous_plan_usable() -> None:
    pso = _RecordingPSO()
    ga = _RecordingGA()
    optimizer = _build_optimizer(pso)
    runner, captured = _capturing_runner(optimizer)

    states: list[Any] = []
    generator = runner.iterate()
    states.append(next(generator))
    states.append(next(generator))
    version_before = captured["environment"].active_version
    plan_before = optimizer._executable_plan
    optimizer.evaluation_planner = object()  # type: ignore[assignment]

    with pytest.raises(ConfigurationError) as error:
        next(generator)
    assert "evaluation_planner must be an EvaluationPlanner" in str(error.value)
    generator.close()
    assert optimizer._executable_plan is plan_before
    assert captured["environment"].active_version == version_before
    assert len(pso.asks) == 1

    optimizer.evaluation_planner = None
    optimizer.set_algorithm(ga)
    for state in runner.iterate_from(states[-1]):
        states.append(state)

    assert states[-1].gen == 10
    assert ga.asks == [4] * 9
    assert len(ga.tell_ids) == 9


def test_in_place_strategy_change_triggers_rebuild() -> None:
    ga = _RecordingGA()
    optimizer = _build_optimizer(ga)
    runner, captured = _capturing_runner(optimizer)

    generator = runner.iterate()
    assert next(generator).gen == 0
    assert next(generator).gen == 1
    version_before = captured["environment"].active_version
    optimizer.strategy.n_offspring = 2
    assert next(generator).gen == 2
    generator.close()

    assert version_before == 0
    assert captured["environment"].active_version == version_before + 1
    assert ga.asks == [4, 2]
    assert len(ga.tell_ids) == 2


def _build_async_optimizer(
    algorithm: Any,
    strategy: Any,
    evaluator: _DeterministicEvaluator,
    *,
    max_pending: int = 1,
    n_init: int = 6,
    population: int = 4,
    termination: Any = None,
    seed: int = 42,
):
    optimizer = (
        Optimizer(_problem(), seed=seed)
        .set_initializer(LHSInitializer(n_init, population, seed))
        .set_algorithm(algorithm)
        .set_strategy(strategy)
        .set_evaluator(evaluator)
        .set_async_evaluation_scheduler(
            AsyncEvaluationScheduler(evaluator, max_pending=max_pending)
        )
        .set_termination(termination or Termination(lambda state: state.fe >= 12))
    )
    return optimizer


# ---------------------------------------------------------------------------
# Asynchronous commit points
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("use_setter", [True, False])
def test_async_direct_replacement_splits_asks_and_tells_by_generation(
    use_setter: bool,
) -> None:
    pso = _RecordingPSO()
    ga = _RecordingGA()
    evaluator = _DeterministicEvaluator(release_after_collects=1, max_collects=4000)
    ten_generations = Termination(lambda state: state.gen >= 10)
    optimizer = _build_async_optimizer(
        pso, DirectStrategy(), evaluator, termination=ten_generations
    )

    swap_generations: list[int] = []

    def on_generation_end(event: Any) -> None:
        if event.ctx.gen == 5:
            swap_generations.append(event.ctx.gen)
            if use_setter:
                optimizer.set_algorithm(ga)
            else:
                optimizer.algorithm = ga

    cbm = CallbackManager()
    cbm.register(GenerationEndEvent, on_generation_end)
    optimizer.cbmanager = cbm
    state = optimizer.run()

    assert swap_generations == [5]
    assert not state.pending_evaluations
    assert len(pso.asks) == 5
    assert len(pso.tell_ids) == 5
    assert len(ga.asks) == 5
    assert len(ga.tell_ids) == 5


def test_setters_stay_inactive_until_publication() -> None:
    pso = _RecordingPSO()
    ga = _RecordingGA()
    evaluator = _DeterministicEvaluator(release_after_collects=1, max_collects=4000)
    optimizer = _build_async_optimizer(
        pso,
        DirectStrategy(),
        evaluator,
        termination=Termination(lambda state: state.gen >= 4),
    )
    scheduler = optimizer.async_evaluation_scheduler

    swapped = False
    for state in optimizer.iterate():
        if state.gen == 2 and not swapped:
            swapped = True
            optimizer.set_algorithm(ga)
            assert scheduler.algorithm is pso

    assert swapped
    assert scheduler.algorithm is ga
    assert len(ga.asks) == 2
    assert len(pso.asks) == 2


def test_steady_state_replacement_drains_outstanding_proposals() -> None:
    pso = _RecordingPSO()
    ga = _RecordingGA()
    evaluator = _DeterministicEvaluator(release_after_collects=1, max_collects=20000)
    optimizer = _build_async_optimizer(
        pso,
        SteadyStateStrategy(),
        evaluator,
        max_pending=2,
        n_init=6,
        population=6,
        termination=Termination(
            lambda state: state.fe + len(state.pending_evaluations) >= 14
        ),
    )
    scheduler = optimizer.async_evaluation_scheduler

    ack_log: list[tuple[int, int, int]] = []

    def on_ack(request_id: int, sequence: int) -> None:
        count = len(ack_log) + 1
        ack_log.append((count, int(request_id), len(evaluator.submitted)))
        if count == 1:
            optimizer.set_algorithm(ga)
            assert scheduler.algorithm is pso

    evaluator.on_ack = on_ack
    state = optimizer.run()

    assert not state.pending_evaluations
    assert len(state.evaluation_handles) == 0
    assert len(ack_log) >= 4
    # the second outstanding request of the pre-swap plan drained with the
    # submission count frozen: no refill while a replacement waits
    assert ack_log[1][2] == ack_log[0][2]
    assert ack_log[2][2] > ack_log[1][2]
    # every proposal was told by the algorithm that asked it
    pso_asked = {row for rows in pso.ask_xs for row in rows}
    ga_asked = {row for rows in ga.ask_xs for row in rows}
    assert pso.tell_xs
    assert ga.tell_xs
    assert all(row in pso_asked for rows in pso.tell_xs for row in rows)
    assert all(row in ga_asked for rows in ga.tell_xs for row in rows)
    assert scheduler.algorithm is ga
    assert len(pso.asks) == 1
    assert len(ga.asks) >= 1


@pytest.mark.parametrize(
    "strategy",
    [SteadyStateStrategy(), DirectStrategy(1)],
)
def test_async_reverting_a_pending_replacement_keeps_the_run_running(
    strategy: Any,
) -> None:
    first = _RecordingGA()
    second = _RecordingGA()
    evaluator = _DeterministicEvaluator(release_after_collects=1, max_collects=1000)
    optimizer = _build_async_optimizer(
        first,
        strategy,
        evaluator,
        max_pending=2,
        n_init=6,
        population=6,
        termination=Termination(lambda state: state.fe >= 14),
    )
    optimizer.resolve_defaults()
    optimizer._compile_plan()
    state = optimizer.initializer.initialize(optimizer, optimizer.problem)
    runtime: Any = create_runtime(optimizer)
    environment = runtime.environment
    session: Any = runtime.initialize(resolve_plan(optimizer), state)
    for _ in range(100):
        session = runtime.advance(session).session
        if session.state.pending_evaluations:
            break
    assert session.state.pending_evaluations

    optimizer.set_algorithm(second)
    session = runtime.advance(session).session

    optimizer.set_algorithm(first)
    for _ in range(100):
        session = runtime.advance(session).session
        if session.finished:
            break

    assert session.finished
    assert not session.state.pending_evaluations
    assert session.state.fe >= 14
    assert environment._publication_deferred is False
    assert len(first.asks) >= 2
