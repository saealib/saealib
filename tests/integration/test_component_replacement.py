"""Replacement of optimizer components between commit points.

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
from saealib.core.runtime import NodeResult
from saealib.core.state import StatePatch
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
from saealib.pipeline import Pipeline, Stage
from saealib.policies.evaluation import RepeatedEvaluation
from saealib.policies.feedback import MixedFeedback, TrueOnlyFeedback
from saealib.problem import Problem
from saealib.strategies import (
    DirectStrategy,
    GenerationBasedStrategy,
    IndividualBasedStrategy,
    PreSelectionStrategy,
    SteadyStateStrategy,
)


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
    told_ids = [
        candidate_id
        for algorithm in (pso, ga)
        for batch in algorithm.tell_ids
        for candidate_id in batch
    ]
    assert len(told_ids) == len(set(told_ids))
    assert state.fe == optimizer.initializer.n_init_archive + len(told_ids)
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


def test_direct_async_step_waits_for_the_active_algorithm_to_tell() -> None:
    from saealib.execution.runtime import execute_strategy_step

    events: list[str] = []

    class EventEvaluator(_DeterministicEvaluator):
        def __init__(self) -> None:
            super().__init__(release_after_collects=3, max_collects=1000)
            self.completed_ids: list[int] = []

        def collect(self, handle: EvaluationHandle, *, wait: bool = True) -> list:
            events.append("collect")
            updates = super().collect(handle, wait=wait)
            for update in updates:
                self.completed_ids.extend(int(value) for value in update.candidate_ids)
            return updates

    class EventGA(_RecordingGA):
        def __init__(self, name: str) -> None:
            super().__init__()
            self.name = name

        def ask(self, request: Any, state: Any) -> Any:
            events.append(f"{self.name}:ask")
            return super().ask(request, state)

        def tell(self, feedback: Any, state: Any) -> Any:
            events.append(f"{self.name}:tell")
            return super().tell(feedback, state)

    evaluator = EventEvaluator()
    first_algorithm = EventGA("first")
    next_algorithm = EventGA("next")
    optimizer = _build_async_optimizer(
        first_algorithm,
        SteadyStateStrategy(),
        evaluator,
        max_pending=2,
        termination=Termination(lambda state: state.gen >= 4),
    )
    optimizer.resolve_defaults()
    state = optimizer.initializer.initialize(optimizer, optimizer.problem)

    state = execute_strategy_step(optimizer.strategy, state, optimizer)

    assert state.pending_evaluations
    pending_ids = tuple(
        sorted(
            int(candidate_id)
            for pending in state.pending_evaluations.values()
            for candidate_id in pending.request.candidate_ids
        )
    )
    evaluations_before_drain = state.fe
    events.clear()
    optimizer.algorithm = next_algorithm

    drained = execute_strategy_step(optimizer.strategy, state, optimizer)

    assert evaluator.completed_ids == list(pending_ids)
    told_ids = [
        candidate_id for batch in first_algorithm.tell_ids for candidate_id in batch
    ]
    assert tuple(sorted(told_ids)) == pending_ids
    assert len(told_ids) == len(set(told_ids))
    assert drained.fe == evaluations_before_drain + len(pending_ids)
    assert not drained.pending_evaluations
    assert "next:ask" not in events

    state = execute_strategy_step(optimizer.strategy, drained, optimizer)

    assert events.index("first:tell") < events.index("next:ask")
    assert len(next_algorithm.asks) == 1


def test_async_algorithm_replacement_drains_before_switching_algorithms() -> None:
    events: list[str] = []

    class EventGA(_RecordingGA):
        def __init__(self, name: str) -> None:
            super().__init__()
            self.name = name

        def ask(self, request: Any, state: Any) -> Any:
            events.append(f"{self.name}:ask")
            return super().ask(request, state)

        def tell(self, feedback: Any, state: Any) -> Any:
            events.append(f"{self.name}:tell")
            return super().tell(feedback, state)

    first_algorithm = EventGA("first")
    next_algorithm = EventGA("next")
    optimizer_ref: Optimizer | None = None
    replacement_issued = False

    class ReplacementOnce(Stage):
        name = "replace_algorithm"
        execution_boundary = "state"

        def __init__(self) -> None:
            super().__init__()

        def execute(self, state: Any):
            nonlocal replacement_issued
            if not replacement_issued:
                replacement_issued = True
                assert optimizer_ref is not None
                optimizer_ref.set_algorithm(next_algorithm)
                return NodeResult(patch=StatePatch(writes={}))
            return NodeResult(patch=StatePatch(writes={}))

    replacement_stage = ReplacementOnce()

    class ReplacementStrategy(DirectStrategy):
        def build_template(self) -> Pipeline:
            pipeline = super().build_template()
            pipeline.steps.insert(4, replacement_stage)
            return pipeline

    evaluator = _DeterministicEvaluator(release_after_collects=2, max_collects=1000)
    optimizer = _build_async_optimizer(
        first_algorithm,
        ReplacementStrategy(),
        evaluator,
        max_pending=2,
        termination=Termination(lambda state: state.gen >= 4),
    )
    optimizer_ref = optimizer

    state = optimizer.run()

    assert replacement_issued
    assert events.index("first:tell") < events.index("next:ask")
    assert len(first_algorithm.asks) == 1
    assert len(first_algorithm.tell_ids) == 1
    assert next_algorithm.asks
    assert not state.pending_evaluations
    told_ids = [
        candidate_id
        for algorithm in (first_algorithm, next_algorithm)
        for batch in algorithm.tell_ids
        for candidate_id in batch
    ]
    assert len(told_ids) == len(set(told_ids))
    assert state.fe == optimizer.initializer.n_init_archive + len(told_ids)


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
    told_ids = [
        candidate_id
        for algorithm in (pso, ga)
        for batch in algorithm.tell_ids
        for candidate_id in batch
    ]
    assert len(told_ids) == len(set(told_ids))
    assert state.fe == optimizer.initializer.n_init_archive + len(told_ids)


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


# ---------------------------------------------------------------------------
# Strategy replacement at commit points, both directions
# ---------------------------------------------------------------------------


def _surrogate_optimizer_for_swap(strategy, seed=7):
    from saealib.acquisition import MeanPrediction
    from saealib.surrogate.manager import LocalSurrogateManager
    from saealib.surrogate.rbf import RBFSurrogate
    from saealib.surrogate.rbf_kernels import GaussianKernel

    return (
        Optimizer(_problem(), seed=seed)
        .set_initializer(LHSInitializer(6, 4, seed))
        .set_algorithm(_RecordingGA())
        .set_surrogate_manager(LocalSurrogateManager(RBFSurrogate(GaussianKernel())))
        .set_acquisition(MeanPrediction())
        .set_strategy(strategy)
        .set_termination(Termination(lambda state: state.gen >= 6))
    )


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


def test_replacing_direct_with_generation_based_mid_run_keeps_running():
    optimizer = _surrogate_optimizer_for_swap(DirectStrategy())
    generator = optimizer.iterate()
    next(generator)
    next(generator)
    optimizer.set_strategy(GenerationBasedStrategy(1))
    state = optimizer.run()
    assert state.gen == 6


def test_replacing_generation_based_with_direct_mid_run_keeps_running():
    optimizer = _surrogate_optimizer_for_swap(GenerationBasedStrategy(1), seed=11)
    generator = optimizer.iterate()
    next(generator)
    next(generator)
    optimizer.set_strategy(DirectStrategy())
    state = optimizer.run()
    assert state.gen == 6


def test_strategy_replacement_publishes_one_binding_version():
    from saealib.execution.runner import Runner
    from saealib.execution.runtime import create_runtime

    optimizer = _surrogate_optimizer_for_swap(DirectStrategy(), seed=23)
    captured = {}

    def factory(optimizer_obj):
        runtime = create_runtime(optimizer_obj)
        captured["environment"] = getattr(runtime, "environment", None)
        return runtime

    generator = Runner(optimizer, runtime_factory=factory).iterate()
    next(generator)
    baseline_version = captured["environment"].active_version

    optimizer.set_strategy(GenerationBasedStrategy(1))
    for _state in generator:
        pass

    assert captured["environment"].active_version - baseline_version == 1


def test_async_generation_based_replacement_waits_for_in_flight_requests() -> None:
    first = _RecordingGA()
    second = _RecordingGA()
    evaluator = _DeterministicEvaluator(release_after_collects=1, max_collects=4000)
    optimizer = _surrogate_optimizer_for_swap(GenerationBasedStrategy(1))
    optimizer.set_algorithm(first).set_evaluator(evaluator)
    optimizer.set_async_evaluation_scheduler(
        AsyncEvaluationScheduler(evaluator, max_pending=1)
    )
    swapped: list[int] = []

    def on_post_ask(event: Any) -> None:
        if not swapped and event.ctx.gen == 3:
            swapped.append(len(first.asks))
            optimizer.set_algorithm(second)

    cbm = CallbackManager()
    cbm.register(PostAskEvent, on_post_ask)
    optimizer.cbmanager = cbm
    state = optimizer.run()

    first_asked = {row for rows in first.ask_xs for row in rows}
    second_asked = {row for rows in second.ask_xs for row in rows}
    assert swapped
    assert state.gen >= 6
    assert first.asks and second.asks and second.tell_xs
    assert first_asked.isdisjoint(second_asked)
    assert all(row in first_asked for rows in first.tell_xs for row in rows)
    assert all(row in second_asked for rows in second.tell_xs for row in rows)


# ---------------------------------------------------------------------------
# Objective source of population rows at commit points
# ---------------------------------------------------------------------------


def _last_true_f_by_id(history: Any) -> dict[int, np.ndarray]:
    status = np.asarray(history.channel("evaluation")["status_code"])
    candidate_blocks = history.blocks("evaluation", "candidate_ids")
    f_blocks = history.blocks("evaluation", "f")
    values: dict[int, np.ndarray] = {}
    for row, code in enumerate(status.tolist()):
        if code not in (2, 3):
            continue
        ids = np.asarray(candidate_blocks[row]).reshape(-1)
        f = np.asarray(f_blocks[row]).reshape(len(ids), -1)
        for position, candidate_id in enumerate(ids.tolist()):
            values[candidate_id] = f[position]
    return values


def _sources_by_id(population: Any) -> dict[int, int]:
    return dict(
        zip(
            population.get_array("id").tolist(),
            population.get_array("objective_source").tolist(),
            strict=True,
        )
    )


def _assert_sources_match_true_evaluations(state: Any) -> set[int]:
    truth = _last_true_f_by_id(state.history)
    population = state.population
    for candidate_id, f, source in zip(
        population.get_array("id").tolist(),
        population.get_array("f"),
        population.get_array("objective_source").tolist(),
        strict=True,
    ):
        evaluated = candidate_id in truth and np.array_equal(truth[candidate_id], f)
        assert source == (0 if evaluated else 1), candidate_id
    return set(population.get_array("objective_source").tolist())


def _generation_based_with_evaluation_history(seed: int = 5) -> Optimizer:
    return (
        _surrogate_optimizer_for_swap(GenerationBasedStrategy(4), seed=seed)
        .set_initializer(LHSInitializer(12, 8, seed))
        .set_history(["summary", "evaluation"])
        .set_termination(Termination(lambda state: state.gen >= 30))
    )


def _assert_inherited_predictions_stay_marked(
    before: dict[int, int], state: Any
) -> None:
    predicted = {cid for cid, source in before.items() if source == 1}
    after = _sources_by_id(state.population)
    inherited = [cid for cid in after if cid in predicted]
    evaluated_by_direct = [cid for cid in after if cid not in before]
    assert inherited and evaluated_by_direct
    assert all(after[cid] == 1 for cid in inherited)
    assert all(after[cid] == 0 for cid in evaluated_by_direct)
    assert np.all(state.offspring.get_array("objective_source") == 0)
    _assert_sources_match_true_evaluations(state)


def test_rows_predicted_before_replacing_generation_based_with_direct_between_steps():
    optimizer = _generation_based_with_evaluation_history()
    generator = optimizer.iterate()
    for _ in range(3):
        state = next(generator)
    before = _sources_by_id(state.population)

    optimizer.set_strategy(DirectStrategy())
    state = next(generator)

    assert state.gen == 11
    _assert_inherited_predictions_stay_marked(before, state)


def test_rows_predicted_before_replacing_generation_based_with_direct_from_a_callback():
    optimizer = _generation_based_with_evaluation_history()
    seen: dict[int, Any] = {}

    def on_generation_end(event: Any) -> None:
        ctx = event.ctx
        if ctx.gen == 10:
            seen[10] = _sources_by_id(ctx.population)
            optimizer.set_strategy(DirectStrategy())
        elif ctx.gen == 11:
            _assert_inherited_predictions_stay_marked(seen[10], ctx)
            seen[11] = True

    cbm = CallbackManager()
    cbm.register(GenerationEndEvent, on_generation_end)
    optimizer.cbmanager = cbm
    optimizer.set_termination(Termination(lambda state: state.gen >= 12))
    optimizer.run()

    assert set(seen) == {10, 11}


def _async_surrogate_optimizer(
    strategy: Any, evaluator: _DeterministicEvaluator, seed: int = 5
) -> Optimizer:
    return (
        _surrogate_optimizer_for_swap(strategy, seed=seed)
        .set_initializer(LHSInitializer(12, 8, seed))
        .set_history(["summary", "evaluation"])
        .set_evaluator(evaluator)
        .set_async_evaluation_scheduler(
            AsyncEvaluationScheduler(evaluator, max_pending=3)
        )
        .set_termination(Termination(lambda state: state.gen >= 12))
    )


@pytest.mark.parametrize("use_setter", [False, True])
@pytest.mark.parametrize(
    ("source_factory", "replacement_factory"),
    [
        (SteadyStateStrategy, DirectStrategy),
        (SteadyStateStrategy, lambda: GenerationBasedStrategy(1)),
        (SteadyStateStrategy, lambda: IndividualBasedStrategy(0.5)),
        (SteadyStateStrategy, lambda: PreSelectionStrategy(5, 2)),
        (DirectStrategy, SteadyStateStrategy),
    ],
)
def test_async_strategy_replacement_drains_in_flight_requests(
    source_factory: Any, replacement_factory: Any, use_setter: bool
) -> None:
    algorithm = _RecordingGA()
    evaluator = _DeterministicEvaluator(order="lifo", max_collects=50_000)
    optimizer = (
        _surrogate_optimizer_for_swap(source_factory(), seed=37)
        .set_algorithm(algorithm)
        .set_evaluator(evaluator)
        .set_async_evaluation_scheduler(
            AsyncEvaluationScheduler(evaluator, max_pending=3)
        )
        .set_evaluation_planner(RepeatedEvaluation(replicates=2))
        .set_feedback_builder(MixedFeedback())
        .set_history(["summary", "evaluation"])
        .set_termination(
            Termination(lambda state: state.fe + len(state.pending_evaluations) >= 30)
        )
    )
    optimizer.resolve_defaults()
    optimizer._compile_plan()
    initializer = optimizer.initializer
    assert initializer is not None
    state = initializer.initialize(optimizer, optimizer.problem)
    runtime: Any = create_runtime(optimizer)
    environment = runtime.environment
    session: Any = runtime.initialize(resolve_plan(optimizer), state)

    for _ in range(1000):
        if len(session.state.pending_evaluations) >= 2:
            break
        session = runtime.advance(session).session
    assert len(session.state.pending_evaluations) >= 2
    pending_at_switch = set(session.state.pending_evaluations)
    submissions_at_switch = len(evaluator.submitted)
    asks_at_switch = len(algorithm.asks)
    replacement = replacement_factory()
    if use_setter:
        optimizer.set_strategy(replacement)
    else:
        optimizer.strategy = replacement

    for _ in range(10_000):
        if session.finished:
            break
        session = runtime.advance(session).session

    assert session.finished
    final = session.state
    assert not final.pending_evaluations
    assert not final.evaluation_handles
    assert final.fe >= 30
    assert len(evaluator.submitted) > submissions_at_switch
    assert len(algorithm.asks) > asks_at_switch
    assert replacement.pipeline is not None
    assert environment.binding_values()["strategy"] is replacement
    acked = [int(request_id) for request_id, _ in evaluator.acks]
    submitted = [int(request.request_id) for request in evaluator.submitted]
    assert len(acked) == len(submitted)
    assert sorted(acked) == sorted(submitted)
    assert pending_at_switch <= set(submitted)
    assert final.history is not None
    summary = final.history.channel("summary")
    evaluation = final.history.channel("evaluation")
    assert int(summary["fe"][-1]) == final.fe
    assert set(submitted) <= set(np.asarray(evaluation["request_id"], dtype=int))
    assert all(
        np.count_nonzero(np.asarray(evaluation["request_id"]) == request_id) == 1
        for request_id in submitted
    )


@pytest.mark.parametrize(
    ("strategy", "asynchronous", "predicted"),
    [
        (DirectStrategy, True, False),
        (lambda: GenerationBasedStrategy(2), True, True),
        (lambda: GenerationBasedStrategy(2), False, True),
    ],
    ids=["direct-async", "generation-based-async", "generation-based-sync"],
)
def test_objective_source_matches_true_evaluations_at_every_commit_point(
    strategy: Any, asynchronous: bool, predicted: bool
) -> None:
    evaluator = _DeterministicEvaluator(order="lifo", max_collects=20000)
    if asynchronous:
        optimizer = _async_surrogate_optimizer(strategy(), evaluator)
    else:
        optimizer = _generation_based_with_evaluation_history()
        optimizer.set_strategy(strategy())
        optimizer.set_termination(Termination(lambda state: state.gen >= 12))
    optimizer.resolve_defaults()
    optimizer._compile_plan()
    initializer = optimizer.initializer
    assert initializer is not None
    state = initializer.initialize(optimizer, optimizer.problem)
    runtime: Any = create_runtime(optimizer)
    session: Any = runtime.initialize(resolve_plan(optimizer), state)

    seen: set[int] = set()
    for _ in range(2000):
        seen |= _assert_sources_match_true_evaluations(session.state)
        if session.finished:
            break
        session = runtime.advance(session).session

    assert session.finished
    assert seen == ({0, 1} if predicted else {0})
    assert np.all(session.state.archive.get_array("objective_source") == 0)
    if asynchronous:
        acknowledged = [request_id for request_id, _ in evaluator.acks]
        assert acknowledged != sorted(acknowledged)
