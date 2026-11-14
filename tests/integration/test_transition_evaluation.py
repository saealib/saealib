"""True evaluations a transition requests at a component replacement's commit point.

A transition yields an ``EvaluateRequest``; the commit point evaluates the
requested population rows, confirms the results at once, and sends the outcome
back into the transition.  The algorithm is not told, and the confirmed results
stay when the transition or a later part of the commit point fails.
"""

from __future__ import annotations

import dataclasses
from typing import Any, cast

import numpy as np
import pytest

from saealib import (
    GA,
    CrossoverBLXAlpha,
    EvaluateRequest,
    EvaluationOutcome,
    LHSInitializer,
    MeanPrediction,
    MutationUniform,
    Optimizer,
    PopulationAttribute,
    PostEvaluationRequestEvent,
    SequentialSelection,
    Termination,
    Transition,
    TransitionContext,
    TruncationSelection,
    max_gen,
)
from saealib.exceptions import EvaluationFatalError, EvaluationProtocolError
from saealib.execution import AsyncEvaluationScheduler, SerialEvaluator
from saealib.execution.evaluator import (
    NO_TELL_METADATA,
    AsyncEvaluator,
    EvaluationRequest,
    PendingEvaluation,
)
from saealib.execution.history import History, record_evaluations
from saealib.execution.runner import Runner
from saealib.execution.runtime import create_runtime, resolve_plan
from saealib.problem import Problem
from saealib.strategies import DirectStrategy, GenerationBasedStrategy
from saealib.surrogate.manager import LocalSurrogateManager
from saealib.surrogate.rbf import RBFSurrogate
from saealib.surrogate.rbf_kernels import GaussianKernel


def _sphere(x: Any) -> np.ndarray:
    return np.array([float(np.sum(np.asarray(x) ** 2))], dtype=np.float64)


def _problem() -> Problem:
    return Problem(
        func=_sphere,
        dim=2,
        n_obj=1,
        direction=np.array([-1.0]),
        lb=[-5.0, -5.0],
        ub=[5.0, 5.0],
        constraints=[],
    )


def _ga() -> GA:
    return GA(
        crossover=CrossoverBLXAlpha(prob=0.9, alpha=0.4),
        mutation=MutationUniform(prob_var=0.1),
        parent_selection=SequentialSelection(),
        survivor_selection=TruncationSelection(),
    )


class _CountingEvaluator(SerialEvaluator):
    def __init__(self) -> None:
        super().__init__()
        self.batches: list[np.ndarray] = []

    def evaluate_batch(self, x: Any, problem: Problem) -> Any:
        self.batches.append(np.arange(len(x)))
        return super().evaluate_batch(x, problem)

    def evaluate_request(self, request: Any, problem: Problem) -> Any:
        self.batches.append(np.arange(len(request.candidate_ids)))
        return super().evaluate_request(request, problem)


class _ReattachableEvaluator(_CountingEvaluator):
    def can_reattach(self, pending: Any) -> bool:
        return True

    def reattach(self, pending: Any, problem: Problem) -> Any:
        return self.submit(pending.request, problem)


class _CountingGA(GA):
    def __init__(self) -> None:
        super().__init__(
            crossover=CrossoverBLXAlpha(prob=0.9, alpha=0.4),
            mutation=MutationUniform(prob_var=0.1),
            parent_selection=SequentialSelection(),
            survivor_selection=TruncationSelection(),
        )
        self.told = 0

    def tell(self, feedback: Any, state: Any) -> Any:
        self.told += 1
        return super().tell(feedback, state)


class _Reevaluate(Transition):
    """Requests the population rows that are not truly evaluated yet."""

    def __init__(self, *, columns: bool = False) -> None:
        self.outcomes: list[EvaluationOutcome] = []
        self.columns = columns
        self.staged_before_apply: list[bool] = []

    def produced_columns(self, new_bindings: Any) -> frozenset[str] | None:
        return frozenset({"transition_score"}) if self.columns else None

    def apply(self, tctx: TransitionContext) -> Any:
        assert tctx.state is not None
        population = tctx.state.population
        assert population is not None
        ids = np.asarray(population.get_array("id"), dtype=np.int64)
        source = np.asarray(population.get_array("objective_source"))
        predicted = ids[source != 0]
        outcome = yield EvaluateRequest(predicted)
        self.outcomes.append(outcome)
        if self.columns and tctx.population is not None:
            self.staged_before_apply.append(
                "transition_score" not in tctx.population.schema
            )
        if self.columns:
            for staged in (tctx.population, tctx.archive, tctx.pareto_archive):
                if staged is not None:
                    staged.add_columns(
                        [PopulationAttribute("transition_score", np.float64, (), 0.0)]
                    )


def _optimizer(evaluator: Any, *, strategy: Any, seed: int = 7) -> Optimizer:
    return (
        Optimizer(_problem(), seed=seed)
        .set_initializer(LHSInitializer(6, 6, seed))
        .set_algorithm(_CountingGA())
        .set_strategy(strategy)
        .set_evaluator(evaluator)
        .set_history(["summary", "evaluation"])
        .set_termination(Termination(lambda state: state.gen >= 40))
    )


def _predicted_state(optimizer: Optimizer) -> Any:
    """Initial state whose population rows all count as predictions."""
    optimizer.resolve_defaults()
    optimizer._compile_plan()
    initializer = optimizer.initializer
    assert initializer is not None
    state = initializer.initialize(optimizer, optimizer.problem)
    population = state.population
    population.update_array(
        "objective_source", np.ones(len(population), dtype=np.uint8)
    )
    state.archive.truncate(0)
    state.pareto_archive.truncate(0)
    return state


def _commit_point(optimizer: Optimizer) -> tuple[Any, dict[str, Any]]:
    captured: dict[str, Any] = {}

    def factory(optimizer_obj: Any) -> Any:
        runtime = create_runtime(optimizer_obj)
        captured["environment"] = getattr(runtime, "environment", None)
        return runtime

    runner = Runner(optimizer, runtime_factory=factory)
    state = _predicted_state(optimizer)
    runtime = factory(optimizer)
    session = runtime.initialize(resolve_plan(optimizer), state)
    environment = captured["environment"]
    environment._last_state = session.state
    environment._mark_checkpoint_state(session.state, phase="running")
    return runner, {"environment": environment, "session": session}


def _history(state: Any) -> Any:
    history = state.history
    assert history is not None
    return history.channel("evaluation")


def _plan_signature(plan: Any) -> tuple[object, ...]:
    graph = getattr(plan, "graph", plan)
    return (
        type(plan).__name__,
        tuple(node.component_id for node in graph.nodes),
    )


def _history_rows(state: Any) -> int:
    return len(_history(state)["size"])


def _three_quantities(state: Any) -> tuple[int, int, int]:
    channel = _history(state)
    sizes = np.asarray(channel["size"], dtype=np.int64)
    fe_after = np.asarray(channel["fe_after"], dtype=np.int64)
    return int(state.fe), int(sizes.sum()), int(fe_after[-1])


def _ga_with_surrogate() -> Optimizer:
    return (
        Optimizer(_problem(), seed=5)
        .set_initializer(LHSInitializer(12, 8, 5))
        .set_algorithm(_ga())
        .set_surrogate_manager(LocalSurrogateManager(RBFSurrogate(GaussianKernel())))
        .set_acquisition(MeanPrediction())
        .set_strategy(GenerationBasedStrategy(4))
        .set_history(["summary", "evaluation"])
        .set_termination(Termination(lambda state: state.gen >= 30))
    )


def test_a_transition_evaluates_predicted_rows_on_a_synchronous_flow() -> None:
    evaluator = _CountingEvaluator()
    optimizer = _ga_with_surrogate().set_evaluator(evaluator)
    generator = Runner(optimizer).iterate()
    state = next(generator)
    for _ in range(3):
        state = next(generator)
    predicted = int(
        np.sum(np.asarray(state.population.get_array("objective_source")) != 0)
    )
    fe_before = state.fe
    archive_before = len(state.archive)
    rows_before = _history_rows(state)
    batches_before = len(evaluator.batches)
    transition = _Reevaluate(columns=True)
    optimizer.set_strategy(DirectStrategy(), transition=transition)
    state = next(generator)
    transition_batches = evaluator.batches[batches_before : batches_before + 1]

    assert predicted == 6
    assert [len(batch) for batch in transition_batches] == [6]
    assert sum(len(batch) for batch in transition_batches) == predicted
    channel = _history(state)
    sizes = [int(value) for value in channel["size"]]
    assert 6 in sizes[rows_before:]
    index = sizes.index(6, rows_before)
    assert (
        int(channel["fe_after"][index]) - int(channel["fe_before"][index]) == predicted
    )
    archived_ids = set(np.asarray(state.archive.get_array("id")).tolist())
    evaluated_ids = set(
        int(value) for value in transition.outcomes[0].candidate_ids.tolist()
    )
    assert evaluated_ids <= archived_ids
    archive_sources = dict(
        zip(
            np.asarray(state.archive.get_array("id")).tolist(),
            np.asarray(state.archive.get_array("objective_source")).tolist(),
            strict=True,
        )
    )
    assert all(archive_sources[candidate_id] == 0 for candidate_id in evaluated_ids)
    assert len(state.archive) >= archive_before + predicted
    assert np.all(np.asarray(state.population.get_array("objective_source")) == 0)
    assert len(transition.outcomes) == 1
    assert len(transition.outcomes[0].candidate_ids) == predicted
    assert transition.staged_before_apply == [True]
    for state in generator:
        pass
    assert state.fe > fe_before
    assert sum(int(value) for value in _history(state)["size"]) == state.fe
    assert int(_history(state)["fe_after"][-1]) == state.fe
    assert "transition_score" in state.population.schema
    assert "transition_score" in state.archive.schema


def test_a_transition_evaluates_predicted_rows_on_an_asynchronous_flow() -> None:
    evaluator = _CountingEvaluator()
    async_evaluator = AsyncEvaluator(evaluator, max_workers=2)
    optimizer = _ga_with_surrogate().set_evaluator(async_evaluator)
    optimizer.set_async_evaluation_scheduler(
        AsyncEvaluationScheduler(async_evaluator, max_pending=3)
    )
    generator = Runner(optimizer).iterate()
    state = next(generator)
    for _ in range(3):
        state = next(generator)
    predicted = int(
        np.sum(np.asarray(state.population.get_array("objective_source")) != 0)
    )
    fe_before = state.fe
    rows_before = _history_rows(state)
    batches_before = len(evaluator.batches)
    transition = _Reevaluate(columns=True)
    optimizer.set_strategy(DirectStrategy(), transition=transition)
    state = next(generator)
    transition_batches = evaluator.batches[batches_before : batches_before + 2]

    assert predicted == 6
    assert [len(batch) for batch in transition_batches] == [3, 3]
    channel = _history(state)
    sizes = [int(value) for value in channel["size"]]
    deltas = [
        int(channel["fe_after"][index]) - int(channel["fe_before"][index])
        for index in range(rows_before, len(sizes))
    ]
    assert deltas.count(3) >= 2
    assert sum(deltas[:2]) == predicted or 3 in deltas
    assert len(transition.outcomes[0].candidate_ids) == predicted
    assert state.fe >= fe_before

    for state in generator:
        pass
    assert "transition_score" in state.population.schema


def _run_commit_point(optimizer: Any) -> tuple[Any, Any]:
    _runner, captured = _commit_point(optimizer)
    environment = captured["environment"]
    session = captured["session"]
    refreshed = environment._refresh_plan_if_needed(session.state)
    state = session.state if refreshed is None else refreshed
    return _runner, state


def test_six_predicted_rows_at_capacity_three_are_evaluated_in_two_requests() -> None:
    evaluator = _CountingEvaluator()
    async_evaluator = AsyncEvaluator(evaluator, max_workers=2)
    optimizer = _optimizer(async_evaluator, strategy=DirectStrategy())
    scheduler = AsyncEvaluationScheduler(async_evaluator, max_pending=3)
    optimizer.set_async_evaluation_scheduler(scheduler)
    transition = _Reevaluate()
    _runner, captured = _commit_point(optimizer)
    environment = captured["environment"]
    state = captured["session"].state
    fe_before = state.fe
    rows_before = _history_rows(state)
    calls_before = len(evaluator.batches)
    plan_before = _plan_signature(environment.plan)
    contract_before = scheduler.contract()

    optimizer.set_strategy(DirectStrategy(), transition=transition)
    refreshed = environment._refresh_plan_if_needed(state)
    assert refreshed is not None
    state = refreshed

    assert len(evaluator.batches) - calls_before == 2
    assert [len(batch) for batch in evaluator.batches[calls_before:]] == [3, 3]
    sizes = [int(value) for value in _history(state)["size"][rows_before:]]
    assert sizes == [3, 3]
    assert state.fe == fe_before + 6
    assert _three_quantities(state) == (fe_before + 6, fe_before + 6, fe_before + 6)
    assert _plan_signature(environment.plan) == plan_before
    assert scheduler.contract() == contract_before


def test_a_synchronous_flow_evaluates_six_predicted_rows_in_one_request() -> None:
    evaluator = _CountingEvaluator()
    optimizer = _optimizer(evaluator, strategy=DirectStrategy())
    transition = _Reevaluate()
    _runner, captured = _commit_point(optimizer)
    environment = captured["environment"]
    state = captured["session"].state
    fe_before = state.fe
    rows_before = _history_rows(state)
    calls_before = len(evaluator.batches)

    optimizer.set_strategy(DirectStrategy(), transition=transition)
    refreshed = environment._refresh_plan_if_needed(state)
    assert refreshed is not None
    state = refreshed

    assert len(evaluator.batches) - calls_before == 1
    assert len(evaluator.batches[-1]) == 6
    sizes = [int(value) for value in _history(state)["size"][rows_before:]]
    assert sizes == [6]
    assert state.fe == fe_before + 6
    assert _three_quantities(state) == (fe_before + 6, fe_before + 6, fe_before + 6)


def test_a_transition_evaluation_is_not_told_to_the_algorithm() -> None:
    evaluator = _CountingEvaluator()
    async_evaluator = AsyncEvaluator(evaluator, max_workers=2)
    scheduler = AsyncEvaluationScheduler(async_evaluator, max_pending=3)
    optimizer = _optimizer(async_evaluator, strategy=DirectStrategy())
    optimizer.set_async_evaluation_scheduler(scheduler)
    transition = _Reevaluate()
    algorithm = cast(Any, optimizer.algorithm)
    deliveries: list[Any] = []
    original_delivery = scheduler._apply_feedback_delivery

    def record_delivery(state: Any, update: Any) -> Any:
        deliveries.append(update)
        return original_delivery(state, update)

    cast(Any, scheduler)._apply_feedback_delivery = record_delivery
    _runner, captured = _commit_point(optimizer)
    environment = captured["environment"]
    state = captured["session"].state
    told_before = algorithm.told
    deliveries_before = len(deliveries)

    optimizer.set_strategy(DirectStrategy(), transition=transition)
    refreshed = environment._refresh_plan_if_needed(state)
    assert refreshed is not None

    assert len(transition.outcomes) == 1
    assert algorithm.told == told_before
    assert len(deliveries) == deliveries_before


def test_the_driver_keeps_offspring_and_the_scheduler_algorithm_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evaluator = _CountingEvaluator()
    async_evaluator = AsyncEvaluator(evaluator, max_workers=2)
    scheduler = AsyncEvaluationScheduler(async_evaluator, max_pending=3)
    optimizer = _optimizer(async_evaluator, strategy=DirectStrategy())
    optimizer.set_async_evaluation_scheduler(scheduler)
    transition = _Reevaluate()
    _runner, captured = _commit_point(optimizer)
    environment = captured["environment"]
    state = captured["session"].state
    offspring = state.offspring
    algorithm = scheduler.algorithm
    contract_before = scheduler.contract()
    algorithm_writes: list[tuple[object, object]] = []
    offspring_writes: list[object] = []
    inside = {"active": False}
    original_scheduler_setattr = type(scheduler).__setattr__
    original_submit = scheduler.submit
    original_poll = scheduler.poll

    def scheduler_setattr(self: Any, name: str, value: Any) -> None:
        if name == "algorithm" and inside["active"]:
            algorithm_writes.append((self.algorithm, value))
        original_scheduler_setattr(self, name, value)

    def submit(state: Any, requests: Any, *, owner: Any = None) -> Any:
        inside["active"] = True
        try:
            return original_submit(state, requests, owner=owner)
        finally:
            inside["active"] = False

    def poll(state: Any, *, wait: bool = False) -> Any:
        inside["active"] = True
        try:
            return original_poll(state, wait=wait)
        finally:
            inside["active"] = False

    monkeypatch.setattr(type(scheduler), "__setattr__", scheduler_setattr)
    monkeypatch.setattr(scheduler, "submit", submit)
    monkeypatch.setattr(scheduler, "poll", poll)
    from saealib.context import OptimizationState

    original_state_setattr = OptimizationState.__setattr__

    def state_setattr(self: Any, name: str, value: Any) -> None:
        if name == "offspring" and inside["active"]:
            offspring_writes.append(value)
        original_state_setattr(self, name, value)

    monkeypatch.setattr(OptimizationState, "__setattr__", state_setattr)
    optimizer.set_strategy(DirectStrategy(), transition=transition)
    refreshed = environment._refresh_plan_if_needed(state)

    assert refreshed is not None
    assert state.offspring is offspring
    assert scheduler.algorithm is algorithm
    assert offspring_writes == []
    assert algorithm_writes == []
    assert len(transition.outcomes[0].candidate_ids) == 6
    assert scheduler.contract() == contract_before


def test_a_failing_transition_keeps_its_evaluations_and_repeats_none() -> None:
    class _Failing(Transition):
        def __init__(self) -> None:
            self.applied = False

        def apply(self, tctx: TransitionContext) -> Any:
            assert tctx.state is not None
            population = tctx.state.population
            ids = np.asarray(population.get_array("id"), dtype=np.int64)
            source = np.asarray(population.get_array("objective_source"))
            yield EvaluateRequest(ids[source != 0])
            self.applied = True
            raise RuntimeError("transition failed after evaluating")

    evaluator = _CountingEvaluator()
    optimizer = _optimizer(evaluator, strategy=DirectStrategy())
    transition = _Failing()
    _runner, captured = _commit_point(optimizer)
    environment = captured["environment"]
    state = captured["session"].state
    fe_before = state.fe
    archive_before = len(state.archive)
    rows_before = _history_rows(state)
    version_before = environment.active_version
    calls_before = len(evaluator.batches)

    optimizer.set_strategy(DirectStrategy(), transition=transition)
    with pytest.raises(RuntimeError):
        environment._refresh_plan_if_needed(state)

    evaluated = environment._last_state
    assert evaluated is not None
    assert len(evaluator.batches) - calls_before == 1
    assert evaluated.fe == fe_before + 6
    assert len(evaluated.archive) == archive_before + 6
    assert _history_rows(evaluated) == rows_before + 1
    assert environment.active_version == version_before
    assert "transition_score" not in evaluated.population.schema
    assert np.all(np.asarray(evaluated.population.get_array("objective_source")) == 0)

    rerun = _Reevaluate(columns=True)
    optimizer.set_strategy(DirectStrategy(), transition=rerun)
    refreshed = environment._refresh_plan_if_needed(evaluated)
    assert refreshed is not None
    assert len(evaluator.batches) - calls_before == 1
    assert refreshed.fe == fe_before + 6
    assert _history_rows(refreshed) == rows_before + 1
    assert len(rerun.outcomes) == 1
    assert len(rerun.outcomes[0].candidate_ids) == 0
    assert "transition_score" in refreshed.population.schema


def test_a_pending_transition_request_survives_a_pickle_checkpoint_without_telling(
    tmp_path: Any,
) -> None:
    evaluator = _ReattachableEvaluator()
    scheduler = AsyncEvaluationScheduler(evaluator, max_pending=6)
    optimizer = _optimizer(evaluator, strategy=DirectStrategy())
    optimizer.set_termination(Termination(max_gen(40)))
    optimizer.set_async_evaluation_scheduler(scheduler)
    _runner, captured = _commit_point(optimizer)
    state = captured["session"].state
    population = state.population
    ids = np.asarray(population.get_array("id"), dtype=np.int64)
    request = EvaluationRequest(
        np.int64(state.request_id_allocator.allocate(1)[0]),
        ids,
        population.genomes,
        metadata={NO_TELL_METADATA: True},
    )
    state = scheduler.submit(state, [request], owner=population)
    algorithm = cast(Any, optimizer.algorithm)
    told_before = algorithm.told
    fe_before = state.fe
    rows_before = _history_rows(state)
    archive_before = len(state.archive)
    pending_fields = {field.name for field in dataclasses.fields(PendingEvaluation)}

    path = tmp_path / "transition.pkl"
    captured["environment"]._mark_checkpoint_state(state, phase="boundary")
    optimizer.save_pickle(state, path)
    restored_optimizer, restored = Optimizer.load_pickle(path)
    restored_pending = next(iter(restored.pending_evaluations.values()))
    assert restored_pending.request.metadata[NO_TELL_METADATA] is True
    assert {
        field.name for field in dataclasses.fields(restored_pending)
    } == pending_fields
    restored_scheduler = restored_optimizer.async_evaluation_scheduler
    assert restored_scheduler is not None
    restored = restored_scheduler.reattach(restored)
    restored = restored_scheduler.poll(restored, wait=True)
    record_evaluations(restored)

    assert cast(Any, restored_optimizer.algorithm).told == told_before
    assert restored.fe == fe_before + len(ids)
    assert _history_rows(restored) == rows_before + 1
    assert len(restored.archive) == archive_before + len(ids)
    assert not restored.pending_evaluations


def test_a_raising_request_event_handler_keeps_the_confirmed_results() -> None:
    evaluator = _CountingEvaluator()
    optimizer = _optimizer(evaluator, strategy=DirectStrategy())
    transition = _Reevaluate()
    _runner, captured = _commit_point(optimizer)
    environment = captured["environment"]
    state = captured["session"].state
    fe_before = int(state.fe)
    rows_before = _history_rows(state)
    archive_before = len(state.archive)

    def raise_from_handler(event: Any) -> None:
        raise RuntimeError("observer failed")

    optimizer.cbmanager.register(PostEvaluationRequestEvent, raise_from_handler)
    optimizer.set_strategy(DirectStrategy(), transition=transition)
    with pytest.raises(EvaluationFatalError):
        environment._refresh_plan_if_needed(state)

    confirmed = environment._last_state
    assert confirmed is not None
    assert int(confirmed.fe) == fe_before + 6
    assert _history_rows(confirmed) == rows_before + 1
    assert len(confirmed.archive) == archive_before + 6
    assert _three_quantities(confirmed) == (fe_before + 6,) * 3
    assert environment.active_version == 0


def test_a_history_append_failure_changes_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evaluator = _CountingEvaluator()
    optimizer = _optimizer(evaluator, strategy=DirectStrategy())
    transition = _Reevaluate(columns=True)
    _runner, captured = _commit_point(optimizer)
    environment = captured["environment"]
    state = captured["session"].state
    fe_before = int(state.fe)
    rows_before = _history_rows(state)
    archive_before = len(state.archive)

    def fail_append(self: Any, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("append failed")

    monkeypatch.setattr(History, "append_block", fail_append)
    optimizer.set_strategy(DirectStrategy(), transition=transition)
    with pytest.raises(RuntimeError):
        environment._refresh_plan_if_needed(state)

    unchanged = environment._last_state
    assert unchanged is not None
    assert int(unchanged.fe) == fe_before
    assert _history_rows(unchanged) == rows_before
    assert len(unchanged.archive) == archive_before
    assert "transition_score" not in unchanged.population.schema
    assert environment.active_version == 0


@pytest.mark.parametrize(
    ("failure", "confirmed_rows"), [("handler", 3), ("history", 0)]
)
def test_the_confirmation_order_holds_on_the_asynchronous_path(
    monkeypatch: pytest.MonkeyPatch, failure: str, confirmed_rows: int
) -> None:
    evaluator = _CountingEvaluator()
    async_evaluator = AsyncEvaluator(evaluator, max_workers=2)
    scheduler = AsyncEvaluationScheduler(async_evaluator, max_pending=3)
    optimizer = _optimizer(async_evaluator, strategy=DirectStrategy())
    optimizer.set_async_evaluation_scheduler(scheduler)
    transition = _Reevaluate()
    _runner, captured = _commit_point(optimizer)
    environment = captured["environment"]
    state = captured["session"].state
    fe_before = int(state.fe)
    rows_before = _history_rows(state)
    archive_before = len(state.archive)
    if failure == "handler":

        def raise_from_handler(event: Any) -> None:
            raise RuntimeError("observer failed")

        optimizer.cbmanager.register(PostEvaluationRequestEvent, raise_from_handler)
    else:

        def fail_append(self: Any, *args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("append failed")

        monkeypatch.setattr(History, "append_block", fail_append)
    optimizer.set_strategy(DirectStrategy(), transition=transition)
    with pytest.raises((RuntimeError, EvaluationFatalError)):
        environment._refresh_plan_if_needed(state)

    confirmed = environment._last_state
    assert confirmed is not None
    assert int(confirmed.fe) == fe_before + confirmed_rows
    assert len(confirmed.archive) == archive_before + confirmed_rows
    assert _history_rows(confirmed) == rows_before + (1 if confirmed_rows else 0)
    if confirmed_rows:
        assert _three_quantities(confirmed) == (fe_before + confirmed_rows,) * 3
    else:
        assert _three_quantities(confirmed) == (fe_before, fe_before, fe_before)


class _FailableEvaluator(_CountingEvaluator):
    fail = False

    def evaluate_batch(self, x: Any, problem: Problem) -> Any:
        if self.fail:
            raise RuntimeError("objective failed")
        return super().evaluate_batch(x, problem)

    def evaluate_request(self, request: Any, problem: Problem) -> Any:
        if self.fail:
            raise RuntimeError("objective failed")
        return super().evaluate_request(request, problem)


def _observable_vector(state: Any, scheduler: Any) -> dict[str, Any]:
    history = state.history
    return {
        "handles": sorted(int(value) for value in state.evaluation_handles),
        "pending": sorted(int(value) for value in state.pending_evaluations),
        "owners": sorted(int(value) for value in state.evaluation_owners),
        "fatal_states": sorted(
            getattr(scheduler, "_fatal_states", {}) if scheduler is not None else ()
        ),
        "fe_counter": int(history._fe_counter),
        "fe_counted_ids": {
            int(request_id): sorted(int(value) for value in counted)
            for request_id, counted in history._fe_counted_ids.items()
        },
        "evaluation_keys": sorted(history._evaluation_keys),
        "surrogate_accuracy_keys": sorted(history._surrogate_accuracy_keys),
        "terminal_request_ids": sorted(history._evaluation_terminal_request_ids),
    }


def _runtime_vector(vector: dict[str, Any]) -> dict[str, Any]:
    return {
        name: vector[name] for name in ("handles", "pending", "owners", "fatal_states")
    }


def _install_failure(
    monkeypatch: pytest.MonkeyPatch, evaluator: Any, optimizer: Any, failure: str
) -> tuple[Any, Any]:
    if failure == "evaluator":
        evaluator.fail = True
        return evaluator, lambda: setattr(evaluator, "fail", False)
    if failure == "handler":

        def raise_from_handler(event: Any) -> None:
            raise RuntimeError("observer failed")

        optimizer.cbmanager.register(PostEvaluationRequestEvent, raise_from_handler)
        return evaluator, lambda: optimizer.cbmanager.unregister(
            PostEvaluationRequestEvent, raise_from_handler
        )

    def fail_append(self: Any, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("append failed")

    monkeypatch.setattr(History, "append_block", fail_append)
    original = (
        History.append_block.__wrapped__
        if hasattr(History.append_block, "__wrapped__")
        else None
    )
    del original

    def restore() -> None:
        monkeypatch.undo()

    return evaluator, restore


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("failure", ["evaluator", "handler", "history"])
def test_a_failed_transition_evaluation_restores_its_observable_state(
    monkeypatch: pytest.MonkeyPatch, asynchronous: bool, failure: str
) -> None:
    evaluator = _FailableEvaluator()
    scheduler = None
    if asynchronous:
        async_evaluator = AsyncEvaluator(evaluator, max_workers=2)
        scheduler = AsyncEvaluationScheduler(async_evaluator, max_pending=3)
        optimizer = _optimizer(async_evaluator, strategy=DirectStrategy())
        optimizer.set_async_evaluation_scheduler(scheduler)
    else:
        optimizer = _optimizer(evaluator, strategy=DirectStrategy())
    _runner, captured = _commit_point(optimizer)
    environment = captured["environment"]
    state = captured["session"].state
    before = _observable_vector(state, scheduler)
    _target, recover = _install_failure(monkeypatch, evaluator, optimizer, failure)

    transition = _Reevaluate()
    optimizer.set_strategy(DirectStrategy(), transition=transition)
    with pytest.raises((RuntimeError, EvaluationFatalError, EvaluationProtocolError)):
        environment._refresh_plan_if_needed(state)

    failed = environment._last_state
    assert failed is not None
    after = _observable_vector(failed, scheduler)
    assert _runtime_vector(after) == _runtime_vector(before)
    if failure == "history":
        assert after == before
    assert _three_quantities(failed) == (int(failed.fe),) * 3
    assert environment.active_version == 0

    recover()
    rerun = _Reevaluate()
    optimizer.set_strategy(DirectStrategy(), transition=rerun)
    resumed = environment._refresh_plan_if_needed(failed)

    assert resumed is not None
    assert environment.active_version == 1
    assert len(rerun.outcomes) == 1
    assert len(rerun.outcomes[0].candidate_ids) == 12 - int(failed.fe)
    assert _three_quantities(resumed) == (12, 12, 12)
    assert not resumed.pending_evaluations
    assert not resumed.evaluation_handles
    assert _runtime_vector(_observable_vector(resumed, scheduler)) == {
        "handles": [],
        "pending": [],
        "owners": [],
        "fatal_states": [],
    }


class _FailAfterEvaluator(_FailableEvaluator):
    def __init__(self, succeed: int) -> None:
        super().__init__()
        self.succeed = succeed

    def evaluate_request(self, request: Any, problem: Problem) -> Any:
        if self.succeed <= 0:
            raise RuntimeError("objective failed")
        self.succeed -= 1
        return super().evaluate_request(request, problem)


def test_a_terminal_failure_of_an_accumulator_flow_fails_the_transition() -> None:
    evaluator = _FailableEvaluator()
    async_evaluator = AsyncEvaluator(evaluator, max_workers=2)
    scheduler = AsyncEvaluationScheduler(async_evaluator, max_pending=3)
    optimizer = _optimizer(async_evaluator, strategy=DirectStrategy())
    optimizer.set_async_evaluation_scheduler(scheduler)
    _runner, captured = _commit_point(optimizer)
    environment = captured["environment"]
    state = captured["session"].state
    evaluator.fail = True
    transition = _Reevaluate()
    optimizer.set_strategy(DirectStrategy(), transition=transition)

    with pytest.raises(EvaluationFatalError):
        environment._refresh_plan_if_needed(state)

    failed = environment._last_state
    assert failed is not None
    assert scheduler._feedback_accumulator is not None
    assert environment.active_version == 0
    assert transition.outcomes == []
    assert int(failed.fe) == 6
    assert _three_quantities(failed) == (6, 6, 6)


def test_a_partially_failed_asynchronous_request_fails_and_keeps_confirmed_rows() -> (
    None
):
    evaluator = _FailAfterEvaluator(succeed=1)
    async_evaluator = AsyncEvaluator(evaluator, max_workers=2)
    scheduler = AsyncEvaluationScheduler(async_evaluator, max_pending=3)
    optimizer = _optimizer(async_evaluator, strategy=DirectStrategy())
    optimizer.set_async_evaluation_scheduler(scheduler)
    _runner, captured = _commit_point(optimizer)
    environment = captured["environment"]
    state = captured["session"].state
    transition = _Reevaluate()
    optimizer.set_strategy(DirectStrategy(), transition=transition)

    with pytest.raises(EvaluationFatalError):
        environment._refresh_plan_if_needed(state)

    failed = environment._last_state
    assert failed is not None
    assert environment.active_version == 0
    assert transition.outcomes == []
    assert int(failed.fe) == 9
    assert _three_quantities(failed) == (9, 9, 9)

    evaluator.succeed = 10
    rerun = _Reevaluate()
    optimizer.set_strategy(DirectStrategy(), transition=rerun)
    resumed = environment._refresh_plan_if_needed(failed)

    assert resumed is not None
    assert environment.active_version == 1
    assert len(rerun.outcomes) == 1
    assert len(rerun.outcomes[0].candidate_ids) == 3
    assert _three_quantities(resumed) == (12, 12, 12)


@pytest.mark.parametrize("max_pending", [1, 2, 3, 6])
def test_a_transition_request_larger_than_the_capacity_still_runs(
    max_pending: int,
) -> None:
    evaluator = _CountingEvaluator()
    async_evaluator = AsyncEvaluator(evaluator, max_workers=2)
    scheduler = AsyncEvaluationScheduler(async_evaluator, max_pending=max_pending)
    optimizer = _optimizer(async_evaluator, strategy=DirectStrategy())
    optimizer.set_async_evaluation_scheduler(scheduler)
    transition = _Reevaluate()
    _runner, captured = _commit_point(optimizer)
    environment = captured["environment"]
    state = captured["session"].state
    batches_before = len(evaluator.batches)

    optimizer.set_strategy(DirectStrategy(), transition=transition)
    refreshed = environment._refresh_plan_if_needed(state)

    assert refreshed is not None
    batches = [len(batch) for batch in evaluator.batches[batches_before:]]
    assert sorted(batches) == sorted(
        [min(max_pending, 6)] * (6 // max_pending)
        + [6 % max_pending] * bool(6 % max_pending)
    )
    assert len(transition.outcomes) == 1
    assert len(transition.outcomes[0].candidate_ids) == 6
    assert _three_quantities(refreshed) == (12, 12, 12)
    assert not refreshed.pending_evaluations
    assert not refreshed.evaluation_handles
