"""The request-level evaluation event and the origin column in History.

``PostEvaluationRequestEvent`` fires once for every evaluation request whose
results are counted, on every evaluation path.  The History ``evaluation``
channel records who requested each evaluation in ``origin_kind``.
"""

from __future__ import annotations

import importlib
import json
import pickle
import time
from typing import Any

import numpy as np
import pytest

from saealib import (
    GA,
    CrossoverBLXAlpha,
    LHSInitializer,
    MeanPrediction,
    MutationUniform,
    Optimizer,
    PostEvaluationEvent,
    PostEvaluationRequestEvent,
    SequentialSelection,
    Termination,
    TruncationSelection,
    max_gen,
)
from saealib.context import (
    CURRENT_CHECKPOINT_SCHEMA_VERSION,
    OptimizationState,
    _begin_evaluation_report,
    _count_evaluated,
    _EvaluationReport,
)
from saealib.exceptions import EvaluationFatalError
from saealib.execution import AsyncEvaluationScheduler, SerialEvaluator
from saealib.execution.evaluator import (
    AsyncEvaluator,
    EvaluationResult,
    EvaluationStatus,
    EvaluationUpdate,
)
from saealib.execution.history import (
    SUPPORTED_HISTORY_CHANNELS,
    History,
    record_evaluations,
)
from saealib.execution.runner import Runner
from saealib.execution.runtime import create_runtime
from saealib.policies.evaluation import RepeatedEvaluation
from saealib.problem import Problem
from saealib.stages import TrueEvaluationStage
from saealib.strategies import DirectStrategy, GenerationBasedStrategy
from saealib.surrogate.manager import LocalSurrogateManager
from saealib.surrogate.rbf import RBFSurrogate
from saealib.surrogate.rbf_kernels import GaussianKernel
from tests.integration.test_transition_evaluation import (
    _commit_point,
    _CountingEvaluator,
    _history,
    _Reevaluate,
)
from tests.integration.test_transition_evaluation import (
    _optimizer as _transition_optimizer,
)


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


def _build(label: str) -> Optimizer:
    optimizer = (
        Optimizer(_problem(), seed=7)
        .set_initializer(LHSInitializer(6, 4, 7))
        .set_algorithm(_ga())
        .set_strategy(DirectStrategy())
        .set_termination(Termination(max_gen(6)))
        .set_history(["summary", "evaluation"])
    )
    if label == "sync-direct":
        return optimizer.set_evaluator(SerialEvaluator())
    if label == "sync-multi":
        return optimizer.set_evaluator(SerialEvaluator()).set_evaluation_planner(
            RepeatedEvaluation(replicates=2)
        )
    evaluator = AsyncEvaluator(SerialEvaluator(), max_workers=2)
    return (
        optimizer.set_surrogate_manager(
            LocalSurrogateManager(RBFSurrogate(GaussianKernel()))
        )
        .set_acquisition(MeanPrediction())
        .set_strategy(GenerationBasedStrategy(2))
        .set_evaluator(evaluator)
        .set_async_evaluation_scheduler(
            AsyncEvaluationScheduler(evaluator, max_pending=2)
        )
    )


def _record(optimizer: Optimizer) -> tuple[list[Any], list[Any]]:
    plan_events: list[Any] = []
    request_events: list[Any] = []
    optimizer.cbmanager.register(PostEvaluationEvent, plan_events.append)
    optimizer.cbmanager.register(PostEvaluationRequestEvent, request_events.append)
    return plan_events, request_events


def _run(
    label: str,
) -> tuple[OptimizationState, OptimizationState, list[Any], list[Any]]:
    optimizer = _build(label)
    captured: dict[str, Any] = {}

    def factory(optimizer_obj: Any) -> Any:
        runtime = create_runtime(optimizer_obj)
        captured["environment"] = getattr(runtime, "environment", None)
        return runtime

    plan_events, request_events = _record(optimizer)
    returned = Runner(optimizer, runtime_factory=factory).run()
    final = captured["environment"]._last_state
    return returned, final, plan_events, request_events


def _history_rows(state: OptimizationState) -> int:
    return len(_history(state)["size"])


def _three_quantities(state: OptimizationState) -> tuple[int, int, int]:
    channel = _history(state)
    return (
        int(state.fe),
        int(np.sum(channel["size"])),
        int(channel["fe_after"][-1]),
    )


def test_the_initializer_reports_one_request_without_a_planner_id() -> None:
    optimizer = _build("sync-direct")
    plan_events, request_events = _record(optimizer)
    initializer = optimizer.initializer
    assert initializer is not None
    state = initializer.initialize(optimizer, optimizer.problem)

    assert state.fe == 6
    assert len(request_events) == 1
    event = request_events[0]
    assert int(event.request_id) == -1
    assert event.size == 6
    assert event.origin_kind == 0
    assert int(_history(state)["origin_kind"][0]) == 0
    assert plan_events == []


def test_the_classic_stage_reports_one_request_per_execution() -> None:
    optimizer = _build("sync-direct")
    optimizer.resolve_defaults()
    optimizer._compile_plan()
    initializer = optimizer.initializer
    assert initializer is not None
    state = initializer.initialize(optimizer, optimizer.problem)
    population = state.population
    assert population is not None
    state = state.replace(offspring=population)
    plan_events, request_events = _record(optimizer)

    counted = TrueEvaluationStage(
        SerialEvaluator(), cbmanager=optimizer.cbmanager
    ).execute(state)
    record_evaluations(counted)

    assert len(plan_events) == 1
    assert plan_events[0].request_id is None
    assert len(request_events) == 1
    event = request_events[0]
    assert int(event.request_id) >= 0
    assert event.size == len(population)
    assert event.origin_kind == 0
    assert len(_history(counted)["size"]) == 2
    assert int(_history(counted)["origin_kind"][-1]) == 0
    assert _three_quantities(counted) == (10, 10, 10)


def test_a_synchronous_plan_reports_one_request_per_generation() -> None:
    returned, final, plan_events, request_events = _run("sync-direct")

    assert len(request_events) == 7
    assert len(plan_events) == 6
    assert [event.size for event in request_events] == [6] + [4] * 6
    assert {event.origin_kind for event in request_events} == {0}
    assert _three_quantities(returned) == (30, 30, 30)
    assert _three_quantities(final) == (30, 30, 30)


def test_a_multi_request_plan_reports_every_request() -> None:
    _returned, final, plan_events, request_events = _run("sync-multi")

    assert len(request_events) == 13
    assert len(plan_events) == 6
    assert [event.size for event in request_events] == [6] + [4] * 12
    assert _three_quantities(_returned) == (54, 54, 54)
    assert _three_quantities(final) == (54, 54, 54)


def test_the_async_scheduler_reports_one_event_per_request() -> None:
    _returned, final, plan_events, request_events = _run("async-gb")

    assert len(plan_events) == 2
    assert len(request_events) == 5
    assert [event.size for event in request_events] == [6, 2, 2, 2, 2]
    assert _three_quantities(final) == (14, 14, 14)


def test_a_transition_evaluation_reports_a_request_and_no_plan_event() -> None:
    evaluator = _CountingEvaluator()
    optimizer = _transition_optimizer(evaluator, strategy=DirectStrategy())
    transition = _Reevaluate()
    _runner, captured = _commit_point(optimizer)
    environment = captured["environment"]
    state = captured["session"].state
    plan_events, request_events = _record(optimizer)
    optimizer.set_strategy(DirectStrategy(), transition=transition)
    refreshed = environment._refresh_plan_if_needed(state)

    assert refreshed is not None
    assert plan_events == []
    assert len(request_events) == 1
    event = request_events[0]
    assert event.size == 6
    assert event.origin_kind == 1
    assert int(_history(refreshed)["origin_kind"][-1]) == 1


def test_an_asynchronous_transition_evaluation_reports_each_request() -> None:
    evaluator = _CountingEvaluator()
    async_evaluator = AsyncEvaluator(evaluator, max_workers=2)
    scheduler = AsyncEvaluationScheduler(async_evaluator, max_pending=3)
    optimizer = _transition_optimizer(async_evaluator, strategy=DirectStrategy())
    optimizer.set_async_evaluation_scheduler(scheduler)
    transition = _Reevaluate()
    _runner, captured = _commit_point(optimizer)
    environment = captured["environment"]
    state = captured["session"].state
    plan_events, request_events = _record(optimizer)
    optimizer.set_strategy(DirectStrategy(), transition=transition)
    refreshed = environment._refresh_plan_if_needed(state)

    assert refreshed is not None
    assert plan_events == []
    assert [event.size for event in request_events] == [3, 3]
    assert {event.origin_kind for event in request_events} == {1}
    assert [int(value) for value in _history(refreshed)["origin_kind"][-2:]] == [1, 1]


def test_every_evaluation_row_records_an_integer_origin_kind() -> None:
    _, final, _, _ = _run("sync-direct")

    channel = _history(final)
    assert "origin_kind" in channel
    assert channel["origin_kind"].dtype == np.dtype(np.int64)
    assert set(np.asarray(channel["origin_kind"]).tolist()) == {0}


def test_a_checkpoint_round_trip_keeps_the_origin_kind(tmp_path: Any) -> None:
    returned, _final, _, _ = _run("sync-direct")
    assert CURRENT_CHECKPOINT_SCHEMA_VERSION == 1
    path = tmp_path / "history.npz"
    returned.save(path)
    restored = OptimizationState.load(path, returned.problem)
    np.testing.assert_array_equal(
        _history(restored)["origin_kind"], _history(returned)["origin_kind"]
    )
    with np.load(path) as data:
        metadata = json.loads(bytes(data["_history_meta"]).decode())
        version = int(np.asarray(data["_checkpoint_schema_version"]))
    assert version == CURRENT_CHECKPOINT_SCHEMA_VERSION
    assert "origin_kind" in metadata["columns"]["evaluation"]


def _remove_origin_kind_column(path: Any, target: Any) -> None:
    with np.load(path) as data:
        arrays = {name: data[name] for name in data.files}
    metadata = json.loads(bytes(arrays["_history_meta"]).decode())
    metadata["columns"]["evaluation"].remove("origin_kind")
    arrays["_history_meta"] = np.frombuffer(
        json.dumps(metadata).encode(), dtype=np.uint8
    )
    arrays.pop("_history__evaluation__origin_kind")
    np.savez(target, **arrays)


def test_an_old_checkpoint_resumes_and_appends_rows_with_origin_zero(
    tmp_path: Any,
) -> None:
    problem = _problem()
    optimizer = (
        Optimizer(problem, seed=7)
        .set_initializer(LHSInitializer(6, 4, 7))
        .set_algorithm(_ga())
        .set_strategy(DirectStrategy())
        .set_termination(Termination(max_gen(4)))
        .set_history(["summary", "evaluation"])
        .set_evaluator(SerialEvaluator())
    )
    state = optimizer.run()
    assert state.gen == 4
    path = tmp_path / "current.npz"
    state.save(path)
    old_path = tmp_path / "old.npz"
    _remove_origin_kind_column(path, old_path)

    restored = OptimizationState.load(old_path, problem)

    assert restored.gen == 4
    channel = _history(restored)
    assert set(np.asarray(channel["origin_kind"]).tolist()) == {0}
    rows_before = len(channel["size"])
    resumed = (
        Optimizer(problem, seed=7)
        .set_initializer(LHSInitializer(6, 4, 7))
        .set_algorithm(_ga())
        .set_strategy(DirectStrategy())
        .set_termination(Termination(max_gen(6)))
        .set_history(["summary", "evaluation"])
        .set_evaluator(SerialEvaluator())
        .run_from(restored)
    )
    assert resumed.gen == 6
    channel = _history(resumed)
    assert len(channel["size"]) > rows_before
    assert set(np.asarray(channel["origin_kind"]).tolist()) == {0}
    assert _three_quantities(resumed) == (
        int(resumed.fe),
        int(resumed.fe),
        int(resumed.fe),
    )


def test_the_event_is_exported_from_the_callback_namespace_and_the_root() -> None:
    import saealib
    import saealib.callback

    assert (
        saealib.PostEvaluationRequestEvent
        is saealib.callback.PostEvaluationRequestEvent
    )
    assert "PostEvaluationRequestEvent" in saealib.__all__
    assert "PostEvaluationRequestEvent" in saealib.callback.__all__
    for module_name in (
        "saealib.execution",
        "saealib.stages",
        "saealib.pipeline",
        "saealib.exceptions",
    ):
        module = importlib.import_module(module_name)
        assert "PostEvaluationRequestEvent" not in getattr(module, "__all__", [])
    assert "configuration" not in SUPPORTED_HISTORY_CHANNELS


def test_dispatching_the_event_without_a_handler_is_cheap() -> None:
    optimizer = _build("sync-direct")
    initializer = optimizer.initializer
    assert initializer is not None
    event = PostEvaluationRequestEvent(
        ctx=initializer.initialize(optimizer, optimizer.problem),
        request_id=np.int64(0),
        candidate_ids=np.asarray([0, 1], dtype=np.int64),
        size=2,
    )
    manager = optimizer.cbmanager
    runs = 20000
    start = time.perf_counter()
    for _ in range(runs):
        manager.dispatch(event)
    elapsed = (time.perf_counter() - start) / runs
    assert elapsed < 25e-6


def test_a_pickle_without_the_origin_registry_resumes_and_appends_origin_zero(
    tmp_path: Any,
) -> None:
    problem = _problem()
    optimizer = (
        Optimizer(problem, seed=7)
        .set_initializer(LHSInitializer(6, 4, 7))
        .set_algorithm(_ga())
        .set_strategy(DirectStrategy())
        .set_termination(Termination(max_gen(2)))
        .set_history(["summary", "evaluation"])
        .set_evaluator(SerialEvaluator())
    )
    state = optimizer.run()
    path = tmp_path / "old.pkl"
    optimizer.save_pickle(state, path)
    with open(path, "rb") as handle:
        old_optimizer, old_state = pickle.load(handle)
    history = old_state.history
    assert history is not None
    del history.__dict__["_evaluation_origin_kinds"]
    history._columns["evaluation"].pop("origin_kind", None)
    with open(path, "wb") as handle:
        pickle.dump((old_optimizer, old_state), handle)

    _restored_optimizer, restored = Optimizer.load_pickle(path)

    channel = _history(restored)
    assert "origin_kind" in channel
    assert set(np.asarray(channel["origin_kind"]).tolist()) == {0}
    rows_before = len(channel["size"])
    resumed = (
        Optimizer(problem, seed=7)
        .set_initializer(LHSInitializer(6, 4, 7))
        .set_algorithm(_ga())
        .set_strategy(DirectStrategy())
        .set_termination(Termination(max_gen(4)))
        .set_history(["summary", "evaluation"])
        .set_evaluator(SerialEvaluator())
        .run_from(restored)
    )
    channel = _history(resumed)
    assert len(channel["size"]) > rows_before
    assert set(np.asarray(channel["origin_kind"]).tolist()) == {0}
    assert _three_quantities(resumed) == (
        int(resumed.fe),
        int(resumed.fe),
        int(resumed.fe),
    )


def _buffered_evaluation(state: Any, ids: np.ndarray) -> None:
    result = EvaluationResult(
        f=np.zeros((len(ids), 1)),
        g=np.empty((len(ids), 0)),
        cv=np.zeros(len(ids)),
        candidate_ids=ids,
    )
    update = EvaluationUpdate(
        np.int64(0),
        EvaluationStatus.COMPLETED,
        ids,
        result,
        sequence=0,
    )
    state.history._observe_evaluation(update, 0, origin_kind=0)


def test_the_counting_function_records_history_before_it_dispatches() -> None:
    optimizer = _build("sync-direct")
    initializer = optimizer.initializer
    assert initializer is not None
    state = initializer.initialize(optimizer, optimizer.problem)
    ids = np.asarray(state.population.get_array("id"), dtype=np.int64)
    _buffered_evaluation(state, ids)
    fe_before = int(state.fe)

    def raise_from_handler(event: Any) -> None:
        raise RuntimeError("observer failed")

    report = _EvaluationReport(
        request_id=0,
        candidate_ids=ids,
        status=EvaluationStatus.COMPLETED,
        origin_kind=0,
        dispatch=raise_from_handler,
    )
    with pytest.raises(EvaluationFatalError) as error:
        _count_evaluated(state, ids, report=report)

    counted = error.value.state
    assert int(counted.fe) == fe_before + len(ids)
    assert _three_quantities(counted) == (
        fe_before + len(ids),
        fe_before + len(ids),
        fe_before + len(ids),
    )


def test_a_history_append_failure_leaves_the_count_and_rows_unchanged(
    monkeypatch: Any,
) -> None:
    optimizer = _build("sync-direct")
    initializer = optimizer.initializer
    assert initializer is not None
    state = initializer.initialize(optimizer, optimizer.problem)
    ids = np.asarray(state.population.get_array("id"), dtype=np.int64)
    _buffered_evaluation(state, ids)
    fe_before = int(state.fe)
    rows_before = _history_rows(state)

    def fail_append(self: Any, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("append failed")

    monkeypatch.setattr(History, "append_block", fail_append)
    report = _EvaluationReport(
        request_id=0,
        candidate_ids=ids,
        status=EvaluationStatus.COMPLETED,
        origin_kind=0,
    )
    with pytest.raises(RuntimeError):
        _begin_evaluation_report(state, report, flush=True)

    history = state.history
    assert history is not None
    assert int(state.fe) == fe_before
    assert _history_rows(state) == rows_before
    assert int(history._fe_counter) == fe_before
