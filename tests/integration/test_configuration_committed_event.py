"""The event a published binding version dispatches.

A commit point that leaves the configuration unchanged publishes nothing; a
publication dispatches one event to the callback manager of the published
version, after the new version is live.
"""

from __future__ import annotations

import dataclasses
from typing import Any, cast

import numpy as np
import pytest

from saealib import (
    GA,
    ConfigurationChange,
    ConfigurationCommittedEvent,
    CrossoverBLXAlpha,
    CrossoverSBX,
    GenerationEndEvent,
    LHSInitializer,
    MutationUniform,
    Optimizer,
    SequentialSelection,
    Termination,
    TruncationSelection,
    max_gen,
)
from saealib.callback import CallbackManager
from saealib.core.graph_builder import build_component_graph
from saealib.core.runtime import SequentialPlan
from saealib.exceptions import ReplacementRejectedError
from saealib.execution import AsyncEvaluationScheduler, SerialEvaluator
from saealib.execution.bindings import BINDING_SLOTS
from saealib.execution.history import SUPPORTED_HISTORY_CHANNELS
from saealib.execution.runner import Runner
from saealib.execution.runtime import _OptimizerEnvironment, create_runtime
from saealib.problem import Problem
from saealib.strategies import DirectStrategy, SteadyStateStrategy


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


class _FlatDirect(DirectStrategy):
    """A Direct flow lowered to a flat graph, so its plan is sequential."""

    def build_graph(self, provider: Any) -> Any:
        return build_component_graph(self.build_pipeline(provider))


def _optimizer(strategy: Any, *, await_: bool = False, seed: int = 7) -> Optimizer:
    optimizer = (
        Optimizer(_problem(), seed=seed)
        .set_initializer(LHSInitializer(6, 4, seed))
        .set_algorithm(_ga())
        .set_strategy(strategy)
        .set_termination(Termination(max_gen(6)))
    )
    if await_:
        evaluator = SerialEvaluator()
        optimizer = optimizer.set_evaluator(evaluator).set_async_evaluation_scheduler(
            AsyncEvaluationScheduler(evaluator, max_pending=2)
        )
    return optimizer


def _capturing_runner(optimizer: Optimizer) -> tuple[Runner, dict[str, Any]]:
    captured: dict[str, Any] = {}

    def factory(optimizer_obj: Any) -> Any:
        runtime = create_runtime(optimizer_obj)
        captured["environment"] = getattr(runtime, "environment", None)
        return runtime

    return Runner(optimizer, runtime_factory=factory), captured


def _record_events(optimizer: Optimizer) -> list[ConfigurationCommittedEvent]:
    events: list[ConfigurationCommittedEvent] = []
    optimizer.cbmanager.register(ConfigurationCommittedEvent, events.append)
    return events


def _replace_at(
    optimizer: Optimizer, generation: int, action: Any
) -> list[ConfigurationCommittedEvent]:
    events = _record_events(optimizer)

    def on_generation_end(event: Any) -> None:
        if event.ctx.gen == generation:
            action()

    optimizer.cbmanager.register(GenerationEndEvent, on_generation_end)
    return events


def _count_commit_point_entries(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    entries = [0]
    original = _OptimizerEnvironment._refresh_plan_if_needed

    def counted(self: Any, state: Any = None) -> None:
        entries[0] += 1
        return original(self, state)

    monkeypatch.setattr(_OptimizerEnvironment, "_refresh_plan_if_needed", counted)
    return entries


def test_a_run_without_a_replacement_enters_commit_points_and_fires_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entries = _count_commit_point_entries(monkeypatch)
    optimizer = _optimizer(DirectStrategy())
    events = _record_events(optimizer)

    state = optimizer.run()

    assert state.gen == 6
    assert entries[0] == 7
    assert events == []


def test_a_replacement_fires_one_event_after_the_new_version_is_published() -> None:
    optimizer = _optimizer(DirectStrategy())
    replacement = _ga()
    observed: list[dict[str, Any]] = []

    def record(event: ConfigurationCommittedEvent) -> None:
        published = optimizer._last_published_bindings
        observed.append(
            {
                "version": event.version,
                "state_version": event.ctx.__dict__.get("_binding_version"),
                "published_version": None if published is None else published.version,
                "changes": event.changes,
                "algorithm": optimizer.algorithm,
            }
        )

    optimizer.cbmanager.register(ConfigurationCommittedEvent, record)
    runner, _captured = _capturing_runner(optimizer)
    generator = runner.iterate()
    next(generator)
    version_before = _captured["environment"].active_version
    optimizer.set_algorithm(replacement)
    for _ in generator:
        pass

    assert version_before == 0
    assert len(observed) == 1
    event = observed[0]
    assert event["version"] == 1
    assert event["state_version"] == event["version"]
    assert event["published_version"] == event["version"]
    assert event["algorithm"] is replacement
    assert len(event["changes"]) == 1
    change = event["changes"][0]
    assert isinstance(change, ConfigurationChange)
    assert change.slot == "algorithm"
    assert change.previous is not replacement
    assert change.current is replacement
    assert change.source == "setter"


def test_several_replaced_slots_share_one_event_in_binding_order() -> None:
    optimizer = _optimizer(DirectStrategy())
    algorithm = _ga()
    strategy = DirectStrategy()
    events = _replace_at(
        optimizer,
        2,
        lambda: (
            optimizer.set_algorithm(algorithm),
            optimizer.set_strategy(strategy),
        ),
    )
    runner, _captured = _capturing_runner(optimizer)
    for _ in runner.iterate():
        pass

    assert len(events) == 1
    slots = [change.slot for change in events[0].changes]
    assert slots == ["algorithm", "strategy"]
    assert slots == sorted(slots, key=BINDING_SLOTS.index)
    assert [change.current for change in events[0].changes] == [algorithm, strategy]
    assert events[0].version == 1


def test_a_rejected_replacement_fires_no_event() -> None:
    optimizer = _optimizer(DirectStrategy())
    events = _record_events(optimizer)
    runner, captured = _capturing_runner(optimizer)
    generator = runner.iterate()
    next(generator)
    published_planner = optimizer.evaluation_planner
    assert published_planner is not None
    optimizer.evaluation_planner = cast(Any, object())

    with pytest.raises(ReplacementRejectedError):
        for _ in generator:
            pass

    assert events == []
    assert captured["environment"].active_version == 0
    assert optimizer.evaluation_planner is published_planner


def test_a_nested_part_change_publishes_one_event_without_replaced_slots() -> None:
    optimizer = _optimizer(DirectStrategy())
    events = _replace_at(
        optimizer,
        2,
        lambda: setattr(
            optimizer.algorithm, "crossover", CrossoverSBX(prob=0.9, eta=15.0)
        ),
    )
    runner, captured = _capturing_runner(optimizer)
    for _ in runner.iterate():
        pass

    assert len(events) == 1
    assert events[0].changes == ()
    assert events[0].version == captured["environment"].active_version


def test_a_direct_assignment_is_reported_as_its_source() -> None:
    optimizer = _optimizer(DirectStrategy())
    replacement = _ga()
    events = _replace_at(
        optimizer, 2, lambda: setattr(optimizer, "algorithm", replacement)
    )
    runner, _captured = _capturing_runner(optimizer)
    for _ in runner.iterate():
        pass

    assert len(events) == 1
    assert [(change.slot, change.source) for change in events[0].changes] == [
        ("algorithm", "direct assignment")
    ]


def test_a_component_the_preset_supplies_is_reported_as_its_source() -> None:
    optimizer = _optimizer(DirectStrategy())
    events = _record_events(optimizer)
    runner, _captured = _capturing_runner(optimizer)
    generator = runner.iterate()
    next(generator)
    next(generator)
    optimizer.set_preset(
        {"feedback_builder": {"type": "saealib.policies.feedback.TrueOnlyFeedback"}}
    )
    for _ in generator:
        pass

    assert len(events) == 1
    assert [(change.slot, change.source) for change in events[0].changes] == [
        ("feedback_builder", "user preset")
    ]


def test_the_event_reaches_the_manager_of_the_published_version() -> None:
    optimizer = _optimizer(DirectStrategy())
    replaced_events: list[ConfigurationCommittedEvent] = []
    replacement_manager = CallbackManager()
    replacement_events: list[ConfigurationCommittedEvent] = []
    replacement_manager.register(ConfigurationCommittedEvent, replacement_events.append)
    optimizer.cbmanager.register(ConfigurationCommittedEvent, replaced_events.append)

    def replace_manager() -> None:
        optimizer.cbmanager = replacement_manager

    events = _replace_at(optimizer, 2, replace_manager)
    runner, _captured = _capturing_runner(optimizer)
    for _ in runner.iterate():
        pass

    assert events == []
    assert replaced_events == []
    assert len(replacement_events) == 1
    assert {
        (change.slot, change.current) for change in replacement_events[0].changes
    } == {("cbmanager", replacement_manager)}


@pytest.mark.filterwarnings(
    "ignore:Overriding build_graph\\(provider\\):DeprecationWarning"
)
@pytest.mark.parametrize(
    ("strategy", "await_"),
    [
        (DirectStrategy(), False),
        (SteadyStateStrategy(), True),
        (_FlatDirect(), False),
    ],
)
def test_the_event_fires_on_every_flow(strategy: Any, await_: bool) -> None:
    optimizer = _optimizer(strategy, await_=await_)
    replacement = _ga()
    events = _replace_at(optimizer, 2, lambda: optimizer.set_algorithm(replacement))
    runner, captured = _capturing_runner(optimizer)
    for _ in runner.iterate():
        pass

    assert isinstance(captured["environment"].plan, SequentialPlan) is isinstance(
        strategy, _FlatDirect
    )
    assert len(events) == 1
    assert events[0].version == captured["environment"].active_version == 1


def _trajectory(optimizer: Optimizer) -> list[tuple[int, int, float]]:
    rows: list[tuple[int, int, float]] = []
    for state in optimizer.iterate():
        objectives = np.asarray(state.archive.get_array("f"), dtype=np.float64)
        rows.append(
            (state.gen, state.fe, float(np.min(objectives)) if objectives.size else 0.0)
        )
    return rows


def test_registering_a_handler_does_not_change_the_trajectory() -> None:
    control = _optimizer(DirectStrategy())
    watched = _optimizer(DirectStrategy())
    _record_events(watched)

    control_rows = _trajectory(control)
    watched_rows = _trajectory(watched)

    assert watched_rows == control_rows


def test_the_event_carries_no_parameters_and_adds_no_history_channel() -> None:
    fields = {field.name for field in dataclasses.fields(ConfigurationCommittedEvent)}

    assert fields == {"ctx", "version", "changes"}
    assert "configuration" not in SUPPORTED_HISTORY_CHANNELS


def test_a_raising_handler_does_not_undo_the_publication() -> None:
    optimizer = _optimizer(DirectStrategy())
    replacement = _ga()

    def raise_from_handler(event: ConfigurationCommittedEvent) -> None:
        raise RuntimeError("observer failed")

    optimizer.cbmanager.register(ConfigurationCommittedEvent, raise_from_handler)
    runner, captured = _capturing_runner(optimizer)
    generator = runner.iterate()
    next(generator)
    version_before = captured["environment"].active_version
    optimizer.set_algorithm(replacement)
    with pytest.raises(RuntimeError):
        for _ in generator:
            pass

    environment = captured["environment"]
    published = optimizer._last_published_bindings
    assert published is not None
    assert environment.active_version == version_before + 1
    assert published.version == version_before + 1
    assert environment.binding_values()["algorithm"] is replacement
    assert optimizer.algorithm is replacement
    assert published.values["algorithm"] is replacement
    state = environment._last_state
    assert state is not None
    assert state.__dict__["_binding_version"] == published.version
