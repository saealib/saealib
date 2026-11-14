"""Parameter writes applied at a component replacement commit point.

A write queued for the commit point reaches the published binding before the
difference check, so a value neither the contract nor the strategy values carry
takes effect without publishing anything, while a value the flow shape carries
goes through the ordinary replacement path.
"""

from __future__ import annotations

import pickle
from typing import Any, cast

import numpy as np
import pytest

from saealib import (
    GA,
    AsyncEvaluationScheduler,
    AsyncEvaluator,
    CrossoverBLXAlpha,
    DirectStrategy,
    GenerationStartEvent,
    LHSInitializer,
    MutationUniform,
    Optimizer,
    Problem,
    SequentialSelection,
    SerialEvaluator,
    Termination,
    TruncationSelection,
    max_gen,
)
from saealib.callback import CallbackManager
from saealib.core.graph_builder import build_component_graph
from saealib.core.runtime import SetParameter
from saealib.exceptions import ReplacementRejectedError
from saealib.execution.runner import Runner
from saealib.execution.runtime import create_runtime


def _sphere(x: Any) -> float:
    return float(np.sum(np.asarray(x, dtype=float) ** 2))


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


class _FlatDirect(DirectStrategy):
    """Direct flow lowered to a flat component graph, so its plan is sequential."""

    def build_graph(self, provider: Any) -> Any:
        return build_component_graph(self.build_pipeline(provider))


def _optimizer(n_gen: int = 4, *, flat: bool = False) -> Optimizer:
    optimizer = (
        Optimizer(_problem(), seed=7)
        .set_initializer(LHSInitializer(6, 4, 7))
        .set_algorithm(
            GA(
                crossover=CrossoverBLXAlpha(prob=0.9, alpha=0.4),
                mutation=MutationUniform(prob_var=0.1),
                parent_selection=SequentialSelection(),
                survivor_selection=TruncationSelection(),
            )
        )
        .set_termination(Termination(max_gen(n_gen)))
    )
    if flat:
        optimizer.set_strategy(_FlatDirect())
    return optimizer


def _async(optimizer: Optimizer) -> Optimizer:
    optimizer.evaluator = AsyncEvaluator(SerialEvaluator())
    optimizer.async_evaluation_scheduler = AsyncEvaluationScheduler(
        optimizer.evaluator, max_pending=3
    )
    return optimizer


class _Capture:
    def __init__(self) -> None:
        self.environment: Any = None

    def __call__(self, optimizer: Any) -> Any:
        runtime = create_runtime(optimizer)
        self.environment = getattr(runtime, "environment", None)
        return runtime


def _queue(optimizer: Optimizer, slot: str, path: str, value: object) -> None:
    optimizer._parameter_writes[(slot, path)] = SetParameter(
        slot=slot, path=path, value=value
    )


def _write_at(optimizer: Optimizer, generation: int, **write: Any) -> None:
    """Queue one parameter write from a generation-start callback."""

    def queue(event: Any) -> None:
        if event.ctx.gen == generation:
            _queue(optimizer, write["slot"], write["path"], write["value"])

    manager = CallbackManager()
    manager.register(GenerationStartEvent, queue)
    optimizer.cbmanager = manager


@pytest.mark.parametrize("asynchronous", [False, True])
def test_a_neutral_parameter_write_takes_effect_without_publishing(
    asynchronous: bool,
) -> None:
    optimizer = _optimizer()
    if asynchronous:
        _async(optimizer)
    _write_at(optimizer, 1, slot="algorithm", path="crossover.prob", value=0.31)
    capture = _Capture()
    states = list(Runner(optimizer, runtime_factory=capture).iterate())
    environment = capture.environment

    assert [state.__dict__["_binding_version"] for state in states] == [0, 0, 0, 0, 0]
    assert environment.active_version == 0
    assert len({state.__dict__["_binding_version"] for state in states}) == 1
    assert cast(Any, optimizer.algorithm).crossover.prob == pytest.approx(0.31)
    if asynchronous:
        assert environment._publication_deferred is False
    environment._refresh_plan_if_needed(states[-1])
    assert cast(Any, optimizer.algorithm).crossover.prob == pytest.approx(0.31)


@pytest.mark.filterwarnings(
    "ignore:Overriding build_graph\\(provider\\):DeprecationWarning"
)
@pytest.mark.parametrize("asynchronous", [False, True])
def test_a_neutral_parameter_write_keeps_a_sequential_plan(asynchronous: bool) -> None:
    optimizer = _optimizer(flat=True)
    if asynchronous:
        _async(optimizer)
    _write_at(optimizer, 1, slot="algorithm", path="crossover.prob", value=0.31)
    capture = _Capture()
    list(Runner(optimizer, runtime_factory=capture).iterate())
    environment = capture.environment

    assert environment.active_version == 0
    assert cast(Any, optimizer.algorithm).crossover.prob == pytest.approx(0.31)


def test_a_neutral_comparator_write_reaches_the_running_state() -> None:
    optimizer = _optimizer()
    capture = _Capture()
    generator = Runner(optimizer, runtime_factory=capture).iterate()
    state = next(generator)
    environment = capture.environment
    published = environment._active_snapshot.values["comparator"]

    _queue(optimizer, "comparator", "eps_cv", 0.5)
    environment._refresh_plan_if_needed(state)

    assert published.eps_cv == pytest.approx(0.5)
    assert state.comparator.eps_cv == pytest.approx(0.5)
    assert environment.active_version == 0
    generator.close()


def test_a_flow_shape_parameter_write_publishes_one_version() -> None:
    optimizer = _optimizer()
    optimizer.resolve_defaults()
    default = getattr(optimizer.strategy, "n_offspring", None)
    written = 3 if default != 3 else 4
    _write_at(optimizer, 1, slot="strategy", path="n_offspring", value=written)
    capture = _Capture()
    states = list(Runner(optimizer, runtime_factory=capture).iterate())
    environment = capture.environment

    assert [state.__dict__["_binding_version"] for state in states] == [0, 0, 1, 1, 1]
    assert environment.active_version == 1
    assert cast(Any, optimizer.strategy).n_offspring == written
    environment._refresh_plan_if_needed(states[-1])
    assert cast(Any, optimizer.strategy).n_offspring == written


def test_a_rejected_commit_restores_a_neutral_parameter_write() -> None:
    optimizer = _optimizer()

    def replace(event: Any) -> None:
        if event.ctx.gen == 1:
            _queue(optimizer, "algorithm", "crossover.prob", 0.31)
            optimizer.evaluation_planner = cast(Any, object())

    manager = CallbackManager()
    manager.register(GenerationStartEvent, replace)
    optimizer.cbmanager = manager
    capture = _Capture()
    states: list[Any] = []
    with pytest.raises(ReplacementRejectedError):
        for state in Runner(optimizer, runtime_factory=capture).iterate():
            states.append(state)

    assert cast(Any, optimizer.algorithm).crossover.prob == pytest.approx(0.9)
    assert list(optimizer._parameter_writes) == [("algorithm", "crossover.prob")]
    assert capture.environment.active_version == 0
    optimizer.evaluation_planner = capture.environment.binding_values()[
        "evaluation_planner"
    ]
    capture.environment._refresh_plan_if_needed(states[-1])
    assert cast(Any, optimizer.algorithm).crossover.prob == pytest.approx(0.31)
    assert optimizer._parameter_writes == {}


def test_the_later_write_wins_and_writing_the_current_value_publishes_nothing() -> None:
    optimizer = _optimizer()
    capture = _Capture()
    generator = Runner(optimizer, runtime_factory=capture).iterate()
    state = next(generator)
    environment = capture.environment

    _queue(optimizer, "algorithm", "crossover.prob", 0.31)
    _queue(optimizer, "algorithm", "crossover.prob", 0.42)
    environment._refresh_plan_if_needed(state)
    assert cast(Any, optimizer.algorithm).crossover.prob == pytest.approx(0.42)
    assert environment.active_version == 0

    _queue(optimizer, "algorithm", "crossover.prob", 0.42)
    environment._refresh_plan_if_needed(state)
    assert environment.active_version == 0
    generator.close()


def test_a_pending_parameter_write_neither_refuses_nor_travels_with_a_checkpoint(
    tmp_path: Any,
) -> None:
    optimizer = _optimizer()
    capture = _Capture()
    generator = Runner(optimizer, runtime_factory=capture).iterate()
    state = next(generator)
    environment = capture.environment
    target = tmp_path / "state.npz"

    _queue(optimizer, "algorithm", "crossover.prob", 0.31)
    state.save(target)
    pickled = pickle.loads(pickle.dumps(optimizer))

    assert pickled._parameter_writes == {}
    assert type(state).load(target, state.problem).step == state.step

    environment._refresh_plan_if_needed(state)
    state.save(target)
    pickled = pickle.loads(pickle.dumps(optimizer))
    loaded = type(state).load(target, state.problem)

    assert cast(Any, optimizer.algorithm).crossover.prob == pytest.approx(0.31)
    assert pickled._parameter_writes == {}
    assert loaded.step == state.step
    resumed = _optimizer()
    final = Runner(resumed).run_from(loaded)
    assert cast(Any, resumed.algorithm).crossover.prob == pytest.approx(0.9)
    assert final.step >= loaded.step
    generator.close()


def test_a_commit_point_applies_a_queued_write_while_evaluations_are_pending() -> None:
    optimizer = _async(_optimizer())
    capture = _Capture()
    generator = Runner(optimizer, runtime_factory=capture).iterate()
    state = next(generator)
    environment = capture.environment
    state.pending_evaluations[0] = cast(Any, object())

    _queue(optimizer, "algorithm", "crossover.prob", 0.31)
    environment._refresh_plan_if_needed(state)

    assert environment._publication_deferred is False
    assert environment.active_version == 0
    assert cast(Any, optimizer.algorithm).crossover.prob == pytest.approx(0.31)
    assert len(state.pending_evaluations) == 1
    state.pending_evaluations.clear()
    generator.close()
