"""Controllers consulted at the component replacement commit point."""

from __future__ import annotations

import pickle
from typing import Any, cast

import numpy as np
import pytest

from saealib import (
    GA,
    AsyncEvaluationScheduler,
    AsyncEvaluator,
    CallbackManager,
    Controller,
    CrossoverBLXAlpha,
    DirectStrategy,
    GenerationBasedStrategy,
    GenerationStartEvent,
    GlobalSurrogateManager,
    IndividualBasedStrategy,
    KFoldAccuracyEvaluator,
    LHSInitializer,
    MeanPrediction,
    MutationUniform,
    Optimizer,
    PreSelectionStrategy,
    Problem,
    ReplaceComponent,
    SequentialSelection,
    SerialEvaluator,
    SetParameter,
    SklearnGPRSurrogate,
    SteadyStateStrategy,
    Termination,
    TruncationSelection,
    max_gen,
)
from saealib.controllers import _accuracy_state_value
from saealib.core.graph_builder import build_component_graph
from saealib.core.state import SURROGATES_ACCURACY, StateKey
from saealib.exceptions import ReplacementRejectedError, ValidationError
from saealib.execution.runner import Runner
from saealib.execution.runtime import create_runtime
from saealib.surrogate import ArchiveObjectiveSet

_STEPS = StateKey[object](namespace="user", name="decision_steps", schema_version=1)


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


def _ga() -> GA:
    return GA(
        crossover=CrossoverBLXAlpha(prob=0.9, alpha=0.4),
        mutation=MutationUniform(prob_var=0.1),
        parent_selection=SequentialSelection(),
        survivor_selection=TruncationSelection(),
    )


class _FlatDirect(DirectStrategy):
    """Direct flow lowered to a flat component graph, so its plan is sequential."""

    def build_graph(self, provider: Any) -> Any:
        return build_component_graph(self.build_pipeline(provider))


class Recording(Controller):
    """Append the step value it is asked about."""

    def __init__(self) -> None:
        self.calls: list[int] = []

    def decide(self, state: Any) -> tuple[()]:
        self.calls.append(state.step)
        record = dict(state.get_state(_STEPS) or {})
        record["steps"] = [*record.get("steps", []), state.step]
        state.set_state(_STEPS, record)
        return ()

    def state_keys(self) -> tuple[StateKey, ...]:
        return (_STEPS,)


class Silent(Controller):
    """Decide nothing."""

    def decide(self, state: Any) -> tuple[()]:
        return ()


def _optimizer(
    strategy: Any = None, *, asynchronous: bool = False, generations: int = 3
) -> Optimizer:
    optimizer = (
        Optimizer(_problem(), seed=3)
        .set_initializer(LHSInitializer(6, 4, 3))
        .set_algorithm(_ga())
        .set_surrogate_manager(GlobalSurrogateManager(SklearnGPRSurrogate()))
        .set_acquisition(MeanPrediction())
        .set_termination(Termination(max_gen(generations)))
    )
    if strategy is not None:
        optimizer.set_strategy(strategy)
    if asynchronous:
        optimizer.evaluator = AsyncEvaluator(SerialEvaluator())
        optimizer.async_evaluation_scheduler = AsyncEvaluationScheduler(
            optimizer.evaluator, max_pending=2
        )
    return optimizer


def _accuracy_manager() -> GlobalSurrogateManager:
    return GlobalSurrogateManager(
        SklearnGPRSurrogate(),
        ArchiveObjectiveSet(),
        accuracy_evaluator=KFoldAccuracyEvaluator(n_splits=2),
    )


@pytest.mark.filterwarnings(
    "ignore:Overriding build_graph\\(provider\\):DeprecationWarning"
)
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize(
    "strategy",
    [
        DirectStrategy(),
        GenerationBasedStrategy(gen_ctrl=1),
        IndividualBasedStrategy(evaluation_ratio=0.1),
        PreSelectionStrategy(n_candidates=8, n_select=4),
        SteadyStateStrategy(),
        _FlatDirect(),
    ],
    ids=[
        "direct",
        "generation-based",
        "individual-based",
        "preselection",
        "steady-state",
        "sequential",
    ],
)
def test_a_controller_decides_once_per_step_value(
    strategy: Any, asynchronous: bool
) -> None:
    optimizer = _optimizer(strategy, asynchronous=asynchronous)
    recorder = Recording()
    optimizer.add_controller(recorder)

    final = None
    for state in Runner(optimizer).iterate():
        final = state

    assert final is not None
    assert recorder.calls == list(range(len(recorder.calls)))
    assert final.get_state(_STEPS)["steps"] == recorder.calls
    assert recorder.calls[-1] >= final.step - 1


def test_a_controller_that_decides_nothing_changes_nothing() -> None:
    plain = list(Runner(_optimizer(DirectStrategy())).iterate())
    optimizer = _optimizer(DirectStrategy())
    optimizer.add_controller(Silent())
    states = list(Runner(optimizer).iterate())

    assert {state.__dict__["_binding_version"] for state in states} == {0}
    assert [(state.gen, state.step, state.fe) for state in states] == [
        (state.gen, state.step, state.fe) for state in plain
    ]


def _observation(state: Any) -> Any:
    try:
        return state.get_state(SURROGATES_ACCURACY)
    except KeyError:
        return "absent"


def test_replacing_the_bound_manager_clears_the_accuracy_observation() -> None:
    primary = _accuracy_manager()
    replacement = _accuracy_manager()

    class ReplaceAtStep(Controller):
        def decide(self, state: Any) -> Any:
            if state.step == 2:
                return (
                    ReplaceComponent(slot="surrogate_manager", component=replacement),
                )
            return ()

    optimizer = _optimizer(
        IndividualBasedStrategy(evaluation_ratio=0.5), generations=5
    ).set_surrogate_manager(primary)
    optimizer.add_controller(ReplaceAtStep())
    observations: dict[int, Any] = {}
    bound: dict[int, Any] = {}

    def record(event: Any) -> None:
        observations[event.ctx.step] = _observation(event.ctx)
        bound[event.ctx.step] = _accuracy_state_value(
            optimizer.surrogate_manager.last_accuracy
        )

    manager = CallbackManager()
    manager.register(GenerationStartEvent, record)
    optimizer.cbmanager = manager
    for _ in Runner(optimizer).iterate():
        pass

    assert optimizer.surrogate_manager is replacement
    assert isinstance(observations[1], dict)
    assert set(observations[1]["metrics"]) == set(
        cast(Any, primary.last_accuracy).metrics
    )
    assert observations[2] is None
    assert isinstance(observations[3], dict)
    assert {step: value for step, value in observations.items() if step} == {
        step: value for step, value in bound.items() if step
    }


def test_the_accuracy_observation_survives_a_portable_checkpoint(tmp_path: Any) -> None:
    optimizer = _optimizer(
        IndividualBasedStrategy(evaluation_ratio=0.5)
    ).set_surrogate_manager(_accuracy_manager())
    optimizer.add_controller(Silent())
    generator = Runner(optimizer).iterate()
    next(generator)
    state = next(generator)
    target = tmp_path / "state.npz"

    state.save(target)
    loaded = type(state).load(target, state.problem)

    assert isinstance(loaded.get_state(SURROGATES_ACCURACY), dict)
    assert loaded.get_state(SURROGATES_ACCURACY) == state.get_state(SURROGATES_ACCURACY)
    generator.close()


@pytest.mark.filterwarnings(
    "ignore:Overriding build_graph\\(provider\\):DeprecationWarning"
)
def _resumed_final(optimizer: Optimizer, recorder: Controller, path: Any) -> list[Any]:
    """Save mid-run, resume with the same controller, return the final state."""
    generator = Runner(optimizer).iterate()
    next(generator)
    state = next(generator)
    state.save(path)
    generator.close()
    loaded = type(state).load(path, _problem())
    resumed = (
        Optimizer(loaded.problem, seed=3)
        .set_initializer(LHSInitializer(6, 4, 3))
        .set_algorithm(_ga())
        .set_surrogate_manager(GlobalSurrogateManager(SklearnGPRSurrogate()))
        .set_acquisition(MeanPrediction())
        .set_termination(Termination(max_gen(3)))
        .set_strategy(DirectStrategy())
    )
    resumed.add_controller(recorder)
    final = Runner(resumed).run_from(loaded)
    return [final.gen, final.fe, final.step, final.population.get_array("x").tolist()]


def test_a_resumed_run_decides_the_same_steps_as_an_uninterrupted_one(
    tmp_path: Any,
) -> None:
    uninterrupted = _optimizer(DirectStrategy())
    uninterrupted.add_controller(Recording())
    reference = Runner(uninterrupted).run()

    resumed = _resumed_final(
        _optimizer(DirectStrategy()), Recording(), tmp_path / "state.npz"
    )

    assert resumed == [
        reference.gen,
        reference.fe,
        reference.step,
        reference.population.get_array("x").tolist(),
    ]


def test_a_pickled_run_keeps_its_controllers() -> None:
    optimizer = _optimizer(DirectStrategy())
    recorder = Recording()
    optimizer.add_controller(recorder)
    generator = Runner(optimizer).iterate()
    next(generator)
    state = next(generator)
    generator.close()

    restored = pickle.loads(pickle.dumps(optimizer))
    final = Runner(restored).run_from(state)

    assert [type(item).__name__ for item in restored.controllers] == ["Recording"]
    assert final.step == 3


def test_a_rejected_controller_command_restores_the_declared_state() -> None:
    class ReplaceWhileRejected(Controller):
        def __init__(self) -> None:
            self.calls: list[int] = []
            self.reject = True

        def decide(self, state: Any) -> Any:
            self.calls.append(state.step)
            record = dict(state.get_state(_STEPS) or {})
            record["steps"] = [*record.get("steps", []), state.step]
            state.set_state(_STEPS, record)
            if state.step == 2 and self.reject:
                return (
                    ReplaceComponent(
                        slot="evaluation_planner", component=cast(Any, object())
                    ),
                )
            return ()

        def state_keys(self) -> tuple[StateKey, ...]:
            return (_STEPS,)

    optimizer = _optimizer(DirectStrategy())
    controller = ReplaceWhileRejected()
    optimizer.add_controller(controller)
    states: list[Any] = []
    with pytest.raises(ReplacementRejectedError) as error:
        for state in Runner(optimizer).iterate():
            states.append(state)
    last = states[-1]

    assert error.value.source == "controller 0 (ReplaceWhileRejected)"
    assert last.get_state(_STEPS)["steps"] == [0, 1]
    assert controller.calls == [0, 1, 2]

    controller.reject = False
    final = Runner(optimizer).run_from(last)

    assert final.get_state(_STEPS)["steps"] == [0, 1, 2, 3]
    assert controller.calls == [0, 1, 2, 2, 3]


def test_a_command_queued_before_a_failing_controller_leaves_no_trace() -> None:
    class _QueueWrite(Controller):
        def decide(self, state: Any) -> tuple[Any, ...]:
            return (SetParameter(slot="algorithm", path="crossover.prob", value=0.31),)

    class _Refuse(Controller):
        def decide(self, state: Any) -> tuple[Any, ...]:
            return (cast(Any, object()),)

    optimizer = _optimizer(DirectStrategy())
    optimizer.add_controller(_QueueWrite())
    optimizer.add_controller(_Refuse())
    generator = Runner(optimizer).iterate()
    state = next(generator)

    with pytest.raises(ValidationError, match="not a replacement"):
        next(generator)

    assert optimizer._parameter_writes == {}
    assert cast(Any, optimizer.algorithm).crossover.prob == pytest.approx(0.9)

    class _Empty(Controller):
        def decide(self, state: Any) -> tuple[()]:
            return ()

    optimizer.controllers = [_Empty()]
    final = Runner(optimizer).run_from(state)

    assert cast(Any, optimizer.algorithm).crossover.prob == pytest.approx(0.9)
    assert final.gen >= state.gen
    generator.close()


def _state_at_step(step: int) -> tuple[Any, Any, Any, Any]:
    optimizer = _optimizer(DirectStrategy())
    capture: dict[str, Any] = {}

    def factory(built: Any) -> Any:
        runtime = create_runtime(built)
        capture["environment"] = cast(Any, runtime).environment
        return runtime

    generator = Runner(optimizer, runtime_factory=factory).iterate()
    state = next(generator)
    for _ in range(step):
        state = next(generator)
    return optimizer, state, capture["environment"], generator


def test_a_declared_state_value_mutated_in_place_is_restored() -> None:
    class InPlace(Controller):
        def decide(self, state: Any) -> tuple[Any, ...]:
            state.get_state(_STEPS)["steps"].append(state.step)
            return (
                ReplaceComponent(
                    slot="evaluation_planner", component=cast(Any, object())
                ),
            )

        def state_keys(self) -> tuple[StateKey, ...]:
            return (_STEPS,)

    optimizer, state, environment, generator = _state_at_step(1)
    state.set_state(_STEPS, {"steps": []})
    optimizer.add_controller(InPlace())

    with pytest.raises(ReplacementRejectedError):
        environment._refresh_plan_if_needed(state)
    generator.close()

    assert state.get_state(_STEPS) == {"steps": []}
    assert environment.active_version == 0


def test_a_rejection_names_the_controller_that_returned_the_command() -> None:
    class _Reject(Controller):
        def decide(self, state: Any) -> tuple[Any, ...]:
            return (
                ReplaceComponent(
                    slot="evaluation_planner", component=cast(Any, object())
                ),
            )

    class _Empty(Controller):
        def decide(self, state: Any) -> tuple[()]:
            return ()

    optimizer, state, environment, generator = _state_at_step(1)
    optimizer.add_controller(_Reject())
    optimizer.add_controller(_Empty())

    with pytest.raises(ReplacementRejectedError) as error:
        environment._refresh_plan_if_needed(state)
    generator.close()

    assert error.value.source is not None
    assert "Reject" in error.value.source


def test_a_rejection_after_a_deferral_restores_the_first_decision() -> None:
    class Deferred(Controller):
        def decide(self, state: Any) -> tuple[Any, ...]:
            record = dict(state.get_state(_STEPS) or {})
            record["steps"] = [*record.get("steps", []), state.step]
            state.set_state(_STEPS, record)
            return (
                SetParameter(slot="algorithm", path="crossover.prob", value=0.31),
                ReplaceComponent(
                    slot="evaluation_planner", component=cast(Any, object())
                ),
            )

        def state_keys(self) -> tuple[StateKey, ...]:
            return (_STEPS,)

    optimizer = _optimizer(DirectStrategy(), asynchronous=True)
    capture: dict[str, Any] = {}

    def factory(built: Any) -> Any:
        runtime = create_runtime(built)
        capture["environment"] = cast(Any, runtime).environment
        return runtime

    generator = Runner(optimizer, runtime_factory=factory).iterate()
    state = next(generator)
    environment = capture["environment"]
    state.set_state(_STEPS, {"steps": []})
    optimizer.add_controller(Deferred())
    state.pending_evaluations[0] = cast(Any, object())

    environment._refresh_plan_if_needed(state)
    assert environment._publication_deferred is True
    assert cast(Any, optimizer.algorithm).crossover.prob == pytest.approx(0.31)

    state.pending_evaluations.clear()
    with pytest.raises(ReplacementRejectedError) as error:
        environment._refresh_plan_if_needed(state)
    generator.close()

    assert cast(Any, optimizer.algorithm).crossover.prob == pytest.approx(0.9)
    assert state.get_state(_STEPS) == {"steps": []}
    assert environment.active_version == 0
    assert error.value.source is not None
    assert "Deferred" in error.value.source


def test_a_deferred_commit_point_keeps_the_controller_state() -> None:
    optimizer = _optimizer(DirectStrategy(), asynchronous=True)
    controller = Recording()
    optimizer.add_controller(controller)
    capture: dict[str, Any] = {}

    def factory(built: Any) -> Any:
        runtime = create_runtime(built)
        capture["environment"] = cast(Any, runtime).environment
        return runtime

    generator = Runner(optimizer, runtime_factory=factory).iterate()
    next(generator)
    state = next(generator)
    environment = capture["environment"]
    state.pending_evaluations[0] = cast(Any, object())
    environment._controller_step = None
    optimizer.strategy = IndividualBasedStrategy(evaluation_ratio=0.1)

    environment._refresh_plan_if_needed(state)

    assert environment._publication_deferred is True
    assert state.get_state(_STEPS)["steps"] == [0, 1]
    assert controller.calls == [0, 1]
    state.pending_evaluations.clear()
    generator.close()
