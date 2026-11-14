"""Behavior of the controller helpers and the controller registration."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest

from saealib import (
    Controller,
    FixedSchedule,
    Optimizer,
    Problem,
    ReplaceComponent,
    SetParameter,
    ThresholdController,
)
from saealib.context import OptimizationState
from saealib.controllers import SmoothedController, StagnationController
from saealib.core.state import (
    RUNTIME_STAGNATION,
    SURROGATES_ACCURACY,
    SURROGATES_SMOOTHED_ERROR,
    StateKey,
)
from saealib.exceptions import ValidationError
from saealib.execution.history import History
from saealib.population import Archive, ParetoArchive, Population, PopulationAttribute
from saealib.registry import build, to_spec

_OBSERVED = StateKey[object](namespace="user", name="observed", schema_version=1)


def _measured(state: Any) -> float:
    """Module-level measure, so a controller spec can name it."""
    return 1.0


def _problem() -> Problem:
    return Problem(
        func=lambda x: float(np.sum(np.asarray(x, dtype=float) ** 2)),
        dim=2,
        n_obj=1,
        direction=np.array([-1.0]),
        lb=[-2.0, -2.0],
        ub=[2.0, 2.0],
        constraints=[],
    )


def _state() -> OptimizationState:
    problem = _problem()
    attributes = [
        PopulationAttribute("x", np.float64, (2,), np.nan),
        PopulationAttribute("f", np.float64, (1,), np.nan),
        PopulationAttribute("id", np.int64, (), -1),
    ]
    rows = {
        "x": np.zeros((1, 2)),
        "f": np.zeros((1, 1)),
        "id": np.full(1, -1, dtype=np.int64),
    }
    population = Population(attributes, 4)
    archive = Archive(attributes, 4)
    population.extend(rows)
    archive.extend(rows)
    return OptimizationState(
        problem=problem,
        population=population,
        archive=archive,
        pareto_archive=ParetoArchive(
            attributes, 4, direction=problem.direction, constraints=[]
        ),
        rng=np.random.default_rng(0),
        history=History(),
    )


def _accuracy(state: OptimizationState, spearman: float) -> None:
    state.set_state(
        SURROGATES_ACCURACY,
        {"metrics": {"spearman": spearman}, "n_samples": 8},
    )


def test_fixed_schedule_emits_the_commands_of_the_keyed_counter() -> None:
    command = SetParameter(slot="algorithm", path="crossover.prob", value=0.5)
    state = _state()
    schedule = FixedSchedule({1: [command]})

    assert schedule.decide(state) == ()
    state.step = 1
    assert schedule.decide(state) == (command,)
    assert FixedSchedule({0: [command]}, key="gen").decide(state) == (command,)


def test_fixed_schedule_rejects_an_unknown_counter() -> None:
    with pytest.raises(ValidationError):
        FixedSchedule({}, key="fe")


def test_threshold_controller_selects_the_level_the_metric_reaches() -> None:
    state = _state()
    low, mid, high = object(), object(), object()
    controller = ThresholdController(
        slot="strategy", levels=[(0.0, low), (0.5, mid), (0.9, high)]
    )

    assert cast(Any, controller.decide(state)[0]).component is low
    _accuracy(state, 0.6)
    assert cast(Any, controller.decide(state)[0]).component is mid
    _accuracy(state, 0.95)
    assert cast(Any, controller.decide(state)[0]).component is high
    _accuracy(state, float("nan"))
    assert cast(Any, controller.decide(state)[0]).component is low


def test_threshold_controller_requires_ascending_levels() -> None:
    with pytest.raises(ValidationError):
        ThresholdController(slot="strategy", levels=[(0.5, object()), (0.1, object())])
    with pytest.raises(ValidationError):
        ThresholdController(slot="strategy", levels=[])


def test_smoothed_controller_writes_the_mapped_parameter_and_keeps_its_state() -> None:
    state = _state()
    controller = SmoothedController(gm_max=5, gm_min=0, update_rate=0.5)

    first = cast(Any, controller.decide(state)[0])
    assert isinstance(first, SetParameter)
    assert first.path == "gen_ctrl"
    assert first.value == round((1.0 - 0.5) * 5)
    assert state.get_state(controller.key) == {"smoothed_error": 0.5}

    _accuracy(state, 1.0)
    second = cast(Any, controller.decide(state)[0])
    assert second.value == round((1.0 - 0.25) * 5)
    assert state.get_state(controller.key)["smoothed_error"] == pytest.approx(0.25)


def test_smoothed_controller_rejects_a_rate_outside_the_unit_interval() -> None:
    with pytest.raises(ValidationError):
        SmoothedController(update_rate=0.0)


def test_stagnation_controller_fires_once_per_patience_window() -> None:
    state = _state()
    command = SetParameter(slot="algorithm", path="crossover.prob", value=0.2)
    controller = StagnationController(
        patience=2, commands=[command], measure=lambda current: 1.0
    )

    fired: list[int] = []
    for step in range(5):
        state.step = step
        if controller.decide(state):
            fired.append(step)

    assert fired == [2, 4]
    assert state.get_state(RUNTIME_STAGNATION)["last_improvement_step"] == 4


def test_stagnation_controller_restarts_its_window_on_improvement() -> None:
    state = _state()
    measures = {0: 1.0, 1: 2.0, 2: 2.0, 3: 3.0, 4: 3.0}
    controller = StagnationController(
        patience=2,
        commands=[SetParameter(slot="algorithm", path="crossover.prob", value=0.2)],
        measure=lambda current: measures[cast(Any, current).step],
    )

    fired: list[int] = []
    for step in range(5):
        state.step = step
        if controller.decide(state):
            fired.append(step)

    assert fired == []


def test_stagnation_controller_requires_a_measure() -> None:
    with pytest.raises(TypeError):
        cast(Any, StagnationController)(patience=2, commands=[])


def test_every_helper_round_trips_through_to_spec() -> None:
    helpers: list[Controller] = [
        FixedSchedule(
            {1: [SetParameter(slot="algorithm", path="crossover.prob", value=0.5)]}
        ),
        FixedSchedule({}, key="gen"),
        ThresholdController(
            slot="strategy", levels=[(0.0, "fallback"), (1.0, "primary")]
        ),
        SmoothedController(),
        StagnationController(patience=2, commands=[], measure=_measured),
    ]

    for helper in helpers:
        spec = to_spec(helper)
        assert to_spec(build(spec)) == spec


def test_the_helpers_declare_their_state_keys() -> None:
    assert SmoothedController().state_keys() == (SURROGATES_SMOOTHED_ERROR,)
    assert StagnationController(
        patience=1, commands=[], measure=_measured
    ).state_keys() == (RUNTIME_STAGNATION,)
    assert (
        ThresholdController(slot="strategy", levels=[(0.0, object())]).state_keys()
        == ()
    )


def test_add_controller_requires_a_callable_decide() -> None:
    optimizer = Optimizer(_problem(), seed=0)

    with pytest.raises(ValidationError):
        optimizer.add_controller(object())

    assert optimizer.controllers == []


def test_a_registered_controller_keeps_its_constructor_arguments() -> None:
    controller = ThresholdController(
        slot="surrogate_manager", levels=[(0.0, "fallback"), (0.5, "primary")]
    )

    assert vars(controller)["slot"] == "surrogate_manager"
    assert not hasattr(controller, "contract")
    assert isinstance(FixedSchedule({}).decide(_state()), tuple)


def test_a_replacement_command_names_the_slot_and_component() -> None:
    command = ReplaceComponent(slot="strategy", component="strategy")

    assert command.slot == "strategy"
    assert command.component == "strategy"
    with pytest.raises(ValidationError):
        ReplaceComponent(slot="", component="strategy")
    with pytest.raises(ValidationError):
        SetParameter(slot="strategy", path="", value=1)
