"""The documented switcher examples rebuilt with controllers.

Each test keeps the loop the documentation shows as the oracle and runs the
same decision as a controller registered on the optimizer.  Both runs are
compared by what the run had bound at each yield, not by reading the helper.
A loop applies its choice at the next commit point, so a controller's bound
value at one yield is the loop's choice at the previous one.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest

from saealib import (
    GA,
    CrossoverBLXAlpha,
    GenerationBasedStrategy,
    GlobalSurrogateManager,
    IndividualBasedStrategy,
    KFoldAccuracyEvaluator,
    LHSInitializer,
    MeanPrediction,
    MutationUniform,
    Optimizer,
    PreSelectionStrategy,
    Problem,
    SequentialSelection,
    SklearnGPRSurrogate,
    Termination,
    ThresholdController,
    TruncationSelection,
    max_fe,
    max_gen,
)
from saealib.controllers import SmoothedController
from saealib.surrogate import ArchiveObjectiveSet
from saealib.surrogate.switching import (
    GenCtrlSwitcher,
    ManagerSwitcher,
    StrategySwitcher,
)


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


def _optimizer(termination: Termination, strategy: Any = None) -> Optimizer:
    optimizer = (
        Optimizer(_problem(), seed=0)
        .set_initializer(LHSInitializer(6, 4, 0))
        .set_algorithm(_ga())
        .set_acquisition(MeanPrediction())
        .set_termination(termination)
    )
    if strategy is not None:
        optimizer.set_strategy(strategy)
    return optimizer


def _manager_pair() -> tuple[Any, Any]:
    evaluator = KFoldAccuracyEvaluator(n_splits=2)
    primary = GlobalSurrogateManager(
        SklearnGPRSurrogate(), ArchiveObjectiveSet(), accuracy_evaluator=evaluator
    )
    fallback = GlobalSurrogateManager(SklearnGPRSurrogate(), ArchiveObjectiveSet())
    return primary, fallback


def _assert_shifted(in_effect: list[Any], applied: list[Any], initial: Any) -> None:
    """The loop's last choice has no step left to affect."""
    assert in_effect[0] is initial
    assert in_effect[1:] == applied[:-1]


@pytest.mark.filterwarnings("ignore:ManagerSwitcher is deprecated:DeprecationWarning")
def test_the_manager_switcher_example_runs_as_a_threshold_controller() -> None:
    oracle_primary, oracle_fallback = _manager_pair()
    oracle = _optimizer(Termination(max_fe(30))).set_surrogate_manager(oracle_primary)
    switcher = ManagerSwitcher(oracle_primary, oracle_fallback)
    applied: list[Any] = []
    loop_states = []
    for state in oracle.iterate():
        loop_states.append(state)
        chosen = switcher.switch(oracle.surrogate_manager.last_accuracy)
        applied.append(0 if chosen is oracle_primary else 1)

    primary, fallback = _manager_pair()
    controller = _optimizer(Termination(max_fe(30))).set_surrogate_manager(primary)
    controller.add_controller(
        ThresholdController(
            slot="surrogate_manager", levels=[(0.0, fallback), (0.5, primary)]
        )
    )
    in_effect: list[Any] = []
    run_states = []
    for state in controller.iterate():
        run_states.append(state)
        in_effect.append(0 if controller.surrogate_manager is primary else 1)

    _assert_shifted(in_effect, applied, 0)
    assert len(run_states) == len(loop_states)
    assert (run_states[-1].fe, run_states[-1].gen) == (
        loop_states[-1].fe,
        loop_states[-1].gen,
    )


def test_the_strategy_switcher_example_runs_as_a_threshold_controller() -> None:
    oracle_primary, _ = _manager_pair()
    ps_strategy = PreSelectionStrategy(n_candidates=8, n_select=4)
    ib_strategy = IndividualBasedStrategy(evaluation_ratio=0.1)
    switcher = StrategySwitcher(primary=ps_strategy, fallback=ib_strategy)

    oracle = _optimizer(Termination(max_fe(90)), ib_strategy).set_surrogate_manager(
        oracle_primary
    )
    applied: list[Any] = []
    loop_states = []
    for state in oracle.iterate():
        loop_states.append(state)
        chosen = switcher.switch(oracle.surrogate_manager.last_accuracy)
        applied.append(type(chosen).__name__)

    primary, _ = _manager_pair()
    controller = _optimizer(Termination(max_fe(90)), ib_strategy).set_surrogate_manager(
        primary
    )
    controller.add_controller(
        ThresholdController(
            slot="strategy", levels=[(0.0, ib_strategy), (0.56, ps_strategy)]
        )
    )
    in_effect: list[Any] = []
    run_states = []
    for state in controller.iterate():
        run_states.append(state)
        in_effect.append(type(controller.strategy).__name__)

    _assert_shifted(in_effect, applied, type(ib_strategy).__name__)
    assert len(run_states) == len(loop_states)
    assert (run_states[-1].fe, run_states[-1].gen) == (
        loop_states[-1].fe,
        loop_states[-1].gen,
    )


def test_the_three_way_switcher_example_runs_as_a_threshold_controller() -> None:
    oracle_primary, _ = _manager_pair()
    ps_strategy = PreSelectionStrategy(n_candidates=8, n_select=4)
    ib_strategy = IndividualBasedStrategy(evaluation_ratio=0.1)
    gb_strategy = GenerationBasedStrategy(gen_ctrl=1)
    switcher = StrategySwitcher(
        primary=ps_strategy,
        fallback=ib_strategy,
        mid=gb_strategy,
        threshold=0.57,
        mid_threshold=1.0,
    )

    oracle = _optimizer(Termination(max_fe(120)), ib_strategy).set_surrogate_manager(
        oracle_primary
    )
    applied: list[Any] = []
    for state in oracle.iterate():
        chosen = switcher.switch(oracle.surrogate_manager.last_accuracy)
        applied.append(type(chosen).__name__)

    primary, _ = _manager_pair()
    controller = _optimizer(
        Termination(max_fe(120)), ib_strategy
    ).set_surrogate_manager(primary)
    controller.add_controller(
        ThresholdController(
            slot="strategy",
            levels=[(0.0, ib_strategy), (0.57, gb_strategy), (1.0, ps_strategy)],
        )
    )
    in_effect: list[Any] = []
    for state in controller.iterate():
        in_effect.append(type(controller.strategy).__name__)

    _assert_shifted(in_effect, applied, type(ib_strategy).__name__)


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.filterwarnings("ignore:GenCtrlSwitcher is deprecated:DeprecationWarning")
def test_the_gen_ctrl_example_runs_as_a_smoothed_controller(
    asynchronous: bool,
) -> None:
    switcher = GenCtrlSwitcher(gm_max=5, gm_min=0)
    switcher.smoothed_error = 0.5

    def build() -> Optimizer:
        primary, _ = _manager_pair()
        optimizer = _optimizer(
            Termination(max_gen(6)), GenerationBasedStrategy(gen_ctrl=0)
        ).set_surrogate_manager(primary)
        if asynchronous:
            from saealib import (
                AsyncEvaluationScheduler,
                AsyncEvaluator,
                SerialEvaluator,
            )

            optimizer.evaluator = AsyncEvaluator(SerialEvaluator())
            optimizer.async_evaluation_scheduler = AsyncEvaluationScheduler(
                optimizer.evaluator, max_pending=2
            )
        return optimizer

    oracle = build()
    applied: list[int] = []
    for state in oracle.iterate():
        applied.append(switcher.switch(oracle.surrogate_manager.last_accuracy))
        cast(Any, oracle.strategy).gen_ctrl = applied[-1]

    controller = build()
    controller.add_controller(SmoothedController(gm_max=5, gm_min=0))
    in_effect: list[int] = []
    versions: list[int] = []
    for state in controller.iterate():
        in_effect.append(cast(Any, controller.strategy).gen_ctrl)
        versions.append(state.__dict__["_binding_version"])

    assert in_effect[0] == 0
    assert in_effect[1:] == applied[:-1]
    assert versions[-1] >= 1
