"""The constraint relaxation written by a generation schedule."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from saealib import (
    GA,
    CrossoverBLXAlpha,
    EpsilonConstraintHandler,
    EpsilonSchedule,
    GenerationBasedStrategy,
    InequalityConstraint,
    LHSInitializer,
    MutationUniform,
    Optimizer,
    Problem,
    SequentialSelection,
    Termination,
    TruncationSelection,
    linear_epsilon_schedule,
    max_gen,
)
from saealib.callback import CallbackManager, GenerationStartEvent
from saealib.execution.runner import Runner
from saealib.registry import build, to_spec
from saealib.surrogate.rbf import RBFSurrogate
from saealib.surrogate.rbf_kernels import GaussianKernel


def halving(index: int) -> float:
    """Module-level schedule, so a controller spec can name it."""
    return 0.4 / (index + 1)


_SCHEDULE = linear_epsilon_schedule(eps0=0.4, n_gen=5)


def _ga() -> GA:
    return GA(
        crossover=CrossoverBLXAlpha(prob=0.9, alpha=0.4),
        mutation=MutationUniform(prob_var=0.1),
        parent_selection=SequentialSelection(),
        survivor_selection=TruncationSelection(),
    )


def _problem() -> Problem:
    return Problem(
        func=lambda x: float(np.sum(np.asarray(x, dtype=float) ** 2)),
        dim=2,
        n_obj=1,
        direction=np.array([-1.0]),
        lb=[-5.0, -5.0],
        ub=[5.0, 5.0],
        constraints=[InequalityConstraint(lambda x: 0.5 - float(np.asarray(x)[0]))],
        handler=EpsilonConstraintHandler(_SCHEDULE),
    )


def _optimizer(*, surrogate: bool = False) -> Optimizer:
    optimizer = (
        Optimizer(_problem(), seed=7)
        .set_initializer(LHSInitializer(6, 4, 7))
        .set_algorithm(_ga())
        .set_termination(Termination(max_gen(5)))
    )
    if surrogate:
        optimizer.set_surrogate(
            RBFSurrogate(GaussianKernel()), n_neighbors=4
        ).set_strategy(GenerationBasedStrategy(gen_ctrl=1))
    return optimizer


def _trajectory(optimizer: Optimizer) -> list[tuple[Any, ...]]:
    rows: list[tuple[Any, ...]] = []
    for state in Runner(optimizer).iterate():
        rows.append(
            (
                state.gen,
                state.step,
                state.fe,
                tuple(np.round(state.population.get_array("f").ravel(), 9)),
                tuple(np.round(state.population.get_array("cv").ravel(), 9)),
                tuple(np.round(state.archive.get_array("f").ravel(), 9)),
            )
        )
    return rows


@pytest.mark.parametrize("surrogate", [False, True], ids=["direct", "generation-based"])
def test_the_schedule_controller_writes_the_epsilon_of_the_step_start(
    surrogate: bool,
) -> None:
    optimizer = _optimizer(surrogate=surrogate)
    optimizer.add_controller(EpsilonSchedule(_SCHEDULE))
    observed: list[tuple[int, float]] = []

    def record(event: Any) -> None:
        observed.append((event.ctx.gen, round(float(event.ctx.comparator.eps_cv), 9)))

    manager = CallbackManager()
    manager.register(GenerationStartEvent, record)
    optimizer.cbmanager = manager
    optimizer.run()

    assert observed[0][0] == 0
    for gen, epsilon in observed:
        assert epsilon == pytest.approx(round(float(_SCHEDULE(gen)), 9))


@pytest.mark.parametrize("surrogate", [False, True], ids=["direct", "generation-based"])
def test_registering_the_schedule_controller_does_not_change_the_run(
    surrogate: bool,
) -> None:
    plain = _trajectory(_optimizer(surrogate=surrogate))
    scheduled = _optimizer(surrogate=surrogate)
    scheduled.add_controller(EpsilonSchedule(_SCHEDULE))

    assert _trajectory(scheduled) == plain


def test_the_epsilon_write_publishes_nothing() -> None:
    optimizer = _optimizer()
    optimizer.add_controller(EpsilonSchedule(_SCHEDULE))
    states = list(Runner(optimizer).iterate())

    assert {state.__dict__["_binding_version"] for state in states} == {0}
    assert optimizer._last_published_bindings is not None
    assert optimizer._last_published_bindings.version == 0
    assert optimizer._pending_parameter_writes() == {}


def test_the_schedule_controller_round_trips_through_to_spec() -> None:
    helper = EpsilonSchedule(halving)

    assert to_spec(build(to_spec(helper))) == to_spec(helper)
