"""The constraint relaxation is written at the commit point, not per generation."""

from __future__ import annotations

from typing import Any, cast

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
    MeanPrediction,
    MutationUniform,
    Optimizer,
    Problem,
    SequentialSelection,
    SetParameter,
    SingleObjectiveComparator,
    Termination,
    TruncationSelection,
    linear_epsilon_schedule,
    max_gen,
)
from saealib.callback import CallbackManager, GenerationStartEvent
from saealib.execution.runner import Runner
from saealib.execution.runtime import create_runtime
from saealib.surrogate.rbf import RBFSurrogate
from saealib.surrogate.rbf_kernels import GaussianKernel

_SCHEDULE = linear_epsilon_schedule(eps0=0.4, n_gen=5)


def _ga() -> GA:
    return GA(
        crossover=CrossoverBLXAlpha(prob=0.9, alpha=0.4),
        mutation=MutationUniform(prob_var=0.1),
        parent_selection=SequentialSelection(),
        survivor_selection=TruncationSelection(),
    )


def _problem(handler: Any = None) -> Problem:
    return Problem(
        func=lambda x: float(np.sum(np.asarray(x, dtype=float) ** 2)),
        dim=2,
        n_obj=1,
        direction=np.array([-1.0]),
        lb=[-5.0, -5.0],
        ub=[5.0, 5.0],
        constraints=[InequalityConstraint(lambda x: 0.5 - float(np.asarray(x)[0]))],
        handler=EpsilonConstraintHandler(_SCHEDULE) if handler is None else handler,
    )


def _optimizer(
    *, surrogate: bool = False, generations: int = 5, handler: Any = None
) -> Optimizer:
    optimizer = (
        Optimizer(_problem(handler), seed=7)
        .set_initializer(LHSInitializer(6, 4, 7))
        .set_algorithm(_ga())
        .set_acquisition(MeanPrediction())
        .set_termination(Termination(max_gen(generations)))
    )
    if surrogate:
        optimizer.set_surrogate(
            RBFSurrogate(GaussianKernel()), n_neighbors=4
        ).set_strategy(GenerationBasedStrategy(gen_ctrl=1))
    return optimizer


def _instrument_sorts(optimizer: Optimizer) -> list[float]:
    """Record the relaxation every population sort sees."""
    sorts: list[float] = []
    optimizer.resolve_defaults()
    comparator = cast(Any, optimizer.comparator)
    original = comparator.sort_population

    def wrapper(population: Any) -> Any:
        sorts.append(round(float(comparator.eps_cv), 9))
        return original(population)

    comparator.sort_population = wrapper
    return sorts


@pytest.mark.parametrize("surrogate", [False, True], ids=["direct", "generation-based"])
def test_every_selection_uses_the_schedule_of_its_generation(
    surrogate: bool,
) -> None:
    optimizer = _optimizer(surrogate=surrogate)
    sorts = _instrument_sorts(optimizer)
    optimizer.add_controller(EpsilonSchedule(_SCHEDULE))

    starts: list[tuple[int, float]] = []

    def record(event: Any) -> None:
        starts.append((event.ctx.gen, round(float(event.ctx.comparator.eps_cv), 9)))

    manager = CallbackManager()
    manager.register(GenerationStartEvent, record)
    optimizer.cbmanager = manager
    optimizer.run()

    assert [epsilon for _, epsilon in starts] == pytest.approx(
        [round(float(_SCHEDULE(gen)), 9) for gen, _ in starts]
    )
    assert sorts[0] == pytest.approx(0.0)
    if surrogate:
        # A generation-based flow selects twice per step and advances gen by two.
        assert sorts[1:] == [epsilon for _, epsilon in starts for _ in (0, 1)]
    else:
        assert sorts[1:] == [epsilon for _, epsilon in starts]


def test_a_handler_without_a_schedule_keeps_the_prepared_relaxation() -> None:
    optimizer = _optimizer(generations=3)
    sorts = _instrument_sorts(optimizer)
    optimizer.run()

    assert sorts[0] == pytest.approx(0.0)
    assert sorts[1:] == pytest.approx([float(_SCHEDULE(0))] * (len(sorts) - 1))


def test_replacing_the_comparator_keeps_the_current_relaxation() -> None:
    optimizer = _optimizer(generations=3)
    capture: dict[str, Any] = {}

    def factory(built: Any) -> Any:
        runtime = create_runtime(built)
        capture["environment"] = cast(Any, runtime).environment
        return runtime

    generator = Runner(optimizer, runtime_factory=factory).iterate()
    next(generator)
    state = next(generator)
    environment = capture["environment"]
    replacement = SingleObjectiveComparator(direction=-1.0)

    optimizer.set_comparator(replacement)
    environment._refresh_plan_if_needed(state)

    assert state.comparator is replacement
    assert replacement.eps_cv == pytest.approx(float(_SCHEDULE(0)))
    generator.close()


def test_a_failure_after_publication_restores_the_relaxation_of_the_entry() -> None:
    handler = EpsilonConstraintHandler(linear_epsilon_schedule(0.4, 4))
    optimizer = _optimizer(handler=handler, generations=4)
    capture: dict[str, Any] = {}

    def factory(built: Any) -> Any:
        runtime = create_runtime(built)
        capture["environment"] = cast(Any, runtime).environment
        return runtime

    generator = Runner(optimizer, runtime_factory=factory).iterate()
    state = next(generator)
    environment = capture["environment"]
    state.comparator.eps_cv = 0.77
    optimizer.add_controller(EpsilonSchedule(lambda generation: 0.25))
    published_epsilon = float(environment._active_snapshot.values["comparator"].eps_cv)
    optimizer.set_comparator(SingleObjectiveComparator(direction=-1.0))
    original = environment._publish_scheduler

    def broken(executable: Any) -> None:
        original(executable)
        raise RuntimeError("fault injected in publish_scheduler")

    environment._publish_scheduler = broken
    with pytest.raises(RuntimeError, match="fault injected"):
        environment._refresh_plan_if_needed(state)
    environment._publish_scheduler = original

    assert float(state.comparator.eps_cv) == pytest.approx(0.77)
    assert float(environment._active_snapshot.values["comparator"].eps_cv) == (
        pytest.approx(published_epsilon)
    )
    assert environment.active_version == 0
    generator.close()


def test_a_failure_after_adopting_the_comparator_restores_both_copies() -> None:
    optimizer = _optimizer(generations=3)
    capture: dict[str, Any] = {}

    def factory(built: Any) -> Any:
        runtime = create_runtime(built)
        capture["environment"] = cast(Any, runtime).environment
        return runtime

    generator = Runner(optimizer, runtime_factory=factory).iterate()
    next(generator)
    state = next(generator)
    environment = capture["environment"]
    published = environment._active_snapshot.values["comparator"]
    state.comparator.eps_cv = 0.77
    optimizer._parameter_writes[("algorithm", "crossover.prob")] = SetParameter(
        slot="algorithm", path="crossover.prob", value=0.13
    )
    optimizer.set_algorithm(_ga())
    original = environment._publish_prepared

    def broken(*args: Any, **kwargs: Any) -> Any:
        original(*args, **kwargs)
        raise RuntimeError("fault injected after adopting the comparator")

    environment._publish_prepared = broken
    with pytest.raises(RuntimeError, match="fault injected"):
        environment._refresh_plan_if_needed(state)
    environment._publish_prepared = original

    assert float(state.comparator.eps_cv) == pytest.approx(0.77)
    assert float(published.eps_cv) == pytest.approx(0.0)
    assert cast(Any, optimizer.algorithm).crossover.prob == pytest.approx(0.9)
    assert environment.active_version == 0
    generator.close()


def test_a_resumed_run_uses_the_schedule_of_the_resumed_generation() -> None:
    optimizer = _optimizer()
    optimizer.add_controller(EpsilonSchedule(_SCHEDULE))
    generator = Runner(optimizer).iterate()
    states = [next(generator) for _ in range(3)]
    generator.close()
    resumed = _optimizer()
    resumed.add_controller(EpsilonSchedule(_SCHEDULE))
    sorts = _instrument_sorts(resumed)

    resumed.run_from(states[-1])

    assert sorts
    assert sorts[0] == pytest.approx(round(float(_SCHEDULE(states[-1].gen)), 9))


def test_a_run_without_a_constraint_handler_keeps_a_zero_relaxation() -> None:
    problem = Problem(
        func=lambda x: float(np.sum(np.asarray(x, dtype=float) ** 2)),
        dim=2,
        n_obj=1,
        direction=np.array([-1.0]),
        lb=[-5.0, -5.0],
        ub=[5.0, 5.0],
        constraints=[],
    )
    optimizer = (
        Optimizer(problem, seed=7)
        .set_initializer(LHSInitializer(6, 4, 7))
        .set_algorithm(_ga())
        .set_termination(Termination(max_gen(2)))
    )
    sorts = _instrument_sorts(optimizer)
    optimizer.run()

    assert sorts == [0.0] * len(sorts)
