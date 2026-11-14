"""Column alignment performed by the default transition of a replacement."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from saealib import (
    PSO,
    CallbackManager,
    CrossoverBLXAlpha,
    GenomeGA,
    GenomeInitializer,
    LHSInitializer,
    MutationUniform,
    Optimizer,
    PymooAlgorithm,
    SPEA2Comparator,
    Termination,
    max_gen,
)
from saealib.algorithms import GA
from saealib.callback import GenerationStartEvent
from saealib.context import OptimizationState
from saealib.core.state import RUNTIME_BASE_COLUMNS
from saealib.execution import AsyncEvaluationScheduler, SerialEvaluator
from saealib.execution.runner import Runner
from saealib.execution.runtime import create_runtime
from saealib.operators import OrderCrossover, SwapMutation
from saealib.operators.selection import SequentialSelection, TruncationSelection
from saealib.problem import Problem
from saealib.space import PermutationSpace
from saealib.strategies import DirectStrategy


def _sphere(x: Any) -> np.ndarray:
    return np.array([float(np.sum(np.asarray(x) ** 2))], dtype=np.float64)


def _problem(dim: int = 2) -> Problem:
    return Problem(
        func=_sphere,
        dim=dim,
        n_obj=1,
        direction=np.array([-1.0]),
        lb=[-5.0] * dim,
        ub=[5.0] * dim,
        constraints=[],
    )


def _permutation_problem(dim: int = 6) -> Problem:
    space = PermutationSpace(dim)
    return Problem(
        func=lambda genome: np.asarray(
            [float(np.sum(np.asarray(genome) * np.arange(dim)))], dtype=np.float64
        ),
        dim=dim,
        n_obj=1,
        direction=np.array([-1.0]),
        space=space,
    )


def _ga() -> GA:
    return GA(
        crossover=CrossoverBLXAlpha(prob=0.9, alpha=0.4),
        mutation=MutationUniform(prob_var=0.1),
        parent_selection=SequentialSelection(),
        survivor_selection=TruncationSelection(),
    )


def _genome_ga() -> GenomeGA:
    return GenomeGA(
        OrderCrossover(),
        SwapMutation(),
        SequentialSelection(),
        TruncationSelection(),
    )


def _optimizer(
    algorithm: Any | None = None,
    *,
    problem: Problem | None = None,
    initializer: Any | None = None,
    comparator: Any | None = None,
    max_generation: int = 6,
    seed: int = 7,
) -> Optimizer:
    problem = _problem() if problem is None else problem
    initializer = LHSInitializer(6, 4, seed) if initializer is None else initializer
    optimizer = (
        Optimizer(problem, seed=seed)
        .set_initializer(initializer)
        .set_algorithm(_ga() if algorithm is None else algorithm)
        .set_strategy(DirectStrategy())
        .set_termination(Termination(max_gen(max_generation)))
    )
    if comparator is not None:
        optimizer.set_comparator(comparator)
    return optimizer


def _async_optimizer(algorithm: Any, *, max_generation: int = 6) -> Optimizer:
    evaluator = SerialEvaluator()
    optimizer = _optimizer(algorithm, max_generation=max_generation)
    optimizer.set_evaluator(evaluator)
    optimizer.set_async_evaluation_scheduler(
        AsyncEvaluationScheduler(evaluator, max_pending=1)
    )
    return optimizer


class _FrontOfStep:
    """Snapshot of a collection's schema and values between commit and ask."""

    def __init__(self, state: Any) -> None:
        self.schema = dict(state.population.schema)
        self.arrays = {
            name: np.array(state.population.get_array(name), copy=True)
            for name in self.schema
            if name != "id"
        }

    def name_set(self) -> frozenset[str]:
        return frozenset(self.schema)


def _observe_generation_starts(
    optimizer: Optimizer, observe: set[int]
) -> dict[int, _FrontOfStep]:
    observed: dict[int, _FrontOfStep] = {}

    def on_generation_start(event: Any) -> None:
        if event.ctx.gen in observe:
            observed[event.ctx.gen] = _FrontOfStep(event.ctx)

    callback_manager = CallbackManager()
    callback_manager.register(GenerationStartEvent, on_generation_start)
    optimizer.cbmanager = callback_manager
    return observed


def _declared_columns(component: Any, problem: Problem) -> list[Any]:
    getter = getattr(component, "get_required_attrs", None)
    return [] if getter is None else list(getter(problem))


def _assert_default_values(observed: _FrontOfStep, attribute: Any) -> None:
    expected = np.broadcast_to(
        np.asarray(attribute.default, dtype=np.float64),
        observed.arrays[attribute.name].shape,
    )
    np.testing.assert_array_equal(
        observed.arrays[attribute.name], expected, strict=True
    )


def test_swapping_ga_for_pso_adds_the_pso_columns_with_their_defaults() -> None:
    optimizer = _optimizer()
    pso = PSO()
    observed = _observe_generation_starts(optimizer, {2})
    for state in Runner(optimizer).iterate():
        if state.gen == 2:
            optimizer.set_algorithm(pso)

    front = observed[2]
    for attribute in _declared_columns(pso, optimizer.problem):
        assert attribute.name in front.schema
        _assert_default_values(front, attribute)


@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
def test_pso_to_ga_to_pso_starts_the_second_pso_from_defaults(
    asynchronous: bool,
) -> None:
    optimizer = _async_optimizer(PSO()) if asynchronous else _optimizer(PSO())
    second_pso = PSO()
    pso_columns = {
        attribute.name for attribute in _declared_columns(PSO(), optimizer.problem)
    }
    observed = _observe_generation_starts(optimizer, {2, 4})
    for state in Runner(optimizer).iterate():
        if state.gen == 2:
            optimizer.set_algorithm(_ga())
        elif state.gen == 4:
            optimizer.set_algorithm(second_pso)

    assert pso_columns
    assert not (pso_columns & observed[2].name_set())
    for attribute in _declared_columns(second_pso, optimizer.problem):
        assert attribute.name in observed[4].schema
        _assert_default_values(observed[4], attribute)


def test_swapping_ga_for_a_pymoo_algorithm_adds_the_index_column() -> None:
    from pymoo.algorithms.moo.nsga2 import NSGA2

    pymoo = PymooAlgorithm(NSGA2(pop_size=4))
    optimizer = _optimizer()
    observed = _observe_generation_starts(optimizer, {2})
    for state in Runner(optimizer).iterate():
        if state.gen == 2:
            optimizer.set_algorithm(pymoo)

    assert "pymoo_idx" in observed[2].schema
    for attribute in _declared_columns(pymoo, optimizer.problem):
        assert attribute.name in observed[2].schema


def test_replacing_a_genome_algorithm_and_strategy_keeps_the_base_columns() -> None:
    optimizer = _optimizer(
        _genome_ga(),
        problem=_permutation_problem(),
        initializer=GenomeInitializer(8, 4, seed=7),
        max_generation=4,
    )
    observed = _observe_generation_starts(optimizer, {1})
    for state in Runner(optimizer).iterate():
        if state.gen == 1:
            optimizer.set_algorithm(_genome_ga())
            optimizer.set_strategy(DirectStrategy())

    assert "x" not in observed[1].schema
    assert observed[1].name_set() == {
        "f",
        "g",
        "cv",
        "id",
        "objective_source",
    }


class _VelocityStrategy(DirectStrategy):
    def get_required_attrs(self, problem: Problem) -> list[Any]:
        return [_declared_columns(PSO(), problem)[0]]


def test_replacing_pso_with_ga_keeps_a_column_the_strategy_declares() -> None:
    optimizer = _optimizer(PSO())
    optimizer.set_strategy(_VelocityStrategy())
    generator = Runner(optimizer).iterate()
    for state in generator:
        if state.gen == 2:
            break
    optimizer.set_algorithm(_ga())

    for final in generator:
        pass

    assert final.gen == 6
    for collection in (final.population, final.archive, final.pareto_archive):
        assert "velocity" in collection.schema


def test_replacing_pso_with_ga_keeps_a_column_the_comparator_declares() -> None:
    comparator = SPEA2Comparator(direction=np.array([-1.0]))
    optimizer = _optimizer(PSO(), comparator=comparator)
    optimizer.resolve_defaults()
    initializer = optimizer.initializer
    assert initializer is not None
    state = initializer.initialize(optimizer, optimizer.problem)
    runtime = create_runtime(optimizer)
    environment = getattr(runtime, "environment")
    environment._remember_state(state)
    optimizer.set_algorithm(_ga())

    environment._refresh_plan_if_needed(state)

    assert "spea2_fitness" in state.population.schema
    assert "velocity" not in state.population.schema


def _dense_declared_columns(component: Any, problem: Problem) -> set[str]:
    return {attribute.name for attribute in _declared_columns(component, problem)}


def test_replacing_pso_with_ga_drops_only_the_declared_non_base_columns() -> None:
    pso = PSO()
    optimizer = _optimizer(pso)
    observed = _observe_generation_starts(optimizer, {2})
    states: list[Any] = []
    generator = Runner(optimizer).iterate()
    for state in generator:
        states.append(state)
        if state.gen == 2:
            break
    before = frozenset(states[-1].population.schema)
    base = frozenset(states[-1].get_state(RUNTIME_BASE_COLUMNS))
    incoming = _ga()
    declared = _dense_declared_columns(pso, optimizer.problem)
    for slot in ("algorithm", "comparator"):
        declared.update(
            _dense_declared_columns(getattr(optimizer, slot, None), optimizer.problem)
        )
    expected_drop = frozenset(declared - base - {"x"})
    optimizer.set_algorithm(incoming)
    next(generator)

    assert expected_drop
    assert expected_drop <= before
    assert not (expected_drop & observed[2].name_set())
    assert base <= observed[2].name_set()
    assert frozenset(_dense_declared_columns(incoming, optimizer.problem)) <= (
        observed[2].name_set()
    )


def _capturing_runner(optimizer: Any) -> tuple[Runner, dict[str, Any]]:
    captured: dict[str, Any] = {}

    def factory(optimizer_obj: Any) -> Any:
        runtime = create_runtime(optimizer_obj)
        captured["environment"] = getattr(runtime, "environment", None)
        return runtime

    return Runner(optimizer, runtime_factory=factory), captured


def test_checkpoint_written_after_the_replacement_resumes_to_the_same_run(
    tmp_path: Any,
) -> None:
    optimizer = _optimizer()
    runner, captured = _capturing_runner(optimizer)
    generator = runner.iterate()
    states: list[Any] = []
    for state in generator:
        states.append(state)
        if state.gen == 2:
            break
    pso = PSO()
    optimizer.set_algorithm(pso)
    captured["environment"]._refresh_plan_if_needed(states[-1])
    path = tmp_path / "state.npz"
    states[-1].save(path)

    uninterrupted = [state for state in generator]
    loaded = OptimizationState.load(path, optimizer.problem)
    resumed_optimizer = _optimizer(PSO())
    resumed = [
        state
        for state in Runner(resumed_optimizer).iterate_from(loaded)
        if state.gen > states[-1].gen
    ]

    assert len(uninterrupted) == len(resumed)
    for expected, actual in zip(uninterrupted, resumed, strict=True):
        assert expected.gen == actual.gen
        assert frozenset(expected.population.schema) == frozenset(
            actual.population.schema
        )
        for name in expected.population.schema:
            np.testing.assert_array_equal(
                expected.population.get_array(name),
                actual.population.get_array(name),
            )
