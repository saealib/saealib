"""Named state invalidation, entry repair, and multi-slot publication."""

from __future__ import annotations

import copy
import pickle
from typing import Any

import numpy as np
import pytest

from saealib import (
    PSO,
    CallbackManager,
    LHSInitializer,
    Optimizer,
    Termination,
    max_gen,
)
from saealib.acquisition import MeanPrediction
from saealib.algorithms import GA
from saealib.callback import GenerationStartEvent
from saealib.context import OptimizationState
from saealib.exceptions import ConfigurationError
from saealib.execution import AsyncEvaluationScheduler, SerialEvaluator
from saealib.execution.runner import Runner
from saealib.execution.runtime import (
    _OptimizerEnvironment,
    _plan_declared_state_keys,
    create_runtime,
)
from saealib.operators import CrossoverBLXAlpha, MutationUniform
from saealib.operators.selection import SequentialSelection, TruncationSelection
from saealib.policies.feedback import MixedFeedback
from saealib.problem import Problem
from saealib.strategies import (
    DirectStrategy,
    GenerationBasedStrategy,
    IndividualBasedStrategy,
    PreSelectionStrategy,
    SteadyStateStrategy,
)
from saealib.surrogate.manager import LocalSurrogateManager
from saealib.surrogate.rbf import RBFSurrogate
from saealib.surrogate.rbf_kernels import GaussianKernel


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


def _ga() -> GA:
    return GA(
        crossover=CrossoverBLXAlpha(prob=0.9, alpha=0.4),
        mutation=MutationUniform(prob_var=0.1),
        parent_selection=SequentialSelection(),
        survivor_selection=TruncationSelection(),
    )


def _surrogate_optimizer(
    strategy: Any,
    *,
    algorithm: Any | None = None,
    seed: int = 7,
    max_generation: int = 4,
    feedback: Any | None = None,
    asynchronous: bool = False,
) -> Optimizer:
    optimizer = (
        Optimizer(_problem(), seed=seed)
        .set_initializer(LHSInitializer(8, 6, seed))
        .set_algorithm(_ga() if algorithm is None else algorithm)
        .set_surrogate_manager(LocalSurrogateManager(RBFSurrogate(GaussianKernel())))
        .set_acquisition(MeanPrediction())
        .set_strategy(strategy)
        .set_termination(Termination(max_gen(max_generation)))
    )
    if feedback is not None:
        optimizer.set_feedback_builder(feedback)
    if asynchronous:
        evaluator = SerialEvaluator()
        optimizer.set_evaluator(evaluator)
        optimizer.set_async_evaluation_scheduler(
            AsyncEvaluationScheduler(evaluator, max_pending=1)
        )
    return optimizer


def _capturing_runner(optimizer: Any) -> tuple[Runner, dict[str, Any]]:
    captured: dict[str, Any] = {}

    def factory(optimizer_obj: Any) -> Any:
        runtime = create_runtime(optimizer_obj)
        captured["environment"] = runtime.environment
        return runtime

    return Runner(optimizer, runtime_factory=factory), captured


def _state_values(state: Any) -> dict[Any, Any]:
    return dict(state._store._values)


def _observe_states(optimizer: Optimizer) -> dict[int, dict[Any, Any]]:
    observed: dict[int, dict[Any, Any]] = {}

    def on_generation_start(event: Any) -> None:
        observed[event.ctx.gen] = _state_values(event.ctx)

    callback_manager = CallbackManager()
    callback_manager.register(GenerationStartEvent, on_generation_start)
    optimizer.cbmanager = callback_manager
    return observed


@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
def test_replacing_generation_based_with_direct_clears_stale_named_state(
    asynchronous: bool,
) -> None:
    optimizer = _surrogate_optimizer(
        GenerationBasedStrategy(1),
        feedback=MixedFeedback(),
        asynchronous=asynchronous,
    )
    observed = _observe_states(optimizer)
    for state in Runner(optimizer).iterate():
        if state.gen == 2:
            replacement_gen = state.gen
            optimizer.set_strategy(DirectStrategy(1))

    assert _stale_values_are_none(observed[replacement_gen])


def _stale_values_are_none(values: dict[Any, Any]) -> bool:
    from saealib.core.state import ACQUISITION_RESULT, SCORES, SURROGATES_PREDICTIONS

    return all(
        values.get(key) is None
        for key in (SURROGATES_PREDICTIONS, SCORES, ACQUISITION_RESULT)
    )


def _five_strategies() -> list[Any]:
    return [
        DirectStrategy(),
        GenerationBasedStrategy(1),
        IndividualBasedStrategy(0.5),
        PreSelectionStrategy(5, 2),
        SteadyStateStrategy(),
    ]


def _strategy_pairs() -> list[tuple[Any, Any]]:
    strategies = _five_strategies()
    return [(first, second) for first in strategies for second in strategies]


@pytest.mark.parametrize(("outgoing", "incoming"), _strategy_pairs())
def test_every_ordered_strategy_pair_continues_and_invalidates_the_declared_keys(
    outgoing: Any, incoming: Any
) -> None:
    optimizer = _surrogate_optimizer(outgoing, max_generation=3)
    runner, captured = _capturing_runner(optimizer)
    observed = _observe_states(optimizer)
    generator = runner.iterate()
    states: list[Any] = []
    for state in generator:
        states.append(state)
        if state.gen == 2:
            break
    replacement_gen = states[-1].gen
    outgoing_plan = captured["environment"].plan
    optimizer.set_strategy(incoming)
    next(generator)
    incoming_plan = captured["environment"].plan
    for _state in generator:
        pass

    outgoing_keys = _plan_declared_state_keys(outgoing_plan)
    incoming_keys = _plan_declared_state_keys(incoming_plan)
    assert outgoing_keys is not None
    assert incoming_keys is not None
    expected = outgoing_keys[1] & incoming_keys[0] - incoming_keys[1]
    values = observed[replacement_gen]
    for key in expected:
        assert values[key] is None


@pytest.mark.parametrize(
    "checkpoint", [False, True], ids=["same process", "checkpoint"]
)
def test_resuming_a_ga_state_with_pso_in_the_slot_matches_the_aligned_run(
    checkpoint: bool, tmp_path: Any
) -> None:
    optimizer = _surrogate_optimizer(DirectStrategy(1), max_generation=4)
    generator = Runner(optimizer).iterate()
    states: list[Any] = []
    for state in generator:
        states.append(state)
        if state.gen == 2:
            break
    ga_state = states[-1]
    if checkpoint:
        path = tmp_path / "ga_state.npz"
        ga_state.save(path)
        loaded = OptimizationState.load(path, _problem())
    else:
        loaded = copy.deepcopy(ga_state)
    optimizer.set_algorithm(PSO())
    reference = [state for state in generator]

    resumed_optimizer = _surrogate_optimizer(
        DirectStrategy(1), algorithm=PSO(), max_generation=4
    )
    resumed_optimizer.resolve_defaults()
    resumed_optimizer._compile_plan(require_initializer=False)
    runner, captured = _capturing_runner(resumed_optimizer)
    resumed = [
        state for state in runner.iterate_from(loaded) if state.gen > ga_state.gen
    ]

    assert captured["environment"].active_version == 0
    assert len(resumed) == len(reference)
    for expected, actual in zip(reference, resumed, strict=True):
        for name in expected.population.schema:
            np.testing.assert_array_equal(
                expected.population.get_array(name),
                actual.population.get_array(name),
            )


def test_a_rejected_component_in_a_pickled_slot_is_rejected_at_the_commit_point() -> (
    None
):
    optimizer = _surrogate_optimizer(DirectStrategy(1), max_generation=4)
    generator = Runner(optimizer).iterate()
    next(generator)
    state = next(generator)
    optimizer.evaluation_planner = object()  # type: ignore[assignment]
    restored = pickle.loads(pickle.dumps(optimizer))

    with pytest.raises(ConfigurationError) as error:
        for _ in Runner(restored).iterate_from(state):
            pass
    assert not isinstance(error.value, AttributeError)


def test_replacing_an_algorithm_and_a_strategy_in_one_step_publishes_once() -> None:
    optimizer = _surrogate_optimizer(GenerationBasedStrategy(1), max_generation=4)
    runner, captured = _capturing_runner(optimizer)
    generator = runner.iterate()
    states: list[Any] = []
    for state in generator:
        states.append(state)
        if state.gen == 2:
            break
    version = captured["environment"].active_version
    pso = PSO()
    direct = DirectStrategy(1)
    optimizer.set_algorithm(pso)
    optimizer.set_strategy(direct)
    next(generator)
    environment = captured["environment"]

    assert environment.active_version == version + 1
    assert environment.binding_values()["algorithm"] is pso
    assert environment.binding_values()["strategy"] is direct


@pytest.mark.parametrize("entry", ["iterate_from", "run_from"])
def test_an_entry_without_changes_publishes_nothing(
    entry: str, monkeypatch: Any
) -> None:
    optimizer = _surrogate_optimizer(DirectStrategy(1), max_generation=4)
    runner, captured = _capturing_runner(optimizer)
    generator = runner.iterate()
    states: list[Any] = []
    for state in generator:
        states.append(state)
        if state.gen == 2:
            break
    environment = captured["environment"]
    version = environment.active_version
    publications: list[int] = []
    original = _OptimizerEnvironment._publish

    def counting_publish(self: Any, *args: Any, **kwargs: Any) -> None:
        publications.append(self.active_version)
        original(self, *args, **kwargs)

    monkeypatch.setattr(_OptimizerEnvironment, "_publish", counting_publish)
    if entry == "iterate_from":
        list(Runner(optimizer).iterate_from(states[-1]))
    else:
        Runner(optimizer).run_from(states[-1])

    assert publications == []
    assert environment.active_version == version


def test_entry_repair_adds_missing_columns_without_publishing() -> None:
    optimizer = _surrogate_optimizer(DirectStrategy(1), max_generation=3)
    optimizer.resolve_defaults()
    state = optimizer.initializer.initialize(optimizer, optimizer.problem)
    optimizer.set_algorithm(PSO())
    runner, captured = _capturing_runner(optimizer)
    publications: list[int] = []
    iterator = runner.iterate_from(state)
    next(iterator)
    environment = captured["environment"]
    original = environment._publish

    def counting_publish(*args: Any, **kwargs: Any) -> None:
        publications.append(environment.active_version)
        original(*args, **kwargs)

    environment._publish = counting_publish
    next(iterator)

    assert publications == []
    assert captured["environment"].active_version == 0
    for attribute in PSO().get_required_attrs(optimizer.problem):
        assert attribute.name in state.population.schema
