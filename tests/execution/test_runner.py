"""Contracts for the state a runner returns when a run completes."""

from __future__ import annotations

from typing import cast

import numpy as np

from saealib import (
    GA,
    CrossoverBLXAlpha,
    LHSInitializer,
    MutationUniform,
    Optimizer,
    SequentialSelection,
    Termination,
    TruncationSelection,
)
from saealib.acquisition import MeanPrediction
from saealib.context import OptimizationState
from saealib.core.compiler.compiler import ExecutablePlan
from saealib.core.runtime import ExecutionRuntime, RuntimeSession, RuntimeStep
from saealib.execution import AsyncEvaluationScheduler, SerialEvaluator
from saealib.execution.initializer import Initializer
from saealib.execution.runner import Runner
from saealib.execution.runtime import create_runtime
from saealib.problem import Problem
from saealib.strategies import DirectStrategy, GenerationBasedStrategy
from saealib.surrogate.manager import LocalSurrogateManager
from saealib.surrogate.rbf import RBFSurrogate
from saealib.surrogate.rbf_kernels import GaussianKernel


class _RecordingRuntime:
    """Delegate to a real runtime and keep every state it produced."""

    def __init__(self, inner: ExecutionRuntime) -> None:
        self.inner = inner
        self.capabilities = inner.capabilities
        self.produced: list[OptimizationState] = []

    def initialize(self, plan: ExecutablePlan, state: OptimizationState):
        session = self.inner.initialize(plan, state)
        self.produced.append(session.state)
        return session

    def advance(self, session: RuntimeSession) -> RuntimeStep:
        step = self.inner.advance(session)
        self.produced.append(step.state)
        return step


def _problem() -> Problem:
    return Problem(
        func=lambda x: np.array([float(np.sum(np.asarray(x) ** 2))], dtype=np.float64),
        dim=2,
        n_obj=1,
        direction=np.array([-1.0]),
        lb=[-5.0] * 2,
        ub=[5.0] * 2,
    )


def _optimizer(strategy: str, generations: int) -> Optimizer:
    evaluator = SerialEvaluator()
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
        .set_evaluator(evaluator)
        .set_termination(Termination(lambda state: state.gen >= generations))
        .set_history(["summary", "evaluation"])
    )
    if strategy == "direct":
        return optimizer.set_strategy(DirectStrategy())
    optimizer.set_strategy(GenerationBasedStrategy(1)).set_surrogate_manager(
        LocalSurrogateManager(RBFSurrogate(GaussianKernel()))
    ).set_acquisition(MeanPrediction())
    return optimizer.set_async_evaluation_scheduler(
        AsyncEvaluationScheduler(evaluator, max_pending=1)
    )


def _recording_runner(optimizer: Optimizer) -> tuple[Runner, list[_RecordingRuntime]]:
    runtimes: list[_RecordingRuntime] = []

    def factory(current: Optimizer) -> _RecordingRuntime:
        recording = _RecordingRuntime(create_runtime(current))
        runtimes.append(recording)
        return recording

    return Runner(optimizer, runtime_factory=factory), runtimes


def test_run_returns_the_state_the_async_region_flow_ended_with() -> None:
    runner, runtimes = _recording_runner(_optimizer("generation_based", 6))

    state = runner.run()

    assert state is runtimes[0].produced[-1]
    assert state.pending_evaluations == {}
    assert state.history is not None
    channel = state.history.channel("evaluation")
    assert int(np.sum(channel["size"])) == state.fe
    assert int(channel["fe_after"][-1]) == state.fe


def test_run_from_returns_the_state_the_async_region_flow_ended_with() -> None:
    optimizer = _optimizer("generation_based", 2)
    initializer = cast(Initializer, optimizer.initializer)
    initial = initializer.initialize(optimizer, optimizer.problem)
    runner, runtimes = _recording_runner(optimizer)

    state = runner.run_from(initial)

    assert state is runtimes[0].produced[-1]
    assert state.history is not None
    assert int(np.sum(state.history.channel("evaluation")["size"])) == state.fe


def test_run_returns_the_state_the_region_less_flow_ended_with() -> None:
    runner, runtimes = _recording_runner(_optimizer("direct", 6))

    state = runner.run()

    assert state is runtimes[0].produced[-1]
    assert state.history is not None
    channel = state.history.channel("evaluation")
    assert int(np.sum(channel["size"])) == state.fe
    assert int(channel["fe_after"][-1]) == state.fe


def test_iterate_keeps_yielding_generation_boundaries_without_the_final_state() -> None:
    optimizer = _optimizer("generation_based", 6)
    runner, runtimes = _recording_runner(optimizer)

    yielded = list(runner.iterate())

    assert [(state.gen, state.fe) for state in yielded] == [(2, 6), (4, 10), (6, 14)]
    final = runtimes[0].produced[-1]
    assert (final.gen, final.fe) == (6, 15)
    assert all(state is not final for state in yielded)
