"""Focused contracts for the asynchronous runtime boundary."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from saealib.context import OptimizationState
from saealib.core.compiler import CompileContext, Compiler
from saealib.core.compiler.graph import (
    ComponentGraph,
    ComponentNode,
    ControlEdge,
    NodeRef,
)
from saealib.core.contracts import ComponentContract, StateContract
from saealib.core.graph_builder import build_component_graph
from saealib.core.runtime import (
    NodeResult,
    PollResult,
    RequestRecompile,
    RuntimeSession,
    SequentialPlan,
)
from saealib.core.state import EVALUATION_HANDLES, PENDING_EVALUATIONS
from saealib.core.state.patch import StatePatch
from saealib.exceptions import EvaluationFatalError
from saealib.execution import AsyncEvaluationScheduler, SerialEvaluator
from saealib.execution.runtime import (
    AsyncPipelineRuntime,
    _OptimizerEnvironment,
    create_runtime,
    default_runtime_registry,
    execute_strategy_step,
)
from saealib.optimizer import Optimizer
from saealib.pipeline import Pipeline, Stage
from saealib.problem import Problem


class _Stage(Stage):
    def __init__(self, name: str) -> None:
        super().__init__(name=name)

    def execute(self, state: OptimizationState) -> OptimizationState:
        return state


def _plan() -> SequentialPlan:
    plan = Compiler().compile(build_component_graph(Pipeline([_Stage("stage")])))
    return SequentialPlan.from_executable_plan(plan)


def _state(*, pending: bool = True) -> OptimizationState:
    state = object.__new__(OptimizationState)
    object.__setattr__(
        state, "pending_evaluations", {"request": object()} if pending else {}
    )
    object.__setattr__(state, "evaluation_handles", {})
    object.__setattr__(state, "async_fatal", None)
    return state


def test_reverting_a_deferred_replacement_clears_the_pending_publication() -> None:
    class Scheduler:
        max_pending = 2
        algorithm = None
        callback_manager = None

    first, second = object(), object()
    optimizer = SimpleNamespace(
        strategy=SimpleNamespace(),
        algorithm=first,
        async_evaluation_scheduler=Scheduler(),
        dispatch=lambda event: None,
    )
    active_plan = _plan()
    environment = _OptimizerEnvironment(optimizer, active_plan)
    prepared: list[dict[str, object]] = []

    def compile_current(bindings: dict[str, object]):
        prepared.append(bindings)
        return active_plan.plan, active_plan

    object.__setattr__(environment, "_compile_current_plan", compile_current)
    inflight = _state(pending=True)
    drained = _state(pending=False)

    optimizer.algorithm = second
    environment._refresh_plan_if_needed(inflight)
    assert environment._publication_deferred
    assert not environment._pending_recompile

    optimizer.algorithm = first
    environment._refresh_plan_if_needed(inflight)

    assert not environment._publication_deferred
    assert not environment._pending_recompile

    environment._refresh_plan_if_needed(drained)

    assert environment.active_version == 0
    assert prepared == []


def test_direct_step_runs_plan_when_handles_have_no_pending_evaluations() -> None:
    class SelfCompletingWork:
        def __init__(self) -> None:
            self.calls = 0
            self.unfinished = True

        def contract(self) -> ComponentContract:
            return ComponentContract()

        def execute(self, view: object) -> NodeResult:
            del view
            self.calls += 1
            self.unfinished = False
            return NodeResult(patch=StatePatch(writes={}))

        def has_async_work(self, state: OptimizationState) -> bool:
            del state
            return self.unfinished

    class Strategy:
        def __init__(self, work: SelfCompletingWork) -> None:
            self.work = work

        def build_graph(self, provider: object) -> ComponentGraph:
            del provider
            return ComponentGraph(
                nodes=(ComponentNode(component_id="work", component=self.work),),
                entry_points=(NodeRef(component_id="work"),),
            )

    problem = Problem(
        func=lambda x: np.array([float(np.sum(np.asarray(x) ** 2))]),
        dim=2,
        n_obj=1,
        direction=np.array([-1.0]),
        lb=[-5.0, -5.0],
        ub=[5.0, 5.0],
    )
    optimizer = Optimizer(problem, seed=29)
    optimizer.resolve_defaults()
    scheduler = AsyncEvaluationScheduler(SerialEvaluator(), max_pending=2)
    optimizer.set_async_evaluation_scheduler(scheduler)
    work = SelfCompletingWork()
    strategy = Strategy(work)
    optimizer.set_strategy(cast(Any, strategy))
    assert optimizer.initializer is not None
    state = optimizer.initializer.initialize(optimizer, problem)
    object.__setattr__(state, "evaluation_handles", {1: object()})

    execute_strategy_step(strategy, state, optimizer)

    assert state.pending_evaluations == {}
    assert work.calls == 1
    assert not work.unfinished


class _Environment:
    capabilities = frozenset()

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.terminated = False
        self.keep_pending_after_poll = False
        self.keep_pending_after_execute = False

    def execute(self, plan, state):
        self.calls.append("execute")
        return state

    def execute_async(self, plan, state):
        self.calls.append("execute_async")
        if not self.keep_pending_after_execute:
            object.__setattr__(state, "pending_evaluations", {})
        return state

    def reattach(self, state):
        self.calls.append("reattach")
        object.__setattr__(state, "evaluation_handles", dict(state.pending_evaluations))
        return state

    def poll(self, state):
        self.calls.append("poll")
        if not self.keep_pending_after_poll:
            object.__setattr__(state, "pending_evaluations", {})
        return PollResult(state=state, progressed=not self.keep_pending_after_poll)

    def is_terminated(self, state):
        self.calls.append("termination")
        return self.terminated

    def can_refill(self, state):
        self.calls.append("capacity")
        return True

    def dispatch(self, event):
        self.calls.append(type(event).__name__)

    def finish_generation(self, state):
        self.calls.append("finish")

    def fatal(self, state):
        self.calls.append("fatal")


def test_optimizer_environment_polls_scheduler_without_waiting() -> None:
    calls: list[bool] = []

    class Scheduler:
        def poll_result(self, state, *, wait):
            calls.append(wait)
            return PollResult(state=state, progressed=False)

    optimizer = SimpleNamespace(
        async_evaluation_scheduler=Scheduler(),
        algorithm=SimpleNamespace(allow_partial_tell=False),
        strategy=SimpleNamespace(),
    )
    environment = _OptimizerEnvironment(optimizer, _plan())

    result = environment.poll(_state())

    assert calls == [False]
    assert isinstance(result, PollResult)
    assert not result.progressed


def test_optimizer_environment_reaches_inserted_feedback_accumulator_seam():
    # The adapter is intentionally represented by insertion metadata here:
    # StageNodeAdapter self-loop execution is outside this boundary.
    enabled: list[bool] = []

    class Scheduler:
        def enable_feedback_accumulator(self):
            enabled.append(True)

    compiled = replace(
        _plan().plan,
        inserted_adapters=(SimpleNamespace(adapter_name="feedback_accumulator"),),
    )
    optimizer = SimpleNamespace(
        async_evaluation_scheduler=Scheduler(),
        algorithm=SimpleNamespace(allow_partial_tell=False),
        strategy=SimpleNamespace(),
    )

    _OptimizerEnvironment(optimizer, SequentialPlan.from_executable_plan(compiled))

    assert enabled == [True]


def test_async_environment_executes_compiled_plan_without_strategy_step() -> None:
    executed: list[OptimizationState] = []

    class AsyncStage(_Stage):
        def execute(self, state: OptimizationState) -> OptimizationState:
            executed.append(state)
            return state

    plan = Compiler().compile(
        build_component_graph(Pipeline([AsyncStage("async_stage")]))
    )
    sequential = SequentialPlan.from_executable_plan(plan)

    class PoisonStrategy:
        def step(self, state: OptimizationState, provider: object) -> OptimizationState:
            raise AssertionError("async runtime must not call strategy.step")

    optimizer = SimpleNamespace(
        async_evaluation_scheduler=object(),
        algorithm=SimpleNamespace(allow_partial_tell=False),
        strategy=PoisonStrategy(),
    )
    state = _state(pending=False)

    result = _OptimizerEnvironment(optimizer, sequential).execute_async(
        sequential, state
    )

    assert result is state
    assert executed == [state]


def test_async_recompile_waits_for_the_state_returned_by_execution() -> None:
    problem = Problem(
        lambda x: float(np.sum(np.asarray(x) ** 2)),
        1,
        1,
        np.array([-1.0]),
        lb=[-1.0],
        ub=[1.0],
    )
    state_optimizer = Optimizer(problem, seed=29)
    state_optimizer.resolve_defaults()
    assert state_optimizer.initializer is not None
    state = state_optimizer.initializer.initialize(state_optimizer, problem)

    class Request:
        def contract(self) -> ComponentContract:
            return ComponentContract(
                state=StateContract(writes=(PENDING_EVALUATIONS, EVALUATION_HANDLES))
            )

        def execute(self, view):
            del view
            return NodeResult(
                patch=StatePatch(
                    writes={
                        PENDING_EVALUATIONS: {1: object()},
                        EVALUATION_HANDLES: {1: object()},
                    }
                ),
                commands=(RequestRecompile(),),
            )

    class AsyncRequest:
        def __init__(self) -> None:
            self.proposal_flags: list[bool] = []

        def contract(self) -> ComponentContract:
            return ComponentContract()

        def execute_async(self, current, *, allow_new_proposals=True, **kwargs):
            del kwargs
            self.proposal_flags.append(allow_new_proposals)
            if allow_new_proposals:
                object.__setattr__(current, "pending_evaluations", {1: object()})
                object.__setattr__(current, "evaluation_handles", {1: object()})
            return current

    class Scheduler:
        max_pending = 1
        algorithm = None
        callback_manager = None

    asynchronous = AsyncRequest()
    graph = ComponentGraph(
        nodes=(
            ComponentNode(component_id="request", component=Request()),
            ComponentNode(component_id="async_request", component=asynchronous),
        ),
        control_edges=(
            ControlEdge(
                source=NodeRef(component_id="request"),
                target=NodeRef(component_id="async_request"),
            ),
        ),
        entry_points=(NodeRef(component_id="request"),),
    )

    class Strategy:
        def __init__(self) -> None:
            self.graph = graph

        def build_graph(self, provider):
            del provider
            return self.graph

    strategy = Strategy()
    optimizer = SimpleNamespace(
        problem=problem,
        strategy=strategy,
        algorithm=None,
        evaluator=None,
        feedback_builder=None,
        cbmanager=None,
        async_evaluation_scheduler=Scheduler(),
        dispatch=lambda event: None,
    )
    plan = Compiler().compile(
        graph,
        CompileContext(
            space=problem.space,
            problem=problem,
            offered_runtime_capabilities=default_runtime_registry.offered_capabilities(
                optimizer
            ),
        ),
    )
    sequential = SequentialPlan.from_executable_plan(plan)
    environment = _OptimizerEnvironment(optimizer, sequential)

    outcome = environment.execute_async_step(sequential, state)

    assert outcome.recompile_requested
    assert environment._last_state is outcome.state
    assert outcome.state.pending_evaluations
    assert environment.recompile(sequential) is sequential
    assert environment._publication_deferred
    assert environment._pending_recompile
    assert asynchronous.proposal_flags == []
    state = outcome.state
    object.__setattr__(state, "pending_evaluations", {})
    object.__setattr__(state, "evaluation_handles", {})
    environment._refresh_plan_if_needed(state)

    assert environment.active_version == 1
    assert not environment._pending_recompile


def test_factory_selects_async_runtime_when_scheduler_is_present() -> None:
    class Strategy:
        def build_graph(self, provider):
            raise AssertionError("existing executable plan should be reused")

        def step(self, state, provider):
            raise AssertionError("async runtime tests use the environment seam")

    optimizer = SimpleNamespace(
        executable_plan=_plan().plan,
        async_evaluation_scheduler=object(),
        strategy=Strategy(),
    )

    assert isinstance(create_runtime(optimizer), AsyncPipelineRuntime)


def test_async_runtime_reattaches_polls_and_finishes_one_generation() -> None:
    environment = _Environment()
    runtime = AsyncPipelineRuntime(environment)
    session = RuntimeSession(plan=_plan(), state=_state(), generation_open=True)

    step = runtime.advance(session)

    assert environment.calls == ["fatal", "reattach", "poll", "finish"]
    assert step.observable
    assert step.session is not None and not step.session.generation_open


def test_async_runtime_refills_pending_capacity_without_observing_step() -> None:
    environment = _Environment()
    environment.keep_pending_after_poll = True
    environment.keep_pending_after_execute = True
    runtime = AsyncPipelineRuntime(environment)
    state = _state()
    object.__setattr__(state, "evaluation_handles", dict(state.pending_evaluations))
    session = RuntimeSession(plan=_plan(), state=state, generation_open=True)

    step = runtime.advance(session)

    assert environment.calls == [
        "fatal",
        "poll",
        "termination",
        "capacity",
        "execute_async",
    ]
    assert not step.observable
    assert step.session is not None and step.session.generation_open


def test_async_runtime_drains_pending_after_termination_without_refill() -> None:
    environment = _Environment()
    environment.keep_pending_after_poll = True
    environment.terminated = True
    runtime = AsyncPipelineRuntime(environment)
    state = _state()
    object.__setattr__(state, "evaluation_handles", dict(state.pending_evaluations))
    session = RuntimeSession(plan=_plan(), state=state, generation_open=True)

    step = runtime.advance(session)

    assert environment.calls == ["fatal", "poll", "termination"]
    assert not step.observable
    assert step.session is not None and not step.session.finished


def test_async_runtime_preserves_fatal_boundary() -> None:
    class FatalEnvironment(_Environment):
        def fatal(self, state):
            self.calls.append("fatal")
            raise EvaluationFatalError("delivery failed", state)

    runtime = AsyncPipelineRuntime(FatalEnvironment())
    with pytest.raises(EvaluationFatalError, match="delivery failed"):
        runtime.advance(RuntimeSession(plan=_plan(), state=_state()))
