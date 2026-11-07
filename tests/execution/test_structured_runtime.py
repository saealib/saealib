from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

import saealib.execution.runtime as runtime_module
from saealib.callback import GenerationEndEvent, GenerationStartEvent, RunStartEvent
from saealib.comparators import SingleObjectiveComparator
from saealib.context import EvaluationPlanState, OptimizationState
from saealib.core import (
    BranchRegion,
    ComponentContract,
    Condition,
    LoopRegion,
    RepeatRegion,
    StateContract,
    StatePatch,
    lower_pipeline,
)
from saealib.core.compiler import CompileContext, Compiler
from saealib.core.runtime import (
    NodeResult,
    NodeStatus,
    PollResult,
    RequestRecompile,
    StructuredPlan,
)
from saealib.core.state import (
    EVALUATION_UPDATES,
    EVALUATIONS_PLAN_UPDATES,
    OPTIMIZATION_STATE_INITIAL_KEYS,
    USER_DATA,
    StateKey,
)
from saealib.exceptions import ValidationError
from saealib.execution.evaluator import (
    EvaluationRequest,
    EvaluationStatus,
    PendingEvaluation,
)
from saealib.execution.runtime import (
    AsyncPipelineRuntime,
    PipelineRuntime,
    _OptimizerEnvironment,
    default_runtime_registry,
)
from saealib.pipeline import Branch, Loop, Pipeline, Repeat, Stage
from saealib.policies.evaluation import EvaluationPlan
from saealib.population import Archive, ParetoArchive, Population, PopulationAttribute
from saealib.problem import Problem
from saealib.stages import stage_component


def _state() -> OptimizationState:
    attrs = [
        PopulationAttribute(name="x", dtype=np.float64, shape=(1,)),
        PopulationAttribute(name="f", dtype=np.float64, shape=(1,)),
        PopulationAttribute(name="g", dtype=np.float64, shape=(0,)),
        PopulationAttribute(name="cv", dtype=np.float64, shape=()),
    ]
    problem = Problem(
        func=lambda x: np.array([x[0] ** 2]),
        dim=1,
        n_obj=1,
        direction=np.array([-1.0]),
        lb=[-1.0],
        ub=[1.0],
        comparator=SingleObjectiveComparator(),
    )
    population = Population(attrs, init_capacity=1)
    archive = Archive(attrs, init_capacity=1)
    pareto_archive = ParetoArchive(attrs, init_capacity=1, direction=np.array([-1.0]))
    state = OptimizationState(
        problem=problem,
        population=population,
        archive=archive,
        pareto_archive=pareto_archive,
        data={"marker": 1},
    )
    state.set_state(USER_DATA, 0)
    return state


def _compile(pipeline: Pipeline):
    return Compiler().compile_pipeline(
        pipeline,
        CompileContext(
            initial_state_keys=frozenset((*OPTIMIZATION_STATE_INITIAL_KEYS, USER_DATA))
        ),
    )


class _Increment:
    name = "increment"

    def contract(self) -> ComponentContract:
        return ComponentContract(
            state=StateContract(reads=(USER_DATA,), writes=(USER_DATA,))
        )

    def execute(self, view):
        return StatePatch(writes={USER_DATA: view.get(USER_DATA) + 1})


class _UntilFive(Condition):
    def contract(self) -> StateContract:
        return StateContract(reads=(USER_DATA,))

    def evaluate(self, view) -> bool:
        return view.get(USER_DATA) >= 5


class _StateCondition(Condition):
    def __init__(self, value: bool) -> None:
        self.value = value

    def contract(self) -> StateContract:
        return StateContract()

    def evaluate(self, view) -> bool:
        del view
        return self.value


class _BlockOnce(_Increment):
    def __init__(self) -> None:
        self.calls = 0

    def execute(self, view):
        self.calls += 1
        patch = super().execute(view)
        if self.calls == 1:
            return NodeResult(patch=patch, status=NodeStatus.BLOCKED)
        return patch


class _StructuredEnvironment:
    capabilities = frozenset()

    def __init__(self) -> None:
        self.dispatched: list[object] = []
        self.finished_generations = 0

    def dispatch(self, event) -> None:
        self.dispatched.append(event)

    def finish_generation(self, state) -> None:
        self.finished_generations += 1
        self.dispatch(GenerationEndEvent(ctx=state))

    def is_terminated(self, state) -> bool:
        del state
        return False

    def fatal(self, state) -> None:
        del state


class _StructuredAsyncPollingEnvironment(_StructuredEnvironment):
    def __init__(self, terminated: bool, progressed: bool = False) -> None:
        super().__init__()
        self.terminated = terminated
        self.progressed = progressed

    def poll(self, state):
        return PollResult(state=state, progressed=self.progressed)

    def is_terminated(self, state) -> bool:
        del state
        return self.terminated


class _StatusOnce(_Increment):
    def __init__(self, status: NodeStatus) -> None:
        self.status = status
        self.calls = 0

    def execute(self, view):
        self.calls += 1
        patch = super().execute(view)
        status = self.status if self.calls == 1 else NodeStatus.COMPLETED
        return NodeResult(patch=patch, status=status)


class _RequestRecompile(_Increment):
    def execute(self, view):
        return NodeResult(
            patch=super().execute(view),
            commands=(RequestRecompile(),),
            status=NodeStatus.BLOCKED,
        )


class _CompletedRequestRecompile(_Increment):
    def execute(self, view):
        return NodeResult(
            patch=super().execute(view),
            commands=(RequestRecompile(),),
        )


class _RequestRecompileOnce(_Increment):
    def __init__(self) -> None:
        self.calls = 0

    def execute(self, view):
        self.calls += 1
        commands = (RequestRecompile(),) if self.calls == 1 else ()
        return NodeResult(patch=super().execute(view), commands=commands)


class _RequestRecompileWithPending(_Increment):
    def __init__(self, state: OptimizationState) -> None:
        self.state = state

    def execute(self, view):
        object.__setattr__(self.state, "pending_evaluations", {1: object()})
        return NodeResult(
            patch=super().execute(view),
            commands=(RequestRecompile(),),
        )


class _AsyncBlockOnce(_Increment):
    def __init__(self) -> None:
        self.calls = 0

    def execute_async(self, view):
        self.calls += 1
        patch = super().execute(view)
        status = NodeStatus.BLOCKED if self.calls == 1 else NodeStatus.COMPLETED
        return NodeResult(patch=patch, status=status)


class _AsyncRequiredWithoutDriver(_Increment):
    requires_async_execution = True


class _AsyncBranchLeaf(_Increment):
    def __init__(self) -> None:
        self.calls = 0

    def execute_async(self, view):
        self.calls += 1
        return super().execute(view)


class _AsyncProtocolDriver:
    name = "async_protocol_driver"
    async_protocol = "evaluation"
    async_protocol_role = "driver"

    def contract(self) -> ComponentContract:
        return ComponentContract()

    def execute(self, view):
        del view
        return StatePatch(writes={})

    def execute_async(self, view):
        return self.execute(view)


class _AsyncProtocolWait:
    name = "async_protocol_wait"

    def contract(self) -> ComponentContract:
        return ComponentContract(
            state=StateContract(
                reads=(EVALUATIONS_PLAN_UPDATES,), writes=(EVALUATION_UPDATES,)
            )
        )

    def execute(self, view):
        raise AssertionError("async protocol wait must be scheduler-owned")


class _AsyncProtocolEnd:
    name = "async_protocol_end"
    async_protocol = "evaluation"
    async_protocol_role = "end"

    def contract(self) -> ComponentContract:
        return ComponentContract()

    def execute(self, view):
        raise AssertionError("async protocol end must be scheduler-owned")


class _AsyncProtocolTail(_Increment):
    name = "async_protocol_tail"

    def __init__(self) -> None:
        self.calls = 0

    def execute(self, view):
        self.calls += 1
        return super().execute(view)


def test_structured_plan_executes_repeat_loop_and_branch() -> None:
    state = _state()
    runtime = PipelineRuntime()
    pipeline = Pipeline(
        [
            Repeat(_Increment(), 3, name="repeat"),
            Loop(_Increment(), until=_UntilFive(), name="loop"),
            Branch(
                _StateCondition(True),
                then=_Increment(),
                else_=_Increment(),
                name="branch",
            ),
        ]
    )
    graph = lower_pipeline(pipeline)
    assert [type(region.region) for region in graph.regions] == [
        RepeatRegion,
        LoopRegion,
        BranchRegion,
    ]

    session = runtime.initialize(_compile(pipeline), state)
    step = runtime.advance(session)

    assert state.get_state(USER_DATA) == 6
    assert step.finished
    assert len(step.node_results) == 6


def test_structured_plan_resumes_blocked_leaf_from_saved_frame() -> None:
    component = _BlockOnce()
    runtime = PipelineRuntime()
    session = runtime.initialize(_compile(Pipeline([component])), _state())

    first = runtime.advance(session)
    assert not first.finished
    assert first.node_results[0].status is NodeStatus.BLOCKED
    assert first.session is not None and first.session.frames

    second = runtime.advance(first.session)
    assert second.finished
    assert component.calls == 2
    assert second.state.get_state(USER_DATA) == 2


def test_async_structured_repeat_resumes_the_same_iteration() -> None:
    component = _AsyncBlockOnce()
    runtime = AsyncPipelineRuntime()
    session = runtime.initialize(
        _compile(Pipeline([Repeat(component, count=2)])), _state()
    )

    first = runtime.advance(session)
    assert first.node_results[0].status is NodeStatus.BLOCKED
    assert first.session is not None
    assert first.session.frames[-1].operation_index == 0

    second = runtime.advance(first.session)

    assert second.finished
    assert component.calls == 3
    assert second.state.get_state(USER_DATA) == 3


def test_async_structured_branch_executes_only_the_selected_leaf() -> None:
    selected = _AsyncBranchLeaf()
    skipped = _AsyncBranchLeaf()
    runtime = AsyncPipelineRuntime()
    session = runtime.initialize(
        _compile(
            Pipeline(
                [
                    Branch(
                        _StateCondition(True),
                        then=selected,
                        else_=skipped,
                    )
                ]
            )
        ),
        _state(),
    )

    step = runtime.advance(session)

    assert step.finished
    assert selected.calls == 1
    assert skipped.calls == 0


def test_async_structured_recompile_waits_for_pending_evaluations() -> None:
    class Environment(_StructuredAsyncPollingEnvironment):
        def __init__(self) -> None:
            super().__init__(terminated=False, progressed=True)
            self.recompiled = 0

        def recompile(self, plan):
            self.recompiled += 1
            return plan

        def poll(self, state):
            object.__setattr__(state, "pending_evaluations", {})
            return PollResult(state=state, progressed=True)

    environment = Environment()
    component = _RequestRecompileOnce()
    state = _state().replace(pending_evaluations={1: object()})
    runtime = AsyncPipelineRuntime(environment=environment)
    first = runtime.advance(runtime.initialize(_compile(Pipeline([component])), state))

    assert environment.recompiled == 1
    assert environment.finished_generations == 1
    assert first.session is not None
    assert not first.session.generation_open
    assert first.session.frames == ()

    second = runtime.advance(first.session)

    assert environment.recompiled == 1
    assert environment.finished_generations == 2
    assert [type(event) for event in environment.dispatched] == [
        RunStartEvent,
        GenerationEndEvent,
        GenerationStartEvent,
        GenerationEndEvent,
    ]
    assert second.session is not None and second.session.frames == ()


def test_async_structured_refuses_recompile_when_leaf_creates_pending_evaluation() -> (
    None
):
    class Environment(_StructuredEnvironment):
        def __init__(self) -> None:
            super().__init__()
            self.recompiled = 0

        def recompile(self, plan):
            self.recompiled += 1
            return plan

    environment = Environment()
    state = _state()
    component = _RequestRecompileWithPending(state)
    runtime = AsyncPipelineRuntime(environment=environment)
    step = runtime.advance(runtime.initialize(_compile(Pipeline([component])), state))

    assert environment.recompiled == 0
    assert environment.finished_generations == 0
    assert not any(
        isinstance(event, GenerationEndEvent) for event in environment.dispatched
    )
    assert step.refused_commands == (RequestRecompile(),)
    assert step.session is not None and step.session.generation_open


def test_async_structured_request_is_published_after_pending_work_drains(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _state()
    original_prepare = runtime_module._prepare_plan
    prepared: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def record_prepare(*args, **kwargs):
        prepared.append((args, kwargs))
        return original_prepare(*args, **kwargs)

    monkeypatch.setattr(runtime_module, "_prepare_plan", record_prepare)

    class RequestOnce:
        def __init__(self) -> None:
            self.calls = 0

        def contract(self) -> ComponentContract:
            return ComponentContract()

        def execute(self, view):
            del view
            self.calls += 1
            if self.calls == 1:
                object.__setattr__(state, "pending_evaluations", {1: object()})
                object.__setattr__(state, "evaluation_handles", {1: object()})
                return NodeResult(
                    patch=StatePatch(writes={}),
                    commands=(RequestRecompile(),),
                )
            return NodeResult(patch=StatePatch(writes={}))

    component = RequestOnce()

    class Strategy:
        def __init__(self) -> None:
            self.component = component

        def build_graph(self, provider):
            del provider
            return lower_pipeline(Pipeline([self.component]))

    class Scheduler:
        max_pending = 1
        algorithm = None
        callback_manager = None

        def poll_result(self, current, *, wait=False):
            del wait
            return PollResult(
                state=current.replace(pending_evaluations={}, evaluation_handles={}),
                progressed=True,
            )

    strategy = Strategy()
    optimizer = SimpleNamespace(
        problem=state.problem,
        strategy=strategy,
        algorithm=None,
        evaluator=None,
        feedback_builder=None,
        cbmanager=None,
        async_evaluation_scheduler=Scheduler(),
        termination=SimpleNamespace(is_terminated=lambda current: False),
        dispatch=lambda event: None,
        surrogate_manager=None,
    )
    graph = strategy.build_graph(optimizer)
    plan = Compiler().compile(
        graph,
        CompileContext(
            space=state.problem.space,
            problem=state.problem,
            offered_runtime_capabilities=default_runtime_registry.offered_capabilities(
                optimizer
            ),
            initial_state_keys=OPTIMIZATION_STATE_INITIAL_KEYS,
        ),
    )
    environment = _OptimizerEnvironment(
        optimizer, StructuredPlan.from_executable_plan(plan)
    )
    runtime = AsyncPipelineRuntime(environment)

    step = runtime.advance(runtime.initialize(plan, state))

    assert step.refused_commands == (RequestRecompile(),)
    assert environment.recompile(environment.plan) is environment.plan
    assert environment.active_version == 0
    assert environment._pending_recompile
    assert environment._publication_deferred
    assert prepared == []
    assert not AsyncPipelineRuntime._async_leaf_kwargs(environment)[
        "allow_new_proposals"
    ]

    object.__setattr__(state, "pending_evaluations", {})
    object.__setattr__(state, "evaluation_handles", {})
    environment._refresh_plan_if_needed(state)

    assert environment.active_version == 1
    assert not environment._pending_recompile
    assert not environment._publication_deferred
    assert len(prepared) == 1
    args, kwargs = prepared[0]
    assert args[0] is strategy
    assert args[1].binding_values()["strategy"] is strategy
    assert kwargs["initial_state_keys"] == OPTIMIZATION_STATE_INITIAL_KEYS
    assert kwargs["offered_runtime_capabilities"] == (
        default_runtime_registry.offered_capabilities(optimizer)
    )
    assert kwargs["_verification_enabled"] is True


def test_async_structured_requires_a_driver_without_sync_fallback() -> None:
    runtime = AsyncPipelineRuntime()
    session = runtime.initialize(
        _compile(Pipeline([_AsyncRequiredWithoutDriver()])), _state()
    )

    with pytest.raises(ValidationError, match="async execution driver"):
        runtime.advance(session)


def test_async_protocol_completion_resumes_after_protocol_end() -> None:
    request = EvaluationRequest(
        np.int64(1), np.empty(0, dtype=np.int64), np.empty((0, 1))
    )
    state = _state().replace(
        evaluation_plan=EvaluationPlan((request,)),
        evaluation_plan_state=EvaluationPlanState(submitted=(1,), completed=(1,)),
    )
    tail = _AsyncProtocolTail()
    pipeline = Pipeline(
        [
            _AsyncProtocolDriver(),
            _AsyncProtocolWait(),
            _AsyncProtocolEnd(),
            tail,
        ]
    )
    runtime = AsyncPipelineRuntime()
    step = runtime.advance(runtime.initialize(_compile(pipeline), state))

    assert step.finished
    assert step.executed_node_ids == (
        "async_protocol_driver",
        "async_protocol_tail",
    )
    assert tail.calls == 1


def test_structured_external_environment_closes_generation_not_runtime_session() -> (
    None
):
    environment = _StructuredEnvironment()
    runtime = PipelineRuntime(environment=environment)
    session = runtime.initialize(_compile(Pipeline([_Increment()])), _state())

    step = runtime.advance(session)

    assert step.finished is False
    assert step.observable is True
    assert step.session is not None
    assert step.session.finished is False
    assert step.session.generation_open is False
    assert environment.finished_generations == 1
    assert [type(event) for event in environment.dispatched] == [
        RunStartEvent,
        GenerationStartEvent,
        GenerationEndEvent,
    ]


@pytest.mark.parametrize("status", [NodeStatus.BLOCKED, NodeStatus.RUNNING])
def test_structured_external_environment_keeps_generation_open_for_resume(
    status: NodeStatus,
) -> None:
    component = _StatusOnce(status)
    environment = _StructuredEnvironment()
    runtime = PipelineRuntime(environment=environment)
    session = runtime.initialize(_compile(Pipeline([component])), _state())

    first = runtime.advance(session)

    assert first.finished is False
    assert first.observable is False
    assert first.session is not None
    assert first.session.finished is False
    assert first.session.generation_open is True
    assert environment.finished_generations == 0
    assert first.node_results[0].status is status

    second = runtime.advance(first.session)

    assert second.finished is False
    assert second.observable is True
    assert second.session is not None
    assert second.session.finished is False
    assert second.session.generation_open is False
    assert environment.finished_generations == 1
    assert component.calls == 2


def test_structured_external_environment_rejects_failed_node() -> None:
    environment = _StructuredEnvironment()
    runtime = PipelineRuntime(environment=environment)
    session = runtime.initialize(
        _compile(Pipeline([_StatusOnce(NodeStatus.FAILED)])), _state()
    )

    with pytest.raises(
        ValidationError, match="structured runtime node reported FAILED"
    ):
        runtime.advance(session)

    assert environment.finished_generations == 0


def test_structured_runtime_refuses_recompile_with_active_frame() -> None:
    plan = _compile(Pipeline([Repeat(_RequestRecompile(), count=2)]))

    step = PipelineRuntime().advance(PipelineRuntime().initialize(plan, _state()))

    assert step.refused_commands == (RequestRecompile(),)
    assert step.session is not None and step.session.frames


def test_structured_runtime_recompiles_at_root_and_discards_frames() -> None:
    class Environment(_StructuredEnvironment):
        def __init__(self) -> None:
            super().__init__()
            self.recompiled: list[object] = []

        def recompile(self, plan):
            self.recompiled.append(plan)
            return plan

    environment = Environment()
    runtime = PipelineRuntime(environment=environment)
    session = runtime.initialize(
        _compile(Pipeline([_CompletedRequestRecompile()])), _state()
    )

    step = runtime.advance(session)

    assert len(environment.recompiled) == 1
    assert step.refused_commands == ()
    assert step.session is not None and step.session.frames == ()


def test_structured_recompile_closes_generation_before_next_generation_start() -> None:
    class Environment(_StructuredEnvironment):
        def __init__(self) -> None:
            super().__init__()
            self.recompiled = 0

        def recompile(self, plan):
            self.recompiled += 1
            return plan

    environment = Environment()
    component = _RequestRecompileOnce()
    runtime = PipelineRuntime(environment=environment)
    first = runtime.advance(
        runtime.initialize(_compile(Pipeline([component])), _state())
    )

    assert environment.recompiled == 1
    assert environment.finished_generations == 1
    assert first.observable is True
    assert [type(event) for event in environment.dispatched] == [
        RunStartEvent,
        GenerationStartEvent,
        GenerationEndEvent,
    ]
    assert first.session is not None
    assert not first.session.generation_open
    assert first.session.frames == ()

    second = runtime.advance(first.session)

    assert environment.recompiled == 1
    assert environment.finished_generations == 2
    assert [type(event) for event in environment.dispatched] == [
        RunStartEvent,
        GenerationStartEvent,
        GenerationEndEvent,
        GenerationStartEvent,
        GenerationEndEvent,
    ]
    assert second.session is not None and second.session.frames == ()


def test_structured_runtime_refuses_recompile_with_pending_evaluation() -> None:
    class Environment(_StructuredEnvironment):
        def __init__(self) -> None:
            super().__init__()
            self.recompiled = 0

        def recompile(self, plan):
            self.recompiled += 1
            return plan

    environment = Environment()
    state = _state().replace(pending_evaluations={1: object()})
    runtime = PipelineRuntime(environment=environment)
    step = runtime.advance(
        runtime.initialize(_compile(Pipeline([_CompletedRequestRecompile()])), state)
    )

    assert environment.recompiled == 0
    assert environment.finished_generations == 0
    assert step.refused_commands == (RequestRecompile(),)
    assert step.session is not None and step.session.generation_open


def test_structured_runtime_refuses_recompile_before_root_completion() -> None:
    class SecondIncrement(_Increment):
        name = "second_increment"

    class Environment(_StructuredEnvironment):
        def __init__(self) -> None:
            super().__init__()
            self.recompiled = 0

        def recompile(self, plan):
            self.recompiled += 1
            return plan

    environment = Environment()
    runtime = PipelineRuntime(environment=environment)
    session = runtime.initialize(
        _compile(Pipeline([_CompletedRequestRecompile(), SecondIncrement()])), _state()
    )

    step = runtime.advance(session)

    assert environment.recompiled == 0
    assert step.refused_commands == (RequestRecompile(),)
    assert step.executed_node_ids == ("increment", "second_increment")


@pytest.mark.parametrize("count", [0, 2])
def test_structured_runtime_repeat_count(count: int) -> None:
    component = _StatusOnce(NodeStatus.COMPLETED)
    step = PipelineRuntime().advance(
        PipelineRuntime().initialize(
            _compile(Pipeline([Repeat(component, count=count)])), _state()
        )
    )

    assert component.calls == count
    assert step.finished


def test_async_runtime_accepts_structured_plan_at_initialization() -> None:
    plan = _compile(Pipeline([_Increment()]))
    runtime = AsyncPipelineRuntime()

    session = runtime.initialize(plan, _state())

    step = runtime.advance(session)

    assert step.finished
    assert step.state.get_state(USER_DATA) == 1


@pytest.mark.parametrize("terminated", [False, True])
@pytest.mark.parametrize("progressed, sleeps", [(False, True), (True, False)])
def test_structured_async_polling_uses_explicit_progress(
    terminated: bool,
    progressed: bool,
    sleeps: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = _StructuredAsyncPollingEnvironment(terminated, progressed)
    request = EvaluationRequest(np.int64(1), np.array([np.int64(1)]), np.empty((1, 1)))
    pending = PendingEvaluation(
        request=request,
        status=EvaluationStatus.PENDING,
        applied_candidate_ids=np.array([np.int64(1)]),
    )
    state = _state().replace(pending_evaluations={1: pending})
    runtime = AsyncPipelineRuntime(environment=environment)
    sleep = Mock()
    monkeypatch.setattr(runtime_module.time, "sleep", sleep)

    step = runtime.advance(
        runtime.initialize(_compile(Pipeline([_Increment()])), state)
    )

    assert step.state is state
    if sleeps:
        sleep.assert_called_once_with(0.001)
    else:
        sleep.assert_not_called()


class _LegacyStage(Stage):
    name = "legacy"

    def execute(self, state):
        return state


def test_structured_plan_rejects_optimization_state_stage_boundary() -> None:
    with pytest.raises(ValidationError, match="stage_component"):
        _compile(Pipeline([_LegacyStage()]))


def test_stage_with_declared_writes_still_rejects_undeclared_writes() -> None:
    extra = StateKey(namespace="user", name="extra", schema_version=1)

    class DeclaredWriter(Stage):
        name = "declared_writer"

        def contract(self) -> ComponentContract:
            return ComponentContract(state=StateContract(writes=(USER_DATA,)))

        def execute(self, state: OptimizationState) -> OptimizationState:
            state.set_state(extra, 1)
            return state

    plan = _compile(Pipeline([stage_component(DeclaredWriter())]))
    state = _state()

    session = PipelineRuntime().initialize(plan, state)
    with pytest.raises(ValidationError, match="undeclared state write"):
        PipelineRuntime().advance(session)


def test_opaque_stage_writes_any_key_without_rejection() -> None:
    extra = StateKey(namespace="user", name="extra", schema_version=1)

    class OpaqueWriter(Stage):
        name = "opaque_writer"

        def execute(self, state: OptimizationState) -> OptimizationState:
            state.set_state(extra, 1)
            return state

    pipeline = Pipeline([stage_component(OpaqueWriter())])
    plan = Compiler().compile_pipeline(
        pipeline,
        CompileContext(
            initial_state_keys=frozenset((*OPTIMIZATION_STATE_INITIAL_KEYS, USER_DATA))
        ),
    )
    assert [
        diagnostic.code
        for diagnostic in plan.diagnostics
        if diagnostic.code == "opaque_state_effects"
    ]
    assert not [
        diagnostic
        for diagnostic in plan.diagnostics
        if diagnostic.severity.name == "ERROR"
    ]

    session = PipelineRuntime().initialize(plan, _state())
    step = PipelineRuntime().advance(session)

    assert step.state._store.get(extra) == 1
