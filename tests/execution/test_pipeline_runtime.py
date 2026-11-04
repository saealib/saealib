"""Focused tests for the synchronous plan bridge."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import Any, cast

import pytest

from saealib import (
    GA,
    CrossoverBLXAlpha,
    MutationUniform,
    SequentialSelection,
    TruncationSelection,
)
from saealib.context import OptimizationState
from saealib.core.compiler import (
    CompileContext,
    Compiler,
    ComponentGraph,
    ComponentNode,
    ControlEdge,
    DataEdge,
    ExecutablePlan,
    StateBinding,
)
from saealib.core.compiler.adapters import Adapter, AdapterComponent
from saealib.core.compiler.graph import NodeRef
from saealib.core.compiler.schema_rules import _FreshenedComponent
from saealib.core.contracts import ComponentContract, DataSpec, StateContract
from saealib.core.graph_builder import (
    StageContractNodeAdapter,
    StageNodeAdapter,
    build_component_graph,
    cached_execution_target,
)
from saealib.core.runtime import (
    NodeResult,
    NodeStatus,
    RequestCheckpoint,
    RequestRecompile,
    RequestTermination,
    SequentialPlan,
)
from saealib.core.state import StateKey, StatePatch, StateStore
from saealib.exceptions import ValidationError
from saealib.execution.evaluator import SerialEvaluator
from saealib.execution.runtime import PipelineRuntime, _OptimizerEnvironment
from saealib.optimizer import ComponentProvider
from saealib.pipeline import Pipeline, Stage
from saealib.policies.evaluation import EvaluateAll
from saealib.stages import AsyncEvaluationSubmitStage
from saealib.strategies.base import OptimizationStrategy, build_runtime_neutral_graph
from saealib.strategies.direct import DirectStrategy


class _Stage(Stage):
    def __init__(self, name: str) -> None:
        super().__init__(name=name)
        self.seen: list[OptimizationState] = []

    def execute(self, state: OptimizationState) -> OptimizationState:
        self.seen.append(state)
        next_state = object.__new__(OptimizationState)
        object.__setattr__(
            next_state, "marker", f"{self.name}({getattr(state, 'marker', '')})"
        )
        return next_state


def _state() -> OptimizationState:
    state = object.__new__(OptimizationState)
    object.__setattr__(state, "marker", "start")
    return state


def _compiled(stages: list[_Stage]):
    graph = build_component_graph(Pipeline(cast(list[Stage], stages)))
    return Compiler().compile(graph)


class _GenericNode:
    def __init__(
        self,
        contract: ComponentContract,
        execute: Callable[[Any], NodeResult | StatePatch],
    ) -> None:
        self._contract = contract
        self._execute = execute

    def contract(self) -> ComponentContract:
        return self._contract

    def execute(self, view: Any) -> NodeResult | StatePatch:
        return self._execute(view)


def _generic_plan(
    nodes: tuple[_GenericNode, ...],
    *,
    data_edges: tuple[DataEdge, ...] = (),
    state_bindings: tuple[StateBinding, ...] = (),
) -> ExecutablePlan:
    component_nodes = tuple(
        ComponentNode(component_id=f"node_{index}", component=node)
        for index, node in enumerate(nodes)
    )
    graph = ComponentGraph(
        nodes=component_nodes,
        data_edges=data_edges,
        state_bindings=state_bindings,
        entry_points=(NodeRef(component_id=component_nodes[0].component_id),),
    )
    return ExecutablePlan(
        graph=graph,
        diagnostics=(),
        required_runtime_capabilities=frozenset(),
        active_rule_namespaces=frozenset(),
        active_rule_names=(),
        contract_snapshots=tuple(
            (node.component_id, node.contract) for node in component_nodes
        ),
    )


def _generic_state(values: dict[StateKey, object]) -> OptimizationState:
    state = _state()
    object.__setattr__(state, "_store", StateStore(values))
    return state


def test_real_default_strategy_graph_compiles_to_stage_order() -> None:
    class Provider:
        algorithm = GA(
            crossover=CrossoverBLXAlpha(prob=0.9, alpha=0.4),
            mutation=MutationUniform(prob_var=0.1),
            parent_selection=SequentialSelection(),
            survivor_selection=TruncationSelection(),
        )
        evaluator = SerialEvaluator()
        cbmanager = None
        async_evaluation_scheduler = None
        evaluation_planner = None
        feedback_builder = None
        feedback_builder_explicit = False

    plan = Compiler().compile(
        DirectStrategy().build_graph(cast(ComponentProvider, Provider()))
    )
    ordered = SequentialPlan.from_executable_plan(plan)

    assert ordered.nodes
    from saealib.core.compiler.regions import RegionNode
    from saealib.core.compiler.structured import StructuredGraph

    assert isinstance(plan.graph, StructuredGraph)
    assert not any(isinstance(op, RegionNode) for op in plan.graph.operations)
    assert [node.component_id for node in ordered.execution_nodes] == [
        "count_generation",
        "ask",
        "evaluation_plan",
        "evaluation_submit",
        "evaluation_collect",
        "evaluation_apply",
        "archive_update",
        "feedback",
        "tell",
        "evaluation_acknowledge",
    ]
    operations = cast(Any, plan.graph.operations)
    assert [node.component_id for node in operations] == [
        node.component_id for node in ordered.execution_nodes
    ]
    assert plan.graph.node_by_id("ask") is plan.graph.nodes[1]
    assert plan.graph.node_by_id("tell") is plan.graph.nodes[8]
    from saealib.core.runtime import StructuredPlan

    structured = StructuredPlan.from_executable_plan(plan)
    targets = getattr(structured, "_execute_targets")

    def _outermost(component: Any) -> Any:
        while getattr(component, "_component", None) is not None:
            component = component._component
        return component

    # every structured node executes through its own Stage adapter (possibly
    # behind schema-freshening wrappers), so the Stage's declared contract
    # and execution boundary stay in force
    assert set(targets) == {node.component_id for node in operations}
    assert all(
        _outermost(targets[node.component_id].__self__) is _outermost(node.component)
        for node in operations
    )


def test_runtime_threads_stage_return_values_and_reports_order_without_events() -> None:
    stages = [_Stage("a"), _Stage("b"), _Stage("c")]
    session = PipelineRuntime().initialize(_compiled(stages), _state())

    step = PipelineRuntime().advance(session)

    assert step.observable
    assert not step.finished
    assert [result.status for result in step.node_results] == [NodeStatus.COMPLETED] * 3
    assert step.executed_node_ids == ("a", "b", "c")
    assert all(result.events == () for result in step.node_results)
    assert stages[0].seen[0].marker == "start"
    assert stages[1].seen[0].marker == "a(start)"
    assert stages[2].seen[0].marker == "b(a(start))"
    assert step.session is not None and step.session.state is step.state


def test_generic_node_result_uses_state_binding_and_executes_termination() -> None:
    abstract = StateKey(namespace="user", name="value", schema_version=1)
    bound = StateKey(namespace="user", name="value_for_node", schema_version=1)

    def execute(view: Any) -> NodeResult:
        assert view.get(abstract) == 1
        return NodeResult(
            patch=StatePatch(writes={abstract: 2}),
            commands=(RequestTermination(reason="done"),),
        )

    node = _GenericNode(
        ComponentContract(
            state=StateContract(reads=(abstract,), writes=(abstract,)),
        ),
        execute,
    )
    plan = _generic_plan(
        (node,),
        state_bindings=(
            StateBinding(node=NodeRef(component_id="node_0"), state_key=bound),
        ),
    )

    step = PipelineRuntime().advance(
        PipelineRuntime().initialize(plan, _generic_state({bound: 1}))
    )

    assert step.finished
    assert step.executed_node_ids == ("node_0",)
    assert step.node_results[0].commands == (RequestTermination(reason="done"),)
    assert step.refused_commands == ()
    assert step.state._store.get(bound) == 2


def test_sequential_graph_node_receives_node_scoped_services() -> None:
    service = object()

    def execute(view: Any) -> NodeResult:
        assert view.context.compiled_service("ScopedService") is service
        return NodeResult(patch=StatePatch(writes={}))

    executable = _generic_plan((_GenericNode(ComponentContract(), execute),))
    node = replace(
        executable.graph.nodes[0], resolved_services={"ScopedService": service}
    )
    executable = replace(executable, graph=replace(executable.graph, nodes=(node,)))

    step = PipelineRuntime().advance(
        PipelineRuntime().initialize(executable, _generic_state({}))
    )

    assert step.executed_node_ids == ("node_0",)


def test_compiler_to_runtime_executes_state_bound_component() -> None:
    abstract = StateKey(namespace="user", name="value", schema_version=1)
    bound = StateKey(namespace="user", name="value_for_node", schema_version=1)
    node = _GenericNode(
        ComponentContract(
            state=StateContract(reads=(abstract,), writes=(abstract,)),
        ),
        lambda view: StatePatch(writes={abstract: view.get(abstract) + 1}),
    )
    graph = ComponentGraph(
        nodes=(ComponentNode(component_id="node", component=node),),
        state_bindings=(
            StateBinding(node=NodeRef(component_id="node"), state_key=bound),
        ),
        entry_points=(NodeRef(component_id="node"),),
    )
    executable = Compiler().compile(
        graph,
        CompileContext(initial_state_keys=frozenset({bound})),
    )

    assert not [
        diagnostic
        for diagnostic in executable.diagnostics
        if diagnostic.severity.name == "ERROR"
    ]
    step = PipelineRuntime().advance(
        PipelineRuntime().initialize(executable, _generic_state({bound: 1}))
    )

    assert step.state._store.get(bound) == 2


def test_non_enumerable_writer_still_writes_the_state_bound_key() -> None:
    declared = StateKey(namespace="user", name="value", schema_version=1)
    bound = StateKey(namespace="user", name="actual", schema_version=1)
    node = _GenericNode(
        ComponentContract(
            state=StateContract(writes=(declared,), writes_enumerable=False),
        ),
        lambda view: NodeResult(patch=StatePatch(writes={declared: 9})),
    )
    plan = _generic_plan(
        (node,),
        state_bindings=(
            StateBinding(node=NodeRef(component_id="node_0"), state_key=bound),
        ),
    )

    step = PipelineRuntime().advance(
        PipelineRuntime().initialize(plan, _generic_state({bound: 1}))
    )

    assert step.state._store.get(bound) == 9


def test_non_enumerable_writer_still_deletes_the_state_bound_key() -> None:
    declared = StateKey(namespace="user", name="value", schema_version=1)
    bound = StateKey(namespace="user", name="actual", schema_version=1)
    node = _GenericNode(
        ComponentContract(
            state=StateContract(writes=(declared,), writes_enumerable=False),
        ),
        lambda view: NodeResult(
            patch=StatePatch(writes={}, deletes=frozenset((declared,)))
        ),
    )
    plan = _generic_plan(
        (node,),
        state_bindings=(
            StateBinding(node=NodeRef(component_id="node_0"), state_key=bound),
        ),
    )

    step = PipelineRuntime().advance(
        PipelineRuntime().initialize(plan, _generic_state({bound: 1}))
    )

    assert not step.state._store.contains(bound)


def test_generic_node_rejects_undeclared_writes_and_does_not_mutate_state() -> None:
    key = StateKey(namespace="user", name="value", schema_version=1)
    node = _GenericNode(
        ComponentContract(),
        lambda view: NodeResult(patch=StatePatch(writes={key: 2})),
    )
    plan = _generic_plan((node,))
    state = _generic_state({key: 1})

    with pytest.raises(ValidationError, match="undeclared state write"):
        PipelineRuntime().advance(PipelineRuntime().initialize(plan, state))

    assert state._store.get(key) == 1


def test_recompile_request_without_provider_fails_at_step_boundary() -> None:
    node = _GenericNode(
        ComponentContract(),
        lambda view: NodeResult(
            patch=StatePatch(writes={}),
            commands=(RequestRecompile(),),
        ),
    )
    plan = _generic_plan((node,))

    with pytest.raises(ValidationError, match="cannot satisfy RequestRecompile"):
        PipelineRuntime().advance(
            PipelineRuntime().initialize(plan, _generic_state({}))
        )


def test_async_generic_execution_preserves_node_outcome_metadata() -> None:
    key = StateKey(namespace="user", name="value", schema_version=1)
    node = _GenericNode(
        ComponentContract(state=StateContract(writes=(key,))),
        lambda view: NodeResult(
            patch=StatePatch(writes={key: 3}),
            commands=(RequestCheckpoint(reason="unsupported"),),
        ),
    )
    executable = _generic_plan((node,))
    sequential = SequentialPlan.from_executable_plan(executable)
    optimizer = type(
        "OptimizerStub",
        (),
        {
            "async_evaluation_scheduler": object(),
            "strategy": type("StrategyStub", (), {})(),
            "dispatch": lambda self, event: None,
        },
    )()

    outcome = _OptimizerEnvironment(optimizer, sequential).execute_async_step(
        sequential, _generic_state({key: 1})
    )

    assert outcome.node_results[0].commands == (
        RequestCheckpoint(reason="unsupported"),
    )
    assert outcome.refused_commands == (RequestCheckpoint(reason="unsupported"),)
    assert outcome.state._store.get(key) == 3


def test_failed_node_patch_is_not_committed_and_commands_are_refused() -> None:
    key = StateKey(namespace="user", name="value", schema_version=1)
    node = _GenericNode(
        ComponentContract(state=StateContract(writes=(key,))),
        lambda view: NodeResult(
            patch=StatePatch(writes={key: 2}),
            commands=(RequestCheckpoint(reason="unsupported"),),
            status=NodeStatus.FAILED,
        ),
    )
    plan = _generic_plan((node,))
    state = _generic_state({key: 1})

    step = PipelineRuntime().advance(PipelineRuntime().initialize(plan, state))

    assert step.node_results[0].status is NodeStatus.FAILED
    assert step.refused_commands == (RequestCheckpoint(reason="unsupported"),)
    assert step.state._store.get(key) == 1


def test_data_edge_alone_reaches_and_orders_generic_nodes() -> None:
    key = StateKey(namespace="user", name="value", schema_version=1)
    first = _GenericNode(
        ComponentContract(state=StateContract(writes=(key,))),
        lambda view: StatePatch(writes={key: 7}),
    )

    def read_second(view: Any) -> StatePatch:
        assert view.get(key) == 7
        return StatePatch(writes={})

    second = _GenericNode(
        ComponentContract(state=StateContract(reads=(key,))),
        read_second,
    )
    plan = _generic_plan(
        (first, second),
        data_edges=(
            DataEdge(
                source=NodeRef(component_id="node_0"),
                target=NodeRef(component_id="node_1"),
                source_port="out",
                target_port="in",
            ),
        ),
    )

    step = PipelineRuntime().advance(
        PipelineRuntime().initialize(plan, _generic_state({}))
    )

    assert step.executed_node_ids == ("node_0", "node_1")


def test_sync_environment_executes_compiled_plan_without_strategy_step() -> None:
    stages = [_Stage("a"), _Stage("b")]
    plan = _compiled(stages)

    class PoisonStrategy:
        def step(self, state: OptimizationState, provider: object) -> OptimizationState:
            raise AssertionError("sync runtime must not call strategy.step")

    class Optimizer:
        strategy = PoisonStrategy()
        async_evaluation_scheduler = None

    environment = _OptimizerEnvironment(
        Optimizer(), SequentialPlan.from_executable_plan(plan)
    )
    result = environment.execute(SequentialPlan.from_executable_plan(plan), _state())

    assert result.marker == "b(a(start))"

    adapter = plan.graph.nodes[0].component

    def poison(_: OptimizationState) -> OptimizationState:
        raise AssertionError("compiled Stage execution was bypassed")

    setattr(adapter, "execute", poison)
    assert environment.execute(
        SequentialPlan.from_executable_plan(plan), _state()
    ).marker == ("b(a(start))")


def test_sequential_plan_caches_bound_execute_targets() -> None:
    stages = [_Stage("a"), _Stage("b")]
    plan = SequentialPlan.from_executable_plan(_compiled(stages))

    assert [cast(Any, target).__self__ for target in plan._execute_targets] == [
        cast(Any, node.component).stage for node in plan.execution_nodes
    ]
    assert [cast(Any, target).__func__ for target in plan._execute_targets] == [
        type(stage).execute for stage in stages
    ]


def test_stage_adapter_binds_stage_execute_target_at_construction() -> None:
    stage = _Stage("a")
    adapter = StageNodeAdapter(stage)

    target = cast(Any, adapter._execute_target)
    assert target.__self__ is stage
    assert target.__func__ is type(stage).execute

    def poison(_: OptimizationState) -> OptimizationState:
        raise AssertionError("adapter wrapper was called")

    cast(Any, adapter).execute = poison
    assert target(_state()).marker == "a(start)"


def test_freshened_stage_adapter_uses_nested_cached_execute_target() -> None:
    stage = _Stage("a")
    adapter = StageContractNodeAdapter(stage)
    freshened = _FreshenedComponent(adapter, adapter.contract())

    target = cached_execution_target(freshened)
    assert cast(Any, target).__self__ is stage
    assert cast(Any, target).__func__ is type(stage).execute

    def poison(_: OptimizationState) -> OptimizationState:
        raise AssertionError("freshened adapter wrapper was called")

    setattr(adapter, "execute", poison)
    assert cast(Any, target)(_state()).marker == "a(start)"


def test_stage_adapter_rejects_non_callable_execute_target() -> None:
    class NonExecutableStage(Stage):
        execute = None  # type: ignore[assignment]

    with pytest.raises(ValidationError, match="stage must be executable"):
        StageNodeAdapter(NonExecutableStage())


def test_manually_constructed_plan_rejects_non_executable_node() -> None:
    compiled = _compiled([_Stage("a")])

    class NonExecutable:
        def contract(self):
            return compiled.graph.nodes[0].contract

    node = ComponentNode(component_id="a", component=NonExecutable())
    malformed = replace(
        compiled,
        graph=replace(compiled.graph, nodes=(node,)),
    )

    with pytest.raises(ValidationError, match="node 'a' is not executable"):
        SequentialPlan(plan=malformed, nodes=(node,), execution_nodes=(node,))


def test_graph_only_strategy_is_the_runtime_neutral_source() -> None:
    class MismatchedStrategy(OptimizationStrategy):
        def build_graph(self, provider: ComponentProvider):
            return build_component_graph(
                Pipeline(
                    [
                        _Stage("prefix"),
                        _Stage("async_only"),
                        AsyncEvaluationSubmitStage(object(), EvaluateAll()),
                    ]
                )
            )

    class Provider:
        async_evaluation_scheduler = object()

    graph = build_runtime_neutral_graph(
        MismatchedStrategy(), cast(ComponentProvider, Provider())
    )
    assert tuple(node.component_id for node in graph.nodes[:3]) == (
        "prefix",
        "async_only",
        "async_evaluation_submit",
    )


def test_order_view_ignores_data_only_synthetic_node_and_stage_self_loop() -> None:
    stages = [_Stage("a"), _Stage("b")]
    plan = _compiled(stages)
    synthetic = AdapterComponent(
        adapter=Adapter(
            name="identity_test",
            source=DataSpec(kind="Population"),
            target=DataSpec(kind="Population"),
            lossless=True,
            auto_insertable=True,
        )
    )
    graph = replace(
        plan.graph,
        nodes=(
            *plan.graph.nodes,
            type(plan.graph.nodes[0])(component_id="adapter", component=synthetic),
        ),
        data_edges=(
            *plan.graph.data_edges,
            DataEdge(
                source=NodeRef(component_id="b"),
                target=NodeRef(component_id="b"),
                source_port="x",
                target_port="x",
            ),
        ),
    )
    ordered = SequentialPlan.from_executable_plan(replace(plan, graph=graph))
    assert [node.component_id for node in ordered.nodes] == ["a", "b"]


@pytest.mark.parametrize("kind", ["cycle", "ambiguous", "missing_entry"])
def test_invalid_control_order_is_rejected(kind: str) -> None:
    stages = [_Stage("a"), _Stage("b"), _Stage("c")]
    plan = _compiled(stages)
    if kind == "cycle":
        edges = (
            ControlEdge(
                source=NodeRef(component_id="a"), target=NodeRef(component_id="b")
            ),
            ControlEdge(
                source=NodeRef(component_id="b"), target=NodeRef(component_id="a")
            ),
        )
        graph = replace(plan.graph, control_edges=edges)
    elif kind == "ambiguous":
        edges = (
            ControlEdge(
                source=NodeRef(component_id="a"), target=NodeRef(component_id="c")
            ),
            ControlEdge(
                source=NodeRef(component_id="b"), target=NodeRef(component_id="c")
            ),
        )
        graph = replace(
            plan.graph,
            control_edges=edges,
            entry_points=(NodeRef(component_id="a"), NodeRef(component_id="b")),
        )
    else:
        graph = replace(plan.graph, entry_points=())
    with pytest.raises(ValidationError):
        SequentialPlan.from_executable_plan(replace(plan, graph=graph))


def test_runtime_rejects_missing_capability() -> None:
    plan = replace(
        _compiled([_Stage("a")]),
        required_runtime_capabilities=frozenset({"partial_feedback"}),
    )
    with pytest.raises(ValidationError, match="capabilities"):
        PipelineRuntime().initialize(plan, _state())
