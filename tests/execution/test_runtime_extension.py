"""Focused tests for the runtime extension boundary."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
from pymoo.algorithms.soo.nonconvex.ga import GA as PymooGA  # noqa: N811

from saealib.algorithms.pymoo_algorithm import PymooAlgorithm
from saealib.context import OptimizationState
from saealib.core.compiler.compiler import CompileContext, Compiler, ExecutablePlan
from saealib.core.compiler.graph import ComponentGraph, ComponentNode, NodeRef
from saealib.core.compiler.structured import StructuredGraph
from saealib.core.contracts import ComponentContract, ExecutionContract
from saealib.core.runtime import (
    NodeResult,
    NodeStatus,
    RequestRecompile,
    RuntimeStep,
    StructuredPlan,
)
from saealib.core.state import OPTIMIZATION_STATE_INITIAL_KEYS
from saealib.core.state.patch import StatePatch
from saealib.exceptions import ConfigurationError, StalePlanError, ValidationError
from saealib.execution import (
    AsyncEvaluationScheduler,
    AsyncEvaluator,
    RuntimeFactory,
    RuntimeRegistration,
    RuntimeRegistry,
    SerialEvaluator,
    create_runtime,
    default_runtime_registry,
)
from saealib.execution.runtime import (
    AsyncPipelineRuntime,
    PipelineRuntime,
)
from saealib.islands import IslandModel
from saealib.optimizer import Optimizer
from saealib.problem import Problem
from saealib.stages import EvaluationPlanStage
from saealib.strategies.direct import DirectStrategy, SteadyStateStrategy
from saealib.strategies.gb import GenerationBasedStrategy
from saealib.strategies.ib import IndividualBasedStrategy
from saealib.strategies.ps import PreSelectionStrategy
from saealib.surrogate.rbf import RBFSurrogate
from saealib.surrogate.rbf_kernels import GaussianKernel
from saealib.termination import Termination, max_gen


def _state() -> OptimizationState:
    return object.__new__(OptimizationState)


def _problem() -> Problem:
    return Problem(
        func=lambda x: np.sum(np.asarray(x) ** 2, axis=-1, keepdims=True),
        dim=3,
        n_obj=1,
        direction=np.array([-1.0]),
        lb=[-1.0] * 3,
        ub=[1.0] * 3,
    )


@dataclass
class _Node:
    capabilities: tuple[str, ...] = ()
    reads: int = 0

    def contract(self) -> ComponentContract:
        self.reads += 1
        return ComponentContract(
            execution=ExecutionContract(required_runtime_capabilities=self.capabilities)
        )


def _plan(*nodes: ComponentNode) -> ExecutablePlan:
    return ExecutablePlan(
        graph=ComponentGraph(
            nodes=nodes,
            entry_points=(NodeRef(component_id=nodes[0].component_id),)
            if nodes
            else (),
        ),
        diagnostics=(),
        required_runtime_capabilities=frozenset(
            capability
            for node in nodes
            for capability in node.contract.execution.required_runtime_capabilities
        ),
        active_rule_namespaces=frozenset(),
        active_rule_names=(),
        contract_snapshots=tuple((node.component_id, node.contract) for node in nodes),
    )


def test_runtime_extension_public_surfaces_are_explicit_and_consistent() -> None:
    import saealib.core as core
    import saealib.execution as execution
    import saealib.execution.runtime as runtime

    canonical = (
        "RuntimeRegistry",
        "RuntimeRegistration",
        "create_runtime",
    )
    advanced_hooks = (
        "RuntimeFactory",
        "default_runtime_registry",
    )

    assert len(canonical) == len(set(canonical)) == 3
    assert all(name in execution.__all__ for name in canonical + advanced_hooks)
    assert all(getattr(execution, name) is getattr(runtime, name) for name in canonical)
    assert all(
        getattr(execution, name) is getattr(runtime, name) for name in advanced_hooks
    )
    assert RuntimeFactory is runtime.RuntimeFactory
    assert all(
        name not in core.__all__ and not hasattr(core, name)
        for name in canonical + advanced_hooks
    )

    stub_path = Path(__file__).parents[2] / "src/saealib/execution/__init__.pyi"
    stub_tree = ast.parse(stub_path.read_text(encoding="utf-8"))
    stub_all = next(
        node.value
        for node in stub_tree.body
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id == "__all__"
    )
    if stub_all is None:
        raise AssertionError("execution stub must define __all__")
    assert ast.literal_eval(stub_all) == execution.__all__
    assert all(hasattr(execution, name) for name in execution.__all__)
    assert set(canonical + advanced_hooks).issubset(runtime.__all__)

    assert runtime.__name__ == "saealib.execution.runtime"
    assert runtime.__name__ != execution.__name__


def test_runtime_registry_selects_an_added_provider_without_consumer_changes() -> None:
    registry = RuntimeRegistry()
    selected = object()
    registry.register(
        RuntimeRegistration(
            name="test",
            matches=lambda provider: provider == "test",
            factory=cast(Any, lambda provider, plan: selected),
        )
    )

    assert registry.create("test", cast(Any, object())) is selected
    with pytest.raises(
        ConfigurationError, match=r"no registered runtime.*registered runtime names"
    ):
        registry.create("other", cast(Any, object()))


def test_runtime_registry_rejects_multiple_matching_providers() -> None:
    registry = RuntimeRegistry(
        (
            RuntimeRegistration(
                name="first",
                matches=lambda provider: provider == "shared",
                factory=cast(Any, lambda provider, plan: object()),
            ),
            RuntimeRegistration(
                name="second",
                matches=lambda provider: provider == "shared",
                factory=cast(Any, lambda provider, plan: object()),
            ),
        )
    )

    with pytest.raises(
        ConfigurationError,
        match=r"multiple registered runtimes.*'first'.*'second'",
    ):
        registry.create("shared", cast(Any, object()))


def test_runtime_registry_replace_leaves_exactly_one_match() -> None:
    remaining = object()
    registry = RuntimeRegistry(
        (
            RuntimeRegistration(
                name="first",
                matches=lambda provider: provider == "shared",
                factory=cast(Any, lambda provider, plan: remaining),
            ),
            RuntimeRegistration(
                name="second",
                matches=lambda provider: provider == "shared",
                factory=cast(Any, lambda provider, plan: object()),
            ),
        )
    )
    registry.replace(
        "second",
        RuntimeRegistration(
            name="second",
            matches=lambda provider: False,
            factory=cast(Any, lambda provider, plan: object()),
        ),
    )

    assert registry.create("shared", cast(Any, object())) is remaining


def test_default_runtime_capability_offers_are_owned_by_runtime_providers() -> None:
    async_optimizer = Optimizer(_problem()).set_async_evaluation_scheduler(
        AsyncEvaluationScheduler(SerialEvaluator())
    )
    sync_optimizer = Optimizer(_problem())

    assert AsyncPipelineRuntime.capabilities == frozenset({"partial_feedback"})
    assert PipelineRuntime.capabilities == frozenset()
    assert default_runtime_registry.offered_capabilities(async_optimizer) == frozenset(
        {"partial_feedback"}
    )
    assert default_runtime_registry.offered_capabilities(sync_optimizer) == frozenset()

    partial_sync = Optimizer(_problem()).set_algorithm(
        PymooAlgorithm(PymooGA(pop_size=4), allow_partial_tell=True)
    )
    assert default_runtime_registry.offered_capabilities(partial_sync) == frozenset(
        {"partial_feedback"}
    )
    partial_sync.set_strategy(DirectStrategy(n_offspring=4))
    partial_sync.resolve_defaults()
    plan = partial_sync._compile_plan()
    assert plan is not None
    runtime = create_runtime(partial_sync)
    initializer = partial_sync.initializer
    assert initializer is not None
    state = initializer.initialize(partial_sync, partial_sync.problem)
    runtime.initialize(plan, state)


def test_runtime_registration_rejects_non_callable_capability_provider() -> None:
    with pytest.raises(ValidationError, match="capability_provider"):
        RuntimeRegistration(
            name="invalid",
            matches=lambda optimizer: True,
            factory=cast(Any, lambda optimizer, plan: object()),
            capability_provider=cast(Any, object()),
        )


def test_real_strategy_graphs_compile_and_initialize_selected_runtimes() -> None:
    island_optimizers = [
        Optimizer(_problem()).set_strategy(DirectStrategy(n_offspring=4)),
        Optimizer(_problem()).set_strategy(DirectStrategy(n_offspring=4)),
    ]
    for optimizer in island_optimizers:
        optimizer.resolve_defaults()
    island_model = IslandModel(island_optimizers, migration_interval=0)
    configurations = {
        "default": [Optimizer(_problem())],
        "surrogate": [
            Optimizer(_problem()).set_surrogate(RBFSurrogate(GaussianKernel()))
        ],
        "async": [
            Optimizer(_problem()).set_async_evaluation_scheduler(
                AsyncEvaluationScheduler(SerialEvaluator(), max_pending=2)
            )
        ],
        # Construct the real model so this coverage includes its compatibility
        # and optimizer ownership wiring, then compile each owned strategy graph.
        "island": island_model.optimizers,
        "pymoo": [
            Optimizer(_problem())
            .set_algorithm(PymooAlgorithm(PymooGA(pop_size=4)))
            .set_strategy(DirectStrategy(n_offspring=4))
        ],
    }
    for name, optimizers in configurations.items():
        for optimizer in optimizers:
            optimizer.resolve_defaults()
            plan = optimizer._compile_plan()
            assert plan is not None, name
            assert not [
                diagnostic
                for diagnostic in plan.diagnostics
                if diagnostic.severity.name == "ERROR"
            ], name
            node_ids = {node.component_id for node in plan.graph.nodes}
            assert len(node_ids) == len(plan.graph.nodes)
            assert all(
                edge.source.component_id in node_ids
                and edge.target.component_id in node_ids
                for edge in plan.graph.data_edges
            )
            connections = {
                (
                    edge.source.role,
                    edge.source_port,
                    edge.target.role,
                    edge.target_port,
                    tuple(item.name for item in edge.transforms),
                )
                for edge in plan.graph.data_edges
                if edge.source.role != "state" and edge.target.role != "state"
            }
            graph = plan.graph
            for edge in graph.data_edges:
                if edge.source.role != "state" or edge.target.role == "state":
                    continue
                state_node = graph.node_by_id(edge.source.component_id)
                flow_origin = getattr(state_node.component, "flow_origin", None)
                if not isinstance(flow_origin, str):
                    continue
                source = graph.node_by_id(flow_origin)
                for producer in graph.data_edges:
                    if (
                        producer.source.component_id == flow_origin
                        and producer.target.component_id == state_node.component_id
                    ):
                        connections.add(
                            (
                                source.role,
                                producer.source_port,
                                edge.target.role,
                                edge.target_port,
                                tuple(
                                    item.name
                                    for item in (*producer.transforms, *edge.transforms)
                                ),
                            )
                        )
            expected_connections = {
                (
                    "proposer",
                    "genomes",
                    "evaluation_planner",
                    "candidates",
                    (),
                )
                if name in {"island", "pymoo"}
                else (
                    "proposer",
                    "genomes",
                    "predictor",
                    "candidates",
                    ("encode_features",),
                )
            }
            if name == "async":
                expected_connections.update(
                    {
                        ("feedback_builder", "feedback", "predictor", "input", ()),
                        ("predictor", "output", "feedback_consumer", "feedback", ()),
                    }
                )
            else:
                expected_connections.update(
                    {
                        (
                            "feedback_builder",
                            "feedback",
                            "feedback_consumer",
                            "feedback",
                            (),
                        ),
                        (
                            "evaluation_planner",
                            "evaluation_requests",
                            "evaluator",
                            "genomes",
                            ("select_requested_candidates", "extract_genomes"),
                        ),
                    }
                )
            assert expected_connections <= connections, name
            runtime = create_runtime(optimizer)
            initializer = optimizer.initializer
            assert initializer is not None
            state = initializer.initialize(optimizer, optimizer.problem)
            session = runtime.initialize(plan, state)
            assert session.state is state, name
            if name == "async":
                assert isinstance(runtime, AsyncPipelineRuntime)
            else:
                assert isinstance(runtime, PipelineRuntime)


def test_generation_strategy_retains_held_algorithm_state_reads() -> None:
    from saealib import LHSInitializer

    problem = Problem(
        func=lambda x: float(np.sum(x * x)),
        dim=3,
        n_obj=1,
        direction=np.array([-1.0]),
        lb=[-5.0] * 3,
        ub=[5.0] * 3,
    )
    state = (
        Optimizer(problem, seed=937)
        .set_strategy(GenerationBasedStrategy(2))
        .set_initializer(
            LHSInitializer(n_init_archive=12, n_init_population=8, seed=937)
        )
        .set_termination(Termination(max_gen(4)))
        .run()
    )

    assert state.gen == 6


def test_generation_strategy_runs_as_structured_async_plan() -> None:
    evaluator = AsyncEvaluator(SerialEvaluator())
    optimizer = (
        Optimizer(_problem())
        .set_strategy(GenerationBasedStrategy(gen_ctrl=0))
        .set_evaluator(evaluator)
        .set_async_evaluation_scheduler(
            AsyncEvaluationScheduler(evaluator, max_pending=2)
        )
        .set_termination(Termination(max_gen(1)))
    )
    try:
        optimizer.resolve_defaults()
        plan = optimizer._compile_plan()
        assert plan is not None
        assert isinstance(plan.graph, StructuredGraph)
        initializer = optimizer.initializer
        assert initializer is not None
        state = initializer.initialize(optimizer, optimizer.problem)
        runtime = create_runtime(optimizer)
        session = runtime.initialize(plan, state)
        assert isinstance(session.plan, StructuredPlan)

        first = runtime.advance(session)

        assert any(result.status is NodeStatus.BLOCKED for result in first.node_results)
        step = first
        for _ in range(10):
            if step.finished:
                break
            assert step.session is not None
            step = runtime.advance(step.session)

        assert step.finished
        assert not step.state.pending_evaluations
    finally:
        evaluator.close()


def _graph_signature(optimizer: Optimizer) -> tuple[object, ...]:
    graph = cast(Any, optimizer.strategy).build_graph(optimizer)
    return (
        tuple(node.component_id for node in graph.nodes),
        tuple(
            (
                edge.source.component_id,
                edge.source.role,
                edge.source_port,
                edge.target.component_id,
                edge.target.role,
                edge.target_port,
            )
            for edge in graph.data_edges
        ),
        tuple(
            (edge.source.component_id, edge.target.component_id)
            for edge in graph.control_edges
        ),
    )


def test_strategy_graph_signatures_are_runtime_neutral() -> None:
    strategy_factories = (
        lambda: DirectStrategy(n_offspring=4),
        SteadyStateStrategy,
        lambda: GenerationBasedStrategy(gen_ctrl=2),
        lambda: IndividualBasedStrategy(evaluation_ratio=0.5),
        lambda: PreSelectionStrategy(n_candidates=8, n_select=2),
    )

    for make_strategy in strategy_factories:
        sync = Optimizer(_problem()).set_strategy(make_strategy())
        async_optimizer = Optimizer(_problem()).set_strategy(make_strategy())
        async_optimizer.set_async_evaluation_scheduler(
            AsyncEvaluationScheduler(SerialEvaluator())
        )
        sync.resolve_defaults()
        async_optimizer.resolve_defaults()

        assert _graph_signature(sync) == _graph_signature(async_optimizer)


def test_five_strategy_graphs_compile_with_runtime_neutral_topology() -> None:
    strategy_factories = (
        lambda: DirectStrategy(n_offspring=4),
        SteadyStateStrategy,
        lambda: GenerationBasedStrategy(gen_ctrl=2),
        lambda: IndividualBasedStrategy(evaluation_ratio=0.5),
        lambda: PreSelectionStrategy(n_candidates=8, n_select=2),
    )

    for make_strategy in strategy_factories:
        sync = Optimizer(_problem()).set_strategy(make_strategy())
        asynchronous = Optimizer(_problem()).set_strategy(make_strategy())
        asynchronous.set_async_evaluation_scheduler(
            AsyncEvaluationScheduler(SerialEvaluator())
        )
        sync.resolve_defaults()
        asynchronous.resolve_defaults()
        sync_graph = cast(Any, sync.strategy).build_graph(sync)
        async_graph = cast(Any, asynchronous.strategy).build_graph(asynchronous)

        assert _graph_signature(sync) == _graph_signature(asynchronous)
        for optimizer, graph in ((sync, sync_graph), (asynchronous, async_graph)):
            plan = Compiler().compile(
                graph,
                CompileContext(
                    space=optimizer.problem.space,
                    problem=optimizer.problem,
                    offered_runtime_capabilities=(
                        frozenset({"partial_feedback"})
                        if optimizer.async_evaluation_scheduler is not None
                        else frozenset()
                    ),
                    initial_state_keys=OPTIMIZATION_STATE_INITIAL_KEYS,
                ),
            )
            assert [
                diagnostic
                for diagnostic in plan.diagnostics
                if diagnostic.severity.name == "ERROR"
            ] == []


def test_async_runtime_uses_canonical_sync_graph_contract() -> None:
    optimizer = Optimizer(_problem()).set_async_evaluation_scheduler(
        AsyncEvaluationScheduler(SerialEvaluator())
    )
    optimizer.resolve_defaults()
    graph = cast(Any, optimizer.strategy).build_graph(optimizer)
    node = graph.node_by_id("evaluation_plan")
    stage = node.component.stage
    assert isinstance(stage, EvaluationPlanStage)
    assert all(
        "runtime" not in type(item.component.stage).__name__.lower()
        for item in graph.nodes
        if hasattr(item.component, "stage")
    )
    stage_ids = [
        item.component_id for item in graph.nodes if hasattr(item.component, "stage")
    ]
    assert stage_ids[-1] == "evaluation_acknowledge"


def test_replaced_default_registration_is_selected() -> None:
    selected = object()
    registry = RuntimeRegistry(default_runtime_registry.registrations())
    registry.replace(
        "sync",
        RuntimeRegistration(
            name="sync",
            matches=lambda provider: provider == "sync",
            factory=cast(Any, lambda provider, plan: selected),
        ),
    )

    assert registry.create("sync", cast(Any, object())) is selected
    with pytest.raises(ValidationError, match="already exists"):
        registry.register(registry.registrations()[-1])


def test_initialize_reads_each_node_contract_once_and_rejects_stale_plan() -> None:
    node = _Node()
    second = _Node()
    component_node = ComponentNode(component_id="node", component=node)
    second_node = ComponentNode(component_id="second", component=second)
    plan = _plan(component_node, second_node)
    # The plan snapshot is valid for the first initialize call.
    runtime = PipelineRuntime()
    runtime.initialize(plan, _state())
    assert node.reads == 2
    assert second.reads == 2

    node.capabilities = ("partial_feedback",)
    with pytest.raises(StalePlanError, match="stale_plan") as error:
        runtime.initialize(plan, _state())
    assert error.value.code == "stale_plan"
    assert node.reads == 3
    assert second.reads == 3


def test_runtime_capabilities_are_checked_as_a_required_subset() -> None:
    node = _Node(capabilities=("partial_feedback",))
    component_node = ComponentNode(component_id="node", component=node)
    plan = _plan(component_node)

    class LimitedRuntime(PipelineRuntime):
        capabilities = frozenset()

    runtime = LimitedRuntime()

    with pytest.raises(ValidationError, match="partial_feedback"):
        runtime.initialize(plan, _state())


def test_refusal_remains_a_step_outcome_and_recompile_is_step_boundary_data() -> None:
    step = RuntimeStep(
        state=_state(),
        refused_commands=(RequestRecompile(reason="policy"),),
    )
    assert step.refused_commands[0].reason == "policy"

    boundary = RuntimeStep(
        state=_state(),
        node_results=(
            NodeResult(
                patch=StatePatch(writes={}),
                commands=(RequestRecompile(),),
            ),
        ),
    )
    assert boundary.recompile_requested
    assert boundary.refused_commands == ()


def test_async_runtime_uses_public_async_seam_without_stage_name_dependencies() -> None:
    source = (Path(__file__).parents[2] / "src/saealib/execution/runtime.py").read_text(
        encoding="utf-8"
    )

    assert '== "evaluation_plan"' not in source
    assert 'getattr(plan_stage, "_planner"' not in source
    assert "AsyncEvaluationSubmitStage" not in source
    assert 'getattr(node.component, "execute_async", None)' in source


def test_runtime_uses_strategy_feedback_builder_when_plan_has_no_feedback_stage():
    builder = object()
    strategy = SimpleNamespace(feedback_builder=builder)
    scheduler = SimpleNamespace(_feedback_accumulator=object())
    executable_plan = ExecutablePlan(
        graph=StructuredGraph(nodes=()),
        diagnostics=(),
        required_runtime_capabilities=frozenset(),
        active_rule_namespaces=frozenset(),
        active_rule_names=(),
    )
    optimizer = SimpleNamespace(
        executable_plan=executable_plan,
        strategy=strategy,
        problem=_problem(),
        algorithm=None,
        evaluator=None,
        feedback_builder=None,
        cbmanager=None,
        async_evaluation_scheduler=scheduler,
    )

    runtime = create_runtime(optimizer)

    assert runtime is not None
    assert scheduler.feedback_builder is builder
