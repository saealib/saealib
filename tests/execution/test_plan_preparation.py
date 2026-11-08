from __future__ import annotations

import copy
import pickle
from dataclasses import replace
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

import saealib.execution.prepare as prepare_module
import saealib.execution.runtime as runtime_module
import saealib.optimizer as optimizer_module
from saealib.algorithms.pso import PSO
from saealib.core.compiler import ComponentGraph, ComponentNode, NodeRef
from saealib.core.compiler.graph import ControlEdge
from saealib.core.compiler.structured import StructuredGraph
from saealib.core.contracts import (
    AssumptionSet,
    ComponentContract,
    PartSpec,
    ServiceRequirement,
)
from saealib.core.runtime import NodeResult, StructuredPlan
from saealib.core.state import OPTIMIZATION_STATE_INITIAL_KEYS, StatePatch
from saealib.exceptions import ConfigurationError, ValidationError
from saealib.execution import (
    AsyncEvaluationScheduler,
    RuntimeRegistration,
    SerialEvaluator,
)
from saealib.execution.runtime import (
    _OptimizerEnvironment,
    create_runtime,
    default_runtime_registry,
    execute_strategy_step,
    resolve_plan,
)
from saealib.optimizer import ComponentProvider, Optimizer
from saealib.problem import Problem
from saealib.strategies.direct import DirectStrategy
from saealib.termination import Termination, max_gen


def _problem() -> Problem:
    return Problem(
        func=lambda x: np.array([float(np.sum(np.asarray(x) ** 2))]),
        dim=2,
        n_obj=1,
        direction=np.array([-1.0]),
        lb=[-5.0, -5.0],
        ub=[5.0, 5.0],
    )


class _BatchEvaluator(SerialEvaluator):
    def supports_batch_rollback(self) -> bool:
        return True


def _search_result(asynchronous: bool, verification_enabled: bool):
    optimizer = Optimizer(_problem(), seed=29).set_termination(Termination(max_gen(3)))
    optimizer._verification_enabled = verification_enabled
    if asynchronous:
        optimizer.set_async_evaluation_scheduler(
            AsyncEvaluationScheduler(SerialEvaluator(), max_pending=2)
        )
    state = optimizer.run()
    assert optimizer.executable_plan is not None
    assert not any(
        item.severity.value == "error" for item in optimizer.executable_plan.diagnostics
    )
    history = state.history
    return {
        "fe": state.fe,
        "gen": state.gen,
        "population_ids": state.population.get_array("id").tolist(),
        "population_x": state.population.get_array("x").tolist(),
        "population_f": state.population.get_array("f").tolist(),
        "archive_ids": state.archive.get_array("id").tolist(),
        "archive_x": state.archive.get_array("x").tolist(),
        "archive_f": state.archive.get_array("f").tolist(),
        "history": {
            channel: {
                name: np.asarray(values).tolist()
                for name, values in history.channel(channel).items()
            }
            for channel in sorted(history.enabled)
        }
        if history is not None
        else None,
        "best_x": state.archive.get_array("x")[
            int(np.argmin(state.archive.get_array("f")[:, 0]))
        ].tolist(),
        "best_f": float(np.min(state.archive.get_array("f")[:, 0])),
    }


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("mutation", ["setter", "assignment"])
def test_graph_building_mutations_use_the_live_optimizer(
    asynchronous: bool, mutation: str
) -> None:
    events: dict[str, list[str]] = {"first": [], "second": []}

    class EventPSO(PSO):
        def __init__(self, name: str) -> None:
            super().__init__()
            self.name = name

        def ask(self, request: Any, state: Any):
            events[self.name].append("ask")
            return super().ask(request, state)

        def tell(self, feedback: Any, state: Any):
            events[self.name].append("tell")
            return super().tell(feedback, state)

    first, second = EventPSO("first"), EventPSO("second")

    class MutatingStrategy(DirectStrategy):
        def __init__(self) -> None:
            super().__init__(n_offspring=2)
            self.changed = False

        def build_graph(self, provider: Any) -> ComponentGraph:
            if not self.changed:
                self.changed = True
                if mutation == "setter":
                    provider.set_algorithm(second)
                else:
                    provider.algorithm = second
            return super().build_graph(provider)

    optimizer = (
        Optimizer(_problem(), seed=29)
        .set_algorithm(first)
        .set_strategy(MutatingStrategy())
        .set_termination(Termination(lambda state: state.gen >= 3))
    )
    if asynchronous:
        optimizer.set_async_evaluation_scheduler(
            AsyncEvaluationScheduler(_BatchEvaluator(), max_pending=2)
        )

    state = optimizer.run()

    assert state.gen >= 3
    assert optimizer.algorithm is second
    assert events["first"] == []
    assert events["second"].count("ask") == events["second"].count("tell")
    assert events["second"].count("ask") >= 3


def test_snapshot_provider_view_supports_copy_and_pickle() -> None:
    view = runtime_module._SnapshotProviderView(
        SimpleNamespace(problem="live-problem"),
        {"algorithm": "snapshot-algorithm", "strategy": "snapshot-strategy"},
    )

    copied = copy.copy(view)
    restored = pickle.loads(pickle.dumps(view))

    assert copied.algorithm == "snapshot-algorithm"
    assert restored.algorithm == "snapshot-algorithm"
    assert restored.strategy == "snapshot-strategy"
    assert restored.problem == "live-problem"


def test_bare_provider_prepares_plan_and_runtime_without_a_problem(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class NoOp:
        def contract(self) -> ComponentContract:
            return ComponentContract()

        def execute(self, view: Any) -> NodeResult:
            del view
            return NodeResult(patch=StatePatch(writes={}))

    class Strategy:
        def build_graph(self, provider: Any) -> ComponentGraph:
            del provider
            return ComponentGraph(
                nodes=(ComponentNode(component_id="noop", component=NoOp()),),
                entry_points=(NodeRef(component_id="noop"),),
            )

    provider = SimpleNamespace(
        strategy=Strategy(), problem=None, dispatch=lambda event: None
    )
    contexts: list[Any] = []
    compiler_type = getattr(prepare_module, "Compiler", None)
    if compiler_type is not None:
        original_compile = compiler_type.compile

        def capture_compile(compiler: Any, graph: Any, context: Any = None):
            contexts.append(context)
            return original_compile(compiler, graph, context)

        monkeypatch.setattr(compiler_type, "compile", capture_compile)
    expected_capabilities = default_runtime_registry.offered_capabilities(provider)

    plan = resolve_plan(provider)
    runtime = create_runtime(provider)

    assert type(plan).__name__ == "ExecutablePlan"
    assert runtime is not None
    if contexts:
        assert len(contexts) == 2
        assert all(
            context.problem is None and context.space is None for context in contexts
        )
        assert all(
            context.offered_runtime_capabilities == expected_capabilities
            and context.initial_state_keys == OPTIMIZATION_STATE_INITIAL_KEYS
            for context in contexts
        )


def test_invalid_strategy_graph_raises_validation_error() -> None:
    class InvalidStrategy:
        def build_graph(self, provider: Any) -> None:
            del provider
            return None

    provider = SimpleNamespace(problem=_problem(), strategy=InvalidStrategy())

    with pytest.raises(ValidationError, match="ComponentGraph"):
        resolve_plan(provider)


def test_direct_step_runs_plan_to_finish_its_unfinished_work() -> None:
    class SelfCompletingWork:
        def __init__(self) -> None:
            self.calls = 0
            self.unfinished = True

        def contract(self) -> ComponentContract:
            return ComponentContract()

        def execute(self, view: Any) -> NodeResult:
            del view
            self.calls += 1
            self.unfinished = False
            return NodeResult(patch=StatePatch(writes={}))

        def has_async_work(self, state: Any) -> bool:
            del state
            return self.unfinished

    class Strategy:
        def __init__(self, work: SelfCompletingWork) -> None:
            self.work = work

        def build_graph(self, provider: Any) -> ComponentGraph:
            del provider
            return ComponentGraph(
                nodes=(ComponentNode(component_id="work", component=self.work),),
                entry_points=(NodeRef(component_id="work"),),
            )

    problem = _problem()
    optimizer = Optimizer(problem, seed=29)
    optimizer.resolve_defaults()
    state = _initialize(optimizer)
    work = SelfCompletingWork()
    strategy = Strategy(work)
    optimizer.set_strategy(cast(Any, strategy))
    optimizer.set_async_evaluation_scheduler(
        AsyncEvaluationScheduler(SerialEvaluator(), max_pending=2)
    )

    with pytest.warns(
        DeprecationWarning, match=r"execute_strategy_step.*Optimizer\.iterate"
    ):
        execute_strategy_step(strategy, state, optimizer)

    assert work.calls == 1
    assert not work.unfinished


@pytest.mark.parametrize("entrypoint", ["optimizer", "resolve", "direct"])
def test_preparation_uses_the_runtime_problem_after_graph_build(
    monkeypatch: pytest.MonkeyPatch, entrypoint: str
) -> None:
    configured_problem = _problem()
    graph_problem = Problem(
        func=lambda x: np.array([float(np.sum(np.asarray(x) ** 2))]),
        dim=2,
        n_obj=1,
        direction=np.array([-1.0]),
        lb=[-1.0, -1.0],
        ub=[1.0, 1.0],
    )
    observed_bounds: list[float] = []

    class UsesBounds:
        def contract(self) -> ComponentContract:
            return ComponentContract(
                required_services=(ServiceRequirement(name="BoundsService"),)
            )

        def execute(self, view: Any) -> NodeResult:
            bounds = view.context.compiled_service("BoundsService").bounds
            observed_bounds.append(float(bounds[1][0]))
            return NodeResult(patch=StatePatch(writes={}))

    class MutatingStrategy:
        def build_graph(self, provider: Any) -> ComponentGraph:
            provider.problem = graph_problem
            return ComponentGraph(
                nodes=(
                    ComponentNode(component_id="uses_bounds", component=UsesBounds()),
                ),
                entry_points=(NodeRef(component_id="uses_bounds"),),
            )

    contexts: list[Any] = []
    plans: list[Any] = []
    original_compile = prepare_module.Compiler.compile

    def capture_compile(compiler: Any, graph: Any, context: Any = None):
        contexts.append(context)
        plan = original_compile(compiler, graph, context)
        plans.append(plan)
        return plan

    def offered_capabilities(provider: Any) -> frozenset[str]:
        return (
            frozenset({"partial_feedback"})
            if provider.problem is graph_problem
            else frozenset()
        )

    monkeypatch.setattr(prepare_module.Compiler, "compile", capture_compile)
    monkeypatch.setattr(
        default_runtime_registry, "offered_capabilities", offered_capabilities
    )

    optimizer = Optimizer(configured_problem, seed=29)
    optimizer.resolve_defaults()
    optimizer.set_strategy(cast(Any, MutatingStrategy()))

    expected_problem = graph_problem
    if entrypoint == "optimizer":
        optimizer._compile_plan()
    elif entrypoint == "resolve":
        resolve_plan(optimizer)
    else:
        state = _initialize(optimizer)
        expected_problem = state.problem
        execute_strategy_step(optimizer.strategy, state, optimizer)

    assert len(contexts) == 1
    assert contexts[0].problem is expected_problem
    assert contexts[0].space is expected_problem.space
    assert contexts[0].offered_runtime_capabilities == frozenset({"partial_feedback"})
    service = plans[0].graph.nodes[0].resolved_services["BoundsService"]
    assert service is expected_problem.space.services.require("BoundsService")
    if entrypoint == "direct":
        assert observed_bounds == [5.0]


def test_direct_step_uses_the_state_problem_when_provider_problem_differs() -> None:
    configured_problem = _problem()
    state_problem = Problem(
        func=lambda x: np.array([float(np.sum(np.asarray(x) ** 2))]),
        dim=2,
        n_obj=1,
        direction=np.array([-1.0]),
        lb=[-1.0, -1.0],
        ub=[1.0, 1.0],
    )
    observed_bounds: list[float] = []

    class UsesBounds:
        def contract(self) -> ComponentContract:
            return ComponentContract(
                required_services=(ServiceRequirement(name="BoundsService"),)
            )

        def execute(self, view: Any) -> NodeResult:
            bounds = view.context.compiled_service("BoundsService").bounds
            observed_bounds.append(float(bounds[1][0]))
            return NodeResult(patch=StatePatch(writes={}))

    class Strategy:
        def build_graph(self, provider: Any) -> ComponentGraph:
            del provider
            return ComponentGraph(
                nodes=(
                    ComponentNode(component_id="uses_bounds", component=UsesBounds()),
                ),
                entry_points=(NodeRef(component_id="uses_bounds"),),
            )

    optimizer = Optimizer(configured_problem, seed=29)
    optimizer.resolve_defaults()
    strategy = Strategy()
    optimizer.set_strategy(cast(Any, strategy))
    state = _initialize(optimizer).replace(problem=state_problem)

    result = execute_strategy_step(strategy, state, optimizer)

    assert optimizer.problem is configured_problem
    assert result.problem is state_problem
    assert observed_bounds == [1.0]


class _UnreachableComponent:
    def contract(self) -> ComponentContract:
        return ComponentContract()

    def execute(self, view: Any) -> NodeResult:
        del view
        return NodeResult(patch=StatePatch(writes={}))


class _UnreachableStrategy(DirectStrategy):
    def build_graph(self, provider: ComponentProvider) -> StructuredGraph:
        graph = cast(StructuredGraph, super().build_graph(provider))
        unreachable = ComponentNode(
            component_id="unreachable", component=_UnreachableComponent()
        )
        return replace(
            graph,
            nodes=(*graph.nodes, unreachable),
            operations=(*graph.operations, unreachable),
        )


class _MalformedStrategy(DirectStrategy):
    def build_graph(self, provider: ComponentProvider) -> ComponentGraph:
        graph = super().build_graph(provider)
        return ComponentGraph(
            nodes=graph.nodes,
            data_edges=graph.data_edges,
            control_edges=(
                *graph.control_edges,
                ControlEdge(
                    source=graph.entry_points[0],
                    target=NodeRef(component_id="missing"),
                ),
            ),
            state_bindings=graph.state_bindings,
            entry_points=graph.entry_points,
        )


class _FailingStrategy(DirectStrategy):
    def build_graph(self, provider: ComponentProvider) -> ComponentGraph:
        del provider
        raise ValueError("strategy graph cannot be built")


class _ContractChild:
    def contract(self) -> ComponentContract:
        return ComponentContract(
            assumptions=AssumptionSet({"evaluation.deterministic": False})
        )


class _MismatchedContract:
    def __init__(self) -> None:
        self.child = _ContractChild()

    def contract(self) -> ComponentContract:
        return ComponentContract(
            parts=(PartSpec(name="child", contract=ComponentContract()),)
        )


def _configured_optimizer(asynchronous: bool) -> tuple[Optimizer, list[object]]:
    calls: list[object] = []
    problem = Problem(
        func=lambda x: calls.append(x) or np.array([float(np.sum(np.asarray(x) ** 2))]),
        dim=2,
        n_obj=1,
        direction=np.array([-1.0]),
        lb=[-5.0, -5.0],
        ub=[5.0, 5.0],
    )
    optimizer = (
        Optimizer(problem, seed=29)
        .set_strategy(_UnreachableStrategy())
        .set_termination(Termination(max_gen(1)))
    )
    if asynchronous:
        optimizer.set_async_evaluation_scheduler(
            AsyncEvaluationScheduler(SerialEvaluator(), max_pending=2)
        )
    optimizer.resolve_defaults()
    return optimizer, calls


def _initialize(optimizer: Optimizer):
    assert optimizer.initializer is not None
    assert optimizer.problem is not None
    return optimizer.initializer.initialize(optimizer, optimizer.problem)


@pytest.mark.parametrize("asynchronous", [False, True])
def test_verification_choice_preserves_seeded_search_results(
    asynchronous: bool,
) -> None:
    verified = _search_result(asynchronous, verification_enabled=True)
    unverified = _search_result(asynchronous, verification_enabled=False)

    assert verified == unverified


def test_skipping_verification_allows_optimizer_evaluation() -> None:
    optimizer, calls = _configured_optimizer(False)
    optimizer._verification_enabled = False

    state = optimizer.run()

    assert calls
    assert state.fe > 0
    assert optimizer.executable_plan is not None
    assert not any(
        item.code == "unreachable_node"
        for item in optimizer.executable_plan.diagnostics
    )


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("entrypoint", ["run", "iterate", "run_from", "iterate_from"])
def test_verification_errors_stop_optimizer_entrypoints_before_evaluation(
    asynchronous: bool, entrypoint: str
) -> None:
    optimizer, calls = _configured_optimizer(asynchronous)
    state = None
    if entrypoint.endswith("_from"):
        state = _initialize(optimizer)
    evaluations_before = len(calls)

    with pytest.raises(ConfigurationError, match="unreachable_node"):
        if entrypoint == "run":
            optimizer.run()
        elif entrypoint == "iterate":
            optimizer.iterate()
        elif entrypoint == "run_from":
            assert state is not None
            optimizer.run_from(state)
        else:
            assert state is not None
            optimizer.iterate_from(state)

    assert len(calls) == evaluations_before


@pytest.mark.parametrize("asynchronous", [False, True])
def test_direct_strategy_step_rejects_verification_errors_before_evaluation(
    asynchronous: bool,
) -> None:
    optimizer, calls = _configured_optimizer(asynchronous)
    state = _initialize(optimizer)
    evaluations_before = len(calls)

    if asynchronous:
        with (
            pytest.warns(
                DeprecationWarning, match=r"execute_strategy_step.*Optimizer\.iterate"
            ),
            pytest.raises(ConfigurationError, match="unreachable_node"),
        ):
            execute_strategy_step(optimizer.strategy, state, optimizer)
    else:
        with pytest.raises(ConfigurationError, match="unreachable_node"):
            execute_strategy_step(optimizer.strategy, state, optimizer)

    assert len(calls) == evaluations_before


def test_skipping_verification_keeps_structural_errors_blocking() -> None:
    optimizer, calls = _configured_optimizer(False)
    optimizer.set_strategy(_MalformedStrategy())
    optimizer._verification_enabled = False

    with pytest.raises(ConfigurationError, match="invalid_graph_edge"):
        optimizer.run()

    assert calls == []


def test_optimizer_preserves_strategy_graph_build_errors() -> None:
    optimizer, _ = _configured_optimizer(False)
    optimizer.set_strategy(_FailingStrategy())

    with pytest.raises(ValueError, match="strategy graph cannot be built"):
        optimizer._compile_plan()


def test_skipping_verification_keeps_component_contract_errors_blocking() -> None:
    optimizer, calls = _configured_optimizer(False)
    optimizer.set_algorithm(cast(Any, _MismatchedContract()))
    optimizer._verification_enabled = False

    with pytest.raises(ConfigurationError, match="part_contract_mismatch"):
        optimizer.run()

    assert calls == []


def _plan_shape(plan: Any) -> tuple[object, ...]:
    graph = plan.graph
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
        tuple(
            (item.adapter_name, item.source_path, item.target_path)
            for item in plan.inserted_adapters
        ),
        tuple(
            (node.component_id, tuple(sorted(node.resolved_services)))
            for node in graph.nodes
            if node.resolved_services
        ),
    )


def _plan_binding_ids(plan: Any) -> tuple[tuple[str, int], ...]:
    return tuple(
        (node.component_id, id(stage._algorithm))
        for node in plan.graph.nodes
        if (stage := getattr(node.component, "stage", None)) is not None
        and hasattr(stage, "_algorithm")
    )


def _plan_service_ids(plan: Any) -> tuple[tuple[str, str, int], ...]:
    return tuple(
        (node.component_id, name, id(service))
        for node in plan.graph.nodes
        for name, service in node.resolved_services.items()
    )


def test_execution_paths_share_plan_preparation_and_runtime_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_sync = next(
        item for item in default_runtime_registry.registrations() if item.name == "sync"
    )
    default_runtime_registry.replace(
        "sync",
        RuntimeRegistration(
            name="sync",
            matches=original_sync.matches,
            factory=original_sync.factory,
            offered_runtime_capabilities=frozenset({"partial_feedback"}),
        ),
    )
    original_prepare = prepare_module._prepare_plan
    original_compiler = prepare_module.Compiler
    compiled: list[Any] = []
    contexts: list[Any] = []

    def record_prepare(*args, **kwargs):
        plan = original_prepare(*args, **kwargs)
        compiled.append(plan)
        return plan

    class CapturingCompiler:
        def compile(self, graph, context=None):
            contexts.append(context)
            return original_compiler().compile(graph, context)

    monkeypatch.setattr(optimizer_module, "_prepare_plan", record_prepare)
    monkeypatch.setattr(runtime_module, "_prepare_plan", record_prepare)
    monkeypatch.setattr(prepare_module, "Compiler", CapturingCompiler)

    try:
        optimizer = Optimizer(_problem(), seed=29).set_strategy(DirectStrategy())
        optimizer.set_termination(Termination(max_gen(1)))
        state = optimizer.run()
        assert optimizer.executable_plan is not None
        old_algorithm = optimizer.algorithm
        old_plan = optimizer.executable_plan

        optimizer._executable_plan = None
        resolved = runtime_module.resolve_plan(optimizer)
        assert _plan_shape(resolved) == _plan_shape(old_plan)

        environment = _OptimizerEnvironment(
            optimizer, StructuredPlan.from_executable_plan(old_plan)
        )
        optimizer.set_algorithm(PSO())
        environment._refresh_plan_if_needed(state)
        replacement_plan = optimizer.executable_plan
        assert replacement_plan is not None
        environment.recompile(environment.plan)

        optimizer.set_algorithm(old_algorithm)
        optimizer.run_from(state)
        optimizer.iterate_from(state).close()
        execute_strategy_step(optimizer.strategy, state, optimizer)

        asynchronous = Optimizer(_problem(), seed=29).set_strategy(DirectStrategy())
        asynchronous.set_termination(Termination(max_gen(1)))
        asynchronous.set_async_evaluation_scheduler(
            AsyncEvaluationScheduler(_BatchEvaluator(), max_pending=2)
        )
        asynchronous.resolve_defaults()
        asynchronous._compile_plan()
        initial = _initialize(asynchronous)
        with pytest.warns(
            DeprecationWarning, match=r"execute_strategy_step.*Optimizer\.iterate"
        ):
            execute_strategy_step(asynchronous.strategy, initial, asynchronous)

        assert len(compiled) == len(contexts)
        assert len(compiled) >= 9
        assert all(context.problem is optimizer.problem for context in contexts[:7])
        assert all(context.problem is asynchronous.problem for context in contexts[7:])
        assert all(
            context.offered_runtime_capabilities == frozenset({"partial_feedback"})
            for context in contexts
        )
        assert all(
            context.initial_state_keys == OPTIMIZATION_STATE_INITIAL_KEYS
            and context._verification_enabled
            for context in contexts
        )
        assert _plan_shape(compiled[0]) == _plan_shape(compiled[1])
        assert _plan_shape(compiled[2]) == _plan_shape(compiled[3])
        assert _plan_shape(compiled[0]) == _plan_shape(compiled[4])
        assert _plan_shape(compiled[0]) == _plan_shape(compiled[5])
        assert _plan_shape(compiled[0]) == _plan_shape(compiled[6])
        for plan in (*compiled[:2], *compiled[4:7]):
            assert _plan_binding_ids(plan) == _plan_binding_ids(compiled[0])
            assert _plan_service_ids(plan) == _plan_service_ids(compiled[0])
        assert _plan_binding_ids(compiled[2]) == _plan_binding_ids(compiled[3])
        assert _plan_service_ids(compiled[2]) == _plan_service_ids(compiled[3])
        assert _plan_shape(compiled[-2]) == _plan_shape(compiled[-1])
        assert _plan_binding_ids(compiled[-2]) == _plan_binding_ids(compiled[-1])
        assert _plan_service_ids(compiled[-2]) == _plan_service_ids(compiled[-1])
    finally:
        default_runtime_registry.replace("sync", original_sync)
