"""Tests for the stage-to-graph bridge."""

from __future__ import annotations

from typing import Any, TypeAlias, cast

import numpy as np

import saealib.core.graph_builder as graph_builder
from saealib import (
    GA,
    CrossoverBLXAlpha,
    MutationUniform,
    SequentialSelection,
    TruncationSelection,
)
from saealib.core.compiler import CompileContext, Compiler, RuleContext
from saealib.core.compiler.diagnostics import DiagnosticBag
from saealib.core.compiler.graph import IdentityRule, NodeRef, ReachabilityRule
from saealib.core.compiler.regions import RegionNode
from saealib.core.compiler.structured import StructuredGraph
from saealib.core.contracts import (
    MANY,
    ComponentContract,
    DataSpec,
    PartSpec,
    PortContract,
    PortDirection,
    PortSpec,
    StateContract,
)
from saealib.core.graph_builder import (
    StageContractNodeAdapter,
    StageNodeAdapter,
    StagePartNodeAdapter,
    build_component_graph,
    build_decomposed_component_graph,
)
from saealib.core.state import (
    OPTIMIZATION_STATE_INITIAL_KEYS,
    SURROGATES_DEFAULT,
)
from saealib.execution.evaluator import SerialEvaluator
from saealib.optimizer import Optimizer
from saealib.pipeline import Pipeline, Stage
from saealib.problem import Problem
from saealib.stages import SurrogatePredictStage
from saealib.strategies.base import OptimizationStrategy
from saealib.strategies.direct import DirectStrategy, SteadyStateStrategy
from saealib.strategies.gb import GenerationBasedStrategy
from saealib.strategies.ib import IndividualBasedStrategy
from saealib.strategies.ps import PreSelectionStrategy

_EdgeSignature: TypeAlias = tuple[str, str, str, str, str, str, tuple[str, ...]]


class _ContractComponent:
    """A no-op component sufficient for graph construction."""

    def __init__(self, *, exports: tuple = (), ports: dict | None = None) -> None:
        self._contract = ComponentContract(
            ports=ports or {}, state=StateContract(exports=exports)
        )

    def contract(self) -> ComponentContract:
        return self._contract


def _port(name: str, direction: PortDirection, kind: str = "Population") -> PortSpec:
    return PortSpec(
        name=name,
        direction=direction,
        data=DataSpec(kind=kind),
        cardinality=MANY,
    )


def _surrogate_manager_contract() -> ComponentContract:
    return ComponentContract(
        ports={
            "fitter": PortContract(inputs=(_port("archive", PortDirection.INPUT),)),
            "predictor": PortContract(
                inputs=(
                    _port("candidates", PortDirection.INPUT, "FeatureBatch"),
                    _port("archive", PortDirection.INPUT),
                ),
                outputs=(
                    _port("prediction", PortDirection.OUTPUT, "SurrogatePrediction"),
                ),
            ),
        },
        state=StateContract(exports=(SURROGATES_DEFAULT,)),
    )


class _Provider:
    def __init__(self) -> None:
        self.algorithm = GA(
            crossover=CrossoverBLXAlpha(prob=0.9, alpha=0.4),
            mutation=MutationUniform(prob_var=0.1),
            parent_selection=SequentialSelection(),
            survivor_selection=TruncationSelection(),
        )
        self.evaluator = SerialEvaluator()
        self.surrogate_manager = _ContractComponent(
            exports=(SURROGATES_DEFAULT,),
            ports={
                "fitter": PortContract(inputs=(_port("archive", PortDirection.INPUT),)),
                "predictor": PortContract(
                    inputs=(
                        _port("candidates", PortDirection.INPUT, "FeatureBatch"),
                        _port("archive", PortDirection.INPUT),
                    ),
                    outputs=(
                        _port(
                            "prediction", PortDirection.OUTPUT, "SurrogatePrediction"
                        ),
                    ),
                ),
            },
        )
        self.acquisition = _ContractComponent(
            ports={
                "acquisition": PortContract(
                    inputs=(
                        _port("prediction", PortDirection.INPUT, "SurrogatePrediction"),
                    ),
                    outputs=(_port("scores", PortDirection.OUTPUT, "RowPredicate"),),
                )
            }
        )
        self.cbmanager = None
        self.async_evaluation_scheduler = None
        self.evaluation_planner = None
        self.feedback_builder = None
        self.feedback_builder_explicit = False


def _strategies():
    return (
        DirectStrategy(),
        SteadyStateStrategy(),
        GenerationBasedStrategy(gen_ctrl=2),
        IndividualBasedStrategy(evaluation_ratio=0.5),
        PreSelectionStrategy(n_candidates=8, n_select=2),
    )


def test_builtins_and_graph_only_strategies_share_graph_pipeline_facade(
    monkeypatch,
):
    provider: Any = _Provider()
    builtins = _strategies()

    def fail_build_stages(*args, **kwargs):
        raise AssertionError("canonical graph construction called _build_stages")

    def fail_stage_materializer(*args, **kwargs):
        raise AssertionError("built-in graph construction used the Stage materializer")

    monkeypatch.setattr(
        graph_builder,
        "build_decomposed_component_graph_from_stages",
        fail_stage_materializer,
    )

    for strategy in builtins:
        monkeypatch.setattr(
            type(strategy), "_build_stages", fail_build_stages, raising=False
        )
        graph = strategy.build_graph(provider)
        pipeline = strategy.build_pipeline(provider)
        if isinstance(graph, StructuredGraph):
            if isinstance(strategy, GenerationBasedStrategy):
                assert [
                    operation.region.qualified_id for operation in graph.operations[1:]
                ] == [
                    "generation_based.surrogate_generations",
                    "generation_based.true_generation",
                ]
                assert pipeline.stages[0].name == "surrogate_fit"
                assert len(pipeline.stages) == 3
            else:
                # Direct and SteadyState are structured plans without regions
                assert not any(
                    isinstance(operation, RegionNode) for operation in graph.operations
                )
                assert pipeline.stages[0].name == "count_generation"
            assert not any(
                isinstance(node.component, StageNodeAdapter) for node in graph.nodes
            )
        else:
            assert [stage.name for stage in pipeline.stages] == [
                node.component.stage.name
                for node in graph.nodes
                if isinstance(node.component, StageNodeAdapter)
            ]

    class GraphOnlyStrategy(OptimizationStrategy):
        def build_graph(self, provider: Any):
            del provider
            return builtins[0].build_graph(_Provider())

    strategy = GraphOnlyStrategy()
    graph = strategy.build_graph(provider)
    pipeline = strategy.build_pipeline(provider)
    stages = cast(list[Stage], pipeline.stages)
    assert [stage.name for stage in stages] == [
        node.component.stage.name
        for node in graph.nodes
        if isinstance(node.component, StageNodeAdapter)
    ]


def _rule_diagnostics(rule, graph):
    result = rule.apply(
        RuleContext(
            graph=graph,
            compile_context=CompileContext(),
            diagnostics=DiagnosticBag(),
        )
    )
    return result.diagnostics


def test_each_strategy_builds_a_reachable_well_formed_graph():
    provider: Any = _Provider()

    for strategy in _strategies():
        graph = strategy.build_graph(provider)

        assert len(graph.well_formedness()) == 0
        assert len(_rule_diagnostics(IdentityRule(), graph)) == 0
        assert len(_rule_diagnostics(ReachabilityRule(), graph)) == 0
        root = (
            "generation_based.surrogate_fit"
            if isinstance(strategy, GenerationBasedStrategy)
            else "count_generation"
        )
        assert NodeRef(component_id=root) in graph.entry_points


def test_graph_nodes_match_actual_pipeline_and_preselection_has_no_top_k_stage():
    provider: Any = _Provider()

    for strategy in _strategies():
        pipeline = strategy.build_pipeline(provider)
        graph = strategy.build_graph(provider)
        stage_nodes = [
            node for node in graph.nodes if isinstance(node.component, StageNodeAdapter)
        ]

        if isinstance(graph, StructuredGraph):
            if isinstance(strategy, GenerationBasedStrategy):
                assert [
                    operation.region.qualified_id for operation in graph.operations[1:]
                ] == [
                    "generation_based.surrogate_generations",
                    "generation_based.true_generation",
                ]
                assert [
                    node.component_id
                    for node in graph.region_nodes[0].region.body.operations
                ] == [
                    "generation_based.surrogate_generations.count_generation",
                    "generation_based.surrogate_generations.ask",
                    "generation_based.surrogate_generations.surrogate_predict",
                    "generation_based.surrogate_generations.acquisition",
                    "generation_based.surrogate_generations.feedback",
                    "generation_based.surrogate_generations.tell",
                ]
            else:
                # Direct lowers to one region-less structured plan whose node
                # IDs are the bare declaration Stage names
                assert not any(
                    isinstance(operation, RegionNode) for operation in graph.operations
                )
                assert [node.component_id for node in graph.operations] == [
                    stage.name for stage in pipeline.stages
                ]
                assert graph.node_by_id("ask") is graph.nodes[1]
            assert all(
                not isinstance(node.component, StageNodeAdapter) for node in graph.nodes
            )
        else:
            assert [node.component.stage for node in stage_nodes] == list(
                pipeline.stages
            )
            assert all(
                node.component.stage.__class__.__name__ != "TopKSelectionStage"
                for node in stage_nodes
            )


def test_data_and_control_edges_are_distinct_and_archive_feedback_is_control_only():
    provider: Any = _Provider()
    raw_graph = DirectStrategy().build_graph(provider)
    plan = Compiler().compile(
        raw_graph,
        CompileContext(initial_state_keys=OPTIMIZATION_STATE_INITIAL_KEYS),
    )
    graph = plan.graph

    assert graph.data_edges
    assert graph.control_edges
    assert all(hasattr(edge, "source_port") for edge in graph.data_edges)
    assert not any(
        edge.source.component_id.split("__", 1)[0] == "archive_update"
        and edge.target.component_id.split("__", 1)[0] == "feedback"
        for edge in graph.data_edges
    )
    state_routes = {
        (
            getattr(
                graph.node_by_id(edge.source.component_id).component,
                "flow_origin",
                None,
            ),
            edge.target.component_id,
        )
        for edge in graph.data_edges
        if edge.source.role == "state" and edge.target.role != "state"
    }
    assert ("ask__proposer", "evaluation_plan__planner") in state_routes
    assert ("feedback__builder", "tell__algorithm") in state_routes


def _resolve_edge_port(graph, endpoint, port_name, direction):
    node = graph.node_by_id(endpoint.component_id)
    contracts = (
        ((endpoint.role, node.contract.ports[endpoint.role]),)
        if endpoint.role is not None and endpoint.role in node.contract.ports
        else ()
        if endpoint.role is not None
        else tuple(node.contract.ports.items())
    )
    matches = [
        port
        for _, contract in contracts
        for port in (*contract.inputs, *contract.outputs)
        if port.name == port_name and port.direction is direction
    ]
    assert len(matches) == 1
    return matches[0]


def test_all_strategy_data_edges_resolve_to_declared_directional_ports():
    common: tuple[_EdgeSignature, ...] = (
        (
            "evaluation_plan",
            "evaluation_planner",
            "evaluation_requests",
            "evaluation_submit",
            "evaluator",
            "genomes",
            ("select_requested_candidates", "extract_genomes"),
        ),
        (
            "feedback",
            "feedback_builder",
            "feedback",
            "tell",
            "feedback_consumer",
            "feedback",
            (),
        ),
    )
    direct_candidates: _EdgeSignature = (
        "ask",
        "proposer",
        "genomes",
        "evaluation_plan",
        "evaluation_planner",
        "candidates",
        (),
    )
    surrogate: tuple[_EdgeSignature, ...] = (
        (
            "ask",
            "proposer",
            "genomes",
            "surrogate_predict",
            "predictor",
            "candidates",
            ("encode_features",),
        ),
        (
            "surrogate_predict",
            "predictor",
            "prediction",
            "acquisition",
            "acquisition",
            "prediction",
            (),
        ),
        (
            "acquisition",
            "acquisition",
            "scores",
            "evaluation_plan",
            "evaluation_planner",
            "acquisition",
            (),
        ),
        (
            "surrogate_predict",
            "predictor",
            "prediction",
            "feedback",
            "feedback_builder",
            "prediction",
            (),
        ),
    )

    for strategy in _strategies():
        provider: Any = _Provider()
        raw_graph = strategy.build_graph(provider)
        plan = Compiler().compile(
            raw_graph,
            CompileContext(initial_state_keys=OPTIMIZATION_STATE_INITIAL_KEYS),
        )
        graph = plan.graph

        def stage_id(component_id: str) -> str:
            return component_id.split("__", 1)[0].rsplit(".", 1)[-1]

        actual: set[_EdgeSignature] = {
            (
                stage_id(edge.source.component_id),
                edge.source.role or "",
                edge.source_port,
                stage_id(edge.target.component_id),
                edge.target.role or "",
                edge.target_port,
                tuple(item.name for item in edge.transforms),
            )
            for edge in graph.data_edges
            if edge.source.role != "state" and edge.target.role != "state"
        }
        for edge in graph.data_edges:
            if edge.source.role != "state" or edge.target.role == "state":
                continue
            state_node = graph.node_by_id(edge.source.component_id)
            flow_origin = getattr(state_node.component, "flow_origin", None)
            if not isinstance(flow_origin, str):
                continue
            source = graph.node_by_id(flow_origin)
            producers = tuple(
                producer
                for producer in graph.data_edges
                if producer.target.component_id == state_node.component_id
                and producer.source.component_id == flow_origin
            )
            for producer in producers:
                actual.add(
                    (
                        stage_id(flow_origin),
                        source.role or "",
                        producer.source_port,
                        stage_id(edge.target.component_id),
                        edge.target.role or "",
                        edge.target_port,
                        tuple(
                            item.name
                            for item in (*producer.transforms, *edge.transforms)
                        ),
                    )
                )
        expected: set[_EdgeSignature] = set(common)
        if isinstance(
            strategy, (DirectStrategy, SteadyStateStrategy, GenerationBasedStrategy)
        ):
            expected.add(direct_candidates)
        if isinstance(
            strategy,
            (GenerationBasedStrategy, IndividualBasedStrategy, PreSelectionStrategy),
        ):
            # GenerationBased reads the surrogate planner's scores when it
            # builds the true-generation evaluation requests.
            expected.update(surrogate)
        assert expected == actual
        for edge in graph.data_edges:
            assert edge.source.role is not None
            assert edge.target.role is not None
            _resolve_edge_port(
                graph, edge.source, edge.source_port, PortDirection.OUTPUT
            )
            _resolve_edge_port(
                graph, edge.target, edge.target_port, PortDirection.INPUT
            )


def test_control_edges_preserve_reachability_without_data_edges():
    provider: Any = _Provider()

    for strategy in _strategies():
        graph = strategy.build_graph(provider)
        control_only = graph.__class__(
            nodes=graph.nodes,
            control_edges=graph.control_edges,
            state_bindings=graph.state_bindings,
            entry_points=graph.entry_points,
        )
        assert len(_rule_diagnostics(ReachabilityRule(), control_only)) == 0


def test_adapter_composes_held_contract_without_declaring_stage_state():
    class _Stage(Stage):
        name = "custom"

        def __init__(self) -> None:
            super().__init__()
            self.component = _ContractComponent(exports=(SURROGATES_DEFAULT,))

        def execute(self, state):
            return state

    contract = StageNodeAdapter(_Stage()).contract()

    assert contract.state.reads == ()
    assert contract.state.writes == ()
    assert contract.state.exports == (SURROGATES_DEFAULT,)


def test_stage_adapter_contract_is_a_construction_snapshot():
    class _CountingComponent(_ContractComponent):
        calls = 0

        def contract(self) -> ComponentContract:
            self.calls += 1
            return super().contract()

    class _Stage(Stage):
        name = "snapshot"

        def __init__(self) -> None:
            super().__init__()
            self.component = _CountingComponent(exports=(SURROGATES_DEFAULT,))

        def execute(self, state):
            return state

    stage = _Stage()
    adapter = StageNodeAdapter(stage)
    first = adapter.contract()
    cast(Any, stage).component = _ContractComponent()

    assert adapter.contract() is first
    assert first.state.exports == (SURROGATES_DEFAULT,)
    assert stage.component is not adapter._held[0].component
    assert cast(_CountingComponent, adapter._held[0].component).calls == 1


def test_stage_contract_adapter_captures_custom_stage_contract_once():
    class _Stage(Stage):
        name = "direct_snapshot"
        calls = 0

        def contract(self) -> ComponentContract:
            self.calls += 1
            return ComponentContract(state=StateContract(reads=(SURROGATES_DEFAULT,)))

        def execute(self, state):
            return state

    stage = _Stage()
    adapter = StageContractNodeAdapter(stage)
    first = adapter.contract()

    assert adapter.contract() is first
    assert stage.calls == 1


def test_named_surrogate_manager_uses_the_predictor_binding():
    class _NamedManager:
        def __init__(self) -> None:
            self.managers = {
                "cheap": _ContractComponent(exports=(SURROGATES_DEFAULT,)),
                "rich": _ContractComponent(exports=(SURROGATES_DEFAULT,)),
            }

        def contract(self) -> ComponentContract:
            return _surrogate_manager_contract()

    manager = _NamedManager()
    stage = SurrogatePredictStage(cast(Any, manager))
    graph = build_component_graph(Pipeline([stage]))
    names = {binding.state_key.name for binding in graph.state_bindings}

    assert names == {"surrogate_predict:predictor"}
    call = graph.node_by_id("surrogate_predict__predictor")
    assert call.component is manager


def test_surrogate_manager_attributes_keep_the_predictor_binding():
    class _Manager:
        def __init__(self) -> None:
            self.cheap_surrogate = _ContractComponent(exports=(SURROGATES_DEFAULT,))
            self.rich_surrogate = _ContractComponent(exports=(SURROGATES_DEFAULT,))

        def contract(self) -> ComponentContract:
            return _surrogate_manager_contract()

    manager = _Manager()
    graph = build_component_graph(Pipeline([SurrogatePredictStage(cast(Any, manager))]))

    assert {binding.state_key.name for binding in graph.state_bindings} == {
        "surrogate_predict:predictor"
    }
    assert graph.node_by_id("surrogate_predict__predictor").component is manager


def test_decomposed_surrogate_manager_uses_predictor_binding():
    class _NamedManager:
        def __init__(self) -> None:
            self.managers = {
                "cheap": _ContractComponent(exports=(SURROGATES_DEFAULT,)),
                "rich": _ContractComponent(exports=(SURROGATES_DEFAULT,)),
            }

        def contract(self) -> ComponentContract:
            return _surrogate_manager_contract()

    manager = _NamedManager()
    graph = build_decomposed_component_graph(
        Pipeline([SurrogatePredictStage(cast(Any, manager))])
    )

    assert {binding.state_key.name for binding in graph.state_bindings} == {
        "surrogate_predict:predictor"
    }
    assert all(
        binding.node.component_id != "surrogate_predict"
        for binding in graph.state_bindings
    )
    assert graph.node_by_id("surrogate_predict__predictor").component is manager


def test_structured_decomposition_uses_the_stage_as_execution_owner():
    class _Stage(Stage):
        name = "declared_contract_stage"

        def __init__(self) -> None:
            super().__init__()
            self._component = _ContractComponent(exports=(SURROGATES_DEFAULT,))

        def contract(self) -> ComponentContract:
            return ComponentContract(
                state=StateContract(reads=(SURROGATES_DEFAULT,)),
                parts=(
                    # This is intentionally a direct Stage contract assertion;
                    # the part is still independently represented below.
                    PartSpec(name="component", contract=self._component.contract()),
                ),
            )

        def execute(self, state):
            return state

    graph = build_decomposed_component_graph(Pipeline([_Stage()]))
    stage = graph.node_by_id("declared_contract_stage")

    assert stage.contract.state.reads == (SURROGATES_DEFAULT,)
    assert type(stage.component).__name__ == "StageContractNodeAdapter"
    stage_adapter = cast(StageContractNodeAdapter, stage.component)
    assert stage_adapter.stage.name == "declared_contract_stage"
    assert len(graph.nodes) == 2
    part = graph.node_by_id("declared_contract_stage__component")
    assert isinstance(part.component, StagePartNodeAdapter)
    assert not graph.data_edges


def test_structured_decomposition_does_not_scan_held_components(monkeypatch):
    pipeline = DirectStrategy().build_pipeline(cast(Any, _Provider()))
    calls = 0
    original = graph_builder._held_components

    def count_held(stage):
        nonlocal calls
        calls += 1
        return original(stage)

    monkeypatch.setattr(graph_builder, "_held_components", count_held)
    graph = build_decomposed_component_graph(pipeline)

    assert calls == 0
    assert graph.nodes


def test_structured_feedback_is_cross_node_and_control_is_not_data():
    graph = build_decomposed_component_graph(
        Pipeline(list(DirectStrategy().build_pipeline(cast(Any, _Provider())).stages))
    )

    builder_nodes = [node for node in graph.nodes if node.role == "feedback_builder"]
    consumer_nodes = [node for node in graph.nodes if node.role == "feedback_consumer"]
    assert len(builder_nodes) == len(consumer_nodes) == 1
    builder_edges = [
        edge
        for edge in graph.data_edges
        if edge.source.component_id == builder_nodes[0].component_id
        and edge.source_port == "feedback"
        and edge.target.role == "state"
    ]
    assert len(builder_edges) == 1
    state_id = builder_edges[0].target.component_id
    consumer_edges = [
        edge
        for edge in graph.data_edges
        if edge.source.component_id == state_id
        and edge.target.component_id == consumer_nodes[0].component_id
        and edge.target_port == "feedback"
    ]
    assert len(consumer_edges) == 1
    assert builder_nodes[0].component_id != consumer_nodes[0].component_id
    assert not any(
        item.source.component_id == item.target.component_id
        for item in graph.data_edges
    )

    control_only = graph.__class__(
        nodes=graph.nodes,
        control_edges=graph.control_edges,
        state_bindings=graph.state_bindings,
        entry_points=graph.entry_points,
    )
    assert len(_rule_diagnostics(ReachabilityRule(), control_only)) == 0

    partial_plan = Compiler().compile(
        graph,
        CompileContext(offered_runtime_capabilities=frozenset({"partial_feedback"})),
    )
    accumulator_edges = [
        item
        for item in partial_plan.graph.data_edges
        if item.source.component_id.startswith("__adapter_feedback_accumulator_")
        or item.target.component_id.startswith("__adapter_feedback_accumulator_")
    ]
    assert len(accumulator_edges) == 2
    assert (
        accumulator_edges[0].source.component_id
        != accumulator_edges[0].target.component_id
    )
    assert (
        accumulator_edges[1].source.component_id
        != accumulator_edges[1].target.component_id
    )
    assert {item.source.component_id for item in accumulator_edges} != {
        item.target.component_id for item in accumulator_edges
    }


def test_five_strategy_graphs_compile_without_errors_or_self_loops():
    problem = Problem(
        func=lambda x: float(np.sum(np.asarray(x) ** 2)),
        dim=2,
        n_obj=1,
        direction=np.array([-1.0]),
        lb=[-1.0, -1.0],
        ub=[1.0, 1.0],
    )

    for strategy in _strategies():
        optimizer = Optimizer(problem).set_strategy(strategy)
        optimizer.resolve_defaults()
        plan = optimizer._compile_plan()
        assert plan is not None
        errors = [item for item in plan.diagnostics if item.severity.value == "error"]
        assert errors == []
        assert all(
            edge.source.component_id != edge.target.component_id
            for edge in plan.graph.data_edges
        )
        compiled_graph = cast(StructuredGraph, plan.graph)
        flow_calls = cast(
            dict[str, dict[str, str]], compiled_graph.metadata["stage_flow_calls"]
        )
        for stage_id, calls in flow_calls.items():
            stage_adapter = cast(
                StageNodeAdapter, compiled_graph.node_by_id(stage_id).component
            )
            stage = stage_adapter.stage
            summary = stage.flow_summary()
            assert summary is not None
            declared = {call.name: call for call in summary.calls}
            assert set(calls) == set(declared)
            for call_name, component_id in calls.items():
                call_node = compiled_graph.node_by_id(component_id)
                assert call_node.component is getattr(
                    stage, declared[call_name].attribute
                )

        repeated_plan = optimizer._compile_plan()
        assert repeated_plan is not None
        node_ids = {node.component_id for node in plan.graph.nodes}
        repeated_ids = {node.component_id for node in repeated_plan.graph.nodes}
        assert node_ids == repeated_ids
        assert len(repeated_ids) == len(repeated_plan.graph.nodes)
