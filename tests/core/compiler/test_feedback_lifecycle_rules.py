"""Tests for lifecycle verification and compile-time accumulation."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
from pymoo.algorithms.soo.nonconvex.ga import GA

from saealib.algorithms.pymoo_algorithm import PymooAlgorithm
from saealib.core.compiler import (
    CompileContext,
    Compiler,
    ComponentGraph,
    ComponentNode,
    DataEdge,
    DiagnosticBag,
    FeedbackAccumulatorRule,
    NodeRef,
    ResolutionResult,
    RuleContext,
)
from saealib.core.compiler.adapters import (
    DEFAULT_ADAPTER_REGISTRY,
    LosslessAdapterRule,
)
from saealib.core.compiler.dataflow_rules import StructuredDataflowRule
from saealib.core.contracts import (
    MANY,
    ComponentContract,
    DataSpec,
    FeedbackContract,
    LifecycleContract,
    PortContract,
    PortDirection,
    PortSpec,
)
from saealib.core.contracts.feedback import COMPLETE_BATCH, PARTIAL_ALLOWED
from saealib.core.state import FEEDBACK_RESULT, OPTIMIZATION_STATE_INITIAL_KEYS
from saealib.execution.evaluator import SerialEvaluator
from saealib.execution.runtime import default_runtime_registry
from saealib.execution.scheduler import AsyncEvaluationScheduler
from saealib.islands import IslandModel
from saealib.optimizer import Optimizer
from saealib.problem import Problem
from saealib.strategies.direct import DirectStrategy
from saealib.surrogate.rbf import RBFSurrogate
from saealib.surrogate.rbf_kernels import GaussianKernel


@dataclass(frozen=True)
class _Endpoint:
    data: DataSpec
    direction: PortDirection
    completion: str | None = None

    def contract(self) -> ComponentContract:
        port = PortSpec(
            name="value",
            direction=self.direction,
            data=self.data,
            cardinality=MANY,
        )
        role = (
            PortContract(outputs=(port,))
            if self.direction is PortDirection.OUTPUT
            else PortContract(inputs=(port,))
        )
        feedback = (
            None
            if self.completion is None
            else FeedbackContract(
                accepted_channels=frozenset({"true"}),
                completion=self.completion,
            )
        )
        return ComponentContract(
            ports={"role": role},
            lifecycle=LifecycleContract(feedback=feedback),
        )


def _problem() -> Problem:
    return Problem(
        func=lambda x: float(np.sum(np.asarray(x) ** 2)),
        dim=2,
        n_obj=1,
        direction=np.array([-1.0]),
        lb=[-1.0, -1.0],
        ub=[1.0, 1.0],
    )


def _feedback_graph(completion: str) -> ComponentGraph:
    return ComponentGraph(
        nodes=(
            ComponentNode(
                component_id="producer",
                component=_Endpoint(
                    data=DataSpec(kind="FeedbackBatch"),
                    direction=PortDirection.OUTPUT,
                ),
            ),
            ComponentNode(
                component_id="consumer",
                component=_Endpoint(
                    data=DataSpec(kind="FeedbackBatch"),
                    direction=PortDirection.INPUT,
                    completion=completion,
                ),
            ),
        ),
        data_edges=(
            DataEdge(
                source=NodeRef(component_id="producer", role="role"),
                target=NodeRef(component_id="consumer", role="role"),
                source_port="value",
                target_port="value",
            ),
        ),
        entry_points=(NodeRef(component_id="producer"),),
    )


def _compile_actual(optimizer: Optimizer):
    optimizer._resolve_defaults()
    graph = cast(Any, optimizer.strategy).build_graph(optimizer)
    plan = Compiler().compile(
        graph,
        CompileContext(
            space=optimizer.problem.space,
            problem=optimizer.problem,
            offered_runtime_capabilities=default_runtime_registry.offered_capabilities(
                optimizer
            ),
            initial_state_keys=OPTIMIZATION_STATE_INITIAL_KEYS,
        ),
    )
    return graph, plan


def test_async_graph_inserts_accumulator_and_describes_it() -> None:
    optimizer = Optimizer(_problem()).set_async_evaluation_scheduler(
        AsyncEvaluationScheduler(SerialEvaluator())
    )

    _, plan = _compile_actual(optimizer)

    signatures = {
        (
            edge.source.role or "",
            edge.source_port,
            edge.target.role or "",
            edge.target_port,
            tuple(item.name for item in edge.transforms),
        )
        for edge in plan.graph.data_edges
    }
    assert signatures == {
        ("state", "value", "predictor", "archive", ()),
        (
            "state",
            "value",
            "feedback_builder",
            "evaluation",
            ("extract_evaluation_observations",),
        ),
        ("state", "value", "feedback_builder", "evaluated_indices", ()),
        (
            "state",
            "value",
            "evaluator",
            "genomes",
            ("select_requested_candidates", "extract_genomes"),
        ),
        ("state", "value", "evaluation_planner", "acquisition", ()),
        ("state", "value", "state", "value", ("apply_observations",)),
        ("state", "value", "predictor", "candidates", ("encode_features",)),
        ("state", "value", "evaluation_planner", "candidates", ()),
        ("state", "value", "feedback_builder", "candidates", ()),
        ("state", "value", "acquisition", "prediction", ()),
        ("state", "value", "feedback_builder", "prediction", ()),
        ("proposer", "genomes", "state", "value", ()),
        ("acquisition", "scores", "state", "value", ()),
        ("evaluator", "observations", "state", "value", ()),
        ("evaluation_planner", "evaluation_requests", "state", "value", ()),
        ("feedback_builder", "feedback", "state", "value", ()),
        ("predictor", "prediction", "state", "value", ()),
        ("predictor", "output", "feedback_consumer", "feedback", ()),
        ("state", "value", "predictor", "input", ()),
    }
    assert len(plan.graph.data_edges) == len(signatures)
    assert plan.graph.data_edges
    assert not [
        diagnostic
        for diagnostic in plan.diagnostics
        if diagnostic.severity.value == "error"
    ]
    assert {
        (
            getattr(item, "adapter_name", None),
            getattr(item, "source_path").role,
            getattr(item, "source_path").port,
            getattr(item, "target_path").role,
            getattr(item, "target_path").port,
        )
        for item in plan.inserted_adapters
    } == {
        (
            "feedback_accumulator",
            "state",
            "value",
            "feedback_consumer",
            "feedback",
        ),
    }
    assert "feedback_accumulator" in plan.describe()
    assert any(
        node.component_id.startswith("__adapter_feedback_accumulator_")
        for node in plan.graph.nodes
    )


def test_stage_internal_feedback_edges_are_claimed() -> None:
    optimizer = Optimizer(_problem()).set_async_evaluation_scheduler(
        AsyncEvaluationScheduler(SerialEvaluator())
    )
    optimizer._resolve_defaults()
    raw_graph = cast(Any, optimizer.strategy).build_graph(optimizer)
    compile_context = CompileContext(
        space=optimizer.problem.space,
        problem=optimizer.problem,
        offered_runtime_capabilities=frozenset({"partial_feedback"}),
        initial_state_keys=OPTIMIZATION_STATE_INITIAL_KEYS,
    )
    flow = StructuredDataflowRule().apply(
        RuleContext(
            graph=raw_graph,
            compile_context=compile_context,
            diagnostics=DiagnosticBag(),
        )
    )
    graph = flow.graph
    result = FeedbackAccumulatorRule().apply(
        RuleContext(
            graph=graph,
            compile_context=compile_context,
            diagnostics=DiagnosticBag(),
        )
    )

    new_edges = tuple(
        edge for edge in result.graph.data_edges if edge not in graph.data_edges
    )
    assert new_edges
    claimed_edges = {claim.key for claim in result.claims if claim.kind == "data_edge"}

    def edge_key(edge: DataEdge) -> str:
        def reference(ref: NodeRef) -> str:
            return ref.component_id + (f"[{ref.role}]" if ref.role else "")

        base = (
            f"{reference(edge.source)}.{edge.source_port}->"
            f"{reference(edge.target)}.{edge.target_port}"
        )
        return base + "|" + ",".join(item.name for item in edge.transforms)

    assert all(edge_key(edge) in claimed_edges for edge in new_edges)
    accumulators = tuple(
        node
        for node in result.graph.nodes
        if getattr(getattr(node.component, "adapter", None), "name", None)
        == "feedback_accumulator"
    )
    assert len(accumulators) == 1
    state_flow = next(
        node
        for node in graph.nodes
        if node.role == "state"
        and getattr(node.component, "state_key", None) == FEEDBACK_RESULT
        and node.component_id.endswith("__flow")
    )
    expected_sources = {(state_flow.component_id, "state", "value")}
    input_edges = tuple(
        edge
        for edge in new_edges
        if edge.target.role == "predictor" and edge.target_port == "input"
    )
    output_edges = tuple(
        edge
        for edge in new_edges
        if edge.source.role == "predictor" and edge.source_port == "output"
    )
    assert len(input_edges) == len(output_edges) == 1
    assert {
        (edge.source.component_id, edge.source.role, edge.source_port)
        for edge in input_edges
    } == expected_sources
    assert {edge.target.component_id for edge in input_edges} == {
        node.component_id for node in accumulators
    }
    assert {edge.source.component_id for edge in output_edges} == {
        node.component_id for node in accumulators
    }
    assert {
        (
            edge.target.component_id,
            edge.target.role,
            edge.target_port,
            tuple(item.name for item in edge.transforms),
        )
        for edge in output_edges
    } == {("tell__algorithm", "feedback_consumer", "feedback", ())}


def test_partial_allowed_consumer_does_not_get_accumulator() -> None:
    plan = Compiler().compile(
        _feedback_graph(PARTIAL_ALLOWED),
        CompileContext(offered_runtime_capabilities=frozenset({"partial_feedback"})),
    )

    assert not plan.inserted_adapters
    assert not any(
        node.component_id.startswith("__adapter_feedback_accumulator_")
        for node in plan.graph.nodes
    )


def test_accumulator_registry_matcher_requires_both_runtime_and_consumer() -> None:
    graph = _feedback_graph(COMPLETE_BATCH)
    producer = graph.node_by_id("producer")
    consumer = graph.node_by_id("consumer")
    source_port = producer.contract.ports["role"].outputs[0]
    target_port = consumer.contract.ports["role"].inputs[0]

    partial = DEFAULT_ADAPTER_REGISTRY.candidates(
        source_port.data,
        target_port.data,
        compile_context=CompileContext(
            offered_runtime_capabilities=frozenset({"partial_feedback"})
        ),
        source_node=producer,
        target_node=consumer,
        source_port=source_port,
        target_port=target_port,
        graph=graph,
    )
    assert tuple(item.name for item in partial) == ("feedback_accumulator",)

    plan = Compiler().compile(
        graph,
        CompileContext(offered_runtime_capabilities=frozenset({"partial_feedback"})),
    )
    assert [getattr(item, "adapter_name", None) for item in plan.inserted_adapters] == [
        "feedback_accumulator"
    ]
    assert not [
        diagnostic
        for diagnostic in plan.diagnostics
        if diagnostic.severity.value == "error"
    ]

    allowed = _feedback_graph(PARTIAL_ALLOWED)
    partial_target = allowed.node_by_id("consumer")
    partial_candidates = DEFAULT_ADAPTER_REGISTRY.candidates(
        source_port.data,
        partial_target.contract.ports["role"].inputs[0].data,
        compile_context=CompileContext(
            offered_runtime_capabilities=frozenset({"partial_feedback"})
        ),
        source_node=allowed.node_by_id("producer"),
        target_node=partial_target,
        source_port=source_port,
        target_port=partial_target.contract.ports["role"].inputs[0],
        graph=allowed,
    )
    assert not partial_candidates


def test_pymoo_complete_batch_partial_runtime_keeps_j8_diagnostic() -> None:
    optimizer = Optimizer(_problem()).set_algorithm(
        PymooAlgorithm(GA(pop_size=4), allow_partial_tell=False)
    )
    optimizer.set_async_evaluation_scheduler(
        AsyncEvaluationScheduler(SerialEvaluator())
    )
    optimizer._resolve_defaults()

    assert [diagnostic.code for diagnostic in optimizer.contract_diagnostics()] == [
        "pymoo_partial_feedback_unsupported"
    ]


def test_sync_graph_does_not_insert_accumulator() -> None:
    graph, plan = _compile_actual(Optimizer(_problem()))

    assert all(
        "partial_feedback" not in node.contract.execution.offered_runtime_capabilities
        for node in graph.nodes
    )
    assert not any(
        getattr(item, "adapter_name", None) == "feedback_accumulator"
        for item in plan.inserted_adapters
    )


def test_resolution_rule_order_is_confluent_for_lifecycle_and_adapters() -> None:
    optimizer = Optimizer(_problem()).set_async_evaluation_scheduler(
        AsyncEvaluationScheduler(SerialEvaluator())
    )
    optimizer._resolve_defaults()
    graph = cast(Any, optimizer.strategy).build_graph(optimizer)
    context = CompileContext(
        space=optimizer.problem.space,
        problem=optimizer.problem,
        initial_state_keys=OPTIMIZATION_STATE_INITIAL_KEYS,
    )

    signatures: list[tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]] = []
    for seed in range(8):
        rules = [FeedbackAccumulatorRule(), LosslessAdapterRule()]
        random.Random(seed).shuffle(rules)
        current = graph
        diagnostics = DiagnosticBag()
        for rule in rules:
            result = rule.apply(
                RuleContext(
                    graph=current,
                    compile_context=context,
                    diagnostics=diagnostics,
                )
            )
            assert isinstance(result, ResolutionResult)
            current = result.graph
            diagnostics.extend(result.diagnostics)
        signatures.append(
            (
                tuple(sorted(node.component_id for node in current.nodes)),
                tuple(sorted(str(edge) for edge in current.data_edges)),
                tuple(sorted(diagnostic.code for diagnostic in diagnostics)),
            )
        )
    assert all(signature == signatures[0] for signature in signatures)


def test_five_actual_configurations_have_zero_compile_errors() -> None:
    problem = _problem()
    configurations: dict[str, list[Optimizer]] = {
        "default": [Optimizer(problem)],
        "surrogate": [Optimizer(problem).set_surrogate(RBFSurrogate(GaussianKernel()))],
        "async": [
            Optimizer(problem).set_async_evaluation_scheduler(
                AsyncEvaluationScheduler(SerialEvaluator())
            )
        ],
        "island": [
            Optimizer(_problem()).set_strategy(DirectStrategy()),
            Optimizer(_problem()).set_strategy(DirectStrategy()),
        ],
        "pymoo": [
            Optimizer(problem)
            .set_algorithm(PymooAlgorithm(GA(pop_size=4)))
            .set_strategy(DirectStrategy())
        ],
    }
    IslandModel(configurations["island"])

    for optimizers in configurations.values():
        for optimizer in optimizers:
            _, plan = _compile_actual(optimizer)
            assert not [
                diagnostic
                for diagnostic in plan.diagnostics
                if diagnostic.severity.value == "error"
            ]
