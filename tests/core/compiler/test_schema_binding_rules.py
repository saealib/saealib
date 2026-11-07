"""Tests for schema binding and lossless adapter resolution."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pytest
from pymoo.algorithms.soo.nonconvex.ga import GA

from saealib.algorithms.pymoo_algorithm import PymooAlgorithm
from saealib.core.compiler import (
    CompileContext,
    Compiler,
    ComponentGraph,
    ComponentNode,
    DataEdge,
    DiagnosticBag,
    NodeRef,
    ResolutionResult,
    RuleContext,
    Severity,
)
from saealib.core.compiler.adapters import (
    ADAPTER_CATEGORIES,
    Adapter,
    AdapterRegistry,
    LosslessAdapterRule,
)
from saealib.core.compiler.schema_rules import SchemaBindingRule
from saealib.core.contracts import (
    MANY,
    ONE,
    ComponentContract,
    Contained,
    DataSpec,
    Fixed,
    PortContract,
    PortDirection,
    PortSpec,
    RepresentationSpec,
    ServiceRequirement,
    Var,
)
from saealib.core.state import OPTIMIZATION_STATE_INITIAL_KEYS
from saealib.exceptions import ValidationError
from saealib.execution.evaluator import SerialEvaluator
from saealib.execution.scheduler import AsyncEvaluationScheduler
from saealib.islands import IslandModel
from saealib.optimizer import Optimizer
from saealib.problem import Problem
from saealib.space import ObjectSpace, VectorSpace
from saealib.strategies.direct import DirectStrategy
from saealib.surrogate.rbf import RBFSurrogate
from saealib.surrogate.rbf_kernels import GaussianKernel


@dataclass(frozen=True)
class _Endpoint:
    data: DataSpec
    direction: PortDirection
    cardinality: str = MANY
    services: tuple[ServiceRequirement, ...] = ()

    def contract(self) -> ComponentContract:
        port = PortSpec(
            name="value",
            direction=self.direction,
            data=self.data,
            cardinality=self.cardinality,
            required_services=self.services,
        )
        role = (
            PortContract(outputs=(port,))
            if self.direction is PortDirection.OUTPUT
            else PortContract(inputs=(port,))
        )
        return ComponentContract(ports={"role": role})


def _edge_graph(source: DataSpec, target: DataSpec) -> ComponentGraph:
    return ComponentGraph(
        nodes=(
            ComponentNode(
                component_id="producer",
                component=_Endpoint(
                    data=source,
                    direction=PortDirection.OUTPUT,
                    cardinality=ONE,
                ),
            ),
            ComponentNode(
                component_id="consumer",
                component=_Endpoint(data=target, direction=PortDirection.INPUT),
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
        entry_points=(NodeRef(component_id="producer", role="role"),),
    )


def _problem() -> Problem:
    return Problem(
        func=lambda x: float(np.sum(x)),
        dim=2,
        n_obj=1,
        direction=np.array([-1.0]),
        lb=[0.0, 0.0],
        ub=[1.0, 1.0],
    )


def _compile_actual_optimizer(optimizer: Optimizer):
    optimizer._resolve_defaults()
    graph = cast(Any, optimizer.strategy).build_graph(optimizer)
    plan = Compiler().compile(
        graph,
        CompileContext(
            space=optimizer.problem.space,
            problem=optimizer.problem,
            initial_state_keys=OPTIMIZATION_STATE_INITIAL_KEYS,
        ),
    )
    return graph, plan


def test_actual_default_graph_uses_declared_surrogate_input_transform() -> None:
    problem = _problem()
    _, plan = _compile_actual_optimizer(Optimizer(problem))

    assert any(
        edge.source.role == "state"
        and edge.target.role == "predictor"
        and edge.target_port == "candidates"
        and tuple(transform.name for transform in edge.transforms)
        == ("encode_features",)
        for edge in plan.graph.data_edges
    )
    assert not [
        diagnostic
        for diagnostic in plan.diagnostics
        if diagnostic.severity.value == "error"
    ]
    assert any(
        tuple(transform.name for transform in edge.transforms) == ("encode_features",)
        for edge in plan.graph.data_edges
    )
    assert not any(
        getattr(item, "adapter_name", None) == "feature_encoder"
        for item in plan.inserted_adapters
    )


def test_object_space_does_not_activate_vector_profile_adapter() -> None:
    problem = _problem()
    graph, _ = _compile_actual_optimizer(Optimizer(problem))
    object_space = ObjectSpace(RepresentationSpec(kind="permutation"))
    plan = Compiler().compile(
        graph,
        CompileContext(space=object_space, problem=problem),
    )

    assert not any(
        diagnostic.code == "incompatible_representation"
        for diagnostic in plan.diagnostics
    )
    assert not any(
        getattr(item, "adapter_name", None) == "dense_numeric_view"
        for item in plan.inserted_adapters
    )


def test_feature_encoder_is_never_automatically_inserted() -> None:
    adapter = Adapter(
        name="feature_encoder",
        category="feature_encoder",
        source=DataSpec(kind="Population"),
        target=DataSpec(kind="FeatureBatch"),
        lossless=True,
        auto_insertable=True,
    )
    graph = _edge_graph(adapter.source, adapter.target)
    plan = Compiler(adapter_registry=AdapterRegistry((adapter,))).compile(graph)

    assert not any(
        diagnostic.code == "incompatible_representation"
        for diagnostic in plan.diagnostics
    )
    assert not plan.inserted_adapters
    assert all("feature_encoder" not in node.component_id for node in plan.graph.nodes)


def test_two_automatic_adapter_candidates_are_ambiguous_and_not_inserted() -> None:
    adapters = tuple(
        Adapter(
            name=name,
            source=DataSpec(kind="Population"),
            target=DataSpec(kind="FeatureBatch"),
            lossless=True,
            auto_insertable=True,
        )
        for name in ("population_view_a", "population_view_b")
    )
    plan = Compiler(adapter_registry=AdapterRegistry(adapters)).compile(
        _edge_graph(adapters[0].source, adapters[0].target)
    )

    findings = [
        diagnostic
        for diagnostic in plan.diagnostics
        if diagnostic.code == "ambiguous_adapter"
    ]
    assert len(findings) == 1
    assert findings[0].severity is Severity.WARNING
    assert "population_view_a" in findings[0].message
    assert "population_view_b" in findings[0].message
    assert not plan.inserted_adapters
    assert not any(
        diagnostic.code == "incompatible_port" for diagnostic in plan.diagnostics
    )
    assert tuple(node.component_id for node in plan.graph.nodes) == (
        "producer",
        "consumer",
    )


@pytest.mark.parametrize(
    ("name", "category"),
    (
        ("ordinary_transform", "text_embedding"),
        ("ordinary_transform", "graph_embedding"),
        ("ordinal", "ordinal_encoding"),
        ("one_hot", "one_hot_encoding"),
        ("latent", "latent_mapping"),
        ("approximate_distance", "approximate_distance"),
    ),
)
def test_explicit_only_adapter_categories_are_not_auto_insertable(
    name: str, category: str
) -> None:
    adapter = Adapter(
        name=name,
        category=category,
        source=DataSpec(kind="Population"),
        target=DataSpec(kind="FeatureBatch"),
        lossless=True,
        auto_insertable=True,
    )

    assert not adapter.eligible_for_automatic_insertion


def test_adapter_category_controls_eligibility_not_adapter_name() -> None:
    explicit = Adapter(
        name="ordinary_transform",
        category="feature_encoder",
        source=DataSpec(kind="Population"),
        target=DataSpec(kind="FeatureBatch"),
        lossless=True,
        auto_insertable=True,
    )
    automatic = Adapter(
        name="approximate_free_dense_view",
        category="lossless_view",
        source=DataSpec(kind="Population"),
        target=DataSpec(kind="FeatureBatch"),
        lossless=True,
        auto_insertable=True,
    )

    assert not explicit.eligible_for_automatic_insertion
    assert automatic.eligible_for_automatic_insertion


def test_adapter_category_must_be_registered() -> None:
    assert ADAPTER_CATEGORIES.names() == (
        "identity",
        "lossless_view",
        "batch_buffering",
        "immutable_clone",
        "partial_feedback_accumulation",
        "text_embedding",
        "graph_embedding",
        "ordinal_encoding",
        "one_hot_encoding",
        "latent_mapping",
        "approximate_distance",
        "feature_encoder",
    )
    with pytest.raises(ValidationError, match="Unknown adapter category"):
        Adapter(
            name="unknown_category",
            category="not_registered",
            source=DataSpec(kind="Population"),
            target=DataSpec(kind="FeatureBatch"),
            lossless=True,
            auto_insertable=True,
        )


def test_schema_variables_are_isolated_per_node_without_wrapping_components() -> None:
    variable = Var(name="representation")
    producer_a = _Endpoint(
        data=DataSpec(kind="Population", bindings={"representation": variable}),
        direction=PortDirection.OUTPUT,
        cardinality=ONE,
    )
    producer_b = _Endpoint(
        data=DataSpec(kind="Population", bindings={"representation": variable}),
        direction=PortDirection.OUTPUT,
        cardinality=ONE,
    )
    graph = ComponentGraph(
        nodes=(
            ComponentNode(
                component_id="producer_a",
                component=producer_a,
            ),
            ComponentNode(
                component_id="consumer_a",
                component=_Endpoint(
                    data=DataSpec(
                        kind="Population",
                        bindings={"representation": Fixed(value="real")},
                    ),
                    direction=PortDirection.INPUT,
                ),
            ),
            ComponentNode(
                component_id="producer_b",
                component=producer_b,
            ),
            ComponentNode(
                component_id="consumer_b",
                component=_Endpoint(
                    data=DataSpec(
                        kind="Population",
                        bindings={"representation": Fixed(value="integer")},
                    ),
                    direction=PortDirection.INPUT,
                ),
            ),
        ),
        data_edges=(
            DataEdge(
                source=NodeRef(component_id="producer_a", role="role"),
                target=NodeRef(component_id="consumer_a", role="role"),
                source_port="value",
                target_port="value",
            ),
            DataEdge(
                source=NodeRef(component_id="producer_b", role="role"),
                target=NodeRef(component_id="consumer_b", role="role"),
                source_port="value",
                target_port="value",
            ),
        ),
        entry_points=(
            NodeRef(component_id="producer_a", role="role"),
            NodeRef(component_id="producer_b", role="role"),
        ),
    )
    plan = Compiler().compile(graph)

    assert not [
        diagnostic
        for diagnostic in plan.diagnostics
        if diagnostic.severity.value == "error"
    ]
    assert plan.graph.node_by_id("producer_a").component is producer_a
    assert plan.graph.node_by_id("producer_b").component is producer_b
    names = [
        plan.graph.node_by_id(node_id)
        .contract.ports["role"]
        .outputs[0]
        .data.bindings["representation"]
        for node_id in ("producer_a", "producer_b")
    ]
    assert all(isinstance(binding, Var) for binding in names)
    assert [binding.name for binding in names if isinstance(binding, Var)] == [
        "representation",
        "representation",
    ]


def _service_chains(
    q2_service: str = "BoundsService",
) -> ComponentGraph:
    """Two independent chains with a shared-name schema variable."""
    nodes: list[ComponentNode] = []
    edges: list[DataEdge] = []
    for index, fixed in ((1, "A"), (2, "B")):
        nodes.append(
            ComponentNode(
                component_id=f"p{index}",
                component=_Endpoint(
                    data=DataSpec(
                        kind="Population",
                        bindings={"representation": Fixed(value=fixed)},
                    ),
                    direction=PortDirection.OUTPUT,
                    cardinality=ONE,
                ),
            )
        )
        nodes.append(
            ComponentNode(
                component_id=f"q{index}",
                component=_Endpoint(
                    data=DataSpec(
                        kind="Population",
                        bindings={"representation": Var(name="representation")},
                    ),
                    direction=PortDirection.INPUT,
                    cardinality=ONE,
                    services=(
                        ServiceRequirement(
                            name="BoundsService" if index == 1 else q2_service
                        ),
                    ),
                ),
            )
        )
        edges.append(
            DataEdge(
                source=NodeRef(component_id=f"p{index}", role="role"),
                target=NodeRef(component_id=f"q{index}", role="role"),
                source_port="value",
                target_port="value",
            )
        )
    graph = ComponentGraph(
        nodes=tuple(nodes),
        data_edges=tuple(edges),
        entry_points=(
            NodeRef(component_id="p1", role="role"),
            NodeRef(component_id="p2", role="role"),
        ),
    )
    return graph


def test_unresolved_services_keep_schema_variables_of_independent_nodes_separate() -> (
    None
):
    graph = _service_chains()

    plan = Compiler().compile(graph, CompileContext())

    assert [diagnostic.code for diagnostic in plan.diagnostics] == [
        "unresolved_service",
        "unresolved_service",
    ]
    assert not [
        diagnostic
        for diagnostic in plan.diagnostics
        if diagnostic.code == "incompatible_port"
    ]


def test_resolvable_services_keep_schema_variables_of_independent_nodes_separate() -> (
    None
):
    graph = _service_chains()

    plan = Compiler().compile(
        graph,
        CompileContext(space=VectorSpace(dim=1, lb=[0.0], ub=[1.0])),
    )

    assert not plan.diagnostics


def test_mixed_services_keep_schema_variables_of_independent_nodes_separate() -> None:
    graph = _service_chains(q2_service="UnresolvableService")

    plan = Compiler().compile(
        graph,
        CompileContext(space=VectorSpace(dim=1, lb=[0.0], ub=[1.0])),
    )

    assert not [
        diagnostic
        for diagnostic in plan.diagnostics
        if diagnostic.code == "incompatible_port"
    ]
    assert [diagnostic.code for diagnostic in plan.diagnostics] == ["unknown_service"]


def test_genuinely_shared_schema_variable_conflict_is_still_an_error() -> None:
    shared = ComponentNode(
        component_id="m",
        component=_Endpoint(
            data=DataSpec(
                kind="Population",
                bindings={"representation": Var(name="representation")},
            ),
            direction=PortDirection.OUTPUT,
            cardinality=ONE,
        ),
    )
    consumers = tuple(
        ComponentNode(
            component_id=f"q{index}",
            component=_Endpoint(
                data=DataSpec(
                    kind="Population", bindings={"representation": Fixed(value=fixed)}
                ),
                direction=PortDirection.INPUT,
            ),
        )
        for index, fixed in ((1, "A"), (2, "B"))
    )
    graph = ComponentGraph(
        nodes=(shared, *consumers),
        data_edges=tuple(
            DataEdge(
                source=NodeRef(component_id="m", role="role"),
                target=NodeRef(component_id=f"q{index}", role="role"),
                source_port="value",
                target_port="value",
            )
            for index in (1, 2)
        ),
        entry_points=(NodeRef(component_id="m", role="role"),),
    )

    plan = Compiler().compile(graph)

    errors = [
        diagnostic
        for diagnostic in plan.diagnostics
        if diagnostic.severity is Severity.ERROR
    ]
    assert [diagnostic.code for diagnostic in errors] == ["incompatible_port"]


def test_unknown_schema_variable_and_unbound_variable_are_warnings() -> None:
    graph = ComponentGraph(
        nodes=(
            ComponentNode(
                component_id="producer",
                component=_Endpoint(
                    data=DataSpec(
                        kind="Population",
                        bindings={
                            "future_variable": Var(name="future_variable"),
                            "representation": Var(name="representation"),
                        },
                    ),
                    direction=PortDirection.OUTPUT,
                    cardinality=ONE,
                ),
            ),
            ComponentNode(
                component_id="consumer",
                component=_Endpoint(
                    data=DataSpec(
                        kind="Population",
                        bindings={"representation": Contained(values=frozenset({"a"}))},
                    ),
                    direction=PortDirection.INPUT,
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
        entry_points=(NodeRef(component_id="producer", role="role"),),
    )

    plan = Compiler().compile(graph)

    codes = [diagnostic.code for diagnostic in plan.diagnostics]
    assert sorted(codes) == ["schema_variable_unbound", "unknown_schema_variable"]
    assert all(
        diagnostic.severity is Severity.WARNING for diagnostic in plan.diagnostics
    )
    unknown = next(d for d in plan.diagnostics if d.code == "unknown_schema_variable")
    assert "future_variable" in unknown.message


def _rule_order_signature(
    order: tuple[Any, ...],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    problem = _problem()
    graph = _edge_graph(DataSpec(kind="Population"), DataSpec(kind="FeatureBatch"))
    current = graph
    diagnostics = DiagnosticBag()
    for rule in order:
        result = rule.apply(
            RuleContext(
                graph=current,
                compile_context=CompileContext(space=problem.space, problem=problem),
                diagnostics=diagnostics,
            )
        )
        assert isinstance(result, ResolutionResult)
        current = result.graph
        diagnostics.extend(result.diagnostics)
    return (
        tuple(node.component_id for node in current.nodes),
        tuple(diagnostic.code for diagnostic in diagnostics),
    )


def test_schema_and_adapter_rule_order_is_deterministic() -> None:
    rules = (SchemaBindingRule(), LosslessAdapterRule())
    expected = None
    for seed in range(8):
        shuffled = list(rules)
        random.Random(seed).shuffle(shuffled)
        signature = _rule_order_signature(tuple(shuffled))
        if expected is None:
            expected = signature
        else:
            assert signature == expected
    assert expected == (
        (
            "producer",
            "consumer",
            "__adapter_dense_numeric_view_producer_role_value_consumer_role_value",
        ),
        (),
    )


def test_actual_graph_configurations_have_no_compile_errors() -> None:
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
            Optimizer(problem).set_strategy(DirectStrategy()),
            Optimizer(_problem()).set_strategy(DirectStrategy()),
        ],
        "pymoo": [
            Optimizer(problem)
            .set_algorithm(PymooAlgorithm(GA(pop_size=4)))
            .set_strategy(DirectStrategy())
        ],
    }
    IslandModel(configurations["island"])

    errors_by_configuration: dict[str, tuple[str, ...]] = {}
    for name, optimizers in configurations.items():
        errors: list[str] = []
        for optimizer in optimizers:
            _, plan = _compile_actual_optimizer(optimizer)
            errors.extend(
                diagnostic.code
                for diagnostic in plan.diagnostics
                if diagnostic.severity.value == "error"
            )
        errors_by_configuration[name] = tuple(errors)

    assert errors_by_configuration == {
        "default": (),
        "surrogate": (),
        "async": (),
        "island": (),
        "pymoo": (),
    }
