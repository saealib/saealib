from __future__ import annotations

import pytest

from saealib.core.compiler import (
    BranchRegion,
    Compiler,
    ComponentGraph,
    ComponentNode,
    ControlEdge,
    DataEdge,
    LoopRegion,
    NodeRef,
    RepeatRegion,
    Severity,
    StructuredGraph,
    lower_structured,
)
from saealib.core.component import Component
from saealib.core.contracts import (
    MANY,
    ONE,
    ComponentContract,
    DataSpec,
    Fixed,
    PortContract,
    PortDirection,
    PortSpec,
    StateContract,
)
from saealib.exceptions import ValidationError
from saealib.pipeline import Pipeline


class _Component:
    def __init__(
        self,
        name: str,
        *,
        output: str | None = None,
        input_kind: str | None = None,
        optional: bool = False,
        output_cardinality: str = MANY,
        input_cardinality: str = MANY,
    ) -> None:
        self.name = name
        self._output = output
        self._input_kind = input_kind
        self._optional = optional
        self._output_cardinality = output_cardinality
        self._input_cardinality = input_cardinality

    def contract(self) -> ComponentContract:
        outputs = (
            ()
            if self._output is None
            else (
                PortSpec(
                    name="value",
                    direction=PortDirection.OUTPUT,
                    data=DataSpec(kind=self._output),
                    cardinality=self._output_cardinality,
                ),
            )
        )
        inputs = (
            ()
            if self._input_kind is None
            else (
                PortSpec(
                    name="value",
                    direction=PortDirection.INPUT,
                    data=DataSpec(kind=self._input_kind),
                    cardinality=self._input_cardinality,
                    optional=self._optional,
                ),
            )
        )
        return ComponentContract(
            ports={"main": PortContract(inputs=inputs, outputs=outputs)}
        )


class _Condition:
    def contract(self):
        return StateContract()

    def evaluate(self, view):
        return True


def _compile(*components: Component):
    return Compiler().compile_pipeline(Pipeline(list(components)))


def _edge_tuple(edge: DataEdge) -> tuple[str, str, str, str]:
    return (
        edge.source.component_id,
        edge.source.role or "",
        edge.target.component_id,
        edge.target.role or "",
    )


def test_structured_dataflow_adds_one_compatible_edge() -> None:
    plan = _compile(
        _Component("producer", output="Population"),
        _Component("consumer", input_kind="Population"),
    )

    assert isinstance(plan.graph, StructuredGraph)
    assert [_edge_tuple(edge) for edge in plan.graph.data_edges] == [
        ("producer", "main", "consumer", "main")
    ]
    assert not plan.diagnostics


def test_required_input_without_compatible_upstream_is_reported() -> None:
    plan = _compile(_Component("consumer", input_kind="Population"))

    assert [diagnostic.code for diagnostic in plan.diagnostics] == ["unresolved_input"]


def test_unknown_target_kind_cannot_be_verified_and_is_not_double_reported() -> None:
    plan = _compile(
        _Component("producer", output="Population"),
        _Component("consumer", input_kind="MagicTensor"),
    )

    assert [diagnostic.code for diagnostic in plan.diagnostics] == [
        "cannot_verify_data_kind"
    ]
    assert plan.diagnostics[0].severity is Severity.WARNING


def test_unknown_producer_kind_cannot_supply_a_known_input() -> None:
    plan = _compile(
        _Component("producer", output="MagicTensor"),
        _Component("consumer", input_kind="Population"),
    )

    assert [diagnostic.code for diagnostic in plan.diagnostics] == [
        "cannot_verify_data_kind"
    ]
    assert plan.diagnostics[0].severity is Severity.WARNING


def test_unknown_kind_with_known_cardinality_mismatch_is_an_error() -> None:
    plan = _compile(
        _Component("producer", output="MagicTensor"),
        _Component("consumer", input_kind="Population", input_cardinality=ONE),
    )

    errors = [
        diagnostic
        for diagnostic in plan.diagnostics
        if diagnostic.severity is Severity.ERROR
    ]
    assert [diagnostic.code for diagnostic in errors] == ["unresolved_input"]
    assert not [
        diagnostic
        for diagnostic in plan.diagnostics
        if diagnostic.code == "cannot_verify_data_kind"
    ]


class _BoundEndpoint:
    def __init__(
        self,
        name: str,
        value: str,
        *,
        direction: PortDirection,
        cardinality: str,
    ) -> None:
        self.name = name
        self._data = DataSpec(
            kind="Population", bindings={"representation": Fixed(value=value)}
        )
        self._direction = direction
        self._cardinality = cardinality

    def contract(self) -> ComponentContract:
        port = PortSpec(
            name="value",
            direction=self._direction,
            data=self._data,
            cardinality=self._cardinality,
        )
        return ComponentContract(
            ports={
                "main": (
                    PortContract(outputs=(port,))
                    if self._direction is PortDirection.OUTPUT
                    else PortContract(inputs=(port,))
                )
            }
        )


def test_unknown_cardinality_with_known_schema_conflict_is_an_error() -> None:
    plan = _compile(
        _BoundEndpoint(
            "producer", "A", direction=PortDirection.OUTPUT, cardinality="many"
        ),
        _BoundEndpoint(
            "consumer", "B", direction=PortDirection.INPUT, cardinality=MANY
        ),
    )

    errors = [
        diagnostic
        for diagnostic in plan.diagnostics
        if diagnostic.severity is Severity.ERROR
    ]
    assert [diagnostic.code for diagnostic in errors] == ["unresolved_input"]
    assert not [
        diagnostic
        for diagnostic in plan.diagnostics
        if diagnostic.code == "cannot_verify_data_kind"
    ]


def test_unknown_cardinality_with_known_kind_mismatch_is_an_error() -> None:
    plan = _compile(
        _Component("producer", output="GenomeBatch"),
        _Component("consumer", input_kind="Population", input_cardinality="many"),
    )

    errors = [
        diagnostic
        for diagnostic in plan.diagnostics
        if diagnostic.severity is Severity.ERROR
    ]
    assert [diagnostic.code for diagnostic in errors] == ["unresolved_input"]
    assert not [
        diagnostic
        for diagnostic in plan.diagnostics
        if diagnostic.code == "cannot_verify_data_kind"
    ]


def test_multiple_compatible_upstream_producers_are_ambiguous() -> None:
    plan = _compile(
        _Component("first", output="Population"),
        _Component("second", output="Population"),
        _Component("consumer", input_kind="Population"),
    )

    assert [diagnostic.code for diagnostic in plan.diagnostics] == ["ambiguous_input"]
    assert all(
        diagnostic.severity is Severity.WARNING for diagnostic in plan.diagnostics
    )
    assert not plan.graph.data_edges


def test_optional_input_without_compatible_upstream_stays_unconnected() -> None:
    plan = _compile(_Component("consumer", input_kind="Population", optional=True))

    assert not plan.graph.data_edges
    assert not plan.diagnostics


def test_incompatible_upstream_producer_is_not_selected() -> None:
    plan = _compile(
        _Component("producer", output="FeatureBatch"),
        _Component("consumer", input_kind="Population"),
    )

    assert not plan.graph.data_edges
    assert [diagnostic.code for diagnostic in plan.diagnostics] == ["unresolved_input"]


def test_direct_component_graph_is_not_auto_resolved() -> None:
    producer = _Component("producer", output="Population")
    consumer = _Component("consumer", input_kind="Population")
    graph = ComponentGraph(
        nodes=(
            ComponentNode(component_id="producer", component=producer),
            ComponentNode(component_id="consumer", component=consumer),
        ),
        control_edges=(
            ControlEdge(
                source=NodeRef(component_id="producer"),
                target=NodeRef(component_id="consumer"),
            ),
        ),
        entry_points=(NodeRef(component_id="producer"),),
    )
    assert not Compiler().compile(graph).graph.data_edges


def test_repeat_zero_does_not_provide_data_to_the_following_region() -> None:
    plan = Compiler().compile(
        lower_structured(
            [
                RepeatRegion(
                    region_id="repeat",
                    count=0,
                    body=(_Component("producer", output="Population"),),
                ),
                _Component("consumer", input_kind="Population"),
            ]
        )
    )

    assert not plan.graph.data_edges
    assert [diagnostic.code for diagnostic in plan.diagnostics] == ["unresolved_input"]


def test_callable_repeat_count_is_rejected() -> None:
    with pytest.raises(ValidationError):
        RepeatRegion(
            region_id="repeat",
            count=lambda view: 1,  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
            body=(_Component("producer", output="Population"),),
        )


def test_loop_body_does_not_provide_data_to_the_following_region() -> None:
    plan = Compiler().compile(
        lower_structured(
            [
                LoopRegion(
                    region_id="loop",
                    condition=_Condition(),
                    body=(_Component("producer", output="Population"),),
                ),
                _Component("consumer", input_kind="Population"),
            ]
        )
    )

    assert not plan.graph.data_edges
    assert [diagnostic.code for diagnostic in plan.diagnostics] == ["unresolved_input"]


def test_branch_single_path_producer_does_not_provide_data_after_branch() -> None:
    plan = Compiler().compile(
        lower_structured(
            [
                BranchRegion(
                    region_id="branch",
                    condition=_Condition(),
                    body=(_Component("producer", output="Population"),),
                ),
                _Component("consumer", input_kind="Population"),
            ]
        )
    )

    assert not plan.graph.data_edges
    assert [diagnostic.code for diagnostic in plan.diagnostics] == ["unresolved_input"]


def _dataflow_diagnostics(plan) -> list[str]:
    return [
        diagnostic.code
        for diagnostic in plan.diagnostics
        if diagnostic.code in ("unresolved_input", "ambiguous_input")
    ]


def test_branch_paths_each_supplying_one_producer_resolve_the_input() -> None:
    plan = Compiler().compile(
        lower_structured(
            [
                BranchRegion(
                    region_id="branch",
                    condition=_Condition(),
                    body=(_Component("a", output="Population"),),
                    otherwise=(_Component("b", output="Population"),),
                ),
                _Component("consumer", input_kind="Population"),
            ]
        )
    )

    assert not _dataflow_diagnostics(plan)
    assert sorted(_edge_tuple(edge) for edge in plan.graph.data_edges) == [
        ("branch.a", "main", "consumer", "main"),
        ("branch.else.b", "main", "consumer", "main"),
    ]


def test_branch_path_with_multiple_compatible_producers_is_ambiguous() -> None:
    plan = Compiler().compile(
        lower_structured(
            [
                BranchRegion(
                    region_id="branch",
                    condition=_Condition(),
                    body=(
                        _Component("a1", output="Population"),
                        _Component("a2", output="Population"),
                    ),
                    otherwise=(_Component("b", output="Population"),),
                ),
                _Component("consumer", input_kind="Population"),
            ]
        )
    )

    assert _dataflow_diagnostics(plan) == ["ambiguous_input"]
    assert not plan.graph.data_edges
    ambiguous = next(
        diagnostic
        for diagnostic in plan.diagnostics
        if diagnostic.code == "ambiguous_input"
    )
    assert {related.components[0] for related in ambiguous.related} == {
        "branch.a1",
        "branch.a2",
    }


def test_nested_branch_paths_each_supplying_one_producer_resolve_the_input() -> None:
    plan = Compiler().compile(
        lower_structured(
            [
                BranchRegion(
                    region_id="outer",
                    condition=_Condition(),
                    body=(
                        BranchRegion(
                            region_id="inner",
                            condition=_Condition(),
                            body=(_Component("a1", output="Population"),),
                            otherwise=(_Component("a2", output="Population"),),
                        ),
                    ),
                    otherwise=(_Component("b", output="Population"),),
                ),
                _Component("consumer", input_kind="Population"),
            ]
        )
    )

    assert not _dataflow_diagnostics(plan)
    assert sorted(_edge_tuple(edge) for edge in plan.graph.data_edges) == [
        ("outer.else.b", "main", "consumer", "main"),
        ("outer.inner.a1", "main", "consumer", "main"),
        ("outer.inner.else.a2", "main", "consumer", "main"),
    ]


def test_upstream_producer_before_branch_keeps_single_definite_edge() -> None:
    plan = Compiler().compile(
        lower_structured(
            [
                _Component("producer", output="Population"),
                BranchRegion(
                    region_id="branch",
                    condition=_Condition(),
                    body=(_Component("a", output="Population"),),
                    otherwise=(_Component("b", output="Population"),),
                ),
                _Component("consumer", input_kind="Population"),
            ]
        )
    )

    assert not _dataflow_diagnostics(plan)
    assert [_edge_tuple(edge) for edge in plan.graph.data_edges] == [
        ("producer", "main", "consumer", "main")
    ]
