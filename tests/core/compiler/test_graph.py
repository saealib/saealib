from dataclasses import FrozenInstanceError
from typing import Any, cast

import pytest

from saealib.core.compiler import (
    CompileContext,
    ComponentBindings,
    ComponentGraph,
    ComponentNode,
    ControlEdge,
    DataEdge,
    GraphTemplate,
    IdentityRule,
    NodeRef,
    ReachabilityRule,
    RuleContext,
    StateBinding,
)
from saealib.core.compiler.diagnostics import DiagnosticBag
from saealib.core.contracts import ComponentContract
from saealib.core.state.keys import USER_DATA
from saealib.exceptions import ValidationError


class Component:
    def contract(self) -> ComponentContract:
        return ComponentContract()


def node(name: str) -> ComponentNode:
    return ComponentNode(component_id=name, component=Component())


def test_component_node_and_edges_normalize_and_are_frozen():
    first = node("first")
    assert first.component is not None
    assert first.contract == ComponentContract()
    assert (
        DataEdge(
            source=NodeRef(component_id="first"),
            target=NodeRef(component_id="second"),
            source_port="out",
            target_port="in",
        ).source.component_id
        == "first"
    )
    assert (
        StateBinding(node=NodeRef(component_id="first"), state_key=USER_DATA).state_key
        == USER_DATA
    )
    with pytest.raises(FrozenInstanceError):
        setattr(first, "component_id", "other")


def test_well_formedness_collects_structure_and_paths():
    graph = ComponentGraph(
        nodes=(node("a"), node("a"), node("orphan")),
        data_edges=(
            DataEdge(
                source=NodeRef(component_id="a"),
                target=NodeRef(component_id="missing"),
                source_port="out",
                target_port="in",
            ),
        ),
        control_edges=(
            ControlEdge(
                source=NodeRef(component_id="a"),
                target=NodeRef(component_id="orphan"),
            ),
        ),
        entry_points=(NodeRef(component_id="a"),),
    )
    diagnostics = tuple(graph.well_formedness())
    codes = {diagnostic.code for diagnostic in diagnostics}
    assert codes == {"invalid_graph_edge"}
    assert "unreachable_node" not in codes
    assert all(diagnostic.path.components for diagnostic in diagnostics)
    assert all(diagnostic.resolutions for diagnostic in diagnostics)


@pytest.mark.parametrize(
    ("nodes", "entry_points"),
    [
        ((), ()),
        ((node("present"),), (NodeRef(component_id="missing"),)),
    ],
)
def test_well_formedness_reports_invalid_entry_points(nodes, entry_points):
    graph = ComponentGraph(nodes=nodes, entry_points=entry_points)

    diagnostics = tuple(graph.well_formedness())

    assert [diagnostic.code for diagnostic in diagnostics] == ["invalid_entry_point"]


def test_well_formedness_reports_state_bindings_to_missing_nodes():
    graph = ComponentGraph(
        nodes=(node("a"),),
        state_bindings=(
            StateBinding(node=NodeRef(component_id="missing"), state_key=USER_DATA),
        ),
        entry_points=(NodeRef(component_id="a"),),
    )

    diagnostics = tuple(graph.well_formedness())

    assert [diagnostic.code for diagnostic in diagnostics] == ["invalid_state_binding"]
    assert diagnostics[0].path.components == ("missing",)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"nodes": cast(Any, (object(),))},
        {
            "nodes": (node("a"),),
            "data_edges": cast(Any, (object(),)),
        },
    ],
)
def test_component_graph_rejects_invalid_structure_types(kwargs):
    with pytest.raises(ValidationError):
        ComponentGraph(**kwargs)


def test_identity_and_reachability_rules_apply_and_data_control_are_separate():
    graph = ComponentGraph(
        nodes=(node("start"), node("data"), node("control"), node("lost")),
        data_edges=(
            DataEdge(
                source=NodeRef(component_id="start"),
                target=NodeRef(component_id="data"),
                source_port="out",
                target_port="in",
            ),
        ),
        control_edges=(
            ControlEdge(
                source=NodeRef(component_id="start"),
                target=NodeRef(component_id="control"),
            ),
        ),
        entry_points=(NodeRef(component_id="start"),),
    )

    def context_for(graph):
        return RuleContext(
            graph=graph,
            compile_context=CompileContext(),
            diagnostics=DiagnosticBag(),
        )

    assert len(IdentityRule().apply(context_for(graph)).diagnostics) == 0
    findings = ReachabilityRule().apply(context_for(graph)).diagnostics
    assert [finding.code for finding in findings] == ["unreachable_node"]
    assert (
        len(
            ComponentGraph(
                nodes=(node("start"),),
                entry_points=(NodeRef(component_id="start"),),
            ).well_formedness()
        )
        == 0
    )


def test_identity_rule_reports_duplicate_component_ids():
    graph = ComponentGraph(
        nodes=(node("duplicate"), node("duplicate")),
        entry_points=(NodeRef(component_id="duplicate"),),
    )

    findings = (
        IdentityRule()
        .apply(
            RuleContext(
                graph=graph,
                compile_context=CompileContext(),
                diagnostics=DiagnosticBag(),
            )
        )
        .diagnostics
    )

    assert [finding.code for finding in findings] == ["duplicate_component_id"]


def test_template_and_bindings_are_minimal():
    class Template(GraphTemplate):
        def build_graph(self, bindings: ComponentBindings) -> ComponentGraph:
            return ComponentGraph(
                nodes=(
                    ComponentNode(component_id="x", component=bindings.components["x"]),
                ),
                entry_points=(NodeRef(component_id="x"),),
            )

    graph = Template().build_graph(ComponentBindings(components={"x": Component()}))
    assert graph.node_by_id("x").component_id == "x"
    with pytest.raises(ValidationError):
        ComponentBindings(components={"bad:name:extra": Component()})
