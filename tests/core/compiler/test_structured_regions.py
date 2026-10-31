from dataclasses import FrozenInstanceError

import pytest

from saealib.core.compiler import (
    BranchRegion,
    Compiler,
    LoopRegion,
    NodeRef,
    RepeatRegion,
    StructuredGraph,
    lower_structured,
)
from saealib.core.contracts import ComponentContract, StateContract
from saealib.core.state.keys import RUNTIME_RNG, USER_DATA
from saealib.exceptions import ValidationError


class Component:
    def __init__(self, name, state=StateContract()):
        self.name = name
        self._state = state

    def contract(self):
        return ComponentContract(state=self._state)


class PipelineLike:
    def __init__(self, name, stages):
        self.name = name
        self.stages = stages


class Condition:
    def __init__(self, state=StateContract()):
        self._state = state

    def contract(self):
        return self._state

    def evaluate(self, context):
        return True


def test_nested_sequence_lowering_qualifies_ids_and_adds_leaf_edges():
    graph = lower_structured(
        PipelineLike(
            "outer",
            [PipelineLike("inner", [Component("first"), Component("last")])],
        )
    )

    assert isinstance(graph, StructuredGraph)
    assert [node.component_id for node in graph.nodes] == [
        "outer.inner.first",
        "outer.inner.last",
    ]
    assert [
        (edge.source.component_id, edge.target.component_id)
        for edge in graph.control_edges
    ] == [("outer.inner.first", "outer.inner.last")]
    assert graph.regions[0].region.qualified_id == "outer.inner"


def test_control_regions_are_retained_and_loop_has_no_cycle_edge():
    graph = lower_structured(
        [
            Component("before"),
            RepeatRegion(region_id="repeat", count=2, body=(Component("r"),)),
            LoopRegion(region_id="loop", condition=Condition(), body=(Component("l"),)),
            BranchRegion(
                region_id="branch", condition=Condition(), body=(Component("b"),)
            ),
            Component("after"),
        ]
    )

    assert [type(node.region) for node in graph.regions] == [
        RepeatRegion,
        LoopRegion,
        BranchRegion,
    ]
    assert all(
        edge.source.component_id != "node2" or edge.target.component_id != "node1"
        for edge in graph.control_edges
    )
    assert all(
        edge.source.component_id != edge.target.component_id
        for edge in graph.control_edges
    )


def test_leading_branch_declares_entry_points_on_every_path():
    graph = lower_structured(
        [
            BranchRegion(
                region_id="branch",
                condition=Condition(),
                body=(Component("then"),),
                otherwise=(Component("otherwise"),),
            ),
            Component("after"),
        ]
    )

    result = Compiler().compile(graph)

    assert graph.entry_points == (
        NodeRef(component_id="branch.then"),
        NodeRef(component_id="branch.else.otherwise"),
    )
    assert not [
        diagnostic
        for diagnostic in result.diagnostics
        if diagnostic.code == "unreachable_node"
    ]


def test_leading_branch_without_otherwise_keeps_single_entry_point():
    graph = lower_structured(
        [
            BranchRegion(
                region_id="branch",
                condition=Condition(),
                body=(Component("then"),),
            ),
            Component("after"),
        ]
    )

    result = Compiler().compile(graph)

    assert graph.entry_points == (NodeRef(component_id="branch.then"),)
    assert not [
        diagnostic
        for diagnostic in result.diagnostics
        if diagnostic.code == "unreachable_node"
    ]


def _leading_branch(region_id):
    return BranchRegion(
        region_id=region_id,
        condition=Condition(),
        body=(Component("then"),),
        otherwise=(Component("otherwise"),),
    )


@pytest.mark.parametrize(
    "body",
    [
        _leading_branch("branch"),
        RepeatRegion(
            region_id="repeat",
            count=2,
            body=(_leading_branch("inner"), Component("tail")),
        ),
        LoopRegion(
            region_id="loop",
            condition=Condition(),
            body=(_leading_branch("inner"), Component("tail")),
        ),
        BranchRegion(
            region_id="branch",
            condition=Condition(),
            body=(_leading_branch("inner"), Component("tail")),
            otherwise=(Component("else_tail"),),
        ),
    ],
)
def test_leading_region_body_starting_with_branch_reaches_inner_paths(body):
    graph = lower_structured(
        [
            BranchRegion(
                region_id="outer",
                condition=Condition(),
                body=(body, Component("tail")),
                otherwise=(Component("else_tail"),),
            )
        ]
    )

    result = Compiler().compile(graph)

    assert not [
        diagnostic
        for diagnostic in result.diagnostics
        if diagnostic.code == "unreachable_node"
    ]


def test_trailing_branch_in_nested_body_exits_on_every_path():
    graph = lower_structured(
        [
            RepeatRegion(
                region_id="repeat",
                count=1,
                body=(
                    Component("before"),
                    BranchRegion(
                        region_id="branch",
                        condition=Condition(),
                        body=(Component("then"),),
                        otherwise=(Component("otherwise"),),
                    ),
                ),
            ),
            Component("after"),
        ]
    )

    result = Compiler().compile(graph)

    edge_ids = [
        (edge.source.component_id, edge.target.component_id)
        for edge in graph.control_edges
    ]
    assert ("repeat.branch.then", "after") in edge_ids
    assert ("repeat.branch.else.otherwise", "after") in edge_ids
    assert not [
        diagnostic
        for diagnostic in result.diagnostics
        if diagnostic.code == "unreachable_node"
    ]


@pytest.mark.parametrize(
    "body",
    [
        RepeatRegion(
            region_id="repeat",
            count=2,
            body=(_leading_branch("inner"), Component("tail")),
        ),
        LoopRegion(
            region_id="loop",
            condition=Condition(),
            body=(_leading_branch("inner"), Component("tail")),
        ),
        BranchRegion(
            region_id="branch",
            condition=Condition(),
            body=(_leading_branch("inner"), Component("tail")),
            otherwise=(Component("else_tail"),),
        ),
    ],
)
def test_trailing_region_body_ending_with_branch_exits_inner_paths(body):
    graph = lower_structured(
        [
            BranchRegion(
                region_id="outer",
                condition=Condition(),
                body=(Component("head"), body),
            )
        ]
    )

    result = Compiler().compile(graph)

    assert not [
        diagnostic
        for diagnostic in result.diagnostics
        if diagnostic.code == "unreachable_node"
    ]


def test_region_effects_compose_component_and_condition_state():
    graph = lower_structured(
        [
            Component("writer", StateContract(writes=(USER_DATA,))),
            LoopRegion(
                region_id="loop",
                condition=Condition(StateContract(reads=(RUNTIME_RNG,))),
                body=(Component("reader", StateContract(reads=(USER_DATA,))),),
            ),
        ]
    )

    assert graph.effect.reads == (USER_DATA, RUNTIME_RNG)
    assert graph.effect.writes == (USER_DATA,)
    assert graph.regions[0].region.effect.reads == (USER_DATA, RUNTIME_RNG)


@pytest.mark.parametrize("count", [-1, True, 1.5])
def test_repeat_rejects_invalid_count(count):
    with pytest.raises(ValidationError):
        RepeatRegion(region_id="repeat", count=count, body=())


@pytest.mark.parametrize(
    "condition", [object(), type("Bad", (), {"contract": lambda self: object()})()]
)
def test_regions_reject_invalid_conditions(condition):
    with pytest.raises(ValidationError):
        LoopRegion(region_id="loop", condition=condition, body=())


def test_structured_vocabulary_is_immutable():
    region = RepeatRegion(region_id="repeat", count=0, body=())
    with pytest.raises(FrozenInstanceError):
        region.region_id = "changed"  # ty: ignore[invalid-assignment]
