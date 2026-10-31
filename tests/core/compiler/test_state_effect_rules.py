from dataclasses import replace

import pytest

from saealib.core.compiler import (
    BranchRegion,
    Compiler,
    ComponentGraph,
    ComponentNode,
    ControlEdge,
    LoopRegion,
    NodeRef,
    RepeatRegion,
    StateBinding,
    lower_structured,
)
from saealib.core.contracts import ComponentContract, StateContract
from saealib.core.state import SURROGATES_DEFAULT, USER_DATA, StateKey, StateView
from saealib.exceptions import ValidationError


class _Component:
    def __init__(self, contract):
        self._contract = contract

    def contract(self):
        return self._contract


class _Condition:
    def __init__(self, state=StateContract()):
        self._state = state

    def contract(self) -> StateContract:
        return self._state

    def evaluate(self, view: StateView) -> bool:
        return True


def _graph(*items, edges=(), bindings=()):
    nodes = tuple(
        ComponentNode(component_id=name, component=_Component(contract))
        for name, contract in items
    )
    return ComponentGraph(
        nodes=nodes,
        control_edges=tuple(edges),
        state_bindings=tuple(bindings),
        entry_points=(NodeRef(component_id=items[0][0]),),
    )


def _codes(graph, *, context=None):
    return [item.code for item in Compiler().compile(graph, context).diagnostics]


def test_ordered_write_then_read_is_valid():
    writer = ComponentContract(state=StateContract(writes=(USER_DATA,)))
    reader = ComponentContract(state=StateContract(reads=(USER_DATA,)))
    graph = _graph(
        ("writer", writer),
        ("reader", reader),
        edges=(
            ControlEdge(
                source=NodeRef(component_id="writer"),
                target=NodeRef(component_id="reader"),
            ),
        ),
    )
    assert "concurrent_state_read_write" not in _codes(graph)
    assert "unreachable_state_read" not in _codes(graph)


def test_read_before_write_and_unwritten_read_are_diagnosed():
    reader = ComponentContract(state=StateContract(reads=(USER_DATA,)))
    writer = ComponentContract(state=StateContract(writes=(USER_DATA,)))
    before = _graph(
        ("reader", reader),
        ("writer", writer),
        edges=(
            ControlEdge(
                source=NodeRef(component_id="reader"),
                target=NodeRef(component_id="writer"),
            ),
        ),
    )
    unwritten = _graph(("reader", reader))
    assert "unreachable_state_read" in _codes(before)
    assert "unreachable_state_read" in _codes(unwritten)


def test_read_write_without_an_initial_value_is_uninitialized():
    graph = _graph(
        (
            "node",
            ComponentContract(
                state=StateContract(reads=(USER_DATA,), writes=(USER_DATA,))
            ),
        )
    )
    assert "uninitialized_state_write" in _codes(graph)


def test_unordered_writes_and_read_write_are_diagnosed():
    writer = ComponentContract(state=StateContract(writes=(USER_DATA,)))
    reader = ComponentContract(state=StateContract(reads=(USER_DATA,)))
    writes = _graph(("left", writer), ("right", writer))
    read_write = _graph(("writer", writer), ("reader", reader))
    assert "concurrent_state_write" in _codes(writes)
    assert "concurrent_state_read_write" in _codes(read_write)


def test_non_enumerable_reads_use_the_graph_state_universe():
    opaque = ComponentContract(state=StateContract(reads_enumerable=False))
    writer = ComponentContract(state=StateContract(writes=(USER_DATA,)))
    graph = _graph(("opaque", opaque), ("writer", writer))
    assert "concurrent_state_read_write" in _codes(graph)
    ordered = replace(
        graph,
        control_edges=(
            ControlEdge(
                source=NodeRef(component_id="writer"),
                target=NodeRef(component_id="opaque"),
            ),
        ),
    )
    assert "concurrent_state_read_write" not in _codes(ordered)


def test_bindings_keep_same_contract_nodes_in_distinct_namespaces():
    contract = ComponentContract(state=StateContract(writes=(USER_DATA,)))
    first_key = StateKey(namespace="user", name="first", schema_version=1)
    second_key = StateKey(namespace="user", name="second", schema_version=1)
    graph = _graph(
        ("node_a", contract),
        ("node_b", contract),
        bindings=(
            StateBinding(node=NodeRef(component_id="node_a"), state_key=first_key),
            StateBinding(node=NodeRef(component_id="node_b"), state_key=second_key),
        ),
    )
    plan = Compiler().compile(graph)
    assert "concurrent_state_write" not in _codes(graph)
    assert not [
        item for item in plan.diagnostics if item.code == "concurrent_state_write"
    ]


def test_bindings_qualify_declared_surrogate_keys_without_string_parsing():
    contract = ComponentContract(state=StateContract(writes=(SURROGATES_DEFAULT,)))
    cheap = StateKey(namespace="surrogates", name="stage:cheap", schema_version=1)
    rich = StateKey(namespace="surrogates", name="stage:rich", schema_version=1)
    graph = _graph(
        ("cheap", contract),
        ("rich", contract),
        bindings=(
            StateBinding(node=NodeRef(component_id="cheap"), state_key=cheap),
            StateBinding(node=NodeRef(component_id="rich"), state_key=rich),
        ),
    )
    plan = Compiler().compile(graph)
    assert not [
        item for item in plan.diagnostics if item.code == "concurrent_state_write"
    ]


def test_endpoint_roles_still_order_state_effects_by_node():
    writer = ComponentContract(state=StateContract(writes=(USER_DATA,)))
    reader = ComponentContract(state=StateContract(reads=(USER_DATA,)))
    graph = _graph(
        ("writer", writer),
        ("reader", reader),
        edges=(
            ControlEdge(
                source=NodeRef(component_id="writer", role="producer"),
                target=NodeRef(component_id="reader", role="consumer"),
            ),
        ),
    )
    assert "unreachable_state_read" not in _codes(graph)


def _structured(*items):
    return lower_structured(items)


def test_structured_condition_reads_require_incoming_initialization():
    graph = _structured(
        LoopRegion(
            region_id="loop",
            condition=_Condition(StateContract(reads=(USER_DATA,))),
            body=(),
        ),
        _Component(ComponentContract()),
    )
    assert "unreachable_state_read" in _codes(graph)


def test_structured_branch_requires_writes_on_both_paths():
    def writer():
        return _Component(ComponentContract(state=StateContract(writes=(USER_DATA,))))

    reader = _Component(ComponentContract(state=StateContract(reads=(USER_DATA,))))
    both = _structured(
        BranchRegion(
            region_id="branch",
            condition=_Condition(),
            body=(writer(),),
            otherwise=(writer(),),
        ),
        reader,
    )
    one = _structured(
        BranchRegion(region_id="branch", condition=_Condition(), body=(writer(),)),
        reader,
    )
    assert "unreachable_state_read" not in _codes(both)
    assert "unreachable_state_read" in _codes(one)
    assert "concurrent_state_write" not in _codes(both)


def test_structured_loop_and_repeat_state_is_sequential():
    writer = _Component(ComponentContract(state=StateContract(writes=(USER_DATA,))))
    reader = _Component(ComponentContract(state=StateContract(reads=(USER_DATA,))))
    loop = _structured(
        LoopRegion(
            region_id="loop",
            condition=_Condition(),
            body=(writer, reader),
        ),
        reader,
    )
    repeat_zero = _structured(
        RepeatRegion(region_id="repeat", count=0, body=(writer,)),
        reader,
    )
    repeat_once = _structured(
        RepeatRegion(region_id="repeat", count=1, body=(writer,)),
        reader,
    )
    assert "unreachable_state_read" in _codes(loop)
    assert "unreachable_state_read" in _codes(repeat_zero)
    assert "unreachable_state_read" not in _codes(repeat_once)


def test_callable_repeat_count_is_rejected() -> None:
    writer = _Component(ComponentContract(state=StateContract(writes=(USER_DATA,))))
    with pytest.raises(ValidationError):
        RepeatRegion(
            region_id="repeat",
            count=lambda view: 1,  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
            body=(writer,),
        )
