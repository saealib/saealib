from __future__ import annotations

import ast
import contextlib
import linecache
import textwrap
from typing import cast

import numpy as np
import pytest
from instrumented_store import Recorder, attach_instrumentation, instrumented_component

from saealib import (
    Archive,
    OptimizationState,
    ParetoArchive,
    Population,
    PopulationAttribute,
    Problem,
)
from saealib.core.state import POPULATIONS_MAIN, StateKey
from saealib.core.state_inference import (
    _ALLOWED_AST_NODES,
    InferenceStatus,
    clear_state_effect_cache,
    infer_state_effects,
)
from saealib.operators.selection import SurvivorSelection
from saealib.pipeline import Stage

PROBE = StateKey(namespace="user", name="probe", schema_version=1)


def _problem() -> Problem:
    return Problem(
        lambda x: np.array([float(np.sum(np.asarray(x) ** 2))]),
        2,
        1,
        np.array([-1.0]),
        lb=[-1.0] * 2,
        ub=[1.0] * 2,
    )


def _attributes() -> list[PopulationAttribute]:
    return [
        PopulationAttribute("x", np.float64, (2,)),
        PopulationAttribute("f", np.float64, (1,)),
        PopulationAttribute("g", np.float64, (0,)),
        PopulationAttribute("cv", np.float64, (), default=0.0),
        PopulationAttribute("id", np.int64, (), default=-1),
    ]


def _state() -> OptimizationState:
    population = Population(_attributes(), init_capacity=2)
    population._extend_internal(
        {
            "x": np.array([[0.1, 0.2], [0.3, 0.4]]),
            "f": np.array([[1.0], [2.0]]),
            "g": np.zeros((2, 0)),
            "cv": np.zeros(2),
            "id": np.array([1, 2]),
        },
        preserve_ids=True,
    )
    archive = Archive(_attributes(), init_capacity=2)
    archive._extend_internal(
        {
            "x": np.array([[0.1, 0.2], [0.3, 0.4]]),
            "f": np.array([[1.0], [2.0]]),
            "g": np.zeros((2, 0)),
            "cv": np.zeros(2),
            "id": np.array([1, 2]),
        },
        preserve_ids=True,
    )
    return OptimizationState(
        problem=_problem(),
        populations={"main": population},
        archives={
            "main": archive,
            "pareto": ParetoArchive(_attributes(), init_capacity=2),
        },
        rng=np.random.default_rng(0),
    )


class _WriteProbeStage(Stage):
    def execute(self, state):
        state.set_state(PROBE, 1)
        return state


class _ArgumentWriter(SurvivorSelection):
    def select(self, pool, n_survivors, *, ctx):
        pool.update_array("x", np.zeros((len(pool), 2)))
        return np.arange(min(n_survivors, len(pool)))


class _BranchWriter(Stage):
    enabled = False

    def execute(self, state):
        if self.enabled:
            state.set_state(PROBE, 1)
        return state


@pytest.fixture(autouse=True)
def _clear_inference_cache():
    clear_state_effect_cache()
    yield
    clear_state_effect_cache()


def _inferred(component):
    result = infer_state_effects(component)
    assert result.status is InferenceStatus.INFERRED
    assert result.effects is not None
    return result.effects


def _observe(name: str, state: OptimizationState, call) -> tuple[set, set]:
    recorder = Recorder()
    attach_instrumentation(state, recorder)
    with instrumented_component(name, recorder):
        call()
    recorder.flush()
    return recorder.keys(name, "read"), recorder.keys(name, "write")


def test_inference_matches_runtime_observation_for_a_state_write() -> None:
    stage = _WriteProbeStage()
    state = _state()
    _, observed_writes = _observe("stage", state, lambda: stage.execute(state))
    inferred = _inferred(stage)

    assert PROBE in inferred.writes
    assert PROBE in observed_writes


def test_explicit_argument_effects_are_not_inferred() -> None:
    selection = _ArgumentWriter()
    state = _state()
    _, observed_writes = _observe(
        "selection",
        state,
        lambda: selection.select(state.population, 1, ctx=state),
    )
    inferred = _inferred(selection)

    assert inferred.writes == ()
    assert POPULATIONS_MAIN in observed_writes


def test_branches_that_do_not_run_are_still_inferred() -> None:
    stage = _BranchWriter()
    state = _state()
    _, observed_writes = _observe("stage", state, lambda: stage.execute(state))
    inferred = _inferred(stage)

    assert PROBE in inferred.writes
    assert PROBE not in observed_writes


# --- contexts and keys that stay outside the modeled forms ----------------

GLOBAL_KEY = StateKey(namespace="user", name="global", schema_version=1)
OTHER = StateKey(namespace="user", name="other", schema_version=1)


def _write_first(items):
    items[0].set_state(OTHER, 1)


class _ConditionalAliasStage(Stage):
    enabled = True

    def execute(self, state):
        alias = state if self.enabled else state  # noqa: RUF034
        alias.set_state(OTHER, 1)
        return state


class _BoundMethodAliasStage(Stage):
    def execute(self, state):
        writer = state.set_state
        writer(OTHER, 1)
        return state


class _BooleanAliasStage(Stage):
    def execute(self, state):
        (state or state).set_state(OTHER, 1)
        return state


class _ContainerArgumentStage(Stage):
    def execute(self, state):
        _write_first([state])
        return state


class _LocallyAssignedKeyStage(Stage):
    def execute(self, state):
        GLOBAL_KEY = OTHER
        state.set_state(GLOBAL_KEY, 1)
        return state


class _DefaultedKeyStage(Stage):
    def execute(self, state, GLOBAL_KEY=OTHER):  # noqa: N803
        state.set_state(GLOBAL_KEY, 1)
        return state


class _LoopBoundKeyStage(Stage):
    def execute(self, state):
        for GLOBAL_KEY in [OTHER]:
            state.set_state(GLOBAL_KEY, 1)
        return state


_UNMODELED_CONTEXT_STAGES = (
    _ConditionalAliasStage,
    _BoundMethodAliasStage,
    _BooleanAliasStage,
    _ContainerArgumentStage,
    _LocallyAssignedKeyStage,
    _DefaultedKeyStage,
    _LoopBoundKeyStage,
)
_UNMODELED_CONTEXT_IDS = (
    "conditional-alias",
    "bound-method-alias",
    "boolean-alias",
    "container-argument",
    "locally-assigned-key",
    "defaulted-key",
    "loop-bound-key",
)


@pytest.mark.parametrize(
    "component_type", _UNMODELED_CONTEXT_STAGES, ids=_UNMODELED_CONTEXT_IDS
)
def test_a_context_outside_the_modeled_forms_is_not_enumerable(component_type) -> None:
    stage = component_type()
    state = _state()
    _, observed_writes = _observe("stage", state, lambda: stage.execute(state))

    assert OTHER in observed_writes
    assert infer_state_effects(stage).status is InferenceStatus.UNKNOWN


# --- every allowed node type can carry the context ------------------------


def _write_key(ctx):
    ctx.set_state(PROBE, 1)


def _read_then_write(expression: str) -> str:
    return f"value = {expression}\nstate.set_state(OTHER, value)\nreturn state\n"


_BINARY_OPERATORS = {
    ast.Add: "+",
    ast.Sub: "-",
    ast.Mult: "*",
    ast.MatMult: "@",
    ast.Div: "/",
    ast.FloorDiv: "//",
    ast.Mod: "%",
    ast.Pow: "**",
    ast.LShift: "<<",
    ast.RShift: ">>",
    ast.BitOr: "|",
    ast.BitXor: "^",
    ast.BitAnd: "&",
}
_COMPARISON_OPERATORS = {
    ast.Eq: "== 1",
    ast.NotEq: "!= 1",
    ast.Lt: "< 1",
    ast.LtE: "<= 1",
    ast.Gt: "> 1",
    ast.GtE: ">= 1",
    ast.Is: "is None",
    ast.IsNot: "is not None",
    ast.In: "in (1,)",
    ast.NotIn: "not in (1,)",
}
_BOOLEAN_OPERATORS = {ast.And: "and", ast.Or: "or"}
_UNARY_OPERATORS = {ast.UAdd: "+", ast.USub: "-", ast.Invert: "~", ast.Not: "not"}

_NODE_CONTEXT_BODIES: dict[type[ast.AST], str] = {
    ast.arguments: "return state\n",
    ast.arg: "return state\n",
    ast.alias: (
        "import contextlib as context_module\nstate.set_state(PROBE, 1)\nreturn state\n"
    ),
    ast.keyword: "_write_key(ctx=state)\nreturn state\n",
    ast.Return: "return state\n",
    ast.Assign: "alias = state\nalias.set_state(PROBE, 1)\nreturn state\n",
    ast.AnnAssign: "alias: object = state\nalias.set_state(PROBE, 1)\nreturn state\n",
    ast.Expr: "state.set_state(PROBE, 1)\nreturn state\n",
    ast.Pass: "if False:\n    pass\nstate.set_state(PROBE, 1)\nreturn state\n",
    ast.If: "if True:\n    state.set_state(PROBE, 1)\nreturn state\n",
    ast.For: "for item in [1]:\n    state.set_state(PROBE, item)\nreturn state\n",
    ast.While: (
        "count = 0\n"
        "while count < 1:\n"
        "    state.set_state(PROBE, count)\n"
        "    count = count + 1\n"
        "return state\n"
    ),
    ast.Break: (
        "for item in [1]:\n    state.set_state(PROBE, item)\n    break\nreturn state\n"
    ),
    ast.Continue: (
        "for item in [1]:\n"
        "    state.set_state(PROBE, item)\n"
        "    continue\n"
        "return state\n"
    ),
    ast.With: (
        "with contextlib.nullcontext():\n    state.set_state(PROBE, 1)\nreturn state\n"
    ),
    ast.withitem: (
        "with contextlib.nullcontext():\n    state.set_state(PROBE, 1)\nreturn state\n"
    ),
    ast.Try: (
        "try:\n"
        "    state.set_state(PROBE, 1)\n"
        "except ValueError:\n"
        "    state.set_state(OTHER, 1)\n"
        "return state\n"
    ),
    ast.ExceptHandler: (
        "try:\n"
        "    state.set_state(PROBE, 1)\n"
        "except ValueError:\n"
        "    state.set_state(OTHER, 1)\n"
        "return state\n"
    ),
    ast.Raise: "raise ValueError(state.get_state(PROBE))\n",
    ast.Assert: "assert state.get_state(PROBE) is None\nreturn state\n",
    ast.Import: "import contextlib\nstate.set_state(PROBE, 1)\nreturn state\n",
    ast.ImportFrom: (
        "from contextlib import nullcontext\nstate.set_state(PROBE, 1)\nreturn state\n"
    ),
    ast.Call: "state.set_state(PROBE, 1)\nreturn state\n",
    ast.Attribute: "generator = state.rng\nstate.set_state(PROBE, 1)\nreturn state\n",
    ast.Name: "state.set_state(PROBE, 1)\nreturn state\n",
    ast.Constant: "state.set_state(PROBE, 7)\nreturn state\n",
    ast.Tuple: "items = (state, 1)\nitems[0].set_state(PROBE, 1)\nreturn state\n",
    ast.List: "items = [state]\nitems[0].set_state(PROBE, 1)\nreturn state\n",
    ast.Set: "items = {state}\nreturn state\n",
    ast.Dict: "items = {1: state}\nreturn state\n",
    ast.IfExp: (
        "alias = state if True else state\nalias.set_state(PROBE, 1)\nreturn state\n"
    ),
    ast.BoolOp: "alias = state or state\nalias.set_state(PROBE, 1)\nreturn state\n",
    ast.UnaryOp: _read_then_write("not state.get_state(PROBE)"),
    ast.BinOp: _read_then_write("state.get_state(PROBE) + 1"),
    ast.Compare: (
        "if state.get_state(PROBE) is None:\n"
        "    state.set_state(PROBE, 1)\n"
        "return state\n"
    ),
    ast.Load: "state.get_state(PROBE)\nreturn state\n",
    ast.Store: "alias = state\nreturn state\n",
    ast.Del: "state.set_state(PROBE, 1)\nreturn state\n",
}
for _operator, _text in _BINARY_OPERATORS.items():
    _NODE_CONTEXT_BODIES[_operator] = _read_then_write(
        f"state.get_state(PROBE) {_text} 1"
    )
for _operator, _text in _COMPARISON_OPERATORS.items():
    _NODE_CONTEXT_BODIES[_operator] = _read_then_write(
        f"state.get_state(PROBE) {_text}"
    )
for _operator, _text in _BOOLEAN_OPERATORS.items():
    _NODE_CONTEXT_BODIES[_operator] = _read_then_write(
        f"state.get_state(PROBE) {_text} 1"
    )
for _operator, _text in _UNARY_OPERATORS.items():
    _NODE_CONTEXT_BODIES[_operator] = _read_then_write(
        f"{_text} state.get_state(PROBE)"
    )


def _node_stage(node_type: type[ast.AST]) -> Stage:
    body = textwrap.indent(_NODE_CONTEXT_BODIES[node_type], "        ")
    source = f"class Probe(Stage):\n    def execute(self, state):\n{body}"
    filename = f"<state-context-node-{node_type.__name__}>"
    linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)
    namespace = {
        "Stage": Stage,
        "contextlib": contextlib,
        "_write_key": _write_key,
        "PROBE": PROBE,
        "OTHER": OTHER,
    }
    exec(compile(source, filename, "exec"), namespace)
    return cast(type, namespace["Probe"])()


def _execute_quietly(stage: Stage, state: OptimizationState) -> None:
    try:
        stage.execute(state)
    except Exception:
        return


def test_every_allowed_node_type_has_a_context_row() -> None:
    assert set(_NODE_CONTEXT_BODIES) == _ALLOWED_AST_NODES


def _node_type_name(node_type: type[ast.AST]) -> str:
    return node_type.__name__


def _row_node_types() -> list[type[ast.AST]]:
    return sorted(_NODE_CONTEXT_BODIES, key=_node_type_name)


@pytest.mark.parametrize("node_type", _row_node_types())
def test_a_context_in_an_allowed_node_type_is_enumerable_or_unknown(node_type) -> None:
    clear_state_effect_cache()
    stage = _node_stage(node_type)
    state = _state()
    observed_reads, observed_writes = _observe(
        "stage", state, lambda: _execute_quietly(stage, state)
    )
    inference = infer_state_effects(stage)

    if inference.status is InferenceStatus.UNKNOWN:
        return
    assert inference.status is InferenceStatus.INFERRED
    assert inference.effects is not None
    assert observed_reads <= set(inference.effects.reads)
    assert observed_writes <= set(inference.effects.writes)
