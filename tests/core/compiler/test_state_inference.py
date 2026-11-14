from __future__ import annotations

import abc
import ast
import contextlib
import dataclasses
import functools
import importlib
import inspect
import linecache
import operator
import pkgutil
import re
import threading
import time
from abc import abstractmethod
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

import saealib
import saealib.core.compiler.graph as graph_module
from saealib import (
    GA,
    PSO,
    CrossoverBLXAlpha,
    DirectStrategy,
    GenerationBasedStrategy,
    IndividualBasedStrategy,
    LHSInitializer,
    MutationUniform,
    Optimizer,
    Pipeline,
    PreSelectionStrategy,
    Problem,
    SequentialSelection,
    SteadyStateStrategy,
    Termination,
    TruncationSelection,
    max_fe,
    max_gen,
)
from saealib.acquisition.base import AcquisitionFunction
from saealib.algorithms.base import Algorithm
from saealib.context import OptimizationState
from saealib.core.compiler import (
    CompileContext,
    Compiler,
    ComponentGraph,
    ComponentNode,
    ControlEdge,
    ExecutablePlan,
    NodeRef,
    Severity,
)
from saealib.core.compiler.state_inference_rules import CONFLICT_CODE
from saealib.core.contracts import ComponentContract, StateContract
from saealib.core.state import StateKey
from saealib.core.state.patch import StatePatch
from saealib.core.state_inference import (
    _ALLOWED_AST_NODES,
    NOT_APPLICABLE,
    InferenceStatus,
    StateInference,
    _Analyzer,
    clear_state_effect_cache,
    infer_state_effects,
)
from saealib.execution.evaluator import Evaluator
from saealib.execution.runtime import _bound_patch
from saealib.operators.crossover import Crossover
from saealib.operators.mutation import Mutation
from saealib.operators.selection import ParentSelection
from saealib.pipeline import Stage
from saealib.policies.evaluation import EvaluateAll, EvaluationPlanner
from saealib.policies.feedback import FeedbackBuilder
from saealib.strategies.base import TemplateStrategy
from saealib.surrogate.manager import SurrogateManager
from saealib.surrogate.training_set import TrainingSet

PROBE = StateKey(namespace="user", name="probe", schema_version=1)
OTHER = StateKey(namespace="user", name="other", schema_version=1)


def _quadratic(x):
    return np.array([float(np.sum(np.asarray(x) ** 2))])


def _problem() -> Problem:
    return Problem(
        func=_quadratic,
        dim=2,
        n_obj=1,
        direction=np.array([-1.0]),
        lb=[-1.0] * 2,
        ub=[1.0] * 2,
    )


def _inferred(component) -> StateContract:
    result = infer_state_effects(component)
    assert result.status is InferenceStatus.INFERRED
    assert result.effects is not None
    return result.effects


class _WriteProbeStage(Stage):
    def execute(self, state):
        return StatePatch(writes={PROBE: 1})


class _ReadProbeStage(Stage):
    def execute(self, state):
        return StatePatch(writes={OTHER: state.get_state(PROBE)})


class _ReadWriteProbeStage(Stage):
    def execute(self, state):
        return StatePatch(writes={PROBE: state.get_state(PROBE) + 1})


class _DeclaredReadProbeStage(Stage):
    def contract(self) -> ComponentContract:
        return ComponentContract(state=StateContract(reads=(PROBE,), writes=(OTHER,)))

    def execute(self, state):
        return StatePatch(writes={OTHER: state.get_state(PROBE)})


class _DeclaredWriteProbeStage(Stage):
    def contract(self) -> ComponentContract:
        return ComponentContract(state=StateContract(writes=(PROBE,)))

    def execute(self, state):
        return StatePatch(writes={PROBE: 1})


class _DeclaredReadWriteStage(Stage):
    def contract(self) -> ComponentContract:
        return ComponentContract(state=StateContract(reads=(PROBE,), writes=(PROBE,)))

    def execute(self, state):
        return StatePatch(writes={PROBE: state.get_state(PROBE) + 1})


class _NoopStage(Stage):
    def execute(self, state):
        return state


class _DeclaredStage(Stage):
    def contract(self) -> ComponentContract:
        return ComponentContract(state=StateContract(reads=(PROBE,), writes=(PROBE,)))

    def execute(self, state):
        return StatePatch(writes={PROBE: 1})


class _SuperOnlyStage(Stage):
    def contract(self) -> ComponentContract:
        return super().contract()

    def execute(self, state):
        return StatePatch(writes={PROBE: 1})


class _AlgorithmDefault(Algorithm):
    def ask(self, request, state):
        raise NotImplementedError

    def tell(self, feedback, state):
        raise NotImplementedError


class _SelectionDefault(ParentSelection):
    def select(
        self, population, n_pair, n_parents, rng=np.random.default_rng(), *, ctx
    ) -> np.ndarray:
        return population.extract([])


class _SubMutationUniform(MutationUniform):
    pass


class _SubSequentialSelection(SequentialSelection):
    pass


class _SubEvaluateAll(EvaluateAll):
    pass


class _GetattrStage(Stage):
    def execute(self, state):
        getattr(state, "probe", None)
        return StatePatch(writes={PROBE: 1})


class _PartialStage(Stage):
    def __init__(self) -> None:
        self._reader = functools.partial(lambda state: state.get_state(PROBE))

    def execute(self, state):
        self._reader(state)
        return StatePatch(writes={PROBE: 1})


class _RaisingContract(Stage):
    def contract(self) -> ComponentContract:
        raise RuntimeError("broken")

    def execute(self, state):
        return StatePatch(writes={PROBE: 1})


class _CastStage(Stage):
    def execute(self, state):
        context = cast(OptimizationState, state)
        return StatePatch(writes={PROBE: context.get_state(PROBE)})


_EVALUATED: list[str] = []


def _probe_key() -> StateKey[object]:
    _EVALUATED.append("called")
    return PROBE


class _EvaluatedStage(Stage):
    def execute(self, state):
        return StatePatch(writes={PROBE: state.get_state(_probe_key())})


class _Keys:
    @property
    def key(self):
        _PROPERTY_CALLS.append("key")
        return PROBE


_PROPERTY_CALLS: list[str] = []
_KEYS = _Keys()


class _PropertyKeyStage(Stage):
    def execute(self, state):
        state.get_state(_KEYS.key)
        return state


class _MethodCallerStage(Stage):
    execute = operator.methodcaller("set_state", PROBE, 3)


class _LambdaStage(Stage):
    def execute(self, state):
        return state


_LambdaStage.execute = lambda self, state: state.set_state(PROBE, 1)


class _OpaqueBaseStage(Stage):
    def execute(self, state):
        return state


setattr(_OpaqueBaseStage, "execute", operator.methodcaller("set_state", PROBE, 1))


class _SuperOpaqueStage(_OpaqueBaseStage):
    def execute(self, state):
        return super().execute(state)


def _add_other(function):
    @functools.wraps(function)
    def wrapper(self, state):
        state.set_state(OTHER, 1)
        return function(self, state)

    return wrapper


class _WrappedStage(Stage):
    @_add_other
    def execute(self, state):
        state.set_state(PROBE, 1)
        return state


class _StaticMethodStage(Stage):
    def execute(self, state):
        return state


setattr(
    _StaticMethodStage,
    "execute",
    staticmethod(lambda state: state.set_state(PROBE, 1)),
)


class _ClassMethodStage(Stage):
    def execute(self, state):
        return state


setattr(
    _ClassMethodStage,
    "execute",
    classmethod(lambda cls, state: state.set_state(PROBE, 1)),
)


class _DynamicGetattrStage(Stage):
    def __getattr__(self, name):
        raise AttributeError(name)

    def execute(self, state):
        return state


class _UnreturnedReplaceStage(Stage):
    def execute(self, state):
        state.replace(data=1)
        return state


class _PositionalStatePatchStage(Stage):
    def execute(self, state):
        return StatePatch({PROBE: 1})  # ty: ignore[missing-argument, too-many-positional-arguments]


class _ContainerStage(Stage):
    def execute(self, state):
        items = [state]
        items.append(state)
        return state


class _RecursiveStage(Stage):
    def execute(self, state):
        return self.helper(state)

    def helper(self, state):
        return self.execute(state)


class _UserAbstractBase(Stage):
    def contract(self) -> ComponentContract:
        return ComponentContract(state=StateContract())

    @abstractmethod
    def execute(self, state): ...


class _UserConcreteStage(_UserAbstractBase):
    def execute(self, state):
        state.set_state(PROBE, 1)
        return state


_INSTANCE_OVERRIDE = _WriteProbeStage()
setattr(_INSTANCE_OVERRIDE, "execute", lambda state: StatePatch(writes={OTHER: 1}))


_EXEC_NAMESPACE: dict[str, object] = {}
exec(
    "from saealib.pipeline import Stage as _Stage\n"
    "from saealib.core.state.patch import StatePatch as _StatePatch\n"
    "class ExecStage(_Stage):\n"
    "    def execute(self, state):\n"
    "        return _StatePatch(writes={})\n",
    _EXEC_NAMESPACE,
)
_ExecStage = cast(type[Stage], _EXEC_NAMESPACE["ExecStage"])


@pytest.fixture(autouse=True)
def _clear_inference_cache():
    clear_state_effect_cache()
    yield
    clear_state_effect_cache()


def _compile_graph(
    *nodes: ComponentNode,
    initial: frozenset[StateKey[object]] = frozenset(),
):
    graph = ComponentGraph(
        nodes=nodes,
        control_edges=(),
        state_bindings=(),
        entry_points=(NodeRef(component_id=nodes[0].component_id),),
    )
    return Compiler().compile(graph, CompileContext(initial_state_keys=initial))


def _builtin_configurations() -> dict[str, Optimizer]:
    ga = GA(
        CrossoverBLXAlpha(0.9, 0.5),
        MutationUniform(0.1),
        SequentialSelection(),
        TruncationSelection(),
    )
    return {
        "ga-direct": (
            Optimizer(_problem(), seed=11)
            .set_initializer(LHSInitializer(2, 2, 11))
            .set_algorithm(ga)
            .set_strategy(DirectStrategy())
            .set_termination(Termination(max_gen(1)))
        ),
        "pso-direct": (
            Optimizer(_problem(), seed=11)
            .set_initializer(LHSInitializer(2, 2, 11))
            .set_algorithm(PSO())
            .set_strategy(DirectStrategy())
            .set_termination(Termination(max_gen(1)))
        ),
        "ga-template": (
            Optimizer(_problem(), seed=11)
            .set_initializer(LHSInitializer(2, 2, 11))
            .set_algorithm(ga)
            .set_strategy(TemplateStrategy(Pipeline([_NoopStage()])))
            .set_termination(Termination(max_gen(1)))
        ),
    }


def _compile_plan(optimizer: Optimizer, *, infer: bool) -> ExecutablePlan:
    optimizer.resolve_defaults()
    if not infer:
        original = graph_module.infer_state_effects
        setattr(graph_module, "infer_state_effects", lambda component: NOT_APPLICABLE)
        try:
            optimizer.resolve_defaults()
            plan = optimizer._compile_plan(require_initializer=True)
        finally:
            setattr(graph_module, "infer_state_effects", original)
    else:
        plan = optimizer._compile_plan(require_initializer=True)
    assert plan is not None
    return plan


def test_inference_applies_only_to_family_defaults_with_empty_state() -> None:
    clear_state_effect_cache()
    assert infer_state_effects(_WriteProbeStage()).status is InferenceStatus.INFERRED
    assert (
        infer_state_effects(_DeclaredStage()).status is InferenceStatus.NOT_APPLICABLE
    )
    assert (
        infer_state_effects(_SuperOnlyStage()).status is InferenceStatus.NOT_APPLICABLE
    )
    assert (
        infer_state_effects(MutationUniform(prob=0.1)).status
        is InferenceStatus.NOT_APPLICABLE
    )
    assert (
        infer_state_effects(_SubMutationUniform(prob=0.1)).status
        is InferenceStatus.NOT_APPLICABLE
    )
    assert (
        infer_state_effects(_SubEvaluateAll()).status is InferenceStatus.NOT_APPLICABLE
    )
    assert (
        infer_state_effects(_AlgorithmDefault()).status
        is InferenceStatus.NOT_APPLICABLE
    )
    assert _inferred(_SelectionDefault()) == StateContract()


@pytest.mark.parametrize(
    "component",
    [
        CrossoverBLXAlpha(0.9, 0.5),
        SequentialSelection(),
        _SubSequentialSelection(),
        MutationUniform(prob=0.1),
        _SubMutationUniform(prob=0.1),
        EvaluateAll(),
        _SubEvaluateAll(),
    ],
)
def test_builtin_classes_keep_their_contract_after_inference(component) -> None:
    before = component.contract()
    inference = infer_state_effects(component)
    after = component.contract()
    assert before == after
    if type(component) in (MutationUniform, _SubMutationUniform, EvaluateAll):
        assert inference.status is InferenceStatus.NOT_APPLICABLE


def test_inferred_effects_do_not_change_contracts_or_the_plan_signature() -> None:
    with_inference = _compile_plan(_builtin_configurations()["ga-template"], infer=True)
    clear_state_effect_cache()
    without_inference = _compile_plan(
        _builtin_configurations()["ga-template"], infer=False
    )

    assert with_inference.contract_snapshots == without_inference.contract_snapshots
    for with_node, without_node in zip(
        with_inference.graph.nodes, without_inference.graph.nodes
    ):
        assert with_node.contract == without_node.contract
        assert with_node.analysis_contract == without_node.analysis_contract
        assert with_node._inferred_effects.status is InferenceStatus.INFERRED
        assert without_node._inferred_effects.status is InferenceStatus.NOT_APPLICABLE


def test_write_gate_decisions_are_identical_with_and_without_inference() -> None:
    with_inference = _compile_plan(_builtin_configurations()["ga-template"], infer=True)
    clear_state_effect_cache()
    without_inference = _compile_plan(
        _builtin_configurations()["ga-template"], infer=False
    )
    fake_plan = SimpleNamespace(
        plan=SimpleNamespace(graph=SimpleNamespace(state_bindings=()))
    )

    def decision(plan, node, key):
        try:
            _bound_patch(cast(Any, fake_plan), node, StatePatch(writes={key: 1}))
            return ("allowed", None)
        except Exception as error:
            return ("rejected", type(error).__name__)

    for with_node, without_node in zip(
        with_inference.graph.nodes, without_inference.graph.nodes
    ):
        for key in (PROBE, OTHER):
            assert decision(with_inference, with_node, key) == decision(
                without_inference, without_node, key
            )


@pytest.mark.parametrize("configuration", sorted(_builtin_configurations()))
def test_builtin_configurations_gain_no_inference_findings(configuration) -> None:
    with_inference = _compile_plan(_builtin_configurations()[configuration], infer=True)
    clear_state_effect_cache()
    without_inference = _compile_plan(
        _builtin_configurations()[configuration], infer=False
    )

    added = [
        diagnostic
        for diagnostic in with_inference.diagnostics
        if diagnostic not in without_inference.diagnostics
    ]
    removed = [
        diagnostic
        for diagnostic in without_inference.diagnostics
        if diagnostic not in with_inference.diagnostics
    ]
    assert added == []
    assert removed == []


def test_inference_adds_warnings_without_removing_findings() -> None:
    optimizer = (
        Optimizer(_problem(), seed=5)
        .set_initializer(LHSInitializer(2, 2, 5))
        .set_algorithm(
            GA(
                CrossoverBLXAlpha(0.9, 0.5),
                MutationUniform(0.1),
                SequentialSelection(),
                TruncationSelection(),
            )
        )
        .set_strategy(TemplateStrategy(Pipeline([_ReadWriteProbeStage()])))
        .set_termination(Termination(max_gen(1)))
    )
    optimizer.resolve_defaults()
    with_inference = optimizer._compile_plan(require_initializer=True)
    clear_state_effect_cache()
    original = graph_module.infer_state_effects
    setattr(graph_module, "infer_state_effects", lambda component: NOT_APPLICABLE)
    try:
        optimizer.resolve_defaults()
        without_inference = optimizer._compile_plan(require_initializer=True)
    finally:
        setattr(graph_module, "infer_state_effects", original)
    assert without_inference is not None
    assert with_inference is not None

    added = [
        diagnostic
        for diagnostic in with_inference.diagnostics
        if diagnostic not in without_inference.diagnostics
    ]
    removed = [
        diagnostic
        for diagnostic in without_inference.diagnostics
        if diagnostic not in with_inference.diagnostics
    ]
    assert removed == []
    assert added
    assert all(diagnostic.severity is Severity.WARNING for diagnostic in added)
    assert all(diagnostic.code == CONFLICT_CODE for diagnostic in added)
    assert any(
        "uninitialized state write" in diagnostic.message for diagnostic in added
    )
    assert any(
        diagnostic.code == "cannot_verify_stage_flow"
        for diagnostic in with_inference.diagnostics
    )


_CONFLICT_KINDS = {
    "uninitialized state write": "uninitialized_state_write",
    "unreachable state read": "unreachable_state_read",
    "concurrent state read/write": "concurrent_state_read_write",
    "concurrent state write": "concurrent_state_write",
}
_STATE_KEY_PATTERN = re.compile(r"[A-Za-z0-9_.]+:[A-Za-z0-9_.]+:\d+")


def _conflict_kind(message: str) -> str:
    for phrase, kind in _CONFLICT_KINDS.items():
        if phrase in message:
            return kind
    raise AssertionError(message)


def _state_findings(plan: ExecutablePlan, *, declared: bool) -> set[tuple]:
    findings = set()
    for diagnostic in plan.diagnostics:
        if declared:
            if (
                diagnostic.severity is not Severity.ERROR
                or diagnostic.code not in _CONFLICT_KINDS.values()
            ):
                continue
            kind = diagnostic.code
        else:
            if (
                diagnostic.severity is not Severity.WARNING
                or diagnostic.code != CONFLICT_CODE
            ):
                continue
            kind = _conflict_kind(diagnostic.message)
        key = _STATE_KEY_PATTERN.search(diagnostic.message)
        if key is None:
            continue
        findings.add((diagnostic.path.components, key.group(0), kind))
    return findings


def _sequential_parity(scenario: str, *, declared: bool) -> ExecutablePlan:
    if scenario == "concurrent-writes":
        left_component = _DeclaredWriteProbeStage() if declared else _WriteProbeStage()
        right_component = _DeclaredWriteProbeStage() if declared else _WriteProbeStage()
        edges = ()
    elif scenario == "read-write":
        left_component = (
            _DeclaredReadWriteStage() if declared else _ReadWriteProbeStage()
        )
        right_component = (
            _DeclaredReadWriteStage() if declared else _ReadWriteProbeStage()
        )
        edges = ()
    else:
        left_component = _DeclaredReadProbeStage() if declared else _ReadProbeStage()
        right_component = _DeclaredWriteProbeStage() if declared else _WriteProbeStage()
        edges = (
            ControlEdge(
                source=NodeRef(component_id="left"),
                target=NodeRef(component_id="right"),
            ),
        )
    graph = ComponentGraph(
        nodes=(
            ComponentNode(component_id="left", component=left_component),
            ComponentNode(component_id="right", component=right_component),
        ),
        control_edges=edges,
        state_bindings=(),
        entry_points=(NodeRef(component_id="left"),),
    )
    return Compiler().compile(graph, CompileContext())


@pytest.mark.parametrize(
    "scenario", ["concurrent-writes", "read-write", "read-before-write"]
)
def test_sequential_inference_matches_a_hand_written_declaration(scenario) -> None:
    declared = _sequential_parity(scenario, declared=True)
    clear_state_effect_cache()
    inferred = _sequential_parity(scenario, declared=False)
    declared_findings = _state_findings(declared, declared=True)
    assert declared_findings
    assert declared_findings == _state_findings(inferred, declared=False)


def _structured_parity(*, declared: bool) -> ExecutablePlan:
    component = _DeclaredReadWriteStage() if declared else _ReadWriteProbeStage()
    optimizer = (
        Optimizer(_problem(), seed=5)
        .set_initializer(LHSInitializer(2, 2, 5))
        .set_algorithm(
            GA(
                CrossoverBLXAlpha(0.9, 0.5),
                MutationUniform(0.1),
                SequentialSelection(),
                TruncationSelection(),
            )
        )
        .set_strategy(TemplateStrategy(Pipeline([component])))
        .set_termination(Termination(max_gen(1)))
    )
    optimizer.resolve_defaults()
    plan = optimizer._compile_plan(require_initializer=True)
    assert plan is not None
    return plan


def test_structured_inference_matches_a_hand_written_declaration() -> None:
    declared = _structured_parity(declared=True)
    clear_state_effect_cache()
    inferred = _structured_parity(declared=False)
    declared_findings = _state_findings(declared, declared=True)
    assert declared_findings
    assert declared_findings == _state_findings(inferred, declared=False)


def test_pure_initialization_does_not_warn() -> None:
    node = ComponentNode(component_id="write", component=_WriteProbeStage())
    result = _compile_graph(node)
    assert not [
        diagnostic
        for diagnostic in result.diagnostics
        if diagnostic.code == CONFLICT_CODE
    ]


def test_sequential_conflicts_are_reported_as_warnings() -> None:
    node = ComponentNode(component_id="read_write", component=_ReadWriteProbeStage())
    result = _compile_graph(node)
    conflicts = [
        diagnostic
        for diagnostic in result.diagnostics
        if diagnostic.code == CONFLICT_CODE
    ]
    assert len(conflicts) == 1
    assert conflicts[0].severity is Severity.WARNING
    assert "uninitialized state write" in conflicts[0].message
    assert PROBE.namespace in conflicts[0].message

    reader = ComponentNode(component_id="read", component=_ReadProbeStage())
    writer = ComponentNode(component_id="write", component=_WriteProbeStage())
    graph = ComponentGraph(
        nodes=(reader, writer),
        control_edges=(
            ControlEdge(
                source=NodeRef(component_id="read"),
                target=NodeRef(component_id="write"),
            ),
        ),
        state_bindings=(),
        entry_points=(NodeRef(component_id="read"),),
    )
    result = Compiler().compile(graph, CompileContext())
    conflicts = [
        diagnostic
        for diagnostic in result.diagnostics
        if diagnostic.code == CONFLICT_CODE
    ]
    assert conflicts
    assert all(diagnostic.severity is Severity.WARNING for diagnostic in conflicts)
    assert any("unreachable state read" in item.message for item in conflicts)


@pytest.mark.parametrize(
    "component",
    [
        _GetattrStage(),
        _PartialStage(),
        _ExecStage(),
        _RaisingContract(),
        _EvaluatedStage(),
        _PropertyKeyStage(),
        _MethodCallerStage(),
        _LambdaStage(),
        _SuperOpaqueStage(),
        _WrappedStage(),
        _RecursiveStage(),
        _StaticMethodStage(),
        _ClassMethodStage(),
        _DynamicGetattrStage(),
        _UnreturnedReplaceStage(),
        _PositionalStatePatchStage(),
        _ContainerStage(),
        _UserConcreteStage(),
        _INSTANCE_OVERRIDE,
    ],
    ids=[
        "getattr",
        "partial",
        "exec",
        "raising-contract",
        "evaluated-key",
        "property-key",
        "methodcaller",
        "lambda",
        "super-opaque-base",
        "wrapped",
        "recursive",
        "staticmethod",
        "classmethod",
        "dynamic-getattr",
        "unreturned-replace",
        "positional-state-patch",
        "container",
        "user-abstract-base",
        "instance-override",
    ],
)
def test_unclassified_constructs_are_non_enumerable(component) -> None:
    result = infer_state_effects(component)
    if type(component) in (_RaisingContract, _UserConcreteStage):
        assert result.status is InferenceStatus.NOT_APPLICABLE
    else:
        assert result.status is InferenceStatus.UNKNOWN


def test_key_resolution_does_not_invoke_descriptors() -> None:
    assert infer_state_effects(_PropertyKeyStage()).status is InferenceStatus.UNKNOWN
    assert _PROPERTY_CALLS == []
    assert _EVALUATED == []


def test_cast_of_the_context_is_followed_as_an_alias() -> None:
    inferred = _inferred(_CastStage())
    assert PROBE in inferred.reads
    assert PROBE in inferred.writes


def test_a_class_is_analyzed_once_per_process(monkeypatch) -> None:
    calls = []
    original = _Analyzer.analyze_class

    def counted(self):
        calls.append(type(self).__name__)
        return original(self)

    monkeypatch.setattr(_Analyzer, "analyze_class", counted)
    clear_state_effect_cache()
    first = infer_state_effects(_WriteProbeStage())
    second = infer_state_effects(_WriteProbeStage())
    assert first == second
    assert len(calls) == 1


def test_concurrent_inference_analyzes_a_class_once(monkeypatch) -> None:
    calls = []
    original = _Analyzer.analyze_class

    def counted(self):
        calls.append(type(self).__name__)
        time.sleep(0.05)
        return original(self)

    monkeypatch.setattr(_Analyzer, "analyze_class", counted)
    clear_state_effect_cache()
    barrier = threading.Barrier(2)
    results: list[StateInference] = []

    def infer():
        barrier.wait()
        results.append(infer_state_effects(_WriteProbeStage()))

    threads = [threading.Thread(target=infer) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(calls) == 1
    assert results[0] == results[1]


def _checkpoint_projection(state, path):
    state.save(path)
    with np.load(path, allow_pickle=False) as data:
        return {name: data[name].copy() for name in data.files}


def _assert_same_projection(left, right) -> None:
    assert set(left) == set(right)
    for name in left:
        assert np.array_equal(left[name], right[name]), name


def test_checkpoint_contents_are_identical_with_and_without_inference(
    tmp_path,
) -> None:
    def run(directory):
        optimizer = (
            Optimizer(_problem(), seed=7)
            .set_initializer(LHSInitializer(2, 2, 7))
            .set_algorithm(
                GA(
                    CrossoverBLXAlpha(0.9, 0.5),
                    MutationUniform(0.1),
                    SequentialSelection(),
                    TruncationSelection(),
                )
            )
            .set_strategy(DirectStrategy())
            .set_termination(Termination(max_gen(1)))
        )
        state = optimizer.run()
        optimizer.save_pickle(state, directory / "optimizer.pkl")
        restored_optimizer, restored_state = Optimizer.load_pickle(
            directory / "optimizer.pkl"
        )
        assert restored_optimizer._executable_plan is None
        return (
            _checkpoint_projection(state, directory / "portable.npz"),
            _checkpoint_projection(restored_state, directory / "restored.npz"),
        )

    with_directory = tmp_path / "with"
    without_directory = tmp_path / "without"
    with_directory.mkdir()
    without_directory.mkdir()
    with_projection = run(with_directory)
    clear_state_effect_cache()
    original = graph_module.infer_state_effects
    setattr(graph_module, "infer_state_effects", lambda component: NOT_APPLICABLE)
    try:
        without_projection = run(without_directory)
    finally:
        setattr(graph_module, "infer_state_effects", original)

    _assert_same_projection(with_projection[0], without_projection[0])
    _assert_same_projection(with_projection[1], without_projection[1])


# --- syntax allowlist and scope handling -----------------------------------


class _NestedFunctionStage(Stage):
    def execute(self, state):
        def helper():
            state.set_state(PROBE, 1)

        return state


class _NestedLambdaStage(Stage):
    def execute(self, state):
        list(map(lambda _: state.set_state(PROBE, 1), range(1)))
        return state


class _NestedClassStage(Stage):
    def execute(self, state):
        class Inner:
            pass

        return state


class _ListCompStage(Stage):
    def execute(self, state):
        values = [state for _ in range(1)]
        if values:
            return state
        return state


class _SetCompStage(Stage):
    def execute(self, state):
        values = {state for _ in range(1)}
        if values:
            return state
        return state


class _DictCompStage(Stage):
    def execute(self, state):
        values = {index: state for index in range(1)}
        if values:
            return state
        return state


class _GeneratorExpStage(Stage):
    def execute(self, state):
        values = (state for _ in range(1))
        if values:
            return state
        return state


class _GlobalStage(Stage):
    def execute(self, state):
        global PROBE
        return state


class _NonlocalStage(Stage):
    def execute(self, state):
        value = 0

        def helper():
            nonlocal value
            value = 1

        return state


class _MatchStage(Stage):
    def execute(self, state):
        match state:
            case _:
                return state
        return state


class _WalrusStage(Stage):
    def execute(self, state):
        if value := 1:
            state.set_state(PROBE, value)
        return state


class _DelStage(Stage):
    def execute(self, state):
        del state


class _FStringStage(Stage):
    def execute(self, state):
        text = f"{state}"
        if text:
            return state
        return state


class _SubscriptStage(Stage):
    def execute(self, state):
        value = [1][0]
        state.set_state(PROBE, value)
        return state


class _SliceStage(Stage):
    def execute(self, state):
        values = [1, 2][0:1]
        if values:
            return state
        return state


class _StarredStage(Stage):
    def execute(self, state):
        values = [1, *[2]]
        if values:
            return state
        return state


class _AugAssignStage(Stage):
    def execute(self, state):
        value = 0
        value += 1
        state.set_state(PROBE, value)
        return state


class _YieldStage(Stage):
    def execute(self, state):
        yield state


class _AsyncStage(Stage):
    async def execute(self, state):  # ty: ignore[invalid-method-override]
        return state


class _IfStage(Stage):
    def execute(self, state):
        if True:
            state.set_state(PROBE, 1)
        return state


class _ForStage(Stage):
    def execute(self, state):
        for _ in range(1):
            state.set_state(PROBE, 1)
        return state


class _WhileStage(Stage):
    def execute(self, state):
        while False:
            state.set_state(PROBE, 1)
        return state


class _TryStage(Stage):
    def execute(self, state):
        try:
            state.set_state(PROBE, 1)
        except ValueError:
            return state
        return state


class _WithStage(Stage):
    def execute(self, state):
        with contextlib.nullcontext():
            state.set_state(PROBE, 1)
        return state


class _RebindStage(Stage):
    def execute(self, state):
        state = object()
        return state


class _ForShadowStage(Stage):
    def execute(self, state):
        for state in [object()]:
            return state
        return state


class _WithShadowStage(Stage):
    def execute(self, state):
        with contextlib.nullcontext() as state:
            return state
        return state


class _IterContextStage(Stage):
    def execute(self, state):
        for item in [state]:
            item.set_state(PROBE, 1)
        return state


@pytest.mark.parametrize(
    "component",
    [
        _NestedFunctionStage(),
        _NestedLambdaStage(),
        _NestedClassStage(),
        _ListCompStage(),
        _SetCompStage(),
        _DictCompStage(),
        _GeneratorExpStage(),
        _GlobalStage(),
        _NonlocalStage(),
        _MatchStage(),
        _WalrusStage(),
        _DelStage(),
        _FStringStage(),
        _SubscriptStage(),
        _SliceStage(),
        _StarredStage(),
        _AugAssignStage(),
        _YieldStage(),
        _AsyncStage(),
    ],
    ids=[
        "nested-function",
        "nested-lambda",
        "nested-class",
        "list-comprehension",
        "set-comprehension",
        "dict-comprehension",
        "generator-expression",
        "global",
        "nonlocal",
        "match",
        "walrus",
        "del",
        "f-string",
        "subscript",
        "slice",
        "starred",
        "augmented-assignment",
        "yield",
        "async",
    ],
)
def test_excluded_syntax_is_non_enumerable(component) -> None:
    assert infer_state_effects(component).status is InferenceStatus.UNKNOWN


@pytest.mark.parametrize(
    "component",
    [
        _IfStage(),
        _ForStage(),
        _WhileStage(),
        _TryStage(),
        _WithStage(),
    ],
    ids=["if", "for", "while", "try", "with"],
)
def test_allowed_syntax_is_inferred(component) -> None:
    inferred = _inferred(component)
    assert set(inferred.reads) == set()
    assert set(inferred.writes) == {PROBE}


@pytest.mark.parametrize(
    "node_type",
    [
        ast.Subscript,
        ast.Slice,
        ast.Starred,
        ast.AugAssign,
        ast.Delete,
        ast.Global,
        ast.Nonlocal,
        ast.Match,
        ast.NamedExpr,
        ast.JoinedStr,
        ast.FormattedValue,
        ast.Lambda,
        ast.FunctionDef,
        ast.AsyncFunctionDef,
        ast.ClassDef,
        ast.ListComp,
        ast.SetComp,
        ast.DictComp,
        ast.GeneratorExp,
        ast.comprehension,
        ast.Yield,
        ast.YieldFrom,
        ast.Await,
        ast.AsyncFor,
        ast.AsyncWith,
    ],
)
def test_excluded_node_types_are_not_allowlisted(node_type) -> None:
    assert node_type not in _ALLOWED_AST_NODES


@pytest.mark.parametrize(
    "node_type",
    [
        ast.Assign,
        ast.AnnAssign,
        ast.Return,
        ast.Call,
        ast.Attribute,
        ast.Name,
        ast.Constant,
        ast.If,
        ast.For,
        ast.While,
        ast.Try,
        ast.With,
        ast.Import,
        ast.ImportFrom,
        ast.Dict,
        ast.Tuple,
        ast.List,
        ast.Set,
        ast.BinOp,
        ast.Compare,
        ast.BoolOp,
        ast.keyword,
        ast.arg,
        ast.arguments,
    ],
)
def test_allowed_node_types_are_allowlisted(node_type) -> None:
    assert node_type in _ALLOWED_AST_NODES


@pytest.mark.parametrize(
    "component",
    [
        _RebindStage(),
        _ForShadowStage(),
        _WithShadowStage(),
        _IterContextStage(),
    ],
    ids=["rebind", "for-shadow", "with-shadow", "iterate-context"],
)
def test_context_rebinding_or_shadowing_is_non_enumerable(component) -> None:
    assert infer_state_effects(component).status is InferenceStatus.UNKNOWN


# --- structural non-execution ----------------------------------------------


_HOOK_CALLS: list[str] = []


class _Descriptor:
    def __get__(self, instance, owner=None):
        _HOOK_CALLS.append("descriptor")
        return None


class _DescriptorContractStage(Stage):
    contract = _Descriptor()

    def execute(self, state):
        return StatePatch(writes={PROBE: 1})


class _PropertyAnalysisStage(Stage):
    @property
    def analysis_contract(self):
        _HOOK_CALLS.append("property")
        return self.contract()

    def execute(self, state):
        return StatePatch(writes={PROBE: 1})


class _GetattrHookStage(Stage):
    def __getattr__(self, name):
        _HOOK_CALLS.append(f"getattr:{name}")
        raise AttributeError(name)

    def execute(self, state):
        return StatePatch(writes={PROBE: 1})


class _GetattributeHookStage(Stage):
    def __getattribute__(self, name):
        _HOOK_CALLS.append(f"getattribute:{name}")
        return object.__getattribute__(self, name)

    def execute(self, state):
        return StatePatch(writes={PROBE: 1})


class _ClassGetitemStage(Stage):
    def __class_getitem__(cls, item):
        _HOOK_CALLS.append("class-getitem")
        return cls

    def execute(self, state):
        return StatePatch(writes={PROBE: 1})


class _DictPropertyStage(Stage):
    @property
    def __dict__(self):
        _HOOK_CALLS.append("dict-property")
        return {}

    def execute(self, state):
        return StatePatch(writes={PROBE: 1})


class _Meta(abc.ABCMeta):
    @property
    def analysis_contract(cls):
        _HOOK_CALLS.append("metaclass")
        return ComponentContract(state=StateContract())


class _MetaclassStage(Stage, metaclass=_Meta):
    def execute(self, state):
        return StatePatch(writes={PROBE: 1})


@pytest.mark.parametrize(
    "component",
    [
        _DescriptorContractStage(),
        _PropertyAnalysisStage(),
        _GetattrHookStage(),
        _GetattributeHookStage(),
        _ClassGetitemStage(),
        _DictPropertyStage(),
        _MetaclassStage(),
    ],
    ids=[
        "descriptor",
        "property",
        "getattr",
        "getattribute",
        "class-getitem",
        "dict-property",
        "metaclass",
    ],
)
def test_no_user_hook_runs_during_inference(component) -> None:
    _HOOK_CALLS.clear()
    inferred = infer_state_effects(component)
    if type(component) is _ClassGetitemStage:
        assert inferred.status is InferenceStatus.INFERRED
    elif type(component) in (
        _GetattrHookStage,
        _GetattributeHookStage,
        _DictPropertyStage,
        _MetaclassStage,
    ):
        assert inferred.status is InferenceStatus.UNKNOWN
    else:
        assert inferred.status is InferenceStatus.NOT_APPLICABLE
    assert _HOOK_CALLS == []


# --- mixed and exhaustive declaration parity -------------------------------


class _DeclaredReaderStage(Stage):
    def contract(self) -> ComponentContract:
        return ComponentContract(state=StateContract(reads=(PROBE,)))

    def execute(self, state):
        return state


class _InferredReaderStage(Stage):
    def execute(self, state):
        state.get_state(PROBE)
        return state


class _DeclaredNoneStage(Stage):
    def contract(self) -> ComponentContract:
        return ComponentContract(state=StateContract())

    def execute(self, state):
        return state


def _state_graph(*components, ordered: bool = False) -> ExecutablePlan:
    nodes = tuple(
        ComponentNode(component_id=f"node{index}", component=component)
        for index, component in enumerate(components)
    )
    edges = (
        (
            ControlEdge(
                source=NodeRef(component_id="node0"),
                target=NodeRef(component_id="node1"),
            ),
        )
        if ordered and len(nodes) > 1
        else ()
    )
    graph = ComponentGraph(
        nodes=nodes,
        control_edges=edges,
        state_bindings=(),
        entry_points=(NodeRef(component_id="node0"),),
    )
    return Compiler().compile(graph, CompileContext())


def test_mixed_inferred_writer_and_declared_reader_parity() -> None:
    inferred = _state_graph(_WriteProbeStage(), _DeclaredReaderStage())
    reference = _state_graph(_DeclaredWriteProbeStage(), _DeclaredReaderStage())
    base = _state_findings(inferred, declared=True)
    new = _state_findings(inferred, declared=False)
    declared = _state_findings(reference, declared=True)

    assert declared <= base | new
    assert new
    assert new.isdisjoint(base)
    assert any(kind == "concurrent_state_read_write" for _, _, kind in new)


def test_mixed_declared_writer_and_inferred_reader_parity() -> None:
    inferred = _state_graph(_DeclaredWriteProbeStage(), _InferredReaderStage())
    reference = _state_graph(_DeclaredWriteProbeStage(), _DeclaredReaderStage())
    base = _state_findings(inferred, declared=True)
    new = _state_findings(inferred, declared=False)
    declared = _state_findings(reference, declared=True)

    assert declared <= base | new
    assert new
    assert new.isdisjoint(base)
    assert any(kind == "concurrent_state_read_write" for _, _, kind in new)


def test_structured_mixed_declaration_parity() -> None:
    inferred = _structured_parity_with(_ReadWriteProbeStage(), _DeclaredReaderStage())
    reference = _structured_parity_with(
        _DeclaredReadWriteStage(), _DeclaredReaderStage()
    )
    base = _state_findings(inferred, declared=True)
    new = _state_findings(inferred, declared=False)
    declared = _state_findings(reference, declared=True)

    assert declared <= base | new
    assert new.isdisjoint(base)


def _structured_parity_with(*components) -> ExecutablePlan:
    optimizer = (
        Optimizer(_problem(), seed=5)
        .set_initializer(LHSInitializer(2, 2, 5))
        .set_algorithm(
            GA(
                CrossoverBLXAlpha(0.9, 0.5),
                MutationUniform(0.1),
                SequentialSelection(),
                TruncationSelection(),
            )
        )
        .set_strategy(TemplateStrategy(Pipeline(list(components))))
        .set_termination(Termination(max_gen(1)))
    )
    optimizer.resolve_defaults()
    plan = optimizer._compile_plan(require_initializer=True)
    assert plan is not None
    return plan


_EFFECT_PAIRS = [
    (_NoopStage, _DeclaredNoneStage),
    (_InferredReaderStage, _DeclaredReaderStage),
    (_WriteProbeStage, _DeclaredWriteProbeStage),
    (_ReadWriteProbeStage, _DeclaredReadWriteStage),
]


@pytest.mark.parametrize("left_index", range(len(_EFFECT_PAIRS)))
@pytest.mark.parametrize("right_index", range(len(_EFFECT_PAIRS)))
@pytest.mark.parametrize("ordered", [False, True], ids=["unordered", "ordered"])
def test_small_graph_parity_with_hand_written_declarations(
    left_index, right_index, ordered
) -> None:
    clear_state_effect_cache()
    inferred = _state_graph(
        _EFFECT_PAIRS[left_index][0](),
        _EFFECT_PAIRS[right_index][0](),
        ordered=ordered,
    )
    clear_state_effect_cache()
    reference = _state_graph(
        _EFFECT_PAIRS[left_index][1](),
        _EFFECT_PAIRS[right_index][1](),
        ordered=ordered,
    )
    assert _state_findings(inferred, declared=False) == _state_findings(
        reference, declared=True
    )


def test_state_effect_rule_converts_the_shared_computation() -> None:
    from saealib.core.compiler.compiler import DiagnosticBag, RuleContext
    from saealib.core.compiler.state_effect_rules import (
        StateEffectRule,
        compute_state_effect_findings,
        declared_state_effects,
    )

    plan = _sequential_parity("read-write", declared=True)
    context = RuleContext(plan.graph, CompileContext(), DiagnosticBag())
    diagnostics = [
        diagnostic
        for diagnostic in StateEffectRule().apply(context).diagnostics
        if diagnostic.code != "opaque_state_effects"
    ]
    findings = compute_state_effect_findings(
        plan.graph, frozenset(), declared_state_effects(plan.graph)
    )
    assert [
        (
            diagnostic.path.components,
            diagnostic.code,
            tuple(item.components for item in diagnostic.related),
        )
        for diagnostic in diagnostics
    ] == [
        (
            (finding.node.component_id,),
            finding.kind,
            tuple((item.component_id,) for item in finding.related),
        )
        for finding in findings
    ]


@pytest.mark.parametrize("configuration", sorted(_builtin_configurations()))
def test_builtin_configurations_have_no_state_effect_findings(configuration) -> None:
    plan = _compile_plan(_builtin_configurations()[configuration], infer=True)
    state_codes = {
        "uninitialized_state_write",
        "unreachable_state_read",
        "concurrent_state_read_write",
        "concurrent_state_write",
    }
    assert not [
        diagnostic for diagnostic in plan.diagnostics if diagnostic.code in state_codes
    ]


def test_try_star_is_not_allowlisted_when_available() -> None:
    try_star = getattr(ast, "TryStar", None)
    if try_star is not None:
        assert try_star not in _ALLOWED_AST_NODES


class _UnusedHelperStage(Stage):
    def execute(self, state):
        return state

    def helper(self, state):
        state.set_state(PROBE, 1)


class _UnreachableBranchStage(Stage):
    def execute(self, state):
        if False:
            state.set_state(PROBE, 1)
        return state


def test_inference_counts_unused_context_methods_and_unexecuted_branches() -> None:
    helper = _inferred(_UnusedHelperStage())
    assert set(helper.reads) == set()
    assert set(helper.writes) == {PROBE}

    branch = _inferred(_UnreachableBranchStage())
    assert set(branch.reads) == set()
    assert set(branch.writes) == {PROBE}


# --- unknown inference at verification -------------------------------------


def _writer(state):
    state.set_state(PROBE, 1)


class _UnknownSelection(ParentSelection):
    def select(
        self, population, n_pair, n_parents, rng=np.random.default_rng(), *, ctx
    ) -> np.ndarray:
        return functools.partial(_writer, ctx)()


class _DeclaredOpaqueSelection(ParentSelection):
    def contract(self) -> ComponentContract:
        return ComponentContract(
            state=StateContract(reads_enumerable=False, writes_enumerable=False)
        )

    def select(
        self, population, n_pair, n_parents, rng=np.random.default_rng(), *, ctx
    ) -> np.ndarray:
        return functools.partial(_writer, ctx)()


class _UnknownStage(Stage):
    def execute(self, state):
        return state

    def helper(self, state):
        return functools.partial(_writer, state)()


class _DeclaredOpaqueStage(Stage):
    def contract(self) -> ComponentContract:
        return ComponentContract(
            state=StateContract(reads_enumerable=False, writes_enumerable=False)
        )

    def execute(self, state):
        return state

    def helper(self, state):
        return functools.partial(_writer, state)()


def test_unknown_inference_matches_a_non_enumerable_declaration_sequential() -> None:
    unknown = _state_graph(_UnknownSelection(), _UnknownSelection())
    reference = _state_graph(_DeclaredOpaqueSelection(), _DeclaredOpaqueSelection())
    unknown_opaque = [
        diagnostic.path.components
        for diagnostic in unknown.diagnostics
        if diagnostic.code == "opaque_state_effects"
    ]
    reference_opaque = [
        diagnostic.path.components
        for diagnostic in reference.diagnostics
        if diagnostic.code == "opaque_state_effects"
    ]
    assert unknown_opaque == reference_opaque
    assert not _state_findings(unknown, declared=True)
    assert not _state_findings(unknown, declared=False)
    assert not _state_findings(reference, declared=True)


def test_unknown_inference_matches_a_non_enumerable_declaration_structured() -> None:
    unknown = _structured_parity_with(_UnknownStage(), _UnknownStage())
    reference = _structured_parity_with(_DeclaredOpaqueStage(), _DeclaredOpaqueStage())
    unknown_opaque = [
        diagnostic.path.components
        for diagnostic in unknown.diagnostics
        if diagnostic.code == "opaque_state_effects"
    ]
    reference_opaque = [
        diagnostic.path.components
        for diagnostic in reference.diagnostics
        if diagnostic.code == "opaque_state_effects"
    ]
    assert unknown_opaque == reference_opaque
    assert not _state_findings(unknown, declared=True)
    assert not _state_findings(unknown, declared=False)


def _default_node_configurations():
    ga = GA(
        CrossoverBLXAlpha(0.9, 0.5),
        MutationUniform(0.1),
        SequentialSelection(),
        TruncationSelection(),
    )
    for algorithm, strategy in (
        (ga, DirectStrategy()),
        (PSO(), DirectStrategy()),
        (ga, SteadyStateStrategy()),
        (ga, IndividualBasedStrategy(0.5)),
        (ga, GenerationBasedStrategy(2)),
        (PSO(), GenerationBasedStrategy(2)),
        (PSO(), PreSelectionStrategy(4, 2)),
    ):
        yield (
            Optimizer(_problem(), seed=3)
            .set_initializer(LHSInitializer(2, 2, 3))
            .set_algorithm(algorithm)
            .set_strategy(strategy)
            .set_termination(Termination(max_fe(6)))
        )


def _default_node_family_bases() -> set[type]:
    families: set[type] = set()
    for optimizer in _default_node_configurations():
        optimizer.resolve_defaults()
        plan = optimizer._compile_plan(require_initializer=True)
        assert plan is not None
        for node in plan.graph.nodes:
            component = node.component
            stage = getattr(component, "stage", None)
            actual = stage if stage is not None else component
            for base in type(actual).__mro__:
                if "contract" in vars(base) and getattr(
                    base, "__abstractmethods__", ()
                ):
                    families.add(base)
    return families


def test_builtin_family_defaults_have_no_unknown_inference() -> None:
    node_families = _default_node_family_bases()
    assert Stage in node_families
    assert AcquisitionFunction in node_families
    assert FeedbackBuilder in node_families
    assert SurrogateManager in node_families
    assert Crossover not in node_families
    assert Mutation not in node_families
    assert TrainingSet not in node_families
    unknown = []
    for info in pkgutil.walk_packages(saealib.__path__, "saealib."):
        if info.name.startswith("saealib.viz"):
            continue
        try:
            module = importlib.import_module(info.name)
        except Exception:
            continue
        for _, cls in inspect.getmembers(module, inspect.isclass):
            if cls.__module__ != module.__name__:
                continue
            if not any(base in type(cls).__mro__ for base in node_families):
                continue
            try:
                instance = cast(Any, cls).__new__(cls)
            except Exception:
                continue
            if infer_state_effects(instance).status is InferenceStatus.UNKNOWN:
                unknown.append(f"{cls.__module__}.{cls.__qualname__}")
    assert unknown == []


# --- compile-level acceptance for unknown inference ------------------------

_MATRIX_FAMILIES = {
    "parent_selection": ParentSelection,
    "mutation": Mutation,
    "crossover": Crossover,
    "feedback_builder": FeedbackBuilder,
    "evaluation_planner": EvaluationPlanner,
    "stage": Stage,
    "acquisition": AcquisitionFunction,
    "surrogate_manager": SurrogateManager,
    "evaluator": Evaluator,
}
_MATRIX_FAILURES = ("partial", "getattr", "exec", "unreadable", "unsupported")


def _matrix_source(family: str, failure: str) -> str:
    context = "state" if family == "stage" else "ctx"
    if failure == "partial":
        statement = f"result = functools.partial(_writer, {context})()"
    elif failure == "getattr":
        statement = f'result = getattr({context}, "state", None)'
    elif failure == "unsupported":
        statement = "result = [1][0]"
    else:
        statement = f"result = {context}.set_state(PROBE, 1)"
    if family == "parent_selection":
        return (
            "class Probe(ParentSelection):\n"
            "    def select(self, population, n_pair, n_parents, "
            "rng=np.random.default_rng(), *, ctx) -> np.ndarray:\n"
            f"        {statement}\n"
            "        return population.extract([])\n"
        )
    if family == "mutation":
        return (
            "class Probe(Mutation):\n"
            "    def mutate_batch(self, candidates_batch, mutate_range, "
            "rng=np.random.default_rng()):\n"
            "        return candidates_batch\n"
            "    def helper(self, ctx):\n"
            f"        {statement}\n"
        )
    if family == "crossover":
        return (
            "class Probe(Crossover):\n"
            "    def crossover_batch(self, *args, **kwargs):\n"
            "        return args[0]\n"
            "    def helper(self, ctx):\n"
            f"        {statement}\n"
        )
    if family == "feedback_builder":
        return (
            "class Probe(FeedbackBuilder):\n"
            "    def build(self, candidates, prediction, evaluation, "
            "evaluated_indices, *, ctx):\n"
            f"        {statement}\n"
            "        return None\n"
        )
    if family == "evaluation_planner":
        return (
            "class Probe(EvaluationPlanner):\n"
            "    def plan(self, candidates, acquisition, *, ctx):\n"
            f"        {statement}\n"
            "        return None\n"
        )
    if family == "acquisition":
        return (
            "class Probe(AcquisitionFunction):\n"
            "    def evaluate(self, candidates_x, prediction, archive, *, "
            "prepared=None, ctx):\n"
            f"        {statement}\n"
            "        return None\n"
        )
    if family == "surrogate_manager":
        return (
            "class Probe(SurrogateManager):\n"
            "    def predict(self, candidates_x, archive, *, refit=True, ctx):\n"
            f"        {statement}\n"
            "        return None\n"
        )
    if family == "evaluator":
        return (
            "class Probe(Evaluator):\n"
            "    def evaluate_batch(self, x, problem):\n"
            "        return None\n"
            "    def helper(self, ctx):\n"
            f"        {statement}\n"
        )
    return (
        "class Probe(Stage):\n"
        "    def execute(self, state):\n"
        f"        {statement}\n"
        "        return state\n"
    )


def _matrix_component(family: str, failure: str) -> type:
    base = _MATRIX_FAMILIES[family]
    source = _matrix_source(family, failure)
    filename = f"<state-inference-{family}-{failure}>"
    if failure not in ("exec", "unreadable"):
        entry = (len(source), None, source.splitlines(True), filename)
        linecache.cache[filename] = entry
    namespace = {
        base.__name__: base,
        "functools": functools,
        "np": np,
        "PROBE": PROBE,
        "_writer": _writer,
    }
    exec(compile(source, filename, "exec"), namespace)
    return cast(type, namespace["Probe"])


@pytest.mark.parametrize("family", sorted(_MATRIX_FAMILIES))
@pytest.mark.parametrize("failure", _MATRIX_FAILURES)
def test_unknown_inference_reports_a_diagnostic_at_compile_time(
    family, failure
) -> None:
    component_type = _matrix_component(family, failure)
    inferred = _state_graph(component_type())
    clear_state_effect_cache()
    original = graph_module.infer_state_effects
    setattr(graph_module, "infer_state_effects", lambda component: NOT_APPLICABLE)
    try:
        declared = _state_graph(component_type())
    finally:
        setattr(graph_module, "infer_state_effects", original)
    opaque = [
        diagnostic
        for diagnostic in inferred.diagnostics
        if diagnostic.code == "opaque_state_effects"
    ]
    assert len(opaque) == 1
    assert inferred.contract_snapshots == declared.contract_snapshots
    for inferred_node, declared_node in zip(inferred.graph.nodes, declared.graph.nodes):
        assert inferred_node.contract == declared_node.contract
        assert inferred_node.analysis_contract == declared_node.analysis_contract
    assert not _state_findings(inferred, declared=False)


# --- undeclared shape x node-eligible family matrix -----------------------

_SHAPE_KINDS = (
    "metaclass",
    "instance_dict",
    "instance_override",
    "getattr_hook",
    "getattribute_hook",
    "slots",
    "exception",
)
_HOOK_NAMES = {
    "parent_selection": "select",
    "mutation": "mutate_batch",
    "crossover": "crossover_batch",
    "feedback_builder": "build",
    "evaluation_planner": "plan",
    "stage": "execute",
    "acquisition": "evaluate",
    "surrogate_manager": "predict",
    "evaluator": "evaluate_batch",
}


class _ShapeMeta(abc.ABCMeta):
    pass


def _shape_extras(kind: str) -> str:
    if kind == "instance_dict":
        return "    @property\n    def __dict__(self):\n        return {}\n"
    if kind == "getattr_hook":
        return "    def __getattr__(self, name):\n        raise AttributeError(name)\n"
    if kind == "getattribute_hook":
        return (
            "    def __getattribute__(self, name):\n"
            "        return object.__getattribute__(self, name)\n"
        )
    if kind == "slots":
        return "    __slots__ = ()\n"
    return ""


def _shape_hook(family: str, statement: str) -> str:
    context = "state" if family == "stage" else "ctx"
    if family == "parent_selection":
        return (
            "    def select(self, population, n_pair, n_parents, "
            "rng=np.random.default_rng(), *, ctx) -> np.ndarray:\n"
            f"        {statement.format(ctx=context)}\n"
            "        return population.extract([])\n"
        )
    if family == "mutation":
        return (
            "    def mutate_batch(self, candidates_batch, mutate_range, "
            "rng=np.random.default_rng()):\n"
            "        return candidates_batch\n"
            "    def helper(self, ctx):\n"
            f"        {statement.format(ctx='ctx')}\n"
        )
    if family == "crossover":
        return (
            "    def crossover_batch(self, *args, **kwargs):\n"
            "        return args[0]\n"
            "    def helper(self, ctx):\n"
            f"        {statement.format(ctx='ctx')}\n"
        )
    if family == "feedback_builder":
        return (
            "    def build(self, candidates, prediction, evaluation, "
            "evaluated_indices, *, ctx):\n"
            f"        {statement.format(ctx='ctx')}\n"
            "        return None\n"
        )
    if family == "evaluation_planner":
        return (
            "    def plan(self, candidates, acquisition, *, ctx):\n"
            f"        {statement.format(ctx='ctx')}\n"
            "        return None\n"
        )
    if family == "acquisition":
        return (
            "    def evaluate(self, candidates_x, prediction, archive, *, "
            "prepared=None, ctx):\n"
            f"        {statement.format(ctx='ctx')}\n"
            "        return None\n"
        )
    if family == "surrogate_manager":
        return (
            "    def predict(self, candidates_x, archive, *, refit=True, ctx):\n"
            f"        {statement.format(ctx='ctx')}\n"
            "        return None\n"
        )
    if family == "evaluator":
        return (
            "    def evaluate_batch(self, x, problem):\n"
            "        return None\n"
            "    def helper(self, ctx):\n"
            f"        {statement.format(ctx='ctx')}\n"
        )
    return (
        "    def execute(self, state):\n"
        f"        {statement.format(ctx='state')}\n"
        "        return state\n"
    )


def _shape_source(family: str, kind: str, declared: bool) -> str:
    if kind == "partial":
        statement = "result = functools.partial(_writer, {ctx})()"
    elif kind == "getattr_context":
        statement = 'result = getattr({ctx}, "state", None)'
    elif kind == "unsupported":
        statement = "result = [1][0]"
    elif kind in ("exec", "unreadable"):
        statement = "result = {ctx}.set_state(PROBE, 1)"
    else:
        statement = "result = {ctx}.set_state(PROBE, 1)"
    metaclass = ", metaclass=ShapeMeta" if kind == "metaclass" else ""
    contract = (
        "    def contract(self):\n"
        "        return replace(\n"
        "            super().contract(),\n"
        "            state=StateContract(writes=(PROBE,)),\n"
        "        )\n"
        if declared
        else ""
    )
    return (
        f"class Probe({_MATRIX_FAMILIES[family].__name__}{metaclass}):\n"
        f"{contract}{_shape_hook(family, statement)}{_shape_extras(kind)}"
    )


def _custom_instance_dict_stage_inference() -> StateInference:
    return infer_state_effects(
        _shape_component("stage", "instance_dict", declared=False)
    )


def _shape_component(family: str, kind: str, declared: bool) -> object:
    source = _shape_source(family, kind, declared)
    filename = f"<state-shape-{family}-{kind}-{declared}>"
    if kind not in ("exec", "unreadable"):
        entry = (len(source), None, source.splitlines(True), filename)
        linecache.cache[filename] = entry
    namespace = {
        _MATRIX_FAMILIES[family].__name__: _MATRIX_FAMILIES[family],
        "ShapeMeta": _ShapeMeta,
        "functools": functools,
        "np": np,
        "PROBE": PROBE,
        "_writer": _writer,
        "replace": dataclasses.replace,
        "StateContract": StateContract,
    }
    exec(compile(source, filename, "exec"), namespace)
    return cast(type, namespace["Probe"])()


def _shape_graph(component: object, graph_type: str) -> ExecutablePlan:
    if graph_type == "sequential":
        return _state_graph(component)
    return _structured_parity_with(component)


@pytest.mark.parametrize("family", sorted(_MATRIX_FAMILIES))
@pytest.mark.parametrize(
    "kind",
    (*_SHAPE_KINDS, "partial", "getattr_context", "exec", "unreadable", "unsupported"),
)
@pytest.mark.parametrize("graph_type", ["sequential", "structured"])
def test_undeclared_shape_matrix(family, kind, graph_type, monkeypatch) -> None:
    from saealib.core import state_inference

    if graph_type == "structured" and family == "stage" and kind == "instance_dict":
        assert _custom_instance_dict_stage_inference().status is InferenceStatus.UNKNOWN
        return

    declared_component = _shape_component(family, kind, declared=True)
    declared_plan = _shape_graph(declared_component, graph_type)
    assert not [
        diagnostic
        for diagnostic in declared_plan.diagnostics
        if diagnostic.code == "opaque_state_effects"
    ]

    with monkeypatch.context() as patch:
        if kind == "exception":

            def raise_decision(*args, **kwargs):
                raise RuntimeError("decision failure")

            patch.setattr(state_inference, "_is_family_default", raise_decision)
        undeclared_component = _shape_component(family, kind, declared=False)
        if kind == "instance_override":
            setattr(undeclared_component, _HOOK_NAMES[family], lambda *a, **k: None)
        undeclared_plan = _shape_graph(undeclared_component, graph_type)
    diagnostics = [
        diagnostic
        for diagnostic in undeclared_plan.diagnostics
        if diagnostic.code in ("opaque_state_effects", "cannot_verify_stage_flow")
    ]
    assert len(diagnostics) == 1

    original = graph_module.infer_state_effects
    setattr(graph_module, "infer_state_effects", lambda component: NOT_APPLICABLE)
    try:
        disabled_plan = _shape_graph(
            _shape_component(family, kind, declared=False), graph_type
        )
    finally:
        setattr(graph_module, "infer_state_effects", original)
    assert undeclared_plan.contract_snapshots == disabled_plan.contract_snapshots
    for inferred_node, disabled_node in zip(
        undeclared_plan.graph.nodes, disabled_plan.graph.nodes
    ):
        assert inferred_node.contract == disabled_node.contract
        assert inferred_node.analysis_contract == disabled_node.analysis_contract


def test_stage_with_a_custom_instance_dict_is_not_enumerable() -> None:
    # Stage resolution assigns ``resolved.__dict__`` when it prepares a stage,
    # so a structured graph cannot carry a stage whose ``__dict__`` is a custom
    # property.  The analysis itself is reachable directly and must stay
    # non-enumerable, since names cannot be resolved through that shape.
    assert _custom_instance_dict_stage_inference().status is InferenceStatus.UNKNOWN


# --- the context may appear only inside a modeled form ---------------------

SHARED_KEY = StateKey(namespace="user", name="shared", schema_version=1)
LOCAL_KEY = StateKey(namespace="user", name="local", schema_version=1)
BOXED_KEY = StateKey(namespace="user", name="boxed", schema_version=1)


def _write_boxed(items):
    items[0].set_state(BOXED_KEY, 1)


class _SharedKeyStage(Stage):
    def execute(self, state):
        state.set_state(SHARED_KEY, 1)
        return state


class _ConditionalAliasStage(Stage):
    enabled = True

    def execute(self, state):
        alias = state if self.enabled else state  # noqa: RUF034
        alias.set_state(BOXED_KEY, 1)
        return state


class _BoundMethodAliasStage(Stage):
    def execute(self, state):
        writer = state.set_state
        writer(BOXED_KEY, 1)
        return state


class _BooleanAliasStage(Stage):
    def execute(self, state):
        (state or state).set_state(BOXED_KEY, 1)
        return state


class _ContainerArgumentStage(Stage):
    def execute(self, state):
        _write_boxed([state])
        return state


class _LocallyAssignedKeyStage(Stage):
    def execute(self, state):
        SHARED_KEY = LOCAL_KEY
        state.set_state(SHARED_KEY, 1)
        return state


class _DefaultedKeyStage(Stage):
    def execute(self, state, SHARED_KEY=LOCAL_KEY):  # noqa: N803
        state.set_state(SHARED_KEY, 1)
        return state


class _LoopBoundKeyStage(Stage):
    def execute(self, state):
        for SHARED_KEY in [LOCAL_KEY]:
            state.set_state(SHARED_KEY, 1)
        return state


def _patch_factory(**kwargs):
    return StatePatch(**kwargs)


def _identity(kind, value):
    return value


class _ShadowedPatchStage(Stage):
    def execute(self, state):
        StatePatch = _patch_factory
        return StatePatch(writes={BOXED_KEY: 1})


class _ShadowedCastStage(Stage):
    def execute(self, state):
        cast = _identity
        alias = cast(object, state)
        alias.set_state(BOXED_KEY, 1)
        return state


class _AnyKeyReaderStage(Stage):
    def contract(self) -> ComponentContract:
        return ComponentContract(
            state=StateContract(reads=(SHARED_KEY, LOCAL_KEY, BOXED_KEY))
        )

    def execute(self, state):
        return state


_UNMODELED_CONTEXT_STAGES = (
    _ConditionalAliasStage,
    _BoundMethodAliasStage,
    _BooleanAliasStage,
    _ContainerArgumentStage,
    _LocallyAssignedKeyStage,
    _DefaultedKeyStage,
    _LoopBoundKeyStage,
    _ShadowedPatchStage,
    _ShadowedCastStage,
)
_UNMODELED_CONTEXT_IDS = (
    "conditional-alias",
    "bound-method-alias",
    "boolean-alias",
    "container-argument",
    "locally-assigned-key",
    "defaulted-key",
    "loop-bound-key",
    "shadowed-state-patch",
    "shadowed-cast",
)


def test_an_unbound_key_name_resolves_through_the_module() -> None:
    assert set(_inferred(_SharedKeyStage()).writes) == {SHARED_KEY}


@pytest.mark.parametrize(
    "component_type", _UNMODELED_CONTEXT_STAGES, ids=_UNMODELED_CONTEXT_IDS
)
def test_a_context_outside_the_modeled_forms_is_not_enumerable(component_type) -> None:
    clear_state_effect_cache()
    assert infer_state_effects(component_type()).status is InferenceStatus.UNKNOWN


@pytest.mark.parametrize(
    "component_type", _UNMODELED_CONTEXT_STAGES, ids=_UNMODELED_CONTEXT_IDS
)
def test_an_unmodeled_context_use_adds_no_conflict_warning(component_type) -> None:
    graph = ComponentGraph(
        nodes=(
            ComponentNode(component_id="stage", component=component_type()),
            ComponentNode(component_id="reader", component=_AnyKeyReaderStage()),
        ),
        control_edges=(
            ControlEdge(
                source=NodeRef(component_id="stage"),
                target=NodeRef(component_id="reader"),
            ),
        ),
        state_bindings=(),
        entry_points=(NodeRef(component_id="stage"),),
    )
    plan = Compiler().compile(graph, CompileContext())
    assert not [
        diagnostic
        for diagnostic in plan.diagnostics
        if diagnostic.code == CONFLICT_CODE
    ]
