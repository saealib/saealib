from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any, ClassVar, cast

import pytest

from saealib import (
    GA,
    CrossoverBLXAlpha,
    MutationUniform,
    SequentialSelection,
    TruncationSelection,
)
from saealib.acquisition import MeanPrediction
from saealib.core.compiler import (
    CompileContext,
    Compiler,
    DiagnosticBag,
    RuleContext,
    lower_pipeline,
)
from saealib.core.compiler.adapters import Adapter, AdapterRegistry, LosslessAdapterRule
from saealib.core.compiler.graph import DataEdge
from saealib.core.contracts import (
    CallInput,
    CallOutput,
    ComponentCall,
    ComponentContract,
    DataSpec,
    ExecutionContract,
    Fixed,
    FlowEdge,
    PartSpec,
    PortContract,
    PortDirection,
    PortSpec,
    ServiceRequirement,
    StageFlowSummary,
    StateContract,
    StateValue,
    Transform,
)
from saealib.core.graph_builder import (
    build_decomposed_component_graph_from_stages,
    decompose_structured_graph,
)
from saealib.core.state import StateKey
from saealib.exceptions import ValidationError
from saealib.pipeline import Branch, Loop, Pipeline, Repeat, Stage
from saealib.stages import (
    EvaluationApplyStage,
    EvaluationCollectStage,
    FeedbackStage,
    SurrogateOnlyLoopStage,
    _builtin_stage_instances_for_contracts,
    discover_builtin_stages,
    stage_component,
)
from saealib.surrogate.manager import LocalSurrogateManager
from saealib.surrogate.rbf import RBFSurrogate
from saealib.surrogate.rbf_kernels import GaussianKernel


class _Component:
    def __init__(self, role: str, *, input_kind: str | None, output_kind: str | None):
        self._role = role
        self._input_kind = input_kind
        self._output_kind = output_kind

    def contract(self) -> ComponentContract:
        inputs = (
            ()
            if self._input_kind is None
            else (
                PortSpec(
                    name="in",
                    direction=PortDirection.INPUT,
                    data=DataSpec(kind=self._input_kind),
                    cardinality="MANY",
                ),
            )
        )
        outputs = (
            ()
            if self._output_kind is None
            else (
                PortSpec(
                    name="out",
                    direction=PortDirection.OUTPUT,
                    data=DataSpec(kind=self._output_kind),
                    cardinality="MANY",
                ),
            )
        )
        return ComponentContract(
            ports={self._role: PortContract(inputs=inputs, outputs=outputs)}
        )


class _SummaryStage(Stage):
    def __init__(self, name: str, *, bad: bool = False, bad_port: bool = False) -> None:
        super().__init__(name=name)
        self.first = _Component(
            "producer", input_kind=None, output_kind="EvaluationRequestBatch"
        )
        self.second = _Component(
            "consumer", input_kind="CandidateSelection", output_kind=None
        )
        self._bad = bad
        self._bad_port = bad_port

    def flow_summary(self) -> StageFlowSummary:
        first = ComponentCall(
            name="first", role="producer", attribute="missing" if self._bad else "first"
        )
        second = ComponentCall(name="second", role="consumer", attribute="second")
        return StageFlowSummary(
            calls=(first, second),
            edges=(
                FlowEdge(
                    source=CallOutput(call=first, port="out"),
                    target=CallInput(
                        call=second, port="missing" if self._bad_port else "in"
                    ),
                    transforms=(Transform(name="select_requested_candidates"),),
                ),
            ),
        )

    def execute(self, state):
        return state


def test_summary_edges_bind_declared_component_objects() -> None:
    stage = _SummaryStage("step")
    graph = build_decomposed_component_graph_from_stages([stage])
    first = next(node for node in graph.nodes if node.component_id == "step__first")
    edge = next(edge for edge in graph.data_edges if isinstance(edge, DataEdge))
    assert first.component is stage.first
    assert edge.source.component_id == "step__first"
    assert edge.target.component_id == "step__second"
    assert edge.transforms == (Transform(name="select_requested_candidates"),)


def test_summary_missing_attribute_is_rejected() -> None:
    with pytest.raises(ValidationError, match="missing attribute"):
        build_decomposed_component_graph_from_stages([_SummaryStage("step", bad=True)])


def test_repeated_stage_names_get_stable_occurrence_ids() -> None:
    stages = (
        _SummaryStage("step"),
        _SummaryStage("step"),
        _SummaryStage("step_2"),
        _SummaryStage("step__first"),
    )
    first = build_decomposed_component_graph_from_stages(stages)
    second = build_decomposed_component_graph_from_stages(stages)
    first_plan = Compiler().compile(first)
    second_plan = Compiler().compile(second)
    first_ids = [node.component_id for node in first_plan.graph.nodes]
    second_ids = [node.component_id for node in second_plan.graph.nodes]
    assert first_ids == second_ids
    assert len(first_ids) == len(set(first_ids))
    assert {"step__first_2", "step_2__first", "step_2_2__first"} <= set(first_ids)


def test_builtin_stages_declare_known_or_state_only_flow() -> None:
    stages = _builtin_stage_instances_for_contracts()
    assert len(stages) == len(discover_builtin_stages()) == 20
    assert all(isinstance(stage.flow_summary(), StageFlowSummary) for stage in stages)
    for stage in stages:
        summary = stage.flow_summary()
        assert summary is not None
        part_names = {part.name for part in stage.contract().parts}
        assert all(call.attribute in part_names for call in summary.calls if part_names)


def test_summary_port_mismatch_is_rejected() -> None:
    with pytest.raises(ValidationError, match="does not declare its input port"):
        build_decomposed_component_graph_from_stages(
            [_SummaryStage("step", bad_port=True)]
        )


def test_repeated_calls_and_same_type_parts_keep_invocation_identity() -> None:
    class RepeatedStage(Stage):
        def __init__(self) -> None:
            super().__init__(name="repeated")
            self.left = _Component(
                "producer", input_kind=None, output_kind="Population"
            )
            self.right = _Component(
                "producer", input_kind=None, output_kind="Population"
            )

        def flow_summary(self) -> StageFlowSummary:
            left = ComponentCall(name="left_call", role="producer", attribute="left")
            right = ComponentCall(name="right_call", role="producer", attribute="right")
            repeated = ComponentCall(
                name="repeated_call", role="producer", attribute="left"
            )
            return StageFlowSummary(calls=(left, right, repeated))

        def execute(self, state):
            return state

    stage = RepeatedStage()
    graph = build_decomposed_component_graph_from_stages([stage])
    calls = [node for node in graph.nodes if node.component_id.startswith("repeated__")]
    assert len({node.component_id for node in calls}) == 3
    assert calls[0].component is stage.left
    assert calls[1].component is stage.right
    assert calls[2].component is stage.left
    assert type(calls[0].component) is type(calls[1].component)


def test_unknown_flow_warns_once_and_stage_still_executes() -> None:
    class UnknownStage(Stage):
        def __init__(self) -> None:
            super().__init__(name="unknown")
            self.called = False

        def execute(self, state):
            self.called = True
            return state

    stage = UnknownStage()
    graph = build_decomposed_component_graph_from_stages([stage])
    plan = Compiler().compile(graph)
    warnings = [item for item in plan.diagnostics if item.severity.value == "warning"]
    assert len(warnings) == 1
    assert warnings[0].code == "cannot_verify_stage_flow"
    assert not [item for item in plan.diagnostics if item.severity.value == "error"]
    assert not plan.graph.data_edges
    marker = object()
    assert cast(Any, graph.node_by_id("unknown").component).execute(marker) is marker
    assert stage.called


def test_unknown_flow_write_does_not_make_a_later_state_read_unreachable() -> None:
    key = StateKey[object](namespace="tests", name="unknown_write", schema_version=1)

    class UnknownWriter(Stage):
        def __init__(self) -> None:
            super().__init__(name="unknown_writer")

        def execute(self, state):
            return state

    class Reader(Stage):
        def __init__(self) -> None:
            super().__init__(name="reader")

        def contract(self) -> ComponentContract:
            return ComponentContract(state=StateContract(reads=(key,)))

        def execute(self, state):
            return state

    stages = [UnknownWriter(), Reader()]
    graphs = (
        decompose_structured_graph(
            lower_pipeline(Pipeline([stage_component(stage) for stage in stages]))
        ),
        build_decomposed_component_graph_from_stages(stages),
    )
    for graph in graphs:
        plan = Compiler().compile(graph)
        assert not [item for item in plan.diagnostics if item.severity.value == "error"]
        assert any(item.code == "cannot_verify_stage_flow" for item in plan.diagnostics)


_FLOW_STATE = StateKey[object](namespace="tests", name="flow_value", schema_version=1)


class _StateComponent:
    def __init__(self, *, output: bool) -> None:
        self._output = output

    def contract(self) -> ComponentContract:
        ports = (
            PortContract(
                outputs=(
                    PortSpec(
                        name="value",
                        direction=PortDirection.OUTPUT,
                        data=DataSpec(kind="CandidateSelection"),
                        cardinality="MANY",
                    ),
                )
            )
            if self._output
            else PortContract(
                inputs=(
                    PortSpec(
                        name="value",
                        direction=PortDirection.INPUT,
                        data=DataSpec(kind="CandidateSelection"),
                        cardinality="MANY",
                    ),
                )
            )
        )
        return ComponentContract(ports={"value_role": ports})


class _StateProducerStage(Stage):
    def __init__(self, name: str) -> None:
        super().__init__(name=name)
        self.part = _StateComponent(output=True)

    def contract(self) -> ComponentContract:
        return ComponentContract(
            parts=(PartSpec(name="part", contract=self.part.contract()),),
            state=StateContract(writes=(_FLOW_STATE,)),
        )

    def flow_summary(self) -> StageFlowSummary:
        call = ComponentCall(name="writer", role="value_role", attribute="part")
        return StageFlowSummary(
            calls=(call,),
            edges=(
                FlowEdge(
                    source=CallOutput(call=call, port="value"),
                    target=StateValue(key=_FLOW_STATE),
                ),
            ),
        )

    def execute(self, state):
        return state


class _StateConsumerStage(Stage):
    def __init__(self, name: str = "reader") -> None:
        super().__init__(name=name)
        self.part = _StateComponent(output=False)

    def contract(self) -> ComponentContract:
        return ComponentContract(
            parts=(PartSpec(name="part", contract=self.part.contract()),),
            state=StateContract(reads=(_FLOW_STATE,)),
        )

    def flow_summary(self) -> StageFlowSummary:
        call = ComponentCall(name="reader", role="value_role", attribute="part")
        return StageFlowSummary(
            calls=(call,),
            edges=(
                FlowEdge(
                    source=StateValue(key=_FLOW_STATE),
                    target=CallInput(call=call, port="value"),
                ),
            ),
        )

    def execute(self, state):
        return state


class _Always:
    def contract(self) -> StateContract:
        return StateContract()

    def evaluate(self, view) -> bool:
        del view
        return True


def _compile_stages(pipeline: Pipeline, *, initial: bool = False):
    def wrap(entry):
        if isinstance(entry, Stage):
            return stage_component(entry)
        if isinstance(entry, Pipeline):
            return Pipeline([wrap(item) for item in entry.stages], name=entry.name)
        if isinstance(entry, Branch):
            return Branch(
                entry.condition,
                then=wrap(entry.then),
                else_=None if entry.else_ is None else wrap(entry.else_),
                name=entry.name,
            )
        if isinstance(entry, Repeat):
            return Repeat(wrap(entry.body), count=entry.count, name=entry.name)
        if isinstance(entry, Loop):
            return Loop(wrap(entry.body), until=entry.until, name=entry.name)
        return entry

    graph = decompose_structured_graph(lower_pipeline(wrap(pipeline)))
    initial_keys = frozenset({_FLOW_STATE}) if initial else frozenset()
    return Compiler().compile(graph, CompileContext(initial_state_keys=initial_keys))


def test_external_state_input_has_a_state_key_source() -> None:
    plan = _compile_stages(Pipeline([_StateConsumerStage()]), initial=True)
    reader = next(
        node
        for node in plan.graph.nodes
        if node.component is not None and node.role == "value_role"
    )
    edges = [
        edge
        for edge in plan.graph.data_edges
        if edge.target.component_id == reader.component_id
    ]
    assert len(edges) == 1
    source = plan.graph.node_by_id(edges[0].source.component_id).component
    assert source.state_key == _FLOW_STATE
    assert not [item for item in plan.diagnostics if item.severity.value == "error"]


def test_branch_producers_remain_alternatives() -> None:
    left = _StateProducerStage("left_writer")
    right = _StateProducerStage("right_writer")
    consumer = _StateConsumerStage()
    plan = _compile_stages(
        Pipeline(
            [
                Branch(
                    _Always(),
                    then=Pipeline([left]),
                    else_=Pipeline([right]),
                    name="choice",
                ),
                consumer,
            ]
        )
    )
    reader_id = next(
        component_id
        for component_id, names in plan.graph.metadata["stage_flow_calls"].items()
        if "reader" in names
    )
    consumer_id = plan.graph.metadata["stage_flow_calls"][reader_id]["reader"]
    sources = {
        getattr(
            plan.graph.node_by_id(edge.source.component_id).component,
            "flow_origin",
            None,
        )
        for edge in plan.graph.data_edges
        if edge.target.component_id == consumer_id
    }
    assert len(sources) == 2
    assert any(
        isinstance(source, str) and "left_writer__writer" in source
        for source in sources
    )
    assert any(
        isinstance(source, str) and "right_writer__writer" in source
        for source in sources
    )


def test_zero_repeat_does_not_supply_a_state_value() -> None:
    writer = _StateProducerStage("writer")
    plan = _compile_stages(
        Pipeline(
            [Repeat(Pipeline([writer]), count=0, name="skip"), _StateConsumerStage()]
        )
    )
    writer_id = next(
        component_id
        for stage_id, names in plan.graph.metadata["stage_flow_calls"].items()
        if plan.graph.node_by_id(stage_id).component.stage.name == "writer"
        for component_id in (names["writer"],)
    )
    consumer_id = next(
        names["reader"]
        for names in plan.graph.metadata["stage_flow_calls"].values()
        if "reader" in names
    )
    assert not any(
        edge.source.component_id == writer_id
        and edge.target.component_id == consumer_id
        for edge in plan.graph.data_edges
    )
    assert any(item.code == "unreachable_state_read" for item in plan.diagnostics)


def test_loop_reader_accepts_entry_and_back_edge_producers() -> None:
    writer = _StateProducerStage("writer")
    consumer = _StateConsumerStage("reader")
    plan = _compile_stages(
        Pipeline(
            [
                Loop(
                    Pipeline([consumer, writer]),
                    until=_Always(),
                    name="cycle",
                )
            ]
        ),
        initial=True,
    )
    ids = plan.graph.metadata["stage_flow_calls"]
    reader_id = next(names["reader"] for names in ids.values() if "reader" in names)
    incoming = [
        edge for edge in plan.graph.data_edges if edge.target.component_id == reader_id
    ]
    assert len(incoming) == 2
    assert any(
        plan.graph.node_by_id(edge.source.component_id).component.state_key
        == _FLOW_STATE
        for edge in incoming
    )
    assert any(
        getattr(
            plan.graph.node_by_id(edge.source.component_id).component,
            "flow_origin",
            None,
        )
        == "cycle.writer__writer"
        for edge in incoming
    )


def test_region_joins_keep_all_state_producers_and_discard_overwritten_values() -> None:
    class TypedWriter(Stage):
        def __init__(self, name: str, kind: str) -> None:
            super().__init__(name=name)
            self.part = _Component("value_role", input_kind=None, output_kind=kind)

        def contract(self) -> ComponentContract:
            return ComponentContract(
                parts=(PartSpec(name="part", contract=self.part.contract()),),
                state=StateContract(writes=(_FLOW_STATE,)),
            )

        def flow_summary(self) -> StageFlowSummary:
            writer = ComponentCall(name="writer", role="value_role", attribute="part")
            return StageFlowSummary(
                calls=(writer,),
                edges=(
                    FlowEdge(
                        source=CallOutput(call=writer, port="out"),
                        target=StateValue(key=_FLOW_STATE),
                    ),
                ),
            )

        def execute(self, state):
            return state

    reader = _StateConsumerStage()
    mismatched = _compile_stages(
        Pipeline(
            [
                Loop(
                    Pipeline(
                        [
                            Branch(
                                _Always(),
                                then=Pipeline(
                                    [TypedWriter("left", "CandidateSelection")]
                                ),
                                else_=Pipeline([TypedWriter("right", "GenomeBatch")]),
                                name="choice",
                            )
                        ]
                    ),
                    until=_Always(),
                    name="cycle",
                ),
                reader,
            ]
        ),
        initial=True,
    )
    right_writer_flow = next(
        node.component_id
        for node in mismatched.graph.nodes
        if getattr(node.component, "flow_origin", "").endswith("right__writer")
    )
    assert any(
        item.code == "incompatible_port"
        and item.path.components == (right_writer_flow,)
        for item in mismatched.diagnostics
    )

    for region, producer_count in (
        (
            Branch(
                _Always(),
                then=Pipeline([TypedWriter("left", "CandidateSelection")]),
                else_=Pipeline([TypedWriter("right", "CandidateSelection")]),
                name="choice",
            ),
            2,
        ),
        (
            Repeat(
                Pipeline([TypedWriter("inside", "CandidateSelection")]),
                count=1,
                name="repeat",
            ),
            1,
        ),
    ):
        plan = _compile_stages(
            Pipeline(
                [TypedWriter("before", "GenomeBatch"), region, _StateConsumerStage()]
            ),
            initial=True,
        )
        assert not [item for item in plan.diagnostics if item.severity.value == "error"]
        reader_id = next(
            names["reader"]
            for names in plan.graph.metadata["stage_flow_calls"].values()
            if "reader" in names
        )
        incoming = [
            edge
            for edge in plan.graph.data_edges
            if edge.target.component_id == reader_id
        ]
        assert len(incoming) == producer_count
        assert {
            plan.graph.node_by_id(edge.source.component_id).role for edge in incoming
        } == {"state"}


def test_state_transform_after_a_branch_keeps_every_producer_edge() -> None:
    source_key = StateKey[object](
        namespace="tests", name="transform_source", schema_version=1
    )
    target_key = StateKey[object](
        namespace="tests", name="transform_target", schema_version=1
    )

    class PopulationWriter(Stage):
        def __init__(self, name: str) -> None:
            super().__init__(name=name)
            self.part = _Component(
                "population_writer", input_kind=None, output_kind="Population"
            )

        def contract(self) -> ComponentContract:
            return ComponentContract(
                parts=(PartSpec(name="part", contract=self.part.contract()),),
                state=StateContract(writes=(source_key,)),
            )

        def flow_summary(self) -> StageFlowSummary:
            call = ComponentCall(
                name="writer", role="population_writer", attribute="part"
            )
            return StageFlowSummary(
                calls=(call,),
                edges=(
                    FlowEdge(
                        source=CallOutput(call=call, port="out"),
                        target=StateValue(key=source_key),
                    ),
                ),
            )

        def execute(self, state):
            return state

    class TransformState(Stage):
        def __init__(self) -> None:
            super().__init__(name="derive")

        def contract(self) -> ComponentContract:
            return ComponentContract(
                state=StateContract(reads=(source_key,), writes=(target_key,))
            )

        def flow_summary(self) -> StageFlowSummary:
            return StageFlowSummary(
                edges=(
                    FlowEdge(
                        source=StateValue(key=source_key),
                        target=StateValue(key=target_key),
                        transforms=(Transform(name="extract_population_genomes"),),
                    ),
                )
            )

        def execute(self, state):
            return state

    branch = Branch(
        _Always(),
        then=Pipeline([PopulationWriter("left")]),
        else_=Pipeline([PopulationWriter("right")]),
        name="choice",
    )
    plan = _compile_stages(Pipeline([branch, TransformState()]))
    transformed = next(
        node
        for node in plan.graph.nodes
        if getattr(node.component, "flow_origin", None) == "derive__state"
    )
    incoming = [
        edge
        for edge in plan.graph.data_edges
        if edge.target.component_id == transformed.component_id
    ]
    assert len(incoming) == 2
    assert {
        getattr(
            plan.graph.node_by_id(edge.source.component_id).component,
            "flow_origin",
            None,
        )
        for edge in incoming
    } == {"choice.left__writer", "choice.else.right__writer"}


def test_region_free_state_edges_match_the_compatibility_graph() -> None:
    stages = (
        _StateProducerStage("first"),
        _StateProducerStage("last"),
        _StateConsumerStage(),
    )
    structured = _compile_stages(Pipeline(stages), initial=True)
    compatibility = Compiler().compile(
        build_decomposed_component_graph_from_stages(stages),
        CompileContext(initial_state_keys=frozenset({_FLOW_STATE})),
    )

    def signatures(plan):
        return sorted(
            (
                edge.source.component_id,
                edge.source.role,
                edge.source_port,
                edge.target.component_id,
                edge.target.role,
                edge.target_port,
                tuple(item.name for item in edge.transforms),
            )
            for edge in plan.graph.data_edges
        )

    assert signatures(structured) == signatures(compatibility)


def test_region_free_external_state_transforms_match_compatibility_graph() -> None:
    population_key = StateKey[object](
        namespace="tests", name="transform_population", schema_version=1
    )
    genome_key = StateKey[object](
        namespace="tests", name="transform_genomes", schema_version=1
    )
    population_data = DataSpec(
        kind="Population", bindings={"representation": Fixed(value="real")}
    )
    genome_data = DataSpec(
        kind="GenomeBatch", bindings={"representation": Fixed(value="real")}
    )

    class Writer:
        def contract(self) -> ComponentContract:
            return ComponentContract(
                ports={
                    "writer": PortContract(
                        outputs=(
                            PortSpec(
                                name="value",
                                direction=PortDirection.OUTPUT,
                                data=population_data,
                                cardinality="MANY",
                            ),
                        )
                    )
                }
            )

    class PopulationWriter(Stage):
        def __init__(self, name: str = "write") -> None:
            super().__init__(name=name)
            self.writer = Writer()

        def contract(self) -> ComponentContract:
            return ComponentContract(
                parts=(PartSpec(name="writer", contract=self.writer.contract()),),
                state=StateContract(writes=(population_key,)),
            )

        def flow_summary(self) -> StageFlowSummary:
            call = ComponentCall(name="writer", role="writer", attribute="writer")
            return StageFlowSummary(
                calls=(call,),
                edges=(
                    FlowEdge(
                        source=CallOutput(call=call, port="value"),
                        target=StateValue(key=population_key),
                    ),
                ),
            )

        def execute(self, state):
            return state

    class StateOnlyPopulation(Stage):
        def __init__(self) -> None:
            super().__init__(name="state_only")

        def contract(self) -> ComponentContract:
            return ComponentContract(state=StateContract(writes=(population_key,)))

        def flow_summary(self) -> StageFlowSummary:
            return StageFlowSummary()

        def execute(self, state):
            return state

    class Noop(Stage):
        def __init__(self) -> None:
            super().__init__(name="noop")

        def contract(self) -> ComponentContract:
            return ComponentContract(state=StateContract())

        def flow_summary(self) -> StageFlowSummary:
            return StageFlowSummary()

        def execute(self, state):
            return state

    class UnknownFlow(Stage):
        def __init__(self) -> None:
            super().__init__(name="unknown")

        def flow_summary(self) -> None:
            return None

        def execute(self, state):
            return state

    class ExtractGenomes(Stage):
        def __init__(self) -> None:
            super().__init__(name="extract")

        def contract(self) -> ComponentContract:
            return ComponentContract(
                state=StateContract(reads=(population_key,), writes=(genome_key,))
            )

        def flow_summary(self) -> StageFlowSummary:
            return StageFlowSummary(
                edges=(
                    FlowEdge(
                        source=StateValue(key=population_key),
                        target=StateValue(key=genome_key),
                        transforms=(Transform(name="extract_population_genomes"),),
                    ),
                )
            )

        def execute(self, state):
            return state

    evaluator_contract = ComponentContract(
        ports={
            "evaluator": PortContract(
                inputs=(
                    PortSpec(
                        name="genomes",
                        direction=PortDirection.INPUT,
                        data=genome_data,
                        cardinality="MANY",
                    ),
                )
            )
        }
    )

    class Evaluator:
        def contract(self) -> ComponentContract:
            return evaluator_contract

    class Evaluate(Stage):
        def __init__(self) -> None:
            super().__init__(name="evaluate")
            self.evaluator = Evaluator()

        def contract(self) -> ComponentContract:
            return ComponentContract(
                parts=(PartSpec(name="evaluator", contract=self.evaluator.contract()),),
                state=StateContract(reads=(genome_key,)),
            )

        def flow_summary(self) -> StageFlowSummary:
            call = ComponentCall(
                name="evaluator", role="evaluator", attribute="evaluator"
            )
            return StageFlowSummary(
                calls=(call,),
                edges=(
                    FlowEdge(
                        source=StateValue(key=genome_key),
                        target=CallInput(call=call, port="genomes"),
                    ),
                ),
            )

        def execute(self, state):
            return state

    def compile_pair(stages: Sequence[Stage]):
        stage_list = list(stages)
        context = CompileContext(initial_state_keys=frozenset({population_key}))
        compatibility = Compiler().compile(
            build_decomposed_component_graph_from_stages(stage_list), context
        )
        structured = Compiler().compile(
            decompose_structured_graph(
                lower_pipeline(
                    Pipeline([stage_component(stage) for stage in stage_list])
                )
            ),
            context,
        )
        return compatibility, structured

    def signatures(plan):
        return sorted(
            (
                edge.source.component_id,
                edge.source.role,
                edge.source_port,
                edge.target.component_id,
                edge.target.role,
                edge.target_port,
                tuple(item.name for item in edge.transforms),
            )
            for edge in plan.graph.data_edges
        )

    def errors(plan):
        return sorted(
            item.code for item in plan.diagnostics if item.severity.value == "error"
        )

    comparisons = []
    cases: tuple[tuple[Stage, ...], ...] = (
        (ExtractGenomes(), Evaluate()),
        (PopulationWriter(), ExtractGenomes(), Evaluate()),
        (PopulationWriter(), StateOnlyPopulation(), ExtractGenomes(), Evaluate()),
        (PopulationWriter(), UnknownFlow(), ExtractGenomes(), Evaluate()),
    )
    for stages in cases:
        compatibility, structured = compile_pair(stages)
        comparisons.append(
            (
                signatures(structured),
                signatures(compatibility),
                errors(structured),
                errors(compatibility),
            )
        )

    def branch_signature(writer_first: bool):
        writer = stage_component(PopulationWriter())
        noop = stage_component(Noop())
        branch = Branch(
            _Always(),
            then=Pipeline([writer if writer_first else noop]),
            else_=Pipeline([noop if writer_first else writer]),
            name="choice",
        )
        plan = Compiler().compile(
            decompose_structured_graph(
                lower_pipeline(
                    Pipeline(
                        [
                            branch,
                            stage_component(ExtractGenomes()),
                            stage_component(Evaluate()),
                        ]
                    )
                )
            ),
            CompileContext(initial_state_keys=frozenset({population_key})),
        )
        reader_id = next(
            node.component_id
            for node in plan.graph.nodes
            if isinstance(node.component, Evaluator)
        )
        incoming = [
            edge
            for edge in plan.graph.data_edges
            if edge.target.component_id == reader_id
        ]

        def signature(edge):
            source_node = plan.graph.node_by_id(edge.source.component_id)
            source = source_node.component
            role = source_node.contract.ports.get(edge.source.role or "")
            output = (
                None
                if role is None
                else next(
                    (port for port in role.outputs if port.name == edge.source_port),
                    None,
                )
            )
            data = None if output is None else output.data
            return (
                repr(getattr(source, "state_key", None)),
                getattr(source, "flow_origin", None),
                None if data is None else data.kind,
                tuple(
                    sorted(
                        (key, repr(value))
                        for key, value in (
                            () if data is None else data.bindings.items()
                        )
                    )
                ),
                tuple(item.name for item in edge.transforms),
                edge.target_port,
            )

        return sorted(signature(edge) for edge in incoming), errors(plan), len(incoming)

    then_signature = branch_signature(True)
    else_signature = branch_signature(False)
    assert then_signature == else_signature
    assert then_signature[1:] == ([], 2)
    for (
        structured_edges,
        compatibility_edges,
        structured_errors,
        compatibility_errors,
    ) in comparisons:
        assert structured_edges == compatibility_edges
        assert structured_errors == compatibility_errors == []


def test_many_consecutive_state_branches_compile_quickly() -> None:
    branches = [
        Branch(
            _Always(),
            then=Pipeline([_StateProducerStage(f"left_{index}")]),
            else_=Pipeline([_StateProducerStage(f"right_{index}")]),
            name=f"choice_{index}",
        )
        for index in range(24)
    ]
    started = time.monotonic()
    plan = _compile_stages(Pipeline([*branches, _StateConsumerStage()]), initial=True)
    assert time.monotonic() - started < 2.0
    assert not [item for item in plan.diagnostics if item.severity.value == "error"]


def test_zero_repeat_keeps_inner_edges_and_does_not_export_inputs() -> None:
    class Node:
        def __init__(self, *, reads: bool = False, writes: bool = False) -> None:
            inputs = (
                (
                    PortSpec(
                        name="value",
                        direction=PortDirection.INPUT,
                        data=DataSpec(kind="Population"),
                        cardinality="MANY",
                    ),
                )
                if reads
                else ()
            )
            outputs = (
                (
                    PortSpec(
                        name="value",
                        direction=PortDirection.OUTPUT,
                        data=DataSpec(kind="Population"),
                        cardinality="MANY",
                    ),
                )
                if writes
                else ()
            )
            self._contract = ComponentContract(
                ports={"node": PortContract(inputs=inputs, outputs=outputs)}
            )

        def contract(self) -> ComponentContract:
            return self._contract

        def execute(self, state):
            return state

    internal = Compiler().compile(
        lower_pipeline(
            Pipeline(
                [
                    Repeat(
                        Pipeline([Node(writes=True), Node(reads=True)]),
                        count=0,
                        name="skip",
                    )
                ]
            )
        )
    )
    assert [
        (edge.source.component_id, edge.target.component_id)
        for edge in internal.graph.data_edges
    ] == [("skip.node0", "skip.node1")]

    unresolved = Compiler().compile(
        lower_pipeline(
            Pipeline([Repeat(Pipeline([Node(reads=True)]), count=0, name="skip")])
        )
    )
    assert any(item.code == "unresolved_input" for item in unresolved.diagnostics)
    assert not unresolved.graph.data_edges


def test_unknown_flow_keeps_held_contract_diagnostics_on_both_graph_paths() -> None:
    class RequiredAlgorithm:
        def contract(self) -> ComponentContract:
            return ComponentContract(
                ports={
                    "algorithm": PortContract(
                        inputs=(
                            PortSpec(
                                name="required",
                                direction=PortDirection.INPUT,
                                data=DataSpec(kind="AlgorithmInput"),
                                cardinality="ONE",
                                required_services=(
                                    ServiceRequirement(name="BoundsService"),
                                ),
                            ),
                        ),
                    ),
                },
                execution=ExecutionContract(
                    required_runtime_capabilities=("gpu_runtime",)
                ),
                state=StateContract(reads_enumerable=False, writes_enumerable=False),
            )

    class Required(Stage):
        def __init__(self) -> None:
            super().__init__(name="required")
            self._algorithm = RequiredAlgorithm()

            class Unlisted:
                def contract(self) -> ComponentContract:
                    return ComponentContract(
                        execution=ExecutionContract(
                            required_runtime_capabilities=("unlisted_runtime",)
                        )
                    )

            self.unlisted = Unlisted()

        def contract(self) -> ComponentContract:
            return ComponentContract(
                parts=(
                    PartSpec(name="_algorithm", contract=self._algorithm.contract()),
                ),
                state=StateContract(),
            )

        def execute(self, state):
            return state

    context = CompileContext(offered_runtime_capabilities=frozenset())
    structured = Compiler().compile(
        decompose_structured_graph(
            lower_pipeline(Pipeline([stage_component(Required())]))
        ),
        context,
    )
    compatibility = Compiler().compile(
        build_decomposed_component_graph_from_stages([Required()]), context
    )
    findings = [
        sorted(
            (item.severity.value, item.code, item.path.components)
            for item in plan.diagnostics
        )
        for plan in (structured, compatibility)
    ]
    assert findings[0] == findings[1]
    for plan in (structured, compatibility):
        assert any(
            item.code == "missing_runtime_capability" for item in plan.diagnostics
        )
        assert [item.code for item in plan.diagnostics].count(
            "cannot_verify_stage_flow"
        ) == 1
        assert any(item.code == "unresolved_service" for item in plan.diagnostics)
        assert any(item.code == "opaque_state_effects" for item in plan.diagnostics)
        assert not any(
            item.code == "missing_runtime_capability"
            and item.path.components[-1] == "required__unlisted"
            for item in plan.diagnostics
        )
        assert not plan.graph.data_edges


def test_unknown_flow_held_state_contract_uses_the_host_stage_position() -> None:
    key = StateKey[object](namespace="tests", name="held_state", schema_version=1)

    class Held:
        def contract(self) -> ComponentContract:
            return ComponentContract(state=StateContract(reads=(key,), writes=(key,)))

    class Unknown(Stage):
        def __init__(self, *, declared: bool) -> None:
            super().__init__(name="unknown")
            self.held = Held()
            self.declared = declared

        def contract(self) -> ComponentContract:
            if not self.declared:
                return super().contract()
            return ComponentContract(
                parts=(PartSpec(name="held", contract=self.held.contract()),),
                state=StateContract(),
            )

        def execute(self, state):
            return state

    from saealib.core.graph_builder import build_component_graph

    for declared in (True, False):
        stage = Unknown(declared=declared)
        graphs = (
            decompose_structured_graph(
                lower_pipeline(Pipeline([stage_component(stage)]))
            ),
            build_decomposed_component_graph_from_stages([stage]),
            build_component_graph(Pipeline([stage])),
        )
        plans = [Compiler().compile(graph) for graph in graphs]
        errors = [
            sorted(
                item.code for item in plan.diagnostics if item.severity.value == "error"
            )
            for plan in plans
        ]
        assert errors[0] == errors[1] == errors[2]
        if declared:
            assert errors[0] == ["uninitialized_state_write"]
        else:
            assert errors[0] == []


def test_state_writes_replace_producers_on_both_graph_paths() -> None:
    class UnknownWrite(Stage):
        def __init__(self) -> None:
            super().__init__(name="unknown_write")

        def contract(self) -> ComponentContract:
            return ComponentContract(state=StateContract(writes=(_FLOW_STATE,)))

        def execute(self, state):
            return state

    class StateOnlyWrite(UnknownWrite):
        def __init__(self) -> None:
            super().__init__()
            self.name = "state_only_write"

        def flow_summary(self) -> StageFlowSummary:
            return StageFlowSummary()

    for middle in (UnknownWrite, StateOnlyWrite):
        signatures = []
        diagnostic_sets = []
        for structured in (True, False):
            writer = _StateProducerStage("writer")
            stages = [writer, middle(), _StateConsumerStage()]
            if structured:
                plan = _compile_stages(Pipeline(stages), initial=True)
            else:
                graph = build_decomposed_component_graph_from_stages(stages)
                plan = Compiler().compile(
                    graph,
                    CompileContext(initial_state_keys=frozenset({_FLOW_STATE})),
                )
            reader_id = next(
                node.component_id
                for node in plan.graph.nodes
                if node.component_id.startswith("reader__reader")
            )
            incoming = [
                edge
                for edge in plan.graph.data_edges
                if edge.target.component_id == reader_id
            ]
            assert len(incoming) == 1
            source = plan.graph.node_by_id(incoming[0].source.component_id).component
            assert getattr(source, "state_key", None) == _FLOW_STATE
            assert source is not writer.part
            signatures.append(
                (
                    incoming[0].source.role,
                    incoming[0].source_port,
                    incoming[0].target_port,
                    tuple(item.name for item in incoming[0].transforms),
                )
            )
            diagnostic_sets.append(
                sorted((item.severity.value, item.code) for item in plan.diagnostics)
            )
        assert signatures[0] == signatures[1]
        assert diagnostic_sets[0] == diagnostic_sets[1]


def test_declared_part_omitted_from_summary_is_rejected() -> None:
    class StageWithParts(Stage):
        def __init__(self) -> None:
            super().__init__(name="parts")
            self.first = _Component(
                "producer", input_kind=None, output_kind="Population"
            )
            self.second = _Component(
                "producer", input_kind=None, output_kind="Population"
            )

        def contract(self) -> ComponentContract:
            return ComponentContract(
                parts=(
                    PartSpec(name="first", contract=self.first.contract()),
                    PartSpec(name="second", contract=self.second.contract()),
                )
            )

        def flow_summary(self) -> StageFlowSummary:
            return StageFlowSummary(
                calls=(ComponentCall(name="first", role="producer", attribute="first"),)
            )

        def execute(self, state):
            return state

    with pytest.raises(ValidationError, match="omits declared components"):
        build_decomposed_component_graph_from_stages([StageWithParts()])


def test_verification_call_and_state_nodes_are_not_sequentially_executable() -> None:
    class Helper:
        def contract(self) -> ComponentContract:
            return ComponentContract(
                ports={
                    "main": PortContract(
                        inputs=(
                            PortSpec(
                                name="in",
                                direction=PortDirection.INPUT,
                                data=DataSpec(kind="Population"),
                                cardinality="MANY",
                            ),
                        ),
                        outputs=(
                            PortSpec(
                                name="out",
                                direction=PortDirection.OUTPUT,
                                data=DataSpec(kind="Population"),
                                cardinality="MANY",
                            ),
                        ),
                    )
                }
            )

        def execute(self, state):
            raise AssertionError("verification-only node executed")

    class StageWithCall(Stage):
        def __init__(self) -> None:
            super().__init__(name="owner")
            self.helper = Helper()

        def contract(self) -> ComponentContract:
            return ComponentContract(
                state=StateContract(reads=(_FLOW_STATE,), writes=(_FLOW_STATE,))
            )

        def flow_summary(self) -> StageFlowSummary:
            helper = ComponentCall(name="helper", role="main", attribute="helper")
            return StageFlowSummary(
                calls=(helper,),
                edges=(
                    FlowEdge(
                        source=StateValue(key=_FLOW_STATE),
                        target=CallInput(call=helper, port="in"),
                    ),
                    FlowEdge(
                        source=CallOutput(call=helper, port="out"),
                        target=StateValue(key=_FLOW_STATE),
                    ),
                ),
            )

        def execute(self, state):
            return state

    graph = build_decomposed_component_graph_from_stages([StageWithCall()])
    plan = Compiler().compile(graph)
    from saealib.core.runtime import SequentialPlan

    sequential = SequentialPlan.from_executable_plan(plan)
    assert [node.component_id for node in sequential.execution_nodes] == ["owner"]
    assert sequential.execution_nodes[0] is plan.graph.node_by_id("owner")
    assert (
        graph.node_by_id("owner__helper").component
        is cast(Any, graph.node_by_id("owner").component).stage.helper
    )
    state_node = next(node for node in graph.nodes if node.role == "state")
    assert getattr(state_node.component, "state_key") == _FLOW_STATE
    assert state_node not in sequential.execution_nodes


def test_framework_component_with_a_role_remains_sequentially_executable() -> None:
    from saealib.core.compiler import ComponentGraph, ComponentNode, NodeRef
    from saealib.core.runtime import SequentialPlan

    class Worker:
        def __init__(self) -> None:
            self.calls = 0

        def contract(self) -> ComponentContract:
            return ComponentContract(state=StateContract())

        def execute(self, state):
            self.calls += 1
            return state

    worker = Worker()
    graph = ComponentGraph(
        nodes=(ComponentNode(component_id="work", component=worker, role="worker"),),
        entry_points=(NodeRef(component_id="work", role="worker"),),
    )
    plan = Compiler().compile(graph)
    sequential = SequentialPlan.from_executable_plan(plan)
    assert [node.component_id for node in sequential.execution_nodes] == ["work"]
    assert sequential.execution_nodes[0] is plan.graph.node_by_id("work")
    cast(Any, sequential.execution_nodes[0].component).execute(None)
    assert worker.calls == 1


def test_extract_population_genomes_preserves_representation_binding() -> None:
    from saealib.core.compiler import ComponentGraph, ComponentNode, NodeRef

    @dataclass
    class Endpoint:
        kind: str
        output: bool

        def contract(self) -> ComponentContract:
            port = PortSpec(
                name="value",
                direction=(
                    PortDirection.OUTPUT if self.output else PortDirection.INPUT
                ),
                data=DataSpec(
                    kind=self.kind,
                    bindings={"representation": Fixed(value="real")},
                ),
                cardinality="MANY",
            )
            return ComponentContract(
                ports={
                    "role": (
                        PortContract(outputs=(port,))
                        if self.output
                        else PortContract(inputs=(port,))
                    )
                }
            )

    graph = ComponentGraph(
        nodes=(
            ComponentNode(
                component_id="population", component=Endpoint("Population", True)
            ),
            ComponentNode(
                component_id="evaluator", component=Endpoint("GenomeBatch", False)
            ),
        ),
        data_edges=(
            DataEdge(
                source=NodeRef(component_id="population", role="role"),
                target=NodeRef(component_id="evaluator", role="role"),
                source_port="value",
                target_port="value",
                transforms=(Transform(name="extract_population_genomes"),),
            ),
        ),
        entry_points=(NodeRef(component_id="population", role="role"),),
    )
    plan = Compiler().compile(graph)
    assert not [
        item
        for item in plan.diagnostics
        if item.code in {"producer_binding_missing", "incompatible_port"}
    ]


def test_structured_repeated_stage_names_get_stable_ids() -> None:
    from saealib.stages import CountGenerationStage

    def build(pipeline):
        return decompose_structured_graph(lower_pipeline(pipeline))

    flat = build(
        Pipeline(
            [
                stage_component(CountGenerationStage()),
                stage_component(CountGenerationStage()),
            ]
        )
    )
    nested = build(
        Pipeline([Repeat(Pipeline([stage_component(CountGenerationStage())]), count=2)])
    )
    assert [node.component_id for node in flat.nodes if node.role is None] == [
        "count_generation",
        "count_generation_2",
    ]
    assert [node.component_id for node in nested.nodes if node.role is None] == [
        "repeat.count_generation",
    ]
    combined = Pipeline(
        [
            stage_component(CountGenerationStage()),
            Repeat(
                Pipeline(
                    [
                        stage_component(CountGenerationStage()),
                        stage_component(CountGenerationStage()),
                    ]
                ),
                count=2,
                name="repeat",
            ),
            stage_component(CountGenerationStage()),
        ]
    )
    combined_first = build(combined)
    combined_second = build(combined)
    combined_ids = [
        node.component_id for node in combined_first.nodes if node.role is None
    ]
    assert combined_ids == [
        node.component_id for node in combined_second.nodes if node.role is None
    ]
    assert len(combined_ids) == len(set(combined_ids))
    assert {
        "count_generation",
        "repeat.count_generation",
        "repeat.count_generation_2",
        "count_generation_2",
    } <= set(combined_ids)


def test_call_role_is_validated_without_an_edge() -> None:
    class StageWithUndeclaredRole(Stage):
        def __init__(self) -> None:
            super().__init__(name="bad_role")
            self.part = _Component(
                "producer", input_kind=None, output_kind="Population"
            )

        def flow_summary(self) -> StageFlowSummary:
            return StageFlowSummary(
                calls=(ComponentCall(name="call", role="evaluator", attribute="part"),)
            )

        def execute(self, state):
            return state

    with pytest.raises(ValidationError, match="unknown role"):
        build_decomposed_component_graph_from_stages([StageWithUndeclaredRole()])


def test_surrogate_only_loop_summary_covers_inner_component_roles() -> None:
    algorithm = GA(
        crossover=CrossoverBLXAlpha(prob=0.9, alpha=0.4),
        mutation=MutationUniform(prob_var=0.2),
        parent_selection=SequentialSelection(),
        survivor_selection=TruncationSelection(),
    )
    manager = LocalSurrogateManager(RBFSurrogate(GaussianKernel()))
    stage = SurrogateOnlyLoopStage(
        algorithm, manager, 2, None, acquisition=MeanPrediction()
    )
    summary = stage.flow_summary()
    assert {(call.attribute, call.role) for call in summary.calls} >= {
        ("_sm", "fitter"),
        ("_sm", "predictor"),
        ("_algorithm", "proposer"),
        ("_algorithm", "feedback_consumer"),
        ("_acquisition", "acquisition"),
        ("_builder", "feedback_builder"),
    }
    graph = build_decomposed_component_graph_from_stages([stage])
    calls = {node.component_id.rsplit("__", 1)[-1]: node for node in graph.nodes}
    assert calls["fitter"].component is stage._sm
    assert calls["predictor"].component is stage._sm
    assert calls["algorithm"].component is stage._algorithm
    assert calls["feedback_consumer"].component is stage._algorithm
    assert calls["acquisition"].component is stage._acquisition
    assert calls["builder"].component is cast(Any, stage.stages[4])._builder

    class RequiredAlgorithm(GA):
        def contract(self) -> ComponentContract:
            contract = super().contract()
            return replace(
                contract,
                execution=replace(
                    contract.execution,
                    required_runtime_capabilities=("gpu_runtime",),
                ),
            )

    required = RequiredAlgorithm(
        crossover=CrossoverBLXAlpha(prob=0.9, alpha=0.4),
        mutation=MutationUniform(prob_var=0.2),
        parent_selection=SequentialSelection(),
        survivor_selection=TruncationSelection(),
    )
    required_stage = SurrogateOnlyLoopStage(
        required, manager, 1, None, acquisition=MeanPrediction()
    )
    required_plan = Compiler().compile(
        build_decomposed_component_graph_from_stages([required_stage]),
        CompileContext(offered_runtime_capabilities=frozenset()),
    )
    assert any(
        item.code == "missing_runtime_capability" for item in required_plan.diagnostics
    )


def test_planned_evaluation_observations_reach_apply_through_state_transforms() -> None:
    from saealib.core.state import EVALUATED_OFFSPRING, EVALUATION_UPDATES

    class Evaluator:
        def __init__(self, output_kind: str = "ObservationBatch") -> None:
            self.output_kind = output_kind

        def contract(self) -> ComponentContract:
            return ComponentContract(
                ports={
                    "evaluator": PortContract(
                        outputs=(
                            PortSpec(
                                name="observations",
                                direction=PortDirection.OUTPUT,
                                data=DataSpec(kind=self.output_kind),
                                cardinality="MANY",
                            ),
                        )
                    )
                }
            )

    graph = build_decomposed_component_graph_from_stages(
        [
            EvaluationCollectStage(cast(Any, Evaluator())),
            EvaluationApplyStage(),
            FeedbackStage(),
        ]
    )
    plan = Compiler().compile(graph)

    def flow_node(key):
        return next(
            node
            for node in plan.graph.nodes
            if node.role == "state"
            and getattr(node.component, "state_key", None) == key
            and node.component_id.endswith("__flow")
        )

    updates = flow_node(EVALUATION_UPDATES)
    evaluated = flow_node(EVALUATED_OFFSPRING)
    collect_id = next(
        node.component_id
        for node in plan.graph.nodes
        if node.component_id.startswith("evaluation_collect__evaluator")
    )
    feedback_id = next(
        node.component_id
        for node in plan.graph.nodes
        if node.component_id.startswith("feedback__builder")
    )
    first = next(
        edge
        for edge in plan.graph.data_edges
        if edge.source.component_id == collect_id
        and edge.target.component_id == updates.component_id
    )
    second = next(
        edge
        for edge in plan.graph.data_edges
        if edge.source.component_id == updates.component_id
        and edge.target.component_id == evaluated.component_id
    )
    third = next(
        edge
        for edge in plan.graph.data_edges
        if edge.source.component_id == evaluated.component_id
        and edge.target.component_id == feedback_id
        and edge.target_port == "evaluation"
    )
    assert first.source_port == "observations"
    assert tuple(item.name for item in second.transforms) == ("apply_observations",)
    assert tuple(item.name for item in third.transforms) == (
        "extract_evaluation_observations",
    )
    assert not [
        item
        for item in plan.diagnostics
        if item.code in {"invalid_flow_transform", "incompatible_port"}
    ]

    invalid = Compiler().compile(
        build_decomposed_component_graph_from_stages(
            [
                EvaluationCollectStage(cast(Any, Evaluator("GenomeBatch"))),
                EvaluationApplyStage(),
                FeedbackStage(),
            ]
        )
    )
    assert any(
        item.code == "invalid_flow_transform" and item.severity.value == "error"
        for item in invalid.diagnostics
    )


def test_state_source_transform_preserves_required_representation_on_both_paths() -> (
    None
):
    key = StateKey[object](
        namespace="tests", name="initial_population", schema_version=1
    )

    class Evaluator:
        def contract(self) -> ComponentContract:
            return ComponentContract(
                ports={
                    "evaluator": PortContract(
                        inputs=(
                            PortSpec(
                                name="genomes",
                                direction=PortDirection.INPUT,
                                data=DataSpec(
                                    kind="GenomeBatch",
                                    bindings={"representation": Fixed(value="real")},
                                ),
                                cardinality="MANY",
                            ),
                        )
                    )
                }
            )

    class Evaluate(Stage):
        def __init__(self) -> None:
            super().__init__(name="evaluate")
            self.evaluator = Evaluator()

        def contract(self) -> ComponentContract:
            return ComponentContract(state=StateContract(reads=(key,)))

        def flow_summary(self) -> StageFlowSummary:
            call = ComponentCall(
                name="evaluator", role="evaluator", attribute="evaluator"
            )
            return StageFlowSummary(
                calls=(call,),
                edges=(
                    FlowEdge(
                        source=StateValue(key=key),
                        target=CallInput(call=call, port="genomes"),
                        transforms=(Transform(name="extract_population_genomes"),),
                    ),
                ),
            )

        def execute(self, state):
            return state

    stage = Evaluate()
    context = CompileContext(initial_state_keys=frozenset({key}))
    graphs = (
        decompose_structured_graph(lower_pipeline(Pipeline([stage_component(stage)]))),
        build_decomposed_component_graph_from_stages([stage]),
    )
    for graph in graphs:
        plan = Compiler().compile(graph, context)
        assert not [item for item in plan.diagnostics if item.severity.value == "error"]
        source = next(
            plan.graph.node_by_id(edge.source.component_id)
            for edge in plan.graph.data_edges
            if edge.target.component_id == "evaluate__evaluator"
        )
        assert source.contract.ports["state"].outputs[0].data.bindings == {
            "representation": Fixed(value="real")
        }


def test_encode_features_does_not_carry_the_input_representation_binding() -> None:
    from saealib.core.compiler import ComponentGraph, ComponentNode, DataEdge, NodeRef

    @dataclass
    class Endpoint:
        kind: str
        output: bool
        representation: str

        def contract(self) -> ComponentContract:
            port = PortSpec(
                name="value",
                direction=(
                    PortDirection.OUTPUT if self.output else PortDirection.INPUT
                ),
                data=DataSpec(
                    kind=self.kind,
                    bindings={"representation": Fixed(value=self.representation)},
                ),
                cardinality="MANY",
            )
            return ComponentContract(
                ports={
                    "role": PortContract(
                        outputs=(port,) if self.output else (),
                        inputs=() if self.output else (port,),
                    )
                }
            )

    graph = ComponentGraph(
        nodes=(
            ComponentNode(
                component_id="population",
                component=Endpoint("Population", True, "categorical"),
            ),
            ComponentNode(
                component_id="model",
                component=Endpoint("FeatureBatch", False, "real"),
            ),
        ),
        data_edges=(
            DataEdge(
                source=NodeRef(component_id="population", role="role"),
                target=NodeRef(component_id="model", role="role"),
                source_port="value",
                target_port="value",
                transforms=(Transform(name="encode_features"),),
            ),
        ),
        entry_points=(NodeRef(component_id="population", role="role"),),
    )
    plan = Compiler().compile(graph)
    schema_errors = [
        item
        for item in plan.diagnostics
        if item.code in {"incompatible_port", "invalid_flow_transform"}
    ]
    assert len(schema_errors) == 1
    assert "producer_binding_missing" in schema_errors[0].message


def test_many_consecutive_loops_compile_without_path_explosion() -> None:
    pipeline = Pipeline(
        [
            Loop(
                Pipeline(
                    [
                        _StateConsumerStage(f"reader_{index}"),
                        _StateProducerStage(f"writer_{index}"),
                    ]
                ),
                until=_Always(),
                name=f"loop_{index}",
            )
            for index in range(16)
        ]
    )
    started = time.monotonic()
    plan = _compile_stages(pipeline, initial=True)
    assert time.monotonic() - started < 2.0
    assert not [item for item in plan.diagnostics if item.severity.value == "error"]


def test_repeat_and_loop_do_not_export_branch_component_producers() -> None:
    class PortNode:
        def __init__(self, *, reads: bool = False, writes: bool = False) -> None:
            self.reads = reads
            self.writes = writes

        def contract(self) -> ComponentContract:
            input_ports = (
                (
                    PortSpec(
                        name="value",
                        direction=PortDirection.INPUT,
                        data=DataSpec(kind="Population"),
                        cardinality="MANY",
                    ),
                )
                if self.reads
                else ()
            )
            output_ports = (
                (
                    PortSpec(
                        name="value",
                        direction=PortDirection.OUTPUT,
                        data=DataSpec(kind="Population"),
                        cardinality="MANY",
                    ),
                )
                if self.writes
                else ()
            )
            return ComponentContract(
                ports={
                    "node": PortContract(
                        inputs=input_ports,
                        outputs=output_ports,
                    )
                },
                state=StateContract(),
            )

        def execute(self, state):
            return state

    def branch() -> Branch:
        return Branch(
            _Always(),
            then=Pipeline([PortNode(writes=True)]),
            else_=Pipeline([PortNode(writes=True)]),
        )

    cases = (
        Pipeline(
            [
                branch(),
                Loop(Pipeline([PortNode(reads=True)]), until=_Always()),
            ]
        ),
        Pipeline(
            [
                branch(),
                Repeat(Pipeline([PortNode()]), count=1),
                PortNode(reads=True),
            ]
        ),
        Pipeline([Repeat(Pipeline([PortNode(reads=True)]), count=0)]),
    )
    for pipeline in cases:
        plan = Compiler().compile(lower_pipeline(pipeline))
        assert any(item.code == "unresolved_input" for item in plan.diagnostics)
        assert not plan.graph.data_edges


def test_loop_back_edges_use_only_final_and_executable_writers() -> None:
    reader = _StateConsumerStage()
    first = _StateProducerStage("first_writer")
    final = _StateProducerStage("final_writer")
    overwritten = _compile_stages(
        Pipeline(
            [
                Loop(
                    Pipeline([reader, first, final]),
                    until=_Always(),
                    name="cycle",
                )
            ]
        ),
        initial=True,
    )
    calls = overwritten.graph.metadata["stage_flow_calls"]
    reader_id = next(names["reader"] for names in calls.values() if "reader" in names)
    incoming = [
        edge
        for edge in overwritten.graph.data_edges
        if edge.target.component_id == reader_id
    ]
    sources = [
        overwritten.graph.node_by_id(edge.source.component_id).component
        for edge in incoming
    ]
    origins = {getattr(source, "flow_origin", None) for source in sources}
    assert "cycle.final_writer__writer" in origins
    assert "cycle.first_writer__writer" not in origins
    assert any(getattr(source, "state_key", None) == _FLOW_STATE for source in sources)

    left = _StateProducerStage("left_writer")
    right = _StateProducerStage("right_writer")
    branch_plan = _compile_stages(
        Pipeline(
            [
                Loop(
                    Pipeline(
                        [
                            Branch(
                                _Always(),
                                then=Pipeline([left]),
                                else_=Pipeline([right]),
                            ),
                            _StateConsumerStage(),
                        ]
                    ),
                    until=_Always(),
                    name="choice_cycle",
                )
            ]
        ),
        initial=True,
    )
    calls = branch_plan.graph.metadata["stage_flow_calls"]
    reader_id = next(names["reader"] for names in calls.values() if "reader" in names)
    branch_sources = {
        getattr(
            branch_plan.graph.node_by_id(edge.source.component_id).component,
            "flow_origin",
            None,
        )
        for edge in branch_plan.graph.data_edges
        if edge.target.component_id == reader_id
    }
    assert any(
        isinstance(source, str) and "left_writer__writer" in source
        for source in branch_sources
    )
    assert any(
        isinstance(source, str) and "right_writer__writer" in source
        for source in branch_sources
    )

    before_branch = _StateProducerStage("before_branch_writer")
    one_sided = _StateProducerStage("one_sided_writer")
    one_sided_plan = _compile_stages(
        Pipeline(
            [
                Loop(
                    Pipeline(
                        [
                            before_branch,
                            Branch(
                                _Always(),
                                then=Pipeline([one_sided]),
                                else_=None,
                            ),
                            _StateConsumerStage(),
                        ]
                    ),
                    until=_Always(),
                    name="one_sided_cycle",
                )
            ]
        ),
        initial=True,
    )
    calls = one_sided_plan.graph.metadata["stage_flow_calls"]
    reader_id = next(names["reader"] for names in calls.values() if "reader" in names)
    one_sided_sources = [
        one_sided_plan.graph.node_by_id(edge.source.component_id).component
        for edge in one_sided_plan.graph.data_edges
        if edge.target.component_id == reader_id
    ]
    one_sided_origins = {
        getattr(source, "flow_origin", None) for source in one_sided_sources
    }
    assert any(
        "before_branch_writer__writer" in origin
        for origin in one_sided_origins
        if isinstance(origin, str)
    )
    assert any(
        "one_sided_writer__writer" in origin
        for origin in one_sided_origins
        if isinstance(origin, str)
    )

    skipped = _StateProducerStage("skipped_writer")
    repeat_zero = _compile_stages(
        Pipeline(
            [
                Loop(
                    Pipeline(
                        [
                            Repeat(Pipeline([skipped]), count=0),
                            _StateConsumerStage(),
                        ]
                    ),
                    until=_Always(),
                    name="skip_cycle",
                )
            ]
        ),
        initial=True,
    )
    calls = repeat_zero.graph.metadata["stage_flow_calls"]
    reader_id = next(names["reader"] for names in calls.values() if "reader" in names)
    skipped_incoming = [
        edge
        for edge in repeat_zero.graph.data_edges
        if edge.target.component_id == reader_id
    ]
    assert len(skipped_incoming) == 1
    assert (
        repeat_zero.graph.node_by_id(
            skipped_incoming[0].source.component_id
        ).component.state_key
        == _FLOW_STATE
    )


def test_registered_transforms_validate_output_and_row_alignment() -> None:
    planner = _Component(
        "evaluation_planner", input_kind=None, output_kind="EvaluationRequestBatch"
    )
    evaluator = _Component("evaluator", input_kind="GenomeBatch", output_kind=None)

    class EvaluationStage(Stage):
        def __init__(self, transforms=None) -> None:
            super().__init__(name="evaluation")
            self.planner = planner
            self.evaluator = evaluator
            self.transforms = transforms or (
                Transform(name="select_requested_candidates"),
                Transform(name="extract_genomes"),
            )

        def flow_summary(self) -> StageFlowSummary:
            plan = ComponentCall(
                name="planner", role="evaluation_planner", attribute="planner"
            )
            evaluate = ComponentCall(
                name="evaluator", role="evaluator", attribute="evaluator"
            )
            return StageFlowSummary(
                calls=(plan, evaluate),
                edges=(
                    FlowEdge(
                        source=CallOutput(call=plan, port="out"),
                        target=CallInput(call=evaluate, port="in"),
                        transforms=self.transforms,
                    ),
                ),
            )

        def execute(self, state):
            return state

    graph = build_decomposed_component_graph_from_stages([EvaluationStage()])
    plan = Compiler().compile(graph)
    edge = next(edge for edge in plan.graph.data_edges if edge.transforms)
    assert [item.name for item in edge.transforms] == [
        "select_requested_candidates",
        "extract_genomes",
    ]
    assert not [
        item
        for item in plan.diagnostics
        if item.code in {"invalid_flow_transform", "incompatible_port"}
    ]

    for invalid_chain in (
        (Transform(name="invented_conversion"),),
        (Transform(name="extract_genomes"),),
    ):
        invalid_plan = Compiler().compile(
            build_decomposed_component_graph_from_stages(
                [EvaluationStage(transforms=invalid_chain)]
            )
        )
        assert any(
            item.code == "invalid_flow_transform" and item.severity.value == "error"
            for item in invalid_plan.diagnostics
        )

    class _Ids:
        def __init__(self) -> None:
            self.next = 1

        def allocate(self, count: int):
            import numpy as np

            values = np.arange(self.next, self.next + count, dtype=np.int64)
            self.next += count
            return values

    class _Population:
        schema: ClassVar[dict[str, object]] = {"id": object()}

        def __init__(self):
            import numpy as np

            from saealib.population.genome import DenseVectorBatch

            self.ids = np.array([31, 17, 24], dtype=np.int64)
            self.genomes = DenseVectorBatch(np.array([[3.0], [1.0], [2.0]]))

        def __len__(self):
            return len(self.ids)

        def get_array(self, name: str):
            assert name == "id"
            return self.ids

    from saealib.acquisition import AcquisitionResult
    from saealib.core.state import EVALUATION_REQUEST
    from saealib.policies.evaluation import TopKEvaluation
    from saealib.stages import EvaluationPlanStage, EvaluationSubmitStage

    class GenomeEvaluator:
        def contract(self) -> ComponentContract:
            return ComponentContract(
                ports={
                    "evaluator": PortContract(
                        inputs=(
                            PortSpec(
                                name="genomes",
                                direction=PortDirection.INPUT,
                                data=DataSpec(kind="GenomeBatch"),
                                cardinality="MANY",
                            ),
                        )
                    )
                }
            )

    evaluation_graph = build_decomposed_component_graph_from_stages(
        [
            EvaluationPlanStage(TopKEvaluation(2)),
            EvaluationSubmitStage(cast(Any, GenomeEvaluator())),
        ]
    )
    evaluation_plan = Compiler().compile(evaluation_graph)
    planner_edge = next(
        edge
        for edge in evaluation_plan.graph.data_edges
        if edge.target.component_id.startswith("evaluation_submit__evaluator")
        and edge.transforms
    )
    assert planner_edge.source.role == "state"
    assert (
        getattr(
            evaluation_plan.graph.node_by_id(
                planner_edge.source.component_id
            ).component,
            "state_key",
            None,
        )
        == EVALUATION_REQUEST
    )
    assert tuple(item.name for item in planner_edge.transforms) == (
        "select_requested_candidates",
        "extract_genomes",
    )

    class _Context:
        candidate_id_allocator = _Ids()
        request_id_allocator = _Ids()

    population = _Population()
    request = (
        TopKEvaluation(2)
        .plan(
            population,
            AcquisitionResult(scores=__import__("numpy").array([0.0, 2.0, 1.0])),
            _Context(),
        )
        .requests[0]
    )
    rows = request.metadata["row_indices"]
    assert tuple(item.name for item in planner_edge.transforms) == (
        "select_requested_candidates",
        "extract_genomes",
    )
    assert request.candidate_ids.tolist() == [
        int(population.ids[index]) for index in rows
    ]
    assert request.payload.array.tolist() == [
        population.genomes.array[index].tolist() for index in rows
    ]


def test_adapter_insertion_uses_and_preserves_transformed_data() -> None:
    planner = _Component(
        "planner", input_kind=None, output_kind="EvaluationRequestBatch"
    )
    consumer = _Component("consumer", input_kind="Population", output_kind=None)

    class AdapterStage(Stage):
        def __init__(self) -> None:
            super().__init__(name="adapted")
            self.planner = planner
            self.consumer = consumer

        def flow_summary(self) -> StageFlowSummary:
            source = ComponentCall(name="planner", role="planner", attribute="planner")
            target = ComponentCall(
                name="consumer", role="consumer", attribute="consumer"
            )
            return StageFlowSummary(
                calls=(source, target),
                edges=(
                    FlowEdge(
                        source=CallOutput(call=source, port="out"),
                        target=CallInput(call=target, port="in"),
                        transforms=(Transform(name="select_requested_candidates"),),
                    ),
                ),
            )

        def execute(self, state):
            return state

    adapter = Adapter(
        name="selection_view",
        source=DataSpec(kind="CandidateSelection"),
        target=DataSpec(kind="Population"),
        lossless=True,
        auto_insertable=True,
        category="lossless_view",
    )
    graph = build_decomposed_component_graph_from_stages([AdapterStage()])
    result = LosslessAdapterRule(AdapterRegistry((adapter,))).apply(
        RuleContext(
            graph=graph,
            compile_context=CompileContext(),
            diagnostics=DiagnosticBag(),
        )
    )
    inserted = [
        node
        for node in result.graph.nodes
        if getattr(node.component, "adapter", None) == adapter
    ]
    assert len(inserted) == 1
    adapter_id = inserted[0].component_id
    incoming = next(
        edge
        for edge in result.graph.data_edges
        if edge.target.component_id == adapter_id
    )
    outgoing = next(
        edge
        for edge in result.graph.data_edges
        if edge.source.component_id == adapter_id
    )
    assert incoming.transforms == (Transform(name="select_requested_candidates"),)
    assert outgoing.transforms == ()
