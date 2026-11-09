"""Compiler verification of the components the built-in flow templates hold."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, cast

import numpy as np
import pytest

from saealib import (
    GA,
    CrossoverBLXAlpha,
    LHSInitializer,
    MutationUniform,
    Optimizer,
    SequentialSelection,
    Termination,
    TruncationSelection,
)
from saealib.acquisition import MeanPrediction
from saealib.core.compiler.adapters import AdapterInsertion
from saealib.exceptions import ConfigurationError, ValidationError
from saealib.execution import AsyncEvaluationScheduler, SerialEvaluator
from saealib.pipeline import Pipeline
from saealib.policies.feedback import TrueOnlyFeedback
from saealib.problem import Problem
from saealib.stages import CountGenerationStage
from saealib.strategies import DirectStrategy, GenerationBasedStrategy, TemplateStrategy
from saealib.surrogate.manager import LocalSurrogateManager
from saealib.surrogate.rbf import RBFSurrogate
from saealib.surrogate.rbf_kernels import GaussianKernel


class _GpuGA(GA):
    def contract(self) -> Any:
        contract = super().contract()
        return replace(
            contract,
            execution=replace(
                contract.execution, required_runtime_capabilities=("gpu_runtime",)
            ),
        )


class _MislabelledFeedback(TrueOnlyFeedback):
    def contract(self) -> Any:
        contract = super().contract()
        return replace(
            contract,
            ports={
                role: replace(
                    port,
                    outputs=tuple(
                        replace(output, data=replace(output.data, kind="Unrelated"))
                        for output in port.outputs
                    ),
                )
                for role, port in contract.ports.items()
            },
        )


def _ga(cls: type[GA] = GA) -> GA:
    return cls(
        crossover=CrossoverBLXAlpha(prob=0.9, alpha=0.4),
        mutation=MutationUniform(prob_var=0.2),
        parent_selection=SequentialSelection(),
        survivor_selection=TruncationSelection(),
    )


def _optimizer(strategy: Any, *, asynchronous: bool = False) -> Optimizer:
    optimizer = (
        Optimizer(
            Problem(
                func=lambda x: np.array([float(np.sum(np.asarray(x) ** 2))]),
                dim=2,
                n_obj=1,
                direction=np.array([-1.0]),
                lb=[-5.0] * 2,
                ub=[5.0] * 2,
            ),
            seed=3,
        )
        .set_initializer(LHSInitializer(8, 4, 3))
        .set_algorithm(_ga())
        .set_strategy(strategy)
        .set_termination(Termination(lambda state: state.gen >= 2))
    )
    if strategy.requires_surrogate:
        optimizer.set_surrogate_manager(
            LocalSurrogateManager(RBFSurrogate(GaussianKernel()))
        ).set_acquisition(MeanPrediction())
    if asynchronous:
        optimizer.set_async_evaluation_scheduler(
            AsyncEvaluationScheduler(SerialEvaluator(), max_pending=2)
        )
    return optimizer


def _findings(optimizer: Optimizer) -> list[tuple[str, str, tuple[str, ...]]]:
    plan = optimizer._compile_plan()
    assert plan is not None
    return sorted(
        (
            diagnostic.severity.name,
            diagnostic.code,
            diagnostic.path.components if diagnostic.path else (),
        )
        for diagnostic in plan.diagnostics
    )


@pytest.mark.parametrize(
    ("strategy", "path_prefix"),
    [
        (DirectStrategy(), ""),
        (DirectStrategy(n_offspring=3), ""),
        (GenerationBasedStrategy(0), "generation_based.true_generation."),
    ],
)
def test_algorithm_requiring_an_unoffered_runtime_capability_stops_the_run(
    strategy: Any, path_prefix: str
) -> None:
    optimizer = _optimizer(strategy).set_algorithm(_ga(_GpuGA))

    findings = _findings(optimizer)
    assert len(findings) == 2
    assert {code for severity, code, _path in findings} == {
        "missing_runtime_capability"
    }
    assert all(severity == "ERROR" for severity, _code, _path in findings)
    assert {
        path[-1].removeprefix(path_prefix)
        for _severity, _code, path in findings
        if path
    } == {"ask__proposer", "tell__algorithm"}
    with pytest.raises(ConfigurationError, match="missing_runtime_capability"):
        optimizer.run()


def test_compiler_diagnostic_reports_selected_source_and_reason() -> None:
    optimizer = _optimizer(DirectStrategy()).set_algorithm(_ga(_GpuGA))
    plan = optimizer._compile_plan()

    assert plan is not None
    findings = [
        diagnostic
        for diagnostic in plan.diagnostics
        if diagnostic.code == "missing_runtime_capability"
    ]
    assert findings
    assert any(
        "algorithm selected _GpuGA from setter: Configured with set_algorithm()"
        in resolution
        for diagnostic in findings
        for resolution in diagnostic.resolutions
    )


@pytest.mark.parametrize("asynchronous", [False, True])
def test_direct_feedback_with_unrelated_output_kind_is_reported(
    asynchronous: bool,
) -> None:
    optimizer = _optimizer(DirectStrategy(), asynchronous=asynchronous)
    optimizer.set_feedback_builder(_MislabelledFeedback())

    findings = _findings(optimizer)
    warnings = [
        item for item in findings if item[:2] == ("WARNING", "cannot_verify_data_kind")
    ]
    assert len(warnings) == 2
    assert sum(path[-1] == "feedback__builder" for _, _, path in warnings) == 1

    if asynchronous:
        errors = [
            item
            for item in findings
            if item[:2] == ("ERROR", "incompatible_feedback_lifecycle")
        ]
        assert len(errors) == 1
        assert {path[-1] for _, _, path in errors} == {"tell__algorithm"}
        with pytest.raises(ConfigurationError, match="incompatible_feedback"):
            optimizer.run()
    else:
        assert not [item for item in findings if item[0] == "ERROR"]


def test_direct_async_plan_inserts_the_feedback_accumulator() -> None:
    optimizer = _optimizer(DirectStrategy(), asynchronous=True)
    plan = optimizer._compile_plan()

    assert plan is not None
    assert len(plan.inserted_adapters) == 1
    assert {
        (
            cast(AdapterInsertion, insertion).adapter_name,
            cast(AdapterInsertion, insertion).source_path.role,
            cast(AdapterInsertion, insertion).source_path.port,
            cast(AdapterInsertion, insertion).target_path.role,
            cast(AdapterInsertion, insertion).target_path.port,
            cast(AdapterInsertion, insertion).target_path.components[-1],
        )
        for insertion in plan.inserted_adapters
    } == {
        (
            "feedback_accumulator",
            "state",
            "value",
            "feedback_consumer",
            "feedback",
            "tell__algorithm",
        ),
    }


def test_generation_based_async_plan_delivers_complete_batches() -> None:
    optimizer = _optimizer(GenerationBasedStrategy(2), asynchronous=True)
    plan = optimizer._compile_plan()

    assert plan is not None
    assert not any(
        getattr(insertion, "adapter_name", None) == "feedback_accumulator"
        for insertion in plan.inserted_adapters
    )
    assert not plan.diagnostics


def test_misspelled_execution_boundary_stops_plan_preparation() -> None:
    stage = CountGenerationStage()
    stage.execution_boundary = "State"
    optimizer = _optimizer(TemplateStrategy(Pipeline(steps=[stage], name="t")))

    with pytest.raises(ValidationError, match="execution_boundary"):
        optimizer._compile_plan()


@pytest.mark.parametrize(
    "strategy",
    [DirectStrategy(), GenerationBasedStrategy(1)],
    ids=["direct", "generation-based"],
)
def test_lowered_graph_binds_each_state_key_once_per_node(strategy: Any) -> None:
    optimizer = _optimizer(strategy)
    plan = optimizer._compile_plan()
    assert plan is not None

    for graph in (strategy.build_graph(optimizer), plan.graph):
        bindings = [
            (binding.node.component_id, binding.state_key)
            for binding in graph.state_bindings
        ]
        assert bindings
        assert len(bindings) == len(set(bindings))


def test_decomposed_graph_rejects_values_that_are_not_node_specs() -> None:
    from saealib.core.graph_builder import build_decomposed_component_graph_from_specs

    with pytest.raises(
        ValidationError, match=r"^graph specs must contain NodeAdapterSpec values"
    ):
        build_decomposed_component_graph_from_specs([cast(Any, object())])


def test_decomposed_graph_rejects_duplicate_component_ids() -> None:
    from saealib.core.graph_builder import (
        NodeAdapterSpec,
        StageContractNodeAdapter,
        build_decomposed_component_graph_from_specs,
    )

    specs = [
        NodeAdapterSpec(
            component_id="count",
            adapter=StageContractNodeAdapter(CountGenerationStage(), node_path="count"),
        )
        for _ in range(2)
    ]

    with pytest.raises(
        ValidationError, match=r"^graph specs must have unique component_id values"
    ):
        build_decomposed_component_graph_from_specs(specs)


def test_structured_graph_without_held_components_is_returned_unchanged() -> None:
    from saealib.core.compiler.lowerer import lower_pipeline
    from saealib.core.graph_builder import decompose_structured_graph
    from saealib.stages import stage_component

    graph = lower_pipeline(Pipeline([stage_component(CountGenerationStage())]))

    assert decompose_structured_graph(graph) is graph
