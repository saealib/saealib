from __future__ import annotations

import pickle
import sys
from pathlib import Path
from typing import cast

import numpy as np
import pytest

import saealib
from saealib.algorithms.pso import PSO
from saealib.context import OptimizationState
from saealib.core.compiler import Severity
from saealib.core.contracts.contract import ComponentContract
from saealib.core.contracts.state import StateContract
from saealib.defaults.keys import POPULATION_SIZE
from saealib.defaults.model import (
    DefaultHint,
    DefaultResolution,
    DefaultStrength,
    ResolvedDefault,
)
from saealib.exceptions import ConfigurationError
from saealib.optimizer import Optimizer
from saealib.pipeline import ComponentRef, Pipeline, Stage
from saealib.policies import EvaluateAll, RatioEvaluation
from saealib.policies.feedback import (
    ComparatorWorstFallback,
    FeedbackBuilder,
    MixedFeedback,
    NoFeedback,
    PredictedFeedback,
    TrueOnlyFeedback,
)
from saealib.problem import Problem
from saealib.stages import EvaluationPlanStage
from saealib.strategies.base import TemplateStrategy
from saealib.strategies.direct import DirectStrategy
from saealib.strategies.ib import IndividualBasedStrategy
from saealib.termination import Termination, max_gen


def _problem() -> Problem:
    return Problem(
        func=lambda x: np.array([float(np.sum(np.asarray(x) ** 2))]),
        dim=3,
        n_obj=1,
        direction=np.array([-1.0]),
        lb=[-1.0] * 3,
        ub=[1.0] * 3,
    )


def _preset(ratio: float) -> dict:
    return {
        "evaluation_planner": {
            "type": "RatioEvaluation",
            "params": {"ratio": ratio, "sanitize_nonfinite": True},
        }
    }


def _bundled(monkeypatch, ratio: float | None) -> None:
    from saealib import defaults

    preset = {"strategy": {"type": "DirectStrategy", "params": {}}}
    if ratio is not None:
        preset.update(
            {
                "evaluation_planner": {
                    "type": "RatioEvaluation",
                    "params": {"ratio": ratio, "sanitize_nonfinite": True},
                },
                "feedback_builder": {
                    "type": "TrueOnlyFeedback",
                    "params": {},
                },
            }
        )
    monkeypatch.setattr(
        defaults,
        "load_defaults",
        lambda: {
            "presets": {"test": preset},
            "by_algorithm": {},
            "by_problem_shape": [],
            "fallback": "test",
        },
    )


def _source(optimizer: Optimizer, slot: str) -> str:
    resolution = optimizer._component_resolution
    assert resolution is not None
    origin = resolution.origins[slot]
    return origin.source


def _origins(optimizer: Optimizer):
    resolution = optimizer._component_resolution
    assert resolution is not None
    return resolution.origins


def test_setter_precedes_user_bundled_strategy_and_stage_defaults(monkeypatch) -> None:
    _bundled(monkeypatch, 0.8)
    selected = RatioEvaluation(0.2)
    optimizer = (
        Optimizer(_problem())
        .set_preset(_preset(0.6))
        .set_strategy(DirectStrategy())
        .set_evaluation_planner(selected)
    )

    optimizer.resolve_defaults()

    assert optimizer.evaluation_planner is selected
    assert _source(optimizer, "evaluation_planner") == "setter"


def test_direct_assignment_precedes_user_preset(monkeypatch) -> None:
    _bundled(monkeypatch, 0.8)
    selected = RatioEvaluation(0.2)
    optimizer = Optimizer(_problem()).set_preset(_preset(0.6))
    optimizer.evaluation_planner = selected

    optimizer.resolve_defaults()

    assert optimizer.evaluation_planner is selected
    assert _source(optimizer, "evaluation_planner") == "direct assignment"


def test_user_preset_precedes_bundled_preset(monkeypatch) -> None:
    _bundled(monkeypatch, 0.8)
    optimizer = Optimizer(_problem()).set_algorithm(PSO()).set_preset(_preset(0.6))

    optimizer.resolve_defaults()

    assert isinstance(optimizer.evaluation_planner, RatioEvaluation)
    assert optimizer.evaluation_planner.ratio == 0.6
    assert _source(optimizer, "evaluation_planner") == "user preset"
    assert _origins(optimizer)["evaluation_planner"].reason


def test_bundled_preset_precedes_strategy_default(monkeypatch) -> None:
    _bundled(monkeypatch, 0.8)
    optimizer = Optimizer(_problem())

    optimizer.resolve_defaults()

    assert isinstance(optimizer.strategy, DirectStrategy)
    assert isinstance(optimizer.evaluation_planner, RatioEvaluation)
    assert optimizer.evaluation_planner.ratio == 0.8
    assert _source(optimizer, "evaluation_planner") == "bundled preset"


def test_explicit_strategy_defaults_precede_bundled_policies(monkeypatch) -> None:
    _bundled(monkeypatch, 0.1)
    optimizer = Optimizer(_problem()).set_strategy(IndividualBasedStrategy(0.7))

    optimizer.resolve_defaults()

    assert isinstance(optimizer.evaluation_planner, RatioEvaluation)
    assert optimizer.evaluation_planner.ratio == 0.7
    assert _source(optimizer, "evaluation_planner") == "strategy default"
    assert isinstance(optimizer.feedback_builder, MixedFeedback)
    assert _source(optimizer, "feedback_builder") == "strategy default"


def test_higher_priority_strategies_use_their_own_policy_defaults(monkeypatch) -> None:
    _bundled(monkeypatch, 0.1)
    preset = {
        "strategy": {
            "type": "IndividualBasedStrategy",
            "params": {"evaluation_ratio": 0.6},
        }
    }
    user_strategy = Optimizer(_problem()).set_preset(preset)
    assigned_strategy = Optimizer(_problem())
    assigned_strategy.strategy = IndividualBasedStrategy(0.7)

    user_strategy.resolve_defaults()
    assigned_strategy.resolve_defaults()

    assert _source(user_strategy, "strategy") == "user preset"
    assert isinstance(user_strategy.evaluation_planner, RatioEvaluation)
    assert user_strategy.evaluation_planner.ratio == 0.6
    assert _source(user_strategy, "evaluation_planner") == "strategy default"
    assert _source(assigned_strategy, "strategy") == "direct assignment"
    assert isinstance(assigned_strategy.evaluation_planner, RatioEvaluation)
    assert assigned_strategy.evaluation_planner.ratio == 0.7
    assert _source(assigned_strategy, "evaluation_planner") == "strategy default"


def test_strategy_default_precedes_stage_fallback(monkeypatch) -> None:
    _bundled(monkeypatch, None)
    optimizer = Optimizer(_problem()).set_strategy(IndividualBasedStrategy(0.7))

    optimizer.resolve_defaults()

    assert isinstance(optimizer.evaluation_planner, RatioEvaluation)
    assert optimizer.evaluation_planner.ratio == 0.7
    assert _source(optimizer, "evaluation_planner") == "strategy default"


def test_stage_fallback_source_and_reason_are_compiler_diagnostics(monkeypatch) -> None:
    _bundled(monkeypatch, None)

    class MinimalStrategy(TemplateStrategy):
        requires_surrogate = False

        def __init__(self) -> None:
            super().__init__(
                Pipeline([EvaluationPlanStage(ComponentRef("evaluation_planner"))])
            )

    optimizer = Optimizer(_problem()).set_strategy(MinimalStrategy())
    plan = optimizer._compile_plan()

    assert plan is not None
    messages = [item.message for item in plan.diagnostics]
    assert any("Stage fallback" in message for message in messages)
    assert any(
        "EvaluationPlanStage supplies EvaluateAll" in message for message in messages
    )


def test_incompatible_user_policy_keeps_selection_and_reports_source(
    monkeypatch,
) -> None:
    _bundled(monkeypatch, 0.1)
    optimizer = (
        Optimizer(_problem())
        .set_algorithm(PSO())
        .set_strategy(DirectStrategy())
        .set_preset(_preset(0.3))
    )

    plan = optimizer._compile_plan()

    assert plan is not None
    assert isinstance(optimizer.evaluation_planner, RatioEvaluation)
    assert optimizer.evaluation_planner.ratio == 0.3
    assert _source(optimizer, "evaluation_planner") == "user preset"
    diagnostics = [
        item
        for item in plan.diagnostics
        if item.code == "unresolved_input" and "evaluation_planner" in item.message
    ]
    assert len(diagnostics) == 1
    assert diagnostics[0].severity is Severity.ERROR
    assert "user preset" in diagnostics[0].message
    assert "Selected from the user preset" in diagnostics[0].message
    assert any("acquisition scores" in issue for issue in optimizer.validate())
    assert any(
        item.severity is Severity.ERROR
        and "Selected from the user preset" in item.message
        for item in optimizer.last_contract_diagnostics
    )


def test_incompatible_user_feedback_keeps_selection_and_reports_source(
    monkeypatch,
) -> None:
    _bundled(monkeypatch, 0.1)
    optimizer = (
        Optimizer(_problem())
        .set_algorithm(PSO())
        .set_strategy(DirectStrategy())
        .set_preset({"feedback_builder": {"type": "PredictedFeedback"}})
    )
    optimizer.resolve_defaults()
    selected = optimizer.feedback_builder

    plan = optimizer._compile_plan()

    assert plan is not None
    assert optimizer.feedback_builder is selected
    assert isinstance(selected, PredictedFeedback)
    inputs = selected.contract().ports["feedback_builder"].inputs
    assert not next(port for port in inputs if port.name == "prediction").optional
    assert next(port for port in inputs if port.name == "evaluation").optional
    assert _source(optimizer, "feedback_builder") == "user preset"
    origin = _origins(optimizer)["feedback_builder"]
    diagnostics = [
        item
        for item in plan.diagnostics
        if item.path.components == ("optimizer", "feedback_builder")
        and item.code == "unresolved_input"
    ]
    assert len(diagnostics) == 1
    assert diagnostics[0].severity is Severity.ERROR
    assert "objective predictions" in diagnostics[0].message
    assert "user preset" in diagnostics[0].message
    assert origin.reason in diagnostics[0].message
    issues = optimizer.validate()
    assert any("objective predictions" in issue for issue in issues)
    assert any("user preset" in issue and origin.reason in issue for issue in issues)
    assert any(item is diagnostics[0] for item in optimizer.last_contract_diagnostics)
    with pytest.raises(ConfigurationError, match="objective predictions"):
        optimizer.run()


class _OptionalFeedback(FeedbackBuilder):
    def build(self, candidates, prediction, evaluation, evaluated_indices, ctx):
        return NoFeedback().build(
            candidates, prediction, evaluation, evaluated_indices, ctx
        )


@pytest.mark.parametrize(
    "builder",
    [
        TrueOnlyFeedback(),
        MixedFeedback(),
        NoFeedback(),
        ComparatorWorstFallback(MixedFeedback()),
        _OptionalFeedback(),
    ],
    ids=["true_only", "mixed", "none", "fallback", "custom"],
)
def test_feedback_without_required_prediction_is_accepted(builder) -> None:
    optimizer = Optimizer(_problem()).set_strategy(DirectStrategy())
    optimizer.set_feedback_builder(builder)

    plan = optimizer._compile_plan()

    assert plan is not None
    builder = optimizer.feedback_builder
    assert builder is not None
    inputs = builder.contract().ports["feedback_builder"].inputs
    assert next(port for port in inputs if port.name == "prediction").optional
    assert next(port for port in inputs if port.name == "evaluation").optional
    assert not any(
        item.severity is Severity.ERROR and item.code == "unresolved_input"
        for item in plan.diagnostics
    )


def test_feedback_wrapper_preserves_required_prediction_fact() -> None:
    optimizer = Optimizer(_problem()).set_strategy(DirectStrategy())
    optimizer.set_feedback_builder(ComparatorWorstFallback(PredictedFeedback()))

    plan = optimizer._compile_plan()

    assert plan is not None
    diagnostics = [
        item
        for item in plan.diagnostics
        if item.path.components == ("optimizer", "feedback_builder")
        and item.code == "unresolved_input"
    ]
    assert len(diagnostics) == 1
    assert "objective predictions" in diagnostics[0].message


class _OpaqueStateStage(Stage):
    def contract(self) -> ComponentContract:
        return ComponentContract(state=StateContract(writes_enumerable=False))

    def execute(self, state: OptimizationState) -> OptimizationState:
        return state


def test_required_feedback_is_not_rejected_when_state_writes_are_unknown() -> None:
    strategy = DirectStrategy()
    feedback_index = strategy.declaration.steps.index(
        strategy.declaration.find("feedback")
    )
    strategy.declaration.steps.insert(feedback_index, _OpaqueStateStage())
    optimizer = (
        Optimizer(_problem())
        .set_algorithm(PSO())
        .set_strategy(strategy)
        .set_preset({"feedback_builder": {"type": "PredictedFeedback"}})
    )

    plan = optimizer._compile_plan()

    assert plan is not None
    assert not any(
        item.severity is Severity.ERROR and item.code == "unresolved_input"
        for item in plan.diagnostics
    )


def test_feedback_error_is_not_hidden_by_downstream_unknown_state_writes() -> None:
    strategy = DirectStrategy()
    feedback_index = strategy.declaration.steps.index(
        strategy.declaration.find("feedback")
    )
    strategy.declaration.steps.insert(feedback_index + 1, _OpaqueStateStage())
    optimizer = (
        Optimizer(_problem())
        .set_algorithm(PSO())
        .set_strategy(strategy)
        .set_preset({"feedback_builder": {"type": "PredictedFeedback"}})
    )

    plan = optimizer._compile_plan()

    assert plan is not None
    diagnostics = [
        item
        for item in plan.diagnostics
        if item.path.components == ("optimizer", "feedback_builder")
        and item.code == "unresolved_input"
    ]
    assert len(diagnostics) == 1
    assert "user preset" in diagnostics[0].message


def test_explicit_direct_strategy_runs_with_bundled_ga() -> None:
    optimizer = Optimizer(_problem(), seed=17).set_strategy(DirectStrategy())
    optimizer.set_termination(Termination(max_gen(1)))

    state = optimizer.run()

    assert state.gen == 1
    assert type(optimizer.algorithm).__name__ == "GA"
    assert isinstance(optimizer.evaluation_planner, EvaluateAll)
    assert _source(optimizer, "evaluation_planner") == "strategy default"
    assert isinstance(optimizer.feedback_builder, TrueOnlyFeedback)
    assert _source(optimizer, "feedback_builder") == "strategy default"


def test_repeated_resolution_keeps_default_origin_and_component(monkeypatch) -> None:
    _bundled(monkeypatch, 0.8)
    optimizer = Optimizer(_problem())

    optimizer.resolve_defaults()
    first = optimizer.evaluation_planner
    first_source = _source(optimizer, "evaluation_planner")
    optimizer.resolve_defaults()

    assert optimizer.evaluation_planner is first
    assert first_source == "bundled preset"
    assert _source(optimizer, "evaluation_planner") == first_source


def test_clearing_a_replaced_slot_selects_the_user_preset(monkeypatch) -> None:
    _bundled(monkeypatch, 0.8)
    optimizer = Optimizer(_problem()).set_preset(_preset(0.6))
    optimizer.resolve_defaults()
    replacement = RatioEvaluation(0.3)
    optimizer.evaluation_planner = replacement
    optimizer.resolve_defaults()
    assert optimizer.evaluation_planner is replacement
    assert _source(optimizer, "evaluation_planner") == "direct assignment"

    optimizer.evaluation_planner = None
    optimizer.resolve_defaults()

    assert isinstance(optimizer.evaluation_planner, RatioEvaluation)
    assert optimizer.evaluation_planner.ratio == 0.6
    assert _source(optimizer, "evaluation_planner") == "user preset"


def test_strategy_default_components_are_owned_per_optimizer() -> None:
    strategy = DirectStrategy()
    first = Optimizer(_problem()).set_strategy(strategy)
    second = Optimizer(_problem()).set_strategy(strategy)

    first.resolve_defaults()
    second.resolve_defaults()

    assert first.strategy is strategy
    assert second.strategy is strategy
    assert first.feedback_builder is not second.feedback_builder
    assert first.evaluation_planner is not second.evaluation_planner


def test_stage_local_component_does_not_replace_a_shared_binding() -> None:
    from saealib.execution.prepare import resolve_declaration

    local = EvaluateAll()
    shared = RatioEvaluation(0.5)
    resolved = resolve_declaration(
        Pipeline([EvaluationPlanStage(local)]),
        {"evaluation_planner": shared},
    )

    assert isinstance(resolved, Pipeline)
    assert isinstance(resolved.steps[0], EvaluationPlanStage)
    assert resolved.steps[0]._planner is local
    assert shared is not local


def test_resolution_metadata_is_not_a_public_optimizer_api() -> None:
    optimizer = Optimizer(_problem())
    optimizer.resolve_defaults()

    assert not hasattr(optimizer, "component_resolutions")
    assert not hasattr(optimizer, "prepare")
    assert not hasattr(saealib, "ComponentResolution")


def test_optimizer_state_without_binding_metadata_keeps_stored_components() -> None:
    optimizer = (
        Optimizer(_problem())
        .set_algorithm(PSO())
        .set_strategy(DirectStrategy())
        .set_termination(Termination(max_gen(2)))
    )
    optimizer.resolve_defaults()
    algorithm = optimizer.algorithm
    strategy = optimizer.strategy
    state = optimizer.__getstate__()
    for name in (
        "_component_origins",
        "_component_resolution",
        "_materialized_components",
        "_configuration_version",
        "_track_component_assignments",
    ):
        state.pop(name, None)
    hint = DefaultHint(
        key=POPULATION_SIZE,
        value=12,
        strength=DefaultStrength.FALLBACK,
        source="BuiltinDefaultProvider",
    )
    record = object.__new__(ResolvedDefault)
    object.__setattr__(record, "key", POPULATION_SIZE)
    object.__setattr__(record, "value", 12)
    object.__setattr__(record, "selected_hint", hint)
    object.__setattr__(record, "alternatives", (hint,))
    stored_resolution = object.__new__(DefaultResolution)
    object.__setattr__(stored_resolution, "values", {POPULATION_SIZE: 12})
    object.__setattr__(stored_resolution, "resolved", {POPULATION_SIZE: record})
    state["_default_resolution"] = pickle.loads(pickle.dumps(stored_resolution))
    stored_resolution = cast(DefaultResolution, state["_default_resolution"])
    stored_hint = cast(
        DefaultHint,
        getattr(stored_resolution.resolved[POPULATION_SIZE], "selected_hint"),
    )
    assert stored_hint.source == "BuiltinDefaultProvider"

    restored = Optimizer.__new__(Optimizer)
    restored.__setstate__(state)
    restored.resolve_defaults()

    assert restored.algorithm is algorithm
    assert restored.strategy is strategy
    assert _origins(restored)["algorithm"].source == "persisted"
    assert _origins(restored)["strategy"].source == "persisted"


def test_saved_direct_strategy_resumes_with_its_policy_defaults(monkeypatch) -> None:
    def sphere(x: object) -> np.ndarray:
        return np.array([float(np.sum(np.asarray(x) ** 2))])

    monkeypatch.setattr(sys.modules["__main__"], "sphere", sphere, raising=False)
    checkpoint = (
        Path(__file__).parent / "fixtures" / ("direct_strategy_missing_policies.pkl")
    )
    optimizer, state = Optimizer.load_pickle(checkpoint)
    strategy = optimizer.strategy

    resumed = optimizer.run_from(state)

    assert (resumed.gen, resumed.fe) == (3, 34)
    assert optimizer.strategy is strategy
    assert isinstance(strategy, DirectStrategy)
    assert isinstance(optimizer.evaluation_planner, EvaluateAll)
    assert _source(optimizer, "evaluation_planner") == "strategy default"
    assert isinstance(optimizer.feedback_builder, TrueOnlyFeedback)
    assert _source(optimizer, "feedback_builder") == "strategy default"


def test_first_yield_and_resume_reuse_component_sources(monkeypatch) -> None:
    _bundled(monkeypatch, None)
    optimizer = (
        Optimizer(_problem(), seed=17)
        .set_algorithm(PSO())
        .set_strategy(DirectStrategy())
        .set_termination(Termination(max_gen(1)))
    )

    iterator = optimizer.iterate()
    sources = {name: origin.source for name, origin in _origins(optimizer).items()}
    state = next(iterator)
    iterator.close()

    assert state.gen == 0
    assert _origins(optimizer)["algorithm"].source == "setter"
    assert _origins(optimizer)["strategy"].source == "setter"
    assert _origins(optimizer)["evaluation_planner"].source == "strategy default"

    optimizer.set_termination(Termination(max_gen(3)))
    resumed = optimizer.run_from(state)

    assert resumed.gen == 3
    assert {
        name: origin.source for name, origin in _origins(optimizer).items()
    } == sources
