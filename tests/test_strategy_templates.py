"""Behavioral checks for surrogate-assisted flow declarations."""

from __future__ import annotations

import pickle
import warnings
from typing import Any, cast

import numpy as np
import pytest
from _evaluation_helpers import _ControlledEvaluator, _Evaluator

import saealib.callback.events as callback_events
from saealib import (
    GA,
    CrossoverBLXAlpha,
    IndividualBasedStrategy,
    LHSInitializer,
    MutationUniform,
    Optimizer,
    PreSelectionStrategy,
    SequentialSelection,
    Termination,
    TruncationSelection,
)
from saealib.acquisition import MeanPrediction
from saealib.callback import CallbackManager, GenerationEndEvent, PostAskEvent
from saealib.callback.events import Event
from saealib.context import OptimizationState
from saealib.core.compiler.structured import StructuredGraph
from saealib.core.contracts import ComponentContract, StageFlowSummary, StateContract
from saealib.core.state.keys import RUNTIME_GENERATION, USER_DATA
from saealib.exceptions import ValidationError
from saealib.execution import (
    AsyncEvaluationScheduler,
    AsyncEvaluator,
    Evaluator,
    SerialEvaluator,
)
from saealib.execution.runtime import create_runtime, resolve_plan
from saealib.pipeline import (
    Branch,
    ComponentRef,
    Condition,
    Loop,
    Pipeline,
    Repeat,
    Stage,
)
from saealib.policies.evaluation import (
    EvaluateAll,
    RatioEvaluation,
    RepeatedEvaluation,
    TopKEvaluation,
)
from saealib.policies.feedback import MixedFeedback, TrueOnlyFeedback
from saealib.problem import Problem
from saealib.registry import build, to_spec
from saealib.stages import (
    AcquisitionStage,
    ArchiveUpdateStage,
    AskStage,
    CountGenerationStage,
    EvaluationAcknowledgeStage,
    EvaluationApplyStage,
    EvaluationCollectStage,
    EvaluationPlanStage,
    EvaluationSubmitStage,
    FeedbackStage,
    SurrogateFitStage,
    SurrogatePredictStage,
    TellStage,
)
from saealib.strategies import (
    DirectStrategy,
    GenerationBasedStrategy,
    SteadyStateStrategy,
    TemplateStrategy,
)
from saealib.strategies.ib import build_ib_template
from saealib.strategies.ps import build_ps_template
from saealib.surrogate.manager import LocalSurrogateManager
from saealib.surrogate.rbf import RBFSurrogate
from saealib.surrogate.rbf_kernels import GaussianKernel


class UserDirectStrategy(DirectStrategy):
    pass


class UserGenerationBasedStrategy(GenerationBasedStrategy):
    pass


class DeclarationCondition(Condition):
    def __init__(self, threshold: int = 1) -> None:
        self.threshold = threshold

    def contract(self):
        from saealib.core.contracts import StateContract

        return StateContract()

    def evaluate(self, view) -> bool:
        return False


class LambdaCondition(Condition):
    def __init__(self, predicate=None) -> None:
        self.predicate = predicate or (lambda value: value)

    def contract(self):
        from saealib.core.contracts import StateContract

        return StateContract()

    def evaluate(self, view) -> bool:
        return bool(self.predicate(view))


class DeclarationStage(Stage):
    def __init__(self, marker: str) -> None:
        super().__init__()
        self.marker = marker

    def execute(self, state: OptimizationState) -> OptimizationState:
        return state


class DeclarationTemplateSubclass(TemplateStrategy):
    def __init__(self, pipeline, marker="default"):
        self.marker = marker
        super().__init__(pipeline)


class InheritedDeclarationTemplate(TemplateStrategy):
    pass


class GenerationBelow(DeclarationCondition):
    def contract(self):
        return StateContract(reads=(RUNTIME_GENERATION,))

    def evaluate(self, view) -> bool:
        return int(view.get(RUNTIME_GENERATION)) < self.threshold


class GenerationReached(DeclarationCondition):
    def contract(self):
        return StateContract(reads=(RUNTIME_GENERATION,))

    def evaluate(self, view) -> bool:
        return int(view.get(RUNTIME_GENERATION)) >= self.threshold


class MarkerStage(Stage):
    def __init__(self, marker: str):
        super().__init__()
        self.marker = marker

    def contract(self):
        return ComponentContract(
            state=StateContract(reads=(USER_DATA,), writes=(USER_DATA,))
        )

    def execute(self, state: OptimizationState) -> OptimizationState:
        data = dict(state.data or {})
        markers = list(data.get("declaration_markers", ()))
        markers.append(self.marker)
        data["declaration_markers"] = markers
        return state.replace(data=data)


class StageCallLog:
    def __init__(self) -> None:
        self.values: list[int] = []


class RecordingCountGenerationStage(CountGenerationStage):
    def __init__(self, calls: StageCallLog) -> None:
        super().__init__()
        self.calls = calls

    def flow_summary(self) -> StageFlowSummary:
        return cast(StageFlowSummary, None)

    def execute(self, state: OptimizationState) -> OptimizationState:
        self.calls.values.append(state.gen)
        return super().execute(state)


class RecordingSurrogateFitStage(SurrogateFitStage):
    def __init__(self, surrogate_manager: Any, calls: StageCallLog) -> None:
        super().__init__(surrogate_manager)
        self.calls = calls

    def flow_summary(self) -> StageFlowSummary:
        return cast(StageFlowSummary, None)

    def execute(self, state: OptimizationState) -> OptimizationState:
        self.calls.values.append(state.gen)
        return super().execute(state)


class RecordingAskStage(AskStage):
    def __init__(self, algorithm: Any, n_offspring: int | None, calls: StageCallLog):
        super().__init__(algorithm, n_offspring=n_offspring)
        self.calls = calls

    def flow_summary(self) -> StageFlowSummary:
        return cast(StageFlowSummary, None)

    def execute(self, state: OptimizationState) -> OptimizationState:
        self.calls.values.append(state.gen)
        return super().execute(state)


def _sphere(x):
    return np.array([float(np.sum(np.asarray(x) ** 2))])


def _one_generation(state):
    return state.gen >= 1


def _two_generations(state):
    return state.gen >= 2


def _four_generations(state):
    return state.gen >= 4


def _twelve_evaluations(state):
    return state.fe >= 12


def _assert_checkpoint_matches(
    actual, expected, *, in_flight_request_ids=(), portable_checkpoint=False
):
    assert (actual.gen, actual.fe) == (expected.gen, expected.fe)
    for collection_name in ("population", "archive"):
        actual_collection = getattr(actual, collection_name)
        expected_collection = getattr(expected, collection_name)
        for column in ("id", "x", "f", "g", "cv", "objective_source"):
            np.testing.assert_array_equal(
                actual_collection.get_array(column),
                expected_collection.get_array(column),
            )
    assert actual.history is not None
    assert expected.history is not None
    assert actual.history.enabled == expected.history.enabled
    for channel in sorted(expected.history.enabled):
        actual_columns = actual.history.channel(channel)
        expected_columns = expected.history.channel(channel)
        assert actual_columns.keys() == expected_columns.keys()
        for name, expected_values in expected_columns.items():
            if (
                portable_checkpoint
                and name in {"origin_decision_count", "origin_gen"}
                and in_flight_request_ids
            ):
                request_ids = actual_columns.get("request_id")
                if request_ids is not None:
                    # Portable checkpoints do not store in-memory request origins.
                    mask = np.isin(request_ids, tuple(in_flight_request_ids))
                    expected_values = np.array(expected_values, copy=True)
                    expected_values[mask] = -1
            np.testing.assert_array_equal(
                actual_columns[name], expected_values, err_msg=f"{channel}.{name}"
            )


def _assert_history_matches_evaluations(state, *, final=True):
    """Check the evaluation channel against the state's evaluation count.

    ``final`` marks the state returned by ``run()``, whose count matches the
    History totals.  A state taken at a generation boundary carries that
    boundary's count, while the History object it shares keeps recording
    later evaluations.
    """
    assert state.history is not None
    assert state.history.is_enabled("summary")
    assert state.history.is_enabled("evaluation")
    evaluation = state.history.channel("evaluation")
    assert int(evaluation["fe_after"][-1]) >= state.fe
    if final:
        assert int(np.sum(evaluation["size"])) == state.fe
        assert int(evaluation["fe_after"][-1]) == state.fe
    else:
        assert int(state.history.channel("summary")["fe"][-1]) == state.fe
    candidate_ids = np.concatenate(
        [
            np.asarray(block).reshape(-1)
            for block in state.history.blocks("evaluation", "candidate_ids")
        ]
    )
    assert len(np.unique(candidate_ids)) == len(candidate_ids)


def _problem() -> Problem:
    return Problem(
        func=_sphere,
        dim=2,
        n_obj=1,
        direction=np.array([-1.0]),
        lb=[-5.0, -5.0],
        ub=[5.0, 5.0],
    )


def _algorithm() -> GA:
    return GA(
        crossover=CrossoverBLXAlpha(prob=0.9, alpha=0.4),
        mutation=MutationUniform(prob_var=0.1),
        parent_selection=SequentialSelection(),
        survivor_selection=TruncationSelection(),
    )


def _optimizer(
    strategy: Any,
    *,
    asynchronous: bool = False,
    evaluator: Evaluator | None = None,
    max_pending: int = 1,
) -> Optimizer:
    optimizer = (
        Optimizer(_problem(), seed=9)
        .set_initializer(LHSInitializer(6, 4, 9))
        .set_algorithm(_algorithm())
        .set_strategy(strategy)
        .set_termination(Termination(lambda state: state.gen >= 2))
    )
    optimizer.set_surrogate_manager(
        LocalSurrogateManager(RBFSurrogate(GaussianKernel()))
    ).set_acquisition(MeanPrediction())
    if asynchronous:
        evaluator = evaluator or _Evaluator()
        optimizer.set_evaluator(evaluator)
        optimizer.set_async_evaluation_scheduler(
            AsyncEvaluationScheduler(evaluator, max_pending=max_pending)
        )
    return optimizer


class _RecordingPlanner(RatioEvaluation):
    def __init__(self) -> None:
        super().__init__(0.5, sanitize_nonfinite=True)
        self.calls = 0

    def plan(self, candidates, acquisition, *, ctx):
        self.calls += 1
        return super().plan(candidates, acquisition, ctx=ctx)


class _RecordingFeedback(MixedFeedback):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def build(self, candidates, prediction, evaluation, evaluated_indices, *, ctx):
        self.calls += 1
        return super().build(
            candidates, prediction, evaluation, evaluated_indices, ctx=ctx
        )


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize(
    ("strategy", "planner", "builder"),
    [
        (
            IndividualBasedStrategy(0.5),
            RatioEvaluation(0.25, sanitize_nonfinite=True),
            MixedFeedback(),
        ),
        (
            PreSelectionStrategy(5, 2),
            TopKEvaluation(1, sanitize_nonfinite=True),
            TrueOnlyFeedback(),
        ),
    ],
)
def test_default_and_explicit_policies_are_bound_for_each_strategy(
    asynchronous, strategy, planner, builder
):
    optimizer = _optimizer(strategy, asynchronous=asynchronous)
    optimizer.run()
    default_planner = strategy.pipeline.find("evaluation_plan")._planner
    assert isinstance(default_planner, type(strategy.evaluation_planner))
    assert isinstance(
        strategy.pipeline.find("feedback")._builder, type(strategy.feedback_builder)
    )
    if isinstance(strategy, IndividualBasedStrategy):
        assert default_planner.ratio == strategy.evaluation_ratio
    else:
        assert default_planner.k == min(strategy.n_select, strategy.n_candidates)

    explicit = _optimizer(strategy, asynchronous=asynchronous)
    explicit.set_evaluation_planner(planner).set_feedback_builder(builder)
    explicit.run()
    resolved = cast(Any, explicit.strategy).pipeline
    assert resolved.find("evaluation_plan")._planner is planner
    assert resolved.find("feedback")._builder is builder


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize(
    "strategy_factory",
    [lambda: IndividualBasedStrategy(0.25), lambda: PreSelectionStrategy(5, 2)],
)
def test_user_preset_policies_are_bound_for_each_strategy(
    asynchronous, strategy_factory
):
    strategy = strategy_factory()
    optimizer = _optimizer(strategy, asynchronous=asynchronous)
    if isinstance(strategy, IndividualBasedStrategy):
        planner = {
            "type": "RatioEvaluation",
            "params": {"ratio": 0.5, "sanitize_nonfinite": True},
        }
        builder = {"type": "TrueOnlyFeedback", "params": {}}
    else:
        planner = {
            "type": "TopKEvaluation",
            "params": {"k": 1, "sanitize_nonfinite": True},
        }
        builder = {"type": "MixedFeedback", "params": {}}
    optimizer.set_preset({"evaluation_planner": planner, "feedback_builder": builder})
    optimizer.run()
    resolved = cast(Any, optimizer.strategy).pipeline
    assert resolved.find("evaluation_plan")._planner is optimizer.evaluation_planner
    assert resolved.find("feedback")._builder is optimizer.feedback_builder


@pytest.mark.parametrize("asynchronous", [False, True])
def test_steady_state_policy_precedence_is_consistent(asynchronous):
    default_strategy = SteadyStateStrategy()
    default_optimizer = _optimizer(default_strategy, asynchronous=asynchronous)
    default_optimizer.run()
    default_pipeline = default_strategy.pipeline
    assert default_pipeline is not None
    default_plan = cast(Any, default_pipeline.find("evaluation_plan"))
    default_feedback = cast(Any, default_pipeline.find("feedback"))
    default_ask = cast(Any, default_pipeline.find("ask"))
    assert isinstance(default_plan._planner, EvaluateAll)
    assert isinstance(default_feedback._builder, TrueOnlyFeedback)
    assert default_ask._n_offspring == 1

    preset_strategy = SteadyStateStrategy()
    preset_optimizer = _optimizer(preset_strategy, asynchronous=asynchronous)
    preset_optimizer.set_preset(
        {
            "evaluation_planner": {
                "type": "RepeatedEvaluation",
                "params": {"replicates": 2},
            },
            "feedback_builder": {"type": "MixedFeedback", "params": {}},
        }
    )
    preset_optimizer.run()
    preset_pipeline = preset_strategy.pipeline
    assert preset_pipeline is not None
    preset_plan = cast(Any, preset_pipeline.find("evaluation_plan"))
    preset_feedback = cast(Any, preset_pipeline.find("feedback"))
    assert preset_plan._planner is preset_optimizer.evaluation_planner
    assert preset_plan._planner.replicates == 2
    assert preset_feedback._builder is preset_optimizer.feedback_builder

    explicit_strategy = SteadyStateStrategy()
    explicit_optimizer = _optimizer(explicit_strategy, asynchronous=asynchronous)
    planner = RepeatedEvaluation(replicates=3)
    builder = MixedFeedback()
    explicit_optimizer.set_preset(
        {
            "evaluation_planner": {
                "type": "RepeatedEvaluation",
                "params": {"replicates": 2},
            },
            "feedback_builder": {"type": "TrueOnlyFeedback", "params": {}},
        }
    )
    explicit_optimizer.set_evaluation_planner(planner).set_feedback_builder(builder)
    explicit_optimizer.run()
    explicit_pipeline = explicit_strategy.pipeline
    assert explicit_pipeline is not None
    explicit_plan = cast(Any, explicit_pipeline.find("evaluation_plan"))
    explicit_feedback = cast(Any, explicit_pipeline.find("feedback"))
    assert explicit_plan._planner is planner
    assert explicit_feedback._builder is builder


def test_steady_state_refills_after_no_delivery_with_active_repeated_plan():
    evaluator = _ControlledEvaluator()
    optimizer = _optimizer(
        SteadyStateStrategy(),
        asynchronous=True,
        evaluator=evaluator,
        max_pending=2,
    ).set_history(["summary", "evaluation"])
    optimizer.set_evaluation_planner(RepeatedEvaluation(replicates=2))
    optimizer.set_termination(Termination(lambda state: state.fe >= 50))
    post_ask_events = []
    callbacks = CallbackManager()
    callbacks.register(PostAskEvent, lambda event: post_ask_events.append(event.ctx.fe))
    optimizer.cbmanager = callbacks
    optimizer.resolve_defaults()
    optimizer._compile_plan()
    initializer = optimizer.initializer
    assert initializer is not None
    state = initializer.initialize(optimizer, optimizer.problem)
    runtime = create_runtime(optimizer)
    session = runtime.initialize(resolve_plan(optimizer), state)
    assert session is not None

    for _ in range(10):
        if len(evaluator.submitted) == 2:
            break
        session = runtime.advance(session).session
        assert session is not None
    assert len(evaluator.submitted) == 2
    assert len(session.state.pending_evaluations) == 2
    assert len(post_ask_events) == 1

    first_request_id = int(evaluator.submitted[0].request_id)
    evaluator.released_request_ids = {first_request_id}
    session = runtime.advance(session).session
    assert session is not None
    assert len(evaluator.submitted) == 3
    assert len(session.state.pending_evaluations) == 2
    assert len(post_ask_events) == 2

    remaining_initial_id = int(evaluator.submitted[1].request_id)
    second_candidate_id = int(evaluator.submitted[2].request_id)
    evaluator.released_request_ids = {remaining_initial_id, second_candidate_id}
    session = runtime.advance(session).session
    assert session is not None
    active_plan = session.state.evaluation_plan
    active_plan_state = session.state.evaluation_plan_state
    assert active_plan is not None
    assert active_plan_state is not None
    active_request_ids = {int(request.request_id) for request in active_plan.requests}
    assert len(active_request_ids & set(active_plan_state.completed)) == 1
    pending_active_ids = active_request_ids & set(session.state.pending_evaluations)
    assert len(pending_active_ids) == 1
    assert len(post_ask_events) == 2

    submitted_before = len(evaluator.submitted)
    evaluator.released_request_ids.clear()
    session = runtime.advance(session).session
    assert session is not None

    assert len(evaluator.submitted) == submitted_before + 1
    assert len(post_ask_events) == 3
    assert pending_active_ids <= set(session.state.pending_evaluations)

    evaluator.release_all = True
    for _ in range(1000):
        if session.finished:
            break
        session = runtime.advance(session).session
        assert session is not None
    assert session.finished


@pytest.mark.parametrize(
    "strategy",
    [
        DirectStrategy(),
        GenerationBasedStrategy(1),
        IndividualBasedStrategy(0.4),
        PreSelectionStrategy(7, 3),
        SteadyStateStrategy(),
    ],
)
def test_strategy_specs_round_trip(strategy):
    rebuilt = build(to_spec(strategy))
    assert type(rebuilt) is type(strategy)
    if isinstance(strategy, IndividualBasedStrategy):
        assert rebuilt.evaluation_ratio == strategy.evaluation_ratio
    elif isinstance(strategy, PreSelectionStrategy):
        assert rebuilt.n_candidates == strategy.n_candidates
        assert rebuilt.n_select == strategy.n_select
    assert to_spec(rebuilt) == to_spec(strategy)


def test_template_strategy_subclass_overriding_pipeline_requires_registry_spec():
    original = DeclarationTemplateSubclass(
        Pipeline([CountGenerationStage()]), marker="preserved"
    )

    with pytest.raises(ValidationError, match=r"pipeline.*_registry_spec"):
        to_spec(original)


def test_template_strategy_subclass_inheriting_init_keeps_declaration():
    original = InheritedDeclarationTemplate(Pipeline([CountGenerationStage()]))

    spec = to_spec(original)
    rebuilt = build(spec)

    assert type(rebuilt) is InheritedDeclarationTemplate
    assert isinstance(rebuilt.declaration, Pipeline)
    assert isinstance(rebuilt.declaration.stages[0], CountGenerationStage)
    assert to_spec(rebuilt) == spec == to_spec(original)


def test_declaration_spec_preserves_controls_components_and_references():
    reference = ComponentRef("proposer", "algorithm")
    declaration = Pipeline(
        [
            AskStage(reference),
            EvaluationPlanStage(RatioEvaluation(0.5, sanitize_nonfinite=True)),
            Branch(
                DeclarationCondition(2),
                then=Pipeline([Repeat(DeclarationStage("then"), 2)]),
                else_=DeclarationStage("else"),
                name="choice",
            ),
            Loop(DeclarationStage("loop"), until=DeclarationCondition(3)),
            Pipeline([DeclarationStage("nested")]),
        ],
        name="declared",
    )
    strategy = TemplateStrategy(declaration)
    spec = to_spec(strategy)
    restored = build(spec)
    assert isinstance(restored, TemplateStrategy)
    assert isinstance(restored.declaration, Pipeline)
    assert isinstance(restored.declaration.stages[0], AskStage)
    assert restored.declaration.stages[0]._refs == (("_algorithm", reference),)
    assert isinstance(restored.declaration.stages[1]._planner, RatioEvaluation)
    assert isinstance(restored.declaration.stages[2], Branch)
    assert isinstance(restored.declaration.stages[2].then, Pipeline)
    assert isinstance(restored.declaration.stages[2].then.stages[0], Repeat)
    assert restored.declaration.stages[2].name == "choice"
    assert restored.declaration.stages[2].condition.threshold == 2
    assert restored.declaration.stages[2].then.stages[0].count == 2
    assert restored.declaration.stages[2].then.stages[0].body.marker == "then"
    assert isinstance(restored.declaration.stages[3], Loop)
    assert restored.declaration.stages[3].until.threshold == 3
    assert isinstance(restored.declaration.stages[4], Pipeline)
    assert restored.declaration.stages[4].stages[0].marker == "nested"
    assert to_spec(restored) == spec
    assert to_spec(strategy) == spec


def test_declaration_spec_rejects_local_stage_with_location():
    class LocalStage(Stage):
        def execute(self, state: OptimizationState) -> OptimizationState:
            return state

    with pytest.raises(ValidationError, match=r"declaration.*stages"):
        to_spec(TemplateStrategy(Pipeline([LocalStage()])))


def test_declaration_spec_rejects_condition_closure_with_location():
    with pytest.raises(ValidationError, match=r"declaration.*condition.*predicate"):
        to_spec(
            TemplateStrategy(
                Pipeline([Branch(LambdaCondition(), then=DeclarationStage("then"))])
            )
        )


def _declaration_for_run() -> Pipeline:
    return Pipeline(
        [
            Loop(
                Pipeline(
                    [
                        Branch(
                            GenerationBelow(1),
                            then=Pipeline([MarkerStage("then")]),
                            else_=Pipeline([MarkerStage("else")]),
                            name="choice",
                        ),
                        CountGenerationStage(),
                    ],
                    name="loop_body",
                ),
                until=GenerationReached(2),
                name="loop",
            ),
            Repeat(
                Pipeline([MarkerStage("repeat")], name="repeat_body"),
                count=2,
                name="repeat",
            ),
            Pipeline([MarkerStage("nested")], name="nested_body"),
            SurrogateFitStage(ComponentRef("fitter", "surrogate_manager")),
            AskStage(ComponentRef("proposer", "algorithm"), n_offspring=2),
            SurrogatePredictStage(ComponentRef("predictor", "surrogate_manager")),
            AcquisitionStage(ComponentRef("acquisition", "acquisition")),
            EvaluationPlanStage(RatioEvaluation(0.5)),
            EvaluationSubmitStage(ComponentRef("evaluator", "evaluator")),
            EvaluationCollectStage(ComponentRef("evaluator", "evaluator")),
            EvaluationApplyStage(),
            ArchiveUpdateStage(),
            FeedbackStage(ComponentRef("feedback_builder", "feedback_builder")),
            TellStage(ComponentRef("feedback_consumer", "algorithm")),
            EvaluationAcknowledgeStage(ComponentRef("evaluator", "evaluator")),
            MarkerStage("tail"),
            CountGenerationStage(),
        ],
        name="round_trip",
    )


def _declaration_optimizer(strategy: Any, asynchronous: bool) -> Optimizer:
    optimizer = (
        Optimizer(_problem(), seed=29)
        .set_initializer(LHSInitializer(6, 4, 29))
        .set_algorithm(_algorithm())
        .set_surrogate_manager(LocalSurrogateManager(RBFSurrogate(GaussianKernel())))
        .set_acquisition(MeanPrediction())
        .set_strategy(strategy)
        .set_termination(Termination(_two_generations))
        .set_history(["summary", "population", "evaluation"])
    )
    if asynchronous:
        evaluator = AsyncEvaluator(SerialEvaluator())
        optimizer.set_evaluator(evaluator)
        optimizer.set_async_evaluation_scheduler(
            AsyncEvaluationScheduler(evaluator, max_pending=2)
        )
    return optimizer


@pytest.mark.parametrize("asynchronous", [False, True])
def test_declaration_round_trip_execution_matches_after_preparation(asynchronous):
    original = InheritedDeclarationTemplate(_declaration_for_run())
    source_spec = to_spec(original)
    original_optimizer = _declaration_optimizer(original, asynchronous)
    original_state = original_optimizer.run()
    prepared_spec = to_spec(original)
    assert prepared_spec == source_spec

    restored = build(source_spec)
    assert type(restored) is InheritedDeclarationTemplate
    restored_optimizer = _declaration_optimizer(restored, asynchronous)
    restored_state = restored_optimizer.run()
    assert restored_state.decision_count == original_state.decision_count
    assert restored_state.data == original_state.data
    assert original_state.data["declaration_markers"].count("then") == 1
    assert original_state.data["declaration_markers"].count("else") == 1
    assert original_state.data["declaration_markers"].count("repeat") == 2
    _assert_checkpoint_matches(restored_state, original_state)

    restored_prepared = build(prepared_spec)
    assert type(restored_prepared) is InheritedDeclarationTemplate
    restored_prepared_optimizer = _declaration_optimizer(
        restored_prepared, asynchronous
    )
    restored_prepared_state = restored_prepared_optimizer.run()
    assert restored_prepared_state.decision_count == original_state.decision_count
    assert restored_prepared_state.data == original_state.data
    _assert_checkpoint_matches(restored_prepared_state, original_state)


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize(
    "strategy_factory",
    [
        lambda: IndividualBasedStrategy(0.5),
        lambda: PreSelectionStrategy(7, 3),
        SteadyStateStrategy,
    ],
)
def test_all_emitted_strategy_events_receive_optimization_state(
    asynchronous, strategy_factory
):
    optimizer = _optimizer(strategy_factory(), asynchronous=asynchronous)
    contexts: dict[type[Event], list[object]] = {}
    callbacks = CallbackManager()
    event_types = [
        value
        for value in vars(callback_events).values()
        if isinstance(value, type) and issubclass(value, Event) and value is not Event
    ]
    for event_type in event_types:
        contexts[event_type] = []
        callbacks.register(
            event_type,
            lambda event, values=contexts[event_type]: values.append(event.ctx),
        )
    optimizer.cbmanager = callbacks

    optimizer.run()

    emitted = [context for values in contexts.values() for context in values]
    assert emitted
    assert all(isinstance(context, OptimizationState) for context in emitted)


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize(
    ("strategy_factory", "update"),
    [
        (
            lambda: IndividualBasedStrategy(0.1),
            lambda strategy: setattr(strategy, "evaluation_ratio", 0.75),
        ),
        (
            lambda: PreSelectionStrategy(5, 2),
            lambda strategy: (
                setattr(strategy, "n_candidates", 3),
                setattr(strategy, "n_select", 1),
            ),
        ),
    ],
)
def test_strategy_parameter_change_rebuilds_next_commit(
    asynchronous, strategy_factory, update
):
    strategy = strategy_factory()
    optimizer = _optimizer(strategy, asynchronous=asynchronous)
    optimizer.set_termination(Termination(_two_generations))
    changed: list[bool] = []
    callbacks = CallbackManager()

    def change(event):
        if not changed and event.ctx.gen >= 1:
            changed.append(True)
            update(strategy)

    callbacks.register(GenerationEndEvent, change)
    optimizer.cbmanager = callbacks
    optimizer.run()

    assert changed
    ask = strategy.pipeline.find("ask")
    planner = strategy.pipeline.find("evaluation_plan")._planner
    if isinstance(strategy, IndividualBasedStrategy):
        assert planner.ratio == 0.75
    else:
        assert ask._n_offspring == 3
        assert planner.k == 1


_PIPELINE_REPLACEMENT_CASES = [
    (DirectStrategy, "count_generation"),
    (lambda: GenerationBasedStrategy(1), "surrogate_fit"),
    (lambda: IndividualBasedStrategy(0.5), "count_generation"),
    (lambda: PreSelectionStrategy(5, 2), "count_generation"),
    (SteadyStateStrategy, "count_generation"),
]


def _recording_pipeline_stage(strategy, name, calls: StageCallLog):
    pipeline = cast(Pipeline, strategy.pipeline)
    stage = pipeline.find(name)
    if name == "surrogate_fit":
        assert isinstance(stage, SurrogateFitStage)
        return RecordingSurrogateFitStage(stage._sm, calls)
    assert isinstance(stage, CountGenerationStage)
    return RecordingCountGenerationStage(calls)


def _assert_pipeline_replacement_was_prepared(optimizer, initial_plan):
    plan = optimizer.executable_plan
    assert plan is not initial_plan
    assert any(
        diagnostic.code == "cannot_verify_stage_flow" for diagnostic in plan.diagnostics
    )
    return plan


@pytest.mark.parametrize(
    ("strategy_factory", "stage_name"), _PIPELINE_REPLACEMENT_CASES
)
def test_strategy_pipeline_replacement_before_run_is_prepared(
    strategy_factory, stage_name
):
    strategy = strategy_factory()
    optimizer = _optimizer(strategy)
    optimizer.resolve_defaults()
    optimizer._compile_plan()
    initial_plan = optimizer.executable_plan
    pipeline = cast(Pipeline, strategy.pipeline)
    calls = StageCallLog()
    replacement = _recording_pipeline_stage(strategy, stage_name, calls)

    pipeline.replace(stage_name, replacement)
    assert pipeline.find(stage_name) is replacement
    assert strategy.declaration.find(stage_name) is replacement

    iterator = optimizer.iterate()
    plan = _assert_pipeline_replacement_was_prepared(optimizer, initial_plan)
    next(iterator)
    next(iterator)

    assert calls.values == [0]
    assert any(
        getattr(node.component, "stage", None) is not None
        and isinstance(
            node.component.stage,
            (RecordingCountGenerationStage, RecordingSurrogateFitStage),
        )
        for node in plan.graph.nodes
    )


@pytest.mark.parametrize(
    ("strategy_factory", "stage_name"), _PIPELINE_REPLACEMENT_CASES
)
def test_strategy_pipeline_replacement_between_yields_uses_next_step(
    strategy_factory, stage_name
):
    strategy = strategy_factory()
    optimizer = _optimizer(strategy).set_termination(
        Termination(lambda state: state.gen >= 6)
    )
    iterator = optimizer.iterate()
    next(iterator)
    first = next(iterator)
    initial_plan = optimizer.executable_plan
    pipeline = cast(Pipeline, strategy.pipeline)
    calls = StageCallLog()
    replacement = _recording_pipeline_stage(strategy, stage_name, calls)

    pipeline.replace(stage_name, replacement)
    assert calls.values == []
    next(iterator)
    plan = _assert_pipeline_replacement_was_prepared(optimizer, initial_plan)

    expected_generation = first.gen - (2 if stage_name == "surrogate_fit" else 1)
    assert calls.values == [expected_generation]
    assert any(
        getattr(node.component, "stage", None) is not None
        and isinstance(
            node.component.stage,
            (RecordingCountGenerationStage, RecordingSurrogateFitStage),
        )
        for node in plan.graph.nodes
    )


@pytest.mark.parametrize(
    ("strategy_factory", "stage_name"), _PIPELINE_REPLACEMENT_CASES
)
def test_strategy_pipeline_replacement_from_callback_uses_next_step(
    strategy_factory, stage_name
):
    strategy = strategy_factory()
    optimizer = _optimizer(strategy).set_termination(
        Termination(lambda state: state.gen >= 6)
    )
    calls = StageCallLog()
    replacement: list[Stage] = []
    callbacks = CallbackManager()

    def replace_after_ask(event: PostAskEvent) -> None:
        if replacement:
            return
        pipeline = cast(Pipeline, strategy.pipeline)
        value = _recording_pipeline_stage(strategy, stage_name, calls)
        replacement.append(value)
        pipeline.replace(stage_name, value)

    callbacks.register(PostAskEvent, replace_after_ask)
    optimizer.cbmanager = callbacks
    iterator = optimizer.iterate()
    next(iterator)
    first = next(iterator)
    initial_plan = optimizer.executable_plan

    assert replacement
    assert calls.values == []
    next(iterator)
    plan = _assert_pipeline_replacement_was_prepared(optimizer, initial_plan)

    expected_generation = first.gen - (2 if stage_name == "surrogate_fit" else 1)
    assert calls.values == [expected_generation]
    assert any(
        getattr(node.component, "stage", None) is not None
        and isinstance(
            node.component.stage,
            (RecordingCountGenerationStage, RecordingSurrogateFitStage),
        )
        for node in plan.graph.nodes
    )


def test_generation_based_nested_pipeline_replacement_runs_only_in_true_flow():
    strategy = GenerationBasedStrategy(1)
    optimizer = _optimizer(strategy).set_termination(
        Termination(lambda state: state.gen >= 4)
    )
    optimizer.resolve_defaults()
    optimizer._compile_plan()
    pipeline = cast(Pipeline, strategy.pipeline)
    true_generation = cast(Pipeline, pipeline.find("true_generation", recursive=True))
    surrogate_generation = cast(
        Pipeline, pipeline.find("surrogate_generation", recursive=True)
    )
    original_ask = cast(AskStage, true_generation.find("ask"))
    calls = StageCallLog()
    replacement = RecordingAskStage(
        original_ask._algorithm, original_ask._n_offspring, calls
    )

    true_generation.replace("ask", replacement)
    assert true_generation.find("ask") is replacement
    assert isinstance(surrogate_generation.find("ask"), AskStage)
    initial_plan = optimizer.executable_plan

    list(optimizer.iterate())

    assert optimizer.executable_plan is not initial_plan
    assert calls.values == [2, 4]
    declared_true_generation = cast(
        Pipeline, strategy.declaration.find("true_generation", recursive=True)
    )
    assert declared_true_generation.find("ask") is replacement
    prepared_true_generation = cast(
        Pipeline,
        cast(Pipeline, strategy.pipeline).find("true_generation", recursive=True),
    )
    assert isinstance(
        prepared_true_generation.find("ask"),
        RecordingAskStage,
    )
    plan = optimizer.executable_plan
    assert plan is not None
    assert any(
        diagnostic.code == "cannot_verify_stage_flow" for diagnostic in plan.diagnostics
    )


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize(
    "strategy_factory",
    [
        DirectStrategy,
        lambda: GenerationBasedStrategy(1),
        lambda: IndividualBasedStrategy(0.5),
        lambda: PreSelectionStrategy(5, 2),
        SteadyStateStrategy,
    ],
)
def test_same_class_ask_replacement_after_start_is_observed(
    asynchronous, strategy_factory
):
    strategy = strategy_factory()
    optimizer = _optimizer(strategy, asynchronous=asynchronous).set_termination(
        Termination(lambda state: state.gen >= 6)
    )
    iterator = optimizer.iterate()
    next(iterator)
    next(iterator)
    initial_plan = optimizer.executable_plan
    pipeline = cast(Pipeline, strategy.pipeline)
    owner = (
        cast(Pipeline, pipeline.find("true_generation", recursive=True))
        if isinstance(strategy, GenerationBasedStrategy)
        else pipeline
    )
    original = cast(AskStage, owner.find("ask"))
    replacement = AskStage(original._algorithm, n_offspring=7)
    revision = strategy._pipeline_revision.value
    owner.replace("ask", replacement)
    assert strategy._pipeline_revision.value == revision + 1
    next(iterator)
    assert optimizer.executable_plan is not initial_plan
    declared_pipeline = (
        cast(Pipeline, strategy.declaration.find("true_generation", recursive=True))
        if isinstance(strategy, GenerationBasedStrategy)
        else strategy.declaration
    )
    assert cast(AskStage, declared_pipeline.find("ask"))._n_offspring == 7


@pytest.mark.parametrize(
    "strategy_factory",
    [
        DirectStrategy,
        UserDirectStrategy,
        lambda: GenerationBasedStrategy(0),
        lambda: UserGenerationBasedStrategy(0),
        lambda: IndividualBasedStrategy(0.5),
        lambda: PreSelectionStrategy(5, 2),
        SteadyStateStrategy,
    ],
)
@pytest.mark.parametrize(
    "state_shape", ["without_declaration", "without_generated_nodes"]
)
def test_strategy_pickle_without_declaration_resumes(strategy_factory, state_shape):
    strategy = strategy_factory()
    revision = strategy._pipeline_revision.value
    if state_shape == "without_declaration":
        strategy.__dict__.pop("declaration", None)
        strategy.__dict__.pop("_pipeline_revision", None)
    else:
        strategy.__dict__["_pipeline_revision"] = revision
        cast(Pipeline, strategy.declaration).__dict__.pop("_pipeline_revision", None)
    strategy.__dict__.pop("_generated_ask", None)
    strategy.__dict__.pop("_generated_repeat", None)
    strategy.__dict__.pop("_built_with_surrogate_generations", None)

    restored = pickle.loads(pickle.dumps(strategy))
    assert isinstance(restored.declaration, Pipeline)
    optimizer = _optimizer(restored).set_termination(
        Termination(lambda state: state.gen >= 1)
    )
    list(optimizer.iterate())


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize(
    "strategy_factory",
    [
        DirectStrategy,
        lambda: GenerationBasedStrategy(1),
        lambda: IndividualBasedStrategy(0.5),
        lambda: PreSelectionStrategy(5, 2),
        SteadyStateStrategy,
    ],
)
def test_same_class_ask_replacement_from_callback_is_observed(
    asynchronous, strategy_factory
):
    strategy = strategy_factory()
    optimizer = _optimizer(strategy, asynchronous=asynchronous).set_termination(
        Termination(lambda state: state.gen >= 6)
    )
    callbacks = CallbackManager()
    replaced = False

    def replace_after_ask(event: PostAskEvent) -> None:
        nonlocal replaced
        if replaced:
            return
        replaced = True
        pipeline = cast(Pipeline, strategy.pipeline)
        owner = (
            cast(Pipeline, pipeline.find("true_generation", recursive=True))
            if isinstance(strategy, GenerationBasedStrategy)
            else pipeline
        )
        original = cast(AskStage, owner.find("ask"))
        revision = strategy._pipeline_revision.value
        owner.replace("ask", AskStage(original._algorithm, n_offspring=7))
        assert strategy._pipeline_revision.value == revision + 1

    callbacks.register(PostAskEvent, replace_after_ask)
    optimizer.cbmanager = callbacks
    iterator = optimizer.iterate()
    next(iterator)
    initial_plan = optimizer.executable_plan
    next(iterator)
    assert replaced
    next(iterator)
    assert optimizer.executable_plan is not initial_plan
    declared_pipeline = (
        cast(Pipeline, strategy.declaration.find("true_generation", recursive=True))
        if isinstance(strategy, GenerationBasedStrategy)
        else strategy.declaration
    )
    assert cast(AskStage, declared_pipeline.find("ask"))._n_offspring == 7


@pytest.mark.parametrize(
    "strategy_factory",
    [DirectStrategy, SteadyStateStrategy, lambda: PreSelectionStrategy(5, 2)],
)
def test_renamed_ask_replacement_keeps_its_own_count(strategy_factory):
    strategy = strategy_factory()
    optimizer = _optimizer(strategy).set_termination(
        Termination(lambda state: state.gen >= 1)
    )
    optimizer.resolve_defaults()
    optimizer._compile_plan()
    calls = StageCallLog()
    replacement = RecordingAskStage(optimizer.algorithm, 7, calls)
    replacement.name = "renamed_ask"
    cast(Pipeline, strategy.pipeline).replace("ask", replacement)

    list(optimizer.iterate())

    assert calls.values == [1]
    assert strategy.declaration.find("renamed_ask") is replacement
    assert replacement._n_offspring == 7


def test_generation_based_renamed_repeat_replacement_is_kept():
    strategy = GenerationBasedStrategy(2)
    optimizer = _optimizer(strategy).set_termination(
        Termination(lambda state: state.gen >= 1)
    )
    optimizer.resolve_defaults()
    optimizer._compile_plan()
    pipeline = cast(Pipeline, strategy.pipeline)
    original = cast(Repeat, pipeline.find("surrogate_generations"))
    replacement = Repeat(original.body, count=0, name="renamed_repeat")
    pipeline.replace("surrogate_generations", replacement)

    states = list(optimizer.iterate())

    assert states[-1].gen == 1
    assert strategy.declaration.find("renamed_repeat") is replacement


def test_generation_based_renamed_stage_replacement_is_kept():
    class ReplacementLoop(RecordingCountGenerationStage):
        name = "replacement_loop"

    strategy = GenerationBasedStrategy(2)
    optimizer = _optimizer(strategy).set_termination(
        Termination(lambda state: state.gen >= 4)
    )
    optimizer.resolve_defaults()
    optimizer._compile_plan()
    calls = StageCallLog()
    replacement = ReplacementLoop(calls)
    cast(Pipeline, strategy.pipeline).replace("surrogate_generations", replacement)

    iterator = optimizer.iterate()
    next(iterator)
    first = next(iterator)

    assert first.gen == 2
    assert calls.values == [0]
    assert strategy.declaration.find("replacement_loop") is replacement


def test_generation_based_toggle_preserves_true_flow_replacement():
    strategy = GenerationBasedStrategy(1)
    optimizer = _optimizer(strategy).set_termination(
        Termination(lambda state: state.gen >= 6)
    )
    optimizer.resolve_defaults()
    optimizer._compile_plan()
    pipeline = cast(Pipeline, strategy.pipeline)
    true_generation = cast(Pipeline, pipeline.find("true_generation"))
    calls = StageCallLog()
    original = cast(AskStage, true_generation.find("ask"))
    replacement = RecordingAskStage(original._algorithm, original._n_offspring, calls)
    replacement.name = "renamed_true_ask"
    true_generation.replace("ask", replacement)

    iterator = optimizer.iterate()
    next(iterator)
    first_gen = next(iterator).gen
    strategy.gen_ctrl = 0
    second_gen = next(iterator).gen
    strategy.gen_ctrl = 1
    third_gen = next(iterator).gen

    assert (first_gen, second_gen, third_gen) == (2, 3, 5)
    assert calls.values == [2, 3, 5]
    assert strategy.declaration.find("renamed_true_ask", recursive=True) is replacement


def test_strategy_pipeline_replacement_unknown_name_raises_key_error():
    strategy = DirectStrategy()
    optimizer = _optimizer(strategy)
    optimizer.resolve_defaults()
    pipeline = strategy.build_pipeline(optimizer)

    with pytest.raises(KeyError, match="missing"):
        pipeline.replace("missing", CountGenerationStage())


def test_async_pipeline_replacement_waits_for_pending_evaluations():
    evaluator = _ControlledEvaluator()
    strategy = SteadyStateStrategy()
    optimizer = _optimizer(
        strategy, asynchronous=True, evaluator=evaluator, max_pending=2
    ).set_termination(Termination(lambda state: state.fe >= 50))
    optimizer.resolve_defaults()
    optimizer._compile_plan()
    initializer = optimizer.initializer
    assert initializer is not None
    state = initializer.initialize(optimizer, optimizer.problem)
    runtime = create_runtime(optimizer)
    session = runtime.initialize(resolve_plan(optimizer), state)
    assert session is not None

    for _ in range(10):
        if session.state.pending_evaluations:
            break
        session = runtime.advance(session).session
        assert session is not None
    assert session.state.pending_evaluations
    initial_plan = optimizer.executable_plan
    pipeline = cast(Pipeline, strategy.pipeline)
    calls = StageCallLog()
    pipeline.replace("count_generation", RecordingCountGenerationStage(calls))

    for _ in range(3):
        session = runtime.advance(session).session
        assert session is not None
        assert optimizer.executable_plan is initial_plan
        assert calls.values == []

    evaluator.release_all = True
    for _ in range(1000):
        if optimizer.executable_plan is not initial_plan and calls.values:
            break
        session = runtime.advance(session).session
        assert session is not None

    assert calls.values
    assert isinstance(
        cast(Pipeline, strategy.pipeline).find("count_generation"),
        RecordingCountGenerationStage,
    )


@pytest.mark.parametrize(
    ("first_factory", "second_factory"),
    [
        (lambda: IndividualBasedStrategy(0.5), DirectStrategy),
        (lambda: PreSelectionStrategy(5, 2), lambda: GenerationBasedStrategy(0)),
        (DirectStrategy, lambda: IndividualBasedStrategy(0.5)),
        (lambda: GenerationBasedStrategy(0), lambda: PreSelectionStrategy(5, 2)),
        (SteadyStateStrategy, DirectStrategy),
        (DirectStrategy, SteadyStateStrategy),
        (SteadyStateStrategy, lambda: GenerationBasedStrategy(0)),
        (lambda: GenerationBasedStrategy(0), SteadyStateStrategy),
        (SteadyStateStrategy, lambda: IndividualBasedStrategy(0.5)),
        (lambda: IndividualBasedStrategy(0.5), SteadyStateStrategy),
        (SteadyStateStrategy, lambda: PreSelectionStrategy(5, 2)),
        (lambda: PreSelectionStrategy(5, 2), SteadyStateStrategy),
    ],
)
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("use_setter", [False, True])
def test_strategy_replacement_from_callback_keeps_running(
    first_factory, second_factory, asynchronous, use_setter
):
    first = first_factory()
    second = second_factory()
    optimizer = _optimizer(first, asynchronous=asynchronous).set_history(
        ["summary", "evaluation"]
    )
    optimizer.set_termination(Termination(_four_generations))
    replaced: list[int] = []
    fe_at_replacement: list[int] = []
    completed_after_switch: list[int] = []
    callbacks = CallbackManager()

    def replace_strategy(event):
        if not replaced and event.ctx.gen >= 1:
            replaced.append(event.ctx.gen)
            fe_at_replacement.append(event.ctx.fe)
            if use_setter:
                optimizer.set_strategy(second)
            else:
                optimizer.strategy = second

    def record_replacement_flow(event):
        if replaced and optimizer.strategy is second and event.ctx.gen > replaced[0]:
            completed_after_switch.append(event.ctx.gen)

    callbacks.register(GenerationEndEvent, replace_strategy)
    callbacks.register(GenerationEndEvent, record_replacement_flow)
    optimizer.cbmanager = callbacks

    state = optimizer.run()

    assert replaced
    assert state.gen == 4
    assert cast(Any, second).pipeline is not None
    assert completed_after_switch
    assert state.fe > fe_at_replacement[0]
    _assert_history_matches_evaluations(state)


@pytest.mark.parametrize(
    "strategy_factory",
    [
        lambda: IndividualBasedStrategy(0.5),
        lambda: PreSelectionStrategy(5, 2),
        SteadyStateStrategy,
    ],
)
def test_template_strategy_build_graph_returns_structured_graph(strategy_factory):
    strategy = strategy_factory()
    optimizer = _optimizer(strategy)
    optimizer.resolve_defaults()

    graph = cast(Any, strategy).build_graph(optimizer)

    assert isinstance(graph, StructuredGraph)
    if isinstance(strategy, SteadyStateStrategy):
        expected = (
            "count_generation",
            "ask",
            "evaluation_plan",
            "evaluation_submit",
            "evaluation_collect",
            "evaluation_apply",
            "archive_update",
            "feedback",
            "tell",
            "evaluation_acknowledge",
        )
        node_ids = {node.component_id for node in graph.nodes}
        assert set(expected) <= node_ids
        pipeline = strategy.pipeline
        assert pipeline is not None
        assert tuple(step.name for step in pipeline.steps) == expected
        assert all(pipeline.find(name) is not None for name in expected)
        optimizer.run()
        prepared = strategy.pipeline
        assert prepared is not None
        assert all(prepared.find(name) is not None for name in expected)


@pytest.mark.parametrize(
    ("first_factory", "second_factory"),
    [
        (lambda: IndividualBasedStrategy(0.5), DirectStrategy),
        (lambda: PreSelectionStrategy(5, 2), lambda: GenerationBasedStrategy(0)),
        (DirectStrategy, lambda: IndividualBasedStrategy(0.5)),
        (lambda: GenerationBasedStrategy(0), lambda: PreSelectionStrategy(5, 2)),
        (SteadyStateStrategy, DirectStrategy),
        (DirectStrategy, SteadyStateStrategy),
        (SteadyStateStrategy, lambda: GenerationBasedStrategy(0)),
        (lambda: GenerationBasedStrategy(0), SteadyStateStrategy),
        (SteadyStateStrategy, lambda: IndividualBasedStrategy(0.5)),
        (lambda: IndividualBasedStrategy(0.5), SteadyStateStrategy),
        (SteadyStateStrategy, lambda: PreSelectionStrategy(5, 2)),
        (lambda: PreSelectionStrategy(5, 2), SteadyStateStrategy),
    ],
)
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("use_setter", [False, True])
def test_strategy_replacement_between_iteration_yields_runs_replacement_flow(
    first_factory, second_factory, asynchronous, use_setter
):
    first = first_factory()
    second = second_factory()
    optimizer = _optimizer(first, asynchronous=asynchronous).set_history(
        ["summary", "evaluation"]
    )
    optimizer.set_termination(Termination(_four_generations))
    iterator = optimizer.iterate()
    next(iterator)
    observed = next(iterator)
    generation_at_switch = observed.gen
    fe_at_switch = observed.fe
    if use_setter:
        optimizer.set_strategy(second)
    else:
        optimizer.strategy = second

    states = list(iterator)
    final = states[-1]

    assert generation_at_switch >= 1
    assert final.gen == 4
    assert final.gen > generation_at_switch
    assert final.fe > fe_at_switch
    _assert_history_matches_evaluations(final, final=False)
    resolved = cast(Any, optimizer.strategy).pipeline
    assert resolved is not None
    if isinstance(second, GenerationBasedStrategy):
        assert resolved["true_generation"].find("ask") is not None
    else:
        assert resolved.find("ask") is not None


@pytest.mark.parametrize("checkpoint_format", ["npz", "pickle"])
@pytest.mark.parametrize(
    "strategy_factory",
    [
        lambda: PreSelectionStrategy(5, 2),
        lambda: IndividualBasedStrategy(0.5),
        SteadyStateStrategy,
    ],
)
def test_sync_checkpoint_resume_matches_uninterrupted_run(
    tmp_path, checkpoint_format: str, strategy_factory
):
    history_channels = ["summary", "population", "evaluation"]
    partial = _optimizer(strategy_factory()).set_history(history_channels)
    partial.set_termination(Termination(_one_generation))
    partial_state = partial.run()
    path = tmp_path / f"checkpoint.{checkpoint_format}"
    if checkpoint_format == "npz":
        partial_state.save(path)
        restored = OptimizationState.load(path, _problem())
        resumed_optimizer = _optimizer(strategy_factory()).set_history(history_channels)
        resumed = resumed_optimizer.run_from(restored)
    else:
        partial.save_pickle(partial_state, path)
        resumed_optimizer, restored = Optimizer.load_pickle(path)
        resumed_optimizer.set_termination(Termination(_two_generations))
        resumed = resumed_optimizer.run_from(restored)

    uninterrupted = _optimizer(strategy_factory()).set_history(history_channels)
    uninterrupted.set_termination(Termination(_two_generations))
    expected = uninterrupted.run()

    _assert_checkpoint_matches(resumed, expected)


@pytest.mark.parametrize("checkpoint_format", ["npz", "pickle"])
@pytest.mark.parametrize(
    "strategy_factory",
    [
        lambda: IndividualBasedStrategy(0.5),
        lambda: PreSelectionStrategy(5, 2),
        SteadyStateStrategy,
    ],
)
def test_async_checkpoint_resume_matches_uninterrupted_run(
    tmp_path, checkpoint_format: str, strategy_factory
):
    history_channels = ["summary", "population", "evaluation"]
    partial = _optimizer(strategy_factory(), asynchronous=True).set_history(
        history_channels
    )
    partial.set_termination(Termination(_one_generation))
    partial_state = partial.run()
    path = tmp_path / f"async-checkpoint.{checkpoint_format}"
    if checkpoint_format == "npz":
        partial_state.save(path)
        restored = OptimizationState.load(path, _problem())
        resumed_optimizer = _optimizer(
            strategy_factory(), asynchronous=True
        ).set_history(history_channels)
    else:
        partial.save_pickle(partial_state, path)
        resumed_optimizer, restored = Optimizer.load_pickle(path)
    resumed_optimizer.set_termination(Termination(_two_generations))
    resumed = resumed_optimizer.run_from(restored)

    uninterrupted_optimizer = _optimizer(
        strategy_factory(), asynchronous=True
    ).set_history(history_channels)
    uninterrupted_optimizer.set_termination(Termination(_two_generations))
    expected = uninterrupted_optimizer.run()

    _assert_checkpoint_matches(resumed, expected)


@pytest.mark.parametrize("checkpoint_format", ["npz", "pickle"])
def test_steady_state_async_checkpoint_resume_with_requests_in_flight(
    tmp_path, checkpoint_format: str
):
    history_channels = ["summary", "population", "evaluation"]
    checkpoint_path = tmp_path / f"steady-state.{checkpoint_format}"

    def configured_optimizer():
        optimizer = _optimizer(
            SteadyStateStrategy(), asynchronous=True, max_pending=3
        ).set_history(history_channels)
        optimizer.set_evaluation_planner(RepeatedEvaluation(replicates=3))
        optimizer.set_termination(Termination(_twelve_evaluations))
        return optimizer

    partial = configured_optimizer()
    partial.resolve_defaults()
    partial._compile_plan()
    state = partial.initializer.initialize(partial, partial.problem)
    runtime = create_runtime(partial)
    session = runtime.initialize(resolve_plan(partial), state)
    for _ in range(100):
        if session.state.pending_evaluations:
            break
        step = runtime.advance(session)
        assert step.session is not None
        session = step.session
    assert session.state.pending_evaluations
    assert len(session.state.evaluation_handles) == len(
        session.state.pending_evaluations
    )
    in_flight_request_ids = set(session.state.pending_evaluations)
    saved = session.state
    if checkpoint_format == "npz":
        saved.save(checkpoint_path)
    else:
        partial.save_pickle(saved, checkpoint_path)

    for _ in range(10000):
        if session.finished:
            break
        step = runtime.advance(session)
        assert step.session is not None
        session = step.session
    assert session.finished
    expected = session.state

    if checkpoint_format == "npz":
        resumed_optimizer = configured_optimizer()
        restored = OptimizationState.load(checkpoint_path, _problem())
    else:
        resumed_optimizer, restored = Optimizer.load_pickle(checkpoint_path)
    actual = resumed_optimizer.run_from(restored)

    _assert_checkpoint_matches(
        actual,
        expected,
        in_flight_request_ids=in_flight_request_ids,
        portable_checkpoint=checkpoint_format == "npz",
    )


def test_steady_state_partial_feedback_accumulator_pickle_resume(tmp_path):
    history_channels = ["summary", "population", "evaluation"]

    def configured_optimizer():
        optimizer = _optimizer(
            SteadyStateStrategy(), asynchronous=True, max_pending=3
        ).set_history(history_channels)
        optimizer.set_evaluation_planner(RepeatedEvaluation(replicates=3))
        optimizer.set_termination(Termination(_twelve_evaluations))
        return optimizer

    partial = configured_optimizer()
    partial.resolve_defaults()
    partial._compile_plan()
    state = partial.initializer.initialize(partial, partial.problem)
    runtime = create_runtime(partial)
    session = runtime.initialize(resolve_plan(partial), state)
    for _ in range(1000):
        accumulator = session.state.feedback_accumulator
        if (
            session.state.pending_evaluations
            and accumulator
            and any(buffer.get("seen") for buffer in accumulator.get("buffers", ()))
        ):
            break
        step = runtime.advance(session)
        assert step.session is not None
        session = step.session
    else:
        pytest.fail("partial repeated-evaluation feedback was not observed")

    in_flight_request_ids = set(session.state.pending_evaluations)
    assert len(in_flight_request_ids) == 3
    checkpoint = tmp_path / "steady-partial.pkl"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        partial.save_pickle(session.state, checkpoint)
        resumed_optimizer, restored = Optimizer.load_pickle(checkpoint)
    resumed = resumed_optimizer.run_from(restored)

    expected = configured_optimizer().run()
    _assert_checkpoint_matches(
        resumed,
        expected,
        in_flight_request_ids=in_flight_request_ids,
        portable_checkpoint=False,
    )


def _archive_x_f(state: Any) -> tuple[np.ndarray, np.ndarray]:
    archive = state.archive
    xs = np.asarray(archive.get_array("x"))
    fs = np.asarray(archive.get_array("f"))
    values = fs if fs.ndim == 1 else fs[:, 0]
    order = np.lexsort((*xs[:, ::-1].T, values))
    return values[order], xs[order]


def _assert_same_run(actual: Any, expected: Any) -> None:
    assert (actual.gen, actual.fe) == (expected.gen, expected.fe)
    actual_x, actual_f = _archive_x_f(actual)
    expected_x, expected_f = _archive_x_f(expected)
    np.testing.assert_array_equal(actual_x, expected_x)
    np.testing.assert_array_equal(actual_f, expected_f)


@pytest.mark.parametrize(
    ("shorthand", "declaration", "planner", "builder"),
    [
        (
            IndividualBasedStrategy(0.3),
            build_ib_template(),
            RatioEvaluation(0.3, sanitize_nonfinite=True),
            MixedFeedback(),
        ),
        (
            PreSelectionStrategy(5, 2),
            build_ps_template(5),
            TopKEvaluation(min(2, 5), sanitize_nonfinite=True),
            TrueOnlyFeedback(),
        ),
    ],
)
def test_shorthand_matches_template_with_explicit_policy_components(
    shorthand, declaration, planner, builder
):
    shorthand_optimizer = _optimizer(shorthand)
    shorthand_state = shorthand_optimizer.run()

    template_optimizer = _optimizer(TemplateStrategy(declaration))
    template_optimizer.set_evaluation_planner(planner)
    template_optimizer.set_feedback_builder(builder)
    template_state = template_optimizer.run()

    _assert_same_run(template_state, shorthand_state)


def test_explicit_policy_components_win_over_the_shorthand_default() -> None:
    planner = TopKEvaluation(3, sanitize_nonfinite=True)
    shorthand_optimizer = _optimizer(IndividualBasedStrategy(0.3))
    shorthand_optimizer.set_evaluation_planner(planner)
    shorthand_state = shorthand_optimizer.run()

    template_optimizer = _optimizer(TemplateStrategy(build_ib_template()))
    template_optimizer.set_evaluation_planner(
        TopKEvaluation(3, sanitize_nonfinite=True)
    )
    template_state = template_optimizer.run()

    _assert_same_run(template_state, shorthand_state)
    origin = shorthand_optimizer._component_origins["evaluation_planner"]
    assert origin.source == "setter"


def test_in_place_shorthand_change_does_not_affect_explicit_components() -> None:
    strategy = IndividualBasedStrategy(0.3)
    optimizer = _optimizer(strategy)
    optimizer.set_evaluation_planner(TopKEvaluation(3, sanitize_nonfinite=True))
    optimizer.run()
    resolved = cast(Any, strategy.pipeline)
    explicit = resolved.find("evaluation_plan")._planner
    assert explicit is optimizer.evaluation_planner
    assert not isinstance(explicit, RatioEvaluation)
