"""Behavioral checks for surrogate-assisted flow declarations."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest

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
from saealib.execution import (
    AsyncEvaluationScheduler,
    EvaluationHandle,
    EvaluationRequest,
    EvaluationStatus,
    EvaluationUpdate,
    Evaluator,
    SerialEvaluator,
)
from saealib.execution.runtime import create_runtime, resolve_plan
from saealib.policies.evaluation import (
    EvaluateAll,
    RatioEvaluation,
    RepeatedEvaluation,
    TopKEvaluation,
)
from saealib.policies.feedback import MixedFeedback, TrueOnlyFeedback
from saealib.problem import Problem
from saealib.registry import build, to_spec
from saealib.strategies import (
    DirectStrategy,
    GenerationBasedStrategy,
    SteadyStateStrategy,
)
from saealib.surrogate.manager import LocalSurrogateManager
from saealib.surrogate.rbf import RBFSurrogate
from saealib.surrogate.rbf_kernels import GaussianKernel


class _Evaluator(Evaluator):
    def __init__(self) -> None:
        self.delivered: dict[int, EvaluationUpdate] = {}
        self.submitted: list[EvaluationRequest] = []
        self._delivered: set[int] = set()
        self._seen_in_pass: set[int] = set()
        self._released_in_pass = False

    def evaluate_batch(self, x, problem):
        return SerialEvaluator().evaluate_batch(x, problem)

    def submit(self, request: EvaluationRequest, problem: Problem):
        self.submitted.append(request)
        return EvaluationHandle(
            request.request_id,
            EvaluationStatus.PENDING,
            backend_token=(request, problem),
        )

    def collect(self, handle: EvaluationHandle, *, wait: bool = True):
        if handle._acknowledged_sequence >= 0:
            return []
        request, problem = handle.backend_token
        request_id = int(request.request_id)
        if request_id in self._seen_in_pass:
            self._seen_in_pass.clear()
            self._released_in_pass = False
        self._seen_in_pass.add(request_id)
        if self._released_in_pass:
            return []
        outstanding = [
            int(item.request_id)
            for item in self.submitted
            if int(item.request_id) not in self._delivered
        ]
        if not outstanding or request_id != min(outstanding):
            return []
        update = self.delivered.get(request_id)
        if update is None:
            result = SerialEvaluator().evaluate_request(request, problem)
            result.candidate_ids = request.candidate_ids
            result.__post_init__()
            update = EvaluationUpdate(
                request.request_id,
                EvaluationStatus.COMPLETED,
                request.candidate_ids,
                result,
                None,
                0,
            )
            self.delivered[request_id] = update
            self._delivered.add(request_id)
        self._released_in_pass = True
        handle._delivered_sequence = update.sequence
        return [update]

    def acknowledge(self, handle: EvaluationHandle, sequence: int) -> None:
        handle._acknowledged_sequence = sequence
        self.delivered.pop(int(handle.request_id), None)

    def supports_batch_rollback(self) -> bool:
        return True

    def can_reattach(self, pending: Any) -> bool:
        return True

    def reattach(self, pending: Any, problem: Problem):
        if pending.request not in self.submitted:
            self.submitted.append(pending.request)
        return EvaluationHandle(
            pending.request.request_id,
            EvaluationStatus.PENDING,
            backend_token=(pending.request, problem),
        )


class _ControlledEvaluator(_Evaluator):
    def __init__(self) -> None:
        super().__init__()
        self.released_request_ids: set[int] = set()
        self.release_all = False

    def collect(self, handle: EvaluationHandle, *, wait: bool = True):
        if handle._acknowledged_sequence >= 0:
            return []
        request, problem = handle.backend_token
        request_id = int(request.request_id)
        if request_id in self._delivered or not (
            self.release_all or request_id in self.released_request_ids
        ):
            return []
        result = SerialEvaluator().evaluate_request(request, problem)
        result.candidate_ids = request.candidate_ids
        result.__post_init__()
        update = EvaluationUpdate(
            request.request_id,
            EvaluationStatus.COMPLETED,
            request.candidate_ids,
            result,
            None,
            0,
        )
        self.delivered[request_id] = update
        self._delivered.add(request_id)
        handle._delivered_sequence = update.sequence
        return [update]


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


def _assert_history_matches_evaluations(state):
    assert state.history is not None
    assert state.history.is_enabled("summary")
    assert state.history.is_enabled("evaluation")
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

    def plan(self, candidates, acquisition, ctx):
        self.calls += 1
        return super().plan(candidates, acquisition, ctx)


class _RecordingFeedback(MixedFeedback):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def build(self, candidates, prediction, evaluation, evaluated_indices, ctx):
        self.calls += 1
        return super().build(candidates, prediction, evaluation, evaluated_indices, ctx)


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
    optimizer._resolve_defaults()

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
    _assert_history_matches_evaluations(final)
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
