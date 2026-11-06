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
from saealib.callback import CallbackManager, GenerationEndEvent
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
from saealib.policies.evaluation import RatioEvaluation, TopKEvaluation
from saealib.policies.feedback import MixedFeedback, TrueOnlyFeedback
from saealib.problem import Problem
from saealib.registry import build, to_spec
from saealib.strategies import DirectStrategy, GenerationBasedStrategy
from saealib.surrogate.manager import LocalSurrogateManager
from saealib.surrogate.rbf import RBFSurrogate
from saealib.surrogate.rbf_kernels import GaussianKernel


class _Evaluator(Evaluator):
    def __init__(self) -> None:
        self.delivered: dict[int, EvaluationUpdate] = {}

    def evaluate_batch(self, x, problem):
        return SerialEvaluator().evaluate_batch(x, problem)

    def submit(self, request: EvaluationRequest, problem: Problem):
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
        return EvaluationHandle(
            pending.request.request_id,
            EvaluationStatus.PENDING,
            backend_token=(pending.request, problem),
        )


def _sphere(x):
    return np.array([float(np.sum(np.asarray(x) ** 2))])


def _one_generation(state):
    return state.gen >= 1


def _two_generations(state):
    return state.gen >= 2


def _four_generations(state):
    return state.gen >= 4


def _assert_checkpoint_matches(actual, expected):
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
            np.testing.assert_array_equal(actual_columns[name], expected_values)


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


def _optimizer(strategy: Any, *, asynchronous: bool = False) -> Optimizer:
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
        evaluator = _Evaluator()
        optimizer.set_evaluator(evaluator)
        optimizer.set_async_evaluation_scheduler(
            AsyncEvaluationScheduler(evaluator, max_pending=1)
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


@pytest.mark.parametrize(
    "strategy", [IndividualBasedStrategy(0.4), PreSelectionStrategy(7, 3)]
)
def test_strategy_specs_round_trip(strategy):
    rebuilt = build(to_spec(strategy))
    assert type(rebuilt) is type(strategy)
    if isinstance(strategy, IndividualBasedStrategy):
        assert rebuilt.evaluation_ratio == strategy.evaluation_ratio
    else:
        assert rebuilt.n_candidates == strategy.n_candidates
        assert rebuilt.n_select == strategy.n_select


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize(
    "strategy_factory",
    [lambda: IndividualBasedStrategy(0.5), lambda: PreSelectionStrategy(7, 3)],
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
    ],
)
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("use_setter", [False, True])
def test_strategy_replacement_from_callback_keeps_running(
    first_factory, second_factory, asynchronous, use_setter
):
    first = first_factory()
    second = second_factory()
    optimizer = _optimizer(first, asynchronous=asynchronous)
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


@pytest.mark.parametrize(
    "strategy_factory",
    [lambda: IndividualBasedStrategy(0.5), lambda: PreSelectionStrategy(5, 2)],
)
def test_surrogate_strategy_build_graph_returns_structured_graph(strategy_factory):
    optimizer = _optimizer(strategy_factory())
    optimizer._resolve_defaults()

    graph = cast(Any, optimizer.strategy).build_graph(optimizer)

    assert isinstance(graph, StructuredGraph)


@pytest.mark.parametrize(
    ("first_factory", "second_factory"),
    [
        (lambda: IndividualBasedStrategy(0.5), DirectStrategy),
        (lambda: PreSelectionStrategy(5, 2), lambda: GenerationBasedStrategy(0)),
        (DirectStrategy, lambda: IndividualBasedStrategy(0.5)),
        (lambda: GenerationBasedStrategy(0), lambda: PreSelectionStrategy(5, 2)),
    ],
)
@pytest.mark.parametrize("asynchronous", [False, True])
def test_strategy_replacement_between_iteration_yields_runs_replacement_flow(
    first_factory, second_factory, asynchronous
):
    first = first_factory()
    second = second_factory()
    optimizer = _optimizer(first, asynchronous=asynchronous)
    optimizer.set_termination(Termination(_four_generations))
    iterator = optimizer.iterate()
    next(iterator)
    observed = next(iterator)
    generation_at_switch = observed.gen
    fe_at_switch = observed.fe
    optimizer.set_strategy(second)

    states = list(iterator)
    final = states[-1]

    assert generation_at_switch >= 1
    assert final.gen == 4
    assert final.gen > generation_at_switch
    assert final.fe > fe_at_switch
    resolved = cast(Any, optimizer.strategy).pipeline
    assert resolved is not None
    if isinstance(second, GenerationBasedStrategy):
        assert resolved["true_generation"].find("ask") is not None
    else:
        assert resolved.find("ask") is not None


@pytest.mark.parametrize("checkpoint_format", ["npz", "pickle"])
@pytest.mark.parametrize(
    "strategy_factory",
    [lambda: PreSelectionStrategy(5, 2), lambda: IndividualBasedStrategy(0.5)],
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
    [lambda: IndividualBasedStrategy(0.5), lambda: PreSelectionStrategy(5, 2)],
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
