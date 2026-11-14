"""Built-in and user flow templates run with and without an async scheduler."""

from __future__ import annotations

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
from saealib.callback import (
    CallbackManager,
    GenerationEndEvent,
    PostAskEvent,
    PostEvaluationEvent,
    RunEndEvent,
)
from saealib.core.compiler.lowerer import lower_pipeline
from saealib.core.contracts import ComponentContract, StateContract
from saealib.core.state import POPULATIONS_MAIN, RUNTIME_RNG
from saealib.execution import (
    AsyncEvaluationScheduler,
    EvaluationHandle,
    EvaluationRequest,
    EvaluationStatus,
    EvaluationUpdate,
    Evaluator,
    SerialEvaluator,
)
from saealib.pipeline import ComponentRef, Pipeline, Stage
from saealib.policies.feedback import ComparatorWorstFallback, TrueOnlyFeedback
from saealib.problem import Problem
from saealib.stages import (
    ArchiveUpdateStage,
    AskStage,
    CountGenerationStage,
    EvaluationAcknowledgeStage,
    EvaluationApplyStage,
    EvaluationCollectStage,
    EvaluationPlanStage,
    EvaluationSubmitStage,
    FeedbackStage,
    TellStage,
    stage_component,
)
from saealib.strategies import DirectStrategy, GenerationBasedStrategy
from saealib.strategies.base import OptimizationStrategy, TemplateStrategy
from saealib.surrogate.manager import LocalSurrogateManager
from saealib.surrogate.rbf import RBFSurrogate
from saealib.surrogate.rbf_kernels import GaussianKernel


class _FifoEvaluator(Evaluator):
    """Deliver each request once, in submission order, and count outstanding."""

    def __init__(self) -> None:
        self._delivered: set[int] = set()
        self.collect_calls = 0
        self.outstanding = 0
        self.max_outstanding = 0

    def evaluate_batch(self, x: Any, problem: Any) -> Any:
        return SerialEvaluator().evaluate_batch(x, problem)

    def submit(self, request: EvaluationRequest, problem: Any) -> EvaluationHandle:
        self.outstanding += 1
        self.max_outstanding = max(self.max_outstanding, self.outstanding)
        return EvaluationHandle(
            request.request_id,
            EvaluationStatus.PENDING,
            backend_token=(request, problem),
        )

    def collect(self, handle: EvaluationHandle, *, wait: bool = True) -> list:
        self.collect_calls += 1
        if self.collect_calls > 5000:
            raise RuntimeError("collect cap exceeded")
        request, problem = handle.backend_token
        request_id = int(request.request_id)
        if handle._acknowledged_sequence >= 0 or request_id in self._delivered:
            return []
        self._delivered.add(request_id)
        self.outstanding -= 1
        result = SerialEvaluator().evaluate_request(request, problem)
        result.candidate_ids = request.candidate_ids
        result.__post_init__()
        handle._delivered_sequence = 0
        return [
            EvaluationUpdate(
                request.request_id,
                EvaluationStatus.COMPLETED,
                request.candidate_ids,
                result,
                None,
                0,
            )
        ]

    def acknowledge(self, handle: EvaluationHandle, sequence: int) -> None:
        handle._acknowledged_sequence = sequence

    def supports_batch_rollback(self) -> bool:
        return True

    def can_reattach(self, pending: Any) -> bool:
        return True

    def reattach(self, pending: Any, problem: Any) -> EvaluationHandle:
        return EvaluationHandle(
            pending.request.request_id,
            EvaluationStatus.PENDING,
            backend_token=(pending.request, problem),
        )


class _CountingFeedback(TrueOnlyFeedback):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def build(self, *args: Any, **kwargs: Any) -> Any:
        self.calls += 1
        return super().build(*args, **kwargs)


class _MarkerStage(Stage):
    name = "marker"

    def contract(self) -> ComponentContract:
        return ComponentContract(
            state=StateContract(reads=(POPULATIONS_MAIN, RUNTIME_RNG), writes=())
        )

    def execute(self, state: Any) -> Any:
        return state


class _LoweredStrategy(OptimizationStrategy):
    """Region-less structured strategy lowered without component part nodes."""

    requires_surrogate = False

    def build_pipeline(self, provider: Any) -> Pipeline:
        return Pipeline(
            name="lowered",
            steps=[
                stage_component(CountGenerationStage()),
                stage_component(AskStage(provider.algorithm, n_offspring=2)),
                stage_component(EvaluationPlanStage(None)),
                stage_component(EvaluationSubmitStage(provider.evaluator)),
                stage_component(EvaluationCollectStage(provider.evaluator)),
                stage_component(EvaluationApplyStage()),
                stage_component(ArchiveUpdateStage()),
                stage_component(FeedbackStage(provider.feedback_builder)),
                stage_component(TellStage(provider.algorithm)),
                stage_component(EvaluationAcknowledgeStage(provider.evaluator)),
            ],
        )

    def build_graph(self, provider: Any) -> Any:
        return lower_pipeline(self.build_pipeline(provider))


def _user_template() -> TemplateStrategy:
    return TemplateStrategy(
        Pipeline(
            name="user_flow",
            steps=[
                CountGenerationStage(),
                AskStage(ComponentRef("proposer"), n_offspring=2),
                _MarkerStage(),
                EvaluationPlanStage(ComponentRef("evaluation_planner")),
                EvaluationSubmitStage(ComponentRef("evaluator")),
                EvaluationCollectStage(ComponentRef("evaluator")),
                EvaluationApplyStage(),
                ArchiveUpdateStage(),
                FeedbackStage(ComponentRef("feedback_builder")),
                TellStage(ComponentRef("feedback_consumer")),
                EvaluationAcknowledgeStage(ComponentRef("evaluator")),
            ],
        )
    )


def _problem() -> Problem:
    return Problem(
        func=lambda x: np.array([float(np.sum(np.asarray(x) ** 2))], dtype=np.float64),
        dim=2,
        n_obj=1,
        direction=np.array([-1.0]),
        lb=[-5.0] * 2,
        ub=[5.0] * 2,
    )


def _ga() -> GA:
    return GA(
        crossover=CrossoverBLXAlpha(prob=0.9, alpha=0.4),
        mutation=MutationUniform(prob_var=0.1),
        parent_selection=SequentialSelection(),
        survivor_selection=TruncationSelection(),
    )


def _optimizer(
    strategy: Any,
    termination: Any,
    *,
    max_pending: int | None = None,
    seed: int = 42,
) -> tuple[Optimizer, _FifoEvaluator]:
    evaluator = _FifoEvaluator()
    optimizer = (
        Optimizer(_problem(), seed=seed)
        .set_initializer(LHSInitializer(6, 4, seed))
        .set_strategy(strategy)
        .set_algorithm(_ga())
        .set_termination(Termination(termination))
        .set_evaluator(evaluator)
    )
    if strategy.requires_surrogate:
        optimizer.set_surrogate_manager(
            LocalSurrogateManager(RBFSurrogate(GaussianKernel()))
        ).set_acquisition(MeanPrediction())
    if max_pending is not None:
        optimizer.set_async_evaluation_scheduler(
            AsyncEvaluationScheduler(evaluator, max_pending=max_pending)
        )
    return optimizer, evaluator


def _record_events(optimizer: Optimizer) -> list[str]:
    events: list[str] = []
    callbacks = CallbackManager()
    callbacks.register(PostEvaluationEvent, lambda event: events.append("eval"))
    callbacks.register(GenerationEndEvent, lambda event: events.append("gen_end"))
    callbacks.register(RunEndEvent, lambda event: events.append("run_end"))
    optimizer.cbmanager = callbacks
    return events


def test_user_region_less_template_reaches_termination_asynchronously() -> None:
    optimizer, _ = _optimizer(
        _user_template(), lambda state: state.fe >= 8, max_pending=2
    )
    events = _record_events(optimizer)

    state = optimizer.run()

    assert (state.gen, state.fe) == (1, 8)
    assert events == ["eval", "eval", "gen_end", "run_end"]


def test_scheduler_accumulates_feedback_only_for_plans_with_an_accumulator() -> None:
    lowered, _ = _optimizer(
        _LoweredStrategy(), lambda state: state.fe >= 8, max_pending=2
    )
    direct, _ = _optimizer(
        DirectStrategy(), lambda state: state.gen >= 2, max_pending=2
    )
    generation_based, _ = _optimizer(
        GenerationBasedStrategy(1), lambda state: state.gen >= 3, max_pending=2
    )

    for optimizer in (lowered, direct, generation_based):
        optimizer.run()

    def accumulates(optimizer: Optimizer) -> bool:
        scheduler = cast(Any, optimizer.async_evaluation_scheduler)
        return scheduler._feedback_accumulator is not None

    assert not accumulates(lowered)
    assert accumulates(direct)
    assert not accumulates(generation_based)


@pytest.mark.parametrize(
    ("gen_ctrl", "generations", "expected"), [(0, 2, (2, 11)), (2, 6, (6, 11))]
)
def test_generation_based_async_submissions_respect_max_pending(
    gen_ctrl: int, generations: int, expected: tuple[int, int]
) -> None:
    optimizer, evaluator = _optimizer(
        GenerationBasedStrategy(gen_ctrl),
        lambda state: state.gen >= generations,
        max_pending=1,
    )

    state = optimizer.run()

    assert evaluator.max_outstanding == 1
    assert (state.gen, state.fe) == expected


@pytest.mark.parametrize("gen_ctrl", [0, 2])
def test_generation_based_true_feedback_uses_the_stage_builder_sync_and_async(
    gen_ctrl: int,
) -> None:
    calls: dict[bool, int] = {}
    for asynchronous in (False, True):
        strategy = GenerationBasedStrategy(gen_ctrl)
        builder = _CountingFeedback()
        strategy.true_feedback_builder = builder
        optimizer, _ = _optimizer(
            strategy,
            lambda state: state.gen >= 4,
            max_pending=2 if asynchronous else None,
        )

        optimizer.run()

        pipeline = cast(Any, strategy.pipeline)
        true_generation = pipeline.find("true_generation", recursive=True)
        assert true_generation.stages[7]._builder is builder
        calls[asynchronous] = builder.calls

    assert calls[False] > 0
    assert calls[True] == calls[False]


def test_bundled_feedback_default_does_not_reach_the_generation_based_true_generation():
    optimizer = Optimizer(_problem(), seed=1).set_termination(
        Termination(lambda state: state.gen >= 2)
    )
    optimizer.resolve_defaults()
    assert isinstance(optimizer.feedback_builder, ComparatorWorstFallback)
    assert not optimizer.feedback_builder_explicit

    strategy = GenerationBasedStrategy(1)
    optimizer.set_strategy(strategy)
    optimizer.resolve_defaults()
    assert not optimizer.feedback_builder_explicit
    pipeline = cast(Any, strategy.build_pipeline(optimizer))

    builder = pipeline.find("true_generation", recursive=True).stages[7]._builder
    assert builder is optimizer.feedback_builder
    assert isinstance(builder, TrueOnlyFeedback)
    resolution = optimizer._component_resolution
    assert resolution is not None
    assert resolution.origins["feedback_builder"].source == "strategy default"


@pytest.mark.parametrize("asynchronous", [False, True])
def test_direct_feedback_assignment_wins_after_generation_based_swap(
    asynchronous: bool,
) -> None:
    slot_builder = _CountingFeedback()
    true_builder = _CountingFeedback()
    generation_based = GenerationBasedStrategy(0)
    generation_based.true_feedback_builder = true_builder
    optimizer, _ = _optimizer(
        DirectStrategy(),
        lambda state: state.gen >= 6,
        max_pending=2 if asynchronous else None,
    )
    optimizer.feedback_builder = slot_builder
    optimizer.set_surrogate_manager(
        LocalSurrogateManager(RBFSurrogate(GaussianKernel()))
    ).set_acquisition(MeanPrediction())
    calls_at_swap: list[int] = []

    def swap(event: Any) -> None:
        if event.ctx.gen == 3:
            calls_at_swap.append(slot_builder.calls)
            optimizer.strategy = generation_based

    callbacks = CallbackManager()
    callbacks.register(GenerationEndEvent, swap)
    optimizer.cbmanager = callbacks

    state = optimizer.run()

    assert state.gen == 6
    assert calls_at_swap and calls_at_swap[0] > 0
    assert slot_builder.calls > calls_at_swap[0]
    assert true_builder.calls == 0
    resolution = optimizer._component_resolution
    assert resolution is not None
    assert resolution.origins["feedback_builder"].source == "direct assignment"


def test_algorithm_reassigned_after_ask_takes_over_from_the_next_generation() -> None:
    class _CountingGA(GA):
        def __init__(self) -> None:
            super().__init__(
                crossover=CrossoverBLXAlpha(prob=0.9, alpha=0.4),
                mutation=MutationUniform(prob_var=0.1),
                parent_selection=SequentialSelection(),
                survivor_selection=TruncationSelection(),
            )
            self.asks = 0

        def ask(self, *args: Any, **kwargs: Any) -> Any:
            self.asks += 1
            return super().ask(*args, **kwargs)

    first, second = _CountingGA(), _CountingGA()
    optimizer = (
        Optimizer(_problem(), seed=7)
        .set_initializer(LHSInitializer(6, 4, 7))
        .set_strategy(DirectStrategy(3))
        .set_termination(Termination(lambda state: state.gen >= 3))
        .set_algorithm(first)
    )
    asks = {"seen": 0}

    def on_ask(event: Any) -> None:
        asks["seen"] += 1
        if asks["seen"] == 2:
            optimizer.algorithm = second

    callbacks = CallbackManager()
    callbacks.register(PostAskEvent, on_ask)
    optimizer.cbmanager = callbacks

    state = optimizer.run()

    assert state.gen == 3
    assert (first.asks, second.asks) == (2, 1)


def _planned_state(optimizer: Optimizer) -> Any:
    from saealib.policies.evaluation import RepeatedEvaluation

    optimizer.resolve_defaults()
    initializer = optimizer.initializer
    assert initializer is not None
    state = initializer.initialize(optimizer, optimizer.problem)
    state = AskStage(optimizer.algorithm, n_offspring=1).execute(state)
    return EvaluationPlanStage(RepeatedEvaluation(3)).execute(state)


def test_submit_stage_submits_planned_requests_within_scheduler_capacity() -> None:
    optimizer, evaluator = _optimizer(DirectStrategy(), lambda state: state.gen >= 1)
    state = _planned_state(optimizer)
    request_ids = [int(item.request_id) for item in state.evaluation_plan.requests]
    scheduler = AsyncEvaluationScheduler(evaluator, max_pending=1)

    submitted = EvaluationSubmitStage(evaluator).execute_async(
        state, scheduler=scheduler
    )

    assert len(request_ids) == 3
    assert list(submitted.pending_evaluations) == request_ids[:1]
    plan_state = submitted.evaluation_plan_state
    assert plan_state is not None
    assert plan_state.submitted == tuple(request_ids[:1])
    assert plan_state.deferred == tuple(request_ids[1:])
    assert evaluator.outstanding == 1


def test_submit_stage_without_a_scheduler_submits_synchronously() -> None:
    optimizer, evaluator = _optimizer(DirectStrategy(), lambda state: state.gen >= 1)
    state = _planned_state(optimizer)
    stage = EvaluationSubmitStage(evaluator)

    submitted = stage.execute_async(state)
    unplanned = state.replace(evaluation_plan=None)

    assert set(submitted.evaluation_handles) == {
        int(item.request_id) for item in state.evaluation_plan.requests
    }
    assert stage.execute_async(unplanned, scheduler=object()) is unplanned


def test_generation_based_async_strategy_step_is_rejected() -> None:
    from saealib.exceptions import ValidationError

    strategy = GenerationBasedStrategy(1)
    optimizer, _ = _optimizer(strategy, lambda state: state.gen >= 2, max_pending=2)
    initializer = optimizer.initializer
    assert initializer is not None
    state = initializer.initialize(optimizer, optimizer.problem)

    with (
        pytest.warns(
            DeprecationWarning, match=r"execute_strategy_step.*Optimizer\.iterate"
        ),
        pytest.raises(
            ValidationError, match=r"^async strategy steps require a sequential graph"
        ),
    ):
        strategy.step(state, optimizer)


def test_steady_state_async_strategy_step_submits_one_candidate() -> None:
    from saealib.strategies import SteadyStateStrategy

    strategy = SteadyStateStrategy()
    optimizer, evaluator = _optimizer(
        strategy, lambda state: state.gen >= 2, max_pending=2
    )
    initializer = optimizer.initializer
    assert initializer is not None
    state = initializer.initialize(optimizer, optimizer.problem)

    with pytest.warns(
        DeprecationWarning, match=r"execute_strategy_step.*Optimizer\.iterate"
    ):
        stepped = strategy.step(state, optimizer)

    assert len(stepped.pending_evaluations) == 1
    assert evaluator.outstanding == 1
