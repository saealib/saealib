"""Direct strategy: plain EA without surrogate evaluation."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from saealib.core.compiler.graph import ComponentGraph
from saealib.core.graph_builder import (
    NodeAdapterSpec,
    StageContractNodeAdapter,
    build_decomposed_component_graph_from_specs,
)
from saealib.exceptions import ValidationError
from saealib.pipeline import Pipeline
from saealib.policies.evaluation import EvaluateAll
from saealib.policies.feedback import FeedbackBuilder, TrueOnlyFeedback
from saealib.registry import register
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
)
from saealib.strategies.base import (
    OptimizationStrategy,
)

if TYPE_CHECKING:
    from saealib.optimizer import ComponentProvider


class DirectStrategy(OptimizationStrategy):
    """Plain EA strategy without surrogate scoring.

    Every candidate produced by :meth:`~saealib.algorithms.Algorithm.ask` is
    evaluated with the true objective function.  No surrogate manager is
    required.
    """

    requires_surrogate: bool = False

    evaluation_planner = EvaluateAll()
    feedback_builder = TrueOnlyFeedback()

    def __init__(self, n_offspring: int | None = None) -> None:
        if n_offspring is not None and n_offspring < 1:
            raise ValidationError("n_offspring must be positive")
        self.n_offspring = n_offspring
        self.pipeline: Pipeline | None = None

    def build_graph(self, provider: ComponentProvider) -> ComponentGraph:
        """Build the canonical direct strategy graph."""
        cbmanager = getattr(provider, "cbmanager", None)
        evaluation_planner = (
            getattr(provider, "evaluation_planner", None) or self.evaluation_planner
        )
        feedback_builder = cast(
            FeedbackBuilder | None, getattr(provider, "feedback_builder", None)
        )
        if feedback_builder is None:
            feedback_builder = self.feedback_builder
        evaluation_tail = [
            EvaluationPlanStage(evaluation_planner),
            EvaluationSubmitStage(provider.evaluator),
            EvaluationCollectStage(provider.evaluator),
            EvaluationApplyStage(),
            ArchiveUpdateStage(),
            FeedbackStage(feedback_builder),
            TellStage(provider.algorithm),
            EvaluationAcknowledgeStage(provider.evaluator, cbmanager),
        ]
        stages = (
            CountGenerationStage(),
            AskStage(
                provider.algorithm,
                n_offspring=self.n_offspring,
                cbmanager=cbmanager,
            ),
            *evaluation_tail,
        )
        stage_ids = tuple(stage.name for stage in stages)
        specs = tuple(
            NodeAdapterSpec(
                component_id=stage_id,
                adapter=StageContractNodeAdapter(stage, node_path=stage_id),
            )
            for stage_id, stage in zip(stage_ids, stages)
        )
        return build_decomposed_component_graph_from_specs(specs)


@register()
class SteadyStateStrategy(DirectStrategy):
    """Evaluate one candidate per step and refill asynchronous worker slots."""

    supports_async_refill: bool = True

    def __init__(self) -> None:
        super().__init__(n_offspring=1)

    def build_graph(self, provider: ComponentProvider) -> ComponentGraph:
        """Build the graph view of the steady-state pipeline."""
        return super().build_graph(provider)
