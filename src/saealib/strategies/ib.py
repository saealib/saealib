"""Individual-based surrogate-assisted optimization strategy."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from saealib.pipeline import ComponentRef, Pipeline
from saealib.policies.evaluation import RatioEvaluation
from saealib.policies.feedback import MixedFeedback
from saealib.registry import register
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
    SurrogatePredictStage,
    TellStage,
)
from saealib.strategies.base import TemplateStrategy

if TYPE_CHECKING:
    from saealib.context import OptimizationState
    from saealib.optimizer import ComponentProvider


def build_ib_template() -> Pipeline:
    """Build the individual-based flow declaration."""
    steps = [
        CountGenerationStage(),
        AskStage(ComponentRef("proposer")),
        SurrogatePredictStage(ComponentRef("predictor")),
        AcquisitionStage(ComponentRef("acquisition")),
        EvaluationPlanStage(ComponentRef("evaluation_planner")),
        EvaluationSubmitStage(ComponentRef("evaluator")),
        EvaluationCollectStage(ComponentRef("evaluator")),
        EvaluationApplyStage(),
        ArchiveUpdateStage(),
        FeedbackStage(ComponentRef("feedback_builder")),
        TellStage(ComponentRef("feedback_consumer")),
        EvaluationAcknowledgeStage(ComponentRef("evaluator")),
    ]
    for stage in steps:
        stage.execution_boundary = "state"
    return Pipeline(name="individual_based", steps=steps)


@register()
class IndividualBasedStrategy(TemplateStrategy):
    """Individual-based strategy with surrogate scoring of every offspring.

    Parameters
    ----------
    evaluation_ratio : float, optional
        Fraction of offspring selected for true objective evaluation.
        Every offspring is passed to ``algorithm.tell()``; candidates without
        true evaluations use their surrogate-predicted objective values.

    Notes
    -----
    Use ``state.archive`` or ``state.pareto_archive`` when only truly
    evaluated solutions are needed. Local surrogate fitting size is
    configured on the :class:`~saealib.surrogate.manager.SurrogateManager`.
    """

    requires_surrogate: bool = True
    feedback_builder = MixedFeedback()
    _node_namespace = ""

    def __init__(self, evaluation_ratio: float = 0.1) -> None:
        self.evaluation_ratio = evaluation_ratio
        super().__init__(build_ib_template())

    def _restore_missing_declaration(self) -> Pipeline:
        return build_ib_template()

    def _restore_generated_node_references(self, declaration: Pipeline) -> None:
        """Individual-based declarations do not cache generated nodes."""
        return None

    @property
    def evaluation_planner(self) -> RatioEvaluation:
        """Return the ratio planner for the current evaluation ratio."""
        return RatioEvaluation(self.evaluation_ratio, sanitize_nonfinite=True)

    def _default_components(self) -> Mapping[str, object]:
        """Return policy components used when their Optimizer slots are empty."""
        return {
            "evaluation_planner": self.evaluation_planner,
            "feedback_builder": self.feedback_builder,
        }

    def build_template(self) -> Pipeline:
        """Return the declaration used to prepare the current flow."""
        return self.declaration

    def step(
        self, ctx: OptimizationState, provider: ComponentProvider
    ) -> OptimizationState:
        """Score each offspring, then evaluate the selected fraction truly.

        Parameters
        ----------
        ctx : OptimizationState
            Current optimization context.
        provider : ComponentProvider
            Component provider.

        Returns
        -------
        OptimizationState
            Updated state after one generation step.
        """
        return super().step(ctx, provider)
