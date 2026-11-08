"""Pre-selection surrogate-assisted optimization strategy."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from saealib.pipeline import ComponentRef, Pipeline
from saealib.policies.evaluation import TopKEvaluation
from saealib.policies.feedback import TrueOnlyFeedback
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


def build_ps_template(n_candidates: int) -> Pipeline:
    """Build the pre-selection flow declaration for *n_candidates*."""
    steps = [
        CountGenerationStage(),
        AskStage(ComponentRef("proposer"), n_offspring=n_candidates),
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
    return Pipeline(name="pre_selection", steps=steps)


@register()
class PreSelectionStrategy(TemplateStrategy):
    """Generate a candidate pool and truly evaluate its top-scoring members.

    Parameters
    ----------
    n_candidates : int
        Number of candidates to generate and score with the surrogate.
    n_select : int
        Number of top-scoring candidates to evaluate with the true objective.
    """

    requires_surrogate: bool = True
    feedback_builder = TrueOnlyFeedback()
    _node_namespace = ""

    def __init__(self, n_candidates: int, n_select: int) -> None:
        self.n_candidates = n_candidates
        self.n_select = n_select
        super().__init__(build_ps_template(n_candidates))
        self._generated_ask = self.declaration.find("ask")

    def _restore_missing_declaration(self) -> Pipeline:
        return build_ps_template(int(getattr(self, "n_candidates")))

    def _restore_generated_node_references(self, declaration: Pipeline) -> None:
        self.__dict__.setdefault("_generated_ask", declaration.find("ask"))

    @property
    def evaluation_planner(self) -> TopKEvaluation:
        """Return the top-k planner for the current candidate counts."""
        return TopKEvaluation(
            min(self.n_select, self.n_candidates), sanitize_nonfinite=True
        )

    def _default_components(self) -> Mapping[str, object]:
        """Return policy components used when their Optimizer slots are empty."""
        return {
            "evaluation_planner": self.evaluation_planner,
            "feedback_builder": self.feedback_builder,
        }

    def build_template(self) -> Pipeline:
        """Return the declaration with the current candidate count."""
        if isinstance(self._generated_ask, AskStage):
            self._generated_ask._set_n_offspring(self.n_candidates)
        return self.declaration

    def step(
        self, ctx: OptimizationState, provider: ComponentProvider
    ) -> OptimizationState:
        """Generate a candidate pool, then evaluate its selected members truly.

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
