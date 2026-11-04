"""Generation-based strategy expressed as a flow template.

Each call to ``step`` runs ``gen_ctrl`` surrogate-only generations followed
by one generation of true objective evaluations.  The strategy's flow is a
declaration whose Stages reference components by ``(slot, role)``; only the
inner-loop feedback builder stays a Stage-local explicit component, so it
never competes with the shared ``feedback_builder`` slot.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from saealib.core.contracts.observation import SURROGATE
from saealib.pipeline import ComponentRef, Pipeline, PipelineEntry, Repeat
from saealib.policies.feedback import (
    MixedFeedback,
    PredictedFeedback,
    TrueOnlyFeedback,
)
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
    SurrogateFitStage,
    SurrogatePredictStage,
    TellStage,
)
from saealib.strategies.base import TemplateStrategy

if TYPE_CHECKING:
    from saealib.context import OptimizationState
    from saealib.optimizer import ComponentProvider


def build_gb_template(gen_ctrl: int) -> Pipeline:
    """Build the generation-based flow declaration for ``gen_ctrl``.

    The declaration holds :class:`~saealib.pipeline.ComponentRef` references
    only.  The ``PredictedFeedback`` of the surrogate-only inner loop is a
    Stage-local explicit component; it never competes with the shared
    ``feedback_builder`` slot.
    """
    surrogate_generation = Pipeline(
        name="surrogate_generation",
        steps=[
            CountGenerationStage(),
            AskStage(ComponentRef("proposer")),
            SurrogatePredictStage(ComponentRef("predictor"), refit=False),
            AcquisitionStage(ComponentRef("acquisition")),
            FeedbackStage(PredictedFeedback()),
            TellStage(ComponentRef("feedback_consumer"), channel=SURROGATE),
        ],
    )
    true_generation = Pipeline(
        name="true_generation",
        steps=[
            CountGenerationStage(),
            AskStage(ComponentRef("proposer")),
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
    steps: list[PipelineEntry] = []
    if gen_ctrl:
        steps.extend(
            (
                SurrogateFitStage(ComponentRef("fitter")),
                Repeat(
                    surrogate_generation,
                    count=gen_ctrl,
                    name="surrogate_generations",
                ),
            )
        )
    steps.append(true_generation)
    return Pipeline(name="generation_based", steps=steps)


@register()
class GenerationBasedStrategy(TemplateStrategy):
    """Generation-based strategy.

    Parameters
    ----------
    gen_ctrl : int
        Number of surrogate-only generations executed inside each :meth:`step`
        call before one generation of true objective evaluation.
    """

    requires_surrogate: bool = True
    feedback_builder = MixedFeedback()
    true_feedback_builder = TrueOnlyFeedback()

    def _default_components(self) -> Mapping[str, object]:
        """Slot defaults when the Optimizer leaves them empty.

        The shared true-generation flow resolves ``feedback_builder`` from
        ``true_feedback_builder``; the strategy's ``feedback_builder`` stays
        a bundled class default and the inner loop's ``PredictedFeedback``
        remains a Stage-local explicit component.
        """
        return {
            "evaluation_planner": self.evaluation_planner,
            "feedback_builder": self.true_feedback_builder,
        }

    def __init__(self, gen_ctrl: int) -> None:
        self.gen_ctrl = gen_ctrl
        super().__init__(build_gb_template(gen_ctrl))

    def snapshot_bindings(self, provider: ComponentProvider) -> dict[str, object]:
        """Read the slots, keeping the true generation's feedback builder.

        The Optimizer's ``feedback_builder`` reaches the true generation only
        when it was configured explicitly (``feedback_builder_explicit``);
        otherwise ``true_feedback_builder`` builds the true feedback.
        """
        bindings = super().snapshot_bindings(provider)
        if not getattr(provider, "feedback_builder_explicit", False):
            bindings["feedback_builder"] = self.true_feedback_builder
        return bindings

    def build_template(self) -> Pipeline:
        """Rebuild the declaration from the current ``gen_ctrl``."""
        self.declaration = build_gb_template(self.gen_ctrl)
        return self.declaration

    def step(
        self, ctx: OptimizationState, provider: ComponentProvider
    ) -> OptimizationState:
        """Run ``gen_ctrl`` surrogate-only generations, then one true-evaluation step.

        Rebuilds the pipeline each call so component/parameter changes take
        effect immediately.

        Parameters
        ----------
        ctx : OptimizationState
            Current optimization context.
        provider : ComponentProvider
            Component provider.
        """
        return super().step(ctx, provider)
