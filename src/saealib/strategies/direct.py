"""Direct strategy expressed as a flow template.

The strategy's flow is a declaration whose Stages reference components by
``(slot, role)``; preparation resolves the references from the binding
snapshot of each plan, so Direct lowers to the same structured plan shape
as the other built-in strategies (a structured plan without regions).
"""

from __future__ import annotations

from collections.abc import Mapping

from saealib.exceptions import ValidationError
from saealib.pipeline import ComponentRef, Pipeline
from saealib.policies.evaluation import EvaluateAll
from saealib.policies.feedback import TrueOnlyFeedback
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
from saealib.strategies.base import TemplateStrategy


def build_direct_template(n_offspring: int | None = None) -> Pipeline:
    """Build the direct flow declaration for ``n_offspring`` per step.

    The declaration holds :class:`~saealib.pipeline.ComponentRef` references
    only; components are resolved per plan at preparation time.  Its Stages
    declare the optimization-state execution boundary: every Stage runs with
    the run's own state as ctx, so component code and callback handlers see
    the run state (callback events carry the state ctx, not the
    declared-reads transaction ctx).
    """
    steps = [
        CountGenerationStage(),
        AskStage(ComponentRef("proposer"), n_offspring=n_offspring),
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
    return Pipeline(name="direct", steps=steps)


@register()
class DirectStrategy(TemplateStrategy):
    """Plain EA strategy without surrogate scoring.

    Every candidate produced by :meth:`~saealib.algorithms.Algorithm.ask` is
    evaluated with the true objective function.  No surrogate manager is
    required.
    """

    requires_surrogate: bool = False

    evaluation_planner = EvaluateAll()
    feedback_builder = TrueOnlyFeedback()

    _node_namespace = ""

    def _default_components(self) -> Mapping[str, object]:
        return {
            "evaluation_planner": self.evaluation_planner,
            "feedback_builder": self.feedback_builder,
        }

    def __init__(self, n_offspring: int | None = None) -> None:
        if n_offspring is not None and n_offspring < 1:
            raise ValidationError("n_offspring must be positive")
        self.n_offspring = n_offspring
        super().__init__(build_direct_template(n_offspring))
        self._generated_ask = self.declaration.find("ask")

    def _restore_missing_declaration(self) -> Pipeline:
        return build_direct_template(getattr(self, "n_offspring", None))

    def _restore_generated_node_references(self, declaration: Pipeline) -> None:
        self.__dict__.setdefault("_generated_ask", declaration.find("ask"))

    def build_template(self) -> Pipeline:
        """Return the declaration with the current offspring count."""
        if isinstance(self._generated_ask, AskStage):
            self._generated_ask._set_n_offspring(self.n_offspring)
        return self.declaration


@register()
class SteadyStateStrategy(DirectStrategy):
    """Evaluate one candidate per step and refill asynchronous worker slots."""

    supports_async_refill: bool = True

    def __init__(self) -> None:
        super().__init__(n_offspring=1)

    def _restore_missing_declaration(self) -> Pipeline:
        return build_direct_template(1)

    def _restore_generated_node_references(self, declaration: Pipeline) -> None:
        self.__dict__.setdefault("_generated_ask", declaration.find("ask"))

    def build_template(self) -> Pipeline:
        """Keep the flow at one candidate per step."""
        if isinstance(self._generated_ask, AskStage):
            self._generated_ask._set_n_offspring(1)
        return self.declaration
