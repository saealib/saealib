"""Direct strategy expressed as a flow template.

The strategy's flow is a declaration whose Stages reference components by
``(slot, role)``; preparation resolves the references from the binding
snapshot of each plan, so Direct lowers to the same structured plan shape
as the other built-in strategies (a structured plan without regions).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from saealib.core.graph_builder import (
    NodeAdapterSpec,
    StageContractNodeAdapter,
    build_decomposed_component_graph_from_specs,
)
from saealib.exceptions import ValidationError
from saealib.pipeline import ComponentRef, Pipeline
from saealib.policies.evaluation import EvaluateAll, EvaluationPlanner
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
from saealib.strategies.base import TemplateStrategy

if TYPE_CHECKING:
    from typing import Any

    from saealib.core.compiler.graph import ComponentGraph
    from saealib.optimizer import ComponentProvider


def _provider_or(value: Any, provider: ComponentProvider, slot: str) -> Any:
    """Return the bound Optimizer slot, or ``value`` when the slot is empty.

    An explicitly bound Optimizer slot wins; an empty slot falls to the
    strategy instance's own component.
    """
    return getattr(provider, slot, None) or value


def _steady_state_component_graph(
    provider: ComponentProvider,
    *,
    evaluation_planner: EvaluationPlanner,
    feedback_builder: FeedbackBuilder,
) -> ComponentGraph:
    """Build the steady-state component graph as a sequential stage chain.

    The refill flow runs one stage after another through the sequential
    execution seam instead of lowering to a declarative structured plan.
    """
    cbmanager = getattr(provider, "cbmanager", None)
    evaluation_planner = _provider_or(
        evaluation_planner, provider, "evaluation_planner"
    )
    feedback_builder = _provider_or(feedback_builder, provider, "feedback_builder")
    stages = (
        CountGenerationStage(),
        AskStage(provider.algorithm, n_offspring=1, cbmanager=cbmanager),
        EvaluationPlanStage(evaluation_planner),
        EvaluationSubmitStage(provider.evaluator),
        EvaluationCollectStage(provider.evaluator),
        EvaluationApplyStage(),
        ArchiveUpdateStage(),
        FeedbackStage(feedback_builder),
        TellStage(provider.algorithm),
        EvaluationAcknowledgeStage(provider.evaluator, cbmanager),
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

    def build_template(self) -> Pipeline:
        """Rebuild the declaration from the current ``n_offspring``."""
        self.declaration = build_direct_template(self.n_offspring)
        return self.declaration


@register()
class SteadyStateStrategy(DirectStrategy):
    """Evaluate one candidate per step and refill asynchronous worker slots."""

    supports_async_refill: bool = True

    def __init__(self) -> None:
        super().__init__(n_offspring=1)

    def build_graph(self, provider: ComponentProvider) -> ComponentGraph:
        """Build the steady-state flow as a sequential stage chain."""
        return _steady_state_component_graph(
            provider,
            evaluation_planner=self.evaluation_planner,
            feedback_builder=self.feedback_builder,
        )

    def build_pipeline(self, provider: ComponentProvider) -> Pipeline:
        """Recover the pipeline facade from the sequential graph."""
        from saealib.strategies.base import build_pipeline_from_graph

        facade = build_pipeline_from_graph(self.build_graph(provider))
        self.pipeline = facade
        return facade
