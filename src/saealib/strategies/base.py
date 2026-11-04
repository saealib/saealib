"""Base class for optimization strategies."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from saealib.context import OptimizationState
from saealib.core.compiler.graph import ComponentGraph
from saealib.core.contracts import ComponentContract, PortContract, StateContract
from saealib.core.state import PENDING_EVALUATIONS
from saealib.execution.prepare import BINDING_SLOTS
from saealib.optimizer import ComponentProvider
from saealib.pipeline import Pipeline
from saealib.policies.evaluation import EvaluateAll, EvaluationPlanner
from saealib.policies.feedback import FeedbackBuilder, MixedFeedback

if TYPE_CHECKING:
    from saealib.pipeline import PipelineEntry


def build_runtime_neutral_graph(
    strategy: OptimizationStrategy, provider: ComponentProvider
) -> ComponentGraph:
    """Build the strategy's one canonical graph topology.

    Runtime selection changes execution policy, never graph construction.
    Built-in strategies materialize their graph directly at this boundary;
    ``build_pipeline`` is reconstructed from this graph only for compatibility
    callers.
    """
    if type(strategy).build_graph is not OptimizationStrategy.build_graph:
        return strategy.build_graph(provider)
    raise TypeError("strategy must implement build_graph")


def build_pipeline_from_graph(graph: ComponentGraph) -> Pipeline:
    """Recover a Pipeline facade from a stage-based strategy graph."""
    from saealib.core.graph_builder import StageNodeAdapter

    stages = [
        node.component.stage
        for node in graph.nodes
        if isinstance(node.component, StageNodeAdapter)
    ]
    return Pipeline(stages)


class OptimizationStrategy:
    """Base class for optimization strategies.

    Strategies expose one canonical :class:`ComponentGraph` topology.  A
    strategy may describe it directly through ``build_pipeline``; the default
    implementation recovers a facade for stage-based graphs.
    """

    # Optimizer.validate() checks this to ensure surrogate_manager is configured.
    requires_surrogate: bool = False

    # Async runtimes may refill a strategy while an earlier proposal is still
    # waiting for complete feedback.  Most strategies need a generation
    # boundary first; steady-state strategies explicitly opt into overlap.
    supports_async_refill: bool = False

    evaluation_planner: EvaluationPlanner = EvaluateAll()

    feedback_builder: FeedbackBuilder = MixedFeedback()

    def contract(self) -> ComponentContract:
        """Return the strategy contract."""
        return ComponentContract(
            ports={"strategy": PortContract()},
            state=StateContract(
                reads=(PENDING_EVALUATIONS,),
            ),
        )

    def build_pipeline(self, provider: ComponentProvider) -> Pipeline:
        """Return the strategy's high-level Pipeline description."""
        return build_pipeline_from_graph(self.build_graph(provider))

    def build_graph(self, provider: ComponentProvider) -> ComponentGraph:
        """Build the strategy's canonical graph.

        Built-in strategies provide node/adapter specs to the canonical graph
        builder; graph-only extensions override this method and need not
        implement a Pipeline builder.
        """
        return build_runtime_neutral_graph(self, provider)

    def step(
        self, ctx: OptimizationState, provider: ComponentProvider
    ) -> OptimizationState:
        """
        Perform one generation step: generate, score, evaluate, and update.

        Parameters
        ----------
        ctx : OptimizationState
            Current optimization context.
        provider : ComponentProvider
            Component provider.

        Returns
        -------
        OptimizationState
            Updated state returned by the pipeline.
        """
        from saealib.execution.runtime import execute_strategy_step

        return execute_strategy_step(self, ctx, provider)


class TemplateStrategy(OptimizationStrategy):
    """
    Run a declared Pipeline whose Stages reference components by ``(slot, role)``.

    The declaration is a shareable value: it holds no components, only
    :class:`~saealib.pipeline.ComponentRef` references (or explicit
    Stage-local components).  Each preparation resolves the references from
    one binding snapshot and rebuilds every Stage, so the same strategy —
    even the same instance — can be prepared again for another Optimizer or
    a new binding version without the earlier plan's components changing.

    Parameters
    ----------
    pipeline : Pipeline
        Declaration to resolve and run.
    requires_surrogate : bool, optional
        Whether the declaration needs a surrogate manager.

    Attributes
    ----------
    declaration : Pipeline
        The declaration passed as *pipeline*.
    pipeline : Pipeline or None
        The declaration resolved for the most recent plan; ``None`` until a
        plan is prepared.
    """

    def __init__(
        self,
        pipeline: Pipeline,
        *,
        requires_surrogate: bool = False,
    ) -> None:
        self.declaration = pipeline
        self.requires_surrogate = requires_surrogate
        self.pipeline: Pipeline | None = None

    def build_pipeline(self, provider: ComponentProvider) -> Pipeline:
        """Return the declaration resolved against the active snapshot."""
        resolved = self._resolve(provider)
        declared = resolved if isinstance(resolved, Pipeline) else Pipeline([resolved])
        self.pipeline = declared
        return declared

    def build_graph(self, provider: ComponentProvider) -> ComponentGraph:
        """Lower the declaration resolved against the active snapshot."""
        from saealib.core.compiler.lowerer import lower_pipeline
        from saealib.execution.prepare import declared_components

        return lower_pipeline(declared_components(self.build_pipeline(provider)))

    def _resolve(self, provider: ComponentProvider) -> PipelineEntry:
        from saealib.execution.prepare import resolve_declaration

        return resolve_declaration(self.declaration, self.snapshot_bindings(provider))

    @staticmethod
    def snapshot_bindings(provider: ComponentProvider) -> dict[str, object]:
        """Read the slots a declaration resolves from.

        The runtime serves slots from the active binding snapshot; a bare
        provider resolves from its current slot values.
        """
        values = getattr(provider, "binding_values", None)
        if callable(values):
            return cast(dict[str, object], values())
        return {name: getattr(provider, name, None) for name in BINDING_SLOTS}
