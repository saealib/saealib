"""Base class for optimization strategies."""

from __future__ import annotations

import weakref
from collections.abc import Mapping
from typing import TYPE_CHECKING, cast

from saealib._deprecation import warn_deprecated
from saealib.context import OptimizationState
from saealib.core.compiler.graph import ComponentGraph
from saealib.core.compiler.regions import _structural_stages
from saealib.core.contracts import ComponentContract, PortContract, StateContract
from saealib.core.state import PENDING_EVALUATIONS
from saealib.execution.prepare import BINDING_SLOTS
from saealib.optimizer import ComponentProvider
from saealib.pipeline import Pipeline
from saealib.policies.evaluation import EvaluateAll, EvaluationPlanner
from saealib.policies.feedback import FeedbackBuilder, MixedFeedback

if TYPE_CHECKING:
    from saealib.pipeline import PipelineEntry


class _PipelineRevision:
    """Mutable revision shared by a strategy and its declaration."""

    def __init__(self, value: int = 0) -> None:
        self.value = value

    def __repr__(self) -> str:
        return f"_PipelineRevision({self.value})"


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


def _entry_uses_surrogate_slot(entry: object) -> bool:
    """Return whether a declaration entry reaches the surrogate manager slot."""
    from saealib.pipeline import Stage
    from saealib.surrogate.manager import SurrogateManager

    children = _structural_stages(entry)
    if children is not None:
        return any(_entry_uses_surrogate_slot(child) for child in children)
    if isinstance(entry, SurrogateManager):
        return True
    if not isinstance(entry, Stage):
        return False
    for _, reference in getattr(entry, "_refs", ()):
        if reference is not None and reference.slot == "surrogate_manager":
            return True
    for value in vars(entry).values():
        if isinstance(value, SurrogateManager):
            return True
        if (
            value is not entry
            and (isinstance(value, Stage) or _structural_stages(value) is not None)
            and _entry_uses_surrogate_slot(value)
        ):
            return True
    nested = getattr(entry, "stages", None)
    if isinstance(nested, (list, tuple)):
        return any(_entry_uses_surrogate_slot(child) for child in nested)
    return False


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

    # Optimizer.validate() checks this to ensure surrogate_manager is
    # configured.  It is a plain attribute: a strategy may declare it as a
    # class attribute, inherit it from a mixin, set it on an instance, or
    # restore it from a pickle or a spec.
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
        builder.  Writing a new strategy is done with
        :class:`TemplateStrategy` and a declared Pipeline; overriding this
        method is deprecated and emits a :class:`DeprecationWarning`.
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
        from saealib.execution.runtime import _execute_strategy_step

        if getattr(provider, "async_evaluation_scheduler", None) is not None:
            # GenerationBased, IndividualBased, and PreSelection forward here
            # through their own step() methods; the other built-ins inherit
            # this method directly.
            stacklevel = 3 if type(self).step is not OptimizationStrategy.step else 2
            warn_deprecated(
                "execute_strategy_step() with asynchronous evaluation",
                since="0.2.0",
                replacement="Optimizer.iterate() or Optimizer.run()",
                stacklevel=stacklevel,
            )
        return _execute_strategy_step(self, ctx, provider)


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
    requires_surrogate : bool or None, optional
        Surrogate requirement declared on the instance.  ``None`` (the
        default) leaves the class-level value in place.  For a subclass that
        keeps the standard ``build_graph``, the configuration diagnostics
        derive the requirement from the declared flow instead, and a declared
        value that contradicts the flow is reported and does not change the
        outcome.

    Attributes
    ----------
    declaration : Pipeline
        The declaration passed as *pipeline*.
    pipeline : Pipeline or None
        The declaration resolved for the most recent plan; ``None`` until a
        plan is prepared.
    requires_surrogate : bool or None
        Plain attribute holding the declared requirement; ``None`` when the
        strategy declares none.  The configuration diagnostics derive the
        effective requirement of a standard-lowering subclass from its
        declaration.
    """

    requires_surrogate: bool | None = None

    def __init__(
        self,
        pipeline: Pipeline,
        *,
        requires_surrogate: bool | None = None,
    ) -> None:
        self.declaration = pipeline
        if requires_surrogate is not None:
            self.requires_surrogate = requires_surrogate
        self.pipeline: Pipeline | None = None
        self._pipeline_revision = _PipelineRevision()
        setattr(pipeline, "_pipeline_revision", self._pipeline_revision)

    def __setstate__(self, state: Mapping[str, object]) -> None:
        self.__dict__.update(state)
        self.__dict__.setdefault("_pipeline_revision", 0)
        declaration = self.__dict__.get("declaration")
        if not isinstance(declaration, Pipeline):
            declaration = self._restore_missing_declaration()
            if isinstance(declaration, Pipeline):
                self.declaration = declaration
        declaration = self.__dict__.get("declaration")
        if isinstance(declaration, Pipeline):
            shared_revision = getattr(declaration, "_pipeline_revision", None)
            if not isinstance(shared_revision, _PipelineRevision):
                stored_revision = self.__dict__.get("_pipeline_revision", 0)
                if isinstance(stored_revision, _PipelineRevision):
                    shared_revision = stored_revision
                else:
                    shared_revision = _PipelineRevision(int(stored_revision))
                declaration._pipeline_revision = shared_revision
            self._pipeline_revision = shared_revision
        pipeline = self.__dict__.get("pipeline")
        if isinstance(declaration, Pipeline) and isinstance(pipeline, Pipeline):
            self._link_pipeline_edits(declaration, pipeline)
        if isinstance(declaration, Pipeline):
            self._restore_generated_node_references(declaration)

    def _restore_missing_declaration(self) -> Pipeline | None:
        """Rebuild the declaration when a state does not contain one."""
        return None

    def _restore_generated_node_references(self, declaration: Pipeline) -> None:
        """Restore caches pointing into *declaration* when they are absent."""

    def _derive_surrogate_requirement(self) -> bool | None:
        """Return the requirement derived from the declared flow.

        The walk covers every Stage of the declaration, including nested
        Repeat, Loop, and Branch regions, and is ``True`` when a Stage refers
        to the surrogate manager slot through a ``ComponentRef`` or holds a
        surrogate manager directly.  The result is ``None`` when the subclass
        replaces ``build_graph``; such a strategy keeps using its declared
        ``requires_surrogate`` attribute.  The comparison is method identity,
        the same check the ``build_graph`` deprecation uses.
        """
        if getattr(type(self), "build_graph", None) is not TemplateStrategy.build_graph:
            return None
        return _entry_uses_surrogate_slot(getattr(self, "declaration", None))

    def build_template(self) -> Pipeline:
        """Return the declaration pipeline prepared for this run.

        Parameterized strategies update their declaration here so in-place
        parameter changes take effect at the next commit point.
        """
        return self.declaration

    def build_pipeline(self, provider: ComponentProvider) -> Pipeline:
        """Return the declaration resolved against the active snapshot."""
        template = self.build_template()
        resolved = self._resolve(provider, template)
        declared = resolved if isinstance(resolved, Pipeline) else Pipeline([resolved])
        self.pipeline = declared
        self._link_pipeline_edits(template, declared)
        return declared

    def _link_pipeline_edits(self, template: object, facade: object) -> None:
        if isinstance(template, Pipeline) and isinstance(facade, Pipeline):
            setattr(template, "_pipeline_revision", self._pipeline_revision)

            def on_replace(name: str, stage: PipelineEntry) -> None:
                if template is not facade:
                    template.replace(name, stage)
                self._pipeline_revision.value += 1
                if self.pipeline is not None:
                    self._link_pipeline_edits(self.declaration, self.pipeline)

            facade._on_replace = on_replace

        template_children = _structural_stages(template)
        facade_children = _structural_stages(facade)
        if template_children is None or facade_children is None:
            return
        for template_child, facade_child in zip(template_children, facade_children):
            self._link_pipeline_edits(template_child, facade_child)

    #: Node-ID naming for the lowered declaration: ``None`` qualifies nested
    #: Stages under the declaration's own name; "" gives them their bare
    #: Stage names as node IDs (``ask``, ``tell``, ...).
    _node_namespace: str | None = None

    def build_graph(self, provider: ComponentProvider) -> ComponentGraph:
        """Lower the declaration resolved against the active snapshot.

        The components each Stage holds become ``(component, role)`` part
        nodes with data edges between their declared ports, so the
        compiler's contract rules apply to them; the Stages stay the
        executable nodes.
        """
        from saealib.core.compiler.lowerer import lower_pipeline
        from saealib.core.graph_builder import decompose_structured_graph
        from saealib.execution.prepare import declared_components

        return decompose_structured_graph(
            lower_pipeline(
                declared_components(self.build_pipeline(provider)),
                namespace=self._node_namespace,
            )
        )

    def _resolve(
        self, provider: ComponentProvider, declaration: PipelineEntry
    ) -> PipelineEntry:
        from saealib.execution.prepare import _resolve_declaration

        selections = getattr(provider, "_stage_selections", None)
        return _resolve_declaration(
            declaration, self.snapshot_bindings(provider), selections
        )

    def _default_components(self) -> Mapping[str, object]:
        """Components the strategy instance provides for empty slots.

        An explicitly bound Optimizer slot always wins; these values cover
        the slots the Optimizer deliberately leaves to the strategy's own
        bundled defaults.  Stage-level fallbacks only apply when neither is
        configured.
        """
        return {}

    def snapshot_bindings(self, provider: ComponentProvider) -> dict[str, object]:
        """Read the slots selected during execution preparation."""
        values = getattr(provider, "binding_values", None)
        if callable(values):
            values = cast(dict[str, object], values())
        else:
            values = {name: getattr(provider, name, None) for name in BINDING_SLOTS}
            if not callable(getattr(provider, "resolve_defaults", None)):
                defaults = self._default_components()
                values = {
                    slot: value if value is not None else defaults.get(slot, value)
                    for slot, value in values.items()
                }
        return values


_BUILD_GRAPH_BASE_IMPLEMENTATIONS = (
    OptimizationStrategy.build_graph,
    TemplateStrategy.build_graph,
)
_WARNED_BUILD_GRAPH_OVERRIDES: weakref.WeakSet[type] = weakref.WeakSet()


def warn_deprecated_build_graph_override(strategy: object) -> None:
    """Warn once per strategy class that lowers its own component graph.

    The class's ``build_graph`` is compared by identity against the base
    implementations, so a plain subclass of a built-in strategy does not
    warn while a custom override does.
    """
    cls = type(strategy)
    build_graph = getattr(cls, "build_graph", None)
    if build_graph in _BUILD_GRAPH_BASE_IMPLEMENTATIONS:
        return
    if cls in _WARNED_BUILD_GRAPH_OVERRIDES:
        return
    _WARNED_BUILD_GRAPH_OVERRIDES.add(cls)
    warn_deprecated(
        f"Overriding build_graph(provider) in {cls.__qualname__}",
        since="0.2.0",
        replacement="TemplateStrategy with a declared Pipeline",
        stacklevel=2,
    )
