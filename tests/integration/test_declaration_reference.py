"""Declaration Stages reference components by (slot, role).

A declaration Pipeline holds ``ComponentRef`` references or explicit
Stage-local components.  Preparing a declaration for an optimizer resolves
every reference from that plan's binding snapshot into plan-owned Stage
instances; the declaration itself is never mutated, so the same declaration
can be prepared for two Optimizers without cross-talk.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))

import test_component_replacement as _replacement_fixtures

from saealib import (
    PSO,
    CrossoverBLXAlpha,
    GaussianKernel,
    LHSInitializer,
    MutationUniform,
    Optimizer,
    Problem,
    SequentialSelection,
    Termination,
    TruncationSelection,
)
from saealib.acquisition import MeanPrediction
from saealib.algorithms import GA
from saealib.callback import CallbackManager
from saealib.core.contracts.observation import SURROGATE
from saealib.core.graph_builder import (
    _compose_contracts,
    _held_components,
)
from saealib.exceptions import ValidationError
from saealib.execution.runner import Runner
from saealib.execution.runtime import create_runtime
from saealib.pipeline import ComponentRef, Pipeline, Repeat, Stage
from saealib.policies.evaluation import (
    EvaluateAll,
    RepeatedEvaluation,
)
from saealib.policies.feedback import (
    MixedFeedback,
    NoFeedback,
    PredictedFeedback,
    TrueOnlyFeedback,
)
from saealib.registry import build, to_spec
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
    SurrogateOnlyLoopStage,
    SurrogatePredictStage,
    TellStage,
    TrueEvaluationStage,
)
from saealib.strategies import (
    DirectStrategy,
    GenerationBasedStrategy,
    TemplateStrategy,
)
from saealib.surrogate import RBFSurrogate
from saealib.surrogate.manager import LocalSurrogateManager


def _problem(dim: int = 2) -> Problem:
    return Problem(
        func=lambda x: np.array([float(np.sum(np.asarray(x) ** 2))], dtype=np.float64),
        dim=dim,
        n_obj=1,
        direction=np.array([-1.0]),
        lb=[-5.0] * dim,
        ub=[5.0] * dim,
        constraints=[],
    )


def _genetic_algorithm() -> GA:
    return GA(
        crossover=CrossoverBLXAlpha(prob=0.9, alpha=0.4),
        mutation=MutationUniform(prob_var=0.1),
        parent_selection=SequentialSelection(),
        survivor_selection=TruncationSelection(),
    )


def _as_x_rows(population: Any) -> tuple[tuple[float, ...], ...]:
    return tuple(
        tuple(float(value) for value in row)
        for row in np.asarray(population.get_array("x")).tolist()
    )


class _RecordingGA(GA):
    def __init__(self) -> None:
        super().__init__(
            crossover=CrossoverBLXAlpha(prob=0.9, alpha=0.4),
            mutation=MutationUniform(prob_var=0.1),
            parent_selection=SequentialSelection(),
            survivor_selection=TruncationSelection(),
        )
        self.asks: list[int] = []
        self.ask_xs: list[tuple[tuple[float, ...], ...]] = []
        self.tell_xs: list[tuple[tuple[float, ...], ...]] = []

    def ask(self, request: Any, state: Any) -> Any:
        batch = super().ask(request, state)
        self.asks.append(len(batch.candidates))
        self.ask_xs.append(_as_x_rows(batch.candidates))
        return batch

    def tell(self, feedback: Any, state: Any) -> Any:
        population = state.context.offspring
        self.tell_xs.append(_as_x_rows(population))
        return super().tell(feedback, state)


# ---------------------------------------------------------------------------
# Declarations
# ---------------------------------------------------------------------------


def _direct_declaration() -> Pipeline:
    return Pipeline(
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
        ]
    )


def _gb_declaration(gen_ctrl: int) -> Pipeline:
    """The built-in generation-based pipeline written as a declaration."""
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
    steps = []
    if gen_ctrl:
        steps.append(SurrogateFitStage(ComponentRef("fitter")))
        steps.append(
            Repeat(surrogate_generation, count=gen_ctrl, name="surrogate_generations")
        )
    steps.append(true_generation)
    return Pipeline(name="generation_based", steps=steps)


def _declaration_stages(entry: Any) -> list[tuple[Any, Any]]:
    """Every Stage of a declaration, with its remaining references."""
    found: list[tuple[Any, Any]] = []
    stack: list[Any] = [entry]
    while stack:
        value = stack.pop()
        if isinstance(value, Pipeline):
            stack.extend(reversed(value.steps))
        elif isinstance(value, Repeat):
            stack.append(value.body)
        elif isinstance(value, Stage):
            found.append((value, getattr(value, "_refs", ())))
    return found


def _archive(state: Any) -> tuple[np.ndarray, np.ndarray]:
    archive = state.archive
    xs = np.asarray(archive.get_array("x"))
    fs = np.asarray(archive.get_array("f"))
    values = fs if fs.ndim == 1 else fs[:, 0]
    order = np.lexsort((*xs[:, ::-1].T, values))
    return values[order], xs[order]


# ---------------------------------------------------------------------------
# Reference value
# ---------------------------------------------------------------------------


class _MarkerStage(Stage):
    name = "marker"

    def execute(self, state: Any) -> Any:
        return state.replace(
            data={**state.data, "markers": [*state.data.get("markers", []), "marker"]}
        )


class _CompositeAsk(AskStage):
    """Ask stage that runs its declared child Stages after asking."""

    def __init__(self, algorithm: Any) -> None:
        super().__init__(algorithm)
        self.seen: list[str] = []

    def execute(self, state: Any) -> Any:
        self.seen.append("ask")
        for child in self.stages:
            state = child.execute(state)
        return state


def test_component_ref_defaults_to_the_well_known_slot() -> None:
    assert ComponentRef("proposer").slot == "algorithm"
    assert ComponentRef("predictor", slot="surrogate_manager").slot == (
        "surrogate_manager"
    )


def test_component_ref_rejects_unregistered_roles() -> None:
    with pytest.raises(ValidationError, match="not a registered role"):
        ComponentRef("nope")
    with pytest.raises(ValidationError, match="no well-known slot"):
        ComponentRef("decomposition")


def test_component_ref_round_trips_through_the_registry() -> None:
    ref = ComponentRef("feedback_builder")
    spec = to_spec(ref)
    assert spec["type"].endswith("ComponentRef")
    assert spec["params"]["role"] == "feedback_builder"
    assert spec["params"]["slot"] == "feedback_builder"
    assert build(spec) == ref


# ---------------------------------------------------------------------------
# Resolution semantics
# ---------------------------------------------------------------------------

_SLOT_VALUE = {
    "algorithm": "algorithm",
    "evaluator": "evaluator",
    "evaluation_planner": "evaluation_planner",
    "feedback_builder": "feedback_builder",
    "surrogate_manager": "surrogate_manager",
    "acquisition": "acquisition",
    "strategy": "strategy",
    "cbmanager": "cbmanager",
}

_EXPECTED_DIRECT_ROLES = (
    "proposer",
    "evaluation_planner",
    "evaluator",
    "evaluator",
    "feedback_builder",
    "feedback_consumer",
    "evaluator",
)


def test_resolution_is_plan_owned_and_never_mutates_the_declaration() -> None:
    from saealib.execution.prepare import resolve_declaration

    declaration = _direct_declaration()
    algorithm = _RecordingGA()
    evaluator = _replacement_fixtures._DeterministicEvaluator()
    bindings = {
        "algorithm": algorithm,
        "evaluator": evaluator,
        "evaluation_planner": None,
        "feedback_builder": None,
        "surrogate_manager": None,
        "acquisition": None,
        "strategy": None,
        "cbmanager": None,
    }

    resolved = resolve_declaration(declaration, bindings)

    declared = _declaration_stages(declaration)
    stages = _declaration_stages(resolved)
    # every declared stage still carries its (slot, role) reference ...
    roles = tuple(refs[0][1].role for _, refs in declared if refs)
    assert roles == _EXPECTED_DIRECT_ROLES
    assert resolved is not declaration
    # ... while the resolved stages hold the snapshot's components
    assert declared[1][0]._algorithm is None
    assert stages[1][0]._algorithm is algorithm
    assert declared[3][0]._evaluator is None
    assert stages[3][0]._evaluator is evaluator
    assert all(
        resolved_stage is not declared_stage
        for (resolved_stage, _), (declared_stage, _) in zip(
            stages, declared, strict=True
        )
    )


def _evaluator_fixture(max_collects: int = 4000) -> Any:
    return _replacement_fixtures._DeterministicEvaluator(
        release_after_collects=1, max_collects=max_collects
    )


def test_plan_defaults_are_fresh_per_plan_and_shared_within_it() -> None:
    from saealib.execution.prepare import resolve_declaration

    declaration = _direct_declaration()
    bindings = {
        "algorithm": _RecordingGA(),
        "evaluator": _evaluator_fixture(),
        "evaluation_planner": None,
        "feedback_builder": None,
        "surrogate_manager": None,
        "acquisition": None,
        "strategy": None,
        "cbmanager": None,
    }

    resolved = resolve_declaration(declaration, bindings)
    stages = [stage for stage, _ in _declaration_stages(resolved)]
    planner, builder = stages[2]._planner, stages[7]._builder
    assert isinstance(planner, EvaluateAll)
    assert isinstance(builder, TrueOnlyFeedback)

    again = resolve_declaration(declaration, bindings)
    again_stages = [stage for stage, _ in _declaration_stages(again)]
    assert again_stages[2]._planner is not planner
    assert again_stages[7]._builder is not builder

    twice = Pipeline(
        steps=[
            FeedbackStage(ComponentRef("feedback_builder")),
            FeedbackStage(ComponentRef("feedback_builder")),
        ]
    )
    shared = [
        stage for stage, _ in _declaration_stages(resolve_declaration(twice, bindings))
    ]
    assert shared[0]._builder is shared[1]._builder


def test_explicit_component_stays_stage_local() -> None:
    from saealib.execution.prepare import resolve_declaration

    algorithm = _RecordingGA()
    explicit = MixedFeedback()
    snapshot_builder = TrueOnlyFeedback()
    declaration = Pipeline(
        steps=[
            AskStage(ComponentRef("proposer")),
            FeedbackStage(ComponentRef("feedback_builder")),
            FeedbackStage(explicit),
        ]
    )
    resolved = resolve_declaration(
        declaration,
        {
            "algorithm": algorithm,
            "feedback_builder": snapshot_builder,
            "cbmanager": None,
        },
    )
    stages = [stage for stage, _ in _declaration_stages(resolved)]
    ask, shared, local = stages

    assert ask._algorithm is algorithm
    assert shared._builder is snapshot_builder
    # the explicit component is used by its own stage only, inside its own
    # plan-owned copy of the stage
    assert local._builder is explicit
    assert local is not declaration.steps[2]
    declared_local = cast(Any, declaration.steps[2])
    assert declared_local._builder is explicit


def test_surrogate_only_loop_resolves_references_into_inner_stages() -> None:
    from saealib.execution.prepare import resolve_declaration

    surrogate_manager = LocalSurrogateManager(RBFSurrogate(GaussianKernel()))
    acquisition = MeanPrediction()
    builder = TrueOnlyFeedback()
    declaration = Pipeline(
        steps=[
            SurrogateOnlyLoopStage(
                ComponentRef("proposer"),
                ComponentRef("predictor"),
                1,
                acquisition=ComponentRef("acquisition"),
                feedback_builder=ComponentRef("feedback_builder"),
            )
        ]
    )
    resolved = resolve_declaration(
        declaration,
        {
            "algorithm": _genetic_algorithm(),
            "surrogate_manager": surrogate_manager,
            "acquisition": acquisition,
            "feedback_builder": builder,
            "cbmanager": None,
        },
    )
    inner = next(stage for stage, _ in _declaration_stages(resolved))
    assert inner._gen_ctrl == 1
    ask = inner.stages[1]
    predict = inner.stages[2]
    score = inner.stages[3]
    feedback = inner.stages[4]
    tell = inner.stages[5]

    assert ask._algorithm is inner.stages[5]._algorithm
    assert predict._sm is surrogate_manager and predict._refit is False
    assert score._acquisition is acquisition
    assert feedback._builder is builder
    assert tell._channel is SURROGATE
    # when the slot is empty, the loop keeps its built-in mixed default
    empty = resolve_declaration(
        Pipeline(
            steps=[
                SurrogateOnlyLoopStage(
                    ComponentRef("proposer"),
                    ComponentRef("predictor"),
                    1,
                    acquisition=ComponentRef("acquisition"),
                    feedback_builder=ComponentRef("feedback_builder"),
                )
            ]
        ),
        {
            "algorithm": _genetic_algorithm(),
            "surrogate_manager": surrogate_manager,
            "acquisition": acquisition,
            "feedback_builder": None,
            "cbmanager": None,
        },
    )
    inner_empty = next(stage for stage, _ in _declaration_stages(empty))
    assert isinstance(inner_empty.stages[4]._builder, MixedFeedback)


def test_resolution_keeps_declared_stage_subclasses_and_metadata() -> None:
    from saealib.execution.prepare import resolve_declaration

    seen: list[str] = []

    class _ProbedAsk(AskStage):
        def execute(self, state: Any) -> Any:
            seen.append("ask")
            return super().execute(state)

    class _ProbedFeedback(FeedbackStage):
        def execute(self, state: Any) -> Any:
            seen.append("feedback")
            return super().execute(state)

    custom_ask = _ProbedAsk(ComponentRef("proposer"), n_offspring=2)
    custom_ask.name = "custom_ask"
    custom_ask.label = "Custom ask"
    custom_ask.notation = "N"
    custom_feedback = _ProbedFeedback(ComponentRef("feedback_builder"))
    custom_feedback.name = "custom_feedback"
    custom_feedback.label = "Custom feedback"
    custom_feedback.notation = r"$N$"

    resolved = cast(
        Pipeline,
        resolve_declaration(
            Pipeline(
                [custom_ask, custom_feedback],
                name="custom",
                label="Custom flow",
            ),
            {
                "algorithm": _genetic_algorithm(),
                "feedback_builder": TrueOnlyFeedback(),
                "cbmanager": None,
            },
        ),
    )

    resolved_ask = cast(Any, resolved.steps[0])
    resolved_feedback = cast(Any, resolved.steps[1])
    assert isinstance(resolved_ask, _ProbedAsk)
    assert isinstance(resolved_feedback, _ProbedFeedback)
    assert resolved_ask.name == "custom_ask"
    assert resolved_ask.label == "Custom ask"
    assert resolved_ask.notation == "N"
    assert resolved_feedback.name == "custom_feedback"
    assert resolved_feedback.label == "Custom feedback"
    assert resolved_feedback.notation == r"$N$"
    # every declared reference is gone and every bound component is attached
    assert resolved_ask._refs == ()
    assert resolved_feedback._refs == ()
    holding_ask = AskStage(resolved_ask._algorithm, n_offspring=2, cbmanager=None)
    assert resolved_ask.contract() == holding_ask.contract()
    assert resolved_ask._state_reads == holding_ask._state_reads
    assert resolved.find("custom_ask") is resolved_ask
    assert resolved.find("custom_feedback") is resolved_feedback
    replacement = _ProbedAsk(ComponentRef("proposer"), n_offspring=1)
    replacement.name = "custom_ask"
    resolved.replace("custom_ask", replacement)
    replaced_ask = resolved.find("custom_ask")
    assert replaced_ask is not None
    assert cast(Any, replaced_ask)._n_offspring == 1
    assert isinstance(replaced_ask, _ProbedAsk)


def test_declared_subclass_stages_run_with_their_overrides_in_a_plan() -> None:
    seen: list[str] = []

    class _ProbedAsk(AskStage):
        def execute(self, state: Any) -> Any:
            seen.append("ask")
            return super().execute(state)

    class _ProbedFeedback(FeedbackStage):
        def execute(self, state: Any) -> Any:
            seen.append("feedback")
            return super().execute(state)

    custom_ask = _ProbedAsk(ComponentRef("proposer"))
    custom_ask.name = "custom_ask"
    custom_feedback = _ProbedFeedback(ComponentRef("feedback_builder"))
    custom_feedback.name = "custom_feedback"
    steps = list(_direct_declaration().steps)
    steps[1] = custom_ask
    steps[7] = custom_feedback
    strategy = TemplateStrategy(Pipeline(steps=steps, name="probed"))
    optimizer = _build_optimizer(_genetic_algorithm(), strategy, seed=9)
    state = optimizer.run()

    assert seen.count("ask") == state.gen
    assert seen.count("feedback") == state.gen
    resolved_pipeline = cast(Any, strategy.build_pipeline(optimizer))
    assert resolved_pipeline.find("custom_ask") is not None
    assert resolved_pipeline.find("custom_feedback") is not None
    assert isinstance(resolved_pipeline.find("custom_ask"), _ProbedAsk)
    assert isinstance(resolved_pipeline.find("custom_feedback"), _ProbedFeedback)


def test_resolution_copies_composite_stage_state_and_keeps_children() -> None:
    import pickle

    from saealib.execution.prepare import resolve_declaration

    declaration_stage = _CompositeAsk(ComponentRef("proposer"))
    declaration_stage.stages = [_MarkerStage()]
    declaration = Pipeline([declaration_stage], name="composite")
    optimizer = _build_optimizer(
        _genetic_algorithm(), TemplateStrategy(_direct_declaration())
    )
    optimizer.resolve_defaults()
    state = cast(Any, optimizer.initializer).initialize(optimizer, optimizer.problem)
    bindings = {"algorithm": optimizer.algorithm, "cbmanager": None}

    first = cast(Any, resolve_declaration(declaration, bindings)).steps[0]
    second = cast(Any, resolve_declaration(declaration, bindings)).steps[0]

    # the declared child Stages survive resolution and run
    assert len(first.stages) == 1 and len(second.stages) == 1
    assert isinstance(first, _CompositeAsk) and isinstance(second, _CompositeAsk)
    first_state = first.execute(state)
    assert first.seen == ["ask"]
    assert first_state.data.get("markers") == ["marker"]
    second.execute(state)
    assert second.seen == ["ask"]
    # each resolved Stage owns its mutable state; the declaration is untouched
    assert first.seen is not second.seen
    assert declaration_stage.seen == []
    assert declaration_stage.stages[0] is not first.stages[0]
    # the same holds after pickling a resolved plan
    third = cast(Any, resolve_declaration(declaration, bindings)).steps[0]
    loaded_stage = pickle.loads(pickle.dumps(third))
    assert isinstance(loaded_stage, _CompositeAsk)
    assert len(loaded_stage.stages) == 1
    loaded_state = loaded_stage.execute(state)
    assert loaded_stage.seen == ["ask"]
    assert loaded_state.data.get("markers") == ["marker"]


def test_surrogate_only_loop_keeps_explicit_components_next_to_references() -> None:
    from saealib.execution.prepare import BINDING_SLOTS, resolve_declaration
    from saealib.policies.feedback import PredictedFeedback

    optimizer = _build_optimizer(
        _genetic_algorithm(), TemplateStrategy(_gb_declaration(1)), surrogate=True
    )
    optimizer.resolve_defaults()
    bindings = {name: getattr(optimizer, name, None) for name in BINDING_SLOTS}

    mixed = SurrogateOnlyLoopStage(
        optimizer.algorithm,
        ComponentRef("fitter"),
        2,
        acquisition=optimizer.acquisition,
    )
    resolved_pipeline = cast(
        Any, resolve_declaration(Pipeline([mixed], name="loop"), bindings)
    )
    resolved = resolved_pipeline.steps[0]
    assert resolved._algorithm is optimizer.algorithm
    assert resolved._acquisition is optimizer.acquisition
    assert resolved._sm is optimizer.surrogate_manager
    inner_asks = [stage for stage in resolved.stages if isinstance(stage, AskStage)]
    assert inner_asks and inner_asks[0]._algorithm is optimizer.algorithm

    builder = PredictedFeedback()
    loop = SurrogateOnlyLoopStage(
        ComponentRef("proposer"),
        ComponentRef("fitter"),
        2,
        acquisition=ComponentRef("acquisition"),
        feedback_builder=builder,
    )
    resolved_pipeline = cast(
        Any, resolve_declaration(Pipeline([loop], name="loop"), bindings)
    )
    resolved = resolved_pipeline.steps[0]
    inner_feedback = [
        stage for stage in resolved.stages if isinstance(stage, FeedbackStage)
    ]
    assert inner_feedback and inner_feedback[0]._builder is builder


def test_surrogate_only_loop_rejects_a_missing_component_and_reference() -> None:
    with pytest.raises(ValidationError, match="algorithm"):
        SurrogateOnlyLoopStage(
            cast(Any, None),
            ComponentRef("fitter"),
            2,
            acquisition=ComponentRef("acquisition"),
        )


def test_unresolved_reference_fails_at_graph_build() -> None:
    stage = AskStage(ComponentRef("proposer"))
    with pytest.raises(ValidationError, match="unresolved"):
        stage.contract()
    with pytest.raises(ValidationError, match="unresolved"):
        _held_components(stage)


def test_resolved_contract_matches_the_component_holding_constructor() -> None:
    from saealib.execution.prepare import resolve_declaration

    algorithm = _RecordingGA()
    cbmanager = CallbackManager()
    declaration = Pipeline(steps=[AskStage(ComponentRef("proposer"), n_offspring=3)])
    resolved = resolve_declaration(
        declaration, {"algorithm": algorithm, "cbmanager": cbmanager}
    )
    resolved_ask = _declaration_stages(resolved)[0][0]
    holding = AskStage(algorithm, n_offspring=3, cbmanager=cbmanager)

    assert resolved_ask.contract() == holding.contract()
    assert resolved_ask._state_reads == holding._state_reads
    # lifecycle, execution, assumptions, ports, and state composition of
    # the bound component survive resolution
    assert _compose_contracts(_held_components(resolved_ask)) == _compose_contracts(
        _held_components(holding)
    )


# ---------------------------------------------------------------------------
# Run-level acceptance
# ---------------------------------------------------------------------------


def _policy_run(
    kind: str,
    planner: Any,
    feedback: Any,
    *,
    expert_planner: Any = None,
    expert_feedback: Any = None,
) -> tuple[Any, int]:
    """Run one policy probe; return the final state and the tell count.

    True evaluations and tells count the components that actually drove the
    run, not just the attributes resolution would report.
    """
    strategy = DirectStrategy() if kind == "direct" else GenerationBasedStrategy(0)
    strategy.evaluation_planner = planner
    if kind == "direct":
        strategy.feedback_builder = feedback
    else:
        cast(Any, strategy).true_feedback_builder = feedback
    algorithm = _RecordingGA()
    optimizer = _build_optimizer(algorithm, strategy, seed=17)
    optimizer.set_termination(Termination(lambda state: state.gen >= 6))
    if expert_planner is not None:
        optimizer.set_evaluation_planner(expert_planner)
    if expert_feedback is not None:
        optimizer.set_feedback_builder(expert_feedback)
    state = optimizer.run()
    return state, len(algorithm.tell_xs)


N_INIT, POPULATION, N_GEN = 6, 4, 6


@pytest.mark.parametrize("kind", ["direct", "gb"])
def test_empty_optimizer_planner_slot_uses_the_strategy_own_planner(kind: str) -> None:
    state, tells = _policy_run(kind, RepeatedEvaluation(2), NoFeedback())
    assert state.fe == N_INIT + 2 * POPULATION * N_GEN
    assert tells == 0


@pytest.mark.parametrize("kind", ["direct", "gb"])
def test_empty_optimizer_feedback_slot_uses_the_strategy_own_feedback(
    kind: str,
) -> None:
    state, tells = _policy_run(kind, EvaluateAll(), NoFeedback())
    assert state.fe == N_INIT + POPULATION * N_GEN
    assert tells == 0


@pytest.mark.parametrize("kind", ["direct", "gb"])
def test_explicit_optimizer_planner_slot_wins_over_the_strategy_own_planner(
    kind: str,
) -> None:
    state, tells = _policy_run(
        kind,
        RepeatedEvaluation(2),
        NoFeedback(),
        expert_planner=RepeatedEvaluation(1),
    )
    assert state.fe == N_INIT + POPULATION * N_GEN
    assert tells == 0


@pytest.mark.parametrize("kind", ["direct", "gb"])
def test_explicit_optimizer_feedback_slot_wins_over_the_strategy_own_feedback(
    kind: str,
) -> None:
    state, tells = _policy_run(
        kind,
        EvaluateAll(),
        NoFeedback(),
        expert_feedback=TrueOnlyFeedback(),
    )
    assert state.fe == N_INIT + POPULATION * N_GEN
    assert tells == N_GEN


def _build_optimizer(
    algorithm: Any,
    strategy: Any,
    seed: int = 7,
    *,
    surrogate: bool = False,
) -> Optimizer:
    optimizer = Optimizer(_problem(), seed=seed)
    optimizer = optimizer.set_initializer(LHSInitializer(6, 4, seed)).set_algorithm(
        algorithm
    )
    if surrogate:
        optimizer = optimizer.set_surrogate_manager(
            LocalSurrogateManager(RBFSurrogate(GaussianKernel()))
        ).set_acquisition(MeanPrediction())
    return optimizer.set_strategy(strategy).set_termination(
        Termination(lambda state: state.gen >= 8)
    )


def _capturing_runner(optimizer: Any) -> tuple[Runner, dict[str, Any]]:
    captured: dict[str, Any] = {}

    def factory(optimizer_obj: Any) -> Any:
        runtime = create_runtime(optimizer_obj)
        captured["environment"] = getattr(runtime, "environment", None)
        return runtime

    return Runner(optimizer, runtime_factory=factory), captured


def _plan_ask_components(plan: Any) -> list[Any]:
    """Components the plan's ask stages resolved to."""
    graph = plan.graph if hasattr(plan, "graph") else plan.plan.graph
    components = []
    for node in graph.nodes:
        stage = getattr(node.component, "stage", None)
        if getattr(stage, "name", "") == "ask":
            components.append(getattr(stage, "_algorithm"))
    return components


@pytest.mark.parametrize("seed", [0, 7])
def test_direct_declaration_matches_the_builtin_strategy(seed: int) -> None:
    control = _build_optimizer(_genetic_algorithm(), DirectStrategy(), seed=seed)
    template = _build_optimizer(
        _genetic_algorithm(), TemplateStrategy(_direct_declaration()), seed=seed
    )
    control_state, template_state = control.run(), template.run()

    assert control_state.gen == template_state.gen
    assert np.array_equal(_archive(control_state)[0], _archive(template_state)[0])
    assert np.array_equal(_archive(control_state)[1], _archive(template_state)[1])


@pytest.mark.parametrize("gen_ctrl", [0, 2])
def test_generation_based_declaration_matches_the_builtin_strategy(
    gen_ctrl: int,
) -> None:
    seed = 3
    control = _build_optimizer(
        _genetic_algorithm(),
        GenerationBasedStrategy(gen_ctrl),
        seed=seed,
        surrogate=True,
    )
    template = _build_optimizer(
        _genetic_algorithm(),
        TemplateStrategy(_gb_declaration(gen_ctrl), requires_surrogate=True),
        seed=seed,
        surrogate=True,
    )
    control_state, template_state = control.run(), template.run()

    assert control_state.gen == template_state.gen
    assert control_state.fe == template_state.fe
    assert np.array_equal(_archive(control_state)[0], _archive(template_state)[0])
    assert np.array_equal(_archive(control_state)[1], _archive(template_state)[1])


def test_shared_declaration_prepared_for_two_optimizers_without_cross_talk() -> None:
    declaration = _direct_declaration()
    shared = TemplateStrategy(declaration)
    algorithm_a = _RecordingGA()
    algorithm_b = PSO()

    optimizer_a = _build_optimizer(algorithm_a, shared)
    optimizer_b = (
        Optimizer(_problem(), seed=11)
        .set_initializer(LHSInitializer(6, 4, 11))
        .set_algorithm(algorithm_b)
        .set_strategy(shared)
        .set_termination(Termination(lambda state: state.gen >= 8))
    )

    runner_b, captured_b = _capturing_runner(optimizer_b)
    generator = runner_b.iterate()
    next(generator)
    next(generator)
    generator.close()
    # B already holds a plan when A's first builder runs
    plan_b_when_a_starts = captured_b["environment"].plan

    runner_a, captured_a = _capturing_runner(optimizer_a)
    generator = runner_a.iterate()
    states_a = []
    for state in generator:
        states_a.append(state)
        if state.gen >= 3:
            break
    state_b = optimizer_b.run()
    for state in runner_a.iterate_from(states_a[-1]):
        states_a.append(state)

    reference_a = _build_optimizer(_genetic_algorithm(), DirectStrategy(), seed=7).run()
    reference_b = (
        Optimizer(_problem(), seed=11)
        .set_initializer(LHSInitializer(6, 4, 11))
        .set_algorithm(PSO())
        .set_strategy(DirectStrategy())
        .set_termination(Termination(lambda state: state.gen >= 8))
        .run()
    )

    assert np.array_equal(_archive(reference_a)[0], _archive(states_a[-1])[0])
    assert np.array_equal(_archive(reference_a)[1], _archive(states_a[-1])[1])
    assert np.array_equal(_archive(reference_b)[0], _archive(state_b)[0])
    assert np.array_equal(_archive(reference_b)[1], _archive(state_b)[1])
    # B's earlier plan kept B's components while A was being prepared
    assert all(
        component is algorithm_b
        for component in _plan_ask_components(plan_b_when_a_starts)
    )
    assert all(
        component is algorithm_b
        for component in _plan_ask_components(optimizer_b._executable_plan)
    )
    assert all(
        component is algorithm_a
        for component in _plan_ask_components(captured_a["environment"].plan)
    )


def test_replacement_waits_for_in_flight_async_requests() -> None:
    from saealib.execution import AsyncEvaluationScheduler

    first_algorithm = _RecordingGA()
    second_algorithm = _RecordingGA()
    captured_runtime: dict[str, Any] = {}
    swap_record: dict[str, Any] = {}

    class _SwapEvaluator(_replacement_fixtures._DeterministicEvaluator):
        def submit(self, request: Any, problem: Any) -> Any:
            handle = super().submit(request, problem)
            if swap_record:
                return handle
            swap_record["version"] = captured_runtime["environment"].active_version
            swap_record["pinned"] = captured_runtime["environment"].binding_values()
            optimizer.set_algorithm(second_algorithm)
            return handle

    evaluator = _SwapEvaluator(release_after_collects=1, max_collects=3000)
    optimizer = (
        Optimizer(_problem(), seed=42)
        .set_initializer(LHSInitializer(6, 4, 42))
        .set_algorithm(first_algorithm)
        .set_strategy(TemplateStrategy(_direct_declaration()))
        .set_evaluator(evaluator)
        .set_async_evaluation_scheduler(
            AsyncEvaluationScheduler(evaluator, max_pending=1)
        )
        .set_termination(Termination(lambda state: state.gen >= 8))
    )
    runner, captured_runtime = _capturing_runner(optimizer)

    state = runner.run()

    # the swap stayed pinned to the active version while requests were in
    # flight, and the drain published one new version afterwards
    assert swap_record["pinned"]["algorithm"] is first_algorithm
    assert captured_runtime["environment"].active_version == swap_record["version"] + 1
    assert len(first_algorithm.asks) == 1
    assert len(second_algorithm.asks) >= 7
    first_asked = {row for rows in first_algorithm.ask_xs for row in rows}
    second_asked = {row for rows in second_algorithm.ask_xs for row in rows}
    assert first_asked.isdisjoint(second_asked)
    assert all(row in first_asked for rows in first_algorithm.tell_xs for row in rows)
    assert all(row in second_asked for rows in second_algorithm.tell_xs for row in rows)
    assert state.gen == 8
    assert captured_runtime["environment"].active_version == (
        swap_record["version"] + 1
    )


def test_plan_rebuild_keeps_the_executing_plan_components() -> None:
    algorithm_first = _RecordingGA()
    algorithm_second = _RecordingGA()
    optimizer = _build_optimizer(
        algorithm_first, TemplateStrategy(_direct_declaration())
    )
    runner, captured = _capturing_runner(optimizer)

    generator = runner.iterate()
    next(generator)
    next(generator)
    plan_before = captured["environment"].plan
    optimizer.set_algorithm(algorithm_second)
    for _state in generator:
        pass

    plan_after = captured["environment"].plan
    assert plan_after is not plan_before
    # the plan that was executing when the replacement was requested kept
    # the components it was resolved with
    assert all(
        component is algorithm_first for component in _plan_ask_components(plan_before)
    )
    assert all(
        component is algorithm_second for component in _plan_ask_components(plan_after)
    )


def _held_stages(entry: Any) -> list[Any]:
    stage = getattr(entry, "stage", entry)
    if isinstance(stage, Stage):
        return [
            stage,
            *(held for child in stage.stages for held in _held_stages(child)),
        ]
    children = [
        *getattr(entry, "steps", ()),
        *(
            getattr(entry, name, None)
            for name in ("body", "then", "else_")
            if getattr(entry, name, None) is not None
        ),
    ]
    return [held for child in children for held in _held_stages(child)]


def _fallback_feedback_template() -> TemplateStrategy:
    return TemplateStrategy(_direct_declaration())


@pytest.mark.parametrize(
    ("strategy_factory", "surrogate"),
    [
        (_fallback_feedback_template, False),
        (DirectStrategy, False),
        (lambda: GenerationBasedStrategy(1), True),
    ],
    ids=["template-with-fallback", "direct", "generation-based"],
)
def test_strategy_step_runs_the_pipeline_it_keeps(
    monkeypatch: pytest.MonkeyPatch, strategy_factory: Any, surrogate: bool
) -> None:
    strategy = strategy_factory()
    optimizer = _build_optimizer(_genetic_algorithm(), strategy, surrogate=surrogate)
    executed: list[Any] = []
    for stage_type in {type(stage) for stage in _held_stages(strategy.declaration)}:
        original = stage_type.execute

        def execute(self: Any, state: Any, _original: Any = original) -> Any:
            executed.append(self)
            return _original(self, state)

        monkeypatch.setattr(stage_type, "execute", execute)

    initializer = optimizer.initializer
    assert initializer is not None
    state = strategy.step(
        initializer.initialize(optimizer, optimizer.problem), optimizer
    )

    kept = _held_stages(strategy.pipeline)
    assert state.gen >= 1
    assert executed
    assert all(any(stage is held for held in kept) for stage in executed)
    if strategy_factory is _fallback_feedback_template:
        assert optimizer.feedback_builder is None
        feedback = [stage for stage in executed if isinstance(stage, FeedbackStage)]
        assert feedback
        assert all(isinstance(stage._builder, TrueOnlyFeedback) for stage in feedback)


class _AlwaysTrue:
    def contract(self) -> Any:
        from saealib.core.contracts import StateContract

        return StateContract()

    def evaluate(self, view: Any) -> bool:
        del view
        return True


@pytest.mark.parametrize(
    ("build_stage", "message"),
    [
        (lambda: AskStage(cast(Any, None)), "AskStage requires an algorithm"),
        (lambda: TellStage(cast(Any, None)), "TellStage requires an algorithm"),
        (
            lambda: SurrogatePredictStage(cast(Any, None)),
            "SurrogatePredictStage requires a surrogate manager",
        ),
        (
            lambda: SurrogateFitStage(cast(Any, None)),
            "SurrogateFitStage requires a surrogate manager",
        ),
        (
            lambda: AcquisitionStage(cast(Any, None)),
            "AcquisitionStage requires an acquisition function",
        ),
        (
            lambda: EvaluationSubmitStage(cast(Any, None)),
            "EvaluationSubmitStage requires an evaluator",
        ),
        (
            lambda: EvaluationCollectStage(cast(Any, None)),
            "EvaluationCollectStage requires an evaluator",
        ),
        (
            lambda: EvaluationAcknowledgeStage(cast(Any, None)),
            "EvaluationAcknowledgeStage requires an evaluator",
        ),
        (
            lambda: TrueEvaluationStage(cast(Any, None)),
            "TrueEvaluationStage requires an evaluator",
        ),
        (
            lambda: SurrogateOnlyLoopStage(
                ComponentRef("proposer"),
                cast(Any, None),
                1,
                acquisition=ComponentRef("acquisition"),
            ),
            "SurrogateOnlyLoopStage requires sm",
        ),
    ],
)
def test_stage_without_component_or_reference_is_rejected(
    build_stage: Any, message: str
) -> None:
    with pytest.raises(ValidationError, match=f"^{message}"):
        build_stage()


@pytest.mark.parametrize(
    ("build_stage", "name"),
    [
        (
            lambda cb: AskStage(ComponentRef("proposer"), cbmanager=cb),
            "AskStage",
        ),
        (
            lambda cb: SurrogatePredictStage(ComponentRef("predictor"), cbmanager=cb),
            "SurrogatePredictStage",
        ),
        (
            lambda cb: SurrogateFitStage(ComponentRef("fitter"), cbmanager=cb),
            "SurrogateFitStage",
        ),
        (
            lambda cb: AcquisitionStage(ComponentRef("acquisition"), cbmanager=cb),
            "AcquisitionStage",
        ),
        (
            lambda cb: EvaluationAcknowledgeStage(
                ComponentRef("evaluator"), cbmanager=cb
            ),
            "EvaluationAcknowledgeStage",
        ),
        (
            lambda cb: TrueEvaluationStage(ComponentRef("evaluator"), cbmanager=cb),
            "TrueEvaluationStage",
        ),
        (
            lambda cb: SurrogateOnlyLoopStage(
                ComponentRef("proposer"),
                ComponentRef("predictor"),
                1,
                cb,
                acquisition=ComponentRef("acquisition"),
            ),
            "SurrogateOnlyLoopStage",
        ),
    ],
)
def test_reference_stage_rejects_an_explicit_callback_manager(
    build_stage: Any, name: str
) -> None:
    with pytest.raises(
        ValidationError, match=f"^{name} takes no cbmanager next to a reference"
    ):
        build_stage(CallbackManager())


def test_reference_to_an_empty_slot_without_a_default_fails_resolution() -> None:
    from saealib.execution.prepare import resolve_declaration

    declaration = Pipeline([AskStage(ComponentRef("proposer"))])

    with pytest.raises(
        ValidationError,
        match=r"^AskStage needs a component for role 'proposer' from slot 'algorithm'",
    ):
        resolve_declaration(declaration, {"algorithm": None})


def test_stage_recording_references_without_resolution_support_fails() -> None:
    from saealib.execution.prepare import resolve_declaration

    class _Unresolvable(_MarkerStage):
        _refs = (("_component", ComponentRef("proposer")),)

    with pytest.raises(
        ValidationError,
        match=r"^_Unresolvable records \(slot, role\) references but does not support",
    ):
        resolve_declaration(Pipeline([_Unresolvable()]), {})


def test_true_evaluation_reference_resolves_from_the_evaluator_slot() -> None:
    from saealib.execution.prepare import resolve_declaration

    declared = TrueEvaluationStage(ComponentRef("evaluator"), n_eval=3)
    evaluator = _evaluator_fixture()
    callbacks = CallbackManager()

    resolved = cast(
        Any,
        resolve_declaration(
            Pipeline([declared]), {"evaluator": evaluator, "cbmanager": callbacks}
        ),
    ).steps[0]

    assert isinstance(resolved, TrueEvaluationStage)
    assert resolved._evaluator is evaluator
    assert resolved._cbmanager is callbacks
    assert resolved._n_eval == 3
    assert declared._evaluator is None


def test_loop_and_branch_declarations_resolve_and_lower_their_bodies() -> None:
    from saealib.execution.prepare import declared_components, resolve_declaration
    from saealib.pipeline import Branch, Loop

    condition = _AlwaysTrue()
    declared_ask = AskStage(ComponentRef("proposer"))
    declaration = Pipeline(
        [
            Loop(Pipeline([declared_ask], name="ask"), until=condition, name="loop"),
            Branch(
                condition,
                then=Pipeline([TellStage(ComponentRef("feedback_consumer"))]),
                else_=Pipeline([TellStage(ComponentRef("feedback_consumer"))]),
                name="branch",
            ),
        ],
        name="declared",
    )
    algorithm = _genetic_algorithm()

    resolved = cast(Any, resolve_declaration(declaration, {"algorithm": algorithm}))
    loop, branch = resolved.steps
    lowered_loop, lowered_branch = declared_components(resolved).steps

    assert loop.name == "loop" and loop.until is condition
    assert loop.body.steps[0]._algorithm is algorithm
    assert branch.name == "branch" and branch.condition is condition
    assert branch.then.steps[0]._algorithm is algorithm
    assert branch.else_.steps[0]._algorithm is algorithm
    assert declared_ask._algorithm is None
    assert lowered_loop.name == "loop" and lowered_loop.until is condition
    assert lowered_loop.body.steps[0].stage is loop.body.steps[0]
    assert lowered_branch.name == "branch"
    assert lowered_branch.then.steps[0].stage is branch.then.steps[0]
    assert lowered_branch.else_.steps[0].stage is branch.else_.steps[0]


def test_resolution_copies_dict_and_set_attributes_of_a_declared_stage() -> None:
    from saealib.execution.prepare import resolve_declaration

    marker = object()
    declared = AskStage(ComponentRef("proposer"))
    cast(Any, declared).options = {"limit": [1, 2], "marker": marker}
    cast(Any, declared).tags = {"a", "b"}

    resolved = cast(
        Any,
        resolve_declaration(Pipeline([declared]), {"algorithm": _genetic_algorithm()}),
    ).steps[0]

    assert resolved.options == {"limit": [1, 2], "marker": marker}
    assert resolved.options is not cast(Any, declared).options
    assert resolved.options["limit"] is not cast(Any, declared).options["limit"]
    assert resolved.options["marker"] is marker
    assert resolved.tags == {"a", "b"}
    assert resolved.tags is not cast(Any, declared).tags


def test_graph_native_component_in_a_declaration_is_kept_by_identity() -> None:
    from saealib.execution.prepare import resolve_declaration
    from saealib.stages import stage_component

    component = stage_component(CountGenerationStage())

    resolved = cast(Any, resolve_declaration(Pipeline([component]), {}))

    assert resolved.steps[0] is component


class _LoweredStrategy:
    """Structured strategy that is not a template and lowers its own Pipeline."""

    requires_surrogate = False

    def __init__(self, *, with_pipeline: bool = True) -> None:
        self.built: list[Pipeline] = []
        if with_pipeline:
            self.build_pipeline = self._build_pipeline

    def _build_pipeline(self, provider: Any) -> Pipeline:
        from saealib.stages import stage_component

        pipeline = Pipeline([stage_component(CountGenerationStage())], name="lowered")
        self.built.append(pipeline)
        return pipeline

    def build_graph(self, provider: Any) -> Any:
        from saealib.core.compiler.lowerer import lower_pipeline

        return lower_pipeline(self._build_pipeline(provider))

    def step(self, state: Any, provider: Any) -> Any:
        from saealib.execution.runtime import execute_strategy_step

        return execute_strategy_step(self, state, provider)


def _initial_state(optimizer: Optimizer) -> Any:
    initializer = optimizer.initializer
    assert initializer is not None
    return initializer.initialize(optimizer, optimizer.problem)


def test_structured_strategy_step_keeps_its_build_pipeline_result() -> None:
    strategy = _LoweredStrategy()
    optimizer = _build_optimizer(_genetic_algorithm(), DirectStrategy())

    state = strategy.step(_initial_state(optimizer), optimizer)

    assert state.gen == 1
    assert len(strategy.built) == 2
    assert cast(Any, strategy).pipeline is strategy.built[-1]


def test_structured_strategy_step_without_build_pipeline_is_rejected() -> None:
    strategy = _LoweredStrategy(with_pipeline=False)
    optimizer = _build_optimizer(_genetic_algorithm(), DirectStrategy())

    with pytest.raises(
        ValidationError,
        match=r"^structured strategy graph requires build_pipeline\(provider\)",
    ):
        strategy.step(_initial_state(optimizer), optimizer)
