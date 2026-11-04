"""Focused acceptance checks for the Stage contract inventory."""

from __future__ import annotations

from typing import Any

from saealib.core.contracts import ComponentContract
from saealib.core.state import (
    ACQUISITION_RESULT,
    ARCHIVES_MAIN,
    ARCHIVES_PARETO,
    EVALUATED_OFFSPRING,
    EVALUATION_HANDLES,
    EVALUATION_NEW_IDS,
    EVALUATION_REQUEST,
    EVALUATION_UPDATE_NEW_IDS,
    EVALUATION_UPDATES,
    EVALUATIONS_COUNT,
    EVALUATIONS_OWNERS,
    EVALUATIONS_PENDING,
    EVALUATIONS_PLAN,
    EVALUATIONS_PLAN_STATE,
    EVALUATIONS_PLAN_UPDATES,
    FEEDBACK_RESULT,
    POPULATIONS_MAIN,
    PROPOSALS_CURRENT,
    PROPOSALS_ID_ALLOCATOR,
    PROPOSALS_OFFSPRING,
    RUNTIME_CANDIDATE_ID_ALLOCATOR,
    RUNTIME_DECISION_COUNT,
    RUNTIME_GENERATION,
    RUNTIME_REQUEST_ID_ALLOCATOR,
    RUNTIME_RNG,
    SCORES,
    SURROGATES_PREDICTIONS,
)
from saealib.pipeline import Stage
from saealib.stages import (
    AcquisitionStage,
    ArchiveUpdateStage,
    AskStage,
    AsyncEvaluationSubmitStage,
    CountGenerationStage,
    EvaluationAcknowledgeStage,
    EvaluationApplyStage,
    EvaluationCollectStage,
    EvaluationPlanStage,
    EvaluationSubmitStage,
    FeedbackStage,
    InitializationStage,
    PendingEvaluationContextStage,
    SortByScoreStage,
    SurrogateFitStage,
    SurrogateOnlyLoopStage,
    SurrogatePredictStage,
    TellStage,
    TopKSelectionStage,
    TrueEvaluationStage,
)


class _ContractComponent:
    """Minimal held component used only to exercise Stage composition."""

    def contract(self) -> ComponentContract:
        return ComponentContract()

    def ask(self, request: object, state: object) -> object:
        del request, state
        return None

    def tell(self, feedback: object, state: object) -> object:
        del feedback, state
        return None


OPERATIONAL_STAGES = (
    CountGenerationStage,
    AskStage,
    SurrogatePredictStage,
    PendingEvaluationContextStage,
    AcquisitionStage,
    SurrogateFitStage,
    TopKSelectionStage,
    SortByScoreStage,
    EvaluationPlanStage,
    AsyncEvaluationSubmitStage,
    EvaluationSubmitStage,
    EvaluationCollectStage,
    EvaluationApplyStage,
    EvaluationAcknowledgeStage,
    TrueEvaluationStage,
    ArchiveUpdateStage,
    FeedbackStage,
    TellStage,
    SurrogateOnlyLoopStage,
    InitializationStage,
)


def _operational_stage_instances() -> tuple[Stage, ...]:
    def held() -> Any:
        return _ContractComponent()

    return (
        CountGenerationStage(),
        AskStage(held()),
        SurrogatePredictStage(held()),
        PendingEvaluationContextStage(held()),
        AcquisitionStage(held()),
        SurrogateFitStage(held()),
        TopKSelectionStage(k=1),
        SortByScoreStage(),
        EvaluationPlanStage(),
        AsyncEvaluationSubmitStage(held(), held()),
        EvaluationSubmitStage(held()),
        EvaluationCollectStage(held()),
        EvaluationApplyStage(),
        EvaluationAcknowledgeStage(held()),
        TrueEvaluationStage(held()),
        ArchiveUpdateStage(),
        FeedbackStage(held()),
        TellStage(held()),
        SurrogateOnlyLoopStage(held(), held(), 0, acquisition=held()),
        InitializationStage(held(), held(), held()),
    )


OPERATIONAL_STAGE_INSTANCES = _operational_stage_instances()


def test_all_operational_stages_have_exact_direct_state_contracts() -> None:
    """Every operational Stage exposes the state it directly touches."""
    expected: dict[type[Stage], tuple[set[Any], set[Any]]] = {
        CountGenerationStage: (
            {RUNTIME_GENERATION, EVALUATIONS_PENDING},
            {RUNTIME_GENERATION},
        ),
        AskStage: (
            {
                PROPOSALS_OFFSPRING,
                PROPOSALS_ID_ALLOCATOR,
                EVALUATIONS_PLAN,
                EVALUATIONS_PLAN_STATE,
                RUNTIME_CANDIDATE_ID_ALLOCATOR,
            },
            {
                PROPOSALS_OFFSPRING,
                PROPOSALS_CURRENT,
                RUNTIME_CANDIDATE_ID_ALLOCATOR,
                EVALUATED_OFFSPRING,
            },
        ),
        SurrogatePredictStage: (
            {PROPOSALS_OFFSPRING, POPULATIONS_MAIN, ARCHIVES_MAIN},
            {PROPOSALS_OFFSPRING, SURROGATES_PREDICTIONS},
        ),
        PendingEvaluationContextStage: (set(), set()),
        AcquisitionStage: (
            {
                PROPOSALS_OFFSPRING,
                SURROGATES_PREDICTIONS,
                ARCHIVES_MAIN,
                RUNTIME_GENERATION,
                RUNTIME_DECISION_COUNT,
                RUNTIME_RNG,
            },
            {SCORES, ACQUISITION_RESULT},
        ),
        SurrogateFitStage: ({POPULATIONS_MAIN, ARCHIVES_MAIN}, set()),
        TopKSelectionStage: ({PROPOSALS_OFFSPRING, SCORES}, {PROPOSALS_OFFSPRING}),
        SortByScoreStage: (
            {PROPOSALS_OFFSPRING, SCORES},
            {PROPOSALS_OFFSPRING, SCORES},
        ),
        EvaluationPlanStage: (
            {
                PROPOSALS_OFFSPRING,
                EVALUATIONS_PENDING,
                EVALUATION_REQUEST,
                EVALUATIONS_PLAN,
                EVALUATIONS_PLAN_STATE,
                EVALUATION_HANDLES,
                EVALUATIONS_OWNERS,
                ACQUISITION_RESULT,
                SCORES,
                SURROGATES_PREDICTIONS,
                RUNTIME_REQUEST_ID_ALLOCATOR,
                RUNTIME_DECISION_COUNT,
            },
            {
                EVALUATION_REQUEST,
                EVALUATIONS_PLAN,
                EVALUATIONS_PLAN_STATE,
                EVALUATIONS_PLAN_UPDATES,
                EVALUATIONS_PENDING,
                EVALUATION_UPDATES,
                EVALUATION_UPDATE_NEW_IDS,
                EVALUATION_NEW_IDS,
                EVALUATION_HANDLES,
                EVALUATIONS_OWNERS,
                RUNTIME_REQUEST_ID_ALLOCATOR,
                RUNTIME_DECISION_COUNT,
            },
        ),
        AsyncEvaluationSubmitStage: (
            {
                PROPOSALS_OFFSPRING,
                ACQUISITION_RESULT,
                SCORES,
                EVALUATIONS_PLAN,
                EVALUATIONS_PLAN_STATE,
                EVALUATIONS_PENDING,
                EVALUATION_HANDLES,
                RUNTIME_REQUEST_ID_ALLOCATOR,
                RUNTIME_DECISION_COUNT,
            },
            {
                EVALUATION_REQUEST,
                EVALUATIONS_PLAN,
                EVALUATIONS_PLAN_STATE,
                EVALUATIONS_PLAN_UPDATES,
                EVALUATIONS_PENDING,
                RUNTIME_REQUEST_ID_ALLOCATOR,
                RUNTIME_DECISION_COUNT,
            },
        ),
        EvaluationSubmitStage: (
            {
                EVALUATION_REQUEST,
                EVALUATIONS_PLAN,
                EVALUATIONS_PLAN_STATE,
                EVALUATIONS_PENDING,
                EVALUATION_HANDLES,
                EVALUATIONS_OWNERS,
            },
            {
                EVALUATIONS_PENDING,
                EVALUATION_HANDLES,
                EVALUATIONS_PLAN_STATE,
            },
        ),
        EvaluationCollectStage: (
            {
                EVALUATIONS_PLAN,
                EVALUATIONS_PLAN_STATE,
                EVALUATIONS_PENDING,
                EVALUATION_HANDLES,
                EVALUATIONS_PLAN_UPDATES,
                EVALUATION_REQUEST,
            },
            {
                EVALUATION_UPDATES,
                EVALUATION_REQUEST,
                EVALUATIONS_PLAN,
                EVALUATIONS_PLAN_STATE,
                EVALUATIONS_PLAN_UPDATES,
                EVALUATIONS_PENDING,
                EVALUATION_UPDATE_NEW_IDS,
                EVALUATION_NEW_IDS,
            },
        ),
        EvaluationApplyStage: (
            {
                PROPOSALS_OFFSPRING,
                EVALUATION_REQUEST,
                EVALUATION_UPDATES,
                EVALUATIONS_PLAN,
                EVALUATIONS_PLAN_STATE,
                EVALUATIONS_PLAN_UPDATES,
                EVALUATIONS_PENDING,
            },
            {
                PROPOSALS_OFFSPRING,
                EVALUATED_OFFSPRING,
                EVALUATION_NEW_IDS,
                EVALUATION_UPDATE_NEW_IDS,
                EVALUATIONS_PENDING,
            },
        ),
        EvaluationAcknowledgeStage: (
            {
                PROPOSALS_OFFSPRING,
                EVALUATION_REQUEST,
                EVALUATIONS_PLAN,
                EVALUATIONS_PLAN_STATE,
                EVALUATIONS_PENDING,
                EVALUATION_HANDLES,
                EVALUATION_UPDATES,
                EVALUATION_UPDATE_NEW_IDS,
                EVALUATIONS_PLAN_UPDATES,
                EVALUATIONS_COUNT,
            },
            {
                EVALUATIONS_PLAN,
                EVALUATIONS_PLAN_STATE,
                EVALUATIONS_PLAN_UPDATES,
                EVALUATIONS_PENDING,
                EVALUATION_HANDLES,
                EVALUATIONS_COUNT,
            },
        ),
        TrueEvaluationStage: (
            {PROPOSALS_OFFSPRING},
            {PROPOSALS_OFFSPRING, EVALUATED_OFFSPRING, EVALUATIONS_COUNT},
        ),
        ArchiveUpdateStage: (
            {
                EVALUATED_OFFSPRING,
                ARCHIVES_MAIN,
                ARCHIVES_PARETO,
                EVALUATIONS_PLAN,
                EVALUATIONS_PLAN_STATE,
            },
            {ARCHIVES_MAIN, ARCHIVES_PARETO, EVALUATED_OFFSPRING},
        ),
        FeedbackStage: (
            {
                PROPOSALS_OFFSPRING,
                EVALUATED_OFFSPRING,
                EVALUATION_NEW_IDS,
                SURROGATES_PREDICTIONS,
                EVALUATIONS_PLAN,
                EVALUATIONS_PLAN_STATE,
            },
            {PROPOSALS_OFFSPRING, FEEDBACK_RESULT},
        ),
        TellStage: (
            {
                PROPOSALS_OFFSPRING,
                PROPOSALS_CURRENT,
                FEEDBACK_RESULT,
                EVALUATED_OFFSPRING,
                EVALUATIONS_PLAN,
                EVALUATIONS_PLAN_STATE,
                EVALUATION_UPDATES,
            },
            set(),
        ),
        SurrogateOnlyLoopStage: ({ARCHIVES_MAIN}, set()),
        InitializationStage: (set(), set()),
    }
    for stage in OPERATIONAL_STAGE_INSTANCES:
        stage_type = type(stage)
        assert "contract" in stage_type.__dict__, stage_type.__name__
        contract = stage.contract()
        assert isinstance(contract, ComponentContract)
        expected_reads, expected_writes = expected[stage_type]
        assert set(contract.state.reads) == expected_reads, stage_type.__name__
        assert set(contract.state.writes) == expected_writes, stage_type.__name__
        for part in contract.parts:
            assert getattr(stage, part.name) is not None


def test_real_stage_instances_expose_the_same_contract_seam() -> None:
    """At least the constructor path, not only synthetic objects, is covered."""
    assert all(
        isinstance(stage.contract(), ComponentContract)
        for stage in OPERATIONAL_STAGE_INSTANCES
    )


def test_transient_contract_keys_are_boundary_vocabulary_only() -> None:
    """Transient stage fields have identity without becoming StateStore fields."""
    from saealib.context import _STORE_FIELDS

    assert EVALUATION_REQUEST not in _STORE_FIELDS.values()
    assert EVALUATION_UPDATES not in _STORE_FIELDS.values()
    assert EVALUATION_REQUEST.namespace == "evaluations"
    assert EVALUATION_UPDATES.name == "updates"


def test_held_component_contract_is_a_part_not_the_stage_state_contract() -> None:
    stage = next(
        stage for stage in OPERATIONAL_STAGE_INSTANCES if isinstance(stage, AskStage)
    )
    ask = stage.contract()
    assert ask.parts
    assert ask.parts[0].name == "_algorithm"
    assert getattr(stage, ask.parts[0].name) is not None
    assert PROPOSALS_OFFSPRING in ask.state.writes
    assert ask.parts[0].contract.state != ask.state


def test_operational_inventory_contains_only_real_stage_types() -> None:
    assert all(issubclass(stage_type, Stage) for stage_type in OPERATIONAL_STAGES)


def test_existing_pseudocode_remains_name_and_notation_stable() -> None:
    stage = CountGenerationStage()
    assert stage.name == "count_generation"
    assert stage.to_pseudocode() == r"\State $gen \leftarrow gen + 1$"
