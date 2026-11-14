from __future__ import annotations

from typing import Any, cast

from saealib.core.contracts import (
    RUNTIME_CAPABILITIES,
    ExecutionContract,
    Fixed,
    StateContract,
    Var,
)
from saealib.core.state import (
    ARCHIVES_MAIN,
    ARCHIVES_PARETO,
    EVALUATIONS_COUNT,
    EVALUATIONS_OWNERS,
    EVALUATIONS_PLAN,
    EVALUATIONS_PLAN_STATE,
    EVALUATIONS_PLAN_UPDATES,
    PENDING_EVALUATIONS,
    POPULATIONS_MAIN,
    PROPOSALS_CURRENT,
    RUNTIME_ASYNC_FATAL,
    RUNTIME_BASE_COLUMNS,
    RUNTIME_CANDIDATE_ID_ALLOCATOR,
    RUNTIME_DECISION_COUNT,
    RUNTIME_GENERATION,
    RUNTIME_REGION_COUNTS,
    RUNTIME_RNG,
    RUNTIME_STEP,
)
from saealib.decomposition import (
    Decomposition,
    PBIDecomposition,
    TchebycheffDecomposition,
    WeightedSumDecomposition,
)
from saealib.execution.evaluator import SerialEvaluator
from saealib.execution.initializer import (
    Initializer,
    LHSInitializer,
    RandomInitializer,
    SobolInitializer,
)
from saealib.execution.scheduler import AsyncEvaluationScheduler
from saealib.termination import (
    Termination,
    f_target,
    max_fe,
    max_gen,
    max_steps,
    stalled,
)


def test_pymoo_partial_tell_capability_is_configuration_dependent() -> None:
    from saealib.algorithms.pymoo_algorithm import PymooAlgorithm

    baseline = PymooAlgorithm(cast(Any, object()), allow_partial_tell=False).contract()
    partial = PymooAlgorithm(cast(Any, object()), allow_partial_tell=True).contract()

    assert baseline.execution == ExecutionContract()
    assert partial.execution == ExecutionContract(
        required_runtime_capabilities=("partial_feedback",)
    )
    assert partial.ports == baseline.ports
    assert partial.state == baseline.state


def test_async_scheduler_does_not_declare_partial_feedback_offer() -> None:
    contract = AsyncEvaluationScheduler(SerialEvaluator()).contract()

    assert contract.execution.offered_runtime_capabilities == ()


def test_only_partial_feedback_runtime_capability_is_registered() -> None:
    assert RUNTIME_CAPABILITIES.names() == ("partial_feedback",)


def test_initializer_contract_declares_population_and_rng_effects() -> None:
    contract = LHSInitializer(5, 5).contract()
    output = contract.ports["initializer"].outputs[0]

    assert output.name == "population"
    assert output.data.kind == "Population"
    assert output.data.bindings["candidate_count"] == Fixed(value=5)
    assert contract.state == StateContract(
        reads=(
            RUNTIME_RNG,
            EVALUATIONS_COUNT,
            RUNTIME_GENERATION,
            RUNTIME_DECISION_COUNT,
            POPULATIONS_MAIN,
            ARCHIVES_MAIN,
            ARCHIVES_PARETO,
            RUNTIME_CANDIDATE_ID_ALLOCATOR,
        ),
        writes=(
            POPULATIONS_MAIN,
            ARCHIVES_MAIN,
            EVALUATIONS_COUNT,
            ARCHIVES_PARETO,
            RUNTIME_GENERATION,
            RUNTIME_STEP,
            RUNTIME_REGION_COUNTS,
            RUNTIME_RNG,
            RUNTIME_BASE_COLUMNS,
            RUNTIME_CANDIDATE_ID_ALLOCATOR,
        ),
    )
    for initializer in (
        RandomInitializer(5, 5),
        SobolInitializer(5, 5),
    ):
        assert initializer.contract().state == contract.state
    base_contract = Initializer.contract(LHSInitializer(5, 5))
    assert base_contract.ports["initializer"].outputs[0].data.bindings[
        "candidate_count"
    ] == Var(name="N")
    assert LHSInitializer.contract is not Initializer.contract
    assert RandomInitializer.contract is not Initializer.contract
    assert SobolInitializer.contract is not Initializer.contract


def test_decomposition_contract_declares_objective_to_score_flow() -> None:
    contract = PBIDecomposition().contract()
    role = contract.ports["decomposition"]

    assert role.inputs[0].name == "objectives"
    assert role.inputs[0].data.kind == "FeatureBatch"
    assert isinstance(role.inputs[0].data.bindings["objective_schema"], Var)
    assert role.outputs[0].name == "scores"
    assert role.outputs[0].data.kind == "RowPredicate"
    assert contract.state == StateContract()
    assert WeightedSumDecomposition.contract is Decomposition.contract
    assert TchebycheffDecomposition.contract is Decomposition.contract
    assert PBIDecomposition.contract is Decomposition.contract


def test_termination_contract_tracks_registered_condition_state() -> None:
    assert Termination(max_fe(10)).contract().state == StateContract(
        reads=(EVALUATIONS_COUNT,)
    )
    assert Termination(max_gen(10)).contract().state == StateContract(
        reads=(RUNTIME_GENERATION,)
    )
    assert Termination(max_steps(10)).contract().state == StateContract(
        reads=(RUNTIME_STEP,)
    )
    assert Termination(f_target(0.0)).contract().state == StateContract(
        reads=(ARCHIVES_MAIN,)
    )
    assert Termination(stalled(2)).contract().state == StateContract(
        reads=(ARCHIVES_MAIN, RUNTIME_GENERATION)
    )
    assert Termination(max_fe(10)).contract().ports == {}


def test_termination_contract_is_conservative_for_unknown_callables() -> None:
    called = False

    def condition(_ctx):
        nonlocal called
        called = True
        return False

    assert Termination(condition).contract().state == StateContract(
        reads=(EVALUATIONS_COUNT, RUNTIME_GENERATION, RUNTIME_STEP, ARCHIVES_MAIN),
        reads_enumerable=False,
    )
    assert not called


def test_evaluator_and_scheduler_contracts_declare_lifecycle_boundary() -> None:
    evaluator = SerialEvaluator()
    role = evaluator.contract().ports["evaluator"]

    assert role.inputs[0].name == "genomes"
    assert role.inputs[0].data.kind == "GenomeBatch"
    assert role.outputs[0].name == "observations"
    assert role.outputs[0].data.kind == "ObservationBatch"

    scheduler_contract = AsyncEvaluationScheduler(evaluator).contract()
    assert scheduler_contract.state == StateContract(
        reads=(
            PENDING_EVALUATIONS,
            EVALUATIONS_OWNERS,
            EVALUATIONS_PLAN,
            EVALUATIONS_PLAN_STATE,
            EVALUATIONS_PLAN_UPDATES,
            EVALUATIONS_COUNT,
            ARCHIVES_MAIN,
            ARCHIVES_PARETO,
            PROPOSALS_CURRENT,
        ),
        writes=(
            PENDING_EVALUATIONS,
            EVALUATIONS_OWNERS,
            EVALUATIONS_PLAN,
            EVALUATIONS_PLAN_STATE,
            EVALUATIONS_PLAN_UPDATES,
            EVALUATIONS_COUNT,
            ARCHIVES_MAIN,
            ARCHIVES_PARETO,
            RUNTIME_ASYNC_FATAL,
        ),
    )
    algorithm_scheduler = AsyncEvaluationScheduler(evaluator, algorithm=object())
    assert algorithm_scheduler.contract().state == StateContract(
        reads=(
            PENDING_EVALUATIONS,
            EVALUATIONS_OWNERS,
            EVALUATIONS_PLAN,
            EVALUATIONS_PLAN_STATE,
            EVALUATIONS_PLAN_UPDATES,
            EVALUATIONS_COUNT,
            ARCHIVES_MAIN,
            ARCHIVES_PARETO,
            PROPOSALS_CURRENT,
            POPULATIONS_MAIN,
            RUNTIME_RNG,
        ),
        writes=(
            PENDING_EVALUATIONS,
            EVALUATIONS_OWNERS,
            EVALUATIONS_PLAN,
            EVALUATIONS_PLAN_STATE,
            EVALUATIONS_PLAN_UPDATES,
            EVALUATIONS_COUNT,
            ARCHIVES_MAIN,
            ARCHIVES_PARETO,
            RUNTIME_ASYNC_FATAL,
            POPULATIONS_MAIN,
            RUNTIME_RNG,
        ),
    )
    assert tuple(part.name for part in scheduler_contract.parts) == ("evaluator",)
    assert scheduler_contract.execution.offered_runtime_capabilities == ()
