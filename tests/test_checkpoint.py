"""Tests for seed, checkpoint, and resume features."""

from __future__ import annotations

import json
import pickle
import warnings
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
from _evaluation_helpers import _ControlledEvaluator

from saealib import (
    GA,
    PSO,
    CheckpointCallback,
    CrossoverBLXAlpha,
    LHSInitializer,
    MutationUniform,
    Optimizer,
    PreSelectionStrategy,
    SequentialSelection,
    Termination,
    TruncationSelection,
    max_gen,
)
from saealib.acquisition import AcquisitionResult, MeanPrediction
from saealib.callback import (
    CallbackManager,
    GenerationEndEvent,
    PostAskEvent,
    PostEvaluationEvent,
    RunEndEvent,
)
from saealib.comparators import SingleObjectiveComparator
from saealib.context import (
    CURRENT_CHECKPOINT_SCHEMA_VERSION,
    OptimizationState,
    _encode_state_value,
)
from saealib.core.contracts import StateContract
from saealib.core.state import StatePatch
from saealib.core.state.keys import ACQUISITION_RESULT, PROPOSALS_CURRENT, StateKey
from saealib.exceptions import CheckpointError, ValidationError
from saealib.execution import (
    AsyncEvaluationScheduler,
    AsyncEvaluator,
    EvaluationErrorInfo,
    EvaluationHandle,
    EvaluationRequest,
    EvaluationResult,
    EvaluationStatus,
    EvaluationUpdate,
    Evaluator,
    PendingEvaluation,
    SerialEvaluator,
)
from saealib.execution.history import (
    History,
    record_evaluations,
    snapshot_generation_for_checkpoint,
)
from saealib.execution.initializer import GenomeInitializer
from saealib.execution.runtime import create_runtime, resolve_plan
from saealib.pipeline import Branch, Pipeline
from saealib.policies.evaluation import EvaluationPlan
from saealib.population.genome import DenseVectorBatch
from saealib.problem import Problem
from saealib.strategies import (
    DirectStrategy,
    GenerationBasedStrategy,
    IndividualBasedStrategy,
    SteadyStateStrategy,
)
from saealib.surrogate.manager import LocalSurrogateManager
from saealib.surrogate.rbf import RBFSurrogate
from saealib.surrogate.rbf_kernels import GaussianKernel

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

DIM = 2
N_INIT_ARCHIVE = 10
N_INIT_POP = 6


def _sphere(x: np.ndarray) -> np.ndarray:
    return np.array([np.sum(x**2)])


def _make_problem() -> Problem:
    return Problem(
        func=_sphere,
        dim=DIM,
        n_obj=1,
        direction=np.array([-1.0]),
        lb=[-5.0] * DIM,
        ub=[5.0] * DIM,
        comparator=SingleObjectiveComparator(),
    )


def _make_optimizer(
    problem: Problem, seed: int | None = None, n_gen: int = 4
) -> Optimizer:
    return (
        Optimizer(problem, seed=seed)
        .set_initializer(LHSInitializer(N_INIT_ARCHIVE, N_INIT_POP))
        .set_algorithm(
            GA(
                crossover=CrossoverBLXAlpha(prob=0.9, alpha=0.5),
                mutation=MutationUniform(prob_var=0.1),
                parent_selection=SequentialSelection(),
                survivor_selection=TruncationSelection(),
            )
        )
        .set_surrogate(RBFSurrogate(GaussianKernel()), n_neighbors=5)
        .set_strategy(IndividualBasedStrategy(evaluation_ratio=0.5))
        .set_termination(Termination(max_gen(n_gen)))
    )


# ---------------------------------------------------------------------------
# Seed
# ---------------------------------------------------------------------------


def test_seed_reproducibility():
    problem = _make_problem()
    ctx1 = _make_optimizer(problem, seed=42).run()
    ctx2 = _make_optimizer(problem, seed=42).run()
    np.testing.assert_array_equal(ctx1.archive.x, ctx2.archive.x)
    np.testing.assert_array_equal(ctx1.archive.f, ctx2.archive.f)


def test_set_seed_method():
    problem = _make_problem()
    opt = _make_optimizer(problem)
    opt.set_seed(99)
    assert opt.seed == 99


def test_seed_none_stored():
    problem = _make_problem()
    opt = Optimizer(problem)
    assert opt.seed is None


def test_lhs_uses_optimizer_seed_over_own():
    """Optimizer.seed takes priority over LHSInitializer.seed."""
    problem = _make_problem()

    opt_a = (
        Optimizer(problem, seed=7)
        .set_initializer(LHSInitializer(N_INIT_ARCHIVE, N_INIT_POP, seed=999))
        .set_algorithm(
            GA(
                crossover=CrossoverBLXAlpha(prob=0.9, alpha=0.5),
                mutation=MutationUniform(prob_var=0.1),
                parent_selection=SequentialSelection(),
                survivor_selection=TruncationSelection(),
            )
        )
        .set_surrogate(RBFSurrogate(GaussianKernel()), n_neighbors=5)
        .set_strategy(IndividualBasedStrategy(evaluation_ratio=0.5))
        .set_termination(Termination(max_gen(2)))
    )
    opt_b = _make_optimizer(problem, seed=7, n_gen=2)

    ctx_a = opt_a.run()
    ctx_b = opt_b.run()
    np.testing.assert_array_equal(ctx_a.archive.x, ctx_b.archive.x)


# ---------------------------------------------------------------------------
# OptimizationState.save / load (npz)
# ---------------------------------------------------------------------------


def test_npz_roundtrip(tmp_path):
    problem = _make_problem()
    ctx = _make_optimizer(problem, seed=0, n_gen=2).run()

    p = tmp_path / "ckpt.npz"
    ctx.save(p)
    assert p.exists()

    loaded = OptimizationState.load(p, problem)
    np.testing.assert_array_equal(loaded.archive.x, ctx.archive.x)
    np.testing.assert_array_equal(loaded.archive.f, ctx.archive.f)
    np.testing.assert_array_equal(loaded.population.x, ctx.population.x)
    assert loaded.fe == ctx.fe
    assert loaded.gen == ctx.gen
    assert ctx.decision_count > 0
    assert loaded.decision_count == ctx.decision_count


def test_npz_load_defaults_decision_count_when_absent(tmp_path):
    """A checkpoint without decision_count resumes with decision_count == 0."""
    problem = _make_problem()
    ctx = _make_optimizer(problem, seed=0, n_gen=2).run()
    assert ctx.decision_count > 0

    p = tmp_path / "ckpt.npz"
    ctx.save(p)
    raw = dict(np.load(p, allow_pickle=False).items())
    # Rename in place (rather than dropping the list entry) so every other
    # entry's index-derived "_entry_N__*" array keys stay aligned.
    entries = json.loads(bytes(raw["_state_entries"]).decode())
    for item in entries:
        if (
            item["key"]["namespace"] == "runtime"
            and item["key"]["name"] == "decision_count"
        ):
            item["key"]["name"] = "decision_count_absent"
            break
    else:
        raise AssertionError("runtime/decision_count entry was not found")
    raw["_state_entries"] = np.frombuffer(json.dumps(entries).encode(), dtype=np.uint8)
    np.savez(p, **raw)

    loaded = OptimizationState.load(p, problem)
    assert loaded.decision_count == 0
    assert loaded.gen == ctx.gen


def test_npz_extension_added(tmp_path):
    problem = _make_problem()
    ctx = _make_optimizer(problem, seed=0, n_gen=2).run()
    ctx.save(tmp_path / "ckpt")
    assert (tmp_path / "ckpt.npz").exists()


def test_npz_rng_state_preserved(tmp_path):
    problem = _make_problem()
    ctx = _make_optimizer(problem, seed=0, n_gen=2).run()
    state_before = ctx.rng.bit_generator.state

    p = tmp_path / "ckpt.npz"
    ctx.save(p)
    loaded = OptimizationState.load(p, problem)
    assert loaded.rng.bit_generator.state == state_before


def test_resumed_flag_npz(tmp_path):
    problem = _make_problem()
    ctx = _make_optimizer(problem, seed=0, n_gen=2).run()
    assert ctx.data.get("resumed", False) is False

    p = tmp_path / "ckpt.npz"
    ctx.save(p)
    loaded = OptimizationState.load(p, problem)
    assert loaded.data.get("resumed") is True


def _rewrite_npz(src: Path, dst: Path, **overrides) -> None:
    """Copy an npz checkpoint to ``dst``, replacing/removing the given keys.

    A value of ``None`` in ``overrides`` deletes that key entirely.
    """
    data = np.load(src, allow_pickle=False)
    rebuilt = dict(data.items())
    for key, value in overrides.items():
        if value is None:
            rebuilt.pop(key, None)
        else:
            rebuilt[key] = value
    np.savez(dst, **rebuilt)


def test_load_missing_schema_version_raises(tmp_path):
    problem = _make_problem()
    ctx = _make_optimizer(problem, seed=0, n_gen=2).run()
    p = tmp_path / "ckpt.npz"
    ctx.save(p)

    stripped = tmp_path / "ckpt_stripped.npz"
    _rewrite_npz(p, stripped, _checkpoint_schema_version=None)

    with pytest.raises(ValidationError, match="_checkpoint_schema_version"):
        OptimizationState.load(stripped, problem)


def test_load_wrong_schema_version_raises(tmp_path):
    problem = _make_problem()
    ctx = _make_optimizer(problem, seed=0, n_gen=2).run()
    p = tmp_path / "ckpt.npz"
    ctx.save(p)

    bad_version = tmp_path / "ckpt_bad_version.npz"
    _rewrite_npz(p, bad_version, _checkpoint_schema_version=np.array(99))

    with pytest.raises(ValidationError, match="Unsupported checkpoint schema version"):
        OptimizationState.load(bad_version, problem)


# ---------------------------------------------------------------------------
# Optimizer.save_pickle / load_pickle
# ---------------------------------------------------------------------------


def test_pickle_roundtrip(tmp_path):
    problem = _make_problem()
    opt = _make_optimizer(problem, seed=0, n_gen=2)
    ctx = opt.run()

    p = tmp_path / "ckpt.pkl"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        opt.save_pickle(ctx, p)
    assert p.exists()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        _opt2, ctx2 = Optimizer.load_pickle(p)

    np.testing.assert_array_equal(ctx2.archive.x, ctx.archive.x)
    assert ctx2.gen == ctx.gen
    np.testing.assert_array_equal(ctx2.archive.id, ctx.archive.id)
    assert ctx.candidate_id_allocator.next_value > 0
    assert (
        ctx2.candidate_id_allocator.next_value == ctx.candidate_id_allocator.next_value
    )
    assert ctx2.request_id_allocator.next_value == ctx.request_id_allocator.next_value


def test_resumed_flag_pickle(tmp_path):
    problem = _make_problem()
    opt = _make_optimizer(problem, seed=0, n_gen=2)
    ctx = opt.run()

    p = tmp_path / "ckpt.pkl"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        opt.save_pickle(ctx, p)
        _, ctx2 = Optimizer.load_pickle(p)
    assert ctx2.data.get("resumed") is True


def test_pickle_extension_added(tmp_path):
    problem = _make_problem()
    opt = _make_optimizer(problem, seed=0, n_gen=2)
    ctx = opt.run()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        opt.save_pickle(ctx, tmp_path / "ckpt")
    assert (tmp_path / "ckpt.pkl").exists()


# ---------------------------------------------------------------------------
# run_from / iterate_from
# ---------------------------------------------------------------------------


def test_run_from_npz_resumed_flag(tmp_path):
    problem = _make_problem()
    ctx_mid = _make_optimizer(problem, seed=42, n_gen=2).run()
    p = tmp_path / "ckpt.npz"
    ctx_mid.save(p)

    loaded = OptimizationState.load(p, problem)
    ctx_final = _make_optimizer(problem, seed=42, n_gen=4).run_from(loaded)
    assert ctx_final.data.get("resumed") is True
    assert ctx_final.gen == 4


def test_run_from_npz_matches_full_run(tmp_path):
    problem = _make_problem()

    # Full run: 4 generations
    ctx_full = _make_optimizer(problem, seed=42, n_gen=4).run()

    # Split: 2 + 2 generations
    ctx_mid = _make_optimizer(problem, seed=42, n_gen=2).run()
    p = tmp_path / "ckpt.npz"
    ctx_mid.save(p)

    ctx_loaded = OptimizationState.load(p, problem)
    ctx_resumed = _make_optimizer(problem, seed=42, n_gen=4).run_from(ctx_loaded)

    np.testing.assert_array_equal(ctx_full.archive.x, ctx_resumed.archive.x)
    np.testing.assert_array_equal(ctx_full.archive.f, ctx_resumed.archive.f)


def test_run_from_pickle_matches_full_run(tmp_path):
    problem = _make_problem()

    ctx_full = _make_optimizer(problem, seed=42, n_gen=4).run()

    opt_mid = _make_optimizer(problem, seed=42, n_gen=2)
    ctx_mid = opt_mid.run()
    p = tmp_path / "ckpt.pkl"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        opt_mid.save_pickle(ctx_mid, p)
        opt_resume, ctx_loaded = Optimizer.load_pickle(p)

    opt_resume.set_termination(Termination(max_gen(4)))
    ctx_resumed = opt_resume.run_from(ctx_loaded)

    np.testing.assert_array_equal(ctx_full.archive.x, ctx_resumed.archive.x)
    np.testing.assert_array_equal(ctx_full.archive.f, ctx_resumed.archive.f)


def _assert_history_equal(
    actual: OptimizationState,
    expected: OptimizationState,
    *,
    in_flight_request_ids=(),
    portable_checkpoint=False,
):
    assert actual.history is not None
    assert expected.history is not None
    assert actual.history.enabled == expected.history.enabled
    for channel in sorted(expected.history.enabled):
        actual_columns = actual.history.channel(channel)
        expected_columns = expected.history.channel(channel)
        assert actual_columns.keys() == expected_columns.keys()
        for name, expected_values in expected_columns.items():
            if (
                portable_checkpoint
                and name in {"origin_decision_count", "origin_gen"}
                and in_flight_request_ids
            ):
                request_ids = actual_columns.get("request_id")
                if request_ids is not None:
                    mask = np.isin(request_ids, tuple(in_flight_request_ids))
                    expected_values = np.array(expected_values, copy=True)
                    expected_values[mask] = -1
            np.testing.assert_array_equal(actual_columns[name], expected_values)


def _assert_run_results_equal(
    actual: OptimizationState,
    expected: OptimizationState,
    *,
    in_flight_request_ids=(),
    portable_checkpoint=False,
):
    assert (actual.gen, actual.fe) == (expected.gen, expected.fe)
    for collection_name in ("population", "archive"):
        actual_collection = getattr(actual, collection_name)
        expected_collection = getattr(expected, collection_name)
        for column in ("id", "x", "f", "g", "cv", "objective_source"):
            np.testing.assert_array_equal(
                actual_collection.get_array(column),
                expected_collection.get_array(column),
            )
    _assert_history_equal(
        actual,
        expected,
        in_flight_request_ids=in_flight_request_ids,
        portable_checkpoint=portable_checkpoint,
    )


def _make_async_direct_optimizer(
    problem: Problem, *, seed: int, n_gen: int
) -> Optimizer:
    evaluator = SerialEvaluator()
    return (
        Optimizer(problem, seed=seed)
        .set_initializer(LHSInitializer(N_INIT_ARCHIVE, N_INIT_POP))
        .set_algorithm(
            GA(
                crossover=CrossoverBLXAlpha(prob=0.9, alpha=0.5),
                mutation=MutationUniform(prob_var=0.1),
                parent_selection=SequentialSelection(),
                survivor_selection=TruncationSelection(),
            )
        )
        .set_strategy(DirectStrategy())
        .set_evaluator(evaluator)
        .set_async_evaluation_scheduler(
            AsyncEvaluationScheduler(evaluator, max_pending=1)
        )
        .set_termination(Termination(max_gen(n_gen)))
    )


class _ReplayEvaluator(Evaluator):
    def __init__(self) -> None:
        self.unacknowledged: dict[int, EvaluationUpdate] = {}

    def evaluate_batch(self, x, problem):
        return SerialEvaluator().evaluate_batch(x, problem)

    def submit(self, request: EvaluationRequest, problem: Problem):
        return EvaluationHandle(
            request.request_id,
            EvaluationStatus.PENDING,
            backend_token=(request, problem),
        )

    def collect(self, handle: EvaluationHandle, *, wait: bool = True):
        if handle._acknowledged_sequence >= 0:
            return []
        request, problem = handle.backend_token
        request_id = int(request.request_id)
        update = self.unacknowledged.get(request_id)
        if update is None:
            result = SerialEvaluator().evaluate_request(request, problem)
            result.candidate_ids = request.candidate_ids
            result.__post_init__()
            update = EvaluationUpdate(
                request.request_id,
                EvaluationStatus.COMPLETED,
                request.candidate_ids,
                result,
                sequence=0,
            )
            self.unacknowledged[request_id] = update
        handle._delivered_sequence = update.sequence
        return [update]

    def acknowledge(self, handle: EvaluationHandle, sequence: int) -> None:
        handle._acknowledged_sequence = sequence
        self.unacknowledged.pop(int(handle.request_id), None)

    def supports_batch_rollback(self) -> bool:
        return True

    def can_reattach(self, pending) -> bool:
        return True

    def reattach(self, pending, problem: Problem):
        return self.submit(pending.request, problem)


class _RunEndCheckpointSaver:
    def __init__(self, checkpoint_format: str, path: Path) -> None:
        self.checkpoint_format = checkpoint_format
        self.path = path
        self.observed: tuple[int, int, int] | None = None
        self.error: str | None = None

    def __call__(self, event: RunEndEvent) -> None:
        state = cast(OptimizationState, event.ctx)
        self.observed = (state.fe, state.gen, len(state.pending_evaluations))
        optimizer = state.__dict__["_checkpoint_guard"].optimizer
        try:
            if self.checkpoint_format == "portable":
                state.save(self.path)
            elif self.checkpoint_format == "pickle":
                optimizer.save_pickle(state, self.path)
            else:
                optimizer.async_evaluation_scheduler.checkpoint(state, str(self.path))
        except CheckpointError as exc:
            self.error = str(exc)


def _make_async_generation_based_optimizer(
    problem: Problem,
    *,
    seed: int,
    n_gen: int,
    gen_ctrl: int,
    max_pending: int = 1,
) -> Optimizer:
    evaluator = _ReplayEvaluator()
    return (
        _make_optimizer(problem, seed=seed, n_gen=n_gen)
        .set_strategy(GenerationBasedStrategy(gen_ctrl))
        .set_evaluator(evaluator)
        .set_async_evaluation_scheduler(
            AsyncEvaluationScheduler(evaluator, max_pending=max_pending)
        )
        .set_history(["summary", "population", "evaluation"])
    )


def test_async_direct_pickle_checkpoint_matches_uninterrupted_run(tmp_path):
    problem = _make_problem()
    uninterrupted_optimizer = _make_async_direct_optimizer(
        problem, seed=42, n_gen=4
    ).set_history(["summary", "population", "evaluation"])
    uninterrupted = uninterrupted_optimizer.run()

    partial_optimizer = _make_async_direct_optimizer(
        problem, seed=42, n_gen=2
    ).set_history(["summary", "population", "evaluation"])
    partial = partial_optimizer.run()
    path = tmp_path / "async-direct-checkpoint.pkl"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        partial_optimizer.save_pickle(partial, path)
        resumed_optimizer, restored = Optimizer.load_pickle(path)

    resumed_optimizer.set_termination(Termination(max_gen(4)))
    resumed = resumed_optimizer.run_from(restored)

    _assert_run_results_equal(resumed, uninterrupted)
    np.testing.assert_array_equal(
        resumed.population.get_array("id"), uninterrupted.population.get_array("id")
    )


@pytest.mark.parametrize("gen_ctrl", [0, 1])
def test_async_generation_based_pickle_resume_with_inflight_request_matches_full_run(
    tmp_path, gen_ctrl
):
    problem = _make_problem()
    uninterrupted = _make_async_generation_based_optimizer(
        problem, seed=42, n_gen=4, gen_ctrl=gen_ctrl
    ).run()

    partial_optimizer = _make_async_generation_based_optimizer(
        problem, seed=42, n_gen=2, gen_ctrl=gen_ctrl
    )
    partial = partial_optimizer.run()
    assert partial.gen == 2
    assert len(partial.pending_evaluations) == 1
    proposal_id = partial.get_state(PROPOSALS_CURRENT)
    path = tmp_path / f"generation-based-{gen_ctrl}.pkl"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        partial_optimizer.save_pickle(partial, path)
        resumed_optimizer, restored = Optimizer.load_pickle(path)

    assert len(restored.pending_evaluations) == 1
    resumed_optimizer.set_termination(Termination(max_gen(4)))
    resumed = resumed_optimizer.run_from(restored)

    assert restored.get_state(PROPOSALS_CURRENT) == proposal_id
    _assert_run_results_equal(resumed, uninterrupted)


@pytest.mark.parametrize("gen_ctrl", [0, 1])
def test_async_generation_based_portable_resume_with_inflight_request_matches_full_run(
    tmp_path, gen_ctrl
):
    problem = _make_problem()
    uninterrupted = _make_async_generation_based_optimizer(
        problem, seed=42, n_gen=4, gen_ctrl=gen_ctrl
    ).run()

    partial_optimizer = _make_async_generation_based_optimizer(
        problem, seed=42, n_gen=2, gen_ctrl=gen_ctrl
    )
    partial = partial_optimizer.run()
    assert partial.gen == 2
    assert len(partial.pending_evaluations) == 1
    assert partial.evaluation_plan_state is not None
    # Deferred requests resume without their in-memory origin columns.
    in_flight_request_ids = set(partial.evaluation_plan_state.deferred)
    proposal_id = partial.get_state(PROPOSALS_CURRENT)
    path = tmp_path / f"generation-based-{gen_ctrl}.npz"
    partial.save(path)
    restored = OptimizationState.load(path, problem)
    assert len(restored.pending_evaluations) == 1

    resumed_optimizer = _make_async_generation_based_optimizer(
        problem, seed=42, n_gen=4, gen_ctrl=gen_ctrl
    )
    resumed = resumed_optimizer.run_from(restored)

    assert restored.get_state(PROPOSALS_CURRENT) == proposal_id
    _assert_run_results_equal(
        resumed,
        uninterrupted,
        in_flight_request_ids=in_flight_request_ids,
        portable_checkpoint=True,
    )


def test_evaluation_plan_pickle_roundtrip_preserves_immutable_artifacts():
    request = EvaluationRequest(
        np.int64(7),
        np.array([11, 12], dtype=np.int64),
        np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float64),
        metadata={"purpose": "evaluation"},
    )
    plan = EvaluationPlan(
        (request,),
        completion_rule={"mode": "all"},
        continuation={"next_request": 8},
        artifacts={"summary": {"count": 2, "complete": True}},
    )

    restored = pickle.loads(pickle.dumps(plan))

    assert len(restored.requests) == len(plan.requests)
    assert restored.requests[0].request_id == request.request_id
    np.testing.assert_array_equal(
        restored.requests[0].candidate_ids, request.candidate_ids
    )
    assert isinstance(restored.requests[0].payload, DenseVectorBatch)
    assert isinstance(request.payload, DenseVectorBatch)
    np.testing.assert_array_equal(
        restored.requests[0].payload.array, request.payload.array
    )
    assert restored.requests[0].outputs == request.outputs
    assert restored.requests[0].metadata == request.metadata
    assert restored.completion_rule == plan.completion_rule
    assert restored.continuation == plan.continuation
    assert restored.artifacts == plan.artifacts
    with pytest.raises(TypeError):
        restored.artifacts["extra"] = "value"


def test_iterate_from_yields_correct_gen(tmp_path):
    problem = _make_problem()
    ctx_mid = _make_optimizer(problem, seed=0, n_gen=2).run()
    p = tmp_path / "ckpt.npz"
    ctx_mid.save(p)

    loaded = OptimizationState.load(p, problem)
    gens = [ctx.gen for ctx in _make_optimizer(problem, n_gen=4).iterate_from(loaded)]
    assert gens[-1] == 4


@pytest.mark.parametrize(
    "strategy_name",
    ["direct", "generation_based", "individual_based", "preselection", "steady_state"],
)
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("checkpoint_format", ["portable", "pickle"])
def test_first_iterate_yield_can_be_saved_and_resumed(
    tmp_path, strategy_name, asynchronous, checkpoint_format
):
    problem = _make_problem()
    uninterrupted = _make_boundary_optimizer(
        problem, strategy_name, asynchronous=asynchronous
    )[0].run()
    optimizer, _ = _make_boundary_optimizer(
        problem, strategy_name, asynchronous=asynchronous
    )
    first = next(optimizer.iterate())
    assert first.gen == 0
    path = tmp_path / f"first-{strategy_name}-{asynchronous}"
    if checkpoint_format == "portable":
        first.save(path)
        restored = OptimizationState.load(path.with_suffix(".npz"), problem)
        resumed_optimizer = _make_boundary_optimizer(
            problem, strategy_name, asynchronous=asynchronous
        )[0]
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            optimizer.save_pickle(first, path)
            resumed_optimizer, restored = Optimizer.load_pickle(
                path.with_suffix(".pkl")
            )
    resumed = resumed_optimizer.run_from(restored)
    _assert_run_results_equal(resumed, uninterrupted)


@pytest.mark.parametrize(
    "strategy_name",
    ["direct", "generation_based", "individual_based", "preselection", "steady_state"],
)
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("checkpoint_format", ["portable", "pickle"])
def test_first_iterate_from_yield_can_be_saved_and_resumed(
    tmp_path, strategy_name, asynchronous, checkpoint_format
):
    problem = _make_problem()
    uninterrupted = _make_boundary_optimizer(
        problem, strategy_name, asynchronous=asynchronous
    )[0].run()
    optimizer, _ = _make_boundary_optimizer(
        problem, strategy_name, asynchronous=asynchronous
    )
    assert optimizer.initializer is not None
    initial = optimizer.initializer.initialize(optimizer, problem)
    first = next(optimizer.iterate_from(initial))
    assert first.gen == 0
    path = tmp_path / f"from-{strategy_name}-{asynchronous}"
    if checkpoint_format == "portable":
        first.save(path)
        restored = OptimizationState.load(path.with_suffix(".npz"), problem)
        resumed_optimizer = _make_boundary_optimizer(
            problem, strategy_name, asynchronous=asynchronous
        )[0]
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            optimizer.save_pickle(first, path)
            resumed_optimizer, restored = Optimizer.load_pickle(
                path.with_suffix(".pkl")
            )
    resumed = resumed_optimizer.run_from(restored)
    _assert_run_results_equal(resumed, uninterrupted)


def test_acquisition_result_without_scores_is_checkpoint_error():
    with pytest.raises(CheckpointError, match="scores and artifacts"):
        _encode_state_value(ACQUISITION_RESULT, object(), {}, "bad")


# ---------------------------------------------------------------------------
# CheckpointCallback
# ---------------------------------------------------------------------------


def test_checkpoint_callback_files_created(tmp_path):
    problem = _make_problem()
    cb = CheckpointCallback(tmp_path / "ckpts", interval=1, format="npz")
    opt = _make_optimizer(problem, seed=0, n_gen=3)
    cb.register(opt.cbmanager)
    opt.run()

    npz_files = sorted((tmp_path / "ckpts").glob("checkpoint_*.npz"))
    assert len(npz_files) == 3


def test_checkpoint_callback_interval(tmp_path):
    problem = _make_problem()
    cb = CheckpointCallback(tmp_path / "ckpts", interval=2, format="npz")
    opt = _make_optimizer(problem, seed=0, n_gen=4)
    cb.register(opt.cbmanager)
    opt.run()

    npz_files = sorted((tmp_path / "ckpts").glob("checkpoint_*.npz"))
    assert len(npz_files) == 2


def test_checkpoint_callback_delete_on_success(tmp_path):
    problem = _make_problem()
    cb = CheckpointCallback(
        tmp_path / "ckpts", interval=1, format="npz", delete_on_success=True
    )
    opt = _make_optimizer(problem, seed=0, n_gen=3)
    cb.register(opt.cbmanager)
    opt.run()

    npz_files = list((tmp_path / "ckpts").glob("checkpoint_*.npz"))
    assert len(npz_files) == 0


def test_checkpoint_callback_both_formats(tmp_path):
    problem = _make_problem()
    opt = _make_optimizer(problem, seed=0, n_gen=2)
    cb = CheckpointCallback(
        tmp_path / "ckpts", interval=1, format="both", optimizer=opt
    )
    cb.register(opt.cbmanager)
    opt.run()

    npz_files = list((tmp_path / "ckpts").glob("checkpoint_*.npz"))
    pkl_files = list((tmp_path / "ckpts").glob("checkpoint_*.pkl"))
    assert len(npz_files) == 2
    assert len(pkl_files) == 2


def test_checkpoint_callback_both_prevalidates_unpicklable_evaluator(tmp_path):
    problem = _make_problem()
    optimizer = _make_async_direct_optimizer(problem, seed=3, n_gen=1)
    evaluator = AsyncEvaluator(SerialEvaluator())
    optimizer.set_evaluator(evaluator).set_async_evaluation_scheduler(
        AsyncEvaluationScheduler(evaluator, max_pending=2)
    )
    checkpoint_dir = tmp_path / "both"
    callback = CheckpointCallback(
        checkpoint_dir, interval=1, format="both", optimizer=optimizer
    )
    callback.register(optimizer.cbmanager)

    with (
        warnings.catch_warnings(),
        pytest.raises(CheckpointError, match="AsyncEvaluator"),
    ):
        warnings.simplefilter("ignore", UserWarning)
        optimizer.run()

    assert not checkpoint_dir.exists()
    assert not list(tmp_path.glob("*.npz"))
    assert not list(tmp_path.glob("*.pkl"))


def test_portable_checkpoint_does_not_pickle_unpicklable_evaluator(tmp_path):
    problem = _make_problem()
    optimizer = _make_async_direct_optimizer(problem, seed=3, n_gen=1)
    evaluator = AsyncEvaluator(SerialEvaluator())
    optimizer.set_evaluator(evaluator).set_async_evaluation_scheduler(
        AsyncEvaluationScheduler(evaluator, max_pending=2)
    )
    state = optimizer.run()
    path = tmp_path / "portable-unpicklable.npz"

    state.save(path)

    assert path.exists()
    restored = OptimizationState.load(path, problem)
    assert (restored.gen, restored.fe) == (state.gen, state.fe)


def test_post_run_component_change_error_describes_configuration_change(tmp_path):
    problem = _make_problem()
    optimizer = _make_optimizer(problem, seed=19, n_gen=2)
    state = optimizer.run()
    optimizer.set_algorithm(
        GA(
            crossover=CrossoverBLXAlpha(prob=0.9, alpha=0.5),
            mutation=MutationUniform(prob_var=0.1),
            parent_selection=SequentialSelection(),
            survivor_selection=TruncationSelection(),
        )
    )
    path = tmp_path / "changed-after-run.npz"

    with pytest.raises(CheckpointError, match="configuration changed after"):
        state.save(path)

    assert not path.exists()


def test_run_with_checkpoint_path(tmp_path):
    problem = _make_problem()
    opt = _make_optimizer(problem, seed=0, n_gen=3)
    opt.run(checkpoint_path=tmp_path / "ckpts", checkpoint_interval=1)

    npz_files = sorted((tmp_path / "ckpts").glob("checkpoint_*.npz"))
    assert len(npz_files) == 3


def test_checkpoint_callback_invalid_format():
    with pytest.raises(ValueError, match="format must be"):
        CheckpointCallback("/tmp", format="json")


def test_checkpoint_callback_pickle_requires_optimizer():
    with pytest.raises(ValueError, match="optimizer must be provided"):
        CheckpointCallback("/tmp", format="pickle")


# ---------------------------------------------------------------------------
# Checkpoints with and without the objective_source column
# ---------------------------------------------------------------------------

UNKNOWN_SOURCE = 255


def _make_pso_optimizer(
    problem: Problem, n_gen: int, channels: tuple[str, ...]
) -> Optimizer:
    return (
        Optimizer(problem, seed=7)
        .set_initializer(LHSInitializer(N_INIT_ARCHIVE, N_INIT_POP))
        .set_algorithm(PSO())
        .set_surrogate(RBFSurrogate(GaussianKernel()), n_neighbors=5)
        .set_strategy(IndividualBasedStrategy(evaluation_ratio=0.25))
        .set_history(list(channels))
        .set_termination(Termination(max_gen(n_gen)))
    )


def _without_objective_source(value):
    if isinstance(value, list):
        return [
            _without_objective_source(item)
            for item in value
            if not (isinstance(item, dict) and item.get("name") == "objective_source")
        ]
    if isinstance(value, dict):
        return {key: _without_objective_source(item) for key, item in value.items()}
    return value


def _write_without_objective_source(src: Path, dst: Path) -> None:
    """Remove the source column from a checkpoint written by the current reader."""
    data = np.load(src, allow_pickle=False)
    arrays = {
        key: data[key] for key in data.files if not key.endswith("__objective_source")
    }
    entries = json.loads(bytes(arrays["_state_entries"]).decode())
    arrays["_state_entries"] = np.frombuffer(
        json.dumps(_without_objective_source(entries), allow_nan=True).encode(),
        dtype=np.uint8,
    )
    arrays["_checkpoint_schema_version"] = np.array(1, dtype=np.int64)
    np.savez(dst, **arrays)


def _sources(population) -> np.ndarray:
    return population.get_array("objective_source")


def _checkpoint_without_objective_source(tmp_path: Path, channels: tuple[str, ...]):
    problem = _make_problem()
    state = _make_pso_optimizer(problem, 3, channels).run()
    written = tmp_path / "current.npz"
    state.save(written)
    without_source = tmp_path / "without-objective-source.npz"
    _write_without_objective_source(written, without_source)
    assert int(np.load(without_source)["_checkpoint_schema_version"]) == 1
    return problem, state, without_source


def _write_without_population_objective_source(src: Path, dst: Path) -> None:
    data = np.load(src, allow_pickle=False)
    arrays = {key: data[key] for key in data.files}
    entries = json.loads(bytes(arrays["_state_entries"]).decode())
    for index, item in enumerate(entries):
        key = item["key"]
        if key["namespace"] == "populations" and key["name"] == "main":
            descriptor = item["value"]["descriptor"]
            descriptor["schema"] = [
                attr
                for attr in descriptor["schema"]
                if attr["name"] != "objective_source"
            ]
            arrays.pop(f"_entry_{index}__objective_source")
            break
    else:
        raise AssertionError("populations/main entry was not found")
    arrays["_state_entries"] = np.frombuffer(
        json.dumps(entries, allow_nan=True).encode(), dtype=np.uint8
    )
    np.savez(dst, **arrays)


class _PartialThenEmptyCompletionEvaluator(Evaluator):
    def evaluate_batch(self, x, problem):
        return SerialEvaluator().evaluate_batch(x, problem)

    def submit(self, request, problem):
        return EvaluationHandle(
            request.request_id,
            EvaluationStatus.PENDING,
            backend_token=(request, problem),
        )

    def collect(self, handle, *, wait=True):
        if handle._acknowledged_sequence >= 0:
            return []
        request, problem = handle.backend_token
        result = SerialEvaluator().evaluate_batch(request.x, problem)
        result.candidate_ids = request.candidate_ids
        result.__post_init__()
        handle._delivered_sequence = 1
        return [
            EvaluationUpdate(
                request.request_id,
                EvaluationStatus.PARTIAL,
                request.candidate_ids,
                result,
                sequence=0,
            ),
            EvaluationUpdate(
                request.request_id,
                EvaluationStatus.COMPLETED,
                np.empty(0, dtype=np.int64),
                sequence=1,
            ),
        ]

    def acknowledge(self, handle, sequence):
        handle._acknowledged_sequence = sequence


def test_checkpoint_without_objective_source_loads_empty_initial_evaluation(tmp_path):
    problem = _make_problem()
    state = (
        Optimizer(problem, seed=7)
        .set_initializer(GenomeInitializer(0, 0))
        .set_strategy(DirectStrategy())
        .set_history(["evaluation"])
        .set_termination(Termination(max_gen(0)))
        .run()
    )
    written = tmp_path / "empty-initial.npz"
    state.save(written)
    checkpoint = tmp_path / "empty-initial-without-source.npz"
    _write_without_objective_source(written, checkpoint)

    loaded = OptimizationState.load(checkpoint, problem)

    assert len(loaded.population) == 0
    assert loaded.population.schema["objective_source"].dtype == np.uint8
    assert len(loaded.archive) == 0


def test_checkpoint_without_objective_source_loads_empty_completion_after_partial(
    tmp_path,
):
    problem = _make_problem()
    evaluator = _PartialThenEmptyCompletionEvaluator()
    state = (
        Optimizer(problem, seed=7)
        .set_initializer(LHSInitializer(6, 2))
        .set_strategy(DirectStrategy())
        .set_evaluator(evaluator)
        .set_async_evaluation_scheduler(
            AsyncEvaluationScheduler(evaluator, max_pending=1)
        )
        .set_history(["evaluation"])
        .set_termination(Termination(max_gen(1)))
        .run()
    )
    written = tmp_path / "empty-completion.npz"
    state.save(written)
    checkpoint = tmp_path / "empty-completion-without-source.npz"
    _write_without_objective_source(written, checkpoint)

    loaded = OptimizationState.load(checkpoint, problem)

    assert loaded.history is not None
    assert np.any(loaded.history.channel("evaluation")["status_code"] == 3)
    assert np.all(_sources(loaded.population) == 0)


def test_checkpoint_without_objective_source_marks_matching_rows_true(
    tmp_path,
):
    problem, state, checkpoint = _checkpoint_without_objective_source(
        tmp_path, ("summary", "evaluation")
    )
    sources = _sources(state.population)
    assert set(sources.tolist()) == {0, 1}

    loaded = OptimizationState.load(checkpoint, problem)

    np.testing.assert_array_equal(
        _sources(loaded.population), np.where(sources == 0, 0, UNKNOWN_SOURCE)
    )
    assert np.all(_sources(loaded.archive) == 0)
    assert np.all(_sources(loaded.pareto_archive) == 0)


def test_checkpoint_without_objective_source_or_evaluation_channel_marks_rows_unknown(
    tmp_path,
):
    problem, _state, checkpoint = _checkpoint_without_objective_source(
        tmp_path, ("summary",)
    )

    loaded = OptimizationState.load(checkpoint, problem)

    assert np.all(_sources(loaded.population) == UNKNOWN_SOURCE)
    assert np.all(_sources(loaded.archive) == UNKNOWN_SOURCE)


@pytest.mark.parametrize(
    "channels", [("summary", "evaluation"), ("summary",)], ids=["evaluation", "none"]
)
def test_checkpoint_without_objective_source_resumes_like_an_uninterrupted_run(
    tmp_path, channels
):
    problem, _state, checkpoint = _checkpoint_without_objective_source(
        tmp_path, channels
    )
    uninterrupted = _make_pso_optimizer(problem, 6, channels).run()

    resumed = _make_pso_optimizer(problem, 6, channels).run_from(
        OptimizationState.load(checkpoint, problem)
    )

    assert (resumed.gen, resumed.fe) == (uninterrupted.gen, uninterrupted.fe)
    np.testing.assert_array_equal(resumed.population.id, uninterrupted.population.id)
    np.testing.assert_array_equal(resumed.population.f, uninterrupted.population.f)
    np.testing.assert_array_equal(resumed.archive.id, uninterrupted.archive.id)
    np.testing.assert_array_equal(
        _sources(resumed.population), _sources(uninterrupted.population)
    )


def _record(state, request_id, status, candidate_id, f=None) -> None:
    result = None
    error = None
    if f is None:
        error = EvaluationErrorInfo("RuntimeError", "evaluation failed")
    else:
        result = EvaluationResult(
            np.atleast_2d(f),
            np.zeros((1, 0)),
            np.zeros(1),
            candidate_ids=np.array([candidate_id]),
        )
    update = EvaluationUpdate(
        np.int64(request_id), status, np.array([candidate_id]), result, error, 0
    )
    state.history._observe_evaluation(update, 0)
    record_evaluations(state)


def test_checkpoint_uses_each_collection_schema_for_objective_source(tmp_path):
    problem = _make_problem()
    state = _make_pso_optimizer(problem, 3, ("summary", "evaluation")).run()
    stored_archive_sources = np.full(len(state.archive), 2, dtype=np.uint8)
    stored_pareto_sources = np.full(len(state.pareto_archive), 255, dtype=np.uint8)
    state.archive.update_array("objective_source", stored_archive_sources)
    state.pareto_archive.update_array("objective_source", stored_pareto_sources)
    written = tmp_path / "current.npz"
    state.save(written)
    without_population_source = tmp_path / "without-population-source.npz"
    _write_without_population_objective_source(written, without_population_source)

    loaded = OptimizationState.load(without_population_source, problem)

    np.testing.assert_array_equal(
        _sources(loaded.population), np.where(_sources(state.population) == 0, 0, 255)
    )
    np.testing.assert_array_equal(_sources(loaded.archive), stored_archive_sources)
    np.testing.assert_array_equal(
        _sources(loaded.pareto_archive), stored_pareto_sources
    )


@pytest.mark.parametrize(
    "status", [EvaluationStatus.FAILED, EvaluationStatus.CANCELLED]
)
def test_checkpoint_counts_result_bearing_failed_or_cancelled_evaluations(
    tmp_path, status
):
    problem = _make_problem()
    state = _make_pso_optimizer(problem, 3, ("summary", "evaluation")).run()
    ids = state.population.get_array("id")
    f = state.population.get_array("f")
    predicted = np.flatnonzero(_sources(state.population) == 1)
    assert len(predicted) >= 2
    result_row, placeholder_row = predicted[:2]
    _record(state, 9101, status, ids[result_row], f[result_row])
    _record(state, 9102, status, ids[placeholder_row])
    written = tmp_path / "current.npz"
    state.save(written)
    without_source = tmp_path / "without-objective-source.npz"
    _write_without_objective_source(written, without_source)

    loaded = _sources(OptimizationState.load(without_source, problem).population)

    assert loaded[result_row] == 0
    assert loaded[placeholder_row] == UNKNOWN_SOURCE


@pytest.mark.parametrize(
    "status", [EvaluationStatus.FAILED, EvaluationStatus.CANCELLED]
)
def test_checkpoint_uses_failed_or_cancelled_result_for_repeated_request_candidate(
    tmp_path, status
):
    problem = _make_problem()
    state = _make_pso_optimizer(problem, 3, ("summary", "evaluation")).run()
    row = int(np.flatnonzero(_sources(state.population) == 1)[0])
    candidate_id = int(state.population.get_array("id")[row])
    earlier_f = np.array([[-1.0]], dtype=np.float64)
    later_f = np.array([[-2.0]], dtype=np.float64)
    request_id = 9201

    _record(state, request_id, EvaluationStatus.PARTIAL, candidate_id, earlier_f)
    _record(state, request_id, status, candidate_id, later_f)
    state.population.update_rows(np.asarray([row], dtype=np.intp), {"f": later_f})
    assert state.history is not None
    evaluation = state.history.channel("evaluation")
    assert evaluation["fe_after"][-1] == evaluation["fe_before"][-1]

    written = tmp_path / "current.npz"
    state.save(written)
    without_source = tmp_path / "without-objective-source.npz"
    _write_without_objective_source(written, without_source)

    loaded = _sources(OptimizationState.load(without_source, problem).population)

    assert loaded[row] == 0


def test_checkpoint_matches_nan_true_evaluation(tmp_path):
    problem = _make_problem()
    state = _make_pso_optimizer(problem, 3, ("summary", "evaluation")).run()
    row = 0
    candidate_id = state.population.get_array("id")[row]
    state.population.update_rows(
        np.asarray([row], dtype=np.intp),
        {"f": np.array([[np.nan]], dtype=np.float64)},
    )
    _record(state, 9103, EvaluationStatus.COMPLETED, candidate_id, np.nan)
    written = tmp_path / "current.npz"
    state.save(written)
    without_source = tmp_path / "without-objective-source.npz"
    _write_without_objective_source(written, without_source)

    loaded = _sources(OptimizationState.load(without_source, problem).population)

    assert loaded[row] == 0


def test_checkpoint_without_objective_source_uses_the_last_true_evaluation(tmp_path):
    problem = _make_problem()
    state = _make_pso_optimizer(problem, 3, ("summary", "evaluation")).run()
    ids = state.population.get_array("id")
    f = state.population.get_array("f")
    sources = _sources(state.population)
    replaced = np.flatnonzero(sources == 0)[0]
    restored = np.flatnonzero(sources == 1)[0]
    _record(state, 9001, EvaluationStatus.COMPLETED, ids[replaced], f[replaced] + 1)
    _record(state, 9002, EvaluationStatus.COMPLETED, ids[restored], f[restored] + 1)
    _record(state, 9003, EvaluationStatus.PARTIAL, ids[restored], f[restored])
    _record(state, 9004, EvaluationStatus.FAILED, ids[restored])
    written = tmp_path / "current.npz"
    state.save(written)
    checkpoint = tmp_path / "without-objective-source.npz"
    _write_without_objective_source(written, checkpoint)

    loaded = _sources(OptimizationState.load(checkpoint, problem).population)

    assert loaded[replaced] == UNKNOWN_SOURCE
    assert loaded[restored] == 0


def test_npz_checkpoint_round_trips_objective_source_and_resumes(
    tmp_path,
):
    problem = _make_problem()
    state = _make_pso_optimizer(problem, 3, ("summary",)).run()
    path = tmp_path / "ckpt.npz"
    state.save(path)

    loaded = OptimizationState.load(path, problem)

    assert (
        int(np.load(path)["_checkpoint_schema_version"])
        == CURRENT_CHECKPOINT_SCHEMA_VERSION
    )
    assert set(_sources(state.population).tolist()) == {0, 1}
    np.testing.assert_array_equal(
        _sources(loaded.population), _sources(state.population)
    )
    np.testing.assert_array_equal(_sources(loaded.archive), _sources(state.archive))
    resumed = _make_pso_optimizer(problem, 6, ("summary",)).run_from(loaded)
    uninterrupted = _make_pso_optimizer(problem, 6, ("summary",)).run()
    np.testing.assert_array_equal(resumed.population.id, uninterrupted.population.id)
    np.testing.assert_array_equal(
        _sources(resumed.population), _sources(uninterrupted.population)
    )


def test_pickle_checkpoint_round_trips_objective_source(tmp_path):
    problem = _make_problem()
    optimizer = _make_pso_optimizer(problem, 3, ("summary",))
    state = optimizer.run()
    path = tmp_path / "ckpt.pkl"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        optimizer.save_pickle(state, path)
        _optimizer, loaded = Optimizer.load_pickle(path)

    assert set(_sources(state.population).tolist()) == {0, 1}
    np.testing.assert_array_equal(
        _sources(loaded.population), _sources(state.population)
    )


def _make_boundary_optimizer(
    problem: Problem,
    strategy_name: str,
    *,
    asynchronous: bool,
    evaluator: _ControlledEvaluator | None = None,
) -> tuple[Optimizer, _ControlledEvaluator | None]:
    strategy = {
        "direct": DirectStrategy,
        "generation_based": lambda: GenerationBasedStrategy(gen_ctrl=2),
        "individual_based": lambda: IndividualBasedStrategy(evaluation_ratio=0.5),
        "preselection": lambda: PreSelectionStrategy(6, 2),
        "steady_state": SteadyStateStrategy,
    }[strategy_name]()
    optimizer = (
        Optimizer(problem, seed=42)
        .set_initializer(LHSInitializer(N_INIT_ARCHIVE, N_INIT_POP))
        .set_algorithm(
            GA(
                crossover=CrossoverBLXAlpha(prob=0.9, alpha=0.5),
                mutation=MutationUniform(prob_var=0.1),
                parent_selection=SequentialSelection(),
                survivor_selection=TruncationSelection(),
            )
        )
        .set_strategy(strategy)
        .set_termination(Termination(max_gen(4)))
        .set_history(["summary", "population", "evaluation"])
    )
    if strategy_name in {"generation_based", "individual_based", "preselection"}:
        optimizer.set_surrogate_manager(
            LocalSurrogateManager(RBFSurrogate(GaussianKernel()))
        ).set_acquisition(MeanPrediction(direction=problem.direction))
    if asynchronous:
        evaluator = evaluator or _ControlledEvaluator()
        evaluator.release_all = True
        optimizer.set_evaluator(evaluator)
        optimizer.set_async_evaluation_scheduler(
            AsyncEvaluationScheduler(evaluator, max_pending=2)
        )
    return optimizer, evaluator


@pytest.mark.parametrize(
    "strategy_name",
    ["direct", "generation_based", "individual_based", "preselection", "steady_state"],
)
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("format_name", ["portable", "pickle"])
def test_step_boundary_checkpoint_resume_matches_uninterrupted_run(
    tmp_path, strategy_name, asynchronous, format_name
):
    problem = _make_problem()
    full_optimizer, full_evaluator = _make_boundary_optimizer(
        problem, strategy_name, asynchronous=asynchronous
    )
    uninterrupted = full_optimizer.run()

    partial_optimizer, partial_evaluator = _make_boundary_optimizer(
        problem, strategy_name, asynchronous=asynchronous
    )
    iterator = partial_optimizer.iterate()
    next(iterator)
    boundary = next(iterator)
    path = tmp_path / f"{strategy_name}-{asynchronous}-{format_name}"
    if format_name == "portable":
        boundary.save(path)
        restored = OptimizationState.load(path.with_suffix(".npz"), problem)
        resumed_optimizer, resumed_evaluator = _make_boundary_optimizer(
            problem, strategy_name, asynchronous=asynchronous
        )
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            partial_optimizer.save_pickle(boundary, path)
            resumed_optimizer, restored = Optimizer.load_pickle(
                path.with_suffix(".pkl")
            )
        resumed_evaluator = (
            cast(
                _ControlledEvaluator,
                resumed_optimizer.async_evaluation_scheduler.evaluator,
            )
            if asynchronous and resumed_optimizer.async_evaluation_scheduler is not None
            else None
        )

    resumed = resumed_optimizer.run_from(restored)

    _assert_run_results_equal(resumed, uninterrupted)
    if strategy_name == "steady_state" and asynchronous:
        assert boundary.feedback_accumulator is not None
        assert restored.feedback_accumulator == boundary.feedback_accumulator
    if asynchronous:
        assert full_evaluator is not None
        assert partial_evaluator is not None
        assert resumed_evaluator is not None
        resumed_order = resumed_evaluator.completion_order
        if format_name == "portable":
            resumed_order = partial_evaluator.completion_order + resumed_order
        assert resumed_order == full_evaluator.completion_order


@pytest.mark.parametrize("event_type", [GenerationEndEvent, RunEndEvent])
def test_generation_based_callback_checkpoint_resumes_to_uninterrupted_run(
    tmp_path, event_type
):
    problem = _make_problem()
    uninterrupted = _make_boundary_optimizer(
        problem, "generation_based", asynchronous=False
    )[0].run()
    optimizer, _ = _make_boundary_optimizer(
        problem, "generation_based", asynchronous=False
    )
    path = tmp_path / f"{event_type.__name__}.npz"
    saved = []
    callbacks = CallbackManager()

    def save_boundary(event):
        if event_type is RunEndEvent or not saved:
            event.ctx.save(path)
            saved.append(event.ctx)

    callbacks.register(event_type, save_boundary)
    optimizer.cbmanager = callbacks
    optimizer.run()

    assert saved
    loaded = OptimizationState.load(path, problem)
    resumed = _make_boundary_optimizer(problem, "generation_based", asynchronous=False)[
        0
    ].run_from(loaded)

    _assert_run_results_equal(resumed, uninterrupted)


@pytest.mark.parametrize("checkpoint_format", ["portable", "pickle", "scheduler"])
def test_async_generation_based_run_end_callback_rejects_nonresumable_context(
    tmp_path, checkpoint_format
):
    problem = _make_problem()
    optimizer = _make_async_generation_based_optimizer(
        problem, seed=42, n_gen=2, gen_ctrl=0
    )
    path = tmp_path / f"run-end-{checkpoint_format}"
    if checkpoint_format != "pickle":
        path = path.with_suffix(".npz")
    else:
        path = path.with_suffix(".pkl")
    saver = _RunEndCheckpointSaver(checkpoint_format, path)
    callbacks = CallbackManager()
    callbacks.register(RunEndEvent, saver)
    optimizer.cbmanager = callbacks

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        returned = optimizer.run()

    assert saver.observed is not None
    callback_fe, _callback_gen, callback_pending = saver.observed
    assert callback_fe != returned.fe
    assert callback_pending != len(returned.pending_evaluations)
    assert saver.error is not None
    assert "requests waiting to be submitted" in saver.error
    assert "save at a GenerationEndEvent instead" in saver.error
    assert "save the state returned by run()" in saver.error
    assert not path.exists()


@pytest.mark.parametrize("checkpoint_format", ["portable", "pickle", "scheduler"])
def test_async_generation_based_run_end_callback_with_capacity_resumes(
    tmp_path, checkpoint_format
):
    problem = _make_problem()
    uninterrupted = _make_async_generation_based_optimizer(
        problem, seed=42, n_gen=4, gen_ctrl=0, max_pending=2
    ).run()
    optimizer = _make_async_generation_based_optimizer(
        problem, seed=42, n_gen=1, gen_ctrl=0, max_pending=2
    )
    path = tmp_path / f"run-end-capacity-{checkpoint_format}"
    path = path.with_suffix(".pkl" if checkpoint_format == "pickle" else ".npz")
    saver = _RunEndCheckpointSaver(checkpoint_format, path)
    optimizer.cbmanager.register(RunEndEvent, saver)
    optimizer.run()
    assert saver.error is None
    assert path.exists()
    if checkpoint_format == "pickle":
        resumed_optimizer, restored = Optimizer.load_pickle(path)
        resumed_optimizer.cbmanager = CallbackManager()
        resumed_optimizer.set_termination(Termination(max_gen(4)))
    else:
        restored = OptimizationState.load(path, problem)
        resumed_optimizer = _make_async_generation_based_optimizer(
            problem, seed=42, n_gen=4, gen_ctrl=0, max_pending=2
        )
    resumed = resumed_optimizer.run_from(restored)
    _assert_run_results_equal(resumed, uninterrupted)


@pytest.mark.parametrize("history_channel", ["population", "front"])
@pytest.mark.parametrize("checkpoint_format", ["portable", "pickle", "scheduler"])
def test_async_generation_based_run_end_callback_records_enabled_history_channel(
    tmp_path, history_channel, checkpoint_format
):
    problem = _make_problem()

    def build(n_gen: int) -> Optimizer:
        return _make_async_generation_based_optimizer(
            problem, seed=42, n_gen=n_gen, gen_ctrl=0, max_pending=2
        ).set_history([history_channel])

    uninterrupted = build(4).run()
    optimizer = build(1)
    path = tmp_path / f"run-end-{history_channel}-{checkpoint_format}"
    path = path.with_suffix(".pkl" if checkpoint_format == "pickle" else ".npz")
    saver = _RunEndCheckpointSaver(checkpoint_format, path)
    optimizer.cbmanager.register(RunEndEvent, saver)
    returned = optimizer.run()

    assert saver.error is None
    _assert_run_results_equal(returned, build(1).run())

    if checkpoint_format == "pickle":
        resumed_optimizer, restored = Optimizer.load_pickle(path)
        resumed_optimizer.cbmanager = CallbackManager()
        resumed_optimizer.set_termination(Termination(max_gen(4)))
    else:
        restored = OptimizationState.load(path, problem)
        resumed_optimizer = build(4)
    resumed = resumed_optimizer.run_from(restored)

    _assert_run_results_equal(resumed, uninterrupted)


@pytest.mark.parametrize("checkpoint_format", ["portable", "pickle", "scheduler"])
def test_async_generation_based_returned_state_resumes_in_all_formats(
    tmp_path, checkpoint_format
):
    problem = _make_problem()
    uninterrupted = _make_async_generation_based_optimizer(
        problem, seed=42, n_gen=4, gen_ctrl=0
    ).run()
    optimizer = _make_async_generation_based_optimizer(
        problem, seed=42, n_gen=2, gen_ctrl=0
    )
    returned = optimizer.run()
    path = tmp_path / f"returned-{checkpoint_format}"
    if checkpoint_format == "pickle":
        path = path.with_suffix(".pkl")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            optimizer.save_pickle(returned, path)
            resumed_optimizer, restored = Optimizer.load_pickle(path)
        resumed_optimizer.set_termination(Termination(max_gen(4)))
    elif checkpoint_format == "portable":
        path = path.with_suffix(".npz")
        returned.save(path)
        restored = OptimizationState.load(path, problem)
        resumed_optimizer = _make_async_generation_based_optimizer(
            problem, seed=42, n_gen=4, gen_ctrl=0
        )
    else:
        path = path.with_suffix(".npz")
        assert optimizer.async_evaluation_scheduler is not None
        optimizer.async_evaluation_scheduler.checkpoint(returned, str(path))
        restored = OptimizationState.load(path, problem)
        resumed_optimizer = _make_async_generation_based_optimizer(
            problem, seed=42, n_gen=4, gen_ctrl=0
        )

    resumed = resumed_optimizer.run_from(restored)
    in_flight_request_ids = (
        set(returned.evaluation_plan_state.deferred)
        if returned.evaluation_plan_state is not None
        else set()
    )
    _assert_run_results_equal(
        resumed,
        uninterrupted,
        in_flight_request_ids=in_flight_request_ids,
        portable_checkpoint=checkpoint_format != "pickle",
    )


def test_gb_async_repeat_wait_point_is_rejected_on_every_checkpoint_path(
    tmp_path,
):
    problem = _make_problem()
    optimizer, _evaluator = _make_boundary_optimizer(
        problem, "generation_based", asynchronous=False
    )
    evaluator = _ReplayEvaluator()
    optimizer.set_evaluator(evaluator).set_async_evaluation_scheduler(
        AsyncEvaluationScheduler(evaluator, max_pending=2)
    )
    optimizer.resolve_defaults()
    optimizer._compile_plan()
    assert optimizer.initializer is not None
    state = optimizer.initializer.initialize(optimizer, problem)
    runtime = create_runtime(optimizer)
    session = runtime.initialize(resolve_plan(optimizer), state)
    step = runtime.advance(session)
    assert step.session is not None
    session = step.session
    assert len(session.frames) == 2
    assert len(session.state.pending_evaluations) == 2

    portable_path = tmp_path / "repeat-wait.npz"
    pickle_path = tmp_path / "repeat-wait.pkl"
    scheduler_path = tmp_path / "repeat-wait-scheduler.npz"
    with pytest.raises(CheckpointError, match="open region"):
        session.state.save(portable_path)
    with (
        warnings.catch_warnings(),
        pytest.raises(CheckpointError, match="open region"),
    ):
        warnings.simplefilter("ignore", UserWarning)
        optimizer.save_pickle(session.state, pickle_path)
    assert optimizer.async_evaluation_scheduler is not None
    with pytest.raises(CheckpointError, match="open region"):
        optimizer.async_evaluation_scheduler.checkpoint(
            session.state, str(scheduler_path)
        )
    assert not portable_path.exists()
    assert not pickle_path.exists()
    assert not scheduler_path.exists()


def test_gb_subclass_repeat_wait_point_is_rejected(tmp_path):
    class MyGB(GenerationBasedStrategy):
        pass

    problem = _make_problem()
    optimizer, _evaluator = _make_boundary_optimizer(
        problem, "generation_based", asynchronous=False
    )
    optimizer.set_strategy(MyGB(2))
    evaluator = _ReplayEvaluator()
    optimizer.set_evaluator(evaluator).set_async_evaluation_scheduler(
        AsyncEvaluationScheduler(evaluator, max_pending=2)
    )
    optimizer.resolve_defaults()
    optimizer._compile_plan()
    assert optimizer.initializer is not None
    state = optimizer.initializer.initialize(optimizer, problem)
    runtime = create_runtime(optimizer)
    session = runtime.initialize(resolve_plan(optimizer), state)
    step = runtime.advance(session)
    assert step.session is not None
    with pytest.raises(CheckpointError, match="open region"):
        step.session.state.save(tmp_path / "subclass.npz")
    assert not (tmp_path / "subclass.npz").exists()


def test_acquisition_result_checkpoint_preserves_artifact_arrays(tmp_path):
    problem = _make_problem()
    state = _make_optimizer(problem, seed=7, n_gen=2).run()
    result = AcquisitionResult(
        scores=np.array([0.25, 0.75], dtype=np.float64),
        artifacts={
            "order": np.array([1, 0], dtype=np.int64),
            "joint": {"values": np.array([[1.0, 2.0]], dtype=np.float32)},
        },
    )
    state.set_state(ACQUISITION_RESULT, result)
    path = tmp_path / "acquisition.npz"
    state.save(path)

    loaded = OptimizationState.load(path, problem)
    restored = loaded.get_state(ACQUISITION_RESULT)

    assert isinstance(restored, AcquisitionResult)
    np.testing.assert_array_equal(restored.scores, result.scores)
    assert restored.scores is not None
    assert result.scores is not None
    assert restored.scores.dtype == result.scores.dtype
    np.testing.assert_array_equal(
        restored.artifacts["order"], result.artifacts["order"]
    )
    assert restored.artifacts["order"].dtype == result.artifacts["order"].dtype
    np.testing.assert_array_equal(
        restored.artifacts["joint"]["values"],
        result.artifacts["joint"]["values"],
    )
    assert (
        restored.artifacts["joint"]["values"].dtype
        == result.artifacts["joint"]["values"].dtype
    )


def test_checkpoint_loads_acquisition_result_key_schema_one(tmp_path):
    problem = _make_problem()
    state = _make_boundary_optimizer(problem, "generation_based", asynchronous=False)[
        0
    ].run()
    path = tmp_path / "current.npz"
    state.save(path)
    raw = dict(np.load(path, allow_pickle=False).items())
    entries = json.loads(bytes(raw["_state_entries"]).decode())
    entry = next(
        item
        for item in entries
        if item["key"]["namespace"] == "evaluations"
        and item["key"]["name"] == "acquisition_result"
    )
    entry["key"]["schema_version"] = 1
    entry["target_schema_version"] = 1
    entry["value"] = {"codec": "json", "value": None}
    raw["_state_entries"] = np.frombuffer(
        json.dumps(entries, allow_nan=True).encode(), dtype=np.uint8
    )
    schema_one_path = tmp_path / "schema-one.npz"
    np.savez(schema_one_path, **raw)

    loaded = OptimizationState.load(schema_one_path, problem)

    assert loaded.get_state(ACQUISITION_RESULT) is None


def test_pickle_v1_acquisition_result_migrates_before_portable_resave(tmp_path):
    problem = _make_problem()
    optimizer = _make_boundary_optimizer(
        problem, "generation_based", asynchronous=False
    )[0]
    state = optimizer.run()
    schema_one_key = StateKey(
        namespace="evaluations", name="acquisition_result", schema_version=1
    )
    result = AcquisitionResult(
        scores=np.array([0.25, 0.75], dtype=np.float64),
        artifacts={"schema_one": np.array([3, 1], dtype=np.int64)},
    )
    state.set_state(schema_one_key, result)

    pickle_path = tmp_path / "schema-one.pkl"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        optimizer.save_pickle(state, pickle_path)
        _restored_optimizer, restored = Optimizer.load_pickle(pickle_path)

    portable_path = tmp_path / "migrated.npz"
    restored.save(portable_path)
    round_tripped = OptimizationState.load(portable_path, problem)
    migrated = round_tripped.get_state(ACQUISITION_RESULT)

    assert isinstance(migrated, AcquisitionResult)
    np.testing.assert_array_equal(migrated.scores, result.scores)
    np.testing.assert_array_equal(migrated.artifacts["schema_one"], [3, 1])
    with pytest.raises(KeyError):
        round_tripped.get_state(schema_one_key)


def _assert_checkpoint_paths_rejected(event, optimizer, tmp_path, *, asynchronous):
    state = event.ctx
    paths = {
        "portable": tmp_path / "portable.npz",
        "pickle": tmp_path / "pickle.pkl",
    }
    operations = [
        (paths["portable"], lambda: state.save(paths["portable"])),
        (paths["pickle"], lambda: optimizer.save_pickle(state, paths["pickle"])),
    ]
    if asynchronous:
        scheduler_path = tmp_path / "scheduler.npz"
        scheduler = optimizer.async_evaluation_scheduler
        operations.append(
            (scheduler_path, lambda: scheduler.checkpoint(state, str(scheduler_path)))
        )
    for path, save in operations:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            try:
                save()
            except CheckpointError:
                assert not path.exists()
            else:
                pytest.fail(f"{path.name} was accepted at an unsupported save point")


@pytest.mark.parametrize("event_type", [PostAskEvent, PostEvaluationEvent])
@pytest.mark.parametrize("asynchronous", [False, True])
def test_non_boundary_callback_rejects_every_checkpoint_path(
    tmp_path, event_type, asynchronous
):
    problem = _make_problem()
    optimizer, _ = _make_boundary_optimizer(
        problem, "direct", asynchronous=asynchronous
    )
    callbacks = CallbackManager()
    attempted = []

    def save_in_callback(event):
        if attempted:
            return
        attempted.append(type(event))
        _assert_checkpoint_paths_rejected(
            event, optimizer, tmp_path, asynchronous=asynchronous
        )

    callbacks.register(event_type, save_in_callback)
    optimizer.cbmanager = callbacks

    optimizer.run()

    assert attempted == [event_type]


@pytest.mark.parametrize("replacement", ["method", "assignment"])
@pytest.mark.parametrize("asynchronous", [False, True])
def test_boundary_callback_rejects_pending_component_replacement(
    tmp_path, replacement, asynchronous
):
    problem = _make_problem()
    optimizer, _ = _make_boundary_optimizer(
        problem, "direct", asynchronous=asynchronous
    )
    callbacks = CallbackManager()
    attempted = []

    def replace_component(event):
        if attempted:
            return
        attempted.append(True)
        algorithm = GA(
            crossover=CrossoverBLXAlpha(prob=0.9, alpha=0.5),
            mutation=MutationUniform(prob_var=0.1),
            parent_selection=SequentialSelection(),
            survivor_selection=TruncationSelection(),
        )
        if replacement == "method":
            optimizer.set_algorithm(algorithm)
        else:
            optimizer.algorithm = algorithm
        _assert_checkpoint_paths_rejected(
            event, optimizer, tmp_path, asynchronous=asynchronous
        )

    callbacks.register(GenerationEndEvent, replace_component)
    optimizer.cbmanager = callbacks

    optimizer.run()

    assert attempted


def test_thread_evaluator_pickle_failure_names_evaluator_and_writes_no_file(
    tmp_path,
):
    problem = _make_problem()
    evaluator = AsyncEvaluator(SerialEvaluator())
    optimizer = (
        Optimizer(problem, seed=3)
        .set_initializer(LHSInitializer(N_INIT_ARCHIVE, N_INIT_POP))
        .set_algorithm(
            GA(
                crossover=CrossoverBLXAlpha(prob=0.9, alpha=0.5),
                mutation=MutationUniform(prob_var=0.1),
                parent_selection=SequentialSelection(),
                survivor_selection=TruncationSelection(),
            )
        )
        .set_strategy(DirectStrategy())
        .set_evaluator(evaluator)
        .set_async_evaluation_scheduler(
            AsyncEvaluationScheduler(evaluator, max_pending=2)
        )
    )
    assert optimizer.initializer is not None
    state = optimizer.initializer.initialize(optimizer, problem)
    path = tmp_path / "thread.pkl"

    with (
        warnings.catch_warnings(),
        pytest.raises(CheckpointError, match="AsyncEvaluator"),
    ):
        warnings.simplefilter("ignore", UserWarning)
        optimizer.save_pickle(state, path)

    assert not path.exists()


def test_pending_evaluation_with_started_callback_rejects_all_checkpoint_paths(
    tmp_path,
):
    problem = _make_problem()
    optimizer = _make_async_generation_based_optimizer(
        problem, seed=12, n_gen=2, gen_ctrl=0
    )
    state = optimizer.run()
    request_id, pending = next(iter(state.pending_evaluations.items()))
    state.pending_evaluations = {
        request_id: replace(
            pending,
            processing={**pending.processing, 0: "callback-started"},
            fatal_error=EvaluationErrorInfo("CallbackFatal", "callback started"),
        )
    }
    portable_path = tmp_path / "started.npz"
    pickle_path = tmp_path / "started.pkl"
    scheduler_path = tmp_path / "started-scheduler.npz"

    with pytest.raises(CheckpointError, match="callback or feedback update"):
        state.save(portable_path)
    with (
        warnings.catch_warnings(),
        pytest.raises(CheckpointError, match="callback or feedback update"),
    ):
        warnings.simplefilter("ignore", UserWarning)
        optimizer.save_pickle(state, pickle_path)
    assert optimizer.async_evaluation_scheduler is not None
    with pytest.raises(CheckpointError, match="callback or feedback update"):
        optimizer.async_evaluation_scheduler.checkpoint(state, str(scheduler_path))

    assert not portable_path.exists()
    assert not pickle_path.exists()
    assert not scheduler_path.exists()


def test_checkpoint_inside_repeat_reports_open_region(tmp_path):
    problem = _make_problem()
    optimizer, _ = _make_boundary_optimizer(
        problem, "generation_based", asynchronous=False
    )
    callbacks = CallbackManager()
    inside = False
    errors = []
    path = tmp_path / "region.npz"

    def save_boundary(event):
        if not inside:
            return
        try:
            event.ctx.save(path)
        except CheckpointError as exc:
            errors.append(str(exc))

    def dispatch_boundary(event):
        nonlocal inside
        inside = True
        optimizer.dispatch(GenerationEndEvent(ctx=event.ctx))
        inside = False

    callbacks.register(GenerationEndEvent, save_boundary)
    callbacks.register(PostAskEvent, dispatch_boundary)
    optimizer.cbmanager = callbacks

    optimizer.run()

    assert errors and "open region" in errors[0]
    assert not path.exists()


def test_loaded_state_save_keeps_fatal_pending_behavior(tmp_path):
    problem = _make_problem()
    optimizer = _make_async_generation_based_optimizer(
        problem, seed=12, n_gen=2, gen_ctrl=0
    )
    state = optimizer.run()
    assert state.pending_evaluations
    path = tmp_path / "pending.npz"
    state.__dict__.pop("_checkpoint_guard", None)
    state.save(path)
    loaded = OptimizationState.load(path, problem)
    pending = next(iter(loaded.pending_evaluations.values()))
    loaded.pending_evaluations = {
        int(pending.request.request_id): PendingEvaluation(
            pending.request,
            pending.status,
            pending.applied_candidate_ids,
            pending.last_delivered_sequence,
            pending.last_acknowledged_sequence,
            pending.processing,
            pending.buffered_updates,
            pending.reserved_cost,
            pending.retry_count,
            pending.checkpointable,
            pending.original_candidate_ids,
            pending.feedback_result,
            EvaluationErrorInfo("CallbackFatal", "callback started"),
            pending.prediction,
        )
    }

    loaded.save(tmp_path / "fatal-pending.npz")


def test_checkpoint_rejects_saving_from_non_boundary_callback(tmp_path):
    optimizer = _make_optimizer(_make_problem(), seed=12, n_gen=1)
    state = optimizer.run()
    path = tmp_path / "callback.npz"
    callbacks = CallbackManager()
    errors = []

    def save_during_callback(event):
        try:
            event.ctx.save(path)
        except CheckpointError as exc:
            errors.append(str(exc))

    callbacks.register(PostAskEvent, save_during_callback)
    optimizer.cbmanager = callbacks

    optimizer.dispatch(PostAskEvent(ctx=state))

    assert errors == ["checkpoint is supported only at a step boundary or run end"]
    assert not path.exists()


def test_nested_event_rejects_save_until_outer_callback_ends(tmp_path):
    optimizer = _make_optimizer(_make_problem(), seed=14, n_gen=1)
    callbacks = CallbackManager()
    optimizer.cbmanager = callbacks
    state = optimizer.run()
    errors = []
    path = tmp_path / "nested-callback.npz"

    def dispatch_nested_event(event):
        optimizer.dispatch(GenerationEndEvent(ctx=event.ctx))
        try:
            event.ctx.save(path)
        except CheckpointError as exc:
            errors.append(str(exc))

    callbacks.register(PostAskEvent, dispatch_nested_event)

    optimizer.dispatch(PostAskEvent(ctx=state))

    assert errors == ["checkpoint is supported only at a step boundary or run end"]
    assert not path.exists()
    state.save(path)
    assert path.exists()


def test_pickle_callback_format_mutation_requires_optimizer(tmp_path):
    optimizer = _make_optimizer(_make_problem(), seed=15, n_gen=1)
    state = optimizer.run()
    callback_path = tmp_path / "checkpoints"
    callback = CheckpointCallback(callback_path)
    callback.format = "pickle"
    callback.register(optimizer.cbmanager)

    with pytest.raises(
        ValidationError,
        match="optimizer must be provided when format is 'pickle' or 'both'",
    ):
        optimizer.dispatch(GenerationEndEvent(ctx=state))

    assert not callback_path.exists()


def test_pickle_callback_requires_optimizer_when_created(tmp_path):
    callback_path = tmp_path / "checkpoints"
    with pytest.raises(ValidationError, match="optimizer must be provided"):
        CheckpointCallback(callback_path, format="pickle")

    assert not callback_path.exists()


def _pending_checkpoint_value(
    *,
    metadata=None,
    checkpointable=True,
    fatal_error=None,
    processing=None,
    buffered_updates=(),
    candidate_ids=(7,),
):
    return SimpleNamespace(
        request=SimpleNamespace(
            metadata={} if metadata is None else metadata,
            candidate_ids=np.asarray(candidate_ids, dtype=np.int64),
        ),
        fatal_error=fatal_error,
        processing={} if processing is None else processing,
        checkpointable=checkpointable,
        original_candidate_ids=None,
        buffered_updates=buffered_updates,
    )


def _partial_feedback_update(candidate_ids=(7,)):
    return SimpleNamespace(
        result=object(),
        status=EvaluationStatus.PARTIAL,
        sequence=0,
        candidate_ids=np.asarray(candidate_ids, dtype=np.int64),
    )


@pytest.mark.parametrize(
    ("case", "message"),
    [
        pytest.param(
            "fatal_pending",
            "callback or feedback update has started",
            id="fatal-pending-evaluation",
        ),
        pytest.param(
            "started_processing",
            "callback or feedback update has started",
            id="started-pending-processing",
        ),
        pytest.param(
            "not_reattachable",
            "pending evaluation work cannot be reattached",
            id="work-cannot-be-reattached",
        ),
        pytest.param(
            "invalid_proposal_identity",
            "invalid pending proposal identity",
            id="invalid-proposal-identity",
        ),
        pytest.param(
            "accumulator_wrong_type",
            "FeedbackAccumulator state is malformed",
            id="accumulator-wrong-type",
        ),
        pytest.param(
            "accumulator_wrong_buffers",
            "FeedbackAccumulator state is malformed",
            id="accumulator-wrong-buffers",
        ),
        pytest.param(
            "accumulator_bad_entry",
            "FeedbackAccumulator proposals are malformed",
            id="accumulator-bad-entry",
        ),
        pytest.param(
            "accumulator_bad_identity",
            "FeedbackAccumulator proposals are malformed",
            id="accumulator-bad-identity",
        ),
        pytest.param(
            "accumulator_duplicate_identity",
            "FeedbackAccumulator proposals are malformed",
            id="accumulator-duplicate-identity",
        ),
        pytest.param(
            "accumulator_without_pending_proposal",
            "buffer has no pending proposal",
            id="accumulator-without-pending-proposal",
        ),
        pytest.param(
            "accumulator_missing_candidate_ids",
            "proposal candidates are malformed",
            id="accumulator-missing-candidate-ids",
        ),
        pytest.param(
            "accumulator_mismatched_candidate_ids",
            "proposal candidates do not match pending requests",
            id="accumulator-mismatched-candidate-ids",
        ),
        pytest.param(
            "partial_feedback_without_proposal_identity",
            "missing proposal identity for partial feedback",
            id="partial-feedback-without-proposal-identity",
        ),
        pytest.param(
            "partial_feedback_without_accumulator",
            "missing FeedbackAccumulator state for partial feedback",
            id="partial-feedback-without-accumulator",
        ),
        pytest.param(
            "partial_feedback_without_delivery",
            "missing a pending feedback delivery",
            id="partial-feedback-without-delivery",
        ),
        pytest.param(
            "partial_feedback_with_extra_candidate",
            "inconsistent FeedbackAccumulator proposal candidates",
            id="partial-feedback-with-extra-candidate",
        ),
    ],
)
def test_checkpoint_rejects_invalid_pending_state_without_writing_file(
    tmp_path, case, message
):
    state = _make_optimizer(_make_problem(), seed=16, n_gen=1).run()
    pending = _pending_checkpoint_value()
    accumulator: Any = None

    if case == "fatal_pending":
        pending.fatal_error = EvaluationErrorInfo("CallbackFatal", "started")
    elif case == "started_processing":
        pending.processing = {0: "tell-started"}
    elif case == "not_reattachable":
        pending.checkpointable = False
    elif case == "invalid_proposal_identity":
        pending.request.metadata = {"proposal_id": "invalid"}
    elif case == "accumulator_wrong_type":
        accumulator = []
    elif case == "accumulator_wrong_buffers":
        accumulator = {"buffers": ()}
    elif case == "accumulator_bad_entry":
        accumulator = {"buffers": [None]}
    elif case == "accumulator_bad_identity":
        accumulator = {"buffers": [{"proposal_id": "invalid"}]}
    elif case == "accumulator_duplicate_identity":
        pending.request.metadata = {"proposal_id": 1}
        accumulator = {
            "buffers": [
                {"proposal_id": 1, "candidate_ids": [7]},
                {"proposal_id": 1, "candidate_ids": [7]},
            ]
        }
    elif case == "accumulator_without_pending_proposal":
        accumulator = {"buffers": [{"proposal_id": 2, "candidate_ids": [7]}]}
    elif case == "accumulator_missing_candidate_ids":
        pending.request.metadata = {"proposal_id": 1}
        accumulator = {"buffers": [{"proposal_id": 1}]}
    elif case == "accumulator_mismatched_candidate_ids":
        pending.request.metadata = {"proposal_id": 1}
        accumulator = {"buffers": [{"proposal_id": 1, "candidate_ids": [8]}]}
    elif case == "partial_feedback_without_proposal_identity":
        pending.request.metadata = {}
        pending.buffered_updates = (_partial_feedback_update(),)
        if state._store.contains(PROPOSALS_CURRENT):
            state._store = state._store.apply_patch(
                StatePatch(writes={}, deletes=frozenset({PROPOSALS_CURRENT}))
            )
    elif case == "partial_feedback_without_accumulator":
        pending.request.metadata = {"proposal_id": 1}
        pending.buffered_updates = (_partial_feedback_update(),)
    elif case == "partial_feedback_without_delivery":
        pending.request.metadata = {"proposal_id": 1}
        pending.buffered_updates = (_partial_feedback_update(),)
        accumulator = {
            "buffers": [{"proposal_id": 1, "candidate_ids": [7], "seen": {}}]
        }
    elif case == "partial_feedback_with_extra_candidate":
        pending.request.metadata = {"proposal_id": 1}
        pending.buffered_updates = (_partial_feedback_update(candidate_ids=(8,)),)
        accumulator = {
            "buffers": [
                {
                    "proposal_id": 1,
                    "candidate_ids": [7],
                    "seen": {"0": "delivered"},
                }
            ]
        }

    state.pending_evaluations = {1: pending}
    cast(Any, state).feedback_accumulator = accumulator
    path = tmp_path / f"{case}.npz"

    with pytest.raises(CheckpointError, match=message):
        state.save(path)

    assert not path.exists()


def test_checkpoint_rejects_runtime_guard_without_validation_method(tmp_path):
    state = _make_optimizer(_make_problem(), seed=17, n_gen=1).run()
    state.__dict__["_checkpoint_guard"] = object()
    path = tmp_path / "invalid-guard.npz"

    with pytest.raises(CheckpointError, match="checkpoint runtime guard is invalid"):
        state.save(path)

    assert not path.exists()


def test_pickle_checkpoint_reports_reattachment_check_failure(tmp_path):
    optimizer = _make_async_generation_based_optimizer(
        _make_problem(), seed=18, n_gen=2, gen_ctrl=0
    )
    state = optimizer.run()
    assert state.pending_evaluations
    scheduler = optimizer.async_evaluation_scheduler
    assert scheduler is not None
    evaluator = scheduler.evaluator

    def fail_reattachment_check(pending: PendingEvaluation) -> bool:
        del pending
        raise RuntimeError("reattachment check failed")

    object.__setattr__(evaluator, "can_reattach", fail_reattachment_check)
    path = tmp_path / "reattachment.pkl"

    with (
        warnings.catch_warnings(),
        pytest.raises(CheckpointError, match="could not verify pending request"),
    ):
        warnings.simplefilter("ignore", UserWarning)
        optimizer.save_pickle(state, path)

    assert not path.exists()


def test_checkpoint_rejects_evaluator_that_cannot_reattach_pending_work(tmp_path):
    optimizer = _make_async_generation_based_optimizer(
        _make_problem(), seed=29, n_gen=2, gen_ctrl=0
    )
    state = optimizer.run()
    assert state.pending_evaluations
    scheduler = optimizer.async_evaluation_scheduler
    assert scheduler is not None
    evaluator = scheduler.evaluator

    def cannot_reattach(pending: PendingEvaluation) -> bool:
        del pending
        return False

    object.__setattr__(evaluator, "can_reattach", cannot_reattach)
    path = tmp_path / "cannot-reattach.npz"

    with pytest.raises(CheckpointError, match="cannot reattach pending request"):
        state.save(path)

    assert not path.exists()


def test_checkpoint_round_trips_pending_updates_without_feedback_results(tmp_path):
    problem = _make_problem()
    state = _make_optimizer(problem, seed=30, n_gen=1).run()
    candidate_ids = state.population.get_array("id")[:1].copy()
    request = EvaluationRequest(
        np.int64(101), candidate_ids, state.population.x[:1].copy()
    )
    update = EvaluationUpdate(
        request.request_id,
        EvaluationStatus.PENDING,
        request.candidate_ids,
        result=None,
        sequence=0,
    )
    pending = PendingEvaluation(
        request,
        EvaluationStatus.PENDING,
        np.empty(0, dtype=np.int64),
        last_delivered_sequence=0,
        buffered_updates=(update,),
        checkpointable=True,
    )
    state.pending_evaluations = {int(request.request_id): pending}
    path = tmp_path / "pending-without-result.npz"

    state.save(path)

    restored = OptimizationState.load(path, problem)
    restored_pending = restored.pending_evaluations[int(request.request_id)]
    assert len(restored_pending.buffered_updates) == 1
    assert restored_pending.buffered_updates[0].result is None


def test_checkpoint_round_trips_none_acquisition_result(tmp_path):
    problem = _make_problem()
    state = _make_optimizer(problem, seed=19, n_gen=1).run()
    state.set_state(ACQUISITION_RESULT, None)
    path = tmp_path / "no-acquisition.npz"

    state.save(path)

    restored = OptimizationState.load(path, problem)
    assert restored.get_state(ACQUISITION_RESULT) is None


def test_checkpoint_round_trips_supported_acquisition_artifact_values(tmp_path):
    problem = _make_problem()
    state = _make_optimizer(problem, seed=20, n_gen=1).run()
    result = AcquisitionResult(
        scores=np.array([0.25, 0.75], dtype=np.float32),
        artifacts={
            "array": np.array([3, 1], dtype=np.int16),
            "scalar": np.float32(2.5),
            "nested": {
                "tuple": (None, True, 4, 1.25, "value"),
                "list": [np.int64(7), {"ready": False}],
            },
        },
    )
    state.set_state(ACQUISITION_RESULT, result)
    path = tmp_path / "acquisition-values.npz"

    state.save(path)

    restored = OptimizationState.load(path, problem).get_state(ACQUISITION_RESULT)
    assert isinstance(restored, AcquisitionResult)
    np.testing.assert_array_equal(restored.scores, result.scores)
    assert restored.scores is not None and result.scores is not None
    assert restored.scores.dtype == result.scores.dtype
    np.testing.assert_array_equal(restored.artifacts["array"], [3, 1])
    assert restored.artifacts["array"].dtype == np.dtype(np.int16)
    assert restored.artifacts["scalar"] == np.float32(2.5)
    assert isinstance(restored.artifacts["scalar"], np.float32)
    assert restored.artifacts["nested"] == {
        "tuple": (None, True, 4, 1.25, "value"),
        "list": [np.int64(7), {"ready": False}],
    }
    assert isinstance(restored.artifacts["nested"]["tuple"], tuple)
    assert isinstance(restored.artifacts["nested"]["list"], list)
    assert isinstance(restored.artifacts["nested"]["list"][0], np.int64)


@pytest.mark.parametrize(
    ("artifacts", "message"),
    [
        pytest.param(
            {"values": np.array([object()], dtype=object)},
            "object dtype is not checkpointable",
            id="object-array",
        ),
        pytest.param(
            {"values": {1: "invalid-key"}},
            "checkpoint metadata keys must be strings",
            id="non-string-mapping-key",
        ),
        pytest.param(
            {"values": object()},
            "non-serializable checkpoint value: object",
            id="unsupported-value",
        ),
    ],
)
def test_checkpoint_rejects_unsupported_acquisition_artifacts(
    tmp_path, artifacts, message
):
    state = _make_optimizer(_make_problem(), seed=21, n_gen=1).run()
    state.set_state(ACQUISITION_RESULT, AcquisitionResult(artifacts=artifacts))
    path = tmp_path / "unsupported-acquisition.npz"

    with pytest.raises(CheckpointError, match=message):
        state.save(path)

    assert not path.exists()


@pytest.mark.parametrize(
    ("stored_value", "message"),
    [
        pytest.param(
            {"scores": {"codec": "json", "value": None}},
            "acquisition result is malformed",
            id="missing-artifacts",
        ),
        pytest.param(
            {"scores": None, "artifacts": {"codec": "json", "value": None}},
            "portable acquisition value is malformed",
            id="malformed-portable-value",
        ),
        pytest.param(
            {
                "scores": {"codec": "mapping", "items": []},
                "artifacts": {"codec": "json", "value": None},
            },
            "portable acquisition mapping is malformed",
            id="malformed-mapping",
        ),
        pytest.param(
            {
                "scores": {"codec": "tuple", "items": {}},
                "artifacts": {"codec": "json", "value": None},
            },
            "portable acquisition sequence is malformed",
            id="malformed-sequence",
        ),
        pytest.param(
            {
                "scores": {"codec": "future"},
                "artifacts": {"codec": "json", "value": None},
            },
            "unknown portable acquisition codec",
            id="unsupported-codec",
        ),
    ],
)
def test_checkpoint_load_rejects_malformed_acquisition_artifacts(
    tmp_path, stored_value, message
):
    problem = _make_problem()
    state = _make_optimizer(problem, seed=22, n_gen=1).run()
    state.set_state(
        ACQUISITION_RESULT,
        AcquisitionResult(scores=np.array([1.0]), artifacts={"value": 1}),
    )
    source = tmp_path / "valid-acquisition.npz"
    state.save(source)
    invalid = tmp_path / "malformed-acquisition.npz"
    write_checkpoint_value(source, invalid, "acquisition_result", stored_value)

    with pytest.raises(CheckpointError, match=message):
        OptimizationState.load(invalid, problem)


def test_checkpoint_load_rejects_acquisition_array_with_missing_storage(tmp_path):
    problem = _make_problem()
    state = _make_optimizer(problem, seed=23, n_gen=1).run()
    state.set_state(
        ACQUISITION_RESULT,
        AcquisitionResult(scores=np.array([1.0]), artifacts={"value": 1}),
    )
    source = tmp_path / "valid-array.npz"
    state.save(source)
    invalid = tmp_path / "missing-array.npz"
    write_checkpoint_value(
        source,
        invalid,
        "acquisition_result",
        {
            "scores": {"codec": "array", "key": "missing-array"},
            "artifacts": {"codec": "json", "value": None},
        },
    )

    with pytest.raises(CheckpointError, match="Invalid checkpoint"):
        OptimizationState.load(invalid, problem)


def test_generation_snapshot_returns_same_state_without_history():
    state = _make_optimizer(_make_problem(), seed=24, n_gen=1).run()
    state = state.replace(history=None)

    assert snapshot_generation_for_checkpoint(state) is state


def test_generation_snapshot_records_enabled_channels_when_behind():
    state = _make_optimizer(_make_problem(), seed=25, n_gen=1).run()
    history = History(("summary", "population", "front"))
    state = state.replace(history=history, gen=state.gen + 1)

    snapshot = snapshot_generation_for_checkpoint(state)

    assert snapshot is not state
    assert snapshot.history is not history
    assert snapshot.history is not None
    for channel in ("summary", "population", "front"):
        np.testing.assert_array_equal(
            snapshot.history.channel(channel)["gen"], [state.gen]
        )


def test_generation_snapshot_keeps_state_when_channel_already_has_generation():
    state = _make_optimizer(_make_problem(), seed=26, n_gen=1).run()
    history = History(("summary", "population"))
    history.append("summary", gen=state.gen)
    state = state.replace(history=history)

    assert snapshot_generation_for_checkpoint(state) is state


class SecondSerializationFailure:
    def __init__(self):
        self.serialization_count = 0

    def __reduce__(self):
        self.serialization_count += 1
        if self.serialization_count > 1:
            raise RuntimeError("second serialization failed")
        return (SecondSerializationFailure, ())


def test_pickle_checkpoint_preflight_rejects_unserializable_state_without_file(
    tmp_path,
):
    optimizer = _make_optimizer(_make_problem(), seed=27, n_gen=1)
    state = optimizer.run()
    state.data["unserializable"] = lambda: None
    path = tmp_path / "unserializable.pkl"

    with (
        warnings.catch_warnings(),
        pytest.raises(CheckpointError, match="checkpoint cannot be pickled"),
    ):
        warnings.simplefilter("ignore", UserWarning)
        optimizer.save_pickle(state, path)

    assert not path.exists()


def test_pickle_checkpoint_wraps_serialization_failure_after_preflight(tmp_path):
    optimizer = _make_optimizer(_make_problem(), seed=28, n_gen=1)
    state = optimizer.run()
    state.data["two_pass"] = SecondSerializationFailure()
    value = state.data["two_pass"]
    path = tmp_path / "second-pass.pkl"

    with (
        warnings.catch_warnings(),
        pytest.raises(CheckpointError, match="second serialization failed"),
    ):
        warnings.simplefilter("ignore", UserWarning)
        optimizer.save_pickle(state, path)

    assert value.serialization_count == 2
    assert not path.exists()


class AlwaysTrueBranchCondition:
    def contract(self) -> StateContract:
        return StateContract()

    def evaluate(self, view, /) -> bool:
        del view
        return True


class NestedGenerationStrategy(GenerationBasedStrategy):
    def build_template(self):
        return Pipeline(
            [
                Branch(
                    AlwaysTrueBranchCondition(),
                    then=super().build_template(),
                    name="generation_branch",
                )
            ],
            name="nested_generation",
        )


def test_nested_repeat_checkpoint_is_rejected_while_work_is_pending(tmp_path):
    problem = _make_problem()
    optimizer, _ = _make_boundary_optimizer(
        problem, "generation_based", asynchronous=False
    )
    optimizer.set_strategy(NestedGenerationStrategy(2))
    evaluator = _ReplayEvaluator()
    optimizer.set_evaluator(evaluator).set_async_evaluation_scheduler(
        AsyncEvaluationScheduler(evaluator, max_pending=2)
    )
    optimizer.resolve_defaults()
    optimizer._compile_plan()
    assert optimizer.initializer is not None
    state = optimizer.initializer.initialize(optimizer, problem)
    runtime = create_runtime(optimizer)
    session = runtime.initialize(resolve_plan(optimizer), state)
    step = runtime.advance(session)

    assert step.session is not None
    assert step.session.state.pending_evaluations
    path = tmp_path / "nested-repeat.npz"
    with pytest.raises(CheckpointError, match="open region"):
        step.session.state.save(path)
    assert not path.exists()


def write_checkpoint_value(source, destination, name, value):
    payload = dict(np.load(source, allow_pickle=False).items())
    entries = json.loads(bytes(payload["_state_entries"]).decode())
    for entry in entries:
        key = entry["key"]
        if key["namespace"] == "evaluations" and key["name"] == name:
            entry["value"]["value"] = value
            break
    else:
        raise AssertionError(f"evaluations/{name} entry was not found")
    payload["_state_entries"] = np.frombuffer(
        json.dumps(entries).encode(), dtype=np.uint8
    )
    np.savez(destination, **payload)
