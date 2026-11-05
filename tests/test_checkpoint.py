"""Tests for seed, checkpoint, and resume features."""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pytest

from saealib import (
    GA,
    PSO,
    CheckpointCallback,
    CrossoverBLXAlpha,
    GaussianKernel,
    IndividualBasedStrategy,
    LHSInitializer,
    MutationUniform,
    Optimizer,
    RBFSurrogate,
    SequentialSelection,
    Termination,
    TruncationSelection,
    max_gen,
)
from saealib.comparators import SingleObjectiveComparator
from saealib.context import OptimizationState
from saealib.exceptions import ValidationError
from saealib.problem import Problem

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


def test_iterate_from_yields_correct_gen(tmp_path):
    problem = _make_problem()
    ctx_mid = _make_optimizer(problem, seed=0, n_gen=2).run()
    p = tmp_path / "ckpt.npz"
    ctx_mid.save(p)

    loaded = OptimizationState.load(p, problem)
    gens = [ctx.gen for ctx in _make_optimizer(problem, n_gen=4).iterate_from(loaded)]
    assert gens[-1] == 4


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
# objective_source column across checkpoint schema versions
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


def _write_as_version_1(src: Path, dst: Path) -> None:
    """Rewrite a checkpoint in the version 1 layout, which has no source column."""
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


def _version_1_checkpoint(tmp_path: Path, channels: tuple[str, ...]):
    problem = _make_problem()
    state = _make_pso_optimizer(problem, 3, channels).run()
    written = tmp_path / "current.npz"
    state.save(written)
    version_1 = tmp_path / "version_1.npz"
    _write_as_version_1(written, version_1)
    assert int(np.load(version_1)["_checkpoint_schema_version"]) == 1
    return problem, state, version_1


def test_version_1_checkpoint_marks_rows_matching_the_evaluation_channel_true(
    tmp_path,
):
    problem, state, version_1 = _version_1_checkpoint(
        tmp_path, ("summary", "evaluation")
    )
    sources = _sources(state.population)
    assert set(sources.tolist()) == {0, 1}

    loaded = OptimizationState.load(version_1, problem)

    np.testing.assert_array_equal(
        _sources(loaded.population), np.where(sources == 0, 0, UNKNOWN_SOURCE)
    )
    assert np.all(_sources(loaded.archive) == 0)
    assert np.all(_sources(loaded.pareto_archive) == 0)


def test_version_1_checkpoint_without_evaluation_channel_marks_rows_unknown(
    tmp_path,
):
    problem, _state, version_1 = _version_1_checkpoint(tmp_path, ("summary",))

    loaded = OptimizationState.load(version_1, problem)

    assert np.all(_sources(loaded.population) == UNKNOWN_SOURCE)
    assert np.all(_sources(loaded.archive) == UNKNOWN_SOURCE)


@pytest.mark.parametrize(
    "channels", [("summary", "evaluation"), ("summary",)], ids=["evaluation", "none"]
)
def test_version_1_checkpoint_resumes_like_an_uninterrupted_run(tmp_path, channels):
    problem, _state, version_1 = _version_1_checkpoint(tmp_path, channels)
    uninterrupted = _make_pso_optimizer(problem, 6, channels).run()

    resumed = _make_pso_optimizer(problem, 6, channels).run_from(
        OptimizationState.load(version_1, problem)
    )

    assert (resumed.gen, resumed.fe) == (uninterrupted.gen, uninterrupted.fe)
    np.testing.assert_array_equal(resumed.population.id, uninterrupted.population.id)
    np.testing.assert_array_equal(resumed.population.f, uninterrupted.population.f)
    np.testing.assert_array_equal(resumed.archive.id, uninterrupted.archive.id)
    np.testing.assert_array_equal(
        _sources(resumed.population), _sources(uninterrupted.population)
    )


def test_version_2_npz_checkpoint_round_trips_objective_source_and_resumes(
    tmp_path,
):
    problem = _make_problem()
    state = _make_pso_optimizer(problem, 3, ("summary",)).run()
    path = tmp_path / "ckpt.npz"
    state.save(path)

    loaded = OptimizationState.load(path, problem)

    assert int(np.load(path)["_checkpoint_schema_version"]) == 2
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
