"""Tests for checkpointability declarations and resume persistence."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from saealib import (
    GA,
    LHSInitializer,
    Optimizer,
    SerialEvaluator,
    Termination,
    max_fe,
    max_gen,
)
from saealib.algorithms import Algorithm, DeapGenerateUpdateAlgorithm
from saealib.context import OptimizationState
from saealib.core.contracts import AssumptionSet
from saealib.exceptions import ConfigurationError, ReplacementRejectedError
from saealib.experiment import (
    AlgorithmEntry,
    ExperimentConfig,
    ExperimentRunner,
    resume_trial,
)
from saealib.operators import (
    CrossoverBLXAlpha,
    MutationUniform,
    SequentialSelection,
    TruncationSelection,
)
from saealib.problem import Problem
from saealib.strategies import DirectStrategy


class _NonCheckpointableEvaluator(SerialEvaluator):
    """Evaluator used to verify pre-run checkpoint rejection."""

    def __init__(self) -> None:
        self.calls = 0

    def contract(self):
        return replace(
            super().contract(),
            assumptions=AssumptionSet({"state.checkpointable": False}),
        )

    def evaluate_batch(self, x, problem):
        self.calls += 1
        return super().evaluate_batch(x, problem)


def _problem() -> Problem:
    return Problem(
        func=lambda x: np.array([np.sum(np.asarray(x) ** 2)]),
        dim=2,
        n_obj=1,
        direction=np.array([-1.0]),
        lb=[-2.0, -2.0],
        ub=[2.0, 2.0],
    )


def _optimizer(
    problem: Problem,
    *,
    seed: int = 0,
    n_gen: int = 2,
    evaluator: SerialEvaluator | None = None,
    algorithm: Algorithm | None = None,
) -> Optimizer:
    if algorithm is None:
        algorithm = GA(
            crossover=CrossoverBLXAlpha(prob=0.9, alpha=0.5),
            mutation=MutationUniform(prob_var=0.1),
            parent_selection=SequentialSelection(),
            survivor_selection=TruncationSelection(),
        )
    return (
        Optimizer(problem, seed=seed)
        .set_initializer(LHSInitializer(6, 6))
        .set_algorithm(algorithm)
        .set_strategy(DirectStrategy())
        .set_evaluator(evaluator or SerialEvaluator())
        .set_termination(Termination(max_gen(n_gen)))
    )


def test_checkpoint_rejects_non_checkpointable_component_before_execution(
    tmp_path: Path,
) -> None:
    evaluator = _NonCheckpointableEvaluator()
    optimizer = _optimizer(_problem(), evaluator=evaluator)

    with pytest.raises(
        ConfigurationError,
        match=r"evaluation_submit___evaluator.*_NonCheckpointableEvaluator",
    ):
        optimizer.run(checkpoint_path=tmp_path / "checkpoint")

    assert evaluator.calls == 0


def test_checkpointing_remains_available_for_unaware_components(tmp_path: Path) -> None:
    optimizer = _optimizer(_problem(), n_gen=2)

    result = optimizer.run(
        checkpoint_path=tmp_path / "checkpoint", checkpoint_interval=1
    )

    files = sorted((tmp_path / "checkpoint").glob("checkpoint_*.npz"))
    assert result.gen == 2
    assert len(files) == result.gen


def test_npz_checkpoint_roundtrip_preserves_nonfinite_state_values(
    tmp_path: Path,
) -> None:
    state = _optimizer(_problem(), n_gen=1).run()
    state = state.replace(data={"nonfinite": [np.nan, np.inf, -np.inf]})
    checkpoint = tmp_path / "nonfinite.npz"

    state.save(checkpoint)
    restored = OptimizationState.load(checkpoint, state.problem)

    values = restored.data["nonfinite"]
    assert np.isnan(values[0])
    assert values[1] == np.inf
    assert values[2] == -np.inf


def test_npz_checkpoint_reproduction_dim3_seed0(
    tmp_path: Path,
) -> None:
    problem = Problem(
        func=lambda x: float(np.sum(np.asarray(x) ** 2)),
        dim=3,
        n_obj=1,
        direction=np.array([-1.0]),
        lb=[-5.0] * 3,
        ub=[5.0] * 3,
    )
    optimizer = Optimizer(problem, seed=0).set_termination(Termination(max_fe(60)))

    result = optimizer.run(
        checkpoint_path=tmp_path / "checkpoint", checkpoint_interval=1
    )

    assert result.gen > 0


def test_deap_algorithm_rejects_portable_checkpointing(tmp_path: Path) -> None:
    pytest.importorskip("deap")
    from deap import cma

    algorithm = DeapGenerateUpdateAlgorithm(cma.Strategy([0.0, 0.0], 1.0, lambda_=6))
    optimizer = _optimizer(_problem(), algorithm=algorithm)

    with pytest.raises(ConfigurationError, match="DeapGenerateUpdateAlgorithm"):
        optimizer.run(checkpoint_path=tmp_path / "checkpoint")


def test_pymoo_algorithm_rejects_portable_checkpointing(tmp_path: Path) -> None:
    pytest.importorskip("pymoo")
    from pymoo.algorithms.soo.nonconvex.ga import GA as PymooGA  # noqa: N811

    from saealib.algorithms.pymoo_algorithm import PymooAlgorithm

    optimizer = _optimizer(_problem(), algorithm=PymooAlgorithm(PymooGA(pop_size=6)))

    with pytest.raises(ConfigurationError, match="PymooAlgorithm"):
        optimizer.run(checkpoint_path=tmp_path / "checkpoint")


@pytest.mark.parametrize("method", ["iterate", "run", "iterate_from", "run_from"])
@pytest.mark.parametrize(
    "checkpoint_format, validates", [("npz", True), ("both", True), ("pickle", False)]
)
def test_checkpointability_validation_is_limited_to_portable_formats(
    monkeypatch, tmp_path: Path, method: str, checkpoint_format: str, validates: bool
) -> None:
    optimizer = _optimizer(_problem(), n_gen=1)
    calls = 0

    def validate() -> None:
        nonlocal calls
        calls += 1
        raise AssertionError("portable checkpoint validation called")

    monkeypatch.setattr(optimizer, "_validate_checkpointable", validate)
    monkeypatch.setattr(optimizer, "_register_checkpoint", lambda *args: None)

    class _Runner:
        def __init__(self, optimizer):
            pass

        def iterate(self):
            return iter(())

        def run(self):
            return None

        def iterate_from(self, ctx):
            return iter(())

        def run_from(self, ctx):
            return None

    monkeypatch.setattr("saealib.optimizer.Runner", _Runner)
    kwargs = {
        "checkpoint_path": tmp_path / "checkpoint",
        "checkpoint_format": checkpoint_format,
    }
    if method.endswith("_from"):

        def invoke():
            return getattr(optimizer, method)(object(), **kwargs)
    else:

        def invoke():
            return getattr(optimizer, method)(**kwargs)

    if validates:
        with pytest.raises(AssertionError, match="portable checkpoint validation"):
            invoke()
    else:
        invoke()
    assert calls == int(validates)


def test_run_from_registers_checkpoints_and_resume_adds_files(tmp_path: Path) -> None:
    problem = _problem()
    checkpoint_dir = tmp_path / "checkpoint"
    mid = _optimizer(problem, seed=4, n_gen=2).run(
        checkpoint_path=checkpoint_dir, checkpoint_interval=1
    )
    before = sorted(checkpoint_dir.glob("checkpoint_*.npz"))
    loaded = OptimizationState.load(before[-1], problem)

    resumed = _optimizer(problem, seed=4, n_gen=4).run_from(
        loaded,
        checkpoint_path=checkpoint_dir,
        checkpoint_interval=1,
    )

    after = sorted(checkpoint_dir.glob("checkpoint_*.npz"))
    assert len(before) == mid.gen == 2
    assert resumed.gen == 4
    assert len(after) > len(before)
    generations = {int(path.stem.rsplit("_", 1)[1]) for path in after}
    assert generations >= {3, 4}


def test_resume_trial_accumulates_prior_wall_time(tmp_path: Path) -> None:
    problem_path = tmp_path / "sphere.py"
    problem_path.write_text(
        "import numpy as np\n"
        "from saealib import Problem\n"
        "problem = Problem(func=lambda x: float(np.sum(x**2)), dim=2, "
        "n_obj=1, direction=np.array([-1.0]), lb=[-2.0, -2.0], "
        "ub=[2.0, 2.0])\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "results"
    initial = ExperimentConfig(
        problems=(problem_path,),
        algorithms=(AlgorithmEntry("default"),),
        seeds=(0,),
        termination={"max_fe": 20},
        output_dir=output_dir,
        checkpoint_interval=1,
    )
    trial = initial.trials()[0]
    ExperimentRunner(initial).run()

    meta_path = output_dir / trial.relative_dir / "meta.json"
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    metadata["wall_time"] = 100.0
    meta_path.write_text(json.dumps(metadata), encoding="utf-8")

    resumed_config = replace(initial, termination={"max_fe": 60})
    result = resume_trial(resumed_config, trial)
    resumed_metadata = json.loads(meta_path.read_text(encoding="utf-8"))

    assert result.wall_time > 100.0
    assert resumed_metadata["wall_time"] > 100.0


def _sphere(x: object) -> float:
    return float(np.sum(np.asarray(x) ** 2))


def _picklable_problem() -> Problem:
    return Problem(
        func=_sphere,
        dim=2,
        n_obj=1,
        direction=np.array([-1.0]),
        lb=[-2.0, -2.0],
        ub=[2.0, 2.0],
    )


@pytest.mark.parametrize("checkpoint_format", ["npz", "both"])
def test_a_portable_checkpoint_run_rejects_a_non_checkpointable_replacement(
    tmp_path: Path, checkpoint_format: str
) -> None:
    optimizer = _optimizer(_picklable_problem(), n_gen=4)
    rejected = _NonCheckpointableEvaluator()

    with pytest.raises(ReplacementRejectedError) as error:
        for state in optimizer.iterate(
            checkpoint_path=tmp_path / "checkpoint",
            checkpoint_format=checkpoint_format,
            checkpoint_interval=1,
        ):
            if state.gen == 1:
                optimizer.set_evaluator(rejected)

    codes = {diagnostic.code for diagnostic in error.value.diagnostics}
    assert codes == {"state_not_checkpointable"}
    assert any(
        "evaluation_submit___evaluator" in str(diagnostic.path)
        for diagnostic in error.value.diagnostics
    )
    published = optimizer._last_published_bindings
    assert published is not None
    assert error.value.version == published.version
    assert optimizer.evaluator is not rejected
    assert isinstance(optimizer.evaluator, SerialEvaluator)
    assert optimizer._configuration_version == published.configuration_version


@pytest.mark.parametrize("checkpoint_format", ["pickle", None])
def test_a_non_portable_run_accepts_a_non_checkpointable_replacement(
    tmp_path: Path, checkpoint_format: str | None
) -> None:
    optimizer = _optimizer(_picklable_problem(), n_gen=4)
    replacement = _NonCheckpointableEvaluator()
    kwargs: dict[str, Any] = (
        {}
        if checkpoint_format is None
        else {
            "checkpoint_path": tmp_path / "checkpoint",
            "checkpoint_format": checkpoint_format,
            "checkpoint_interval": 1,
        }
    )

    final = None
    for final in optimizer.iterate(**kwargs):
        if final.gen == 1:
            optimizer.set_evaluator(replacement)

    assert final is not None
    assert final.gen == 4
    assert optimizer.evaluator is replacement
    published = optimizer._last_published_bindings
    assert published is not None
    assert published.version == 1


def test_a_portable_checkpoint_run_accepts_a_checkpointable_replacement(
    tmp_path: Path,
) -> None:
    optimizer = _optimizer(_picklable_problem(), n_gen=4)
    replacement = SerialEvaluator()

    final = None
    for final in optimizer.iterate(
        checkpoint_path=tmp_path / "checkpoint",
        checkpoint_format="npz",
        checkpoint_interval=1,
    ):
        if final.gen == 1:
            optimizer.set_evaluator(replacement)

    assert final is not None
    assert final.gen == 4
    assert optimizer.evaluator is replacement
    published = optimizer._last_published_bindings
    assert published is not None
    assert published.version == 1
    files = sorted((tmp_path / "checkpoint").glob("checkpoint_*.npz"))
    assert len(files) == final.gen
