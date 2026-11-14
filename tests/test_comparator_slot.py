"""Tests for the comparator as an Optimizer binding slot."""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest

from saealib import (
    GA,
    CrossoverBLXAlpha,
    DirectStrategy,
    EpsilonConstraintHandler,
    EqualityConstraint,
    IndividualBasedStrategy,
    LHSInitializer,
    MutationUniform,
    Optimizer,
    Problem,
    SequentialSelection,
    Termination,
    TruncationSelection,
    linear_epsilon_schedule,
    max_gen,
)
from saealib.comparators import (
    NSGA2Comparator,
    NSGA3Comparator,
    ParetoComparator,
    SingleObjectiveComparator,
    WeightedSumComparator,
)
from saealib.context import OptimizationState
from saealib.core.compiler import ContractPath, Severity
from saealib.core.compiler.contract_diagnostics import (
    _optimizer_configuration_diagnostics,
)
from saealib.execution import SerialEvaluator
from saealib.execution.bindings import resolve_components
from saealib.registry import build, to_spec
from saealib.surrogate.rbf import RBFSurrogate
from saealib.surrogate.rbf_kernels import GaussianKernel

_SRC = Path(__file__).parents[1] / "src" / "saealib"


def _problem(n_obj: int = 1, dim: int = 2) -> Problem:
    return Problem(
        func=lambda x: np.sum(np.asarray(x, dtype=float) ** 2) * np.ones(n_obj),
        dim=dim,
        n_obj=n_obj,
        direction=np.array([-1.0] * n_obj),
        lb=[-2.0] * dim,
        ub=[2.0] * dim,
    )


def _optimizer(problem: Problem, n_gen: int = 3) -> Optimizer:
    return (
        Optimizer(problem, seed=7)
        .set_initializer(LHSInitializer(n_init_archive=6, n_init_population=4, seed=3))
        .set_termination(Termination(max_gen(n_gen)))
    )


def test_set_comparator_is_used_and_recorded_as_an_explicit_setting() -> None:
    comparator = ParetoComparator(direction=np.array([1.0]))
    optimizer = _optimizer(_problem()).set_comparator(comparator)

    state = optimizer.run()

    assert state.comparator is comparator
    assert optimizer.comparator is comparator
    origin = optimizer._component_origins["comparator"]
    assert origin.source == "setter"
    assert "set_comparator" in origin.reason


def test_default_comparator_follows_the_objective_count_and_records_its_origin() -> (
    None
):
    single = _optimizer(_problem(n_obj=1))
    single.resolve_defaults()
    multi = _optimizer(_problem(n_obj=2))
    multi.resolve_defaults()

    assert isinstance(single.comparator, SingleObjectiveComparator)
    assert isinstance(multi.comparator, NSGA2Comparator)
    for optimizer in (single, multi):
        origin = optimizer._component_origins["comparator"]
        assert origin.source == "framework default"
        assert "number of objectives" in origin.reason


def test_explicit_comparator_takes_precedence_over_the_default() -> None:
    problem = _problem(n_obj=2)
    explicit = ParetoComparator(direction=np.array([-1.0, -1.0]))
    optimizer = _optimizer(problem)
    default_multi = optimizer.comparator
    optimizer.resolve_defaults()
    default_multi = optimizer.comparator
    optimizer.set_comparator(explicit)

    state = optimizer.run()

    assert isinstance(default_multi, NSGA2Comparator)
    assert state.comparator is explicit


def test_comparator_direction_is_bound_from_the_problem_on_entry() -> None:
    optimizer = _optimizer(_problem(n_obj=2)).set_comparator(ParetoComparator())

    optimizer.resolve_defaults()

    resolved = optimizer.comparator
    assert resolved is not None
    np.testing.assert_array_equal(resolved.direction, np.array([-1.0, -1.0]))


def test_runtime_reads_the_comparator_from_the_state() -> None:
    problems = sorted(_SRC.rglob("*.py"))
    direct_reads = {
        path.relative_to(_SRC).as_posix(): line
        for path in problems
        for line in path.read_text(encoding="utf-8").splitlines()
        if "problem.comparator" in line
    }

    assert set(direct_reads) <= {"optimizer.py"}, direct_reads


def test_deprecated_problem_comparator_keeps_driving_the_run() -> None:
    comparator = ParetoComparator(direction=np.array([1.0]))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        problem = _problem()
        problem = Problem(
            func=problem.func,
            dim=problem.dim,
            n_obj=problem.n_obj,
            direction=problem.direction,
            lb=problem.lb,
            ub=problem.ub,
            comparator=comparator,
        )
    assert any(
        "Problem(comparator=...)" in str(item.message)
        and issubclass(item.category, DeprecationWarning)
        for item in caught
    )
    optimizer = _optimizer(problem)

    assert optimizer.comparator is comparator
    assert (
        optimizer._component_origins["comparator"].source
        == "deprecated Problem argument"
    )

    state = optimizer.run()

    assert state.comparator is comparator


def test_deprecated_problem_eps_obj_keeps_seeding_the_default_comparator() -> None:
    with pytest.warns(DeprecationWarning, match=r"Problem\(eps_obj=\.\.\.\)"):
        problem = Problem(
            func=lambda x: float(np.sum(np.asarray(x, dtype=float) ** 2)),
            dim=2,
            n_obj=1,
            direction=np.array([-1.0]),
            lb=[-2.0, -2.0],
            ub=[2.0, 2.0],
            eps_obj=1e-3,
        )
    optimizer = _optimizer(problem)
    optimizer.resolve_defaults()

    resolved = optimizer.comparator
    assert resolved is not None
    assert resolved.eps_obj == pytest.approx(1e-3)
    assert problem.comparator.eps_obj == pytest.approx(1e-3)


def _constrained_problem(n_gen: int) -> Problem:
    handler = EpsilonConstraintHandler(linear_epsilon_schedule(eps0=0.4, n_gen=n_gen))
    return Problem(
        func=lambda x: float(x[0]) ** 2,
        dim=1,
        n_obj=1,
        direction=np.array([-1.0]),
        lb=[-1.0],
        ub=[1.0],
        constraints=[EqualityConstraint(lambda x: float(x[0]), tolerance=0.0)],
        handler=handler,
    )


def _constrained_optimizer(n_gen: int) -> Optimizer:
    return (
        Optimizer(_constrained_problem(n_gen), seed=11)
        .set_initializer(LHSInitializer(n_init_archive=6, n_init_population=4, seed=2))
        .set_algorithm(
            GA(
                crossover=CrossoverBLXAlpha(prob=0.9, alpha=0.4),
                mutation=MutationUniform(prob_var=0.1),
                parent_selection=SequentialSelection(),
                survivor_selection=TruncationSelection(),
            )
        )
        .set_surrogate(RBFSurrogate(GaussianKernel()), n_neighbors=4)
        .set_strategy(IndividualBasedStrategy(evaluation_ratio=1.0))
        .set_termination(Termination(max_gen(n_gen)))
    )


def test_threshold_copy_keeps_the_comparator_binding_and_plan_signature(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from saealib import GenerationEndEvent, RunStartEvent
    from saealib import optimizer as optimizer_module

    n_gen = 4
    optimizer = _constrained_optimizer(n_gen)
    preparations: list[object] = []
    real_prepare = optimizer_module._prepare_plan

    def counted_prepare(*args, **kwargs):
        preparations.append(args)
        return real_prepare(*args, **kwargs)

    monkeypatch.setattr(optimizer_module, "_prepare_plan", counted_prepare)

    seen: list[object] = []
    signatures: list[tuple[object, ...]] = []
    versions: list[int] = []

    def signature() -> tuple[object, ...]:
        resolution = optimizer._component_resolution
        assert resolution is not None
        return optimizer._diagnostic_plan_signature(resolution)

    def on_start(event) -> None:
        signatures.append(signature())
        seen.append(event.ctx.comparator)

    def on_end(event) -> None:
        signatures.append(signature())
        seen.append(event.ctx.comparator)
        versions.append(optimizer._configuration_version)

    optimizer.cbmanager.register(RunStartEvent, on_start)
    optimizer.cbmanager.register(GenerationEndEvent, on_end)
    state = optimizer.run()

    assert len(seen) == n_gen + 1
    assert all(comparator is state.comparator for comparator in seen)
    assert len(set(signatures)) == 1
    assert len(set(versions)) == 1
    assert len(preparations) == 1


def test_islands_create_a_default_comparator_per_optimizer() -> None:
    problem = _problem(n_obj=2)
    first = _optimizer(problem, n_gen=1)
    second = _optimizer(problem, n_gen=1)
    first.resolve_defaults()
    second.resolve_defaults()

    assert first.comparator is not second.comparator


def test_islands_keep_an_explicitly_shared_comparator_shared() -> None:
    problem = _problem(n_obj=2)
    shared = ParetoComparator(direction=np.array([-1.0, -1.0]))
    first = _optimizer(problem, n_gen=1).set_comparator(shared)
    second = _optimizer(problem, n_gen=1).set_comparator(shared)
    first.resolve_defaults()
    second.resolve_defaults()

    assert first.comparator is second.comparator is shared


def test_nsga3_set_with_set_comparator_defines_the_default_population() -> None:
    reference_points = np.full((13, 3), 1.0 / 3.0)
    optimizer = Optimizer(_problem(n_obj=3)).set_comparator(
        NSGA3Comparator(reference_points=reference_points)
    )

    optimizer.resolve_defaults()

    initializer = optimizer.initializer
    assert isinstance(initializer, LHSInitializer)
    assert initializer.n_init_population == 16


def test_state_loaded_without_an_optimizer_uses_the_problem_default(tmp_path) -> None:
    problem = _problem(n_obj=1)
    optimizer = _optimizer(problem, n_gen=1)
    state = optimizer.run()
    path = tmp_path / "state"
    state.save(path)

    loaded = OptimizationState.load(path, problem)

    assert isinstance(loaded.comparator, SingleObjectiveComparator)


def test_comparator_spec_round_trip() -> None:
    comparator = SingleObjectiveComparator(direction=1.0, eps_cv=0.5, eps_obj=1e-3)

    rebuilt = build(to_spec(comparator))

    assert isinstance(rebuilt, SingleObjectiveComparator)
    assert rebuilt.eps_cv == pytest.approx(0.5)
    assert rebuilt.eps_obj == pytest.approx(1e-3)


def test_reference_point_dimension_mismatch_is_reported() -> None:
    problem = _problem(n_obj=2)
    comparator = NSGA3Comparator(reference_points=np.full((4, 3), 1.0 / 3.0))
    resolution = resolve_components(Optimizer(problem).set_comparator(comparator))

    findings = _optimizer_configuration_diagnostics(
        resolution.components,
        {},
        problem,
        None,
    )

    mismatch = [
        item for item in findings if item.code == "reference_points_dimension_mismatch"
    ]
    assert len(mismatch) == 1
    assert mismatch[0].severity is Severity.ERROR
    assert mismatch[0].path == ContractPath(components=("optimizer", "comparator"))


def test_objective_count_mismatch_is_reported() -> None:
    problem = _problem(n_obj=1)
    comparator = WeightedSumComparator(direction=np.array([1.0, 1.0]))
    optimizer = Optimizer(problem).set_comparator(comparator)

    issues = optimizer.validate()

    assert any("objective_count_mismatch" in issue for issue in issues)
    assert any(
        item.code == "objective_count_mismatch"
        for item in optimizer._last_contract_diagnostics
    )


def test_resuming_with_an_optimizer_uses_that_optimizer_comparator(tmp_path) -> None:
    problem = _problem(n_obj=2)
    state = _optimizer(problem, n_gen=1).run()
    path = tmp_path / "state"
    state.save(path)
    loaded = OptimizationState.load(path, problem)
    replacement = ParetoComparator(direction=np.array([-1.0, -1.0]))
    resuming = _optimizer(problem, n_gen=1).set_comparator(replacement)

    resumed = resuming.run_from(loaded)

    assert resumed.comparator is replacement


def test_loading_a_pickle_without_comparator_slot_adopts_the_problem_comparator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys

    def sphere(x):
        return np.array([float(np.sum(np.asarray(x) ** 2))])

    monkeypatch.setattr(sys.modules["__main__"], "sphere", sphere, raising=False)
    checkpoint = (
        Path(__file__).parent
        / "fixtures"
        / "optimizer_pickle_without_comparator_slot.pkl"
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        optimizer, state = Optimizer.load_pickle(checkpoint)

    assert optimizer.problem.comparator.eps_obj == pytest.approx(0.25)
    assert optimizer.comparator is optimizer.problem.comparator
    assert optimizer._component_origins["comparator"].source == "persisted"
    assert optimizer.comparator.compare(np.array([1.0]), 0.0, np.array([1.2]), 0.0) == 0

    resumed = optimizer.run_from(state)

    assert resumed.comparator.eps_obj == pytest.approx(0.25)


def test_restored_problem_comparator_is_explicit_only_when_it_differs() -> None:
    from types import SimpleNamespace

    from saealib.optimizer import _restored_problem_comparator

    default = SingleObjectiveComparator(direction=-1.0, eps_cv=1e-6)
    restored_default = SimpleNamespace(
        n_obj=1,
        direction=np.array([-1.0]),
        eps_cv=1e-6,
        eps_obj=1e-6,
        comparator=SingleObjectiveComparator(direction=-1.0, eps_cv=1e-6),
    )
    restored_explicit = SimpleNamespace(
        n_obj=1,
        direction=np.array([-1.0]),
        eps_cv=1e-6,
        eps_obj=1e-6,
        comparator=SingleObjectiveComparator(direction=-1.0, eps_cv=1e-6, eps_obj=0.25),
    )

    assert _restored_problem_comparator(restored_default) is None
    assert _restored_problem_comparator(restored_explicit) is not None
    assert (
        _restored_problem_comparator(
            SimpleNamespace(comparator=default, _comparator_explicit=False)
        )
        is None
    )


class _RollbackEvaluator(SerialEvaluator):
    def supports_batch_rollback(self) -> bool:
        return True


def _swap_optimizer(mode: str) -> tuple[Optimizer, object, list[dict]]:
    from saealib import GenerationEndEvent, GenerationStartEvent
    from saealib.execution import AsyncEvaluationScheduler

    problem = Problem(
        func=lambda x: np.array([float(np.sum(np.asarray(x, dtype=float) ** 2))]),
        dim=2,
        n_obj=1,
        direction=np.array([-1.0]),
        lb=[-1.0, -1.0],
        ub=[1.0, 1.0],
    )
    replacement = SingleObjectiveComparator(direction=1.0)
    optimizer = (
        Optimizer(problem, seed=7)
        .set_comparator(SingleObjectiveComparator(direction=-1.0))
        .set_initializer(LHSInitializer(8, 4))
        .set_strategy(DirectStrategy())
        .set_termination(Termination(max_gen(3)))
    )
    if mode == "async":
        evaluator = _RollbackEvaluator()
        optimizer = optimizer.set_evaluator(evaluator).set_async_evaluation_scheduler(
            AsyncEvaluationScheduler(evaluator, max_pending=8)
        )
    observations: list[dict] = []

    def on_start(event) -> None:
        observations.append(
            {
                "phase": "start",
                "gen": event.ctx.gen,
                "state": event.ctx.comparator is replacement,
                "binding_version": event.ctx.__dict__.get("_binding_version"),
            }
        )

    def on_end(event) -> None:
        observations.append(
            {
                "phase": "end",
                "gen": event.ctx.gen,
                "state": event.ctx.comparator is replacement,
                "optimizer": optimizer.comparator is replacement,
                "binding_version": event.ctx.__dict__.get("_binding_version"),
            }
        )
        if event.ctx.gen == 1:
            optimizer.set_comparator(replacement)

    optimizer.cbmanager.register(GenerationStartEvent, on_start)
    optimizer.cbmanager.register(GenerationEndEvent, on_end)
    return optimizer, replacement, observations


@pytest.mark.parametrize("mode", ("sync", "async"))
def test_replacing_the_comparator_during_a_run_updates_the_state(mode: str) -> None:
    optimizer, replacement, observations = _swap_optimizer(mode)

    state = optimizer.run()

    first_end = next(
        item for item in observations if item["phase"] == "end" and item["gen"] == 1
    )
    assert first_end["state"] is False
    assert first_end["optimizer"] is False
    for item in observations:
        if item["gen"] >= 2:
            assert item["state"] is True
            assert item["binding_version"] == 1
        if item["phase"] == "end" and item["gen"] >= 2:
            assert item["optimizer"] is True
    assert state.comparator is replacement
    assert optimizer.comparator is replacement
    assert replacement.eps_cv == pytest.approx(
        state.problem.handler.feasibility_threshold
    )


@pytest.mark.parametrize("mode", ("sync", "async"))
def test_replacing_the_comparator_follows_the_running_schedule(mode: str) -> None:
    from saealib import (
        EpsilonConstraintHandler,
        EpsilonSchedule,
        GenerationEndEvent,
        GenerationStartEvent,
    )
    from saealib.execution import AsyncEvaluationScheduler

    schedule = linear_epsilon_schedule(eps0=0.4, n_gen=3)
    handler = EpsilonConstraintHandler(schedule)
    problem = Problem(
        func=lambda x: float(np.asarray(x, dtype=float)[0]) ** 2,
        dim=1,
        n_obj=1,
        direction=np.array([-1.0]),
        lb=[-1.0],
        ub=[1.0],
        constraints=[
            EqualityConstraint(lambda x: float(np.asarray(x)[0]), tolerance=0.0)
        ],
        handler=handler,
    )
    replacement = SingleObjectiveComparator(direction=-1.0)
    optimizer = (
        Optimizer(problem, seed=11)
        .set_initializer(LHSInitializer(6, 4, seed=1))
        .set_strategy(DirectStrategy())
        .set_termination(Termination(max_gen(3)))
    )
    if mode == "async":
        evaluator = _RollbackEvaluator()
        optimizer = optimizer.set_evaluator(evaluator).set_async_evaluation_scheduler(
            AsyncEvaluationScheduler(evaluator, max_pending=8)
        )
    optimizer.add_controller(EpsilonSchedule(schedule))
    thresholds: list[tuple[int, float, float]] = []

    def on_start(event) -> None:
        if event.ctx.gen >= 2:
            thresholds.append(
                (
                    event.ctx.gen,
                    event.ctx.comparator.eps_cv,
                    float(schedule(event.ctx.gen)),
                )
            )

    def on_end(event) -> None:
        if event.ctx.gen == 1:
            optimizer.set_comparator(replacement)

    optimizer.cbmanager.register(GenerationStartEvent, on_start)
    optimizer.cbmanager.register(GenerationEndEvent, on_end)

    state = optimizer.run()

    assert thresholds
    for _, eps_cv, scheduled in thresholds:
        assert eps_cv == pytest.approx(scheduled)
    assert state.comparator is replacement
    assert state.comparator.eps_cv == pytest.approx(schedule(state.gen))
