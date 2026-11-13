"""Tests for constraint-defined feasibility and separable cv aggregation."""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from saealib import (
    GA,
    CrossoverBLXAlpha,
    DirectStrategy,
    EpsilonConstraintHandler,
    EqualityConstraint,
    InequalityConstraint,
    LHSInitializer,
    MutationUniform,
    Optimizer,
    Problem,
    SequentialSelection,
    StaticToleranceHandler,
    Termination,
    TruncationSelection,
    linear_epsilon_schedule,
    max_gen,
)
from saealib.comparators import NSGA2Comparator, SingleObjectiveComparator
from saealib.context import OptimizationState
from saealib.decomposition import (
    DecompositionComparator,
    TchebycheffDecomposition,
)
from saealib.execution import SerialEvaluator
from saealib.execution.history import History, record_generation
from saealib.population import Archive, ParetoArchive, Population, PopulationAttribute
from saealib.problem import PymooProblem, sum_violation
from saealib.problem.constraint import _feasible_mask, _feasible_rows
from saealib.result import Result
from saealib.surrogate.training_set import FeasibilityClassificationSet

_DIM = 2


def _constraints() -> list[InequalityConstraint]:
    return [
        InequalityConstraint(lambda x: float(np.asarray(x)[0])),
        EqualityConstraint(lambda x: float(np.asarray(x)[1])),
    ]


def _problem(
    constraints: list[InequalityConstraint] | None = None,
    *,
    cv_aggregator=None,
    n_obj: int = 1,
) -> Problem:
    return Problem(
        func=lambda x: np.sum(np.asarray(x, dtype=float) ** 2) * np.ones(n_obj),
        dim=_DIM,
        n_obj=n_obj,
        direction=np.full(n_obj, -1.0),
        lb=[-1.0, -1.0],
        ub=[1.0, 1.0],
        constraints=_constraints() if constraints is None else constraints,
        cv_aggregator=cv_aggregator,
    )


def _attrs(n_constraints: int = 2, n_obj: int = 1) -> list[PopulationAttribute]:
    return [
        PopulationAttribute("x", np.float64, (_DIM,), np.nan),
        PopulationAttribute("f", np.float64, (n_obj,), np.nan),
        PopulationAttribute("g", np.float64, (n_constraints,), 0.0),
        PopulationAttribute("cv", np.float64, (), 0.0),
        PopulationAttribute("id", np.int64, (), -1),
    ]


def _rows(g_rows: np.ndarray, f_rows: np.ndarray, cv: np.ndarray):
    return {
        "x": np.arange(g_rows.shape[0] * _DIM, dtype=float).reshape(-1, _DIM),
        "f": np.asarray(f_rows, dtype=float),
        "g": np.asarray(g_rows, dtype=float),
        "cv": np.asarray(cv, dtype=float),
    }


def _state(problem: Problem, g_rows, f_rows, cv) -> OptimizationState:
    attrs = _attrs(g_rows.shape[1], np.asarray(f_rows).shape[1])
    population = Population(attrs, 1)
    archive = Archive(attrs, 1)
    population.extend(_rows(g_rows, f_rows, cv))
    archive.extend(_rows(g_rows, f_rows, cv))
    return OptimizationState(
        problem=problem,
        population=population,
        archive=archive,
        pareto_archive=ParetoArchive(
            attrs, 1, direction=problem.direction, constraints=problem.constraints
        ),
        rng=np.random.default_rng(0),
        history=History(),
    )


def test_default_tolerances_follow_research_convention() -> None:
    inequality = InequalityConstraint(lambda x: 0.0)
    equality = EqualityConstraint(lambda x: 0.0)

    assert inequality.threshold == pytest.approx(0.0)
    assert equality.tolerance == pytest.approx(1e-4)
    assert inspect.signature(PymooProblem.__init__).parameters[
        "eq_tolerance"
    ].default == pytest.approx(1e-4)


def test_default_aggregation_is_bit_identical_to_the_static_handler() -> None:
    problem = _problem()
    handler = StaticToleranceHandler()
    points = [
        np.array([0.0, 0.0]),
        np.array([0.1, 1e-4]),
        np.array([-0.2, 9e-5]),
        np.array([-0.3, 2e-4]),
        np.array([5.0, -7.0]),
    ]

    for point in points:
        g, cv = problem.evaluate_constraints(point)
        assert cv == handler.compute_cv(problem.constraints, point, g)
        assert cv == sum_violation(problem.constraints, point, g)

    explicit = _problem(cv_aggregator=sum_violation)
    for point in points:
        _, cv = explicit.evaluate_constraints(point)
        assert cv == problem.evaluate_constraints(point)[1]


def test_plain_function_aggregator_replaces_the_handler_aggregation() -> None:
    def worst_violation(constraints, x, g) -> float:
        return max(
            (c.violation_from_value(float(gi)) for gi, c in zip(g, constraints)),
            default=0.0,
        )

    problem = _problem(cv_aggregator=worst_violation)
    point = np.array([0.3, 5e-4])

    _, cv = problem.evaluate_constraints(point)

    assert cv == pytest.approx(0.3)
    assert cv == worst_violation(problem.constraints, point, point)
    assert cv != StaticToleranceHandler().compute_cv(problem.constraints, point, point)


class _BatchProblem(Problem):
    def evaluate_batch(self, x):
        values = np.asarray(x, dtype=float)
        f = np.sum(values**2, axis=1)[:, None]
        g = np.column_stack([values[:, 0], values[:, 1]])
        return f, g


def test_batch_evaluation_uses_the_configured_aggregator() -> None:
    def doubled(constraints, x, g) -> float:
        return 2.0 * sum_violation(constraints, x, g)

    problems = [
        _problem(cv_aggregator=doubled),
        _problem(cv_aggregator=doubled),
    ]
    batch_points = np.array([[0.0, 0.0], [0.5, 1e-4], [0.5, 5e-4]])
    problems[0] = _BatchProblem(
        func=lambda x: np.sum(np.asarray(x, dtype=float) ** 2),
        dim=_DIM,
        n_obj=1,
        direction=np.array([-1.0]),
        lb=[-1.0, -1.0],
        ub=[1.0, 1.0],
        constraints=_constraints(),
        cv_aggregator=doubled,
    )

    result = SerialEvaluator().evaluate_batch(batch_points, problems[0])

    expected = [
        doubled(problems[0].constraints, point, np.array([point[0], point[1]]))
        for point in batch_points
    ]
    np.testing.assert_array_equal(result.cv, np.asarray(expected, dtype=float))


_BOUNDARY_ROWS = np.array(
    [
        [0.0, 0.0],
        [0.0, 1e-4],
        [0.0, 1.0001e-4],
        [-1e-12, 0.0],
        [1e-12, 0.0],
        [-0.5, -1e-4],
    ]
)
_BOUNDARY_FEASIBLE = np.array([True, True, False, True, False, True])
_BOUNDARY_F = np.array([[index] for index in range(len(_BOUNDARY_ROWS))], dtype=float)


def test_result_history_training_set_and_archive_agree_at_tolerances() -> None:
    problem = _problem()
    cv = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    state = _state(problem, _BOUNDARY_ROWS, _BOUNDARY_F, cv)

    np.testing.assert_array_equal(
        _feasible_mask(problem.constraints, _BOUNDARY_ROWS), _BOUNDARY_FEASIBLE
    )
    np.testing.assert_array_equal(
        _feasible_rows(state.archive, problem.constraints), _BOUNDARY_FEASIBLE
    )

    expected_best = float(np.min(_BOUNDARY_F[_BOUNDARY_FEASIBLE]))
    assert Result.from_state(state).f[0] == pytest.approx(expected_best)

    record_generation(state)
    assert state.history is not None
    summary = state.history.channel("summary")
    assert summary["best_f"][0] == pytest.approx(expected_best)
    assert summary["feasible_ratio"][0] == pytest.approx(4 / 6)

    labels = (
        FeasibilityClassificationSet(source="archive")
        .build(state.archive, None, ctx=state)
        .train_y
    )
    np.testing.assert_array_equal(labels, _BOUNDARY_FEASIBLE.astype(float))

    archive = ParetoArchive(
        _attrs(), 10, direction=problem.direction, constraints=problem.constraints
    )
    archive.add(
        x=np.array([0.0, 0.0]),
        f=np.array([2.0]),
        g=np.array([0.0, 9e-5]),
        cv=9e-5,
        id=1,
    )
    archive.add(
        x=np.array([1.0, 1.0]),
        f=np.array([0.5]),
        g=np.array([1e-5, 0.0]),
        cv=1e-5,
        id=2,
    )

    np.testing.assert_array_equal(archive.get_array("id"), [1])


def test_epsilon_aggregation_keeps_its_formula_and_is_a_search_side_exception() -> None:
    handler = EpsilonConstraintHandler(linear_epsilon_schedule(eps0=0.5, n_gen=4))
    constraints = _constraints()
    point = np.array([0.3, 5e-5])
    g = np.array([0.3, 5e-5])

    cv = handler.compute_cv(constraints, point, g)

    assert cv == pytest.approx(0.3**2 + 5e-5)
    assert cv > 0.0
    assert bool(_feasible_mask(constraints, g[np.newaxis, :])[0]) is False
    equality_feasible = np.array([-0.1, 5e-5])
    assert (
        bool(_feasible_mask(constraints, equality_feasible[np.newaxis, :])[0]) is True
    )
    assert handler.compute_cv(
        constraints, equality_feasible, equality_feasible
    ) == pytest.approx(abs(5e-5))


def test_reported_feasibility_ignores_the_epsilon_relaxation() -> None:
    handler = EpsilonConstraintHandler(linear_epsilon_schedule(eps0=0.5, n_gen=4))
    problem = _problem()
    problem.handler = handler
    g_rows = np.array([[0.0, 5e-5], [0.3, 0.0]])
    state = _state(problem, g_rows, np.zeros((2, 1)), np.array([5e-5, 0.09]))

    assert handler.feasibility_threshold == pytest.approx(0.5)
    state.comparator.eps_cv = handler.feasibility_threshold

    record_generation(state)

    assert state.history is not None
    summary = state.history.channel("summary")
    assert summary["feasible_ratio"][0] == pytest.approx(0.5)
    assert Result.from_state(state).f[0] == pytest.approx(0.0)


def test_problem_eps_cv_is_deprecated_and_relaxes_search_only() -> None:
    with pytest.warns(DeprecationWarning, match=r"Problem\(eps_cv=\.\.\.\)"):
        problem = _problem()
        deprecated = Problem(
            func=problem.func,
            dim=_DIM,
            n_obj=1,
            direction=problem.direction,
            lb=[-1.0, -1.0],
            ub=[1.0, 1.0],
            eps_cv=1e-3,
            constraints=_constraints(),
        )

    assert _problem().eps_cv == pytest.approx(0.0)
    assert deprecated.eps_cv == pytest.approx(1e-3)
    assert deprecated.handler.feasibility_threshold == pytest.approx(1e-3)
    assert deprecated.comparator.eps_cv == pytest.approx(1e-3)

    g_rows = np.array([[0.0, 5e-5]])
    state = _state(deprecated, g_rows, np.zeros((1, 1)), np.array([5e-4]))
    record_generation(state)

    assert state.history is not None
    summary = state.history.channel("summary")
    assert summary["feasible_ratio"][0] == pytest.approx(1.0)


def test_comparator_archive_and_decomposition_share_one_default() -> None:
    decomposition = TchebycheffDecomposition()
    archive = ParetoArchive(_attrs(), 4, direction=np.array([-1.0]))

    assert SingleObjectiveComparator().eps_cv == pytest.approx(0.0)
    assert NSGA2Comparator().eps_cv == pytest.approx(0.0)
    comparator = DecompositionComparator(decomposition, np.array([0.5, 0.5]))
    assert comparator.eps_cv == pytest.approx(0.0)
    assert archive.eps_cv == pytest.approx(0.0)
    assert _problem().eps_cv == pytest.approx(0.0)
    assert StaticToleranceHandler().feasibility_threshold == pytest.approx(0.0)


def test_run_created_archive_carries_the_problem_constraints() -> None:
    problem = _problem()
    optimizer = (
        Optimizer(problem, seed=3)
        .set_initializer(LHSInitializer(6, 4, seed=1))
        .set_strategy(DirectStrategy())
        .set_termination(Termination(max_gen(1)))
    )

    state = optimizer.run()

    assert state.pareto_archive.constraints == problem.constraints
    assert state.comparator.eps_cv == pytest.approx(0.0)


def test_checkpoint_round_trip_keeps_archive_constraints(tmp_path) -> None:
    problem = _problem()
    optimizer = (
        Optimizer(problem, seed=5)
        .set_initializer(LHSInitializer(6, 4, seed=2))
        .set_strategy(DirectStrategy())
        .set_termination(Termination(max_gen(1)))
    )
    state = optimizer.run()
    path = tmp_path / "state"
    state.save(path)

    loaded = OptimizationState.load(path, problem)

    assert loaded.pareto_archive.constraints == problem.constraints
    assert loaded.pareto_archive.eps_cv == pytest.approx(state.pareto_archive.eps_cv)


class _ArchiveFactoryWithoutConstraintsParameter(GA):
    """Algorithm whose Pareto archive factory takes no ``constraints`` argument."""

    def create_pareto_archive(self, attrs, init_capacity, problem):
        return ParetoArchive(attrs, init_capacity, direction=problem.direction)


def _archive_factory_optimizer(problem: Problem, n_gen: int = 1) -> Optimizer:
    algorithm = _ArchiveFactoryWithoutConstraintsParameter(
        crossover=CrossoverBLXAlpha(prob=0.9, alpha=0.4),
        mutation=MutationUniform(prob_var=0.1),
        parent_selection=SequentialSelection(),
        survivor_selection=TruncationSelection(),
    )
    return (
        Optimizer(problem, seed=9)
        .set_initializer(LHSInitializer(6, 4, seed=3))
        .set_algorithm(algorithm)
        .set_strategy(DirectStrategy())
        .set_termination(Termination(max_gen(n_gen)))
    )


def test_archive_factory_without_constraints_parameter_still_gets_them() -> None:
    problem = _problem()

    state = _archive_factory_optimizer(problem).run()

    assert state.pareto_archive.constraints == problem.constraints
    flags = state.pareto_archive._feasible_flags([1e-5], [[0.0, 5e-5]])
    assert bool(flags[0]) is True


def test_archive_factory_without_constraints_parameter_survives_checkpoint(
    tmp_path,
) -> None:
    problem = _problem()
    optimizer = _archive_factory_optimizer(problem)
    state = optimizer.run()
    path = tmp_path / "state"
    state.save(path)

    loaded = OptimizationState.load(path, problem)
    resumed = _archive_factory_optimizer(problem, n_gen=2).run_from(loaded)

    assert loaded.pareto_archive.constraints == problem.constraints
    assert resumed.pareto_archive.constraints == problem.constraints


_RELAXED_CV = (0.0534, 0.01, 0.5, 0.3677, 0.3827, 0.1312, 0.4369)


def _restore_objective(x) -> np.ndarray:
    first = np.asarray(x, dtype=float)[0]
    return np.array([first, 0.51 - first])


def _restore_inequality(x) -> float:
    return float(np.asarray(x, dtype=float)[0])


def _restore_problem() -> Problem:
    return Problem(
        func=_restore_objective,
        dim=2,
        n_obj=2,
        direction=np.array([-1.0, -1.0]),
        lb=[0.01, 0.01],
        ub=[0.5, 0.5],
        constraints=[InequalityConstraint(_restore_inequality)],
    )


def _relaxed_state(problem: Problem) -> OptimizationState:
    """State whose Pareto archive kept rows under a relaxed ``eps_cv``."""
    attrs = _attrs(1, 2)
    archive = ParetoArchive(attrs, 10, direction=problem.direction, eps_cv=0.5)
    for index, cv in enumerate(_RELAXED_CV):
        archive.add(
            x=np.array([cv, 0.5 - cv]),
            f=np.array([cv, 0.51 - cv]),
            g=np.array([cv]),
            cv=cv,
            id=index,
        )
    assert len(archive) == len(_RELAXED_CV)
    main = Archive(attrs, 1)
    main._extend_internal(
        {name: archive.get_array(name).copy() for name in ("x", "f", "g", "cv", "id")},
        preserve_ids=True,
    )
    return OptimizationState(
        problem=problem,
        population=Population(attrs, 1),
        archive=main,
        pareto_archive=archive,
        rng=np.random.default_rng(0),
    )


def test_portable_checkpoint_rebuilds_the_archive_under_the_bound_constraints(
    tmp_path,
) -> None:
    problem = _restore_problem()
    state = _relaxed_state(problem)
    path = tmp_path / "state"
    state.save(path)

    loaded = OptimizationState.load(path, problem)

    archive = loaded.pareto_archive
    assert len(archive) == 1
    assert archive.get_array("cv")[0] == pytest.approx(min(_RELAXED_CV))
    assert archive.get_array("id")[0] == _RELAXED_CV.index(min(_RELAXED_CV))
    np.testing.assert_array_equal(
        archive.get_array("x")[0], np.array([min(_RELAXED_CV), 0.5 - min(_RELAXED_CV)])
    )
    np.testing.assert_array_equal(
        archive.get_array("g")[0], np.array([min(_RELAXED_CV)])
    )
    result = Result.from_state(loaded)
    assert result.f.shape == (1, 2)
    assert Result.from_state(loaded).f[0, 0] == pytest.approx(min(_RELAXED_CV))


def test_pickle_restore_rebuilds_the_archive_under_the_bound_constraints(
    tmp_path,
) -> None:
    import warnings

    from saealib.execution.runner import Runner

    problem = _restore_problem()
    state = _relaxed_state(problem)
    optimizer = (
        Optimizer(problem, seed=1)
        .set_algorithm(
            GA(
                crossover=CrossoverBLXAlpha(prob=0.9, alpha=0.4),
                mutation=MutationUniform(prob_var=0.1),
                parent_selection=SequentialSelection(),
                survivor_selection=TruncationSelection(),
            )
        )
        .set_strategy(DirectStrategy())
        .set_termination(Termination(max_gen(1)))
    )
    path = tmp_path / "state.pkl"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        optimizer.save_pickle(state, path)
        restored, loaded = Optimizer.load_pickle(path)

    iterator = Runner(restored).iterate_from(loaded)
    try:
        resumed = next(iterator)
    finally:
        iterator.close()

    assert len(resumed.pareto_archive) == 1
    assert resumed.pareto_archive.get_array("cv")[0] == pytest.approx(min(_RELAXED_CV))


class _ArchiveWithoutConstraintsParameter(ParetoArchive):
    def __init__(self, attrs, init_capacity, direction=None):
        super().__init__(attrs=attrs, init_capacity=init_capacity, direction=direction)


class _CustomArchiveAlgorithm(GA):
    @property
    def pareto_archive_class(self):
        return _ArchiveWithoutConstraintsParameter


def _custom_archive_optimizer(problem: Problem, n_gen: int = 2) -> Optimizer:
    return (
        Optimizer(problem, seed=3)
        .set_initializer(LHSInitializer(6, 4, seed=1))
        .set_algorithm(
            _CustomArchiveAlgorithm(
                crossover=CrossoverBLXAlpha(prob=0.9, alpha=0.4),
                mutation=MutationUniform(prob_var=0.1),
                parent_selection=SequentialSelection(),
                survivor_selection=TruncationSelection(),
            )
        )
        .set_strategy(DirectStrategy())
        .set_termination(Termination(max_gen(n_gen)))
    )


def test_archive_class_without_constraints_parameter_runs(tmp_path) -> None:
    problem = _problem()
    state = _custom_archive_optimizer(problem).run()

    assert isinstance(state.pareto_archive, _ArchiveWithoutConstraintsParameter)
    assert state.pareto_archive.constraints == problem.constraints

    path = tmp_path / "state"
    state.save(path)
    loaded = OptimizationState.load(path, problem)

    assert loaded.pareto_archive.constraints == problem.constraints
    np.testing.assert_array_equal(
        loaded.pareto_archive.get_array("x"), state.pareto_archive.get_array("x")
    )


def _constrained_sphere(x) -> np.ndarray:
    return np.array([float(np.sum(np.asarray(x, dtype=float) ** 2))])


def _constrained_sphere_constraint(x) -> float:
    return float(np.asarray(x, dtype=float)[0] - 0.2)


def test_pickle_without_cv_aggregator_resumes_constrained_evaluations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys
    import warnings
    from pathlib import Path

    monkeypatch.setattr(
        sys.modules["__main__"], "sphere", _constrained_sphere, raising=False
    )
    monkeypatch.setattr(
        sys.modules["__main__"],
        "constraint",
        _constrained_sphere_constraint,
        raising=False,
    )
    checkpoint = Path(__file__).parent / "fixtures" / "constrained_problem_pickle.pkl"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        optimizer, state = Optimizer.load_pickle(checkpoint)
    problem = optimizer.problem
    assert problem.constraints
    assert problem.cv_aggregator is None
    g, cv = problem.evaluate_constraints(np.array([0.5, 0.5]))
    assert g[0] == pytest.approx(0.3)
    assert cv == pytest.approx(0.3)

    optimizer.set_termination(Termination(max_gen(3)))
    resumed = optimizer.run_from(state)

    assert (resumed.gen, resumed.fe) == (3, 20)
    x = resumed.archive.get_array("x")
    f = resumed.archive.get_array("f")
    assert np.all(np.isfinite(x))
    assert np.all(np.isfinite(f))
    np.testing.assert_allclose(
        resumed.archive.get_array("cv"),
        np.maximum(0.0, resumed.archive.get_array("g")[:, 0]),
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        again, again_state = Optimizer.load_pickle(checkpoint)
    again.set_termination(Termination(max_gen(3)))
    repeated = again.run_from(again_state)
    np.testing.assert_array_equal(repeated.archive.get_array("x"), x)
    np.testing.assert_array_equal(repeated.archive.get_array("f"), f)
