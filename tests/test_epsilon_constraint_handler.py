"""
Tests for EpsilonConstraintHandler and schedule helpers.

Tests cover:
- linear_epsilon_schedule / exponential_epsilon_schedule: decay behaviour
- EpsilonConstraintHandler: initial eps, on_generation_end, compute_cv
- compute_cv for equality (bypasses tolerance) and inequality (uses threshold)
- feasibility_threshold tracks internal eps
"""

import numpy as np
import pytest

from saealib import (
    ConstraintHandler,
    EpsilonConstraintHandler,
    EqualityConstraint,
    InequalityConstraint,
    Problem,
    exponential_epsilon_schedule,
    linear_epsilon_schedule,
)

# ---------------------------------------------------------------------------
# linear_epsilon_schedule
# ---------------------------------------------------------------------------


def test_linear_schedule_at_zero():
    s = linear_epsilon_schedule(eps0=1.0, n_gen=10)
    assert s(0) == pytest.approx(1.0)


def test_linear_schedule_at_T():
    s = linear_epsilon_schedule(eps0=1.0, n_gen=10)
    assert s(10) == pytest.approx(0.0)


def test_linear_schedule_clips_zero():
    s = linear_epsilon_schedule(eps0=1.0, n_gen=10)
    assert s(15) == pytest.approx(0.0)


def test_linear_schedule_midpoint():
    s = linear_epsilon_schedule(eps0=2.0, n_gen=10)
    assert s(5) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# exponential_epsilon_schedule
# ---------------------------------------------------------------------------


def test_exponential_schedule_at_zero():
    s = exponential_epsilon_schedule(eps0=2.0, decay=0.9)
    assert s(0) == pytest.approx(2.0)


def test_exponential_schedule_decays():
    s = exponential_epsilon_schedule(eps0=2.0, decay=0.5)
    assert s(1) == pytest.approx(1.0)
    assert s(2) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# EpsilonConstraintHandler — initialisation and lifecycle
# ---------------------------------------------------------------------------


def _make_handler(eps0=1.0, n_gen=10):
    return EpsilonConstraintHandler(linear_epsilon_schedule(eps0=eps0, n_gen=n_gen))


def test_is_constraint_handler():
    assert isinstance(_make_handler(), ConstraintHandler)


def test_initial_eps():
    h = _make_handler(eps0=0.5, n_gen=10)
    assert h.feasibility_threshold == pytest.approx(0.5)


def test_on_generation_end_updates_eps():
    h = _make_handler(eps0=1.0, n_gen=10)
    h.on_generation_end(gen=5, population=None)
    assert h.feasibility_threshold == pytest.approx(0.5)


def test_on_generation_end_reaches_zero():
    h = _make_handler(eps0=1.0, n_gen=10)
    h.on_generation_end(gen=10, population=None)
    assert h.feasibility_threshold == pytest.approx(0.0)


def test_eps_decreases_over_generations():
    h = _make_handler(eps0=1.0, n_gen=5)
    prev = h.feasibility_threshold
    for gen in range(1, 6):
        h.on_generation_end(gen=gen, population=None)
        assert h.feasibility_threshold <= prev
        prev = h.feasibility_threshold
    assert h.feasibility_threshold == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# EpsilonConstraintHandler.compute_cv
# ---------------------------------------------------------------------------

_x = np.zeros(1)


def test_compute_cv_inequality_feasible():
    h = _make_handler()
    c = InequalityConstraint(lambda x: 0.3, threshold=0.5)
    cv = h.compute_cv([c], _x, np.array([0.3]))
    assert cv == pytest.approx(0.0)


def test_compute_cv_inequality_violated():
    h = _make_handler()
    c = InequalityConstraint(lambda x: 0.8, threshold=0.5)
    cv = h.compute_cv([c], _x, np.array([0.8]))
    assert cv == pytest.approx(0.3**2)


def test_compute_cv_equality_raw_absolute_ignores_tolerance():
    """EpsilonConstraintHandler uses |h|, not max(0, |h| - tolerance)."""
    h = _make_handler()
    # tolerance=1e-6 would make cv ≈ 0 with StaticToleranceHandler,
    # but EpsilonConstraintHandler should return |0.01| = 0.01.
    c = EqualityConstraint(lambda x: 0.01, tolerance=1e-6)
    cv = h.compute_cv([c], _x, np.array([0.01]))
    assert cv == pytest.approx(0.01)


def test_compute_cv_equality_negative_value():
    h = _make_handler()
    c = EqualityConstraint(lambda x: -0.05, tolerance=0.0)
    cv = h.compute_cv([c], _x, np.array([-0.05]))
    assert cv == pytest.approx(0.05)


def test_compute_cv_mixed_constraints():
    h = _make_handler()
    ineq = InequalityConstraint(lambda x: 0.3, threshold=0.0)
    eq = EqualityConstraint(lambda x: -0.2, tolerance=0.0)
    cv = h.compute_cv([ineq, eq], _x, np.array([0.3, -0.2]))
    assert cv == pytest.approx(0.3**2 + 0.2)


def test_compute_cv_no_constraints():
    h = _make_handler()
    cv = h.compute_cv([], _x, np.empty(0))
    assert cv == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Problem integration
# ---------------------------------------------------------------------------


def _make_constrained_problem(handler):
    return Problem(
        func=lambda x: float(x[0]),
        dim=1,
        n_obj=1,
        direction=np.array([-1.0]),
        lb=[-1.0],
        ub=[1.0],
        constraints=[EqualityConstraint(lambda x: float(x[0]), tolerance=0.0)],
        handler=handler,
    )


def test_problem_uses_epsilon_handler_compute_cv():
    h = _make_handler(eps0=1.0, n_gen=10)
    prob = _make_constrained_problem(h)
    x = np.array([0.3])
    _, cv = prob.evaluate_constraints(x)
    assert cv == pytest.approx(0.3)


def test_problem_feasibility_threshold_from_handler():
    h = _make_handler(eps0=0.5, n_gen=10)
    prob = _make_constrained_problem(h)
    assert prob.handler.feasibility_threshold == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Runner integration: comparator.eps_cv tracks handler.feasibility_threshold
# ---------------------------------------------------------------------------


def _make_optimizer(problem, n_gen: int):
    from saealib import (
        GA,
        CrossoverBLXAlpha,
        IndividualBasedStrategy,
        LHSInitializer,
        MutationUniform,
        Optimizer,
        SequentialSelection,
        Termination,
        TruncationSelection,
        max_gen,
    )
    from saealib.surrogate.rbf import RBFSurrogate
    from saealib.surrogate.rbf_kernels import GaussianKernel

    return (
        Optimizer(problem)
        .set_initializer(LHSInitializer(n_init_archive=8, n_init_population=6, seed=0))
        .set_algorithm(
            GA(
                crossover=CrossoverBLXAlpha(prob=0.9, alpha=0.4),
                mutation=MutationUniform(prob_var=0.1),
                parent_selection=SequentialSelection(),
                survivor_selection=TruncationSelection(),
            )
        )
        .set_surrogate(RBFSurrogate(GaussianKernel()), n_neighbors=5)
        .set_strategy(IndividualBasedStrategy(evaluation_ratio=1.0))
        .set_termination(Termination(max_gen(n_gen)))
    )


def test_comparator_eps_cv_synced_at_start():
    from saealib import RunStartEvent

    eps0 = 0.5
    h = EpsilonConstraintHandler(linear_epsilon_schedule(eps0=eps0, n_gen=20))
    prob = _make_constrained_problem(h)
    opt = _make_optimizer(prob, n_gen=1)

    eps_at_start = []

    def _capture(e):
        eps_at_start.append(e.ctx.comparator.eps_cv)

    opt.cbmanager.register(RunStartEvent, _capture)
    opt.run()

    assert len(eps_at_start) == 1
    assert eps_at_start[0] == pytest.approx(eps0)


def test_the_schedule_controller_decides_the_relaxation_of_each_generation():
    from saealib import EpsilonSchedule, GenerationStartEvent

    n_gen = 5
    schedule = linear_epsilon_schedule(eps0=1.0, n_gen=n_gen)
    h = EpsilonConstraintHandler(schedule)
    prob = _make_constrained_problem(h)
    opt = _make_optimizer(prob, n_gen=n_gen)
    opt.add_controller(EpsilonSchedule(schedule))

    eps_history = []

    def _capture(e):
        eps_history.append((e.ctx.gen, e.ctx.comparator.eps_cv))

    opt.cbmanager.register(GenerationStartEvent, _capture)
    opt.run()

    assert eps_history
    for gen, epsilon in eps_history:
        assert epsilon == pytest.approx(schedule(gen))


def test_epsilon_relaxation_reaches_the_comparator_only():
    from saealib import EpsilonSchedule, GenerationStartEvent

    n_gen = 5
    schedule = linear_epsilon_schedule(eps0=1.0, n_gen=n_gen)
    h = EpsilonConstraintHandler(schedule)
    prob = _make_constrained_problem(h)
    opt = _make_optimizer(prob, n_gen=n_gen)
    opt.add_controller(EpsilonSchedule(schedule))
    archive_eps = []

    def _check(e):
        assert e.ctx.comparator.eps_cv == pytest.approx(schedule(e.ctx.gen))
        archive_eps.append(e.ctx.pareto_archive.eps_cv)

    opt.cbmanager.register(GenerationStartEvent, _check)
    opt.run()

    assert archive_eps
    assert all(value == pytest.approx(0.0) for value in archive_eps)


def test_standalone_pareto_archive_eps_cv_default_unaffected():
    from saealib.population import ParetoArchive, PopulationAttribute

    attrs = [
        PopulationAttribute(name="x", dtype=np.float64, shape=(1,)),
        PopulationAttribute(name="f", dtype=np.float64, shape=(1,)),
        PopulationAttribute(name="cv", dtype=np.float64, shape=()),
    ]
    archive = ParetoArchive(attrs, init_capacity=5)
    assert archive.eps_cv == pytest.approx(0.0)
