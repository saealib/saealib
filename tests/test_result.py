"""Tests for result construction and state delegation."""

from types import SimpleNamespace

import numpy as np
import pytest

from saealib import (
    DirectStrategy,
    GenomeInitializer,
    LHSInitializer,
    Optimizer,
    Problem,
    Termination,
    max_fe,
    max_gen,
    maximize,
    minimize,
)
from saealib.algorithms import GenomeGA
from saealib.context import OptimizationState
from saealib.exceptions import ValidationError
from saealib.operators import (
    OrderCrossover,
    SequentialSelection,
    SwapMutation,
    TruncationSelection,
)
from saealib.result import Result
from saealib.space import PermutationSpace


class _Archive:
    def __init__(self, x, f, cv):
        self._arrays = {
            "x": np.asarray(x, dtype=float),
            "f": np.asarray(f, dtype=float),
            "cv": np.asarray(cv, dtype=float),
        }

    def get_array(self, name):
        return self._arrays[name]

    def __len__(self):
        return len(self._arrays["x"])


def _state(f, cv, direction, pareto=None):
    x = np.arange(len(f) * 2, dtype=float).reshape(len(f), 2)
    archive = _Archive(x, f, cv)
    pareto_archive = (
        _Archive(
            np.asarray(pareto[0], dtype=float),
            np.asarray(pareto[1], dtype=float),
            np.zeros(len(pareto[0])) if pareto else [],
        )
        if pareto
        else _Archive([], np.empty((0, len(direction))), [])
    )
    return SimpleNamespace(
        problem=SimpleNamespace(
            n_obj=len(direction),
            direction=np.asarray(direction, dtype=float),
            eps_cv=1e-6,
        ),
        archive=archive,
        pareto_archive=pareto_archive,
        population=object(),
        history=object(),
        fe=np.int64(7),
        gen=np.int64(3),
    )


def test_from_state_matches_minimize_and_maximize():
    kwargs = {
        "dim": 2,
        "lb": [-1, -1],
        "ub": [1, 1],
        "max_fe": 12,
        "pop_size": 4,
        "seed": 0,
        "verbose": False,
    }
    for optimize in (minimize, maximize):
        result = optimize(lambda x: np.sum(x**2), **kwargs)  # ty: ignore[invalid-argument-type]
        rebuilt = Result.from_state(result.ctx)
        np.testing.assert_array_equal(rebuilt.x, result.x)
        np.testing.assert_array_equal(rebuilt.f, result.f)
        assert rebuilt.fe == result.fe
        assert rebuilt.gen == result.gen


def test_single_objective_min_and_max():
    state = _state([[4], [1], [2]], [0, 0, 0], [-1])
    assert Result.from_state(state).f[0] == 1
    state = _state([[4], [1], [2]], [0, 0, 0], [1])
    assert Result.from_state(state).f[0] == 4


def test_multi_objective_uses_nonempty_pareto_archive():
    state = _state(
        [[1, 9], [9, 1]],
        [0, 0],
        [-1, -1],
        pareto=([[3, 3]], [[2, 2]]),
    )
    result = Result.from_state(state)
    np.testing.assert_array_equal(result.f, [[2, 2]])


def test_multi_objective_empty_pareto_archive_uses_first_front():
    state = _state([[1, 9], [9, 1], [8, 10]], [0, 0, 0], [-1, -1])
    result = Result.from_state(state)
    np.testing.assert_array_equal(result.f, [[1, 9], [9, 1]])


def test_feasibility_is_prioritized_and_impossible_uses_min_cv():
    state = _state([[100], [1], [2]], [0.1, 0, 0], [-1])
    assert Result.from_state(state).f[0] == 1
    state = _state([[100], [1], [2]], [0.5, 0.2, 0.3], [-1])
    assert Result.from_state(state).f[0] == 1


def test_empty_archive_raises_validation_error():
    with pytest.raises(ValidationError):
        Result.from_state(_state([], [], [-1]))


def test_optimizer_run_state_can_build_result():
    problem = Problem(
        func=lambda x: np.array([np.sum(np.asarray(x) ** 2)]),
        dim=2,
        n_obj=1,
        direction=np.array([-1.0]),
        lb=[-1, -1],
        ub=[1, 1],
    )
    state = (
        Optimizer(problem, seed=0)
        .set_initializer(LHSInitializer(8, 4))
        .set_termination(Termination(max_fe(12)))
        .run()
    )
    result = Result.from_state(state)
    assert result.ctx is state
    assert isinstance(state, OptimizationState)
    assert isinstance(result.x, np.ndarray)


def _permutation_state(n_obj: int) -> OptimizationState:
    space = PermutationSpace(8)
    problem = Problem(
        func=lambda x: np.asarray(
            [
                float(sum(i * v for i, v in enumerate(x))),
                float(sum((7 - i) * v for i, v in enumerate(x))),
            ][:n_obj]
        ),
        dim=space.dim,
        n_obj=n_obj,
        direction=np.full(n_obj, -1.0),
        space=space,
    )
    ga = GenomeGA(
        OrderCrossover(),
        SwapMutation(),
        SequentialSelection(),
        TruncationSelection(),
    )
    opt = (
        Optimizer(problem, seed=1)
        .set_algorithm(ga)
        .set_strategy(DirectStrategy())
        .set_initializer(GenomeInitializer(24, 24))
        .set_termination(Termination(max_gen(3)))
        .set_history(["summary", "front", "population"])
    )
    return opt.run()


def test_non_dense_single_objective_result_has_no_x():
    result = Result.from_state(_permutation_state(1))

    assert result.x is None
    assert isinstance(result.f, np.ndarray)


def test_non_dense_multi_objective_result_has_no_x():
    result = Result.from_state(_permutation_state(2))

    assert result.x is None
    assert result.f.ndim == 2


def test_iterate_result_copies_are_stable_after_continuing():
    problem = Problem(
        func=lambda x: np.array([np.sum(np.asarray(x) ** 2)]),
        dim=2,
        n_obj=1,
        direction=np.array([-1.0]),
        lb=[-1, -1],
        ub=[1, 1],
    )
    iterator = (
        Optimizer(problem, seed=0)
        .set_initializer(LHSInitializer(8, 4))
        .set_termination(Termination(max_gen(2)))
        .iterate()
    )
    try:
        state = next(iterator)
        result = Result.from_state(state)
        assert result.x is not None
        x_before = result.x.copy()
        f_before = result.f.copy()
        next(iterator)
    finally:
        iterator.close()

    np.testing.assert_array_equal(result.x, x_before)
    np.testing.assert_array_equal(result.f, f_before)


def test_direct_result_from_state_copies_arrays_and_normalizes_counters():
    state = _state([[1]], [0], [-1])
    result = Result.from_state(state)
    assert result.x is not None
    state.archive._arrays["x"][0, 0] = 99
    state.archive._arrays["f"][0, 0] = 99
    assert result.x[0] != 99
    assert result.f[0] != 99
    assert isinstance(result.fe, int)
    assert isinstance(result.gen, int)


def test_state_properties_delegate_by_identity():
    state = _state([[1]], [0], [-1])
    result = Result.from_state(state)
    assert result.problem is state.problem
    assert result.archive is state.archive
    assert result.pareto_archive is state.pareto_archive
    assert result.population is state.population
