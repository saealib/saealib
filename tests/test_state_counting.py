"""Behaviour of the evaluation count entry points on OptimizationState."""

from __future__ import annotations

import numpy as np

from saealib import context as context_module
from saealib.context import OptimizationState
from saealib.population import Archive, ParetoArchive, Population, PopulationAttribute
from saealib.problem import Problem

_ATTRS = [
    PopulationAttribute(name="x", dtype=np.float64, shape=(2,)),
    PopulationAttribute(name="f", dtype=np.float64, shape=(1,)),
    PopulationAttribute(name="g", dtype=np.float64, shape=(0,)),
    PopulationAttribute(name="cv", dtype=np.float64, shape=()),
]


def _problem() -> Problem:
    return Problem(
        func=lambda x: np.array([float(np.sum(np.asarray(x) ** 2))]),
        dim=2,
        n_obj=1,
        direction=np.array([-1.0]),
        lb=[-5.0, -5.0],
        ub=[5.0, 5.0],
    )


def _rows() -> dict[str, np.ndarray]:
    return {
        "x": np.zeros((2, 2), dtype=np.float64),
        "f": np.zeros((2, 1), dtype=np.float64),
        "g": np.zeros((2, 0), dtype=np.float64),
        "cv": np.zeros(2, dtype=np.float64),
    }


def _state() -> OptimizationState:
    population = Population(_ATTRS, 2)
    population.extend(_rows())
    archive = Archive(_ATTRS, 2)
    archive.extend(_rows())
    pareto_archive = ParetoArchive(_ATTRS, 2, direction=np.array([-1.0]))
    pareto_archive.extend(_rows())
    return OptimizationState(
        problem=_problem(),
        population=population,
        archive=archive,
        pareto_archive=pareto_archive,
        rng=np.random.default_rng(0),
    )


def _spy_on_counting(monkeypatch) -> list[tuple[OptimizationState, int, bool]]:
    calls: list[tuple[OptimizationState, int, bool]] = []
    original = context_module._add_evaluations

    def spy(state: OptimizationState, count: int, *, in_place: bool = False):
        calls.append((state, count, in_place))
        return original(state, count, in_place=in_place)

    monkeypatch.setattr(context_module, "_add_evaluations", spy)
    return calls


def test_count_fe_adds_the_manual_count_through_the_central_counting_function(
    monkeypatch,
):
    state = _state()
    calls = _spy_on_counting(monkeypatch)

    assert state.count_fe(3) is None

    assert len(calls) == 1
    counted_state, count, in_place = calls[0]
    assert counted_state is state
    assert count == 3
    assert in_place is True
    assert state.fe == 3


def test_count_fe_adds_to_the_count_the_state_already_has(monkeypatch):
    state = _state().replace(fe=4)
    calls = _spy_on_counting(monkeypatch)

    state.count_fe(2)

    assert len(calls) == 1
    assert state.fe == 6
