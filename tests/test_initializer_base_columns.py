"""The initializer's base-column record in the optimization state."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from saealib import (
    GA,
    PSO,
    GenomeInitializer,
    LHSInitializer,
    Optimizer,
    Termination,
    max_gen,
)
from saealib.context import OptimizationState
from saealib.core.state import RUNTIME_BASE_COLUMNS
from saealib.identity import PopulationAttribute
from saealib.operators import (
    CrossoverBLXAlpha,
    MutationUniform,
    SequentialSelection,
    TruncationSelection,
)
from saealib.population import Archive, ParetoArchive, Population
from saealib.problem import Problem
from saealib.strategies import DirectStrategy


def _sphere(x: np.ndarray) -> np.ndarray:
    return np.array([float(np.sum(np.asarray(x) ** 2))])


def _problem() -> Problem:
    return Problem(
        func=_sphere,
        dim=2,
        n_obj=1,
        direction=np.array([-1.0]),
        lb=[-2.0, -2.0],
        ub=[2.0, 2.0],
    )


def _ga() -> GA:
    return GA(
        crossover=CrossoverBLXAlpha(prob=0.9, alpha=0.4),
        mutation=MutationUniform(prob_var=0.1),
        parent_selection=SequentialSelection(),
        survivor_selection=TruncationSelection(),
    )


def _initial_state(initializer, algorithm=None) -> OptimizationState:
    optimizer = (
        Optimizer(_problem())
        .set_initializer(initializer)
        .set_algorithm(algorithm or _ga())
        .set_strategy(DirectStrategy())
        .set_termination(Termination(max_gen(1)))
    )
    return next(optimizer.iterate())


def test_base_columns_record_the_columns_no_component_declared() -> None:
    state = _initial_state(LHSInitializer(4, 4))

    assert state.get_state(RUNTIME_BASE_COLUMNS) == [
        "cv",
        "f",
        "g",
        "id",
        "objective_source",
        "x",
    ]


def test_base_columns_omit_the_columns_the_algorithm_declares() -> None:
    state = _initial_state(LHSInitializer(4, 4), PSO())

    base = state.get_state(RUNTIME_BASE_COLUMNS)

    assert "velocity" not in base
    assert "pbest_x" not in base
    assert "velocity" in state.population.schema
    assert base == ["cv", "f", "g", "id", "objective_source", "x"]


def test_genome_initializer_base_columns_omit_the_design_variable_column() -> None:
    state = _initial_state(GenomeInitializer(4, 4))

    assert state.get_state(RUNTIME_BASE_COLUMNS) == [
        "cv",
        "f",
        "g",
        "id",
        "objective_source",
    ]


class _ExtraColumnInitializer(LHSInitializer):
    """A user-defined initializer that adds one column of its own."""

    def _create_attrs(self, problem, provider):
        return [
            *super()._create_attrs(problem, provider),
            PopulationAttribute("user_column", np.int64, (), default=3),
        ]


def test_a_user_defined_initializer_records_its_own_columns() -> None:
    state = _initial_state(_ExtraColumnInitializer(4, 4))

    assert "user_column" in state.get_state(RUNTIME_BASE_COLUMNS)


def test_base_columns_survive_a_portable_checkpoint(tmp_path: Path) -> None:
    state = _initial_state(LHSInitializer(4, 4))
    path = tmp_path / "state"
    state.save(path)

    loaded = OptimizationState.load(path, _problem())

    assert loaded.get_state(RUNTIME_BASE_COLUMNS) == [
        "cv",
        "f",
        "g",
        "id",
        "objective_source",
        "x",
    ]


def test_base_columns_survive_a_pickle_round_trip() -> None:
    state = _initial_state(GenomeInitializer(4, 4))

    restored = pickle.loads(pickle.dumps(state))

    assert restored.get_state(RUNTIME_BASE_COLUMNS) == [
        "cv",
        "f",
        "g",
        "id",
        "objective_source",
    ]


def test_a_state_without_the_record_loads_and_reports_no_base_columns(
    tmp_path: Path,
) -> None:
    problem = _problem()
    attrs = [
        PopulationAttribute("x", np.float64, (problem.dim,), default=np.nan),
        PopulationAttribute("f", np.float64, (problem.n_obj,), default=np.nan),
        PopulationAttribute("g", np.float64, (problem.n_constraints,), default=0.0),
        PopulationAttribute("cv", np.float64, (), default=0.0),
        PopulationAttribute("id", np.int64, (), default=-1),
    ]
    population = Population(attrs, 4)
    archive = Archive(attrs, 4)
    pareto_archive = ParetoArchive(attrs, 4, direction=problem.direction)
    state = OptimizationState(
        problem=problem,
        population=population,
        archive=archive,
        pareto_archive=pareto_archive,
        rng=np.random.default_rng(0),
    )
    path = tmp_path / "state"
    state.save(path)

    loaded = OptimizationState.load(path, problem)

    try:
        loaded.get_state(RUNTIME_BASE_COLUMNS)
    except KeyError:
        recorded = None
    else:  # pragma: no cover
        recorded = loaded.get_state(RUNTIME_BASE_COLUMNS)
    assert recorded is None
