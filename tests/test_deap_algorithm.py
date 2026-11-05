"""Tests for the optional DEAP generate/update adapter."""

from __future__ import annotations

import builtins
from typing import Any, cast

import numpy as np
import pytest

deap = pytest.importorskip("deap")

from _algorithm_boundary import ask as algorithm_ask  # noqa: E402
from _algorithm_boundary import tell as algorithm_tell  # noqa: E402
from deap import cma  # noqa: E402

from saealib.algorithms import DeapGenerateUpdateAlgorithm  # noqa: E402
from saealib.comparators import SingleObjectiveComparator  # noqa: E402
from saealib.context import OptimizationState  # noqa: E402
from saealib.exceptions import ConfigurationError  # noqa: E402
from saealib.population import (  # noqa: E402
    Archive,
    ParetoArchive,
    Population,
    PopulationAttribute,
)
from saealib.problem import Problem  # noqa: E402


class _Provider:
    def dispatch(self, event: object) -> None:
        del event


def _problem(dim: int = 3) -> Problem:
    return Problem(
        func=lambda x: np.array([np.sum(np.asarray(x) ** 2)]),
        dim=dim,
        n_obj=1,
        direction=np.array([-1.0]),
        lb=np.full(dim, -5.0).tolist(),
        ub=np.full(dim, 5.0).tolist(),
        comparator=SingleObjectiveComparator(direction=-1.0),
    )


def _state(
    algo: DeapGenerateUpdateAlgorithm, problem: Problem, seed: int
) -> OptimizationState:
    attrs = [
        PopulationAttribute("x", float, (problem.dim,), default=np.nan),
        PopulationAttribute("f", float, (1,), default=np.nan),
        PopulationAttribute("g", float, (0,), default=0.0),
        PopulationAttribute("cv", float, (), default=0.0),
        PopulationAttribute("objective_source", np.uint8, (), default=0),
        *algo.get_required_attrs(problem),
    ]
    population = Population(attrs, init_capacity=8)
    rng = np.random.default_rng(seed)
    x = rng.uniform(problem.lb, problem.ub, size=(8, problem.dim))
    population.extend({"x": x, "f": np.array([[problem.func(row)[0]] for row in x])})
    archive = Archive(attrs, init_capacity=8)
    pareto_archive = ParetoArchive(attrs, init_capacity=8, direction=problem.direction)
    return OptimizationState(
        problem=problem,
        population=population,
        archive=archive,
        pareto_archive=pareto_archive,
        rng=np.random.default_rng(seed),
    )


def _run(
    seed: int, rounds: int = 8
) -> tuple[np.ndarray, list[float], np.ndarray, np.ndarray]:
    problem = _problem()
    algo = DeapGenerateUpdateAlgorithm(
        cma.Strategy([3.0] * problem.dim, 1.0, lambda_=8)
    )
    state = _state(algo, problem, seed)
    provider = _Provider()
    best: list[float] = []
    last_x = np.empty((0, problem.dim))
    for _ in range(rounds):
        offspring = algorithm_ask(algo, state, provider)
        last_x = offspring.get_array("x").copy()
        offspring.update_array(
            "f", np.array([problem.func(row) for row in offspring.x])
        )
        best.append(float(np.min(offspring.get_array("f"))))
        algorithm_tell(algo, state, offspring, provider)
    centroid = np.asarray(cast(Any, algo.strategy).centroid, dtype=float).copy()
    return state.population.get_array("x").copy(), best, last_x, centroid


def test_strategy_optimizes_and_population_is_replaced_by_told_offspring():
    x, best, last_x, centroid = _run(7, rounds=10)
    assert np.isfinite(x).all()
    assert len(x) == 8
    assert min(best[2:]) < best[0]
    np.testing.assert_array_equal(x, last_x)
    assert np.linalg.norm(centroid) < np.linalg.norm(np.full(3, 3.0))


def test_same_seed_reproduces_and_different_seed_changes_result():
    first, _, _, _ = _run(11)
    second, _, _, _ = _run(11)
    third, _, _, _ = _run(12)
    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first, third)


def test_partial_tell_is_rejected_by_default():
    problem = _problem()
    algo = DeapGenerateUpdateAlgorithm(cma.Strategy([0.0] * problem.dim, 1.0))
    state = _state(algo, problem, 3)
    offspring = algorithm_ask(algo, state)
    offspring.update_array("f", np.array([problem.func(row) for row in offspring.x]))
    with pytest.raises(ConfigurationError, match="received"):
        algorithm_tell(algo, state, offspring.extract(np.arange(len(offspring) - 1)))


def test_tell_keeps_the_objective_source_of_told_offspring():
    problem = _problem()
    algo = DeapGenerateUpdateAlgorithm(
        cma.Strategy([0.0] * problem.dim, 1.0, lambda_=8)
    )
    state = _state(algo, problem, 5)
    offspring = algorithm_ask(algo, state)
    offspring.update_array("f", np.array([problem.func(row) for row in offspring.x]))
    sources = (np.arange(len(offspring)) % 2).astype(np.uint8)
    offspring.update_array("objective_source", sources)

    algorithm_tell(algo, state, offspring)

    np.testing.assert_array_equal(
        state.population.get_array("objective_source"), sources
    )


def test_missing_deap_has_install_command(monkeypatch: pytest.MonkeyPatch):
    real_import = builtins.__import__

    def blocked(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "deap":
            raise ImportError("blocked")
        del args, kwargs
        return real_import(name)

    monkeypatch.setattr(builtins, "__import__", blocked)
    with pytest.raises(ImportError, match=r"pip install.*deap"):
        DeapGenerateUpdateAlgorithm(cast(Any, object()))
