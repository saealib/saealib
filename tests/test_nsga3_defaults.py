import inspect
import math

import numpy as np

from saealib.comparators.comparators import (
    Comparator,
    NSGA3Comparator,
    RNSGA2Comparator,
)
from saealib.defaults.keys import INITIAL_ARCHIVE_SIZE, MAX_EVALUATIONS, POPULATION_SIZE
from saealib.execution.initializer import LHSInitializer
from saealib.optimizer import Optimizer
from saealib.problem import Problem


def _problem(n_obj: int = 3, dim: int = 3, n_ref: int = 13) -> Problem:
    reference_points = np.full((n_ref, n_obj), 1.0 / n_obj)
    return Problem(
        func=lambda x: np.sum(x) * np.ones(n_obj),
        dim=dim,
        n_obj=n_obj,
        direction=np.array([-1.0] * n_obj),
        lb=[-5.0] * dim,
        ub=[5.0] * dim,
        comparator=NSGA3Comparator(reference_points=reference_points),
    )


def _defaults(optimizer: Optimizer):
    resolution = optimizer._default_resolution
    assert resolution is not None
    return resolution


def test_population_and_archive_sizes_use_composition_formulas() -> None:
    optimizer = Optimizer(_problem(dim=3, n_ref=13))

    optimizer.resolve_defaults()

    assert isinstance(optimizer.initializer, LHSInitializer)
    initializer = optimizer.initializer
    assert isinstance(initializer, LHSInitializer)
    assert initializer.n_init_population == 4 * math.ceil(13 / 4)
    assert initializer.n_init_archive == max(5 * 3, 4 * math.ceil(13 / 4))
    assert _defaults(optimizer).get(POPULATION_SIZE) == 16
    assert _defaults(optimizer).get(INITIAL_ARCHIVE_SIZE) == 16
    assert _defaults(optimizer).resolved[POPULATION_SIZE].source == "composition rule"
    assert "reference points" in _defaults(optimizer).resolved[POPULATION_SIZE].reason


def test_population_and_archive_use_dimension_fallback_without_reference_points() -> (
    None
):
    problem = Problem(
        func=lambda x: np.sum(x),
        dim=4,
        n_obj=1,
        direction=np.array([-1.0]),
        lb=[-5.0] * 4,
        ub=[5.0] * 4,
    )
    optimizer = Optimizer(problem)

    optimizer.resolve_defaults()

    initializer = optimizer.initializer
    assert isinstance(initializer, LHSInitializer)
    assert initializer.n_init_population == 4 * 4
    assert initializer.n_init_archive == 5 * 4
    assert _defaults(optimizer).get(MAX_EVALUATIONS) == 200 * 4


def test_explicit_initializer_keeps_identity_and_sizes() -> None:
    explicit = LHSInitializer(n_init_archive=100, n_init_population=50, seed=42)
    optimizer = Optimizer(_problem()).set_initializer(explicit)

    optimizer.resolve_defaults()

    initializer = optimizer.initializer
    assert initializer is explicit
    assert initializer.n_init_population == 50
    assert initializer.n_init_archive == 100
    assert _defaults(optimizer).get(POPULATION_SIZE) == 16


def test_comparator_exposes_reference_points_without_resolving_population_size() -> (
    None
):
    comparator = _problem().comparator
    assert hasattr(comparator, "reference_points")
    assert not hasattr(comparator, "default_hints")

    module = inspect.getmodule(NSGA3Comparator)
    assert module is not None
    source = inspect.getsource(module)
    assert "saealib.defaults" not in source


def _rnsga2_problem(dim: int = 10, n_ref: int = 2) -> Problem:
    reference_points = np.full((n_ref, 2), 0.5)
    return Problem(
        func=lambda x: np.stack([np.sum(x**2), np.sum((x - 1) ** 2)]),
        dim=dim,
        n_obj=2,
        direction=np.array([-1.0, -1.0]),
        lb=[-1.0] * dim,
        ub=[1.0] * dim,
        comparator=RNSGA2Comparator(reference_points=reference_points),
    )


def test_reference_point_alignment_is_a_comparator_fact() -> None:
    assert Comparator.aligns_population_to_reference_points is False
    assert RNSGA2Comparator.aligns_population_to_reference_points is False
    assert NSGA3Comparator.aligns_population_to_reference_points is True


def test_rnsga2_keeps_dimension_based_population_and_archive() -> None:
    optimizer = Optimizer(_rnsga2_problem(dim=10))

    optimizer.resolve_defaults()

    initializer = optimizer.initializer
    assert isinstance(initializer, LHSInitializer)
    assert initializer.n_init_population == 4 * 10
    assert initializer.n_init_archive == 5 * 10
    assert _defaults(optimizer).get(POPULATION_SIZE) == 40
    assert _defaults(optimizer).get(INITIAL_ARCHIVE_SIZE) == 50
    assert _defaults(optimizer).resolved[POPULATION_SIZE].source == "composition rule"
    assert (
        "reference points" not in _defaults(optimizer).resolved[POPULATION_SIZE].reason
    )


def test_custom_comparator_can_request_reference_point_alignment() -> None:
    class AlignedRNSGA2Comparator(RNSGA2Comparator):
        aligns_population_to_reference_points = True

    problem = Problem(
        func=lambda x: np.stack([np.sum(x**2), np.sum((x - 1) ** 2)]),
        dim=4,
        n_obj=2,
        direction=np.array([-1.0, -1.0]),
        lb=[-1.0] * 4,
        ub=[1.0] * 4,
        comparator=AlignedRNSGA2Comparator(np.full((13, 2), 0.5)),
    )
    optimizer = Optimizer(problem)

    optimizer.resolve_defaults()

    initializer = optimizer.initializer
    assert isinstance(initializer, LHSInitializer)
    assert initializer.n_init_population == 4 * math.ceil(13 / 4)
    assert _defaults(optimizer).get(POPULATION_SIZE) == 16
    assert "reference points" in _defaults(optimizer).resolved[POPULATION_SIZE].reason
