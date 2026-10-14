"""Tests for RandomInitializer and SobolInitializer."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest

from saealib import (
    GA,
    CrossoverBLXAlpha,
    LHSInitializer,
    MutationUniform,
    SequentialSelection,
    Termination,
    TruncationSelection,
    max_gen,
)
from saealib.comparators import Comparator, SingleObjectiveComparator, SPEA2Comparator
from saealib.exceptions import ConfigurationError, ValidationError
from saealib.execution.evaluator import SerialEvaluator
from saealib.execution.initializer import (
    GenomeInitializer,
    Initializer,
    RandomInitializer,
    SobolInitializer,
)
from saealib.population import Population, PopulationAttribute
from saealib.problem import Problem
from saealib.strategies.base import OptimizationStrategy
from saealib.strategies.direct import DirectStrategy
from saealib.surrogate import SurrogateManager

DIM = 3
N_ARCHIVE = 8
N_POP = 4

LB = [-2.0] * DIM
UB = [3.0] * DIM


def _make_problem(comparator: Comparator | None = None) -> Problem:
    return Problem(
        func=lambda x: np.array([np.sum(x**2)]),
        dim=DIM,
        n_obj=1,
        direction=np.array([-1.0]),
        lb=LB,
        ub=UB,
        comparator=(
            comparator if comparator is not None else SingleObjectiveComparator()
        ),
    )


class _MockSurrogateManager:
    def fit(self, archive, ctx=None):
        pass


class _RequiredAttrsComparator(Comparator):
    def __init__(self, required_attrs: list[PopulationAttribute]):
        super().__init__(weights=np.empty(0), eps_cv=1e-6, eps_obj=1e-6)
        self.required_attrs = required_attrs

    def get_required_attrs(self, problem: Problem) -> list[PopulationAttribute]:
        return list(self.required_attrs)

    def sort_population(self, population: Population) -> np.ndarray:
        return np.arange(len(population))

    def compare_population(self, population: Population, idx_a: int, idx_b: int) -> int:
        return 0

    def compare(self, fa: np.ndarray, cv_a: float, fb: np.ndarray, cv_b: float) -> int:
        return 0


class _RequiredAttrsGA(GA):
    def __init__(self, required_attrs: list[PopulationAttribute]):
        super().__init__(
            crossover=CrossoverBLXAlpha(prob=0.9, alpha=0.4),
            mutation=MutationUniform(prob_var=0.1),
            parent_selection=SequentialSelection(),
            survivor_selection=TruncationSelection(),
        )
        self.required_attrs = required_attrs

    def get_required_attrs(self, problem: Problem) -> list[PopulationAttribute]:
        return list(self.required_attrs)


class _MockProvider:
    seed: int | None = None
    strategy: OptimizationStrategy = DirectStrategy()
    termination: Termination = Termination(max_gen(100_000))

    def __init__(self):
        from saealib.callback import CallbackManager

        self.algorithm = GA(
            crossover=CrossoverBLXAlpha(prob=0.9, alpha=0.4),
            mutation=MutationUniform(prob_var=0.1),
            parent_selection=SequentialSelection(),
            survivor_selection=TruncationSelection(),
        )
        self.surrogate_manager: SurrogateManager = _MockSurrogateManager()  # type: ignore
        self.evaluator = SerialEvaluator()
        self.cbmanager = CallbackManager()

    def dispatch(self, event):
        self.cbmanager.dispatch(event)


@pytest.fixture
def problem():
    return _make_problem()


@pytest.fixture
def provider():
    return _MockProvider()


def _make_required_attrs_provider(
    required_attrs: list[PopulationAttribute],
) -> _MockProvider:
    provider = _MockProvider()
    provider.algorithm = _RequiredAttrsGA(required_attrs)
    return provider


# ---------------------------------------------------------------------------
# Shared behaviour: all initializers must satisfy these properties
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "initializer",
    [
        RandomInitializer(N_ARCHIVE, N_POP, seed=0),
        SobolInitializer(N_ARCHIVE, N_POP, seed=0),
        LHSInitializer(N_ARCHIVE, N_POP, seed=0),
    ],
    ids=["random", "sobol", "lhs"],
)
class TestInitializerContract:
    def test_archive_size(self, initializer, problem, provider):
        ctx = initializer.initialize(provider, problem)
        assert len(ctx.archive) == N_ARCHIVE

    def test_population_size(self, initializer, problem, provider):
        ctx = initializer.initialize(provider, problem)
        assert len(ctx.population) == N_POP

    def test_fe_equals_n_archive(self, initializer, problem, provider):
        ctx = initializer.initialize(provider, problem)
        assert ctx.fe == N_ARCHIVE

    def test_archive_x_within_bounds(self, initializer, problem, provider):
        ctx = initializer.initialize(provider, problem)
        x = ctx.archive.get_array("x")
        lb = np.array(LB)
        ub = np.array(UB)
        assert np.all(x >= lb - 1e-9)
        assert np.all(x <= ub + 1e-9)

    def test_is_initializer_subclass(self, initializer, problem, provider):
        assert isinstance(initializer, Initializer)


@pytest.mark.parametrize(
    "initializer_class",
    [GenomeInitializer, RandomInitializer, SobolInitializer, LHSInitializer],
    ids=["genome", "random", "sobol", "lhs"],
)
class TestInitializerSizeValidation:
    def test_population_larger_than_archive_raises(self, initializer_class):
        with pytest.raises(ValidationError, match="n_init_population"):
            initializer_class(3, 5)

    def test_negative_sizes_raise(self, initializer_class):
        with pytest.raises(ValidationError, match="non-negative"):
            initializer_class(-1, 0)

    def test_equal_sizes_are_allowed(self, initializer_class):
        initializer_class(5, 5)


# ---------------------------------------------------------------------------
# Importability
# ---------------------------------------------------------------------------


def test_random_initializer_importable_from_top_level():
    import saealib

    assert hasattr(saealib, "RandomInitializer")
    assert saealib.RandomInitializer is RandomInitializer  # type: ignore[attr-defined]


def test_sobol_initializer_importable_from_top_level():
    import saealib

    assert hasattr(saealib, "SobolInitializer")
    assert saealib.SobolInitializer is SobolInitializer  # type: ignore[attr-defined]


def test_random_initializer_importable_from_execution():
    import saealib.execution.initializer as m

    assert m.RandomInitializer is RandomInitializer


def test_sobol_initializer_importable_from_execution():
    import saealib.execution.initializer as m

    assert m.SobolInitializer is SobolInitializer


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def test_random_initializer_same_seed_reproducible(problem, provider):
    ctx1 = RandomInitializer(N_ARCHIVE, N_POP, seed=42).initialize(
        cast(Any, provider), problem
    )
    provider2 = _MockProvider()
    ctx2 = RandomInitializer(N_ARCHIVE, N_POP, seed=42).initialize(
        cast(Any, provider2), problem
    )
    np.testing.assert_array_equal(
        ctx1.archive.get_array("x"), ctx2.archive.get_array("x")
    )


def test_sobol_initializer_same_seed_reproducible(problem, provider):
    ctx1 = SobolInitializer(N_ARCHIVE, N_POP, seed=42).initialize(
        cast(Any, provider), problem
    )
    provider2 = _MockProvider()
    ctx2 = SobolInitializer(N_ARCHIVE, N_POP, seed=42).initialize(
        cast(Any, provider2), problem
    )
    np.testing.assert_array_equal(
        ctx1.archive.get_array("x"), ctx2.archive.get_array("x")
    )


def test_comparator_required_attrs_are_added_to_population_and_archive():
    marker = PopulationAttribute("marker", np.float64, (), default=np.nan)
    problem = _make_problem(_RequiredAttrsComparator([marker]))
    provider = _make_required_attrs_provider([])

    ctx = LHSInitializer(N_ARCHIVE, N_POP, seed=0).initialize(
        cast(Any, provider), problem
    )

    assert "marker" in ctx.population.schema
    assert "marker" in ctx.archive.schema


def test_genome_initializer_merges_comparator_required_attrs():
    marker = PopulationAttribute("marker", np.float64, (), default=np.nan)
    problem = _make_problem(_RequiredAttrsComparator([marker]))
    provider = _make_required_attrs_provider([])

    attrs = GenomeInitializer(N_ARCHIVE, N_POP, seed=0)._create_attrs(
        problem, cast(Any, provider)
    )

    assert [attr.name for attr in attrs].count("marker") == 1


def test_genome_initializer_resolves_dense_numeric_view_on_pareto_archive(
    problem, provider
):
    ctx = GenomeInitializer(N_ARCHIVE, N_POP, seed=0).initialize(
        cast(Any, provider), problem
    )

    assert ctx.pareto_archive._dense_numeric_view is (
        problem.space.services.get("DenseNumericView")
    )


def test_compatible_algorithm_and_comparator_attrs_are_deduplicated():
    algorithm_attr = PopulationAttribute("marker", np.float64, (), default=np.nan)
    comparator_attr = PopulationAttribute("marker", float, (), default=np.nan)
    problem = _make_problem(_RequiredAttrsComparator([comparator_attr]))
    provider = _make_required_attrs_provider([algorithm_attr])

    ctx = LHSInitializer(N_ARCHIVE, N_POP, seed=0).initialize(
        cast(Any, provider), problem
    )

    assert sum(attr.name == "marker" for attr in ctx.population.attrs) == 1
    assert sum(attr.name == "marker" for attr in ctx.archive.attrs) == 1


def test_conflicting_algorithm_and_comparator_attrs_raise_configuration_error():
    algorithm_attr = PopulationAttribute("marker", np.float64, (), default=np.nan)
    comparator_attr = PopulationAttribute("marker", np.int64, (), default=0)
    problem = _make_problem(_RequiredAttrsComparator([comparator_attr]))
    provider = _make_required_attrs_provider([algorithm_attr])

    with pytest.raises(ConfigurationError, match="marker"):
        LHSInitializer(N_ARCHIVE, N_POP, seed=0).initialize(
            cast(Any, provider), problem
        )


def test_conflicting_default_raises_configuration_error():
    algorithm_attr = PopulationAttribute("marker", np.float64, (), default=np.nan)
    comparator_attr = PopulationAttribute("marker", np.float64, (), default=0.0)
    problem = _make_problem(_RequiredAttrsComparator([comparator_attr]))
    provider = _make_required_attrs_provider([algorithm_attr])

    with pytest.raises(ConfigurationError, match="marker"):
        LHSInitializer(N_ARCHIVE, N_POP, seed=0).initialize(
            cast(Any, provider), problem
        )


def test_builtin_comparator_keeps_the_default_initializer_schema():
    ctx = LHSInitializer(N_ARCHIVE, N_POP, seed=0).initialize(
        cast(Any, _MockProvider()), _make_problem()
    )
    expected = ["x", "f", "g", "cv", "id"]

    assert [attr.name for attr in ctx.population.attrs] == expected
    assert [attr.name for attr in ctx.archive.attrs] == expected


def test_lhs_initializer_populates_spea2_fitness_before_return():
    direction = np.array([-1.0, -1.0])
    problem = Problem(
        func=lambda x: np.array([np.sum(x**2), np.sum((x - 1.0) ** 2)]),
        dim=DIM,
        n_obj=2,
        direction=direction,
        lb=LB,
        ub=UB,
        comparator=SPEA2Comparator(direction=direction),
    )
    ctx = LHSInitializer(N_ARCHIVE, N_POP, seed=0).initialize(
        cast(Any, _MockProvider()), problem
    )

    assert np.all(np.isfinite(ctx.population.get_array("spea2_fitness")))


def test_spea2_fitness_is_nan_on_new_population_rows_until_prepared():
    comparator = SPEA2Comparator()
    problem = _make_problem(comparator)
    ctx = LHSInitializer(N_ARCHIVE, N_POP, seed=0).initialize(
        cast(Any, _MockProvider()), problem
    )

    ctx.population.append(x=np.zeros(DIM), f=np.array([1.0]), cv=0.0)
    assert np.isnan(ctx.population.get_array("spea2_fitness")[-1])

    comparator.prepare_population(ctx.population)
    assert np.all(np.isfinite(ctx.population.get_array("spea2_fitness")))
