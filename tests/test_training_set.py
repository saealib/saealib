"""Tests for surrogate training set strategy objects."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pytest

from saealib.comparators import SingleObjectiveComparator
from saealib.context import OptimizationState
from saealib.exceptions import ValidationError
from saealib.population import Archive, ParetoArchive, Population, PopulationAttribute
from saealib.population.genome import PermutationBatch, VariableLengthBatch
from saealib.problem import Problem
from saealib.space import PermutationSpace, SequenceSpace
from saealib.space.space import ServiceRegistry
from saealib.surrogate.training_set import (
    ArchiveObjectiveSet,
    ConstraintObjectiveSet,
    FeasibilityClassificationSet,
    KNNConstraintObjectiveSet,
    KNNObjectiveSet,
    LevelBasedSet,
    PairwiseComparisonSet,
    ReferencePointComparisonSet,
    TopKBipartitionSet,
    TrainingSet,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

DIM = 2
N_OBJ = 1
N_CONSTRAINTS = 2

_ATTRS = [
    PopulationAttribute(name="x", dtype=np.float64, shape=(DIM,)),
    PopulationAttribute(name="f", dtype=np.float64, shape=(N_OBJ,)),
    PopulationAttribute(name="g", dtype=np.float64, shape=(0,)),
    PopulationAttribute(name="cv", dtype=np.float64, shape=()),
]

_ATTRS_G = [
    PopulationAttribute(name="x", dtype=np.float64, shape=(DIM,)),
    PopulationAttribute(name="f", dtype=np.float64, shape=(N_OBJ,)),
    PopulationAttribute(name="g", dtype=np.float64, shape=(N_CONSTRAINTS,)),
    PopulationAttribute(name="cv", dtype=np.float64, shape=()),
]


def _make_problem(eps_cv: float = 1e-6) -> Problem:
    return Problem(
        func=lambda x: np.array([np.sum(x**2)]),
        dim=DIM,
        n_obj=N_OBJ,
        direction=np.array([-1.0]),
        lb=[-5.0] * DIM,
        ub=[5.0] * DIM,
        eps_cv=eps_cv,
        comparator=SingleObjectiveComparator(),
    )


def _make_archive(n: int = 10) -> Archive:
    arc = Archive(_ATTRS, init_capacity=n + 5)
    rng = np.random.default_rng(42)
    for _ in range(n):
        x = rng.uniform(-2.0, 2.0, size=DIM)
        f = np.array([np.sum(x**2)])
        arc.add(x=x, f=f, cv=0.0)
    return arc


def _make_archive_with_g(n: int = 10) -> Archive:
    """Archive with 2 constraint columns filled with known values."""
    arc = Archive(_ATTRS_G, init_capacity=n + 5)
    rng = np.random.default_rng(7)
    for i in range(n):
        x = rng.uniform(-2.0, 2.0, size=DIM)
        f = np.array([np.sum(x**2)])
        g = np.array([float(i), float(-i)])  # known values for assertion
        arc.add(x=x, f=f, g=g, cv=0.0)
    return arc


def _make_archive_with_cv(cvs: list[float]) -> Archive:
    """Archive where each point has a prescribed cv value."""
    arc = Archive(_ATTRS, init_capacity=len(cvs) + 5)
    rng = np.random.default_rng(0)
    for cv in cvs:
        x = rng.uniform(-2.0, 2.0, size=DIM)
        f = np.array([np.sum(x**2)])
        arc.add(x=x, f=f, cv=float(cv))
    return arc


def _make_population_with_cv(cvs: list[float]) -> Population:
    """Population where each individual has a prescribed cv value."""
    pop = Population(_ATTRS, init_capacity=len(cvs) + 5)
    rng = np.random.default_rng(1)
    xs = rng.uniform(-2.0, 2.0, size=(len(cvs), DIM))
    fs = np.array([[np.sum(x**2)] for x in xs])
    cv_arr = np.array(cvs, dtype=float)
    pop.extend({"x": xs, "f": fs, "cv": cv_arr})
    return pop


def _make_ctx(
    archive: Archive | None = None,
    population: Population | None = None,
    eps_cv: float = 1e-6,
) -> OptimizationState:
    problem = _make_problem(eps_cv=eps_cv)
    arc = archive if archive is not None else _make_archive()
    pop = population if population is not None else _make_population_with_cv([0.0] * 5)
    pareto_arc = ParetoArchive(_ATTRS, init_capacity=20, direction=np.array([-1.0]))
    return OptimizationState(
        problem=problem,
        population=pop,
        archive=arc,
        pareto_archive=pareto_arc,
        rng=np.random.default_rng(0),
        fe=10,
        gen=1,
    )


# ===========================================================================
# TrainingSet ABC
# ===========================================================================
class TestTrainingSetABC:
    def test_cannot_instantiate_abstract(self) -> None:
        with pytest.raises(TypeError):
            TrainingSet()  # type: ignore[abstract]  # intentional: testing abstract instantiation raises TypeError


# ===========================================================================
# ArchiveObjectiveSet
# ===========================================================================
class TestArchiveObjectiveSet:
    def test_returns_all_archive_points(self) -> None:
        arc = _make_archive(10)
        ts = ArchiveObjectiveSet()
        data = ts.build(arc, None, ctx=None)
        assert data.train_x.shape == (10, DIM)
        assert data.train_y.shape == (10, N_OBJ)

    def test_values_match_archive(self) -> None:
        arc = _make_archive(8)
        ts = ArchiveObjectiveSet()
        data = ts.build(arc, None, ctx=None)
        np.testing.assert_array_equal(data.train_x, arc.x)
        np.testing.assert_array_equal(data.train_y, arc.f)

    def test_candidate_x_ignored(self) -> None:
        arc = _make_archive(5)
        ts = ArchiveObjectiveSet()
        cand = np.zeros(DIM)
        data = ts.build(arc, None, candidate_x=cand, ctx=None)
        assert data.train_x.shape == (5, DIM)


def _make_attrs(dim: int) -> list[PopulationAttribute]:
    return [
        PopulationAttribute(name="x", dtype=np.float64, shape=(dim,)),
        PopulationAttribute(name="f", dtype=np.float64, shape=(N_OBJ,)),
        PopulationAttribute(name="g", dtype=np.float64, shape=(0,)),
        PopulationAttribute(name="cv", dtype=np.float64, shape=()),
    ]


def _make_space_problem(space: Any) -> Problem:
    return Problem(
        func=lambda x: np.array([0.0]),
        dim=None,
        n_obj=N_OBJ,
        direction=np.array([-1.0]),
        space=space,
        comparator=SingleObjectiveComparator(),
    )


def _make_genome_archive(space: Any, genomes: Any, dim: int) -> Archive:
    """Archive backed by a real genome batch for a non-vector space."""
    arc = Archive(_make_attrs(dim), init_capacity=4, genomes=genomes, space=space)
    n = len(genomes)
    arc.update_rows(
        np.arange(n),
        {"f": np.arange(1, n + 1, dtype=float).reshape(-1, 1), "cv": np.zeros(n)},
    )
    return arc


def _make_space_ctx(problem: Problem, archive: Archive, dim: int) -> OptimizationState:
    attrs = _make_attrs(dim)
    return OptimizationState(
        problem=problem,
        population=Population(attrs, init_capacity=4),
        archive=archive,
        pareto_archive=ParetoArchive(
            attrs, init_capacity=8, direction=np.array([-1.0])
        ),
        rng=np.random.default_rng(0),
        fe=2,
        gen=1,
    )


class _DoublingEncoder:
    def encode(self, genomes: Any) -> np.ndarray:
        return 2.0 * np.asarray(genomes.array, dtype=np.float64)


class _FailingEncoder:
    def encode(self, genomes: Any) -> np.ndarray:
        raise ValueError("encoder exploded")


class _UnsupportedSpace:
    """Search space without a supported default encoder."""

    def __init__(self) -> None:
        self.services = ServiceRegistry()
        self.dim = DIM


class TestArchiveObjectiveSetFeatureEncoding:
    def test_standalone_state_uses_default_encoder_and_warns(self) -> None:
        arc = _make_archive(4)
        ctx = _make_ctx(archive=arc)
        space = ctx.problem.space
        assert space.services.get("FeatureEncoder") is None

        with pytest.warns(
            UserWarning, match="encode_features used the default FeatureEncoder"
        ) as emitted:
            data = ArchiveObjectiveSet().build(arc, None, ctx=ctx)

        assert len(emitted) == 1
        assert emitted[0].filename == __file__
        np.testing.assert_array_equal(data.train_x, arc.x)
        np.testing.assert_array_equal(data.train_y, arc.f)
        assert space.services.get("FeatureEncoder") is None

    def test_permutation_archive_uses_default_encoder_and_warns(self) -> None:
        space = PermutationSpace(length=3)
        genomes = PermutationBatch([[0, 2, 1], [2, 1, 0]], length=3)
        arc = _make_genome_archive(space, genomes, 3)
        ctx = _make_space_ctx(_make_space_problem(space), arc, 3)
        assert space.services.get("FeatureEncoder") is None

        with pytest.warns(
            UserWarning, match="encode_features used the default FeatureEncoder"
        ) as emitted:
            data = ArchiveObjectiveSet().build(arc, None, ctx=ctx)

        assert len(emitted) == 1
        assert emitted[0].filename == __file__
        np.testing.assert_array_equal(data.train_x, genomes.array)
        np.testing.assert_array_equal(data.train_y, arc.f)
        assert space.services.get("FeatureEncoder") is None

    def test_sequence_archive_uses_default_encoder_and_warns(self) -> None:
        space = SequenceSpace(["a", "b"], max_length=3)
        genomes = VariableLengthBatch([["a", "b"], ["b"]])
        arc = _make_genome_archive(space, genomes, 4)
        ctx = _make_space_ctx(_make_space_problem(space), arc, 4)
        assert space.services.get("FeatureEncoder") is None

        with pytest.warns(
            UserWarning, match="encode_features used the default FeatureEncoder"
        ) as emitted:
            data = ArchiveObjectiveSet().build(arc, None, ctx=ctx)

        assert len(emitted) == 1
        assert emitted[0].filename == __file__
        np.testing.assert_array_equal(data.train_x, [[2, 1, 2, 0], [1, 2, 0, 0]])
        np.testing.assert_array_equal(data.train_y, arc.f)
        assert space.services.get("FeatureEncoder") is None

    def test_registered_encoder_is_used_without_default_warning(self) -> None:
        arc = _make_archive(4)
        ctx = _make_ctx(archive=arc)
        encoder = _DoublingEncoder()
        ctx.problem.space.services.register("FeatureEncoder", encoder)

        with warnings.catch_warnings(record=True) as emitted:
            warnings.simplefilter("always")
            data = ArchiveObjectiveSet().build(arc, None, ctx=ctx)

        assert emitted == []
        np.testing.assert_array_equal(data.train_x, 2.0 * arc.x)
        assert ctx.problem.space.services.get("FeatureEncoder") is encoder

    def test_prepared_encoder_takes_priority_without_warning(self) -> None:
        arc = _make_archive(4)
        ctx = _make_ctx(archive=arc)
        registered = _FailingEncoder()
        prepared = _DoublingEncoder()
        ctx.problem.space.services.register("FeatureEncoder", registered)
        ctx.bind_compiled_services({"FeatureEncoder": prepared})

        with warnings.catch_warnings(record=True) as emitted:
            warnings.simplefilter("always")
            data = ArchiveObjectiveSet().build(arc, None, ctx=ctx)

        assert emitted == []
        np.testing.assert_array_equal(data.train_x, 2.0 * arc.x)
        assert ctx.compiled_service("FeatureEncoder") is prepared
        assert ctx.problem.space.services.get("FeatureEncoder") is registered

    def test_ctx_none_keeps_archive_columns_without_warning(self) -> None:
        arc = _make_archive(5)

        with warnings.catch_warnings(record=True) as emitted:
            warnings.simplefilter("always")
            data = ArchiveObjectiveSet().build(arc, None, ctx=None)

        assert emitted == []
        np.testing.assert_array_equal(data.train_x, arc.x)

    def test_unsupported_space_keeps_encoder_error_without_warning(self) -> None:
        space = _UnsupportedSpace()
        arc = _make_archive(3)
        ctx = _make_space_ctx(_make_space_problem(space), arc, DIM)

        with warnings.catch_warnings(record=True) as emitted:
            warnings.simplefilter("always")
            with pytest.raises(
                ValidationError, match="require the FeatureEncoder service"
            ):
                ArchiveObjectiveSet().build(arc, None, ctx=ctx)

        assert emitted == []
        assert space.services.get("FeatureEncoder") is None

    def test_incompatible_registered_encoder_is_not_replaced_by_default(self) -> None:
        arc = _make_archive(3)
        ctx = _make_ctx(archive=arc)
        encoder = object()
        ctx.problem.space.services.register("FeatureEncoder", encoder)

        with warnings.catch_warnings(record=True) as emitted:
            warnings.simplefilter("always")
            with pytest.raises(
                ValidationError, match="require the FeatureEncoder service"
            ):
                ArchiveObjectiveSet().build(arc, None, ctx=ctx)

        assert emitted == []
        assert ctx.problem.space.services.get("FeatureEncoder") is encoder

    def test_registered_encoder_failure_propagates(self) -> None:
        arc = _make_archive(3)
        ctx = _make_ctx(archive=arc)
        ctx.problem.space.services.register("FeatureEncoder", _FailingEncoder())

        with pytest.raises(ValueError, match="encoder exploded"):
            ArchiveObjectiveSet().build(arc, None, ctx=ctx)

    def test_prepared_encoder_failure_propagates(self) -> None:
        arc = _make_archive(3)
        ctx = _make_ctx(archive=arc)
        ctx.bind_compiled_services({"FeatureEncoder": _FailingEncoder()})

        with pytest.raises(ValueError, match="encoder exploded"):
            ArchiveObjectiveSet().build(arc, None, ctx=ctx)


# ===========================================================================
# KNNObjectiveSet
# ===========================================================================
class TestKNNObjectiveSet:
    def test_returns_k_neighbours(self) -> None:
        arc = _make_archive(20)
        ts = KNNObjectiveSet(n_neighbors=5)
        cand = np.zeros(DIM)
        data = ts.build(arc, None, candidate_x=cand, ctx=None)
        assert data.train_x.shape == (5, DIM)
        assert data.train_y.shape == (5, N_OBJ)

    def test_requires_candidate_x(self) -> None:
        arc = _make_archive(10)
        ts = KNNObjectiveSet(n_neighbors=5)
        with pytest.raises(ValueError, match="requires candidate_x"):
            ts.build(arc, None, candidate_x=None, ctx=None)

    def test_default_n_neighbors(self) -> None:
        ts = KNNObjectiveSet()
        assert ts.n_neighbors == 50

    def test_clamped_to_archive_size(self) -> None:
        arc = _make_archive(8)
        ts = KNNObjectiveSet(n_neighbors=50)
        cand = np.zeros(DIM)
        data = ts.build(arc, None, candidate_x=cand, ctx=None)
        assert data.train_x.shape[0] <= 8

    def test_dense_rows_are_read_only_and_match_archive(self) -> None:
        arc = _make_archive(8)
        data = KNNObjectiveSet(n_neighbors=4).build(
            arc, None, candidate_x=np.zeros(DIM), ctx=None
        )
        assert not data.train_x.flags.writeable
        assert not data.train_y.flags.writeable
        np.testing.assert_array_equal(
            data.train_x, arc.x[arc.get_knn(np.zeros(DIM), 4)[0]]
        )
        np.testing.assert_array_equal(
            data.train_y, arc.f[arc.get_knn(np.zeros(DIM), 4)[0]]
        )

    def test_custom_archive_subclass_uses_property_fallback(self) -> None:
        class CustomArchive(Archive):
            @property
            def x(self):
                return super().x

            @property
            def f(self):
                return super().f

        arc = CustomArchive(_ATTRS, init_capacity=8)
        for i in range(4):
            arc.add(
                x=np.full(DIM, i, dtype=float), f=np.array([i], dtype=float), cv=0.0
            )
        data = KNNObjectiveSet(n_neighbors=2).build(
            arc, None, candidate_x=np.zeros(DIM), ctx=None
        )
        assert data.train_x.flags.writeable
        np.testing.assert_array_equal(
            data.train_x, arc.x[arc.get_knn(np.zeros(DIM), 2)[0]]
        )


# ===========================================================================
# ConstraintObjectiveSet
# ===========================================================================
class TestConstraintObjectiveSet:
    def test_returns_all_archive_points(self) -> None:
        arc = _make_archive_with_g(8)
        ts = ConstraintObjectiveSet()
        data = ts.build(arc, None, ctx=None)
        assert data.train_x.shape == (8, DIM)
        assert data.train_y.shape == (8, N_CONSTRAINTS)
        np.testing.assert_array_equal(data.train_y, arc.g)

    def test_train_y_is_g_not_f(self) -> None:
        arc = _make_archive_with_g(5)
        ts = ConstraintObjectiveSet()
        data = ts.build(arc, None, ctx=None)
        # train_y must differ from f (different columns)
        assert data.train_y.shape[1] == N_CONSTRAINTS
        assert data.train_y.shape[1] != arc.f.shape[1] or not np.allclose(
            data.train_y, arc.f
        )

    def test_raises_on_zero_constraints(self) -> None:
        arc = _make_archive(5)  # uses _ATTRS with shape=(0,) for g
        ts = ConstraintObjectiveSet()
        with pytest.raises(ValueError, match="0 columns"):
            ts.build(arc, None, ctx=None)

    def test_candidate_x_is_ignored(self) -> None:
        arc = _make_archive_with_g(6)
        ts = ConstraintObjectiveSet()
        candidate = np.zeros(DIM)
        data = ts.build(arc, None, candidate_x=candidate, ctx=None)
        assert data.train_x.shape[0] == 6


# ===========================================================================
# KNNConstraintObjectiveSet
# ===========================================================================
class TestKNNConstraintObjectiveSet:
    def test_returns_k_neighbours(self) -> None:
        arc = _make_archive_with_g(10)
        ts = KNNConstraintObjectiveSet(n_neighbors=5)
        candidate = np.zeros(DIM)
        data = ts.build(arc, None, candidate_x=candidate, ctx=None)
        assert data.train_x.shape == (5, DIM)
        assert data.train_y.shape == (5, N_CONSTRAINTS)

    def test_requires_candidate_x(self) -> None:
        arc = _make_archive_with_g(5)
        ts = KNNConstraintObjectiveSet(n_neighbors=3)
        with pytest.raises(ValueError, match="candidate_x"):
            ts.build(arc, None, candidate_x=None, ctx=None)

    def test_raises_on_zero_constraints(self) -> None:
        arc = _make_archive(5)
        ts = KNNConstraintObjectiveSet(n_neighbors=3)
        with pytest.raises(ValueError, match="0 columns"):
            ts.build(arc, None, candidate_x=np.zeros(DIM), ctx=None)

    def test_clamped_to_archive_size(self) -> None:
        arc = _make_archive_with_g(4)
        ts = KNNConstraintObjectiveSet(n_neighbors=50)
        data = ts.build(arc, None, candidate_x=np.zeros(DIM), ctx=None)
        assert data.train_x.shape[0] == 4


# ===========================================================================
# FeasibilityClassificationSet
# ===========================================================================
class TestFeasibilityClassificationSet:
    def test_all_feasible(self) -> None:
        arc = _make_archive_with_cv([0.0, 0.0, 0.0, 0.0])
        ts = FeasibilityClassificationSet(source="archive")
        data = ts.build(arc, None, ctx=None)
        np.testing.assert_array_equal(data.train_y, np.ones(4))

    def test_all_infeasible(self) -> None:
        arc = _make_archive_with_cv([1.0, 2.0, 0.5])
        ts = FeasibilityClassificationSet(source="archive")
        data = ts.build(arc, None, ctx=None)
        np.testing.assert_array_equal(data.train_y, np.zeros(3))

    def test_mixed_labels(self) -> None:
        arc = _make_archive_with_cv([0.0, 1.0, 0.0, 2.0])
        ts = FeasibilityClassificationSet(source="archive")
        data = ts.build(arc, None, ctx=None)
        np.testing.assert_array_equal(data.train_y, [1.0, 0.0, 1.0, 0.0])

    def test_eps_from_ctx(self) -> None:
        arc = _make_archive_with_cv([0.1, 0.2, 0.3])
        ctx = _make_ctx(archive=arc, eps_cv=0.15)
        ts = FeasibilityClassificationSet(source="archive")
        data = ts.build(arc, None, ctx=ctx)
        # cv=0.1 <= 0.15 → 1, cv=0.2 > 0.15 → 0, cv=0.3 > 0.15 → 0
        np.testing.assert_array_equal(data.train_y, [1.0, 0.0, 0.0])

    def test_eps_default_when_ctx_none(self) -> None:
        arc = _make_archive_with_cv([0.0, 1e-7, 1e-5])
        ts = FeasibilityClassificationSet(source="archive")
        data = ts.build(arc, None, ctx=None)
        # 0.0 <= 1e-6 → 1, 1e-7 <= 1e-6 → 1, 1e-5 > 1e-6 → 0
        np.testing.assert_array_equal(data.train_y, [1.0, 1.0, 0.0])

    def test_source_population(self) -> None:
        pop = _make_population_with_cv([0.0, 0.5, 0.0])
        arc = _make_archive()
        ts = FeasibilityClassificationSet(source="population")
        data = ts.build(arc, pop, ctx=None)
        np.testing.assert_array_equal(data.train_y, [1.0, 0.0, 1.0])

    def test_source_population_none_raises(self) -> None:
        arc = _make_archive()
        ts = FeasibilityClassificationSet(source="population")
        with pytest.raises(ValueError, match="population"):
            ts.build(arc, None, ctx=None)

    def test_train_x_shape(self) -> None:
        arc = _make_archive_with_cv([0.0, 1.0, 0.0])
        ts = FeasibilityClassificationSet(source="archive")
        data = ts.build(arc, None, ctx=None)
        assert data.train_x.shape == (3, DIM)
        assert data.train_y.shape == (3,)

    def test_labels_are_float(self) -> None:
        arc = _make_archive_with_cv([0.0, 1.0])
        ts = FeasibilityClassificationSet(source="archive")
        data = ts.build(arc, None, ctx=None)
        assert data.train_y.dtype == float


# ===========================================================================
# TopKBipartitionSet
# ===========================================================================


def _make_archive_with_f(
    f_values: list[float], cvs: list[float] | None = None
) -> Archive:
    """Archive where each point has a prescribed scalar objective."""
    if cvs is None:
        cvs = [0.0] * len(f_values)
    arc = Archive(_ATTRS, init_capacity=len(f_values) + 5)
    rng = np.random.default_rng(2)
    for fv, cv in zip(f_values, cvs):
        x = rng.uniform(-2.0, 2.0, size=DIM)
        arc.add(x=x, f=np.array([fv]), cv=float(cv))
    return arc


class TestTopKBipartitionSet:
    def test_label_count(self) -> None:
        arc = _make_archive_with_f([1.0, 2.0, 3.0, 4.0])
        ctx = _make_ctx(archive=arc)
        ts = TopKBipartitionSet(source="archive", top_ratio=0.5)
        data = ts.build(arc, None, ctx=ctx)
        assert data.train_y.shape == (4,)
        assert int(data.train_y.sum()) == 2  # top 2 = label 1

    def test_best_get_label_one(self) -> None:
        # f=1.0 is best (minimization), f=4.0 is worst
        arc = _make_archive_with_f([4.0, 1.0, 3.0, 2.0])
        ctx = _make_ctx(archive=arc)
        ts = TopKBipartitionSet(source="archive", top_ratio=0.5)
        data = ts.build(arc, None, ctx=ctx)
        # sorted order: f=1.0, f=2.0 → label 1; f=3.0, f=4.0 → label 0
        np.testing.assert_array_equal(data.train_y[:2], [1.0, 1.0])
        np.testing.assert_array_equal(data.train_y[2:], [0.0, 0.0])

    def test_at_least_one_label_one(self) -> None:
        arc = _make_archive_with_f([1.0, 2.0, 3.0])
        ctx = _make_ctx(archive=arc)
        ts = TopKBipartitionSet(source="archive", top_ratio=0.0)
        data = ts.build(arc, None, ctx=ctx)
        assert data.train_y.sum() >= 1.0

    def test_ctx_none_raises(self) -> None:
        arc = _make_archive_with_f([1.0, 2.0])
        ts = TopKBipartitionSet(source="archive")
        with pytest.raises(ValueError, match="ctx"):
            ts.build(arc, None, ctx=None)

    def test_source_population_none_raises(self) -> None:
        arc = _make_archive_with_f([1.0, 2.0])
        ctx = _make_ctx(archive=arc)
        ts = TopKBipartitionSet(source="population")
        with pytest.raises(ValueError, match="population"):
            ts.build(arc, None, ctx=ctx)

    def test_source_population(self) -> None:
        pop = _make_population_with_cv([0.0, 0.0, 0.0, 0.0])
        arc = _make_archive()
        ctx = _make_ctx(archive=arc, population=pop)
        ts = TopKBipartitionSet(source="population", top_ratio=0.5)
        data = ts.build(arc, pop, ctx=ctx)
        assert data.train_x.shape == (4, DIM)
        assert data.train_y.shape == (4,)

    def test_train_x_shape(self) -> None:
        arc = _make_archive_with_f([1.0, 2.0, 3.0])
        ctx = _make_ctx(archive=arc)
        ts = TopKBipartitionSet(source="archive")
        data = ts.build(arc, None, ctx=ctx)
        assert data.train_x.shape == (3, DIM)

    def test_labels_are_float(self) -> None:
        arc = _make_archive_with_f([1.0, 2.0])
        ctx = _make_ctx(archive=arc)
        ts = TopKBipartitionSet(source="archive")
        data = ts.build(arc, None, ctx=ctx)
        assert data.train_y.dtype == float


# ===========================================================================
# LevelBasedSet
# ===========================================================================


class TestLevelBasedSet:
    def test_three_levels_equal_split(self) -> None:
        arc = _make_archive_with_f([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        ctx = _make_ctx(archive=arc)
        ts = LevelBasedSet(source="archive", n_levels=3)
        data = ts.build(arc, None, ctx=ctx)
        np.testing.assert_array_equal(data.train_y, [2.0, 2.0, 1.0, 1.0, 0.0, 0.0])

    def test_remainder_goes_to_last_level(self) -> None:
        arc = _make_archive_with_f([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        ctx = _make_ctx(archive=arc)
        ts = LevelBasedSet(source="archive", n_levels=3)
        data = ts.build(arc, None, ctx=ctx)
        np.testing.assert_array_equal(data.train_y, [2.0, 2.0, 1.0, 1.0, 0.0, 0.0, 0.0])

    def test_best_individuals_get_highest_label(self) -> None:
        arc = _make_archive_with_f([5.0, 1.0, 3.0, 2.0])
        ctx = _make_ctx(archive=arc)
        ts = LevelBasedSet(source="archive", n_levels=2)
        data = ts.build(arc, None, ctx=ctx)
        # sorted order: f=1.0, f=2.0 → label 1; f=3.0, f=5.0 → label 0
        np.testing.assert_array_equal(data.train_y[:2], [1.0, 1.0])
        np.testing.assert_array_equal(data.train_y[2:], [0.0, 0.0])

    def test_default_n_levels(self) -> None:
        ts = LevelBasedSet()
        assert ts.n_levels == 5

    def test_ctx_none_raises(self) -> None:
        arc = _make_archive_with_f([1.0, 2.0])
        ts = LevelBasedSet(source="archive")
        with pytest.raises(ValueError, match="ctx"):
            ts.build(arc, None, ctx=None)

    def test_source_population_none_raises(self) -> None:
        arc = _make_archive_with_f([1.0, 2.0])
        ctx = _make_ctx(archive=arc)
        ts = LevelBasedSet(source="population")
        with pytest.raises(ValueError, match="population"):
            ts.build(arc, None, ctx=ctx)

    def test_train_x_shape(self) -> None:
        arc = _make_archive_with_f([1.0, 2.0, 3.0, 4.0, 5.0])
        ctx = _make_ctx(archive=arc)
        ts = LevelBasedSet(source="archive", n_levels=5)
        data = ts.build(arc, None, ctx=ctx)
        assert data.train_x.shape == (5, DIM)
        assert data.train_y.shape == (5,)

    def test_labels_are_float(self) -> None:
        arc = _make_archive_with_f([1.0, 2.0, 3.0])
        ctx = _make_ctx(archive=arc)
        ts = LevelBasedSet(source="archive", n_levels=3)
        data = ts.build(arc, None, ctx=ctx)
        assert data.train_y.dtype == float

    def test_score_increases_with_quality(self) -> None:
        from saealib.acquisition.mean import MeanPrediction
        from saealib.surrogate.prediction import SurrogatePrediction

        arc = _make_archive_with_f([1.0, 2.0, 3.0, 4.0])
        ctx = _make_ctx(archive=arc)
        ts = LevelBasedSet(source="archive", n_levels=2)
        data = ts.build(arc, None, ctx=ctx)
        pred = SurrogatePrediction.objective(value=data.train_y.reshape(-1, 1))
        acq = MeanPrediction(direction=np.array([1.0]))
        scores = acq.score(pred)
        assert scores[:2].min() > scores[2:].max()


# ===========================================================================
# PairwiseComparisonSet
# ===========================================================================


class TestPairwiseComparisonSet:
    def test_all_pairs_by_default(self) -> None:
        arc = _make_archive_with_f([1.0, 2.0, 3.0, 4.0])
        ctx = _make_ctx(archive=arc)
        ts = PairwiseComparisonSet(source="archive")
        data = ts.build(arc, None, ctx=ctx)
        n = 4
        assert data.train_x.shape == (n * (n - 1) // 2, DIM * 2)
        assert data.train_y.shape == (n * (n - 1) // 2,)

    def test_n_pairs_limits_count(self) -> None:
        arc = _make_archive_with_f([1.0, 2.0, 3.0, 4.0, 5.0])
        ctx = _make_ctx(archive=arc)
        ts = PairwiseComparisonSet(source="archive", n_pairs=3)
        data = ts.build(arc, None, ctx=ctx)
        assert data.train_x.shape == (3, DIM * 2)
        assert data.train_y.shape == (3,)

    def test_n_pairs_exceeds_total_uses_all(self) -> None:
        arc = _make_archive_with_f([1.0, 2.0, 3.0])
        ctx = _make_ctx(archive=arc)
        ts = PairwiseComparisonSet(source="archive", n_pairs=999)
        data = ts.build(arc, None, ctx=ctx)
        assert data.train_x.shape == (3, DIM * 2)  # 3*(3-1)/2 = 3

    def test_train_x_shape_is_2_dim(self) -> None:
        arc = _make_archive_with_f([1.0, 2.0, 3.0])
        ctx = _make_ctx(archive=arc)
        ts = PairwiseComparisonSet(source="archive")
        data = ts.build(arc, None, ctx=ctx)
        assert data.train_x.shape[1] == DIM * 2

    def test_label_ordering_best_vs_worst(self) -> None:
        arc = _make_archive_with_f([1.0, 10.0])  # idx 0 = best, idx 1 = worst
        ctx = _make_ctx(archive=arc)
        ts = PairwiseComparisonSet(source="archive")
        data = ts.build(arc, None, ctx=ctx)
        # Only one pair (0, 1): f=1.0 beats f=10.0 → label 1
        assert data.train_y[0] == pytest.approx(1.0)

    def test_labels_are_binary(self) -> None:
        arc = _make_archive_with_f([1.0, 2.0, 3.0, 4.0])
        ctx = _make_ctx(archive=arc)
        ts = PairwiseComparisonSet(source="archive")
        data = ts.build(arc, None, ctx=ctx)
        assert set(data.train_y.tolist()).issubset({0.0, 1.0})

    def test_ctx_none_raises(self) -> None:
        arc = _make_archive_with_f([1.0, 2.0])
        ts = PairwiseComparisonSet(source="archive")
        with pytest.raises(ValueError, match="ctx"):
            ts.build(arc, None, ctx=None)

    def test_source_population_none_raises(self) -> None:
        arc = _make_archive_with_f([1.0, 2.0])
        ctx = _make_ctx(archive=arc)
        ts = PairwiseComparisonSet(source="population")
        with pytest.raises(ValueError, match="population"):
            ts.build(arc, None, ctx=ctx)

    def test_custom_rng(self) -> None:
        """Custom rng produces reproducible sampling."""
        arc = _make_archive_with_f([1.0, 2.0, 3.0, 4.0, 5.0])
        ctx = _make_ctx(archive=arc)
        rng1 = np.random.default_rng(99)
        rng2 = np.random.default_rng(99)
        ts1 = PairwiseComparisonSet(source="archive", n_pairs=4, rng=rng1)
        ts2 = PairwiseComparisonSet(source="archive", n_pairs=4, rng=rng2)
        d1 = ts1.build(arc, None, ctx=ctx)
        d2 = ts2.build(arc, None, ctx=ctx)
        np.testing.assert_array_equal(d1.train_x, d2.train_x)
        np.testing.assert_array_equal(d1.train_y, d2.train_y)

    def test_no_duplicate_pairs(self) -> None:
        """Each (i, j) pair appears at most once."""
        arc = _make_archive_with_f([1.0, 2.0, 3.0, 4.0, 5.0])
        ctx = _make_ctx(archive=arc)
        ts = PairwiseComparisonSet(source="archive", n_pairs=8)
        data = ts.build(arc, None, ctx=ctx)
        # Each row in train_x is unique (half-vector x_a or full [x_a, x_b])
        assert len(data.train_x) == len(np.unique(data.train_x, axis=0))


# ===========================================================================
# ReferencePointComparisonSet
# ===========================================================================


class TestReferencePointComparisonSet:
    def test_shape_archive_best(self) -> None:
        arc = _make_archive_with_f([1.0, 2.0, 3.0, 4.0])
        ctx = _make_ctx(archive=arc)
        ts = ReferencePointComparisonSet(ref_source="archive_best")
        data = ts.build(arc, None, ctx=ctx)
        assert data.train_x.shape == (4, DIM)
        assert data.train_y.shape == (4,)

    def test_shape_population_best(self) -> None:
        arc = _make_archive_with_f([1.0, 2.0, 3.0])
        pop = _make_population_with_cv([0.0, 0.0, 0.0])
        ctx = _make_ctx(archive=arc, population=pop)
        ts = ReferencePointComparisonSet(ref_source="population_best")
        data = ts.build(arc, pop, ctx=ctx)
        assert data.train_x.shape == (3, DIM)
        assert data.train_y.shape == (3,)

    def test_labels_are_binary(self) -> None:
        arc = _make_archive_with_f([1.0, 2.0, 3.0, 4.0])
        ctx = _make_ctx(archive=arc)
        ts = ReferencePointComparisonSet(ref_source="archive_best")
        data = ts.build(arc, None, ctx=ctx)
        assert set(data.train_y.tolist()).issubset({0.0, 1.0})

    def test_ctx_none_raises(self) -> None:
        arc = _make_archive_with_f([1.0, 2.0])
        ts = ReferencePointComparisonSet(ref_source="archive_best")
        with pytest.raises(ValueError, match="ctx"):
            ts.build(arc, None, ctx=None)

    def test_population_none_raises_for_population_best(self) -> None:
        arc = _make_archive_with_f([1.0, 2.0])
        ctx = _make_ctx(archive=arc)
        ts = ReferencePointComparisonSet(ref_source="population_best")
        with pytest.raises(ValueError, match="population"):
            ts.build(arc, None, ctx=ctx)

    def test_archive_best_reference_gets_label_one(self) -> None:
        """The best archive point dominates itself, so its label must be 1."""
        arc = _make_archive_with_f([1.0, 5.0, 10.0])
        ctx = _make_ctx(archive=arc)
        ts = ReferencePointComparisonSet(ref_source="archive_best")
        data = ts.build(arc, None, ctx=ctx)
        # The point with f=1.0 is best (minimization); it equals the reference
        # so compare(...) == 0 → label 1
        assert 1.0 in data.train_y

    def test_default_ref_source(self) -> None:
        ts = ReferencePointComparisonSet()
        assert ts.ref_source == "population_best"

    def test_e2e_global_surrogate_manager(self) -> None:
        """ReferencePointComparisonSet works end-to-end with GlobalSurrogateManager."""
        from saealib.acquisition.mean import MeanPrediction
        from saealib.surrogate.manager import GlobalSurrogateManager
        from saealib.surrogate.sklearn_surrogate import SklearnRFRSurrogate

        arc = _make_archive_with_f([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        pop = _make_population_with_cv([0.0, 0.0, 0.0])
        ctx = _make_ctx(archive=arc, population=pop)
        ts = ReferencePointComparisonSet(ref_source="population_best")
        # MeanPrediction with direction=[1.0]: higher predicted label = higher score
        mgr = GlobalSurrogateManager(
            surrogate=SklearnRFRSurrogate(),
            training_set=ts,
        )
        acquisition = MeanPrediction(direction=np.array([1.0]))
        candidates_x = np.random.default_rng(0).uniform(-2.0, 2.0, size=(5, DIM))
        prediction = mgr.predict(candidates_x, arc, ctx=ctx)
        scores = acquisition.evaluate(candidates_x, prediction, arc, ctx=ctx).scores
        assert scores is not None
        assert scores.shape == (5,)
        assert prediction.value.shape == (5, 1)
