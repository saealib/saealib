"""
Tests for evolutionary operators.

Tests cover:
- CrossoverBLXAlpha: alpha rename, output shape, determinism
- CrossoverSBX: output shape, center preservation, determinism
- CrossoverUniform: output shape, swap_rate=0/1 boundary cases
- CrossoverOnePoint: output shape, segment integrity
- CrossoverTwoPoint: output shape, segment integrity
- Crossover (base): n_children default and consistency with output shape
- MutationPolynomial: output shape, within-bounds, zero-rate
- MutationGaussian: output shape, zero-rate, zero-sigma
- LinearRankSelection: output shape, probability bias
"""

from typing import cast

import numpy as np
import pytest
from _algorithm_boundary import ask as algorithm_ask

from saealib import GA, SequentialSelection, TruncationSelection, minimize
from saealib.comparators import NSGA2Comparator, SingleObjectiveComparator
from saealib.context import OptimizationState
from saealib.operators.crossover import (
    Crossover,
    CrossoverBLXAlpha,
    CrossoverCategorical,
    CrossoverIntegerSBX,
    CrossoverOnePoint,
    CrossoverSBX,
    CrossoverTwoPoint,
    CrossoverUniform,
)
from saealib.operators.mutation import (
    Mutation,
    MutationCategorical,
    MutationGaussian,
    MutationIntegerUniform,
    MutationPolynomial,
    MutationUniform,
)
from saealib.operators.selection import LinearRankSelection, TournamentSelection
from saealib.population import Archive, ParetoArchive, Population, PopulationAttribute
from saealib.problem import Problem

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

DIM = 6
_ATTRS = [
    PopulationAttribute(name="x", dtype=np.float64, shape=(DIM,)),
    PopulationAttribute(name="f", dtype=np.float64, shape=(1,)),
    PopulationAttribute(name="g", dtype=np.float64, shape=(0,)),
    PopulationAttribute(name="cv", dtype=np.float64, shape=()),
]


def _make_parents(rng: np.random.Generator) -> np.ndarray:
    """Two parent individuals, shape=(2, DIM)."""
    return rng.uniform(-1.0, 1.0, size=(2, DIM))


def _make_problem() -> Problem:
    return Problem(
        func=lambda x: np.array([np.sum(x**2)]),
        dim=DIM,
        n_obj=1,
        direction=np.array([-1.0]),
        lb=[-5.0] * DIM,
        ub=[5.0] * DIM,
        comparator=SingleObjectiveComparator(),
    )


def _make_ctx(n_pop: int = 10, rng_seed: int = 0) -> OptimizationState:
    problem = _make_problem()
    rng = np.random.default_rng(rng_seed)
    pop = Population(_ATTRS, init_capacity=n_pop + 5)
    xs = rng.uniform(-3.0, 3.0, size=(n_pop, DIM))
    fs = np.array([[np.sum(x**2)] for x in xs])
    gs = np.zeros((n_pop, 0))
    cvs = np.zeros(n_pop)
    pop.extend({"x": xs, "f": fs, "g": gs, "cv": cvs})
    arc = Archive(_ATTRS, init_capacity=5)
    pareto_arc = ParetoArchive(_ATTRS, init_capacity=5, direction=np.array([-1.0]))
    return OptimizationState(
        problem=problem,
        population=pop,
        archive=arc,
        pareto_archive=pareto_arc,
        rng=np.random.default_rng(rng_seed),
    )


# ---------------------------------------------------------------------------
# CrossoverBLXAlpha
# ---------------------------------------------------------------------------


class TestCrossoverBLXAlpha:
    def test_alpha_attribute(self):
        op = CrossoverBLXAlpha(prob=0.7, alpha=0.4)
        assert op.alpha == pytest.approx(0.4)

    def test_output_shape(self):
        op = CrossoverBLXAlpha(prob=0.7, alpha=0.4)
        rng = np.random.default_rng(0)
        p = _make_parents(rng)
        c = op.crossover(p, rng=rng)
        assert c.shape == (2, DIM)

    def test_deterministic_with_seed(self):
        op = CrossoverBLXAlpha(prob=0.7, alpha=0.4)
        p = _make_parents(np.random.default_rng(1))
        c1 = op.crossover(p, rng=np.random.default_rng(42))
        c2 = op.crossover(p, rng=np.random.default_rng(42))
        np.testing.assert_array_equal(c1, c2)

    def test_crossover_batch_output_shape(self):
        op = CrossoverBLXAlpha(prob=0.7, alpha=0.4)
        rng = np.random.default_rng(0)
        n_pair = 5
        parents_batch = rng.uniform(-1.0, 1.0, size=(n_pair, 2, DIM))
        c = op.crossover_batch(parents_batch, rng=rng)
        assert c is not None
        assert c.shape == (n_pair, 2, DIM)

    def test_crossover_batch_matches_single_at_n_pair_one(self):
        op = CrossoverBLXAlpha(prob=0.7, alpha=0.4)
        p = _make_parents(np.random.default_rng(1))
        parents_batch = p[np.newaxis, :, :]
        c_batch = op.crossover_batch(parents_batch, rng=np.random.default_rng(42))
        assert c_batch is not None
        c_single = op.crossover(p, rng=np.random.default_rng(42))
        np.testing.assert_array_equal(c_batch[0], c_single)

    def test_crossover_batch_deterministic_with_seed(self):
        op = CrossoverBLXAlpha(prob=0.7, alpha=0.4)
        rng = np.random.default_rng(0)
        n_pair = 5
        parents_batch = rng.uniform(-1.0, 1.0, size=(n_pair, 2, DIM))
        c1 = op.crossover_batch(parents_batch, rng=np.random.default_rng(42))
        c2 = op.crossover_batch(parents_batch, rng=np.random.default_rng(42))
        np.testing.assert_array_equal(c1, c2)


# ---------------------------------------------------------------------------
# CrossoverSBX
# ---------------------------------------------------------------------------


class TestCrossoverSBX:
    def test_output_shape(self):
        op = CrossoverSBX(prob=0.9, eta=2.0)
        rng = np.random.default_rng(0)
        p = _make_parents(rng)
        c = op.crossover(p, rng=rng)
        assert c.shape == (2, DIM)

    def test_center_preservation(self):
        """Mid-point of children equals mid-point of parents (SBX property)."""
        op = CrossoverSBX(prob=0.9, eta=2.0)
        rng = np.random.default_rng(5)
        p = _make_parents(rng)
        c = op.crossover(p, rng=rng)
        np.testing.assert_allclose((c[0] + c[1]) / 2, (p[0] + p[1]) / 2, atol=1e-10)

    def test_deterministic_with_seed(self):
        op = CrossoverSBX(prob=0.9, eta=2.0)
        p = _make_parents(np.random.default_rng(1))
        c1 = op.crossover(p, rng=np.random.default_rng(42))
        c2 = op.crossover(p, rng=np.random.default_rng(42))
        np.testing.assert_array_equal(c1, c2)

    def test_crossover_batch_output_shape(self):
        op = CrossoverSBX(prob=0.9, eta=2.0)
        rng = np.random.default_rng(0)
        n_pair = 5
        parents_batch = rng.uniform(-1.0, 1.0, size=(n_pair, 2, DIM))
        c = op.crossover_batch(parents_batch, rng=rng)
        assert c is not None
        assert c.shape == (n_pair, 2, DIM)

    def test_crossover_batch_matches_single_at_n_pair_one(self):
        """Unbounded branch: draws are u, then do_cross (2 phases)."""
        op = CrossoverSBX(prob=0.9, eta=2.0)
        p = _make_parents(np.random.default_rng(1))
        parents_batch = p[np.newaxis, :, :]
        c_batch = op.crossover_batch(parents_batch, rng=np.random.default_rng(42))
        assert c_batch is not None
        c_single = op.crossover(p, rng=np.random.default_rng(42))
        np.testing.assert_array_equal(c_batch[0], c_single)

    def test_crossover_batch_matches_single_at_n_pair_one_bounded(self):
        """Bounded branch: draws are u, then swap, then do_cross (3 phases)
        -- the branch GA actually exercises, since GA always supplies bounds."""
        op = CrossoverSBX(prob=1.0, eta=20.0, prob_var=1.0)
        lb = np.zeros(DIM)
        ub = np.ones(DIM)
        p = np.random.default_rng(1).uniform(0.0, 1.0, size=(2, DIM))
        parents_batch = p[np.newaxis, :, :]
        c_batch = op.crossover_batch(
            parents_batch, (lb, ub), rng=np.random.default_rng(42)
        )
        assert c_batch is not None
        c_single = op.crossover(p, (lb, ub), rng=np.random.default_rng(42))
        np.testing.assert_array_equal(c_batch[0], c_single)

    def test_crossover_batch_deterministic_with_seed(self):
        op = CrossoverSBX(prob=0.9, eta=2.0)
        rng = np.random.default_rng(0)
        n_pair = 5
        parents_batch = rng.uniform(-1.0, 1.0, size=(n_pair, 2, DIM))
        c1 = op.crossover_batch(parents_batch, rng=np.random.default_rng(42))
        c2 = op.crossover_batch(parents_batch, rng=np.random.default_rng(42))
        np.testing.assert_array_equal(c1, c2)

    def test_crossover_batch_respects_bounds(self):
        op = CrossoverSBX(prob=1.0, eta=20.0, prob_var=1.0)
        lb = np.zeros(DIM)
        ub = np.ones(DIM)
        rng = np.random.default_rng(0)
        n_pair = 8
        for _ in range(20):
            parents_batch = rng.uniform(0.0, 1.0, size=(n_pair, 2, DIM))
            c = op.crossover_batch(parents_batch, (lb, ub), rng=rng)
            assert c is not None
            assert np.all(c >= lb) and np.all(c <= ub)

    def test_crossover_batch_mixed_identical_and_separated_rows(self):
        """A batch with one row of identical parents and one row of
        separated parents: only the identical row must come back unchanged
        (batch mirror of TestCrossoverSBXBounded.test_identical_parents_unchanged
        -- confirms `separated` masks per row, not collapsed across the batch)."""
        op = CrossoverSBX(prob=1.0, eta=20.0, prob_var=1.0)
        lb = np.zeros(3)
        ub = np.ones(3)
        identical = np.array([[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]])
        separated = np.array([[0.1, 0.2, 0.3], [0.8, 0.7, 0.9]])
        parents_batch = np.stack([identical, separated], axis=0)
        c = op.crossover_batch(parents_batch, (lb, ub), rng=np.random.default_rng(0))
        assert c is not None
        np.testing.assert_array_equal(c[0, 0], identical[0])
        np.testing.assert_array_equal(c[0, 1], identical[1])
        assert np.all(c[1] >= lb) and np.all(c[1] <= ub)
        # discriminates per-row masking from a collapsed `separated` mask:
        # if row 1's separated-ness were incorrectly suppressed by row 0's
        # identical-ness, c[1] would also come back as unmodified parents
        assert not np.array_equal(c[1, 0], separated[0])


# ---------------------------------------------------------------------------
# CrossoverUniform
# ---------------------------------------------------------------------------


class TestCrossoverUniform:
    def test_output_shape(self):
        op = CrossoverUniform(prob=0.8)
        rng = np.random.default_rng(0)
        p = _make_parents(rng)
        c = op.crossover(p, rng=rng)
        assert c.shape == (2, DIM)

    def test_swap_rate_zero(self):
        """swap_rate=0: c1==p1 and c2==p2."""
        op = CrossoverUniform(prob=0.8, swap_rate=0.0)
        rng = np.random.default_rng(0)
        p = _make_parents(rng)
        c = op.crossover(p, rng=rng)
        np.testing.assert_array_equal(c[0], p[0])
        np.testing.assert_array_equal(c[1], p[1])

    def test_swap_rate_one(self):
        """swap_rate=1: c1==p2 and c2==p1."""
        op = CrossoverUniform(prob=0.8, swap_rate=1.0)
        rng = np.random.default_rng(0)
        p = _make_parents(rng)
        c = op.crossover(p, rng=rng)
        np.testing.assert_array_equal(c[0], p[1])
        np.testing.assert_array_equal(c[1], p[0])

    def test_crossover_batch_output_shape(self):
        op = CrossoverUniform(prob=0.8)
        rng = np.random.default_rng(0)
        n_pair = 5
        parents_batch = rng.uniform(-1.0, 1.0, size=(n_pair, 2, DIM))
        c = op.crossover_batch(parents_batch, rng=rng)
        assert c is not None
        assert c.shape == (n_pair, 2, DIM)

    def test_crossover_batch_matches_single_at_n_pair_one(self):
        op = CrossoverUniform(prob=0.8)
        p = _make_parents(np.random.default_rng(1))
        parents_batch = p[np.newaxis, :, :]
        c_batch = op.crossover_batch(parents_batch, rng=np.random.default_rng(42))
        assert c_batch is not None
        c_single = op.crossover(p, rng=np.random.default_rng(42))
        np.testing.assert_array_equal(c_batch[0], c_single)

    def test_crossover_batch_deterministic_with_seed(self):
        op = CrossoverUniform(prob=0.8)
        rng = np.random.default_rng(0)
        n_pair = 5
        parents_batch = rng.uniform(-1.0, 1.0, size=(n_pair, 2, DIM))
        c1 = op.crossover_batch(parents_batch, rng=np.random.default_rng(42))
        c2 = op.crossover_batch(parents_batch, rng=np.random.default_rng(42))
        np.testing.assert_array_equal(c1, c2)

    def test_crossover_batch_swap_rate_zero(self):
        """swap_rate=0: c1==p1 and c2==p2 across every row."""
        op = CrossoverUniform(prob=0.8, swap_rate=0.0)
        rng = np.random.default_rng(0)
        n_pair = 5
        parents_batch = rng.uniform(-1.0, 1.0, size=(n_pair, 2, DIM))
        c = op.crossover_batch(parents_batch, rng=rng)
        assert c is not None
        np.testing.assert_array_equal(c[:, 0, :], parents_batch[:, 0, :])
        np.testing.assert_array_equal(c[:, 1, :], parents_batch[:, 1, :])

    def test_crossover_batch_swap_rate_one(self):
        """swap_rate=1: c1==p2 and c2==p1 across every row."""
        op = CrossoverUniform(prob=0.8, swap_rate=1.0)
        rng = np.random.default_rng(0)
        n_pair = 5
        parents_batch = rng.uniform(-1.0, 1.0, size=(n_pair, 2, DIM))
        c = op.crossover_batch(parents_batch, rng=rng)
        assert c is not None
        np.testing.assert_array_equal(c[:, 0, :], parents_batch[:, 1, :])
        np.testing.assert_array_equal(c[:, 1, :], parents_batch[:, 0, :])


# ---------------------------------------------------------------------------
# CrossoverOnePoint
# ---------------------------------------------------------------------------


class TestCrossoverOnePoint:
    def test_output_shape(self):
        op = CrossoverOnePoint(prob=0.8)
        rng = np.random.default_rng(0)
        p = _make_parents(rng)
        c = op.crossover(p, rng=rng)
        assert c.shape == (2, DIM)

    def test_segment_intact(self):
        """Before cut point, c1 == p1; from cut point onward, c1 == p2."""
        op = CrossoverOnePoint(prob=0.8)
        rng = np.random.default_rng(0)
        p = _make_parents(rng)
        # Fix cut point by seeding: integers(1, DIM) with seed=7 gives a known value
        rng2 = np.random.default_rng(7)
        point = rng2.integers(1, DIM)
        rng3 = np.random.default_rng(7)
        c = op.crossover(p, rng=rng3)
        np.testing.assert_array_equal(c[0, :point], p[0, :point])
        np.testing.assert_array_equal(c[0, point:], p[1, point:])
        np.testing.assert_array_equal(c[1, :point], p[1, :point])
        np.testing.assert_array_equal(c[1, point:], p[0, point:])

    def test_crossover_batch_output_shape(self):
        op = CrossoverOnePoint(prob=0.8)
        rng = np.random.default_rng(0)
        n_pair = 5
        parents_batch = rng.uniform(-1.0, 1.0, size=(n_pair, 2, DIM))
        c = op.crossover_batch(parents_batch, rng=rng)
        assert c is not None
        assert c.shape == (n_pair, 2, DIM)

    def test_crossover_batch_deterministic_with_seed(self):
        op = CrossoverOnePoint(prob=0.8)
        rng = np.random.default_rng(0)
        n_pair = 5
        parents_batch = rng.uniform(-1.0, 1.0, size=(n_pair, 2, DIM))
        c1 = op.crossover_batch(parents_batch, rng=np.random.default_rng(42))
        c2 = op.crossover_batch(parents_batch, rng=np.random.default_rng(42))
        np.testing.assert_array_equal(c1, c2)

    def test_crossover_batch_matches_loop_over_many_rows(self):
        """crossover_batch's single rng.integers(1, dim, size=n_pair) call
        consumes the rng stream identically to n_pair sequential
        rng.integers(1, dim) calls, so every row (not just n_pair == 1)
        must match a loop of crossover() calls sharing one rng."""
        op = CrossoverOnePoint(prob=0.8)
        n_pair = 20
        parents_batch = np.random.default_rng(3).uniform(
            -1.0, 1.0, size=(n_pair, 2, DIM)
        )
        c_batch = op.crossover_batch(parents_batch, rng=np.random.default_rng(42))
        assert c_batch is not None

        shared_rng = np.random.default_rng(42)
        for i in range(n_pair):
            c_single = op.crossover(parents_batch[i], rng=shared_rng)
            np.testing.assert_array_equal(c_batch[i], c_single)


# ---------------------------------------------------------------------------
# CrossoverTwoPoint
# ---------------------------------------------------------------------------


class TestCrossoverTwoPoint:
    def test_output_shape(self):
        op = CrossoverTwoPoint(prob=0.8)
        rng = np.random.default_rng(0)
        p = _make_parents(rng)
        c = op.crossover(p, rng=rng)
        assert c.shape == (2, DIM)

    def test_segment_intact(self):
        """Between the two cut points, c1 == p2; outside, c1 == p1."""
        op = CrossoverTwoPoint(prob=0.8)
        rng2 = np.random.default_rng(3)
        pts = np.sort(rng2.choice(DIM + 1, size=2, replace=False))
        pt1, pt2 = pts[0], pts[1]
        rng3 = np.random.default_rng(3)
        p = _make_parents(np.random.default_rng(99))
        c = op.crossover(p, rng=rng3)
        np.testing.assert_array_equal(c[0, :pt1], p[0, :pt1])
        np.testing.assert_array_equal(c[0, pt1:pt2], p[1, pt1:pt2])
        np.testing.assert_array_equal(c[0, pt2:], p[0, pt2:])

    def test_crossover_batch_output_shape(self):
        op = CrossoverTwoPoint(prob=0.8)
        rng = np.random.default_rng(0)
        n_pair = 5
        parents_batch = rng.uniform(-1.0, 1.0, size=(n_pair, 2, DIM))
        c = op.crossover_batch(parents_batch, rng=rng)
        assert c is not None
        assert c.shape == (n_pair, 2, DIM)

    def test_crossover_batch_deterministic_with_seed(self):
        op = CrossoverTwoPoint(prob=0.8)
        rng = np.random.default_rng(0)
        n_pair = 5
        parents_batch = rng.uniform(-1.0, 1.0, size=(n_pair, 2, DIM))
        c1 = op.crossover_batch(parents_batch, rng=np.random.default_rng(42))
        c2 = op.crossover_batch(parents_batch, rng=np.random.default_rng(42))
        np.testing.assert_array_equal(c1, c2)

    def test_crossover_batch_matches_loop_over_many_rows(self):
        """crossover_batch draws its two cut points per row via the same
        rng.choice(dim + 1, size=2, replace=False) call and in the same
        order as a loop of crossover() calls sharing one rng, so every row
        (not just n_pair == 1) must match exactly."""
        op = CrossoverTwoPoint(prob=0.8)
        n_pair = 20
        parents_batch = np.random.default_rng(3).uniform(
            -1.0, 1.0, size=(n_pair, 2, DIM)
        )
        c_batch = op.crossover_batch(parents_batch, rng=np.random.default_rng(42))
        assert c_batch is not None

        shared_rng = np.random.default_rng(42)
        for i in range(n_pair):
            c_single = op.crossover(parents_batch[i], rng=shared_rng)
            np.testing.assert_array_equal(c_batch[i], c_single)

    def test_crossover_batch_dim_two_boundary(self):
        """dim == 2: only possible sorted cut points from choice(3, 2,
        replace=False) are {0,1}, {0,2}, {1,2} -- exercises the mask logic
        at the smallest useful boundary."""
        op = CrossoverTwoPoint(prob=0.8)
        n_pair = 10
        parents_batch = np.random.default_rng(5).uniform(-1.0, 1.0, size=(n_pair, 2, 2))
        c_batch = op.crossover_batch(parents_batch, rng=np.random.default_rng(7))
        assert c_batch is not None
        assert c_batch.shape == (n_pair, 2, 2)

        shared_rng = np.random.default_rng(7)
        for i in range(n_pair):
            c_single = op.crossover(parents_batch[i], rng=shared_rng)
            np.testing.assert_array_equal(c_batch[i], c_single)


# ---------------------------------------------------------------------------
# Crossover base: n_children
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "op",
    [
        CrossoverBLXAlpha(prob=0.9, alpha=0.4),
        CrossoverSBX(prob=0.9, eta=2.0),
        CrossoverUniform(prob=0.9),
        CrossoverOnePoint(prob=0.9),
        CrossoverTwoPoint(prob=0.9),
    ],
)
class TestCrossoverNChildren:
    def test_n_children_default(self, op):
        assert op.n_children == 2

    def test_output_rows_match_n_children(self, op):
        rng = np.random.default_rng(0)
        c = op.crossover(_make_parents(rng), rng=rng)
        assert c.shape[0] == op.n_children


# ---------------------------------------------------------------------------
# Crossover base: crossover_batch requirement
# ---------------------------------------------------------------------------


class _CrossoverUnbatched(Crossover):
    """Minimal dummy Crossover that does not implement crossover_batch."""

    def __init__(self, prob: float = 0.9):
        super().__init__()
        self.prob = prob

    def crossover(
        self,
        parent: np.ndarray,
        bounds: tuple[np.ndarray, np.ndarray] | None = None,
        rng: np.random.Generator = np.random.default_rng(),
    ) -> np.ndarray:
        return np.array([parent[0], parent[1]])


class TestCrossoverBatchRequired:
    def test_crossover_batch_is_required(self):
        with pytest.raises(TypeError, match="crossover_batch"):
            _CrossoverUnbatched(prob=0.9)


class TestCrossoverDerived:
    """Confirm built-ins inherit the scalar operation from the batch primitive."""

    @pytest.mark.parametrize(
        "op",
        [
            CrossoverSBX(prob=0.9, eta=2.0),
            CrossoverUniform(prob=0.8),
            CrossoverCategorical(prob=1.0),
            CrossoverIntegerSBX(prob=1.0, eta=2.0),
            CrossoverBLXAlpha(prob=0.9, alpha=0.4),
            CrossoverOnePoint(prob=0.9),
            CrossoverTwoPoint(prob=0.9),
        ],
    )
    def test_inherits_crossover(self, op):
        assert "crossover" not in vars(type(op))
        assert op.crossover.__func__ is Crossover.crossover


# ---------------------------------------------------------------------------
# MutationPolynomial
# ---------------------------------------------------------------------------


class TestMutationPolynomial:
    def _range(self):
        lb = np.full(DIM, -5.0)
        ub = np.full(DIM, 5.0)
        return lb, ub

    def test_output_shape(self):
        op = MutationPolynomial(prob_var=0.5, eta=20.0)
        rng = np.random.default_rng(0)
        p = rng.uniform(-2.0, 2.0, size=DIM)
        c = op.mutate(p, self._range(), rng=rng)
        assert c.shape == (DIM,)

    def test_within_bounds(self):
        op = MutationPolynomial(prob_var=1.0, eta=20.0)
        lb, ub = self._range()
        rng = np.random.default_rng(0)
        for _ in range(20):
            p = rng.uniform(-4.0, 4.0, size=DIM)
            c = op.mutate(p, (lb, ub), rng=rng)
            assert np.all(c >= lb) and np.all(c <= ub)

    def test_zero_rate_no_change(self):
        op = MutationPolynomial(prob_var=0.0, eta=20.0)
        rng = np.random.default_rng(0)
        p = rng.uniform(-2.0, 2.0, size=DIM)
        c = op.mutate(p, self._range(), rng=rng)
        np.testing.assert_array_equal(c, p)

    def test_asymmetric_delta_allows_larger_boundary_excursion(self):
        # p near lb, ub far away: the old shared min(delta1, delta2)
        # formula reused the (tiny) lower-bound delta for the upward
        # (u>0.5) branch too, capping upward excursions near zero. The
        # asymmetric formula uses delta2 (upper-bound distance) for that
        # branch instead, matching nsga2-gnuplot-v1.1.6 / pymoo / DEAP.
        op = MutationPolynomial(prob_var=1.0, eta=20.0)
        lb = np.array([0.0])
        ub = np.array([100.0])
        p = np.array([1.0])
        rng = np.random.default_rng(0)
        max_excursion = 0.0
        for _ in range(50):
            c = op.mutate(p, (lb, ub), rng=rng)
            max_excursion = max(max_excursion, float(c[0] - p[0]))
        assert max_excursion > 5.0


# ---------------------------------------------------------------------------
# MutationGaussian
# ---------------------------------------------------------------------------


class TestMutationGaussian:
    def _range(self):
        lb = np.full(DIM, -5.0)
        ub = np.full(DIM, 5.0)
        return lb, ub

    def test_output_shape(self):
        op = MutationGaussian(prob_var=0.5, sigma=0.1)
        rng = np.random.default_rng(0)
        p = rng.uniform(-2.0, 2.0, size=DIM)
        c = op.mutate(p, self._range(), rng=rng)
        assert c.shape == (DIM,)

    def test_zero_rate_no_change(self):
        op = MutationGaussian(prob_var=0.0, sigma=1.0)
        rng = np.random.default_rng(0)
        p = rng.uniform(-2.0, 2.0, size=DIM)
        c = op.mutate(p, self._range(), rng=rng)
        np.testing.assert_array_equal(c, p)

    def test_zero_sigma_no_change(self):
        op = MutationGaussian(prob_var=1.0, sigma=0.0)
        rng = np.random.default_rng(0)
        p = rng.uniform(-2.0, 2.0, size=DIM)
        c = op.mutate(p, self._range(), rng=rng)
        np.testing.assert_array_equal(c, p)


# ---------------------------------------------------------------------------
# Mutation base: mutate_batch requirement
# ---------------------------------------------------------------------------


class _MutationUnbatched(Mutation):
    """Mutation dummy used by the GA fallback-loop interleaving test.

    ``mutate()`` draws exactly one ``rng.random()`` value per call (gated on
    ``prob`` like every real Mutation, though the outcome is always
    ``p.copy()`` either way) so that RNG-consumption ordering is
    observable to callers -- a version that drew zero values from ``rng``
    would make any test relying on interleaving/reordering of RNG draws
    vacuously pass regardless of correctness.
    """

    def __init__(self, prob: float = 1.0):
        super().__init__()
        self.prob = prob
        self.batch_calls = 0

    def mutate(
        self,
        p: np.ndarray,
        mutate_range: tuple,
        rng: np.random.Generator = np.random.default_rng(),
    ) -> np.ndarray:
        # Both branches return p.copy(): the gate draw exists solely so this
        # dummy consumes exactly one rng value per call, making RNG-ordering
        # bugs observable to callers (see class docstring).
        if rng.random() >= self.prob:
            return p.copy()
        return p.copy()

    def mutate_batch(
        self,
        candidates_batch: np.ndarray,
        mutate_range: tuple,
        rng: np.random.Generator = np.random.default_rng(),
    ) -> np.ndarray:
        self.batch_calls += 1
        return candidates_batch.copy()


class _MutationWithoutBatch(Mutation):
    """Minimal dummy Mutation that does not implement mutate_batch."""

    def mutate(
        self,
        p: np.ndarray,
        mutate_range: tuple,
        rng: np.random.Generator = np.random.default_rng(),
    ) -> np.ndarray:
        return p.copy()


class TestMutationBatchRequired:
    def test_mutate_batch_is_required(self):
        with pytest.raises(TypeError, match="mutate_batch"):
            _MutationWithoutBatch()


class TestMutationDerived:
    """Confirm built-ins inherit the scalar operation from the batch primitive."""

    @pytest.mark.parametrize(
        "op",
        [
            MutationUniform(),
            MutationPolynomial(eta=20.0),
            MutationGaussian(sigma=1.0),
            MutationIntegerUniform(),
            MutationCategorical(),
        ],
    )
    def test_inherits_mutate(self, op):
        assert "mutate" not in vars(type(op))
        assert op.mutate.__func__ is Mutation.mutate


# ---------------------------------------------------------------------------
# LinearRankSelection
# ---------------------------------------------------------------------------


class TestLinearRankSelection:
    def test_output_shape(self):
        op = LinearRankSelection()
        ctx = _make_ctx(n_pop=10)
        rng = np.random.default_rng(0)
        idx = op.select(ctx.population, n_pair=4, n_parents=2, rng=rng, ctx=ctx)
        assert idx.shape == (4, 2)

    def test_indices_in_range(self):
        op = LinearRankSelection()
        ctx = _make_ctx(n_pop=10)
        rng = np.random.default_rng(0)
        idx = op.select(ctx.population, n_pair=5, n_parents=2, rng=rng, ctx=ctx)
        assert np.all(idx >= 0) and np.all(idx < len(ctx.population))

    def test_best_selected_more_often(self):
        """Best individual (lowest f) should appear more than worst over many trials."""
        op = LinearRankSelection()
        n_pop = 10
        ctx = _make_ctx(n_pop=n_pop, rng_seed=0)
        # Identify best and worst indices via comparator
        sorted_idx = ctx.comparator.sort_population(ctx.population)
        best_idx = sorted_idx[0]
        worst_idx = sorted_idx[-1]

        rng = np.random.default_rng(1)
        counts = np.zeros(n_pop, dtype=int)
        for _ in range(500):
            idx = op.select(ctx.population, n_pair=10, n_parents=1, rng=rng, ctx=ctx)
            for i in idx.flatten():
                counts[i] += 1

        assert counts[best_idx] > counts[worst_idx]


# ---------------------------------------------------------------------------
# TruncationSelection
# ---------------------------------------------------------------------------


def _make_moo_ctx(f_values: np.ndarray, rng_seed: int = 0) -> OptimizationState:
    """MOO context with an NSGA2Comparator and explicit objective values."""
    n, n_obj = f_values.shape
    attrs = [
        PopulationAttribute(name="x", dtype=np.float64, shape=(n_obj,)),
        PopulationAttribute(name="f", dtype=np.float64, shape=(n_obj,)),
        PopulationAttribute(name="cv", dtype=np.float64, shape=(), default=0.0),
    ]
    pop = Population(attrs, init_capacity=n + 1)
    for row in f_values:
        pop.append(x=row, f=row, cv=0.0)
    direction = np.full(n_obj, -1.0)
    comparator = NSGA2Comparator(direction=direction)
    problem = Problem(
        func=lambda x: x,
        dim=n_obj,
        n_obj=n_obj,
        direction=direction,
        lb=[-10.0] * n_obj,
        ub=[10.0] * n_obj,
        comparator=comparator,
    )
    arc = Archive(attrs, init_capacity=5)
    pareto_arc = ParetoArchive(attrs, init_capacity=5, direction=direction)
    return OptimizationState(
        problem=problem,
        population=pop,
        archive=arc,
        pareto_archive=pareto_arc,
        rng=np.random.default_rng(rng_seed),
    )


class TestTruncationSelection:
    # Front with a genuine crowding-distance tie between rows 2 and 4
    # (both f=[0.5, 0.5], rank=0, cd=1.0), plus a dominated point (row 5).
    # Verified via direct inspection of NSGA2Comparator's cached rank/cd:
    # order=[0,1,2,4,3,5], rank=[0,0,0,0,0,1], cd=[inf,inf,1,0,1,inf].
    _F_WITH_TIE = np.array(
        [
            [0.0, 1.0],
            [1.0, 0.0],
            [0.5, 0.5],
            [0.5, 0.5],
            [0.5, 0.5],
            [2.0, 2.0],
        ]
    )
    # Unequal spacing along the tradeoff line so crowding distances differ
    # (evenly-spaced points give symmetric, tied cd values by construction).
    # Verified: order=[0,5,4,3,2,1], rank=all 0, cd=[inf,0.7,0.8,0.9,1.0,inf].
    _F_NO_TIE = np.array(
        [
            [0.0, 1.0],
            [0.1, 0.9],
            [0.35, 0.65],
            [0.5, 0.5],
            [0.8, 0.2],
            [1.0, 0.0],
        ]
    )

    def test_default_matches_plain_sort(self):
        op = TruncationSelection()
        assert op.randomize_ties is False
        ctx = _make_moo_ctx(self._F_WITH_TIE)
        expected = ctx.comparator.sort_population(ctx.population)[:3]
        result = op.select(ctx.population, 3, ctx=ctx)
        np.testing.assert_array_equal(result, expected)

    def test_randomize_ties_no_ties_matches_plain_sort(self):
        op = TruncationSelection(randomize_ties=True)
        ctx = _make_moo_ctx(self._F_NO_TIE)
        expected = ctx.comparator.sort_population(ctx.population)[:3]
        result = op.select(ctx.population, 3, ctx=ctx)
        np.testing.assert_array_equal(result, expected)

    def test_randomize_ties_preserves_non_tied_boundary(self):
        op = TruncationSelection(randomize_ties=True)
        for seed in range(10):
            ctx = _make_moo_ctx(self._F_WITH_TIE, rng_seed=seed)
            result = op.select(ctx.population, 3, ctx=ctx)
            assert result.shape == (3,)
            assert 0 in result and 1 in result  # cd=inf, always kept
            assert 5 not in result  # dominated, never kept
            assert set(result.tolist()) <= {0, 1, 2, 4}

    def test_randomize_ties_varies_across_seeds(self):
        op = TruncationSelection(randomize_ties=True)
        outcomes = set()
        for seed in range(20):
            ctx = _make_moo_ctx(self._F_WITH_TIE, rng_seed=seed)
            result = op.select(ctx.population, 3, ctx=ctx)
            outcomes.add(frozenset(result.tolist()))
        assert outcomes == {frozenset({0, 1, 2}), frozenset({0, 1, 4})}

    def test_n_survivors_exceeds_pool(self):
        op = TruncationSelection(randomize_ties=True)
        ctx = _make_moo_ctx(self._F_WITH_TIE)
        result = op.select(ctx.population, 10, ctx=ctx)
        assert set(result.tolist()) == set(range(6))


# ---------------------------------------------------------------------------
# Crossover lifecycle hooks
# ---------------------------------------------------------------------------


class TestCrossoverHooks:
    def _op(self):
        return CrossoverBLXAlpha(prob=0.9, alpha=0.4)

    def _parents(self, rng=None):
        rng = rng if rng is not None else np.random.default_rng(0)
        return rng.uniform(-1.0, 1.0, size=(2, DIM))

    def test_post_crossover_default_is_noop(self):
        op = self._op()
        rng = np.random.default_rng(0)
        parents = self._parents(rng)
        offspring = op.crossover(parents, rng=rng)
        result = op.post_crossover(offspring.copy(), parents, rng)
        np.testing.assert_array_equal(result, offspring)

    def test_with_post_transforms_offspring(self):
        op = self._op().with_post(lambda o, p, rng, ctx: np.zeros_like(o))
        rng = np.random.default_rng(0)
        parents = self._parents(rng)
        offspring = op.crossover(parents, rng=rng)
        result = op.post_crossover(offspring, parents, rng)
        np.testing.assert_array_equal(result, np.zeros_like(offspring))

    def test_with_post_fn_receives_correct_args(self):
        received = {}

        def hook(offspring, parents, rng, ctx):
            received["offspring_shape"] = offspring.shape
            received["parents_shape"] = parents.shape
            received["ctx"] = ctx
            return offspring

        ctx = _make_ctx()
        op = self._op().with_post(hook)
        rng = np.random.default_rng(0)
        parents = self._parents(rng)
        offspring = op.crossover(parents, rng=rng)
        op.post_crossover(offspring, parents, rng, ctx)
        assert received["offspring_shape"] == (2, DIM)
        assert received["parents_shape"] == (2, DIM)
        assert received["ctx"] is ctx

    def test_with_post_chains_in_order(self):
        log = []
        op = (
            self._op()
            .with_post(lambda o, p, rng, ctx: (log.append(1), o)[1])
            .with_post(lambda o, p, rng, ctx: (log.append(2), o)[1])
        )
        rng = np.random.default_rng(0)
        parents = self._parents(rng)
        offspring = op.crossover(parents, rng=rng)
        op.post_crossover(offspring, parents, rng)
        assert log == [1, 2]

    def test_with_post_does_not_mutate_original(self):
        op = self._op()
        _ = op.with_post(lambda o, p, rng, ctx: np.zeros_like(o))
        rng = np.random.default_rng(0)
        parents = self._parents(rng)
        offspring = op.crossover(parents, rng=rng)
        result = op.post_crossover(offspring.copy(), parents, rng)
        np.testing.assert_array_equal(result, offspring)


# ---------------------------------------------------------------------------
# Mutation lifecycle hooks
# ---------------------------------------------------------------------------


class TestMutationHooks:
    def _op(self):
        return MutationPolynomial(prob_var=1.0, eta=20.0)

    def _individual(self, rng=None):
        rng = rng if rng is not None else np.random.default_rng(0)
        return rng.uniform(-3.0, 3.0, size=DIM)

    def _range(self):
        return np.full(DIM, -5.0), np.full(DIM, 5.0)

    def test_post_mutation_default_is_noop(self):
        op = self._op()
        rng = np.random.default_rng(0)
        p = self._individual(rng)
        mutated = op.mutate(p, self._range(), rng=rng)
        result = op.post_mutation(mutated.copy(), self._range(), rng)
        np.testing.assert_array_equal(result, mutated)

    def test_with_post_transforms_individual(self):
        op = self._op().with_post(lambda o, mr, rng, ctx: np.zeros_like(o))
        rng = np.random.default_rng(0)
        p = self._individual(rng)
        mutated = op.mutate(p, self._range(), rng=rng)
        result = op.post_mutation(mutated, self._range(), rng)
        np.testing.assert_array_equal(result, np.zeros(DIM))

    def test_with_post_fn_receives_correct_args(self):
        received = {}

        def hook(offspring, mutate_range, rng, ctx):
            received["shape"] = offspring.shape
            received["ctx"] = ctx
            return offspring

        ctx = _make_ctx()
        op = self._op().with_post(hook)
        rng = np.random.default_rng(0)
        p = self._individual(rng)
        mutated = op.mutate(p, self._range(), rng=rng)
        op.post_mutation(mutated, self._range(), rng, ctx)
        assert received["shape"] == (DIM,)
        assert received["ctx"] is ctx

    def test_with_post_chains_in_order(self):
        log = []
        op = (
            self._op()
            .with_post(lambda o, mr, rng, ctx: (log.append(1), o)[1])
            .with_post(lambda o, mr, rng, ctx: (log.append(2), o)[1])
        )
        rng = np.random.default_rng(0)
        p = self._individual(rng)
        mutated = op.mutate(p, self._range(), rng=rng)
        op.post_mutation(mutated, self._range(), rng)
        assert log == [1, 2]

    def test_with_post_does_not_mutate_original(self):
        op = self._op()
        _ = op.with_post(lambda o, mr, rng, ctx: np.zeros_like(o))
        rng = np.random.default_rng(0)
        p = self._individual(rng)
        mutated = op.mutate(p, self._range(), rng=rng)
        result = op.post_mutation(mutated.copy(), self._range(), rng)
        np.testing.assert_array_equal(result, mutated)


# ---------------------------------------------------------------------------
# GA hook invocation (integration)
# ---------------------------------------------------------------------------


class _DispatchOnlyProvider:
    def dispatch(self, event):
        pass


class TestGAHookInvocation:
    """Verify that GA.ask() calls post_crossover and post_mutation hooks."""

    def _make_ctx(self):
        return _make_ctx()

    def test_post_crossover_called_during_ask(self):
        call_count = [0]

        def hook(offspring, parents, rng, ctx):
            call_count[0] += 1
            return offspring

        crossover = CrossoverBLXAlpha(prob=1.0, alpha=0.4).with_post(hook)
        ga = GA(
            crossover=crossover,
            mutation=MutationUniform(prob_var=0.0),
            parent_selection=SequentialSelection(),
            survivor_selection=TruncationSelection(),
        )
        provider = _DispatchOnlyProvider()
        ctx = _make_ctx()
        algorithm_ask(ga, ctx, provider)
        assert call_count[0] > 0

    def test_post_mutation_called_once_per_offspring(self):
        call_count = [0]

        def hook(offspring, mutate_range, rng, ctx):
            call_count[0] += 1
            return offspring

        mutation = MutationUniform(prob_var=0.0).with_post(hook)
        ga = GA(
            crossover=CrossoverBLXAlpha(prob=0.9, alpha=0.4),
            mutation=mutation,
            parent_selection=SequentialSelection(),
            survivor_selection=TruncationSelection(),
        )
        provider = _DispatchOnlyProvider()
        ctx = _make_ctx()
        candidates = algorithm_ask(ga, ctx, provider)
        assert call_count[0] == len(candidates)


# ---------------------------------------------------------------------------
# CrossoverIntegerSBX
# ---------------------------------------------------------------------------


class TestCrossoverIntegerSBX:
    _rng = np.random.default_rng(0)
    _parents = np.array([[1.0, 3.0, 5.0], [3.0, 7.0, 9.0]])

    def test_output_shape(self):
        op = CrossoverIntegerSBX(prob=1.0, eta=2.0)
        c = op.crossover(self._parents, rng=np.random.default_rng(0))
        assert c.shape == (2, 3)

    def test_offspring_are_integers(self):
        op = CrossoverIntegerSBX(prob=1.0, eta=2.0)
        rng = np.random.default_rng(42)
        for _ in range(20):
            c = op.crossover(self._parents, rng=rng)
            assert np.all(c == np.round(c)), "offspring must be integers"

    def test_crossover_rate_attribute(self):
        op = CrossoverIntegerSBX(prob=0.8, eta=5.0)
        assert op.prob == 0.8

    def test_n_children(self):
        assert CrossoverIntegerSBX(prob=1.0, eta=2.0).n_children == 2

    def test_determinism(self):
        op = CrossoverIntegerSBX(prob=1.0, eta=2.0)
        c1 = op.crossover(self._parents, rng=np.random.default_rng(7))
        c2 = op.crossover(self._parents, rng=np.random.default_rng(7))
        np.testing.assert_array_equal(c1, c2)

    def test_crossover_batch_output_shape(self):
        op = CrossoverIntegerSBX(prob=1.0, eta=2.0)
        rng = np.random.default_rng(0)
        n_pair = 5
        parents_batch = rng.integers(0, 10, size=(n_pair, 2, 3)).astype(float)
        c = op.crossover_batch(parents_batch, rng=rng)
        assert c is not None
        assert c.shape == (n_pair, 2, 3)

    def test_crossover_batch_matches_single_at_n_pair_one(self):
        """Unbounded branch: draws are u, then do_cross (2 phases)."""
        op = CrossoverIntegerSBX(prob=1.0, eta=2.0)
        parents_batch = self._parents[np.newaxis, :, :]
        c_batch = op.crossover_batch(parents_batch, rng=np.random.default_rng(7))
        assert c_batch is not None
        c_single = op.crossover(self._parents, rng=np.random.default_rng(7))
        np.testing.assert_array_equal(c_batch[0], c_single)

    def test_crossover_batch_matches_single_at_n_pair_one_bounded(self):
        """Bounded branch: draws are u, then swap, then do_cross (3 phases)
        -- the branch GA actually exercises, since GA always supplies bounds."""
        op = CrossoverIntegerSBX(prob=1.0, eta=20.0, prob_var=1.0)
        lb = np.zeros(3)
        ub = np.full(3, 9.0)
        parents_batch = self._parents[np.newaxis, :, :]
        c_batch = op.crossover_batch(
            parents_batch, (lb, ub), rng=np.random.default_rng(7)
        )
        assert c_batch is not None
        c_single = op.crossover(self._parents, (lb, ub), rng=np.random.default_rng(7))
        np.testing.assert_array_equal(c_batch[0], c_single)

    def test_crossover_batch_deterministic_with_seed(self):
        op = CrossoverIntegerSBX(prob=1.0, eta=2.0)
        rng = np.random.default_rng(0)
        n_pair = 5
        parents_batch = rng.integers(0, 10, size=(n_pair, 2, 3)).astype(float)
        c1 = op.crossover_batch(parents_batch, rng=np.random.default_rng(42))
        c2 = op.crossover_batch(parents_batch, rng=np.random.default_rng(42))
        np.testing.assert_array_equal(c1, c2)

    def test_crossover_batch_offspring_are_integers(self):
        op = CrossoverIntegerSBX(prob=1.0, eta=2.0)
        rng = np.random.default_rng(42)
        n_pair = 8
        for _ in range(20):
            parents_batch = rng.integers(0, 10, size=(n_pair, 2, 3)).astype(float)
            c = op.crossover_batch(parents_batch, rng=rng)
            assert c is not None
            assert np.all(c == np.round(c)), "offspring must be integers"

    def test_crossover_batch_respects_bounds(self):
        op = CrossoverIntegerSBX(prob=1.0, eta=20.0, prob_var=1.0)
        lb = np.zeros(3)
        ub = np.full(3, 9.0)
        rng = np.random.default_rng(0)
        n_pair = 8
        for _ in range(20):
            parents_batch = rng.integers(0, 10, size=(n_pair, 2, 3)).astype(float)
            c = op.crossover_batch(parents_batch, (lb, ub), rng=rng)
            assert c is not None
            assert np.all(c >= lb) and np.all(c <= ub)


# ---------------------------------------------------------------------------
# CrossoverCategorical
# ---------------------------------------------------------------------------


class TestCrossoverCategorical:
    _parents = np.array([[0.0, 2.0, 1.0], [2.0, 0.0, 2.0]])

    def test_output_shape(self):
        op = CrossoverCategorical(prob=1.0)
        c = op.crossover(self._parents, rng=np.random.default_rng(0))
        assert c.shape == (2, 3)

    def test_offspring_values_from_parents(self):
        op = CrossoverCategorical(prob=1.0)
        rng = np.random.default_rng(0)
        for _ in range(50):
            c = op.crossover(self._parents, rng=rng)
            for dim in range(self._parents.shape[1]):
                valid = {self._parents[0, dim], self._parents[1, dim]}
                assert c[0, dim] in valid
                assert c[1, dim] in valid

    def test_complementary_swap(self):
        op = CrossoverCategorical(prob=1.0)
        rng = np.random.default_rng(0)
        for _ in range(50):
            c = op.crossover(self._parents, rng=rng)
            for dim in range(self._parents.shape[1]):
                # if c1 took p2's value, c2 must have taken p1's
                p_sum = self._parents[0, dim] + self._parents[1, dim]
                assert c[0, dim] + c[1, dim] == p_sum

    def test_crossover_rate_attribute(self):
        assert CrossoverCategorical(prob=0.9).prob == 0.9

    def test_n_children(self):
        assert CrossoverCategorical(prob=1.0).n_children == 2

    def test_determinism(self):
        op = CrossoverCategorical(prob=1.0)
        c1 = op.crossover(self._parents, rng=np.random.default_rng(3))
        c2 = op.crossover(self._parents, rng=np.random.default_rng(3))
        np.testing.assert_array_equal(c1, c2)

    def test_crossover_batch_output_shape(self):
        op = CrossoverCategorical(prob=1.0)
        rng = np.random.default_rng(0)
        n_pair = 5
        parents_batch = np.tile(self._parents, (n_pair, 1, 1))
        c = op.crossover_batch(parents_batch, rng=rng)
        assert c is not None
        assert c.shape == (n_pair, 2, 3)

    def test_crossover_batch_matches_single_at_n_pair_one(self):
        op = CrossoverCategorical(prob=1.0)
        parents_batch = self._parents[np.newaxis, :, :]
        c_batch = op.crossover_batch(parents_batch, rng=np.random.default_rng(3))
        assert c_batch is not None
        c_single = op.crossover(self._parents, rng=np.random.default_rng(3))
        np.testing.assert_array_equal(c_batch[0], c_single)

    def test_crossover_batch_deterministic_with_seed(self):
        op = CrossoverCategorical(prob=1.0)
        n_pair = 5
        parents_batch = np.tile(self._parents, (n_pair, 1, 1))
        c1 = op.crossover_batch(parents_batch, rng=np.random.default_rng(3))
        c2 = op.crossover_batch(parents_batch, rng=np.random.default_rng(3))
        np.testing.assert_array_equal(c1, c2)

    def test_crossover_batch_complementary_swap(self):
        op = CrossoverCategorical(prob=1.0)
        rng = np.random.default_rng(0)
        n_pair = 5
        parents_batch = np.tile(self._parents, (n_pair, 1, 1))
        for _ in range(50):
            c = op.crossover_batch(parents_batch, rng=rng)
            assert c is not None
            # if c1 took p2's value, c2 must have taken p1's, for every row
            p_sum = parents_batch[:, 0, :] + parents_batch[:, 1, :]
            np.testing.assert_array_equal(c[:, 0, :] + c[:, 1, :], p_sum)


# ---------------------------------------------------------------------------
# MutationIntegerUniform
# ---------------------------------------------------------------------------


class TestMutationIntegerUniform:
    _lb = np.array([0.0, 1.0, 3.0])
    _ub = np.array([5.0, 4.0, 8.0])

    def test_output_shape(self):
        op = MutationIntegerUniform(prob_var=1.0)
        p = np.array([2.0, 2.0, 5.0])
        c = op.mutate(p, (self._lb, self._ub), rng=np.random.default_rng(0))
        assert c.shape == (3,)

    def test_offspring_are_integers(self):
        op = MutationIntegerUniform(prob_var=1.0)
        rng = np.random.default_rng(0)
        p = np.array([2.0, 2.0, 5.0])
        for _ in range(30):
            c = op.mutate(p, (self._lb, self._ub), rng=rng)
            assert np.all(c == np.round(c))

    def test_values_within_bounds(self):
        op = MutationIntegerUniform(prob_var=1.0)
        rng = np.random.default_rng(1)
        p = np.array([2.0, 2.0, 5.0])
        for _ in range(50):
            c = op.mutate(p, (self._lb, self._ub), rng=rng)
            assert np.all(c >= self._lb)
            assert np.all(c <= self._ub)

    def test_zero_rate_unchanged(self):
        op = MutationIntegerUniform(prob_var=0.0)
        p = np.array([2.0, 3.0, 6.0])
        c = op.mutate(p, (self._lb, self._ub), rng=np.random.default_rng(0))
        np.testing.assert_array_equal(c, p)

    def test_determinism(self):
        op = MutationIntegerUniform(prob_var=0.5)
        p = np.array([2.0, 2.0, 5.0])
        c1 = op.mutate(p, (self._lb, self._ub), rng=np.random.default_rng(5))
        c2 = op.mutate(p, (self._lb, self._ub), rng=np.random.default_rng(5))
        np.testing.assert_array_equal(c1, c2)


# ---------------------------------------------------------------------------
# MutationCategorical
# ---------------------------------------------------------------------------


class TestMutationCategorical:
    # 3 categorical dims with 3, 2, 4 categories respectively
    _lb = np.array([0.0, 0.0, 0.0])
    _ub = np.array([2.0, 1.0, 3.0])

    def test_output_shape(self):
        op = MutationCategorical(prob_var=1.0)
        p = np.array([1.0, 0.0, 2.0])
        c = op.mutate(p, (self._lb, self._ub), rng=np.random.default_rng(0))
        assert c.shape == (3,)

    def test_offspring_are_valid_indices(self):
        op = MutationCategorical(prob_var=1.0)
        rng = np.random.default_rng(0)
        p = np.array([1.0, 0.0, 2.0])
        for _ in range(50):
            c = op.mutate(p, (self._lb, self._ub), rng=rng)
            assert np.all(c >= self._lb)
            assert np.all(c <= self._ub)
            assert np.all(c == np.round(c))

    def test_uniform_distribution(self):
        op = MutationCategorical(prob_var=1.0)
        rng = np.random.default_rng(0)
        p = np.array([0.0])
        lb = np.array([0.0])
        ub = np.array([3.0])  # 4 categories
        counts = np.zeros(4)
        for _ in range(4000):
            c = op.mutate(p, (lb, ub), rng=rng)
            counts[int(c[0])] += 1
        # Each category should appear roughly 25% of the time
        assert np.all(counts > 800), f"distribution not uniform: {counts}"

    def test_zero_rate_unchanged(self):
        op = MutationCategorical(prob_var=0.0)
        p = np.array([1.0, 0.0, 2.0])
        c = op.mutate(p, (self._lb, self._ub), rng=np.random.default_rng(0))
        np.testing.assert_array_equal(c, p)

    def test_determinism(self):
        op = MutationCategorical(prob_var=0.5)
        p = np.array([1.0, 0.0, 2.0])
        c1 = op.mutate(p, (self._lb, self._ub), rng=np.random.default_rng(9))
        c2 = op.mutate(p, (self._lb, self._ub), rng=np.random.default_rng(9))
        np.testing.assert_array_equal(c1, c2)


# ---------------------------------------------------------------------------
# CrossoverSBX: bounded variant and prob_var
# ---------------------------------------------------------------------------


class TestCrossoverSBXBounded:
    def _parents(self):
        return np.array([[0.1, 0.2, 0.5], [0.8, 0.9, 0.6]])

    def _bounds(self):
        lb = np.zeros(3)
        ub = np.ones(3)
        return lb, ub

    def test_bounded_offspring_within_bounds(self):
        op = CrossoverSBX(prob=1.0, eta=20.0, prob_var=1.0)
        lb, ub = self._bounds()
        rng = np.random.default_rng(0)
        for _ in range(50):
            c = op.crossover(self._parents(), (lb, ub), rng=rng)
            assert np.all(c >= lb) and np.all(c <= ub)

    def test_unbounded_fallback_when_bounds_none(self):
        op = CrossoverSBX(prob=1.0, eta=2.0, prob_var=1.0)
        rng1 = np.random.default_rng(7)
        rng2 = np.random.default_rng(7)
        p = self._parents()
        c_none = op.crossover(p, None, rng=rng1)
        c_default = op.crossover(p, rng=rng2)
        np.testing.assert_array_equal(c_none, c_default)

    def test_prob_var_zero_returns_parents(self):
        op = CrossoverSBX(prob=1.0, eta=20.0, prob_var=0.0)
        lb, ub = self._bounds()
        p = self._parents()
        c = op.crossover(p, (lb, ub), rng=np.random.default_rng(0))
        np.testing.assert_array_equal(c[0], p[0])
        np.testing.assert_array_equal(c[1], p[1])

    def test_symmetric_margins_preserve_center(self):
        # Equal margins to lb/ub make beta_q identical for both children,
        # so the offspring center matches the parent center (special case
        # of the asymmetric formula, not a general guarantee).
        op = CrossoverSBX(prob=1.0, eta=20.0, prob_var=1.0)
        rng = np.random.default_rng(3)
        for _ in range(30):
            p = rng.uniform(0.4, 0.6, size=(2, 3))
            margin = rng.uniform(0.05, 0.3, size=3)
            y1 = np.minimum(p[0], p[1])
            y2 = np.maximum(p[0], p[1])
            lb = y1 - margin
            ub = y2 + margin
            c = op.crossover(p, (lb, ub), rng=rng)
            mid_p = 0.5 * (p[0] + p[1])
            mid_c = 0.5 * (c[0] + c[1])
            np.testing.assert_allclose(mid_c, mid_p, atol=1e-9)

    def test_asymmetric_bounds_produce_unequal_offsets(self):
        # With lb close to the parents and ub far away, beta_q must differ
        # between c1 (constrained by lb) and c2 (constrained by ub), so the
        # offspring are not symmetric about the parent center.
        op = CrossoverSBX(prob=1.0, eta=20.0, prob_var=1.0)
        lb = np.array([0.0])
        ub = np.array([1.0])
        p = np.array([[0.01], [0.5]])
        rng = np.random.default_rng(3)
        max_gap = 0.0
        for _ in range(30):
            c = op.crossover(p, (lb, ub), rng=rng)
            mid = 0.5 * (p[0] + p[1])
            offset1 = mid - c[0]
            offset2 = c[1] - mid
            max_gap = max(max_gap, np.abs(offset1 - offset2).max())
        assert max_gap > 1e-6

    def test_identical_parents_unchanged(self):
        op = CrossoverSBX(prob=1.0, eta=20.0, prob_var=1.0)
        lb, ub = self._bounds()
        p = np.array([[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]])
        c = op.crossover(p, (lb, ub), rng=np.random.default_rng(0))
        np.testing.assert_array_equal(c[0], p[0])
        np.testing.assert_array_equal(c[1], p[1])


# ---------------------------------------------------------------------------
# Mutation: individual-level prob gate
# ---------------------------------------------------------------------------


class TestMutationProbGate:
    def _range(self):
        lb = np.zeros(5)
        ub = np.ones(5)
        return lb, ub

    def test_prob_zero_returns_parent_unchanged(self):
        op = MutationPolynomial(prob=0.0, eta=20.0, prob_var=1.0)
        rng = np.random.default_rng(0)
        p = rng.uniform(0.1, 0.9, size=5)
        for _ in range(20):
            c = op.mutate(p, self._range(), rng=rng)
            np.testing.assert_array_equal(c, p)

    def test_prob_zero_uniform(self):
        op = MutationUniform(prob=0.0, prob_var=1.0)
        rng = np.random.default_rng(0)
        p = rng.uniform(0.1, 0.9, size=5)
        for _ in range(20):
            c = op.mutate(p, self._range(), rng=rng)
            np.testing.assert_array_equal(c, p)

    def test_prob_var_none_uses_adaptive_default(self):
        dim = 10
        op = MutationPolynomial(prob=1.0, eta=20.0, prob_var=None)
        lb = np.zeros(dim)
        ub = np.ones(dim)
        p = np.full(dim, 0.5)
        changed = np.zeros(dim)
        rng = np.random.default_rng(1)
        n_trials = 500
        for _ in range(n_trials):
            c = op.mutate(p, (lb, ub), rng=rng)
            changed += (c != p).astype(float)
        expected_rate = min(0.5, 1.0 / dim)
        observed_rate = changed / n_trials
        np.testing.assert_allclose(observed_rate, expected_rate, atol=0.05)


# ---------------------------------------------------------------------------
# MutationUniform.mutate_batch
#
# NOTE: unlike every Crossover.crossover_batch override, mutate_batch's
# output is NOT expected -- and cannot be expected, by construction -- to be
# bit-identical to a loop calling mutate() once per candidate, for any batch
# size (see the "Notes" section of MutationUniform.mutate_batch's
# docstring: the scalar loop interleaves a data-dependent number of draws
# per dimension, the vectorized version always draws a full (k, dim) gate
# array and a full (k, dim) replacement array). None of the tests below
# assert or rely on any such equivalence; they only check the
# statistical/distributional semantics and the exact pass-through of
# ungated rows/dimensions.
# ---------------------------------------------------------------------------


class TestMutationUniformBatch:
    def _range(self, dim):
        lb = np.full(dim, -5.0)
        ub = np.full(dim, 5.0)
        return lb, ub

    def test_output_shape(self):
        op = MutationUniform(prob=1.0, prob_var=0.5)
        rng = np.random.default_rng(0)
        n, dim = 7, DIM
        candidates_batch = rng.uniform(-2.0, 2.0, size=(n, dim))
        result = op.mutate_batch(candidates_batch, self._range(dim), rng=rng)
        assert result is not None
        assert result.shape == (n, dim)

    def test_prob_zero_returns_input_unchanged(self):
        op = MutationUniform(prob=0.0, prob_var=1.0)
        rng = np.random.default_rng(0)
        n, dim = 6, DIM
        candidates_batch = rng.uniform(-2.0, 2.0, size=(n, dim))
        result = op.mutate_batch(candidates_batch, self._range(dim), rng=rng)
        np.testing.assert_array_equal(result, candidates_batch)

    def test_prob_one_prob_var_one_replaces_every_value(self):
        # Input values sit far outside [lb, ub], so a continuous Uniform(lb,
        # ub) replacement draw can never coincidentally equal the original
        # input -- "every value differs" is guaranteed by construction, not
        # by seed luck. lb/ub are distinct, non-overlapping per-dimension
        # bands (rather than the same [-5, 5] for every column, as
        # self._range() would give) so that the bounds check pins each
        # column to its own band -- this would fail loudly under a
        # transposed/mis-broadcast rng.uniform(lb, ub, size=(k, dim)) call,
        # which a same-band-per-column check could not detect.
        op = MutationUniform(prob=1.0, prob_var=1.0)
        rng = np.random.default_rng(0)
        n, dim = 6, DIM
        lb = np.arange(dim) * 10.0
        ub = lb + 1.0
        candidates_batch = np.full((n, dim), 100.0)
        result = op.mutate_batch(candidates_batch, (lb, ub), rng=rng)
        assert result is not None
        assert np.all(result != candidates_batch)
        assert np.all(result >= lb) and np.all(result <= ub)

    def test_prob_var_fractional_matches_expected_rate(self):
        dim = 6
        n = 20
        n_trials = 300
        prob_var = 0.4
        op = MutationUniform(prob=1.0, prob_var=prob_var)
        lb, ub = self._range(dim)
        candidates_batch = np.full((n, dim), 100.0)
        rng = np.random.default_rng(2)
        changed = np.zeros((n, dim))
        for _ in range(n_trials):
            result = op.mutate_batch(candidates_batch, (lb, ub), rng=rng)
            changed += result != candidates_batch
        observed_rate = changed.sum() / (n * dim * n_trials)
        np.testing.assert_allclose(observed_rate, prob_var, atol=0.05)

    def test_fractional_prob_gates_some_rows_not_others(self):
        # prob_var=1.0 makes "changed" exactly track the row-level gate (no
        # coincidental per-dimension pass-through to muddy the signal), and
        # the out-of-bounds input value again rules out coincidental
        # equality for gated rows.
        op = MutationUniform(prob=0.5, prob_var=1.0)
        rng = np.random.default_rng(3)
        n, dim = 20, DIM
        lb, ub = self._range(dim)
        candidates_batch = np.full((n, dim), 100.0)
        result = op.mutate_batch(candidates_batch, (lb, ub), rng=rng)
        assert result is not None
        row_unchanged = (result == candidates_batch).all(axis=1)
        row_changed = (result != candidates_batch).any(axis=1)
        assert row_unchanged.any()
        assert row_changed.any()

    def test_determinism_same_seed_same_output(self):
        op = MutationUniform(prob=0.7, prob_var=0.4)
        dim = DIM
        lb, ub = self._range(dim)
        candidates_batch = np.random.default_rng(0).uniform(-2.0, 2.0, size=(15, dim))
        result_a = op.mutate_batch(
            candidates_batch, (lb, ub), rng=np.random.default_rng(42)
        )
        result_b = op.mutate_batch(
            candidates_batch, (lb, ub), rng=np.random.default_rng(42)
        )
        np.testing.assert_array_equal(result_a, result_b)

    def test_overrides_mutate_batch(self):
        assert type(MutationUniform()).mutate_batch is not Mutation.mutate_batch


class TestMutationUniformBatchUnboundedDimensions:
    """Draw replacements only for dimensions selected for mutation.

    Unbounded dimensions must not be passed to ``rng.uniform`` unless their
    per-dimension gate selected them. This preserves scalar mutation semantics
    and allows unbounded dimensions when no replacement is needed.
    """

    def test_prob_var_zero_with_infinite_bounds_does_not_raise(self):
        op = MutationUniform(prob=1.0, prob_var=0.0)
        lb = np.array([-np.inf, -np.inf])
        ub = np.array([np.inf, np.inf])
        batch = np.array([[1.0, 2.0], [3.0, 4.0]])
        rng = np.random.default_rng(0)
        result = op.mutate_batch(batch, (lb, ub), rng=rng)
        np.testing.assert_array_equal(result, batch)

    def test_mixed_bounded_unbounded_dims_no_raise_bounded_dim_in_bounds(self):
        # seed=0 with this exact (n=5, dim=2, prob=1.0, prob_var=0.3)
        # configuration draws a var_gate that never selects the unbounded
        # column (1) but does select the bounded column (0) for rows 3-4 --
        # verified via a standalone reconstruction of the gate/var_gate draw
        # sequence, asserted below as the test's precondition so the test
        # cannot silently go vacuous if the draw order ever changes.
        lb = np.array([-5.0, -np.inf])
        ub = np.array([5.0, np.inf])
        n, dim = 5, 2
        prob, prob_var = 1.0, 0.3

        check_rng = np.random.default_rng(0)
        gate = check_rng.random(n) < prob
        var_gate = check_rng.random((int(gate.sum()), dim)) < prob_var
        assert not var_gate[:, 1].any(), "precondition: unbounded col never gated"
        assert var_gate[:, 0].any(), "precondition: bounded col gated at least once"

        op = MutationUniform(prob=prob, prob_var=prob_var)
        batch = np.full((n, dim), 100.0)
        rng = np.random.default_rng(0)
        result = op.mutate_batch(batch, (lb, ub), rng=rng)
        assert result is not None

        # Unbounded column is untouched (never gated in, and no crash).
        np.testing.assert_array_equal(result[:, 1], batch[:, 1])
        # Bounded column: gated positions land within [lb, ub]; ungated
        # positions are byte-identical to the input.
        changed = result[gate][var_gate[:, 0], 0]
        assert changed.size > 0
        assert np.all(changed >= lb[0]) and np.all(changed <= ub[0])
        unchanged_mask = ~var_gate[:, 0]
        np.testing.assert_array_equal(
            result[gate][unchanged_mask, 0], batch[gate][unchanged_mask, 0]
        )


# ---------------------------------------------------------------------------
# MutationPolynomial.mutate_batch
#
# NOTE: same caveat as TestMutationUniformBatch above -- mutate_batch's
# output is NOT expected to be bit-identical to a loop calling mutate() once
# per candidate, for any batch size (see the "Notes" section of
# MutationPolynomial.mutate_batch's docstring). None of the tests below
# assert or rely on any such equivalence. The formula-correctness test
# instead pins down mutate_batch's random draws with a scripted fake rng, so
# delta1/u are fully controlled and the resulting delta_q can be checked
# against an independently re-derived reference computation, without
# depending on mutate()'s and mutate_batch()'s RNG streams aligning.
# ---------------------------------------------------------------------------


class _ScriptedRNG:
    """Fake rng exposing ``.random(size)``, ``.normal(loc, scale, size)``, and
    ``.integers(low, high, size)``, each returning a fixed queue of pre-set
    arrays in call order.

    Used to fully control the sequence of random draws mutate_batch makes
    internally (gate, var_gate, u / noise / integer replacement) so a chosen
    set of values can be fed through the real formula deterministically.
    ``normal_values``/``integers_values`` are optional and only needed by
    callers whose ``mutate_batch`` calls ``rng.normal``/``rng.integers``
    (e.g. ``MutationGaussian``/``_MutationDiscreteUniform``);
    ``MutationPolynomial``'s formula only ever calls ``.random``, so it never
    touches either queue.
    """

    def __init__(self, values, normal_values=None, integers_values=None):
        self._values = list(values)
        self._normal_values = list(normal_values) if normal_values is not None else []
        self._integers_values = (
            list(integers_values) if integers_values is not None else []
        )

    def random(self, size=None):
        return self._values.pop(0)

    def normal(self, loc=0.0, scale=1.0, size=None):
        return self._normal_values.pop(0)

    def integers(self, low, high, size=None):
        return self._integers_values.pop(0)


def _reference_delta_q(
    delta1: np.ndarray, delta2: np.ndarray, u: np.ndarray, eta: float
):
    """Independent scalar-loop reimplementation of the polynomial mutation
    delta_q formula.

    Transcribed directly from ``MutationPolynomial.mutate()``'s inner loop
    (not from ``mutate_batch``'s vectorized derivation) to catch any
    transcription mistake in the vectorized version. Deliberately a plain
    Python loop over flattened arrays -- correctness over speed, since this
    is test-only reference code.
    """
    d1_flat = delta1.ravel()
    d2_flat = delta2.ravel()
    u_flat = u.ravel()
    out = np.empty_like(d1_flat, dtype=float)
    for i in range(d1_flat.shape[0]):
        d1 = d1_flat[i]
        d2 = d2_flat[i]
        uu = u_flat[i]
        if uu <= 0.5:
            dq = (2.0 * uu + (1.0 - 2.0 * uu) * (1.0 - d1) ** (eta + 1.0)) ** (
                1.0 / (eta + 1.0)
            ) - 1.0
        else:
            dq = 1.0 - (
                2.0 * (1.0 - uu) + 2.0 * (uu - 0.5) * (1.0 - d2) ** (eta + 1.0)
            ) ** (1.0 / (eta + 1.0))
        out[i] = dq
    return out.reshape(delta1.shape)


class TestMutationPolynomialBatch:
    def _range(self, dim):
        lb = np.full(dim, -5.0)
        ub = np.full(dim, 5.0)
        return lb, ub

    def test_formula_matches_independent_reference(self):
        # delta1 + delta2 == 1 always (both are derived from one point's
        # position between lb/ub), so gridding delta1 (via `sub`) and u
        # covers every reachable (delta1, delta2, u) combination without
        # asserting an invalid geometry.
        #
        # lb/ub deliberately use two columns with distinct, non-unit-width,
        # non-zero-origin bands (rather than a shared [-5, 5] or a [0, 1]
        # band) for two reasons: (a) [0, 1] would make delta1 == sub and
        # (ub - lb) == 1 numerically, silently passing even if the
        # implementation dropped the "* (ub - lb)" scale factor or swapped
        # `sub` for `delta1` in the final expression; (b) a shared band
        # across columns cannot catch a transposed/mis-broadcast
        # `(dim,)`-against-`(k, dim)` computation (same guard rationale as
        # MutationUniform.mutate_batch's test_prob_one_prob_var_one_replaces
        # _every_value).
        delta1_vals = np.linspace(0.01, 0.99, 20)
        u_vals = np.linspace(0.01, 0.99, 20)
        d1_grid, u_grid = np.meshgrid(delta1_vals, u_vals, indexing="ij")
        d1_flat = d1_grid.ravel()
        u_flat = u_grid.ravel()
        n = d1_flat.shape[0]
        dim = 2
        lb = np.array([-3.0, 2.0])
        ub = np.array([7.0, 2.5])
        sub = lb + d1_flat[:, None] * (ub - lb)
        u_arr = np.repeat(u_flat[:, None], dim, axis=1)

        for eta in (2.0, 20.0):
            op = MutationPolynomial(prob=1.0, eta=eta, prob_var=1.0)
            scripted = _ScriptedRNG(
                [
                    np.zeros(n),  # gate draw: 0 < prob=1.0 -> all rows pass
                    np.zeros((n, dim)),  # var_gate draw: all dims pass
                    u_arr,  # u draw: fully controlled, same per column
                ]
            )
            # _ScriptedRNG only duck-types Generator's `.random(size)`; cast
            # so `ty`/static type checkers don't flag the substitution.
            result = op.mutate_batch(
                sub, (lb, ub), rng=cast(np.random.Generator, scripted)
            )
            assert result is not None

            expected_delta_q = _reference_delta_q(
                np.repeat(d1_flat[:, None], dim, axis=1),
                np.repeat((1.0 - d1_flat)[:, None], dim, axis=1),
                u_arr,
                eta,
            )
            expected = np.clip(sub + expected_delta_q * (ub - lb), lb, ub)
            np.testing.assert_allclose(result, expected, rtol=1e-9, atol=1e-9)

    def test_var_gate_pass_through_exact(self):
        # Confirms np.where(var_gate, mutated, sub) leaves dimensions that
        # fail the per-dimension gate exactly as `sub` (no recomputation),
        # while gated dimensions match the expected formula output --
        # pinned via the same scripted-rng control as the formula test
        # above, rather than inferred from the statistical rate test.
        n, dim = 5, 4
        lb = np.array([-3.0, 2.0, 0.0, -10.0])
        ub = np.array([7.0, 2.5, 1.0, 10.0])
        rng_data = np.random.default_rng(7)
        delta1 = rng_data.uniform(0.05, 0.95, size=(n, dim))
        sub = lb + delta1 * (ub - lb)
        u_arr = rng_data.uniform(0.05, 0.95, size=(n, dim))
        # Checkerboard var_gate pattern with prob_var=0.5: threshold draw of
        # 0.0 passes (0.0 < 0.5), 0.9 fails (0.9 >= 0.5).
        var_draw = np.where((np.arange(n)[:, None] + np.arange(dim)) % 2 == 0, 0.0, 0.9)
        passed = var_draw < 0.5
        assert passed.any() and (~passed).any()

        op = MutationPolynomial(prob=1.0, eta=15.0, prob_var=0.5)
        scripted = _ScriptedRNG(
            [
                np.zeros(n),  # gate draw: all rows pass
                var_draw,  # var_gate draw: checkerboard
                u_arr,  # u draw: fully controlled
            ]
        )
        result = op.mutate_batch(sub, (lb, ub), rng=cast(np.random.Generator, scripted))
        assert result is not None

        expected_delta_q = _reference_delta_q(delta1, 1.0 - delta1, u_arr, 15.0)
        expected_mutated = np.clip(sub + expected_delta_q * (ub - lb), lb, ub)

        np.testing.assert_array_equal(result[~passed], sub[~passed])
        np.testing.assert_allclose(
            result[passed], expected_mutated[passed], rtol=1e-9, atol=1e-9
        )

    def test_output_shape(self):
        op = MutationPolynomial(prob=1.0, eta=20.0, prob_var=0.5)
        rng = np.random.default_rng(0)
        n, dim = 7, DIM
        candidates_batch = rng.uniform(-2.0, 2.0, size=(n, dim))
        result = op.mutate_batch(candidates_batch, self._range(dim), rng=rng)
        assert result is not None
        assert result.shape == (n, dim)

    def test_prob_zero_returns_input_unchanged(self):
        op = MutationPolynomial(prob=0.0, eta=20.0, prob_var=1.0)
        rng = np.random.default_rng(0)
        n, dim = 6, DIM
        candidates_batch = rng.uniform(-2.0, 2.0, size=(n, dim))
        result = op.mutate_batch(candidates_batch, self._range(dim), rng=rng)
        np.testing.assert_array_equal(result, candidates_batch)

    def test_respects_bounds(self):
        op = MutationPolynomial(prob=1.0, eta=20.0, prob_var=1.0)
        lb, ub = self._range(DIM)
        rng = np.random.default_rng(0)
        for _ in range(20):
            candidates_batch = rng.uniform(-4.0, 4.0, size=(10, DIM))
            result = op.mutate_batch(candidates_batch, (lb, ub), rng=rng)
            assert result is not None
            assert np.all(result >= lb) and np.all(result <= ub)

    def test_prob_var_fractional_matches_expected_rate(self):
        # var_gate=True dimensions change value with overwhelming
        # probability under a continuous delta_q, so "value changed from
        # input" is a reasonable proxy for "was gated" here.
        dim = 6
        n = 20
        n_trials = 300
        prob_var = 0.4
        op = MutationPolynomial(prob=1.0, eta=20.0, prob_var=prob_var)
        lb, ub = self._range(dim)
        rng = np.random.default_rng(2)
        changed = np.zeros((n, dim))
        for _ in range(n_trials):
            candidates_batch = rng.uniform(-2.0, 2.0, size=(n, dim))
            result = op.mutate_batch(candidates_batch, (lb, ub), rng=rng)
            changed += result != candidates_batch
        observed_rate = changed.sum() / (n * dim * n_trials)
        np.testing.assert_allclose(observed_rate, prob_var, atol=0.05)

    def test_fractional_prob_gates_some_rows_not_others(self):
        op = MutationPolynomial(prob=0.5, eta=20.0, prob_var=1.0)
        rng = np.random.default_rng(3)
        n, dim = 20, DIM
        lb, ub = self._range(dim)
        candidates_batch = rng.uniform(-2.0, 2.0, size=(n, dim))
        result = op.mutate_batch(candidates_batch, (lb, ub), rng=rng)
        assert result is not None
        row_unchanged = (result == candidates_batch).all(axis=1)
        row_changed = (result != candidates_batch).any(axis=1)
        assert row_unchanged.any()
        assert row_changed.any()

    def test_determinism_same_seed_same_output(self):
        op = MutationPolynomial(prob=0.7, eta=20.0, prob_var=0.4)
        dim = DIM
        lb, ub = self._range(dim)
        candidates_batch = np.random.default_rng(0).uniform(-2.0, 2.0, size=(15, dim))
        result_a = op.mutate_batch(
            candidates_batch, (lb, ub), rng=np.random.default_rng(42)
        )
        result_b = op.mutate_batch(
            candidates_batch, (lb, ub), rng=np.random.default_rng(42)
        )
        np.testing.assert_array_equal(result_a, result_b)

    def test_overrides_mutate_batch(self):
        assert (
            type(MutationPolynomial(eta=20.0)).mutate_batch is not Mutation.mutate_batch
        )


# ---------------------------------------------------------------------------
# MutationGaussian.mutate_batch
#
# NOTE: same caveat as TestMutationUniformBatch/TestMutationPolynomialBatch
# above -- mutate_batch's output is NOT expected to be bit-identical to a
# loop calling mutate() once per candidate, for any batch size (see the
# "Notes" section of MutationGaussian.mutate_batch's docstring). None of the
# tests below assert or rely on any such equivalence.
# ---------------------------------------------------------------------------


class TestMutationGaussianBatch:
    def _range(self, dim):
        lb = np.full(dim, -5.0)
        ub = np.full(dim, 5.0)
        return lb, ub

    def test_output_shape(self):
        op = MutationGaussian(prob=1.0, sigma=0.1, prob_var=0.5)
        rng = np.random.default_rng(0)
        n, dim = 7, DIM
        candidates_batch = rng.uniform(-2.0, 2.0, size=(n, dim))
        result = op.mutate_batch(candidates_batch, self._range(dim), rng=rng)
        assert result is not None
        assert result.shape == (n, dim)

    def test_prob_zero_returns_input_unchanged(self):
        op = MutationGaussian(prob=0.0, sigma=1.0, prob_var=1.0)
        rng = np.random.default_rng(0)
        n, dim = 6, DIM
        candidates_batch = rng.uniform(-2.0, 2.0, size=(n, dim))
        result = op.mutate_batch(candidates_batch, self._range(dim), rng=rng)
        np.testing.assert_array_equal(result, candidates_batch)

    def test_var_gate_pass_through_and_noise_exact(self):
        # Fully scripted rng: gate/var_gate/noise draws are all fixed, so
        # dimensions failing var_gate must come back exactly as `sub` (no
        # noise applied), and dimensions passing var_gate must come back as
        # exactly `sub + noise` for the scripted noise array. No branching
        # formula to verify here (unlike MutationPolynomial), so this is a
        # plain addition/gating check.
        n, dim = 5, 4
        lb = np.array([-3.0, 2.0, 0.0, -10.0])
        ub = np.array([7.0, 2.5, 1.0, 10.0])
        rng_data = np.random.default_rng(7)
        sub = rng_data.uniform(-2.0, 2.0, size=(n, dim))
        noise = rng_data.normal(0.0, 1.0, size=(n, dim))
        # Checkerboard var_gate pattern with prob_var=0.5: threshold draw of
        # 0.0 passes (0.0 < 0.5), 0.9 fails (0.9 >= 0.5).
        var_draw = np.where((np.arange(n)[:, None] + np.arange(dim)) % 2 == 0, 0.0, 0.9)
        passed = var_draw < 0.5
        assert passed.any() and (~passed).any()

        op = MutationGaussian(prob=1.0, sigma=1.0, prob_var=0.5)
        scripted = _ScriptedRNG(
            [
                np.zeros(n),  # gate draw: all rows pass
                var_draw,  # var_gate draw: checkerboard
            ],
            normal_values=[noise],  # noise draw: fully controlled
        )
        result = op.mutate_batch(sub, (lb, ub), rng=cast(np.random.Generator, scripted))
        assert result is not None

        expected_mutated = sub + noise
        np.testing.assert_array_equal(result[~passed], sub[~passed])
        np.testing.assert_array_equal(result[passed], expected_mutated[passed])

    def test_zero_sigma_no_change(self):
        # prob=1.0/prob_var=1.0 forces every element through the noise path,
        # so sigma=0.0 returning the input byte-identically is what pins
        # self.sigma as the scale actually passed to rng.normal (rather
        # than, say, a hardcoded scale=1.0).
        op = MutationGaussian(prob=1.0, sigma=0.0, prob_var=1.0)
        rng = np.random.default_rng(0)
        n, dim = 6, DIM
        candidates_batch = rng.uniform(-2.0, 2.0, size=(n, dim))
        result = op.mutate_batch(candidates_batch, self._range(dim), rng=rng)
        np.testing.assert_array_equal(result, candidates_batch)

    def test_no_bounds_clipping(self):
        # Deliberately large sigma and candidates near lb/ub: mutated values
        # must be able to legitimately land outside [lb, ub], pinning the
        # absence of clipping as a regression guard (mutate() itself never
        # clips Gaussian mutation output; mutate_batch must not "fix" this
        # by adding an np.clip that would be a silent behavior change).
        op = MutationGaussian(prob=1.0, sigma=100.0, prob_var=1.0)
        lb, ub = self._range(DIM)
        candidates_batch = np.tile(ub, (20, 1))
        rng = np.random.default_rng(0)
        result = op.mutate_batch(candidates_batch, (lb, ub), rng=rng)
        assert result is not None
        assert np.any(result > ub) or np.any(result < lb)

    def test_prob_var_fractional_matches_expected_rate(self):
        # var_gate=True dimensions change value with overwhelming
        # probability under continuous Gaussian noise (exactly 0.0 noise has
        # probability zero), so "value changed from input" is a reasonable
        # proxy for "was gated" here.
        dim = 6
        n = 20
        n_trials = 300
        prob_var = 0.4
        op = MutationGaussian(prob=1.0, sigma=1.0, prob_var=prob_var)
        lb, ub = self._range(dim)
        rng = np.random.default_rng(2)
        changed = np.zeros((n, dim))
        for _ in range(n_trials):
            candidates_batch = rng.uniform(-2.0, 2.0, size=(n, dim))
            result = op.mutate_batch(candidates_batch, (lb, ub), rng=rng)
            changed += result != candidates_batch
        observed_rate = changed.sum() / (n * dim * n_trials)
        np.testing.assert_allclose(observed_rate, prob_var, atol=0.05)

    def test_fractional_prob_gates_some_rows_not_others(self):
        op = MutationGaussian(prob=0.5, sigma=1.0, prob_var=1.0)
        rng = np.random.default_rng(3)
        n, dim = 20, DIM
        lb, ub = self._range(dim)
        candidates_batch = rng.uniform(-2.0, 2.0, size=(n, dim))
        result = op.mutate_batch(candidates_batch, (lb, ub), rng=rng)
        assert result is not None
        row_unchanged = (result == candidates_batch).all(axis=1)
        row_changed = (result != candidates_batch).any(axis=1)
        assert row_unchanged.any()
        assert row_changed.any()

    def test_determinism_same_seed_same_output(self):
        op = MutationGaussian(prob=0.7, sigma=0.5, prob_var=0.4)
        dim = DIM
        lb, ub = self._range(dim)
        candidates_batch = np.random.default_rng(0).uniform(-2.0, 2.0, size=(15, dim))
        result_a = op.mutate_batch(
            candidates_batch, (lb, ub), rng=np.random.default_rng(42)
        )
        result_b = op.mutate_batch(
            candidates_batch, (lb, ub), rng=np.random.default_rng(42)
        )
        np.testing.assert_array_equal(result_a, result_b)

    def test_overrides_mutate_batch(self):
        assert (
            type(MutationGaussian(sigma=1.0)).mutate_batch is not Mutation.mutate_batch
        )


# ---------------------------------------------------------------------------
# _MutationDiscreteUniform.mutate_batch (shared base for MutationIntegerUniform
# and MutationCategorical)
#
# NOTE: same caveat as the batch test classes above -- mutate_batch's output
# is NOT expected to be bit-identical to a loop calling mutate() once per
# candidate, for any batch size (see the "Notes" section of
# _MutationDiscreteUniform.mutate_batch's docstring). None of the tests below
# assert or rely on any such equivalence.
#
# mutate_batch is implemented exactly once, on _MutationDiscreteUniform
# itself, and inherited unchanged by both MutationIntegerUniform and
# MutationCategorical (neither subclass overrides it). Every test here is
# therefore parametrized over both concrete classes, to confirm the
# inheritance actually works end to end rather than merely that the shared
# base class's own logic is correct in isolation.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cls", [MutationIntegerUniform, MutationCategorical])
class TestMutationDiscreteUniformBatch:
    def _range(self, dim):
        # ub - lb == 9 (10 possible integer values per dimension) -- wide
        # enough that a coincidental "new == old" draw on a gated dimension
        # is rare and won't meaningfully bias the shape/bounds/pass-through
        # checks below.
        lb = np.zeros(dim)
        ub = np.full(dim, 9.0)
        return lb, ub

    def test_output_shape(self, cls):
        op = cls(prob=1.0, prob_var=0.5)
        rng = np.random.default_rng(0)
        n, dim = 7, DIM
        lb, ub = self._range(dim)
        candidates_batch = rng.integers(0, 10, size=(n, dim)).astype(float)
        result = op.mutate_batch(candidates_batch, (lb, ub), rng=rng)
        assert result is not None
        assert result.shape == (n, dim)

    def test_prob_zero_returns_input_unchanged(self, cls):
        op = cls(prob=0.0, prob_var=1.0)
        rng = np.random.default_rng(0)
        n, dim = 6, DIM
        lb, ub = self._range(dim)
        candidates_batch = rng.integers(0, 10, size=(n, dim)).astype(float)
        result = op.mutate_batch(candidates_batch, (lb, ub), rng=rng)
        np.testing.assert_array_equal(result, candidates_batch)

    def test_replaced_values_are_integers_within_bounds(self, cls):
        # lb/ub are distinct, non-overlapping per-dimension bands (rather
        # than a shared range for every column) so the bounds check pins
        # each column to its own band -- this would fail loudly under a
        # transposed/mis-broadcast `rng.integers(lb, ub + 1, size=(k, dim))`
        # call, which a same-band-per-column check could not detect (same
        # guard rationale as
        # TestMutationUniformBatch.test_prob_one_prob_var_one_replaces_every
        # _value). The non-integer input value (2.5, outside every band)
        # also makes "result is an integer" itself proof that every value
        # was actually replaced, rather than merely consistent with the
        # (already-integer) input being left untouched.
        op = cls(prob=1.0, prob_var=1.0)
        rng = np.random.default_rng(1)
        n, dim = 20, DIM
        lb = np.arange(dim) * 10.0
        ub = lb + 4.0
        candidates_batch = np.full((n, dim), 2.5)
        for _ in range(20):
            result = op.mutate_batch(candidates_batch, (lb, ub), rng=rng)
            assert result is not None
            assert np.all(result == np.round(result))
            assert np.all(result >= lb) and np.all(result <= ub)

    def test_var_gate_pass_through_and_replacement_exact(self, cls):
        # Fully scripted rng: gate/var_gate/integer-replacement draws are all
        # fixed, so dimensions failing var_gate must come back exactly as
        # `sub` (no replacement applied), and dimensions passing var_gate
        # must come back as exactly the scripted integer replacement (cast
        # to float).
        n, dim = 5, 4
        lb = np.zeros(dim)
        ub = np.full(dim, 9.0)
        rng_data = np.random.default_rng(7)
        sub = rng_data.integers(0, 10, size=(n, dim)).astype(float)
        replacement = rng_data.integers(0, 10, size=(n, dim)).astype(float)
        # Checkerboard var_gate pattern with prob_var=0.5: threshold draw of
        # 0.0 passes (0.0 < 0.5), 0.9 fails (0.9 >= 0.5).
        var_draw = np.where((np.arange(n)[:, None] + np.arange(dim)) % 2 == 0, 0.0, 0.9)
        passed = var_draw < 0.5
        assert passed.any() and (~passed).any()

        op = cls(prob=1.0, prob_var=0.5)
        scripted = _ScriptedRNG(
            [
                np.zeros(n),  # gate draw: all rows pass
                var_draw,  # var_gate draw: checkerboard
            ],
            integers_values=[replacement.astype(int)],  # replacement draw
        )
        result = op.mutate_batch(sub, (lb, ub), rng=cast(np.random.Generator, scripted))
        assert result is not None

        np.testing.assert_array_equal(result[~passed], sub[~passed])
        np.testing.assert_array_equal(result[passed], replacement[passed])

    def test_prob_var_fractional_matches_expected_rate(self, cls):
        # Use a wide integer range (100 possible values per dimension) so a
        # gated dimension coincidentally redrawing its own value has only a
        # 1% chance, keeping that bias well below the atol used to compare
        # the observed "changed" rate against prob_var.
        dim = 6
        n = 20
        n_trials = 300
        prob_var = 0.4
        op = cls(prob=1.0, prob_var=prob_var)
        lb = np.zeros(dim)
        ub = np.full(dim, 99.0)
        candidates_batch = np.zeros((n, dim))
        rng = np.random.default_rng(2)
        changed = np.zeros((n, dim))
        for _ in range(n_trials):
            result = op.mutate_batch(candidates_batch, (lb, ub), rng=rng)
            changed += result != candidates_batch
        observed_rate = changed.sum() / (n * dim * n_trials)
        np.testing.assert_allclose(observed_rate, prob_var, atol=0.05)

    def test_fractional_prob_gates_some_rows_not_others(self, cls):
        op = cls(prob=0.5, prob_var=1.0)
        rng = np.random.default_rng(3)
        n, dim = 20, DIM
        lb, ub = self._range(dim)
        candidates_batch = np.zeros((n, dim))
        result = op.mutate_batch(candidates_batch, (lb, ub), rng=rng)
        assert result is not None
        row_unchanged = (result == candidates_batch).all(axis=1)
        row_changed = (result != candidates_batch).any(axis=1)
        assert row_unchanged.any()
        assert row_changed.any()

    def test_determinism_same_seed_same_output(self, cls):
        op = cls(prob=0.7, prob_var=0.4)
        dim = DIM
        lb, ub = self._range(dim)
        candidates_batch = (
            np.random.default_rng(0).integers(0, 10, size=(15, dim)).astype(float)
        )
        result_a = op.mutate_batch(
            candidates_batch, (lb, ub), rng=np.random.default_rng(42)
        )
        result_b = op.mutate_batch(
            candidates_batch, (lb, ub), rng=np.random.default_rng(42)
        )
        np.testing.assert_array_equal(result_a, result_b)

    def test_overrides_mutate_batch(self, cls):
        assert type(cls()).mutate_batch is not Mutation.mutate_batch


# ---------------------------------------------------------------------------
# DuplicateElimination
# ---------------------------------------------------------------------------


class TestDuplicateElimination:
    def test_exact_duplicate_detected(self):
        from saealib.operators.dedup import DuplicateElimination

        de = DuplicateElimination(atol=1e-16, rtol=0.0)
        pop = np.array([[0.1, 0.2], [0.3, 0.4]])
        off = np.array([[0.1, 0.2], [0.5, 0.6]])
        mask = de.find_duplicates(off, pop)
        assert mask[0]
        assert not mask[1]

    def test_within_tolerance_detected(self):
        from saealib.operators.dedup import DuplicateElimination

        de = DuplicateElimination(atol=1e-6, rtol=0.0)
        pop = np.array([[0.1, 0.2]])
        off = np.array([[0.1 + 1e-7, 0.2]])
        mask = de.find_duplicates(off, pop)
        assert mask[0]

    def test_outside_tolerance_not_detected(self):
        from saealib.operators.dedup import DuplicateElimination

        de = DuplicateElimination(atol=1e-10, rtol=0.0)
        pop = np.array([[0.1, 0.2]])
        off = np.array([[0.1 + 1e-9, 0.2]])
        mask = de.find_duplicates(off, pop)
        assert not mask[0]

    def test_empty_offspring_returns_empty_mask(self):
        from saealib.operators.dedup import DuplicateElimination

        de = DuplicateElimination()
        pop = np.array([[0.1, 0.2]])
        off = np.empty((0, 2))
        mask = de.find_duplicates(off, pop)
        assert mask.shape == (0,)
        assert mask.dtype == bool

    def test_no_duplicates_all_false(self):
        from saealib.operators.dedup import DuplicateElimination

        de = DuplicateElimination()
        pop = np.array([[0.1, 0.2], [0.3, 0.4]])
        off = np.array([[0.5, 0.6], [0.7, 0.8]])
        mask = de.find_duplicates(off, pop)
        assert not mask.any()

    def test_ga_stores_duplicate_elimination_attribute(self):
        from saealib.operators.dedup import DuplicateElimination

        de = DuplicateElimination(atol=1e-10, rtol=0.0, max_retries=5)
        ga = GA(
            crossover=CrossoverBLXAlpha(prob=0.9, alpha=0.4),
            mutation=MutationPolynomial(prob=1.0, eta=20.0, prob_var=1.0),
            parent_selection=SequentialSelection(),
            survivor_selection=TruncationSelection(),
            duplicate_elimination=de,
        )
        assert ga.duplicate_elimination is de

    def test_ga_with_dedup_returns_correct_count(self):
        """GA with duplicate_elimination set produces expected offspring count."""
        from saealib.operators.dedup import DuplicateElimination

        de = DuplicateElimination()
        ga = GA(
            crossover=CrossoverBLXAlpha(prob=0.9, alpha=0.4),
            mutation=MutationPolynomial(prob=1.0, eta=20.0, prob_var=1.0),
            parent_selection=SequentialSelection(),
            survivor_selection=TruncationSelection(),
            duplicate_elimination=de,
        )
        provider = _DispatchOnlyProvider()
        ctx = _make_ctx(n_pop=10)
        candidates = algorithm_ask(ga, ctx, provider)
        assert len(candidates) == 10

    def test_ga_dedup_retries_when_offspring_are_copies(self):
        """With no-op crossover and no-op mutation, retries exhaust without crash."""
        from saealib.operators.dedup import DuplicateElimination

        # prob=0.0 crossover → offspring = parent copies (always duplicates)
        # prob_var=0.0 mutation → no variables mutated → still copies after each retry
        # dedup exhausts max_retries without fixing any, but must not raise
        de = DuplicateElimination(atol=1e-14, rtol=0.0, max_retries=3)
        ga = GA(
            crossover=CrossoverBLXAlpha(prob=0.0, alpha=0.4),
            mutation=MutationUniform(prob_var=0.0),
            parent_selection=SequentialSelection(),
            survivor_selection=TruncationSelection(),
            duplicate_elimination=de,
        )
        provider = _DispatchOnlyProvider()
        ctx = _make_ctx(n_pop=10)
        candidates = algorithm_ask(ga, ctx, provider)
        assert len(candidates) == 10

    def test_ga_none_dedup_preserves_behavior(self):
        """GA with duplicate_elimination=None must not raise."""
        ga = GA(
            crossover=CrossoverBLXAlpha(prob=0.9, alpha=0.4),
            mutation=MutationPolynomial(prob=1.0, eta=20.0, prob_var=1.0),
            parent_selection=SequentialSelection(),
            survivor_selection=TruncationSelection(),
            duplicate_elimination=None,
        )
        provider = _DispatchOnlyProvider()
        ctx = _make_ctx(n_pop=10)
        candidates = algorithm_ask(ga, ctx, provider)
        assert len(candidates) == 10


# ---------------------------------------------------------------------------
# GA + batched default crossover operators: end-to-end smoke test
# ---------------------------------------------------------------------------


class TestGABatchedCrossoverSmoke:
    def test_ga_with_sbx_runs_several_generations_on_sphere(self):
        """Run batched SBX through GA and keep generated candidates valid."""
        dim = 5
        lb = [-5.0] * dim
        ub = [5.0] * dim
        result = minimize(
            lambda x: np.sum(x**2),
            dim=dim,
            lb=lb,
            ub=ub,
            algorithm=GA(
                crossover=CrossoverSBX(prob=0.9, eta=15.0),
                mutation=MutationPolynomial(prob=1.0, eta=20.0, prob_var=0.2),
                parent_selection=TournamentSelection(2),
                survivor_selection=TruncationSelection(),
            ),
            surrogate="rbf",
            max_fe=150,
            pop_size=10,
            seed=0,
            verbose=False,
        )
        assert result.fe > 0
        assert np.isfinite(result.f).all()
        assert result.x is not None
        assert np.all(result.x >= lb) and np.all(result.x <= ub)
