"""
Tests for the acquisition module.

Tests cover:
- MeanPrediction: no weights (first objective), with weights (scalarization), shape
- MaxUncertainty: no weights (mean std), with weights, requires uncertainty
- ExpectedImprovement: basic EI formula, xi parameter, requires uncertainty
- LowerConfidenceBound: negated LCB, kappa parameter, requires uncertainty
- LowerConfidenceBound beta_schedule: round-index-based kappa, gp_ucb_beta_schedule
- ProbabilityOfFeasibility: P(g<=0), requires uncertainty
- CORSDistance: distance-constrained mean prediction, decision-count beta cycling
- AcquisitionFunction: abstract base class cannot be instantiated
- direction-aware minimize-space conversion for EI/LCB
"""

import math
import pickle
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from scipy.stats import norm

from saealib.acquisition import (
    AcquisitionFunction,
    CORSReference,
    ExpectedImprovement,
    LowerConfidenceBound,
    MaxUncertainty,
    MeanPrediction,
    ProbabilityOfFeasibility,
    ProductOfFeasibility,
    gp_ucb_beta_schedule,
)
from saealib.acquisition.mean import CORSDistance
from saealib.exceptions import ValidationError
from saealib.population import Archive, PopulationAttribute
from saealib.surrogate.prediction import SurrogatePrediction


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _pred(value, std=None):
    """Build a SurrogatePrediction from plain arrays."""
    m = np.asarray(value, dtype=float)
    s = np.asarray(std, dtype=float) if std is not None else None
    return SurrogatePrediction.objective(value=m, std=s)


def _archive(rows):
    """Build a single-objective Archive from objective-value rows."""
    attrs = [
        PopulationAttribute(name="x", dtype=np.float64, shape=(1,)),
        PopulationAttribute(name="f", dtype=np.float64, shape=(1,)),
    ]
    arc = Archive(attrs, init_capacity=len(rows) + 5)
    for i, f_row in enumerate(rows):
        arc.add(x=np.array([float(i)]), f=np.asarray(f_row, dtype=float))
    return arc


def _pred_x(value, x):
    """Build a SurrogatePrediction with candidate coordinates in the x field."""
    m = np.asarray(value, dtype=float)
    return SurrogatePrediction.objective(value=m, x=np.asarray(x, dtype=float))


def _archive_x(xs):
    """Build a 1-feature Archive containing exactly the given x coordinates."""
    attrs = [
        PopulationAttribute(name="x", dtype=np.float64, shape=(1,)),
        PopulationAttribute(name="f", dtype=np.float64, shape=(1,)),
    ]
    arc = Archive(attrs, init_capacity=len(xs) + 5)
    for x_val in xs:
        arc.add(x=np.array([float(x_val)]), f=np.array([0.0]))
    return arc


def _decision_ctx(decision_count: int) -> Any:
    """Minimal duck-typed OptimizationState stand-in exposing ctx.decision_count/rng."""
    return SimpleNamespace(decision_count=decision_count, rng=np.random.default_rng(0))


# ===========================================================================
# AcquisitionFunction (abstract base class) Tests
# ===========================================================================
class TestAcquisitionFunctionABC:
    """Tests for the AcquisitionFunction abstract base class."""

    def test_cannot_instantiate_directly(self) -> None:
        with pytest.raises(TypeError):
            AcquisitionFunction()  # type: ignore[abstract]  # intentional: testing abstract instantiation raises TypeError

    def test_concrete_subclass_must_implement_score(self) -> None:
        class IncompleteAF(AcquisitionFunction):
            pass

        with pytest.raises(TypeError):
            IncompleteAF()  # type: ignore[abstract]  # intentional: testing abstract instantiation raises TypeError


# ===========================================================================
# MeanPrediction Tests
# ===========================================================================
class TestMeanPrediction:
    """Tests for MeanPrediction acquisition function."""

    def test_no_weights_returns_first_objective(self) -> None:
        """Without weights, returns prediction.mean[:, 0]."""
        pred = _pred(value=[[1.0, 5.0], [2.0, 6.0], [3.0, 7.0]])
        scores = MeanPrediction().score(pred, reference=None)
        np.testing.assert_array_equal(scores, [1.0, 2.0, 3.0])

    def test_with_weights_returns_dot_product(self) -> None:
        """With weights, returns mean @ weights."""
        pred = _pred(value=[[1.0, 2.0], [3.0, 4.0]])
        weights = np.array([-1.0, -1.0])
        scores = MeanPrediction(weights=weights).score(pred, reference=None)
        np.testing.assert_array_almost_equal(scores, [-3.0, -7.0])

    def test_output_shape(self) -> None:
        rng = np.random.default_rng(0)
        pred = _pred(value=rng.random((5, 2)))
        scores = MeanPrediction().score(pred, reference=None)
        assert scores.shape == (5,)

    def test_output_shape_with_weights(self) -> None:
        rng = np.random.default_rng(0)
        pred = _pred(value=rng.random((7, 3)))
        scores = MeanPrediction(weights=np.array([-1.0, -1.0, -1.0])).score(
            pred, reference=None
        )
        assert scores.shape == (7,)

    def test_reference_not_used(self) -> None:
        """reference parameter is accepted but ignored."""
        pred = _pred(value=[[2.0]])
        s1 = MeanPrediction().score(pred, reference=None)
        s2 = MeanPrediction().score(pred, reference=np.array([99.0]))
        np.testing.assert_array_equal(s1, s2)

    def test_single_sample(self) -> None:
        pred = _pred(value=[[4.0]])
        scores = MeanPrediction().score(pred, reference=None)
        assert scores.shape == (1,)
        assert scores[0] == pytest.approx(4.0)

    def test_single_objective_with_weight(self) -> None:
        pred = _pred(value=[[3.0]])
        scores = MeanPrediction(weights=np.array([-1.0])).score(pred, reference=None)
        assert scores[0] == pytest.approx(-3.0)


# ===========================================================================
# MaxUncertainty Tests
# ===========================================================================
class TestMaxUncertainty:
    """Tests for MaxUncertainty acquisition function."""

    def test_no_weights_returns_mean_std(self) -> None:
        """Without weights, returns std.mean(axis=1)."""
        pred = _pred(
            value=[[0.0, 0.0], [0.0, 0.0]],
            std=[[1.0, 3.0], [2.0, 4.0]],
        )
        scores = MaxUncertainty().score(pred, reference=None)
        np.testing.assert_array_almost_equal(scores, [2.0, 3.0])

    def test_with_weights(self) -> None:
        pred = _pred(
            value=[[0.0, 0.0]],
            std=[[1.0, 2.0]],
        )
        scores = MaxUncertainty(weights=np.array([1.0, 0.0])).score(
            pred, reference=None
        )
        assert scores[0] == pytest.approx(1.0)

    def test_requires_uncertainty(self) -> None:
        pred = _pred(value=[[1.0, 2.0]])
        with pytest.raises(TypeError, match="uncertainty"):
            MaxUncertainty().score(pred, reference=None)

    def test_output_shape(self) -> None:
        pred = _pred(
            value=np.zeros((6, 2)),
            std=np.ones((6, 2)),
        )
        scores = MaxUncertainty().score(pred, reference=None)
        assert scores.shape == (6,)

    def test_single_objective(self) -> None:
        pred = _pred(value=[[0.0]], std=[[0.5]])
        scores = MaxUncertainty().score(pred, reference=None)
        assert scores[0] == pytest.approx(0.5)


# ===========================================================================
# ExpectedImprovement Tests
# ===========================================================================
class TestExpectedImprovement:
    """Tests for ExpectedImprovement acquisition function."""

    def test_requires_uncertainty(self) -> None:
        pred = _pred(value=[[1.0]])
        with pytest.raises(TypeError, match="uncertainty"):
            ExpectedImprovement().score(pred, reference=1.0)

    def test_output_shape(self) -> None:
        pred = _pred(value=np.zeros((5, 1)), std=np.ones((5, 1)))
        scores = ExpectedImprovement().score(pred, reference=0.0)
        assert scores.shape == (5,)

    def test_ei_nonnegative(self) -> None:
        """EI scores are always >= 0."""
        rng = np.random.default_rng(0)
        mean = rng.standard_normal((20, 1))
        std = np.abs(rng.standard_normal((20, 1))) + 0.1
        pred = _pred(value=mean, std=std)
        scores = ExpectedImprovement().score(pred, reference=0.0)
        assert np.all(scores >= 0.0)

    def test_ei_higher_for_better_candidate(self) -> None:
        """Candidate with lower mean scores higher EI."""
        # reference best = 2.0; mean=1.5 should score higher than mean=3.0
        pred = _pred(
            value=[[3.0], [1.5]],
            std=[[0.5], [0.5]],
        )
        scores = ExpectedImprovement(xi=0.0).score(pred, reference=2.0)
        assert scores[1] > scores[0]

    def test_ei_zero_for_worse_candidate(self) -> None:
        """EI is clipped to 0 when mu >> f_best."""
        pred = _pred(value=[[100.0]], std=[[0.001]])
        scores = ExpectedImprovement(xi=0.0).score(pred, reference=1.0)
        assert scores[0] == pytest.approx(0.0, abs=1e-6)

    def test_ei_formula_manual(self) -> None:
        """Verify EI matches the analytical formula."""
        mu, sigma, f_best, xi = 1.0, 0.5, 2.0, 0.01
        z = (f_best - mu - xi) / sigma
        expected = (f_best - mu - xi) * norm.cdf(z) + sigma * norm.pdf(z)
        pred = _pred(value=[[mu]], std=[[sigma]])
        scores = ExpectedImprovement(xi=xi).score(pred, reference=f_best)
        assert scores[0] == pytest.approx(max(expected, 0.0), rel=1e-5)

    def test_ei_obj_idx(self) -> None:
        """obj_idx selects which objective to compute EI for."""
        pred = _pred(
            value=[[10.0, 1.0]],
            std=[[0.5, 0.5]],
        )
        score_obj0 = ExpectedImprovement(obj_idx=0).score(
            pred, reference=np.array([2.0, 2.0])
        )
        score_obj1 = ExpectedImprovement(obj_idx=1).score(
            pred, reference=np.array([2.0, 2.0])
        )
        # obj1: mu=1.0 < f_best=2.0, positive EI; obj0: mu=10.0 >> f_best=2.0, ~0
        assert score_obj1[0] > score_obj0[0]

    def test_ei_xi_increases_exploration(self) -> None:
        """Higher xi shifts Z down, generally reducing score near f_best."""
        pred = _pred(value=[[1.9]], std=[[0.1]])
        score_low_xi = ExpectedImprovement(xi=0.0).score(pred, reference=2.0)
        score_high_xi = ExpectedImprovement(xi=1.0).score(pred, reference=2.0)
        # xi=1.0 makes z=(2.0-1.9-1.0)/0.1 negative → lower EI
        assert score_low_xi[0] >= score_high_xi[0]


# ===========================================================================
# LowerConfidenceBound Tests
# ===========================================================================
class TestLowerConfidenceBound:
    """Tests for LowerConfidenceBound acquisition function."""

    def test_requires_uncertainty(self) -> None:
        pred = _pred(value=[[1.0]])
        with pytest.raises(TypeError, match="uncertainty"):
            LowerConfidenceBound().score(pred, reference=None)

    def test_output_shape(self) -> None:
        pred = _pred(value=np.zeros((4, 1)), std=np.ones((4, 1)))
        scores = LowerConfidenceBound().score(pred, reference=None)
        assert scores.shape == (4,)

    def test_negated_lcb_formula(self) -> None:
        """score = -(mu - kappa * sigma) = -mu + kappa * sigma."""
        mu, sigma, kappa = 2.0, 0.5, 3.0
        pred = _pred(value=[[mu]], std=[[sigma]])
        scores = LowerConfidenceBound(kappa=kappa).score(pred, reference=None)
        expected = -(mu - kappa * sigma)
        assert scores[0] == pytest.approx(expected)

    def test_higher_score_for_lower_mean(self) -> None:
        """Candidate with lower predicted mean gets a higher score."""
        pred = _pred(value=[[1.0], [3.0]], std=[[0.1], [0.1]])
        scores = LowerConfidenceBound(kappa=0.0).score(pred, reference=None)
        # kappa=0 → score = -mu; lower mu → higher score
        assert scores[0] > scores[1]

    def test_higher_score_for_higher_uncertainty(self) -> None:
        """Candidate with higher std gets a higher score (exploration)."""
        pred = _pred(value=[[1.0], [1.0]], std=[[0.1], [1.0]])
        scores = LowerConfidenceBound(kappa=2.0).score(pred, reference=None)
        assert scores[1] > scores[0]

    def test_kappa_zero_equals_negative_mean(self) -> None:
        pred = _pred(value=[[2.5], [3.5]], std=[[0.5], [0.5]])
        scores = LowerConfidenceBound(kappa=0.0).score(pred, reference=None)
        np.testing.assert_array_almost_equal(scores, [-2.5, -3.5])

    def test_obj_idx(self) -> None:
        """obj_idx selects which objective to compute LCB for."""
        pred = _pred(
            value=[[1.0, 5.0]],
            std=[[0.1, 0.1]],
        )
        s0 = LowerConfidenceBound(kappa=0.0, obj_idx=0).score(pred, reference=None)
        s1 = LowerConfidenceBound(kappa=0.0, obj_idx=1).score(pred, reference=None)
        assert s0[0] == pytest.approx(-1.0)
        assert s1[0] == pytest.approx(-5.0)

    def test_compute_reference_returns_none(self) -> None:
        archive = _archive_x([0.0])
        assert LowerConfidenceBound().compute_reference(archive) is None

    def test_fixed_kappa_unaffected_by_repeated_calls(self) -> None:
        af = LowerConfidenceBound(kappa=2.0)
        pred = _pred(value=[[1.0]], std=[[0.1]])
        first = af.score(pred, reference=None)
        af.score(pred, reference=None)
        third = af.score(pred, reference=None)
        np.testing.assert_array_almost_equal(first, third)

    def test_prepare_resolves_kappa_from_decision_count(self) -> None:
        af = LowerConfidenceBound(beta_schedule=lambda t: 4.0)
        archive = _archive_x([0.0])
        assert af.prepare(archive, ctx=_decision_ctx(0)) == pytest.approx(
            2.0
        )  # sqrt(4.0)

    def test_prepare_raises_when_beta_schedule_returns_negative(self) -> None:
        af = LowerConfidenceBound(beta_schedule=lambda t: -1.0)
        archive = _archive_x([0.0])
        with pytest.raises(ValidationError, match="beta_schedule"):
            af.prepare(archive, ctx=_decision_ctx(0))

    def test_prepare_raises_when_beta_schedule_returns_nan(self) -> None:
        af = LowerConfidenceBound(beta_schedule=lambda t: float("nan"))
        archive = _archive_x([0.0])
        with pytest.raises(ValidationError, match="beta_schedule"):
            af.prepare(archive, ctx=_decision_ctx(0))

    def test_prepare_raises_when_beta_schedule_returns_inf(self) -> None:
        af = LowerConfidenceBound(beta_schedule=lambda t: float("inf"))
        archive = _archive_x([0.0])
        with pytest.raises(ValidationError, match="beta_schedule"):
            af.prepare(archive, ctx=_decision_ctx(0))

    def test_prepare_passes_decision_count_plus_one_as_t(self) -> None:
        seen: list[int] = []

        def schedule(t: int) -> float:
            seen.append(t)
            return 4.0

        af = LowerConfidenceBound(beta_schedule=schedule)
        archive = _archive_x([0.0])
        af.prepare(archive, ctx=_decision_ctx(0))
        af.prepare(archive, ctx=_decision_ctx(5))
        af.prepare(archive, ctx=_decision_ctx(5))
        assert seen == [1, 6, 6]

    def test_prepare_raises_without_ctx_when_beta_schedule_set(self) -> None:
        af = LowerConfidenceBound(beta_schedule=lambda t: 4.0)
        archive = _archive_x([0.0])
        with pytest.raises(ValidationError, match="beta_schedule"):
            af.prepare(archive, ctx=None)

    def test_prepare_delegates_to_compute_reference_without_beta_schedule(
        self,
    ) -> None:
        af = LowerConfidenceBound()
        archive = _archive_x([0.0])
        assert af.prepare(archive, ctx=None) is None
        assert af.prepare(archive, ctx=_decision_ctx(3)) is None

    def test_score_raises_when_beta_schedule_set_and_reference_is_none(self) -> None:
        af = LowerConfidenceBound(beta_schedule=lambda t: 4.0)
        pred = _pred(value=[[1.0]], std=[[0.1]])
        with pytest.raises(ValidationError, match="beta_schedule"):
            af.score(pred, reference=None)

    def test_score_uses_reference_as_kappa_when_beta_schedule_set(self) -> None:
        pred = _pred(value=[[2.0]], std=[[0.5]])
        scores_fixed = LowerConfidenceBound(kappa=2.0).score(pred, reference=None)
        scores_schedule = LowerConfidenceBound(
            kappa=999.0, beta_schedule=lambda t: 4.0
        ).score(pred, reference=2.0)
        assert scores_schedule[0] == pytest.approx(scores_fixed[0])

    def test_repeated_evaluate_with_unchanged_ctx_does_not_advance_t(self) -> None:
        seen: list[int] = []

        def schedule(t: int) -> float:
            seen.append(t)
            return 4.0

        af = LowerConfidenceBound(beta_schedule=schedule)
        pred = _pred(value=[[1.0]], std=[[0.1]])
        candidates_x = np.zeros((1, 1))
        archive = _archive_x([0.0])
        ctx = _decision_ctx(2)

        first = af.evaluate(candidates_x, pred, archive, ctx=ctx)
        second = af.evaluate(candidates_x, pred, archive, ctx=ctx)

        assert seen == [3, 3]
        assert first.scores is not None
        assert second.scores is not None
        np.testing.assert_array_almost_equal(first.scores, second.scores)


class TestGpUcbBetaSchedule:
    def test_matches_theorem1_formula(self) -> None:
        assert gp_ucb_beta_schedule(domain_size=1000, delta=0.1)(5) == pytest.approx(
            2.0 * math.log(1000 * 5**2 * math.pi**2 / (6.0 * 0.1))
        )

    def test_increasing_in_round_index(self) -> None:
        schedule = gp_ucb_beta_schedule(domain_size=100)
        assert schedule(1) < schedule(2) < schedule(10)

    def test_default_delta_is_point_one(self) -> None:
        assert gp_ucb_beta_schedule(domain_size=100)(3) == pytest.approx(
            gp_ucb_beta_schedule(domain_size=100, delta=0.1)(3)
        )

    def test_rejects_non_positive_domain_size(self) -> None:
        with pytest.raises(ValidationError, match="domain_size"):
            gp_ucb_beta_schedule(domain_size=0)
        with pytest.raises(ValidationError, match="domain_size"):
            gp_ucb_beta_schedule(domain_size=-1)

    def test_rejects_non_integer_domain_size(self) -> None:
        with pytest.raises(ValidationError, match="domain_size"):
            gp_ucb_beta_schedule(domain_size=100.0)  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]

    def test_rejects_bool_domain_size(self) -> None:
        with pytest.raises(ValidationError, match="domain_size"):
            gp_ucb_beta_schedule(domain_size=True)

    def test_rejects_non_integer_round_index(self) -> None:
        schedule = gp_ucb_beta_schedule(domain_size=100)
        with pytest.raises(ValidationError, match="t"):
            schedule(1.5)  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]

    def test_rejects_bool_round_index(self) -> None:
        schedule = gp_ucb_beta_schedule(domain_size=100)
        with pytest.raises(ValidationError, match="t"):
            schedule(True)

    def test_rejects_delta_outside_open_unit_interval(self) -> None:
        with pytest.raises(ValidationError, match="delta"):
            gp_ucb_beta_schedule(domain_size=100, delta=0.0)
        with pytest.raises(ValidationError, match="delta"):
            gp_ucb_beta_schedule(domain_size=100, delta=1.0)

    def test_rejects_round_index_below_one(self) -> None:
        schedule = gp_ucb_beta_schedule(domain_size=100)
        with pytest.raises(ValidationError, match="t"):
            schedule(0)

    def test_is_picklable(self) -> None:
        schedule = gp_ucb_beta_schedule(domain_size=100, delta=0.05)
        restored = pickle.loads(pickle.dumps(schedule))
        assert restored(5) == pytest.approx(schedule(5))


# ===========================================================================
# CORSDistance tests
# ===========================================================================
class TestCORSDistance:
    """Tests for the CORS distance-constrained acquisition function."""

    def test_compute_reference_returns_public_cors_reference(self) -> None:
        arc = _archive_x([0.0, 5.0, 10.0])
        af = CORSDistance(delta=10.0)
        first = af.compute_reference(arc)
        second = af.compute_reference(arc)

        assert isinstance(first, CORSReference)
        assert isinstance(second, CORSReference)
        np.testing.assert_array_equal(
            np.sort(first.evaluated_x.ravel()), [0.0, 5.0, 10.0]
        )
        assert first.beta == pytest.approx(0.95)
        assert second.beta == pytest.approx(0.95)
        assert not hasattr(af, "_cycle")

    def test_score_rejects_raw_compute_reference_array(self) -> None:
        arc = _archive_x([0.0])
        pred = _pred_x(value=[[5.0]], x=[[0.05]])
        af = CORSDistance(delta=10.0, search_pattern=(1.0,))

        with pytest.raises(ValidationError, match="CORSReference"):
            af.score(pred, reference=arc.x)

    def test_far_candidate_scores_by_predicted_mean(self) -> None:
        """A candidate far from every evaluated point is unaffected by the constraint."""  # noqa: E501
        arc = _archive_x([0.0, 1.0, 2.0])
        pred = _pred_x(value=[[5.0]], x=[[100.0]])
        af = CORSDistance(delta=1.0)
        prepared = af.prepare(arc, ctx=_decision_ctx(0))
        scores = af.score(pred, reference=prepared)
        assert scores[0] == pytest.approx(5.0)

    def test_close_candidate_gets_worst_score(self) -> None:
        """A candidate violating beta_1 * delta gets -inf, never the predicted mean."""
        arc = _archive_x([0.0, 5.0, 10.0])
        # The first prepare uses beta_1 = 0.95 (default SP1); threshold = 9.5.
        pred = _pred_x(value=[[5.0]], x=[[0.05]])
        af = CORSDistance(delta=10.0)
        prepared = af.prepare(arc, ctx=_decision_ctx(0))
        scores = af.score(pred, reference=prepared)
        assert scores[0] == -np.inf

    def test_beta_cycles_across_prepare_calls(self) -> None:
        """beta_i cycles through search_pattern, advancing once per prepare() call."""
        arc = _archive_x([0.0])
        pred = _pred_x(value=[[5.0]], x=[[0.05]])
        af = CORSDistance(delta=10.0, search_pattern=(1.0, 0.0))

        # First prepare: beta_1 = 1.0 -> threshold = 10.0 -> violates.
        prepared = af.prepare(arc, ctx=_decision_ctx(0))
        assert af.score(pred, reference=prepared)[0] == -np.inf
        # Second prepare: beta_2 = 0.0 -> Eq. (1) is trivially satisfied.
        prepared = af.prepare(arc, ctx=_decision_ctx(1))
        assert af.score(pred, reference=prepared)[0] == pytest.approx(5.0)
        # Third prepare wraps back to beta_1 = 1.0 -> violates again.
        prepared = af.prepare(arc, ctx=_decision_ctx(2))
        assert af.score(pred, reference=prepared)[0] == -np.inf

    def test_prepare_resolves_beta_from_decision_count(self) -> None:
        """prepare() derives beta from the runtime decision count."""
        arc = _archive_x([0.0])
        pattern = (0.95, 0.25, 0.05, 0.03, 0.0)
        af = CORSDistance(delta=10.0, search_pattern=pattern)

        betas = [
            af.prepare(arc, ctx=_decision_ctx(decision_count)).beta
            for decision_count in range(7)
        ]

        assert betas == [0.95, 0.25, 0.05, 0.03, 0.0, 0.95, 0.25]

    @pytest.mark.parametrize("archive_size", [23, 25])
    def test_first_beta_does_not_depend_on_archive_size(
        self, archive_size: int
    ) -> None:
        """A-4 starts at SP1's first beta regardless of initial archive size."""
        arc = _archive_x(np.arange(archive_size, dtype=float))
        af = CORSDistance()

        assert af.prepare(arc, ctx=_decision_ctx(0)).beta == pytest.approx(0.95)

    def test_score_is_read_only_for_prepared_beta(self) -> None:
        """Repeated score() calls do not advance the prepared beta."""
        arc = _archive_x([0.0])
        pred = _pred_x(value=[[5.0]], x=[[0.05]])
        af = CORSDistance(delta=10.0, search_pattern=(1.0, 0.0))
        prepared = af.prepare(arc, ctx=_decision_ctx(0))

        first = af.score(pred, reference=prepared)
        second = af.score(pred, reference=prepared)

        assert first[0] == -np.inf
        np.testing.assert_array_equal(second, first)
        assert prepared.beta == 1.0

    def test_prepare_requires_ctx(self) -> None:
        """Runtime preparation requires a decision count source."""
        arc = _archive_x([0.0])
        af = CORSDistance(delta=10.0)

        with pytest.raises(ValidationError, match="decision_count"):
            af.prepare(arc, ctx=None)

    def test_context_decision_count_determines_beta_phase(self) -> None:
        """The runtime decision count determines the CORS phase."""
        arc = _archive_x([0.0])
        pattern = (0.95, 0.25, 0.05, 0.03, 0.0)
        af = CORSDistance(delta=10.0, search_pattern=pattern)

        betas = [
            af.prepare(arc, ctx=_decision_ctx(decision_count)).beta
            for decision_count in (4, 0, 4, 1, 4)
        ]

        assert betas == [0.0, 0.95, 0.0, 0.25, 0.0]

    def test_default_search_pattern_is_cors_sp1(self) -> None:
        """The default search pattern remains Regis & Shoemaker's SP1."""
        assert CORSDistance(delta=1.0).search_pattern == (
            0.95,
            0.25,
            0.05,
            0.03,
            0.0,
        )

    def test_beta_zero_never_excludes(self) -> None:
        """A search_pattern of all zeros never enforces the distance constraint."""
        arc = _archive_x([0.0])
        pred = _pred_x(value=[[5.0]], x=[[0.0]])
        af = CORSDistance(delta=10.0, search_pattern=(0.0,))
        prepared = af.prepare(arc, ctx=_decision_ctx(0))
        for _ in range(3):
            assert af.score(pred, reference=prepared)[0] == pytest.approx(5.0)

    def test_empty_archive_no_constraint(self) -> None:
        arc = _archive_x([])
        pred = _pred_x(value=[[5.0]], x=[[0.0]])
        af = CORSDistance(delta=10.0)
        prepared = af.prepare(arc, ctx=_decision_ctx(0))
        scores = af.score(pred, reference=prepared)
        assert scores[0] == pytest.approx(5.0)

    def test_missing_x_raises(self) -> None:
        arc = _archive_x([0.0])
        pred = _pred(value=[[5.0]])
        af = CORSDistance(delta=10.0)
        prepared = af.prepare(arc, ctx=_decision_ctx(0))
        with pytest.raises(ValueError, match="requires prediction"):
            af.score(pred, reference=prepared)

    def test_x_row_mismatch_raises(self) -> None:
        with pytest.raises(ValidationError, match="shape"):
            _pred_x(value=[[5.0], [6.0]], x=[[0.0], [1.0], [2.0]])

    def test_direction_scalarizes_base_score(self) -> None:
        """The unconstrained base score respects direction, like MeanPrediction."""
        arc = _archive_x([0.0, 1.0])
        pred = _pred_x(value=[[3.0]], x=[[100.0]])
        af = CORSDistance(delta=1.0, direction=np.array([-1.0]))
        prepared = af.prepare(arc, ctx=_decision_ctx(0))
        scores = af.score(pred, reference=prepared)
        assert scores[0] == pytest.approx(-3.0)

    def test_beta_one_keeps_unique_maximin_candidate_at_boundary(self) -> None:
        """At beta=1, the unique maximin candidate passes the strict inequality."""
        arc = _archive_x([0.0, 10.0])
        pred = _pred_x(value=[[1.0], [2.0], [3.0]], x=[[1.0], [5.0], [20.0]])
        af = CORSDistance(search_pattern=(1.0,))
        prepared = af.prepare(arc, ctx=_decision_ctx(0))

        scores = af.score(pred, reference=prepared)

        # Candidate distances are [1, 5, 10], so Delta_i=10 and the
        # beta=1 threshold is 10. Equality is feasible; only the maximin
        # candidate remains finite.
        np.testing.assert_array_equal(scores, [-np.inf, -np.inf, 3.0])
        assert np.count_nonzero(np.isfinite(scores)) == 1

    def test_delta_none_keeps_at_least_one_candidate_feasible_at_beta_095(self) -> None:
        """The pool-derived maximin scale prevents an impossible beta=.95 constraint."""
        arc = _archive_x([0.0, 10.0])
        pred = _pred_x(value=[[1.0], [2.0], [3.0]], x=[[1.0], [5.0], [20.0]])
        af = CORSDistance(search_pattern=(0.95, 0.0))
        prepared = af.prepare(arc, ctx=_decision_ctx(0))

        scores = af.score(pred, reference=prepared)

        assert np.count_nonzero(np.isfinite(scores)) >= 1
        assert scores[2] == pytest.approx(3.0)

    def test_delta_none_uses_candidate_pool_maximin_distance(self) -> None:
        """The threshold uses max_c min_j ||candidate_c - evaluated_j||."""
        arc = _archive_x([0.0, 10.0])
        pred = _pred_x(value=[[1.0], [2.0], [3.0]], x=[[1.0], [5.0], [20.0]])
        af = CORSDistance(search_pattern=(0.5, 0.0))
        prepared = af.prepare(arc, ctx=_decision_ctx(0))

        scores = af.score(pred, reference=prepared)

        # Candidate distances are [1, 5, 10], so Delta_i=10 and the
        # beta=.5 threshold is 5. Equality is feasible by Eq. (1).
        np.testing.assert_array_equal(scores, [-np.inf, 2.0, 3.0])

    def test_explicit_delta_preserves_fixed_distance_scale(self) -> None:
        """An explicit delta keeps the legacy fixed-threshold behavior."""
        arc = _archive_x([0.0, 10.0])
        pred = _pred_x(value=[[1.0], [2.0]], x=[[5.0], [20.0]])
        af = CORSDistance(delta=5.0, search_pattern=(0.95, 0.0))
        prepared = af.prepare(arc, ctx=_decision_ctx(0))

        scores = af.score(pred, reference=prepared)

        # Fixed delta gives threshold=.95*5=4.75; both candidates pass.
        np.testing.assert_array_equal(scores, [1.0, 2.0])

    def test_delta_none_reflects_pool_bias_in_the_approximation(self) -> None:
        """A pool concentrated near the archive yields a smaller Delta_i."""
        arc = _archive_x([0.0, 10.0])
        pred = _pred_x(
            value=[[1.0], [2.0], [3.0], [4.0]],
            x=[[0.1], [0.2], [9.8], [9.9]],
        )
        af = CORSDistance(search_pattern=(0.95, 0.0))
        prepared = af.prepare(arc, ctx=_decision_ctx(0))

        scores = af.score(pred, reference=prepared)

        # Pool maximin distance is .2, so beta=.95 permits the two .2-away
        # points while excluding the two .1-away points.
        np.testing.assert_array_equal(scores, [-np.inf, 2.0, 3.0, -np.inf])

    @pytest.mark.parametrize("delta", [0.0, -1.0, float("nan"), float("inf")])
    def test_rejects_non_positive_or_non_finite_delta(self, delta: float) -> None:
        with pytest.raises(ValidationError, match="delta"):
            CORSDistance(delta=delta)

    @pytest.mark.parametrize(
        "search_pattern",
        [(1.1,), (-0.1,), (float("nan"),), (float("inf"),)],
    )
    def test_search_pattern_rejects_nonfinite_or_out_of_range_values(
        self, search_pattern: tuple[float, ...]
    ) -> None:
        with pytest.raises(ValidationError, match="search_pattern"):
            CORSDistance(search_pattern=search_pattern)

    def test_search_pattern_rejects_empty_tuple(self) -> None:
        with pytest.raises(ValidationError, match="search_pattern"):
            CORSDistance(search_pattern=())

    def test_search_pattern_accepts_all_zero_pattern(self) -> None:
        af = CORSDistance(search_pattern=(0.0,))

        assert af.search_pattern == (0.0,)


# ===========================================================================
# ProbabilityOfFeasibility Tests
# ===========================================================================
class TestProbabilityOfFeasibility:
    """Tests for ProbabilityOfFeasibility acquisition function."""

    def test_requires_uncertainty(self) -> None:
        pred = _pred(value=[[0.5]])
        with pytest.raises(TypeError, match="uncertainty"):
            ProbabilityOfFeasibility().score(pred, reference=None)

    def test_output_shape(self) -> None:
        pred = _pred(value=np.zeros((5, 1)), std=np.ones((5, 1)))
        scores = ProbabilityOfFeasibility().score(pred, reference=None)
        assert scores.shape == (5,)

    def test_scores_in_0_1(self) -> None:
        """PoF scores are always in [0, 1]."""
        rng = np.random.default_rng(1)
        pred = _pred(
            value=rng.standard_normal((20, 1)),
            std=np.abs(rng.standard_normal((20, 1))) + 0.01,
        )
        scores = ProbabilityOfFeasibility().score(pred, reference=None)
        assert np.all(scores >= 0.0)
        assert np.all(scores <= 1.0)

    def test_pof_formula(self) -> None:
        """PoF = Phi(-mu / sigma)."""
        mu, sigma = -1.0, 0.5
        expected = norm.cdf((0.0 - mu) / sigma)
        pred = _pred(value=[[mu]], std=[[sigma]])
        scores = ProbabilityOfFeasibility().score(pred, reference=None)
        assert scores[0] == pytest.approx(expected, rel=1e-5)

    def test_feasible_candidate_scores_near_one(self) -> None:
        """If mu << 0 (clearly feasible), PoF is near 1."""
        pred = _pred(value=[[-10.0]], std=[[0.1]])
        scores = ProbabilityOfFeasibility().score(pred, reference=None)
        assert scores[0] > 0.99

    def test_infeasible_candidate_scores_near_zero(self) -> None:
        """If mu >> 0 (clearly infeasible), PoF is near 0."""
        pred = _pred(value=[[10.0]], std=[[0.1]])
        scores = ProbabilityOfFeasibility().score(pred, reference=None)
        assert scores[0] < 0.01

    def test_obj_idx(self) -> None:
        """obj_idx selects which objective (constraint) to evaluate."""
        pred = _pred(
            value=[[-5.0, 5.0]],
            std=[[0.1, 0.1]],
        )
        s0 = ProbabilityOfFeasibility(obj_idx=0).score(pred, reference=None)
        s1 = ProbabilityOfFeasibility(obj_idx=1).score(pred, reference=None)
        assert s0[0] > 0.99  # mu=-5: clearly feasible
        assert s1[0] < 0.01  # mu=+5: clearly infeasible


# ===========================================================================
# ProductOfFeasibility Tests
# ===========================================================================
class TestProductOfFeasibility:
    """Tests for ProductOfFeasibility acquisition function."""

    def test_requires_uncertainty(self) -> None:
        pred = _pred(value=[[0.5, 0.5]])
        with pytest.raises(TypeError, match="uncertainty"):
            ProductOfFeasibility().score(pred, reference=None)

    def test_output_shape(self) -> None:
        pred = _pred(value=np.zeros((5, 2)), std=np.ones((5, 2)))
        scores = ProductOfFeasibility().score(pred, reference=None)
        assert scores.shape == (5,)

    def test_scores_in_0_1(self) -> None:
        rng = np.random.default_rng(2)
        pred = _pred(
            value=rng.standard_normal((20, 3)),
            std=np.abs(rng.standard_normal((20, 3))) + 0.01,
        )
        scores = ProductOfFeasibility().score(pred, reference=None)
        assert np.all(scores >= 0.0)
        assert np.all(scores <= 1.0)

    def test_single_constraint_matches_pof(self) -> None:
        """With one constraint column, equals ProbabilityOfFeasibility(obj_idx=0)."""
        rng = np.random.default_rng(3)
        mu = rng.standard_normal((10, 1))
        sigma = np.abs(rng.standard_normal((10, 1))) + 0.01
        pred = _pred(value=mu, std=sigma)
        pof_scores = ProbabilityOfFeasibility(obj_idx=0).score(pred)
        product_scores = ProductOfFeasibility().score(pred)
        np.testing.assert_allclose(product_scores, pof_scores, rtol=1e-6)

    def test_all_feasible_scores_near_one(self) -> None:
        """If all mu << 0, joint PoF is near 1."""
        pred = _pred(value=[[-10.0, -10.0]], std=[[0.1, 0.1]])
        scores = ProductOfFeasibility().score(pred)
        assert scores[0] > 0.99

    def test_one_infeasible_constraint_pulls_score_to_zero(self) -> None:
        """Even if one constraint has mu >> 0, joint PoF is near 0."""
        pred = _pred(value=[[-10.0, 10.0]], std=[[0.1, 0.1]])
        scores = ProductOfFeasibility().score(pred)
        assert scores[0] < 0.01

    def test_product_formula(self) -> None:
        """PoF_joint = Phi(-mu1/s1) * Phi(-mu2/s2)."""
        mu1, s1, mu2, s2 = -1.0, 0.5, 0.5, 1.0
        expected = norm.cdf(-mu1 / s1) * norm.cdf(-mu2 / s2)
        pred = _pred(value=[[mu1, mu2]], std=[[s1, s2]])
        scores = ProductOfFeasibility().score(pred)
        assert scores[0] == pytest.approx(expected, rel=1e-5)


# ===========================================================================
# Direction-aware minimize-space conversion
# ===========================================================================
class TestDirectionSensitivity:
    """
    EI/LCB internally convert to minimize-space via ``direction`` before
    running their (minimize-only) formulas. Scoring (mu, sigma) under
    ``direction=+1`` (maximize) must mirror scoring (-mu, sigma) under
    ``direction=-1`` (minimize) exactly, since ``direction=-1`` is a no-op
    conversion (sign=+1.0), anchoring the maximize side to already-verified
    minimize behavior.
    """

    def test_ei_maximize_mirrors_minimize_negated(self) -> None:
        archive_rows = [[2.0], [0.0], [-1.0]]
        mu = np.array([[1.0], [3.0], [-2.0], [0.5]])
        sigma = np.full((4, 1), 0.5)

        af_max = ExpectedImprovement(xi=0.0, direction=np.array([1.0]))
        af_min = ExpectedImprovement(xi=0.0, direction=np.array([-1.0]))

        ref_max = af_max.compute_reference(_archive(archive_rows))
        ref_min = af_min.compute_reference(_archive([[-v[0]] for v in archive_rows]))

        scores_max = af_max.score(_pred(mu, sigma), ref_max)
        scores_min = af_min.score(_pred(-mu, sigma), ref_min)

        np.testing.assert_allclose(scores_max, scores_min)
        assert np.argsort(scores_max).tolist() == np.argsort(scores_min).tolist()

    def test_lcb_maximize_mirrors_minimize_negated(self) -> None:
        mu = np.array([[1.0], [3.0], [-2.0], [0.5]])
        sigma = np.full((4, 1), 0.3)

        af_max = LowerConfidenceBound(kappa=1.5, direction=np.array([1.0]))
        af_min = LowerConfidenceBound(kappa=1.5, direction=np.array([-1.0]))

        scores_max = af_max.score(_pred(mu, sigma), reference=None)
        scores_min = af_min.score(_pred(-mu, sigma), reference=None)

        np.testing.assert_allclose(scores_max, scores_min)
        assert np.argsort(scores_max).tolist() == np.argsort(scores_min).tolist()

    def test_ei_direction_none_matches_direction_minus_one(self) -> None:
        """Default (direction=None) is a no-op, same as an explicit -1."""
        pred = _pred(value=[[1.5]], std=[[0.5]])
        s_default = ExpectedImprovement(xi=0.0).score(pred, reference=2.0)
        s_explicit = ExpectedImprovement(xi=0.0, direction=np.array([-1.0])).score(
            pred, reference=2.0
        )
        np.testing.assert_allclose(s_default, s_explicit)


class TestDirectionSensitiveOptOut:
    """PoF/ProductOfFeasibility/MaxUncertainty opt out via direction_sensitive=False."""

    def test_probability_of_feasibility_opts_out(self) -> None:
        assert ProbabilityOfFeasibility.direction_sensitive is False

    def test_product_of_feasibility_opts_out(self) -> None:
        assert ProductOfFeasibility.direction_sensitive is False

    def test_max_uncertainty_opts_out(self) -> None:
        assert MaxUncertainty.direction_sensitive is False

    def test_ei_lcb_parego_smsego_ehvi_are_direction_sensitive(self) -> None:
        assert ExpectedImprovement.direction_sensitive is True
        assert LowerConfidenceBound.direction_sensitive is True
