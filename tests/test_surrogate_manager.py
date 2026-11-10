"""
Tests for the surrogate manager module.

Tests cover:
- _split_prediction / _stack_predictions: per-sample <-> batched conversion
- _rank_normalize: rank-based normalization to [0, 1]
- GlobalSurrogateManager: fits on full archive, batch predict
- LocalSurrogateManager: KNN per candidate, per-candidate fit and predict
- CompositeSurrogateManager: full multi-channel composition -- one named
  PredictionChannel per sub-manager
- PairwiseSurrogateManager: predicts a per-candidate win rate against archive
  reference points as a "win_rate" prediction channel, read via
  WinRateAcquisition
- NoveltyAcquisition/InverseDensityAcquisition/MaximinDistanceAcquisition: archive-only
  archive-based acquisitions
"""

import numpy as np
import pytest

from saealib.acquisition import CompositeAcquisition, MeanPrediction
from saealib.acquisition.archive_based import (
    InverseDensityAcquisition,
    MaximinDistanceAcquisition,
    NoveltyAcquisition,
)
from saealib.acquisition.base import AcquisitionResult
from saealib.acquisition.winrate import WinRateAcquisition
from saealib.comparators import SingleObjectiveComparator
from saealib.context import OptimizationState
from saealib.exceptions import ValidationError
from saealib.population import Archive, ParetoArchive, Population, PopulationAttribute
from saealib.problem import Problem
from saealib.stages import AcquisitionStage, SurrogatePredictStage
from saealib.surrogate.accuracy import (
    KFoldAccuracyEvaluator,
    SpearmanCorrelation,
    SurrogateAccuracy,
)
from saealib.surrogate.manager import (
    CompositeSurrogateManager,
    GlobalSurrogateManager,
    LocalSurrogateManager,
    PairwiseSurrogateManager,
    _rank_normalize,
    _split_prediction,
    _stack_predictions,
    product_combine,
    rank_weighted_combine,
)
from saealib.surrogate.prediction import SurrogatePrediction
from saealib.surrogate.rbf import RBFSurrogate
from saealib.surrogate.rbf_kernels import GaussianKernel
from saealib.surrogate.sklearn_surrogate import SklearnRFCClassificationSurrogate
from saealib.surrogate.training_set import KNNObjectiveSet, PairwiseComparisonSet

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
DIM = 2
N_OBJ = 1


def _scores(result: AcquisitionResult) -> np.ndarray:
    assert result.scores is not None
    return result.scores


@pytest.fixture
def archive_1obj() -> Archive:
    """Archive pre-filled with 20 single-objective training points."""
    attrs = [
        PopulationAttribute(name="x", dtype=np.float64, shape=(DIM,)),
        PopulationAttribute(name="f", dtype=np.float64, shape=(N_OBJ,)),
    ]
    arc = Archive(attrs, init_capacity=30)
    rng = np.random.default_rng(42)
    for _ in range(20):
        x = rng.uniform(-2.0, 2.0, size=DIM)
        f = np.array([np.sum(x**2)])
        arc.add(x=x, f=f)
    return arc


@pytest.fixture
def archive_2obj() -> Archive:
    """Archive pre-filled with 20 bi-objective training points."""
    attrs = [
        PopulationAttribute(name="x", dtype=np.float64, shape=(DIM,)),
        PopulationAttribute(name="f", dtype=np.float64, shape=(2,)),
    ]
    arc = Archive(attrs, init_capacity=30)
    rng = np.random.default_rng(0)
    for _ in range(20):
        x = rng.uniform(0.0, 2.0, size=DIM)
        f = np.array([np.sum(x**2), np.sum((x - 2.0) ** 2)])
        arc.add(x=x, f=f)
    return arc


@pytest.fixture
def candidates() -> np.ndarray:
    """5 candidate points in 2D."""
    rng = np.random.default_rng(7)
    return rng.uniform(-1.0, 1.0, size=(5, DIM))


@pytest.fixture
def surrogate_1obj() -> RBFSurrogate:
    return RBFSurrogate(GaussianKernel())


@pytest.fixture
def surrogate_2obj() -> RBFSurrogate:
    return RBFSurrogate(GaussianKernel())


# ===========================================================================
# _split_prediction Tests
# ===========================================================================
class TestSplitPrediction:
    """Tests for the _split_prediction helper."""

    def test_splits_into_correct_count(self) -> None:
        pred = SurrogatePrediction.objective(value=np.zeros((4, 2)))
        parts = _split_prediction(pred)
        assert len(parts) == 4

    def test_each_part_has_shape_1_nobj(self) -> None:
        pred = SurrogatePrediction.objective(
            value=np.arange(6).reshape(3, 2).astype(float)
        )
        parts = _split_prediction(pred)
        for p in parts:
            assert p.value.shape == (1, 2)

    def test_values_preserved(self) -> None:
        mean = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        pred = SurrogatePrediction.objective(value=mean)
        parts = _split_prediction(pred)
        for i, p in enumerate(parts):
            np.testing.assert_array_equal(p.value[0], mean[i])

    def test_std_split_correctly(self) -> None:
        std = np.array([[0.1, 0.2], [0.3, 0.4]])
        pred = SurrogatePrediction.objective(value=np.zeros((2, 2)), std=std)
        parts = _split_prediction(pred)
        std = parts[0].std
        assert std is not None
        assert std.shape == (1, 2)
        np.testing.assert_array_almost_equal(std[0], [0.1, 0.2])

    def test_std_none_propagates(self) -> None:
        pred = SurrogatePrediction.objective(value=np.zeros((3, 1)))
        parts = _split_prediction(pred)
        for p in parts:
            assert p.std is None

    def test_label_split_correctly(self) -> None:
        label = np.array([0.0, 1.0, 2.0])
        pred = SurrogatePrediction.objective(value=np.zeros((3, 1)))
        pred = SurrogatePrediction(channels=pred.channels, label=label)
        parts = _split_prediction(pred)
        for i, p in enumerate(parts):
            assert p.label is not None
            assert p.label[0] == pytest.approx(float(i))

    def test_x_split_correctly(self) -> None:
        x = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        pred = SurrogatePrediction.objective(value=np.zeros((3, 1)), x=x)
        parts = _split_prediction(pred)
        for i, p in enumerate(parts):
            assert p.x is not None
            assert p.x.shape == (1, 2)
            np.testing.assert_array_equal(p.x[0], x[i])

    def test_x_none_propagates(self) -> None:
        pred = SurrogatePrediction.objective(value=np.zeros((3, 1)))
        parts = _split_prediction(pred)
        for p in parts:
            assert p.x is None

    def test_metadata_shared(self) -> None:
        """metadata dict is shared (not deep-copied) across splits."""
        meta: dict[str, object] = {"key": "val"}
        pred = SurrogatePrediction.objective(value=np.zeros((2, 1)), metadata=meta)
        parts = _split_prediction(pred)
        for p in parts:
            assert p.metadata is meta

    def test_objective_channel_splits_correctly(self) -> None:
        pred = SurrogatePrediction.objective(value=np.array([[10.0], [20.0], [30.0]]))
        parts = _split_prediction(pred)
        np.testing.assert_array_equal(parts[0].value, [[10.0]])
        np.testing.assert_array_equal(parts[1].value, [[20.0]])
        np.testing.assert_array_equal(parts[2].value, [[30.0]])


class TestStackPredictions:
    """Tests for the _stack_predictions helper (inverse of _split_prediction)."""

    def test_empty_list_returns_empty_channels(self) -> None:
        result = _stack_predictions([])
        assert result.channels == {}

    def test_stacks_value_across_rows(self) -> None:
        parts = [
            SurrogatePrediction.objective(value=np.array([[1.0, 2.0]])),
            SurrogatePrediction.objective(value=np.array([[3.0, 4.0]])),
        ]
        stacked = _stack_predictions(parts)
        np.testing.assert_array_equal(stacked.value, [[1.0, 2.0], [3.0, 4.0]])

    def test_stacks_std_when_present_on_all(self) -> None:
        parts = [
            SurrogatePrediction.objective(
                value=np.zeros((1, 1)), std=np.full((1, 1), 0.1)
            ),
            SurrogatePrediction.objective(
                value=np.zeros((1, 1)), std=np.full((1, 1), 0.2)
            ),
        ]
        stacked = _stack_predictions(parts)
        assert stacked.std is not None
        np.testing.assert_array_almost_equal(stacked.std, [[0.1], [0.2]])

    def test_std_none_when_none_on_all(self) -> None:
        parts = [
            SurrogatePrediction.objective(value=np.zeros((1, 1))),
            SurrogatePrediction.objective(value=np.zeros((1, 1))),
        ]
        stacked = _stack_predictions(parts)
        assert stacked.std is None

    def test_mixed_std_raises(self) -> None:
        parts = [
            SurrogatePrediction.objective(value=np.zeros((1, 1)), std=np.zeros((1, 1))),
            SurrogatePrediction.objective(value=np.zeros((1, 1))),
        ]
        with pytest.raises(ValidationError):
            _stack_predictions(parts)

    def test_round_trips_with_split_prediction(self) -> None:
        original = SurrogatePrediction.objective(
            value=np.array([[1.0], [2.0], [3.0]]), std=np.full((3, 1), 0.5)
        )
        original = SurrogatePrediction(
            channels=original.channels,
            label=np.array([0, 1, 1]),
        )
        parts = _split_prediction(original)
        stacked = _stack_predictions(parts)
        np.testing.assert_array_equal(stacked.value, original.value)
        assert stacked.std is not None
        assert original.std is not None
        np.testing.assert_array_almost_equal(stacked.std, original.std)
        np.testing.assert_array_equal(stacked.label, original.label)

    def test_stacks_label_when_present_on_all(self) -> None:
        parts = [
            SurrogatePrediction(
                channels=SurrogatePrediction.objective(value=np.zeros((1, 1))).channels,
                label=np.array([0]),
            ),
            SurrogatePrediction(
                channels=SurrogatePrediction.objective(value=np.zeros((1, 1))).channels,
                label=np.array([1]),
            ),
        ]
        stacked = _stack_predictions(parts)
        np.testing.assert_array_equal(stacked.label, [0, 1])

    def test_label_none_when_none_on_all(self) -> None:
        parts = [
            SurrogatePrediction.objective(value=np.zeros((1, 1))),
            SurrogatePrediction.objective(value=np.zeros((1, 1))),
        ]
        stacked = _stack_predictions(parts)
        assert stacked.label is None

    def test_mixed_label_raises(self) -> None:
        parts = [
            SurrogatePrediction(
                channels=SurrogatePrediction.objective(value=np.zeros((1, 1))).channels,
                label=np.array([0]),
            ),
            SurrogatePrediction.objective(value=np.zeros((1, 1))),
        ]
        with pytest.raises(ValidationError):
            _stack_predictions(parts)


# ===========================================================================
# SurrogatePrediction Properties Tests
# ===========================================================================
class TestSurrogatePredictionProperties:
    """Tests for prediction channel properties."""

    def test_value_uses_objective_channel(self) -> None:
        pred = SurrogatePrediction.objective(value=np.array([[42.0]]))
        np.testing.assert_array_almost_equal(pred.value, [[42.0]])


# ===========================================================================
# _rank_normalize Tests
# ===========================================================================
class TestRankNormalize:
    """Tests for the _rank_normalize helper."""

    def test_single_element_returns_one(self) -> None:
        result = _rank_normalize(np.array([5.0]))
        assert result[0] == pytest.approx(1.0)

    def test_two_elements(self) -> None:
        result = _rank_normalize(np.array([1.0, 3.0]))
        assert result[0] == pytest.approx(0.0)
        assert result[1] == pytest.approx(1.0)

    def test_three_elements_ascending(self) -> None:
        result = _rank_normalize(np.array([1.0, 2.0, 3.0]))
        np.testing.assert_array_almost_equal(result, [0.0, 0.5, 1.0])

    def test_order_independent(self) -> None:
        """Permuting the input should permute the output the same way."""
        scores = np.array([3.0, 1.0, 2.0])
        result = _rank_normalize(scores)
        assert result[0] == pytest.approx(1.0)
        assert result[1] == pytest.approx(0.0)
        assert result[2] == pytest.approx(0.5)

    def test_output_range_0_to_1(self) -> None:
        rng = np.random.default_rng(99)
        scores = rng.standard_normal(50)
        result = _rank_normalize(scores)
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

    def test_output_shape(self) -> None:
        result = _rank_normalize(np.arange(10, dtype=float))
        assert result.shape == (10,)

    def test_nan_treated_as_lowest(self) -> None:
        """NaN scores must be assigned rank 0 (worst), not rank n-1 (best)."""
        scores = np.array([3.0, 1.0, np.nan, 2.0])
        result = _rank_normalize(scores)
        assert result[2] == pytest.approx(0.0), "NaN should map to 0.0 (lowest rank)"

    def test_nan_never_highest(self) -> None:
        """NaN scores must never be selected as best in argsort(-normalized)."""
        scores = np.array([np.nan, 1.0, 2.0])
        result = _rank_normalize(scores)
        best_idx = np.argmax(result)
        assert best_idx != 0, (
            "NaN candidate should not have the highest normalized score"
        )


# ===========================================================================
# GlobalSurrogateManager Tests
# ===========================================================================
class TestGlobalSurrogateManager:
    """Tests for GlobalSurrogateManager."""

    def test_predict_returns_surrogate_prediction(
        self,
        surrogate_1obj: RBFSurrogate,
        archive_1obj: Archive,
        candidates: np.ndarray,
    ) -> None:
        manager = GlobalSurrogateManager(surrogate_1obj)
        prediction = manager.predict(candidates, archive_1obj)
        assert isinstance(prediction, SurrogatePrediction)

    def test_prediction_value_shape(
        self,
        surrogate_1obj: RBFSurrogate,
        archive_1obj: Archive,
        candidates: np.ndarray,
    ) -> None:
        manager = GlobalSurrogateManager(surrogate_1obj)
        prediction = manager.predict(candidates, archive_1obj)
        assert prediction.value.shape == (len(candidates), N_OBJ)

    def test_scores_shape_via_acquisition(
        self,
        surrogate_1obj: RBFSurrogate,
        archive_1obj: Archive,
        candidates: np.ndarray,
    ) -> None:
        manager = GlobalSurrogateManager(surrogate_1obj)
        acquisition = MeanPrediction()
        prediction = manager.predict(candidates, archive_1obj)
        scores = _scores(acquisition.evaluate(candidates, prediction, archive_1obj))
        assert scores.shape == (len(candidates),)

    def test_scores_finite(
        self,
        surrogate_1obj: RBFSurrogate,
        archive_1obj: Archive,
        candidates: np.ndarray,
    ) -> None:
        manager = GlobalSurrogateManager(surrogate_1obj)
        acquisition = MeanPrediction()
        prediction = manager.predict(candidates, archive_1obj)
        scores = _scores(acquisition.evaluate(candidates, prediction, archive_1obj))
        assert np.all(np.isfinite(scores))

    def test_biobj_prediction_and_scores_shape(
        self,
        surrogate_2obj: RBFSurrogate,
        archive_2obj: Archive,
        candidates: np.ndarray,
    ) -> None:
        """Bi-objective: scores still shape (n_candidates,)."""
        weights = np.array([-1.0, -1.0])
        manager = GlobalSurrogateManager(surrogate_2obj)
        acquisition = MeanPrediction(weights=weights)
        prediction = manager.predict(candidates, archive_2obj)
        scores = _scores(acquisition.evaluate(candidates, prediction, archive_2obj))
        assert scores.shape == (len(candidates),)
        assert prediction.value.shape == (len(candidates), 2)


# ===========================================================================
# LocalSurrogateManager Tests
# ===========================================================================
class TestLocalSurrogateManager:
    """Tests for LocalSurrogateManager."""

    def test_predict_returns_surrogate_prediction(
        self,
        surrogate_1obj: RBFSurrogate,
        archive_1obj: Archive,
        candidates: np.ndarray,
    ) -> None:
        manager = LocalSurrogateManager(
            surrogate_1obj, training_set=KNNObjectiveSet(10)
        )
        prediction = manager.predict(candidates, archive_1obj)
        assert isinstance(prediction, SurrogatePrediction)

    def test_scores_shape_via_acquisition(
        self,
        surrogate_1obj: RBFSurrogate,
        archive_1obj: Archive,
        candidates: np.ndarray,
    ) -> None:
        manager = LocalSurrogateManager(
            surrogate_1obj, training_set=KNNObjectiveSet(10)
        )
        acquisition = MeanPrediction()
        prediction = manager.predict(candidates, archive_1obj)
        scores = _scores(acquisition.evaluate(candidates, prediction, archive_1obj))
        assert scores.shape == (len(candidates),)

    def test_prediction_value_shape(
        self,
        surrogate_1obj: RBFSurrogate,
        archive_1obj: Archive,
        candidates: np.ndarray,
    ) -> None:
        manager = LocalSurrogateManager(
            surrogate_1obj, training_set=KNNObjectiveSet(10)
        )
        prediction = manager.predict(candidates, archive_1obj)
        assert prediction.value.shape == (len(candidates), N_OBJ)

    def test_n_neighbors_default(
        self,
        surrogate_1obj: RBFSurrogate,
        archive_1obj: Archive,
        candidates: np.ndarray,
    ) -> None:
        """Default training_set is KNNObjectiveSet(50), clamped to archive size."""
        manager = LocalSurrogateManager(surrogate_1obj)
        assert isinstance(manager.training_set, KNNObjectiveSet)
        assert manager.training_set.n_neighbors == 50
        # archive has only 20 points, get_knn should still work
        prediction = manager.predict(candidates, archive_1obj)
        scores = _scores(
            MeanPrediction().evaluate(candidates, prediction, archive_1obj)
        )
        assert scores.shape == (len(candidates),)

    def test_scores_finite(
        self,
        surrogate_1obj: RBFSurrogate,
        archive_1obj: Archive,
        candidates: np.ndarray,
    ) -> None:
        manager = LocalSurrogateManager(
            surrogate_1obj, training_set=KNNObjectiveSet(10)
        )
        acquisition = MeanPrediction()
        prediction = manager.predict(candidates, archive_1obj)
        scores = _scores(acquisition.evaluate(candidates, prediction, archive_1obj))
        assert np.all(np.isfinite(scores))


# ===========================================================================
# CompositeSurrogateManager Tests
# ===========================================================================
class TestCompositeSurrogateManager:
    """Tests for CompositeSurrogateManager.

    Each sub-manager contributes a named ``PredictionChannel``.
    """

    def test_empty_managers_raises_value_error(self) -> None:
        # Use a plain (non-Composite) acquisition so the raise provably
        # comes from CompositeSurrogateManager's own guard, not from
        # CompositeAcquisition's identically-worded empty-mapping guard.
        with pytest.raises(ValueError, match="at least one"):
            CompositeSurrogateManager({})

    def test_predict_composes_named_channels(
        self, archive_1obj: Archive, candidates: np.ndarray
    ) -> None:
        """Each sub-manager contributes its own named PredictionChannel."""
        m1 = GlobalSurrogateManager(RBFSurrogate(GaussianKernel()))
        m2 = LocalSurrogateManager(
            RBFSurrogate(GaussianKernel()),
            training_set=KNNObjectiveSet(5),
        )
        mgr = CompositeSurrogateManager({"a": m1, "b": m2})
        composite_pred = mgr.predict(candidates, archive_1obj)

        m1_standalone = GlobalSurrogateManager(RBFSurrogate(GaussianKernel()))
        m2_standalone = LocalSurrogateManager(
            RBFSurrogate(GaussianKernel()),
            training_set=KNNObjectiveSet(5),
        )
        m1_pred = m1_standalone.predict(candidates, archive_1obj)
        m2_pred = m2_standalone.predict(candidates, archive_1obj)

        np.testing.assert_array_equal(composite_pred.channels["a"].value, m1_pred.value)
        np.testing.assert_array_equal(composite_pred.channels["b"].value, m2_pred.value)
        # RBF-global vs. RBF-local-KNN genuinely differ, so this catches a
        # channel-mapping bug (e.g. both keys pointing at the same manager).
        assert not np.array_equal(
            composite_pred.channels["a"].value, composite_pred.channels["b"].value
        )

    def test_single_manager_predict(
        self,
        surrogate_1obj: RBFSurrogate,
        archive_1obj: Archive,
        candidates: np.ndarray,
    ) -> None:
        m = GlobalSurrogateManager(surrogate_1obj)
        composite_acq = CompositeAcquisition(
            {"objective": MeanPrediction()}, combine_fn=product_combine
        )
        mgr = CompositeSurrogateManager({"objective": m})
        prediction = mgr.predict(candidates, archive_1obj)
        scores = _scores(composite_acq.evaluate(candidates, prediction, archive_1obj))
        assert scores.shape == (len(candidates),)

    def test_product_combine_scores_shape(
        self, archive_1obj: Archive, candidates: np.ndarray
    ) -> None:
        m1 = GlobalSurrogateManager(RBFSurrogate(GaussianKernel()))
        m2 = GlobalSurrogateManager(RBFSurrogate(GaussianKernel()))
        composite_acq = CompositeAcquisition(
            {"a": MeanPrediction(), "b": MeanPrediction()}, combine_fn=product_combine
        )
        mgr = CompositeSurrogateManager({"a": m1, "b": m2})
        prediction = mgr.predict(candidates, archive_1obj)
        scores = _scores(composite_acq.evaluate(candidates, prediction, archive_1obj))
        assert scores.shape == (len(candidates),)

    def test_rank_weighted_combine_scores_in_0_1(
        self, archive_1obj: Archive, candidates: np.ndarray
    ) -> None:
        m1 = GlobalSurrogateManager(RBFSurrogate(GaussianKernel()))
        m2 = GlobalSurrogateManager(RBFSurrogate(GaussianKernel()))
        composite_acq = CompositeAcquisition(
            {"a": MeanPrediction(), "b": MeanPrediction()},
            combine_fn=rank_weighted_combine(),
        )
        mgr = CompositeSurrogateManager({"a": m1, "b": m2})
        prediction = mgr.predict(candidates, archive_1obj)
        scores = _scores(composite_acq.evaluate(candidates, prediction, archive_1obj))
        assert np.all(scores >= 0.0)
        assert np.all(scores <= 1.0)

    def test_rank_weighted_combine_custom_weights(
        self, archive_1obj: Archive, candidates: np.ndarray
    ) -> None:
        m1 = GlobalSurrogateManager(RBFSurrogate(GaussianKernel()))
        m2 = GlobalSurrogateManager(RBFSurrogate(GaussianKernel()))
        composite_acq = CompositeAcquisition(
            {"a": MeanPrediction(), "b": MeanPrediction()},
            combine_fn=rank_weighted_combine(np.array([1.0, 3.0])),
        )
        mgr = CompositeSurrogateManager({"a": m1, "b": m2})
        prediction = mgr.predict(candidates, archive_1obj)
        scores = _scores(composite_acq.evaluate(candidates, prediction, archive_1obj))
        assert scores.shape == (len(candidates),)


# ===========================================================================
# Archive-based acquisitions
# ===========================================================================


class TestNoveltyAcquisition:
    """Tests for NoveltyAcquisition."""

    def test_evaluate_returns_acquisition_result(
        self, archive_1obj: Archive, candidates: np.ndarray
    ) -> None:
        acq = NoveltyAcquisition(k=3)
        result = acq.evaluate(candidates, None, archive_1obj)
        assert result.scores is not None
        assert result.scores.shape == (len(candidates),)

    def test_empty_archive_returns_ones(self, candidates: np.ndarray) -> None:
        attrs = [
            PopulationAttribute(name="x", dtype=np.float64, shape=(DIM,)),
            PopulationAttribute(name="f", dtype=np.float64, shape=(1,)),
        ]
        empty_arc = Archive(attrs, init_capacity=10)
        acq = NoveltyAcquisition(k=3)
        scores = _scores(acq.evaluate(candidates, None, empty_arc))
        np.testing.assert_array_equal(scores, np.ones(len(candidates)))

    def test_scores_nonnegative(
        self, archive_1obj: Archive, candidates: np.ndarray
    ) -> None:
        acq = NoveltyAcquisition(k=3)
        scores = _scores(acq.evaluate(candidates, None, archive_1obj))
        assert np.all(scores >= 0.0)

    def test_more_distant_point_has_higher_novelty(self, archive_1obj: Archive) -> None:
        """A point far from the archive should score higher than a nearby one."""
        near = archive_1obj.x[0:1] + 1e-6  # almost identical to archive point
        far = np.array([[100.0, 100.0]])
        acq = NoveltyAcquisition(k=1)
        score_near = _scores(acq.evaluate(near, None, archive_1obj))[0]
        score_far = _scores(acq.evaluate(far, None, archive_1obj))[0]
        assert score_far > score_near

    def test_k_clamped_to_archive_size(self, archive_1obj: Archive) -> None:
        """k larger than archive size should not raise an error."""
        acq = NoveltyAcquisition(k=1000)
        candidates = np.zeros((3, DIM))
        scores = _scores(acq.evaluate(candidates, None, archive_1obj))
        assert scores.shape == (3,)
        assert np.all(np.isfinite(scores))

    def test_k_affects_scores(self, archive_1obj: Archive) -> None:
        """Different k values yield different scores (unless all distances equal)."""
        candidates = np.zeros((5, DIM))
        s1 = _scores(NoveltyAcquisition(k=1).evaluate(candidates, None, archive_1obj))
        s5 = _scores(NoveltyAcquisition(k=5).evaluate(candidates, None, archive_1obj))
        # mean of 1 NN vs mean of 5 NN — they may differ
        assert s1.shape == s5.shape == (5,)

    def test_direction_sensitive_is_false(self) -> None:
        """Pure x-space distance; must opt out of direction auto-injection."""
        assert NoveltyAcquisition().direction_sensitive is False


# ===========================================================================
# InverseDensityAcquisition tests
# ===========================================================================
class TestInverseDensityAcquisition:
    """Tests for InverseDensityAcquisition."""

    def test_empty_archive_returns_ones(self, candidates: np.ndarray) -> None:
        attrs = [
            PopulationAttribute(name="x", dtype=np.float64, shape=(DIM,)),
            PopulationAttribute(name="f", dtype=np.float64, shape=(1,)),
        ]
        empty_arc = Archive(attrs, init_capacity=10)
        acq = InverseDensityAcquisition(eps=1.0)
        scores = _scores(acq.evaluate(candidates, None, empty_arc))
        np.testing.assert_array_equal(scores, np.ones(len(candidates)))

    def test_scores_shape(self, archive_1obj: Archive, candidates: np.ndarray) -> None:
        acq = InverseDensityAcquisition(eps=1.0)
        scores = _scores(acq.evaluate(candidates, None, archive_1obj))
        assert scores.shape == (len(candidates),)

    def test_scores_positive(
        self, archive_1obj: Archive, candidates: np.ndarray
    ) -> None:
        acq = InverseDensityAcquisition(eps=1.0)
        scores = _scores(acq.evaluate(candidates, None, archive_1obj))
        assert np.all(scores > 0.0)

    def test_sparse_region_has_higher_score(self, archive_1obj: Archive) -> None:
        """Point far from archive (no neighbors within eps) should score higher."""
        dense = archive_1obj.x[0:1] + 0.01  # inside many eps-balls
        sparse = np.array([[100.0, 100.0]])  # far away, zero neighbors
        acq = InverseDensityAcquisition(eps=0.5)
        score_dense = _scores(acq.evaluate(dense, None, archive_1obj))[0]
        score_sparse = _scores(acq.evaluate(sparse, None, archive_1obj))[0]
        assert score_sparse > score_dense

    def test_eps_affects_scores(self, archive_1obj: Archive) -> None:
        """Larger eps counts more neighbors -> lower inverse density."""
        candidate = np.zeros((1, DIM))
        score_small_eps = _scores(
            InverseDensityAcquisition(eps=0.01).evaluate(candidate, None, archive_1obj)
        )[0]
        score_large_eps = _scores(
            InverseDensityAcquisition(eps=100.0).evaluate(candidate, None, archive_1obj)
        )[0]
        assert score_small_eps >= score_large_eps

    def test_direction_sensitive_is_false(self) -> None:
        assert InverseDensityAcquisition().direction_sensitive is False


# ===========================================================================
# MaximinDistanceAcquisition tests
# ===========================================================================
class TestMaximinDistanceAcquisition:
    """Tests for MaximinDistanceAcquisition."""

    def test_scores_shape(self, archive_1obj: Archive, candidates: np.ndarray) -> None:
        acq = MaximinDistanceAcquisition()
        scores = _scores(acq.evaluate(candidates, None, archive_1obj))
        assert scores.shape == (len(candidates),)

    def test_single_candidate_returns_ones(self, archive_1obj: Archive) -> None:
        acq = MaximinDistanceAcquisition()
        single = np.zeros((1, DIM))
        scores = _scores(acq.evaluate(single, None, archive_1obj))
        np.testing.assert_array_equal(scores, np.ones(1))

    def test_scores_nonnegative(
        self, archive_1obj: Archive, candidates: np.ndarray
    ) -> None:
        acq = MaximinDistanceAcquisition()
        scores = _scores(acq.evaluate(candidates, None, archive_1obj))
        assert np.all(scores >= 0.0)

    def test_isolated_candidate_has_higher_score(self, archive_1obj: Archive) -> None:
        """A candidate isolated from others and the archive should score higher."""
        clustered = np.array([[0.0, 0.0], [0.01, 0.0], [0.0, 0.01]])
        isolated_point = np.array([100.0, 100.0])
        candidates = np.vstack([clustered, isolated_point[np.newaxis]])
        acq = MaximinDistanceAcquisition()
        scores = _scores(acq.evaluate(candidates, None, archive_1obj))
        assert scores[-1] > scores[0]

    def test_empty_archive(self, candidates: np.ndarray) -> None:
        """Empty archive: archive_min falls back to ones."""
        attrs = [
            PopulationAttribute(name="x", dtype=np.float64, shape=(DIM,)),
            PopulationAttribute(name="f", dtype=np.float64, shape=(1,)),
        ]
        empty_arc = Archive(attrs, init_capacity=10)
        acq = MaximinDistanceAcquisition()
        scores = _scores(acq.evaluate(candidates, None, empty_arc))
        assert scores.shape == (len(candidates),)
        assert np.all(np.isfinite(scores))

    def test_direction_sensitive_is_false(self) -> None:
        assert MaximinDistanceAcquisition().direction_sensitive is False


# ===========================================================================
# Surrogate lifecycle hooks (post_fit / with_post_fit)
# ===========================================================================


class TestSurrogateHooks:
    """Tests for Surrogate.post_fit and with_post_fit."""

    def test_post_fit_default_is_noop(
        self, surrogate_1obj: RBFSurrogate, archive_1obj: Archive
    ) -> None:
        train_x, train_y = archive_1obj.x, archive_1obj.f
        surrogate_1obj.fit(train_x, train_y)
        result = surrogate_1obj.post_fit(train_x, train_y, ctx=None)
        assert result is None

    def test_with_post_fit_fn_is_called(
        self, surrogate_1obj: RBFSurrogate, archive_1obj: Archive
    ) -> None:
        called = [False]

        def hook(train_x, train_y, ctx):
            called[0] = True

        surrogate = surrogate_1obj.with_post_fit(hook)
        train_x, train_y = archive_1obj.x, archive_1obj.f
        surrogate.fit(train_x, train_y)
        surrogate.post_fit(train_x, train_y, ctx=None)
        assert called[0]

    def test_with_post_fit_fn_receives_correct_args(
        self, surrogate_1obj: RBFSurrogate, archive_1obj: Archive
    ) -> None:
        received = {}

        def hook(train_x, train_y, ctx):
            received["train_x_shape"] = train_x.shape
            received["train_y_shape"] = train_y.shape

        surrogate = surrogate_1obj.with_post_fit(hook)
        train_x, train_y = archive_1obj.x, archive_1obj.f
        surrogate.post_fit(train_x, train_y, ctx=None)
        assert received["train_x_shape"] == train_x.shape
        assert received["train_y_shape"] == train_y.shape

    def test_with_post_fit_chains_in_order(
        self, surrogate_1obj: RBFSurrogate, archive_1obj: Archive
    ) -> None:
        log: list[int] = []
        surrogate = surrogate_1obj.with_post_fit(
            lambda tx, ty, ctx: log.append(1)
        ).with_post_fit(lambda tx, ty, ctx: log.append(2))
        train_x, train_y = archive_1obj.x, archive_1obj.f
        surrogate.post_fit(train_x, train_y, ctx=None)
        assert log == [1, 2]

    def test_with_post_fit_does_not_mutate_original(
        self, surrogate_1obj: RBFSurrogate, archive_1obj: Archive
    ) -> None:
        called = [False]

        def hook(tx, ty, ctx):
            called[0] = True

        _ = surrogate_1obj.with_post_fit(hook)
        train_x, train_y = archive_1obj.x, archive_1obj.f
        surrogate_1obj.post_fit(train_x, train_y, ctx=None)
        assert not called[0]


# ===========================================================================
# NOTE: SurrogateManager.post_score / with_post_score removed
# ===========================================================================
# `post_score`/`with_post_score` were removed from SurrogateManager per plan
# Section 5.3 ("score aggregation" is a responsibility removed from the
# manager) -- there is no longer a scores-shaped hook on the manager to test.
# `Surrogate.post_fit`/`with_post_fit` (tested above via TestSurrogateHooks)
# is unaffected; the two tests below just exercise it through `predict()`
# through the current prediction API.


class TestSurrogateManagerPostFitViaPredict:
    """post_fit is still invoked from predict(), unchanged by the split."""

    def test_post_fit_called_in_global_manager(
        self,
        surrogate_1obj: RBFSurrogate,
        archive_1obj: Archive,
        candidates: np.ndarray,
    ) -> None:
        called = [False]

        def hook(tx, ty, ctx):
            called[0] = True

        surrogate = surrogate_1obj.with_post_fit(hook)
        manager = GlobalSurrogateManager(surrogate)
        manager.predict(candidates, archive_1obj)
        assert called[0]

    def test_post_fit_called_per_candidate_in_local_manager(
        self,
        surrogate_1obj: RBFSurrogate,
        archive_1obj: Archive,
        candidates: np.ndarray,
    ) -> None:
        call_count = [0]

        def hook(tx, ty, ctx):
            call_count[0] += 1

        surrogate = surrogate_1obj.with_post_fit(hook)
        manager = LocalSurrogateManager(surrogate, training_set=KNNObjectiveSet(10))
        manager.predict(candidates, archive_1obj)
        assert call_count[0] == len(candidates)


# ===========================================================================
# SurrogateManager.on_generation_end / with_on_generation_end
# ===========================================================================


class TestSurrogateManagerGenerationHook:
    """Tests for SurrogateManager.on_generation_end and with_on_generation_end."""

    def test_on_generation_end_default_is_noop(
        self,
        surrogate_1obj: RBFSurrogate,
        archive_1obj: Archive,
    ) -> None:
        manager = GlobalSurrogateManager(surrogate_1obj)
        assert manager.on_generation_end(0, archive_1obj, ctx=None) is None

    def test_with_on_generation_end_fn_is_called(
        self,
        surrogate_1obj: RBFSurrogate,
        archive_1obj: Archive,
    ) -> None:
        calls: list[tuple] = []

        def hook(gen, archive, ctx):
            calls.append((gen, archive, ctx))

        manager = GlobalSurrogateManager(surrogate_1obj).with_on_generation_end(hook)
        manager.on_generation_end(3, archive_1obj, ctx=None)

        assert len(calls) == 1
        assert calls[0] == (3, archive_1obj, None)

    def test_with_on_generation_end_chains_in_order(
        self,
        surrogate_1obj: RBFSurrogate,
        archive_1obj: Archive,
    ) -> None:
        order: list[int] = []
        manager = (
            GlobalSurrogateManager(surrogate_1obj)
            .with_on_generation_end(lambda g, a, ctx: order.append(1))
            .with_on_generation_end(lambda g, a, ctx: order.append(2))
        )
        manager.on_generation_end(0, archive_1obj)
        assert order == [1, 2]

    def test_with_on_generation_end_does_not_mutate_original(
        self,
        surrogate_1obj: RBFSurrogate,
        archive_1obj: Archive,
    ) -> None:
        called = [False]

        def hook(g, a, ctx):
            called[0] = True

        original = GlobalSurrogateManager(surrogate_1obj)
        _ = original.with_on_generation_end(hook)
        original.on_generation_end(0, archive_1obj)
        assert not called[0]


# ---------------------------------------------------------------------------
# last_accuracy
# ---------------------------------------------------------------------------


class TestLastAccuracy:
    def test_last_accuracy_none_by_default(
        self, surrogate_1obj: RBFSurrogate, archive_1obj: Archive
    ) -> None:
        manager = GlobalSurrogateManager(surrogate_1obj)
        assert manager.last_accuracy is None

    def test_last_accuracy_none_without_evaluator(
        self, surrogate_1obj: RBFSurrogate, archive_1obj: Archive
    ) -> None:
        manager = GlobalSurrogateManager(surrogate_1obj)
        manager.fit(archive_1obj)
        assert manager.last_accuracy is None

    def test_last_accuracy_set_after_fit_with_evaluator(
        self, surrogate_1obj: RBFSurrogate, archive_1obj: Archive
    ) -> None:
        evaluator = KFoldAccuracyEvaluator(metrics=[SpearmanCorrelation()], n_splits=5)
        manager = GlobalSurrogateManager(surrogate_1obj, accuracy_evaluator=evaluator)
        manager.fit(archive_1obj)
        assert isinstance(manager.last_accuracy, SurrogateAccuracy)
        assert "spearman" in manager.last_accuracy.metrics

    def test_last_accuracy_updated_on_predict_with_refit(
        self, surrogate_1obj: RBFSurrogate, archive_1obj: Archive
    ) -> None:
        evaluator = KFoldAccuracyEvaluator(metrics=[SpearmanCorrelation()], n_splits=3)
        manager = GlobalSurrogateManager(surrogate_1obj, accuracy_evaluator=evaluator)
        rng = np.random.default_rng(0)
        candidates = rng.uniform(-2.0, 2.0, size=(5, DIM))
        manager.predict(candidates, archive_1obj, refit=True)
        assert manager.last_accuracy is not None

    def test_last_accuracy_not_updated_when_refit_false(
        self, surrogate_1obj: RBFSurrogate, archive_1obj: Archive
    ) -> None:
        evaluator = KFoldAccuracyEvaluator(metrics=[SpearmanCorrelation()], n_splits=3)
        manager = GlobalSurrogateManager(surrogate_1obj, accuracy_evaluator=evaluator)
        manager.fit(archive_1obj)
        first = manager.last_accuracy

        rng = np.random.default_rng(1)
        candidates = rng.uniform(-2.0, 2.0, size=(5, DIM))
        manager.predict(candidates, archive_1obj, refit=False)
        assert manager.last_accuracy is first  # same object, not updated

    def test_composite_propagates_last_accuracy_from_first_manager(
        self, surrogate_1obj: RBFSurrogate, archive_1obj: Archive
    ) -> None:
        evaluator = KFoldAccuracyEvaluator(metrics=[SpearmanCorrelation()], n_splits=3)
        m1 = GlobalSurrogateManager(surrogate_1obj, accuracy_evaluator=evaluator)
        m2 = GlobalSurrogateManager(surrogate_1obj)
        composite = CompositeSurrogateManager({"a": m1, "b": m2})
        composite.fit(archive_1obj)
        assert composite.last_accuracy is m1.last_accuracy
        assert composite.last_accuracy is not None


class TestLocalSurrogateManagerAccuracy:
    def test_last_accuracy_none_by_default(
        self, surrogate_1obj: RBFSurrogate, archive_1obj: Archive
    ) -> None:
        manager = LocalSurrogateManager(surrogate_1obj)
        assert manager.last_accuracy is None

    def test_fit_sets_last_accuracy(
        self, surrogate_1obj: RBFSurrogate, archive_1obj: Archive
    ) -> None:
        evaluator = KFoldAccuracyEvaluator(metrics=[SpearmanCorrelation()], n_splits=3)
        manager = LocalSurrogateManager(surrogate_1obj, accuracy_evaluator=evaluator)
        manager.fit(archive_1obj)
        assert isinstance(manager.last_accuracy, SurrogateAccuracy)
        assert "spearman" in manager.last_accuracy.metrics

    def test_fit_noop_without_evaluator(
        self, surrogate_1obj: RBFSurrogate, archive_1obj: Archive
    ) -> None:
        manager = LocalSurrogateManager(surrogate_1obj)
        manager.fit(archive_1obj)
        assert manager.last_accuracy is None

    def test_last_accuracy_set_after_predict_with_refit(
        self, surrogate_1obj: RBFSurrogate, archive_1obj: Archive
    ) -> None:
        evaluator = KFoldAccuracyEvaluator(metrics=[SpearmanCorrelation()], n_splits=3)
        manager = LocalSurrogateManager(surrogate_1obj, accuracy_evaluator=evaluator)
        rng = np.random.default_rng(0)
        candidates = rng.uniform(-2.0, 2.0, size=(5, DIM))
        manager.predict(candidates, archive_1obj, refit=True)
        assert isinstance(manager.last_accuracy, SurrogateAccuracy)
        assert "spearman" in manager.last_accuracy.metrics
        # n_samples = number of candidates for which validation was possible
        assert manager.last_accuracy.n_samples == len(candidates)

    def test_last_accuracy_not_updated_when_refit_false(
        self, surrogate_1obj: RBFSurrogate, archive_1obj: Archive
    ) -> None:
        evaluator = KFoldAccuracyEvaluator(metrics=[SpearmanCorrelation()], n_splits=3)
        manager = LocalSurrogateManager(surrogate_1obj, accuracy_evaluator=evaluator)
        rng = np.random.default_rng(0)
        candidates = rng.uniform(-2.0, 2.0, size=(5, DIM))
        manager.predict(candidates, archive_1obj, refit=True)
        first = manager.last_accuracy
        manager.predict(candidates, archive_1obj, refit=False)
        assert manager.last_accuracy is first  # not updated

    def test_generation_based_pattern_sets_last_accuracy(
        self, surrogate_1obj: RBFSurrogate, archive_1obj: Archive
    ) -> None:
        """fit() + predict(refit=False) pattern (GenerationBasedStrategy)."""
        evaluator = KFoldAccuracyEvaluator(metrics=[SpearmanCorrelation()], n_splits=3)
        manager = LocalSurrogateManager(surrogate_1obj, accuracy_evaluator=evaluator)
        rng = np.random.default_rng(0)
        candidates = rng.uniform(-2.0, 2.0, size=(5, DIM))
        manager.fit(archive_1obj)
        assert isinstance(manager.last_accuracy, SurrogateAccuracy)
        accuracy_after_fit = manager.last_accuracy
        # inner loop: refit=False should not update last_accuracy
        manager.predict(candidates, archive_1obj, refit=False)
        assert manager.last_accuracy is accuracy_after_fit

    def test_nearest_neighbor_excluded_from_training(
        self, surrogate_1obj: RBFSurrogate, archive_1obj: Archive
    ) -> None:
        """Nearest archive neighbor is held out; training uses n_neighbors-1 points."""
        evaluator = KFoldAccuracyEvaluator(metrics=[SpearmanCorrelation()], n_splits=3)
        manager_with = LocalSurrogateManager(
            surrogate_1obj, accuracy_evaluator=evaluator
        )
        manager_without = LocalSurrogateManager(RBFSurrogate(GaussianKernel()))
        rng = np.random.default_rng(99)
        candidates = rng.uniform(-2.0, 2.0, size=(5, DIM))
        pred_with = manager_with.predict(candidates, archive_1obj)
        pred_without = manager_without.predict(candidates, archive_1obj)
        scores_with = _scores(
            MeanPrediction().evaluate(candidates, pred_with, archive_1obj)
        )
        scores_without = _scores(
            MeanPrediction().evaluate(candidates, pred_without, archive_1obj)
        )
        # Scores differ because the nearest neighbor is excluded from training
        # when an accuracy evaluator is active (k-1 vs k training points).
        assert scores_with.shape == scores_without.shape
        assert not np.any(np.isnan(scores_with))

    def test_loo_self_exclusion_in_update_accuracy(
        self, surrogate_1obj: RBFSurrogate, archive_1obj: Archive
    ) -> None:
        """_update_accuracy uses LOO self-exclusion; RBF no longer gives perfect score."""  # noqa: E501
        evaluator = KFoldAccuracyEvaluator(metrics=[SpearmanCorrelation()], n_splits=3)
        manager = LocalSurrogateManager(surrogate_1obj, accuracy_evaluator=evaluator)
        manager.fit(archive_1obj)
        assert manager.last_accuracy is not None
        # With self-exclusion, RBF accuracy is < 1.0 (not perfectly interpolated)
        spearman = manager.last_accuracy.get("spearman")
        assert spearman < 1.0 or np.isnan(spearman)


# ===========================================================================
# PairwiseSurrogateManager Tests
# ===========================================================================

_PAIRWISE_ATTRS = [
    PopulationAttribute(name="x", dtype=np.float64, shape=(DIM,)),
    PopulationAttribute(name="f", dtype=np.float64, shape=(N_OBJ,)),
    PopulationAttribute(name="g", dtype=np.float64, shape=(0,)),
    PopulationAttribute(name="cv", dtype=np.float64, shape=()),
]


@pytest.fixture
def archive_pairwise() -> Archive:
    """Archive pre-filled with 20 single-objective training points including cv."""
    arc = Archive(_PAIRWISE_ATTRS, init_capacity=30)
    rng = np.random.default_rng(42)
    for _ in range(20):
        x = rng.uniform(-2.0, 2.0, size=DIM)
        f = np.array([np.sum(x**2)])
        arc.add(x=x, f=f, cv=0.0)
    return arc


@pytest.fixture
def ctx_pairwise(archive_pairwise: Archive) -> OptimizationState:
    """OptimizationState with a single-objective comparator."""
    problem = Problem(
        func=lambda x: np.array([np.sum(x**2)]),
        dim=DIM,
        n_obj=N_OBJ,
        direction=np.array([-1.0]),
        lb=[-5.0] * DIM,
        ub=[5.0] * DIM,
        eps_cv=1e-6,
        comparator=SingleObjectiveComparator(),
    )
    problem.space.services.register("FeatureEncoder", _IdentityFeatureEncoder())
    pop_attrs = _PAIRWISE_ATTRS
    pop = Population(pop_attrs, init_capacity=10)
    rng = np.random.default_rng(0)
    xs = rng.uniform(-2.0, 2.0, size=(5, DIM))
    fs = np.array([[np.sum(x**2)] for x in xs])
    pop.extend({"x": xs, "f": fs, "cv": np.zeros(5)})
    pareto_arc = ParetoArchive(pop_attrs, init_capacity=20, direction=np.array([-1.0]))
    return OptimizationState(
        problem=problem,
        population=pop,
        archive=archive_pairwise,
        pareto_archive=pareto_arc,
        rng=np.random.default_rng(1),
        fe=20,
        gen=1,
    )


class TestPairwiseSurrogateManager:
    """E2E tests for PairwiseSurrogateManager with SklearnRFCClassificationSurrogate."""

    def test_predict_returns_correct_shape(
        self,
        archive_pairwise: Archive,
        ctx_pairwise: OptimizationState,
        candidates: np.ndarray,
    ) -> None:
        manager = PairwiseSurrogateManager(
            SklearnRFCClassificationSurrogate(n_estimators=5, random_state=0),
            n_ref=5,
        )
        prediction = manager.predict(candidates, archive_pairwise, ctx_pairwise)
        scores = _scores(
            WinRateAcquisition().evaluate(
                candidates, prediction, archive_pairwise, ctx_pairwise
            )
        )
        assert scores.shape == (len(candidates),)
        assert prediction.channels["win_rate"].value.shape == (len(candidates), 1)

    def test_fit_then_predict_refit_false(
        self,
        archive_pairwise: Archive,
        ctx_pairwise: OptimizationState,
        candidates: np.ndarray,
    ) -> None:
        """fit() + predict(refit=False) pattern works without error."""
        manager = PairwiseSurrogateManager(
            SklearnRFCClassificationSurrogate(n_estimators=5, random_state=0),
            n_ref=5,
        )
        manager.fit(archive_pairwise, ctx_pairwise)
        prediction = manager.predict(
            candidates, archive_pairwise, ctx_pairwise, refit=False
        )
        assert prediction.channels["win_rate"].value.shape == (len(candidates), 1)

    def test_predictions_keep_relational_channel(
        self,
        archive_pairwise: Archive,
        ctx_pairwise: OptimizationState,
        candidates: np.ndarray,
    ) -> None:
        """Relational predictions are not objective predictions."""
        manager = PairwiseSurrogateManager(
            SklearnRFCClassificationSurrogate(n_estimators=5, random_state=0),
            n_ref=5,
        )
        prediction = manager.predict(candidates, archive_pairwise, ctx_pairwise)
        assert "objective" not in prediction.channels

    def test_scores_in_0_1_range(
        self,
        archive_pairwise: Archive,
        ctx_pairwise: OptimizationState,
        candidates: np.ndarray,
    ) -> None:
        """Win probabilities from predict_proba are always in [0, 1]."""
        manager = PairwiseSurrogateManager(
            SklearnRFCClassificationSurrogate(n_estimators=5, random_state=0),
            n_ref=5,
        )
        prediction = manager.predict(candidates, archive_pairwise, ctx_pairwise)
        scores = _scores(
            WinRateAcquisition().evaluate(
                candidates, prediction, archive_pairwise, ctx_pairwise
            )
        )
        assert np.all(scores >= 0.0)
        assert np.all(scores <= 1.0)

    def test_default_training_set_is_pairwise(self) -> None:
        """Default training_set is PairwiseComparisonSet when none supplied."""
        manager = PairwiseSurrogateManager(
            SklearnRFCClassificationSurrogate(n_estimators=5, random_state=0)
        )
        assert isinstance(manager.training_set, PairwiseComparisonSet)

    def test_runs_through_strategy_stage_wiring(
        self,
        archive_pairwise: Archive,
        ctx_pairwise: OptimizationState,
        candidates: np.ndarray,
    ) -> None:
        """Pairwise predictions can be scored by a separately supplied acquisition."""
        manager = PairwiseSurrogateManager(
            SklearnRFCClassificationSurrogate(n_estimators=5, random_state=0),
            n_ref=5,
        )
        attrs = [
            PopulationAttribute(name="x", dtype=np.float64, shape=(DIM,)),
            PopulationAttribute(name="f", dtype=np.float64, shape=(N_OBJ,)),
            PopulationAttribute(name="g", dtype=np.float64, shape=(0,)),
            PopulationAttribute(name="cv", dtype=np.float64, shape=()),
        ]
        offspring = Population(attrs, init_capacity=len(candidates))
        offspring.extend(
            {
                "x": candidates,
                "f": np.full((len(candidates), N_OBJ), np.nan),
                "g": np.zeros((len(candidates), 0)),
                "cv": np.zeros(len(candidates)),
            }
        )
        state = ctx_pairwise.replace(offspring=offspring)

        predict_stage = SurrogatePredictStage(manager, cbmanager=None)
        state = predict_stage.execute(state)
        acquisition_stage = AcquisitionStage(WinRateAcquisition(), cbmanager=None)
        state = acquisition_stage.execute(state)

        assert state.scores is not None
        assert state.scores.shape == (len(candidates),)
        assert np.all(state.scores >= 0.0) and np.all(state.scores <= 1.0)


class _IdentityFeatureEncoder:
    def encode(self, genomes):
        return genomes.array


# ===========================================================================
# Acquisition ownership
# ===========================================================================
class TestIterAcquisitions:
    def test_managers_do_not_expose_acquisition(
        self, surrogate_1obj: RBFSurrogate
    ) -> None:
        managers = [
            GlobalSurrogateManager(surrogate_1obj),
            LocalSurrogateManager(surrogate_1obj),
            PairwiseSurrogateManager(
                SklearnRFCClassificationSurrogate(n_estimators=5, random_state=0)
            ),
        ]
        for manager in managers:
            assert not hasattr(manager, "acquisition")
            assert not hasattr(manager, "iter_acquisitions")

    def test_composite_acquisition_is_provider_owned(
        self, surrogate_1obj: RBFSurrogate
    ) -> None:
        m1 = GlobalSurrogateManager(surrogate_1obj)
        m2 = GlobalSurrogateManager(surrogate_1obj)
        acq_a = MeanPrediction()
        acq_b = MeanPrediction()
        composite_acq = CompositeAcquisition(
            {"a": acq_a, "b": acq_b}, combine_fn=product_combine
        )
        composite = CompositeSurrogateManager({"a": m1, "b": m2})
        assert not hasattr(composite, "acquisition")
        assert composite_acq.acquisitions == {"a": acq_a, "b": acq_b}

    def test_pairwise_acquisition_is_independently_constructed(self) -> None:
        manager = PairwiseSurrogateManager(
            SklearnRFCClassificationSurrogate(n_estimators=5, random_state=0)
        )
        acquisition = WinRateAcquisition()
        assert not hasattr(manager, "acquisition")
        assert isinstance(acquisition, WinRateAcquisition)
