"""
Tests for sklearn-based surrogate adapters.

Covers:
- SklearnSurrogate: fit/predict, single/multi-objective, missing sklearn
- SklearnSVMSurrogate, SklearnNNSurrogate, SklearnRFRSurrogate: smoke tests
- SklearnXGBSurrogate, SklearnLGBMSurrogate: smoke tests
- Integration with GlobalSurrogateManager
"""

import sys

import numpy as np
import pytest

from saealib.acquisition import MeanPrediction
from saealib.population import Archive, PopulationAttribute
from saealib.surrogate.manager import GlobalSurrogateManager
from saealib.surrogate.prediction import SurrogatePrediction
from saealib.surrogate.sklearn_surrogate import (
    SklearnClassificationSurrogate,
    SklearnLGBMSurrogate,
    SklearnNNSurrogate,
    SklearnRFCClassificationSurrogate,
    SklearnRFRSurrogate,
    SklearnSurrogate,
    SklearnSVCClassificationSurrogate,
    SklearnSVMSurrogate,
    SklearnXGBSurrogate,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
DIM = 2
N_SAMPLES = 20


@pytest.fixture
def train_data_1obj():
    rng = np.random.default_rng(0)
    X = rng.uniform(-2.0, 2.0, size=(N_SAMPLES, DIM))
    y = np.sum(X**2, axis=1)
    return X, y


@pytest.fixture
def train_data_2obj():
    rng = np.random.default_rng(1)
    X = rng.uniform(0.0, 2.0, size=(N_SAMPLES, DIM))
    y = np.column_stack([np.sum(X**2, axis=1), np.sum((X - 2.0) ** 2, axis=1)])
    return X, y


@pytest.fixture
def test_x():
    rng = np.random.default_rng(7)
    return rng.uniform(-1.0, 1.0, size=(5, DIM))


@pytest.fixture
def archive_1obj():
    attrs = [
        PopulationAttribute(name="x", dtype=np.float64, shape=(DIM,)),
        PopulationAttribute(name="f", dtype=np.float64, shape=(1,)),
    ]
    arc = Archive(attrs, init_capacity=30)
    rng = np.random.default_rng(42)
    for _ in range(N_SAMPLES):
        x = rng.uniform(-2.0, 2.0, size=DIM)
        f = np.array([np.sum(x**2)])
        arc.add(x=x, f=f)
    return arc


@pytest.fixture
def candidates():
    rng = np.random.default_rng(7)
    return rng.uniform(-1.0, 1.0, size=(5, DIM))


# ===========================================================================
# SklearnSurrogate Tests
# ===========================================================================
class TestSklearnSurrogate:
    def test_fit_predict_1obj(self, train_data_1obj, test_x) -> None:
        from sklearn.linear_model import Ridge

        X, y = train_data_1obj
        s = SklearnSurrogate(Ridge())
        s.fit(X, y)
        pred = s.predict(test_x)
        assert isinstance(pred, SurrogatePrediction)
        assert pred.value.shape == (5, 1)

    def test_fit_predict_2obj(self, train_data_2obj, test_x) -> None:
        from sklearn.linear_model import Ridge

        X, y = train_data_2obj
        s = SklearnSurrogate(Ridge())
        s.fit(X, y)
        pred = s.predict(test_x)
        assert pred.value.shape == (5, 2)

    def test_predict_1d_input(self, train_data_1obj) -> None:
        from sklearn.linear_model import Ridge

        X, y = train_data_1obj
        s = SklearnSurrogate(Ridge())
        s.fit(X, y)
        pred = s.predict(X[0])  # 1D input
        assert pred.value.shape == (1, 1)

    def test_std_is_none(self, train_data_1obj, test_x) -> None:
        from sklearn.linear_model import Ridge

        X, y = train_data_1obj
        s = SklearnSurrogate(Ridge())
        s.fit(X, y)
        assert s.predict(test_x).std is None

    def test_refit_different_nobj(
        self, train_data_1obj, train_data_2obj, test_x
    ) -> None:
        from sklearn.linear_model import Ridge

        X1, y1 = train_data_1obj
        X2, y2 = train_data_2obj
        s = SklearnSurrogate(Ridge())
        s.fit(X1, y1)
        s.fit(X2, y2)
        pred = s.predict(test_x)
        assert pred.value.shape == (5, 2)

    def test_missing_sklearn_raises(self, monkeypatch) -> None:
        monkeypatch.setitem(sys.modules, "sklearn", None)
        monkeypatch.setitem(sys.modules, "sklearn.base", None)
        with pytest.raises(ImportError, match="scikit-learn"):
            from sklearn.linear_model import Ridge

            SklearnSurrogate(Ridge())

    def test_estimator_is_cloned_per_objective(self, train_data_2obj) -> None:
        from sklearn.linear_model import Ridge

        X, y = train_data_2obj
        estimator = Ridge()
        s = SklearnSurrogate(estimator)
        s.fit(X, y)
        assert s._models is not None
        assert s._models[0] is not s._models[1]
        assert s._models[0] is not estimator


# ===========================================================================
# SklearnSVMSurrogate Tests
# ===========================================================================
class TestSklearnSVMSurrogate:
    def test_fit_predict_1obj(self, train_data_1obj, test_x) -> None:
        s = SklearnSVMSurrogate()
        X, y = train_data_1obj
        s.fit(X, y)
        pred = s.predict(test_x)
        assert pred.value.shape == (5, 1)

    def test_fit_predict_2obj(self, train_data_2obj, test_x) -> None:
        s = SklearnSVMSurrogate()
        X, y = train_data_2obj
        s.fit(X, y)
        assert s.predict(test_x).value.shape == (5, 2)

    def test_kwargs_forwarded(self) -> None:
        from sklearn.svm import SVR

        s = SklearnSVMSurrogate(kernel="linear", C=10.0)
        assert isinstance(s.estimator, SVR)
        assert s.estimator.kernel == "linear"
        assert s.estimator.C == 10.0


# ===========================================================================
# SklearnNNSurrogate Tests
# ===========================================================================
class TestSklearnNNSurrogate:
    def test_fit_predict_1obj(self, train_data_1obj, test_x) -> None:
        s = SklearnNNSurrogate(max_iter=200, random_state=0)
        X, y = train_data_1obj
        s.fit(X, y)
        pred = s.predict(test_x)
        assert pred.value.shape == (5, 1)

    def test_fit_predict_2obj(self, train_data_2obj, test_x) -> None:
        s = SklearnNNSurrogate(max_iter=200, random_state=0)
        X, y = train_data_2obj
        s.fit(X, y)
        assert s.predict(test_x).value.shape == (5, 2)


# ===========================================================================
# SklearnRFRSurrogate Tests
# ===========================================================================
class TestSklearnRFRSurrogate:
    def test_fit_predict_1obj(self, train_data_1obj, test_x) -> None:
        s = SklearnRFRSurrogate(n_estimators=10, random_state=0)
        X, y = train_data_1obj
        s.fit(X, y)
        pred = s.predict(test_x)
        assert pred.value.shape == (5, 1)

    def test_fit_predict_2obj(self, train_data_2obj, test_x) -> None:
        s = SklearnRFRSurrogate(n_estimators=10, random_state=0)
        X, y = train_data_2obj
        s.fit(X, y)
        assert s.predict(test_x).value.shape == (5, 2)


# ===========================================================================
# SklearnXGBSurrogate Tests
# ===========================================================================
class TestSklearnXGBSurrogate:
    def test_fit_predict_1obj(self, train_data_1obj, test_x) -> None:
        s = SklearnXGBSurrogate(n_estimators=10, random_state=0, verbosity=0)
        X, y = train_data_1obj
        s.fit(X, y)
        pred = s.predict(test_x)
        assert pred.value.shape == (5, 1)

    def test_fit_predict_2obj(self, train_data_2obj, test_x) -> None:
        s = SklearnXGBSurrogate(n_estimators=10, random_state=0, verbosity=0)
        X, y = train_data_2obj
        s.fit(X, y)
        assert s.predict(test_x).value.shape == (5, 2)

    def test_missing_xgboost_raises(self, monkeypatch) -> None:
        monkeypatch.setitem(sys.modules, "xgboost", None)
        with pytest.raises(ImportError, match="xgboost"):
            SklearnXGBSurrogate()


# ===========================================================================
# SklearnLGBMSurrogate Tests
# ===========================================================================
class TestSklearnLGBMSurrogate:
    def test_fit_predict_1obj(self, train_data_1obj, test_x) -> None:
        s = SklearnLGBMSurrogate(n_estimators=10, random_state=0, verbose=-1)
        X, y = train_data_1obj
        s.fit(X, y)
        pred = s.predict(test_x)
        assert pred.value.shape == (5, 1)

    def test_fit_predict_2obj(self, train_data_2obj, test_x) -> None:
        s = SklearnLGBMSurrogate(n_estimators=10, random_state=0, verbose=-1)
        X, y = train_data_2obj
        s.fit(X, y)
        assert s.predict(test_x).value.shape == (5, 2)

    def test_missing_lightgbm_raises(self, monkeypatch) -> None:
        monkeypatch.setitem(sys.modules, "lightgbm", None)
        with pytest.raises(ImportError, match="lightgbm"):
            SklearnLGBMSurrogate()


# ===========================================================================
# Integration: GlobalSurrogateManager
# ===========================================================================
class TestSklearnSurrogateIntegration:
    def test_global_manager_svm(self, archive_1obj, candidates) -> None:
        acquisition = MeanPrediction()
        manager = GlobalSurrogateManager(SklearnSVMSurrogate())
        prediction = manager.predict(candidates, archive_1obj, ctx=None)
        scores = acquisition.evaluate(
            candidates, prediction, archive_1obj, ctx=None
        ).scores
        assert scores is not None
        assert scores.shape == (len(candidates),)
        assert prediction.value.shape == (len(candidates), 1)

    def test_global_manager_rfr(self, archive_1obj, candidates) -> None:
        acquisition = MeanPrediction()
        manager = GlobalSurrogateManager(
            SklearnRFRSurrogate(n_estimators=10, random_state=0)
        )
        prediction = manager.predict(candidates, archive_1obj, ctx=None)
        scores = acquisition.evaluate(
            candidates, prediction, archive_1obj, ctx=None
        ).scores
        assert scores is not None
        assert scores.shape == (len(candidates),)

    def test_global_manager_xgb(self, archive_1obj, candidates) -> None:
        acquisition = MeanPrediction()
        manager = GlobalSurrogateManager(
            SklearnXGBSurrogate(n_estimators=10, verbosity=0)
        )
        prediction = manager.predict(candidates, archive_1obj, ctx=None)
        scores = acquisition.evaluate(
            candidates, prediction, archive_1obj, ctx=None
        ).scores
        assert scores is not None
        assert scores.shape == (len(candidates),)

    def test_global_manager_lgbm(self, archive_1obj, candidates) -> None:
        acquisition = MeanPrediction()
        manager = GlobalSurrogateManager(
            SklearnLGBMSurrogate(n_estimators=10, verbose=-1)
        )
        prediction = manager.predict(candidates, archive_1obj, ctx=None)
        scores = acquisition.evaluate(
            candidates, prediction, archive_1obj, ctx=None
        ).scores
        assert scores is not None
        assert scores.shape == (len(candidates),)


# ===========================================================================
# SklearnClassificationSurrogate Tests
# ===========================================================================


@pytest.fixture
def binary_labels(train_data_1obj: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    _, y = train_data_1obj
    return (y > np.median(y)).astype(float)


class TestSklearnClassificationSurrogate:
    def test_fit_predict_proba_shape(
        self,
        train_data_1obj: tuple[np.ndarray, np.ndarray],
        binary_labels: np.ndarray,
    ) -> None:
        from sklearn.ensemble import RandomForestClassifier

        X, _ = train_data_1obj
        sur = SklearnClassificationSurrogate(
            RandomForestClassifier(n_estimators=5, random_state=0)
        )
        sur.fit(X, binary_labels)
        pred = sur.predict_proba(X[:3])
        assert pred.value.shape == (3, 1)
        assert np.all(pred.value >= 0.0)
        assert np.all(pred.value <= 1.0)

    def test_predict_delegates_to_predict_proba(
        self,
        train_data_1obj: tuple[np.ndarray, np.ndarray],
        binary_labels: np.ndarray,
    ) -> None:
        from sklearn.ensemble import RandomForestClassifier

        X, _ = train_data_1obj
        sur = SklearnClassificationSurrogate(
            RandomForestClassifier(n_estimators=5, random_state=0)
        )
        sur.fit(X, binary_labels)
        np.testing.assert_array_equal(
            sur.predict(X[:3]).value, sur.predict_proba(X[:3]).value
        )

    def test_fit_twice_refits_model(
        self,
        train_data_1obj: tuple[np.ndarray, np.ndarray],
        binary_labels: np.ndarray,
    ) -> None:
        from sklearn.ensemble import RandomForestClassifier

        X, _ = train_data_1obj
        sur = SklearnClassificationSurrogate(
            RandomForestClassifier(n_estimators=5, random_state=0)
        )
        sur.fit(X, binary_labels)
        sur.fit(X, binary_labels)
        assert sur.predict_proba(X[:3]).value.shape == (3, 1)

    def test_predict_proba_1d_input(
        self,
        train_data_1obj: tuple[np.ndarray, np.ndarray],
        binary_labels: np.ndarray,
    ) -> None:
        from sklearn.ensemble import RandomForestClassifier

        X, _ = train_data_1obj
        sur = SklearnClassificationSurrogate(
            RandomForestClassifier(n_estimators=5, random_state=0)
        )
        sur.fit(X, binary_labels)
        pred = sur.predict_proba(X[0])
        assert pred.value.shape == (1, 1)


class TestSklearnRFCClassificationSurrogate:
    def test_fit_predict_proba(
        self,
        train_data_1obj: tuple[np.ndarray, np.ndarray],
        binary_labels: np.ndarray,
    ) -> None:
        X, _ = train_data_1obj
        sur = SklearnRFCClassificationSurrogate(n_estimators=5, random_state=0)
        sur.fit(X, binary_labels)
        pred = sur.predict_proba(X[:3])
        assert pred.value.shape == (3, 1)
        assert np.all(pred.value >= 0.0)
        assert np.all(pred.value <= 1.0)


class TestSklearnSVCClassificationSurrogate:
    def test_fit_predict_proba(
        self,
        train_data_1obj: tuple[np.ndarray, np.ndarray],
        binary_labels: np.ndarray,
    ) -> None:
        X, _ = train_data_1obj
        sur = SklearnSVCClassificationSurrogate(random_state=0)
        sur.fit(X, binary_labels)
        pred = sur.predict_proba(X[:3])
        assert pred.value.shape == (3, 1)
        assert np.all(pred.value >= 0.0)
        assert np.all(pred.value <= 1.0)
