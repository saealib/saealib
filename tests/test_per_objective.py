"""
Tests for PerObjectiveSurrogate.

Covers:
- Empty surrogates raises ValueError
- fit/predict with single surrogate (1-obj)
- fit/predict with multiple surrogates (2-obj, mixed types)
- n_obj mismatch raises ValueError at fit time
- provides_uncertainty: False when any surrogate lacks it
- provides_uncertainty: True when all surrogates provide it, std is populated
- Integration with GlobalSurrogateManager
"""

import numpy as np
import pytest

from saealib.acquisition import MeanPrediction
from saealib.population import Archive, PopulationAttribute
from saealib.surrogate.base import Surrogate
from saealib.surrogate.manager import GlobalSurrogateManager
from saealib.surrogate.per_objective import PerObjectiveSurrogate
from saealib.surrogate.prediction import SurrogatePrediction
from saealib.surrogate.rbf import RBFSurrogate
from saealib.surrogate.rbf_kernels import GaussianKernel
from saealib.surrogate.sklearn_surrogate import SklearnRFRSurrogate, SklearnSVMSurrogate

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
DIM = 2
N_SAMPLES = 20


@pytest.fixture
def train_data_2obj():
    rng = np.random.default_rng(1)
    X = rng.uniform(0.0, 2.0, size=(N_SAMPLES, DIM))
    y = np.column_stack([np.sum(X**2, axis=1), np.sum((X - 2.0) ** 2, axis=1)])
    return X, y


@pytest.fixture
def train_data_1obj():
    rng = np.random.default_rng(0)
    X = rng.uniform(-2.0, 2.0, size=(N_SAMPLES, DIM))
    y = np.sum(X**2, axis=1)
    return X, y


@pytest.fixture
def test_x():
    rng = np.random.default_rng(7)
    return rng.uniform(-1.0, 1.0, size=(5, DIM))


@pytest.fixture
def archive_2obj():
    attrs = [
        PopulationAttribute(name="x", dtype=np.float64, shape=(DIM,)),
        PopulationAttribute(name="f", dtype=np.float64, shape=(2,)),
    ]
    arc = Archive(attrs, init_capacity=30)
    rng = np.random.default_rng(0)
    for _ in range(N_SAMPLES):
        x = rng.uniform(0.0, 2.0, size=DIM)
        f = np.array([np.sum(x**2), np.sum((x - 2.0) ** 2)])
        arc.add(x=x, f=f)
    return arc


@pytest.fixture
def candidates():
    rng = np.random.default_rng(7)
    return rng.uniform(-1.0, 1.0, size=(5, DIM))


# ---------------------------------------------------------------------------
# Stub surrogate with uncertainty
# ---------------------------------------------------------------------------
class _StubSurrogateWithUncertainty(Surrogate):
    """Stub that always returns a fixed mean and std."""

    provides_uncertainty = True

    def fit(self, train_x, train_y) -> None:
        pass

    def predict(self, test_x) -> SurrogatePrediction:
        test = np.asarray(test_x)
        if test.ndim == 1:
            test = test.reshape(1, -1)
        n = len(test)
        return SurrogatePrediction.objective(
            value=np.ones((n, 1)),
            std=np.full((n, 1), 0.1),
        )


# ===========================================================================
# PerObjectiveSurrogate Tests
# ===========================================================================
class TestPerObjectiveSurrogate:
    def test_empty_surrogates_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            PerObjectiveSurrogate([])

    def test_fit_predict_1obj_single_surrogate(self, train_data_1obj, test_x) -> None:
        X, y = train_data_1obj
        s = PerObjectiveSurrogate([RBFSurrogate(GaussianKernel())])
        s.fit(X, y)
        pred = s.predict(test_x)
        assert pred.value.shape == (5, 1)

    def test_fit_predict_2obj_two_surrogates(self, train_data_2obj, test_x) -> None:
        X, y = train_data_2obj
        s = PerObjectiveSurrogate(
            [RBFSurrogate(GaussianKernel()), RBFSurrogate(GaussianKernel())]
        )
        s.fit(X, y)
        pred = s.predict(test_x)
        assert pred.value.shape == (5, 2)

    def test_mixed_surrogate_types(self, train_data_2obj, test_x) -> None:
        X, y = train_data_2obj
        s = PerObjectiveSurrogate(
            [SklearnSVMSurrogate(), SklearnRFRSurrogate(n_estimators=5, random_state=0)]
        )
        s.fit(X, y)
        pred = s.predict(test_x)
        assert pred.value.shape == (5, 2)

    def test_nobj_mismatch_raises(self, train_data_2obj) -> None:
        X, y = train_data_2obj
        s = PerObjectiveSurrogate([RBFSurrogate(GaussianKernel())])
        with pytest.raises(ValueError, match="n_obj=2"):
            s.fit(X, y)

    def test_predict_1d_input(self, train_data_1obj) -> None:
        X, y = train_data_1obj
        s = PerObjectiveSurrogate([RBFSurrogate(GaussianKernel())])
        s.fit(X, y)
        pred = s.predict(X[0])
        assert pred.value.shape == (1, 1)

    def test_std_none_when_no_uncertainty(self, train_data_2obj, test_x) -> None:
        X, y = train_data_2obj
        s = PerObjectiveSurrogate(
            [RBFSurrogate(GaussianKernel()), RBFSurrogate(GaussianKernel())]
        )
        s.fit(X, y)
        assert s.predict(test_x).std is None

    def test_provides_uncertainty_false_if_any_missing(self) -> None:
        s = PerObjectiveSurrogate(
            [_StubSurrogateWithUncertainty(), RBFSurrogate(GaussianKernel())]
        )
        assert s.provides_uncertainty is False

    def test_provides_uncertainty_true_if_all_provide(self) -> None:
        s = PerObjectiveSurrogate(
            [_StubSurrogateWithUncertainty(), _StubSurrogateWithUncertainty()]
        )
        assert s.provides_uncertainty is True

    def test_std_populated_when_all_provide_uncertainty(self, test_x) -> None:
        rng = np.random.default_rng(0)
        X = rng.uniform(-1.0, 1.0, size=(10, DIM))
        y = np.column_stack([np.ones(10), np.zeros(10)])

        s = PerObjectiveSurrogate(
            [_StubSurrogateWithUncertainty(), _StubSurrogateWithUncertainty()]
        )
        s.fit(X, y)
        pred = s.predict(test_x)
        assert pred.std is not None
        assert pred.std.shape == (5, 2)

    def test_1d_train_y_accepted(self, train_data_1obj, test_x) -> None:
        X, y = train_data_1obj
        assert y.ndim == 1
        s = PerObjectiveSurrogate([RBFSurrogate(GaussianKernel())])
        s.fit(X, y)
        assert s.predict(test_x).value.shape == (5, 1)

    def test_x_populated_from_test_x_directly(self, train_data_2obj, test_x) -> None:
        """prediction.x must be set from test_x regardless of the wrapped
        surrogates, so acquisition functions like CORSDistance still work
        when an RBFSurrogate is wrapped in PerObjectiveSurrogate."""
        X, y = train_data_2obj
        s = PerObjectiveSurrogate(
            [RBFSurrogate(GaussianKernel()), RBFSurrogate(GaussianKernel())]
        )
        s.fit(X, y)
        pred = s.predict(test_x)
        assert pred.x is not None
        np.testing.assert_array_equal(pred.x, test_x)

    def test_x_populated_even_when_wrapped_surrogates_lack_it(self, test_x) -> None:
        """SklearnSVMSurrogate doesn't populate prediction.x itself; the wrapper
        must still set it from its own test_x argument."""
        rng = np.random.default_rng(0)
        X = rng.uniform(-1.0, 1.0, size=(10, DIM))
        y = np.ones(10)
        s = PerObjectiveSurrogate([SklearnSVMSurrogate()])
        s.fit(X, y)
        pred = s.predict(test_x)
        assert pred.x is not None
        np.testing.assert_array_equal(pred.x, test_x)


# ===========================================================================
# Integration: GlobalSurrogateManager
# ===========================================================================
class TestPerObjectiveSurrogateIntegration:
    def test_global_manager_2obj(self, archive_2obj, candidates) -> None:
        s = PerObjectiveSurrogate(
            [
                SklearnSVMSurrogate(),
                SklearnRFRSurrogate(n_estimators=10, random_state=0),
            ]
        )
        weights = np.array([-1.0, -1.0])
        acquisition = MeanPrediction(weights=weights)
        manager = GlobalSurrogateManager(s)
        prediction = manager.predict(candidates, archive_2obj, ctx=None)
        scores = acquisition.evaluate(
            candidates, prediction, archive_2obj, ctx=None
        ).scores
        assert scores is not None
        assert scores.shape == (len(candidates),)
        assert prediction.value.shape == (len(candidates), 2)
