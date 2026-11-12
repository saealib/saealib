"""
Tests for SklearnGPRSurrogate.

Covers:
- SklearnGPRSurrogate: fit/predict, single/multi-objective, std population
- provides_uncertainty flag
- Custom kernel
- Integration with EI, LCB, MaxUncertainty, PoF acquisition functions
- Integration with GlobalSurrogateManager
"""

import numpy as np
import pytest

from saealib.acquisition import (
    ExpectedImprovement,
    LowerConfidenceBound,
    MaxUncertainty,
    ProbabilityOfFeasibility,
)
from saealib.population import Archive, PopulationAttribute
from saealib.surrogate.manager import GlobalSurrogateManager
from saealib.surrogate.prediction import SurrogatePrediction
from saealib.surrogate.sklearn_surrogate import SklearnGPRSurrogate

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
def fitted_gp(train_data_1obj):
    X, y = train_data_1obj
    gp = SklearnGPRSurrogate()
    gp.fit(X, y)
    return gp


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
# SklearnGPRSurrogate Tests
# ===========================================================================
class TestSklearnGPRSurrogate:
    def test_fit_predict_1obj(self, train_data_1obj, test_x) -> None:
        X, y = train_data_1obj
        gp = SklearnGPRSurrogate()
        gp.fit(X, y)
        pred = gp.predict(test_x)
        assert isinstance(pred, SurrogatePrediction)
        assert pred.value.shape == (5, 1)

    def test_fit_predict_2obj(self, train_data_2obj, test_x) -> None:
        X, y = train_data_2obj
        gp = SklearnGPRSurrogate()
        gp.fit(X, y)
        pred = gp.predict(test_x)
        assert pred.value.shape == (5, 2)

    def test_predict_1d_input(self, fitted_gp, train_data_1obj) -> None:
        X, _ = train_data_1obj
        pred = fitted_gp.predict(X[0])
        assert pred.value.shape == (1, 1)

    def test_std_populated(self, fitted_gp, test_x) -> None:
        pred = fitted_gp.predict(test_x)
        assert pred.std is not None

    def test_std_shape_matches_value(self, fitted_gp, test_x) -> None:
        pred = fitted_gp.predict(test_x)
        assert pred.std.shape == pred.value.shape

    def test_std_shape_2obj(self, train_data_2obj, test_x) -> None:
        X, y = train_data_2obj
        gp = SklearnGPRSurrogate()
        gp.fit(X, y)
        pred = gp.predict(test_x)
        assert pred.std is not None
        assert pred.std.shape == (5, 2)

    def test_std_nonnegative(self, fitted_gp, test_x) -> None:
        pred = fitted_gp.predict(test_x)
        assert pred.std is not None
        assert np.all(pred.std >= 0.0)

    def test_provides_uncertainty_class_attribute(self) -> None:
        assert SklearnGPRSurrogate.provides_uncertainty is True

    def test_has_uncertainty_true(self, fitted_gp, test_x) -> None:
        pred = fitted_gp.predict(test_x)
        assert pred.has_uncertainty is True

    def test_custom_kernel(self, train_data_1obj, test_x) -> None:
        from sklearn.gaussian_process.kernels import Matern

        X, y = train_data_1obj
        gp = SklearnGPRSurrogate(kernel=Matern(nu=2.5))
        gp.fit(X, y)
        pred = gp.predict(test_x)
        assert pred.value.shape == (5, 1)
        assert pred.std is not None

    def test_refit_different_nobj(
        self, train_data_1obj, train_data_2obj, test_x
    ) -> None:
        X1, y1 = train_data_1obj
        X2, y2 = train_data_2obj
        gp = SklearnGPRSurrogate()
        gp.fit(X1, y1)
        gp.fit(X2, y2)
        pred = gp.predict(test_x)
        assert pred.value.shape == (5, 2)
        assert pred.std is not None
        assert pred.std.shape == (5, 2)

    def test_models_are_cloned_per_objective(self, train_data_2obj) -> None:
        X, y = train_data_2obj
        gp = SklearnGPRSurrogate()
        gp.fit(X, y)
        assert gp._models is not None
        assert gp._models[0] is not gp._models[1]


# ===========================================================================
# SklearnGPRSurrogate + Acquisition Function Integration Tests
# ===========================================================================
class TestSklearnGPRSurrogateWithAcquisition:
    def test_ei_integration(self, fitted_gp, test_x) -> None:
        pred = fitted_gp.predict(test_x)
        scores = ExpectedImprovement().score(pred, reference=1.0)
        assert scores.shape == (5,)
        assert np.all(scores >= 0.0)

    def test_lcb_integration(self, fitted_gp, test_x) -> None:
        pred = fitted_gp.predict(test_x)
        scores = LowerConfidenceBound().score(pred, reference=None)
        assert scores.shape == (5,)

    def test_max_uncertainty_integration(self, fitted_gp, test_x) -> None:
        pred = fitted_gp.predict(test_x)
        scores = MaxUncertainty().score(pred, reference=None)
        assert scores.shape == (5,)
        assert np.all(scores >= 0.0)

    def test_pof_integration(self, fitted_gp, test_x) -> None:
        pred = fitted_gp.predict(test_x)
        scores = ProbabilityOfFeasibility().score(pred, reference=None)
        assert scores.shape == (5,)
        assert np.all(scores >= 0.0)
        assert np.all(scores <= 1.0)


# ===========================================================================
# Integration: GlobalSurrogateManager
# ===========================================================================
class TestSklearnGPRSurrogateManager:
    def test_global_manager_ei(self, archive_1obj, candidates) -> None:
        acquisition = ExpectedImprovement()
        manager = GlobalSurrogateManager(SklearnGPRSurrogate())
        prediction = manager.predict(candidates, archive_1obj, ctx=None)
        scores = acquisition.evaluate(
            candidates, prediction, archive_1obj, ctx=None
        ).scores
        assert scores is not None
        assert scores.shape == (len(candidates),)
        assert np.all(scores >= 0.0)
        assert prediction.value.shape == (len(candidates), 1)
        assert prediction.std is not None

    def test_global_manager_lcb(self, archive_1obj, candidates) -> None:
        acquisition = LowerConfidenceBound()
        manager = GlobalSurrogateManager(SklearnGPRSurrogate())
        prediction = manager.predict(candidates, archive_1obj, ctx=None)
        scores = acquisition.evaluate(
            candidates, prediction, archive_1obj, ctx=None
        ).scores
        assert scores is not None
        assert scores.shape == (len(candidates),)

    def test_global_manager_max_uncertainty(self, archive_1obj, candidates) -> None:
        acquisition = MaxUncertainty()
        manager = GlobalSurrogateManager(SklearnGPRSurrogate())
        prediction = manager.predict(candidates, archive_1obj, ctx=None)
        scores = acquisition.evaluate(
            candidates, prediction, archive_1obj, ctx=None
        ).scores
        assert scores is not None
        assert scores.shape == (len(candidates),)


# ===========================================================================
# GPSurrogate removed alias
# ===========================================================================
class TestGPSurrogateRemoved:
    def test_deprecated_module_no_longer_exists(self) -> None:
        import importlib

        with pytest.raises(ModuleNotFoundError):
            importlib.import_module("saealib.surrogate._deprecated")

    def test_not_accessible_from_surrogate_package(self) -> None:
        import saealib.surrogate

        assert not hasattr(saealib.surrogate, "GPSurrogate")

    def test_not_accessible_from_top_level(self) -> None:
        import saealib

        assert not hasattr(saealib, "GPSurrogate")
