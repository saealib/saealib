"""
Tests for TorchSurrogate.

Covers:
- fit/predict: single/multi-objective, 1D input
- std is None
- fit resets weights to initial state
- missing torch raises ImportError
- Integration with GlobalSurrogateManager
"""

import sys

import numpy as np
import pytest
import torch
import torch.nn as nn

from saealib.acquisition import MeanPrediction
from saealib.population import Archive, PopulationAttribute
from saealib.surrogate.manager import GlobalSurrogateManager
from saealib.surrogate.prediction import SurrogatePrediction
from saealib.surrogate.torch_surrogate import TorchSurrogate

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
DIM = 2
N_SAMPLES = 20


def _make_model_1obj() -> nn.Module:
    return nn.Sequential(nn.Linear(DIM, 16), nn.ReLU(), nn.Linear(16, 1))


def _make_model_2obj() -> nn.Module:
    return nn.Sequential(nn.Linear(DIM, 16), nn.ReLU(), nn.Linear(16, 2))


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
# TorchSurrogate Tests
# ===========================================================================
class TestTorchSurrogate:
    def test_fit_predict_1obj(self, train_data_1obj, test_x) -> None:
        X, y = train_data_1obj
        s = TorchSurrogate(_make_model_1obj(), epochs=10)
        s.fit(X, y)
        pred = s.predict(test_x)
        assert isinstance(pred, SurrogatePrediction)
        assert pred.value.shape == (5, 1)

    def test_fit_predict_2obj(self, train_data_2obj, test_x) -> None:
        X, y = train_data_2obj
        s = TorchSurrogate(_make_model_2obj(), epochs=10)
        s.fit(X, y)
        pred = s.predict(test_x)
        assert pred.value.shape == (5, 2)

    def test_predict_1d_input(self, train_data_1obj) -> None:
        X, y = train_data_1obj
        s = TorchSurrogate(_make_model_1obj(), epochs=10)
        s.fit(X, y)
        pred = s.predict(X[0])  # 1D input
        assert pred.value.shape == (1, 1)

    def test_std_is_none(self, train_data_1obj, test_x) -> None:
        X, y = train_data_1obj
        s = TorchSurrogate(_make_model_1obj(), epochs=10)
        s.fit(X, y)
        assert s.predict(test_x).std is None

    def test_1d_train_y_accepted(self, train_data_1obj, test_x) -> None:
        X, y = train_data_1obj
        assert y.ndim == 1
        s = TorchSurrogate(_make_model_1obj(), epochs=10)
        s.fit(X, y)
        assert s.predict(test_x).value.shape == (5, 1)

    def test_fit_resets_weights(self, train_data_1obj, test_x) -> None:
        X, y = train_data_1obj
        model = _make_model_1obj()
        s = TorchSurrogate(model, epochs=50)

        s.fit(X, y)
        pred_after_fit = s.predict(test_x).value.copy()

        # Second fit should start from the same initial weights
        s.fit(X, y)
        pred_after_refit = s.predict(test_x).value.copy()

        np.testing.assert_array_almost_equal(pred_after_fit, pred_after_refit)

    def test_initial_weights_not_modified(self, train_data_1obj) -> None:
        X, y = train_data_1obj
        model = _make_model_1obj()
        s = TorchSurrogate(model, epochs=50)
        initial_state = {k: v.clone() for k, v in s._initial_state.items()}

        s.fit(X, y)

        for key in initial_state:
            torch.testing.assert_close(s._initial_state[key], initial_state[key])

    def test_custom_optimizer(self, train_data_1obj, test_x) -> None:
        X, y = train_data_1obj
        s = TorchSurrogate(
            _make_model_1obj(),
            optimizer_cls=torch.optim.SGD,
            optimizer_kwargs={"lr": 1e-3},
            epochs=10,
        )
        s.fit(X, y)
        assert s.predict(test_x).value.shape == (5, 1)

    def test_custom_loss_fn(self, train_data_1obj, test_x) -> None:
        X, y = train_data_1obj
        s = TorchSurrogate(
            _make_model_1obj(),
            loss_fn=nn.L1Loss(),
            epochs=10,
        )
        s.fit(X, y)
        assert s.predict(test_x).value.shape == (5, 1)

    def test_missing_torch_raises(self, monkeypatch) -> None:
        monkeypatch.setitem(sys.modules, "torch", None)
        with pytest.raises(ImportError, match="PyTorch"):
            TorchSurrogate(_make_model_1obj())


# ===========================================================================
# Integration: GlobalSurrogateManager
# ===========================================================================
class TestTorchSurrogateIntegration:
    def test_global_manager_1obj(self, archive_1obj, candidates) -> None:
        s = TorchSurrogate(_make_model_1obj(), epochs=10)
        acquisition = MeanPrediction()
        manager = GlobalSurrogateManager(s)
        prediction = manager.predict(candidates, archive_1obj, ctx=None)
        scores = acquisition.evaluate(
            candidates, prediction, archive_1obj, ctx=None
        ).scores
        assert scores is not None
        assert scores.shape == (len(candidates),)
        assert prediction.value.shape == (len(candidates), 1)
