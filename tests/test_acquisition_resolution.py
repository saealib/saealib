"""Acquisition selection paired with the selected surrogate manager."""

from __future__ import annotations

import pickle

import numpy as np

import saealib
from saealib.acquisition.mean import MeanPrediction
from saealib.acquisition.winrate import WinRateAcquisition
from saealib.optimizer import Optimizer
from saealib.problem import Problem
from saealib.surrogate.base import ComparisonSurrogate
from saealib.surrogate.manager import (
    GlobalSurrogateManager,
    LocalSurrogateManager,
    PairwiseSurrogateManager,
)
from saealib.surrogate.prediction import PredictionChannel, SurrogatePrediction
from saealib.surrogate.rbf import RBFSurrogate
from saealib.surrogate.rbf_kernels import GaussianKernel
from saealib.surrogate.training_set import ArchiveObjectiveSet
from saealib.termination import Termination, max_fe

DIM = 2


class QuietComparisonSurrogate(ComparisonSurrogate):
    """Deterministic comparison surrogate; keeps pairwise prediction working."""

    def fit(self, train_x, train_y) -> None:
        self.n_train = len(train_x)

    def predict_proba(self, test_x) -> SurrogatePrediction:
        n = len(test_x)
        return SurrogatePrediction(
            channels={
                "objective": PredictionChannel(
                    value=np.linspace(0.2, 0.8, n).reshape(n, 1)
                )
            }
        )


class UnregisteredPairwiseManager(PairwiseSurrogateManager):
    """Concrete pairwise manager that is deliberately absent from the registry."""


def _sphere(x) -> np.ndarray:
    return np.array([float(np.sum(np.asarray(x) ** 2))])


def _problem() -> Problem:
    return Problem(
        func=_sphere,
        dim=DIM,
        n_obj=1,
        direction=np.array([-1.0]),
        lb=[-1.0] * DIM,
        ub=[1.0] * DIM,
    )


def _pairwise(
    manager_type: type[PairwiseSurrogateManager] = PairwiseSurrogateManager,
) -> PairwiseSurrogateManager:
    return manager_type(surrogate=QuietComparisonSurrogate(), n_ref=3)


def _local_preset(acquisition: dict | None = None) -> dict:
    params: dict = {
        "training_set": {"type": "KNNObjectiveSet", "params": {"n_neighbors": 2}}
    }
    if acquisition is not None:
        params["acquisition"] = acquisition
    return {"surrogate_manager": {"type": "LocalSurrogateManager", "params": params}}


def _origin(optimizer: Optimizer, slot: str):
    resolution = optimizer._component_resolution
    assert resolution is not None
    return resolution.origins[slot]


def _resolved(kind: str, preset: dict | None, manager=None) -> Optimizer:
    optimizer = Optimizer(_problem(), seed=1)
    if preset is not None:
        optimizer.set_preset(preset)
    if manager is not None:
        if kind == "setter":
            optimizer.set_surrogate_manager(manager)
        else:
            optimizer.surrogate_manager = manager
    return optimizer


def test_explicit_setter_pairwise_manager_derives_win_rate_across_resolution() -> None:
    optimizer = _resolved("setter", None, _pairwise())

    optimizer.resolve_defaults()
    acquisition = optimizer.acquisition
    origin = _origin(optimizer, "acquisition")
    assert isinstance(acquisition, WinRateAcquisition)

    optimizer.resolve_defaults()

    assert optimizer.acquisition is acquisition
    assert isinstance(optimizer.acquisition, WinRateAcquisition)
    assert _origin(optimizer, "acquisition") == origin


def test_directly_assigned_pairwise_manager_derives_win_rate() -> None:
    optimizer = _resolved("assignment", None, _pairwise())

    optimizer.resolve_defaults()

    assert isinstance(optimizer.acquisition, WinRateAcquisition)
    optimizer.resolve_defaults()
    assert isinstance(optimizer.acquisition, WinRateAcquisition)


def test_unregistered_pairwise_subclass_derives_win_rate_acquisition() -> None:
    optimizer = _resolved("assignment", None, _pairwise(UnregisteredPairwiseManager))

    optimizer.resolve_defaults()

    assert isinstance(optimizer.acquisition, WinRateAcquisition)
    optimizer.resolve_defaults()
    assert isinstance(optimizer.acquisition, WinRateAcquisition)


def test_displaced_preset_manager_does_not_supply_its_paired_acquisition() -> None:
    optimizer = _resolved("setter", _local_preset(), _pairwise())

    optimizer.resolve_defaults()

    assert isinstance(optimizer.surrogate_manager, PairwiseSurrogateManager)
    assert isinstance(optimizer.acquisition, WinRateAcquisition)
    assert _origin(optimizer, "acquisition").source != "user preset"


def test_assigned_manager_displaced_preset_manager_does_not_supply_pair() -> None:
    optimizer = _resolved("assignment", _local_preset(), _pairwise())

    optimizer.resolve_defaults()

    assert isinstance(optimizer.acquisition, WinRateAcquisition)
    assert _origin(optimizer, "acquisition").source != "user preset"


def test_selected_preset_manager_supplies_its_paired_acquisition() -> None:
    optimizer = _resolved("setter", _local_preset())

    optimizer.resolve_defaults()
    acquisition = optimizer.acquisition

    assert isinstance(optimizer.surrogate_manager, LocalSurrogateManager)
    assert isinstance(acquisition, MeanPrediction)
    assert _origin(optimizer, "acquisition").source == "user preset"

    optimizer.resolve_defaults()

    assert optimizer.acquisition is acquisition
    assert _origin(optimizer, "acquisition").source == "user preset"


def test_displaced_preset_manager_keeps_its_declared_acquisition() -> None:
    optimizer = _resolved(
        "setter",
        _local_preset({"type": "MeanPrediction", "params": {"weights": [2.0]}}),
        GlobalSurrogateManager(
            surrogate=RBFSurrogate(kernel=GaussianKernel()),
            training_set=ArchiveObjectiveSet(),
        ),
    )

    optimizer.resolve_defaults()

    assert isinstance(optimizer.surrogate_manager, GlobalSurrogateManager)
    assert isinstance(optimizer.acquisition, MeanPrediction)
    assert _origin(optimizer, "acquisition").source == "user preset"
    np.testing.assert_array_equal(optimizer.acquisition.weights, [2.0])


def test_explicit_acquisition_precedes_preset_pair() -> None:
    selected = WinRateAcquisition()
    optimizer = (
        Optimizer(_problem(), seed=1)
        .set_preset(_local_preset())
        .set_surrogate_manager(_pairwise())
        .set_acquisition(selected)
    )

    optimizer.resolve_defaults()

    assert optimizer.acquisition is selected
    assert _origin(optimizer, "acquisition").source == "setter"


def test_pairwise_manager_run_uses_win_rate_acquisition() -> None:
    optimizer = (
        Optimizer(_problem(), seed=1)
        .set_surrogate_manager(_pairwise())
        .set_termination(Termination(max_fe(60)))
    )

    optimizer.run()

    assert isinstance(optimizer.acquisition, WinRateAcquisition)


def test_directly_assigned_pairwise_manager_run_uses_win_rate_acquisition() -> None:
    optimizer = Optimizer(_problem(), seed=1)
    optimizer.surrogate_manager = _pairwise()
    optimizer.set_termination(Termination(max_fe(60)))

    optimizer.run()

    assert isinstance(optimizer.acquisition, WinRateAcquisition)


def test_pickle_round_trip_keeps_win_rate_acquisition() -> None:
    optimizer = _resolved("setter", None, _pairwise())
    optimizer.resolve_defaults()

    restored = pickle.loads(pickle.dumps(optimizer))
    restored.resolve_defaults()

    assert isinstance(restored.acquisition, WinRateAcquisition)


def _restored_without_binding_metadata(
    optimizer: Optimizer, *, clear_acquisition: bool = False
) -> Optimizer:
    state = optimizer.__getstate__()
    for name in (
        "_component_origins",
        "_component_resolution",
        "_materialized_components",
        "_configuration_version",
        "_track_component_assignments",
    ):
        state.pop(name, None)
    if clear_acquisition:
        state["acquisition"] = None
    restored = Optimizer.__new__(Optimizer)
    restored.__setstate__(state)
    return restored


def test_persisted_pairwise_manager_derives_win_rate_acquisition() -> None:
    optimizer = _resolved("setter", None, _pairwise())
    optimizer.resolve_defaults()

    restored = _restored_without_binding_metadata(optimizer, clear_acquisition=True)
    restored.resolve_defaults()

    assert isinstance(restored.acquisition, WinRateAcquisition)
    assert _origin(restored, "acquisition").source != "bundled preset"


def test_persisted_manager_displaced_preset_manager_does_not_supply_pair() -> None:
    optimizer = (
        Optimizer(_problem(), seed=1)
        .set_preset(_local_preset())
        .set_surrogate_manager(_pairwise())
    )
    optimizer.resolve_defaults()

    restored = _restored_without_binding_metadata(optimizer, clear_acquisition=True)
    restored.resolve_defaults()

    assert isinstance(restored.surrogate_manager, PairwiseSurrogateManager)
    assert isinstance(restored.acquisition, WinRateAcquisition)
    assert _origin(restored, "acquisition").source != "user preset"


def test_persisted_manager_keeps_declared_preset_acquisition() -> None:
    optimizer = (
        Optimizer(_problem(), seed=1)
        .set_preset(
            _local_preset({"type": "MeanPrediction", "params": {"weights": [2.0]}})
        )
        .set_surrogate_manager(_pairwise())
    )
    optimizer.resolve_defaults()

    restored = _restored_without_binding_metadata(optimizer, clear_acquisition=True)
    restored.resolve_defaults()

    assert isinstance(restored.surrogate_manager, PairwiseSurrogateManager)
    assert isinstance(restored.acquisition, MeanPrediction)
    assert _origin(restored, "acquisition").source == "user preset"
    np.testing.assert_array_equal(restored.acquisition.weights, [2.0])


def test_minimize_with_pairwise_manager_uses_win_rate_acquisition() -> None:
    result = saealib.minimize(
        _sphere,
        surrogate=_pairwise(),
        dim=DIM,
        lb=[-1.0] * DIM,
        ub=[1.0] * DIM,
        max_fe=60,
        seed=1,
        verbose=False,
    )

    assert np.isfinite(np.asarray(result.f)).all()
