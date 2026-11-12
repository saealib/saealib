"""Acquisition and archive boundary contracts."""

from typing import Any, cast

import numpy as np
import pytest

from saealib.acquisition import MeanPrediction
from saealib.acquisition.base import (
    AcquisitionFunction,
    AcquisitionResult,
    CompositeAcquisition,
    PointwiseAcquisition,
)
from saealib.acquisition.batch import BatchExpectedImprovement
from saealib.exceptions import ValidationError
from saealib.population import Archive, PopulationAttribute
from saealib.surrogate.manager import GlobalSurrogateManager
from saealib.surrogate.prediction import PredictionChannel, SurrogatePrediction


def _archive() -> Archive:
    attrs = [
        PopulationAttribute("x", np.float64, (1,)),
        PopulationAttribute("f", np.float64, (1,)),
        PopulationAttribute("id", np.int64, (), -1),
    ]
    return Archive(attrs, duplicate_policy="keep_first")


def test_prediction_owns_arrays_and_normalizes_dtype() -> None:
    value = np.array([[1]], dtype=np.int32)
    pred = SurrogatePrediction(channels={"objective": PredictionChannel(value)})
    value[0, 0] = 9
    assert pred.value.dtype == np.float64
    assert pred.value.flags.owndata
    assert pred.value[0, 0] == 1


def test_prediction_rejects_channel_row_mismatch() -> None:
    with pytest.raises(ValidationError):
        SurrogatePrediction(
            channels={
                "a": PredictionChannel(np.zeros((2, 1))),
                "b": PredictionChannel(np.zeros((3, 1))),
            }
        )


def test_manager_does_not_own_acquisition_and_provider_can_exchange_it() -> None:
    manager = GlobalSurrogateManager(cast(Any, object()))
    assert not hasattr(manager, "acquisition")
    assert not hasattr(manager, "iter_acquisitions")
    first = MeanPrediction()
    second = MeanPrediction()
    assert first is not second


def test_pointwise_empty_batch_is_owned_float64_and_does_not_prepare() -> None:
    class Counting(MeanPrediction):
        def __init__(self) -> None:
            super().__init__()
            self.prepares = 0

        def prepare(self, archive, *, ctx):
            self.prepares += 1
            return super().prepare(archive, ctx=ctx)

    acq = Counting()
    result = acq.evaluate(
        np.empty((0, 1)),
        SurrogatePrediction.objective(np.empty((0, 1))),
        _archive(),
        ctx=None,
    )
    assert result.scores is not None
    assert result.scores.shape == (0,)
    assert result.scores.dtype == np.float64
    assert acq.prepares == 0


def test_composite_rejects_missing_channel_before_combining() -> None:
    acq = CompositeAcquisition(
        {"constraint": MeanPrediction()}, lambda scores: scores[0]
    )
    pred = SurrogatePrediction.objective(np.zeros((2, 1)))
    with pytest.raises(ValidationError, match="missing configured channel"):
        acq.evaluate(np.zeros((2, 1)), pred, _archive(), ctx=None)


def test_pointwise_and_composite_validate_prediction_and_score_shapes() -> None:
    class BadPointwise(PointwiseAcquisition):
        def compute_reference(self, archive: Any, rng: Any = None) -> Any:
            return "reference"

        def score(self, prediction: Any, reference: Any, rng: Any = None) -> np.ndarray:
            return np.zeros(1, dtype=np.float64)

    pointwise = BadPointwise()
    candidates = np.zeros((2, 1), dtype=np.float64)
    archive = _archive()
    with pytest.raises(TypeError, match="requires a prediction"):
        pointwise.evaluate(candidates, None, archive, ctx=None)
    with pytest.raises(ValidationError, match="returned shape"):
        pointwise.evaluate(
            candidates,
            SurrogatePrediction.objective(np.zeros((2, 1))),
            archive,
            ctx=None,
        )

    class NoScores(AcquisitionFunction):
        def evaluate(
            self,
            candidates_x: np.ndarray,
            prediction: Any,
            archive: Any,
            *,
            prepared: Any = None,
            ctx: Any,
        ) -> AcquisitionResult:
            return AcquisitionResult(scores=None)

    class BadCombiner(AcquisitionFunction):
        def evaluate(
            self,
            candidates_x: np.ndarray,
            prediction: Any,
            archive: Any,
            *,
            prepared: Any = None,
            ctx: Any,
        ) -> AcquisitionResult:
            return AcquisitionResult(scores=np.zeros(len(candidates_x) - 1))

    prediction = SurrogatePrediction.objective(np.zeros((2, 1)))
    with pytest.raises(ValidationError, match="returned no scores"):
        CompositeAcquisition(
            {"objective": NoScores()}, lambda values: values[0]
        ).evaluate(candidates, prediction, archive, ctx=None)
    with pytest.raises(ValidationError, match="returned scores of shape"):
        CompositeAcquisition(
            {"objective": BadCombiner()}, lambda values: values[0]
        ).evaluate(candidates, prediction, archive, ctx=None)
    with pytest.raises(TypeError, match="requires a prediction"):
        CompositeAcquisition(
            {"objective": MeanPrediction()}, lambda values: values[0]
        ).evaluate(candidates, None, archive, ctx=None)


def test_composite_rejects_empty_and_invalid_combined_scores() -> None:
    with pytest.raises(ValueError, match="at least one"):
        CompositeAcquisition({}, lambda values: np.empty(0))

    candidates = np.zeros((2, 1), dtype=np.float64)
    prediction = SurrogatePrediction.objective(np.zeros((2, 1)))
    acq = CompositeAcquisition(
        {"objective": MeanPrediction()}, lambda values: np.zeros(1)
    )
    with pytest.raises(ValidationError, match="combine_fn returned shape"):
        acq.evaluate(candidates, prediction, _archive(), ctx=None)


def test_batch_expected_improvement_validates_joint_prediction_contract() -> None:
    candidates = np.zeros((2, 1), dtype=np.float64)
    archive = _archive()
    with pytest.raises(ValidationError, match="n_draws"):
        BatchExpectedImprovement(n_draws=0)
    with pytest.raises(ValidationError, match="one objective"):
        BatchExpectedImprovement(n_draws=2).evaluate(
            candidates,
            SurrogatePrediction.objective(np.zeros((2, 2)), np.ones((2, 2))),
            archive,
            ctx=None,
        )
    with pytest.raises(ValidationError, match="requires uncertainty"):
        BatchExpectedImprovement(n_draws=2).evaluate(
            candidates,
            SurrogatePrediction.objective(np.zeros((2, 1))),
            archive,
            ctx=None,
        )
    with pytest.raises(ValidationError, match="covariance"):
        BatchExpectedImprovement(n_draws=2).evaluate(
            candidates,
            SurrogatePrediction(
                {
                    "objective": PredictionChannel(
                        np.zeros((2, 1)), np.ones((2, 1)), covariance=np.eye(1)
                    )
                }
            ),
            archive,
            ctx=None,
        )


@pytest.mark.parametrize(
    "factory",
    [
        lambda: PredictionChannel(np.zeros((2, 1)), covariance=np.zeros(1)),
        lambda: PredictionChannel(np.zeros((2, 1)), samples=np.zeros(1)),
    ],
)
def test_prediction_channel_auxiliary_arrays_align_with_values(factory: Any) -> None:
    with pytest.raises(ValidationError, match="same leading dimension"):
        factory()


def test_prediction_validates_names_optional_arrays_and_channel_projection() -> None:
    with pytest.raises(ValidationError, match="non-empty"):
        SurrogatePrediction({"": PredictionChannel(np.zeros((1, 1)))})
    with pytest.raises(ValidationError, match="not a PredictionChannel"):
        SurrogatePrediction(cast(Any, {"objective": np.zeros((1, 1))}))
    with pytest.raises(ValidationError, match="prediction x"):
        SurrogatePrediction(
            {"objective": PredictionChannel(np.zeros((2, 1)))},
            x=np.zeros((1, 1)),
        )
    with pytest.raises(ValidationError, match="prediction label"):
        SurrogatePrediction(
            {"objective": PredictionChannel(np.zeros((2, 1)))},
            label=np.zeros(1),
        )

    prediction = SurrogatePrediction(
        {
            "objective": PredictionChannel(np.zeros((2, 1))),
            "win_rate": PredictionChannel(np.ones((2, 1))),
        },
        label=np.array([0, 1]),
    )
    assert not prediction.has_uncertainty
    assert prediction.has_label
    projected = prediction.select_channel("win_rate", as_objective=False)
    assert set(projected.channels) == {"win_rate"}
    with pytest.raises(KeyError):
        prediction.select_channel("missing")


@pytest.mark.parametrize(
    "policy, expected", [("keep_first", 1), ("replace", 1), ("append", 2)]
)
def test_archive_duplicate_policy(policy: str, expected: int) -> None:
    archive = Archive(
        [
            PopulationAttribute("x", np.float64, (1,)),
            PopulationAttribute("f", np.float64, (1,)),
            PopulationAttribute("id", np.int64, (), -1),
            PopulationAttribute("request_id", np.int64, (), -1),
        ],
        duplicate_policy=policy,
    )
    archive.add(x=[0.0], f=[1.0], id=10, request_id=20)
    archive.add(x=[0.0], f=[2.0], id=11, request_id=21)
    assert len(archive) == expected
    if policy == "replace":
        assert archive.id[0] == 11
        assert archive.f[0, 0] == 2.0


def test_duplicate_history_api_is_removed() -> None:
    archive = _archive()
    archive.add(x=[0.0], f=[1.0], id=1)
    archive.add(x=[0.0], f=[2.0], id=2)
    assert not hasattr(archive, "get_duplicated_population")
