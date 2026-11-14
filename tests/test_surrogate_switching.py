"""Tests for surrogate/switching.py."""

from unittest.mock import MagicMock

import pytest

from saealib.surrogate.accuracy import SurrogateAccuracy
from saealib.surrogate.switching import (
    AccuracyBasedSurrogateSwitcher,
    GenCtrlSwitcher,
    ManagerSwitcher,
    StrategySwitcher,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _acc(spearman: float, n: int = 10) -> SurrogateAccuracy:
    return SurrogateAccuracy(metrics={"spearman": spearman}, n_samples=n)


# ---------------------------------------------------------------------------
# ABC
# ---------------------------------------------------------------------------


def test_abc_is_abstract() -> None:
    with pytest.raises(TypeError):
        AccuracyBasedSurrogateSwitcher()  # type: ignore[abstract]  # intentional: testing abstract instantiation raises TypeError


# ---------------------------------------------------------------------------
# ManagerSwitcher
# ---------------------------------------------------------------------------


class TestManagerSwitcher:
    def _make(self, threshold=0.5):
        primary, fallback = MagicMock(name="primary"), MagicMock(name="fallback")
        sw = ManagerSwitcher(primary, fallback, threshold=threshold)
        return primary, fallback, sw

    def test_returns_primary_above_threshold(self) -> None:
        primary, _, sw = self._make()
        assert sw.switch(_acc(0.8)) is primary

    def test_returns_primary_at_threshold(self) -> None:
        primary, _, sw = self._make(threshold=0.5)
        assert sw.switch(_acc(0.5)) is primary

    def test_returns_fallback_below_threshold(self) -> None:
        _, fallback, sw = self._make()
        assert sw.switch(_acc(0.3)) is fallback

    def test_returns_fallback_when_accuracy_none(self) -> None:
        _, fallback, sw = self._make()
        assert sw.switch(None) is fallback

    def test_returns_fallback_when_metric_missing(self) -> None:
        _, fallback, sw = self._make()
        acc = SurrogateAccuracy(metrics={}, n_samples=5)
        assert sw.switch(acc) is fallback

    def test_custom_metric(self) -> None:
        primary, fallback = MagicMock(), MagicMock()
        sw = ManagerSwitcher(primary, fallback, metric="r2", threshold=0.7)
        high = SurrogateAccuracy(metrics={"r2": 0.9}, n_samples=5)
        low = SurrogateAccuracy(metrics={"r2": 0.5}, n_samples=5)
        assert sw.switch(high) is primary
        assert sw.switch(low) is fallback

    def test_defaults(self) -> None:
        sw = ManagerSwitcher(MagicMock(), MagicMock())
        assert sw.metric == "spearman"
        assert sw.threshold == 0.5

    def test_sequence(self) -> None:
        primary, fallback, sw = self._make(threshold=0.6)
        snapshots = [None, _acc(0.4), _acc(0.7), _acc(0.55), _acc(0.9)]
        expected = [fallback, fallback, primary, fallback, primary]
        for acc, exp in zip(snapshots, expected):
            assert sw.switch(acc) is exp


# ---------------------------------------------------------------------------
# StrategySwitcher
# ---------------------------------------------------------------------------


class TestStrategySwitcher:
    def _make(self, threshold=0.56):
        primary, fallback = MagicMock(name="ps"), MagicMock(name="ib")
        sw = StrategySwitcher(primary, fallback, threshold=threshold)
        return primary, fallback, sw

    def test_returns_primary_above_threshold(self) -> None:
        primary, _, sw = self._make()
        assert sw.switch(_acc(0.8)) is primary

    def test_returns_fallback_below_threshold(self) -> None:
        _, fallback, sw = self._make()
        assert sw.switch(_acc(0.3)) is fallback

    def test_returns_fallback_when_none(self) -> None:
        _, fallback, sw = self._make()
        assert sw.switch(None) is fallback

    def test_default_threshold_is_056(self) -> None:
        sw = StrategySwitcher(MagicMock(), MagicMock())
        assert sw.threshold == 0.56

    def test_default_metric_is_spearman(self) -> None:
        sw = StrategySwitcher(MagicMock(), MagicMock())
        assert sw.metric == "spearman"

    def test_default_mid_is_none(self) -> None:
        sw = StrategySwitcher(MagicMock(), MagicMock())
        assert sw.mid is None
        assert sw.mid_threshold is None

    def test_two_way_mode_unchanged_at_threshold(self) -> None:
        primary, _, sw = self._make()
        assert sw.switch(_acc(0.56)) is primary

    def test_two_way_mode_unchanged_below_threshold(self) -> None:
        _, fallback, sw = self._make()
        assert sw.switch(_acc(0.55)) is fallback

    def test_two_way_mode_unchanged_when_metric_non_finite(self) -> None:
        _, fallback, sw = self._make()
        acc = SurrogateAccuracy(metrics={"spearman": float("nan")}, n_samples=5)
        assert sw.switch(acc) is fallback

    def test_two_way_mode_sequence_pinned(self) -> None:
        # Pins the pre-existing 2-way switch() behavior unchanged by the
        # addition of the optional mid/mid_threshold parameters.
        primary, fallback, sw = self._make(threshold=0.56)
        snapshots = [None, _acc(0.3), _acc(0.56), _acc(0.99), _acc(1.0)]
        expected = [fallback, fallback, primary, primary, primary]
        for acc, exp in zip(snapshots, expected):
            assert sw.switch(acc) is exp


class TestStrategySwitcherThreeWay:
    def _make(self, threshold=0.57, mid_threshold=1.0):
        primary = MagicMock(name="ps")
        fallback = MagicMock(name="ib")
        mid = MagicMock(name="gb")
        sw = StrategySwitcher(
            primary,
            fallback,
            threshold=threshold,
            mid=mid,
            mid_threshold=mid_threshold,
        )
        return primary, fallback, mid, sw

    def test_mid_without_mid_threshold_raises(self) -> None:
        with pytest.raises(ValueError):
            StrategySwitcher(MagicMock(), MagicMock(), mid=MagicMock())

    def test_mid_threshold_without_mid_raises(self) -> None:
        with pytest.raises(ValueError):
            StrategySwitcher(MagicMock(), MagicMock(), mid_threshold=1.0)

    def test_routes_to_fallback_below_threshold(self) -> None:
        _, fallback, _, sw = self._make()
        assert sw.switch(_acc(0.5)) is fallback

    def test_routes_to_mid_at_threshold(self) -> None:
        _, _, mid, sw = self._make()
        assert sw.switch(_acc(0.57)) is mid

    def test_routes_to_mid_just_below_mid_threshold(self) -> None:
        _, _, mid, sw = self._make()
        assert sw.switch(_acc(0.99)) is mid

    def test_routes_to_primary_at_mid_threshold(self) -> None:
        primary, _, _, sw = self._make()
        assert sw.switch(_acc(1.0)) is primary

    def test_routes_to_fallback_when_none(self) -> None:
        _, fallback, _, sw = self._make()
        assert sw.switch(None) is fallback

    def test_routes_to_fallback_when_metric_non_finite(self) -> None:
        _, fallback, _, sw = self._make()
        acc = SurrogateAccuracy(metrics={"spearman": float("nan")}, n_samples=5)
        assert sw.switch(acc) is fallback

    def test_sequence(self) -> None:
        primary, fallback, mid, sw = self._make()
        snapshots = [None, _acc(0.3), _acc(0.57), _acc(0.8), _acc(1.0)]
        expected = [fallback, fallback, mid, mid, primary]
        for acc, exp in zip(snapshots, expected):
            assert sw.switch(acc) is exp


# ---------------------------------------------------------------------------
# GenCtrlSwitcher
# ---------------------------------------------------------------------------


class TestGenCtrlSwitcherInit:
    def test_invalid_update_rate_zero(self) -> None:
        with pytest.raises(ValueError):
            GenCtrlSwitcher(update_rate=0.0)

    def test_invalid_update_rate_above_one(self) -> None:
        with pytest.raises(ValueError):
            GenCtrlSwitcher(update_rate=1.1)

    def test_invalid_gm_min_negative(self) -> None:
        with pytest.raises(ValueError):
            GenCtrlSwitcher(gm_min=-1)

    def test_invalid_gm_max_less_than_min(self) -> None:
        with pytest.raises(ValueError):
            GenCtrlSwitcher(gm_max=2, gm_min=5)


class TestGenCtrlSwitcherSwitch:
    def test_perfect_accuracy_returns_gm_max(self) -> None:
        sw = GenCtrlSwitcher(gm_max=5, update_rate=1.0)
        assert sw.switch(_acc(1.0)) == 5

    def test_worst_accuracy_returns_gm_min(self) -> None:
        sw = GenCtrlSwitcher(gm_max=5, gm_min=0, update_rate=1.0)
        assert sw.switch(_acc(-1.0)) == 0

    def test_medium_accuracy_intermediate(self) -> None:
        # spearman=0.0 -> eps=0.5 -> quality=0.5 -> gm=5
        sw = GenCtrlSwitcher(gm_max=10, update_rate=1.0)
        assert sw.switch(_acc(0.0)) == 5

    def test_none_accuracy_does_not_change_state(self) -> None:
        sw = GenCtrlSwitcher(gm_max=10, update_rate=1.0, initial_error=0.5)
        before = sw.smoothed_error
        sw.switch(None)
        assert sw.smoothed_error == before

    def test_smoothing_converges_to_gm_max(self) -> None:
        sw = GenCtrlSwitcher(gm_max=10, update_rate=0.5)
        for _ in range(30):
            result = sw.switch(_acc(1.0))
        assert result == 10

    def test_gm_min_clamp(self) -> None:
        sw = GenCtrlSwitcher(gm_max=5, gm_min=2, update_rate=1.0)
        assert sw.switch(_acc(-1.0)) >= 2

    def test_gm_max_clamp(self) -> None:
        sw = GenCtrlSwitcher(gm_max=3, update_rate=1.0)
        assert sw.switch(_acc(1.0)) <= 3

    def test_smoothed_error_is_public(self) -> None:
        sw = GenCtrlSwitcher(initial_error=0.3)
        assert sw.smoothed_error == 0.3


# ---------------------------------------------------------------------------
# Deprecation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("constructor", "replacement"),
    [
        (lambda: ManagerSwitcher(MagicMock(), MagicMock()), "ThresholdController"),
        (
            lambda: StrategySwitcher(MagicMock(), MagicMock()),
            "ThresholdController",
        ),
        (lambda: GenCtrlSwitcher(), "SmoothedController"),
    ],
    ids=["manager", "strategy", "gen-ctrl"],
)
def test_the_concrete_switchers_name_their_controller_replacement(
    constructor, replacement: str
) -> None:
    with pytest.warns(DeprecationWarning, match=replacement):
        constructor()
