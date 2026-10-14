"""
Tests for the termination module.

Tests cover:
- Termination: init validation, is_terminated with single/multiple conditions
- max_fe: factory function for function evaluation termination
- max_gen: factory function for generation termination
- Custom callable conditions
"""

from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest

from saealib.context import OptimizationState
from saealib.exceptions import ValidationError
from saealib.registry import build, get, to_spec
from saealib.termination import (
    Termination,
    TerminationCondition,
    f_target,
    max_fe,
    max_gen,
    stalled,
)

_DUMMY_ARCHIVE = SimpleNamespace(get=lambda key: np.zeros((0, 1)))
_DUMMY_DIRECTION = np.array([-1.0], dtype=float)


def _make_ctx(fe: int = 0, gen: int = 0) -> OptimizationState:
    """Create a minimal context stub with fe and gen for termination tests."""
    return cast(
        OptimizationState,
        SimpleNamespace(
            fe=fe,
            gen=gen,
            archive=_DUMMY_ARCHIVE,
            direction=_DUMMY_DIRECTION,
        ),
    )


def _make_obj_ctx(
    f_values, weight: float = -1.0, gen: int = 0, fe: int = 0
) -> OptimizationState:
    """
    Create a context stub with an archive of objective values.

    ``f_values`` is a sequence of single-objective values (shape ``(n,)``);
    ``weight`` is the optimization direction (-1 minimize, +1 maximize).
    """
    f_arr = np.asarray(f_values, dtype=float).reshape(-1, 1)
    archive = SimpleNamespace(get=lambda key: f_arr if key == "f" else None)
    return cast(
        OptimizationState,
        SimpleNamespace(
            archive=archive,
            direction=np.array([weight], dtype=float),
            gen=gen,
            fe=fe,
        ),
    )


# ===========================================================================
# Termination.__init__ Tests
# ===========================================================================
class TestTerminationInit:
    """Tests for Termination initialization and validation."""

    def test_single_condition(self) -> None:
        t = Termination(max_fe(100))
        assert isinstance(t.condition, TerminationCondition)

    def test_multiple_conditions(self) -> None:
        t = Termination(max_fe(100), max_gen(50))
        assert isinstance(t.condition, TerminationCondition)

    def test_lambda_condition(self) -> None:
        t = Termination(lambda ctx: ctx.fe >= 10)
        assert isinstance(t.condition, TerminationCondition)

    def test_no_conditions_raises_validation_error(self) -> None:
        with pytest.raises(ValidationError, match="At least one"):
            Termination()

    def test_non_callable_raises_type_error(self) -> None:
        with pytest.raises(TypeError, match="must be callable"):
            Termination(42)  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]

    def test_mixed_callable_and_non_callable_raises(self) -> None:
        with pytest.raises(TypeError, match="must be callable"):
            Termination(max_fe(100), "not_callable")  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]

    def test_condition_is_read_only(self) -> None:
        t = Termination(max_fe(100))
        with pytest.raises(AttributeError):
            t.condition = max_gen(50)  # type: ignore  # assigning condition slot; declared type is wider than TerminationCondition


# ===========================================================================
# Termination.is_terminated Tests
# ===========================================================================
class TestIsTerminated:
    """Tests for Termination.is_terminated logic."""

    def test_not_terminated_below_threshold(self) -> None:
        t = Termination(max_fe(100))
        ctx = _make_ctx(fe=50)
        assert t.is_terminated(ctx) is False

    def test_terminated_at_threshold(self) -> None:
        t = Termination(max_fe(100))
        ctx = _make_ctx(fe=100)
        assert t.is_terminated(ctx) is True

    def test_terminated_above_threshold(self) -> None:
        t = Termination(max_fe(100))
        ctx = _make_ctx(fe=150)
        assert t.is_terminated(ctx) is True

    def test_multiple_conditions_or_logic_first_met(self) -> None:
        """Terminates when ANY condition is met (OR logic)."""
        t = Termination(max_fe(100), max_gen(50))
        ctx = _make_ctx(fe=100, gen=10)
        assert t.is_terminated(ctx) is True

    def test_multiple_conditions_or_logic_second_met(self) -> None:
        t = Termination(max_fe(100), max_gen(50))
        ctx = _make_ctx(fe=10, gen=50)
        assert t.is_terminated(ctx) is True

    def test_multiple_conditions_or_logic_both_met(self) -> None:
        t = Termination(max_fe(100), max_gen(50))
        ctx = _make_ctx(fe=100, gen=50)
        assert t.is_terminated(ctx) is True

    def test_multiple_conditions_none_met(self) -> None:
        t = Termination(max_fe(100), max_gen(50))
        ctx = _make_ctx(fe=10, gen=10)
        assert t.is_terminated(ctx) is False

    def test_custom_callable_condition(self) -> None:
        def custom(ctx):
            return ctx.fe > 0 and ctx.gen > 0

        t = Termination(custom)
        assert t.is_terminated(_make_ctx(fe=0, gen=0)) is False
        assert t.is_terminated(_make_ctx(fe=1, gen=1)) is True

    def test_lambda_condition(self) -> None:
        t = Termination(lambda ctx: ctx.fe >= 5)
        assert t.is_terminated(_make_ctx(fe=4)) is False
        assert t.is_terminated(_make_ctx(fe=5)) is True


# ===========================================================================
# max_fe Tests
# ===========================================================================
class TestMaxFe:
    """Tests for the max_fe factory function."""

    def test_returns_callable(self) -> None:
        cond = max_fe(100)
        assert callable(cond)

    def test_below_threshold(self) -> None:
        cond = max_fe(100)
        assert cond(_make_ctx(fe=99)) is False

    def test_at_threshold(self) -> None:
        cond = max_fe(100)
        assert cond(_make_ctx(fe=100)) is True

    def test_above_threshold(self) -> None:
        cond = max_fe(100)
        assert cond(_make_ctx(fe=200)) is True

    def test_zero_threshold(self) -> None:
        cond = max_fe(0)
        assert cond(_make_ctx(fe=0)) is True

    def test_qualname(self) -> None:
        cond = max_fe(500)
        assert cond.__qualname__ == "max_fe(500)"

    def test_doc(self) -> None:
        cond = max_fe(500)
        assert cond.__doc__ is not None and "500" in cond.__doc__


# ===========================================================================
# max_gen Tests
# ===========================================================================
class TestMaxGen:
    """Tests for the max_gen factory function."""

    def test_returns_callable(self) -> None:
        cond = max_gen(50)
        assert callable(cond)

    def test_below_threshold(self) -> None:
        cond = max_gen(50)
        assert cond(_make_ctx(gen=49)) is False

    def test_at_threshold(self) -> None:
        cond = max_gen(50)
        assert cond(_make_ctx(gen=50)) is True

    def test_above_threshold(self) -> None:
        cond = max_gen(50)
        assert cond(_make_ctx(gen=100)) is True

    def test_zero_threshold(self) -> None:
        cond = max_gen(0)
        assert cond(_make_ctx(gen=0)) is True

    def test_qualname(self) -> None:
        cond = max_gen(200)
        assert cond.__qualname__ == "max_gen(200)"

    def test_doc(self) -> None:
        cond = max_gen(200)
        assert cond.__doc__ is not None and "200" in cond.__doc__


# ===========================================================================
# Edge Case / Combination Tests
# ===========================================================================
class TestTerminationEdgeCases:
    """Tests for edge cases and advanced usage patterns."""

    def test_three_conditions(self) -> None:
        """Three conditions: terminates when any one is met."""
        t = Termination(
            max_fe(100),
            max_gen(50),
            lambda ctx: ctx.fe + ctx.gen > 80,
        )
        # Only third condition is met (fe=50 + gen=40 = 90 > 80)
        assert t.is_terminated(_make_ctx(fe=50, gen=40)) is True
        # No condition met
        assert t.is_terminated(_make_ctx(fe=10, gen=10)) is False

    def test_condition_with_class_method(self) -> None:
        """A class with __call__ can be used as a condition."""

        class StagnationCheck:
            def __call__(self, ctx):
                return ctx.gen >= 10

        t = Termination(StagnationCheck())
        assert t.is_terminated(_make_ctx(gen=9)) is False
        assert t.is_terminated(_make_ctx(gen=10)) is True

    def test_state_changes_over_iterations(self) -> None:
        """Simulate iteration: is_terminated returns False then True."""
        t = Termination(max_fe(5))
        ctx = _make_ctx(fe=0)
        for i in range(10):
            ctx.fe = i
            if i < 5:
                assert t.is_terminated(ctx) is False
            else:
                assert t.is_terminated(ctx) is True


# ===========================================================================
# TerminationCondition Operator Tests
# ===========================================================================
class TestTerminationConditionOperators:
    """Tests for the composable TerminationCondition class."""

    def test_factories_return_termination_condition(self) -> None:
        assert isinstance(max_fe(100), TerminationCondition)
        assert isinstance(max_gen(50), TerminationCondition)

    def test_call_returns_bool(self) -> None:
        cond = TerminationCondition(lambda ctx: ctx.fe)  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
        result = cond(_make_ctx(fe=5))
        assert result is True
        assert isinstance(result, bool)

    def test_or_operator(self) -> None:
        cond = max_fe(100) | max_gen(50)
        assert cond(_make_ctx(fe=10, gen=10)) is False
        assert cond(_make_ctx(fe=100, gen=10)) is True
        assert cond(_make_ctx(fe=10, gen=50)) is True

    def test_and_operator(self) -> None:
        cond = max_fe(100) & max_gen(50)
        assert cond(_make_ctx(fe=100, gen=10)) is False
        assert cond(_make_ctx(fe=10, gen=50)) is False
        assert cond(_make_ctx(fe=100, gen=50)) is True

    def test_invert_operator(self) -> None:
        cond = ~max_gen(50)
        assert cond(_make_ctx(gen=10)) is True
        assert cond(_make_ctx(gen=50)) is False

    def test_operators_accept_plain_callable(self) -> None:
        cond = max_fe(100) & (lambda ctx: ctx.gen >= 5)
        assert cond(_make_ctx(fe=100, gen=4)) is False
        assert cond(_make_ctx(fe=100, gen=5)) is True

    def test_reflected_operators_with_plain_callable_on_left(self) -> None:
        and_cond = (lambda ctx: ctx.gen >= 5) & max_fe(100)
        assert and_cond(_make_ctx(fe=100, gen=4)) is False
        assert and_cond(_make_ctx(fe=100, gen=5)) is True

        or_cond = (lambda ctx: ctx.gen >= 5) | max_fe(100)
        assert or_cond(_make_ctx(fe=10, gen=4)) is False
        assert or_cond(_make_ctx(fe=10, gen=5)) is True

    def test_non_callable_raises_type_error(self) -> None:
        with pytest.raises(TypeError, match="must be callable"):
            TerminationCondition(42)  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]

    def test_qualname_reflects_composition(self) -> None:
        assert "max_fe(100)" in (max_fe(100) | max_gen(50)).__qualname__
        assert "max_gen(50)" in (max_fe(100) | max_gen(50)).__qualname__

    def test_repr(self) -> None:
        assert repr(max_fe(100)) == "TerminationCondition(max_fe(100))"


# ===========================================================================
# Termination classmethod Tests (any_of / all_of / not_)
# ===========================================================================
class TestTerminationClassMethods:
    """Tests for Termination.any_of / all_of / not_."""

    def test_any_of_or_logic(self) -> None:
        t = Termination.any_of(max_fe(100), max_gen(50))
        assert t.is_terminated(_make_ctx(fe=10, gen=10)) is False
        assert t.is_terminated(_make_ctx(fe=100, gen=10)) is True

    def test_all_of_and_logic(self) -> None:
        t = Termination.all_of(max_fe(100), max_gen(50))
        assert t.is_terminated(_make_ctx(fe=100, gen=10)) is False
        assert t.is_terminated(_make_ctx(fe=100, gen=50)) is True

    def test_all_of_single_condition(self) -> None:
        t = Termination.all_of(max_fe(100))
        assert t.is_terminated(_make_ctx(fe=99)) is False
        assert t.is_terminated(_make_ctx(fe=100)) is True

    def test_all_of_no_conditions_raises_validation_error(self) -> None:
        with pytest.raises(ValidationError, match="At least one"):
            Termination.all_of()

    def test_not_negates(self) -> None:
        t = Termination.not_(max_gen(50))
        assert t.is_terminated(_make_ctx(gen=10)) is True
        assert t.is_terminated(_make_ctx(gen=50)) is False

    def test_not_accepts_plain_callable(self) -> None:
        t = Termination.not_(lambda ctx: ctx.fe >= 100)
        assert t.is_terminated(_make_ctx(fe=10)) is True
        assert t.is_terminated(_make_ctx(fe=100)) is False

    def test_returns_termination_instance(self) -> None:
        assert isinstance(Termination.any_of(max_fe(100)), Termination)
        assert isinstance(Termination.all_of(max_fe(100)), Termination)
        assert isinstance(Termination.not_(max_fe(100)), Termination)


# ===========================================================================
# f_target Tests
# ===========================================================================
class TestFTarget:
    """Tests for the f_target factory function."""

    def test_returns_termination_condition(self) -> None:
        assert isinstance(f_target(1e-6), TerminationCondition)

    def test_minimize_not_reached(self) -> None:
        cond = f_target(1e-6)
        assert cond(_make_obj_ctx([0.5, 0.1], weight=-1.0)) is False

    def test_minimize_reached(self) -> None:
        cond = f_target(1e-6)
        assert cond(_make_obj_ctx([0.5, 1e-9], weight=-1.0)) is True

    def test_maximize_not_reached(self) -> None:
        cond = f_target(100.0)
        assert cond(_make_obj_ctx([10.0, 50.0], weight=1.0)) is False

    def test_maximize_reached(self) -> None:
        cond = f_target(100.0)
        assert cond(_make_obj_ctx([10.0, 150.0], weight=1.0)) is True

    def test_empty_archive_returns_false(self) -> None:
        cond = f_target(1e-6)
        assert cond(_make_obj_ctx([], weight=-1.0)) is False


# ===========================================================================
# stalled Tests
# ===========================================================================
class TestStalled:
    """Tests for the stalled factory function."""

    def test_returns_termination_condition(self) -> None:
        assert isinstance(stalled(5), TerminationCondition)

    def test_terminates_after_window(self) -> None:
        cond = stalled(3)
        # First call sets the baseline best; no improvement afterwards.
        assert cond(_make_obj_ctx([1.0], weight=-1.0, gen=0)) is False
        assert cond(_make_obj_ctx([1.0], weight=-1.0, gen=1)) is False
        assert cond(_make_obj_ctx([1.0], weight=-1.0, gen=2)) is False
        assert cond(_make_obj_ctx([1.0], weight=-1.0, gen=3)) is True

    def test_improvement_resets_counter(self) -> None:
        cond = stalled(2)
        assert cond(_make_obj_ctx([1.0], weight=-1.0, gen=0)) is False
        assert cond(_make_obj_ctx([1.0], weight=-1.0, gen=1)) is False
        # Improvement (lower is better when minimizing) resets the stall gen.
        assert cond(_make_obj_ctx([0.5], weight=-1.0, gen=2)) is False
        assert cond(_make_obj_ctx([0.5], weight=-1.0, gen=4)) is True

    def test_empty_archive_returns_false(self) -> None:
        cond = stalled(3)
        assert cond(_make_obj_ctx([], weight=-1.0, gen=10)) is False


# ===========================================================================
# Registry integration: to_spec()/build() round-tripping
# ===========================================================================
class TestRegistryIntegration:
    """Factories record a _registry_spec; Termination keeps the original
    conditions tuple, so both are serializable via Registry.to_spec()/build()."""

    @pytest.mark.parametrize("name", ["max_fe", "max_gen", "f_target", "stalled"])
    def test_factories_are_registered(self, name) -> None:
        assert get(name).__name__ == name

    def test_termination_is_registered(self) -> None:
        assert get("Termination") is Termination

    @pytest.mark.parametrize(
        ("factory", "kwargs"),
        [
            (max_fe, {"value": 100}),
            (max_gen, {"value": 50}),
            (f_target, {"value": 1e-6}),
            (stalled, {"window": 5, "tol": 1e-8}),
        ],
    )
    def test_factory_condition_spec_round_trips(self, factory, kwargs) -> None:
        cond = factory(**kwargs)
        rebuilt = build(to_spec(cond))
        assert isinstance(rebuilt, TerminationCondition)
        assert rebuilt.__qualname__ == cond.__qualname__

    def test_composed_condition_has_no_spec(self) -> None:
        composed = max_fe(100) | max_gen(50)
        with pytest.raises(ValidationError):
            to_spec(composed)

    def test_raw_lambda_condition_has_no_spec(self) -> None:
        cond = TerminationCondition(lambda ctx: ctx.fe >= 10)
        with pytest.raises(ValidationError):
            to_spec(cond)

    def test_termination_single_condition_round_trips(self) -> None:
        t = Termination(max_fe(200))
        spec = to_spec(t)
        assert spec == {
            "type": "Termination",
            "params": [{"type": "max_fe", "params": {"value": 200}}],
        }
        rebuilt = build(spec)
        assert isinstance(rebuilt, Termination)
        assert rebuilt.is_terminated(_make_ctx(fe=200)) is True
        assert rebuilt.is_terminated(_make_ctx(fe=199)) is False

    def test_termination_multiple_conditions_round_trip(self) -> None:
        t = Termination(max_fe(100), max_gen(50))
        rebuilt = build(to_spec(t))
        assert isinstance(rebuilt, Termination)
        assert len(rebuilt.conditions) == 2
        assert rebuilt.is_terminated(_make_ctx(fe=10, gen=50)) is True
        assert rebuilt.is_terminated(_make_ctx(fe=10, gen=10)) is False

    def test_termination_with_unspecced_condition_raises(self) -> None:
        t = Termination(max_fe(100), lambda ctx: ctx.fe >= 10)
        with pytest.raises(ValidationError):
            to_spec(t)

    def test_all_of_composition_is_not_serializable(self) -> None:
        """all_of() collapses conditions into one AND-composed condition
        before storing it — that composed condition carries no spec."""
        t = Termination.all_of(max_fe(100), max_gen(50))
        with pytest.raises(ValidationError):
            to_spec(t)
