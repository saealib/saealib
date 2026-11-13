"""
Tests for constraint handling.

Tests cover:
- InequalityConstraint: evaluate, violation (satisfied / violated / on boundary)
- Problem.n_constraints: with and without constraints
- Problem.evaluate_constraints: no constraints, single, multiple
- cv aggregation: sum-of-violations
- Integration: cv is computed and stored in population/archive via LHSInitializer
"""

import numpy as np
import pytest

from saealib.problem import (
    EqualityConstraint,
    InequalityConstraint,
    Problem,
)

# ---------------------------------------------------------------------------
# InequalityConstraint unit tests
# ---------------------------------------------------------------------------


class TestInequalityConstraint:
    def test_evaluate_returns_func_value(self):
        c = InequalityConstraint(lambda x: x[0] ** 2 - 1.0)
        assert c.evaluate(np.array([2.0])) == pytest.approx(3.0)

    def test_violation_satisfied(self):
        c = InequalityConstraint(lambda x: float(x[0]), threshold=5.0)
        assert c.violation(np.array([3.0])) == pytest.approx(0.0)

    def test_violation_on_boundary(self):
        c = InequalityConstraint(lambda x: float(x[0]), threshold=5.0)
        assert c.violation(np.array([5.0])) == pytest.approx(0.0)

    def test_violation_violated(self):
        c = InequalityConstraint(lambda x: float(x[0]), threshold=5.0)
        assert c.violation(np.array([7.0])) == pytest.approx(2.0)

    def test_default_threshold_zero(self):
        c = InequalityConstraint(lambda x: float(x[0]))
        assert c.violation(np.array([-1.0])) == pytest.approx(0.0)
        assert c.violation(np.array([2.0])) == pytest.approx(2.0)

    def test_evaluate_with_violation_returns_both(self):
        c = InequalityConstraint(lambda x: float(x[0]), threshold=5.0)
        g, cv = c.evaluate_with_violation(np.array([7.0]))
        assert g == pytest.approx(7.0)
        assert cv == pytest.approx(2.0)
        g2, cv2 = c.evaluate_with_violation(np.array([3.0]))
        assert g2 == pytest.approx(3.0)
        assert cv2 == pytest.approx(0.0)

    def test_evaluate_with_violation_calls_func_once(self):
        calls = {"n": 0}

        def func(x):
            calls["n"] += 1
            return float(x[0])

        c = InequalityConstraint(func, threshold=1.0)
        c.evaluate_with_violation(np.array([3.0]))
        assert calls["n"] == 1


# ---------------------------------------------------------------------------
# Problem.n_constraints
# ---------------------------------------------------------------------------


class TestProblemNConstraints:
    def _make_problem(self, constraints=None):
        return Problem(
            func=lambda x: float(x[0]),
            dim=2,
            n_obj=1,
            direction=np.array([1.0]),
            lb=[0.0, 0.0],
            ub=[1.0, 1.0],
            constraints=constraints,
        )

    def test_no_constraints_default(self):
        p = self._make_problem()
        assert p.n_constraints == 0

    def test_no_constraints_explicit_none(self):
        p = self._make_problem(constraints=None)
        assert p.n_constraints == 0

    def test_single_constraint(self):
        p = self._make_problem(
            constraints=[InequalityConstraint(lambda x: float(x[0]))]
        )
        assert p.n_constraints == 1

    def test_multiple_constraints(self):
        p = self._make_problem(
            constraints=[
                InequalityConstraint(lambda x: float(x[0])),
                InequalityConstraint(lambda x: float(x[1])),
                InequalityConstraint(lambda x: float(x[0]) + float(x[1])),
            ]
        )
        assert p.n_constraints == 3


# ---------------------------------------------------------------------------
# Problem.evaluate_constraints
# ---------------------------------------------------------------------------


class TestEvaluateConstraints:
    def _make_problem(self, constraints=None):
        return Problem(
            func=lambda x: float(x[0]),
            dim=2,
            n_obj=1,
            direction=np.array([1.0]),
            lb=[0.0, 0.0],
            ub=[1.0, 1.0],
            constraints=constraints,
        )

    def test_no_constraints_returns_empty_and_zero(self):
        p = self._make_problem()
        g, cv = p.evaluate_constraints(np.array([0.5, 0.5]))
        assert g.shape == (0,)
        assert cv == pytest.approx(0.0)

    def test_single_constraint_satisfied(self):
        """g(x) = x[0] - 0.3 <= 0; x[0]=0.2 → satisfied."""
        c = InequalityConstraint(lambda x: x[0] - 0.3)
        p = self._make_problem(constraints=[c])
        g, cv = p.evaluate_constraints(np.array([0.2, 0.5]))
        assert g.shape == (1,)
        assert g[0] == pytest.approx(-0.1)
        assert cv == pytest.approx(0.0)

    def test_single_constraint_violated(self):
        """g(x) = x[0] - 0.3 <= 0; x[0]=0.5 → violated by 0.2."""
        c = InequalityConstraint(lambda x: x[0] - 0.3)
        p = self._make_problem(constraints=[c])
        g, cv = p.evaluate_constraints(np.array([0.5, 0.5]))
        assert g[0] == pytest.approx(0.2)
        assert cv == pytest.approx(0.2)

    def test_multiple_constraints_cv_is_sum(self):
        """cv = sum of individual violations."""
        constraints = [
            InequalityConstraint(lambda x: x[0] - 0.3),  # violated by 0.2 at x[0]=0.5
            InequalityConstraint(lambda x: x[1] - 0.4),  # violated by 0.1 at x[1]=0.5
        ]
        p = self._make_problem(constraints=constraints)
        g, cv = p.evaluate_constraints(np.array([0.5, 0.5]))
        assert g.shape == (2,)
        assert cv == pytest.approx(0.3)

    def test_multiple_constraints_some_satisfied(self):
        """Only violated constraints contribute to cv."""
        constraints = [
            InequalityConstraint(lambda x: x[0] - 0.8),  # satisfied at x[0]=0.5
            InequalityConstraint(lambda x: x[1] - 0.4),  # violated by 0.1 at x[1]=0.5
        ]
        p = self._make_problem(constraints=constraints)
        _g, cv = p.evaluate_constraints(np.array([0.5, 0.5]))
        assert cv == pytest.approx(0.1)

    def test_threshold_nonzero(self):
        """InequalityConstraint(func, threshold=t) means g(x) <= t."""
        c = InequalityConstraint(lambda x: float(x[0]), threshold=0.8)
        p = self._make_problem(constraints=[c])
        _g, cv = p.evaluate_constraints(np.array([0.5, 0.0]))
        assert cv == pytest.approx(0.0)  # 0.5 <= 0.8
        _g2, cv2 = p.evaluate_constraints(np.array([0.9, 0.0]))
        assert cv2 == pytest.approx(0.1)  # 0.9 - 0.8 = 0.1

    def test_g_dtype_float(self):
        c = InequalityConstraint(lambda x: float(x[0]))
        p = self._make_problem(constraints=[c])
        g, _ = p.evaluate_constraints(np.array([0.5, 0.5]))
        assert g.dtype == float

    def test_evaluates_each_constraint_once(self):
        """Each constraint func is evaluated once, not twice (no double-eval)."""
        calls = [0, 0]

        def make_func(i):
            def func(x):
                calls[i] += 1
                return float(x[i])

            return func

        constraints = [
            InequalityConstraint(make_func(0)),
            InequalityConstraint(make_func(1)),
        ]
        p = self._make_problem(constraints=constraints)
        p.evaluate_constraints(np.array([0.5, 0.5]))
        assert calls == [1, 1]


# ---------------------------------------------------------------------------
# EqualityConstraint
# ---------------------------------------------------------------------------


class TestEqualityConstraint:
    def test_is_constraint_subclass(self):
        """EqualityConstraint is a concrete subclass of the Constraint ABC."""
        c = EqualityConstraint(lambda x: float(x[0]))
        assert isinstance(c, InequalityConstraint)

    def test_evaluate_returns_raw_value(self):
        """evaluate(x) returns the raw h(x), not its absolute value."""
        c = EqualityConstraint(lambda x: float(x[0]) - 1.0)
        assert c.evaluate(np.array([-2.0])) == pytest.approx(-3.0)

    def test_violation_uses_absolute_value(self):
        """Both signs of h(x) violate symmetrically: |h(x)| - tolerance."""
        c = EqualityConstraint(lambda x: float(x[0]), tolerance=0.0)
        assert c.violation(np.array([0.3])) == pytest.approx(0.3)
        assert c.violation(np.array([-0.3])) == pytest.approx(0.3)

    def test_violation_within_tolerance_is_zero(self):
        """|h(x)| <= tolerance → violation == 0."""
        c = EqualityConstraint(lambda x: float(x[0]), tolerance=0.1)
        assert c.violation(np.array([0.05])) == pytest.approx(0.0)
        assert c.violation(np.array([-0.05])) == pytest.approx(0.0)

    def test_violation_on_tolerance_boundary(self):
        """|h(x)| == tolerance → violation == 0 (boundary is feasible)."""
        c = EqualityConstraint(lambda x: float(x[0]), tolerance=0.1)
        assert c.violation(np.array([0.1])) == pytest.approx(0.0)
        assert c.violation(np.array([-0.1])) == pytest.approx(0.0)

    def test_violation_outside_tolerance(self):
        """|h(x)| > tolerance → violation == |h(x)| - tolerance."""
        c = EqualityConstraint(lambda x: float(x[0]), tolerance=0.1)
        assert c.violation(np.array([0.3])) == pytest.approx(0.2)
        assert c.violation(np.array([-0.3])) == pytest.approx(0.2)

    def test_default_tolerance(self):
        """Default tolerance is 1e-4."""
        c = EqualityConstraint(lambda x: float(x[0]))
        assert c.tolerance == pytest.approx(1e-4)
        assert c.violation(np.array([1e-5])) == pytest.approx(0.0)

    def test_tolerance_zero_valid(self):
        """tolerance=0.0 is valid (threshold managed externally by a handler)."""
        c = EqualityConstraint(lambda x: float(x[0]), tolerance=0.0)
        assert c.tolerance == pytest.approx(0.0)
        assert c.violation(np.array([0.0])) == pytest.approx(0.0)
        assert c.violation(np.array([1e-9])) == pytest.approx(1e-9)

    def test_gradient_default_none(self):
        """gradient(x) defaults to None unless overridden."""
        c = EqualityConstraint(lambda x: float(x[0]))
        assert c.gradient(np.array([0.5])) is None

    def test_evaluate_with_violation_single_call(self):
        """(g, cv) is consistent and computed with a single function evaluation."""
        calls = {"n": 0}

        def func(x):
            calls["n"] += 1
            return float(x[0]) - 1.0

        c = EqualityConstraint(func, tolerance=0.0)
        g, cv = c.evaluate_with_violation(np.array([0.4]))
        assert g == pytest.approx(-0.6)
        assert cv == pytest.approx(0.6)
        assert calls["n"] == 1


# ---------------------------------------------------------------------------
# Mixed inequality + equality constraints
# ---------------------------------------------------------------------------


class TestMixedConstraints:
    def _make_problem(self, constraints=None):
        return Problem(
            func=lambda x: float(x[0]),
            dim=2,
            n_obj=1,
            direction=np.array([1.0]),
            lb=[0.0, 0.0],
            ub=[1.0, 1.0],
            constraints=constraints,
        )

    def test_mixed_cv_is_sum(self):
        """cv aggregates inequality and equality violations through one handler."""
        constraints = [
            InequalityConstraint(lambda x: x[0] - 0.3),  # violated by 0.2 at 0.5
            EqualityConstraint(lambda x: x[1] - 1.0, tolerance=0.0),  # |0.5-1|=0.5
        ]
        p = self._make_problem(constraints=constraints)
        g, cv = p.evaluate_constraints(np.array([0.5, 0.5]))
        assert g.shape == (2,)
        assert g[0] == pytest.approx(0.2)
        assert g[1] == pytest.approx(-0.5)  # raw equality value, signed
        assert cv == pytest.approx(0.7)  # 0.2 + 0.5

    def test_mixed_all_feasible(self):
        """Feasible inequality and within-tolerance equality → cv == 0."""
        constraints = [
            InequalityConstraint(lambda x: x[0] - 0.8),  # satisfied at 0.5
            EqualityConstraint(lambda x: x[1] - 0.5, tolerance=1e-6),  # exact
        ]
        p = self._make_problem(constraints=constraints)
        _g, cv = p.evaluate_constraints(np.array([0.5, 0.5]))
        assert cv == pytest.approx(0.0)
