"""
Tests for the ConstraintHandler abstraction.

Tests cover:
- InequalityConstraint: rename, gradient() default, removed Constraint alias
- StaticToleranceHandler: sum-of-violations cv, feasibility_threshold, identity augment
- Problem: default handler reproduces previous behavior; custom handler is honored
- ConstraintHandler: compute_cv delegation and augment_objective in Problem.evaluate
"""

import numpy as np
import pytest

from saealib import (
    ConstraintHandler,
    InequalityConstraint,
    Problem,
    StaticToleranceHandler,
)


def _make_problem(constraints=None, handler=None):
    return Problem(
        func=lambda x: float(x[0]),
        dim=2,
        n_obj=1,
        direction=np.array([1.0]),
        lb=[0.0, 0.0],
        ub=[1.0, 1.0],
        constraints=constraints,
        handler=handler,
    )


# ---------------------------------------------------------------------------
# InequalityConstraint / removed alias
# ---------------------------------------------------------------------------


class TestInequalityConstraint:
    def test_gradient_default_none(self):
        c = InequalityConstraint(lambda x: float(x[0]))
        assert c.gradient(np.array([0.5, 0.5])) is None

    def test_gradient_overridable(self):
        class Linear(InequalityConstraint):
            def gradient(self, x):
                return np.array([1.0, 0.0])

        c = Linear(lambda x: float(x[0]))
        np.testing.assert_array_equal(c.gradient(np.zeros(2)), [1.0, 0.0])

    def test_constraint_alias_not_accessible_from_top_level(self):
        import saealib

        assert not hasattr(saealib, "Constraint")

    def test_constraint_alias_not_accessible_from_problem_package(self):
        import saealib.problem

        assert not hasattr(saealib.problem, "Constraint")


# ---------------------------------------------------------------------------
# StaticToleranceHandler
# ---------------------------------------------------------------------------


class TestStaticToleranceHandler:
    def test_is_constraint_handler(self):
        assert isinstance(StaticToleranceHandler(), ConstraintHandler)

    def test_compute_cv_sum_of_violations(self):
        constraints = [
            InequalityConstraint(lambda x: x[0] - 0.3),
            InequalityConstraint(lambda x: x[1] - 0.4),
        ]
        x = np.array([0.5, 0.5])
        g = np.array([0.2, 0.1])
        cv = StaticToleranceHandler().compute_cv(constraints, x, g)
        assert cv == pytest.approx(0.3)

    def test_compute_cv_respects_threshold(self):
        c = [InequalityConstraint(lambda x: float(x[0]), threshold=0.8)]
        cv = StaticToleranceHandler().compute_cv(c, np.array([0.0]), np.array([0.9]))
        assert cv == pytest.approx(0.1)

    def test_feasibility_threshold_default(self):
        assert StaticToleranceHandler().feasibility_threshold == pytest.approx(0.0)

    def test_feasibility_threshold_custom(self):
        assert StaticToleranceHandler(
            eps_cv=1e-3
        ).feasibility_threshold == pytest.approx(1e-3)

    def test_augment_objective_identity(self):
        f = np.array([1.0, 2.0])
        out = StaticToleranceHandler().augment_objective(
            f, [], np.zeros(2), np.empty(0)
        )
        np.testing.assert_array_equal(out, f)


# ---------------------------------------------------------------------------
# ConstraintHandler.repair default
# ---------------------------------------------------------------------------


class TestConstraintHandlerRepair:
    def test_default_repair_clips_to_bounds(self):
        h = StaticToleranceHandler()
        x = np.array([1.5, -0.5])
        lb = np.zeros(2)
        ub = np.ones(2)
        out = h.repair(x, [], lb, ub)
        np.testing.assert_array_equal(out, [1.0, 0.0])

    def test_default_repair_within_bounds_unchanged(self):
        h = StaticToleranceHandler()
        x = np.array([0.3, 0.7])
        lb, ub = np.zeros(2), np.ones(2)
        np.testing.assert_array_equal(h.repair(x, [], lb, ub), x)

    def test_default_repair_does_not_mutate_input(self):
        h = StaticToleranceHandler()
        x = np.array([2.0, -1.0])
        x_orig = x.copy()
        h.repair(x, [], np.zeros(2), np.ones(2))
        np.testing.assert_array_equal(x, x_orig)

    def test_custom_repair_overridable(self):
        class ReflectHandler(ConstraintHandler):
            def compute_cv(self, constraints, x, g):
                return 0.0

            def repair(self, x, constraints, lb, ub, **kwargs):
                return np.clip(2 * ub - x, lb, ub)

        h = ReflectHandler()
        x = np.array([1.2, 0.5])
        out = h.repair(x, [], np.zeros(2), np.ones(2))
        # 2*ub - x = [0.8, 1.5], clipped to [0,1] -> [0.8, 1.0]
        np.testing.assert_array_almost_equal(out, [0.8, 1.0])


# ---------------------------------------------------------------------------
# Problem integration
# ---------------------------------------------------------------------------


class TestProblemHandlerIntegration:
    def test_default_handler_is_static_tolerance(self):
        p = _make_problem()
        assert isinstance(p.handler, StaticToleranceHandler)

    def test_default_handler_eps_cv_matches(self):
        p = _make_problem()
        with pytest.warns(DeprecationWarning, match=r"Problem\(eps_cv=\.\.\.\)"):
            p2 = Problem(
                func=lambda x: float(x[0]),
                dim=1,
                n_obj=1,
                direction=np.array([1.0]),
                lb=[0.0],
                ub=[1.0],
                eps_cv=1e-3,
            )
        assert p.handler.feasibility_threshold == pytest.approx(0.0)
        assert p2.handler.feasibility_threshold == pytest.approx(1e-3)

    def test_evaluate_constraints_delegates_cv(self):
        c = [InequalityConstraint(lambda x: x[0] - 0.3)]
        p = _make_problem(constraints=c)
        g, cv = p.evaluate_constraints(np.array([0.5, 0.5]))
        assert g[0] == pytest.approx(0.2)
        assert cv == pytest.approx(0.2)

    def test_custom_handler_cv(self):
        class MaxHandler(ConstraintHandler):
            def compute_cv(self, constraints, x, g):
                return float(max((max(0.0, gi) for gi in g), default=0.0))

        constraints = [
            InequalityConstraint(lambda x: x[0] - 0.3),
            InequalityConstraint(lambda x: x[1] - 0.4),
        ]
        p = _make_problem(constraints=constraints, handler=MaxHandler())
        _g, cv = p.evaluate_constraints(np.array([0.5, 0.5]))
        # max(0.2, 0.1) instead of sum 0.3
        assert cv == pytest.approx(0.2)

    def test_evaluate_applies_augment_objective(self):
        class PenaltyHandler(ConstraintHandler):
            def compute_cv(self, constraints, x, g):
                return float(sum(max(0.0, gi) for gi in g))

            def augment_objective(self, f, constraints, x, g):
                cv = self.compute_cv(constraints, x, g)
                return f + 100.0 * cv

        c = [InequalityConstraint(lambda x: x[0] - 0.3)]
        p = _make_problem(constraints=c, handler=PenaltyHandler())
        # f = x[0] = 0.5, cv = 0.2 -> f' = 0.5 + 100 * 0.2 = 20.5
        f = p.evaluate(np.array([0.5, 0.5]))
        assert f[0] == pytest.approx(20.5)

    def test_evaluate_with_precomputed_g(self):
        class PenaltyHandler(ConstraintHandler):
            def compute_cv(self, constraints, x, g):
                return 0.0

            def augment_objective(self, f, constraints, x, g):
                return f + float(g[0])

        c = [InequalityConstraint(lambda x: x[0])]
        p = _make_problem(constraints=c, handler=PenaltyHandler())
        f = p.evaluate(np.array([0.5, 0.5]), g=np.array([7.0]))
        # augment uses the passed g (7.0), not a recomputed one (0.5)
        assert f[0] == pytest.approx(0.5 + 7.0)

    def test_default_evaluate_unchanged(self):
        c = [InequalityConstraint(lambda x: x[0] - 0.3)]
        p = _make_problem(constraints=c)
        f = p.evaluate(np.array([0.5, 0.5]))
        assert f[0] == pytest.approx(0.5)
