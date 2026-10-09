"""
Tests for Problem.evaluate_batch.

Tests cover:
- Default Problem.evaluate_batch returns None (batch evaluation unsupported)
- A subclass overriding evaluate_batch satisfies the (f_batch, g_batch) shape
  contract, both without and with constraints
"""

import numpy as np
import pytest

from saealib.exceptions import ValidationError
from saealib.problem import InequalityConstraint, Problem


def _sphere_problem(constraints=None):
    return Problem(
        func=lambda x: np.sum(x**2),
        dim=2,
        n_obj=1,
        direction=np.array([-1.0]),
        lb=[-5.0, -5.0],
        ub=[5.0, 5.0],
        constraints=constraints,
    )


def test_rejects_lower_bound_with_wrong_length():
    with pytest.raises(ValidationError, match=r"lb.*length=1.*dim=2"):
        Problem(
            func=lambda x: np.sum(x**2),
            dim=2,
            n_obj=1,
            direction=np.array([-1.0]),
            lb=[-5.0],
            ub=[5.0, 5.0],
        )


def test_rejects_upper_bound_with_wrong_length():
    with pytest.raises(ValidationError, match=r"ub.*length=3.*dim=2"):
        Problem(
            func=lambda x: np.sum(x**2),
            dim=2,
            n_obj=1,
            direction=np.array([-1.0]),
            lb=[-5.0, -5.0],
            ub=[5.0, 5.0, 5.0],
        )


def test_rejects_multidimensional_lower_bound():
    with pytest.raises(ValidationError, match=r"lb.*shape=\(1, 2\).*dim=2"):
        Problem(
            func=lambda x: np.sum(x**2),
            dim=2,
            n_obj=1,
            direction=np.array([-1.0]),
            lb=np.array([[-5.0, -5.0]]),
            ub=[5.0, 5.0],
        )


def test_broadcasts_scalar_bounds():
    problem = Problem(
        func=lambda x: np.sum(x**2),
        dim=2,
        n_obj=1,
        direction=np.array([-1.0]),
        lb=-5.0,
        ub=5.0,
    )

    np.testing.assert_array_equal(problem.lb, [-5.0, -5.0])
    np.testing.assert_array_equal(problem.ub, [5.0, 5.0])


def test_accepts_bounds_with_length_dim():
    problem = Problem(
        func=lambda x: np.sum(x**2),
        dim=2,
        n_obj=1,
        direction=np.array([-1.0]),
        lb=np.array([-5.0, -1.0]),
        ub=np.array([5.0, 1.0]),
    )

    np.testing.assert_array_equal(problem.lb, [-5.0, -1.0])
    np.testing.assert_array_equal(problem.ub, [5.0, 1.0])


class TestDefaultEvaluateBatch:
    def test_returns_none(self):
        p = _sphere_problem()
        x = np.array([[0.0, 0.0], [1.0, 2.0], [-3.0, 4.0]])
        assert p.evaluate_batch(x) is None

    def test_returns_none_with_constraints(self):
        p = _sphere_problem(constraints=[InequalityConstraint(lambda x: x[0] - 1.0)])
        x = np.array([[0.0, 0.0], [2.0, 0.0]])
        assert p.evaluate_batch(x) is None


class _BatchSphereProblem(Problem):
    """Toy subclass whose evaluate_batch scores all rows in one call."""

    def evaluate_batch(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
        x = np.asarray(x, dtype=float)
        f_batch = np.sum(x**2, axis=1, keepdims=True)
        g_batch = np.empty((x.shape[0], len(self.constraints)), dtype=float)
        for i, c in enumerate(self.constraints):
            g_batch[:, i] = [c.evaluate(row) for row in x]
        return f_batch, g_batch


class TestOverriddenEvaluateBatch:
    def test_shapes_without_constraints(self):
        p = _BatchSphereProblem(
            func=lambda x: np.sum(x**2),
            dim=2,
            n_obj=1,
            direction=np.array([-1.0]),
            lb=[-5.0, -5.0],
            ub=[5.0, 5.0],
        )
        n = 4
        x = np.array([[0.0, 0.0], [1.0, 2.0], [-3.0, 4.0], [5.0, -5.0]])
        result = p.evaluate_batch(x)
        assert result is not None
        f_batch, g_batch = result
        assert f_batch.shape == (n, p.n_obj)
        assert g_batch.shape == (n, p.n_constraints)
        assert g_batch.shape == (n, 0)

    def test_shapes_with_constraints(self):
        constraints = [
            InequalityConstraint(lambda x: x[0] - 1.0),
            InequalityConstraint(lambda x: x[1] + 1.0),
        ]
        p = _BatchSphereProblem(
            func=lambda x: np.sum(x**2),
            dim=2,
            n_obj=1,
            direction=np.array([-1.0]),
            lb=[-5.0, -5.0],
            ub=[5.0, 5.0],
            constraints=constraints,
        )
        n = 3
        x = np.array([[0.0, 0.0], [2.0, 0.0], [-3.0, 1.0]])
        result = p.evaluate_batch(x)
        assert result is not None
        f_batch, g_batch = result
        assert f_batch.shape == (n, p.n_obj)
        assert g_batch.shape == (n, p.n_constraints)
        for i, xi in enumerate(x):
            assert f_batch[i] == pytest.approx(np.sum(xi**2))
            for j, c in enumerate(constraints):
                assert g_batch[i, j] == pytest.approx(c.evaluate(xi))
