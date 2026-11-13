"""Regression tests for saealib.exceptions hierarchy and public boundary raises."""

import pytest

import saealib
from saealib.exceptions import ConfigurationError, SaealibError, ValidationError


class TestHierarchy:
    def test_saealib_error_is_exception(self):
        assert issubclass(SaealibError, Exception)

    def test_validation_error_bases(self):
        assert issubclass(ValidationError, SaealibError)
        assert issubclass(ValidationError, ValueError)

    def test_configuration_error_bases(self):
        assert issubclass(ConfigurationError, SaealibError)
        assert issubclass(ConfigurationError, ValueError)

    def test_validation_error_catchable_as_value_error(self):
        with pytest.raises(ValueError):
            raise ValidationError("bad input")

    def test_configuration_error_catchable_as_value_error(self):
        with pytest.raises(ValueError):
            raise ConfigurationError("misconfigured")

    def test_both_catchable_as_saealib_error(self):
        with pytest.raises(SaealibError):
            raise ValidationError("x")
        with pytest.raises(SaealibError):
            raise ConfigurationError("x")


class TestTopLevelExport:
    def test_exported_in_all(self):
        assert "SaealibError" in saealib.__all__
        assert "ValidationError" in saealib.__all__
        assert "ConfigurationError" in saealib.__all__

    def test_importable_from_top_level(self):
        from saealib import ConfigurationError as CfgErr
        from saealib import SaealibError as BaseErr
        from saealib import ValidationError as ValErr

        assert BaseErr is SaealibError
        assert ValErr is ValidationError
        assert CfgErr is ConfigurationError


class TestMinimizeBoundary:
    def test_unknown_algorithm(self):
        with pytest.raises(ValidationError):
            saealib.minimize(lambda x: x, dim=1, lb=[0], ub=[1], algorithm="unknown")

    def test_unknown_algorithm_still_value_error(self):
        with pytest.raises(ValueError):
            saealib.minimize(lambda x: x, dim=1, lb=[0], ub=[1], algorithm="unknown")

    def test_surrogate_none(self):
        with pytest.raises(ValidationError):
            saealib.minimize(lambda x: x, dim=1, lb=[0], ub=[1], surrogate=None)

    def test_unknown_surrogate(self):
        with pytest.raises(ValidationError):
            saealib.minimize(lambda x: x, dim=1, lb=[0], ub=[1], surrogate="unknown")

    def test_unknown_strategy(self):
        with pytest.raises(ValidationError):
            saealib.minimize(lambda x: x, dim=1, lb=[0], ub=[1], strategy="unknown")

    def test_missing_dim(self):
        with pytest.raises(ValidationError):
            saealib.minimize(lambda x: x, lb=[0], ub=[1])

    def test_unknown_direction(self):
        with pytest.raises(ValidationError):
            saealib.minimize(lambda x: x, dim=1, lb=[0], ub=[1], direction=["minimise"])


class TestOptimizerBoundary:
    def _misconfigured_optimizer(self) -> saealib.Optimizer:
        """An Optimizer with an inconsistency that library defaults cannot
        resolve (comparator direction length mismatched with n_obj), as
        opposed to merely-unset components, which are auto-filled by
        Optimizer.resolve_defaults() as of the Registry-based defaults."""
        import numpy as np

        from saealib.comparators import ParetoComparator

        problem = saealib.Problem(
            func=lambda x: np.sum(x),
            dim=2,
            n_obj=1,
            direction=np.array([-1.0]),
            lb=[-5.0, -5.0],
            ub=[5.0, 5.0],
        )
        return saealib.Optimizer(problem).set_comparator(
            ParetoComparator(direction=np.array([1.0, 1.0]))
        )

    def test_misconfigured_run_raises_configuration_error(self):
        with pytest.raises(ConfigurationError, match="Optimizer misconfigured"):
            self._misconfigured_optimizer().run()

    def test_misconfigured_run_still_value_error(self):
        with pytest.raises(ValueError, match="Optimizer misconfigured"):
            self._misconfigured_optimizer().run()

    def test_misconfigured_iterate_raises_configuration_error(self):
        with pytest.raises(ConfigurationError, match="Optimizer misconfigured"):
            next(self._misconfigured_optimizer().iterate())
