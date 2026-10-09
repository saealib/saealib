import warnings

import numpy as np
import pytest

from saealib.exceptions import ValidationError
from saealib.registry import build, to_spec
from saealib.surrogate.rbf import RBFSurrogate
from saealib.surrogate.rbf_kernels import (
    CubicKernel,
    GaussianKernel,
    LinearKernel,
    MaternKernel,
    MultiquadricKernel,
    RBFKernel,
    ThinPlateSplineKernel,
)

TRAIN_X = np.array([[-1.0], [0.0], [1.0]])
TRAIN_Y = np.array([1.0, 2.0, -0.5])


def test_n_features_in_is_none_before_fit():
    surrogate = RBFSurrogate(kernel=GaussianKernel())

    assert surrogate.n_features_in_ is None


def test_n_features_in_tracks_most_recent_fit():
    surrogate = RBFSurrogate(kernel=GaussianKernel())
    surrogate.fit(TRAIN_X, TRAIN_Y)

    assert surrogate.n_features_in_ == 1


def test_default_degree_and_solver_interpolate_training_points():
    surrogate = RBFSurrogate(kernel=GaussianKernel())

    assert surrogate.solver == "lstsq"

    surrogate.fit(TRAIN_X, TRAIN_Y)

    np.testing.assert_allclose(
        surrogate.predict(TRAIN_X).value[:, 0],
        TRAIN_Y,
        rtol=1e-10,
        atol=1e-10,
    )


@pytest.mark.parametrize("polynomial_degree", [0, 1])
def test_polynomial_terms_interpolate_training_points(polynomial_degree):
    surrogate = RBFSurrogate(
        kernel=GaussianKernel(),
        polynomial_degree=polynomial_degree,
    )
    surrogate.fit(TRAIN_X, TRAIN_Y)

    np.testing.assert_allclose(
        surrogate.predict(TRAIN_X).value[:, 0],
        TRAIN_Y,
        rtol=1e-10,
        atol=1e-10,
    )


def test_degree_one_reproduces_linear_extrapolation_but_default_does_not():
    coefficient = 2.0
    intercept = 3.0
    train_y = coefficient * TRAIN_X[:, 0] + intercept
    outside_x = np.array([[3.0]])
    expected = coefficient * outside_x[:, 0] + intercept

    polynomial_surrogate = RBFSurrogate(
        kernel=GaussianKernel(),
        polynomial_degree=1,
    )
    polynomial_surrogate.fit(TRAIN_X, train_y)
    polynomial_prediction = polynomial_surrogate.predict(outside_x).value[:, 0]

    default_surrogate = RBFSurrogate(kernel=GaussianKernel())
    default_surrogate.fit(TRAIN_X, train_y)
    default_prediction = default_surrogate.predict(outside_x).value[:, 0]

    np.testing.assert_allclose(
        polynomial_prediction,
        expected,
        rtol=1e-10,
        atol=1e-10,
    )
    assert np.max(np.abs(default_prediction - expected)) > 1e-3


def test_thin_plate_spline_kernel_with_degree_one_interpolates_training_points():
    surrogate = RBFSurrogate(
        kernel=ThinPlateSplineKernel(),
        polynomial_degree=1,
    )
    surrogate.fit(TRAIN_X, TRAIN_Y)

    np.testing.assert_allclose(
        surrogate.predict(TRAIN_X).value[:, 0],
        TRAIN_Y,
        rtol=1e-10,
        atol=1e-10,
    )


def test_tikhonov_residual_increases_with_alpha_and_approaches_solve():
    solve_surrogate = RBFSurrogate(kernel=GaussianKernel())
    solve_surrogate.fit(TRAIN_X, TRAIN_Y)
    solve_prediction = solve_surrogate.predict(TRAIN_X).value[:, 0]

    predictions = []
    residuals = []
    for alpha in (1e-6, 1e-3, 1.0):
        surrogate = RBFSurrogate(
            kernel=GaussianKernel(),
            solver="tikhonov",
            alpha=alpha,
        )
        surrogate.fit(TRAIN_X, TRAIN_Y)
        prediction = surrogate.predict(TRAIN_X).value[:, 0]
        predictions.append(prediction)
        residuals.append(np.linalg.norm(prediction - TRAIN_Y))

    assert residuals[0] > 1e-10
    assert residuals[0] < residuals[1] < residuals[2]
    np.testing.assert_allclose(
        predictions[0],
        solve_prediction,
        rtol=1e-5,
        atol=1e-5,
    )


def test_lstsq_returns_finite_values_for_rank_deficient_training_data():
    train_x = np.array([[-1.0], [0.0], [0.0], [1.0]])
    train_y = np.array([1.0, 2.0, 2.0, 3.0])
    test_x = np.array([[-1.0], [0.0], [1.0], [2.0]])
    surrogate = RBFSurrogate(
        kernel=GaussianKernel(),
        polynomial_degree=1,
        solver="lstsq",
    )

    surrogate.fit(train_x, train_y)

    assert np.isfinite(surrogate.predict(test_x).value).all()


def test_nonfinite_solve_result_uses_singular_nan_fallback(monkeypatch):
    surrogate = RBFSurrogate(kernel=GaussianKernel(), solver="solve")
    monkeypatch.setattr(
        np.linalg,
        "solve",
        lambda a, rhs: np.resize((np.inf, -np.inf), rhs.shape),
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        surrogate.fit(TRAIN_X, TRAIN_Y)
        prediction = surrogate.predict(TRAIN_X).value

    assert surrogate._models is not None
    assert surrogate._models[0]._last_singular
    assert np.isnan(prediction).all()


@pytest.mark.parametrize("polynomial_degree", [-1, 2, 0.5, True, False])
def test_invalid_polynomial_degree_raises_validation_error(polynomial_degree):
    with pytest.raises(ValidationError):
        RBFSurrogate(
            kernel=GaussianKernel(),
            polynomial_degree=polynomial_degree,
        )


@pytest.mark.parametrize("polynomial_degree", [-1, 2, 0.5, True, False])
def test_invalid_polynomial_degree_assignment_raises_validation_error(
    polynomial_degree,
):
    surrogate = RBFSurrogate(kernel=GaussianKernel())

    with pytest.raises(ValidationError):
        surrogate.polynomial_degree = polynomial_degree


@pytest.mark.parametrize("solver", ["invalid", "qr"])
def test_invalid_solver_raises_validation_error(solver):
    with pytest.raises(ValidationError):
        RBFSurrogate(kernel=GaussianKernel(), solver=solver)


@pytest.mark.parametrize("solver", ["solve", "lstsq", "tikhonov"])
@pytest.mark.parametrize("alpha", [0.0, -1.0])
def test_nonpositive_alpha_raises_validation_error(solver, alpha):
    with pytest.raises(ValidationError):
        RBFSurrogate(kernel=GaussianKernel(), solver=solver, alpha=alpha)


@pytest.mark.parametrize(
    "alpha",
    [0, -1, float("nan"), float("inf"), float("-inf"), True, False, "0.1", None],
)
def test_invalid_alpha_values_raise_validation_error(alpha):
    with pytest.raises(ValidationError):
        RBFSurrogate(kernel=GaussianKernel(), alpha=alpha)


@pytest.mark.parametrize("alpha", [1e-8, 1, np.float32(1e-8), np.int32(2)])
def test_valid_numeric_alpha_types_are_accepted(alpha):
    surrogate = RBFSurrogate(kernel=GaussianKernel(), alpha=alpha)

    assert surrogate.alpha == alpha


@pytest.mark.parametrize(
    ("kernel", "polynomial_degree"),
    [
        (LinearKernel(), 0),
        (CubicKernel(), 1),
        (MultiquadricKernel(), 0),
    ],
)
def test_additional_cpd_kernels_interpolate_training_points(kernel, polynomial_degree):
    surrogate = RBFSurrogate(
        kernel=kernel,
        polynomial_degree=polynomial_degree,
    )
    surrogate.fit(TRAIN_X, TRAIN_Y)

    np.testing.assert_allclose(
        surrogate.predict(TRAIN_X).value[:, 0],
        TRAIN_Y,
        rtol=1e-10,
        atol=1e-10,
    )


@pytest.mark.parametrize("nu", [0.5, 1.5, 2.5])
def test_matern_kernels_interpolate_training_points_with_default_polynomial_term(nu):
    surrogate = RBFSurrogate(kernel=MaternKernel(nu=nu))
    surrogate.fit(TRAIN_X, TRAIN_Y)

    np.testing.assert_allclose(
        surrogate.predict(TRAIN_X).value[:, 0],
        TRAIN_Y,
        rtol=1e-10,
        atol=1e-10,
    )


def test_matern_kernel_rejects_invalid_nu_at_construction_time():
    with pytest.raises(ValidationError, match=r"nu must be 0\.5, 1\.5, or 2\.5"):
        MaternKernel(nu=1.0)


def test_repeated_singular_fits_warn_only_once(caplog):
    duplicate_x = np.array([[0.0], [0.0], [0.0]])
    duplicate_y = np.array([1.0, 2.0, 3.0])
    surrogate = RBFSurrogate(kernel=GaussianKernel(length_scale=1.0), solver="solve")

    with caplog.at_level("WARNING", logger="saealib.surrogate.rbf"):
        surrogate.fit(duplicate_x, duplicate_y)
        surrogate.fit(duplicate_x, duplicate_y)
        surrogate.fit(duplicate_x, duplicate_y)

    warnings = [r for r in caplog.records if r.levelname == "WARNING"]
    assert len(warnings) == 1


def test_singular_fit_after_clean_fit_warns_again(caplog):
    duplicate_x = np.array([[0.0], [0.0], [0.0]])
    duplicate_y = np.array([1.0, 2.0, 3.0])
    surrogate = RBFSurrogate(kernel=GaussianKernel(length_scale=1.0), solver="solve")

    with caplog.at_level("WARNING", logger="saealib.surrogate.rbf"):
        surrogate.fit(duplicate_x, duplicate_y)
        surrogate.fit(TRAIN_X, TRAIN_Y)
        surrogate.fit(duplicate_x, duplicate_y)

    warnings = [r for r in caplog.records if r.levelname == "WARNING"]
    assert len(warnings) == 2


def test_singular_fit_after_failed_fit_warns_again(caplog):
    # RBF kernel block is rank-deficient (duplicate x=0.0 rows) while the
    # median pairwise distance is nonzero, so kernel.resolve() succeeds and
    # the singularity surfaces at the linear-solve step instead.
    train_x = np.array([[-1.0], [0.0], [0.0], [1.0]])
    train_y = np.array([1.0, 2.0, 2.0, 3.0])
    surrogate = RBFSurrogate(kernel=GaussianKernel(), solver="solve")

    with caplog.at_level("WARNING", logger="saealib.surrogate.rbf"):
        surrogate.fit(train_x, train_y)
        # A single training point makes kernel.resolve()'s auto length_scale
        # heuristic fail (no pairwise distance to take a median of) — an
        # internal fit failure that goes through _invalidate_fit(), unlike a
        # public-boundary validation failure which leaves prior state alone.
        with pytest.raises(ValidationError):
            surrogate.fit(np.array([[0.0]]), np.array([5.0]))
        # Pin the precondition this test depends on: the failure above must
        # actually invalidate fitted state (and thus _last_singular) via the
        # atomic-fit except branch, not merely raise. If a future change made
        # this an early boundary-validation rejection instead, prior state
        # (including _last_singular) would survive and this test would keep
        # passing at 2 warnings for the wrong reason.
        assert surrogate._models is None
        surrogate.fit(train_x, train_y)

    warnings = [r for r in caplog.records if r.levelname == "WARNING"]
    assert len(warnings) == 2


@pytest.mark.parametrize("kernel", [GaussianKernel(), MaternKernel()])
def test_explicit_none_disables_polynomial_term_for_kernel_without_requirement(
    kernel,
):
    surrogate = RBFSurrogate(kernel=kernel, polynomial_degree=None)

    assert surrogate.polynomial_degree is None
    assert surrogate.resolved_polynomial_degree is None


@pytest.mark.parametrize(
    ("kernel", "expected_degree"),
    [
        (GaussianKernel(), 0),
        (LinearKernel(), 0),
        (CubicKernel(), 1),
        (ThinPlateSplineKernel(), 1),
        (MultiquadricKernel(), 0),
        (MaternKernel(), 0),
    ],
)
def test_auto_polynomial_degree_resolution(kernel, expected_degree):
    surrogate = RBFSurrogate(kernel=kernel)

    assert surrogate.polynomial_degree == "auto"
    assert surrogate.resolved_polynomial_degree == expected_degree


def test_polynomial_degree_exposes_configured_value_separately():
    surrogate = RBFSurrogate(kernel=GaussianKernel())

    assert surrogate.polynomial_degree == "auto"
    assert surrogate.resolved_polynomial_degree == 0

    surrogate.polynomial_degree = 1
    assert surrogate.polynomial_degree == 1
    assert surrogate.resolved_polynomial_degree == 1

    surrogate.polynomial_degree = None
    assert surrogate.polynomial_degree is None
    assert surrogate.resolved_polynomial_degree is None


def test_default_polynomial_degree_round_trips_through_registry():
    rebuilt = build(to_spec(RBFSurrogate(kernel=GaussianKernel())))

    assert rebuilt.polynomial_degree == "auto"
    assert rebuilt.resolved_polynomial_degree == 0


def test_resolved_kernel_tracks_fit_and_clears_after_failed_fit():
    configured_kernel = GaussianKernel()
    surrogate = RBFSurrogate(kernel=configured_kernel)

    assert surrogate.resolved_kernel is None
    expected_kernel = surrogate.kernel.resolve(TRAIN_X)
    surrogate.fit(TRAIN_X, TRAIN_Y)

    assert surrogate.resolved_kernel == expected_kernel
    assert surrogate.resolved_kernel is not configured_kernel
    assert configured_kernel.length_scale is None
    assert surrogate.resolved_kernel.length_scale == expected_kernel.length_scale

    degenerate_x = np.array([[1.0], [1.0], [1.0]])
    with pytest.raises(ValidationError):
        surrogate.fit(degenerate_x, np.ones(3))

    assert surrogate.resolved_kernel is None


def test_no_polynomial_term_fits_large_offset_without_centering():
    train_x = np.linspace(-20.0, 20.0, 40).reshape(-1, 1)
    train_y = 1e8 + train_x[:, 0] ** 2
    surrogate = RBFSurrogate(
        kernel=GaussianKernel(length_scale=2.0),
        polynomial_degree=None,
    )

    surrogate.fit(train_x, train_y)
    test_x = np.array([[-19.5], [-10.25], [-0.5], [0.5], [10.25], [19.5]])
    prediction = surrogate.predict(test_x).value[:, 0]
    kernel_matrix = np.exp(-0.5 * ((train_x - train_x.T) / 2.0) ** 2)
    test_kernel = np.exp(-0.5 * ((train_x - test_x.T) / 2.0) ** 2)
    expected = test_kernel.T @ np.linalg.solve(kernel_matrix, train_y)

    np.testing.assert_allclose(prediction, expected, rtol=1e-12, atol=1e-5)


def test_default_polynomial_term_preserves_holdout_offset_invariance():
    train_x = np.linspace(-3.0, 3.0, 9).reshape(-1, 1)
    test_x = np.array([[-2.75], [-1.8], [-1.25], [-0.4], [0.4], [1.25], [1.8], [2.75]])
    train_values = np.sin(train_x[:, 0]) + 0.1 * train_x[:, 0] ** 2
    test_values = np.sin(test_x[:, 0]) + 0.1 * test_x[:, 0] ** 2
    errors = []

    for offset in (0.0, 1e3, 1e6):
        surrogate = RBFSurrogate(kernel=GaussianKernel(length_scale=1.0))
        surrogate.fit(train_x, train_values + offset)
        prediction = surrogate.predict(test_x).value[:, 0]
        errors.append(np.max(np.abs(prediction - (test_values + offset))))

    np.testing.assert_allclose(errors, errors[0], rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize(
    ("kernel", "polynomial_degree"),
    [
        (CubicKernel(), 0),
        (ThinPlateSplineKernel(), 0),
        (ThinPlateSplineKernel(), None),
        (LinearKernel(), None),
        (CubicKernel(), None),
        (MultiquadricKernel(), None),
    ],
)
def test_invalid_polynomial_degree_kernel_combinations_raise_validation_error(
    kernel, polynomial_degree
):
    with pytest.raises(ValidationError):
        RBFSurrogate(
            kernel=kernel,
            polynomial_degree=polynomial_degree,
        )


def test_linear_kernel_with_degree_one_is_allowed():
    surrogate = RBFSurrogate(
        kernel=LinearKernel(),
        polynomial_degree=1,
    )

    surrogate.fit(TRAIN_X, TRAIN_Y)
    surrogate.predict(TRAIN_X)


@pytest.mark.parametrize(
    ("attribute", "value"),
    [
        ("kernel", GaussianKernel(length_scale=0.5)),
        ("polynomial_degree", 0),
        ("solver", "lstsq"),
        ("alpha", 1e-3),
    ],
)
def test_configuration_replacement_invalidates_fitted_state(attribute, value):
    surrogate = RBFSurrogate(kernel=GaussianKernel())
    surrogate.fit(TRAIN_X, TRAIN_Y)
    setattr(surrogate, attribute, value)

    with pytest.raises(RuntimeError):
        surrogate.predict(TRAIN_X)

    surrogate.fit(TRAIN_X, TRAIN_Y)
    surrogate.predict(TRAIN_X)


def _old_style_callable_kernel(x1, x2):
    return np.exp(-np.linalg.norm(x1 - x2))


@pytest.mark.parametrize(
    ("attribute", "value"),
    [
        ("polynomial_degree", 2),
        ("solver", "bad"),
        ("alpha", 0.0),
        ("kernel", _old_style_callable_kernel),
        ("kernel", object()),
    ],
)
def test_invalid_configuration_replacement_preserves_fitted_state(attribute, value):
    surrogate = RBFSurrogate(kernel=GaussianKernel())
    surrogate.fit(TRAIN_X, TRAIN_Y)

    with pytest.raises(ValidationError):
        setattr(surrogate, attribute, value)

    surrogate.predict(TRAIN_X)


@pytest.mark.parametrize("kernel", [_old_style_callable_kernel, object(), "gaussian"])
def test_non_rbf_kernel_rejected_at_construction(kernel):
    with pytest.raises(ValidationError, match="RBFKernel"):
        RBFSurrogate(kernel=kernel)


def test_incompatible_kernel_swap_preserves_fitted_state():
    surrogate = RBFSurrogate(
        kernel=GaussianKernel(),
        polynomial_degree=None,
    )
    surrogate.fit(TRAIN_X, TRAIN_Y)

    with pytest.raises(ValidationError):
        surrogate.kernel = ThinPlateSplineKernel()

    assert isinstance(surrogate.kernel, GaussianKernel)
    surrogate.predict(TRAIN_X)


def test_auto_polynomial_degree_reresolves_on_kernel_swap():
    surrogate = RBFSurrogate(kernel=GaussianKernel())
    assert surrogate.resolved_polynomial_degree == 0

    surrogate.kernel = LinearKernel()

    assert surrogate.polynomial_degree == "auto"
    assert surrogate.resolved_polynomial_degree == 0


def test_failed_fit_clears_previous_fitted_state():
    surrogate = RBFSurrogate(
        kernel=GaussianKernel(length_scale=1.0),
        polynomial_degree=1,
    )
    surrogate.fit(TRAIN_X, TRAIN_Y)

    degenerate_x = np.array([[1.0], [1.0], [1.0]])
    with pytest.raises(ValidationError, match=r"rank|degenerate"):
        surrogate.fit(degenerate_x, np.ones(3))

    assert surrogate.resolved_kernel is None
    assert surrogate.n_features_in_ is None
    with pytest.raises(RuntimeError):
        surrogate.predict(TRAIN_X)


@pytest.mark.parametrize(
    ("train_x", "train_y"),
    [
        (TRAIN_X[:, 0], TRAIN_Y),
        (np.empty((0, 1)), np.empty(0)),
        (np.array([[np.nan], [0.0], [1.0]]), TRAIN_Y),
        (np.array([[np.inf], [0.0], [1.0]]), TRAIN_Y),
        (TRAIN_X, np.array([1.0, np.nan, 3.0])),
        (TRAIN_X, np.array([1.0, np.inf, 3.0])),
        (TRAIN_X, np.ones(2)),
        (np.empty((3, 0)), TRAIN_Y),
        (TRAIN_X, np.empty((3, 0))),
    ],
    ids=[
        "train_x_1d",
        "empty_training_set",
        "train_x_nan",
        "train_x_inf",
        "train_y_nan",
        "train_y_inf",
        "sample_count_mismatch",
        "train_x_zero_features",
        "train_y_zero_objectives",
    ],
)
def test_fit_rejects_invalid_training_inputs(train_x, train_y):
    surrogate = RBFSurrogate(kernel=GaussianKernel(length_scale=1.0))

    with pytest.raises(ValidationError):
        surrogate.fit(train_x, train_y)


def test_fit_rejects_non_1d_or_2d_training_targets():
    surrogate = RBFSurrogate(kernel=GaussianKernel(length_scale=1.0))

    with pytest.raises(ValidationError):
        surrogate.fit(TRAIN_X, np.ones((3, 1, 1)))


def test_predict_rejects_nonfinite_inputs_and_feature_count_mismatch():
    surrogate = RBFSurrogate(kernel=GaussianKernel())
    surrogate.fit(TRAIN_X, TRAIN_Y)

    for test_x in (np.array([[np.nan]]), np.array([[np.inf]])):
        with pytest.raises(ValidationError):
            surrogate.predict(test_x)

    with pytest.raises(ValidationError, match="features"):
        surrogate.predict(np.ones((1, 2)))


def test_predict_before_fit_raises_runtime_error():
    surrogate = RBFSurrogate(kernel=GaussianKernel())

    with pytest.raises(RuntimeError):
        surrogate.predict(TRAIN_X)


class _CustomKernel(RBFKernel):
    def evaluate(self, r: np.ndarray) -> np.ndarray:
        return np.exp(-r)


def test_custom_rbf_kernel_subclass_works_with_surrogate():
    surrogate = RBFSurrogate(kernel=_CustomKernel())

    surrogate.fit(TRAIN_X, TRAIN_Y)
    prediction = surrogate.predict(TRAIN_X).value

    assert np.isfinite(prediction).all()


def test_fixed_length_scale_bypasses_auto_resolution():
    resolved = GaussianKernel(length_scale=0.5).resolve(TRAIN_X)

    assert resolved.length_scale == 0.5


@pytest.mark.parametrize(
    "train_x",
    [
        np.array([[0.0]]),
        np.array([[1.0], [1.0], [1.0]]),
    ],
)
def test_degenerate_auto_length_scale_raises_validation_error(train_x):
    surrogate = RBFSurrogate(kernel=GaussianKernel())

    with pytest.raises(ValidationError):
        surrogate.fit(train_x, np.ones(len(train_x)))
