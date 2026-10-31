"""Scikit-learn surrogate adapter and convenience classes."""

from __future__ import annotations

from typing import Any

import numpy as np

from saealib.surrogate.base import ComparisonSurrogate, RegressionSurrogate
from saealib.surrogate.prediction import SurrogatePrediction


class SklearnSurrogate(RegressionSurrogate):
    """
    Surrogate adapter for scikit-learn compatible estimators.

    Wraps any estimator that implements ``fit(X, y)`` and ``predict(X)``.
    Multi-objective problems are handled by fitting one cloned estimator
    per objective, following the same pattern as ``RBFSurrogate``.

    Parameters
    ----------
    estimator : sklearn estimator
        A scikit-learn compatible regressor instance (e.g. ``SVR()``).
        The estimator is cloned once per objective on the first ``fit`` call.

    Raises
    ------
    ImportError
        If scikit-learn is not installed.
    """

    def __init__(self, estimator: object) -> None:
        try:
            import sklearn  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "scikit-learn is required for SklearnSurrogate. "
                "Install it with: pip install saealib[sklearn]"
            ) from e
        self.estimator = estimator
        self._models: list | None = None
        self.n_obj: int | None = None

    def fit(self, train_x: np.ndarray, train_y: np.ndarray) -> None:
        """
        Fit the surrogate model.

        Parameters
        ----------
        train_x : np.ndarray
            Training input data. shape: (n_samples, n_features)
        train_y : np.ndarray
            Training output data. shape: (n_samples, n_obj) or (n_samples,).
        """
        from sklearn.base import clone

        arr = np.asarray(train_y, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        n_obj = arr.shape[1]

        if self._models is None or n_obj != self.n_obj:
            self.n_obj = n_obj
            self._models = [clone(self.estimator) for _ in range(n_obj)]

        for i, model in enumerate(self._models):
            model.fit(train_x, arr[:, i])

    def predict(self, test_x: np.ndarray) -> SurrogatePrediction:
        """
        Predict using the surrogate model.

        Parameters
        ----------
        test_x : np.ndarray
            Input data. shape: (n_samples, n_features) or (n_features,)

        Returns
        -------
        SurrogatePrediction
            prediction.value shape: (n_samples, n_obj)
            prediction.std is None
        """
        test = np.asarray(test_x)
        if test.ndim == 1:
            test = test.reshape(1, -1)
        assert self._models is not None
        preds = [m.predict(test) for m in self._models]
        value = np.column_stack(preds)
        return SurrogatePrediction.objective(value=value)


class SklearnSVMSurrogate(SklearnSurrogate):
    """
    Support Vector Regression surrogate.

    Convenience wrapper around ``SklearnSurrogate`` using
    ``sklearn.svm.SVR``.

    Parameters
    ----------
    **kwargs
        Keyword arguments forwarded to ``sklearn.svm.SVR``.
    """

    def __init__(self, **kwargs: object) -> None:
        from sklearn.svm import SVR

        self.kwargs = kwargs
        super().__init__(SVR(**kwargs))


class SklearnNNSurrogate(SklearnSurrogate):
    """
    Multi-layer Perceptron surrogate.

    Convenience wrapper around ``SklearnSurrogate`` using
    ``sklearn.neural_network.MLPRegressor``.

    Parameters
    ----------
    **kwargs
        Keyword arguments forwarded to ``sklearn.neural_network.MLPRegressor``.
    """

    def __init__(self, **kwargs: object) -> None:
        from sklearn.neural_network import MLPRegressor

        self.kwargs = kwargs
        super().__init__(MLPRegressor(**kwargs))


class SklearnRFRSurrogate(SklearnSurrogate):
    """
    Random Forest Regressor surrogate.

    Convenience wrapper around ``SklearnSurrogate`` using
    ``sklearn.ensemble.RandomForestRegressor``.

    Parameters
    ----------
    **kwargs
        Keyword arguments forwarded to
        ``sklearn.ensemble.RandomForestRegressor``.
    """

    def __init__(self, **kwargs: object) -> None:
        from sklearn.ensemble import RandomForestRegressor

        self.kwargs = kwargs
        super().__init__(RandomForestRegressor(**kwargs))


class SklearnGPRSurrogate(SklearnSurrogate):
    """
    Gaussian Process Regressor surrogate.

    Convenience wrapper around ``SklearnSurrogate`` using
    ``sklearn.gaussian_process.GaussianProcessRegressor``.
    Provides uncertainty estimates via ``SurrogatePrediction.std``,
    enabling compatibility with acquisition functions such as EI, LCB,
    PoF, and MaxUncertainty.

    Parameters
    ----------
    **kwargs
        Keyword arguments forwarded to
        ``sklearn.gaussian_process.GaussianProcessRegressor``
        (e.g. ``kernel``, ``alpha``, ``n_restarts_optimizer``).

    References
    ----------
    :cite:`sacks1989dace`: Sacks, J., Welch, W. J., Mitchell, T. J., & Wynn,
    H. P. (1989). Design and analysis of computer experiments. *Statistical
    Science*, 4(4), 409-423.

    :cite:`rasmussen2006gpml`: Rasmussen, C. E., & Williams, C. K. I.
    (2006). *Gaussian Processes for Machine Learning*. MIT Press.
    """

    provides_uncertainty: bool = True

    def __init__(self, **kwargs: object) -> None:
        from sklearn.gaussian_process import GaussianProcessRegressor

        self.kwargs = kwargs
        super().__init__(GaussianProcessRegressor(**kwargs))

    def predict(self, test_x: np.ndarray) -> SurrogatePrediction:
        """
        Predict mean and standard deviation using the GP models.

        Parameters
        ----------
        test_x : np.ndarray
            Input data. shape: (n_samples, n_features) or (n_features,)

        Returns
        -------
        SurrogatePrediction
            prediction.value shape: (n_samples, n_obj)
            prediction.std  shape: (n_samples, n_obj)
        """
        test = np.asarray(test_x)
        if test.ndim == 1:
            test = test.reshape(1, -1)
        assert self._models is not None
        means, stds = [], []
        for m in self._models:
            mu, sigma = m.predict(test, return_std=True)
            means.append(mu)
            stds.append(sigma)
        value = np.column_stack(means)
        std = np.column_stack(stds)
        return SurrogatePrediction.objective(value=value, std=std)


class SklearnXGBSurrogate(SklearnSurrogate):
    """
    XGBoost surrogate.

    Convenience wrapper around ``SklearnSurrogate`` using
    ``xgboost.XGBRegressor``.

    Parameters
    ----------
    **kwargs
        Keyword arguments forwarded to ``xgboost.XGBRegressor``.

    Raises
    ------
    ImportError
        If xgboost is not installed.
    """

    def __init__(self, **kwargs: object) -> None:
        try:
            from xgboost import XGBRegressor
        except ImportError as e:
            raise ImportError(
                "xgboost is required for SklearnXGBSurrogate. "
                "Install it with: pip install saealib[xgboost]"
            ) from e
        self.kwargs = kwargs
        super().__init__(XGBRegressor(**kwargs))


class SklearnLGBMSurrogate(SklearnSurrogate):
    """
    LightGBM surrogate.

    Convenience wrapper around ``SklearnSurrogate`` using
    ``lightgbm.LGBMRegressor``.

    Parameters
    ----------
    **kwargs
        Keyword arguments forwarded to ``lightgbm.LGBMRegressor``.

    Raises
    ------
    ImportError
        If lightgbm is not installed.
    """

    def __init__(self, **kwargs: object) -> None:
        try:
            from lightgbm import LGBMRegressor
        except ImportError as e:
            raise ImportError(
                "lightgbm is required for SklearnLGBMSurrogate. "
                "Install it with: pip install saealib[lightgbm]"
            ) from e
        self.kwargs = kwargs
        super().__init__(LGBMRegressor(**kwargs))  # type: ignore  # LightGBM stubs mistype __init__ kwargs


class SklearnClassificationSurrogate(ComparisonSurrogate):
    """
    Surrogate adapter for scikit-learn compatible binary classifiers.

    Wraps any estimator that implements ``fit(X, y)`` and
    ``predict_proba(X)``.  ``train_y`` must be binary labels ``{0, 1}``.

    Parameters
    ----------
    estimator : sklearn estimator
        A scikit-learn compatible classifier with ``predict_proba`` support
        (e.g. ``SVC(probability=True)`` or ``RandomForestClassifier()``).

    Raises
    ------
    ImportError
        If scikit-learn is not installed.
    """

    def __init__(self, estimator: object) -> None:
        try:
            import sklearn  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "scikit-learn is required for SklearnClassificationSurrogate. "
                "Install it with: pip install saealib[sklearn]"
            ) from e
        self.estimator = estimator
        self._model: Any = None

    def fit(self, train_x: np.ndarray, train_y: np.ndarray) -> None:
        """
        Fit the classifier.

        Parameters
        ----------
        train_x : np.ndarray
            Training input data. shape: (n_samples, n_features)
        train_y : np.ndarray
            Binary labels. shape: (n_samples,), values in ``{0, 1}``.
        """
        from sklearn.base import clone

        labels = np.asarray(train_y, dtype=float).ravel()
        if self._model is None:
            self._model = clone(self.estimator)
        self._model.fit(train_x, labels)

    def predict_proba(self, test_x: np.ndarray) -> SurrogatePrediction:
        """
        Return class-1 probability estimates.

        Parameters
        ----------
        test_x : np.ndarray
            Input data. shape: (n_samples, n_features) or (n_features,)

        Returns
        -------
        SurrogatePrediction
            ``value`` shape: ``(n_samples, 1)``, values in ``[0, 1]``.
        """
        test = np.asarray(test_x)
        if test.ndim == 1:
            test = test.reshape(1, -1)
        assert self._model is not None
        proba = self._model.predict_proba(test)
        return SurrogatePrediction.objective(value=proba[:, 1:2])


class SklearnSVCClassificationSurrogate(SklearnClassificationSurrogate):
    """
    Support Vector Classification surrogate.

    Convenience wrapper around ``SklearnClassificationSurrogate`` using
    ``sklearn.svm.SVC(probability=True)``.

    Parameters
    ----------
    **kwargs
        Keyword arguments forwarded to ``sklearn.svm.SVC``.
        ``probability=True`` is always set.
    """

    def __init__(self, **kwargs: object) -> None:
        from sklearn.svm import SVC

        kwargs["probability"] = True
        self.kwargs = kwargs
        super().__init__(SVC(**kwargs))


class SklearnRFCClassificationSurrogate(SklearnClassificationSurrogate):
    """
    Random Forest Classification surrogate.

    Convenience wrapper around ``SklearnClassificationSurrogate`` using
    ``sklearn.ensemble.RandomForestClassifier``.

    Parameters
    ----------
    **kwargs
        Keyword arguments forwarded to
        ``sklearn.ensemble.RandomForestClassifier``.
    """

    def __init__(self, **kwargs: object) -> None:
        from sklearn.ensemble import RandomForestClassifier

        self.kwargs = kwargs
        super().__init__(RandomForestClassifier(**kwargs))
