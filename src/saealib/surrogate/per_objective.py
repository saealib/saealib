"""PerObjectiveSurrogate: assign a different surrogate per objective."""

from __future__ import annotations

import numpy as np

from saealib.exceptions import ValidationError
from saealib.surrogate.base import RegressionSurrogate, Surrogate
from saealib.surrogate.prediction import SurrogatePrediction


class PerObjectiveSurrogate(RegressionSurrogate):
    """
    Assigns a different surrogate model to each objective function.

    Each surrogate in ``surrogates`` is fitted on the corresponding column
    of ``train_y``. Predictions are concatenated along the objective axis.

    If every surrogate has ``provides_uncertainty = True``, the returned
    ``SurrogatePrediction.std`` is populated by stacking each surrogate's
    ``std`` output; otherwise ``std`` is ``None``.

    Parameters
    ----------
    surrogates : list[Surrogate]
        One surrogate per objective. Must be non-empty.

    Raises
    ------
    ValueError
        If ``surrogates`` is empty, or if the number of objectives in
        ``train_y`` does not match ``len(surrogates)`` at fit time.
    """

    def __init__(self, surrogates: list[Surrogate]) -> None:
        if not surrogates:
            raise ValidationError(
                "PerObjectiveSurrogate requires at least one surrogate."
            )
        self.surrogates = surrogates

    @property
    def provides_uncertainty(self) -> bool:
        """True only when every constituent surrogate provides uncertainty."""
        return all(s.provides_uncertainty for s in self.surrogates)

    def fit(self, train_x: np.ndarray, train_y: np.ndarray) -> None:
        """
        Fit each surrogate on its corresponding objective column.

        Parameters
        ----------
        train_x : np.ndarray
            Training input data. shape: (n_samples, n_features)
        train_y : np.ndarray
            Training output data. shape: (n_samples, n_obj) or (n_samples,).
        """
        arr = np.asarray(train_y, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        n_obj = arr.shape[1]
        if n_obj != len(self.surrogates):
            raise ValidationError(
                f"n_obj={n_obj} does not match len(surrogates)={len(self.surrogates)}"
            )
        for i, s in enumerate(self.surrogates):
            s.fit(train_x, arr[:, i])

    def predict(self, test_x: np.ndarray) -> SurrogatePrediction:
        """
        Predict by combining each surrogate's output.

        Parameters
        ----------
        test_x : np.ndarray
            Input data. shape: (n_samples, n_features) or (n_features,)

        Returns
        -------
        SurrogatePrediction
            prediction.value shape: (n_samples, n_obj)
            prediction.std shape: (n_samples, n_obj) if all surrogates provide
            uncertainty, otherwise None.
            prediction.x holds the (n_samples, n_features) query points
            passed to this call, set directly from ``test_x`` regardless of
            whether the wrapped surrogates populate their own ``x`` (needed
            by acquisition functions such as CORSDistance).
        """
        preds = [s.predict(test_x) for s in self.surrogates]
        value = np.column_stack([p.value for p in preds])
        std = None
        if self.provides_uncertainty:
            std = np.column_stack([p.std for p in preds])  # type: ignore  # provides_uncertainty guarantees non-None; ty can't narrow in list comp
        x = np.atleast_2d(np.asarray(test_x, dtype=float))
        return SurrogatePrediction.objective(value=value, std=std, x=x)
