---
primary_layer: layer3
page_type: concept
---

# Surrogate

`saealib` restricts the responsibility of the predictive model approximating the objective function to `Surrogate`, a swappable component.
`Surrogate` knows only fit/predict — it knows nothing about how a prediction is converted into a score ([AcquisitionFunction](acquisition_functions.md)), or where training data comes from ([TrainingSet](training_set.md), via [SurrogateManager](surrogate_manager.md)).

## Surrogate's role

`Surrogate` requires two methods to be implemented.

**`fit(train_x, train_y) -> None`**: Trains the model on inputs of shape `(n_samples, n_features)` and outputs of shape `(n_samples, n_obj)` (or `(n_samples,)` for single-objective).

**`predict(test_x) -> SurrogatePrediction`**: Returns predictions for inputs of shape `(n_samples, n_features)`.

The class attribute `provides_uncertainty: bool = False` indicates whether predictions carry uncertainty (standard deviation).
The default is `False`; only Gaussian Process implementations override it to `True`.

There are two marker base classes between `Surrogate` and `predict()`.

**`RegressionSurrogate`**: A marker for regression surrogates where `train_y` is a real-valued objective function output.

**`ComparisonSurrogate`**: For comparison surrogates where `train_y` is a `{0, 1}`-valued binary comparison label.
`predict_proba(test_x) -> SurrogatePrediction` (values are win probabilities in `[0, 1]`) is the primary interface, with `predict()` delegating to `predict_proba()` by default.

## SurrogatePrediction

`predict()`'s return value is `SurrogatePrediction`, a unified dataclass.

| Field | Content |
|---|---|
| `value` | The predicted value. Shape `(n_samples, n_obj)` |
| `std` | Uncertainty (standard deviation). `None` for surrogates that don't provide it |
| `label` | Class labels, only present for classification models |
| `metadata` | A `dict` holding implementation-specific auxiliary information, such as SHAP values |

`value` and `std` are convenience properties for the objective channel.
When a surrogate's prediction represents a quantity that isn't the objective value (such as a novelty score), this mechanism prevents contaminating things like pbest.
Archive-based criteria such as `NoveltyAcquisition` instead receive candidate and archive design points directly.

## Built-in Surrogates

**`RBFSurrogate(kernel, polynomial_degree="auto", solver="lstsq", alpha=1e-8)`**: A surrogate using RBF interpolation {cite}`gutmann2001rbf,regis2005cors,rasmussen2006gpml` (the origin of RBF interpolation itself is Hardy, 1971).
`kernel` is a required `RBFKernel` object with no default (built-in kernels are listed below); a custom kernel is a subclass implementing `evaluate(r)`.
`predict()` explicitly returns `std=None` (RBF interpolation provides no uncertainty).

| Parameter | Values | Role |
|---|---|---|
| `polynomial_degree` | `"auto"` (default) / `None` / `0` / `1` | Augments the system with a polynomial term (from `kernel.auto_polynomial_degree` / none / constant / linear). A conditionally positive definite kernel such as `ThinPlateSplineKernel` requires one; a strictly positive definite kernel such as `GaussianKernel` does not, but resolves `"auto"` to a constant term so that predictions stay unchanged when a constant is added to every objective value. |
| `solver` | `"lstsq"` (default) / `"solve"` / `"tikhonov"` | How the (possibly augmented) linear system is solved: least squares (tolerant of rank-deficient systems) / direct / ridge-regularized via `alpha`. |
| `alpha` | `1e-8` (default) | Ridge regularization strength, used only when `solver="tikhonov"`. |

`kernel` and `polynomial_degree` are the values as configured; `resolved_kernel` and `resolved_polynomial_degree` expose what a `fit()` call actually resolves and uses, keeping the configured `RBFKernel` instance itself immutable.
Reassigning `kernel`/`polynomial_degree`/`solver`/`alpha` invalidates any fitted state; `predict()` then raises `RuntimeError` until the next `fit()`.

Kernels carrying a `length_scale` field (in the same distance units as the input data) resolve it automatically to the median pairwise training-point distance when left `None`.

Each kernel declares two independent degrees. `min_polynomial_degree` is the lowest degree the kernel accepts — passing anything below it, `None` included, raises `ValidationError`. `auto_polynomial_degree` is what `polynomial_degree="auto"` resolves to, which may be higher than the minimum.

| Kernel | Formula | `min_polynomial_degree` | `"auto"` resolves to |
|---|---|---|---|
| `GaussianKernel` | $\exp\left(-\dfrac{r^2}{2\,\ell^2}\right)$ | none | `0` (constant) |
| `ThinPlateSplineKernel` | $r^2 \log r$ (no `length_scale`) | `1` (linear) | `1` (linear) |
| `LinearKernel` | $r$ (no `length_scale`) | `0` (constant) | `0` (constant) |
| `CubicKernel` | $r^3$ (no `length_scale`) | `1` (linear) | `1` (linear) |
| `MultiquadricKernel` | $\sqrt{r^2 + \ell^2}$ | `0` (constant) | `0` (constant) |
| `MaternKernel` (`nu=0.5/1.5/2.5`) | Matérn covariance, `length_scale` = $\ell$ | none | `0` (constant) |

All six kernels and all three `solver` values are supported: each is covered by tests and usable as shipped. Passing `polynomial_degree=None` to `GaussianKernel` or `MaternKernel` is likewise supported, and drops the constant term that `"auto"` would have added.

**`PerObjectiveSurrogate(surrogates)`**: A `RegressionSurrogate` subclass, a composite class that assigns a different surrogate per objective.
Raises `ValueError` at `fit` time if `train_y`'s column count doesn't match `len(surrogates)`.
`provides_uncertainty` is a composite judgment, returning `True` only if every constituent surrogate is `True`.

### External library adapters

Regression surrogates via a scikit-learn-compatible API carry a `Sklearn` prefix.

| Class | Model |
|---|---|
| `SklearnGPRSurrogate` | Gaussian Process {cite}`sacks1989dace,rasmussen2006gpml`. The sole implementation with `provides_uncertainty=True` |
| `SklearnRFRSurrogate` | Random Forest regression |
| `SklearnSVMSurrogate` | SVM |
| `SklearnNNSurrogate` | MLP |
| `SklearnXGBSurrogate` | XGBoost (the `xgboost` extra) |
| `SklearnLGBMSurrogate` | LightGBM (the `lightgbm` extra) |
| `TorchSurrogate` | PyTorch-based models (the `torch` extra) |

`SklearnGPRSurrogate` computes standard deviation from the GP kernel via `return_std=True`, and returns `provides_uncertainty=True`.

The following classes are classification surrogates aimed at feasibility prediction.

| Class | Model |
|---|---|
| `SklearnClassificationSurrogate` | Classification models compatible with scikit-learn in general |
| `SklearnRFCClassificationSurrogate` | Random Forest classification |
| `SklearnSVCClassificationSurrogate` | SVM classification |

See [TrainingSet](training_set.md)'s `FeasibilityClassificationSet` for how training data is extracted for these classification surrogates.
For pairwise comparison, use the dedicated `ComparisonSurrogate`-family implementations rather than these classification surrogates, paired with [SurrogateManager](surrogate_manager.md)'s `PairwiseSurrogateManager` and `PairwiseComparisonSet`.

See [Installation](../../getting_started/installation.md) for how to install each extra.

```{note}
Adapters for BoTorch/SMT surrogates, beyond scikit-learn/XGBoost/LightGBM/PyTorch, are not currently implemented in `saealib`.
`pyproject.toml` has no corresponding extra either.
pymoo has adapters, but at the [Problem](../problem_and_ranking/problem.md)/[Crossover](../search_algorithms/crossover.md)/[Mutation](../search_algorithms/mutation.md)/[Algorithm](../search_algorithms/algorithm.md) level, not as a `Surrogate` — pymoo doesn't provide surrogate models.
```

## Extension hooks

If you just want to add post-fit processing, you can add it to an existing `Surrogate` instance with `with_post_fit(fn)` instead of creating a new subclass.
`with_post_fit` doesn't modify the original instance — it returns a copy with `fn` added.

```python
from saealib import GaussianKernel, RBFSurrogate


def log_fit(train_x, train_y, ctx=None):
    print(f"fit on {len(train_x)} samples")


base = RBFSurrogate(kernel=GaussianKernel())
logged = base.with_post_fit(log_fit)
```

`fn`'s signature is `fn(train_x, train_y, ctx) -> None`.

## Implementing a custom Surrogate

If you need a custom predictive model, subclass `Surrogate` and implement `fit()`/`predict()`.
Choose `RegressionSurrogate` for regression, or `ComparisonSurrogate` (implementing `predict_proba()`) for comparison, as the class to subclass.

The following example is a simple regression surrogate that returns the nearest training point's objective value directly as the prediction.

```python
import numpy as np
from saealib import RegressionSurrogate, SurrogatePrediction


class NearestNeighborSurrogate(RegressionSurrogate):
    """A simple surrogate that returns the nearest point's objective value directly as the prediction."""

    def fit(self, train_x, train_y):
        self.train_x = np.asarray(train_x, dtype=float)
        self.train_y = np.asarray(train_y, dtype=float)

    def predict(self, test_x):
        test_x = np.atleast_2d(test_x)
        dists = np.linalg.norm(self.train_x[None, :, :] - test_x[:, None, :], axis=2)
        nearest = dists.argmin(axis=1)
        value = self.train_y[nearest]
        return SurrogatePrediction(value=value)
```

## Uncertainty support table

To use an uncertainty-based [AcquisitionFunction](acquisition_functions.md), `Surrogate` needs to return `std`.

| Class | `provides_uncertainty` |
|---|---|
| `SklearnGPRSurrogate` | `True` |
| `RBFSurrogate` / `SklearnRFRSurrogate` / `SklearnSVMSurrogate` / `SklearnNNSurrogate` / `SklearnXGBSurrogate` / `SklearnLGBMSurrogate` / `TorchSurrogate` | `False` |
| `PerObjectiveSurrogate` | `True` only if every constituent surrogate is `True` |

`Optimizer.validate()` checks uncertainty support per prediction channel. If an acquisition requires uncertainty and its matching surrogate does not provide it, validation reports an ERROR diagnostic.

## Related components

- [SurrogateManager](surrogate_manager.md): Coordinates `Surrogate`'s fit/predict
- [TrainingSet](training_set.md): How training data passed to `Surrogate` is extracted
- [AcquisitionFunction](acquisition_functions.md): Converts `predict()`'s result into a score
- [Surrogate accuracy evaluation and dynamic switching](surrogate_switching.md): Evaluating a surrogate's generalization performance
- [Installation](../../getting_started/installation.md): How to install each extra

## References

- {py:class}`saealib.Surrogate`
- {py:class}`saealib.RegressionSurrogate`
- {py:class}`saealib.ComparisonSurrogate`
- {py:class}`saealib.SurrogatePrediction`
- {py:class}`saealib.RBFSurrogate`
- {py:class}`saealib.RBFKernel`
- {py:class}`saealib.GaussianKernel`
- {py:class}`saealib.ThinPlateSplineKernel`
- {py:class}`saealib.LinearKernel`
- {py:class}`saealib.CubicKernel`
- {py:class}`saealib.MultiquadricKernel`
- {py:class}`saealib.MaternKernel`
- {py:class}`saealib.PerObjectiveSurrogate`
- {py:class}`saealib.SklearnGPRSurrogate`
- {py:class}`saealib.SklearnRFRSurrogate`
- {py:class}`saealib.SklearnSVMSurrogate`
- {py:class}`saealib.SklearnNNSurrogate`
- {py:class}`saealib.SklearnXGBSurrogate`
- {py:class}`saealib.SklearnLGBMSurrogate`
- {py:class}`saealib.TorchSurrogate`
- {py:class}`saealib.SklearnClassificationSurrogate`
- {py:class}`saealib.SklearnRFCClassificationSurrogate`
- {py:class}`saealib.SklearnSVCClassificationSurrogate`
