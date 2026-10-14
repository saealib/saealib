"""Surrogate managers for fitting models and producing predictions.

Responsibility split:
  Surrogate           -- fits a model and predicts SurrogatePrediction
  AcquisitionFunction -- converts SurrogatePrediction to scalar scores
  SurrogateManager    -- coordinates training and prediction; exposes predict()

Classes
-------
GlobalSurrogateManager
    Fits once on the full archive; batch predict.
LocalSurrogateManager
    Fits a local KNN model per candidate.
CompositeSurrogateManager
    Composes named predictions from multiple sub-managers.
PairwiseSurrogateManager
    Fits a pairwise classifier and predicts per-candidate win rates.

Combine functions
-----------------
product_combine
    Element-wise product of score arrays (e.g. EI x PoF).
rank_weighted_combine
    Returns a combine function that rank-normalises then takes a weighted average.
"""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np

from saealib.core.contracts import (
    MANY,
    ComponentContract,
    DataSpec,
    PartSpec,
    PortContract,
    PortDirection,
    PortSpec,
    ServiceRequirement,
    StateContract,
    Var,
)
from saealib.core.state import POPULATIONS_MAIN, RUNTIME_RNG
from saealib.exceptions import ValidationError
from saealib.registry import register
from saealib.surrogate.accuracy import AccuracyEvaluator, SurrogateAccuracy
from saealib.surrogate.base import ComparisonSurrogate, Surrogate
from saealib.surrogate.prediction import PredictionChannel, SurrogatePrediction
from saealib.surrogate.training_set import (
    ArchiveObjectiveSet,
    KNNObjectiveSet,
    PairwiseComparisonSet,
    TrainingSet,
)

if TYPE_CHECKING:
    from saealib.context import OptimizationState
    from saealib.population import Archive


class SurrogateManager(ABC):
    """
    Abstract base class for surrogate managers.

    A SurrogateManager is narrowed to training and prediction management
    It coordinates ``TrainingSet`` construction, model fitting, and prediction,
    and returns a ``SurrogatePrediction`` for callers.

    Attributes
    ----------
    last_accuracy : SurrogateAccuracy or None
        Accuracy metrics computed after the most recent :meth:`fit` call.
        ``None`` until the first fit or when no evaluator is configured.
    """

    last_accuracy: SurrogateAccuracy | None = None

    def contract(self) -> ComponentContract:
        """Return the surrogate-manager contract."""
        feature_schema = Var(name="F")
        objective_schema = Var(name="O")
        return ComponentContract(
            required_services=(ServiceRequirement(name="FeatureEncoder"),),
            ports={
                "fitter": PortContract(
                    inputs=(
                        PortSpec(
                            name="archive",
                            direction=PortDirection.INPUT,
                            data=DataSpec(kind="Population"),
                            cardinality=MANY,
                        ),
                    ),
                ),
                "predictor": PortContract(
                    inputs=(
                        PortSpec(
                            name="candidates",
                            direction=PortDirection.INPUT,
                            data=DataSpec(
                                kind="FeatureBatch",
                                bindings={"feature_schema": feature_schema},
                            ),
                            cardinality=MANY,
                        ),
                        PortSpec(
                            name="archive",
                            direction=PortDirection.INPUT,
                            data=DataSpec(kind="Population"),
                            cardinality=MANY,
                        ),
                    ),
                    outputs=(
                        PortSpec(
                            name="prediction",
                            direction=PortDirection.OUTPUT,
                            data=DataSpec(
                                kind="SurrogatePrediction",
                                bindings={"objective_schema": objective_schema},
                            ),
                            cardinality=MANY,
                        ),
                    ),
                ),
            },
        )

    def fit(
        self,
        archive: Archive,
        ctx: OptimizationState | None = None,
    ) -> None:
        """
        Pre-fit the surrogate on the archive.

        Call once before a sequence of ``predict(..., refit=False)``
        calls when the archive does not change between calls (e.g. the
        surrogate-only inner loop of ``GenerationBasedStrategy``).
        The default implementation is a no-op; override in managers that
        maintain a fitted surrogate model.

        Parameters
        ----------
        archive : Archive
            Archive of evaluated solutions used for surrogate training.
        ctx : OptimizationState or None, optional
            Current optimization context.
        """

    @abstractmethod
    def predict(
        self,
        candidates_x: np.ndarray,
        archive: Archive,
        ctx: OptimizationState | None = None,
        *,
        refit: bool = True,
    ) -> SurrogatePrediction:
        """
        Predict candidate solutions using the surrogate model.

        Parameters
        ----------
        candidates_x : np.ndarray
            Candidate design variable matrix. shape: (n_candidates, dim)
        archive : Archive
            Archive of evaluated solutions used for surrogate training.
        ctx : OptimizationState or None, optional
            Current optimization context. Passed to ``TrainingSet.build``
            for strategies that require comparator or population access.
        refit : bool, optional
            If ``True`` (default), fit the surrogate before predicting.
            Pass ``False`` after an explicit :meth:`fit` call to skip
            redundant re-fitting when the archive has not changed.

        Returns
        -------
        SurrogatePrediction
            One batched prediction covering every row of ``candidates_x``.
            Acquisition scoring is not performed here — pass the result to
            an :class:`~saealib.acquisition.base.AcquisitionFunction`.
        """
        ...

    def on_generation_end(
        self,
        gen: int,
        archive: Archive,
        ctx: OptimizationState | None = None,
    ) -> None:
        """End-of-generation hook; override to update internal state."""

    def with_on_generation_end(
        self,
        fn: Callable[[int, Archive, OptimizationState | None], None],
    ) -> SurrogateManager:
        """Return a copy of this manager with ``fn`` appended to the hook.

        Parameters
        ----------
        fn : callable
            ``fn(gen, archive, ctx) -> None``

        Returns
        -------
        SurrogateManager
            Shallow copy with the hook registered.
        """
        new = copy.copy(self)
        prev = self.on_generation_end

        def _chained(
            gen: int,
            archive: Archive,
            ctx: OptimizationState | None = None,
        ) -> None:
            prev(gen, archive, ctx)
            fn(gen, archive, ctx)

        new.on_generation_end = _chained  # type: ignore  # chained callable; hook slot type narrower than Callable
        return new


class GlobalSurrogateManager(SurrogateManager):
    """
    Surrogate manager that fits once on the full archive.

    Fits the surrogate on all archived solutions, then predicts and scores
    all candidates in a single batch. Suitable when global approximation
    quality is sufficient.

    Parameters
    ----------
    surrogate : Surrogate
        Surrogate model instance.
    training_set : TrainingSet or None
        Strategy object for building training data. Defaults to
        ``ArchiveObjectiveSet()``.
    accuracy_evaluator : AccuracyEvaluator or None
        If provided, :meth:`fit` computes accuracy metrics after each fit
        and stores them in :attr:`last_accuracy`.
    """

    def __init__(
        self,
        surrogate: Surrogate,
        training_set: TrainingSet | None = None,
        accuracy_evaluator: AccuracyEvaluator | None = None,
    ):
        self.surrogate = surrogate
        self.training_set: TrainingSet = (
            training_set if training_set is not None else ArchiveObjectiveSet()
        )
        self.accuracy_evaluator = accuracy_evaluator

    def contract(self) -> ComponentContract:
        """Return the global-manager contract."""
        parts = [
            PartSpec(name="surrogate", contract=self.surrogate.contract()),
            PartSpec(name="training_set", contract=self.training_set.contract()),
        ]
        return replace(super().contract(), parts=tuple(parts))

    def fit(
        self,
        archive: Archive,
        ctx: OptimizationState | None = None,
    ) -> None:
        """Fit the surrogate on the full archive.

        If an ``accuracy_evaluator`` was supplied at construction, accuracy
        metrics are computed immediately after fitting and stored in
        :attr:`last_accuracy`.
        """
        population = ctx.population if ctx is not None else None
        data = self.training_set.build(archive, population, ctx, candidate_x=None)
        self.surrogate.fit(data.train_x, data.train_y)
        self.surrogate.post_fit(data.train_x, data.train_y, ctx)
        if self.accuracy_evaluator is not None:
            self.last_accuracy = self.accuracy_evaluator.evaluate(
                self.surrogate, data.train_x, data.train_y
            )

    def predict(
        self,
        candidates_x: np.ndarray,
        archive: Archive,
        ctx: OptimizationState | None = None,
        *,
        refit: bool = True,
    ) -> SurrogatePrediction:
        """Fit on full archive, then predict all candidates at once."""
        if refit:
            self.fit(archive, ctx)
        return self.surrogate.predict(candidates_x)  # mean: (n_candidates, n_obj)


@register()
class LocalSurrogateManager(SurrogateManager):
    """
    Surrogate manager that fits a local model per candidate (pre-selection).

    For each candidate, retrieves the k nearest neighbors from the archive,
    fits the surrogate on that local neighborhood, and predicts the
    candidate's objective value. This corresponds to the individual-based
    local modeling strategy common in SAEA literature.

    NOTE: The same surrogate instance is reused across candidates (re-fit
    each iteration). This is not thread-safe. For parallel use, provide a
    surrogate factory instead (future work).

    Parameters
    ----------
    surrogate : Surrogate
        Surrogate model instance (re-fit per candidate).
    training_set : TrainingSet or None
        Strategy object for building training data per candidate. Defaults to
        ``KNNObjectiveSet(n_neighbors=50)``.
    accuracy_evaluator : AccuracyEvaluator or None
        If provided, :meth:`predict` computes accuracy metrics when
        ``refit=True`` and stores them in :attr:`last_accuracy`.  Accuracy is
        estimated by fitting a local model for each archive point via the same
        ``training_set`` and comparing the prediction against the true value.
    """

    def __init__(
        self,
        surrogate: Surrogate,
        training_set: TrainingSet | None = None,
        accuracy_evaluator: AccuracyEvaluator | None = None,
    ):
        self.surrogate = surrogate
        self.training_set: TrainingSet = (
            training_set
            if training_set is not None
            else KNNObjectiveSet(n_neighbors=50)
        )
        self.accuracy_evaluator = accuracy_evaluator

    def contract(self) -> ComponentContract:
        """Return the local-manager contract."""
        parts = [
            PartSpec(name="surrogate", contract=self.surrogate.contract()),
            PartSpec(name="training_set", contract=self.training_set.contract()),
        ]
        return replace(super().contract(), parts=tuple(parts))

    def fit(
        self,
        archive: Archive,
        ctx: OptimizationState | None = None,
    ) -> None:
        """Compute accuracy metrics from the archive (no global model is fitted).

        For ``LocalSurrogateManager``, there is no persistent global surrogate
        to pre-fit; local models are always built per candidate inside
        :meth:`predict`.  However, calling :meth:`fit` explicitly
        (as ``GenerationBasedStrategy`` does before its inner loop) still
        triggers accuracy evaluation so that :attr:`last_accuracy` is available
        regardless of which strategy is used.
        """
        if self.accuracy_evaluator is not None:
            population = ctx.population if ctx is not None else None
            self._update_accuracy(archive, population, ctx)

    def predict(
        self,
        candidates_x: np.ndarray,
        archive: Archive,
        ctx: OptimizationState | None = None,
        *,
        refit: bool = True,
    ) -> SurrogatePrediction:
        """Fit a local model per candidate and predict each individually.

        When ``refit=True`` and an ``accuracy_evaluator`` is configured,
        accuracy is estimated inline using a nearest-neighbor holdout method:
        for each candidate, the closest archive neighbor is reserved as a
        validation point and excluded from training.  Metrics are averaged
        over all candidates and stored in :attr:`last_accuracy`.

        Because the holdout point is excluded from training, the effective
        training-set size is ``n_neighbors - 1`` when an accuracy evaluator is
        active.  Without an accuracy evaluator the full ``n_neighbors`` points
        are always used.

        A different local model is fit per candidate, but the per-candidate
        single-row predictions are stacked (see :func:`_stack_predictions`)
        into one batched ``SurrogatePrediction`` covering every row of
        ``candidates_x``, matching the base class contract.
        """
        predictions: list[SurrogatePrediction] = []
        population = ctx.population if ctx is not None else None
        compute_accuracy = refit and self.accuracy_evaluator is not None
        y_true_list: list[np.ndarray] = []
        y_pred_list: list[np.ndarray] = []

        for x in candidates_x:
            data = self.training_set.build(archive, population, ctx, candidate_x=x)

            if compute_accuracy and len(data.train_x) >= 2:
                # Hold out the nearest neighbor (index 0 when KNN-sorted) as
                # validation to get an unbiased estimate of local model accuracy.
                val_x = data.train_x[0:1]
                val_y = np.atleast_2d(data.train_y[0:1].T).T
                train_x = data.train_x[1:]
                train_y = data.train_y[1:]
            else:
                val_x = val_y = None
                train_x = data.train_x
                train_y = data.train_y

            self.surrogate.fit(train_x, train_y)
            self.surrogate.post_fit(train_x, train_y, ctx)
            pred = self.surrogate.predict(x)  # mean: (1, n_obj)
            predictions.append(pred)

            if val_x is not None and val_y is not None:
                self._record_accuracy_prediction(
                    self.surrogate, val_x, val_y, y_true_list, y_pred_list
                )

        if compute_accuracy:
            if y_true_list:
                y_true = np.stack(y_true_list)
                y_pred = np.stack(y_pred_list)
                metrics = self.accuracy_evaluator._compute_metrics(y_true, y_pred)
                self.last_accuracy = SurrogateAccuracy(
                    metrics=metrics, n_samples=len(y_true_list)
                )
            else:
                self.last_accuracy = SurrogateAccuracy(n_samples=0)

        return _stack_predictions(predictions)

    def _record_accuracy_prediction(
        self,
        surrogate: Surrogate,
        query_x: np.ndarray,
        true_y: np.ndarray,
        y_true_list: list[np.ndarray],
        y_pred_list: list[np.ndarray],
    ) -> None:
        try:
            prediction = surrogate.predict(query_x)
            y_true_list.append(true_y[0])
            y_pred_list.append(prediction.value[0])
        except Exception:
            pass

    def _update_accuracy(
        self,
        archive: Archive,
        population: object,
        ctx: OptimizationState | None,
    ) -> None:
        """Compute accuracy via LOO with self-exclusion on archive points.

        Called by :meth:`fit` for the ``GenerationBasedStrategy`` pattern.
        For each archive point ``x_i``, the local model is built from the
        ``n_neighbors`` nearest neighbors of ``x_i`` **excluding ``x_i``
        itself**, and then predicts ``x_i``.  This avoids the self-inclusion
        bias that occurs when an interpolating surrogate (e.g. RBF) is fitted
        on data that includes the query point.
        """
        archive_x = archive.x
        archive_y = np.atleast_2d(archive.f.T).T
        n = len(archive_x)

        if n < 2:
            self.last_accuracy = SurrogateAccuracy(n_samples=n)
            return

        n_neighbors = min(getattr(self.training_set, "n_neighbors", n), n - 1)
        y_true_list: list[np.ndarray] = []
        y_pred_list: list[np.ndarray] = []

        for i in range(n):
            # Exclude x_i from its own neighborhood (LOO self-exclusion)
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            loo_x = archive_x[mask]
            loo_y = archive_y[mask]

            dists = np.sum((loo_x - archive_x[i]) ** 2, axis=1)
            k = min(n_neighbors, len(loo_x))
            idx = np.argsort(dists)[:k]
            train_x = loo_x[idx]
            train_y = loo_y[idx]

            surrogate_copy = copy.deepcopy(self.surrogate)
            try:
                surrogate_copy.fit(train_x, train_y)
            except Exception:
                continue
            self._record_accuracy_prediction(
                surrogate_copy,
                archive_x[i : i + 1],
                archive_y[i : i + 1],
                y_true_list,
                y_pred_list,
            )

        if not y_true_list:
            self.last_accuracy = SurrogateAccuracy(n_samples=n)
            return

        y_true = np.stack(y_true_list)
        y_pred = np.stack(y_pred_list)
        metrics = self.accuracy_evaluator._compute_metrics(y_true, y_pred)  # type: ignore  # caller guarantees non-None; ty doesn't narrow across method boundary
        self.last_accuracy = SurrogateAccuracy(metrics=metrics, n_samples=n)


def product_combine(scores: list[np.ndarray]) -> np.ndarray:
    """Combine scores by element-wise product.

    Parameters
    ----------
    scores : list[np.ndarray]
        Score arrays, each shape ``(n_candidates,)``.

    Returns
    -------
    np.ndarray
        Element-wise product. shape: ``(n_candidates,)``.
    """
    return np.prod(np.stack(scores, axis=0), axis=0)


def rank_weighted_combine(
    weights: np.ndarray | None = None,
) -> Callable[[list[np.ndarray]], np.ndarray]:
    """Return a combine function that rank-normalises then takes a weighted average.

    Parameters
    ----------
    weights : np.ndarray or None
        Weights for each manager. If None, uniform weights are used.
        Need not sum to 1; they are normalised internally.

    Returns
    -------
    callable
        A function ``(list[np.ndarray]) -> np.ndarray`` suitable for
        ``CompositeAcquisition``.
    """
    if weights is not None:
        w = np.asarray(weights, dtype=float)
        w = w / w.sum()
    else:
        w = None  # resolved lazily to uniform at call time

    def _combine(score_list: list[np.ndarray]) -> np.ndarray:
        normalized = [_rank_normalize(s) for s in score_list]
        _w = w if w is not None else np.full(len(normalized), 1.0 / len(normalized))
        return (np.stack(normalized, axis=0) * _w[:, None]).sum(axis=0)

    return _combine


class CompositeSurrogateManager(SurrogateManager):
    """Surrogate manager that composes predictions from multiple named sub-managers.

    This manager owns named child managers, calls each child's :meth:`predict`, and
    returns a single :class:`~saealib.surrogate.prediction.SurrogatePrediction`
    whose ``channels`` mapping contains one named
    :class:`~saealib.surrogate.prediction.PredictionChannel` per child. It
    never combines acquisition scores itself -- pair it with
    :class:`~saealib.acquisition.base.CompositeAcquisition`, which evaluates
    each configured channel's acquisition and combines the resulting score
    arrays::

        ei_manager = GlobalSurrogateManager(GP())
        pof_manager = GlobalSurrogateManager(
            PerObjectiveSurrogate([GP()] * n_constraints)
        )
        composite_acq = CompositeAcquisition(
            {
                "objective": ExpectedImprovement(),
                "feasibility": ProductOfFeasibility(),
            },
            combine_fn=product_combine,
        )
        surrogate_manager = CompositeSurrogateManager(
            {"objective": ei_manager, "feasibility": pof_manager},
        )

    Parameters
    ----------
    managers : dict[str, SurrogateManager]
        Named sub-managers to compose. Must be non-empty. Each key becomes
        the corresponding ``PredictionChannel``'s name in :meth:`predict`'s
        result.
    """

    def __init__(
        self,
        managers: dict[str, SurrogateManager],
    ):
        if not managers:
            raise ValidationError(
                "CompositeSurrogateManager requires at least one manager."
            )
        self.managers = managers

    def fit(
        self,
        archive: Archive,
        ctx: OptimizationState | None = None,
    ) -> None:
        """Pre-fit all sub-managers.

        :attr:`last_accuracy` is propagated from the first sub-manager (in
        configuration insertion order) so callers can read a representative
        accuracy value from this composite manager.
        """
        for manager in self.managers.values():
            manager.fit(archive, ctx)
        self.last_accuracy = next(iter(self.managers.values())).last_accuracy

    def predict(
        self,
        candidates_x: np.ndarray,
        archive: Archive,
        ctx: OptimizationState | None = None,
        *,
        refit: bool = True,
    ) -> SurrogatePrediction:
        """Predict via every sub-manager, one named channel each.

        Returns
        -------
        SurrogatePrediction
            ``channels`` maps each configuration key to a
            ``PredictionChannel`` built from that sub-manager's own
            ``value``/``std`` (i.e. its own primary/objective channel). If a
            sub-manager's ``predict()`` exposes no ``"objective"`` channel,
            accessing ``.value``/``.std`` raises ``KeyError`` -- a genuine
            misconfiguration, not handled defensively here.
        """
        channels: dict[str, PredictionChannel] = {}
        for name, manager in self.managers.items():
            child_pred = manager.predict(candidates_x, archive, ctx, refit=refit)
            channels[name] = PredictionChannel(
                value=child_pred.value, std=child_pred.std
            )
        return SurrogatePrediction(channels=channels)


class PairwiseSurrogateManager(SurrogateManager):
    """Surrogate manager for pairwise comparison surrogates.

    :meth:`predict` pairs each candidate with reference points sampled once
    per call from the archive and averages the predicted win probability
    over all pairs, returning the result as a ``"win_rate"`` prediction
    channel. Pair with :class:`~saealib.acquisition.winrate.WinRateAcquisition`
    This manager performs the
    full reference-sampling + pair-construction + ``predict_proba()`` +
    per-candidate win-rate-aggregation sequence itself, rather than the usual
    "manager predicts, then acquisition scores" split -- the
    ``(candidate, reference)`` pairs are not known until this manager's own
    reference-sampling logic runs, and ``AcquisitionFunction`` deliberately
    has no ``Surrogate``/``SurrogateManager`` access to construct and predict
    on them itself.

    The surrogate must be a :class:`~saealib.surrogate.base.ComparisonSurrogate`
    that implements ``predict_proba()``.

    Parameters
    ----------
    surrogate : ComparisonSurrogate
        Pairwise comparison surrogate (e.g. ``SklearnRFCClassificationSurrogate``).
    training_set : TrainingSet or None
        Training data builder.  Defaults to ``PairwiseComparisonSet()``.
        ``ctx`` is required when using ``PairwiseComparisonSet``.
    n_ref : int
        Number of archive points sampled, once per :meth:`predict` call, as
        the shared reference set every candidate in that call is compared
        against. When the archive has fewer than ``n_ref`` points all are
        used.
    """

    def __init__(
        self,
        surrogate: ComparisonSurrogate,
        training_set: TrainingSet | None = None,
        n_ref: int = 10,
    ):
        self.surrogate = surrogate
        self.training_set: TrainingSet = (
            training_set if training_set is not None else PairwiseComparisonSet()
        )
        self.n_ref = n_ref

    def contract(self) -> ComponentContract:
        """Return the pairwise-manager contract."""
        return replace(
            super().contract(),
            parts=(
                PartSpec(name="surrogate", contract=self.surrogate.contract()),
                PartSpec(name="training_set", contract=self.training_set.contract()),
            ),
            state=StateContract(
                reads=(POPULATIONS_MAIN, RUNTIME_RNG),
                writes=(RUNTIME_RNG,),
            ),
        )

    def fit(
        self,
        archive: Archive,
        ctx: OptimizationState | None = None,
    ) -> None:
        """Fit the surrogate on pairwise comparison training data.

        Parameters
        ----------
        archive : Archive
            Archive of evaluated solutions.
        ctx : OptimizationState or None
            Required when the training set is ``PairwiseComparisonSet``
            (provides the comparator and rng).
        """
        population = ctx.population if ctx is not None else None
        data = self.training_set.build(archive, population, ctx, candidate_x=None)
        self.surrogate.fit(data.train_x, data.train_y)
        self.surrogate.post_fit(data.train_x, data.train_y, ctx)

    def predict(
        self,
        candidates_x: np.ndarray,
        archive: Archive,
        ctx: OptimizationState | None = None,
        *,
        refit: bool = True,
    ) -> SurrogatePrediction:
        """Predict per-candidate mean win rate against archive reference points.

        Draws a reference subset from the archive exactly once per call
        (not once per candidate) -- this RNG-consumption timing is a
        once per candidate batch.

        Parameters
        ----------
        candidates_x : np.ndarray
            Candidate design variable matrix. shape: (n_candidates, dim)
        archive : Archive
            Archive of evaluated solutions.
        ctx : OptimizationState or None
            Optimization context.  Required when ``refit=True`` and the
            training set is ``PairwiseComparisonSet``, and to supply
            ``ctx.rng`` for reference sampling (falls back to a fresh
            unseeded generator when ``ctx`` is ``None``).
        refit : bool
            If ``True`` (default), fit the surrogate before predicting.

        Returns
        -------
        SurrogatePrediction
            A single ``"win_rate"`` channel, deliberately not named
            ``"objective"``: naming it ``"objective"`` would make
            predicted objective. ``value`` shape: ``(n_candidates, 1)``.
        """
        if refit:
            self.fit(archive, ctx)

        rng = ctx.rng if ctx is not None else np.random.default_rng()

        archive_x = archive.x
        n_archive = len(archive_x)
        n_ref = min(self.n_ref, n_archive)
        if n_ref < n_archive:
            ref_idx = rng.choice(n_archive, size=n_ref, replace=False)
            ref_x = archive_x[ref_idx]
        else:
            ref_x = archive_x

        win_rates = np.empty(len(candidates_x))
        for i, x_c in enumerate(candidates_x):
            pairs = np.stack([np.concatenate([x_c, x_r]) for x_r in ref_x])
            pred = self.surrogate.predict_proba(pairs)
            win_rates[i] = float(np.mean(pred.value[:, 0]))

        return SurrogatePrediction(
            channels={"win_rate": PredictionChannel(value=win_rates[:, None])}
        )


def _split_prediction(prediction: SurrogatePrediction) -> list[SurrogatePrediction]:
    """Split a batch SurrogatePrediction into per-sample SurrogatePrediction objects.

    Inverse of :func:`_stack_predictions`. Every named channel in
    ``prediction.channels`` is sliced per row; ``covariance``/``samples`` are
    implementation-specific joint quantities and are carried over unsliced
    (reused, not copied) on each per-sample channel, matching how ``x``/
    ``label``/``metadata`` are handled below.
    """
    if prediction.channels:
        n = next(iter(prediction.channels.values())).value.shape[0]
    elif prediction.x is not None:
        n = prediction.x.shape[0]
    else:
        n = 0

    result = []
    for i in range(n):
        channels_i = {
            name: PredictionChannel(
                value=channel.value[i : i + 1],
                std=channel.std[i : i + 1] if channel.std is not None else None,
                covariance=channel.covariance,
                samples=channel.samples,
                metadata=channel.metadata,
            )
            for name, channel in prediction.channels.items()
        }
        label_i = prediction.label[i : i + 1] if prediction.label is not None else None
        x_i = prediction.x[i : i + 1] if prediction.x is not None else None
        result.append(
            SurrogatePrediction(
                channels=channels_i,
                x=x_i,
                label=label_i,
                metadata=prediction.metadata,
            )
        )
    return result


def _stack_predictions(predictions: list[SurrogatePrediction]) -> SurrogatePrediction:
    """Stack per-sample SurrogatePrediction objects into one batched prediction.

    Inverse of :func:`_split_prediction`. Every prediction in *predictions*
    must share the same channel names (e.g. ``LocalSurrogateManager`` always
    predicts through the same ``self.surrogate.predict()``, so this always
    holds in practice). ``covariance``/``samples`` are not stacked (they are
    implementation-specific joint quantities that do not concatenate
    meaningfully row-by-row); only ``value``/``std`` are stacked per channel.
    ``label`` is stacked the same way ``std`` is (all-or-nothing).

    Parameters
    ----------
    predictions : list[SurrogatePrediction]
        Per-sample predictions, each with ``value``/``std`` shaped
        ``(1, n_output)`` per channel.

    Returns
    -------
    SurrogatePrediction
        One batched prediction with ``value``/``std`` shaped
        ``(n, n_output)`` per channel. Returns an empty-channels prediction
        for an empty input list.

    Raises
    ------
    ValidationError
        If a channel's ``std`` is present (non-``None``) on some but not all
        predictions -- a partial ``std`` cannot be stacked without silently
        fabricating missing values. The same rule applies to ``label``.
    """
    if not predictions:
        return SurrogatePrediction(channels={})

    channel_names = predictions[0].channels.keys()
    channels: dict[str, PredictionChannel] = {}
    for name in channel_names:
        values = [p.channels[name].value for p in predictions]
        stds = [p.channels[name].std for p in predictions]
        if all(s is None for s in stds):
            std = None
        elif all(s is not None for s in stds):
            std = np.concatenate(stds, axis=0)  # type: ignore  # narrowed by all() check above
        else:
            raise ValidationError(
                f"Cannot stack channel {name!r}: std is present on some "
                "predictions but not others."
            )
        channels[name] = PredictionChannel(
            value=np.concatenate(values, axis=0),
            std=std,
        )

    x = None
    if all(p.x is not None for p in predictions):
        x = np.concatenate([p.x for p in predictions], axis=0)

    labels = [p.label for p in predictions]
    if all(label is None for label in labels):
        label = None
    elif all(label is not None for label in labels):
        label = np.concatenate(labels, axis=0)  # type: ignore  # narrowed by all() check above
    else:
        raise ValidationError(
            "Cannot stack predictions: label is present on some but not others."
        )

    return SurrogatePrediction(channels=channels, x=x, label=label)


def _rank_normalize(scores: np.ndarray) -> np.ndarray:
    """
    Normalize scores to [0, 1] via rank transform.

    Rank 0 (lowest score) -> 0.0, rank n-1 (highest score) -> 1.0.
    NaN scores are treated as the lowest rank (0.0) so that candidates
    with failed surrogate predictions are never selected.
    """
    n = len(scores)
    if n == 1:
        return np.ones(1)
    safe = np.where(np.isnan(scores), -np.inf, scores)
    order = np.argsort(safe)
    ranks = np.empty(n)
    ranks[order] = np.arange(n)
    return ranks / (n - 1)
