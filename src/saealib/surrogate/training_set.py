"""
Training set strategy objects for surrogate model fitting.

A ``TrainingSet`` abstracts the construction of ``(train_x, train_y)`` pairs
that are passed to ``Surrogate.fit``.  The two orthogonal axes are:

* **Data source** (where samples come from):
  archive, population, k-NN neighbourhood, pairs, per-individual reference points.
* **Labelling rule** (how target values are assigned):
  raw objectives (regression), binary classification, multi-level ranking,
  pairwise comparison.

Literature patterns covered by this module:

+----+----------------------------------------+----------------------------------+
| P1 | CA-LLSO (Wei et al., 2021)             | ``LevelBasedSet``                |
+----+----------------------------------------+----------------------------------+
| P2 | CPS-MOEA (Zhang et al., 2018)          | ``TopKBipartitionSet``           |
+----+----------------------------------------+----------------------------------+
| P3 | Relation-model SAEA (Hao et al., 2024) | ``PairwiseComparisonSet``        |
+----+----------------------------------------+----------------------------------+
| P4 | SAPSO pbest (Tian et al., 2019)        | ``ReferencePointComparisonSet``  |
+----+----------------------------------------+----------------------------------+
| P5 | CSEA / pre-selection (general)         | ``KNNObjectiveSet``,             |
|    |                                        | ``ArchiveObjectiveSet``          |
+----+----------------------------------------+----------------------------------+
| P6 | Constraint BO (Regis & Shoemaker 2005) | ``ConstraintObjectiveSet``,      |
|    | Letham et al. (2019)                   | ``KNNConstraintObjectiveSet``    |
+----+----------------------------------------+----------------------------------+
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import numpy as np

from saealib.core.contracts import (
    MANY,
    ComponentContract,
    DataSpec,
    PortContract,
    PortDirection,
    PortSpec,
    StateContract,
    Var,
)
from saealib.core.state import RUNTIME_RNG
from saealib.exceptions import ValidationError
from saealib.population.archive import Archive
from saealib.population.genome import DenseVectorBatch
from saealib.registry import register
from saealib.space.space import encode_features

if TYPE_CHECKING:
    from saealib.context import OptimizationState
    from saealib.population import Archive, Population


@dataclass
class TrainingData:
    """Immutable container for the training data passed to ``Surrogate.fit``.

    Attributes
    ----------
    train_x : np.ndarray
        Design variable matrix. shape: ``(n_train, dim)``.
        For ``PairwiseComparisonSet`` the shape is ``(n_train, 2 * dim)``.
    train_y : np.ndarray
        Target values. shape: ``(n_train, n_obj)`` for regression,
        ``(n_train,)`` for classification or ranking.
    """

    train_x: np.ndarray
    train_y: np.ndarray


def _canonical_dense_archive_data(
    archive: Archive,
) -> dict[str, np.ndarray] | None:
    """Return canonical dense archive storage, or ``None`` for custom paths.

    This is deliberately limited to the exact built-in ``Archive`` and its
    built-in dense genome storage.  Custom archives and subclasses retain the
    public-property path, since their properties or storage may have different
    semantics.
    """
    if type(archive) is not Archive:
        return None
    values = archive.__dict__
    if not isinstance(values.get("_genome_batch"), DenseVectorBatch):
        return None
    data = values.get("_data")
    if not isinstance(data, dict):
        return None
    return data


def _dense_archive_rows(
    archive: Archive, indices: np.ndarray, target: str
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return dense genome and column rows without going through properties."""
    data = _canonical_dense_archive_data(archive)
    if data is None or "x" not in data or target not in data:
        return None

    size = len(archive)
    x = data["x"][:size][indices]
    y = data[target][:size][indices]
    x.setflags(write=False)
    y.setflags(write=False)
    return x, y


class TrainingSet(ABC):
    """Strategy object that builds ``(train_x, train_y)`` for a surrogate.

    Inject an instance into ``LocalSurrogateManager`` or
    ``GlobalSurrogateManager`` via their ``training_set`` constructor argument.
    """

    def contract(self) -> ComponentContract:
        """Return the training-set contract."""
        feature_schema = Var(name="F")
        objective_schema = Var(name="O")
        return ComponentContract(
            ports={
                "training_set": PortContract(
                    inputs=(
                        PortSpec(
                            name="archive",
                            direction=PortDirection.INPUT,
                            data=DataSpec(kind="Population"),
                            cardinality=MANY,
                        ),
                        PortSpec(
                            name="population",
                            direction=PortDirection.INPUT,
                            data=DataSpec(kind="Population"),
                            cardinality=MANY,
                            optional=True,
                        ),
                    ),
                    outputs=(
                        PortSpec(
                            name="features",
                            direction=PortDirection.OUTPUT,
                            data=DataSpec(
                                kind="FeatureBatch",
                                bindings={"feature_schema": feature_schema},
                            ),
                            cardinality=MANY,
                        ),
                        PortSpec(
                            name="targets",
                            direction=PortDirection.OUTPUT,
                            data=DataSpec(
                                kind="FeatureBatch",
                                bindings={"objective_schema": objective_schema},
                            ),
                            cardinality=MANY,
                        ),
                    ),
                )
            }
        )

    @abstractmethod
    def build(
        self,
        archive: Archive,
        population: Population | None,
        ctx: OptimizationState | None,
        candidate_x: np.ndarray | None = None,
    ) -> TrainingData:
        """Build and return the training data.

        Parameters
        ----------
        archive:
            Evaluated solutions archive.
        population:
            Current population. Not used by all implementations.
        ctx:
            Optimization context. Provides ``comparator``, ``problem.eps_cv``,
            etc. Required by ranking- and comparison-based implementations.
        candidate_x:
            Centre point for k-NN queries (shape ``(dim,)``). Passed by
            ``LocalSurrogateManager`` for each candidate; ``None`` when called
            from ``GlobalSurrogateManager``.

        Returns
        -------
        TrainingData
        """
        ...


class ArchiveObjectiveSet(TrainingSet):
    """Use the entire archive as training data with raw objective values."""

    def build(
        self,
        archive: Archive,
        population: Population | None,
        ctx: OptimizationState | None,
        candidate_x: np.ndarray | None = None,
    ) -> TrainingData:
        """Return all archive points with raw objective values."""
        genomes = getattr(archive, "genomes", None)
        if ctx is not None and genomes is not None:
            train_x = encode_features(ctx.problem.space, genomes)
        else:
            train_x = archive.x
        return TrainingData(train_x=train_x, train_y=archive.f)


@register()
class KNNObjectiveSet(TrainingSet):
    """Retrieve the *k* nearest archive neighbours of ``candidate_x``.

    This is the default training set for ``LocalSurrogateManager``.

    Parameters
    ----------
    n_neighbors:
        Number of nearest neighbours to retrieve.
    """

    def __init__(self, n_neighbors: int = 50) -> None:
        self.n_neighbors = n_neighbors

    def build(
        self,
        archive: Archive,
        population: Population | None,
        ctx: OptimizationState | None,
        candidate_x: np.ndarray | None = None,
    ) -> TrainingData:
        """Return k nearest-neighbour archive points around ``candidate_x``."""
        if candidate_x is None:
            raise ValidationError("KNNObjectiveSet requires candidate_x.")
        idx, _ = archive.get_knn(candidate_x, self.n_neighbors)
        rows = _dense_archive_rows(archive, idx, "f")
        if rows is not None:
            return TrainingData(train_x=rows[0], train_y=rows[1])
        return TrainingData(train_x=archive.x[idx], train_y=archive.f[idx])


class ConstraintObjectiveSet(TrainingSet):
    """Use the entire archive as training data with raw constraint values.

    Analogue of :class:`ArchiveObjectiveSet` for constraint surrogates.
    Returns ``(archive.x, archive.g)`` so that a surrogate can learn to
    predict the raw constraint values ``g(x)``.

    Raises
    ------
    ValueError
        If the archive has no constraint columns (``archive.g.shape[1] == 0``).
        Use this class only with problems that define at least one constraint.
    """

    def build(
        self,
        archive: Archive,
        population: Population | None,
        ctx: OptimizationState | None,
        candidate_x: np.ndarray | None = None,
    ) -> TrainingData:
        """Return all archive points with raw constraint values."""
        g = archive.g
        if g.shape[1] == 0:
            raise ValidationError(
                "ConstraintObjectiveSet requires at least one constraint "
                "(archive.g has 0 columns)."
            )
        return TrainingData(train_x=archive.x, train_y=g)


class KNNConstraintObjectiveSet(TrainingSet):
    """Retrieve the *k* nearest archive neighbours of ``candidate_x``.

    Analogue of :class:`KNNObjectiveSet` for constraint surrogates.
    Returns the k-NN subset of ``(archive.x, archive.g)`` centred on
    ``candidate_x``.

    Parameters
    ----------
    n_neighbors:
        Number of nearest neighbours to retrieve.

    Raises
    ------
    ValueError
        If ``candidate_x`` is None, or if the archive has no constraint columns.
    """

    def __init__(self, n_neighbors: int = 50) -> None:
        self.n_neighbors = n_neighbors

    def build(
        self,
        archive: Archive,
        population: Population | None,
        ctx: OptimizationState | None,
        candidate_x: np.ndarray | None = None,
    ) -> TrainingData:
        """Return k nearest-neighbour archive points with raw constraint values."""
        if candidate_x is None:
            raise ValidationError("KNNConstraintObjectiveSet requires candidate_x.")
        dense_data = _canonical_dense_archive_data(archive)
        if dense_data is not None and "g" in dense_data:
            g = dense_data["g"][: len(archive)]
        else:
            g = archive.g
        if g.shape[1] == 0:
            raise ValidationError(
                "KNNConstraintObjectiveSet requires at least one constraint "
                "(archive.g has 0 columns)."
            )
        idx, _ = archive.get_knn(candidate_x, self.n_neighbors)
        rows = _dense_archive_rows(archive, idx, "g")
        if rows is not None:
            return TrainingData(train_x=rows[0], train_y=rows[1])
        return TrainingData(train_x=archive.x[idx], train_y=g[idx])


class FeasibilityClassificationSet(TrainingSet):
    """Binary labels derived from constraint violation: 1 if feasible, 0 otherwise.

    A sample is feasible when ``cv <= eps_cv``, where ``eps_cv`` is taken from
    ``ctx.problem.eps_cv`` (defaults to ``1e-6`` when ``ctx`` is ``None``).

    This set is algo-agnostic and works with any surrogate that accepts binary
    classification targets (shape ``(n_train,)``).

    Parameters
    ----------
    source:
        ``"archive"`` to use the evaluated archive, or ``"population"`` to
        use the current population.

    Raises
    ------
    ValueError
        If ``source="population"`` but ``population`` is ``None``.
    """

    def __init__(self, source: str = "archive") -> None:
        self.source = source

    def build(
        self,
        archive: Archive,
        population: Population | None,
        ctx: OptimizationState | None,
        candidate_x: np.ndarray | None = None,
    ) -> TrainingData:
        """Return feasibility binary labels (1 = feasible, 0 = infeasible)."""
        if self.source == "population":
            if population is None:
                raise ValidationError(
                    "FeasibilityClassificationSet: source='population' requires "
                    "population to be provided."
                )
            src = population
        else:
            src = archive
        eps = ctx.problem.eps_cv if ctx is not None else 1e-6
        labels = (src.get_array("cv") <= eps).astype(float)
        return TrainingData(train_x=src.get_array("x"), train_y=labels)


class TopKBipartitionSet(TrainingSet):
    """Binary labels from a sorted bipartition: 1 for top-*k*, 0 for the rest.

    Individuals are sorted by ``ctx.comparator.sort_population`` (best-first),
    then the top ``floor(n * top_ratio)`` receive label 1 (promising) and the
    remainder receive label 0 (unpromising).

    Literature patterns: CPS-MOEA (P2), CSEA / pre-selection (P5).

    Parameters
    ----------
    source:
        ``"archive"`` or ``"population"``.
    top_ratio:
        Fraction of individuals labelled as promising. Clamped to ``[0, 1]``.
        At least one individual always receives label 1.

    Raises
    ------
    ValueError
        If ``ctx`` is ``None`` or ``source="population"`` with ``population=None``.
    """

    def __init__(self, source: str = "archive", top_ratio: float = 0.5) -> None:
        self.source = source
        self.top_ratio = top_ratio

    def build(
        self,
        archive: Archive,
        population: Population | None,
        ctx: OptimizationState | None,
        candidate_x: np.ndarray | None = None,
    ) -> TrainingData:
        """Return bipartition labels based on sorted rank."""
        if ctx is None:
            raise ValidationError("TopKBipartitionSet requires ctx.")
        if self.source == "population":
            if population is None:
                raise ValidationError(
                    "TopKBipartitionSet: source='population' requires population."
                )
            src = population
        else:
            src = archive
        sorted_idx = ctx.comparator.sort_population(src)
        x = src.get_array("x")[sorted_idx]
        n = len(x)
        k = max(1, int(n * self.top_ratio))
        labels = np.concatenate([np.ones(k), np.zeros(n - k)])
        return TrainingData(train_x=x, train_y=labels)


class LevelBasedSet(TrainingSet):
    """Multi-class labels from equal-division of a sorted population (CA-LLSO, P1).

    Individuals are sorted by ``ctx.comparator.sort_population`` (best-first),
    then divided into ``n_levels`` equally-sized groups.  The best group receives
    label ``n_levels - 1`` and the worst group receives label 0.  Remainder
    individuals are appended to the worst (label 0) group.

    Parameters
    ----------
    source:
        ``"archive"`` or ``"population"``.
    n_levels:
        Number of classification levels (≥ 2).

    Raises
    ------
    ValueError
        If ``ctx`` is ``None`` or ``source="population"`` with ``population=None``.
    """

    def __init__(self, source: str = "population", n_levels: int = 5) -> None:
        self.source = source
        self.n_levels = n_levels

    def build(
        self,
        archive: Archive,
        population: Population | None,
        ctx: OptimizationState | None,
        candidate_x: np.ndarray | None = None,
    ) -> TrainingData:
        """Return level labels from equal-size sorted groups."""
        if ctx is None:
            raise ValidationError("LevelBasedSet requires ctx.")
        if self.source == "population":
            if population is None:
                raise ValidationError(
                    "LevelBasedSet: source='population' requires population."
                )
            src = population
        else:
            src = archive
        sorted_idx = ctx.comparator.sort_population(src)
        x = src.get_array("x")[sorted_idx]
        n = len(x)
        per = n // self.n_levels
        labels = np.repeat(np.arange(self.n_levels - 1, -1, -1), per)
        # Remainder goes to the worst (label 0) group
        remainder = n - len(labels)
        if remainder > 0:
            labels = np.concatenate([labels, np.full(remainder, 0)])
        return TrainingData(train_x=x, train_y=labels.astype(float))


class PairwiseComparisonSet(TrainingSet):
    """Pairwise labels: ``train_x`` is ``[x_a, x_b]``, label is 1 if *a* beats *b*.

    For each pair ``(a, b)``, the feature vector is the concatenation
    ``[x_a, x_b]`` (shape ``(n_pairs, 2 * dim)``), and the label is 1 if
    ``comparator.compare(f_a, cv_a, f_b, cv_b) <= 0`` (a dominates or equals b),
    0 otherwise.

    .. warning::
        ``train_x`` has shape ``(n_pairs, 2 * dim)``, which is incompatible with
        standard regression surrogates (e.g. ``RBFSurrogate``).  Use a
        pairwise-specific surrogate that expects concatenated feature pairs.

    Parameters
    ----------
    source:
        ``"archive"`` or ``"population"``.
    n_pairs:
        Number of pairs to sample.  ``None`` (default) uses all
        ``n * (n-1) / 2`` pairs.  If the requested count exceeds the total
        available pairs, all pairs are used.
    rng:
        Random number generator for pair sampling.  Falls back to
        ``ctx.rng`` when ``ctx`` is provided, otherwise creates a new one.

    Raises
    ------
    ValueError
        If ``ctx`` is ``None`` (comparator required) or
        ``source="population"`` with ``population=None``.
    """

    def __init__(
        self,
        source: str = "archive",
        n_pairs: int | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.source = source
        self.n_pairs = n_pairs
        self.rng = rng

    def contract(self) -> ComponentContract:
        """Return the pairwise-training contract."""
        state = (
            StateContract(reads=(RUNTIME_RNG,), writes=(RUNTIME_RNG,))
            if self.n_pairs is not None and self.rng is None
            else StateContract()
        )
        return replace(super().contract(), state=state)

    def build(
        self,
        archive: Archive,
        population: Population | None,
        ctx: OptimizationState | None,
        candidate_x: np.ndarray | None = None,
    ) -> TrainingData:
        """Return pairwise comparison training data."""
        if ctx is None:
            raise ValidationError("PairwiseComparisonSet requires ctx.")
        if self.source == "population":
            if population is None:
                raise ValidationError(
                    "PairwiseComparisonSet: source='population' requires population."
                )
            src = population
        else:
            src = archive
        x = src.get_array("x")
        f = src.get_array("f")
        cv = src.get_array("cv")
        cmp = ctx.comparator
        rng = self.rng if self.rng is not None else ctx.rng
        n = len(x)
        n_all = n * (n - 1) // 2
        if self.n_pairs is None or self.n_pairs >= n_all:
            pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        else:
            pairs: list[tuple[int, int]] = []
            seen: set[tuple[int, int]] = set()
            while len(pairs) < self.n_pairs:
                idx = rng.choice(n, size=2, replace=False)
                key = (int(min(idx[0], idx[1])), int(max(idx[0], idx[1])))
                if key not in seen:
                    seen.add(key)
                    pairs.append(key)
        train_x = np.stack([np.concatenate([x[i], x[j]]) for i, j in pairs])
        train_y = np.array(
            [
                1.0 if cmp.compare(f[i], cv[i], f[j], cv[j]) <= 0 else 0.0
                for i, j in pairs
            ]
        )
        return TrainingData(train_x=train_x, train_y=train_y)


class ReferencePointComparisonSet(TrainingSet):
    """Binary labels: 1 if archive point dominates the reference point, 0 otherwise.

    Unlike PairwiseComparisonSet, train_x has shape (n_archive, dim),
    making it compatible with GlobalSurrogateManager and LocalSurrogateManager.

    Parameters
    ----------
    ref_source : str
        How to select the reference point:
        - ``"population_best"`` (default): best individual in the current population
        - ``"archive_best"``: best individual in the archive

    Raises
    ------
    ValueError
        If ctx is None (comparator required).
    ValueError
        If ref_source="population_best" and population is None.
    """

    def __init__(self, ref_source: str = "population_best") -> None:
        self.ref_source = ref_source

    def build(
        self,
        archive: Archive,
        population: Population | None,
        ctx: OptimizationState | None,
        candidate_x: np.ndarray | None = None,
    ) -> TrainingData:
        """Return binary labels: 1 if archive point dominates reference, 0 otherwise."""
        if ctx is None:
            raise ValidationError("ReferencePointComparisonSet requires ctx.")
        if self.ref_source == "population_best":
            if population is None:
                raise ValidationError(
                    "ReferencePointComparisonSet: ref_source='population_best' "
                    "requires population."
                )
            ref_src = population
        else:
            ref_src = archive
        sorted_idx = ctx.comparator.sort_population(ref_src)
        ref_idx = sorted_idx[0]
        f_ref = ref_src.get_array("f")[ref_idx]
        cv_ref = ref_src.get_array("cv")[ref_idx]
        f_arc = archive.get_array("f")
        cv_arc = archive.get_array("cv")
        cmp = ctx.comparator
        n = len(archive)
        labels = np.array(
            [
                1.0 if cmp.compare(f_arc[i], cv_arc[i], f_ref, cv_ref) <= 0 else 0.0
                for i in range(n)
            ]
        )
        return TrainingData(train_x=archive.x, train_y=labels)
