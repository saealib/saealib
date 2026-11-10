"""VectorSpace: search space for dense real-valued vectors."""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from typing import TYPE_CHECKING, cast

import numpy as np
from scipy.spatial import cKDTree  # type: ignore

from saealib.core.contracts.data import Fixed
from saealib.core.contracts.representation import ParameterSpec, RepresentationSpec
from saealib.exceptions import ValidationError
from saealib.population.genome import DenseVectorBatch, GenomeBatch
from saealib.space.space import (
    DerivedSamplingService,
    DerivedValidationService,
    ServiceRegistry,
    ValidationResult,
)

if TYPE_CHECKING:
    pass

__all__ = ["VectorSpace"]


_CANONICAL_NAN = np.uint64(0x7FF8000000000000)


def _canonical_rows(array: np.ndarray) -> np.ndarray:
    bits = np.array(array, dtype=np.float64, copy=True).view(np.uint64)
    bits[bits == np.uint64(0x8000000000000000)] = 0
    nan = np.isnan(array)
    bits[nan] = _CANONICAL_NAN
    return bits


class _VectorBoundsService:
    def __init__(self, lb: np.ndarray, ub: np.ndarray) -> None:
        self._lb = lb
        self._ub = ub

    @property
    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        return self._lb, self._ub


class _VectorDenseNumericView:
    _canonical_identity_backing = True

    def get_view(self, genomes: GenomeBatch) -> np.ndarray:
        if not isinstance(genomes, DenseVectorBatch):
            raise ValidationError(
                "DenseNumericView requires DenseVectorBatch, got "
                f"{type(genomes).__name__}"
            )
        # Returns the read-only array view which shares memory with the batch
        return genomes.array


class _VectorFeatureEncoder:
    """Lossless identity encoding for dense vector genomes."""

    def encode(self, genomes: GenomeBatch) -> np.ndarray:
        if not isinstance(genomes, DenseVectorBatch):
            raise ValidationError(
                "FeatureEncoder requires DenseVectorBatch, got "
                f"{type(genomes).__name__}"
            )
        return genomes.array


class _VectorCloneService:
    def clone(self, genomes: GenomeBatch) -> GenomeBatch:
        if not isinstance(genomes, DenseVectorBatch):
            raise ValidationError(
                f"CloneService requires DenseVectorBatch, got {type(genomes).__name__}"
            )
        # Produce an independent copy of the array data
        return DenseVectorBatch(genomes.array.copy())


class _VectorGenomeCodec:
    """Checkpoint codec for dense vector genome batches."""

    def encode(self, genomes: GenomeBatch) -> dict[str, object]:
        if not isinstance(genomes, DenseVectorBatch):
            raise ValidationError(
                f"GenomeCodec requires DenseVectorBatch, got {type(genomes).__name__}"
            )
        return {"array": np.array(genomes.array, dtype=np.float64, copy=True)}

    def decode(self, payload: dict[str, object]) -> GenomeBatch:
        array = payload.get("array")
        if not isinstance(array, np.ndarray):
            raise ValidationError("dense GenomeCodec payload is missing its array")
        if array.dtype != np.float64 or array.ndim != 2:
            raise ValidationError("dense GenomeCodec payload has invalid array")
        return DenseVectorBatch(array)


class _VectorDistanceService:
    def pairwise_distance(
        self, batch1: GenomeBatch, batch2: GenomeBatch | None = None
    ) -> np.ndarray:
        if not isinstance(batch1, DenseVectorBatch):
            raise ValidationError(
                "DistanceService requires DenseVectorBatch for batch1"
            )
        x1 = batch1.array
        if batch2 is None:
            x2 = x1
        else:
            if not isinstance(batch2, DenseVectorBatch):
                raise ValidationError(
                    "DistanceService requires DenseVectorBatch for batch2"
                )
            x2 = batch2.array

        if x1.shape[1] != x2.shape[1]:
            raise ValidationError(
                f"Dimension mismatch in DistanceService: {x1.shape[1]} vs {x2.shape[1]}"
            )

        diff = x1[:, np.newaxis, :] - x2[np.newaxis, :, :]
        return np.sqrt(np.sum(diff**2, axis=-1))

    def create_index(self, genomes: GenomeBatch) -> object:
        return _VectorDistanceIndex(_dense_array(genomes))

    def query_knn(
        self, index: object, genomes: GenomeBatch | np.ndarray, k: int
    ) -> tuple[np.ndarray, np.ndarray]:
        if not isinstance(index, _VectorDistanceIndex):
            raise ValidationError("DistanceService received an invalid index")
        if k < 1 or index.size == 0:
            return np.array([], dtype=np.intp), np.array([], dtype=np.float64)
        query = (
            genomes.array
            if isinstance(genomes, DenseVectorBatch)
            else np.asarray(genomes)
        )
        if query.ndim == 1:
            query = query.reshape(1, -1)
        if query.ndim != 2 or len(query) == 0:
            return np.array([], dtype=np.intp), np.array([], dtype=np.float64)
        return index.query(query[0], k)


class _VectorFingerprintService:
    def fingerprint(self, genomes: GenomeBatch) -> tuple[Hashable, ...]:
        if not isinstance(genomes, DenseVectorBatch):
            raise ValidationError(
                "FingerprintService requires DenseVectorBatch, got "
                f"{type(genomes).__name__}"
            )
        canonical = _canonical_rows(genomes.array)
        return tuple(row.tobytes() for row in canonical)

    def create_index(self) -> object:
        return _FingerprintIndex()

    def add_to_index(self, index: object, genomes: GenomeBatch) -> None:
        if not isinstance(index, _FingerprintIndex):
            raise ValidationError("FingerprintService received an invalid index")
        for fingerprint in self.fingerprint(genomes):
            index.values.setdefault(cast(bytes, fingerprint), index.size)
            index.size += 1

    def find_matches(self, index: object, genomes: GenomeBatch) -> np.ndarray:
        if not isinstance(index, _FingerprintIndex):
            raise ValidationError("FingerprintService received an invalid index")
        return np.asarray(
            [index.values.get(key, -1) for key in self.fingerprint(genomes)],
            dtype=np.intp,
        )


class _VectorEquivalenceService:
    def __init__(self, atol: float = 1e-16, rtol: float = 0.0) -> None:
        self._atol = atol
        self._rtol = rtol

    def find_duplicates(self, genomes: GenomeBatch) -> np.ndarray:
        if not isinstance(genomes, DenseVectorBatch):
            raise ValidationError(
                "EquivalenceService requires DenseVectorBatch, got "
                f"{type(genomes).__name__}"
            )
        arr = genomes.array
        n = len(arr)
        is_dup = np.zeros(n, dtype=bool)
        for i in range(n):
            if is_dup[i]:
                continue
            for j in range(i + 1, n):
                if not is_dup[j] and np.all(
                    np.isclose(arr[i], arr[j], atol=self._atol, rtol=self._rtol)
                ):
                    is_dup[j] = True
        return is_dup

    def find_matches(self, collection: GenomeBatch, genomes: GenomeBatch) -> np.ndarray:
        stored = _dense_array(collection)
        query = _dense_array(genomes)
        result = np.full(len(query), -1, dtype=np.intp)
        if len(stored) == 0:
            return result
        for position, row in enumerate(query):
            matches = np.all(
                np.isclose(stored, row, atol=self._atol, rtol=self._rtol), axis=1
            )
            first = int(matches.argmax())
            if matches[first]:
                result[position] = first
        return result


def _dense_array(genomes: GenomeBatch) -> np.ndarray:
    if not isinstance(genomes, DenseVectorBatch):
        raise ValidationError("Vector service requires DenseVectorBatch")
    return genomes.array


class _VectorDistanceIndex:
    """Lazy cKDTree handle owned by the vector DistanceService."""

    def __init__(self, genomes: np.ndarray) -> None:
        if genomes.ndim != 2:
            raise ValidationError("Distance index genomes must be a 2-D array")
        self.rows = np.array(genomes, dtype=np.float64, copy=True)
        self.tree: cKDTree | None = None
        self._finite_indices = np.empty(0, dtype=np.intp)

    @property
    def size(self) -> int:
        return len(self.rows)

    def query(self, point: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        if self.tree is None:
            finite = np.all(np.isfinite(self.rows), axis=1)
            self.tree = cKDTree(self.rows[finite]) if np.any(finite) else None
            self._finite_indices = np.flatnonzero(finite).astype(np.intp)
        if self.tree is None:
            return np.array([], dtype=np.intp), np.array([], dtype=np.float64)
        count = min(k, len(self.tree.data))
        distances, indices = self.tree.query(point, k=count)
        indices = np.atleast_1d(indices)
        distances = np.atleast_1d(distances)
        if indices.dtype != np.intp:
            indices = indices.astype(np.intp)
        if distances.dtype != np.float64:
            distances = distances.astype(np.float64)
        return (
            self._finite_indices[indices],
            distances,
        )


class _FingerprintIndex:
    def __init__(self) -> None:
        self.values: dict[bytes, int] = {}
        self.size = 0


class VectorSpace:
    """A search space for continuous numeric vectors defined by variable bounds.

    Parameters
    ----------
    dim : int
        Dimension of the vector space.
    lb : Sequence[float] | np.ndarray
        Lower bounds per dimension.
    ub : Sequence[float] | np.ndarray
        Upper bounds per dimension.
    atol : float, optional
        Absolute tolerance for approximate equivalence matching (default: 1e-16).
    rtol : float, optional
        Relative tolerance for approximate equivalence matching (default: 0.0).
    """

    def __init__(
        self,
        dim: int,
        lb: Sequence[float] | np.ndarray,
        ub: Sequence[float] | np.ndarray,
        *,
        atol: float = 1e-16,
        rtol: float = 0.0,
    ) -> None:
        if not isinstance(dim, int) or dim <= 0:
            raise ValidationError(f"dim must be a positive integer, got {dim!r}")

        lb_arr = np.asarray(lb, dtype=np.float64)
        ub_arr = np.asarray(ub, dtype=np.float64)

        if lb_arr.shape != (dim,) or ub_arr.shape != (dim,):
            raise ValidationError(
                f"lb and ub must have shape ({dim},), got {lb_arr.shape} "
                f"and {ub_arr.shape}"
            )

        if np.any(lb_arr > ub_arr):
            raise ValidationError(
                "Lower bound (lb) cannot be greater than upper bound (ub)"
            )

        lb_arr.setflags(write=False)
        ub_arr.setflags(write=False)
        self._dim = dim
        self._lb = lb_arr
        self._ub = ub_arr
        self._atol = atol
        self._rtol = rtol

        self._representation = RepresentationSpec(
            kind="vector",
            parameters=(ParameterSpec(name="dim", value=Fixed(value=dim)),),
        )

        self._services = ServiceRegistry()
        self._services.register("SamplingService", DerivedSamplingService(self))
        self._services.register("ValidationService", DerivedValidationService(self))
        self._services.register(
            "BoundsService", _VectorBoundsService(self._lb, self._ub)
        )
        self._services.register("DenseNumericView", _VectorDenseNumericView())
        self._services.register("GenomeCodec", _VectorGenomeCodec())
        self._services.register("CloneService", _VectorCloneService())
        self._services.register("DistanceService", _VectorDistanceService())
        self._services.register("FingerprintService", _VectorFingerprintService())
        self._services.register(
            "EquivalenceService",
            _VectorEquivalenceService(atol=self._atol, rtol=self._rtol),
        )

    @property
    def dim(self) -> int:
        """Return space dimension."""
        return self._dim

    @property
    def lb(self) -> np.ndarray:
        """Return lower bounds array."""
        return self._lb

    @property
    def ub(self) -> np.ndarray:
        """Return upper bounds array."""
        return self._ub

    @property
    def atol(self) -> float:
        """Return absolute tolerance for equivalence matching."""
        return self._atol

    @property
    def rtol(self) -> float:
        """Return relative tolerance for equivalence matching."""
        return self._rtol

    @property
    def representation(self) -> RepresentationSpec:
        """Return RepresentationSpec of this vector space."""
        return self._representation

    @property
    def services(self) -> ServiceRegistry:
        """Return ServiceRegistry offered by this vector space."""
        return self._services

    def sample(
        self, n: int, rng: np.random.Generator | None = None
    ) -> DenseVectorBatch:
        """Sample n random genomes uniformly within bounds."""
        if not isinstance(n, int) or n < 0:
            raise ValidationError(f"n must be a non-negative integer, got {n!r}")
        generator = rng if rng is not None else np.random.default_rng()
        data = generator.uniform(self._lb, self._ub, size=(n, self._dim))
        return DenseVectorBatch(data)

    def validate(self, genomes: GenomeBatch) -> ValidationResult:
        """Check if genomes batch is well-formed and within bounds."""
        if not isinstance(genomes, DenseVectorBatch):
            raise ValidationError(
                "VectorSpace validation requires DenseVectorBatch, got "
                f"{type(genomes).__name__}"
            )

        arr = genomes.array
        if arr.shape[1] != self._dim:
            raise ValidationError(
                "VectorSpace dimension mismatch: expected "
                f"{self._dim}, got {arr.shape[1]}"
            )

        if len(arr) == 0:
            return ValidationResult(valid_mask=(), errors=())

        finite_mask = np.all(np.isfinite(arr), axis=1)
        in_bounds_mask = np.all((arr >= self._lb) & (arr <= self._ub), axis=1)
        valid_mask_arr = finite_mask & in_bounds_mask

        valid = bool(np.all(valid_mask_arr))
        errors = (
            () if valid else ("One or more genomes are out of bounds or non-finite",)
        )
        return ValidationResult(
            valid_mask=tuple(bool(v) for v in valid_mask_arr),
            errors=errors,
        )
