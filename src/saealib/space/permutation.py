"""Permutation search space and its representation services."""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from typing import Any, cast

import numpy as np

from saealib.core.contracts.data import Fixed
from saealib.core.contracts.representation import ParameterSpec, RepresentationSpec
from saealib.exceptions import ValidationError
from saealib.population.genome import GenomeBatch, PermutationBatch
from saealib.space.space import (
    DerivedSamplingService,
    DerivedValidationService,
    ServiceRegistry,
    ValidationResult,
)

__all__ = ["PermutationSpace"]


class _PermutationCloneService:
    def clone(self, genomes: GenomeBatch) -> PermutationBatch:
        if not isinstance(genomes, PermutationBatch):
            raise ValidationError("Permutation clone requires PermutationBatch")
        return PermutationBatch(genomes.array.copy(), length=genomes.length)


class _PermutationFingerprintService:
    def fingerprint(self, genomes: GenomeBatch) -> tuple[Hashable, ...]:
        if not isinstance(genomes, PermutationBatch):
            raise ValidationError("Permutation fingerprint requires PermutationBatch")
        return tuple(tuple(int(value) for value in row) for row in genomes.array)

    def create_index(self) -> dict[Hashable, int]:
        return {}

    def add_to_index(self, index: object, genomes: GenomeBatch) -> None:
        if not isinstance(index, dict):
            raise ValidationError("Permutation fingerprint index is invalid")
        offset = max(index.values(), default=-1) + 1
        for row, fingerprint in enumerate(self.fingerprint(genomes)):
            index.setdefault(fingerprint, offset + row)

    def find_matches(self, index: object, genomes: GenomeBatch) -> np.ndarray:
        if not isinstance(index, dict):
            raise ValidationError("Permutation fingerprint index is invalid")
        return np.asarray(
            [
                int(index.get(fingerprint, -1))
                for fingerprint in self.fingerprint(genomes)
            ],
            dtype=np.intp,
        )


class _PermutationCodec:
    def encode(self, genomes: GenomeBatch) -> dict[str, object]:
        if not isinstance(genomes, PermutationBatch):
            raise ValidationError("Permutation GenomeCodec requires PermutationBatch")
        return {
            "length": genomes.length,
            "items": genomes.array.astype(np.int64, copy=True).tolist(),
        }

    def decode(self, payload: dict[str, object]) -> PermutationBatch:
        try:
            length = int(cast(Any, payload["length"]))
            items = payload["items"]
        except (KeyError, TypeError, ValueError) as exc:
            raise ValidationError(
                "Permutation GenomeCodec payload is malformed"
            ) from exc
        return PermutationBatch(cast(Sequence[Sequence[int]], items), length=length)


class _PermutationDistanceService:
    def _array(self, genomes: GenomeBatch) -> np.ndarray:
        if not isinstance(genomes, PermutationBatch):
            raise ValidationError("Permutation distance requires PermutationBatch")
        return genomes.array

    def pairwise_distance(
        self, batch1: GenomeBatch, batch2: GenomeBatch | None = None
    ) -> np.ndarray:
        first = self._array(batch1)
        second = first if batch2 is None else self._array(batch2)
        return np.asarray(
            np.sum(first[:, None, :] != second[None, :, :], axis=2, dtype=np.float64)
        )

    def create_index(self, genomes: GenomeBatch) -> object:
        return self._array(genomes).copy()

    def query_knn(
        self, index: object, genomes: GenomeBatch | np.ndarray, k: int
    ) -> tuple[np.ndarray, np.ndarray]:
        if not isinstance(index, np.ndarray) or index.ndim != 2:
            raise ValidationError("Permutation distance index is invalid")
        query = (
            genomes
            if isinstance(genomes, PermutationBatch)
            else PermutationBatch(cast(np.ndarray, genomes))
        )
        if k < 1 or len(index) == 0 or len(query) == 0:
            return np.empty(0, dtype=np.intp), np.empty(0, dtype=np.float64)
        distances = np.sum(index != query.array[0], axis=1, dtype=np.float64)
        order = np.argsort(distances, kind="stable")[: min(k, len(index))]
        return order.astype(np.intp), np.asarray(distances)[order]


class _PermutationEquivalenceService:
    def find_duplicates(self, genomes: GenomeBatch) -> np.ndarray:
        fingerprints = _PermutationFingerprintService().fingerprint(genomes)
        seen: set[Hashable] = set()
        result = np.zeros(len(fingerprints), dtype=bool)
        for index, fingerprint in enumerate(fingerprints):
            result[index] = fingerprint in seen
            seen.add(fingerprint)
        return result

    def find_matches(self, collection: GenomeBatch, genomes: GenomeBatch) -> np.ndarray:
        existing = _PermutationFingerprintService().fingerprint(collection)
        positions = {fingerprint: index for index, fingerprint in enumerate(existing)}
        return np.asarray(
            [
                positions.get(fingerprint, -1)
                for fingerprint in _PermutationFingerprintService().fingerprint(genomes)
            ],
            dtype=np.intp,
        )


class _PermutationFeatureEncoder:
    def encode(self, genomes: GenomeBatch) -> np.ndarray:
        if not isinstance(genomes, PermutationBatch):
            raise ValidationError(
                "Permutation FeatureEncoder requires PermutationBatch"
            )
        return np.asarray(genomes.array, dtype=np.float64).copy()


class PermutationSpace:
    """A search space whose genomes are permutations of ``range(length)``."""

    def __init__(self, length: int) -> None:
        if not isinstance(length, int) or length < 1:
            raise ValidationError("PermutationSpace length must be a positive integer")
        self._length = length
        self._representation = RepresentationSpec(
            kind="permutation",
            parameters=(ParameterSpec(name="length", value=Fixed(value=length)),),
        )
        self._services = ServiceRegistry()
        self._services.register("SamplingService", DerivedSamplingService(self))
        self._services.register("ValidationService", DerivedValidationService(self))
        self._services.register("CloneService", _PermutationCloneService())
        self._services.register("FingerprintService", _PermutationFingerprintService())
        self._services.register("EquivalenceService", _PermutationEquivalenceService())
        self._services.register("GenomeCodec", _PermutationCodec())
        self._services.register("DistanceService", _PermutationDistanceService())

    @property
    def length(self) -> int:
        """Return the fixed permutation length."""
        return self._length

    @property
    def dim(self) -> int:
        """Return the fixed representation width for generic initializers."""
        return self._length

    @property
    def representation(self) -> RepresentationSpec:
        """Return the permutation RepresentationSpec."""
        return self._representation

    @property
    def services(self) -> ServiceRegistry:
        """Return the services offered by this space."""
        return self._services

    def sample(
        self, n: int, rng: np.random.Generator | None = None
    ) -> PermutationBatch:
        """Draw ``n`` uniformly shuffled permutations."""
        if not isinstance(n, int) or n < 0:
            raise ValidationError(f"n must be a non-negative integer, got {n!r}")
        generator = rng if rng is not None else np.random.default_rng()
        result = np.empty((n, self._length), dtype=np.int64)
        for index in range(n):
            result[index] = generator.permutation(self._length)
        return PermutationBatch(result, length=self._length)

    def validate(self, genomes: GenomeBatch) -> ValidationResult:
        """Validate the batch type and each permutation row."""
        if not isinstance(genomes, PermutationBatch):
            raise ValidationError(
                "PermutationSpace validation requires PermutationBatch"
            )
        if genomes.length != self._length:
            raise ValidationError(
                f"PermutationSpace expected length {self._length}, got {genomes.length}"
            )
        return ValidationResult(
            valid_mask=tuple(True for _ in range(len(genomes))), errors=()
        )
