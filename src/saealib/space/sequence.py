"""Variable-length sequence search space and representation services."""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from copy import deepcopy

import numpy as np

from saealib.core.contracts.data import Fixed
from saealib.core.contracts.representation import ParameterSpec, RepresentationSpec
from saealib.exceptions import ValidationError
from saealib.population.genome import GenomeBatch, VariableLengthBatch
from saealib.space.space import (
    DerivedSamplingService,
    DerivedValidationService,
    ServiceRegistry,
    ValidationResult,
)

__all__ = ["SequenceSpace"]


def _edit_distance(first: Sequence[object], second: Sequence[object]) -> int:
    previous = list(range(len(second) + 1))
    for i, left in enumerate(first, start=1):
        current = [i]
        for j, right in enumerate(second, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[j] + 1,
                    previous[j - 1] + (left != right),
                )
            )
        previous = current
    return previous[-1]


class _SequenceCloneService:
    def clone(self, genomes: GenomeBatch) -> VariableLengthBatch:
        if not isinstance(genomes, VariableLengthBatch):
            raise ValidationError("Sequence clone requires VariableLengthBatch")
        return VariableLengthBatch(deepcopy(genomes.sequences))


class _SequenceFingerprintService:
    def fingerprint(self, genomes: GenomeBatch) -> tuple[Hashable, ...]:
        if not isinstance(genomes, VariableLengthBatch):
            raise ValidationError("Sequence fingerprint requires VariableLengthBatch")
        try:
            return tuple(tuple(row) for row in genomes.sequences)
        except TypeError as exc:
            raise ValidationError(
                "Sequence elements must be hashable for fingerprinting"
            ) from exc

    def create_index(self) -> dict[Hashable, int]:
        return {}

    def add_to_index(self, index: object, genomes: GenomeBatch) -> None:
        if not isinstance(index, dict):
            raise ValidationError("Sequence fingerprint index is invalid")
        offset = max(index.values(), default=-1) + 1
        for row, fingerprint in enumerate(self.fingerprint(genomes)):
            index.setdefault(fingerprint, offset + row)

    def find_matches(self, index: object, genomes: GenomeBatch) -> np.ndarray:
        if not isinstance(index, dict):
            raise ValidationError("Sequence fingerprint index is invalid")
        return np.asarray(
            [
                int(index.get(fingerprint, -1))
                for fingerprint in self.fingerprint(genomes)
            ],
            dtype=np.intp,
        )


class _SequenceCodec:
    def encode(self, genomes: GenomeBatch) -> dict[str, object]:
        if not isinstance(genomes, VariableLengthBatch):
            raise ValidationError("Sequence GenomeCodec requires VariableLengthBatch")
        return {"items": [list(row) for row in genomes.sequences]}

    def decode(self, payload: dict[str, object]) -> VariableLengthBatch:
        items = payload.get("items")
        if not isinstance(items, list):
            raise ValidationError("Sequence GenomeCodec payload is malformed")
        try:
            return VariableLengthBatch(items)
        except (TypeError, ValueError) as exc:
            raise ValidationError("Sequence GenomeCodec payload is malformed") from exc


class _SequenceDistanceService:
    def pairwise_distance(
        self, batch1: GenomeBatch, batch2: GenomeBatch | None = None
    ) -> np.ndarray:
        if not isinstance(batch1, VariableLengthBatch):
            raise ValidationError("Sequence distance requires VariableLengthBatch")
        second = batch1 if batch2 is None else batch2
        if not isinstance(second, VariableLengthBatch):
            raise ValidationError("Sequence distance requires VariableLengthBatch")
        return np.asarray(
            [
                [_edit_distance(left, right) for right in second.sequences]
                for left in batch1.sequences
            ],
            dtype=np.float64,
        )

    def create_index(self, genomes: GenomeBatch) -> object:
        if not isinstance(genomes, VariableLengthBatch):
            raise ValidationError("Sequence distance requires VariableLengthBatch")
        return genomes

    def query_knn(
        self, index: object, genomes: GenomeBatch | np.ndarray, k: int
    ) -> tuple[np.ndarray, np.ndarray]:
        if not isinstance(index, VariableLengthBatch):
            raise ValidationError("Sequence distance index is invalid")
        query = genomes
        if not isinstance(query, VariableLengthBatch) or not len(query):
            raise ValidationError(
                "Sequence distance query requires VariableLengthBatch"
            )
        if k < 1 or not len(index):
            return np.empty(0, dtype=np.intp), np.empty(0, dtype=np.float64)
        distances = np.asarray(
            [_edit_distance(query.sequences[0], row) for row in index.sequences],
            dtype=np.float64,
        )
        order = np.argsort(distances, kind="stable")[: min(k, len(index))]
        return order.astype(np.intp), distances[order]


class _SequenceEquivalenceService:
    def __init__(self, threshold: int) -> None:
        self.threshold = threshold

    def find_duplicates(self, genomes: GenomeBatch) -> np.ndarray:
        if not isinstance(genomes, VariableLengthBatch):
            raise ValidationError("Sequence equivalence requires VariableLengthBatch")
        result = np.zeros(len(genomes), dtype=bool)
        for i in range(len(genomes)):
            result[i] = any(
                _edit_distance(genomes.sequences[i], genomes.sequences[j])
                <= self.threshold
                for j in range(i)
            )
        return result

    def find_matches(self, collection: GenomeBatch, genomes: GenomeBatch) -> np.ndarray:
        if not isinstance(collection, VariableLengthBatch) or not isinstance(
            genomes, VariableLengthBatch
        ):
            raise ValidationError("Sequence equivalence requires VariableLengthBatch")
        return np.asarray(
            [
                next(
                    (
                        index
                        for index, existing in enumerate(collection.sequences)
                        if _edit_distance(existing, query) <= self.threshold
                    ),
                    -1,
                )
                for query in genomes.sequences
            ],
            dtype=np.intp,
        )


class _SequenceFeatureEncoder:
    def __init__(self, alphabet: tuple[Hashable, ...], max_length: int) -> None:
        self._indices = {value: index + 1 for index, value in enumerate(alphabet)}
        self._max_length = max_length

    def encode(self, genomes: GenomeBatch) -> np.ndarray:
        if not isinstance(genomes, VariableLengthBatch):
            raise ValidationError(
                "Sequence FeatureEncoder requires VariableLengthBatch"
            )
        result = np.zeros((len(genomes), self._max_length + 1), dtype=np.float64)
        for row_index, sequence in enumerate(genomes.sequences):
            if len(sequence) > self._max_length:
                raise ValidationError("Sequence exceeds FeatureEncoder max_length")
            result[row_index, 0] = len(sequence)
            result[row_index, 1 : len(sequence) + 1] = [
                self._indices[value] for value in sequence
            ]
        return result


class SequenceSpace:
    """A search space for finite-alphabet sequences with a length range."""

    def __init__(
        self,
        alphabet: Sequence[Hashable],
        min_length: int = 0,
        max_length: int | None = None,
        *,
        equivalence_threshold: int = 0,
    ) -> None:
        values = tuple(alphabet)
        if not values or len(set(values)) != len(values):
            raise ValidationError("SequenceSpace alphabet must contain unique values")
        if not isinstance(min_length, int) or min_length < 0:
            raise ValidationError("min_length must be a non-negative integer")
        if max_length is None:
            max_length = min_length
        if not isinstance(max_length, int) or max_length < min_length:
            raise ValidationError("max_length must be an integer >= min_length")
        if not isinstance(equivalence_threshold, int) or equivalence_threshold < 0:
            raise ValidationError("equivalence_threshold must be non-negative")
        self._alphabet = values
        self._min_length = min_length
        self._max_length = max_length
        self._representation = RepresentationSpec(
            kind="sequence",
            parameters=(
                ParameterSpec(name="alphabet", value=Fixed(value=values)),
                ParameterSpec(name="min_length", value=Fixed(value=min_length)),
                ParameterSpec(name="max_length", value=Fixed(value=max_length)),
            ),
        )
        self._services = ServiceRegistry()
        self._services.register("SamplingService", DerivedSamplingService(self))
        self._services.register("ValidationService", DerivedValidationService(self))
        self._services.register("CloneService", _SequenceCloneService())
        self._services.register("FingerprintService", _SequenceFingerprintService())
        self._services.register(
            "EquivalenceService", _SequenceEquivalenceService(equivalence_threshold)
        )
        self._services.register("GenomeCodec", _SequenceCodec())
        self._services.register("DistanceService", _SequenceDistanceService())

    @property
    def alphabet(self) -> tuple[Hashable, ...]:
        """Return the immutable alphabet."""
        return self._alphabet

    @property
    def min_length(self) -> int:
        """Return the minimum sequence length."""
        return self._min_length

    @property
    def max_length(self) -> int:
        """Return the maximum sequence length."""
        return self._max_length

    @property
    def dim(self) -> int:
        """Return the feature width used by generic optimizer defaults."""
        return self._max_length + 1

    @property
    def representation(self) -> RepresentationSpec:
        """Return the sequence RepresentationSpec."""
        return self._representation

    @property
    def services(self) -> ServiceRegistry:
        """Return the services offered by this space."""
        return self._services

    def sample(
        self, n: int, rng: np.random.Generator | None = None
    ) -> VariableLengthBatch:
        """Draw ``n`` sequences with uniformly selected lengths."""
        if not isinstance(n, int) or n < 0:
            raise ValidationError(f"n must be a non-negative integer, got {n!r}")
        generator = rng if rng is not None else np.random.default_rng()
        rows: list[tuple[Hashable, ...]] = []
        for _ in range(n):
            length = int(generator.integers(self._min_length, self._max_length + 1))
            indices = generator.integers(0, len(self._alphabet), size=length)
            rows.append(tuple(self._alphabet[int(index)] for index in indices))
        return VariableLengthBatch(rows)

    def validate(self, genomes: GenomeBatch) -> ValidationResult:
        """Validate sequence type, alphabet membership, and length bounds."""
        if not isinstance(genomes, VariableLengthBatch):
            raise ValidationError(
                "SequenceSpace validation requires VariableLengthBatch"
            )
        valid: list[bool] = []
        for sequence in genomes.sequences:
            valid.append(
                self._min_length <= len(sequence) <= self._max_length
                and all(value in self._alphabet for value in sequence)
            )
        return ValidationResult(
            valid_mask=tuple(valid),
            errors=() if all(valid) else ("One or more sequences are invalid",),
        )
