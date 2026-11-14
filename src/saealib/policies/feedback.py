from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from functools import cache
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from saealib.context import OptimizationState
from saealib.core.contracts import (
    MANY,
    ComponentContract,
    DataSpec,
    FeedbackBatch,
    PartSpec,
    PortContract,
    PortDirection,
    PortSpec,
    ServiceRequirement,
)
from saealib.core.contracts.observation import (
    CONSTRAINT,
    CV,
    HUMAN,
    IMPUTED,
    OBJECTIVE,
    OBSERVATION_SOURCES,
    OBSERVATION_STATUSES,
    OK,
    SIMULATOR,
    SURROGATE,
    TRUE,
)
from saealib.core.contracts.observations import (
    ObservationBatch,
    ObservationRecords,
    ObservationSchema,
)
from saealib.core.state import POPULATIONS_MAIN
from saealib.exceptions import ValidationError
from saealib.execution.evaluator import EvaluationResult
from saealib.population.population import OBJECTIVE_SOURCE_COLUMN
from saealib.registry import register

if TYPE_CHECKING:
    from saealib.population import Population


@dataclass(frozen=True)
class FeedbackResult:
    """Validated values supplied to an algorithm tell operation."""

    candidate_ids: np.ndarray
    f: np.ndarray
    g: np.ndarray | None
    cv: np.ndarray | None
    evaluated_mask: np.ndarray
    source: np.ndarray
    artifacts: dict[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Normalize owned arrays."""
        ids = _array(self.candidate_ids, np.int64, "candidate_ids")
        f = _array(self.f, np.float64, "f")
        mask = _array(self.evaluated_mask, bool, "evaluated_mask")
        source = _array(self.source, np.uint8, "source")
        if ids.ndim != 1 or f.ndim != 2 or len(ids) != len(f):
            raise ValidationError("feedback candidate_ids and f are misaligned")
        if mask.shape != ids.shape or source.shape != ids.shape:
            raise ValidationError("feedback masks must align with candidate_ids")
        if len(np.unique(ids)) != len(ids):
            raise ValidationError("feedback candidate_ids must be unique")
        if self.g is not None:
            g = _array(self.g, np.float64, "g")
            if g.ndim != 2 or g.shape[0] != len(ids):
                raise ValidationError("feedback g has an invalid shape")
            object.__setattr__(self, "g", _readonly(g))
        if self.cv is not None:
            cv = _array(self.cv, np.float64, "cv")
            if cv.shape != (len(ids),):
                raise ValidationError("feedback cv has an invalid shape")
            object.__setattr__(self, "cv", _readonly(cv))
        for name, arr in (
            ("candidate_ids", ids),
            ("f", f),
            ("evaluated_mask", mask),
            ("source", source),
        ):
            object.__setattr__(self, name, _readonly(arr))
        artifacts = {}
        for name, value in self.artifacts.items():
            arr = np.array(value, copy=True)
            if arr.dtype == object or arr.ndim == 0 or arr.shape[0] != len(ids):
                raise ValidationError(
                    "feedback artifacts must have the candidate row count"
                )
            if not arr.flags.c_contiguous:
                arr = np.ascontiguousarray(arr)
            artifacts[name] = _readonly(arr)
        object.__setattr__(self, "artifacts", artifacts)


def _readonly(arr: np.ndarray) -> np.ndarray:
    arr.flags.writeable = False
    return arr


def _array(value, dtype, name: str) -> np.ndarray:
    arr = np.asarray(value)
    expected = np.dtype(dtype)
    if arr.dtype == object or arr.dtype != expected:
        raise ValidationError(f"feedback {name} must have dtype {expected}")
    return np.array(arr, dtype=expected, order="C", copy=True)


@dataclass(frozen=True)
class SelectionPolicy:
    """Named total ordering for records competing for one value."""

    name: str = "source-fidelity-sequence-status-batch-index"
    # Measured evidence precedes model output; human and simulator sources are
    # ordered before surrogate output.
    source_priority: tuple[str, ...] = (
        TRUE,
        HUMAN,
        SIMULATOR,
        SURROGATE,
        IMPUTED,
    )
    status_priority: tuple[str, ...] = (OK, "failed", "timeout", "cancelled")
    _source_rank_table: np.ndarray = field(init=False, repr=False, compare=False)
    _status_rank_table: np.ndarray = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Cache the tiny vocabulary rank tables for repeated hot-path calls."""
        object.__setattr__(
            self,
            "_source_rank_table",
            self._rank_table(OBSERVATION_SOURCES.names(), self.source_priority),
        )
        object.__setattr__(
            self,
            "_status_rank_table",
            self._rank_table(OBSERVATION_STATUSES.names(), self.status_priority),
        )

    @staticmethod
    def _rank_table(
        vocabulary_names: tuple[str, ...], priority: tuple[str, ...]
    ) -> np.ndarray:
        table = np.zeros(len(vocabulary_names), dtype=np.int64)
        for rank, name in enumerate(reversed(priority), start=1):
            if name in vocabulary_names:
                table[vocabulary_names.index(name)] = rank
        return table

    def source_rank(self, values: np.ndarray) -> np.ndarray:
        """Return descending priority ranks for source vocabulary values."""
        array = np.asarray(values)
        if array.dtype.kind in "iu":
            return self._source_rank_table[array.astype(np.intp, copy=False)]
        ranks = {
            name: len(self.source_priority) - i
            for i, name in enumerate(self.source_priority)
        }
        return np.fromiter(
            (ranks.get(str(value), 0) for value in array), dtype=np.int64
        )

    def status_rank(self, values: np.ndarray) -> np.ndarray:
        """Return descending priority ranks for status vocabulary values."""
        array = np.asarray(values)
        if array.dtype.kind in "iu":
            return self._status_rank_table[array.astype(np.intp, copy=False)]
        ranks = {
            name: len(self.status_priority) - i
            for i, name in enumerate(self.status_priority)
        }
        return np.fromiter(
            (ranks.get(str(value), 0) for value in array), dtype=np.int64
        )

    def select(
        self,
        candidate_positions: np.ndarray,
        source: np.ndarray,
        fidelity: np.ndarray,
        sequence: np.ndarray,
        status: np.ndarray,
        batch_index: np.ndarray,
    ) -> np.ndarray:
        """Return local rows selected by the declared lexicographic order."""
        source_rank = self.source_rank(source)
        status_rank = self.status_rank(status)
        order = np.lexsort(
            (
                batch_index,
                -status_rank,
                -sequence,
                -fidelity,
                -source_rank,
                candidate_positions,
            )
        )
        ordered_positions = candidate_positions[order]
        first = np.r_[
            True,
            ordered_positions[1:] != ordered_positions[:-1],
        ]
        return order[first]


@dataclass(frozen=True)
class FallbackPolicy:
    """Named policy for values which have no usable observation record."""

    name: str = "missing-value-mark-unusable"
    source: str = IMPUTED

    def initialize(
        self, n: int, n_obj: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Create materialized missing values, source labels, and masks."""
        return (
            np.full((n, n_obj), np.nan, dtype=np.float64),
            np.full(n, OBJECTIVE_SOURCE_CODES.get(self.source, 2), dtype=np.uint8),
            np.zeros(n, dtype=bool),
        )


DEFAULT_SELECTION_POLICY = SelectionPolicy()
MISSING_VALUE_FALLBACK_POLICY = FallbackPolicy()


def _legacy_batch(evaluation: EvaluationResult) -> ObservationBatch:
    if evaluation.candidate_ids is None:
        raise ValidationError("evaluation candidate IDs are required")
    schema = ObservationSchema(
        objective_count=evaluation.f.shape[1],
        constraint_count=evaluation.g.shape[1],
        quantities={CV: (0,)},
    )
    return ObservationBatch.from_dense(
        schema,
        evaluation.candidate_ids,
        evaluation.f,
        evaluation.g,
        evaluation.cv,
        source=TRUE,
        status=OK,
    )


def _candidate_ids_and_positions(candidates) -> tuple[np.ndarray, np.ndarray]:
    ids = _ids(candidates)
    order = np.argsort(ids, kind="stable")
    return ids, order


def _record_candidate_ids(records: ObservationRecords) -> np.ndarray:
    """Read single-candidate subjects without materializing record objects."""
    payload = records.column("subject_payload")
    array = np.asarray(payload)
    if array.ndim == 1:
        return np.asarray(array, dtype=np.int64)
    return np.asarray(array.reshape(len(records), -1)[:, 0], dtype=np.int64)


def _optional_column(records: ObservationRecords, name: str, default) -> np.ndarray:
    """Read an optional column while keeping dense batches compatible."""
    try:
        return records.column(name)
    except KeyError:
        result = np.empty(len(records), dtype=object)
        result[:] = default
    return result


def _metadata_numbers(
    values: np.ndarray, key: str | None, default: float
) -> np.ndarray:
    """Normalize optional numeric metadata without a hot-path object loop."""
    if len(values) == 0:
        return np.empty(0, dtype=np.float64)
    if values.dtype != object:
        return np.asarray(values, dtype=np.float64)
    if np.all(values == None):  # noqa: E711  # intentional object-column check
        return np.full(len(values), default, dtype=np.float64)
    try:
        return np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        if key is None:
            return np.full(len(values), default, dtype=np.float64)
        return np.fromiter(
            (
                float(
                    item.get(key, default)
                    if isinstance(item, Mapping)
                    else getattr(item, key, default)
                )
                if item is not None
                else default
                for item in values
            ),
            dtype=np.float64,
            count=len(values),
        )


OBJECTIVE_SOURCE_CODES = {TRUE: 0, SURROGATE: 1, IMPUTED: 2}


def _set_objective_source(
    population: Population,
    values: dict[str, Any],
    codes: np.ndarray | int,
    n_rows: int,
) -> None:
    """Add per-row objective source codes to an ``update_rows`` value mapping.

    Does nothing when the population schema has no objective source column.
    """
    if OBJECTIVE_SOURCE_COLUMN in population.schema:
        values[OBJECTIVE_SOURCE_COLUMN] = np.array(
            np.broadcast_to(np.asarray(codes, dtype=np.uint8), (n_rows,))
        )


def _set_objectives_true(
    population: Population, values: dict[str, Any], n_rows: int
) -> None:
    """Mark the rows of an ``update_rows`` mapping as truly evaluated."""
    _set_objective_source(population, values, OBJECTIVE_SOURCE_CODES[TRUE], n_rows)


def _feedback_batch_from_result(
    result: FeedbackResult,
    *,
    proposal_id: int,
    channel: str,
    final: bool,
    sequence: int,
) -> FeedbackBatch:
    """Wrap a materialized legacy result in the columnar feedback contract.

    ``FeedbackResult.source`` is row-level provenance, while ``channel`` is
    the delivery path.  Dense groups are built through ``from_dense`` so the
    common-source fast path remains vectorized; mixed provenance groups are
    concatenated and restored to the original candidate order.
    """
    candidate_ids = np.asarray(result.candidate_ids, dtype=np.int64)
    f = np.asarray(result.f, dtype=np.float64)
    g = (
        np.empty((len(candidate_ids), 0), dtype=np.float64)
        if result.g is None
        else np.asarray(result.g, dtype=np.float64)
    )
    cv = None if result.cv is None else np.asarray(result.cv, dtype=np.float64)
    if f.ndim != 2 or f.shape[0] != len(candidate_ids):
        raise ValidationError("feedback objective values are not row-aligned")
    if g.ndim != 2 or g.shape[0] != len(candidate_ids):
        raise ValidationError("feedback constraint values are not row-aligned")
    if cv is not None and (cv.ndim != 1 or len(cv) != len(candidate_ids)):
        raise ValidationError("feedback cv values are not row-aligned")
    source_codes = np.asarray(result.source, dtype=np.uint8)
    if source_codes.shape != (len(candidate_ids),):
        raise ValidationError("feedback source values are not row-aligned")
    schema = ObservationSchema(
        objective_count=f.shape[1],
        constraint_count=g.shape[1],
        quantities={CV: (0,)} if cv is not None else {},
    )
    source_names = {0: TRUE, 1: SURROGATE, 2: IMPUTED}
    parts: list[ObservationBatch] = []
    part_positions: list[np.ndarray] = []
    for code, source in source_names.items():
        positions = np.flatnonzero(source_codes == code)
        if len(positions) == 0:
            continue
        parts.append(
            ObservationBatch.from_dense(
                schema,
                candidate_ids[positions],
                f[positions],
                g[positions],
                None if cv is None else cv[positions],
                source=source,
                status=OK,
            )
        )
        part_positions.append(positions)
    if sum(len(positions) for positions in part_positions) != len(candidate_ids):
        raise ValidationError("feedback contains an unknown legacy source code")
    if not parts:
        empty = ObservationBatch.from_dense(
            schema,
            candidate_ids,
            f,
            g,
            cv,
            source=TRUE,
            status=OK,
        )
        return FeedbackBatch(
            proposal_id=proposal_id,
            observations=empty,
            channel=channel,
            final=final,
            sequence=sequence,
        )
    if len(parts) == 1:
        observations = parts[0]
    else:
        quantity_count = f.shape[1] + g.shape[1] + (cv is not None)
        records = ObservationRecords.concat([part.records for part in parts])
        original_positions = np.concatenate(part_positions)
        record_positions = np.repeat(original_positions, quantity_count)
        records = records.take(np.argsort(record_positions, kind="stable"))
        observations = ObservationBatch(schema=schema, records=records)
    return FeedbackBatch(
        proposal_id=proposal_id,
        observations=observations,
        channel=channel,
        final=final,
        sequence=sequence,
    )


@cache
def _objective_schema(n_obj: int) -> ObservationSchema:
    return ObservationSchema(objective_count=n_obj)


def _select_objective_records(
    records: ObservationRecords,
    record_positions: np.ndarray,
    valid_ids: np.ndarray,
    n_obj: int,
    selection_policy: SelectionPolicy,
) -> np.ndarray:
    """Select all objective indices with one columnar quantity filter."""
    objective_records = records.select(quantity_kind=OBJECTIVE)
    objective_indices = np.flatnonzero(records.column("quantity_kind") == OBJECTIVE)
    objective_valid = valid_ids[objective_indices]
    if not np.all(objective_valid):
        objective_records = objective_records.take(np.flatnonzero(objective_valid))
        objective_indices = objective_indices[objective_valid]
    if len(objective_records) == 0:
        return np.empty(0, dtype=np.intp)
    quantity_index = np.asarray(
        objective_records.column("quantity_index"), dtype=np.intp
    )
    group = record_positions[objective_indices] * n_obj + quantity_index
    chosen = selection_policy.select(
        group,
        objective_records.columns["source"],
        np.zeros(len(objective_records), dtype=np.float64),
        np.zeros(len(objective_records), dtype=np.int64),
        objective_records.columns["status"],
        objective_indices.astype(np.int64),
    )
    return objective_indices[chosen]


def _materialize_legacy_mixed(
    candidates,
    prediction,
    evaluation: EvaluationResult | None,
    selection_policy: SelectionPolicy,
    fallback_policy: FallbackPolicy,
    n_obj: int,
) -> FeedbackResult:
    """Materialize the legacy dense path without building a merged record set."""
    ids, id_order = _candidate_ids_and_positions(candidates)
    n = len(ids)
    if prediction is not None and "objective" in prediction.channels:
        predicted_values = prediction.channels["objective"].value
        if predicted_values.shape != (n, n_obj):
            raise ValidationError(
                "prediction objective shape does not match candidates"
            )
    else:
        predicted_values = np.full((n, n_obj), np.nan, dtype=np.float64)
    if evaluation is not None and evaluation.candidate_ids is None:
        evaluation = None
    if evaluation is not None and evaluation.f.shape[1] != n_obj:
        raise ValidationError("evaluation objective shape does not match candidates")

    candidate_positions = np.repeat(np.arange(n, dtype=np.intp), n_obj)
    quantity_indices = np.tile(np.arange(n_obj, dtype=np.intp), n)
    values = predicted_values.reshape(-1)
    sources = np.full(len(values), 1, dtype=np.int8)
    statuses = np.zeros(len(values), dtype=np.int8)
    batch_index = np.arange(len(values), dtype=np.int64)

    if evaluation is not None:
        # This is the legacy-to-observation adapter.  The select/column calls
        # are the same boundary used by the general ObservationBatch path.
        assert evaluation.candidate_ids is not None
        true_batch = ObservationBatch.from_dense(
            _objective_schema(n_obj),
            evaluation.candidate_ids,
            evaluation.f,
            np.empty((len(evaluation.f), 0), dtype=np.float64),
            source=TRUE,
            status=OK,
        )
        true_records = true_batch.records.select(quantity_kind=OBJECTIVE)
        true_ids = _record_candidate_ids(true_records)
        true_values = np.asarray(true_records.column("value"), dtype=np.float64)
        true_indices = np.asarray(true_records.column("quantity_index"), dtype=np.intp)
        true_sorted = np.searchsorted(ids[id_order], true_ids)
        if len(ids):
            true_valid = (true_sorted < len(ids)) & (
                ids[id_order[np.minimum(true_sorted, len(ids) - 1)]] == true_ids
            )
        else:
            true_valid = np.zeros(len(true_ids), dtype=bool)
        true_sorted = true_sorted[true_valid]
        true_indices = true_indices[true_valid]
        true_values = true_values[true_valid]
        true_positions = id_order[true_sorted]
        true_count = len(true_values)
        if true_count:
            candidate_positions = np.concatenate((candidate_positions, true_positions))
            quantity_indices = np.concatenate((quantity_indices, true_indices))
            values = np.concatenate((values, true_values))
            sources = np.concatenate((sources, np.zeros(true_count, dtype=np.int8)))
            statuses = np.concatenate((statuses, np.zeros(true_count, dtype=np.int8)))
            batch_index = np.concatenate(
                (batch_index, np.arange(true_count, dtype=np.int64) + len(batch_index))
            )

    group = candidate_positions * n_obj + quantity_indices
    selected = selection_policy.select(
        group,
        sources,
        np.zeros(len(values), dtype=np.float64),
        np.zeros(len(values), dtype=np.int64),
        statuses,
        batch_index,
    )
    f, selected_source, evaluated = fallback_policy.initialize(n, n_obj)
    selected_values = values[selected]
    usable = np.isfinite(selected_values)
    selected_positions = candidate_positions[selected]
    selected_indices = quantity_indices[selected]
    f[selected_positions[usable], selected_indices[usable]] = selected_values[usable]
    selected_source[selected_positions[usable]] = np.where(
        sources[selected][usable] == 0,
        OBJECTIVE_SOURCE_CODES[TRUE],
        OBJECTIVE_SOURCE_CODES[SURROGATE],
    ).astype(np.uint8)
    evaluated[selected_positions[usable]] = sources[selected][usable] == 0
    return FeedbackResult(ids, f, None, None, evaluated, selected_source)


def _materialize_feedback(
    candidates,
    prediction,
    evaluation,
    selection_policy: SelectionPolicy,
    fallback_policy: FallbackPolicy,
    n_obj: int,
    constraint_count: int = 0,
    include_constraints: bool = False,
) -> FeedbackResult:
    """Select and materialize dense feedback using columnar operations."""
    ids, id_order = _candidate_ids_and_positions(candidates)
    if not include_constraints and (
        evaluation is None or isinstance(evaluation, EvaluationResult)
    ):
        return _materialize_legacy_mixed(
            candidates,
            prediction,
            evaluation,
            selection_policy,
            fallback_policy,
            n_obj,
        )
    batches: list[ObservationBatch] = []
    if evaluation is not None:
        batches.append(
            evaluation
            if isinstance(evaluation, ObservationBatch)
            else _legacy_batch(evaluation)
        )
    if prediction is not None and "objective" in prediction.channels:
        values = prediction.channels["objective"].value
        if values.shape[0] != len(ids):
            raise ValidationError("prediction rows do not match candidates")
        batches.append(
            ObservationBatch.from_dense(
                ObservationSchema(objective_count=values.shape[1]),
                ids,
                values,
                np.empty((len(ids), 0), dtype=np.float64),
                source=SURROGATE,
                status=OK,
            )
        )
    if not batches:
        f, source, evaluated = fallback_policy.initialize(len(ids), n_obj)
        return FeedbackResult(ids, f, None, None, evaluated, source)

    records = ObservationRecords.concat([batch.records for batch in batches])
    if len(records) == 0:
        f, source, evaluated = fallback_policy.initialize(len(ids), n_obj)
        return FeedbackResult(ids, f, None, None, evaluated, source)
    record_ids = _record_candidate_ids(batches[0].records)
    if len(batches) > 1:
        record_ids = np.concatenate(
            [_record_candidate_ids(batch.records) for batch in batches]
        )
    sorted_positions = np.searchsorted(ids[id_order], record_ids)
    if len(ids):
        valid_ids = (sorted_positions < len(ids)) & (
            ids[id_order[np.minimum(sorted_positions, len(ids) - 1)]] == record_ids
        )
    else:
        valid_ids = np.zeros(len(record_ids), dtype=bool)
    positions = np.full(len(record_ids), -1, dtype=np.intp)
    positions[valid_ids] = id_order[sorted_positions[valid_ids]]
    all_kind = records.column("quantity_kind")
    all_index = records.column("quantity_index")
    all_values = np.asarray(records.column("value"), dtype=np.float64)
    all_source = records.column("source")
    all_status = records.column("status")
    if not include_constraints:
        selected = _select_objective_records(
            records, positions, valid_ids, n_obj, selection_policy
        )
    else:
        quantity_keys = np.empty(
            len(records), dtype=[("kind", "U32"), ("index", "U64")]
        )
        quantity_keys["kind"] = all_kind
        quantity_keys["index"] = np.asarray(all_index, dtype=str)
        unique_keys = np.unique(quantity_keys)
        selected_parts: list[np.ndarray] = []
        for key in unique_keys:
            kind = str(key["kind"])
            index_token = str(key["index"])
            try:
                index = int(index_token)
            except ValueError:
                index = index_token
            mask = (all_kind == kind) & (
                np.asarray(all_index, dtype=str) == index_token
            )
            mask &= valid_ids
            full_indices = np.flatnonzero(mask)
            if len(full_indices) == 0:
                continue
            # select() is the columnar filtering boundary; no ObservationRecord
            # is materialized while selecting a quantity.
            subset = records.select(quantity_kind=kind, quantity_index=index)
            subset_indices = np.flatnonzero(
                (all_kind == kind) & (np.asarray(all_index, dtype=str) == index_token)
            )
            subset_valid = valid_ids[subset_indices]
            if not np.all(subset_valid):
                subset = subset.take(np.flatnonzero(subset_valid))
                subset_indices = subset_indices[subset_valid]
            if len(subset) == 0:
                continue
            subset_ids = positions[subset_indices]
            fidelity = _metadata_numbers(
                _optional_column(subset, "fidelity", 0.0), None, 0.0
            )
            sequence = _metadata_numbers(
                _optional_column(subset, "provenance", None), "sequence", 0.0
            ).astype(np.int64)
            chosen = selection_policy.select(
                subset_ids,
                subset.columns["source"],
                fidelity,
                sequence,
                subset.columns["status"],
                subset_indices.astype(np.int64),
            )
            selected_parts.append(subset_indices[chosen])
        selected = (
            np.concatenate(selected_parts)
            if selected_parts
            else np.empty(0, dtype=np.intp)
        )
    f, selected_source, evaluated = fallback_policy.initialize(len(ids), n_obj)
    g = (
        np.full((len(ids), constraint_count), np.nan, dtype=np.float64)
        if include_constraints and constraint_count
        else None
    )
    cv = (
        np.full(len(ids), np.nan, dtype=np.float64)
        if include_constraints and np.any(all_kind[selected] == CV)
        else None
    )
    if len(selected):
        selected_positions = positions[selected]
        selected_kind = all_kind[selected]
        selected_index = np.asarray(all_index[selected], dtype=np.intp)
        selected_value = all_values[selected]
        selected_ok = all_status[selected] == OK
        selected_finite = np.isfinite(selected_value)
        usable = selected_ok & selected_finite
        objective = usable & (selected_kind == OBJECTIVE)
        if np.any(selected_index[objective] >= n_obj):
            raise ValidationError("feedback objective index exceeds n_obj")
        f[selected_positions[objective], selected_index[objective]] = selected_value[
            objective
        ]
        source_names = all_source[selected]
        objective_source = source_names[objective]
        selected_source[selected_positions[objective]] = np.where(
            objective_source == TRUE,
            OBJECTIVE_SOURCE_CODES[TRUE],
            np.where(
                objective_source == SURROGATE,
                OBJECTIVE_SOURCE_CODES[SURROGATE],
                OBJECTIVE_SOURCE_CODES.get(fallback_policy.source, 2),
            ),
        ).astype(np.uint8)
        evaluated[selected_positions[objective]] = objective_source == TRUE
        if g is not None:
            constraint = usable & (selected_kind == CONSTRAINT)
            if np.any(selected_index[constraint] >= constraint_count):
                raise ValidationError(
                    "feedback constraint index exceeds constraint_count"
                )
            g[selected_positions[constraint], selected_index[constraint]] = (
                selected_value[constraint]
            )
        if cv is not None:
            cv_mask = usable & (selected_kind == CV)
            cv[selected_positions[cv_mask]] = selected_value[cv_mask]
    return FeedbackResult(ids, f, g, cv, evaluated, selected_source)


def _ids(candidates: Population) -> np.ndarray:
    if "id" not in candidates.schema:
        return np.arange(len(candidates), dtype=np.int64)
    return np.array(candidates.get_array("id"), dtype=np.int64, copy=True)


def _true_rows(
    candidates: Population, evaluation: EvaluationResult | None
) -> tuple[np.ndarray, np.ndarray]:
    ids = _ids(candidates)
    if evaluation is None or evaluation.candidate_ids is None:
        return ids, np.zeros(len(ids), dtype=bool)
    lookup = {int(value): i for i, value in enumerate(evaluation.candidate_ids)}
    mask = np.array([int(value) in lookup for value in ids], dtype=bool)
    return ids, mask


def _empty(n_obj: int) -> FeedbackResult:
    return FeedbackResult(
        np.empty(0, dtype=np.int64),
        np.empty((0, n_obj), dtype=np.float64),
        None,
        None,
        np.empty(0, dtype=bool),
        np.empty(0, dtype=np.uint8),
    )


class FeedbackBuilder(ABC):
    """Build algorithm feedback from true and predicted values."""

    selection_policy = DEFAULT_SELECTION_POLICY
    fallback_policy = MISSING_VALUE_FALLBACK_POLICY

    def contract(self) -> ComponentContract:
        """Return the feedback-builder family contract."""
        return ComponentContract(
            ports={
                "feedback_builder": PortContract(
                    inputs=(
                        PortSpec(
                            name="candidates",
                            direction=PortDirection.INPUT,
                            data=DataSpec(kind="Population"),
                            cardinality=MANY,
                        ),
                        PortSpec(
                            name="prediction",
                            direction=PortDirection.INPUT,
                            data=DataSpec(kind="SurrogatePrediction"),
                            cardinality=MANY,
                            optional=True,
                        ),
                        PortSpec(
                            name="evaluation",
                            direction=PortDirection.INPUT,
                            data=DataSpec(kind="ObservationBatch"),
                            cardinality=MANY,
                            optional=True,
                        ),
                        PortSpec(
                            name="evaluated_indices",
                            direction=PortDirection.INPUT,
                            data=DataSpec(kind="Ordering"),
                            cardinality=MANY,
                        ),
                    ),
                    outputs=(
                        PortSpec(
                            name="feedback",
                            direction=PortDirection.OUTPUT,
                            data=DataSpec(kind="FeedbackBatch"),
                            cardinality=MANY,
                        ),
                    ),
                ),
            }
        )

    @abstractmethod
    def build(
        self,
        candidates,
        prediction,
        evaluation,
        evaluated_indices,
        *,
        ctx: OptimizationState | None,
    ):
        """Return feedback aligned to candidate IDs."""


@register()
class TrueOnlyFeedback(FeedbackBuilder):
    """Return completed true objective rows."""

    def __init__(self) -> None:
        pass

    def contract(self) -> ComponentContract:
        """Return the feedback contract; the builder reads no state."""
        return super().contract()

    def build(
        self,
        candidates,
        prediction,
        evaluation,
        evaluated_indices,
        *,
        ctx: OptimizationState | None,
    ):
        """Build true-only feedback."""
        ctx = cast(OptimizationState, ctx)
        if evaluation is None or evaluation.candidate_ids is None:
            return _empty(ctx.n_obj)
        if isinstance(evaluation, ObservationBatch):
            evaluation = ObservationBatch(
                schema=evaluation.schema,
                records=evaluation.records.select(source=TRUE),
            )
        result = _materialize_feedback(
            candidates,
            None,
            evaluation,
            self.selection_policy,
            self.fallback_policy,
            ctx.n_obj,
            constraint_count=evaluation.g.shape[1]
            if isinstance(evaluation, EvaluationResult)
            else len(evaluation.schema.indices(CONSTRAINT)),
            include_constraints=True,
        )
        rows = result.evaluated_mask
        return FeedbackResult(
            result.candidate_ids[rows],
            result.f[rows],
            None if result.g is None else result.g[rows],
            None if result.cv is None else result.cv[rows],
            result.evaluated_mask[rows],
            result.source[rows],
        )


@register()
class PredictedFeedback(FeedbackBuilder):
    """Return the objective prediction channel."""

    def __init__(self) -> None:
        pass

    def contract(self) -> ComponentContract:
        """Return the feedback contract requiring objective predictions."""
        family = super().contract()
        builder = family.ports["feedback_builder"]
        inputs = tuple(
            replace(port, optional=False) if port.name == "prediction" else port
            for port in builder.inputs
        )
        return replace(
            family,
            ports={
                **family.ports,
                "feedback_builder": replace(builder, inputs=inputs),
            },
        )

    def build(
        self,
        candidates,
        prediction,
        evaluation,
        evaluated_indices,
        *,
        ctx: OptimizationState | None,
    ):
        """Build predicted feedback."""
        if prediction is None or "objective" not in prediction.channels:
            raise ValidationError("PredictedFeedback requires an objective channel")
        ids = _ids(candidates)
        n = len(ids)
        values = prediction.value
        if values.shape[0] != n:
            raise ValidationError("prediction rows do not match candidates")
        mask = np.ones(n, dtype=bool)
        return FeedbackResult(
            ids,
            np.array(values, dtype=np.float64, copy=True),
            None,
            None,
            mask,
            np.ones(n, dtype=np.uint8),
        )


@register()
class MixedFeedback(FeedbackBuilder):
    """Prefer true rows and fill the remainder from objective predictions."""

    def __init__(self) -> None:
        pass

    def contract(self) -> ComponentContract:
        """Return the feedback contract; the builder reads no state."""
        return super().contract()

    def build(
        self,
        candidates,
        prediction,
        evaluation,
        evaluated_indices,
        *,
        ctx: OptimizationState | None,
    ):
        """Build mixed feedback."""
        ctx = cast(OptimizationState, ctx)
        if (
            isinstance(evaluation, EvaluationResult)
            and evaluation.candidate_ids is None
        ):
            evaluation = None
        ids = _ids(candidates)
        n = len(ids)
        predicted = None if prediction is None else prediction.channels.get("objective")
        if predicted is None:
            if isinstance(evaluation, ObservationBatch):
                n_obj = len(evaluation.schema.indices(OBJECTIVE))
            else:
                n_obj = evaluation.f.shape[1] if evaluation is not None else ctx.n_obj
        else:
            n_obj = predicted.value.shape[1]
        if (
            predicted is not None
            and evaluation is None
            and predicted.value.shape[1] != ctx.n_obj
        ):
            raise ValidationError(
                "prediction objective shape does not match candidates"
            )
        if predicted is not None and predicted.value.shape[1] != n_obj:
            raise ValidationError(
                "prediction objective shape does not match candidates"
            )
        result = _materialize_feedback(
            candidates,
            prediction,
            evaluation,
            self.selection_policy,
            self.fallback_policy,
            n_obj,
            constraint_count=(
                evaluation.g.shape[1]
                if isinstance(evaluation, EvaluationResult)
                else len(evaluation.schema.indices(CONSTRAINT))
                if isinstance(evaluation, ObservationBatch)
                else 0
            ),
            include_constraints=isinstance(evaluation, ObservationBatch),
        )
        # A prediction-only batch has no fallback values, preserving the legacy
        # source=1/NaN behavior until an explicit fallback wrapper is applied.
        if predicted is None and len(result.f) == n and result.g is None:
            result = FeedbackResult(
                result.candidate_ids,
                result.f,
                result.g,
                result.cv,
                result.evaluated_mask,
                np.where(np.isfinite(result.f).any(axis=1), result.source, 1),
            )
        return result


@register()
class NoFeedback(FeedbackBuilder):
    """Return an empty feedback batch."""

    def __init__(self) -> None:
        pass

    def build(
        self,
        candidates,
        prediction,
        evaluation,
        evaluated_indices,
        *,
        ctx: OptimizationState | None,
    ):
        """Build empty feedback."""
        ctx = cast(OptimizationState, ctx)
        return _empty(ctx.n_obj)


@register()
class ComparatorWorstFallback(FeedbackBuilder):
    """Fill missing rows with the comparator's worst population objective."""

    def __init__(self, inner: FeedbackBuilder | None = None) -> None:
        self.inner = inner or MixedFeedback()

    def contract(self) -> ComponentContract:
        """Return the feedback contract requiring population comparison."""
        family = super().contract()
        builder = family.ports["feedback_builder"]
        contract_method = getattr(self.inner, "contract", None)
        try:
            inner_contract = contract_method() if callable(contract_method) else None
        except Exception:
            inner_contract = None
        inner_builder = (
            inner_contract.ports.get("feedback_builder")
            if isinstance(inner_contract, ComponentContract)
            else None
        )
        required_inputs = (
            {port.name for port in inner_builder.inputs if not port.optional}
            if inner_builder is not None
            else set()
        )
        candidates = replace(
            builder.inputs[0],
            required_services=(ServiceRequirement(name="ComparisonService"),),
        )
        inputs = []
        for port in builder.inputs:
            if port.name == "candidates":
                inputs.append(candidates)
            elif port.name in required_inputs:
                inputs.append(replace(port, optional=False))
            else:
                inputs.append(port)
        parts = family.parts
        if inner_contract is not None:
            parts = (*parts, PartSpec(name="inner", contract=inner_contract))
        return replace(
            family,
            ports={
                **family.ports,
                "feedback_builder": replace(builder, inputs=tuple(inputs)),
            },
            parts=parts,
            state=replace(
                family.state,
                reads=(*family.state.reads, POPULATIONS_MAIN),
            ),
        )

    def build(
        self,
        candidates,
        prediction,
        evaluation,
        evaluated_indices,
        *,
        ctx: OptimizationState | None,
    ):
        """Build feedback and replace missing rows."""
        ctx = cast(OptimizationState, ctx)
        result = self.inner.build(
            candidates, prediction, evaluation, evaluated_indices, ctx=ctx
        )
        if len(result.candidate_ids) == 0:
            return result
        missing = np.flatnonzero(
            (result.source != 0) & np.any(~np.isfinite(result.f), axis=1)
        )
        if len(missing) == 0:
            return result
        order = ctx.comparator.sort_population(ctx.population)
        fallback = np.array(
            ctx.population.get_array("f")[order[-1]], dtype=np.float64, copy=True
        )
        f = np.array(result.f, copy=True)
        f[missing] = fallback
        source = np.array(result.source, copy=True)
        source[missing] = np.uint8(
            {TRUE: 0, SURROGATE: 1, self.inner.fallback_policy.source: 2}.get(
                self.inner.fallback_policy.source, 2
            )
        )
        return FeedbackResult(
            result.candidate_ids,
            f,
            result.g,
            result.cv,
            result.evaluated_mask,
            source,
            result.artifacts,
        )


def _result_from_evaluation(evaluation, positions, mask, source):
    g = None if evaluation.g is None else evaluation.g[positions]
    cv = None if evaluation.cv is None else evaluation.cv[positions]
    return FeedbackResult(
        evaluation.candidate_ids[positions],
        evaluation.f[positions],
        g,
        cv,
        np.array(mask, dtype=bool, copy=True),
        np.full(len(positions), source, dtype=np.uint8),
    )
