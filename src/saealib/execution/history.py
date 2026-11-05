"""Execution history containers and generation summary recording."""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from saealib.exceptions import ValidationError
from saealib.execution.evaluator import EvaluationStatus, EvaluationUpdate
from saealib.space import DenseNumericView

if TYPE_CHECKING:
    from saealib.context import OptimizationState


SUPPORTED_HISTORY_CHANNELS = frozenset(
    {
        "summary",
        "front",
        "population",
        "surrogate_accuracy",
        "decision_candidates",
        "evaluation",
    }
)
_INTEGER_COLUMNS = frozenset(
    {
        "gen",
        "fe",
        "fe_before",
        "fe_after",
        "decision_count",
        "front_size",
        "size",
        "request_id",
        "sequence",
        "status_code",
        "attempt",
        "origin_decision_count",
        "origin_gen",
    }
)


@dataclass(frozen=True)
class _ObservedEvaluation:
    """One true evaluation update delivered at a runtime boundary.

    The observation sink buffers these so :func:`record_evaluations` has a single
    source of truth regardless of the sync/async delivery path.  ``attempt`` is
    the retry attempt that produced the update (``0`` on the sync path, which
    has no retries).
    """

    update: EvaluationUpdate
    attempt: int
    final_for_request: bool = True


class History:
    """Store rows for a selected set of named execution channels."""

    def __init__(self, channels: Sequence[str] = ("summary",)) -> None:
        names = tuple(channels)
        unknown = sorted(set(names) - SUPPORTED_HISTORY_CHANNELS)
        if unknown:
            raise ValidationError(f"Unknown history channel(s): {', '.join(unknown)}")
        self._enabled = frozenset(names)
        self._rows: dict[str, int] = {name: 0 for name in self._enabled}
        self._capacity: dict[str, int] = {name: 0 for name in self._enabled}
        self._columns: dict[str, dict[str, np.ndarray]] = {
            name: {} for name in self._enabled
        }
        self._record_modes: dict[str, str | None] = {
            name: None for name in self._enabled
        }
        self._block_columns: dict[str, dict[str, np.ndarray]] = {
            name: {} for name in self._enabled
        }
        self._block_offsets: dict[str, dict[str, np.ndarray]] = {
            name: {} for name in self._enabled
        }
        self._block_capacities: dict[str, dict[str, int]] = {
            name: {} for name in self._enabled
        }
        self._block_record_capacities: dict[str, dict[str, int]] = {
            name: {} for name in self._enabled
        }
        self._evaluation_keys: set[tuple[int, int, int, int]] = set()
        self._evaluation_origins: dict[int, tuple[int, int]] = {}
        self._evaluation_terminal_request_ids: set[int] = set()
        self._evaluation_resume_initialized = False
        self._evaluation_resume_request_ids: set[int] = set()
        self._surrogate_accuracy_keys: set[tuple[int, int, int, int]] = set()
        self._surrogate_predictions: dict[tuple[int, int], np.ndarray] = {}
        self._surrogate_resume_initialized = False
        self._surrogate_resume_candidate_ids: set[int] = set()
        self._observations: list[_ObservedEvaluation] = []
        self._fe_counter: int = 0
        self._fe_counted_ids: dict[int, set[int]] = {}
        self._fe_restored: bool = False

    @property
    def enabled(self) -> frozenset[str]:
        """Return the enabled channel names."""
        return self._enabled

    def is_enabled(self, name: str) -> bool:
        """Return whether a channel is enabled."""
        return name in self._enabled

    def _observe_evaluation(self, update: EvaluationUpdate, attempt: int) -> None:
        """Buffer one delivered true evaluation for the recorder.

        Delivery sources (the sync collect stage and the async scheduler commit
        path) call this; :func:`record_evaluations` is the only consumer.  When
        neither the ``evaluation`` nor ``surrogate_accuracy`` channel is enabled
        the observation is dropped immediately, matching the recorder's own
        early return.
        """
        if not (self.is_enabled("evaluation") or self.is_enabled("surrogate_accuracy")):
            return
        self._observations.append(_ObservedEvaluation(update, attempt))

    def _reopen_request(self, request_id: int) -> None:
        """Un-finalize the last observation of a request that will be retried."""
        for index in range(len(self._observations) - 1, -1, -1):
            observation = self._observations[index]
            if int(observation.update.request_id) == request_id:
                self._observations[index] = dataclasses.replace(
                    observation, final_for_request=False
                )
                return

    def append(self, channel: str, **columns: float | int) -> None:
        """Append one row to an enabled channel."""
        if not self.is_enabled(channel):
            return

        if self._record_modes[channel] == "block":
            raise ValidationError(
                f"History channel {channel!r} requires append_block()"
            )

        stored = self._columns[channel]
        names = tuple(columns)
        if self._record_modes[channel] == "scalar" and set(names) != set(stored):
            raise ValidationError(
                f"History channel {channel!r} columns changed from "
                f"{sorted(stored)!r} to {sorted(names)!r}"
            )
        if self._record_modes[channel] is None:
            self._record_modes[channel] = "scalar"
            stored.update(
                {name: np.empty(0, dtype=self._dtype_for(name)) for name in names}
            )
        elif not stored:
            stored.update(
                {name: np.empty(0, dtype=self._dtype_for(name)) for name in names}
            )

        values: dict[str, Any] = {}
        for name, value in columns.items():
            converted = np.asarray(value, dtype=self._dtype_for(name))
            if converted.ndim != 0:
                raise ValidationError(
                    f"History column {name!r} must receive one scalar value"
                )
            values[name] = converted.item()

        row = self._rows[channel]
        capacity = self._capacity[channel]
        if row == capacity:
            new_capacity = 1 if capacity == 0 else capacity * 2
            for name, array in stored.items():
                resized = np.empty(new_capacity, dtype=array.dtype)
                if row:
                    resized[:row] = array[:row]
                stored[name] = resized
            self._capacity[channel] = new_capacity

        for name, value in values.items():
            stored[name][row] = value
        self._rows[channel] = row + 1

    def append_block(
        self,
        channel: str,
        blocks: Mapping[str, np.ndarray],
        **columns: float | int,
    ) -> None:
        """Append one row containing scalar and variable-length block columns."""
        if not self.is_enabled(channel):
            return

        if self._record_modes[channel] == "scalar":
            raise ValidationError(f"History channel {channel!r} requires append()")

        if not isinstance(blocks, Mapping):
            raise ValidationError("History blocks must be a mapping")
        block_names = tuple(blocks)
        if any(not isinstance(name, str) for name in block_names):
            raise ValidationError("History block column names must be strings")

        stored = self._columns[channel]
        names = tuple(columns)
        if self._record_modes[channel] == "block":
            if set(names) != set(stored):
                raise ValidationError(
                    f"History channel {channel!r} columns changed from "
                    f"{sorted(stored)!r} to {sorted(names)!r}"
                )
            if set(block_names) != set(self._block_columns[channel]):
                raise ValidationError(
                    f"History channel {channel!r} block columns changed from "
                    f"{sorted(self._block_columns[channel])!r} to "
                    f"{sorted(block_names)!r}"
                )

        scalar_values: dict[str, Any] = {}
        for name, value in columns.items():
            try:
                converted = np.asarray(value, dtype=self._dtype_for(name))
            except (TypeError, ValueError) as exc:
                raise ValidationError(
                    f"History column {name!r} cannot receive one scalar value"
                ) from exc
            if converted.ndim != 0:
                raise ValidationError(
                    f"History column {name!r} must receive one scalar value"
                )
            scalar_values[name] = converted.item()

        block_values: dict[str, np.ndarray] = {}
        for name in block_names:
            try:
                value_array = np.asarray(blocks[name])
            except (TypeError, ValueError) as exc:
                raise ValidationError(
                    f"History block column {name!r} must be a 2-D float64, "
                    "int64, or bool array"
                ) from exc
            if value_array.ndim != 2:
                raise ValidationError(
                    f"History block column {name!r} must be a 2-D float64, "
                    "int64, or bool array"
                )
            existing = self._block_columns[channel].get(name)
            dtype = self._block_dtype(value_array)
            if dtype is None:
                raise ValidationError(
                    f"History block column {name!r} must be a 2-D float64, "
                    "int64, or bool array"
                )
            if existing is not None and dtype != existing.dtype:
                raise ValidationError(
                    f"History block column {name!r} changed its dtype category"
                )
            if existing is not None and value_array.shape[1] != existing.shape[1]:
                raise ValidationError(
                    f"History block column {name!r} changed its column count"
                )
            try:
                converted = np.asarray(value_array, dtype=dtype)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValidationError(
                    f"History block column {name!r} must be a 2-D float64, "
                    "int64, or bool array"
                ) from exc
            block_values[name] = converted

        if self._record_modes[channel] is None:
            self._record_modes[channel] = "block"
            stored.update(
                {name: np.empty(0, dtype=self._dtype_for(name)) for name in names}
            )
            for name, values in block_values.items():
                width = values.shape[1]
                self._block_columns[channel][name] = np.empty(
                    (0, width), dtype=values.dtype
                )
                self._block_offsets[channel][name] = np.empty(0, dtype=np.int64)
                self._block_capacities[channel][name] = 0
                self._block_record_capacities[channel][name] = 0

        row = self._rows[channel]
        capacity = self._capacity[channel]
        if row == capacity:
            new_capacity = 1 if capacity == 0 else capacity * 2
            for name, array in stored.items():
                resized = np.empty(new_capacity, dtype=array.dtype)
                if row:
                    resized[:row] = array[:row]
                stored[name] = resized
            self._capacity[channel] = new_capacity

        for name, values in block_values.items():
            value_buffer = self._block_columns[channel][name]
            offset_buffer = self._block_offsets[channel][name]
            value_capacity = self._block_capacities[channel][name]
            record_capacity = self._block_record_capacities[channel][name]
            start = 0 if row == 0 else int(offset_buffer[row])
            end = start + len(values)

            if end > value_capacity:
                new_value_capacity = 1 if value_capacity == 0 else value_capacity
                while new_value_capacity < end:
                    new_value_capacity *= 2
                resized_values = np.empty(
                    (new_value_capacity, value_buffer.shape[1]),
                    dtype=value_buffer.dtype,
                )
                if start:
                    resized_values[:start] = value_buffer[:start]
                value_buffer = resized_values
                self._block_columns[channel][name] = value_buffer
                self._block_capacities[channel][name] = new_value_capacity

            if row + 1 > record_capacity:
                new_record_capacity = 1 if record_capacity == 0 else record_capacity
                while new_record_capacity < row + 1:
                    new_record_capacity *= 2
                resized_offsets = np.empty(new_record_capacity + 1, dtype=np.int64)
                if row:
                    resized_offsets[: row + 1] = offset_buffer[: row + 1]
                else:
                    resized_offsets[0] = 0
                offset_buffer = resized_offsets
                self._block_offsets[channel][name] = offset_buffer
                self._block_record_capacities[channel][name] = new_record_capacity

            if len(values):
                value_buffer[start:end] = values
            offset_buffer[row + 1] = end

        for name, value in scalar_values.items():
            stored[name][row] = value
        self._rows[channel] = row + 1

    def channel(self, name: str) -> Mapping[str, np.ndarray]:
        """Return active scalar-column rows as read-only views.

        Block columns are not included; use :meth:`blocks` to read them.
        """
        if not self.is_enabled(name):
            raise ValidationError(f"History channel {name!r} is not enabled")
        row = self._rows[name]
        result: dict[str, np.ndarray] = {}
        for column, array in self._columns[name].items():
            view = array[:row]
            view.flags.writeable = False
            result[column] = view
        return MappingProxyType(result)

    def blocks(self, channel: str, column: str) -> tuple[np.ndarray, ...]:
        """Return each record's read-only two-dimensional block view."""
        if not self.is_enabled(channel):
            raise ValidationError(f"History channel {channel!r} is not enabled")
        if column not in self._block_columns[channel]:
            raise ValidationError(
                f"History block column {channel!r}/{column!r} is not available"
            )

        values = self._block_columns[channel][column]
        offsets = self._block_offsets[channel][column]
        result: list[np.ndarray] = []
        for index in range(self._rows[channel]):
            start = int(offsets[index])
            end = int(offsets[index + 1])
            view = values[start:end]
            view.flags.writeable = False
            result.append(view)
        return tuple(result)

    def get(self, channel: str, column: str) -> np.ndarray | tuple[np.ndarray, ...]:
        """Return a scalar or block column using its existing read-only views."""
        if not self.is_enabled(channel):
            raise ValidationError(f"History channel {channel!r} is not enabled")

        if column in self._columns[channel]:
            return self.channel(channel)[column]
        if column in self._block_columns[channel]:
            return self.blocks(channel, column)

        scalar_names = sorted(self._columns[channel])
        block_names = sorted(self._block_columns[channel])
        raise ValidationError(
            f"History column {channel!r}/{column!r} is not available; "
            f"available scalar columns: {scalar_names!r}; "
            f"available block columns: {block_names!r}"
        )

    def records(self, channel: str) -> Iterator[Mapping[str, Any]]:
        """Iterate over read-only, row-aligned records without copying data."""
        if not self.is_enabled(channel):
            raise ValidationError(f"History channel {channel!r} is not enabled")

        scalar_columns = self.channel(channel)
        block_columns = {
            column: self.blocks(channel, column)
            for column in self._block_columns[channel]
        }
        row_count = self._rows[channel]

        def iter_records() -> Iterator[Mapping[str, Any]]:
            for row in range(row_count):
                record: dict[str, Any] = {
                    column: values[row].item()
                    for column, values in scalar_columns.items()
                }
                for column, blocks in block_columns.items():
                    record.setdefault(column, blocks[row])
                yield MappingProxyType(record)

        return iter_records()

    def _restore_channel(self, name: str, columns: Mapping[str, np.ndarray]) -> None:
        """Restore a channel's columns without appending rows one at a time."""
        if not self.is_enabled(name):
            raise ValidationError(f"History channel {name!r} is not enabled")

        restored: dict[str, np.ndarray] = {}
        row_count: int | None = None
        for column, values in columns.items():
            if not isinstance(column, str):
                raise ValidationError("History column names must be strings")
            try:
                array = np.asarray(values)
                if array.ndim != 1:
                    raise ValidationError(
                        f"History column {column!r} must be one-dimensional"
                    )
                if row_count is None:
                    row_count = len(array)
                elif len(array) != row_count:
                    raise ValidationError(
                        f"History channel {name!r} columns have different lengths"
                    )
                restored[column] = np.asarray(
                    array, dtype=self._dtype_for(column)
                ).copy()
            except (TypeError, ValueError) as exc:
                raise ValidationError(
                    f"History column {column!r} cannot be restored"
                ) from exc

        rows = 0 if row_count is None else row_count
        self._columns[name] = restored
        self._rows[name] = rows
        self._capacity[name] = rows
        self._record_modes[name] = "scalar" if restored else None

    def _restore_blocks(
        self,
        name: str,
        columns: Mapping[str, tuple[np.ndarray, np.ndarray]],
    ) -> None:
        """Restore block columns without appending rows one at a time."""
        if not self.is_enabled(name):
            raise ValidationError(f"History channel {name!r} is not enabled")

        restored_values: dict[str, np.ndarray] = {}
        restored_offsets: dict[str, np.ndarray] = {}
        record_count: int | None = None
        for column, pair in columns.items():
            if not isinstance(column, str):
                raise ValidationError("History block column names must be strings")
            try:
                values, offsets = pair
                value_array = np.asarray(values)
                offset_array = np.asarray(offsets)
            except (TypeError, ValueError) as exc:
                raise ValidationError(
                    f"History block column {column!r} cannot be restored"
                ) from exc
            if value_array.ndim != 2 or value_array.dtype not in {
                np.dtype(np.float64),
                np.dtype(np.int64),
                np.dtype(np.bool_),
            }:
                raise ValidationError(
                    f"History block column {column!r} must be a 2-D float64, "
                    "int64, or bool array"
                )
            if offset_array.ndim != 1 or offset_array.dtype != np.int64:
                raise ValidationError(
                    f"History block column {column!r} offsets must be int64"
                )
            if len(offset_array) == 0:
                raise ValidationError(
                    f"History block column {column!r} offsets are empty"
                )
            if offset_array[0] != 0:
                raise ValidationError(
                    f"History block column {column!r} offsets must start at zero"
                )
            if np.any(np.diff(offset_array) < 0):
                raise ValidationError(
                    f"History block column {column!r} offsets are not monotonic"
                )
            if int(offset_array[-1]) != len(value_array):
                raise ValidationError(
                    f"History block column {column!r} offsets do not match values"
                )
            current_records = len(offset_array) - 1
            if record_count is None:
                record_count = current_records
            elif current_records != record_count:
                raise ValidationError(
                    "History block columns have different record counts"
                )
            restored_values[column] = value_array.copy()
            restored_offsets[column] = offset_array.copy()

        if not restored_values:
            return

        expected_records = self._rows[name]
        if self._columns[name]:
            if record_count != expected_records:
                raise ValidationError(
                    "History block columns have a different number of records"
                )
        else:
            self._rows[name] = 0 if record_count is None else record_count
            self._capacity[name] = self._rows[name]

        self._block_columns[name] = restored_values
        self._block_offsets[name] = restored_offsets
        self._block_capacities[name] = {
            column: len(values) for column, values in restored_values.items()
        }
        self._block_record_capacities[name] = {
            column: len(offsets) - 1 for column, offsets in restored_offsets.items()
        }
        self._record_modes[name] = "block"

    def _block_storage(self, name: str) -> Mapping[str, tuple[np.ndarray, np.ndarray]]:
        """Return active block storage for checkpoint serialization."""
        if not self.is_enabled(name):
            raise ValidationError(f"History channel {name!r} is not enabled")
        row = self._rows[name]
        return MappingProxyType(
            {
                column: (
                    values[: int(offsets[row])],
                    offsets[: row + 1],
                )
                for column, values in self._block_columns[name].items()
                for offsets in (self._block_offsets[name][column],)
            }
        )

    @staticmethod
    def _dtype_for(name: str) -> np.dtype:
        return np.dtype(np.int64 if name in _INTEGER_COLUMNS else np.float64)

    @staticmethod
    def _block_dtype(array: np.ndarray) -> np.dtype | None:
        if array.dtype.kind == "b":
            return np.dtype(np.bool_)
        if array.dtype.kind in "iu":
            return np.dtype(np.int64)
        if array.dtype.kind == "f":
            return np.dtype(np.float64)
        return None


def record_decision(state: OptimizationState) -> None:
    """Append one row for a confirmed evaluation decision."""
    _register_surrogate_predictions(state)
    history = state.history
    if history is None or not history.is_enabled("decision_candidates"):
        return

    recorded = history.channel("decision_candidates").get("decision_count")
    if (
        recorded is not None
        and len(recorded)
        and int(np.max(recorded)) >= state.decision_count
    ):
        return

    candidates = state.offspring
    plan = state.evaluation_plan
    if candidates is None or plan is None:
        return

    size = len(candidates)
    n_obj = state.problem.n_obj
    candidate_ids = np.array(candidates.get_array("id"), copy=True).reshape((size, 1))
    selected = np.zeros((size, 1), dtype=bool)
    candidate_rows = {
        int(candidate_id): row
        for row, candidate_id in enumerate(candidates.get_array("id"))
    }
    for request in plan.requests:
        for candidate_id in request.candidate_ids:
            row = candidate_rows.get(int(candidate_id))
            if row is not None:
                selected[row, 0] = True

    acquisition_scores = np.full((size, 1), np.nan, dtype=np.float64)
    acquisition = state.acquisition_result
    scores = acquisition.scores if acquisition is not None else state.scores
    if scores is not None:
        try:
            score_values = np.array(scores, dtype=np.float64, copy=True)
        except (TypeError, ValueError):
            score_values = None
        if score_values is not None and score_values.shape == (size,):
            acquisition_scores[:, 0] = score_values

    prediction_mean = np.full((size, n_obj), np.nan, dtype=np.float64)
    prediction_std = np.full((size, n_obj), np.nan, dtype=np.float64)
    predictions = state.predictions
    if predictions is not None:
        objective_channel = predictions.channels.get("objective")
        if objective_channel is not None:
            try:
                mean_values = np.array(
                    objective_channel.value, dtype=np.float64, copy=True
                )
            except (TypeError, ValueError):
                mean_values = None
            if mean_values is not None and mean_values.shape == (size, n_obj):
                prediction_mean = mean_values

            if objective_channel.std is not None:
                try:
                    std_values = np.array(
                        objective_channel.std, dtype=np.float64, copy=True
                    )
                except (TypeError, ValueError):
                    std_values = None
                if std_values is not None and std_values.shape == (size, n_obj):
                    prediction_std = std_values

    blocks: dict[str, np.ndarray] = {
        "candidate_ids": candidate_ids,
        "selected": selected,
        "acquisition_scores": acquisition_scores,
        "prediction_mean": prediction_mean,
        "prediction_std": prediction_std,
    }
    dense_view = cast(
        DenseNumericView | None,
        state.problem.space.services.get("DenseNumericView"),
    )
    if dense_view is not None:
        blocks["candidates"] = np.array(
            dense_view.get_view(candidates.genomes), dtype=np.float64, copy=True
        )

    history.append_block(
        "decision_candidates",
        blocks,
        decision_count=state.decision_count,
        gen=state.gen,
        fe=state.fe,
        size=size,
    )


def _request_candidate_ids(requests: Iterable[object]) -> set[int]:
    """Return candidate IDs from an iterable of evaluation requests."""
    result: set[int] = set()
    try:
        for request in requests:
            candidate_ids = np.asarray(getattr(request, "candidate_ids"))
            if candidate_ids.ndim != 1:
                continue
            result.update(int(candidate_id) for candidate_id in candidate_ids)
    except (AttributeError, TypeError, ValueError, OverflowError):
        return result
    return result


def _state_plan_candidate_ids(state: OptimizationState) -> set[int]:
    """Return candidate IDs covered by the state's current evaluation plan."""
    try:
        plan = state.evaluation_plan
        requests = getattr(plan, "requests", ()) if plan is not None else ()
    except (AttributeError, TypeError, ValueError, OverflowError):
        return set()
    return _request_candidate_ids(requests)


def _register_surrogate_predictions(state: OptimizationState) -> None:
    """Register objective predictions for the current confirmed plan.

    The mapping is held only in :class:`History` memory.  It is deliberately
    independent of the ``decision_candidates`` channel and is not persisted;
    therefore a plan restored in flight has no predictions to pair.
    """
    try:
        history = state.history
        if history is None or not history.is_enabled("surrogate_accuracy"):
            return
        data = state.data
        resumed = isinstance(data, Mapping) and bool(data.get("resumed", False))
    except (
        AttributeError,
        TypeError,
        ValueError,
        OverflowError,
        ValidationError,
    ):
        return

    if resumed and not history._surrogate_resume_initialized:
        history._surrogate_resume_candidate_ids.update(_state_plan_candidate_ids(state))
        history._surrogate_resume_initialized = True

    try:
        candidates = state.offspring
        plan = state.evaluation_plan
        predictions = state.predictions
        if candidates is None or plan is None or predictions is None:
            return
        objective_channel = predictions.channels.get("objective")
        if objective_channel is None:
            return
        candidate_ids = np.asarray(candidates.get_array("id"))
        predicted_values = np.asarray(objective_channel.value, dtype=np.float64)
        n_obj = int(state.problem.n_obj)
        if (
            candidate_ids.ndim != 1
            or len(candidate_ids) != len(candidates)
            or predicted_values.shape != (len(candidates), n_obj)
        ):
            return
        candidate_rows = {
            int(candidate_id): row for row, candidate_id in enumerate(candidate_ids)
        }
        resume_ids = history._surrogate_resume_candidate_ids
        for request in getattr(plan, "requests", ()):
            request_id = int(getattr(request, "request_id"))
            if request_id in history._evaluation_terminal_request_ids:
                continue
            request_candidate_ids = np.asarray(getattr(request, "candidate_ids"))
            if request_candidate_ids.ndim != 1:
                continue
            for candidate_id in np.asarray(request_candidate_ids, dtype=np.int64):
                cid = int(candidate_id)
                if cid in resume_ids:
                    continue
                row = candidate_rows.get(cid)
                if row is not None:
                    history._surrogate_predictions[(request_id, cid)] = np.array(
                        predicted_values[row], dtype=np.float64, copy=True
                    )
    except (
        AttributeError,
        TypeError,
        ValueError,
        OverflowError,
        ValidationError,
    ):
        return


_EVALUATION_STATUS_CODES = {
    EvaluationStatus.PENDING: 0,
    EvaluationStatus.RUNNING: 1,
    EvaluationStatus.PARTIAL: 2,
    EvaluationStatus.COMPLETED: 3,
    EvaluationStatus.FAILED: 4,
    EvaluationStatus.CANCELLED: 5,
}
_TERMINAL_EVALUATION_STATUS_CODES = frozenset({3, 4, 5})


def _evaluation_status_code(status: object) -> int:
    try:
        return _EVALUATION_STATUS_CODES.get(EvaluationStatus(status), -1)
    except (TypeError, ValueError):
        return -1


def _true_objectives_by_candidate(history: History) -> dict[int, np.ndarray]:
    """Return each candidate's last truly evaluated ``f`` from the History.

    Partial and completed records carry results. Failed and cancelled records
    carry results only when their evaluation count advances. A later result
    for the same candidate replaces an earlier one. Empty when the
    ``evaluation`` channel is not recorded in block mode.
    """
    if (
        not history.is_enabled("evaluation")
        or history._record_modes["evaluation"] != "block"
    ):
        return {}
    with_result = {
        _EVALUATION_STATUS_CODES[EvaluationStatus.PARTIAL],
        _EVALUATION_STATUS_CODES[EvaluationStatus.COMPLETED],
    }
    status = np.asarray(history.channel("evaluation")["status_code"])
    columns = history.channel("evaluation")
    fe_before = columns.get("fe_before")
    fe_after = columns.get("fe_after")
    candidate_blocks = history.blocks("evaluation", "candidate_ids")
    f_blocks = history.blocks("evaluation", "f")
    values: dict[int, np.ndarray] = {}
    for row, code in enumerate(status.tolist()):
        ids = np.asarray(candidate_blocks[row]).reshape(-1)
        if ids.size == 0:
            continue
        if code not in with_result:
            has_result = (
                code
                in {
                    _EVALUATION_STATUS_CODES[EvaluationStatus.FAILED],
                    _EVALUATION_STATUS_CODES[EvaluationStatus.CANCELLED],
                }
                and fe_before is not None
                and fe_after is not None
                and int(fe_after[row]) > int(fe_before[row])
            )
            if not has_result:
                continue
        f = np.asarray(f_blocks[row]).reshape(len(ids), -1)
        for position, candidate_id in enumerate(ids.tolist()):
            values[candidate_id] = f[position]
    return values


def _restore_evaluation_keys(history: History) -> None:
    """Rebuild the in-memory evaluation de-duplication keys from its rows."""
    if history._evaluation_keys:
        return
    try:
        columns = history.channel("evaluation")
        request_ids = columns.get("request_id")
        sequences = columns.get("sequence")
        status_codes = columns.get("status_code")
        attempts = columns.get("attempt")
        if request_ids is None or sequences is None or status_codes is None:
            return
        if attempts is None:
            attempts = np.zeros(len(request_ids), dtype=np.int64)
        if (
            len(request_ids) != len(sequences)
            or len(request_ids) != len(status_codes)
            or len(request_ids) != len(attempts)
        ):
            return
        for request_id, sequence, status_code, attempt in zip(
            request_ids, sequences, status_codes, attempts, strict=True
        ):
            try:
                history._evaluation_keys.add(
                    (int(request_id), int(sequence), int(status_code), int(attempt))
                )
            except (TypeError, ValueError, OverflowError):
                continue
    except (AttributeError, TypeError, ValueError, OverflowError, ValidationError):
        return


def _restore_surrogate_accuracy_keys(history: History) -> None:
    """Rebuild surrogate de-duplication keys from restored channel rows."""
    if history._surrogate_accuracy_keys:
        return
    try:
        columns = history.channel("surrogate_accuracy")
        request_ids = columns.get("request_id")
        sequences = columns.get("sequence")
        if (
            request_ids is None
            or sequences is None
            or len(request_ids) != len(sequences)
        ):
            return
        status_codes = columns.get("status_code")
        if status_codes is None:
            status_codes = np.full(len(request_ids), -1, dtype=np.int64)
        attempts = columns.get("attempt")
        if attempts is None:
            attempts = np.zeros(len(request_ids), dtype=np.int64)
        for request_id, sequence, status_code, attempt in zip(
            request_ids, sequences, status_codes, attempts, strict=True
        ):
            try:
                history._surrogate_accuracy_keys.add(
                    (int(request_id), int(sequence), int(status_code), int(attempt))
                )
            except (TypeError, ValueError, OverflowError):
                continue
    except (AttributeError, TypeError, ValueError, OverflowError, ValidationError):
        return


def _fill_missing_evaluation_columns(
    history: History, n_obj: int, n_constraints: int
) -> None:
    """NaN-fill evaluation block columns that older checkpoints lack."""
    if not history.is_enabled("evaluation"):
        return
    if history._record_modes["evaluation"] != "block":
        return
    block_columns = history._block_columns["evaluation"]
    missing = {"g", "noise"} - set(block_columns)
    if not missing:
        return
    reference_offsets = history._block_offsets["evaluation"]
    if not reference_offsets:
        return
    template = next(iter(reference_offsets.values()))
    row_count = history._rows["evaluation"]
    total = int(template[min(row_count, len(template) - 1)])
    widths = {"g": n_constraints, "noise": n_obj}
    for column in sorted(missing):
        block_columns[column] = np.full(
            (total, widths[column]), np.nan, dtype=np.float64
        )
        offsets = template[: row_count + 1].copy()
        history._block_offsets["evaluation"][column] = offsets
        history._block_capacities["evaluation"][column] = total
        history._block_record_capacities["evaluation"][column] = row_count


def _register_evaluation_origins(state: OptimizationState, history: History) -> None:
    """Remember request origins without adding state or checkpoint fields."""
    try:
        plan = state.evaluation_plan
        requests = getattr(plan, "requests", ()) if plan is not None else ()
        decision_count = int(state.decision_count)
        gen = int(state.gen)
        data = state.data
        resumed = isinstance(data, Mapping) and bool(data.get("resumed", False))
    except (AttributeError, TypeError, ValueError, OverflowError):
        return

    try:
        request_ids: list[int] = []
        for request in requests:
            try:
                request_id = int(request.request_id)
            except (AttributeError, TypeError, ValueError, OverflowError):
                continue
            request_ids.append(request_id)
        if resumed and not history._evaluation_resume_initialized:
            # WHY: Restored plans predate this session; current decision_count is wrong.
            history._evaluation_resume_request_ids.update(request_ids)
            history._evaluation_resume_initialized = True
        for request_id in request_ids:
            if request_id in history._evaluation_resume_request_ids:
                continue
            history._evaluation_origins.setdefault(request_id, (decision_count, gen))
    except (AttributeError, TypeError, ValueError, OverflowError):
        return


def _evaluation_blocks(
    state: OptimizationState, update: object
) -> tuple[int, np.ndarray, dict[str, np.ndarray]] | None:
    """Normalize one terminal update into the evaluation channel blocks."""
    try:
        result = getattr(update, "result")
        if result is None:
            candidate_ids = np.asarray(getattr(update, "candidate_ids"))
            if candidate_ids.ndim != 1:
                return None
            candidate_ids = np.asarray(candidate_ids, dtype=np.int64)
            size = len(candidate_ids)
            n_obj = int(state.problem.n_obj)
            n_constraints = int(state.problem.n_constraints)
            if n_obj < 0 or n_constraints < 0:
                return None
            nan_f = np.full((size, n_obj), np.nan, dtype=np.float64)
            nan_g = np.full((size, n_constraints), np.nan, dtype=np.float64)
            nan_noise = np.full((size, n_obj), np.nan, dtype=np.float64)
            nan_vector = np.full((size, 1), np.nan, dtype=np.float64)
            return (
                size,
                candidate_ids,
                {
                    "candidate_ids": candidate_ids.reshape((size, 1)),
                    "f": nan_f,
                    "g": nan_g,
                    "noise": nan_noise,
                    "cv": nan_vector,
                    "cost": nan_vector.copy(),
                },
            )

        result_candidate_ids = getattr(result, "candidate_ids")
        if result_candidate_ids is None:
            return None
        candidate_ids = np.asarray(result_candidate_ids)
        if candidate_ids.ndim != 1:
            return None
        candidate_ids = np.asarray(candidate_ids, dtype=np.int64)
        blocks = _result_blocks(state, candidate_ids, result)
        if blocks is None:
            return None
        return len(candidate_ids), candidate_ids, blocks
    except (AttributeError, TypeError, ValueError, OverflowError):
        return None


def _result_blocks(
    state: OptimizationState, candidate_ids: np.ndarray, result: object
) -> dict[str, np.ndarray] | None:
    """Shape one evaluation result into the evaluation channel blocks."""
    size = len(candidate_ids)
    n_obj = int(state.problem.n_obj)
    n_constraints = int(state.problem.n_constraints)
    f = np.asarray(getattr(result, "f"), dtype=np.float64)
    g = np.asarray(getattr(result, "g"), dtype=np.float64)
    cv = np.asarray(getattr(result, "cv"), dtype=np.float64)
    if f.shape != (size, n_obj) or cv.shape != (size,):
        return None
    if g.shape != (size, n_constraints):
        return None
    cost_value = getattr(result, "cost", None)
    if cost_value is None:
        cost = np.full(size, np.nan, dtype=np.float64)
    else:
        cost = np.asarray(cost_value, dtype=np.float64)
        if cost.shape != (size,):
            return None
    noise_value = getattr(result, "noise", None)
    if noise_value is None:
        noise = np.full((size, n_obj), np.nan, dtype=np.float64)
    else:
        noise = np.asarray(noise_value, dtype=np.float64)
        if noise.shape != (size, n_obj):
            return None
        noise = np.array(noise, dtype=np.float64, copy=True)
    return {
        "candidate_ids": candidate_ids.reshape((size, 1)),
        "f": np.array(f, dtype=np.float64, copy=True),
        "g": np.array(g, dtype=np.float64, copy=True),
        "noise": noise,
        "cv": cv.reshape((size, 1)),
        "cost": cost.reshape((size, 1)),
    }


def _empty_surrogate_accuracy_blocks(n_obj: int) -> dict[str, np.ndarray]:
    """Return empty float64 blocks with the objective width preserved."""
    empty = np.empty((0, n_obj), dtype=np.float64)
    return {"predicted": empty, "true": empty.copy()}


def _surrogate_accuracy_blocks(
    state: OptimizationState, update: object, request_id: int
) -> dict[str, np.ndarray]:
    """Build surrogate/true pairs for one terminal evaluation update."""
    try:
        n_obj = int(state.problem.n_obj)
    except (AttributeError, TypeError, ValueError, OverflowError):
        return _empty_surrogate_accuracy_blocks(0)

    empty = _empty_surrogate_accuracy_blocks(n_obj)
    try:
        result = getattr(update, "result")
        if result is None:
            return empty
        result_candidate_ids = getattr(result, "candidate_ids")
        if result_candidate_ids is None:
            return empty
        candidate_ids = np.asarray(result_candidate_ids)
        if candidate_ids.ndim != 1:
            return empty
        candidate_ids = np.asarray(candidate_ids, dtype=np.int64)
        true_values = np.asarray(getattr(result, "f"), dtype=np.float64)
        if true_values.shape != (len(candidate_ids), n_obj):
            return empty

        predicted_rows: list[np.ndarray] = []
        true_rows: list[np.ndarray] = []
        history = state.history
        if history is None:
            return empty
        for row, candidate_id in enumerate(candidate_ids):
            predicted = history._surrogate_predictions.get(
                (request_id, int(candidate_id))
            )
            if predicted is None:
                continue
            predicted_array = np.asarray(predicted, dtype=np.float64)
            if predicted_array.shape != (n_obj,):
                continue
            predicted_rows.append(np.array(predicted_array, copy=True))
            true_rows.append(np.array(true_values[row], copy=True))
        if not predicted_rows:
            return empty
        predicted_block = np.asarray(predicted_rows, dtype=np.float64)
        true_block = np.asarray(true_rows, dtype=np.float64)
        if (
            predicted_block.shape != true_block.shape
            or predicted_block.ndim != 2
            or predicted_block.shape[1] != n_obj
        ):
            return empty
        return {"predicted": predicted_block, "true": true_block}
    except (AttributeError, TypeError, ValueError, OverflowError):
        return empty


def _surrogate_terminal_candidate_ids(
    state: OptimizationState, update: object, request_id: int
) -> set[int]:
    """Return all candidate IDs whose terminal request must release mappings."""
    candidate_ids: set[int] = set()
    try:
        update_ids = np.asarray(getattr(update, "candidate_ids"))
        if update_ids.ndim == 1:
            candidate_ids.update(
                int(candidate_id)
                for candidate_id in np.asarray(update_ids, dtype=np.int64)
            )
    except (AttributeError, TypeError, ValueError, OverflowError):
        pass

    try:
        plan = state.evaluation_plan
        requests = getattr(plan, "requests", ()) if plan is not None else ()
        for request in requests:
            if int(getattr(request, "request_id")) == request_id:
                candidate_ids.update(_request_candidate_ids((request,)))
        pending = state.pending_evaluations.get(request_id)
        if pending is not None:
            original = getattr(pending, "original_candidate_ids", None)
            if original is not None:
                candidate_ids.update(
                    int(candidate_id)
                    for candidate_id in np.asarray(original, dtype=np.int64)
                )
            request = getattr(pending, "request", None)
            if request is not None:
                candidate_ids.update(_request_candidate_ids((request,)))
        request = state.evaluation_request
        if request is not None and int(request.request_id) == request_id:
            candidate_ids.update(_request_candidate_ids((request,)))
    except (AttributeError, TypeError, ValueError, OverflowError):
        pass
    return candidate_ids


def _result_candidate_ids(update: object) -> Iterable[int]:
    """Yield integer candidate IDs carried by an update's evaluation result.

    Updates without a result (FAILED / CANCELLED) yield nothing, so the
    recorder's ``fe`` counter does not advance for them.
    """
    try:
        result = getattr(update, "result", None)
        if result is None:
            return
        candidate_ids = getattr(result, "candidate_ids", None)
        if candidate_ids is None:
            return
        for candidate_id in np.asarray(candidate_ids, dtype=np.int64):
            yield int(candidate_id)
    except (AttributeError, TypeError, ValueError, OverflowError):
        return


def _advance_fe(
    history: History, request_id: int, candidate_ids: Iterable[int]
) -> tuple[int, int]:
    """Advance the recorder's cumulative ``fe`` counter by new candidate IDs.

    Returns ``(fe_before, fe_after)`` for one recorded row.  Only candidate IDs
    not yet counted under the same ``request_id`` advance the counter; the
    request's counted set is released when its terminal update is recorded.
    """
    before = history._fe_counter
    counted = history._fe_counted_ids.setdefault(request_id, set())
    new_count = 0
    for candidate_id in candidate_ids:
        if candidate_id not in counted:
            counted.add(candidate_id)
            new_count += 1
    after = before + new_count
    history._fe_counter = after
    return before, after


def _restore_fe_counter(history: History) -> None:
    """Resume the ``fe`` counter from persisted ``fe_after`` rows.

    The counter resumes from the maximum ``fe_after`` across the enabled
    ``evaluation`` / ``surrogate_accuracy`` channels.  The per-request counted
    candidate-ID sets are intentionally *not* restored: a resumable checkpoint
    predates this session, so candidate IDs already counted before the pause
    cannot be matched reliably.  This mirrors the documented ``origin = -1``
    resume limitation and only affects ``fe_before`` / ``fe_after`` alignment
    for evaluations that straddle the pause, not correctness of recorded rows.
    """
    if history._fe_restored:
        return
    max_fe_after = -1
    for channel in ("evaluation", "surrogate_accuracy"):
        if history.is_enabled(channel):
            columns = history.channel(channel)
            fe_after = columns.get("fe_after")
            if fe_after is not None and len(fe_after):
                max_fe_after = max(max_fe_after, int(np.max(fe_after)))
    if max_fe_after >= 0:
        history._fe_counter = max_fe_after
    history._fe_restored = True


def record_evaluations(state: OptimizationState) -> None:
    """Append evaluation updates observed at a runtime boundary.

    The recorder reads only the observation sink buffered by
    :meth:`History._observe_evaluation`; the sync collect stage and the async
    scheduler commit path are the only writers.  This gives a single source
    of truth independent of delivery path, so aggregated replicate updates
    and result-less async terminal updates never reach the recorder.

    ``gen``, ``fe_before``, ``fe_after``, and ``decision_count`` describe the
    state at which completion was observed.  ``fe_before`` / ``fe_after`` are
    not copied from ``state.fe``; they come from the recorder's own cumulative
    counter, which advances by the number of new candidate IDs (per
    ``request_id``) carried by each recorded update's result, so the boundary
    is mode-independent.  ``origin_decision_count`` and ``origin_gen`` describe
    the observation at which the request was issued, when that request was
    present in ``state.evaluation_plan``.  The status codes are:

    ``0 = pending, 1 = running, 2 = partial, 3 = completed,
    4 = failed, 5 = cancelled``; unknown values are ``-1``.

    Terminal updates (completed, failed, cancelled) are recorded.  PARTIAL
    updates that carry a true evaluation result (status code ``2``) are also
    recorded, because the delivered true evaluation is meaningful; PARTIAL
    updates without a result are progress notifications and are skipped.
    The de-duplication key is ``(request_id, sequence, status_code, attempt)``,
    so retry attempts that reuse a request ID are kept as distinct rows; the
    in-memory request-origin map, de-duplication set, and ``fe`` counter are
    intentionally not checkpointed.  After resume, requests already present in
    the restored plan have origin columns equal to ``-1``; the ``fe`` counter
    resumes from the maximum persisted ``fe_after`` (candidate-ID sets are not
    restored, matching the origin limitation).  Recorded keys are reconstructed
    from the channel columns.  ``state.data["resumed"]`` is the documented
    marker set by checkpoint loading (``OptimizationState.load`` in
    ``context.py``); all three checkpoint-loader paths set it.

    When enabled, ``surrogate_accuracy`` is recorded at the same update
    boundary (terminal updates and PARTIAL updates with a result).  Its
    candidate-to-prediction mapping is in-memory only; it is not checkpointed,
    so an in-flight update restored by resume records a row with ``size = 0``.
    The mapping is released only when the request's terminal update is
    observed, so a PARTIAL update does not discard the predictions that a
    later COMPLETED update for the same request still needs.

    Malformed updates are ignored so history recording cannot stop execution.
    """
    try:
        history = state.history
        if history is None:
            return
        evaluation_enabled = history.is_enabled("evaluation")
        surrogate_accuracy_enabled = history.is_enabled("surrogate_accuracy")
        if not (evaluation_enabled or surrogate_accuracy_enabled):
            return
        if evaluation_enabled:
            _restore_evaluation_keys(history)
            _register_evaluation_origins(state, history)
        if surrogate_accuracy_enabled:
            _restore_surrogate_accuracy_keys(history)
            _register_surrogate_predictions(state)
        _restore_fe_counter(history)
    except (AttributeError, TypeError, ValueError, OverflowError, ValidationError):
        return

    observations = list(history._observations)
    history._observations.clear()

    for observation in observations:
        update = observation.update
        attempt = observation.attempt
        try:
            request_id = int(getattr(update, "request_id"))
            sequence = int(getattr(update, "sequence"))
            status_code = _evaluation_status_code(getattr(update, "status"))
            is_terminal = status_code in _TERMINAL_EVALUATION_STATUS_CODES
            is_partial_result = (
                status_code == _EVALUATION_STATUS_CODES[EvaluationStatus.PARTIAL]
                and getattr(update, "result") is not None
            )
            if not (is_terminal or is_partial_result):
                continue
            key = (request_id, sequence, status_code, attempt)
        except (AttributeError, TypeError, ValueError, OverflowError):
            continue

        fe_before: int | None = None
        fe_after: int | None = None
        if evaluation_enabled and key not in history._evaluation_keys:
            normalized = _evaluation_blocks(state, update)
            if normalized is not None:
                fe_before, fe_after = _advance_fe(
                    history, request_id, _result_candidate_ids(update)
                )
                try:
                    size, _, blocks = normalized
                    origin_decision_count, origin_gen = history._evaluation_origins.get(
                        request_id, (-1, -1)
                    )
                    history.append_block(
                        "evaluation",
                        blocks,
                        request_id=request_id,
                        sequence=sequence,
                        status_code=status_code,
                        attempt=attempt,
                        gen=int(state.gen),
                        fe_before=fe_before,
                        fe_after=fe_after,
                        decision_count=int(state.decision_count),
                        origin_decision_count=origin_decision_count,
                        origin_gen=origin_gen,
                        size=size,
                    )
                except (
                    AttributeError,
                    TypeError,
                    ValueError,
                    OverflowError,
                    ValidationError,
                ):
                    pass
                else:
                    history._evaluation_keys.add(key)

        if surrogate_accuracy_enabled and key not in history._surrogate_accuracy_keys:
            if fe_before is None:
                fe_before, fe_after = _advance_fe(
                    history, request_id, _result_candidate_ids(update)
                )
            assert fe_before is not None and fe_after is not None
            try:
                blocks = _surrogate_accuracy_blocks(state, update, request_id)
                history.append_block(
                    "surrogate_accuracy",
                    blocks,
                    request_id=request_id,
                    sequence=sequence,
                    status_code=status_code,
                    attempt=attempt,
                    gen=int(state.gen),
                    fe_before=fe_before,
                    fe_after=fe_after,
                    decision_count=int(state.decision_count),
                    size=len(blocks["predicted"]),
                )
            except (
                AttributeError,
                TypeError,
                ValueError,
                OverflowError,
                ValidationError,
            ):
                pass
            else:
                history._surrogate_accuracy_keys.add(key)

        if is_terminal and observation.final_for_request:
            if surrogate_accuracy_enabled:
                history._evaluation_terminal_request_ids.add(request_id)
                for candidate_id in _surrogate_terminal_candidate_ids(
                    state, update, request_id
                ):
                    history._surrogate_predictions.pop((request_id, candidate_id), None)
            history._fe_counted_ids.pop(request_id, None)


def record_initial_evaluation(
    state: OptimizationState, result: object, ids: object
) -> None:
    """Record the initial evaluation as one synthetic completed row.

    Every Initializer calls this before ``ctx.count_fe()``, so the row carries
    ``fe_before = 0`` and ``fe_after = len(ids)`` from the recorder's own
    cumulative counter (the same counter used by :func:`record_evaluations`).
    ``request_id = -1`` marks it as not issued by a planner; ``IDAllocator``
    never allocates a negative ID.  No surrogate prediction exists yet, so
    ``surrogate_accuracy`` records nothing here.
    """
    try:
        history = state.history
        if history is None or not (
            history.is_enabled("evaluation") or history.is_enabled("surrogate_accuracy")
        ):
            return
        candidate_ids = np.asarray(ids, dtype=np.int64)
        if candidate_ids.ndim != 1:
            return
        if not history._fe_restored:
            history._fe_counter = 0
        history._fe_counter += len(candidate_ids)
        history._fe_restored = True
        if not history.is_enabled("evaluation"):
            return
        blocks = _result_blocks(state, candidate_ids, result)
        if blocks is None:
            return
        fe_before = history._fe_counter - len(candidate_ids)
        fe_after = history._fe_counter
        history.append_block(
            "evaluation",
            blocks,
            request_id=-1,
            sequence=0,
            status_code=_EVALUATION_STATUS_CODES[EvaluationStatus.COMPLETED],
            attempt=0,
            gen=0,
            fe_before=fe_before,
            fe_after=fe_after,
            decision_count=0,
            origin_decision_count=0,
            origin_gen=0,
            size=len(candidate_ids),
        )
        history._evaluation_keys.add(
            (-1, 0, _EVALUATION_STATUS_CODES[EvaluationStatus.COMPLETED], 0)
        )
    except (AttributeError, TypeError, ValueError, OverflowError, ValidationError):
        return


def record_generation(state: OptimizationState) -> None:
    """Append enabled generation-channel rows for the current state.

    ``best_f`` is the reported answer: it selects from ``state.archive`` using
    the strict user-declared ``problem.eps_cv``, preserving reported solutions
    and ``history_series("best")`` from relaxed ε values during exploration.
    ``feasible_ratio`` is an exploration diagnostic from ``state.population``
    using the running ``problem.handler.feasibility_threshold`` to expose ε's
    effect. These are different questions despite sharing a summary row.
    ``min_cv`` is the raw minimum cv in ``state.population`` and uses no threshold.

    The ``front`` channel stores the Pareto archive's objective values alone and
    has no cv column, so a front recorded while that archive holds no feasible
    solution is indistinguishable from a feasible one; ``front_size`` collapsing
    to a single point is the only sign, and ``min_cv`` cannot serve here because
    it describes the population rather than the archive.
    """
    history = state.history
    if history is None:
        return

    summary_enabled = history.is_enabled("summary")
    front_enabled = history.is_enabled("front")
    population_enabled = history.is_enabled("population")
    if not (summary_enabled or front_enabled or population_enabled):
        return

    if summary_enabled:
        pareto_archive = state.pareto_archive
        front_size = len(pareto_archive)
        n_obj = state.problem.n_obj
        columns: dict[str, float | int] = {
            "gen": state.gen,
            "fe": state.fe,
            "decision_count": state.decision_count,
            "front_size": front_size,
        }
        if n_obj == 1 and len(state.archive):
            archive_f = state.archive.get_array("f")
            cv = state.archive.get_array("cv")
            feasible = np.where(cv <= state.problem.eps_cv)[0]
            pool = feasible if len(feasible) else np.array([int(np.argmin(cv))])
            best = pool[np.argmax(archive_f[pool] @ state.direction)]
            columns["best_f"] = float(archive_f[best, 0])
        else:
            columns["best_f"] = np.nan
        if front_size:
            objective_values = pareto_archive.f
            f_min = np.min(objective_values, axis=0)
            f_max = np.max(objective_values, axis=0)
            for index in range(n_obj):
                columns[f"f_min_{index}"] = float(f_min[index])
                columns[f"f_max_{index}"] = float(f_max[index])
        else:
            for index in range(n_obj):
                columns[f"f_min_{index}"] = np.nan
                columns[f"f_max_{index}"] = np.nan

        population = state.population
        if len(population):
            cv = population.cv
            threshold = float(state.problem.handler.feasibility_threshold)
            columns["feasible_ratio"] = float(
                np.count_nonzero(cv <= threshold) / len(cv)
            )
            columns["min_cv"] = float(np.min(cv))
        else:
            columns["feasible_ratio"] = np.nan
            columns["min_cv"] = np.nan

        history.append("summary", **columns)

    if front_enabled:
        pareto_archive = state.pareto_archive
        snapshot = np.array(pareto_archive.f, dtype=np.float64, copy=True)
        history.append_block(
            "front",
            {"f": snapshot},
            gen=state.gen,
            fe=state.fe,
            decision_count=state.decision_count,
        )

    if population_enabled:
        population = state.population
        size = len(population)
        blocks: dict[str, np.ndarray] = {
            "f": np.array(population.f, dtype=np.float64, copy=True)
        }
        dense_view = cast(
            DenseNumericView | None,
            state.problem.space.services.get("DenseNumericView"),
        )
        if dense_view is not None:
            blocks["x"] = np.array(
                dense_view.get_view(population.genomes),
                dtype=np.float64,
                copy=True,
            )
        history.append_block(
            "population",
            blocks,
            gen=state.gen,
            fe=state.fe,
            decision_count=state.decision_count,
            size=size,
        )
