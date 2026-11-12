from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from saealib.acquisition.base import AcquisitionResult
from saealib.context import OptimizationState
from saealib.core.contracts import (
    MANY,
    ComponentContract,
    DataSpec,
    PortContract,
    PortDirection,
    PortSpec,
)
from saealib.core.state import (
    RUNTIME_CANDIDATE_ID_ALLOCATOR,
    RUNTIME_REQUEST_ID_ALLOCATOR,
)
from saealib.exceptions import ValidationError
from saealib.execution.evaluator import (
    EvaluationRequest,
    EvaluationResult,
    EvaluationStatus,
    EvaluationUpdate,
)
from saealib.population.genome import DenseVectorBatch
from saealib.registry import register

if TYPE_CHECKING:
    from saealib.population import Population


@dataclass(frozen=True)
class EvaluationPlan:
    """Composable evaluation work returned by a planner.

    A plan can contain one request (the ordinary fast path) or several
    requests such as replicates.  ``continuation`` is an opaque, serializable
    workflow descriptor for promotion/racing decisions; execution remains the
    evaluator/scheduler responsibility.
    """

    requests: tuple[EvaluationRequest, ...]
    completion_rule: Any = None
    continuation: Any = None
    artifacts: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate and own the request tuple and artifact mapping."""
        requests = tuple(self.requests)
        if not requests:
            raise ValidationError("an evaluation plan must contain a request")
        request_ids = [int(request.request_id) for request in requests]
        if len(request_ids) != len(set(request_ids)):
            raise ValidationError("an evaluation plan contains duplicate request IDs")
        object.__setattr__(self, "requests", requests)
        object.__setattr__(self, "artifacts", MappingProxyType(dict(self.artifacts)))

    def __reduce__(self):
        """Restore the immutable artifact mapping when pickling a plan."""
        return (
            type(self),
            (
                self.requests,
                self.completion_rule,
                self.continuation,
                dict(self.artifacts),
            ),
        )


class EvaluationPlanner(ABC):
    """Public contract for planners that may produce multiple requests."""

    _requires_acquisition_scores = False

    def contract(self) -> ComponentContract:
        """Return the evaluation-planner family contract."""
        return ComponentContract(
            ports={
                "evaluation_planner": PortContract(
                    inputs=(
                        PortSpec(
                            name="candidates",
                            direction=PortDirection.INPUT,
                            data=DataSpec(kind="Population"),
                            cardinality=MANY,
                        ),
                        PortSpec(
                            name="acquisition",
                            direction=PortDirection.INPUT,
                            data=DataSpec(kind="RowPredicate"),
                            cardinality=MANY,
                            optional=not self._requires_acquisition_scores,
                        ),
                    ),
                    outputs=(
                        PortSpec(
                            name="evaluation_requests",
                            direction=PortDirection.OUTPUT,
                            data=DataSpec(kind="EvaluationRequestBatch"),
                            cardinality=MANY,
                        ),
                    ),
                ),
            }
        )

    @abstractmethod
    def plan(
        self,
        candidates: Population,
        acquisition: AcquisitionResult | None,
        *,
        ctx: OptimizationState | None,
    ) -> EvaluationPlan:
        """Return composable evaluation work."""
        ...


def _scores(
    acquisition: AcquisitionResult | None, n: int, *, sanitize_nonfinite: bool = False
) -> np.ndarray:
    if acquisition is None or acquisition.scores is None:
        raise ValidationError("an acquisition score array is required")
    scores = np.asarray(acquisition.scores)
    if scores.shape != (n,) or scores.dtype != np.float64:
        raise ValidationError("acquisition scores must be float64 with shape (n,)")
    if not np.all(np.isfinite(scores)) and sanitize_nonfinite:
        scores = np.array(scores, copy=True)
        scores[~np.isfinite(scores)] = np.finfo(np.float64).min
    return scores


def select_top_k(scores: np.ndarray, k: int) -> np.ndarray:
    """Return stable descending top-k indices."""
    scores = np.asarray(scores)
    if scores.ndim != 1 or scores.dtype != np.float64:
        raise ValidationError("scores must be a float64 vector")
    if not np.all(np.isfinite(scores)):
        raise ValidationError("acquisition scores must be finite")
    if not isinstance(k, (int, np.integer)) or isinstance(k, bool):
        raise ValidationError("k must be an integer")
    if k < 0 or k > len(scores):
        raise ValidationError("k must be within the score vector")
    return np.array(np.argsort(-scores, kind="stable")[:k], dtype=np.intp, copy=True)


def select_ratio(scores: np.ndarray, ratio: float) -> np.ndarray:
    """Return stable descending indices for the ratio-selected prefix."""
    scores = np.asarray(scores)
    if scores.ndim != 1 or scores.dtype != np.float64:
        raise ValidationError("scores must be a float64 vector")
    if not np.isfinite(ratio) or ratio < 0.0 or ratio > 1.0:
        raise ValidationError("ratio must be finite and in [0, 1]")
    n = max(1, int(ratio * len(scores))) if len(scores) else 0
    return select_top_k(scores, n)


class _RequestPlanner(EvaluationPlanner):
    def contract(self) -> ComponentContract:
        """Return the request-planner contract declaring allocator state use."""
        family = super().contract()
        state = replace(
            family.state,
            reads=(
                *family.state.reads,
                RUNTIME_CANDIDATE_ID_ALLOCATOR,
                RUNTIME_REQUEST_ID_ALLOCATOR,
            ),
            writes=(
                *family.state.writes,
                RUNTIME_CANDIDATE_ID_ALLOCATOR,
                RUNTIME_REQUEST_ID_ALLOCATOR,
            ),
        )
        return replace(family, state=state)

    @staticmethod
    def _request(
        candidates: Population,
        indices: np.ndarray,
        ctx: OptimizationState | None,
    ) -> EvaluationRequest:
        ctx = cast(OptimizationState, ctx)
        if "id" not in candidates.schema:
            ids = np.arange(len(candidates), dtype=np.int64)
        else:
            current = np.asarray(candidates.get_array("id"), dtype=np.int64)
            missing = np.flatnonzero(current < 0)
            if len(missing):
                ids = ctx.candidate_id_allocator.allocate(len(missing))
                candidates._assign_ids(missing, ids)
        if "id" in candidates.schema:
            ids = np.array(
                candidates.get_array("id")[indices], dtype=np.int64, copy=True
            )
        else:
            ids = np.array(indices, dtype=np.int64, copy=True)
        if len(ids) != len(np.unique(ids)) or np.any(ids < 0):
            raise ValidationError("candidate IDs must be unique and assigned")
        request_id = ctx.request_id_allocator.allocate(1)[0]
        genomes = getattr(candidates, "genomes", None)
        if genomes is not None:
            payload = genomes.take(indices)
        else:
            # Keep compatibility with candidate fixtures that provide ``x``
            # instead of a GenomeBatch.
            # while ensuring the public request payload remains a GenomeBatch.
            try:
                payload = DenseVectorBatch(np.asarray(candidates.x)[indices])
            except AttributeError as exc:
                raise ValidationError(
                    "candidates must provide genomes or a legacy x array"
                ) from exc
        return EvaluationRequest(
            np.int64(request_id),
            ids,
            payload,
            metadata={"row_indices": indices.tolist()},
        )


@register()
class EvaluateAll(_RequestPlanner):
    """Evaluate every candidate."""

    def plan(self, candidates, acquisition, *, ctx: OptimizationState | None):
        """Build a request containing every candidate."""
        indices = np.arange(len(candidates), dtype=np.intp)
        return EvaluationPlan((self._request(candidates, indices, ctx),))


@register()
class TopKEvaluation(_RequestPlanner):
    """Evaluate the stable top-k acquisition scores."""

    _requires_acquisition_scores = True

    def __init__(self, k: int, sanitize_nonfinite: bool = False) -> None:
        self.k = k
        self.sanitize_nonfinite = sanitize_nonfinite

    def plan(self, candidates, acquisition, *, ctx: OptimizationState | None):
        """Build a request for the selected prefix."""
        indices = select_top_k(
            _scores(
                acquisition,
                len(candidates),
                sanitize_nonfinite=self.sanitize_nonfinite,
            ),
            self.k,
        )
        return EvaluationPlan((self._request(candidates, indices, ctx),))


@register()
class RatioEvaluation(_RequestPlanner):
    """Evaluate the stable prefix selected by a ratio."""

    _requires_acquisition_scores = True

    def __init__(self, ratio: float, sanitize_nonfinite: bool = False) -> None:
        self.ratio = ratio
        self.sanitize_nonfinite = sanitize_nonfinite

    def plan(self, candidates, acquisition, *, ctx: OptimizationState | None):
        """Build a request for the ratio-selected prefix."""
        scores = _scores(
            acquisition,
            len(candidates),
            sanitize_nonfinite=self.sanitize_nonfinite,
        )
        indices = select_ratio(scores, self.ratio)
        return EvaluationPlan((self._request(candidates, indices, ctx),))


@register()
class RepeatedEvaluation(_RequestPlanner):
    """Request candidates with an explicit replicate number."""

    def __init__(self, replicates: int = 2):
        if (
            not isinstance(replicates, int)
            or isinstance(replicates, bool)
            or replicates < 1
        ):
            raise ValidationError("replicates must be a positive integer")
        self.replicates = replicates

    def plan(self, candidates, acquisition, *, ctx: OptimizationState | None):
        """Build one request per replicate."""
        requests = self.plan_replicates(candidates, ctx)
        plan_id = int(requests[0].request_id)
        requests = tuple(
            EvaluationRequest(
                request.request_id,
                request.candidate_ids,
                request.payload,
                request.outputs,
                {**request.metadata, "plan_id": plan_id},
            )
            for request in requests
        )
        return EvaluationPlan(
            requests,
            completion_rule="all_requests_completed",
            artifacts={"replicates": self.replicates},
        )

    def plan_replicates(self, candidates, ctx):
        """Build one request per replicate for the same candidate IDs."""
        first = EvaluateAll().plan(candidates, None, ctx=ctx).requests[0]
        requests = []
        for replicate in range(self.replicates):
            request_id = first.request_id
            if replicate:
                request_id = np.int64(ctx.request_id_allocator.allocate(1)[0])
            requests.append(
                EvaluationRequest(
                    request_id,
                    first.candidate_ids,
                    first.payload,
                    first.outputs,
                    {**first.metadata, "replicate": replicate},
                )
            )
        return tuple(requests)


@dataclass(frozen=True)
class ReplicateSummary:
    """Aggregated observations for one candidate batch."""

    candidate_ids: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    count: np.ndarray

    def __post_init__(self):
        """Own arrays and validate aggregate shapes."""
        ids = np.array(self.candidate_ids, dtype=np.int64, copy=True)
        mean = np.array(self.mean, dtype=np.float64, copy=True)
        std = np.array(self.std, dtype=np.float64, copy=True)
        count = np.array(self.count, dtype=np.int64, copy=True)
        if mean.ndim != 2 or std.shape != mean.shape or count.shape != (len(ids),):
            raise ValidationError("replicate summary shapes are inconsistent")
        object.__setattr__(self, "candidate_ids", ids)
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "std", std)
        object.__setattr__(self, "count", count)


def aggregate_replicates(candidate_ids, observations) -> ReplicateSummary:
    """Aggregate repeated objective observations by stable candidate ID."""
    ids = np.asarray(candidate_ids, dtype=np.int64)
    values = np.asarray(observations, dtype=np.float64)
    if values.ndim != 3 or values.shape[1] != len(ids):
        raise ValidationError(
            "observations must have shape (replicate, candidate, objective)"
        )
    return ReplicateSummary(
        ids,
        values.mean(axis=0),
        values.std(axis=0),
        np.full(len(ids), values.shape[0], dtype=np.int64),
    )


def _aggregate_repeated_updates(
    plan: EvaluationPlan,
    updates_by_request: Mapping[int, Iterable[EvaluationUpdate]],
    final_update: EvaluationUpdate,
) -> EvaluationUpdate:
    """Aggregate a completed repeated plan for the standard lifecycle."""
    if final_update.status is not EvaluationStatus.COMPLETED:
        return final_update
    observations: dict[int, list[tuple[np.ndarray, np.ndarray, np.ndarray]]] = {}
    for request in plan.requests:
        for update in updates_by_request.get(int(request.request_id), ()):
            if update.result is None:
                continue
            for row, candidate_id in enumerate(update.candidate_ids):
                observations.setdefault(int(candidate_id), []).append(
                    (
                        np.asarray(update.result.f[row], dtype=np.float64),
                        np.asarray(update.result.g[row], dtype=np.float64),
                        np.asarray(update.result.cv[row], dtype=np.float64),
                    )
                )
    candidate_ids = np.asarray(plan.requests[0].candidate_ids, dtype=np.int64)
    if any(int(candidate_id) not in observations for candidate_id in candidate_ids):
        return final_update
    f = np.asarray(
        [
            np.mean([item[0] for item in observations[int(candidate_id)]], axis=0)
            for candidate_id in candidate_ids
        ],
        dtype=np.float64,
    )
    g = np.asarray(
        [
            np.mean([item[1] for item in observations[int(candidate_id)]], axis=0)
            for candidate_id in candidate_ids
        ],
        dtype=np.float64,
    )
    cv = np.asarray(
        [
            np.mean([item[2] for item in observations[int(candidate_id)]])
            for candidate_id in candidate_ids
        ],
        dtype=np.float64,
    )
    return EvaluationUpdate(
        final_update.request_id,
        final_update.status,
        candidate_ids,
        EvaluationResult(f, g, cv, candidate_ids=candidate_ids),
        final_update.error,
        final_update.sequence,
    )


def _combine_plan_updates(
    plan: EvaluationPlan,
    updates_by_request: Mapping[int, Iterable[EvaluationUpdate]],
    final_update: EvaluationUpdate,
) -> EvaluationUpdate:
    """Combine the latest result for each candidate in a multi-request plan."""
    values: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    ordered_requests = sorted(
        enumerate(plan.requests),
        key=lambda item: (
            float(item[1].metadata.get("fidelity", -np.inf)),
            item[0],
        ),
    )
    for _, request in ordered_requests:
        for update in updates_by_request.get(int(request.request_id), ()):
            if update.result is None:
                continue
            for row, candidate_id in enumerate(update.candidate_ids):
                values[int(candidate_id)] = (
                    np.asarray(update.result.f[row], dtype=np.float64),
                    np.asarray(update.result.g[row], dtype=np.float64),
                    np.asarray(update.result.cv[row], dtype=np.float64),
                )
    candidate_order: list[int] = []
    seen: set[int] = set()
    for request in plan.requests:
        for candidate_id in request.candidate_ids:
            candidate_id = int(candidate_id)
            if candidate_id in values and candidate_id not in seen:
                candidate_order.append(candidate_id)
                seen.add(candidate_id)
    candidate_ids = np.asarray(candidate_order, dtype=np.int64)
    if len(candidate_ids) == 0:
        return final_update
    f = np.asarray([values[int(candidate_id)][0] for candidate_id in candidate_ids])
    g = np.asarray([values[int(candidate_id)][1] for candidate_id in candidate_ids])
    cv = np.asarray([values[int(candidate_id)][2] for candidate_id in candidate_ids])
    return EvaluationUpdate(
        final_update.request_id,
        final_update.status,
        candidate_ids,
        EvaluationResult(f, g, cv, candidate_ids=candidate_ids),
        final_update.error,
        final_update.sequence,
    )


def _continue_fidelity_plan(
    plan: EvaluationPlan,
    updates_by_request: Mapping[int, Iterable[EvaluationUpdate]],
    ctx: OptimizationState,
) -> EvaluationPlan | None:
    """Add the selected high-fidelity request to a promotion plan."""
    continuation = plan.continuation
    if not isinstance(continuation, Mapping):
        return None
    if continuation.get("kind") != "fidelity_promotion":
        return None
    if len(plan.requests) != 1:
        return None
    low_request = plan.requests[0]
    low_update = None
    for update in updates_by_request.get(int(low_request.request_id), ()):
        if update.result is not None:
            low_update = update
    if low_update is None or low_update.result is None:
        return None
    count = continuation.get("promotion_count")
    if count is None:
        fraction = float(continuation.get("promotion_fraction", 0.5))
        count = max(1, int(np.ceil(len(low_update.candidate_ids) * fraction)))
    count = min(int(count), len(low_update.candidate_ids))
    if count < 1:
        return None
    direction = np.asarray(ctx.problem.direction, dtype=np.float64)
    order = np.argsort(
        -direction[0] * np.asarray(low_update.result.f[:, 0], dtype=np.float64),
        kind="stable",
    )[:count]
    selected_ids = np.asarray(low_update.candidate_ids[order], dtype=np.int64)
    low_rows = np.asarray(
        [
            int(np.flatnonzero(low_request.candidate_ids == value)[0])
            for value in selected_ids
        ],
        dtype=np.intp,
    )
    request_id = np.int64(ctx.request_id_allocator.allocate(1)[0])
    high_request = EvaluationRequest(
        request_id,
        selected_ids,
        low_request.payload.take(low_rows),
        low_request.outputs,
        {
            **low_request.metadata,
            "fidelity": int(continuation["next_fidelity"]),
            "promotion_of": int(low_request.request_id),
        },
    )
    return EvaluationPlan(
        (low_request, high_request),
        completion_rule="all_requests_completed",
        continuation={"kind": "fidelity_promotion_complete"},
        artifacts={
            **dict(plan.artifacts),
            "promoted_candidate_ids": selected_ids.tolist(),
            "promoted_request_id": int(high_request.request_id),
        },
    )


@register()
class FidelityEvaluation(_RequestPlanner):
    """Attach an explicit fidelity level to an evaluation request."""

    def __init__(self, fidelity: int = 0):
        if not isinstance(fidelity, int) or isinstance(fidelity, bool) or fidelity < 0:
            raise ValidationError("fidelity must be a non-negative integer")
        self.fidelity = fidelity

    def plan(self, candidates, acquisition, *, ctx: OptimizationState | None):
        """Build one request and attach its fidelity level."""
        request = EvaluateAll().plan(candidates, acquisition, ctx=ctx).requests[0]
        metadata = dict(request.metadata)
        metadata["fidelity"] = self.fidelity
        return EvaluationPlan(
            (
                EvaluationRequest(
                    request.request_id,
                    request.candidate_ids,
                    request.payload,
                    request.outputs,
                    metadata,
                ),
            )
        )


@register()
class FidelityPromotion(FidelityEvaluation):
    """Represent an explicit promotion from one fidelity level to another."""

    def __init__(
        self,
        fidelity: int = 0,
        next_fidelity: int | None = None,
        *,
        promotion_count: int | None = None,
        promotion_fraction: float = 0.5,
    ):
        super().__init__(fidelity)
        if next_fidelity is not None and (
            not isinstance(next_fidelity, int)
            or isinstance(next_fidelity, bool)
            or next_fidelity <= fidelity
        ):
            raise ValidationError("next_fidelity must exceed fidelity")
        if promotion_count is not None and (
            not isinstance(promotion_count, int)
            or isinstance(promotion_count, bool)
            or promotion_count < 1
        ):
            raise ValidationError("promotion_count must be positive")
        if not np.isfinite(promotion_fraction) or not 0.0 < promotion_fraction <= 1.0:
            raise ValidationError("promotion_fraction must be in (0, 1]")
        self.next_fidelity = next_fidelity
        self.promotion_count = promotion_count
        self.promotion_fraction = promotion_fraction

    def plan(self, candidates, acquisition, *, ctx: OptimizationState | None):
        """Build a low-fidelity request with a standard continuation marker."""
        plan = super().plan(candidates, acquisition, ctx=ctx)
        if self.next_fidelity is None:
            return plan
        return EvaluationPlan(
            plan.requests,
            completion_rule="fidelity_promotion",
            continuation={
                "kind": "fidelity_promotion",
                "next_fidelity": self.next_fidelity,
                "promotion_count": self.promotion_count,
                "promotion_fraction": self.promotion_fraction,
            },
            artifacts={"fidelity": self.fidelity},
        )
