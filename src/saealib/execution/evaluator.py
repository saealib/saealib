"""
Evaluator abstraction.

An ``Evaluator`` is the single entry point through which Strategies and
Initializers turn a batch of design vectors into objective values, raw
constraint values, and aggregate constraint violations. Centralizing
evaluation here enables pluggable execution backends (serial, parallel, ...)
without touching the pipeline code.
"""

from __future__ import annotations

import pickle
import time
from abc import ABC, abstractmethod
from collections.abc import Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Protocol, cast

import numpy as np

from saealib.core.contracts import (
    MANY,
    ComponentContract,
    DataSpec,
    PortContract,
    PortDirection,
    PortSpec,
    Var,
)
from saealib.exceptions import EvaluationProtocolError, ValidationError
from saealib.population.genome import DenseVectorBatch, GenomeBatch, genome_value

if TYPE_CHECKING:
    from saealib.problem import Problem


class EvaluationStatus(str, Enum):
    """State of a submitted evaluation."""

    PENDING = "pending"
    RUNNING = "running"
    PARTIAL = "partial"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class EvaluationQuery:
    """Request context supplied to an :class:`EvaluationAdapter`."""

    request_id: np.int64
    candidate_ids: np.ndarray
    metadata: Mapping[str, Any] = field(default_factory=dict)


class EvaluationAdapter(Protocol):
    """Transform genomes into the payload consumed by a problem evaluation."""

    def transform(self, genomes: GenomeBatch, request: EvaluationQuery) -> Any:
        """Return an evaluation payload batch for *genomes*."""
        ...


def _owned_array(value: Any, *, dtype: np.dtype, ndim: int, name: str) -> np.ndarray:
    arr = np.asarray(value)
    if arr.dtype != dtype or arr.ndim != ndim:
        raise ValidationError(f"{name} must have dtype {dtype} and ndim {ndim}")
    result = np.array(arr, dtype=dtype, order="C", copy=True)
    result.flags.writeable = False
    return result


def _payload_value(payload: Any, index: int) -> object:
    if isinstance(payload, GenomeBatch):
        return genome_value(payload, index)
    return payload[index]


@dataclass
class EvaluationResult:
    """Validated numeric result for a batch of candidates."""

    f: np.ndarray
    g: np.ndarray
    cv: np.ndarray
    candidate_ids: np.ndarray | None = None
    cost: np.ndarray | None = None
    noise: np.ndarray | None = None
    outputs: dict[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.f = _owned_array(self.f, dtype=np.dtype(np.float64), ndim=2, name="f")
        self.g = _owned_array(self.g, dtype=np.dtype(np.float64), ndim=2, name="g")
        self.cv = _owned_array(self.cv, dtype=np.dtype(np.float64), ndim=1, name="cv")
        n = len(self.f)
        if len(self.g) != n or len(self.cv) != n:
            raise ValidationError("EvaluationResult channel lengths must match")
        if self.candidate_ids is not None:
            self.candidate_ids = _owned_array(
                self.candidate_ids,
                dtype=np.dtype(np.int64),
                ndim=1,
                name="candidate_ids",
            )
            if len(self.candidate_ids) != n:
                raise ValidationError("candidate_ids length must match result rows")
            if len(np.unique(self.candidate_ids)) != n:
                raise ValidationError("candidate_ids must be unique")
        if self.cost is not None:
            self.cost = _owned_array(
                self.cost, dtype=np.dtype(np.float64), ndim=1, name="cost"
            )
            if len(self.cost) != n:
                raise ValidationError("cost length must match result rows")
        if self.noise is not None:
            self.noise = _owned_array(
                self.noise, dtype=np.dtype(np.float64), ndim=2, name="noise"
            )
            if self.noise.shape[0] != n or self.noise.shape[1] != self.f.shape[1]:
                raise ValidationError("noise must have shape (n, n_obj)")
        normalized: dict[str, np.ndarray] = {}
        for name, value in self.outputs.items():
            arr = np.asarray(value)
            if arr.dtype != np.float64 or arr.ndim == 0 or arr.shape[0] != n:
                raise ValidationError(
                    f"outputs[{name!r}] must be float64 with leading size {n}"
                )
            arr = np.array(arr, dtype=np.float64, order="C", copy=True)
            arr.flags.writeable = False
            normalized[name] = arr
        pickle.dumps(normalized)
        self.outputs = normalized


@dataclass(frozen=True)
class EvaluationErrorInfo:
    """Serializable error information for a failed evaluation."""

    error_type: str
    message: str
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        pickle.dumps(dict(self.details))


@dataclass(frozen=True, init=False)
class EvaluationRequest:
    """Input payload whose snapshot semantics depend on how it was built.

    Passing an existing ``GenomeBatch`` directly, such as ``pop.genomes``,
    does not guarantee that the request owns an immutable snapshot of the
    input: a Population-backed batch may remain a view of live storage.  The
    normal planner path instead calls ``genomes.take(indices)`` to construct
    an independent batch value, which can be treated as the request's input
    snapshot.
    """

    request_id: np.int64
    candidate_ids: np.ndarray
    payload: GenomeBatch
    outputs: tuple[str, ...] = ("f", "g", "cv")
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __init__(
        self,
        request_id: np.int64,
        candidate_ids: np.ndarray,
        payload: GenomeBatch | np.ndarray | None = None,
        outputs: tuple[str, ...] = ("f", "g", "cv"),
        metadata: Mapping[str, Any] | None = None,
        *,
        x: np.ndarray | None = None,
    ) -> None:
        if np.asarray(request_id).shape != ():
            raise ValidationError("request_id must be scalar")
        if x is not None:
            if payload is not None:
                raise ValidationError("EvaluationRequest accepts either payload or x")
            payload = x
        if payload is None:
            raise ValidationError("EvaluationRequest requires a payload")
        if isinstance(payload, np.ndarray):
            payload = DenseVectorBatch(
                _owned_array(
                    payload,
                    dtype=np.dtype(np.float64),
                    ndim=2,
                    name="x",
                )
            )
        if not isinstance(payload, GenomeBatch):
            raise ValidationError(
                "EvaluationRequest payload must implement the GenomeBatch protocol"
            )
        object.__setattr__(self, "request_id", np.int64(request_id))
        ids = _owned_array(
            candidate_ids, dtype=np.dtype(np.int64), ndim=1, name="candidate_ids"
        )
        if len(ids) != len(payload) or len(ids) != len(np.unique(ids)):
            raise ValidationError(
                "request candidate_ids must be unique and match payload"
            )
        object.__setattr__(self, "candidate_ids", ids)
        object.__setattr__(self, "payload", payload)
        object.__setattr__(self, "outputs", tuple(outputs))
        metadata = {} if metadata is None else dict(metadata)
        pickle.dumps(metadata)
        object.__setattr__(self, "metadata", metadata)

    @property
    def x(self) -> np.ndarray:
        """Return the dense compatibility view when the payload is numeric."""
        if isinstance(self.payload, DenseVectorBatch):
            return self.payload.array
        raise ValidationError(
            "EvaluationRequest.x requires a DenseNumericView-compatible "
            "DenseVectorBatch payload; this request carries a non-dense "
            f"{type(self.payload).__name__} payload"
        )


@dataclass
class EvaluationHandle:
    """Runtime reference to submitted evaluation work."""

    request_id: np.int64
    status: EvaluationStatus
    backend_token: Any = None
    _sync_result: EvaluationResult | None = None
    _sync_error: EvaluationErrorInfo | None = None
    _delivered_sequence: int = -1
    _acknowledged_sequence: int = -1
    _submitted_at: float = field(default_factory=time.monotonic, repr=False)
    _candidate_ids: np.ndarray | None = field(default=None, repr=False)
    _detached: bool = field(default=False, repr=False)
    _completed_at: float | None = field(default=None, repr=False)
    _completion_key: int | None = field(default=None, repr=False)


@dataclass(frozen=True)
class EvaluationUpdate:
    """One delivered evaluation update."""

    request_id: np.int64
    status: EvaluationStatus
    candidate_ids: np.ndarray
    result: EvaluationResult | None = None
    error: EvaluationErrorInfo | None = None
    sequence: int = 0

    def __post_init__(self) -> None:
        if self.sequence < 0:
            raise EvaluationProtocolError("update sequence must be non-negative")
        ids = _owned_array(
            self.candidate_ids, dtype=np.dtype(np.int64), ndim=1, name="candidate_ids"
        )
        if len(ids) != len(np.unique(ids)):
            raise EvaluationProtocolError("update candidate_ids must be unique")
        object.__setattr__(self, "candidate_ids", ids)
        if self.result is not None and self.result.candidate_ids is None:
            raise EvaluationProtocolError("lifecycle result requires candidate_ids")


@dataclass(frozen=True)
class PendingEvaluation:
    """Serializable record for submitted work."""

    request: EvaluationRequest
    status: EvaluationStatus
    applied_candidate_ids: np.ndarray
    last_delivered_sequence: int = -1
    last_acknowledged_sequence: int = -1
    processing: Mapping[int, str] = field(default_factory=dict)
    buffered_updates: tuple[EvaluationUpdate, ...] = ()
    reserved_cost: float = 0.0
    retry_count: int = 0
    checkpointable: bool = False
    original_candidate_ids: np.ndarray | None = None
    feedback_result: Any = None
    fatal_error: EvaluationErrorInfo | None = None
    prediction: Any = None

    def __post_init__(self) -> None:
        ids = _owned_array(
            self.applied_candidate_ids,
            dtype=np.dtype(np.int64),
            ndim=1,
            name="applied_candidate_ids",
        )
        object.__setattr__(self, "applied_candidate_ids", ids)
        object.__setattr__(self, "processing", dict(self.processing))
        original = (
            self.request.candidate_ids
            if self.original_candidate_ids is None
            else self.original_candidate_ids
        )
        original = _owned_array(
            original,
            dtype=np.dtype(np.int64),
            ndim=1,
            name="original_candidate_ids",
        )
        if len(original) != len(np.unique(original)):
            raise ValidationError("original_candidate_ids must be unique")
        object.__setattr__(self, "original_candidate_ids", original)
        if self.reserved_cost < 0 or not np.isfinite(self.reserved_cost):
            raise ValidationError("reserved_cost must be finite and non-negative")
        if self.retry_count < 0:
            raise ValidationError("retry_count must be non-negative")


class Evaluator(ABC):
    """Base class for batch evaluators."""

    def contract(self) -> ComponentContract:
        """Return the evaluator family contract."""
        return ComponentContract(
            ports={
                "evaluator": PortContract(
                    inputs=(
                        PortSpec(
                            name="genomes",
                            direction=PortDirection.INPUT,
                            data=DataSpec(
                                kind="GenomeBatch",
                                bindings={"representation": Var(name="R")},
                            ),
                            cardinality=MANY,
                        ),
                    ),
                    outputs=(
                        PortSpec(
                            name="observations",
                            direction=PortDirection.OUTPUT,
                            data=DataSpec(
                                kind="ObservationBatch",
                                bindings={
                                    "objective_schema": Var(name="O"),
                                    "constraint_schema": Var(name="C"),
                                },
                            ),
                            cardinality=MANY,
                        ),
                    ),
                )
            }
        )

    @abstractmethod
    def evaluate_batch(
        self, x: GenomeBatch | np.ndarray, problem: Problem
    ) -> EvaluationResult:
        """
        Evaluate a batch of design vectors.

        Parameters
        ----------
        x : np.ndarray
            Design vectors to evaluate. shape = (n, dim)
        problem : Problem
            The optimization problem providing the objective and constraints.

        Returns
        -------
        EvaluationResult
            Batched objective values, raw constraint values, and violations.
        """
        ...

    def submit(self, request: EvaluationRequest, problem: Problem) -> EvaluationHandle:
        """Submit a request through the synchronous adapter."""
        try:
            result = self.evaluate_request(request, problem)
        except Exception as exc:
            error = EvaluationErrorInfo(type(exc).__name__, str(exc))
            return EvaluationHandle(
                request.request_id, EvaluationStatus.FAILED, _sync_error=error
            )
        if result.candidate_ids is None:
            result.candidate_ids = request.candidate_ids
            result.__post_init__()
        elif not np.array_equal(result.candidate_ids, request.candidate_ids):
            raise EvaluationProtocolError("result candidate_ids do not match request")
        return EvaluationHandle(
            request.request_id, EvaluationStatus.COMPLETED, _sync_result=result
        )

    def evaluate_request(
        self, request: EvaluationRequest, problem: Problem
    ) -> EvaluationResult:
        """Evaluate a request, including its public metadata boundary."""
        return self.evaluate_batch(request.payload, problem)

    @classmethod
    def has_partial_lifecycle_override(cls) -> bool:
        """Return whether lifecycle adapter methods are only partly overridden."""
        methods = ("submit", "collect", "acknowledge")
        overridden = [
            getattr(cls, name) is not getattr(Evaluator, name) for name in methods
        ]
        return any(overridden) and not all(overridden)

    def supports_batch_rollback(self) -> bool:
        """Return whether failed batch submission can be rolled back."""
        return False

    def collect(
        self, handle: EvaluationHandle, *, wait: bool = True
    ) -> list[EvaluationUpdate]:
        """Return unacknowledged updates for a handle."""
        if handle._detached:
            return []
        if handle._sync_result is None and handle._sync_error is None:
            return []
        if handle._acknowledged_sequence >= 0:
            return []
        ids = (
            handle._sync_result.candidate_ids
            if handle._sync_result is not None
            and handle._sync_result.candidate_ids is not None
            else np.empty(0, dtype=np.int64)
        )
        update = EvaluationUpdate(
            request_id=handle.request_id,
            status=handle.status,
            candidate_ids=ids,
            result=handle._sync_result,
            error=handle._sync_error,
            sequence=0,
        )
        handle._delivered_sequence = 0
        return [update]

    def acknowledge(self, handle: EvaluationHandle, sequence: int) -> None:
        """Acknowledge one contiguous update sequence."""
        if (
            sequence != handle._acknowledged_sequence + 1
            or sequence > handle._delivered_sequence
        ):
            raise EvaluationProtocolError("evaluation sequences must be contiguous")
        handle._acknowledged_sequence = sequence

    def cancel(self, handle: EvaluationHandle) -> bool:
        """Cancel submitted work when the backend supports cancellation."""
        return False

    def detach(self, handle: EvaluationHandle) -> bool:
        """Detach work whose result must no longer be delivered."""
        return False

    def can_reattach(self, pending: PendingEvaluation) -> bool:
        """Return whether pending work can be restored after checkpoint load."""
        return False

    def reattach(
        self, pending: PendingEvaluation, problem: Problem
    ) -> EvaluationHandle:
        """Restore a pending request or raise when unsupported."""
        raise EvaluationProtocolError(
            f"evaluator cannot reattach request {int(pending.request.request_id)}"
        )


class AsyncEvaluator(Evaluator):
    """Thread-backed asynchronous evaluator."""

    def __init__(
        self,
        evaluator: Evaluator | None = None,
        *,
        max_workers: int = 1,
    ) -> None:
        if max_workers < 1:
            raise ValidationError("max_workers must be positive")
        self._evaluator = evaluator
        self._executor = ThreadPoolExecutor(max_workers=max_workers)

    def evaluate_batch(
        self, x: GenomeBatch | np.ndarray, problem: Problem
    ) -> EvaluationResult:
        """Evaluate one batch on the wrapped evaluator."""
        if self._evaluator is None:
            raise EvaluationProtocolError("AsyncEvaluator requires a batch evaluator")
        return self._evaluator.evaluate_batch(x, problem)

    def supports_batch_rollback(self) -> bool:
        """Return whether submitted futures can be detached."""
        return True

    def submit(self, request: EvaluationRequest, problem: Problem) -> EvaluationHandle:
        """Submit work without blocking the caller."""

        def run() -> EvaluationResult:
            if self._evaluator is not None:
                return self._evaluator.evaluate_request(request, problem)
            return self.evaluate_request(request, problem)

        future = self._executor.submit(run)
        handle = EvaluationHandle(
            request.request_id,
            EvaluationStatus.PENDING,
            backend_token=future,
            _candidate_ids=request.candidate_ids,
        )
        future.add_done_callback(
            lambda _: setattr(handle, "_completed_at", time.monotonic())
        )
        return handle

    def collect(
        self, handle: EvaluationHandle, *, wait: bool = True
    ) -> list[EvaluationUpdate]:
        """Collect one completed update, optionally without waiting."""
        future = handle.backend_token
        if not isinstance(future, Future):
            raise EvaluationProtocolError("async handle has no Future backend token")
        if not wait and not future.done():
            return []
        if handle._sync_result is None and handle._sync_error is None:
            try:
                result = future.result()
                if handle._completed_at is None:
                    handle._completed_at = time.monotonic()
                if result.candidate_ids is None:
                    result.candidate_ids = handle._candidate_ids
                    result.__post_init__()
                elif handle._candidate_ids is not None and not np.array_equal(
                    result.candidate_ids, handle._candidate_ids
                ):
                    raise EvaluationProtocolError("async result candidate_ids mismatch")
                handle._sync_result = result
                handle.status = EvaluationStatus.COMPLETED
            except Exception as exc:
                if handle._completed_at is None:
                    handle._completed_at = time.monotonic()
                handle._sync_error = EvaluationErrorInfo(type(exc).__name__, str(exc))
                handle.status = EvaluationStatus.FAILED
        if handle._acknowledged_sequence >= 0:
            return []
        ids = (
            handle._sync_result.candidate_ids
            if handle._sync_result is not None
            and handle._sync_result.candidate_ids is not None
            else np.empty(0, dtype=np.int64)
        )
        handle._delivered_sequence = 0
        return [
            EvaluationUpdate(
                handle.request_id,
                handle.status,
                ids,
                handle._sync_result,
                handle._sync_error,
                0,
            )
        ]

    def cancel(self, handle: EvaluationHandle) -> bool:
        """Cancel work that has not started."""
        future = handle.backend_token
        if not isinstance(future, Future):
            return False
        cancelled = future.cancel()
        if cancelled:
            handle.status = EvaluationStatus.CANCELLED
        return cancelled

    def detach(self, handle: EvaluationHandle) -> bool:
        """Detach a running future and discard its eventual result."""
        future = handle.backend_token
        if not isinstance(future, Future):
            return False
        handle._detached = True
        handle.status = EvaluationStatus.CANCELLED
        return True

    def acknowledge(self, handle: EvaluationHandle, sequence: int) -> None:
        """Acknowledge one asynchronous update."""
        super().acknowledge(handle, sequence)

    def close(self) -> None:
        """Release worker resources."""
        self._executor.shutdown(wait=True)


ThreadPoolEvaluator = AsyncEvaluator


class SerialEvaluator(Evaluator):
    """Default evaluator: evaluates each candidate sequentially."""

    def evaluate_batch(
        self, x: GenomeBatch | np.ndarray, problem: Problem
    ) -> EvaluationResult:
        """Evaluate a batch, using the problem's optional evaluation adapter."""
        genomes = (
            x
            if isinstance(x, GenomeBatch)
            else DenseVectorBatch(np.atleast_2d(np.asarray(x, dtype=float)))
        )
        query = EvaluationQuery(
            request_id=np.int64(0),
            candidate_ids=np.arange(len(genomes), dtype=np.int64),
        )
        return self._evaluate_batch(genomes, problem, query)

    def evaluate_request(
        self, request: EvaluationRequest, problem: Problem
    ) -> EvaluationResult:
        """Evaluate a request while exposing its context to the adapter."""
        query = EvaluationQuery(
            request_id=request.request_id,
            candidate_ids=request.candidate_ids,
            metadata=request.metadata,
        )
        return self._evaluate_batch(request.payload, problem, query)

    def _evaluate_batch(
        self,
        x: GenomeBatch,
        problem: Problem,
        request: EvaluationQuery,
    ) -> EvaluationResult:
        genomes = x
        payload: Any = genomes
        if problem.evaluation_adapter is not None:
            payload = problem.evaluation_adapter.transform(genomes, request)
        elif isinstance(genomes, DenseVectorBatch):
            payload = genomes.array
        n = len(genomes)
        n_constraints = problem.n_constraints

        f = np.empty((n, problem.n_obj), dtype=float)
        g = np.empty((n, n_constraints), dtype=float)
        cv = np.zeros(n, dtype=float)

        raw = problem.evaluate_batch(payload)
        if raw is not None:
            f_raw, g_raw = raw
            for i in range(n):
                xi = cast(np.ndarray, _payload_value(payload, i))
                cv[i] = problem._aggregate_cv(xi, g_raw[i])
                f[i] = problem.handler.augment_objective(
                    f_raw[i], problem.constraints, xi, g_raw[i]
                )
                g[i] = g_raw[i]
            return EvaluationResult(f=f, g=g, cv=cv)

        for i in range(n):
            xi = _payload_value(payload, i)
            g_i, cv_i = problem.evaluate_constraints(xi)
            f[i] = problem.evaluate(xi, g_i)
            g[i] = g_i
            cv[i] = cv_i

        return EvaluationResult(f=f, g=g, cv=cv)


class JoblibEvaluator(Evaluator):
    """
    Parallel evaluator backed by `joblib <https://joblib.readthedocs.io>`_.

    Candidates in each batch are evaluated in parallel using joblib's
    ``Parallel`` / ``delayed`` interface.  The default backend ``"loky"``
    uses cloudpickle, so problem functions defined as lambdas or closures
    are handled without extra serialisation work.

    Switching to Dask or Ray is a single-parameter change::

        JoblibEvaluator(n_jobs=-1, backend="dask")   # Dask cluster
        JoblibEvaluator(n_jobs=-1, backend="ray")    # Ray cluster

    Parameters
    ----------
    n_jobs : int
        Number of parallel workers.  ``-1`` uses all available CPU cores
        (joblib convention).  ``1`` disables parallelism (equivalent to
        :class:`SerialEvaluator` but with joblib overhead).
    backend : str
        joblib backend name.  ``"loky"`` (default) launches fresh worker
        processes with cloudpickle serialisation.  Other built-in options:
        ``"threading"``, ``"multiprocessing"``.  Third-party backends
        ``"dask"`` and ``"ray"`` require the corresponding packages and a
        running cluster.
    **joblib_kwargs
        Additional keyword arguments forwarded verbatim to
        ``joblib.Parallel``.  Useful for ``verbose``, ``prefer``,
        ``require``, ``timeout``, etc.

    Raises
    ------
    ImportError
        If joblib is not installed.

    Notes
    -----
    **Island-model nested parallelism**: when multiple islands each own a
    ``JoblibEvaluator``, CPU cores may be over-subscribed.  Set
    ``n_jobs=1`` per island evaluator and let the island-level parallelism
    control concurrency, or use ``joblib.parallel_backend`` as a context
    manager to limit inner workers.

    Async evaluation is out of scope for this class.  Asynchronous
    candidate dispatch would require changes to the ``Algorithm.ask`` /
    ``Algorithm.tell`` interface and is tracked separately.

    This evaluator does not use ``Problem.evaluate_batch`` even when a
    ``Problem`` overrides it; combining per-row parallel dispatch (this
    class) with a single vectorized batch call is tracked separately.
    """

    def __init__(
        self,
        n_jobs: int = -1,
        backend: str = "loky",
        **joblib_kwargs: object,
    ) -> None:
        try:
            import joblib as _joblib  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "JoblibEvaluator requires joblib. "
                "Install it with: pip install saealib[parallel]"
            ) from exc
        self._n_jobs = n_jobs
        self._backend = backend
        self._joblib_kwargs = joblib_kwargs

    @property
    def n_jobs(self) -> int:
        """Number of parallel workers (joblib convention; ``-1`` = all cores)."""
        return self._n_jobs

    @property
    def backend(self) -> str:
        """Joblib backend name."""
        return self._backend

    def evaluate_batch(
        self, x: GenomeBatch | np.ndarray, problem: Problem
    ) -> EvaluationResult:
        """
        Evaluate candidates in parallel using joblib.

        Parameters
        ----------
        x : np.ndarray
            Design vectors to evaluate.  shape = (n, dim)
        problem : Problem
            The optimization problem providing the objective and constraints.

        Returns
        -------
        EvaluationResult
            Batched objective values, raw constraint values, and violations.
        """
        from joblib import Parallel, delayed

        x = np.atleast_2d(np.asarray(x, dtype=float))
        n = len(x)

        def _eval_one(xi: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
            g_i, cv_i = problem.evaluate_constraints(xi)
            f_i = problem.evaluate(xi, g_i)
            return f_i, g_i, cv_i

        results = Parallel(
            n_jobs=self._n_jobs,
            backend=self._backend,
            **self._joblib_kwargs,
        )(delayed(_eval_one)(x[i]) for i in range(n))

        f = np.array([r[0] for r in results], dtype=float).reshape(n, problem.n_obj)
        g = np.array([r[1] for r in results], dtype=float).reshape(
            n, problem.n_constraints
        )
        cv = np.array([r[2] for r in results], dtype=float)

        return EvaluationResult(f=f, g=g, cv=cv)
