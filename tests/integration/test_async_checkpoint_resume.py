"""Resuming an asynchronous run from an NPZ checkpoint."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from saealib import (
    GA,
    PSO,
    CrossoverBLXAlpha,
    LHSInitializer,
    MutationUniform,
    Optimizer,
    SequentialSelection,
    Termination,
    TruncationSelection,
)
from saealib.acquisition import MeanPrediction
from saealib.context import OptimizationState
from saealib.execution import (
    AsyncEvaluationScheduler,
    EvaluationHandle,
    EvaluationRequest,
    EvaluationStatus,
    EvaluationUpdate,
    Evaluator,
    SerialEvaluator,
)
from saealib.problem import InequalityConstraint, Problem
from saealib.strategies import DirectStrategy, GenerationBasedStrategy
from saealib.surrogate.manager import LocalSurrogateManager
from saealib.surrogate.rbf import RBFSurrogate
from saealib.surrogate.rbf_kernels import GaussianKernel


class _FifoEvaluator(Evaluator):
    """Deliver each request once, in submission order, on its first collect."""

    def __init__(self) -> None:
        self._delivered: set[int] = set()
        self.collect_calls = 0

    def evaluate_batch(self, x: Any, problem: Any) -> Any:
        return SerialEvaluator().evaluate_batch(x, problem)

    def submit(self, request: EvaluationRequest, problem: Any) -> EvaluationHandle:
        return EvaluationHandle(
            request.request_id,
            EvaluationStatus.PENDING,
            backend_token=(request, problem),
        )

    def collect(self, handle: EvaluationHandle, *, wait: bool = True) -> list:
        self.collect_calls += 1
        if self.collect_calls > 5000:
            raise RuntimeError("collect cap exceeded")
        request, problem = handle.backend_token
        request_id = int(request.request_id)
        if handle._acknowledged_sequence >= 0 or request_id in self._delivered:
            return []
        self._delivered.add(request_id)
        result = SerialEvaluator().evaluate_request(request, problem)
        result.candidate_ids = request.candidate_ids
        result.__post_init__()
        handle._delivered_sequence = 0
        return [
            EvaluationUpdate(
                request.request_id,
                EvaluationStatus.COMPLETED,
                request.candidate_ids,
                result,
                None,
                0,
            )
        ]

    def acknowledge(self, handle: EvaluationHandle, sequence: int) -> None:
        handle._acknowledged_sequence = sequence

    def supports_batch_rollback(self) -> bool:
        return True

    def can_reattach(self, pending: Any) -> bool:
        return True

    def reattach(self, pending: Any, problem: Any) -> EvaluationHandle:
        return EvaluationHandle(
            pending.request.request_id,
            EvaluationStatus.PENDING,
            backend_token=(pending.request, problem),
        )


def _distance_from_axis(x: Any) -> float:
    return 1.0 - abs(float(x[0]))


def _problem() -> Problem:
    return Problem(
        func=lambda x: np.array([float(np.sum(np.asarray(x) ** 2))], dtype=np.float64),
        dim=2,
        n_obj=1,
        direction=np.array([-1.0]),
        lb=[-5.0] * 2,
        ub=[5.0] * 2,
        constraints=[InequalityConstraint(_distance_from_axis)],
    )


def _algorithm(name: str) -> Any:
    if name == "pso":
        return PSO()
    return GA(
        crossover=CrossoverBLXAlpha(prob=0.9, alpha=0.4),
        mutation=MutationUniform(prob_var=0.1),
        parent_selection=SequentialSelection(),
        survivor_selection=TruncationSelection(),
    )


def _optimizer(
    strategy: str, algorithm: str, generations: int, max_pending: int
) -> Optimizer:
    evaluator = _FifoEvaluator()
    optimizer = (
        Optimizer(_problem(), seed=7)
        .set_initializer(LHSInitializer(6, 4, 7))
        .set_algorithm(_algorithm(algorithm))
        .set_termination(Termination(lambda state: state.gen >= generations))
        .set_evaluator(evaluator)
    )
    if strategy == "generation_based":
        optimizer.set_strategy(GenerationBasedStrategy(0))
        optimizer.set_surrogate_manager(
            LocalSurrogateManager(RBFSurrogate(GaussianKernel()))
        ).set_acquisition(MeanPrediction())
    else:
        optimizer.set_strategy(DirectStrategy(n_offspring=3))
    optimizer.set_async_evaluation_scheduler(
        AsyncEvaluationScheduler(evaluator, max_pending=max_pending)
    )
    return optimizer


def _last_generation_boundary(optimizer: Optimizer) -> OptimizationState:
    """Return the state at the last generation boundary of a completed run."""
    state = None
    for state in optimizer.iterate():
        pass
    if state is None:
        raise AssertionError("optimizer produced no generation boundary")
    return state


@pytest.mark.parametrize(
    ("strategy", "algorithm", "max_pending"),
    [
        ("generation_based", "ga", 2),
        ("generation_based", "ga", 1),
        ("generation_based", "pso", 1),
        ("direct", "ga", 2),
        ("direct", "ga", 1),
        ("direct", "pso", 2),
    ],
)
def test_async_npz_checkpoint_at_a_generation_boundary_resumes_like_uninterrupted_run(
    tmp_path, strategy: str, algorithm: str, max_pending: int
) -> None:
    path = tmp_path / "checkpoint.npz"
    _last_generation_boundary(_optimizer(strategy, algorithm, 1, max_pending)).save(
        path
    )
    restored = OptimizationState.load(path, _problem())

    resumed = _optimizer(strategy, algorithm, 3, max_pending).run_from(restored)
    uninterrupted = _optimizer(strategy, algorithm, 3, max_pending).run()

    assert (resumed.gen, resumed.fe) == (uninterrupted.gen, uninterrupted.fe)
    assert (
        resumed.population.get_array("id").tolist()
        == uninterrupted.population.get_array("id").tolist()
    )


def _objective_sources(population: Any) -> list[int]:
    return population.get_array("objective_source").tolist()


@pytest.mark.parametrize(
    ("strategy", "algorithm", "in_flight"),
    [("generation_based", "ga", True), ("direct", "pso", False)],
)
def test_async_npz_checkpoint_resume_keeps_objective_source(
    tmp_path, strategy: str, algorithm: str, in_flight: bool
) -> None:
    path = tmp_path / "checkpoint.npz"
    saved = _last_generation_boundary(_optimizer(strategy, algorithm, 1, 2))
    saved.save(path)
    restored = OptimizationState.load(path, _problem())

    assert bool(saved.pending_evaluations) is in_flight
    assert _objective_sources(restored.population) == _objective_sources(
        saved.population
    )
    for request_id, owner in saved.evaluation_owners.items():
        assert _objective_sources(
            restored.evaluation_owners[request_id]
        ) == _objective_sources(owner)

    resumed = _optimizer(strategy, algorithm, 3, 2).run_from(restored)
    uninterrupted = _optimizer(strategy, algorithm, 3, 2).run()

    assert _objective_sources(resumed.population) == _objective_sources(
        uninterrupted.population
    )
    finite = np.isfinite(resumed.population.get_array("f")).all(axis=1)
    sources = np.asarray(_objective_sources(resumed.population))
    assert np.all(sources[finite] == 0)
    assert np.all(sources[~finite] == 255)
