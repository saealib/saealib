"""Integration regressions for CORS beta cadence.

These tests verify the source-faithful CORS configuration:
- 1 decision = 1 true evaluation
- beta is derived from the runtime decision count
- search pattern cycles correctly
- delta=None approximation works
- checkpoint restore preserves the beta phase
"""

from __future__ import annotations

from itertools import pairwise
from typing import Any

import numpy as np

from saealib import (
    GA,
    CrossoverBLXAlpha,
    GaussianKernel,
    LHSInitializer,
    MutationUniform,
    Optimizer,
    RBFSurrogate,
    SequentialSelection,
    Termination,
    TruncationSelection,
    max_gen,
)
from saealib.acquisition.mean import CORSDistance
from saealib.callback import AcquisitionEndEvent
from saealib.comparators import SingleObjectiveComparator
from saealib.context import OptimizationState
from saealib.policies.evaluation import TopKEvaluation
from saealib.problem import Problem
from saealib.strategies.ps import PreSelectionStrategy

SEARCH_PATTERN = (0.9, 0.4, 0.0)
DIM = 2


class _RecordingCORSDistance(CORSDistance):
    """Expose the beta resolved by the real CORS implementation."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.records: list[float] = []

    def prepare(self, archive, *, ctx):
        reference = super().prepare(archive, ctx=ctx)
        self.records.append(reference.beta)
        return reference


def _make_problem() -> Problem:
    return Problem(
        func=lambda x: np.array([np.sum(x**2)]),
        dim=DIM,
        n_obj=1,
        direction=np.array([-1.0]),
        lb=[-5.0] * DIM,
        ub=[5.0] * DIM,
        comparator=SingleObjectiveComparator(),
    )


def _make_canonical_optimizer(
    problem: Problem, acquisition: CORSDistance, n_gen: int
) -> Optimizer:
    """Build a canonical CORS optimizer: n_select=1, TopKEvaluation(1)."""
    return (
        Optimizer(problem, seed=7)
        .set_initializer(LHSInitializer(6, 4))
        .set_algorithm(
            GA(
                crossover=CrossoverBLXAlpha(prob=0.9, alpha=0.5),
                mutation=MutationUniform(prob_var=0.1),
                parent_selection=SequentialSelection(),
                survivor_selection=TruncationSelection(),
            )
        )
        .set_surrogate(RBFSurrogate(GaussianKernel()), n_neighbors=5)
        .set_acquisition(acquisition)
        .set_strategy(PreSelectionStrategy(n_candidates=4, n_select=1))
        .set_evaluation_planner(TopKEvaluation(1, sanitize_nonfinite=True))
        .set_termination(Termination(max_gen(n_gen)))
    )


def test_canonical_cors_evaluates_one_candidate_per_decision():
    """Canonical CORS: 1 decision = 1 FE, beta advances correctly."""
    acquisition = _RecordingCORSDistance(delta=1.0, search_pattern=SEARCH_PATTERN)
    optimizer = _make_canonical_optimizer(_make_problem(), acquisition, n_gen=3)
    score_calls: list[int] = []
    optimizer.cbmanager.register(
        AcquisitionEndEvent,
        lambda event: score_calls.append(int(event.ctx.gen)),
    )

    states = list(optimizer.iterate())

    assert [(state.gen, state.decision_count) for state in states] == [
        (0, 0),
        (1, 1),
        (2, 2),
        (3, 3),
    ]
    assert score_calls == [1, 2, 3]
    assert len(acquisition.records) == len(score_calls)
    np.testing.assert_allclose(acquisition.records, SEARCH_PATTERN)
    assert [current.fe - previous.fe for previous, current in pairwise(states)] == [
        1,
        1,
        1,
    ]


def test_canonical_cors_with_delta_none():
    """Canonical CORS with delta=None: candidate-pool maximin approximation."""
    acquisition = _RecordingCORSDistance(search_pattern=SEARCH_PATTERN)
    optimizer = _make_canonical_optimizer(_make_problem(), acquisition, n_gen=3)

    states = list(optimizer.iterate())

    assert [(state.gen, state.decision_count) for state in states] == [
        (0, 0),
        (1, 1),
        (2, 2),
        (3, 3),
    ]
    np.testing.assert_allclose(acquisition.records, SEARCH_PATTERN)
    assert [current.fe - previous.fe for previous, current in pairwise(states)] == [
        1,
        1,
        1,
    ]


def test_cors_search_pattern_cycles():
    """Search pattern repeats cyclically across multiple cycles."""
    short_pattern = (0.8, 0.0)
    acquisition = _RecordingCORSDistance(delta=1.0, search_pattern=short_pattern)
    optimizer = _make_canonical_optimizer(_make_problem(), acquisition, n_gen=5)

    states = list(optimizer.iterate())

    expected_betas = [0.8, 0.0, 0.8, 0.0, 0.8]
    np.testing.assert_allclose(acquisition.records, expected_betas)
    assert states[-1].decision_count == 5


def test_cors_beta_phase_survives_checkpoint_restore(tmp_path):
    """The persisted decision count selects the same beta after restore."""
    acquisition = _RecordingCORSDistance(delta=1.0, search_pattern=SEARCH_PATTERN)
    optimizer = _make_canonical_optimizer(_make_problem(), acquisition, n_gen=3)
    states = list(optimizer.iterate())
    saved = states[2]
    checkpoint = tmp_path / "cors.npz"

    saved.save(checkpoint)
    restored = OptimizationState.load(checkpoint, _make_problem())
    resumed_acquisition = CORSDistance(delta=1.0, search_pattern=SEARCH_PATTERN)

    assert restored.decision_count == saved.decision_count == 2
    assert (
        resumed_acquisition.prepare(restored.archive, ctx=restored).beta
        == SEARCH_PATTERN[2]
    )
