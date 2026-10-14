from typing import cast

import numpy as np
import pytest

from saealib.algorithms import GenomeGA
from saealib.context import OptimizationState
from saealib.core.contracts import FeedbackBatch
from saealib.exceptions import ConfigurationError
from saealib.operators import OrderCrossover, SwapMutation
from saealib.operators.selection import SequentialSelection, TruncationSelection
from saealib.population import Archive, ParetoArchive, Population, PopulationAttribute
from saealib.problem import Problem
from saealib.space import FingerprintService, PermutationSpace

DIM = 4
N_POP = 4

_ARCHIVE_ATTRS = [
    PopulationAttribute(name="f", dtype=np.float64, shape=(1,)),
    PopulationAttribute(name="cv", dtype=np.float64, shape=()),
]


def _make_problem() -> Problem:
    space = PermutationSpace(DIM)
    return Problem(
        func=lambda g: np.asarray([float(np.sum(g, dtype=float))]),
        dim=space.dim,
        n_obj=1,
        direction=np.array([-1.0]),
        space=space,
    )


def _make_genome_ga() -> GenomeGA:
    return GenomeGA(
        OrderCrossover(),
        SwapMutation(),
        SequentialSelection(),
        TruncationSelection(),
    )


def _make_ctx(rng_seed: int = 1) -> OptimizationState:
    problem = _make_problem()
    space = cast(PermutationSpace, problem.space)
    fingerprint = cast(FingerprintService, space.services.require("FingerprintService"))
    genomes = space.sample(N_POP, np.random.default_rng(rng_seed))
    pop = Population([], genomes=genomes, space=space)
    arc = Archive([], fingerprint_service=fingerprint)
    pareto_arc = ParetoArchive(
        _ARCHIVE_ATTRS,
        direction=np.array([-1.0]),
        key_attr="f",
        fingerprint_service=fingerprint,
    )
    return OptimizationState(
        problem=problem,
        population=pop,
        archive=arc,
        pareto_archive=pareto_arc,
        rng=np.random.default_rng(rng_seed + 1),
    )


def _view(ga: GenomeGA, ctx: OptimizationState):
    return ctx._store.view(
        ga.contract().state,
        context=ctx,
        dispatch=lambda event: None,
    )


class TestGenomeGATellGuards:
    def test_missing_offspring_raises_configuration_error(self):
        ga = _make_genome_ga()
        ctx = _make_ctx()
        with pytest.raises(ConfigurationError, match="offspring population"):
            ga.tell(cast(FeedbackBatch, object()), _view(ga, ctx))
