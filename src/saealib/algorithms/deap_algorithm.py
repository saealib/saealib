"""Adapter exposing a DEAP generate/update strategy as an Algorithm."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import Any, Protocol

import numpy as np

from saealib.algorithms.base import Algorithm, ProposalRequest, algorithm_context
from saealib.callback import PostAskEvent
from saealib.core.contracts import (
    AssumptionSet,
    ComponentContract,
    FeedbackBatch,
    FeedbackRequirement,
    ProposalBatch,
    ProposalRelations,
)
from saealib.core.state import POPULATIONS_MAIN, StatePatch, StateView
from saealib.exceptions import ConfigurationError
from saealib.operators._deap_rng import seeded_global_numpy_random
from saealib.population import PopulationAttribute
from saealib.problem import Problem


class _DeapGenerateUpdateLike(Protocol):
    def generate(
        self, ind_init: Callable[[np.ndarray], _DeapIndividual]
    ) -> list[_DeapIndividual]:
        """Generate individuals using ``ind_init``."""

    def update(self, population: list[_DeapIndividual]) -> None:
        """Update strategy state from evaluated individuals."""


class _DeapIndividual(np.ndarray):
    fitness: Any

    def __array_finalize__(self, obj: object) -> None:
        self.fitness = getattr(obj, "fitness", None)


class DeapGenerateUpdateAlgorithm(Algorithm):
    """Wrap a DEAP ``generate``/``update`` strategy.

    The wrapped strategy owns its evolving state; this adapter does not mirror
    a strategy population. Generated individuals are repaired before evaluation
    and the told offspring becomes saealib's main population.
    """

    def __init__(
        self,
        strategy: _DeapGenerateUpdateLike,
        *,
        allow_partial_tell: bool = False,
    ) -> None:
        try:
            import deap  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "DeapGenerateUpdateAlgorithm requires DEAP; install it with "
                "pip install 'saealib[deap]'"
            ) from exc
        self.strategy = strategy
        self.allow_partial_tell = allow_partial_tell
        self._fitness_class: type | None = None
        self._last_individuals: list[_DeapIndividual] | None = None

    def contract(self) -> ComponentContract:
        """Return the contract, rejecting portable checkpoints.

        The DEAP strategy owns evolving centroid, sigma, and related state
        that this adapter cannot export to a portable checkpoint.
        """
        return replace(
            super().contract(),
            assumptions=AssumptionSet({"state.checkpointable": False}),
        )

    def _ensure_setup(self, problem: Problem) -> None:
        if problem.n_constraints:
            raise ConfigurationError(
                "DeapGenerateUpdateAlgorithm supports unconstrained problems only"
            )
        if problem.n_obj != 1:
            raise ConfigurationError(
                "DeapGenerateUpdateAlgorithm supports single-objective problems only"
            )
        if self._fitness_class is None:
            from deap import base

            self._fitness_class = type(
                f"_SaealibDeapFitness_{id(self)}",
                (base.Fitness,),
                {"weights": tuple(problem.direction)},
            )

    def _ind_init(self, row: np.ndarray) -> _DeapIndividual:
        assert self._fitness_class is not None
        individual = np.asarray(row, dtype=float).view(_DeapIndividual)
        individual.fitness = self._fitness_class()
        return individual

    def get_required_attrs(self, problem: Problem) -> list[PopulationAttribute]:
        """Return the ask-order tracking attribute required by the adapter."""
        self._ensure_setup(problem)
        return [PopulationAttribute("deap_idx", np.int64, (), default=-1)]

    @property
    def ask_notation(self) -> list[str]:
        """Describe strategy generation."""
        return [r"$\mathcal{Q} \leftarrow \text{strategy.generate(ind\_init)}$"]

    @property
    def tell_notation(self) -> list[str]:
        """Describe strategy update and population adoption."""
        return [r"$\text{strategy.update}(\mathcal{Q})$", r"$P \leftarrow \mathcal{Q}$"]

    def ask(self, request: ProposalRequest, state: StateView) -> ProposalBatch:
        """Generate, repair, and return the strategy's next individuals."""
        del request
        ctx = algorithm_context(state)
        self._ensure_setup(ctx.problem)
        with seeded_global_numpy_random(ctx.rng):
            individuals = list(self.strategy.generate(self._ind_init))
        self._last_individuals = individuals
        if not individuals:
            raise ConfigurationError("The DEAP strategy generated no individuals")
        try:
            x = np.stack([np.asarray(ind, dtype=float) for ind in individuals])
        except (TypeError, ValueError) as exc:
            raise ConfigurationError(
                "The DEAP strategy generated invalid individuals"
            ) from exc
        if x.ndim != 2 or x.shape[1] != ctx.problem.dim:
            raise ConfigurationError(
                f"The DEAP strategy generated shape {x.shape}; expected "
                f"(n, {ctx.problem.dim})"
            )
        for i in range(len(x)):
            x[i] = ctx.problem.handler.repair(
                x[i], ctx.problem.constraints, ctx.problem.lb, ctx.problem.ub
            )
        x = ctx.problem.repair(x)
        for individual, row in zip(individuals, x, strict=True):
            individual[:] = row
        state.dispatch(PostAskEvent(ctx=ctx, candidates=x))
        candidates = ctx.population.empty_like(capacity=len(x))
        candidates.extend({"x": x, "deap_idx": np.arange(len(x), dtype=np.int64)})
        return ProposalBatch.from_allocator(
            ctx.proposal_id_allocator,
            candidates=candidates,
            relations=ProposalRelations(row_count=len(candidates)),
            requirements=FeedbackRequirement(quantities=()),
        )

    def tell(self, feedback: FeedbackBatch, state: StateView) -> StatePatch:
        """Set fitness values, update the strategy, and adopt told offspring."""
        del feedback
        ctx = algorithm_context(state)
        offspring = ctx.offspring
        if offspring is None:
            raise ConfigurationError(
                "DeapGenerateUpdateAlgorithm.tell() requires an offspring population"
            )
        if self._last_individuals is None:
            raise ConfigurationError(
                "DeapGenerateUpdateAlgorithm.tell() called before ask()"
            )
        idx = offspring.get_array("deap_idx").astype(np.int64, copy=False)
        if (
            idx.ndim != 1
            or np.any(idx < 0)
            or np.any(idx >= len(self._last_individuals))
        ):
            raise ConfigurationError("deap_idx is outside the generated population")
        if len(np.unique(idx)) != len(idx):
            raise ConfigurationError("deap_idx contains duplicate positions")
        if len(idx) != len(self._last_individuals) and not self.allow_partial_tell:
            raise ConfigurationError(
                f"DeapGenerateUpdateAlgorithm.tell() received {len(idx)} of "
                f"{len(self._last_individuals)} generated candidates"
            )
        f = offspring.get_array("f")
        told: list[_DeapIndividual] = []
        for row, position in enumerate(idx):
            individual = self._last_individuals[int(position)]
            individual.fitness.values = tuple(float(value) for value in f[row])
            told.append(individual)
        self.strategy.update(told)
        ctx.population.clear()
        ctx.population._extend_internal(offspring, preserve_ids=True)
        return StatePatch(writes={POPULATIONS_MAIN: ctx.population})
