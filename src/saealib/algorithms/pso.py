"""Particle Swarm Optimization module."""

from __future__ import annotations

from dataclasses import replace
from typing import Protocol, cast

import numpy as np

from saealib.algorithms.base import (
    AskTellAlgorithm,
    ProposalRequest,
    algorithm_context,
)
from saealib.callback import PostAskEvent
from saealib.comparators import Comparator
from saealib.core.contracts import (
    ComponentContract,
    FeedbackBatch,
    FeedbackRequirement,
    ProposalBatch,
    ProposalRelations,
    ServiceRequirement,
)
from saealib.core.state import POPULATIONS_MAIN, StatePatch, StateView
from saealib.exceptions import ConfigurationError
from saealib.population import Archive, Population, PopulationAttribute
from saealib.problem import Problem
from saealib.registry import register
from saealib.space import BoundsService


class _LeaderContext(Protocol):
    """Capabilities required to construct and rank PSO personal bests."""

    @property
    def population(self) -> Population: ...

    @property
    def problem(self) -> Problem: ...

    @property
    def comparator(self) -> Comparator: ...

    @property
    def dim(self) -> int: ...


@register()
class PSO(AskTellAlgorithm):
    """
    Particle Swarm Optimization (PSO) for single-objective problems.

    The global best (leader) is selected from the personal bests of all
    particles using ``ctx.comparator``, so the ranking adapts automatically
    to any single-objective Comparator.

    Multi-objective PSO (MOPSO) requires a dedicated subclass with a
    separate Pareto archive for leader selection.

    Attributes
    ----------
    w : float
        Inertia weight.
    c1 : float
        Cognitive coefficient (personal best attraction).
    c2 : float
        Social coefficient (global best attraction).
    v_max : float or None
        Maximum velocity magnitude per dimension. ``None`` disables clamping.

    References
    ----------
    :cite:`kennedy1995pso`: Kennedy, J., & Eberhart, R. (1995). Particle
    swarm optimization. *Proceedings of the IEEE International Conference
    on Neural Networks (ICNN)*, 4, 1942-1948.

    :cite:`shi1998inertia`: Shi, Y., & Eberhart, R. (1998). A modified
    particle swarm optimizer. *Proceedings of the IEEE International
    Conference on Evolutionary Computation (ICEC)*, 69-73. (Introduces
    the inertia weight ``w`` used here.)
    """

    # PSO's tell operation updates particle state row-for-row.  A
    # pre-selection strategy may evaluate only a subset, but shrinking the
    # particle population to that subset would silently change the algorithm's
    # state dimension.  The canonical tell view therefore retains the full
    # proposal for this built-in algorithm.
    tell_requires_full_proposal = True

    def __init__(
        self,
        w: float = 0.7,
        c1: float = 1.5,
        c2: float = 1.5,
        v_max: float | None = None,
    ):
        """
        Initialize PSO.

        Parameters
        ----------
        w : float, optional
            Inertia weight. Defaults to 0.7.
        c1 : float, optional
            Cognitive coefficient. Defaults to 1.5.
        c2 : float, optional
            Social coefficient. Defaults to 1.5.
        v_max : float or None, optional
            Maximum velocity per dimension. Defaults to ``None`` (no clamping).
        """
        super().__init__()
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.v_max = v_max

    def contract(self) -> ComponentContract:
        """Return the PSO contract with comparator-backed feedback."""
        family = super().contract()
        feedback = family.ports["feedback_consumer"]
        offspring = replace(
            feedback.inputs[0],
            required_services=(ServiceRequirement(name="ComparisonService"),),
        )
        return replace(
            family,
            ports={
                **family.ports,
                "feedback_consumer": replace(feedback, inputs=(offspring,)),
            },
        )

    def get_required_attrs(self, problem: Problem) -> list[PopulationAttribute]:
        """
        Return PSO-specific attributes required by the Population.

        Parameters
        ----------
        problem : Problem
            The Problem object being referenced.

        Returns
        -------
        list[PopulationAttribute]
            velocity, pbest_x, pbest_f, and pbest_cv attributes.
        """
        dim = problem.dim
        n_obj = problem.n_obj
        return [
            PopulationAttribute(
                name="velocity", dtype=np.float64, shape=(dim,), default=0.0
            ),
            PopulationAttribute(
                name="pbest_x", dtype=np.float64, shape=(dim,), default=np.nan
            ),
            PopulationAttribute(
                name="pbest_f", dtype=np.float64, shape=(n_obj,), default=np.nan
            ),
            PopulationAttribute(
                name="pbest_cv", dtype=np.float64, shape=(), default=np.nan
            ),
        ]

    @property
    def population_class(self):
        """Return the population class."""
        return Population

    @property
    def archive_class(self):
        """Return the archive class."""
        return Archive

    @property
    def ask_notation(self) -> list[str]:
        """LaTeX notation lines for PSO.ask(): velocity update and position update."""
        return [
            r"$v \leftarrow w v + c_1 r_1 (p_{best} - x) + c_2 r_2 (g_{best} - x)$",
            r"$x \leftarrow x + v$",
        ]

    @property
    def tell_notation(self) -> list[str]:
        """LaTeX notation lines for PSO.tell(): pbest update."""
        return [
            r"$p_{best,i} \leftarrow x_i$ if $f(x_i) \prec f(p_{best,i})$",
            r"$P \leftarrow \mathcal{Q}$",
        ]

    def ask(
        self,
        request: ProposalRequest,
        state: StateView,
    ) -> ProposalBatch:
        """
        Update particle velocities and positions.

        Parameters
        ----------
        request : ProposalRequest
            Request-specific offspring count; PSO uses its population size.
        state : StateView
            Read-only algorithm state view.

        Returns
        -------
        ProposalBatch
            Candidates with updated ``x`` and ``velocity``.
        """
        del request
        ctx = algorithm_context(state)
        pop = ctx.population
        popsize = len(pop)
        bounds_srv = cast(BoundsService, ctx.compiled_service("BoundsService"))
        lb, ub = bounds_srv.bounds

        x = pop.get_array("x").copy()
        f = pop.get_array("f").copy()
        cv = pop.get_array("cv").copy()
        v = pop.get_array("velocity").copy()
        pbest_x = pop.get_array("pbest_x").copy()
        pbest_f = pop.get_array("pbest_f").copy()
        pbest_cv = pop.get_array("pbest_cv").copy()

        # Initialize pbest for particles that have not yet been evaluated.
        # pbest_f has shape (popsize, n_obj); NaN in any objective means uninitialised.
        uninit = np.isnan(pbest_f).any(axis=-1)
        pbest_x[uninit] = x[uninit]
        pbest_f[uninit] = f[uninit]
        pbest_cv[uninit] = cv[uninit]

        leader = self._select_leader(ctx, pbest_x, pbest_f, pbest_cv)

        r1 = ctx.rng.uniform(0.0, 1.0, size=(popsize, ctx.dim))
        r2 = ctx.rng.uniform(0.0, 1.0, size=(popsize, ctx.dim))

        v_new = self.w * v + self.c1 * r1 * (pbest_x - x) + self.c2 * r2 * (leader - x)

        if self.v_max is not None:
            v_new = np.clip(v_new, -self.v_max, self.v_max)

        x_new = x + v_new
        handler = ctx.problem.handler
        constraints = ctx.problem.constraints
        for i in range(len(x_new)):
            x_new[i] = handler.repair(x_new[i], constraints, lb, ub)

        cand_pop = pop.empty_like(capacity=popsize)
        cand_pop.extend(
            {
                "x": x_new,
                "f": np.full((popsize, ctx.n_obj), np.nan),
                "g": np.zeros((popsize, ctx.problem.n_constraints)),
                "cv": np.zeros(popsize),
                "velocity": v_new,
                "pbest_x": pbest_x,
                "pbest_f": pbest_f,
                "pbest_cv": pbest_cv,
            }
        )

        state.dispatch(PostAskEvent(ctx=ctx, candidates=x_new))

        return ProposalBatch.from_allocator(
            ctx.proposal_id_allocator,
            candidates=cand_pop,
            relations=ProposalRelations(row_count=len(cand_pop)),
            requirements=FeedbackRequirement(quantities=()),
        )

    def tell(
        self,
        feedback: FeedbackBatch,
        state: StateView,
    ) -> StatePatch:
        """
        Update the population and personal bests from evaluated offspring.

        Parameters
        ----------
        feedback : FeedbackBatch
            Feedback delivered for the current proposal.
        state : StateView
            Read-only algorithm state view.
        """
        del feedback
        ctx = algorithm_context(state)
        offspring = ctx.offspring
        if offspring is None:
            raise ConfigurationError("PSO.tell() requires an offspring population")
        x_new = offspring.get_array("x")
        f_new = offspring.get_array("f")
        g_new = offspring.get_array("g")
        cv_new = offspring.get_array("cv")
        v_new = offspring.get_array("velocity")
        pbest_x = offspring.get_array("pbest_x").copy()
        pbest_f = offspring.get_array("pbest_f").copy()
        pbest_cv = offspring.get_array("pbest_cv").copy()
        has_id = "id" in offspring.schema
        if has_id:
            id_new = offspring.get_array("id")

        popsize = len(offspring)
        population = ctx.population
        cmp = ctx.comparator
        for i in range(popsize):
            if np.any(np.isnan(f_new[i])):
                continue
            if (
                np.any(np.isnan(pbest_f[i]))
                or cmp.compare(f_new[i], cv_new[i], pbest_f[i], pbest_cv[i]) == -1
            ):
                pbest_x[i] = x_new[i]
                pbest_f[i] = f_new[i]
                pbest_cv[i] = cv_new[i]

        new_pop_data = {
            "x": x_new,
            "f": f_new,
            "g": g_new,
            "cv": cv_new,
            "velocity": v_new,
            "pbest_x": pbest_x,
            "pbest_f": pbest_f,
            "pbest_cv": pbest_cv,
        }
        if has_id:
            new_pop_data["id"] = id_new

        population.clear()
        population._extend_internal(new_pop_data, preserve_ids=True)
        return StatePatch(writes={POPULATIONS_MAIN: population})

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _select_leader(
        self,
        ctx: _LeaderContext,
        pbest_x: np.ndarray,
        pbest_f: np.ndarray,
        pbest_cv: np.ndarray,
    ) -> np.ndarray:
        """
        Select the global-best leader from the personal bests.

        The personal bests are sorted by ``ctx.comparator`` and the
        top-ranked particle's position is returned as the leader.

        Parameters
        ----------
        ctx : _LeaderContext
            Optimization context.
        pbest_x : np.ndarray
            Personal best positions, shape (popsize, dim).
        pbest_f : np.ndarray
            Personal best objective values, shape (popsize, n_obj).
        pbest_cv : np.ndarray
            Personal best constraint violations, shape (popsize,).

        Returns
        -------
        np.ndarray
            Leader position, shape (dim,).
        """
        popsize = len(pbest_x)
        pbest_pop = ctx.population.empty_like(capacity=popsize)
        pbest_pop.extend(
            {
                "x": pbest_x,
                "f": pbest_f,
                "g": np.zeros((popsize, ctx.problem.n_constraints)),
                "cv": pbest_cv,
                "velocity": np.zeros((popsize, ctx.dim)),
                "pbest_x": pbest_x,
                "pbest_f": pbest_f,
                "pbest_cv": pbest_cv,
            }
        )
        sorted_idx = ctx.comparator.sort_population(pbest_pop)
        return pbest_x[sorted_idx[0]]
