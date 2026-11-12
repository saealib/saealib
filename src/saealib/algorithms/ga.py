"""Genetic Algorithm module."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import TYPE_CHECKING, Literal, cast

import numpy as np

from saealib.algorithms.base import (
    AskTellAlgorithm,
    ProposalRequest,
    algorithm_context,
    algorithm_state,
)
from saealib.callback import PostAskEvent, PostCrossoverEvent, PostMutationEvent
from saealib.core.contracts import (
    ComponentContract,
    FeedbackBatch,
    FeedbackRequirement,
    PartSpec,
    ProposalBatch,
    ProposalRelations,
)
from saealib.core.state import POPULATIONS_MAIN, ExecutionContext, StatePatch, StateView
from saealib.exceptions import ConfigurationError
from saealib.operators.crossover import (
    Crossover,
    CrossoverCategorical,
    CrossoverIntegerSBX,
)
from saealib.operators.dedup import DuplicateElimination
from saealib.operators.mutation import (
    Mutation,
    MutationCategorical,
    MutationIntegerUniform,
)
from saealib.population import Population
from saealib.population.genome import DenseVectorBatch
from saealib.problem import Problem
from saealib.registry import register
from saealib.space import BoundsService

if TYPE_CHECKING:
    from saealib.operators.selection import ParentSelection, SurvivorSelection


def _resolve_variation_execution(
    variation_execution: (
        Literal["batch", "sequential"] | dict[str, Literal["batch", "sequential"]]
    ),
    kind: Literal["crossover", "mutation"],
) -> Literal["batch", "sequential"]:
    valid_modes = ("batch", "sequential")
    if isinstance(variation_execution, str):
        if variation_execution not in valid_modes:
            raise ConfigurationError(
                "variation_execution must be 'batch', 'sequential', or a dict "
                "mapping 'crossover' and/or 'mutation' to one of those modes"
            )
        return variation_execution

    if not isinstance(variation_execution, dict):
        raise ConfigurationError(
            "variation_execution must be 'batch', 'sequential', or a dict "
            "mapping 'crossover' and/or 'mutation' to one of those modes"
        )

    unknown_keys = set(variation_execution) - {"crossover", "mutation"}
    invalid_values = [
        value
        for value in variation_execution.values()
        if not isinstance(value, str) or value not in valid_modes
    ]
    if unknown_keys or invalid_values:
        raise ConfigurationError(
            "variation_execution must be 'batch', 'sequential', or a dict "
            "mapping 'crossover' and/or 'mutation' to one of those modes"
        )
    return variation_execution.get(kind, "batch")


def _canonical_merge_pool(
    population: Population, offspring: Population, capacity: int
) -> Population | None:
    """Allocate the private dense pool used only by :meth:`GA.tell`.

    The two callers immediately append populations with the same schema, so
    every live cell is overwritten before the pool is observed.  Keep this
    deliberately narrower than ``Population.empty_like``: exact built-in
    populations, the normal ``get_array`` implementation, and canonical
    identity-backed dense genomes are required.  Any customisation returns
    ``None`` and the caller uses the public-compatible factory.
    """
    if (
        type(population) is not Population
        or type(offspring) is not Population
        or type(population).get_array is not Population.get_array
        or type(offspring).get_array is not Population.get_array
        or population._schema != offspring._schema
        or not isinstance(population._genome_batch, DenseVectorBatch)
        or not isinstance(offspring._genome_batch, DenseVectorBatch)
        or getattr(population._dense_numeric_view, "_canonical_identity_backing", False)
        is not True
        or getattr(offspring._dense_numeric_view, "_canonical_identity_backing", False)
        is not True
        or population._legacy_scalar_x
        or offspring._legacy_scalar_x
        or population._genome_items is not None
        or offspring._genome_items is not None
    ):
        return None

    pool = Population.__new__(Population)
    pool._capacity = capacity
    pool.space = population.space
    pool._size = 0
    pool._structure_version = 0
    pool._value_version = 0
    pool._data = {
        attr.name: np.empty((capacity, *attr.shape), dtype=attr.dtype, order="C")
        for attr in population.attrs
    }
    pool._schema = dict(population._schema)
    pool._cache = {}
    pool._dense_genomes_view_cache = None
    pool._dense_numeric_view = population._dense_numeric_view
    pool._genome_items = None
    pool._legacy_scalar_x = False
    pool._genome_batch = DenseVectorBatch._from_borrowed_view(pool._data["x"])
    return pool


def _route_crossover(
    parent: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    rng: np.random.Generator,
    problem: Problem,
    cont_op: Crossover,
    int_op: Crossover,
    cat_op: Crossover,
) -> np.ndarray:
    i_mask = problem.integer_mask
    cat_mask = problem.categorical_mask
    if not i_mask.any() and not cat_mask.any():
        return cont_op.crossover(parent, (lb, ub), rng=rng)

    n_children = cont_op.n_children
    dim = parent.shape[1]
    offspring = np.empty((n_children, dim))
    c_mask = problem.continuous_mask

    if c_mask.any():
        offspring[:, c_mask] = cont_op.crossover(
            parent[:, c_mask], (lb[c_mask], ub[c_mask]), rng=rng
        )
    if i_mask.any():
        offspring[:, i_mask] = int_op.crossover(
            parent[:, i_mask], (lb[i_mask], ub[i_mask]), rng=rng
        )
    if cat_mask.any():
        offspring[:, cat_mask] = cat_op.crossover(
            parent[:, cat_mask], (lb[cat_mask], ub[cat_mask]), rng=rng
        )

    return offspring


def _route_mutation(
    p: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    rng: np.random.Generator,
    problem: Problem,
    cont_op: Mutation,
    int_op: Mutation,
    cat_op: Mutation,
) -> np.ndarray:
    i_mask = problem.integer_mask
    cat_mask = problem.categorical_mask
    if not i_mask.any() and not cat_mask.any():
        return cont_op.mutate(p, (lb, ub), rng=rng)

    result = p.copy()
    c_mask = problem.continuous_mask

    if c_mask.any():
        result[c_mask] = cont_op.mutate(p[c_mask], (lb[c_mask], ub[c_mask]), rng=rng)
    if i_mask.any():
        result[i_mask] = int_op.mutate(p[i_mask], (lb[i_mask], ub[i_mask]), rng=rng)
    if cat_mask.any():
        result[cat_mask] = cat_op.mutate(
            p[cat_mask], (lb[cat_mask], ub[cat_mask]), rng=rng
        )

    return result


@register()
class GA(AskTellAlgorithm):
    """
    Genetic Algorithm class.

    Attributes
    ----------
    crossover : Crossover
        Crossover operator for continuous dimensions.
    mutation : Mutation
        Mutation operator for continuous dimensions.
    parent_selection : ParentSelection
        Parent selection operator.
    survivor_selection : SurvivorSelection
        Survivor selection operator.
    integer_crossover : Crossover
        Crossover operator for integer dimensions.
    integer_mutation : Mutation
        Mutation operator for integer dimensions.
    categorical_crossover : Crossover
        Crossover operator for categorical dimensions.
    categorical_mutation : Mutation
        Mutation operator for categorical dimensions.
    duplicate_elimination : DuplicateElimination or None
        When set, offspring that duplicate any member of the current population
        are replaced by re-generated candidates (up to ``max_retries`` attempts).
        ``None`` disables duplicate elimination (default behaviour).
    variation_execution : {"batch", "sequential"} or dict
        Execution mode for crossover and mutation. A string applies to both;
        a dict may set ``"crossover"`` and ``"mutation"`` independently, with
        omitted keys defaulting to ``"batch"``. In batch mode, all gate
        decisions and operations complete before any post-operation hook runs,
        so a stateful hook cannot influence a later individual's operation.
        Sequential mode completes each individual's operation and hook before
        starting the next, preserving the pre-batching behavior byte-for-byte.
        Mixed-variable problems now use per-type batch operations by default,
        which changes their historical random-number sequence; select
        sequential mode to preserve it. This is a semantic choice, not only a
        performance setting.
    """

    def __init__(
        self,
        crossover: Crossover,
        mutation: Mutation,
        parent_selection: ParentSelection,
        survivor_selection: SurvivorSelection,
        *,
        duplicate_elimination: DuplicateElimination | None = None,
        variation_execution: (
            Literal["batch", "sequential"] | dict[str, Literal["batch", "sequential"]]
        ) = "batch",
        integer_crossover: Crossover | None = None,
        integer_mutation: Mutation | None = None,
        categorical_crossover: Crossover | None = None,
        categorical_mutation: Mutation | None = None,
    ):
        """
        Initialize GA (Genetic Algorithm) class.

        Parameters
        ----------
        crossover : Crossover
            Crossover operator for continuous dimensions.
        mutation : Mutation
            Mutation operator for continuous dimensions.
        parent_selection : ParentSelection
            Parent selection operator.
        survivor_selection : SurvivorSelection
            Survivor selection operator.
        duplicate_elimination : DuplicateElimination, optional
            When provided, offspring that duplicate any member of the current
            population are replaced by re-generated candidates.  ``None``
            (default) disables duplicate elimination.
        variation_execution : {"batch", "sequential"} or dict, optional
            Execution mode for crossover and mutation. A string applies to
            both operators. A dict may contain ``"crossover"`` and
            ``"mutation"`` independently; an omitted key defaults to
            ``"batch"``. In batch mode, every individual's gate decision and
            operation is computed before any ``post_crossover`` or
            ``post_mutation`` hook runs for any individual in that call. A
            stateful hook may observe earlier hooks in the subsequent
            post-processing pass, but cannot influence a later individual's
            operation. In sequential mode, each individual's complete
            operation-then-hook cycle finishes before the next begins,
            matching the library's pre-batching behavior byte-for-byte. This
            is a semantic choice, not only a performance setting.
            Mixed-variable problems use per-type batch operations by default,
            changing their historical random-number sequence; use sequential
            mode to preserve it.
        integer_crossover : Crossover, optional
            Crossover operator for integer dimensions.
            Defaults to ``CrossoverIntegerSBX`` with the same rate as *crossover*.
        integer_mutation : Mutation, optional
            Mutation operator for integer dimensions.
            Defaults to ``MutationIntegerUniform`` with the same rate as *mutation*.
        categorical_crossover : Crossover, optional
            Crossover operator for categorical dimensions.
            Defaults to ``CrossoverCategorical`` with the same rate as *crossover*.
        categorical_mutation : Mutation, optional
            Mutation operator for categorical dimensions.
            Defaults to ``MutationCategorical`` with the same rate as *mutation*.
        """
        super().__init__()
        self.crossover = crossover
        self.mutation = mutation
        self.parent_selection = parent_selection
        self.survivor_selection = survivor_selection
        self.duplicate_elimination = duplicate_elimination
        self.variation_execution = variation_execution
        _resolve_variation_execution(self.variation_execution, "crossover")

        _cr = getattr(crossover, "prob", 1.0)
        _pv = getattr(mutation, "prob_var", None)
        self.integer_crossover: Crossover = (
            integer_crossover
            if integer_crossover is not None
            else CrossoverIntegerSBX(_cr, eta=20.0)
        )
        self.integer_mutation: Mutation = (
            integer_mutation
            if integer_mutation is not None
            else MutationIntegerUniform(prob_var=_pv)
        )
        self.categorical_crossover: Crossover = (
            categorical_crossover
            if categorical_crossover is not None
            else CrossoverCategorical(_cr)
        )
        self.categorical_mutation: Mutation = (
            categorical_mutation
            if categorical_mutation is not None
            else MutationCategorical(prob_var=_pv)
        )

        for _name, _op in [
            ("integer_crossover", self.integer_crossover),
            ("categorical_crossover", self.categorical_crossover),
        ]:
            if _op.n_children != self.crossover.n_children:
                raise ConfigurationError(
                    f"{_name}.n_children={_op.n_children} must equal "
                    f"crossover.n_children={self.crossover.n_children} "
                    "for mixed-variable routing"
                )
            if _op.n_parents != self.crossover.n_parents:
                raise ConfigurationError(
                    f"{_name}.n_parents={_op.n_parents} must equal "
                    f"crossover.n_parents={self.crossover.n_parents} "
                    "for mixed-variable routing"
                )

    def contract(self) -> ComponentContract:
        """Return the genetic-algorithm contract."""
        return replace(
            super().contract(),
            parts=(
                PartSpec(name="crossover", contract=self.crossover.contract()),
                PartSpec(name="mutation", contract=self.mutation.contract()),
                PartSpec(
                    name="parent_selection", contract=self.parent_selection.contract()
                ),
                PartSpec(
                    name="survivor_selection",
                    contract=self.survivor_selection.contract(),
                ),
                PartSpec(
                    name="integer_crossover",
                    contract=self.integer_crossover.contract(),
                ),
                PartSpec(
                    name="integer_mutation", contract=self.integer_mutation.contract()
                ),
                PartSpec(
                    name="categorical_crossover",
                    contract=self.categorical_crossover.contract(),
                ),
                PartSpec(
                    name="categorical_mutation",
                    contract=self.categorical_mutation.contract(),
                ),
            ),
        )

    @property
    def ask_notation(self) -> list[str]:
        """LaTeX notation lines for GA.ask(): select → crossover → mutate."""
        return [
            r"$I_m \leftarrow \mathrm{select}(P,\, n_{pair})$",
            r"$\mathcal{Q} \leftarrow \mathrm{crossover}(P[I_m])$",
            r"$\mathcal{Q} \leftarrow \mathrm{mutate}(\mathcal{Q})$",
        ]

    @property
    def tell_notation(self) -> list[str]:
        r"""LaTeX notation lines for GA.tell(): $(\mu + \lambda)$ survivor selection."""
        return [
            r"$P \leftarrow \mathrm{select}_{(\mu+\lambda)}"
            r"(P \cup \mathcal{Q},\, \mu)$",
        ]

    def ask(
        self,
        request: ProposalRequest,
        state: StateView,
    ) -> ProposalBatch:
        """
        Generate offspring via crossover and mutation.

        Parameters
        ----------
        request : ProposalRequest
            Request-specific offspring count.
        state : StateView
            Read-only algorithm state view.

        Returns
        -------
        ProposalBatch
        """
        ctx = algorithm_context(state)
        selection_ctx = algorithm_state(ctx)
        n_offspring = request.n_offspring
        # Re-validate per-type crossover consistency here because operators may be
        # replaced or mutated after __init__, bypassing constructor checks.
        if ctx.problem.integer_mask.any() or ctx.problem.categorical_mask.any():
            for _name, _op in [
                ("integer_crossover", self.integer_crossover),
                ("categorical_crossover", self.categorical_crossover),
            ]:
                if _op.n_children != self.crossover.n_children:
                    raise ConfigurationError(
                        f"{_name}.n_children={_op.n_children} must equal "
                        f"crossover.n_children={self.crossover.n_children} "
                        "for mixed-variable routing"
                    )
                if _op.n_parents != self.crossover.n_parents:
                    raise ConfigurationError(
                        f"{_name}.n_parents={_op.n_parents} must equal "
                        f"crossover.n_parents={self.crossover.n_parents} "
                        "for mixed-variable routing"
                    )

        mixed = bool(
            ctx.problem.integer_mask.any() or ctx.problem.categorical_mask.any()
        )

        pop = ctx.population.get_array("x")
        popsize = len(pop)
        target = n_offspring if n_offspring is not None else popsize
        bounds_srv = cast(BoundsService, ctx.compiled_service("BoundsService"))
        lb, ub = bounds_srv.bounds
        n_children = self.crossover.n_children
        n_pair = math.ceil(target / n_children)
        parent_idx_m = (
            self.parent_selection.select(
                ctx.population,
                n_pair=n_pair,
                n_parents=self.crossover.n_parents,
                rng=ctx.rng,
                ctx=selection_ctx,
            )
            % popsize
        )
        handler = ctx.problem.handler
        constraints = ctx.problem.constraints

        parents_batch = pop[parent_idx_m]
        cand = self._crossover_pairs(ctx, parents_batch, lb, ub, mixed)
        cand = self._repair_batch(ctx, cand, handler, constraints, lb, ub)
        state.dispatch(PostCrossoverEvent(ctx=ctx, candidates=cand))

        cand = self._mutate_candidates(ctx, cand, lb, ub, mixed)
        cand = self._repair_batch(ctx, cand, handler, constraints, lb, ub)
        state.dispatch(PostMutationEvent(ctx=ctx, candidates=cand))

        if self.duplicate_elimination is not None:
            pop_x = ctx.population.get_array("x")
            de = self.duplicate_elimination
            for _ in range(de.max_retries):
                dup = de.find_duplicates(cand[:target], pop_x)
                if not dup.any():
                    break
                dup_idx = np.where(dup)[0]
                repl = self._make_offspring(
                    ctx, len(dup_idx), pop, popsize, lb, ub, handler, constraints
                )
                cand[dup_idx] = repl[: len(dup_idx)]

        state.dispatch(PostAskEvent(ctx=ctx, candidates=cand))

        cand_pop = ctx.population.empty_like(capacity=target)
        cand_pop.extend({"x": cand[:target]})
        return ProposalBatch.from_allocator(
            ctx.proposal_id_allocator,
            candidates=cand_pop,
            relations=ProposalRelations(row_count=len(cand_pop)),
            requirements=FeedbackRequirement(quantities=()),
        )

    def _crossover_pairs(
        self,
        ctx: ExecutionContext,
        parents_batch: np.ndarray,
        lb: np.ndarray,
        ub: np.ndarray,
        mixed: bool,
    ) -> np.ndarray:
        """Generate post-crossover offspring for a batch of parent groups.

        Batch mode operates on all dimensions together for continuous-only
        problems and by variable-type column masks for mixed problems.
        Sequential mode runs the per-pair loop. Mixed problems in batch mode
        do not reproduce historical sequential random-number sequences.

        Parameters
        ----------
        ctx : ExecutionContext
            Current optimization context.
        parents_batch : np.ndarray
            Batch of parent groups. shape = (n_pair, n_parents, dim)
        lb : np.ndarray
            Lower bounds.
        ub : np.ndarray
            Upper bounds.
        mixed : bool
            Whether the problem has any integer or categorical dimensions.

        Returns
        -------
        np.ndarray
            Offspring. shape = (n_pair * n_children, dim)
        """
        n_pair = parents_batch.shape[0]
        n_children = self.crossover.n_children
        dim = ctx.dim
        mode = _resolve_variation_execution(self.variation_execution, "crossover")
        use_batch = (not mixed) and (mode == "batch")

        if use_batch:
            gate = ctx.rng.random(n_pair) < self.crossover.prob
            if gate.any():
                batch_offspring = self.crossover.crossover_batch(
                    parents_batch[gate], (lb, ub), rng=ctx.rng
                )
            else:
                batch_offspring = np.empty((0, n_children, dim))
            cand = parents_batch[:, :n_children].copy()
            cand[gate] = batch_offspring
            cand = self.crossover.post_crossover_batch(
                cand, parents_batch, ctx.rng, ctx
            )
            return cand.reshape(n_pair * n_children, dim)

        if mixed and mode == "batch":
            gate = ctx.rng.random(n_pair) < self.crossover.prob
            cand = parents_batch[:, :n_children].copy()
            if gate.any():
                gated_parents = parents_batch[gate]
                gated_offspring = np.empty((int(gate.sum()), n_children, dim))
                c_mask = ctx.problem.continuous_mask
                i_mask = ctx.problem.integer_mask
                cat_mask = ctx.problem.categorical_mask
                if c_mask.any():
                    gated_offspring[:, :, c_mask] = self.crossover.crossover_batch(
                        gated_parents[:, :, c_mask],
                        (lb[c_mask], ub[c_mask]),
                        rng=ctx.rng,
                    )
                if i_mask.any():
                    gated_offspring[:, :, i_mask] = (
                        self.integer_crossover.crossover_batch(
                            gated_parents[:, :, i_mask],
                            (lb[i_mask], ub[i_mask]),
                            rng=ctx.rng,
                        )
                    )
                if cat_mask.any():
                    gated_offspring[:, :, cat_mask] = (
                        self.categorical_crossover.crossover_batch(
                            gated_parents[:, :, cat_mask],
                            (lb[cat_mask], ub[cat_mask]),
                            rng=ctx.rng,
                        )
                    )
                cand[gate] = gated_offspring
            cand = self.crossover.post_crossover_batch(
                cand, parents_batch, ctx.rng, ctx
            )
            return cand.reshape(n_pair * n_children, dim)

        cand = np.empty((n_pair * n_children, dim))
        for i in range(n_pair):
            parent = parents_batch[i]
            if ctx.rng.random() < self.crossover.prob:
                c = _route_crossover(
                    parent,
                    lb,
                    ub,
                    ctx.rng,
                    ctx.problem,
                    self.crossover,
                    self.integer_crossover,
                    self.categorical_crossover,
                )
            else:
                c = parent[:n_children].copy()
            c = self.crossover.post_crossover(c, parent, ctx.rng, ctx)
            cand[i * n_children : (i + 1) * n_children] = c
        return cand

    def _mutate_candidates(
        self,
        ctx: ExecutionContext,
        cand: np.ndarray,
        lb: np.ndarray,
        ub: np.ndarray,
        mixed: bool,
    ) -> np.ndarray:
        """Generate post-mutation offspring for a batch of candidates.

        Batch mode operates on all dimensions together for continuous-only
        problems and by variable-type column masks for mixed problems.
        Sequential mode runs the per-individual loop. Each type's batch
        mutation gates independently. Mixed problems in batch mode do not
        reproduce historical sequential random-number sequences.

        Parameters
        ----------
        ctx : ExecutionContext
            Current optimization context.
        cand : np.ndarray
            Candidates to mutate. shape = (n, dim)
        lb : np.ndarray
            Lower bounds.
        ub : np.ndarray
            Upper bounds.
        mixed : bool
            Whether the problem has any integer or categorical dimensions.

        Returns
        -------
        np.ndarray
            Mutated candidates. shape = (n, dim)
        """
        mode = _resolve_variation_execution(self.variation_execution, "mutation")
        use_batch = (not mixed) and (mode == "batch")

        if use_batch:
            cand = self.mutation.mutate_batch(cand, (lb, ub), rng=ctx.rng)
            return self.mutation.post_mutation_batch(cand, (lb, ub), ctx.rng, ctx)

        if mixed and mode == "batch":
            result = cand.copy()
            c_mask = ctx.problem.continuous_mask
            i_mask = ctx.problem.integer_mask
            cat_mask = ctx.problem.categorical_mask
            if c_mask.any():
                result[:, c_mask] = self.mutation.mutate_batch(
                    cand[:, c_mask], (lb[c_mask], ub[c_mask]), rng=ctx.rng
                )
            if i_mask.any():
                result[:, i_mask] = self.integer_mutation.mutate_batch(
                    cand[:, i_mask], (lb[i_mask], ub[i_mask]), rng=ctx.rng
                )
            if cat_mask.any():
                result[:, cat_mask] = self.categorical_mutation.mutate_batch(
                    cand[:, cat_mask], (lb[cat_mask], ub[cat_mask]), rng=ctx.rng
                )
            return self.mutation.post_mutation_batch(result, (lb, ub), ctx.rng, ctx)

        for i in range(len(cand)):
            cand[i] = _route_mutation(
                cand[i],
                lb,
                ub,
                ctx.rng,
                ctx.problem,
                self.mutation,
                self.integer_mutation,
                self.categorical_mutation,
            )
            cand[i] = self.mutation.post_mutation(cand[i], (lb, ub), ctx.rng, ctx)

        return cand

    def _repair_batch(
        self,
        ctx: ExecutionContext,
        cand: np.ndarray,
        handler,
        constraints,
        lb: np.ndarray,
        ub: np.ndarray,
    ) -> np.ndarray:
        """Repair *cand* via the constraint handler (row-wise), then the problem.

        ``handler.repair`` stays row-by-row (mirrors how ``PSO`` already
        leaves it), while ``ctx.problem.repair`` is called once for the
        whole batch since it already accepts ``(n, dim)`` arrays.
        """
        for i in range(len(cand)):
            cand[i] = handler.repair(cand[i], constraints, lb, ub)
        return ctx.problem.repair(cand)

    def _make_offspring(
        self,
        ctx: ExecutionContext,
        n_target: int,
        pop: np.ndarray,
        popsize: int,
        lb: np.ndarray,
        ub: np.ndarray,
        handler,
        constraints,
    ) -> np.ndarray:
        """Generate *n_target* offspring without dispatching events.

        Used exclusively by the duplicate-elimination retry loop in
        :meth:`ask` to silently replace duplicate candidates.
        """
        mixed = bool(
            ctx.problem.integer_mask.any() or ctx.problem.categorical_mask.any()
        )
        n_children = self.crossover.n_children
        n_pair = math.ceil(n_target / n_children)
        parent_idx = (
            self.parent_selection.select(
                ctx.population,
                n_pair=n_pair,
                n_parents=self.crossover.n_parents,
                rng=ctx.rng,
                ctx=algorithm_state(ctx),
            )
            % popsize
        )
        parents_batch = pop[parent_idx]
        batch = self._crossover_pairs(ctx, parents_batch, lb, ub, mixed)
        batch = self._repair_batch(ctx, batch, handler, constraints, lb, ub)
        batch = self._mutate_candidates(ctx, batch, lb, ub, mixed)
        batch = self._repair_batch(ctx, batch, handler, constraints, lb, ub)
        return batch

    def tell(
        self,
        feedback: FeedbackBatch,
        state: StateView,
    ) -> StatePatch:
        """
        Update the population using (μ+λ) survivor selection.

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
            raise ConfigurationError("GA.tell() requires an offspring population")
        population = ctx.population
        popsize = len(population)

        pool = _canonical_merge_pool(population, offspring, popsize + len(offspring))
        if pool is None:
            pool = population.empty_like(capacity=popsize + len(offspring))
        pool._extend_internal(population, preserve_ids=True)
        pool._extend_internal(offspring, preserve_ids=True)

        survivor_idx = self.survivor_selection.select(
            pool, popsize, ctx=algorithm_state(ctx)
        )

        if not population._replace_from_population(
            pool, survivor_idx, preserve_ids=True
        ):
            population.clear()
            population._extend_internal(pool.extract(survivor_idx), preserve_ids=True)
        return StatePatch(writes={POPULATIONS_MAIN: population})
