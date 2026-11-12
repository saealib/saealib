"""Adapter exposing an already-constructed pymoo Algorithm as saealib's Algorithm."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Protocol, cast

import numpy as np

from saealib.algorithms.base import (
    AskTellAlgorithm,
    ProposalRequest,
    algorithm_context,
)
from saealib.callback import PostAskEvent
from saealib.core.contracts import (
    PARTIAL_ALLOWED,
    REPEATED_ALLOWED,
    AssumptionSet,
    ComponentContract,
    ExecutionContract,
    FeedbackBatch,
    FeedbackRequirement,
    ProposalBatch,
    ProposalRelations,
)
from saealib.core.state import (
    POPULATIONS_MAIN,
    RUNTIME_CANDIDATE_ID_ALLOCATOR,
    StatePatch,
    StateView,
)
from saealib.exceptions import ConfigurationError
from saealib.identity import IDAllocator
from saealib.population import Population, PopulationAttribute
from saealib.population.population import (
    OBJECTIVE_SOURCE_COLUMN,
    UNKNOWN_SOURCE_CODE,
)
from saealib.problem.constraint import EqualityConstraint
from saealib.problem.problem import Problem
from saealib.space import BoundsService

if TYPE_CHECKING:
    from pymoo.core.problem import Problem as PymooCoreProblem

    from saealib.problem import Problem


class _PymooPopulationLike(Protocol):
    def get(self, *args: object, **kwargs: object) -> object:
        """Read one or more fields (e.g. "X", "F", "G", "H", "CV")."""
        ...

    def set(self, *args: object, **kwargs: object) -> object:
        """Write one or more fields (e.g. "X", "F", "G", "H")."""
        ...

    def __len__(self) -> int: ...


class _PymooContext(Protocol):
    """Capabilities used by the pymoo adapter's synchronization helpers."""

    @property
    def problem(self) -> Problem: ...

    @property
    def population(self) -> Population: ...

    @property
    def offspring(self) -> Population | None: ...

    @property
    def rng(self) -> np.random.Generator: ...

    @property
    def candidate_id_allocator(self) -> IDAllocator: ...

    @property
    def proposal_id_allocator(self) -> IDAllocator: ...

    def compiled_service(self, name: str) -> object: ...


class _PymooAlgorithmLike(Protocol):
    @property
    def pop(self) -> _PymooPopulationLike | None:
        """Internal population; None before setup(). Read-only from here."""
        ...

    def setup(self, problem: PymooCoreProblem, **kwargs: object) -> object:
        """Bind the algorithm to a problem and initialize its internal state."""
        ...

    def ask(self) -> _PymooPopulationLike | None:
        """Return the next infill population, or None if exhausted."""
        ...

    def tell(self, *args: object, **kwargs: object) -> object:
        """Advance internal state given evaluated infills."""
        ...


class PymooAlgorithm(AskTellAlgorithm):
    """
    Adapter wrapping a pymoo Algorithm (e.g. ``NSGA2()``) as saealib's ``Algorithm``.

    Lets researchers who already have a pymoo algorithm reuse it unchanged
    inside saealib's ask-tell loop and surrogate-assisted strategies.

    Parameters
    ----------
    pymoo_algorithm : pymoo.core.algorithm.Algorithm
        An already-constructed pymoo algorithm instance (not yet ``setup()``).
    allow_partial_tell : bool, optional
        ``PreSelectionStrategy`` truncates offspring before ``tell()``, so
        ``tell()`` may receive only a subset of what ``ask()`` produced. By
        default this raises :class:`~saealib.exceptions.ConfigurationError`,
        since index-coupled pymoo algorithms (e.g. differential evolution)
        would silently misalign parents and offspring under a subset. Set
        ``True`` to opt in and pass the subset through anyway. Default: False.

    Notes
    -----
    **Engine mode.** The wrapped pymoo algorithm owns its own population and
    internal state (survival mechanics, index-coupled operators, adaptive
    parameters). ``ctx.population`` is a *mirror*, refreshed from the pymoo
    algorithm's own ``.pop`` at the end of every :meth:`tell`, not the source
    of truth. This is the only way to reuse a pymoo algorithm's own
    tested logic unchanged, but it comes with real limitations:

    - **No checkpoint/resume.** ``OptimizationState.save()`` only
      serializes saealib's own arrays; the wrapped pymoo algorithm's
      internal state is not captured.
    - **``n_offspring`` is ignored.** The offspring count is fixed by the
      wrapped algorithm's own configuration (e.g. its ``pop_size``);
      strategies that request a larger candidate pool for pre-selection
      cannot enlarge it here.
    - **Constrained problems combined with surrogate-only generations are
      unsupported.** When a candidate never receives a true evaluation,
      saealib leaves its ``g``/``cv`` at schema defaults (``0.0``, looking
      falsely feasible) — this is an existing, library-wide limitation
      (:class:`~saealib.surrogate.prediction.SurrogatePrediction` has no
      ``cv`` field), not something specific to this adapter, but it means
      ``PymooAlgorithm``'s constraint handling should not be trusted under
      ``SurrogateOnlyLoopStage``/IB's untrue tail.

    Initialization is handled by mirroring saealib's already-evaluated
    initial population into the pymoo algorithm's own initial ``ask()``
    result before calling ``tell()`` on it, so the initial design-of-
    experiments is not duplicated or discarded.

    A ``pymoo_idx`` column tracks each candidate's position in the pymoo
    infill population, so that :meth:`tell` can recover the correct mapping
    even after ``SortByScoreStage``/``TopKSelectionStage`` reorder or
    truncate ``state.offspring`` between ``ask`` and ``tell``.
    """

    def __init__(
        self,
        pymoo_algorithm: _PymooAlgorithmLike,
        *,
        allow_partial_tell: bool = False,
    ) -> None:
        super().__init__()
        self.pymoo_algorithm = pymoo_algorithm
        self.allow_partial_tell = allow_partial_tell
        self._initialized = False
        self._ieq_idx: np.ndarray = np.empty(0, dtype=np.int64)
        self._eq_idx: np.ndarray = np.empty(0, dtype=np.int64)
        self._infills: _PymooPopulationLike | None = None

    def contract(self) -> ComponentContract:
        """Return the family contract with optional partial-feedback capability."""
        family = super().contract()
        family = replace(
            family,
            assumptions=AssumptionSet({"state.checkpointable": False}),
            state=replace(
                family.state,
                reads=(*family.state.reads, RUNTIME_CANDIDATE_ID_ALLOCATOR),
                writes=(*family.state.writes, RUNTIME_CANDIDATE_ID_ALLOCATOR),
            ),
        )
        if not self.allow_partial_tell:
            return family
        feedback = family.lifecycle.feedback
        assert feedback is not None
        return replace(
            family,
            lifecycle=replace(
                family.lifecycle,
                feedback=replace(
                    feedback,
                    completion=PARTIAL_ALLOWED,
                    multiplicity=REPEATED_ALLOWED,
                ),
            ),
            execution=ExecutionContract(
                required_runtime_capabilities=("partial_feedback",),
            ),
        )

    _candidate_id_attr = "saealib_candidate_id"

    def get_required_attrs(self, problem: Problem) -> list[PopulationAttribute]:
        """Add a ``pymoo_idx`` column tracking position in the pymoo infill pop."""
        return [PopulationAttribute("pymoo_idx", np.int64, (), default=-1)]

    @property
    def ask_notation(self) -> list[str]:
        """LaTeX notation line for ask(): delegate to the wrapped pymoo algorithm."""
        return [r"$\mathcal{Q} \leftarrow \text{pymoo\_algorithm.ask()}$"]

    @property
    def tell_notation(self) -> list[str]:
        """LaTeX notation lines for tell(): delegate, then mirror the population."""
        return [
            r"$\text{pymoo\_algorithm.tell}(\mathcal{Q})$",
            r"$P \leftarrow \text{pymoo\_algorithm.pop}$",
        ]

    def _build_pymoo_problem(self, ctx: _PymooContext) -> PymooCoreProblem:
        from pymoo.core.problem import Problem as PymooProblem

        problem = ctx.problem

        eq_mask = np.array(
            [isinstance(c, EqualityConstraint) for c in problem.constraints],
            dtype=bool,
        )
        self._ieq_idx = np.where(~eq_mask)[0]
        self._eq_idx = np.where(eq_mask)[0]

        bounds_srv = cast(BoundsService, ctx.compiled_service("BoundsService"))
        lb, ub = bounds_srv.bounds

        return PymooProblem(
            n_var=problem.dim,
            n_obj=problem.n_obj,
            n_ieq_constr=len(self._ieq_idx),
            n_eq_constr=len(self._eq_idx),
            xl=np.asarray(lb, dtype=float),
            xu=np.asarray(ub, dtype=float),
        )

    def _assign_objectives(
        self,
        pymoo_pop: _PymooPopulationLike,
        offspring: Population,
        problem: Problem,
        scatter_order: np.ndarray | None = None,
    ) -> None:
        """Write F (sign-converted) and split G into pymoo's G/H onto pymoo_pop.

        ``scatter_order`` maps each row ``k`` of ``offspring`` to its
        original ask-order position ``scatter_order[k]`` in ``pymoo_pop``
        (``pymoo_pop`` has ``offspring``'s *original*, un-reordered length).
        This is a scatter, not a gather: row ``k`` of ``offspring`` writes to
        position ``scatter_order[k]`` of ``pymoo_pop``, preserving each
        pymoo individual's original identity/order — required for
        index-coupled algorithms (e.g. DE) that compare ``pop[k]`` against
        ``infills[k]`` positionally. ``None`` means ``offspring`` is already
        in the same order as ``pymoo_pop`` (the initialization handshake, or
        an already-realigned partial-tell subset).
        """
        f = offspring.get_array("f")
        g = offspring.get_array("g")
        if scatter_order is not None:
            scatter_order = np.asarray(scatter_order, dtype=np.int64)
            if (
                len(scatter_order) != len(f)
                or np.any(scatter_order < 0)
                or np.any(scatter_order >= len(pymoo_pop))
            ):
                raise ConfigurationError("pymoo_idx is outside the infill population")
            if len(np.unique(scatter_order)) != len(scatter_order):
                raise ConfigurationError("pymoo_idx contains duplicate positions")
            n = len(pymoo_pop)
            try:
                f_full = np.array(pymoo_pop.get("F"), dtype=float, copy=True)
                if f_full.shape != (n, f.shape[1]):
                    f_full = np.zeros((n, f.shape[1]), dtype=float)
            except (AttributeError, KeyError, TypeError, ValueError):
                f_full = np.zeros((n, f.shape[1]), dtype=float)
            f_full[scatter_order] = f
            f = f_full
            if g.shape[1] > 0:
                g_full = np.zeros((n, g.shape[1]), dtype=float)
                try:
                    old_g = np.asarray(pymoo_pop.get("G"), dtype=float)
                    if old_g.shape == g_full.shape:
                        g_full[:] = old_g
                except (AttributeError, KeyError, TypeError, ValueError):
                    pass
                try:
                    old_h = np.asarray(pymoo_pop.get("H"), dtype=float)
                    if old_h.shape == (n, len(self._eq_idx)):
                        g_full[:, self._eq_idx] = old_h
                except (AttributeError, KeyError, TypeError, ValueError):
                    pass
                g_full[scatter_order] = g
                g = g_full
        pymoo_pop.set("F", -problem.direction * f)
        if len(self._ieq_idx) > 0:
            pymoo_pop.set("G", g[:, self._ieq_idx])
        if len(self._eq_idx) > 0:
            pymoo_pop.set("H", g[:, self._eq_idx])

    def _ensure_initialized(self, ctx: _PymooContext) -> None:
        if self._initialized:
            return
        from pymoo.core.termination import NoTermination

        pymoo_problem = self._build_pymoo_problem(ctx)
        self.pymoo_algorithm.setup(
            pymoo_problem,
            termination=NoTermination(),
            seed=int(ctx.rng.integers(0, 2**31 - 1)),
            verbose=False,
        )
        init = self.pymoo_algorithm.ask()
        if init is None or len(init) != len(ctx.population):
            got = 0 if init is None else len(init)
            raise ConfigurationError(
                f"pymoo algorithm's initial population size ({got}, from its own "
                f"pop_size) must equal saealib's initial population size "
                f"({len(ctx.population)}, from Initializer/Optimizer.set_popsize). "
                "Set both to the same value."
            )
        init.set("X", ctx.population.get_array("x"))
        if "id" in ctx.population.schema:
            self._set_pymoo_candidate_ids(init, ctx.population.get_array("id"))
        self._assign_objectives(init, ctx.population, ctx.problem)
        self.pymoo_algorithm.tell(infills=init)
        self._initialized = True

    def ask(
        self,
        request: ProposalRequest,
        state: StateView,
    ) -> ProposalBatch:
        """
        Generate offspring via the wrapped pymoo algorithm's own ``ask()``.

        Parameters
        ----------
        ctx : ExecutionContext
            Current optimization context.
        provider : Dispatchable
            Component provider.
        n_offspring : int or None, optional
            Ignored; the offspring count is fixed by the wrapped pymoo
            algorithm's own configuration.

        Returns
        -------
        Population
            Candidates with ``x`` and ``pymoo_idx`` set.
        """
        del request
        ctx = algorithm_context(state)
        self._ensure_initialized(ctx)
        infills = self.pymoo_algorithm.ask()
        if infills is None:
            raise ConfigurationError(
                "The wrapped pymoo algorithm's ask() returned None (has_next() is "
                "False). Check its termination/pop_size configuration."
            )
        self._infills = infills
        x = np.asarray(infills.get("X"), dtype=float)

        handler = ctx.problem.handler
        constraints = ctx.problem.constraints
        lb, ub = ctx.problem.lb, ctx.problem.ub
        for i in range(len(x)):
            x[i] = handler.repair(x[i], constraints, lb, ub)
            x[i] = ctx.problem.repair(x[i])

        state.dispatch(PostAskEvent(ctx=ctx, candidates=x))

        population = ctx.population
        cand = population.empty_like(capacity=len(x))
        data = {"x": x, "pymoo_idx": np.arange(len(x), dtype=np.int64)}
        if "id" in population.schema:
            candidate_ids = ctx.candidate_id_allocator.allocate(len(x))
            self._set_pymoo_candidate_ids(infills, candidate_ids)
            data["id"] = candidate_ids
        cand._extend_internal(data, preserve_ids=True)
        return ProposalBatch.from_allocator(
            ctx.proposal_id_allocator,
            candidates=cand,
            relations=ProposalRelations(row_count=len(cand)),
            requirements=FeedbackRequirement(quantities=()),
        )

    def tell(
        self,
        feedback: FeedbackBatch,
        state: StateView,
    ) -> StatePatch:
        """
        Update the wrapped pymoo algorithm, then mirror its population into ``ctx``.

        Parameters
        ----------
        ctx : ExecutionContext
            Current optimization context.
        provider : Dispatchable
            Component provider.
        offspring : Population
            Offspring population, possibly reordered or truncated relative
            to what :meth:`ask` produced.
        """
        del feedback
        ctx = algorithm_context(state)
        offspring = ctx.offspring
        if offspring is None:
            raise ConfigurationError(
                "PymooAlgorithm.tell() requires an offspring population"
            )
        assert self._infills is not None
        idx = offspring.get_array("pymoo_idx").astype(np.int64, copy=False)
        if idx.ndim != 1 or np.any(idx < 0) or np.any(idx >= len(self._infills)):
            raise ConfigurationError("pymoo_idx is outside the infill population")
        if len(np.unique(idx)) != len(idx):
            raise ConfigurationError("pymoo_idx contains duplicate positions")

        if len(idx) == len(self._infills):
            # Full reorder (possibly identity). Scatter F/G/H back into
            # self._infills's *original* ask-order positions so the object
            # passed to tell() has the exact identity/order ask() produced,
            # which index-coupled algorithms (e.g. DE) rely on positionally.
            self._assign_objectives(
                self._infills, offspring, ctx.problem, scatter_order=idx
            )
            if "id" in offspring.schema:
                full_ids = self._pymoo_candidate_ids(self._infills)
                if full_ids is None:
                    raise ConfigurationError(
                        "pymoo infill candidate provenance is missing"
                    )
                full_ids = full_ids.copy()
                full_ids[idx] = offspring.get_array("id")
                self._set_pymoo_candidate_ids(self._infills, full_ids)
            infills = self._infills
        else:
            if not self.allow_partial_tell:
                raise ConfigurationError(
                    f"PymooAlgorithm.tell() received {len(idx)} of "
                    f"{len(self._infills)} candidates ask() produced. This happens "
                    "with PreSelectionStrategy's top-k truncation; index-coupled "
                    "pymoo algorithms (e.g. differential evolution) would silently "
                    "misalign parents and offspring under a partial tell. Pass "
                    "allow_partial_tell=True to opt in anyway, or use "
                    "DirectStrategy/IndividualBasedStrategy instead."
                )
            # offspring row k already corresponds to original ask-position
            # idx[k]; subsetting self._infills by idx realigns it to match
            # offspring directly, so no further reordering is needed here.
            infills = self._infills[idx]  # type: ignore  # pymoo Population is ndarray-subscriptable; not worth modeling numpy's full __getitem__ overload set in the Protocol
            self._assign_objectives(infills, offspring, ctx.problem)
            if "id" in offspring.schema:
                full_ids = self._pymoo_candidate_ids(self._infills)
                if full_ids is None:
                    raise ConfigurationError(
                        "pymoo infill candidate provenance is missing"
                    )
                full_ids = full_ids.copy()
                full_ids[idx] = offspring.get_array("id")
                self._set_pymoo_candidate_ids(self._infills, full_ids)

        self.pymoo_algorithm.tell(infills=infills)
        self._sync_population(ctx)
        return StatePatch(writes={POPULATIONS_MAIN: ctx.population})

    @classmethod
    def _set_pymoo_candidate_ids(
        cls, pymoo_pop: _PymooPopulationLike, candidate_ids: np.ndarray
    ) -> None:
        candidate_ids = np.asarray(candidate_ids, dtype=np.int64)
        if candidate_ids.ndim != 1 or len(candidate_ids) != len(pymoo_pop):
            raise ConfigurationError("pymoo candidate provenance has invalid shape")
        pymoo_pop.set(
            cls._candidate_id_attr,
            candidate_ids,
        )

    @classmethod
    def _pymoo_candidate_ids(cls, pymoo_pop: _PymooPopulationLike) -> np.ndarray | None:
        try:
            values = pymoo_pop.get(cls._candidate_id_attr)
        except (AttributeError, KeyError, TypeError):
            return None
        if values is None:
            return None
        raw_values = np.asarray(values).reshape(-1)
        if raw_values.dtype == object and any(value is None for value in raw_values):
            return None
        try:
            values = raw_values.astype(np.int64, copy=False)
        except (TypeError, ValueError):
            return None
        if len(values) != len(pymoo_pop) or np.any(values < 0):
            return None
        if len(np.unique(values)) != len(values):
            raise ConfigurationError("pymoo survivor candidate IDs are not unique")
        return values

    def _sync_population(self, ctx: _PymooContext) -> None:
        alg_pop = self.pymoo_algorithm.pop
        assert alg_pop is not None  # setup()/tell() already ran by this point
        population = ctx.population
        if len(alg_pop) != len(population):
            raise ConfigurationError(
                f"Wrapped pymoo algorithm's internal population size ({len(alg_pop)}) "
                f"no longer matches saealib's population size ({len(population)}). "
                "This can happen with archive-growing pymoo algorithms that are not "
                "supported by this adapter."
            )
        problem = ctx.problem
        n = len(alg_pop)
        x = np.asarray(alg_pop.get("X"), dtype=float)
        f = -problem.direction * np.asarray(alg_pop.get("F"), dtype=float)
        g = np.zeros((n, problem.n_constraints), dtype=float)
        if len(self._ieq_idx) > 0:
            g[:, self._ieq_idx] = np.asarray(alg_pop.get("G"), dtype=float)
        if len(self._eq_idx) > 0:
            g[:, self._eq_idx] = np.asarray(alg_pop.get("H"), dtype=float)
        cv = np.asarray(alg_pop.get("CV"), dtype=float).reshape(n)
        survivor_ids = self._pymoo_candidate_ids(alg_pop)

        new_pop_data = {
            "x": x,
            "f": f,
            "g": g,
            "cv": cv,
            "pymoo_idx": np.full(n, -1, dtype=np.int64),
        }
        if "id" in population.schema:
            if survivor_ids is None:
                raise ConfigurationError(
                    "pymoo survivor candidate provenance is missing; "
                    "survival must preserve saealib_candidate_id"
                )
            new_pop_data["id"] = survivor_ids
        if OBJECTIVE_SOURCE_COLUMN in population.schema:
            new_pop_data[OBJECTIVE_SOURCE_COLUMN] = _survivor_objective_sources(
                n, survivor_ids, population, ctx.offspring
            )

        population.clear()
        population._extend_internal(new_pop_data, preserve_ids=True)


def _survivor_objective_sources(
    n: int,
    survivor_ids: np.ndarray | None,
    population: Population,
    offspring: Population | None,
) -> np.ndarray:
    """Look up each survivor's objective source code by candidate ID."""
    sources = np.full(n, UNKNOWN_SOURCE_CODE, dtype=np.uint8)
    if survivor_ids is None or "id" not in population.schema:
        return sources
    known: dict[int, int] = {}
    for rows in (population, offspring):
        if (
            rows is None
            or "id" not in rows.schema
            or OBJECTIVE_SOURCE_COLUMN not in rows.schema
        ):
            continue
        known.update(
            zip(
                rows.get_array("id").tolist(),
                rows.get_array(OBJECTIVE_SOURCE_COLUMN).tolist(),
                strict=True,
            )
        )
    for row, candidate_id in enumerate(survivor_ids.tolist()):
        sources[row] = known.get(candidate_id, UNKNOWN_SOURCE_CODE)
    return sources
