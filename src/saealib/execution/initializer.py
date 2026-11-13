"""Initializer: builds the initial Population, Archive, and OptimizationState."""

from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from dataclasses import replace
from typing import cast

import numpy as np
import scipy.stats

from saealib.callback import InitialEvaluationEndEvent, InitialEvaluationStartEvent
from saealib.comparators import Comparator, NSGA3Comparator
from saealib.context import OptimizationState
from saealib.core.contracts import (
    MANY,
    ComponentContract,
    DataSpec,
    Fixed,
    PortContract,
    PortDirection,
    PortSpec,
    StateContract,
    Var,
)
from saealib.core.state import (
    ARCHIVES_MAIN,
    ARCHIVES_PARETO,
    EVALUATIONS_COUNT,
    POPULATIONS_MAIN,
    RUNTIME_CANDIDATE_ID_ALLOCATOR,
    RUNTIME_DECISION_COUNT,
    RUNTIME_GENERATION,
    RUNTIME_REGION_COUNTS,
    RUNTIME_RNG,
    RUNTIME_STEP,
)
from saealib.exceptions import ConfigurationError, ValidationError
from saealib.execution.history import (
    History,
    record_generation,
    record_initial_evaluation,
)
from saealib.optimizer import ComponentProvider
from saealib.policies.feedback import _set_objectives_true
from saealib.population import Archive, ParetoArchive, Population, PopulationAttribute
from saealib.population.population import (
    OBJECTIVE_SOURCE_COLUMN,
    UNKNOWN_SOURCE_CODE,
)
from saealib.problem import Problem
from saealib.space import BoundsService, DenseNumericView


def _make_population(factory, attrs, capacity, problem, genomes=None):
    dense_view = problem.space.services.get("DenseNumericView")
    try:
        signature = inspect.signature(factory)
        accepts = any(
            name in signature.parameters
            for name in ("dense_numeric_view", "space", "genomes")
        ) or any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )
    except (TypeError, ValueError):
        accepts = True
        signature = None
    kwargs = {"attrs": attrs, "init_capacity": capacity}
    if accepts:
        kwargs["dense_numeric_view"] = dense_view
        if (
            signature is None
            or "space" in signature.parameters
            or any(
                parameter.kind == inspect.Parameter.VAR_KEYWORD
                for parameter in signature.parameters.values()
            )
        ):
            kwargs["space"] = problem.space
        if genomes is not None and (
            signature is None
            or "genomes" in signature.parameters
            or any(
                parameter.kind == inspect.Parameter.VAR_KEYWORD
                for parameter in signature.parameters.values()
            )
        ):
            kwargs["genomes"] = genomes
    population = factory(**kwargs)
    if isinstance(population, Population):
        # Factories without the optional service parameter still need the
        # resolved view attached before evaluation begins.
        population._dense_numeric_view = dense_view
    return population


def _resolved_dense_view(problem: Problem) -> DenseNumericView | None:
    return cast(
        DenseNumericView | None,
        problem.space.services.get("DenseNumericView"),
    )


def _provider_comparator(provider: ComponentProvider) -> Comparator | None:
    """Return the comparator the provider resolved for this plan."""
    comparator = getattr(provider, "comparator", None)
    return comparator if isinstance(comparator, Comparator) else None


def _create_pareto_archive(
    provider: ComponentProvider,
    attrs: list[PopulationAttribute],
    init_capacity: int,
    problem: Problem,
):
    """Create the Pareto archive and bind the problem's constraints to it."""
    pareto_archive = provider.algorithm.create_pareto_archive(
        attrs=attrs, init_capacity=init_capacity, problem=problem
    )
    if pareto_archive.__dict__.get("constraints") is None:
        pareto_archive.constraints = (
            list(problem.constraints) if problem.constraints else None
        )
    return pareto_archive


def _defaults_equal(a: object, b: object) -> bool:
    try:
        return bool(np.array_equal(np.asarray(a), np.asarray(b), equal_nan=True))
    except TypeError:
        return a == b


def _merge_required_attrs(
    attrs: list[PopulationAttribute],
    problem: Problem,
    provider: ComponentProvider,
) -> list[PopulationAttribute]:
    merged = list(attrs)
    if not any(attr.name == OBJECTIVE_SOURCE_COLUMN for attr in merged):
        merged.append(
            PopulationAttribute(
                OBJECTIVE_SOURCE_COLUMN, np.uint8, (), default=UNKNOWN_SOURCE_CODE
            )
        )
    by_name = {attr.name: attr for attr in merged}
    sources: list[PopulationAttribute] = []
    if provider.algorithm is not None:
        sources.extend(provider.algorithm.get_required_attrs(problem))
    comparator = _provider_comparator(provider)
    if comparator is not None:
        sources.extend(comparator.get_required_attrs(problem))
    for attr in sources:
        existing = by_name.get(attr.name)
        if existing is None:
            merged.append(attr)
            by_name[attr.name] = attr
            continue
        if (
            np.dtype(existing.dtype) != np.dtype(attr.dtype)
            or existing.shape != attr.shape
            or not _defaults_equal(existing.default, attr.default)
        ):
            raise ConfigurationError(
                f"Conflicting PopulationAttribute definitions for {attr.name!r}: "
                f"{existing} vs {attr}"
            )
    return merged


def _mark_initial_rows_true(*collections: Population) -> None:
    for collection in collections:
        values: dict[str, np.ndarray] = {}
        _set_objectives_true(collection, values, len(collection))
        if values:
            collection.update_rows(np.arange(len(collection)), values)


class Initializer(ABC):
    """Abstract base for classes that set up the initial optimization context."""

    def contract(self) -> ComponentContract:
        """Return the initializer contract."""
        candidate_count = Var(name="N")
        return ComponentContract(
            ports={
                "initializer": PortContract(
                    outputs=(
                        PortSpec(
                            name="population",
                            direction=PortDirection.OUTPUT,
                            data=DataSpec(
                                kind="Population",
                                bindings={"candidate_count": candidate_count},
                            ),
                            cardinality=MANY,
                        ),
                    ),
                ),
            },
            state=StateContract(
                reads=(
                    RUNTIME_RNG,
                    EVALUATIONS_COUNT,
                    RUNTIME_GENERATION,
                    RUNTIME_DECISION_COUNT,
                    POPULATIONS_MAIN,
                    ARCHIVES_MAIN,
                    ARCHIVES_PARETO,
                ),
                writes=(
                    POPULATIONS_MAIN,
                    ARCHIVES_MAIN,
                    EVALUATIONS_COUNT,
                    ARCHIVES_PARETO,
                    RUNTIME_GENERATION,
                    RUNTIME_STEP,
                    RUNTIME_REGION_COUNTS,
                    RUNTIME_RNG,
                ),
            ),
        )

    def _create_attrs(
        self, problem: Problem, provider: ComponentProvider
    ) -> list[PopulationAttribute]:
        attrs = [
            PopulationAttribute("x", float, (problem.dim,), default=np.nan),
            PopulationAttribute("f", float, (problem.n_obj,), default=np.nan),
            PopulationAttribute("g", float, (problem.n_constraints,), default=0.0),
            PopulationAttribute("cv", float, (), default=0.0),
            PopulationAttribute("id", np.int64, (), default=-1),
        ]
        return _merge_required_attrs(attrs, problem, provider)

    def _create_context(
        self,
        problem: Problem,
        provider: ComponentProvider,
        archive: Archive,
        pareto_archive: ParetoArchive,
        population: Population,
        rng: np.random.Generator,
    ) -> OptimizationState:
        comparator = _provider_comparator(provider)
        if isinstance(comparator, NSGA3Comparator) and comparator._rng is None:
            comparator.rng = rng.spawn(1)[0]

        return OptimizationState(
            problem=problem,
            population=population,
            archive=archive,
            pareto_archive=pareto_archive,
            rng=rng,
            comparator=comparator,
            fe=len(archive),
            gen=0,
            history=History(
                channels=getattr(provider, "history_channels", ("summary",))
            ),
        )

    @abstractmethod
    def initialize(
        self, provider: ComponentProvider, problem: Problem
    ) -> OptimizationState:
        """
        Initialize Population and Archive.

        Use them to generate an OptimizationState.

        Parameters
        ----------
        provider : ComponentProvider
            The component provider instance.
        problem : Problem
            The problem instance.

        Returns
        -------
        OptimizationState
            The optimization context.
        """
        pass


class GenomeInitializer(Initializer):
    """Initialize a population by sampling the problem's ``SearchSpace``."""

    def __init__(
        self, n_init_archive: int, n_init_population: int, seed: int | None = None
    ) -> None:
        if n_init_archive < 0 or n_init_population < 0:
            raise ValidationError("initial sizes must be non-negative")
        if n_init_population > n_init_archive:
            raise ValidationError("n_init_population cannot exceed n_init_archive")
        self.n_init_archive = n_init_archive
        self.n_init_population = n_init_population
        self.seed = seed

    def _create_attrs(
        self, problem: Problem, provider: ComponentProvider
    ) -> list[PopulationAttribute]:
        attrs = [
            PopulationAttribute("f", float, (problem.n_obj,), default=np.nan),
            PopulationAttribute("g", float, (problem.n_constraints,), default=0.0),
            PopulationAttribute("cv", float, (), default=0.0),
            PopulationAttribute("id", np.int64, (), default=-1),
        ]
        return _merge_required_attrs(attrs, problem, provider)

    def initialize(
        self, provider: ComponentProvider, problem: Problem
    ) -> OptimizationState:
        """Create the initial population and archive for a problem."""
        provider_seed = getattr(provider, "seed", None)
        rng = np.random.default_rng(
            provider_seed if provider_seed is not None else self.seed
        )
        attrs = self._create_attrs(problem, provider)
        empty_genomes = problem.space.sample(0, rng)
        population = _make_population(
            provider.algorithm.population_class,
            attrs,
            self.n_init_population,
            problem,
            empty_genomes,
        )
        archive = _make_population(
            provider.algorithm.archive_class,
            attrs,
            self.n_init_archive,
            problem,
            empty_genomes,
        )
        pareto_archive = _create_pareto_archive(
            provider, attrs, self.n_init_archive, problem
        )
        ctx = self._create_context(
            problem, provider, archive, pareto_archive, population, rng
        )

        genomes = problem.space.sample(self.n_init_archive, rng)
        validation = problem.space.validate(genomes)
        if not validation.valid:
            raise ValidationError(
                "SearchSpace.sample returned invalid genomes: "
                + "; ".join(validation.errors)
            )
        result = provider.evaluator.evaluate_batch(genomes, problem)
        ids = ctx.candidate_id_allocator.allocate(self.n_init_archive)

        for i in range(self.n_init_archive):
            data = {
                "genome": genomes.take([i]),
                "f": result.f[i],
                "g": result.g[i],
                "cv": float(result.cv[i]),
                "id": int(ids[i]),
            }
            archive.add(data)
            pareto_archive.add(data)
        _mark_initial_rows_true(archive, pareto_archive)

        record_initial_evaluation(ctx, result, ids)
        ctx.count_fe(self.n_init_archive)
        provider.dispatch(InitialEvaluationEndEvent(ctx=ctx, archive=archive))

        sorted_idx = ctx.comparator.rank_population(archive)
        archive_sorted = archive.extract(sorted_idx)
        archive.clear()
        archive._extend_internal(archive_sorted, preserve_ids=True)
        population._extend_internal(
            archive[: self.n_init_population], preserve_ids=True
        )
        record_generation(ctx)
        return ctx


class LHSInitializer(Initializer):
    """
    Latin Hypercube Sampling Initializer.

    Attributes
    ----------
    n_init_archive : int
        The number of individuals to initialize in the archive.
    n_init_population : int
        The number of individuals to initialize in the population.
    seed : int | None
        The seed for the random number generator.
    """

    def __init__(
        self, n_init_archive: int, n_init_population: int, seed: int | None = None
    ):
        if n_init_archive < 0 or n_init_population < 0:
            raise ValidationError("initial sizes must be non-negative")
        if n_init_population > n_init_archive:
            raise ValidationError("n_init_population cannot exceed n_init_archive")
        self.n_init_archive = n_init_archive
        self.n_init_population = n_init_population
        self.seed = seed

    def contract(self) -> ComponentContract:
        """Return the LHS initializer contract."""
        contract = super().contract()
        role = contract.ports["initializer"]
        output = role.outputs[0]
        return replace(
            contract,
            state=replace(
                contract.state,
                reads=(
                    *contract.state.reads,
                    RUNTIME_CANDIDATE_ID_ALLOCATOR,
                ),
                writes=(*contract.state.writes, RUNTIME_CANDIDATE_ID_ALLOCATOR),
            ),
            ports={
                **contract.ports,
                "initializer": replace(
                    role,
                    outputs=(
                        replace(
                            output,
                            data=replace(
                                output.data,
                                bindings={
                                    "candidate_count": Fixed(
                                        value=self.n_init_population
                                    )
                                },
                            ),
                        ),
                    ),
                ),
            },
        )

    def initialize(
        self, provider: ComponentProvider, problem: Problem
    ) -> OptimizationState:
        """
        Initialize Population and Archive with LHS samples.

        Parameters
        ----------
        provider : ComponentProvider
            The component provider instance.
        problem : Problem
            The problem instance.

        Returns
        -------
        OptimizationState
        """
        provider_seed = getattr(provider, "seed", None)
        rng = np.random.default_rng(
            provider_seed if provider_seed is not None else self.seed
        )
        attrs = self._create_attrs(problem, provider)

        population = _make_population(
            provider.algorithm.population_class,
            attrs,
            self.n_init_population,
            problem,
        )
        archive = _make_population(
            provider.algorithm.archive_class,
            attrs,
            self.n_init_archive,
            problem,
        )
        pareto_archive = _create_pareto_archive(
            provider, attrs, self.n_init_archive, problem
        )
        pareto_archive._dense_numeric_view = _resolved_dense_view(problem)

        ctx = self._create_context(
            problem, provider, archive, pareto_archive, population, rng
        )

        bounds_srv = cast(
            BoundsService, problem.space.services.require("BoundsService")
        )
        lb, ub = bounds_srv.bounds
        archive_x = scipy.stats.qmc.LatinHypercube(d=problem.dim, rng=rng).random(
            self.n_init_archive
        )
        archive_x = scipy.stats.qmc.scale(archive_x, lb, ub)

        provider.dispatch(InitialEvaluationStartEvent(ctx=ctx, candidates_x=archive_x))

        result = provider.evaluator.evaluate_batch(archive_x, problem)

        ids = ctx.candidate_id_allocator.allocate(self.n_init_archive)

        for i in range(self.n_init_archive):
            data = {
                "x": archive_x[i],
                "f": result.f[i],
                "g": result.g[i],
                "cv": float(result.cv[i]),
                "id": int(ids[i]),
            }
            archive.add(data)
            pareto_archive.add(data)
        _mark_initial_rows_true(archive, pareto_archive)

        record_initial_evaluation(ctx, result, ids)
        ctx.count_fe(self.n_init_archive)

        provider.dispatch(InitialEvaluationEndEvent(ctx=ctx, archive=archive))

        sorted_idx = ctx.comparator.rank_population(archive)
        archive_sorted = archive.extract(sorted_idx)
        archive.clear()
        archive._extend_internal(archive_sorted, preserve_ids=True)

        population._extend_internal(
            archive[: self.n_init_population], preserve_ids=True
        )
        record_generation(ctx)

        return ctx


class RandomInitializer(Initializer):
    """
    Random uniform sampling initializer.

    Attributes
    ----------
    n_init_archive : int
        The number of individuals to initialize in the archive.
    n_init_population : int
        The number of individuals to initialize in the population.
    seed : int | None
        The seed for the random number generator.
    """

    def __init__(
        self, n_init_archive: int, n_init_population: int, seed: int | None = None
    ):
        if n_init_archive < 0 or n_init_population < 0:
            raise ValidationError("initial sizes must be non-negative")
        if n_init_population > n_init_archive:
            raise ValidationError("n_init_population cannot exceed n_init_archive")
        self.n_init_archive = n_init_archive
        self.n_init_population = n_init_population
        self.seed = seed

    def contract(self) -> ComponentContract:
        """Return the random initializer contract."""
        contract = super().contract()
        role = contract.ports["initializer"]
        output = role.outputs[0]
        return replace(
            contract,
            state=replace(
                contract.state,
                reads=(
                    *contract.state.reads,
                    RUNTIME_CANDIDATE_ID_ALLOCATOR,
                ),
                writes=(*contract.state.writes, RUNTIME_CANDIDATE_ID_ALLOCATOR),
            ),
            ports={
                **contract.ports,
                "initializer": replace(
                    role,
                    outputs=(
                        replace(
                            output,
                            data=replace(
                                output.data,
                                bindings={
                                    "candidate_count": Fixed(
                                        value=self.n_init_population
                                    )
                                },
                            ),
                        ),
                    ),
                ),
            },
        )

    def initialize(
        self, provider: ComponentProvider, problem: Problem
    ) -> OptimizationState:
        """
        Initialize Population and Archive with uniform random samples.

        Parameters
        ----------
        provider : ComponentProvider
            The component provider instance.
        problem : Problem
            The problem instance.

        Returns
        -------
        OptimizationState
        """
        provider_seed = getattr(provider, "seed", None)
        rng = np.random.default_rng(
            provider_seed if provider_seed is not None else self.seed
        )
        attrs = self._create_attrs(problem, provider)

        population = _make_population(
            provider.algorithm.population_class,
            attrs,
            self.n_init_population,
            problem,
        )
        archive = _make_population(
            provider.algorithm.archive_class,
            attrs,
            self.n_init_archive,
            problem,
        )
        pareto_archive = _create_pareto_archive(
            provider, attrs, self.n_init_archive, problem
        )
        pareto_archive._dense_numeric_view = _resolved_dense_view(problem)

        ctx = self._create_context(
            problem, provider, archive, pareto_archive, population, rng
        )

        bounds_srv = cast(
            BoundsService, problem.space.services.require("BoundsService")
        )
        lb, ub = bounds_srv.bounds
        archive_x = rng.uniform(lb, ub, size=(self.n_init_archive, problem.dim))

        provider.dispatch(InitialEvaluationStartEvent(ctx=ctx, candidates_x=archive_x))

        result = provider.evaluator.evaluate_batch(archive_x, problem)

        ids = ctx.candidate_id_allocator.allocate(self.n_init_archive)

        for i in range(self.n_init_archive):
            data = {
                "x": archive_x[i],
                "f": result.f[i],
                "g": result.g[i],
                "cv": float(result.cv[i]),
                "id": int(ids[i]),
            }
            archive.add(data)
            pareto_archive.add(data)
        _mark_initial_rows_true(archive, pareto_archive)

        record_initial_evaluation(ctx, result, ids)
        ctx.count_fe(self.n_init_archive)

        provider.dispatch(InitialEvaluationEndEvent(ctx=ctx, archive=archive))

        sorted_idx = ctx.comparator.rank_population(archive)
        archive_sorted = archive.extract(sorted_idx)
        archive.clear()
        archive._extend_internal(archive_sorted, preserve_ids=True)

        population._extend_internal(
            archive[: self.n_init_population], preserve_ids=True
        )
        record_generation(ctx)

        return ctx


class SobolInitializer(Initializer):
    """
    Sobol quasi-random sequence initializer.

    Attributes
    ----------
    n_init_archive : int
        The number of individuals to initialize in the archive.
    n_init_population : int
        The number of individuals to initialize in the population.
    seed : int | None
        The seed for the random number generator.
    """

    def __init__(
        self, n_init_archive: int, n_init_population: int, seed: int | None = None
    ):
        if n_init_archive < 0 or n_init_population < 0:
            raise ValidationError("initial sizes must be non-negative")
        if n_init_population > n_init_archive:
            raise ValidationError("n_init_population cannot exceed n_init_archive")
        self.n_init_archive = n_init_archive
        self.n_init_population = n_init_population
        self.seed = seed

    def contract(self) -> ComponentContract:
        """Return the Sobol initializer contract."""
        contract = super().contract()
        role = contract.ports["initializer"]
        output = role.outputs[0]
        return replace(
            contract,
            state=replace(
                contract.state,
                reads=(
                    *contract.state.reads,
                    RUNTIME_CANDIDATE_ID_ALLOCATOR,
                ),
                writes=(*contract.state.writes, RUNTIME_CANDIDATE_ID_ALLOCATOR),
            ),
            ports={
                **contract.ports,
                "initializer": replace(
                    role,
                    outputs=(
                        replace(
                            output,
                            data=replace(
                                output.data,
                                bindings={
                                    "candidate_count": Fixed(
                                        value=self.n_init_population
                                    )
                                },
                            ),
                        ),
                    ),
                ),
            },
        )

    def initialize(
        self, provider: ComponentProvider, problem: Problem
    ) -> OptimizationState:
        """
        Initialize Population and Archive with scrambled Sobol samples.

        Parameters
        ----------
        provider : ComponentProvider
            The component provider instance.
        problem : Problem
            The problem instance.

        Returns
        -------
        OptimizationState
        """
        provider_seed = getattr(provider, "seed", None)
        rng = np.random.default_rng(
            provider_seed if provider_seed is not None else self.seed
        )
        attrs = self._create_attrs(problem, provider)

        population = _make_population(
            provider.algorithm.population_class,
            attrs,
            self.n_init_population,
            problem,
        )
        archive = _make_population(
            provider.algorithm.archive_class,
            attrs,
            self.n_init_archive,
            problem,
        )
        pareto_archive = _create_pareto_archive(
            provider, attrs, self.n_init_archive, problem
        )
        pareto_archive._dense_numeric_view = _resolved_dense_view(problem)

        ctx = self._create_context(
            problem, provider, archive, pareto_archive, population, rng
        )

        bounds_srv = cast(
            BoundsService, problem.space.services.require("BoundsService")
        )
        lb, ub = bounds_srv.bounds
        archive_x = scipy.stats.qmc.Sobol(
            d=problem.dim, scramble=True, seed=rng
        ).random(self.n_init_archive)
        archive_x = scipy.stats.qmc.scale(archive_x, lb, ub)

        provider.dispatch(InitialEvaluationStartEvent(ctx=ctx, candidates_x=archive_x))

        result = provider.evaluator.evaluate_batch(archive_x, problem)

        ids = ctx.candidate_id_allocator.allocate(self.n_init_archive)

        for i in range(self.n_init_archive):
            data = {
                "x": archive_x[i],
                "f": result.f[i],
                "g": result.g[i],
                "cv": float(result.cv[i]),
                "id": int(ids[i]),
            }
            archive.add(data)
            pareto_archive.add(data)
        _mark_initial_rows_true(archive, pareto_archive)

        record_initial_evaluation(ctx, result, ids)
        ctx.count_fe(self.n_init_archive)

        provider.dispatch(InitialEvaluationEndEvent(ctx=ctx, archive=archive))

        sorted_idx = ctx.comparator.rank_population(archive)
        archive_sorted = archive.extract(sorted_idx)
        archive.clear()
        archive._extend_internal(archive_sorted, preserve_ids=True)

        population._extend_internal(
            archive[: self.n_init_population], preserve_ids=True
        )
        record_generation(ctx)

        return ctx
