"""Selection operators for evolutionary algorithms."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np

from saealib.core.contracts import (
    MANY,
    ComponentContract,
    DataSpec,
    PortContract,
    PortDirection,
    PortSpec,
    ServiceRequirement,
    StateContract,
    Var,
)
from saealib.core.state import RUNTIME_RNG
from saealib.exceptions import ValidationError
from saealib.population import Population
from saealib.registry import register

if TYPE_CHECKING:
    from saealib.context import OptimizationState


class ParentSelection(ABC):
    """Base class for parent selection operators."""

    def contract(self) -> ComponentContract:
        """Return the parent-selection contract."""
        parents = DataSpec(
            kind="Ordering",
            bindings={"parent_count": Var(name="P")},
        )
        return ComponentContract(
            ports={
                "parent_selection": PortContract(
                    inputs=(
                        PortSpec(
                            name="population",
                            direction=PortDirection.INPUT,
                            data=DataSpec(kind="Population"),
                            cardinality=MANY,
                        ),
                    ),
                    outputs=(
                        PortSpec(
                            name="parents",
                            direction=PortDirection.OUTPUT,
                            data=parents,
                            cardinality=MANY,
                        ),
                    ),
                )
            }
        )

    @abstractmethod
    def select(
        self,
        population: Population,
        n_pair: int,
        n_parents: int,
        rng: np.random.Generator = np.random.default_rng(),
        *,
        ctx: OptimizationState | None,
    ) -> np.ndarray:
        """
        Select parents for reproduction.

        Parameters
        ----------
        population : Population
            Population to select from.
        n_pair : int
            Number of pairs to select.
        n_parents : int
            Number of parents per pair.
        rng : np.random.Generator, optional
            Random number generator, by default np.random.default_rng()
        ctx : OptimizationState or None
            Optimization context, providing the comparator and runtime
            random generator. Selection requires comparison and randomness,
            so every concrete implementation raises ValidationError when
            ``ctx`` is ``None``.

        Returns
        -------
        np.ndarray
            Selected parent indices. shape = (n_pair, n_parents)

        Raises
        ------
        ValidationError
            If ``ctx`` is ``None``.
        """
        pass


class TournamentSelection(ParentSelection):
    """
    Tournament selection operator.

    References
    ----------
    :cite:`miller1995tournament`: Miller, B. L., & Goldberg, D. E. (1995).
    Genetic algorithms, tournament selection, and the effects of noise.
    *Complex Systems*, 9(3), 193-212.

    :cite:`blickle1996selection`: Blickle, T., & Thiele, L. (1996). A
    comparison of selection schemes used in evolutionary algorithms.
    *Evolutionary Computation*, 4(4), 361-394. (Basis for the
    ``tournament_size`` design.)
    """

    def __init__(self, tournament_size: int):
        super().__init__()
        self.tournament_size = tournament_size

    def contract(self) -> ComponentContract:
        """Return the tournament-selection contract."""
        base = super().contract()
        role = base.ports["parent_selection"]
        return replace(
            base,
            ports={
                "parent_selection": replace(
                    role,
                    inputs=(
                        replace(
                            role.inputs[0],
                            required_services=(
                                ServiceRequirement(name="ComparisonService"),
                            ),
                        ),
                    ),
                )
            },
        )

    def select(
        self,
        population: Population,
        n_pair: int,
        n_parents: int,
        rng: np.random.Generator = np.random.default_rng(),
        *,
        ctx: OptimizationState | None,
    ) -> np.ndarray:
        """
        Execute tournament selection.

        Parameters
        ----------
        population : Population
            Population to select from.
        n_pair : int
            Number of pairs to select.
        n_parents : int
            Number of parents per pair.
        rng : np.random.Generator, optional
            Random number generator, by default np.random.default_rng()
        ctx : OptimizationState or None
            Optimization context, providing the comparator and runtime
            random generator.

        Returns
        -------
        np.ndarray
            Selected parent indices. shape = (n_pair, n_parents)

        Raises
        ------
        ValidationError
            If ``ctx`` is ``None``.
        """
        if ctx is None:
            raise ValidationError("TournamentSelection requires ctx.")
        n_pop = len(population)
        cmp = ctx.comparator
        selected_idx = np.zeros((n_pair, n_parents), dtype=int)
        for i in range(n_pair):
            for j in range(n_parents):
                tournament_idx = rng.choice(
                    n_pop, size=self.tournament_size, replace=False
                )
                best_idx = tournament_idx[0]
                for idx in tournament_idx[1:]:
                    if (
                        cmp.compare_population(
                            population,
                            idx,
                            best_idx,
                        )
                        < 0
                    ):
                        best_idx = idx
                selected_idx[i, j] = best_idx
        return selected_idx


@register()
class SequentialSelection(ParentSelection):
    """Sequential selection operator."""

    def __init__(self):
        super().__init__()

    def select(
        self,
        population: Population,
        n_pair: int,
        n_parents: int,
        rng: np.random.Generator = np.random.default_rng(),
        *,
        ctx: OptimizationState | None,
    ) -> np.ndarray:
        """
        Execute sequential selection.

        Assign parents sequentially without any fitness-based comparison.

        Parameters
        ----------
        population : Population
            Population to select from.
        n_pair : int
            Number of pairs to select.
        n_parents : int
            Number of parents per pair.
        rng : np.random.Generator, optional
            Random number generator, by default np.random.default_rng()
        ctx : OptimizationState or None
            Optimization context. Required even though sequential selection
            performs no comparison.

        Returns
        -------
        np.ndarray
            Selected parent indices. shape = (n_pair, n_parents)

        Raises
        ------
        ValidationError
            If ``ctx`` is ``None``.
        """
        if ctx is None:
            raise ValidationError("SequentialSelection requires ctx.")
        selected_idx = np.zeros((n_pair, n_parents), dtype=int)
        i_grid, j_grid = np.meshgrid(
            np.arange(n_pair), np.arange(n_parents), indexing="ij"
        )
        selected_idx = i_grid * n_parents + j_grid
        return selected_idx


class LinearRankSelection(ParentSelection):
    """
    Parent selection using linear rank-based probabilities.

    Selection probability is proportional to rank: the best individual has
    the highest probability and the worst has the lowest. This avoids
    numerical issues with raw fitness values (negative values, scale
    sensitivity) while preserving a fitness-proportionate behaviour.
    """

    def __init__(self):
        super().__init__()

    def contract(self) -> ComponentContract:
        """Return the linear-rank-selection contract."""
        base = super().contract()
        role = base.ports["parent_selection"]
        return replace(
            base,
            ports={
                "parent_selection": replace(
                    role,
                    inputs=(
                        replace(
                            role.inputs[0],
                            required_services=(
                                ServiceRequirement(name="ComparisonService"),
                            ),
                        ),
                    ),
                )
            },
        )

    def select(
        self,
        population: Population,
        n_pair: int,
        n_parents: int,
        rng: np.random.Generator = np.random.default_rng(),
        *,
        ctx: OptimizationState | None,
    ) -> np.ndarray:
        """
        Execute linear rank selection.

        Parameters
        ----------
        population : Population
            Population to select from.
        n_pair : int
            Number of pairs to select.
        n_parents : int
            Number of parents per pair.
        rng : np.random.Generator, optional
            Random number generator, by default np.random.default_rng()
        ctx : OptimizationState or None
            Optimization context, providing the comparator and runtime
            random generator.

        Returns
        -------
        np.ndarray
            Selected parent indices. shape = (n_pair, n_parents)

        Raises
        ------
        ValidationError
            If ``ctx`` is ``None``.
        """
        if ctx is None:
            raise ValidationError("LinearRankSelection requires ctx.")
        n_pop = len(population)
        sorted_idx = ctx.comparator.sort_population(population)
        weights = np.arange(n_pop, 0, -1, dtype=float)
        probs = weights / weights.sum()
        rank_probs = np.empty(n_pop)
        rank_probs[sorted_idx] = probs
        chosen = rng.choice(n_pop, size=n_pair * n_parents, p=rank_probs)
        return chosen.reshape(n_pair, n_parents)


class SurvivorSelection(ABC):
    """Base class for survivor selection operators."""

    def contract(self) -> ComponentContract:
        """Return the survivor-selection contract."""
        return ComponentContract(
            ports={
                "survivor_selection": PortContract(
                    inputs=(
                        PortSpec(
                            name="pool",
                            direction=PortDirection.INPUT,
                            data=DataSpec(kind="Population"),
                            cardinality=MANY,
                        ),
                    ),
                    outputs=(
                        PortSpec(
                            name="survivors",
                            direction=PortDirection.OUTPUT,
                            data=DataSpec(kind="Ordering"),
                            cardinality=MANY,
                        ),
                    ),
                )
            }
        )

    @abstractmethod
    def select(
        self,
        pool: Population,
        n_survivors: int,
        *,
        ctx: OptimizationState | None,
    ) -> np.ndarray:
        """
        Select survivors from the selection pool.

        The pool is a merged Population (e.g., parents + offspring) constructed
        by the Algorithm. The replacement strategy (μ+λ or μ,λ) is determined
        by the Algorithm, not by this class.

        Parameters
        ----------
        pool : Population
            Selection pool to choose survivors from.
        n_survivors : int
            Number of survivors to select.
        ctx : OptimizationState or None
            Optimization context, providing the comparator and runtime
            random generator. Selection requires comparison and randomness,
            so every concrete implementation raises ValidationError when
            ``ctx`` is ``None``.

        Returns
        -------
        np.ndarray
            Indices of selected survivors in the pool.

        Raises
        ------
        ValidationError
            If ``ctx`` is ``None``.
        """
        pass


@register()
class TruncationSelection(SurvivorSelection):
    """
    Truncation selection operator.

    Attributes
    ----------
    randomize_ties : bool
        When True, individuals tied at the truncation boundary (equal under
        ``comparator.compare_population``) are shuffled before truncating,
        matching pymoo's ``randomized_argsort`` treatment of the splitting
        front. When False (default), ties keep the order returned by
        ``comparator.rank_population`` (deterministic).
    """

    def __init__(self, randomize_ties: bool = False):
        """
        Initialize truncation selection operator.

        Parameters
        ----------
        randomize_ties : bool, optional
            Randomize the order of individuals tied at the truncation
            boundary, by default False.
        """
        self.randomize_ties = randomize_ties

    def contract(self) -> ComponentContract:
        """Return the truncation-selection contract."""
        base = super().contract()
        role = base.ports["survivor_selection"]
        state = (
            StateContract(reads=(RUNTIME_RNG,), writes=(RUNTIME_RNG,))
            if self.randomize_ties
            else StateContract()
        )
        return replace(
            base,
            state=state,
            ports={
                "survivor_selection": replace(
                    role,
                    inputs=(
                        replace(
                            role.inputs[0],
                            required_services=(
                                ServiceRequirement(name="ComparisonService"),
                            ),
                        ),
                    ),
                )
            },
        )

    def select(
        self,
        pool: Population,
        n_survivors: int,
        *,
        ctx: OptimizationState | None,
    ) -> np.ndarray:
        """
        Select survivors by truncating the sorted pool.

        Parameters
        ----------
        pool : Population
            Selection pool to choose survivors from.
        n_survivors : int
            Number of survivors to select.
        ctx : OptimizationState or None
            Optimization context, providing the comparator and runtime
            random generator.

        Returns
        -------
        np.ndarray
            Indices of selected survivors in the pool.

        Raises
        ------
        ValidationError
            If ``ctx`` is ``None``.
        """
        if ctx is None:
            raise ValidationError("TruncationSelection requires ctx.")
        cmp = ctx.comparator
        sorted_idx = cmp.rank_population(pool)
        if not self.randomize_ties or n_survivors >= len(sorted_idx):
            return sorted_idx[:n_survivors]

        boundary = sorted_idx[n_survivors - 1]
        lo = n_survivors - 1
        while (
            lo > 0 and cmp.compare_population(pool, sorted_idx[lo - 1], boundary) == 0
        ):
            lo -= 1
        hi = n_survivors
        while (
            hi < len(sorted_idx)
            and cmp.compare_population(pool, sorted_idx[hi], boundary) == 0
        ):
            hi += 1
        if hi - lo <= 1:
            return sorted_idx[:n_survivors]

        tie_block = sorted_idx[lo:hi].copy()
        ctx.rng.shuffle(tie_block)
        result = sorted_idx.copy()
        result[lo:hi] = tie_block
        return result[:n_survivors]
