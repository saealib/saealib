"""Island-model execution for independent optimizers."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

import numpy as np

from saealib.context import OptimizationState
from saealib.exceptions import ValidationError
from saealib.population.population import (
    OBJECTIVE_SOURCE_COLUMN,
    UNKNOWN_SOURCE_CODE,
)

if TYPE_CHECKING:
    from saealib.optimizer import Optimizer


__all__ = ["IslandModel"]


class IslandModel:
    """Run optimizers in parallel and periodically exchange individuals.

    Migration selects the best individuals from a source island using its
    problem comparator and overwrites the worst rows of the target population.
    The target rows' existing ``id`` values are intentionally retained: an
    island migration is a value transfer, not a candidate-identity transfer.

    Parameters
    ----------
    optimizers : iterable of Optimizer
        Configured optimizers, one per island.
    topology : {"ring", "fully_connected"} or array-like, optional
        Directed migration topology. An array-like value is an explicit
        adjacency matrix, where a true entry ``[source, target]`` enables
        migration in that direction.
    migration_interval : int, optional
        Migration period in generations. Non-positive values disable migration.
    migration_size : int, optional
        Number of individuals sent along each enabled edge.
    adjacency : array-like, optional
        Explicit adjacency matrix, as an alternative to ``topology``.
    """

    def __init__(
        self,
        optimizers: Iterable[Optimizer] = (),
        topology: str | Iterable[Iterable[object]] = "ring",
        migration_interval: int = 1,
        migration_size: int = 1,
        *,
        adjacency: Iterable[Iterable[object]] | None = None,
    ) -> None:
        self.optimizers = tuple(optimizers)
        if any(not hasattr(optimizer, "strategy") for optimizer in self.optimizers):
            raise ValidationError("optimizers must be configured Optimizer instances")
        if not isinstance(migration_interval, (int, np.integer)):
            raise ValidationError("migration_interval must be an integer")
        if not isinstance(migration_size, (int, np.integer)) or migration_size <= 0:
            raise ValidationError("migration_size must be a positive integer")
        if adjacency is not None and not (
            isinstance(topology, str) and topology == "ring"
        ):
            raise ValidationError("specify either topology or adjacency, not both")
        topology_value = adjacency if adjacency is not None else topology
        self._topology = topology_value
        self.migration_interval = int(migration_interval)
        self.migration_size = int(migration_size)
        self._adjacency = self._validate_topology(topology_value)
        self._validate_compatibility()
        self.migration_events: list[tuple[int, int]] = []

    def _validate_compatibility(self) -> None:
        """Reject island problems whose evaluated rows cannot be exchanged."""
        if not self.optimizers:
            return
        reference = getattr(self.optimizers[0], "problem", None)
        if reference is None:
            return
        for index, optimizer in enumerate(self.optimizers[1:], start=1):
            problem = getattr(optimizer, "problem", None)
            if problem is None:
                continue
            for attribute in ("dim", "n_obj", "n_constraints"):
                expected = getattr(reference, attribute)
                actual = getattr(problem, attribute)
                if actual != expected:
                    raise ValidationError(
                        f"island {index} problem {attribute}={actual!r} "
                        f"does not match island 0 ({expected!r})"
                    )
            if not np.array_equal(problem.direction, reference.direction):
                raise ValidationError(
                    f"island {index} problem direction={problem.direction!r} "
                    f"does not match island 0 ({reference.direction!r})"
                )
        if self.migration_interval > 0:
            from saealib.algorithms.pymoo_algorithm import PymooAlgorithm

            for index, optimizer in enumerate(self.optimizers):
                if isinstance(getattr(optimizer, "algorithm", None), PymooAlgorithm):
                    raise ValidationError(
                        f"island {index} uses engine-mode PymooAlgorithm, which "
                        "is incompatible with enabled migration"
                    )

    def _validate_topology(
        self, topology: str | Iterable[Iterable[object]]
    ) -> np.ndarray:
        n_islands = len(self.optimizers)
        if isinstance(topology, str):
            if topology not in {"ring", "fully_connected"}:
                raise ValidationError(f"invalid island topology: {topology!r}")
            if topology == "ring":
                adjacency = np.zeros((n_islands, n_islands), dtype=bool)
                if n_islands > 1:
                    adjacency[
                        np.arange(n_islands),
                        (np.arange(n_islands) + 1) % n_islands,
                    ] = True
                return adjacency
            adjacency = np.ones((n_islands, n_islands), dtype=bool)
            np.fill_diagonal(adjacency, False)
            return adjacency

        try:
            adjacency = np.asarray(topology)
        except (TypeError, ValueError) as exc:
            raise ValidationError(
                "explicit topology must be a square adjacency matrix"
            ) from exc
        if adjacency.ndim != 2 or adjacency.shape != (n_islands, n_islands):
            raise ValidationError(
                "explicit topology shape must match the number of optimizers"
            )
        if adjacency.dtype.kind not in "biuf":
            raise ValidationError(
                "explicit topology must contain numeric or boolean values"
            )
        if not np.all(np.isfinite(adjacency)):
            raise ValidationError("explicit topology cannot contain non-finite values")
        return adjacency.astype(bool, copy=True)

    def run(self) -> tuple[OptimizationState, ...]:
        """Run all islands, returning their final states."""
        self.migration_events = []
        if len(self.optimizers) <= 1 or self.migration_interval <= 0:
            return tuple(optimizer.run() for optimizer in self.optimizers)

        generators = [optimizer.iterate() for optimizer in self.optimizers]
        states = [next(generator) for generator in generators]
        live = [True] * len(generators)
        last_migrations: dict[tuple[int, int], int] = {}
        while any(live):
            for source, generator in enumerate(generators):
                if not live[source]:
                    continue
                try:
                    states[source] = next(generator)
                except StopIteration:
                    live[source] = False
                    continue
                self._migrate_ready(states, live, last_migrations)
        return tuple(states)

    def _migrate_ready(
        self,
        states: list[OptimizationState],
        live: list[bool],
        last_migrations: dict[tuple[int, int], int],
    ) -> None:
        """Apply migrations once both endpoints have reached a period."""
        migrations = []
        for source, target in np.argwhere(self._adjacency):
            source, target = int(source), int(target)
            if not live[source] or not live[target]:
                continue
            generation = min(states[source].gen, states[target].gen)
            generation -= generation % self.migration_interval
            edge = (source, target)
            if generation > last_migrations.get(edge, 0):
                source_population = states[source].population
                target_population = states[target].population
                count = min(
                    self.migration_size,
                    len(source_population),
                    len(target_population),
                )
                if count:
                    best = states[source].comparator.sort_population(source_population)[
                        :count
                    ]
                    worst = states[target].comparator.sort_population(
                        target_population
                    )[-count:]
                    values = self._migration_values(
                        source_population, target_population, best
                    )
                    migrations.append((target_population, worst.copy(), values))
                last_migrations[edge] = generation
                self.migration_events.append(edge)
        for target_population, target_indices, values in migrations:
            target_population.update_rows(target_indices, values)

    def _migrate(self, source: OptimizationState, target: OptimizationState) -> None:
        source_population = source.population
        target_population = target.population
        count = min(self.migration_size, len(source_population), len(target_population))
        if count == 0:
            return
        best = source.comparator.sort_population(source_population)[:count]
        worst = target.comparator.sort_population(target_population)[-count:]
        values = self._migration_values(source_population, target_population, best)
        target_population.update_rows(worst, values)

    @staticmethod
    def _migration_values(source, target, indices: np.ndarray) -> dict[str, np.ndarray]:
        values: dict[str, np.ndarray] = {}
        for name, target_attr in target.schema.items():
            if name == "id":
                continue
            if name not in source.schema:
                if name == OBJECTIVE_SOURCE_COLUMN:
                    values[name] = np.full(
                        (len(indices), *target_attr.shape),
                        UNKNOWN_SOURCE_CODE,
                        dtype=target_attr.dtype,
                    )
                continue
            source_values = source.get_array(name)[indices]
            expected_shape = (len(indices), *target_attr.shape)
            if source_values.shape != expected_shape:
                raise ValidationError(
                    f"migration column {name!r} shape {source_values.shape!r} "
                    f"does not match target schema shape {expected_shape!r}"
                )
            values[name] = np.asarray(source_values, dtype=target_attr.dtype).copy()
        return values
