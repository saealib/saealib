"""Abstract base class for evolutionary algorithms."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Protocol, cast, runtime_checkable

import numpy as np

from saealib.core.contracts import (
    MANY,
    TRUE,
    ComponentContract,
    DataSpec,
    FeedbackBatch,
    FeedbackContract,
    LifecycleContract,
    PortContract,
    PortDirection,
    PortSpec,
    ProposalBatch,
    ServiceRequirement,
    StateContract,
)
from saealib.core.state import (
    POPULATIONS_MAIN,
    PROPOSALS_ID_ALLOCATOR,
    PROPOSALS_OFFSPRING,
    RUNTIME_RNG,
    ExecutionContext,
    StatePatch,
    StateView,
)
from saealib.exceptions import ValidationError
from saealib.population import Archive, ParetoArchive, Population, PopulationAttribute
from saealib.problem import Problem
from saealib.space import DenseNumericView


class Algorithm(ABC):
    """Base class for evolutionary algorithms."""

    def contract(self) -> ComponentContract:
        """Return the evolutionary-algorithm family contract."""
        return ComponentContract(
            ports={
                "proposer": PortContract(
                    outputs=(
                        PortSpec(
                            name="genomes",
                            direction=PortDirection.OUTPUT,
                            data=DataSpec(kind="Population"),
                            cardinality=MANY,
                            required_services=(
                                ServiceRequirement(name="BoundsService"),
                            ),
                        ),
                    ),
                ),
                "feedback_consumer": PortContract(
                    inputs=(
                        PortSpec(
                            name="feedback",
                            direction=PortDirection.INPUT,
                            data=DataSpec(kind="FeedbackBatch"),
                            cardinality=MANY,
                        ),
                    ),
                ),
            },
            state=StateContract(
                reads=(
                    POPULATIONS_MAIN,
                    PROPOSALS_OFFSPRING,
                    PROPOSALS_ID_ALLOCATOR,
                    RUNTIME_RNG,
                ),
                writes=(
                    POPULATIONS_MAIN,
                    PROPOSALS_ID_ALLOCATOR,
                    RUNTIME_RNG,
                ),
            ),
            lifecycle=LifecycleContract(
                feedback=FeedbackContract(accepted_channels=frozenset({TRUE}))
            ),
        )

    @abstractmethod
    def get_required_attrs(self, problem: Problem) -> list[PopulationAttribute]:
        """Return the list of attributes required for Population."""
        pass

    @property
    @abstractmethod
    def population_class(self) -> type[Population]:
        """Return the population class."""
        pass

    @property
    @abstractmethod
    def archive_class(self) -> type[Archive]:
        """Return the archive class."""
        pass

    @property
    def pareto_archive_class(self) -> type[ParetoArchive]:
        """Return the Pareto archive class."""
        return ParetoArchive

    def create_pareto_archive(
        self,
        attrs: list[PopulationAttribute],
        init_capacity: int,
        problem: Problem,
    ) -> ParetoArchive:
        """Create a ParetoArchive with the correct direction for the problem."""
        kwargs: dict[str, Any] = {
            "attrs": attrs,
            "init_capacity": init_capacity,
            "direction": problem.direction,
        }
        if not any(attr.name == "x" for attr in attrs):
            kwargs.update(
                {
                    "key_attr": "id",
                    "space": problem.space,
                    "genomes": problem.space.sample(0),
                    "dense_numeric_view": cast(
                        DenseNumericView | None,
                        problem.space.services.get("DenseNumericView"),
                    ),
                }
            )
        return self.pareto_archive_class(
            **kwargs,
        )

    @property
    def ask_notation(self) -> list[str] | None:
        r"""LaTeX notation lines describing the internal steps of :meth:`ask`.

        Returns ``None`` by default; :class:`~saealib.stages.AskStage` then
        renders a single collapsed ``\\State`` line.  Override to return a list
        of LaTeX math strings (one per logical step) so that
        ``AskStage.to_pseudocode(expand=True)`` expands them inside a
        ``\\Comment`` block.

        Example (GA)::

            return [
                r"$I_m \\leftarrow \\mathrm{select}(P,\\, n_{pair})$",
                r"$\\mathcal{Q} \\leftarrow \\mathrm{crossover}(P[I_m])$",
                r"$\\mathcal{Q} \\leftarrow \\mathrm{mutate}(\\mathcal{Q})$",
            ]
        """
        return None

    @abstractmethod
    def ask(
        self,
        request: ProposalRequest,
        state: StateView,
    ) -> ProposalBatch:
        """Generate one proposal batch from a read-only state view."""
        pass

    @property
    def tell_notation(self) -> list[str] | None:
        r"""LaTeX notation lines describing the internal steps of :meth:`tell`.

        Returns ``None`` by default; :class:`~saealib.stages.TellStage` then
        renders a single collapsed ``\State`` line.  Override to return a list
        of LaTeX math strings so that ``TellStage.to_pseudocode(expand=True)``
        expands them inside a ``\Comment`` block.
        """
        return None

    @abstractmethod
    def tell(
        self,
        feedback: FeedbackBatch,
        state: StateView,
    ) -> StatePatch:
        """Consume feedback and return the state changes it produced."""
        pass


@dataclass(frozen=True, kw_only=True)
class ProposalRequest:
    """The request-specific part of the proposer interface."""

    n_offspring: int | None = None

    def __post_init__(self) -> None:
        if self.n_offspring is None:
            return
        if isinstance(self.n_offspring, (bool, np.bool_)) or not isinstance(
            self.n_offspring, (int, np.integer)
        ):
            raise ValidationError("n_offspring must be a non-negative integer or None")
        n_offspring = int(self.n_offspring)
        if n_offspring < 0:
            raise ValidationError("n_offspring must be a non-negative integer or None")
        object.__setattr__(self, "n_offspring", n_offspring)


@runtime_checkable
class Proposer(Protocol):
    """Generate proposals without owning the feedback lifecycle."""

    def ask(self, request: ProposalRequest, state: StateView) -> ProposalBatch:
        """Return a batch of candidates and its feedback requirement."""
        ...


@runtime_checkable
class FeedbackConsumer(Protocol):
    """Consume feedback independently from proposal generation."""

    def tell(self, feedback: FeedbackBatch, state: StateView) -> StatePatch:
        """Return state changes caused by a feedback batch."""
        ...


class AskTellAlgorithm(Algorithm):
    """Stable combined proposer and feedback-consumer API."""


def algorithm_context(state: StateView) -> ExecutionContext:
    """Return the execution context bound to a built-in algorithm call."""
    return state.context
