"""Metadata defaults for Algorithm and the algorithm-slot duck typing.

The base ``Algorithm`` provides standard ``get_required_attrs``,
``population_class``, and ``archive_class`` defaults, so a minimal algorithm
implements only ``ask``/``tell``.  The algorithm slot is duck-typed: no
execution path requires ``Algorithm`` inheritance or ``ask``/``tell``.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np

from saealib import (
    CrossoverBLXAlpha,
    DirectStrategy,
    LHSInitializer,
    MutationUniform,
    Optimizer,
    ParetoArchive,
    Population,
    PopulationAttribute,
    Problem,
    SequentialSelection,
    SingleObjectiveComparator,
    Termination,
    TruncationSelection,
    max_gen,
)
from saealib.algorithms.base import Algorithm, algorithm_context
from saealib.algorithms.deap_algorithm import DeapGenerateUpdateAlgorithm
from saealib.algorithms.ga import GA
from saealib.algorithms.genome_ga import GenomeGA
from saealib.algorithms.pso import PSO
from saealib.algorithms.pymoo_algorithm import PymooAlgorithm
from saealib.core.contracts import (
    ComponentContract,
    FeedbackBatch,
    FeedbackRequirement,
    ProposalBatch,
    ProposalRelations,
    StateContract,
)
from saealib.core.state import RUNTIME_GENERATION, StatePatch
from saealib.pipeline import Pipeline, Stage
from saealib.population import Archive
from saealib.strategies.base import TemplateStrategy


def _problem() -> Problem:
    return Problem(
        func=lambda x: np.array([np.sum(np.asarray(x) ** 2)]),
        dim=2,
        n_obj=1,
        direction=np.array([-1.0]),
        lb=[-2.0] * 2,
        ub=[2.0] * 2,
        comparator=SingleObjectiveComparator(),
    )


class _KeepPopulationAlgorithm(Algorithm):
    """Minimal ask/tell-only algorithm that keeps the population as-is."""

    def ask(self, request, state) -> ProposalBatch:
        ctx = algorithm_context(state)
        candidates = ctx.population.empty_like(capacity=0)
        return ProposalBatch.from_allocator(
            ctx.proposal_id_allocator,
            candidates=candidates,
            relations=ProposalRelations(row_count=0),
            requirements=FeedbackRequirement(quantities=()),
        )

    def tell(self, feedback: FeedbackBatch, state) -> StatePatch:
        del feedback, state
        return StatePatch(writes={})


def test_minimal_ask_tell_only_algorithm_uses_standard_population_and_archive() -> None:
    algorithm = _KeepPopulationAlgorithm()
    assert algorithm.get_required_attrs(_problem()) == []
    assert algorithm.population_class is Population
    assert algorithm.archive_class is Archive

    optimizer = (
        Optimizer(_problem(), seed=1)
        .set_initializer(LHSInitializer(4, 4, seed=1))
        .set_algorithm(algorithm)
        .set_strategy(DirectStrategy())
        .set_termination(Termination(max_gen(2)))
    )
    result = optimizer.run()

    assert result.gen == 2
    assert result.history is not None


def test_metadata_defaults_are_overridable_for_extra_attributes() -> None:
    extra = PopulationAttribute("velocity", np.float64, (2,), default=0.0)

    class _ExtraAttrsAlgorithm(_KeepPopulationAlgorithm):
        def get_required_attrs(self, problem: Problem):
            return [extra]

    optimizer = (
        Optimizer(_problem(), seed=2)
        .set_initializer(LHSInitializer(4, 4, seed=2))
        .set_algorithm(_ExtraAttrsAlgorithm())
        .set_strategy(DirectStrategy())
        .set_termination(Termination(max_gen(1)))
    )
    state = next(iter(optimizer.iterate()))

    assert "velocity" in state.population.schema


def test_metadata_defaults_are_overridable_for_custom_collections() -> None:
    class _TrackedPopulation(Population):
        pass

    class _CustomCollectionsAlgorithm(_KeepPopulationAlgorithm):
        @property
        def population_class(self) -> type[Population]:
            return _TrackedPopulation

    optimizer = (
        Optimizer(_problem(), seed=3)
        .set_initializer(LHSInitializer(4, 4, seed=3))
        .set_algorithm(_CustomCollectionsAlgorithm())
        .set_strategy(DirectStrategy())
        .set_termination(Termination(max_gen(1)))
    )
    states = [state for state in optimizer.iterate()]

    assert states, "custom collections must not prevent a run"
    assert states[-1].population.__class__ is _TrackedPopulation
    assert states[-1].archive.__class__ is Archive


def test_builtin_metadata_overrides_restate_the_defaults() -> None:
    for algorithm_class in (GA, GenomeGA):
        assert "get_required_attrs" not in vars(algorithm_class)
        assert "population_class" not in vars(algorithm_class)
        assert "archive_class" not in vars(algorithm_class)

    for algorithm_class in (PSO, PymooAlgorithm, DeapGenerateUpdateAlgorithm):
        assert "get_required_attrs" in vars(algorithm_class)
        assert "population_class" not in vars(algorithm_class)
        assert "archive_class" not in vars(algorithm_class)


def test_base_defaults_match_the_documented_contract() -> None:
    algorithm = _KeepPopulationAlgorithm()
    assert Algorithm.get_required_attrs(algorithm, _problem()) == []
    assert Algorithm.population_class.fget(algorithm) is Population
    assert Algorithm.archive_class.fget(algorithm) is Archive


def test_ga_declares_every_held_child_component() -> None:
    ga = GA(
        crossover=CrossoverBLXAlpha(0.9, 0.5),
        mutation=MutationUniform(0.1),
        parent_selection=SequentialSelection(),
        survivor_selection=TruncationSelection(),
    )
    declared = {part.name for part in ga.contract().parts}
    for name, value in vars(ga).items():
        child_contract = getattr(value, "contract", None)
        if not callable(child_contract):
            continue
        assert name in declared, f"GA holds undeclared child component {name!r}"


@__import__("pytest").mark.parametrize(
    "population_class_name", ["population_class", "archive_class"]
)
def test_initializer_uses_the_algorithm_metadata_hooks(population_class_name) -> None:
    observed: list[str] = []

    class _RecordingAlgorithm(_KeepPopulationAlgorithm):
        @property
        def population_class(self):
            observed.append("population_class")
            return Population

        @property
        def archive_class(self):
            observed.append("archive_class")
            return Archive

    optimizer = (
        Optimizer(_problem(), seed=5)
        .set_initializer(LHSInitializer(4, 4, seed=5))
        .set_algorithm(_RecordingAlgorithm())
        .set_strategy(DirectStrategy())
        .set_termination(Termination(max_gen(1)))
    )
    next(iter(optimizer.iterate()))

    assert population_class_name in observed


class _ControlAlgorithm:
    """Algorithm-slot object with only the initializer metadata hooks."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    @property
    def population_class(self) -> type[Population]:
        self.calls.append("population_class")
        return Population

    @property
    def archive_class(self) -> type:
        self.calls.append("archive_class")
        return Archive

    def get_required_attrs(self, problem: Problem):
        self.calls.append("get_required_attrs")
        return []

    def create_pareto_archive(self, attrs, init_capacity, problem):
        self.calls.append("create_pareto_archive")
        return ParetoArchive(
            attrs=attrs,
            init_capacity=init_capacity,
            direction=problem.direction,
        )

    def contract(self) -> ComponentContract:
        return ComponentContract()


class _CountStage(Stage):
    name = "count_generation"
    label = "Count generation"

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[str] = []

    def contract(self) -> ComponentContract:
        return ComponentContract(
            state=StateContract(
                reads=(RUNTIME_GENERATION,), writes=(RUNTIME_GENERATION,)
            )
        )

    def execute(self, state):
        self.calls.append("execute")
        return state.replace(gen=state.gen + 1)


class _ControlStrategy(TemplateStrategy):
    def __init__(self) -> None:
        super().__init__(Pipeline(name="control", steps=[]))
        self.stages: list[_CountStage] = []

    def build_pipeline(self, provider) -> Pipeline:
        stage = _CountStage()
        self.stages.append(stage)
        return Pipeline(name="control", steps=[stage])


def test_algorithm_object_without_ask_tell_runs_from_custom_strategy() -> None:
    control = _ControlAlgorithm()
    strategy = _ControlStrategy()

    optimizer = (
        Optimizer(_problem(), seed=4)
        .set_initializer(LHSInitializer(4, 4, seed=4))
        .set_algorithm(cast(Any, control))
        .set_strategy(strategy)
        .set_termination(Termination(max_gen(2)))
    )
    result = optimizer.run()

    assert result.gen == 2
    assert strategy.stages[0].calls == ["execute", "execute"]
    assert "population_class" in control.calls
    assert "create_pareto_archive" in control.calls
    assert not hasattr(control, "ask")
    assert not hasattr(control, "tell")
