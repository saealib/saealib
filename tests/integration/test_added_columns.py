"""Columns added or dropped between generations while a run continues."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from saealib import (
    GA,
    PSO,
    CrossoverBLXAlpha,
    LHSInitializer,
    MutationUniform,
    Optimizer,
    SequentialSelection,
    Termination,
    TruncationSelection,
    max_gen,
)
from saealib.context import OptimizationState
from saealib.exceptions import ValidationError
from saealib.identity import PopulationAttribute
from saealib.problem import Problem
from saealib.strategies import (
    DirectStrategy,
    GenerationBasedStrategy,
    IndividualBasedStrategy,
    PreSelectionStrategy,
    SteadyStateStrategy,
)

COLUMN = "probe_column"
DEFAULT = 3.5
COLLECTIONS = ("population", "archive", "pareto_archive")


def _sphere(x: Any) -> np.ndarray:
    return np.array([float(np.sum(np.asarray(x) ** 2))], dtype=np.float64)


def _problem() -> Problem:
    return Problem(
        func=_sphere,
        dim=2,
        n_obj=1,
        direction=np.array([-1.0]),
        lb=[-5.0, -5.0],
        ub=[5.0, 5.0],
    )


def _ga() -> GA:
    return GA(
        crossover=CrossoverBLXAlpha(prob=0.9, alpha=0.4),
        mutation=MutationUniform(prob_var=0.1),
        parent_selection=SequentialSelection(),
        survivor_selection=TruncationSelection(),
    )


def _optimizer(algorithm: Any, strategy: Any, n_gen: int = 4) -> Optimizer:
    return (
        Optimizer(_problem(), seed=0)
        .set_initializer(LHSInitializer(6, 6))
        .set_algorithm(algorithm)
        .set_strategy(strategy)
        .set_termination(Termination(max_gen(n_gen)))
    )


def _spec() -> PopulationAttribute:
    return PopulationAttribute(COLUMN, np.float64, (), default=DEFAULT)


def _add_everywhere(state: OptimizationState) -> None:
    for name in COLLECTIONS:
        getattr(state, name).add_columns([_spec()])


def _assert_column_kept(state: OptimizationState) -> None:
    for name in COLLECTIONS:
        collection = getattr(state, name)
        assert COLUMN in collection.schema
        values = collection.get_array(COLUMN)
        assert len(values) == len(collection)
        np.testing.assert_array_equal(values, np.full(len(collection), DEFAULT))


STRATEGIES = {
    "direct": DirectStrategy,
    "steady_state": SteadyStateStrategy,
    "individual_based": lambda: IndividualBasedStrategy(evaluation_ratio=0.5),
    "generation_based": lambda: GenerationBasedStrategy(gen_ctrl=1),
    "preselection": lambda: PreSelectionStrategy(n_candidates=8, n_select=4),
}


@pytest.mark.parametrize("strategy_name", sorted(STRATEGIES))
def test_an_added_column_survives_later_generations(strategy_name: str) -> None:
    optimizer = _optimizer(_ga(), STRATEGIES[strategy_name]())
    added = False
    observed = 0
    for state in optimizer.iterate():
        if not added:
            _add_everywhere(state)
            added = True
            continue
        _assert_column_kept(state)
        observed += 1

    assert observed >= 2


def test_an_added_column_survives_a_pso_tell_that_replaces_the_population() -> None:
    optimizer = _optimizer(PSO(), DirectStrategy())
    added = False
    observed = 0
    for state in optimizer.iterate():
        if not added:
            _add_everywhere(state)
            added = True
            continue
        _assert_column_kept(state)
        observed += 1

    assert observed >= 2


def test_adding_the_same_column_twice_leaves_the_structure_version_unchanged() -> None:
    optimizer = _optimizer(_ga(), DirectStrategy())
    added = False
    yields = 0
    for state in optimizer.iterate():
        if not added:
            added = True
            state.population.add_columns([_spec()])
            after_first = state.population.structure_version
            assert state.population.add_columns([_spec()]) == []
            assert state.population.structure_version == after_first
        yields += 1

    assert yields >= 3


def test_a_column_added_between_generations_survives_a_portable_checkpoint(
    tmp_path: Path,
) -> None:
    optimizer = _optimizer(_ga(), DirectStrategy())
    for state in optimizer.iterate():
        if state.gen == 1:
            _add_everywhere(state)
            state.save(tmp_path / "state")
            break

    loaded = OptimizationState.load(tmp_path / "state", _problem())
    assert COLUMN in loaded.population.schema

    resumed = _optimizer(_ga(), DirectStrategy()).run_from(loaded)

    assert resumed.gen == 4
    _assert_column_kept(resumed)


def test_a_column_added_between_generations_survives_a_pickle_checkpoint(
    tmp_path: Path,
) -> None:
    optimizer = _optimizer(_ga(), DirectStrategy())
    for state in optimizer.iterate():
        if state.gen == 1:
            _add_everywhere(state)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                optimizer.save_pickle(state, tmp_path / "state")
            break

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reloaded_optimizer, loaded = Optimizer.load_pickle(tmp_path / "state")
    assert COLUMN in loaded.population.schema

    resumed = reloaded_optimizer.run_from(loaded)

    assert resumed.gen == 4
    _assert_column_kept(resumed)


def test_dropping_a_column_the_algorithm_declares_is_repaired_at_the_first_commit() -> (
    None
):
    optimizer = _optimizer(PSO(), DirectStrategy())
    final = None

    for state in optimizer.iterate():
        final = state
        if state.gen == 0:
            state.population.drop_columns(["velocity"])

    assert final is not None
    assert "velocity" in final.population.schema


def test_every_column_of_a_builtin_schema_is_reserved_or_keeps_the_run_alive() -> None:
    probe = _optimizer(PSO(), DirectStrategy(), 3)
    for state in probe.iterate():
        state.population.add_columns([_spec()])
        schema = sorted(state.population.schema)
        break

    outcomes: dict[str, str] = {}
    for name in schema:
        optimizer = _optimizer(PSO(), DirectStrategy(), 3)
        try:
            for state in optimizer.iterate():
                if state.gen == 0:
                    state.population.add_columns([_spec()])
                    state.population.drop_columns([name])
            outcomes[name] = "continued"
        except ValidationError:
            outcomes[name] = "rejected"
        except KeyError as exc:
            outcomes[name] = f"stopped: KeyError {exc}"

    for name, outcome in outcomes.items():
        assert outcome in {"continued", "rejected"} or outcome.startswith(
            "stopped: KeyError"
        ), (name, outcome)
    assert {"continued", "rejected"} <= set(outcomes.values())


def test_adding_the_columns_pso_declares_lets_pso_continue() -> None:
    optimizer = _optimizer(_ga(), DirectStrategy(), n_gen=5)
    swapped = False
    state: OptimizationState | None = None
    for current in optimizer.iterate():
        if not swapped and current.gen == 1:
            specs = PSO().get_required_attrs(current.problem)
            for name in COLLECTIONS:
                getattr(current, name).add_columns(specs)
            optimizer.set_algorithm(PSO())
            swapped = True
        state = current

    assert swapped
    assert state is not None
    assert "velocity" in state.population.schema
    assert state.gen == 5
