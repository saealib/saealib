from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest

from saealib import (
    PSO,
    DirectStrategy,
    InequalityConstraint,
    IslandModel,
    LHSInitializer,
    Optimizer,
    Population,
    PopulationAttribute,
    Problem,
    PymooAlgorithm,
    SingleObjectiveComparator,
    Termination,
    max_fe,
)
from saealib.exceptions import ValidationError


def _optimizer(seed: int, max_evaluations: int = 4) -> Optimizer:
    def evaluate(x):
        return np.array([np.sum(x**2)], dtype=np.float64)

    return (
        Optimizer(
            Problem(
                func=evaluate,
                dim=2,
                n_obj=1,
                direction=np.array([-1.0]),
                lb=[-1.0, -1.0],
                ub=[1.0, 1.0],
            ),
            seed=seed,
        )
        .set_initializer(LHSInitializer(2, 2, seed))
        .set_algorithm(PSO())
        .set_strategy(DirectStrategy(n_offspring=2))
        .set_termination(Termination(max_fe(max_evaluations)))
    )


def _run(**kwargs):
    model = IslandModel((_optimizer(7), _optimizer(11)), **kwargs)
    return model, model.run()


_MIGRATION_ATTRS = [
    PopulationAttribute("id", np.int64, (), -1),
    PopulationAttribute("x", np.float64, (1,)),
    PopulationAttribute("f", np.float64, (1,)),
    PopulationAttribute("g", np.float64, (0,)),
    PopulationAttribute("cv", np.float64, (), 0.0),
]


def _migration_state(values, ids):
    values = np.asarray(values, dtype=np.float64)
    population = Population(_MIGRATION_ATTRS, len(values))
    population._extend_internal(
        {
            "id": np.asarray(ids, dtype=np.int64),
            "x": values[:, None],
            "f": values[:, None],
            "g": np.empty((len(values), 0), dtype=np.float64),
            "cv": np.zeros(len(values), dtype=np.float64),
        },
        preserve_ids=True,
    )
    return SimpleNamespace(
        population=population,
        comparator=SingleObjectiveComparator(direction=-1.0),
        gen=1,
    )


def _migration_state_with_objective_source(values, ids, sources):
    values = np.asarray(values, dtype=np.float64)
    attrs = [
        *_MIGRATION_ATTRS,
        PopulationAttribute("objective_source", np.uint8, (), default=255),
    ]
    population = Population(attrs, len(values))
    population._extend_internal(
        {
            "id": np.asarray(ids, dtype=np.int64),
            "x": values[:, None],
            "f": values[:, None],
            "g": np.empty((len(values), 0), dtype=np.float64),
            "cv": np.zeros(len(values), dtype=np.float64),
            "objective_source": np.asarray(sources, dtype=np.uint8),
        },
        preserve_ids=True,
    )
    return SimpleNamespace(
        population=population,
        comparator=SingleObjectiveComparator(direction=-1.0),
        gen=1,
    )


def _migration_model(n_islands, **kwargs):
    optimizers = tuple(SimpleNamespace(strategy=object()) for _ in range(n_islands))
    return IslandModel(cast(tuple[Optimizer, ...], optimizers), **kwargs)


def _optimizer_without_initializer(seed: int) -> Optimizer:
    def evaluate(x):
        return np.array([np.sum(x**2)], dtype=np.float64)

    return (
        Optimizer(
            Problem(
                func=evaluate,
                dim=2,
                n_obj=1,
                direction=np.array([-1.0]),
                lb=[-1.0, -1.0],
                ub=[1.0, 1.0],
            ),
            seed=seed,
        )
        .set_algorithm(PSO())
        .set_strategy(DirectStrategy(n_offspring=2))
        .set_termination(Termination(max_fe(8)))
    )


def test_islands_resolve_default_initializer_without_manual_initialization():
    migration_model = IslandModel(
        (_optimizer_without_initializer(7), _optimizer_without_initializer(11)),
        migration_interval=1,
    )
    migration_states = migration_model.run()

    fallback_states = IslandModel(
        (_optimizer_without_initializer(13),), migration_interval=0
    ).run()

    assert len(migration_states) == 2
    assert len(fallback_states) == 1
    assert all(state.fe >= 8 for state in (*migration_states, *fallback_states))


def test_migration_copies_values_and_preserves_target_ids():
    _, independent = _run(migration_interval=0)
    model, migrated = _run(migration_interval=1)
    target = migrated[1].population
    source = migrated[0].population
    old_target = independent[1].population

    assert not np.array_equal(target.get_array("f"), old_target.get_array("f"))
    matches = [
        (source_index, target_index)
        for source_index in range(len(source))
        for target_index in range(len(target))
        if np.array_equal(
            source.get_array("x")[source_index], target.get_array("x")[target_index]
        )
        and np.array_equal(
            source.get_array("f")[source_index], target.get_array("f")[target_index]
        )
    ]
    assert matches
    source_index, target_index = matches[0]
    assert np.array_equal(
        target.get_array("f")[target_index], source.get_array("f")[source_index]
    )
    assert not np.array_equal(
        target.get_array("f")[target_index], old_target.get_array("f")[target_index]
    )
    for column in ("x", "g", "cv"):
        assert np.array_equal(
            target.get_array(column)[target_index],
            source.get_array(column)[source_index],
        )
    assert np.array_equal(target.get_array("id"), old_target.get_array("id"))
    assert model.migration_events


def test_migration_from_population_without_objective_source_marks_target_unknown():
    source = _migration_state([0.0, 100.0], [0, 1])
    target = _migration_state_with_objective_source([10.0, 110.0], [10, 11], [0, 1])
    model = _migration_model(2, migration_interval=1, migration_size=1)

    model._migrate(source, target)

    assert target.population.get_array("id").tolist() == [10, 11]
    assert target.population.get_array("f")[1, 0] == 0.0
    assert target.population.get_array("objective_source")[1] == 255


@pytest.mark.parametrize(
    ("topology", "expected"),
    [
        (
            "ring",
            ([0.0, 20.0], [10.0, 0.0], [20.0, 10.0]),
        ),
        (
            "fully_connected",
            ([0.0, 20.0], [10.0, 20.0], [20.0, 10.0]),
        ),
    ],
)
def test_migration_collects_all_edges_before_writing(topology, expected):
    states = [
        _migration_state([0.0, 100.0], [0, 1]),
        _migration_state([10.0, 110.0], [10, 11]),
        _migration_state([20.0, 120.0], [20, 21]),
    ]
    model = _migration_model(3, topology=topology, migration_interval=1)

    model._migrate_ready(states, [True, True, True], {})

    for state, values in zip(states, expected):
        np.testing.assert_array_equal(state.population.get_array("f")[:, 0], values)


@pytest.mark.parametrize(
    ("migration_size", "source_values", "target_values", "expected"),
    [
        (2, [1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0], [10, 20, 1, 2]),
        (4, [1.0, 2.0], [10.0, 20.0], [1, 2]),
    ],
)
def test_migration_size_copies_multiple_rows_and_clamps(
    migration_size, source_values, target_values, expected
):
    states = [
        _migration_state(source_values, range(len(source_values))),
        _migration_state(target_values, range(10, 10 + len(target_values))),
    ]
    model = _migration_model(
        2,
        migration_interval=1,
        migration_size=migration_size,
        topology=[[0, 1], [0, 0]],
    )

    model._migrate_ready(states, [True, True], {})

    np.testing.assert_array_equal(states[1].population.get_array("f")[:, 0], expected)


def test_explicit_topology_controls_migration_direction():
    _, baseline = _run(migration_interval=0)
    model, states = _run(topology=[[0, 1], [0, 0]], migration_interval=1)

    assert model.migration_events == [(0, 1)]
    for column in ("x", "f", "g", "cv", "id"):
        assert np.array_equal(
            states[0].population.get_array(column),
            baseline[0].population.get_array(column),
        )
    best = states[0].comparator.sort_population(states[0].population)[0]
    assert any(
        all(
            np.array_equal(
                states[0].population.get_array(column)[best],
                states[1].population.get_array(column)[index],
            )
            for column in ("x", "f", "g", "cv")
        )
        for index in range(len(states[1].population))
    )


def test_fully_connected_migration_sends_multiple_rows_per_edge():
    optimizers = tuple(_optimizer(seed, 6) for seed in (7, 11, 13))
    model = IslandModel(
        optimizers,
        topology="fully_connected",
        migration_interval=1,
        migration_size=2,
    )
    model.run()

    assert set(model.migration_events) == {
        (source, target)
        for source in range(3)
        for target in range(3)
        if source != target
    }
    assert len(model.migration_events) >= 6


def test_migration_interval_is_in_generations():
    model = IslandModel((_optimizer(7, 6), _optimizer(11, 6)), migration_interval=2)
    model.run()

    assert model.migration_events == [(0, 1), (1, 0)]


def test_fixed_seeds_reproduce_islands():
    _, first = _run()
    _, second = _run()

    for left, right in zip(first, second):
        assert np.array_equal(
            left.population.get_array("x"), right.population.get_array("x")
        )
        assert np.array_equal(
            left.population.get_array("f"), right.population.get_array("f")
        )


@pytest.mark.parametrize(
    "topology",
    [
        "unknown",
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        [["yes", 0], [0, 1]],
        [[0.0, np.nan], [0.0, 1.0]],
        [[0.0, np.inf], [0.0, 1.0]],
    ],
)
def test_topology_validation(topology):
    with pytest.raises(ValidationError):
        IslandModel((_optimizer(1), _optimizer(2)), topology=topology)


@pytest.mark.parametrize("interval", [1.5, "1"])
def test_migration_interval_must_be_an_integer(interval):
    with pytest.raises(ValidationError):
        IslandModel((_optimizer(1), _optimizer(2)), migration_interval=interval)


@pytest.mark.parametrize("migration_size", [0, -1, 1.5])
def test_migration_size_must_be_positive_integer(migration_size):
    with pytest.raises(ValidationError):
        IslandModel((_optimizer(1), _optimizer(2)), migration_size=migration_size)


def test_topology_and_adjacency_cannot_both_be_specified():
    with pytest.raises(ValidationError):
        IslandModel(
            (_optimizer(1), _optimizer(2)),
            topology="fully_connected",
            adjacency=[[0, 1], [0, 0]],
        )


def _problem(*, dim=2, n_obj=1, direction=None, constraints=None):
    return Problem(
        func=lambda x: np.asarray([np.sum(x**2)] * n_obj, dtype=np.float64),
        dim=dim,
        n_obj=n_obj,
        direction=np.full(n_obj, -1.0) if direction is None else direction,
        lb=[-1.0] * dim,
        ub=[1.0] * dim,
        constraints=[] if constraints is None else constraints,
    )


def test_migration_copies_all_shared_algorithm_columns_and_pso_state():
    source = _migration_state([0.0, 10.0], [0, 1])
    target = _migration_state([20.0, 30.0], [10, 11])
    for state, offset in ((source, 0.0), (target, 200.0)):
        attrs = [
            *state.population.attrs,
            PopulationAttribute("pbest_x", np.float64, (1,)),
            PopulationAttribute("pbest_f", np.float64, (1,)),
            PopulationAttribute("pbest_cv", np.float64, (), 0.0),
            PopulationAttribute("velocity", np.float64, (1,)),
        ]
        upgraded = Population(attrs, len(state.population))
        upgraded._extend_internal(
            {
                name: state.population.get_array(name).copy()
                for name in state.population.schema
            },
            preserve_ids=True,
        )
        upgraded.update_array("pbest_x", state.population.get_array("x") + offset)
        upgraded.update_array("pbest_f", state.population.get_array("f") + offset)
        upgraded.update_array(
            "pbest_cv", np.arange(len(state.population), dtype=np.float64) + offset
        )
        upgraded.update_array("velocity", np.full((len(state.population), 1), offset))
        state.population = upgraded

    model = _migration_model(2, topology=[[0, 1], [0, 0]], migration_interval=1)
    model._migrate_ready([source, target], [True, True], {})
    source_index = source.comparator.sort_population(source.population)[0]
    target_index = 1
    for name in target.population.schema:
        if name != "id":
            np.testing.assert_array_equal(
                target.population.get_array(name)[target_index],
                source.population.get_array(name)[source_index],
            )
    np.testing.assert_array_equal(target.population.get_array("id"), [10, 11])
    np.testing.assert_array_equal(
        target.population.get_array("pbest_x")[target_index],
        target.population.get_array("x")[target_index],
    )
    np.testing.assert_array_equal(
        target.population.get_array("pbest_f")[target_index],
        target.population.get_array("f")[target_index],
    )


@pytest.mark.parametrize(
    ("attribute", "left", "right"),
    [("dim", 2, 3), ("n_obj", 1, 2), ("n_constraints", 0, 1)],
)
def test_island_model_rejects_incompatible_problem_shapes(attribute, left, right):
    if attribute == "n_constraints":
        first_problem = _problem(constraints=[])
        second_problem = _problem(
            constraints=[InequalityConstraint(lambda x: np.sum(x), threshold=0.0)]
        )
    else:
        first_problem = _problem(**{attribute: left})
        second_problem = _problem(**{attribute: right})
    first = SimpleNamespace(strategy=object(), problem=first_problem)
    second = SimpleNamespace(strategy=object(), problem=second_problem)
    with pytest.raises(ValidationError, match=rf"island 1 problem {attribute}"):
        IslandModel(cast(tuple[Optimizer, ...], (first, second)), migration_interval=0)


def test_island_model_rejects_opposite_directions():
    first = SimpleNamespace(
        strategy=object(), problem=_problem(direction=np.array([-1.0]))
    )
    second = SimpleNamespace(
        strategy=object(), problem=_problem(direction=np.array([1.0]))
    )
    with pytest.raises(ValidationError, match="island 1 problem direction"):
        IslandModel(cast(tuple[Optimizer, ...], (first, second)), migration_interval=0)


def test_migration_rejects_engine_mode_pymoo_algorithm():
    from pymoo.algorithms.soo.nonconvex.ga import GA

    optimizer = SimpleNamespace(
        strategy=object(), problem=_problem(), algorithm=PymooAlgorithm(GA())
    )
    with pytest.raises(ValidationError, match=r"island 0.*PymooAlgorithm"):
        IslandModel(
            cast(tuple[Optimizer, ...], (optimizer, _optimizer(2))),
            migration_interval=1,
        )


def test_migration_disabled_allows_engine_mode_pymoo_algorithm():
    from pymoo.algorithms.soo.nonconvex.ga import GA

    optimizer = SimpleNamespace(
        strategy=object(), problem=_problem(), algorithm=PymooAlgorithm(GA())
    )
    IslandModel(
        cast(tuple[Optimizer, ...], (optimizer, _optimizer(2))), migration_interval=0
    )
