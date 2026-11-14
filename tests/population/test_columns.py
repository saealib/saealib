"""Column add and drop operations on Population, Archive, and ParetoArchive."""

from __future__ import annotations

import pickle

import numpy as np
import pytest

from saealib.comparators import NSGA2Comparator
from saealib.exceptions import ValidationError
from saealib.identity import PopulationAttribute
from saealib.population import Archive, ParetoArchive, Population

DIM = 2


def _attrs() -> list[PopulationAttribute]:
    return [
        PopulationAttribute("x", np.float64, (DIM,), default=np.nan),
        PopulationAttribute("f", np.float64, (1,), default=np.nan),
        PopulationAttribute("g", np.float64, (0,), default=0.0),
        PopulationAttribute("cv", np.float64, (), default=0.0),
        PopulationAttribute("id", np.int64, (), default=-1),
    ]


def _data(rows: int = 3) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(0)
    return {
        "x": rng.uniform(-1.0, 1.0, size=(rows, DIM)),
        "f": rng.uniform(0.0, 1.0, size=(rows, 1)),
        "g": np.zeros((rows, 0)),
        "cv": np.zeros(rows),
        "id": np.arange(rows, dtype=np.int64),
    }


def _population(rows: int = 3) -> Population:
    population = Population(_attrs(), rows)
    population._extend_internal(_data(rows), preserve_ids=True)
    return population


def test_add_columns_creates_the_column_filled_with_the_default() -> None:
    population = _population()

    added = population.add_columns(
        [PopulationAttribute("velocity", np.float64, (DIM,), default=0.5)]
    )

    assert added == ["velocity"]
    assert "velocity" in population.schema
    np.testing.assert_array_equal(
        population.get_array("velocity"), np.full((3, DIM), 0.5)
    )
    assert len(population) == 3


def test_add_columns_fills_the_storage_capacity_for_later_rows() -> None:
    population = _population()

    population.add_columns([PopulationAttribute("tag", np.int64, (), default=7)])
    population.append({"x": np.zeros(DIM), "f": np.zeros(1)})

    np.testing.assert_array_equal(population.get_array("tag"), np.full(4, 7))


def test_add_columns_bumps_the_structure_version_once_for_a_batch() -> None:
    population = _population()
    structure = population.structure_version
    value = population.value_version

    population.add_columns(
        [
            PopulationAttribute("a", np.float64, (), default=0.0),
            PopulationAttribute("b", np.int64, (), default=-1),
        ]
    )

    assert population.structure_version == structure + 1
    assert population.value_version == value + 1


def test_add_columns_of_an_equal_definition_is_a_no_op() -> None:
    population = _population()
    spec = PopulationAttribute("a", np.float64, (), default=np.nan)
    population.add_columns([spec])
    structure = population.structure_version
    value = population.value_version

    added = population.add_columns([spec])

    assert added == []
    assert population.structure_version == structure
    assert population.value_version == value


def test_add_columns_of_a_conflicting_definition_raises() -> None:
    population = _population()
    population.add_columns([PopulationAttribute("a", np.float64, (), default=0.0)])

    with pytest.raises(ValidationError, match="Conflicting PopulationAttribute"):
        population.add_columns([PopulationAttribute("a", np.float64, (DIM,))])


@pytest.mark.parametrize(
    "specs",
    [
        [
            PopulationAttribute("a", np.float64, ()),
            PopulationAttribute("a", np.int64, ()),
        ],
        [PopulationAttribute("", np.float64, ())],
        [PopulationAttribute("a", object, ())],
        [PopulationAttribute("a", np.float64, (-1,))],
        [PopulationAttribute("a", np.float64, (2.5,))],  # ty: ignore[invalid-argument-type]
        [PopulationAttribute("a", np.float64, (), default="not a number")],
        [None],
    ],
)
def test_add_columns_rejects_an_invalid_definition(specs: list[object]) -> None:
    population = _population()
    structure = population.structure_version

    with pytest.raises(ValidationError):
        population.add_columns(specs)  # ty: ignore[invalid-argument-type]

    assert population.structure_version == structure
    assert "a" not in population.schema


def test_add_columns_with_no_specs_is_a_no_op() -> None:
    population = _population()
    structure = population.structure_version

    assert population.add_columns([]) == []
    assert population.structure_version == structure


def test_add_columns_invalidates_individual_handles() -> None:
    population = _population()
    individual = population[0]

    population.add_columns([PopulationAttribute("a", np.float64, ())])

    with pytest.raises(RuntimeError, match="Invalid Individual reference"):
        individual.get_readonly_value("f")


def test_drop_columns_removes_the_column_and_bumps_the_structure_version() -> None:
    population = _population()
    population.add_columns([PopulationAttribute("a", np.float64, ())])
    structure = population.structure_version

    dropped = population.drop_columns(["a"])

    assert dropped == ["a"]
    assert "a" not in population.schema
    assert "a" not in population.columns
    assert population.structure_version == structure + 1


def test_drop_columns_keeps_the_remaining_columns_intact() -> None:
    population = _population()
    expected = population.get_array("f").copy()
    population.add_columns([PopulationAttribute("a", np.float64, ())])

    population.drop_columns(["a"])

    np.testing.assert_array_equal(population.get_array("f"), expected)
    assert len(population) == 3


@pytest.mark.parametrize(
    ("names", "match"),
    [
        (["missing"], "unknown column"),
        (["a", "a"], "must not repeat"),
        ([""], "non-empty string column name"),
        (["x"], "reserved column"),
        (["f"], "reserved column"),
        (["cv"], "reserved column"),
        (["id"], "reserved column"),
        (["objective_source"], "reserved column"),
    ],
)
def test_drop_columns_rejects_the_name(names: list[str], match: str) -> None:
    attrs = [
        *_attrs(),
        PopulationAttribute("objective_source", np.uint8, (), default=0),
    ]
    population = Population(attrs, 3)
    population._extend_internal(
        {**_data(3), "objective_source": np.zeros(3, dtype=np.uint8)},
        preserve_ids=True,
    )
    population.add_columns([PopulationAttribute("a", np.float64, ())])
    structure = population.structure_version

    with pytest.raises(ValidationError, match=match):
        population.drop_columns(names)

    assert population.structure_version == structure


def test_drop_columns_with_no_names_is_a_no_op() -> None:
    population = _population()

    assert population.drop_columns([]) == []


def test_add_and_drop_invalidate_a_population_keyed_cache() -> None:
    population = _population()
    population.set_cache(("entry",), "value")

    population.add_columns([PopulationAttribute("a", np.float64, ())])

    assert population.get_cache(("entry",)) is None

    population.set_cache(("entry",), "value")
    population.add_columns([PopulationAttribute("a", np.float64, ())])

    assert population.get_cache(("entry",)) == "value"

    population.drop_columns(["a"])

    assert population.get_cache(("entry",)) is None


def test_add_and_drop_invalidate_the_comparator_sort_cache() -> None:
    population = _population()
    comparator = NSGA2Comparator()
    comparator.sort_population(population)
    cached = ("pareto_sort", comparator._criterion_key())
    assert population.get_cache(cached) is not None

    population.add_columns([PopulationAttribute("a", np.float64, ())])

    assert population.get_cache(cached) is None

    comparator.sort_population(population)
    population.add_columns([PopulationAttribute("a", np.float64, ())])

    assert population.get_cache(cached) is not None

    population.drop_columns(["a"])

    assert population.get_cache(cached) is None


def test_archive_refuses_to_drop_the_identity_column() -> None:
    archive = Archive(_attrs(), 3)

    with pytest.raises(ValidationError, match="reserved column"):
        archive.drop_columns(["x"])


def test_pareto_archive_refuses_to_drop_the_identity_column() -> None:
    archive = ParetoArchive(_attrs(), 3, direction=np.array([-1.0]))

    with pytest.raises(ValidationError, match="reserved column"):
        archive.drop_columns(["x"])


def test_append_archive_refuses_to_drop_its_observation_columns() -> None:
    attrs = [
        *_attrs(),
        PopulationAttribute("request_id", np.int64, (), default=-1),
    ]
    archive = Archive(attrs, 3, duplicate_policy="append")
    archive.add_columns([PopulationAttribute("a", np.float64, ())])

    for name in ("id", "request_id"):
        with pytest.raises(ValidationError, match="reserved column"):
            archive.drop_columns([name])

    assert archive.drop_columns(["a"]) == ["a"]


def test_a_column_dropped_from_an_archive_is_gone_from_its_schema() -> None:
    archive = Archive(_attrs(), 3)
    archive.add_columns([PopulationAttribute("a", np.float64, ())])

    archive.drop_columns(["a"])

    assert set(archive.schema) == {attr.name for attr in _attrs()}


def test_added_columns_survive_a_pickle_round_trip() -> None:
    archive = Archive(_attrs(), 3)
    archive.add_columns([PopulationAttribute("a", np.float64, (), default=1.5)])

    restored = pickle.loads(pickle.dumps(archive))

    assert "a" in restored.schema
    assert restored.add_columns([PopulationAttribute("a", np.float64, (), 1.5)]) == []
