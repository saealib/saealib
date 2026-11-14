"""Transitions attached to a component replacement and their commit point.

A replacement records the transition its setter carried; the commit point runs
it after static validation and before publication, with collection writes
staged, so a transition that raises leaves the published configuration as it
was.
"""

from __future__ import annotations

import pickle
from typing import Any, cast

import numpy as np
import pytest

from saealib import (
    PSO,
    Chain,
    CrossoverBLXAlpha,
    CrossoverSBX,
    LHSInitializer,
    MutationUniform,
    NSGA3Comparator,
    Optimizer,
    PopulationAttribute,
    SequentialSelection,
    SingleObjectiveComparator,
    Termination,
    Transition,
    TransitionContext,
    TruncationSelection,
    chain,
)
from saealib.algorithms import GA
from saealib.exceptions import (
    CheckpointError,
    ConfigurationError,
    ReplacementRejectedError,
    ValidationError,
)
from saealib.execution.bindings import BINDING_SLOTS
from saealib.execution.runner import Runner
from saealib.execution.runtime import create_runtime
from saealib.problem import Problem
from saealib.strategies import DirectStrategy, GenerationBasedStrategy


def _sphere(x: Any) -> np.ndarray:
    return np.array([float(np.sum(np.asarray(x) ** 2))], dtype=np.float64)


def _problem(dim: int = 2) -> Problem:
    return Problem(
        func=_sphere,
        dim=dim,
        n_obj=1,
        direction=np.array([-1.0]),
        lb=[-5.0] * dim,
        ub=[5.0] * dim,
        constraints=[],
    )


def _six_generations(state: Any) -> bool:
    return state.gen >= 6


def _ga() -> GA:
    return GA(
        crossover=CrossoverBLXAlpha(prob=0.9, alpha=0.4),
        mutation=MutationUniform(prob_var=0.1),
        parent_selection=SequentialSelection(),
        survivor_selection=TruncationSelection(),
    )


def _optimizer(algorithm: Any | None = None, *, seed: int = 7) -> Optimizer:
    return (
        Optimizer(_problem(), seed=seed)
        .set_initializer(LHSInitializer(6, 4, seed))
        .set_algorithm(_ga() if algorithm is None else algorithm)
        .set_strategy(DirectStrategy())
        .set_termination(Termination(_six_generations))
    )


def _capturing_runner(optimizer: Any) -> tuple[Runner, dict[str, Any]]:
    captured: dict[str, Any] = {}

    def factory(optimizer_obj: Any) -> Any:
        runtime = create_runtime(optimizer_obj)
        captured["environment"] = getattr(runtime, "environment", None)
        return runtime

    return Runner(optimizer, runtime_factory=factory), captured


def _pso_columns() -> list[PopulationAttribute]:
    return [
        PopulationAttribute("velocity", np.float64, (2,), 0.0),
        PopulationAttribute("pbest_x", np.float64, (2,), np.nan),
        PopulationAttribute("pbest_f", np.float64, (1,), np.nan),
        PopulationAttribute("pbest_cv", np.float64, (), np.nan),
    ]


class _Recorder(Transition):
    """Records its application and optionally produces population columns."""

    def __init__(
        self,
        events: list[str],
        name: str = "recorder",
        *,
        columns: list[PopulationAttribute] | None = None,
        produced: frozenset[str] | None = None,
    ) -> None:
        self.events = events
        self.name = name
        self.columns = columns
        self.produced = produced

    def produced_columns(self, new_bindings: Any) -> frozenset[str] | None:
        return self.produced

    def apply(self, tctx: TransitionContext) -> Any:
        self.events.append(f"{self.name}:apply")
        if self.columns:
            for staged in (tctx.population, tctx.archive, tctx.pareto_archive):
                if staged is not None:
                    staged.add_columns(self.columns)
        yield from ()


def _pso_transition(events: list[str] | None = None) -> _Recorder:
    return _Recorder(
        [] if events is None else events,
        "pso",
        columns=_pso_columns(),
        produced=frozenset({"velocity", "pbest_x", "pbest_f", "pbest_cv"}),
    )


# ---------------------------------------------------------------------------
# Recorded replacements
# ---------------------------------------------------------------------------


def test_a_setter_records_the_transition_and_origin_of_a_replacement() -> None:
    optimizer = _optimizer()
    algorithm = _ga()
    transition = Transition()

    optimizer.set_algorithm(algorithm, transition=transition)

    records = optimizer._pending_replacements()
    assert set(records) == {"algorithm"}
    record = records["algorithm"]
    assert record.slot == "algorithm"
    assert record.component is algorithm
    assert record.transition is transition
    assert record.source == "setter"


def test_a_later_setter_without_a_transition_replaces_the_record() -> None:
    optimizer = _optimizer()
    optimizer.set_algorithm(_ga(), transition=Transition())

    optimizer.set_algorithm(new_algorithm := _ga())

    assert optimizer._pending_replacements() == {}
    assert optimizer.algorithm is new_algorithm


def test_a_transition_argument_must_be_a_transition() -> None:
    optimizer = _optimizer()

    with pytest.raises(ValidationError):
        optimizer.set_algorithm(_ga(), transition=cast(Any, object()))


def test_a_record_is_consumed_when_its_replacement_is_published() -> None:
    optimizer = _optimizer()
    runner, captured = _capturing_runner(optimizer)
    generator = runner.iterate()
    next(generator)
    environment = captured["environment"]
    version = environment.active_version
    optimizer.set_algorithm(_ga(), transition=Transition())
    assert set(optimizer._pending_replacements()) == {"algorithm"}

    next(generator)

    assert environment.active_version == version + 1
    assert optimizer._pending_replacements() == {}


def test_a_record_is_dropped_when_the_commit_point_rejects_the_replacement() -> None:
    optimizer = _optimizer()
    runner, captured = _capturing_runner(optimizer)
    generator = runner.iterate()
    states = [next(generator)]
    environment = captured["environment"]
    version = environment.active_version
    optimizer.set_algorithm(PSO(), transition=Transition())

    with pytest.raises(ConfigurationError):
        for state in generator:
            states.append(state)
    generator.close()

    assert environment.active_version == version
    assert optimizer._pending_replacements() == {}


def test_a_record_for_the_published_component_is_dropped_without_publishing() -> None:
    optimizer = _optimizer()
    runner, captured = _capturing_runner(optimizer)
    generator = runner.iterate()
    next(generator)
    environment = captured["environment"]
    version = environment.active_version
    published = optimizer.algorithm
    optimizer.set_algorithm(published, transition=Transition())
    assert set(optimizer._pending_replacements()) == {"algorithm"}

    next(generator)

    assert environment.active_version == version
    assert optimizer._pending_replacements() == {}


def test_a_publication_of_another_slot_does_not_run_a_stale_transition() -> None:
    events: list[str] = []
    optimizer = _optimizer()
    runner, captured = _capturing_runner(optimizer)
    generator = runner.iterate()
    next(generator)
    environment = captured["environment"]
    version = environment.active_version
    optimizer.set_algorithm(optimizer.algorithm, transition=_Recorder(events))
    optimizer.set_strategy(DirectStrategy())

    next(generator)

    assert environment.active_version == version + 1
    assert events == []
    assert optimizer._pending_replacements() == {}


def test_a_contract_difference_publication_does_not_run_a_stale_transition() -> None:
    events: list[str] = []
    optimizer = _optimizer()
    runner, captured = _capturing_runner(optimizer)
    generator = runner.iterate()
    state = next(generator)
    environment = captured["environment"]
    version = environment.active_version
    optimizer.set_algorithm(optimizer.algorithm, transition=_Recorder(events))
    assert state.gen == 0

    cast(GA, optimizer.algorithm).crossover = CrossoverSBX(prob=0.9, eta=20.0)
    next(generator)

    assert environment.active_version == version + 1
    assert events == []
    assert optimizer._pending_replacements() == {}


def test_a_pickle_does_not_carry_a_pending_record() -> None:
    optimizer = _optimizer()
    runner, _ = _capturing_runner(optimizer)
    generator = runner.iterate()
    next(generator)
    optimizer.set_algorithm(_ga(), transition=Transition())

    restored = pickle.loads(pickle.dumps(optimizer))

    assert optimizer._pending_replacements() != {}
    assert restored._pending_replacements() == {}


def test_a_checkpoint_cannot_be_written_while_a_record_is_pending(
    tmp_path: Any,
) -> None:
    optimizer = _optimizer()
    runner, _ = _capturing_runner(optimizer)
    generator = runner.iterate()
    state = next(generator)
    target = tmp_path / "checkpoint.npz"
    state.save(target)
    optimizer.set_algorithm(_ga(), transition=Transition())

    with pytest.raises(CheckpointError):
        state.save(target)


# ---------------------------------------------------------------------------
# Running a transition at the commit point
# ---------------------------------------------------------------------------


class _BoomError(RuntimeError):
    pass


def test_a_transition_that_raises_stops_the_run_with_its_own_exception() -> None:
    optimizer = _optimizer()
    runner, captured = _capturing_runner(optimizer)
    generator = runner.iterate()
    states = [next(generator)]
    environment = captured["environment"]
    version = environment.active_version
    resolution = optimizer._component_resolution
    counter = optimizer._configuration_version
    columns = sorted(states[-1].population.schema)

    def explode(tctx: TransitionContext) -> Any:
        raise _BoomError("refused")
        yield

    optimizer.set_algorithm(PSO(), transition=Transition.from_function(explode))

    with pytest.raises(_BoomError):
        for state in generator:
            states.append(state)
    generator.close()

    assert environment.active_version == version
    assert optimizer._component_resolution is resolution
    assert optimizer._configuration_version == counter
    assert sorted(states[-1].population.schema) == columns
    assert states[-1].__dict__["_binding_version"] == version
    assert type(optimizer.algorithm) is GA


def test_resuming_after_a_raising_transition_keeps_the_old_configuration() -> None:
    optimizer = _optimizer()
    runner, _ = _capturing_runner(optimizer)
    generator = runner.iterate()
    states = [next(generator)]

    def explode(tctx: TransitionContext) -> Any:
        raise _BoomError("refused")
        yield

    optimizer.set_algorithm(PSO(), transition=Transition.from_function(explode))
    with pytest.raises(_BoomError):
        for state in generator:
            states.append(state)
    generator.close()

    calls: list[int] = []

    def record(tctx: TransitionContext) -> Any:
        calls.append(1)
        yield from ()

    resumed, resumed_captured = _capturing_runner(optimizer)
    final: Any = None
    for final in resumed.iterate_from(states[-1]):
        pass

    assert final.gen == 6
    assert resumed_captured["environment"].active_version == 0
    assert calls == []


def test_setting_the_same_replacement_again_runs_the_transition_again() -> None:
    optimizer = _optimizer()
    runner, captured = _capturing_runner(optimizer)
    generator = runner.iterate()
    states = [next(generator)]
    environment = captured["environment"]
    calls: list[int] = []

    def explode(tctx: TransitionContext) -> Any:
        raise _BoomError("refused")
        yield

    optimizer.set_algorithm(PSO(), transition=Transition.from_function(explode))
    with pytest.raises(_BoomError):
        for state in generator:
            states.append(state)
    generator.close()

    def record(tctx: TransitionContext) -> Any:
        calls.append(1)
        yield from ()

    replacement = PSO()
    optimizer.set_algorithm(replacement, transition=Transition.from_function(record))
    with pytest.raises(ConfigurationError):
        for _ in _capturing_runner(optimizer)[0].iterate_from(states[-1]):
            pass

    assert calls == [1]
    assert environment.active_version == 0


def test_a_yielded_value_reports_that_transition_evaluation_is_unavailable() -> None:
    optimizer = _optimizer()
    runner, captured = _capturing_runner(optimizer)
    generator = runner.iterate()
    next(generator)
    environment = captured["environment"]
    version = environment.active_version

    def request_evaluation(tctx: TransitionContext) -> Any:
        yield "evaluate"

    optimizer.set_algorithm(
        PSO(), transition=Transition.from_function(request_evaluation)
    )

    with pytest.raises(ConfigurationError) as error:
        next(generator)
    generator.close()

    assert "evaluation requested by a transition is not available" in str(error.value)
    assert environment.active_version == version
    assert type(optimizer.algorithm) is GA


def test_a_transition_runs_after_validation_and_before_publication(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    optimizer = _optimizer()
    runner, captured = _capturing_runner(optimizer)
    generator = runner.iterate()
    next(generator)
    environment = captured["environment"]
    original_check = environment._check_staged_changes
    original_publish = environment._publish_prepared

    def check(*args: Any, **kwargs: Any) -> Any:
        events.append("post-validation")
        return original_check(*args, **kwargs)

    def publish(*args: Any, **kwargs: Any) -> Any:
        events.append(f"publish:{environment.active_version + 1}")
        return original_publish(*args, **kwargs)

    monkeypatch.setattr(environment, "_check_staged_changes", check)
    monkeypatch.setattr(environment, "_publish_prepared", publish)
    optimizer.set_algorithm(_ga(), transition=_Recorder(events, "algorithm"))

    next(generator)

    assert events == ["algorithm:apply", "post-validation", "publish:1"]
    assert environment.active_version == 1


def test_columns_a_declared_transition_produces_complete_the_replacement() -> None:
    optimizer = _optimizer()
    runner, captured = _capturing_runner(optimizer)
    generator = runner.iterate()
    state = next(generator)
    environment = captured["environment"]
    identities = (
        id(state.population),
        id(state.archive),
        id(state.pareto_archive),
    )
    optimizer.set_algorithm(PSO(), transition=_pso_transition())

    replacement = next(generator)

    assert environment.active_version == 1
    schema = replacement.population.schema
    assert {"velocity", "pbest_x", "pbest_f", "pbest_cv"} <= set(schema)
    assert float(replacement.population.get_array("velocity")[0, 0]) == 0.0
    assert (
        id(replacement.population),
        id(replacement.archive),
        id(replacement.pareto_archive),
    ) == identities


def test_a_replacement_whose_produced_columns_are_missing_is_rejected() -> None:
    optimizer = _optimizer()
    runner, captured = _capturing_runner(optimizer)
    generator = runner.iterate()
    states = [next(generator)]
    environment = captured["environment"]
    columns = sorted(states[-1].population.schema)
    declaring = _Recorder(
        [],
        "declares",
        produced=frozenset({"velocity", "pbest_x", "pbest_f", "pbest_cv"}),
    )
    optimizer.set_algorithm(PSO(), transition=declaring)

    with pytest.raises(ReplacementRejectedError) as error:
        for state in generator:
            states.append(state)
    generator.close()

    assert [item.code for item in error.value.diagnostics] == [
        "missing_population_column"
    ]
    assert "velocity" in str(error.value)
    assert environment.active_version == 0
    assert type(optimizer.algorithm) is GA
    assert sorted(states[-1].population.schema) == columns


def test_a_produced_column_with_the_wrong_specification_is_rejected() -> None:
    optimizer = _optimizer()
    runner, _ = _capturing_runner(optimizer)
    generator = runner.iterate()
    next(generator)
    columns = _pso_columns()
    columns[0] = PopulationAttribute("velocity", np.int64, (), 0)
    wrong = _Recorder(
        [],
        "wrong",
        columns=columns,
        produced=frozenset({"velocity", "pbest_x", "pbest_f", "pbest_cv"}),
    )
    optimizer.set_algorithm(PSO(), transition=wrong)

    with pytest.raises(ReplacementRejectedError) as error:
        next(generator)
    generator.close()

    assert "population_column_specification_mismatch" in [
        item.code for item in error.value.diagnostics
    ]
    assert type(optimizer.algorithm) is GA


def test_a_transition_without_a_declaration_warns_and_is_still_verified(
    caplog: pytest.LogCaptureFixture,
) -> None:
    optimizer = _optimizer()
    runner, captured = _capturing_runner(optimizer)
    generator = runner.iterate()
    next(generator)
    environment = captured["environment"]
    transition = _Recorder([], "undeclared", columns=_pso_columns())
    optimizer.set_algorithm(PSO(), transition=transition)

    with caplog.at_level("WARNING", logger="saealib.execution.runtime"):
        replacement = next(generator)

    assert "declares no produced columns" in caplog.text
    assert environment.active_version == 1
    assert "velocity" in replacement.population.schema


# ---------------------------------------------------------------------------
# Composing transitions
# ---------------------------------------------------------------------------


def test_a_chain_runs_its_parts_in_order() -> None:
    events: list[str] = []
    composed = chain(_Recorder(events, "first"), _Recorder(events, "second"))
    context = TransitionContext(
        state=None,
        problem=None,
        rng=None,
        outgoing={},
        incoming={},
        replaced_slots=(),
        write_targets=frozenset(),
    )

    assert isinstance(composed, Chain)
    list(composed.apply(context))
    assert events == ["first:apply", "second:apply"]


def test_a_chain_declares_the_union_of_its_parts_columns() -> None:
    class Declares(Transition):
        def __init__(self, name: str) -> None:
            self.name = name

        def produced_columns(self, new_bindings: Any) -> frozenset[str] | None:
            return frozenset({self.name})

        def apply(self, tctx: TransitionContext) -> Any:
            yield from ()

    composed = chain(Declares("velocity"), Declares("pbest_f"))

    assert composed.produced_columns({}) == frozenset({"velocity", "pbest_f"})
    assert composed.parts == (composed.parts[0], composed.parts[1])


def test_a_chain_ignores_parts_that_declare_nothing() -> None:
    class Declares(Transition):
        def produced_columns(self, new_bindings: Any) -> frozenset[str] | None:
            return frozenset({"velocity"})

        def apply(self, tctx: TransitionContext) -> Any:
            yield from ()

    assert chain(Declares(), Transition()).produced_columns({}) == frozenset(
        {"velocity"}
    )


def test_replacing_two_slots_runs_their_transitions_in_binding_order() -> None:
    events: list[str] = []
    optimizer = _optimizer()
    runner, captured = _capturing_runner(optimizer)
    generator = runner.iterate()
    next(generator)
    environment = captured["environment"]
    optimizer.set_strategy(DirectStrategy(), transition=_Recorder(events, "strategy"))
    optimizer.set_algorithm(_ga(), transition=_Recorder(events, "algorithm"))

    next(generator)

    assert events == ["algorithm:apply", "strategy:apply"]
    assert environment.active_version == 1


def test_a_chain_of_transitions_runs_at_one_commit_point() -> None:
    events: list[str] = []
    optimizer = _optimizer()
    runner, captured = _capturing_runner(optimizer)
    generator = runner.iterate()
    next(generator)
    environment = captured["environment"]
    optimizer.set_algorithm(
        _ga(),
        transition=chain(_Recorder(events, "first"), _Recorder(events, "second")),
    )

    next(generator)

    assert events == ["first:apply", "second:apply"]
    assert environment.active_version == 1


def test_a_transition_runs_at_a_commit_point_outside_a_run() -> None:
    from saealib.core.runtime import StructuredPlan
    from saealib.execution.runtime import _OptimizerEnvironment, resolve_plan

    seen: list[str] = []
    optimizer = _optimizer()
    optimizer.run()
    plan = resolve_plan(optimizer)
    environment = _OptimizerEnvironment(
        optimizer, StructuredPlan.from_executable_plan(plan)
    )
    optimizer._executable_plan = None
    optimizer.set_algorithm(_ga(), transition=_Recorder(seen, "outside"))

    environment._refresh_plan_if_needed(None)

    assert seen == ["outside:apply"]
    assert environment.active_version == 1
    assert environment._last_state is None


class _AddThenDropReserved(Transition):
    """Stages a column addition and then a drop the collection refuses."""

    def apply(self, tctx: TransitionContext) -> Any:
        staged = tctx.population
        assert staged is not None
        staged.add_columns(
            [PopulationAttribute("review_probe", np.float64, (), np.nan)]
        )
        staged.drop_columns(["x"])
        yield from ()


class _AddUnfillable(Transition):
    """Stages two additions; the second cannot fill its column."""

    def apply(self, tctx: TransitionContext) -> Any:
        staged = tctx.population
        assert staged is not None
        staged.add_columns(
            [PopulationAttribute("review_probe", np.float64, (), np.nan)]
        )
        staged.add_columns([PopulationAttribute("review_bad", np.int64, (), "text")])
        yield from ()


def _collection_snapshot(state: Any) -> list[tuple[Any, ...]]:
    snapshot = []
    for collection in (state.population, state.archive, state.pareto_archive):
        snapshot.append(
            (
                sorted(collection.schema),
                {name: collection.get_array(name).copy() for name in collection.schema},
                collection.structure_version,
                collection.value_version,
            )
        )
    return snapshot


def _assert_staged_apply_failure_keeps_the_state(
    optimizer: Any, environment: Any, state: Any, transition: Any
) -> None:
    snapshot = _collection_snapshot(state)
    version = environment.active_version
    published = optimizer._last_published_bindings
    optimizer.set_algorithm(_ga(), transition=transition)

    with pytest.raises(ValidationError):
        for _state in _capturing_runner(optimizer)[0].iterate_from(state):
            pass

    after = _collection_snapshot(state)
    for (names, values, structure, value), (later, later_values, s2, v2) in zip(
        snapshot, after
    ):
        assert later == names
        assert s2 == structure
        assert v2 == value
        for name, array in values.items():
            np.testing.assert_array_equal(later_values[name], array)
    assert environment.active_version == version
    assert type(optimizer.algorithm) is GA
    assert published is not None
    assert published.unchanged_by(
        environment._bindings(),
        environment._strategy_values(),
        environment._feature_encoder_selection(),
        optimizer._preset,
    )


def test_a_staged_reserved_drop_leaves_the_collections_unchanged() -> None:
    optimizer = _optimizer()
    runner, captured = _capturing_runner(optimizer)
    state = next(runner.iterate())

    _assert_staged_apply_failure_keeps_the_state(
        optimizer, captured["environment"], state, _AddThenDropReserved()
    )


def test_a_staged_column_that_cannot_be_filled_restores_the_applied_columns() -> None:
    optimizer = _optimizer()
    runner, captured = _capturing_runner(optimizer)
    state = next(runner.iterate())

    _assert_staged_apply_failure_keeps_the_state(
        optimizer, captured["environment"], state, _AddUnfillable()
    )


# ---------------------------------------------------------------------------
# One commit memento: every failure restores the same state vector
# ---------------------------------------------------------------------------

_COMMIT_FAULT_STEPS = (
    "compile",
    "gate",
    "candidate_checks",
    "region_history",
    "transition",
    "post_validation",
    "staged_apply",
    "publish_resolution",
    "publish_record",
    "publish_comparator",
    "publish_scheduler",
)


def _four_generations(state: Any) -> bool:
    return state.gen >= 4


def _surrogate_optimizer(seed: int = 7) -> Optimizer:
    from saealib.acquisition import MeanPrediction
    from saealib.surrogate import GaussianKernel, LocalSurrogateManager, RBFSurrogate
    from saealib.surrogate.training_set import KNNObjectiveSet

    return (
        Optimizer(_problem(), seed=seed)
        .set_initializer(LHSInitializer(8, 6, seed))
        .set_algorithm(_ga())
        .set_surrogate_manager(
            LocalSurrogateManager(
                RBFSurrogate(GaussianKernel()), training_set=KNNObjectiveSet(6)
            )
        )
        .set_acquisition(MeanPrediction(direction=_problem().direction))
        .set_strategy(DirectStrategy())
        .set_termination(Termination(_four_generations))
    )


class _ProbeColumn(Transition):
    """Adds one column to the three collections, declaring what it produces."""

    def produced_columns(self, new_bindings: Any) -> frozenset[str]:
        return frozenset({"probe"})

    def apply(self, tctx: TransitionContext) -> Any:
        for name in ("population", "archive", "pareto_archive"):
            staged = getattr(tctx, name)
            assert staged is not None
            staged.add_columns([PopulationAttribute("probe", np.float64, (), 0.0)])
        yield from ()


class _RefusingComparator(SingleObjectiveComparator):
    """Comparator whose ``eps_cv`` cannot be bound once ``refuse`` is set."""

    refuse = False

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "eps_cv" and getattr(self, "refuse", False):
            raise RuntimeError("comparator cannot bind eps_cv")
        super().__setattr__(name, value)


def _collection_values(state: Any) -> list[tuple[Any, ...]]:
    return [
        (
            tuple(sorted(collection.schema)),
            {name: collection.get_array(name).copy() for name in collection.schema},
            collection.structure_version,
            collection.value_version,
        )
        for collection in (state.population, state.archive, state.pareto_archive)
    ]


def _commit_vector(
    optimizer: Any, environment: Any, state: Any
) -> tuple[dict[str, Any], list[tuple[Any, ...]]]:
    snapshot = environment._active_snapshot
    history = state.history
    encoder_default = optimizer._feature_encoder_default
    comparator_default = optimizer._comparator_default
    return (
        {
            "slots": {
                name: id(getattr(optimizer, name, None)) for name in BINDING_SLOTS
            },
            "counter": optimizer._configuration_version,
            "origins": {
                name: (id(origin.value), origin.source, origin.reason)
                for name, origin in optimizer._component_origins.items()
            },
            "materialized": {
                name: id(value)
                for name, value in optimizer._materialized_components.items()
            },
            "binding_record": id(optimizer._last_published_bindings),
            "published": id(environment._published),
            "snapshot": id(snapshot),
            "snapshot_version": snapshot.version,
            "comparator": id(state.comparator),
            "executable_plan": id(optimizer._executable_plan),
            "resolution": id(optimizer._component_resolution),
            "default_resolution": id(optimizer._default_resolution),
            "feature_encoder_default": (
                None
                if encoder_default is None
                else (id(encoder_default[0]), id(encoder_default[1]))
            ),
            "comparator_default": (
                None
                if comparator_default is None
                else (id(comparator_default[0]), id(comparator_default[1]))
            ),
            "feedback_flag": optimizer.feedback_builder_explicit,
            "region_counts": dict(state.region_counts),
            "region_ids": () if history is None else tuple(history._region_ids),
            "history_columns": (
                () if history is None else tuple(sorted(history.channel("summary")))
            ),
            "compiled_services": {
                name: id(value)
                for name, value in (
                    state.__dict__.get("_compiled_services") or {}
                ).items()
            }
            | {"__present__": "_compiled_services" in state.__dict__},
            "scheduler": {
                name: id(value)
                for name, value in environment._scheduler_state().items()
                if not name.endswith("accumulator_state")
            },
            "active_version": environment.active_version,
            "binding_version": state.__dict__.get("_binding_version"),
            "unchanged_by": snapshot.unchanged_by(
                environment._bindings(),
                environment._strategy_values(),
                environment._feature_encoder_selection(),
                optimizer._preset,
            ),
        },
        _collection_values(state),
    )


def _assert_commit_restored(before: Any, after: Any) -> None:
    before_state, before_collections = before
    after_state, after_collections = after
    assert after_state == before_state
    assert len(after_collections) == len(before_collections)
    for before_collection, after_collection in zip(
        before_collections, after_collections
    ):
        names, values, structure_version, value_version = before_collection
        later_names, later_values, later_structure, later_value = after_collection
        assert later_names == names
        assert (later_structure, later_value) == (structure_version, value_version)
        for name, array in values.items():
            np.testing.assert_array_equal(later_values[name], array)


def _wrap_after(monkeypatch: Any, target: Any, name: str, step: str) -> None:
    original = getattr(target, name)

    def replacement(*args: Any, **kwargs: Any) -> Any:
        original(*args, **kwargs)
        raise RuntimeError(f"fault injected in {step}")

    monkeypatch.setattr(target, name, replacement)


def _inject_commit_fault(
    monkeypatch: Any, step: str, environment: Any, optimizer: Any
) -> None:
    runtime_module = __import__(
        "saealib.execution.runtime", fromlist=["_StagedCollections"]
    )
    if step == "compile":
        _wrap_after(monkeypatch, environment, "_compile_current_plan", step)
    elif step == "gate":
        _wrap_after(monkeypatch, environment, "_gate", step)
    elif step == "candidate_checks":
        _wrap_after(monkeypatch, environment, "_check_candidate_state", step)
    elif step == "region_history":
        _wrap_after(monkeypatch, runtime_module, "_ensure_region_history", step)
    elif step == "transition":
        _wrap_after(monkeypatch, environment, "_run_transitions", step)
    elif step == "post_validation":
        _wrap_after(monkeypatch, environment, "_check_staged_changes", step)
    elif step == "staged_apply":
        _wrap_after(monkeypatch, runtime_module._StagedCollections, "apply", step)
    elif step == "publish_resolution":
        _wrap_after(monkeypatch, optimizer, "_publish_component_resolution", step)
    elif step == "publish_record":
        _wrap_after(monkeypatch, environment, "_remember_published_bindings", step)
    elif step == "publish_comparator":
        _wrap_after(monkeypatch, runtime_module, "prepare_state_comparator", step)
    elif step == "publish_scheduler":
        _wrap_after(monkeypatch, environment, "_publish_scheduler", step)
    else:  # pragma: no cover - the parametrization lists every step
        raise AssertionError(f"unknown commit step {step!r}")


@pytest.mark.parametrize("step", _COMMIT_FAULT_STEPS)
def test_a_failure_at_any_commit_step_restores_the_published_state(
    step: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    optimizer = _surrogate_optimizer()
    runner, captured = _capturing_runner(optimizer)
    generator = runner.iterate()
    state = next(generator)
    environment = captured["environment"]
    before = _commit_vector(optimizer, environment, state)
    optimizer.set_strategy(GenerationBasedStrategy(1), transition=_ProbeColumn())
    optimizer.set_algorithm(_ga())
    optimizer.set_comparator(SingleObjectiveComparator())
    _inject_commit_fault(monkeypatch, step, environment, optimizer)

    with pytest.raises(RuntimeError, match="fault injected"):
        next(generator)
    generator.close()

    _assert_commit_restored(before, _commit_vector(optimizer, environment, state))
    assert optimizer._pending_replacements() == {}


def test_a_comparator_that_refuses_its_binding_restores_the_commit() -> None:
    optimizer = _optimizer()
    runner, captured = _capturing_runner(optimizer)
    generator = runner.iterate()
    state = next(generator)
    environment = captured["environment"]
    before = _commit_vector(optimizer, environment, state)
    optimizer.set_algorithm(_ga(), transition=_ProbeColumn())
    optimizer.set_strategy(DirectStrategy())
    comparator = _RefusingComparator()
    comparator.refuse = True
    optimizer.set_comparator(comparator)

    with pytest.raises(RuntimeError, match="cannot bind eps_cv"):
        next(generator)
    generator.close()

    _assert_commit_restored(before, _commit_vector(optimizer, environment, state))
    assert environment.active_version == 0
    assert type(optimizer.algorithm) is GA


def test_a_failure_while_publishing_the_scheduler_restores_its_attributes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from saealib.execution import AsyncEvaluationScheduler, SerialEvaluator

    optimizer = _optimizer()
    optimizer.set_async_evaluation_scheduler(
        AsyncEvaluationScheduler(SerialEvaluator(), max_pending=2)
    )
    optimizer.resolve_defaults()
    initializer = optimizer.initializer
    assert initializer is not None
    state = initializer.initialize(optimizer, optimizer.problem)
    runtime = cast(Any, create_runtime(optimizer))
    environment = runtime.environment
    assert environment is not None
    environment._remember_state(state)
    scheduler = cast(Any, optimizer.async_evaluation_scheduler)
    before = _commit_vector(optimizer, environment, state)
    expected = environment._scheduler_state()
    assert expected["algorithm"] is not None
    optimizer.set_algorithm(_ga(), transition=_ProbeColumn())

    def fail(*args: Any, **kwargs: Any) -> None:
        scheduler.algorithm = "published"
        scheduler.callback_manager = "published"
        scheduler.feedback_builder = "published"
        raise RuntimeError("fault injected in publish_scheduler")

    monkeypatch.setattr(environment, "_publish_scheduler", fail)

    with pytest.raises(RuntimeError, match="fault injected"):
        environment._refresh_plan_if_needed(state)

    _assert_commit_restored(before, _commit_vector(optimizer, environment, state))
    restored = environment._scheduler_state()
    for name, value in expected.items():
        if name.endswith("accumulator_state"):
            continue
        assert restored[name] is value


def test_a_rejected_strategy_transition_adds_no_region_history() -> None:
    optimizer = _surrogate_optimizer()
    runner, captured = _capturing_runner(optimizer)
    generator = runner.iterate()
    state = next(generator)
    environment = captured["environment"]
    before = _commit_vector(optimizer, environment, state)
    assert before[0]["region_counts"] == {}

    def explode(tctx: TransitionContext) -> Any:
        raise RuntimeError("strategy transition refused")
        yield

    optimizer.set_strategy(
        GenerationBasedStrategy(1), transition=Transition.from_function(explode)
    )

    with pytest.raises(RuntimeError, match="refused"):
        next(generator)
    generator.close()

    history = state.history
    assert history is not None
    assert dict(state.region_counts) == {}
    assert tuple(history._region_ids) == ()
    assert not any(
        column.startswith("region_") for column in history.channel("summary")
    )
    _assert_commit_restored(before, _commit_vector(optimizer, environment, state))
    assert type(optimizer.strategy) is DirectStrategy


# ---------------------------------------------------------------------------
# A rejected randomized comparator leaves the parent seed sequence alone
# ---------------------------------------------------------------------------


def _three_objective_problem(dim: int = 3) -> Problem:
    def objectives(x: Any) -> np.ndarray:
        values = np.asarray(x, dtype=float)
        return np.array(
            [
                float(np.sum(values**2)),
                float(np.sum((values - 0.5) ** 2)),
                float(np.sum((values + 0.5) ** 2)),
            ],
            dtype=np.float64,
        )

    return Problem(
        func=objectives,
        dim=dim,
        n_obj=3,
        direction=np.array([-1.0, -1.0, -1.0]),
        lb=[-5.0] * dim,
        ub=[5.0] * dim,
        constraints=[],
    )


def _nsga3_optimizer(seed: int = 7) -> Optimizer:
    return (
        Optimizer(_three_objective_problem(), seed=seed)
        .set_initializer(LHSInitializer(6, 4, seed))
        .set_algorithm(_ga())
        .set_strategy(DirectStrategy())
        .set_termination(Termination(_four_generations))
    )


def _publish_randomized_comparator(
    monkeypatch: pytest.MonkeyPatch, *, fail_first: bool
) -> dict[str, Any]:
    optimizer = _nsga3_optimizer()
    runner, captured = _capturing_runner(optimizer)
    generator = runner.iterate()
    state = next(generator)
    parent = cast(Any, state.rng.bit_generator.seed_seq)
    children_before = parent.n_children_spawned
    rejected = NSGA3Comparator(np.eye(3))
    optimizer.set_comparator(rejected)
    if fail_first:
        environment = captured["environment"]
        original = environment._publish_scheduler

        def fail(*args: Any, **kwargs: Any) -> None:
            original(*args, **kwargs)
            raise RuntimeError("fault injected in publish_scheduler")

        monkeypatch.setattr(environment, "_publish_scheduler", fail)
        with pytest.raises(RuntimeError, match="fault injected"):
            next(generator)
        assert parent.n_children_spawned == children_before
        assert state.comparator is not rejected
        monkeypatch.undo()
        accepted = NSGA3Comparator(np.eye(3))
        optimizer.set_comparator(accepted)
        resumed, resumed_captured = _capturing_runner(optimizer)
        resumed_generator = resumed.iterate_from(state)
        next(resumed_generator)
        published = next(resumed_generator)
        resumed_generator.close()
        environment = resumed_captured["environment"]
    else:
        accepted = NSGA3Comparator(np.eye(3))
        optimizer.set_comparator(accepted)
        published = next(generator)
        generator.close()
        environment = captured["environment"]
    return {
        "spawned": parent.n_children_spawned - children_before,
        "spawn_key": tuple(cast(Any, accepted.rng.bit_generator.seed_seq).spawn_key),
        "ordering": tuple(
            int(index) for index in accepted.sort_population(published.population)
        ),
        "values": np.asarray(published.population.get_array("f")).copy(),
        "version": environment.active_version,
    }


def test_a_rejected_nsga3_replacement_keeps_the_next_child_seed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    control = _publish_randomized_comparator(monkeypatch, fail_first=False)
    rejected = _publish_randomized_comparator(monkeypatch, fail_first=True)

    np.testing.assert_array_equal(rejected["values"], control["values"])
    assert control["spawned"] == 1
    for key in ("spawned", "spawn_key", "ordering", "version"):
        assert rejected[key] == control[key]
