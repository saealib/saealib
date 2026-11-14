"""Rebuilding re-inserted engines and checking the pymoo population size."""

from __future__ import annotations

import pickle
import threading
from typing import Any

import numpy as np
import pytest

from saealib import (
    PSO,
    LHSInitializer,
    Optimizer,
    PymooAlgorithm,
    Termination,
    max_gen,
)
from saealib.algorithms import GA
from saealib.exceptions import ConfigurationError, ReplacementRejectedError
from saealib.execution import AsyncEvaluationScheduler, SerialEvaluator
from saealib.execution.runner import Runner
from saealib.operators import CrossoverBLXAlpha, MutationUniform
from saealib.operators.selection import SequentialSelection, TruncationSelection
from saealib.problem import Problem
from saealib.strategies import DirectStrategy


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


def _ga() -> GA:
    return GA(
        crossover=CrossoverBLXAlpha(prob=0.9, alpha=0.4),
        mutation=MutationUniform(prob_var=0.1),
        parent_selection=SequentialSelection(),
        survivor_selection=TruncationSelection(),
    )


def _optimizer(
    algorithm: Any,
    *,
    population: int = 4,
    max_generation: int = 8,
    asynchronous: bool = False,
) -> Optimizer:
    optimizer = (
        Optimizer(_problem(), seed=7)
        .set_initializer(LHSInitializer(6, population, 7))
        .set_algorithm(algorithm)
        .set_strategy(DirectStrategy())
        .set_termination(Termination(max_gen(max_generation)))
    )
    if asynchronous:
        evaluator = SerialEvaluator()
        optimizer.set_evaluator(evaluator)
        optimizer.set_async_evaluation_scheduler(
            AsyncEvaluationScheduler(evaluator, max_pending=1)
        )
    return optimizer


def _assert_published_configuration_is_restored(optimizer: Any) -> None:
    from saealib.execution.prepare import BINDING_SLOTS

    snapshot = optimizer._last_published_bindings
    assert snapshot is not None
    current = {name: getattr(optimizer, name, None) for name in BINDING_SLOTS}
    assert snapshot.unchanged_by(
        current,
        snapshot.strategy_values,
        snapshot.feature_encoder_selection,
        snapshot.preset,
    )


def _pymoo(pop_size: int = 4, *, allow_partial_tell: bool = False) -> Any:
    from pymoo.algorithms.moo.nsga2 import NSGA2

    return PymooAlgorithm(
        NSGA2(pop_size=pop_size), allow_partial_tell=allow_partial_tell
    )


def _run_reinsertion(
    first: Any,
    replacement_factory: Any,
    *,
    asynchronous: bool = False,
    max_generation: int = 8,
) -> list[dict[str, np.ndarray]]:
    optimizer = _optimizer(
        first, asynchronous=asynchronous, max_generation=max_generation
    )
    yields: list[dict[str, np.ndarray]] = []
    for state in Runner(optimizer).iterate():
        yields.append(
            {
                name: np.array(state.population.get_array(name), copy=True)
                for name in ("x", "f", "id")
            }
        )
        if state.gen == 2:
            optimizer.set_algorithm(_ga())
        elif state.gen == 4:
            optimizer.set_algorithm(replacement_factory())
    return yields


def test_reinserted_pymoo_matches_a_fresh_instance_every_generation() -> None:
    engine = _pymoo()
    reinserted = _run_reinsertion(engine, lambda: engine)
    fresh = _run_reinsertion(_pymoo(), _pymoo)

    assert len(reinserted) == len(fresh)
    for expected, actual in zip(reinserted, fresh, strict=True):
        for name in expected:
            np.testing.assert_array_equal(expected[name], actual[name])


def test_async_reinserted_pymoo_with_partial_tell_matches_a_fresh_instance() -> None:
    engine = _pymoo(allow_partial_tell=True)
    reinserted = _run_reinsertion(
        engine, lambda: engine, asynchronous=True, max_generation=6
    )
    fresh = _run_reinsertion(
        _pymoo(allow_partial_tell=True),
        lambda: _pymoo(allow_partial_tell=True),
        asynchronous=True,
        max_generation=6,
    )

    assert len(reinserted) == len(fresh)
    for expected, actual in zip(reinserted, fresh, strict=True):
        for name in expected:
            np.testing.assert_array_equal(expected[name], actual[name])


def test_first_insertion_of_a_modified_pymoo_instance_keeps_its_pop_size() -> None:
    pymoo = _pymoo()
    pymoo.pymoo_algorithm.pop_size = 6
    optimizer = _optimizer(_ga(), population=6, max_generation=4)

    for state in Runner(optimizer).iterate():
        if state.gen == 2:
            optimizer.set_algorithm(pymoo)

    assert pymoo._used
    assert pymoo.pymoo_algorithm.pop_size == 6


def test_a_pymoo_population_size_mismatch_is_rejected_at_the_commit_point() -> None:
    pymoo = _pymoo(pop_size=4)
    optimizer = _optimizer(_ga(), population=6, max_generation=4)
    generator = Runner(optimizer).iterate()
    states: list[Any] = []
    for state in generator:
        states.append(state)
        if state.gen == 2:
            break
    snapshot = optimizer._last_published_bindings
    assert snapshot is not None
    version = snapshot.version
    columns = frozenset(states[-1].population.schema)
    evaluations = states[-1].fe
    optimizer.set_algorithm(pymoo)

    with pytest.raises(ReplacementRejectedError) as error:
        next(generator)

    assert "pymoo_pop_size_mismatch" in {
        diagnostic.code for diagnostic in error.value.diagnostics
    }
    assert optimizer.algorithm is not pymoo
    kept = optimizer._last_published_bindings
    assert kept is not None
    assert kept.version == version
    assert frozenset(states[-1].population.schema) == columns
    assert states[-1].fe == evaluations

    final = Runner(optimizer).run_from(states[-1])
    assert final.gen == 4


def test_a_pymoo_population_size_that_matches_the_run_is_published() -> None:
    optimizer = _optimizer(_ga(), population=6, max_generation=4)
    pymoo = _pymoo(pop_size=6)
    generator = Runner(optimizer).iterate()
    for state in generator:
        if state.gen == 2:
            optimizer.set_algorithm(pymoo)
            break
    next(generator)

    assert optimizer.algorithm is pymoo
    assert "pymoo_idx" in state.population.schema


def test_a_pymoo_algorithm_with_the_wrong_pop_size_stops_at_the_entry() -> None:
    pymoo = _pymoo(pop_size=6)
    optimizer = _optimizer(pymoo, population=4, max_generation=4)

    with pytest.raises(ConfigurationError) as error:
        list(Runner(optimizer).iterate())

    assert "pymoo_pop_size_mismatch" in str(error.value)
    assert not pymoo._used


@pytest.mark.parametrize("case", ["modified_settings", "missing_pristine"])
@pytest.mark.parametrize("operation", ["strategy", "resume"])
def test_a_continuing_pymoo_engine_is_not_checked_as_a_reinsertion(
    case: str, operation: str
) -> None:
    pymoo = _pymoo(pop_size=4)
    population = 4
    if case == "modified_settings":
        pymoo.pymoo_algorithm.pop_size = 6
        population = 6
    else:
        pymoo._pristine = None
    optimizer = _optimizer(pymoo, population=population, max_generation=6)
    generator = Runner(optimizer).iterate()
    state = None
    for state in generator:
        if state.gen == 2:
            break
    if operation == "strategy":
        optimizer.set_strategy(DirectStrategy(1))
        for _state in generator:
            pass
        final = _state
        assert optimizer.algorithm is pymoo
    else:
        assert state is not None
        final = Runner(optimizer).run_from(state)
    assert final.gen == 6


def test_a_pymoo_algorithm_without_pop_size_is_checked_by_its_first_population() -> (
    None
):
    from pymoo.algorithms.soo.nonconvex.nelder import NelderMead

    optimizer = _optimizer(_ga(), population=4, max_generation=4)
    generator = Runner(optimizer).iterate()
    for state in generator:
        if state.gen == 2:
            optimizer.set_algorithm(PymooAlgorithm(NelderMead()))
            break

    with pytest.raises(ReplacementRejectedError) as error:
        next(generator)
    assert "pymoo_pop_size_mismatch" in {
        diagnostic.code for diagnostic in error.value.diagnostics
    }


def test_a_pymoo_instance_without_a_construction_copy_is_rejected_on_reinsertion() -> (
    None
):
    pymoo = _pymoo()
    pymoo._pristine = None
    optimizer = _optimizer(_ga(), population=4, max_generation=4)
    generator = Runner(optimizer).iterate()
    states: list[Any] = []
    for state in generator:
        states.append(state)
        if state.gen == 2:
            break
    optimizer.set_algorithm(pymoo)
    next(generator)
    optimizer.set_algorithm(_ga())
    next(generator)
    optimizer.set_algorithm(pymoo)

    with pytest.raises(ReplacementRejectedError) as error:
        next(generator)

    assert "pristine_copy_unavailable" in {
        diagnostic.code for diagnostic in error.value.diagnostics
    }
    final = Runner(optimizer).run_from(states[-1])
    assert final.gen == 4


def test_a_rebuild_transition_does_not_make_alignment_declare_nothing() -> None:
    from saealib import AlignColumns, chain

    pymoo = _pymoo()
    composed = chain(AlignColumns(_problem()), pymoo._replacement_transition())

    declared = composed.produced_columns({"algorithm": pymoo})

    assert declared == frozenset({"pymoo_idx"})


def test_a_failing_alignment_restores_the_published_bindings(
    monkeypatch: Any,
) -> None:
    from saealib.transition import AlignColumns

    def failing_apply(self: Any, tctx: Any) -> Any:
        del self, tctx
        raise RuntimeError("alignment failed")
        yield from ()

    monkeypatch.setattr(AlignColumns, "apply", failing_apply)
    optimizer = _optimizer(_ga(), max_generation=4)
    generator = Runner(optimizer).iterate()
    for state in generator:
        if state.gen == 2:
            optimizer.set_algorithm(PSO())
            break

    with pytest.raises(RuntimeError, match="alignment failed"):
        next(generator)

    snapshot = optimizer._last_published_bindings
    assert snapshot is not None
    assert optimizer.algorithm is not None
    assert snapshot.values["algorithm"] is optimizer.algorithm
    assert optimizer._configuration_version == snapshot.configuration_version
    _assert_published_configuration_is_restored(optimizer)


def test_pickling_an_optimizer_with_a_pymoo_algorithm_round_trips() -> None:
    pymoo = _pymoo()
    optimizer = _optimizer(pymoo, max_generation=3)
    restored = pickle.loads(pickle.dumps(optimizer))

    state = Runner(restored).run()

    assert state.gen == 3


def _deap_strategy(lambda_: int = 4, dim: int = 2) -> Any:
    pytest.importorskip("deap")
    from deap import cma

    return cma.Strategy([0.0] * dim, 1.0, lambda_=lambda_)


def _deap(strategy: Any | None = None) -> Any:
    from saealib.algorithms import DeapGenerateUpdateAlgorithm

    return DeapGenerateUpdateAlgorithm(strategy or _deap_strategy())


def _run_deap_reinsertion(
    first: Any,
    replacement_factory: Any,
    *,
    asynchronous: bool = False,
    max_generation: int = 8,
) -> list[dict[str, np.ndarray]]:
    optimizer = _optimizer(
        first, asynchronous=asynchronous, max_generation=max_generation
    )
    yields: list[dict[str, np.ndarray]] = []
    for state in Runner(optimizer).iterate():
        yields.append(
            {
                name: np.array(state.population.get_array(name), copy=True)
                for name in ("x", "f", "id")
            }
        )
        if state.gen == 2:
            optimizer.set_algorithm(_ga())
        elif state.gen == 4:
            optimizer.set_algorithm(replacement_factory())
    return yields


@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
def test_reinserted_deap_strategy_matches_a_fresh_strategy(
    asynchronous: bool,
) -> None:
    engine = _deap()
    reinserted = _run_deap_reinsertion(
        engine, lambda: engine, asynchronous=asynchronous, max_generation=6
    )
    fresh = _run_deap_reinsertion(
        _deap(), _deap, asynchronous=asynchronous, max_generation=6
    )

    assert len(reinserted) == len(fresh)
    for expected, actual in zip(reinserted, fresh, strict=True):
        for name in expected:
            np.testing.assert_array_equal(expected[name], actual[name])


class _LockedStrategy:
    """A generate/update strategy whose lock prevents a construction copy."""

    def __init__(self) -> None:
        self._strategy = _deap_strategy()
        self._lock = threading.Lock()
        self.centroid = self._strategy.centroid
        self.sigma = self._strategy.sigma

    def generate(self, ind_init: Any) -> Any:
        with self._lock:
            return self._strategy.generate(ind_init)

    def update(self, population: Any) -> None:
        with self._lock:
            self._strategy.update(population)


def test_a_strategy_without_a_construction_copy_runs_first_and_is_rejected_later() -> (
    None
):
    from saealib.algorithms import DeapGenerateUpdateAlgorithm

    locked = _LockedStrategy()
    adapter = DeapGenerateUpdateAlgorithm(locked)
    assert adapter._pristine is None
    optimizer = _optimizer(_ga(), population=4, max_generation=4)
    generator = Runner(optimizer).iterate()
    states: list[Any] = []
    for state in generator:
        states.append(state)
        if state.gen == 2:
            break
    optimizer.set_algorithm(adapter)
    next(generator)
    optimizer.set_algorithm(_ga())
    next(generator)
    optimizer.set_algorithm(adapter)

    with pytest.raises(ReplacementRejectedError) as error:
        next(generator)

    assert "pristine_copy_unavailable" in {
        diagnostic.code for diagnostic in error.value.diagnostics
    }
    _assert_published_configuration_is_restored(optimizer)
    final = Runner(optimizer).run_from(states[-1])
    assert final.gen == 4


def test_pickling_an_optimizer_with_a_deap_algorithm_round_trips() -> None:
    optimizer = _optimizer(_deap(), max_generation=3)
    restored = pickle.loads(pickle.dumps(optimizer))

    state = Runner(restored).run()

    assert state.gen == 3
