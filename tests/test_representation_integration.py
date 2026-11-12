"""Longitudinal checks for representation-neutral execution."""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from saealib.algorithms import GA, GenomeGA
from saealib.core.contracts import (
    Fixed,
    ParameterSpec,
    RepresentationKind,
    RepresentationSpec,
)
from saealib.core.contracts.representation import REPRESENTATION_KINDS
from saealib.exceptions import ValidationError
from saealib.execution import AsyncEvaluationScheduler, AsyncEvaluator, SerialEvaluator
from saealib.execution.evaluator import EvaluationQuery
from saealib.operators import OrderCrossover, SequenceMutation, SwapMutation
from saealib.operators.crossover import Crossover, CrossoverSBX
from saealib.operators.mutation import Mutation, MutationUniform
from saealib.operators.selection import SequentialSelection, TruncationSelection
from saealib.optimizer import Optimizer
from saealib.population import ObjectBatch
from saealib.population.genome import GenomeBatch, PermutationBatch, VariableLengthBatch
from saealib.problem import Problem
from saealib.space import (
    PermutationSpace,
    SequenceSpace,
    ServiceRegistry,
    ValidationResult,
    VectorSpace,
)
from saealib.space.space import encode_features
from saealib.strategies.direct import DirectStrategy
from saealib.strategies.ib import IndividualBasedStrategy
from saealib.surrogate.base import RegressionSurrogate
from saealib.surrogate.manager import GlobalSurrogateManager
from saealib.surrogate.prediction import PredictionChannel, SurrogatePrediction
from saealib.termination import Termination, max_gen


def test_permutation_profile_sampling_validation_codec_and_variation() -> None:
    space = PermutationSpace(4)
    batch = space.sample(4, np.random.default_rng(4))
    assert isinstance(batch, PermutationBatch)
    assert space.validate(batch).valid
    with pytest.raises(ValidationError):
        space.validate(ObjectBatch([None]))
    codec = cast(Any, space.services.require("GenomeCodec"))
    restored = codec.decode(codec.encode(batch))
    np.testing.assert_array_equal(restored.array, batch.array)
    children = OrderCrossover().apply(
        PermutationBatch([[0, 1, 2, 3], [3, 2, 1, 0]], length=4),
        np.random.default_rng(1),
    )
    assert isinstance(children, PermutationBatch)
    assert space.validate(children).valid
    mutated = SwapMutation().apply(children, np.random.default_rng(2))
    assert isinstance(mutated, PermutationBatch)
    assert space.validate(mutated).valid


def test_sequence_profile_services_and_mutation() -> None:
    space = SequenceSpace(("a", "b", "c"), min_length=1, max_length=3)
    batch = space.sample(5, np.random.default_rng(5))
    assert isinstance(batch, VariableLengthBatch)
    assert space.validate(batch).valid
    invalid = VariableLengthBatch([("z",), ("a", "b", "c", "a")])
    result = space.validate(invalid)
    assert result.valid_mask == (False, False) and not result.valid
    codec = cast(Any, space.services.require("GenomeCodec"))
    assert codec.decode(codec.encode(batch)).sequences == batch.sequences
    distance = cast(Any, space.services.require("DistanceService"))
    left = VariableLengthBatch([("a", "b"), ("a",)])
    right = VariableLengthBatch([("a", "c"), ("a", "b", "c")])
    np.testing.assert_array_equal(
        distance.pairwise_distance(left, right), [[1, 1], [1, 2]]
    )
    assert space.services.get("FeatureEncoder") is None
    with pytest.warns(
        UserWarning, match="encode_features used the default FeatureEncoder"
    ):
        features = encode_features(space, left)
    np.testing.assert_array_equal(features, [[2, 1, 2, 0], [1, 1, 0, 0]])
    assert space.services.get("FeatureEncoder") is None
    mutated = SequenceMutation(
        alphabet=space.alphabet, min_length=1, max_length=3
    ).apply(left, np.random.default_rng(7))
    assert space.validate(mutated).valid


def test_vector_standalone_feature_encoder_is_lossless_identity() -> None:
    space = VectorSpace(dim=2, lb=[-1.0, -1.0], ub=[1.0, 1.0])
    batch = space.sample(3, np.random.default_rng(8))
    assert space.services.get("FeatureEncoder") is None
    with pytest.warns(
        UserWarning, match="encode_features used the default FeatureEncoder"
    ):
        features = encode_features(space, batch)
    np.testing.assert_array_equal(features, batch.array)
    assert np.shares_memory(features, batch.array)
    assert space.services.get("FeatureEncoder") is None


def _vector_problem() -> Problem:
    return Problem(
        func=lambda x: np.asarray([np.sum(np.asarray(x, dtype=float) ** 2)]),
        dim=2,
        n_obj=1,
        direction=np.array([-1.0]),
        lb=[-1.0, -1.0],
        ub=[1.0, 1.0],
    )


def _genome_problem(space: Any, func) -> Problem:
    return Problem(
        func=func, dim=space.dim, n_obj=1, direction=np.array([-1.0]), space=space
    )


def _compile(problem: Problem, algorithm: Any, strategy: Any) -> Optimizer:
    optimizer = (
        Optimizer(problem, seed=1).set_algorithm(algorithm).set_strategy(strategy)
    )
    optimizer.resolve_defaults()
    plan = optimizer._compile_plan()
    assert plan is not None
    assert not [
        diagnostic
        for diagnostic in plan.diagnostics
        if diagnostic.severity.value == "error"
    ]
    return optimizer


def test_five_real_graphs_compile_with_visible_defaults() -> None:
    vector_ga = GA(
        CrossoverSBX(0.9, 15.0),
        MutationUniform(),
        SequentialSelection(),
        TruncationSelection(),
    )
    permutation = PermutationSpace(3)
    genome_ga = GenomeGA(
        OrderCrossover(), SwapMutation(), SequentialSelection(), TruncationSelection()
    )
    sequence = SequenceSpace((0, 1), 1, 3)
    sequence_ga = GenomeGA(
        _ExternalCrossover(),
        SequenceMutation(alphabet=(0, 1), min_length=1, max_length=3),
        SequentialSelection(),
        TruncationSelection(),
    )
    _register_custom_kind()
    custom = _CustomSpace()
    profiles = (
        (_vector_problem(), vector_ga, DirectStrategy()),
        (_vector_problem(), _vector_ga(), IndividualBasedStrategy(0.5)),
        (
            _genome_problem(permutation, lambda x: np.asarray([sum(x)])),
            genome_ga,
            DirectStrategy(),
        ),
        (
            _genome_problem(sequence, lambda x: np.asarray([len(x)])),
            sequence_ga,
            DirectStrategy(),
        ),
        (
            _genome_problem(custom, lambda x: np.asarray([float(x)])),
            _custom_ga(),
            DirectStrategy(),
        ),
    )
    assert len(profiles) == 5
    compiled = [
        _compile(problem, algorithm, strategy)
        for problem, algorithm, strategy in profiles
    ]
    findings = [
        tuple(
            diagnostic
            for diagnostic in optimizer.contract_diagnostics()
            if diagnostic.code == "feature_encoder_default_selected"
        )
        for optimizer in compiled
    ]
    assert [len(items) for items in findings] == [0, 1, 0, 0, 0]
    assert findings[1][0].severity.value == "info"


class _CustomBatch:
    def __init__(self, values: Sequence[int] = ()) -> None:
        self.values = tuple(int(v) for v in values)

    def __len__(self) -> int:
        return len(self.values)

    def take(self, indices: Sequence[int]) -> _CustomBatch:
        return type(self)([self.values[int(i)] for i in indices])

    @classmethod
    def concat(cls, batches: Sequence[_CustomBatch]) -> _CustomBatch:
        return cls([value for batch in batches for value in batch.values])


class _CustomCodec:
    def encode(self, genomes: Any) -> dict[str, Any]:
        values = getattr(genomes, "values", None)
        if values is None:
            values = np.asarray(genomes.array)[:, 0]
        return {"values": [int(value) for value in values]}

    def decode(self, payload: dict[str, Any]) -> _CustomBatch:
        return _CustomBatch(payload["values"])


class _CustomFeatureEncoder:
    def encode(self, genomes: Any) -> np.ndarray:
        values = getattr(genomes, "values", None)
        if values is not None:
            return np.asarray(values, dtype=float).reshape(-1, 1)
        return np.asarray(genomes.array, dtype=float).reshape(len(genomes), -1)


class _CustomDistance:
    def pairwise_distance(self, batch1: Any, batch2: Any = None) -> np.ndarray:
        other = batch1 if batch2 is None else batch2
        return np.abs(np.subtract.outer(batch1.values, other.values)).astype(float)

    def create_index(self, genomes: GenomeBatch) -> object:
        return genomes

    def query_knn(
        self, index: Any, genomes: Any, k: int
    ) -> tuple[np.ndarray, np.ndarray]:
        distances = self.pairwise_distance(genomes, index)[0]
        order = np.argsort(distances)[:k]
        return order.astype(np.intp), distances[order]


def _batch_values(genomes: Any) -> tuple[object, ...]:
    values = getattr(genomes, "values", None)
    if values is not None:
        return tuple(values)
    return tuple(tuple(row) for row in genomes.array)


class _CustomFingerprint:
    def fingerprint(self, genomes: Any) -> tuple[object, ...]:
        return _batch_values(genomes)

    def create_index(self) -> dict[object, int]:
        return {}

    def add_to_index(self, index: dict[object, int], genomes: Any) -> None:
        index.update({value: i for i, value in enumerate(_batch_values(genomes))})

    def find_matches(
        self, index: dict[object, int], genomes: GenomeBatch
    ) -> np.ndarray:
        return np.asarray(
            [index.get(value, -1) for value in _batch_values(genomes)], dtype=np.intp
        )


class _CustomEquivalence:
    def find_duplicates(self, genomes: GenomeBatch) -> np.ndarray:
        values = _batch_values(genomes)
        return np.asarray([value in values[:i] for i, value in enumerate(values)])

    def find_matches(self, collection: GenomeBatch, genomes: GenomeBatch) -> np.ndarray:
        values = _batch_values(collection)
        return np.asarray(
            [
                values.index(value) if value in values else -1
                for value in _batch_values(genomes)
            ],
            dtype=np.intp,
        )


class _CustomSpace:
    dim = 1

    def __init__(self) -> None:
        self._representation = RepresentationSpec(
            kind="external_custom",
            parameters=(ParameterSpec(name="width", value=Fixed(value=1)),),
        )
        self._services = ServiceRegistry()
        self._services.register("GenomeCodec", _CustomCodec())
        self._services.register("FeatureEncoder", _CustomFeatureEncoder())
        self._services.register(
            "BoundsService",
            type(
                "Bounds",
                (),
                {
                    "bounds": (np.array([0.0]), np.array([100.0])),
                },
            )(),
        )
        self._services.register("DistanceService", _CustomDistance())
        self._services.register(
            "CloneService",
            type("Clone", (), {"clone": lambda _, g: _CustomBatch(g.values)})(),
        )
        self._services.register("FingerprintService", _CustomFingerprint())
        self._services.register("EquivalenceService", _CustomEquivalence())

    @property
    def representation(self) -> RepresentationSpec:
        return self._representation

    @property
    def services(self) -> ServiceRegistry:
        return self._services

    def sample(self, n: int, rng=None) -> _CustomBatch:
        return _CustomBatch(range(n))

    def validate(self, genomes: GenomeBatch) -> ValidationResult:
        return ValidationResult(
            valid_mask=tuple(
                isinstance(genomes, _CustomBatch) for _ in range(len(genomes))
            )
        )


def _register_custom_kind() -> None:
    if REPRESENTATION_KINDS.get("external_custom") is None:
        REPRESENTATION_KINDS.register(
            "external_custom",
            RepresentationKind(
                name="external_custom",
                description="test-only external representation",
                parameters=(ParameterSpec(name="width", value=Fixed(value=1)),),
            ),
        )


class _ExternalCrossover(Crossover):
    def crossover_batch(self, parents_batch, bounds=None, rng=np.random.default_rng()):
        return np.asarray(parents_batch)[:, ::-1, :]

    def apply(self, parents: Any, rng=np.random.default_rng()) -> Any:
        sequences = getattr(parents, "sequences", None)
        if sequences is not None:
            return VariableLengthBatch(sequences)
        values = getattr(parents, "values", None)
        if values is not None:
            return _CustomBatch(values)
        return _CustomBatch([int(value) for value in np.asarray(parents.array)[:, 0]])


class _ExternalMutation(Mutation):
    def contract(self):
        role = super().contract().ports["mutation"]
        return replace(
            super().contract(),
            ports={
                "mutation": replace(
                    role,
                    inputs=tuple(replace(p, required_services=()) for p in role.inputs),
                )
            },
        )

    def mutate_batch(
        self, candidates_batch, mutate_range=None, rng=np.random.default_rng()
    ):
        return np.asarray(candidates_batch)

    def apply(self, candidates: Any, rng=np.random.default_rng()) -> _CustomBatch:
        return _CustomBatch([v + 1 for v in candidates.values])


def _custom_ga() -> GenomeGA:
    return GenomeGA(
        _ExternalCrossover(),
        _ExternalMutation(),
        SequentialSelection(),
        TruncationSelection(),
    )


def _vector_ga() -> GA:
    return GA(
        CrossoverSBX(0.9, 15.0),
        MutationUniform(),
        SequentialSelection(),
        TruncationSelection(),
    )


class _ExternalAdapter:
    def __init__(self) -> None:
        self.calls = 0

    def transform(self, genomes: Any, request: EvaluationQuery) -> list[float]:
        self.calls += 1
        values = getattr(genomes, "values", None)
        if values is not None:
            return [float(value) for value in values]
        return [float(row[0]) for row in genomes.array]


class _TinySurrogate(RegressionSurrogate):
    def fit(self, train_x: np.ndarray, train_y: np.ndarray) -> None:
        self.mean = float(np.mean(train_y))

    def predict(self, test_x: np.ndarray) -> SurrogatePrediction:
        return SurrogatePrediction(
            {"objective": PredictionChannel(np.full((len(test_x), 1), self.mean))}
        )


def test_external_representation_runs_archive_checkpoint_async_and_global_surrogate(
    tmp_path: Path,
) -> None:
    _register_custom_kind()
    space = _CustomSpace()
    space.services.register(
        "DenseNumericView",
        type(
            "View",
            (),
            {
                "get_view": lambda _, g: _CustomFeatureEncoder().encode(g),
            },
        )(),
    )
    problem = _genome_problem(
        space,
        lambda x: np.asarray([float(x)]),
    )
    problem.evaluation_adapter = _ExternalAdapter()
    evaluator = SerialEvaluator()
    optimizer = (
        Optimizer(problem, seed=3)
        .set_algorithm(_custom_ga())
        .set_strategy(DirectStrategy())
        .set_evaluator(evaluator)
        .set_async_evaluation_scheduler(
            AsyncEvaluationScheduler(evaluator, max_pending=1)
        )
        .set_termination(Termination(max_gen(1)))
    )
    ctx = optimizer.run()
    assert len(ctx.archive) > 0
    manager = GlobalSurrogateManager(_TinySurrogate())
    manager.fit(ctx.archive, ctx=ctx)
    prediction = manager.predict(
        cast(Any, space.services.require("FeatureEncoder")).encode(
            ctx.population.genomes
        ),
        ctx.archive,
        ctx=ctx,
    )
    assert prediction.value.shape[0] == len(ctx.population)
    checkpoint = tmp_path / "external_representation.npz"
    ctx.save(checkpoint)
    loaded = type(ctx).load(checkpoint, problem)
    assert len(loaded.archive) == len(ctx.archive)
    codec = cast(Any, space.services.require("GenomeCodec"))
    assert codec.decode(codec.encode(_CustomBatch([1, 2]))).values == (1, 2)

    async_adapter = _ExternalAdapter()
    problem.evaluation_adapter = async_adapter
    async_evaluator = AsyncEvaluator(SerialEvaluator(), max_workers=1)
    async_optimizer = (
        Optimizer(problem, seed=3)
        .set_algorithm(_custom_ga())
        .set_strategy(DirectStrategy())
        .set_evaluator(async_evaluator)
        .set_termination(Termination(max_gen(1)))
    )
    try:
        async_optimizer.run()
        assert async_adapter.calls > 0
    finally:
        async_evaluator.close()


def test_external_representation_requires_explicit_feature_encoder() -> None:
    _register_custom_kind()
    space = _CustomSpace()
    genomes = space.sample(3, np.random.default_rng(3))
    features = encode_features(cast(Any, space), cast(Any, genomes))
    assert features.shape == (3, 1)
    codec = cast(Any, space.services.require("GenomeCodec"))
    assert codec.decode(codec.encode(_CustomBatch([1, 2]))).values == (1, 2)


def test_external_representation_rejects_missing_feature_encoder() -> None:
    _register_custom_kind()
    space = _CustomSpace()
    services = cast(Any, space.services)
    feature_encoder = services._services.pop("FeatureEncoder")
    try:
        with warnings.catch_warnings(record=True) as emitted:
            warnings.simplefilter("always")
            with pytest.raises(
                ValidationError,
                match="require the FeatureEncoder service",
            ):
                encode_features(cast(Any, space), cast(Any, space.sample(3)))
        assert emitted == []
    finally:
        services.register("FeatureEncoder", feature_encoder)


def test_compiler_has_no_builtin_representation_batch_branches() -> None:
    root = Path(__file__).parents[1] / "src" / "saealib" / "core" / "compiler"
    source = "\n".join(path.read_text(encoding="utf-8") for path in root.glob("*.py"))
    for name in (
        "PermutationBatch",
        "VariableLengthBatch",
        "ObjectBatch",
        "DenseVectorBatch",
    ):
        assert (
            "isinstance(" not in source or name not in source.split("isinstance(", 1)[1]
        )
    assert "isinstance(genomes, PermutationBatch)" not in source
    assert "isinstance(genomes, VariableLengthBatch)" not in source
    assert "isinstance(genomes, ObjectBatch)" not in source
