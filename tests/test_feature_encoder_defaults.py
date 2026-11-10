from __future__ import annotations

import warnings
from typing import cast

import numpy as np
import pytest

from saealib.context import OptimizationState
from saealib.exceptions import CheckpointError, ValidationError
from saealib.execution.bindings import _feature_encoder_signature
from saealib.optimizer import Optimizer
from saealib.population.genome import (
    DenseVectorBatch,
    PermutationBatch,
    VariableLengthBatch,
)
from saealib.problem import Problem
from saealib.registry import register
from saealib.space import (
    ObjectSpace,
    PermutationSpace,
    SearchSpace,
    SequenceSpace,
    VectorSpace,
)
from saealib.space.space import _encode_features, encode_features
from saealib.strategies import DirectStrategy, GenerationBasedStrategy
from saealib.termination import Termination, max_gen


def _sphere(x: np.ndarray) -> np.ndarray:
    return np.array([float(np.sum(np.asarray(x) ** 2))])


@register(name="CountingFeatureEncoder")
class CountingFeatureEncoder:
    def __init__(self) -> None:
        self.calls = 0

    def encode(self, genomes):
        self.calls += 1
        return np.asarray(genomes.array)


@register(name="ScaleFeatureEncoder")
class ScaleFeatureEncoder:
    def __init__(self, scale: float = 2.0) -> None:
        self.scale = scale
        self.calls = 0

    @property
    def _registry_spec(self) -> dict[str, object]:
        return {"type": "ScaleFeatureEncoder", "params": {"scale": self.scale}}

    def encode(self, genomes):
        self.calls += 1
        return np.asarray(genomes.array) * self.scale


class ExtendedVectorSpace(VectorSpace):
    pass


def _optimizer(space: object, seed: int = 11) -> Optimizer:
    problem = Problem(
        func=_sphere,
        dim=getattr(space, "dim", 2),
        n_obj=1,
        direction=np.array([-1.0]),
        space=cast(SearchSpace, space),
    )
    return Optimizer(problem, seed=seed)


def _feature_encoder(optimizer: Optimizer) -> object:
    plan = optimizer._compile_plan()
    assert plan is not None
    return next(
        node.resolved_services["FeatureEncoder"]
        for node in plan.graph.nodes
        if "FeatureEncoder" in node.resolved_services
    )


@pytest.mark.parametrize(
    ("space", "severity", "required_message"),
    [
        (VectorSpace(dim=2, lb=[-1, -1], ub=[1, 1]), "info", "identity"),
        (PermutationSpace(length=3), "warning", "explicit choice is recommended"),
        (
            SequenceSpace(["a", "b"], max_length=3),
            "warning",
            "explicit choice is recommended",
        ),
    ],
)
def test_optimizer_selects_visible_default_feature_encoders(
    space: object, severity: str, required_message: str
) -> None:
    services = getattr(space, "services")
    assert services.get("FeatureEncoder") is None
    optimizer = _optimizer(space)

    plan = optimizer._compile_plan()

    assert plan is not None
    findings = [
        item
        for item in plan.diagnostics
        if item.code == "feature_encoder_default_selected"
    ]
    assert len(findings) == 1
    diagnostic = findings[0]
    assert diagnostic.severity.value == severity
    assert required_message in diagnostic.message
    assert "framework default" in diagnostic.message
    assert not any(item.severity.value == "error" for item in findings)
    assert services.get("FeatureEncoder") is None
    issues = optimizer.validate()
    assert not any("feature_encoder_default_selected" in issue for issue in issues)
    assert any(
        item.code == "feature_encoder_default_selected"
        and item.severity.value == severity
        for item in optimizer.last_contract_diagnostics
    )


def test_inherited_vector_space_uses_vector_default() -> None:
    class DerivedVectorSpace(VectorSpace):
        pass

    optimizer = _optimizer(DerivedVectorSpace(dim=2, lb=[-1, -1], ub=[1, 1]))
    plan = optimizer._compile_plan()

    assert plan is not None
    assert any(
        item.code == "feature_encoder_default_selected"
        and item.severity.value == "info"
        for item in plan.diagnostics
    )


def test_custom_search_space_does_not_receive_an_inferred_encoder() -> None:
    class CustomSearchSpace(ObjectSpace):
        @property
        def dim(self) -> int:
            return 2

    representation = VectorSpace(dim=2, lb=[-1, -1], ub=[1, 1]).representation
    space = CustomSearchSpace(representation)
    optimizer = _optimizer(space)

    plan = optimizer._compile_plan()

    assert plan is not None
    assert not any(
        item.code == "feature_encoder_default_selected" for item in plan.diagnostics
    )
    assert any(
        item.code == "unresolved_service" and "FeatureEncoder" in item.message
        for item in plan.diagnostics
    )


def test_direct_strategy_does_not_select_an_unused_default(monkeypatch) -> None:
    import saealib.execution.bindings as bindings

    original = bindings._feature_encoder_default
    calls: list[object] = []

    def record_default(space: object):
        calls.append(space)
        return original(space)

    monkeypatch.setattr(bindings, "_feature_encoder_default", record_default)
    optimizer = (
        _optimizer(VectorSpace(dim=2, lb=[-1, -1], ub=[1, 1]))
        .set_strategy(DirectStrategy())
        .set_termination(Termination(max_gen(1)))
    )

    plan = optimizer._compile_plan()

    assert plan is not None
    assert calls == []
    assert optimizer._feature_encoder_default is None
    resolution = optimizer._component_resolution
    assert resolution is not None
    assert resolution.feature_encoder is None
    assert resolution.feature_encoder_origin is None
    assert not any(
        item.code == "feature_encoder_default_selected" for item in plan.diagnostics
    )
    assert not any(
        "FeatureEncoder" in node.resolved_services for node in plan.graph.nodes
    )
    state = optimizer.run()
    assert calls == []
    assert optimizer._feature_encoder_default is None
    assert "FeatureEncoder" not in state.__dict__.get("_compiled_services", {})


def test_local_encoder_state_does_not_invalidate_portable_checkpoints(
    tmp_path,
) -> None:
    class LocalEncoder:
        def __init__(self) -> None:
            self.calls = 0

        def encode(self, genomes):
            self.calls += 1
            return np.asarray(genomes.array)

    space = VectorSpace(dim=2, lb=[-1, -1], ub=[1, 1])
    encoder = LocalEncoder()
    space.services.register("FeatureEncoder", encoder)
    optimizer = _optimizer(space).set_termination(Termination(max_gen(5)))
    versions: list[int] = []

    for index, state in enumerate(optimizer.iterate()):
        environment = state.__dict__["_checkpoint_guard"]
        versions.append(environment.active_version)
        state.save(tmp_path / f"state-{index}.npz")

    assert encoder.calls > 0
    assert versions == [0, 0, 0, 0, 0, 0]
    assert len(list(tmp_path.glob("state-*.npz"))) == 6


def test_registry_encoder_state_does_not_change_its_configuration_signature(
    tmp_path,
) -> None:
    space = VectorSpace(dim=2, lb=[-1, -1], ub=[1, 1])
    encoder = CountingFeatureEncoder()
    space.services.register("FeatureEncoder", encoder)
    optimizer = _optimizer(space).set_termination(Termination(max_gen(5)))
    versions: list[int] = []

    for index, state in enumerate(optimizer.iterate()):
        environment = state.__dict__["_checkpoint_guard"]
        versions.append(environment.active_version)
        state.save(tmp_path / f"state-{index}.npz")

    assert encoder.calls > 0
    assert versions == [0, 0, 0, 0, 0, 0]
    assert len(list(tmp_path.glob("state-*.npz"))) == 6


def test_restored_encoder_has_unknown_origin_until_re_registered(tmp_path) -> None:
    space = VectorSpace(dim=2, lb=[-1, -1], ub=[1, 1])
    encoder = CountingFeatureEncoder()
    space.services.register("FeatureEncoder", encoder)
    del space.services._feature_encoder_registration_count
    optimizer = _optimizer(space).set_termination(Termination(max_gen(2)))
    first = next(optimizer.iterate())
    path = tmp_path / "optimizer.pkl"

    with pytest.warns(UserWarning, match="version-sensitive"):
        optimizer.save_pickle(first, path)
        restored, _ = Optimizer.load_pickle(path)

    saved_encoder = restored.problem.space.services.get("FeatureEncoder")
    assert saved_encoder is not None
    plan = restored._compile_plan()
    assert plan is not None
    resolution = restored._component_resolution
    assert resolution is not None
    origin = resolution.feature_encoder_origin
    assert origin is not None
    assert origin.source == "persisted"
    assert "original selection source is unavailable" in origin.reason
    assert origin.value is saved_encoder
    assert not any(
        item.code == "feature_encoder_default_selected" for item in plan.diagnostics
    )
    assert (
        next(
            node.resolved_services["FeatureEncoder"]
            for node in plan.graph.nodes
            if "FeatureEncoder" in node.resolved_services
        )
        is saved_encoder
    )

    restored.problem.space.services.register("FeatureEncoder", saved_encoder)
    plan = restored._compile_plan()
    assert plan is not None
    resolution = restored._component_resolution
    assert resolution is not None
    origin = resolution.feature_encoder_origin
    assert origin is not None
    assert origin.source == "explicit SearchSpace configuration"
    assert origin.value is saved_encoder
    assert not any(
        item.code == "feature_encoder_default_selected" for item in plan.diagnostics
    )


@pytest.mark.parametrize("missing_registration_count", [False, True])
def test_restored_encoder_signature_tracks_declared_configuration(
    tmp_path, missing_registration_count: bool
) -> None:
    space = VectorSpace(dim=2, lb=[-1, -1], ub=[1, 1])
    encoder = ScaleFeatureEncoder()
    space.services.register("FeatureEncoder", encoder)
    if missing_registration_count:
        del space.services._feature_encoder_registration_count
    optimizer = _optimizer(space).set_termination(Termination(max_gen(4)))
    first = next(optimizer.iterate())
    pickle_path = tmp_path / f"scale-{missing_registration_count}.pkl"

    with pytest.warns(UserWarning, match="version-sensitive"):
        optimizer.save_pickle(first, pickle_path)
        restored, saved_state = Optimizer.load_pickle(pickle_path)

    restored_encoder = restored.problem.space.services.get("FeatureEncoder")
    assert isinstance(restored_encoder, ScaleFeatureEncoder)
    iterator = restored.iterate_from(saved_state)
    boundary = next(iterator)
    environment = boundary.__dict__["_checkpoint_guard"]
    origin = environment._active_snapshot.provenance["feature_encoder"]
    assert origin.value is restored_encoder
    assert origin.signature == _feature_encoder_signature(restored_encoder)
    if missing_registration_count:
        assert origin.source == "persisted"
        assert "original selection source is unavailable" in origin.reason
    else:
        assert origin.source == "explicit SearchSpace configuration"

    boundary.save(tmp_path / f"unchanged-{missing_registration_count}.npz")
    restored_encoder.scale = 3.0
    with pytest.raises(CheckpointError, match="configuration changed"):
        boundary.save(tmp_path / f"stale-{missing_registration_count}.npz")

    next_state = next(iterator)
    updated = environment._active_snapshot.provenance["feature_encoder"]
    assert environment.active_version == 1
    assert updated.value is restored_encoder
    assert updated.signature == _feature_encoder_signature(restored_encoder)
    assert "3.0" in updated.signature
    assert environment.binding_values()["feature_encoder"] is restored_encoder
    assert next_state.compiled_service("FeatureEncoder") is restored_encoder
    next_state.save(tmp_path / f"published-{missing_registration_count}.npz")


def test_nonconsumer_preparation_clears_an_automatic_encoder_cache(monkeypatch) -> None:
    import saealib.execution.bindings as bindings

    original = bindings._feature_encoder_default
    calls: list[object] = []

    def record_default(space: object):
        calls.append(space)
        return original(space)

    monkeypatch.setattr(bindings, "_feature_encoder_default", record_default)
    optimizer = _optimizer(VectorSpace(dim=2, lb=[-1, -1], ub=[1, 1]))
    consumer_strategy = GenerationBasedStrategy(1)
    optimizer.set_strategy(consumer_strategy)

    consumer_plan = optimizer._compile_plan()

    assert consumer_plan is not None
    assert len(calls) == 1
    assert optimizer._feature_encoder_default is not None

    optimizer.set_strategy(DirectStrategy())
    direct_plan = optimizer._compile_plan()

    assert direct_plan is not None
    assert optimizer._feature_encoder_default is None
    assert optimizer._component_resolution.feature_encoder is None
    assert optimizer._component_resolution.feature_encoder_origin is None
    assert not any(
        item.code == "feature_encoder_default_selected"
        for item in direct_plan.diagnostics
    )

    optimizer.set_strategy(consumer_strategy)
    resumed_consumer_plan = optimizer._compile_plan()

    assert resumed_consumer_plan is not None
    assert len(calls) == 2


def test_unused_explicit_encoder_survives_until_a_consumer_is_prepared() -> None:
    class CustomEncoder:
        def encode(self, genomes):
            return np.asarray(genomes.array)

    space = VectorSpace(dim=2, lb=[-1, -1], ub=[1, 1])
    encoder = CustomEncoder()
    space.services.register("FeatureEncoder", encoder)
    optimizer = _optimizer(space)
    strategy = GenerationBasedStrategy(1)
    optimizer.set_strategy(strategy)
    optimizer.set_strategy(DirectStrategy())

    direct_plan = optimizer._compile_plan()

    assert direct_plan is not None
    assert space.services.get("FeatureEncoder") is encoder
    resolution = optimizer._component_resolution
    assert resolution is not None
    assert resolution.feature_encoder is encoder
    assert resolution.feature_encoder_origin is not None
    assert (
        resolution.feature_encoder_origin.source == "explicit SearchSpace configuration"
    )
    assert optimizer._feature_encoder_default is None
    assert not any(
        item.code == "feature_encoder_default_selected"
        for item in direct_plan.diagnostics
    )

    optimizer.set_strategy(strategy)
    consumer_plan = optimizer._compile_plan()

    assert consumer_plan is not None
    assert _feature_encoder(optimizer) is encoder
    assert not any(
        item.code == "feature_encoder_default_selected"
        for item in consumer_plan.diagnostics
    )


def test_same_encoder_explicit_registration_updates_prepared_provenance() -> None:
    space = VectorSpace(dim=2, lb=[-1, -1], ub=[1, 1])
    optimizer = _optimizer(space).set_termination(Termination(max_gen(5)))
    states = optimizer.iterate()
    first = next(states)
    encoder = first.compiled_service("FeatureEncoder")
    environment = first.__dict__["_checkpoint_guard"]
    first_version = environment.active_version

    space.services.register("FeatureEncoder", encoder)
    second = next(states)

    assert environment.active_version == first_version + 1
    assert second.compiled_service("FeatureEncoder") is encoder
    assert (
        environment._active_snapshot.provenance["feature_encoder"].source
        == "explicit SearchSpace configuration"
    )
    active_plan = optimizer.executable_plan
    assert active_plan is not None
    assert not any(
        item.code == "feature_encoder_default_selected"
        for item in active_plan.diagnostics
    )
    optimizer.contract_diagnostics()
    assert not any(
        item.code == "feature_encoder_default_selected"
        for item in optimizer.last_contract_diagnostics
    )

    space.services.register("FeatureEncoder", encoder)
    third = next(states)

    assert environment.active_version == first_version + 1
    assert third.compiled_service("FeatureEncoder") is encoder

    class CustomEncoder:
        def encode(self, genomes):
            return np.asarray(genomes.array)

    replacement = CustomEncoder()
    space.services.register("FeatureEncoder", replacement)
    fourth = next(states)

    assert environment.active_version == first_version + 2
    assert fourth.compiled_service("FeatureEncoder") is replacement
    assert environment.binding_values()["feature_encoder"] is replacement
    active_plan = optimizer.executable_plan
    assert active_plan is not None
    assert not any(
        item.code == "feature_encoder_default_selected"
        for item in active_plan.diagnostics
    )


def test_explicit_custom_encoder_keeps_identity_and_behavior() -> None:
    class CustomEncoder:
        def encode(self, genomes):
            return np.full((len(genomes), 1), 7.0)

    space = VectorSpace(dim=2, lb=[-1, -1], ub=[1, 1])
    encoder = CustomEncoder()
    space.services.register("FeatureEncoder", encoder)
    optimizer = _optimizer(space)

    assert _feature_encoder(optimizer) is encoder
    plan = optimizer.executable_plan
    assert plan is not None
    assert not any(
        item.code == "feature_encoder_default_selected" for item in plan.diagnostics
    )
    with warnings.catch_warnings(record=True) as emitted:
        warnings.simplefilter("always")
        encoded = encode_features(space, DenseVectorBatch([[0.25, 0.5]]))
    assert emitted == []
    np.testing.assert_array_equal(encoded, [[7.0]])


def test_standalone_does_not_replace_an_incompatible_explicit_encoder() -> None:
    space = VectorSpace(dim=2, lb=[-1, -1], ub=[1, 1])
    encoder = object()
    space.services.register("FeatureEncoder", encoder)
    genomes = DenseVectorBatch([[0.25, 0.5]])

    with warnings.catch_warnings(record=True) as emitted:
        warnings.simplefilter("always")
        with pytest.raises(ValidationError, match="require the FeatureEncoder service"):
            encode_features(space, genomes)

    assert emitted == []
    assert space.services.get("FeatureEncoder") is encoder


def test_incompatible_explicit_encoder_is_reported_without_replacement() -> None:
    space = VectorSpace(dim=2, lb=[-1, -1], ub=[1, 1])
    encoder = object()
    space.services.register("FeatureEncoder", encoder)
    optimizer = _optimizer(space)

    optimizer.validate()

    findings = optimizer.last_contract_diagnostics
    assert any(
        item.code == "component_protocol_mismatch"
        and item.path.components == ("optimizer", "feature_encoder")
        for item in findings
    )
    assert not any(item.code == "feature_encoder_default_selected" for item in findings)
    assert _feature_encoder(optimizer) is encoder


def test_default_encoders_preserve_transformation_results() -> None:
    vector_space = VectorSpace(dim=2, lb=[-1, -1], ub=[1, 1])
    vector_encoder = _feature_encoder(_optimizer(vector_space))
    vectors = DenseVectorBatch([[0.25, 0.5], [-0.75, 1.0]])
    encoded_vectors = _encode_features(vector_space, vectors, vector_encoder)
    np.testing.assert_array_equal(encoded_vectors, vectors.array)
    assert np.shares_memory(encoded_vectors, vectors.array)

    permutation_space = PermutationSpace(length=3)
    permutation_encoder = _feature_encoder(_optimizer(permutation_space))
    permutations = PermutationBatch([[0, 2, 1], [2, 1, 0]], length=3)
    encoded_permutations = _encode_features(
        permutation_space, permutations, permutation_encoder
    )
    np.testing.assert_array_equal(encoded_permutations, permutations.array)
    assert not np.shares_memory(encoded_permutations, permutations.array)

    sequence_space = SequenceSpace(["a", "b"], max_length=3)
    sequence_encoder = _feature_encoder(_optimizer(sequence_space))
    sequences = VariableLengthBatch([["a", "b"], ["b"]])
    encoded_sequences = _encode_features(sequence_space, sequences, sequence_encoder)
    np.testing.assert_array_equal(encoded_sequences, [[2, 1, 2, 0], [1, 2, 0, 0]])


@pytest.mark.parametrize(
    "space_kind", ["vector", "extended_vector", "permutation", "sequence"]
)
def test_standalone_feature_encoder_defaults_warn_and_preserve_results(
    space_kind: str,
) -> None:
    source_array: np.ndarray | None = None
    if space_kind in {"vector", "extended_vector"}:
        space = (
            ExtendedVectorSpace(dim=2, lb=[-1, -1], ub=[1, 1])
            if space_kind == "extended_vector"
            else VectorSpace(dim=2, lb=[-1, -1], ub=[1, 1])
        )
        genomes = DenseVectorBatch([[0.25, 0.5], [-0.75, 1.0]])
        expected = genomes.array
        source_array = genomes.array
        shares_memory = True
    elif space_kind == "permutation":
        space = PermutationSpace(length=3)
        genomes = PermutationBatch([[0, 2, 1], [2, 1, 0]], length=3)
        expected = np.array([[0, 2, 1], [2, 1, 0]], dtype=np.float64)
        source_array = genomes.array
        shares_memory = False
    else:
        space = SequenceSpace(["a", "b"], max_length=3)
        genomes = VariableLengthBatch([["a", "b"], ["b"]])
        expected = np.array([[2, 1, 2, 0], [1, 2, 0, 0]], dtype=np.float64)
        shares_memory = None

    assert space.services.get("FeatureEncoder") is None
    with pytest.warns(
        UserWarning, match="encode_features used the default FeatureEncoder"
    ) as emitted:
        encoded = encode_features(space, genomes)

    assert emitted[0].filename == __file__
    assert space.services.get("FeatureEncoder") is None
    np.testing.assert_array_equal(encoded, expected)
    if source_array is not None:
        assert shares_memory is not None
        assert np.shares_memory(encoded, source_array) is shares_memory


def test_optimizer_default_does_not_emit_standalone_warning() -> None:
    space = VectorSpace(dim=2, lb=[-1, -1], ub=[1, 1])
    optimizer = _optimizer(space).set_termination(Termination(max_gen(2)))

    with warnings.catch_warnings(record=True) as emitted:
        warnings.simplefilter("always")
        list(optimizer.iterate())

    assert not any(
        "encode_features used the default FeatureEncoder" in str(item.message)
        for item in emitted
    )
    assert space.services.get("FeatureEncoder") is None


def test_vector_default_rebinds_after_portable_and_pickle_resume(tmp_path) -> None:
    problem_space = VectorSpace(dim=2, lb=[-1, -1], ub=[1, 1])
    optimizer = _optimizer(problem_space).set_termination(Termination(max_gen(3)))
    first = next(optimizer.iterate())
    assert first.compiled_service("FeatureEncoder") is not None

    portable_path = tmp_path / "portable.npz"
    first.save(portable_path)
    restored = OptimizationState.load(portable_path, optimizer.problem)
    resumed = optimizer.run_from(restored)
    assert resumed.gen >= first.gen
    assert resumed.compiled_service("FeatureEncoder") is not None
    assert any(
        item.code == "feature_encoder_default_selected"
        and item.severity.value == "info"
        for item in optimizer.last_contract_diagnostics
    )

    pickle_path = tmp_path / "optimizer.pkl"
    optimizer.save_pickle(resumed, pickle_path)
    restored_optimizer, restored_pickle = Optimizer.load_pickle(pickle_path)
    resumed_pickle = restored_optimizer.run_from(restored_pickle)
    assert resumed_pickle.compiled_service("FeatureEncoder") is not None
