"""Cross-component acceptance guards for the extension boundary."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, cast

import numpy as np

from saealib.core.compiler import CompileContext, Compiler
from saealib.core.compiler.diagnostics import Severity
from saealib.core.state import OPTIMIZATION_STATE_INITIAL_KEYS
from saealib.optimizer import Optimizer
from saealib.space import PermutationSpace, SequenceSpace
from saealib.strategies.direct import DirectStrategy
from saealib.strategies.ib import IndividualBasedStrategy

_REPRESENTATION_PATH = Path(__file__).parents[1] / "test_representation_integration.py"
_REPRESENTATION_SPEC = importlib.util.spec_from_file_location(
    "representation_integration", _REPRESENTATION_PATH
)
assert _REPRESENTATION_SPEC is not None and _REPRESENTATION_SPEC.loader is not None
representations = importlib.util.module_from_spec(_REPRESENTATION_SPEC)
_REPRESENTATION_SPEC.loader.exec_module(representations)
external_plugin = importlib.import_module("test_external_plugin_integration")


# Keep these names explicit: this inventory guard covers the focused extension
# tests that exercise the external boundary.
EXTENSION_FOCUSED_TESTS = (
    "test_external_plugin_proposes_evaluates_and_delivers_feedback",
    "test_external_plugin_component_compiles_without_error_diagnostics",
    "test_differential_evolution_preserves_targets_and_replaces_improvements",
    "test_multifidelity_accepts_sparse_low_and_selects_high_promotion",
    "test_cmaes_updates_only_on_complete_generation_feedback",
    "test_moead_updates_only_the_child_subproblem_neighborhood",
    "test_map_elites_uses_behavior_cells_and_quality_replacement",
    "test_coevolution_joins_named_blocks_and_updates_one_block",
)


def _representation_profiles() -> tuple[tuple[str, Any, Any, Any], ...]:
    """Reuse representation profiles and external-space helpers."""
    permutation = PermutationSpace(3)
    sequence = SequenceSpace((0, 1), 1, 3)
    representations._register_custom_kind()
    custom = representations._CustomSpace()
    genome_ga = representations.GenomeGA(
        representations.OrderCrossover(),
        representations.SwapMutation(),
        representations.SequentialSelection(),
        representations.TruncationSelection(),
    )
    sequence_ga = representations.GenomeGA(
        representations._ExternalCrossover(),
        representations.SequenceMutation(alphabet=(0, 1), min_length=1, max_length=3),
        representations.SequentialSelection(),
        representations.TruncationSelection(),
    )
    return (
        (
            "vector-direct",
            representations._vector_problem(),
            representations._vector_ga(),
            DirectStrategy(),
        ),
        (
            "vector-individual-based",
            representations._vector_problem(),
            representations._vector_ga(),
            IndividualBasedStrategy(0.5),
        ),
        (
            "permutation",
            representations._genome_problem(
                permutation, lambda x: np.asarray([sum(x)])
            ),
            genome_ga,
            DirectStrategy(),
        ),
        (
            "variable-length-sequence",
            representations._genome_problem(sequence, lambda x: np.asarray([len(x)])),
            sequence_ga,
            DirectStrategy(),
        ),
        (
            "external-custom-object-graph",
            representations._genome_problem(custom, lambda x: np.asarray([float(x)])),
            representations._custom_ga(),
            DirectStrategy(),
        ),
    )


def test_extension_inventory_and_profiles_are_complete() -> None:
    assert len(EXTENSION_FOCUSED_TESTS) == 8
    assert all(
        callable(getattr(external_plugin, name, None))
        for name in EXTENSION_FOCUSED_TESTS
    )
    assert len(_representation_profiles()) == 5


def test_representation_graph_profiles_compile_through_current_public_path() -> None:
    profile_names = tuple(profile[0] for profile in _representation_profiles())
    assert profile_names == (
        "vector-direct",
        "vector-individual-based",
        "permutation",
        "variable-length-sequence",
        "external-custom-object-graph",
    )

    for name, problem, algorithm, strategy in _representation_profiles():
        optimizer = (
            Optimizer(problem, seed=1).set_algorithm(algorithm).set_strategy(strategy)
        )
        optimizer.resolve_defaults()
        graph = cast(Any, optimizer.strategy).build_graph(optimizer)
        plan = Compiler().compile(
            graph,
            CompileContext(
                space=optimizer.problem.space,
                problem=optimizer.problem,
                initial_state_keys=OPTIMIZATION_STATE_INITIAL_KEYS,
            ),
        )
        assert [
            diagnostic
            for diagnostic in plan.diagnostics
            if diagnostic.severity is Severity.ERROR
        ] == [], name


def test_plugin_uses_core_compiler_without_representation_branch() -> None:
    compiler_root = Path(__file__).parents[2] / "src" / "saealib" / "core" / "compiler"
    compiler_source = "\n".join(
        path.read_text(encoding="utf-8") for path in compiler_root.glob("*.py")
    )
    assert "external_plugin" not in compiler_source
    assert "ExternalPlugin" not in compiler_source
