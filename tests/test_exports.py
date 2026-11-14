"""Drift-guard tests for saealib's top-level export surfaces.

See the root export-surface comment at the top of src/saealib/__init__.py for
the eager/lazy policy enforced here. Namespace ``__all__`` values are checked
on their own package and are not required to appear at the root.
"""

import importlib
import importlib.util
import logging
from pathlib import Path
from typing import Any, cast

import pytest

import saealib

# Subpackages whose namespace __all__ values are checked independently of the
# root surfaces. saealib.registry is deliberately excluded: it defines no
# __all__, and its get/build/to_spec namespace-only status is covered by
# tests/test_registry.py::TestTopLevelExport instead.
SCANNED_SUBPACKAGES = [
    "saealib.acquisition",
    "saealib.algorithms",
    "saealib.benchmarks",
    "saealib.callback",
    "saealib.comparators",
    "saealib.controllers",
    "saealib.defaults",
    "saealib.exceptions",
    "saealib.execution",
    "saealib.operators",
    "saealib.population",
    "saealib.problem",
    "saealib.space",
    "saealib.strategies",
    "saealib.surrogate",
    "saealib.utils",
    "saealib.variables",
]

# These are intentionally public only through their subpackage. Keeping a
# small explicit set makes the independence contract visible and prevents
# this test from passing only because one incidental symbol happens to differ.
NAMESPACE_ONLY_EXPORTS = {
    "saealib.benchmarks": {"sphere"},
    "saealib.defaults": {"load_defaults"},
    "saealib.space": {"SearchSpace"},
}

REMOVED_EXPERIMENTAL_EXPORTS = {
    "ArchiveSnapshot",
    "CooperativeCoevolution",
    "CorrelatedQuadraticSurrogate",
    "DynamicArchiveSelector",
    "EvaluationWorkflowResult",
    "FidelityEvaluator",
    "FidelityPromotionRunner",
    "FidelityWorkflowResult",
    "MigrationPolicy",
    "RepeatedEvaluationRunner",
    "SeededNoiseEvaluator",
    "reference_problem",
}

REMOVED_CANONICAL_ALIASES = {
    "EvaluationPolicy",
    "FeedbackPolicy",
    "DensityAcquisition",
    "NichingAcquisition",
    "RouletteWheelSelection",
    "MigrationPolicy",
    "AsyncScheduler",
}


def test_root_logger_is_private_and_null_handler_is_attached():
    root_logger = logging.getLogger("saealib")

    assert not hasattr(saealib, "logger")
    assert "logger" not in saealib.__all__
    assert any(
        isinstance(handler, logging.NullHandler) for handler in root_logger.handlers
    )


def test_eager_and_lazy_exports_do_not_overlap():
    assert set(saealib.__all__).isdisjoint(saealib._LAZY_EXPORTS)


def test_eager_and_lazy_exports_resolve():
    for name in list(saealib.__all__) + list(saealib._LAZY_EXPORTS):
        assert hasattr(saealib, name), f"{name} listed but not resolvable"


def test_lazy_exports_resolve_from_their_canonical_modules():
    for name, module_name in saealib._LAZY_EXPORTS.items():
        module = importlib.import_module(module_name)
        assert getattr(saealib, name) is getattr(module, name)


def test_experimental_generality_code_is_not_a_library_export():
    assert REMOVED_EXPERIMENTAL_EXPORTS.isdisjoint(vars(saealib))
    assert importlib.util.find_spec("saealib.generality") is None
    assert not (Path(__file__).parents[1] / "src/saealib/generality.py").exists()


def test_removed_aliases_are_not_importable():
    for name in REMOVED_CANONICAL_ALIASES:
        assert not hasattr(saealib, name)
    for module_name in (
        "saealib.acquisition",
        "saealib.execution",
        "saealib.operators",
        "saealib.policies",
    ):
        module = importlib.import_module(module_name)
        assert all(not hasattr(module, name) for name in REMOVED_CANONICAL_ALIASES)


@pytest.mark.parametrize(
    ("module_name", "name"),
    [("saealib", name) for name in sorted(REMOVED_CANONICAL_ALIASES)],
)
def test_removed_root_names_fail_import(module_name, name):
    with pytest.raises(ImportError):
        exec(f"from {module_name} import {name}", {})


def test_removed_parameter_names_fail_construction():
    from saealib.execution import AsyncEvaluationScheduler, SerialEvaluator
    from saealib.stages import FeedbackStage

    with pytest.raises(TypeError):
        cast(Any, AsyncEvaluationScheduler)(SerialEvaluator(), feedback_policy=None)
    with pytest.raises(TypeError):
        cast(Any, FeedbackStage)(policy=None)


def test_namespace_exports_resolve_independently_from_root_surface():
    root_surface = set(saealib.__all__) | set(saealib._LAZY_EXPORTS)
    namespace_exports: list[tuple[str, str]] = []
    for modname in SCANNED_SUBPACKAGES:
        mod = importlib.import_module(modname)
        for name in getattr(mod, "__all__", []):
            assert name in vars(mod) or hasattr(mod, name), (
                f"{modname}.{name} is listed but unresolved"
            )
            namespace_exports.append((modname, name))

    independent = [
        (modname, name)
        for modname, name in namespace_exports
        if name not in root_surface
    ]
    assert independent, "namespace public APIs must not be coupled to root exports"
    for _, name in independent:
        assert name not in saealib.__all__
        assert name not in saealib._LAZY_EXPORTS

    for modname, names in NAMESPACE_ONLY_EXPORTS.items():
        mod = importlib.import_module(modname)
        for name in names:
            assert name in mod.__all__
            assert name not in root_surface
            assert getattr(mod, name) is not getattr(saealib, name, None)


def test_root_overlap_uses_the_namespace_canonical_object():
    root_surface = set(saealib.__all__) | set(saealib._LAZY_EXPORTS)
    for modname in SCANNED_SUBPACKAGES:
        mod = importlib.import_module(modname)
        for name in getattr(mod, "__all__", []):
            if name in root_surface:
                assert getattr(saealib, name) is getattr(mod, name)
