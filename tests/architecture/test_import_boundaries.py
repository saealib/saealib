"""AST-enforced dependency boundaries for the low-level core package."""

from __future__ import annotations

import ast
from pathlib import Path

KNOWN_CORE_REVERSE_IMPORTS = frozenset(
    {
        ("src/saealib/core/adapters/accumulator.py", "saealib.policies.feedback"),
        ("src/saealib/core/runtime.py", "saealib.callback.events"),
        ("src/saealib/core/runtime.py", "saealib.context"),
        ("src/saealib/core/state/context.py", "saealib.context"),
    }
)

FORBIDDEN_FACADE_PACKAGES = {
    "context",
    "callback",
    "policies",
}

FORBIDDEN_FEATURE_PACKAGES = {
    "algorithms",
    "operators",
    "surrogate",
    "acquisition",
    "strategies",
    "problem",
    "population",
    "pipeline",
}

# Keep exceptions exact and intentional. These are implementation leaves that
# still need population runtime types; shared identity types live in core.
INTENTIONAL_CORE_BRIDGES: dict[tuple[str, str], str] = {
    (
        "src/saealib/core/state/patch.py",
        "saealib.population.genome",
    ): "StatePatch stores the Population genome batch representation.",
    (
        "src/saealib/core/state/store.py",
        "saealib.population",
    ): "StateStore owns the runtime Population state value.",
    (
        "src/saealib/core/graph_builder.py",
        "saealib.pipeline",
    ): "The graph builder retains the Stage/Pipeline compatibility bridge.",
}


class _Imports(ast.NodeVisitor):
    """Collect imports resolved to absolute saealib module names.

    Covers the root-package alias form (``from saealib import context``) and
    relative forms (``from ..context import OptimizationState``) in addition
    to fully qualified modules.
    """

    def __init__(self, package: tuple[str, ...]) -> None:
        self.package = package
        self.imports: list[tuple[ast.Import | ast.ImportFrom, str]] = []

    def visit_Import(self, node: ast.Import) -> None:
        self.imports.extend((node, alias.name) for alias in node.names)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.level == 0:
            if node.module == "saealib":
                for alias in node.names:
                    self.imports.append((node, f"saealib.{alias.name}"))
            elif node.module is not None:
                self.imports.append((node, node.module))
            return
        depth = node.level - 1
        if depth >= len(self.package):
            return
        base = self.package[: len(self.package) - depth]
        if node.module is not None:
            self.imports.append((node, ".".join([*base, *node.module.split(".")])))
        else:
            for alias in node.names:
                self.imports.append((node, ".".join([*base, alias.name])))


def _core_files() -> list[Path]:
    root = Path(__file__).parents[2]
    return sorted((root / "src/saealib/core").glob("**/*.py"))


def _core_package(relative: str) -> tuple[str, ...]:
    module_parts = Path(relative).with_suffix("").parts[2:]
    return ("saealib", *module_parts[:-1])


def _core_imports() -> list[tuple[str, ast.Import | ast.ImportFrom, str]]:
    root = Path(__file__).parents[2]
    collected = []
    for path in _core_files():
        relative = path.relative_to(root).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        collector = _Imports(_core_package(relative))
        collector.visit(tree)
        collected.extend((relative, node, module) for node, module in collector.imports)
    return collected


def test_core_has_no_forbidden_module_level_imports() -> None:
    violations: list[str] = []
    seen_allowlist: set[tuple[str, str]] = set()

    for relative, node, module in _core_imports():
        parts = module.split(".")
        if len(parts) < 2 or parts[0] != "saealib":
            continue
        package = parts[1]
        if package not in FORBIDDEN_FEATURE_PACKAGES:
            continue
        key = (relative, module)
        if key in INTENTIONAL_CORE_BRIDGES:
            seen_allowlist.add(key)
        else:
            violations.append(f"{relative}:{node.lineno}: from {module}")

    assert not violations, "Forbidden core imports:\n" + "\n".join(violations)
    assert seen_allowlist == set(INTENTIONAL_CORE_BRIDGES), (
        "Allowlist contains stale or unobserved entries: "
        + repr(set(INTENTIONAL_CORE_BRIDGES) - seen_allowlist)
    )


def test_core_has_no_reverse_facade_imports() -> None:
    violations: list[str] = []
    seen_allowlist: set[tuple[str, str]] = set()

    for relative, node, module in _core_imports():
        parts = module.split(".")
        if len(parts) < 2 or parts[0] != "saealib":
            continue
        package = parts[1]
        if package not in FORBIDDEN_FACADE_PACKAGES:
            continue
        key = (relative, module)
        if key in KNOWN_CORE_REVERSE_IMPORTS:
            seen_allowlist.add(key)
        else:
            violations.append(f"{relative}:{node.lineno}: from {module}")

    assert not violations, "Forbidden reverse core imports:\n" + "\n".join(violations)
    assert seen_allowlist == KNOWN_CORE_REVERSE_IMPORTS, (
        "Allowlist contains stale or unobserved entries: "
        + repr(KNOWN_CORE_REVERSE_IMPORTS - seen_allowlist)
    )


def test_facade_import_resolution_covers_root_and_relative_forms() -> None:
    source = "\n".join(
        [
            "from saealib import context",
            "from ..context import OptimizationState",
            "from ..callback import events",
            "from ..policies.feedback import SelectionPolicy",
            "from .. import policies",
            "from . import callback",
            "from saealib.core.state import StateKey",
        ]
    )
    collector = _Imports(("saealib", "core"))
    collector.visit(ast.parse(source))
    modules = {module for _, module in collector.imports}

    assert {
        "saealib.context",
        "saealib.callback",
        "saealib.policies",
        "saealib.policies.feedback",
    } <= modules
    assert "saealib.core.callback" in modules
    assert "saealib.core.state" in modules


def test_candidate_ids_use_one_identity_across_compatibility_exports() -> None:
    from saealib.core.contracts import CandidateIds as CoreCandidateIds
    from saealib.core.contracts.identity import CandidateIds as LeafCandidateIds
    from saealib.identity import CandidateIds as IdentityCandidateIds
    from saealib.population import CandidateIds as PopulationPackageCandidateIds
    from saealib.population.population import CandidateIds as PopulationCandidateIds

    assert CoreCandidateIds is LeafCandidateIds
    assert CoreCandidateIds is IdentityCandidateIds
    assert PopulationCandidateIds is IdentityCandidateIds
    assert PopulationPackageCandidateIds is IdentityCandidateIds
