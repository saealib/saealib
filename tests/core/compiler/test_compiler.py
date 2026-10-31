from __future__ import annotations

import ast
import random
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

import pytest

from saealib.core.compiler import (
    BranchRegion,
    CompileContext,
    Compiler,
    ResolutionResult,
    RuleContext,
    RuleRegistry,
    VerificationResult,
    lower_structured,
)
from saealib.core.compiler.compiler import _merge_graphs
from saealib.core.compiler.diagnostics import ContractPath, Diagnostic, Severity
from saealib.core.compiler.graph import (
    ComponentGraph,
    ComponentNode,
    ControlEdge,
    DataEdge,
    NodeRef,
    StateBinding,
)
from saealib.core.compiler.structured import StructuredGraph
from saealib.core.contracts import ComponentContract, ExecutionContract, StateContract
from saealib.core.state.keys import StateKey
from saealib.exceptions import ConfigurationError


class _Component:
    def contract(self) -> ComponentContract:
        return ComponentContract()


def _graph() -> ComponentGraph:
    return ComponentGraph(
        nodes=(ComponentNode(component_id="start", component=_Component()),),
        entry_points=(NodeRef(component_id="start"),),
    )


def _reference_merge_graphs(
    base: ComponentGraph, proposals: tuple[ResolutionResult, ...]
) -> ComponentGraph:
    """Reference implementation of the pre-indexed merge behavior."""

    def merge_values(original, current, candidate):
        merged = list(current)

        def same_slot(left, right):
            if type(left) is type(right) and hasattr(left, "component_id"):
                return getattr(left, "component_id") == getattr(right, "component_id")
            return left == right

        matched = {}
        used_candidates = set()
        for original_index, original_value in enumerate(original):
            for candidate_index, candidate_value in enumerate(candidate):
                if candidate_index in used_candidates:
                    continue
                if same_slot(original_value, candidate_value):
                    matched[original_index] = candidate_index
                    used_candidates.add(candidate_index)
                    break

        for original_index in reversed(range(len(original))):
            if original_index in matched:
                continue
            original_value = original[original_index]
            for current_index, current_value in enumerate(merged):
                if same_slot(original_value, current_value):
                    del merged[current_index]
                    break

        for original_index, candidate_index in matched.items():
            original_value = original[original_index]
            candidate_value = candidate[candidate_index]
            if candidate_value == original_value:
                continue
            for current_index, current_value in enumerate(merged):
                if same_slot(original_value, current_value):
                    merged[current_index] = candidate_value
                    break

        def stable_value_key(value):
            return (
                type(value).__name__,
                str(getattr(value, "component_id", "")),
                repr(value),
            )

        for candidate_index, candidate_value in enumerate(candidate):
            if candidate_index in used_candidates:
                continue
            if not any(same_slot(candidate_value, value) for value in merged):
                merged.append(candidate_value)
        existing = [
            value
            for value in merged
            if any(same_slot(value, original_value) for original_value in original)
        ]
        added = [
            value
            for value in merged
            if not any(same_slot(value, original_value) for original_value in original)
        ]
        return tuple((*existing, *sorted(added, key=stable_value_key)))

    graph = base
    for proposal in proposals:
        candidate = proposal.graph
        graph = replace(
            graph,
            nodes=merge_values(base.nodes, graph.nodes, candidate.nodes),
            data_edges=merge_values(
                base.data_edges, graph.data_edges, candidate.data_edges
            ),
            control_edges=merge_values(
                base.control_edges, graph.control_edges, candidate.control_edges
            ),
            state_bindings=merge_values(
                base.state_bindings, graph.state_bindings, candidate.state_bindings
            ),
            entry_points=merge_values(
                base.entry_points, graph.entry_points, candidate.entry_points
            ),
        )
    return graph


def test_merge_graphs_matches_reference_with_duplicates_and_adapters() -> None:
    component = _Component()
    start = ComponentNode(component_id="start", component=component)
    duplicate = ComponentNode(component_id="duplicate", component=component)
    adapter = ComponentNode(component_id="adapter", component=component)
    base = ComponentGraph(
        nodes=(start, duplicate, duplicate),
        data_edges=(
            DataEdge(
                source=NodeRef(component_id="start"),
                target=NodeRef(component_id="duplicate"),
                source_port="x",
                target_port="y",
            ),
        ),
        control_edges=(
            ControlEdge(
                source=NodeRef(component_id="start"),
                target=NodeRef(component_id="duplicate"),
            ),
        ),
        state_bindings=(
            StateBinding(
                node=NodeRef(component_id="start"),
                state_key=StateKey(namespace="user", name="value", schema_version=1),
            ),
        ),
        entry_points=(NodeRef(component_id="start"), NodeRef(component_id="start")),
    )
    proposals = (
        ResolutionResult(
            graph=replace(
                base,
                nodes=(
                    replace(start, role="producer"),
                    duplicate,
                    adapter,
                    duplicate,
                ),
                entry_points=(NodeRef(component_id="start", role="producer"),),
            )
        ),
        ResolutionResult(
            graph=replace(
                base,
                nodes=(replace(start, role="consumer"), duplicate, adapter),
                data_edges=(),
            )
        ),
    )

    assert _merge_graphs(base, proposals) == _reference_merge_graphs(base, proposals)


def test_conflicting_claims_apply_neither_rewrite() -> None:
    graph = _graph()
    registry = RuleRegistry(
        [
            _ClaimingRule("a", role="left"),
            _ClaimingRule("b", role="right"),
        ]
    )
    plan = Compiler(registry).compile(
        graph, CompileContext(enabled_rule_namespaces=frozenset({"test"}))
    )

    assert any(
        diagnostic.code == "conflicting_rewrite" for diagnostic in plan.diagnostics
    )
    assert plan.graph is not graph
    assert graph.nodes[0].role is None


@dataclass
class _ClaimingRule:
    name: str
    role: str = "left"
    namespace: str = "test"
    phase: Literal["resolution"] = "resolution"

    def apply(self, context: RuleContext) -> ResolutionResult:
        node = replace(context.graph.nodes[0], role=self.role)
        return ResolutionResult(
            graph=replace(context.graph, nodes=(node,)),
            claims=frozenset({context.claim("node", "start")}),
        )


@dataclass
class _UnclaimedRule:
    name: str = "unclaimed"
    namespace: str = "test"
    phase: Literal["resolution"] = "resolution"

    def apply(self, context: RuleContext) -> ResolutionResult:
        node = replace(context.graph.nodes[0], role="unclaimed")
        return ResolutionResult(graph=replace(context.graph, nodes=(node,)))


def test_unclaimed_graph_changes_are_rejected() -> None:
    graph = _graph()
    plan = Compiler(RuleRegistry([_UnclaimedRule()])).compile(
        graph, CompileContext(enabled_rule_namespaces=frozenset({"test"}))
    )

    assert [
        diagnostic.code
        for diagnostic in plan.diagnostics
        if diagnostic.code == "unclaimed_rewrite"
    ] == ["unclaimed_rewrite"]
    assert plan.graph is not graph


@dataclass
class _VerificationRule:
    name: str
    namespace: str = "test"
    phase: Literal["verification"] = "verification"

    def apply(self, context: RuleContext) -> VerificationResult:
        return VerificationResult()


def test_compiler_result_is_execution_free_and_describable() -> None:
    registry = RuleRegistry([_VerificationRule("check")])
    plan = Compiler(registry).compile(
        _graph(), CompileContext(enabled_rule_namespaces=frozenset({"test"}))
    )

    assert "ExecutablePlan" in plan.describe()
    assert not hasattr(plan, "execute")


@dataclass
class _FindingRule:
    name: str
    namespace: str = "test"
    phase: Literal["verification"] = "verification"

    def apply(self, context: RuleContext) -> VerificationResult:
        return VerificationResult(
            diagnostics=(
                Diagnostic(
                    severity=Severity.WARNING,
                    code="unknown_data_spec",
                    message=f"finding from {self.name}",
                    path=ContractPath(components=("start",)),
                    resolutions=("Inspect the test rule.",),
                ),
            )
        )


def _diagnostic_signature(plan) -> tuple[tuple[str, str, str], ...]:
    return tuple(
        (diagnostic.code, str(diagnostic.path), diagnostic.message)
        for diagnostic in plan.diagnostics
    )


def test_rule_enumeration_order_is_deterministic() -> None:
    rules = [_FindingRule("a"), _FindingRule("b"), _FindingRule("c")]
    expected = None
    for seed in range(8):
        shuffled = list(rules)
        random.Random(seed).shuffle(shuffled)
        plan = Compiler(RuleRegistry(shuffled)).compile(
            _graph(), CompileContext(enabled_rule_namespaces=frozenset({"test"}))
        )
        signature = (_diagnostic_signature(plan), plan.active_rule_names)
        if expected is None:
            expected = signature
        else:
            assert signature == expected


@dataclass
class _DisjointRule:
    name: str
    target: str
    role: str
    namespace: str = "test"
    phase: Literal["resolution"] = "resolution"

    def apply(self, context: RuleContext) -> ResolutionResult:
        nodes = tuple(
            replace(node, role=self.role) if node.component_id == self.target else node
            for node in context.graph.nodes
        )
        return ResolutionResult(
            graph=replace(context.graph, nodes=nodes),
            claims=frozenset({context.claim("node", self.target)}),
        )


def test_disjoint_resolution_claims_are_order_independent() -> None:
    graph = replace(
        _graph(),
        nodes=(
            ComponentNode(component_id="start", component=_Component()),
            ComponentNode(component_id="other", component=_Component()),
        ),
    )
    rules = [
        _DisjointRule("a", "start", "first"),
        _DisjointRule("b", "other", "second"),
    ]
    expected = None
    for seed in range(8):
        shuffled = list(rules)
        random.Random(seed).shuffle(shuffled)
        plan = Compiler(RuleRegistry(shuffled)).compile(
            graph, CompileContext(enabled_rule_namespaces=frozenset({"test"}))
        )
        signature = tuple((node.component_id, node.role) for node in plan.graph.nodes)
        if expected is None:
            expected = signature
        else:
            assert signature == expected
    assert expected == (("start", "first"), ("other", "second"))


@dataclass
class _AddingRule:
    name: str
    target: str
    namespace: str = "test"
    phase: Literal["resolution"] = "resolution"

    def apply(self, context: RuleContext) -> ResolutionResult:
        if any(node.component_id == self.target for node in context.graph.nodes):
            return ResolutionResult(
                graph=context.graph,
                claims=frozenset(
                    {
                        context.claim("node", self.target),
                        context.claim("entry_point", self.target),
                    }
                ),
            )
        node = ComponentNode(component_id=self.target, component=_Component())
        return ResolutionResult(
            graph=replace(
                context.graph,
                nodes=(*context.graph.nodes, node),
                entry_points=(
                    *context.graph.entry_points,
                    NodeRef(component_id=self.target),
                ),
            ),
            claims=frozenset(
                {
                    context.claim("node", self.target),
                    context.claim("entry_point", self.target),
                }
            ),
        )


def test_disjoint_resolution_additions_are_order_independent() -> None:
    graph = _graph()
    rules = [_AddingRule("add_b", "b"), _AddingRule("add_a", "a")]
    expected = None
    for seed in range(8):
        shuffled = list(rules)
        random.Random(seed).shuffle(shuffled)
        plan = Compiler(RuleRegistry(shuffled)).compile(
            graph, CompileContext(enabled_rule_namespaces=frozenset({"test"}))
        )
        signature = tuple(node.component_id for node in plan.graph.nodes)
        if expected is None:
            expected = signature
        else:
            assert signature == expected
    assert expected == ("start", "a", "b")


@dataclass
class _OneShotRule:
    name: str = "one_shot"
    namespace: str = "test"
    phase: Literal["resolution"] = "resolution"
    calls: int = 0

    def apply(self, context: RuleContext) -> ResolutionResult:
        self.calls += 1
        if any(node.component_id == "added" for node in context.graph.nodes):
            return ResolutionResult(
                graph=context.graph,
                claims=frozenset(
                    {
                        context.claim("node", "added"),
                        context.claim("entry_point", "added"),
                    }
                ),
            )
        added = ComponentNode(component_id="added", component=_Component())
        return ResolutionResult(
            graph=replace(
                context.graph,
                nodes=(*context.graph.nodes, added),
                entry_points=(
                    *context.graph.entry_points,
                    NodeRef(component_id="added"),
                ),
            ),
            claims=frozenset(
                {
                    context.claim("node", "added"),
                    context.claim("entry_point", "added"),
                }
            ),
        )


def test_resolution_reaches_fixed_point() -> None:
    rule = _OneShotRule()
    plan = Compiler(RuleRegistry([rule])).compile(
        _graph(), CompileContext(enabled_rule_namespaces=frozenset({"test"}))
    )

    assert rule.calls == 2
    assert {node.component_id for node in plan.graph.nodes} == {"start", "added"}
    assert not any(
        diagnostic.code == "unstable_compilation" for diagnostic in plan.diagnostics
    )


@dataclass
class _StructuredNodeRewriteRule:
    name: str = "structured_node_rewrite"
    namespace: str = "test"
    phase: Literal["resolution"] = "resolution"

    def apply(self, context: RuleContext) -> ResolutionResult:
        nodes = tuple(
            node.with_resolved_services({"marker": self})
            for node in context.graph.nodes
        )
        return ResolutionResult(
            graph=replace(context.graph, nodes=nodes),
            claims=frozenset(
                context.claim("node", node.component_id) for node in context.graph.nodes
            ),
        )


def test_structured_rewrite_rebinds_operations_and_alternate_regions() -> None:
    graph = lower_structured(
        [
            BranchRegion(
                region_id="branch",
                condition=type(
                    "Condition",
                    (),
                    {
                        "contract": lambda self: StateContract(),
                        "evaluate": lambda self, view: True,
                    },
                )(),
                body=(_Component(),),
                otherwise=(_Component(),),
            ),
            _Component(),
        ]
    )
    plan = Compiler(RuleRegistry([_StructuredNodeRewriteRule()])).compile(
        graph, CompileContext(enabled_rule_namespaces=frozenset({"test"}))
    )

    assert isinstance(plan.graph, StructuredGraph)
    assert all(
        operation
        is next(
            node
            for node in plan.graph.nodes
            if node.component_id == operation.component_id
        )
        for operation in plan.graph.operations
        if isinstance(operation, ComponentNode)
    )
    branch = plan.graph.region_nodes[0].region
    assert isinstance(branch, BranchRegion)
    assert isinstance(branch.body, StructuredGraph)
    assert all(
        operation
        is next(
            node
            for node in plan.graph.nodes
            if node.component_id == operation.component_id
        )
        for operation in branch.body.operations
        if isinstance(operation, ComponentNode)
    )
    assert isinstance(branch.otherwise, StructuredGraph)
    assert all(
        operation
        is next(
            node
            for node in plan.graph.nodes
            if node.component_id == operation.component_id
        )
        for operation in branch.otherwise.operations
        if isinstance(operation, ComponentNode)
    )


@dataclass
class _StructuredOperationMutationRule:
    name: str = "structured_operation_mutation"
    namespace: str = "test"
    phase: Literal["resolution"] = "resolution"

    def apply(self, context: RuleContext) -> ResolutionResult:
        assert isinstance(context.graph, StructuredGraph)
        return ResolutionResult(
            graph=replace(
                context.graph,
                operations=tuple(reversed(context.graph.operations)),
            )
        )


def test_structured_resolution_cannot_change_operation_order() -> None:
    graph = lower_structured([_Component(), _Component()])
    plan = Compiler(RuleRegistry([_StructuredOperationMutationRule()])).compile(
        graph,
        CompileContext(enabled_rule_namespaces=frozenset({"test"})),
    )

    assert any(
        diagnostic.code == "structured_execution_mutation"
        for diagnostic in plan.diagnostics
    )


@dataclass
class _StructuredExecutableOperationMutationRule:
    name: str = "structured_executable_operation_mutation"
    namespace: str = "test"
    phase: Literal["resolution"] = "resolution"

    def apply(self, context: RuleContext) -> ResolutionResult:
        assert isinstance(context.graph, StructuredGraph)
        extra = ComponentNode(component_id="extra", component=_Component())
        return ResolutionResult(
            graph=replace(
                context.graph,
                nodes=(*context.graph.nodes, extra),
                operations=(*context.graph.operations, extra),
            )
        )


def test_structured_resolution_cannot_add_executable_operation() -> None:
    plan = Compiler(
        RuleRegistry([_StructuredExecutableOperationMutationRule()])
    ).compile(
        lower_structured([_Component(), _Component()]),
        CompileContext(enabled_rule_namespaces=frozenset({"test"})),
    )

    assert any(
        diagnostic.code == "structured_execution_mutation"
        for diagnostic in plan.diagnostics
    )


@dataclass
class _StructuredSemanticNodeRule:
    name: str = "structured_semantic_node"
    namespace: str = "test"
    phase: Literal["resolution"] = "resolution"

    def apply(self, context: RuleContext) -> ResolutionResult:
        if any(node.component_id == "semantic" for node in context.graph.nodes):
            return ResolutionResult(
                graph=context.graph,
                claims=frozenset({context.claim("node", "semantic")}),
            )
        semantic = ComponentNode(component_id="semantic", component=_Component())
        return ResolutionResult(
            graph=replace(context.graph, nodes=(*context.graph.nodes, semantic)),
            claims=frozenset({context.claim("node", "semantic")}),
        )


def test_structured_resolution_allows_semantic_only_nodes() -> None:
    plan = Compiler(RuleRegistry([_StructuredSemanticNodeRule()])).compile(
        lower_structured([_Component(), _Component()]),
        CompileContext(enabled_rule_namespaces=frozenset({"test"})),
    )

    assert "semantic" in {node.component_id for node in plan.graph.nodes}
    assert not any(
        diagnostic.code == "structured_execution_mutation"
        for diagnostic in plan.diagnostics
    )


@dataclass
class _StructuredNestedTopologyMutationRule:
    name: str = "structured_nested_topology_mutation"
    namespace: str = "test"
    phase: Literal["resolution"] = "resolution"

    def apply(self, context: RuleContext) -> ResolutionResult:
        assert isinstance(context.graph, StructuredGraph)
        region_node = context.graph.region_nodes[0]
        region = region_node.region
        assert isinstance(region.body, StructuredGraph)
        body = replace(region.body, operations=tuple(reversed(region.body.operations)))
        mutated_region_node = replace(region_node, region=replace(region, body=body))
        return ResolutionResult(
            graph=replace(
                context.graph,
                region_nodes=(mutated_region_node,),
                operations=(mutated_region_node,),
            )
        )


def test_structured_resolution_cannot_change_nested_topology() -> None:
    graph = lower_structured(
        [
            BranchRegion(
                region_id="branch",
                condition=type(
                    "Condition",
                    (),
                    {
                        "contract": lambda self: StateContract(),
                        "evaluate": lambda self, view: True,
                    },
                )(),
                body=(_Component(), _Component()),
            )
        ]
    )
    plan = Compiler(RuleRegistry([_StructuredNestedTopologyMutationRule()])).compile(
        graph, CompileContext(enabled_rule_namespaces=frozenset({"test"}))
    )

    assert any(
        diagnostic.code == "structured_execution_mutation"
        for diagnostic in plan.diagnostics
    )


@dataclass
class _NeverStableRule:
    name: str = "never_stable"
    namespace: str = "test"
    phase: Literal["resolution"] = "resolution"
    calls: int = 0

    def apply(self, context: RuleContext) -> ResolutionResult:
        self.calls += 1
        node = replace(
            context.graph.nodes[0],
            role=f"role_{self.calls}",
        )
        return ResolutionResult(
            graph=replace(context.graph, nodes=(node,)),
            claims=frozenset({context.claim("node", "start")}),
        )


class _ShortCompiler(Compiler):
    MAX_RESOLUTION_ITERATIONS = 3


def test_resolution_iteration_cap_reports_unstable_compilation() -> None:
    rule = _NeverStableRule()
    plan = _ShortCompiler(RuleRegistry([rule])).compile(
        _graph(), CompileContext(enabled_rule_namespaces=frozenset({"test"}))
    )

    assert rule.calls == 3
    assert [
        diagnostic.code
        for diagnostic in plan.diagnostics
        if diagnostic.code == "unstable_compilation"
    ] == ["unstable_compilation"]


def test_verification_diagnostics_are_collected() -> None:
    plan = Compiler(RuleRegistry([_FindingRule("a"), _FindingRule("b")])).compile(
        _graph(), CompileContext(enabled_rule_namespaces=frozenset({"test"}))
    )

    assert [
        diagnostic.message
        for diagnostic in plan.diagnostics
        if diagnostic.message.startswith("finding")
    ] == ["finding from a", "finding from b"]


def _graph_with_missing_reference(kind: str) -> ComponentGraph:
    nodes = (
        ComponentNode(component_id="a", component=_Component()),
        ComponentNode(component_id="b", component=_Component()),
    )
    if kind == "data_source":
        return ComponentGraph(
            nodes=nodes,
            data_edges=(
                DataEdge(
                    source=NodeRef(component_id="ghost"),
                    target=NodeRef(component_id="b"),
                    source_port="out",
                    target_port="in",
                ),
            ),
            entry_points=(NodeRef(component_id="a"),),
        )
    if kind == "data_target":
        return ComponentGraph(
            nodes=nodes,
            data_edges=(
                DataEdge(
                    source=NodeRef(component_id="a"),
                    target=NodeRef(component_id="ghost"),
                    source_port="out",
                    target_port="in",
                ),
            ),
            entry_points=(NodeRef(component_id="a"),),
        )
    if kind == "control":
        return ComponentGraph(
            nodes=nodes,
            control_edges=(
                ControlEdge(
                    source=NodeRef(component_id="a"),
                    target=NodeRef(component_id="ghost"),
                ),
            ),
            entry_points=(NodeRef(component_id="a"),),
        )
    if kind == "entry_point":
        return ComponentGraph(
            nodes=nodes,
            entry_points=(NodeRef(component_id="ghost"),),
        )
    return ComponentGraph(
        nodes=nodes,
        state_bindings=(
            StateBinding(
                node=NodeRef(component_id="ghost"),
                state_key=StateKey(namespace="user", name="value", schema_version=1),
            ),
        ),
        entry_points=(NodeRef(component_id="a"),),
    )


@pytest.mark.parametrize(
    ("kind", "expected_code"),
    [
        ("data_source", "invalid_graph_edge"),
        ("data_target", "invalid_graph_edge"),
        ("control", "invalid_graph_edge"),
        ("entry_point", "invalid_entry_point"),
        ("state_binding", "invalid_state_binding"),
    ],
)
def test_compile_reports_missing_node_references_as_diagnostics(
    kind, expected_code
) -> None:
    plan = Compiler().compile(_graph_with_missing_reference(kind))

    findings = [
        diagnostic
        for diagnostic in plan.diagnostics
        if diagnostic.code == expected_code
    ]
    assert findings
    assert all(diagnostic.severity is Severity.ERROR for diagnostic in findings)
    assert plan.active_rule_namespaces == frozenset()


def test_core_rules_are_registered_once_and_not_duplicated_by_well_formedness() -> None:
    graph = ComponentGraph(
        nodes=(
            ComponentNode(component_id="start", component=_Component()),
            ComponentNode(component_id="start", component=_Component()),
            ComponentNode(component_id="orphan", component=_Component()),
        ),
        entry_points=(NodeRef(component_id="start"),),
    )
    plan = Compiler().compile(graph)

    assert [
        diagnostic.code
        for diagnostic in plan.diagnostics
        if diagnostic.code == "duplicate_component_id"
    ] == ["duplicate_component_id"]
    assert [
        diagnostic.code
        for diagnostic in plan.diagnostics
        if diagnostic.code == "unreachable_node"
    ] == ["unreachable_node"]


def test_compiler_rejects_registration_collision_with_core_rule() -> None:
    with pytest.raises(ConfigurationError, match="core:identity"):
        Compiler(RuleRegistry([_VerificationRule(name="identity", namespace="core")]))


@dataclass
class _NamespacedVerificationRule:
    name: str = "verification"
    namespace: str = "extension"
    phase: Literal["verification"] = "verification"
    calls: int = 0

    def apply(self, context: RuleContext) -> VerificationResult:
        self.calls += 1
        return VerificationResult()


@dataclass
class _NamespacedResolutionRule:
    name: str = "resolution"
    namespace: str = "extension"
    phase: Literal["resolution"] = "resolution"
    calls: int = 0

    def apply(self, context: RuleContext) -> ResolutionResult:
        self.calls += 1
        return ResolutionResult(
            graph=context.graph,
            claims=frozenset({context.claim("node", "start")}),
        )


def test_third_party_rule_activation_differs_by_phase() -> None:
    verification = _NamespacedVerificationRule()
    resolution = _NamespacedResolutionRule()
    registry = RuleRegistry([verification, resolution])
    plain = _graph()
    namespaced = replace(
        plain,
        nodes=(ComponentNode(component_id="extension:start", component=_Component()),),
        entry_points=(NodeRef(component_id="extension:start"),),
    )

    Compiler(registry).compile(namespaced)
    assert verification.calls == 1
    assert resolution.calls == 0
    Compiler(registry).compile(plain)
    assert verification.calls == 1
    assert resolution.calls == 0
    Compiler(registry).compile(
        plain, CompileContext(enabled_rule_namespaces=frozenset({"extension"}))
    )
    assert verification.calls == 2
    assert resolution.calls == 1


def test_compiler_package_does_not_import_concrete_component_packages() -> None:
    compiler_root = Path(__file__).parents[3] / "src" / "saealib" / "core" / "compiler"
    imports: list[str] = []
    for path in compiler_root.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                imports.append(node.module)

    saealib_imports = tuple(
        imported
        for imported in imports
        if imported == "saealib" or imported.startswith("saealib.")
    )
    allowed = (
        "saealib.core",
        "saealib.exceptions",
    )
    unexpected = tuple(
        imported
        for imported in saealib_imports
        if not any(
            imported == package or imported.startswith(f"{package}.")
            for package in allowed
        )
    )
    assert not unexpected, (
        f"compiler package crossed the contract boundary: {unexpected}"
    )


def test_executable_plan_aggregates_required_capabilities_without_runtime_check() -> (
    None
):
    component = type(
        "CapabilityComponent",
        (),
        {
            "contract": lambda self: ComponentContract(
                execution=ExecutionContract(
                    required_runtime_capabilities=("partial_feedback",),
                )
            )
        },
    )()
    graph = ComponentGraph(
        nodes=(ComponentNode(component_id="start", component=component),),
        entry_points=(NodeRef(component_id="start"),),
    )

    plan = Compiler().compile(graph)

    assert plan.required_runtime_capabilities == frozenset({"partial_feedback"})
    assert not any(
        diagnostic.code == "unknown_runtime_capability"
        for diagnostic in plan.diagnostics
    )
