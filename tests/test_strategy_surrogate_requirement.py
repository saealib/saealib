"""Flow-derived surrogate requirement for template strategies."""

from __future__ import annotations

import copy
import pickle
from unittest.mock import patch

import numpy as np
import pytest

from saealib import (
    GA,
    CrossoverBLXAlpha,
    LHSInitializer,
    MutationUniform,
    Optimizer,
    SequentialSelection,
    Termination,
    TruncationSelection,
)
from saealib.comparators import SingleObjectiveComparator
from saealib.core.compiler import ContractPath, Severity
from saealib.core.compiler.contract_diagnostics import (
    _optimizer_configuration_diagnostics,
)
from saealib.core.compiler.diagnostics import Diagnostic
from saealib.core.compiler.strategy_requirements import resolve_requires_surrogate
from saealib.execution.evaluator import SerialEvaluator
from saealib.pipeline import ComponentRef, Pipeline, Repeat
from saealib.problem import Problem
from saealib.stages import AskStage, CountGenerationStage, SurrogatePredictStage
from saealib.strategies import (
    DirectStrategy,
    GenerationBasedStrategy,
    IndividualBasedStrategy,
    PreSelectionStrategy,
    SteadyStateStrategy,
    TemplateStrategy,
)
from saealib.strategies.base import OptimizationStrategy
from saealib.strategies.direct import build_direct_template
from saealib.strategies.ib import build_ib_template


class DeclaredSurrogateRequirement(TemplateStrategy):
    """Template subclass declaring the requirement at class level."""

    requires_surrogate = True


class DeclaredNoSurrogateRequirement(TemplateStrategy):
    """Template subclass declaring no requirement at class level."""

    requires_surrogate = False


class MixinSurrogateRequirement:
    """Mixin declaring the requirement, placed before the template base."""

    requires_surrogate = True


class MixinNoSurrogateRequirement:
    """Mixin declaring no requirement, placed before the template base."""

    requires_surrogate = False


class InheritedSurrogateRequirement(MixinSurrogateRequirement, TemplateStrategy):
    """Template subclass inheriting the requirement from a mixin."""


class InheritedNoSurrogateRequirement(MixinNoSurrogateRequirement, TemplateStrategy):
    """Template subclass inheriting no requirement from a mixin."""


class LoweringOverride(TemplateStrategy):
    """Template subclass lowering its own graph and declaring the requirement."""

    requires_surrogate = False

    def build_graph(self, provider):
        return TemplateStrategy.build_graph(self, provider)


def _problem() -> Problem:
    return Problem(
        func=lambda _: np.zeros(1),
        dim=2,
        n_obj=1,
        direction=np.array([-1.0]),
        lb=[-1.0, -1.0],
        ub=[1.0, 1.0],
        comparator=SingleObjectiveComparator(),
    )


def _bindings(strategy: object) -> dict[str, object]:
    return {
        "algorithm": object(),
        "strategy": strategy,
        "initializer": object(),
        "termination": object(),
        "surrogate_manager": None,
        "acquisition": None,
        "evaluation_planner": None,
        "feedback_builder": None,
        "evaluator": SerialEvaluator(),
        "cbmanager": object(),
    }


def _findings(strategy: object) -> dict[str, Diagnostic]:
    findings = _optimizer_configuration_diagnostics(
        _bindings(strategy), {}, _problem(), SerialEvaluator()
    )
    return {finding.code: finding for finding in findings}


def _surrogate_stage() -> SurrogatePredictStage:
    return SurrogatePredictStage(ComponentRef("predictor"))


def _surrogate_declaration() -> Pipeline:
    return Pipeline(
        steps=[
            CountGenerationStage(),
            AskStage(ComponentRef("proposer")),
            _surrogate_stage(),
        ]
    )


@pytest.mark.parametrize(
    ("strategy", "expected"),
    [
        (DirectStrategy(), False),
        (SteadyStateStrategy(), False),
        (IndividualBasedStrategy(), True),
        (PreSelectionStrategy(8, 2), True),
        (GenerationBasedStrategy(1), True),
        (GenerationBasedStrategy(0), False),
    ],
)
def test_builtin_strategies_derive_the_requirement_from_their_flow(
    strategy: OptimizationStrategy, expected: bool
) -> None:
    assert resolve_requires_surrogate(strategy).effective is expected


def test_template_declaration_with_surrogate_reference_requires_a_manager() -> None:
    strategy = TemplateStrategy(_surrogate_declaration())
    assert strategy.requires_surrogate is None
    requirement = resolve_requires_surrogate(strategy)
    assert requirement.declares is True
    assert requirement.declared is None
    assert requirement.effective is True


def test_declaration_without_surrogate_reference_does_not_require_a_manager() -> None:
    strategy = TemplateStrategy(build_direct_template())
    assert resolve_requires_surrogate(strategy).effective is False


def test_surrogate_reference_inside_a_nested_repeat_requires_a_manager() -> None:
    declaration = Pipeline(
        steps=[
            Repeat(
                Pipeline(
                    steps=[_surrogate_stage()],
                    name="inner",
                ),
                count=3,
                name="outer",
            )
        ]
    )
    assert resolve_requires_surrogate(TemplateStrategy(declaration)).effective is True


def test_directly_held_surrogate_manager_requires_a_manager() -> None:
    from saealib.surrogate.manager import GlobalSurrogateManager
    from saealib.surrogate.rbf import RBFSurrogate
    from saealib.surrogate.rbf_kernels import GaussianKernel

    declaration = Pipeline(
        steps=[
            SurrogatePredictStage(
                GlobalSurrogateManager(RBFSurrogate(GaussianKernel()))
            )
        ]
    )
    assert resolve_requires_surrogate(TemplateStrategy(declaration)).effective is True


def test_derived_requirement_drives_the_missing_manager_error() -> None:
    findings = _findings(IndividualBasedStrategy())
    assert findings["missing_surrogate_manager"].severity is Severity.ERROR
    assert findings["missing_surrogate_manager"].path == ContractPath(
        components=("optimizer", "surrogate_manager")
    )
    assert "missing_acquisition" in findings


def test_derived_false_flow_reports_no_missing_manager_error() -> None:
    findings = _findings(TemplateStrategy(build_direct_template()))
    assert "missing_surrogate_manager" not in findings
    assert "missing_acquisition" not in findings


def test_declared_requirement_contradicting_the_flow_is_reported_not_used() -> None:
    strategy = TemplateStrategy(build_direct_template(), requires_surrogate=True)
    assert strategy.requires_surrogate is True

    requirement = resolve_requires_surrogate(strategy)
    assert requirement.declared is True
    assert requirement.effective is False

    findings = _findings(strategy)
    contradiction = findings["contradicting_surrogate_requirement"]
    assert contradiction.severity is Severity.WARNING
    assert contradiction.path == ContractPath(components=("optimizer", "strategy"))
    assert "requires_surrogate=True" in contradiction.message
    assert "missing_surrogate_manager" not in findings


def test_declared_requirement_matching_the_flow_is_not_reported() -> None:
    strategy = TemplateStrategy(build_ib_template(), requires_surrogate=True)
    requirement = resolve_requires_surrogate(strategy)
    assert requirement.declared is True
    assert requirement.effective is True
    assert "contradicting_surrogate_requirement" not in _findings(strategy)


def test_plain_subclass_of_a_builtin_keeps_the_derived_requirement() -> None:
    class UserIndividualBased(IndividualBasedStrategy):
        pass

    class UserGenerationBased(GenerationBasedStrategy):
        pass

    assert resolve_requires_surrogate(UserIndividualBased()).effective is True
    assert resolve_requires_surrogate(UserGenerationBased(0)).effective is False


def test_class_attribute_true_on_a_flow_without_surrogate_stages_is_not_used() -> None:
    strategy = DeclaredSurrogateRequirement(build_direct_template())
    assert strategy.requires_surrogate is True

    requirement = resolve_requires_surrogate(strategy)
    assert requirement.declared is True
    assert requirement.effective is False

    findings = _findings(strategy)
    contradiction = findings["contradicting_surrogate_requirement"]
    assert contradiction.severity is Severity.WARNING
    assert "requires_surrogate=True" in contradiction.message
    assert "missing_surrogate_manager" not in findings


def test_class_attribute_false_on_a_flow_with_surrogate_stages_is_not_used() -> None:
    strategy = DeclaredNoSurrogateRequirement(_surrogate_declaration())
    assert strategy.requires_surrogate is False

    requirement = resolve_requires_surrogate(strategy)
    assert requirement.declared is False
    assert requirement.effective is True

    findings = _findings(strategy)
    contradiction = findings["contradicting_surrogate_requirement"]
    assert contradiction.severity is Severity.WARNING
    assert "requires_surrogate=False" in contradiction.message
    assert findings["missing_surrogate_manager"].severity is Severity.ERROR
    assert "missing_acquisition" in findings


def test_class_attribute_matching_the_flow_is_not_reported() -> None:
    strategy = DeclaredSurrogateRequirement(_surrogate_declaration())
    assert resolve_requires_surrogate(strategy).effective is True
    assert "contradicting_surrogate_requirement" not in _findings(strategy)


def test_mixin_inherited_requirement_contradicting_the_flow_is_reported() -> None:
    strategy = InheritedNoSurrogateRequirement(_surrogate_declaration())
    assert strategy.requires_surrogate is False

    requirement = resolve_requires_surrogate(strategy)
    assert requirement.declared is False
    assert requirement.effective is True

    findings = _findings(strategy)
    assert "contradicting_surrogate_requirement" in findings
    assert findings["missing_surrogate_manager"].severity is Severity.ERROR


def test_mixin_inherited_requirement_on_a_direct_flow_is_reported() -> None:
    strategy = InheritedSurrogateRequirement(build_direct_template())
    assert strategy.requires_surrogate is True

    requirement = resolve_requires_surrogate(strategy)
    assert requirement.declared is True
    assert requirement.effective is False

    findings = _findings(strategy)
    assert "contradicting_surrogate_requirement" in findings
    assert "missing_surrogate_manager" not in findings


def test_instance_assignment_is_a_declaration() -> None:
    strategy = DeclaredSurrogateRequirement(build_direct_template())
    strategy.requires_surrogate = True
    assert strategy.requires_surrogate is True

    requirement = resolve_requires_surrogate(strategy)
    assert requirement.declared is True
    assert requirement.effective is False
    assert "contradicting_surrogate_requirement" in _findings(strategy)


def test_pickle_keeps_the_declared_requirement_and_derives_the_flow() -> None:
    strategy = TemplateStrategy(build_direct_template(), requires_surrogate=True)
    restored = pickle.loads(pickle.dumps(strategy))
    assert restored.requires_surrogate is True
    assert resolve_requires_surrogate(restored).declared is True
    assert resolve_requires_surrogate(restored).effective is False
    assert "contradicting_surrogate_requirement" in _findings(restored)


def test_pickle_derives_the_requirement_of_builtin_strategies() -> None:
    restored = pickle.loads(pickle.dumps(IndividualBasedStrategy(0.3)))
    assert resolve_requires_surrogate(restored).effective is True
    restored = pickle.loads(pickle.dumps(GenerationBasedStrategy(0)))
    assert resolve_requires_surrogate(restored).effective is False


def test_pickle_of_a_template_strategy_still_runs() -> None:
    strategy = TemplateStrategy(build_direct_template())
    restored = pickle.loads(pickle.dumps(strategy))
    with patch("saealib.defaults.load_defaults", return_value=_manager_free_defaults()):
        optimizer = _manager_free_optimizer(restored)
        assert optimizer.validate(require_initializer=False) == []
        state = optimizer.run()
    assert state.gen == 2
    assert state.fe > 0


def test_spec_round_trip_keeps_the_declared_requirement() -> None:
    from saealib.registry import build, to_spec

    strategy = TemplateStrategy(build_direct_template(), requires_surrogate=True)
    rebuilt = build(to_spec(strategy))
    assert rebuilt.requires_surrogate is True
    assert resolve_requires_surrogate(rebuilt).effective is False


def test_spec_round_trip_of_a_builtin_derives_the_requirement() -> None:
    from saealib.registry import build, to_spec

    rebuilt = build(to_spec(GenerationBasedStrategy(0)))
    assert resolve_requires_surrogate(rebuilt).effective is False


def test_lowering_override_keeps_its_declared_requirement() -> None:
    strategy = LoweringOverride(build_direct_template())
    assert strategy.requires_surrogate is False
    assert resolve_requires_surrogate(strategy).effective is False
    assert "missing_surrogate_manager" not in _findings(strategy)


def test_lowering_override_keeps_an_instance_requirement_after_pickle() -> None:
    strategy = LoweringOverride(build_direct_template())
    strategy.requires_surrogate = True
    restored = pickle.loads(pickle.dumps(strategy))
    assert restored.requires_surrogate is True

    requirement = resolve_requires_surrogate(restored)
    assert requirement.declared is True
    assert requirement.effective is True
    findings = _findings(restored)
    assert findings["missing_surrogate_manager"].severity is Severity.ERROR
    assert "contradicting_surrogate_requirement" not in findings


def test_lowering_override_keeps_an_instance_requirement_in_a_spec_round_trip() -> None:
    from saealib.registry import build, to_spec

    strategy = LoweringOverride(build_direct_template())
    strategy.requires_surrogate = True
    spec = to_spec(strategy)
    assert spec["params"]["requires_surrogate"] is True
    rebuilt = build(spec)
    assert rebuilt.requires_surrogate is True
    assert resolve_requires_surrogate(rebuilt).effective is True
    assert "missing_surrogate_manager" in _findings(rebuilt)


def _algorithm() -> GA:
    return GA(
        crossover=CrossoverBLXAlpha(prob=0.9, alpha=0.4),
        mutation=MutationUniform(prob_var=0.1),
        parent_selection=SequentialSelection(),
        survivor_selection=TruncationSelection(),
    )


def _manager_free_optimizer(strategy: TemplateStrategy) -> Optimizer:
    optimizer = (
        Optimizer(_problem(), seed=19)
        .set_initializer(LHSInitializer(6, 4, 19))
        .set_algorithm(_algorithm())
        .set_strategy(strategy)
        .set_termination(Termination(lambda state: state.gen >= 2))
        .set_evaluator(SerialEvaluator())
    )
    setattr(optimizer, "surrogate_manager", None)
    setattr(optimizer, "acquisition", None)
    return optimizer


def _manager_free_defaults() -> dict[str, object]:
    from saealib.defaults.loader import load_defaults

    defaults = copy.deepcopy(load_defaults())
    for preset in defaults["presets"].values():
        preset.pop("surrogate_manager", None)
    return defaults


def test_optimizer_accepts_a_class_attribute_true_flow_without_a_manager() -> None:
    with patch("saealib.defaults.load_defaults", return_value=_manager_free_defaults()):
        optimizer = _manager_free_optimizer(
            DeclaredSurrogateRequirement(build_direct_template())
        )
        issues = optimizer.validate(require_initializer=False)
        assert issues == []
        state = optimizer.run()

    assert state.gen == 2
    assert state.fe > 0


def test_optimizer_reports_derived_requirement_for_class_attribute_false_flow() -> None:
    with patch("saealib.defaults.load_defaults", return_value=_manager_free_defaults()):
        optimizer = _manager_free_optimizer(
            DeclaredNoSurrogateRequirement(_surrogate_declaration())
        )
        issues = optimizer.validate(require_initializer=False)
        assert any("missing_surrogate_manager" in issue for issue in issues)
        assert any("missing_acquisition" in issue for issue in issues)

        with pytest.raises(Exception, match="surrogate_manager"):
            optimizer.run()
