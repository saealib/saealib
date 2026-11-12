"""Signature contract for the context-dependent public hooks.

Every context-dependent hook accepts a trailing, keyword-only, required
``ctx`` annotated ``OptimizationState | None``. Selection implementations
reject ``ctx=None`` with ``ValidationError``; training-set classes that need
a comparator keep their explicit ``requires ctx`` rejections.
"""

import inspect
from typing import cast

import pytest

import saealib
from saealib.acquisition.archive_based import (
    InverseDensityAcquisition,
    MaximinDistanceAcquisition,
    NoveltyAcquisition,
)
from saealib.acquisition.base import (
    AcquisitionFunction,
    CompositeAcquisition,
    PointwiseAcquisition,
)
from saealib.acquisition.batch import BatchExpectedImprovement
from saealib.acquisition.lcb import LowerConfidenceBound
from saealib.acquisition.mean import CORSDistance
from saealib.acquisition.winrate import WinRateAcquisition
from saealib.exceptions import ValidationError
from saealib.operators.selection import (
    LinearRankSelection,
    ParentSelection,
    SequentialSelection,
    SurvivorSelection,
    TournamentSelection,
    TruncationSelection,
)
from saealib.policies.evaluation import (
    EvaluateAll,
    EvaluationPlanner,
    FidelityEvaluation,
    FidelityPromotion,
    RatioEvaluation,
    RepeatedEvaluation,
    TopKEvaluation,
)
from saealib.policies.feedback import (
    ComparatorWorstFallback,
    FeedbackBuilder,
    MixedFeedback,
    NoFeedback,
    PredictedFeedback,
    TrueOnlyFeedback,
)
from saealib.population import Archive, Population
from saealib.surrogate.manager import (
    CompositeSurrogateManager,
    GlobalSurrogateManager,
    LocalSurrogateManager,
    PairwiseSurrogateManager,
    SurrogateManager,
)
from saealib.surrogate.training_set import (
    ArchiveObjectiveSet,
    ConstraintObjectiveSet,
    FeasibilityClassificationSet,
    KNNConstraintObjectiveSet,
    KNNObjectiveSet,
    LevelBasedSet,
    PairwiseComparisonSet,
    ReferencePointComparisonSet,
    TopKBipartitionSet,
    TrainingSet,
)

# (hook owner class, hook method name) pairs for every built-in override.
HOOK_OWNERS: list[tuple[type, str]] = [
    (ParentSelection, "select"),
    (TournamentSelection, "select"),
    (SequentialSelection, "select"),
    (LinearRankSelection, "select"),
    (SurvivorSelection, "select"),
    (TruncationSelection, "select"),
    (EvaluationPlanner, "plan"),
    (EvaluateAll, "plan"),
    (TopKEvaluation, "plan"),
    (RatioEvaluation, "plan"),
    (RepeatedEvaluation, "plan"),
    (FidelityEvaluation, "plan"),
    (FidelityPromotion, "plan"),
    (FeedbackBuilder, "build"),
    (TrueOnlyFeedback, "build"),
    (PredictedFeedback, "build"),
    (MixedFeedback, "build"),
    (NoFeedback, "build"),
    (ComparatorWorstFallback, "build"),
    (TrainingSet, "build"),
    (ArchiveObjectiveSet, "build"),
    (KNNObjectiveSet, "build"),
    (ConstraintObjectiveSet, "build"),
    (KNNConstraintObjectiveSet, "build"),
    (FeasibilityClassificationSet, "build"),
    (TopKBipartitionSet, "build"),
    (LevelBasedSet, "build"),
    (PairwiseComparisonSet, "build"),
    (ReferencePointComparisonSet, "build"),
    (AcquisitionFunction, "prepare"),
    (PointwiseAcquisition, "prepare"),
    (CompositeAcquisition, "prepare"),
    (CORSDistance, "prepare"),
    (LowerConfidenceBound, "prepare"),
    (AcquisitionFunction, "evaluate"),
    (PointwiseAcquisition, "evaluate"),
    (CompositeAcquisition, "evaluate"),
    (NoveltyAcquisition, "evaluate"),
    (InverseDensityAcquisition, "evaluate"),
    (MaximinDistanceAcquisition, "evaluate"),
    (WinRateAcquisition, "evaluate"),
    (BatchExpectedImprovement, "evaluate"),
    (SurrogateManager, "fit"),
    (GlobalSurrogateManager, "fit"),
    (LocalSurrogateManager, "fit"),
    (CompositeSurrogateManager, "fit"),
    (PairwiseSurrogateManager, "fit"),
    (SurrogateManager, "predict"),
    (GlobalSurrogateManager, "predict"),
    (LocalSurrogateManager, "predict"),
    (CompositeSurrogateManager, "predict"),
    (PairwiseSurrogateManager, "predict"),
]


@pytest.mark.parametrize(("owner", "hook"), HOOK_OWNERS, ids=lambda value: value)
def test_hook_ctx_is_trailing_keyword_only_required(owner, hook):
    signature = inspect.signature(getattr(owner, hook))
    parameters = list(signature.parameters.values())
    assert parameters[-1].name == "ctx"
    ctx = parameters[-1]
    assert ctx.kind is inspect.Parameter.KEYWORD_ONLY
    assert ctx.default is inspect.Parameter.empty


@pytest.mark.parametrize(("owner", "hook"), HOOK_OWNERS, ids=lambda value: value)
def test_hook_ctx_is_annotated_optional_optimization_state(owner, hook):
    signature = inspect.signature(getattr(owner, hook))
    annotation = signature.parameters["ctx"].annotation
    assert annotation == "OptimizationState | None"


def test_selection_context_protocol_is_not_public():
    assert not hasattr(saealib, "_SelectionContext")
    for owner, hook in HOOK_OWNERS:
        signature = inspect.signature(getattr(owner, hook))
        for parameter in signature.parameters.values():
            assert "_SelectionContext" not in str(parameter.annotation)
    assert "_SelectionContext" not in dir(saealib)


def _call_dynamic(fn, *args, **kwargs):
    """Call a hook so wrong-argument shapes stay outside static checking."""
    return fn(*args, **kwargs)


def test_target_hooks_reject_omitted_and_positional_ctx():
    pop = cast(Population, object())
    arc = cast(Archive, object())
    truncation = TruncationSelection()
    with pytest.raises(TypeError):
        _call_dynamic(TournamentSelection(2).select, pop, 1, 1)
    with pytest.raises(TypeError):
        _call_dynamic(SequentialSelection().select, pop, 1, 1)
    with pytest.raises(TypeError):
        _call_dynamic(LinearRankSelection().select, pop, 1, 1)
    with pytest.raises(TypeError):
        _call_dynamic(truncation.select, pop, 1)
    with pytest.raises(TypeError):
        _call_dynamic(truncation.select, pop, pop, 1)
    with pytest.raises(TypeError):
        _call_dynamic(EvaluateAll().plan, pop, None, pop)
    with pytest.raises(TypeError):
        _call_dynamic(MixedFeedback().build, pop, None, None, [], pop)
    with pytest.raises(TypeError):
        _call_dynamic(ArchiveObjectiveSet().build, arc, None, arc)


def test_concrete_selections_reject_none_ctx():
    population = object()
    with pytest.raises(ValidationError, match="TournamentSelection requires ctx"):
        _call_dynamic(TournamentSelection(2).select, population, 1, 1, ctx=None)
    with pytest.raises(ValidationError, match="SequentialSelection requires ctx"):
        _call_dynamic(SequentialSelection().select, population, 1, 1, ctx=None)
    with pytest.raises(ValidationError, match="LinearRankSelection requires ctx"):
        _call_dynamic(LinearRankSelection().select, population, 1, 1, ctx=None)
    with pytest.raises(ValidationError, match="TruncationSelection requires ctx"):
        _call_dynamic(TruncationSelection().select, population, 1, ctx=None)


def test_training_sets_keep_their_required_ctx_rejections():
    archive = object()
    with pytest.raises(ValidationError, match="TopKBipartitionSet requires ctx"):
        _call_dynamic(TopKBipartitionSet().build, archive, None, ctx=None)
    with pytest.raises(ValidationError, match="LevelBasedSet requires ctx"):
        _call_dynamic(LevelBasedSet().build, archive, None, ctx=None)
    with pytest.raises(ValidationError, match="PairwiseComparisonSet requires ctx"):
        _call_dynamic(PairwiseComparisonSet().build, archive, None, ctx=None)
    with pytest.raises(
        ValidationError, match="ReferencePointComparisonSet requires ctx"
    ):
        _call_dynamic(ReferencePointComparisonSet().build, archive, None, ctx=None)


def test_optimization_state_stays_exported():
    assert "OptimizationState" in saealib._LAZY_EXPORTS
