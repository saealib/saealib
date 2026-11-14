"""saealib: Surrogate-Assisted Evolutionary Algorithms (SAEAs) Library."""

import logging
from importlib.metadata import version

__version__ = version("saealib")

# Root exports are curated independently from each subpackage's namespace.
# Names in ``_LAZY_EXPORTS`` are root conveniences, not an automatic mirror;
# subpackage ``__all__`` entries remain valid public APIs at their own path.

from saealib.acquisition import (
    AcquisitionFunction,
    AcquisitionResult,
    BatchExpectedImprovement,
    ExpectedImprovement,
    PointwiseAcquisition,
)
from saealib.algorithms import GA, PSO, Algorithm, AskTellAlgorithm, GenomeGA
from saealib.api import maximize, minimize
from saealib.callback import (
    CallbackManager,
    ConfigurationChange,
    ConfigurationCommittedEvent,
    Event,
    GenerationEndEvent,
    GenerationStartEvent,
    InitialEvaluationEndEvent,
    InitialEvaluationStartEvent,
    PostEvaluationEvent,
    RunEndEvent,
    RunStartEvent,
)
from saealib.checkpoint import CheckpointCallback
from saealib.comparators import Comparator, NSGA2Comparator, SingleObjectiveComparator
from saealib.exceptions import (
    CheckpointError,
    ConfigurationError,
    EvaluationFatalError,
    EvaluationProtocolError,
    EvaluationSubmissionError,
    RejectedChange,
    ReplacementRejectedError,
    SaealibError,
    ValidationError,
)
from saealib.execution.evaluator import (
    AsyncEvaluator,
    EvaluationErrorInfo,
    EvaluationHandle,
    EvaluationRequest,
    EvaluationResult,
    EvaluationStatus,
    EvaluationUpdate,
    Evaluator,
    PendingEvaluation,
    SerialEvaluator,
    ThreadPoolEvaluator,
)
from saealib.execution.initializer import (
    GenomeInitializer,
    Initializer,
    LHSInitializer,
    RandomInitializer,
    SobolInitializer,
)
from saealib.execution.scheduler import AsyncEvaluationScheduler
from saealib.islands import IslandModel
from saealib.operators import (
    Crossover,
    CrossoverSBX,
    DuplicateElimination,
    Mutation,
    MutationPolynomial,
    ParentSelection,
    SurvivorSelection,
    TournamentSelection,
    TruncationSelection,
)
from saealib.optimizer import Optimizer
from saealib.pipeline import (
    Branch,
    Condition,
    Loop,
    Pipeline,
    PipelineEntry,
    Repeat,
    Stage,
)
from saealib.policies import (
    ComparatorWorstFallback,
    EvaluateAll,
    EvaluationPlan,
    EvaluationPlanner,
    FeedbackBuilder,
    FeedbackResult,
    FidelityEvaluation,
    FidelityPromotion,
    MixedFeedback,
    NoFeedback,
    PredictedFeedback,
    RatioEvaluation,
    RepeatedEvaluation,
    ReplicateSummary,
    TopKEvaluation,
    TrueOnlyFeedback,
    aggregate_replicates,
)
from saealib.population import (
    Archive,
    Individual,
    ParetoArchive,
    Population,
    PopulationAttribute,
)
from saealib.problem import (
    ConstraintHandler,
    EpsilonConstraintHandler,
    EqualityConstraint,
    InequalityConstraint,
    Problem,
    StaticToleranceHandler,
)

# Activate the standard vector profile's adapters without making the
# framework compiler import vector implementation modules.
from saealib.profiles.vector import activate as _activate_vector_profile
from saealib.registry import register
from saealib.result import Result
from saealib.stages import (
    AcquisitionStage,
    ArchiveUpdateStage,
    AskStage,
    AsyncEvaluationSubmitStage,
    CountGenerationStage,
    EvaluationAcknowledgeStage,
    EvaluationApplyStage,
    EvaluationCollectStage,
    EvaluationPlanStage,
    EvaluationSubmitStage,
    FeedbackStage,
    InitializationStage,
    SurrogateFitStage,
    SurrogateOnlyLoopStage,
    SurrogatePredictStage,
    TellStage,
)
from saealib.strategies import (
    DirectStrategy,
    GenerationBasedStrategy,
    IndividualBasedStrategy,
    OptimizationStrategy,
    PreSelectionStrategy,
    SteadyStateStrategy,
)
from saealib.surrogate import Surrogate, SurrogateManager
from saealib.termination import (
    Termination,
    TerminationCondition,
    f_target,
    max_fe,
    max_gen,
    max_steps,
    stalled,
)
from saealib.utils.indicators import hypervolume, hypervolume_contributions
from saealib.variables import (
    CategoricalVariable,
    ContinuousVariable,
    IntegerVariable,
    Variable,
)

_activate_vector_profile()

_logger = logging.getLogger(__name__)
_logger.addHandler(logging.NullHandler())

__all__ = [
    "GA",
    "PSO",
    "AcquisitionFunction",
    "AcquisitionResult",
    "AcquisitionStage",
    "Algorithm",
    "Archive",
    "ArchiveUpdateStage",
    "AskStage",
    "AskTellAlgorithm",
    "AsyncEvaluationScheduler",
    "AsyncEvaluationSubmitStage",
    "AsyncEvaluator",
    "BatchExpectedImprovement",
    "Branch",
    "CallbackManager",
    "CategoricalVariable",
    "CheckpointCallback",
    "CheckpointError",
    "Comparator",
    "ComparatorWorstFallback",
    "Condition",
    "ConfigurationChange",
    "ConfigurationCommittedEvent",
    "ConfigurationError",
    "ConstraintHandler",
    "ContinuousVariable",
    "CountGenerationStage",
    "Crossover",
    "CrossoverSBX",
    "DirectStrategy",
    "DuplicateElimination",
    "EpsilonConstraintHandler",
    "EqualityConstraint",
    "EvaluateAll",
    "EvaluationErrorInfo",
    "EvaluationFatalError",
    "EvaluationHandle",
    "EvaluationPlan",
    "EvaluationPlanner",
    "EvaluationProtocolError",
    "EvaluationRequest",
    "EvaluationResult",
    "EvaluationStatus",
    "EvaluationSubmissionError",
    "EvaluationUpdate",
    "Evaluator",
    "Event",
    "ExpectedImprovement",
    "FeedbackBuilder",
    "FeedbackResult",
    "FeedbackStage",
    "FidelityEvaluation",
    "FidelityPromotion",
    "GenerationBasedStrategy",
    "GenerationEndEvent",
    "GenerationStartEvent",
    "GenomeGA",
    "GenomeInitializer",
    "Individual",
    "IndividualBasedStrategy",
    "InequalityConstraint",
    "InitialEvaluationEndEvent",
    "InitialEvaluationStartEvent",
    "InitializationStage",
    "Initializer",
    "IntegerVariable",
    "IslandModel",
    "LHSInitializer",
    "Loop",
    "MixedFeedback",
    "Mutation",
    "MutationPolynomial",
    "NSGA2Comparator",
    "NoFeedback",
    "OptimizationStrategy",
    "Optimizer",
    "ParentSelection",
    "ParetoArchive",
    "PendingEvaluation",
    "Pipeline",
    "PipelineEntry",
    "PointwiseAcquisition",
    "Population",
    "PopulationAttribute",
    "PostEvaluationEvent",
    "PreSelectionStrategy",
    "PredictedFeedback",
    "Problem",
    "RandomInitializer",
    "RatioEvaluation",
    "RejectedChange",
    "Repeat",
    "RepeatedEvaluation",
    "ReplacementRejectedError",
    "ReplicateSummary",
    "Result",
    "RunEndEvent",
    "RunStartEvent",
    "SaealibError",
    "SerialEvaluator",
    "SingleObjectiveComparator",
    "SobolInitializer",
    "Stage",
    "StaticToleranceHandler",
    "SteadyStateStrategy",
    "Surrogate",
    "SurrogateFitStage",
    "SurrogateManager",
    "SurrogateOnlyLoopStage",
    "SurrogatePredictStage",
    "SurvivorSelection",
    "TellStage",
    "Termination",
    "TerminationCondition",
    "ThreadPoolEvaluator",
    "TopKEvaluation",
    "TournamentSelection",
    "TrueOnlyFeedback",
    "TruncationSelection",
    "ValidationError",
    "Variable",
    "aggregate_replicates",
    "f_target",
    "hypervolume",
    "hypervolume_contributions",
    "max_fe",
    "max_gen",
    "max_steps",
    "maximize",
    "minimize",
    "register",
    "stalled",
]

# ---------------------------------------------------------------------------
# Lazy root surface — imported on first access and shown in dir()
# ---------------------------------------------------------------------------

_LAZY_EXPORTS: dict[str, str] = {
    # algorithms (less common)
    "PymooAlgorithm": "saealib.algorithms",
    # transitions
    "AlignColumns": "saealib.transition",
    "Chain": "saealib.transition",
    "EvaluateRequest": "saealib.transition",
    "EvaluationOutcome": "saealib.transition",
    "Transition": "saealib.transition",
    "TransitionContext": "saealib.transition",
    "chain": "saealib.transition",
    # context
    "OptimizationState": "saealib.context",
    # comparators (less common)
    "Dominator": "saealib.comparators",
    "EpsilonDominanceComparator": "saealib.comparators",
    "EpsilonDominator": "saealib.comparators",
    "HypervolumeComparator": "saealib.comparators",
    "NonDominatedSorter": "saealib.comparators",
    "NSGA3Comparator": "saealib.comparators",
    "ParetoComparator": "saealib.comparators",
    "ParetoDominator": "saealib.comparators",
    "RNSGA2Comparator": "saealib.comparators",
    "SPEA2Comparator": "saealib.comparators",
    "WeightedSumComparator": "saealib.comparators",
    "crowding_distance": "saealib.comparators",
    "crowding_distance_all_fronts": "saealib.comparators",
    "dda_non_dominated_sort": "saealib.comparators",
    "non_dominated_sort": "saealib.comparators",
    "spea2_fitness": "saealib.comparators",
    # execution (parallel)
    "JoblibEvaluator": "saealib.execution.evaluator",
    # execution (evaluation protocol)
    "EvaluationAdapter": "saealib.execution.evaluator",
    "EvaluationQuery": "saealib.execution.evaluator",
    # decomposition
    "Decomposition": "saealib.decomposition",
    "DecompositionComparator": "saealib.decomposition",
    "PBIDecomposition": "saealib.decomposition",
    "TchebycheffDecomposition": "saealib.decomposition",
    "WeightedSumDecomposition": "saealib.decomposition",
    # operators (less common)
    "CrossoverBLXAlpha": "saealib.operators",
    "CrossoverCategorical": "saealib.operators",
    "CrossoverIntegerSBX": "saealib.operators",
    "CrossoverOnePoint": "saealib.operators",
    "OrderCrossover": "saealib.operators",
    "CrossoverTwoPoint": "saealib.operators",
    "CrossoverUniform": "saealib.operators",
    "MutationCategorical": "saealib.operators",
    "MutationGaussian": "saealib.operators",
    "MutationIntegerUniform": "saealib.operators",
    "MutationUniform": "saealib.operators",
    "SequenceMutation": "saealib.operators",
    "SwapMutation": "saealib.operators",
    "PymooCrossover": "saealib.operators",
    "PymooMutation": "saealib.operators",
    "LinearRankSelection": "saealib.operators",
    "SequentialSelection": "saealib.operators",
    "repair_clipping": "saealib.operators",
    # acquisition (less common)
    "CompositeAcquisition": "saealib.acquisition",
    "CORSDistance": "saealib.acquisition",
    "InverseDensityAcquisition": "saealib.acquisition",
    "MaximinDistanceAcquisition": "saealib.acquisition",
    "EHVIAcquisition": "saealib.acquisition",
    "LowerConfidenceBound": "saealib.acquisition",
    "MaxUncertainty": "saealib.acquisition",
    "MeanPrediction": "saealib.acquisition",
    "NoveltyAcquisition": "saealib.acquisition",
    "ParEGOAcquisition": "saealib.acquisition",
    "ProbabilityOfFeasibility": "saealib.acquisition",
    "ProductOfFeasibility": "saealib.acquisition",
    "SMSEGOAcquisition": "saealib.acquisition",
    "WinRateAcquisition": "saealib.acquisition",
    "gp_ucb_beta_schedule": "saealib.acquisition",
    # surrogate (specialized)
    "CompositeSurrogateManager": "saealib.surrogate",
    "GlobalSurrogateManager": "saealib.surrogate",
    "LocalSurrogateManager": "saealib.surrogate",
    "PerObjectiveSurrogate": "saealib.surrogate",
    "PredictionChannel": "saealib.surrogate",
    "RBFSurrogate": "saealib.surrogate",
    "SklearnGPRSurrogate": "saealib.surrogate",
    "SklearnLGBMSurrogate": "saealib.surrogate",
    "SklearnNNSurrogate": "saealib.surrogate",
    "SklearnRFRSurrogate": "saealib.surrogate",
    "SklearnSurrogate": "saealib.surrogate",
    "SklearnSVMSurrogate": "saealib.surrogate",
    "SklearnXGBSurrogate": "saealib.surrogate",
    "SurrogatePrediction": "saealib.surrogate",
    "TorchSurrogate": "saealib.surrogate",
    "product_combine": "saealib.surrogate",
    "rank_weighted_combine": "saealib.surrogate",
    # surrogate (training-set builders)
    "ArchiveObjectiveSet": "saealib.surrogate",
    "ConstraintObjectiveSet": "saealib.surrogate",
    "FeasibilityClassificationSet": "saealib.surrogate",
    "KNNConstraintObjectiveSet": "saealib.surrogate",
    "KNNObjectiveSet": "saealib.surrogate",
    "LevelBasedSet": "saealib.surrogate",
    "PairwiseComparisonSet": "saealib.surrogate",
    "ReferencePointComparisonSet": "saealib.surrogate",
    "TopKBipartitionSet": "saealib.surrogate",
    "TrainingData": "saealib.surrogate",
    "TrainingSet": "saealib.surrogate",
    # surrogate (accuracy evaluation)
    "AccuracyEvaluator": "saealib.surrogate",
    "HeldOutAccuracyEvaluator": "saealib.surrogate",
    "KFoldAccuracyEvaluator": "saealib.surrogate",
    "LOOAccuracyEvaluator": "saealib.surrogate",
    "R2Score": "saealib.surrogate",
    "RMSE": "saealib.surrogate",
    "SpearmanCorrelation": "saealib.surrogate",
    "SurrogateAccuracy": "saealib.surrogate",
    "SurrogateAccuracyMetric": "saealib.surrogate",
    # surrogate (switching)
    "AccuracyBasedSurrogateSwitcher": "saealib.surrogate",
    "GenCtrlSwitcher": "saealib.surrogate",
    "ManagerSwitcher": "saealib.surrogate",
    "StrategySwitcher": "saealib.surrogate",
    # surrogate (other)
    "ComparisonSurrogate": "saealib.surrogate",
    "PairwiseSurrogateManager": "saealib.surrogate",
    "RegressionSurrogate": "saealib.surrogate",
    "SklearnClassificationSurrogate": "saealib.surrogate",
    "SklearnRFCClassificationSurrogate": "saealib.surrogate",
    "SklearnSVCClassificationSurrogate": "saealib.surrogate",
    # problem (less common)
    "GradientRepairHandler": "saealib.problem",
    "PymooProblem": "saealib.problem",
    "exponential_epsilon_schedule": "saealib.problem",
    "linear_epsilon_schedule": "saealib.problem",
    # population (mixins)
    "ArchiveMixin": "saealib.population",
    "ParetoMixin": "saealib.population",
    # callbacks (less common)
    "AcquisitionEndEvent": "saealib.callback",
    "AcquisitionStartEvent": "saealib.callback",
    "PostAskEvent": "saealib.callback",
    "PostCrossoverEvent": "saealib.callback",
    "PostMutationEvent": "saealib.callback",
    "PostSurrogateFitEvent": "saealib.callback",
    "SurrogateEndEvent": "saealib.callback",
    "SurrogateStartEvent": "saealib.callback",
    "logging_generation": "saealib.callback",
    "logging_generation_hv": "saealib.callback",
    # stages (less common)
    "stage_component": "saealib.stages",
    # surrogate (RBF kernels)
    "RBFKernel": "saealib.surrogate.rbf_kernels",
    "GaussianKernel": "saealib.surrogate.rbf_kernels",
    "ThinPlateSplineKernel": "saealib.surrogate.rbf_kernels",
    "LinearKernel": "saealib.surrogate.rbf_kernels",
    "CubicKernel": "saealib.surrogate.rbf_kernels",
    "MultiquadricKernel": "saealib.surrogate.rbf_kernels",
    "MaternKernel": "saealib.surrogate.rbf_kernels",
    "gd": "saealib.utils.indicators",
    "gd_plus": "saealib.utils.indicators",
    "igd": "saealib.utils.indicators",
    "igd_plus": "saealib.utils.indicators",
    "spacing": "saealib.utils.indicators",
    "spread": "saealib.utils.indicators",
    # utils
    "uniform_weight_vectors": "saealib.utils.weight_vectors",
}


def __getattr__(name: str) -> object:
    if name in _LAZY_EXPORTS:
        import importlib

        mod = importlib.import_module(_LAZY_EXPORTS[name])
        obj = getattr(mod, name)
        globals()[name] = obj  # cache to avoid repeated lookup
        return obj
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__ + list(_LAZY_EXPORTS))
