"""Type stubs for saealib's curated top-level package surface.

Covers both the eager root surface and the lazy-loaded root surface. A
subpackage's ``__all__`` is its own namespace contract and is not mirrored
here automatically.
"""

__all__: list[str]
__version__: str

# ---------------------------------------------------------------------------
# Eager root surface
# ---------------------------------------------------------------------------

from saealib.acquisition import AcquisitionFunction as AcquisitionFunction
from saealib.acquisition import AcquisitionResult as AcquisitionResult
from saealib.acquisition import BatchExpectedImprovement as BatchExpectedImprovement

# acquisition (less common)
from saealib.acquisition import CompositeAcquisition as CompositeAcquisition
from saealib.acquisition import CORSDistance as CORSDistance
from saealib.acquisition import EHVIAcquisition as EHVIAcquisition
from saealib.acquisition import ExpectedImprovement as ExpectedImprovement
from saealib.acquisition import InverseDensityAcquisition as InverseDensityAcquisition
from saealib.acquisition import LowerConfidenceBound as LowerConfidenceBound
from saealib.acquisition import MaximinDistanceAcquisition as MaximinDistanceAcquisition
from saealib.acquisition import MaxUncertainty as MaxUncertainty
from saealib.acquisition import MeanPrediction as MeanPrediction
from saealib.acquisition import NoveltyAcquisition as NoveltyAcquisition
from saealib.acquisition import ParEGOAcquisition as ParEGOAcquisition
from saealib.acquisition import PointwiseAcquisition as PointwiseAcquisition
from saealib.acquisition import ProbabilityOfFeasibility as ProbabilityOfFeasibility
from saealib.acquisition import ProductOfFeasibility as ProductOfFeasibility
from saealib.acquisition import SMSEGOAcquisition as SMSEGOAcquisition
from saealib.acquisition import WinRateAcquisition as WinRateAcquisition
from saealib.acquisition import gp_ucb_beta_schedule as gp_ucb_beta_schedule
from saealib.algorithms import GA as GA
from saealib.algorithms import PSO as PSO
from saealib.algorithms import Algorithm as Algorithm
from saealib.algorithms import AskTellAlgorithm as AskTellAlgorithm
from saealib.algorithms import GenomeGA as GenomeGA

# algorithms (less common)
from saealib.algorithms import PymooAlgorithm as PymooAlgorithm
from saealib.api import maximize as maximize
from saealib.api import minimize as minimize
from saealib.callback import AcquisitionEndEvent as AcquisitionEndEvent
from saealib.callback import AcquisitionStartEvent as AcquisitionStartEvent
from saealib.callback import CallbackManager as CallbackManager
from saealib.callback import Event as Event
from saealib.callback import GenerationEndEvent as GenerationEndEvent
from saealib.callback import GenerationStartEvent as GenerationStartEvent
from saealib.callback import InitialEvaluationEndEvent as InitialEvaluationEndEvent
from saealib.callback import InitialEvaluationStartEvent as InitialEvaluationStartEvent
from saealib.callback import PostAskEvent as PostAskEvent
from saealib.callback import PostCrossoverEvent as PostCrossoverEvent
from saealib.callback import PostEvaluationEvent as PostEvaluationEvent
from saealib.callback import PostMutationEvent as PostMutationEvent
from saealib.callback import PostSurrogateFitEvent as PostSurrogateFitEvent
from saealib.callback import RunEndEvent as RunEndEvent
from saealib.callback import RunStartEvent as RunStartEvent
from saealib.callback import SurrogateEndEvent as SurrogateEndEvent
from saealib.callback import SurrogateStartEvent as SurrogateStartEvent

# callbacks (less common)
from saealib.callback import logging_generation as logging_generation
from saealib.callback import logging_generation_hv as logging_generation_hv
from saealib.checkpoint import CheckpointCallback as CheckpointCallback
from saealib.comparators import Comparator as Comparator
from saealib.comparators import Dominator as Dominator
from saealib.comparators import EpsilonDominanceComparator as EpsilonDominanceComparator
from saealib.comparators import EpsilonDominator as EpsilonDominator
from saealib.comparators import HypervolumeComparator as HypervolumeComparator
from saealib.comparators import NonDominatedSorter as NonDominatedSorter
from saealib.comparators import NSGA2Comparator as NSGA2Comparator
from saealib.comparators import NSGA3Comparator as NSGA3Comparator
from saealib.comparators import ParetoComparator as ParetoComparator
from saealib.comparators import ParetoDominator as ParetoDominator
from saealib.comparators import RNSGA2Comparator as RNSGA2Comparator
from saealib.comparators import SingleObjectiveComparator as SingleObjectiveComparator
from saealib.comparators import SPEA2Comparator as SPEA2Comparator
from saealib.comparators import WeightedSumComparator as WeightedSumComparator

# ---------------------------------------------------------------------------
# Lazy root surface (lazy-loaded at runtime via __getattr__)
# ---------------------------------------------------------------------------
# comparators
from saealib.comparators import crowding_distance as crowding_distance
from saealib.comparators import (
    crowding_distance_all_fronts as crowding_distance_all_fronts,
)
from saealib.comparators import dda_non_dominated_sort as dda_non_dominated_sort
from saealib.comparators import non_dominated_sort as non_dominated_sort
from saealib.comparators import spea2_fitness as spea2_fitness

# context
from saealib.context import OptimizationState as OptimizationState

# decomposition
from saealib.decomposition import Decomposition as Decomposition
from saealib.decomposition import DecompositionComparator as DecompositionComparator
from saealib.decomposition import PBIDecomposition as PBIDecomposition
from saealib.decomposition import TchebycheffDecomposition as TchebycheffDecomposition
from saealib.decomposition import WeightedSumDecomposition as WeightedSumDecomposition
from saealib.exceptions import CheckpointError as CheckpointError
from saealib.exceptions import ConfigurationError as ConfigurationError
from saealib.exceptions import EvaluationFatalError as EvaluationFatalError
from saealib.exceptions import EvaluationProtocolError as EvaluationProtocolError
from saealib.exceptions import EvaluationSubmissionError as EvaluationSubmissionError
from saealib.exceptions import RejectedChange as RejectedChange
from saealib.exceptions import ReplacementRejectedError as ReplacementRejectedError
from saealib.exceptions import SaealibError as SaealibError
from saealib.exceptions import ValidationError as ValidationError
from saealib.execution.evaluator import AsyncEvaluator as AsyncEvaluator
from saealib.execution.evaluator import EvaluationAdapter as EvaluationAdapter
from saealib.execution.evaluator import EvaluationErrorInfo as EvaluationErrorInfo
from saealib.execution.evaluator import EvaluationHandle as EvaluationHandle
from saealib.execution.evaluator import EvaluationQuery as EvaluationQuery
from saealib.execution.evaluator import EvaluationRequest as EvaluationRequest
from saealib.execution.evaluator import EvaluationResult as EvaluationResult
from saealib.execution.evaluator import EvaluationStatus as EvaluationStatus
from saealib.execution.evaluator import EvaluationUpdate as EvaluationUpdate
from saealib.execution.evaluator import Evaluator as Evaluator
from saealib.execution.evaluator import JoblibEvaluator as JoblibEvaluator
from saealib.execution.evaluator import PendingEvaluation as PendingEvaluation
from saealib.execution.evaluator import SerialEvaluator as SerialEvaluator
from saealib.execution.evaluator import ThreadPoolEvaluator as ThreadPoolEvaluator
from saealib.execution.initializer import GenomeInitializer as GenomeInitializer
from saealib.execution.initializer import Initializer as Initializer
from saealib.execution.initializer import LHSInitializer as LHSInitializer
from saealib.execution.initializer import RandomInitializer as RandomInitializer
from saealib.execution.initializer import SobolInitializer as SobolInitializer
from saealib.execution.scheduler import (
    AsyncEvaluationScheduler as AsyncEvaluationScheduler,
)
from saealib.islands import IslandModel as IslandModel
from saealib.operators import Crossover as Crossover

# operators (less common)
from saealib.operators import CrossoverBLXAlpha as CrossoverBLXAlpha
from saealib.operators import CrossoverCategorical as CrossoverCategorical
from saealib.operators import CrossoverIntegerSBX as CrossoverIntegerSBX
from saealib.operators import CrossoverOnePoint as CrossoverOnePoint
from saealib.operators import CrossoverSBX as CrossoverSBX
from saealib.operators import CrossoverTwoPoint as CrossoverTwoPoint
from saealib.operators import CrossoverUniform as CrossoverUniform
from saealib.operators import DuplicateElimination as DuplicateElimination
from saealib.operators import LinearRankSelection as LinearRankSelection
from saealib.operators import Mutation as Mutation
from saealib.operators import MutationCategorical as MutationCategorical
from saealib.operators import MutationGaussian as MutationGaussian
from saealib.operators import MutationIntegerUniform as MutationIntegerUniform
from saealib.operators import MutationPolynomial as MutationPolynomial
from saealib.operators import MutationUniform as MutationUniform
from saealib.operators import OrderCrossover as OrderCrossover
from saealib.operators import ParentSelection as ParentSelection
from saealib.operators import PymooCrossover as PymooCrossover
from saealib.operators import PymooMutation as PymooMutation
from saealib.operators import SequenceMutation as SequenceMutation
from saealib.operators import SequentialSelection as SequentialSelection
from saealib.operators import SurvivorSelection as SurvivorSelection
from saealib.operators import SwapMutation as SwapMutation
from saealib.operators import TournamentSelection as TournamentSelection
from saealib.operators import TruncationSelection as TruncationSelection
from saealib.operators import repair_clipping as repair_clipping
from saealib.optimizer import Optimizer as Optimizer
from saealib.pipeline import Branch as Branch
from saealib.pipeline import Condition as Condition
from saealib.pipeline import Loop as Loop
from saealib.pipeline import Pipeline as Pipeline
from saealib.pipeline import PipelineEntry as PipelineEntry
from saealib.pipeline import Repeat as Repeat
from saealib.pipeline import Stage as Stage
from saealib.policies import ComparatorWorstFallback as ComparatorWorstFallback
from saealib.policies import EvaluateAll as EvaluateAll
from saealib.policies import EvaluationPlan as EvaluationPlan
from saealib.policies import EvaluationPlanner as EvaluationPlanner
from saealib.policies import FeedbackBuilder as FeedbackBuilder
from saealib.policies import FeedbackResult as FeedbackResult
from saealib.policies import FidelityEvaluation as FidelityEvaluation
from saealib.policies import FidelityPromotion as FidelityPromotion
from saealib.policies import MixedFeedback as MixedFeedback
from saealib.policies import NoFeedback as NoFeedback
from saealib.policies import PredictedFeedback as PredictedFeedback
from saealib.policies import RatioEvaluation as RatioEvaluation
from saealib.policies import RepeatedEvaluation as RepeatedEvaluation
from saealib.policies import ReplicateSummary as ReplicateSummary
from saealib.policies import TopKEvaluation as TopKEvaluation
from saealib.policies import TrueOnlyFeedback as TrueOnlyFeedback
from saealib.policies import aggregate_replicates as aggregate_replicates
from saealib.population import Archive as Archive

# population (mixins)
from saealib.population import ArchiveMixin as ArchiveMixin
from saealib.population import Individual as Individual
from saealib.population import ParetoArchive as ParetoArchive
from saealib.population import ParetoMixin as ParetoMixin
from saealib.population import Population as Population
from saealib.population import PopulationAttribute as PopulationAttribute

# problem (less common)
from saealib.problem import ConstraintHandler as ConstraintHandler
from saealib.problem import EpsilonConstraintHandler as EpsilonConstraintHandler
from saealib.problem import EqualityConstraint as EqualityConstraint
from saealib.problem import GradientRepairHandler as GradientRepairHandler
from saealib.problem import InequalityConstraint as InequalityConstraint
from saealib.problem import Problem as Problem
from saealib.problem import PymooProblem as PymooProblem
from saealib.problem import StaticToleranceHandler as StaticToleranceHandler
from saealib.problem import exponential_epsilon_schedule as exponential_epsilon_schedule
from saealib.problem import linear_epsilon_schedule as linear_epsilon_schedule
from saealib.registry import register as register
from saealib.result import Result as Result

# stages
from saealib.stages import AcquisitionStage as AcquisitionStage
from saealib.stages import ArchiveUpdateStage as ArchiveUpdateStage
from saealib.stages import AskStage as AskStage
from saealib.stages import AsyncEvaluationSubmitStage as AsyncEvaluationSubmitStage
from saealib.stages import CountGenerationStage as CountGenerationStage
from saealib.stages import EvaluationAcknowledgeStage as EvaluationAcknowledgeStage
from saealib.stages import EvaluationApplyStage as EvaluationApplyStage
from saealib.stages import EvaluationCollectStage as EvaluationCollectStage
from saealib.stages import EvaluationPlanStage as EvaluationPlanStage
from saealib.stages import EvaluationSubmitStage as EvaluationSubmitStage
from saealib.stages import FeedbackStage as FeedbackStage
from saealib.stages import InitializationStage as InitializationStage
from saealib.stages import SurrogateFitStage as SurrogateFitStage
from saealib.stages import SurrogateOnlyLoopStage as SurrogateOnlyLoopStage
from saealib.stages import SurrogatePredictStage as SurrogatePredictStage
from saealib.stages import TellStage as TellStage
from saealib.stages import stage_component as stage_component
from saealib.strategies import DirectStrategy as DirectStrategy
from saealib.strategies import GenerationBasedStrategy as GenerationBasedStrategy
from saealib.strategies import IndividualBasedStrategy as IndividualBasedStrategy
from saealib.strategies import OptimizationStrategy as OptimizationStrategy
from saealib.strategies import PreSelectionStrategy as PreSelectionStrategy
from saealib.strategies import SteadyStateStrategy as SteadyStateStrategy
from saealib.surrogate import RMSE as RMSE

# surrogate (switching)
from saealib.surrogate import (
    AccuracyBasedSurrogateSwitcher as AccuracyBasedSurrogateSwitcher,
)

# surrogate (accuracy evaluation)
from saealib.surrogate import AccuracyEvaluator as AccuracyEvaluator

# surrogate (training-set builders)
from saealib.surrogate import ArchiveObjectiveSet as ArchiveObjectiveSet

# surrogate (other)
from saealib.surrogate import ComparisonSurrogate as ComparisonSurrogate
from saealib.surrogate import CompositeSurrogateManager as CompositeSurrogateManager
from saealib.surrogate import ConstraintObjectiveSet as ConstraintObjectiveSet
from saealib.surrogate import (
    FeasibilityClassificationSet as FeasibilityClassificationSet,
)
from saealib.surrogate import GenCtrlSwitcher as GenCtrlSwitcher
from saealib.surrogate import GlobalSurrogateManager as GlobalSurrogateManager
from saealib.surrogate import HeldOutAccuracyEvaluator as HeldOutAccuracyEvaluator
from saealib.surrogate import KFoldAccuracyEvaluator as KFoldAccuracyEvaluator
from saealib.surrogate import KNNConstraintObjectiveSet as KNNConstraintObjectiveSet
from saealib.surrogate import KNNObjectiveSet as KNNObjectiveSet
from saealib.surrogate import LevelBasedSet as LevelBasedSet
from saealib.surrogate import LocalSurrogateManager as LocalSurrogateManager
from saealib.surrogate import LOOAccuracyEvaluator as LOOAccuracyEvaluator
from saealib.surrogate import ManagerSwitcher as ManagerSwitcher
from saealib.surrogate import PairwiseComparisonSet as PairwiseComparisonSet
from saealib.surrogate import PairwiseSurrogateManager as PairwiseSurrogateManager
from saealib.surrogate import PerObjectiveSurrogate as PerObjectiveSurrogate
from saealib.surrogate import PredictionChannel as PredictionChannel
from saealib.surrogate import R2Score as R2Score
from saealib.surrogate import RBFKernel as RBFKernel
from saealib.surrogate import RBFSurrogate as RBFSurrogate
from saealib.surrogate import ReferencePointComparisonSet as ReferencePointComparisonSet
from saealib.surrogate import RegressionSurrogate as RegressionSurrogate
from saealib.surrogate import (
    SklearnClassificationSurrogate as SklearnClassificationSurrogate,
)
from saealib.surrogate import SklearnGPRSurrogate as SklearnGPRSurrogate
from saealib.surrogate import SklearnLGBMSurrogate as SklearnLGBMSurrogate
from saealib.surrogate import SklearnNNSurrogate as SklearnNNSurrogate
from saealib.surrogate import (
    SklearnRFCClassificationSurrogate as SklearnRFCClassificationSurrogate,
)
from saealib.surrogate import SklearnRFRSurrogate as SklearnRFRSurrogate
from saealib.surrogate import SklearnSurrogate as SklearnSurrogate
from saealib.surrogate import (
    SklearnSVCClassificationSurrogate as SklearnSVCClassificationSurrogate,
)
from saealib.surrogate import SklearnSVMSurrogate as SklearnSVMSurrogate
from saealib.surrogate import SklearnXGBSurrogate as SklearnXGBSurrogate
from saealib.surrogate import SpearmanCorrelation as SpearmanCorrelation
from saealib.surrogate import StrategySwitcher as StrategySwitcher
from saealib.surrogate import Surrogate as Surrogate
from saealib.surrogate import SurrogateAccuracy as SurrogateAccuracy
from saealib.surrogate import SurrogateAccuracyMetric as SurrogateAccuracyMetric
from saealib.surrogate import SurrogateManager as SurrogateManager
from saealib.surrogate import SurrogatePrediction as SurrogatePrediction
from saealib.surrogate import TopKBipartitionSet as TopKBipartitionSet
from saealib.surrogate import TorchSurrogate as TorchSurrogate
from saealib.surrogate import TrainingData as TrainingData
from saealib.surrogate import TrainingSet as TrainingSet
from saealib.surrogate import product_combine as product_combine
from saealib.surrogate import rank_weighted_combine as rank_weighted_combine
from saealib.surrogate.rbf_kernels import CubicKernel as CubicKernel
from saealib.surrogate.rbf_kernels import GaussianKernel as GaussianKernel
from saealib.surrogate.rbf_kernels import LinearKernel as LinearKernel
from saealib.surrogate.rbf_kernels import MaternKernel as MaternKernel
from saealib.surrogate.rbf_kernels import MultiquadricKernel as MultiquadricKernel
from saealib.surrogate.rbf_kernels import (
    ThinPlateSplineKernel as ThinPlateSplineKernel,
)
from saealib.termination import Termination as Termination
from saealib.termination import TerminationCondition as TerminationCondition
from saealib.termination import f_target as f_target
from saealib.termination import max_fe as max_fe
from saealib.termination import max_gen as max_gen
from saealib.termination import max_steps as max_steps
from saealib.termination import stalled as stalled
from saealib.transition import Chain as Chain
from saealib.transition import Transition as Transition
from saealib.transition import TransitionContext as TransitionContext
from saealib.transition import chain as chain
from saealib.utils.indicators import hypervolume as hypervolume
from saealib.utils.indicators import (
    hypervolume_contributions as hypervolume_contributions,
)
from saealib.utils.weight_vectors import (
    uniform_weight_vectors as uniform_weight_vectors,
)
from saealib.variables import CategoricalVariable as CategoricalVariable
from saealib.variables import ContinuousVariable as ContinuousVariable
from saealib.variables import IntegerVariable as IntegerVariable
from saealib.variables import Variable as Variable

_LAZY_EXPORTS: dict[str, str]
