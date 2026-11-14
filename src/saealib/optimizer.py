"""Optimizer: assembles and runs the surrogate-assisted EA pipeline."""

from __future__ import annotations

import copy
import dataclasses
import importlib.util
import logging
import pickle
import warnings
from collections.abc import Callable, Generator, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

from typing_extensions import Self

from saealib.acquisition.base import AcquisitionFunction, CompositeAcquisition
from saealib.acquisition.mean import MeanPrediction
from saealib.acquisition.winrate import WinRateAcquisition
from saealib.callback import (
    CallbackManager,
    Event,
    GenerationStartEvent,
    logging_generation,
)
from saealib.context import (
    OptimizationState,
    _validate_checkpoint_pending,
    _validate_checkpoint_runtime,
    _validate_pending_reattachable,
)
from saealib.core.compiler import (
    ContractPath,
    Diagnostic,
    DiagnosticBag,
    ExecutablePlan,
    Severity,
    check_component_contract,
)
from saealib.core.compiler.contract_diagnostics import (
    _optimizer_configuration_diagnostics,
    _strategy_graph_diagnostic,
    check_pymoo_feedback_compatibility,
)
from saealib.core.contracts import ComponentContract
from saealib.core.runtime import ReplaceComponent
from saealib.core.state import OPTIMIZATION_STATE_INITIAL_KEYS
from saealib.exceptions import CheckpointError, ConfigurationError, ValidationError
from saealib.execution.bindings import (
    BINDING_SLOTS,
    _comparator_signature,
    _ComponentOrigin,
    _ComponentResolution,
    _feature_encoder_signature,
    _signature,
    resolve_components,
)
from saealib.execution.evaluator import Evaluator, SerialEvaluator
from saealib.execution.history import SUPPORTED_HISTORY_CHANNELS
from saealib.execution.prepare import (
    _graph_requires_service,
    _InvalidStrategyGraphError,
    _prepare_plan,
)
from saealib.execution.runner import Runner
from saealib.execution.runtime import (
    _BindingSnapshot,
    _non_checkpointable_components,
    default_runtime_registry,
)
from saealib.execution.scheduler import AsyncEvaluationScheduler
from saealib.policies.evaluation import EvaluationPlanner
from saealib.policies.feedback import FeedbackBuilder
from saealib.surrogate.manager import (
    LocalSurrogateManager,
    PairwiseSurrogateManager,
    SurrogateManager,
)
from saealib.surrogate.rbf_kernels import GaussianKernel
from saealib.termination import Termination
from saealib.transition import Transition

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from saealib.algorithms.base import Algorithm
    from saealib.comparators import Comparator
    from saealib.defaults.model import DefaultResolution
    from saealib.execution.initializer import Initializer
    from saealib.problem import Problem
    from saealib.strategies.base import OptimizationStrategy
    from saealib.surrogate.base import Surrogate


def _default_acquisition_spec(manager_type: str | type | None, direction) -> dict:
    from saealib.registry import get

    if isinstance(manager_type, type):
        manager_cls = manager_type
    else:
        try:
            manager_cls = get(manager_type) if manager_type is not None else None
        except ValidationError:
            manager_cls = None
    if isinstance(manager_cls, type) and issubclass(
        manager_cls, PairwiseSurrogateManager
    ):
        return {"type": "WinRateAcquisition", "params": {}}
    return {"type": "MeanPrediction", "params": {"direction": direction}}


def _restored_problem_comparator(problem: object) -> object | None:
    """Return the comparator a restored Problem kept as an explicit setting.

    A restored Problem without a ``_comparator_explicit`` attribute carries no
    record of the source of its comparator, so the stored comparator is
    compared with the one this Problem builds by default: a different
    specification marks an explicit comparator that belongs in the optimizer
    slot.
    """
    if problem is None or "_comparator_explicit" in vars(problem):
        return None
    stored = getattr(problem, "comparator", None)
    if stored is None:
        return None
    from saealib.comparators import default_comparator

    try:
        expected = default_comparator(
            int(getattr(problem, "n_obj")),
            getattr(problem, "direction", None),
            eps_cv=float(getattr(problem, "eps_cv", 0.0)),
            eps_obj=float(getattr(problem, "eps_obj", 1e-6)),
        )
    except Exception:
        return stored
    if _signature(stored) == _signature(expected):
        return None
    return stored


class Dispatchable(Protocol):
    """Minimal interface for objects that can dispatch callback events."""

    def dispatch(self, event: Event) -> None:
        """Dispatch a callback event."""
        ...


class ComponentProvider(Protocol):
    """The interface for components that can be used by the Optimizer."""

    @property
    def seed(self) -> int | None:
        """Return the master random seed."""
        ...

    @property
    def algorithm(self) -> Algorithm:
        """Return the algorithm instance."""
        ...

    @property
    def strategy(self) -> OptimizationStrategy:
        """Return the optimization strategy instance."""
        ...

    @property
    def surrogate_manager(self) -> SurrogateManager:
        """Return the surrogate manager instance."""
        ...

    @property
    def acquisition(self) -> AcquisitionFunction:
        """Acquisition function owned by the provider, not the manager."""
        ...

    @property
    def evaluation_planner(self) -> EvaluationPlanner | None:
        """Return the evaluation planner."""
        ...

    @property
    def feedback_builder(self) -> FeedbackBuilder | None:
        """Return the configured feedback builder."""
        ...

    @property
    def async_evaluation_scheduler(self) -> AsyncEvaluationScheduler | None:
        """Return the optional asynchronous evaluation scheduler."""
        ...

    @property
    def history_channels(self) -> Sequence[str]:
        """Return the enabled execution history channels."""
        ...

    @property
    def feedback_builder_explicit(self) -> bool:
        """Return whether the feedback builder was explicitly configured."""
        ...

    @property
    def history_channels_explicit(self) -> bool:
        """Return whether history channels were explicitly configured."""
        ...

    @property
    def evaluator(self) -> Evaluator:
        """Return the evaluator instance."""
        ...

    @property
    def termination(self) -> Termination:
        """Return the termination condition."""
        ...

    @property
    def cbmanager(self) -> CallbackManager:
        """Return the callback manager."""
        ...

    def dispatch(self, event: Event) -> None:
        """Dispatch a callback event."""
        ...


class Optimizer:
    """
    Optimizer class for evolutionary algorithms.

    Integrates problem definition, evolutionary algorithm, surrogate model,
    model manager, and termination condition, and manages the optimization process.

    Attributes
    ----------
    problem : Problem
        The optimization problem.
    algorithm : Algorithm
        The evolutionary algorithm.
    surrogate : Surrogate
        The surrogate model.
    strategy : OptimizationStrategy
        The optimization strategy.
    termination : Termination
        The termination condition.
    archive : Archive
        The archive of evaluated solutions.
    popsize : int
        The population size.
    seed : int or None
        The master random seed.  ``None`` means non-deterministic.
    fe : int
        The current number of function evaluations.
    gen : int
        The current generation number.
    cbmanager : CallbackManager
        The callback event manager.
    instance_name : str
        The name of the optimizer instance.
    """

    _COMPONENT_ASSIGNMENTS = frozenset((*BINDING_SLOTS, "initializer", "termination"))

    if TYPE_CHECKING:
        algorithm: Algorithm
        strategy: OptimizationStrategy
        surrogate_manager: SurrogateManager
        termination: Termination

    def __setattr__(self, name: str, value: object) -> None:
        tracking = self.__dict__.get("_track_component_assignments", False)
        prior_value = getattr(self, name, None) if tracking else None
        object.__setattr__(self, name, value)
        if tracking and name in self._COMPONENT_ASSIGNMENTS:
            origins = self.__dict__.setdefault("_component_origins", {})
            origins[name] = _ComponentOrigin(
                value,
                "direct assignment",
                "Assigned directly to the optimizer.",
                "",
            )
            self.__dict__.setdefault("_materialized_components", {}).pop(name, None)
            if prior_value is not value:
                object.__setattr__(
                    self,
                    "_configuration_version",
                    self.__dict__.get("_configuration_version", 0) + 1,
                )
            if name == "feedback_builder" and value is not None:
                object.__setattr__(self, "feedback_builder_explicit", True)

    def _set_component(
        self,
        name: str,
        value: object,
        reason: str,
        transition: Transition | None = None,
    ) -> None:
        if transition is not None and not isinstance(transition, Transition):
            raise ValidationError("transition must be a Transition or None")
        prior_value = getattr(self, name, None)
        object.__setattr__(self, name, value)
        self._component_origins[name] = _ComponentOrigin(value, "setter", reason, "")
        self._materialized_components.pop(name, None)
        if transition is None:
            self._replacement_records.pop(name, None)
        else:
            self._replacement_records[name] = ReplaceComponent(
                slot=name,
                component=value,
                transition=transition,
                source="setter",
            )
        if prior_value is not value:
            self._configuration_version += 1

    def _set_framework_component(self, name: str, value: object, reason: str) -> None:
        object.__setattr__(self, name, value)
        self._component_origins[name] = _ComponentOrigin(
            value, "framework default", reason, ""
        )
        self._materialized_components.pop(name, None)

    def _pending_replacements(self) -> dict[str, ReplaceComponent]:
        """Return the recorded replacements no commit point consumed yet."""
        return dict(self._replacement_records)

    def _consume_replacements(self) -> dict[str, ReplaceComponent]:
        """Take the recorded replacements and leave the queue empty."""
        records = dict(self._replacement_records)
        self._replacement_records.clear()
        return records

    def __init__(self, problem: Problem, seed: int | None = None):
        """
        Initialize the Optimizer.

        Parameters
        ----------
        problem : Problem
            The optimization problem.
        seed : int or None, optional
            Master random seed propagated to the initializer.  ``None`` (default)
            means non-deterministic.
        """
        self.problem: Problem = problem
        self.seed: int | None = seed
        self.cbmanager: CallbackManager = CallbackManager()
        self.cbmanager.register(GenerationStartEvent, logging_generation)
        self.initializer: Initializer | None = None
        self.evaluator: Evaluator = SerialEvaluator()
        self.acquisition: AcquisitionFunction = cast(AcquisitionFunction, None)
        self._evaluation_planner: EvaluationPlanner | None = None
        self.feedback_builder: FeedbackBuilder | None = None
        self.feedback_builder_explicit = False
        self.history_channels_explicit = False
        self.async_evaluation_scheduler: AsyncEvaluationScheduler | None = None
        self.history_channels: tuple[str, ...] = ("summary",)
        self.instance_name: str = ""
        self._preset: dict | None = None
        self._default_resolution: DefaultResolution | None = None
        self._component_resolution: _ComponentResolution | None = None
        self._component_origins: dict[str, _ComponentOrigin] = {}
        self._materialized_components: dict[str, object] = {}
        self._feature_encoder_default: tuple[object, object, str] | None = None
        self._configuration_version = 0
        self._replacement_records: dict[str, ReplaceComponent] = {}
        self._last_contract_diagnostics: tuple[Diagnostic, ...] = ()
        self._executable_plan: ExecutablePlan | None = None
        self._last_published_bindings: _BindingSnapshot | None = None
        self._prepared_diagnostic_cache: (
            tuple[tuple[object, ...], tuple[Diagnostic, ...]] | None
        ) = None
        self._verification_enabled = True
        self._portable_checkpointing = False
        self._component_origins["evaluator"] = _ComponentOrigin(
            self.evaluator,
            "bundled preset",
            "Uses serial objective evaluation.",
            _signature(self.evaluator),
        )
        self._component_origins["cbmanager"] = _ComponentOrigin(
            self.cbmanager,
            "bundled preset",
            "Uses the optimizer callback manager.",
            _signature(self.cbmanager),
        )
        self.comparator: Comparator | None = None
        self._comparator_default: tuple[Problem, object] | None = None
        if getattr(problem, "_comparator_explicit", False):
            self.comparator = problem.comparator
            self._component_origins["comparator"] = _ComponentOrigin(
                self.comparator,
                "deprecated Problem argument",
                "Adopted from the deprecated Problem(comparator=...) argument.",
                _comparator_signature(self.comparator),
            )
        self._track_component_assignments = True

    # --- setters (all return self for chaining) ---

    def contract_diagnostics(self) -> DiagnosticBag:
        """Return diagnostics for the prepared optimizer composition."""
        return self._collect_contract_diagnostics(require_initializer=True)

    def _collect_contract_diagnostics(
        self, *, require_initializer: bool
    ) -> DiagnosticBag:
        resolution = resolve_components(self)
        diagnostics = DiagnosticBag()
        contracts: dict[str, ComponentContract] = {}
        provenance = {
            slot: (origin.source, origin.reason)
            for slot, origin in resolution.origins.items()
        }
        component_names = (
            "initializer",
            "algorithm",
            "surrogate_manager",
            "termination",
            "evaluator",
            "evaluation_planner",
            "feedback_builder",
            "acquisition",
            "async_evaluation_scheduler",
            "strategy",
        )
        for name in component_names:
            component = (
                getattr(self, name, None)
                if name == "async_evaluation_scheduler"
                else resolution.components.get(name)
            )
            if component is None:
                continue
            missing = object()
            try:
                contract_method = getattr(component, "contract", missing)
            except Exception as error:
                diagnostics.append(
                    Diagnostic(
                        severity=Severity.ERROR,
                        code="contract_unavailable",
                        message=(
                            f"{name}.contract attribute could not be read: "
                            f"{type(error).__name__}: {error}."
                        ),
                        path=ContractPath(components=(name,)),
                        resolutions=(
                            "Make the component's contract attribute readable and "
                            "provide a callable contract() method.",
                        ),
                    )
                )
                continue
            if contract_method is missing:
                continue
            if not callable(contract_method):
                diagnostics.append(
                    Diagnostic(
                        severity=Severity.ERROR,
                        code="contract_unavailable",
                        message=(
                            f"{name}.contract exists on {type(component).__name__} "
                            "but is not callable."
                        ),
                        path=ContractPath(components=(name,)),
                        resolutions=(
                            "Provide a callable contract() method returning "
                            "ComponentContract.",
                        ),
                    )
                )
                continue
            try:
                contract = cast(Callable[[], object], contract_method)()
            except Exception as error:
                diagnostics.append(
                    Diagnostic(
                        severity=Severity.ERROR,
                        code="contract_unavailable",
                        message=(
                            f"{name}.contract() on {type(component).__name__} "
                            f"raised {type(error).__name__}: {error}."
                        ),
                        path=ContractPath(components=(name,)),
                        resolutions=(
                            "Make contract() return a ComponentContract without "
                            "raising.",
                        ),
                    )
                )
                continue
            if not isinstance(contract, ComponentContract):
                diagnostics.append(
                    Diagnostic(
                        severity=Severity.ERROR,
                        code="contract_unavailable",
                        message=(
                            f"{name}.contract() on {type(component).__name__} "
                            f"returned {type(contract).__name__}, not "
                            "ComponentContract."
                        ),
                        path=ContractPath(components=(name,)),
                        resolutions=(
                            "Make contract() return a ComponentContract instance.",
                        ),
                    )
                )
                continue
            contracts[name] = contract
            diagnostics.extend(
                check_component_contract(
                    contract,
                    component=component,
                    path=ContractPath(components=(name,)),
                )
            )
        diagnostics.extend(
            _optimizer_configuration_diagnostics(
                resolution.components,
                provenance,
                self.problem,
                resolution.components.get("evaluator", self.evaluator),
                require_initializer=require_initializer,
            )
        )
        graph_blocked = any(
            diagnostic.severity is Severity.ERROR
            and (
                diagnostic.code
                in {"contract_unavailable", "component_protocol_mismatch"}
                or diagnostic.code
                in {"missing_surrogate_manager", "missing_acquisition"}
                or (
                    diagnostic.code == "missing_required_component"
                    and diagnostic.path.components
                    in {
                        ("optimizer", "algorithm"),
                        ("optimizer", "strategy"),
                    }
                )
            )
            for diagnostic in diagnostics
        )
        if not graph_blocked:
            strategy = resolution.components.get("strategy")
            if strategy is not None:
                try:
                    if not callable(getattr(strategy, "build_graph", None)):
                        raise TypeError(
                            f"{type(strategy).__name__} does not provide a callable "
                            "build_graph() operation"
                        )
                    cached = self._prepared_diagnostic_cache
                    signature_resolution = resolution
                    cached_default = self._feature_encoder_default
                    if (
                        cached is not None
                        and resolution.feature_encoder_origin is None
                        and isinstance(cached_default, tuple)
                        and len(cached_default) == 3
                        and cached_default[0] is self.problem.space
                        and cached_default[1] is not None
                        and self._executable_plan is not None
                        and _graph_requires_service(
                            self._executable_plan.graph, "FeatureEncoder"
                        )
                    ):
                        encoder, reason = cached_default[1], cached_default[2]
                        signature_resolution = dataclasses.replace(
                            resolution,
                            feature_encoder=encoder,
                            feature_encoder_origin=_ComponentOrigin(
                                encoder,
                                "framework default",
                                reason,
                                f"{id(self.problem.space)}:{_feature_encoder_signature(encoder)}",
                            ),
                            feature_encoder_space=self.problem.space,
                        )
                    signature = self._diagnostic_plan_signature(
                        signature_resolution,
                        require_initializer=require_initializer,
                    )
                    if cached is not None and cached[0] == signature:
                        plan_diagnostics = cached[1]
                        self._prepared_diagnostic_cache = None
                    else:
                        validation_strategy = copy.copy(strategy)
                        validation_provider = copy.copy(self)
                        validation_provider._component_origins = (
                            self._component_origins.copy()
                        )
                        validation_provider._materialized_components = (
                            self._materialized_components.copy()
                        )
                        object.__setattr__(
                            validation_provider, "strategy", validation_strategy
                        )
                        plan = _prepare_plan(
                            validation_strategy,
                            validation_provider,
                            problem=None,
                            offered_runtime_capabilities=None,
                            _context_factory=lambda: (
                                getattr(validation_provider, "problem", None),
                                default_runtime_registry.offered_capabilities(
                                    validation_provider
                                ),
                            ),
                            initial_state_keys=OPTIMIZATION_STATE_INITIAL_KEYS,
                            _verification_enabled=self._verification_enabled,
                            _component_resolution=resolution,
                            _require_initializer=require_initializer,
                        )
                        plan_diagnostics = plan.diagnostics
                    diagnostics.extend(plan_diagnostics)
                except Exception as error:
                    diagnostics.append(
                        _strategy_graph_diagnostic(strategy, provenance, error)
                    )
        algorithm = resolution.components.get("algorithm")
        scheduler = self.async_evaluation_scheduler
        from saealib.algorithms.pymoo_algorithm import PymooAlgorithm

        if (
            isinstance(algorithm, PymooAlgorithm)
            and not getattr(algorithm, "allow_partial_tell", False)
            and scheduler is not None
        ):
            algorithm_contract = contracts.get("algorithm")
            if algorithm_contract is not None:
                diagnostics.extend(
                    check_pymoo_feedback_compatibility(
                        algorithm_contract,
                        consumer_path=ContractPath(components=("algorithm",)),
                        runtime_path=ContractPath(
                            components=("async_evaluation_scheduler",)
                        ),
                    )
                )
        return DiagnosticBag(dict.fromkeys(diagnostics))

    def _diagnostic_plan_signature(
        self,
        resolution: _ComponentResolution,
        *,
        require_initializer: bool = True,
    ) -> tuple[object, ...]:
        return (
            self._configuration_version,
            tuple(
                (
                    name,
                    _comparator_signature(value)
                    if name == "comparator"
                    else _signature(value),
                )
                for name, value in resolution.components.items()
            ),
            tuple(
                (name, origin.source, origin.reason)
                for name, origin in resolution.origins.items()
            ),
            (
                None
                if resolution.feature_encoder_origin is None
                else (
                    resolution.feature_encoder_origin.source,
                    resolution.feature_encoder_origin.reason,
                    resolution.feature_encoder_origin.signature,
                )
            ),
            repr(self.problem.direction),
            self._verification_enabled,
            require_initializer,
        )

    @property
    def last_contract_diagnostics(self) -> tuple[Diagnostic, ...]:
        """Return the diagnostics collected by the most recent ``validate``."""
        return self._last_contract_diagnostics

    @property
    def executable_plan(self) -> ExecutablePlan | None:
        """Return the plan produced by the most recent execution preparation."""
        return self._executable_plan

    def describe(self) -> str:
        """Describe the most recently compiled plan, if one exists."""
        if self._executable_plan is None:
            return "Optimizer(uncompiled)"
        return self._executable_plan.describe()

    def _compile_plan(
        self, *, require_initializer: bool = True
    ) -> ExecutablePlan | None:
        """Build and compile the configured strategy graph once per run."""
        self.resolve_defaults()
        self._prepared_diagnostic_cache = None
        strategy = getattr(self, "strategy", None)
        if not callable(getattr(strategy, "build_graph", None)):
            self._executable_plan = None
            return None
        for _ in range(8):
            configuration_version = self._configuration_version
            resolution = cast(_ComponentResolution, self._component_resolution)
            try:
                plan = _prepare_plan(
                    strategy,
                    self,
                    problem=None,
                    offered_runtime_capabilities=None,
                    _context_factory=lambda: (
                        getattr(self, "problem", None),
                        default_runtime_registry.offered_capabilities(self),
                    ),
                    initial_state_keys=OPTIMIZATION_STATE_INITIAL_KEYS,
                    _verification_enabled=getattr(self, "_verification_enabled", True),
                    _component_resolution=resolution,
                    _require_initializer=require_initializer,
                )
            except _InvalidStrategyGraphError:
                self._executable_plan = None
                return None
            if configuration_version == self._configuration_version:
                resolution = cast(_ComponentResolution, self._component_resolution)
                consumes_feature_encoder = _graph_requires_service(
                    plan.graph, "FeatureEncoder"
                )
                self._publish_component_resolution(
                    resolution,
                    feature_encoder_consumed=consumes_feature_encoder,
                )
                self._executable_plan = plan
                self._prepared_diagnostic_cache = (
                    self._diagnostic_plan_signature(
                        resolution, require_initializer=require_initializer
                    ),
                    plan.diagnostics,
                )
                return plan
            self.resolve_defaults()
            strategy = getattr(self, "strategy", None)
        raise ConfigurationError("component bindings changed during plan preparation")

    def set_seed(self, seed: int | None) -> Self:
        """Set the master random seed. Returns self."""
        self.seed = seed
        return self

    def set_history(self, channels: Sequence[str]) -> Self:
        """Set execution history channels and return self.

        On resume, the checkpoint's channels take precedence unless this method
        was called; calling it with a different set fails validation. Assigning
        to ``history_channels`` directly does not count as calling it.
        """
        requested = tuple(channels)
        unknown = sorted(set(requested) - SUPPORTED_HISTORY_CHANNELS)
        if unknown:
            raise ValidationError(f"Unknown history channel(s): {', '.join(unknown)}")
        self.history_channels = requested
        self.history_channels_explicit = True
        return self

    def set_comparator(
        self, comparator: Comparator, *, transition: Transition | None = None
    ) -> Self:
        """Set the comparator that orders solutions. Returns self.

        Parameters
        ----------
        comparator : Comparator
            The component to bind.
        transition : Transition or None, optional
            Transition to run at the commit point of this replacement, after
            the candidate configuration passes static validation and before
            the new binding version is published.
        """
        self._set_component(
            "comparator",
            comparator,
            "Configured with set_comparator().",
            transition=transition,
        )
        return self

    def set_initializer(self, initializer: Initializer) -> Self:
        """Set the initializer. Returns self."""
        self._set_component(
            "initializer", initializer, "Configured with set_initializer()."
        )
        return self

    def set_algorithm(
        self, algorithm: Algorithm, *, transition: Transition | None = None
    ) -> Self:
        """Set the evolutionary algorithm. Returns self.

        Parameters
        ----------
        algorithm : Algorithm
            The component to bind.
        transition : Transition or None, optional
            Transition to run at the commit point of this replacement, after
            the candidate configuration passes static validation and before
            the new binding version is published.
        """
        self._set_component(
            "algorithm",
            algorithm,
            "Configured with set_algorithm().",
            transition=transition,
        )
        return self

    def set_surrogate_manager(
        self, manager: SurrogateManager, *, transition: Transition | None = None
    ) -> Self:
        """Set the surrogate manager. Returns self.

        Parameters
        ----------
        manager : SurrogateManager
            The component to bind.
        transition : Transition or None, optional
            Transition to run at the commit point of this replacement.
        """
        self._set_component(
            "surrogate_manager",
            manager,
            "Configured with set_surrogate_manager().",
            transition=transition,
        )
        if isinstance(manager, PairwiseSurrogateManager) and self.acquisition is None:
            self._set_framework_component(
                "acquisition",
                WinRateAcquisition(),
                "Uses the pairwise manager's win-rate acquisition.",
            )
        return self

    def set_acquisition(
        self, acquisition, *, transition: Transition | None = None
    ) -> Self:
        """Set the independent acquisition component.

        Parameters
        ----------
        acquisition : AcquisitionFunction
            The component to bind.
        transition : Transition or None, optional
            Transition to run at the commit point of this replacement.
        """
        self._set_component(
            "acquisition",
            acquisition,
            "Configured with set_acquisition().",
            transition=transition,
        )
        return self

    @property
    def evaluation_planner(self) -> EvaluationPlanner | None:
        """Return the configured evaluation planner."""
        return self._evaluation_planner

    @evaluation_planner.setter
    def evaluation_planner(self, planner: EvaluationPlanner | None) -> None:
        self._evaluation_planner = planner

    def set_evaluation_planner(
        self, planner: EvaluationPlanner, *, transition: Transition | None = None
    ) -> Self:
        """Set the evaluation planner.

        Parameters
        ----------
        planner : EvaluationPlanner
            The component to bind.
        transition : Transition or None, optional
            Transition to run at the commit point of this replacement.
        """
        self._set_component(
            "evaluation_planner",
            planner,
            "Configured with set_evaluation_planner().",
            transition=transition,
        )
        return self

    def set_feedback_builder(
        self, builder: FeedbackBuilder, *, transition: Transition | None = None
    ) -> Self:
        """Set the feedback builder.

        Parameters
        ----------
        builder : FeedbackBuilder
            The component to bind.
        transition : Transition or None, optional
            Transition to run at the commit point of this replacement.
        """
        self._set_component(
            "feedback_builder",
            builder,
            "Configured with set_feedback_builder().",
            transition=transition,
        )
        self.feedback_builder_explicit = True
        return self

    def set_surrogate(self, surrogate: Surrogate, n_neighbors: int = 50) -> Self:
        """
        Wrap a raw Surrogate in a LocalSurrogateManager. Returns self.

        Equivalent to ``set_surrogate_manager(LocalSurrogateManager(surrogate, ...))``.
        """
        from saealib.surrogate.training_set import KNNObjectiveSet

        self._set_component(
            "surrogate_manager",
            LocalSurrogateManager(
                surrogate,
                training_set=KNNObjectiveSet(n_neighbors=n_neighbors),
            ),
            "Configured with set_surrogate().",
        )
        self._set_framework_component(
            "acquisition",
            MeanPrediction(direction=self.problem.direction),
            "Uses the prediction acquisition paired with set_surrogate().",
        )
        return self

    def set_strategy(
        self, strategy: OptimizationStrategy, *, transition: Transition | None = None
    ) -> Self:
        """Set the optimization strategy. Returns self.

        Parameters
        ----------
        strategy : OptimizationStrategy
            The component to bind.
        transition : Transition or None, optional
            Transition to run at the commit point of this replacement.
        """
        self._set_component(
            "strategy",
            strategy,
            "Configured with set_strategy().",
            transition=transition,
        )
        return self

    def set_evaluator(
        self, evaluator: Evaluator, *, transition: Transition | None = None
    ) -> Self:
        """Set the evaluator. Returns self.

        Parameters
        ----------
        evaluator : Evaluator
            The component to bind.
        transition : Transition or None, optional
            Transition to run at the commit point of this replacement.
        """
        if evaluator.has_partial_lifecycle_override():
            raise ConfigurationError(
                "submit(), collect(), and acknowledge() must be overridden together"
            )
        self._set_component(
            "evaluator",
            evaluator,
            "Configured with set_evaluator().",
            transition=transition,
        )
        return self

    def set_async_evaluation_scheduler(
        self, scheduler: AsyncEvaluationScheduler | None
    ) -> Self:
        """Configure asynchronous evaluation for built-in strategies."""
        self.async_evaluation_scheduler = scheduler
        if scheduler is not None:
            scheduler.algorithm = getattr(self, "algorithm", None)
            scheduler.callback_manager = self.cbmanager
            scheduler.feedback_builder = self.feedback_builder
        return self

    def set_termination(self, termination: Termination) -> Self:
        """Set the termination condition. Returns self."""
        self._set_component(
            "termination", termination, "Configured with set_termination()."
        )
        return self

    def set_instance_name(self, name: str) -> Self:
        """Set the instance name. Returns self."""
        self.instance_name = name
        return self

    def set_preset(self, preset: str | Path | dict) -> Self:
        """
        Set a user-defined preset. Returns self.

        The preset fills components that are not already configured via
        ``set_*()``; explicitly set components take precedence.
        """
        from saealib.defaults import load_preset

        self._preset = load_preset(preset)
        self._configuration_version += 1
        return self

    # --- preset export ---

    def save_preset(self, path: str | Path) -> Path:
        """
        Save the currently configured components as a reusable preset file.

        Serializes ``algorithm``, ``surrogate_manager``, ``strategy``, and
        ``termination`` (whichever are set via ``set_*()``) into a preset
        dict and writes it as YAML. Problem-owned parameters (``dim``,
        ``direction``) are stripped so the preset can be reused across
        problems of different dimensionality.

        Parameters
        ----------
        path : str or Path
            Destination file path. The ``.yaml`` extension is added if absent.

        Returns
        -------
        Path
            The path the preset was written to.

        Raises
        ------
        ValidationError
            If no components are configured, or a configured component
            cannot be serialized (e.g. holds a raw lambda).
        """
        from saealib.defaults import dump_preset
        from saealib.registry import _strip_params, to_spec

        preset: dict = {}
        for name in (
            "algorithm",
            "surrogate_manager",
            "strategy",
            "evaluation_planner",
            "feedback_builder",
            "termination",
        ):
            component = getattr(self, name, None)
            if component is None:
                continue
            try:
                spec = to_spec(component)
            except ValidationError as e:
                raise ValidationError(
                    f"Cannot save preset: {name} is not serializable: {e}"
                ) from e
            preset[name] = _strip_params(spec, "dim", "direction")

        if not preset:
            raise ValidationError(
                "Cannot save preset: no components are configured. Call "
                "set_algorithm()/set_strategy()/set_surrogate_manager()/"
                "set_termination() first."
            )
        return dump_preset(preset, path)

    # --- callbacks ---

    def dispatch(self, event: Event) -> None:
        """Dispatch an event to the callback manager."""
        context = getattr(event, "ctx", None)
        marker = object()
        saved_marker = marker
        if isinstance(context, OptimizationState):
            saved_marker = context.__dict__.get("_checkpoint_callback", marker)
            context.__dict__["_checkpoint_callback"] = type(event).__name__
        try:
            self.cbmanager.dispatch(event)
        finally:
            if isinstance(context, OptimizationState):
                if saved_marker is marker:
                    context.__dict__.pop("_checkpoint_callback", None)
                else:
                    context.__dict__["_checkpoint_callback"] = saved_marker

    # --- run ---

    def validate(self, *, require_initializer: bool = True) -> list[str]:
        """
        Check configuration consistency. Returns list of issues.

        Component defaults are resolved before checking the composition.

        Parameters
        ----------
        require_initializer : bool, optional
            When False, skip the initializer presence check.  Use this when
            resuming from a checkpoint where initialization is not needed.
        """
        diagnostics = tuple(
            self.contract_diagnostics()
            if require_initializer
            else self._collect_contract_diagnostics(require_initializer=False)
        )
        self._last_contract_diagnostics = diagnostics
        return [
            str(diagnostic)
            for diagnostic in self._last_contract_diagnostics
            if diagnostic.severity is Severity.ERROR
        ]

    def _publish_component_resolution(
        self,
        resolution: _ComponentResolution,
        *,
        feature_encoder_consumed: bool | None = None,
    ) -> None:
        """Publish resolved bindings for this optimizer execution session."""
        for name, value in resolution.components.items():
            object.__setattr__(self, name, value)
            origin = resolution.origins.get(name)
            if origin is None:
                self._component_origins.pop(name, None)
                self._materialized_components.pop(name, None)
                continue
            self._component_origins[name] = origin
            if origin.source in {"setter", "direct assignment"}:
                self._materialized_components.pop(name, None)
            else:
                self._materialized_components[name] = value
        self._component_resolution = resolution
        self._default_resolution = resolution.defaults
        comparator_origin = resolution.origins.get("comparator")
        if (
            comparator_origin is not None
            and comparator_origin.source == "framework default"
        ):
            self._comparator_default = (
                self.problem,
                resolution.components.get("comparator"),
            )
        elif comparator_origin is not None:
            self._comparator_default = None
        if (
            resolution.feature_encoder_origin is not None
            and resolution.feature_encoder_origin.source == "framework default"
        ):
            self._feature_encoder_default = (
                resolution.feature_encoder_space,
                resolution.feature_encoder,
                resolution.feature_encoder_origin.reason,
            )
        elif (
            resolution.feature_encoder_origin is not None
            or feature_encoder_consumed is False
        ):
            self._feature_encoder_default = None
        feature_encoder_origin = resolution.feature_encoder_origin
        if (
            feature_encoder_origin is not None
            and feature_encoder_origin.source == "persisted"
        ):
            self._component_origins["feature_encoder"] = feature_encoder_origin
        elif feature_encoder_origin is not None or feature_encoder_consumed is False:
            self._component_origins.pop("feature_encoder", None)
        feedback_origin = resolution.origins.get("feedback_builder")
        self.feedback_builder_explicit = feedback_origin is not None and (
            feedback_origin.source in {"setter", "direct assignment", "user preset"}
        )
        if self.async_evaluation_scheduler is not None:
            self.async_evaluation_scheduler.algorithm = self.algorithm

    def resolve_defaults(self) -> None:
        """Resolve component and semantic defaults for the current composition."""
        resolution = resolve_components(self)
        self._publish_component_resolution(resolution)
        component_types = {
            name: type(component).__name__
            for name, component in resolution.components.items()
            if component is not None
        }
        selected_sources = {
            name: f"{origin.source}: {origin.reason}"
            for name, origin in resolution.origins.items()
        }
        logger.debug(
            "Resolved defaults: bundled_preset=%s components=%s values=%s sources=%s",
            resolution.bundled_preset,
            component_types,
            {key.name: value for key, value in resolution.defaults.values.items()},
            selected_sources,
        )

    def _inject_acquisition_directions(self) -> None:
        """Auto-inject ``problem.direction`` into unset acquisition directions.

        Mirrors the "inherit from problem unless explicitly set" contract used
        for ``NSGA3Comparator.rng`` and ``SingleObjectiveComparator.direction``:
        an acquisition function that already has an explicit ``direction`` (or
        opts out via ``direction_sensitive = False``) is left untouched.
        """
        for acq in self._iter_acquisitions():
            if (
                getattr(acq, "direction_sensitive", True)
                and getattr(acq, "direction", None) is None
            ):
                acq.direction = self.problem.direction

    def _iter_acquisitions(self):
        """Yield provider-owned acquisitions, including composite children."""
        acquisition = self.acquisition
        if acquisition is None:
            return
        if isinstance(acquisition, CompositeAcquisition):
            yield from acquisition.acquisitions.values()
        else:
            yield acquisition

    def _select_preset_name(self, defaults: dict, algorithm: Algorithm | None) -> str:
        from saealib.defaults.loader import select_preset_name

        algorithm_name = None if algorithm is None else type(algorithm).__name__
        return select_preset_name(defaults, self.problem, algorithm_name)

    def _build_components_from_spec(
        self, spec: dict
    ) -> tuple[SurrogateManager, AcquisitionFunction]:
        return self._build_components_from_spec_static(
            spec, self.problem.dim, self.problem.direction
        )

    @staticmethod
    def _build_components_from_spec_static(
        spec: dict, dim: int, direction, *, surrogate: Surrogate | None = None
    ) -> tuple[SurrogateManager, AcquisitionFunction]:
        from saealib.registry import _inject_params, build

        manager_spec = copy.deepcopy(spec)
        params = manager_spec.setdefault("params", {})
        if surrogate is None:
            params.setdefault(
                "surrogate",
                {
                    "type": "RBFSurrogate",
                    "params": {"kernel": GaussianKernel()},
                },
            )
        else:
            params["surrogate"] = surrogate
        acquisition_spec = params.pop("acquisition", None)
        if acquisition_spec is None:
            acquisition_spec = _default_acquisition_spec(
                manager_spec.get("type"), direction
            )
        manager_spec = _inject_params(manager_spec, dim=dim, direction=direction)
        acquisition_spec = _inject_params(
            acquisition_spec, dim=dim, direction=direction
        )
        return build(manager_spec), build(acquisition_spec)

    def _register_checkpoint(
        self,
        path: str | Path,
        interval: int,
        format: str,
        delete_on_success: bool,
    ) -> None:
        from saealib.checkpoint import CheckpointCallback

        cb = CheckpointCallback(
            path=path,
            interval=interval,
            format=format,
            delete_on_success=delete_on_success,
            optimizer=self if format in ("pickle", "both") else None,
        )
        cb.register(self.cbmanager)

    def _validate_checkpointable(self, plan: ExecutablePlan | None = None) -> None:
        """Reject portable checkpoints for components with non-exportable state.

        Parameters
        ----------
        plan : ExecutablePlan or None, optional
            Plan to inspect.  ``None`` (default) uses the compiled plan of the
            configured components.  A commit point passes its candidate plan,
            so the check covers a replacement before it is published.
        """
        if plan is None:
            plan = self._executable_plan
        if plan is None:
            return
        rejected = [
            f"{component_id} ({type_name})"
            for component_id, type_name in _non_checkpointable_components(plan)
        ]
        if rejected:
            raise ConfigurationError(
                "Portable checkpointing requires state.checkpointable=True; "
                "non-checkpointable component(s): " + ", ".join(rejected)
            )

    def iterate(
        self,
        checkpoint_path: str | Path | None = None,
        checkpoint_interval: int = 1,
        checkpoint_format: str = "npz",
        checkpoint_delete_on_success: bool = False,
    ) -> Generator[OptimizationState, None, OptimizationState]:
        """
        Iterate the optimization process.

        Parameters
        ----------
        checkpoint_path : str, Path, or None, optional
            If provided, checkpoints are saved to this directory every
            *checkpoint_interval* generations.
        checkpoint_interval : int, optional
            Generations between checkpoints.  Default: 1.
        checkpoint_format : {'npz', 'pickle', 'both'}, optional
            Checkpoint format.  Default: ``'npz'``.
        checkpoint_delete_on_success : bool, optional
            Delete checkpoints on normal termination.  Default: False.

        Returns
        -------
        Generator[OptimizationState]
            Generator of OptimizationState; it returns the state the runtime
            ended with.
        """
        self.resolve_defaults()
        self._compile_plan()
        issues = self.validate()
        if issues:
            raise ConfigurationError(
                "Optimizer misconfigured:\n" + "\n".join(f"  - {m}" for m in issues)
            )
        self._inject_acquisition_directions()
        self._portable_checkpointing = checkpoint_path is not None and (
            checkpoint_format in ("npz", "both")
        )
        if checkpoint_path is not None:
            if checkpoint_format in ("npz", "both"):
                self._validate_checkpointable()
            self._register_checkpoint(
                checkpoint_path,
                checkpoint_interval,
                checkpoint_format,
                checkpoint_delete_on_success,
            )
        return Runner(self).iterate()

    def run(
        self,
        checkpoint_path: str | Path | None = None,
        checkpoint_interval: int = 1,
        checkpoint_format: str = "npz",
        checkpoint_delete_on_success: bool = False,
    ) -> OptimizationState:
        """
        Run the optimization process.

        Parameters
        ----------
        checkpoint_path : str, Path, or None, optional
            If provided, checkpoints are saved to this directory every
            *checkpoint_interval* generations.
        checkpoint_interval : int, optional
            Generations between checkpoints.  Default: 1.
        checkpoint_format : {'npz', 'pickle', 'both'}, optional
            Checkpoint format.  Default: ``'npz'``.
        checkpoint_delete_on_success : bool, optional
            Delete checkpoints on normal termination.  Default: False.

        Returns
        -------
        OptimizationState
            The optimization context.
        """
        logger.info(
            "Optimization run started: dim=%s n_obj=%s",
            self.problem.dim,
            self.problem.n_obj,
        )
        self.resolve_defaults()
        self._compile_plan()
        issues = self.validate()
        if issues:
            raise ConfigurationError(
                "Optimizer misconfigured:\n" + "\n".join(f"  - {m}" for m in issues)
            )
        self._inject_acquisition_directions()
        self._portable_checkpointing = checkpoint_path is not None and (
            checkpoint_format in ("npz", "both")
        )
        if checkpoint_path is not None:
            if checkpoint_format in ("npz", "both"):
                self._validate_checkpointable()
            self._register_checkpoint(
                checkpoint_path,
                checkpoint_interval,
                checkpoint_format,
                checkpoint_delete_on_success,
            )
        state = Runner(self).run()
        logger.info(
            "Optimization run finished: fe=%s gen=%s",
            getattr(state, "fe", None),
            getattr(state, "gen", None),
        )
        return state

    def iterate_from(
        self,
        ctx: OptimizationState,
        checkpoint_path: str | Path | None = None,
        checkpoint_interval: int = 1,
        checkpoint_format: str = "npz",
        checkpoint_delete_on_success: bool = False,
    ) -> Generator[OptimizationState, None, OptimizationState]:
        """
        Resume iteration from an existing context (e.g. loaded from checkpoint).

        Does not call the initializer; all other components must be configured.

        Parameters
        ----------
        ctx : OptimizationState
            Context to resume from.
        checkpoint_path : str, Path, or None, optional
            If provided, checkpoints are saved to this directory every
            *checkpoint_interval* generations.
        checkpoint_interval : int, optional
            Generations between checkpoints.  Default: 1.
        checkpoint_format : {'npz', 'pickle', 'both'}, optional
            Checkpoint format.  Default: ``'npz'``.
        checkpoint_delete_on_success : bool, optional
            Delete checkpoints on normal termination.  Default: False.

        Returns
        -------
        Generator[OptimizationState, None, OptimizationState]
            Generator of OptimizationState; it returns the state the runtime
            ended with.
        """
        self._compile_plan(require_initializer=False)
        issues = self.validate(require_initializer=False)
        if issues:
            raise ConfigurationError(
                "Optimizer misconfigured:\n" + "\n".join(f"  - {m}" for m in issues)
            )
        self._inject_acquisition_directions()
        self._portable_checkpointing = checkpoint_path is not None and (
            checkpoint_format in ("npz", "both")
        )
        if checkpoint_path is not None:
            if checkpoint_format in ("npz", "both"):
                self._validate_checkpointable()
            self._register_checkpoint(
                checkpoint_path,
                checkpoint_interval,
                checkpoint_format,
                checkpoint_delete_on_success,
            )
        return Runner(self).iterate_from(ctx)

    def run_from(
        self,
        ctx: OptimizationState,
        checkpoint_path: str | Path | None = None,
        checkpoint_interval: int = 1,
        checkpoint_format: str = "npz",
        checkpoint_delete_on_success: bool = False,
    ) -> OptimizationState:
        """
        Resume and run to completion from an existing context.

        Parameters
        ----------
        ctx : OptimizationState
            Context to resume from.
        checkpoint_path : str, Path, or None, optional
            If provided, checkpoints are saved to this directory every
            *checkpoint_interval* generations.
        checkpoint_interval : int, optional
            Generations between checkpoints.  Default: 1.
        checkpoint_format : {'npz', 'pickle', 'both'}, optional
            Checkpoint format.  Default: ``'npz'``.
        checkpoint_delete_on_success : bool, optional
            Delete checkpoints on normal termination.  Default: False.

        Returns
        -------
        OptimizationState
            The final optimization context.
        """
        self._compile_plan(require_initializer=False)
        issues = self.validate(require_initializer=False)
        if issues:
            raise ConfigurationError(
                "Optimizer misconfigured:\n" + "\n".join(f"  - {m}" for m in issues)
            )
        self._inject_acquisition_directions()
        self._portable_checkpointing = checkpoint_path is not None and (
            checkpoint_format in ("npz", "both")
        )
        if checkpoint_path is not None:
            if checkpoint_format in ("npz", "both"):
                self._validate_checkpointable()
            self._register_checkpoint(
                checkpoint_path,
                checkpoint_interval,
                checkpoint_format,
                checkpoint_delete_on_success,
            )
        return Runner(self).run_from(ctx)

    # ------------------------------------------------------------------
    # Checkpoint: pickle (limited complete reproducibility)
    # ------------------------------------------------------------------

    _PICKLE_WARNING = (
        "Pickle checkpoints are version-sensitive. "
        "Reproducibility is only guaranteed within the same Python "
        "and library versions."
    )

    def __getstate__(self) -> dict[str, object]:
        """Exclude the compiled graph plan from legacy pickle checkpoints."""
        state = self.__dict__.copy()
        state["_executable_plan"] = None
        state["_prepared_diagnostic_cache"] = None
        state["_feature_encoder_default"] = None
        state["_replacement_records"] = {}
        return state

    def __setstate__(self, state: dict[str, object]) -> None:
        """Restore an optimizer without a stale compiled execution plan."""
        self.__dict__.update(state)
        self.__dict__.setdefault("_executable_plan", None)
        self.__dict__.setdefault("_last_published_bindings", None)
        self.__dict__.setdefault("_prepared_diagnostic_cache", None)
        self.__dict__.setdefault("_default_resolution", None)
        self.__dict__.setdefault("_verification_enabled", True)
        self.__dict__.setdefault("_portable_checkpointing", False)
        self.__dict__.setdefault("_component_resolution", None)
        self.__dict__.setdefault("_component_origins", {})
        self.__dict__.setdefault("_materialized_components", {})
        self.__dict__.setdefault("_feature_encoder_default", None)
        self.__dict__.setdefault("_replacement_records", {})
        self.__dict__.setdefault("comparator", None)
        self.__dict__.setdefault("_comparator_default", None)
        self.__dict__.setdefault("_configuration_version", 0)
        self.__dict__.setdefault("_track_component_assignments", True)
        if self.__dict__.get("comparator") is None:
            stored_comparator = _restored_problem_comparator(
                self.__dict__.get("problem")
            )
            if stored_comparator is not None:
                object.__setattr__(self, "comparator", stored_comparator)
        if "_component_origins" not in state:
            source = (
                "persisted"
                if self._default_resolution is not None
                else "direct assignment"
            )
            for name in self._COMPONENT_ASSIGNMENTS:
                value = getattr(self, name, None)
                if value is not None:
                    self._component_origins[name] = _ComponentOrigin(
                        value,
                        source,
                        "Retained the component stored in optimizer state.",
                        _signature(value),
                    )
        comparator = self.__dict__.get("comparator")
        if comparator is not None and "comparator" not in self._component_origins:
            self._component_origins["comparator"] = _ComponentOrigin(
                comparator,
                "persisted",
                "Adopted from the comparator stored on the restored problem.",
                _comparator_signature(comparator),
            )
        space = getattr(getattr(self, "problem", None), "space", None)
        services = getattr(space, "services", None)
        getter = getattr(services, "get", None)
        feature_encoder = getter("FeatureEncoder") if callable(getter) else None
        if (
            feature_encoder is not None
            and self._component_origins.get("feature_encoder") is None
            and getattr(services, "_feature_encoder_registration_count", None) is None
        ):
            self._component_origins["feature_encoder"] = _ComponentOrigin(
                feature_encoder,
                "persisted",
                "Retained the FeatureEncoder stored with the SearchSpace; its "
                "original selection source is unavailable.",
                _feature_encoder_signature(feature_encoder),
            )

    def save_pickle(self, ctx: OptimizationState, path: str | Path) -> None:
        """
        Save the optimizer and context together as a pickle checkpoint.

        This preserves fitted surrogate state and all component objects,
        offering complete reproducibility within the same environment.

        .. warning::
            Pickle files are tied to specific Python and library versions.
            Use :meth:`OptimizationState.save` for a more portable format.

        Parameters
        ----------
        ctx : OptimizationState
            Current optimization context.
        path : str or Path
            Destination file path.  The ``.pkl`` extension is added if absent.
        """
        warnings.warn(self._PICKLE_WARNING, UserWarning, stacklevel=2)
        p = Path(path)
        if not p.suffix:
            p = p.with_suffix(".pkl")
        self._validate_pickle_checkpoint(ctx)
        save_ctx = ctx
        guard = ctx.__dict__.get("_checkpoint_guard")
        if getattr(guard, "_checkpoint_run_end_callback", False):
            from saealib.execution.history import snapshot_generation_for_checkpoint

            save_ctx = snapshot_generation_for_checkpoint(ctx)
        try:
            payload = pickle.dumps((self, save_ctx))
        except Exception as exc:
            raise CheckpointError(f"checkpoint cannot be pickled: {exc}") from exc
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "wb") as f:
            f.write(payload)

    def _validate_pickle_checkpoint(self, ctx: OptimizationState) -> None:
        _validate_checkpoint_runtime(ctx)
        _validate_checkpoint_pending(ctx)
        scheduler = getattr(self, "async_evaluation_scheduler", None)
        evaluator = getattr(scheduler, "evaluator", None) or getattr(
            self, "evaluator", None
        )
        if evaluator is not None:
            _validate_pending_reattachable(ctx, evaluator)
            try:
                pickle.dumps(evaluator)
            except Exception as exc:
                raise CheckpointError(
                    f"evaluator {type(evaluator).__name__} cannot be pickled: {exc}"
                ) from exc
        # Check the complete object graph before any callback creates a file.
        try:
            pickle.dumps((self, ctx))
        except Exception as exc:
            raise CheckpointError(f"checkpoint cannot be pickled: {exc}") from exc

    @classmethod
    def load_pickle(cls, path: str | Path) -> tuple[Optimizer, OptimizationState]:
        """
        Load an optimizer and context from a pickle checkpoint.

        The returned context has ``resumed=True``.  Call
        :meth:`run_from` or :meth:`iterate_from` on the returned optimizer
        to continue the optimization.

        .. warning::
            Pickle files are tied to specific Python and library versions.

        Parameters
        ----------
        path : str or Path
            Path to the ``.pkl`` file.  The extension is added if absent.

        Returns
        -------
        tuple[Optimizer, OptimizationState]
        """
        warnings.warn(cls._PICKLE_WARNING, UserWarning, stacklevel=2)
        p = Path(path)
        if not p.suffix:
            p = p.with_suffix(".pkl")
        with open(p, "rb") as f:
            optimizer, ctx = pickle.load(f)
        ctx = dataclasses.replace(ctx, data={**ctx.data, "resumed": True})
        return optimizer, ctx

    # ------------------------------------------------------------------
    # Construction: from a problem definition file
    # ------------------------------------------------------------------

    @classmethod
    def from_problem_file(
        cls,
        problem_path: str | Path,
        preset: str | Path | dict | None = None,
    ) -> Optimizer:
        """
        Build an Optimizer from a Python file defining a problem.

        The file at *problem_path* is executed as a standalone module (it is not
        registered in ``sys.modules``) and must define a top-level ``problem``
        variable holding a :class:`~saealib.problem.Problem` instance. If
        *preset* is given, it is applied first via :meth:`set_preset`; any of
        the top-level variables ``algorithm``, ``strategy``,
        ``surrogate_manager``, ``termination`` defined in the file are then
        applied via the corresponding ``set_*()`` method, so the file can
        override individual components of the preset. A top-level ``seed``
        variable, if present, is passed to the ``Optimizer`` constructor.

        Parameters
        ----------
        problem_path : str or Path
            Path to a ``.py`` file defining a top-level ``problem`` variable.
        preset : str, Path, dict, or None, optional
            Preset applied before the file's own component definitions, so
            that the file can override individual components. See
            :meth:`set_preset`.

        Returns
        -------
        Optimizer
            A configured, not-yet-run ``Optimizer`` instance. Call ``run()``
            or ``iterate()`` on it.

        Raises
        ------
        ValidationError
            If the file does not define a top-level ``problem`` variable, or
            it is not a ``Problem`` instance. Also raised if a top-level
            ``algorithm``, ``strategy``, ``surrogate_manager``,
            ``termination``, or ``seed`` variable is defined but is not an
            instance of its expected type.
        FileNotFoundError
            If *problem_path* does not exist.

        Examples
        --------
        >>> # problem.py:
        >>> #     problem = Problem(func=..., dim=2, n_obj=1, direction=[-1])
        >>> #     seed = 42
        >>> opt = Optimizer.from_problem_file("problem.py", preset="preset.yaml")
        >>> ctx = opt.run()  # doctest: +SKIP
        """
        from saealib.algorithms.base import Algorithm
        from saealib.problem import Problem
        from saealib.strategies.base import OptimizationStrategy
        from saealib.surrogate.manager import SurrogateManager
        from saealib.termination import Termination

        p = Path(problem_path)
        spec = importlib.util.spec_from_file_location(p.stem, p)
        if spec is None or spec.loader is None:
            raise ValidationError(f"Cannot load module from {p}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        problem = getattr(module, "problem", None)
        if not isinstance(problem, Problem):
            raise ValidationError(
                f"{p} must define a top-level `problem` variable holding a "
                "Problem instance"
            )

        seed = getattr(module, "seed", None)
        if seed is not None and not isinstance(seed, int):
            raise ValidationError(
                f"{p}: top-level 'seed' must be an int instance, "
                f"got {type(seed).__name__}"
            )
        opt = cls(problem, seed=seed) if seed is not None else cls(problem)

        if preset is not None:
            opt.set_preset(preset)

        for name, setter, expected in (
            ("algorithm", opt.set_algorithm, Algorithm),
            ("strategy", opt.set_strategy, OptimizationStrategy),
            ("surrogate_manager", opt.set_surrogate_manager, SurrogateManager),
            ("evaluation_planner", opt.set_evaluation_planner, EvaluationPlanner),
            ("feedback_builder", opt.set_feedback_builder, FeedbackBuilder),
            ("termination", opt.set_termination, Termination),
        ):
            component = getattr(module, name, None)
            if component is not None:
                if not isinstance(component, expected):
                    raise ValidationError(
                        f"{p}: top-level '{name}' must be a {expected.__name__} "
                        f"instance, got {type(component).__name__}"
                    )
                setter(component)

        return opt
