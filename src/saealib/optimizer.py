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
from typing import TYPE_CHECKING, Any, Protocol, cast

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
from saealib.context import OptimizationState
from saealib.core.compiler import (
    CompileContext,
    Compiler,
    ContractPath,
    Diagnostic,
    DiagnosticBag,
    ExecutablePlan,
    Severity,
    check_component_contract,
)
from saealib.core.compiler.contract_diagnostics import (
    check_pymoo_feedback_compatibility,
)
from saealib.core.compiler.graph import ComponentGraph
from saealib.core.contracts import ComponentContract
from saealib.core.state import OPTIMIZATION_STATE_INITIAL_KEYS
from saealib.exceptions import ConfigurationError, ValidationError
from saealib.execution.evaluator import Evaluator, SerialEvaluator
from saealib.execution.history import SUPPORTED_HISTORY_CHANNELS
from saealib.execution.runner import Runner
from saealib.execution.runtime import default_runtime_registry
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
from saealib.termination import max_fe as max_fe_cond

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from saealib.algorithms.base import Algorithm
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
        self._last_contract_diagnostics: tuple[Diagnostic, ...] = ()
        self._executable_plan: ExecutablePlan | None = None

    # --- setters (all return self for chaining) ---

    def contract_diagnostics(self) -> DiagnosticBag:
        """Return diagnostics for the currently configured component graph."""
        diagnostics = DiagnosticBag()
        contracts: dict[str, ComponentContract] = {}
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
            component = getattr(self, name, None)
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
        algorithm = getattr(self, "algorithm", None)
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
        return diagnostics

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

    def _compile_plan(self) -> ExecutablePlan | None:
        """Build and compile the configured strategy graph once per run."""
        strategy = getattr(self, "strategy", None)
        build_graph = getattr(strategy, "build_graph", None)
        if not callable(build_graph):
            self._executable_plan = None
            return None

        graph = build_graph(self)
        if not isinstance(graph, ComponentGraph):
            self._executable_plan = None
            return None
        plan = Compiler().compile(
            graph,
            CompileContext(
                space=self.problem.space,
                problem=self.problem,
                offered_runtime_capabilities=default_runtime_registry.offered_capabilities(
                    self
                ),
                initial_state_keys=OPTIMIZATION_STATE_INITIAL_KEYS,
            ),
        )
        self._executable_plan = plan
        return plan

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

    def set_initializer(self, initializer: Initializer) -> Self:
        """Set the initializer. Returns self."""
        self.initializer = initializer
        return self

    def set_algorithm(self, algorithm: Algorithm) -> Self:
        """Set the evolutionary algorithm. Returns self."""
        self.algorithm = algorithm
        return self

    def set_surrogate_manager(self, manager: SurrogateManager) -> Self:
        """Set the surrogate manager. Returns self."""
        self.surrogate_manager = manager
        if isinstance(manager, PairwiseSurrogateManager) and self.acquisition is None:
            self.acquisition = WinRateAcquisition()
        return self

    def set_acquisition(self, acquisition) -> Self:
        """Set the independent acquisition component."""
        self.acquisition = acquisition
        return self

    @property
    def evaluation_planner(self) -> EvaluationPlanner | None:
        """Return the configured evaluation planner."""
        return self._evaluation_planner

    @evaluation_planner.setter
    def evaluation_planner(self, planner: EvaluationPlanner | None) -> None:
        self._evaluation_planner = planner

    def set_evaluation_planner(self, planner: EvaluationPlanner) -> Self:
        """Set the evaluation planner."""
        self.evaluation_planner = planner
        return self

    def set_feedback_builder(self, builder: FeedbackBuilder) -> Self:
        """Set the feedback builder."""
        self.feedback_builder = builder
        self.feedback_builder_explicit = True
        return self

    def set_surrogate(self, surrogate: Surrogate, n_neighbors: int = 50) -> Self:
        """
        Wrap a raw Surrogate in a LocalSurrogateManager. Returns self.

        Equivalent to ``set_surrogate_manager(LocalSurrogateManager(surrogate, ...))``.
        """
        from saealib.surrogate.training_set import KNNObjectiveSet

        self.surrogate_manager = LocalSurrogateManager(
            surrogate,
            training_set=KNNObjectiveSet(n_neighbors=n_neighbors),
        )
        self.acquisition = MeanPrediction(direction=self.problem.direction)
        return self

    def set_strategy(self, strategy: OptimizationStrategy) -> Self:
        """Set the optimization strategy. Returns self."""
        self.strategy = strategy
        return self

    def set_evaluator(self, evaluator: Evaluator) -> Self:
        """Set the evaluator. Returns self."""
        if evaluator.has_partial_lifecycle_override():
            raise ConfigurationError(
                "submit(), collect(), and acknowledge() must be overridden together"
            )
        self.evaluator = evaluator
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
        self.termination = termination
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
        self.cbmanager.dispatch(event)

    # --- run ---

    def validate(self, *, require_initializer: bool = True) -> list[str]:
        """
        Check configuration consistency. Returns list of issues.

        Call :meth:`resolve_defaults` first when validation results should
        reflect the components that will be used for execution.

        Parameters
        ----------
        require_initializer : bool, optional
            When False, skip the initializer presence check.  Use this when
            resuming from a checkpoint where initialization is not needed.
        """
        issues: list[str] = []

        self._validate_evaluator_lifecycle(issues)

        algorithm = getattr(self, "algorithm", None)
        strategy = getattr(self, "strategy", None)
        termination = getattr(self, "termination", None)
        surrogate_manager = getattr(self, "surrogate_manager", None)

        self._validate_required_components(
            issues, algorithm, strategy, termination, require_initializer
        )
        self._validate_component_protocols(issues)
        self._validate_strategy_requirements(issues, strategy, surrogate_manager)
        self._validate_comparator_direction(issues)

        if surrogate_manager is not None:
            self._validate_surrogate_compatibility(issues, surrogate_manager)

        diagnostics = tuple(self.contract_diagnostics())
        if self._executable_plan is not None:
            diagnostics = tuple(
                dict.fromkeys((*diagnostics, *self._executable_plan.diagnostics))
            )
        self._last_contract_diagnostics = diagnostics
        issues.extend(
            str(diagnostic)
            for diagnostic in self._last_contract_diagnostics
            if diagnostic.severity is Severity.ERROR
        )

        return issues

    def _validate_evaluator_lifecycle(self, issues: list[str]) -> None:
        if self.evaluator.has_partial_lifecycle_override():
            issues.append(
                "submit(), collect(), and acknowledge() must be overridden together"
            )

    def _validate_required_components(
        self,
        issues: list[str],
        algorithm: object,
        strategy: object,
        termination: object,
        require_initializer: bool,
    ) -> None:
        if algorithm is None:
            issues.append("algorithm is not set; call set_algorithm()")
        if strategy is None:
            issues.append("strategy is not set; call set_strategy()")
        if require_initializer and self.initializer is None:
            issues.append("initializer is not set; call set_initializer()")
        if termination is None:
            issues.append("termination is not set; call set_termination()")

    def _validate_component_protocols(self, issues: list[str]) -> None:
        if self.evaluation_planner is not None and not isinstance(
            self.evaluation_planner, EvaluationPlanner
        ):
            issues.append("evaluation_planner must be an EvaluationPlanner")
        if self.feedback_builder is not None and not isinstance(
            self.feedback_builder, FeedbackBuilder
        ):
            issues.append("feedback_builder must be a FeedbackBuilder")

    def _validate_strategy_requirements(
        self,
        issues: list[str],
        strategy: object,
        surrogate_manager: object,
    ) -> None:
        if (
            strategy is not None
            and getattr(strategy, "requires_surrogate", False)
            and surrogate_manager is None
        ):
            issues.append(
                f"{type(strategy).__name__} requires a surrogate_manager; "
                "call set_surrogate_manager() or set_surrogate()"
            )
        if (
            strategy is not None
            and getattr(strategy, "requires_surrogate", False)
            and self.acquisition is None
        ):
            issues.append(
                "a provider-owned acquisition is required; call set_acquisition()"
            )

    def _validate_comparator_direction(self, issues: list[str]) -> None:
        _dim = getattr(self.problem.comparator, "direction", None)
        if (
            _dim is not None
            and hasattr(_dim, "__len__")
            and len(_dim) > 0
            and len(_dim) != self.problem.n_obj
        ):
            issues.append(
                f"comparator direction length ({len(_dim)}) does not match "
                f"problem.n_obj ({self.problem.n_obj})"
            )

    def _validate_surrogate_compatibility(
        self, issues: list[str], surrogate_manager: object
    ) -> None:
        acq = self.acquisition
        surr = getattr(surrogate_manager, "surrogate", None)
        if (
            acq is not None
            and surr is not None
            and getattr(acq, "requires_uncertainty", False)
            and not getattr(surr, "provides_uncertainty", False)
        ):
            issues.append(
                f"{type(acq).__name__} requires uncertainty estimates but "
                f"{type(surr).__name__} does not provide them "
                "(provides_uncertainty=False)"
            )

        for acq in self._iter_acquisitions():
            _adir = getattr(acq, "direction", None)
            if (
                _adir is not None
                and hasattr(_adir, "__len__")
                and len(_adir) > 0
                and len(_adir) != self.problem.n_obj
            ):
                issues.append(
                    f"{type(acq).__name__} direction length ({len(_adir)}) does "
                    f"not match problem.n_obj ({self.problem.n_obj})"
                )

    def _discover_default_hint_providers(self) -> tuple[list[Any], dict[str, object]]:
        """Discover default providers from the configured component graph.

        The component contracts describe constructor-held parts, so following
        ``ComponentContract.parts`` keeps discovery independent of concrete
        component classes and supports arbitrarily nested custom components.
        """
        from saealib.defaults.builtin import BUILTIN_DEFAULT_PROVIDER

        roots = (
            ("initializer", getattr(self, "initializer", None)),
            ("algorithm", getattr(self, "algorithm", None)),
            ("strategy", getattr(self, "strategy", None)),
            ("surrogate_manager", getattr(self, "surrogate_manager", None)),
            ("acquisition", getattr(self, "acquisition", None)),
            ("comparator", self.problem.comparator),
            ("termination", getattr(self, "termination", None)),
            ("evaluator", getattr(self, "evaluator", None)),
            ("evaluation_planner", getattr(self, "evaluation_planner", None)),
            ("feedback_builder", getattr(self, "feedback_builder", None)),
            (
                "async_evaluation_scheduler",
                getattr(self, "async_evaluation_scheduler", None),
            ),
        )
        providers: list[Any] = [BUILTIN_DEFAULT_PROVIDER]
        components: dict[str, object] = {}
        visited: set[int] = set()
        provider_ids = {id(BUILTIN_DEFAULT_PROVIDER)}

        def visit(
            component: object,
            path: str,
            declared_contract: ComponentContract | None = None,
        ) -> None:
            if component is None:
                return
            # Keep every reachable path in the context even when a component
            # instance is shared by multiple roots or parts.  Providers and
            # child contracts are still visited only once per instance.
            components[path] = component
            if id(component) in visited:
                return
            visited.add(id(component))

            try:
                default_hints = getattr(component, "default_hints", None)
            except Exception as error:
                raise ConfigurationError(
                    f"Cannot inspect {path}.default_hints: "
                    f"{type(error).__name__}: {error}"
                ) from error
            if callable(default_hints) and id(component) not in provider_ids:
                providers.append(component)
                provider_ids.add(id(component))

            contract = declared_contract
            if contract is None:
                try:
                    contract_method = getattr(component, "contract", None)
                except Exception as error:
                    raise ConfigurationError(
                        f"Cannot inspect {path}.contract: "
                        f"{type(error).__name__}: {error}"
                    ) from error
                if not callable(contract_method):
                    return
                try:
                    contract = cast(ComponentContract, contract_method())
                except Exception as error:
                    raise ConfigurationError(
                        f"Cannot inspect {path}.contract(): "
                        f"{type(error).__name__}: {error}"
                    ) from error
                if not isinstance(contract, ComponentContract):
                    raise ConfigurationError(
                        f"{path}.contract() returned "
                        f"{type(contract).__name__}, expected ComponentContract"
                    )

            for part in contract.parts:
                part_path = f"{path}.{part.name}"
                try:
                    child = getattr(component, part.name)
                except AttributeError as error:
                    if part.optional:
                        continue
                    raise ConfigurationError(
                        f"Cannot discover component part {part_path!r}: "
                        f"attribute is missing"
                    ) from error
                except Exception as error:
                    raise ConfigurationError(
                        f"Cannot inspect component part {part_path!r}: "
                        f"{type(error).__name__}: {error}"
                    ) from error
                if child is None:
                    if part.optional:
                        continue
                    raise ConfigurationError(
                        f"Cannot discover component part {part_path!r}: "
                        "required part is None"
                    )
                visit(child, part_path, part.contract)

        for name, component in roots:
            if component is not None:
                visit(component, name)
        return providers, components

    def resolve_defaults(self) -> None:
        """Fill unset components with library defaults (Registry + presets file).

        Components already set via ``set_*()`` are never overwritten. Gaps
        are filled first from a user preset (``set_preset()``), if any, then
        from a bundled preset selected by (1) the algorithm's registered
        name if ``algorithm`` is set, else (2) a Problem-shape rule, else
        (3) the universal fallback preset. ``initializer`` is computed
        from semantic defaults and is not part of any preset;
        ``termination`` falls back to the resolved maximum-evaluations default
        (``200 * problem.dim`` unless a provider recommends another value)
        only if neither ``set_*()`` nor a preset supplies one.
        """
        from saealib.defaults import load_defaults
        from saealib.registry import _inject_params, build

        algorithm = getattr(self, "algorithm", None)
        strategy = getattr(self, "strategy", None)
        surrogate_manager = getattr(self, "surrogate_manager", None)
        bundled_preset_name = None

        user_preset = getattr(self, "_preset", None)
        if user_preset is not None:
            dim = self.problem.dim
            direction = self.problem.direction
            if algorithm is None and "algorithm" in user_preset:
                algorithm = build(
                    _inject_params(
                        user_preset["algorithm"], dim=dim, direction=direction
                    )
                )
                self.algorithm = algorithm
            if surrogate_manager is None and "surrogate_manager" in user_preset:
                surrogate_manager, acquisition = self._build_components_from_spec(
                    user_preset["surrogate_manager"]
                )
                self.surrogate_manager = surrogate_manager
                if self.acquisition is None:
                    self.acquisition = acquisition
            if strategy is None and "strategy" in user_preset:
                strategy = build(
                    _inject_params(
                        user_preset["strategy"], dim=dim, direction=direction
                    )
                )
                self.strategy = strategy
            if self.evaluation_planner is None:
                spec = user_preset.get("evaluation_planner")
                if spec is not None:
                    self.evaluation_planner = build(spec)
            if self.feedback_builder is None and "feedback_builder" in user_preset:
                self.feedback_builder = build(user_preset["feedback_builder"])
                self.feedback_builder_explicit = True
            if (
                getattr(self, "termination", None) is None
                and "termination" in user_preset
            ):
                self.termination = build(
                    _inject_params(
                        user_preset["termination"], dim=dim, direction=direction
                    )
                )

        use_bundled_policies = strategy is None
        if (
            algorithm is None
            or strategy is None
            or surrogate_manager is None
            or self.evaluation_planner is None
            or self.feedback_builder is None
        ):
            defaults = load_defaults()
            bundled_preset_name = self._select_preset_name(defaults, algorithm)
            preset = defaults["presets"][bundled_preset_name]
            if algorithm is None and "algorithm" in preset:
                self.algorithm = build(
                    _inject_params(
                        preset["algorithm"],
                        dim=self.problem.dim,
                        direction=self.problem.direction,
                    )
                )
            if surrogate_manager is None and "surrogate_manager" in preset:
                manager, acquisition = self._build_components_from_spec(
                    preset["surrogate_manager"]
                )
                self.surrogate_manager = manager
                if self.acquisition is None:
                    self.acquisition = acquisition
            if strategy is None and "strategy" in preset:
                self.strategy = build(
                    _inject_params(
                        preset["strategy"],
                        dim=self.problem.dim,
                        direction=self.problem.direction,
                    )
                )
            if (
                use_bundled_policies
                and self.evaluation_planner is None
                and "evaluation_planner" in preset
            ):
                self.evaluation_planner = build(preset["evaluation_planner"])
            if (
                use_bundled_policies
                and self.feedback_builder is None
                and "feedback_builder" in preset
            ):
                self.feedback_builder = build(preset["feedback_builder"])

        from saealib.defaults.context import DefaultContext
        from saealib.defaults.keys import (
            INITIAL_ARCHIVE_SIZE,
            MAX_EVALUATIONS,
            POPULATION_SIZE,
        )
        from saealib.defaults.model import (
            DefaultHint,
            DefaultResolution,
            DefaultStrength,
            ResolvedDefault,
        )
        from saealib.defaults.resolver import DEFAULT_RESOLVER
        from saealib.execution.initializer import LHSInitializer

        # Resolve semantic defaults for the complete configured composition,
        # even when materialization of one of the defaults is unnecessary
        # because the user supplied that component explicitly.
        providers, components = self._discover_default_hint_providers()
        ctx = DefaultContext(
            problem=self.problem,
            seed=self.seed,
            components=components,
        )
        resolution = DEFAULT_RESOLVER.resolve(ctx, providers)

        dim = self.problem.dim
        n_population = resolution.get(POPULATION_SIZE, 4 * dim)
        n_archive = resolution.get(INITIAL_ARCHIVE_SIZE, 5 * dim)

        # Ensure archive is at least as large as population, and retain that
        # invariant in the provenance trace as well.  This semantic step must
        # run even when the initializer was supplied explicitly: otherwise a
        # second resolve_defaults() call would replace the normalized trace
        # with the raw resolver result.
        if n_archive < n_population:
            archive_resolution = resolution.resolved.get(INITIAL_ARCHIVE_SIZE)
            if (
                archive_resolution is not None
                and archive_resolution.selected_hint.strength
                == DefaultStrength.REQUIRED
            ):
                selected = archive_resolution.selected_hint
                raise ConfigurationError(
                    "REQUIRED default for "
                    f"{INITIAL_ARCHIVE_SIZE.name!r} ({n_archive}) from "
                    f"{selected.source!r} is smaller than the resolved "
                    f"population size ({n_population})"
                )

            adjusted_hint = DefaultHint(
                key=INITIAL_ARCHIVE_SIZE,
                value=n_population,
                strength=DefaultStrength.REQUIRED,
                source="optimizer",
                reason=(
                    f"Raised to match the resolved population size: {n_population}"
                ),
            )
            resolution = DefaultResolution(
                values={
                    **resolution.values,
                    INITIAL_ARCHIVE_SIZE: n_population,
                },
                resolved={
                    **resolution.resolved,
                    INITIAL_ARCHIVE_SIZE: ResolvedDefault(
                        key=INITIAL_ARCHIVE_SIZE,
                        value=n_population,
                        selected_hint=adjusted_hint,
                        alternatives=(
                            archive_resolution.alternatives
                            if archive_resolution is not None
                            else ()
                        ),
                    ),
                },
            )
            n_archive = n_population

        if self.initializer is None:
            self.initializer = LHSInitializer(
                n_init_archive=n_archive, n_init_population=n_population, seed=self.seed
            )

        self._default_resolution = resolution

        if getattr(self, "termination", None) is None:
            max_evaluations = resolution.get(MAX_EVALUATIONS, 200 * dim)
            self.termination = Termination(max_fe_cond(max_evaluations))
        if self.async_evaluation_scheduler is not None:
            self.async_evaluation_scheduler.algorithm = self.algorithm

        component_types = {
            name: type(component).__name__
            for name in (
                "algorithm",
                "strategy",
                "surrogate_manager",
                "acquisition",
                "evaluation_planner",
                "feedback_builder",
                "termination",
                "initializer",
            )
            if (component := getattr(self, name, None)) is not None
        }
        logger.debug(
            "Resolved defaults: bundled_preset=%s components=%s",
            bundled_preset_name,
            component_types,
        )

    def _resolve_defaults(self) -> None:
        """Compatibility wrapper for :meth:`resolve_defaults`."""
        self.resolve_defaults()

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

    def _validate_checkpointable(self) -> None:
        """Reject portable checkpoints for components with non-exportable state."""
        plan = self._executable_plan
        if plan is None:
            return
        snapshots = dict(plan.contract_snapshots)
        rejected = []
        for component_id, contract in snapshots.items():
            if not contract.assumptions["state.checkpointable"]:
                component = plan.graph.node_by_id(component_id).component
                component_name = getattr(component, "component", component)
                rejected.append(f"{component_id} ({type(component_name).__name__})")
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
    ) -> Generator[OptimizationState, None, None]:
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
            Generator of OptimizationState.
        """
        self.resolve_defaults()
        self._compile_plan()
        issues = self.validate()
        if issues:
            raise ConfigurationError(
                "Optimizer misconfigured:\n" + "\n".join(f"  - {m}" for m in issues)
            )
        self._inject_acquisition_directions()
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
    ) -> Generator[OptimizationState, None, None]:
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
        Generator[OptimizationState, None, None]
        """
        self._compile_plan()
        issues = self.validate(require_initializer=False)
        if issues:
            raise ConfigurationError(
                "Optimizer misconfigured:\n" + "\n".join(f"  - {m}" for m in issues)
            )
        self._inject_acquisition_directions()
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
        self._compile_plan()
        issues = self.validate(require_initializer=False)
        if issues:
            raise ConfigurationError(
                "Optimizer misconfigured:\n" + "\n".join(f"  - {m}" for m in issues)
            )
        self._inject_acquisition_directions()
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
        return state

    def __setstate__(self, state: dict[str, object]) -> None:
        """Restore an optimizer without a stale compiled execution plan."""
        self.__dict__.update(state)
        self.__dict__.setdefault("_executable_plan", None)
        self.__dict__.setdefault("_default_resolution", None)

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
        with open(p, "wb") as f:
            pickle.dump((self, ctx), f)

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
