from __future__ import annotations

import ast
import importlib
import inspect
import pkgutil
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import saealib
from saealib.core.compiler.diagnostics import DIAGNOSTIC_CODES
from saealib.core.component import Component
from saealib.core.contracts import (
    ASSUMPTION_KEYS,
    CARDINALITIES,
    DATA_SPEC_KINDS,
    ROLES,
    RUNTIME_CAPABILITIES,
    SCHEMA_VARIABLES,
    SERVICE_VOCABULARY,
    AssumptionSet,
    ComponentContract,
    DataSpec,
    ExecutionContract,
    Fixed,
    PartSpec,
    PortContract,
    PortDirection,
    PortSpec,
    Product,
    ServiceRequirement,
    StateContract,
    Var,
)
from saealib.core.state import SURROGATES_DEFAULT
from saealib.pipeline import Stage
from saealib.registry import build, to_spec
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
    PendingEvaluationContextStage,
    SortByScoreStage,
    StageStateViewAdapter,
    SurrogateFitStage,
    SurrogateOnlyLoopStage,
    SurrogatePredictStage,
    TellStage,
    TopKSelectionStage,
    TrueEvaluationStage,
)
from saealib.surrogate.rbf_kernels import GaussianKernel

_REGISTRY = saealib.registry._REGISTRY
_NON_CLASS_REGISTRY_NAMES = frozenset(
    {"max_fe", "max_gen", "max_steps", "f_target", "stalled"}
)
_OPTIONAL_IMPORT_MODULES = frozenset(
    {
        "saealib.algorithms.pymoo_algorithm",
        "saealib.operators.pymoo_crossover",
        "saealib.operators.pymoo_mutation",
        "saealib.problem.pymoo_problem",
        "saealib.surrogate.sklearn_surrogate",
        "saealib.surrogate.torch_surrogate",
    }
)
_DEAP_OPTIONAL_CONTRACT_MODULES = frozenset(
    {
        "saealib.algorithms.deap_algorithm",
        "saealib.operators.deap_crossover",
        "saealib.operators.deap_mutation",
    }
)
try:
    importlib.import_module("deap")
except ImportError:
    _DEAP_AVAILABLE = False
else:
    _DEAP_AVAILABLE = True


class _PymooVariableStub:
    value = 0.5


class _PymooCrossoverStub:
    n_parents = 3
    n_offsprings = 4
    prob = _PymooVariableStub()

    def _do(
        self,
        problem: object,
        x: np.ndarray,
        *args: object,
        random_state: object = None,
        **kwargs: object,
    ) -> np.ndarray:
        return x


class _PymooMutationStub:
    def _do(
        self,
        problem: object,
        x: np.ndarray,
        *args: object,
        random_state: object = None,
        **kwargs: object,
    ) -> np.ndarray:
        return x


class _StageHeldComponent:
    def contract(self) -> ComponentContract:
        return ComponentContract()

    def ask(self, request: Any, state: Any) -> Any:
        del request, state
        return None

    def tell(self, feedback: Any, state: Any) -> Any:
        del feedback, state
        return None


def _operational_stage_instances() -> tuple[Stage, ...]:
    def held() -> Any:
        return _StageHeldComponent()

    return (
        CountGenerationStage(),
        AskStage(held()),
        SurrogatePredictStage(held()),
        PendingEvaluationContextStage(held()),
        AcquisitionStage(held()),
        SurrogateFitStage(held()),
        TopKSelectionStage(k=1),
        SortByScoreStage(),
        EvaluationPlanStage(),
        AsyncEvaluationSubmitStage(held(), held()),
        EvaluationSubmitStage(held()),
        EvaluationCollectStage(held()),
        EvaluationApplyStage(),
        EvaluationAcknowledgeStage(held()),
        TrueEvaluationStage(held()),
        ArchiveUpdateStage(),
        FeedbackStage(held()),
        TellStage(held()),
        SurrogateOnlyLoopStage(held(), held(), 0, acquisition=held()),
        InitializationStage(held(), held(), held()),
    )


_OPERATIONAL_STAGE_INSTANCES = _operational_stage_instances()


_IMPORT_FAILURES: dict[str, str] = {}


def _qualified_name(cls: type[Any]) -> str:
    return f"{cls.__module__}.{cls.__qualname__}"


def _discover_contract_classes() -> tuple[type[Any], ...]:
    discovered: dict[type[Any], None] = {}
    for module_info in pkgutil.walk_packages(saealib.__path__, saealib.__name__ + "."):
        if not _DEAP_AVAILABLE and module_info.name in _DEAP_OPTIONAL_CONTRACT_MODULES:
            continue
        try:
            module = importlib.import_module(module_info.name)
        except ImportError as error:
            _IMPORT_FAILURES[module_info.name] = str(error)
            continue
        for obj in vars(module).values():
            if (
                inspect.isclass(obj)
                and obj.__module__.startswith("saealib.")
                and (
                    _DEAP_AVAILABLE
                    or obj.__module__ not in _DEAP_OPTIONAL_CONTRACT_MODULES
                )
                and not getattr(obj, "_is_protocol", False)
                # Stage contracts are audited by the component inventory
                # tests.  They are a separate execution-boundary protocol,
                # not registry-built components; their constructors require
                # runtime services and must not enter this recipe discovery.
                and not issubclass(obj, Stage)
                and callable(getattr(obj, "contract", None))
                and obj.contract is not Component.contract
            ):
                discovered[obj] = None
    return tuple(sorted(discovered, key=_qualified_name))


_DISCOVERED_CONTRACT_CLASSES = _discover_contract_classes()
_CONTRACT_ABCS = tuple(
    cls for cls in _DISCOVERED_CONTRACT_CLASSES if inspect.isabstract(cls)
)
_DISCOVERED_CONCRETE_CLASSES = tuple(
    cls for cls in _DISCOVERED_CONTRACT_CLASSES if not inspect.isabstract(cls)
)


def _recipe_local_surrogate_manager() -> Any:
    return build(
        {
            "type": "LocalSurrogateManager",
            "params": {
                "surrogate": {
                    "type": "RBFSurrogate",
                    "params": {"kernel": GaussianKernel()},
                },
                "training_set": {
                    "type": "KNNObjectiveSet",
                    "params": {"n_neighbors": 50},
                },
            },
        }
    )


def _recipe_rbf_surrogate() -> Any:
    return build(
        {
            "type": "RBFSurrogate",
            "params": {"kernel": GaussianKernel()},
        }
    )


def _recipe_global_surrogate_manager() -> Any:
    from saealib.surrogate.manager import GlobalSurrogateManager

    return GlobalSurrogateManager(_recipe_rbf_surrogate())


def _recipe_composite_surrogate_manager() -> Any:
    from saealib.surrogate.manager import CompositeSurrogateManager

    return CompositeSurrogateManager({"default": _recipe_global_surrogate_manager()})


def _recipe_pairwise_surrogate_manager() -> Any:
    from sklearn.svm import SVC

    from saealib.surrogate.manager import PairwiseSurrogateManager
    from saealib.surrogate.sklearn_surrogate import SklearnClassificationSurrogate

    return PairwiseSurrogateManager(
        SklearnClassificationSurrogate(SVC(probability=True))
    )


def _recipe_per_objective_surrogate() -> Any:
    from saealib.surrogate.per_objective import PerObjectiveSurrogate

    return PerObjectiveSurrogate([_recipe_rbf_surrogate()])


def _recipe_sklearn_surrogate() -> Any:
    from sklearn.svm import SVR

    from saealib.surrogate.sklearn_surrogate import SklearnSurrogate

    return SklearnSurrogate(SVR())


def _recipe_sklearn_classification_surrogate() -> Any:
    from sklearn.svm import SVC

    from saealib.surrogate.sklearn_surrogate import SklearnClassificationSurrogate

    return SklearnClassificationSurrogate(SVC(probability=True))


def _recipe_torch_surrogate() -> Any:
    import torch

    from saealib.surrogate.torch_surrogate import TorchSurrogate

    return TorchSurrogate(torch.nn.Linear(5, 1))


def _recipe_composite_acquisition() -> Any:
    from saealib.acquisition.base import CompositeAcquisition
    from saealib.acquisition.mean import MeanPrediction

    return CompositeAcquisition(
        {"objective": MeanPrediction()}, lambda scores: scores[0]
    )


def _recipe_manager_switcher() -> Any:
    from saealib.surrogate.switching import ManagerSwitcher

    manager = _recipe_global_surrogate_manager()
    return ManagerSwitcher(manager, manager)


def _recipe_strategy_switcher() -> Any:
    from saealib.strategies.ib import IndividualBasedStrategy
    from saealib.surrogate.switching import StrategySwitcher

    strategy = IndividualBasedStrategy()
    return StrategySwitcher(strategy, strategy)


def _recipe_ga() -> Any:
    return build(
        {
            "type": "GA",
            "params": {
                "crossover": {
                    "type": "CrossoverBLXAlpha",
                    "params": {"prob": 0.7, "alpha": 0.4},
                },
                "mutation": {
                    "type": "MutationUniform",
                    "params": {"prob_var": 0.3},
                },
                "parent_selection": {"type": "SequentialSelection"},
                "survivor_selection": {"type": "TruncationSelection"},
            },
        }
    )


def _recipe_genome_ga() -> Any:
    from saealib.algorithms import GenomeGA
    from saealib.operators import OrderCrossover, SwapMutation
    from saealib.operators.selection import SequentialSelection, TruncationSelection

    return GenomeGA(
        OrderCrossover(),
        SwapMutation(),
        SequentialSelection(),
        TruncationSelection(),
    )


def _recipe_termination() -> Any:
    from saealib.termination import Termination, max_fe

    return Termination(max_fe(100))


def _recipe_lhs_initializer() -> Any:
    from saealib.execution.initializer import LHSInitializer

    return LHSInitializer(1, 1)


def _recipe_random_initializer() -> Any:
    from saealib.execution.initializer import RandomInitializer

    return RandomInitializer(1, 1)


def _recipe_genome_initializer() -> Any:
    from saealib.execution.initializer import GenomeInitializer

    return GenomeInitializer(0, 0)


def _recipe_sobol_initializer() -> Any:
    from saealib.execution.initializer import SobolInitializer

    return SobolInitializer(1, 1)


def _recipe_async_evaluation_scheduler() -> Any:
    from saealib.execution.evaluator import SerialEvaluator
    from saealib.execution.scheduler import AsyncEvaluationScheduler

    return AsyncEvaluationScheduler(SerialEvaluator())


def _recipe_pymoo_algorithm() -> Any:
    from saealib.algorithms.pymoo_algorithm import PymooAlgorithm

    instance = PymooAlgorithm.__new__(PymooAlgorithm)
    instance.allow_partial_tell = False
    return instance


def _recipe_stage_node_adapter() -> Any:
    from saealib.core.graph_builder import StageNodeAdapter
    from saealib.pipeline import Stage

    class _RecipeStage(Stage):
        def execute(self, state: Any) -> Any:
            return state

    return StageNodeAdapter(_RecipeStage(name="recipe_stage"))


def _recipe_stage_state_view_adapter() -> Any:
    from saealib.pipeline import Stage

    class _RecipeStage(Stage):
        def execute(self, state: Any) -> Any:
            return state

    return StageStateViewAdapter(_RecipeStage(name="recipe_stage"))


def _recipe_stage_contract_node_adapter() -> Any:
    from saealib.core.graph_builder import StageContractNodeAdapter

    return StageContractNodeAdapter(_operational_stage_instances()[1])


def _recipe_stage_part_node_adapter() -> Any:
    from saealib.core.graph_builder import StagePartNodeAdapter

    return StagePartNodeAdapter(_StageHeldComponent(), ComponentContract())


def _recipe_adapter_component() -> Any:
    from saealib.core.compiler.adapters import Adapter, AdapterComponent

    adapter = Adapter(
        name="recipe_adapter",
        source=DataSpec(kind="Population"),
        target=DataSpec(kind="Population"),
        lossless=True,
        auto_insertable=True,
    )
    return AdapterComponent(adapter=adapter)


def _build_qualified(path: str, **params: Any) -> Any:
    return build({"type": path, "params": params})


class _DeapCrossoverStub:
    def __call__(
        self, ind1: list[float], ind2: list[float]
    ) -> tuple[list[float], list[float]]:
        return ind1, ind2


class _DeapMutationStub:
    def __call__(self, individual: list[float]) -> tuple[list[float]]:
        return (individual,)


class _DeapStrategyStub:
    def generate(self, ind_init: Any) -> list[Any]:
        return [ind_init(np.zeros(1, dtype=float))]

    def update(self, population: list[Any]) -> None:
        del population


def _recipe_template_strategy() -> Any:
    from saealib.pipeline import Pipeline
    from saealib.strategies.base import TemplateStrategy

    return TemplateStrategy(Pipeline())


_RECIPES: dict[str, Callable[[], Any]] = {
    _qualified_name(
        _REGISTRY["LocalSurrogateManager"]
    ): _recipe_local_surrogate_manager,
    _qualified_name(_REGISTRY["RBFSurrogate"]): _recipe_rbf_surrogate,
    "saealib.acquisition.base.CompositeAcquisition": _recipe_composite_acquisition,
    (
        "saealib.surrogate.manager.CompositeSurrogateManager"
    ): _recipe_composite_surrogate_manager,
    (
        "saealib.surrogate.manager.GlobalSurrogateManager"
    ): _recipe_global_surrogate_manager,
    (
        "saealib.surrogate.manager.PairwiseSurrogateManager"
    ): _recipe_pairwise_surrogate_manager,
    (
        "saealib.surrogate.per_objective.PerObjectiveSurrogate"
    ): _recipe_per_objective_surrogate,
    (
        "saealib.surrogate.sklearn_surrogate.SklearnClassificationSurrogate"
    ): _recipe_sklearn_classification_surrogate,
    ("saealib.surrogate.sklearn_surrogate.SklearnSurrogate"): _recipe_sklearn_surrogate,
    "saealib.surrogate.switching.ManagerSwitcher": _recipe_manager_switcher,
    "saealib.surrogate.switching.StrategySwitcher": _recipe_strategy_switcher,
    "saealib.surrogate.torch_surrogate.TorchSurrogate": _recipe_torch_surrogate,
    _qualified_name(_REGISTRY["CrossoverBLXAlpha"]): lambda: build(
        {
            "type": "CrossoverBLXAlpha",
            "params": {"prob": 0.7, "alpha": 0.4},
        }
    ),
    _qualified_name(_REGISTRY["GA"]): _recipe_ga,
    "saealib.algorithms.genome_ga.GenomeGA": _recipe_genome_ga,
    _qualified_name(_REGISTRY["TopKEvaluation"]): lambda: build(
        {"type": "TopKEvaluation", "params": {"k": 1}}
    ),
    _qualified_name(_REGISTRY["RatioEvaluation"]): lambda: build(
        {"type": "RatioEvaluation", "params": {"ratio": 0.5}}
    ),
    _qualified_name(_REGISTRY["Termination"]): _recipe_termination,
    "saealib.execution.initializer.LHSInitializer": _recipe_lhs_initializer,
    "saealib.execution.initializer.RandomInitializer": _recipe_random_initializer,
    "saealib.execution.initializer.GenomeInitializer": _recipe_genome_initializer,
    "saealib.execution.initializer.SobolInitializer": _recipe_sobol_initializer,
    (
        "saealib.execution.scheduler.AsyncEvaluationScheduler"
    ): _recipe_async_evaluation_scheduler,
    "saealib.algorithms.pymoo_algorithm.PymooAlgorithm": _recipe_pymoo_algorithm,
    "saealib.core.graph_builder.StageNodeAdapter": _recipe_stage_node_adapter,
    "saealib.stages.StageStateViewAdapter": _recipe_stage_state_view_adapter,
    (
        "saealib.core.graph_builder.StageContractNodeAdapter"
    ): _recipe_stage_contract_node_adapter,
    "saealib.core.graph_builder.StagePartNodeAdapter": (
        _recipe_stage_part_node_adapter
    ),
    "saealib.core.compiler.adapters.AdapterComponent": _recipe_adapter_component,
    "saealib.strategies.base.TemplateStrategy": _recipe_template_strategy,
    _qualified_name(_REGISTRY["GenerationBasedStrategy"]): lambda: build(
        {"type": "GenerationBasedStrategy", "params": {"gen_ctrl": 5}}
    ),
    _qualified_name(_REGISTRY["PreSelectionStrategy"]): lambda: build(
        {
            "type": "PreSelectionStrategy",
            "params": {"n_candidates": 40, "n_select": 4},
        }
    ),
    "saealib.operators.crossover.CrossoverCategorical": lambda: _build_qualified(
        "saealib.operators.crossover.CrossoverCategorical", prob=1.0
    ),
    "saealib.operators.crossover.CrossoverIntegerSBX": lambda: _build_qualified(
        "saealib.operators.crossover.CrossoverIntegerSBX", prob=1.0, eta=20.0
    ),
    "saealib.operators.crossover.CrossoverOnePoint": lambda: _build_qualified(
        "saealib.operators.crossover.CrossoverOnePoint", prob=1.0
    ),
    "saealib.operators.crossover.CrossoverSBX": lambda: _build_qualified(
        "saealib.operators.crossover.CrossoverSBX", prob=1.0, eta=20.0
    ),
    "saealib.operators.crossover.CrossoverTwoPoint": lambda: _build_qualified(
        "saealib.operators.crossover.CrossoverTwoPoint", prob=1.0
    ),
    "saealib.operators.crossover.CrossoverUniform": lambda: _build_qualified(
        "saealib.operators.crossover.CrossoverUniform", prob=1.0
    ),
    "saealib.operators.mutation.MutationGaussian": lambda: _build_qualified(
        "saealib.operators.mutation.MutationGaussian", sigma=1.0
    ),
    "saealib.operators.mutation.MutationPolynomial": lambda: _build_qualified(
        "saealib.operators.mutation.MutationPolynomial", eta=20.0
    ),
    "saealib.operators.pymoo_crossover.PymooCrossover": lambda: _build_qualified(
        "saealib.operators.pymoo_crossover.PymooCrossover",
        operator=_PymooCrossoverStub(),
    ),
    "saealib.operators.pymoo_mutation.PymooMutation": lambda: _build_qualified(
        "saealib.operators.pymoo_mutation.PymooMutation",
        operator=_PymooMutationStub(),
    ),
    "saealib.operators.deap_crossover.DeapCrossover": lambda: _build_qualified(
        "saealib.operators.deap_crossover.DeapCrossover",
        operator=_DeapCrossoverStub(),
    ),
    "saealib.operators.deap_mutation.DeapMutation": lambda: _build_qualified(
        "saealib.operators.deap_mutation.DeapMutation",
        operator=_DeapMutationStub(),
    ),
    "saealib.algorithms.deap_algorithm.DeapGenerateUpdateAlgorithm": (
        lambda: _build_qualified(
            "saealib.algorithms.deap_algorithm.DeapGenerateUpdateAlgorithm",
            strategy=_DeapStrategyStub(),
        )
    ),
    "saealib.operators.selection.TournamentSelection": lambda: _build_qualified(
        "saealib.operators.selection.TournamentSelection", tournament_size=2
    ),
}
if not _DEAP_AVAILABLE:
    _RECIPES = {
        path: recipe
        for path, recipe in _RECIPES.items()
        if path.rsplit(".", 1)[0] not in _DEAP_OPTIONAL_CONTRACT_MODULES
    }


def _builtin_classes() -> dict[str, type[Any]]:
    return {
        name: obj
        for name, obj in _REGISTRY.items()
        if inspect.isclass(obj) and obj.__module__.startswith("saealib.")
    }


_BUILTIN_CLASSES = _builtin_classes()
_BUILTIN_CONSTRUCTION_FAILURES: dict[str, type[BaseException]] = {}
_REACHED_RECIPE_PATHS: set[str] = set()


def _run_recipe(path: str) -> Any | None:
    recipe = _RECIPES.get(path)
    if recipe is None:
        return None
    _REACHED_RECIPE_PATHS.add(path)
    return recipe()


def _construct_builtin(name: str) -> Any | None:
    try:
        return build(name)
    except (TypeError, ValueError) as error:
        cls = _BUILTIN_CLASSES[name]
        path = _qualified_name(cls)
        _BUILTIN_CONSTRUCTION_FAILURES[path] = type(error)
        return _run_recipe(path)


_BUILTIN_INSTANCES = tuple(
    (name, _construct_builtin(name)) for name in _BUILTIN_CLASSES
)

_DISCOVERY_CONSTRUCTION_FAILURES: dict[str, type[BaseException]] = {}


def _construct_discovered(cls: type[Any]) -> Any | None:
    path = _qualified_name(cls)
    try:
        return cls()
    except (TypeError, ValueError) as error:
        _DISCOVERY_CONSTRUCTION_FAILURES[path] = type(error)
        return _run_recipe(path)


_DISCOVERED_INSTANCES = tuple(
    (cls, _construct_discovered(cls)) for cls in _DISCOVERED_CONCRETE_CLASSES
)


def _implements_contract(instance: Any) -> bool:
    method = getattr(type(instance), "contract", None)
    return callable(method) and method is not Component.contract


_BUILTIN_COMPONENTS = tuple(
    (name, instance)
    for name, instance in _BUILTIN_INSTANCES
    if instance is not None and _implements_contract(instance)
)

_IMPLEMENTED_COMPONENTS = tuple(
    (_qualified_name(cls), instance)
    for cls, instance in _DISCOVERED_INSTANCES
    if instance is not None and _implements_contract(instance)
)

_CONTRACT_INVARIANT_INSTANCES = _IMPLEMENTED_COMPONENTS + tuple(
    (
        f"{type(stage).__module__}.{type(stage).__qualname__}",
        stage,
    )
    for stage in _OPERATIONAL_STAGE_INSTANCES
)


def _freeze(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return (value.dtype.str, value.shape, value.tobytes())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return tuple(
            sorted(
                ((repr(key), _freeze(item)) for key, item in value.items()),
                key=lambda item: item[0],
            )
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze(item) for item in value)
    try:
        hash(value)
    except TypeError:
        return repr(value)
    return value


def _snapshot(instance: Any) -> Any:
    try:
        return ("to_spec", _freeze(to_spec(instance)))
    except Exception:
        for name in ("state_export", "export_state"):
            exporter = getattr(instance, name, None)
            if callable(exporter):
                try:
                    return (name, _freeze(exporter()))
                except Exception:
                    break
        return ("vars", _freeze(vars(instance)))


def _check_purity(instance: Any) -> None:
    before = _snapshot(instance)
    first = instance.contract()
    between = _snapshot(instance)
    second = instance.contract()
    after = _snapshot(instance)
    assert first == second
    assert before == between == after


class _PoisonedGenerator:
    def __getattr__(self, name: str) -> Any:
        raise AssertionError(f"contract accessed poisoned RNG attribute {name!r}")


@contextmanager
def _poisoned_generators(root: Any) -> Iterator[None]:
    replacements: list[tuple[Any, str, np.random.Generator]] = []
    visited: set[int] = set()

    def visit(value: Any) -> None:
        if id(value) in visited:
            return
        visited.add(id(value))
        try:
            attributes = tuple(vars(value).items())
        except TypeError:
            return
        for name, attribute in attributes:
            if isinstance(attribute, np.random.Generator):
                setattr(value, name, _PoisonedGenerator())
                replacements.append((value, name, attribute))
            elif callable(getattr(attribute, "contract", None)):
                visit(attribute)

    try:
        visit(root)
        yield
    finally:
        for owner, name, generator in reversed(replacements):
            setattr(owner, name, generator)


def _check_state_free(instance: Any) -> None:
    with _poisoned_generators(instance):
        instance.contract()


@contextmanager
def _record_abc_contract_calls(
    abc: type[Any],
) -> Iterator[set[type[Any]]]:
    original = abc.contract
    had_local_contract = "contract" in abc.__dict__
    called_by: set[type[Any]] = set()

    def recording_contract(instance: Any, *args: Any, **kwargs: Any) -> Any:
        called_by.add(type(instance))
        return original(instance, *args, **kwargs)

    setattr(abc, "contract", recording_contract)
    try:
        yield called_by
    finally:
        if had_local_contract:
            setattr(abc, "contract", original)
        else:
            delattr(abc, "contract")


def _check_port_names(instance: Any) -> None:
    contract = instance.contract()
    for role_contract in contract.ports.values():
        for ports in (role_contract.inputs, role_contract.outputs):
            names = tuple(port.name for port in ports)
            assert all(isinstance(name, str) and name for name in names)
            assert len(names) == len(set(names))
    for part in contract.parts:
        child = getattr(instance, part.name)
        _check_port_names(child)


def _check_vocabularies(instance: Any) -> None:
    contract = instance.contract()
    for role, role_contract in contract.ports.items():
        assert ROLES.get(role) is not None
        for port in (*role_contract.inputs, *role_contract.outputs):
            assert DATA_SPEC_KINDS.get(port.data.kind) is not None
            assert CARDINALITIES.get(port.cardinality) is not None
            for variable in port.data.bindings:
                assert SCHEMA_VARIABLES.get(variable) is not None
            for service in port.required_services:
                assert SERVICE_VOCABULARY.get(service.name) is not None
    for name in contract.assumptions:
        assert ASSUMPTION_KEYS.get(name) is not None
    for capability in (
        *contract.execution.required_runtime_capabilities,
        *contract.execution.offered_runtime_capabilities,
    ):
        assert RUNTIME_CAPABILITIES.get(capability) is not None
    for part in contract.parts:
        child = getattr(instance, part.name)
        _check_vocabularies(child)


def _check_parts(instance: Any) -> None:
    contract = instance.contract()
    for part in contract.parts:
        assert isinstance(part.contract, ComponentContract)
        try:
            child = getattr(instance, part.name)
        except AttributeError as error:
            raise AssertionError(f"missing part attribute {part.name!r}") from error
        assert callable(getattr(child, "contract", None))
        assert part.contract == child.contract()
        _check_parts(child)


def _port(name: str, direction: PortDirection) -> PortSpec:
    return PortSpec(
        name=name,
        direction=direction,
        data=DataSpec(kind="GenomeBatch"),
        cardinality="ONE",
    )


class _PureDummy:
    def __init__(self, value: int = 1) -> None:
        self.value = value

    def contract(self) -> ComponentContract:
        return ComponentContract()


class _ImpureDummy:
    def __init__(self, value: int = 1) -> None:
        self.value = value

    def contract(self) -> ComponentContract:
        self.value += 1
        return ComponentContract()


class _StateFreeChild:
    def __init__(self) -> None:
        self.rng = np.random.default_rng(0)

    def contract(self) -> ComponentContract:
        return ComponentContract()


class _StateReadingChild(_StateFreeChild):
    def contract(self) -> ComponentContract:
        self.rng.random()
        return ComponentContract()


class _StateFreeParent:
    def __init__(self) -> None:
        self.child = _StateFreeChild()

    def contract(self) -> ComponentContract:
        return self.child.contract()


class _StateReadingParent:
    def __init__(self) -> None:
        self.child = _StateReadingChild()

    def contract(self) -> ComponentContract:
        return self.child.contract()


class _PortNamesPassDummy:
    def contract(self) -> ComponentContract:
        port = _port("input", PortDirection.INPUT)
        return ComponentContract(
            ports={
                "proposer": PortContract(inputs=(port,)),
                "predictor": PortContract(inputs=(port,)),
            }
        )


class _PortNamesFailDummy:
    def contract(self) -> ComponentContract:
        port = _port("input", PortDirection.INPUT)
        role_contract = PortContract(inputs=(port,))
        object.__setattr__(role_contract, "inputs", (port, port))
        return ComponentContract(ports={"proposer": role_contract})


class _VocabularyChild:
    def contract(self) -> ComponentContract:
        return ComponentContract()


class _VocabularyPassDummy:
    def __init__(self) -> None:
        self.child = _VocabularyChild()

    def contract(self) -> ComponentContract:
        return ComponentContract(
            ports=_vocabulary_contract().ports,
            parts=(PartSpec(name="child", contract=self.child.contract()),),
            assumptions=_vocabulary_contract().assumptions,
        )


class _VocabularyDummy:
    def __init__(self, component_contract: ComponentContract) -> None:
        self.component_contract = component_contract

    def contract(self) -> ComponentContract:
        return self.component_contract


def _vocabulary_contract(
    *,
    role: str = "proposer",
    kind: str = "GenomeBatch",
    bindings: Mapping[str, Any] | None = None,
    cardinality: str = "ONE",
    services: tuple[str, ...] = ("SamplingService",),
    assumptions: Mapping[str, bool] | None = None,
    capabilities: tuple[str, ...] = (),
) -> ComponentContract:
    if bindings is None:
        bindings = {
            "representation": Product(
                elements=(
                    Var(name="feature_schema"),
                    Product(
                        elements=(
                            Var(name="objective_schema"),
                            Fixed(value="dense"),
                        )
                    ),
                )
            )
        }
    if assumptions is None:
        assumptions = {"evaluation.deterministic": False}
    port = PortSpec(
        name="input",
        direction=PortDirection.INPUT,
        data=DataSpec(kind=kind, bindings=bindings),
        cardinality=cardinality,
        required_services=tuple(
            ServiceRequirement(name=service) for service in services
        ),
    )
    return ComponentContract(
        ports={role: PortContract(inputs=(port,))},
        execution=ExecutionContract(required_runtime_capabilities=capabilities),
        assumptions=AssumptionSet(assumptions),
    )


def _unknown_role_contract() -> ComponentContract:
    return _vocabulary_contract(role="unknown_role")


def _unknown_data_kind_contract() -> ComponentContract:
    return _vocabulary_contract(kind="unknown_kind")


def _unknown_cardinality_contract() -> ComponentContract:
    return _vocabulary_contract(cardinality="UNKNOWN_CARDINALITY")


def _unknown_top_level_schema_variable_contract() -> ComponentContract:
    return _vocabulary_contract(
        bindings={
            "unknown_variable": Product(
                elements=(
                    Var(name="anything"),
                    Fixed(value="dense"),
                )
            )
        }
    )


def _unknown_required_service_contract() -> ComponentContract:
    return _vocabulary_contract(services=("UnknownService",))


def _unknown_assumption_contract() -> ComponentContract:
    return _vocabulary_contract(assumptions={"unknown.assumption": True})


def _unknown_runtime_capability_contract() -> ComponentContract:
    return _vocabulary_contract(capabilities=("missing_capability",))


class _PartsChild:
    def contract(self) -> ComponentContract:
        return ComponentContract(
            assumptions=AssumptionSet({"evaluation.deterministic": False})
        )


class _PartsPassDummy:
    def __init__(self) -> None:
        self.child = _PartsChild()

    def contract(self) -> ComponentContract:
        return ComponentContract(
            parts=(PartSpec(name="child", contract=self.child.contract()),)
        )


class _PartsMismatchDummy(_PartsPassDummy):
    def contract(self) -> ComponentContract:
        return ComponentContract(
            parts=(PartSpec(name="child", contract=ComponentContract()),)
        )


class _PartsMissingDummy:
    def contract(self) -> ComponentContract:
        return ComponentContract(
            parts=(PartSpec(name="child", contract=ComponentContract()),)
        )


def test_builtin_contract_implementation_set_is_not_empty() -> None:
    assert _BUILTIN_COMPONENTS


def test_discovered_contract_implementation_set_is_not_empty() -> None:
    assert _IMPLEMENTED_COMPONENTS


def test_discovered_contract_classes_are_unique_by_identity() -> None:
    assert len(_DISCOVERED_CONTRACT_CLASSES) == len(
        {id(cls) for cls in _DISCOVERED_CONTRACT_CLASSES}
    )


def test_import_failures_are_allowlisted_optional_modules() -> None:
    assert set(_IMPORT_FAILURES) <= _OPTIONAL_IMPORT_MODULES


def test_abstract_contract_classes_are_excluded_from_instance_checks() -> None:
    assert all(
        not inspect.isabstract(type(instance))
        for _, instance in _IMPLEMENTED_COMPONENTS
    )


@pytest.mark.parametrize(
    "abc",
    _CONTRACT_ABCS,
    ids=[_qualified_name(cls) for cls in _CONTRACT_ABCS],
)
def test_each_contract_abc_is_exercised_by_discovered_subclasses(
    abc: type[Any],
) -> None:
    instances = tuple(
        instance
        for cls, instance in _DISCOVERED_INSTANCES
        if issubclass(cls, abc) and instance is not None
    )
    with _record_abc_contract_calls(abc) as called_by:
        for instance in instances:
            instance.contract()
    assert called_by


def test_non_class_registry_entries_are_explicitly_allowed() -> None:
    actual = {
        name
        for name, obj in _REGISTRY.items()
        if not inspect.isclass(obj)
        and getattr(obj, "__module__", "").startswith("saealib.")
    }
    assert actual == _NON_CLASS_REGISTRY_NAMES


def test_every_builtin_class_has_a_construction_path() -> None:
    constructed = {
        name for name, instance in _BUILTIN_INSTANCES if instance is not None
    }
    builtin_paths = {_qualified_name(cls) for cls in _BUILTIN_CLASSES.values()}
    discovered_paths = {_qualified_name(cls) for cls in _DISCOVERED_CONTRACT_CLASSES}
    assert set(_BUILTIN_CONSTRUCTION_FAILURES) <= set(_RECIPES)
    assert set(_RECIPES) <= builtin_paths | discovered_paths
    assert constructed == set(_BUILTIN_CLASSES)


def test_every_discovered_contract_class_has_a_construction_path() -> None:
    constructed = {
        _qualified_name(cls)
        for cls, instance in _DISCOVERED_INSTANCES
        if instance is not None
    }
    expected = {_qualified_name(cls) for cls in _DISCOVERED_CONCRETE_CLASSES}
    assert set(_DISCOVERY_CONSTRUCTION_FAILURES) <= set(_RECIPES)
    assert constructed == expected


def test_every_recipe_is_reached_after_no_arg_construction_attempt() -> None:
    assert set(_RECIPES) == _REACHED_RECIPE_PATHS


@pytest.mark.parametrize(
    ("name", "instance"),
    _CONTRACT_INVARIANT_INSTANCES,
    ids=[name for name, _ in _CONTRACT_INVARIANT_INSTANCES],
)
def test_discovered_contract_is_state_free(name: str, instance: Any) -> None:
    del name
    _check_state_free(instance)


def test_surrogate_contract_exports_model_state_only() -> None:
    contract = _recipe_rbf_surrogate().contract()
    assert contract.state == StateContract(exports=(SURROGATES_DEFAULT,))


@pytest.mark.parametrize(
    ("name", "instance"),
    _CONTRACT_INVARIANT_INSTANCES,
    ids=[name for name, _ in _CONTRACT_INVARIANT_INSTANCES],
)
def test_discovered_contract_is_pure(name: str, instance: Any) -> None:
    del name
    _check_purity(instance)


@pytest.mark.parametrize(
    ("name", "instance"),
    _CONTRACT_INVARIANT_INSTANCES,
    ids=[name for name, _ in _CONTRACT_INVARIANT_INSTANCES],
)
def test_discovered_port_names_are_valid(name: str, instance: Any) -> None:
    del name
    _check_port_names(instance)


@pytest.mark.parametrize(
    ("name", "instance"),
    _CONTRACT_INVARIANT_INSTANCES,
    ids=[name for name, _ in _CONTRACT_INVARIANT_INSTANCES],
)
def test_discovered_contract_vocabulary_references_are_registered(
    name: str, instance: Any
) -> None:
    del name
    _check_vocabularies(instance)


@pytest.mark.parametrize(
    ("name", "instance"),
    _CONTRACT_INVARIANT_INSTANCES,
    ids=[name for name, _ in _CONTRACT_INVARIANT_INSTANCES],
)
def test_discovered_parts_match_held_components(name: str, instance: Any) -> None:
    del name
    _check_parts(instance)


def test_purity_check_accepts_dummy() -> None:
    _check_purity(_PureDummy())


def test_purity_check_rejects_dummy() -> None:
    with pytest.raises(AssertionError):
        _check_purity(_ImpureDummy())


def test_state_free_check_accepts_nested_dummy() -> None:
    _check_state_free(_StateFreeParent())


def test_state_free_check_rejects_nested_dummy() -> None:
    with pytest.raises(AssertionError):
        _check_state_free(_StateReadingParent())


def test_port_name_check_accepts_same_name_in_distinct_roles() -> None:
    _check_port_names(_PortNamesPassDummy())


def test_port_name_check_rejects_duplicate_name_in_direction() -> None:
    with pytest.raises(AssertionError):
        _check_port_names(_PortNamesFailDummy())


def test_vocabulary_check_accepts_nested_registered_references() -> None:
    _check_vocabularies(_VocabularyPassDummy())


def test_vocabulary_check_ignores_nested_variable_names() -> None:
    _check_vocabularies(
        _VocabularyDummy(
            _vocabulary_contract(
                bindings={
                    "representation": Product(
                        elements=(
                            Var(name="anything"),
                            Fixed(value="dense"),
                        )
                    )
                }
            )
        )
    )


@pytest.mark.parametrize(
    ("vocabulary_name", "contract_factory"),
    [
        ("role", _unknown_role_contract),
        ("data kind", _unknown_data_kind_contract),
        ("cardinality", _unknown_cardinality_contract),
        ("top-level schema variable", _unknown_top_level_schema_variable_contract),
        ("required service", _unknown_required_service_contract),
        ("assumption key", _unknown_assumption_contract),
        ("runtime capability", _unknown_runtime_capability_contract),
    ],
)
def test_vocabulary_check_rejects_each_unregistered_reference(
    vocabulary_name: str, contract_factory: Callable[[], ComponentContract]
) -> None:
    del vocabulary_name
    with pytest.raises(AssertionError):
        _check_vocabularies(_VocabularyDummy(contract_factory()))


def test_parts_check_accepts_matching_child_contract() -> None:
    _check_parts(_PartsPassDummy())


def test_parts_check_rejects_mismatched_child_contract() -> None:
    with pytest.raises(AssertionError):
        _check_parts(_PartsMismatchDummy())


def test_parts_check_rejects_missing_child_attribute() -> None:
    with pytest.raises(AssertionError):
        _check_parts(_PartsMissingDummy())


def test_custom_diagnostics_have_paths_and_registered_codes() -> None:
    """Every compiler diagnostic declares a location, resolution, and code."""
    compiler_root = Path(saealib.__file__).parent / "core" / "compiler"
    literal_codes: set[str] = set()
    diagnostic_calls = 0

    for source_path in compiler_root.glob("*.py"):
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            function = node.func
            if not (
                (isinstance(function, ast.Name) and function.id == "Diagnostic")
                or (
                    isinstance(function, ast.Attribute)
                    and function.attr == "Diagnostic"
                )
            ):
                continue
            diagnostic_calls += 1
            keywords = {keyword.arg for keyword in node.keywords}
            assert "path" in keywords, f"missing path in {source_path}:{node.lineno}"
            assert "resolutions" in keywords, (
                f"missing resolutions in {source_path}:{node.lineno}"
            )
            for keyword in node.keywords:
                if (
                    keyword.arg == "code"
                    and isinstance(keyword.value, ast.Constant)
                    and isinstance(keyword.value.value, str)
                ):
                    literal_codes.add(keyword.value.value)

    assert diagnostic_calls > 0
    assert all(DIAGNOSTIC_CODES.get(code) is not None for code in literal_codes)
