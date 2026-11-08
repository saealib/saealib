"""High-level API: minimize / maximize functions."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

import numpy as np

from saealib.acquisition.base import AcquisitionFunction
from saealib.callback import GenerationStartEvent, logging_generation
from saealib.exceptions import ValidationError
from saealib.execution.initializer import LHSInitializer
from saealib.optimizer import Optimizer, _default_acquisition_spec
from saealib.problem import Problem
from saealib.result import Result
from saealib.strategies.gb import GenerationBasedStrategy
from saealib.strategies.ib import IndividualBasedStrategy
from saealib.strategies.ps import PreSelectionStrategy
from saealib.surrogate.manager import SurrogateManager
from saealib.termination import Termination
from saealib.termination import max_fe as max_fe_cond

if TYPE_CHECKING:
    from pathlib import Path

    from saealib.algorithms.base import Algorithm
    from saealib.strategies.base import OptimizationStrategy
    from saealib.surrogate.base import Surrogate


# Sentinel distinguishing "argument omitted" (defer to Optimizer's own
# default resolution) from an explicit ``None`` passed by the caller (which
# keeps its pre-existing meaning, e.g. surrogate=None raises ValidationError).
class _UnsetType:
    def __repr__(self) -> str:
        return "UNSET"


_UNSET = _UnsetType()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_STRATEGY_TYPE_NAMES = {
    "ib": "IndividualBasedStrategy",
    "gb": "GenerationBasedStrategy",
    "ps": "PreSelectionStrategy",
}


def _resolve_direction(
    direction: np.ndarray | list[str] | None,
    n_obj: int,
    default: float,
) -> np.ndarray:
    """Convert direction argument to a ±1 float array."""
    if direction is None:
        return np.full(n_obj, default)
    if isinstance(direction, np.ndarray):
        return direction
    _map = {"minimize": -1.0, "maximize": 1.0}
    try:
        return np.array([_map[d] for d in direction])
    except KeyError as e:
        raise ValidationError(
            f"Unknown direction {e}. Use 'minimize' or 'maximize'."
        ) from e


def _ensure_problem(
    func: Callable | Problem,
    dim: int | None,
    lb,
    ub,
    n_obj: int,
    direction: np.ndarray | list[str] | None,
    default_direction: float,
) -> Problem:
    """Return a Problem, building one from a callable if needed."""
    if isinstance(func, Problem):
        if direction is not None:
            raise ValidationError(
                "direction cannot be passed when func is a Problem; configure the "
                "Problem's own direction."
            )
        return func
    if dim is None or lb is None or ub is None:
        raise ValidationError("dim, lb, and ub are required when func is a callable.")
    direction_arr = _resolve_direction(direction, n_obj, default=default_direction)
    return Problem(
        func=func, dim=dim, n_obj=n_obj, direction=direction_arr, lb=lb, ub=ub
    )


def _resolve_algorithm(
    algorithm: str | Algorithm | None, preset: dict, problem: Problem
) -> Algorithm | None:
    if isinstance(algorithm, str):
        from saealib.registry import _inject_params, build, get

        name = algorithm.upper()
        if name not in ("GA", "PSO"):
            raise ValidationError(
                f"Unknown algorithm: {algorithm!r}. "
                "Use 'GA', 'PSO', or an Algorithm instance."
            )
        spec = preset.get("algorithm")
        if isinstance(spec, dict) and spec.get("type") == name:
            return build(
                _inject_params(spec, dim=problem.dim, direction=problem.direction)
            )
        return get(name)()
    return algorithm


def _resolve_surrogate(
    surrogate: str | Surrogate | SurrogateManager | None,
    problem: Problem,
    preset: dict,
) -> tuple[SurrogateManager, AcquisitionFunction | None]:
    if surrogate is None:
        raise ValidationError(
            "surrogate=None is not supported. "
            "Use 'rbf' or a Surrogate/SurrogateManager instance."
        )
    if isinstance(surrogate, SurrogateManager):
        from saealib.registry import build

        acquisition = build(
            _default_acquisition_spec(type(surrogate), problem.direction)
        )
        return surrogate, acquisition
    if isinstance(surrogate, str):
        if surrogate.lower() == "rbf":
            return Optimizer._build_components_from_spec_static(
                preset["surrogate_manager"], problem.dim, problem.direction
            )
        raise ValidationError(
            f"Unknown surrogate: {surrogate!r}. "
            "Use 'rbf' or a Surrogate/SurrogateManager instance."
        )
    return Optimizer._build_components_from_spec_static(
        preset["surrogate_manager"],
        problem.dim,
        problem.direction,
        surrogate=surrogate,
    )


def _resolve_strategy(
    strategy: str | OptimizationStrategy | None,
    pop_size: int,
    preset: dict,
    problem: Problem,
) -> OptimizationStrategy | None:
    if not isinstance(strategy, str):
        return strategy
    name = strategy.lower()
    type_name = _STRATEGY_TYPE_NAMES.get(name)
    if type_name is None:
        raise ValidationError(
            f"Unknown strategy: {strategy!r}. "
            "Use 'ib', 'gb', 'ps', or an OptimizationStrategy instance."
        )
    spec = preset.get("strategy")
    if isinstance(spec, dict) and spec.get("type") == type_name:
        from saealib.registry import _inject_params, build

        return build(_inject_params(spec, dim=problem.dim, direction=problem.direction))
    if name == "ib":
        return IndividualBasedStrategy()
    if name == "gb":
        return GenerationBasedStrategy(gen_ctrl=5)
    if name == "ps":
        n_select = max(1, pop_size // 10)
        return PreSelectionStrategy(n_candidates=pop_size, n_select=n_select)


def _run(
    problem: Problem,
    algorithm: str | Algorithm | None,
    surrogate: str | Surrogate | SurrogateManager | None,
    strategy: str | OptimizationStrategy | None,
    max_fe: int | None,
    pop_size: int | None,
    seed: int | None,
    verbose: bool,
    preset: str | Path | dict | None,
    history_channels: Sequence[str] | None,
) -> Result:
    dim = problem.dim

    from saealib.defaults import load_defaults
    from saealib.defaults.loader import select_preset_name

    if algorithm is _UNSET or algorithm is None:
        algorithm_name = None
    elif isinstance(algorithm, str):
        algorithm_name = algorithm.upper()
    else:
        algorithm_name = type(algorithm).__name__
    defaults = load_defaults()
    preset_name = select_preset_name(defaults, problem, algorithm_name)
    bundled_preset = defaults["presets"][preset_name]

    opt = Optimizer(problem, seed=seed)
    if pop_size is not None:
        initializer = LHSInitializer(
            n_init_archive=max(5 * dim, pop_size),
            n_init_population=pop_size,
            seed=seed,
        )
        opt.set_initializer(initializer)
    if max_fe is not None:
        opt.set_termination(Termination(max_fe_cond(max_fe)))
    if history_channels is not None:
        opt.set_history(history_channels)

    if preset is not None:
        opt.set_preset(preset)

    # Arguments left at _UNSET are not passed to set_*(); Optimizer.run()'s
    # resolve_defaults() then fills them from the library's bundled preset.
    if algorithm is not _UNSET:
        # An explicit None resolves to None here, same as never calling
        # set_algorithm(); Optimizer.resolve_defaults() then fills it in.
        opt.set_algorithm(_resolve_algorithm(algorithm, bundled_preset, problem))  # type: ignore
    if surrogate is not _UNSET:
        manager, acquisition = _resolve_surrogate(surrogate, problem, bundled_preset)
        opt.set_surrogate_manager(manager)
        if opt.acquisition is None and acquisition is not None:
            opt.set_acquisition(acquisition)
    if strategy is not _UNSET:
        opt.set_strategy(
            _resolve_strategy(
                strategy,
                pop_size if pop_size is not None else 4 * dim,
                bundled_preset,
                problem,
            )  # type: ignore
        )

    if not verbose:
        opt.cbmanager.unregister(GenerationStartEvent, logging_generation)

    return Result.from_state(opt.run())


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def minimize(
    func: Callable | Problem,
    algorithm: str | Algorithm | None = _UNSET,  # type: ignore  # sentinel default
    *,
    dim: int | None = None,
    lb=None,
    ub=None,
    n_obj: int = 1,
    direction: np.ndarray | list[str] | None = None,
    surrogate: str | Surrogate | SurrogateManager | None = _UNSET,  # type: ignore
    strategy: str | OptimizationStrategy | None = _UNSET,  # type: ignore
    preset: str | Path | dict | None = None,
    max_fe: int | None = None,
    pop_size: int | None = None,
    seed: int | None = None,
    history_channels: Sequence[str] | None = None,
    verbose: bool = True,
) -> Result:
    """Run surrogate-assisted minimization.

    Parameters
    ----------
    func : callable or Problem
        Objective function ``f(x) -> float | array``, or a fully configured
        :class:`Problem` instance (in which case ``dim``, ``lb``, ``ub``, and
        ``n_obj`` are ignored; its own ``direction`` is respected, and passing
        ``direction`` is invalid).
    algorithm : str, Algorithm, or None
        ``'GA'``, ``'PSO'``, or an :class:`Algorithm` instance. If omitted,
        the library's bundled default preset resolves it (currently GA with
        BLX-alpha crossover and uniform mutation).
    dim : int, optional
        Number of design variables. Required when *func* is a callable.
    lb : array-like, optional
        Lower bounds of length *dim*. Required when *func* is a callable.
    ub : array-like, optional
        Upper bounds of length *dim*. Required when *func* is a callable.
    n_obj : int
        Number of objectives. Ignored when *func* is a :class:`Problem`. Default: 1.
    direction : np.ndarray, list[str], or None
        Optimization direction per objective. Each element is ``-1``/``"minimize"``
        (minimize) or ``+1``/``"maximize"`` (maximize). Default: all minimize for
        callable objectives. A :class:`Problem` uses its own direction.
    surrogate : str, Surrogate, SurrogateManager, or None
        ``'rbf'``, a :class:`Surrogate`, or a :class:`SurrogateManager`. If
        omitted, the library's bundled default preset resolves it (currently
        an RBF surrogate with mean-prediction acquisition). Passing ``None``
        explicitly is not supported and raises :class:`ValidationError`.
    strategy : str, OptimizationStrategy, or None
        ``'ib'``, ``'gb'``, ``'ps'``, or an :class:`OptimizationStrategy`. If
        omitted, the library's bundled default preset resolves it (currently
        individual-based).
    preset : str, Path, dict, or None, optional
        A preset (YAML file path or dict) providing default component
        configuration. See :meth:`Optimizer.set_preset`. Components explicitly
        passed via *algorithm*/*surrogate*/*strategy* still take precedence.
    max_fe : int or None
        Maximum true function evaluations. If omitted, semantic defaults or the
        ``200 * dim`` fallback are used.
    pop_size : int or None
        Population size. Semantic defaults from the configured composition are
        used when omitted, with ``4 * dim`` as the fallback.
    seed : int or None
        Random seed for :class:`LHSInitializer`.
    history_channels : sequence of str or None
        Execution history channels to record. ``None`` keeps the default
        summary history. Include ``"evaluation"`` to record every true
        evaluation result.
    verbose : bool
        If ``False``, suppress per-generation log output. Default: ``True``.

    Returns
    -------
    Result

    Examples
    --------
    >>> from saealib import minimize
    >>> import numpy as np
    >>> result = minimize(lambda x: np.sum(x**2), dim=5, lb=[-5]*5, ub=[5]*5,
    ...                   max_fe=500, seed=0, verbose=False)
    >>> result.x, result.f
    """
    problem = _ensure_problem(
        func, dim, lb, ub, n_obj, direction, default_direction=-1.0
    )
    return _run(
        problem,
        algorithm,
        surrogate,
        strategy,
        max_fe,
        pop_size,
        seed,
        verbose,
        preset,
        history_channels,
    )


def maximize(
    func: Callable | Problem,
    algorithm: str | Algorithm | None = _UNSET,  # type: ignore  # sentinel default
    *,
    dim: int | None = None,
    lb=None,
    ub=None,
    n_obj: int = 1,
    direction: np.ndarray | list[str] | None = None,
    surrogate: str | Surrogate | SurrogateManager | None = _UNSET,  # type: ignore
    strategy: str | OptimizationStrategy | None = _UNSET,  # type: ignore
    preset: str | Path | dict | None = None,
    max_fe: int | None = None,
    pop_size: int | None = None,
    seed: int | None = None,
    history_channels: Sequence[str] | None = None,
    verbose: bool = True,
) -> Result:
    """Run surrogate-assisted maximization.

    Identical to :func:`minimize` except that all objectives are maximized
    (``direction = +1``) for callable objectives. When a :class:`Problem` is
    passed, its own direction is respected.

    Parameters
    ----------
    func : callable or Problem
        Objective function ``f(x) -> float | array``, or a fully configured
        :class:`Problem` instance (in which case ``dim``, ``lb``, ``ub``, and
        ``n_obj`` are ignored; its own ``direction`` is respected, and passing
        ``direction`` is invalid).
    algorithm : str, Algorithm, or None
        ``'GA'``, ``'PSO'``, or an :class:`Algorithm` instance. If omitted,
        the library's bundled default preset resolves it (currently GA with
        BLX-alpha crossover and uniform mutation).
    dim : int, optional
        Number of design variables. Required when *func* is a callable.
    lb : array-like, optional
        Lower bounds of length *dim*. Required when *func* is a callable.
    ub : array-like, optional
        Upper bounds of length *dim*. Required when *func* is a callable.
    n_obj : int
        Number of objectives. Ignored when *func* is a :class:`Problem`. Default: 1.
    direction : np.ndarray, list[str], or None
        Optimization direction per objective. Each element is ``-1``/``"minimize"``
        (minimize) or ``+1``/``"maximize"`` (maximize). Default: all maximize for
        callable objectives. A :class:`Problem` uses its own direction.
    surrogate : str, Surrogate, SurrogateManager, or None
        ``'rbf'``, a :class:`Surrogate`, or a :class:`SurrogateManager`. If
        omitted, the library's bundled default preset resolves it (currently
        an RBF surrogate with mean-prediction acquisition). Passing ``None``
        explicitly is not supported and raises :class:`ValidationError`.
    strategy : str, OptimizationStrategy, or None
        ``'ib'``, ``'gb'``, ``'ps'``, or an :class:`OptimizationStrategy`. If
        omitted, the library's bundled default preset resolves it (currently
        individual-based).
    preset : str, Path, dict, or None, optional
        A preset (YAML file path or dict) providing default component
        configuration. See :meth:`Optimizer.set_preset`. Components explicitly
        passed via *algorithm*/*surrogate*/*strategy* still take precedence.
    max_fe : int or None
        Maximum true function evaluations. If omitted, semantic defaults or the
        ``200 * dim`` fallback are used.
    pop_size : int or None
        Population size. Semantic defaults from the configured composition are
        used when omitted, with ``4 * dim`` as the fallback.
    seed : int or None
        Random seed for :class:`LHSInitializer`.
    history_channels : sequence of str or None
        Execution history channels to record. ``None`` keeps the default
        summary history. Include ``"evaluation"`` to record every true
        evaluation result.
    verbose : bool
        If ``False``, suppress per-generation log output. Default: ``True``.

    Returns
    -------
    Result

    Examples
    --------
    >>> from saealib import maximize
    >>> import numpy as np
    >>> result = maximize(lambda x: -np.sum(x**2) + 10, dim=5, lb=[-5]*5, ub=[5]*5,
    ...                   max_fe=500, seed=0, verbose=False)
    >>> result.x, result.f
    """
    problem = _ensure_problem(
        func, dim, lb, ub, n_obj, direction, default_direction=+1.0
    )
    return _run(
        problem,
        algorithm,
        surrogate,
        strategy,
        max_fe,
        pop_size,
        seed,
        verbose,
        preset,
        history_channels,
    )
