"""Decision rules consulted once per step at a component replacement commit point.

A controller returns :class:`~saealib.core.runtime.ReplaceComponent` and
:class:`~saealib.core.runtime.SetParameter` commands.  The runtime applies them
at the commit point of the step, so a controller never touches the plan, the
binding version, or the state counters itself.

Supported
---------
Registering controllers with :meth:`saealib.Optimizer.add_controller` and
having each one decide once per step value at the commit point, in
registration order, for structured and sequential plans run synchronously or
asynchronously.  A controller replaces a binding slot
(:class:`~saealib.core.runtime.ReplaceComponent`, including a transition) or
writes one parameter value (:class:`~saealib.core.runtime.SetParameter`), which
either leaves the plan and the binding version alone or joins the ordinary
replacement path.  The declared state keys are seeded, restored when a commit
point fails, and survive a portable checkpoint, and the accuracy a bound
surrogate manager reported is available as ``SURROGATES_ACCURACY``.  The
helpers in this module cover the accuracy threshold, smoothing and stagnation
rules.

Extensible toward
-----------------
A policy that continues after a rejected command instead of stopping the run,
events and History entries for a controller decision, saving controller
specifications in a preset or a checkpoint, propagating a controller to the
optimizers of an ``IslandModel`` or an ``Experiment``, and a controller whose
``decide`` runs while evaluations are still in flight.

Out of scope
------------
A controller that requests an evaluation, a checkpoint, or termination, and a
controller that changes anything other than components and parameter values.
"""

from __future__ import annotations

import itertools
import math
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any

from saealib.core.runtime import ReplaceComponent, RuntimeCommand, SetParameter
from saealib.core.state import (
    RUNTIME_STAGNATION,
    SURROGATES_ACCURACY,
    SURROGATES_SMOOTHED_ERROR,
    StateKey,
)
from saealib.exceptions import ValidationError

if TYPE_CHECKING:
    from saealib.context import OptimizationState
    from saealib.surrogate.accuracy import SurrogateAccuracy

__all__ = [
    "Controller",
    "FixedSchedule",
    "SmoothedController",
    "StagnationController",
    "ThresholdController",
]

_MISSING = object()


class Controller(ABC):
    """One decision rule a run consults at every step boundary.

    A controller is called once per step value at the commit point, before the
    binding difference is read, and returns the commands the runtime applies
    there.  ``decide`` must depend only on the run state and on the state keys
    the controller declares in :meth:`state_keys`: a run that resumes after a
    rejected commit decides the same step again, and the runtime puts the
    declared keys back first.  Randomness therefore belongs to
    ``state.rng``, not to the controller.

    Mutable controller state belongs to the declared state keys only.  An
    attribute of the controller instance survives a pickle checkpoint but not
    a portable one, and a controller is never part of the plan.

    Notes
    -----
    A controller is not a plan node and declares no contract.  Register it
    with :meth:`saealib.Optimizer.add_controller`.  What the framework supports
    today, what the seam is meant to allow later, and what a controller cannot
    do are stated in the module documentation.
    """

    @abstractmethod
    def decide(self, state: OptimizationState) -> Sequence[RuntimeCommand]:
        """Return the commands this step applies.

        Parameters
        ----------
        state : OptimizationState
            The running state at the start of the step.

        Returns
        -------
        Sequence[RuntimeCommand]
            Replacements and parameter writes to apply at this commit point.
        """

    def state_keys(self) -> tuple[StateKey[Any], ...]:
        """Return the state keys this controller owns.

        Returns
        -------
        tuple[StateKey[Any], ...]
            Keys the runtime seeds with ``None`` when absent, and restores to
            their previous value when the commit point rejects a command.
        """
        return ()


def _accuracy_state_value(accuracy: SurrogateAccuracy | None) -> dict[str, Any] | None:
    """Return the JSON-safe observation a run keeps for an accuracy snapshot."""
    if accuracy is None:
        return None
    metrics = getattr(accuracy, "metrics", None)
    values = {} if not isinstance(metrics, Mapping) else metrics
    return {
        "metrics": {str(name): float(value) for name, value in values.items()},
        "n_samples": int(getattr(accuracy, "n_samples", 0) or 0),
    }


def _metric_from_state(
    state: OptimizationState, key: StateKey[Any], metric: str
) -> float:
    """Return one accuracy metric from an observation, or ``nan`` when absent."""
    try:
        record = state.get_state(key)
    except KeyError:
        return math.nan
    if not isinstance(record, Mapping):
        return math.nan
    metrics = record.get("metrics")
    if not isinstance(metrics, Mapping):
        return math.nan
    value = metrics.get(metric, _MISSING)
    if value is _MISSING or value is None:
        return math.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _observed_state(
    state: OptimizationState, key: StateKey[Any]
) -> Mapping[str, Any] | None:
    """Return a controller's own state record, or ``None`` when unset."""
    try:
        record = state.get_state(key)
    except KeyError:
        return None
    return record if isinstance(record, Mapping) else None


class FixedSchedule(Controller):
    """Emit a fixed command sequence at chosen steps or generations.

    Parameters
    ----------
    schedule : Mapping[int, Sequence[RuntimeCommand]]
        Commands keyed by the step value (or by the generation when ``key`` is
        ``"gen"``).  A value that has no entry emits nothing.
    key : {"step", "gen"}, optional
        Counter the schedule is keyed by. Default: ``"step"``.  A
        generation-based strategy advances ``gen`` by more than one per step,
        so keying by ``"gen"`` skips entries in that flow.

    Notes
    -----
    ``decide`` never advances the counters it reads, so a schedule keyed by
    ``gen`` in a flow that skips generations simply fires fewer times.
    """

    def __init__(
        self,
        schedule: Mapping[int, Sequence[RuntimeCommand]],
        key: str = "step",
    ) -> None:
        if key not in {"step", "gen"}:
            raise ValidationError("FixedSchedule key must be 'step' or 'gen'")
        self.schedule = dict(schedule)
        self.key = key

    def decide(self, state: OptimizationState) -> Sequence[RuntimeCommand]:
        """Return the commands scheduled for the current counter value."""
        index = state.step if self.key == "step" else state.gen
        return tuple(self.schedule.get(index, ()))


class ThresholdController(Controller):
    """Replace a component when an accuracy metric crosses a threshold.

    The levels are ``(lower_bound, component)`` pairs in ascending order: the
    component of the highest bound that does not exceed the metric is written
    to ``slot``.  A missing or non-finite metric selects the first level, so
    the first entry is the fallback.  Two entries give a binary switch, three
    entries give a three-way split.

    Parameters
    ----------
    slot : str
        Binding slot to replace, for example ``"surrogate_manager"`` or
        ``"strategy"``.
    levels : Sequence[tuple[float, object]]
        Ascending ``(lower_bound, component)`` pairs.
    metric : str, optional
        Metric key inside the accuracy observation. Default: ``"spearman"``.
    key : StateKey, optional
        State key holding the accuracy observation. Default:
        ``SURROGATES_ACCURACY``, the observation a run writes at every
        generation end while it has controllers.

    Notes
    -----
    The accuracy observation a commit point reads comes from the previous
    generation's fit, so a threshold never sees the accuracy of the step it
    affects. Selecting the component that is already bound publishes nothing.
    """

    def __init__(
        self,
        slot: str,
        levels: Sequence[tuple[float, object]],
        metric: str = "spearman",
        key: StateKey[Any] = SURROGATES_ACCURACY,
    ) -> None:
        if not levels:
            raise ValidationError("ThresholdController levels must not be empty")
        bounds = [float(bound) for bound, _ in levels]
        if any(later <= earlier for earlier, later in itertools.pairwise(bounds)):
            raise ValidationError(
                "ThresholdController levels must be in ascending order"
            )
        self.slot = slot
        self.levels = tuple((float(bound), component) for bound, component in levels)
        self.metric = metric
        self.key = key

    def decide(self, state: OptimizationState) -> Sequence[RuntimeCommand]:
        """Return the replacement the current accuracy selects."""
        value = _metric_from_state(state, self.key, self.metric)
        component = self.levels[0][1]
        if math.isfinite(value):
            for bound, candidate in self.levels:
                if value >= bound:
                    component = candidate
        return (ReplaceComponent(slot=self.slot, component=component),)


class SmoothedController(Controller):
    r"""Smooth an accuracy metric and write it to an integer parameter.

    The error is estimated from the metric, exponentially smoothed, and mapped
    linearly to the parameter:

    .. math::

        \\varepsilon_t = (1-r_u)\\varepsilon_{t-1} + r_u\\varepsilon_{new}

        g_m = \\operatorname{round}((1-\\varepsilon_t) g_{m,max})

    For Spearman rho and R2, ``epsilon = 0.5 * (1 - value)``; any other metric
    is clamped to ``[0, 1]``.

    Parameters
    ----------
    slot : str, optional
        Binding slot holding the component to write. Default: ``"strategy"``.
    path : str, optional
        Attribute path on that component. Default: ``"gen_ctrl"``.
    gm_max : int, optional
        Upper bound of the written value. Default: ``5``.
    gm_min : int, optional
        Lower bound of the written value. Default: ``0``.
    update_rate : float, optional
        Smoothing rate in ``(0, 1]``. Default: ``0.5``.
    metric : str, optional
        Metric key inside the accuracy observation. Default: ``"spearman"``.
    initial_error : float, optional
        Smoothed error before the first observation. Default: ``0.5``.
    key : StateKey, optional
        State key this controller keeps ``smoothed_error`` under. Default:
        ``SURROGATES_SMOOTHED_ERROR``.  Two instances of this controller in one
        run collide on the default key, so pass a distinct ``key`` for each.

    Notes
    -----
    ``gen_ctrl`` decides the shape of the flow, so a change to it republishes
    the bindings and, while asynchronous evaluations are pending, defers the
    commit point until they drain.  A run that changes the value every
    generation therefore pays one republication and, on asynchronous flows, one
    drain per change.  The write reaches the plan from the next step.
    """

    def __init__(
        self,
        slot: str = "strategy",
        path: str = "gen_ctrl",
        gm_max: int = 5,
        gm_min: int = 0,
        update_rate: float = 0.5,
        metric: str = "spearman",
        initial_error: float = 0.5,
        key: StateKey[Any] = SURROGATES_SMOOTHED_ERROR,
    ) -> None:
        if not 0 < update_rate <= 1.0:
            raise ValidationError("update_rate must be in (0, 1]")
        if gm_min < 0:
            raise ValidationError("gm_min must be >= 0")
        if gm_max < gm_min:
            raise ValidationError("gm_max must be >= gm_min")
        self.slot = slot
        self.path = path
        self.gm_max = gm_max
        self.gm_min = gm_min
        self.update_rate = update_rate
        self.metric = metric
        self.initial_error = float(initial_error)
        self.key = key

    def state_keys(self) -> tuple[StateKey[Any], ...]:
        """Return the state key holding the smoothed error."""
        return (self.key,)

    def decide(self, state: OptimizationState) -> Sequence[RuntimeCommand]:
        """Update the smoothed error and write the mapped parameter value."""
        record = _observed_state(state, self.key)
        smoothed = self.initial_error
        if record is not None and record.get("smoothed_error") is not None:
            smoothed = float(record["smoothed_error"])
        value = _metric_from_state(state, SURROGATES_ACCURACY, self.metric)
        if math.isfinite(value):
            smoothed = (1.0 - self.update_rate) * smoothed + self.update_rate * (
                _error_from_metric(value, self.metric)
            )
        written = max(
            self.gm_min, min(self.gm_max, round((1.0 - smoothed) * self.gm_max))
        )
        state.set_state(self.key, {"smoothed_error": smoothed})
        return (SetParameter(slot=self.slot, path=self.path, value=written),)


def _error_from_metric(value: float, metric: str) -> float:
    """Map an accuracy metric to an error in ``[0, 1]``."""
    if metric in ("spearman", "r2"):
        return 0.5 * (1.0 - float(value))
    return float(max(0.0, min(1.0, value)))


class StagnationController(Controller):
    """Emit a command sequence once per patience window without improvement.

    Parameters
    ----------
    patience : int
        Number of steps without improvement before the commands fire.
    commands : Sequence[RuntimeCommand]
        Commands this controller emits. An empty sequence records progress
        without changing anything.
    measure : Callable[[OptimizationState], float]
        Quality of the current state, where a larger value is better.  The
        measure is required because what counts as progress depends on the
        problem.
    key : StateKey, optional
        State key this controller keeps ``best`` and
        ``last_improvement_step`` under. Default: ``RUNTIME_STAGNATION``.  Two
        instances of this controller in one run collide on the default key, so
        pass a distinct ``key`` for each.

    Notes
    -----
    The first decision records the measure without firing.  After that, a
    measure above ``best`` moves ``last_improvement_step`` to the current step,
    and a window of ``patience`` steps without improvement fires the commands
    and starts a new window.
    """

    def __init__(
        self,
        patience: int,
        commands: Sequence[RuntimeCommand],
        measure: Callable[[OptimizationState], float],
        key: StateKey[Any] = RUNTIME_STAGNATION,
    ) -> None:
        if isinstance(patience, bool) or not isinstance(patience, int) or patience < 1:
            raise ValidationError("patience must be a positive integer")
        if not callable(measure):
            raise ValidationError("measure must be callable")
        self.patience = patience
        self.commands = tuple(commands)
        self.measure = measure
        self.key = key

    def state_keys(self) -> tuple[StateKey[Any], ...]:
        """Return the state key holding the progress record."""
        return (self.key,)

    def decide(self, state: OptimizationState) -> Sequence[RuntimeCommand]:
        """Return the commands when the current window has run out."""
        record = _observed_state(state, self.key) or {}
        best = record.get("best")
        last = record.get("last_improvement_step")
        step = state.step
        value = float(self.measure(state))
        if best is None or last is None:
            state.set_state(self.key, {"best": value, "last_improvement_step": step})
            return ()
        if value > float(best):
            state.set_state(self.key, {"best": value, "last_improvement_step": step})
            return ()
        if step - int(last) >= self.patience:
            state.set_state(
                self.key, {"best": float(best), "last_improvement_step": step}
            )
            return tuple(self.commands)
        return ()
