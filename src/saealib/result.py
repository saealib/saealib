"""Optimization result returned by the high-level API."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from saealib.exceptions import ValidationError

if TYPE_CHECKING:
    from saealib.context import OptimizationState
    from saealib.execution.history import History


_HISTORY_X_NAMES = {
    "fe": "Function evaluations",
    "gen": "Generations",
    "decision_count": "Decisions",
}
_HISTORY_X = frozenset(_HISTORY_X_NAMES)


@dataclass(frozen=True)
class HistorySeries:
    """A history series and the names used to describe its coordinates.

    Parameters
    ----------
    x : np.ndarray
        Values for the independent coordinate.
    y : np.ndarray
        Values for the dependent coordinate.
    x_name : str
        Display name for the independent coordinate.
    y_name : str
        Display name for the dependent coordinate.
    """

    x: np.ndarray
    y: np.ndarray
    x_name: str
    y_name: str


@dataclass
class Result:
    """Optimization result returned by :func:`minimize` / :func:`maximize`.

    ``x``/``f``/``fe``/``gen`` are fixed at creation time (the arrays are
    copied).  ``ctx``/``history``/``archive``/``pareto_archive``/``population``
    are references derived from the state and may change while iteration is
    still in progress.

    Attributes
    ----------
    x : np.ndarray or None
        Best design variables. Shape ``(dim,)`` for single-objective,
        ``(n_pareto, dim)`` for multi-objective. ``None`` for non-dense
        spaces; use ``problem.space.services`` to convert
        ``result.ctx.archive.genomes`` when dense design variables are needed.
    f : np.ndarray
        Best objective values. Shape ``(n_obj,)`` for single-objective,
        ``(n_pareto, n_obj)`` for multi-objective.
    fe : int
        Total number of true function evaluations used.
    gen : int
        Total number of generations completed.
    history : History or None
        Execution history recorded during the run.
    ctx : OptimizationState
        Full optimization context providing access to the archive and more.
    """

    x: np.ndarray | None
    f: np.ndarray
    fe: int
    gen: int
    ctx: OptimizationState
    history: History | None = None

    @classmethod
    def from_state(cls, state: OptimizationState) -> Result:
        """Build a result from the final or currently observed state.

        Result feasibility uses ``problem.eps_cv`` and intentionally differs
        from ``problem.handler.feasibility_threshold``.
        """
        archive_f = state.archive.get_array("f")
        archive_cv = state.archive.get_array("cv")
        direction = state.problem.direction
        eps = state.problem.eps_cv

        feasible = np.where(archive_cv <= eps)[0]
        if len(feasible):
            pool = feasible
        elif len(archive_cv):
            pool = np.array([int(np.argmin(archive_cv))])
        else:
            raise ValidationError(
                "Cannot build a result from a state whose archive is empty. "
                "Evaluate at least one candidate before requesting a result."
            )

        if state.problem.n_obj == 1:
            scores = archive_f[pool] @ direction
            best_idx = pool[int(np.argmax(scores))]
            try:
                best_x = state.archive.get_array("x")[best_idx]
            except AttributeError:
                best_x = None
            best_f = archive_f[best_idx]
        else:
            if len(state.pareto_archive) > 0:
                try:
                    best_x = state.pareto_archive.get_array("x")
                except AttributeError:
                    best_x = None
                best_f = state.pareto_archive.get_array("f")
            else:
                from saealib.comparators import non_dominated_sort

                _, fronts = non_dominated_sort(archive_f[pool], direction=direction)
                pareto_idx = pool[fronts[0]]
                try:
                    best_x = state.archive.get_array("x")[pareto_idx]
                except AttributeError:
                    best_x = None
                best_f = archive_f[pareto_idx]

        return cls(
            x=None if best_x is None else np.array(best_x, copy=True),
            f=np.array(best_f, copy=True),
            fe=int(state.fe),
            gen=int(state.gen),
            history=state.history,
            ctx=state,
        )

    @property
    def problem(self):
        """Return the problem referenced by the optimization state."""
        return self.ctx.problem

    @property
    def archive(self):
        """Return the archive referenced by the optimization state."""
        return self.ctx.archive

    @property
    def pareto_archive(self):
        """Return the Pareto archive referenced by the optimization state."""
        return self.ctx.pareto_archive

    @property
    def population(self):
        """Return the population referenced by the optimization state."""
        return self.ctx.population

    def history_series(
        self,
        value: str | Callable[[Mapping[str, Any]], float],
        *,
        x: str = "fe",
        channel: str | None = None,
        **value_kwargs: Any,
    ) -> HistorySeries:
        """Return one scalar value per recorded history generation.

        Parameters
        ----------
        value : str or callable
            Registered value name, or a callable mapping one history record to
            a scalar. A callable also requires *channel*.
        x : {"fe", "gen", "decision_count"}, optional
            Column supplying the independent coordinate.
        channel : str, optional
            History channel whose records feed a callable *value*.
        **value_kwargs
            Arguments required by the registered value: ``reference_point``
            for ``"hypervolume"``, ``reference_front`` for ``"gd"``,
            ``"gd_plus"``, ``"igd"``, ``"igd_plus"`` and ``"spread"``.

        Returns
        -------
        HistorySeries
            The series and the display names of its two coordinates.

        Raises
        ------
        ValidationError
            If *x* or *value* is unknown, the channel the value needs was not
            recorded, or a reference argument has the wrong shape.

        Notes
        -----
        A reference is supplied in the objective space of the problem, the same
        orientation as :attr:`Result.f`. Front and reference are both converted
        to minimize-space before an indicator runs, and neither is normalized,
        so an indicator value keeps raw objective units.

        Indicators read the ``front`` channel, a per-generation snapshot of the
        Pareto archive holding objective values alone, with no constraint
        violation column. That archive judges feasibility by its own ``eps_cv``,
        which the built-in algorithms leave at ``0.0``, so it admits a solution
        as feasible only at exactly zero violation. While it holds no such
        solution, constrained dominance reduces the front to the lowest-cv
        solutions and an indicator over that generation describes an infeasible
        front; ``front_size`` collapsing to a single point is the visible sign.

        Feasibility elsewhere in the series rests on different thresholds:
        ``"best"`` selects from the archive under ``problem.eps_cv``, and
        ``"feasible_ratio"`` counts the population under the running
        ``problem.handler.feasibility_threshold``. The three are independent
        and can disagree, most visibly while epsilon is relaxed.
        """
        if x not in _HISTORY_X:
            valid = ", ".join(f'"{name}"' for name in _HISTORY_X_NAMES)
            raise ValidationError(f"x must be one of: {valid}.")
        if self.history is None:
            raise ValidationError(
                "history_series requires execution history, but the result has none. "
                "Record it by passing history_channels=[...] to minimize() or by "
                "enabling it on the Optimizer with set_history([...])."
            )

        if isinstance(value, str):
            from saealib._series_values import _HISTORY_VALUES, _builtin_values

            try:
                spec = _HISTORY_VALUES[value]
            except KeyError as exc:
                valid = ", ".join(_HISTORY_VALUES)
                raise ValidationError(
                    f"Unknown history value {value!r}. Choose one of: {valid}."
                ) from exc
            if not self.history.is_enabled(spec.channel):
                raise ValidationError(
                    f'history_series value "{value}" requires the '
                    f'"{spec.channel}" history channel. '
                    "Enable it with minimize(..., history_channels=[...]) or "
                    "Optimizer.set_history([...]), then rerun the optimization."
                )
            y = _builtin_values(self, value, spec, value_kwargs)
            y_name = value
        elif callable(value):
            if channel is None:
                raise ValidationError(
                    "A callable history value requires channel=... so its "
                    "records can be selected."
                )
            if not self.history.is_enabled(channel):
                raise ValidationError(
                    f'history_series callable value requires the "{channel}" '
                    "history channel. Enable it with "
                    "minimize(..., history_channels=[...]) or "
                    "Optimizer.set_history([...]), then rerun the optimization."
                )
            records = list(self.history.records(channel))
            try:
                y = [float(value(record)) for record in records]
                x_values = [float(record[x]) for record in records]
            except KeyError as exc:
                raise ValidationError(
                    f'history channel {channel!r} is missing the "{x}" column.'
                ) from exc
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValidationError(
                    "A callable history value must return a scalar number."
                ) from exc
            return HistorySeries(
                x=np.asarray(x_values, dtype=float),
                y=np.asarray(y, dtype=float),
                x_name=_HISTORY_X_NAMES[x],
                y_name=getattr(value, "__name__", "callable"),
            )
        else:
            raise ValidationError("value must be a registered name or callable.")

        x_values = self.history.get(spec.channel, x)
        return HistorySeries(
            x=np.asarray(x_values, dtype=float).copy(),
            y=np.asarray(y, dtype=float),
            x_name=_HISTORY_X_NAMES[x],
            y_name=y_name,
        )
