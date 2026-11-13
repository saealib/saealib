"""Optimization result returned by the high-level API."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from saealib.exceptions import ValidationError

if TYPE_CHECKING:
    from saealib.comparators import Comparator
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
        Best objective values under the default criterion. Shape ``(n_obj,)``
        for single-objective, ``(n_pareto, n_obj)`` for multi-objective. Use
        :meth:`best` or :meth:`front` to select a different criterion.
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

        ``x`` and ``f`` are the default-criterion view: the best solution for a
        single-objective problem and the non-dominated set of the truly
        evaluated main-archive rows for a multi-objective problem. The
        default criterion is feasibility first, then Pareto dominance; see
        :meth:`best` and :meth:`front`.
        """
        if len(state.archive) == 0:
            raise ValidationError(
                "Cannot build a result from a state whose archive is empty. "
                "Evaluate at least one candidate before requesting a result."
            )

        result = cls(
            x=None,
            f=np.empty((0,), dtype=float),
            fe=int(state.fe),
            gen=int(state.gen),
            history=state.history,
            ctx=state,
        )
        if state.problem.n_obj == 1:
            x, f = result.best()
        else:
            x, f = result.front()
        result.x = x
        result.f = f
        return result

    def best(
        self, criterion: Comparator | None = None
    ) -> tuple[np.ndarray | None, np.ndarray]:
        """Return the best solution under *criterion*.

        Parameters
        ----------
        criterion : Comparator, optional
            Comparison criterion. When omitted, the default criterion is used:
            a solution is feasible when every per-constraint violation is zero
            after its own tolerance, feasible solutions rank before infeasible
            ones, and feasible solutions are ranked by Pareto dominance. When
            no solution is feasible, the default criterion uses the first row
            with the minimum ``cv`` for a single-objective problem and the
            non-dominated rows among the minimum ``cv`` rows for a
            multi-objective problem. A supplied criterion keeps its own
            ``direction`` and ``eps_cv`` and ranks every row with the minimum
            ``cv`` when no row is feasible.

        Returns
        -------
        tuple of (np.ndarray or None, np.ndarray)
            Design variables with shape ``(dim,)`` and objective values with
            shape ``(n_obj,)``. The design variables are ``None`` for a
            non-dense search space.

        Raises
        ------
        ValidationError
            If *criterion* is not a :class:`~saealib.Comparator` or the
            main archive is empty.
        """
        is_default = criterion is None
        criterion = self._resolve_criterion(criterion)
        f, _ = self._archive_values()
        pool = self._selection_pool(criterion, is_default)

        if is_default and self.problem.n_obj == 1:
            scores = f[pool] @ np.asarray(self.problem.direction, dtype=float)
            index = int(pool[int(np.argmax(scores))])
        elif is_default:
            index = int(self._first_front(criterion, f, pool)[0])
        else:
            order = criterion.sort_population(self.archive.extract(pool))
            index = int(pool[int(order[0])])
        return self._archive_row(index)

    def front(
        self, criterion: Comparator | None = None
    ) -> tuple[np.ndarray | None, np.ndarray]:
        """Return the non-dominated solutions under *criterion*.

        Parameters
        ----------
        criterion : Comparator, optional
            Comparison criterion. When omitted, the default criterion is used:
            a solution is feasible when every per-constraint violation is zero
            after its own tolerance, and the feasible solutions are filtered by
            Pareto dominance with the criterion's own sorter. When no solution
            is feasible, the default criterion returns the first row with the
            minimum ``cv`` for a single-objective problem and the non-dominated
            rows among the minimum ``cv`` rows for a multi-objective problem.
            A supplied criterion keeps its own ``direction`` and ``eps_cv`` and
            ranks every row with the minimum ``cv`` when no row is feasible.

        Returns
        -------
        tuple of (np.ndarray or None, np.ndarray)
            Design variables with shape ``(n_front, dim)`` and objective values
            with shape ``(n_front, n_obj)``. The design variables are ``None``
            for a non-dense search space.

        Raises
        ------
        ValidationError
            If *criterion* is not a :class:`~saealib.Comparator` or the
            main archive is empty.
        """
        is_default = criterion is None
        criterion = self._resolve_criterion(criterion)
        f, _ = self._archive_values()
        pool = self._selection_pool(criterion, is_default)
        indices = self._first_front(criterion, f, pool)
        return self._archive_rows(indices)

    def _resolve_criterion(self, criterion: Comparator | None) -> Comparator:
        from saealib.comparators import Comparator, ParetoComparator

        if criterion is None:
            return ParetoComparator(
                direction=np.asarray(self.problem.direction, dtype=float),
                eps_cv=0.0,
            )
        if not isinstance(criterion, Comparator):
            raise ValidationError("criterion must be a Comparator instance or None.")
        return criterion

    def _archive_values(self) -> tuple[np.ndarray, np.ndarray]:
        archive = self.ctx.archive
        if len(archive) == 0:
            raise ValidationError(
                "Cannot select solutions from a result whose archive is empty."
            )
        f = np.asarray(archive.get_array("f"), dtype=float)
        cv = np.asarray(archive.get_array("cv"), dtype=float)
        return f, cv

    def _selection_pool(self, criterion: Comparator, is_default: bool) -> np.ndarray:
        _, cv = self._archive_values()
        if is_default:
            from saealib.problem.constraint import _feasible_rows

            feasible = _feasible_rows(self.ctx.archive, self.problem.constraints)
        else:
            feasible = cv <= float(criterion.eps_cv)
        indices = np.flatnonzero(feasible)
        if len(indices):
            return indices
        if is_default and self.problem.n_obj == 1:
            return np.asarray([int(np.argmin(cv))])
        minimum = np.flatnonzero(cv == np.min(cv))
        return minimum if len(minimum) else np.arange(len(cv))

    def _first_front(
        self, criterion: Comparator, f: np.ndarray, pool: np.ndarray
    ) -> np.ndarray:
        from saealib.comparators import Dominator, non_dominated_sort

        dominator = getattr(criterion, "dominator", None)
        if isinstance(dominator, Dominator):
            sorter = getattr(criterion, "sorter", None)
            if not callable(sorter):
                sorter = non_dominated_sort
            ranks, _ = sorter(
                f[pool],
                direction=getattr(criterion, "direction", None),
                dominator=dominator,
            )
            ranks = np.asarray(ranks)
            return pool[np.flatnonzero(ranks == np.min(ranks))]
        order = criterion.sort_population(self.ctx.archive.extract(pool))
        return pool[np.asarray(order[:1], dtype=int)]

    def _archive_row(self, index: int) -> tuple[np.ndarray | None, np.ndarray]:
        x, f = self._archive_rows(np.asarray([index], dtype=int))
        return (None if x is None else x[0], f[0])

    def _archive_rows(
        self, indices: np.ndarray
    ) -> tuple[np.ndarray | None, np.ndarray]:
        archive = self.ctx.archive
        f = np.array(archive.get_array("f")[indices], dtype=float, copy=True)
        try:
            x = np.array(archive.get_array("x")[indices], dtype=float, copy=True)
        except AttributeError:
            x = None
        return x, f

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
        """Return the population referenced by the optimization state.

        The ``objective_source`` column tells true from predicted rows; see
        :class:`~saealib.context.OptimizationState` for its codes.
        """
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
        violation column. The archive decides feasibility from the problem's
        constraints, so it admits a solution only when every per-constraint
        violation is zero. While it holds no such solution, constrained
        dominance reduces the front to the lowest-cv solutions and an indicator
        over that generation describes an infeasible front; ``front_size``
        collapsing to a single point is the visible sign.

        ``"best"`` and ``"feasible_ratio"`` use the same constraint-based
        decision, so relaxing the comparator during an epsilon-constrained run
        changes neither; ``min_cv`` describes the population's raw violation.
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
