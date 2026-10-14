"""
Termination module.

Termination class defines criteria to stop the optimization process.
Users can specify arbitrary termination conditions as callable objects.
"""

from __future__ import annotations

import operator
from collections.abc import Callable
from functools import partial, reduce
from typing import TYPE_CHECKING

from saealib.core.contracts import ComponentContract, StateContract
from saealib.core.state import ARCHIVES_MAIN, EVALUATIONS_COUNT, RUNTIME_GENERATION
from saealib.exceptions import ValidationError
from saealib.registry import register

if TYPE_CHECKING:
    from saealib.context import OptimizationState


#: Type alias for a plain termination condition function.
#: A callable that takes an ``OptimizationState`` and returns ``True``
#: when the optimization should stop. Accepted everywhere a
#: :class:`TerminationCondition` is, and wrapped automatically when composed.
ConditionFunc = Callable[["OptimizationState"], bool]


class TerminationCondition:
    """
    Composable termination condition.

    Wraps a callable ``ctx -> bool`` and adds logical operators so that
    conditions can be combined declaratively:

    - ``a | b`` : terminate when **either** ``a`` or ``b`` is met (OR)
    - ``a & b`` : terminate when **both** ``a`` and ``b`` are met (AND)
    - ``~a``    : terminate when ``a`` is **not** met (NOT)

    The other operand may be a plain ``Callable[[ctx], bool]``; it is
    wrapped automatically. Reflected operators (``__ror__`` / ``__rand__``)
    let a plain callable appear on the left-hand side as well.

    Parameters
    ----------
    func : Callable[[OptimizationState], bool]
        The underlying condition callable.
    name : str, optional
        Human-readable name used for ``repr`` and ``__qualname__``.
        Defaults to ``func.__qualname__`` when available.
    doc : str, optional
        Docstring for the wrapped condition. Defaults to ``func.__doc__``.

    Raises
    ------
    TypeError
        If ``func`` is not callable.

    Examples
    --------
    >>> cond = max_fe(2000) & (lambda ctx: ctx.archive.get("f").min() < 1e-6)
    >>> cond = max_gen(100) | max_fe(2000)
    >>> cond = ~stalled(20)
    """

    def __init__(
        self,
        func: ConditionFunc,
        *,
        name: str | None = None,
        doc: str | None = None,
        spec: dict | None = None,
    ) -> None:
        if not callable(func):
            raise TypeError(
                f"Termination condition must be callable, got {type(func).__name__}."
            )
        self._func = func
        self.__qualname__ = (
            name if name is not None else getattr(func, "__qualname__", repr(func))
        )
        resolved_doc = doc if doc is not None else getattr(func, "__doc__", None)
        if resolved_doc is not None:
            self.__doc__ = resolved_doc
        # Registry.to_spec() reads this instead of reflecting the constructor
        # (``func`` is a closure/partial and cannot be serialized generically).
        # ``None`` for compositions (|, &, ~) and raw callables not built via
        # a registered factory (max_fe, max_gen, f_target, stalled).
        self._registry_spec = spec

    @staticmethod
    def _coerce(cond: ConditionFunc | TerminationCondition) -> TerminationCondition:
        if isinstance(cond, TerminationCondition):
            return cond
        return TerminationCondition(cond)

    def __call__(self, ctx: OptimizationState) -> bool:
        """Evaluate the wrapped condition against ``ctx``."""
        return bool(self._func(ctx))

    def __or__(
        self, other: ConditionFunc | TerminationCondition
    ) -> TerminationCondition:
        """Compose with OR: terminate when either condition is met."""
        other = self._coerce(other)
        return TerminationCondition(
            lambda ctx: self(ctx) or other(ctx),
            name=f"({self.__qualname__} | {other.__qualname__})",
        )

    def __ror__(
        self, other: ConditionFunc | TerminationCondition
    ) -> TerminationCondition:
        """Compose with OR when a plain callable is on the left-hand side."""
        return self._coerce(other) | self

    def __and__(
        self, other: ConditionFunc | TerminationCondition
    ) -> TerminationCondition:
        """Compose with AND: terminate when both conditions are met."""
        other = self._coerce(other)
        return TerminationCondition(
            lambda ctx: self(ctx) and other(ctx),
            name=f"({self.__qualname__} & {other.__qualname__})",
        )

    def __rand__(
        self, other: ConditionFunc | TerminationCondition
    ) -> TerminationCondition:
        """Compose with AND when a plain callable is on the left-hand side."""
        return self._coerce(other) & self

    def __invert__(self) -> TerminationCondition:
        """Negate the condition with NOT."""
        return TerminationCondition(
            lambda ctx: not self(ctx),
            name=f"~{self.__qualname__}",
        )

    def __repr__(self) -> str:
        """Return a readable representation including the condition name."""
        return f"TerminationCondition({self.__qualname__})"


@register()
class Termination:
    """
    Termination class to determine when to stop the optimization process.

    Accepts one or more callable conditions. Each condition receives
    an ``OptimizationState`` and returns ``True`` when the process
    should terminate. The optimization stops when **any** condition
    evaluates to ``True``.

    Parameters
    ----------
    *conditions : ConditionFunc or TerminationCondition
        One or more callable conditions. Each must accept an
        ``OptimizationState`` and return ``bool``.

    Raises
    ------
    ValueError
        If no conditions are provided.
    TypeError
        If any condition is not callable.

    Examples
    --------
    >>> termination = Termination(max_fe(2000))
    >>> termination = Termination(max_fe(2000), max_gen(100))
    >>> termination = Termination(
    ...     max_fe(2000),
    ...     lambda ctx: ctx.archive.get("f").min() < 1e-6,
    ... )
    >>> termination = Termination.all_of(max_fe(2000), max_gen(100))
    >>> termination = Termination.not_(max_gen(100))
    """

    def __init__(self, *conditions: ConditionFunc | TerminationCondition):
        if not conditions:
            raise ValidationError(
                "At least one termination condition must be provided."
            )
        for cond in conditions:
            if not callable(cond):
                raise TypeError(
                    "Termination condition must be callable, "
                    f"got {type(cond).__name__}."
                )
        # Kept alongside the composed condition below so Registry.to_spec()
        # can serialize each original condition (see TerminationCondition.spec).
        self.conditions: tuple[TerminationCondition, ...] = tuple(
            TerminationCondition._coerce(cond) for cond in conditions
        )
        # Combine all conditions with OR into a single composed
        # ``TerminationCondition``, which is the sole source of truth used by
        # ``is_terminated``.
        self._condition: TerminationCondition = reduce(operator.or_, self.conditions)

    @property
    def condition(self) -> TerminationCondition:
        """Return the composed termination condition (read-only)."""
        return self._condition

    def contract(self) -> ComponentContract:
        """Return the termination contract inferred from registered conditions."""
        state_keys = {
            "max_fe": (EVALUATIONS_COUNT,),
            "max_gen": (RUNTIME_GENERATION,),
            "f_target": (ARCHIVES_MAIN,),
            "stalled": (ARCHIVES_MAIN, RUNTIME_GENERATION),
        }
        conservative = (EVALUATIONS_COUNT, RUNTIME_GENERATION, ARCHIVES_MAIN)
        reads: list = []
        reads_enumerable = True
        for condition in self.conditions:
            spec = condition._registry_spec
            condition_reads = (
                state_keys.get(spec.get("type")) if isinstance(spec, dict) else None
            )
            reads.extend(conservative if condition_reads is None else condition_reads)
            reads_enumerable = reads_enumerable and condition_reads is not None
        return ComponentContract(
            state=StateContract(
                reads=tuple(dict.fromkeys(reads)),
                reads_enumerable=reads_enumerable,
            ),
        )

    @classmethod
    def any_of(cls, *conditions: ConditionFunc | TerminationCondition) -> Termination:
        """
        Build a Termination that stops when **any** condition is met (OR).

        Equivalent to ``Termination(*conditions)``; provided for symmetry
        with :meth:`all_of` and :meth:`not_`.

        Parameters
        ----------
        *conditions : ConditionFunc or TerminationCondition
            One or more callable conditions.

        Returns
        -------
        Termination
            A termination combining the conditions with OR.
        """
        return cls(*conditions)

    @classmethod
    def all_of(cls, *conditions: ConditionFunc | TerminationCondition) -> Termination:
        """
        Build a Termination that stops when **all** conditions are met (AND).

        Parameters
        ----------
        *conditions : ConditionFunc or TerminationCondition
            One or more callable conditions.

        Returns
        -------
        Termination
            A termination combining the conditions with AND.

        Raises
        ------
        ValueError
            If no conditions are provided.
        """
        if not conditions:
            raise ValidationError(
                "At least one termination condition must be provided."
            )
        combined = reduce(
            operator.and_,
            (TerminationCondition._coerce(cond) for cond in conditions),
        )
        return cls(combined)

    @classmethod
    def not_(cls, condition: ConditionFunc | TerminationCondition) -> Termination:
        """
        Build a Termination that stops when ``condition`` is **not** met (NOT).

        Parameters
        ----------
        condition : ConditionFunc or TerminationCondition
            The condition to negate.

        Returns
        -------
        Termination
            A termination that negates the given condition.
        """
        return cls(~TerminationCondition._coerce(condition))

    def is_terminated(self, ctx: OptimizationState) -> bool:
        """
        Check if the composed termination condition is met.

        Parameters
        ----------
        ctx : OptimizationState
            The current optimization context.

        Returns
        -------
        bool
            True if the termination condition is met, False otherwise.
        """
        return self._condition(ctx)


# Built-in termination condition factories


def _fe_reached(ctx: OptimizationState, value: int) -> bool:
    return ctx.fe >= value


def _gen_reached(ctx: OptimizationState, value: int) -> bool:
    return ctx.gen >= value


@register()
def max_fe(value: int) -> TerminationCondition:
    """
    Create a termination condition based on maximum function evaluations.

    Parameters
    ----------
    value : int
        Maximum number of function evaluations.

    Returns
    -------
    TerminationCondition
        A composable condition that returns True when ``ctx.fe >= value``.

    Examples
    --------
    >>> termination = Termination(max_fe(2000))
    >>> condition = max_fe(2000) & max_gen(100)
    """
    return TerminationCondition(
        partial(_fe_reached, value=value),
        name=f"max_fe({value})",
        doc=f"Terminate when fe >= {value}.",
        spec={"type": "max_fe", "params": {"value": value}},
    )


@register()
def max_gen(value: int) -> TerminationCondition:
    """
    Create a termination condition based on maximum generations.

    Parameters
    ----------
    value : int
        Maximum number of generations.

    Returns
    -------
    TerminationCondition
        A composable condition that returns True when ``ctx.gen >= value``.

    Examples
    --------
    >>> termination = Termination(max_gen(100))
    >>> condition = max_gen(100) | max_fe(2000)
    """
    return TerminationCondition(
        partial(_gen_reached, value=value),
        name=f"max_gen({value})",
        doc=f"Terminate when gen >= {value}.",
        spec={"type": "max_gen", "params": {"value": value}},
    )


@register()
def f_target(value: float) -> TerminationCondition:
    """
    Create a termination condition based on a target objective value.

    Intended for single-objective problems. The direction is taken from
    ``ctx.direction`` (``-1`` for minimization, ``+1`` for maximization), so the
    condition is met when the best objective found reaches ``value`` from the
    correct side (``best <= value`` when minimizing, ``best >= value`` when
    maximizing). Returns ``False`` while the archive is empty.

    Parameters
    ----------
    value : float
        Target objective value.

    Returns
    -------
    TerminationCondition
        A composable condition that returns True when the best objective in
        the archive reaches ``value``.

    Examples
    --------
    >>> termination = Termination(max_fe(2000), f_target(1e-6))
    """

    def _condition(ctx: OptimizationState) -> bool:
        f = ctx.archive.get("f")
        if f is None or len(f) == 0:
            return False
        direction = ctx.direction
        # Work in score space (higher is better) so a single comparison covers
        # both minimization and maximization.
        best_score = float((f @ direction).max())
        return best_score >= value * float(direction[0])

    return TerminationCondition(
        _condition,
        name=f"f_target({value})",
        doc=f"Terminate when the best objective reaches {value}.",
        spec={"type": "f_target", "params": {"value": value}},
    )


@register()
def stalled(window: int, tol: float = 1e-8) -> TerminationCondition:
    """
    Create a termination condition based on lack of improvement (stagnation).

    Terminates when the best score (``f @ direction``, higher is better) has not
    improved by more than ``tol`` for ``window`` consecutive generations.
    Improvement is tracked across calls using ``ctx.gen``; the returned
    condition is therefore stateful and intended to be used once per run.

    Parameters
    ----------
    window : int
        Number of generations without improvement before terminating.
    tol : float, optional
        Minimum score increase counted as an improvement, by default ``1e-8``.

    Returns
    -------
    TerminationCondition
        A composable condition that returns True after ``window`` stagnant
        generations.

    Examples
    --------
    >>> termination = Termination(max_fe(2000), stalled(20))
    """
    state: dict[str, float | None] = {"best": None, "stall_gen": None}

    def _condition(ctx: OptimizationState) -> bool:
        f = ctx.archive.get("f")
        if f is None or len(f) == 0:
            return False
        best_score = float((f @ ctx.direction).max())
        if state["best"] is None or best_score > state["best"] + tol:
            state["best"] = best_score
            state["stall_gen"] = ctx.gen
            return False
        stall_gen = state["stall_gen"]
        assert stall_gen is not None
        return (ctx.gen - stall_gen) >= window

    return TerminationCondition(
        _condition,
        name=f"stalled({window})",
        doc=f"Terminate after {window} generations without improvement.",
        spec={"type": "stalled", "params": {"window": window, "tol": tol}},
    )
