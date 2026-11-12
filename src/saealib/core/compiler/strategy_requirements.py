"""Resolve a strategy's declared surrogate requirement.

The declared flow itself is inspected by the strategy layer, which exposes
:meth:`~saealib.strategies.base.TemplateStrategy._derive_surrogate_requirement`;
this module applies the decision that diagnostics and runtime share, so the
compiler package stays free of feature-package imports.
"""

from __future__ import annotations

from typing import NamedTuple

__all__ = ["SurrogateRequirement", "resolve_requires_surrogate"]


_MISSING = object()


class SurrogateRequirement(NamedTuple):
    """Resolved surrogate requirement of an optimization strategy.

    Attributes
    ----------
    effective : bool or None
        The requirement diagnostics and runtime use; ``None`` when the
        strategy declares none and no declared flow derives one.
    declared : bool or None
        The plain ``requires_surrogate`` attribute when it holds a boolean,
        so a class attribute, a mixin attribute, an instance assignment, and
        a value restored from a pickle or a spec all count.
    declares : bool
        Whether the strategy exposes a ``requires_surrogate`` attribute at
        all.
    """

    effective: bool | None
    declared: bool | None
    declares: bool


def resolve_requires_surrogate(strategy: object) -> SurrogateRequirement:
    """Resolve the surrogate requirement of *strategy*.

    ``requires_surrogate`` stays a plain attribute.  A strategy that lowers a
    declared flow with the standard ``build_graph`` reports the requirement
    derived from that flow, and a declaration that contradicts it is reported
    by the configuration diagnostics and does not change the outcome.  Every
    other strategy keeps using its declared attribute.
    """
    declared_value = getattr(strategy, "requires_surrogate", _MISSING)
    declared = declared_value if isinstance(declared_value, bool) else None
    derive = getattr(strategy, "_derive_surrogate_requirement", None)
    derived = derive() if callable(derive) else None
    return SurrogateRequirement(
        effective=derived if isinstance(derived, bool) else declared,
        declared=declared,
        declares=declared_value is not _MISSING,
    )
