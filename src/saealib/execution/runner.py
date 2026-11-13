"""Thin facade over the configured execution runtime."""

from __future__ import annotations

from collections.abc import Callable, Generator
from typing import Any

from saealib.context import OptimizationState
from saealib.core.runtime import ExecutionRuntime, RuntimeSession
from saealib.exceptions import ValidationError
from saealib.execution.history import History
from saealib.execution.runtime import create_runtime, resolve_plan


class Runner:
    """Drive a runtime and yield only its observable state boundaries."""

    def __init__(
        self,
        optimizer: Any,
        runtime_factory: Callable[[Any], ExecutionRuntime] = create_runtime,
    ) -> None:
        self.optimizer = optimizer
        self._runtime_factory = runtime_factory

    def run(self) -> OptimizationState:
        """Run to completion and return the final state."""
        for state in self.iterate():
            pass
        return state

    def run_from(self, state: OptimizationState) -> OptimizationState:
        """Resume from an existing state and run to completion."""
        for state in self.iterate_from(state):
            pass
        return state

    def iterate(self) -> Generator[OptimizationState, None, None]:
        """Initialize the runtime and yield observable runtime states."""
        initializer = getattr(self.optimizer, "initializer", None)
        if initializer is None:
            raise RuntimeError("Runner requires an optimizer initializer")
        state = initializer.initialize(self.optimizer, self.optimizer.problem)
        yield from self.iterate_from(state)

    def iterate_from(
        self, state: OptimizationState
    ) -> Generator[OptimizationState, None, None]:
        """Drive runtime steps from an initial or checkpoint state."""
        comparator = getattr(self.optimizer, "comparator", None)
        if comparator is not None and state.comparator is not comparator:
            state.comparator = comparator
        configured_channels = tuple(
            getattr(self.optimizer, "history_channels", ("summary",))
        )
        requested_channels = frozenset(configured_channels)
        if state.history is None:
            state.history = History(configured_channels)
        elif getattr(self.optimizer, "history_channels_explicit", False) and (
            requested_channels != state.history.enabled
        ):
            raise ValidationError(
                "Cannot resume with different history channel sets: "
                f"requested channels={sorted(requested_channels)!r}, "
                f"checkpoint channels={sorted(state.history.enabled)!r}. "
                "Use set_history() to match the checkpoint channels, "
                "or start a new run."
            )
        plan = resolve_plan(self.optimizer)
        runtime = self._runtime_factory(self.optimizer)
        session: RuntimeSession = runtime.initialize(plan, state)
        if session.observable:
            yield session.state
        while not session.finished:
            step = runtime.advance(session)
            if step.session is None:
                raise RuntimeError("RuntimeStep must provide its next session")
            session = step.session
            if step.observable:
                yield step.state
