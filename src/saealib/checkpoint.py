"""CheckpointCallback: automatic checkpoint saving during optimization."""

from __future__ import annotations

import contextlib
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, cast

from saealib.exceptions import ValidationError

if TYPE_CHECKING:
    from saealib.callback import CallbackManager, GenerationEndEvent, RunEndEvent
    from saealib.optimizer import Optimizer


class CheckpointCallback:
    """
    Callback that saves checkpoints at regular generation intervals.

    Two formats are supported:

    - ``'npz'``: portable numpy format (best-effort reproducibility).
    - ``'pickle'``: full optimizer + context state (limited complete
      reproducibility; version-sensitive).
    - ``'both'``: saves both formats each checkpoint.

    Register this callback with :meth:`register`, or pass it implicitly
    via the ``checkpoint_*`` parameters of :meth:`~saealib.Optimizer.run`
    and :meth:`~saealib.Optimizer.iterate`.

    Parameters
    ----------
    path : str or Path
        Directory where checkpoint files are saved.  Created if absent.
    interval : int, optional
        Save a checkpoint every *interval* generations.  Default: 1.
    format : {'npz', 'pickle', 'both'}, optional
        Checkpoint format.  Default: ``'npz'``.
    delete_on_success : bool, optional
        If True, all checkpoint files produced by this callback are
        deleted when the run terminates normally (``RunEndEvent``).
        Useful to avoid accumulating files after a successful run.
        Default: False.
    optimizer : Optimizer or None, optional
        Required when *format* is ``'pickle'`` or ``'both'``.
    """

    _VALID_FORMATS = frozenset({"npz", "pickle", "both"})

    def __init__(
        self,
        path: str | Path,
        interval: int = 1,
        format: str = "npz",
        delete_on_success: bool = False,
        optimizer: Optimizer | None = None,
    ) -> None:
        if format not in self._VALID_FORMATS:
            raise ValidationError(
                f"format must be one of {sorted(self._VALID_FORMATS)!r}, got {format!r}"
            )
        if format in ("pickle", "both") and optimizer is None:
            raise ValidationError(
                "optimizer must be provided when format is 'pickle' or 'both'"
            )
        self._base = Path(path)
        self.interval = interval
        self.format = format
        self.delete_on_success = delete_on_success
        self._optimizer = optimizer
        self._saved: list[Path] = []

    def register(self, cbmanager: CallbackManager) -> None:
        """Register generation-end and run-end handlers on *cbmanager*."""
        from saealib.callback import GenerationEndEvent, RunEndEvent

        cbmanager.register(GenerationEndEvent, self._on_generation_end)
        cbmanager.register(RunEndEvent, self._on_run_end)

    def _on_generation_end(self, event: GenerationEndEvent) -> None:
        ctx = event.ctx
        if ctx.gen % self.interval != 0:
            return

        self._base.mkdir(parents=True, exist_ok=True)
        stem = f"checkpoint_{ctx.gen:06d}"

        if self.format in ("npz", "both"):
            from saealib.context import OptimizationState

            state = cast(OptimizationState, ctx)
            p = self._base / f"{stem}.npz"
            state.save(p)
            self._saved.append(p)

        if self.format in ("pickle", "both"):
            p = self._base / f"{stem}.pkl"
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                self._optimizer.save_pickle(ctx, p)  # type: ignore  # save_pickle typing incomplete in stubs
            self._saved.append(p)

    def _on_run_end(self, event: RunEndEvent) -> None:
        if not self.delete_on_success:
            return
        for p in self._saved:
            with contextlib.suppress(OSError):
                p.unlink(missing_ok=True)
        self._saved.clear()
