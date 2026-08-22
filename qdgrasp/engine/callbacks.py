"""Callback contract for the QDGrasp runner.

Callbacks observe the run; they never own it.  Every hook receives the runner
state mapping so a callback stays usable when new fields are added.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable, Protocol, runtime_checkable


LOGGER = logging.getLogger("qdgrasp.engine")


@runtime_checkable
class Callback(Protocol):
    """Lifecycle hooks invoked by :class:`qdgrasp.engine.runner.Runner`."""

    def on_train_start(self, state: dict[str, Any]) -> None: ...

    def on_step_end(self, state: dict[str, Any]) -> None: ...

    def on_validation_end(self, state: dict[str, Any]) -> None: ...

    def on_train_end(self, state: dict[str, Any]) -> None: ...


class BaseCallback:
    """No-op implementation of every hook; subclass and override what you need."""

    def on_train_start(self, state: dict[str, Any]) -> None:
        """Called once before the first optimisation step."""

    def on_step_end(self, state: dict[str, Any]) -> None:
        """Called after each optimiser step with the current loss and step index."""

    def on_validation_end(self, state: dict[str, Any]) -> None:
        """Called after each validation pass with the resulting metrics."""

    def on_train_end(self, state: dict[str, Any]) -> None:
        """Called once after the final step, before the result bundle is written."""


class CallbackList(BaseCallback):
    """Fan one hook call out to an ordered list of callbacks."""

    def __init__(self, callbacks: Iterable[Callback] | None = None) -> None:
        self._callbacks: list[Callback] = list(callbacks or ())

    def append(self, callback: Callback) -> None:
        """Add a callback to the end of the chain."""

        self._callbacks.append(callback)

    def __len__(self) -> int:
        return len(self._callbacks)

    def _dispatch(self, hook: str, state: dict[str, Any]) -> None:
        for callback in self._callbacks:
            getattr(callback, hook)(state)

    def on_train_start(self, state: dict[str, Any]) -> None:
        self._dispatch("on_train_start", state)

    def on_step_end(self, state: dict[str, Any]) -> None:
        self._dispatch("on_step_end", state)

    def on_validation_end(self, state: dict[str, Any]) -> None:
        self._dispatch("on_validation_end", state)

    def on_train_end(self, state: dict[str, Any]) -> None:
        self._dispatch("on_train_end", state)


class ProgressLogger(BaseCallback):
    """Log the effective runtime once and then one line every ``interval`` steps."""

    def __init__(self, interval: int = 10) -> None:
        if interval < 1:
            raise ValueError("interval must be >= 1")
        self.interval = interval

    def on_train_start(self, state: dict[str, Any]) -> None:
        LOGGER.info(
            "train start: device=%s precision=%s max_steps=%d seed=%d",
            state["runtime"]["effective"]["device"],
            state["runtime"]["effective"]["precision"],
            state["max_steps"],
            state["runtime"]["effective"]["seed"],
        )

    def on_step_end(self, state: dict[str, Any]) -> None:
        step = state["global_step"]
        if step % self.interval == 0 or step == state["max_steps"]:
            LOGGER.info("step %d/%d loss=%.6f", step, state["max_steps"], state["loss"])

    def on_validation_end(self, state: dict[str, Any]) -> None:
        metrics = ", ".join(f"{key}={value:.6f}" for key, value in sorted(state["metrics"].items()))
        LOGGER.info("validation at step %d: %s", state["global_step"], metrics)

    def on_train_end(self, state: dict[str, Any]) -> None:
        LOGGER.info("train end: %d step(s), final loss=%.6f", state["global_step"], state["loss"])


class LossHistory(BaseCallback):
    """Collect ``(step, loss)`` pairs so tests and reports can assert on them."""

    def __init__(self) -> None:
        self.history: list[tuple[int, float]] = []

    def on_step_end(self, state: dict[str, Any]) -> None:
        self.history.append((int(state["global_step"]), float(state["loss"])))
