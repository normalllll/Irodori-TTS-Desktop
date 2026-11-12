"""Optional process events and cooperative cancellation shared by CLI and desktop."""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from pathlib import Path

from irodori_tts.desktop.messages import t


class TaskCancelled(Exception):
    """Raised only at a computation boundary where no result has been published."""


def check_cancel(cancel: Callable[[], bool] | None) -> None:
    if cancel is not None and cancel():
        raise TaskCancelled(t("task_control.task_cancelled"))


class TaskChannel:
    _sequences: dict[str, int] = {}

    def __init__(self) -> None:
        self.task_id = os.environ.get("IRODORI_TASK_ID", "cli")
        self.events = os.environ.get("IRODORI_TASK_EVENTS")
        self.stop_file = os.environ.get("IRODORI_TASK_STOP")
        self.sequence = 0

    def stopped(self) -> bool:
        return bool(self.stop_file and Path(self.stop_file).exists())

    def emit(self, kind: str, **values) -> None:
        self.sequence = self._sequences.get(self.task_id, 0) + 1
        self._sequences[self.task_id] = self.sequence
        from irodori_tts.desktop.messages import persist

        event = persist(
            {"task_id": self.task_id, "sequence": self.sequence, "type": kind, **values}
        )
        if self.events:
            with Path(self.events).open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(event, ensure_ascii=False) + "\n")
                stream.flush()
