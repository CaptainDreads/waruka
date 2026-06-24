# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Stefan Lewis
"""Step-aware progress tracking shared by long-running commands.

Writes a single JSON (PROGRESS_PATH) that always reflects the current step
of whatever pipeline command is running, with a wall-clock heartbeat so the
live monitor (waruka.monitor) never goes stale during long silent steps.
Writes are atomic (write-temp-then-rename).
"""
from __future__ import annotations

import json
import os
import tempfile
import time

PROGRESS_PATH = "_progress.json"


class Progress:
    def __init__(self, command: str, source: str | None = None,
                 out_path: str | None = None, heartbeat_s: float = 1.5,
                 path: str = PROGRESS_PATH):
        self.path = path
        self.state: dict = {
            "pid": os.getpid(),
            "command": command,
            "status": "running",
            "step": "starting",
            "step_progress": None,
            "step_detail": "",
            "step_started_at": time.time(),
            "overall_started_at": time.time(),
            "last_update": time.time(),
            "elapsed_s": 0.0,
            "source": source,
            "out_path": out_path,
        }
        self.heartbeat_s = heartbeat_s
        self._last_flush = 0.0
        self.flush(force=True)

    def set_step(self, name: str, *, detail: str = "",
                 progress: float | None = None, **extra):
        self.state["step"] = name
        self.state["step_started_at"] = time.time()
        self.state["step_progress"] = progress
        self.state["step_detail"] = detail
        for k in ("current_frame", "fps_observed", "eta_s",
                  "merge_alive", "merge_merged"):
            self.state.pop(k, None)
        self.state.update(extra)
        self.flush(force=True)

    def update(self, *, progress: float | None = None,
               detail: str | None = None, **extra):
        if progress is not None:
            self.state["step_progress"] = progress
        if detail is not None:
            self.state["step_detail"] = detail
        self.state.update(extra)
        self.flush()

    def heartbeat(self):
        self.flush()

    def done(self, **extra):
        self.state["status"] = "done"
        self.state["step"] = "done"
        self.state["step_progress"] = 1.0
        self.state.update(extra)
        self.flush(force=True)

    def fail(self, msg: str):
        self.state["status"] = "failed"
        self.state["step_detail"] = msg
        self.flush(force=True)

    def flush(self, force: bool = False):
        now = time.time()
        if not force and (now - self._last_flush) < self.heartbeat_s:
            return
        self._last_flush = now
        self.state["last_update"] = now
        self.state["elapsed_s"] = now - self.state["overall_started_at"]
        self.state["step_elapsed_s"] = now - self.state["step_started_at"]
        try:
            d = os.path.dirname(os.path.abspath(self.path)) or "."
            fd, tmp = tempfile.mkstemp(prefix=".waruka_progress_", dir=d)
            with os.fdopen(fd, "w") as f:
                json.dump(self.state, f)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, self.path)
        except Exception:
            pass
