# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Stefan Lewis
"""Tkinter live-progress monitor for `waruka track`.

Reads `_progress.json` written by run_perception every progress_every frames
and displays a progress bar, ETA, throughput, per-tile track counts, and a
Kill button that sends SIGTERM to the running perception process.

Standalone: launch with `python -m waruka monitor` (optionally `--path` to
point at a non-default progress file). Run it alongside `waruka track`.
"""
from __future__ import annotations

import json
import os
import signal
import time
import tkinter as tk
from tkinter import ttk, messagebox

DEFAULT_PATH = "_progress.json"
POLL_MS = 500   # how often to re-read the progress file
STALE_S = 30.0  # if last_update is older than this, mark stale (process likely died)


def _fmt_hms(seconds):
    if seconds is None or seconds < 0:
        return "--:--:--"
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


class MonitorApp:
    def __init__(self, root: tk.Tk, progress_path: str):
        self.root = root
        self.path = progress_path
        self.last_pid = None
        self.last_status = None

        root.title("Waruka perception monitor")
        root.geometry("520x420")
        root.resizable(False, False)

        self._build_ui()
        self._poll()

    def _build_ui(self):
        self.root.geometry("560x500")
        pad = {"padx": 12, "pady": 3}
        big = ("Segoe UI", 11)
        mono = ("Consolas", 10)

        self.status_var = tk.StringVar(value="waiting for progress file...")
        ttk.Label(self.root, textvariable=self.status_var,
                  font=("Segoe UI", 10, "bold")
                  ).grid(row=0, column=0, columnspan=4, sticky="w", **pad)

        # Current step (prominent)
        ttk.Label(self.root, text="step:", font=("Segoe UI", 12)
                  ).grid(row=1, column=0, sticky="w", **pad)
        self.step_var = tk.StringVar(value="-")
        ttk.Label(self.root, textvariable=self.step_var,
                  font=("Segoe UI", 14, "bold")
                  ).grid(row=1, column=1, columnspan=3, sticky="w", **pad)

        self.step_detail_var = tk.StringVar(value="")
        ttk.Label(self.root, textvariable=self.step_detail_var, font=mono,
                  foreground="#555"
                  ).grid(row=2, column=0, columnspan=4, sticky="w", **pad)

        # Step progress bar
        self.bar = ttk.Progressbar(self.root, length=520, mode="determinate")
        self.bar.grid(row=3, column=0, columnspan=4, **pad)

        # Overall elapsed + step elapsed
        self.elapsed_var = tk.StringVar(value="elapsed --:--:--")
        self.step_elapsed_var = tk.StringVar(value="step --:--:--")
        ttk.Label(self.root, textvariable=self.elapsed_var, font=big
                  ).grid(row=4, column=0, sticky="w", **pad)
        ttk.Label(self.root, textvariable=self.step_elapsed_var, font=big
                  ).grid(row=4, column=3, sticky="e", **pad)

        # Context-sensitive (frame / merge etc.) — populated when applicable
        self.ctx1_var = tk.StringVar(value="")
        self.ctx2_var = tk.StringVar(value="")
        ttk.Label(self.root, textvariable=self.ctx1_var, font=big
                  ).grid(row=5, column=0, sticky="w", **pad)
        ttk.Label(self.root, textvariable=self.ctx2_var, font=big
                  ).grid(row=5, column=3, sticky="e", **pad)

        self.source_var = tk.StringVar(value="source: -")
        self.out_var = tk.StringVar(value="out: -")
        self.pid_var = tk.StringVar(value="pid: -")
        ttk.Label(self.root, textvariable=self.source_var, font=mono
                  ).grid(row=6, column=0, columnspan=4, sticky="w", **pad)
        ttk.Label(self.root, textvariable=self.out_var, font=mono
                  ).grid(row=7, column=0, columnspan=4, sticky="w", **pad)
        ttk.Label(self.root, textvariable=self.pid_var, font=mono
                  ).grid(row=8, column=0, columnspan=4, sticky="w", **pad)

        ttk.Separator(self.root, orient="horizontal").grid(
            row=9, column=0, columnspan=4, sticky="ew", padx=12, pady=8)

        ttk.Label(self.root, text="live tracks per tile:", font=mono
                  ).grid(row=10, column=0, columnspan=4, sticky="w", **pad)
        self.tile_var = tk.StringVar(value="")
        ttk.Label(self.root, textvariable=self.tile_var, font=mono,
                  justify="left").grid(row=11, column=0, columnspan=4,
                                        sticky="w", **pad)

        btns = ttk.Frame(self.root)
        btns.grid(row=12, column=0, columnspan=4, pady=12)
        self.kill_btn = ttk.Button(btns, text="Kill run", command=self._kill,
                                    state="disabled")
        self.kill_btn.pack(side="left", padx=6)
        ttk.Button(btns, text="Close", command=self.root.destroy
                   ).pack(side="left", padx=6)

    def _kill(self):
        if self.last_pid is None:
            return
        if not messagebox.askyesno("Confirm kill",
                                    f"Send SIGTERM to PID {self.last_pid}?"):
            return
        try:
            os.kill(int(self.last_pid), signal.SIGTERM)
            self.status_var.set(f"sent SIGTERM to {self.last_pid}")
        except Exception as e:
            messagebox.showerror("Kill failed", str(e))

    def _poll(self):
        try:
            self._update_once()
        except Exception as e:
            self.status_var.set(f"monitor error: {e}")
        self.root.after(POLL_MS, self._poll)

    def _update_once(self):
        if not os.path.exists(self.path):
            self.status_var.set(f"no {self.path} yet")
            self.bar["value"] = 0
            self.kill_btn.state(["disabled"])
            return
        try:
            with open(self.path) as f:
                p = json.load(f)
        except Exception:
            return  # mid-write, try again next tick

        status = p.get("status", "?")
        step = p.get("step", "?")
        step_detail = p.get("step_detail", "") or ""
        step_progress = p.get("step_progress")  # 0-1 or null
        now = time.time()
        last = float(p.get("last_update", now))
        stale = (now - last) > STALE_S and status not in ("done", "failed")

        # Header line
        if stale:
            header = f"STALE — no update for {int(now - last)} s"
        elif status == "failed":
            header = f"FAILED: {step_detail}"
        elif status == "done":
            header = "done"
        else:
            header = f"{p.get('command', '?')}: {status}"
        self.status_var.set(header)

        # Step name + detail + per-step progress bar
        self.step_var.set(step)
        self.step_detail_var.set(step_detail[:80])
        if step_progress is None:
            self.bar.configure(mode="indeterminate")
            try:
                self.bar.start(60)
            except tk.TclError:
                pass
        else:
            try:
                self.bar.stop()
            except tk.TclError:
                pass
            self.bar.configure(mode="determinate")
            self.bar["value"] = max(0.0, min(100.0, 100.0 * step_progress))

        self.elapsed_var.set(f"elapsed {_fmt_hms(p.get('elapsed_s'))}")
        self.step_elapsed_var.set(
            f"step {_fmt_hms(p.get('step_elapsed_s'))}")

        # Context-sensitive: which step → what to show
        ctx1, ctx2 = "", ""
        if step in ("detect_and_track", "render_frames"):
            ctx1 = (f"frame {p.get('current_frame', '-')} / "
                    f"{p.get('f_end', '-')}")
            fps_obs = p.get("fps_observed")
            eta = p.get("eta_s")
            ctx2 = (f"fps {fps_obs:.2f}  eta {_fmt_hms(eta)}"
                    if fps_obs is not None else "")
        elif step == "cross_tile_merge":
            ctx1 = f"alive {p.get('merge_alive', '-')}"
            ctx2 = f"merged {p.get('merge_merged', '-')}"
        elif step == "load_models":
            ctx1 = step_detail
        self.ctx1_var.set(ctx1)
        self.ctx2_var.set(ctx2)

        self.source_var.set(f"source: {p.get('source', '-')}")
        self.out_var.set(f"out: {p.get('out_path', '-')}")
        self.last_pid = p.get("pid")
        self.pid_var.set(f"pid: {self.last_pid}")

        per_tile = p.get("per_tile_track_counts", [])
        if per_tile:
            half = (len(per_tile) + 1) // 2
            lines = ["  ".join(f"t{i}:{n:>3}"
                                for i, n in enumerate(per_tile[:half]))]
            if len(per_tile) > half:
                lines.append("  ".join(
                    f"t{i}:{n:>3}"
                    for i, n in enumerate(per_tile[half:], half)))
            self.tile_var.set("\n".join(lines))
        else:
            self.tile_var.set("(no tile counts)")

        if status == "running" and not stale and self.last_pid is not None:
            self.kill_btn.state(["!disabled"])
        else:
            self.kill_btn.state(["disabled"])
        self.last_status = status


def run_monitor(progress_path: str = DEFAULT_PATH):
    root = tk.Tk()
    MonitorApp(root, progress_path)
    root.mainloop()


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default=DEFAULT_PATH)
    a = ap.parse_args()
    run_monitor(a.path)
