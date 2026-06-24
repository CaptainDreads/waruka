# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Stefan Lewis
"""Waruka GUI -- end-to-end pipeline driver.

Launches with ``python -m waruka gui``. Provides a stepped flow:

    1. Open source video (any path on disk).
    2. Calibrate dewarp (auto-skipped if project.json already has pano).
    3. Mark field (auto-skipped if project.json already has homography).
    4. Tweak tracking parameters.
    5. Run the pipeline; tracked broadcast video is written to
       ``<source_dir>/<basename>_broadcast.mp4`` next to the source.

Intermediate artefacts (project.json, tracks.json, players*.json,
campath.json, per-chunk files) live in
``<source_dir>/waruka_tracking/<basename>/`` so the source directory
stays clean apart from the final ``_broadcast.mp4`` output.

Calibrate + markfield are launched as subprocesses pointing at the
existing CLI commands; their OpenCV windows handle the interactive
work. Track/classify/campath/render (or the chunked pipeline) also
run as a subprocess and surface live progress via the same
``_progress.json`` mechanism ``waruka.monitor`` uses.

Framework: PySide6 (Qt 6). Design intentionally leaves the door
open for embedding the calibrate/markfield UIs natively later --
each step is its own widget so swapping the subprocess launcher
for an embedded widget is a localised change.
"""
from __future__ import annotations

import json
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from PySide6 import QtCore, QtGui, QtWidgets

from .jobqueue import (
    Job, JobStage, JobQueue, stage_command, artefact_dir,
    assert_output_safe, _norm_path, _job_keeps_audio,
    STATUS_PENDING, STATUS_RUNNING, STATUS_DONE, STATUS_FAILED,
    STATUS_INTERRUPTED, STAGE_PENDING, STAGE_RUNNING, STAGE_DONE,
    STAGE_FAILED, STAGE_SKIPPED,
)

# Path to the Python interpreter currently running this GUI. Subprocess
# launches reuse the same interpreter so we don't end up running the
# CLI commands under a different Python with different installed deps.
PYTHON = sys.executable

# When the windowed waruka.exe (or any --windowed PyInstaller bundle)
# spawns a console-subsystem binary like ffmpeg.exe, Windows allocates
# a fresh console window for the child by default -- visible as a brief
# flash. CREATE_NO_WINDOW suppresses that without changing any other
# behaviour. Use this flag-dict in `creationflags=` on any subprocess
# call to a console binary.
_NO_WINDOW_KW: dict = {}
if sys.platform == "win32":
    import subprocess as _sp_for_flags
    _NO_WINDOW_KW = {"creationflags": _sp_for_flags.CREATE_NO_WINDOW}

# Directory CONTAINING the waruka package (i.e. the project root, not the
# package dir itself). We point subprocess PYTHONPATH at this so that
# ``python -m waruka ...`` resolves even when cwd is the artefact dir
# (which lives several levels away from the source tree).
WARUKA_PARENT = str(Path(__file__).resolve().parent.parent)


# --------------------------------------------------------------------------
# Path derivation and video probing
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class WarukaPaths:
    """All derived paths for a given source video.

    Naming rules (from user spec):
      * artefacts: ``<source_dir>/waruka_tracking/<basename>/``
      * final output: ``<source_dir>/<basename>_broadcast.mp4`` (right next to
        the source, suffix added before .mp4).
    """
    source: Path
    source_dir: Path
    basename: str
    artefact_dir: Path
    project_json: Path
    output_video: Path

    @classmethod
    def for_video(cls, source_video: str | Path) -> "WarukaPaths":
        src = Path(source_video).resolve()
        basename = src.stem
        source_dir = src.parent
        artefact_dir = source_dir / "waruka_tracking" / basename
        return cls(
            source=src,
            source_dir=source_dir,
            basename=basename,
            artefact_dir=artefact_dir,
            project_json=artefact_dir / "project.json",
            output_video=source_dir / f"{basename}_broadcast.mp4",
        )


@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    fps: float
    n_frames: int
    duration_s: float
    has_audio: bool

    @classmethod
    def probe(cls, path: str | Path) -> "VideoInfo | None":
        """Read video metadata. Geometry via OpenCV (fast), audio
        presence via the bundled ffmpeg binary (parses -i stderr for
        ``Stream #N:M[...]: Audio:`` lines)."""
        import cv2
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            cap.release()
            return None
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        dur = (n / fps) if fps > 0 else 0.0
        return cls(width=w, height=h, fps=fps, n_frames=n,
                   duration_s=dur, has_audio=_probe_audio(path))


def _probe_audio(path: str | Path) -> bool:
    """Return True if the file has at least one audio stream.

    Uses imageio_ffmpeg's bundled ffmpeg binary so we don't need a
    separate ffprobe on PATH. ``ffmpeg -i <file>`` exits non-zero
    because there's no output, but its stderr lists the input streams
    -- we just grep that for an Audio stream line. Returns False on
    any error (treated as 'no detectable audio')."""
    import subprocess
    try:
        import imageio_ffmpeg
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return False
    try:
        out = subprocess.run(
            [ffmpeg, "-hide_banner", "-i", str(path)],
            capture_output=True, text=True, timeout=10,
            **_NO_WINDOW_KW,
        )
    except Exception:
        return False
    # ffmpeg writes stream info to stderr regardless of exit code.
    text = (out.stderr or "") + (out.stdout or "")
    for line in text.splitlines():
        if "Stream #" in line and ": Audio:" in line:
            return True
    return False


# --------------------------------------------------------------------------
# Step status detection
# --------------------------------------------------------------------------

STEP_PENDING = "pending"
STEP_DONE = "done"
STEP_RUNNING = "running"
STEP_BLOCKED = "blocked"   # depends on an earlier step


def detect_step_status(paths: WarukaPaths) -> dict[str, str]:
    """Inspect on-disk state and return per-step status.

    'calibrate' is considered done as soon as a project.json exists at
    the expected location (the calibrate UI is the only thing that
    saves it; an existence check is enough).

    'markfield' is considered done when the project file has a
    homography written -- that field is None until markfield runs.

    'params' is always 'pending' (it's a form, never "done"). The
    process step is blocked until markfield is done.
    """
    status = {
        "calibrate": STEP_PENDING,
        "markfield": STEP_PENDING,
        "params": STEP_PENDING,
        "process": STEP_BLOCKED,
    }
    pj = paths.project_json
    if not pj.exists():
        return status

    # Project file present -- calibrate has at least been saved once.
    status["calibrate"] = STEP_DONE

    try:
        from .config import ProjectConfig
        cfg = ProjectConfig.load(pj)
    except Exception:
        # File exists but is unreadable -- treat downstream as blocked.
        return status

    if cfg.homography is not None and len(cfg.field_marks.get("corners", [])) >= 4:
        status["markfield"] = STEP_DONE
        status["process"] = STEP_PENDING

    return status


# --------------------------------------------------------------------------
# Widgets
# --------------------------------------------------------------------------

class VideoPickerWidget(QtWidgets.QGroupBox):
    """File-picker for the source video. Emits ``video_selected`` once a
    valid video has been picked + probed.

    Also accepts external drag-drop of a single video file onto the
    whole group box; the drop flows through ``set_video`` so
    artefact-dir detection / project.json handling / status update
    all behave exactly as if the user had clicked Open."""

    video_selected = QtCore.Signal(object, object)  # (WarukaPaths, VideoInfo)

    _BASE_TITLE = "Source video"
    _DROP_TITLE = "Source video  -- drop here to load"

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(self._BASE_TITLE, parent)
        self._paths: WarukaPaths | None = None
        self._info: VideoInfo | None = None
        self.setAcceptDrops(True)

        layout = QtWidgets.QGridLayout(self)
        layout.setVerticalSpacing(4)

        # Row 0: path label + open button
        self.path_label = QtWidgets.QLabel("<no video selected>")
        self.path_label.setStyleSheet("font-family: Consolas, monospace;")
        self.path_label.setWordWrap(True)
        layout.addWidget(self.path_label, 0, 0)

        self.open_btn = QtWidgets.QPushButton("Open video...")
        self.open_btn.clicked.connect(self._pick_video)
        layout.addWidget(self.open_btn, 0, 1)

        # Row 1: probe info
        self.info_label = QtWidgets.QLabel("")
        self.info_label.setStyleSheet("color: #555;")
        layout.addWidget(self.info_label, 1, 0, 1, 2)

        # Row 2-3: artefact dir + output path
        self.artefact_label = QtWidgets.QLabel("")
        self.artefact_label.setStyleSheet(
            "font-family: Consolas, monospace; color: #444;")
        self.artefact_label.setWordWrap(True)
        layout.addWidget(self.artefact_label, 2, 0, 1, 2)

        self.output_label = QtWidgets.QLabel("")
        self.output_label.setStyleSheet(
            "font-family: Consolas, monospace; color: #444;")
        self.output_label.setWordWrap(True)
        layout.addWidget(self.output_label, 3, 0, 1, 2)

    def _pick_video(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Choose source video",
            "",
            "Video files (*.mp4 *.mov *.mkv *.avi);;All files (*.*)",
        )
        if not path:
            return
        self.set_video(path)

    def set_video(self, path: str | Path) -> None:
        """Load a video by path (used by both the picker and external code)."""
        paths = WarukaPaths.for_video(path)
        info = VideoInfo.probe(paths.source)
        if info is None:
            QtWidgets.QMessageBox.warning(
                self, "Couldn't open video",
                f"OpenCV could not open:\n{paths.source}",
            )
            return

        # Ensure artefact dir exists so calibrate/markfield can write into it.
        paths.artefact_dir.mkdir(parents=True, exist_ok=True)

        self._paths = paths
        self._info = info
        self.path_label.setText(str(paths.source))
        self.info_label.setText(
            f"{info.width}x{info.height}  {info.fps:.2f} fps  "
            f"{info.duration_s:.1f} s  ({info.n_frames} frames)"
        )
        self.artefact_label.setText(f"artefacts: {paths.artefact_dir}")
        self.output_label.setText(f"output:    {paths.output_video}")
        self.video_selected.emit(paths, info)

    @property
    def paths(self) -> WarukaPaths | None:
        return self._paths

    @property
    def info(self) -> VideoInfo | None:
        return self._info

    # Drag-drop wiring ----------------------------------------------------

    def _drop_video_path(self, ev) -> str | None:
        md = ev.mimeData()
        if not md.hasUrls():
            return None
        for u in md.urls():
            if not u.isLocalFile():
                continue
            path = u.toLocalFile()
            if Path(path).suffix.lower() in ConcatTab.VIDEO_EXTS:
                return path
        return None

    def dragEnterEvent(self, ev: QtGui.QDragEnterEvent) -> None:
        if self._drop_video_path(ev):
            self.setTitle(self._DROP_TITLE)
            ev.acceptProposedAction()
            return
        super().dragEnterEvent(ev)

    def dragMoveEvent(self, ev: QtGui.QDragMoveEvent) -> None:
        if self._drop_video_path(ev):
            ev.acceptProposedAction()
            return
        super().dragMoveEvent(ev)

    def dragLeaveEvent(self, ev: QtGui.QDragLeaveEvent) -> None:
        self.setTitle(self._BASE_TITLE)
        super().dragLeaveEvent(ev)

    def dropEvent(self, ev: QtGui.QDropEvent) -> None:
        self.setTitle(self._BASE_TITLE)
        path = self._drop_video_path(ev)
        if path is not None:
            ev.acceptProposedAction()
            self.set_video(path)
            return
        super().dropEvent(ev)


class StepCardWidget(QtWidgets.QFrame):
    """One step in the vertical step list.

    Each card has a fixed label, a status indicator, and an action
    button. The action callback is supplied at construction time; the
    parent rebuilds the status indicator via ``set_status`` whenever
    on-disk state changes.
    """

    def __init__(
        self,
        number: int,
        title: str,
        action_label: str,
        on_action: Callable[[], None],
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.setStyleSheet(
            "StepCardWidget { background: #fafafa; border: 1px solid #ddd; "
            "border-radius: 4px; }"
        )
        self._on_action = on_action

        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)

        self.status_dot = QtWidgets.QLabel()
        self.status_dot.setFixedWidth(18)
        layout.addWidget(self.status_dot)

        title_label = QtWidgets.QLabel(f"{number}. {title}")
        title_label.setStyleSheet("font-size: 13px; font-weight: 500;")
        layout.addWidget(title_label, 1)

        self.status_text = QtWidgets.QLabel("pending")
        self.status_text.setStyleSheet("color: #888; font-size: 11px;")
        layout.addWidget(self.status_text)

        self.action_btn = QtWidgets.QPushButton(action_label)
        self.action_btn.clicked.connect(self._fire)
        self.action_btn.setEnabled(False)
        layout.addWidget(self.action_btn)

        self.set_status(STEP_BLOCKED)

    def _fire(self) -> None:
        self._on_action()

    def set_status(self, status: str) -> None:
        colour = {
            STEP_PENDING: "#888",
            STEP_DONE: "#2a7",
            STEP_RUNNING: "#c84",
            STEP_BLOCKED: "#bbb",
        }.get(status, "#888")
        glyph = {
            STEP_PENDING: "•",   # bullet
            STEP_DONE: "✓",      # check
            STEP_RUNNING: "▶",   # play triangle
            STEP_BLOCKED: "•",
        }.get(status, "•")
        self.status_dot.setText(
            f'<span style="color: {colour}; font-size: 18px;">{glyph}</span>'
        )
        self.status_text.setText(status)
        # Action button: disabled when blocked, enabled otherwise.
        self.action_btn.setEnabled(status != STEP_BLOCKED)
        # Update button label so re-runs are obvious.
        if status == STEP_DONE and self.action_btn.text().startswith("Run "):
            self.action_btn.setText("Re-run")


# --------------------------------------------------------------------------
# Processing parameters
# --------------------------------------------------------------------------

# Auto-pick threshold: clips shorter than this run sequential (pixel-perfect,
# no chunk-0 residual). Anything longer runs the chunked pipeline for the
# wall-time saving. 120 s lines up with the v0.12 handover guidance.
PIPELINE_AUTO_MIN_DURATION_S = 120.0


@dataclass
class ProcessingParams:
    """User-tweakable knobs surfaced in the params dialog.

    Mirrors the high-value CLI defaults from ``waruka track`` / ``waruka pipeline``
    rather than re-deriving them. Anything not in this dataclass is left at
    the CLI default (production-tuned in v0.12).
    """
    # Both time bounds are optional. None means "use the CLI's natural
    # default" -- t0 omitted -> 0 (start of video); t1 omitted -> end of
    # video. Sidesteps the off-by-one-frame issue with cv2's container
    # frame count at the very last frame: if the user never types an
    # explicit end value, we never pass --t1 to the subprocess and the
    # reader stops when the actual stream ends.
    t0: float | None = None
    t1: float | None = None
    # Default sequential: pixel-perfect, no chunk-0 residual. Pipeline is
    # the opt-in secondary mode for long matches where the ~33-50% wall
    # saving matters and the first-20s drift is acceptable.
    mode: str = "sequential"      # "sequential" | "pipeline" | "auto" (legacy)
    stride: int = 3
    view_mode: str = "default"    # "default" | "wide"
    # When True AND the source has audio, the process step ALSO writes a
    # silent companion file next to the main output (named with the
    # _no_audio.mp4 suffix). When source has no audio this is forced
    # False at the UI layer.
    create_no_audio_copy: bool = True
    output_path: str = ""         # absolute path; pre-filled from WarukaPaths
    # Optional post-render frame interpolation (#18). When interpolate_fps
    # is 0 the interpolate step is skipped and render writes straight to
    # output_path. When > 0 (and != src_fps) render writes to a raw
    # intermediate file and `waruka interpolate` brings it to the target.
    interpolate_fps: int = 0
    interpolate_backend: str = "rife"  # "rife" (default, fast) | "film" (slow)
    # Optional source-crop super-resolution during render (#41). When True
    # the GpuRenderer runs Real-ESRGAN x2 on the per-frame source crop,
    # then resamples to output. Auto-bypassed per-frame when the source
    # crop is already large enough relative to output.
    sr_enabled: bool = False

    def processed_duration(self, video_duration_s: float) -> float:
        """How many seconds we'll actually process (after applying any
        user-specified trim). Used for mode auto-resolution."""
        t0 = self.t0 if self.t0 is not None else 0.0
        t1 = self.t1 if self.t1 is not None else video_duration_s
        return max(0.0, t1 - t0)

    def effective_mode(self, video_duration_s: float) -> str:
        """Resolve 'auto' to either 'sequential' or 'pipeline' based on
        the actual processed duration (not the full clip length)."""
        if self.mode != "auto":
            return self.mode
        return ("pipeline"
                if self.processed_duration(video_duration_s)
                   >= PIPELINE_AUTO_MIN_DURATION_S
                else "sequential")


class ParamsDialog(QtWidgets.QDialog):
    """Modal form for the tracking parameters.

    Pre-fills sensible defaults derived from the loaded video (t1 = full
    duration, output_path = the standard _broadcast.mp4 location). Returns
    the configured ProcessingParams via ``result_params`` when the user
    clicks OK.
    """

    def __init__(
        self,
        paths: WarukaPaths,
        info: VideoInfo,
        initial: ProcessingParams,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Tracking parameters")
        self.setModal(True)
        self.resize(560, 0)
        self._paths = paths
        self._info = info
        self.result_params: ProcessingParams | None = None

        form = QtWidgets.QFormLayout()
        form.setLabelAlignment(QtCore.Qt.AlignRight)

        # --- Time window ---------------------------------------------------
        # Both bounds are optional. Leaving t1 empty means "let the
        # subprocess run to the natural end of the stream" -- avoids
        # the off-by-one-frame issue with cv2's container frame count.
        # Validator caps explicit values at the probed duration.
        time_validator = QtGui.QDoubleValidator(0.0, info.duration_s, 3)
        time_validator.setNotation(QtGui.QDoubleValidator.StandardNotation)

        self.t0_edit = QtWidgets.QLineEdit()
        self.t0_edit.setPlaceholderText("beginning of clip")
        self.t0_edit.setValidator(time_validator)
        if initial.t0 is not None:
            self.t0_edit.setText(f"{initial.t0:g}")
        form.addRow("Start time (s):", self.t0_edit)

        self.t1_edit = QtWidgets.QLineEdit()
        self.t1_edit.setPlaceholderText("end of clip")
        self.t1_edit.setValidator(time_validator)
        if initial.t1 is not None:
            self.t1_edit.setText(f"{initial.t1:g}")
        form.addRow(f"End time (s, max {info.duration_s:.1f}):",
                    self.t1_edit)

        # --- Mode ----------------------------------------------------------
        # Sequential is the safe default (pixel-perfect, no chunk-0 drift).
        # Pipeline is the secondary opt-in for long matches where the wall-
        # time saving outweighs the known small first-20s framing drift.
        self.mode_combo = QtWidgets.QComboBox()
        self.mode_combo.addItem(
            "Sequential -- pixel-perfect (default)", "sequential")
        self.mode_combo.addItem(
            "Pipeline -- chunked, ~33-50% faster (slight first-20s drift)",
            "pipeline")
        idx = self.mode_combo.findData(initial.mode)
        if idx < 0:
            idx = 0  # falls back to sequential if initial.mode is legacy "auto"
        self.mode_combo.setCurrentIndex(idx)
        form.addRow("Processing mode:", self.mode_combo)

        # --- Stride --------------------------------------------------------
        self.stride_spin = QtWidgets.QSpinBox()
        self.stride_spin.setRange(1, 10)
        self.stride_spin.setValue(initial.stride)
        self.stride_spin.setToolTip(
            "Detection runs every Nth frame; output is interpolated to every "
            "frame. Higher = faster but more track fragmentation. v0.12 "
            "production default 3.")
        form.addRow("Detection stride:", self.stride_spin)

        # --- View mode -----------------------------------------------------
        self.view_combo = QtWidgets.QComboBox()
        self.view_combo.addItem("default (tight, natural)", "default")
        self.view_combo.addItem("wide (more breathing room)", "wide")
        idx = self.view_combo.findData(initial.view_mode)
        self.view_combo.setCurrentIndex(max(0, idx))
        form.addRow("View mode:", self.view_combo)

        # --- Audio copy ----------------------------------------------------
        # The main tracked output preserves source audio (via post-render
        # ffmpeg mux). If the source has audio AND this is checked, we
        # ALSO write a silent companion named <basename>_broadcast_no_audio.mp4.
        # If the source has no audio at all, the checkbox is disabled and
        # an explanatory note appears.
        audio_widget = QtWidgets.QWidget()
        audio_layout = QtWidgets.QVBoxLayout(audio_widget)
        audio_layout.setContentsMargins(0, 0, 0, 0)
        audio_layout.setSpacing(2)
        self.no_audio_check = QtWidgets.QCheckBox(
            "also write a silent copy (<output>_no_audio.mp4)")
        if info.has_audio:
            self.no_audio_check.setChecked(initial.create_no_audio_copy)
            self.no_audio_check.setEnabled(True)
            note_text = ("Source has an audio track; tracked output "
                         "will preserve it by default.")
        else:
            self.no_audio_check.setChecked(False)
            self.no_audio_check.setEnabled(False)
            note_text = ("Source has no audio track; tracked output "
                         "will be silent.")
        audio_layout.addWidget(self.no_audio_check)
        note = QtWidgets.QLabel(note_text)
        note.setStyleSheet("color: #777; font-size: 10px;")
        note.setWordWrap(True)
        audio_layout.addWidget(note)
        form.addRow("Audio:", audio_widget)

        # --- Frame interpolation (post-render step, #18) -------------------
        interpolate_widget = QtWidgets.QWidget()
        interpolate_layout = QtWidgets.QVBoxLayout(interpolate_widget)
        interpolate_layout.setContentsMargins(0, 0, 0, 0)
        interpolate_layout.setSpacing(2)

        row = QtWidgets.QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        self.interpolate_fps_combo = QtWidgets.QComboBox()
        self.interpolate_fps_combo.addItem("Off (keep render fps)", 0)
        self.interpolate_fps_combo.addItem("40 fps (2x)", 40)
        self.interpolate_fps_combo.addItem("60 fps (3x, recommended)", 60)
        self.interpolate_fps_combo.addItem("80 fps (4x)", 80)
        idx = self.interpolate_fps_combo.findData(initial.interpolate_fps)
        self.interpolate_fps_combo.setCurrentIndex(max(0, idx))
        row.addWidget(self.interpolate_fps_combo, 1)
        self.interpolate_backend_combo = QtWidgets.QComboBox()
        self.interpolate_backend_combo.addItem("RIFE 4.25 (fast)", "rife")
        self.interpolate_backend_combo.addItem("FILM-Style (very slow!)", "film")
        bidx = self.interpolate_backend_combo.findData(initial.interpolate_backend)
        self.interpolate_backend_combo.setCurrentIndex(max(0, bidx))
        row.addWidget(self.interpolate_backend_combo, 1)
        interpolate_layout.addLayout(row)

        self.interpolate_note = QtWidgets.QLabel()
        self.interpolate_note.setWordWrap(True)
        self.interpolate_note.setStyleSheet("color: #777; font-size: 10px;")
        interpolate_layout.addWidget(self.interpolate_note)

        # Live update of the note as the user changes either combo.
        self.interpolate_fps_combo.currentIndexChanged.connect(self._update_interpolate_note)
        self.interpolate_backend_combo.currentIndexChanged.connect(self._update_interpolate_note)
        self._update_interpolate_note()
        form.addRow("Frame interp:", interpolate_widget)

        # --- Source-crop super-resolution (#41) ---------------------------
        sr_widget = QtWidgets.QWidget()
        sr_layout = QtWidgets.QVBoxLayout(sr_widget)
        sr_layout.setContentsMargins(0, 0, 0, 0)
        sr_layout.setSpacing(2)
        self.sr_check = QtWidgets.QCheckBox(
            "Real-ESRGAN x2 upscale on the source crop")
        self.sr_check.setChecked(initial.sr_enabled)
        sr_layout.addWidget(self.sr_check)
        self.sr_note = QtWidgets.QLabel()
        self.sr_note.setWordWrap(True)
        self.sr_note.setStyleSheet("color: #777; font-size: 10px;")
        sr_layout.addWidget(self.sr_note)
        self.sr_check.toggled.connect(self._update_sr_note)
        self._update_sr_note()
        form.addRow("Upscale:", sr_widget)

        # --- Output path ---------------------------------------------------
        out_widget = QtWidgets.QWidget()
        out_layout = QtWidgets.QHBoxLayout(out_widget)
        out_layout.setContentsMargins(0, 0, 0, 0)
        self.output_edit = QtWidgets.QLineEdit(
            initial.output_path or str(paths.output_video))
        self.output_edit.setStyleSheet("font-family: Consolas, monospace;")
        browse_btn = QtWidgets.QPushButton("Browse...")
        browse_btn.clicked.connect(self._pick_output)
        out_layout.addWidget(self.output_edit, 1)
        out_layout.addWidget(browse_btn)
        form.addRow("Output video:", out_widget)

        # --- Buttons -------------------------------------------------------
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)

        outer = QtWidgets.QVBoxLayout(self)
        outer.addLayout(form)
        outer.addWidget(buttons)

    def _update_sr_note(self) -> None:
        if self.sr_check.isChecked():
            self.sr_note.setText(
                "ON -- Real-ESRGAN x2 runs on every frame's source crop "
                "before the final resample. Constant sharpening across "
                "all zooms (no per-frame bypass, so no visible 'pop' at "
                "framing transitions). Adds ~150-1200 ms per frame "
                "depending on framing; roughly 5-10x slower render. "
                "~10-20 h for a 100-min match overnight.")
            self.sr_note.setStyleSheet("color: #777; font-size: 10px;")
        else:
            self.sr_note.setText(
                "Off -- broadcast video is resampled directly to output "
                "resolution (faster; some softness at very tight zooms).")
            self.sr_note.setStyleSheet("color: #777; font-size: 10px;")

    def _update_interpolate_note(self) -> None:
        fps = self.interpolate_fps_combo.currentData()
        backend = self.interpolate_backend_combo.currentData()
        # Backend combo only matters when interp is on. Disable it visually
        # otherwise.
        self.interpolate_backend_combo.setEnabled(fps != 0)
        if fps == 0:
            self.interpolate_note.setText(
                "Off -- the broadcast video is written at the render's "
                "native fps (20 fps).")
            self.interpolate_note.setStyleSheet("color: #777; font-size: 10px;")
            return
        if backend == "film":
            self.interpolate_note.setText(
                f"WARNING: FILM at {fps} fps for a 100-min match takes "
                f"roughly {[None, None, 33, 66, 133][fps // 20]} h on an "
                "RTX 2080 Ti. RIFE is ~4x faster and visually equivalent on "
                "ultimate-frisbee footage. Only pick FILM for special "
                "renders you're prepared to wait days for.")
            self.interpolate_note.setStyleSheet(
                "color: #b00; font-size: 10px; font-weight: bold;")
        else:
            est_h = {40: 8, 60: 16, 80: 25}[fps]
            self.interpolate_note.setText(
                f"RIFE 4.25 at {fps} fps -- about {est_h} h of compute for "
                "a 100-min match on an RTX 2080 Ti (single-overnight at "
                "60 fps; comfortable for batch use).")
            self.interpolate_note.setStyleSheet("color: #777; font-size: 10px;")

    def _pick_output(self) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Output video", self.output_edit.text(),
            "MP4 files (*.mp4);;All files (*.*)")
        if path:
            self.output_edit.setText(path)

    def _accept(self) -> None:
        # Parse the optional time inputs. Empty string -> None means
        # "use the CLI's natural default" (start-of-clip / end-of-clip).
        def _parse_optional(text: str, label: str) -> float | None:
            s = text.strip()
            if not s:
                return None
            try:
                return float(s)
            except ValueError:
                QtWidgets.QMessageBox.warning(
                    self, "Invalid number",
                    f"{label} must be a number or left blank.")
                raise
        try:
            t0 = _parse_optional(self.t0_edit.text(), "Start time")
            t1 = _parse_optional(self.t1_edit.text(), "End time")
        except ValueError:
            return
        if t0 is not None and t1 is not None and t1 <= t0:
            QtWidgets.QMessageBox.warning(
                self, "Invalid time window",
                f"End time ({t1:g}s) must be greater than start ({t0:g}s).")
            return
        # Validator caps individual values to [0, duration], but a user
        # could still leave the field empty (validator only fires on type)
        # so we don't need an out-of-range check here -- empty is allowed
        # and explicit values were range-checked at edit time.
        out_path = self.output_edit.text().strip()
        if not out_path:
            QtWidgets.QMessageBox.warning(
                self, "Missing output path",
                "Please specify an output video path.")
            return
        try:
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            QtWidgets.QMessageBox.warning(
                self, "Output path not writable",
                f"Couldn't create parent directory:\n{e}")
            return
        self.result_params = ProcessingParams(
            t0=t0,                      # may be None -> CLI default 0
            t1=t1,                      # may be None -> CLI default = end
            mode=self.mode_combo.currentData(),
            stride=int(self.stride_spin.value()),
            view_mode=self.view_combo.currentData(),
            create_no_audio_copy=self.no_audio_check.isChecked(),
            output_path=out_path,
            interpolate_fps=int(self.interpolate_fps_combo.currentData()),
            interpolate_backend=str(self.interpolate_backend_combo.currentData()),
            sr_enabled=self.sr_check.isChecked(),
        )
        self.accept()


# --------------------------------------------------------------------------
# Subprocess runner -- shared by all step launchers
# --------------------------------------------------------------------------

class StepRunner(QtCore.QObject):
    """Wraps QProcess for one-shot CLI launches.

    Emits ``started`` when the subprocess begins, ``finished(exit_code)``
    when it exits. Captures stdout/stderr to a log buffer the caller can
    surface in an error dialog if the exit code is non-zero.

    Only one StepRunner should run at a time per MainWindow; the parent
    is responsible for serialising launches.
    """

    started = QtCore.Signal()
    finished = QtCore.Signal(int)
    line = QtCore.Signal(str)

    def __init__(self, parent: QtCore.QObject | None = None) -> None:
        super().__init__(parent)
        self._proc: QtCore.QProcess | None = None
        self._log: list[str] = []

    def start(self, args: list[str], cwd: str | Path | None = None,
              program: str | None = None) -> None:
        """Spawn ``program args...``. Default program is ``PYTHON``
        (the GUI's own interpreter) so the typical case --
        ``StepRunner.start(["-m", "waruka", "calibrate", ...])`` -- still
        works without callers having to specify the binary.

        For non-Python launches (ffmpeg etc.) pass ``program=`` to
        override. PYTHONPATH is only injected when running PYTHON.

        ``cwd`` (optional) sets the subprocess working directory, useful
        when the binary writes side-files (e.g. ``_progress.json``) and
        you want them to land somewhere specific.
        """
        if self._proc is not None:
            raise RuntimeError("StepRunner is already running")
        self._log = []
        proc = QtCore.QProcess(self)
        # In the frozen bundle, sys.executable is waruka.exe (windowed
        # subsystem). Launching subcommands through it works (the launcher
        # strips the "-m waruka" prefix) BUT the windowed exe redirects
        # stderr to a log file via SetStdHandle, so QProcess sees no
        # output and the error dialog shows "(no output captured)".
        # Switch to waruka-cli.exe (console subsystem) which keeps the
        # standard pipes attached to its parent. Drop "-m waruka" from
        # args since waruka-cli.exe doesn't need it.
        if program is None and getattr(sys, "frozen", False):
            cli_exe = Path(sys.executable).with_name("waruka-cli.exe")
            if cli_exe.is_file():
                program = str(cli_exe)
                if len(args) >= 2 and args[0] == "-m" and args[1] == "waruka":
                    args = args[2:]
        prog = program if program is not None else PYTHON
        proc.setProgram(prog)
        proc.setArguments(args)
        if cwd is not None:
            proc.setWorkingDirectory(str(cwd))

        # Only patch PYTHONPATH when actually invoking the Python
        # interpreter -- it's irrelevant to ffmpeg and other binaries.
        if prog == PYTHON:
            env = QtCore.QProcessEnvironment.systemEnvironment()
            existing_pp = env.value("PYTHONPATH", "")
            sep = ";" if sys.platform == "win32" else ":"
            if existing_pp:
                env.insert("PYTHONPATH",
                           f"{WARUKA_PARENT}{sep}{existing_pp}")
            else:
                env.insert("PYTHONPATH", WARUKA_PARENT)
            proc.setProcessEnvironment(env)

        proc.setProcessChannelMode(QtCore.QProcess.MergedChannels)
        proc.readyReadStandardOutput.connect(self._drain)
        proc.finished.connect(self._on_finished)
        proc.errorOccurred.connect(self._on_error)
        # Suppress console allocation for console-subsystem children
        # (waruka-cli.exe, ffmpeg.exe). Same reason as `_NO_WINDOW_KW`
        # above: windowed parent + console child = Windows allocates a
        # fresh console = visible flash. Qt exposes this via the
        # creation-args modifier callback (Windows-only no-op elsewhere).
        if sys.platform == "win32":
            def _no_window(args):
                import subprocess as _sp
                args.flags |= _sp.CREATE_NO_WINDOW
                return args
            try:
                proc.setCreateProcessArgumentsModifier(_no_window)
            except AttributeError:
                # PySide6 < 6.2 doesn't have this; safe to skip.
                pass
        self._proc = proc
        proc.start()
        self.started.emit()

    def is_running(self) -> bool:
        return self._proc is not None

    def log(self) -> str:
        return "".join(self._log)

    def _drain(self) -> None:
        if self._proc is None:
            return
        data = bytes(self._proc.readAllStandardOutput()).decode(
            "utf-8", errors="replace")
        if not data:
            return
        self._log.append(data)
        for ln in data.splitlines():
            self.line.emit(ln)

    def _on_finished(self, exit_code: int, exit_status) -> None:
        self._drain()  # one last grab in case anything's buffered
        proc, self._proc = self._proc, None
        if proc is not None:
            proc.deleteLater()
        self.finished.emit(int(exit_code))

    def _on_error(self, err) -> None:
        # Surfaced as a non-zero exit (Qt also fires `finished` separately
        # for most error paths, but failedToStart skips it).
        self._log.append(f"\n[QProcess error] {err}\n")
        if self._proc is not None and self._proc.state() == QtCore.QProcess.NotRunning:
            # No finished signal will come -- synthesise one.
            proc, self._proc = self._proc, None
            proc.deleteLater()
            self.finished.emit(-1)


# --------------------------------------------------------------------------
# Process dialog -- orchestrates the multi-stage tracking pipeline
# --------------------------------------------------------------------------

# Polling interval for the live progress file.
PROGRESS_POLL_MS = 500


def _fmt_hms(seconds: float | None) -> str:
    if seconds is None or seconds < 0:
        return "--:--:--"
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


class ProcessDialog(QtWidgets.QDialog):
    """Modal dialog that runs the tracking pipeline + audio post-process.

    State machine (sequential mode):
        track -> classify -> campath -> render -> audio_mux -> done

    Pipeline mode collapses track/classify/campath/render into one
    ``waruka pipeline`` subprocess, but keeps the final audio_mux step.

    Each stage is a subprocess. Live progress for the long stages
    (track, render, pipeline) is read from the artefact dir's
    ``_progress.json`` (the same file ``waruka.monitor`` polls).
    Short stages (classify, campath, audio_mux) just show an
    indeterminate spinner.
    """

    # State labels -- also used as the "stage" header text.
    STATE_IDLE = "idle"
    STATE_DONE = "done"
    STATE_FAILED = "failed"
    # Per-stage states are dynamic (set when entering each stage).

    def __init__(
        self,
        paths: WarukaPaths,
        info: VideoInfo,
        params: ProcessingParams,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Processing")
        self.setModal(True)
        self.resize(640, 360)
        self._paths = paths
        self._info = info
        self._params = params

        # Effective processing mode after auto-resolution
        self._mode = params.effective_mode(info.duration_s)

        # Build the queue of stages we'll execute in order.
        self._stages: list[dict] = self._plan_stages()
        self._stage_idx = 0
        self._state = self.STATE_IDLE
        self._final_outputs: list[Path] = []
        self._failure_log: str = ""

        # Subprocess runner + progress poller
        self._runner = StepRunner(self)
        self._runner.line.connect(self._on_runner_line)
        self._runner.finished.connect(self._on_stage_finished)

        self._poll_timer = QtCore.QTimer(self)
        self._poll_timer.setInterval(PROGRESS_POLL_MS)
        self._poll_timer.timeout.connect(self._poll_progress)

        # --- UI -----------------------------------------------------------
        outer = QtWidgets.QVBoxLayout(self)

        # Header: stage name + counter
        self.stage_label = QtWidgets.QLabel("Ready to run")
        self.stage_label.setStyleSheet(
            "font-size: 14px; font-weight: 600;")
        outer.addWidget(self.stage_label)

        self.stage_detail = QtWidgets.QLabel("")
        self.stage_detail.setStyleSheet(
            "color: #555; font-family: Consolas, monospace;")
        outer.addWidget(self.stage_detail)

        self.bar = QtWidgets.QProgressBar()
        self.bar.setRange(0, 100)
        self.bar.setValue(0)
        outer.addWidget(self.bar)

        # Two-line status row: elapsed / eta on one row, fps / current on other
        row = QtWidgets.QHBoxLayout()
        self.elapsed_label = QtWidgets.QLabel("elapsed --:--:--")
        self.eta_label = QtWidgets.QLabel("eta --:--:--")
        self.fps_label = QtWidgets.QLabel("")
        for w in (self.elapsed_label, self.eta_label, self.fps_label):
            row.addWidget(w)
        row.addStretch(1)
        outer.addLayout(row)

        # Log toggle + log area (hidden by default)
        log_row = QtWidgets.QHBoxLayout()
        self.show_log_btn = QtWidgets.QPushButton("Show details")
        self.show_log_btn.setCheckable(True)
        self.show_log_btn.toggled.connect(self._toggle_log)
        log_row.addWidget(self.show_log_btn)
        log_row.addStretch(1)
        outer.addLayout(log_row)

        self.log_view = QtWidgets.QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(2000)
        self.log_view.setStyleSheet(
            "font-family: Consolas, monospace; font-size: 10px; "
            "background: #1e1e1e; color: #d0d0d0;")
        self.log_view.setVisible(False)
        outer.addWidget(self.log_view, 1)

        # Output paths preview (populated when done)
        self.output_list = QtWidgets.QListWidget()
        self.output_list.setVisible(False)
        self.output_list.setMaximumHeight(80)
        outer.addWidget(self.output_list)

        # Button row
        btns = QtWidgets.QHBoxLayout()
        self.kill_btn = QtWidgets.QPushButton("Kill run")
        self.kill_btn.setEnabled(False)
        self.kill_btn.clicked.connect(self._on_kill)
        btns.addWidget(self.kill_btn)

        self.open_folder_btn = QtWidgets.QPushButton("Open output folder")
        self.open_folder_btn.setVisible(False)
        self.open_folder_btn.clicked.connect(self._open_output_folder)
        btns.addWidget(self.open_folder_btn)

        btns.addStretch(1)

        self.close_btn = QtWidgets.QPushButton("Close")
        self.close_btn.setEnabled(False)  # disabled while running
        self.close_btn.clicked.connect(self.close)
        btns.addWidget(self.close_btn)

        outer.addLayout(btns)

    # ----- stage planning --------------------------------------------------

    def _plan_stages(self) -> list[dict]:
        """Build the ordered list of stages to run, each described as:

            {"name": str, "args": [...post-python argv...], "uses_progress": bool}

        The argv is what gets passed to StepRunner.start (i.e. after
        ``python``). Paths are absolute so cwd doesn't matter for
        resolution, but cwd is still set to artefact_dir so each
        stage's ``_progress.json`` lands there.
        """
        p = self._paths
        a = self._params
        proj = str(p.project_json)
        video = str(p.source)
        # Only include --t0/--t1 if the user explicitly set them; otherwise
        # let the CLI's defaults apply (t0=0, t1=end-of-stream). Sidesteps
        # the off-by-one-frame issue with container frame counts.
        time_args: list[str] = []
        if a.t0 is not None:
            time_args += ["--t0", f"{a.t0:.3f}"]
        if a.t1 is not None:
            time_args += ["--t1", f"{a.t1:.3f}"]

        # When interpolation is on, render writes to a raw intermediate
        # file in the artefact dir; the interpolate stage brings it to the
        # final user-chosen output_path. Audio mux then runs on the final.
        interp_on = (a.interpolate_fps and a.interpolate_fps > 0)
        if interp_on:
            raw_output = str(p.artefact_dir / "broadcast_raw.mp4")
            render_target = raw_output
        else:
            raw_output = None
            render_target = a.output_path

        stages: list[dict] = []
        if self._mode == "pipeline":
            stages.append({
                "name": "pipeline",
                "args": [
                    "-m", "waruka", "pipeline",
                    "--project", proj, "--video", video,
                    *time_args,
                    "--chunk", "30",
                    "--out", render_target,
                ],
                "uses_progress": True,
                "label": "Pipeline (chunked track + render)",
            })
        else:
            tracks_json = str(p.artefact_dir / "tracks.json")
            players_json = str(p.artefact_dir / "players.json")
            campath_json = str(p.artefact_dir / "campath.json")
            n_main = 5 if interp_on else 4
            stages += [
                {
                    "name": "track",
                    "args": [
                        "-m", "waruka", "track",
                        "--project", proj, "--video", video,
                        *time_args,
                        "--stride", str(a.stride),
                        "--out", tracks_json,
                    ],
                    "uses_progress": True,
                    "label": f"Stage 1/{n_main}: Detect + track players",
                },
                {
                    "name": "classify",
                    "args": [
                        "-m", "waruka", "classify", tracks_json,
                        "--project", proj, "--out", players_json,
                    ],
                    "uses_progress": False,
                    "label": f"Stage 2/{n_main}: Classify on-field vs sideline",
                },
                {
                    "name": "campath",
                    "args": [
                        "-m", "waruka", "campath", players_json,
                        "--project", proj,
                        "--view-mode", a.view_mode,
                        "--out", campath_json,
                    ],
                    "uses_progress": False,
                    "label": f"Stage 3/{n_main}: Plan camera path",
                },
                {
                    "name": "render",
                    "args": [
                        "-m", "waruka", "render", campath_json,
                        "--project", proj, "--video", video,
                        "--out", render_target,
                        *(["--sr"] if a.sr_enabled else []),
                    ],
                    "uses_progress": True,
                    "label": f"Stage 4/{n_main}: Render broadcast video",
                },
            ]

        if interp_on:
            stages.append({
                "name": "interpolate",
                "args": [
                    "-m", "waruka", "interpolate", raw_output,
                    "--fps", str(int(a.interpolate_fps)),
                    "--backend", a.interpolate_backend,
                    "--out", a.output_path,
                ],
                "uses_progress": True,
                "label": (f"Stage 5/5: Frame-interpolate to "
                          f"{a.interpolate_fps} fps ({a.interpolate_backend.upper()})"),
            })

        # Audio post-process: only meaningful when the source has audio.
        if self._info.has_audio:
            stages.append({
                "name": "audio_mux",
                "args": None,   # runs via ffmpeg directly, not waruka CLI
                "uses_progress": False,
                "label": "Final: mux source audio into tracked video",
            })

        return stages

    # ----- run lifecycle ---------------------------------------------------

    def start(self) -> None:
        """Kick off the pipeline. Called by MainWindow after exec()."""
        if not self._stages:
            self._finish_done()
            return
        self._stage_idx = 0
        self._enter_stage()

    def _enter_stage(self) -> None:
        stage = self._stages[self._stage_idx]
        self.stage_label.setText(stage["label"])
        self.stage_detail.setText("")
        self.bar.setRange(0, 100)
        self.bar.setValue(0)
        self.kill_btn.setEnabled(True)
        self._state = stage["name"]
        self._append_log(f"\n=== {stage['label']} ===\n")

        if stage["name"] == "audio_mux":
            # Special-cased; not a waruka CLI subprocess but a direct
            # ffmpeg invocation. We still run it via StepRunner so the
            # finished/log signals work uniformly.
            self._run_audio_mux()
            return

        # Standard CLI stage
        if stage["uses_progress"]:
            # Wipe any stale progress file so we don't read the previous run.
            try:
                (self._paths.artefact_dir / "_progress.json").unlink(
                    missing_ok=True)
            except Exception:
                pass
            self._poll_timer.start()

        self._runner.start(stage["args"], cwd=self._paths.artefact_dir)

    def _on_stage_finished(self, exit_code: int) -> None:
        self._poll_timer.stop()
        stage = self._stages[self._stage_idx]
        if exit_code != 0:
            self._failure_log = self._runner.log()
            self._finish_failed(f"{stage['name']} exited with code {exit_code}")
            return
        # Stage success.
        self._stage_idx += 1
        if self._stage_idx >= len(self._stages):
            self._finish_done()
            return
        self._enter_stage()

    def _on_runner_line(self, ln: str) -> None:
        self._append_log(ln + "\n")

    def _poll_progress(self) -> None:
        """Read the artefact-dir _progress.json and update the bar/labels.

        Tolerates missing/half-written files (returns silently); doesn't
        replace the indeterminate state if the file isn't there yet.
        """
        import json
        path = self._paths.artefact_dir / "_progress.json"
        if not path.exists():
            return
        try:
            with open(path) as f:
                p = json.load(f)
        except Exception:
            return
        prog = p.get("step_progress")
        if prog is None:
            self.bar.setRange(0, 0)  # indeterminate
        else:
            self.bar.setRange(0, 100)
            self.bar.setValue(int(round(100 * float(prog))))
        detail = p.get("step_detail", "")
        step = p.get("step", "")
        if step:
            self.stage_detail.setText(f"{step}  {detail}"[:160])
        elapsed = p.get("elapsed_s")
        eta = p.get("eta_s")
        fps = p.get("fps_observed")
        self.elapsed_label.setText(f"elapsed {_fmt_hms(elapsed)}")
        self.eta_label.setText(f"eta {_fmt_hms(eta)}")
        self.fps_label.setText(
            f"fps {fps:.2f}" if isinstance(fps, (int, float)) else "")

    # ----- audio mux ------------------------------------------------------

    def _run_audio_mux(self) -> None:
        """Run ffmpeg to mux source audio into the tracked output.

        Layout:
            silent_intermediate = <output>.silent_tmp.mp4
            We move the current output (which is silent) to that path,
            then run:
                ffmpeg -i silent_intermediate -i source -map 0:v -map 1:a
                       -c copy -shortest <output>
            On success, delete silent_intermediate unless the user asked
            for a _no_audio.mp4 copy, in which case rename it to that.
        """
        try:
            import imageio_ffmpeg
            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception as e:
            self._finish_failed(f"ffmpeg not available for audio mux: {e}")
            return

        out_path = Path(self._params.output_path)
        silent_tmp = out_path.with_suffix(".silent_tmp.mp4")
        try:
            if silent_tmp.exists():
                silent_tmp.unlink()
            out_path.rename(silent_tmp)
        except Exception as e:
            self._finish_failed(f"could not stage silent intermediate: {e}")
            return
        # Stash these so _on_stage_finished can finalise after ffmpeg exits.
        self._mux_silent_tmp = silent_tmp
        self._mux_out_path = out_path

        args = [
            "-y",  # overwrite output without prompting
            "-i", str(silent_tmp),
            "-i", str(self._paths.source),
            "-map", "0:v", "-map", "1:a",
            "-c", "copy", "-shortest",
            str(out_path),
        ]
        # We need to run ffmpeg directly, not via PYTHON. Build a thin
        # variant by swapping the runner's program for this one call.
        proc = QtCore.QProcess(self)
        proc.setProgram(ffmpeg)
        proc.setArguments(args)
        proc.setProcessChannelMode(QtCore.QProcess.MergedChannels)
        proc.readyReadStandardOutput.connect(
            lambda: self._on_runner_line(
                bytes(proc.readAllStandardOutput()).decode(
                    "utf-8", errors="replace").rstrip()))
        proc.finished.connect(lambda code, _st: self._on_audio_finished(code))
        self.bar.setRange(0, 0)
        self.stage_detail.setText("ffmpeg muxing audio...")
        self._ffmpeg_proc = proc
        proc.start()

    def _on_audio_finished(self, exit_code: int) -> None:
        proc = getattr(self, "_ffmpeg_proc", None)
        if proc is not None:
            proc.deleteLater()
            self._ffmpeg_proc = None
        silent_tmp = getattr(self, "_mux_silent_tmp", None)
        out_path = getattr(self, "_mux_out_path", None)
        if exit_code != 0:
            # Restore the silent intermediate as the main output so the
            # user at least has the silent version.
            if silent_tmp is not None and out_path is not None:
                try:
                    if out_path.exists():
                        out_path.unlink()
                    silent_tmp.rename(out_path)
                except Exception:
                    pass
            self._finish_failed(f"ffmpeg audio mux failed (exit {exit_code})")
            return
        # Mux succeeded. Either keep the silent intermediate as a
        # no-audio companion, or delete it.
        if self._params.create_no_audio_copy and silent_tmp is not None and out_path is not None:
            no_audio_path = out_path.with_name(
                out_path.stem + "_no_audio" + out_path.suffix)
            try:
                if no_audio_path.exists():
                    no_audio_path.unlink()
                silent_tmp.rename(no_audio_path)
                self._final_outputs.append(no_audio_path)
            except Exception:
                # Non-fatal; just keep the silent_tmp around.
                pass
        else:
            try:
                if silent_tmp is not None and silent_tmp.exists():
                    silent_tmp.unlink()
            except Exception:
                pass
        self._finish_done()

    # ----- finish helpers --------------------------------------------------

    def _finish_done(self) -> None:
        self._state = self.STATE_DONE
        self.stage_label.setText("Done.")
        self.stage_detail.setText("")
        self.bar.setRange(0, 100)
        self.bar.setValue(100)
        self.kill_btn.setEnabled(False)
        self.close_btn.setEnabled(True)
        self.open_folder_btn.setVisible(True)
        # Always include the main output; the no-audio copy was appended
        # during _on_audio_finished if applicable.
        main_out = Path(self._params.output_path)
        if main_out.exists() and main_out not in self._final_outputs:
            self._final_outputs.insert(0, main_out)
        self._show_outputs()

    def _finish_failed(self, msg: str) -> None:
        self._state = self.STATE_FAILED
        self.stage_label.setText(f"Failed: {msg}")
        self.bar.setRange(0, 100)
        self.bar.setValue(0)
        self.kill_btn.setEnabled(False)
        self.close_btn.setEnabled(True)
        # Force the log view open so the user can see what happened.
        self.show_log_btn.setChecked(True)
        if self._failure_log:
            self._append_log(
                "\n--- captured output from failing stage ---\n"
                + self._failure_log[-4000:])

    def _show_outputs(self) -> None:
        self.output_list.clear()
        for p in self._final_outputs:
            self.output_list.addItem(str(p))
        self.output_list.setVisible(bool(self._final_outputs))

    def _open_output_folder(self) -> None:
        # Use Qt's url-based opener so we don't have to shell out.
        if not self._final_outputs:
            return
        folder = self._final_outputs[0].parent
        QtGui.QDesktopServices.openUrl(
            QtCore.QUrl.fromLocalFile(str(folder)))

    # ----- log + kill -----------------------------------------------------

    def _toggle_log(self, on: bool) -> None:
        self.log_view.setVisible(on)
        self.show_log_btn.setText("Hide details" if on else "Show details")

    def _append_log(self, text: str) -> None:
        self.log_view.appendPlainText(text.rstrip())

    def _on_kill(self) -> None:
        # Confirm before sending SIGTERM/kill.
        if QtWidgets.QMessageBox.question(
            self, "Kill run",
            "Stop the currently-running stage?\n"
            "Partial outputs may be left on disk.",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No,
        ) != QtWidgets.QMessageBox.Yes:
            return
        # Kill the waruka runner if active.
        if self._runner.is_running():
            try:
                # Pull the worker PID from the progress file if available
                # so we kill the right process tree.
                import json, os, signal
                pj = self._paths.artefact_dir / "_progress.json"
                if pj.exists():
                    pid = json.loads(pj.read_text()).get("pid")
                    if pid is not None:
                        try:
                            os.kill(int(pid), signal.SIGTERM)
                        except Exception:
                            pass
            except Exception:
                pass
            # Also fall back to killing the QProcess itself.
            if self._runner._proc is not None:
                self._runner._proc.kill()
        # Kill the ffmpeg proc if in audio_mux.
        proc = getattr(self, "_ffmpeg_proc", None)
        if proc is not None:
            proc.kill()
        self._finish_failed("killed by user")

    def closeEvent(self, ev: QtGui.QCloseEvent) -> None:
        # Don't allow closing while a stage is running -- forces the
        # user to either kill explicitly or wait for completion.
        if self._state not in (self.STATE_IDLE, self.STATE_DONE,
                                self.STATE_FAILED):
            ev.ignore()
            QtWidgets.QMessageBox.information(
                self, "Still running",
                "A stage is currently running. Kill it first if you want "
                "to abort, or wait for it to finish.")
            return
        super().closeEvent(ev)


# --------------------------------------------------------------------------
# Track tab -- owns the calibrate -> markfield -> params -> process flow
# --------------------------------------------------------------------------

class TrackTab(QtWidgets.QWidget):
    """The original end-to-end tracking flow, now living inside a tab.

    Owns the video picker + the 4-step pipeline card list + all step
    launcher methods. Emits ``status_message`` instead of touching the
    main window's status bar directly so the parent QTabWidget can
    route it however it likes.
    """

    # Emitted whenever the tab wants to surface a status-bar message.
    # The int is a timeout in milliseconds (0 = sticky).
    status_message = QtCore.Signal(str, int)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)

        self._runner = StepRunner(self)
        self._current_step: str | None = None
        self._runner.finished.connect(self._on_step_finished)

        # Tracking params -- populated when the user opens the form
        # dialog and clicks OK. None until then; the process step
        # refuses to launch without one.
        self._params: ProcessingParams | None = None

        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(16, 16, 16, 16)
        outer.setSpacing(12)

        # --- Video picker --------------------------------------------------
        self.picker = VideoPickerWidget()
        self.picker.video_selected.connect(self._on_video_selected)
        outer.addWidget(self.picker)

        # --- Step list -----------------------------------------------------
        step_box = QtWidgets.QGroupBox("Pipeline steps")
        step_layout = QtWidgets.QVBoxLayout(step_box)
        step_layout.setSpacing(6)

        self.step_cards: dict[str, StepCardWidget] = {
            "calibrate": StepCardWidget(
                1, "Calibrate dewarp", "Run calibrate",
                self._run_calibrate,
            ),
            "markfield": StepCardWidget(
                2, "Mark field", "Run markfield",
                self._run_markfield,
            ),
            "params": StepCardWidget(
                3, "Tracking parameters", "Open form",
                self._open_params_dialog,
            ),
            "process": StepCardWidget(
                4, "Process (track + render)", "Run pipeline",
                self._run_process,
            ),
        }
        for card in self.step_cards.values():
            step_layout.addWidget(card)
        step_layout.addStretch(1)
        outer.addWidget(step_box, 1)

    # ----- public API for cross-tab handover ------------------------------

    def load_video(self, path: str | Path) -> None:
        """Load a video into the picker. Called by ConcatTab after a
        concat+trim job lands a final output."""
        self.picker.set_video(path)

    # ----- video + step status --------------------------------------------

    def _on_video_selected(self, paths: WarukaPaths, info: VideoInfo) -> None:
        self.status_message.emit(f"Loaded {paths.source.name}", 0)
        self._refresh_step_status()

    def _refresh_step_status(self) -> None:
        """Re-read on-disk state and update the step cards."""
        paths = self.picker.paths
        if paths is None:
            return
        status = detect_step_status(paths)
        for name, card in self.step_cards.items():
            card.set_status(status.get(name, STEP_PENDING))

    # ----- Step launchers -------------------------------------------------

    def _run_calibrate(self) -> None:
        self._launch_step(
            "calibrate",
            ["-m", "waruka", "calibrate"],
            extra_args=[],
        )

    def _run_markfield(self) -> None:
        self._launch_step(
            "markfield",
            ["-m", "waruka", "markfield"],
            extra_args=[],
        )

    def _run_process(self) -> None:
        """Launch the ProcessDialog. Requires video + markfield + params."""
        paths = self.picker.paths
        info = self.picker.info
        if paths is None or info is None:
            QtWidgets.QMessageBox.information(
                self, "No video loaded",
                "Open a video first using the picker above.")
            return
        if self._params is None:
            QtWidgets.QMessageBox.information(
                self, "Set parameters first",
                "Open the tracking-parameters form (step 3) before running.")
            return
        st = detect_step_status(paths)
        if st.get("markfield") != STEP_DONE:
            QtWidgets.QMessageBox.warning(
                self, "Mark field first",
                "Run the markfield step before processing.")
            return

        dlg = ProcessDialog(paths, info, self._params, parent=self)
        dlg.show()
        QtCore.QTimer.singleShot(0, dlg.start)
        dlg.exec()
        self._refresh_step_status()

    def _open_params_dialog(self) -> None:
        paths = self.picker.paths
        info = self.picker.info
        if paths is None or info is None:
            QtWidgets.QMessageBox.information(
                self, "No video loaded",
                "Open a video first using the picker above.")
            return
        initial = self._params or ProcessingParams()
        if not initial.output_path:
            initial.output_path = str(paths.output_video)
        dlg = ParamsDialog(paths, info, initial, parent=self)
        if dlg.exec() == QtWidgets.QDialog.Accepted and dlg.result_params:
            self._params = dlg.result_params
            mode_resolved = self._params.effective_mode(info.duration_s)
            t0_str = (f"{self._params.t0:.1f}"
                       if self._params.t0 is not None else "start")
            t1_str = (f"{self._params.t1:.1f}"
                       if self._params.t1 is not None else "end")
            self.status_message.emit(
                f"Params set: t0={t0_str} t1={t1_str} mode={mode_resolved}",
                5000)
            self.step_cards["params"].set_status(STEP_DONE)

    def _launch_step(
        self,
        step: str,
        cmd_args: list[str],
        extra_args: list[str],
    ) -> None:
        paths = self.picker.paths
        if paths is None:
            QtWidgets.QMessageBox.information(
                self, "No video loaded",
                "Open a video first using the picker above.")
            return
        if self._runner.is_running():
            QtWidgets.QMessageBox.information(
                self, "Step in progress",
                f"Currently running '{self._current_step}'. "
                "Wait for it to finish first.")
            return

        args = (
            list(cmd_args)
            + [str(paths.source), "--project", str(paths.project_json)]
            + list(extra_args)
        )

        self._current_step = step
        card = self.step_cards.get(step)
        if card is not None:
            card.set_status(STEP_RUNNING)
        self.status_message.emit(
            f"Running {step}...  (interactive window will open separately)",
            0)
        self._runner.start(args, cwd=paths.artefact_dir)

    def _on_step_finished(self, exit_code: int) -> None:
        step = self._current_step or "?"
        self._current_step = None

        if exit_code != 0:
            log = self._runner.log() or "(no output captured)"
            QtWidgets.QMessageBox.warning(
                self,
                f"{step} failed (exit {exit_code})",
                f"Subprocess exited with code {exit_code}.\n\n"
                f"Last output:\n\n{log[-2000:]}",
            )
            self.status_message.emit(
                f"{step} failed (exit {exit_code})", 5000)
        else:
            self.status_message.emit(f"{step} finished", 3000)

        self._refresh_step_status()


# --------------------------------------------------------------------------
# Scrubber widget -- video player + in/out marker capture for trim
# --------------------------------------------------------------------------

class ScrubberWidget(QtWidgets.QWidget):
    """Video scrubber with IN/OUT capture for trim selection.

    Backend: ``cv2.VideoCapture`` for frame access (same approach used
    by calibrate / markfield / render) + ``QLabel`` for display + a
    ``QTimer`` for playback. We tried QtMultimedia's ``QMediaPlayer``
    first; on Windows-Store-Python + PySide6 it returns ResourceError
    on every file we throw at it (H.264 included), so we fell back
    to cv2 which is rock-solid and handles HEVC natively via its
    bundled FFmpeg. The trade-off: no audio playback during preview.
    Acceptable here because the user is picking visual cues (match
    start / end), not listening for them.

    Public API (matches the original QMediaPlayer-backed version so
    callers in ConcatTab + the smoketests don't need to change):
      * ``load_video(path)``
      * ``in_seconds()`` / ``out_seconds()``  -> float | None
      * ``duration_seconds()`` / ``trim_duration_seconds()`` -> float
      * ``stop()``
      * ``in_out_changed`` signal

    For backward-compat with the original tests that hand-poked
    ``_player.position()`` / ``_player.setPosition()``, we still
    expose an internal player-like object via the same name -- but
    it's just a thin wrapper over our position state.
    """

    in_out_changed = QtCore.Signal()

    # Step sizes for the seek buttons, in milliseconds.
    STEP_SMALL_MS = 1000
    STEP_LARGE_MS = 10_000

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._in_ms: int | None = None
        self._out_ms: int | None = None
        self._duration_ms: int = 0
        self._position_ms: int = 0
        self._fps: float = 25.0          # filled at load_video
        self._n_frames: int = 0
        # cv2.VideoCapture handle. We lazy-import cv2 below so the
        # module stays importable without it.
        self._cap = None                  # type: ignore
        # Playback timer: fires every (1000/fps) ms, advances + reads
        # the next frame, mirrors the position back to the slider.
        self._play_timer = QtCore.QTimer(self)
        self._play_timer.timeout.connect(self._on_play_tick)
        # Used to suppress recursive valueChanged emissions when WE
        # update the slider after a programmatic seek.
        self._slider_internal_update = False
        # Cached qimage of the most recent frame, used so rescaling
        # on resize doesn't have to re-decode.
        self._last_qimage: QtGui.QImage | None = None

        # Display label -- replaces the original QVideoWidget.
        self._display = QtWidgets.QLabel(self)
        self._display.setAlignment(QtCore.Qt.AlignCenter)
        self._display.setMinimumSize(640, 280)
        self._display.setStyleSheet(
            "background: #000; color: #888;")
        self._display.setText("(no video loaded)")
        self._display.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Expanding)

        # --- Layout -------------------------------------------------------
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(6)
        outer.addWidget(self._display, 1)

        # Transport row
        transport = QtWidgets.QHBoxLayout()
        self._back_large_btn = QtWidgets.QPushButton("⏮ 10s")
        self._back_large_btn.clicked.connect(
            lambda: self._seek_relative(-self.STEP_LARGE_MS))
        self._back_small_btn = QtWidgets.QPushButton("◀ 1s")
        self._back_small_btn.clicked.connect(
            lambda: self._seek_relative(-self.STEP_SMALL_MS))
        self._play_btn = QtWidgets.QPushButton("▶ Play")
        self._play_btn.clicked.connect(self._toggle_play)
        self._fwd_small_btn = QtWidgets.QPushButton("1s ▶")
        self._fwd_small_btn.clicked.connect(
            lambda: self._seek_relative(+self.STEP_SMALL_MS))
        self._fwd_large_btn = QtWidgets.QPushButton("10s ⏭")
        self._fwd_large_btn.clicked.connect(
            lambda: self._seek_relative(+self.STEP_LARGE_MS))
        for b in (self._back_large_btn, self._back_small_btn,
                   self._play_btn, self._fwd_small_btn, self._fwd_large_btn):
            transport.addWidget(b)
        transport.addStretch(1)
        self._time_label = QtWidgets.QLabel("00:00:00 / 00:00:00")
        self._time_label.setStyleSheet(
            "font-family: Consolas, monospace; color: #444;")
        transport.addWidget(self._time_label)
        outer.addLayout(transport)

        # Timeline slider (range = 0..duration_ms; we use ms directly)
        self._slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self._slider.setRange(0, 0)
        self._slider.sliderMoved.connect(self._on_slider_moved)
        # sliderMoved fires only while the user drags; clicks on the
        # track fire valueChanged with a synthetic value. Handle both.
        self._slider.valueChanged.connect(self._on_slider_value_changed)
        outer.addWidget(self._slider)

        # IN/OUT row
        in_out_row = QtWidgets.QHBoxLayout()
        self._in_label = QtWidgets.QLabel("IN:  start of clip")
        self._in_label.setStyleSheet(
            "font-family: Consolas, monospace; color: #444;")
        self._set_in_btn = QtWidgets.QPushButton("Set IN here")
        self._set_in_btn.clicked.connect(self._set_in_at_current)
        self._out_label = QtWidgets.QLabel("OUT: end of clip")
        self._out_label.setStyleSheet(
            "font-family: Consolas, monospace; color: #444;")
        self._set_out_btn = QtWidgets.QPushButton("Set OUT here")
        self._set_out_btn.clicked.connect(self._set_out_at_current)
        self._clear_btn = QtWidgets.QPushButton("Clear in/out")
        self._clear_btn.clicked.connect(self._clear_in_out)
        for w in (self._in_label, self._set_in_btn):
            in_out_row.addWidget(w)
        in_out_row.addSpacing(20)
        for w in (self._out_label, self._set_out_btn):
            in_out_row.addWidget(w)
        in_out_row.addStretch(1)
        in_out_row.addWidget(self._clear_btn)
        outer.addLayout(in_out_row)

        # Player shim (for backward-compat with smoketests that poked
        # _player.position() / _player.setSource() / _player.stop()).
        self._player = self._PlayerShim(self)

        self._refresh_in_out_labels()

    # ----- backward-compat shim -------------------------------------------

    class _PlayerShim:
        """Stand-in for the old QMediaPlayer attribute. Forwards
        position/seek/stop to the cv2-backed parent so existing
        smoketests + in-process callers keep working unchanged."""
        def __init__(self, owner: "ScrubberWidget") -> None:
            self._owner = owner

        def position(self) -> int:
            return self._owner._position_ms

        def setPosition(self, p: int) -> None:
            self._owner._seek_to(int(p))

        def stop(self) -> None:
            self._owner._play_timer.stop()
            self._owner._play_btn.setText("▶ Play")

        def setSource(self, *_args, **_kwargs) -> None:
            # Original code called this with QUrl() to release the
            # file handle. The cv2 backend has its own release path
            # (load_video resetting / explicit close()). No-op here
            # is fine because the caller-side cleanup logic in
            # ConcatTab calls .stop() and overwrites the path.
            self._owner._close_capture()

        def play(self) -> None:
            self._owner._start_playing()

        def pause(self) -> None:
            self._owner._play_timer.stop()
            self._owner._play_btn.setText("▶ Play")

        def playbackState(self) -> str:
            # Returns a string to avoid importing QMediaPlayer's enum.
            return ("PlayingState" if self._owner._play_timer.isActive()
                    else "StoppedState")

    # ----- public API ------------------------------------------------------

    def load_video(self, path: str | Path) -> None:
        """Open ``path`` via cv2.VideoCapture, populate duration / fps
        from the container, reset in/out + position, render the first
        frame."""
        import cv2
        self._close_capture()
        self._in_ms = None
        self._out_ms = None
        self._position_ms = 0
        self._last_qimage = None

        self._cap = cv2.VideoCapture(str(path))
        if not self._cap.isOpened():
            QtWidgets.QMessageBox.warning(
                self, "Couldn't open video",
                f"cv2 failed to open: {path}")
            self._cap = None
            return

        self._fps = float(self._cap.get(cv2.CAP_PROP_FPS)) or 25.0
        self._n_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._duration_ms = int(round(1000 * self._n_frames / self._fps))
        self._slider.setRange(0, self._duration_ms)
        self._slider.setValue(0)
        self._update_time_label()
        self._refresh_in_out_labels()
        # Render the first frame so the user sees something.
        self._render_current_frame()
        self.in_out_changed.emit()

    def in_seconds(self) -> float | None:
        return None if self._in_ms is None else self._in_ms / 1000.0

    def out_seconds(self) -> float | None:
        return None if self._out_ms is None else self._out_ms / 1000.0

    def duration_seconds(self) -> float:
        return self._duration_ms / 1000.0

    def trim_duration_seconds(self) -> float:
        in_s = self.in_seconds() or 0.0
        out_s = (self.out_seconds()
                  if self.out_seconds() is not None
                  else self.duration_seconds())
        return max(0.0, out_s - in_s)

    def stop(self) -> None:
        """Stop playback. Called by ConcatTab during cleanup."""
        self._play_timer.stop()
        self._play_btn.setText("▶ Play")

    # ----- internals -------------------------------------------------------

    def _close_capture(self) -> None:
        """Release the cv2 handle so the underlying file can be
        unlinked / overwritten."""
        self._play_timer.stop()
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None

    def _toggle_play(self) -> None:
        if self._play_timer.isActive():
            self._play_timer.stop()
            self._play_btn.setText("▶ Play")
        else:
            self._start_playing()

    def _start_playing(self) -> None:
        if self._cap is None or self._fps <= 0:
            return
        # If we're at the end, restart from the beginning.
        if self._position_ms >= self._duration_ms - int(1000 / self._fps):
            self._seek_to(0)
        self._play_timer.start(int(1000 / self._fps))
        self._play_btn.setText("⏸ Pause")

    def _on_play_tick(self) -> None:
        """Advance by one frame; stop at the end of the clip."""
        if self._cap is None:
            self._play_timer.stop()
            return
        ok, frame = self._cap.read()
        if not ok or frame is None:
            self._play_timer.stop()
            self._play_btn.setText("▶ Play")
            return
        # cv2's POS_FRAMES is the index of the NEXT frame; the one we
        # just read is at index (POS_FRAMES - 1). Trust the time it
        # tells us.
        import cv2
        pos_ms = int(self._cap.get(cv2.CAP_PROP_POS_MSEC))
        if pos_ms <= 0:
            pos_ms = self._position_ms + int(1000 / self._fps)
        self._position_ms = min(pos_ms, self._duration_ms)
        self._display_frame(frame)
        self._mirror_position_to_slider()
        self._update_time_label()
        if self._position_ms >= self._duration_ms - 10:
            self._play_timer.stop()
            self._play_btn.setText("▶ Play")

    def _seek_relative(self, delta_ms: int) -> None:
        target = max(0, min(self._duration_ms,
                              self._position_ms + delta_ms))
        self._seek_to(target)

    def _seek_to(self, target_ms: int) -> None:
        """Seek the cv2 capture to ``target_ms`` (clamped) and render
        the frame at that position."""
        if self._cap is None or self._duration_ms <= 0:
            self._position_ms = target_ms
            self._mirror_position_to_slider()
            self._update_time_label()
            return
        import cv2
        target_ms = max(0, min(self._duration_ms, target_ms))
        # CAP_PROP_POS_MSEC seeks to the nearest keyframe before the
        # target in many cv2 backends; for HEVC that's typically within
        # ~2s. Acceptable for in/out marker placement.
        self._cap.set(cv2.CAP_PROP_POS_MSEC, float(target_ms))
        self._position_ms = target_ms
        self._render_current_frame()
        self._mirror_position_to_slider()
        self._update_time_label()

    def _render_current_frame(self) -> None:
        """Read + display the frame at the current capture position."""
        if self._cap is None:
            return
        ok, frame = self._cap.read()
        if not ok or frame is None:
            return
        self._display_frame(frame)
        # cv2.read() advances the position by one frame; for our seek
        # logic to be idempotent we don't bother stepping it back --
        # the next play tick / seek will resync.

    def _display_frame(self, frame) -> None:
        """Convert a cv2 BGR frame to QPixmap + paint on the display."""
        import cv2
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        # bytesPerLine matters for non-contiguous strides; cv2 frames
        # are contiguous so width*3 is correct.
        qimg = QtGui.QImage(rgb.data, w, h, w * ch,
                            QtGui.QImage.Format_RGB888).copy()
        # Copy() because the numpy array's memory may be GC'd; we want
        # QImage to own its own buffer.
        self._last_qimage = qimg
        self._paint_last_qimage()

    def _paint_last_qimage(self) -> None:
        if self._last_qimage is None:
            return
        pix = QtGui.QPixmap.fromImage(self._last_qimage)
        # Scale to the label's current size, keep aspect ratio.
        pix = pix.scaled(
            self._display.size(),
            QtCore.Qt.KeepAspectRatio,
            QtCore.Qt.SmoothTransformation)
        self._display.setPixmap(pix)

    def resizeEvent(self, ev: QtGui.QResizeEvent) -> None:
        super().resizeEvent(ev)
        self._paint_last_qimage()

    def _on_slider_moved(self, value: int) -> None:
        """User dragged the slider thumb."""
        self._seek_to(int(value))

    def _on_slider_value_changed(self, value: int) -> None:
        """Catches clicks on the track (which fire valueChanged but
        NOT sliderMoved). Guarded by _slider_internal_update so our
        own programmatic setValue calls don't re-trigger seek."""
        if self._slider_internal_update:
            return
        # Only seek if the value differs meaningfully from where we
        # already are -- avoids feedback loops on programmatic seek.
        if abs(value - self._position_ms) > 50:
            self._seek_to(int(value))

    def _mirror_position_to_slider(self) -> None:
        self._slider_internal_update = True
        try:
            self._slider.setValue(self._position_ms)
        finally:
            self._slider_internal_update = False

    def _update_time_label(self) -> None:
        pos = self._position_ms / 1000.0
        dur = self._duration_ms / 1000.0
        self._time_label.setText(
            f"{_fmt_hms(pos)} / {_fmt_hms(dur)}")

    # ----- in/out capture -------------------------------------------------

    def _set_in_at_current(self) -> None:
        pos = self._position_ms
        if self._out_ms is not None and pos >= self._out_ms:
            QtWidgets.QApplication.beep()
            return
        self._in_ms = pos if pos > 0 else None
        self._refresh_in_out_labels()
        self.in_out_changed.emit()

    def _set_out_at_current(self) -> None:
        pos = self._position_ms
        if self._in_ms is not None and pos <= self._in_ms:
            QtWidgets.QApplication.beep()
            return
        # If user set OUT at the very end (within 100ms), treat it as
        # "use natural end" -- avoids the off-by-one container-frame
        # issue the tracking dialog had with t0/t1.
        if (self._duration_ms - pos) <= 100:
            self._out_ms = None
        else:
            self._out_ms = pos
        self._refresh_in_out_labels()
        self.in_out_changed.emit()

    def _clear_in_out(self) -> None:
        self._in_ms = None
        self._out_ms = None
        self._refresh_in_out_labels()
        self.in_out_changed.emit()

    def _refresh_in_out_labels(self) -> None:
        if self._in_ms is None:
            in_text = "IN:  start of clip"
        else:
            in_text = f"IN:  {_fmt_hms(self._in_ms / 1000.0)}"
        if self._out_ms is None:
            if self._duration_ms > 0:
                out_text = f"OUT: end of clip ({_fmt_hms(self.duration_seconds())})"
            else:
                out_text = "OUT: end of clip"
        else:
            out_text = f"OUT: {_fmt_hms(self._out_ms / 1000.0)}"
        self._in_label.setText(in_text)
        self._out_label.setText(out_text)


# --------------------------------------------------------------------------
# Concat tab -- multi-clip concat + trim, hands off to Track tab on done
# --------------------------------------------------------------------------

@dataclass
class ClipEntry:
    """One row in the concat clip list. Carries the source path plus
    the metadata probed at add time so the row display + later codec-
    mismatch checks don't have to re-probe."""
    path: Path
    info: VideoInfo
    # Per-clip video codec. Used by the codec-consistency check to
    # decide between lossless ``-c copy`` concat and re-encode.
    video_codec: str = ""
    # Best-effort recording-start datetime, formatted "YYYY-MM-DD HH:MM:SS".
    # Empty string if no source had it. Populated by _extract_clip_datetime
    # at add time. Shown in the QListWidget because Reolink filenames are
    # cryptic.
    created_at: str = ""


def _extract_clip_datetime(path: Path) -> str:
    """Best-effort recording-start datetime as 'YYYY-MM-DD HH:MM:SS'.

    Tries three sources in order:
      1. Reolink-style filename pattern  ``..._DST<YYYYMMDD>_<HHMMSS>_...``
         (records the *start* of the chunk; what the user actually wants).
      2. ffmpeg ``creation_time`` metadata tag (records the *write*
         time, which for some encoders is the end of the chunk).
      3. Filesystem mtime fallback.

    Returns '' if all three fail.
    """
    import re
    import datetime as _dt

    # 1. Reolink filename pattern.
    m = re.search(r"_DST(\d{8})_(\d{6})_", path.name)
    if m:
        d, t = m.group(1), m.group(2)
        return f"{d[:4]}-{d[4:6]}-{d[6:8]} {t[:2]}:{t[2:4]}:{t[4:6]}"

    # 2. ffmpeg creation_time metadata.
    try:
        import subprocess
        import imageio_ffmpeg
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        out = subprocess.run(
            [ffmpeg, "-hide_banner", "-i", str(path)],
            capture_output=True, text=True, timeout=10,
            **_NO_WINDOW_KW,
        )
        text = (out.stderr or "") + (out.stdout or "")
        for line in text.splitlines():
            if "creation_time" in line.lower():
                # Format: "    creation_time   : 2026-05-17T08:50:29.000000Z"
                m2 = re.search(
                    r"(\d{4}-\d{2}-\d{2})T(\d{2}:\d{2}:\d{2})", line)
                if m2:
                    return f"{m2.group(1)} {m2.group(2)}"
    except Exception:
        pass

    # 3. mtime fallback.
    try:
        ts = _dt.datetime.fromtimestamp(path.stat().st_mtime)
        return ts.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return ""


def _probe_video_codec(path: str | Path) -> str:
    """Return the video stream codec name (e.g. 'h264', 'hevc') or '' on failure.

    Same approach as ``_probe_audio``: scrape ffmpeg's -i stderr for
    the ``Stream #N:M ... Video: <codec>`` line and pull the codec
    token. Cheap enough to do at file-add time."""
    import subprocess
    try:
        import imageio_ffmpeg
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return ""
    try:
        out = subprocess.run(
            [ffmpeg, "-hide_banner", "-i", str(path)],
            capture_output=True, text=True, timeout=10,
            **_NO_WINDOW_KW,
        )
    except Exception:
        return ""
    text = (out.stderr or "") + (out.stdout or "")
    for line in text.splitlines():
        if "Stream #" in line and "Video:" in line:
            # Format: "Stream #0:0[0x1](und): Video: h264 (High) ...
            #   ^^^^^^^                       ^^^^^^ ^^^^
            #   prefix                        marker codec
            try:
                after = line.split("Video:", 1)[1].strip()
                # token before the next space or '('
                tok = after.split(" ", 1)[0].split("(", 1)[0]
                return tok.strip().lower()
            except Exception:
                pass
    return ""


class ClipListWidget(QtWidgets.QListWidget):
    """QListWidget that accepts external file drag-drop.

    Emits ``files_dropped(list[str])`` when the user drops one or
    more files onto the widget. Internal reorder via drag is enabled
    so the user can shuffle clip order; the explicit up/down buttons
    in ConcatTab cover the keyboard-friendly path."""

    files_dropped = QtCore.Signal(list)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setSelectionMode(self.SelectionMode.ExtendedSelection)
        self.setDragDropMode(self.DragDropMode.DragDrop)
        self.setDefaultDropAction(QtCore.Qt.MoveAction)

    # Drag-drop wiring ----------------------------------------------------

    def dragEnterEvent(self, ev: QtGui.QDragEnterEvent) -> None:
        # External drops carry URLs; internal moves don't.
        if ev.mimeData().hasUrls():
            ev.acceptProposedAction()
            return
        super().dragEnterEvent(ev)

    def dragMoveEvent(self, ev: QtGui.QDragMoveEvent) -> None:
        if ev.mimeData().hasUrls():
            ev.acceptProposedAction()
            return
        super().dragMoveEvent(ev)

    def dropEvent(self, ev: QtGui.QDropEvent) -> None:
        md = ev.mimeData()
        if md.hasUrls():
            paths = [u.toLocalFile() for u in md.urls() if u.isLocalFile()]
            if paths:
                self.files_dropped.emit(paths)
                ev.acceptProposedAction()
                return
        # Internal reorder -- let QListWidget handle it
        super().dropEvent(ev)


class PostProcessTab(QtWidgets.QWidget):
    """Standalone post-processing tab: run frame interpolation or 2x
    Real-ESRGAN upscale on any existing video file.

    Independent from the Track tab's project + step pipeline. Picks an
    input file, picks an operation (interpolate / upscale), runs the
    corresponding ``waruka`` CLI subcommand as a subprocess, surfaces
    progress in a log pane.
    """

    status_message = QtCore.Signal(str, int)

    VIDEO_EXTS = (".mp4", ".mov", ".mkv", ".avi", ".m4v")

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._runner = StepRunner(self)
        self._runner.line.connect(self._on_log_line)
        self._runner.finished.connect(self._on_finished)

        # Stage chaining state. Each entry is (label, args, output_path).
        # When the user picks both operations, this holds [upscale, interp]
        # and we advance after each successful exit. Temp files created
        # for intermediate stages are tracked so they can be deleted at
        # the end (or on failure).
        self._stages: list[tuple[str, list[str], str]] = []
        self._stage_total = 0
        self._stage_index = 0
        self._temp_files: list[str] = []

        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(16, 16, 16, 16)
        outer.setSpacing(10)

        hint = QtWidgets.QLabel(
            "Run frame interpolation, 2x super-resolution, or both on any "
            "existing video file. When both are checked, the source is "
            "upscaled first, then the upscaled output is interpolated -- "
            "faster than the reverse order. Output is written to the "
            "chosen path without touching the source.")
        hint.setStyleSheet("color: #555;")
        hint.setWordWrap(True)
        outer.addWidget(hint)

        # --- File pickers ---------------------------------------------
        form = QtWidgets.QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setLabelAlignment(QtCore.Qt.AlignRight)

        in_row = QtWidgets.QHBoxLayout()
        self.input_edit = QtWidgets.QLineEdit()
        self.input_edit.setPlaceholderText("path to source video")
        in_row.addWidget(self.input_edit, 1)
        in_browse = QtWidgets.QPushButton("Browse...")
        in_browse.clicked.connect(self._browse_input)
        in_row.addWidget(in_browse)
        in_w = QtWidgets.QWidget(); in_w.setLayout(in_row)
        form.addRow("Input video:", in_w)

        out_row = QtWidgets.QHBoxLayout()
        self.output_edit = QtWidgets.QLineEdit()
        self.output_edit.setPlaceholderText(
            "auto-suggested from input + operation")
        out_row.addWidget(self.output_edit, 1)
        out_browse = QtWidgets.QPushButton("Browse...")
        out_browse.clicked.connect(self._browse_output)
        out_row.addWidget(out_browse)
        out_w = QtWidgets.QWidget(); out_w.setLayout(out_row)
        form.addRow("Output:", out_w)
        outer.addLayout(form)

        # --- Operation selector ---------------------------------------
        # Checkboxes (not radio) -- the two operations are independent
        # and can be combined. If both are checked, we run upscale first
        # then interpolate the upscaled file (faster ordering: SR runs N
        # times instead of 3N).
        op_box = QtWidgets.QGroupBox("Operations")
        op_layout = QtWidgets.QVBoxLayout(op_box)
        self.do_interp = QtWidgets.QCheckBox(
            "Interpolate frames (RIFE / FILM)")
        self.do_interp.setChecked(True)
        self.do_upscale = QtWidgets.QCheckBox(
            "Upscale 2x (Real-ESRGAN) -- ~10 s/frame at 1440p input, "
            "use short clips")
        op_layout.addWidget(self.do_interp)
        op_layout.addWidget(self.do_upscale)
        self.do_interp.toggled.connect(self._update_visibility)
        self.do_upscale.toggled.connect(self._update_visibility)
        outer.addWidget(op_box)

        # --- Interpolate params ---------------------------------------
        self.interp_box = QtWidgets.QGroupBox("Interpolate parameters")
        interp_layout = QtWidgets.QFormLayout(self.interp_box)
        self.interp_fps_combo = QtWidgets.QComboBox()
        self.interp_fps_combo.addItem("40 fps (2x)", 40)
        self.interp_fps_combo.addItem("60 fps (3x, recommended)", 60)
        self.interp_fps_combo.addItem("80 fps (4x)", 80)
        self.interp_fps_combo.setCurrentIndex(1)
        interp_layout.addRow("Target fps:", self.interp_fps_combo)
        self.interp_backend_combo = QtWidgets.QComboBox()
        self.interp_backend_combo.addItem("RIFE 4.25 (fast)", "rife")
        self.interp_backend_combo.addItem("FILM-Style (very slow!)", "film")
        interp_layout.addRow("Backend:", self.interp_backend_combo)
        self.interp_cq_spin = QtWidgets.QSpinBox()
        self.interp_cq_spin.setRange(15, 35); self.interp_cq_spin.setValue(23)
        interp_layout.addRow("NVENC CQ (lower=sharper):",
                              self.interp_cq_spin)
        outer.addWidget(self.interp_box)

        # --- Upscale params -------------------------------------------
        self.upscale_box = QtWidgets.QGroupBox("Upscale parameters")
        up_layout = QtWidgets.QFormLayout(self.upscale_box)
        up_layout.addRow("Scale:", QtWidgets.QLabel("2x (fixed, "
                                                       "Real-ESRGAN x2plus)"))
        self.up_cq_spin = QtWidgets.QSpinBox()
        self.up_cq_spin.setRange(15, 35); self.up_cq_spin.setValue(23)
        up_layout.addRow("NVENC CQ (lower=sharper):", self.up_cq_spin)
        warn = QtWidgets.QLabel(
            "Output may exceed 4K, in which case HEVC is used "
            "automatically (better-quality codec; some older "
            "players may need an update).")
        warn.setStyleSheet("color: #777; font-size: 10px;")
        warn.setWordWrap(True)
        up_layout.addRow(warn)
        outer.addWidget(self.upscale_box)

        # --- Run row --------------------------------------------------
        run_row = QtWidgets.QHBoxLayout()
        self.run_btn = QtWidgets.QPushButton("Run")
        self.run_btn.clicked.connect(self._run)
        run_row.addWidget(self.run_btn)
        self.cancel_btn = QtWidgets.QPushButton("Cancel")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self._cancel)
        run_row.addWidget(self.cancel_btn)
        run_row.addStretch(1)
        self.status_label = QtWidgets.QLabel("Ready.")
        self.status_label.setStyleSheet("color: #555;")
        run_row.addWidget(self.status_label)
        outer.addLayout(run_row)

        # --- Log ------------------------------------------------------
        self.log_view = QtWidgets.QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setStyleSheet(
            "font-family: Consolas, monospace; font-size: 11px;")
        self.log_view.setMaximumBlockCount(2000)
        outer.addWidget(self.log_view, 1)

        self.input_edit.textChanged.connect(self._suggest_output)
        self._update_visibility()

    # ----- visibility + paths --------------------------------------

    def _update_visibility(self) -> None:
        self.interp_box.setVisible(self.do_interp.isChecked())
        self.upscale_box.setVisible(self.do_upscale.isChecked())
        self._suggest_output()

    # Suffixes the auto-suggested output filename uses. Tracked so we
    # only overwrite a previous auto-suggestion, not a user-typed path.
    _SUFFIXES = ("_interp.mp4", "_upscale.mp4", "_upscale_interp.mp4")

    def _suggest_output(self) -> None:
        cur = self.output_edit.text().strip()
        src = self.input_edit.text().strip()
        if not src:
            return
        if self.do_upscale.isChecked() and self.do_interp.isChecked():
            suffix = "_upscale_interp.mp4"
        elif self.do_upscale.isChecked():
            suffix = "_upscale.mp4"
        elif self.do_interp.isChecked():
            suffix = "_interp.mp4"
        else:
            return
        suggestion = str(Path(src).with_suffix("")) + suffix
        if not cur or any(cur.endswith(s) for s in self._SUFFIXES):
            self.output_edit.setText(suggestion)

    def _browse_input(self) -> None:
        exts = " ".join(f"*{e}" for e in self.VIDEO_EXTS)
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select source video", "",
            f"Video files ({exts});;All files (*)")
        if path:
            self.input_edit.setText(path)

    def _browse_output(self) -> None:
        cur = self.output_edit.text().strip()
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Output file", cur or "", "MP4 files (*.mp4)")
        if path:
            self.output_edit.setText(path)

    # ----- run / cancel --------------------------------------------

    def _interp_args(self, src: str, dst: str) -> list[str]:
        return ["-m", "waruka", "interpolate", src,
                 "--out", dst,
                 "--fps", str(self.interp_fps_combo.currentData()),
                 "--backend", self.interp_backend_combo.currentData(),
                 "--cq", str(self.interp_cq_spin.value())]

    def _upscale_args(self, src: str, dst: str) -> list[str]:
        return ["-m", "waruka", "upscale", src,
                 "--out", dst,
                 "--cq", str(self.up_cq_spin.value())]

    def _run(self) -> None:
        if self._runner.is_running():
            return
        src = self.input_edit.text().strip()
        dst = self.output_edit.text().strip()
        if not src or not Path(src).exists():
            QtWidgets.QMessageBox.warning(self, "Waruka",
                                            "Pick a source video that "
                                            "exists on disk.")
            return
        if not dst:
            QtWidgets.QMessageBox.warning(self, "Waruka",
                                            "Pick an output path.")
            return
        if not (self.do_interp.isChecked() or self.do_upscale.isChecked()):
            QtWidgets.QMessageBox.warning(self, "Waruka",
                                            "Tick at least one operation "
                                            "(Interpolate and/or Upscale).")
            return

        # Build the stage list.
        # Upscale before interpolate when both: SR runs N times instead of
        # 3N, and even though interp on 2x is slower per pair, the model
        # cost scales sub-linearly with pixels -- net ~3x faster overall.
        self._stages = []
        self._temp_files = []
        if self.do_upscale.isChecked() and self.do_interp.isChecked():
            tmp = str(Path(dst).with_name(
                Path(dst).stem + "_upscaled_tmp.mp4"))
            self._stages.append(
                ("Upscale 2x", self._upscale_args(src, tmp), tmp))
            self._stages.append(
                ("Interpolate", self._interp_args(tmp, dst), dst))
            self._temp_files.append(tmp)
        elif self.do_upscale.isChecked():
            self._stages.append(
                ("Upscale 2x", self._upscale_args(src, dst), dst))
        else:
            self._stages.append(
                ("Interpolate", self._interp_args(src, dst), dst))

        self._stage_total = len(self._stages)
        self._stage_index = 0
        self.log_view.clear()
        self.run_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self._launch_next_stage()

    def _launch_next_stage(self) -> None:
        if not self._stages:
            # All stages finished successfully.
            self._cleanup_temp_files()
            self.run_btn.setEnabled(True)
            self.cancel_btn.setEnabled(False)
            self.status_label.setText("Done.")
            self.status_message.emit("Post-process finished.", 3000)
            return
        label, args, _out = self._stages.pop(0)
        self._stage_index += 1
        self.log_view.appendPlainText(
            f"\n=== Stage {self._stage_index}/{self._stage_total}: "
            f"{label} ===\n$ {' '.join(args)}\n")
        self.status_label.setText(
            f"Stage {self._stage_index}/{self._stage_total}: {label}...")
        self.status_message.emit(
            f"{label} started "
            f"({self._stage_index}/{self._stage_total}).", 3000)
        self._runner.start(args)

    def _cleanup_temp_files(self) -> None:
        for p in self._temp_files:
            try:
                Path(p).unlink(missing_ok=True)
            except Exception as e:  # noqa: BLE001
                self.log_view.appendPlainText(
                    f"[warn] could not remove temp file {p}: {e}")
        self._temp_files = []

    def _cancel(self) -> None:
        # Drop any pending stages so we don't auto-advance after the kill.
        self._stages = []
        if self._runner.is_running():
            self._runner._proc.kill()

    def _on_log_line(self, line: str) -> None:
        self.log_view.appendPlainText(line)

    def _on_finished(self, exit_code: int) -> None:
        if exit_code != 0:
            self.run_btn.setEnabled(True)
            self.cancel_btn.setEnabled(False)
            self.status_label.setText(f"Failed (exit {exit_code}).")
            self.status_message.emit(
                f"Stage failed (exit {exit_code}); "
                f"remaining stages skipped.", 5000)
            self._stages = []
            self._cleanup_temp_files()
            return
        # Stage OK -- advance to next, or finish.
        self._launch_next_stage()


class ConcatTab(QtWidgets.QWidget):
    """Multi-clip concatenation tool.

    Current scope (task #10): the file list + add/remove/reorder
    controls. Codec-mismatch detection (#11), ffmpeg orchestration
    (#12), scrubber (#13), trim + naming (#14), and Track-tab
    handover (#15) land in subsequent tasks.
    """

    status_message = QtCore.Signal(str, int)

    # Video file extensions accepted in both the file picker and the
    # drag-drop handler. Lowercased; comparison is case-insensitive.
    VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".m4v"}

    def __init__(self, main_window: "MainWindow",
                 parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._main = main_window
        # Source of truth: list[ClipEntry] in the order they'll be
        # concatenated. The QListWidget shows the same order; reorder
        # buttons mutate both in lockstep.
        self._clips: list[ClipEntry] = []

        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(16, 16, 16, 16)
        outer.setSpacing(10)

        # --- Header / instructions ----------------------------------------
        hint = QtWidgets.QLabel(
            "Add 5-minute clips below (drag-drop or 'Add files...'). "
            "Use ↑ / ↓ to reorder. Concat will play them top-to-bottom.")
        hint.setStyleSheet("color: #555;")
        hint.setWordWrap(True)
        outer.addWidget(hint)

        # --- File list ----------------------------------------------------
        # Header label uses the same monospace font + widths as the rows
        # so columns line up visually. _refresh_list_widget rebuilds
        # both header and rows in lockstep.
        self.list_header = QtWidgets.QLabel("")
        self.list_header.setStyleSheet(
            "font-family: Consolas, monospace; font-size: 11px; "
            "color: #555; padding: 4px 6px;")
        outer.addWidget(self.list_header)

        self.list_widget = ClipListWidget()
        self.list_widget.files_dropped.connect(self._on_files_dropped)
        self.list_widget.setStyleSheet(
            "QListWidget { font-family: Consolas, monospace; "
            "font-size: 11px; }")
        outer.addWidget(self.list_widget, 1)

        # --- Row of action buttons ----------------------------------------
        btn_row = QtWidgets.QHBoxLayout()
        self.add_btn = QtWidgets.QPushButton("Add files...")
        self.add_btn.clicked.connect(self._on_add_clicked)
        btn_row.addWidget(self.add_btn)

        self.up_btn = QtWidgets.QPushButton("↑ Move up")
        self.up_btn.clicked.connect(lambda: self._move_selected(-1))
        btn_row.addWidget(self.up_btn)

        self.down_btn = QtWidgets.QPushButton("↓ Move down")
        self.down_btn.clicked.connect(lambda: self._move_selected(+1))
        btn_row.addWidget(self.down_btn)

        self.remove_btn = QtWidgets.QPushButton("Remove")
        self.remove_btn.clicked.connect(self._on_remove_clicked)
        btn_row.addWidget(self.remove_btn)

        self.clear_btn = QtWidgets.QPushButton("Clear")
        self.clear_btn.clicked.connect(self._on_clear_clicked)
        btn_row.addWidget(self.clear_btn)

        btn_row.addStretch(1)

        # Total-duration summary on the right
        self.summary_label = QtWidgets.QLabel("0 clips")
        self.summary_label.setStyleSheet("color: #444;")
        btn_row.addWidget(self.summary_label)

        outer.addLayout(btn_row)

        # --- Stream consistency summary -----------------------------------
        # One label per concern, hidden until clips are present.
        # Updated by _refresh_consistency_panel() whenever clips change.
        consistency_box = QtWidgets.QGroupBox("Stream consistency")
        cons_layout = QtWidgets.QVBoxLayout(consistency_box)
        cons_layout.setSpacing(4)

        self.codec_label = QtWidgets.QLabel()
        self.codec_label.setWordWrap(True)
        cons_layout.addWidget(self.codec_label)

        self.audio_label = QtWidgets.QLabel()
        self.audio_label.setWordWrap(True)
        cons_layout.addWidget(self.audio_label)

        # Mixed-audio remediation button. Only shown when audio state
        # is "mixed". Clicking it removes the no-audio clips so the
        # remaining set is all-with-audio (user can also use the Remove
        # button to drop the with-audio ones; we just default to keeping
        # the larger group).
        mixed_row = QtWidgets.QHBoxLayout()
        mixed_row.addStretch(1)
        self.drop_no_audio_btn = QtWidgets.QPushButton(
            "Drop clips without audio")
        self.drop_no_audio_btn.clicked.connect(self._drop_no_audio_clips)
        mixed_row.addWidget(self.drop_no_audio_btn)
        cons_layout.addLayout(mixed_row)

        self._consistency_box = consistency_box
        outer.addWidget(consistency_box)

        # --- Audio choices (sub-panel) ------------------------------------
        # Mirrors the tracking dialog's audio convention:
        #   * Default: preserve audio in the output (when source has it).
        #   * Optional checkbox: also write a silent _no_audio.mp4 copy.
        # Both are locked off when no clip has audio (single output, silent).
        audio_box = QtWidgets.QGroupBox("Audio")
        audio_box_layout = QtWidgets.QVBoxLayout(audio_box)
        audio_box_layout.setSpacing(2)

        self.preserve_audio_check = QtWidgets.QCheckBox(
            "Preserve audio in concatenated output")
        self.preserve_audio_check.setChecked(True)
        # Connect ONCE here; _refresh_consistency_panel manipulates
        # enabled-state only so re-connecting on every refresh is
        # unnecessary (and prints noisy 'failed to disconnect' warnings
        # on PySide6 when the connection doesn't exist yet).
        self.preserve_audio_check.toggled.connect(
            self._on_preserve_audio_toggled)
        audio_box_layout.addWidget(self.preserve_audio_check)

        self.no_audio_copy_check = QtWidgets.QCheckBox(
            "Also write a silent copy (<output>_no_audio.mp4)")
        self.no_audio_copy_check.setChecked(True)
        audio_box_layout.addWidget(self.no_audio_copy_check)

        outer.addWidget(audio_box)
        self._audio_box = audio_box

        # --- Concatenate action + live progress panel ---------------------
        # The button is enabled only when the clip list is in a
        # runnable state (clips present, no mixed audio, no codec
        # mismatch -- or user has acknowledged re-encode).
        concat_box = QtWidgets.QGroupBox("Concatenate")
        concat_layout = QtWidgets.QVBoxLayout(concat_box)
        concat_layout.setSpacing(6)

        # Button row
        btn_outer = QtWidgets.QHBoxLayout()
        self.concat_btn = QtWidgets.QPushButton("▶  Concatenate clips")
        self.concat_btn.setStyleSheet(
            "QPushButton { font-size: 13px; padding: 6px 16px; }")
        self.concat_btn.clicked.connect(self._on_concat_clicked)
        btn_outer.addWidget(self.concat_btn)
        btn_outer.addStretch(1)
        self.kill_concat_btn = QtWidgets.QPushButton("Kill")
        self.kill_concat_btn.clicked.connect(self._on_kill_concat_clicked)
        self.kill_concat_btn.setVisible(False)
        btn_outer.addWidget(self.kill_concat_btn)
        concat_layout.addLayout(btn_outer)

        # Progress section (hidden until a run starts)
        self.concat_stage_label = QtWidgets.QLabel("")
        self.concat_stage_label.setStyleSheet("color: #444;")
        self.concat_stage_label.setVisible(False)
        concat_layout.addWidget(self.concat_stage_label)

        self.concat_progress = QtWidgets.QProgressBar()
        self.concat_progress.setRange(0, 100)
        self.concat_progress.setValue(0)
        self.concat_progress.setVisible(False)
        concat_layout.addWidget(self.concat_progress)

        self.concat_eta_label = QtWidgets.QLabel("")
        self.concat_eta_label.setStyleSheet("color: #777; font-size: 10px;")
        self.concat_eta_label.setVisible(False)
        concat_layout.addWidget(self.concat_eta_label)

        # Success line (small, above the scrubber when shown).
        self.concat_result_label = QtWidgets.QLabel("")
        self.concat_result_label.setStyleSheet(
            "color: #2a7; font-family: Consolas, monospace; font-size: 10px;")
        self.concat_result_label.setWordWrap(True)
        self.concat_result_label.setVisible(False)
        concat_layout.addWidget(self.concat_result_label)

        # Scrubber widget for trim. Hidden until concat completes; loaded
        # with the concat_temp.mp4 in _on_concat_finished.
        self.scrubber = ScrubberWidget(self)
        self.scrubber.setVisible(False)
        self.scrubber.in_out_changed.connect(self._refresh_save_preview)
        concat_layout.addWidget(self.scrubber, 1)

        # --- Save (trim + emit final output) panel ----------------------
        # Sub-panel visible only once concat is done. The name input
        # is pre-filled with the first clip's date (YYYYMMDD format,
        # with trailing space so the user can type team names after).
        save_box = QtWidgets.QGroupBox("Save trimmed output")
        save_layout = QtWidgets.QVBoxLayout(save_box)
        save_layout.setSpacing(4)

        name_row = QtWidgets.QHBoxLayout()
        name_row.addWidget(QtWidgets.QLabel("Filename:"))
        self.save_name_edit = QtWidgets.QLineEdit()
        self.save_name_edit.setPlaceholderText("(uses fallback name)")
        self.save_name_edit.textChanged.connect(
            self._refresh_save_preview)
        name_row.addWidget(self.save_name_edit, 1)
        name_row.addWidget(QtWidgets.QLabel(".mp4"))
        save_layout.addLayout(name_row)

        self.save_preview_label = QtWidgets.QLabel("")
        self.save_preview_label.setStyleSheet(
            "font-family: Consolas, monospace; color: #444; font-size: 10px;")
        self.save_preview_label.setWordWrap(True)
        save_layout.addWidget(self.save_preview_label)

        save_btn_row = QtWidgets.QHBoxLayout()
        self.save_btn = QtWidgets.QPushButton("💾  Save")
        self.save_btn.setStyleSheet(
            "QPushButton { font-size: 13px; padding: 6px 18px; }")
        self.save_btn.clicked.connect(self._on_save_clicked)
        save_btn_row.addWidget(self.save_btn)
        self.discard_btn = QtWidgets.QPushButton("Discard concat")
        self.discard_btn.clicked.connect(self._on_discard_clicked)
        save_btn_row.addWidget(self.discard_btn)
        save_btn_row.addStretch(1)
        save_layout.addLayout(save_btn_row)

        save_box.setVisible(False)
        concat_layout.addWidget(save_box)
        self._save_box = save_box

        outer.addWidget(concat_box, 1)
        self._concat_box = concat_box

        # Trim state -- populated when the trim subprocess is in flight.
        # Re-uses the concat progress widgets to avoid double UI.
        self._trim_runner = StepRunner(self)
        self._trim_runner.finished.connect(self._on_trim_finished)
        self._trim_runner.line.connect(self._on_concat_line)  # same parser
        self._trim_phase: str = "idle"   # "idle" | "trim" | "no_audio_copy"
        self._final_output_path: Path | None = None
        self._no_audio_output_path: Path | None = None
        self._trim_total_s: float = 0.0
        self._trim_started_at: float = 0.0

        # Subprocess + state for the running concat ffmpeg call.
        self._concat_runner = StepRunner(self)
        self._concat_runner.finished.connect(self._on_concat_finished)
        self._concat_runner.line.connect(self._on_concat_line)
        # Total output duration (sum of clip durations) and the most
        # recent out_time we saw, used to drive the progress bar.
        self._concat_total_s: float = 0.0
        self._concat_started_at: float = 0.0
        self._concat_last_out_s: float = 0.0
        self._concat_output_path: Path | None = None

        self._refresh_buttons()
        self._refresh_consistency_panel()

    # ----- file add paths -------------------------------------------------

    def _on_add_clicked(self) -> None:
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self,
            "Add clips to concatenate",
            "",
            "Video files (*.mp4 *.mov *.mkv *.avi *.m4v);;All files (*.*)",
        )
        if paths:
            self._add_paths(paths)

    def _on_files_dropped(self, paths: list[str]) -> None:
        self._add_paths(paths)

    def _add_paths(self, paths: list[str]) -> None:
        """Probe each path and append to the clip list.

        Filters by extension, dedupes against the existing list, sorts
        the *newly-added* batch by basename (so dropping a folder's
        worth of Reolink chunks comes out in timestamp order), then
        appends. The user can still reorder afterwards via ↑/↓.
        """
        # 1. Extension filter
        candidates: list[Path] = []
        for s in paths:
            p = Path(s)
            if p.suffix.lower() in self.VIDEO_EXTS and p.exists():
                candidates.append(p)
        if not candidates:
            QtWidgets.QMessageBox.information(
                self, "Nothing added",
                "Drop or select MP4 / MOV / MKV / AVI / M4V files.")
            return

        # 2. Dedupe vs already-present
        existing = {c.path.resolve() for c in self._clips}
        new_paths = [p for p in candidates if p.resolve() not in existing]
        skipped = len(candidates) - len(new_paths)
        if not new_paths:
            self.status_message.emit(
                f"All {skipped} dropped file(s) were already in the list.",
                4000)
            return

        # 3. Sort the new batch alphabetically (Reolink timestamps are
        # baked into the filename, so basename sort = timestamp sort).
        new_paths.sort(key=lambda p: p.name.lower())

        # 4. Probe metadata + codec under a busy cursor (~100-300ms per
        # file due to ffmpeg startup; 16 Reolink chunks ~= 3s).
        QtWidgets.QApplication.setOverrideCursor(
            QtCore.Qt.CursorShape.WaitCursor)
        try:
            for p in new_paths:
                info = VideoInfo.probe(p)
                if info is None:
                    print(f"[concat] failed to probe {p}; skipping")
                    continue
                codec = _probe_video_codec(p)
                created = _extract_clip_datetime(p)
                self._clips.append(ClipEntry(path=p, info=info,
                                              video_codec=codec,
                                              created_at=created))
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

        self._refresh_list_widget()
        self._refresh_buttons()
        self._refresh_consistency_panel()

        added = len(new_paths)
        msg = f"Added {added} clip(s)"
        if skipped:
            msg += f"; skipped {skipped} duplicate(s)"
        self.status_message.emit(msg, 4000)

    # ----- list mutations -------------------------------------------------

    def _on_remove_clicked(self) -> None:
        rows = sorted({i.row() for i in self.list_widget.selectedIndexes()},
                       reverse=True)
        if not rows:
            return
        for r in rows:
            del self._clips[r]
        self._refresh_list_widget()
        self._refresh_buttons()
        self._refresh_consistency_panel()

    def _on_clear_clicked(self) -> None:
        if not self._clips:
            return
        if QtWidgets.QMessageBox.question(
            self, "Clear list",
            f"Remove all {len(self._clips)} clip(s) from the list?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No,
        ) != QtWidgets.QMessageBox.Yes:
            return
        self._clips.clear()
        self._refresh_list_widget()
        self._refresh_buttons()
        self._refresh_consistency_panel()

    def _move_selected(self, delta: int) -> None:
        """Move every selected row by ``delta`` positions (preserving
        relative order within the selection). Clamps at the edges."""
        rows = sorted({i.row() for i in self.list_widget.selectedIndexes()})
        if not rows:
            return
        # Direction determines iteration order so we don't overwrite
        # items we haven't moved yet.
        if delta < 0:
            # moving up -- iterate top-to-bottom
            for r in rows:
                new_r = r + delta
                if new_r < 0:
                    continue
                self._clips[new_r], self._clips[r] = (
                    self._clips[r], self._clips[new_r])
        else:
            for r in reversed(rows):
                new_r = r + delta
                if new_r >= len(self._clips):
                    continue
                self._clips[new_r], self._clips[r] = (
                    self._clips[r], self._clips[new_r])
        self._refresh_list_widget()
        # Restore the selection on the moved rows so successive ↑↑↑
        # keeps the same selection.
        new_rows = [max(0, min(len(self._clips) - 1, r + delta))
                    for r in rows]
        self.list_widget.clearSelection()
        for r in new_rows:
            self.list_widget.item(r).setSelected(True)

    # ----- redraw ---------------------------------------------------------

    def _refresh_list_widget(self) -> None:
        """Repaint header + every row. Columns are padded to fixed widths
        derived from the current set of clips so values line up
        regardless of filename length variance (Reolink names differ by
        1-3 chars due to the trailing checksum).

        Column order:
          filename | recorded | duration | audio | codec | resolution | fps
        Text columns left-aligned, numeric columns right-aligned.
        """
        self.list_widget.clear()
        if not self._clips:
            self.list_header.setText("")
            return

        # Compute column widths from the longest value currently present
        # (with sensible lower bounds so columns don't squish to zero).
        max_name = max(len(c.path.name) for c in self._clips)
        max_name = max(max_name, len("Filename"))
        max_codec = max(len(c.video_codec or "?") for c in self._clips)
        max_codec = max(max_codec, 5)   # at least "Codec"
        max_res = max(len(f"{c.info.width}x{c.info.height}")
                      for c in self._clips)
        max_res = max(max_res, 10)     # at least "Resolution"

        def row(name: str, created: str, dur: str, audio: str,
                codec: str, res: str, fps: str) -> str:
            return (f"{name.ljust(max_name)}  "
                    f"{created.ljust(19)}  "
                    f"{dur.rjust(8)}  "
                    f"{audio.center(3)}  "
                    f"{codec.ljust(max_codec)}  "
                    f"{res.ljust(max_res)}  "
                    f"{fps.rjust(7)}")

        self.list_header.setText(
            row("Filename", "Recorded", "Duration",
                "Aud", "Codec", "Resolution", "FPS"))

        for c in self._clips:
            i = c.info
            text = row(
                c.path.name,
                c.created_at or "—",
                _fmt_hms(i.duration_s),
                "♪" if i.has_audio else "—",
                c.video_codec or "?",
                f"{i.width}x{i.height}",
                f"{i.fps:.1f}fps",
            )
            it = QtWidgets.QListWidgetItem(text)
            it.setToolTip(str(c.path))
            self.list_widget.addItem(it)

    def _refresh_buttons(self) -> None:
        have = bool(self._clips)
        self.up_btn.setEnabled(have)
        self.down_btn.setEnabled(have)
        self.remove_btn.setEnabled(have)
        self.clear_btn.setEnabled(have)

        total = sum(c.info.duration_s for c in self._clips)
        n = len(self._clips)
        self.summary_label.setText(
            f"{n} clip{'s' if n != 1 else ''}, total {_fmt_hms(total)}")

    # ----- consistency checks --------------------------------------------

    def _audio_state(self) -> str:
        """Classify the current clip set's audio presence.

        Returns one of:
          ``"empty"``      -- no clips loaded
          ``"all_have"``   -- every clip carries an audio stream
          ``"none_have"``  -- no clip carries an audio stream
          ``"mixed"``      -- some do, some don't (ffmpeg ``-c copy``
                              concat will refuse this without remediation)
        """
        if not self._clips:
            return "empty"
        with_audio = sum(1 for c in self._clips if c.info.has_audio)
        if with_audio == len(self._clips):
            return "all_have"
        if with_audio == 0:
            return "none_have"
        return "mixed"

    def _codec_state(self) -> tuple[str, str]:
        """Classify codec/resolution/fps consistency.

        Returns ``(state, message)`` where state is:
          ``"empty"``       -- no clips loaded
          ``"consistent"``  -- all clips share codec + WxH + fps
          ``"mismatch"``    -- at least one differs; lossless concat
                                won't work, would need re-encode
        and message is a human-readable summary suitable for the
        consistency panel label.
        """
        if not self._clips:
            return "empty", ""

        def sig(c: ClipEntry) -> tuple:
            return (c.video_codec, c.info.width, c.info.height,
                    round(c.info.fps, 2))
        sigs = {sig(c) for c in self._clips}
        if len(sigs) == 1:
            codec, w, h, fps = next(iter(sigs))
            return ("consistent",
                    f"{codec} {w}x{h} @ {fps:g}fps -- "
                    "lossless concat (-c copy)")
        # Mismatch: enumerate the unique signatures so the user knows
        # which clips differ.
        parts = []
        for codec, w, h, fps in sorted(sigs):
            n = sum(1 for c in self._clips if sig(c) == (codec, w, h, fps))
            parts.append(f"{n}x {codec} {w}x{h} @ {fps:g}fps")
        return ("mismatch", "Mixed: " + " | ".join(parts)
                + ". Will need re-encode (slower).")

    def _refresh_consistency_panel(self) -> None:
        """Update the consistency-box labels + audio-box state.

        Also drives:
          - Visibility of the 'Drop clips without audio' button.
          - Whether the audio checkboxes are enabled.
          - Whether the preserve-audio checkbox auto-defaults on/off.
        """
        # Codec line
        codec_state, codec_msg = self._codec_state()
        if codec_state == "empty":
            self._consistency_box.setVisible(False)
        else:
            self._consistency_box.setVisible(True)
            if codec_state == "consistent":
                self.codec_label.setText(f"<b>Video:</b> ✓ {codec_msg}")
                self.codec_label.setStyleSheet("color: #2a7;")
            else:
                self.codec_label.setText(f"<b>Video:</b> ⚠ {codec_msg}")
                self.codec_label.setStyleSheet("color: #c84;")

        # Audio line
        audio_state = self._audio_state()
        if audio_state == "all_have":
            self.audio_label.setText(
                "<b>Audio:</b> ✓ all clips have audio")
            self.audio_label.setStyleSheet("color: #2a7;")
            self.drop_no_audio_btn.setVisible(False)
        elif audio_state == "none_have":
            self.audio_label.setText(
                "<b>Audio:</b> — no clip has audio (output will be silent)")
            self.audio_label.setStyleSheet("color: #888;")
            self.drop_no_audio_btn.setVisible(False)
        elif audio_state == "mixed":
            with_n = sum(1 for c in self._clips if c.info.has_audio)
            without_n = len(self._clips) - with_n
            self.audio_label.setText(
                f"<b>Audio:</b> ⚠ {with_n} clip(s) have audio, "
                f"{without_n} don't. Lossless concat needs all-or-none. "
                "Drop the silent ones or remove the with-audio ones.")
            self.audio_label.setStyleSheet("color: #c84;")
            self.drop_no_audio_btn.setVisible(True)

        # Audio-box checkbox state: lock off when nothing to preserve.
        # Policy: when audio can be preserved (all_have) -> enable +
        # default-check. When it can't (empty / mixed / none_have) ->
        # disable + uncheck so the user can see clearly what will
        # happen. Programmatic setChecked is signal-blocked so the
        # toggled handler only fires on real user clicks.
        can_preserve = (audio_state == "all_have")
        self.preserve_audio_check.setEnabled(can_preserve)
        self.preserve_audio_check.blockSignals(True)
        self.preserve_audio_check.setChecked(can_preserve)
        self.preserve_audio_check.blockSignals(False)
        # The "also write silent copy" checkbox is only useful when
        # we're actually preserving audio in the main output.
        self.no_audio_copy_check.setEnabled(
            can_preserve and self.preserve_audio_check.isChecked())

    def _on_preserve_audio_toggled(self, on: bool) -> None:
        self.no_audio_copy_check.setEnabled(on)
        if not on:
            self.no_audio_copy_check.setChecked(False)

    def _drop_no_audio_clips(self) -> None:
        """Remove every clip lacking an audio stream. Mixed-state
        remediation; user can also remove the with-audio ones via
        the regular Remove button if they prefer to go silent."""
        before = len(self._clips)
        self._clips = [c for c in self._clips if c.info.has_audio]
        dropped = before - len(self._clips)
        if dropped:
            self._refresh_list_widget()
            self._refresh_buttons()
            self._refresh_consistency_panel()
            self.status_message.emit(
                f"Dropped {dropped} clip(s) without audio.", 4000)

    # ----- concat orchestration ------------------------------------------

    def _concat_workspace(self) -> Path:
        """Per-session workspace dir for the concat temp file + ffmpeg
        concat list. Created next to the first clip so the user can
        find it if anything goes wrong; cleaned up after a successful
        trim in task #14."""
        first = self._clips[0].path
        ws = first.parent / "waruka_concat_workspace"
        ws.mkdir(exist_ok=True)
        return ws

    def _can_run_concat(self) -> tuple[bool, str]:
        """Return (ok, reason). Reason is shown to the user when not ok."""
        if not self._clips:
            return False, "Add at least one clip first."
        if len(self._clips) < 2:
            return (False,
                    "Need 2+ clips for concat. (A single clip can be "
                    "dropped straight into the Track tab.)")
        if self._audio_state() == "mixed":
            return (False,
                    "Audio state is mixed -- some clips have audio, "
                    "some don't. Drop the silent ones (or remove the "
                    "with-audio ones) before concatenating.")
        return True, ""

    def _on_concat_clicked(self) -> None:
        ok, reason = self._can_run_concat()
        if not ok:
            QtWidgets.QMessageBox.information(
                self, "Can't concatenate yet", reason)
            return

        # Detect codec mismatch -- needs re-encode. Confirm with the
        # user since it's the slow path.
        codec_state, codec_msg = self._codec_state()
        re_encode = (codec_state == "mismatch")
        if re_encode:
            est_min = max(1, int(sum(c.info.duration_s
                                      for c in self._clips) / 60.0))
            if QtWidgets.QMessageBox.question(
                self, "Re-encode required",
                f"Clips don't share the same codec/resolution/fps:\n\n"
                f"{codec_msg}\n\n"
                f"Lossless concat (-c copy) won't work; the tool will "
                f"re-encode the combined stream instead. Rough estimate: "
                f"~{est_min} min wall time on a CPU encode. Continue?",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                QtWidgets.QMessageBox.Yes,
            ) != QtWidgets.QMessageBox.Yes:
                return

        # Decide audio handling for ffmpeg.
        keep_audio = (self._audio_state() == "all_have"
                       and self.preserve_audio_check.isChecked())

        # Build the concat list file + output path.
        ws = self._concat_workspace()
        list_file = ws / "list.txt"
        with open(list_file, "w", encoding="utf-8") as f:
            for c in self._clips:
                # Single-quote escape per ffmpeg concat demuxer syntax.
                p = str(c.path).replace("'", r"'\''")
                f.write(f"file '{p}'\n")

        out_path = ws / "concat_temp.mp4"
        if out_path.exists():
            out_path.unlink()
        self._concat_output_path = out_path

        # Build the ffmpeg argv.
        args = ["-y",
                "-hide_banner",
                "-f", "concat", "-safe", "0",
                "-i", str(list_file)]
        if re_encode:
            args += ["-c:v", "libx264", "-crf", "18",
                     "-preset", "veryfast", "-pix_fmt", "yuv420p"]
            args += ["-c:a", "aac"] if keep_audio else ["-an"]
        else:
            args += ["-c", "copy"]
            if not keep_audio:
                args += ["-an"]
        # -progress writes key=value lines to stdout; -nostats suppresses
        # the usual two-line carriage-returning status spam.
        args += ["-progress", "pipe:1", "-nostats", str(out_path)]

        # Locate the ffmpeg binary.
        try:
            import imageio_ffmpeg
            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception as e:
            QtWidgets.QMessageBox.warning(
                self, "ffmpeg not found",
                f"Couldn't locate the bundled ffmpeg binary: {e}")
            return

        # Tee a header into the log + set up live progress state.
        self._concat_total_s = sum(c.info.duration_s for c in self._clips)
        self._concat_last_out_s = 0.0
        import time
        self._concat_started_at = time.monotonic()
        mode = "re-encode" if re_encode else "lossless -c copy"
        keep_audio_str = "preserve audio" if keep_audio else "no audio"
        self._set_concat_running_ui(True)
        self.concat_stage_label.setText(
            f"Concatenating {len(self._clips)} clips ({mode}, "
            f"{keep_audio_str})...")
        self.status_message.emit(
            f"Running ffmpeg concat ({mode})...", 0)

        self._concat_runner.start(args, cwd=ws, program=ffmpeg_exe)

    def _set_concat_running_ui(self, running: bool) -> None:
        """Toggle the run-state-dependent UI: hide/show progress widgets,
        disable/enable the file-list buttons + concat button."""
        self.concat_stage_label.setVisible(running)
        self.concat_progress.setVisible(running)
        self.concat_eta_label.setVisible(running)
        self.concat_result_label.setVisible(False)
        self.kill_concat_btn.setVisible(running)
        # File-list mutations during a run would corrupt the in-flight
        # list.txt; lock them down.
        for w in (self.add_btn, self.up_btn, self.down_btn,
                   self.remove_btn, self.clear_btn,
                   self.preserve_audio_check, self.no_audio_copy_check,
                   self.concat_btn, self.list_widget):
            w.setEnabled(not running)
        if not running:
            self._refresh_buttons()           # restore correct enabled state
            self._refresh_consistency_panel()  # restore audio toggles

    def _on_concat_line(self, line: str) -> None:
        """Parse the ``-progress pipe:1`` key=value stream.

        Useful keys:
          ``out_time_ms`` -- output media time in microseconds
          ``progress``    -- 'continue' (mid-run) or 'end' (finished)
        Everything else goes to the captured log as-is for debugging.
        """
        if "=" not in line:
            return
        key, val = line.split("=", 1)
        if key == "out_time_ms":
            try:
                # ffmpeg confusingly labels microseconds as "_ms"
                us = int(val)
                self._concat_last_out_s = us / 1e6
                self._update_concat_progress()
            except ValueError:
                pass
        elif key == "progress" and val.strip() == "end":
            # The 'end' tick is just informational; finished() handles
            # the real completion.
            self._concat_last_out_s = self._concat_total_s

    def _update_concat_progress(self) -> None:
        import time
        if self._concat_total_s <= 0:
            return
        frac = min(1.0, self._concat_last_out_s / self._concat_total_s)
        self.concat_progress.setValue(int(round(100 * frac)))
        elapsed = time.monotonic() - self._concat_started_at
        if frac > 1e-4:
            eta = elapsed * (1.0 - frac) / frac
        else:
            eta = None
        self.concat_eta_label.setText(
            f"out_time {_fmt_hms(self._concat_last_out_s)} / "
            f"{_fmt_hms(self._concat_total_s)}    "
            f"elapsed {_fmt_hms(elapsed)}    "
            f"eta {_fmt_hms(eta)}"
        )

    def _on_concat_finished(self, exit_code: int) -> None:
        self._set_concat_running_ui(False)
        out = self._concat_output_path
        if exit_code != 0:
            log = self._concat_runner.log() or "(no output captured)"
            QtWidgets.QMessageBox.warning(
                self,
                f"Concat failed (exit {exit_code})",
                f"ffmpeg exited with code {exit_code}.\n\n"
                f"Last output:\n\n{log[-2000:]}",
            )
            self.status_message.emit(
                f"Concat failed (exit {exit_code})", 5000)
            self.concat_stage_label.setText(
                f"Failed (exit {exit_code}). "
                "See the message box for ffmpeg output.")
            self.concat_stage_label.setVisible(True)
            return

        if out is None or not out.exists():
            self.status_message.emit("Concat finished but no output file?",
                                       5000)
            return

        # Sanity check: probe the output's duration to confirm it
        # matches the expected total (within 1 second).
        info = VideoInfo.probe(out)
        if info is not None and self._concat_total_s > 0:
            drift = abs(info.duration_s - self._concat_total_s)
            if drift > 1.0:
                print(f"[concat] duration drift: expected "
                      f"{self._concat_total_s:.1f}s, got "
                      f"{info.duration_s:.1f}s (diff {drift:.1f}s)")

        self.status_message.emit("Concat finished.", 4000)
        self.concat_stage_label.setText(
            f"✓ Concatenated {len(self._clips)} clips. "
            "Scrub below to set trim points, then click Save.")
        self.concat_stage_label.setVisible(True)
        self.concat_result_label.setText(f"Intermediate: {out}")
        self.concat_result_label.setVisible(True)
        # Load the temp file into the scrubber, reveal the save panel.
        self.scrubber.load_video(out)
        self.scrubber.setVisible(True)
        self._show_save_panel()

    def _on_kill_concat_clicked(self) -> None:
        if not self._concat_runner.is_running():
            return
        if QtWidgets.QMessageBox.question(
            self, "Kill concat",
            "Stop the running ffmpeg concat?\n"
            "Partial output will be deleted.",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No,
        ) != QtWidgets.QMessageBox.Yes:
            return
        # ffmpeg gets killed; the finished signal will then fire with
        # a non-zero exit code, which surfaces the error path normally.
        if self._concat_runner._proc is not None:
            self._concat_runner._proc.kill()
        # Clean up any partial output.
        out = self._concat_output_path
        if out is not None and out.exists():
            try:
                out.unlink()
            except Exception:
                pass

    # ----- save / trim ---------------------------------------------------

    def _fallback_output_basename(self) -> str:
        """Default output basename when the user leaves the name input
        empty. Convention: ``<first_clip_stem>_full`` (so dropping the
        16 RecM03_..._D0D41CE chunks gives ``..._D0D41CE_full.mp4``)."""
        if not self._clips:
            return "concat_output"
        return f"{self._clips[0].path.stem}_full"

    def _date_prefix_from_first_clip(self) -> str:
        """Extract a YYYYMMDD prefix from the first clip's recording
        timestamp, ready to use as a typing prompt in the name input.
        Returns '' if no usable date was extracted."""
        if not self._clips:
            return ""
        created = self._clips[0].created_at
        if not created:
            return ""
        # created_at is "YYYY-MM-DD HH:MM:SS"; strip out the dashes
        # from the date half to match the user's "20260516 TeamA..."
        # convention.
        try:
            date_part = created.split(" ", 1)[0]
            return date_part.replace("-", "")
        except Exception:
            return ""

    def _final_output_for_name(self, name: str) -> Path:
        """Resolve the user-typed (or fallback) name to an absolute path
        next to the first clip. Appends .mp4 if the user omits it."""
        name = name.strip() or self._fallback_output_basename()
        if not name.lower().endswith(".mp4"):
            name += ".mp4"
        # First clip's parent is the chosen output dir.
        return self._clips[0].path.parent / name

    def _refresh_save_preview(self) -> None:
        """Update the path + duration preview shown above the Save
        button. Called whenever the name input or scrubber in/out
        changes."""
        # NB: isHidden() (not isVisible()) -- isVisible returns False
        # for any unshown ancestor chain, which would silently swallow
        # preview updates under offscreen Qt (and in the smoketests).
        if self._save_box.isHidden() or not self._clips:
            return
        name = self.save_name_edit.text() or self._fallback_output_basename()
        path = self._final_output_for_name(name)
        dur = self.scrubber.trim_duration_seconds()
        full_dur = self.scrubber.duration_seconds()
        lines = [f"→ {path}"]
        if full_dur > 0:
            in_s = self.scrubber.in_seconds() or 0.0
            out_s = (self.scrubber.out_seconds()
                      if self.scrubber.out_seconds() is not None
                      else full_dur)
            lines.append(
                f"  duration: {_fmt_hms(dur)}  "
                f"(trimmed from {_fmt_hms(full_dur)};  "
                f"in={_fmt_hms(in_s)} out={_fmt_hms(out_s)})")
        if self._audio_state() == "all_have" and self.preserve_audio_check.isChecked():
            if self.no_audio_copy_check.isChecked():
                lines.append("  + silent copy: "
                              f"{path.with_name(path.stem + '_no_audio' + path.suffix).name}")
        self.save_preview_label.setText("\n".join(lines))
        # Warn (visually) if the file already exists -- Save will block it
        # (Waruka won't overwrite), so prompt for a different name.
        if path.exists():
            self.save_preview_label.setStyleSheet(
                "font-family: Consolas, monospace; color: #c84; "
                "font-size: 10px;")
            self.save_preview_label.setText(
                self.save_preview_label.text()
                + "\n  ⚠ file exists -- choose another name (Save will block this)")
        else:
            self.save_preview_label.setStyleSheet(
                "font-family: Consolas, monospace; color: #444; "
                "font-size: 10px;")

    def _show_save_panel(self) -> None:
        """Populate + reveal the save sub-panel. Called after concat
        completes successfully."""
        # Pre-fill date prefix as a typing-prompt: user can extend it
        # with team names, or clear it entirely to fall back to
        # <first_stem>_full.
        prefix = self._date_prefix_from_first_clip()
        if prefix:
            self.save_name_edit.setText(f"{prefix} ")
            # Move cursor to the end so subsequent typing appends.
            self.save_name_edit.setCursorPosition(len(prefix) + 1)
        else:
            self.save_name_edit.clear()
        # Placeholder shows the fallback so it's clear what gets used
        # when the input is blank.
        self.save_name_edit.setPlaceholderText(
            self._fallback_output_basename())
        self._save_box.setVisible(True)
        self._refresh_save_preview()

    def _on_discard_clicked(self) -> None:
        """Throw away the concat temp file + reset the UI back to the
        clip-list stage. User can then re-run concat with different
        clips or options."""
        if QtWidgets.QMessageBox.question(
            self, "Discard concat",
            "Delete the intermediate concat file and start over?\n"
            "(Source clips are not affected.)",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No,
        ) != QtWidgets.QMessageBox.Yes:
            return
        self.scrubber.stop()
        out = self._concat_output_path
        if out is not None and out.exists():
            try:
                out.unlink()
            except Exception:
                pass
        # Hide success UI, leave the clip list intact.
        self.scrubber.setVisible(False)
        self._save_box.setVisible(False)
        self.concat_result_label.setVisible(False)
        self.concat_stage_label.setVisible(False)

    def _on_save_clicked(self) -> None:
        """Kick off the trim ffmpeg pass. On success we may also write
        the silent companion copy + then hand off to the Track tab
        (task #15)."""
        if self._concat_output_path is None or not self._concat_output_path.exists():
            QtWidgets.QMessageBox.warning(
                self, "No concat output", "Concat temp file is missing.")
            return

        name = self.save_name_edit.text()
        final = self._final_output_for_name(name)
        try:
            final.parent.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            QtWidgets.QMessageBox.warning(
                self, "Output path not writable",
                f"Couldn't create parent directory:\n{e}")
            return

        self._final_output_path = final
        # If the user wants the silent copy AND we're preserving audio,
        # plan it for after the main trim.
        wants_no_audio_copy = (
            self._audio_state() == "all_have"
            and self.preserve_audio_check.isChecked()
            and self.no_audio_copy_check.isChecked())
        self._no_audio_output_path = (
            final.with_name(final.stem + "_no_audio" + final.suffix)
            if wants_no_audio_copy else None)

        # Output-safety validation (parity with the queue; fail closed).
        # Never overwrite a source clip, and never silently clobber an
        # existing file -- the user must pick a clear name.
        inputs = [c.path for c in self._clips]
        targets = [("output", final)]
        if self._no_audio_output_path is not None:
            targets.append(("silent copy", self._no_audio_output_path))
        for label, pth in targets:
            try:
                assert_output_safe(pth, inputs)
            except ValueError:
                QtWidgets.QMessageBox.critical(
                    self, "Waruka -- unsafe output name",
                    f"The {label} would overwrite one of your source "
                    f"clips, destroying footage:\n\n{pth}\n\n"
                    "Choose a different name.")
                return
            if Path(pth).exists():
                QtWidgets.QMessageBox.critical(
                    self, "Waruka -- output already exists",
                    f"The {label} already exists:\n\n{pth}\n\n"
                    "Pick a different name (or delete the existing file "
                    "first); Waruka will not overwrite it.")
                return

        # Locate ffmpeg
        try:
            import imageio_ffmpeg
            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception as e:
            QtWidgets.QMessageBox.warning(
                self, "ffmpeg not found",
                f"Couldn't locate the bundled ffmpeg binary: {e}")
            return

        # Build trim argv. -ss before -i = fast keyframe seek (no
        # re-encode required under -c copy). Output start may snap to
        # the nearest keyframe before the requested in-point, which on
        # Reolink with ~2s GOPs is acceptable for tactics review.
        in_s = self.scrubber.in_seconds()
        out_s = self.scrubber.out_seconds()
        args = ["-y", "-hide_banner"]
        if in_s is not None and in_s > 0:
            args += ["-ss", f"{in_s:.3f}"]
        if out_s is not None:
            args += ["-to", f"{out_s:.3f}"]
        args += ["-i", str(self._concat_output_path), "-c", "copy"]
        if not (self._audio_state() == "all_have"
                 and self.preserve_audio_check.isChecked()):
            args += ["-an"]
        args += ["-progress", "pipe:1", "-nostats", str(final)]

        # Switch into trim phase. Hide save controls, swap progress
        # widgets back into the visible-running state, drive the
        # progress bar from the same _on_concat_line handler.
        self._trim_phase = "trim"
        self._trim_total_s = self.scrubber.trim_duration_seconds()
        import time
        self._trim_started_at = time.monotonic()
        self._concat_total_s = self._trim_total_s  # for the progress parser
        self._concat_last_out_s = 0.0
        self._concat_started_at = self._trim_started_at
        self._save_box.setEnabled(False)
        self.scrubber.stop()
        self.concat_stage_label.setText(
            f"Trimming to {final.name}...")
        self.concat_stage_label.setVisible(True)
        self.concat_progress.setRange(0, 100)
        self.concat_progress.setValue(0)
        self.concat_progress.setVisible(True)
        self.concat_eta_label.setVisible(True)
        self.kill_concat_btn.setVisible(True)
        self.status_message.emit(
            f"Trimming to {final.name}...", 0)
        # Wire kill to the trim runner instead of the concat one for
        # the duration of the trim phase.
        try:
            self.kill_concat_btn.clicked.disconnect()
        except (TypeError, RuntimeError):
            pass
        self.kill_concat_btn.clicked.connect(self._on_kill_trim_clicked)

        self._trim_runner.start(args, cwd=final.parent, program=ffmpeg)

    def _on_kill_trim_clicked(self) -> None:
        if not self._trim_runner.is_running():
            return
        if QtWidgets.QMessageBox.question(
            self, "Kill trim",
            "Stop the trim/save? Partial output will be deleted.",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No,
        ) != QtWidgets.QMessageBox.Yes:
            return
        if self._trim_runner._proc is not None:
            self._trim_runner._proc.kill()
        for p in (self._final_output_path, self._no_audio_output_path):
            if p is not None and p.exists():
                try:
                    p.unlink()
                except Exception:
                    pass

    def _on_trim_finished(self, exit_code: int) -> None:
        # Restore kill button binding back to the concat path for next time.
        try:
            self.kill_concat_btn.clicked.disconnect()
        except (TypeError, RuntimeError):
            pass
        self.kill_concat_btn.clicked.connect(self._on_kill_concat_clicked)

        phase = self._trim_phase
        self._trim_phase = "idle"

        if exit_code != 0:
            log = self._trim_runner.log() or "(no output captured)"
            QtWidgets.QMessageBox.warning(
                self,
                f"Trim failed (exit {exit_code})",
                f"ffmpeg exited with code {exit_code}.\n\n"
                f"Last output:\n\n{log[-2000:]}",
            )
            self.concat_stage_label.setText(
                f"Trim failed (exit {exit_code}).")
            self.kill_concat_btn.setVisible(False)
            self.concat_progress.setVisible(False)
            self.concat_eta_label.setVisible(False)
            self._save_box.setEnabled(True)
            return

        # Trim phase done. Optionally kick off the silent-copy pass.
        if phase == "trim" and self._no_audio_output_path is not None:
            self._run_no_audio_copy()
            return

        # All done. Hide the running-state UI and surface the result.
        self.kill_concat_btn.setVisible(False)
        self.concat_progress.setVisible(False)
        self.concat_eta_label.setVisible(False)
        self._save_box.setEnabled(True)
        final = self._final_output_path
        extra = ""
        if self._no_audio_output_path is not None and self._no_audio_output_path.exists():
            extra = f"\n+ silent copy: {self._no_audio_output_path.name}"
        self.concat_stage_label.setText(
            f"✓ Saved {final.name}{extra}")
        self.status_message.emit(f"Saved {final.name}", 5000)

        # Release the scrubber's cv2 capture handle so the temp file
        # can be unlinked. cv2 releases synchronously so no defer
        # needed -- but keep the singleShot tick to give pending paint
        # events a moment to settle.
        try:
            self.scrubber._close_capture()
        except Exception:
            pass
        QtCore.QTimer.singleShot(50, self._cleanup_workspace)

        # Hand off to the Track tab (#15 hooks in here next).
        self._handoff_to_track()

    def _run_no_audio_copy(self) -> None:
        """Second ffmpeg pass: copy the final trimmed output, strip
        audio, write as <name>_no_audio.mp4. Cheap -- just remuxes."""
        try:
            import imageio_ffmpeg
            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception as e:
            print(f"[concat] no_audio copy skipped, ffmpeg gone: {e}")
            self._finish_after_no_audio_copy()
            return

        self._trim_phase = "no_audio_copy"
        args = ["-y", "-hide_banner",
                "-i", str(self._final_output_path),
                "-c", "copy", "-an",
                "-progress", "pipe:1", "-nostats",
                str(self._no_audio_output_path)]
        import time
        self._concat_total_s = self._trim_total_s   # same duration
        self._concat_last_out_s = 0.0
        self._concat_started_at = time.monotonic()
        self.concat_stage_label.setText(
            f"Writing silent copy ({self._no_audio_output_path.name})...")
        self._trim_runner.start(
            args, cwd=self._no_audio_output_path.parent, program=ffmpeg)

    def _finish_after_no_audio_copy(self) -> None:
        """Wrap-up shared between successful no_audio copy and the
        fallback when no copy was requested."""
        self._on_trim_finished(0)

    def _cleanup_workspace(self) -> None:
        """Delete the concat_temp.mp4 + list.txt + (empty) workspace
        dir. Scheduled via QTimer.singleShot after _on_trim_finished
        so Qt's media framework has time to release the file handle
        (see comment in _on_trim_finished)."""
        try:
            if self._concat_output_path and self._concat_output_path.exists():
                self._concat_output_path.unlink()
            ws = self._concat_workspace()
            (ws / "list.txt").unlink(missing_ok=True)
            try:
                ws.rmdir()  # only succeeds if empty
            except OSError:
                pass
        except Exception as e:
            # Non-fatal -- the temp will be overwritten on next concat
            # run. Worst case: a 400+ MB file sits in the workspace
            # until the user does another concat or deletes it manually.
            print(f"[concat] cleanup warning: {e}")

    def _handoff_to_track(self) -> None:
        """Switch to the Track tab and load the freshly-trimmed output.

        The Track tab's set_video() handler will derive a new artefact
        dir for the trimmed file (alongside the source dir, not the
        original Reolink chunks' dir), reset all step cards to pending,
        and the user can immediately calibrate -> markfield -> process.

        Source clip list is left intact in the Concat tab so the user
        can do another run if they want; manually Clear to reset."""
        if self._final_output_path is None or not self._final_output_path.exists():
            return
        try:
            self._main.activate_track_with(self._final_output_path)
            self.status_message.emit(
                f"Loaded {self._final_output_path.name} into Track tab.",
                5000)
        except Exception as e:
            print(f"[concat] handoff failed: {e}")

    # ----- accessor used by later tasks ----------------------------------

    @property
    def clips(self) -> list[ClipEntry]:
        return list(self._clips)

    def preserve_audio(self) -> bool:
        """Whether the concat output should carry source audio."""
        return self.preserve_audio_check.isChecked()

    def create_no_audio_copy(self) -> bool:
        """Whether to ALSO write a silent companion .mp4."""
        return self.no_audio_copy_check.isChecked()


# --------------------------------------------------------------------------
# Queue tab -- overnight batch processing (#35)
# --------------------------------------------------------------------------

# Where per-job logs are archived. One file per job, appended on each run.
_QUEUE_LOG_DIR = Path.home() / ".waruka" / "logs"


class QueueRunner(QtCore.QObject):
    """Sequential batch runner for a JobQueue.

    Wires a StepRunner to launch one stage at a time, persists state on
    every transition, polls ``_progress.json`` in the broadcast output's
    parent for per-stage progress, and honours pause requests at stage
    boundaries (i.e. the current stage runs to completion, then the
    queue stops). Emits Qt signals the QueueTab subscribes to.
    """

    job_started = QtCore.Signal(str)                  # job_id
    job_finished = QtCore.Signal(str, bool)           # job_id, success
    stage_started = QtCore.Signal(str, str)           # job_id, stage_name
    stage_finished = QtCore.Signal(str, str, bool)    # job_id, stage_name, ok
    stage_progress = QtCore.Signal(str, str, float, dict)
        # job_id, stage_name, fraction (-1 = indeterminate), extras dict
    queue_idle = QtCore.Signal()
    queue_state_changed = QtCore.Signal()
    log_line = QtCore.Signal(str, str)                # job_id, line

    def __init__(self, queue: JobQueue,
                 parent: QtCore.QObject | None = None) -> None:
        super().__init__(parent)
        self.queue = queue
        self.runner = StepRunner(self)
        self.runner.finished.connect(self._on_stage_exit)
        self.runner.line.connect(self._on_log_line)
        self._current_job: Optional[Job] = None
        self._current_stage_name: Optional[str] = None
        self._current_cleanup_paths: list[str] = []
        self._progress_dir: Optional[Path] = None
        self._stopping: bool = False
        self._poll = QtCore.QTimer(self)
        self._poll.setInterval(500)
        self._poll.timeout.connect(self._poll_progress)
        self._log_file = None
        # Per-job log path -- now derived from the job's artefact dir so
        # the log lives next to the chunks / JSON intermediates rather
        # than in ~/.waruka/logs. Kept current for the "Open log..."
        # button in the QueueTab.
        self._current_log_path: Optional[Path] = None

    # ----- Public controls --------------------------------------------

    def is_running(self) -> bool:
        return self._current_job is not None

    def start(self) -> None:
        """Begin / continue running the queue. No-op if already running
        or queue is paused."""
        if self.runner.is_running():
            return
        if self.queue.paused:
            return
        self._launch_next_job()

    def pause(self) -> None:
        self.queue.set_paused(True)
        self.queue_state_changed.emit()

    def resume(self) -> None:
        self.queue.set_paused(False)
        self.queue_state_changed.emit()
        self.start()

    def stop_current(self) -> None:
        """Kill the running subprocess. Marks the current job as
        interrupted and stops the queue (no auto-advance)."""
        if not self.runner.is_running():
            return
        self._stopping = True
        proc = self.runner._proc
        if proc is not None:
            proc.kill()

    # ----- Internal stage machinery -----------------------------------

    def _launch_next_job(self) -> None:
        if self.queue.paused:
            return
        job = self.queue.next_runnable()
        if job is None:
            self.queue_idle.emit()
            return
        self._current_job = job
        job.status = STATUS_RUNNING
        # Find first not-done stage (resumes mid-job if any were already
        # marked done -- e.g. a retry after a render failure preserves
        # the earlier concat).
        for i, s in enumerate(job.stages):
            if s.status != STAGE_DONE:
                job.current_stage_idx = i
                break
        self.queue.save()
        self._open_log_file(job)
        self.job_started.emit(job.id)
        self._launch_next_stage()

    def _launch_next_stage(self) -> None:
        job = self._current_job
        if job is None:
            return
        # Skip past any DONE stages
        while (job.current_stage_idx < len(job.stages)
               and job.stages[job.current_stage_idx].status == STAGE_DONE):
            job.current_stage_idx += 1
        if job.current_stage_idx >= len(job.stages):
            self._finalize_job(success=True)
            return
        # Pause-at-boundary check (before launching, after at least one
        # stage has completed since pause was requested).
        if self.queue.paused:
            job.status = STATUS_PENDING
            self.queue.save()
            self._current_job = None
            self._current_stage_name = None
            self._close_log_file()
            return

        stage = job.stages[job.current_stage_idx]
        # ffmpeg-based stages (concat, audio_mux) use imageio_ffmpeg's
        # bundled binary -- matches what ConcatTab does, so we don't
        # require ffmpeg to be on PATH. Falls back to bare "ffmpeg" if
        # the imageio_ffmpeg lookup fails for any reason.
        ffmpeg_bin = "ffmpeg"
        if stage.name in ("concat", "audio_mux"):
            try:
                import imageio_ffmpeg
                ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
            except Exception:
                pass
        try:
            args, cleanup = stage_command(
                job, stage.name, ffmpeg_bin=ffmpeg_bin)
        except Exception as e:  # noqa: BLE001
            stage.status = STAGE_FAILED
            stage.error = f"command build failed: {e}"
            stage.finished_at = time.time()
            self.queue.save()
            self.stage_finished.emit(job.id, stage.name, False)
            self._finalize_job(success=False)
            return

        # Run each stage with cwd = artefact dir so _progress.json and
        # any other transient files land in the per-job subdir rather
        # than next to the final broadcast.
        cwd_path = artefact_dir(job)
        cwd_path.mkdir(parents=True, exist_ok=True)
        cwd = str(cwd_path)
        self._progress_dir = cwd_path
        try:
            (self._progress_dir / "_progress.json").unlink(missing_ok=True)
        except Exception:
            pass

        self._current_stage_name = stage.name
        self._current_cleanup_paths = list(cleanup)
        stage.status = STAGE_RUNNING
        stage.started_at = time.time()
        stage.finished_at = None
        stage.exit_code = None
        stage.error = None
        self.queue.save()
        self.stage_started.emit(job.id, stage.name)

        # ffmpeg stages launch the binary directly; everything else is
        # `python -m waruka <subcommand>` via StepRunner's default.
        if stage.name in ("concat", "audio_mux"):
            program = args[0]
            args_rest = args[1:]
            self._log_to_file(
                f"\n$ {program} {' '.join(args_rest)}\n")
            self.runner.start(args_rest, cwd=cwd, program=program)
        else:
            args_rest = args[1:]  # drop the python_bin (StepRunner adds it)
            self._log_to_file(
                f"\n$ python {' '.join(args_rest)}\n")
            self.runner.start(args_rest, cwd=cwd)

        self._poll.start()

    def _on_stage_exit(self, exit_code: int) -> None:
        self._poll.stop()
        job = self._current_job
        if job is None:
            return
        for p in self._current_cleanup_paths:
            try:
                Path(p).unlink(missing_ok=True)
            except Exception:
                pass
        self._current_cleanup_paths = []

        stage = job.stage(self._current_stage_name or "")
        if stage is None:
            self._current_job = None
            return
        stage.finished_at = time.time()
        stage.exit_code = int(exit_code)

        was_killed = self._stopping
        if was_killed:
            self._stopping = False
            stage.status = STAGE_FAILED
            stage.error = "Stopped by user."
            job.error = "Stopped by user."
            job.status = STATUS_INTERRUPTED
            self.queue.save()
            self.stage_finished.emit(job.id, stage.name, False)
            self.job_finished.emit(job.id, False)
            self._current_job = None
            self._current_stage_name = None
            self._close_log_file()
            # Stay stopped (don't auto-advance after a manual stop).
            self.queue_idle.emit()
            return

        if exit_code == 0:
            stage.status = STAGE_DONE
            # Post-stage hook: when audio_mux succeeds, the latest
            # silent intermediate (silent_interp if interpolate ran,
            # else silent_render) is no longer needed for the broadcast
            # itself. If the user asked for a silent-copy companion
            # (`create_no_audio_copy`), rename it to
            # <broadcast>_no_audio.mp4; otherwise delete it. Same
            # behaviour matches the Track tab.
            #
            # The OTHER silent file (silent_render when interp ran)
            # is now scratch -- delete it unconditionally.
            if stage.name == "audio_mux":
                from .jobqueue import (
                    silent_render_path as _silent_render,
                    silent_interp_path as _silent_interp,
                    audio_mux_video_input as _final_silent,
                )
                final_silent = Path(_final_silent(job))
                if final_silent.exists():
                    tp = job.tracking_params or {}
                    if tp.get("create_no_audio_copy"):
                        bp = Path(job.broadcast_output_path)
                        no_audio = bp.with_name(
                            f"{bp.stem}_no_audio{bp.suffix}")
                        try:
                            if no_audio.exists():
                                no_audio.unlink()
                            final_silent.rename(no_audio)
                        except Exception as e:  # noqa: BLE001
                            self._log_to_file(
                                f"[warn] couldn't keep silent copy "
                                f"at {no_audio}: {e}\n")
                    else:
                        try:
                            final_silent.unlink(missing_ok=True)
                        except Exception:
                            pass
                # Drop the OTHER silent intermediate (which is now
                # always scratch -- if interp ran, silent_render became
                # an in-between buffer; if it didn't, silent_render IS
                # the final_silent we already handled above).
                other = Path(_silent_render(job))
                if str(other) != str(final_silent) and other.exists():
                    try:
                        other.unlink(missing_ok=True)
                    except Exception:
                        pass
            # Post-concat hook: if the user asked for a silent companion of
            # the concat'd pano, produce it now via a quick ffmpeg `-an`
            # stream-copy beside the source. Gated on the dedicated
            # concat_no_audio_copy flag and on audio actually being kept.
            if (stage.name == "concat"
                    and getattr(job, "concat_no_audio_copy", False)
                    and _job_keeps_audio(job)
                    and len(job.concat_files) > 1):
                self._produce_concat_no_audio_companion(job)
            self.queue.save()
            self.stage_finished.emit(job.id, stage.name, True)
            job.current_stage_idx += 1
            self._current_stage_name = None
            self._launch_next_stage()
        else:
            stage.status = STAGE_FAILED
            stage.error = f"Exit code {exit_code}"
            job.error = (f"Stage '{stage.name}' failed "
                          f"(exit {exit_code}).")
            self.queue.save()
            self.stage_finished.emit(job.id, stage.name, False)
            self._finalize_job(success=False)

    def _produce_concat_no_audio_companion(self, job: Job) -> None:
        """After concat: emit a silent companion `<base>_no_audio.mp4`
        beside the with-audio concat output. Stream-copy with `-an`
        so it's near-instant.
        """
        try:
            import imageio_ffmpeg
            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception as e:  # noqa: BLE001
            self._log_to_file(
                f"[warn] no-audio concat companion skipped (ffmpeg "
                f"unavailable: {e})\n")
            return
        concat_path = Path(job.concat_output_path)
        if not concat_path.exists():
            return
        out_path = concat_path.with_name(
            f"{concat_path.stem}_no_audio{concat_path.suffix}")
        cmd = [ffmpeg, "-y", "-loglevel", "error",
                "-i", str(concat_path), "-an",
                "-c:v", "copy", str(out_path)]
        try:
            subprocess.run(cmd, check=True, **_NO_WINDOW_KW)
        except Exception as e:  # noqa: BLE001
            self._log_to_file(
                f"[warn] failed to produce {out_path.name}: {e}\n")

    def _finalize_job(self, success: bool) -> None:
        job = self._current_job
        if job is None:
            return
        if success:
            job.status = STATUS_DONE
            # Note: concat_output_path is now a FINAL artefact (sits
            # beside the source as <basename>.mp4 in the v1.x layout),
            # so we do NOT delete it on success. The `keep_intermediates`
            # flag is reserved for any genuinely-scratch outputs the
            # job left behind.
        else:
            job.status = STATUS_FAILED
        self.queue.save()
        self.job_finished.emit(job.id, success)
        self._current_job = None
        self._current_stage_name = None
        self._close_log_file()
        # Move on to next pending job. Failed jobs auto-skip; user can
        # retry from the UI. Honour pause if it was set.
        if not self.queue.paused:
            self._launch_next_job()

    # ----- Progress polling -------------------------------------------

    def _poll_progress(self) -> None:
        if (self._progress_dir is None or self._current_job is None
                or self._current_stage_name is None):
            return
        path = self._progress_dir / "_progress.json"
        if not path.exists():
            return
        try:
            with open(path) as f:
                p = json.load(f)
        except Exception:
            return
        frac_raw = p.get("step_progress")
        try:
            frac = float(frac_raw) if frac_raw is not None else -1.0
        except (TypeError, ValueError):
            frac = -1.0
        extras = {
            "step": p.get("step", ""),
            "detail": p.get("step_detail", ""),
            "elapsed_s": p.get("elapsed_s"),
            "eta_s": p.get("eta_s"),
            "fps": p.get("fps_observed"),
        }
        self.stage_progress.emit(
            self._current_job.id, self._current_stage_name, frac, extras)

    # ----- Per-job log archive ----------------------------------------

    def _open_log_file(self, job: Job) -> None:
        self._close_log_file()
        # Log lives next to the per-job intermediates under the artefact
        # dir. Fall back to ~/.waruka/logs/<job_id>.log if something goes
        # wrong creating the artefact dir (very rare; permission issue
        # on the broadcast-output drive).
        try:
            ad = artefact_dir(job)
            ad.mkdir(parents=True, exist_ok=True)
            log_path = ad / "job.log"
        except Exception:
            _QUEUE_LOG_DIR.mkdir(parents=True, exist_ok=True)
            log_path = _QUEUE_LOG_DIR / f"{job.id}.log"
        self._current_log_path = log_path
        try:
            self._log_file = open(
                log_path, "a", encoding="utf-8", buffering=1)
            self._log_file.write(
                f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} "
                f"-- {job.name} (id {job.id}) ===\n")
        except Exception:
            self._log_file = None

    def _close_log_file(self) -> None:
        if self._log_file is not None:
            try:
                self._log_file.close()
            except Exception:
                pass
            self._log_file = None

    def _log_to_file(self, line: str) -> None:
        if self._log_file is not None:
            try:
                self._log_file.write(line)
            except Exception:
                pass

    def _on_log_line(self, line: str) -> None:
        job = self._current_job
        if job is None:
            return
        self.log_line.emit(job.id, line)
        self._log_to_file(line + "\n")


# --------------------------------------------------------------------------
# Add-job dialog
# --------------------------------------------------------------------------

class FilePathLineEdit(QtWidgets.QLineEdit):
    """QLineEdit that accepts a single file drop. Drops set the text
    to the dropped file's local path. Multiple files use the first
    one; folders are ignored. Used for the project / output fields
    in the AddJobDialog so the user can drag files straight from
    Explorer instead of clicking Browse.
    """

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAcceptDrops(True)

    def dragEnterEvent(self, ev: QtGui.QDragEnterEvent) -> None:
        if ev.mimeData().hasUrls() and any(
                u.isLocalFile() for u in ev.mimeData().urls()):
            ev.acceptProposedAction()
            return
        super().dragEnterEvent(ev)

    def dragMoveEvent(self, ev: QtGui.QDragMoveEvent) -> None:
        if ev.mimeData().hasUrls():
            ev.acceptProposedAction()
            return
        super().dragMoveEvent(ev)

    def dropEvent(self, ev: QtGui.QDropEvent) -> None:
        md = ev.mimeData()
        if md.hasUrls():
            for u in md.urls():
                if u.isLocalFile():
                    self.setText(u.toLocalFile())
                    # Fire textEdited so listeners (auto-derive,
                    # status refresh) react as if typed.
                    self.textEdited.emit(self.text())
                    ev.acceptProposedAction()
                    return
        super().dropEvent(ev)


def _project_is_calibrated(project_path: Path) -> bool:
    """Calibrate is 'done' once project.json exists -- the calibrate UI is
    the only thing that creates it."""
    return project_path.exists()


def _project_is_markfielded(project_path: Path) -> bool:
    """Markfield is 'done' once project.json has a homography + corners."""
    if not project_path.exists():
        return False
    try:
        from .config import ProjectConfig
        cfg = ProjectConfig.load(project_path)
    except Exception:
        return False
    return (cfg.homography is not None
            and len(cfg.field_marks.get("corners", [])) >= 4)


class AddJobDialog(QtWidgets.QDialog):
    """Form for creating or editing a Job, mirroring the Track tab's
    step flow:

      1. Pick input chunks (will be concatenated).
      2. Pick output broadcast path; project.json path auto-derives
         alongside it.
      3. Run calibrate (first chunk + project path) -> sets dewarp.
      4. Run markfield -> sets homography + field corners.
      5. Optional: enable interpolate + tweak its settings.
      6. Add to queue.

    Calibrate / Markfield launch subprocesses via the dialog's own
    StepRunner; the dialog stays open while they're running. The CLI
    windows handle the interactive work. When they finish the status
    pips refresh from the project.json on disk.

    'Reuse project from previous job' jumps straight past steps 3-4 by
    pointing at the most-recently-added job's project file -- intended
    for back-to-back games on the same field/mount.

    On accept, the assembled Job lands at ``self.result_job``.
    """

    def __init__(self, queue: JobQueue,
                 existing: Job | None = None,
                 parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.queue = queue
        self.existing = existing
        self.result_job: Job | None = None
        self.setWindowTitle("Add job" if existing is None else "Edit job")
        self.setMinimumWidth(640)

        # Subprocess runner for inline calibrate / markfield. Mirrors the
        # TrackTab pattern. Only one step runs at a time.
        self._runner = StepRunner(self)
        self._runner.finished.connect(self._on_step_finished)
        self._current_step: str | None = None
        # When user manually edits the project path, stop auto-deriving
        # it from the output path. Same convention PostProcessTab uses
        # for its output-path auto-suggest.
        self._project_user_edited = False

        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(14, 14, 14, 14)
        outer.setSpacing(10)

        # --- Name ---
        name_row = QtWidgets.QFormLayout()
        self.name_edit = QtWidgets.QLineEdit()
        self.name_edit.setPlaceholderText(
            "auto-derived from output filename if blank")
        name_row.addRow("Job name:", self.name_edit)
        outer.addLayout(name_row)

        # --- Input chunks ---
        chunks_box = QtWidgets.QGroupBox(
            "1. Input chunks (concatenated in this order -- "
            "drag-drop files in)")
        cb = QtWidgets.QVBoxLayout(chunks_box)
        # Reuse the Concat tab's drag-drop-aware list widget so files
        # can be dropped straight in from Explorer; internal reorder
        # via drag is also supported.
        self.chunks_list = ClipListWidget()
        self.chunks_list.setMaximumHeight(120)
        self.chunks_list.files_dropped.connect(self._on_chunks_dropped)
        cb.addWidget(self.chunks_list)
        chunk_btns = QtWidgets.QHBoxLayout()
        add_chunks_btn = QtWidgets.QPushButton("Add files...")
        add_chunks_btn.clicked.connect(self._on_add_chunks)
        chunk_btns.addWidget(add_chunks_btn)
        for label, slot in [("↑ Up", lambda: self._move_chunks(-1)),
                              ("↓ Down", lambda: self._move_chunks(+1)),
                              ("Remove", self._on_remove_chunks)]:
            b = QtWidgets.QPushButton(label)
            b.clicked.connect(slot)
            chunk_btns.addWidget(b)
        chunk_btns.addStretch(1)
        cb.addLayout(chunk_btns)
        outer.addWidget(chunks_box)

        # --- Output ---
        # The user names the CONCAT file; the broadcast (tracked) output
        # is that name + "_broadcast". Folder and filename are split so
        # only the name needs typing (the folder defaults to the source
        # clips' folder). Naming the concat directly -- rather than
        # deriving it by stripping "_broadcast" off the broadcast name --
        # is what makes the data-loss collision impossible.
        out_box = QtWidgets.QGroupBox("2. Output (concat + broadcast)")
        ob = QtWidgets.QVBoxLayout(out_box)
        dir_row = QtWidgets.QHBoxLayout()
        dir_row.addWidget(QtWidgets.QLabel("Folder:"))
        self.dir_edit = FilePathLineEdit()
        self.dir_edit.setPlaceholderText(
            "output folder (defaults to the first clip's folder)")
        dir_row.addWidget(self.dir_edit, 1)
        dir_browse = QtWidgets.QPushButton("Browse...")
        dir_browse.clicked.connect(self._on_browse_dir)
        dir_row.addWidget(dir_browse)
        ob.addLayout(dir_row)

        cname_row = QtWidgets.QHBoxLayout()
        cname_row.addWidget(QtWidgets.QLabel("Concat name:"))
        self.concat_name_edit = QtWidgets.QLineEdit()
        self.concat_name_edit.setPlaceholderText("e.g. 20260614 Curve")
        cname_row.addWidget(self.concat_name_edit, 1)
        cname_row.addWidget(QtWidgets.QLabel(".mp4"))
        ob.addLayout(cname_row)

        # Audio choices for the concat step (parity with the Concat tab).
        self.preserve_audio_check = QtWidgets.QCheckBox(
            "Preserve audio in concatenated output")
        self.preserve_audio_check.setChecked(True)
        self.preserve_audio_check.toggled.connect(
            self._on_preserve_audio_toggled)
        ob.addWidget(self.preserve_audio_check)
        self.concat_no_audio_check = QtWidgets.QCheckBox(
            "Also write a silent copy of the concat (<name>_no_audio.mp4)")
        self.concat_no_audio_check.setChecked(False)
        self.concat_no_audio_check.toggled.connect(
            lambda *_: self._update_paths_preview())
        ob.addWidget(self.concat_no_audio_check)

        self.paths_preview = QtWidgets.QLabel("")
        self.paths_preview.setStyleSheet(
            "font-family: Consolas, monospace; color: #555; font-size: 10px;")
        self.paths_preview.setWordWrap(True)
        ob.addWidget(self.paths_preview)
        outer.addWidget(out_box)

        # --- Setup (calibrate + markfield) ---
        setup_box = QtWidgets.QGroupBox(
            "3. Setup -- calibrate + mark field")
        sb = QtWidgets.QVBoxLayout(setup_box)
        proj_row = QtWidgets.QHBoxLayout()
        proj_row.addWidget(QtWidgets.QLabel("Project file:"))
        self.project_edit = FilePathLineEdit()
        self.project_edit.setPlaceholderText(
            "auto-derived from output (e.g. game1.project.json) "
            "-- or drag one here")
        proj_row.addWidget(self.project_edit, 1)
        proj_browse = QtWidgets.QPushButton("Browse...")
        proj_browse.clicked.connect(self._on_browse_project)
        proj_row.addWidget(proj_browse)
        sb.addLayout(proj_row)

        # Calibrate row
        cal_row = QtWidgets.QHBoxLayout()
        self.calibrate_status = QtWidgets.QLabel("✗ Calibrate")
        self.calibrate_status.setMinimumWidth(180)
        self.calibrate_status.setStyleSheet(
            "font-family: Consolas, monospace;")
        cal_row.addWidget(self.calibrate_status)
        self.calibrate_btn = QtWidgets.QPushButton("Run calibrate")
        self.calibrate_btn.clicked.connect(
            lambda: self._launch_step("calibrate"))
        cal_row.addWidget(self.calibrate_btn)
        cal_row.addStretch(1)
        sb.addLayout(cal_row)

        # Markfield row
        mf_row = QtWidgets.QHBoxLayout()
        self.markfield_status = QtWidgets.QLabel("✗ Mark field")
        self.markfield_status.setMinimumWidth(180)
        self.markfield_status.setStyleSheet(
            "font-family: Consolas, monospace;")
        mf_row.addWidget(self.markfield_status)
        self.markfield_btn = QtWidgets.QPushButton("Run markfield")
        self.markfield_btn.clicked.connect(
            lambda: self._launch_step("markfield"))
        mf_row.addWidget(self.markfield_btn)
        mf_row.addStretch(1)
        sb.addLayout(mf_row)

        # Reuse + hint
        reuse_row = QtWidgets.QHBoxLayout()
        self.reuse_btn = QtWidgets.QPushButton(
            "Reuse project from previous job")
        self.reuse_btn.clicked.connect(self._on_reuse_project)
        reuse_row.addWidget(self.reuse_btn)
        reuse_row.addStretch(1)
        sb.addLayout(reuse_row)
        setup_hint = QtWidgets.QLabel(
            "Calibrate + markfield open interactive OpenCV windows on "
            "the first input chunk. 'Reuse' points at the previous job's "
            "project file -- use it for back-to-back games on the same "
            "field/mount where the camera hasn't moved.")
        setup_hint.setStyleSheet("color: #666; font-size: 11px;")
        setup_hint.setWordWrap(True)
        sb.addWidget(setup_hint)
        outer.addWidget(setup_box)

        # --- Tracking parameters ---
        # Reuse the Track tab's ParamsDialog so the queue stays in parity
        # with the existing tracking flow: stride, mode (sequential vs
        # pipeline), t0/t1, view mode, SR toggle, audio companion file,
        # and interpolation pre-selection all live there.
        tp_box = QtWidgets.QGroupBox("4. Tracking parameters")
        tpb = QtWidgets.QVBoxLayout(tp_box)
        tp_row = QtWidgets.QHBoxLayout()
        self.tp_btn = QtWidgets.QPushButton("Open tracking parameters...")
        self.tp_btn.clicked.connect(self._on_open_track_params)
        tp_row.addWidget(self.tp_btn)
        tp_row.addStretch(1)
        tpb.addLayout(tp_row)
        self.tp_summary = QtWidgets.QLabel("Using defaults.")
        self.tp_summary.setStyleSheet(
            "color: #555; font-family: Consolas, monospace; "
            "font-size: 11px;")
        self.tp_summary.setWordWrap(True)
        tpb.addWidget(self.tp_summary)
        outer.addWidget(tp_box)

        # In-dialog state for tracking parameters (dict form mirroring
        # ProcessingParams). Updated by _on_open_track_params; consumed
        # by _on_accept.
        self._tracking_params: dict = {}
        self._has_audio: bool = False  # set when params dialog probes

        # --- Interpolation (optional) ---
        self.interp_box = QtWidgets.QGroupBox(
            "5. Frame interpolation (optional)")
        self.interp_box.setCheckable(True)
        self.interp_box.setChecked(False)
        ibl = QtWidgets.QFormLayout(self.interp_box)
        self.interp_fps = QtWidgets.QComboBox()
        self.interp_fps.addItem("40 fps (2x)", 40)
        self.interp_fps.addItem("60 fps (3x, recommended)", 60)
        self.interp_fps.addItem("80 fps (4x)", 80)
        self.interp_fps.setCurrentIndex(1)
        ibl.addRow("Target fps:", self.interp_fps)
        self.interp_backend = QtWidgets.QComboBox()
        self.interp_backend.addItem("RIFE 4.25 (fast)", "rife")
        self.interp_backend.addItem("FILM-Style (very slow!)", "film")
        ibl.addRow("Backend:", self.interp_backend)
        self.interp_cq = QtWidgets.QSpinBox()
        self.interp_cq.setRange(0, 51)
        self.interp_cq.setValue(23)
        ibl.addRow("NVENC CQ (lower = sharper, bigger):",
                   self.interp_cq)
        outer.addWidget(self.interp_box)

        # --- OK / Cancel ---
        btns = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok |
            QtWidgets.QDialogButtonBox.Cancel)
        self._ok_btn = btns.button(QtWidgets.QDialogButtonBox.Ok)
        self._ok_btn.setText(
            "Save" if existing is not None else "Add to queue")
        btns.accepted.connect(self._on_accept)
        btns.rejected.connect(self.reject)
        outer.addWidget(btns)

        # --- Wire up reactivity ---
        self.chunks_list.model().rowsInserted.connect(
            lambda *_: self._on_anything_changed())
        self.chunks_list.model().rowsRemoved.connect(
            lambda *_: self._on_anything_changed())
        self.dir_edit.textEdited.connect(self._on_naming_changed)
        self.concat_name_edit.textEdited.connect(self._on_naming_changed)
        self.project_edit.textEdited.connect(self._on_project_edited)

        if existing is not None:
            self._load_from(existing)
        self._refresh_status()
        self._update_paths_preview()

    # ----- chunk list helpers -----------------------------------------

    def _on_add_chunks(self) -> None:
        files, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, "Add input chunks", "",
            "Videos (*.mp4 *.mov *.mkv *.avi *.m4v);;All files (*)")
        for f in files:
            self.chunks_list.addItem(f)

    def _on_chunks_dropped(self, paths: list[str]) -> None:
        """Handle external file drop on the chunks list. Filters to
        plausible video extensions; appends in drop order."""
        video_exts = {".mp4", ".mov", ".mkv", ".avi", ".m4v"}
        for p in paths:
            if Path(p).suffix.lower() in video_exts and Path(p).is_file():
                self.chunks_list.addItem(p)

    def _move_chunks(self, direction: int) -> None:
        rows = sorted(
            [self.chunks_list.row(i) for i in self.chunks_list.selectedItems()],
            reverse=(direction > 0))
        for r in rows:
            nr = r + direction
            if 0 <= nr < self.chunks_list.count():
                item = self.chunks_list.takeItem(r)
                self.chunks_list.insertItem(nr, item)
                self.chunks_list.setCurrentRow(nr)

    def _on_remove_chunks(self) -> None:
        for i in self.chunks_list.selectedItems():
            self.chunks_list.takeItem(self.chunks_list.row(i))

    # ----- Output / project path -------------------------------------

    # ----- Naming helpers (concat name -> derived artefact paths) -----

    def _out_dir(self) -> str:
        return self.dir_edit.text().strip()

    def _concat_base(self) -> str:
        """User's concat filename, sans any .mp4 suffix they typed."""
        b = self.concat_name_edit.text().strip()
        if b.lower().endswith(".mp4"):
            b = b[:-4]
        return b.strip()

    def _concat_path(self) -> str:
        d, b = self._out_dir(), self._concat_base()
        return str(Path(d) / f"{b}.mp4") if d and b else ""

    def _broadcast_path(self) -> str:
        d, b = self._out_dir(), self._concat_base()
        return str(Path(d) / f"{b}_broadcast.mp4") if d and b else ""

    def _concat_no_audio_path(self) -> str:
        d, b = self._out_dir(), self._concat_base()
        return str(Path(d) / f"{b}_no_audio.mp4") if d and b else ""

    def _broadcast_no_audio_path(self) -> str:
        d, b = self._out_dir(), self._concat_base()
        return str(Path(d) / f"{b}_broadcast_no_audio.mp4") if d and b else ""

    def _on_naming_changed(self, *_args) -> None:
        if not self._project_user_edited:
            # Auto-derive project inside the artefact subdir, keyed off the
            # BROADCAST stem so it matches jobqueue.artefact_dir
            # (<dir>/waruka_tracking/<base>_broadcast/).
            bp = self._broadcast_path()
            if bp:
                op = Path(bp)
                proj = str(op.parent / "waruka_tracking" / op.stem
                            / "project.json")
                # Block our own signal handler so this doesn't flip the
                # "user edited" flag.
                self.project_edit.blockSignals(True)
                self.project_edit.setText(proj)
                self.project_edit.blockSignals(False)
        self._update_paths_preview()
        self._refresh_status()

    def _on_preserve_audio_toggled(self, on: bool) -> None:
        # The silent-concat-copy option is only meaningful when audio is
        # being preserved in the first place.
        self.concat_no_audio_check.setEnabled(on)
        self._update_paths_preview()

    def _update_paths_preview(self) -> None:
        d, b = self._out_dir(), self._concat_base()
        if not (d and b):
            self.paths_preview.setText(
                "Set a folder and concat name to see the output files.")
            return
        keeps_audio = self.preserve_audio_check.isChecked()
        lines = []
        if self.chunks_list.count() > 1:
            lines.append(f"concat:     {b}.mp4"
                          + ("" if keeps_audio else "   (silent)"))
            if keeps_audio and self.concat_no_audio_check.isChecked():
                lines.append(f"            {b}_no_audio.mp4")
        lines.append(f"broadcast:  {b}_broadcast.mp4")
        if keeps_audio and (self._tracking_params or {}).get(
                "create_no_audio_copy"):
            lines.append(f"            {b}_broadcast_no_audio.mp4")
        self.paths_preview.setText("\n".join(lines))

    def _on_project_edited(self, _text: str) -> None:
        self._project_user_edited = True
        self._refresh_status()

    def _on_browse_dir(self) -> None:
        start = self._out_dir() or ""
        d = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Output folder", start)
        if d:
            self.dir_edit.setText(d)
            self._on_naming_changed()

    def _on_browse_project(self) -> None:
        start = self.project_edit.text().strip() or ""
        f, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Pick project.json", start, "Project JSON (*.json)")
        if f:
            self.project_edit.setText(f)
            self._project_user_edited = True
            self._refresh_status()

    def _on_reuse_project(self) -> None:
        if not self.queue.jobs:
            QtWidgets.QMessageBox.information(
                self, "Waruka",
                "No previous jobs in the queue to reuse a project from.")
            return
        latest = max(self.queue.jobs, key=lambda j: j.added_at)
        self.project_edit.setText(latest.project_path)
        self._project_user_edited = True
        self._refresh_status()

    # ----- Reactivity hook -------------------------------------------

    def _on_anything_changed(self) -> None:
        # Auto-fill folder + concat name from the first clip when blank.
        if self.chunks_list.count() > 0:
            first = self.chunks_list.item(0).text()
            if first:
                if not self.dir_edit.text().strip():
                    self.dir_edit.setText(str(Path(first).parent))
                if not self.concat_name_edit.text().strip():
                    # Prefill "YYYYMMDD " from the clip's recording date as
                    # a typing prompt (mirrors the Concat tab); the user
                    # types the team names after it.
                    prefix = ""
                    try:
                        dt = _extract_clip_datetime(Path(first))
                        if dt:
                            prefix = dt.split(" ", 1)[0].replace("-", "") + " "
                    except Exception:
                        prefix = ""
                    if prefix:
                        self.concat_name_edit.setText(prefix)
                self._on_naming_changed()
        self._refresh_status()

    # ----- Inline calibrate / markfield ------------------------------

    def _launch_step(self, step: str) -> None:
        """Run `waruka calibrate` or `waruka markfield` on the first input
        chunk + the current project path. Subprocess opens its own
        OpenCV window; the dialog stays open."""
        if self._runner.is_running():
            QtWidgets.QMessageBox.information(
                self, "Waruka",
                f"Currently running '{self._current_step}'. Wait for it "
                "to finish first.")
            return
        if self.chunks_list.count() == 0:
            QtWidgets.QMessageBox.warning(
                self, "Waruka",
                "Add at least one input chunk first; the first chunk is "
                "used as the calibration source.")
            return
        first_chunk = self.chunks_list.item(0).text()
        if not Path(first_chunk).exists():
            QtWidgets.QMessageBox.warning(
                self, "Waruka",
                f"First input chunk does not exist:\n{first_chunk}")
            return
        proj = self.project_edit.text().strip()
        if not proj:
            QtWidgets.QMessageBox.warning(
                self, "Waruka",
                "Set the output folder and concat name first; the project "
                "file path auto-derives from them.")
            return
        # Ensure project directory exists (calibrate writes the file there).
        Path(proj).parent.mkdir(parents=True, exist_ok=True)
        # cwd: the project's directory. Calibrate / markfield write
        # transient files (last_scrub_t cache, etc.) there.
        cwd = str(Path(proj).parent)

        args = ["-m", "waruka", step, first_chunk, "--project", proj]
        self._current_step = step
        self._set_step_running(step)
        self._runner.start(args, cwd=cwd)

    def _on_step_finished(self, exit_code: int) -> None:
        step = self._current_step
        self._current_step = None
        if step is None:
            return
        if exit_code != 0:
            log = self._runner.log() or "(no output captured)"
            QtWidgets.QMessageBox.warning(
                self,
                f"{step} failed (exit {exit_code})",
                f"Subprocess exited with code {exit_code}.\n\n"
                f"Last output:\n\n{log[-2000:]}")
        self._refresh_status()

    def _set_step_running(self, step: str) -> None:
        label = self.calibrate_status if step == "calibrate" \
            else self.markfield_status
        verb = "Calibrate" if step == "calibrate" else "Mark field"
        label.setText(f"... {verb} running")
        label.setStyleSheet(
            "font-family: Consolas, monospace; color: #187;")
        self._refresh_buttons()

    # ----- Status refresh --------------------------------------------

    def _refresh_status(self) -> None:
        """Update calibrate/markfield status pips + Add-to-queue button
        enablement from the current project.json's on-disk state."""
        # Don't overwrite the "Running" labels mid-step
        if self._runner.is_running():
            self._refresh_buttons()
            return
        proj = self.project_edit.text().strip()
        proj_path = Path(proj) if proj else None
        cal_done = bool(proj_path) and _project_is_calibrated(proj_path)
        mf_done = bool(proj_path) and _project_is_markfielded(proj_path)
        self._set_status_label(self.calibrate_status, "Calibrate", cal_done)
        self._set_status_label(self.markfield_status, "Mark field", mf_done)
        self._refresh_buttons()

    def _set_status_label(self, label: QtWidgets.QLabel,
                            verb: str, done: bool) -> None:
        if done:
            label.setText(f"✓ {verb}")
            label.setStyleSheet(
                "font-family: Consolas, monospace; color: #080;")
        else:
            label.setText(f"✗ {verb}")
            label.setStyleSheet(
                "font-family: Consolas, monospace; color: #888;")

    def _refresh_buttons(self) -> None:
        running = self._runner.is_running()
        has_chunks = self.chunks_list.count() > 0
        proj = self.project_edit.text().strip()
        has_out = bool(self._broadcast_path())
        proj_path = Path(proj) if proj else None
        cal_done = bool(proj_path) and _project_is_calibrated(proj_path)
        mf_done = bool(proj_path) and _project_is_markfielded(proj_path)

        self.calibrate_btn.setEnabled(
            not running and has_chunks and has_out)
        # Markfield requires calibrate to have produced a project.json
        self.markfield_btn.setEnabled(
            not running and has_chunks and has_out and cal_done)
        # Reuse-project disabled while running
        self.reuse_btn.setEnabled(not running)
        # Add-to-queue requires the full setup
        self._ok_btn.setEnabled(
            not running and has_chunks and has_out and mf_done)
        if self._ok_btn.isEnabled():
            self._ok_btn.setToolTip("")
        elif running:
            self._ok_btn.setToolTip("Subprocess is running.")
        elif not has_chunks:
            self._ok_btn.setToolTip("Add at least one input chunk.")
        elif not has_out:
            self._ok_btn.setToolTip(
                "Set the output folder and concat name.")
        elif not mf_done:
            self._ok_btn.setToolTip(
                "Calibrate + Mark field must be done before queueing "
                "(or pick an existing project).")

    # ----- Close-while-running guard ---------------------------------

    def reject(self) -> None:
        if self._runner.is_running():
            if QtWidgets.QMessageBox.question(
                self, "Subprocess still running",
                f"'{self._current_step}' is still running. Closing this "
                "dialog will kill it. Continue?",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                QtWidgets.QMessageBox.No,
            ) != QtWidgets.QMessageBox.Yes:
                return
            proc = self._runner._proc
            if proc is not None:
                proc.kill()
        super().reject()

    # ----- Tracking parameters dialog --------------------------------

    def _on_open_track_params(self) -> None:
        """Probe the first input chunk, open the Track tab's params
        dialog with sensible defaults, and capture the result on
        ``self._tracking_params``. Interp fps/backend round-trip with
        the in-dialog interp combos so there's a single source of
        truth for what eventually feeds the interpolate stage."""
        chunks = [self.chunks_list.item(i).text()
                   for i in range(self.chunks_list.count())]
        if not chunks:
            QtWidgets.QMessageBox.warning(
                self, "Waruka",
                "Add at least one input chunk first -- the params "
                "dialog needs the first chunk to probe duration + "
                "audio presence.")
            return
        first = chunks[0]
        if not Path(first).exists():
            QtWidgets.QMessageBox.warning(
                self, "Waruka", f"Missing input file:\n{first}")
            return
        out = self._broadcast_path()
        if not out:
            QtWidgets.QMessageBox.warning(
                self, "Waruka",
                "Set the output folder and concat name first -- the params "
                "dialog needs the broadcast path for its default output "
                "field.")
            return

        # Build a VideoInfo-like view: audio from first chunk, total
        # duration summed across all chunks so t0/t1 validator covers
        # the full concatenated timeline.
        info_first = VideoInfo.probe(first)
        if info_first is None:
            QtWidgets.QMessageBox.warning(
                self, "Waruka",
                f"Could not probe first input chunk:\n{first}")
            return
        total_dur = info_first.duration_s
        for c in chunks[1:]:
            ci = VideoInfo.probe(c)
            if ci is not None:
                total_dur += ci.duration_s
        info = VideoInfo(
            width=info_first.width, height=info_first.height,
            fps=info_first.fps,
            n_frames=int(round(total_dur * info_first.fps)),
            duration_s=total_dur,
            has_audio=info_first.has_audio,
        )

        # Synthesise a WarukaPaths whose output_video is the queue's
        # broadcast output (so the params dialog's Output field
        # pre-fills the right place).
        import dataclasses as _dc
        paths_first = WarukaPaths.for_video(first)
        paths = _dc.replace(paths_first, output_video=Path(out))

        # Pre-fill initial params from whatever's in self._tracking_params
        # plus the AddJobDialog's current interp combo state (so opening
        # the dialog without ever having set params still reflects what
        # the user picked in the interp box).
        existing = self._tracking_params
        if self.interp_box.isChecked():
            init_interp_fps = int(self.interp_fps.currentData())
            init_interp_backend = self.interp_backend.currentData()
        else:
            init_interp_fps = int(existing.get("interpolate_fps", 0))
            init_interp_backend = str(
                existing.get("interpolate_backend", "rife"))
        initial = ProcessingParams(
            t0=existing.get("t0"),
            t1=existing.get("t1"),
            mode=str(existing.get("mode", "sequential")),
            stride=int(existing.get("stride", 3)),
            view_mode=str(existing.get("view_mode", "default")),
            create_no_audio_copy=bool(existing.get(
                "create_no_audio_copy", info.has_audio)),
            output_path=out,
            interpolate_fps=init_interp_fps,
            interpolate_backend=init_interp_backend,
            sr_enabled=bool(existing.get("sr_enabled", False)),
        )

        dlg = ParamsDialog(paths, info, initial, parent=self)
        if dlg.exec() != QtWidgets.QDialog.Accepted or dlg.result_params is None:
            return
        p = dlg.result_params

        # Save structured params + audio flag
        self._tracking_params = {
            "t0": p.t0, "t1": p.t1,
            "mode": p.mode,
            "stride": p.stride,
            "view_mode": p.view_mode,
            "create_no_audio_copy": bool(p.create_no_audio_copy),
            "sr_enabled": bool(p.sr_enabled),
            "interpolate_fps": int(p.interpolate_fps),
            "interpolate_backend": p.interpolate_backend,
        }
        self._has_audio = bool(info.has_audio)

        # Sync interp settings back to the AddJobDialog's own combos.
        # The interp box stays the authoritative source for the
        # interpolate stage (so the CQ knob lives in one place).
        if p.interpolate_fps and p.interpolate_fps > 0:
            self.interp_box.setChecked(True)
            idx = self.interp_fps.findData(int(p.interpolate_fps))
            if idx >= 0:
                self.interp_fps.setCurrentIndex(idx)
            bidx = self.interp_backend.findData(p.interpolate_backend)
            if bidx >= 0:
                self.interp_backend.setCurrentIndex(bidx)
        else:
            # User picked "Off" in params dialog -> turn off the box
            self.interp_box.setChecked(False)

        # The broadcast path is governed by this dialog's concat-name
        # field (single source of truth), so we intentionally do NOT pull
        # an output-path edit back from the params dialog. Refresh the
        # preview, since the silent-broadcast-copy choice may have changed.
        self._update_paths_preview()

        self._refresh_track_params_summary()

    def _refresh_track_params_summary(self) -> None:
        tp = self._tracking_params
        if not tp:
            self.tp_summary.setText("Using defaults.")
            return
        bits = [
            f"mode={tp.get('mode', 'sequential')}",
            f"stride={tp.get('stride', 3)}",
            f"view={tp.get('view_mode', 'default')}",
        ]
        if tp.get("t0") is not None:
            bits.append(f"t0={tp['t0']:g}")
        if tp.get("t1") is not None:
            bits.append(f"t1={tp['t1']:g}")
        if tp.get("sr_enabled"):
            bits.append("SR on")
        if self._has_audio:
            if tp.get("create_no_audio_copy"):
                bits.append("also writes _no_audio.mp4")
            else:
                bits.append("audio kept")
        else:
            bits.append("no source audio")
        self.tp_summary.setText("  ".join(bits))

    # ----- Accept (build Job) ----------------------------------------

    def _on_accept(self) -> None:
        chunks = [self.chunks_list.item(i).text()
                   for i in range(self.chunks_list.count())]
        if not chunks:
            QtWidgets.QMessageBox.warning(
                self, "Waruka", "Add at least one input chunk.")
            return
        for c in chunks:
            if not Path(c).exists():
                QtWidgets.QMessageBox.warning(
                    self, "Waruka",
                    f"Missing input file:\n{c}")
                return
        project = self.project_edit.text().strip()
        if not project:
            QtWidgets.QMessageBox.warning(
                self, "Waruka",
                "Project file is empty. Set the output broadcast first.")
            return
        proj_path = Path(project)
        if not proj_path.exists():
            QtWidgets.QMessageBox.warning(
                self, "Waruka",
                f"Project file does not exist:\n{project}\n\n"
                "Run Calibrate + Mark field first.")
            return
        if not _project_is_markfielded(proj_path):
            QtWidgets.QMessageBox.warning(
                self, "Waruka",
                "Project has no field marks yet. Run Mark field before "
                "adding to the queue.")
            return
        out_dir = self._out_dir()
        base = self._concat_base()
        if not out_dir or not base:
            QtWidgets.QMessageBox.warning(
                self, "Waruka", "Set the output folder and concat name.")
            return
        out = self._broadcast_path()           # <dir>/<base>_broadcast.mp4
        name = self.name_edit.text().strip() or base
        # The concat output is a FINAL, user-named artefact next to the
        # source (<dir>/<base>.mp4). It is NEVER derived by stripping a
        # suffix off another name, so it cannot silently land on an input
        # clip. Single-input jobs don't concat; the lone file IS the source.
        concat_out = self._concat_path() if len(chunks) > 1 else chunks[0]

        # --- Output-safety validation (fail closed; never clobber data) ---
        # Build the list of artefacts this job will write.
        planned: list[tuple[str, str]] = []
        if len(chunks) > 1:
            planned.append(("concat", concat_out))
            if (self.preserve_audio_check.isChecked()
                    and self.concat_no_audio_check.isChecked()):
                planned.append(
                    ("concat silent copy", self._concat_no_audio_path()))
        planned.append(("broadcast", out))
        if (self._tracking_params or {}).get("create_no_audio_copy"):
            planned.append(
                ("broadcast silent copy", self._broadcast_no_audio_path()))
        # 1. Catastrophic: an output landing on an input clip would let
        #    `ffmpeg -y` truncate that source mid-concat. Always block.
        for label, pth in planned:
            try:
                assert_output_safe(pth, chunks)
            except ValueError:
                QtWidgets.QMessageBox.critical(
                    self, "Waruka -- unsafe output name",
                    f"The {label} output would overwrite one of your input "
                    f"clips, destroying source footage:\n\n{pth}\n\n"
                    "Choose a different concat name or folder.")
                return
        # 2. Existence: block a NEW name that already exists. A job being
        #    re-saved/retried may legitimately reuse its OWN prior outputs,
        #    so those paths are exempted.
        own = set()
        if self.existing is not None:
            for p in (self.existing.concat_output_path,
                      self.existing.broadcast_output_path):
                if not p:
                    continue
                own.add(_norm_path(p))
                # Also exempt the job's own _no_audio companions, which a
                # partially-run job may already have written.
                pp = Path(p)
                own.add(_norm_path(
                    pp.with_name(pp.stem + "_no_audio" + pp.suffix)))
        for label, pth in planned:
            if _norm_path(pth) in own:
                continue
            if Path(pth).exists():
                QtWidgets.QMessageBox.critical(
                    self, "Waruka -- output already exists",
                    f"The {label} output already exists:\n\n{pth}\n\n"
                    "Pick a different concat name or folder (or delete the "
                    "existing file first). Waruka will not overwrite it.")
                return

        # If the user hasn't opened the params dialog, we still want
        # audio_mux to fire when the source has audio. Probe the first
        # chunk now as a fallback (fast: just runs ffmpeg -i and greps
        # for an Audio: stream line).
        if not self._tracking_params and not self._has_audio:
            try:
                self._has_audio = _probe_audio(chunks[0])
            except Exception:
                self._has_audio = False

        # In v1.x and later, interpolate writes to broadcast_output_path
        # (after audio_mux runs on its output) rather than to a separate
        # `_smooth.mp4`. The final post-interpolate-with-audio file IS
        # the broadcast output. We keep `interp_path` set to None here;
        # the queue uses `tracking_params.interpolate_fps` to decide
        # whether to schedule the interpolate stage.
        interp_path: Optional[str] = None
        interp_args: dict = {}
        if self.interp_box.isChecked():
            interp_args = {
                "fps": int(self.interp_fps.currentData()),
                "backend": self.interp_backend.currentData(),
                "cq": int(self.interp_cq.value()),
            }

        job_id = self.existing.id if self.existing else uuid.uuid4().hex[:12]
        retry_count = self.existing.retry_count if self.existing else 0
        added_at = self.existing.added_at if self.existing else time.time()

        self.result_job = Job(
            id=job_id, name=name,
            concat_files=chunks,
            project_path=project,
            concat_output_path=concat_out,
            broadcast_output_path=out,
            interp_output_path=interp_path,
            pipeline_args={},
            interp_args=interp_args,
            tracking_params=dict(self._tracking_params),
            has_audio=bool(self._has_audio),
            preserve_concat_audio=self.preserve_audio_check.isChecked(),
            concat_no_audio_copy=self.concat_no_audio_check.isChecked(),
            keep_intermediates=False,
            status=STATUS_PENDING,
            stages=[],   # rebuilt on add
            current_stage_idx=0,
            retry_count=retry_count,
            error=None,
            added_at=added_at,
        )
        self.accept()

    def _load_from(self, job: Job) -> None:
        self.name_edit.setText(job.name)
        for c in job.concat_files:
            self.chunks_list.addItem(c)
        # Editing an existing job -- restore the folder + concat name from
        # the job's concat output (or, for single-input jobs, from the
        # broadcast name with the _broadcast suffix removed). The project
        # path was already set, so don't let the auto-derive clobber it.
        cp = job.concat_output_path
        if cp and len(job.concat_files) > 1:
            self.dir_edit.setText(str(Path(cp).parent))
            self.concat_name_edit.setText(Path(cp).stem)
        else:
            bp = Path(job.broadcast_output_path)
            self.dir_edit.setText(str(bp.parent))
            stem = bp.stem
            if stem.endswith("_broadcast"):
                stem = stem[: -len("_broadcast")]
            self.concat_name_edit.setText(stem)
        # Legacy persisted jobs (pre-this-feature) deserialise these as
        # None; treat None as the default (preserve audio, no extra copy).
        pv = getattr(job, "preserve_concat_audio", True)
        self.preserve_audio_check.setChecked(True if pv is None else bool(pv))
        self.concat_no_audio_check.setChecked(
            bool(getattr(job, "concat_no_audio_copy", False)))
        self.project_edit.setText(job.project_path)
        self._project_user_edited = True
        # Interpolate-on signal lives in any of three places depending
        # on when the job was persisted:
        #   - tracking_params['interpolate_fps']  (v1.x, structured form)
        #   - interp_args['fps']                  (legacy queue dialog)
        #   - interp_output_path truthy           (pre-v1.x persisted jobs)
        tp = job.tracking_params or {}
        ia = job.interp_args or {}
        interp_fps_persisted = (
            tp.get("interpolate_fps")
            or ia.get("fps")
            or (60 if job.interp_output_path else None))
        if interp_fps_persisted:
            self.interp_box.setChecked(True)
            idx = self.interp_fps.findData(int(interp_fps_persisted))
            if idx >= 0:
                self.interp_fps.setCurrentIndex(idx)
            bidx = self.interp_backend.findData(
                str(tp.get("interpolate_backend") or ia.get("backend", "rife")))
            if bidx >= 0:
                self.interp_backend.setCurrentIndex(bidx)
            self.interp_cq.setValue(
                int(job.interp_args.get("cq", 23)))
        # Tracking parameters captured on the original job
        self._tracking_params = dict(job.tracking_params or {})
        self._has_audio = bool(job.has_audio)
        self._refresh_track_params_summary()


# --------------------------------------------------------------------------
# Queue tab
# --------------------------------------------------------------------------

# Maps Job status -> (display label, prefix glyph, colour hex)
_JOB_STATUS_META = {
    STATUS_PENDING:     ("pending",     "  ", "#666"),
    STATUS_RUNNING:     ("running",     "▶ ", "#187"),
    STATUS_DONE:        ("done",        "✓ ", "#080"),
    STATUS_FAILED:      ("failed",      "✗ ", "#c00"),
    STATUS_INTERRUPTED: ("interrupted", "⚠ ", "#c70"),
}


class QueueTab(QtWidgets.QWidget):
    """Overnight batch queue UI (#35).

    Top: status header + global Start/Pause/Resume/Stop controls.
    Middle: jobs table + Add/Edit/Move/Remove/Retry buttons.
    Bottom: selected-job details and live log.
    """

    status_message = QtCore.Signal(str, int)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.queue = JobQueue()
        self.runner = QueueRunner(self.queue, self)
        self.runner.job_started.connect(self._on_job_started)
        self.runner.job_finished.connect(self._on_job_finished)
        self.runner.stage_started.connect(self._on_stage_started)
        self.runner.stage_finished.connect(self._on_stage_finished)
        self.runner.stage_progress.connect(self._on_stage_progress)
        self.runner.queue_idle.connect(self._on_queue_idle)
        self.runner.queue_state_changed.connect(self._refresh_status)
        self.runner.log_line.connect(self._on_log_line)
        self._latest_progress: dict = {}

        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(16, 16, 16, 16)
        outer.setSpacing(10)

        # --- Header (status + global controls) --------------------------
        header = QtWidgets.QHBoxLayout()
        self.status_label = QtWidgets.QLabel("Idle.")
        self.status_label.setStyleSheet(
            "font-weight: bold; font-size: 13px;")
        header.addWidget(self.status_label, 1)

        self.start_btn = QtWidgets.QPushButton("▶ Start queue")
        self.start_btn.clicked.connect(self._on_start)
        header.addWidget(self.start_btn)
        self.pause_btn = QtWidgets.QPushButton("⏸ Pause")
        self.pause_btn.clicked.connect(self._on_pause)
        header.addWidget(self.pause_btn)
        self.resume_btn = QtWidgets.QPushButton("▶ Resume")
        self.resume_btn.clicked.connect(self._on_resume)
        header.addWidget(self.resume_btn)
        self.stop_btn = QtWidgets.QPushButton("■ Stop current")
        self.stop_btn.clicked.connect(self._on_stop)
        header.addWidget(self.stop_btn)
        outer.addLayout(header)

        # --- Jobs table -----------------------------------------------
        self.table = QtWidgets.QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(
            ["#", "Status", "Name", "Stage", "Progress", "Output"])
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(2, QtWidgets.QHeaderView.Stretch)
        hh.setSectionResizeMode(5, QtWidgets.QHeaderView.Stretch)
        hh.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(3, QtWidgets.QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(4, QtWidgets.QHeaderView.ResizeToContents)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectRows)
        self.table.setSelectionMode(
            QtWidgets.QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(
            QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.itemSelectionChanged.connect(self._refresh_details)
        outer.addWidget(self.table, 2)

        # --- Row buttons ----------------------------------------------
        row_btns = QtWidgets.QHBoxLayout()
        for label, slot in [
            ("Add job...", self._on_add_job),
            ("Edit", self._on_edit_job),
            ("↑ Up", lambda: self._move_selected(-1)),
            ("↓ Down", lambda: self._move_selected(+1)),
            ("Remove", self._on_remove_job),
            ("Retry", self._on_retry_job),
            ("Open log...", self._on_open_log),
        ]:
            b = QtWidgets.QPushButton(label)
            b.clicked.connect(slot)
            row_btns.addWidget(b)
        row_btns.addStretch(1)
        outer.addLayout(row_btns)

        # --- Details panel --------------------------------------------
        details_box = QtWidgets.QGroupBox("Selected job")
        dl = QtWidgets.QVBoxLayout(details_box)
        self.details_label = QtWidgets.QLabel(
            "Select a job above for details.")
        self.details_label.setWordWrap(True)
        self.details_label.setTextInteractionFlags(
            QtCore.Qt.TextSelectableByMouse)
        self.details_label.setStyleSheet(
            "font-family: Consolas, monospace; font-size: 11px;")
        dl.addWidget(self.details_label)
        outer.addWidget(details_box)

        # --- Live log -------------------------------------------------
        log_box = QtWidgets.QGroupBox("Live log (current job)")
        lll = QtWidgets.QVBoxLayout(log_box)
        self.log_view = QtWidgets.QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(2000)
        self.log_view.setStyleSheet(
            "font-family: Consolas, monospace; font-size: 10px;")
        lll.addWidget(self.log_view)
        outer.addWidget(log_box, 1)

        self._refresh_table()
        self._refresh_status()

    # ----- Controls (top header) --------------------------------------

    def _on_start(self) -> None:
        if self.queue.paused:
            # Friendly nudge: Start has no effect while paused. Use Resume.
            QtWidgets.QMessageBox.information(
                self, "Waruka",
                "Queue is paused. Use Resume to continue.")
            return
        if not self.queue.next_runnable():
            QtWidgets.QMessageBox.information(
                self, "Waruka",
                "No pending jobs. Add a job first.")
            return
        self.runner.start()
        self.status_message.emit("Queue started.", 3000)
        self._refresh_status()

    def _on_pause(self) -> None:
        self.runner.pause()
        self.status_message.emit(
            "Pause requested. Current stage will finish.", 4000)

    def _on_resume(self) -> None:
        self.runner.resume()
        self.status_message.emit("Queue resumed.", 3000)

    def _on_stop(self) -> None:
        if not self.runner.is_running():
            return
        if QtWidgets.QMessageBox.question(
            self, "Stop current job",
            "Kill the currently running stage and mark the job as "
            "interrupted?\n\nCompleted stages of this job are preserved "
            "and the queue stops. Use Retry to pick the job up later.",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No,
        ) != QtWidgets.QMessageBox.Yes:
            return
        self.runner.stop_current()

    # ----- Row buttons -------------------------------------------------

    def _on_add_job(self) -> None:
        dlg = AddJobDialog(self.queue, parent=self)
        if dlg.exec() == QtWidgets.QDialog.Accepted and dlg.result_job:
            self.queue.add(dlg.result_job)
            self._refresh_table()
            self.status_message.emit(
                f"Added job '{dlg.result_job.name}'.", 3000)

    def _on_edit_job(self) -> None:
        job = self._selected_job()
        if job is None:
            return
        if job.status == STATUS_RUNNING:
            QtWidgets.QMessageBox.information(
                self, "Waruka",
                "Can't edit a running job. Stop it first.")
            return
        dlg = AddJobDialog(self.queue, existing=job, parent=self)
        if dlg.exec() == QtWidgets.QDialog.Accepted and dlg.result_job:
            # Replace in place
            for i, j in enumerate(self.queue.jobs):
                if j.id == job.id:
                    new = dlg.result_job
                    # Reset stages so they get rebuilt on next pick-up
                    new.stages = []
                    new.build_stages()
                    new.status = STATUS_PENDING
                    new.current_stage_idx = 0
                    new.error = None
                    self.queue.jobs[i] = new
                    break
            self.queue.save()
            self._refresh_table()

    def _move_selected(self, direction: int) -> None:
        job = self._selected_job()
        if job is None:
            return
        if self.queue.move(job.id, direction):
            self._refresh_table()
            self._select_job_id(job.id)

    def _on_remove_job(self) -> None:
        job = self._selected_job()
        if job is None:
            return
        if job.status == STATUS_RUNNING:
            QtWidgets.QMessageBox.information(
                self, "Waruka",
                "Can't remove a running job. Stop it first.")
            return
        if QtWidgets.QMessageBox.question(
            self, "Remove job",
            f"Remove '{job.name}' from the queue?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
        ) != QtWidgets.QMessageBox.Yes:
            return
        self.queue.remove(job.id)
        self._refresh_table()

    def _on_retry_job(self) -> None:
        job = self._selected_job()
        if job is None:
            return
        if job.status not in (STATUS_FAILED, STATUS_INTERRUPTED):
            QtWidgets.QMessageBox.information(
                self, "Waruka",
                "Only failed or interrupted jobs can be retried.")
            return
        self.queue.reset_for_retry(job.id)
        self._refresh_table()
        self.status_message.emit(
            f"Job '{job.name}' queued for retry.", 3000)

    def _on_open_log(self) -> None:
        job = self._selected_job()
        if job is None:
            return
        # New layout: log lives at <artefact_dir>/job.log. Fall back to
        # the legacy ~/.waruka/logs/<id>.log location for jobs that
        # finished before the move so old logs are still openable.
        log_path = artefact_dir(job) / "job.log"
        if not log_path.exists():
            legacy = _QUEUE_LOG_DIR / f"{job.id}.log"
            if legacy.exists():
                log_path = legacy
        if not log_path.exists():
            QtWidgets.QMessageBox.information(
                self, "Waruka", "No log yet for this job.")
            return
        QtGui.QDesktopServices.openUrl(
            QtCore.QUrl.fromLocalFile(str(log_path)))

    # ----- Runner signal handlers --------------------------------------

    def _on_job_started(self, job_id: str) -> None:
        self.log_view.clear()
        self.log_view.appendPlainText(
            f"--- Started job {self._job_label(job_id)} ---")
        self._refresh_table()
        self._refresh_status()
        self._select_job_id(job_id)

    def _on_job_finished(self, job_id: str, success: bool) -> None:
        self._refresh_table()
        self._refresh_status()
        verb = "succeeded" if success else "failed"
        self.status_message.emit(
            f"Job {self._job_label(job_id)} {verb}.", 5000)

    def _on_stage_started(self, job_id: str, stage_name: str) -> None:
        self.log_view.appendPlainText(
            f"\n[stage] {stage_name} starting...")
        self._refresh_table()
        self._refresh_status()

    def _on_stage_finished(self, job_id: str, stage_name: str,
                            ok: bool) -> None:
        glyph = "ok" if ok else "FAILED"
        self.log_view.appendPlainText(
            f"[stage] {stage_name} {glyph}")
        self._refresh_table()

    def _on_stage_progress(self, job_id: str, stage_name: str,
                            frac: float, extras: dict) -> None:
        self._latest_progress = {
            "job_id": job_id, "stage": stage_name,
            "frac": frac, "extras": extras,
        }
        # Update the table row's Progress column without a full rebuild.
        row = self._row_for_job(job_id)
        if row >= 0:
            self.table.setItem(
                row, 4, QtWidgets.QTableWidgetItem(
                    self._fmt_progress(frac, extras)))

    def _on_queue_idle(self) -> None:
        self._refresh_status()
        self.status_message.emit("Queue idle.", 3000)

    def _on_log_line(self, job_id: str, line: str) -> None:
        if self._current_running_job_id() == job_id:
            self.log_view.appendPlainText(line)

    # ----- Refresh helpers ---------------------------------------------

    def _refresh_table(self) -> None:
        prev_id = None
        sel = self._selected_job()
        if sel is not None:
            prev_id = sel.id
        self.table.setRowCount(len(self.queue.jobs))
        for r, j in enumerate(self.queue.jobs):
            label, glyph, colour = _JOB_STATUS_META.get(
                j.status, (j.status, "  ", "#666"))
            # # column
            it_n = QtWidgets.QTableWidgetItem(str(r + 1))
            it_n.setData(QtCore.Qt.UserRole, j.id)
            self.table.setItem(r, 0, it_n)
            # Status
            it_s = QtWidgets.QTableWidgetItem(f"{glyph}{label}")
            it_s.setForeground(QtGui.QColor(colour))
            self.table.setItem(r, 1, it_s)
            # Name
            self.table.setItem(
                r, 2, QtWidgets.QTableWidgetItem(j.name))
            # Stage
            cur_stage = "-"
            if j.stages:
                idx = min(j.current_stage_idx, len(j.stages) - 1)
                cur_stage = j.stages[idx].name
            self.table.setItem(
                r, 3, QtWidgets.QTableWidgetItem(cur_stage))
            # Progress
            if (self._latest_progress.get("job_id") == j.id
                    and j.status == STATUS_RUNNING):
                prog_text = self._fmt_progress(
                    self._latest_progress.get("frac", -1.0),
                    self._latest_progress.get("extras", {}))
            else:
                prog_text = self._summarise_stages(j)
            self.table.setItem(
                r, 4, QtWidgets.QTableWidgetItem(prog_text))
            # Output
            self.table.setItem(
                r, 5, QtWidgets.QTableWidgetItem(
                    Path(j.broadcast_output_path).name))
        if prev_id is not None:
            self._select_job_id(prev_id)
        self._refresh_details()

    def _refresh_details(self) -> None:
        job = self._selected_job()
        if job is None:
            self.details_label.setText("Select a job above for details.")
            return
        lines = []
        lines.append(f"Name:      {job.name}")
        lines.append(f"Status:    {job.status}")
        if job.error:
            lines.append(f"Error:     {job.error}")
        lines.append(f"Project:   {job.project_path}")
        lines.append(f"Output:    {job.broadcast_output_path}")
        if len(job.concat_files) > 1:
            lines.append(f"Concat:    {job.concat_output_path}")
        # v1.x: interpolate output IS the broadcast_output_path (after
        # audio_mux). Surface the fps/backend so the user can confirm
        # what's configured without opening the edit dialog.
        from .jobqueue import _job_has_interp
        if _job_has_interp(job):
            tp = job.tracking_params or {}
            ia = job.interp_args or {}
            fps_ = tp.get("interpolate_fps") or ia.get("fps")
            backend_ = tp.get("interpolate_backend") or ia.get("backend")
            lines.append(f"Interp:    fps={fps_} backend={backend_}")
        lines.append(f"Inputs:    {len(job.concat_files)} chunk(s)")
        for c in job.concat_files[:6]:
            lines.append(f"             {c}")
        if len(job.concat_files) > 6:
            lines.append(f"             ... and "
                          f"{len(job.concat_files) - 6} more")
        lines.append("")
        lines.append("Stages:")
        for s in job.stages:
            tag = {
                STAGE_PENDING: "  ",
                STAGE_RUNNING: "▶ ",
                STAGE_DONE: "✓ ",
                STAGE_FAILED: "✗ ",
                STAGE_SKIPPED: "- ",
            }.get(s.status, "  ")
            timing = ""
            if s.started_at and s.finished_at:
                dt = s.finished_at - s.started_at
                timing = f"  ({_fmt_hms(dt)})"
            err = f"  [{s.error}]" if s.error else ""
            lines.append(f"  {tag}{s.name}{timing}{err}")
        if job.retry_count:
            lines.append(f"\nRetried {job.retry_count} time(s).")
        self.details_label.setText("\n".join(lines))

    def _refresh_status(self) -> None:
        if self.runner.is_running():
            verb = "Paused after current stage" if self.queue.paused else "Running"
            self.status_label.setText(f"{verb}.")
        elif self.queue.paused:
            self.status_label.setText("Paused.")
        elif self.queue.next_runnable() is None:
            done = sum(1 for j in self.queue.jobs
                        if j.status == STATUS_DONE)
            fail = sum(1 for j in self.queue.jobs
                        if j.status == STATUS_FAILED)
            if not self.queue.jobs:
                self.status_label.setText("Idle. Add jobs to begin.")
            else:
                self.status_label.setText(
                    f"Idle. {done} done, {fail} failed.")
        else:
            self.status_label.setText("Idle. Press Start.")

        # Button enable/visible state
        self.start_btn.setEnabled(
            (not self.runner.is_running()) and (not self.queue.paused))
        self.pause_btn.setEnabled(
            self.runner.is_running() and not self.queue.paused)
        self.resume_btn.setEnabled(self.queue.paused)
        self.stop_btn.setEnabled(self.runner.is_running())

    # ----- Misc helpers ------------------------------------------------

    def _selected_job(self) -> Job | None:
        rows = self.table.selectionModel().selectedRows()
        if not rows:
            return None
        r = rows[0].row()
        if r < 0 or r >= len(self.queue.jobs):
            return None
        return self.queue.jobs[r]

    def _row_for_job(self, job_id: str) -> int:
        for r, j in enumerate(self.queue.jobs):
            if j.id == job_id:
                return r
        return -1

    def _select_job_id(self, job_id: str) -> None:
        row = self._row_for_job(job_id)
        if row >= 0:
            self.table.selectRow(row)

    def _job_label(self, job_id: str) -> str:
        j = self.queue.find(job_id)
        return f"'{j.name}'" if j else job_id

    def _current_running_job_id(self) -> str | None:
        for j in self.queue.jobs:
            if j.status == STATUS_RUNNING:
                return j.id
        return None

    @staticmethod
    def _fmt_progress(frac: float, extras: dict) -> str:
        step = extras.get("step", "")
        eta = extras.get("eta_s")
        if frac is None or frac < 0:
            return f"{step}".strip() or "..."
        pct = max(0, min(100, int(round(frac * 100))))
        eta_s = f"  eta {_fmt_hms(eta)}" if eta else ""
        return f"{step} {pct}%{eta_s}"

    @staticmethod
    def _summarise_stages(job: Job) -> str:
        if not job.stages:
            return ""
        done = sum(1 for s in job.stages if s.status == STAGE_DONE)
        return f"{done}/{len(job.stages)} stages"


# --------------------------------------------------------------------------
# Main window -- tabs shell only
# --------------------------------------------------------------------------

class MainWindow(QtWidgets.QMainWindow):
    """Tabbed shell. Holds the Track tab (end-to-end tracking flow) and
    the Concat tab (multi-clip concat + trim). Status messages from
    either tab funnel into the shared status bar."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Waruka")
        # Load the title-bar / taskbar icon. Lives at REPO_ROOT/icons/
        # in dev and at <bundle>/_internal/icons/ in the frozen build
        # (placed there by build_exe.py's DATA_FILES). Same _REPO_ROOT
        # pattern as the other runtime resources -- works in both modes.
        _ico = Path(WARUKA_PARENT) / "icons" / "waruka.ico"
        if _ico.is_file():
            self.setWindowIcon(QtGui.QIcon(str(_ico)))
            # Also set the application-wide icon so any future dialog
            # without an explicit parent inherits it.
            QtWidgets.QApplication.setWindowIcon(QtGui.QIcon(str(_ico)))
        self.resize(900, 640)

        self.tabs = QtWidgets.QTabWidget(self)
        self.track_tab = TrackTab(self)
        self.concat_tab = ConcatTab(self, self)
        self.postprocess_tab = PostProcessTab(self)
        self.queue_tab = QueueTab(self)

        self.tabs.addTab(self.track_tab, "Track")
        self.tabs.addTab(self.concat_tab, "Concat")
        self.tabs.addTab(self.postprocess_tab, "Post-process")
        self.tabs.addTab(self.queue_tab, "Queue")
        self.setCentralWidget(self.tabs)

        # Funnel status messages from any tab into the status bar.
        for tab in (self.track_tab, self.concat_tab,
                     self.postprocess_tab, self.queue_tab):
            tab.status_message.connect(
                lambda msg, ms: self.statusBar().showMessage(msg, ms))

        # Backward-compat shim: a few smoketests reach in for
        # ``MainWindow.picker`` / ``MainWindow._params`` etc., which
        # historically lived on the main window directly. Map them
        # through to the track tab so existing tests keep working.
        # New code should reach in via ``MainWindow.track_tab`` instead.

        self.statusBar().showMessage("Open a video to begin.")

    # ----- backward-compat properties -------------------------------------
    # The original MainWindow had these on itself; some smoketests use
    # them. New code: prefer ``self.track_tab.<...>`` directly.

    @property
    def picker(self) -> VideoPickerWidget:
        return self.track_tab.picker

    @property
    def step_cards(self) -> dict[str, StepCardWidget]:
        return self.track_tab.step_cards

    @property
    def _runner(self) -> StepRunner:
        return self.track_tab._runner

    @property
    def _params(self) -> ProcessingParams | None:
        return self.track_tab._params

    @_params.setter
    def _params(self, value: ProcessingParams | None) -> None:
        self.track_tab._params = value

    def _refresh_step_status(self) -> None:
        self.track_tab._refresh_step_status()

    def _run_process(self) -> None:
        self.track_tab._run_process()

    def _launch_step(self, step: str, cmd_args: list[str],
                     extra_args: list[str]) -> None:
        self.track_tab._launch_step(step, cmd_args, extra_args)

    # ----- cross-tab handover (used by ConcatTab) -------------------------

    def activate_track_with(self, path: str | Path) -> None:
        """Switch to the Track tab and load the given video."""
        self.tabs.setCurrentWidget(self.track_tab)
        self.track_tab.load_video(path)


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

def run_gui(argv: list[str] | None = None) -> int:
    """Entry point invoked by ``python -m waruka gui``."""
    app = QtWidgets.QApplication(argv or sys.argv)
    win = MainWindow()
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(run_gui())
