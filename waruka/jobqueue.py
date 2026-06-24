# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Stefan Lewis
"""Job queue for overnight batch processing (#35).

A Job is one game end-to-end: concat input chunks -> waruka pipeline
(track + classify + campath + render in one chunked call) ->
optional waruka interpolate. JobQueue is JSON-backed at
~/.waruka/queue.json so jobs survive Waruka restarts. QueueRunner
launches stages sequentially via subprocess, honours pause requests
at stage boundaries, and emits Qt signals the QueueTab listens to.

On reload, any job that was 'running' when the app crashed is
flipped to 'interrupted' so the user can retry it from the UI.
"""
from __future__ import annotations

import dataclasses
import json
import os
import shlex
import sys
import time
import uuid
from pathlib import Path
from typing import Optional


_PERSIST_DIR = Path.home() / ".waruka"
PERSIST_PATH = _PERSIST_DIR / "queue.json"


# --------------------------------------------------------------------------
# Output-safety guard (critical: never overwrite a source clip)
# --------------------------------------------------------------------------

def _norm_path(p: str | Path) -> str:
    """Normalised, case-folded absolute path for robust comparison.

    Resolves symlinks/`..`, then `os.path.normcase` so the compare is
    case-insensitive on Windows (where SOURCE.MP4 == source.mp4)."""
    try:
        resolved = Path(p).resolve()
    except Exception:
        resolved = Path(os.path.abspath(str(p)))
    return os.path.normcase(str(resolved))


def assert_output_safe(out_path: str | Path,
                       input_paths) -> None:
    """Raise ValueError if `out_path` is one of `input_paths`.

    Belt-and-braces backend guard for the concat/ffmpeg paths: writing a
    concatenated output onto one of its own inputs truncates that input
    mid-read and destroys the source footage (see BACKLOG critical bug).
    We fail the stage rather than risk data loss."""
    out_norm = _norm_path(out_path)
    for ip in input_paths or []:
        if _norm_path(ip) == out_norm:
            raise ValueError(
                "Refusing to write the concatenated output onto one of its "
                f"own input clips -- this would destroy source footage:\n"
                f"  output: {out_path}\n  input:  {ip}")


# --------------------------------------------------------------------------
# Data model
# --------------------------------------------------------------------------

# Per-job status values. running and paused are transient (live state);
# the rest persist across restarts.
STATUS_PENDING = "pending"
STATUS_RUNNING = "running"
STATUS_DONE = "done"
STATUS_FAILED = "failed"
STATUS_INTERRUPTED = "interrupted"  # crashed mid-job


# Per-stage status values.
STAGE_PENDING = "pending"
STAGE_RUNNING = "running"
STAGE_DONE = "done"
STAGE_FAILED = "failed"
STAGE_SKIPPED = "skipped"


@dataclasses.dataclass
class JobStage:
    """One stage of a job's pipeline."""
    name: str                              # display name and key
    status: str = STAGE_PENDING
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    exit_code: Optional[int] = None
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "JobStage":
        return cls(**{k: d.get(k) for k in cls.__dataclass_fields__})


@dataclasses.dataclass
class Job:
    """One game to be processed end-to-end."""
    id: str                                # uuid hex
    name: str                              # display name
    concat_files: list[str]                # input chunks (may be 1 file)
    project_path: str                      # project.json with calib + marks
    concat_output_path: str                # concatenated video (intermediate)
    broadcast_output_path: str             # final broadcast (post-render)
    interp_output_path: Optional[str]      # final smooth (post-interpolate)

    # Legacy raw-CLI-flag dict (kept for backward compat with persisted
    # jobs; new fields go in `tracking_params` instead).
    pipeline_args: dict = dataclasses.field(default_factory=dict)
    interp_args: dict = dataclasses.field(default_factory=dict)
    # Structured tracking parameters captured via the Track tab's
    # ParamsDialog -- same form, same defaults, kept in parity. Keys
    # mirror ProcessingParams' fields: t0, t1, mode, stride, view_mode,
    # create_no_audio_copy, sr_enabled, interpolate_fps,
    # interpolate_backend. Empty means "CLI defaults".
    tracking_params: dict = dataclasses.field(default_factory=dict)
    # Probed at job-creation time from the first input chunk. Drives the
    # audio_mux stage: if True, render writes a silent intermediate and
    # audio_mux merges the source audio back in to produce the final
    # broadcast.
    has_audio: bool = False
    # Concat-step audio choices (parity with the Concat tab). When
    # preserve_concat_audio is False the concat'd pano is written silent
    # and -- because audio_mux sources its audio from that concat -- the
    # whole job is treated as audio-free (no audio_mux, silent broadcast).
    # concat_no_audio_copy additionally emits a silent <name>_no_audio.mp4
    # companion of the (with-audio) concat. Defaults keep the prior
    # behaviour: audio preserved, no extra silent copy.
    preserve_concat_audio: bool = True
    concat_no_audio_copy: bool = False
    # When True, the concatenated intermediate is deleted after the job
    # completes successfully. False keeps it for debugging.
    keep_intermediates: bool = False

    status: str = STATUS_PENDING
    stages: list[JobStage] = dataclasses.field(default_factory=list)
    current_stage_idx: int = 0
    retry_count: int = 0
    error: Optional[str] = None
    added_at: float = dataclasses.field(default_factory=time.time)

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["stages"] = [s.to_dict() if isinstance(s, JobStage)
                        else s for s in self.stages]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Job":
        d = dict(d)
        d["stages"] = [JobStage.from_dict(s) for s in d.get("stages", [])]
        return cls(**{k: d.get(k) for k in cls.__dataclass_fields__})

    def build_stages(self) -> None:
        """Initialise self.stages based on inputs + tracking_params.

        Stage list shape:
            (concat?) -> processing-core -> (interpolate?) -> (audio_mux?)

        processing-core is sequential by default (4 stages: track +
        classify + campath + render). User can opt into the chunked
        `waruka pipeline` collapse by setting tracking_params['mode'] =
        'pipeline'; until #21 lands, sequential stays default.

        Stage ordering rationale: audio_mux comes AFTER interpolate so
        the final broadcast is the interpolated version with audio (vs
        the pre-1.0 v1 ordering, which audio_mux'd the un-interpolated
        render and produced a silent interpolated `_smooth.mp4` as a
        separate output).
        """
        if self.stages:
            return
        out: list[JobStage] = []
        if len(self.concat_files) > 1:
            out.append(JobStage(name="concat"))
        mode = (self.tracking_params or {}).get("mode", "sequential")
        if mode == "pipeline":
            out.append(JobStage(name="pipeline"))
        else:
            for n in ("track", "classify", "campath", "render"):
                out.append(JobStage(name=n))
        if _job_has_interp(self):
            out.append(JobStage(name="interpolate"))
        if _job_keeps_audio(self):
            out.append(JobStage(name="audio_mux"))
        self.stages = out

    def stage(self, name: str) -> Optional[JobStage]:
        for s in self.stages:
            if s.name == name:
                return s
        return None

    def remaining_stages(self) -> list[JobStage]:
        return [s for s in self.stages
                 if s.status in (STAGE_PENDING, STAGE_FAILED)]


# --------------------------------------------------------------------------
# Queue
# --------------------------------------------------------------------------

class JobQueue:
    """Persistent queue. Atomic JSON write on every mutation."""

    def __init__(self, persist_path: Path = PERSIST_PATH) -> None:
        self.persist_path = Path(persist_path)
        self.jobs: list[Job] = []
        self.paused: bool = False
        self.load()

    # ----- Persistence -----

    def load(self) -> None:
        if not self.persist_path.exists():
            return
        try:
            with open(self.persist_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:  # noqa: BLE001
            print(f"[jobqueue] failed to load {self.persist_path}: {e}",
                  flush=True)
            return
        self.paused = bool(data.get("paused", False))
        self.jobs = []
        for jd in data.get("jobs", []):
            try:
                job = Job.from_dict(jd)
            except Exception as e:  # noqa: BLE001
                print(f"[jobqueue] skipping malformed job: {e}", flush=True)
                continue
            # A 'running' job on load means the app crashed mid-job.
            # Flip to 'interrupted' so the user sees it and can retry.
            if job.status == STATUS_RUNNING:
                job.status = STATUS_INTERRUPTED
                job.error = "Waruka exited while this job was running."
                for s in job.stages:
                    if s.status == STAGE_RUNNING:
                        s.status = STAGE_FAILED
                        s.error = "Interrupted."
            # Legacy migration: early v1+ queue jobs were built around a
            # single "pipeline" stage. We've since split that into
            # track + classify + campath + render (so we can avoid the
            # chunk-0 issue in `waruka pipeline` until #21 lands). Rebuild
            # stages for any job that still carries the old layout.
            if any(s.name == "pipeline" for s in job.stages):
                # Reset to pending; any stage progress before the legacy
                # pipeline stage (just 'concat' really) is dropped, which
                # is fine since concat is fast.
                job.stages = []
                job.build_stages()
                job.current_stage_idx = 0
                if job.status not in (STATUS_DONE,):
                    job.status = STATUS_PENDING
                    job.error = ("Stages migrated from legacy 'pipeline' "
                                  "layout to sequential.")
            self.jobs.append(job)

    def save(self) -> None:
        self.persist_path.parent.mkdir(parents=True, exist_ok=True)
        # Atomic write: temp file + rename. On Windows os.replace is
        # atomic for files on the same volume.
        tmp = self.persist_path.with_suffix(self.persist_path.suffix + ".tmp")
        data = {
            "paused": self.paused,
            "jobs": [j.to_dict() for j in self.jobs],
        }
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, self.persist_path)

    # ----- Mutators -----

    def add(self, job: Job) -> None:
        job.build_stages()
        self.jobs.append(job)
        self.save()

    def remove(self, job_id: str) -> bool:
        for i, j in enumerate(self.jobs):
            if j.id == job_id:
                self.jobs.pop(i)
                self.save()
                return True
        return False

    def move(self, job_id: str, direction: int) -> bool:
        """direction = -1 for up, +1 for down."""
        for i, j in enumerate(self.jobs):
            if j.id == job_id:
                ni = i + direction
                if 0 <= ni < len(self.jobs):
                    self.jobs[i], self.jobs[ni] = self.jobs[ni], self.jobs[i]
                    self.save()
                    return True
                return False
        return False

    def find(self, job_id: str) -> Optional[Job]:
        for j in self.jobs:
            if j.id == job_id:
                return j
        return None

    def set_paused(self, paused: bool) -> None:
        self.paused = paused
        self.save()

    def next_runnable(self) -> Optional[Job]:
        """Next job that's pending, interrupted, or paused mid-pipeline.

        Failed jobs are skipped -- user must explicitly retry them.
        """
        for j in self.jobs:
            if j.status in (STATUS_PENDING, STATUS_INTERRUPTED):
                return j
        return None

    def reset_for_retry(self, job_id: str) -> bool:
        """Retry from the first failed/incomplete stage of this job.
        Completed stages stay 'done' (saves their work)."""
        j = self.find(job_id)
        if j is None:
            return False
        for s in j.stages:
            if s.status in (STAGE_FAILED,):
                s.status = STAGE_PENDING
                s.started_at = None
                s.finished_at = None
                s.exit_code = None
                s.error = None
        j.status = STATUS_PENDING
        j.error = None
        j.retry_count += 1
        # Resume stage index at the first not-yet-done stage.
        for i, s in enumerate(j.stages):
            if s.status != STAGE_DONE:
                j.current_stage_idx = i
                break
        self.save()
        return True


# --------------------------------------------------------------------------
# Stage command builders
# --------------------------------------------------------------------------

def _flag_args(d: dict) -> list[str]:
    """{key: val} -> ['--key', 'val'] pairs. Boolean True -> bare flag;
    False or None -> skip."""
    out: list[str] = []
    for k, v in d.items():
        if v is None or v is False:
            continue
        flag = "--" + k.replace("_", "-")
        if v is True:
            out.append(flag)
        else:
            out.extend([flag, str(v)])
    return out


def concat_cmd(job: Job, ffmpeg_bin: str) -> tuple[list[str], list[str]]:
    """Returns (cmd_args, extra_files_to_cleanup).

    Builds an ffmpeg concat-demuxer command. The concat list file lives
    inside the artefact dir alongside the (also-intermediate) concat
    output. The list is returned as an extra cleanup path so the runner
    can remove it once the stage finishes.
    """
    # Belt-and-braces: never let the concat output land on a source clip.
    assert_output_safe(job.concat_output_path, job.concat_files)
    list_path = str(artefact_dir(job) / "concat_list.txt")
    # Ensure artefact dir exists before we write the list file.
    Path(list_path).parent.mkdir(parents=True, exist_ok=True)
    with open(list_path, "w", encoding="utf-8") as f:
        for p in job.concat_files:
            esc = str(Path(p).resolve()).replace("'", "'\\''")
            f.write(f"file '{esc}'\n")
    args = [ffmpeg_bin, "-y", "-f", "concat", "-safe", "0",
             "-i", list_path, "-c", "copy"]
    # Drop audio from the concat when the source has none, or the user
    # opted out of preserving it. (audio_mux sources its audio from this
    # concat, so a silent concat => silent broadcast -- see _job_keeps_audio.)
    if not _job_keeps_audio(job):
        args.append("-an")
    args.append(job.concat_output_path)
    return args, [list_path]


def artefact_dir(job: Job) -> Path:
    """Subdirectory holding all of a job's non-final outputs:
    intermediate JSONs, concat list, concat'd video, silent broadcast,
    pipeline chunks, per-job log. Matches the Track tab's
    ``waruka_tracking/<basename>/`` convention so the broadcast output's
    parent stays clean.
    """
    bp = Path(job.broadcast_output_path)
    return bp.parent / "waruka_tracking" / bp.stem


def _job_video_source(job: Job) -> str:
    """Whichever video the downstream stages should consume:
    the concatenated intermediate when multiple inputs are present,
    otherwise the single input file."""
    return (job.concat_output_path if len(job.concat_files) > 1
            else job.concat_files[0])


def _job_intermediate(job: Job, suffix: str) -> str:
    """Path for an intermediate (JSON, list file, ...) inside the
    job's artefact dir."""
    return str(artefact_dir(job) / suffix)


def _job_has_interp(job: Job) -> bool:
    """True when interpolate should be planned into the job.

    Triggered by either:
      - structured `tracking_params['interpolate_fps']` (current GUI form)
      - legacy `interp_args['fps']` (older persisted jobs)
      - legacy `interp_output_path` truthy (jobs persisted under v0.16
        and earlier; kept for queue migration)
    """
    tp = job.tracking_params or {}
    if tp.get("interpolate_fps"):
        return True
    if (job.interp_args or {}).get("fps"):
        return True
    if job.interp_output_path:
        return True
    return False


def _job_keeps_audio(job: Job) -> bool:
    """Effective audio flag for the whole job.

    True only when the source actually has audio AND the user kept it in
    the concat. Because audio_mux pulls its audio from the concat output,
    dropping audio at the concat step makes the entire job audio-free
    (no audio_mux stage, silent broadcast). Older persisted jobs lack the
    `preserve_concat_audio` field (and Job.from_dict fills missing keys
    with None, not the dataclass default), so we treat both absent and
    None as 'preserve' to keep their behaviour unchanged."""
    preserve = getattr(job, "preserve_concat_audio", True)
    if preserve is None:
        preserve = True
    return bool(job.has_audio and preserve)


def silent_render_path(job: Job) -> str:
    """Render stage's video sink when either audio_mux or interpolate
    follow. Lives in the artefact dir; cleaned up when the job
    completes."""
    return str(artefact_dir(job) / "silent_render.mp4")


def silent_interp_path(job: Job) -> str:
    """Interpolate stage's video sink when audio_mux follows. Lives
    in the artefact dir; cleaned up (or renamed to the
    `_broadcast_no_audio.mp4` companion) on success."""
    return str(artefact_dir(job) / "silent_interp.mp4")


def render_output_path(job: Job) -> str:
    """Where the render stage writes.

    Silent intermediate when EITHER interpolate or audio_mux follows
    (render is video-only either way -- audio gets muxed at the very
    end, after interpolate). Direct write to broadcast_output_path
    only when neither follows."""
    if _job_keeps_audio(job) or _job_has_interp(job):
        return silent_render_path(job)
    return job.broadcast_output_path


def interp_output_target(job: Job) -> str:
    """Where the interpolate stage writes.

    Silent intermediate when audio_mux follows; direct write to the
    final broadcast_output_path otherwise."""
    if _job_keeps_audio(job):
        return silent_interp_path(job)
    return job.broadcast_output_path


def audio_mux_video_input(job: Job) -> str:
    """Which video file audio_mux should mux audio into. Picks the
    latest silent intermediate: silent_interp if interpolate ran,
    otherwise silent_render."""
    return (silent_interp_path(job) if _job_has_interp(job)
            else silent_render_path(job))


def _tp(job: Job) -> dict:
    return job.tracking_params or {}


def _time_flags(job: Job) -> list[str]:
    out = []
    tp = _tp(job)
    if tp.get("t0") is not None:
        out.extend(["--t0", str(tp["t0"])])
    if tp.get("t1") is not None:
        out.extend(["--t1", str(tp["t1"])])
    return out


def track_cmd(job: Job, python_bin: str) -> list[str]:
    args = [python_bin, "-m", "waruka", "track",
             "--project", job.project_path,
             "--video", _job_video_source(job),
             "--out", _job_intermediate(job, "tracks.json")]
    tp = _tp(job)
    if tp.get("stride") is not None:
        args.extend(["--stride", str(tp["stride"])])
    args.extend(_time_flags(job))
    # Legacy raw-flag dict still honoured for forward-compat.
    args.extend(_flag_args(job.pipeline_args))
    return args


def classify_cmd(job: Job, python_bin: str) -> list[str]:
    return [python_bin, "-m", "waruka", "classify",
             _job_intermediate(job, "tracks.json"),
             "--project", job.project_path,
             "--out", _job_intermediate(job, "players.json")]


def campath_cmd(job: Job, python_bin: str) -> list[str]:
    return [python_bin, "-m", "waruka", "campath",
             _job_intermediate(job, "players.json"),
             "--project", job.project_path,
             "--out", _job_intermediate(job, "campath.json")]


def render_cmd(job: Job, python_bin: str) -> list[str]:
    args = [python_bin, "-m", "waruka", "render",
             _job_intermediate(job, "campath.json"),
             "--project", job.project_path,
             "--video", _job_video_source(job),
             "--out", render_output_path(job)]
    tp = _tp(job)
    if tp.get("sr_enabled"):
        args.append("--sr")
    args.extend(_time_flags(job))
    return args


def pipeline_cmd(job: Job, python_bin: str) -> list[str]:
    """`waruka pipeline` over the (already-concatenated) input.

    Each job gets its own work-dir under the broadcast output's parent
    so concurrent / retried jobs don't collide on per-chunk
    intermediates. Honours mode == 'pipeline' from tracking_params.
    """
    work_dir = str(artefact_dir(job) / "chunks")
    args = [python_bin, "-m", "waruka", "pipeline",
             "--project", job.project_path,
             "--video", _job_video_source(job),
             "--out", render_output_path(job),
             "--work-dir", work_dir]
    tp = _tp(job)
    if tp.get("stride") is not None:
        args.extend(["--stride", str(tp["stride"])])
    args.extend(_time_flags(job))
    if tp.get("sr_enabled"):
        args.append("--sr")
    args.extend(_flag_args(job.pipeline_args))
    return args


def audio_mux_cmd(job: Job, ffmpeg_bin: str) -> list[str]:
    """Mux source audio into the latest silent video intermediate.

    Reads video from `audio_mux_video_input(job)` (the silent interp
    output if interpolate ran, else the silent render output) and
    audio from the original source. Output is the final
    broadcast_output_path.

    ffmpeg picks the first input's video + the second input's audio,
    stream-copies both, stops at the shorter of the two so we don't
    extend the video to match a longer audio track from the concat.
    """
    return [ffmpeg_bin, "-y",
             "-i", audio_mux_video_input(job),
             "-i", _job_video_source(job),
             "-map", "0:v", "-map", "1:a",
             "-c", "copy", "-shortest",
             job.broadcast_output_path]


def interp_cmd(job: Job, python_bin: str) -> list[str]:
    """`waruka interpolate` over the silent render output.

    Reads `silent_render_path(job)` (the pre-audio render output) and
    writes to `interp_output_target(job)` -- which is either a silent
    intermediate (if audio_mux follows) or the final broadcast path
    (if there's no audio in the source).
    """
    args = [python_bin, "-m", "waruka", "interpolate",
             silent_render_path(job),
             "--out", interp_output_target(job)]
    # Pull interpolate args from BOTH structured tracking_params (new
    # GUI form) and the legacy `interp_args` dict.
    tp = job.tracking_params or {}
    interp_args: dict = {}
    if tp.get("interpolate_fps"):
        interp_args["fps"] = tp["interpolate_fps"]
    if tp.get("interpolate_backend"):
        interp_args["backend"] = tp["interpolate_backend"]
    if tp.get("interpolate_cq"):
        interp_args["cq"] = tp["interpolate_cq"]
    interp_args.update(job.interp_args or {})
    args.extend(_flag_args(interp_args))
    return args


def stage_command(job: Job, stage_name: str,
                    python_bin: str = sys.executable,
                    ffmpeg_bin: str = "ffmpeg"
                    ) -> tuple[list[str], list[str]]:
    """Returns (cmd_args, extra_cleanup_paths) for the named stage."""
    if stage_name == "concat":
        return concat_cmd(job, ffmpeg_bin)
    if stage_name == "track":
        return track_cmd(job, python_bin), []
    if stage_name == "classify":
        return classify_cmd(job, python_bin), []
    if stage_name == "campath":
        return campath_cmd(job, python_bin), []
    if stage_name == "render":
        return render_cmd(job, python_bin), []
    if stage_name == "pipeline":
        return pipeline_cmd(job, python_bin), []
    if stage_name == "audio_mux":
        return audio_mux_cmd(job, ffmpeg_bin), []
    if stage_name == "interpolate":
        return interp_cmd(job, python_bin), []
    raise ValueError(f"unknown stage {stage_name!r}")
