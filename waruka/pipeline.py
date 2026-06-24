# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Stefan Lewis
"""Cross-stage chunked pipeline (added v0.12 for #20b).

Splits the input video's [t0, t1] range into chunks of `chunk_seconds`
and runs the four pipeline stages (track -> classify -> campath ->
render) as concurrent worker threads. Each worker dequeues a chunk
from its input queue, processes, and posts to the next stage's queue.

Architecture:
    track       --chunk--> classify_q
    classify    --chunk--> campath_q
    campath     --chunk--> render_q
    render      --chunk--> chunk_outputs

So track of chunk N+1 runs concurrently with classify/campath/render
of chunks N, N-1, .... For a long match (50min track + 50min render
serial = ~100min), wall-clock approaches max(total_track, total_render)
because the stages pipeline -- typically ~50% saving.

GPU contention: track and render both use CUDA compute. They share
the default stream so don't truly overlap, BUT their CPU-bound
portions (pad_source_for_blur in render, post-processing in track)
DO run while the other stage's GPU work is in flight, recovering most
of the lost time. CPU stages (classify, campath) run on dedicated
threads and overlap freely.

Output: per-chunk MP4s concatenated via `ffmpeg -f concat` at the end.
All chunks use the same Waruka render path (same codec, resolution,
fps), so concat without re-encoding works.

Chunk boundaries are HARD: each chunk has independent tracker state,
classifier lifetime stats, and campath smoothing. Cross-boundary
tracks split into per-chunk pieces. For typical chunk sizes (>=30s)
the boundary artefacts are small and localised; pick larger chunks
(60-120s) for offline runs where continuity matters more than
parallelism. For live mode (future), chunks become rolling windows
on the stream input.
"""
from __future__ import annotations

import os
import queue
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path


def _concat_mp4s(paths, out_path):
    """Use ffmpeg concat demuxer to merge MP4s without re-encoding.
    Uses imageio_ffmpeg's bundled ffmpeg binary so we don't depend on
    system ffmpeg being in PATH (Waruka already uses imageio_ffmpeg for
    rendering)."""
    import imageio_ffmpeg
    # Belt-and-braces: never let the concat output land on one of its
    # own inputs (ffmpeg -y would truncate that input mid-read and
    # destroy it). See the critical data-loss bug in BACKLOG.md.
    from .jobqueue import assert_output_safe
    assert_output_safe(out_path, paths)
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    list_file = Path(out_path).with_suffix(".concat.txt")
    list_file.write_text(
        "\n".join(f"file '{p.absolute().as_posix()}'" for p in paths),
        encoding="utf-8")
    cmd = [ffmpeg, "-y", "-loglevel", "error", "-f", "concat",
           "-safe", "0", "-i", str(list_file), "-c", "copy", str(out_path)]
    # Suppress fresh-console allocation on Windows when this pipeline
    # is invoked from the windowed waruka.exe (otherwise this concat
    # would flash a console window).
    _kw = {}
    if sys.platform == "win32":
        _kw["creationflags"] = subprocess.CREATE_NO_WINDOW
    subprocess.run(cmd, check=True, **_kw)
    list_file.unlink()


def run_pipeline(
    project_path: str,
    video: str | None = None,
    t0: float = 0.0,
    t1: float | None = None,
    chunk_seconds: float = 30.0,
    # Pre/post overlap seconds (now default 0 because cross-chunk
    # state continuity supersedes them). When cross_chunk_state=True
    # the tracker carries IDs + Kalman states forward and the
    # classifier sees cumulative track frames, so no extra evidence
    # window is needed at boundaries. The overlap knobs remain for
    # opt-out / legacy comparison.
    post_overlap_seconds: float = 0.0,
    pre_overlap_seconds: float = 0.0,
    # Cross-chunk state continuity (v0.12 #20b final). When True
    # (default), tracker state (active tracks + IDs + Kalman) flows
    # forward, classifier reads cumulative tracks across all chunks
    # so far, and smoother state is bridged via initial_smoother_state
    # (already wired). Result: chunked output matches single-pass
    # output exactly up to numerical noise. Set False to revert to
    # the broken (boundary-divergent) behaviour for A/B comparison.
    cross_chunk_state: bool = True,
    out_path: str = "broadcast.mp4",
    work_dir: str = "_pipeline_chunks",
    cleanup_chunks: bool = True,
    track_kwargs: dict | None = None,
    classify_buffer_m: float = 1.0,
    campath_kwargs: dict | None = None,
    render_kwargs: dict | None = None,
):
    """Run the chunked pipeline. See module docstring for design."""
    from . import perception, classify as classify_mod
    from . import campath as campath_mod
    from . import render as render_mod
    track_kwargs = dict(track_kwargs or {})
    campath_kwargs = dict(campath_kwargs or {})
    render_kwargs = dict(render_kwargs or {})

    if t1 is None:
        import cv2
        from .config import ProjectConfig
        cfg = ProjectConfig.load(project_path)
        src = video or cfg.source_video
        cap = cv2.VideoCapture(src)
        fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
        nfr = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        t1 = nfr / fps

    chunks = []  # list of (cid, render_t0, render_t1, proc_t0, proc_t1)
    t = float(t0)
    cid = 0
    while t < t1:
        c_t0 = t  # rendered start
        c_t1 = min(t + chunk_seconds, t1)  # rendered end
        # pre-overlap: chunk 0 has no preceding chunk so pre=0
        pre = 0.0 if cid == 0 else float(pre_overlap_seconds)
        proc_t0 = max(t0, c_t0 - pre)
        proc_t1 = min(c_t1 + post_overlap_seconds, t1)
        chunks.append((cid, c_t0, c_t1, proc_t0, proc_t1))
        t = c_t1
        cid += 1
    print(f"pipeline: {len(chunks)} chunks of {chunk_seconds:.1f}s "
          f"(+{pre_overlap_seconds:.1f}s pre / "
          f"+{post_overlap_seconds:.1f}s post overlap), "
          f"t={t0:.1f}-{t1:.1f}s", flush=True)

    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)

    classify_q: queue.Queue = queue.Queue()
    campath_q: queue.Queue = queue.Queue()
    render_q: queue.Queue = queue.Queue()
    sentinel = object()

    timings: dict[int, dict[str, float]] = {cid_: {} for cid_, _, _, _, _ in chunks}
    t_lock = threading.Lock()
    error_evt = threading.Event()
    errors: list[tuple[str, BaseException]] = []
    chunk_outputs: list[tuple[int, Path]] = []

    def _record(cid_, stage, elapsed):
        with t_lock:
            timings[cid_][stage] = elapsed

    def _log(msg):
        print(f"  {msg}", flush=True)

    def _track_worker():
        prev_state_path: str | None = None
        # Track previous chunk's render range so chunk N can back-emit
        # it -- chunk N's tracker has hits in both chunks (resumed state
        # + new) so the back-emit interpolates chunk N-1's last frames
        # with future-hit context, matching single-pass.
        prev_chunk_frame_range: tuple[int, int] | None = None
        fps_assumed = 20.0  # for frame conversion (real fps known after
                            # first track but assumed here for chunk planning)
        try:
            for cid_, c_t0, c_t1, proc_t0, proc_t1 in chunks:
                if error_evt.is_set():
                    break
                t_start = time.time()
                tracks_out = work / f"tracks_{cid_:04d}.json"
                state_out = (str(work / f"tracker_state_{cid_:04d}.json")
                             if cross_chunk_state else None)
                # Back-emit setup: chunk N back-emits chunk N-1's
                # render range. Skip on chunk 0 (no predecessor).
                back_emit_range = None
                back_emit_out = None
                if cross_chunk_state and prev_chunk_frame_range is not None:
                    back_emit_range = prev_chunk_frame_range
                    back_emit_out = str(
                        work / f"backemit_{cid_ - 1:04d}.json")
                _log(f"[track]    chunk {cid_:3d} "
                     f"render({c_t0:.1f}-{c_t1:.1f}s) "
                     f"proc({proc_t0:.1f}-{proc_t1:.1f}s) start")
                perception.run_perception(
                    project_path, video=video, t0=proc_t0, t1=proc_t1,
                    out_path=str(tracks_out),
                    initial_tracker_state=prev_state_path,
                    tracker_state_out=state_out,
                    back_emit_range=back_emit_range,
                    back_emit_out=back_emit_out,
                    **track_kwargs)
                if cross_chunk_state:
                    prev_state_path = state_out
                # Record this chunk's RENDER frame range for the NEXT
                # chunk's back-emit to use.
                this_f0 = int(c_t0 * fps_assumed)
                this_f1 = int(c_t1 * fps_assumed) - 1
                prev_chunk_frame_range = (this_f0, this_f1)
                elapsed = time.time() - t_start
                _record(cid_, "track", elapsed)
                _log(f"[track]    chunk {cid_:3d} done in {elapsed:.1f}s")
                classify_q.put((cid_, c_t0, c_t1, proc_t0, proc_t1,
                                tracks_out, back_emit_out))
        except BaseException as e:
            errors.append(("track", e))
            error_evt.set()
        finally:
            classify_q.put(sentinel)

    def _classify_worker():
        # Cumulative tracks file grows with each chunk. The classifier
        # reads it and sees full per-track lifetimes (tracker IDs
        # persist across chunks via cross-chunk state handoff).
        #
        # IMPORTANT: chunk N's classify is delayed until chunk N+1's
        # tracks are also in the cumulative. This gives chunk N the
        # SAME future evidence chunk N+1 has at the boundary, so the
        # per-frame classify outputs at chunk N's last frame and chunk
        # N+1's first frame are consistent. Without this delay, chunks
        # diverged by ~3 deg at boundaries even with state passing
        # because chunk N's classifier saw less future history.
        # The last chunk has no successor to wait for and runs
        # immediately on its sentinel.
        cumulative_path = work / "tracks_cumulative.json"
        import json as _json
        pending: tuple | None = None
        # KNOWN LIMITATION: chunk 0 has ~5 deg RMS yaw divergence from
        # single-pass for the first ~20 seconds of its render range.
        # Root cause is chunked tracker behaviour differing from
        # single-pass tracker behaviour (chunked produces a few extra
        # track IDs at the start). Tried fix #1 (defer chunk 0 classify
        # until ALL chunks done so it sees full cumulative) -- made
        # things WORSE (max 38 deg) because giving the classifier
        # more evidence amplifies the tracker-divergence's effect on
        # the foreign rule. Fix requires solving the tracker-init
        # divergence, which is out of scope for v0.12.
        def _run_classify(itm):
            cid_, c_t0, c_t1, proc_t0, proc_t1, tracks_out, _be = itm
            t_start = time.time()
            players_out = work / f"players_{cid_:04d}.json"
            if cross_chunk_state:
                classify_input = str(cumulative_path)
            else:
                classify_input = str(tracks_out)
            classify_mod.classify_tracks(
                classify_input, project_path,
                classify_buffer_m, out_path=str(players_out))
            elapsed = time.time() - t_start
            _record(cid_, "classify", elapsed)
            _log(f"[classify] chunk {cid_:3d} done in {elapsed:.1f}s "
                 f"({'lookahead-by-1' if cross_chunk_state else 'self-only'})")
            campath_q.put((cid_, c_t0, c_t1, proc_t0, proc_t1, players_out))

        def _merge_back_emit(back_emit_path):
            """Replace alive-tracks frame entries in cumulative for the
            back-emit's frame range with the back-emit's entries.
            Tracks that died in the prior chunk (and don't appear in
            back_emit) keep their entries from cumulative."""
            with open(str(cumulative_path)) as _f:
                cum = _json.load(_f)
            with open(back_emit_path) as _f:
                be = _json.load(_f)
            be_by_frame = {f["frame"]: f for f in be["frames"]}
            be_ids = set()
            for f in be["frames"]:
                for p in f["players"]:
                    be_ids.add(p["id"])
            replaced = 0
            for f in cum["frames"]:
                if f["frame"] in be_by_frame:
                    # Drop alive-track entries (these are the ones
                    # back_emit has BETTER positions for).
                    f["players"] = [p for p in f["players"]
                                    if p["id"] not in be_ids]
                    # Add the back-emit's improved entries.
                    f["players"].extend(be_by_frame[f["frame"]]["players"])
                    replaced += 1
            with open(str(cumulative_path), "w") as _f:
                _json.dump(cum, _f)
            return replaced
        try:
            while True:
                item = classify_q.get()
                if item is sentinel:
                    if pending is not None:
                        _run_classify(pending)
                        pending = None
                    return
                if error_evt.is_set():
                    continue
                (cid_, c_t0, c_t1, proc_t0, proc_t1,
                 tracks_out, back_emit_path) = item
                if cross_chunk_state:
                    # Append this chunk's frames to the cumulative.
                    with open(str(tracks_out)) as _f:
                        this_chunk = _json.load(_f)
                    if cumulative_path.exists():
                        with open(str(cumulative_path)) as _f:
                            cum = _json.load(_f)
                        cum["frames"].extend(this_chunk["frames"])
                    else:
                        cum = this_chunk
                    with open(str(cumulative_path), "w") as _f:
                        _json.dump(cum, _f)
                    # Apply back-emit (if this chunk produced one) --
                    # patches PREVIOUS chunk's last-frame positions in
                    # the cumulative using THIS chunk's tracker hits.
                    if back_emit_path and Path(back_emit_path).exists():
                        n_patched = _merge_back_emit(back_emit_path)
                        _log(f"[classify] back-emit chunk {cid_-1:3d}: "
                             f"patched {n_patched} frames in cumulative")
                    # Lookahead-by-1: process previously-buffered chunk
                    # now that this chunk's tracks are in cumulative.
                    if pending is not None:
                        _run_classify(pending)
                    pending = item
                else:
                    _run_classify(item)
        except BaseException as e:
            errors.append(("classify", e))
            error_evt.set()
        finally:
            campath_q.put(sentinel)

    def _campath_worker():
        # State handoff: each chunk's smoother state at the RENDER
        # boundary (not the end of processing) seeds the next chunk.
        # We extract from path entries via finite difference because
        # the smoother_final_state field captures end-of-processing
        # state, which includes the overlap region we don't want.
        prev_state: dict | None = None
        import json as _json
        try:
            while True:
                item = campath_q.get()
                if item is sentinel:
                    return
                if error_evt.is_set():
                    continue
                cid_, c_t0, c_t1, proc_t0, proc_t1, players_out = item
                t_start = time.time()
                campath_out = work / f"campath_{cid_:04d}.json"
                campath_mod.plan_campath(
                    str(players_out), project_path,
                    out_path=str(campath_out),
                    initial_smoother_state=prev_state,
                    **campath_kwargs)
                with open(str(campath_out)) as _f:
                    cp_data = _json.load(_f)
                # Find the path entry closest to the render boundary
                # (c_t1) and capture state for the next chunk's init.
                fps_cp = cp_data["fps"]
                boundary_frame = int(c_t1 * fps_cp)
                path_entries = cp_data["path"]
                # find idx of last entry with frame < boundary_frame
                # (so it's still within the rendered range)
                bidx = None
                for i, e in enumerate(path_entries):
                    if e["frame"] < boundary_frame:
                        bidx = i
                    else:
                        break
                if bidx is None or bidx == 0:
                    prev_state = cp_data.get("smoother_final_state")
                else:
                    dt_cp = 1.0 / fps_cp
                    p1 = path_entries[bidx]
                    p0 = path_entries[bidx - 1]
                    prev_state = {
                        "yaw_pos": p1["yaw"],
                        "yaw_vel": (p1["yaw"] - p0["yaw"]) / dt_cp,
                        "pitch_pos": p1["pitch"],
                        "pitch_vel": (p1["pitch"] - p0["pitch"]) / dt_cp,
                        "hfov_pos": p1["hfov"],
                        "hfov_vel": (p1["hfov"] - p0["hfov"]) / dt_cp,
                        "d_pos": p1["d"],
                        "d_vel": (p1["d"] - p0["d"]) / dt_cp,
                    }
                elapsed = time.time() - t_start
                _record(cid_, "campath", elapsed)
                _log(f"[campath]  chunk {cid_:3d} done in {elapsed:.1f}s")
                render_q.put((cid_, c_t0, c_t1, campath_out))
        except BaseException as e:
            errors.append(("campath", e))
            error_evt.set()
        finally:
            render_q.put(sentinel)

    def _render_worker():
        try:
            while True:
                item = render_q.get()
                if item is sentinel:
                    return
                if error_evt.is_set():
                    continue
                cid_, c_t0, c_t1, campath_out = item
                t_start = time.time()
                mp4_out = work / f"broadcast_{cid_:04d}.mp4"
                # Render ONLY the rendered range, not the overlap.
                # The campath JSON contains entries up to proc_t1 but
                # render_path's t0/t1 clip the output to [c_t0, c_t1].
                render_mod.render_path(
                    str(campath_out), project_path,
                    video=video, out_path=str(mp4_out),
                    t0=c_t0, t1=c_t1,
                    **render_kwargs)
                elapsed = time.time() - t_start
                _record(cid_, "render", elapsed)
                _log(f"[render]   chunk {cid_:3d} done in {elapsed:.1f}s")
                chunk_outputs.append((cid_, mp4_out))
        except BaseException as e:
            errors.append(("render", e))
            error_evt.set()

    threads = [
        threading.Thread(target=_track_worker, name="track", daemon=True),
        threading.Thread(target=_classify_worker, name="classify", daemon=True),
        threading.Thread(target=_campath_worker, name="campath", daemon=True),
        threading.Thread(target=_render_worker, name="render", daemon=True),
    ]
    pipeline_start = time.time()
    for thr in threads:
        thr.start()
    for thr in threads:
        thr.join()
    pipeline_total = time.time() - pipeline_start

    if errors:
        raise RuntimeError(f"pipeline errors: {errors}")

    chunk_outputs.sort(key=lambda x: x[0])
    if len(chunk_outputs) == 1:
        shutil.copy(str(chunk_outputs[0][1]), out_path)
    else:
        _concat_mp4s([p for _, p in chunk_outputs], out_path)

    if cleanup_chunks:
        for cid_, _ in chunk_outputs:
            for prefix, ext in [("tracks", ".json"), ("players", ".json"),
                                ("players", "_labeled.json"),
                                ("campath", ".json"),
                                ("broadcast", ".mp4")]:
                f = work / f"{prefix}_{cid_:04d}{ext}"
                if f.exists():
                    f.unlink()
        try:
            work.rmdir()
        except OSError:
            pass

    # Timing summary.
    print(f"\npipeline done in {pipeline_total:.1f}s", flush=True)
    print("per-chunk timings (sec):")
    print(f"  {'chunk':>5}  {'track':>6}  {'classify':>8}  "
          f"{'campath':>7}  {'render':>6}")
    for cid_ in sorted(timings):
        ts = timings[cid_]
        print(f"  {cid_:>5}  "
              f"{ts.get('track', 0):>6.1f}  "
              f"{ts.get('classify', 0):>8.1f}  "
              f"{ts.get('campath', 0):>7.1f}  "
              f"{ts.get('render', 0):>6.1f}")
    serial = sum(sum(t.values()) for t in timings.values())
    print(f"  {'sum':>5}  "
          f"{sum(t.get('track', 0) for t in timings.values()):>6.1f}  "
          f"{sum(t.get('classify', 0) for t in timings.values()):>8.1f}  "
          f"{sum(t.get('campath', 0) for t in timings.values()):>7.1f}  "
          f"{sum(t.get('render', 0) for t in timings.values()):>6.1f}")
    if serial > 0:
        saved = (serial - pipeline_total) / serial * 100
        print(f"\nserial-equivalent: {serial:.1f}s, pipelined: "
              f"{pipeline_total:.1f}s ({saved:.1f}% saving)")
    return out_path
