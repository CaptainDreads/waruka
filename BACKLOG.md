# Waruka backlog (post-1.0)

Items captured during 1.0 release prep. Each is self-contained enough
for a contributor (or future self) to pick up cold. When the GitHub
repo is up, these can be cracked open into individual Issues.

## Queue: post-process-only job type

**Status:** Designed, not implemented. Low complexity (~2 hours).

The Queue tab currently only accepts full-pipeline jobs (track →
classify → campath → render → interpolate). When a user has an
already-rendered broadcast and just wants to run `interpolate` (or
`upscale`) on it, the only path is the Post-process tab -- which is
one-job-at-a-time, no queueing.

**What to build:**

- Add a new job type to `waruka/jobqueue.py` ("post_process_only") that
  skips track/classify/campath/render and runs straight into the
  interpolate / upscale stages on an existing rendered MP4.
- Extend the Queue tab's "Add job" dialog (`waruka/gui.py`, the queue
  add-job dialog) with a "post-process only" option that opens a video
  picker for the rendered broadcast input.
- Reuse the existing Post-process tab's params widgets for the
  interpolate / upscale flags.
- `jobqueue.py:stage_command()` already returns commands per stage;
  the change is adding a stage-selector to job construction, not new
  command logic.

**Why it's worth doing:** Overnight batch interpolation of a day's
matches. Letting users tweak parameters once, then queue 8 jobs and
walk away.

## Remote / web processing interface

**Status:** MVP designed, not implemented. Tournament-driven deadline
deferred to remote-desktop for the immediate need; "proper" solution
is post-tournament.

The use case is being away from the dev machine (tournament hotel) but
wanting to process and watch matches on the same machine.

**Short-term workaround (chosen for the imminent tournament):**
Remote desktop into the dev machine.

**Long-term solution (MVP):**

- [Tailscale](https://tailscale.com/) or equivalent zero-config VPN on
  the dev machine and any remote device. Skips port-forward / TLS /
  dynamic-DNS overhead.
- Thin Flask or FastAPI app on the dev machine:
  - `GET /` -- list source videos on disk + status of each
    (untouched / processing / done)
  - `POST /process?video=...` -- adds a queue job (reuses
    `waruka/jobqueue.py`)
  - `GET /download/<name>` -- streams the broadcast output back over HTTP
- Access from any device at `http://<tailnet-ip>:8000`.
- File transfer in: Tailscale share, SFTP, or a watched-folder pattern.
- File transfer out: browser download via the endpoint above.

~200 lines of Python. Ship the MVP first.

**Stretch (post-MVP):**

- Direct cloud-storage integration (S3, Google Drive, Dropbox)
  triggered by the user picking a cloud-hosted video URL. Removes the
  need for any inbound network access on the dev machine.
- A simple progress page that polls `.waruka_progress_*.json` so the
  user can see what stage each job is in from their phone.
- Optional auth (a single shared secret in a config file is enough
  for personal use).

**Why it's worth doing:** Tournament mid-tactics review is a real
workflow. The dev machine has the GPU; phones/tablets don't. A web UI
unlocks that workflow without making remote desktop into a daily tool.

## Auto-fit camera roll from near sideline marks

**Status:** Designed, not implemented. UX decision made.

Currently `roll0` (the camera roll relative to ground) is set manually
in the `calibrate` slider tool. The near sideline -- being a known
straight line at z=0 in field coordinates, white-painted (high
click-precision), and close to the camera (high px/m) -- is a strong
constraint that should be able to back out the camera roll
automatically during markfield.

**UX decision (confirmed with user):** Fit only on "fit" key press,
not live as the user clicks. A large jump on fit is already expected
from the LSQ refit, so the additional roll change is not jarring.

**Algorithm sketch:**

1. After N near-sideline marks placed (N >= 4), back-project each
   click through the current dewarp model to get its 3D ray direction
   (use `pano.src_to_direction()` or equivalent).
2. Those rays should all lie on a single vertical plane passing
   through the camera origin (the sideline's plane). The plane's
   normal vector encodes the camera's roll relative to that sideline.
3. Fit the plane via SVD of the ray matrix (smallest singular vector
   is the plane normal).
4. Compute the rotation that aligns the fitted plane's normal with
   the world's "across-the-field" direction at the camera's current
   yaw. The angle component around the camera's forward axis IS the
   roll correction.
5. Apply as a delta to `cfg.roll0` in `project.json`.

**Stretch:** the same machinery can derive `pitch0` from both
sidelines (their plane normals' difference encodes camera tilt) and
yaw from corner-to-corner constraints. Probably not needed if roll
alone gets the sidelines visually horizontal post-fit.

**Where to implement:**

- New helper in `waruka/ground.py` (e.g. `fit_camera_roll_from_marks`)
  that takes the pano model + near-sideline marks and returns a
  roll-correction angle.
- Call it from `waruka/markfield.py` inside the existing fit-key
  handler, before `refine_homography`. Apply the roll delta to
  `cfg.roll0`, then let `refine_homography` do the LSQ fit on top.

**Risk:** the camera model isn't perfect at the lens edges. Per
[feedback_marks_over_camera_model.md] the marks should always win
over model assumptions; the auto-fit must not over-correct based on
edge-distorted marks. Mitigation: weight near-sideline marks by their
distance from the camera (closer = more trusted), same way the
existing MLE weights work in `compute_mle_weights`.

**Why it's worth doing:** Currently `calibrate` is the only place
`roll0` is set, and re-mounting the camera (different match, different
sideline) silently invalidates it. Auto-fit during markfield makes
re-calibration a non-event for new recording sessions.

## Regenerate user-facing PDFs + screenshots post-rename

**Status:** DONE (v1.0.0). PDFs regenerated; screenshots verified clean.

The Vemos -> Waruka rename pass updated the `.md` sources but left the
binary `.pdf` versions (`docs/cli_reference.pdf`,
`docs/gui_walkthrough.pdf`) saying "Vemos" throughout.

**Resolved:**

- The PDF generator (previously a scratch script in `dev/misc/`) was
  promoted to a tracked, SPDX-headed `scripts/build_docs.py`. Run
  `python scripts/build_docs.py` to rebuild both PDFs from their `.md`
  sources. Verified: 0 "Vemos" occurrences post-regen.
- Fixed stale doc cross-references along the way: `gui_walkthrough.md`
  image paths (`_doc_screenshots/` -> `screenshots/`) and the
  `_cli_documentation.md` / `_gui_documentation.md` cross-links.
- Screenshots in `docs/screenshots/` were checked and are cropped
  tab-content captures with no window title bar -- no "Vemos" text is
  visible, so no re-capture was needed.

**Remaining (optional):**

- Wire `scripts/build_docs.py` into `scripts/build_exe.py`'s prereq
  check so a stale PDF can't ship.
- A screenshot-capture script (there has never been one in the repo);
  only needed if a future GUI change makes the current shots stale.

## Auto-detect playing field from cones

**Status:** Designed, not implemented. Medium complexity.

Most matches have **8 orange/red cones** on a green grass background:
4 at the field corners + 4 at the endzone-line corners. Cones are
static (or near-static -- occasionally kicked) for the whole match.
This is a strong, almost-tagged signal that calibrate + markfield
could exploit to skip the manual marking step entirely.

**What to build:**

A new pipeline stage / CLI command that, given a video, returns the
8 cone centroids in raw pano pixel space (then handed to the existing
markfield homography fit):

1. Sample N frames across the match (e.g. 30 frames over 10 minutes).
2. Per frame, HSV-threshold for orange/red against green grass.
   Adaptive thresholds based on a few-second average to handle sunny
   vs cloudy lighting.
3. Connected-components -> blob centroids per frame.
4. Multi-frame averaging: keep only blobs that persist at ~the same
   pano location across most frames (filters players' jerseys,
   spectator gear, sideline bags, kicked cones).
5. Cluster the surviving centroids; you should get 8 stable clusters.
6. Geometric assignment: use field dimensions + the camera's expected
   sideline-mid position to assign each cluster to its corresponding
   field landmark (NW corner, near-endzone NW corner, etc.).
7. Hand the 8 (pano_x, pano_y, world_x, world_y) pairs to the existing
   `refine_homography` LSQ fit.

**Where to implement:**

- New module `waruka/autodetect_marks.py` with
  `detect_field_cones(video_path, sample_frames=30) -> list[Mark]`.
- New CLI command in `waruka/__main__.py`:
  `python -m waruka autodetect-marks <video> --project P.json`.
- Optional GUI affordance: a "Detect cones" button in the markfield UI
  that prefills the marks for user confirmation, instead of
  fully-automatic. Hybrid approach lets the user catch lighting-edge
  failures before they corrupt the calibration.

**Risk:**

- Lighting variation -- sunny vs cloudy, golden hour -- can shift
  perceived cone colour significantly. Per-clip threshold adaptation
  is essential; a static HSV range will fail.
- Cones get partially or fully occluded by players multiple times per
  match. Multi-frame median (not mean) helps; the user-noted
  "near-static" property does the heavy lifting.
- Cones get kicked. The multi-frame stability filter should drop
  outliers, but if a cone is kicked early and never replaced, the
  count drops below 8 and the geometric assignment fails -- need a
  fallback to "fewer marks, still fit homography".
- Some leagues use yellow cones, some pink. Make the colour configurable.

**Why it's worth doing:**

- Fire-and-forget setup for routine matches -- no human in the loop.
- Prerequisite for the live-streaming roadmap; live mode needs
  unattended calibration since the operator can't be at the laptop.
- Useful on-site quick setup before a tactics-review pass without
  needing the full calibrate / markfield interactive UIs. The raw
  video can still be re-calibrated more precisely afterwards.

## Auto-detect score from lineup formations

**Status:** Designed, not implemented. Medium complexity. Won't be
100% accurate but should catch most goals.

The insight: after a goal, both teams line up at their respective
endzone for the pull. The lineup formation -- a dense cluster of
players at one endzone line + another dense cluster at the opposite
endzone line -- is visually distinctive vs gameplay (where players
are spread across the field). The transition INTO this formation is
a strong signal that a point ended; the direction of the cluster's
formation tells you which endzone scored.

**Algorithm sketch:**

Post-process pass over `tracks.json` (or `players_<N>.json` from the
classifier):

1. Per-frame compute aggregate stats over the framing-pool tracks:
   - `x_std` -- standard deviation of player X coordinates
     (lengthwise). Low = clustered, high = spread out.
   - `x_bimodality` -- e.g. dip statistic. High = two clusters, low
     = one or none.
   - Cluster centroids when bimodality > threshold.
2. Define "lineup-formation" state: `x_std < L*0.3` AND
   `x_bimodality > T` AND both centroids near opposite endzone lines.
3. Detect state transitions into lineup-formation from gameplay
   (where `x_std > L*0.4` typically). The transition timestamp is the
   approximate goal moment.
4. Scoring team direction: in the 5-10 seconds BEFORE the lineup
   forms, which endzone did the player density move toward? That's
   the scoring team's endzone.
5. Emit a goals list: `[(timestamp, scoring_end, confidence), ...]`.

**Where to implement:**

- New module `waruka/score_detect.py`.
- New CLI command:
  `python -m waruka detect-goals tracks.json --project P.json --out goals.json`.
- GUI integration: the Concat tab or a new Highlights tab could
  surface the goals list as navigation chapters in the broadcast
  output. Pair with `waruka interpolate` to make a "goals only"
  highlight reel automatically.

**Risk + edge cases:**

- **Non-goal pauses** (injury, time-out, OB hard catch dispute) also
  trigger lineup-like formations. Without disc tracking we can't
  distinguish. Output confidence should be reduced when the
  preceding-state pattern is ambiguous.
- **Half-time / time-out**: extended lineup-like formations that
  aren't goals. A duration-based filter (lineup-formation lasting
  >2 minutes is probably half-time) can mask these.
- **Spread offence pulls**: some teams pull with a horizontal stack
  near the brick mark rather than a tight endzone line. Detection
  threshold needs to be tunable, not hard-coded.
- **Substitution mid-point**: rare but possible -- one team's lineup
  formation can briefly appear without a goal. Confidence weighting
  helps; the bimodality requirement (BOTH teams lined up at OPPOSITE
  endzones) catches most cases.

**Why it's worth doing:**

- Jump-to-goal navigation in long match recordings. A 90-minute match
  becomes 12 navigation anchors instead of one long timeline.
- Half-time identification for free as a side effect.
- Foundation for downstream "auto-highlight reels" (interpolate +
  cut around each detected goal +/- 20s).
- The hard alternative -- actually tracking the disc -- is genuinely
  hard (small, fast, often occluded), so a downstream-formation-based
  proxy is a pragmatic substitute.

## Radar / minimap view

**Status:** Designed, not implemented. Medium complexity. Most of the
data already exists -- this is mostly a rendering pass.

A top-down stylised field with player dots, FIFA-radar style. Useful
as a tactics-review view alongside the broadcast: at a glance you see
the whole field's state, including off-camera players that the
auto-broadcast crop has chosen to leave out of frame.

**Data we already have:**

- Per-frame player positions in field (X, Z) coordinates -- from
  `tracks.json`.
- Per-frame classifier labels (player / sideline / foreign / probation)
  from `players_<N>_labeled.json`.
- Field dimensions in `project.json`.

**Data we don't have (yet):**

- **Team assignment** (red vs black, light vs dark). The classifier
  groups all players together. Solving this is the main extra work
  this feature needs. Approach: per-track, sample box pixels at every
  detection frame, compute a robust jersey-colour signature
  (HSV mode), then 2-means cluster across all player tracks. Cluster
  centroids name the teams.
- **Disc position**. Defer -- explicitly out of scope; the radar
  works without it.

**What to build:**

- New CLI: `python -m waruka radar tracks.json --project P.json --out radar.mp4`
- Output is a 1920x1080 (or 1280x720) video, same fps as the source,
  showing:
  - A clean stylised field outline (white-on-green) at the canonical
    field dimensions, including endzone lines + brick marks.
  - Player dots coloured by team (after the team-clustering step)
    AND/OR by classifier label (faded for inactive / probation).
  - Optional short trails (~1 second of recent positions) for motion
    cues.
  - Optional jersey numbers / track IDs as small text labels (start
    OFF; expensive on render and rarely needed).
- Optional ffmpeg post-step that picture-in-picture composites the
  radar into the bottom-right of the broadcast.

**Where to implement:**

- New module `waruka/radar.py` for the per-frame draw.
- New CLI entry in `waruka/__main__.py`.
- New jersey-classifier helper -- could live in `waruka/classify.py`
  (it sits in the classifier's natural workflow), exposed as a new
  per-track `team` field on the labelled output.

**Why it's worth doing:**

- Tactics analysis directly answerable from a single panel: "where
  was the deep cutter when the disc swung wide?" The broadcast crop
  doesn't show all 14 players simultaneously by design -- the radar
  does.
- Pairs naturally with the `interpolate` highlight workflow:
  goal-clip + radar inset is a complete analytical artifact.
- Modest engineering -- a few days of work versus weeks for richer
  analytics features.

## Cloud-deployable processor

**Status:** Designed, not implemented. Big lift -- Linux port is the
main blocker.

The use case: people without a CUDA-capable GPU (or sufficient
patience for CPU-only processing) should be able to upload their
match, pay a cloud provider for an hour of GPU time, and get the
broadcast back.

**Recommended path: Docker image + deployment guide.** OSS-friendly,
no service to operate, user pays cloud cost directly. Avoids the
auth/billing/queue/storage overhead of a hosted SaaS.

**What to build:**

1. **Linux port of the CLI.** Most of the codebase is already portable
   via `pathlib.Path`. The known Windows-isms:
   - `waruka/nvdecode.py` loads `cudart64_12.dll` by name; Linux uses
     `libcudart.so.12`. Add a platform branch.
   - Any `os.add_dll_directory()` calls (in the launcher and nvdecode)
     are Windows-only; on Linux use `LD_LIBRARY_PATH` injection.
   - PyInstaller bundle: skip on Linux -- the Docker image IS the
     distribution mechanism. Just `pip install -e .` inside the image.
2. **Docker image.** Base: `nvidia/cuda:12.1-runtime-ubuntu22.04`.
   Layers: Python 3.13, requirements.txt, waruka source, weights.
   `~3 GB` image. Push to ghcr.io / Docker Hub.
3. **Headless invocation.** The CLI doesn't need a display server, but
   `calibrate` / `markfield` do (cv2 windows). Cloud workflow:
   user pre-calibrates locally, ships only `project.json` to the
   cloud machine alongside the video. Cloud machine runs
   `pipeline` only.
4. **Deployment guide** for one or two cost-effective providers.
   Candidates worth evaluating:
   - **RunPod / Vast.ai** -- $0.20-0.50/hr for consumer 30xx/40xx
     class GPUs. Cheapest. Less reliable / preemptible.
   - **Lambda Labs** -- $0.40-1/hr similar; better reliability,
     simpler UX.
   - **AWS g4dn.xlarge (T4 16GB)** -- $0.526/hr on-demand. Mainstream,
     reliable, expensive.
   - **Google Cloud L4 (24 GB)** -- $0.81/hr, well-suited for the
     interpolate stage where VRAM matters.
   Pick **one** to document the happy path; mention the rest.
5. **Storage flow.** Source video to cloud (uploads / pre-signed URLs);
   broadcast output back (downloads / pre-signed URLs). Keep this in
   the user's hands -- don't run a storage tier.

**Per-match cost estimate (pessimistic 4h, 100-min match):**

| Provider | $/hr | Per match |
|---|---|---|
| RunPod consumer RTX 3090 | 0.20 | 0.80 |
| Lambda A10 | 0.75 | 3.00 |
| AWS T4 | 0.53 | 2.12 |
| GCP L4 | 0.81 | 3.24 |

`$1-3` is a reasonable price point for casual users -- equivalent
to a single coffee per match.

**Risk + open questions:**

- **GPU support matrix.** The cu118 torch wheel ships SM 5.0-9.0. T4
  is 7.5, L4 is 8.9, A10 is 8.6, consumer 30xx is 8.6, 40xx is 8.9.
  All covered. Blackwell (50xx, A100/H100 = 8.0/9.0) covered. Good.
- **NVENC availability.** Free / consumer tier providers may
  restrict NVENC concurrent sessions; cloud-server GPUs (T4, L4, A10)
  generally have it unrestricted. Verify on the chosen provider.
- **Linux build of `imageio-ffmpeg`** ships an LGPL ffmpeg; should
  Just Work for our encoder targets (h264_nvenc / libx264).
- **Interactive calibrate / markfield** can't run on a headless cloud
  box. The cloud path is "calibrate once locally, ship project.json,
  process forever in the cloud." Document this clearly.

**Why it's worth doing:**

- Removes the hardware barrier to entry. A GTX-1060-bound user with
  a $5 cloud budget can process a season in a weekend.
- Pairs with the future live-streaming roadmap: live streams could
  push to a cloud worker rather than requiring the user's PC to be
  on at game time.
- A clean Docker image is also useful for the dev workflow -- contributors
  can iterate without a Windows VM.

## Resume from checkpoint

**Status:** Designed, not implemented. Medium complexity.

Long renders that die mid-flight (power blip, OOM, accidental kill,
NVENC driver hiccup) currently restart from scratch. With a full
match render at ~3.6 h, a death at hour 2 costs you both halves.
Track and pipeline-mode already chunk their work; the missing piece
is "detect existing chunks at startup and skip them."

**What to build:**

- Per-stage: emit a small `_checkpoint.json` in the artefact dir after
  each completed chunk, listing the chunks done + their output paths.
- On restart, read `_checkpoint.json`, skip chunks already present
  with valid (non-zero-byte, parseable) output, resume from the
  first missing.
- Add a `--resume` flag (or make it default-on with `--no-resume` to
  opt out). When `--resume` is on AND a `_checkpoint.json` exists,
  print "resuming from chunk N of M" instead of "starting fresh."

**Where to implement:**

- `waruka/pipeline.py` already does chunk splitting -- extend it to
  read/write the checkpoint file.
- `waruka/track.py`, `waruka/render.py` -- add per-chunk emission and
  startup detection.
- GUI -- on next launch after a crash, surface "resume <project>?"
  inline.

**Risk:** stale chunks from an aborted run with different params get
silently reused. Mitigation: include a hash of the relevant args
(stride, rows, conf, model name) in the checkpoint header; mismatched
hash = invalidate + restart fresh.

**Why it's worth doing:** Reliability. Long renders are the painful
ones to lose. This is the single feature that turns "leave it running
overnight" from gamble to default.

## Match metadata sidecar

**Status:** Designed, not implemented. Low complexity.

Currently a project has only `project.json` (camera calibration). The
match itself -- date, opponent, venue, final score, etc. -- isn't
tracked anywhere. As a result, output filenames are generic
(`broadcast.mp4`) and the user has to manually rename / file each
output.

**What to build:**

- A `match.yaml` sidecar in each project's artefact dir with
  free-form fields:
  ```yaml
  date: 2026-08-15
  team: Waruka
  opponent: Toronto Goat
  venue: GHO Park field 3
  format: 7v7 outdoor
  final_score: { team: 15, opponent: 13 }
  notes: |
    Sectional final. Bid back from 9-12.
  ```
- Optional in the GUI: a "Match details" form on the Track tab that
  populates / edits this file.
- Used in:
  - Output filename: `Waruka-vs-Toronto-2026-08-15.mp4`
  - MP4 metadata (`-metadata title=...` in the ffmpeg encode step) so
    YouTube auto-titles look right.
  - Optional burnable scoreboard overlay if scores are present.

**Where to implement:**

- New module `waruka/match_meta.py` for read/write.
- Hook into `waruka/render.py`'s output filename construction.
- Hook into `waruka/gui.py`'s Track tab.

**Why it's worth doing:** Cheap polish that makes long-term match
libraries actually browsable. Coaches archiving years of footage
benefit disproportionately.

## Practice-clip extractor

**Status:** Designed, not implemented. Medium complexity. Compounds
with auto-score detection + future annotation work.

Given a list of `(timestamp, tag, duration)` tuples, auto-extract
short clips per category. Pairs naturally with auto-score detection
("export all 12 goals as a single highlight reel") and with whatever
external annotation tool the user uses (`insights.gg` etc).

**What to build:**

- CLI: `python -m waruka clips broadcast.mp4 --timestamps clips.json --out highlights/`
- Input format:
  ```json
  [
    {"t": 132.5, "duration": 8, "tag": "deep cut score"},
    {"t": 487.2, "duration": 12, "tag": "force break"}
  ]
  ```
- Per clip: ffmpeg seek-and-cut + optional re-mux (smart-cut at
  keyframes where possible, transcode only at boundaries).
- Output: individual `.mp4` files named by tag + index, plus an
  optional concat output as a single reel.

**Where to implement:**

- New module `waruka/clips.py`.
- Reuses `waruka/jobqueue.py`'s ffmpeg invocation helper.

**Integration angles:**

- **From auto-score detection:** consume the goals list directly.
- **From an external annotation tool:** import `insights.gg` /
  similar export formats (if they offer one; worth checking the API).
- **From an in-app annotation layer** (if ever built): consume the
  bookmarks list directly.

**Why it's worth doing:** Tactics review benefits from short clips
the coach can replay 5 times in a row, not from scrubbing through a
90-minute timeline. Pairs naturally with whatever upstream produces
the timestamps.

## Possession statistics (low priority)

**Status:** Designed, deferred pending disc tracking feasibility.

The clean version of possession stats requires knowing where the
disc is per frame -- "team X has possession at time T means team X
has the disc." Without disc tracking, possession is an inference
problem from team positions (who is closer to where the disc-thrower
last was, who is in throwing posture, etc), and the inference is
noisy.

**Inference-only fallback (no disc tracking required):**

- "Team in possession" approximated as "the team with more players in
  the offensive half over the last N seconds" -- works during normal
  flow, fails on transitions and stall counts.
- Combine with stack identification: the team RUNNING the stack
  pattern is on offence.

**Risk:** Without disc tracking, accuracy is probably 70-80% --
useful for high-level statistics but not for individual play review.

**Recommendation:** wait until disc-tracking feasibility is known. If
disc tracking lands, possession is trivial. If it doesn't, decide
whether the inference-only version is worth shipping.

## Force / stack identification

**Status:** Designed, not implemented. Hard problem, ultimate-specific.

Detect:

- **Defensive force direction** (which side of the offensive handler
  is being denied). Cue: relative position of the marker (1-3m from
  handler, deliberately on one side) and the body orientation of
  the marker.
- **Offensive stack pattern**: vertical (players lined up downfield
  of the disc, single file) vs horizontal (players spread side-to-
  side at one depth) vs split-stack variations.

**Algorithm sketch:**

- Per-frame, identify the handler (the player nearest the disc -- or
  if disc tracking isn't available, the player who appears most
  stationary while a defender is right next to them).
- Identify the marker (the defender within 3m of the handler with
  hips/shoulders rotated toward them).
- Force direction = the vector from marker to handler, normalised to
  the field's break/home convention.
- Stack pattern = clustering analysis on the offensive team's
  positions: pairwise distance histogram, principal component of
  the cluster.
- Smooth over short windows (5-10s) so the output isn't frame-flicker.

**Where to implement:**

- New module `waruka/tactics.py`. Run after `classify` /
  team-classification.
- Output: per-frame `(force_direction, stack_type, confidence)`.

**Prerequisites:**

- **Team classification** (red vs black) -- listed under the radar
  feature; necessary here too.
- **Persistent player IDs** -- see architectural note below; without
  these the force/stack labels stutter at every ID switch.

**Why it's worth doing:**

- Genuinely novel in this product space. No commercial product
  targets ultimate this deeply.
- Coaches care -- "did we hold force on 80% of points" is a real
  conversation that today requires hand-counting.

## Disc tracking (investigation)

**Status:** Feasibility investigation, not implementation. Hard
problem, transformative if it lands.

The user has anecdotal experience that yolo11 + Duo 2 struggles with
the disc due to small angular size, high speed, and shape variability
(end-on view vs face-on view). This is consistent with general
small-object-detection literature.

**Approaches worth evaluating, easiest first:**

1. **Motion-based detection.** The disc creates a 2-3 pixel motion
   blob at frame rate. Subtract consecutive frames + threshold +
   filter by velocity / trajectory continuity. Cheap, doesn't need
   training data, but fails on stationary disc and noisy with player
   motion.
2. **Specialised small-object YOLO.** Fine-tune a small detection
   head specifically for the disc class on a couple thousand
   labelled frames. Requires labelling effort (a tagging session
   could generate ~5000 frames in a day) but tractable.
3. **Multi-frame tracking with Kalman + motion prior.** Even an
   imperfect per-frame detector becomes more reliable when you
   constrain by frisbee dynamics (parabolic flight, hover phases,
   high-spin curves). The disc moves predictably between throws.
4. **Hybrid:** detector for hover / hand-held phases, motion-based
   for in-flight. Different physics, different best techniques.

**Plan:**

- Spend a week on a v0 motion-only detector. Render results on the
  debug-pano with disc trace. If it visibly tracks 50%+ of in-flight
  frames, the foundation is there.
- If v0 is encouraging, label ~2000 disc frames (UI tool: click the
  disc centre on each, ~1 disc/sec of labelling time) and train a
  small specialised model.
- If v0 is hopeless, write up the negative result in the backlog
  entry and close.

**Why it's worth doing:**

- Unlocks possession statistics directly (no team-inference needed).
- Unlocks per-throw analytics: throw distance, throw type
  (forehand / backhand from release motion), turnover detection
  (who released, where was nearest defender at release).
- Enables true highlight-quality clips automatically -- "the disc
  flew from A to B, follow it."
- Even if it only works on in-flight (not hover), that's enough for
  most tactical insights.

## Field heat maps

**Status:** Designed, blocked on point segmentation. Add to backlog
proper once auto-score detection lands.

Per-team or per-player occupancy density, plotted on a top-down
field. Trivial to compute from per-frame ground positions.

**Blocker:** ultimate alternates field direction every point. A
single heat map over the full match would be meaningless ("we mostly
played near the lines"). Need to either:

- Split by point (requires auto-score detection to find point
  boundaries), then average over per-point heat maps with the field
  flipped so "our endzone" is always the same end.
- Or require the user to annotate point boundaries by hand
  (unappealing).

**Prerequisite:** Auto-score / point-boundary detection. Build that
first, this becomes a quick downstream feature.

## Vertical 9:16 output (low priority)

**Status:** Designed, low priority.

User already has a manual workflow for vertical clips for
social-media sharing. Native support would shave a few minutes off
that workflow but isn't a high-value addition. Listed here so it's
captured.

## insights.gg integration (light)

**Status:** Open question -- depends on whether insights.gg has a
usable export / API.

The user already uses insights.gg for annotation. Rather than
re-implement an in-app annotation layer, integrate with insights.gg:

- If insights.gg has a clip-export API, import the exported list as
  input to the practice-clip extractor.
- If it has a "timestamped notes" export, surface those as
  navigation chapters in the broadcast output (MP4 chapter markers).
- Both are read-only integrations -- no write path needed.

**Action:** Check insights.gg's API docs. If they have one, the
integration is ~half a day. If not, close this entry.

## Smoother Panini-projection transitions (#40)

**Status:** Tuning, not implementation. Panini already exists in the
renderer; the values need adjustment for smoother visual feel.

The cylindrical/rectilinear blend is configured by
`projection_blend` (currently 0.3) and the underlying Panini
mathematics in `waruka/projection.py`. The transition between
projections during pan/zoom changes can look slightly "jumpy"; the
fix is in the blend parameter and the framing-change smoothing, not
in re-implementing the projection.

**What to build:**

- Identify the specific transitions that feel jumpy. Likely candidates:
  - hfov changes during rapid framing pulls
  - yaw changes through extreme yaw values where Panini's distortion
    profile shifts most aggressively
- Tune `projection_blend` and the campath smoothing constants
  (`smooth_t`, `lookahead_s`, the `_smooth_signal` deadzones) so the
  perceived motion through the Panini blend stays continuous.
- Possibly add a `--projection-blend-curve` mode that varies the
  blend with hfov (more cylindrical at wide framings, more
  rectilinear at narrow) instead of a single fixed blend.

**Why it's worth doing:** Pure cosmetic, but the current behaviour is
noticeable enough that the user flagged it explicitly. The render
quality is the user-facing surface; small polish here is felt every
time the broadcast is watched.

## Field overlay rendered onto broadcast (#46)

**Status:** Designed, not implemented. Pure rendering work.

Burn the field's lines (sidelines, endzone lines, brick marks) onto
the broadcast output so geometric context is unambiguous. Useful for
both tactics review (where exactly is the disc relative to the
endzone?) and for highlighting that the pipeline IS correctly
calibrated (visible alignment confirms the homography).

**What to build:**

- Use the existing `world_to_view` projection to map known field
  landmarks into per-frame pano-pixel coordinates, then into the
  cropped output frame.
- Render thin white lines for sideline / endzone / brick marks with
  a slight glow.
- Toggleable via a new render flag (`--field-overlay`) and a GUI
  checkbox in the tracking-params dialog.

**Risk:** Lines look great when the calibration is accurate and
horrible when it's slightly off. Probably auto-disable the overlay
if `far_rms > 5m` or some similar quality signal.

**Why it's worth doing:** A calibration confidence signal as a side
effect, plus it makes broadcast output look more like a TV product
than a debug overlay.

## Markfield loupe / zoom-on-click (#39)

**Status:** Designed, not implemented. Pure UX.

When marking field corners and sideline points, sub-pixel accuracy
matters -- a 0.5 px click error at the far sideline translates to
several metres of ground error. A loupe / zoom overlay during click
placement would let the user nail the centre of the cone /
intersection without zooming the whole canvas.

**What to build:**

- During mark placement, a small 5-10x magnified circular overlay
  follows the cursor showing pixel-level detail under the click
  point.
- A crosshair at the centre of the loupe so the user sees exactly
  what pixel will be marked.
- Toggleable on/off (some users prefer the whole-canvas view).

**Where to implement:** `waruka/markfield.py` -- adding a paint pass
that renders the loupe over the current preview frame when the
relevant mode is active.

**Why it's worth doing:** Calibration quality is gated by mark
precision. The marks-win principle assumes marks are precise; the
loupe makes that easier to live up to.

## TensorRT inference path (#44)

**Status:** Designed, not implemented. Performance work; meaningful
speed gains expected.

Compile the YOLO11n detection model (and possibly RIFE / FILM) through
TensorRT for ~2-3x faster inference. TensorRT specialises kernels
for the actual model + input shape combination, eliminating overhead
from generic PyTorch ops.

**What to build:**

- One-time TensorRT engine compilation step that produces a `.trt`
  file specific to the GPU model (engines aren't portable between
  GPU generations).
- Runtime loader that detects the GPU + tries the matching `.trt`
  engine; falls back to torch if missing.
- CLI flag: `--engine trt` (or `--engine torch` to force the slower
  path for debugging).
- Bundled engine files per supported GPU? Probably not -- too many
  combinations. Generate on first run if needed and cache.

**Risk:**

- TensorRT engines are NOT portable between driver versions, CUDA
  versions, or GPU SM versions. Caching strategy must invalidate
  reliably.
- TensorRT install overhead in the Docker image / build is
  substantial.
- The numerical output can differ slightly from torch fp16; need to
  verify detection recall hasn't regressed on the test clips.

**Why it's worth doing:** The track stage is the biggest wall-time
contributor in the pipeline. A 2-3x speedup there meaningfully
shrinks the full-match time and brings live mode within reach.

## Render-fps verification (#36)

**Status:** Investigation, not implementation.

The documented render-fps numbers (e.g. "~25 fps on RTX 2080 Ti
debug-pano") are anecdotal benchmarks from old test sessions. We
don't know if current code still hits those numbers, or if changes
to the renderer / GpuRenderer / SR integration over v0.15 -> v1.0
have regressed perf without anyone noticing.

**What to build:**

- A microbenchmark script under `scripts/` that runs the render
  loop on a short fixed clip and emits per-stage timing (decode,
  GpuRenderer, encode, etc.).
- Compare against the documented numbers; flag any regression.
- Run as a pre-release check.

**Why it's worth doing:** Catches perf bugs that don't break tests
but degrade the user experience. Especially relevant before live
mode work, where 1-2 fps lost in render is the difference between
"works" and "doesn't."

## FILM backend performance investigation

**Status:** Investigation, not implementation.

The FILM-Style interpolation backend runs ~4x the cost of RIFE. The
GUI shows a startup warning when FILM is picked. Whether the 4x is
inherent to the model or just an unoptimised inference path isn't
known.

**What to build:**

- Profile a single FILM dt to localise the bottleneck (Python
  overhead? Specific layer? Memory bandwidth?).
- Try the obvious optimisations: half precision, larger batch, ONNX
  export + TensorRT (related to #44).
- If it stays at 4x with no further headroom, document that as
  "FILM is inherently expensive, here's why" and leave it.

**Why it's worth doing:** FILM produces slightly better-looking
interpolation than RIFE for certain motion types. If 4x can be
reduced to 2x it becomes a tractable opt-in; today the 4x cost is
too steep for routine use.

## GUI debug surface improvements (#34)

**Status:** Partially addressed by this session's file-logging work;
remainder still open.

The file logging in [scripts/waruka_launcher.py](scripts/waruka_launcher.py)
now captures subprocess output reliably even in windowed mode, which
fixed the "(no output captured)" failure dialog in the Track tab.
What's still missing:

- A built-in log viewer in the GUI (not just "open the file in
  Notepad"). Tail-style live update during long-running jobs.
- A "diagnostic dump" button on the GUI that bundles the most
  recent log file + project.json + calibration into a zip for
  easy issue-reporting.
- Inline error annotations on the Pipeline-step cards (show the
  last error line directly on the card instead of in a popup).

**Why it's worth doing:** OSS users will report issues. A
diagnostic-dump bundle reduces the "send me the error" back-and-forth
to one button click.

## Concat tab intermediates: filename consistency

**Status:** Cleanup, not feature work.

The concat tool's intermediate filenames (`out_concat.concat.txt`,
the temporary trimmed-and-encoded chunks it produces during
concat-with-trim) don't follow the same naming conventions as the
rest of the pipeline's intermediates. Hard-to-find when debugging,
inconsistent on disk.

**What to build:**

- Audit the concat tool's intermediate output paths.
- Rename to match the per-stage `_artefact_dir/<stem>_<stage>.<ext>`
  pattern used by the rest of the pipeline.
- Update the concat-success message in the GUI to point at the right
  files.

**Why it's worth doing:** Reduces "where did that file go?" friction
when debugging. Small polish, accretes over time.

## Live mode (future scope)

**Status:** Future scope. Years out, but worth capturing the shape.

Process the camera feed in real time rather than from a recorded
file. Output: a continuously updated broadcast crop that's only a
few seconds behind reality. Use cases include in-game tactics review
between points, sideline coaching dashboards, livestreaming with
auto-broadcast camera.

**What this requires (rough order, each item is a meaningful chunk
of work):**

1. **Camera ingestion**: pull live HEVC from Reolink Duo 2 via RTSP
   directly into NVDEC, skip the file-on-disk step.
2. **Sliding-window pipeline**: track/classify/campath/render all
   need to run on a moving N-second window, not a complete clip.
   This is a fundamental rewrite of the chunked pipeline mode in
   `waruka/pipeline.py`.
3. **Unattended calibration**: auto-detect field from cones (already
   in this backlog) is the prerequisite -- the operator can't
   manually calibrate live.
4. **Latency budget**: target ~5 second behind-real-time. Hardware
   has to be fast enough; current track stage at 5 fps source-frame
   throughput would mean the live output runs at 1/4 speed of the
   game, which doesn't work. TensorRT (#44) is on the critical path.
5. **Output**: RTMP push to a streaming service (Twitch, YouTube
   Live, or self-hosted with nginx-rtmp), OR live preview window in
   the GUI for coaching dashboards.

**Why it's worth doing:**

- Different use case unlock: "watch our own game right now from a
  better camera angle" is something no other ultimate tool offers.
- Pairs with the cloud-deployable processor + Tailscale work to
  serve the live broadcast to remote viewers.
- Forces multiple foundational improvements (TensorRT, auto-cones,
  proper async pipeline) that benefit the offline use case too.

**Don't start until:** the auto-cones, TensorRT, and cloud-deployable
items are far enough along that live mode is "compose the pieces"
rather than "invent everything from scratch."

## Trim-before-process in the Queue add-job dialog

**Status:** Designed, not implemented. Small UX gap with a clever UX
refinement.

The trim-in/out widget already exists in the Concat tab and is wired
into the concat-with-trim flow. The Queue tab's add-job dialog does
NOT expose it. As a result, when you queue several matches overnight
you can't say "skip the first 3 minutes of warmup on each."

**UX refinement (user-suggested, preferred):**

For multi-clip queue jobs, asking the user to specify trim points as
"seconds into the concatenated video" is awkward -- they'd have to
mentally compute the cumulative duration. Instead:

- Load **only the first and last clip** of the selected job into the
  scrubber.
- Let the user mark IN on the first clip (where the match actually
  starts, somewhere in the middle of that clip) and OUT on the last
  clip (where the match actually ends).
- Internally translate those two clip-local positions into the
  global `(t0, t1)` for the concatenated video.

This works because the user always loads first/last clips that
contain the genuine match start and end somewhere within them --
the trim is really "where in clip 1 does the action start" + "where
in the final clip does the action end."

**What to build:**

- Compact first/last scrubber widgets in the Queue tab's add-job
  dialog (reuse the existing scrubber from Concat).
- Map local clip times -> global concat times using each clip's
  duration (probed at add time, same `_probe_audio` style helper).
- Pass the resulting `(t0, t1)` trim points down to the queued job's
  command construction in `waruka/jobqueue.py`. The downstream
  stages (`track`, `render`, etc.) already accept `--t0` / `--t1`
  flags, so nothing past command-construction needs to change.

**Where to implement:** `waruka/gui.py` (queue add-job dialog),
`waruka/jobqueue.py` (arg-passing through `track_cmd`, `render_cmd`,
`pipeline_cmd`).

**Why it's worth doing:** Closes a real workflow gap. ~1 day fix
including the global-trim translation.

# Architectural notes

The features above interact through a few shared dependencies. Worth
calling out so they're tackled in the right order.

## Persistent player IDs across the match

Currently the tracker is tuned for **broadcast framing** -- it
maintains continuity well enough for the camera to follow players,
but accepts ID switches when a player is temporarily occluded or
leaves frame. The classifier's three-gate architecture is similarly
tolerant of fragment-length tracks.

**This works fine for the auto-broadcast use case. It does not work
for individual-player analytics.** Specifically, the following
features need persistent IDs to be useful:

- Force / stack identification (the marker becomes a different ID
  every 30 seconds; force-hold percentages would be meaningless).
- Possession statistics by player.
- Player heat maps (if per-player).
- Player distance / speed stats.
- Lineup-composition tracking (which players were on field together).

**Path forward, when these features become a priority:**

1. Add a separate "long-association" pass that runs AFTER classify.
   Inputs: the existing tracks + per-track jersey-colour signature
   (used for team classification). Output: merge tracks that share
   a jersey colour, are near each other in time, and have
   plausible-continuous trajectories.
2. Use the merged IDs as the basis for all per-player analytics.
3. The broadcast / framing pipeline keeps using the original
   fragmented tracks -- no regression.

This is a foundational gap. It's worth investing the time once,
before stacking multiple analytics features on top of unmerged
tracks.

## Team classification

Mentioned individually for the radar view, but the same machinery
is needed for: force/stack, possession, any team-segmented analytics.
The clustering approach (HSV signature per track, 2-means cluster
across all tracks) is shared infrastructure. Build it once, use it
many times.

# Considered and rejected

Captured so future brainstorming sessions don't re-propose them.
Each has a specific reason it's not worth pursuing as of 2026-06.

## All-field framing mode

**Reason:** The raw source video and the existing `--debug-pano` mode
already cover the "see everyone simultaneously" use case. Producing a
dewarped first-class all-field output would cost meaningful processing
time for marginal additional value (cleaner horizon, straighter
sidelines).

See [all_field_vs_broadcast.png](docs/screenshots/all_field_vs_broadcast.png)
for the visual comparison that informed the decision.

## Camera calibration presets

**Reason:** Per-match camera setup varies enough (distance, tilt, roll
all change between mounts) that saved presets are rarely reusable.
The occasional case of "two games on the same field back-to-back"
isn't enough to justify the UI work. The auto-roll fit feature
captures most of the same value in the cases where it matters.

## In-app annotation / bookmark layer

**Reason:** Already covered by external tools (`insights.gg` etc).
Re-implementing reinvents the wheel for an audience of one.

**Open exception:** if `insights.gg` exposes a clip-list export or
API, a read-only integration that imports their tags as input to
the practice-clip extractor would be worth ~half a day. Tracked
as a separate live entry above.

## Multi-camera fusion (two Duo 2s)

**Reason:** Budget constraint -- a second camera + mount + cabling
is hundreds of dollars per setup. The marginal benefit over the
single-camera path is small unless the goal is 3D field
reconstruction, which is not a Waruka goal.

## Player jersey-number OCR

**Reason:** Two compounding practical issues:

1. Many kits have numbers that are small, low-contrast, or stylised
   in ways that resist OCR at field distances. The Duo 2's px/m at
   the far sideline means jersey numbers can be <8 pixels tall on
   far players -- below where any practical OCR works.
2. The underlying tracker is tuned to be tolerant of ID switches
   (good for camera framing, bad for individual identity tracking).
   Even a perfect OCR pass wouldn't help on its own; you'd need to
   fix persistent IDs first (see architectural notes), and once
   that's done, team-classification + jersey-colour clustering
   provides most analytics value without needing per-player
   identification.

## AI-generated tactical commentary

**Reason:** Explicit user rejection. Not a direction the project
wants to go.
