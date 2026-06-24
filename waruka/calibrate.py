# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Stefan Lewis
"""Interactive calibration tool (OpenCV HighGUI).

Run on a machine with a display:

    python -m waruka calibrate input_video_short.mp4 --time 2.0

Two ways to get a natural broadcast dewarp, usable together:

  * Manual: drag the sliders (k1/k2 radial, hfov/vfov, pitch0/roll0) and
    watch the live preview until lines look straight and players upright.
    This is the primary path when the pitch has no clear straight markings.
  * Assisted: click points along things that are genuinely straight in
    reality (the floodlight pylon is ideal; sidelines/fences if marked),
    press F to fit the radial distortion and level the horizon. Avoid
    ragged references like a treeline on a slope.

The camera is fixed, so calibration is done once per recording and saved.

Preview-window helpers (added 2026-05-30):
  * Left-click + drag to pan (yaw/pitch); mouse wheel to zoom (vfov).
  * L toggles a translucent perfectly-horizontal reference line through
    the preview -- a visual ruler to check the dewarped horizon is level.
    Right-drag in the preview window slides the line up or down; key 0
    re-centres it. The vertical position is persisted in project.json.
  * O toggles a translucent overlay of the marked calibration lines
    reprojected into the preview -- shows how straight they actually
    are under the current k1/k2/pitch0/roll0 settings, without leaving
    the preview window.
  * Scrubbing: , and . step the video by +-1s; < and > step by +-10s.
    Useful when the reference (pylon, sideline) is briefly occluded.
    The current scrub time is persisted in project.json so reopening
    resumes where you left off.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np

from .config import ProjectConfig
from .calib import fit_distortion, level_horizon


def _move_os_cursor(screen_x: int, screen_y: int) -> bool:
    """Warp the OS cursor to a screen position. Windows-only via Win32
    SetCursorPos; returns True on success, False on non-Windows or
    permission failure. Used to sync the OS cursor with the virtual
    cursor when arrow keys nudge it."""
    if sys.platform != "win32":
        return False
    try:
        import ctypes
        return bool(ctypes.windll.user32.SetCursorPos(
            int(screen_x), int(screen_y)))
    except Exception:
        return False


def _get_os_cursor_pos() -> tuple[int, int] | None:
    """Return the OS cursor's current screen position (x, y) via Win32
    ``GetCursorPos``. Returns None on non-Windows or failure.

    Used as the authoritative source for "where the cursor actually
    is" when centering the magnifier loupe. Avoids the round-trip
    through cv2 mouse coords + getWindowImageRect, which has shown
    a sub-pixel-to-noticeable offset depending on cv2 version /
    DPI scaling / window-resize state."""
    if sys.platform != "win32":
        return None
    try:
        import ctypes

        class _POINT(ctypes.Structure):
            _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]

        pt = _POINT()
        if ctypes.windll.user32.GetCursorPos(ctypes.byref(pt)):
            return (int(pt.x), int(pt.y))
    except Exception:
        pass
    return None


# Win32 constants for the loupe styling trick.
_GCLP_HCURSOR      = -12
_GWL_STYLE         = -16
_GWL_EXSTYLE       = -20
_WS_CAPTION        = 0x00C00000
_WS_THICKFRAME     = 0x00040000
_WS_SYSMENU        = 0x00080000
_WS_MINIMIZEBOX    = 0x00020000
_WS_MAXIMIZEBOX    = 0x00010000
_WS_EX_LAYERED     = 0x00080000
_WS_EX_TRANSPARENT = 0x00000020
_LWA_ALPHA         = 0x00000002
_HWND_TOPMOST      = -1
_HWND_NOTOPMOST    = -2
_SWP_NOMOVE        = 0x0002
_SWP_NOSIZE        = 0x0001
_SWP_NOACTIVATE    = 0x0010
_SWP_FRAMECHANGED  = 0x0020
# Standard system cursor IDs used with LoadCursorW(NULL, ...). The
# arrow has its hotspot at the upper-left tip; the cross has its
# hotspot at the centre. We don't actually use the cross any more in
# magnifier mode -- _create_invisible_cursor produces a fully
# transparent one so only the loupe's drawn crosshair is visible --
# but the constants stay around for reference.
_IDC_ARROW         = 32512
_IDC_CROSS         = 32515


def _create_invisible_cursor() -> int:
    """Create a 32x32 fully-transparent HCURSOR for use as the class
    cursor while magnifier mode is on. The cursor's hit-testing /
    logical position still works (clicks register at GetCursorPos
    coords); only the visible arrow/cross disappears, so the loupe's
    drawn crosshair is the sole on-screen position indicator.

    Builds the cursor via the classic ``CreateCursor`` API with an
    AND mask of all 1s (transparent) and an XOR mask of all 0s (no
    inversion). Returns 0 on non-Windows / failure -- the caller
    falls back to leaving the cursor alone in that case."""
    if sys.platform != "win32":
        return 0
    try:
        import ctypes
        w, h = 32, 32
        n_bytes = (w * h) // 8   # one bit per pixel, 8 px per byte = 128 B
        and_bits = (ctypes.c_ubyte * n_bytes)(*([0xFF] * n_bytes))
        xor_bits = (ctypes.c_ubyte * n_bytes)(*([0x00] * n_bytes))
        hinst = ctypes.windll.kernel32.GetModuleHandleW(None)
        return int(ctypes.windll.user32.CreateCursor(
            hinst, 0, 0, w, h, and_bits, xor_bits))
    except Exception:
        return 0


def _swap_window_class_cursor_invisible(window_title: str,
                                         saved_cursor_holder: list) -> bool:
    """Swap the window class cursor to a fully invisible one. Pairs
    with :func:`_restore_window_class_cursor`. Returns True on
    success.

    Slightly different from :func:`_swap_window_class_cursor`: we
    create a custom HCURSOR (rather than loading a system one) so we
    have to remember to destroy it on revert."""
    if sys.platform != "win32":
        return False
    if saved_cursor_holder[0] is not None:
        return True
    try:
        import ctypes
        hwnd = ctypes.windll.user32.FindWindowW(None, window_title)
        if not hwnd:
            return False
        invisible = _create_invisible_cursor()
        if not invisible:
            return False
        prev_cursor = ctypes.windll.user32.SetClassLongPtrW(
            hwnd, _GCLP_HCURSOR, invisible)
        # Tuple ``(prev_cursor, our_cursor)``: prev_cursor goes back on
        # revert; our_cursor needs DestroyCursor to free GDI resources.
        saved_cursor_holder[0] = (prev_cursor, invisible)
        return True
    except Exception:
        return False


def _swap_window_class_cursor(window_title: str,
                               cursor_id: int,
                               saved_cursor_holder: list) -> bool:
    """Change the cursor for ``window_title``'s window class to the
    system cursor identified by ``cursor_id``. Saves the previous
    HCURSOR into ``saved_cursor_holder[0]`` so it can be restored.

    Affects every cv2 window of the same class because OpenCV creates
    them all under one class; the preview window picks up the same
    cursor change. That's acceptable in magnifier mode since the
    crosshair is also useful in the preview. Windows-only no-op
    elsewhere."""
    if sys.platform != "win32":
        return False
    if saved_cursor_holder[0] is not None:
        return True
    try:
        import ctypes
        hwnd = ctypes.windll.user32.FindWindowW(None, window_title)
        if not hwnd:
            return False
        new_cursor = ctypes.windll.user32.LoadCursorW(0, cursor_id)
        if not new_cursor:
            return False
        # SetClassLongPtrW returns the previous value -- exactly what
        # we want to remember for the revert. Use SetClassLongPtrW
        # rather than SetClassLongW so the call is correct on 64-bit
        # Python (HCURSOR is a pointer-sized handle).
        prev_cursor = ctypes.windll.user32.SetClassLongPtrW(
            hwnd, _GCLP_HCURSOR, new_cursor)
        saved_cursor_holder[0] = prev_cursor
        return True
    except Exception:
        return False


def _restore_window_class_cursor(window_title: str,
                                  saved_cursor_holder: list) -> None:
    """Pair with :func:`_swap_window_class_cursor` /
    :func:`_swap_window_class_cursor_invisible`. Restores the
    original HCURSOR saved by either, and destroys the temporary
    cursor we created (only the invisible variant) so GDI resources
    don't leak. No-op when nothing was saved."""
    if sys.platform != "win32" or saved_cursor_holder[0] is None:
        return
    saved = saved_cursor_holder[0]
    # Two saved-state shapes:
    #   * plain int -- previous HCURSOR (from IDC_* swap; nothing to free)
    #   * tuple (prev_hcursor, our_hcursor) -- invisible swap; our
    #     custom cursor needs DestroyCursor after the class is reverted
    if isinstance(saved, tuple):
        prev_cursor, our_cursor = saved
    else:
        prev_cursor, our_cursor = saved, 0
    try:
        import ctypes
        hwnd = ctypes.windll.user32.FindWindowW(None, window_title)
        if hwnd:
            ctypes.windll.user32.SetClassLongPtrW(
                hwnd, _GCLP_HCURSOR, prev_cursor)
        if our_cursor:
            ctypes.windll.user32.DestroyCursor(our_cursor)
    except Exception:
        pass
    finally:
        saved_cursor_holder[0] = None


def _set_window_topmost_clickthrough(window_title: str,
                                      saved_style_holder: list,
                                      size: int | None = None) -> bool:
    """Style ``window_title`` for magnifying-glass mode:

      * **topmost** -- always above the main window (HWND_TOPMOST)
      * **click-through** -- mouse events pass to the window below
        (WS_EX_LAYERED | WS_EX_TRANSPARENT, alpha=255)
      * **borderless** -- no title bar, no resize frame
        (strip WS_CAPTION/WS_THICKFRAME/WS_SYSMENU/min/max boxes)
      * **circular** (if ``size`` given) -- the window is clipped
        to an ellipse of ``size`` diameter via ``SetWindowRgn``;
        the rectangular corners become transparent

    Saves everything needed for a clean revert into
    ``saved_style_holder[0]`` as a dict: ``ex_style``, ``style``,
    ``circular``. :func:`_revert_window_styles` restores all of it.
    Windows-only no-op elsewhere."""
    if sys.platform != "win32":
        return False
    if saved_style_holder[0] is not None:
        # Already applied; nothing to do.
        return True
    try:
        import ctypes
        hwnd = ctypes.windll.user32.FindWindowW(None, window_title)
        if not hwnd:
            return False
        ex_style = ctypes.windll.user32.GetWindowLongW(hwnd, _GWL_EXSTYLE)
        style    = ctypes.windll.user32.GetWindowLongW(hwnd, _GWL_STYLE)
        saved_style_holder[0] = {
            "ex_style": ex_style,
            "style":    style,
            "circular": bool(size),
        }
        # 1) Borderless: clear caption + frame styles. SWP_FRAMECHANGED
        # tells Windows to recompute the non-client area so the title
        # bar actually disappears.
        new_style = style & ~(
            _WS_CAPTION | _WS_THICKFRAME | _WS_SYSMENU
            | _WS_MINIMIZEBOX | _WS_MAXIMIZEBOX)
        ctypes.windll.user32.SetWindowLongW(hwnd, _GWL_STYLE, new_style)
        # 2) Topmost + click-through ex_style. The layered alpha is
        # required for the window to remain visible after the
        # WS_EX_LAYERED flag is set.
        new_ex = ex_style | _WS_EX_LAYERED | _WS_EX_TRANSPARENT
        ctypes.windll.user32.SetWindowLongW(hwnd, _GWL_EXSTYLE, new_ex)
        ctypes.windll.user32.SetLayeredWindowAttributes(
            hwnd, 0, 255, _LWA_ALPHA)
        ctypes.windll.user32.SetWindowPos(
            hwnd, _HWND_TOPMOST, 0, 0, 0, 0,
            _SWP_NOMOVE | _SWP_NOSIZE | _SWP_NOACTIVATE | _SWP_FRAMECHANGED)
        # 3) Circular clip via elliptic region. SetWindowRgn takes
        # ownership of the region handle, so no DeleteObject needed.
        if size:
            region = ctypes.windll.gdi32.CreateEllipticRgn(
                0, 0, int(size), int(size))
            ctypes.windll.user32.SetWindowRgn(hwnd, region, True)
        return True
    except Exception:
        return False


def _revert_window_styles(window_title: str,
                           saved_style_holder: list) -> None:
    """Undo :func:`_set_window_topmost_clickthrough`: drop the topmost
    flag, restore the original GWL_STYLE + GWL_EXSTYLE, and remove
    the elliptic clipping region (back to rectangular). No-op when
    nothing was applied."""
    if sys.platform != "win32":
        return
    saved = saved_style_holder[0]
    if saved is None:
        return
    try:
        import ctypes
        hwnd = ctypes.windll.user32.FindWindowW(None, window_title)
        if hwnd:
            # Region back to rectangular (pass NULL to clear).
            if saved.get("circular"):
                ctypes.windll.user32.SetWindowRgn(hwnd, 0, True)
            # Styles back to originals.
            ctypes.windll.user32.SetWindowLongW(
                hwnd, _GWL_STYLE,    saved["style"])
            ctypes.windll.user32.SetWindowLongW(
                hwnd, _GWL_EXSTYLE,  saved["ex_style"])
            # Drop topmost + recompute frame so the title bar comes back.
            ctypes.windll.user32.SetWindowPos(
                hwnd, _HWND_NOTOPMOST, 0, 0, 0, 0,
                _SWP_NOMOVE | _SWP_NOSIZE | _SWP_NOACTIVATE
                | _SWP_FRAMECHANGED)
    except Exception:
        pass
    finally:
        saved_style_holder[0] = None


def _virtual_to_screen(cursor_src, scale: float, src_w: int, src_h: int,
                       win_name: str) -> tuple[int, int] | None:
    """Convert a virtual-cursor (source-pixel) position to screen
    coords using the OpenCV window's current image rect. Returns None
    if the window isn't laid out yet or cv2 doesn't support
    ``getWindowImageRect`` on this build.

    Accounts for user-resized windows: ``getWindowImageRect`` returns
    the on-screen rect of the displayed image, which may be larger or
    smaller than the natural display size if the user dragged a
    window corner."""
    try:
        rect = cv2.getWindowImageRect(win_name)
    except (cv2.error, AttributeError):
        return None
    if rect is None or rect[2] <= 0 or rect[3] <= 0:
        return None
    x_win, y_win, w_rect, h_rect = rect
    # Natural display size for the source frame at the picked scale.
    disp_w = max(1, int(src_w * scale))
    disp_h = max(1, int(src_h * scale))
    # Window may have been resized -- scale our source-px coords
    # to match the actual on-screen image size.
    px_x = cursor_src[0] * scale * (w_rect / disp_w)
    px_y = cursor_src[1] * scale * (h_rect / disp_h)
    return int(x_win + px_x), int(y_win + px_y)

HELP = [
    "Calibrate window: L-click=add pt   N=new line   U=undo   C=clear",
    "Arrows=nudge cursor 1 src px   Enter/Space=commit at cursor",
    "H=tag HORIZON   K=toggle k2 fit   F=fit radial+level",
    "Preview: L-drag=pan  R-drag=move level line  wheel=zoom  0=re-centre",
    "L=level line   O=overlay lines   ,/.=scrub +-1s   < >=scrub +-10s",
    "Z=toggle loupe   M=magnifier mode (loupe follows cursor)",
    "S=save   Q/ESC=quit",
]

# Window between SetCursorPos and the synthetic MOUSEMOVE it generates.
# During this window the calibrate mouse callback ignores movement
# events so the OS-cursor warp doesn't write back stale display coords
# to cursor_src (which would undo the arrow-nudge precision).
SETCURSOR_FILTER_S = 0.10

# OpenCV arrow key codes via cv2.waitKeyEx. Windows + Linux/X11
# return different full-width codes; both sets are accepted so the
# nudge behaviour works regardless of platform.
ARROW_LEFT_CODES  = (2424832, 65361)  # Win 0x250000, X11 0xFF51
ARROW_UP_CODES    = (2490368, 65362)  # Win 0x260000, X11 0xFF52
ARROW_RIGHT_CODES = (2555904, 65363)  # Win 0x270000, X11 0xFF53
ARROW_DOWN_CODES  = (2621440, 65364)  # Win 0x280000, X11 0xFF54

# Default preview window output size. Used by the renderer and the
# drag-pan pixel->degree conversion. Scaled down on small screens at
# session start via _fit_window_sizes(); the runtime values used in
# the loop are local variables prev_w / prev_h (not these constants).
PREV_W, PREV_H = 1100, 619

# Loupe (zoom window) magnification + crop radius in source-frame
# pixels. Matches the markfield loupe convention (5x with INTER_NEAREST
# so individual pixels stay sharp; 70px radius -> ~700px loupe window).
LOUPE_ZOOM = 5
LOUPE_RADIUS_SRC_PX = 70
# In magnifier mode the loupe shrinks (still 5x, just less area). At
# 40 src-px radius the loupe window is 400x400 -- big enough to read
# easily, small enough to not block most of the calibrate image.
MAGNIFIER_RADIUS_SRC_PX = 40


def _detect_screen_size(default: tuple[int, int] = (1366, 768)
                         ) -> tuple[int, int]:
    """Best-effort screen size in pixels. Used to scale the calibrate
    + preview windows so they fit on small laptops without forcing the
    user to drag them around. Falls back to a conservative default
    matching the smallest laptop screen we want to support (1366x768)."""
    try:
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()
        w = root.winfo_screenwidth()
        h = root.winfo_screenheight()
        root.destroy()
        if w > 0 and h > 0:
            return (w, h)
    except Exception:
        pass
    return default


def _fit_window_sizes(src_w: int, src_h: int,
                       screen_w: int, screen_h: int,
                       ) -> tuple[float, int, int]:
    """Pick a display scale + preview dimensions that fit the screen.

    Returns ``(scale, prev_w, prev_h)`` where:
      * ``scale`` is the multiplier applied to the source frame for the
        main calibrate window. The result must fit in roughly the left
        ~55% of the screen, leaving room for the preview to the right
        and the loupe below it. Capped at 1.0 so we never upscale.
      * ``prev_w`` / ``prev_h`` are the preview render size, keeping the
        original 1100:619 aspect ratio. Targeted at ~40% screen width.

    Both leave headroom for trackbars (~340px above the calibrate
    image), window chrome, taskbar, and the loupe (~750px tall when
    open).
    """
    # Calibrate-window image: aim for ~55% screen width, ~50% screen
    # height (leaves vertical room for trackbars + chrome).
    max_calib_w = int(screen_w * 0.55)
    max_calib_h = int(screen_h * 0.50)
    scale = min(1.0, max_calib_w / max(1, src_w),
                 max_calib_h / max(1, src_h))

    # Preview: ~40% screen width, keeping the original 1100:619 aspect.
    prev_w = min(PREV_W, int(screen_w * 0.40))
    prev_h = max(1, int(round(prev_w * PREV_H / PREV_W)))
    # Also clamp height to leave room for the loupe stack.
    max_prev_h = int(screen_h * 0.45)
    if prev_h > max_prev_h:
        prev_h = max_prev_h
        prev_w = int(round(prev_h * PREV_W / PREV_H))
    return scale, prev_w, prev_h


def _draw(disp, lines, cur, horizon_idx, scale, model, rms, fit_k2,
          t_now, total_seconds, show_level, show_overlay):
    img = disp.copy()
    alll = lines + ([cur] if cur else [])
    for i, ln in enumerate(alll):
        is_cur = cur and i == len(lines)
        col = (0, 215, 255) if is_cur else (
            (255, 80, 0) if i == horizon_idx else (0, 200, 0))
        pts = [(int(x * scale), int(y * scale)) for x, y in ln]
        for j, p in enumerate(pts):
            cv2.circle(img, p, 4, col, -1)
            if j:
                cv2.line(img, pts[j - 1], p, col, 1)
    lines_txt = (
        f"lines={len(lines)} fit_k2={'ON' if fit_k2 else 'off'}"
        + (f"  straightness_rms={rms:.4f} rad" if rms is not None else "")
    )
    time_txt = (f"t={t_now:6.2f}s / {total_seconds:.1f}s   "
                f"level={'ON' if show_level else 'OFF'}   "
                f"overlay={'ON' if show_overlay else 'OFF'}")
    y = 18
    for t in [lines_txt, time_txt, *HELP]:
        cv2.putText(img, t, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, t, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
        y += 21
    return img


def _draw_preview_overlays(view, model, lines, cur, yaw, pitch, vfov,
                            show_level, show_overlay, level_y_frac=0.0):
    """Apply translucent level-reference line and reprojected calibration
    lines on top of the rendered preview view.

    level_y_frac: signed fraction of height offset from centre (positive
    = below centre, negative = above). Right-drag in the preview window
    updates this; 0.0 is the natural centred position.
    """
    h, w = view.shape[:2]
    out = view
    if show_level:
        overlay = out.copy()
        cy = int(h * (0.5 + max(-0.49, min(0.49, level_y_frac))))
        cv2.line(overlay, (0, cy), (w, cy), (0, 255, 255), 2, cv2.LINE_AA)
        out = cv2.addWeighted(overlay, 0.45, out, 0.55, 0)
    if show_overlay and (lines or (cur and len(cur) >= 2)):
        overlay = out.copy()
        groups = list(lines) + ([cur] if cur and len(cur) >= 2 else [])
        for ln in groups:
            if len(ln) < 2:
                continue
            pts_arr = np.array(ln, dtype=float)
            rays = model.src_to_direction(pts_arr[:, 0], pts_arr[:, 1])
            proj = model.world_to_view(rays, yaw, pitch, vfov, w, h)
            valid = ~np.isnan(proj).any(axis=1)
            for i in range(1, len(proj)):
                if valid[i] and valid[i - 1]:
                    p1 = (int(proj[i - 1, 0]), int(proj[i - 1, 1]))
                    p2 = (int(proj[i, 0]), int(proj[i, 1]))
                    cv2.line(overlay, p1, p2, (255, 80, 0), 2, cv2.LINE_AA)
            for i, ok in enumerate(valid):
                if ok:
                    p = (int(proj[i, 0]), int(proj[i, 1]))
                    cv2.circle(overlay, p, 3, (255, 80, 0), -1, cv2.LINE_AA)
        out = cv2.addWeighted(overlay, 0.5, out, 0.5, 0)
    return out


def run_calibrator(video: str, project_path: str, t_seconds: float = 2.0):
    pp = Path(project_path)
    # Probe video dimensions for either loading or creating the config.
    cap_tmp = cv2.VideoCapture(video)
    if not cap_tmp.isOpened():
        raise RuntimeError(f"cannot open {video}")
    w0 = int(cap_tmp.get(cv2.CAP_PROP_FRAME_WIDTH))
    h0 = int(cap_tmp.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap_tmp.release()

    if pp.exists():
        cfg = ProjectConfig.load(pp)
        cfg.source_video = video
    else:
        cfg = ProjectConfig.for_video(video, w0, h0)
    model = cfg.pano

    # Keep VideoCapture open for the whole session so scrubbing is fast.
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    total_seconds = (total_frames / fps) if total_frames else 0.0

    # Resume at the previously-scrubbed time if available, else the
    # CLI argument.
    t_now = [float(cfg.last_scrub_t)
             if cfg.last_scrub_t is not None else float(t_seconds)]

    def grab_frame_at(t):
        if total_frames > 0:
            target = max(0, min(total_frames - 1, int(t * fps)))
        else:
            target = int(t * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, target)
        ok_, fr = cap.read()
        return fr if ok_ else None

    frame = grab_frame_at(t_now[0])
    if frame is None:
        cap.release()
        raise RuntimeError(f"cannot read {video} at {t_now[0]}s")
    h, w = frame.shape[:2]
    # Adaptive sizing -- scale the calibrate display + preview to fit
    # whatever screen the user is on. Was hard-coded at scale=1500/w
    # before, which produced a 1500x880 calibrate window that didn't
    # fit on 1366x768 laptops. See [#32].
    screen_w, screen_h = _detect_screen_size()
    scale, prev_w, prev_h = _fit_window_sizes(w, h, screen_w, screen_h)
    frame_ref = [frame]
    disp_ref = [cv2.resize(frame, (int(w * scale), int(h * scale)))]

    def scrub(delta_s):
        t_new = max(0.0, min(max(total_seconds - 1e-3, 0.0),
                             t_now[0] + delta_s))
        fr = grab_frame_at(t_new)
        if fr is None:
            print(f"scrub failed at t={t_new:.2f}s")
            return
        t_now[0] = t_new
        frame_ref[0] = fr
        disp_ref[0] = cv2.resize(fr, (int(w * scale), int(h * scale)))

    lines: list[list[tuple[float, float]]] = [
        [tuple(p) for p in l] for l in cfg.calib_lines
    ]
    cur: list[tuple[float, float]] = []
    horizon_idx = -1
    rms = None
    fit_k2 = False

    # Overlay toggles (persisted in cfg).
    show_level = [bool(cfg.show_level_line)]
    show_overlay = [bool(cfg.show_calib_overlay)]
    level_y_frac = [float(cfg.level_line_y_frac)]
    # Preview-window drag state for click+drag pan.
    prev_drag: list = [None]   # (x, y) while LEFT-dragging to pan
    level_drag: list = [None]  # (x, y) while RIGHT-dragging level line

    # Virtual cursor in SOURCE-frame coordinates (floats, so arrow
    # nudges of 1 source pixel work even when scale < 1.0). Drives
    # both the loupe crosshair and the point committed by L-click /
    # Enter. Mouse moves overwrite it; arrow keys nudge it. Click
    # commits at the *virtual* cursor (not the click position) so a
    # nudge applied right before clicking is preserved.
    cursor_src = [0.0, 0.0]
    # Loupe visibility toggle. Defaults on; user can hide with Z so the
    # loupe window doesn't waste screen space when not needed.
    show_loupe = [True]
    # Magnifier mode: when ON, the loupe window is centred on the
    # cursor, stays above the main window, and ignores mouse events
    # (clicks fall through to the main calibrate window underneath).
    # Toggled with M. Reproduces the feel of a real magnifying glass.
    magnifier_mode = [False]
    # Saved ex-style of the loupe window before the topmost +
    # click-through tweaks. Restored when magnifier mode is turned
    # off so the loupe behaves normally again.
    loupe_saved_style: list = [None]
    # Saved HCURSOR of the calibrate window class. Swapped to
    # IDC_CROSS in magnifier mode so the OS cursor (whose hotspot is
    # the centre of the cross) lines up with the loupe's drawn
    # crosshair. The arrow cursor's hotspot is at the upper-left tip
    # which makes the two crosses look offset.
    saved_class_cursor: list = [None]
    # Timestamp of the most recent SetCursorPos call. The calibrate
    # mouse callback uses this to swallow the synthetic MOUSEMOVE that
    # immediately follows -- otherwise the nudge would be overwritten
    # by a (display->source) round-trip with rounding loss.
    last_setcursor_t = [0.0]

    win, prev = "waruka calibrate", "waruka preview (dewarp)"
    loupe_win = "loupe (zoom)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.namedWindow(prev, cv2.WINDOW_NORMAL)
    cv2.namedWindow(loupe_win, cv2.WINDOW_NORMAL)

    # Position the three windows so they fit on the detected screen:
    # calibrate top-left, preview top-right, loupe bottom-right. The
    # user can still drag them around afterwards.
    try:
        cv2.resizeWindow(win,
                          int(w * scale), int(h * scale) + 340)
        cv2.resizeWindow(prev, prev_w, prev_h)
        loupe_side = LOUPE_RADIUS_SRC_PX * 2 * LOUPE_ZOOM
        cv2.resizeWindow(loupe_win, loupe_side, loupe_side)
        cv2.moveWindow(win, 0, 0)
        cv2.moveWindow(prev, int(w * scale) + 20, 0)
        cv2.moveWindow(loupe_win,
                        int(w * scale) + 20, prev_h + 60)
    except cv2.error:
        # Some OpenCV builds raise here on hi-DPI Windows. Non-fatal.
        pass

    # Float trackbars via (pos + lo) / div.
    specs = {
        "k1 x1000": (-1000, 500, 1000.0, model.k1),
        "k2 x1000": (-500, 500, 1000.0, model.k2),
        "hfov": (120, 260, 1.0, model.hfov_deg),
        "vfov": (40, 140, 1.0, model.vfov_deg),
        "pitch0 x10": (-450, 450, 10.0, model.pitch0_deg),
        "roll0 x10": (-150, 150, 10.0, model.roll0_deg),
        "prev_yaw": (-90, 90, 1.0, 0.0),
        "prev_pitch": (-25, 25, 1.0, 0.0),
        "prev_vfov": (25, 110, 1.0, 75.0),
    }
    div = {}
    for name, (lo, hi, d, init) in specs.items():
        cv2.createTrackbar(name, win, 0, hi - lo, lambda v: None)
        cv2.setTrackbarPos(name, win, int(np.clip(init * d, lo, hi)) - lo)
        div[name] = (lo, d)

    def get(name):
        lo, d = div[name]
        return (cv2.getTrackbarPos(name, win) + lo) / d

    def setf(name, val):
        lo, d = div[name]
        cv2.setTrackbarPos(name, win, int(np.clip(val * d, lo, lo + 1e9)) - lo)

    def on_calibrate_mouse(ev, x, y, *_):
        if ev == cv2.EVENT_MOUSEMOVE:
            # Suppress the synthetic mousemove generated by our own
            # SetCursorPos call after an arrow nudge -- otherwise the
            # warp causes a display->source round-trip that loses the
            # 1-source-pixel precision the nudge was meant to give.
            if time.monotonic() - last_setcursor_t[0] < SETCURSOR_FILTER_S:
                return
            # Update the virtual cursor to follow the mouse. Float
            # coords so the loupe + arrow-nudge stay sub-display-pixel
            # accurate.
            cursor_src[0] = x / scale
            cursor_src[1] = y / scale
        elif ev == cv2.EVENT_LBUTTONDOWN:
            # Commit at the VIRTUAL cursor, not the click position.
            # On Windows + Linux, a single arrow keypress just before a
            # click typically wouldn't fire a MOUSEMOVE -- so the click
            # would otherwise undo the nudge. Reading cursor_src here
            # picks up the nudged position.
            cur.append((cursor_src[0], cursor_src[1]))

    def commit_at_cursor():
        """Append the current virtual cursor position to cur. Wired to
        Enter / Space below; useful when the user has nudged the
        cursor with arrows and wants to commit without touching the
        mouse (which would jump the cursor back)."""
        cur.append((cursor_src[0], cursor_src[1]))

    def sync_os_cursor_to_virtual():
        """Move the OS cursor on screen to match the current virtual
        cursor. Called from every arrow-nudge handler so the OS
        cursor visibly tracks the loupe crosshair. Side effect: the
        OS will fire a synthetic MOUSEMOVE -- we stamp
        ``last_setcursor_t`` so the calibrate mouse callback ignores
        that event."""
        screen = _virtual_to_screen(cursor_src, scale, w, h, win)
        if screen is None:
            return
        last_setcursor_t[0] = time.monotonic()
        _move_os_cursor(*screen)

    def position_magnifier_loupe():
        """Centre the loupe window on the OS cursor every loop tick
        while magnifier mode is on, and ensure the magnifier styles
        (topmost + click-through + borderless + circular) are applied.
        Styles applied lazily once per magnifier-on transition;
        ``loupe_saved_style[0]`` is None until applied."""
        if not (show_loupe[0] and magnifier_mode[0]):
            return
        magnifier_side = MAGNIFIER_RADIUS_SRC_PX * 2 * LOUPE_ZOOM
        if loupe_saved_style[0] is None:
            # Resize BEFORE applying styles so the elliptic region is
            # computed at the right diameter. WINDOW_NORMAL respects
            # cv2.resizeWindow even with the title-bar styles still
            # present; once the borderless tweak fires the new size
            # is the whole on-screen footprint.
            try:
                cv2.resizeWindow(loupe_win, magnifier_side, magnifier_side)
            except cv2.error:
                pass
            _set_window_topmost_clickthrough(
                loupe_win, loupe_saved_style, size=magnifier_side)
        # Centre on the OS cursor as reported by Win32 GetCursorPos --
        # the authoritative answer to "where is the cursor RIGHT NOW".
        # Going through cv2 mouse coords + getWindowImageRect showed
        # a real offset on some setups. Fall back to the cursor_src
        # round-trip on non-Windows.
        pos = _get_os_cursor_pos()
        if pos is None:
            pos = _virtual_to_screen(cursor_src, scale, w, h, win)
        if pos is None:
            return
        sx, sy = pos
        # Clicks pass through to the calibrate window underneath
        # thanks to the WS_EX_TRANSPARENT style.
        lx = sx - magnifier_side // 2
        ly = sy - magnifier_side // 2
        try:
            cv2.moveWindow(loupe_win, lx, ly)
        except cv2.error:
            pass

    def draw_loupe():
        """Render the 5x zoom window of the area around the virtual
        cursor. Mirrors the markfield loupe pattern -- crops on the
        SOURCE frame (not the downscaled display) so the magnification
        shows real source-frame pixels at INTER_NEAREST.

        In magnifier mode the crop radius shrinks to
        ``MAGNIFIER_RADIUS_SRC_PX`` so the floating loupe stays
        compact (~400x400) rather than swamping the calibrate image."""
        if not show_loupe[0]:
            return
        sx = int(cursor_src[0])
        sy = int(cursor_src[1])
        r = (MAGNIFIER_RADIUS_SRC_PX
              if magnifier_mode[0] else LOUPE_RADIUS_SRC_PX)
        x0, y0 = max(0, sx - r), max(0, sy - r)
        x1 = min(frame_ref[0].shape[1], sx + r)
        y1 = min(frame_ref[0].shape[0], sy + r)
        crop = frame_ref[0][y0:y1, x0:x1]
        if crop.size == 0:
            return
        z = cv2.resize(
            crop,
            (crop.shape[1] * LOUPE_ZOOM, crop.shape[0] * LOUPE_ZOOM),
            interpolation=cv2.INTER_NEAREST,
        )
        cx = (sx - x0) * LOUPE_ZOOM
        cy = (sy - y0) * LOUPE_ZOOM
        cv2.drawMarker(z, (cx, cy), (0, 215, 255),
                        cv2.MARKER_CROSS, 24, 2)
        # Tiny source-coord readout in the corner so the user knows
        # where the loupe is centred (useful when re-locating a mark).
        cv2.putText(z, f"src ({sx},{sy})", (8, 18),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                     (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(z, f"src ({sx},{sy})", (8, 18),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                     (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow(loupe_win, z)

    def on_preview_mouse(ev, x, y, flags, *_):
        if ev == cv2.EVENT_LBUTTONDOWN:
            prev_drag[0] = (x, y)
        elif ev == cv2.EVENT_RBUTTONDOWN:
            # Start dragging the level reference line. Only meaningful
            # when it's visible; if hidden, this is a no-op until L is
            # pressed to show it.
            level_drag[0] = (x, y)
        elif ev == cv2.EVENT_MOUSEMOVE:
            if prev_drag[0] is not None:
                last_x, last_y = prev_drag[0]
                dx, dy = x - last_x, y - last_y
                prev_drag[0] = (x, y)
                vf = get("prev_vfov")
                # vfov / preview_height = degrees per output pixel near
                # the centre (rectilinear, so accurate near centre).
                deg_per_px = vf / PREV_H
                # Drag pulls the world with the cursor: drag right ->
                # camera yaw LEFT (yaw decreases). Drag down -> camera
                # tilts UP (pitch decreases, since pitch+ = look down).
                setf("prev_yaw", get("prev_yaw") - dx * deg_per_px)
                setf("prev_pitch", get("prev_pitch") - dy * deg_per_px)
            if level_drag[0] is not None and show_level[0]:
                _last_x, last_y = level_drag[0]
                dy = y - last_y
                level_drag[0] = (x, y)
                # Convert px delta into fraction-of-height delta so the
                # line follows the cursor directly. Clamp to keep the
                # line on screen.
                new_frac = level_y_frac[0] + dy / PREV_H
                level_y_frac[0] = max(-0.49, min(0.49, new_frac))
        elif ev == cv2.EVENT_LBUTTONUP:
            prev_drag[0] = None
        elif ev == cv2.EVENT_RBUTTONUP:
            if level_drag[0] is not None:
                cfg.level_line_y_frac = level_y_frac[0]
            level_drag[0] = None
        elif ev == cv2.EVENT_MOUSEWHEEL:
            # OpenCV packs the wheel delta into the high 16 bits of flags
            # (signed). Positive = wheel-up = zoom in (vfov shrinks).
            delta = (flags >> 16) & 0xFFFF
            if delta > 32767:
                delta -= 65536
            cur_vfov = get("prev_vfov")
            if delta > 0:
                new_vfov = max(25.0, cur_vfov * 0.92)
            else:
                new_vfov = min(110.0, cur_vfov * 1.08)
            setf("prev_vfov", new_vfov)

    cv2.setMouseCallback(win, on_calibrate_mouse)
    cv2.setMouseCallback(prev, on_preview_mouse)

    # Render caches. The dewarp `model.render(...)` is the dominant cost
    # in this loop -- for a 4608x1728 source it takes long enough that
    # without caching the loupe lags noticeably behind the cursor.
    # Re-render only when the input state changes; cursor movement
    # leaves the state untouched so we hit the cache. The calibrate
    # window draw is cheaper but we cache it the same way for
    # consistency.
    cached_preview: list = [None]
    cached_preview_key: list = [None]
    cached_calib: list = [None]
    cached_calib_key: list = [None]

    def _lines_signature():
        # Only material when overlays / mark drawings depend on them.
        # Flatten to a tuple so the key stays hashable + cheap to
        # compare. The latest cur point matters because clicks grow it.
        return (tuple(tuple(p) for ln in lines for p in ln),
                tuple(cur))

    while True:
        model.k1 = get("k1 x1000")
        model.k2 = get("k2 x1000")
        model.hfov_deg = get("hfov")
        model.vfov_deg = get("vfov")
        model.pitch0_deg = get("pitch0 x10")
        model.roll0_deg = get("roll0 x10")

        prev_yaw = get("prev_yaw")
        prev_pitch = get("prev_pitch")
        prev_vfov = get("prev_vfov")

        # ---- calibrate window (cheap, but cache for symmetry) ----------
        ln_sig = _lines_signature()
        calib_key = (id(disp_ref[0]), ln_sig, horizon_idx,
                      t_now[0], show_level[0], show_overlay[0],
                      rms, fit_k2)
        if calib_key != cached_calib_key[0]:
            cached_calib[0] = _draw(
                disp_ref[0], lines, cur, horizon_idx, scale,
                model, rms, fit_k2,
                t_now[0], total_seconds,
                show_level[0], show_overlay[0])
            cached_calib_key[0] = calib_key
        cv2.imshow(win, cached_calib[0])

        # ---- preview window (expensive; aggressively cached) -----------
        prev_key = (
            id(frame_ref[0]),
            model.k1, model.k2,
            model.hfov_deg, model.vfov_deg,
            model.pitch0_deg, model.roll0_deg,
            prev_yaw, prev_pitch, prev_vfov,
            prev_w, prev_h,
            show_level[0], show_overlay[0], level_y_frac[0],
            # Lines/cur only affect the overlay layer, but cheap to
            # include unconditionally.
            ln_sig,
        )
        if prev_key != cached_preview_key[0]:
            view = model.render(frame_ref[0], prev_yaw, prev_pitch,
                                 prev_vfov, prev_w, prev_h,
                                 interp=cv2.INTER_LINEAR)
            view = _draw_preview_overlays(
                view, model, lines, cur,
                prev_yaw, prev_pitch, prev_vfov,
                show_level[0], show_overlay[0],
                level_y_frac=level_y_frac[0])
            cached_preview[0] = view
            cached_preview_key[0] = prev_key
        cv2.imshow(prev, cached_preview[0])

        draw_loupe()
        position_magnifier_loupe()

        # Use waitKeyEx so arrow keys come through with their full
        # platform-specific code (waitKey masks to the low byte, which
        # truncates arrow codes to 0 on Windows).
        k_full = cv2.waitKeyEx(20)
        if k_full == -1:
            continue
        k = k_full & 0xFF

        # Arrow nudges: one source pixel per press. After updating
        # cursor_src, also warp the OS cursor so the mouse pointer
        # visibly tracks the loupe crosshair. Repeat-fire is handled
        # by the OS auto-repeat -- holding an arrow scrolls smoothly.
        if k_full in ARROW_LEFT_CODES:
            cursor_src[0] = max(0.0, cursor_src[0] - 1.0)
            sync_os_cursor_to_virtual()
            continue
        if k_full in ARROW_RIGHT_CODES:
            cursor_src[0] = min(float(w - 1), cursor_src[0] + 1.0)
            sync_os_cursor_to_virtual()
            continue
        if k_full in ARROW_UP_CODES:
            cursor_src[1] = max(0.0, cursor_src[1] - 1.0)
            sync_os_cursor_to_virtual()
            continue
        if k_full in ARROW_DOWN_CODES:
            cursor_src[1] = min(float(h - 1), cursor_src[1] + 1.0)
            sync_os_cursor_to_virtual()
            continue
        # Enter (13) or Space (32) commits at the virtual cursor --
        # lets the user nudge precisely then commit without touching
        # the mouse (which would jump the cursor back).
        if k_full in (13, 32):
            commit_at_cursor()
            continue

        if k == 255 or k == 0:
            continue
        if k in (ord("q"), 27):
            break
        elif k == ord("n"):
            if len(cur) >= 2:
                lines.append(cur[:])
            cur.clear()
        elif k == ord("u"):
            if cur:
                cur.pop()
            elif lines and lines[-1]:
                lines[-1].pop()
        elif k == ord("c"):
            lines.clear(); cur.clear(); horizon_idx = -1; rms = None
        elif k == ord("h"):
            if len(cur) >= 2:
                lines.append(cur[:]); cur.clear()
            horizon_idx = len(lines) - 1
        elif k == ord("k"):
            fit_k2 = not fit_k2
        elif k == ord("l"):
            show_level[0] = not show_level[0]
            cfg.show_level_line = show_level[0]
            print(f"level reference line: {'ON' if show_level[0] else 'OFF'}")
        elif k == ord("o"):
            show_overlay[0] = not show_overlay[0]
            cfg.show_calib_overlay = show_overlay[0]
            print(f"calib lines overlay: {'ON' if show_overlay[0] else 'OFF'}")
        elif k == ord("0"):
            level_y_frac[0] = 0.0
            cfg.level_line_y_frac = 0.0
            print("level line re-centred")
        elif k == ord("z"):
            show_loupe[0] = not show_loupe[0]
            if not show_loupe[0]:
                # Drop the existing window so it doesn't sit stale.
                # Re-created next time the user toggles back on.
                try:
                    cv2.destroyWindow(loupe_win)
                except cv2.error:
                    pass
            else:
                cv2.namedWindow(loupe_win, cv2.WINDOW_NORMAL)
                try:
                    loupe_side = LOUPE_RADIUS_SRC_PX * 2 * LOUPE_ZOOM
                    cv2.resizeWindow(loupe_win, loupe_side, loupe_side)
                except cv2.error:
                    pass
            print(f"loupe: {'ON' if show_loupe[0] else 'OFF'}")
        elif k == ord("m"):
            magnifier_mode[0] = not magnifier_mode[0]
            print(f"magnifier mode: "
                  f"{'ON (centred + borderless + circular + click-through, cross cursor)' if magnifier_mode[0] else 'OFF (loupe at fixed slot)'}")
            if magnifier_mode[0]:
                # Hide the OS cursor (transparent class cursor) so
                # the loupe's drawn crosshair is the sole on-screen
                # indicator. The cursor's logical position still
                # works (clicks register at GetCursorPos coords);
                # just the visible arrow disappears, sidestepping any
                # hotspot-vs-visual-centre mismatch.
                _swap_window_class_cursor_invisible(
                    win, saved_class_cursor)
            else:
                # Restore everything in reverse: cursor first (cheap),
                # then loupe styles + size + position.
                _restore_window_class_cursor(win, saved_class_cursor)
                _revert_window_styles(loupe_win, loupe_saved_style)
                normal_side = LOUPE_RADIUS_SRC_PX * 2 * LOUPE_ZOOM
                try:
                    cv2.resizeWindow(loupe_win, normal_side, normal_side)
                    cv2.moveWindow(loupe_win,
                                    int(w * scale) + 20, prev_h + 60)
                except cv2.error:
                    pass
        elif k == ord(","):
            scrub(-1.0)
        elif k == ord("."):
            scrub(+1.0)
        elif k == ord("<"):
            scrub(-10.0)
        elif k == ord(">"):
            scrub(+10.0)
        elif k == ord("f"):
            alll = lines + ([cur] if len(cur) >= 3 else [])
            try:
                fitted, rms = fit_distortion(model, alll, fit_k2=fit_k2)
                hidx = horizon_idx if horizon_idx >= 0 else 0
                fitted = level_horizon(fitted, alll[min(hidx, len(alll) - 1)])
                setf("k1 x1000", fitted.k1)
                setf("k2 x1000", fitted.k2)
                setf("pitch0 x10", fitted.pitch0_deg)
                setf("roll0 x10", fitted.roll0_deg)
                print(f"fit: k1={fitted.k1:+.4f} k2={fitted.k2:+.4f} "
                      f"pitch0={fitted.pitch0_deg:+.2f} rms={rms:.4f}")
            except ValueError as e:
                print("fit needs >=2 references with >=3 points each:", e)
        elif k == ord("s"):
            cfg.pano = model
            cfg.calib_lines = [[list(p) for p in ln] for ln in lines]
            cfg.last_scrub_t = float(t_now[0])
            cfg.show_level_line = show_level[0]
            cfg.show_calib_overlay = show_overlay[0]
            cfg.level_line_y_frac = float(level_y_frac[0])
            cfg.save(pp)
            print("saved", pp)

    cap.release()
    cv2.destroyAllWindows()
