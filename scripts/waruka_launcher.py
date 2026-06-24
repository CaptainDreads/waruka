# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Stefan Lewis
"""PyInstaller entry point for both `waruka.exe` (GUI, windowed) and
`waruka-cli.exe` (CLI, console). Both binaries run this script; the only
difference is the PyInstaller subsystem flag (`--windowed` vs `--console`).

Behaviour:
- `waruka.exe` (no args)         -> launch the GUI
- `waruka.exe gui`               -> launch the GUI
- `waruka.exe track --project P` -> CLI subcommand (forwards to waruka.__main__)
- `waruka-cli.exe ...`           -> same dispatch, but with a real console
                                   so stdout/stderr is visible live

Logging:
- Every launch writes `<exe_dir>/logs/waruka-<timestamp>-<pid>.log`.
- In windowed mode (no console), all `print()` output goes ONLY to that
  log file -- standard streams aren't connected to anything.
- In console mode, output tees to both the terminal AND the log file.

Bundled paths note: when frozen, `sys._MEIPASS` is the extracted bundle
root. `third_party/`, `yolo11n.pt`, and the `waruka/` package all live
directly under it. The runtime helpers in `waruka.interpolate`,
`waruka.sr`, and `waruka.perception` compute their default weight paths
as `Path(__file__).resolve().parent.parent / ...`, which resolves to
`_MEIPASS / third_party / ...` in the frozen bundle -- no special-casing
needed here.
"""
from __future__ import annotations

import io
import os
import sys
from datetime import datetime
from pathlib import Path


def _bundle_root() -> Path | None:
    """Return the bundle root if running under PyInstaller, else None."""
    if getattr(sys, "frozen", False):
        return Path(getattr(sys, "_MEIPASS", sys.executable)).resolve()
    return None


def _exe_dir() -> Path:
    """Directory containing the exe (or the launcher .py in dev mode).

    Logs land here so the bundle stays portable -- nothing writes to
    %APPDATA% or %LOCALAPPDATA%, and there's no install step required.
    """
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


def _set_app_user_model_id() -> None:
    """Tell Windows this is its own app, not a generic Python launcher.

    Without this, Windows groups Waruka's taskbar entry under whichever
    interpreter / launcher started it (a generic Python icon in the
    frozen bundle, falling back to a feature-less placeholder). Setting
    an explicit AppUserModelID makes the taskbar show our icon as a
    distinct pinnable entry.

    No-op on non-Windows. Best-effort: a failure here is cosmetic
    (icon shows up wrong) and must not gate startup.
    """
    if sys.platform != "win32":
        return
    try:
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
            "com.waruka.waruka")
    except Exception:
        pass


def _augment_dll_search_path() -> None:
    """Make sure bundled CUDA runtime + NVDEC DLLs are findable.

    `waruka.nvdecode` walks site-packages for `nvidia/cuda_runtime/bin` --
    in the frozen bundle those wheels' bin dirs end up inside
    `_internal/`, not on `site.getsitepackages()`. Pre-register them via
    `os.add_dll_directory` so the explicit loader still works.
    """
    root = _bundle_root()
    if root is None:
        return
    candidates = []
    for sub in (
        root / "nvidia" / "cuda_runtime" / "bin",
        root / "_internal" / "nvidia" / "cuda_runtime" / "bin",
        root,
        root / "_internal",
    ):
        if sub.is_dir():
            candidates.append(sub)
    for d in candidates:
        try:
            os.add_dll_directory(str(d))
        except (OSError, AttributeError):
            pass


class _Tee:
    """Write to multiple streams. Tolerates one of them failing
    (windowed bootloader sometimes hands out broken stdio handles).
    """
    def __init__(self, *streams):
        self.streams = [s for s in streams if s is not None]

    def write(self, s: str) -> int:
        for st in self.streams:
            try:
                st.write(s)
            except (OSError, ValueError, AttributeError):
                pass
        return len(s)

    def flush(self) -> None:
        for st in self.streams:
            try:
                st.flush()
            except (OSError, ValueError, AttributeError):
                pass


def _is_windowed_subsystem() -> bool:
    """Read our own exe's PE Subsystem field to decide redirect strategy.

    PyInstaller's windowed bootloader is built as Windows GUI subsystem
    (PE Subsystem = 2); the console bootloader is Windows console
    (Subsystem = 3). The exe's own PE header is the unambiguous answer.

    Runtime GetConsoleWindow() is unreliable: it returns 0 for a console
    exe launched from a noninteractive PowerShell or under a harness
    that doesn't allocate a real console (which would incorrectly trip
    full-redirect mode and lose terminal output for the CLI exe).
    """
    if not getattr(sys, "frozen", False):
        return False  # dev mode: keep terminal output
    try:
        with open(sys.executable, "rb") as f:
            f.seek(0x3c)
            pe_off = int.from_bytes(f.read(4), "little")
            # PE\0\0 sig (4 B) + COFF file header (20 B) +
            # optional-header offset to Subsystem (68 B, same for PE32/PE32+).
            f.seek(pe_off + 4 + 20 + 68)
            subsystem = int.from_bytes(f.read(2), "little")
        # 2 = IMAGE_SUBSYSTEM_WINDOWS_GUI  (no console attached)
        # 3 = IMAGE_SUBSYSTEM_WINDOWS_CUI  (console subsystem)
        return subsystem == 2
    except Exception:
        # Safer to default to console on read failure: dropping terminal
        # output for the CLI exe is worse than missing subprocess capture
        # on the GUI exe.
        return False


def _redirect_os_stdio_to_log(log_f) -> None:
    """Aim every layer of stdout/stderr at the log file (windowed mode).

    Three layers, each catching a different kind of write:
      1. Python `sys.stdout`/`sys.stderr` -- catches `print()` calls
      2. OS-level fds 1 and 2 (`os.dup2`) -- catches C-level writes from
         native extensions (PyTorch, OpenCV, etc.) that bypass Python's
         file objects
      3. Windows STD_OUTPUT_HANDLE / STD_ERROR_HANDLE (`SetStdHandle`) --
         catches subprocesses (ffmpeg via imageio_ffmpeg) that inherit
         the parent's stdio when spawned with default `stdout=None`

    Without layer 3, ffmpeg's encoder progress would go to a dead handle
    in windowed mode and silently disappear -- the log file would have
    no record of why the encode stalled.
    """
    # Layer 3: tell child processes to use the log file as their stdio.
    try:
        import ctypes
        import msvcrt
        h = msvcrt.get_osfhandle(log_f.fileno())
        STD_OUTPUT_HANDLE = -11
        STD_ERROR_HANDLE = -12
        ctypes.windll.kernel32.SetStdHandle(STD_OUTPUT_HANDLE, h)
        ctypes.windll.kernel32.SetStdHandle(STD_ERROR_HANDLE, h)
    except Exception:
        pass

    # Layer 2: parent C-level writes via fd 1 / fd 2.
    try:
        os.dup2(log_f.fileno(), 1)
        os.dup2(log_f.fileno(), 2)
    except OSError:
        pass

    # Layer 1: parent Python writes via sys.stdout / sys.stderr.
    sys.stdout = log_f
    sys.stderr = log_f


def _setup_logging() -> Path | None:
    """Set up the per-launch log file at `<exe_dir>/logs/`.

    In console mode (waruka-cli.exe) the log is a copy of what the user
    already sees on the terminal -- terminal stays primary, log is the
    auditable record. In windowed mode (waruka.exe) the log is the only
    sink for everything: Python prints, native-extension writes, AND
    subprocess output from ffmpeg etc.

    Returns the log file path, or None on failure (logging never gates
    startup -- the bundle should still launch if we can't open the log).
    """
    try:
        log_dir = _exe_dir() / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        log_path = log_dir / f"waruka-{ts}-{os.getpid()}.log"
        f = open(log_path, "w", encoding="utf-8", buffering=1)  # line-buffered

        windowed = _is_windowed_subsystem()
        if windowed:
            # No terminal: redirect every stdio layer so subprocess
            # output (ffmpeg, etc.) is also captured.
            _redirect_os_stdio_to_log(f)
        else:
            # Keep the terminal as the primary channel; tee Python writes.
            # Don't touch fd 1/2 or STD_*_HANDLE -- subprocess output
            # stays on the terminal where the user expects to see it.
            sys.stdout = _Tee(sys.stdout, f)
            sys.stderr = _Tee(sys.stderr, f)

        # Header makes the log readable when diagnosing a specific run.
        print(f"=== waruka launch {datetime.now().isoformat()} ===")
        print(f"argv: {sys.argv}")
        print(f"frozen: {getattr(sys, 'frozen', False)}")
        print(f"exe_dir: {_exe_dir()}")
        print(f"mode: {'windowed (full redirect)' if windowed else 'console (tee)'}")
        return log_path
    except Exception as e:  # pragma: no cover - best-effort logging
        # Last-resort fallback: keep going without logging.
        try:
            sys.stderr.write(f"warning: failed to set up logging: {e}\n")
        except Exception:
            pass
        return None


def main() -> int:
    _set_app_user_model_id()
    _augment_dll_search_path()
    _setup_logging()
    from waruka.__main__ import main as waruka_main

    argv = sys.argv[1:]
    # The GUI builds subprocess invocations as `python -m waruka <cmd>`,
    # which becomes `waruka.exe -m waruka <cmd>` in the frozen bundle.
    # `-m waruka` is meaningless when the binary IS waruka -- strip it
    # so argparse doesn't reject the unknown `-m` flag with exit code 2.
    if len(argv) >= 2 and argv[0] == "-m" and argv[1] == "waruka":
        argv = argv[2:]
    if not argv:
        argv = ["gui"]
    waruka_main(argv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
