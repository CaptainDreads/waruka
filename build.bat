@echo off
REM ---------------------------------------------------------------------------
REM One-click Waruka build wrapper.
REM
REM Runs `python scripts/build_exe.py --zip` from the repo root.
REM
REM Equivalent to opening PowerShell and typing:
REM     python scripts/build_exe.py --zip
REM
REM Output:
REM     dist\waruka\                  -- working bundle (5 GB; waruka.exe + waruka-cli.exe)
REM     dist\waruka-<version>.zip     -- zipped distributable (3 GB)
REM
REM Wall time: ~15-20 minutes. The build prints progress; if a step fails it
REM stops with a clear error message pointing at the fix.
REM
REM Prerequisites are documented in BUILDING.md.
REM ---------------------------------------------------------------------------

setlocal
cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 (
    echo ERROR: 'python' not found on PATH. Install Python 3.13 from python.org
    echo        and tick "Add to PATH" during install, or use the launcher 'py'.
    exit /b 1
)

python scripts\build_exe.py --zip
exit /b %errorlevel%
