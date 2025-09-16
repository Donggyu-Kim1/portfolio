# main.py
"""
Pipeline runner for:
1) src/price.py
2) src/return.py
3) src/beta.py
4) src/ewma.py
5) src/cov.py
6) src/self_cov.py
7) src/var.py

Usage:
  python main.py               # fail-fast (default)
  python main.py --keep-going  # continue even if a step fails
  python main.py --            # pass-through args after -- to every step (optional)

Logs:
  Console + data/_logs/pipeline_YYYYmmdd_HHMMSS.log
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import sys
from pathlib import Path
from subprocess import Popen, PIPE
from typing import List, Tuple

# ---- Config: scripts to run in order (relative to project root) ----
SCRIPTS = [
    Path("src/price.py"),
    Path("src/return.py"),
    Path("src/beta.py"),
    Path("src/ewma.py"),
    Path("src/cov.py"),
    Path("src/self_cov.py"),
    Path("src/var.py"),
]


def _timestamp() -> str:
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _ts_for_filename() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def run_step(
    script_path: Path, log_fp: Path, passthrough: List[str]
) -> Tuple[int, float]:
    """
    Run one script with streaming stdout/stderr to console and log.
    Returns (return_code, elapsed_seconds).
    """
    start = dt.datetime.now()

    # Ensure unbuffered output from child
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    # Resolve to absolute path (project root is this file's parent)
    project_root = Path(__file__).resolve().parent
    script_abs = (project_root / script_path).resolve()

    if not script_abs.exists():
        msg = f"[{_timestamp()}] [ERROR] File not found: {script_abs}\n"
        sys.stderr.write(msg)
        with open(log_fp, "a", encoding="utf-8") as log:
            log.write(msg)
        return 127, 0.0

    cmd = [sys.executable, str(script_abs)]
    if passthrough:
        cmd += passthrough

    header = (
        f"[{_timestamp()}] [RUN ] {script_path.as_posix()}  (cmd: {' '.join(cmd)})\n"
    )
    sys.stdout.write(header)
    with open(log_fp, "a", encoding="utf-8") as log:
        log.write(header)

    # Stream output
    proc = Popen(
        cmd, stdout=PIPE, stderr=PIPE, cwd=str(project_root), env=env, text=True
    )
    # Interleave stdout/stderr
    with open(log_fp, "a", encoding="utf-8") as log:
        # Read until process ends
        while True:
            out_line = proc.stdout.readline() if proc.stdout else ""
            err_line = proc.stderr.readline() if proc.stderr else ""
            if not out_line and not err_line and proc.poll() is not None:
                break
            if out_line:
                sys.stdout.write(out_line)
                log.write(out_line)
            if err_line:
                sys.stderr.write(err_line)
                log.write(err_line)

    rc = proc.wait()
    elapsed = (dt.datetime.now() - start).total_seconds()
    footer = f"[{_timestamp()}] [DONE] {script_path.name}  exit={rc}  elapsed={elapsed:.1f}s\n"
    # Color-ish marker for failures
    if rc != 0:
        footer = f"[{_timestamp()}] [FAIL] {script_path.name}  exit={rc}  elapsed={elapsed:.1f}s\n"

    sys.stdout.write(footer if rc == 0 else footer)
    with open(log_fp, "a", encoding="utf-8") as log:
        log.write(footer)

    return rc, elapsed


def main():
    parser = argparse.ArgumentParser(description="Sequential pipeline runner")
    parser.add_argument(
        "--keep-going", action="store_true", help="Do not stop on step failure"
    )
    # everything after -- is passed to each step, unchanged
    if "--" in sys.argv:
        idx = sys.argv.index("--")
        args_ns = parser.parse_args(sys.argv[1:idx])
        passthrough = sys.argv[idx + 1 :]
    else:
        args_ns = parser.parse_args()
        passthrough = []

    # Prepare logs dir
    project_root = Path(__file__).resolve().parent
    logs_dir = project_root / "data" / "_logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_fp = logs_dir / f"pipeline_{_ts_for_filename()}.log"

    # Summary header
    header = (
        f"[{_timestamp()}] === PIPELINE START ===\n"
        f"Steps: {len(SCRIPTS)}\n"
        f"Keep-going: {args_ns.keep_going}\n"
        f"Log file : {log_fp}\n"
        f"Python   : {sys.executable}\n"
    )
    sys.stdout.write(header)
    with open(log_fp, "a", encoding="utf-8") as log:
        log.write(header)

    # Run steps
    results = []
    for i, script in enumerate(SCRIPTS, 1):
        banner = (
            f"[{_timestamp()}] --- Step {i}/{len(SCRIPTS)}: {script.as_posix()} ---\n"
        )
        sys.stdout.write(banner)
        with open(log_fp, "a", encoding="utf-8") as log:
            log.write(banner)

        rc, elapsed = run_step(script, log_fp, passthrough)
        results.append((script.name, rc, elapsed))
        if rc != 0 and not args_ns.keep_going:
            msg = f"[{_timestamp()}] Pipeline aborted due to failure at step {i}: {script.name}\n"
            sys.stderr.write(msg)
            with open(log_fp, "a", encoding="utf-8") as log:
                log.write(msg)
            break

    # Summary footer
    ok = sum(1 for _, rc, _ in results if rc == 0)
    fail = sum(1 for _, rc, _ in results if rc != 0)
    total_elapsed = sum(t for _, _, t in results)
    summary = (
        f"[{_timestamp()}] === PIPELINE END ===  ok={ok}, fail={fail}, "
        f"elapsed_total={total_elapsed:.1f}s\n"
    )
    sys.stdout.write(summary)
    with open(log_fp, "a", encoding="utf-8") as log:
        log.write(summary)

    # Non-zero exit if any failed (unless keep-going? still return non-zero to signal issues)
    if fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
