#!/usr/bin/env python3

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description='Run one command and report elapsed time and OS peak resident memory.')
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command
    if command[:1] == ["--"]:
        command = command[1:]
    if not command:
        parser.error("provide a command after --")
    if sys.platform not in ("linux", "darwin") or not hasattr(os, "wait4"):
        parser.error("resource measurement supports Linux and macOS")

    started = datetime.now(timezone.utc)
    label = re.sub(r"[^A-Za-z0-9_.-]+", "-", args.label).strip("-") or "run"
    output = args.output or (
        Path(__file__).resolve().parents[1] / "benchmark_logs" / "resources"
        / (label + "_" + started.strftime("%Y%m%d_%H%M%S_%f") + ".json"))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as handle:
        record = dict(label=args.label, command=command, cwd=str(Path.cwd()),
                      started_utc=started.isoformat(), platform=sys.platform,
                      hostname=os.uname().nodename, status="RUNNING")
        json.dump(record, handle, indent=2)
    print(f"Resource record: {output}", file=sys.stderr, flush=True)

    start = time.monotonic()
    peak_bytes = None
    returncode = 127
    try:
        process = subprocess.Popen(command)
        while True:
            try:
                _, status, usage = os.wait4(process.pid, 0)
                break
            except KeyboardInterrupt:
                try:
                    os.kill(process.pid, signal.SIGINT)
                except ProcessLookupError:
                    pass
        returncode = (os.WEXITSTATUS(status) if os.WIFEXITED(status)
                      else -os.WTERMSIG(status))
        process.returncode = returncode
        peak_bytes = int(usage.ru_maxrss) * (1 if sys.platform == "darwin" else 1024)
    except OSError as error:
        record["error"] = str(error)
        returncode = 127 if isinstance(error, FileNotFoundError) else 126
    elapsed = time.monotonic() - start
    exit_code = returncode if returncode >= 0 else 128 - returncode
    peak_gib = None if peak_bytes is None else peak_bytes / (1024 ** 3)
    record.update(status="COMPLETED" if exit_code == 0 else "FAILED",
                  wall_seconds=elapsed, peak_rss_bytes=peak_bytes,
                  peak_rss_gib=peak_gib, exit_code=exit_code,
                  signal=-returncode if returncode < 0 else None,
                  memory_scope="OS ru_maxrss; not aggregate concurrent process-tree memory")
    with output.open("w") as handle:
        json.dump(record, handle, indent=2)
        handle.write("\n")
    peak_text = "N/A" if peak_gib is None else f"{peak_gib:.6f} GiB"
    print(f"\n[Process Resources]\nlabel: {args.label}\n"
          f"wall: {elapsed:.6f} s\npeak RSS: {peak_text}\n"
          f"exit code: {exit_code}\nrecord: {output}",
          file=sys.stderr, flush=True)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
