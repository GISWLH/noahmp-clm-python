"""Supervise independent single-core benchmark repeats until a UTC deadline.

Example: python benchmarks/run_cpu_batch.py --deadline 2026-10-13T10:45:00+00:00
    --job 10000:2,4,6 --job 100000:8,10,12 --job 300000:14,16,18
    --output-dir ../logs/h5-overnight

Each child owns a new process group. Only groups created by this supervisor are
terminated at the deadline. State and all partial measurements are preserved.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def now():
    return datetime.now(timezone.utc)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deadline", required=True)
    parser.add_argument("--job", action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    deadline = datetime.fromisoformat(args.deadline)
    if deadline.tzinfo is None or deadline <= now():
        parser.error("deadline must be a future timezone-aware date")
    specs = [(int(job.split(":")[0]), [int(x) for x in job.split(":")[1].split(",")])
             for job in args.job]
    cores = [cpu for _, group in specs for cpu in group]
    if len(set(cores)) != len(cores) or any(len(group) != 3 for _, group in specs):
        parser.error("every job needs three CPU IDs, disjoint across jobs")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    repo = Path(__file__).resolve().parents[1]
    state_path = args.output_dir.resolve() / "state.json"
    state = {"supervisor_pid": os.getpid(), "started_utc": now().isoformat(),
             "deadline_utc": deadline.isoformat(), "status": "running", "jobs": []}
    processes = []

    def save():
        state["updated_utc"] = now().isoformat()
        temporary = state_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
        temporary.replace(state_path)

    for n, group in specs:
        output = state_path.parent / f"numpy-{n}.jsonl"
        console = state_path.parent / f"numpy-{n}.log"
        command = [sys.executable, "-u", str(repo / "benchmarks/bench_numpy_parallel_repeats.py"),
                   "--sizes", str(n), "--cores", *map(str, group), "--steps", "48",
                   "--output", str(output)]
        environment = dict(os.environ, PYTHONPATH=str(repo / "src"))
        with console.open("x", encoding="utf-8") as stream:
            process = subprocess.Popen(command, cwd=repo, env=environment, stdin=subprocess.DEVNULL,
                                       stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        row = {"n": n, "cores": group, "pid": process.pid, "pgid": process.pid,
               "command": command, "output": str(output), "console": str(console), "exit_code": None}
        processes.append((process, row))
        state["jobs"].append(row)
        save()
    while any(process.poll() is None for process, _ in processes):
        for process, row in processes:
            row["exit_code"] = process.poll()
        save()
        if now() >= deadline:
            for process, row in processes:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                    row["deadline_terminated"] = True
            time.sleep(5)
            for process, _ in processes:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGKILL)
            state["status"] = "deadline"
            break
        time.sleep(15)
    for process, row in processes:
        row["exit_code"] = process.wait()
    if state["status"] == "running":
        state["status"] = "completed" if all(row["exit_code"] == 0 for _, row in processes) else "failed"
    save()
    print(json.dumps(state), flush=True)
    return 0 if state["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
