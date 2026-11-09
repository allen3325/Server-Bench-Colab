#!/usr/bin/env python3
"""Keep a running evolve's evidence off the VM while it runs.

A Colab runtime can be reclaimed at any moment, and with it everything on its
disk. This sidecar runs beside `local-evolve.sh`'s driver, as root, in the
HOST network namespace, and pushes the run to the rclone remote (`persist.sh
save`):

* every `--interval` seconds, and soon after every new version (the journal's
  version events), so at most one interval of work is ever lost;
* never while a measurement is running (an `aiperf` replay, or the verdict's
  worker): an upload competes for CPU, disk and network with the server being
  timed. A deferred checkpoint waits for the measurement to end;
* once more after the driver exits, with the verdict.

It also copies the agent's Claude Code transcripts (`~agent/.claude/projects`,
which live outside the run directory) into `<run>/claude-projects`: that is
where a make-up run (`run.py --resume-run`) looks for the conversation to
continue. Codex keeps its sessions in `<run>/codex-sessions` already.

Nothing here touches the agent's sandbox: no mount, no route, no process in
its namespace. The remote's token stays in the root-only secrets directory.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
PERSIST = REPO / "serverbench/scripts/colab/persist.sh"
AGENT_PROJECTS = Path("/home/agent/.claude/projects")
VERSION_EVENTS = {"version_registered", "head_advanced", "version_outcome",
                  "candidate_captured", "final_validation"}
MEASURING = ("aiperf", "final_validate.modal_worker", "final_validate.local_verdict")


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(message: str) -> None:
    print(f"{now()} checkpoint: {message}", flush=True)


def alive(pid: int) -> bool:
    if pid <= 0:
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    state = Path(f"/proc/{pid}/stat")
    try:
        return state.read_text().split(") ", 1)[1][0] != "Z"
    except (OSError, IndexError):
        return False


def measuring() -> str:
    """The first running measurement process, or ''."""
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            cmdline = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode(
                "utf-8", "replace")
        except OSError:
            continue
        for marker in MEASURING:
            if marker in cmdline and "checkpoint.py" not in cmdline:
                return marker
    return ""


def created_at(run: Path) -> float:
    """The run's own creation time (its journal's `run_created`), or 0."""
    try:
        with (run / "journal.jsonl").open() as handle:
            for line in handle:
                record = json.loads(line)
                if record.get("event") == "run_created":
                    stamp = record.get("created_at") or record.get("at_utc") or ""
                    return datetime.strptime(stamp, "%Y-%m-%dT%H:%M:%SZ").replace(
                        tzinfo=timezone.utc).timestamp()
    except (OSError, ValueError):
        pass
    return 0.0


def newest_run(task: str, since: float) -> Path | None:
    """The run this driver started: created after this sidecar was. A run that
    merely ended recently (its directory still being touched) is not it."""
    root = REPO / "serverbench/evolve_space/evolve" / task.lower()
    if not root.is_dir():
        return None
    runs = [(created_at(p), p) for p in root.iterdir() if (p / "journal.jsonl").is_file()]
    runs = [(t, p) for t, p in runs if t >= since - 5]
    return max(runs)[1] if runs else None


def copy_transcripts(run: Path) -> int:
    """Regular files only, symlinks never followed; root-only at the target."""
    if not AGENT_PROJECTS.is_dir():
        return 0
    target = run / "claude-projects"
    copied = 0
    for dirpath, dirnames, filenames in os.walk(AGENT_PROJECTS, followlinks=False):
        here = Path(dirpath)
        dest = target / here.relative_to(AGENT_PROJECTS)
        dest.mkdir(parents=True, exist_ok=True)
        os.chmod(dest, 0o700)
        for name in filenames:
            src = here / name
            if src.is_symlink() or not src.is_file():
                continue
            out = dest / name
            st = src.stat()
            if out.exists() and out.stat().st_size == st.st_size \
                    and out.stat().st_mtime >= st.st_mtime:
                continue
            shutil.copyfile(src, out, follow_symlinks=False)
            os.chmod(out, 0o600)
            copied += 1
    return copied


class Journal:
    def __init__(self, path: Path):
        self.path, self.offset = path, 0
        self.last_event, self.versions = "", []

    def new_events(self) -> set[str]:
        events = set()
        try:
            with self.path.open("rb") as handle:
                handle.seek(self.offset)
                for raw in handle:
                    if not raw.endswith(b"\n"):
                        break                       # a line still being written
                    self.offset += len(raw)
                    try:
                        record = json.loads(raw)
                    except ValueError:
                        continue
                    event = str(record.get("event", ""))
                    events.add(event)
                    self.last_event = event
                    if event == "head_advanced" and record.get("version_id"):
                        self.versions.append(record["version_id"])
        except OSError:
            pass
        return events


def push(args, run: Path | None, journal: Journal | None, reason: str) -> bool:
    started = time.monotonic()
    copied = copy_transcripts(run) if run else 0
    env = dict(os.environ, TASK=args.task)
    env.setdefault("RCLONE_CONFIG", str(REPO / "serverbench/artifacts/secrets/rclone.conf"))
    got = subprocess.run(["bash", str(PERSIST), "save", args.remote], env=env,
                         capture_output=True, text=True)
    ok = got.returncode == 0
    marker = {
        "at_utc": now(), "reason": reason, "ok": ok,
        "run_id": run.name if run else "", "run_root": str(run) if run else "",
        "last_event": journal.last_event if journal else "",
        "versions": journal.versions if journal else [],
        "transcripts_copied": copied,
        "seconds": round(time.monotonic() - started, 1),
    }
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
        json.dump(marker, handle, indent=1)
    subprocess.run([os.environ.get("RCLONE", "/root/bin/rclone"), "copyto",
                    handle.name, f"{args.remote}/checkpoint.json"],
                   env=env, capture_output=True)
    os.unlink(handle.name)
    tail = (got.stdout + got.stderr).strip().splitlines()[-1:] or [""]
    log(f"{'saved' if ok else 'FAILED'} ({reason}, {marker['seconds']}s"
        f"{', ' + str(copied) + ' transcript file(s)' if copied else ''}) {tail[0][:160]}")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--task", required=True)
    ap.add_argument("--driver-pid", type=int, default=0,
                    help="exit (after a last checkpoint) when this process ends")
    ap.add_argument("--interval", type=float, default=600)
    ap.add_argument("--since", type=float, default=0,
                    help="epoch seconds before the driver started (default: now)")
    ap.add_argument("--remote", default=os.environ.get("SB_REMOTE", "sb:serverbench-colab"))
    args = ap.parse_args()

    since = args.since or time.time()
    stop = {"now": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.update(now=True))
    signal.signal(signal.SIGINT, lambda *_: stop.update(now=True))

    run: Path | None = None
    journal: Journal | None = None
    last_push, pending, deferred_by = time.monotonic(), "", ""
    log(f"watching {args.task} every {args.interval:.0f}s -> {args.remote}")
    while True:
        if run is None:
            run = newest_run(args.task, since)
            if run is not None:
                journal = Journal(run / "journal.jsonl")
                log(f"run {run.name}")
        if journal is not None and journal.new_events() & VERSION_EVENTS:
            pending = pending or f"after {journal.last_event}"
        # A stop signal (usually the same one the driver got) does not end the
        # watch while the driver is still preserving its evidence: the last
        # checkpoint has to come after the driver's teardown, not before it.
        finished = (not alive(args.driver_pid) if args.driver_pid > 0
                    else stop["now"])
        if finished:
            push(args, run, journal, "the run ended")
            return 0
        due = pending or (time.monotonic() - last_push >= args.interval and "interval")
        if due:
            busy = measuring()
            if busy:
                if deferred_by != busy:
                    log(f"deferred: {busy} is measuring")
                deferred_by = busy
            else:
                push(args, run, journal, due)
                last_push, pending, deferred_by = time.monotonic(), "", ""
        time.sleep(15)


if __name__ == "__main__":
    sys.exit(main())
