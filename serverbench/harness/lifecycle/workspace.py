"""Build the workspace, prove it boots, and own what that took.

One workspace, built from a manifest rather than a directory walk, and proven
against its own boot contract before an agent is put in it. `Ownership` is the
ledger of what that cost -- a container, a device -- recorded by exact id so the
only cleanup that can touch it is this run's. The request file is the agent's
end of the same workspace: it asks for a version, and the harness does not
decide on a clock.

## Version requests
The agent asks for a version. The harness does not decide on a clock.

A round is a unit of the agent's work, and only the agent knows when one has
finished. A boundary every N minutes would let the HARNESS decide what a
version was: it would freeze trees mid-edit and make the cadence a property of
a launch flag rather than of the work.

Standard workspaces get one atomic command:

    ./version "what changed and why it is worth keeping"

It writes a request and returns immediately. The agent is never blocked, and it
may ask as often as it likes -- the brief encourages exactly that. The harness
notices the request on its next poll and does the rest: capture, check,
materialise, publish.

Instructed workspaces instead get ``./checkpoint`` to perform capture and
materialisation without changing the final head, and ``./promote`` to select an
existing checkpoint without capturing the live worktree.

The request is a file rather than a socket call because it has to survive the
agent's process, work from a shell, and be readable afterwards as evidence of
what the agent thought it was doing at that moment.

## Ownership
What this run owns, by exact id, and the only cleanup that may touch it.

Cleanup by name prefix (``docker ps --filter name=<prefix>-``) can kill a
*live* agent belonging to a concurrent run. Prefix matching cannot tell two
runs apart, so there is no prefix matching here.

A resource is released only if this run's journal says this run acquired it.
Nothing is discovered by scanning, and nothing is signalled that we did not
record starting.

## The workspace
Build one workspace, then prove it works before an agent is put in it.

Small and explicit. What the agent gets is listed here; what it does not get is
absent because it was never copied, not because a runtime scan removes it
afterwards.

Two things this refuses to do:

* **No environment is copied.** The agent builds its own virtualenv from a
  read-only wheelhouse, and ``.venv`` is not in the candidate manifest, so it
  cannot be captured, materialised or checked. A directory walk would map a
  multi-gigabyte ``.venv`` as candidate source and make every boundary slow.
* **No interpreter stub.** If the declared interpreter is not there, this
  reports ``HARNESS_SETUP_FAILED``. An executable stub where a venv should be
  would let ``serve.sh`` silently fall back to the system python and measure
  the wrong environment.

The candidate manifest is the spine of the whole lifecycle: capture, checks,
materialisation and module attribution all read it, and none of them walks the
tree. Standard tasks get the one-step ``./version`` control. Instructed
tasks receive ``./checkpoint`` and ``./promote`` so saving an immutable version
cannot accidentally select the final head.
"""
from __future__ import annotations

import contextlib
import dataclasses
import json
import os
import pathlib
import shutil
import signal
import subprocess
import time
from pathlib import Path

from . import backend
from .record import Journal
from .utils import ensure_harness_on_path

# --- version requests ------------------------------------------------------------

#: Inside the workspace, so the agent can write it, and dot-prefixed so it is
#: never candidate source.
REQUEST_DIR = ".versions"

#: The script the agent runs. Deliberately tiny: it must not be able to fail in
#: a way that costs the agent its work, and it must not need the harness.
SCRIPT = '''#!/usr/bin/env bash
# Ask the harness to publish the current state of this workspace as a version.
#
#   ./version "what changed in this version"
#
# The request is written at once. Nothing you do afterwards changes this
# checkpoint: the harness freezes the tree as it stands and performs
# deterministic/form checks. This script then waits up to a minute for those
# checks and prints the time left and whether a full test has passed on
# exactly this tree.
#
# Any number of versions may be requested; the harness imposes no publication
# cadence.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$here/.versions"
note="${*:-}"
stamp="$(date -u +%Y%m%dT%H%M%SZ)-$$"
printf '{"requested_utc": "%s", "action": "publish", "note": %s}\\n' \\
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \\
    "$(printf '%s' "$note" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))')" \\
    > "$here/.versions/$stamp.json"
echo "version requested; the harness freezes this tree and writes BOUNDARY_CHECKS.md here"
if [[ -x "$here/clock" ]]; then
    "$here/clock" 2>/dev/null | grep -E '^(deadline_utc|remaining)=' || true
fi
for _ in $(seq 60); do
    if [[ "$here/BOUNDARY_CHECKS.md" -nt "$here/.versions/$stamp.json" ]]; then
        awk '/^## (Failed checks|All checks)/{exit} {print}' "$here/BOUNDARY_CHECKS.md" || true
        exit 0
    fi
    sleep 1
done
echo "(its checks are not written yet -- an unchanged tree writes none; read BOUNDARY_CHECKS.md later)"
'''

CHECKPOINT_SCRIPT = '''#!/usr/bin/env bash
# Freeze the current source as an immutable version without changing the final
# published head. The harness completes the capture asynchronously.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$here/.versions"
note="${*:-}"
stamp="$(date -u +%Y%m%dT%H%M%SZ)-checkpoint-$$"
printf '{"requested_utc": "%s", "action": "checkpoint", "note": %s}\\n' \\
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \\
    "$(printf '%s' "$note" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))')" \\
    > "$here/.versions/$stamp.json"
echo "checkpoint requested; it will create a version without moving the final head"
if [[ -x "$here/clock" ]]; then
    "$here/clock" 2>/dev/null | grep -E '^(deadline_utc|remaining)=' || true
fi
for _ in $(seq 60); do
    if [[ "$here/BOUNDARY_CHECKS.md" -nt "$here/.versions/$stamp.json" ]]; then
        awk '/^## (Failed checks|All checks)/{exit} {print}' "$here/BOUNDARY_CHECKS.md" || true
        exit 0
    fi
    sleep 1
done
echo "(its checks are not written yet -- an unchanged tree writes none; read BOUNDARY_CHECKS.md later)"
'''

PROMOTE_SCRIPT = '''#!/usr/bin/env bash
# Move the final published head to an already saved checkpoint.
# Usage: ./promote v7 "best public evidence"; a candidate id such as r007-...
# also works. `latest` selects this run's newest completed checkpoint. Promotion
# is optional unless you want an older checkpoint: at run end the harness uses
# the newest saved checkpoint when no explicit selection was made.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$here/.versions"
target="${1:-latest}"
if [[ $# -gt 0 ]]; then shift; fi
note="${*:-}"
stamp="$(date -u +%Y%m%dT%H%M%SZ)-promote-$$"
printf '{"requested_utc": "%s", "action": "promote", "target": %s, "note": %s}\\n' \\
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \\
    "$(printf '%s' "$target" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))')" \\
    "$(printf '%s' "$note" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))')" \\
    > "$here/.versions/$stamp.json"
echo "head promotion requested for $target; no new checkpoint will be captured"
'''

CLOCK_SCRIPT = '''#!/usr/bin/env python3
"""Report this run's immutable deadline and live remaining budget."""
import signal
from datetime import datetime, timezone

# `./clock | head` closes the pipe early; die quietly rather than print a
# BrokenPipeError traceback that reads as the clock being broken.
signal.signal(signal.SIGPIPE, signal.SIG_DFL)

DEADLINE_TEXT = __DEADLINE_JSON__
deadline = datetime.fromisoformat(DEADLINE_TEXT.replace("Z", "+00:00"))
now = datetime.now(timezone.utc)
remaining = max(0, int((deadline - now).total_seconds()))
hours, rest = divmod(remaining, 3600)
minutes, seconds = divmod(rest, 60)

print(f"now_utc={now.strftime('%Y-%m-%dT%H:%M:%SZ')}")
print(f"deadline_utc={deadline.astimezone(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}")
print(f"remaining={hours:02d}:{minutes:02d}:{seconds:02d}")
print(f"remaining_seconds={remaining}")
print(f"status={'RUNNING' if now < deadline else 'EXPIRED'}")
'''

FULL_TEST_SCRIPT = r'''#!/usr/bin/env python3
"""Run the verdict's own measurement on the current tree, in this container.

Freezes the tree as publication would, cold-starts its serve.sh on its own
port, runs the protocol/context probes and an AgentX replay of --duration
seconds (300-1800; the final verdict always runs 1800) at concurrency.json,
and reports replay fidelity, the request scores and dollars_per_dollar.
Stop your own GPU servers and profilers first: the card must be free.
"""
import argparse
import json
from pathlib import Path
import subprocess
import time
import uuid

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--wait', action='store_true')
parser.add_argument('--duration', type=int, default=1800,
                    help='replay seconds, 300-1800 (default 1800, the verdict length)')
args = parser.parse_args()
if not 300 <= args.duration <= 1800:
    parser.error('--duration must be 300-1800 seconds')
root = Path(__file__).resolve().parent
request_id = 'full-test-' + uuid.uuid4().hex
request = root / '.versions' / (request_id + '.json')
temporary = request.with_suffix('.tmp')
temporary.write_text(json.dumps({'action': 'full_test', 'note': 'public full verifier',
                                 'duration_s': args.duration}))
temporary.replace(request)
report = root / 'agentx/full-tests' / request_id / 'full-test.json'
fallback = root / '.versions' / (request_id + '.result')


def time_left():
    """The run's clock, so a test is never started or read blind to the deadline."""
    try:
        out = subprocess.run([str(root / 'clock')], capture_output=True, text=True,
                             timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return ''
    return ' '.join(line for line in out.splitlines()
                    if line.startswith(('deadline_utc=', 'remaining=')))


print(f'Requested {request_id}. Stop/avoid GPU work until completion. Report: {report}', flush=True)
print(time_left(), flush=True)
while args.wait:
    try:
        body = json.loads((fallback if fallback.exists() else report).read_text())
        if body.get('status') != 'running':
            print(json.dumps(body, indent=2))
            print(time_left())
            print('A pass covers exactly the tree this test froze; a version or '
                  'checkpoint captures the tree as it stands when you request it.')
            raise SystemExit(0 if body.get('status') == 'passed' else 1)
    except (OSError, ValueError):
        pass
    time.sleep(1)
'''


def deliver(workspace: Path, *, instructed: bool = False) -> Path:
    """Deliver either atomic publication or split checkpoint/head controls."""
    workspace = Path(workspace)
    (workspace / REQUEST_DIR).mkdir(parents=True, exist_ok=True)
    # Every profile can run the verdict's cold start and full replay in its
    # own container before it publishes.
    full_test = workspace / 'full-test'
    full_test.write_text(FULL_TEST_SCRIPT)
    full_test.chmod(0o755)
    if instructed:
        checkpoint = workspace / "checkpoint"
        checkpoint.write_text(CHECKPOINT_SCRIPT)
        checkpoint.chmod(0o755)
        promote = workspace / "promote"
        promote.write_text(PROMOTE_SCRIPT)
        promote.chmod(0o755)
        return checkpoint
    script = workspace / "version"
    script.write_text(SCRIPT)
    script.chmod(0o755)
    return script


# RAW, so `\n` inside the generated script stays two characters.
# Without the r-prefix Python would resolve it when THIS module is parsed, the
# written file would get a real newline inside an f-string, and
# `./measurements` would die on a SyntaxError.
MEASUREMENTS_SCRIPT = r'''#!/usr/bin/env python3
"""List the AgentX measurements this run has completed, with dollars_per_dollar.

Reads the agent's OWN aiperf artifacts under `agentx/`, and the evaluator's
`development-evidence.json` beside each (written by `--measure-only` and by
`./full-test`). It computes nothing: the request scores, the multiplier and
`dollars_per_dollar` are the evaluator's own numbers, and the p90 values are
read straight out of the export.

It exists because the evidence is spread across one directory per measurement
and a run makes a dozen, so "which of these held every gate" is a question
that otherwise costs a shell pipeline to answer.
"""
import signal

# `./tool | head` closes the pipe early. Die quietly, as a Unix tool
# does, instead of printing a BrokenPipeError traceback that reads as
# the tool failing.
signal.signal(signal.SIGPIPE, signal.SIG_DFL)
import json
import pathlib
import sys

GATES = __GATES_JSON__
ACCURACY = __ACCURACY_JSON__
HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE / "agentx"

# WHEN THE OUTPUT ARRIVED, judged by the SAME module the verdict uses, read
# from the evaluator copy in `tests/final_verdict/`. Not reimplemented here:
# two definitions of "was this streamed" is how this tool and the verdict come
# to disagree about a gate.
#
# A server that holds every token until the request ends and then flushes them
# reports a near-zero TPOT that is the speed of a socket write. The verdict
# withdraws that TPOT; so does this.
sys.path.insert(0, str(HERE / "tests"))
try:
    from final_verdict import stream_timing
except Exception:                                   # noqa: BLE001
    stream_timing = None


def scalar(body, key, stat="avg"):
    value = body.get(key)
    if isinstance(value, dict):
        return value.get(stat)
    return value


rows = []
for export in sorted(ROOT.rglob("profile_export_aiperf.json")):
    # The replay verifier deliberately writes an AgentX aggregate so it can
    # apply the same OSL accounting. Its eight-request public fixture is not a
    # full-window measurement and must not appear in this performance ledger.
    if "lightweight-agentx" in export.parts:
        continue
    try:
        body = json.loads(export.read_text("utf-8"))
    except (OSError, ValueError):
        continue
    window = scalar(body, "benchmark_duration") or 0.0
    got = {
        "p90 TTFT": (body.get("time_to_first_token", {}) or {}).get("p90"),
        "p95 TTFT": (body.get("time_to_first_token", {}) or {}).get("p95"),
        "p90 TPOT": (body.get("inter_token_latency", {}) or {}).get("p90"),
        "p95 TPOT": (body.get("inter_token_latency", {}) or {}).get("p95"),
    }
    timing = (stream_timing.assess(export.parent)
              if stream_timing is not None else {})
    if timing.get("buffered"):
        # Not measured, and an unmeasured gate never holds -- exactly as the
        # verdict judges it.
        got["p90 TPOT"] = got["p95 TPOT"] = None
    verdicts = {name: (False if got[name] is None and timing.get("buffered")
                       else None if got[name] is None
                       else got[name] <= limit)
                for name, limit in GATES.items()}
    decided = [v for v in verdicts.values() if v is not None]
    # The evaluator's own report for this measurement, when it wrote one.
    evidence, label = {}, str(export.relative_to(ROOT)).split("/")[0]
    for parent in export.parents:
        candidate = parent / "development-evidence.json"
        if candidate.is_file():
            try:
                evidence = json.loads(candidate.read_text("utf-8"))
            except (OSError, ValueError):
                evidence = {}
            label = str(parent.relative_to(ROOT))
            break
        if parent == ROOT:
            break
    rows.append({
        "evidence": evidence,
        "timing": timing,
        "run": label,
        "top": str(export.relative_to(ROOT)).split("/")[0],
        "mtime": export.stat().st_mtime,
        "window_s": window,
        "requests": scalar(body, "request_count"),
        "tok_per_s": scalar(body, "output_token_throughput"),
        "osl_mismatch": scalar(body, "osl_mismatch_count"),
        "errors": scalar(body, "error_request_count"),
        "got": got,
        "held": bool(decided) and all(decided),
    })



# ATTEMPTS THAT NEVER PRODUCED A MEASUREMENT, and aiperf's own reason.
#
# Listed because they are the most useful feedback there is. A server that
# cuts a long-context response off mid-transfer aborts every attempt at
# warmup; reporting only "0 completed" would hide that the server, not the
# measurement path, is the problem -- and the verdict's first request fails
# the same way.
REASONS = ("Run aborted", "TransferEncodingError", "ServerDisconnectedError",
           "ClientPayloadError", "Connection refused", "ClientConnectorError",
           "Traceback", "not a valid performance benchmark", "ERROR")
completed = {row["top"] for row in rows}
failed = []
for log in sorted(ROOT.rglob("aiperf.log")):
    if "logs" not in log.parts:
        continue
    run = str(log.relative_to(ROOT)).split("/")[0]
    if run in completed:
        continue
    reason = ""
    try:
        lines = log.read_text("utf-8", errors="replace").splitlines()
    except OSError:
        lines = []
    for marker in REASONS:
        hits = [line for line in lines if marker in line]
        if hits:
            reason = hits[0].split(" - ", 3)[-1].strip()
            break
    failed.append((log.stat().st_mtime, run,
                   reason or "aiperf wrote no aggregate and logged no reason"))
failed.sort()
if not rows and not failed:
    print("no completed measurement in this run yet.")
    print(f"looked under {ROOT}")
    sys.exit(0)

rows.sort(key=lambda r: r["mtime"])
print(f"{len(rows)} completed AgentX measurement(s), oldest first\n")
for row in rows:
    # CORRECTNESS FIRST, the same order the gate is judged in. A run where
    # most requests failed still produces latency percentiles, and reading
    # those as a performance result is how a broken server reads as a slow
    # one: a run where most requests errored looks like a latency miss.
    broken = (row["errors"] or 0) or (row["osl_mismatch"] or 0)
    evidence = row["evidence"]
    value = evidence.get("objective_value")
    if broken:
        mark = "REPLAY GATE FAILED: osl mismatches or errored requests"
    elif evidence and value is not None:
        mark = (f"dollars_per_dollar {value:.4f}  (request attainment "
                f"{evidence.get('request_multiplier') or 0:.4f}; diagnostic gross "
                f"{evidence.get('gross_dollars_per_dollar') or 0:.4f}); "
                "output correctness is verified privately")
    elif evidence:
        mark = ("dollars_per_dollar NOT MEASURED: "
                + str(evidence.get("incomplete_why")
                      or evidence.get("request_scoring_why") or "see the run's log"))
    else:
        mark = ("no evaluator report beside this export (not run through "
                "--measure-only or ./full-test), so no dollars_per_dollar")
    print(f"{row['run']}  window {row['window_s']:.0f}s  "
          f"requests {row['requests'] or 0:.0f}  "
          f"{row['tok_per_s'] or 0:.1f} tok/s  "
          f"osl_mismatch {row['osl_mismatch'] or 0:.0f}  "
          f"errors {row['errors'] or 0:.0f}")
    if row["timing"].get("buffered"):
        print(f"    stream not measurable: {row['timing']['why']}")
    elif row["timing"] and not row["timing"].get("evaluable"):
        # Say so rather than let a TPOT through silently. The verdict always
        # writes per-request records; a development run may not (it can leave
        # only its warmup requests), and then nobody can tell whether its
        # output was streamed.
        print(f"    delivery timing not checked: {row['timing']['why']}")
    counts = ((evidence.get("request_attainment") or {})
              .get("request_score_counts") or {})
    if counts:
        total = (evidence.get("request_attainment") or {}).get("request_count")
        print(f"    requests meeting both p90 bars: {counts.get('p90_both', 0)}/{total}")
    for name, limit in GATES.items():
        value = row["got"][name]
        if value is None:
            if row["timing"].get("buffered") and "TPOT" in name:
                print(f"    {name:9} NOT MEASURED (output was not streamed)"
                      f"   limit {limit:>9,}   MISS")
            else:
                print(f"    {name:9} absent")
            continue
        print(f"    {name:9} {value:12,.1f}   bar {limit:>9,}   "
              f"(aggregate percentile, diagnostic; each request is scored on its own)")
    print(f"    -> {mark}\n")

print("Each value is read from the export named by its row.")

if failed:
    print(f"\n{len(failed)} measurement attempt(s) that did NOT complete, "
          f"oldest first -- a run still in progress also appears here")
    for _, run, reason in failed:
        print(f"  {run:16} {reason[:220]}")

print("\nOutput correctness is verified privately after the run; "
      "nothing in this workspace measures it.")
'''


def _frozen_gates(task_dir: Path | None) -> dict:
    """This task's p90 bars (TTFT and TPOT per percentile), in the order a
    reader expects them.

    Returns an empty mapping when the task declares none, which makes
    `./measurements` print the numbers without verdicts rather than inventing
    a bar. A tool that guesses a threshold is worse than one that omits it.
    """
    if task_dir is None:
        return {}
    try:
        import yaml
        spec = yaml.safe_load(
            (Path(task_dir) / "contract.yaml").read_text("utf-8")) or {}
    except (OSError, ValueError, ImportError):
        return {}
    levels = spec.get("service_levels") or []
    gates: dict[str, float] = {}
    for level in sorted(levels, key=lambda r: int(r.get("percentile", 0))):
        pct = int(level.get("percentile", 0))
        if level.get("ttft_ms") is not None:
            gates[f"p{pct} TTFT"] = float(level["ttft_ms"])
        if level.get("tpot_ms") is not None:
            gates[f"p{pct} TPOT"] = float(level["tpot_ms"])
    return gates


def _frozen_accuracy(task_dir: Path | None) -> dict:
    """Nothing, deliberately: how output correctness is judged is private, so
    `./measurements` states no bar."""
    del task_dir
    return {}


def deliver_measurements(workspace: Path, gates: dict,
                         accuracy: dict | None = None) -> Path:
    """Put `./measurements` in the workspace and make it runnable.

    The task's p90 bars are baked in at build time rather than read from
    the task at run time, for the same reason the clock bakes its deadline:
    the tool must answer with the bar this run is judged against, and a tool
    that re-derives the bar can disagree with the verdict.
    """
    workspace = Path(workspace)
    script = workspace / "measurements"
    script.write_text(
        MEASUREMENTS_SCRIPT
        .replace("__GATES_JSON__", json.dumps(gates))
        .replace("__ACCURACY_JSON__", json.dumps(accuracy or {})),
        encoding="utf-8")
    script.chmod(0o755)
    return script


def deliver_clock(workspace: Path, deadline_utc: str) -> Path:
    """Put a live, agent-readable time-budget clock in the workspace."""
    workspace = Path(workspace)
    script = workspace / "clock"
    body = CLOCK_SCRIPT.replace("__DEADLINE_JSON__", json.dumps(deadline_utc))
    script.write_text(body, encoding="utf-8")
    script.chmod(0o755)
    return script


def deliver_gpu_clock(workspace: Path) -> Path:
    """Deliver diagnostics, not an automatic reset or a GPU lease."""
    source = Path(__file__).resolve().parents[2] / "scripts/agent/gpu_clock.py"
    script = Path(workspace) / "gpu-clock"
    shutil.copyfile(source, script)
    script.chmod(0o755)
    return script


def pending(workspace: Path, seen: set[str]) -> list[dict]:
    """Requests the agent has made that the harness has not acted on yet."""
    where = Path(workspace) / REQUEST_DIR
    if not where.is_dir():
        return []
    out = []
    for path in sorted(where.glob("*.json")):
        if path.name in seen:
            continue
        try:
            body = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            # A half-written request is read on the next poll, not discarded.
            continue
        action = str(body.get("action", "publish"))
        if action not in {"publish", "checkpoint", "promote", "full_test"}:
            continue
        request = {"file": path.name,
                   "requested_utc": str(body.get("requested_utc", "")),
                   "action": action,
                   "target": str(body.get("target", "")),
                   "note": str(body.get("note", ""))}
        if action == "full_test":
            try:
                request["duration_s"] = int(body.get("duration_s") or 1800)
            except (TypeError, ValueError):
                request["duration_s"] = 0          # refused by the full test
        out.append(request)
    return out


# --- ownership -------------------------------------------------------------------

def _docker(*args: str, timeout: float = 120.0) -> subprocess.CompletedProcess:
    return subprocess.run(["docker", *args], capture_output=True, text=True,
                          timeout=timeout, check=False)


class Ownership:
    """The run's resource ledger. Acquire before use, release by exact id."""

    def __init__(self, journal: Journal) -> None:
        self.journal = journal

    # -- acquire -----------------------------------------------------------
    def took_process(self, pid: int, what: str, argv: list[str]) -> None:
        self.journal.append("owned_process", pid=int(pid), what=what,
                            argv=list(argv))

    def took_container(self, name: str, container_id: str = "") -> None:
        """Claim a container by name, and record WHICH one that name meant.

        A name is chosen and reused -- a compose project recreates `main` under
        the same name on the next start -- while an id is what docker actually
        made, so without it a teardown reading the journal could not tell the
        container it claimed from a later one wearing its name. It is also the
        cheapest evidence of which backend produced it.

        Looked up when the caller does not supply it, and never fatal: the run
        HAS taken the container by this point, and failing the claim because
        docker did not answer would lose the record of a container that is
        running.
        """
        if not container_id:
            # A Modal sandbox id IS the name, minus its prefix. Asking docker
            # for it would fail and record an empty id.
            if backend.is_modal(name):
                container_id = backend.sandbox_id(name)
            elif backend.is_local(name):
                container_id = name
            else:
                try:
                    container_id = container_id_of(name)
                except Exception:                               # noqa: BLE001
                    container_id = ""
        self.journal.append("owned_container", name=name,
                            container_id=container_id)

    #: Never claimed, whatever asks -- READ FROM CONFIGURATION, not hardcoded.
    #:
    #: A second, hardcoded copy of the exclusion policy would drift from
    #: `SERVERBENCH_EXCLUDED_GPUS`: a run could then hold a card whose claim
    #: was refused, so nothing outside the run would know it was taken.
    @classmethod
    def forbidden_devices(cls) -> frozenset[int]:
        """Cards this fleet must never claim, from `SERVERBENCH_EXCLUDED_GPUS`.

        The same variable `scripts/env.sh` exports and `evolve-common.sh`
        excludes from selection, so "unselectable" and "unclaimable" cannot
        drift apart.
        """
        raw = os.environ.get("SERVERBENCH_EXCLUDED_GPUS", "")
        out = set()
        for part in raw.replace(" ", "").split(","):
            if part.isdigit():
                out.add(int(part))
        return frozenset(out)

    #: Where the fleet's card claims live. One ownership mechanism means the
    #: evolve driver writes the same claim `scripts/deploy/with-gpu` and
    #: `final_validate.lease` read.
    CLAIMS = Path(os.environ.get("SERVERBENCH_CARD_CLAIMS",
                                 "/tmp/serverbench-card-claims"))

    def took_device(self, index: int, uuid: str) -> bool:
        """Record the card AND claim it where the rest of the fleet can see.

        A device recorded only in the run's own journal is invisible to the
        rest of the fleet, which could take it mid-run.

        Pre-claiming cannot fix it from outside, because the selector declines
        claimed cards -- so claiming before launch locks the run out of its own
        device. The claim has to be written by the run, after selection, which
        is here.

        Returns whether the shared claim was acquired. The caller records the
        attempt either way and must refuse before container setup when it is
        false; proceeding with a card the rest of the fleet sees as available
        recreates the collision this claim exists to prevent.
        """
        self.journal.append("owned_device", index=int(index), uuid=uuid)
        claimed, why = "", ""
        if int(index) in self.forbidden_devices():
            # An excluded card must never be held. It is excluded by
            # `evolve-common.sh` before selection, so the shell path cannot
            # reach here -- but a claim written for it would tell every
            # scheduler on this host that WE hold the one device we may never
            # hold, and it outlives whatever wrote it.
            #
            # Refused here rather than only upstream, because "unselectable" and
            # "unclaimable" are different guarantees and only the second one
            # holds against a caller that already has an index in hand.
            self.journal.append(
                "device_claim_refused", index=int(index),
                why=(f"device {index} is the operator's card on this fleet and "
                     f"is never claimable, whatever handed us the index"))
            return False
        try:
            self.CLAIMS.mkdir(parents=True, exist_ok=True)
            path = self.CLAIMS / f"gpu{int(index)}"
            path.mkdir()
            (path / "owner").write_text(str(os.getpid()), "utf-8")
            (path / "what").write_text("evolve", "utf-8")
            (path / "since").write_text(utc_now(), "utf-8")
            claimed = str(path)
        except FileExistsError:
            # A claim already there. Whose, and is it alive?
            #
            # A LIVE claim is theirs and stays: the selector should have
            # declined the card, and a claim is its owner's to remove.
            #
            # A DEAD claim (an owner long gone, or an empty claim directory
            # with no owner file) is garbage, and leaving it is not caution --
            # it is how a card ends up in use by a run that cannot claim it,
            # with nothing on the fleet recording it.
            #
            # Taken over, never deleted-and-recreated: the directory stays and
            # the owner file is rewritten, so a reader mid-swap sees one owner
            # or the other and never no claim at all.
            owner = ""
            with contextlib.suppress(OSError):
                owner = (path / "owner").read_text("utf-8").strip()
            # ALIVE means the process EXISTS, not that we may signal it.
            #
            # `os.kill(pid, 0)` raises ProcessLookupError when there is no such
            # process and PermissionError when there is one we do not own.
            # Catching OSError for both would read another USER's live claim
            # as dead and steal their card -- the precise failure the whole
            # claim mechanism exists to prevent. Only ProcessLookupError means
            # gone.
            alive = False
            if owner.isdigit():
                try:
                    os.kill(int(owner), 0)
                    alive = True
                except ProcessLookupError:
                    alive = False
                except (PermissionError, OSError, ValueError):
                    # It answers to somebody. Not ours to take.
                    alive = True
            if alive:
                why = (f"a claim already exists for this card (owner {owner}, "
                       f"still running) and is not ours to take")
            else:
                try:
                    (path / "owner").write_text(str(os.getpid()), "utf-8")
                    (path / "what").write_text("evolve", "utf-8")
                    (path / "since").write_text(utc_now(), "utf-8")
                    claimed = str(path)
                    why = (f"took over a DEAD claim "
                           f"(previous owner {owner or 'none recorded'} is not "
                           f"running). A claim nobody owns is not protection, "
                           f"it is a card held with no record of who holds it")
                except OSError as exc:
                    why = (f"a dead claim could not be taken over: "
                           f"{type(exc).__name__}: {exc}")
        except OSError as exc:
            why = f"{type(exc).__name__}: {exc}"
        self.journal.append("device_claimed", index=int(index),
                            claim=claimed, why=why)
        return bool(claimed)

    # -- read back ---------------------------------------------------------
    @property
    def processes(self) -> list[dict]:
        return self.journal.events("owned_process")

    @property
    def containers(self) -> list[str]:
        return [e["name"] for e in self.journal.events("owned_container")]

    @property
    def devices(self) -> list[dict]:
        return self.journal.events("owned_device")

    def owns_container(self, name: str) -> bool:
        return name in self.containers

    def owns_process(self, pid: int) -> bool:
        return any(int(e["pid"]) == int(pid) for e in self.processes)

    # -- release -----------------------------------------------------------
    def stop_process(self, pid: int, *, sig: int = signal.SIGTERM) -> str:
        """Signal one pid this run recorded starting. Anything else refuses."""
        if not self.owns_process(pid):
            raise PermissionError(
                f"pid {pid} is not in this run's journal. A run may only "
                f"signal a process it recorded starting")
        try:
            os.kill(pid, sig)
        except ProcessLookupError:
            self.journal.append("process_released", pid=pid, how="already gone")
            return "already gone"
        self.journal.append("process_released", pid=pid, how=f"signal {sig}")
        return f"signal {sig}"

    def stop_container(self, name: str, *, remove: bool = True) -> str:
        """Stop one container by exact name, and only one this run started."""
        if not self.owns_container(name):
            raise PermissionError(
                f"container {name!r} is not in this run's journal. There is no "
                f"prefix or pattern form of this call, deliberately")
        # A SANDBOX IS NOT STOPPED BY DOCKER, AND THIS IS THE ONLY TEARDOWN AN
        # ABORTED MODAL RUN REACHES.
        #
        # `run.py` releases the agent's sandbox through `backend.remove` on the
        # way to final validation. `release_all` is the other path: a run that
        # fails in setup, an operator's Ctrl-C, a HarnessSetupFailed. There
        # `docker stop modal:sb-...` would name no docker container, and the
        # sandbox would keep its GPUs until its own timeout. A leaked local
        # card is wasted; a leaked sandbox is billed.
        if backend.is_modal(name) or backend.is_local(name):
            how = backend.remove(name)
            self.journal.append("container_released", name=name,
                                backend="modal" if backend.is_modal(name)
                                else "local", how=how)
            return how
        got = _docker("stop", name)
        if remove:
            _docker("rm", name)
        how = "stopped" if got.returncode == 0 else f"stop rc={got.returncode}"
        self.journal.append("container_released", name=name, how=how)
        return how

    def _drop_claim(self, index: int) -> str:
        """Remove this run's own card claim. Never anyone else's.

        Checked by recorded owner, because removing a claim we did not write is
        how a card gets handed to two jobs at once.
        """
        path = self.CLAIMS / f"gpu{int(index)}"
        try:
            owner = (path / "owner").read_text("utf-8").strip()
        except OSError:
            return "no claim"
        if owner != str(os.getpid()):
            return f"claim belongs to {owner or 'someone else'}; left alone"
        shutil.rmtree(path, ignore_errors=True)
        return "claim released"

    def release_device(self, index: int) -> str:
        """Record that this run has given a card back, and whether it is clear.

        A device is not stopped the way a process is -- it comes free when the
        container and the server holding it are gone -- so this releases nothing
        by itself and is deliberately called LAST, after both. What it adds is
        the evidence, the counterpart of `took_device`: "did this run give the
        card back?" is answered per device rather than inferred from
        `container_released`, so a four-device run that freed three of them
        does not look like one that freed all four.

        Reports what is still on the card rather than acting on it: this run may
        only signal what it started, and by here it has already stopped all of
        that. Anything remaining belongs to somebody else, or is ours and did
        not die -- and those are different problems that a number cannot tell
        apart.
        """
        if not any(int(e["index"]) == int(index) for e in self.devices):
            raise PermissionError(
                f"device {index} is not in this run's journal. A run may only "
                f"release a card it recorded taking")
        remaining = compute_processes(int(index))
        how = "clear" if not remaining else f"{len(remaining)} process(es) remain"
        # And give the card claim back, so the next job's selector sees the
        # card free. A claim outliving its run makes the card's ownership
        # ambiguous.
        claim = self._drop_claim(int(index))
        self.journal.append("device_released", index=int(index), how=how,
                            remaining=remaining, claim=claim)
        return how

    def release_all(self) -> list[dict]:
        """Release exactly what this run took, and report each outcome.

        Order matters and is the reason devices come last: a card is only
        genuinely free once the container and the server process on it are gone,
        so asking about it before that would record a card as busy on every run.

        Idempotent by construction -- every step is `suppress`ed and each
        records its own outcome, so a second call after a partial teardown
        reports "already gone" rather than raising.
        """
        out = []
        for name in dict.fromkeys(self.containers):
            with contextlib.suppress(Exception):
                out.append({"container": name, "how": self.stop_container(name)})
        for e in self.processes:
            with contextlib.suppress(Exception):
                out.append({"pid": e["pid"],
                            "how": self.stop_process(int(e["pid"]))})
        for e in self.devices:
            with contextlib.suppress(Exception):
                out.append({"device": e["index"],
                            "how": self.release_device(int(e["index"]))})
        return out


def container_id_of(name: str) -> str:
    """The full id docker gave this container, or "" if it cannot be asked."""
    got = _docker("inspect", "--format", "{{.Id}}", name)
    return got.stdout.strip() if got.returncode == 0 else ""


def container_is_up(name: str) -> bool:
    got = _docker("inspect", "--format", "{{.State.Running}}", name)
    return got.returncode == 0 and got.stdout.strip() == "true"


def device_uuid(index: int) -> str:
    got = subprocess.run(
        ["nvidia-smi", f"--id={index}", "--query-gpu=uuid",
         "--format=csv,noheader"], capture_output=True, text=True, check=False)
    return got.stdout.strip() if got.returncode == 0 else ""


def compute_processes(index: int) -> list[str]:
    """Every compute process on a device, ours included."""
    uuid = device_uuid(index)
    if not uuid:
        return []
    got = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory",
         "--format=csv,noheader"], capture_output=True, text=True, check=False)
    return [line.strip() for line in got.stdout.splitlines()
            if line.strip().startswith(uuid)]


def foreign_processes(index: int) -> list[str]:
    """Compute processes on a device that are NOT this run's.

    Used to refuse a device before taking it, never to clear one. Called at run
    start, when nothing of ours is on the card yet, so everything it sees
    belongs to somebody else.
    """
    return compute_processes(index)


def container_reservations(index: int, uuid: str = "") -> list[str]:
    """Running containers that reserve this card while CUDA may be idle."""
    from final_validate.gpu_reservations import reservations_for

    return list(reservations_for(index, uuid or device_uuid(index)))


def _container_of(pid: str) -> str:
    """The container id a pid belongs to, or "" for a host process.

    Read from `/proc/<pid>/cgroup`, which names the container for anything
    docker started.
    """
    try:
        body = pathlib.Path(f"/proc/{int(pid)}/cgroup").read_text("utf-8")
    except (OSError, ValueError):
        return ""
    for line in body.splitlines():
        for part in line.replace("/", " ").split():
            if len(part) >= 64 and all(c in "0123456789abcdef" for c in part):
                return part
            if part.startswith("docker-") and part.endswith(".scope"):
                return part[len("docker-"):-len(".scope")]
    return ""


def holder_interpreter() -> str:
    """An interpreter on this host that can import torch AND see a GPU.

    `$SERVERBENCH_PY` deliberately carries no torch -- the harness core must
    import inside a bare image, and `tests/repo` enforces it -- so the holder
    cannot run under the interpreter that starts it. Resolved at run time and
    never hardcoded to a path from a document, the same rule as every other
    interpreter on this fleet.

    Returns "" when nothing here can hold a card, which is not fatal: the card
    is still claimed, it simply will not LOOK busy to a neighbour reading
    nvidia-smi, and the run says so in its journal.
    """
    named = os.environ.get("SERVERBENCH_HOLDER_PY", "").strip()
    candidates = [named] if named else []
    candidates += [
        str(pathlib.Path.home() / "anaconda3" / "bin" / "python"),
        "/usr/bin/python3",
    ]
    for candidate in candidates:
        if not candidate or not pathlib.Path(candidate).is_file():
            continue
        probe = subprocess.run(
            [candidate, "-c",
             "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 9)"],
            capture_output=True, text=True, timeout=120, check=False)
        if probe.returncode == 0:
            return candidate
    return ""


def _container_id(name_or_id: str) -> str:
    """The full container ID for a name, cached. "" if docker cannot say."""
    if not name_or_id:
        return ""
    cached = _CONTAINER_IDS.get(name_or_id)
    if cached is not None:
        return cached
    got = subprocess.run(["docker", "inspect", "-f", "{{.Id}}", name_or_id],
                         capture_output=True, text=True, check=False)
    found = got.stdout.strip() if got.returncode == 0 else ""
    _CONTAINER_IDS[name_or_id] = found
    return found


#: One `docker inspect` per container, not one per tick per card.
_CONTAINER_IDS: dict[str, str] = {}


def processes_not_in(index: int, container: str,
                     *, mine: tuple[str, ...] = ()) -> list[str]:
    """Compute processes on a device that are NOT inside `container`.

    MID-RUN, `foreign_processes` IS THE WRONG QUESTION, and its own docstring
    says so: "called at run start, when nothing of ours is on the card yet, so
    everything it sees belongs to somebody else." Mid-run it would record the
    agent's own server -- which restarts with a new pid on every iteration --
    as a stranger.

    So ownership is decided by the CONTAINER, which is the only thing that
    actually knows: the agent's server runs inside it, whatever pid it has this
    minute, and a neighbour's job does not.
    """
    if not container and not mine:
        return []                      # nothing to compare against; say nothing
    # A NAME IS NOT AN ID, and comparing them matches nothing.
    #
    # `container` is the NAME the driver records -- `sb<run>-main-1` -- and
    # `_container_of` returns the 64-hex cgroup ID, so the name is resolved
    # through docker before comparing.
    wanted = _container_id(container) if container else ""
    if not wanted and not mine:
        return []                      # cannot establish ours; say nothing
    out = []
    for line in compute_processes(index):
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        if parts[1] in mine:
            continue                      # a process this run started itself
        found = _container_of(parts[1])
        if found and (found.startswith(wanted[:12])
                      or wanted.startswith(found[:12])):
            continue
        out.append(line)
    return out


# --- building and booting --------------------------------------------------------

ensure_harness_on_path()


# One candidate policy is used by host publication and the packaged
# development verifier. This module itself still needs only stdlib.
from final_validate import candidate_bundle as _candidate_bundle  # noqa: E402

from . import delivery  # noqa: E402
from .record import RunIdentity, utc_now  # noqa: E402

#: Never candidate source, whatever appears in the workspace.
NEVER_CANDIDATE = _candidate_bundle.NEVER_CANDIDATE

class HarnessSetupFailed(RuntimeError):
    """The workspace is wrong. Never charged to the candidate."""


@dataclasses.dataclass(frozen=True)
class BuiltWorkspace:
    path: Path
    workspace_id: str
    manifest: tuple[str, ...]
    delivered_files: int
    build_seconds: float
    size_bytes: int
    excluded: tuple[str, ...]
    external_mounts: tuple[str, ...]
    built_utc: str

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["path"] = str(self.path)
        for key in ("manifest", "excluded", "external_mounts"):
            d[key] = list(getattr(self, key))
        d["kind"] = "workspace"
        d["schema_version"] = 1
        return d

    def for_agent(self, *, task_id: str, deadline_utc: str,
                  stop: str) -> dict:
        """What `WORKSPACE.json` tells the agent: the run's facts, as data.

        The brief says this file records the run's deadline, so it does. The
        host's record (`to_dict`, journalled as `workspace_built`) keeps the
        host path, the harness's exclusion list and the delivered mounts;
        none of those is true or useful inside the container, where the
        workspace is `/workspace` and the delivery rules are in
        `DELIVERY_POLICY.json`.
        """
        return {"kind": "workspace", "schema_version": 2,
                "path": AGENT_WORKSPACE, "task_id": task_id,
                "deadline_utc": deadline_utc, "stop": stop,
                "manifest": list(self.manifest)}


#: Where the workspace is mounted inside the agent's container
#: (`scripts/agent/run_agent.py::WORKSPACE`, `modal_env.py::WORKSPACE`).
AGENT_WORKSPACE = "/workspace"


def _stop_rule(journal: Journal) -> str:
    """The run's `--stop`, as `run_created` recorded it; `timebox` when the
    journal has none (a builder called outside a run, as tests do)."""
    try:
        created = journal.last("run_created") or {}
    except Exception:                                          # noqa: BLE001
        created = {}
    return str(created.get("stop") or "timebox")


def candidate_manifest(workspace: Path) -> list[str]:
    """The candidate's own files, workspace-relative and sorted.

    Explicit rules, applied to the top `candidate_bundle.MAX_CANDIDATE_DEPTH`
    levels. A scheduler, a KV cache, a batching policy and kernels are
    separate concerns belonging in separate modules, and the obvious layout
    for that is `engine/attention/flash.py`, three deep; a shallower limit
    would drop those files silently.

    Four is enough for any layout a person would actually write and still
    excludes the deep trees that are caches and vendored packages -- and
    `NEVER_CANDIDATE` catches those by name regardless of depth.
    """
    return candidate_delivery_report(workspace)["files"]


def candidate_delivery_report(workspace: Path) -> dict:
    return _candidate_bundle.audit(Path(workspace))


#: Names a seed may carry that the WORKSPACE already owns, or that belong to a
#: published version's record rather than to its source.
#:
#: A `--seed` is usually a task's `demo/` starter, which is pure source. But a
#: run started from a prototype extends the lineage that path is inside, and a
#: version directory is source PLUS the reproduction kit PLUS its own
#: evidence. Promoting all of it would do two bad things at once:
#:
#: * `os.replace` raises `FileExistsError` on a directory that is already
#:   there, and the harness has already delivered `tests/` and `wait/`, so
#:   seeding from a version would crash.
#: * A version's datasheet and measurements are run evidence; a successor
#:   seeded from its predecessor would find them in its own workspace root.
#:
#: The harness's copies win because they are delivered fresh for THIS run; the
#: version's are a snapshot taken when it was published.
PROMOTION_SKIPPED: tuple[str, ...] = (
    # the harness delivers these itself, for this run
    "baselines", "measure", "tests", "wait", "knowledge", "MEMORY",
    "microbench", "task_context", "clock", "BOUNDARY_CHECKS.md", ".feedback",
    # A SECOND DOCUMENT ADDRESSED TO THE AGENT. `instruction.md` is the one
    # document the agent is told anything in; a `DEMO.md` restating its gates
    # is how one copy goes stale.
    "DEMO.md",
    # a published version's own record, which is evidence and not source
    "DATASHEET.json", "DATASHEET.md", "MEASUREMENTS.jsonl", "REVIEW.md",
    "TRAJECTORY.md", "trajectory.atif.json", "VERSION.json",
)


def promotable(names) -> list[str]:
    """Which seed entries may be promoted to the workspace root.

    A pure function so the rule is testable without building a workspace, and
    so the list above is the only place it is stated.
    """
    return [n for n in names if n not in PROMOTION_SKIPPED]


#: A file whose name carries one of these is HELD OUT and never travels.
#:
#: A held-out trace, a reference answer or a consensus file inside an
#: agent-readable workspace is the one leak that invalidates every result the
#: task ever produces. Matched on the NAME, so `-final`
#: catches `trace-final.json` and leaves `tests/final_verdict/` alone.
HELD_OUT_MARKERS = ("-qualification", "-final", "-aa", "consensus-",
                    "references")

#: The two files a task's data lives in, and the only two delivered as its
#: contract. `contract.yaml` is the frozen bars; `reference.yaml` is the
#: measured evidence. No answer-bearing sample is delivered at all:
#: `reference.yaml`'s correctness block is stripped from the delivered copy
#: (`PRIVATE_KEYS`), and `tests/final_verdict/` carries no pool.
TASK_DATA_FILES = ("contract.yaml", "reference.yaml")

#: Build artefacts that accumulate in a task tree and must never travel.
#:
#: `.pytest_cache` appears the moment anyone runs the suite, which imports
#: each task's packaged verdict. Both are gitignored, so they are invisible in
#: review and would arrive in the agent's workspace anyway: a stale `.pyc`
#: beside a module the agent is meant to read, and a pytest cache naming test
#: ids from the host.
BUILD_NOISE = ("__pycache__", ".pytest_cache")


def _is_build_noise(path: Path) -> bool:
    return any(part in BUILD_NOISE for part in path.parts)


#: Keys stripped from a task data file on its way into a workspace. How output
#: correctness is judged is private: the agent is told it is verified
#: independently and builds its own checks.
PRIVATE_KEYS = {"reference.yaml": ("correctness",)}


def _write_without(source: Path, target: Path, keys: tuple[str, ...]) -> None:
    import yaml

    body = yaml.safe_load(source.read_text("utf-8")) or {}
    for key in keys:
        body.pop(key, None)
    target.write_text(
        "# Delivered copy of this task's reference.yaml.\n"
        + yaml.safe_dump(body, sort_keys=False, allow_unicode=True), "utf-8")


def _deliver_task_context(task_dir: Path, workspace: Path, *,
                          seed: Path | None = None) -> list[str]:
    """Give the evaluator its task contract: two files, and nothing else.

    The agent is shipped `tests/final_verdict/` so it can measure its own
    server against the real gates before its run ends, and that copy reads the
    task through `spine.load_contract`. It therefore needs the task's data
    laid out where `resolve_task_dir` globs for it -- hence
    `task_context/<TASK_ID>/`, which is what makes the task findable by the
    delivered verdict.

    ## What travels

    `contract.yaml` -- the p90 bars, the prices, the devices, the window.
    `reference.yaml` -- the SGLang concurrency search, the reference arms and
    their launch lines. Its `correctness` block is private and is stripped on
    the way in (`PRIVATE_KEYS`).

    That is the whole of a task's data.

    The held-out check is cheap, reads the delivered tree rather than
    trusting this loop, and keeps a task that ever gains a frozen trace from
    leaking it.
    """
    task_dir = Path(task_dir)
    into = Path(workspace) / "task_context" / task_dir.name
    into.mkdir(parents=True, exist_ok=True)

    written: list[str] = []
    for name in TASK_DATA_FILES:
        source = task_dir / name
        if not source.is_file():
            continue
        if name in PRIVATE_KEYS:
            _write_without(source, into / name, PRIVATE_KEYS[name])
        else:
            shutil.copy2(source, into / name)
        written.append(name)

    # Checked by READING the delivered tree, not by trusting the loop above.
    # A leak found here is a run that must not start.
    leaked = [str(p.relative_to(into)) for p in into.rglob("*")
              if p.is_file() and any(m in p.name for m in HELD_OUT_MARKERS)]
    if leaked:
        raise RuntimeError(
            f"held-out files reached the workspace: {leaked[:5]}. Every result "
            f"this task has produced would be unciteable. Nothing was started")
    return written


def _task_model_id(task_dir: Path) -> str:
    """The model the task serves, from its own harness module."""
    try:
        from final_validate import tasks as task_modules
        return str(task_modules.python_module(Path(task_dir).name).MODEL_ID)
    except Exception:                                          # noqa: BLE001
        return ""


def build(*, identity: RunIdentity, task_dir: Path, repo: Path,
          journal: Journal, seed: Path | None = None,
          from_tree: tuple = (), agents_dir: Path | None = None,
          claude_project_dir: Path | None = None,
          skill_dir: Path | None = None,
          ncu_skill_dir: Path | None = None,
          kernel_skill_dir: Path | None = None,
          kernelwiki_skill_dir: Path | None = None,
          kernelwiki_overlay_dir: Path | None = None,
          brief: str = "") -> BuiltWorkspace:
    """Materialise the workspace for one run."""
    started = time.monotonic()
    workspace = identity.root / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    task_dir = Path(task_dir)
    delivered = delivery.deliver(workspace, delivery.Sources(
        seed=Path(seed) if seed else None,
        public_tests=task_dir / "tests",
        agents=Path(agents_dir) if agents_dir else None,
        claude_project=Path(claude_project_dir) if claude_project_dir else None,
        serving_skill=Path(skill_dir) if skill_dir else None,
        ncu_report_skill=Path(ncu_skill_dir) if ncu_skill_dir else None,
        kernel_skill=Path(kernel_skill_dir) if kernel_skill_dir else None,
        kernelwiki_skill=(Path(kernelwiki_skill_dir)
                          if kernelwiki_skill_dir else None),
        kernelwiki_overlay=(Path(kernelwiki_overlay_dir)
                            if kernelwiki_overlay_dir else None),
        model_id=_task_model_id(task_dir),
    ))
    _deliver_task_context(task_dir, workspace,
                          seed=Path(seed) if seed else None)
    # No short-prompt microbenchmark is delivered: decode cost depends on KV
    # context length, so it is not comparable with this long-context scenario.
    journal.append(
        "microbenchmark_not_delivered",
        why="short-prompt timing is not comparable with the long-context "
            "scenario",
        note="the task's full scenario window is the comparable measurement")

    # The seed is the candidate's starting source, so it lives at the root
    # where `serve.sh` and `server.py` are what the brief names. Delivered into
    # `seed/` and promoted here, rather than left one directory down where the
    # brief's commands would not resolve.
    seed_dir = workspace / "seed"
    if seed_dir.is_dir():
        names = sorted(p.name for p in seed_dir.iterdir())
        keep = set(promotable(names))
        skipped_from_seed: list[str] = []
        for item in sorted(seed_dir.iterdir()):
            if item.name in keep:
                item.replace(workspace / item.name)
            else:
                # Journal every seed entry that is intentionally not promoted.
                skipped_from_seed.append(item.name)
                if item.is_dir():
                    shutil.rmtree(item, ignore_errors=True)
                else:
                    item.unlink(missing_ok=True)
        shutil.rmtree(seed_dir, ignore_errors=True)
        if skipped_from_seed:
            journal.append(
                "seed_entries_not_promoted", names=sorted(skipped_from_seed),
                why=("a published version carries the reproduction kit and its "
                     "own record as well as its source. The harness delivers "
                     "the kit fresh for this run, and a version's datasheet "
                     "and measurements are run evidence, not "
                     "source"))

    # A published predecessor head, when present, replaces the seed source.
    for rel, blob in from_tree:
        target = workspace / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(blob)

    # Standard tasks use `./version`; instructed tasks deliberately separate
    # immutable checkpoint creation from final-head selection.
    deliver(workspace, instructed="_instructed_" in task_dir.name)
    deliver_clock(workspace, identity.deadline_utc)
    deliver_gpu_clock(workspace)
    # `./measurements` -- the agent reading back its OWN aiperf exports
    # against this task's p90 bars. The bars are baked in at build
    # time, like the clock's deadline: a tool that re-derives the bar can
    # disagree with the verdict, and then there are two bars.
    deliver_measurements(workspace, _frozen_gates(task_dir),
                         _frozen_accuracy(task_dir))

    # The brief is written LAST, after everything it describes exists, and is
    # composed from the workspace itself, so it cannot name a directory the
    # workspace does not have.
    if callable(brief):
        brief = brief(workspace)
    if brief:
        (workspace / "AGENT_TASK.md").write_text(brief)
    bundle = candidate_delivery_report(workspace)
    files = bundle['files']
    (workspace / 'DELIVERY_POLICY.json').write_text(
        json.dumps(bundle['policy'], indent=2) + '\n')
    (workspace / 'DELIVERY_REPORT.json').write_text(
        json.dumps(bundle, indent=2) + '\n')
    (workspace / "MAP.md").write_text(_map(workspace, files))

    built = BuiltWorkspace(
        path=workspace, workspace_id=identity.workspace_id,
        manifest=tuple(files),
        delivered_files=sum(1 for p in workspace.rglob("*") if p.is_file()),
        build_seconds=round(time.monotonic() - started, 2),
        size_bytes=sum(p.stat().st_size for p in workspace.rglob("*")
                       if p.is_file()),
        excluded=tuple(NEVER_CANDIDATE),
        external_mounts=tuple(sorted(delivered.mounted)),
        built_utc=utc_now())
    (workspace / "WORKSPACE.json").write_text(json.dumps(
        built.for_agent(task_id=identity.task_id,
                        deadline_utc=identity.deadline_utc,
                        stop=_stop_rule(journal)),
        indent=1, sort_keys=True) + "\n")
    journal.append("workspace_built", **built.to_dict())
    return built


def _map(workspace: Path, files: list[str]) -> str:
    lines = ["# Workspace map", "",
             "Generated at build time. These are the files you own; everything",
             "else here is delivered and is not yours to change.", ""]
    lines += [f"- `{f}`" for f in files]
    delivered = (
        ("tests", "public tests and the agent-readable evaluator copy"),
        ("microbench", "the development sample and its scope manifest"),
        ("task_context", "the public task contract and non-held-out context"),
        ("baselines", "baseline artifacts supplied for this task"),
        ("MEMORY", "historical lineage notes"),
        ("wait", "server readiness helpers"),
        ("clock", "the live remaining-time clock"),
        ("gpu-clock", "read-only clock diagnosis and guarded single-GPU recovery"),
        (".claude/skills", "reference skills, discovered by the CLI"),
    )
    present = [(name, why) for name, why in delivered
               if (Path(workspace) / name).exists()]
    lines += ["", "## Delivered, not yours"]
    lines += ([f"- `{name}{'/' if (Path(workspace) / name).is_dir() else ''}` -- {why}."
               for name, why in present]
              or ["- No additional delivered files are present."])
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------- boot contract
@dataclasses.dataclass(frozen=True)
class BootReport:
    ok: bool
    checks: tuple[dict, ...]
    seconds: float

    def to_dict(self) -> dict:
        return {"kind": "workspace_boot", "schema_version": 1, "ok": self.ok,
                "seconds": self.seconds, "checks": [dict(c) for c in self.checks]}

    def failure(self) -> str:
        bad = [c for c in self.checks if not c["ok"]]
        return "; ".join(f"{c['check']}: {c['detail']}" for c in bad)


def foreign_mounts(docker_mounts: str, workspace: Path) -> list[str]:
    """Mounts sourced from a run that is not this one.

    `docker_mounts` is `docker inspect -f '{{range .Mounts}}{{.Source}}->
    {{.Destination}} {{end}}'`.

    A mount is foreign when its SOURCE sits beside this run rather than inside
    it. Derived from the run's own layout rather than matched against path
    literals, which rot whenever a run id or the run space's location changes.

    A helper, not an inline expression, because the whole check runs behind
    `docker inspect` and could otherwise only be exercised by starting a
    container.
    """
    run_root = Path(workspace).resolve().parent
    namespace_root = run_root.parent
    out: list[str] = []
    for mount in docker_mounts.split():
        source = mount.split("->")[0]
        if not source:
            continue
        try:
            resolved = Path(source).resolve()
        except OSError:
            continue
        if not resolved.is_relative_to(namespace_root):
            continue
        if resolved.is_relative_to(run_root):
            continue
        out.append(mount)
    return out


def boot_contract(*, workspace: Path, container: str, python: str,
                  journal: Journal,
                  withheld: tuple[str, ...] = ()) -> BootReport:
    """Run the workspace from the environment the agent will actually use.

    Real execution, in the container, through the declared interpreter. Every
    failure here is HARNESS_SETUP_FAILED: the candidate has not been asked a
    question yet, so it cannot have got one wrong.
    """
    started = time.monotonic()
    checks: list[dict] = []

    def check(name: str, script: str, want: str = "") -> bool:
        got = subprocess.run(backend.exec_argv(container, "", script),
                             capture_output=True, text=True,
                             timeout=300, check=False)
        ok = got.returncode == 0 and (not want or want in got.stdout)
        checks.append({"check": name, "ok": ok,
                       "detail": (got.stdout + got.stderr).strip()[:300]})
        return ok

    # The declared interpreter, resolved the way a shell resolves it. `test -x`
    # on a bare name always fails, and a boot check that always fails is as
    # useless as one that always passes.
    check("the declared interpreter exists and is real",
          f'command -v {python} >/dev/null && {python} -c '
          f'"import sys; print(sys.executable)"', want="/")
    # One instrument: `aiperf`, the pinned AgentX, which is the same engine
    # the grader copy and the final verdict run.
    # `command -v` prints the resolved PATH, which contains the binary's name;
    # `aiperf --version` prints a bare version with no name in it, so it
    # cannot be matched on "aiperf" alone.
    check("the pinned AgentX the agent measures with is on PATH",
          'command -v aiperf && aiperf --version', want="aiperf")
    check("an advertised AgentX source mount contains AgentX source",
          'if [ -z "${AGENTX_SOURCE:-}" ]; then '
          'echo agentx-source-not-mounted; '
          'elif test -f "$AGENTX_SOURCE/pyproject.toml" '
          '&& test -f "$AGENTX_SOURCE/src/aiperf/__init__.py"; then '
          'echo agentx-source-ok; else echo agentx-source-empty >&2; exit 1; fi',
          want="agentx-source-")
    check("the CUDA compiler promised by the brief is on PATH",
          'command -v nvcc && nvcc --version', want="nvcc")
    check("the candidate skeleton is present and parses",
          f'cd /workspace && {python} -c "import ast, pathlib; '
          f'ast.parse(pathlib.Path(\'server.py\').read_text()); '
          f'print(\'skeleton-ok\')"', want="skeleton-ok")
    # THE DEVELOPMENT LOOP THE BRIEF NAMES.
    #
    # The brief tells the agent to measure with
    # `tests/final_verdict.scenario --measure-only`, which is the SAME
    # evaluator and duration the verdict uses. What must be present is the
    # evaluator, and it must import inside the container. A brief naming a
    # module the workspace cannot import is the defect this whole block exists
    # to catch.
    check("the evaluator the brief tells the agent to measure with imports",
          f'cd /workspace/tests && PYTHONDONTWRITEBYTECODE=1 {python} -c '
          f'"import final_verdict.scenario; print(\'evaluator-ok\')"',
          want="evaluator-ok")
    # THE TOKENIZER AgentX MEASURES WITH, LOADED OFFLINE THE WAY ./full-test LOADS IT.
    #
    # A model snapshot missing files such as `.gitattributes` or `README.md`
    # makes the huggingface_hub inside the pinned AgentX raise
    # IncompleteSnapshotError offline, and every `./full-test` replay would die
    # hours in with "Tokenizer ... is not present in the HuggingFace cache".
    check("AgentX loads the task's pinned tokenizer offline",
          'f=$(ls /workspace/tests/final_verdict/tasks/*.py 2>/dev/null '
          '| grep -v -e /_ -e __init__ | head -1); '
          'id=$(sed -n \'s/^MODEL_ID = "\\(.*\\)"$/\\1/p\' "$f"); '
          'rev=$(sed -n \'s/^MODEL_REVISION = "\\(.*\\)"$/\\1/p\' "$f"); '
          'ax=/opt/agentx/venv/bin/python; [ -x "$ax" ] || ax=python3; '
          '"$ax" -c "import sys; from aiperf.common.tokenizer import Tokenizer; '
          'Tokenizer.from_pretrained(sys.argv[1], revision=sys.argv[2] or \'main\'); '
          'print(\'tokenizer-ok\', sys.argv[1])" "$id" "$rev"',
          want="tokenizer-ok")
    check("the lightweight AgentX replay verifier is present and imports",
          f'cd /workspace && test -x tests/replay-verify && cd tests && '
          f'PYTHONDONTWRITEBYTECODE=1 {python} -c '
          f'"from final_verdict import replay_verify; '
          f'assert replay_verify.REQUEST_COUNT > 0; '
          f'print(\'replay-verifier-ok\')"',
          want="replay-verifier-ok")
    check("no correctness code or answer pool reached the workspace",
          'cd /workspace/tests/final_verdict && '
          'leak=0; for f in agentx_reference.py private_correctness.py '
          'sample.py correctness.py answer_match.py pool; do '
          'if test -e "$f"; then echo "LEAK $f"; leak=1; fi; done; '
          'test "$leak" = 0 && echo no-private-correctness',
          want="no-private-correctness")
    instructed_tools = (workspace / '.claude/skills/'
                        'serving-system-evolution-instructed/scripts')
    if instructed_tools.is_dir():
        check("mandatory instructed profiler and design-check tools import",
              'cd /workspace/.claude/skills/serving-system-evolution-instructed/scripts && '
              f'PYTHONDONTWRITEBYTECODE=1 {python} -c '
              '"import torch.profiler, profile_candidate, design_review; '
              'assert len(design_review.AXES) == 7; '
              'print(\'design-tools-ok\')"', want='design-tools-ok')
    # Every command the generated brief shows the agent must be here. A brief
    # that names something absent tells the agent to use a workflow the
    # harness does not have.
    from .instructions import commands_missing_from, executables_the_brief_names
    missing = commands_missing_from(workspace)
    checks.append({"check": "every command the brief names exists",
                   "ok": not missing,
                   "detail": ("all present" if not missing
                              else f"absent from the workspace: {missing}")})

    # ...and every executable it names resolves IN THE CONTAINER, in BOTH
    # shells. The check above looks in the workspace on the host, so it says
    # nothing about PATH. Debian's `/etc/profile` ASSIGNS PATH rather than
    # appending, so a login shell can discard the container's, and which shell
    # a tool happens to use would decide whether the brief was true.
    for name in executables_the_brief_names(workspace):
        check(f"the brief names `{name}`, and both shells resolve it",
              f"command -v {name} && bash -c 'command -v {name}'", want="/")

    # No other run may be reachable from inside. Each evolve starts fresh:
    # the workspace is built for this run -- never another run's workspace,
    # another run's candidates, or another run's evidence.
    others = subprocess.run(
        backend.exec_argv(
            container, "",
            "ls -a / /workspace 2>/dev/null | grep -E '^(run-[0-9]+|evolve|"
            "evolve-scratch|evolve_space|campaigns)$' | head -5"),
        capture_output=True, text=True, timeout=120, check=False)
    checks.append({"check": "no other run is visible from the workspace",
                   "ok": not others.stdout.strip(),
                   "detail": (others.stdout.strip()[:200]
                              or "no run directory, campaign tree or evolve "
                                 "namespace is mounted or present")})
    mounts = subprocess.run(
        ["docker", "inspect", "-f",
         "{{range .Mounts}}{{.Source}}->{{.Destination}} {{end}}", container],
        capture_output=True, text=True, timeout=60, check=False
    ) if not (backend.is_modal(container) or backend.is_local(container)
              ) else subprocess.CompletedProcess(
        # A SANDBOX HAS NO BIND MOUNTS. Nothing from this host is visible
        # inside it -- the workspace was uploaded and the checkpoint is a
        # volume -- so there is no host path that could be foreign, which is
        # exactly what this check exists to find.
        args=[], returncode=0, stdout="", stderr="")
    foreign = foreign_mounts(mounts.stdout, workspace)
    checks.append({"check": "no other run's directory is mounted",
                   "ok": not foreign,
                   "detail": " ".join(foreign)[:200] or "only this run's"})

    for name in withheld:
        got = subprocess.run(
            backend.exec_argv(
                container, "",
                f"test -e /workspace/{name} && echo present || echo absent"),
            capture_output=True, text=True, timeout=120, check=False)
        checks.append({"check": f"withheld: {name} is not mounted",
                       "ok": "absent" in got.stdout,
                       "detail": got.stdout.strip()[:120]})

    report = BootReport(ok=all(c["ok"] for c in checks), checks=tuple(checks),
                        seconds=round(time.monotonic() - started, 2))
    journal.append("workspace_boot", **report.to_dict())
    return report
