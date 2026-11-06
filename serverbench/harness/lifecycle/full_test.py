"""Agent-requested, source-bound full test on public traffic, run in the container.

The host freezes and hashes the delivery bundle (file bookkeeping only). The
cold start, the protocol/context probes and the AgentX replay (the agent's
`--duration`, 300-1,800 s; the verdict always runs 1,800 s) all run INSIDE the
workspace container, with the packaged evaluator and the
card the run already holds -- nothing the agent triggers executes on the host.
Output correctness is never checked here; that stage is private to the final
verdict. Never writes a final CSV/result/head.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import shutil
import subprocess
import time
from pathlib import Path

from . import agent, backend


def execute(controller, request: dict) -> None:
    """Never let a background setup/reporting failure disappear silently."""
    try:
        _execute(controller, request)
    except Exception as exc:
        controller.journal.append(
            "development_full_test_failed", request_file=request["file"], why=f"{type(exc).__name__}: {exc}"
        )
        # This separate path also works when the requested evidence directory
        # was unusable. The client polls it rather than waiting indefinitely.
        report = {
            "schema": "serving_evolution.full_test/1",
            "status": "failed",
            "complete": False,
            "why": f"{type(exc).__name__}: {exc}",
        }
        request_id = Path(request["file"]).stem
        destination = controller.built.path / ".versions" / (request_id + ".result")
        if (
            not destination.resolve().is_relative_to(controller.built.path.resolve())
            or destination.is_symlink()
        ):
            raise ValueError("full-test failure report escapes workspace") from exc
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report))
        if backend.is_modal(controller.container):
            backend.put_file(
                controller.container, json.dumps(report),
                f"/workspace/agentx/full-tests/{request_id}/full-test.json"
            )
            backend.put_file(
                controller.container, json.dumps(report), "/workspace/.versions/" + destination.name
            )


#: Where each development full test leaves its report and frozen source,
#: relative to the workspace.
REPORTS = Path("agentx") / "full-tests"


def passed_on(tree: Path, files: list[str], reports: Path) -> dict:
    """Whether a development full test passed on exactly this tree.

    A version published minutes before the deadline may be untested since a
    full test several versions earlier. Compared on delivered names and bytes
    against each passed test's
    frozen `source/`: a captured tree's file modes come back normalised by
    `git archive`, so the full test's mode-bound fingerprint cannot be reused.
    """
    want = {name: (Path(tree) / name).read_bytes() for name in files}
    newest = None
    for path in sorted(Path(reports).glob("*/full-test.json"),
                       key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            report = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if report.get("status") != "passed" or not isinstance(report.get("files"), dict):
            continue
        source = path.parent / "source"
        names = set(report["files"])
        changed = sorted(names ^ set(want))
        for name in sorted(names & set(want)):
            try:
                if (source / name).read_bytes() != want[name]:
                    changed.append(name)
            except OSError:
                changed.append(name)
        found = {"full_test": path.parent.name,
                 "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                               time.gmtime(path.stat().st_mtime))}
        if not changed:
            return {"tested": True, **found}
        if newest is None:
            newest = {**found, "changed": changed}
    return {"tested": False, "newest_passed": newest}


#: A port the agent's own server is not expected to use.
FULL_TEST_PORT = 18731
#: Memory already resident on the card above which a cold start cannot be a
#: fair test: the agent's own server is still up, the snapshot's server would
#: OOM, and the test would wait out its whole readiness timeout before
#: reporting "not ready".
BUSY_CARD_MIB = 3072
#: The verdict's own readiness limit for a candidate
#: (`deploy.CANDIDATE_READY_TIMEOUT_S`): a
#: development cold start allowed longer than the verdict allows would pass
#: here and fail there.
READY_TIMEOUT_S = 900


def _in_container_script(*, task: str, inside: str, tree_sha: str,
                         duration_s: int = 1800) -> str:
    """Cold-start the frozen bundle and measure it, inside the container."""
    return f"""set -u
SRC={inside}/source
EV={inside}/evaluation
mkdir -p "$EV"
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | sort -n | tail -1)
if [ -n "$used" ] && [ "$used" -gt {BUSY_CARD_MIB} ]; then
  echo "GPU_BUSY: the card already holds $used MiB before the cold start; stop your own server and profilers, then request the test again" | tee "$EV/gpu-busy.txt"
  exit 5
fi
set -m
( cd "$SRC" && exec env APP_DIR="$SRC" PORT={FULL_TEST_PORT} bash serve.sh ) \\
    > "$EV/candidate.serverlog.txt" 2>&1 &
SERVER=$!
stop() {{ kill -TERM -- -$SERVER 2>/dev/null; sleep 5; kill -KILL -- -$SERVER 2>/dev/null; }}
trap stop EXIT
for _ in $(seq 1 {READY_TIMEOUT_S}); do
  curl -sf http://127.0.0.1:{FULL_TEST_PORT}/health >/dev/null && break
  kill -0 $SERVER 2>/dev/null || {{ echo "serve.sh exited before readiness; see candidate.serverlog.txt" | tee "$EV/startup-failed.txt"; exit 3; }}
  sleep 1
done
curl -sf http://127.0.0.1:{FULL_TEST_PORT}/health >/dev/null || {{ echo "not ready after {READY_TIMEOUT_S} s; see candidate.serverlog.txt" | tee "$EV/startup-failed.txt"; exit 4; }}
cd /workspace/tests
python3 -m final_verdict.scenario {task} --target candidate \\
    --endpoint 127.0.0.1:{FULL_TEST_PORT} --model-dir "$MODEL_PATH" \\
    --run-dir "$EV/protocol" --check-protocol-context > "$EV/protocol.log" 2>&1
echo $? > "$EV/protocol.rc"
python3 -m final_verdict.scenario {task} --target candidate \\
    --endpoint 127.0.0.1:{FULL_TEST_PORT} --model-dir "$MODEL_PATH" \\
    --run-dir "$EV" --measure-only --duration {duration_s} \\
    --candidate-snapshot "$SRC" --candidate-tree-sha {tree_sha} \\
    > "$EV/measure.log" 2>&1
echo $? > "$EV/measure.rc"
# The verdict probes the same server after its window; a server that died or
# hung during the replay fails here rather than looking healthy.
python3 -m final_verdict.scenario {task} --target candidate \\
    --endpoint 127.0.0.1:{FULL_TEST_PORT} --model-dir "$MODEL_PATH" \\
    --run-dir "$EV/protocol-after" --check-protocol-context > "$EV/protocol-after.log" 2>&1
echo $? > "$EV/protocol-after.rc"
exit 0
"""


#: Exceptions a server logs when a client goes away -- the replay cancels its
#: in-flight requests when the window closes. They say nothing about health.
CLIENT_DISCONNECT = ("BrokenPipeError", "ConnectionResetError",
                     "ConnectionAbortedError", "CancelledError",
                     "ClientDisconnect")


def _eos_gate(stage: Path) -> dict:
    """The EOS gate's result from one `--check-protocol-context` stage, or {} if absent."""
    found = stage / "agentx" / "candidate" / "eos-check.json"
    try:
        return json.loads(found.read_text())
    except (OSError, ValueError):
        return {}


def server_log_faults(log_text: str) -> list[str]:
    """Unhandled exceptions and CUDA out-of-memory errors in a server log.

    One entry per distinct fault: the exception line that ends each Python
    traceback, and any CUDA out-of-memory line. A client disconnecting is not
    a fault.
    """
    faults: list[str] = []
    lines = log_text.splitlines()
    for index, line in enumerate(lines):
        if line.startswith("Traceback (most recent call last)"):
            ending = next((later.strip() for later in lines[index + 1:]
                           if later and not later.startswith((" ", "\t"))
                           and not later.startswith("Traceback")), "")
            if ending and not any(name in ending for name in CLIENT_DISCONNECT):
                faults.append(ending[:200])
        elif "CUDA out of memory" in line or "OutOfMemoryError" in line:
            faults.append(line.strip()[:200])
    return list(dict.fromkeys(faults))


def health_problems(*, attainment: dict | None, window_seconds: int,
                    after_probe_ok: bool | None,
                    log_text: str) -> list[str]:
    """Why a completed window does not show a healthy server, if it does not.

    A window whose replay gate held can still have served almost nothing, or
    been served by an engine that died part-way, and an agent would read such
    a window as a pass. Fewer than one completed profiling request
    per minute, a server that fails the protocol probes after the window, and
    an unhandled exception or CUDA out-of-memory error in the server log are
    each a failure, with the reason stated.
    """
    problems: list[str] = []
    attainment = attainment or {}
    count = attainment.get("request_count")
    if count is not None:
        completed = (int(count) - int(attainment.get("error_request_count") or 0)
                     - int(attainment.get("cancelled_request_count") or 0))
        floor = max(1, int(window_seconds) // 60)
        if completed < floor:
            problems.append(
                f"only {completed} profiling request(s) completed in the "
                f"{window_seconds} s window (fewer than one per minute); see "
                "evaluation/candidate.serverlog.txt and the per-request export")
    if after_probe_ok is False:
        problems.append("the protocol probes failed after the window: the server "
                        "stopped serving correctly during or after the replay; "
                        "see evaluation/protocol-after.log")
    faults = server_log_faults(log_text)
    if faults:
        problems.append(f"the server log records {len(faults)} fault(s), first: "
                        f"{faults[0]}; see evaluation/candidate.serverlog.txt")
    return problems


def _execute(controller, request: dict) -> None:
    from final_validate.candidate_bundle import audit

    workspace = controller.built.path
    request_id = Path(request["file"]).stem
    root = workspace / "agentx" / "full-tests" / request_id
    if not root.resolve().is_relative_to(workspace.resolve()):
        raise ValueError("full-test destination escapes workspace")
    root.mkdir(parents=True, exist_ok=False)
    destination = root / "full-test.json"
    report = {
        "schema": "serving_evolution.full_test/1",
        "task_id": controller.plan.task_dir.name,
        "status": "running",
        "evaluation_scope": "development_full_test",
        "final_verdict": False,
        "complete": False,
        "source_snapshot": str(root / "source"),
    }

    def save():
        temporary = destination.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(destination)
        if backend.is_modal(controller.container):
            backend.put_file(
                controller.container, json.dumps(report),
                f"/workspace/agentx/full-tests/{request_id}/full-test.json"
            )
            backend.put_file(
                controller.container, json.dumps(report), "/workspace/.versions/" + request_id + ".result"
            )

    save()
    child = None
    try:
        if backend.is_modal(controller.container):
            # A test must never proceed against a stale host mirror after a
            # failed transfer.
            backend.pull_workspace(controller.container, workspace)
        else:
            controller._bring_the_tree_home("agent requested a development full test")
        delivery = audit(workspace)
        (root / "delivery-report.json").write_text(json.dumps(delivery, indent=2) + "\n")
        if delivery["errors"]:
            raise RuntimeError("DELIVERY_FAILED: " + "; ".join(delivery["errors"]))
        source = root / "source"
        source.mkdir()
        files = {}
        for name in delivery["files"]:
            path = source / name
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(workspace / name, path)
            files[name] = hashlib.sha256(
                str(path.stat().st_mode & 0o777).encode() + b"\0" + path.read_bytes()
            ).hexdigest()
        fingerprint = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
        report.update(candidate_tree_sha=fingerprint, files=files)
        save()
        duration = int(request.get("duration_s", 1800))
        if not 300 <= duration <= 1800:
            raise RuntimeError(f"BAD_DURATION: {duration} s; use 300-1800")
        report["window_seconds"] = duration
        remaining = agent.remaining_seconds(controller.identity)
        if remaining < duration + 300:
            raise RuntimeError(
                f"INSUFFICIENT_TIME: {remaining:.0f} s remain; a {duration} s "
                "replay needs about 5 more minutes for cold start and probes")
        # EVERYTHING BELOW RUNS INSIDE THE WORKSPACE CONTAINER. The host only
        # froze and hashed the bundle above; the cold start, the probes and the
        # 1,800 s replay use the container's own interpreter, the packaged
        # evaluator and the card the run already holds.
        inside = f"/workspace/agentx/full-tests/{request_id}"
        if backend.is_modal(controller.container):
            backend.push_directory(controller.container, root, inside)
        if backend.is_local(controller.container):
            # The driver is root here; the frozen source and the evidence
            # directory have to be the agent's, as they are on docker.
            from serving_evolve import local_sandbox
            local_sandbox.give_to_agent(root)
        script = _in_container_script(
            task=controller.plan.task_dir.name, inside=inside,
            tree_sha=fingerprint, duration_s=duration)
        command = backend.exec_argv(controller.container, "", script)
        report["command"] = ["<workspace container>", script]
        with (root / "verifier.log").open("w") as output:
            child = subprocess.Popen(command, stdout=output,
                                     stderr=subprocess.STDOUT)
            deadline = time.monotonic() + min(remaining, 14400)
            while child.poll() is None:
                if controller._ending.is_set() or time.monotonic() >= deadline:
                    raise TimeoutError(
                        "full test interrupted at evolve deadline; partial evidence is not a pass"
                    )
                with contextlib.suppress(subprocess.TimeoutExpired):
                    child.wait(timeout=1)
        if backend.is_modal(controller.container):
            backend.pull_workspace(controller.container, workspace)
        for name, expected in files.items():
            captured = source / name
            if (
                captured.is_symlink()
                or not captured.is_file()
                or hashlib.sha256(
                    str(captured.stat().st_mode & 0o777).encode() + b"\0" + captured.read_bytes()
                ).hexdigest()
                != expected
            ):
                raise RuntimeError("SOURCE_CHANGED: full-test snapshot changed during evaluation")
        evaluation = root / "evaluation"
        evidence_path = evaluation / "development-evidence.json"
        if child.returncode != 0 or not evidence_path.is_file():
            stated = [p.read_text().strip() for p in
                      (evaluation / "gpu-busy.txt", evaluation / "startup-failed.txt")
                      if p.is_file()]
            raise RuntimeError(
                (stated[0] + "; " if stated else "")
                + f"full test did not finish (exit {child.returncode}); see "
                "verifier.log and evaluation/*.log")
        evidence = json.loads(evidence_path.read_text())
        # `--check-protocol-context` exits 0 when both pass, 1 when a replay-gate
        # probe fails and 4 when only the EOS gate fails; the EOS gate's own result
        # is read from its file so the two gates are never reported as one.
        protocol_rc = (evaluation / "protocol.rc").read_text().strip()
        protocol_ok = protocol_rc in ("0", "4")
        after_rc = evaluation / "protocol-after.rc"
        after_ok = (after_rc.read_text().strip() in ("0", "4")) if after_rc.is_file() else None
        eos = _eos_gate(evaluation / "protocol")
        eos_after = _eos_gate(evaluation / "protocol-after")
        serverlog = evaluation / "candidate.serverlog.txt"
        health = health_problems(
            attainment=evidence.get("request_attainment"),
            window_seconds=int(evidence.get("window_seconds") or duration),
            after_probe_ok=after_ok,
            log_text=(serverlog.read_text("utf-8", errors="replace")
                      if serverlog.is_file() else ""))
        complete = bool(evidence.get("full_window_complete"))
        replay_ok = evidence.get("replay_fidelity_holds") is True
        measured = evidence.get("objective_value") is not None
        # A PASS is the verdict's checks that can run here -- cold start,
        # protocol/context probes before and after the window, the replay
        # gate, the objective measured -- and a server that stayed healthy:
        # requests kept completing and its log records no fault. A low
        # dollars_per_dollar from a healthy server is a result, not a failure.
        why = "; ".join([reason for bad, reason in (
            (not protocol_ok, "the replay gate's protocol/context probes failed: "
             "see evaluation/protocol.log"),
            # An explicit EOS-gate failure fails the test; a stage that wrote no
            # EOS file reports `None` (not run), which alone does not fail it.
            (eos.get("holds") is False or protocol_rc == "4",
             (eos.get("why") or "the EOS gate failed") + ": see evaluation/protocol.log"),
            (eos.get("holds") is not False and eos_after.get("holds") is False,
             (eos_after.get("why") or "the EOS gate failed") + " after the window: "
             "see evaluation/protocol-after.log"),
            (not replay_ok, "the replay gate failed: see evaluation/measure.log"),
            (not complete, "the replay did not complete: "
             + str(evidence.get("incomplete_why") or "see evaluation/measure.log")),
            # A failed replay gate withholds the score; that reason is above.
            (complete and replay_ok and not measured,
             "dollars_per_dollar could not be computed: "
             + str(evidence.get("request_scoring_why") or "see evaluation/measure.log")),
        ) if bad] + health)
        report.update(
            complete=complete,
            protocol_context_holds=protocol_ok,
            protocol_after_window_holds=after_ok,
            eos_gate_holds=eos.get("holds"),
            eos_gate_after_window_holds=eos_after.get("holds"),
            server_log_faults=server_log_faults(
                serverlog.read_text("utf-8", errors="replace")
                if serverlog.is_file() else ""),
            replay_fidelity_holds=evidence.get("replay_fidelity_holds"),
            objective=evidence.get("objective"),
            objective_value=evidence.get("objective_value"),
            dollars_per_dollar=evidence.get("objective_value"),
            gross_dollars_per_dollar=evidence.get("gross_dollars_per_dollar"),
            request_multiplier=evidence.get("request_multiplier"),
            request_attainment=evidence.get("request_attainment"),
            billable_input_tokens=evidence.get("billable_input_tokens"),
            billable_output_tokens=evidence.get("billable_output_tokens"),
            billable_value_usd=evidence.get("billable_value_usd"),
            billing_gpu_cost_usd=evidence.get("billing_gpu_cost_usd"),
            window_seconds=evidence.get("window_seconds") or duration,
            status="passed" if not why else "failed",
            why=why,
        )
    except Exception as exc:  # background work must leave a terminal diagnostic
        report.update(status="failed", why=f"{type(exc).__name__}: {exc}")
    finally:
        if child is not None and child.poll() is None:
            child.terminate()  # the in-container script stops its own server
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        if backend.is_modal(controller.container):
            # Public evidence only; make detailed logs available at the exact
            # paths advertised to the agent, not only a terminal status stub.
            try:
                backend.push_directory(controller.container, root,
                                       f"/workspace/agentx/full-tests/{request_id}")
            except Exception as exc:
                report["evidence_transfer_error"] = str(exc)
        save()
        controller.journal.append(
            "development_full_test_finished",
            request_file=request["file"],
            status=report["status"],
            complete=report["complete"],
            report=str(destination),
        )
