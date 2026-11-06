"""The final verdict on this host, in fresh namespace sandboxes (no docker).

The local counterpart of `modal_verdict.py`, for runs whose environment was a
`local:` sandbox (`serving_evolve/local_sandbox.py`, Google Colab). It reuses
the Modal transport's in-sandbox worker (`modal_worker.py`) unchanged -- the
same two phases, the same shared measurement and scoring code -- and replaces
only how a "sandbox" is made:

* **candidate phase.** A fresh sandbox with NO egress at all (no broker). The
  frozen version is copied to `/app` and owned by `agent`; the evaluator and
  this task's contract are copied to a root-owned, read-only tree; the held-out
  questions go to a root-only directory. The worker runs as root inside the
  sandbox's network namespace and starts `serve.sh` as `agent` there, so the
  candidate reaches nothing but loopback and cannot read the evaluator's
  private inputs or outputs (`/evaluation` is root-only).
* **reference phase.** After every candidate process is gone, a second fresh
  sandbox runs the pinned SGLang (its own interpreter, `SERVERBENCH_LOCAL_SGLANG_BIN`)
  over the candidate's committed replies and the held-out answers. No
  candidate code runs there.

The agent's own sandbox is terminated before this runs, and every process the
agent user owns is killed and confirmed gone first: the verdict refuses to start
while one survives.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import secrets
import shutil
import signal
import subprocess
import sys
from pathlib import Path

from . import candidate_bundle, spine
from .modal_verdict import preflight

REPO = Path(__file__).resolve().parents[3]
VERIFIER = Path("/opt/serverbench-verifier")
PRIVATE = Path("/opt/serverbench-private")
EVIDENCE = Path("/evaluation")
SOURCE = Path("/app")
MODEL = Path("/models/target")
AGENTX_PYTHON = "/opt/agentx/venv/bin/python"

TOOLS_BIN = os.environ.get("SERVERBENCH_LOCAL_TOOLS_BIN", "/opt/serverbench/bin")
AGENT_PYTHON_BIN = os.environ.get("SERVERBENCH_LOCAL_AGENT_PYTHON_BIN",
                                  "/opt/serverbench/agent-python/bin")
SGLANG_BIN = os.environ.get("SERVERBENCH_LOCAL_SGLANG_BIN",
                            "/opt/serverbench/sglang-venv/bin")
CUDA_HOME = os.environ.get("SERVERBENCH_LOCAL_CUDA_HOME", "/usr/local/cuda")


def _sandbox():
    harness = str(REPO / "serverbench" / "harness")
    if harness not in sys.path:
        sys.path.insert(0, harness)
    from serving_evolve import local_sandbox
    return local_sandbox


def _harbor_env():
    scripts = str(REPO / "serverbench" / "scripts" / "agent")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    import harbor_env
    return harbor_env


def _copy_verifier(task: str) -> None:
    """Only trusted evaluator source and this task's contract, root-owned."""
    if VERIFIER.exists():
        shutil.rmtree(VERIFIER)
    source = REPO / "serverbench/harness/final_validate"
    shutil.copytree(source, VERIFIER / "harness" / "final_validate",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    context = VERIFIER / "task_context" / task
    context.mkdir(parents=True)
    for name in ("contract.yaml", "reference.yaml"):
        shutil.copy2(REPO / "serving_tasks" / task / name, context / name)
    pin = VERIFIER / "artifacts" / "agentx"
    pin.mkdir(parents=True)
    shutil.copy2(REPO / "serverbench/artifacts/agentx/aiperf-pin.json", pin)
    for path in [VERIFIER, *VERIFIER.rglob("*")]:
        os.lchown(path, 0, 0)
        if not path.is_symlink():
            os.chmod(path, (path.stat().st_mode & ~0o022) | 0o444
                     | (0o111 if path.is_dir() else 0))


def _fresh(path: Path, mode: int) -> None:
    if path.is_symlink():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)
    os.chmod(path, mode)


def _place_inputs(task: str, model_dir: Path) -> None:
    """The model, AgentX and the task's HF repository at the brief's paths."""
    harbor_env = _harbor_env()
    sys.path.insert(0, str(REPO / "serverbench" / "scripts" / "agent"))
    import local_env
    binding = local_env.can_bind()
    mounts = [m for m in harbor_env.mounts_for(
        workspace=Path("/nonexistent"), model=model_dir,
        env_input=Path("/nonexistent"), uv_cache=Path("/nonexistent"),
        task=REPO / "serving_tasks" / task)
        if m["target"] not in (harbor_env.WORKSPACE, harbor_env.ENV_INPUT_MOUNT,
                               harbor_env.UV_CACHE_MOUNT)]
    for mount in mounts:
        if (mount["target"].startswith("/models/")
                and mount["target"] != str(MODEL)):
            continue
        local_env.place(mount["source"], mount["target"],
                        read_only=bool(mount.get("read_only")), binding=binding)


def _environment(task: str, *, reference: bool) -> dict[str, str]:
    path = (f"/opt/agentx/bin:{CUDA_HOME}/bin:{AGENT_PYTHON_BIN}:{TOOLS_BIN}:"
            f"/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin")
    if reference:
        path = f"{SGLANG_BIN}:" + path
    env = {
        "PATH": path, "HOME": "/root", "LANG": "C.UTF-8",
        "APP_DIR": str(SOURCE), "PORT": "8000", "MODEL_PATH": str(MODEL),
        "CUDA_HOME": CUDA_HOME, "LD_LIBRARY_PATH": f"{CUDA_HOME}/lib64",
        "HF_HOME": "/opt/hf-cache", "HF_DATASETS_CACHE": "/tmp/hf-datasets-verdict",
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false", "PYTHONUNBUFFERED": "1",
        "AIPERF_DATASET_MMAP_CACHE_ENABLED": "false",
        "PIP_NO_INDEX": "1", "UV_OFFLINE": "1", "UV_NO_INDEX": "1",
        "UV_PYTHON_DOWNLOADS": "never", "UV_LINK_MODE": "copy",
        "UV_CACHE_DIR": "/tmp/uv-cache-verdict",
        "PYTHONPATH": f"{VERIFIER}/harness",
        "SERVERBENCH_TASK_CONTEXT": f"{VERIFIER}/task_context",
        "HF_DATASETS_OFFLINE": "1",
    }
    module = _harbor_env()._task_module(REPO / "serving_tasks" / task)
    if module is not None:
        env["MODEL_ID"] = str(getattr(module, "MODEL_ID", "") or "")
        env["MODEL_REVISION"] = str(getattr(module, "MODEL_REVISION", "") or "")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        env["CUDA_VISIBLE_DEVICES"] = visible
    return env


def _datasets_cache() -> None:
    cache = Path("/tmp/hf-datasets-verdict")
    _fresh(cache, 0o755)
    for directory in sorted(Path("/opt/hf-cache/datasets").glob("*")):
        (cache / directory.name).symlink_to(directory)


def _agent_is_gone() -> None:
    sandbox = _sandbox()
    left = sandbox.kill_agent_processes()
    if left:
        raise RuntimeError(
            f"INFRASTRUCTURE_FAILURE: {len(left)} process(es) of the agent user "
            f"survived SIGKILL ({left[:5]}); refusing to measure next to them")


def _run_phase(name: str, command: list[str], *, env: dict, log: Path) -> int:
    sandbox = _sandbox()
    body = sandbox.create(name, env={}, proxy_socket="")
    try:
        argv = sandbox.enter_argv(body, as_root=True) + command
        with log.open("w") as handle:
            process = subprocess.Popen(argv, env=env, cwd=str(VERIFIER),
                                       stdout=handle, stderr=subprocess.STDOUT,
                                       start_new_session=True)
            try:
                return process.wait(timeout=6 * 3600)
            except BaseException:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
                raise
    finally:
        sandbox.terminate(name)
        _agent_is_gone()


def judge(task: str, version: Path, out: Path, *, development: bool = False,
          candidate_tree_sha: str = "", target: str = "candidate",
          model_dir: Path | None = None) -> dict:
    task = spine.resolve_task_dir(task).name
    if os.geteuid() != 0:
        raise RuntimeError("INFRASTRUCTURE_FAILURE: the local verdict runs as root")
    if not Path(version).is_dir():
        raise ValueError(f"candidate directory does not exist: {version}")
    record = Path(version) / "VERSION.json"
    if record.is_file() and json.loads(record.read_text()).get("mutable") is True:
        raise ValueError("DELIVERY_FAILED: final validation requires an immutable published version")
    delivery = (candidate_bundle.audit(version) if target == "candidate"
                else {"files": [], "errors": []})
    if delivery["errors"]:
        raise ValueError("DELIVERY_FAILED: " + "; ".join(delivery["errors"]))
    if target == "candidate" and "serve.sh" not in delivery["files"]:
        raise ValueError("DELIVERY_FAILED: candidate has no delivered serve.sh")
    preflight(task, Path(version), target)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "delivery-report.json").write_text(json.dumps(delivery, indent=2))
    _agent_is_gone()
    if model_dir is not None:
        _place_inputs(task, Path(model_dir))
    _copy_verifier(task)
    _datasets_cache()
    fingerprint = candidate_bundle.fingerprint(version, delivery["files"])
    if candidate_tree_sha and fingerprint != candidate_tree_sha:
        raise ValueError("SOURCE_CHANGED: the candidate differs from the requested source digest")
    candidate_tree_sha = fingerprint
    uid, gid = _sandbox().agent_ids()

    # The held-out questions: drawn here, root-only; the answers stay in `out`
    # until the reference phase, which runs no candidate code.
    _fresh(PRIVATE, 0o700)
    if not development:
        from . import private_correctness
        size, _ = private_correctness.answer_settings(spine.load_contract(task))
        questions = private_correctness.select_questions(size, secrets.randbits(64))
        (out / "reference-answers.json").write_text(json.dumps(
            {q["id"]: q["answer"] for q in questions}))
        (PRIVATE / "questions.json").write_text(json.dumps(
            [{"id": q["id"], "question": q["question"]} for q in questions]))
        os.chmod(PRIVATE / "questions.json", 0o600)

    phase = "candidate"
    try:
        # ------------------------------------------------------ candidate
        _fresh(EVIDENCE, 0o700)
        _fresh(SOURCE, 0o755)
        if target == "candidate":
            for name in delivery["files"]:
                destination = SOURCE / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(Path(version) / name, destination)
            for path in [SOURCE, *SOURCE.rglob("*")]:
                os.lchown(path, uid, gid)
                if path.is_dir() and not path.is_symlink():
                    os.chmod(path, path.stat().st_mode | 0o700)
        command = [AGENTX_PYTHON, "-m", "final_validate.modal_worker", "candidate",
                   "--task", task, "--target", target,
                   "--candidate-tree-sha", candidate_tree_sha]
        if development:
            command.append("--development-workload")
        else:
            command += ["--questions", str(PRIVATE / "questions.json")]
        (out / "candidate-sandbox.json").write_text(json.dumps({
            "backend": "local", "task": task, "network": "none (loopback only)"}))
        code = _run_phase("verdict-candidate", command,
                          env=_environment(task, reference=False),
                          log=out / "candidate.phase.log")
        _collect(out)
        if code:
            tail = (out / "candidate.phase.log").read_text(errors="replace")[-4000:]
            raise RuntimeError(f"candidate phase exited {code}: {tail}")

        # ------------------------------------------------------ reference
        phase = "reference"
        _fresh(PRIVATE, 0o700)
        _fresh(SOURCE, 0o755)
        _fresh(EVIDENCE, 0o700)
        for name in ("candidate-phase.json", "reference-answers.json"):
            if (out / name).is_file():
                shutil.copy2(out / name, EVIDENCE / name)
        from . import agentx_reference
        artifacts = out / "agentx" / target
        with contextlib.suppress(FileNotFoundError):
            raw = agentx_reference._raw_export(artifacts)
            if raw is not None:
                destination = EVIDENCE / raw.relative_to(out)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(raw, destination)
        command = [AGENTX_PYTHON, "-m", "final_validate.modal_worker", "reference",
                   "--task", task, "--target", target,
                   "--candidate-tree-sha", candidate_tree_sha]
        code = _run_phase("verdict-reference", command,
                          env=_environment(task, reference=True),
                          log=out / "reference.phase.log")
        _collect(out)
        if code:
            tail = (out / "reference.phase.log").read_text(errors="replace")[-4000:]
            raise RuntimeError(f"reference phase exited {code}: {tail}")
        result = json.loads((out / f"scenario-{target}.json").read_text())
        result["backend"] = "local"
        (out / f"scenario-{target}.json").write_text(json.dumps(result, indent=1) + "\n")
        return result
    except BaseException as exc:
        (out / "local-failure.json").write_text(json.dumps({
            "phase": phase, "complete": False,
            "why": f"{type(exc).__name__}: {exc}"[:4000]}, indent=2))
        if isinstance(exc, Exception) and phase == "reference":
            raise RuntimeError(f"INFRASTRUCTURE_FAILURE: reference audit did not complete: {exc}") from exc
        if isinstance(exc, Exception) and (out / "candidate-ready.json").is_file():
            raise RuntimeError(
                f"MEASUREMENT_FAILED: server started but verification did not finish: {exc}") from exc
        raise
    finally:
        _fresh(PRIVATE, 0o700)


def _collect(out: Path) -> None:
    """Copy the phase's evidence out of /evaluation (files only)."""
    for path in sorted(EVIDENCE.rglob("*")):
        if path.is_symlink() or not (path.is_file() or path.is_dir()):
            continue
        destination = out / path.relative_to(EVIDENCE)
        if path.is_dir():
            destination.mkdir(parents=True, exist_ok=True)
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--version", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=None)
    parser.add_argument("--development-workload", action="store_true")
    parser.add_argument("--candidate-tree-sha", default="")
    parser.add_argument("--target", default="candidate", choices=("candidate", *spine.ARMS))
    args = parser.parse_args(argv)
    try:
        judge(args.task, args.version, args.out,
              development=args.development_workload,
              candidate_tree_sha=args.candidate_tree_sha, target=args.target,
              model_dir=args.model_dir)
    except (RuntimeError, ValueError, OSError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
