#!/usr/bin/env python3
"""Stand a run's environment up as a namespace sandbox on this host.

The local counterpart of `harbor_env.py` (docker) and `modal_env.py` (Modal),
for a host with root and a GPU but no container runtime -- a Google Colab
runtime. It writes the same `container.json` / `harbor_environment.json` and
names the container `local:<name>`, so `lifecycle/backend.py` dispatches on it
and nothing downstream learns a second vocabulary.

## What is the same as docker, deliberately

* the agent runs as the unprivileged `agent` user, in `/workspace`, with the
  model at `/models/target`, the wheelhouse at `/opt/env-input`, the pinned
  AgentX at `/opt/agentx` and the task's Hugging Face repository at
  `/opt/hf-cache` -- the paths the brief names;
* the environment variables the image and the compose overlay set
  (offline flags, `MODEL_PATH`, `PATH`, ...), from the same functions;
* the egress broker: the agent reaches its model provider through a CONNECT
  proxy that refuses every other host, and every attempt is logged;
* the credential routing: one provider, one credential, the rest stripped.

## What differs, and why

* **The network boundary is a network namespace, not iptables.** The host has
  no `CAP_NET_ADMIN`. The sandbox's namespace has only `lo`; the broker is
  reached through a unix socket relayed to 127.0.0.1:3128 inside it. See
  `serving_evolve/local_sandbox.py`.
* **Paths are bind mounts when the host allows them, symlinks when not.** With
  symlinks, the directories on the way to the workspace are made traversable
  (0711) and every private directory (verdict evidence, the answer pool,
  other runs, published lineages, secrets) is made root-only (0700).
* **There is one tenant.** The agent's home and its leftovers in /tmp are
  wiped before each run, because the previous run on this machine may have
  been a different agent on the same task.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "serverbench" / "harness"))

import harbor_env  # noqa: E402  (pure helpers only; nothing here drives Harbor)
from serving_evolve import local_sandbox as sandbox  # noqa: E402

WORKSPACE = harbor_env.WORKSPACE
MODEL_MOUNT = harbor_env.MODEL_MOUNT
UV_CACHE_MOUNT = harbor_env.UV_CACHE_MOUNT

#: Where the Colab setup puts the interpreter that stands in for the image's
#: `python3` (torch, transformers and the tokenizer stack at the image's pins),
#: the agent CLIs and `uv`. Overridable for other hosts.
TOOLS_BIN = os.environ.get("SERVERBENCH_LOCAL_TOOLS_BIN", "/opt/serverbench/bin")
AGENT_PYTHON_BIN = os.environ.get("SERVERBENCH_LOCAL_AGENT_PYTHON_BIN",
                                  "/opt/serverbench/agent-python/bin")
CUDA_HOME = os.environ.get("SERVERBENCH_LOCAL_CUDA_HOME", "/usr/local/cuda")

#: What the agent image's Dockerfiles set with ENV, which a sandbox has to set
#: itself (serving-agent + evolve-agent).
IMAGE_ENV = {
    "DISABLE_TELEMETRY": "1", "DISABLE_AUTOUPDATER": "1",
    "DISABLE_ERROR_REPORTING": "1",
    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
    "PYTHONUNBUFFERED": "1", "PIP_NO_INDEX": "1", "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false",
    "PYTHONNOUSERSITE": "1", "PYTHONPATH": "", "UV_OFFLINE": "1",
    "UV_NO_INDEX": "1", "UV_LINK_MODE": "copy",
    "UV_PYTHON_DOWNLOADS": "never",
}

#: Directories under the repository an agent must never read, made root-only.
PRIVATE_DIRS = ("server_output", "evaluation_progress",
                "serverbench/artifacts/final_validation",
                "serverbench/artifacts/eval_samples",
                "serverbench/artifacts/secrets",
                "serverbench/artifacts/operations",
                "serving_tasks/_concurrency_search_archive")


def can_bind() -> bool:
    probe_src = Path("/run/serverbench-local-bindprobe-src")
    probe_dst = Path("/run/serverbench-local-bindprobe-dst")
    probe_src.mkdir(parents=True, exist_ok=True)
    probe_dst.mkdir(parents=True, exist_ok=True)
    ok = subprocess.run(["mount", "--bind", str(probe_src), str(probe_dst)],
                        capture_output=True, check=False).returncode == 0
    if ok:
        subprocess.run(["umount", str(probe_dst)], check=False)
    return ok


def place(source: str | Path, target: str, *, read_only: bool, binding: bool) -> str:
    """Make `target` show `source`: a bind mount, or else a symlink.

    Only sources inside the repository are bound (it is root-only when binds
    are available). Everything outside it -- the model and the Hugging Face
    cache under /opt/serverbench -- is public input and is SYMLINKED: a Hugging
    Face snapshot is a farm of relative links (`../../blobs/...`, and in newer
    hub layouts a second hop into a hub-wide `blobs/`), which resolve only at
    their real location.
    """
    binding = binding and Path(source).resolve().is_relative_to(REPO.resolve())
    target_path = Path(target)
    if Path(source).resolve() == target_path.resolve() and not target_path.is_symlink():
        # Already at the path the brief names (the uv-managed interpreter an
        # AgentX venv links to, installed under /opt): nothing to place.
        make_traversable(target_path)
        return "in place"
    sandbox.unbind(target_path)
    if target_path.is_symlink():
        target_path.unlink()
    if binding:
        sandbox.bind(source, target_path, read_only=read_only)
        return "bind"
    if target_path.exists():
        if target_path.is_dir() and not any(target_path.iterdir()):
            target_path.rmdir()
        else:
            raise RuntimeError(f"{target} exists and is not ours to replace")
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.symlink_to(Path(source))
    return "symlink"


def make_traversable(path: Path, stop: Path = Path("/")) -> None:
    """o+x on every directory from `path` up to `/`, never o+r."""
    for directory in [path, *path.parents]:
        if directory == stop or str(directory) == "/":
            break
        with contextlib.suppress(OSError):
            mode = directory.stat().st_mode & 0o7777
            if not mode & 0o001:
                os.chmod(directory, mode | 0o001)


def harden(workspace: Path) -> list[str]:
    """Root-only private directories; other runs' trees closed off."""
    done = []
    for relative in PRIVATE_DIRS:
        path = REPO / relative
        path.mkdir(parents=True, exist_ok=True)
        os.chmod(path, 0o700)
        done.append(relative)
    run_root = workspace.parent
    namespace = run_root.parent
    for sibling in namespace.iterdir() if namespace.is_dir() else ():
        if sibling.is_dir() and sibling != run_root:
            os.chmod(sibling, 0o700)
    os.chmod(run_root, 0o711)
    return done


def wipe_agent_leftovers(uid: int, gid: int) -> dict:
    """A clean home and no files of a previous agent in /tmp or /dev/shm."""
    left = sandbox.kill_agent_processes(uid)
    home = Path(sandbox.AGENT_HOME)
    if home.exists():
        shutil.rmtree(home, ignore_errors=True)
    home.mkdir(parents=True, exist_ok=True)
    for name in (".bashrc", ".profile"):
        skel = Path("/etc/skel") / name
        if skel.is_file():
            shutil.copy2(skel, home / name)
    for path in [home, *home.rglob("*")]:
        os.chown(path, uid, gid)
    os.chmod(home, 0o700)
    removed = 0
    for root in (Path("/tmp"), Path("/dev/shm"), Path("/var/tmp")):
        for entry in list(root.iterdir()) if root.is_dir() else ():
            with contextlib.suppress(OSError):
                if entry.lstat().st_uid == uid:
                    if entry.is_dir() and not entry.is_symlink():
                        shutil.rmtree(entry, ignore_errors=True)
                    else:
                        entry.unlink()
                    removed += 1
    return {"processes_left": left, "tmp_entries_removed": removed}


def chown_tree(path: Path, uid: int, gid: int) -> None:
    for item in [path, *path.rglob("*")]:
        with contextlib.suppress(OSError):
            os.lchown(item, uid, gid)


def run_in(container: str, command: str, *, timeout: float = 240) -> subprocess.CompletedProcess:
    return subprocess.run(sandbox.exec_argv(container, "", command),
                          capture_output=True, text=True, timeout=timeout,
                          check=False)


def egress_probes(container: str, provider: str) -> dict:
    """The network boundary, observed from inside, as the agent sees it."""
    probes = {
        "direct_pypi_blocked": (
            "curl -sS -m 8 --noproxy '*' -o /dev/null -w '%{http_code}' "
            "https://pypi.org || echo BLOCKED", "BLOCKED"),
        "direct_huggingface_blocked": (
            "curl -sS -m 8 --noproxy '*' -o /dev/null -w '%{http_code}' "
            "https://huggingface.co || echo BLOCKED", "BLOCKED"),
        "proxied_pypi_refused": (
            "curl -sS -m 15 -o /dev/null -w '%{http_code}' https://pypi.org "
            "|| echo REFUSED", "REFUSED"),
        "proxied_github_refused": (
            "curl -sS -m 15 -o /dev/null -w '%{http_code}' https://github.com "
            "|| echo REFUSED", "REFUSED"),
        "provider_reachable": (
            f"curl -sS -m 20 -o /dev/null -w 'HTTP%{{http_code}}' "
            f"https://{provider}/ || true", "HTTP"),
        "private_root_unreadable": (
            "ls /root >/dev/null 2>&1 && echo READABLE || echo UNREADABLE",
            "UNREADABLE"),
    }
    out = {}
    for name, (command, want) in probes.items():
        got = run_in(container, command, timeout=60)
        text = (got.stdout + got.stderr).strip()
        ok = want in text and not (want == "HTTP" and "HTTP000" in text)
        out[name] = {"ok": ok, "detail": text[-200:]}
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workspace", type=Path, required=True)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--env-input", type=Path, required=True)
    ap.add_argument("--gpu", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--task", type=Path, default=None)
    ap.add_argument("--name", required=True)
    ap.add_argument("--agent-cli", choices=("codex", "claude"), default="claude")
    ap.add_argument("--agent-model", default="")
    ap.add_argument("--agent-effort", default="")
    ap.add_argument("--allow", action="append", default=[])
    ap.add_argument("--credentials", default=None)
    ap.add_argument("--agent-env", default="")
    ap.add_argument("--skip-codex-preflight", action="store_true")
    args = ap.parse_args(argv)

    if os.geteuid() != 0:
        print("REFUSED: the local backend is set up by root", file=sys.stderr)
        return 2
    if args.credentials is None:
        args.credentials = harbor_env.CREDENTIAL_DEFAULTS[args.agent_cli]
    if args.agent_cli == "codex" and not args.agent_effort:
        print("Codex requires a pinned --agent-effort", file=sys.stderr)
        return 2
    args.out.mkdir(parents=True, exist_ok=True)
    began = time.time()
    uid, gid = sandbox.agent_ids()
    container = sandbox.PREFIX + args.name
    devices = [int(g) for g in str(args.gpu).split(",") if g.strip()]

    # ------------------------------------------------------------ environment
    env = {
        "MODEL_PATH": MODEL_MOUNT,
        "MODEL_ID": harbor_env._declared_model_id(args.task),
        "PORT": "8000",
        "UV_CACHE_DIR": UV_CACHE_MOUNT,
        "APP_DIR": WORKSPACE,
        "CUDA_VISIBLE_DEVICES": ",".join(str(d) for d in devices),
        **IMAGE_ENV,
    }
    path = (f"{CUDA_HOME}/bin:{AGENT_PYTHON_BIN}:{TOOLS_BIN}:/usr/local/sbin:"
            f"/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin")
    env["CUDA_HOME"] = CUDA_HOME
    env["LD_LIBRARY_PATH"] = f"{CUDA_HOME}/lib64"
    runtime = harbor_env.agentx_runtime()
    if runtime is not None:
        env["AGENTX_ROOT"] = harbor_env.AGENTX_MOUNT
        if harbor_env.agentx_source() is not None:
            env["AGENTX_SOURCE"] = harbor_env.AGENTX_SOURCE_MOUNT
        if harbor_env.hf_cache_for(args.model) is not None:
            env["HF_HOME"] = harbor_env.HF_CACHE_MOUNT
        env["AIPERF_DATASET_MMAP_CACHE_ENABLED"] = "false"
        env["HF_DATASETS_CACHE"] = harbor_env.HF_DATASETS_CACHE
        path = f"{harbor_env.AGENTX_MOUNT}/bin:" + path
    env["PATH"] = path

    sources = [REPO / "serverbench/artifacts/secrets/agent.env"]
    if args.agent_env:
        named = Path(args.agent_env).expanduser()
        if not named.is_file():
            raise SystemExit(f"--agent-env names {named}, which does not exist")
        sources.append(named)
    for token_file in sources:
        if not token_file.is_file():
            continue
        named_keys = []
        for line in token_file.read_text("utf-8").splitlines():
            key, _, value = line.partition("=")
            if key.strip() and not key.strip().startswith("#"):
                env[key.strip()] = value.strip()
                named_keys.append(key.strip())
        print(f"token supplied via {token_file.name} -> {sorted(named_keys)} "
              f"(values not read into evidence)")
    pinned = harbor_env.pin_model_routes(env, args.agent_model, args.agent_cli)
    if pinned:
        print(f"every Claude CLI model route pinned to {pinned}")
    hosts = harbor_env._broker_allowlist(args.agent_cli, args.allow)
    hosts = harbor_env.narrow_to_provider(hosts, env, args.agent_cli, args.allow)
    route = harbor_env.credential_route(env, args.agent_cli)
    print(f"egress broker relays only to {sorted(hosts)}")

    # -------------------------------------------------------- clean slate
    for leftover in sandbox.STATE_ROOT.glob("*/sandbox.json") if sandbox.STATE_ROOT.is_dir() else ():
        sandbox.terminate(leftover.parent.name)
    wiped = wipe_agent_leftovers(uid, gid)
    print(f"agent home wiped; {wiped['tmp_entries_removed']} leftover tmp "
          f"entr(ies) of a previous agent removed")

    # ------------------------------------------------------------- the paths
    binding = can_bind()
    workspace = args.workspace.resolve()
    chown_tree(workspace, uid, gid)
    uv_cache = args.out / "uv-cache"
    uv_cache.mkdir(parents=True, exist_ok=True)
    os.chown(uv_cache, uid, gid)
    codex_sessions = None
    if args.agent_cli == "codex":
        codex_sessions = args.out / "codex-sessions"
        codex_sessions.mkdir(parents=True, exist_ok=True)
        os.chown(codex_sessions, uid, gid)
    mounts = harbor_env.mounts_for(
        workspace=workspace, model=args.model, env_input=args.env_input,
        uv_cache=uv_cache, codex_sessions=codex_sessions, task=args.task)
    placed = []
    for mount in mounts:
        if (mount["target"].startswith("/models/")
                and mount["target"] != MODEL_MOUNT):
            # The snapshot's blob directories, which docker has to mount
            # beside it; a symlinked snapshot resolves its own links.
            continue
        how = place(mount["source"], mount["target"],
                    read_only=bool(mount.get("read_only")), binding=binding)
        placed.append({**mount, "how": how})
        if how == "symlink":
            make_traversable(Path(mount["source"]))
    hardened = harden(workspace)
    if binding:
        # Everything the agent needs is mounted, and a bind does not depend on
        # its source's ancestors being traversable: close the repository.
        os.chmod(REPO, 0o700)
        hardened.append(". (the repository root)")
    else:
        os.chmod(REPO, 0o711)
        make_traversable(workspace)
    print(f"paths       {len(placed)} placed by "
          f"{'bind mount' if binding else 'symlink (no mount permission here)'}")

    # --------------------------------------------------------------- broker
    socket_path = sandbox.STATE_ROOT / f"{args.name}.egress.sock"
    sandbox.STATE_ROOT.mkdir(parents=True, exist_ok=True)
    os.chmod(sandbox.STATE_ROOT, 0o700)
    broker_log = args.out / "egress-broker.jsonl"
    broker_argv = [sys.executable, str(HERE / "egress_broker.py"),
                   "--unix", str(socket_path), "--log", str(broker_log)]
    for host in sorted(hosts):
        broker_argv += ["--allow-host", host]
    broker = subprocess.Popen(broker_argv, stdout=subprocess.PIPE,
                              stderr=subprocess.DEVNULL, text=True,
                              start_new_session=True)
    if not (broker.stdout.readline() or "").startswith("unix:"):
        broker.kill()
        print("the egress broker did not come up", file=sys.stderr)
        return 1

    try:
        body = sandbox.create(args.name, env=env, proxy_socket=str(socket_path),
                              path=path)

        # --------------------------------------------------- credentials
        credentials = Path(args.credentials).expanduser()
        staged = None
        if route in ("oauth", "codex") and credentials.is_file():
            staged = harbor_env.lifecycle_mod("credentials").stage(
                credentials, args.out)
        claude_home = Path(sandbox.AGENT_HOME) / ".claude"
        claude_home.mkdir(parents=True, exist_ok=True)
        os.chown(claude_home, uid, gid)
        os.chmod(claude_home, 0o700)
        if staged is not None and args.agent_cli == "claude":
            sandbox.put_file(container, staged.read_text("utf-8"),
                             f"{claude_home}/{credentials.name}")
            print(f"credentials copied from {credentials.name} into the "
                  f"agent's home (refreshed in place by the driver)")
        elif route == "api_key":
            print("credentials: the run's API key only")
        codex_skills = 0
        if args.agent_cli == "codex":
            codex_home = Path(sandbox.AGENT_HOME) / ".codex"
            (codex_home / "skills").mkdir(parents=True, exist_ok=True)
            sessions = codex_home / "sessions"
            if sessions.exists() or sessions.is_symlink():
                if sessions.is_dir() and not sessions.is_symlink():
                    shutil.rmtree(sessions)
                else:
                    sessions.unlink()
            sessions.symlink_to("/home/agent/codex-sessions")
            skills = Path(WORKSPACE) / ".claude" / "skills"
            for skill in sorted(skills.iterdir()) if skills.is_dir() else ():
                if skill.is_dir():
                    link = codex_home / "skills" / skill.name
                    if link.is_symlink():
                        link.unlink()
                    link.symlink_to(skill)
                    codex_skills += 1
            if staged is not None:
                shutil.copy2(staged, codex_home / "auth.json")
                os.chmod(codex_home / "auth.json", 0o600)
            chown_tree(codex_home, uid, gid)
            os.chmod(codex_home, 0o700)
            print(f"codex       {codex_skills} workspace skill(s) linked")
        if not staged and route != "api_key" and not (
                env.get("CLAUDE_CODE_OAUTH_TOKEN") or env.get("OPENAI_API_KEY")
                or env.get("CODEX_API_KEY")):
            print(f"WARNING: no credentials at {args.credentials} and none in "
                  f"the env file; the agent will start and report that it is "
                  f"not logged in")

        # ------------------------------------------ datasets cache (writable)
        datasets_linked = 0
        cache = Path(harbor_env.HF_DATASETS_CACHE)
        if env.get("HF_HOME"):
            cache.mkdir(parents=True, exist_ok=True)
            source_root = Path(harbor_env.HF_CACHE_MOUNT) / "datasets"
            for directory in sorted(source_root.glob("*")) if source_root.is_dir() else ():
                link = cache / directory.name
                if link.is_symlink():
                    link.unlink()
                link.symlink_to(directory)
                datasets_linked += 1
            chown_tree(cache, uid, gid)

        # ------------------------------------------------------------ probes
        provider = sorted(hosts)[0] if hosts else "api.anthropic.com"
        if args.agent_cli == "claude" and "api.anthropic.com" in hosts:
            provider = "api.anthropic.com"
        if args.agent_cli == "codex":
            provider = "api.openai.com"
        egress = egress_probes(container, provider)
        egress_ok = all(r["ok"] for r in egress.values())
        for name, result in egress.items():
            print(f"egress      {'ok  ' if result['ok'] else 'FAIL'} {name}: "
                  f"{result['detail'][-60:]}")
        drive = run_in(container, "ls /content/drive/MyDrive >/dev/null 2>&1 "
                                  "&& echo DRIVE_VISIBLE || echo no-drive")
        if "DRIVE_VISIBLE" in drive.stdout:
            raise RuntimeError(
                "Google Drive is mounted and readable from the sandbox. Unmount "
                "it (drive.flush_and_unmount()) before starting an agent: it "
                "would see everything in your Drive, including other runs")
        preflight = run_in(
            container, f"whoami; python3 -c 'import torch; "
                       f"print(torch.cuda.device_count())'; "
                       f"{args.agent_cli} --version", timeout=300)
        agent_preflight = None
        if args.agent_cli == "codex" and not args.skip_codex_preflight:
            agent_preflight = codex_preflight(container, args.agent_model,
                                              args.agent_effort)
            print(f"agent       {agent_preflight['detail'][:200]}")
            if not agent_preflight["ok"]:
                raise RuntimeError("Codex preflight failed before the run budget "
                                   f"was spent: {agent_preflight['detail']}")
        if not egress_ok:
            raise RuntimeError("the sandbox's egress policy is not what was asked for")
    except Exception as exc:                                    # noqa: BLE001
        sandbox.terminate(args.name)
        broker.terminate()
        print(f"could not stand the local environment up: {exc}", file=sys.stderr)
        return 1

    gpu = {str(d): harbor_env.run_agent.device_uuid(d) for d in devices}
    names = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                           capture_output=True, text=True, check=False).stdout
    record = {
        "container": container, "ip": "127.0.0.1", "backend": "local",
        "sandbox": body, "gpu_index": devices[0], "gpu_indices": devices,
        "gpu_uuids": gpu, "gpu_names": names.splitlines(),
        "image": "local namespace sandbox (no container image)",
        "model_path": MODEL_MOUNT, "model_dir": str(args.model.resolve()),
        "env_input": str(args.env_input.resolve()),
        "agent_cli": args.agent_cli, "agent_model": args.agent_model,
        "reasoning_effort": args.agent_effort,
        "agent_preflight": agent_preflight,
        "agent_probe": (preflight.stdout + preflight.stderr).strip().splitlines()[-6:],
        "codex_workspace_skills_linked": codex_skills,
        "paths": placed, "path_mechanism": "bind" if binding else "symlink",
        "private_dirs_root_only": hardened,
        "datasets_linked": datasets_linked,
        "previous_agent_cleanup": wiped,
        "egress": egress, "egress_ok": egress_ok,
        "egress_broker": {
            "endpoint": f"unix:{socket_path} (127.0.0.1:{sandbox.PROXY_PORT} "
                        f"inside the sandbox)",
            "pid": broker.pid, "log": str(broker_log),
            "allowed_hosts": sorted(hosts)},
        "credentials_copied": bool(staged),
        "isolation_limitation": (
            "The sandbox's network namespace has only loopback; its one way "
            "out is a relay to a harness-owned CONNECT broker that allows the "
            "model provider's hosts and refuses everything else. There is no "
            "mount namespace: the agent sees this host's filesystem with an "
            "unprivileged user's permissions, and every private directory is "
            "root-only."),
        "seconds_to_stand_up": round(time.time() - began, 1),
    }
    text = json.dumps(record, indent=1, sort_keys=True) + "\n"
    (args.out / "container.json").write_text(text, "utf-8")
    (args.out / "harbor_environment.json").write_text(text, "utf-8")
    print(f"container   {container}")
    print(f"stood up in {record['seconds_to_stand_up']}s")
    return 0


def codex_preflight(container: str, model: str, effort: str) -> dict:
    marker = "SERVERBENCH_AGENT_PREFLIGHT_OK"
    import shlex
    command = ["codex", "exec", "--ephemeral", "--ignore-user-config",
               "--strict-config", "--skip-git-repo-check",
               "--dangerously-bypass-approvals-and-sandbox", "--json"]
    command += harbor_env.lifecycle_mod("agent").codex_config(model, effort)
    command += [f"Use the shell to run `printf {marker}`; then answer with "
                f"the single word OK."]
    got = run_in(container, shlex.join(command), timeout=300)
    events = []
    for line in (got.stdout or "").splitlines():
        with contextlib.suppress(json.JSONDecodeError):
            events.append(json.loads(line))
    tool_marker = any(
        event.get("type") == "item.completed"
        and (event.get("item") or {}).get("type") == "command_execution"
        and marker in str((event.get("item") or {}).get("aggregated_output") or "")
        for event in events)
    ok = got.returncode == 0 and tool_marker
    return {"ok": ok, "tool_marker": tool_marker, "exit_code": got.returncode,
            "detail": ("Codex authenticated and executed a shell command" if ok
                       else ((got.stdout or "") + (got.stderr or ""))[-500:])}


if __name__ == "__main__":
    raise SystemExit(main())
