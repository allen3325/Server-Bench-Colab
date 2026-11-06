#!/usr/bin/env python3
"""Prove, on this machine, that the local backend's isolation holds.

Stands a throwaway sandbox up behind a throwaway egress broker and checks, from
inside it, as the agent user:

* the agent is not root and cannot read /root or the repository's private dirs;
* no host on the internet is reachable directly (no route, no DNS);
* through the broker, only the provider is reachable (pypi/github refused);
* loopback works and the GPU is usable;
* every process it leaves behind is killed by `terminate`.

Also reports whether bind mounts are available (they are preferred; symlinks
plus permissions are the fallback). Exit status 0 only if every check holds.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "serverbench" / "harness"))
sys.path.insert(0, str(REPO / "serverbench" / "scripts" / "agent"))

from serving_evolve import local_sandbox as sandbox  # noqa: E402
import local_env  # noqa: E402

NAME = "selfcheck"


def main() -> int:
    sandbox.agent_ids()
    sock = sandbox.STATE_ROOT / f"{NAME}.egress.sock"
    sandbox.STATE_ROOT.mkdir(parents=True, exist_ok=True)
    broker = subprocess.Popen(
        [sys.executable, str(REPO / "serverbench/scripts/agent/egress_broker.py"),
         "--unix", str(sock), "--allow-host", "api.anthropic.com",
         "--allow-host", "api.openai.com"],
        stdout=subprocess.PIPE, text=True, start_new_session=True)
    broker.stdout.readline()
    results = {}
    try:
        sandbox.create(NAME, env={}, proxy_socket=str(sock),
                       path=(f"{local_env.AGENT_PYTHON_BIN}:{local_env.TOOLS_BIN}:"
                             "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"))
        container = sandbox.PREFIX + NAME
        probes = local_env.egress_probes(container, "api.anthropic.com")
        results.update(probes)

        def check(name, command, want):
            got = local_env.run_in(container, command, timeout=120)
            text = (got.stdout + got.stderr).strip()
            results[name] = {"ok": want in text, "detail": text[-200:]}

        check("runs_as_agent_not_root", "id -un", "agent")
        check("loopback_tcp",
              "python3 -c \"import socket;s=socket.socket();s.bind(('127.0.0.1',8000));"
              "s.listen();socket.create_connection(('127.0.0.1',8000));print('LOOPBACK_OK')\"",
              "LOOPBACK_OK")
        check("gpu_visible", "nvidia-smi -L", "GPU 0")
        check("repo_private_dirs_unreadable",
              f"ls {REPO}/serverbench/artifacts/eval_samples >/dev/null 2>&1 "
              f"&& echo READABLE || echo UNREADABLE", "UNREADABLE")
        local_env.run_in(container, "nohup sleep 600 >/dev/null 2>&1 &")
        time.sleep(1)
    finally:
        how = sandbox.terminate(NAME)
        broker.terminate()
    left = sandbox.kill_agent_processes(timeout=5)
    results["terminate_leaves_nothing"] = {"ok": how == "terminated" and not left,
                                           "detail": f"{how}; left {left}"}
    binding = local_env.can_bind()
    width = max(len(k) for k in results)
    for name, result in results.items():
        mark = "\033[32mok  \033[0m" if result["ok"] else "\033[31mFAIL\033[0m"
        print(f"   {mark} {name:<{width}}  {result['detail'][-70:]}")
    print(f"   path mechanism: {'bind mounts' if binding else 'symlinks + permissions (no mount permission)'}")
    (REPO / "serverbench/artifacts").mkdir(exist_ok=True)
    (REPO / "serverbench/artifacts/local-selfcheck.json").write_text(
        json.dumps({"results": results, "bind_mounts": binding}, indent=1))
    return 0 if all(r["ok"] for r in results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
