"""Where a run's container lives, and how the driver reaches into it.

A local run's environment is a docker container, reachable by `docker exec`
and addressable by `docker inspect`. A Modal sandbox has no docker daemon in
front of it and no bridge IP, so those two operations are the only seam, and
everything above them is shared.

There is deliberately no second implementation of the lifecycle. A run's
container is named by a string, and that string says which backend it belongs
to:

    sb260916t172128z646511-main-1     a local docker container
    modal:sb-NIgdDshqEHsPMcgticF5NJ   a Modal sandbox
    local:sb260916t172128z646511      namespaces on this host, no docker
                                      (`serving_evolve/local_sandbox.py`;
                                      Google Colab is the host it is for)

Anything that would shell out to `docker exec` asks this module for an argv
instead.

## Where things run

During an evolve, everything the agent builds or triggers runs inside the
run's container (Docker) or sandbox (Modal): the agent CLI, its own servers,
`./full-test` (cold start, probes and the AgentX replay) and `./measurements`.
The host only composes the container, freezes and hashes published trees, and
keeps the logs. The final verdict, after the deadline, runs on the host in a
fresh container of its own (Docker) or in fresh sandboxes (Modal: candidate,
then an independent pinned reference sandbox after the candidate's ends).
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
from pathlib import Path

#: A container name carrying this prefix is a Modal sandbox id.
MODAL_PREFIX = "modal:"

#: Where the harness reaches a server when it measures from inside the same
#: machine as the server. Not a default -- a deliberate consequence of there
#: being no routable address into a sandbox.
IN_SANDBOX_ADDRESS = "127.0.0.1"


def is_modal(container: str) -> bool:
    return str(container or "").startswith(MODAL_PREFIX)


#: A container name carrying this prefix is a local namespace sandbox.
LOCAL_PREFIX = "local:"


def is_local(container: str) -> bool:
    """Is this a namespace sandbox on this host (`serving_evolve.local_sandbox`)?

    Like docker, the workspace is the host directory itself (bind-mounted at
    /workspace), so nothing is pulled or pushed; unlike docker, there is no
    daemon to ask about mounts, reservations or container ids.
    """
    return str(container or "").startswith(LOCAL_PREFIX)


def _local():
    from serving_evolve import local_sandbox
    return local_sandbox


def sandbox_id(container: str) -> str:
    if not is_modal(container):
        raise ValueError(f"{container!r} is not a Modal container name")
    return str(container)[len(MODAL_PREFIX):]


def modal_python() -> str:
    """An interpreter that can `import modal`.

    The harness interpreter cannot and is not being taught to: it is shared
    with whatever else is running on this host, and pulling modal's dependency
    tree into it could upgrade grpcio or protobuf underneath a live driver.
    """
    override = os.environ.get("SERVERBENCH_MODAL_PY", "")
    if override:
        return override
    here = Path(__file__).resolve().parents[3]
    candidate = here / "harbor" / ".venv" / "bin" / "python"
    return str(candidate) if candidate.is_file() else "python3"


def _helper() -> Path:
    return (Path(__file__).resolve().parents[2]
            / "scripts" / "agent" / "modal_exec.py")


def exec_argv(container: str, workdir: str, command: str, *,
              interactive: bool = False) -> list[str]:
    """The argv that runs `command` inside this run's container.

    `interactive` means the caller will write the command's stdin, which is how
    the agent's prompt is staged without putting it on a command line.
    """
    if is_modal(container):
        argv = [modal_python(), str(_helper()), "exec",
                "--sandbox", sandbox_id(container), "--workdir", workdir,
                "--command", command]
        return [*argv, "--stdin"] if interactive else argv
    if is_local(container):
        # stdin passes straight through nsenter, so `interactive` needs nothing.
        return _local().exec_argv(container, workdir, command)
    inner = f"cd {shlex.quote(workdir)} && {command}" if workdir else command
    return ["docker", "exec", *(["-i"] if interactive else []),
            container, "bash", "-lc", inner]


def measures_from_inside(container: str) -> bool:
    """Does the evaluator run in the same container as the candidate?

    Callers that record provenance need this: it is the difference between a
    measurement the agent could not reach and one it merely did not.
    """
    return is_modal(container)


def remove(container: str, *, timeout: float = 120.0) -> str:
    """Stop and discard the container. Returns how it went, never raises."""
    try:
        if is_modal(container):
            got = subprocess.run(
                [modal_python(), str(_helper()), "terminate",
                 "--sandbox", sandbox_id(container)],
                capture_output=True, text=True, timeout=timeout, check=False)
        elif is_local(container):
            how = _local().terminate(_local().name_of(container))
            return "removed" if how == "terminated" else how
        else:
            got = subprocess.run(["docker", "rm", "-f", container],
                                 capture_output=True, text=True,
                                 timeout=timeout, check=False)
    except (OSError, subprocess.SubprocessError) as exc:
        return f"{type(exc).__name__}: {exc}"
    return "removed" if got.returncode == 0 else (
        (got.stderr or got.stdout or "").strip()[:200])


def put_file(container: str, text: str, path: str, *,
             timeout: float = 300.0) -> None:
    """Write a file into the container, without a shell seeing its contents.

    The agent's brief goes through here: quoting inside it must not be able to
    terminate the command that carries it, and it must not appear on a command
    line where a pattern drawn from the task could match the process running
    it.
    """
    if is_modal(container):
        got = subprocess.run(
            [modal_python(), str(_helper()), "put",
             "--sandbox", sandbox_id(container), "--path", path],
            input=text, text=True, capture_output=True, timeout=timeout,
            check=False)
    elif is_local(container):
        try:
            _local().put_file(container, text, path)
        except OSError as exc:
            raise RuntimeError(f"could not write {path} into {container}: "
                               f"{exc}") from exc
        return
    else:
        got = subprocess.run(
            ["docker", "exec", "-i", container, "bash", "-c",
             f"cat > {shlex.quote(path)} && chmod 600 {shlex.quote(path)}"],
            input=text, text=True, capture_output=True, timeout=timeout,
            check=False)
    if got.returncode != 0:
        raise RuntimeError(
            f"could not write {path} into {container}: "
            f"{(got.stderr or got.stdout).strip()[:400]}")


def pull_workspace(container: str, destination: Path, *,
                   source: str = "/workspace", timeout: float = 1800.0) -> str:
    """Bring the agent's tree back from the container, if it is not already here.

    On docker it already is: the workspace is a bind mount, so the host
    directory IS what the agent edited and there is nothing to copy.

    On Modal there is no bind mount. The tree was uploaded once and lives only
    in the sandbox, so without a pull a capture would read whatever the host
    still holds -- the seed. That is data loss, not staleness, once the
    sandbox is terminated.

    Returns a short description of what happened, for the journal.
    """
    if not is_modal(container):
        return "bind mount; nothing to pull"
    got = subprocess.run(
        [modal_python(), str(_helper()), "pull",
         "--sandbox", sandbox_id(container), "--source", source,
         "--dest", str(destination)],
        capture_output=True, text=True, timeout=timeout, check=False)
    if got.returncode != 0:
        raise RuntimeError(
            f"could not pull {source} out of {container}: "
            f"{(got.stderr or got.stdout).strip()[:400]}")
    return (got.stdout or "").strip()[-200:]


def describe(container: str) -> dict:
    """What this container is, for the run's own record."""
    return {"container": container,
            "backend": ("modal" if is_modal(container)
                        else "local" if is_local(container) else "docker"),
            "measures_from_inside": measures_from_inside(container),
            "address": (IN_SANDBOX_ADDRESS
                        if is_modal(container) or is_local(container)
                        else "bridge")}


def sync_requests(container: str, destination: Path) -> None:
    """Copy only tiny request envelopes; never scan/copy GPU evidence on a tick."""
    if not is_modal(container):
        return
    got = subprocess.run([modal_python(), str(_helper()), "requests",
                          "--sandbox", sandbox_id(container)],
                         capture_output=True, text=True, timeout=30, check=False)
    if got.returncode:
        raise RuntimeError(f"Modal request sync failed: {got.stderr[-400:]}")
    root = Path(destination) / ".versions"
    root.mkdir(exist_ok=True)
    for name, body in json.loads(got.stdout).items():
        if Path(name).name != name or not name.endswith(".json"):
            raise ValueError("unsafe Modal request name")
        path = root / name
        if not path.resolve().is_relative_to(root.resolve()):
            raise ValueError("Modal request escapes workspace")
        temporary = path.with_suffix(".sync-tmp")
        temporary.write_text(body)
        temporary.replace(path)


def push_directory(container: str, source: Path, destination: str) -> None:
    if not is_modal(container):
        return
    got = subprocess.run([modal_python(), str(_helper()), "push",
                          "--sandbox", sandbox_id(container),
                          "--source", str(source), "--dest", destination],
                         capture_output=True, text=True, timeout=600, check=False)
    if got.returncode:
        raise RuntimeError(f"Modal evidence transfer failed: {got.stderr[-400:]}")
