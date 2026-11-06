"""A run's environment as namespaces on THIS machine, for hosts with no Docker.

The third backend, beside the Harbor docker container and the Modal sandbox.
It exists for hosts where the operator is root but cannot run a container
runtime -- a Google Colab runtime is the case it was written for: root, a GPU,
`CAP_SYS_ADMIN`, but no `CAP_NET_ADMIN` (so no iptables) and no Docker.

## What a sandbox is

A long-lived HOLDER process, owned by root, that has unshared a new user
namespace and a new network namespace:

* the network namespace has only `lo`. There is no route out, no DNS and no
  address but 127.0.0.1, so a `pip install`, a `git clone` or a Hugging Face
  download inside it fails however it is attempted -- unsetting a proxy
  variable changes nothing;
* the user namespace maps exactly one id, the unprivileged `agent` user, to
  itself. It exists only because creating a network namespace whose `lo` can
  be brought up needs `CAP_NET_ADMIN` over it, and the user namespace is how a
  host without that capability gets it for a namespace it owns. Processes
  inside run as `agent` (uid unchanged), never as root;
* the holder relays one TCP port inside the namespace (127.0.0.1:3128) to a
  unix socket outside it, where the egress broker listens. A unix socket is a
  file, so it crosses network namespaces; the broker then allows CONNECT to the
  model provider's hosts and nothing else, exactly as on docker.

Every command the driver runs "in the container" is `nsenter` into that holder's
namespaces with the agent's uid and the run's environment (`exec_argv`). The
working tree, the model and the wheelhouse are bind-mounted at the paths the
brief names (`/workspace`, `/models/target`, `/opt/env-input`, ...), the same
paths the docker image has.

## What it is not

Not a mount namespace: the agent sees the host's filesystem with the agent
user's permissions. The verdict's private material is kept from it by
permissions (root-only directories) and by not being on the machine until the
agent's processes are gone, which `terminate` establishes.

Stdlib only, like the rest of the harness core.
"""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import fcntl
import json
import os
import pwd
import select
import shlex
import signal
import socket
import struct
import subprocess
import sys
import threading
import time
from pathlib import Path

#: The container-name prefix that says "this run's environment is a local
#: sandbox". `lifecycle/backend.py` dispatches on it.
PREFIX = "local:"

#: Root-only state: one directory per sandbox, holding its record and the
#: environment its commands run with (which carries the provider credential).
STATE_ROOT = Path(os.environ.get("SERVERBENCH_LOCAL_STATE", "/run/serverbench-local"))

AGENT_USER = "agent"
AGENT_HOME = "/home/agent"

#: The port, inside every sandbox, that reaches the egress broker.
PROXY_PORT = 3128

CLONE_NEWUSER = 0x10000000
CLONE_NEWNET = 0x40000000
SIOCGIFFLAGS = 0x8913
SIOCSIFFLAGS = 0x8914
IFF_UP = 0x1
IFF_RUNNING = 0x40


# ----------------------------------------------------------------- the user
def agent_ids() -> tuple[int, int]:
    """The agent user's uid and gid, creating the user if it does not exist."""
    try:
        entry = pwd.getpwnam(AGENT_USER)
    except KeyError:
        subprocess.run(["useradd", "-m", "-d", AGENT_HOME, "-s", "/bin/bash",
                        AGENT_USER], check=True)
        entry = pwd.getpwnam(AGENT_USER)
    return entry.pw_uid, entry.pw_gid


def name_of(container: str) -> str:
    if not str(container).startswith(PREFIX):
        raise ValueError(f"{container!r} is not a local sandbox name")
    name = str(container)[len(PREFIX):]
    if not name or "/" in name or name.startswith("."):
        raise ValueError(f"unsafe sandbox name {name!r}")
    return name


def state_dir(name: str) -> Path:
    return STATE_ROOT / name


def record(name: str) -> dict:
    return json.loads((state_dir(name) / "sandbox.json").read_text("utf-8"))


# --------------------------------------------------------------- the holder
def _libc():
    libc = ctypes.CDLL(None, use_errno=True)
    libc.unshare.argtypes = [ctypes.c_int]
    return libc


def _bring_up_loopback() -> None:
    """`ip link set lo up`, by ioctl, while this process still holds
    CAP_NET_ADMIN over the namespace it just created (an exec would drop it)."""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        request = struct.pack("16sH14s", b"lo", 0, b"\0" * 14)
        got = fcntl.ioctl(sock, SIOCGIFFLAGS, request)
        flags = struct.unpack("16sH", got[:18])[1]
        fcntl.ioctl(sock, SIOCSIFFLAGS,
                    struct.pack("16sH14s", b"lo", flags | IFF_UP | IFF_RUNNING,
                                b"\0" * 14))


def _splice(a: socket.socket, b: socket.socket) -> None:
    try:
        while True:
            ready, _, _ = select.select([a, b], [], [], 3600)
            if not ready:
                return
            for sock in ready:
                data = sock.recv(65536)
                if not data:
                    return
                (b if sock is a else a).sendall(data)
    except OSError:
        return
    finally:
        for sock in (a, b):
            with contextlib.suppress(OSError):
                sock.close()


def _relay(listener: socket.socket, unix_path: str) -> None:
    while True:
        try:
            client, _ = listener.accept()
        except OSError:
            return
        try:
            upstream = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            upstream.connect(unix_path)
        except OSError:
            client.close()
            continue
        threading.Thread(target=_splice, args=(client, upstream),
                         daemon=True).start()


def _holder(args) -> int:
    """Runs as root. Unshare, wait for the id maps, bring `lo` up, relay."""
    libc = _libc()
    if libc.unshare(CLONE_NEWUSER | CLONE_NEWNET) != 0:
        err = ctypes.get_errno()
        print(f"unshare failed: {os.strerror(err)}", flush=True)
        return 1
    print("unshared", flush=True)
    # The launcher writes our uid_map/gid_map (it holds CAP_SETUID in the
    # parent namespace; we do not) and then tells us to go on.
    if sys.stdin.readline().strip() != "go":
        return 1
    _bring_up_loopback()
    listener = None
    if args.proxy_socket:
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", PROXY_PORT))
        listener.listen(64)
        threading.Thread(target=_relay, args=(listener, args.proxy_socket),
                         daemon=True).start()
    print("ready", flush=True)
    sys.stdout.close()
    signal.signal(signal.SIGTERM, lambda *_: os._exit(0))
    while True:
        time.sleep(3600)


# ---------------------------------------------------------------- lifecycle
def create(name: str, *, env: dict[str, str], proxy_socket: str = "",
           path: str = "") -> dict:
    """Stand a sandbox up and return its record. Root only."""
    if os.geteuid() != 0:
        raise PermissionError("a local sandbox is created by root")
    uid, gid = agent_ids()
    directory = state_dir(name)
    STATE_ROOT.mkdir(parents=True, exist_ok=True)
    os.chmod(STATE_ROOT, 0o700)
    directory.mkdir(exist_ok=True)
    os.chmod(directory, 0o700)
    holder = subprocess.Popen(
        [sys.executable, "-I", str(Path(__file__).resolve()), "_holder",
         "--proxy-socket", proxy_socket],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, start_new_session=True, close_fds=True)
    try:
        first = holder.stdout.readline().strip()
        if first != "unshared":
            raise RuntimeError(f"sandbox holder did not unshare: {first!r} "
                               f"{holder.stdout.read()[-400:]!r}")
        Path(f"/proc/{holder.pid}/uid_map").write_text(f"{uid} {uid} 1\n")
        Path(f"/proc/{holder.pid}/gid_map").write_text(f"{gid} {gid} 1\n")
        holder.stdin.write("go\n")
        holder.stdin.flush()
        second = holder.stdout.readline().strip()
        if second != "ready":
            raise RuntimeError(f"sandbox holder did not come up: {second!r}")
    except Exception:
        holder.kill()
        raise
    body = {
        "name": name, "container": PREFIX + name, "holder_pid": holder.pid,
        "uid": uid, "gid": gid, "home": AGENT_HOME,
        "net_ns": os.readlink(f"/proc/{holder.pid}/ns/net"),
        "user_ns": os.readlink(f"/proc/{holder.pid}/ns/user"),
        "proxy": f"http://127.0.0.1:{PROXY_PORT}" if proxy_socket else "",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    full_env = {"HOME": AGENT_HOME, "USER": AGENT_USER, "LOGNAME": AGENT_USER,
                "SHELL": "/bin/bash", "LANG": "C.UTF-8",
                "PATH": path or "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"}
    if body["proxy"]:
        for key in ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy"):
            full_env[key] = body["proxy"]
        full_env["NO_PROXY"] = full_env["no_proxy"] = "localhost,127.0.0.1"
    full_env.update(env)
    # `bash -lc` runs /etc/profile, which ASSIGNS PATH; the profile hook
    # written by `install_profile_hook` puts the sandbox's back.
    full_env["SERVERBENCH_LOCAL_PATH"] = full_env["PATH"]
    env_file = directory / "env.json"
    env_file.write_text(json.dumps(full_env))
    os.chmod(env_file, 0o600)
    (directory / "sandbox.json").write_text(json.dumps(body, indent=1))
    install_profile_hook()
    return body


PROFILE_HOOK = Path("/etc/profile.d/00-serverbench-local.sh")


def install_profile_hook() -> None:
    text = ('# Written by serverbench local_sandbox: a login shell in a sandbox\n'
            '# keeps the PATH the sandbox was given (/etc/profile assigns one).\n'
            'if [ -n "${SERVERBENCH_LOCAL_PATH:-}" ]; then\n'
            '    export PATH="$SERVERBENCH_LOCAL_PATH"\n'
            'fi\n')
    with contextlib.suppress(OSError):
        if not PROFILE_HOOK.is_file() or PROFILE_HOOK.read_text() != text:
            PROFILE_HOOK.write_text(text)
            os.chmod(PROFILE_HOOK, 0o644)


def alive(name: str) -> bool:
    try:
        body = record(name)
        return os.readlink(f"/proc/{body['holder_pid']}/ns/net") == body["net_ns"]
    except (OSError, ValueError, KeyError):
        return False


def _zombie(entry: Path) -> bool:
    """A zombie has already exited; it cannot be signalled and holds nothing.
    (Where PID 1 does not reap -- a bare container -- they linger.)"""
    try:
        return (entry / "stat").read_text().rsplit(")", 1)[1].split()[0] == "Z"
    except (OSError, IndexError):
        return True


def _members(body: dict) -> list[int]:
    """Every process in the sandbox's network namespace or owned by the agent."""
    found = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if pid == os.getpid() or _zombie(entry):
            continue
        try:
            net = os.readlink(entry / "ns" / "net")
            uid = entry.stat().st_uid
        except OSError:
            continue
        if net == body["net_ns"] or uid == body["uid"]:
            found.append(pid)
    return found


def kill_agent_processes(uid: int | None = None, *, timeout: float = 30.0) -> list[int]:
    """SIGKILL every process owned by the agent user, and wait until none is
    left. Returns what was still there at the end (empty means clean)."""
    if uid is None:
        uid = agent_ids()[0]
    deadline = time.monotonic() + timeout
    while True:
        left = []
        for entry in Path("/proc").iterdir():
            if not entry.name.isdigit():
                continue
            with contextlib.suppress(OSError):
                if entry.stat().st_uid == uid and not _zombie(entry):
                    left.append(int(entry.name))
        for pid in left:
            with contextlib.suppress(OSError):
                os.kill(pid, signal.SIGKILL)
        if not left or time.monotonic() >= deadline:
            return left
        time.sleep(0.5)


def terminate(name: str, *, timeout: float = 30.0) -> str:
    """Kill everything in the sandbox, the holder last. Never raises."""
    try:
        body = record(name)
    except (OSError, ValueError):
        return "no such sandbox"
    deadline = time.monotonic() + timeout
    while True:
        members = [p for p in _members(body) if p != body["holder_pid"]]
        for pid in members:
            with contextlib.suppress(OSError):
                os.kill(pid, signal.SIGKILL)
        if not members or time.monotonic() >= deadline:
            break
        time.sleep(0.5)
    with contextlib.suppress(OSError):
        os.kill(int(body["holder_pid"]), signal.SIGKILL)
    leftover = [p for p in _members(body) if p != body["holder_pid"]]
    with contextlib.suppress(OSError):
        (state_dir(name) / "env.json").unlink()
    return "terminated" if not leftover else f"{len(leftover)} process(es) survived"


# -------------------------------------------------------------- running in it
def exec_argv(container: str, workdir: str, command: str) -> list[str]:
    """The argv the driver runs to execute `command` in the sandbox as the agent."""
    name_of(container)
    return [sys.executable, "-I", str(Path(__file__).resolve()), "exec",
            "--name", name_of(container), "--workdir", workdir or "",
            "--command", command]


def enter_argv(body: dict, *, as_root: bool = False) -> list[str]:
    """nsenter into the sandbox: as the agent, or (for the verdict's own
    evaluator) as root in the network namespace only."""
    pid = str(body["holder_pid"])
    if as_root:
        return ["nsenter", f"--target={pid}", "--net", "--"]
    return ["nsenter", f"--target={pid}", "--user", "--net",
            f"--setuid={body['uid']}", f"--setgid={body['gid']}", "--"]


def _exec(args) -> int:
    body = record(args.name)
    if not alive(args.name):
        print(f"sandbox {args.name} is not running", file=sys.stderr)
        return 125
    env = json.loads((state_dir(args.name) / "env.json").read_text("utf-8"))
    inner = (f"cd {shlex.quote(args.workdir)} && {args.command}"
             if args.workdir else args.command)
    argv = enter_argv(body) + ["/bin/bash", "-lc", inner]
    os.chdir("/")
    os.execvpe(argv[0], argv, env)
    return 127


def put_file(container: str, text: str, path: str) -> None:
    """Write a file the agent owns (0600), creating agent-owned parents."""
    body = record(name_of(container))
    target = Path(path)
    missing = []
    parent = target.parent
    while not parent.exists():
        missing.append(parent)
        parent = parent.parent
    for directory in reversed(missing):
        directory.mkdir()
        os.chown(directory, body["uid"], body["gid"])
    temporary = target.with_name(target.name + ".sbtmp")
    temporary.write_text(text, "utf-8")
    os.chown(temporary, body["uid"], body["gid"])
    os.chmod(temporary, 0o600)
    temporary.replace(target)


def give_to_agent(path: str | Path) -> None:
    """chown -R a tree the (root) driver wrote to the agent user.

    On docker the driver and the agent share a uid, so a directory the driver
    creates inside the workspace is the agent's. Here the driver is root, and
    without this the agent could not write into, say, the frozen source a
    development full test cold-starts.
    """
    uid, gid = agent_ids()
    root = Path(path)
    for item in [root, *root.rglob("*")]:
        with contextlib.suppress(OSError):
            os.lchown(item, uid, gid)


# -------------------------------------------------------------------- mounts
def bind(source: str | Path, target: str | Path, *, read_only: bool = False) -> None:
    """Bind `source` at `target` on this host (root), replacing an older bind."""
    target = Path(target)
    unbind(target)
    if Path(source).is_dir():
        target.mkdir(parents=True, exist_ok=True)
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.touch(exist_ok=True)
    subprocess.run(["mount", "--bind", str(source), str(target)], check=True)
    if read_only:
        subprocess.run(["mount", "-o", "remount,bind,ro", str(target)], check=True)


def unbind(target: str | Path) -> None:
    target = str(target)
    while subprocess.run(["mountpoint", "-q", target], check=False).returncode == 0:
        if subprocess.run(["umount", target], check=False).returncode != 0:
            subprocess.run(["umount", "-l", target], check=False)
            break


# ------------------------------------------------------------------------ CLI
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="verb", required=True)
    h = sub.add_parser("_holder")
    h.add_argument("--proxy-socket", default="")
    e = sub.add_parser("exec")
    e.add_argument("--name", required=True)
    e.add_argument("--workdir", default="")
    e.add_argument("--command", required=True)
    t = sub.add_parser("terminate")
    t.add_argument("--name", required=True)
    args = ap.parse_args(argv)
    if args.verb == "_holder":
        return _holder(args)
    if args.verb == "exec":
        return _exec(args)
    if args.verb == "terminate":
        print(terminate(args.name))
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
