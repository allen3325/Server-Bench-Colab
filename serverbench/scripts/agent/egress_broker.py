#!/usr/bin/env python3
"""The agent's only way out, and it is not the internet.

WHY
===

A coding agent needs its model provider, and a direct route to the provider
from inside the task network would also be a route out. With this broker the
container's allowlist names ONE address -- the docker bridge gateway, which is
the host, not the internet -- and the firewall REJECTs everything else. The
agent has no route to PyPI, to GitHub, to Hugging Face, to the PyTorch index,
or directly to its provider endpoints. What it has is a relay, running outside
its network namespace, that will open exactly one kind of connection on its
behalf. The image's `restrict-network.sh` grants the relay only because the
runner names it in ALLOWED_HOSTS; a URL in the environment grants nothing.

WHAT IT IS
==========

An HTTP CONNECT proxy, and nothing else. It speaks no HTTP verb but CONNECT, so
it cannot fetch a URL for anybody; it refuses any host outside the exact set
selected by the host-side launcher; and it never sees plaintext, because what
it relays is the client's own TLS session. There is no certificate to install
and no interception.

WHAT IT ADDS BEYOND ISOLATION
=============================

Every connection is recorded. `--log` writes one JSON line per CONNECT --- when,
which host, allowed or refused, and how many bytes moved in each direction ---
so a run can say afterwards exactly what its agent's model traffic was: a rule
that is never exercised looks identical to a rule that is never needed.

WHAT IT IS NOT
==============

Not an authorization boundary against a hostile agent that has already escaped
the container: it is reachable from inside, and an agent could talk to it. What
it bounds is WHERE that traffic can go: the small provider-specific allowlist.
An agent cannot use it to reach a package index or a model hub.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import select
import socket
import socketserver
import sys
import threading
import time
from pathlib import Path

#: Provider-specific model/auth endpoints. The host-side launcher selects one
#: exact set; nothing in the container can widen it through an environment var.
#:
#: `openrouter.ai` is in the `claude` set because the CLI speaks the Anthropic
#: Messages API and OpenRouter serves it at `/api/v1/messages`, which is how
#: third-party-model runs reach their model. `api.fireworks.ai` is here for the
#: same reason: it exposes an Anthropic-compatible `/inference/v1/messages`.
#:
#: A host earns a place here by being a MODEL endpoint the agent CLI must
#: reach. Nothing else goes in: the point of the list is that an agent cannot
#: reach a package index or a model hub, and every addition is one more place
#: it can reach.
PROVIDER_HOSTS: dict[str, frozenset[str]] = {
    "claude": frozenset({
        "api.anthropic.com", "statsig.anthropic.com", "sentry.io",
        "openrouter.ai", "api.fireworks.ai",
    }),
    "codex": frozenset({
        "chatgpt.com", "auth.openai.com", "api.openai.com",
    }),
}

# Backwards-compatible default for direct library/CLI users. Production passes
# the selected provider's explicit set to `serve`.
ALLOWED: frozenset[str] = PROVIDER_HOSTS["claude"]

#: The only port this broker will open. HTTPS and nothing else: a tunnel to
#: port 22 or 8080 on an allowed host is still a tunnel to somewhere nobody
#: authorised, and the model API is reached over 443.
ALLOWED_PORTS: frozenset[int] = frozenset({443})

#: How long to wait for the upstream TCP connect before giving up.
CONNECT_TIMEOUT_S = 20.0
#: Idle time after which a relayed session is closed.
IDLE_TIMEOUT_S = 900.0


class _Handler(socketserver.BaseRequestHandler):
    """One CONNECT, checked, then two sockets spliced together."""

    def handle(self) -> None:                                    # noqa: C901
        started = time.time()
        client = self.request
        client.settimeout(30.0)
        host, port, raw = "", 0, b""
        try:
            raw = self._read_request_head(client)
        except (OSError, TimeoutError):
            return
        first = raw.split(b"\r\n", 1)[0].decode("latin-1", "replace")
        parts = first.split()
        if len(parts) < 2 or parts[0].upper() != "CONNECT":
            # Not a tunnel request. This proxy has no other verb on purpose:
            # a proxy that will GET a URL is a proxy that will fetch a wheel.
            self._refuse(client, 405, "this proxy speaks CONNECT and nothing else")
            self._record(host="", port=0, allowed=False, why="not-connect",
                         started=started)
            return
        target = parts[1]
        host, _, port_text = target.rpartition(":")
        host = (host or target).strip("[]").lower()
        try:
            port = int(port_text)
        except ValueError:
            port = 443
        allowed_hosts = getattr(self.server, "allowed_hosts", ALLOWED)
        if host not in allowed_hosts or port not in ALLOWED_PORTS:
            self._refuse(client, 403, f"{host}:{port} is not on this run's allowlist")
            self._record(host=host, port=port, allowed=False, why="not-allowed",
                         started=started)
            return
        try:
            upstream = socket.create_connection((host, port), CONNECT_TIMEOUT_S)
        except OSError as exc:
            self._refuse(client, 502, f"upstream refused: {exc}")
            self._record(host=host, port=port, allowed=True, why=f"upstream:{exc}",
                         started=started)
            return
        try:
            client.sendall(b"HTTP/1.1 200 Connection established\r\n\r\n")
            up, down = self._relay(client, upstream)
        finally:
            for sock in (upstream, client):
                with contextlib.suppress(OSError):
                    sock.close()
        self._record(host=host, port=port, allowed=True, why="ok",
                     started=started, sent=up, received=down)

    @staticmethod
    def _read_request_head(sock: socket.socket) -> bytes:
        """Up to the blank line, and never more than 8 KiB.

        A bound, because the head is attacker-controlled input and an unbounded
        read is a way to make this process the thing that falls over.
        """
        buf = b""
        while b"\r\n\r\n" not in buf and len(buf) < 8192:
            chunk = sock.recv(1024)
            if not chunk:
                break
            buf += chunk
        return buf

    @staticmethod
    def _refuse(sock: socket.socket, code: int, why: str) -> None:
        body = why.encode("utf-8", "replace")
        with contextlib.suppress(OSError):
            sock.sendall(
                f"HTTP/1.1 {code} Forbidden\r\nContent-Length: {len(body)}\r\n"
                f"Connection: close\r\n\r\n".encode("latin-1") + body)

    @staticmethod
    def _relay(a: socket.socket, b: socket.socket) -> tuple[int, int]:
        """Splice two sockets until either end goes away. Counts bytes."""
        a.settimeout(None)
        b.settimeout(None)
        sent = received = 0
        while True:
            try:
                ready, _, bad = select.select([a, b], [], [a, b], IDLE_TIMEOUT_S)
            except (OSError, ValueError):
                break
            if bad or not ready:
                break
            for sock in ready:
                other = b if sock is a else a
                try:
                    data = sock.recv(65536)
                except OSError:
                    return sent, received
                if not data:
                    return sent, received
                if sock is a:
                    sent += len(data)
                else:
                    received += len(data)
                try:
                    other.sendall(data)
                except OSError:
                    return sent, received
        return sent, received

    def _record(self, **fields) -> None:
        log = getattr(self.server, "log_path", None)
        if not log:
            return
        started = fields.pop("started", time.time())
        line = json.dumps({
            "kind": "egress_broker_connection", "schema_version": 1,
            "at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
            "seconds": round(time.time() - started, 3),
            **fields}, sort_keys=True)
        # Append-only and line-at-a-time: the driver reads this while the run
        # is still going, and a torn line would be indistinguishable from a
        # connection nobody made.
        with open(log, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()


class _Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True
    log_path: Path | None = None
    allowed_hosts: frozenset[str] = ALLOWED


class _UnixServer(socketserver.ThreadingUnixStreamServer):
    """The same broker on a unix socket, for the local (namespace) backend.

    A sandbox's network namespace has no route to this host's addresses, but a
    unix socket is a file and crosses network namespaces; the sandbox relays
    127.0.0.1:3128 to it. See `serving_evolve/local_sandbox.py`.
    """
    daemon_threads = True
    log_path: Path | None = None
    allowed_hosts: frozenset[str] = ALLOWED


def serve(bind: str, port: int, log: Path | None,
          allowed_hosts: frozenset[str] | None = None,
          unix: Path | None = None) -> _Server:
    if unix is not None:
        with contextlib.suppress(FileNotFoundError):
            Path(unix).unlink()
        server = _UnixServer(str(unix), _Handler)
        # The sandbox reaches it as the agent user.
        os.chmod(unix, 0o666)
    else:
        server = _Server((bind, port), _Handler)
    server.log_path = log
    server.allowed_hosts = allowed_hosts or ALLOWED
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bind", default="0.0.0.0",
                    help="address to listen on; the container reaches this "
                         "through the docker bridge gateway")
    ap.add_argument("--port", type=int, default=0,
                    help="0 picks a free port and prints it")
    ap.add_argument("--log", type=Path, default=None,
                    help="JSONL of every connection, allowed or refused")
    ap.add_argument("--allow-host", action="append", default=[],
                    help="exact HTTPS hostname to relay; repeatable")
    ap.add_argument("--print-port", action="store_true")
    ap.add_argument("--unix", type=Path, default=None,
                    help="listen on this unix socket instead of TCP (the "
                         "local namespace backend)")
    args = ap.parse_args(argv)

    allowed_hosts = frozenset(args.allow_host) or ALLOWED
    server = serve(args.bind, args.port, args.log, allowed_hosts,
                   unix=args.unix)
    if args.unix is not None:
        print(f"unix:{args.unix}", flush=True)
        chosen = 0
    else:
        chosen = server.server_address[1]
    if args.unix is not None:
        pass
    elif args.print_port:
        print(chosen, flush=True)
    else:
        print(f"egress broker on {args.bind}:{chosen}; "
              f"allowing {sorted(allowed_hosts)}", flush=True)
    # The parent is a setup script that exits; this process outlives it and is
    # owned by the driver, which recorded its pid and stops it at teardown.
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
    return 0


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    raise SystemExit(main(sys.argv[1:]))
