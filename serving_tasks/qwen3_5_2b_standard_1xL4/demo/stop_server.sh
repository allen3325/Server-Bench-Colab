#!/bin/bash
# Stop the server started by start_server.sh -- by its PROCESS GROUP (recorded in
# $SERVER_PIDFILE, default /tmp/server.pid). This kills ONLY the server (and any workers it
# spawned), never the agent harness, so it is always safe.
#
# Do NOT stop the server with a broad `pkill -f` (e.g. `pkill -f server.py`): `pkill -f`
# matches full command lines, and the agent harness's own command line contains the task
# text (which mentions server.py) -- a broad pkill will kill the agent itself and end
# your session. Use this script instead.
APP_DIR="${APP_DIR:-/app}"
PIDFILE="${SERVER_PIDFILE:-/tmp/server.pid}"
if [ -f "$PIDFILE" ]; then
  pgid=$(cat "$PIDFILE" 2>/dev/null)
  if [ -n "$pgid" ]; then
    # The file normally holds a process-GROUP id (start_server.sh writes the setsid leader).
    # If something wrote a plain PID instead, -PGID signals nothing -- fall back to the PID so
    # the server still stops rather than silently surviving and holding the GPU.
    if kill -0 -"$pgid" 2>/dev/null; then
      kill -TERM -"$pgid" 2>/dev/null
      for _ in $(seq 1 40); do kill -0 -"$pgid" 2>/dev/null || break; sleep 0.5; done
      kill -KILL -"$pgid" 2>/dev/null
      echo "server stopped (pgid $pgid)"
    else
      kill -TERM "$pgid" 2>/dev/null
      for _ in $(seq 1 40); do kill -0 "$pgid" 2>/dev/null || break; sleep 0.5; done
      kill -KILL "$pgid" 2>/dev/null
      echo "server stopped (pid $pgid; pidfile held a PID, not a process group)"
    fi
  else
    echo "empty pidfile; nothing to stop"
  fi
  rm -f "$PIDFILE"
else
  echo "no server pidfile ($PIDFILE); nothing to stop"
fi
