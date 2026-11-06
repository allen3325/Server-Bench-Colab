#!/bin/bash
# (Re)start YOUR server (runs $APP_DIR/serve.sh) DETACHED so it survives across the
# harness's short-lived per-command shells. The harness runs every command in a fresh
# subshell with a short per-command timeout, and loading the model (and compiling any
# custom kernel) takes longer than that -- so a foreground "launch + wait" would be
# killed. This detaches via `setsid`, which reparents the server to its own session;
# it keeps running after this command returns.
#
# RESTART-SAFE: this first STOPS any server it previously started (by PROCESS GROUP,
# recorded in $SERVER_PIDFILE), then launches a fresh one. So to restart after editing
# server.py / serve.sh, just run this script again -- you do NOT need to kill anything
# yourself.
#
# ┌─ IMPORTANT ────────────────────────────────────────────────────────────────────┐
# │ NEVER stop the server with a broad `pkill -f` (e.g. `pkill -f server.py` or      │
# │ `pkill -f python`). `pkill -f` matches FULL COMMAND LINES, and the agent harness │
# │ running you carries the task text (which mentions server.py) in its own command  │
# │ line -- a broad pkill will match and KILL THE AGENT ITSELF, ending your session  │
# │ mid-task. To stop the server use `bash $APP_DIR/stop_server.sh` (kills only the  │
# │ server's process group by PID). If you ever must use pkill, target a precise     │
# │ full path, e.g. `pkill -f "python3 $APP_DIR/server.py"`.                        │
# └─────────────────────────────────────────────────────────────────────────────────┘
#
# Returns INSTANTLY. Then poll readiness in a separate command:
#     curl -s localhost:8000/v1/models
# Once that answers, measure throughput with:  bash $APP_DIR/bench.sh
# Logs stream to $SERVER_LOG (default /tmp/server.log) -- tail it to debug startup.
APP_DIR="${APP_DIR:-/app}"
PIDFILE="${SERVER_PIDFILE:-/tmp/server.pid}"
LOGFILE="${SERVER_LOG:-/tmp/server.log}"

# Stop a previously-started server (its whole process group), if any.
if [ -f "$PIDFILE" ]; then
  pgid=$(cat "$PIDFILE" 2>/dev/null)
  if [ -n "$pgid" ]; then
    kill -TERM -"$pgid" 2>/dev/null
    for _ in $(seq 1 40); do kill -0 -"$pgid" 2>/dev/null || break; sleep 0.5; done
    kill -KILL -"$pgid" 2>/dev/null
  fi
  rm -f "$PIDFILE"
fi

# Launch detached in its OWN session/process group; record the group id (== leader PID).
# `setsid bash -c 'echo $$ > pidfile; exec bash serve.sh'` makes $$ the new session
# leader's PID (== PGID); exec preserves that PID through serve.sh and server.py, and any
# workers the server spawns inherit the group -- so stop_server.sh can kill them all.
setsid bash -c 'echo $$ > '"$PIDFILE"'; exec bash '"$APP_DIR"'/serve.sh' >"$LOGFILE" 2>&1 &
sleep 0.3
echo "launched (server pgid $(cat "$PIDFILE" 2>/dev/null), logs: $LOGFILE)"
echo "poll readiness:  curl -s localhost:8000/v1/models"
echo "then benchmark:  bash $APP_DIR/bench.sh"
echo "to stop/restart: bash $APP_DIR/stop_server.sh  /  bash $APP_DIR/start_server.sh   (NEVER 'pkill -f')"
