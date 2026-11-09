#!/bin/bash
# Drive ServerBench on a Google Colab VM from your own terminal, with colab-cli
# (https://github.com/googlecolab/google-colab-cli) instead of the notebook.
# Runs on YOUR machine; every step on the VM is the same script the notebook
# runs (setup.sh, persist.sh, hardware.py, calibrate.py, local-evolve.sh,
# verdict-only.sh).
#
#   uv tool install google-colab-cli && colab usage     # once: OAuth login
#   export CLAUDE_CODE_OAUTH_TOKEN=...                  # credentials, see below
#   export RCLONE_CONF=~/.config/rclone/rclone.conf
#   serverbench/scripts/colab/remote.sh up --gpu L4     # VM, repo, setup, secrets
#   serverbench/scripts/colab/remote.sh restore
#   serverbench/scripts/colab/remote.sh task            # this card's task
#   serverbench/scripts/colab/remote.sh run claude --hours 3 --down
#
# Commands ([-s NAME] picks the session, default `sb`):
#   up [--gpu G] [--high-mem]  open the session if it is not there, push, setup,
#                              secrets
#   push                       send the working tree (tracked and untracked files
#                              git does not ignore, plus .git) to the VM
#   setup                      setup.sh, as a job
#   secrets                    write credentials from your environment into
#                              root-only files on the VM (never on a command line)
#   codex-login                codex login --device-auth on the VM (subscription)
#   restore | save | status    persist.sh restore / save / status
#   task [--usd-per-gpu-hour R]   print the card's task; create it when missing
#   calibrate search|seeds|record [--down]     calibrate.py, as a job
#   run claude|codex [--hours N] [--model M] [--effort E] [--down]
#                              local-evolve.sh, as a job
#   verdict [RUN_DIR] [--down] verdict-only.sh on a restored run, as a job
#   follow [JOB]               follow a job (default: the newest) until it ends
#   jobs                       the jobs on the VM and whether they still run
#   sh [CMD...]                a shell on the VM (or one command), in the repo,
#                              with the notebook's environment
#   pull REMOTE_PATH [LOCAL]   copy a file or directory back (relative to the repo)
#   keepalive start|stop|status
#   down                       stop the keepalive and release the VM
#
# Credentials, read from your environment by `secrets` (same as the notebook's):
#   CLAUDE_CODE_OAUTH_TOKEN | ANTHROPIC_API_KEY      Claude
#   CODEX_AUTH_JSON (file or content) | OPENAI_API_KEY   Codex (or codex-login)
#   RCLONE_CONF (file or content, a remote named `sb`)   off-VM checkpoints
#
# Jobs. Anything long runs detached on the VM (setsid nohup, log and exit code
# under /content/sb-jobs), and this script follows it by polling over short ssh
# connections every FOLLOW_EVERY seconds (30). Ctrl-C stops following, never
# the job; `follow` picks it up again.
#
# One connection at a time. colab-cli admits one `colab ssh` per runtime (a
# second gets HTTP 429), and while one is open `colab exec` fails ("Connection
# was lost"). So every ssh and every keepalive exec here takes the same local
# lock and holds it only for one short call -- except an interactive `sh` or
# `codex-login`, which holds it (and pauses the keepalive) until you leave.
#
# Keepalive. colab-cli keeps a VM while its kernel is active, but a job started
# over ssh is not the kernel's: with no notebook tab open, the VM could be taken
# for idle in the middle of a run. While a job runs, a loop on this machine
# sends the kernel a trivial `colab exec` every KEEPALIVE_EVERY seconds (300);
# it stops after KEEPALIVE_IDLE_TICKS (2) checks with no job running, so a
# finished run does not hold the VM and spend compute units. It needs this
# machine awake; the job itself does not depend on it.
#
# Stopping the VM. Nothing calls `colab stop` unless asked: `down` does, and
# `--down` on run, calibrate or verdict has the keepalive loop do it once no job
# runs -- after `persist.sh save` succeeds, never without rclone or after a
# failed save (a stopped VM's disk is gone).
set -euo pipefail

SESSION="${SB_COLAB_SESSION:-sb}"
if [ "${1:-}" = "-s" ]; then SESSION="$2"; shift 2; fi
CMD="${1:-help}"; [ $# -gt 0 ] && shift
DOWN=0; args=()
for a in "$@"; do if [ "$a" = --down ]; then DOWN=1; else args+=("$a"); fi; done
set -- ${args[@]+"${args[@]}"}

LOCAL_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
REPO=/content/Server-Bench-Colab
JOBS=/content/sb-jobs
STATE="${XDG_STATE_HOME:-$HOME/.local/state}/serverbench-colab"
KEEPALIVE_EVERY="${KEEPALIVE_EVERY:-300}"
FOLLOW_EVERY="${FOLLOW_EVERY:-30}"
LOCK="$STATE/$SESSION.lock"
mkdir -p "$STATE"

say() { printf '\033[1m== %s\033[0m\n' "$*" >&2; }
die() { printf '\033[31merror\033[0m %s\n' "$*" >&2; exit 1; }
command -v colab >/dev/null || die "colab-cli not found: uv tool install google-colab-cli"
command -v flock >/dev/null || die "flock not found (util-linux)"

# ------------------------------------------------------------- connection ---
SSH_OPTS=(-o "ProxyCommand=colab ssh --proxy-mode -s $SESSION"
          -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null
          -o LogLevel=ERROR -o ServerAliveInterval=30 -o ConnectTimeout=90)

# vm_in: stdin goes over (no retry: it would be spent). vm_n: no stdin, retried
# when ssh itself fails to connect (255). vm_tty: interactive.
vm_in() { ( flock -w 900 9 || exit 255; ssh "${SSH_OPTS[@]}" "root@colab-$SESSION" "$@" ) 9>"$LOCK"; }
vm_tty() { ( flock -w 900 9 || exit 255; ssh -t "${SSH_OPTS[@]}" "root@colab-$SESSION" "$@" ) 9>"$LOCK"; }
vm_n() {
    local i rc=255
    for i in 1 2 3 4 5; do
        ( flock -w 900 9 || exit 255; ssh -n "${SSH_OPTS[@]}" "root@colab-$SESSION" "$@" ) 9>"$LOCK" && return 0
        rc=$?; [ "$rc" = 255 ] || return "$rc"
        sleep $((i * 3))
    done
    return "$rc"
}

# The notebook's environment (cell 0) and its task choice (cell 1), on the VM.
PRELUDE="set -e; cd $REPO
export HF_HOME=/opt/serverbench/hf SERVERBENCH_TASK_ROOT=$REPO/serving_tasks
export CALIBRATION=\${CALIBRATION:-/content/sb-calibration} SB_REMOTE=\${SB_REMOTE:-sb:serverbench-colab}
export PATH=/opt/serverbench/bin:\$PATH
export TASK=${TASK:+$(printf %q "$TASK")}
[ -n \"\$TASK\" ] || TASK=\$(python3 serverbench/scripts/colab/hardware.py task)"
on_vm() { vm_n "$PRELUDE
$*"; }

session_exists() { colab sessions 2>/dev/null | grep -q "^\[$SESSION\]"; }

# ------------------------------------------------------------------- jobs ---
# start_job NAME: the job's script comes on stdin, after the prelude. The
# script travels over stdin, so no pgrep ever sees its text on a command line.
start_job() {
    local name="$1"
    { printf '%s\n' "$PRELUDE"; cat; } | vm_in "set -e; mkdir -p $JOBS; cd $JOBS
        if [ -f $name.pid ] && kill -0 \$(cat $name.pid) 2>/dev/null; then
            cat > /dev/null; echo '$name is already running'; exit 0; fi
        cat > $name.sh; rm -f $name.rc
        setsid nohup bash -c 'bash $JOBS/$name.sh; echo \$? > $JOBS/$name.rc' > $name.log 2>&1 < /dev/null &
        echo \$! > $name.pid"
    say "job $name started on $SESSION (log $JOBS/$name.log)"
}

# follow_job NAME [GREP]: print the log as it grows until the job ends; return
# its exit code. Each poll is one short ssh, so the keepalive gets its turns.
follow_job() {
    local name="$1" filter="${2:-}" off=-1 out head body size alive rc
    say "following $name every ${FOLLOW_EVERY}s (Ctrl-C stops following, not the job)"
    while :; do
        out=$(vm_n "f=$JOBS/$name.log; s=\$(stat -c %s \$f 2>/dev/null || echo 0)
            a=0; kill -0 \$(cat $JOBS/$name.pid 2>/dev/null) 2>/dev/null && a=1
            o=$off; [ \$o -ge 0 ] || o=\$(( s > 6000 ? s - 6000 : 0 ))
            echo \"\$s \$a \$o \$(cat $JOBS/$name.rc 2>/dev/null)\"
            tail -c +\$((o + 1)) \$f 2>/dev/null | head -c \$((s - o)); echo .") \
            || { echo "(poll failed; retrying)" >&2; sleep "$FOLLOW_EVERY"; continue; }
        head=${out%%$'\n'*}; body=${out#*$'\n'}; body=${body%.}
        read -r size alive _ rc <<<"$head"
        if [ -n "$filter" ]; then printf '%s' "$body" | grep -E --line-buffered "$filter" || true
        else printf '%s' "$body"; fi
        off=$size
        if [ "$alive" != 1 ]; then
            say "$name ended with exit code ${rc:-unknown}"
            return "${rc:-1}"
        fi
        sleep "$FOLLOW_EVERY"
    done
}

# --------------------------------------------------------------- keepalive ---
ka_pidfile() { echo "$STATE/$SESSION.keepalive.pid"; }
ka_running() { local p; p=$(cat "$(ka_pidfile)" 2>/dev/null) && kill -0 "$p" 2>/dev/null; }
down_marker() { echo "$STATE/$SESSION.down-when-idle"; }

# BUSY while a job (or a driver started some other way) runs on the VM, IDLE
# when none does, nothing when it cannot tell -- which the loop counts as busy.
vm_work() {
    vm_n "for p in $JOBS/*.pid /run/serverbench-local-driver.pid; do
              [ -f \$p ] && kill -0 \$(cat \$p) 2>/dev/null && { echo BUSY; exit 0; }
          done; echo IDLE" 2>/dev/null || true
}

# `colab exec` takes a second or two, but now and then hangs or loses its
# connection (about one call in six, measured) -- so a short leash, three tries.
ka_poke() {
    ( flock -w 120 9 || { echo "$(date +%T) keepalive skipped: the connection is in use"; exit 0; }
      for i in 1 2 3; do
          out=$(echo 'import time; print("keepalive", time.strftime("%H:%M:%S"))' \
                | timeout 45 colab exec -s "$SESSION" --timeout 30 2>&1 | tail -1) || true
          case "$out" in keepalive*) echo "$out"; exit 0 ;; esac
          echo "$(date +%T) keepalive try $i failed: ${out:-timed out}"
          sleep 5
      done
    ) 9>"$LOCK"
}

# Release the VM once its work is done -- but only after the results are off it.
down_when_idle() {
    echo "$(date +%T) work finished; saving before colab stop"
    if vm_n "cd $REPO && [ -f serverbench/artifacts/secrets/rclone.conf ] && bash serverbench/scripts/colab/persist.sh save"; then
        echo "$(date +%T) saved; colab stop -s $SESSION"
        colab stop -s "$SESSION"
    else
        echo "$(date +%T) NOT stopping $SESSION: no rclone config or the save failed -- the results are only on the VM" \
             "(pull them, then: $0 -s $SESSION down)"
    fi
    rm -f "$(down_marker)"
}

ka_loop() {
    local gone=0 idle=0 work
    while :; do
        if session_exists; then
            gone=0
            work=$(vm_work)
            if [ "$work" = IDLE ]; then idle=$((idle + 1)); else idle=0; fi
            if [ "$idle" -ge "${KEEPALIVE_IDLE_TICKS:-2}" ]; then
                echo "$(date +%T) no job on $SESSION for $idle checks; keepalive stops"
                # A marker newer than one interval may belong to a job that is
                # still being started; and look once more before letting go.
                if [ -f "$(down_marker)" ]; then
                    if [ $(( $(date +%s) - $(stat -c %Y "$(down_marker)") )) -lt "$KEEPALIVE_EVERY" ] \
                       || [ "$(vm_work)" != IDLE ]; then
                        idle=0; sleep "$KEEPALIVE_EVERY"; continue
                    fi
                    down_when_idle
                fi
                break
            fi
            ka_poke
            echo "$(date +%T) vm: ${work:-unknown}"
        else
            gone=$((gone + 1))      # a listing can fail once; three misses mean gone
            echo "$(date +%T) session $SESSION not listed ($gone/3)"
            [ "$gone" -ge 3 ] && break
        fi
        sleep "$KEEPALIVE_EVERY"
    done
    rm -f "$(ka_pidfile)"
}
ka_start() {
    ka_running && return 0
    nohup "$0" -s "$SESSION" _keepalive_loop >> "$STATE/$SESSION.keepalive.log" 2>&1 < /dev/null &
    echo $! > "$(ka_pidfile)"
    say "keepalive every ${KEEPALIVE_EVERY}s while a job runs (log: $STATE/$SESSION.keepalive.log)"
}
ka_stop() { local p; p=$(cat "$(ka_pidfile)" 2>/dev/null) && kill "$p" 2>/dev/null; rm -f "$(ka_pidfile)"; }

# job NAME [GREP]: start it from stdin, keep the VM alive, follow it.
job() { start_job "$1"; ka_start; follow_job "$@"; }

# ------------------------------------------------------------------- steps ---
push() {
    say "push $LOCAL_REPO -> $SESSION:$REPO"
    (cd "$LOCAL_REPO" && { git ls-files -z --cached --others --exclude-standard; printf '.git\0'; } \
        | tar --null -T - -czf -) \
        | vm_in "set -e; mkdir -p $REPO; tar -xzf - --no-same-owner -C $REPO
              git config --global --get-all safe.directory | grep -qx '\\*' || git config --global --add safe.directory '*'
              git -C $REPO log --oneline -1"
}

setup() {
    job setup '^\S*== |ok |fail|rror|attempt' <<<"bash serverbench/scripts/colab/setup.sh" \
        || die "setup.sh failed: $0 -s $SESSION sh tail -60 $JOBS/setup.log"
}

secrets() {
    local d; d=$(mktemp -d); trap 'rm -rf "$d"' RETURN
    (
        umask 077
        if [ -n "${CLAUDE_CODE_OAUTH_TOKEN:-}" ]; then
            printf 'CLAUDE_CODE_OAUTH_TOKEN=%s\n' "$CLAUDE_CODE_OAUTH_TOKEN" > "$d/claude.env"
        elif [ -n "${ANTHROPIC_API_KEY:-}" ]; then
            printf 'ANTHROPIC_BASE_URL=https://api.anthropic.com\nANTHROPIC_API_KEY=%s\n' "$ANTHROPIC_API_KEY" > "$d/claude.env"
        fi
        from_file_or_text() { if [ -f "$1" ]; then cat "$1"; else printf '%s\n' "$1"; fi; }
        [ -n "${RCLONE_CONF:-}" ] && from_file_or_text "$RCLONE_CONF" > "$d/rclone.conf"
        [ -n "${CODEX_AUTH_JSON:-}" ] && from_file_or_text "$CODEX_AUTH_JSON" > "$d/codex-auth.json"
        [ -n "${OPENAI_API_KEY:-}" ] && printf '%s' "$OPENAI_API_KEY" > "$d/openai.key"
        true
    )
    say "secrets: $(cd "$d" && ls | tr '\n' ' ')"
    [ -s "$d/rclone.conf" ] && ! grep -q '^\[sb\]' "$d/rclone.conf" && echo "warning: RCLONE_CONF has no [sb] remote" >&2
    # Over ssh's stdin into a root-only staging dir; nothing touches argv or a log.
    (cd "$d" && tar -cf - .) | vm_in "set -e; umask 077; t=\$(mktemp -d); tar -xf - -C \$t
        s=$REPO/serverbench/artifacts/secrets; mkdir -p \$s; chmod 700 \$s
        [ -f \$t/claude.env ]  && install -m 600 \$t/claude.env  \$s/claude.env  && echo 'Claude: '\$(grep -q OAUTH \$s/claude.env && echo subscription || echo 'API key')
        [ -f \$t/rclone.conf ] && install -m 600 \$t/rclone.conf \$s/rclone.conf && echo 'Checkpoints: rclone'
        mkdir -p /root/.codex; chmod 700 /root/.codex
        [ -f \$t/codex-auth.json ] && install -m 600 \$t/codex-auth.json /root/.codex/auth.json && echo 'Codex: subscription (auth.json)'
        if [ -f \$t/openai.key ] && [ ! -f \$t/codex-auth.json ]; then
            PATH=/opt/serverbench/bin:\$PATH codex login --with-api-key < \$t/openai.key >/dev/null && echo 'Codex: API key'
        fi
        rm -rf \$t
        [ -f \$s/rclone.conf ] || echo 'Checkpoints: NO rclone config -- a reclaimed VM loses the run'"
}

run_agent() {
    local cli="${1:?claude|codex}"; shift
    local hours=3 model effort
    case "$cli" in
        claude) model=claude-opus-5; effort=xhigh ;;
        codex)  model=gpt-5.6-sol;   effort=max ;;
        *) die "agent is claude or codex" ;;
    esac
    while [ $# -gt 0 ]; do case "$1" in
        --hours) hours="$2"; shift 2 ;;
        --model) model="$2"; shift 2 ;;
        --effort) effort="$2"; shift 2 ;;
        *) die "unknown option $1" ;;
    esac; done
    job "run-$cli" <<EOF
p=\$(cat /run/serverbench-local-driver.pid 2>/dev/null || true)
if [ -n "\$p" ] && [ -d /proc/\$p ]; then echo "a run is already going (driver \$p)" >&2; exit 1; fi
env=$REPO/serverbench/artifacts/secrets/claude.env
if [ $cli = claude ] && [ -f \$env ]; then export AGENT_ENV=\$env; fi
echo "TASK=\$TASK $cli $model $effort ${hours}h"
AGENT_CLI=$cli AGENT_MODEL=$model AGENT_EFFORT=$effort \
    exec bash serverbench/scripts/server_evolve/local-evolve.sh "\$TASK" --hours $hours
EOF
}

if [ "$DOWN" = 1 ]; then
    case "$CMD" in run|calibrate|verdict) ;; *) die "--down goes with run, calibrate or verdict" ;; esac
    : > "$(down_marker)"
    say "--down: colab stop once no job runs and persist.sh save succeeds"
fi

case "$CMD" in
    up)
        new_args=()
        while [ $# -gt 0 ]; do case "$1" in
            --gpu) new_args+=(--gpu "$2"); shift 2 ;;
            --high-mem) new_args+=(--high-mem); shift ;;
            *) die "unknown option $1" ;;
        esac; done
        [ ${#new_args[@]} -gt 0 ] || new_args=(--gpu L4)
        if session_exists; then say "session $SESSION is there"
        else say "colab new -s $SESSION ${new_args[*]}"; colab new -s "$SESSION" "${new_args[@]}"; fi
        push
        vm_n "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; python3 $REPO/serverbench/scripts/colab/hardware.py detect"
        setup
        secrets
        ;;
    push) push ;;
    setup) setup ;;
    secrets) secrets ;;
    codex-login) vm_tty "PATH=/opt/serverbench/bin:\$PATH; codex login --device-auth && codex login status" ;;
    restore|save|status) on_vm "bash serverbench/scripts/colab/persist.sh $CMD" ;;
    task)
        rate=0.80; [ "${1:-}" = "--usd-per-gpu-hour" ] && rate="$2"
        on_vm "if [ ! -d serving_tasks/\$TASK ]; then
                   serverbench/.venv/bin/python serverbench/scripts/colab/hardware.py new-task --usd-per-gpu-hour $rate
                   if [ -f serverbench/artifacts/secrets/rclone.conf ]; then bash serverbench/scripts/colab/persist.sh save; fi
               fi
               echo \"TASK=\$TASK\"; grep -E '^(title|hardware|usd_per_gpu_hour):' serving_tasks/\$TASK/contract.yaml"
        ;;
    calibrate)
        step="${1:?search|seeds|record}"
        case "$step" in search|seeds|record) ;; *) die "calibrate search|seeds|record" ;; esac
        job "calibrate-$step" <<EOF
extra=
if [ $step != record ]; then
    extra="--model-dir \$(ls -d /opt/serverbench/hf/hub/models--Qwen--Qwen3.5-2B/snapshots/*/ | head -1)"
fi
serverbench/.venv/bin/python serverbench/scripts/colab/calibrate.py $step --task "\$TASK" \$extra --out-dir "\$CALIBRATION"
if [ -f serverbench/artifacts/secrets/rclone.conf ]; then bash serverbench/scripts/colab/persist.sh save; fi
EOF
        ;;
    run) run_agent "$@" ;;
    verdict) job verdict <<<"bash serverbench/scripts/colab/verdict-only.sh $(printf '%q ' "$@")" ;;
    follow)
        name="${1:-$(vm_n "ls -t $JOBS/*.log 2>/dev/null | head -1 | xargs -r basename | sed 's/\.log\$//'")}"
        [ -n "$name" ] || die "no job on $SESSION"
        follow_job "$name"
        ;;
    jobs)
        vm_n "cd $JOBS 2>/dev/null || { echo 'no jobs'; exit 0; }
              for p in \$(ls -t *.pid); do n=\${p%.pid}
                  if kill -0 \$(cat \$p) 2>/dev/null; then s=running; else s=\"exit \$(cat \$n.rc 2>/dev/null || echo '?')\"; fi
                  printf '%-22s %-10s %s\n' \$n \"\$s\" \"\$(date -r \$n.log +%F' '%T)\"
              done"
        ;;
    sh)
        if [ $# -eq 0 ]; then vm_tty "cd $REPO; exec bash -l"
        else on_vm "$*"; fi
        ;;
    pull)
        src="${1:?REMOTE_PATH}"; dst="${2:-.}"; mkdir -p "$dst"
        vm_n "cd $REPO && tar -czf - $(printf %q "$src")" | tar -xzf - -C "$dst"
        say "pulled $src into $dst"
        ;;
    keepalive)
        case "${1:-status}" in
            start) ka_start ;;
            stop) ka_stop ;;
            status) if ka_running; then echo "running (pid $(cat "$(ka_pidfile)"))"; else echo "not running"; fi
                    tail -3 "$STATE/$SESSION.keepalive.log" 2>/dev/null || true ;;
        esac
        ;;
    _keepalive_loop) ka_loop ;;
    down) ka_stop; rm -f "$(down_marker)"; colab stop -s "$SESSION" ;;
    help|-h|--help) sed -n '2,/^set -euo/p' "$0" | sed '$d; s/^# \{0,1\}//' ;;
    *) die "unknown command $CMD (try: $0 help)" ;;
esac
