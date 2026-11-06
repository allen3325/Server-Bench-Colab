#!/usr/bin/env bash
# Bounded functional check for qwen3_5_2b_standard_1xL4. Proves the server answers; grades
# nothing. Fails loudly rather than hanging: a smoke test that waits forever is
# indistinguishable from a server that never came up.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE="${BASE_URL:-http://127.0.0.1:8000}"

for i in $(seq 1 60); do
    code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 3 "$BASE/v1/models" || true)
    [ "$code" = "200" ] && break
    [ "$i" = "60" ] && { echo "server did not answer /v1/models within 300s" >&2; exit 1; }
    sleep 5
done

echo "server is up; sending qwen3_5_2b_standard_1xL4 requests"
python3 "$HERE/example_requests.py"
echo "smoke ok"
