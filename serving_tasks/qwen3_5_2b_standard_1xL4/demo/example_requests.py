#!/usr/bin/env python3
"""Small chat requests that exercise qwen3_5_2b_standard_1xL4's measured endpoint shape."""

import json
import os
import urllib.request

BASE = os.environ.get("BASE_URL", "http://127.0.0.1:8000")

_FILLER = ("scheduler batch kv cache prefill decode throughput latency "
           "tokenizer attention checkpoint quantization ") * 6

REQUESTS = [
    {"content": _FILLER + "\n\nSummarise the terms above in one sentence.",
      "max_tokens": 128},
    {"content": _FILLER + "\n\nName the two that concern memory.",
      "max_tokens": 128},
]


def main() -> int:
    for i, body in enumerate(REQUESTS):
        body = {"model": "target", "temperature": 0.0, "stream": False,
                "messages": [{"role": "user", "content": body["content"]}],
                "max_tokens": body["max_tokens"],
                "chat_template_kwargs": {"enable_thinking": False}}
        req = urllib.request.Request(
            f"{BASE}/v1/chat/completions", method="POST",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as resp:
            out = json.loads(resp.read())
        message = (out.get("choices") or [{}])[0].get("message") or {}
        text = message.get("content", "")
        print(f"[{i}] {len(text)} chars: {text[:120]!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
