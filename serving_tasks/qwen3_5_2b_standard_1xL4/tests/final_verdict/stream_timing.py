"""Distinguish streamed generation from output buffered until completion.

TPOT is measured from token arrival times.  Holding generated tokens and then
flushing them back-to-back makes TPOT describe socket writes rather than decode
and can therefore make a non-streaming server appear to pass a TPOT gate.

A request is burst-delivered when its implied decode rate exceeds
``BURST_TOKENS_PER_S``.  An arm is buffered only when that happens for at least
``BUFFERED_SHARE`` of eligible requests.  The share rule permits occasional
multi-token chunks from speculative decoding while detecting systematic
end-of-request flushing.

A buffered arm's TPOT is unmeasured, so its TPOT gate cannot hold.  TTFT stays
as observed because the client genuinely received its first token that late.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

SCHEMA = "final_validate.stream_timing/1"

#: Above this, one request's decode cannot have been streamed as generated.
BURST_TOKENS_PER_S = 2000.0

#: Share of eligible requests that must be burst-delivered to call the arm
#: buffered.  Requiring a share avoids firing on occasional multi-token chunks.
BUFFERED_SHARE = 0.25

#: Shorter outputs are ambiguous: a handful of tokens can legitimately arrive
#: in one or two chunks from any server.
MIN_OUTPUT_TOKENS = 64

#: Fewer eligible requests than this is not enough to call anything.
MIN_ELIGIBLE = 3


def _value(metrics: dict, name: str):
    entry = metrics.get(name)
    return entry.get("value") if isinstance(entry, dict) else None


def assess(artifact_dir: Path) -> dict[str, Any]:
    """Read aiperf's per-request export and say whether output was streamed.

    Never raises. An export that is missing or unreadable returns
    `evaluable: False`, which leaves TPOT exactly as measured -- the absence of
    evidence of buffering is not evidence of it.
    """
    path = Path(artifact_dir) / "profile_export.jsonl"
    result: dict[str, Any] = {
        "schema": SCHEMA, "export": str(path), "evaluable": False,
        "buffered": False, "eligible": 0, "burst_delivered": 0,
        "burst_share": None,
        "rule": (f"burst-delivered = (output tokens - 1) / decode span > "
                 f"{BURST_TOKENS_PER_S:g} tok/s, on requests with >= "
                 f"{MIN_OUTPUT_TOKENS} output tokens; buffered when >= "
                 f"{BUFFERED_SHARE:.0%} of at least {MIN_ELIGIBLE} are"),
        "why": "",
    }
    if not path.is_file():
        result["why"] = "no per-request export, so delivery timing is unknown"
        return result
    rates: list[float] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    record = json.loads(line)
                except ValueError:
                    continue
                if (record.get("metadata") or {}).get(
                        "benchmark_phase") != "profiling":
                    continue
                metrics = record.get("metrics") or {}
                tokens = _value(metrics, "output_sequence_length")
                span_ms = _value(metrics, "decode_duration")
                if (not isinstance(tokens, (int, float))
                        or tokens < MIN_OUTPUT_TOKENS
                        or not isinstance(span_ms, (int, float))):
                    continue
                rates.append(float("inf") if span_ms <= 0
                             else (tokens - 1) / (span_ms / 1000.0))
    except OSError as exc:
        result["why"] = f"could not read the export: {exc}"
        return result

    burst = sum(rate > BURST_TOKENS_PER_S for rate in rates)
    result["eligible"] = len(rates)
    result["burst_delivered"] = burst
    if len(rates) < MIN_ELIGIBLE:
        result["why"] = (f"only {len(rates)} request(s) produced >= "
                         f"{MIN_OUTPUT_TOKENS} tokens; too few to judge")
        return result
    share = burst / len(rates)
    result.update({
        "evaluable": True,
        "burst_share": round(share, 4),
        "buffered": share >= BUFFERED_SHARE,
    })
    if result["buffered"]:
        # No "the streamed requests decoded at X ms/token" here, deliberately.
        # A response flushed in three bursts instead of one clears the burst
        # threshold and still is not a streamed one: such requests imply a
        # per-token time no honest decode reaches. Quoting it as a decode
        # speed would be a second false number in place of the first.
        result["why"] = (
            f"{burst} of {len(rates)} requests delivered their whole output "
            f"in a burst after generation (> {BURST_TOKENS_PER_S:g} tok/s "
            f"implied), so TPOT measured a socket write, not decode, and is "
            f"NOT MEASURED")
    return result
