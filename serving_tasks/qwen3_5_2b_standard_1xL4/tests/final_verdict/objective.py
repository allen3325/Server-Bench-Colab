"""Economic objective shared by development measurements and the verdict.

Both paths call this module so they use the same token totals, prices and GPU
cost.  The work count is completed-request input plus generated output; the
task's declared tariff converts that work to dollars: ``gross_dollars_per_dollar``
counts every completed request. ``dollars_per_dollar`` bills only the input
and output tokens of requests that meet BOTH latency bars, over the full GPU
cost. The request-count attainment fraction is diagnostic, not a multiplier.

## What counts, and what must never be counted

* **Each completed request's full logical input, once** -- cached prefixes
  included. The tokens are part of the work the trace offered whether or not
  the server had to recompute them.
* **Actual generated output**, under the frozen token-counting convention.
* **Never** cache-read tokens, internal re-prefill, or backend-only padding. A
  re-prefill after an eviction burns GPU-seconds and delivers nothing new;
  those tokens were counted when the turn that first read them completed.
* **Never** an estimate from throughput x duration when the real totals exist.
  Where they do not, the result says so and is marked not comparable.
* **Never** the server's own `usage` self-report. aiperf's `total_isl` /
  `total_osl` are measured by the client; a system under test must not be the
  source of the number it is scored on.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

#: The objective, and the diagnostic it is derived from. The gross figure
#: counts every request as if it met the bars; it decides nothing.
OBJECTIVE = "dollars_per_dollar"
GROSS = "gross_dollars_per_dollar"

try:
    from final_validate import token_prices
except ImportError:                      # the in-Harbor copy
    from . import token_prices

#: aiperf's names for the two totals, and ours. The export carries both
#: `total_isl`/`total_osl` (the client's count) and `total_usage_*` (the
#: server's self-report). We read the client's.
CLIENT_INPUT = "total_isl"
CLIENT_OUTPUT = "total_osl"

#: What `scenario.py`'s aggregate calls them once mapped.
AGGREGATE_INPUT = "total_input_tokens"
AGGREGATE_OUTPUT = "total_output_tokens"

#: aiperf's prefix-reuse metric. A PERCENTAGE of 64-token blocks, with `count`
#: the blocks offered and `sum` the blocks that were a hit -- derived from the
#: TRACE's own block hashes, not from anything the server reports about itself.
#: That provenance is the reason it may be scored on: a system under test must
#: not be the source of the number it is scored on, and this is a property of
#: the workload.
CACHE_HIT = "theoretical_prefix_cache_hit"


def cached_input_tokens(body: dict[str, Any],
                        input_tokens: float | None) -> float | None:
    """Tokens of `input_tokens` that the trace says were a cache hit, or None.

    `theoretical_prefix_cache_hit` carries `{count: blocks offered, sum: blocks
    hit}`, so the hit FRACTION is sum/count and the token count is that
    fraction of the measured input. Derived from the ratio rather than from
    `sum x 64` because the block size is the scenario's business, and a
    hardcoded 64 here would be a second definition of it.

    None when the export does not carry the metric -- never 0, which would
    claim nothing was cached and, on this corpus, overstate the value
    several-fold.
    """
    entry = body.get(CACHE_HIT)
    if not isinstance(entry, dict) or input_tokens is None:
        return None
    count = _number(entry.get("count"))
    hit = _number(entry.get("sum"))
    if count is None or hit is None or count <= 0:
        return None
    return max(0.0, min(1.0, hit / count)) * float(input_tokens)


def _number(value: Any) -> float | None:
    """A real number, or None. `bool` is not a measurement and NaN is not one."""
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    return number


def totals_from(body: dict[str, Any]) -> tuple[float | None, float | None]:
    """`(input, output)` totals out of either shape that carries them.

    Accepts a `scenario.py` aggregate (`total_input_tokens`) or an aiperf
    export summary (`total_isl`), because the verdict reads one and the
    trajectory reads the other and they must not disagree about the answer.
    """
    def pick(*names: str) -> float | None:
        for name in names:
            entry = body.get(name)
            if isinstance(entry, dict):          # aiperf writes {"avg": ...}
                entry = entry.get("avg")
            number = _number(entry)
            if number is not None:
                return number
        return None

    return (pick(AGGREGATE_INPUT, CLIENT_INPUT),
            pick(AGGREGATE_OUTPUT, CLIENT_OUTPUT))


def score(body: dict[str, Any], *, usd_per_hour: float,
          duration_s: float | None = None,
          prices: dict[str, Any] | None = None) -> dict[str, Any]:
    """Total tokens per dollar, with the three counts reported separately.

    `body` is an aggregate or an aiperf summary. `duration_s` overrides the
    duration found in the body, for a caller that knows the scored window.

    Returns every field absent rather than zero when the evidence is not there:
    a zero is a claim about a system, and "we could not tell" is not.
    """
    out: dict[str, Any] = {"tokens_per_usd": None, "total_tokens": None,
                           "input_tokens": None, "output_tokens": None}
    seconds = _number(duration_s)
    if seconds is None:
        entry = body.get("benchmark_duration_s", body.get("benchmark_duration"))
        if isinstance(entry, dict):
            entry = entry.get("avg")
        seconds = _number(entry)

    isl, osl = totals_from(body)
    if not seconds or (isl is None and osl is None):
        return out
    total = (isl or 0.0) + (osl or 0.0)
    hours = seconds / 3600.0
    declared_prices = ((prices or {}).get("token_prices") or {})
    out["input_tokens"] = round(isl) if isl is not None else None
    out["output_tokens"] = round(osl) if osl is not None else None
    out["total_tokens"] = round(total)
    out["hours"] = round(hours, 4)
    out["usd"] = round(hours * usd_per_hour, 4) if usd_per_hour else None
    if hours and usd_per_hour:
        out["tokens_per_usd"] = round(total / (hours * usd_per_hour), 1)
    out["basis"] = (
        f"(input {out['input_tokens']} + output {out['output_tokens']}) "
        f"/ ({hours:.4f} h x ${usd_per_hour:.2f} per hour, the task's frozen "
        f"rate). TOTAL tokens")

    # GROSS DOLLARS PER DOLLAR: what ALL the work was worth at the task's
    # prices, over what the GPU-hours cost -- every request counted as if it
    # met the latency bars. `request_scored` turns it into the objective.
    cached = cached_input_tokens(body, isl)
    priced = token_prices.value_of(
        input_tokens=isl, output_tokens=osl, cached_input_tokens=cached,
        prices=token_prices.table(prices),
        price_source=declared_prices.get("source"))
    out["value_usd"] = priced.get("value_usd")
    out["cached_input_tokens"] = priced.get("cached_input_tokens")
    out["uncached_input_tokens"] = priced.get("uncached_input_tokens")
    out["token_prices_usd_per_million"] = priced.get("prices_usd_per_million")
    out["token_price_source"] = priced.get("price_source")
    if priced.get("cache_evidence"):
        out["cache_evidence"] = priced["cache_evidence"]
    cost = hours * usd_per_hour
    if priced.get("value_usd") is not None and cost:
        out[GROSS] = round(priced["value_usd"] / cost, 4)
        # Derive the stated cache share from the rates actually used so the
        # explanation cannot disagree with the arithmetic.
        rates_m = priced.get("prices_usd_per_million") or {}
        inp, cach = rates_m.get("input"), rates_m.get("cached_input")
        share = (f"{cach / inp * 100:.0f}% of input"
                 if isinstance(inp, (int, float)) and inp
                 and isinstance(cach, (int, float)) else "a stated share of input")
        out[GROSS + "_basis"] = (
            f"${priced['value_usd']:.4f} of tokens at list price / "
            f"${cost:.4f} of GPU-time ({hours:.4f} h x ${usd_per_hour:.2f}/h). "
            f"Below 1.0 means this service loses money at list price on this "
            f"hardware. The cached-input rate is a CONVENTION ({share}), "
            f"not a published price, and moving it moves this ratio severalfold")
    return out


#: THE REQUEST-SCORING POLICY. One latency target, the p90 row,
#: and a binary request score: a profiling request that meets BOTH the p90
#: TTFT and the p90 TPOT bar earns 1, every other request -- including errors
#: and cancellations -- earns 0. Only qualifying requests' token value is
#: billed; there is no hard latency gate.
REQUEST_SCORING_KIND = "request_p90_billable_tokens_v4"
REQUEST_SCORES = {"p90_both": 1.0, "otherwise": 0.0}


def request_scoring_error(config: dict[str, Any] | None) -> str:
    """Validate the task's `request_scoring` block; unknown policies never score."""
    cfg = config or {}
    if (cfg.get("kind") != REQUEST_SCORING_KIND
            or cfg.get("gross_objective") != GROSS
            or cfg.get("aggregation") != "sum_token_value"
            or cfg.get("attainment_basis") != "profiling_request_count"):
        return (f"request scoring requires kind {REQUEST_SCORING_KIND}, "
                f"gross_objective {GROSS}, profiling request counts and sum_token_value "
                "aggregation")
    scores = cfg.get("request_scores")
    if not isinstance(scores, dict) or set(scores) != set(REQUEST_SCORES):
        return f"request scoring requires request scores {sorted(REQUEST_SCORES)}"
    for name, expected in REQUEST_SCORES.items():
        if _number(scores.get(name)) != expected:
            return ("request scores must be "
                    + ", ".join(f"{k}={v:g}" for k, v in REQUEST_SCORES.items()))
    return ""


def _count(value: Any) -> int | None:
    number = _number(value)
    return (int(number) if number is not None and number >= 0
            and number.is_integer() else None)


def request_attainment(artifact_dir: str | Path | None,
                       checks: list[dict[str, Any]], *,
                       expected_requests: Any,
                       expected_errors: Any) -> dict[str, Any]:
    """Count the client's profiling records, never interpolate percentiles.

    Error/cancelled records remain in N and fail both targets. A successful
    one-token reply has no TPOT: it scores 0 and is counted, never a
    zero-latency pass. Any other successful record without a finite latency
    makes the evidence unavailable.
    Aggregate counts bind the export so a truncated or duplicate export cannot
    improve attainment. Aiperf's request_count counts non-error records; its
    separate error_request_count is absent on clean runs. N is their SUM.
    Warmup is excluded explicitly, not by missing metrics.
    """
    out: dict[str, Any] = {
        "schema": "final_validate.request_attainment/4", "evaluable": False,
        "request_count": 0, "error_request_count": 0,
        "cancelled_request_count": 0, "single_token_request_count": 0,
        "terms": {},
        "request_score_counts": {"p90_both": 0, "otherwise": 0},
        "completed_input_tokens": 0, "completed_output_tokens": 0,
        "billable_input_tokens": 0, "billable_output_tokens": 0,
        "why": "",
    }
    def unavailable(reason: str) -> dict[str, Any]:
        out["why"] = reason
        return out

    expected = _count(expected_requests)
    errors = 0 if expected_errors is None else _count(expected_errors)
    if expected is None or errors is None or expected + errors == 0:
        return unavailable("missing or invalid aggregate request/error counts")
    if not artifact_dir:
        return unavailable("no per-request export directory")
    path = Path(artifact_dir) / "profile_export.jsonl"
    out["export"] = str(path)
    by_percentile = {row.get("percentile"): row for row in checks}
    if len(checks) != 1 or set(by_percentile) != {90}:
        return unavailable("expected exactly the p90 latency targets")
    for percentile in (90,):
        for metric in ("ttft", "tpot"):
            limit = _number(by_percentile[percentile].get(f"{metric}_ms_gate"))
            if limit is None or limit <= 0:
                return unavailable("missing or invalid latency target")
            out["terms"][f"p{percentile}_{metric}"] = {
                "target_fraction": percentile / 100, "threshold_ms": limit,
                "met_requests": 0,
            }
    ids: set[str] = set()
    try:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                if not isinstance(record, dict):
                    return unavailable("non-object per-request record")
                metadata = record.get("metadata")
                if not isinstance(metadata, dict):
                    return unavailable("per-request record lacks metadata")
                phase = metadata.get("benchmark_phase")
                if phase == "warmup":
                    continue
                if phase != "profiling":
                    return unavailable("unknown or missing benchmark phase")
                request_id = metadata.get("x_request_id")
                if not isinstance(request_id, str) or not request_id or request_id in ids:
                    return unavailable("missing or duplicate profiling request ID")
                ids.add(request_id)
                out["request_count"] += 1
                if record.get("error"):
                    out["error_request_count"] += 1
                    continue
                if metadata.get("was_cancelled"):
                    out["cancelled_request_count"] += 1
                    continue
                if metadata.get("context_overflow_skip"):
                    return unavailable("profiling request skipped for context overflow")
                metrics = record.get("metrics")
                if not isinstance(metrics, dict):
                    return unavailable("successful request lacks metrics")
                tokens = {}
                for name in ("input", "output"):
                    entry = metrics.get(f"{name}_sequence_length")
                    count = (_count(entry.get("value"))
                             if isinstance(entry, dict) else None)
                    if count is None or entry.get("unit") != "tokens":
                        return unavailable(f"missing/invalid client {name} token count")
                    tokens[name] = count
                    out[f"completed_{name}_tokens"] += count
                values = {}
                for metric, name in (("ttft", "time_to_first_token"),
                                     ("tpot", "inter_token_latency")):
                    entry = metrics.get(name)
                    value = _number(entry.get("value")) if isinstance(entry, dict) else None
                    if value is None or value < 0 or entry.get("unit") != "ms":
                        values[metric] = None
                    else:
                        values[metric] = value
                if values["ttft"] is None:
                    return unavailable("missing/invalid ttft in successful request")
                if values["tpot"] is None:
                    # A ONE-TOKEN REPLY HAS NO INTER-TOKEN INTERVAL. The trace
                    # can ask for exactly one output token; such a request is
                    # successful and has no TPOT. It scores 0 (the brief: "a
                    # one-token reply with no TPOT is not a zero-latency
                    # pass") and is not a malformed record. Treating it as
                    # malformed would make a whole run's dollars_per_dollar
                    # unavailable because of one such turn -- in the verdict
                    # too, since it scores with this function.
                    length = metrics.get("output_sequence_length")
                    tokens = (_number(length.get("value"))
                              if isinstance(length, dict) else None)
                    if tokens != 1:
                        return unavailable("missing/invalid tpot in successful request")
                    out["single_token_request_count"] += 1
                    for key, term in out["terms"].items():
                        if key.endswith("_ttft") and values["ttft"] <= term["threshold_ms"]:
                            term["met_requests"] += 1
                    out["request_score_counts"]["otherwise"] += 1
                    continue
                for key, term in out["terms"].items():
                    if values[key.split("_")[1]] <= term["threshold_ms"]:
                        term["met_requests"] += 1
                p90 = by_percentile[90]
                if all(values[metric] <= p90[f"{metric}_ms_gate"]
                       for metric in ("ttft", "tpot")):
                    out["request_score_counts"]["p90_both"] += 1
                    for name, count in tokens.items():
                        out[f"billable_{name}_tokens"] += count
                else:
                    out["request_score_counts"]["otherwise"] += 1
    except (OSError, ValueError) as exc:
        return unavailable(f"unreadable per-request export: {exc}")
    n = out["request_count"]
    if n != expected + errors or out["error_request_count"] != errors:
        return unavailable("per-request export and aggregate request/error counts disagree")
    # Error and cancelled requests remain in N and earn zero. They were
    # deliberately skipped above because they carry no trustworthy latency.
    scored = sum(out["request_score_counts"].values())
    out["request_score_counts"]["otherwise"] += n - scored
    for term in out["terms"].values():
        misses = n - term["met_requests"]
        term.update(request_count=n, missed_requests=misses,
                    attainment_fraction=term["met_requests"] / n)
    counts = out["request_score_counts"]
    out["request_score_sum"] = counts["p90_both"]
    out["request_multiplier"] = out["request_score_sum"] / n
    out["evaluable"] = True
    return out


def request_scored(gross: Any, checks: list[dict[str, Any]],
                   config: dict[str, Any] | None, *,
                   artifact_dir: str | Path | None = None,
                   expected_requests: Any = None,
                   expected_errors: Any = None,
                   expected_input_tokens: Any = None,
                   expected_output_tokens: Any = None,
                   prices: dict[str, Any] | None = None,
                   usd_per_hour: Any = None,
                   duration_s: Any = None,
                   unavailable_reason: str = "") -> dict[str, Any]:
    """Bill qualifying requests' client token counts over the whole GPU cost.

    ``gross`` is never used to calculate or infer billable revenue. All live
    tasks price cached input at full input price. Unequal rates require
    per-request cache evidence which the pinned export does not supply; refuse
    rather than allocate aggregate hits to a selected subset or silently
    change the tariff.
    """
    out: dict[str, Any] = {
        "request_scoring_policy": (config or {}).get("kind") or REQUEST_SCORING_KIND,
        OBJECTIVE: None,
        "request_multiplier": None,
        "request_attainment": None,
        "request_scoring_why": "",
    }
    reason = unavailable_reason or request_scoring_error(config)
    if reason:
        out["request_scoring_why"] = reason
        return out
    seconds, rate = _number(duration_s), _number(usd_per_hour)
    try:
        rates = token_prices.table(prices)
    except (TypeError, ValueError) as exc:
        out["request_scoring_why"] = str(exc)
        return out
    if any(_number(v) is None or v < 0 for v in rates.values()):
        out["request_scoring_why"] = "invalid token prices"
        return out
    if not seconds or seconds <= 0 or not rate or rate <= 0:
        out["request_scoring_why"] = "missing or invalid GPU rate/window"
        return out
    if rates["usd_per_input_token"] != rates["usd_per_cached_input_token"]:
        out["request_scoring_why"] = (
            "unequal input/cache prices require per-request cache evidence")
        return out
    evidence = request_attainment(
        artifact_dir, checks, expected_requests=expected_requests,
        expected_errors=expected_errors)
    out["request_attainment"] = evidence
    if not evidence["evaluable"]:
        out["request_scoring_why"] = evidence["why"]
        return out
    for name, expected in (("input", expected_input_tokens),
                           ("output", expected_output_tokens)):
        count = _count(expected)
        if count is None or count != evidence[f"completed_{name}_tokens"]:
            out["request_scoring_why"] = (
                f"per-request client {name} tokens and aggregate disagree or are missing")
            return out
    value = (evidence["billable_input_tokens"] * rates["usd_per_input_token"]
             + evidence["billable_output_tokens"] * rates["usd_per_output_token"])
    cost = seconds / 3600 * rate
    if not math.isfinite(value) or not math.isfinite(cost) or cost <= 0:
        out["request_scoring_why"] = "nonfinite billable revenue or GPU cost"
        return out
    if not math.isfinite(value / cost):
        out["request_scoring_why"] = "nonfinite billable revenue/cost ratio"
        return out
    multiplier = evidence["request_multiplier"]
    out.update({
        OBJECTIVE: round(value / cost, 4),
        "request_multiplier": multiplier,
        "billable_input_tokens": evidence["billable_input_tokens"],
        "billable_output_tokens": evidence["billable_output_tokens"],
        "billable_value_usd": value,
        "billing_gpu_cost_usd": cost,
        "request_scoring_basis": (
            f"${value:.8g} of SLO-qualified request tokens / ${cost:.8g} GPU cost; "
            "only completed profiling requests meeting BOTH p90 bars are billed; "
            "request_multiplier is request-count attainment, diagnostic only"),
    })
    return out
