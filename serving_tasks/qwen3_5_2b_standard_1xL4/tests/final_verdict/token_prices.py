"""Value served tokens at the task's declared API prices.

`dollars_per_dollar` -- revenue over cost. The numerator is what a commercial
API would have charged for the work this server actually did; the denominator
is what the GPU-hours cost to do it. A ratio below 1.0 means the service loses
money at list price on this hardware.

    value = (input - cached) x input_price
          + cached          x cache_price
          + output          x output_price

    cost  = hours x devices x usd_per_gpu_hour

CACHED TOKENS ARE A SUBSET OF INPUT, NOT A THIRD CATEGORY.  Billing them at
both rates would count one token twice.  Every task declares its own tariff in
`contract.yaml`; there is no fallback price -- a task without one is refused
rather than priced as some other model.
"""

from __future__ import annotations

from typing import Any


def table(detail: dict[str, Any] | None = None) -> dict[str, float]:
    """The task's own per-token prices. Refuses a task that declares none."""
    declared = ((detail or {}).get("token_prices") or {})
    out = {}
    for name in ("usd_per_input_token", "usd_per_output_token",
                 "usd_per_cached_input_token"):
        value = declared.get(name)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(f"the task declares no {name}; its contract.yaml "
                             "must state its own token prices")
        out[name] = float(value)
    return out


def value_of(*, input_tokens: float | None, output_tokens: float | None,
             cached_input_tokens: float | None,
             prices: dict[str, float] | None = None,
             price_source: str | None = None) -> dict[str, Any]:
    """What a commercial API would have charged for this work.

    `cached_input_tokens` is the part of `input_tokens` that was a cache hit --
    a SUBSET, billed at the cache rate instead of the input rate, never as
    well as it.

    A `None` cache count is not zero. Zero would claim every token was billed
    at full price, which on this corpus overstates the value several-fold;
    absent says the evidence was not there, and the caller decides.
    """
    rates = prices or table()
    if input_tokens is None and output_tokens is None:
        return {"value_usd": None,
                "why": "no token counts, so there is nothing to price"}

    billed_in = float(input_tokens or 0.0)
    cached = cached_input_tokens
    out: dict[str, Any] = {}
    if cached is None:
        out["cache_evidence"] = (
            "absent: no prefix-cache-hit measurement in this export, so the "
            "whole input is priced at the full input rate. That is the "
            "CONSERVATIVE reading and it OVERSTATES the value on a workload "
            "with high prefix reuse -- it is not a claim that nothing was "
            "cached")
        cached_tokens = 0.0
    else:
        cached_tokens = max(0.0, min(float(cached), billed_in))
        if float(cached) > billed_in:
            out["cache_evidence"] = (
                f"clamped: the cache-hit count ({float(cached):.0f}) exceeded "
                f"the input count ({billed_in:.0f}), which cannot happen if "
                f"cached tokens are a subset of input. Clamped rather than "
                f"trusted")
    uncached = max(0.0, billed_in - cached_tokens)

    value = (uncached * rates["usd_per_input_token"]
             + cached_tokens * rates["usd_per_cached_input_token"]
             + float(output_tokens or 0.0) * rates["usd_per_output_token"])
    out.update({
        "value_usd": round(value, 6),
        "uncached_input_tokens": round(uncached),
        "cached_input_tokens": round(cached_tokens),
        "output_tokens": round(float(output_tokens or 0.0)),
        "prices_usd_per_million": {
            "input": round(rates["usd_per_input_token"] * 1e6, 4),
            "cached_input": round(rates["usd_per_cached_input_token"] * 1e6, 4),
            "output": round(rates["usd_per_output_token"] * 1e6, 4),
        },
        "price_source": price_source or "the task's declared tariff",
    })
    return out
