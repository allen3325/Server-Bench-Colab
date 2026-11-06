"""The frozen latency bars, in one place, for every task that declares them.

A task's `service_levels` is a list of percentile rows, and each row
carries BOTH a time-to-first-token ceiling and a time-per-output-token ceiling:

    service_levels:
    - percentile: 90
      ttft_ms: 2700
      tpot_ms: 12

Each row is two bars. The tasks here declare one row, p90, and its two bars
are what each profiling request is scored against: a request that meets both
earns 1, any other request earns 0 (`objective.request_scored`). The bars are
not hard gates -- the hard gates are replay fidelity, the EOS stop rule and
output correctness -- so a missed bar lowers the objective and never
disqualifies a run.

`check` also compares the run's aggregate percentiles with the same bars, one
comparison per bar. That aggregate comparison is a diagnostic: it is reported
beside a verdict and decides nothing.

WHY EACH ROW HAS TWO INDEPENDENT BARS
-------------------------------------
The second threshold on each row is a TIME-PER-OUTPUT-TOKEN ceiling. TPOT is a
per-request latency like TTFT, so it has a distribution and each row reads its
own percentile of it. The bars are independent comparisons against two
distributions:

    p90 TTFT <= ceiling     p90 TPOT <= ceiling

A throughput floor would ask something else: it is a statement about the
SERVICE's aggregate rate, which on a single-stream workload is mostly a
statement about one request. A TPOT ceiling is a statement about what
a WAITING USER feels between tokens, and it does not improve just because the
service is idle.

Every comparison is INCLUSIVE: `ttft <= ceiling`, `tpot <= ceiling`. A
measurement sitting exactly on a threshold holds it.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

#: What a service-level row must carry. A row missing either threshold is a
#: broken contract, not a row with a default -- a defaulted ceiling is a bar
#: nobody chose.
REQUIRED_FIELDS = ("percentile", "ttft_ms", "tpot_ms")


class ContractError(ValueError):
    """A task's frozen service levels are missing or malformed."""


def read(slo: dict[str, Any] | None, *, task_id: str = "") -> list[dict]:
    """The frozen rows, validated, in ascending percentile order.

    Raises rather than defaulting. A verdict with no bar is not a verdict, and
    a bar invented at read time is worse than none: it looks like the task's.
    """
    where = f"{task_id}'s " if task_id else "the task's "
    levels = list((slo or {}).get("service_levels") or [])
    if not levels:
        raise ContractError(
            f"{where}contract.yaml declares no `service_levels`, so there is "
            f"nothing to decide against")
    rows = []
    for index, level in enumerate(levels):
        if not isinstance(level, dict):
            raise ContractError(
                f"{where}service_levels[{index}] is {type(level).__name__}, "
                f"not a mapping")
        missing = [f for f in REQUIRED_FIELDS if level.get(f) is None]
        if missing:
            raise ContractError(
                f"{where}service_levels[{index}] is missing {missing}. Each "
                f"entry carries its OWN ttft ceiling and its OWN tpot "
                f"ceiling -- two rows never share one")
        try:
            rows.append({
                "percentile": int(level["percentile"]),
                "ttft_ms": float(level["ttft_ms"]),
                "tpot_ms": float(level["tpot_ms"]),
            })
        except (TypeError, ValueError) as exc:
            raise ContractError(
                f"{where}service_levels[{index}] holds a non-numeric "
                f"threshold: {level!r}") from exc
    rows.sort(key=lambda r: r["percentile"])
    percentiles = [r["percentile"] for r in rows]
    if len(set(percentiles)) != len(percentiles):
        raise ContractError(
            f"{where}service_levels declares percentile {percentiles} twice. "
            f"One row per percentile, each with its own pair of thresholds")
    return rows


def _finite(value: Any) -> float | None:
    """A real number, or None.

    `bool` is not a measurement; neither NaN nor infinity is a finite observed
    latency.
    """
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        # A STRING IS NOT A NUMBER HERE, even a string that parses.
        # `float("12")` succeeds, and accepting it means a malformed aggregate
        # -- a server reporting `"usage": {"completion_tokens": "12"}`, a field
        # someone stringified on the way through a template -- is silently
        # treated as a measurement. The aggregate is JSON numbers; anything
        # else is evidence this pipeline should refuse rather than repair.
        return None
    number = float(value)
    if number != number or number in (float("inf"), float("-inf")):
        return None
    return number


def check(levels: Sequence[dict], *, ttft_by_percentile: dict[int, Any],
          tpot_by_percentile: dict[int, Any]) -> list[dict]:
    """One row per service level, each carrying BOTH of its verdicts.

    `ttft_by_percentile` maps 90 -> measured p90 TTFT in ms, and
    `tpot_by_percentile` maps 90 -> measured p90 time-per-output-token in ms.
    Both are per-request latency distributions, so each row reads its OWN
    percentile of each, so every bar is an independent comparison.

    An absent or non-finite measurement never holds a gate. It is not evidence
    of a miss either, and `measured` says which of the two happened.
    """
    rows = []
    for level in levels:
        percentile = int(level["percentile"])
        ttft_ceiling = float(level["ttft_ms"])
        tpot_ceiling = float(level["tpot_ms"])
        ttft = _finite(ttft_by_percentile.get(percentile))
        tpot = _finite(tpot_by_percentile.get(percentile))
        ttft_ok = ttft is not None and ttft <= ttft_ceiling
        tpot_ok = tpot is not None and tpot <= tpot_ceiling
        rows.append({
            "percentile": percentile,
            "ttft_ms": ttft,
            "ttft_ms_gate": ttft_ceiling,
            "ttft_holds": ttft_ok,
            "tpot_ms": tpot,
            "tpot_ms_gate": tpot_ceiling,
            "tpot_holds": tpot_ok,
            "measured": ttft is not None and tpot is not None,
        })
    return rows


def holds(checks: Iterable[dict]) -> bool:
    """Every bar in every row holds, by the aggregate percentiles."""
    rows = list(checks)
    return bool(rows) and all(row["ttft_holds"] and row["tpot_holds"]
                              for row in rows)


def why(checks: Iterable[dict]) -> str:
    """The misses, named one by one, in the order the rows are frozen.

    Every failed gate is reported SEPARATELY. A run that misses both ceilings
    at the same percentile is told both: an operator who reads only
    "p95 missed" cannot tell whether the system is slow to start or slow to
    produce, and those have opposite fixes.
    """
    parts = []
    for row in checks:
        p = row["percentile"]
        if not row["ttft_holds"]:
            if row["ttft_ms"] is None:
                parts.append(f"p{p} ttft not measured (gate "
                             f"{row['ttft_ms_gate']:.0f} ms)")
            else:
                parts.append(f"p{p} ttft {row['ttft_ms']:.0f} > "
                             f"{row['ttft_ms_gate']:.0f} ms")
        if not row["tpot_holds"]:
            if row["tpot_ms"] is None:
                parts.append(f"p{p} tpot not measured (gate "
                             f"{row['tpot_ms_gate']:.0f} ms)")
            else:
                parts.append(f"p{p} tpot {row['tpot_ms']:.1f} > "
                             f"{row['tpot_ms_gate']:.0f} ms")
    return "; ".join(parts)


def table(levels: Sequence[dict]) -> str:
    """The frozen bar as markdown, for a brief or a README.

    Rendered from the contract rather than typed beside it: a threshold table
    written by hand beside the one that is judged is a second bar, and the two
    drift on the first revision.
    """
    lines = ["| percentile | time to first token | time per output token |",
             "|---|---|---|"]
    for level in levels:
        lines.append(
            f"| **p{int(level['percentile'])}** "
            f"| <= **{float(level['ttft_ms']):.0f} ms** "
            f"| <= **{float(level['tpot_ms']):.0f} ms** |")
    return "\n".join(lines)
