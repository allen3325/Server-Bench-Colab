"""Report and bound an AgentX warmup that stops making progress.

Elapsed time alone cannot distinguish a legitimate long-context prefill from a
hung request.  This watcher therefore treats an increase in ``returned`` as
progress, reports the counters after a period without progress, and eventually
ends a stalled warmup with explicit evidence instead of waiting silently.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

#: How long `returned` may stay unchanged before the warmup is called stalled.
#:
#: Long enough for an honest long-context prefill.  Diagnosis begins much
#: earlier; this deadline is only a backstop so a hung warmup cannot consume
#: the whole run.
STALL_S = 900.0

#: When to start saying that nothing is happening. Early enough that an agent
#: reading its own measurement's output can still do something about it.
SAY_SOMETHING_S = 120.0

#: How often to repeat that, once it is being said.
REPEAT_EVERY_S = 120.0

#: AgentX's own warmup progress line. Matched on the counters rather than the
#: surrounding text, which carries a source location that moves between builds.
PROGRESS = re.compile(
    r"returned=(?P<returned>\d+)/(?P<expected>\d+)\s*\|"
    r"\s*sent=(?P<sent>\d+)\s*\|"
    r"\s*in_flight=(?P<in_flight>\d+)\s*\|"
    r"\s*errors=(?P<errors>\d+)\s*\|"
    r"\s*elapsed=(?P<elapsed>[\d.]+)s")

#: Warmup is over once the timed replay starts, and after that this module has
#: no opinion. Either marker is enough; AgentX prints the first and the
#: progress bar carries the second.
PROFILING_STARTED = ("PROFILING execute", "Profiling:")


@dataclass
class WarmupWatch:
    """What the warmup has done, from the log AgentX writes as it goes."""

    returned: int = 0
    expected: int = 0
    sent: int = 0
    in_flight: int = 0
    errors: int = 0
    elapsed_s: float = 0.0
    #: Wall time at which `returned` last increased, or warmup began.
    #:
    #: None rather than 0.0 for "not yet set". With 0.0 the obvious idiom
    #: `self.progressed_at or now` reads a legitimate zero timestamp as
    #: absent and reports the stall as zero seconds forever -- invisible under
    #: `time.time()`, which is never zero, and immediate under a test clock
    #: that starts at the origin.
    progressed_at: float | None = None
    profiling: bool = False
    saw_progress_line: bool = False
    _said_at: float | None = field(default=None, repr=False)

    def feed(self, chunk: str, now: float) -> None:
        """Absorb newly written log text.

        `now` is wall time from the caller rather than read here, so a test
        can drive an hour of stall in a millisecond.
        """
        if any(marker in chunk for marker in PROFILING_STARTED):
            self.profiling = True
        matches = list(PROGRESS.finditer(chunk))
        if not matches:
            return
        if self.progressed_at is None:
            # Setup/dataset loading is not warmup. The first progress line
            # usually arrives 30 seconds into warmup, possibly after minutes
            # of setup. Anchor to AgentX's phase elapsed time, not the first
            # arbitrary chunk (which may even be empty). A read can contain
            # several buffered progress lines; the latest describes `now`.
            self.progressed_at = now - max(
                0.0, float(matches[-1].group("elapsed")))
        for match in matches:
            self.saw_progress_line = True
            returned = int(match.group("returned"))
            if returned > self.returned:
                self.returned = returned
                self.progressed_at = now
            self.expected = int(match.group("expected"))
            self.sent = int(match.group("sent"))
            self.in_flight = int(match.group("in_flight"))
            self.errors = int(match.group("errors"))
            self.elapsed_s = float(match.group("elapsed"))

    def stuck_for(self, now: float) -> float:
        """Seconds since `returned` last moved. Zero once profiling begins."""
        if self.profiling or not self.saw_progress_line:
            return 0.0
        if self.progressed_at is None:
            return 0.0
        return max(0.0, now - self.progressed_at)

    def stalled(self, now: float, *, stall_s: float | None = None) -> bool:
        """Has the warmup produced nothing for long enough to call it stuck?

        `stall_s` resolves at CALL time rather than in the signature. A
        default argument binds once, at import, so :data:`STALL_S` would be a
        constant that merely LOOKED adjustable -- and a caller that lowered it
        would watch a hung process run to the outer timeout while every line
        it printed said the stall had been detected.
        """
        limit = STALL_S if stall_s is None else stall_s
        return self.stuck_for(now) >= limit

    def should_say_something(self, now: float) -> bool:
        """Is it time to tell the reader that nothing is happening?

        True once at :data:`SAY_SOMETHING_S`, then every
        :data:`REPEAT_EVERY_S`. Calling this RECORDS that it was said, so a
        supervisor can drive it straight from its poll loop.
        """
        stuck = self.stuck_for(now)
        if stuck < SAY_SOMETHING_S:
            return False
        if self._said_at is not None and now - self._said_at < REPEAT_EVERY_S:
            return False
        self._said_at = now
        return True

    def describe(self, now: float) -> str:
        """One line naming what is stuck, and whose problem it is."""
        stuck = self.stuck_for(now)
        outstanding = self.in_flight or max(0, self.sent - self.returned)
        return (
            f"warmup stuck: {self.returned}/{self.expected} requests have "
            f"returned, {outstanding} still in flight, and nothing has come "
            f"back for {stuck:.0f}s ({self.elapsed_s:.0f}s into warmup). "
            f"The timed window has NOT started. These counters track completed "
            f"requests, not prefill or streamed-token progress. A slow "
            f"long-context request and a hung request can look identical here. "
            f"Check server progress or a thread dump before attributing a "
            f"deadlock; setup time is excluded from this clock.")

    def why_it_failed(self, now: float, *, stall_s: float | None = None) -> str:
        """The recorded reason, for a run that this watch ended."""
        stall_s = STALL_S if stall_s is None else stall_s
        return (
            f"warmup_stalled: {self.returned} of {self.expected} warmup "
            f"requests returned and then nothing moved for {self.stuck_for(now):.0f}s "
            f"(limit {stall_s:.0f}s), with {self.in_flight} still in flight "
            f"{self.elapsed_s:.0f}s into warmup. The timed window never "
            f"opened, so there is no measurement. The completion watchdog "
            f"expired; this alone does not establish a deadlock or incorrect "
            f"output. A slow long-context prefill may also exceed this limit. "
            f"Setup time is excluded; inspect server progress before retrying.")
