"""Cached memory-source selection, independent of public ps presentation."""

from copy import deepcopy
from dataclasses import asdict
import time


class MemoryTelemetry:
    """Own native observations; keep log/native measurements separate.

    A valid partial/unavailable native report remains authoritative in auto
    mode. Only a failed observation selects the log fallback. Receipt age is
    separate from the native snapshot's original age and scope.
    """

    def __init__(self, log, *, mode="auto", compare=False, clock=time.monotonic):
        self.log = log
        self.mode = mode
        self.compare = compare
        self._clock = clock
        self._native = None
        self._received = None
        self._reason = "not_observed"
        self._http_status = None
        self._support = "unknown"
        self._sequence = 0

    @property
    def needs_native(self):
        return (self.mode != "log" or self.compare) and self._support != "unsupported"

    @property
    def failure_reason(self):
        return self._reason

    def accept(self, report):
        sequence = report.snapshot()["snapshot_sequence"]
        if sequence < self._sequence:
            self.fail("snapshot_sequence_regressed", http_status=200)
            return
        self._sequence = sequence
        self._native = report
        self._received = self._clock()
        self._reason = None
        self._http_status = 200
        self._support = "supported"

    def fail(self, reason, *, http_status=None):
        # Do not keep an old known inventory selected after a failed read.
        self._native = None
        self._received = None
        self._reason = reason
        self._http_status = http_status
        if reason == "endpoint_not_supported":
            self._support = "unsupported"
        elif http_status == 200:
            self._support = "incompatible"

    def _native_snapshot(self):
        if self._native is None:
            return None
        elapsed = max(0, int((self._clock() - self._received) * 1000))
        return {"report": self._native.snapshot(), "totals": asdict(self._native.totals()),
                "observation_age_ms": elapsed}

    def snapshot(self):
        """Copy cached observations without HTTP, leases or lifecycle changes."""
        native = self._native_snapshot()
        log = self.log.snapshot()
        use_native = self.mode != "log" and native is not None
        use_log = self.mode == "log" or self.mode == "auto" and native is None
        result = {
            "mode": self.mode,
            "source": "native_memory_api" if use_native else "native_load_log" if use_log else None,
            "fallback_reason": self._reason if self.mode == "auto" and use_log else None,
            "native": {"support": self._support, "reason": self._reason, "http_status": self._http_status},
            "measurement": native if use_native else {"report": log} if use_log else None,
        }
        if self.compare:
            result["comparison"] = {"native": native, "log": {"report": log}}
        return deepcopy(result)
