"""Tiny in-memory sliding-window rate limiter (per client key, per process).

/ask spends paid LLM tokens, so an unthrottled endpoint lets anyone who can reach
the port burn the quota. This is deliberately simple: no Redis, no dependencies.
It is per-process (fine for one uvicorn worker, the documented setup) and keyed by
the client IP the server sees (behind a reverse proxy configure the proxy to limit
instead, or all users share one key).
"""
from __future__ import annotations

import math
import threading
import time
from collections import deque
from typing import Callable


class SlidingWindowLimiter:
    def __init__(
        self,
        limit: int,
        window_seconds: float = 60.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.limit = limit              # max requests per window; 0 disables
        self.window = window_seconds
        self._clock = clock
        self._hits: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def check(self, key: str) -> tuple[bool, int]:
        """Record a request. Returns (allowed, retry_after_seconds)."""
        if self.limit <= 0:
            return True, 0
        now = self._clock()
        cutoff = now - self.window
        with self._lock:
            # Drop clients with no recent activity so the dict can't grow forever.
            for k in [k for k, q in self._hits.items() if not q or q[-1] <= cutoff]:
                del self._hits[k]
            q = self._hits.setdefault(key, deque())
            while q and q[0] <= cutoff:
                q.popleft()
            if len(q) >= self.limit:
                return False, max(1, math.ceil(q[0] + self.window - now))
            q.append(now)
            return True, 0
