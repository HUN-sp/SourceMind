"""Quota protection for LIVE LLM calls (the Groq free tier: ~1,000 requests/day,
~12,000 tokens/minute).

This is separate from rag/ratelimit.py (a short per-minute anti-abuse throttle on
/ask). Here we count only requests that actually reach the LLM: answers served
from the preset file or the answer cache, and refusals from the similarity gate,
cost nothing and are not counted. That is why this hooks into the pipeline's
`generate` call rather than decorating the endpoint (a slowapi decorator would
also charge visitors for the free preset chips). It uses `limits`, the
rate-limiting engine slowapi is built on, with in-memory storage (one uvicorn
worker, the documented setup).

Two buckets: per client IP (default 15/hour) and global across all visitors
(default 800/day, under Groq's daily request cap).
"""
from __future__ import annotations

import math
import time

from limits import parse
from limits.storage import MemoryStorage
from limits.strategies import MovingWindowRateLimiter


class QuotaExceeded(Exception):
    def __init__(self, message: str, retry_after: int) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class LLMQuota:
    def __init__(self, per_ip: str = "15/hour", global_: str = "800/day") -> None:
        self._limiter = MovingWindowRateLimiter(MemoryStorage())
        self.per_ip = parse(per_ip) if per_ip else None
        self.global_ = parse(global_) if global_ else None

    def _retry_after(self, item, *ids: str) -> int:
        reset = self._limiter.get_window_stats(item, *ids).reset_time
        return max(1, math.ceil(reset - time.time()))

    def charge(self, client_ip: str) -> None:
        """Record one live LLM call for this IP, or raise QuotaExceeded (nothing recorded)."""
        if self.global_ and not self._limiter.test(self.global_, "global"):
            raise QuotaExceeded(
                "The demo's shared daily question budget is used up. The sample questions "
                "still work instantly; please try your own question tomorrow.",
                self._retry_after(self.global_, "global"),
            )
        if self.per_ip and not self._limiter.test(self.per_ip, "ip", client_ip):
            wait = self._retry_after(self.per_ip, "ip", client_ip)
            raise QuotaExceeded(
                f"To protect the free API quota, each visitor can ask "
                f"{self.per_ip.amount} live questions per {self.per_ip.GRANULARITY.name}. "
                f"Please try again in about {max(1, wait // 60)} min. "
                "The sample questions still work instantly.",
                wait,
            )
        if self.per_ip:
            self._limiter.hit(self.per_ip, "ip", client_ip)
        if self.global_:
            self._limiter.hit(self.global_, "global")

    def remaining(self, client_ip: str) -> int | None:
        if not self.per_ip:
            return None
        return self._limiter.get_window_stats(self.per_ip, "ip", client_ip).remaining
