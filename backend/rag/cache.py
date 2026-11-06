"""Tiny thread-safe LRU for answered questions.

The Groq free tier caps daily tokens, and every /ask sends several passages as
context, so re-asking the same question (example chips, retries, page reloads)
should not spend tokens again. Only successful answers are stored — never refusals,
which may be a transient LLM hiccup — and entries are keyed on the index
generation so re-indexing invalidates them.
"""
from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Any


class AnswerCache:
    def __init__(self, maxsize: int = 128) -> None:
        self.maxsize = maxsize          # 0 disables caching
        self._data: OrderedDict[Any, Any] = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    @staticmethod
    def key(question: str, doc: str | None, generation: int) -> tuple:
        return (" ".join(question.split()).casefold(), doc, generation)

    def get(self, key):
        with self._lock:
            if key in self._data:
                self._data.move_to_end(key)
                self.hits += 1
                return self._data[key]
            self.misses += 1
            return None

    def put(self, key, value) -> None:
        if self.maxsize <= 0:
            return
        with self._lock:
            self._data[key] = value
            self._data.move_to_end(key)
            while len(self._data) > self.maxsize:
                self._data.popitem(last=False)

    def __len__(self) -> int:
        return len(self._data)

    def stats(self) -> dict:
        return {"entries": len(self), "hits": self.hits, "misses": self.misses, "max_entries": self.maxsize}
