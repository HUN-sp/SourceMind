"""LLM wrapper — the ONLY place that talks to a model provider.

Everything else calls `generate(system, user)` and gets back text. Swapping
Groq for OpenAI, or for a local Ollama, is a change here and nowhere else.

We default to Groq's free tier (OpenAI-compatible API) serving an open-weight
model. temperature=0 for determinism, and JSON response mode so the caller gets
structured output it can trust instead of parsing prose.
"""
from __future__ import annotations

from functools import lru_cache

from .config import CONFIG


class LLMError(RuntimeError):
    """Raised when the model call fails (bad key, rate limit, network)."""


@lru_cache(maxsize=1)
def _client():
    from groq import Groq

    if not CONFIG.groq_api_key or CONFIG.groq_api_key.startswith("gsk_your_"):
        raise LLMError(
            "GROQ_API_KEY is not set. Copy .env.example to .env and add a free "
            "key from https://console.groq.com/keys"
        )
    return Groq(api_key=CONFIG.groq_api_key)


def generate(system: str, user: str, *, json_mode: bool = True) -> str:
    """Single call to the LLM. Returns the raw text content."""
    try:
        resp = _client().chat.completions.create(
            model=CONFIG.llm_model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0,
            seed=42,  # request reproducibility; LLM inference is still not 100% deterministic
            response_format={"type": "json_object"} if json_mode else None,
        )
        return resp.choices[0].message.content or ""
    except LLMError:
        raise
    except Exception as e:  # noqa: BLE001 - surface any provider error uniformly
        raise LLMError(f"LLM call failed: {e}") from e
