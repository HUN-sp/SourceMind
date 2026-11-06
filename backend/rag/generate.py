"""LLM wrapper — the ONLY place that talks to a model provider.

Everything else calls `generate(system, user)` and gets back text. Swapping
Groq for OpenAI, or for a local Ollama, is a change here and nowhere else.

We default to Groq's free tier (OpenAI-compatible API) serving an open-weight
model. temperature=0 for determinism, and JSON response mode so the caller gets
structured output it can trust instead of parsing prose.
"""
from __future__ import annotations

import logging
from functools import lru_cache

from .config import CONFIG

log = logging.getLogger(__name__)


class LLMError(RuntimeError):
    """The model call failed. The message is SAFE to show to API clients — raw
    provider text is logged server-side, never forwarded."""


class LLMConfigError(LLMError):
    """The server isn't configured for LLM calls (e.g. no API key)."""


class LLMAuthError(LLMError):
    """The provider rejected the server's API key."""

    def __init__(self, message: str = "The language-model provider rejected the server's API key."):
        super().__init__(message)


class LLMRateLimitError(LLMError):
    """Provider rate limit or daily token quota hit."""

    def __init__(self, message: str | None = None, retry_after: int | None = None):
        self.retry_after = retry_after
        wait = f" Try again in about {retry_after}s." if retry_after else " Please try again later."
        super().__init__(message or ("The language model is rate-limited or its daily token quota is used up." + wait))


class LLMModelError(LLMError):
    """The configured model doesn't exist / isn't available to this key (providers
    retire models). The fix is an operator change to LLM_MODEL, so name it."""

    def __init__(self, model: str):
        super().__init__(
            f"The configured language model '{model}' is not available from the provider. "
            "Set LLM_MODEL in .env to a model your key can use, then restart the server."
        )


class LLMUnavailableError(LLMError):
    """Provider unreachable, timed out, or returned a server error."""

    def __init__(self, message: str = "The language model is temporarily unavailable. Please try again."):
        super().__init__(message)


def llm_configured() -> bool:
    """True if a real-looking API key is set (not blank, not the .env.example
    placeholder). Never exposes the key itself."""
    key = CONFIG.groq_api_key
    return bool(key) and not key.startswith("gsk_your_")


@lru_cache(maxsize=1)
def _client():
    from groq import Groq

    if not llm_configured():
        raise LLMConfigError(
            "GROQ_API_KEY is not set. Copy .env.example to .env and add a free "
            "key from https://console.groq.com/keys"
        )
    return Groq(api_key=CONFIG.groq_api_key)


def _scrub(text: str) -> str:
    """Remove the API key from text before it is logged (defence in depth)."""
    key = CONFIG.groq_api_key
    return text.replace(key, "***") if key else text


def _translate(e: Exception) -> LLMError:
    """Map a provider/SDK exception to one of our safe errors; log the raw detail."""
    import groq

    log.warning("LLM call failed: %s: %s", type(e).__name__, _scrub(str(e)))
    if isinstance(e, groq.RateLimitError):
        retry = None
        try:
            retry = int(float(e.response.headers.get("retry-after", "")))
        except (ValueError, AttributeError):
            pass
        return LLMRateLimitError(retry_after=retry)
    if isinstance(e, groq.AuthenticationError):
        return LLMAuthError()
    if isinstance(e, groq.NotFoundError):
        return LLMModelError(CONFIG.llm_model)
    if isinstance(e, (groq.APIConnectionError, groq.APITimeoutError, groq.InternalServerError)):
        return LLMUnavailableError()
    return LLMError("The language-model call failed. Please try again.")


def generate(system: str, user: str, *, json_mode: bool = True) -> str:
    """Single call to the LLM. Returns the raw text content."""
    kwargs = {"response_format": {"type": "json_object"}} if json_mode else {}
    try:
        resp = _client().chat.completions.create(
            model=CONFIG.llm_model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0,
            seed=42,  # request reproducibility; LLM inference is still not 100% deterministic
            **kwargs,
        )
        return resp.choices[0].message.content or ""
    except LLMError:
        raise
    except Exception as e:  # noqa: BLE001 - surface any provider error uniformly
        raise _translate(e) from e
