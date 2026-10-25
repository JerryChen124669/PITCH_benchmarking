"""Retry and max_tokens-escalation helpers for the Together AI calls of the ablation generation (extracted unchanged from the project's run_ablation_together.py; the rest of that script is not needed).

  * transient provider errors (5xx, connection, timeout, rate limit) are retried after TRANSIENT_RETRY_DELAYS seconds (run_repeats_v2.py sets 30, 60, 120, 240, 480);
  * a reasoning model that returns no text because it used its whole output limit on hidden reasoning is retried with a larger max_tokens (MAX_TOKENS_ESCALATION), if the cap allows.
"""
from __future__ import annotations

import sys
import time

MAX_TOKENS_ESCALATION = [32000, 64000, 100000]
TRANSIENT_RETRY_DELAYS = [10, 30, 60]


def _is_transient(exc: Exception) -> bool:
    from together._exceptions import APIConnectionError, APITimeoutError, InternalServerError, RateLimitError
    if isinstance(exc, (InternalServerError, APIConnectionError, APITimeoutError, RateLimitError)):
        return True
    status = getattr(exc, "status_code", None)
    return status is not None and 500 <= status < 600


def _error_text(exc: Exception) -> str:
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        msg = body.get("error", {}).get("message") if isinstance(body.get("error"), dict) else body.get("message")
        if msg:
            return str(msg)
    return str(getattr(exc, "message", exc))


def _looks_unservable(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None)
    text = _error_text(exc).lower()
    if status in (403, 404):
        return True
    return any(p in text for p in ["no serverless", "not available", "no healthy", "dedicated endpoint", "does not exist"])


def _looks_context_overflow(exc: Exception) -> bool:
    text = _error_text(exc).lower()
    return any(p in text for p in ["context length", "context_length", "maximum context", "too many tokens", "input is too long"])


def _call_with_retries(client, kwargs: dict):
    last_exc = None
    for attempt, delay in enumerate([0] + TRANSIENT_RETRY_DELAYS):
        if delay:
            print(f"  NOTE [{kwargs['model']}]: transient error ({type(last_exc).__name__}: {_error_text(last_exc)}); "
                  f"retrying in {delay}s (attempt {attempt + 1}/{len(TRANSIENT_RETRY_DELAYS) + 1})...", file=sys.stderr)
            time.sleep(delay)
        try:
            return client.chat.completions.create(**kwargs)
        except Exception as exc:  # noqa: BLE001
            if not _is_transient(exc):
                raise
            last_exc = exc
    raise last_exc


def _content_and_finish(response) -> tuple[str | None, str | None]:
    d = response.model_dump()
    choice = (d.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    return message.get("content"), choice.get("finish_reason")


def _create_with_escalation(client, kwargs: dict):
    response = _call_with_retries(client, kwargs)
    content, finish_reason = _content_and_finish(response)
    if content or finish_reason != "length":
        return response
    for cap in MAX_TOKENS_ESCALATION:
        if kwargs.get("max_tokens") is not None and cap <= kwargs["max_tokens"]:
            continue
        print(f"  NOTE [{kwargs['model']}]: empty content at finish_reason='length'; retrying with max_tokens={cap}.", file=sys.stderr)
        response = _call_with_retries(client, {**kwargs, "max_tokens": cap})
        content, finish_reason = _content_and_finish(response)
        if content or finish_reason != "length":
            return response
    return response


def _extract_allowed_values(text: str) -> list[str]:
    known = {"low", "medium", "high", "xhigh", "max"}
    found = [v.lower() for v in re.findall(r"['\"]([a-zA-Z]+)['\"]", text) if v.lower() in known]
    if found:
        return found
    return [w for w in re.findall(r"[a-zA-Z]+", text.lower()) if w in known]
