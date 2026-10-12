"""
Second batch of Together AI models interpreting the 6 decision trees.

Differences from official_trial.py (the original batch):
  - New, separate TOGETHER_MODELS roster (name -> repo id) instead of
    MODEL_REASONING -- these are all new models, none overlap the first batch.
  - --repeats defaults to 1 (single generation per (model, prompt); no
    self-consistency selection needed downstream for these).
  - Reasoning effort is auto-probed per model instead of hardcoded: the first
    prompt call for a model tries reasoning_effort candidates from highest to
    lowest (REASONING_PRIORITY). If the API's 400 error names the accepted
    enum values, the highest of THOSE is used. If the field isn't accepted at
    all, the model is called with no reasoning_effort param (it may still
    reason by default). The resolved value is cached and reused for that
    model's remaining prompts (no re-probing).
  - Resilient to unrunnable models: if a model can't be served at all
    (no serverless endpoint -- 404/403, or a persistent error on the first
    call), this prints a warning and moves on to the next model instead of
    aborting the whole run.

Output layout (same schema as official_trial.py):
    trial_results/<model_slug>/<prompt_stem>_result.json
        { "prompt_file", "model", "reasoning_effort", "answers": [a1] }

Usage:
    python official_trial_batch2.py                 # all 7 models x all prompts, 1 repeat
    python official_trial_batch2.py --models "Qwen/Qwen3.7-Max"
    python official_trial_batch2.py --overwrite
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

from official_trial import (
    PROMPT_DIR,
    RESULTS_DIR,
    model_slug,
    parse_prompt_file,
)

TOGETHER_MODELS = {
    "Inkling": "thinkingmachines/Inkling",
    "Qwen3.7-Max": "Qwen/Qwen3.7-Max",
    "DeepSeek-V4.1-Flash": "deepseek-ai/DeepSeek-V4.1-Flash",
    "GLM-5.3-Flash": "zai-org/GLM-5.3-Flash",
    "Qwen3.8-Flash": "Qwen/Qwen3.8-Flash",
    "Llama-3.3-70B": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
    "Gemma-4-31B": "google/gemma-4-31B-it",
}

REPEATS_DEFAULT = 1

# Highest-to-lowest guess order for "use the highest thinking mode" -- probed
# live against the API since we don't know each new model's accepted enum
# values in advance (same situation official_trial.py's MODEL_REASONING was
# built from).
REASONING_PRIORITY = ["max", "xhigh", "high", "medium", "low"]

# At "max"/"xhigh" thinking, a model can burn the entire default 8192-token
# completion cap on hidden reasoning before writing any visible content,
# coming back with an empty (not None) message.content and
# finish_reason="length" (confirmed empirically for thinkingmachines/Inkling:
# 8778 reasoning tokens alone). Escalate max_tokens and retry rather than
# silently saving an empty answer -- same failure mode official_trial.py
# documents for Kimi-K3, but handled adaptively here since we don't know in
# advance which of these new models will hit it.
MAX_TOKENS_ESCALATION = [32000, 64000, 100000]

# Transient, worth-retrying-as-is errors -- server-side hiccups where the
# same request can simply succeed on a later attempt. Backoff in seconds
# between attempts (3 retries beyond the first try).
TRANSIENT_RETRY_DELAYS = [10, 30, 60]


def _is_transient(exc: Exception) -> bool:
    from together._exceptions import APIConnectionError, APITimeoutError, InternalServerError, RateLimitError

    if isinstance(exc, (InternalServerError, APIConnectionError, APITimeoutError, RateLimitError)):
        return True
    status = getattr(exc, "status_code", None)
    return status is not None and 500 <= status < 600


def _call_with_retries(client, kwargs: dict):
    """client.chat.completions.create with automatic retries on transient
    (5xx / connection / timeout / rate-limit) errors -- same request,
    unmodified, since these are server-side hiccups rather than something
    wrong with the request itself. Non-transient errors (400s, permission
    errors) are raised immediately with no retry."""
    last_exc = None
    for attempt, delay in enumerate([0] + TRANSIENT_RETRY_DELAYS):
        if delay:
            print(
                f"  NOTE [{kwargs['model']}]: transient error ({type(last_exc).__name__}: {_error_text(last_exc)}); "
                f"retrying in {delay}s (attempt {attempt + 1}/{len(TRANSIENT_RETRY_DELAYS) + 1})...",
                file=sys.stderr,
            )
            time.sleep(delay)
        try:
            return client.chat.completions.create(**kwargs)
        except Exception as exc:  # noqa: BLE001 - re-raised below if not transient/exhausted
            if not _is_transient(exc):
                raise
            last_exc = exc
    raise last_exc


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
    return any(
        phrase in text
        for phrase in ["no serverless", "not available", "no healthy", "dedicated endpoint", "does not exist"]
    )


def _extract_allowed_values(text: str) -> list[str]:
    """Best-effort parse of Together's 400 error body for an enum list, e.g.
    "...must be one of: low, medium, xhigh" or "...['low','medium','xhigh']"."""
    values = re.findall(r"['\"]([a-zA-Z]+)['\"]", text)
    known = {"low", "medium", "high", "xhigh", "max"}
    found = [v.lower() for v in values if v.lower() in known]
    if found:
        return found
    # fallback: bare words without quotes, e.g. "low, medium or xhigh"
    words = re.findall(r"[a-zA-Z]+", text.lower())
    return [w for w in words if w in known]


def _content_and_finish(response) -> tuple[str | None, str | None]:
    d = response.model_dump()
    choice = (d.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    return message.get("content"), choice.get("finish_reason")


def _create_with_escalation(client, kwargs: dict):
    """Calls chat.completions.create, and if the model came back with empty
    content because it ran out of completion tokens on hidden reasoning
    (finish_reason == "length"), retries with an escalating max_tokens cap
    until real content shows up or the escalation ladder is exhausted."""
    response = _call_with_retries(client, kwargs)
    content, finish_reason = _content_and_finish(response)

    if content or finish_reason != "length":
        return response

    for cap in MAX_TOKENS_ESCALATION:
        if kwargs.get("max_tokens") is not None and cap <= kwargs["max_tokens"]:
            continue
        print(
            f"  NOTE [{kwargs['model']}]: empty content at finish_reason='length' "
            f"(likely exhausted completion tokens on hidden reasoning); retrying with max_tokens={cap}.",
            file=sys.stderr,
        )
        retry_kwargs = {**kwargs, "max_tokens": cap}
        response = _call_with_retries(client, retry_kwargs)
        content, finish_reason = _content_and_finish(response)
        if content or finish_reason != "length":
            return response

    return response


def probe_reasoning_effort(client, model: str, messages: list[dict]) -> tuple[str | None, dict | None]:
    """Try REASONING_PRIORITY highest-to-lowest against a real call. Returns
    (resolved_reasoning_effort_or_None, response_or_None). Raises the
    underlying exception if the model itself can't be served at all."""
    tried: set[str] = set()
    candidate: str | None = REASONING_PRIORITY[0]

    while True:
        kwargs = {"model": model, "messages": messages}
        if candidate is not None:
            kwargs["reasoning_effort"] = candidate
        try:
            response = _create_with_escalation(client, kwargs)
            return candidate, response
        except Exception as exc:  # noqa: BLE001 - inspected below, re-raised if not recoverable
            if _looks_unservable(exc):
                raise

            from together.error import BadRequestError

            if not isinstance(exc, BadRequestError) or candidate is None:
                # Either a non-400 transient error, or we already tried with
                # no param at all -- nothing left to adapt, surface it.
                raise

            text = _error_text(exc)
            allowed = _extract_allowed_values(text)
            tried.add(candidate)

            if allowed:
                remaining = [v for v in REASONING_PRIORITY if v in allowed and v not in tried]
                if remaining:
                    candidate = remaining[0]
                    continue
                # exhausted the API's own accepted list -- give up on the param
                candidate = None
                continue

            # No enum list in the error -- assume the param itself is
            # unsupported for this model and drop it.
            rejected_value = candidate
            candidate = None
            print(
                f"  NOTE [{model}]: reasoning_effort='{rejected_value}' rejected "
                f"({text!r}); retrying without reasoning_effort.",
                file=sys.stderr,
            )
            continue


def run_one(client, model: str, prompt_path: Path, out_path: Path, cached_effort: dict) -> None:
    system_prompt, user_prompt = parse_prompt_file(prompt_path)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    if model not in cached_effort:
        reasoning_effort, response = probe_reasoning_effort(client, model, messages)
        cached_effort[model] = reasoning_effort
        print(f"  [{model}] resolved reasoning_effort = {reasoning_effort!r}")
    else:
        reasoning_effort = cached_effort[model]
        kwargs = {"model": model, "messages": messages}
        if reasoning_effort is not None:
            kwargs["reasoning_effort"] = reasoning_effort
        response = _create_with_escalation(client, kwargs)

    d = response.model_dump()
    choice = (d.get("choices") or [{}])[0]
    answer = choice.get("message", {}).get("content")
    if not answer:
        print(
            f"WARNING: model='{model}' file='{prompt_path.name}' returned empty/no content "
            f"(finish_reason={choice.get('finish_reason')!r}) even after max_tokens escalation; saving as-is.",
            file=sys.stderr,
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "prompt_file": prompt_path.name,
                "model": model,
                "reasoning_effort": reasoning_effort,
                "answers": [answer],
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--models", nargs="+", default=None,
        help="Subset of repo ids to run (default: all in TOGETHER_MODELS). Use the repo id, "
             f"e.g. {next(iter(TOGETHER_MODELS.values()))!r}.",
    )
    parser.add_argument("--repeats", type=int, default=REPEATS_DEFAULT, help=f"Default: {REPEATS_DEFAULT}.")
    parser.add_argument("--overwrite", action="store_true", help="Re-run combos that already have a saved result.")
    args = parser.parse_args()

    import os

    api_key = os.environ.get("TOGETHER_API_KEY")
    if not api_key:
        raise SystemExit("ERROR: Environment variable TOGETHER_API_KEY is not set.")

    if not PROMPT_DIR.is_dir():
        raise SystemExit(f"ERROR: Prompt folder not found: {PROMPT_DIR}")

    models = args.models if args.models else list(TOGETHER_MODELS.values())
    unknown = [m for m in models if m not in TOGETHER_MODELS.values()]
    if unknown:
        raise SystemExit(f"ERROR: Unknown model(s): {unknown}\nKnown: {list(TOGETHER_MODELS.values())}")

    prompt_files = sorted(PROMPT_DIR.glob("*.txt"))
    if not prompt_files:
        raise SystemExit(f"ERROR: No .txt files found in prompt folder: {PROMPT_DIR}")

    from together import Together

    client = Together(timeout=900.0, api_key=api_key)

    done = 0
    skipped = 0
    unservable: list[str] = []
    cached_effort: dict[str, str | None] = {}

    for model in models:
        model_dir = RESULTS_DIR / model_slug(model)  # created lazily on first successful save, so failed models leave no empty folder
        model_failed = False

        for prompt_path in prompt_files:
            out_path = model_dir / f"{prompt_path.stem}_result.json"

            if out_path.exists() and not args.overwrite:
                try:
                    existing_answers = len(json.loads(out_path.read_text(encoding="utf-8")).get("answers", []))
                except (json.JSONDecodeError, OSError):
                    existing_answers = 0
                if existing_answers >= args.repeats:
                    print(f"SKIP (already has {existing_answers}/{args.repeats} answer(s)): {out_path}")
                    skipped += 1
                    continue

            print(f"Running: [{model}] {prompt_path.name} ...", flush=True)
            try:
                run_one(client, model, prompt_path, out_path, cached_effort)
            except Exception as exc:  # noqa: BLE001 - decide skip-model vs skip-combo below
                if _looks_unservable(exc):
                    print(
                        f"WARNING: model '{model}' cannot be run (serverless endpoint unavailable): "
                        f"{_error_text(exc)}. Skipping this model entirely, continuing with the rest.",
                        file=sys.stderr,
                    )
                    model_failed = True
                    break
                print(
                    f"WARNING: request failed for model='{model}' file='{prompt_path.name}': "
                    f"{type(exc).__name__}: {_error_text(exc)}. Skipping this (model, prompt) combo.",
                    file=sys.stderr,
                )
                continue

            done += 1
            print(f"Saved: {out_path}")

        if model_failed:
            unservable.append(model)

    print(f"\nDone. {done} combo(s) run, {skipped} skipped (already existed).")
    if unservable:
        print(f"Could not run (no serverless endpoint / persistent error): {unservable}")


if __name__ == "__main__":
    main()
