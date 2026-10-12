"""
Official multi-model trial: runs every prompt in prompt/ through several
Together AI models, repeating each prompt 3x per model, and saves the raw
answers (no token/cost tracking -- see pilot_run.py for that).

Output layout:
    trial_results/<model_slug>/<prompt_stem>_result.json
        { "prompt_file", "model", "reasoning_effort", "answers": [a1, a2, a3] }

Usage:
    python official_trial.py                 # full matrix: all models x all prompts x 3 repeats
    python official_trial.py --models MODEL_ID [MODEL_ID ...]
    python official_trial.py --prompt-file NAME.txt
    python official_trial.py --repeats 1
    python official_trial.py --overwrite      # re-run combos that already have a saved result

Already-completed (model, prompt) combos are skipped by default (their JSON
already exists), so a failed/interrupted run can be safely re-launched
without re-paying for work already done.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROMPT_DIR = SCRIPT_DIR.parent / "prompt"
RESULTS_DIR = SCRIPT_DIR / "trial_results"

REPEATS_DEFAULT = 3

SYSTEM_HEADER = "--- SYSTEM PROMPT ---"
USER_START_HEADER = "--- EXPERIMENT CONTEXT & DATA PIPELINE ---"

# reasoning_effort value per model; None means "don't pass the parameter"
# (either the model doesn't support it, or it reasons by default regardless).
#
# Excluded for now -- not available as serverless/pay-per-token endpoints,
# each would require a separate dedicated (hourly-billed) endpoint:
#   - nvidia/nemotron-3-ultra-550b-a55b  (skipped per user decision)
#   - google/gemma-4-31B-it              (paused -- user is checking Together
#                                          dashboard/support; add back here
#                                          once resolved)
MODEL_REASONING: dict[str, str | None] = {
    "zai-org/GLM-5.3": "xhigh",  # "high" measured as ~low-effort for this model; xhigh/max are the real top tier
    "Qwen/Qwen3.8-2.4T-A95B": "xhigh",  # only low/medium/xhigh accepted; xhigh = highest
    "deepseek-ai/DeepSeek-V4-Pro-0813": "max",  # only high/max accepted; max = highest
    "moonshotai/Kimi-K3": None,  # reasoning_effort unsupported; reasons by default
    "MiniMaxAI/MiniMax-M3": None,  # reasoning_effort unsupported; reasons by default
    "deepseek-ai/DeepSeek-V4-Flash-0731": "max",  # only high/max accepted; max = highest
    "openai/gpt-oss-120b": "medium",  # explicit override, not highest
}

# Per-model completion token cap override; None means "let the API default
# (8192) apply". Kimi-K3 reasons by default with no way to cap/tune it via
# reasoning_effort, and on these long biology prompts its hidden
# reasoning_content alone can run past 8192 tokens, hitting finish_reason
# "length" with message.content == None (verified empirically: one prompt
# used 17.6k reasoning tokens + content before finishing at finish_reason
# "stop" under a 32000 cap). Without this override, Kimi-K3's answers are
# silently null instead of erroring, which is easy to miss downstream.
MODEL_MAX_TOKENS: dict[str, int | None] = {
    "moonshotai/Kimi-K3": 64000,
}


def parse_prompt_file(path: Path) -> tuple[str, str]:
    """Split a prompt .txt file into (system_prompt, user_prompt). Same
    parsing contract as pilot_run.py."""
    text = path.read_text(encoding="utf-8")

    if not text.lstrip().startswith(SYSTEM_HEADER):
        raise ValueError(
            f"Could not parse system-prompt section in '{path.name}': "
            f"file does not start with '{SYSTEM_HEADER}'."
        )

    after_header = text.lstrip()[len(SYSTEM_HEADER):]

    split_idx = after_header.find(USER_START_HEADER)
    if split_idx == -1:
        raise ValueError(
            f"Could not parse system-prompt section in '{path.name}': "
            f"marker '{USER_START_HEADER}' was not found."
        )

    system_prompt = after_header[:split_idx].strip()
    user_prompt = after_header[split_idx:].strip()

    if not system_prompt:
        raise ValueError(f"Parsed system prompt is empty in '{path.name}'.")
    if not user_prompt:
        raise ValueError(f"Parsed user prompt is empty in '{path.name}'.")

    return system_prompt, user_prompt


def model_slug(model: str) -> str:
    """Filesystem-safe subfolder name for a model id, e.g.
    'deepseek-ai/DeepSeek-V4-Flash-0731' -> 'deepseek-ai__DeepSeek-V4-Flash-0731'."""
    return model.replace("/", "__")


def run_one(client, model: str, reasoning_effort: str | None, prompt_path: Path, repeats: int, out_path: Path) -> dict:
    """Run a single prompt file through one model `repeats` times, saving
    progress to out_path after every individual repeat -- so a failure on a
    later repeat (e.g. a timeout) doesn't discard earlier, already-paid-for
    answers. Resumes from an existing partial result if out_path already has
    fewer than `repeats` answers saved."""
    system_prompt, user_prompt = parse_prompt_file(prompt_path)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    answers: list = []
    if out_path.exists():
        try:
            existing = json.loads(out_path.read_text(encoding="utf-8"))
            answers = existing.get("answers", [])
        except (json.JSONDecodeError, OSError):
            answers = []

    def save():
        out_path.write_text(
            json.dumps(
                {
                    "prompt_file": prompt_path.name,
                    "model": model,
                    "reasoning_effort": reasoning_effort,
                    "answers": answers,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    for i in range(len(answers) + 1, repeats + 1):
        print(f"  [{model}] {prompt_path.name} - run {i}/{repeats}...", flush=True)
        kwargs = {"model": model, "messages": messages}
        if reasoning_effort is not None:
            kwargs["reasoning_effort"] = reasoning_effort
        max_tokens = MODEL_MAX_TOKENS.get(model)
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens

        try:
            response = client.chat.completions.create(**kwargs)
        except Exception as exc:  # noqa: BLE001 - surface the real API error, no fallback
            print(
                f"ERROR: request failed for model='{model}' file='{prompt_path.name}' "
                f"run={i}: {type(exc).__name__}: {exc}\n"
                f"{len(answers)} of {repeats} repeat(s) already saved to {out_path} "
                "and will not be re-run.",
                file=sys.stderr,
            )
            raise SystemExit(1) from exc

        d = response.model_dump()
        choice = (d.get("choices") or [{}])[0]
        answer = choice.get("message", {}).get("content")
        if answer is None:
            print(
                f"WARNING: model='{model}' file='{prompt_path.name}' run={i} returned "
                f"no content (finish_reason={choice.get('finish_reason')!r}); "
                "likely ran out of completion tokens on hidden reasoning before "
                "writing the final answer. Saving None -- raise MODEL_MAX_TOKENS "
                "for this model and re-run with --overwrite.",
                file=sys.stderr,
            )
        answers.append(answer)
        save()

    return {
        "prompt_file": prompt_path.name,
        "model": model,
        "reasoning_effort": reasoning_effort,
        "answers": answers,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Subset of model ids to run (default: all models in MODEL_REASONING).",
    )
    parser.add_argument(
        "--prompt-file",
        type=str,
        default=None,
        help="Run only this prompt filename from prompt/ (default: all .txt files).",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=REPEATS_DEFAULT,
        help=f"Number of repeats per (model, prompt) pair (default: {REPEATS_DEFAULT}).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-run (model, prompt) combos that already have a saved result file (default: skip them).",
    )
    args = parser.parse_args()

    api_key = os.environ.get("TOGETHER_API_KEY")
    if not api_key:
        raise SystemExit(
            "ERROR: Environment variable TOGETHER_API_KEY is not set.\n"
            'Set it with: [System.Environment]::SetEnvironmentVariable('
            '"TOGETHER_API_KEY", "your_api_key", "User")\n'
            "then restart your terminal."
        )

    if not PROMPT_DIR.is_dir():
        raise SystemExit(f"ERROR: Prompt folder not found: {PROMPT_DIR}")

    models = args.models if args.models else list(MODEL_REASONING.keys())
    unknown = [m for m in models if m not in MODEL_REASONING]
    if unknown:
        raise SystemExit(
            f"ERROR: Unknown model(s) not in MODEL_REASONING config: {unknown}\n"
            f"Known models: {list(MODEL_REASONING.keys())}"
        )

    if args.prompt_file:
        prompt_files = [PROMPT_DIR / args.prompt_file]
        if not prompt_files[0].is_file():
            raise SystemExit(f"ERROR: Prompt file not found: {prompt_files[0]}")
    else:
        prompt_files = sorted(PROMPT_DIR.glob("*.txt"))
        if not prompt_files:
            raise SystemExit(f"ERROR: No .txt files found in prompt folder: {PROMPT_DIR}")

    from together import Together

    # Default SDK timeout (60s) is far too short for high-reasoning-effort
    # calls on large models; even 300s wasn't enough for Qwen3.8 at "xhigh".
    client = Together(timeout=900.0)

    total_combos = len(models) * len(prompt_files)
    done = 0
    skipped = 0

    for model in models:
        reasoning_effort = MODEL_REASONING[model]
        model_dir = RESULTS_DIR / model_slug(model)
        model_dir.mkdir(parents=True, exist_ok=True)

        for prompt_path in prompt_files:
            out_path = model_dir / f"{prompt_path.stem}_result.json"

            existing_answers = 0
            if out_path.exists():
                try:
                    existing_answers = len(json.loads(out_path.read_text(encoding="utf-8")).get("answers", []))
                except (json.JSONDecodeError, OSError):
                    existing_answers = 0

            if existing_answers >= args.repeats and not args.overwrite:
                print(f"SKIP (already has {existing_answers}/{args.repeats} answers): {out_path}")
                skipped += 1
                continue

            if args.overwrite and out_path.exists():
                out_path.unlink()

            run_one(client, model, reasoning_effort, prompt_path, args.repeats, out_path)
            done += 1
            print(f"Saved: {out_path}")

    print(f"\nDone. {done} combo(s) run, {skipped} skipped (already existed), out of {total_combos} total.")


if __name__ == "__main__":
    main()
