"""
Self-consistency selection: for each (model, prompt) result in trial_results/
with 2 or more repeated answers, ask an LLM judge to determine whether at
least two of them reach the same core biological conclusion. If so, keep the
more complete one of that agreeing pair. If none agree, pick one at random
(code-side, not LLM-side -- LLMs are unreliable at true randomness).

Saves one file per (model, prompt) in the SAME model subfolder:
    trial_results/<model_slug>/<prompt_stem>_selected_answer.json
        { "prompt_file", "model", "source_answer_index", "selection_method",
          "selection_notes", "answer" }

This intermediate file lets judge_evaluation.py (and any other downstream
analysis) read a single canonical answer per combo instead of re-doing this
comparison every time.

Usage:
    python select_best_answer.py                  # process every result in trial_results/
    python select_best_answer.py --models MODEL_SLUG [MODEL_SLUG ...]
    python select_best_answer.py --overwrite       # redo combos that already have a selection
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import string
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
TRIAL_RESULTS_DIR = SCRIPT_DIR / "trial_results"

# Cheap-ish, capable OpenAI model for the consistency check. Reuses the same
# judge model as judge_evaluation.py since the user only gave us one model id
# to work with; this call is much shorter than the full rubric grading call.
SELECTION_MODEL = "gpt-5.6-sol"

SELECTION_SYSTEM_PROMPT = """You compare several independently generated biological analyses that all responded to the exact same question (a decision-tree interpretation task). You are NOT grading quality here -- you are only checking whether at least two of them reach the SAME core conclusion.

"Same core conclusion" means: they identify the same primary zone/rule (or an equivalent one) as the most important, AND propose substantively the same biological mechanism to explain it -- not just similar vocabulary or shared gene names (which is expected since they answer the same prompt).

You will be shown the candidates, each labeled with a single letter, in random order. Determine:
1. Do at least two of them agree on the core conclusion as defined above?
2. If yes, which of that agreeing pair is more complete and better-supported (more directly tied to the tree's actual logic, more specific mechanism)?

Respond with ONLY a markdown JSON block in this exact format:
```json
{
  "consistent_pair_found": true,
  "agreeing_labels": ["X", "Y"],
  "preferred_label": "X",
  "reason": "one or two sentences"
}
```
If no two candidates agree on the core conclusion, respond:
```json
{
  "consistent_pair_found": false,
  "agreeing_labels": [],
  "preferred_label": null,
  "reason": "one or two sentences"
}
```
"""


def load_client():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit(
            "ERROR: Environment variable OPENAI_API_KEY is not set.\n"
            'Set it with: [System.Environment]::SetEnvironmentVariable('
            '"OPENAI_API_KEY", "your_api_key", "User")\n'
            "then restart your terminal."
        )
    from openai import OpenAI

    return OpenAI(api_key=api_key, timeout=300.0)


def select_one(client, answers: list[str]) -> dict:
    """Given 2 or more answer strings, return
    {"index": int, "method": str, "notes": str}."""
    n = len(answers)
    if n < 2:
        raise ValueError(f"Expected at least 2 answers, got {n}")
    if n > 26:
        raise ValueError(f"select_one only supports up to 26 answers (A-Z), got {n}")

    labels = list(string.ascii_uppercase[:n])
    order = list(range(n))
    random.shuffle(order)  # avoid positional bias
    label_to_index = {labels[i]: order[i] for i in range(n)}

    user_content = "\n\n".join(
        f"--- CANDIDATE {labels[i]} ---\n{answers[order[i]]}" for i in range(n)
    )

    response = client.chat.completions.create(
        model=SELECTION_MODEL,
        messages=[
            {"role": "system", "content": SELECTION_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
    )
    raw_text = response.choices[0].message.content or ""

    json_match = re.search(r"```json\s*(.*?)\s*```", raw_text, re.DOTALL)
    clean_json_str = json_match.group(1) if json_match else raw_text.strip().strip("`")
    parsed = json.loads(clean_json_str)

    if parsed.get("consistent_pair_found") and parsed.get("preferred_label") in label_to_index:
        chosen_index = label_to_index[parsed["preferred_label"]]
        method = "consistent_pair"
    else:
        # Prefer a non-empty candidate over a failed (None) generation when
        # picking at random -- no reason to hand downstream grading a blank
        # answer if a real one was on the table.
        non_empty = [i for i in range(n) if answers[i]]
        chosen_index = random.choice(non_empty or list(range(n)))
        method = "random_no_consensus"

    return {
        "index": chosen_index,
        "method": method,
        "notes": parsed.get("reason", ""),
    }


def process_file(client, result_path: Path, overwrite: bool) -> bool:
    """Returns True if a selection was written, False if skipped."""
    out_path = result_path.parent / f"{result_path.stem.replace('_result', '')}_selected_answer.json"

    if out_path.exists() and not overwrite:
        print(f"SKIP (already selected): {out_path}")
        return False

    data = json.loads(result_path.read_text(encoding="utf-8"))
    answers = data.get("answers", [])
    if len(answers) < 2:
        print(f"SKIP (has only {len(answers)} answer(s), need at least 2): {result_path}", file=sys.stderr)
        return False

    if not any(answers):
        # All raw generations came back empty (e.g. the model hit its
        # completion-token cap on hidden reasoning before writing an answer --
        # see official_trial.py's MODEL_MAX_TOKENS). Don't spend a judge call
        # comparing blanks; flag it so it's re-run at the generation
        # step instead of quietly scoring as a real (bad) answer downstream.
        print(
            f"WARNING: all {len(answers)} answers are empty for {data['model']} / {data['prompt_file']} -- "
            "writing a null selection instead of calling the judge. Fix generation and re-run "
            "with --overwrite.",
            file=sys.stderr,
        )
        out = {
            "prompt_file": data["prompt_file"],
            "model": data["model"],
            "source_answer_index": None,
            "selection_method": "all_answers_empty",
            "selection_notes": f"All {len(answers)} raw generations were empty/null; likely a generation failure upstream.",
            "answer": None,
        }
        out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"  -> all_answers_empty: {out_path}")
        return True

    print(f"Selecting: {data['model']} / {data['prompt_file']} ...", flush=True)

    try:
        selection = select_one(client, answers)
    except Exception as exc:  # noqa: BLE001 - surface the real error, no silent fallback
        print(f"ERROR: selection failed for {result_path}: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    out = {
        "prompt_file": data["prompt_file"],
        "model": data["model"],
        "source_answer_index": selection["index"],
        "selection_method": selection["method"],
        "selection_notes": selection["notes"],
        "answer": answers[selection["index"]],
    }
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"  -> {selection['method']} (index {selection['index']}): {out_path}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Subset of model subfolder names (e.g. 'zai-org__GLM-5.3') to process (default: all).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Redo combos that already have a selected-answer file (default: skip them).",
    )
    args = parser.parse_args()

    if not TRIAL_RESULTS_DIR.is_dir():
        raise SystemExit(f"ERROR: trial_results folder not found: {TRIAL_RESULTS_DIR}")

    client = load_client()

    model_dirs = sorted(d for d in TRIAL_RESULTS_DIR.iterdir() if d.is_dir())
    if args.models:
        wanted = set(args.models)
        model_dirs = [d for d in model_dirs if d.name in wanted]
        missing = wanted - {d.name for d in model_dirs}
        if missing:
            raise SystemExit(f"ERROR: unknown model subfolder(s): {missing}")

    if not model_dirs:
        raise SystemExit(f"ERROR: no model subfolders found under {TRIAL_RESULTS_DIR}")

    done = 0
    skipped = 0
    for model_dir in model_dirs:
        for result_path in sorted(model_dir.glob("*_result.json")):
            if process_file(client, result_path, args.overwrite):
                done += 1
            else:
                skipped += 1

    print(f"\nDone. {done} selection(s) written, {skipped} skipped.")


if __name__ == "__main__":
    main()
