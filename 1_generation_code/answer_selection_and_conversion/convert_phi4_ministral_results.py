"""
One-off conversion: phi4_14B_results/<prompt_stem>.json and
ministral3_14B_results/<prompt_stem>.json (externally generated locally, one
file per prompt, a single response each -- schema differs slightly between
the two: phi4's files embed their own system_prompt/user_prompt, ministral's
don't) into the standard trial_results/ layout, same convention as
convert_gemma12b_results.py / convert_qwen9b_results.py -- exactly one
generation per prompt, so select_best_answer.py's self-consistency check is
bypassed entirely.

Writes, per model:
    trial_results/<model_slug>/<prompt_stem>_result.json
        {prompt_file, model, answers: [the single answer]}
    trial_results/<model_slug>/<prompt_stem>_selected_answer.json
        {prompt_file, model, source_answer_index: 0,
         selection_method: "single_sample_no_consistency_check", answer}

model_slug here is a short display name (phi4-14B / ministral3-14B), NOT the
raw HF repo id from the source files' "model" field -- that field is only
ever used for our own bookkeeping in these two output files, never sent to
judge_evaluation.py or pairwise_similarity_v2.py's LLM calls (both scripts
only ever read "answer"/"prompt_file" from *_selected_answer.json).
"""

from __future__ import annotations

import json
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
TRIAL_RESULTS_DIR = SCRIPT_DIR / "trial_results"

SOURCE_DIRS = {
    "phi4_14B_results": "phi4-14B",
    "ministral3_14B_results": "ministral3-14B",
}


def convert_one(src_dir: Path, model_slug: str) -> int:
    out_dir = TRIAL_RESULTS_DIR / model_slug
    out_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for src_path in sorted(src_dir.glob("*.json")):
        rec = json.loads(src_path.read_text(encoding="utf-8"))
        prompt_file = rec["prompt_file"]
        prompt_stem = Path(prompt_file).stem
        answer = rec["response"]

        result_path = out_dir / f"{prompt_stem}_result.json"
        result_path.write_text(
            json.dumps(
                {"prompt_file": prompt_file, "model": model_slug, "answers": [answer]},
                indent=2,
            ),
            encoding="utf-8",
        )

        selected_path = out_dir / f"{prompt_stem}_selected_answer.json"
        selected_path.write_text(
            json.dumps(
                {
                    "prompt_file": prompt_file,
                    "model": model_slug,
                    "source_answer_index": 0,
                    "selection_method": "single_sample_no_consistency_check",
                    "selection_notes": "Only one generation was provided for this (model, prompt) combo -- no self-consistency check was possible or necessary.",
                    "answer": answer,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        count += 1
    return count


def main() -> None:
    for src_name, model_slug in SOURCE_DIRS.items():
        src_dir = SCRIPT_DIR / src_name
        if not src_dir.is_dir():
            raise SystemExit(f"ERROR: not found: {src_dir}")
        n = convert_one(src_dir, model_slug)
        print(f"{src_name}/*.json -> trial_results/{model_slug}/  ({n} prompts)")


if __name__ == "__main__":
    main()
