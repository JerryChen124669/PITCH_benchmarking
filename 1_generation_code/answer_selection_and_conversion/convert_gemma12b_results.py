"""
One-off conversion: gemma12B/<prompt_stem>.json (externally generated, one
file per prompt, keys: model/answer) into the standard trial_results/ layout,
the same way convert_qwen9b_results.py handles the qwen jsonl exports --
exactly one generation per prompt, so select_best_answer.py's 3-way
self-consistency check is bypassed entirely.

Writes:
    trial_results/<model_slug>/<prompt_stem>_result.json
        {prompt_file, model, answers: [the single answer]} -- audit trail,
        matching the schema official_trial.py writes.
    trial_results/<model_slug>/<prompt_stem>_selected_answer.json
        {prompt_file, model, source_answer_index: 0,
         selection_method: "single_sample_no_consistency_check", answer}
        -- judge_evaluation.py and pairwise_similarity.py, which only read
        this file's "answer"/"model"/"prompt_file" fields, work unmodified.
"""

from __future__ import annotations

import json
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR / "gemma12B"
TRIAL_RESULTS_DIR = SCRIPT_DIR / "trial_results"


def convert_all(src_dir: Path) -> int:
    count = 0
    for src_path in sorted(src_dir.glob("*.json")):
        rec = json.loads(src_path.read_text(encoding="utf-8"))
        model_name = rec["model"]
        answer = rec["answer"]
        prompt_stem = src_path.stem

        out_dir = TRIAL_RESULTS_DIR / model_name
        out_dir.mkdir(parents=True, exist_ok=True)

        result_path = out_dir / f"{prompt_stem}_result.json"
        result_path.write_text(
            json.dumps(
                {"prompt_file": f"{prompt_stem}.txt", "model": model_name, "answers": [answer]},
                indent=2,
            ),
            encoding="utf-8",
        )

        selected_path = out_dir / f"{prompt_stem}_selected_answer.json"
        selected_path.write_text(
            json.dumps(
                {
                    "prompt_file": f"{prompt_stem}.txt",
                    "model": model_name,
                    "source_answer_index": 0,
                    "selection_method": "single_sample_no_consistency_check",
                    "selection_notes": "Only one generation was provided for this (model, prompt) combo -- no 3-way self-consistency check was possible or necessary.",
                    "answer": answer,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        count += 1
    return count


def main() -> None:
    if not SRC_DIR.is_dir():
        raise SystemExit(f"ERROR: not found: {SRC_DIR}")
    n = convert_all(SRC_DIR)
    print(f"gemma12B/*.json -> trial_results/<model_slug>/  ({n} prompts)")


if __name__ == "__main__":
    main()
