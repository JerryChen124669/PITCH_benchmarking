"""
For models with exactly ONE raw generation per prompt (e.g. official_trial_batch2.py
runs), write trial_results/<model_slug>/<prompt_stem>_selected_answer.json
directly from <prompt_stem>_result.json, the same schema the single-sample
converters (convert_gemma12b_results.py etc.) use. select_best_answer.py
needs >=2 answers, so it can't handle these.

Usage:
    python select_single_sample_answers.py --models MODEL_SLUG [MODEL_SLUG ...]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

TRIAL_RESULTS_DIR = Path(__file__).resolve().parent / "trial_results"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", required=True, help="trial_results subfolder names.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    for slug in args.models:
        model_dir = TRIAL_RESULTS_DIR / slug
        if not model_dir.is_dir():
            raise SystemExit(f"ERROR: not found: {model_dir}")
        written = 0
        for result_path in sorted(model_dir.glob("*_result.json")):
            data = json.loads(result_path.read_text(encoding="utf-8"))
            answers = data.get("answers", [])
            if len(answers) != 1 or not answers[0]:
                print(f"SKIP (need exactly 1 non-empty answer, got {len(answers)}): {result_path}")
                continue
            out_path = model_dir / f"{result_path.stem.replace('_result', '')}_selected_answer.json"
            if out_path.exists() and not args.overwrite:
                continue
            out_path.write_text(
                json.dumps(
                    {
                        "prompt_file": data["prompt_file"],
                        "model": data["model"],
                        "source_answer_index": 0,
                        "selection_method": "single_sample_no_consistency_check",
                        "selection_notes": "Only one generation was produced for this (model, prompt) combo -- no self-consistency check was possible or necessary.",
                        "answer": answers[0],
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            written += 1
        print(f"{slug}: {written} selected-answer file(s) written")


if __name__ == "__main__":
    main()
