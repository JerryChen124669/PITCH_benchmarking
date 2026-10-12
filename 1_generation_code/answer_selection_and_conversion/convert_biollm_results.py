"""
One-off conversion: bioLLM/<model_key>/<prompt_stem>.jsonl (externally
generated, one line per prompt, 5 raw generations per line under
"responses": [{run_index, seed, response, generation_time_sec}, ...]) into
the standard trial_results/ layout, matching the schema official_trial.py
writes so select_best_answer.py's N-way self-consistency check can run
against these 5 candidates per (model, prompt) combo.

Writes:
    trial_results/<model_key>/<prompt_stem>_result.json
        {prompt_file, model, answers: [5 raw generations]}

select_best_answer.py is run separately afterwards to pick 1 of the 5 and
write the corresponding *_selected_answer.json.
"""

from __future__ import annotations

import json
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR / "bioLLM"
TRIAL_RESULTS_DIR = SCRIPT_DIR / "trial_results"

MODEL_DIRS = ["biomistral", "pmc_llama"]


def convert_one(src_dir: Path, model_name: str) -> int:
    out_dir = TRIAL_RESULTS_DIR / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for src_path in sorted(src_dir.glob("*.jsonl")):
        lines = [l for l in src_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        if len(lines) != 1:
            raise SystemExit(f"ERROR: expected exactly 1 JSON line in {src_path}, got {len(lines)}")
        rec = json.loads(lines[0])

        prompt_file = rec["prompt_file"]
        prompt_stem = Path(prompt_file).stem
        answers = [r["response"] for r in rec["responses"]]

        result_path = out_dir / f"{prompt_stem}_result.json"
        result_path.write_text(
            json.dumps(
                {"prompt_file": prompt_file, "model": model_name, "answers": answers},
                indent=2,
            ),
            encoding="utf-8",
        )
        count += 1
    return count


def main() -> None:
    for model_name in MODEL_DIRS:
        src_dir = SRC_DIR / model_name
        if not src_dir.is_dir():
            raise SystemExit(f"ERROR: not found: {src_dir}")
        n = convert_one(src_dir, model_name)
        print(f"bioLLM/{model_name}/*.jsonl -> trial_results/{model_name}/  ({n} prompts)")


if __name__ == "__main__":
    main()
