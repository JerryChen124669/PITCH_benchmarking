"""
One-off conversion: qwen3.8_9B/real_qwen9b_think_temp{0,1}.jsonl (externally
generated, one line per prompt, keys: key/id/thinking/answer) into the
standard trial_results/ layout, treating each temperature as its own model
since there's exactly one generation per prompt -- no 3 repeats to run
select_best_answer.py's self-consistency check against.

Writes, per temperature variant:
    trial_results/<model_slug>/<prompt_stem>_result.json
        {prompt_file, model, answers: [the single answer]} -- audit trail,
        matching the schema official_trial.py writes (minus the "thinking"
        trace, dropped the same way official_trial.py only keeps
        message.content and discards reasoning).
    trial_results/<model_slug>/<prompt_stem>_selected_answer.json
        {prompt_file, model, source_answer_index: 0,
         selection_method: "single_sample_no_consistency_check", answer}
        -- select_best_answer.py is bypassed entirely (its algorithm requires
        exactly 3 answers); this file is written directly so
        judge_evaluation.py and pairwise_similarity.py, which only read this
        file's "answer"/"model"/"prompt_file" fields, work unmodified.
"""

from __future__ import annotations

import json
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR / "qwen3.8_9B"
TRIAL_RESULTS_DIR = SCRIPT_DIR / "trial_results"

FILES = {
    "real_qwen9b_think_temp0.jsonl": "Qwen3.8-9B-think-temp0",
    "real_qwen9b_think_temp1.jsonl": "Qwen3.8-9B-think-temp1",
}


def convert_one(src_path: Path, model_name: str) -> int:
    out_dir = TRIAL_RESULTS_DIR / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for line in src_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        prompt_stem = rec["key"]
        answer = rec["answer"]

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
    for fname, model_name in FILES.items():
        src_path = SRC_DIR / fname
        if not src_path.is_file():
            raise SystemExit(f"ERROR: not found: {src_path}")
        n = convert_one(src_path, model_name)
        print(f"{fname} -> trial_results/{model_name}/  ({n} prompts)")


if __name__ == "__main__":
    main()
