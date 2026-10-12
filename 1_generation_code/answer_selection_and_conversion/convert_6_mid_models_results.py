"""
One-off conversion: 6_mid_models/real_*.jsonl (externally generated, one line
per prompt, keys: key/id/thinking/answer) into the standard trial_results/
layout, one model per file, exactly one generation per prompt -- same
convention as convert_qwen9b_results.py.

Writes, per model:
    trial_results/<model_slug>/<prompt_stem>_result.json
        {prompt_file, model, answers: [the single answer]}
    trial_results/<model_slug>/<prompt_stem>_selected_answer.json
        {prompt_file, model, source_answer_index: 0,
         selection_method: "single_sample_no_consistency_check", answer}

The "thinking" trace is intentionally dropped (judges are only ever shown the
final answer), and model_slug is a display slug, never the HF repo id.
"""

from __future__ import annotations

import json
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR / "6_mid_models"
TRIAL_RESULTS_DIR = SCRIPT_DIR / "trial_results"

FILES = {
    "real_qwen38_27b.jsonl": "Qwen3.8-27B-think",
    "real_qwen36_35b_a3b.jsonl": "Qwen3.6-35B-A3B-think",
    "real_nemotron3_nano_30b_a3b.jsonl": "Nemotron3-Nano-30B-A3B",
    "real_gemma4_31b.jsonl": "gemma-4-31B-it",
    "real_mistral_small31_24b.jsonl": "Mistral-Small-3.1-24B",
    "real_qwen3next_80b_a3b.jsonl": "Qwen3-Next-80B-A3B-Instruct",
}


def convert_one(src_path: Path, model_slug: str) -> int:
    out_dir = TRIAL_RESULTS_DIR / model_slug
    out_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for line in src_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        prompt_stem = rec["key"]
        answer = rec["answer"]
        if not answer or not answer.strip():
            raise SystemExit(f"ERROR: empty answer in {src_path.name} for {prompt_stem}")

        (out_dir / f"{prompt_stem}_result.json").write_text(
            json.dumps({"prompt_file": f"{prompt_stem}.txt", "model": model_slug, "answers": [answer]}, indent=2),
            encoding="utf-8",
        )
        (out_dir / f"{prompt_stem}_selected_answer.json").write_text(
            json.dumps(
                {
                    "prompt_file": f"{prompt_stem}.txt",
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
    for fname, slug in FILES.items():
        src = SRC_DIR / fname
        if not src.is_file():
            raise SystemExit(f"ERROR: not found: {src}")
        print(f"{fname} -> trial_results/{slug}/  ({convert_one(src, slug)} prompts)")


if __name__ == "__main__":
    main()
