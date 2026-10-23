"""
Export the raw JSON records in results/ as minimal
.json files: prompt_file, model, generation_settings, reasoning, and
response, all verbatim (only .strip()'d of leading/trailing whitespace) --
no other reprocessing, filtering, or added text.

Usage:
    python export_clean_reasoning_results.py
"""

import glob
import json
import os

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
CLEAN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_clean")


def main():
    os.makedirs(CLEAN_DIR, exist_ok=True)

    json_paths = sorted(glob.glob(os.path.join(RESULTS_DIR, "*.json")))
    if not json_paths:
        raise SystemExit(f"No result files found in {RESULTS_DIR}")

    for path in json_paths:
        with open(path, encoding="utf-8") as f:
            record = json.load(f)

        stem = os.path.splitext(os.path.basename(path))[0]
        out_path = os.path.join(CLEAN_DIR, f"{stem}.json")

        clean_record = {
            "prompt_file": record["prompt_file"],
            "model": record["model"],
            "generation_settings": record["generation_settings"],
            "reasoning": record["reasoning"].strip(),
            "response": record["response"].strip(),
        }

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(clean_record, f, indent=2, ensure_ascii=False)

        print(f"[info] {path} -> {out_path}")

    print(f"\n[info] {len(json_paths)} clean result files written to {os.path.abspath(CLEAN_DIR)}")


if __name__ == "__main__":
    main()
