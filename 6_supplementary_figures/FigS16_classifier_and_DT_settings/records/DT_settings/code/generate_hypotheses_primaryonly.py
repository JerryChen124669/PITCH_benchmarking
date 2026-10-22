"""Same as generate_hypotheses.py but for the *_primaryonly_prompt.txt
files (regenerate_primary_only.py), saving to separate *_primaryonly_
hypothesis.jsonl files so nothing from the original run is touched.

Usage:
    python generate_hypotheses_primaryonly.py [--runs N]
"""
import argparse
import json
import time

import config
from generate_hypotheses import API_KEY, MODEL_ID, TEMPERATURE, SLEEP_BETWEEN_CALLS, parse_prompt_file
from google import genai
from google.genai import types


def main(n_runs):
    client = genai.Client(api_key=API_KEY)

    jobs = [(ds["key"], setting["name"]) for ds in config.DATASETS for setting in config.DT_SETTINGS]
    print(f"Found {len(jobs)} primary-only prompt files. {n_runs} run(s) each = {len(jobs) * n_runs} API calls.")

    for job_idx, (dataset_key, setting_name) in enumerate(jobs, 1):
        out_dir = config.RESULTS_DIR / dataset_key
        prompt_path = out_dir / f"{dataset_key}_{setting_name}_primaryonly_prompt.txt"
        hyp_path = out_dir / f"{dataset_key}_{setting_name}_primaryonly_hypothesis.jsonl"

        if not prompt_path.exists():
            print(f"[{job_idx}/{len(jobs)}] SKIP {dataset_key}/{setting_name}: prompt file not found")
            continue

        print(f"\n[{job_idx}/{len(jobs)}] {dataset_key} / {setting_name}")
        raw_content = prompt_path.read_text(encoding="utf-8")
        system_prompt, user_content = parse_prompt_file(raw_content)
        gen_config = types.GenerateContentConfig(system_instruction=system_prompt, temperature=TEMPERATURE)

        with open(hyp_path, "w", encoding="utf-8") as outfile:
            for run in range(1, n_runs + 1):
                print(f"  -> run {run}/{n_runs} via {MODEL_ID}...", end=" ", flush=True)
                try:
                    response = client.models.generate_content(model=MODEL_ID, contents=user_content, config=gen_config)
                    response_text = response.text
                    usage = response.usage_metadata
                    print(f"tokens: {usage.prompt_token_count} in / {usage.candidates_token_count} out" if usage else "ok")
                except Exception as e:
                    print(f"ERROR: {e}")
                    response_text = f"Error processing request: {e}"

                outfile.write(json.dumps({
                    "dataset_key": dataset_key, "setting_name": setting_name, "run": run, "final_answer": response_text,
                }) + "\n")
                outfile.flush()
                time.sleep(SLEEP_BETWEEN_CALLS)

        print(f"  Saved -> {hyp_path}")

    print("\nAll primary-only hypotheses generated.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=1)
    args = parser.parse_args()
    main(args.runs)
