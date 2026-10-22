"""Sends each of the 12 DT-sweep prompts (run_evaluation.py's output) to
Gemini 3.1 Pro and saves the generated hypothesis text.

This does NOT drop/ablate any prompt content -- unlike
gemini_3.1_pro_hypothesis/ablation.py (a different experiment testing which
prompt sections matter), this sends each of the 4 DT-setting prompts intact,
because the question here is "does the biological hypothesis change when the
DT's feature-count/depth changes", not "does removing a prompt section hurt".

Reuses the same google-genai client/model-id/system-instruction-splitting
convention as gemini_3.1_pro_hypothesis/LLMinterpret_gemini.py for consistency
with the rest of the project.

Usage:
    python generate_hypotheses.py [--runs N]
"""
import argparse
import json
import os
import re
import time

from google import genai
from google.genai import types

import config

API_KEY = os.environ["GEMINI_API_KEY"]
MODEL_ID = "gemini-3.1-pro-preview"
TEMPERATURE = 0.3
SLEEP_BETWEEN_CALLS = 4  # seconds, rate-limit safety margin (matches existing scripts)


def parse_prompt_file(content):
    """Same section split as LLMinterpret_gemini.py: system prompt is
    everything between '--- SYSTEM PROMPT ---' and the next '--- ' header;
    user content is everything from there on."""
    match = re.search(r'--- SYSTEM PROMPT ---\s*\n(.*?)\n--- ', content, re.DOTALL)
    if match:
        system_prompt = match.group(1).strip()
        end_of_sys_prompt = content.find(system_prompt) + len(system_prompt)
        user_content = content[end_of_sys_prompt:].strip()
    else:
        system_prompt = "You are an expert computational biologist tasked with interpreting decision tree classification rules."
        user_content = content
    return system_prompt, user_content


def main(n_runs):
    client = genai.Client(api_key=API_KEY)

    jobs = [
        (ds["key"], setting["name"])
        for ds in config.DATASETS
        for setting in config.DT_SETTINGS
    ]
    print(f"Found {len(jobs)} (dataset, setting) prompt files. {n_runs} run(s) each = {len(jobs) * n_runs} API calls.")

    for job_idx, (dataset_key, setting_name) in enumerate(jobs, 1):
        out_dir = config.RESULTS_DIR / dataset_key
        prompt_path = out_dir / f"{dataset_key}_{setting_name}_prompt.txt"
        hyp_path = out_dir / f"{dataset_key}_{setting_name}_hypothesis.jsonl"

        if not prompt_path.exists():
            print(f"[{job_idx}/{len(jobs)}] SKIP {dataset_key}/{setting_name}: prompt file not found (run run_evaluation.py first)")
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
                    if usage:
                        print(f"tokens: {usage.prompt_token_count} in / {usage.candidates_token_count} out")
                    else:
                        print("ok")
                except Exception as e:
                    print(f"ERROR: {e}")
                    response_text = f"Error processing request: {e}"

                outfile.write(json.dumps({
                    "dataset_key": dataset_key,
                    "setting_name": setting_name,
                    "run": run,
                    "final_answer": response_text,
                }) + "\n")
                outfile.flush()
                time.sleep(SLEEP_BETWEEN_CALLS)

        print(f"  Saved -> {hyp_path}")

    print("\nAll hypotheses generated.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=1, help="Number of Gemini runs per (dataset, setting).")
    args = parser.parse_args()
    main(args.runs)
