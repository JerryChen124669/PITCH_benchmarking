"""Sends each of the 12 (dataset, model_type) prompts from run_evaluation.py
to Gemini 3.1 Pro and saves the generated hypothesis text. Same
client/model-id/system-instruction-splitting convention as
DTEvaluation/generate_hypotheses.py and gemini_3.1_pro_hypothesis/LLMinterpret_gemini.py.

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
    match = re.search(r'--- SYSTEM PROMPT ---\s*\n(.*?)\n--- ', content, re.DOTALL)
    if match:
        system_prompt = match.group(1).strip()
        end_of_sys_prompt = content.find(system_prompt) + len(system_prompt)
        user_content = content[end_of_sys_prompt:].strip()
    else:
        system_prompt = "You are an expert computational biologist tasked with interpreting classifier results."
        user_content = content
    return system_prompt, user_content


def main(n_runs):
    client = genai.Client(api_key=API_KEY)

    jobs = [
        (ds["key"], model["key"])
        for ds in config.DATASETS
        for model in config.MODEL_SETTINGS
    ]
    print(f"Found {len(jobs)} (dataset, model_type) prompt files. {n_runs} run(s) each = {len(jobs) * n_runs} API calls.")

    for job_idx, (dataset_key, model_key) in enumerate(jobs, 1):
        out_dir = config.RESULTS_DIR / dataset_key
        prompt_path = out_dir / f"{dataset_key}_{model_key}_prompt.txt"
        hyp_path = out_dir / f"{dataset_key}_{model_key}_hypothesis.jsonl"

        if not prompt_path.exists():
            print(f"[{job_idx}/{len(jobs)}] SKIP {dataset_key}/{model_key}: prompt file not found (run run_evaluation.py first)")
            continue

        print(f"\n[{job_idx}/{len(jobs)}] {dataset_key} / {model_key}")
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
                    "model_key": model_key,
                    "run": run,
                    "final_answer": response_text,
                }) + "\n")
                outfile.flush()
                time.sleep(SLEEP_BETWEEN_CALLS)

        print(f"  Saved -> {hyp_path}")

    print("\nAll hypotheses generated.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=1, help="Number of Gemini runs per (dataset, model_type).")
    args = parser.parse_args()
    main(args.runs)
