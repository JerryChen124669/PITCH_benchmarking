"""
Run the five-response variability experiment: for every prompt file in
exp_prompts/, generate 5 independent responses from Gemma 4 12B, using
identical generation settings across the 5 runs (only the seed differs).

Usage:
    python run_experiment.py
    python run_experiment.py --prompt-file ../../exp_prompts/example.txt

Output: one JSON file per prompt at ../results/<prompt_stem>.json,
containing all 5 responses together (see README.md for the schema). Each
prompt's output is a separate file -- results are never merged together.
"""

import argparse
import glob
import json
import os
import time

from inference import build_inputs, generate_once, load_model, parse_prompt_file
from model_config import BASE_SEED, HF_REPO, NUM_REPEATS

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PROMPT_DIR = os.path.join(SCRIPT_DIR, "..", "..", "exp_prompts")
DEFAULT_RESULTS_DIR = os.path.join(SCRIPT_DIR, "..", "results")


def run_for_prompt(model, tokenizer, prompt_path, max_new_tokens=None):
    system_prompt, user_prompt = parse_prompt_file(prompt_path)
    input_ids, attention_mask = build_inputs(tokenizer, system_prompt, user_prompt)
    prompt_len = input_ids.shape[1]

    responses = []
    actual_gen_kwargs = None
    for i in range(NUM_REPEATS):
        seed = BASE_SEED + i
        t0 = time.time()
        completion, gen_kwargs, _ = generate_once(
            model, tokenizer, input_ids, attention_mask, seed=seed, max_new_tokens=max_new_tokens
        )
        actual_gen_kwargs = gen_kwargs
        responses.append(
            {
                "run_index": i,
                "seed": seed,
                "response": completion,
                "generation_time_sec": round(time.time() - t0, 2),
            }
        )
        print(f"    run {i + 1}/{NUM_REPEATS} done ({responses[-1]['generation_time_sec']}s)")

    return {
        "prompt_file": os.path.basename(prompt_path),
        "model_repo": HF_REPO,
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "prompt_token_length": prompt_len,
        "generation_settings": actual_gen_kwargs,
        "num_responses": NUM_REPEATS,
        "responses": responses,
    }


def main():
    parser = argparse.ArgumentParser(description="Run 5x generation over exp_prompts with Gemma 4 12B.")
    parser.add_argument("--prompt-file", default=None, help="Run a single prompt file instead of the whole folder.")
    parser.add_argument("--prompt-dir", default=DEFAULT_PROMPT_DIR)
    parser.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    args = parser.parse_args()

    if args.prompt_file:
        prompt_paths = [args.prompt_file]
    else:
        prompt_paths = sorted(glob.glob(os.path.join(args.prompt_dir, "*.txt")))
    if not prompt_paths:
        raise SystemExit(f"No prompt files found in {args.prompt_dir}")

    os.makedirs(args.results_dir, exist_ok=True)

    print("[info] loading gemma-4-12B-it ...")
    model, tokenizer = load_model()

    for prompt_path in prompt_paths:
        stem = os.path.splitext(os.path.basename(prompt_path))[0]
        out_path = os.path.join(args.results_dir, f"{stem}.json")
        print(f"[info] {prompt_path} -> {out_path}")
        record = run_for_prompt(model, tokenizer, prompt_path, args.max_new_tokens)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)

    print("[info] done.")


if __name__ == "__main__":
    main()
