"""
Run the five-response variability experiment: for every prompt file in
exp_prompts/, generate 5 independent responses from one model, using
identical generation settings across the 5 runs (only the seed differs).

Usage:
    python run_experiment.py --model biomistral
    python run_experiment.py --model pmc_llama
    python run_experiment.py --model biomistral --prompt-file ../../exp_prompts/PTMA-Ki-67_Interpretation_Prompt.txt

Output: one JSONL file per model at
    ../results/<model>/<prompt_stem>.jsonl
Each line is one prompt's record containing all 5 responses together
(see README.md for the schema).
"""

import argparse
import glob
import json
import os
import time

from inference import build_inputs, generate_once, load_model, parse_prompt_file
from model_config import (
    BASE_SEED,
    MODEL_REGISTRY,
    NUM_REPEATS,
    RESPONSE_FORMAT_DIRECTIVE_CHAT,
    RESPONSE_FORMAT_DIRECTIVE_COMPLETION,
)

RESPONSE_FORMAT_DIRECTIVE_BY_STYLE = {
    "mistral_chat": RESPONSE_FORMAT_DIRECTIVE_CHAT,
    "plain_concat": RESPONSE_FORMAT_DIRECTIVE_COMPLETION,
}

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PROMPT_DIR = os.path.join(SCRIPT_DIR, "..", "..", "exp_prompts")
DEFAULT_RESULTS_DIR = os.path.join(SCRIPT_DIR, "..", "results")


def run_for_prompt(model_key, model, tokenizer, cfg, prompt_path, max_new_tokens=None):
    system_prompt, user_prompt = parse_prompt_file(prompt_path)
    input_ids, attention_mask, rendered_prompt = build_inputs(model_key, tokenizer, system_prompt, user_prompt)
    prompt_len = input_ids.shape[1]

    responses = []
    actual_gen_kwargs = None
    for i in range(NUM_REPEATS):
        seed = BASE_SEED + i
        t0 = time.time()
        completion, gen_kwargs, _ = generate_once(
            model, tokenizer, input_ids, attention_mask, cfg, seed=seed, max_new_tokens=max_new_tokens
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

    record = {
        "prompt_file": os.path.basename(prompt_path),
        "model_key": model_key,
        "model_repo": MODEL_REGISTRY[model_key]["hf_repo"],
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        # Appended to user_prompt at the model-input layer only -- never
        # written into exp_prompts/*.txt. See model_config.py for rationale.
        "response_format_directive": RESPONSE_FORMAT_DIRECTIVE_BY_STYLE[cfg["prompt_style"]],
        "prompt_token_length": prompt_len,
        "generation_settings": actual_gen_kwargs,
        "num_responses": NUM_REPEATS,
        "responses": responses,
    }
    return record


def main():
    parser = argparse.ArgumentParser(description="Run 5x generation over exp_prompts for one model.")
    parser.add_argument("--model", required=True, choices=list(MODEL_REGISTRY))
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

    out_dir = os.path.join(args.results_dir, args.model)
    os.makedirs(out_dir, exist_ok=True)

    print(f"[info] loading {args.model} ...")
    model, tokenizer, cfg = load_model(args.model)

    for prompt_path in prompt_paths:
        stem = os.path.splitext(os.path.basename(prompt_path))[0]
        out_path = os.path.join(out_dir, f"{stem}.jsonl")
        print(f"[info] {prompt_path} -> {out_path}")
        record = run_for_prompt(args.model, model, tokenizer, cfg, prompt_path, args.max_new_tokens)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print("[info] done.")


if __name__ == "__main__":
    main()
