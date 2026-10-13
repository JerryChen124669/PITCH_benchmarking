"""
One-off inference run: interpret the decision-tree prompts in exp_prompts/
with the base (non-instruction-tuned) Ministral 3 14B model.

Prompt composition follows the same convention as bioLLM/scripts/inference.py:
each exp_prompts/*.txt file is split on its "--- SECTION ---" headers, the
first section is the system prompt, and the literal remainder (including its
own headers) is the user prompt. Ministral-3-14B-Base-2512 is a base/
completion-style model -- it ships no chat_template (unlike the
Reasoning-2512 checkpoint) -- so system + user are concatenated into one
continuous document and fed as a single completion prompt, the same
plain_concat handling bioLLM uses for PMC-LLaMA.

This is a single run per prompt (no repeated-seed variability experiment).

Usage:
    python run_inference.py
"""

import glob
import json
import os
import re
import time

import torch
from transformers import AutoTokenizer, Mistral3ForConditionalGeneration

MODEL_PATH = "/mnt/data/models/Ministral-3-14B-Base-2512"
PROMPTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "exp_prompts")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
SEED = 1234
MAX_NEW_TOKENS = 1536

SECTION_HEADER_RE = re.compile(r"^---\s*(.+?)\s*---\s*$")


def parse_prompt_file(path):
    with open(path, encoding="utf-8") as f:
        lines = f.readlines()

    header_idxs = [i for i, line in enumerate(lines) if SECTION_HEADER_RE.match(line.strip())]
    if len(header_idxs) < 2:
        raise ValueError(f"{path}: expected at least 2 '--- SECTION ---' headers, found {len(header_idxs)}")

    first, second = header_idxs[0], header_idxs[1]
    system_prompt = "".join(lines[first + 1:second]).strip()
    user_prompt = "".join(lines[second:]).strip()
    return system_prompt, user_prompt


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    prompt_paths = sorted(glob.glob(os.path.join(PROMPTS_DIR, "*.txt")))
    if not prompt_paths:
        raise SystemExit(f"No prompt files found in {PROMPTS_DIR}")
    print(f"[info] found {len(prompt_paths)} prompt files in {PROMPTS_DIR}")

    print(f"[info] loading base model from {MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = Mistral3ForConditionalGeneration.from_pretrained(MODEL_PATH, dtype=torch.bfloat16)
    model.to("cuda")
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    for prompt_path in prompt_paths:
        stem = os.path.splitext(os.path.basename(prompt_path))[0]
        out_path = os.path.join(RESULTS_DIR, f"{stem}.json")
        print(f"[info] {prompt_path} -> {out_path}")

        system_prompt, user_prompt = parse_prompt_file(prompt_path)
        rendered_prompt = f"{system_prompt}\n\n{user_prompt}"

        inputs = tokenizer(rendered_prompt, return_tensors="pt", add_special_tokens=True).to("cuda")
        prompt_len = inputs["input_ids"].shape[1]

        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)

        t0 = time.time()
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=1.0,
                top_p=0.9,
                repetition_penalty=1.05,
                no_repeat_ngram_size=3,
                pad_token_id=tokenizer.pad_token_id,
            )
        gen_time = round(time.time() - t0, 2)

        completion_ids = output_ids[0][prompt_len:]
        completion_text = tokenizer.decode(completion_ids, skip_special_tokens=True)

        record = {
            "prompt_file": os.path.basename(prompt_path),
            "model": "mistralai/Ministral-3-14B-Base-2512",
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "rendered_prompt": rendered_prompt,
            "prompt_token_length": prompt_len,
            "generation_settings": {
                "max_new_tokens": MAX_NEW_TOKENS,
                "do_sample": True,
                "temperature": 1.0,
                "top_p": 0.9,
                "repetition_penalty": 1.05,
                "no_repeat_ngram_size": 3,
                "seed": SEED,
            },
            "generation_time_sec": gen_time,
            "response": completion_text,
        }

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2, ensure_ascii=False)
        print(f"  done ({gen_time}s, {len(completion_ids)} tokens)")

    del model
    torch.cuda.empty_cache()
    print("[info] all prompts processed.")


if __name__ == "__main__":
    main()
