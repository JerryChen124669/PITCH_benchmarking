"""
One-off inference run: interpret the decision-tree prompts in exp_prompts/
with the instruction/reasoning-tuned Ministral-3-14B-Reasoning-2512
checkpoint (4-bit, unsloth/Ministral-3-14B-Reasoning-2512-bnb-4bit),
following the same system/user prompt composition as
bioLLM/scripts/inference.py.

Unlike Ministral-3-14B-Base-2512, this checkpoint ships a chat_template
with a native system role and a [THINK]...[/THINK] reasoning scratchpad, so
system_prompt/user_prompt are passed through tokenizer.apply_chat_template
as separate messages, and the model's [THINK] block is split out from the
final answer in the saved record.

This is a single run per prompt (no repeated-seed variability experiment).

Usage:
    python run_inference_reasoning.py
"""

import glob
import json
import os
import re
import time

import torch
from transformers import AutoTokenizer, Mistral3ForConditionalGeneration

MODEL_PATH = "unsloth/Ministral-3-14B-Reasoning-2512-bnb-4bit"
PROMPTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "exp_prompts")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
SEED = 1234
MAX_NEW_TOKENS = 4096

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


def split_reasoning(text):
    for marker in ("[/THINK]", "</think>"):
        if marker in text:
            reasoning, final_answer = text.split(marker, 1)
            return reasoning.strip(), final_answer.strip()
    return "", text.strip()


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    prompt_paths = sorted(glob.glob(os.path.join(PROMPTS_DIR, "*.txt")))
    if not prompt_paths:
        raise SystemExit(f"No prompt files found in {PROMPTS_DIR}")
    print(f"[info] found {len(prompt_paths)} prompt files in {PROMPTS_DIR}")

    print(f"[info] loading {MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = Mistral3ForConditionalGeneration.from_pretrained(MODEL_PATH, dtype=torch.bfloat16, device_map={"": 0})
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    for prompt_path in prompt_paths:
        stem = os.path.splitext(os.path.basename(prompt_path))[0]
        out_path = os.path.join(RESULTS_DIR, f"{stem}.json")
        print(f"[info] {prompt_path} -> {out_path}")

        system_prompt, user_prompt = parse_prompt_file(prompt_path)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to(model.device)
        prompt_len = inputs["input_ids"].shape[1]

        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)

        t0 = time.time()
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=0.3,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
            )
        gen_time = round(time.time() - t0, 2)

        completion_ids = output_ids[0][prompt_len:]
        completion_text = tokenizer.decode(completion_ids, skip_special_tokens=True)
        reasoning, final_answer = split_reasoning(completion_text)

        record = {
            "prompt_file": os.path.basename(prompt_path),
            "model": MODEL_PATH,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "prompt_token_length": prompt_len,
            "generation_settings": {
                "max_new_tokens": MAX_NEW_TOKENS,
                "do_sample": True,
                "temperature": 0.3,
                "top_p": 0.9,
                "seed": SEED,
            },
            "generation_time_sec": gen_time,
            "reasoning": reasoning,
            "response": final_answer,
        }

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2, ensure_ascii=False)
        print(f"  done ({gen_time}s, {len(completion_ids)} tokens)")

    del model
    torch.cuda.empty_cache()
    print("[info] all prompts processed.")


if __name__ == "__main__":
    main()
