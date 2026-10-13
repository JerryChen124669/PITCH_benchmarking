"""
Reusable local inference module for Gemma 4 12B (google/gemma-4-12B-it).

Inference only -- no training/fine-tuning. Text-only usage (the checkpoint
is a multimodal text+vision+audio "unified" model, but only the text path is
used here; AutoModelForCausalLM covers that without pulling in the image/
audio processors).

CLI usage (quick smoke test on a single prompt file):

    python inference.py --prompt-file ../../exp_prompts/example.txt
"""

import argparse
import re
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from model_config import GENERATION_DEFAULTS, MODEL_CONFIG, MODEL_PATH

SECTION_HEADER_RE = re.compile(r"^---\s*(.+?)\s*---\s*$")


def parse_prompt_file(path):
    """Split a prompt file into (system_prompt, user_prompt).

    If the file uses the '--- SECTION ---' header convention (>=2 headers),
    the first section is the system prompt and everything from the second
    header onward (headers included) is the user prompt. Otherwise the whole
    file is treated as the user prompt with no system prompt.
    """
    with open(path, encoding="utf-8") as f:
        lines = f.readlines()

    header_idxs = [i for i, line in enumerate(lines) if SECTION_HEADER_RE.match(line.strip())]
    if len(header_idxs) >= 2:
        first, second = header_idxs[0], header_idxs[1]
        system_prompt = "".join(lines[first + 1:second]).strip()
        user_prompt = "".join(lines[second:]).strip()
        return system_prompt, user_prompt

    return None, "".join(lines).strip()


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype=MODEL_CONFIG["torch_dtype"])
    model.to("cuda")
    model.eval()
    return model, tokenizer


def build_inputs(tokenizer, system_prompt, user_prompt):
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})

    encoded = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
    )
    return encoded["input_ids"], encoded["attention_mask"]


def generate_once(model, tokenizer, input_ids, attention_mask, seed, max_new_tokens=None):
    """Run a single generation with a fixed seed. Returns the decoded
    completion text (prompt excluded) and the actual generation kwargs used."""
    input_ids = input_ids.to(model.device)
    attention_mask = attention_mask.to(model.device)
    prompt_len = input_ids.shape[1]

    budget = max_new_tokens or MODEL_CONFIG["default_max_new_tokens"]
    headroom = MODEL_CONFIG["effective_context"] - prompt_len - 16
    budget = max(1, min(budget, headroom))

    gen_kwargs = dict(GENERATION_DEFAULTS)
    gen_kwargs["max_new_tokens"] = budget

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    with torch.no_grad():
        output_ids = model.generate(input_ids=input_ids, attention_mask=attention_mask, **gen_kwargs)

    completion_ids = output_ids[0][prompt_len:]
    completion_text = tokenizer.decode(completion_ids, skip_special_tokens=True)
    return completion_text, gen_kwargs, prompt_len


def main():
    parser = argparse.ArgumentParser(description="Smoke-test a single prompt against Gemma 4 12B.")
    parser.add_argument("--prompt-file", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    system_prompt, user_prompt = parse_prompt_file(args.prompt_file)

    print("[info] loading gemma-4-12B-it ...", file=sys.stderr)
    model, tokenizer = load_model()

    input_ids, attention_mask = build_inputs(tokenizer, system_prompt, user_prompt)
    print(f"[info] prompt length: {input_ids.shape[1]} tokens", file=sys.stderr)

    completion, gen_kwargs, prompt_len = generate_once(
        model, tokenizer, input_ids, attention_mask, seed=args.seed, max_new_tokens=args.max_new_tokens
    )
    print(f"[info] generation kwargs: {gen_kwargs}", file=sys.stderr)
    print("\n----- COMPLETION -----\n")
    print(completion)


if __name__ == "__main__":
    main()
