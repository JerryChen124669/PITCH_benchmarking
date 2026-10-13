"""
Reusable local inference module for PMC-LLaMA 13B and BioMistral 7B.

Handles model loading and model-specific prompt formatting so that the same
biological reasoning prompt (system + user text) can be sent to either model
without modification. Inference only -- no training/fine-tuning.

CLI usage (quick smoke test on a single prompt file):

    python inference.py --model biomistral --prompt-file ../../exp_prompts/PTMA-Ki-67_Interpretation_Prompt.txt
    python inference.py --model pmc_llama  --prompt-file ../../exp_prompts/PTMA-Ki-67_Interpretation_Prompt.txt
"""

import argparse
import re
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer

from model_config import (
    GENERATION_DEFAULTS,
    MODEL_REGISTRY,
    RESPONSE_FORMAT_DIRECTIVE_CHAT,
    RESPONSE_FORMAT_DIRECTIVE_COMPLETION,
)

SECTION_HEADER_RE = re.compile(r"^---\s*(.+?)\s*---\s*$")


def parse_prompt_file(path):
    """Split an exp_prompts/*.txt file into (system_prompt, user_prompt).

    Files follow a fixed structure:
        --- SYSTEM PROMPT ---
        <system prompt text>
        --- <next section header> ---
        <everything else: experiment context, focused zones, decision
         tree, task instructions>

    The system prompt is the text of the first section. The user prompt is
    the literal remainder of the file (including its section headers, which
    are part of the task content, not chat scaffolding). Nothing is
    reworded or summarized.
    """
    with open(path, encoding="utf-8") as f:
        lines = f.readlines()

    header_idxs = [i for i, line in enumerate(lines) if SECTION_HEADER_RE.match(line.strip())]
    if len(header_idxs) < 2:
        raise ValueError(f"{path}: expected at least 2 '--- SECTION ---' headers, found {len(header_idxs)}")

    first, second = header_idxs[0], header_idxs[1]
    system_prompt = "".join(lines[first + 1:second]).strip()
    user_prompt = "".join(lines[second:]).strip()
    return system_prompt, user_prompt


def load_model(model_key):
    if model_key not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{model_key}'. Choices: {list(MODEL_REGISTRY)}")
    cfg = MODEL_REGISTRY[model_key]

    if cfg["architecture"] == "llama":
        # AutoTokenizer's fast-tokenizer auto-conversion mis-segments this
        # repo's raw sentencepiece tokenizer.model (verified: it splits
        # "Return" into "Ret"/"ur"/"nt" and similar nonsense), even though
        # tokenizer_config.json explicitly declares tokenizer_class =
        # LlamaTokenizer, use_fast = false. Loading the slow/sentencepiece
        # class directly avoids the broken conversion.
        tokenizer = LlamaTokenizer.from_pretrained(cfg["local_path"])
    else:
        tokenizer = AutoTokenizer.from_pretrained(cfg["local_path"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {"dtype": cfg["torch_dtype"]}
    if cfg["rope_parameters"] is not None:
        model_kwargs["rope_parameters"] = cfg["rope_parameters"]
    if cfg.get("max_position_embeddings_override") is not None:
        model_kwargs["max_position_embeddings"] = cfg["max_position_embeddings_override"]

    model = AutoModelForCausalLM.from_pretrained(cfg["local_path"], **model_kwargs)
    model.to("cuda")
    model.eval()
    if model.generation_config.pad_token_id is None or model.generation_config.pad_token_id < 0:
        model.generation_config.pad_token_id = tokenizer.pad_token_id

    return model, tokenizer, cfg


def build_inputs(model_key, tokenizer, system_prompt, user_prompt):
    """Render (system_prompt, user_prompt) into model input_ids + attention_mask,
    applying model-specific chat/prompt formatting only -- the biological
    content is passed through verbatim.

    A response-format directive is appended to the user turn here, at the
    model-input layer only -- it is never written into exp_prompts/*.txt.
    It is placed last (highest-recency instruction) to counter the observed
    tendency of both models to spend most of the output restating the
    zone/tree data already given instead of committing to one primary +
    one secondary zone as the task asks. Which variant depends on
    prompt_style: see RESPONSE_FORMAT_DIRECTIVE_CHAT/_COMPLETION in
    model_config.py for why they differ."""
    cfg = MODEL_REGISTRY[model_key]

    if cfg["prompt_style"] == "mistral_chat":
        # BioMistral's chat template only supports user/assistant roles
        # (no system role), so we fold the system prompt into the first
        # user turn -- the standard adaptation for Mistral-style models.
        user_prompt_for_model = f"{user_prompt}\n\n{RESPONSE_FORMAT_DIRECTIVE_CHAT}"
        combined = f"{system_prompt}\n\n{user_prompt_for_model}"
        messages = [{"role": "user", "content": combined}]
        encoded = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
        )
        input_ids = encoded["input_ids"]
        rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    elif cfg["prompt_style"] == "plain_concat":
        # PMC-LLaMA has no chat template; it is a completion-style
        # instruction-tuned model. Feed the system + user text as one
        # continuous document, matching its documented usage
        # (add_special_tokens=False in the model card's example). The
        # directive ends mid-sentence (see RESPONSE_FORMAT_DIRECTIVE_COMPLETION)
        # so generation is a natural continuation, not a command to obey.
        user_prompt_for_model = f"{user_prompt}\n\n{RESPONSE_FORMAT_DIRECTIVE_COMPLETION}"
        rendered = f"{system_prompt}\n\n{user_prompt_for_model}"
        input_ids = tokenizer(rendered, return_tensors="pt", add_special_tokens=False)["input_ids"]

    else:
        raise ValueError(f"Unknown prompt_style '{cfg['prompt_style']}'")

    # Single unpadded sequence per call, so the mask is simply all-ones.
    attention_mask = torch.ones_like(input_ids)
    return input_ids, attention_mask, rendered


def generate_once(model, tokenizer, input_ids, attention_mask, cfg, seed, max_new_tokens=None):
    """Run a single generation with a fixed seed. Returns the decoded
    completion text (prompt excluded) and the actual generation kwargs used."""
    input_ids = input_ids.to(model.device)
    attention_mask = attention_mask.to(model.device)
    prompt_len = input_ids.shape[1]

    budget = max_new_tokens or cfg["default_max_new_tokens"]
    if cfg.get("effective_context"):
        # Leave headroom so prompt + generation never exceeds the model's
        # (possibly RoPE-extended) context window.
        headroom = cfg["effective_context"] - prompt_len - 16
        budget = max(1, min(budget, headroom))

    gen_kwargs = dict(GENERATION_DEFAULTS)
    gen_kwargs["max_new_tokens"] = budget
    if cfg.get("min_new_tokens"):
        gen_kwargs["min_new_tokens"] = min(cfg["min_new_tokens"], budget)
    if cfg.get("repetition_penalty") is not None:
        gen_kwargs["repetition_penalty"] = cfg["repetition_penalty"]
    if cfg.get("no_repeat_ngram_size") is not None:
        gen_kwargs["no_repeat_ngram_size"] = cfg["no_repeat_ngram_size"]

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    with torch.no_grad():
        output_ids = model.generate(input_ids=input_ids, attention_mask=attention_mask, **gen_kwargs)

    completion_ids = output_ids[0][prompt_len:]
    completion_text = tokenizer.decode(completion_ids, skip_special_tokens=True)
    if cfg.get("response_prefix"):
        # The prefix (e.g. "PRIMARY ZONE:") was the tail of the prompt used
        # to prime a completion-style model -- re-attach it so the stored
        # response reads as a complete answer, not fabricated content.
        completion_text = cfg["response_prefix"] + completion_text
    return completion_text, gen_kwargs, prompt_len


def main():
    parser = argparse.ArgumentParser(description="Smoke-test a single prompt against a local model.")
    parser.add_argument("--model", required=True, choices=list(MODEL_REGISTRY))
    parser.add_argument("--prompt-file", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    system_prompt, user_prompt = parse_prompt_file(args.prompt_file)

    print(f"[info] loading {args.model} ...", file=sys.stderr)
    model, tokenizer, cfg = load_model(args.model)

    input_ids, attention_mask, rendered = build_inputs(args.model, tokenizer, system_prompt, user_prompt)
    print(f"[info] prompt length: {input_ids.shape[1]} tokens", file=sys.stderr)

    completion, gen_kwargs, prompt_len = generate_once(
        model, tokenizer, input_ids, attention_mask, cfg, seed=args.seed, max_new_tokens=args.max_new_tokens
    )
    print(f"[info] generation kwargs: {gen_kwargs}", file=sys.stderr)
    print("\n----- COMPLETION -----\n")
    print(completion)


if __name__ == "__main__":
    main()
