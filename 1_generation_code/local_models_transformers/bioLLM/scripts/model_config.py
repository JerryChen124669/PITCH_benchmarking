"""
Central configuration for local biomedical LLM inference.

Both models are run at their officially released full precision (no
quantization). See README.md for the disk/VRAM sizing that motivated these
choices.
"""

import torch

MODELS_DIR = "/mnt/data/models"

MODEL_REGISTRY = {
    "pmc_llama": {
        "hf_repo": "axiong/PMC_LLaMA_13B",
        "local_path": f"{MODELS_DIR}/PMC_LLaMA_13B",
        "architecture": "llama",
        # Native dtype as released (config.json: torch_dtype = float32).
        "torch_dtype": torch.float32,
        "prompt_style": "plain_concat",
        # Native max_position_embeddings is 2048, which is smaller than
        # several exp_prompts/*.txt files once tokenized. We extend the
        # context window via RoPE linear scaling so the full prompt text
        # survives intact. PMC-LLaMA was never trained or evaluated beyond
        # 2048 positions, so generation quality in the extended range is
        # unverified -- treat outputs as experimental and flag this in any
        # write-up. See README.md "Known limitations".
        "rope_parameters": {"rope_type": "linear", "factor": 2.0, "rope_theta": 10000.0},
        "max_position_embeddings_override": 4096,
        "effective_context": 4096,
        "default_max_new_tokens": 1024,
        # Kept low: forcing this completion-style model well past where it
        # wants to stop (we tried 400) made it spill into unrelated
        # memorized pretraining text (news articles, book reviews) once it
        # ran out of relevant conditioning. The completion-style priming
        # cue (RESPONSE_FORMAT_DIRECTIVE_COMPLETION, ending mid-sentence at
        # "PRIMARY ZONE:") does the real work of keeping it on-topic and
        # writing at length; this is just a floor against trivially short
        # output.
        "min_new_tokens": 50,
        "repetition_penalty": 1.05,
        "no_repeat_ngram_size": 3,
        # Re-attached to the decoded completion for readability: this text
        # was the tail of the prompt (a completion-style priming cue), not
        # part of exp_prompts/*.txt, so nothing is fabricated here.
        "response_prefix": "PRIMARY ZONE:",
    },
    "biomistral": {
        "hf_repo": "BioMistral/BioMistral-7B",
        "local_path": f"{MODELS_DIR}/BioMistral-7B",
        "architecture": "mistral",
        # Native dtype as released (config.json: torch_dtype = bfloat16).
        "torch_dtype": torch.bfloat16,
        "prompt_style": "mistral_chat",
        "rope_parameters": None,
        "max_position_embeddings_override": None,
        "effective_context": 32768,
        "default_max_new_tokens": 1024,
        "min_new_tokens": 400,
        # Added after observing BioMistral fall into literal repetition
        # loops on some of the 5 seeds for a given prompt (e.g. re-emitting
        # "the regulation of neuronal excitability by X, Y, Z" with one more
        # gene appended each time, dozens of times in a row).
        "repetition_penalty": 1.15,
        "no_repeat_ngram_size": 4,
        "response_prefix": None,
    },
}

# Generation settings held constant across the 5 repeated runs per prompt,
# per model, and shared between the two models. repetition_penalty and
# no_repeat_ngram_size are NOT here -- they are set per-model above, because
# the same value that fixed BioMistral's repetition loops caused PMC-LLaMA
# to derail into unrelated memorized web text (a Reuters Health article)
# instead of the biological task. Only the random seed differs between the
# 5 runs of a given model.
GENERATION_DEFAULTS = {
    "do_sample": True,
    "temperature": 0.7,
    "top_p": 0.9,
    "num_return_sequences": 1,
}

# Appended to the user prompt at the model-input layer only (never written
# into exp_prompts/*.txt). Both models were observed to spend most of their
# output restating the zone definitions, precision/TP/FP numbers, or the
# raw decision tree already present in the input, rather than committing to
# the single primary + secondary zone selection the task already asks for.
# This reinforces that instruction with an explicit "don't restate the
# input" rule and a fixed answer skeleton, placed last so it has the
# strongest effect on what gets generated next.
#
# Two variants, not one -- BioMistral is chat/instruction-tuned and follows
# a closed "output ONLY this structure" directive well. PMC-LLaMA is a
# completion-style model: the same closed directive made it respond
# "---END INSTRUCTIONS---" and stop almost immediately (verified via direct
# ablation), because a "---"-delimited all-caps instructions block isn't
# something it recognizes as a thing to obey, only as a document boundary
# to close out. The _COMPLETION variant instead ends the prompt mid-sentence
# at "PRIMARY ZONE:", priming it the way base/completion models actually
# work: predict what comes next, rather than follow a standalone command.
RESPONSE_FORMAT_DIRECTIVE_CHAT = """--- OUTPUT INSTRUCTIONS ---
Do not restate, copy, paraphrase, or re-list the decision tree, the focused/target zone definitions, their logic rules, or their Precision/TP/FP numbers -- the reader already has all of that. Do not describe the classifier's training setup or overall accuracy. Your response is incomplete unless it fully covers BOTH the PRIMARY ZONE and SECONDARY ZONE sections below -- never stop after the primary zone alone. Output ONLY your biological analysis, committing to exactly one primary zone and one secondary zone (not a listing of every zone), using exactly this structure and nothing else:

PRIMARY ZONE: <node number>
MECHANISTIC RATIONALE: <3-6 sentences of causal biological reasoning connecting this zone's features to the target response, grounded in known pathways>
LITERATURE-CONFIRMED FINDINGS: <bullet list, each with supporting publication(s)>
NEW TESTABLE MECHANISMS: <bullet list of novel hypotheses, each with supporting literature and a suggested experiment>

SECONDARY ZONE: <node number, biologically distinct from the primary zone>
MECHANISTIC RATIONALE: <3-6 sentences>
LITERATURE-CONFIRMED FINDINGS: <bullet list with publications>
NEW TESTABLE MECHANISMS: <bullet list with supporting literature and a suggested experiment>"""

RESPONSE_FORMAT_DIRECTIVE_COMPLETION = """Do not repeat, copy, or re-list the decision tree, the focused/target zone definitions, their logic rules, or their Precision/TP/FP numbers above -- go straight into the analysis below. Select exactly one primary zone and one biologically distinct secondary zone (not a listing of every zone). For each zone, give the causal biological mechanism connecting its features to the target response, the findings that confirm known literature (with the supporting publication), and new testable mechanisms supported by literature.

--- RESULTS & DISCUSSION ---
PRIMARY ZONE:"""

NUM_REPEATS = 5
BASE_SEED = 1234
