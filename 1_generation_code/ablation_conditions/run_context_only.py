"""Generates the CONTEXT-ONLY condition (cond4_context_only_softened, label "Context-only"): 4 LLMs x 3 datasets x 3 repeats = 36 calls, with the same call settings as every other condition.

It reuses run_repeats_v2.py unchanged (temperature 0.5 for all LLMs; Together reasoning effort Qwen xhigh, DeepSeek V4.1 Flash max; output cap 64,000 tokens, DeepSeek 130,000; Gemini default thinking;
same retry / max_tokens-escalation helpers in together_helpers.py). Differences: the prompt set (2_prompts/ablation_conditions/<dataset>/<dataset>_cond4_context_only_softened_prompt.txt: the context-only prompts without the honesty
instruction, asking for one specific mechanism), the system/user split (these prompts use the header `--- EXPERIMENT CONTEXT ---`; the user message starts at that header, as for the other conditions) and GLM-5.3:
its API no longer accepts reasoning_effort = xhigh (used for the other conditions), so GLM-5.3 uses max here (--glm-effort xhigh to try the old value); the value is stored in every answer record.
Output: 3_answers/ablation_conditions/all_attempts/<model_slug>/<dataset>_cond4_context_only_softened_run<k>.json (existing non-empty answers are skipped, so re-running does not overwrite them;
use --answers-dir for a fresh run).
Usage : python run_context_only.py --dry-run                       # plan only, no API calls
        python run_context_only.py --answers-dir ../../scratch_answers   # needs TOGETHER_API_KEY and GEMINI_API_KEY in the environment (never stored)
"""
import sys
from pathlib import Path

import run_repeats_v2 as rr

COND = "cond4_context_only_softened"
HEAD, USER_HDR = "--- SYSTEM PROMPT ---", "--- EXPERIMENT CONTEXT ---"


def split(text):
    assert text.lstrip().startswith(HEAD)
    after = text.lstrip()[len(HEAD):]
    i = after.find(USER_HDR)
    assert i != -1
    return after[:i].strip(), after[i:].strip()


rr._split = split
rr.CONDS = [COND]
rr.DATASETS = ["CITEseq_CD45RA_naive", "CosMx_CancerKi67", "Gouwens_GABAergic"]      # pERK is not part of the ablation analysis
rr.VARIANT_TAG = "context_only_softened_prompt"
REPO = Path(__file__).resolve().parents[2]
rr.PROMPTS = REPO / "2_prompts" / "ablation_conditions"
rr.ANSWERS = REPO / "3_answers" / "ablation_conditions" / "all_attempts"
rr.TOGETHER_EFFORT["zai-org/GLM-5.3"] = "max"

if __name__ == "__main__":
    # GLM-5.3's API now rejects reasoning_effort="xhigh" (used for the earlier conditions; "only low, high or max is allowed"): --glm-effort max uses the highest level it still accepts.
    # The value used is stored in every answer record (`reasoning_effort`).
    if "--glm-effort" in sys.argv:
        i = sys.argv.index("--glm-effort")
        rr.TOGETHER_EFFORT["zai-org/GLM-5.3"] = sys.argv[i + 1]
        del sys.argv[i:i + 2]
    if "--variant-tag" not in sys.argv:
        sys.argv += ["--variant-tag", "context_only_softened_prompt"]
    rr.main()
