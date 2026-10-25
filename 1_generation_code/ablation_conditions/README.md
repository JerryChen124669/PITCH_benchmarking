# Generation code: ablation of the evidence conditions

Runner of the ablation answers (4 LLMs x 3 datasets x 3 conditions + the Context-only condition), the exact settings and how the Context-only answers were generated.

| File | Purpose |
|---|---|
| `run_repeats_v2.py` | calls the four LLMs (Gemini-3.1-pro-preview through the Google API; GLM-5.3, DeepSeek V4.1 Flash and Qwen3.8-2.4T-A95B through Together AI) for every prompt and repeat and writes one JSON record per call (`final_answer`, `finish_reason`, `usage`, `temperature`, `reasoning_effort`, `prompt_variant`). Resumable (existing non-empty answers are skipped). |
| `together_helpers.py` | retry of transient provider errors and max_tokens escalation when a reasoning model returns no text at its output limit (used by `run_repeats_v2.py`). |
| `run_context_only.py` | **Context-only condition** (`cond4_context_only_softened`): wraps `run_repeats_v2.py` with the context-only prompts, the `--- EXPERIMENT CONTEXT ---` system/user split and GLM-5.3 at reasoning effort `max`. |

**Call settings (all conditions):** temperature 0.5; Gemini with its default thinking configuration; Together reasoning effort Qwen xhigh and DeepSeek V4.1 Flash max; output cap 64,000 tokens (DeepSeek 130,000); 3 repeats per prompt and LLM.
**Exception:** the Together API no longer accepts `reasoning_effort = xhigh` for GLM-5.3 ("only low, high or max is allowed"), so the Context-only answers of GLM-5.3 were generated with `max`; the value is stored in every answer record. The GLM-5.3 answers of PITCH, MLP + IG and measurement-only used `xhigh`.

```
python run_context_only.py --dry-run                              # plan only (36 calls), no API call
python run_context_only.py --answers-dir <empty folder>           # needs TOGETHER_API_KEY and GEMINI_API_KEY in the environment
```
API keys are read from the environment only and are never stored. By default the script reads the prompts from `../../2_prompts/ablation_conditions/` and writes to `../../3_answers/ablation_conditions/all_attempts/`
(it skips answers that already exist, so use `--answers-dir` for a new run). Provider-served generations are stochastic and are not expected to be reproduced verbatim.
For the other three conditions use `run_repeats_v2.py --prompts-dir ../../2_prompts/ablation_conditions --conds cond0_full_pipeline cond1_ig_only cond2_raw_data --datasets CITEseq_CD45RA_naive CosMx_CancerKi67 Gouwens_GABAergic --answers-dir <folder>`.
