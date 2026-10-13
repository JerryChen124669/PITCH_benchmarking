# PITCH: biological hypothesizing from decision-tree prompts

Code, prompts and generated answers for the 24-model benchmark of hypotheses generated from decision-tree
(DT) interpretation prompts.

```
1_generation_code/
  api_models_together/        official_trial.py, official_trial_batch2.py   (11 models served by Together AI)
  api_model_gemini/           LLMinterpret_gemini.py                        (gemini-3.1-pro-preview, Google GenAI API)
  local_models_vllm/          vLLM runner, run scripts and run configs for Qwen3.8-27B, Qwen3.6-35B-A3B,
                              Nemotron 3 Nano 30B-A3B, Gemma 4 31B-it, Mistral Small 3.1 24B,
                              Qwen3-Next-80B-A3B-Instruct and the 9B Qwen run
  local_models_transformers/  Hugging Face transformers scripts for Gemma 4 12B-it, Phi-4 14B, Ministral 3 14B,
                              BioMistral-7B and PMC-LLaMA-13B, with their parameters (README.md, parameters.csv)
  answer_selection_and_conversion/   self-consistency selection and the converters that put externally
                              generated answers into the common layout
  model_settings/             llm_generation_settings.xlsx (one row per model: identifier, sizes, serving route,
                              samples per prompt, sampling and reasoning settings, precision, seed, dates, source file),
                              model registry and size list
2_prompts/                    the six prompt files (system prompt + experiment context + target zones + task)
3_answers/<model>/            <case>_selected_answer.json = the single answer per case that was passed on to evaluation
                              (24 models x 6 cases = 144 files; raw repeated generations are not included)
```

## What is and is not included
- One selected answer per model and case, for all six cases: Fxyd6, HUMAN (GNLY-CD45RA), PTMA (Ki-67), pERK (EGF 10 ng/mL),
  IL2RA-CCR5 and EGR1-pRPS6.
- 24 models are included. A temperature-0 variant of the 9B Qwen run is not part of the 24 and is not included.
- Models with several raw generations (7 Together models, Gemini, BioMistral, PMC-LLaMA) have one selected answer per
  prompt, chosen by an LLM self-consistency check (`select_best_answer.py`). Gemma 4 12B-it was sampled five times and the
  answer closest to the others in embedding space (`select_consistent.py`) is the one stored. The other models have one
  generation. Replicates are not averaged anywhere.
- Ministral 3 14B: the stored answers were generated with `unsloth/Ministral-3-14B-Reasoning-2512-bnb-4bit` (4-bit,
  temperature 0.3); the script in `local_models_transformers/ministral3_14B` runs the base model
  `Ministral-3-14B-Base-2512` (bf16, temperature 1.0) and did not produce them. See that folder's README.
- Settings that were not set explicitly (for example temperature on the Together models) are listed as such in the
  workbook; nothing there is inferred.

## Running the scripts
- API keys are read from environment variables and are not stored in the repository:
  `TOGETHER_API_KEY` (Together models), `GEMINI_API_KEY` (Gemini), `OPENAI_API_KEY` (answer selection).
- The scripts were run from the original project layout, where the prompts are in `../prompt` and the outputs go to
  `trial_results/`. To rerun from this repository, copy `2_prompts` to `../prompt` or edit `PROMPT_DIR` / `RESULTS_DIR`
  at the top of `official_trial.py`; `LLMinterpret_gemini.py` reads prompts from `data/` and writes `results/`.
- `local_models_vllm/README.md` documents the local vLLM runs (vLLM 0.22.1, bfloat16, seed 0, 4 x RTX A6000).
- `local_models_transformers/README.md` documents the transformers runs (one RTX PRO 6000 Blackwell, 96 GB).

## Repository layout
| Folder | Content |
|---|---|
| `1_generation_code/` | scripts and settings used to generate the hypotheses (24 models) |
| `2_prompts/` | the six prompt files |
| `3_answers/` | the selected answer of each of the 24 models for each of the six prompts |
| `4_evaluation_code/` | rubric judging, pairwise similarity, clustering and selection scripts; `reproduce_figures.sh` regenerates the five evaluation figures |
| `5_evaluation_results/` | judged scores and pairwise similarity records for all six prompts, and the result tables and figures |

See `4_evaluation_code/README.md` for how each figure is produced.