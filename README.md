# PITCH: biological hypothesizing from decision-tree prompts

Code, prompts, generated answers, similarity & quality evaluation and evaluation results for the 24-model benchmark of hypotheses generated from decision-tree interpretation prompts.

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

## Repository layout
| Folder | Content |
|---|---|
| `1_generation_code/` | scripts and settings used to generate the hypotheses (24 models) |
| `2_prompts/` | the six empirical prompt files |
| `3_answers/` | the answer used for evaluation of 24 models for each of the six prompts |
| `4_evaluation_code/` | rubric judging, pairwise similarity, clustering and selection scripts; `reproduce_figures.sh` regenerates the five evaluation figures.|
| `5_evaluation_results/` | judged scores and pairwise similarity records for all six prompts, and the result tables and figures |
