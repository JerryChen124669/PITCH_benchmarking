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
| `6_supplementary_figures/` | data tables, plotting scripts and records behind Supplementary Figures S15 (ablation) and S16 (classifier choice, decision-tree settings) |

## Ablation of the evidence conditions (generation rate, five-criteria radar)
Beyond the 24-model benchmark, the folders `ablation_conditions/` inside `2_prompts/`, `3_answers/`, `4_evaluation_code/` and `5_evaluation_results/` hold an ablation of what the LLM is given (PITCH decision-tree output vs the MLP + IG ranking vs measurements only; four LLMs, three datasets):
the prompts, all attempts and selected answers, the scripts for the generation rate, the five-criteria quality radar and the hypothesis composition agreement, and their results. See the README in each folder.

## Canonical outputs, provenance and expert assessment
- **The 144 selected responses are the canonical evaluated outputs.** `3_answers/<model>/<case>_selected_answer.json` (24 models x 6 cases) is the single response per model and case that every score, similarity and figure in this repository was computed from. These files are not regenerated or edited.
- **Raw candidates and selection provenance are stored separately** in `3_answers/selection_provenance/`: `selection_manifest.csv` (one row per canonical response: candidate count, selected candidate, method, selector judgment, seed, retries, date), `manual_exceptions.csv`, the selector judgments and the raw candidate files that could be recovered.
  Selection methods: 78 single-sample, 57 consistent-pair, 6 embedding-medoid of five (Gemma 4 12B), 2 random no-consensus, 1 manual selection from the archive (Gemini, Fxyd6 - its selection criterion is unrecorded). The responses themselves are unchanged; only the label of the six Gemma files was corrected.
- **Some settings or raw files are unavailable and are labelled `unrecorded`** (or `not available`) rather than inferred: for example the seed of the two random selections, the retry counts of the vLLM and second-batch Together runs (seeds and retries are recorded where a record exists, see `3_answers/selection_provenance/README.md`), checkpoint revisions, the five raw Gemma 4 12B candidates and the candidates of one MiniMax case. `1_generation_code/model_settings/benchmark_version_manifest.csv` lists, per model, the settings that were explicitly recorded, those left to the provider, and those unrecorded; `benchmark_repository_commit.txt` records the repository commits of the archived benchmark.
- **Provider-served and stochastic generations are not expected to be reproduced verbatim.** Sampling, provider-side model updates and unrecorded defaults mean that re-running a generation will give different text; the archived candidates and selected responses are the record.
- **The expert-assessment records are de-identified** (`5_evaluation_results/expert_assessment/`): assessors are anonymous (`Assessor_1` ...), no names, contact details or timestamps are kept, and the original questionnaire (PDF), the background documents given to the assessors, the hypotheses as shown, the label-to-model mapping and the per-assessor ratings are archived; items that could not be recovered are listed in that folder's README.
  `aggregate_expert_ratings.py` reproduces the expert-score aggregation and `figure_7E_plotting_input.csv` (the input of the Figure 7E plot); `validate_expert_archive.py` and `3_answers/selection_provenance/validate_selection_provenance.py` check the archive.

Supplementary Figures S15 (ablation) and S16 (classifier choice, decision-tree settings): the data tables and plotting scripts that redraw every panel are in `6_supplementary_figures/`; the records behind S15 are in the `ablation_conditions/` folders above, those behind S16 in `6_supplementary_figures/FigS16_classifier_and_DT_settings/records/`.
