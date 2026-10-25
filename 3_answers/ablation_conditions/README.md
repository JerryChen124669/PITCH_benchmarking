# Ablation-condition answers

| Folder | Content |
|---|---|
| `all_attempts/<model>/<dataset>_<condition>_run<k>.json` | every saved model attempt for the three datasets and three conditions (176 files; fields include `final_answer`, `finish_reason`, `usage`, `prompt_variant`, `temperature`). Run numbers above 3 are the additional attempts (Gemini with MLP + IG). Attempts that hit the output limit have an empty `final_answer`. |
| `selected_answers/<condition>/<model>/<dataset>_<condition>_picked_run<k>.json` | the single answer per model, condition and dataset used for the quality and agreement evaluations (a committed answer drawn at random with seed 20; `selection_manifest.json` lists the candidates and the pick). Cells without a committed answer are absent (measurement-only: 7 of 12 cells). |

Models: Gemini-3.1-pro-preview, GLM-5.3, DeepSeek V4.1 Flash, Qwen3.8-2.4T-A95B (temperature 0.5).
Condition identifiers in file names: `cond0_full_pipeline` = PITCH, `cond1_ig_only` = MLP + IG, `cond2_raw_data` = measurement-only, `cond4_context_only_softened` = context-only (36 attempts: 4 models x 3 datasets x 3 repeats; GLM-5.3 at reasoning effort `max`, see `1_generation_code/ablation_conditions/README.md`).

The folder `ablation_conditions/` contains no file named `*_selected_answer.json`, so the benchmark scripts in `4_evaluation_code/` (which glob that pattern inside each model folder of `3_answers/`) skip it.
The verdict of each attempt (stated / tentative / declined) is in `5_evaluation_results/ablation_conditions/answer_screening.csv`.

**Context-only answers.** 35 of the 36 attempts returned text and committed to a hypothesis; one DeepSeek V4.1 Flash call (Patch-seq, run 2) reached the output limit on hidden reasoning and returned no text (counted as an attempt and as a failure). Twelve answers (one per model and dataset, drawn at random with seed 20) are in `selected_answers/cond4_context_only_softened/`; they are kept as control records and are not used in Table S3 or Figure S15. Gemini set `sufficient_information = false` in its three CosMx answers although the text commits to a specific hypothesis (confidence 8); they are counted as committed.
