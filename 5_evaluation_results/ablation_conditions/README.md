# Evaluation results: ablation of the evidence conditions

Condition labels: **PITCH** = MLP + IG + decision tree + LLM; **MLP + IG** = MLP + integrated gradients + LLM (no decision tree); **measurement-only** = LLM given sampled measurements only.
Datasets: CITE-seq CD45RA, CosMx Ki-67, Gouwens input resistance. Models: Gemini-3.1-pro-preview, GLM-5.3, DeepSeek V4.1 Flash, Qwen3.8-2.4T-A95B.

| Path | Content |
|---|---|
| `answer_screening.csv` | verdict of every attempt (stated / tentative / declined) with notes, from reading the answers |
| `1_success_rate_table/` | generation rate table (`.xlsx`, `.csv`, `.md`): % of attempts that formulated a hypothesis, 95% Wilson CIs, attempts and per-dataset breakdown |
| `2_quality_radar/scorecards/` | judge scores of the selected answers (`pass1`, `pass2` for PITCH and MLP + IG; `raw_pass1`, `raw_pass2` for measurement-only) |
| `2_quality_radar/radar_all_models_average_6axes.*` | 6-axis radar, PITCH vs MLP + IG (datasheet `radar_6axes_datasheet.xlsx`) |
| `2_quality_radar/radar_all_models_average_6axes_3conditions.*` | 6-axis radar for the three conditions; the sixth axis of measurement-only is plotted as 0 (not measured) (datasheet `radar_3conditions_datasheet.xlsx`) |
| `4_hypothesis_composition_agreement/` | agreement scores across the four models (`data/` records, rationale CSV, datasheet, bar plot) |

## Generation rate
Percentage of attempts (all completed calls, including retries) that committed to a hypothesis, 95% Wilson CI in parentheses, successes/attempts in brackets.

| Condition | Gemini-3.1-pro-preview (successfully generated a hypothesis for the question) | Qwen3.8-2.4T-A95B (successfully generated a hypothesis for the question) | GLM-5.3 (successfully generated a hypothesis for the question) | DeepSeek V4.1 Flash (successfully generated a hypothesis for the question) |
|---|---|---|---|---|
| PITCH | 100% (70-100%)  [9/9] | 100% (70-100%)  [9/9] | 100% (70-100%)  [9/9] | 100% (70-100%)  [9/9] |
| MLP + IG | 17% (9-31%)  [7/41] | 100% (70-100%)  [9/9] | 100% (70-100%)  [9/9] | 100% (70-100%)  [9/9] |
| Measurement-only | 11% (2-44%)  [1/9] | 44% (19-73%)  [4/9] | 100% (70-100%)  [9/9] | 22% (6-55%)  [2/9] |


## Six-axis radar (all-model averages, 0-10)
| Axis | PITCH | MLP + IG | Measurement-only (n = 7) |
|---|---|---|---|
| Feature Comprehensiveness | 9.58 | 8.50 | 8.57 |
| Biological Plausibility | 8.29 | 8.21 | 7.93 |
| Hypothesis Integration & Specificity | 7.33 | 5.83 | 4.86 |
| Mechanistic Synthesis | 7.17 | 5.75 | 4.79 |
| Literature Grounding | 7.42 | 7.96 | 7.64 |
| Hypothesis composition agreement (across the 4 models) | 7.67 | 4.33 | not measured (plotted as 0) |

Notes: quality scores are the mean of two judge passes on one selected committed answer per model, condition and dataset (12 answers per condition; 7 for measurement-only, where a committed answer exists in 7 of 12 cells);
the agreement score is the mean of three random-order judging passes per prompt (3 prompts per condition). The PITCH prompt still asks for four contextual features whereas the MLP + IG prompt does not.
