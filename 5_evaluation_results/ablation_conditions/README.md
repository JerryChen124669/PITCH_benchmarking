# Evaluation results: ablation of the evidence conditions

Condition labels: **PITCH** = MLP + IG + decision tree + LLM; **MLP + IG** = MLP + integrated gradients + LLM (no decision tree); **measurement-only** = LLM given sampled measurements only; **context-only** = LLM given only the study system, design and target response (no data).
Datasets: CITE-seq CD45RA, CosMx Ki-67, Gouwens input resistance. Models: Gemini-3.1-pro-preview, GLM-5.3, DeepSeek V4.1 Flash, Qwen3.8-2.4T-A95B.

**Where the supplementary items come from**

| Supplement | What | Data here | Drawn by |
|---|---|---|---|
| Table S2 | hypothesis generation rates | `answer_screening.csv`, `1_success_rate_table/` | `../../6_supplementary_figures/TableS2_generation_rate/` |
| Table S3 | model-level hypothesis-quality scores | `2_quality_radar/scorecards/`, `2_quality_radar/per_model_scores_table.*` | `../../6_supplementary_figures/TableS3_quality_by_model/` |
| Figure S15 | five criteria + hypothesis composition agreement (radar) | `2_quality_radar/scorecards/`, `4_hypothesis_composition_agreement/` | `../../6_supplementary_figures/FigS15_ablation/` |

| Path | Content |
|---|---|
| `answer_screening.csv` | verdict of every attempt (stated / tentative / declined) with notes, from reading the answers (PITCH, MLP + IG, measurement-only, context-only) |
| `1_success_rate_table/` | generation rate table (`.xlsx`, `.csv`, `.md`): % of attempts that formulated a hypothesis, 95% Wilson CIs, attempts and per-dataset breakdown |
| `2_quality_radar/scorecards/` | judge scores of the selected answers (`pass1`, `pass2` for PITCH and MLP + IG; `raw_pass1`, `raw_pass2` for measurement-only; `ctx2_pass1`, `ctx2_pass2` for context-only) |
| `2_quality_radar/radar_all_models_average_6axes_4conditions_all_axes.*`, `..._5axes_4conditions_all_axes.*` | **four-condition radars, all axes, context-only drawn on every axis** (datasheet `radar_4conditions_datasheet.xlsx`) |
| `2_quality_radar/radar_all_models_average_6axes_4conditions.*`, `..._5axes_4conditions.*` | the same without Feature Comprehensiveness and Mechanistic Synthesis for context-only (not applicable) |
| `2_quality_radar/per_model_scores_table.{csv,md}` | per-model, per-condition scores (n, FC, BP, Spec., MS, LG): the source of Table S3 |
| `2_quality_radar/radar_all_models_average_6axes.*`, `..._6axes_3conditions.*` | earlier radars: PITCH vs MLP + IG, and the three data-fed conditions |
| `4_hypothesis_composition_agreement/` | agreement scores across the four models (`data/` records, rationale CSV, datasheet, bar plot); `data_available_models/` = agreement of measurement-only (2-3 LLMs per prompt) and context-only (4 LLMs per prompt), records of 3 prompts x 3 passes, scores and rationales |

## Generation rate (Table S2)
Percentage of attempts (all completed calls, including retries) that committed to a hypothesis, 95% Wilson CI in parentheses, successes/attempts in brackets.

| Condition | Gemini-3.1-pro-preview | Qwen3.8-2.4T-A95B | GLM-5.3 | DeepSeek V4.1 Flash |
|---|---|---|---|---|
| PITCH | 100% (70-100%)  [9/9] | 100% (70-100%)  [9/9] | 100% (70-100%)  [9/9] | 100% (70-100%)  [9/9] |
| MLP + IG | 17% (9-31%)  [7/41] | 100% (70-100%)  [9/9] | 100% (70-100%)  [9/9] | 100% (70-100%)  [9/9] |
| Measurement-only | 11% (2-44%)  [1/9] | 44% (19-73%)  [4/9] | 100% (70-100%)  [9/9] | 22% (6-55%)  [2/9] |
| Context-only | 100% (70-100%)  [9/9] | 100% (70-100%)  [9/9] | 100% (70-100%)  [9/9] | 89% (56-98%)  [8/9] |

## Quality and agreement (Figure S15; all-model averages, 0-10)
The criterion is called **Hypothesis Specificity** in the manuscript.

| Axis | PITCH (n = 12) | MLP + IG (n = 12) | Measurement-only (n = 7) | Context-only (n = 12) |
|---|---|---|---|---|
| Literature Grounding | 7.42 | 7.96 | 7.64 | 7.79 |
| Biological Plausibility | 8.29 | 8.21 | 7.93 | 8.25 |
| Hypothesis Specificity | 7.33 | 5.83 | 4.86 | 8.21 |
| Mechanistic Synthesis | 7.17 | 5.75 | 4.79 | 8.04 (n/a) |
| Feature Comprehensiveness | 9.58 | 8.50 | 8.57 | 9.54 (n/a) |
| Hypothesis composition agreement | 7.67 (4 LLMs) | 4.33 (4 LLMs) | 2.89 (2-3 LLMs) | 3.44 (4 LLMs) |

Notes: quality scores are the mean of two judge passes on one selected committed answer per model, condition and dataset (12 answers per condition; 7 for measurement-only, where a committed answer exists in 7 of 12 cells); the agreement score is the mean of three random-order judging passes per prompt (3 prompts per condition).
Context-only: Feature Comprehensiveness and Mechanistic Synthesis are defined on supplied focus / contextual features and do not apply when no data are given (shown for reference; n/a); the context-only prompts have no honesty / abstention paragraph (`2_prompts/ablation_conditions/README.md`).
Measurement-only agreement rests on groups of 2-3 answers, which agree more easily than groups of four. The PITCH prompt still asks for four contextual features whereas the other prompts do not.
Specificity and Mechanistic Synthesis correlate r = 0.96 across answers (they are almost the same measurement), and neither criterion tests whether the data support the hypothesis: a context-only answer can score as high as a PITCH answer on both; agreement is the only axis that depends on the data.

**Legacy zone-identification fields.** The scorecards produced by the earlier seven-criterion judge (benchmark scores `evaluation_results_rubric_v12/`, ablation scorecards `ablation_conditions/.../scorecards/`) still contain a `Zone Identification score_zone1` field. Zone identification is not reported: this field is excluded from every analysis, figure and table in the manuscript, and the scripts in this repository no longer request or use it.
