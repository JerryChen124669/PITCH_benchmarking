# Evaluation results: ablation of the evidence conditions

Condition labels: **PITCH** = MLP + IG + decision tree + LLM; **MLP + IG** = MLP + integrated gradients + LLM (no decision tree); **measurement-only** = LLM given sampled measurements only.
Datasets: CITE-seq CD45RA, CosMx Ki-67, Gouwens input resistance. Models: Gemini-3.1-pro-preview, GLM-5.3, DeepSeek V4.1 Flash, Qwen3.8-2.4T-A95B.
A fourth **context-only** control (no data given) was run; its records are kept here but it is not part of Table S3 or Figure S15 (section at the end).

**Where the supplementary items come from**

| Supplement | What | Data here | Drawn by |
|---|---|---|---|
| Table S2 | hypothesis generation rates | `answer_screening.csv`, `1_success_rate_table/` | `../../6_supplementary_figures/TableS2_generation_rate/` |
| Table S3 | model-level hypothesis-quality scores | `2_quality_radar/scorecards/` | `../../6_supplementary_figures/TableS3_quality_by_model/` |
| Figure S15 | five criteria + hypothesis composition agreement (radar) | `2_quality_radar/scorecards/`, `4_hypothesis_composition_agreement/` | `../../6_supplementary_figures/FigS15_ablation/` |

| Path | Content |
|---|---|
| `answer_screening.csv` | verdict of every attempt (stated / tentative / declined) with notes, from reading the answers |
| `1_success_rate_table/` | generation rate table (`.xlsx`, `.csv`, `.md`): % of attempts that formulated a hypothesis, 95% Wilson CIs, attempts and per-dataset breakdown |
| `2_quality_radar/scorecards/` | judge scores of the selected answers (`pass1`, `pass2` for PITCH and MLP + IG; `raw_pass1`, `raw_pass2` for measurement-only; `ctx2_pass1`, `ctx2_pass2` for the context-only control) |
| `2_quality_radar/radar_all_models_average_6axes.*`, `..._6axes_3conditions.*` | 6-axis radars: PITCH vs MLP + IG, and PITCH, MLP + IG and measurement-only (datasheets `radar_6axes_datasheet.xlsx`, `radar_3conditions_datasheet.xlsx`) |
| `4_hypothesis_composition_agreement/` | records of the agreement judgments (see its `README.md`): `data/` = PITCH and MLP + IG (4 LLMs per prompt; datasheet and bar plot); `data_available_models/` = measurement-only (2-3 LLMs per prompt) and the context-only control; `agreement_all_conditions.csv` = PITCH, MLP + IG and measurement-only in one table with the LLMs of each group |

## Generation rate (Table S2)
Percentage of attempts (all completed calls, including retries) that committed to a hypothesis, 95% Wilson CI in parentheses, successes/attempts in brackets. The last row is the context-only control.

| Condition | Gemini-3.1-pro-preview | Qwen3.8-2.4T-A95B | GLM-5.3 | DeepSeek V4.1 Flash |
|---|---|---|---|---|
| PITCH | 100% (70-100%)  [9/9] | 100% (70-100%)  [9/9] | 100% (70-100%)  [9/9] | 100% (70-100%)  [9/9] |
| MLP + IG | 17% (9-31%)  [7/41] | 100% (70-100%)  [9/9] | 100% (70-100%)  [9/9] | 100% (70-100%)  [9/9] |
| Measurement-only | 11% (2-44%)  [1/9] | 44% (19-73%)  [4/9] | 100% (70-100%)  [9/9] | 22% (6-55%)  [2/9] |
| Context-only (control) | 100% (70-100%)  [9/9] | 100% (70-100%)  [9/9] | 100% (70-100%)  [9/9] | 89% (56-98%)  [8/9] |

## Quality and agreement (Figure S15; all-model averages, 0-10)
The criterion is called **Hypothesis Specificity** in the manuscript.

| Axis | PITCH (n = 12) | MLP + IG (n = 12) | Measurement-only (n = 7) |
|---|---|---|---|
| Feature Comprehensiveness | 9.58 | 8.50 | 8.57 |
| Biological Plausibility | 8.29 | 8.21 | 7.93 |
| Hypothesis Specificity | 7.33 | 5.83 | 4.86 |
| Mechanistic Synthesis | 7.17 | 5.75 | 4.79 |
| Literature Grounding | 7.42 | 7.96 | 7.64 |
| Hypothesis composition agreement (PITCH, MLP + IG: across the 4 LLMs; measurement-only: among the 2-3 LLMs with a successfully generated answer) | 7.67 | 4.33 | 2.89 |

Notes: quality scores are the mean of two judge passes on one selected committed answer per model, condition and dataset (12 answers per condition; 7 for measurement-only, where a committed answer exists in 7 of 12 cells); the agreement score is the mean of three random-order judging passes per prompt (3 prompts per condition).
Measurement-only agreement rests on groups of 2-3 answers, which agree more easily than groups of four. The PITCH prompt still asks for four contextual features whereas the other prompts do not.

## Context-only control (records kept; results not reported here)
The control gives the LLM only the study system, design and target response (prompts in `../../2_prompts/ablation_conditions/`). It is not used for Table S3 or Figure S15, and none of its quality or agreement results is tabulated or plotted in this folder.
Kept so that it can be inspected and re-run: the judge scorecards `2_quality_radar/scorecards/ctx2_pass1`, `ctx2_pass2`; the agreement records `4_hypothesis_composition_agreement/data_available_models/cond4_context_only_softened/`; the verdicts in `answer_screening.csv` (its generation rate is the last row of Table S2);
the scripts `02b_judge_quality.py --conds cond4_context_only_softened`, `04c_...py --conds cond4_context_only_softened`, `02h_plot_radars_4conditions.py` and `04d_summarize_agreement_all_conditions.py --with-context-only` in `../../4_evaluation_code/ablation_conditions/`.

**Legacy zone-identification fields.** The scorecards produced by the earlier seven-criterion judge (benchmark scores `evaluation_results_rubric_v12/`, ablation scorecards `ablation_conditions/.../scorecards/`) still contain a `Zone Identification score_zone1` field. Zone identification is not reported: this field is excluded from every analysis, figure and table in the manuscript, and the scripts in this repository no longer request or use it.
