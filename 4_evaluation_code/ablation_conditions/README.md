# Evaluation code: ablation of the evidence conditions

Scripts for the generation rate, the five-criteria quality evaluation and the hypothesis composition agreement of the ablation (PITCH vs MLP + IG vs measurement-only; CITE-seq CD45RA, CosMx Ki-67, Gouwens input resistance; four LLMs). The scripts also support a fourth, context-only control that is **not used in the manuscript** (see below).
**They produce the data behind Table S2 (generation rates), Table S3 (model-level quality scores) and Figure S15 (radar); the tables and the figure themselves are drawn from the records in `5_evaluation_results/ablation_conditions/` by the scripts in `6_supplementary_figures/`. Table S3 and Figure S15 contain PITCH, MLP + IG and measurement-only.**
Run from this folder (`python <script>`); paths resolve through `common.py` to `2_prompts/`, `3_answers/` and `5_evaluation_results/` of this repository. Judging scripts need `OPENAI_API_KEY` in the environment (no key is stored).
Condition identifiers: `cond0_full_pipeline` PITCH, `cond1_ig_only` MLP + IG, `cond2_raw_data` measurement-only, `cond4_context_only_softened` context-only.

| Script | Purpose | API |
|---|---|---|
| `01_build_success_rate_table.py` | **Generation rate** (Table S2 source): % of attempts that formulated a hypothesis per model and condition, 95% Wilson CI -> `5_evaluation_results/ablation_conditions/1_success_rate_table/` | no |
| `02a_select_answers.py` | keeps 3 answers per prompt (committed first) and draws one committed answer per model, condition and dataset (seed 20) -> `3_answers/ablation_conditions/selected_answers/` (earlier picks are unchanged when a condition is added) | no |
| `02b_judge_quality.py` | blind 5-criteria quality judge (gpt-5.6-sol; unmodified rubric with the five-criterion preamble), `--conds`, `--out-dir`; two passes. PITCH and MLP + IG: `scorecards/pass1`, `pass2`; measurement-only: `--conds cond2_raw_data` -> `raw_pass1`, `raw_pass2`; context-only: `--conds cond4_context_only_softened` -> `ctx2_pass1`, `ctx2_pass2` | yes |
| `02c_plot_quality_radars.py` | mean of the two passes; `quality_radar_datasheet.xlsx` (PITCH and MLP + IG) | no |
| `04a_judge_hypothesis_composition_agreement.py` | **hypothesis composition agreement** for PITCH and MLP + IG: the four LLMs' answers judged together in random order, 3 passes; `--dry-run` checks the rubric hash (0ffd5598c1bf...) | yes |
| `04b_plot_hypothesis_composition_agreement.py` | bar plot and datasheet of the agreement score (PITCH, MLP + IG) | no |
| `04c_judge_hypothesis_composition_agreement_available_models.py` | agreement of **measurement-only** (and of the context-only control with `--conds cond4_context_only_softened`): same rubric, judge and random-order passes as `04a`, applied to the LLMs that have a committed answer for the prompt (measurement-only 2, 3 and 2 answers); `--conds`, `--dry-run`, `--group-from` | yes |
| `04d_summarize_agreement_all_conditions.py` | one table of the agreement of PITCH, MLP + IG and measurement-only (LLMs judged together, pass scores, mean, record folder) -> `4_hypothesis_composition_agreement/agreement_all_conditions.{csv,md}`; `--with-context-only` adds the control | no |
| `02e_plot_radar_6axes.py` | 6-axis radar, PITCH vs MLP + IG | no |
| `02f_plot_radars_3conditions.py` | radars for PITCH, MLP + IG and measurement-only (and the helper functions used by `02h`) | no |
| `02h_plot_radars_4conditions.py` | **exploratory, context-only control (not used in the manuscript)**: four-condition radars, datasheet and per-model score table including the control; its outputs are not kept in the repository, the records it reads are | no |
| `common.py`, `raw_prompt_versions.py`, `evaluation/`, `sampling/` | shared paths and helpers; exact measurement-only prompt wordings; judge rubric module (`judge_evaluation.py` is the verbatim 7-criterion rubric, sha256 522c88817cdc...; it is not the same file as `../judge_evaluation.py` of the benchmark) and rubrics; measurement-only sampling | - |

Order: `02a` -> `01`; `02b` (pass 1 and 2 for each condition, see above) -> `02c`; `04a` -> `04b`; `04c` (measurement-only) -> `02f`. Then `6_supplementary_figures/` (Tables S2, S3, Figure S15).

**Context-only control (not used in the manuscript).** A fourth condition gives the LLM no data. Its generation, selection and judging are supported by the same scripts (`run_context_only.py` in `../../1_generation_code/ablation_conditions/`; `02a`, `02b --conds cond4_context_only_softened`, `04c --conds cond4_context_only_softened`), and its records are kept (`../../5_evaluation_results/ablation_conditions/`), but Figure S15, Table S3 and the repository result tables do not contain it; Table S2 reports its generation rate. Feature Comprehensiveness and Mechanistic Synthesis are defined on supplied focus / contextual features and do not apply when no data are given.
Agreement groups of different size (2-4 answers) are not strictly comparable.
The displayed name of the specificity criterion is **Hypothesis Specificity** everywhere; the judge rubric file keeps its original heading "Hypothesis Integration and Specificity" (it is the text the judge was given).
Generation code: `../../1_generation_code/ablation_conditions/`.
