# Evaluation code: ablation of the evidence conditions

Scripts for the generation rate, the five-criteria quality radar plots and the hypothesis composition agreement of the evidence-condition ablation (PITCH vs MLP + IG vs measurement-only; CITE-seq CD45RA, CosMx Ki-67, Gouwens input resistance; four LLMs).
Run from this folder (`python <script>`); paths resolve through `common.py` to `2_prompts/`, `3_answers/` and `5_evaluation_results/` of this repository. Judging scripts need `OPENAI_API_KEY` in the environment (no key is stored).

| Script | Purpose | API |
|---|---|---|
| `01_build_success_rate_table.py` | **Generation rate**: % of attempts that formulated a hypothesis per model and condition, 95% Wilson CI -> `1_success_rate_table/` | no |
| `02a_select_answers.py` | keeps 3 answers per prompt (committed first) and draws one committed answer per model, condition and dataset (seed 20) -> `3_answers/ablation_conditions/selected_answers/` | no |
| `02b_judge_quality.py` | blind 5-criteria quality judge (gpt-5.6-sol; unmodified rubric with the five-criterion preamble), `--conds`, `--out-dir`; run twice (two passes) | yes |
| `02c_plot_quality_radars.py` | mean of the two passes; writes `quality_radar_datasheet.xlsx` (all-model averages used by the 6-axis radars) and the 5-axis radars | no |
| `04a_judge_hypothesis_composition_agreement.py` | **hypothesis composition agreement**: four models' answers judged together in random order, 3 passes; `--dry-run` checks the rubric hash (0ffd5598c1bf...) | yes |
| `04b_plot_hypothesis_composition_agreement.py` | bar plot and datasheet of the agreement score | no |
| `02e_plot_radar_6axes.py` | **6-axis radar**, PITCH vs MLP + IG: 5 quality criteria + hypothesis composition agreement | no |
| `02f_plot_radars_3conditions.py` | radars for all three conditions; its 6-axis version plots the measurement-only agreement as 0 (not measured) | no |
| `sampling/` | **measurement-only sampling**: `raw_data_sampler.py`, `run_ablation.py` and the modules it imports; rule, seed (42) and the sampled row indices (`measurement_only_sampled_cells.csv`) | token counting only |
| `common.py`, `raw_prompt_versions.py`, `evaluation/` | shared paths and helpers; exact measurement-only prompt wordings; judge rubric module (`judge_evaluation.py` is the verbatim 7-criterion rubric, sha256 522c88817cdc...; it is not the same file as `../judge_evaluation.py` of the benchmark), rubrics | - |

Order: `02a` -> `01`; `02b` (pass 1, pass 2; measurement-only with `--conds cond2_raw_data`) -> `02c`; `04a` -> `04b` -> `02e`, `02f`.
Results are in `../5_evaluation_results/ablation_conditions/`.
