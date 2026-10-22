# Evaluation results

All six prompts are included for the judged data (the figures that use four prompts drop IL2RA and pRPS6 inside the scripts).

| Folder | Content |
|---|---|
| `evaluation_results_rubric_v12/` | `all_scores.csv` (one row per model and prompt: seven rubric scores for the primary zone, with the judge's short reasons for plausibility and synthesis), one `*_scorecard.json` per model and prompt, `total_score_6cases_excl_zone_identification/` (figure C: script, data and plot of the total score, six criteria without zone identification, six prompts) and `1_pillar_plots_4cases_excl_pRPS6_IL2RA/` (per-criterion charts, four prompts). The `final_score_zone1` field in the scorecards is an older total that includes zone identification and counts literature grounding at half weight; it is not used in the figures. |
| `pairwise_similarity_results_v2/<prompt>/` | One judged record per unordered model pair and prompt: six similarity scores (P1-P6) and their mean. Input to both similarity analyses. |
| `pairwise_similarity_results_v2/FINAL_3categories_similarity_and_performance/` | Total similarity (figures A, B): `cluster_assignments.csv`, similarity and distance matrices, dendrogram heat map, cluster map, category profile tables and figures. |
| `pairwise_similarity_results_v2/FINAL_all_models_no_temp0/selected_models_4cases/` | Mechanistic similarity (figures D, E): `data_selection/` (model table, selected models), `data_diversity/` (correlation tests and tables) and `plots/` (PNG, SVG, PDF). `model_table.csv` regenerated with the final code has four extra columns (lowest plausibility, specificity and synthesis per model and the number of prompts that pass a gate). |

24 models are included; the temperature-0 Qwen run is not one of them.

**Legacy zone-identification fields.** The scorecards produced by the earlier seven-criterion judge (benchmark scores `evaluation_results_rubric_v12/`, ablation scorecards `ablation_conditions/.../scorecards/`) still contain a `Zone Identification score_zone1` field. Zone identification is not reported: this field is excluded from every analysis, figure and table in the manuscript, and the scripts in this repository no longer request or use it.
