# Hypothesis composition agreement: records of all four conditions

For one prompt (condition x dataset) the selected answers of the LLMs are shown together, anonymised, in a random order, to the judge (gpt-5.6-sol; rubric sha256 0ffd5598c1bf...), which returns a one-sentence rationale and an integer score 0-10 for the agreement of the molecular contributors. Three passes per prompt (new random order each pass); the prompt score is the mean of the three passes.

| Condition | LLMs judged together | Records |
|---|---|---|
| PITCH | all four LLMs on each prompt | `data/cond0_full_pipeline/` |
| MLP + IG | all four LLMs on each prompt | `data/cond1_ig_only/` |
| Measurement-only | only the LLMs that have a committed answer for the prompt: CITE-seq GLM-5.3 + Qwen3.8-2.4T-A95B; CosMx GLM-5.3 + DeepSeek V4.1 Flash + Qwen3.8-2.4T-A95B; Gouwens Gemini-3.1-pro-preview + GLM-5.3 | `data_available_models/cond2_raw_data/` |
| Context-only | all four LLMs on each prompt | `data_available_models/cond4_context_only_softened/` |

Each record (`<dataset>_pass<k>.json`) holds the judged order of the answers, the judge's rationale and the score; for `data_available_models/` it also lists the LLMs of the group.

| File | Content |
|---|---|
| `agreement_all_conditions.csv` / `.md` | **all four conditions in one table**: per prompt the LLMs judged together, the three pass scores, the mean, and the record folder (produced by `4_evaluation_code/ablation_conditions/04d_summarize_agreement_all_conditions.py`) |
| `data/` | records, scores (`agreement_scores.csv`), rationales (`justifications.csv`) and metadata of PITCH and MLP + IG (script `04a`) |
| `data_available_models/` | records of Measurement-only and Context-only, with `agreement_scores_available_models.csv`, `all_agreement_available_models.json` and `meta.json` (script `04c`) |
| `hypothesis_composition_agreement_datasheet.xlsx`, `hypothesis_composition_agreement_bars.*` | datasheet and bar plot of the **PITCH and MLP + IG** comparison only (the two conditions judged on all four LLMs); they do not contain Measurement-only or Context-only |

Mean over the three prompts: PITCH 7.67, MLP + IG 4.33, Measurement-only 2.89 (groups of 2-3 answers, which agree more easily than groups of four), Context-only 3.44.
The agreement score measures similarity between LLMs, not correctness. Radar plots that combine these scores with the quality criteria: `../2_quality_radar/` and `../../../6_supplementary_figures/FigS15_ablation/`.
