# Evaluation code

Everything needed to reproduce the five evaluation figures from the judged data, plus the two scripts that produced the
judgements. `reproduce_figures.sh` holds the exact command for each figure; all five regenerate identically (SVG text and
bars, PDF/JPG pixels) from the data in `../5_evaluation_results/`.

| Figure | File | Script |
|---|---|---|
| A. Total-similarity dendrogram and heat map | `.../FINAL_3categories_similarity_and_performance/dendrogram_heatmap.svg` | `analyze_pairwise_clustering.py` |
| B. Category profile heat map | `.../FINAL_3categories_similarity_and_performance/category_profile_heatmap.svg` | `characterize_categories.py` |
| C. Overall rank, excluding specificity | `evaluation_results_rubric_v12/1_pillar_plots_4cases_excl_pRPS6_IL2RA/overall_rank_excl_specificity.pdf` | `plot_evaluation_pillars.py` |
| D. Quality eligibility vs mechanistic diversity | `.../selected_models_4cases/plots/1_quality_vs_diversity_selection.svg` | `quality_vs_diversity_selection.py` |
| E. Mean quality vs mean distinctiveness | `.../selected_models_4cases/plots/2_mean_quality_vs_mean_distinctiveness.svg` | `diversity_vs_quality.py` |

## Working directory
The scripts import each other and read their inputs from sibling folders. Put in one directory: all files from this
folder, and `evaluation_results_rubric_v12/` and `pairwise_similarity_results_v2/` from `../5_evaluation_results/`.
To rerun the judges (not needed for the figures) the selected answers from `../3_answers/` are also needed, as `trial_results/`.

## Judging scripts (already run; their outputs are in the results folder)
| Script | Purpose |
|---|---|
| `judge_evaluation.py` | Blind rubric judging of each selected answer. Judge `gpt-5.6-sol` (OpenAI chat completions; no temperature, reasoning effort, token limit or seed set; 300 s timeout), one call per answer, primary zone only, integer scores 0-10 per criterion. The rubric and all scoring anchors are `JUDGE_SYSTEM_PROMPT` (line 50). Key from `OPENAI_API_KEY`. |
| `pairwise_similarity_v2.py` | Blind pairwise similarity of two models' selected answers on one prompt, six criteria P1-P6, integer 0-10. `gpt-5.6-sol`; one call per unordered pair and prompt, answers shown as Response A / B in a random order (one order only). Criteria and anchors: `PAIRWISE_SYSTEM_PROMPT` (line 70). |

## Two similarity analyses
| | Total similarity (figures A, B) | Mechanistic similarity (figures D, E) |
|---|---|---|
| Question | Which models respond alike overall? | Among the high-quality models, are the hypotheses different from each other? |
| Criteria | all six, P1-P6, averaged | P3 (mechanistic hypothesis) and P4 (functional / causal role), averaged |
| Prompts | all six | four: Fxyd6, HUMAN, PTMA, pERK |
| Models | 22 (all except BioMistral, PMC-LLaMA and the temperature-0 Qwen run) | the models that pass the quality line (10 models) |
| Method | model-pair score = mean over prompts; distance = 1 - S/10; average-linkage clustering | per prompt: distinctiveness of a model = 10 - mean (P3+P4)/2 similarity to the other eligible models; averaged over prompts |

**Total similarity.** The number of clusters is forced to 5; clusters 2, 3 and 5 of that cut are then merged after the fact
into one category, giving 3 categories of 3, 6 and 13 models (silhouette of the merged partition 0.219). The merge is a
judgement, not a dendrogram cut. Category names are provisional. Figure B profiles the categories with the rubric scores
(model score = mean over prompts; category value = unweighted mean of its models; Kruskal-Wallis and Cliff's delta in the tables).

**Mechanistic similarity.** Case quality = mean(biological plausibility, specificity, mechanistic synthesis). A model's quality
floor is its second-lowest case quality (so reached in at least 3 of 4 prompts); models with a floor above 7.0 are eligible.
A model whose distinctiveness is below 2 (mean P3+P4 similarity to the other eligible models of 8 or more) is dropped as
redundant; none is (the selected models range from 2.25 to 3.14). Ten models are selected. `diversity_vs_quality.py` then tests, among them, whether distinctiveness relates
to quality (Spearman with permutation tests: per model rho = -0.38, p = 0.27, n = 10; per model pair rho = +0.07, p = 0.78, n = 45).
The scripts also contain other options explored during analysis (`--gate`, `--floor-case lowest`, `--quality-stat lowest`);
none of them is part of the reported figures.

## Overall rank in figure C
Sum of six criteria, each 0-10, at full weight: zone identification, logic rigor, feature comprehensiveness, biological
plausibility, mechanistic synthesis and literature grounding (specificity is not included), maximum 60, computed per answer
and averaged over the four prompts. A companion figure with all seven criteria (`overall_rank_all_7_pillars`) and one chart per
criterion are in the same folder.

Rubric scoring and pairwise similarity need `OPENAI_API_KEY` in the environment; no key is stored in the code.
