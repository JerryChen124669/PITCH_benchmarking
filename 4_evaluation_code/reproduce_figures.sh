#!/bin/bash
# Reproduces the five figures of the paper from the judged data in ../5_evaluation_results.
# Run from a working directory that holds the scripts in this folder (including model_registry.json) together with
# evaluation_results_rubric_v12/ and pairwise_similarity_results_v2/ from ../5_evaluation_results/ (see README.md).
# Needs numpy, pandas, scipy, scikit-learn, matplotlib, Pillow (no API key: nothing is re-judged).
set -e
FINAL=pairwise_similarity_results_v2/FINAL_3categories_similarity_and_performance
SEL=pairwise_similarity_results_v2/FINAL_all_models_no_temp0/selected_models_4cases

# Figure A  (similarity dendrogram + heat map): total similarity, all six criteria, six prompts, 22 models
python analyze_pairwise_clustering.py --min-total-b 0 --exclude-models Qwen3.8-9B-think-temp0 biomistral pmc_llama \
    --add-pillars P6 --k 5 --merge-clusters 2,3,5 --plots cluster_map dendrogram_heatmap \
    --category-names "Rule-faithful descriptive annotators" "Integrative, weakly grounded hypothesis generators" "Mechanistic, literature-grounded integrators" \
    --out-dir "$FINAL"                                   # -> $FINAL/dendrogram_heatmap.svg

# Figure B  (category profile heat map): rubric scores of the three categories, six prompts
python characterize_categories.py                        # -> $FINAL/category_profile_heatmap.svg

# Figure C  (overall rank, excluding specificity): four prompts
python plot_evaluation_pillars.py --eval-dir evaluation_results_rubric_v12 --exclude-prompts pRPS6 IL2RA \
    --out-dir evaluation_results_rubric_v12/1_pillar_plots_4cases_excl_pRPS6_IL2RA   # -> .../overall_rank_excl_specificity.pdf

# Figure D  (quality eligibility vs mechanistic diversity): four prompts
python quality_vs_diversity_selection.py --assignments "$FINAL/cluster_assignments.csv" --exclude-cases IL2RA pRPS6 --strict --redundant-below 2 \
    --out-dir "$SEL/data_selection"                      # -> $SEL/data_selection/quality_vs_diversity_selection.svg  (= plots/1_...)

# Figure E  (mean quality vs mean distinctiveness of the selected models): four prompts
python diversity_vs_quality.py --assignments "$SEL/data_selection/selected_models_assignments.csv" --exclude-cases IL2RA pRPS6 \
    --label "selected; IL2RA and pRPS6 excluded" --out-dir "$SEL/data_diversity"   # -> .../diversity_vs_quality_panel_B_models.svg  (= plots/2_...)
