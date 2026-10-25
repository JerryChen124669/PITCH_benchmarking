# Supplementary Tables S2 and S3 and Figures S15 and S16: data and code to reproduce them

Everything needed to redraw the tables and panels from data tables is in `TableS2_generation_rate/`, `TableS3_quality_by_model/`, `FigS15_ablation/` and `FigS16_classifier_and_DT_settings/`. Run the scripts in the respective folder
(Python 3, `numpy`, `pandas`, `matplotlib`, `openpyxl`, `scikit-learn`; no API key, no model call). Each script reads only the files in its folder (Table S2's `export_TableS2_attempts.py` reads the saved attempts of the ablation), writes
the tables as `csv`, `md` and `xlsx` and the panels as `svg` (live text), `pdf` and `png`, and writes the numbers it draws. Names in the data tables and on the plots are the names used in the manuscript.
Supplementary Table S2, Table S3 and Figure S15 are the **ablation of the evidence conditions** (PITCH, MLP + IG, measurement-only, context-only); their records are in the `ablation_conditions/` folders of `1_generation_code/`, `2_prompts/`, `3_answers/`, `4_evaluation_code/` and `5_evaluation_results/`.

```
TableS2_generation_rate/
  export_TableS2_attempts.py            step 1: one row per completed attempt (condition, LLM, prompt, run, outcome) from ../../3_answers/ablation_conditions/ and answer_screening.csv
  data/TableS2_attempts.csv             the exported attempts (176)
  make_TableS2.py                       step 2: -> TableS2_generation_rates.{csv,md,xlsx}  (percentage, Wilson 95% CI, successes and attempts)
TableS3_quality_by_model/
  make_TableS3.py                       -> TableS3_quality_by_model.{csv,md,xlsx}   (reads ../FigS15_ablation/data/FigS15_scores_per_answer.csv)
FigS15_ablation/
  plot_FigS15.py                        -> FigS15_ablation_radar.{svg,pdf,png}  (all axes, all four conditions)
                                           FigS15_ablation_radar_applicable_axes.{svg,pdf,png}  (context-only without the two criteria that do not apply)
  data/FigS15_scores_per_answer.csv     one row per (answer, judge pass): five criteria
  data/FigS15_composition_agreement_per_prompt.csv   hypothesis composition agreement per prompt and condition (and the LLMs in each group)
  data/FigS15_plot_values.csv           the plotted values and n per condition
  (records: prompts `../2_prompts/ablation_conditions/`, answers `../3_answers/ablation_conditions/`, judge scorecards, screening and agreement records
   `../5_evaluation_results/ablation_conditions/`, code incl. measurement-only sampling `../4_evaluation_code/ablation_conditions/`, generation code `../1_generation_code/ablation_conditions/`)
FigS16_classifier_and_DT_settings/
  plot_FigS16.py                        -> FigS16A_balanced_accuracy, FigS16B_criteria_by_classifier, FigS16C_similarity_between_classifiers,
                                           FigS16D_criteria_by_DT_setting, FigS16E_similarity_between_DT_settings   (.svg/.pdf/.png)
  data/FigS16A_oof_predictions.csv.gz       out-of-fold prediction (class and positive-class probability) of LR, DT, RF for every cell, with its fold
  data/FigS16A_balanced_accuracy_by_fold.csv, FigS16A_plot_values.csv   balanced accuracy per fold and the plotted values (%)
  data/FigS16A_majority_baseline.csv, FigS16A_heldout_metrics_by_dataset.csv, FigS16A_heldout_summary_across_datasets.csv
                                            prevalence, all-background accuracy, accuracy / balanced accuracy / positive-class precision and recall
  data/FigS16A_accuracy_by_fold.csv         plain accuracy per dataset, classifier and fold (earlier version of panel A)
  data/FigS16B_scores_per_case.csv          five criteria per (dataset, classifier)
  data/FigS16C_similarity_per_case.csv      pairwise similarity (six criteria + their mean) per (dataset, classifier pair)
  data/FigS16D_scores_per_setting.csv       six criteria per (dataset, decision-tree setting)
  data/FigS16E_similarity_between_settings.csv   pairwise similarity (three criteria + their mean) per (dataset, setting pair)
  data/FigS16*_plot_values.csv              what each panel shows
  records/classifier_comparison/            prompts, evidence, answers, judge outputs, logs, code for panels A-C
  records/DT_settings/                      tree nodes, prompts, answers, judge outputs, logs, code for panels D-E
```

## Table S2. Hypothesis generation rates across four LLMs under input conditions
Values indicate the percentage of attempts that produced a specific hypothesis, with 95% Wilson confidence intervals in parentheses, pooled across the CITE-seq CD45RA, CosMx Ki-67, and Patch-seq input-resistance tasks.
All completed attempts, including retries and empty responses, were counted. Context-only: the LLM received only the study system, design and target response (no data).

| Input condition | Gemini 3.1 Pro Preview | Qwen3.8 2.4T-A95B | GLM-5.3 | DeepSeek V4.1 Flash |
|---|---|---|---|---|
| PITCH | 100 (70-100) | 100 (70-100) | 100 (70-100) | 100 (70-100) |
| MLP + IG | 17 (9-31) | 100 (70-100) | 100 (70-100) | 100 (70-100) |
| Measurement-only | 11 (2-44) | 44 (19-73) | 100 (70-100) | 22 (6-55) |
| Context-only | 100 (70-100) | 100 (70-100) | 100 (70-100) | 89 (56-98) |

How it is computed (`make_TableS2.py`): rate = attempts with outcome `committed` / completed attempts (3 per prompt and LLM = 9 per cell; Gemini with MLP + IG: 41 because it was retried; one DeepSeek context-only call returned no text and counts as a failure), Wilson interval.
`committed` = the answer commits to one specific hypothesis (caveats and low confidence allowed); `tentative` (hedged "if forced" guess), `declined` and `no text` (output limit reached) are not generated hypotheses. Verdicts are from reading each answer (`5_evaluation_results/ablation_conditions/answer_screening.csv`).

## Table S3. Model-level hypothesis-quality scores across ablation conditions
Scores are means across the successfully generated responses for each model; n denotes the number of prompts yielding an evaluable hypothesis. PITCH represents the complete pipeline, MLP + IG excludes DT rules, measurement-only input excludes both IG attribution and DT rules, and context-only input provides no data.
FC, feature comprehensiveness; BP, biological plausibility; Spec., hypothesis specificity; MS, mechanistic synthesis; LG, literature grounding. All criteria were scored from 0 to 10 by GPT-5.6-sol.
Qwen3.8, Qwen3.8-2.4T-A95B; DeepSeek, DeepSeek-V4.1-Flash; Gemini, Gemini-3.1-Pro. For context-only, FC and MS (dagger) are defined on supplied focus / contextual features and do not apply when no data are given; they are shown for reference.
How it is computed (`make_TableS3.py`): each selected answer was judged in two blind passes, the score of an answer is the mean of its two passes, a cell is the mean over the datasets for which the LLM has an answer (`data` = `../FigS15_ablation/data/FigS15_scores_per_answer.csv`).

## Figure S15. Ablation analysis of PITCH intermediate representations
Mean scores across five hypothesis-quality criteria and hypothesis-composition agreement were compared for PITCH (blue, n=12), MLP + IG without DT rules (orange, n=12), measurement-only input (green, n=7) and context-only input (purple, n=12); n denotes successfully generated responses.
Hypothesis-composition agreement was evaluated within each prompt and then averaged across three prompts: across the four LLMs for PITCH, MLP + IG and context-only, and across the LLMs that successfully generated a response for measurement-only (two or three per prompt), because no prompt has a measurement-only response from all four LLMs (agreement among fewer responses is easier to reach).
All metrics used 0-10 scales. In the main panel context-only is drawn on all axes; feature comprehensiveness and mechanistic synthesis are defined on supplied focus / contextual features and do not apply to context-only (`FigS15_ablation_radar_applicable_axes` omits them for that condition).

| Quantity | How it is computed (`plot_FigS15.py`) |
|---|---|
| Five criteria | each selected answer was judged twice by the same blind judge (gpt-5.6-sol); answer score = mean of the two passes; condition value = mean over its answers |
| n | PITCH 12 and MLP + IG 12 (4 LLMs x 3 prompts); Measurement-only 7 (the answers that committed to a hypothesis: GLM 3, Qwen 2, DeepSeek 1, Gemini 1); Context-only 12 (4 LLMs x 3 prompts) |
| Hypothesis composition agreement | judged among the LLMs within each prompt (3 passes, mean), then mean over the 3 prompts: PITCH 7.67, MLP + IG 4.33, Measurement-only 2.89 (CITE-seq GLM-5.3 + Qwen3.8; CosMx GLM-5.3 + DeepSeek V4.1 Flash + Qwen3.8; Gouwens Gemini-3.1-pro-preview + GLM-5.3; code `04c_judge_hypothesis_composition_agreement_available_models.py`, records `5_evaluation_results/ablation_conditions/4_hypothesis_composition_agreement/data_available_models/`), Context-only 3.44 (4 LLMs per prompt) |
| Axis order | Literature Grounding, Biological Plausibility, Hypothesis Specificity, Mechanistic Synthesis, Feature Comprehensiveness, Hypothesis composition agreement |
| Prompts | CITE-seq CD45RA, CosMx Ki-67, Gouwens input resistance (3 prompts) |
| LLMs | Gemini-3.1-pro-preview, GLM-5.3, DeepSeek V4.1 Flash, Qwen3.8-2.4T-A95B |

Which answer entered each cell: one randomly picked committed answer per (LLM, condition, prompt) (seed 20, `4_evaluation_code/ablation_conditions/02a_select_answers.py`); every attempt, the screening verdict
(committed / tentative / declined) and the selected answers are in `5_evaluation_results/ablation_conditions/answer_screening.csv` and `3_answers/ablation_conditions/`. Judge scorecards:
`5_evaluation_results/ablation_conditions/2_quality_radar/scorecards/`. Measurement-only cells: decile-stratified sample of the target, seed 42, 45 / 152 / 77 cells, code and row indices in
`4_evaluation_code/ablation_conditions/sampling/`.

What the four conditions were given (see `2_prompts/ablation_conditions/README.md`): the dataset and predicted response are fixed, but only PITCH names a focus feature (GNLY, PTMA, Fxyd6) and asks why it
attributes to the target, and only PITCH requests four contextual features; MLP + IG and measurement-only ask the LLM to identify the relevant features itself; context-only gives no data and no honesty / abstention paragraph and asks for one specific mechanism.
Two criteria deserve care when reading the figure: Hypothesis Specificity and Mechanistic Synthesis correlate r = 0.96 across answers, and neither tests whether the data support the hypothesis, so a context-only answer can score as high as a PITCH answer on both; the hypothesis composition agreement is the axis that depends on the data.

## Figure S16. Effects of classifier choice and decision-tree settings on PITCH hypothesis generation
(A) Logistic regression (LR), decision tree (DT), and random forest (RF) achieve comparable out-of-fold balanced accuracy in predicting IG-derived labels (bars: mean ± SD across the CITE-seq, CosMx and Gouwens datasets; points: individual datasets; dashed line: 50%, the balanced-accuracy no-discrimination baseline).
(n = 3 empirical prompts) (B) Hypothesis-generation performance remains comparable across classifiers using the 5 criteria metric. (C) Pairwise comparisons show
substantial agreement among hypotheses generated using different classifiers. Therefore, (B) and (C) support flexibility in classifier choice according to
analytical priorities. The default setting comprises 30 candidate features and a maximum tree depth of 4. (D) Varying DT depth and candidate feature-pool size
yields similar hypothesis-generation quality across six criteria (n = 3 datasets); (E) Pairwise hypothesis similarity shows substantial similarity between the
tested settings on a ten-point scale (n = 3 datasets).

| Panel | Data | Computation | Datasets |
|---|---|---|---|
| A | `FigS16A_oof_predictions.csv.gz` | balanced accuracy of each classifier from the out-of-fold predictions of the stratified 5-fold CV (seed 42, identical folds for LR, DT, RF), per fold, mean of the 5 folds per dataset; bar = mean ± SD (n-1) over the 3 datasets, dots = the 3 dataset values; dashed line = 50% (all-background or random prediction) | 3: CITE-seq CD45RA, CosMx Ki-67, Gouwens input resistance (pERK EGF10 predictions are in the file but not plotted) |
| B | `FigS16B_scores_per_case.csv` | five criteria (Feature Comprehensiveness, Biological Plausibility, Mechanistic Synthesis, Hypothesis Specificity, Literature Grounding); mean over datasets | 3 (rows with `Plotted in Fig S16B` = True; pERK EGF10 is in the table but not plotted) |
| C | `FigS16C_similarity_per_case.csv` | Hypothesis Similarity = mean of the six pairwise criteria, then mean over datasets | 3 (`Plotted in Fig S16C`) |
| D | `FigS16D_scores_per_setting.csv` | six criteria (B's five plus Logic Rigor); mean over datasets | 3: CITE-seq CD45RA, CosMx Ki-67, Gouwens input resistance |
| E | `FigS16E_similarity_between_settings.csv` | Hypothesis Similarity = mean of three pairwise criteria (Feature Interpretation & Coverage, Mechanistic Hypothesis, Functional/Causal-Role Similarity), then mean over datasets | 3 |

Setting names in the data: `30 genes, depth 4`, `60 genes, depth 4`, `30 genes, depth 6`, `60 genes, depth 6` (30/60 candidate features x maximum depth 4/6; default 30 genes, depth 4).

## Plotting command for Figure S16A (balanced accuracy)
```
cd FigS16_classifier_and_DT_settings
python plot_FigS16.py        # writes FigS16A_balanced_accuracy.svg / .pdf / .png (300 dpi) and data/FigS16A_plot_values.csv; also panels B-E
```
The out-of-fold predictions were produced by `records/classifier_comparison/heldout_evaluation/run_heldout_evaluation.py` (same data, label, features, classifier settings and folds as the original comparison; reproduces the original per-fold accuracies exactly). Positive-class prevalence is 10% in every dataset, so an all-background classifier has 90% accuracy and 50% balanced accuracy; plain accuracy of the classifiers (87-91%) is therefore at the all-background level and is not used as the measure of detection (see `data/FigS16A_heldout_metrics_by_dataset.csv`: positive-class precision 0.48-0.54 and recall 0.70-0.77 on average).

## Facts a reader should know before relying on a panel
- The classifiers are compared on the same IG-derived binary label (top 10% of positive IG attribution = class 1) and the same candidate features, using stratified 5-fold
  cross-validation over all cells (seed 42; class weights {0: 1, 1: 4}; LR standardized inside each fold; DT depth 4; RF 200 trees, depth 4). The MLP, IG ranking and
  label are computed once on all cells before the cross-validation. The out-of-fold prediction of every cell (class and positive-class probability, with its fold) is saved in `data/FigS16A_oof_predictions.csv.gz`; panel A therefore measures how well each classifier reproduces the IG-derived label on held-out cells, not an independent test of the label.
- One answer was generated per (dataset, classifier) and per (dataset, setting) with Gemini-3.1-pro-preview (temperature 0.3 in the saved scripts). The Literature
  Grounding score of the classifier comparison is the best of three judge calls (`records/classifier_comparison/judge_outputs/literature_grounding_bestof3_log.json`
  holds all three).
- The judge and pairwise calls were not seeded; the A/B order of each pairwise comparison was randomized.
- pERK EGF10 uses a curated 22-feature panel instead of an IG-ranked list; its rows are in the data tables but it is not plotted in any panel of S16 (all panels use the same 3 datasets).
- The scripts in `records/*/code/` are the generation and judging code; they read the API keys from environment variables and need the original datasets.

## Figure S16: where each required record is
| Record | Location |
|---|---|
| Datasets, prompts, predicted response, target marker | `records/classifier_comparison/answers_and_prompts/<dataset>/` (`*_prompt.txt`); `records/DT_settings/answers_and_prompts/<dataset>/`. S16A-C focus features: GNLY (CITE-seq CD45RA), PTMA (CosMx Ki-67), Fxyd6 (Gouwens input resistance), Cytoplasm Intensity mean Actin (pERK EGF10); the same prompts name the focus feature for LR, DT and RF. |
| Exact LR, DT and RF representations given to the LLM | the prompts above; the machine-readable evidence is `*_evidence.json` (LR: standardized coefficients ranked by absolute value; DT: tree rules and node statistics; RF: mean decrease in impurity over 200 trees, top-ranked first); builder `records/classifier_comparison/code/prompt_builder.py` |
| Candidate feature lists, ranking, 30 / 60 features, depth 4 / 6, rules | `records/DT_settings/answers_and_prompts/<dataset>/*_nodes.json` and `*_primaryonly_prompt.txt` (ranking = mean absolute IG; `code/config.py`: 4 settings, `min_samples_leaf` 5, no leaf cap, `random_state` 42) |
| Generated responses (one per dataset x classifier / setting) | `*_hypothesis.jsonl` next to the prompts; generation scripts `code/generate_hypotheses*.py` (temperature 0.3) |
| Classifier evaluation: splits, prevalence, hyperparameters, class weights, threshold, per-fold results | `records/classifier_comparison/code/evaluate_classifier_performance.py`, `config.py`, `classifier_perf_run.log`, `data/FigS16A_accuracy_by_fold.csv` |
| Judge prompts, scores | `records/classifier_comparison/code/judge_evaluation.py`, `run_new_rubric_openai.py`, `judge_outputs/`; `records/DT_settings/code/judge_evaluation.py`, `judge_outputs/judge_6criteria/`; individual scores in `data/FigS16B/C/D/E_*.csv` |
| Pairwise similarity scores (S16C, S16E) | `data/FigS16C_similarity_per_case.csv`, `data/FigS16E_similarity_between_settings.csv`; raw judgements in `judge_outputs/` |
| Averaging, uncertainty, plots | `plot_FigS16.py` and the table above |

## Legacy fields
Older scorecards (benchmark scores and the ablation scorecards) still hold a zone-identification field written by the earlier seven-criterion judge. It is excluded from all reported analyses; none of the data tables or plots here uses it.
