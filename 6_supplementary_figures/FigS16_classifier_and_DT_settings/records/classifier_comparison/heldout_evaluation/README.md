# Majority-class baseline and held-out positive-class performance (Figure S16A follow-up)

`run_heldout_evaluation.py` re-runs the classifier comparison with the same data, MLP + IG step, features, binary label and classifier settings as `../evaluate_classifier_performance.py`
and the **same stratified 5-fold splits** (`StratifiedKFold(5, shuffle=True, random_state=42)`, identical folds for LR, DT and RF), and saves every out-of-fold prediction.
The accuracies recomputed here equal the original per-fold accuracies (max difference 1e-16 over 60 fold results, `out/reproduction_check.csv`).

## What the scores are
All scores are out-of-fold: each cell is scored by a model that was trained on the other four folds. The MLP + IG ranking, the candidate features and the label itself
(top 10% of positive IG attribution = positive class) were computed once on all cells before the split, so the scores measure how well each classifier reproduces the IG-derived label
on unseen cells; they are not an independent test of the label.

## Results (CITE-seq CD45RA, CosMx Ki-67, Gouwens input resistance; mean of 5 folds per dataset)
Positive prevalence is 10.0% in every dataset (862/8,617; 3,000/30,000; 340/3,395), so an all-background classifier has 90.0% accuracy, 0.5 balanced accuracy and 0 positive recall.

| Dataset | Classifier | Accuracy | Balanced accuracy | Positive precision | Positive recall |
|---|---|---|---|---|---|
| CITE-seq CD45RA | LR / DT / RF | 0.970 / 0.966 / 0.969 | 0.970 / 0.965 / 0.971 | 0.78 / 0.77 / 0.77 | 0.97 / 0.96 / 0.97 |
| CosMx Ki-67 | LR / DT / RF | 0.887 / 0.844 / 0.877 | 0.776 / 0.757 / 0.770 | 0.46 / 0.35 / 0.42 | 0.64 / 0.65 / 0.64 |
| Gouwens input resistance | LR / DT / RF | 0.842 / 0.810 / 0.877 | 0.757 / 0.761 / 0.703 | 0.35 / 0.31 / 0.41 | 0.65 / 0.70 / 0.49 |

Mean +/- SD across the three datasets (`out/summary_across_datasets.csv`):

| | Accuracy | Balanced accuracy | Positive precision | Positive recall |
|---|---|---|---|---|
| All-background | 0.900 | 0.500 | - | 0.00 |
| LR | 0.900 +/- 0.065 | 0.834 +/- 0.118 | 0.53 +/- 0.23 | 0.75 +/- 0.19 |
| DT | 0.874 +/- 0.082 | 0.828 +/- 0.119 | 0.48 +/- 0.25 | 0.77 +/- 0.17 |
| RF | 0.908 +/- 0.053 | 0.815 +/- 0.139 | 0.54 +/- 0.21 | 0.70 +/- 0.25 |

pERK EGF10 (not part of Figure S16) is in `out/metrics_by_dataset.csv`: accuracy 0.89 for all three, balanced accuracy 0.80-0.81, precision 0.47-0.48, recall 0.69-0.71.

## Reading
- The mean accuracy of LR (0.900), DT (0.874) and RF (0.908) is at the all-background level (0.900). In CosMx and Gouwens every classifier is below it (0.81-0.89), because the positive class carries a weight of 4 in all three classifiers, which gives up accuracy to find positives.
- The positive class is detected clearly better than chance: balanced accuracy 0.82-0.83 against 0.5, with recall 0.70-0.77. Precision is moderate (0.48-0.54), i.e. about half of the predicted positives are positive, against a 10% prevalence.
- The three classifiers are similar in balanced accuracy (0.815-0.834, SD across datasets 0.12-0.14). Dataset matters more than classifier: CITE-seq is near 0.97, CosMx and Gouwens 0.70-0.78.

## Files
`out/oof_predictions.csv.gz` (cell, fold, true label, predicted class and positive-class probability for LR, DT, RF), `out/fold_assignments.csv`, `out/majority_baseline.csv`, `out/metrics_by_fold.csv`, `out/metrics_by_dataset.csv`, `out/summary_across_datasets.csv`, `out/reproduction_check.csv`, `run_log.txt`.
Cell indices are row positions in the data after loading with `pipeline.load_data` (the datasets are not in the repository).
