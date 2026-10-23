"""Majority-class baseline and held-out positive-class performance of LR, DT and RF (Figure S16A follow-up).

Re-uses the data loading, MLP + IG, feature selection, binary label and classifier definitions of the original classifier comparison
(../evaluate_classifier_performance.py, ../pipeline.py, ../config.py) and the SAME stratified 5-fold splits
(StratifiedKFold(n_splits=5, shuffle=True, random_state=42), the same folds for LR, DT and RF).

For every cell, the out-of-fold prediction (the class and the positive-class probability from the model that did not see that cell) is saved.
From these predictions: positive-class prevalence, the all-background (majority-class) accuracy, and accuracy, balanced accuracy, positive-class
precision, recall and F1 per classifier and dataset (mean of the 5 folds, and pooled over all out-of-fold predictions).

Outputs (out/):
  oof_predictions.csv.gz        one row per cell: dataset, cell_index, fold, y_true, <clf>_pred, <clf>_prob
  fold_assignments.csv          dataset, cell_index, fold  (the splits)
  majority_baseline.csv         prevalence and all-background accuracy per dataset
  metrics_by_fold.csv           per dataset x classifier x fold
  metrics_by_dataset.csv        per dataset x classifier (mean over folds; pooled out-of-fold values in *_pooled columns)
  summary_across_datasets.csv   mean +/- SD (ddof = 1) across the three datasets of Figure S16 (pERK EGF10 is computed but excluded)
  reproduction_check.csv        accuracy recomputed here vs the original per-fold accuracy file

Usage:  python run_heldout_evaluation.py        (needs the original datasets; the MLP + IG step takes a few minutes per dataset)
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import config  # noqa: E402
from evaluate_classifier_performance import CV_SEED, N_SPLITS, build_classifier, load_dataset  # noqa: E402

OUT = HERE / "out"
OUT.mkdir(exist_ok=True)
CLASSIFIERS = [("logreg", "LR"), ("dt", "DT"), ("rf", "RF")]
FIG_DATASETS = ["CITEseq_CD45RA_naive", "CosMx_CancerKi67", "Gouwens_GABAergic"]          # the three datasets used in Figure S16
ORIGINAL_FOLDS = HERE.parent / "new_rubric_results" / "FINAL_organized" / "0_classifier_performance" / "classifier_performance_by_fold.csv"


def metrics(y, p):
    return {
        "accuracy": accuracy_score(y, p),
        "balanced_accuracy": balanced_accuracy_score(y, p),
        "positive_precision": precision_score(y, p, pos_label=1, zero_division=0),
        "positive_recall": recall_score(y, p, pos_label=1, zero_division=0),
        "positive_f1": f1_score(y, p, pos_label=1, zero_division=0),
        "n_predicted_positive": int((p == 1).sum()),
    }


def main():
    oof_frames, fold_frames, base_rows, fold_rows, pooled = [], [], [], [], []
    for ds in config.DATASETS:
        key = ds["key"]
        t0 = time.time()
        X, y, features, target = load_dataset(ds)
        y = np.asarray(y).astype(int)
        n, n_pos = len(y), int(y.sum())
        print(f"\n=== {key}: {n} cells x {X.shape[1]} features, target={target}, positives {n_pos} ({n_pos / n:.2%}) [{time.time() - t0:.0f}s]")
        base_rows.append({"dataset": key, "n_cells": n, "n_positive": n_pos, "positive_prevalence": n_pos / n,
                          "all_background_accuracy": 1 - n_pos / n, "all_background_balanced_accuracy": 0.5,
                          "all_background_positive_recall": 0.0, "used_in_figure_S16": key in FIG_DATASETS})

        cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=CV_SEED)
        folds = list(cv.split(X, y))
        fold_id = np.empty(n, dtype=int)
        for k, (_, te) in enumerate(folds, start=1):
            fold_id[te] = k
        oof = pd.DataFrame({"dataset": key, "cell_index": np.arange(n), "fold": fold_id, "y_true": y})
        fold_frames.append(oof[["dataset", "cell_index", "fold"]])

        for ckey, clabel in CLASSIFIERS:
            pred = np.zeros(n, dtype=int)
            prob = np.zeros(n)
            for k, (tr, te) in enumerate(folds, start=1):
                m = clone(build_classifier(ckey)).fit(X[tr], y[tr])
                pred[te] = m.predict(X[te])
                prob[te] = m.predict_proba(X[te])[:, 1]
                fold_rows.append({"dataset": key, "classifier": clabel, "fold": k, "n_test": len(te), "n_test_positive": int(y[te].sum()),
                                  **metrics(y[te], pred[te])})
            oof[f"{clabel}_pred"] = pred
            oof[f"{clabel}_prob"] = prob.round(6)
            pooled.append({"dataset": key, "classifier": clabel, **{f"{k}_pooled": v for k, v in metrics(y, pred).items() if k != "n_predicted_positive"}})
            print(f"  [{clabel}] pooled acc={accuracy_score(y, pred):.3f} bal.acc={balanced_accuracy_score(y, pred):.3f} "
                  f"precision={precision_score(y, pred, zero_division=0):.3f} recall={recall_score(y, pred):.3f}")
        oof_frames.append(oof)
        print(f"  done in {time.time() - t0:.0f}s")

    pd.concat(oof_frames).to_csv(OUT / "oof_predictions.csv.gz", index=False)
    pd.concat(fold_frames).to_csv(OUT / "fold_assignments.csv", index=False)
    base = pd.DataFrame(base_rows)
    base.to_csv(OUT / "majority_baseline.csv", index=False)
    byfold = pd.DataFrame(fold_rows)
    byfold.to_csv(OUT / "metrics_by_fold.csv", index=False)
    cols = ["accuracy", "balanced_accuracy", "positive_precision", "positive_recall", "positive_f1"]
    bydata = byfold.groupby(["dataset", "classifier"], sort=False)[cols].mean().reset_index().merge(pd.DataFrame(pooled), on=["dataset", "classifier"])
    bydata = bydata.merge(base[["dataset", "positive_prevalence", "all_background_accuracy"]], on="dataset")
    bydata["accuracy_minus_all_background"] = bydata["accuracy"] - bydata["all_background_accuracy"]
    bydata["folds_with_no_predicted_positive"] = byfold.assign(z=byfold.n_predicted_positive == 0).groupby(["dataset", "classifier"], sort=False)["z"].sum().values
    bydata.to_csv(OUT / "metrics_by_dataset.csv", index=False)

    fig = bydata[bydata.dataset.isin(FIG_DATASETS)]
    rows = [{"classifier": "All-background", **{"accuracy_mean": base[base.used_in_figure_S16].all_background_accuracy.mean(),
                                                  "accuracy_sd": base[base.used_in_figure_S16].all_background_accuracy.std(ddof=1),
                                                  "balanced_accuracy_mean": 0.5, "balanced_accuracy_sd": 0.0,
                                                  "positive_precision_mean": np.nan, "positive_precision_sd": np.nan,
                                                  "positive_recall_mean": 0.0, "positive_recall_sd": 0.0,
                                                  "positive_prevalence_mean": base[base.used_in_figure_S16].positive_prevalence.mean(), "n_datasets": 3}}]
    for _, clabel in CLASSIFIERS:
        g = fig[fig.classifier == clabel]
        r = {"classifier": clabel}
        for c in ["accuracy", "balanced_accuracy", "positive_precision", "positive_recall", "positive_f1", "accuracy_minus_all_background"]:
            r[f"{c}_mean"], r[f"{c}_sd"] = g[c].mean(), g[c].std(ddof=1)
        r["positive_prevalence_mean"], r["n_datasets"] = g["positive_prevalence"].mean(), len(g)
        rows.append(r)
    pd.DataFrame(rows).to_csv(OUT / "summary_across_datasets.csv", index=False)

    orig = pd.read_csv(ORIGINAL_FOLDS)
    lab = {"logreg": "LR", "dt": "DT", "rf": "RF"}
    orig["classifier"] = orig["classifier"].map(lab)
    ds_map = {k: k for k in orig.dataset.unique()}
    chk = byfold.merge(orig.rename(columns={"accuracy": "accuracy_original"})[["dataset", "classifier", "fold", "accuracy_original"]], on=["dataset", "classifier", "fold"])
    chk["abs_diff"] = (chk["accuracy"] - chk["accuracy_original"]).abs()
    chk[["dataset", "classifier", "fold", "accuracy", "accuracy_original", "abs_diff"]].to_csv(OUT / "reproduction_check.csv", index=False)
    print(f"\nReproduction check vs original per-fold accuracies: max |difference| = {chk.abs_diff.max():.2e} over {len(chk)} fold results")
    print(pd.DataFrame(rows).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
