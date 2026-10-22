"""Classifier-quality evaluation (Logistic Regression vs Decision Tree vs
Random Forest), addressing the reviewer question: "the manuscript motivates
decision tree ... but does not establish that this representation is
necessary or superior ... compare decision tree with simpler interpretable
classifiers and stronger predictive classifiers, evaluating classification
performance."

Reuses pipeline.py's EXACT data-loading / focused-feature / binary-label
logic (same MLP+IG, same y_binary, same attr_features per dataset) so the
comparison is apples-to-apples: all 3 classifiers see the same features and
the same binary label on each dataset, only the model class differs.

For each (dataset, classifier), runs stratified 5-fold CV computing:
  - Accuracy
  - F1 (macro)
  - ROC-AUC
  - Average Precision (PR-AUC)
Logistic Regression is wrapped in a Pipeline(StandardScaler, LogisticRegression)
per fold to avoid scaling leakage (same convention as pipeline.py's cv_f1).

Outputs (under new_rubric_results/FINAL_organized/0_classifier_performance/):
  - classifier_performance_by_fold.csv   (raw per-fold scores)
  - classifier_performance_summary.csv   (mean +/- std per dataset x classifier x metric)
  - logreg_threshold_example.json        (worked example of the LogReg decision rule)
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

import config
import pipeline

OUT_DIR = config.BASE_DIR / "new_rubric_results" / "FINAL_organized" / "0_classifier_performance"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SCORING = {
    "accuracy": "accuracy",
    "f1_macro": "f1_macro",
    "roc_auc": "roc_auc",
    "average_precision": "average_precision",
}
N_SPLITS = 5
CV_SEED = 42


def build_classifier(model_key):
    if model_key == "logreg":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(class_weight={0: 1, 1: config.CLASS_1_WEIGHT}, **config.LOGREG_FIXED_PARAMS)),
        ])
    if model_key == "dt":
        return DecisionTreeClassifier(
            max_depth=config.DT_MAX_DEPTH, class_weight={0: 1, 1: config.CLASS_1_WEIGHT}, **config.DT_FIXED_PARAMS
        )
    if model_key == "rf":
        return RandomForestClassifier(
            n_estimators=config.RF_N_ESTIMATORS, max_depth=config.RF_MAX_DEPTH,
            class_weight={0: 1, 1: config.CLASS_1_WEIGHT}, **config.RF_FIXED_PARAMS
        )
    raise ValueError(model_key)


def load_dataset(ds_meta):
    if ds_meta["mode"] == "ig_auto":
        X_raw, y_raw, headers, target_name = pipeline.load_data(ds_meta["path"])
        model, corr = pipeline.train_mlp(X_raw, y_raw)
        attr = pipeline.compute_ig(model, X_raw)
        ranking, target_prot = pipeline.pick_focused_feature(attr, headers)
        attr_features = pipeline.select_attributing_features(ranking, headers, target_prot, ds_meta["n_features"])
    else:
        X_raw, y_raw, headers, target_name, attr, target_prot, attr_features = pipeline.load_pERK_data(ds_meta)
    y_binary, _ = pipeline.make_binary_label(attr, headers, target_prot)
    indices = [np.where(headers == f)[0][0] for f in attr_features]
    X = X_raw[:, indices]
    return X, y_binary, attr_features, target_prot


def evaluate_dataset(ds_meta, fold_rows, logreg_example):
    key = ds_meta["key"]
    print(f"\n=== {key} (mode={ds_meta['mode']}) ===")
    t0 = time.time()
    X, y_binary, attr_features, target_prot = load_dataset(ds_meta)
    n_pos = int(y_binary.sum())
    print(f"  {X.shape[0]} cells x {X.shape[1]} features, target={target_prot}, "
          f"class balance={n_pos}/{len(y_binary)} positive ({time.time()-t0:.1f}s)")

    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=CV_SEED)

    for model_setting in config.MODEL_SETTINGS:
        model_key, model_label = model_setting["key"], model_setting["label"]
        clf = build_classifier(model_key)
        scores = cross_validate(clf, X, y_binary, cv=cv, scoring=SCORING, n_jobs=1)
        for fold_idx in range(N_SPLITS):
            row = {"dataset": key, "classifier": model_key, "classifier_label": model_label, "fold": fold_idx + 1}
            for metric in SCORING:
                row[metric] = scores[f"test_{metric}"][fold_idx]
            fold_rows.append(row)
        means = {m: scores[f"test_{m}"].mean() for m in SCORING}
        print(f"  [{model_key:7s}] " + " ".join(f"{m}={v:.3f}" for m, v in means.items()))

        if model_key == "logreg":
            full_pipe = build_classifier("logreg")
            full_pipe.fit(X, y_binary)
            lr = full_pipe.named_steps["clf"]
            scaler = full_pipe.named_steps["scaler"]
            logreg_example[key] = {
                "target_prot": target_prot,
                "attr_features": attr_features,
                "intercept": float(lr.intercept_[0]),
                "standardized_coefficients": {f: float(c) for f, c in zip(attr_features, lr.coef_[0])},
                "scaler_mean": {f: float(m) for f, m in zip(attr_features, scaler.mean_)},
                "scaler_scale": {f: float(s) for f, s in zip(attr_features, scaler.scale_)},
                "decision_rule_note": (
                    "Predicted class = 1 iff intercept + sum(coef_i * (x_i - mean_i)/scale_i) > 0 "
                    "(equivalently, predicted probability > 0.5). This IS a valid threshold rule, but "
                    "it thresholds a single linear combination of all standardized features at once, "
                    "not one feature at a time -- so it selects a half-space of cells (a hyperplane "
                    "boundary in 22-30 dimensional standardized space), not a small set of simple, "
                    "human-readable 'feature > value' conditions the way a decision-tree leaf does."
                ),
            }
    print(f"  Done with {key} in {time.time()-t0:.1f}s")


def main():
    fold_rows = []
    logreg_example = {}
    for ds_meta in config.DATASETS:
        evaluate_dataset(ds_meta, fold_rows, logreg_example)

    df = pd.DataFrame(fold_rows)
    fold_path = OUT_DIR / "classifier_performance_by_fold.csv"
    df.to_csv(fold_path, index=False)
    print("\nSaved ->", fold_path)

    summary_rows = []
    for (ds, clf), g in df.groupby(["dataset", "classifier"], sort=False):
        row = {"dataset": ds, "classifier": clf}
        for metric in SCORING:
            row[f"{metric}_mean"] = g[metric].mean()
            row[f"{metric}_std"] = g[metric].std(ddof=1)
        summary_rows.append(row)
    df_summary = pd.DataFrame(summary_rows)
    summary_path = OUT_DIR / "classifier_performance_summary.csv"
    df_summary.to_csv(summary_path, index=False)
    print("Saved ->", summary_path)

    example_path = OUT_DIR / "logreg_threshold_example.json"
    example_path.write_text(json.dumps(logreg_example, indent=2), encoding="utf-8")
    print("Saved ->", example_path)


if __name__ == "__main__":
    main()
