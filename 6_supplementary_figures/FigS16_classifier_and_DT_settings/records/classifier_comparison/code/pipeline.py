"""Headless MLP -> IG -> {LogReg, Decision Tree, Random Forest} -> LLM-prompt
pipeline. Shares its MLP/IG/tree/node-report logic with
DTEvaluation/pipeline.py (copied, not imported, to keep the two evaluation
studies independently runnable/editable); adds Logistic Regression and
Random Forest fitting + evidence extraction, and a manual-feature-selection
data loader for the pERK/EGF dataset (mirrors classification.py).
"""
import math

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from captum.attr import IntegratedGradients
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text

import config


# --- 1. MODEL DEFINITION (copied from app_v4.py / DTEvaluation) -----------
class MLPRegression(nn.Module):
    def __init__(self, input_size, hidden_size, dropout):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        self.fc2 = nn.Linear(hidden_size, 1)
        self.initialize()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x.squeeze()

    def initialize(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.constant_(self.fc1.bias, 0.0)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.constant_(self.fc2.bias, 0.0)


# --- 2a. DATA / TRAINING / IG -- ig_auto datasets (xlsx, own MLP) ---------
def load_data(path):
    df = pd.read_excel(path)
    headers = df.columns[:-1].values
    target_name = df.columns[-1]
    X_raw = df.iloc[:, :-1].values.astype(np.float32)
    y_raw = df.iloc[:, -1].values.astype(np.float32)
    return X_raw, y_raw, headers, target_name


def train_mlp(X_raw, y_raw, params=config.MLP_PARAMS, test_size=config.TEST_SIZE):
    seed = params["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)

    X_train, X_test, y_train, y_test = train_test_split(
        X_raw, y_raw, test_size=test_size, random_state=seed
    )

    model = MLPRegression(X_raw.shape[1], params["hidden_size"], params["dropout"])
    optimizer = optim.Adam(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])
    criterion = nn.MSELoss()

    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.float32)

    model.train()
    for _ in range(params["epochs"] + 1):
        optimizer.zero_grad()
        preds = model(X_t)
        loss = criterion(preds, y_t)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        test_preds = model(torch.tensor(X_test, dtype=torch.float32)).numpy()
    corr, _ = pearsonr(y_test, test_preds)
    return model, corr


def compute_ig(model, X_raw, n_steps=config.IG_N_STEPS):
    model.eval()
    ig = IntegratedGradients(model, multiply_by_inputs=False)
    W_tensor = ig.attribute(torch.from_numpy(X_raw), n_steps=n_steps)
    W = W_tensor.detach().numpy()
    attr = X_raw * W
    return attr


def pick_focused_feature(attr, headers):
    """Top-ranked feature by mean |IG attribution| (ig_auto datasets only)."""
    avg_abs = np.mean(np.abs(attr), axis=0)
    ranking = np.argsort(avg_abs)[::-1]
    return ranking, headers[ranking[0]]


def select_attributing_features(ranking, headers, target_prot, n_features):
    top_names = headers[ranking[:n_features]].tolist()
    return [f for f in top_names if f != target_prot]


# --- 2b. DATA -- manual dataset (pERK/EGF, precomputed IG) -----------------
def load_pERK_data(ds_meta):
    """Reconstructs X_raw / y_raw / headers / attr / attr_features for the
    pERK/EGF dataset exactly as classification.py does, but also recovers
    the continuous target (LogResponseData column) so correlation_direction
    can be computed the same way as for the ig_auto datasets. classification.py
    never loads y itself (it only needs the precomputed attr column to build
    the binary label), so this alignment was verified independently against
    Data_1.mat before writing this function: filtering LinearIndex to
    linear_index_keep gives exactly len(FeatureData-z-510.npy) rows in the
    same order, and the reconstructed feature_idx=75 vs. pERK correlation
    comes out positive, matching the pre-existing
    gemini_3.1_pro_hypothesis/data/pERK_EGF10_DT_prompt.txt.
    """
    mat = h5py.File(str(ds_meta["mat_path"]), "r")
    LinearIndex_raw = np.array(mat["LinearIndex"]).T.squeeze()
    LogResponseData_raw = np.array(mat["LogResponseData"]).T
    FeatureData = np.load(str(ds_meta["feature_data_path"]))

    con_idx = ds_meta["concentrate_idx"]
    response_idx = ds_meta["response_idx"]
    feature_idx = ds_meta["feature_idx"]

    data_idx1 = np.where(np.isin(LinearIndex_raw, ds_meta["linear_index_keep"]))[0]
    LinearIndex = LinearIndex_raw[data_idx1]
    LogResponseData = LogResponseData_raw[data_idx1, :]

    attr_dir = ds_meta["attr_dir"]
    conc_val = ds_meta["concentration_values"][con_idx]
    d = FeatureData.shape[1]
    attr_full = np.load(str(attr_dir / f"attr_{response_idx}-{conc_val}-dim{d}-z.npy"))
    rand_int = np.load(str(attr_dir / f"rand_int_{response_idx}-{conc_val}-dim{d}-z.npy"))

    data_idx2 = np.where(np.isin(LinearIndex, ds_meta["datanum_by_conc"][con_idx]))[0]
    X_full = FeatureData[data_idx2, :][rand_int, :]
    y_full = LogResponseData[data_idx2, response_idx][rand_int]
    attr_full = attr_full  # already produced from the same data_idx2/rand_int selection

    with open(str(ds_meta["feature_name_path"]), "r") as f:
        all_names = [line.strip() for line in f.readlines()]

    use_list = list(ds_meta["use_list"])
    headers = np.array([all_names[i] for i in use_list])
    X_raw = X_full[:, use_list].astype(np.float32)
    attr = attr_full[:, use_list]
    y_raw = y_full.astype(np.float32)

    target_prot = all_names[feature_idx]
    assert target_prot in headers, f"feature_idx {feature_idx} not in use_list"

    attr_features = [all_names[i] for i in use_list if i != feature_idx] if ds_meta["self_exclude"] \
        else [all_names[i] for i in use_list]

    target_name = ds_meta["target_name"]
    return X_raw, y_raw, headers, target_name, attr, target_prot, attr_features


# --- 3. FOCUSED FEATURE + BINARY LABEL (app_v4.py Step 3 defaults) --------
def make_binary_label(attr, headers, target_prot, direction=config.ATTR_DIRECTION,
                       percentile=config.IG_PERCENTILE):
    target_idx = np.where(headers == target_prot)[0][0]
    y_ig_raw = attr[:, target_idx]
    if direction == "Positive":
        thresh = np.percentile(y_ig_raw, percentile)
        y_binary = np.where(y_ig_raw > thresh, 1, 0)
    else:
        thresh = np.percentile(y_ig_raw, 100 - percentile)
        y_binary = np.where(y_ig_raw < thresh, 1, 0)
    return y_binary, y_ig_raw


def correlation_direction(X_raw, y_raw, headers, target_prot):
    idx = np.where(headers == target_prot)[0][0]
    r, _ = pearsonr(X_raw[:, idx], y_raw)
    return "Direct" if r >= 0 else "Inverse", r


# --- 4a. DECISION TREE + PER-NODE QUANTITATIVE EVIDENCE (from DTEvaluation) -
def fit_decision_tree(X_raw, headers, attr_features, y_binary, max_depth=config.DT_MAX_DEPTH,
                       fixed_params=config.DT_FIXED_PARAMS, class_1_weight=config.CLASS_1_WEIGHT):
    indices = [np.where(headers == f)[0][0] for f in attr_features]
    X_dt = X_raw[:, indices]
    clf = DecisionTreeClassifier(
        max_depth=max_depth,
        class_weight={0: 1, 1: class_1_weight},
        **fixed_params,
    )
    clf.fit(X_dt, y_binary)
    cv_scores = cross_val_score(clf, X_dt, y_binary, cv=5, scoring="f1_macro")
    return clf, X_dt, cv_scores.mean()


def _get_node_path(tree, feature_names, target_node):
    left, right, threshold = tree.children_left, tree.children_right, tree.threshold
    features = [feature_names[i] if i >= 0 else "undefined" for i in tree.feature]
    total_samples = tree.weighted_n_node_samples[0]

    def local_importance(node):
        n_node = tree.weighted_n_node_samples[node]
        n_l, n_r = tree.weighted_n_node_samples[left[node]], tree.weighted_n_node_samples[right[node]]
        g_node, g_l, g_r = tree.impurity[node], tree.impurity[left[node]], tree.impurity[right[node]]
        weighted_child_impurity = (n_l / n_node) * g_l + (n_r / n_node) * g_r
        return float(max(0.0, (n_node / total_samples) * (g_node - weighted_child_impurity)))

    def recurse(node, path, depth):
        if node == target_node:
            return path, depth
        if left[node] != -1:
            imp = local_importance(node)
            step_l = {"feature": features[node], "condition": f"{features[node]} <= {threshold[node]:.3f}", "local_importance": imp}
            res = recurse(left[node], path + [step_l], depth + 1)
            if res:
                return res
            step_r = {"feature": features[node], "condition": f"{features[node]} > {threshold[node]:.3f}", "local_importance": imp}
            res = recurse(right[node], path + [step_r], depth + 1)
            if res:
                return res
        return None

    return recurse(0, [], 0)


def build_node_report(clf, X_dt, y_binary, attr_features, target_ig_scores, total_dataset_size):
    """Per-leaf quantitative evidence table (identical spec to DTEvaluation's
    build_node_report): split conditions, depth, support, Gini impurity/
    reduction, target-cell lift, IG enrichment. Used for the Decision Tree
    model directly, and for Random Forest's single representative tree."""
    t_ = clf.tree_
    total_samples = len(y_binary)
    overall_target_rate = float(np.mean(y_binary))
    overall_mean_ig = float(np.mean(target_ig_scores))
    overall_std_ig = float(np.std(target_ig_scores))

    full_paths = clf.apply(X_dt)
    is_leaf = t_.children_left == -1

    parent_of = {}
    for node in range(t_.node_count):
        if t_.children_left[node] != -1:
            parent_of[t_.children_left[node]] = node
            parent_of[t_.children_right[node]] = node

    def split_gini_reduction(node):
        if node not in parent_of:
            return 0.0
        p = parent_of[node]
        n_p = t_.weighted_n_node_samples[p]
        n_l = t_.weighted_n_node_samples[t_.children_left[p]]
        n_r = t_.weighted_n_node_samples[t_.children_right[p]]
        g_p, g_l, g_r = t_.impurity[p], t_.impurity[t_.children_left[p]], t_.impurity[t_.children_right[p]]
        return float(max(0.0, g_p - (n_l / n_p) * g_l - (n_r / n_p) * g_r))

    min_support = max(config.MIN_SUPPORT_ABS, math.ceil(config.MIN_SUPPORT_FRACTION * total_dataset_size))

    records = []
    for node in range(t_.node_count):
        if not is_leaf[node]:
            continue
        in_node = full_paths == node
        n = int(np.sum(in_node))
        if n == 0:
            continue

        tp = int(np.sum(in_node & (y_binary == 1)))
        target_fraction = tp / n
        lift = target_fraction / overall_target_rate if overall_target_rate > 0 else float("nan")

        node_mean_ig = float(np.mean(target_ig_scores[in_node]))
        ig_enrichment_sd = (node_mean_ig - overall_mean_ig) / overall_std_ig if overall_std_ig > 0 else float("nan")

        path, depth = _get_node_path(t_, attr_features, node)
        rule_str = " AND ".join(step["condition"] for step in path) if path else "Root Node"
        feature_confidence = sorted(path, key=lambda s: -s["local_importance"])

        records.append({
            "node_id": int(node),
            "rule": rule_str,
            "feature_confidence": feature_confidence,
            "depth": depth,
            "n_cells": n,
            "pop_fraction": n / total_samples,
            "gini_impurity": float(t_.impurity[node]),
            "gini_reduction": split_gini_reduction(node),
            "target_fraction": target_fraction,
            "overall_target_rate": overall_target_rate,
            "target_lift": lift,
            "mean_ig": node_mean_ig,
            "overall_mean_ig": overall_mean_ig,
            "overall_std_ig": overall_std_ig,
            "ig_enrichment_sd": ig_enrichment_sd,
            "flag_low_support": n < min_support,
        })

    return records, min_support


# --- 4b. RANDOM FOREST: forest-wide importances + 1 representative tree ---
def fit_random_forest(X_raw, headers, attr_features, y_binary,
                       n_estimators=config.RF_N_ESTIMATORS, max_depth=config.RF_MAX_DEPTH,
                       fixed_params=config.RF_FIXED_PARAMS, class_1_weight=config.CLASS_1_WEIGHT):
    indices = [np.where(headers == f)[0][0] for f in attr_features]
    X = X_raw[:, indices]
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight={0: 1, 1: class_1_weight},
        **fixed_params,
    )
    clf.fit(X, y_binary)
    cv_scores = cross_val_score(clf, X, y_binary, cv=5, scoring="f1_macro")

    # A single shallow tree (same depth/leaf/weight budget as the standalone
    # Decision Tree run) fit on the identical data, kept ONLY to produce
    # human-readable rule "zones" for the LLM prompt. The authoritative
    # ranking is the forest-wide feature_importances_ below -- the
    # representative tree is explicitly labeled as illustrative in the prompt.
    rep_tree = DecisionTreeClassifier(
        max_depth=max_depth,
        class_weight={0: 1, 1: class_1_weight},
        **config.DT_FIXED_PARAMS,
    )
    rep_tree.fit(X, y_binary)

    return clf, rep_tree, X, cv_scores.mean()


def build_forest_importance_ranking(rf_clf, attr_features):
    importances = rf_clf.feature_importances_
    ranking = sorted(zip(attr_features, importances), key=lambda t: -t[1])
    return ranking


# --- 4c. LOGISTIC REGRESSION: standardized coefficients --------------------
def fit_logistic_regression(X_raw, headers, attr_features, y_binary,
                             fixed_params=config.LOGREG_FIXED_PARAMS, class_1_weight=config.CLASS_1_WEIGHT):
    indices = [np.where(headers == f)[0][0] for f in attr_features]
    X = X_raw[:, indices]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    clf = LogisticRegression(class_weight={0: 1, 1: class_1_weight}, **fixed_params)
    clf.fit(X_scaled, y_binary)

    # cross_val_score must re-fit the scaler on each training fold to avoid
    # leaking test-fold statistics into the standardization.
    cv_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(class_weight={0: 1, 1: class_1_weight}, **fixed_params)),
    ])
    cv_scores = cross_val_score(cv_pipe, X, y_binary, cv=5, scoring="f1_macro")

    return clf, X_scaled, cv_scores.mean()


def build_logreg_coefficient_ranking(clf, attr_features):
    coefs = clf.coef_[0]
    ranking = sorted(zip(attr_features, coefs), key=lambda t: -abs(t[1]))
    return ranking
