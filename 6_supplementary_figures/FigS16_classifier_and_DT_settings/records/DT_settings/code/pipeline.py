"""Headless MLP -> Integrated Gradients -> Decision Tree -> LLM-prompt pipeline.

Trimmed, non-Streamlit re-implementation of the model / IG / DT / prompt
logic in app_v4.py. Only the pieces needed to sweep DT (top-K features x
max_depth) settings and emit an interpretation prompt per setting are kept:
no Streamlit, no SHAP, no UMAP, no network plot, no matplotlib.
"""
import math

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from captum.attr import IntegratedGradients
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, export_text

import config


# --- 1. MODEL DEFINITION (copied from app_v4.py) ---------------------------
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


# --- 2. DATA / TRAINING / IG ------------------------------------------------
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


# --- 3. FOCUSED FEATURE + BINARY LABEL (app_v4.py Step 3 defaults) --------
def pick_focused_feature(attr, headers):
    """Top-ranked feature by mean |IG attribution| -- the app's default
    'focused feature' (index 0 of the ranking dropdown)."""
    avg_abs = np.mean(np.abs(attr), axis=0)
    ranking = np.argsort(avg_abs)[::-1]
    return ranking, headers[ranking[0]]


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


# --- 4. DECISION TREE + PER-NODE QUANTITATIVE EVIDENCE ---------------------
def select_attributing_features(ranking, headers, target_prot, n_features):
    top_names = headers[ranking[:n_features]].tolist()
    return [f for f in top_names if f != target_prot]


def fit_tree(X_raw, headers, attr_features, y_binary, max_depth,
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
    """Returns (steps, depth) for the path from root to target_node. Each step
    is {feature, condition, local_importance}: local_importance is that
    ancestor split's own weighted Gini-impurity decrease (the same quantity
    sklearn sums across the whole tree for feature_importances_, but kept
    per-occurrence here so a single node's rule can be ranked internally --
    e.g. to pick the top ~4 conditions out of a depth-6 path of 6)."""
    left, right, threshold = tree.children_left, tree.children_right, tree.threshold
    features = [feature_names[i] if i >= 0 else "undefined" for i in tree.feature]
    # weighted_n_node_samples (not n_node_samples): the tree was fit with
    # class_weight, so impurity is computed from class-weighted proportions.
    # Mixing weighted impurity with raw counts breaks the non-negativity
    # guarantee of the impurity-decrease identity.
    total_samples = tree.weighted_n_node_samples[0]

    def local_importance(node):
        n_node = tree.weighted_n_node_samples[node]
        n_l, n_r = tree.weighted_n_node_samples[left[node]], tree.weighted_n_node_samples[right[node]]
        g_node, g_l, g_r = tree.impurity[node], tree.impurity[left[node]], tree.impurity[right[node]]
        weighted_child_impurity = (n_l / n_node) * g_l + (n_r / n_node) * g_r
        # Should be >= 0 since CART always picks the impurity-maximizing split;
        # clip tiny residual floating-point noise for a clean display.
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
    """Per-leaf quantitative evidence table, per the 7-point spec:
    (1) split conditions, (2) depth, (3) support (n + fraction of total pop),
    (4) node Gini impurity, (5) Gini reduction from the originating split,
    (6) target-cell fraction + lift vs. population rate,
    (7) mean IG score + enrichment vs. population mean.
    Nodes below the min support threshold are flagged, not discarded.
    """
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
        # weighted_n_node_samples (not n_node_samples) since the tree was fit
        # with class_weight: impurity is computed from class-weighted
        # proportions, so mixing it with raw counts breaks the non-negativity
        # guarantee of the impurity-decrease identity (produces spurious
        # negative "reductions").
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
        # Ratio-to-mean is numerically unstable when the population's mean IG
        # attribution sits near zero (common, since IG scores are roughly
        # symmetric). Report enrichment as a z-score in population SD units instead.
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
