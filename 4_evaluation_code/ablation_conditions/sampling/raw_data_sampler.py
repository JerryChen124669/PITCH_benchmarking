"""Downsamples a raw cells x features table to fit within Gemini's token
budget for condition 2 ("skip IG and DT, feed the raw pre-MLP data").

Two things this deliberately does NOT do, to keep the ablation honest:
  - No column/feature trimming. Trimming features would itself be a form of
    feature selection -- exactly what this condition is supposed to remove.
  - No token-budget guessing. The final row count is verified against
    Gemini's real tokenizer via client.models.count_tokens on the actual
    assembled table, not a linear extrapolation -- extrapolation is only
    used to pick a starting point, then corrected empirically.

Row selection is decile-stratified by target value (not random), because
the achievable sample is a small fraction of the full dataset (as low as
~0.5% for the 1500-feature CITEseq table) -- pure random sampling at that
rate risks missing the tails of the target distribution entirely, which
would cripple the LLM's ability to see any real relationship.
"""
import io
import csv
import time

import numpy as np

import config


def stratified_indices(y, n_target, n_bins=10, seed=42):
    """Picks n_target row indices, allocated as evenly as possible across
    n_bins quantile bins of y, so the sample spans the full target range."""
    rng = np.random.default_rng(seed)
    n_total = len(y)
    n_target = min(n_target, n_total)

    quantile_edges = np.quantile(y, np.linspace(0, 1, n_bins + 1))
    quantile_edges[-1] += 1e-9  # include the max value in the last bin
    bin_ids = np.digitize(y, quantile_edges[1:-1], right=False)

    base_per_bin = n_target // n_bins
    remainder = n_target - base_per_bin * n_bins

    chosen = []
    for b in range(n_bins):
        bin_members = np.where(bin_ids == b)[0]
        take = base_per_bin + (1 if b < remainder else 0)
        take = min(take, len(bin_members))
        if take > 0:
            chosen.extend(rng.choice(bin_members, size=take, replace=False).tolist())

    # If some bins were too small to fill their quota, top up from the
    # largest remaining pool so we still hit n_target where possible.
    chosen = set(chosen)
    if len(chosen) < n_target:
        remaining_pool = np.array([i for i in range(n_total) if i not in chosen])
        rng.shuffle(remaining_pool)
        for i in remaining_pool:
            if len(chosen) >= n_target:
                break
            chosen.add(int(i))

    chosen = sorted(chosen)
    rng.shuffle(chosen)  # don't hand the LLM a suspiciously sorted-by-bin table
    return chosen


def format_csv_table(headers, target_name, X, y, indices):
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(list(headers) + [target_name])
    for i in indices:
        row = [f"{v:.4g}" for v in X[i]] + [f"{y[i]:.4g}"]
        w.writerow(row)
    return buf.getvalue()


def fit_to_token_budget(count_tokens_fn, headers, target_name, X, y,
                         budget_tokens=config.RAW_DATA_TOKEN_BUDGET,
                         n_bins=config.RAW_DATA_STRATIFY_BINS, seed=42,
                         calibration_rows=20, max_iters=6):
    """Returns (indices, csv_text, actual_tokens, n_rows_used)."""
    n_total = len(y)

    # 1. Calibrate: measure real per-row token cost from a small sample.
    calib_idx = stratified_indices(y, min(calibration_rows, n_total), n_bins=n_bins, seed=seed)
    header_text = format_csv_table(headers, target_name, X, y, [])
    calib_text = format_csv_table(headers, target_name, X, y, calib_idx)
    tok_header = count_tokens_fn(header_text)
    tok_calib = count_tokens_fn(calib_text)
    tok_per_row = max(1.0, (tok_calib - tok_header) / len(calib_idx))

    # 2. Starting estimate.
    n_rows = int(max(1, (budget_tokens - tok_header) // tok_per_row))
    n_rows = min(n_rows, n_total)

    # 3. Verify empirically, shrink if it overshoots (usually converges in 1-2 rounds).
    for _ in range(max_iters):
        idx = stratified_indices(y, n_rows, n_bins=n_bins, seed=seed)
        text = format_csv_table(headers, target_name, X, y, idx)
        actual_tokens = count_tokens_fn(text)
        if actual_tokens <= budget_tokens or n_rows <= 1:
            return idx, text, actual_tokens, len(idx)
        overshoot_ratio = budget_tokens / actual_tokens
        n_rows = max(1, int(n_rows * overshoot_ratio * 0.97))  # small extra safety margin

    return idx, text, actual_tokens, len(idx)
