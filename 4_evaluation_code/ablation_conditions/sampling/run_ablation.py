"""Builds both ablation-condition prompts (ig_only, raw_data) for all 4
datasets = 8 prompt files. For condition 2, the raw-data row count is
determined empirically per dataset via Gemini's real tokenizer (see
raw_data_sampler.py) -- the final cell counts actually used are printed and
saved to *_sample_meta.json.

Usage:
    python run_ablation.py
"""
import json
import os
import time

import numpy as np
from google import genai

import config
import pipeline
import raw_data_sampler
from prompt_builder import build_ig_only_prompt, build_raw_data_prompt

API_KEY = os.environ["GEMINI_API_KEY"]
_client = genai.Client(api_key=API_KEY)


def _count_tokens(text):
    return _client.models.count_tokens(model=config.MODEL_ID, contents=text).total_tokens


def run_dataset(ds_meta):
    key = ds_meta["key"]
    print(f"\n=== Dataset: {key} (mode={ds_meta['mode']}) ===")
    t0 = time.time()

    if ds_meta["mode"] == "ig_auto":
        X_raw, y_raw, headers, target_name = pipeline.load_data(ds_meta["path"])
        print(f"  Loaded {X_raw.shape[0]} cells x {X_raw.shape[1]} features. Target: {target_name}")

        model, corr = pipeline.train_mlp(X_raw, y_raw)
        print(f"  MLP trained. Test Pearson r = {corr:.4f} ({time.time() - t0:.1f}s elapsed)")

        attr = pipeline.compute_ig(model, X_raw)
        ranking, avg_abs, target_prot = pipeline.pick_focused_feature(attr, headers)
        n_features_requested = ds_meta["n_features"]
    else:  # manual (pERK) -- precomputed attr, no MLP retrain needed
        X_raw, y_raw, headers, target_name, attr = pipeline.load_pERK_raw_table(ds_meta)
        print(f"  Loaded {X_raw.shape[0]} cells x {X_raw.shape[1]} raw features (pre-MLP). Target: {target_name}")
        ranking, avg_abs, target_prot = pipeline.pick_focused_feature(attr, headers, manual_idx=ds_meta["feature_idx"])
        n_features_requested = 30  # rank the full 510-dim IG output same as the other datasets

    print(f"  Focused feature: {target_prot}")
    corr_label, r_val = pipeline.correlation_direction(X_raw, y_raw, headers, target_prot)

    out_dir = config.RESULTS_DIR / key
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------------- Condition 1: IG-only, no classifier -----------------
    ig_ranking = pipeline.select_attributing_features(ranking, avg_abs, headers, target_prot, n_features_requested)
    prompt1 = build_ig_only_prompt(
        dataset_meta=ds_meta, target_name=target_name, target_prot=target_prot,
        corr_label=corr_label, n_features_requested=n_features_requested, ig_ranking=ig_ranking,
    )
    (out_dir / f"{key}_cond1_ig_only_prompt.txt").write_text(prompt1, encoding="utf-8")
    print(f"  [cond1_ig_only] {len(ig_ranking)} features -> {key}_cond1_ig_only_prompt.txt "
          f"({_count_tokens(prompt1):,} tokens)")

    # ---------------- Condition 2: raw data only, no IG/classifier --------
    idx, csv_text, actual_tokens, n_rows = raw_data_sampler.fit_to_token_budget(
        _count_tokens, headers, target_name, X_raw, y_raw,
    )
    prompt2 = build_raw_data_prompt(
        dataset_meta=ds_meta, target_name=target_name, target_prot=target_prot,
        n_total_cells=X_raw.shape[0], n_sample_cells=n_rows, sample_tokens=actual_tokens,
        csv_table_text=csv_text,
    )
    (out_dir / f"{key}_cond2_raw_data_prompt.txt").write_text(prompt2, encoding="utf-8")
    total_prompt2_tokens = _count_tokens(prompt2)
    print(f"  [cond2_raw_data] {n_rows} / {X_raw.shape[0]} cells ({n_rows / X_raw.shape[0] * 100:.2f}%), "
          f"data block = {actual_tokens:,} tokens, full prompt = {total_prompt2_tokens:,} tokens "
          f"-> {key}_cond2_raw_data_prompt.txt")

    meta = {
        "dataset_key": key, "target_prot": target_prot, "target_name": target_name,
        "n_total_cells": int(X_raw.shape[0]), "n_features_total": int(X_raw.shape[1]),
        "cond1_n_features": len(ig_ranking),
        "cond2_n_sample_cells": n_rows, "cond2_sample_fraction_pct": round(n_rows / X_raw.shape[0] * 100, 3),
        "cond2_data_block_tokens": actual_tokens, "cond2_full_prompt_tokens": total_prompt2_tokens,
        "cond2_sample_row_indices": [int(i) for i in idx],
        "token_budget_used": config.RAW_DATA_TOKEN_BUDGET,
        "gemini_input_token_limit": config.GEMINI_INPUT_TOKEN_LIMIT,
    }
    (out_dir / f"{key}_sample_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"  Done with {key} in {time.time() - t0:.1f}s")
    return meta


def main():
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_meta = []
    for ds_meta in config.DATASETS:
        all_meta.append(run_dataset(ds_meta))

    print("\n=== SUMMARY: condition-2 raw-data cell counts ===")
    for m in all_meta:
        print(f"  {m['dataset_key']}: {m['cond2_n_sample_cells']} / {m['n_total_cells']} cells "
              f"({m['cond2_sample_fraction_pct']}%), {m['cond2_data_block_tokens']:,} tokens")

    print("\nAll datasets complete. 8 prompts (4 datasets x 2 conditions) written under:", config.RESULTS_DIR)


if __name__ == "__main__":
    main()
