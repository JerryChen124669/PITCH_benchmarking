"""Builds the "primary-zone-only" variant of the 12 prompts: same tree, same
node evidence, same everything -- except the LLM is never asked to name a
secondary zone at all. Tests whether dropping that ask changes cross-setting
robustness (does removing a possibly-noisy secondary hypothesis make the
overall result look more consistent?).

No retraining needed: reuses the node evidence already saved in
*_nodes.json and the tree-rules text already saved in the original
*_prompt.txt (extracted, not recomputed), so this only touches text.

Does NOT overwrite the original *_prompt.txt / *_nodes.json files --
writes new *_primaryonly_prompt.txt files alongside them.

Usage:
    python regenerate_primary_only.py
"""
import json
import re

import config
from prompt_builder import build_prompt


def extract_corr_label(prompt_text):
    m = re.search(r"Correlation:\s*(\w+)\s*correlation", prompt_text)
    return m.group(1) if m else "Direct"


def extract_tree_rules(prompt_text):
    m = re.search(r"--- FULL DECISION TREE RAW RULES ---\n(.*?)\n\n--- TASK ---", prompt_text, re.DOTALL)
    if m:
        return m.group(1)
    # fallback: from the marker to the next "--- " header
    start = prompt_text.find("--- FULL DECISION TREE RAW RULES ---")
    if start == -1:
        return ""
    start += len("--- FULL DECISION TREE RAW RULES ---\n")
    end = prompt_text.find("\n--- ", start)
    return prompt_text[start:end if end != -1 else None]


def main():
    for ds in config.DATASETS:
        for setting in config.DT_SETTINGS:
            out_dir = config.RESULTS_DIR / ds["key"]
            nodes_path = out_dir / f"{ds['key']}_{setting['name']}_nodes.json"
            orig_prompt_path = out_dir / f"{ds['key']}_{setting['name']}_prompt.txt"

            if not nodes_path.exists() or not orig_prompt_path.exists():
                print(f"SKIP {ds['key']}/{setting['name']}: missing original files (run run_evaluation.py first)")
                continue

            nodes_data = json.loads(nodes_path.read_text(encoding="utf-8"))
            orig_text = orig_prompt_path.read_text(encoding="utf-8")

            corr_label = extract_corr_label(orig_text)
            tree_rules_text = extract_tree_rules(orig_text)

            prompt_text = build_prompt(
                dataset_meta=ds,
                target_name=nodes_data["target_name"],
                target_prot=nodes_data["target_prot"],
                corr_label=corr_label,
                attr_direction=config.ATTR_DIRECTION,
                ig_percentile=config.IG_PERCENTILE,
                n_features_requested=setting["n_features"],
                attr_features=nodes_data["attr_features"],
                max_depth=setting["max_depth"],
                class_1_weight=config.CLASS_1_WEIGHT,
                clf=None,
                tree_rules_override=tree_rules_text,
                cv_f1=nodes_data["cv_f1"],
                node_records=nodes_data["nodes"],
                min_support=nodes_data["min_support"],
                include_secondary=False,
            )

            out_path = out_dir / f"{ds['key']}_{setting['name']}_primaryonly_prompt.txt"
            out_path.write_text(prompt_text, encoding="utf-8")
            print(f"{ds['key']}/{setting['name']}: corr_label={corr_label} -> {out_path.name}")

    print("\nDone. Original *_prompt.txt files untouched.")


if __name__ == "__main__":
    main()
