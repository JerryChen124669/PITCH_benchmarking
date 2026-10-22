"""Evaluates reliability/robustness of the LLM-generated hypotheses across
the 4 DT settings (top30/top60 IG features x depth4/depth6), per dataset.

There is no ground truth here (these are hypotheses, not verifiable facts),
so this does NOT reuse ablation_eval.py's "grade against a reference answer"
design. Instead it combines two independent checks:

  (A) Evidence Alignment (deterministic, no LLM call): for the zone/features
      each setting's response claims as primary/secondary and "top 4
      contexts", look up that exact node's own precomputed evidence
      (run_evaluation.py's *_nodes.json: support, Gini reduction, target-cell
      lift, IG enrichment, and the node's own feature-confidence ranking) and
      report whether the model picked a well-supported zone/features or a
      flagged/low-support one, and how many of its "top 4" match the node's
      own top-4-by-local-Gini-reduction ranking.

  (B) LLM-Judged Robustness (Gemini 3.1 Pro as judge, independent scoring):
      for each dataset, the 4 settings' hypotheses are graded independently
      (no reference answer) on biological plausibility, evidence grounding,
      and hypothesis specificity/testability, PLUS one shared
      "Cross-Setting Robustness" score answering the actual research
      question -- does adding more genes (top30->top60) or tree depth
      (4->6) change the core biological conclusion, or does it converge?

Usage:
    python evaluate_reliability.py
"""
import json
import os
import re
import time

import pandas as pd
from google import genai
from google.genai import types

import config

API_KEY = os.environ["GEMINI_API_KEY"]
MODEL_ID = "gemini-3.1-pro-preview"
SLEEP_BETWEEN_CALLS = 4

OUTPUT_EXCEL = config.RESULTS_DIR / "reliability_eval.xlsx"


# ==========================================================================
# PART A: EVIDENCE ALIGNMENT (deterministic)
# ==========================================================================
def extract_json_footer(text):
    match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    if not match:
        match = re.search(r"(\{[^{}]*\"primary_zone_node_id\"[\s\S]*\})", text)
    if not match:
        return None
    try:
        return json.loads(match.group(1))
    except Exception:
        return None


def node_top_features(node, k=4):
    return [step["condition"].split(" ")[0] for step in node["feature_confidence"][:k]]


def collect_hypotheses():
    """Full text of every generated hypothesis (run 1), narrative and
    structured footer split into separate columns for readability."""
    rows = []
    for ds in config.DATASETS:
        for setting in config.DT_SETTINGS:
            hyp_path = config.RESULTS_DIR / ds["key"] / f"{ds['key']}_{setting['name']}_hypothesis.jsonl"
            if not hyp_path.exists():
                continue
            with open(hyp_path, "r", encoding="utf-8") as f:
                record = json.loads(f.readline())
            full_text = record["final_answer"]
            parsed = extract_json_footer(full_text)
            rows.append({
                "dataset": ds["key"],
                "setting": setting["name"],
                "hypothesis_text": strip_json_footer(full_text),
                "primary_zone_node_id": parsed.get("primary_zone_node_id") if parsed else None,
                "secondary_zone_node_id": parsed.get("secondary_zone_node_id") if parsed else None,
                "top_4_contexts": ", ".join(parsed.get("top_4_contexts", [])) if parsed else None,
            })
    return pd.DataFrame(rows)


def evaluate_zone_pick(nodes_by_id, node_id, llm_features, zone_label):
    if node_id is None:
        return {"zone": zone_label, "node_id": None, "found": False}
    node = nodes_by_id.get(int(node_id))
    if node is None:
        return {"zone": zone_label, "node_id": node_id, "found": False}

    node_top4 = set(node_top_features(node))
    llm_set = {f.strip() for f in (llm_features or [])}
    overlap = len(node_top4 & llm_set)

    return {
        "zone": zone_label,
        "node_id": node_id,
        "found": True,
        "depth": node["depth"],
        "flagged_low_support": node["flag_low_support"],
        "pop_fraction_pct": round(node["pop_fraction"] * 100, 2),
        "n_cells": node["n_cells"],
        "target_lift": round(node["target_lift"], 2),
        "gini_reduction": round(node["gini_reduction"], 4),
        "ig_enrichment_sd": round(node["ig_enrichment_sd"], 2),
        "llm_top4_contexts": ", ".join(llm_features or []),
        "node_own_top4_contexts": ", ".join(node_top4),
        "feature_overlap_of_4": overlap,
    }


def run_evidence_alignment():
    rows = []
    for ds in config.DATASETS:
        for setting in config.DT_SETTINGS:
            out_dir = config.RESULTS_DIR / ds["key"]
            nodes_path = out_dir / f"{ds['key']}_{setting['name']}_nodes.json"
            hyp_path = out_dir / f"{ds['key']}_{setting['name']}_hypothesis.jsonl"

            if not nodes_path.exists() or not hyp_path.exists():
                print(f"  SKIP {ds['key']}/{setting['name']}: missing nodes.json or hypothesis.jsonl")
                continue

            nodes_data = json.loads(nodes_path.read_text(encoding="utf-8"))
            nodes_by_id = {n["node_id"]: n for n in nodes_data["nodes"]}

            with open(hyp_path, "r", encoding="utf-8") as f:
                first_line = f.readline()
            record = json.loads(first_line)  # run 1
            parsed = extract_json_footer(record["final_answer"])

            base_row = {"dataset": ds["key"], "setting": setting["name"]}
            if parsed is None:
                rows.append({**base_row, "zone": "PARSE_ERROR", "found": False})
                continue

            primary = evaluate_zone_pick(nodes_by_id, parsed.get("primary_zone_node_id"),
                                          parsed.get("top_4_contexts"), "primary")
            rows.append({**base_row, **primary})

            sec_id = parsed.get("secondary_zone_node_id")
            if sec_id is not None:
                secondary = evaluate_zone_pick(nodes_by_id, sec_id, parsed.get("top_4_contexts"), "secondary")
                rows.append({**base_row, **secondary})

    return pd.DataFrame(rows)


# ==========================================================================
# PART B: LLM-JUDGED ROBUSTNESS (independent scoring, no ground truth)
# ==========================================================================
JUDGE_SYSTEM_PROMPT = """You are an independent scientific reviewer evaluating computational-biology \
hypotheses generated from a decision-tree analysis of single-cell data. There is NO ground-truth answer -- \
these are hypotheses, not verifiable facts -- so do not grade against a reference. Instead:

1. Score each of the 4 hypotheses INDEPENDENTLY on 3 pillars (1-10 each):
   - Biological Plausibility: is the proposed mechanism consistent with known biology for these genes/features?
   - Evidence Grounding: does it correctly and specifically cite the quantitative node evidence (support, \
Gini reduction, target-cell lift, IG enrichment) it was given, rather than hand-waving generic claims?
   - Hypothesis Specificity & Testability: is the conclusion a concrete, falsifiable, experimentally testable \
hypothesis, not generic textbook biology?

2. Then answer the actual research question with ONE shared score: Cross-Setting Robustness (1-10) -- \
across the 4 hypotheses (which differ only in how many candidate genes fed the decision tree, and how deep \
the tree was allowed to grow), do they converge on the same core biological mechanism/pathway, or does the \
conclusion change substantially with these hyperparameters? A high score means the hypothesis is reliable \
regardless of this modeling choice; a low score means it is fragile / hyperparameter-sensitive. Give a \
rationale citing SPECIFICALLY what stayed the same and what changed across the 4 hypotheses."""


def build_judge_prompt(dataset_key, hypotheses_by_setting):
    setting_names = list(hypotheses_by_setting.keys())
    body = "\n\n".join(
        f"--- HYPOTHESIS: {name} ---\n{text}" for name, text in hypotheses_by_setting.items()
    )
    schema_settings = ",\n".join(
        f'  "{name}": {{"biological_plausibility": {{"score": <1-10>, "rationale": "<one sentence>"}}, '
        f'"evidence_grounding": {{"score": <1-10>, "rationale": "<one sentence>"}}, '
        f'"specificity_testability": {{"score": <1-10>, "rationale": "<one sentence>"}}}}'
        for name in setting_names
    )
    return (
        f"Dataset: {dataset_key}\n\n{body}\n\n"
        f"Return ONLY valid JSON, no markdown fences, no surrounding text:\n"
        f"{{\n{schema_settings},\n"
        f'  "cross_setting_robustness": {{"score": <1-10>, "rationale": "<what stayed consistent, what changed, across settings>"}}\n'
        f"}}"
    )


def strip_json_footer(text):
    return re.sub(r"```json\s*\{.*?\}\s*```", "", text, flags=re.DOTALL).strip()


def parse_judge_json(text):
    try:
        return json.loads(text.strip())
    except Exception:
        pass
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        try:
            return json.loads(match.group())
        except Exception:
            pass
    return None


def run_llm_judge(client):
    judge_rows = []
    robustness_rows = []
    judge_config = types.GenerateContentConfig(system_instruction=JUDGE_SYSTEM_PROMPT, temperature=0.1)

    for ds in config.DATASETS:
        dataset_key = ds["key"]
        hypotheses_by_setting = {}
        for setting in config.DT_SETTINGS:
            hyp_path = config.RESULTS_DIR / dataset_key / f"{dataset_key}_{setting['name']}_hypothesis.jsonl"
            if not hyp_path.exists():
                continue
            with open(hyp_path, "r", encoding="utf-8") as f:
                record = json.loads(f.readline())
            hypotheses_by_setting[setting["name"]] = strip_json_footer(record["final_answer"])

        if len(hypotheses_by_setting) < 2:
            print(f"  SKIP judging {dataset_key}: fewer than 2 hypotheses available")
            continue

        print(f"  Judging {dataset_key} ({len(hypotheses_by_setting)} settings)...", end=" ", flush=True)
        prompt = build_judge_prompt(dataset_key, hypotheses_by_setting)
        try:
            response = client.models.generate_content(model=MODEL_ID, contents=prompt, config=judge_config)
            parsed = parse_judge_json(response.text)
        except Exception as e:
            print(f"ERROR: {e}")
            parsed = None

        if parsed is None:
            print("PARSE ERROR")
            time.sleep(SLEEP_BETWEEN_CALLS)
            continue
        print("ok")

        for setting_name in hypotheses_by_setting:
            s = parsed.get(setting_name, {})
            judge_rows.append({
                "dataset": dataset_key,
                "setting": setting_name,
                "biological_plausibility": s.get("biological_plausibility", {}).get("score"),
                "biological_plausibility_rationale": s.get("biological_plausibility", {}).get("rationale"),
                "evidence_grounding": s.get("evidence_grounding", {}).get("score"),
                "evidence_grounding_rationale": s.get("evidence_grounding", {}).get("rationale"),
                "specificity_testability": s.get("specificity_testability", {}).get("score"),
                "specificity_testability_rationale": s.get("specificity_testability", {}).get("rationale"),
            })

        robust = parsed.get("cross_setting_robustness", {})
        robustness_rows.append({
            "dataset": dataset_key,
            "cross_setting_robustness_score": robust.get("score"),
            "rationale": robust.get("rationale"),
        })

        time.sleep(SLEEP_BETWEEN_CALLS)

    return pd.DataFrame(judge_rows), pd.DataFrame(robustness_rows)


# ==========================================================================
# MAIN
# ==========================================================================
def main():
    print("=== Collecting generated hypotheses (full text) ===")
    hypotheses_df = collect_hypotheses()

    print("=== Part A: Evidence Alignment (deterministic) ===")
    alignment_df = run_evidence_alignment()

    print("\n=== Part B: LLM-Judged Robustness (Gemini 3.1 Pro) ===")
    client = genai.Client(api_key=API_KEY)
    judge_df, robustness_df = run_llm_judge(client)

    with pd.ExcelWriter(OUTPUT_EXCEL, engine="openpyxl") as writer:
        hypotheses_df.to_excel(writer, sheet_name="Hypotheses", index=False)
        alignment_df.to_excel(writer, sheet_name="Evidence_Alignment", index=False)
        judge_df.to_excel(writer, sheet_name="LLM_Judge_Scores", index=False)
        robustness_df.to_excel(writer, sheet_name="Cross_Setting_Robustness", index=False)

    print(f"\nSaved -> {OUTPUT_EXCEL}")
    print(f"  Hypotheses: {len(hypotheses_df)} rows")
    print(f"  Evidence_Alignment: {len(alignment_df)} rows")
    print(f"  LLM_Judge_Scores: {len(judge_df)} rows")
    print(f"  Cross_Setting_Robustness: {len(robustness_df)} rows")


if __name__ == "__main__":
    main()
