"""Hypothesis composition agreement of the conditions in one table (PITCH, MLP + IG, Measurement-only; with --with-context-only also the Context-only control, which is not used in the manuscript): which LLMs were judged together, the three pass scores and the mean, per condition and prompt.

Reads the two record sets of 5_evaluation_results/ablation_conditions/4_hypothesis_composition_agreement/:
  data/                    PITCH and MLP + IG: the answers of all four LLMs judged together (04a)
  data_available_models/   Measurement-only and Context-only: the LLMs that have a committed answer for the prompt (04c)
and writes agreement_all_conditions.{csv,md} into that folder (no API call).
Usage: python 04d_summarize_agreement_all_conditions.py [--with-context-only]
"""
import argparse
import csv
import json
import statistics as st

from common import COND_LABEL, DATASET_LABEL, DATASETS, MODEL_LABEL, MODELS, ROOT

FOLDER = ROOT / "4_hypothesis_composition_agreement"
ORDER = ["cond0_full_pipeline", "cond1_ig_only", "cond2_raw_data"]          # conditions of the manuscript
CONTEXT_ONLY = "cond4_context_only_softened"                                # control, not used in the manuscript (records kept)
RECORDS = {"cond0_full_pipeline": "data/", "cond1_ig_only": "data/", "cond2_raw_data": "data_available_models/", "cond4_context_only_softened": "data_available_models/"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-context-only", action="store_true", help="also tabulate the Context-only control (not used in the manuscript)")
    order = ORDER + ([CONTEXT_ONLY] if ap.parse_args().with_context_only else [])
    recs = json.loads((FOLDER / "data" / "all_agreement.json").read_text(encoding="utf-8"))
    recs += json.loads((FOLDER / "data_available_models" / "all_agreement_available_models.json").read_text(encoding="utf-8"))
    rows = []
    for c in order:
        means = []
        for d in DATASETS:
            rr = sorted([r for r in recs if r["condition"] == c and r["dataset"] == d], key=lambda r: r["pass"])
            assert len(rr) == 3, (c, d, len(rr))
            models = rr[0].get("models", MODELS)
            scores = [r["score"] for r in rr]
            means.append(st.mean(scores))
            rows.append([COND_LABEL[c], DATASET_LABEL[d], len(models), "; ".join(MODEL_LABEL[m] for m in models), " / ".join(str(s) for s in scores), f"{st.mean(scores):.2f}", RECORDS[c] + c + "/" + d + "_pass1-3.json"])
        rows.append([COND_LABEL[c], "mean of the 3 prompts", "", "", "", f"{st.mean(means):.2f}", ""])
    head = ["Input condition", "Prompt", "Answers judged together", "LLMs judged together", "Score per pass (1 / 2 / 3)", "Score (mean of 3 passes)", "Records (relative to this folder)"]
    with open(FOLDER / "agreement_all_conditions.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(head)
        w.writerows(rows)
    md = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)] + ["| " + " | ".join(str(v) for v in r) + " |" for r in rows]
    (FOLDER / "agreement_all_conditions.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
