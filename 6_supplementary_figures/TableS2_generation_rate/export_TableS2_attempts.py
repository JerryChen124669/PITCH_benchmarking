"""Table S2 - step 1: export one row per completed attempt (condition, LLM, prompt, run, verdict) to data/TableS2_attempts.csv.

Reads every saved attempt of the ablation (../../3_answers/ablation_conditions/all_attempts/<LLM>/<dataset>_<condition>_run<k>.json) and the verdict of each answer
(../../5_evaluation_results/ablation_conditions/answer_screening.csv: stated / tentative / declined, from reading the answers).
An attempt is a completed call: it either returned an answer or returned no text because it reached the output limit ("no text"). Calls that never ran are not attempts.
pERK/EGF attempts are not part of this analysis. Usage: python export_TableS2_attempts.py
"""
import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
ATT = REPO / "3_answers" / "ablation_conditions" / "all_attempts"
SCREEN = REPO / "5_evaluation_results" / "ablation_conditions" / "answer_screening.csv"
COND = {"cond0_full_pipeline": "PITCH", "cond1_ig_only": "MLP + IG", "cond2_raw_data": "Measurement-only", "cond4_context_only_softened": "Context-only"}
MODEL = {"gemini-3.1-pro-preview": ("Gemini", "Gemini 3.1 Pro Preview"), "zai-org/GLM-5.3": ("GLM", "GLM-5.3"), "deepseek-ai/DeepSeek-V4.1-Flash": ("DeepSeek", "DeepSeek V4.1 Flash"),
         "Qwen/Qwen3.8-2.4T-A95B": ("Qwen", "Qwen3.8 2.4T-A95B")}
PROMPT = {"CITEseq_CD45RA_naive": "CITE-seq CD45RA", "CosMx_CancerKi67": "CosMx Ki-67", "Gouwens_GABAergic": "Patch-seq input resistance"}


def main():
    verdict = {(r["model"], r["condition"], r["dataset"], int(r["run"])): r["verdict"] for r in csv.DictReader(open(SCREEN, encoding="utf-8"))}
    rows = []
    for f in sorted(ATT.glob("*/*_run*.json")):
        r = json.loads(f.read_text(encoding="utf-8"))
        if r["condition"] not in COND or r["dataset"] not in PROMPT or r["model"] not in MODEL:
            continue
        short, label = MODEL[r["model"]]
        if (r.get("final_answer") or "").strip():
            v = {"stated": "committed"}.get(verdict[(short, r["condition"], r["dataset"], int(r["run"]))], verdict[(short, r["condition"], r["dataset"], int(r["run"]))])
        else:
            assert r.get("finish_reason") == "length", f.name
            v = "no text"
        rows.append([COND[r["condition"]], label, PROMPT[r["dataset"]], int(r["run"]), v])
    order = list(COND.values())
    rows.sort(key=lambda x: (order.index(x[0]), x[1], x[2], x[3]))
    with open(HERE / "data" / "TableS2_attempts.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["Input condition", "LLM", "Prompt", "Run", "Outcome"])
        w.writerows(rows)
    print(len(rows), "attempts")


if __name__ == "__main__":
    main()
