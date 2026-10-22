"""Result 1 - success rate of generating a hypothesis, per model and evidence condition (CITE-seq CD45RA, CosMx Ki-67, Gouwens; no pERK).

attempt   = a completed model call: the answer was saved, or the call ran but returned no text because it hit the output cap.
            Calls that never ran (provider outage / credit errors) are not attempts. Every retry counts, so retrying until success cannot inflate the rate.
success   = the answer commits to a hypothesis: verdict "stated" in 5_evaluation_results/ablation_conditions/answer_screening.csv (my reading of each answer).
            "tentative" (hedged / 'if forced') and "declined" (says the data cannot support a hypothesis) count as not successful.
rate      = successes / attempts, with a 95% Wilson confidence interval.

Output: 1_success_rate_table/success_rate_table.{xlsx,csv,md}   (xlsx sheets: Success_rate, Attempts_n, By_dataset, All_attempts, README)
Usage : python 01_build_success_rate_table.py
"""
import csv
import json
import math

import pandas as pd

from common import ALL_CONDS, COND_LABEL, DATASET_LABEL, DATASETS, INPUTS, MODEL_LABEL, MODEL_SHORT, MODELS, ROOT, slug

OUT = ROOT / "1_success_rate_table"
OUT.mkdir(exist_ok=True)
# the model order of the table: Gemini, Qwen, GLM, DeepSeek (as in the earlier version)
ORDER = [m for m in MODELS if MODEL_SHORT[m] == "Gemini"] + [m for m in MODELS if MODEL_SHORT[m] == "Qwen"] + \
        [m for m in MODELS if MODEL_SHORT[m] == "GLM"] + [m for m in MODELS if MODEL_SHORT[m] == "DeepSeek"]


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, (c - h) / d), min(1.0, (c + h) / d)


def md_table(df):
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def main():
    verdict = {(r["model"], r["condition"], r["dataset"], int(r["run"])): r["verdict"] for r in csv.DictReader(open(INPUTS / "answer_screening.csv", encoding="utf-8"))}
    rows, unscreened = [], 0
    for m in MODELS:
        for f in sorted((INPUTS / "answers" / slug(m)).glob("*.json")):
            r = json.loads(f.read_text(encoding="utf-8"))
            if r["dataset"] not in DATASETS or r["condition"] not in ALL_CONDS:
                continue
            has_text = bool((r.get("final_answer") or "").strip())
            if not has_text:
                if r.get("finish_reason") != "length":
                    continue                      # the call never produced an attempt (no text, not a cap hit)
                v, success = "no text (hit output cap)", False
            else:
                v = verdict.get((MODEL_SHORT[m], r["condition"], r["dataset"], int(r["run"])))
                if v is None:
                    unscreened += 1
                    continue
                success = v == "stated"
            rows.append({"model": MODEL_LABEL[m], "condition": COND_LABEL[r["condition"]], "dataset": DATASET_LABEL[r["dataset"]], "run": int(r["run"]),
                         "verdict": v, "success": success})
    assert unscreened == 0, f"{unscreened} answers have no verdict yet"
    att = pd.DataFrame(rows)

    head = {MODEL_LABEL[m]: f"{MODEL_LABEL[m]} (successfully generated a hypothesis for the question)" for m in ORDER}
    conds = [COND_LABEL[c] for c in ALL_CONDS]
    rate, nn = {}, {}
    for c in conds:
        rate[c], nn[c] = {}, {}
        for m in ORDER:
            g = att[(att.condition == c) & (att.model == MODEL_LABEL[m])]
            k, n = int(g.success.sum()), len(g)
            lo, hi = wilson(k, n)
            empty = int((g.verdict == "no text (hit output cap)").sum())
            rate[c][head[MODEL_LABEL[m]]] = f"{100 * k / n:.0f}% ({100 * lo:.0f}-{100 * hi:.0f}%)  [{k}/{n}]"
            nn[c][head[MODEL_LABEL[m]]] = f"{n}" + (f" (incl. {empty} with no text at the output cap)" if empty else "")
    t_rate = pd.DataFrame(rate).T.rename_axis("Condition").reset_index()
    t_n = pd.DataFrame(nn).T.rename_axis("Condition").reset_index()

    by = (att.groupby(["condition", "model", "dataset"]).agg(attempts=("success", "size"), successes=("success", "sum")).reset_index())
    by["success_rate_%"] = (100 * by.successes / by.attempts).round(1)
    readme = pd.DataFrame({"item": ["Definition", "Attempts", "Success", "Interval", "Datasets", "Notes"], "text": [
        "Share of attempts in which the model committed to a hypothesis for the question, per evidence condition.",
        "Completed calls, including every retry (Gemini MLP + IG has many because it was retried on CITE-seq until committed answers appeared) and calls that hit the output cap without text.",
        "Verdict 'stated' = commits to a hypothesis (caveats allowed). 'tentative' and 'declined' = not successful. Verdicts are my reading of each answer (5_evaluation_results/ablation_conditions/answer_screening.csv, column note).",
        "95% Wilson score interval. Cells in square brackets are successes/attempts.",
        "CITE-seq CD45RA, CosMx Ki-67, Gouwens input resistance. pERK is excluded.",
        "Gemini borderline case: CITE-seq MLP + IG run 5 is counted as a success although it declines a regulation hypothesis and offers only a data-structure hypothesis (confidence 1)."]})
    with pd.ExcelWriter(OUT / "success_rate_table.xlsx", engine="openpyxl") as w:
        t_rate.to_excel(w, sheet_name="Success_rate", index=False)
        t_n.to_excel(w, sheet_name="Attempts_n", index=False)
        by.to_excel(w, sheet_name="By_dataset", index=False)
        att.sort_values(["model", "condition", "dataset", "run"]).to_excel(w, sheet_name="All_attempts", index=False)
        readme.to_excel(w, sheet_name="README", index=False)
        for s in w.sheets.values():
            for col in s.columns:
                s.column_dimensions[col[0].column_letter].width = 34
    t_rate.to_csv(OUT / "success_rate_table.csv", index=False)
    md = ["Share of attempts that produced a committed hypothesis, % (95% Wilson CI) [successes/attempts]\n", md_table(t_rate),
          "\nAttempts behind each cell\n", md_table(t_n)]
    (OUT / "success_rate_table.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
