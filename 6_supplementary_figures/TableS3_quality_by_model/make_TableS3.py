"""Table S3 - five-criteria hypothesis quality by LLM and input condition.

Reads ../FigS15_ablation/data/FigS15_scores_per_answer.csv (one row per answer and judge pass; no API calls) and writes
  TableS3_quality_by_model.csv   numeric values (the table as plotted data)
  TableS3_quality_by_model.md    formatted table with footnotes
  TableS3_quality_by_model.xlsx  formatted table
Computation: the score of an answer = mean of its two judge passes; a cell = mean over the datasets for which the LLM has a successfully generated (graded) answer;
n = number of such answers (3 prompts at most). Context-only: Feature Comprehensiveness (FC) and Mechanistic Synthesis (MS) are defined on supplied focus / contextual features and do not apply
when no data are given; they are shown for reference and marked with a dagger.
Usage: python make_TableS3.py
"""
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "FigS15_ablation" / "data" / "FigS15_scores_per_answer.csv"
CONDITIONS = ["PITCH", "MLP + IG", "Measurement-only", "Context-only"]
MODELS = ["Qwen3.8-2.4T-A95B", "DeepSeek V4.1 Flash", "GLM-5.3", "Gemini-3.1-pro-preview"]
SHORT_MODEL = {"Qwen3.8-2.4T-A95B": "Qwen3.8", "DeepSeek V4.1 Flash": "DeepSeek", "GLM-5.3": "GLM-5.3", "Gemini-3.1-pro-preview": "Gemini"}
COLS = {"Feature Comprehensiveness": "FC", "Biological Plausibility": "BP", "Hypothesis Specificity": "Spec.", "Mechanistic Synthesis": "MS", "Literature Grounding": "LG"}
DAGGER = {"Context-only": ["FC", "MS"]}


def main():
    sc = pd.read_csv(SRC)
    per_answer = sc.groupby(["Condition", "Model", "Prompt"], as_index=False)[list(COLS)].mean()          # mean of the two judge passes
    g = per_answer.groupby(["Condition", "Model"])
    num, shown = [], []
    for c in CONDITIONS:
        for k, m in enumerate(MODELS):
            if (c, m) in g.groups:
                d = g.get_group((c, m))
                vals = {COLS[a]: d[a].mean() for a in COLS}
                n = len(d)
            else:
                vals, n = {v: float("nan") for v in COLS.values()}, 0
            num.append({"Input condition": c, "Model": SHORT_MODEL[m], "n": n, **{k_: round(v, 2) for k_, v in vals.items()}})
            shown.append({"Input condition": c if k == 0 else "", "Model": SHORT_MODEL[m], "n": n,
                          **{k_: ("-" if pd.isna(v) else f"{v:.2f}" + ("\u2020" if k_ in DAGGER.get(c, []) else "")) for k_, v in vals.items()}})
    pd.DataFrame(num).to_csv(HERE / "TableS3_quality_by_model.csv", index=False)
    sh = pd.DataFrame(shown)
    foot = ("Scores are 0-10 (gpt-5.6-sol, blind, two passes per answer); each cell is the mean over the datasets with a successfully generated answer; n = number of such answers per LLM (3 prompts at most).\n"
            "FC, Feature Comprehensiveness; BP, Biological Plausibility; Spec., Hypothesis Specificity; MS, Mechanistic Synthesis; LG, Literature Grounding.\n"
            "\u2020 Not applicable to Context-only (no focus / contextual features are supplied); shown for reference only.")
    cols = list(sh.columns)
    md = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)] + ["| " + " | ".join(str(v) for v in r) + " |" for r in sh.values]
    (HERE / "TableS3_quality_by_model.md").write_text("\n".join(md) + "\n\n" + foot.replace("\n", "  \n") + "\n", encoding="utf-8")
    with pd.ExcelWriter(HERE / "TableS3_quality_by_model.xlsx", engine="openpyxl") as w:
        sh.to_excel(w, sheet_name="Table S3", index=False)
        pd.DataFrame({"note": foot.split("\n")}).to_excel(w, sheet_name="Notes", index=False)
        for s in w.sheets.values():
            for col in s.columns:
                s.column_dimensions[col[0].column_letter].width = 22
    print(sh.to_string(index=False))


if __name__ == "__main__":
    main()
