"""Table S3 - five-criteria hypothesis quality by LLM and input condition.

Reads ../FigS15_ablation/data/FigS15_scores_per_answer.csv (one row per answer and judge pass; no API calls) and writes
  TableS3_quality_by_model.csv   numeric values (the table as plotted data)
  TableS3_quality_by_model.md    formatted table with footnotes
  TableS3_quality_by_model.xlsx  formatted table
  TableS3_quality_by_model.png   the table as a picture in the layout of the SI (header rule, shaded alternate rows, rule between conditions)
Computation: the score of an answer = mean of its two judge passes; a cell = mean over the datasets for which the LLM has a successfully generated (graded) answer;
n = number of such answers (3 prompts at most).
Usage: python make_TableS3.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "FigS15_ablation" / "data" / "FigS15_scores_per_answer.csv"
CONDITIONS = ["PITCH", "MLP + IG", "Measurement-only"]
MODELS = ["Qwen3.8-2.4T-A95B", "DeepSeek V4.1 Flash", "GLM-5.3", "Gemini-3.1-pro-preview"]
SHORT_MODEL = {"Qwen3.8-2.4T-A95B": "Qwen3.8", "DeepSeek V4.1 Flash": "DeepSeek", "GLM-5.3": "GLM-5.3", "Gemini-3.1-pro-preview": "Gemini"}
COLS = {"Feature Comprehensiveness": "FC", "Biological Plausibility": "BP", "Hypothesis Specificity": "Spec.", "Mechanistic Synthesis": "MS", "Literature Grounding": "LG"}


def render_png(sh, path):
    """Table as a picture: bold header between two rules, lavender shading on alternate rows within each condition, a thin rule between conditions."""
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"]})
    cols = list(sh.columns)
    xs = [0.0, 2.35, 4.05, 4.65, 5.75, 6.85, 7.95, 9.05, 10.15]               # left edge of each column (centre for the numeric columns)
    width, row_h = 10.6, 0.34
    n = len(sh)
    fig, ax = plt.subplots(figsize=(width * 0.78, (n + 1) * row_h * 0.78 + 0.2))
    ax.set_xlim(0, width); ax.set_ylim(-(n + 1) * row_h, 0.1); ax.axis("off")
    ax.plot([0, width], [0.05, 0.05], color="black", lw=1.1)
    for j, c in enumerate(cols):
        ha = "left" if j < 2 else "center"
        x = xs[j] + (0.1 if j < 2 else 0.5)
        ax.text(x, -row_h / 2, c, ha=ha, va="center", fontsize=11, fontweight="bold", style="italic" if c == "n" else "normal")
    ax.plot([0, width], [-row_h, -row_h], color="black", lw=0.9)
    block_first = [i for i, v in enumerate(sh["Input condition"]) if v]
    for i, r in enumerate(sh.itertuples(index=False)):
        y0, y1 = -(i + 1) * row_h, -(i + 2) * row_h
        pos = i - max(b for b in block_first if b <= i)
        if pos % 2 == 1:
            ax.add_patch(plt.Rectangle((0, y1), width, row_h, color="#E8E8F8", lw=0, zorder=0))
        if i in block_first[1:]:
            ax.plot([0, width], [y0, y0], color="black", lw=0.9)
        for j, v in enumerate(r):
            ha = "left" if j < 2 else "center"
            x = xs[j] + (0.1 if j < 2 else 0.5)
            ax.text(x, (y0 + y1) / 2, str(v), ha=ha, va="center", fontsize=11)
    ax.plot([0, width], [-(n + 1) * row_h, -(n + 1) * row_h], color="black", lw=1.1)
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


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
                          **{k_: ("-" if pd.isna(v) else f"{v:.2f}") for k_, v in vals.items()}})
    pd.DataFrame(num).to_csv(HERE / "TableS3_quality_by_model.csv", index=False)
    sh = pd.DataFrame(shown)
    foot = ("Scores are 0-10 (gpt-5.6-sol, blind, two passes per answer); each cell is the mean over the datasets with a successfully generated answer; n = number of such answers per LLM (3 prompts at most).\n"
            "FC, Feature Comprehensiveness; BP, Biological Plausibility; Spec., Hypothesis Specificity; MS, Mechanistic Synthesis; LG, Literature Grounding.")
    cols = list(sh.columns)
    md = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)] + ["| " + " | ".join(str(v) for v in r) + " |" for r in sh.values]
    (HERE / "TableS3_quality_by_model.md").write_text("\n".join(md) + "\n\n" + foot.replace("\n", "  \n") + "\n", encoding="utf-8")
    with pd.ExcelWriter(HERE / "TableS3_quality_by_model.xlsx", engine="openpyxl") as w:
        sh.to_excel(w, sheet_name="Table S3", index=False)
        pd.DataFrame({"note": foot.split("\n")}).to_excel(w, sheet_name="Notes", index=False)
        for s in w.sheets.values():
            for col in s.columns:
                s.column_dimensions[col[0].column_letter].width = 22
    render_png(sh, HERE / "TableS3_quality_by_model.png")
    print(sh.to_string(index=False))


if __name__ == "__main__":
    main()
