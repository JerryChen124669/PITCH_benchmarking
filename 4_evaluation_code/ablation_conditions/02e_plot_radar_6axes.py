"""Result 2c - all-model-average hexagonal radar: the 5 hypothesis-quality pillars (result 2) + hypothesis composition agreement (result 4).

Axes 1-5  quality pillars (0-10 blind-judge scores; mean over 4 models x 3 datasets = 12 answers per condition)       <- 2_quality_radar/quality_radar_datasheet.xlsx
Axis 6    hypothesis composition agreement across the 4 models (0-10, LLM-judged; mean over 3 prompts)                       <- 4_hypothesis_composition_agreement/hypothesis_composition_agreement_datasheet.xlsx
Axis 6 describes agreement ACROSS models for a prompt (it does not exist per model), so only the all-model average can show it. Radars show means only; the bar plot of result 4
carries the SD across prompts.
Output: 2_quality_radar/radar_all_models_average_6axes.{png,svg,pdf}, radar_6axes_datasheet.xlsx
Usage:  python 02e_plot_radar_6axes.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import COND_COLOR, COND_LABEL, CONDS, ROOT

OUT = ROOT / "2_quality_radar"
PILLARS = ["Feature Comprehensiveness", "Biological Plausibility", "Hypothesis Integration & Specificity", "Mechanistic Synthesis", "Literature Grounding"]
AXES = ["Feature\nComprehensiveness", "Biological\nPlausibility", "Hypothesis Integration\n& Specificity", "Mechanistic\nSynthesis", "Literature\nGrounding",
        "Hypothesis composition agreement\nacross models (0-10)"]
STYLE = {"cond0_full_pipeline": ("o", "-"), "cond1_ig_only": ("s", "--")}


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"})
    q = pd.read_excel(OUT / "quality_radar_datasheet.xlsx", sheet_name="All_models_average").set_index("Condition")
    m = pd.read_excel(ROOT / "4_hypothesis_composition_agreement" / "hypothesis_composition_agreement_datasheet.xlsx", sheet_name="Bars").set_index("condition")
    vals = {c: [float(q.loc[COND_LABEL[c], p]) for p in PILLARS] + [float(m.loc[COND_LABEL[c], "mean_of_3_prompts"])] for c in CONDS}

    ang = np.linspace(0, 2 * np.pi, len(AXES), endpoint=False)
    angc = np.concatenate([ang, ang[:1]])
    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, polar=True)
    ax.set_theta_offset(np.pi / 2); ax.set_theta_direction(-1)
    ax.set_ylim(0, 10); ax.set_yticks([2, 4, 6, 8, 10]); ax.set_yticklabels(["2", "4", "6", "8", "10"], fontsize=8, color="#666")
    ax.set_rlabel_position(30)
    ax.set_xticks(ang); ax.set_xticklabels(AXES, fontsize=10)
    ax.get_xticklabels()[5].set_color("#8a2b2b"); ax.get_xticklabels()[5].set_fontweight("bold")      # the across-model axis
    ax.tick_params(axis="x", pad=32)
    ax.grid(color="#d0d0d0", linewidth=0.7); ax.spines["polar"].set_color("#b0b0b0")
    th = np.linspace(ang[5] - np.pi / 6, ang[5] + np.pi / 6, 40)                                       # light shading of the across-model sector
    ax.fill_between(th, 0, 10, color="#8a2b2b", alpha=0.045, zorder=0)
    for c in CONDS:
        v = np.concatenate([vals[c], [vals[c][0]]])
        mk, ls = STYLE[c]
        ax.plot(angc, v, color=COND_COLOR[c], lw=2.2, ls=ls, marker=mk, ms=6, label=COND_LABEL[c], zorder=3)
        ax.fill(angc, v, color=COND_COLOR[c], alpha=0.08, zorder=2)
    ax.set_title("Hypothesis quality and cross-model hypothesis composition agreement\nAll 4 models averaged (3 datasets)", fontsize=11.5, fontweight="bold", pad=42)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False, fontsize=10, handlelength=2.6)
    fig.text(0.5, 0.012, "Axes 1-5: per-answer quality, 12 answers per condition. Axis 6 (shaded): agreement across the 4 models, 3 prompts per condition. Radars show means only.",
             ha="center", fontsize=8, color="#444")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    for ext, kw in (("png", {"dpi": 600}), ("svg", {}), ("pdf", {})):
        fig.savefig(OUT / f"radar_all_models_average_6axes.{ext}", bbox_inches="tight", facecolor="white", **kw)
    plt.close(fig)

    sheet = pd.DataFrame({COND_LABEL[c]: vals[c] for c in CONDS}, index=[a.replace("\n", " ") for a in AXES]).rename_axis("Axis").reset_index()
    sheet.insert(1, "Group", ["per-answer quality"] * 5 + ["across-model consistency"])
    sheet.insert(2, "Source", ["quality_radar_datasheet.xlsx (All_models_average)"] * 5 + ["hypothesis_composition_agreement_datasheet.xlsx (Bars, mean_of_3_prompts)"])
    with pd.ExcelWriter(OUT / "radar_6axes_datasheet.xlsx", engine="openpyxl") as w:
        sheet.round(3).to_excel(w, sheet_name="All_models_6axes", index=False)
    print(sheet.round(2).to_string(index=False))
    print(f"-> {OUT / 'radar_all_models_average_6axes.png'}")


if __name__ == "__main__":
    main()
