"""Figure S15 - ablation of the PITCH intermediate representations (PITCH, MLP + IG, Measurement-only, Context-only).

Reads data/FigS15_scores_per_answer.csv and data/FigS15_composition_agreement_per_prompt.csv (no API calls) and writes
  FigS15_ablation_radar.{svg,pdf,png}                  MAIN figure: all six axes for all four conditions (Context-only is drawn on every axis)
  FigS15_ablation_radar_applicable_axes.{svg,pdf,png}  the same, but Context-only only on the axes that apply to it (see below)
  data/FigS15_plot_values.csv                           the numbers on the radar

Averaging, in this order:
  1. every answer was judged in two passes (gpt-5.6-sol, blind, five criteria); the answer's score = mean of its two passes;
  2. condition mean = mean of the answers' scores (PITCH n = 12, MLP + IG n = 12, Measurement-only n = 7, Context-only n = 12; n = successfully generated answers;
     12 = 4 LLMs x 3 prompts);
  3. hypothesis composition agreement: judged among the LLMs within each prompt (3 passes, mean), then averaged over the three prompts.
     PITCH, MLP + IG and Context-only: across all four LLMs. Measurement-only: across the LLMs that successfully generated an answer for that prompt
     (CITE-seq GLM-5.3 + Qwen3.8; CosMx GLM-5.3 + DeepSeek V4.1 Flash + Qwen3.8; Gouwens Gemini-3.1-pro-preview + GLM-5.3), because no prompt has
     a Measurement-only answer from all four LLMs. Groups of two or three answers agree more easily than groups of four, so this point is not strictly comparable.
Context-only gives the LLM only the study system, the experimental design and the target response (no data). Feature Comprehensiveness and Mechanistic Synthesis are defined on supplied
focus / contextual features and do not apply to it; the main figure still draws them (for reference), the "applicable axes" figure leaves them out.
Axis order: Literature Grounding, Biological Plausibility, Hypothesis Specificity, Mechanistic Synthesis, Feature Comprehensiveness, Hypothesis composition agreement.
Usage: python plot_FigS15.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"

CONDITIONS = ["PITCH", "MLP + IG", "Measurement-only", "Context-only"]
STYLE = {"PITCH": ("#2b78d4", "o", "-"), "MLP + IG": ("#ec6a36", "s", "--"), "Measurement-only": ("#1db07a", "^", "-."), "Context-only": ("#8b5fbf", "D", ":")}
CRITERIA = ["Literature Grounding", "Biological Plausibility", "Hypothesis Specificity", "Mechanistic Synthesis", "Feature Comprehensiveness"]
AGREEMENT = "Hypothesis composition agreement"
AXES = CRITERIA + [AGREEMENT]
NOT_APPLICABLE = {"Context-only": ["Feature Comprehensiveness", "Mechanistic Synthesis"]}
AXIS_LABEL = {"Feature Comprehensiveness": "Feature\nComprehensiveness", "Biological Plausibility": "Biological\nPlausibility",
              "Hypothesis Specificity": "Hypothesis\nSpecificity", "Mechanistic Synthesis": "Mechanistic\nSynthesis",
              "Literature Grounding": "Literature\nGrounding", AGREEMENT: "Hypothesis composition\nagreement"}

plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"], "svg.fonttype": "none", "pdf.fonttype": 42})


def values():
    sc = pd.read_csv(DATA / "FigS15_scores_per_answer.csv")
    per_answer = sc.groupby(["Condition", "Model", "Prompt"], as_index=False)[CRITERIA].mean()            # 1. mean of the two judge passes
    n = per_answer.groupby("Condition").size()
    mean = per_answer.groupby("Condition")[CRITERIA].mean()                                                 # 2. mean over answers
    ag = pd.read_csv(DATA / "FigS15_composition_agreement_per_prompt.csv")
    ag = ag.groupby("Condition")["Hypothesis composition agreement (mean of passes)"].mean()                # 3. mean over the three prompts
    v = pd.DataFrame(index=CONDITIONS, columns=["n answers"] + AXES, dtype=float)
    for c in CONDITIONS:
        v.loc[c, "n answers"] = n[c]
        v.loc[c, CRITERIA] = mean.loc[c, CRITERIA].values
        v.loc[c, AGREEMENT] = ag[c]
    v.index.name = "Condition"
    return v


def radar(v, stem, applicable_only):
    ang = np.linspace(0, 2 * np.pi, len(AXES), endpoint=False)
    fig = plt.figure(figsize=(7.4, 7.6))
    ax = fig.add_subplot(111, polar=True)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_ylim(0, 10)
    ax.set_yticks([2, 4, 6, 8, 10])
    ax.set_yticklabels(["2", "4", "6", "8", "10"], fontsize=9, color="#666666")
    ax.set_rlabel_position(30)
    ax.set_xticks(ang)
    ax.set_xticklabels([AXIS_LABEL[a] for a in AXES], fontsize=11)
    ax.tick_params(axis="x", pad=18)
    ax.grid(color="#d0d0d0", linewidth=0.7)
    ax.spines["polar"].set_color("#b0b0b0")
    th = np.linspace(ang[-1] - np.pi / 6, ang[-1] + np.pi / 6, 40)          # shaded sector: axis defined across models only
    ax.fill_between(th, 0, 10, color="#8a2b2b", alpha=0.045, zorder=0)
    m = len(AXES)
    for c in CONDITIONS:
        col, mk, ls = STYLE[c]
        vals = v.loc[c, AXES].astype(float).values.copy()
        if applicable_only and c in NOT_APPLICABLE:
            vals[[AXES.index(a) for a in NOT_APPLICABLE[c]]] = np.nan
        have = ~np.isnan(vals)
        for i in range(m):                                                   # line only between neighbouring axes that both have a value (gap stays visible)
            j = (i + 1) % m
            if have[i] and have[j]:
                ax.plot([ang[i], ang[i] + (ang[j] - ang[i]) % (2 * np.pi)], [vals[i], vals[j]], color=col, lw=2.2, ls=ls, zorder=3)
        ax.plot(ang[have], vals[have], color=col, lw=0, marker=mk, ms=6, zorder=4)
        if have.all():
            ax.fill(np.append(ang, ang[0]), np.append(vals, vals[0]), color=col, alpha=0.08, zorder=2)
    handles = [Line2D([0], [0], color=STYLE[c][0], lw=2.2, ls=STYLE[c][2], marker=STYLE[c][1], ms=6, label=c) for c in CONDITIONS]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False, fontsize=11, handlelength=2.6, columnspacing=1.8)
    fig.tight_layout()
    for ext, kw in (("svg", {}), ("pdf", {}), ("png", {"dpi": 300})):
        fig.savefig(HERE / f"{stem}.{ext}", bbox_inches="tight", facecolor="white", **kw)
    plt.close(fig)


def main():
    v = values()
    v.round(3).to_csv(DATA / "FigS15_plot_values.csv")
    radar(v, "FigS15_ablation_radar", applicable_only=False)                          # main figure: all axes, all conditions
    radar(v, "FigS15_ablation_radar_applicable_axes", applicable_only=True)           # Context-only only where its criteria apply
    print(v.round(2).to_string())


if __name__ == "__main__":
    main()
