"""Result 2 - radar plots + datasheet of the 5-pillar hypothesis-quality scores (CITE-seq CD45RA, CosMx Ki-67, Gouwens; PITCH vs MLP + IG).

Scores: 0-10 blind-judge ratings per pillar (02b_judge_quality.py). Every (model, condition, dataset) cell is ONE randomly picked committed answer; each
answer was judged in two independent passes and the plotted score is the mean of the two passes (the passes agree on the pillar scores within ~0.4 on average).
  * one radar per MODEL      (2 conditions, mean of the 3 datasets)           -> radar_model_<model>.{png,svg,pdf}      x4
  * one radar per CONDITION  (4 models overlaid, mean of the 3 datasets)       -> radar_condition_<condition>.{png,svg,pdf} x2
  * one all-model average    (2 conditions, mean of 4 models x 3 datasets)     -> radar_all_models_average.{png,svg,pdf}
  * datasheet reproducing every plotted number                                  -> quality_radar_datasheet.xlsx
Usage: python 02c_plot_quality_radars.py
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import COND_COLOR, COND_LABEL, CONDS, DATASETS, MODEL_LABEL, MODELS, ROOT, slug

OUT = ROOT / "2_quality_radar"
SC = OUT / "scorecards"
PILLARS = ["Feature\nComprehensiveness", "Biological\nPlausibility", "Hypothesis Integration\n& Specificity", "Mechanistic\nSynthesis", "Literature\nGrounding"]
PL = [p.replace("\n", " ") for p in PILLARS]
KEYS = ["Feature Comprehensiveness_zone1", "Biological Plausibility_zone1", "Specificity_zone1", "Grounded Mechanistic Synthesis_zone1", "Literature Grounding_zone1"]
STYLE = {"cond0_full_pipeline": ("o", "-"), "cond1_ig_only": ("s", "--")}
MODEL_STYLE = {m: (c, mk, ls) for m, (c, mk, ls) in zip(MODELS, [("#9b2fae", "D", "-"), ("#2b78d4", "o", "-"), ("#ec6a36", "s", "--"), ("#1db07a", "^", "-.")])}


def radar(series, title, stem, ncol=2):
    """series: list of (label, values[5], color, marker, linestyle)."""
    ang = np.linspace(0, 2 * np.pi, len(PILLARS), endpoint=False)
    angc = np.concatenate([ang, ang[:1]])
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, polar=True)
    ax.set_theta_offset(np.pi / 2); ax.set_theta_direction(-1)
    ax.set_ylim(0, 10); ax.set_yticks([2, 4, 6, 8, 10]); ax.set_yticklabels(["2", "4", "6", "8", "10"], fontsize=8, color="#666")
    ax.set_rlabel_position(36)
    ax.set_xticks(ang); ax.set_xticklabels(PILLARS, fontsize=10.5)
    ax.tick_params(axis="x", pad=26)
    ax.grid(color="#d0d0d0", linewidth=0.7); ax.spines["polar"].set_color("#b0b0b0")
    for label, vals, col, mk, ls in series:
        v = np.concatenate([np.asarray(vals, float), [vals[0]]])
        ax.plot(angc, v, color=col, lw=2.2, ls=ls, marker=mk, ms=6, label=label, zorder=3)
        ax.fill(angc, v, color=col, alpha=0.08, zorder=2)
    ax.set_title(title, fontsize=11.5, fontweight="bold", pad=40)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=ncol, frameon=False, fontsize=9.5, handlelength=2.6)
    fig.tight_layout()
    for ext, kw in (("png", {"dpi": 600}), ("svg", {}), ("pdf", {})):
        fig.savefig(OUT / f"{stem}.{ext}", bbox_inches="tight", facecolor="white", **kw)
    plt.close(fig)


def load(name):
    rows = json.loads((SC / name / "all_scores.json").read_text(encoding="utf-8"))
    return pd.DataFrame([{"model": r["model"], "condition": r["condition"], "dataset": r["dataset"], "picked_run": r["picked_run"],
                          **{k: r["scores"][k] for k in KEYS}} for r in rows if r["dataset"] in DATASETS and r["condition"] in CONDS])


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"})
    p1, p2 = load("pass1"), load("pass2")
    key = ["model", "condition", "dataset", "picked_run"]
    m = p1.merge(p2, on=key, suffixes=("_p1", "_p2"))
    assert len(m) == len(p1) == len(p2) == len(MODELS) * len(CONDS) * len(DATASETS)
    df = m[key].copy()
    for k in KEYS:
        df[k] = (m[k + "_p1"] + m[k + "_p2"]) / 2
    mean = df.groupby(["model", "condition"])[KEYS].mean()
    allm = df.groupby("condition")[KEYS].mean()
    n_ans = len(DATASETS)

    with pd.ExcelWriter(OUT / "quality_radar_datasheet.xlsx", engine="openpyxl") as w:
        a = mean.reset_index()
        a["Model"] = a.model.map(MODEL_LABEL); a["Condition"] = a.condition.map(COND_LABEL); a["n_answers"] = n_ans
        a.rename(columns=dict(zip(KEYS, PL)))[["Model", "Condition", "n_answers"] + PL].round(3).to_excel(w, sheet_name="Per_model_x_condition", index=False)
        b = allm.reindex(CONDS); b.columns = PL; b.index = [COND_LABEL[c] for c in b.index]; b.index.name = "Condition"
        b.insert(0, "n_answers", len(MODELS) * n_ans)
        b.round(3).reset_index().to_excel(w, sheet_name="All_models_average", index=False)
        for nm, d in [("Individual_answers_mean", df), ("Pass1_scores", p1), ("Pass2_scores", p2)]:
            d = d.copy(); d["model"] = d.model.map(MODEL_LABEL); d["condition"] = d.condition.map(COND_LABEL)
            d.rename(columns=dict(zip(KEYS, PL))).round(3).to_excel(w, sheet_name=nm, index=False)
        pd.DataFrame({"note": [
            "Pillar scores are 0-10 blind-judge ratings (gpt-5.6-sol, unmodified 7-pillar rubric, five-criterion preamble; see code/02b_judge_quality.py).",
            "Each (model, condition, dataset) cell is ONE randomly picked committed answer (3_answers/ablation_conditions/selected_answers/selection_manifest.json, seed 20).",
            "Plotted value = mean of two independent judging passes over the same answers (sheets Pass1_scores / Pass2_scores hold the passes).",
            f"Datasets: {', '.join(DATASETS)} (pERK is excluded). Per-model and per-condition values are means over {n_ans} answers; the all-model average is over {len(MODELS) * n_ans} answers.",
            "Reproduce: python 02c_plot_quality_radars.py"]}).to_excel(w, sheet_name="README", index=False)

    for mdl in MODELS:
        radar([(COND_LABEL[c], mean.loc[(mdl, c)][KEYS].values, COND_COLOR[c], *STYLE[c]) for c in CONDS],
              f"Hypothesis quality by evidence condition\n{MODEL_LABEL[mdl]} -- 5-pillar blind judge, 1 random answer x {n_ans} datasets", f"radar_model_{slug(mdl)}")
    for c in CONDS:
        radar([(MODEL_LABEL[mdl], mean.loc[(mdl, c)][KEYS].values, *MODEL_STYLE[mdl]) for mdl in MODELS],
              f"Hypothesis quality: all 4 models, {COND_LABEL[c]}\n5-pillar blind judge, 1 random answer x {n_ans} datasets", f"radar_condition_{c}")
    radar([(COND_LABEL[c], allm.loc[c][KEYS].values, COND_COLOR[c], *STYLE[c]) for c in CONDS],
          f"Hypothesis quality by evidence condition\nAll 4 models averaged -- 5-pillar blind judge, {len(MODELS) * n_ans} answers per condition (4 models x {n_ans} datasets)",
          "radar_all_models_average")
    print(b.round(2).to_string())
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
