"""Result 2d - radar plots with all THREE conditions (PITCH, MLP + IG, measurement-only).

Measurement-only scores: only 7 of the 12 (model, prompt) cells have a committed measurement-only answer (GLM 3, Qwen 2, DeepSeek 1, Gemini 1), so for measurement-only ONE randomly picked committed answer per
available cell was judged (same 5-pillar judge, two passes; 3_answers/ablation_conditions/selected_answers/cond2_raw_data, 02a + 02b). Plotted value = mean of the two passes; the measurement-only average is over the
7 answers (not balanced across models). PITCH and MLP + IG are the 12 answers per condition of result 2.
  * radar_all_models_average_5axes_3conditions  5 pillars, 3 conditions (all-model average)
  * radar_all_models_average_6axes_3conditions  the same + hypothesis composition agreement. PITCH and MLP + IG: agreement across all 4 models (result 4, 04a). MEASUREMENT-ONLY: agreement among the
                                                  models that have a successfully generated answer for the prompt (3, 2 and 2 answers per prompt; 04c), because measurement-only answers do
                                                  not exist for all four models on any prompt. Groups of 2-3 answers agree more easily than groups of 4, so this point is not strictly comparable.
  * radar_model_<model>_5axes_3conditions       5 pillars, 3 conditions per model (raw n: Gemini 1, GLM 3, DeepSeek 1, Qwen 2)
  * radar_3conditions_datasheet.xlsx            every plotted number, the n behind it and the individual measurement-only scores
Usage: python 02f_plot_radars_3conditions.py
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import ALL_CONDS, COND_COLOR, COND_LABEL, DATASET_LABEL, DATASETS, MODEL_LABEL, MODELS, ROOT, slug
CONDS3 = ALL_CONDS[:3]                                           # PITCH, MLP + IG, measurement-only (the context-only condition is added by 02h)

OUT = ROOT / "2_quality_radar"
SC = OUT / "scorecards"
PILLARS = ["Feature Comprehensiveness", "Biological Plausibility", "Hypothesis Specificity", "Mechanistic Synthesis", "Literature Grounding"]
KEYS = ["Feature Comprehensiveness_zone1", "Biological Plausibility_zone1", "Specificity_zone1", "Grounded Mechanistic Synthesis_zone1", "Literature Grounding_zone1"]
AX5 = ["Feature\nComprehensiveness", "Biological\nPlausibility", "Hypothesis\nSpecificity", "Mechanistic\nSynthesis", "Literature\nGrounding"]
AX6 = AX5 + ["Hypothesis composition agreement\nacross models (0-10)"]
STYLE = {"cond0_full_pipeline": ("o", "-"), "cond1_ig_only": ("s", "--"), "cond2_raw_data": ("^", "-."), "cond3_context_only": ("v", (0, (5, 2))), "cond4_context_only_softened": ("D", ":")}


def load(dirname, conds):
    rows = json.loads((SC / dirname / "all_scores.json").read_text(encoding="utf-8"))
    return pd.DataFrame([{"model": r["model"], "condition": r["condition"], "dataset": r["dataset"], "picked_run": r["picked_run"], **{k: r["scores"][k] for k in KEYS}}
                         for r in rows if r["dataset"] in DATASETS and r["condition"] in conds])


def mean_of_passes(a, b):
    key = ["model", "condition", "dataset", "picked_run"]
    m = a.merge(b, on=key, suffixes=("_p1", "_p2"))
    assert len(m) == len(a) == len(b)
    out = m[key].copy()
    for k in KEYS:
        out[k] = (m[k + "_p1"] + m[k + "_p2"]) / 2
    return out


def radar(ax_labels, series, title, stem, footnote, highlight_last=False, ncol=3, size=(9, 8)):
    ang = np.linspace(0, 2 * np.pi, len(ax_labels), endpoint=False)
    angc = np.concatenate([ang, ang[:1]])
    fig = plt.figure(figsize=size)
    ax = fig.add_subplot(111, polar=True)
    ax.set_theta_offset(np.pi / 2); ax.set_theta_direction(-1)
    ax.set_ylim(0, 10); ax.set_yticks([2, 4, 6, 8, 10]); ax.set_yticklabels(["2", "4", "6", "8", "10"], fontsize=8, color="#666")
    ax.set_rlabel_position(30)
    ax.set_xticks(ang); ax.set_xticklabels(ax_labels, fontsize=10)
    ax.tick_params(axis="x", pad=32)
    if highlight_last:
        ax.get_xticklabels()[-1].set_color("#8a2b2b"); ax.get_xticklabels()[-1].set_fontweight("bold")
        th = np.linspace(ang[-1] - np.pi / 6, ang[-1] + np.pi / 6, 40)
        ax.fill_between(th, 0, 10, color="#8a2b2b", alpha=0.045, zorder=0)
    ax.grid(color="#d0d0d0", linewidth=0.7); ax.spines["polar"].set_color("#b0b0b0")
    for label, vals, c in series:
        v = np.concatenate([vals, [vals[0]]])
        mk, ls = STYLE[c]
        ax.plot(angc, v, color=COND_COLOR[c], lw=2.2, ls=ls, marker=mk, ms=6, label=label, zorder=3)
        ax.fill(angc, v, color=COND_COLOR[c], alpha=0.08, zorder=2)
    ax.set_title(title, fontsize=11.5, fontweight="bold", pad=42)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=ncol, frameon=False, fontsize=9.5, handlelength=2.6)
    fig.text(0.5, 0.012, footnote, ha="center", fontsize=8, color="#444", wrap=True)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    for ext, kw in (("png", {"dpi": 600}), ("svg", {}), ("pdf", {})):
        fig.savefig(OUT / f"{stem}.{ext}", bbox_inches="tight", facecolor="white", **kw)
    plt.close(fig)


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"})
    main2 = mean_of_passes(load("pass1", CONDS3), load("pass2", CONDS3))
    raw = mean_of_passes(load("raw_pass1", ["cond2_raw_data"]), load("raw_pass2", ["cond2_raw_data"]))
    assert len(raw) == 7 and len(main2) == 24
    df = pd.concat([main2, raw], ignore_index=True)
    n_all = df.groupby("condition").size()
    allm = df.groupby("condition")[KEYS].mean()
    mdl = df.groupby(["model", "condition"])[KEYS].mean()
    n_mdl = df.groupby(["model", "condition"]).size()
    conc = pd.read_excel(ROOT / "4_hypothesis_composition_agreement" / "hypothesis_composition_agreement_datasheet.xlsx", sheet_name="Bars").set_index("condition")["mean_of_3_prompts"]
    recs = json.loads((ROOT / "4_hypothesis_composition_agreement" / "data_available_models" / "all_agreement_available_models.json").read_text(encoding="utf-8"))
    avail = pd.DataFrame([{"dataset": r["dataset"], "pass": r["pass"], "n_models": len(r["models"]), "score": r["score"]} for r in recs if r["condition"] == "cond2_raw_data"])
    assert len(avail) == 9 and avail.score.notna().all()
    avail_prompt = avail.groupby("dataset").score.mean()                 # mean of the 3 passes per prompt
    conc_val = {c: float(conc[COND_LABEL[c]]) for c in CONDS3 if c != "cond2_raw_data"}
    conc_val["cond2_raw_data"] = float(avail_prompt.mean())              # mean of the 3 prompts

    label = lambda c: f"{COND_LABEL[c].replace('Measurement-only', 'Measurement-only')} (n = {n_all[c]} answers)"
    foot5 = "Measurement-only: one committed answer per available (model, prompt) cell, 7 of 12 cells (GLM 3, Qwen 2, DeepSeek 1, Gemini 1); PITCH and MLP + IG: 12 answers each."
    radar(AX5, [(label(c), allm.loc[c][KEYS].values, c) for c in CONDS3],
          "Hypothesis quality by evidence condition\nAll models averaged (3 datasets)", "radar_all_models_average_5axes_3conditions", foot5, ncol=1)
    foot6 = (foot5 + "\nAxis 6: agreement across the 4 models for PITCH and MLP + IG; for measurement-only, among the models with a successfully generated answer (3, 2 and 2 answers per prompt).")
    radar(AX6, [(label(c), np.append(allm.loc[c][KEYS].values, conc_val[c]), c) for c in CONDS3],
          "Hypothesis quality and cross-model hypothesis composition agreement\nAll models averaged (3 datasets)", "radar_all_models_average_6axes_3conditions", foot6,
          highlight_last=True, ncol=1)
    for m in MODELS:
        ser = [(f"{COND_LABEL[c]} (n = {int(n_mdl.get((m, c), 0))})", mdl.loc[(m, c)][KEYS].values, c) for c in CONDS3 if (m, c) in mdl.index]
        radar(AX5, ser, f"Hypothesis quality by evidence condition\n{MODEL_LABEL[m]} (3 datasets)", f"radar_model_{slug(m)}_5axes_3conditions",
              "n = number of answers (one per prompt); measurement-only only where the model produced a committed hypothesis.", ncol=1)

    with pd.ExcelWriter(OUT / "radar_3conditions_datasheet.xlsx", engine="openpyxl") as w:
        a = allm.reindex(CONDS3); a.columns = PILLARS
        a.insert(0, "n_answers", [int(n_all[c]) for c in CONDS3]); a["Hypothesis composition agreement (measurement-only: available models only)"] = [conc_val[c] for c in CONDS3]
        a.index = [COND_LABEL[c] for c in a.index]; a.index.name = "Condition"
        a.round(3).reset_index().to_excel(w, sheet_name="All_models_average", index=False)
        mm = mdl.reset_index(); mm["n_answers"] = [int(n_mdl[(r.model, r.condition)]) for r in mm.itertuples()]
        mm["model"] = mm.model.map(MODEL_LABEL); mm["condition"] = mm.condition.map(COND_LABEL)
        mm.rename(columns=dict(zip(KEYS, PILLARS))).round(3).to_excel(w, sheet_name="Per_model_x_condition", index=False)
        bal = raw.groupby("model")[KEYS].mean().mean()
        pd.DataFrame({"Measurement-only, model-balanced mean (mean of the 4 model means)": bal.values}, index=PILLARS).round(3).to_excel(w, sheet_name="Raw_model_balanced")
        r = raw.copy(); r["model"] = r.model.map(MODEL_LABEL); r["dataset"] = r.dataset.map(DATASET_LABEL)
        r.rename(columns=dict(zip(KEYS, PILLARS))).round(3).to_excel(w, sheet_name="Raw_individual_answers", index=False)
        ap = avail_prompt.rename(index=DATASET_LABEL).reset_index(); ap.columns = ["Prompt", "Measurement-only agreement (mean of 3 passes)"]
        ap["Models in the group"] = [int(avail[avail.dataset == d].n_models.iloc[0]) for d in avail_prompt.index]
        ap.round(3).to_excel(w, sheet_name="Raw_agreement_per_prompt", index=False)
        pd.DataFrame({"note": [
            "Same 5-pillar blind judge as result 2 (gpt-5.6-sol, unmodified rubric, five-criterion preamble); plotted = mean of two passes.",
            "Measurement-only: 7 answers (one committed answer per available model x prompt cell). The measurement-only averages are over these 7 answers: GLM 3, Qwen 2, DeepSeek 1, Gemini 1 (sheet Raw_model_balanced shows the model-balanced mean).",
            "Most measurement-only answers were generated under earlier wordings of the measurement-only prompt (the judge was shown the wording each answer actually saw). Gemini's only raw answer (Gouwens) rests on a false claim about the data.",
            "6-axis radar, axis 6: PITCH and MLP + IG = hypothesis composition agreement across the 4 models (04a, mean of 3 prompts); measurement-only = agreement among the models with a successfully generated answer (CITE-seq GLM + Qwen, CosMx GLM + DeepSeek + Qwen, Gouwens Gemini + GLM; 04c; 3 passes per prompt, mean of 3 prompts). Smaller groups agree more easily, so this point is not strictly comparable.",
            "Reproduce: 02a_select_answers.py, 02b_judge_quality.py --conds cond2_raw_data --out-dir ../../5_evaluation_results/ablation_conditions/2_quality_radar/scorecards/raw_pass1 (and raw_pass2), 04c_judge_hypothesis_composition_agreement_available_models.py, 02f_plot_radars_3conditions.py"]}).to_excel(w, sheet_name="README", index=False)
    print(a.round(2).to_string())
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
