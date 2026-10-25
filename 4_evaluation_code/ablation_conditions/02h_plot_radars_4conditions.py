"""EXPLORATORY CONTROL, NOT USED IN THE MANUSCRIPT (Figure S15 and Table S3 show PITCH, MLP + IG and measurement-only only).
The outputs of this script (four-condition radars, datasheet, per-model table with the Context-only control) are not kept in the repository; the records it reads
(Context-only answers, scorecards `ctx2_pass1/2`, agreement records) are, so that the control can be inspected and re-run.
Result 2f - 6-axis radar with the context-only condition added, and the per-model score table.

Conditions: PITCH, MLP + IG, Measurement-only, Context-only (the LLM receives only the study system, the design and the target response; softened prompt, condition cond4:
no honesty instruction, a specific mechanism requested). The first context-only prompt (cond3) is archived and not analysed here.
Quality axes 1-5: mean over all successfully generated (committed) answers of the condition (one random committed answer per model x prompt cell, judged in two passes, mean of the passes).
Axis 6, hypothesis composition agreement (mean of the 3 prompts, each the mean of 3 judge passes):
  PITCH, MLP + IG                   judged across all four LLMs (04a)
  Measurement-only, Context-only    judged among the LLMs with a committed answer for the prompt (04c)
Groups of 2-3 answers agree more easily than groups of four, so the measurement-only point (3, 2, 2 answers) is not strictly comparable with the others; Context-only uses 4, 4, 4 answers.
Context-only is drawn without Feature Comprehensiveness and Mechanistic Synthesis (not applicable: defined on supplied focus / contextual features).
Axis order: Literature Grounding, Biological Plausibility, Hypothesis Specificity, Mechanistic Synthesis, Feature Comprehensiveness, agreement (the two criteria not applicable to Context-only are neighbours).
Output (2_quality_radar/): radar_all_models_average_{5,6}axes_4conditions[_all_axes].{png,svg,pdf}, radar_4conditions_datasheet.xlsx, per_model_scores_table.{csv,md}
Usage: python 02h_plot_radars_4conditions.py
"""
import importlib.util
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import ALL_CONDS, COND_COLOR, COND_LABEL, DATASET_LABEL, HERE, MODEL_LABEL, ROOT

spec = importlib.util.spec_from_file_location("radars02f", HERE / "02f_plot_radars_3conditions.py")
r02 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r02)          # reuse the loaders and the radar drawing of 02f

OUT = ROOT / "2_quality_radar"
KEYS, PILLARS, AX5, AX6 = r02.KEYS, r02.PILLARS, r02.AX5, r02.AX6
MAIN = ["cond0_full_pipeline", "cond1_ig_only", "cond2_raw_data", "cond4_context_only_softened"]
AVAIL = ("cond2_raw_data", "cond4_context_only_softened")
CTX = "cond4_context_only_softened"
NA_KEYS = ["Feature Comprehensiveness_zone1", "Grounded Mechanistic Synthesis_zone1"]   # not applicable to Context-only (no supplied focus / contextual features)
ORDER = [4, 1, 2, 3, 0]   # axis order of the radars: Literature Grounding, Biological Plausibility, Specificity, Mechanistic Synthesis, Feature Comprehensiveness
SHORT = {"Feature Comprehensiveness_zone1": "FC", "Biological Plausibility_zone1": "BP", "Specificity_zone1": "Spec.", "Grounded Mechanistic Synthesis_zone1": "MS", "Literature Grounding_zone1": "LG"}
TABLE_MODEL = {"Qwen/Qwen3.8-2.4T-A95B": "Qwen3.8", "deepseek-ai/DeepSeek-V4.1-Flash": "DeepSeek", "zai-org/GLM-5.3": "GLM-5.3", "gemini-3.1-pro-preview": "Gemini"}
TABLE_ORDER = ["Qwen/Qwen3.8-2.4T-A95B", "deepseek-ai/DeepSeek-V4.1-Flash", "zai-org/GLM-5.3", "gemini-3.1-pro-preview"]
PROMPTS = ["CITEseq_CD45RA_naive", "CosMx_CancerKi67", "Gouwens_GABAergic"]


def md_table(df):
    cols = list(df.columns)
    return "\n".join(["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)] + ["| " + " | ".join(str(v) for v in row) + " |" for row in df.values])


def radar_partial(ax_labels, series, title, stem, footnote, highlight_last=False, ncol=1, size=(9, 8)):
    """Like 02f's radar(), but a series may have NaN on some axes (criteria that do not apply to it): its markers are drawn only where it has a value,
    its line only between neighbouring axes that both have a value (so the gap over the missing axes stays visible), and it gets no shaded fill."""
    from matplotlib.lines import Line2D
    ang = np.linspace(0, 2 * np.pi, len(ax_labels), endpoint=False)
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
    n = len(ang)
    for label, vals, c in series:
        vals = np.asarray(vals, dtype=float)
        mk, ls = r02.STYLE[c]
        col = COND_COLOR[c]
        have = ~np.isnan(vals)
        for i in range(n):                                     # line segments between neighbouring axes (including the wrap-around) that both have a value
            j = (i + 1) % n
            if have[i] and have[j]:
                ax.plot([ang[i], ang[i] + (ang[j] - ang[i]) % (2 * np.pi)], [vals[i], vals[j]], color=col, lw=2.2, ls=ls, zorder=3)
        ax.plot(ang[have], vals[have], color=col, lw=0, marker=mk, ms=6, zorder=4)
        if have.all():
            v = np.concatenate([vals, vals[:1]]); a = np.concatenate([ang, ang[:1]])
            ax.fill(a, v, color=col, alpha=0.08, zorder=2)
    handles = [Line2D([0], [0], color=COND_COLOR[c], lw=2.2, ls=r02.STYLE[c][1], marker=r02.STYLE[c][0], ms=6, label=label) for label, _v, c in series]
    ax.set_title(title, fontsize=11.5, fontweight="bold", pad=42)
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=ncol, frameon=False, fontsize=9.5, handlelength=2.6)
    fig.text(0.5, 0.012, footnote, ha="center", fontsize=8, color="#444", wrap=True)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    for ext, kw in (("png", {"dpi": 600}), ("svg", {}), ("pdf", {})):
        fig.savefig(OUT / f"{stem}.{ext}", bbox_inches="tight", facecolor="white", **kw)
    plt.close(fig)


def per_model_table(conds, mdl, n_mdl):
    rows = []
    for c in conds:
        for m in TABLE_ORDER:
            ok = (m, c) in mdl.index
            rows.append({"Input condition": COND_LABEL[c] if m == TABLE_ORDER[0] else "", "Model": TABLE_MODEL[m], "n": int(n_mdl[(m, c)]) if ok else 0,
                         **{SHORT[k]: (f"{mdl.loc[(m, c)][k]:.2f}" if ok else "-") for k in KEYS}})
    return pd.DataFrame(rows)


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"})
    main2 = r02.mean_of_passes(r02.load("pass1", ALL_CONDS), r02.load("pass2", ALL_CONDS))
    raw = r02.mean_of_passes(r02.load("raw_pass1", ["cond2_raw_data"]), r02.load("raw_pass2", ["cond2_raw_data"]))
    ctx = r02.mean_of_passes(r02.load("ctx2_pass1", ["cond4_context_only_softened"]), r02.load("ctx2_pass2", ["cond4_context_only_softened"]))
    assert len(main2) == 24 and len(raw) == 7 and len(ctx) == 12
    df = pd.concat([main2, raw, ctx], ignore_index=True)
    n_all = df.groupby("condition").size()
    allm = df.groupby("condition")[KEYS].mean()
    mdl = df.groupby(["model", "condition"])[KEYS].mean()
    n_mdl = df.groupby(["model", "condition"]).size()

    # axis 6
    agr = ROOT / "4_hypothesis_composition_agreement"
    four = pd.read_excel(agr / "hypothesis_composition_agreement_datasheet.xlsx", sheet_name="Bars").set_index("condition")["mean_of_3_prompts"]
    recs = json.loads((agr / "data_available_models" / "all_agreement_available_models.json").read_text(encoding="utf-8"))
    conc_val, per_prompt = {}, {}
    for c in AVAIL:
        a = pd.DataFrame([{"dataset": r["dataset"], "pass": r["pass"], "n_models": len(r["models"]), "score": r["score"]} for r in recs if r["condition"] == c])
        assert a.score.notna().all() and a.dataset.nunique() == 3 and (a.groupby("dataset").size() == 3).all(), c
        per_prompt[c] = a.groupby("dataset").agg(score=("score", "mean"), n_models=("n_models", "first"))
        conc_val[c] = float(per_prompt[c].score.mean())
    for c in ("cond0_full_pipeline", "cond1_ig_only"):
        conc_val[c] = float(four[COND_LABEL[c]])

    sizes = lambda c: ", ".join(str(int(x)) for x in per_prompt[c].n_models.reindex(PROMPTS))
    label = lambda c: f"{COND_LABEL[c]} (n = {n_all[c]} answers)"
    foot5 = "n = successfully generated answers (one random committed answer per model and prompt):\n" + ", ".join(f"{COND_LABEL[c]} {n_all[c]}" for c in MAIN) + "."
    foot6 = (foot5 + "\nAxis 6: PITCH and MLP + IG across all 4 models; Measurement-only and Context-only among the models with a\n"
             f"successfully generated answer for the prompt (answers per prompt, CITE-seq / CosMx / Gouwens: Measurement-only {sizes('cond2_raw_data')}; Context-only {sizes('cond4_context_only_softened')}).")
    na_note = ("\nContext-only has no value on Feature Comprehensiveness and Mechanistic Synthesis (these criteria are defined on\n"
               "the supplied focus / contextual features and do not apply when no data are given).")
    # axis order: Literature Grounding, Biological Plausibility, Hypothesis Specificity, Mechanistic Synthesis, Feature Comprehensiveness (+ agreement), so that the two criteria that do not
    # apply to Context-only (Mechanistic Synthesis, Feature Comprehensiveness) are neighbours and the gap in its line is one clear break
    KO = [KEYS[i] for i in ORDER]
    ax5 = [AX5[i] for i in ORDER]
    ax6 = ax5 + [AX6[-1]]

    def vals(c, with_agreement, drop_na=True):
        v = allm.loc[c][KO].values.astype(float).copy()
        if c == CTX and drop_na:
            v[[KO.index(k) for k in NA_KEYS]] = np.nan
        return np.append(v, conc_val[c]) if with_agreement else v
    radar_partial(ax5, [(label(c), vals(c, False), c) for c in MAIN], "Hypothesis quality by input condition\nAll models averaged (3 datasets)",
                  "radar_all_models_average_5axes_4conditions", foot5 + na_note)
    radar_partial(ax6, [(label(c), vals(c, True), c) for c in MAIN],
                  "Hypothesis quality and cross-model hypothesis composition agreement\nAll models averaged (3 datasets)", "radar_all_models_average_6axes_4conditions", foot6 + na_note, highlight_last=True)
    # all-axes versions (for reference / internal discussion): Context-only drawn on all five criteria
    all_note = ("\nContext-only is drawn on all five criteria for reference; Feature Comprehensiveness and Mechanistic Synthesis are defined on supplied\n"
                "focus / contextual features and do not apply when no data are given, so they should not be read as comparable to the other conditions.")
    radar_partial(ax5, [(label(c), vals(c, False, drop_na=False), c) for c in MAIN], "Hypothesis quality by input condition\nAll models averaged (3 datasets); all axes",
                  "radar_all_models_average_5axes_4conditions_all_axes", foot5 + all_note)
    radar_partial(ax6, [(label(c), vals(c, True, drop_na=False), c) for c in MAIN],
                  "Hypothesis quality and cross-model hypothesis composition agreement\nAll models averaged (3 datasets); all axes", "radar_all_models_average_6axes_4conditions_all_axes", foot6 + all_note, highlight_last=True)

    tab = per_model_table(MAIN, mdl, n_mdl)
    tab.to_csv(OUT / "per_model_scores_table.csv", index=False)
    (OUT / "per_model_scores_table.md").write_text("Mean score (0-10) of the successfully generated answers per input condition and model; n = number of such answers (3 prompts per model).\n"
                                                   "FC and MS of the Context-only rows are shown for reference only: these criteria are defined on supplied focus / contextual features and are not applicable to Context-only (no value in the radar).\n\n" + md_table(tab) + "\n", encoding="utf-8")

    with pd.ExcelWriter(OUT / "radar_4conditions_datasheet.xlsx", engine="openpyxl") as w:
        a = allm.reindex(MAIN); a.columns = PILLARS
        a.insert(0, "n_answers", [int(n_all[c]) for c in MAIN]); a["Hypothesis composition agreement"] = [conc_val[c] for c in MAIN]
        a.index = [COND_LABEL[c] for c in a.index]; a.index.name = "Condition"
        shown = a.round(3).reset_index().astype({PILLARS[0]: object, PILLARS[3]: object})
        shown.loc[shown.Condition == COND_LABEL[CTX], [PILLARS[0], PILLARS[3]]] = "n/a"     # not applicable to Context-only (figure shows no value); numbers kept in sheet Individual_answers
        shown.to_excel(w, sheet_name="All_models_average", index=False)
        tab.to_excel(w, sheet_name="Per_model_table", index=False)
        ap = pd.concat([per_prompt[c].rename(index=DATASET_LABEL).assign(condition=COND_LABEL[c]) for c in per_prompt]).reset_index()
        ap.columns = ["Prompt", "Agreement (mean of 3 passes)", "Models in the group", "Condition"]
        ap[["Condition", "Prompt", "Models in the group", "Agreement (mean of 3 passes)"]].round(3).to_excel(w, sheet_name="Agreement_available_models", index=False)
        d = df.copy(); d["model"] = d.model.map(MODEL_LABEL); d["dataset"] = d.dataset.map(DATASET_LABEL); d["condition"] = d.condition.map(COND_LABEL)
        d.rename(columns=dict(zip(KEYS, PILLARS))).round(3).to_excel(w, sheet_name="Individual_answers", index=False)
        pd.DataFrame({"note": [
            "Five criteria: gpt-5.6-sol blind judge, unmodified rubric with the five-criterion preamble, two passes; plotted = mean of the passes; condition value = mean over successfully generated answers.",
            "Context-only = the LLM received only the study system, design and target (no data); softened prompt (honesty instruction removed, a specific mechanism requested); 12 answers (all 12 cells committed).",
            "Hypothesis composition agreement: PITCH and MLP + IG across the 4 LLMs (04a); Measurement-only and Context-only among the LLMs with a committed answer (04c). Groups of different size are not strictly comparable.",
            "GLM-5.3 context-only answers were generated with reasoning_effort = max (its API no longer accepts xhigh, used for the other conditions); all other call settings are identical."]}).to_excel(w, sheet_name="README", index=False)
    pd.set_option("display.width", 250)
    print(a.round(2).to_string())


if __name__ == "__main__":
    main()
