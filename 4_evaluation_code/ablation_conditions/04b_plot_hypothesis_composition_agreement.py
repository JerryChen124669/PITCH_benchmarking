"""Result 4 - bar plot + datasheet of the Hypothesis Composition Agreement score (4_hypothesis_composition_agreement/data/all_agreement.json).

Hypothesis Composition Agreement (0-10) = how much of each model's marker-level explanation (which specific molecules the hypothesis is built from, and how they are used) is shared by the
other models, judged for one prompt across the 4 models. Bar = mean over the 3 prompts (CITE-seq CD45RA, CosMx Ki-67, Gouwens input resistance) of the prompt score (mean of 3
random-order judging passes); error bar = SD across the prompts; one marker per prompt.
Output: 4_hypothesis_composition_agreement/hypothesis_composition_agreement_bars.{png,svg,pdf}, hypothesis_composition_agreement_datasheet.xlsx (Bars, Per_prompt, Every_pass_with_rationale, README)
Usage:  python 04b_plot_hypothesis_composition_agreement.py
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import COND_COLOR, COND_LABEL, CONDS, DATASET_LABEL, DATASET_MARKER, DATASETS, ROOT

OUT = ROOT / "4_hypothesis_composition_agreement"
NAME = "Hypothesis Composition Agreement"


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "svg.fonttype": "none"})
    recs = json.loads((OUT / "data" / "all_agreement.json").read_text(encoding="utf-8"))
    meta = json.loads((OUT / "data" / "agreement_meta.json").read_text(encoding="utf-8"))
    df = pd.DataFrame([{"condition": r["condition"], "dataset": r["dataset"], "pass": r["pass"], "score": r["score"],
                        "rationale": r["judge"].get("rationale", r["judge"].get("null_reason", ""))} for r in recs])
    assert len(df) == len(CONDS) * len(DATASETS) * meta["passes"] and df.score.notna().all()
    per_prompt = df.groupby(["condition", "dataset"]).score.agg(["mean", "std", "count"]).reset_index()

    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    bars = []
    for i, c in enumerate(CONDS):
        vals = np.array([per_prompt[(per_prompt.condition == c) & (per_prompt.dataset == d)]["mean"].iloc[0] for d in DATASETS])
        mean, sd = vals.mean(), vals.std(ddof=1)
        ax.bar(i, mean, width=0.55, color=COND_COLOR[c], alpha=0.85, zorder=2)
        ax.errorbar(i, mean, yerr=sd, color="black", capsize=7, lw=1.8, zorder=4)
        for j, d in enumerate(DATASETS):
            ax.scatter(i + (j - 1) * 0.1, vals[j], marker=DATASET_MARKER[d], s=48, facecolor="white", edgecolor="black", lw=1.2, zorder=5, label=DATASET_LABEL[d] if i == 0 else None)
        ax.text(i, max(mean + sd, vals.max()) + 0.3, f"{mean:.2f}", ha="center", fontsize=11, fontweight="bold")
        bars.append({"condition": COND_LABEL[c], "mean_of_3_prompts": mean, "sd_across_prompts": sd, "n_prompts": len(vals), **{DATASET_LABEL[d]: v for d, v in zip(DATASETS, vals)}})
    ax.set_xticks(range(len(CONDS))); ax.set_xticklabels([COND_LABEL[c].replace(" (", "\n(") for c in CONDS], fontsize=11)
    ax.set_xlim(-0.6, 1.6); ax.set_ylim(0, 10); ax.set_yticks(range(0, 11, 2))
    ax.set_ylabel(f"{NAME} score\n(0-10, LLM-judged; 10 = identical marker use)", fontsize=10.5)
    ax.set_title(f"{NAME} across the 4 models\nmean over 3 prompts, error bars = SD; judge {meta['judge_model']}, {meta['passes']} random-order passes", fontsize=10, fontweight="bold")
    ax.grid(axis="y", color="#e0e0e0", lw=0.7, zorder=0); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(title="Prompt (marker = score of that prompt)", loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, frameon=False, fontsize=9, title_fontsize=9)
    fig.tight_layout()
    for ext, kw in (("png", {"dpi": 600}), ("svg", {}), ("pdf", {})):
        fig.savefig(OUT / f"hypothesis_composition_agreement_bars.{ext}", bbox_inches="tight", facecolor="white", **kw)
    plt.close(fig)

    names = {"mean": "prompt_score_mean_of_passes", "std": "sd_across_passes", "count": "n_passes"}
    with pd.ExcelWriter(OUT / "hypothesis_composition_agreement_datasheet.xlsx", engine="openpyxl") as w:
        pd.DataFrame(bars).round(3).to_excel(w, sheet_name="Bars", index=False)
        pp = per_prompt.rename(columns=names)
        pp["condition"] = pp.condition.map(COND_LABEL); pp["dataset"] = pp.dataset.map(DATASET_LABEL)
        pp.round(3).to_excel(w, sheet_name="Per_prompt", index=False)
        ev = df.copy(); ev["condition"] = ev.condition.map(COND_LABEL); ev["dataset"] = ev.dataset.map(DATASET_LABEL)
        ev.sort_values(["condition", "dataset", "pass"]).to_excel(w, sheet_name="Every_pass_with_rationale", index=False)
        pd.DataFrame({"item": ["Score", "Judge input", "Aggregation", "Rubric", "Datasets"], "text": [
            f"{NAME} (0-10): agreement across the 4 models in the specific molecular markers used to construct the primary hypotheses (which exact markers, in comparable roles). Shared target / "
            "program / pathway earns no credit; different genes with similar functions are different contributors.",
            "Per prompt, the 4 picked answers (Gemini, GLM-5.3, DeepSeek V4.1 Flash, Qwen3.8), anonymized and shown together in a random order that changes every pass; no generation prompt, no model names.",
            "Prompt score = mean of 3 passes. Bar = mean of the 3 prompt scores; error bar = SD across prompts.",
            f"code/evaluation/hypothesis_composition_agreement_rubric.txt (sha256 {meta['rubric_sha256'][:12]}...): inventory of all qualifying molecular contributors, shared-coverage bands in quarters, five mandatory ceilings; "
            "each judgment comes with a one-sentence rationale (sheet Every_pass_with_rationale).",
            "CITE-seq CD45RA, CosMx Ki-67, Gouwens input resistance (pERK excluded)."]}).to_excel(w, sheet_name="README", index=False)
    print(pd.DataFrame(bars).round(2).to_string(index=False))
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
