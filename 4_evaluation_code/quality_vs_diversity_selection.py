"""
Quality-eligibility vs diversity plot: which models give diverse hypotheses that are ALSO reliably good?

x  robust hypothesis quality = the quality a model reaches in at least 5 of its 6 cases (the 2nd-lowest case),
   where a hypothesis' quality = mean(Biological Plausibility, Specificity, Mechanistic Synthesis). One bad
   case therefore cannot sink a model, but a model that is weak in two cases cannot qualify.
y  diversity = 10 - mean (P3+P4)/2 similarity (mechanistic hypothesis + functional/causal role) to the OTHER
   quality-eligible models, averaged over the cases. Eligible models are compared with the other eligible
   models; ineligible models are compared with all eligible models.

Regions (all fixed by rubric anchors, not tuned to the data)
  eligibility   robust quality >= --quality-min (default 7.0: lower bound of the rubric's 'strong or mostly
                reliable' band) -> shaded.
  redundant     diversity < --redundant-below (default 2.0 = mean P3+P4 similarity to the other eligible models of 8
                or more, the 'strongly similar' band of the judge's anchors): adds little the others do not already say.
                (Earlier versions of this script used 1.0 = mean similarity >= 9, 'nearly equivalent'; with the data in
                this repository both thresholds select the same models.)
Selected subset = eligible AND not redundant.
Low-quality but highly divergent = NOT eligible and more distinct than every eligible model: diversity that is
more likely to reflect errors than creativity (a caution, not a proof).

Pool = the models of a cluster_assignments.csv (default: the 22-model dendrogram heatmap).

Outputs (--out-dir): quality_vs_diversity_selection.{png,pdf,svg}, model_table.csv, summary.txt
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["svg.fonttype"] = "none"
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from analyze_pairwise_clustering import PAIRWISE_DIR, _place_labels, model_short_name
from qualified_diverse_subset import DEFAULT_SCORES, load_data

DEFAULT_ASSIGNMENTS = PAIRWISE_DIR / "FINAL_3categories_similarity_and_performance" / "cluster_assignments.csv"
OUT_DIR = PAIRWISE_DIR / "FINAL_all_models_no_temp0" / "quality_vs_diversity_selection"
BLUE, RED, ORANGE, GREY, INK, MUTED = "#4575B4", "#D73027", "#FC8D59", "#8c8c8c", "#222222", "#666666"
PURPLE = "#7B3294"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--assignments", type=str, default=None, help=f"Default: {DEFAULT_ASSIGNMENTS}")
    ap.add_argument("--quality-min", type=float, default=7.0)
    ap.add_argument("--redundant-below", type=float, default=2.0,
                    help="A model whose distinctiveness is below this value is dropped as redundant (default 2.0).")
    ap.add_argument("--floor-case", choices=["second", "lowest"], default="second",
                    help="Quality floor = the 2nd-lowest case quality (default; tolerates one bad case) or the lowest case (every case counts).")
    ap.add_argument("--gate", nargs=3, type=int, metavar=("BP", "SP", "MS"), default=None,
                    help="Rubric gate: eligible only if EVERY case has Biological Plausibility >= BP, Specificity >= SP and "
                         "Mechanistic Synthesis >= MS (e.g. 7 6 5 = the lower edges of the rubric anchors). The quality floor is still drawn; "
                         "--quality-min is then only the reference line (the mean of the gate minimums).")
    ap.add_argument("--strict", action="store_true", help="Eligible only if robust quality is strictly ABOVE --quality-min (a model exactly on the line is excluded).")
    ap.add_argument("--out-dir", type=str, default=None)
    ap.add_argument("--exclude-cases", nargs="+", default=[], help="Substrings of case names to drop from both axes (e.g. IL2RA pRPS6).")
    args = ap.parse_args()
    out_dir = Path(args.out_dir) if args.out_dir else OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    pool = set(pd.read_csv(Path(args.assignments) if args.assignments else DEFAULT_ASSIGNMENTS)["model"])
    temp0 = ["Qwen3.5-9B-think-temp0"]  # excluded everywhere; its pair data is incomplete
    all_models = sorted(set(load_data(temp0, DEFAULT_SCORES)[0]))
    models, cases, S3, S34, Q = load_data(temp0 + [m for m in all_models if m not in pool], DEFAULT_SCORES)
    if args.exclude_cases:  # drop whole cases from BOTH axes (quality and similarity)
        keep = [i for i, c in enumerate(cases) if not any(x.lower() in c.lower() for x in args.exclude_cases)]
        cases, S3, S34, Q = [cases[i] for i in keep], S3[keep], S34[keep], {p: v[keep] for p, v in Q.items()}
    n, k = len(models), len(cases)
    names = [model_short_name(m).replace("gemini-3.1-pro-preview", "gemini") for m in models]

    q = (Q["BP"] + Q["SP"] + Q["MS"]) / 3  # (cases x models) hypothesis quality
    robust = np.sort(q, axis=0)[0 if args.floor_case == "lowest" else 1]  # quality floor: the lowest case, or the 2nd-lowest (tolerates one bad case)
    cmp = ">" if args.strict else ">="
    eligible = robust > args.quality_min + 1e-9 if args.strict else robust >= args.quality_min - 1e-9
    failgate = np.zeros(n, dtype=bool)
    cases_pass = np.full(n, k)
    if args.gate:
        gb, gs, gm = args.gate
        per_case_ok = (Q["BP"] >= gb) & (Q["SP"] >= gs) & (Q["MS"] >= gm)
        cases_pass = per_case_ok.sum(axis=0)
        gate_ok = per_case_ok.all(axis=0)
        failgate = ~gate_ok & eligible  # clears the average-quality line but not the rubric gate
        eligible = gate_ok

    div = np.zeros(n)
    for m in range(n):
        others = [j for j in np.where(eligible)[0] if j != m]
        div[m] = np.mean([10 - np.mean([S34[c, m, j] for j in others]) for c in range(k)])
    redundant = div < args.redundant_below
    selected = eligible & ~redundant
    diverge = ~eligible & (div > div[eligible].max())
    status = np.where(selected, "selected", np.where(failgate, "above line, fails rubric gate", np.where(diverge, "below line, highly divergent", np.where(eligible, "eligible, redundant", "not eligible"))))

    table = pd.DataFrame(dict(model=names, quality_floor=robust, mean_quality=q.mean(axis=0), lowest_case_quality=q.min(axis=0),
                              min_BP=Q["BP"].min(axis=0), min_Specificity=Q["SP"].min(axis=0), min_MechSynthesis=Q["MS"].min(axis=0),
                              cases_passing_gate=cases_pass, diversity=div, status=status)).sort_values(["status", "quality_floor"], ascending=[True, False])
    table.to_csv(out_dir / "model_table.csv", index=False)
    # the selected models as an assignments file, so other scripts (diversity_vs_quality.py --assignments) can use them as a pool
    pd.DataFrame({"model": np.array(models)[selected], "cluster": 1}).to_csv(out_dir / "selected_models_assignments.csv", index=False)

    # ---- figure ---------------------------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(11.5, 8.2))
    xmax, ymax = robust.max() + 0.45, div.max() + 0.7
    xmin = robust.min() - 0.4
    ax.axvspan(args.quality_min, xmax, color="#c6dbef", alpha=0.45, lw=0, zorder=0)
    ax.axvline(args.quality_min, color=BLUE, ls="--", lw=1.2, zorder=1)
    ax.axhspan(0, args.redundant_below, color="#e6e6e6", alpha=0.8, lw=0, zorder=0)
    ax.text(xmin + 0.1, 0.12, f"redundant: distinctiveness < {args.redundant_below:g} (mean P3+P4 similarity to the other eligible models >= {10 - args.redundant_below:g})", ha="left", va="bottom", fontsize=8.5, color=MUTED)
    if args.gate:
        ax.text(args.quality_min + 0.06, ymax - 0.12,
                f"QUALITY-ELIGIBLE: in EVERY case\nBio. plausibility >= {args.gate[0]}, Specificity >= {args.gate[1]},\nMech. synthesis >= {args.gate[2]}\n"
                f"(line {args.quality_min:.1f} = mean of the 3 gate minimums;\nthe gate, not the line, decides)", ha="left", va="top", fontsize=8.8, color=BLUE, fontweight="bold")
    else:
        ax.text(args.quality_min + 0.06, ymax - 0.12, f"QUALITY-ELIGIBLE\n(quality floor {cmp} {args.quality_min:.1f})", ha="left", va="top", fontsize=9.5, color=BLUE, fontweight="bold")
    for m in range(n):
        if failgate[m]:
            ax.scatter(robust[m], div[m], s=130, facecolor="white", edgecolor=PURPLE, linewidth=2.6, zorder=4)
        elif selected[m]:
            ax.scatter(robust[m], div[m], s=190, color=BLUE, edgecolor=RED, linewidth=2.6, zorder=5)
        elif eligible[m]:
            ax.scatter(robust[m], div[m], s=110, color=BLUE, edgecolor="white", linewidth=1.0, zorder=4)
        elif diverge[m]:
            ax.scatter(robust[m], div[m], s=130, facecolor="white", edgecolor=ORANGE, linewidth=2.6, zorder=4)
        else:
            ax.scatter(robust[m], div[m], s=80, facecolor="white", edgecolor=GREY, linewidth=1.5, zorder=3)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(0, ymax)
    floor_txt = (f"Quality floor = lowest case quality  (reached in all {k} cases; " if args.floor_case == "lowest"
                 else f"Quality floor = 2nd-lowest case quality  (reached in at least {k - 1} of {k} cases; ")
    ax.set_xlabel(floor_txt + "case quality = mean of Bio. plausibility, Specificity, Mech. synthesis)", fontsize=10)
    ax.set_ylabel("Mechanistic + functional-role diversity  (10 - mean P3+P4 similarity to the other eligible models)", fontsize=10)
    ax.set_title(f"Isolating the high-quality, high-diversity subset  ({int(selected.sum())} of {n} models selected)", loc="left", fontsize=13, color=INK, pad=14)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, color="#ececec", lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    gate_handle = ([Line2D([], [], marker="o", ls="", markerfacecolor="white", markeredgecolor=PURPLE, markeredgewidth=2.4, markersize=10,
                           label="average quality above the line, but fails the rubric gate in at least one case")] if failgate.any() else [])
    ax.legend(handles=gate_handle + [
        Line2D([], [], marker="o", ls="", color=BLUE, markeredgecolor=RED, markeredgewidth=2.4, markersize=11, label="selected: eligible and not redundant"),
        Line2D([], [], marker="o", ls="", markerfacecolor="white", markeredgecolor=ORANGE, markeredgewidth=2.4, markersize=10, label="below the quality line but more divergent than any eligible model (possible error-driven)"),
        Line2D([], [], marker="o", ls="", markerfacecolor="white", markeredgecolor=GREY, markeredgewidth=1.6, markersize=9, label="not eligible, not unusually divergent"),
    ], loc="lower left", bbox_to_anchor=(0.0, 0.17), fontsize=8.8, frameon=False)
    fig.canvas.draw()
    _place_labels(fig, ax, np.column_stack([robust, div]), names)
    fig.text(0.07, 0.005, f"Cases used ({k}): {', '.join(c.split('_')[0] for c in cases)}. Eligibility line comes from the rubric/judge anchors, not tuned to the data; redundancy band: distinctiveness < {args.redundant_below:g}. "
             "Diversity of low-quality models is measured against the eligible pool. Distinct is not creative: no check yet that a distinct hypothesis is new to the literature or true.",
             fontsize=7.8, color=MUTED, wrap=True)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    stem = out_dir / "quality_vs_diversity_selection"
    for ext in (".png", ".pdf", ".svg"):
        fig.savefig(stem.with_suffix(ext), dpi=250 if ext == ".png" else None, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {stem.with_suffix('.png')} / .pdf / .svg")

    lines = ["QUALITY vs DIVERSITY SELECTION", "=" * 70,
             (f"{n} models; eligible = in every one of the {k} cases BP >= {args.gate[0]}, Specificity >= {args.gate[1]}, MechSynthesis >= {args.gate[2]}; "
              f"redundant = diversity < {args.redundant_below}. Above the {args.quality_min} floor line but failing the gate: " + (", ".join(np.array(names)[failgate]) or "none"))
             if args.gate else
             f"{n} models; eligible = quality floor ({'lowest' if args.floor_case == 'lowest' else '2nd-lowest'} of {k} case qualities) {cmp} {args.quality_min}; redundant = diversity < {args.redundant_below}.",
             f"Selected ({int(selected.sum())}): " + ", ".join(np.array(names)[selected]),
             "Eligible but redundant: " + (", ".join(np.array(names)[eligible & redundant]) or "none"),
             "Below the quality line but highly divergent: " + (", ".join(np.array(names)[diverge]) or "none"), "",
             table.round(2).to_string(index=False)]
    (out_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
