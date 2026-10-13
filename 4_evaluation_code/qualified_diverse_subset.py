"""
Which LLMs give DIVERSE yet HIGH-QUALITY hypotheses?  A two-stage rule on the existing data.

Stage 1  QUALIFY (quality gate, set from the judge rubric's own anchors, applied before looking at diversity)
    A hypothesis (one model on one case) is VALID when
        Biological Plausibility >= --bp-min       (default 7: above the 'hallucination ceiling' band, <= 6)
        Specificity            >= --sp-min        (default 6: 'a meaningful central mechanism', not feature-by-feature)
        Mechanistic Synthesis  >= --ms-min        (default 5: at least 'partial' synthesis, not 'weak')
    A MODEL is QUALIFIED when at least --reliability of its hypotheses (of the cases) are valid (default 5 of 6).

Stage 2  DIVERSIFY (only among VALID hypotheses of QUALIFIED models, so differences cannot be errors)
    novelty of a hypothesis     = 10 - mean (P3+P4)/2 similarity to the other valid hypotheses on the same case
    mechanistic BACKBONES       = per case, average-linkage clusters of the valid hypotheses on P3 (mechanistic
                                  hypothesis similarity), cut at P3 >= --backbone-sim (default 7 = the judge's
                                  'same mechanistic backbone, one meaningful difference' anchor). Hypotheses in
                                  different clusters propose different mechanisms.
    unique backbone             = a valid hypothesis that no other valid hypothesis shares a backbone with.
    coverage of a model subset  = mean over cases of (distinct valid backbones the subset reaches / all backbones).
    SUBSET = the smallest greedy set of qualified models reaching --target-coverage (default 0.9).

Robustness: the whole procedure is repeated over a grid of gate strictness (BP >= 7/8), reliability (4/5/6 of 6)
and backbone threshold (P3 >= 6/7/8); the output reports how often each model is qualified and selected.

This is a creativity PROXY (novelty x value), not a measurement of creativity: nothing here checks that a
distinct hypothesis is new to the literature or experimentally true.

Outputs (under --out-dir): qualified_diverse_subset.{png,pdf,svg}, model_table.csv, coverage_curve.csv,
sensitivity.csv, summary.txt

Usage:
    python qualified_diverse_subset.py
"""

from __future__ import annotations

import argparse
import itertools
import warnings
from pathlib import Path

import matplotlib

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["svg.fonttype"] = "none"
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from analyze_pairwise_clustering import PAIRWISE_DIR, _place_labels, load_long_table, model_short_name

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SCORES = SCRIPT_DIR / "evaluation_results_rubric_v12" / "all_scores.csv"
OUT_DIR = PAIRWISE_DIR / "FINAL_all_models_no_temp0" / "qualified_diverse_subset"
BLUE, RED, ORANGE, GREY, INK, MUTED = "#4575B4", "#D73027", "#FC8D59", "#9a9a9a", "#222222", "#666666"


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_data(exclude: list[str], scores_csv: Path):
    df = load_long_table(PAIRWISE_DIR)
    df = df[~df["model_a"].isin(exclude) & ~df["model_b"].isin(exclude)]
    models = sorted(set(df["model_a"]) | set(df["model_b"]))
    cases = sorted(df["dataset"].unique())
    n, k = len(models), len(cases)
    if len(df) != n * (n - 1) // 2 * k:
        raise SystemExit(f"ERROR: incomplete pair data ({len(df)} records for {n} models x {k} cases).")
    idx = {m: i for i, m in enumerate(models)}
    S3, S34 = np.full((k, n, n), np.nan), np.full((k, n, n), np.nan)
    for r in df.itertuples():
        c, i, j = cases.index(r.dataset), idx[r.model_a], idx[r.model_b]
        S3[c, i, j] = S3[c, j, i] = r.P3
        S34[c, i, j] = S34[c, j, i] = (r.P3 + r.P4) / 2

    sc = pd.read_csv(scores_csv)
    sc["model"] = sc["model"].astype(str).str.replace("/", "__", regex=False)
    sc["case"] = sc["prompt_file"].astype(str).str.replace(r"\.txt$", "", regex=True)
    sc = sc.rename(columns={"Biological Plausibility_zone1": "BP", "Specificity_zone1": "SP", "Grounded Mechanistic Synthesis_zone1": "MS"})
    g = sc.set_index(["case", "model"])
    missing = [(c, m) for c in cases for m in models if (c, m) not in g.index]
    if missing:
        raise SystemExit(f"ERROR: missing rubric scores {missing[:4]} ...")
    Q = {p: np.array([[g.loc[(c, m), p] for m in models] for c in cases], dtype=float) for p in ("BP", "SP", "MS")}
    return models, cases, S3, S34, Q


# ---------------------------------------------------------------------------
# Procedure
# ---------------------------------------------------------------------------

def qualify(Q, bp_min, sp_min, ms_min, reliability):
    valid = (Q["BP"] >= bp_min) & (Q["SP"] >= sp_min) & (Q["MS"] >= ms_min)  # (cases x models)
    return valid, valid.sum(axis=0) >= reliability


def novelty_and_backbones(S3, S34, valid, qualified, backbone_sim):
    """novelty[c, m] (NaN unless valid & qualified), label[c, m] backbone cluster id (-1 if not in the valid pool),
    n_backbones[c], unique[c, m] (sole member of its backbone)."""
    k, n = valid.shape
    use = valid & qualified[None, :]
    nov = np.full((k, n), np.nan)
    label = np.full((k, n), -1)
    unique = np.zeros((k, n), dtype=bool)
    nb = np.zeros(k, dtype=int)
    for c in range(k):
        V = np.where(use[c])[0]
        if len(V) == 0:
            continue
        if len(V) == 1:
            label[c, V[0]], nb[c], unique[c, V[0]] = 0, 1, True
            continue
        for i in V:
            nov[c, i] = 10 - np.mean([S34[c, i, j] for j in V if j != i])
        D = 10 - S3[c][np.ix_(V, V)]
        np.fill_diagonal(D, 0.0)
        lab = fcluster(linkage(squareform(D, checks=False), "average"), t=10 - backbone_sim, criterion="distance")
        for v, l in zip(V, lab):
            label[c, v] = l
        nb[c] = len(set(lab))
        sizes = {l: int((lab == l).sum()) for l in set(lab)}
        for v, l in zip(V, lab):
            unique[c, v] = sizes[l] == 1
    return nov, label, nb, unique


def coverage(subset, label, nb) -> float:
    cs = []
    for c in range(label.shape[0]):
        if nb[c] == 0:
            continue
        got = {label[c, m] for m in subset if label[c, m] >= 0}
        cs.append(len(got) / nb[c])
    return float(np.mean(cs))


def greedy_subset(candidates, label, nb, quality):
    chosen, curve = [], []
    left = list(candidates)
    while left:
        best = max(left, key=lambda m: (coverage(chosen + [m], label, nb), quality[m]))
        chosen.append(best)
        left.remove(best)
        curve.append(coverage(chosen, label, nb))
    return chosen, curve


def minimal_subset(chosen, curve, target):
    for i, c in enumerate(curve):
        if c >= target - 1e-9:
            return chosen[: i + 1]
    return chosen


def run_config(S3, S34, Q, bp_min, sp_min, ms_min, reliability, backbone_sim, target):
    valid, qualified = qualify(Q, bp_min, sp_min, ms_min, reliability)
    if qualified.sum() < 2:
        return None
    nov, label, nb, unique = novelty_and_backbones(S3, S34, valid, qualified, backbone_sim)
    quality = ((Q["BP"] + Q["SP"] + Q["MS"]) / 3).mean(axis=0)
    chosen, curve = greedy_subset(list(np.where(qualified)[0]), label, nb, quality)
    return dict(valid=valid, qualified=qualified, nov=nov, label=label, nb=nb, unique=unique, quality=quality,
                chosen=chosen, curve=curve, subset=minimal_subset(chosen, curve, target))


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def make_figure(models, cases, S34, base, rand_mean, rand_lo, rand_hi, qfirst_curve, sens_sel, sens_qual, args, out_stem: Path):
    names = [model_short_name(m).replace("gemini-3.1-pro-preview", "gemini") for m in models]
    n = len(models)
    qual, sub = base["qualified"], set(base["subset"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        nov_model = np.nanmean(base["nov"], axis=0)
    # models that fail the gate: distinctiveness of ALL their hypotheses vs the valid qualified pool (shows error-driven distance)
    k = len(cases)
    for m in np.where(~qual)[0]:
        vals = []
        for c in range(k):
            pool = [j for j in np.where(base["valid"][c] & qual)[0]]
            if pool:
                vals.append(10 - np.mean([S34[c, m, j] for j in pool]))
        nov_model[m] = np.mean(vals)
    uniq = base["unique"].sum(axis=0)

    fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(19, 6.6), gridspec_kw={"width_ratios": [1.15, 1, 0.95]})
    # A -----------------------------------------------------------------------------------------
    for m in range(n):
        if qual[m]:
            axA.scatter(base["quality"][m], nov_model[m], s=70 + 55 * uniq[m], color=BLUE, edgecolor=RED if m in sub else "white",
                        linewidth=2.4 if m in sub else 0.9, zorder=4)
        else:
            axA.scatter(base["quality"][m], nov_model[m], s=70, facecolor="white", edgecolor=GREY, linewidth=1.6, zorder=3)
    axA.set_xlabel("Mean hypothesis quality (mean of Bio. plausibility, Specificity, Mech. synthesis)", fontsize=9.5)
    axA.set_ylabel("Distinctiveness vs the pool of VALID hypotheses (0-10)", fontsize=9.5)
    axA.set_title("A   Qualify first, then look at diversity", loc="left", fontsize=11, color=INK)
    axA.margins(0.14)
    axA.spines[["top", "right"]].set_visible(False)
    axA.grid(True, color="#ececec", zorder=0)
    from matplotlib.lines import Line2D
    axA.legend(handles=[
        Line2D([], [], marker="o", ls="", color=BLUE, markeredgecolor="white", markersize=8, label=f"qualified (>= {args.reliability}/{len(cases)} valid)"),
        Line2D([], [], marker="o", ls="", color=BLUE, markeredgecolor=RED, markeredgewidth=2.2, markersize=8, label="selected subset"),
        Line2D([], [], marker="o", ls="", markerfacecolor="white", markeredgecolor=GREY, markeredgewidth=1.6, markersize=8, label="not qualified (distinct from errors)"),
    ], loc="upper right", fontsize=8.5, frameon=False)
    axA.text(0.02, 0.02, "marker size = hypotheses no other valid model shares", transform=axA.transAxes, fontsize=7.8, color=MUTED)
    fig.canvas.draw()
    keep = [m for m in range(n) if (m in sub) or (not qual[m]) or uniq[m] >= 3]
    _place_labels(fig, axA, np.array([[base["quality"][m], nov_model[m]] for m in keep]), [names[m] for m in keep])

    # B -----------------------------------------------------------------------------------------
    xs = np.arange(1, len(base["curve"]) + 1)
    axB.fill_between(xs, rand_lo, rand_hi, color=GREY, alpha=0.25, lw=0, label="random order (5-95%)")
    axB.plot(xs, rand_mean, color=GREY, lw=1.6)
    axB.plot(xs, qfirst_curve, color=ORANGE, lw=2, label="highest quality first")
    axB.plot(xs, base["curve"], color=BLUE, lw=2.4, marker="o", ms=5, label="greedy: most new mechanisms first")
    axB.axhline(args.target_coverage, color=MUTED, ls="--", lw=0.9)
    s = len(base["subset"])
    axB.axvline(s, color=RED, ls=":", lw=1.4)
    axB.text(s + 0.15, 0.36, f"{s} models reach\n{base['curve'][s - 1] * 100:.0f}% of valid mechanisms", fontsize=8.8, color=RED, va="bottom")
    axB.set_ylim(0, 1.03)
    axB.set_xlabel("Number of qualified models used", fontsize=9.5)
    axB.set_ylabel("Share of distinct valid mechanistic backbones covered", fontsize=9.5)
    axB.set_title("B   A small subset covers most distinct mechanisms", loc="left", fontsize=11, color=INK)
    axB.legend(loc="lower right", fontsize=8.5, frameon=False)
    axB.spines[["top", "right"]].set_visible(False)
    axB.grid(True, color="#ececec", zorder=0)

    # C -----------------------------------------------------------------------------------------
    order = [m for m in np.argsort(-(sens_sel + 1e-3 * sens_qual)) if sens_qual[m] > 0][:16][::-1]
    ypos = np.arange(len(order))
    axC.barh(ypos, [sens_qual[m] for m in order], color="#c6dbef", height=0.72, label="qualified")
    axC.barh(ypos, [sens_sel[m] for m in order], color=BLUE, height=0.72, label="in the selected subset")
    axC.set_yticks(ypos)
    axC.set_yticklabels([names[m] for m in order], fontsize=8.8)
    axC.set_xlim(0, 1.02)
    axC.set_xlabel("Share of the 18 settings (gate x reliability x backbone threshold)", fontsize=9.5)
    axC.set_title("C   Robustness of the selection", loc="left", fontsize=11, color=INK)
    axC.legend(loc="lower right", fontsize=8.5, frameon=False)
    axC.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Isolating LLMs that give diverse yet high-quality hypotheses", x=0.04, y=1.0, ha="left", fontsize=14, color=INK)
    fig.text(0.04, 0.945,
             f"Valid hypothesis: Bio. plausibility >= {args.bp_min}, Specificity >= {args.sp_min}, Mech. synthesis >= {args.ms_min}.  Qualified model: >= {args.reliability} of {len(cases)} valid.  "
             f"Backbone = same mechanism at P3 >= {args.backbone_sim}.  {n} models, {len(cases)} cases.", fontsize=9, color=MUTED)
    fig.tight_layout(rect=(0, 0.02, 1, 0.93))
    for ext in (".png", ".pdf", ".svg"):
        fig.savefig(out_stem.with_suffix(ext), dpi=250 if ext == ".png" else None, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_stem.with_suffix('.png')} / .pdf / .svg")


# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--exclude-models", nargs="+", default=["Qwen3.5-9B-think-temp0"])
    p.add_argument("--scores", type=str, default=None)
    p.add_argument("--out-dir", type=str, default=None)
    p.add_argument("--bp-min", type=float, default=7)
    p.add_argument("--sp-min", type=float, default=6)
    p.add_argument("--ms-min", type=float, default=5)
    p.add_argument("--reliability", type=int, default=5, help="Qualified = at least this many valid hypotheses (default 5 of 6).")
    p.add_argument("--backbone-sim", type=float, default=7.0)
    p.add_argument("--target-coverage", type=float, default=0.9)
    args = p.parse_args()
    out_dir = Path(args.out_dir) if args.out_dir else OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    models, cases, S3, S34, Q = load_data(args.exclude_models, Path(args.scores) if args.scores else DEFAULT_SCORES)
    n, k = len(models), len(cases)
    base = run_config(S3, S34, Q, args.bp_min, args.sp_min, args.ms_min, args.reliability, args.backbone_sim, args.target_coverage)
    if base is None:
        raise SystemExit("ERROR: fewer than 2 qualified models with these gates.")
    short = [model_short_name(m) for m in models]

    # baselines on the same qualified pool
    cand = list(np.where(base["qualified"])[0])
    rng = np.random.default_rng(0)
    perms = []
    for _ in range(2000):
        order = list(rng.permutation(cand))
        perms.append([coverage(order[: i + 1], base["label"], base["nb"]) for i in range(len(order))])
    perms = np.array(perms)
    qorder = sorted(cand, key=lambda m: -base["quality"][m])
    qfirst = [coverage(qorder[: i + 1], base["label"], base["nb"]) for i in range(len(qorder))]

    # sensitivity grid
    sel_count, qual_count, rows, used = np.zeros(n), np.zeros(n), [], 0
    for bp, rel, bb in itertools.product((7, 8), (4, 5, 6), (6, 7, 8)):
        r = run_config(S3, S34, Q, bp, args.sp_min, args.ms_min, rel, bb, args.target_coverage)
        if r is None:
            rows.append(dict(bp_min=bp, reliability=rel, backbone_sim=bb, n_qualified=int(((Q["BP"] >= bp) & (Q["SP"] >= args.sp_min) & (Q["MS"] >= args.ms_min)).sum(axis=0).__ge__(rel).sum()), subset_size=0, subset=""))
            continue
        used += 1
        qual_count += r["qualified"]
        for m in r["subset"]:
            sel_count[m] += 1
        rows.append(dict(bp_min=bp, reliability=rel, backbone_sim=bb, n_qualified=int(r["qualified"].sum()), subset_size=len(r["subset"]),
                         coverage=r["curve"][len(r["subset"]) - 1], subset=", ".join(short[m] for m in r["subset"])))
    sens = pd.DataFrame(rows)
    sens_qual, sens_sel = qual_count / len(rows), sel_count / len(rows)

    # tables
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # models with no valid hypothesis have no novelty
        nov_model = np.nanmean(base["nov"], axis=0)
    table = pd.DataFrame(dict(
        model=short, mean_quality=base["quality"], valid_hypotheses=base["valid"].sum(axis=0), qualified=base["qualified"],
        novelty_among_valid=nov_model, unique_valid_backbones=base["unique"].sum(axis=0),
        in_selected_subset=[m in set(base["subset"]) for m in range(n)], share_settings_qualified=sens_qual, share_settings_selected=sens_sel))
    table.sort_values(["qualified", "share_settings_selected", "mean_quality"], ascending=False).to_csv(out_dir / "model_table.csv", index=False)
    sens.to_csv(out_dir / "sensitivity.csv", index=False)
    pd.DataFrame(dict(n_models=range(1, len(base["curve"]) + 1), greedy=base["curve"], quality_first=qfirst, random_mean=perms.mean(axis=0),
                      greedy_model=[short[m] for m in base["chosen"]])).to_csv(out_dir / "coverage_curve.csv", index=False)

    s = len(base["subset"])
    lines = ["QUALIFIED + DIVERSE SUBSET", "=" * 70,
             f"Gate: BP >= {args.bp_min}, Specificity >= {args.sp_min}, MechSynthesis >= {args.ms_min}; qualified = >= {args.reliability}/{k} valid hypotheses; backbone = P3 >= {args.backbone_sim}.",
             f"{n} models -> {int(base['qualified'].sum())} qualified; not qualified: " + ", ".join(short[m] for m in np.where(~base['qualified'])[0]),
             f"Distinct valid backbones per case: {base['nb'].tolist()} (cases: {[c.split('_')[0] for c in cases]})",
             f"Selected subset ({s} models reach {base['curve'][s - 1] * 100:.0f}% coverage of distinct valid mechanisms): " + ", ".join(short[m] for m in base["subset"]),
             f"For comparison, {s} models in random order reach {perms.mean(axis=0)[s - 1] * 100:.0f}% on average; the {s} highest-quality models reach {qfirst[s - 1] * 100:.0f}%.", "",
             "Robustness over 18 settings (BP gate 7/8 x reliability 4/5/6 x backbone P3 6/7/8):", sens.to_string(index=False), "",
             table.sort_values(["qualified", "share_settings_selected"], ascending=False).round(2).to_string(index=False)]
    (out_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    make_figure(models, cases, S34, base, perms.mean(axis=0), np.percentile(perms, 5, axis=0), np.percentile(perms, 95, axis=0), qfirst,
                sens_sel, sens_qual, args, out_dir / "qualified_diverse_subset")


if __name__ == "__main__":
    main()
