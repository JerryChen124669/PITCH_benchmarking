"""
Is between-model DIVERSITY of hypotheses proportional to hypothesis QUALITY?

Pool: the models >= --min-total-b (default 200B) plus the anchor (gemini), i.e. the 10-model pool of
FINAL_ge200B_10models_with_gemini. Uses only cached data (pairwise similarity JSONs + judge rubric scores);
makes no API calls.

Definitions
  similarity    mean of pillars P3 (mechanistic hypothesis) and P4 (functional/causal role), 0-10, per
                model pair per case.
  distinctiveness of a hypothesis (one model, one case) = 10 - its mean similarity to the other models
                (same case). Higher = more different from the rest of the pool.
  quality of a hypothesis = mean(Biological Plausibility, Specificity, Grounded Mechanistic Synthesis),
                0-10, from all_scores.csv (the three hypothesis pillars; Zone/Logic/Feature coverage are
                near ceiling and describe tree reading, not the hypothesis).

Tests (all rank-based, permutation/bootstrap so the dependence between pairs is respected)
  A  per hypothesis (n = models x cases): Spearman rho of case-centered distinctiveness vs quality;
     permutation p (quality shuffled across models within each case); 95% CI by bootstrapping models.
  B  per model (n = models): Spearman rho of mean distinctiveness vs mean quality; permutation p;
     95% CI by bootstrapping cases.
  C  per pair: Mantel-type Spearman rho between mean similarity and |quality gap| of the two models;
     permutation of model labels.
  D  pair-case similarity by the quality of the WEAKER hypothesis in the pair (distribution).
With ~10 models the tests have little power: a non-significant rho is NOT evidence of no relationship;
read the confidence intervals.

Outputs (under --out-dir, default .../FINAL_ge200B_10models_with_gemini/diversity_vs_quality/):
    diversity_vs_quality_scatter  .png/.pdf/.svg   4-panel figure (text live, vector)
    hypothesis_level.csv, model_level.csv, pair_level.csv, pair_case_level.csv, stats_summary.txt

Usage:
    python diversity_vs_quality.py
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["svg.fonttype"] = "none"  # live text in .svg, editable in Illustrator
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from analyze_pairwise_clustering import PAIRWISE_DIR, _place_labels, load_long_table, model_short_name
from model_registry import load_registry

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SCORES = SCRIPT_DIR / "evaluation_results_rubric_v12" / "all_scores.csv"
OUT_DIR = PAIRWISE_DIR / "FINAL_ge200B_10models_with_gemini" / "diversity_vs_quality"

SIM_PILLARS = ["P3", "P4"]
QUALITY_COLUMNS = {
    "Biological Plausibility_zone1": "BP",
    "Specificity_zone1": "SP",
    "Grounded Mechanistic Synthesis_zone1": "MS",
}
TIER_LABELS = ["weaker member < 7", "weaker member 7-8", "both members >= 8"]

BLUE = "#4575B4"  # the ramp's blue, used as the single series color
TIER_COLORS = ["#c6dbef", "#6baed6", "#2171b5"]  # one hue, light -> dark = lower -> higher quality
INK = "#222222"
MUTED = "#666666"


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def select_pool(min_total_b: float, exclude: list[str]) -> list[str]:
    reg = load_registry()
    anchor = reg["anchor_slug"]
    pool = [m["slug"] for m in reg["models"]
            if m["slug"] and m["total_b"] >= min_total_b and m["status"] == "evaluated" and m["slug"] not in exclude]
    if anchor not in exclude and anchor not in pool:
        pool.append(anchor)  # the anchor's size is undisclosed, so it is exempt from the size filter
    return sorted(pool)


def pool_from_assignments(path: Path, group: int | None) -> list[str]:
    """Pool = the models in a cluster_assignments.csv (optionally only one category of it)."""
    asg = pd.read_csv(path)
    if group is not None:
        asg = asg[asg["cluster"] == group]
        if asg.empty:
            raise SystemExit(f"ERROR: no models in category {group} of {path}")
    return sorted(asg["model"].tolist())


def load_similarity(pool: list[str]) -> tuple[np.ndarray, list[str]]:
    """S[case, i, j] = mean(P3, P4) similarity of models i and j (symmetric, NaN diagonal)."""
    df = load_long_table(PAIRWISE_DIR)
    df = df[df["model_a"].isin(pool) & df["model_b"].isin(pool)].copy()
    cases = sorted(df["dataset"].unique())
    expected = len(pool) * (len(pool) - 1) // 2 * len(cases)
    if len(df) != expected or df.duplicated(["dataset", "model_a", "model_b"]).any():
        raise SystemExit(f"ERROR: expected {expected} unique pair-case records for {len(pool)} models x {len(cases)} cases, got {len(df)}.")
    df["sim"] = df[SIM_PILLARS].mean(axis=1)
    if df["sim"].isna().any():
        raise SystemExit("ERROR: missing similarity scores.")
    idx = {m: i for i, m in enumerate(pool)}
    S = np.full((len(cases), len(pool), len(pool)), np.nan)
    for r in df.itertuples():
        c = cases.index(r.dataset)
        S[c, idx[r.model_a], idx[r.model_b]] = S[c, idx[r.model_b], idx[r.model_a]] = r.sim
    return S, cases


def load_quality(scores_csv: Path, pool: list[str], cases: list[str]) -> np.ndarray:
    """Q[case, model] = mean(BP, Specificity, Mechanistic Synthesis)."""
    sc = pd.read_csv(scores_csv)
    # all_scores.csv stores raw model ids ("MiniMaxAI/MiniMax-M3"); the pairwise data uses "MiniMaxAI__MiniMax-M3".
    sc["model"] = sc["model"].astype(str).str.replace("/", "__", regex=False)
    sc["case"] = sc["prompt_file"].astype(str).str.replace(r"\.txt$", "", regex=True)
    sc = sc.rename(columns=QUALITY_COLUMNS)
    sc["quality"] = sc[list(QUALITY_COLUMNS.values())].mean(axis=1)
    g = sc.set_index(["case", "model"])["quality"]
    missing = [(c, m) for c in cases for m in pool if (c, m) not in g.index]
    if missing:
        raise SystemExit(f"ERROR: no rubric score for {missing[:5]} ... ({len(missing)} missing). Refusing to continue.")
    return np.array([[g[(c, m)] for m in pool] for c in cases])


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

QUALITY_STAT = "mean"  # how a model's case qualities are combined for the per-model and per-pair views: "mean" or "lowest"


def model_quality(Q: np.ndarray) -> np.ndarray:
    """Per-model quality from the (cases x models) hypothesis qualities: the mean, or the lowest case (the 'quality floor')."""
    return Q.min(axis=0) if QUALITY_STAT == "lowest" else Q.mean(axis=0)


def _qnote() -> str:
    base = "case quality = mean(Biological Plausibility, Specificity, Mechanistic Synthesis)"
    return ("Quality = quality floor = the LOWEST of the model's " + base) if QUALITY_STAT == "lowest" else "Quality = mean(Biological Plausibility, Specificity, Mechanistic Synthesis)"


def _qlabel() -> str:
    return "Quality floor = lowest case quality (0-10)" if QUALITY_STAT == "lowest" else "Mean hypothesis quality (0-10)"


def _rho(x: np.ndarray, y: np.ndarray) -> float:
    return spearmanr(x.ravel(), y.ravel()).statistic


def _centered(M: np.ndarray) -> np.ndarray:
    return M - M.mean(axis=1, keepdims=True)  # remove each case's mean (rows = cases)


def run_tests(D: np.ndarray, Q: np.ndarray, S: np.ndarray, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    k, n = D.shape
    out: dict = {}
    if n < 5:  # a rank test on <5 models is meaningless (rho can only take a handful of values)
        nan = float("nan")
        return {"A": dict(rho=nan, p=nan, ci=(nan, nan), n=k * n), "B": dict(rho=nan, p=nan, ci=(nan, nan), n=n),
                "C": dict(rho=nan, p=nan, n=n * (n - 1) // 2)}

    # A) per hypothesis, case effects removed
    r_a = _rho(_centered(D), _centered(Q))
    perm = [_rho(_centered(D), _centered(np.array([rng.permutation(Q[c]) for c in range(k)]))) for _ in range(10000)]
    boots = []
    for _ in range(4000):
        idx = rng.integers(0, n, n)
        boots.append(_rho(_centered(D[:, idx]), _centered(Q[:, idx])))
    out["A"] = dict(rho=r_a, p=float((np.abs(perm) >= abs(r_a)).mean()), ci=tuple(np.nanpercentile(boots, [2.5, 97.5])), n=k * n)

    # B) per model
    Dm, Qm = D.mean(axis=0), model_quality(Q)
    r_b = spearmanr(Dm, Qm).statistic
    p_b = float(np.mean([abs(spearmanr(Dm, rng.permutation(Qm)).statistic) >= abs(r_b) for _ in range(20000)]))
    bb = []
    for _ in range(4000):
        idx = rng.integers(0, k, k)
        bb.append(spearmanr(D[idx].mean(axis=0), model_quality(Q[idx])).statistic)
    out["B"] = dict(rho=r_b, p=p_b, ci=tuple(np.nanpercentile(bb, [2.5, 97.5])), n=n)

    # C) per pair (Mantel-type): mean similarity vs |quality gap|, permuting model labels
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # the all-NaN diagonal
        Sbar = np.nanmean(S, axis=0)
    iu = np.triu_indices(n, 1)
    dQ = np.abs(Qm[:, None] - Qm[None, :])
    r_c = spearmanr(Sbar[iu], dQ[iu]).statistic
    p_c = float(np.mean([abs(spearmanr(Sbar[iu], dQ[np.ix_(pp, pp)][iu]).statistic) >= abs(r_c)
                         for pp in (rng.permutation(n) for _ in range(20000))]))
    out["C"] = dict(rho=r_c, p=p_c, n=len(iu[0]))
    return out


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _short(name: str) -> str:
    return name.replace("gemini-3.1-pro-preview", "gemini")


def _style(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, color="#ececec", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=9, colors=INK)


def _stat_text(s: dict, ci: bool = True) -> str:
    if not np.isfinite(s["rho"]):
        return "too few models\nfor a rank test"
    t = f"Spearman rho = {s['rho']:+.2f}"
    if ci:
        t += f"\n95% CI [{s['ci'][0]:+.2f}, {s['ci'][1]:+.2f}]"
    return t + f"\npermutation p = {s['p']:.2f}"


def _stat_box(ax, text: str, loc: str = "upper right") -> None:
    x, ha = (0.97, "right") if "right" in loc else (0.03, "left")
    y, va = (0.96, "top") if "upper" in loc else (0.04, "bottom")
    ax.text(x, y, text, transform=ax.transAxes, ha=ha, va=va, fontsize=8.5, color=INK, zorder=6,
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#cccccc", lw=0.6, alpha=0.92))


def _draw_pair_panel(ax, S: np.ndarray, Qm: np.ndarray, names: list[str], stats: dict, title: str) -> None:
    """Per model pair: mean similarity (P3+P4, over the cases) vs |gap in mean hypothesis quality|."""
    n = len(names)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # the all-NaN diagonal
        Sbar = np.nanmean(S, axis=0)
    iu = np.triu_indices(n, 1)
    gap = np.abs(Qm[:, None] - Qm[None, :])[iu]
    sim = Sbar[iu]
    ax.scatter(gap, sim, s=52, color=BLUE, edgecolor="white", linewidth=0.9, alpha=0.9, zorder=3)
    c = stats["C"]
    _stat_box(ax, _stat_text(c, ci=False), "upper right")
    # label the three least-similar pairs, staggered by their x order (left one up-left, middle one down-right,
    # right one up-right) so the labels cannot collide
    lowest = sorted(np.argsort(sim)[:3], key=lambda j: gap[j])
    for j, (dx, dy, ha) in zip(lowest, [(-10, 10, "right"), (10, -8, "left"), (12, 12, "left")]):
        ax.annotate(f"{_short(names[iu[0][j]])} - {_short(names[iu[1][j]])}", (gap[j], sim[j]), xytext=(dx, dy),
                    textcoords="offset points", ha=ha, va="center", fontsize=8, color=INK, zorder=5,
                    arrowprops=dict(arrowstyle="-", color="#b0b0b0", lw=0.6, shrinkA=0, shrinkB=4))
    ax.set_xlabel(f"|quality gap| between the two models ({'quality floor = lowest case quality' if QUALITY_STAT == 'lowest' else 'mean hypothesis quality'})", fontsize=9.5)
    ax.set_ylabel("Mean similarity of the pair (P3+P4, 0-10)", fontsize=9.5)
    ax.set_title(title.format(n=c["n"]), loc="left", fontsize=11, color=INK)
    ax.margins(0.12)
    _style(ax)


def _draw_model_panel(fig, ax, D, Q, names, stats, title: str) -> None:
    """Per model: mean hypothesis quality (x) vs mean distinctiveness (y); bars = +/-1 SE across the cases."""
    k = D.shape[0]
    Qm, Dm = model_quality(Q), D.mean(axis=0)
    # SE bars for the mean; a minimum has no standard error, so the floor view shows distinctiveness bars only
    Qs = Q.std(axis=0, ddof=1) / np.sqrt(k) if QUALITY_STAT == "mean" else None
    Ds = D.std(axis=0, ddof=1) / np.sqrt(k)
    ax.errorbar(Qm, Dm, xerr=Qs, yerr=Ds, fmt="none", ecolor="#bdbdbd", elinewidth=0.9, capsize=0, zorder=2)
    ax.scatter(Qm, Dm, s=110, color=BLUE, edgecolor="white", linewidth=1.0, zorder=3)
    slope, icpt = np.polyfit(Qm, Dm, 1)
    xs = np.linspace(Qm.min() - 0.1, Qm.max() + 0.1, 50)
    ax.plot(xs, icpt + slope * xs, color=MUTED, ls="--", lw=1.2, zorder=1)
    b = stats["B"]
    _stat_box(ax, _stat_text(b), "upper right")
    ax.set_xlabel(_qlabel(), fontsize=9.5)
    ax.set_ylabel("Mean distinctiveness (0-10)", fontsize=9.5)
    ax.set_title(title.format(n=b["n"], k=k), loc="left", fontsize=11, color=INK)
    ax.margins(0.16)
    _style(ax)
    fig.canvas.draw()
    _place_labels(fig, ax, np.column_stack([Qm, Dm]), [_short(nm) for nm in names])


def make_panel_b_figure(D, Q, pool, stats, out_stem: Path, title_note: str) -> None:
    """Panel B on its own: a standalone figure (png / pdf / svg)."""
    names = [model_short_name(m) for m in pool]
    fig, ax = plt.subplots(figsize=(9.2, 7.2))
    _draw_model_panel(fig, ax, D, Q, names, stats, "Per model (n = {n}; bars = +/-1 SE across the {k} cases)")
    fig.suptitle(("Quality floor" if QUALITY_STAT == "lowest" else "Mean hypothesis quality") + " vs mean distinctiveness",
                 x=0.06, y=0.995, ha="left", fontsize=13, color=INK)
    fig.text(0.06, 0.945, title_note, ha="left", fontsize=9, color=MUTED)
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))
    fig.text(0.06, 0.012,
             "Distinctiveness = 10 - mean P3+P4 similarity to the other models on the same case. " + _qnote() + ".\n"
             "Spearman rho with a permutation test; bars are +/-1 standard error across cases; dashed line is a visual reference.",
             fontsize=7.8, color=MUTED)
    for ext in (".png", ".pdf", ".svg"):
        fig.savefig(out_stem.with_suffix(ext), dpi=250 if ext == ".png" else None, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_stem.with_suffix('.png')} / .pdf / .svg")


def make_panel_c_figure(S, Q, pool, stats, out_stem: Path, title_note: str) -> None:
    """Panel C on its own: a standalone figure (png / pdf / svg)."""
    names = [model_short_name(m) for m in pool]
    fig, ax = plt.subplots(figsize=(8.6, 6.9))
    _draw_pair_panel(ax, S, model_quality(Q), names, stats, "Per model pair (n = {n}): similarity vs quality gap")
    fig.suptitle("Is model-pair similarity related to the quality gap?", x=0.06, y=0.995, ha="left", fontsize=13, color=INK)
    fig.text(0.06, 0.945, title_note, ha="left", fontsize=9, color=MUTED)
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))
    fig.text(0.06, 0.012,
             "Each dot = one pair of models. Similarity = mean P3+P4 over the cases; "
             + ("quality gap = |difference of the models' quality|.\n" + _qnote() + ". " if QUALITY_STAT == "lowest" else
                "quality gap = |difference of the models' mean hypothesis quality|\n(mean of Biological Plausibility, Specificity, Mechanistic Synthesis). ")
             + "Spearman rho; Mantel-type permutation test (model labels permuted).",
             fontsize=7.8, color=MUTED)
    for ext in (".png", ".pdf", ".svg"):
        fig.savefig(out_stem.with_suffix(ext), dpi=250 if ext == ".png" else None, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_stem.with_suffix('.png')} / .pdf / .svg")


def make_figure(D, Q, S, pool, cases, stats, pair_cases: pd.DataFrame, out_stem: Path, title_note: str) -> None:
    k, n = D.shape
    names = [model_short_name(m) for m in pool]
    fig, axs = plt.subplots(2, 2, figsize=(13.5, 11.2))
    axA, axB, axC, axD = axs.ravel()

    # ---- A: per hypothesis (case-centered) ------------------------------------------------------
    Dc, Qc = _centered(D), _centered(Q)
    x, y = Qc.ravel(), Dc.ravel()
    axA.axhline(0, color="#cccccc", lw=0.8, zorder=1)
    axA.axvline(0, color="#cccccc", lw=0.8, zorder=1)
    axA.scatter(x, y, s=52, color=BLUE, edgecolor="white", linewidth=0.9, alpha=0.9, zorder=3)
    slope, icpt = np.polyfit(x, y, 1)
    xs = np.linspace(x.min(), x.max(), 50)
    axA.plot(xs, icpt + slope * xs, color=MUTED, ls="--", lw=1.2, zorder=2)
    for order, idx in enumerate(sorted(np.where(y > 1.5)[0], key=lambda q: -y[q])[:5]):  # name the (up to 5) most distinctive hypotheses
        i, j = divmod(int(idx), n)
        dy = (0, 6, -7, 6, -7)[order % 5]  # nudge neighbouring labels apart
        axA.annotate(f"{_short(names[j])} / {cases[i].split('_')[0].split('-')[0]}", (x[idx], y[idx]), xytext=(8, dy),
                     textcoords="offset points", va="center", fontsize=8, color=INK, zorder=5)
    a = stats["A"]
    _stat_box(axA, _stat_text(a), "upper left")  # upper right hides the labels of the most distinctive hypotheses
    axA.set_xlabel("Hypothesis quality, relative to the case mean\n(mean of Bio. plausibility, Specificity, Mech. synthesis)", fontsize=9.5)
    axA.set_ylabel("Distinctiveness, relative to the case mean\n(10 - mean similarity to the other models)", fontsize=9.5)
    axA.set_title(f"A   Per hypothesis (n = {a['n']}; each dot = one model on one case)", loc="left", fontsize=11, color=INK)
    _style(axA)

    # ---- B: per model -----------------------------------------------------------------------------
    Qm = model_quality(Q)
    _draw_model_panel(fig, axB, D, Q, names, stats, "B   Per model (n = {n}; bars = +/-1 SE across the {k} cases)")

    # ---- C: per pair ------------------------------------------------------------------------------
    _draw_pair_panel(axC, S, Qm, names, stats, "C   Per model pair (n = {n}; similarity vs quality gap)")

    # ---- D: similarity distribution by quality tier ---------------------------------------------
    rng = np.random.default_rng(1)
    y_floor = float(np.floor(pair_cases["sim"].min())) - 2.1  # free band under the data for the tier summaries
    for i, tier in enumerate(TIER_LABELS):
        v = pair_cases.loc[pair_cases["tier"] == tier, "sim"].to_numpy()
        if len(v) == 0:  # no pair in this quality tier (e.g. a group whose members are all strong)
            axD.text(i, y_floor + 0.12, "n = 0", ha="center", va="bottom", fontsize=8.5, color=MUTED)
            continue
        axD.scatter(i + rng.uniform(-0.28, 0.28, len(v)), v + rng.uniform(-0.06, 0.06, len(v)), s=26, color=TIER_COLORS[i],
                    edgecolor="white", linewidth=0.5, alpha=0.9, zorder=3)
        axD.hlines(np.median(v), i - 0.36, i + 0.36, color=INK, lw=2.2, zorder=5)
        axD.text(i, y_floor + 0.12, f"n = {len(v)}\nmedian {np.median(v):.1f}\n{(v <= 6).mean() * 100:.0f}% at <= 6",
                 ha="center", va="bottom", fontsize=8.5, color=INK)
    axD.axhline(6, color=MUTED, ls="--", lw=1.0, zorder=2)
    axD.text(2.5, 6.0, "<= 6:\nsubstantially\ndifferent\nintegration", ha="left", va="center", fontsize=8, color=MUTED)
    axD.set_xticks(range(len(TIER_LABELS)))
    axD.set_xticklabels([t.replace(" member", "\nmember") for t in TIER_LABELS], fontsize=9.5)
    axD.set_xlim(-0.6, 3.2)
    axD.set_ylim(y_floor, 10.4)
    axD.set_yticks(range(2, 11, 2))  # no ticks in the band reserved for the tier summaries
    axD.set_ylabel("Pair similarity on one case (P3+P4, 0-10)", fontsize=9.5)
    axD.set_xlabel("Quality of the weaker hypothesis in the pair", fontsize=9.5)
    axD.set_title(f"D   Similarity by quality tier ({len(pair_cases)} pair-cases; line = median)", loc="left", fontsize=11, color=INK)
    _style(axD)
    axD.grid(False, axis="x")

    fig.suptitle("Is between-model diversity proportional to hypothesis quality?", x=0.06, y=0.995, ha="left", fontsize=14, color=INK)
    fig.text(0.06, 0.955, title_note, ha="left", fontsize=9.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 0.955))
    fig.text(0.06, 0.012,
             "Distinctiveness = 10 - mean P3+P4 similarity to the other models. "
             + ("Panels A, D: per-hypothesis quality; panels B, C: " if QUALITY_STAT == "lowest" else "") + _qnote() + ". "
             "Rank-based tests; dashed lines are visual reference only."
             + (" With ~10 models a non-significant rho is not evidence of no relationship." if n < 20 else ""),
             fontsize=7.8, color=MUTED, wrap=True)
    for ext in (".png", ".pdf", ".svg"):
        fig.savefig(out_stem.with_suffix(ext), dpi=250 if ext == ".png" else None, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_stem.with_suffix('.png')} / .pdf / .svg")


# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--min-total-b", type=float, default=200.0, help="Pool = models with total parameters >= this (billions) + the anchor.")
    parser.add_argument("--exclude-models", nargs="+", default=["Qwen3.8-9B-think-temp0"])
    parser.add_argument("--assignments", type=str, default=None,
                        help="Use the models of this cluster_assignments.csv as the pool (overrides --min-total-b).")
    parser.add_argument("--group", type=int, default=None,
                        help="With --assignments: only this category's models (distinctiveness is then measured against its members only).")
    parser.add_argument("--label", type=str, default=None, help="Text for the figure subtitle describing the pool.")
    parser.add_argument("--exclude-cases", nargs="+", default=[], help="Substrings of case names to drop (e.g. IL2RA pRPS6).")
    parser.add_argument("--quality-stat", choices=["mean", "lowest"], default="mean",
                        help="Per-model quality in the per-model and per-pair views: the mean over the cases (default) or the "
                             "lowest case (the 'quality floor'). Per-hypothesis views are unaffected.")
    parser.add_argument("--scores", type=str, default=None, help=f"Default: {DEFAULT_SCORES}")
    parser.add_argument("--out-dir", type=str, default=None, help=f"Default: {OUT_DIR}")
    args = parser.parse_args()
    global QUALITY_STAT
    QUALITY_STAT = args.quality_stat

    out_dir = Path(args.out_dir) if args.out_dir else OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    pool = pool_from_assignments(Path(args.assignments), args.group) if args.assignments else select_pool(args.min_total_b, args.exclude_models)
    S, cases = load_similarity(pool)
    Q = load_quality(Path(args.scores) if args.scores else DEFAULT_SCORES, pool, cases)
    if args.exclude_cases:  # drop whole cases from both the similarity and the quality side
        keep = [i for i, c in enumerate(cases) if not any(x.lower() in c.lower() for x in args.exclude_cases)]
        cases, S, Q = [cases[i] for i in keep], S[keep], Q[keep]
    n, k = len(pool), len(cases)
    print(f"Pool: {n} models x {k} cases: {[model_short_name(m) for m in pool]}")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        D = 10 - np.nanmean(S, axis=2)  # (cases x models): distinctiveness of each hypothesis
    stats = run_tests(D, Q, S)

    # ---- tables -------------------------------------------------------------------------------------
    short = [model_short_name(m) for m in pool]
    Dc, Qc = _centered(D), _centered(Q)
    pd.DataFrame([dict(model=short[j], case=cases[i], quality=Q[i, j], distinctiveness=D[i, j],
                       quality_case_centered=Qc[i, j], distinctiveness_case_centered=Dc[i, j])
                  for i in range(k) for j in range(n)]).to_csv(out_dir / "hypothesis_level.csv", index=False)
    pd.DataFrame(dict(model=short, mean_quality=Q.mean(axis=0), quality_floor_lowest_case=Q.min(axis=0), sd_quality=Q.std(axis=0, ddof=1),
                      mean_distinctiveness=D.mean(axis=0), sd_distinctiveness=D.std(axis=0, ddof=1))
                 ).sort_values("mean_distinctiveness", ascending=False).to_csv(out_dir / "model_level.csv", index=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        Sbar = np.nanmean(S, axis=0)
    Qm = model_quality(Q)
    iu = np.triu_indices(n, 1)
    pd.DataFrame([dict(model_a=short[i], model_b=short[j], mean_similarity=Sbar[i, j], quality_gap=abs(Qm[i] - Qm[j]))
                  for i, j in zip(*iu)]).to_csv(out_dir / "pair_level.csv", index=False)
    rows = [dict(case=cases[c], model_a=short[i], model_b=short[j], sim=S[c, i, j], quality_a=Q[c, i], quality_b=Q[c, j],
                 min_quality=min(Q[c, i], Q[c, j])) for c in range(k) for i, j in zip(*iu)]
    pc = pd.DataFrame(rows)
    pc["tier"] = pd.cut(pc["min_quality"], [0, 7, 8, 11], labels=TIER_LABELS, right=False).astype(str)
    pc.to_csv(out_dir / "pair_case_level.csv", index=False)

    tier_tab = pc.groupby("tier")["sim"].agg(n="count", mean="mean", sd="std", share_le_6=lambda s: (s <= 6).mean()).reindex(TIER_LABELS)
    lines = ["DIVERSITY vs HYPOTHESIS QUALITY", "=" * 70,
             f"Pool: {n} models x {k} cases = {n * k} hypotheses; {len(pc)} pair-cases.",
             "distinctiveness = 10 - mean P3+P4 similarity to the other models; hypothesis quality = mean(BP, Specificity, MechSynthesis)",
             f"per-model / per-pair views use quality = {'the LOWEST case (quality floor)' if QUALITY_STAT == 'lowest' else 'the mean over the cases'}", ""]
    for key, label in (("A", "per hypothesis, case-centered"), ("B", "per model"), ("C", "per pair (similarity vs |quality gap|)")):
        s = stats[key]
        ci = f"; 95% CI [{s['ci'][0]:+.2f}, {s['ci'][1]:+.2f}]" if "ci" in s else ""
        lines.append(f"{key}) {label} (n={s['n']}): Spearman rho = {s['rho']:+.2f}{ci}; permutation p = {s['p']:.3f}")
    lines += ["", "D) pair-case P3+P4 similarity by the quality of the weaker hypothesis:", tier_tab.round(2).to_string(), "",
              "model-level (most distinctive first):"]
    ml = pd.read_csv(out_dir / "model_level.csv")
    lines += [f"   {r.model:24s} distinctiveness {r.mean_distinctiveness:.2f}   quality {r.mean_quality:.2f}" for r in ml.itertuples()]
    (out_dir / "stats_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))

    size_desc = args.label or (f">= {args.min_total_b:g}B incl. gemini" if args.min_total_b else "all sizes")
    note = f"{n} models ({size_desc}), {k} cases; similarity = P3+P4"
    make_figure(D, Q, S, pool, cases, stats, pc, out_dir / "diversity_vs_quality_scatter", note)
    make_panel_c_figure(S, Q, pool, stats, out_dir / "diversity_vs_quality_panel_C_pairs", note)
    make_panel_b_figure(D, Q, pool, stats, out_dir / "diversity_vs_quality_panel_B_models", note)


if __name__ == "__main__":
    main()
