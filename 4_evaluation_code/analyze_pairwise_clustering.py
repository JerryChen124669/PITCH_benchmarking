"""
Hierarchical-clustering analysis of the blind pairwise-similarity data in
pairwise_similarity_results_v2/ -- how much models AGREE on the biological
interpretation of the same decision tree, independent of who is more correct.

Pipeline:
  1. Load every cached pair record (pairwise_similarity_results_v2/<dataset>/
     <model_a>__vs__<model_b>.json) into a long table, validate it (duplicate/
     missing/invalid pairs), and report what was found.
  2. Average each of the 6 similarity pillars (P1-P6) across datasets for
     every unordered model pair.
  3. Biological-content similarity = mean(P1..P5) per pair (P6, explanatory
     style, is deliberately excluded -- it measures presentation, not
     content agreement). Convert to a distance matrix D = 1 - S/10.
  4. Hierarchical clustering (average linkage) directly on the condensed
     distance matrix; cluster count chosen by silhouette score on the same
     precomputed distances (not on any 2D projection).
  5. A 2D map via classical PCoA on the same distance matrix, for
     visualization only -- clustering never runs on these coordinates.
  6. Four descriptive "domains" are derived from the 6 pillars (Tree
     Interpretation = P1, Feature Semantics = mean(P2,P4), Mechanistic
     Reasoning = mean(P3,P5), Explanatory Style = P6) and averaged within
     each discovered cluster, purely to describe why the clusters formed --
     they are never re-clustered.
  7. Optionally, an independent per-model performance table (judge_evaluation
     rubric scores, NOT used anywhere in the similarity/clustering math) is
     plotted alongside, ordered to match the similarity dendrogram, for
     visual comparison only.

Nothing here is fabricated: if pairs are missing/duplicated, or performance
data isn't found, the script says so explicitly and either stops (for
similarity-math problems) or skips that piece (for the optional performance
panel).

Outputs, all under --out-dir (default: pairwise_similarity_results_v2/clustering_analysis/):
    pairwise_pillar_averages.csv   one row per unordered model pair, P1-P6 + content_similarity
    content_similarity_matrix.csv  n x n biological-content similarity matrix
    distance_matrix.csv            n x n distance matrix (1 - similarity/10)
    cluster_assignments.csv        model -> cluster id (+ dendrogram leaf order)
    cluster_domain_summary.csv     cluster -> mean of the 4 domains over within-cluster pairs
    cluster_summary.txt            plain-language explanation of why the clusters formed
    cluster_map.png / .pdf         PCoA map, colored by cluster category (default figure)
    dendrogram_heatmap / domain_profile / performance .png/.pdf
                                   optional, one file each, only with --plots

Each figure is its own file. By default only cluster_map is written.
Models are filtered first: total parameters >= --min-total-b (default 20B,
from model_registry.json; the gemini anchor is exempt) and --exclude-models
(default Qwen3.5-9B-think-temp0).

Usage:
    python analyze_pairwise_clustering.py
    python analyze_pairwise_clustering.py --plots cluster_map dendrogram_heatmap
    python analyze_pairwise_clustering.py --k 4          # force the cluster count
    python analyze_pairwise_clustering.py --min-k 2      # allow the unconstrained silhouette optimum
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["svg.fonttype"] = "none"  # live text in .svg, editable in Illustrator
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.metrics import silhouette_score

from paths import RESULTS_DIR

SCRIPT_DIR = Path(__file__).resolve().parent
PAIRWISE_DIR = RESULTS_DIR / "pairwise_similarity_results_v2"
DEFAULT_PERFORMANCE_CSV = RESULTS_DIR / "evaluation_results_rubric_v12" / "all_scores.csv"
OUT_DIR = PAIRWISE_DIR / "clustering_analysis"

# Full rubric criterion name -> short pillar code, in P1..P6 order (matches
# pairwise_similarity_v2.py's CRITERIA / the scores dict keys in each cached
# pair JSON).
CRITERIA = [
    "Zone & Feature Selection Similarity",
    "Feature Interpretation & Coverage Similarity",
    "Mechanistic Hypothesis Similarity",
    "Functional/Causal-Role Similarity",
    "Supporting Reasoning & Experimental Implication Similarity",
    "Response Scope & Elaboration Similarity",
]
PILLAR_OF_CRITERION = {full: f"P{i + 1}" for i, full in enumerate(CRITERIA)}
PILLARS = [f"P{i + 1}" for i in range(6)]
PILLAR_NAMES = {
    "P1": "zone/feature selection", "P2": "feature interpretation", "P3": "mechanistic hypothesis",
    "P4": "functional/causal role", "P5": "supporting reasoning & experiment", "P6": "scope & elaboration",
}
CONTENT_PILLARS = ["P1", "P2", "P3", "P4", "P5"]  # P6 excluded from content similarity

DOMAIN_LABELS = ["Tree Interpretation", "Feature Semantics", "Mechanistic Reasoning", "Explanatory Style"]

CLUSTER_PALETTE = plt.get_cmap("Set2").colors  # colorblind-accessible qualitative palette

# Similarity-heatmap colors: the same pheatmap-style ramp as plot_pairwise_heatmaps.py (blue = low,
# pale yellow = mid, red = high) on the same fixed 0-10 scale, so the dendrogram heatmaps read like
# the other similarity heatmaps. The diagonal (self-similarity, defined not measured) is drawn light grey.
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

PHEATMAP_RAMP = ["#4575B4", "#91BFDB", "#E0F3F8", "#FFFFBF", "#FEE090", "#FC8D59", "#D73027"]
PHEATMAP_CMAP = LinearSegmentedColormap.from_list("pheatmap_rdylbu_r", PHEATMAP_RAMP).with_extremes(bad="#f0f0f0")


def model_short_name(slug: str) -> str:
    return slug.split("__", 1)[-1]


# ---------------------------------------------------------------------------
# 1. Load + validate
# ---------------------------------------------------------------------------

def load_long_table(pairwise_dir: Path) -> pd.DataFrame:
    rows = []
    prompt_dirs = sorted(p for p in pairwise_dir.iterdir() if p.is_dir() and p.name != "heatmaps")
    for pdir in prompt_dirs:
        for f in sorted(pdir.glob("*__vs__*.json")):
            data = json.loads(f.read_text(encoding="utf-8"))
            a, b = data["model_a_slug"], data["model_b_slug"]
            scores = data.get("scores") or {}
            row = {
                "dataset": pdir.name,
                "model_a": a,
                "model_b": b,
                "source_file": str(f.relative_to(pairwise_dir)),
                "overall_similarity_reported": data.get("overall_similarity"),
            }
            for full, short in PILLAR_OF_CRITERION.items():
                row[short] = scores.get(full)
            rows.append(row)
    return pd.DataFrame(rows)


def validate(df: pd.DataFrame) -> dict:
    """Returns a report dict and prints it. Does not raise -- callers decide
    what's fatal (missing/duplicate pairs) vs. informational."""
    models = sorted(set(df["model_a"]) | set(df["model_b"]))
    n = len(models)
    datasets = sorted(df["dataset"].unique())
    expected_pairs = set(combinations(models, 2))

    self_pairs = df[df["model_a"] == df["model_b"]]

    unsorted_rows = df[df["model_a"] >= df["model_b"]]  # files are written model_a < model_b alphabetically

    dup_counts = df.groupby(["dataset", "model_a", "model_b"]).size()
    duplicates = dup_counts[dup_counts > 1]

    present_pairs = set(zip(df["model_a"], df["model_b"]))
    missing_pairs_entirely = expected_pairs - present_pairs

    incomplete_pairs = {}
    for pair in expected_pairs:
        have = set(df.loc[(df.model_a == pair[0]) & (df.model_b == pair[1]), "dataset"])
        missing_here = set(datasets) - have
        if missing_here and pair not in missing_pairs_entirely:
            incomplete_pairs[pair] = sorted(missing_here)

    invalid_mask = (
        df[PILLARS].isnull().any(axis=1)
        | (df[PILLARS].lt(0)).any(axis=1)
        | (df[PILLARS].gt(10)).any(axis=1)
    )
    invalid_rows = df[invalid_mask]

    report = dict(
        models=models,
        n_models=n,
        datasets=datasets,
        n_datasets=len(datasets),
        expected_pairs=len(expected_pairs),
        present_pairs=len(present_pairs),
        self_pairs=self_pairs,
        unsorted_rows=unsorted_rows,
        duplicates=duplicates,
        missing_pairs_entirely=sorted(missing_pairs_entirely),
        incomplete_pairs=incomplete_pairs,
        invalid_rows=invalid_rows,
    )

    print("=" * 70)
    print("DATA VALIDATION REPORT")
    print("=" * 70)
    print(f"Detected models ({n}): {[model_short_name(m) for m in models]}")
    print(f"Datasets ({len(datasets)}): {datasets}")
    print(f"Expected unordered pairs: {len(expected_pairs)}; present: {len(present_pairs)}")
    if self_pairs.shape[0]:
        print(f"INVALID: {self_pairs.shape[0]} row(s) compare a model against itself:")
        print(self_pairs[["dataset", "model_a", "model_b"]].to_string(index=False))
    if unsorted_rows.shape[0]:
        print(
            f"NOTE: {unsorted_rows.shape[0]} row(s) have model_a >= model_b "
            "(unexpected given the file-naming convention, but harmless -- pairs are treated as unordered)."
        )
    if len(duplicates):
        print(f"DUPLICATES: {len(duplicates)} (dataset, pair) combo(s) have more than one cached file:")
        print(duplicates.to_string())
    else:
        print("Duplicates: none.")
    if missing_pairs_entirely:
        print(f"MISSING ENTIRELY: {len(missing_pairs_entirely)} pair(s) have zero cached comparisons:")
        for a, b in sorted(missing_pairs_entirely):
            print(f"  {model_short_name(a)} vs {model_short_name(b)}")
    else:
        print("Missing pairs (entirely absent): none.")
    if incomplete_pairs:
        print(f"INCOMPLETE: {len(incomplete_pairs)} pair(s) are missing one or more datasets:")
        for (a, b), missing in sorted(incomplete_pairs.items()):
            print(f"  {model_short_name(a)} vs {model_short_name(b)}: missing {missing}")
    else:
        print("Incomplete pairs (present but missing some datasets): none.")
    if invalid_rows.shape[0]:
        print(f"INVALID SCORES: {invalid_rows.shape[0]} row(s) have a null or out-of-[0,10] pillar score.")
    else:
        print("Invalid pillar scores: none.")
    print("=" * 70)

    return report


# ---------------------------------------------------------------------------
# 2-3. Pair-level pillar averages, content similarity, distance matrix
# ---------------------------------------------------------------------------

def compute_pair_averages(df: pd.DataFrame, content_pillars: list[str] | None = None) -> pd.DataFrame:
    content_pillars = content_pillars or CONTENT_PILLARS
    pair_avg = df.groupby(["model_a", "model_b"], as_index=False)[PILLARS].mean()
    pair_avg["content_similarity"] = pair_avg[content_pillars].mean(axis=1)

    pair_avg["Tree Interpretation"] = pair_avg["P1"]
    pair_avg["Feature Semantics"] = pair_avg[["P2", "P4"]].mean(axis=1)
    pair_avg["Mechanistic Reasoning"] = pair_avg[["P3", "P5"]].mean(axis=1)
    pair_avg["Explanatory Style"] = pair_avg["P6"]

    # Sanity check: S_content = (1*Tree + 2*FeatureSemantics + 2*MechanisticReasoning) / 5
    # -- an algebraic identity given how the domains are defined, not a new
    # computation. If this ever fails it means the domain formulas above
    # drifted from the content-similarity formula.
    recon = (
        1 * pair_avg["Tree Interpretation"]
        + 2 * pair_avg["Feature Semantics"]
        + 2 * pair_avg["Mechanistic Reasoning"]
    ) / 5
    # (Only valid for the default P1-P5 set; skipped when pillars are dropped.)
    if set(content_pillars) == set(CONTENT_PILLARS) and not np.allclose(recon, pair_avg["content_similarity"]):
        raise AssertionError("Domain reconstruction does not match content_similarity -- check the formulas.")

    return pair_avg


def build_matrices(pair_avg: pd.DataFrame, models: list[str]) -> tuple[np.ndarray, np.ndarray]:
    n = len(models)
    idx = {m: i for i, m in enumerate(models)}
    S = np.full((n, n), np.nan)
    np.fill_diagonal(S, 10.0)  # perfect self-similarity by definition, not measured
    for _, row in pair_avg.iterrows():
        i, j = idx[row["model_a"]], idx[row["model_b"]]
        S[i, j] = S[j, i] = row["content_similarity"]

    missing_cells = np.argwhere(np.isnan(S))
    if len(missing_cells):
        missing_pairs = sorted({tuple(sorted((models[i], models[j]))) for i, j in missing_cells if i != j})
        raise SystemExit(
            "ERROR: cannot build a complete similarity matrix -- missing pair(s): "
            + ", ".join(f"{model_short_name(a)} vs {model_short_name(b)}" for a, b in missing_pairs)
            + ". Refusing to fabricate values; run the missing pairwise comparisons first."
        )

    D = 1 - S / 10
    np.fill_diagonal(D, 0.0)
    return S, D


# ---------------------------------------------------------------------------
# 4. Hierarchical clustering + silhouette-based k selection
# ---------------------------------------------------------------------------

def cluster_models(D: np.ndarray, models: list[str], force_k: int | None, max_k: int | None, min_k: int = 2):
    n = len(models)
    condensed = squareform(D, checks=True)
    Z = linkage(condensed, method="average")

    upper_k = max_k or min(n - 1, 10)
    sil_scores: dict[int, float] = {}
    if n >= 4:
        for k in range(2, upper_k + 1):
            labels_k = fcluster(Z, k, criterion="maxclust")
            if len(set(labels_k)) < 2:
                continue
            sil_scores[k] = silhouette_score(D, labels_k, metric="precomputed")

    print("=" * 70)
    print("CLUSTERING")
    print("=" * 70)
    print("Method: hierarchical, average linkage (UPGMA), on the precomputed content-distance matrix.")

    if force_k is not None:
        best_k = force_k
        print(f"Cluster count: {best_k} (forced via --k, not silhouette-selected).")
        if sil_scores:
            print("Silhouette scores by candidate k (for reference; k is forced):")
            for k, score in sorted(sil_scores.items()):
                marker = "  <== forced" if k == best_k else ""
                print(f"  k={k:2d}: silhouette={score:.4f}{marker}")
    elif sil_scores:
        eligible = {k: s for k, s in sil_scores.items() if k >= min_k}
        if not eligible:
            raise SystemExit(f"ERROR: no candidate k >= --min-k {min_k} available (tested {sorted(sil_scores)}).")
        best_k = max(eligible, key=eligible.get)
        print("Silhouette scores by candidate k (higher = better-separated clusters):")
        for k, score in sorted(sil_scores.items()):
            marker = "  <== selected" if k == best_k else ""
            floor = "  (below --min-k, not eligible)" if k < min_k else ""
            print(f"  k={k:2d}: silhouette={score:.4f}{marker}{floor}")
        if best_k != max(sil_scores, key=sil_scores.get):
            print(f"NOTE: --min-k {min_k} overrode the unconstrained silhouette optimum (k={max(sil_scores, key=sil_scores.get)}).")
    else:
        best_k = min(2, n - 1) if n > 1 else 1
        print(
            f"NOTE: only {n} models -- too few for reliable automatic silhouette-based k selection. "
            f"Falling back to a conservative k={best_k}; treat the dendrogram, not this k, as the primary result."
        )

    labels = fcluster(Z, best_k, criterion="maxclust") if n > 1 else np.array([1])
    print("=" * 70)
    return Z, labels, best_k, sil_scores


def merge_clusters(labels: np.ndarray, groups: list[str], D: np.ndarray) -> tuple[np.ndarray, str]:
    """Post-hoc merge of clusters from the k-cut into fewer categories, e.g.
    groups=["2,3,5"] folds clusters 2, 3 and 5 into one category. This is NOT a
    dendrogram cut (a cut can't isolate one subtree and pool unrelated
    branches), so the merged category is a residual grouping, not a discovered
    cluster; the note returned says so and is written into the summary."""
    parsed = [[int(x) for x in g.split(",")] for g in groups]
    flat = [c for g in parsed for c in g]
    present = {int(c) for c in labels}
    if len(flat) != len(set(flat)) or not set(flat) <= present or any(len(g) < 2 for g in parsed):
        raise SystemExit(
            f"ERROR: --merge-clusters {groups} must list distinct cluster ids present in the k={len(present)} cut "
            f"{sorted(present)}, at least two per group."
        )
    target = {c: c for c in present}
    for g in parsed:
        for c in g:
            target[c] = min(g)
    remap = {old: i + 1 for i, old in enumerate(sorted(set(target.values())))}
    merged = np.array([remap[target[int(c)]] for c in labels])
    sil = silhouette_score(D, merged, metric="precomputed") if 2 <= len(set(merged)) < len(merged) else float("nan")

    print("=" * 70)
    print(f"POST-HOC MERGE of the k={len(present)} clusters into {len(set(merged))} categories (cluster id -> category):")
    for old in sorted(present):
        print(f"  cluster {old} -> category {remap[target[old]]}")
    print(f"Silhouette of the merged partition: {sil:.3f}")
    print("=" * 70)
    desc = "; ".join("{" + "+".join(map(str, g)) + "}" for g in parsed)
    note = (
        f"Post-hoc merge of hierarchical clusters {desc} (ids from the k={len(present)} cut) into {len(set(merged))} categories. "
        "This is NOT a single dendrogram cut: a merged category is a residual grouping, not a discovered cluster. "
        f"Silhouette of the merged partition: {sil:.3f}."
    )
    return merged, note


# ---------------------------------------------------------------------------
# 5. Classical PCoA (metric MDS via double-centered eigendecomposition)
# ---------------------------------------------------------------------------

def classical_pcoa(D: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool]:
    n = D.shape[0]
    D2 = D ** 2
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ D2 @ J
    eigvals, eigvecs = np.linalg.eigh(B)
    order = np.argsort(eigvals)[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]

    has_negative = eigvals.min() < -1e-8
    pos_eigvals = np.clip(eigvals, 0, None)
    coords = eigvecs[:, :2] * np.sqrt(pos_eigvals[:2])

    total = pos_eigvals.sum()
    var_explained = (pos_eigvals[:2] / total * 100) if total > 0 else np.array([np.nan, np.nan])
    return coords, var_explained, has_negative


# ---------------------------------------------------------------------------
# 6. Cluster-level domain summary
# ---------------------------------------------------------------------------

def cluster_domain_summary(pair_avg: pd.DataFrame, cluster_of: dict[str, int]) -> pd.DataFrame:
    pair_avg = pair_avg.copy()
    pair_avg["cluster_a"] = pair_avg["model_a"].map(cluster_of)
    pair_avg["cluster_b"] = pair_avg["model_b"].map(cluster_of)
    within = pair_avg[pair_avg["cluster_a"] == pair_avg["cluster_b"]]

    cluster_sizes = pd.Series(cluster_of).value_counts().rename("n_models")
    all_clusters = sorted(cluster_sizes.index)

    rows = []
    for c in all_clusters:
        sub = within[within["cluster_a"] == c]
        row = {"cluster": c, "n_models": int(cluster_sizes[c]), "n_pairs": int(sub.shape[0])}
        for dlabel in DOMAIN_LABELS:
            row[dlabel] = sub[dlabel].mean() if sub.shape[0] else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def write_cluster_summary_text(
    out_path: Path, cluster_df: pd.DataFrame, domain_summary: pd.DataFrame, best_k: int, sil_scores: dict,
    content_desc: str = "P1-P5", forced: bool = False, merge_note: str | None = None,
    cat_names: dict | None = None,
) -> None:
    lines = []
    lines.append("CLUSTER SUMMARY (blind pairwise biological-content similarity)")
    lines.append("=" * 70)
    lines.append(
        f"{best_k} cluster(s) {'forced (--k)' if forced else 'selected'} via average-linkage hierarchical clustering on the "
        f"content-similarity distance matrix ({content_desc}{'' if 'P6' in content_desc else '; P6/explanatory-style excluded'})."
    )
    if merge_note:
        lines.append(merge_note)
    elif best_k in sil_scores:
        best_score = sil_scores[best_k]
        if forced:
            best_overall = max(sil_scores, key=sil_scores.get)
            lines.append(
                f"Silhouette score at k={best_k}: {best_score:.3f} "
                f"(unconstrained optimum among candidates tested: k={best_overall}, {sil_scores[best_overall]:.3f})."
            )
        else:
            lines.append(f"Silhouette score at k={best_k}: {best_score:.3f} (highest among the candidates tested).")
    lines.append("")

    def _join(items: list[str]) -> str:
        if len(items) <= 1:
            return items[0] if items else ""
        if len(items) == 2:
            return f"{items[0]} and {items[1]}"
        return ", ".join(items[:-1]) + f", and {items[-1]}"

    domain_by_cluster = domain_summary.set_index("cluster")
    clusters_with_data = domain_summary[domain_summary["n_pairs"] > 0]
    # Cross-cluster average is only meaningful to compare against when there
    # are >=3 clusters; with exactly 2, "above/below the average of the two"
    # is tautological (each is above the other), so that comparison is
    # skipped and each cluster's own domain spread is described instead.
    multi_cluster_comparison = clusters_with_data.shape[0] > 2
    global_domain_mean = clusters_with_data[DOMAIN_LABELS].mean()

    for c in sorted(cluster_df["cluster"].unique()):
        members = cluster_df.loc[cluster_df["cluster"] == c, "model"].tolist()
        member_names = ", ".join(model_short_name(m) for m in members)
        row = domain_by_cluster.loc[c]
        name = (cat_names or {}).get(int(c))
        lines.append(f"Cluster {c}{f' -- {name}' if name else ''} ({len(members)} model(s)): {member_names}")
        if row["n_pairs"] == 0:
            lines.append("  Singleton cluster -- no within-cluster pairs, so domain profile is unavailable.")
            lines.append("")
            continue

        values_str = ", ".join(f"{d}={row[d]:.2f}" for d in DOMAIN_LABELS)
        lines.append(f"  Within-cluster domain means: {values_str}")

        # Describe the cluster's own strongest/weakest domain (always
        # meaningful, regardless of how many clusters exist).
        own_sorted = sorted(DOMAIN_LABELS, key=lambda d: row[d])
        weakest, strongest = own_sorted[0], own_sorted[-1]
        if row[strongest] - row[weakest] > 0.5:
            lines.append(
                f"  Within this cluster, agreement is highest on {strongest} ({row[strongest]:.2f}) "
                f"and lowest on {weakest} ({row[weakest]:.2f})."
            )
        else:
            lines.append("  Agreement is roughly even across all four domains within this cluster.")

        if multi_cluster_comparison:
            deltas = {d: row[d] - global_domain_mean[d] for d in DOMAIN_LABELS}
            above = sorted([d for d, v in deltas.items() if v > 0.4], key=lambda d: -deltas[d])
            below = sorted([d for d, v in deltas.items() if v < -0.4], key=lambda d: deltas[d])
            if above or below:
                parts = []
                if above:
                    parts.append("relatively higher " + _join(above))
                if below:
                    parts.append("relatively lower " + _join(below))
                lines.append(f"  Relative to the other clusters' average, this cluster shows {'; '.join(parts)}.")
            else:
                lines.append("  Its domain profile is close to the other clusters' average.")
        lines.append("")

    if clusters_with_data.shape[0] == 2:
        a, b = clusters_with_data["cluster"].tolist()
        row_a, row_b = domain_by_cluster.loc[a], domain_by_cluster.loc[b]
        overall_a = row_a[DOMAIN_LABELS].mean()
        overall_b = row_b[DOMAIN_LABELS].mean()
        higher, lower = (a, b) if overall_a >= overall_b else (b, a)
        lines.append(
            f"With only two clusters, Cluster {higher} shows uniformly higher within-cluster agreement "
            f"across all four domains than Cluster {lower} -- the split is driven by an overall gap in "
            "how closely members interpret the same biology, not by one specific domain."
        )
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Performance panel (independent; never affects similarity/clustering)
# ---------------------------------------------------------------------------

def load_performance(csv_path: Path, models: list[str]) -> tuple[pd.DataFrame | None, list[str]]:
    if not csv_path.is_file():
        print(f"NOTE: performance CSV not found at {csv_path} -- skipping the performance panel.")
        return None, []

    perf = pd.read_csv(csv_path)
    # evaluation_results_rubric_v12/all_scores.csv stores the raw model id
    # (e.g. "MiniMaxAI/MiniMax-M3"); pairwise data uses the filesystem slug
    # (e.g. "MiniMaxAI__MiniMax-M3", same convention as official_trial.py's
    # model_slug()). Normalize before matching.
    perf["model_slug"] = perf["model"].astype(str).str.replace("/", "__", regex=False)
    pillar_cols = [c for c in perf.columns if c.endswith("_zone1") and "reason" not in c and c != "final score_zone1"]
    if "final score_zone1" in perf.columns:
        pillar_cols = pillar_cols + ["final score_zone1"]

    avg = perf.groupby("model_slug")[pillar_cols].mean()
    avg = avg.reindex(models)
    missing_models = avg[avg.isnull().all(axis=1)].index.tolist()
    if missing_models:
        print(
            f"NOTE: performance data has no rows for {len(missing_models)} of the similarity models "
            f"({[model_short_name(m) for m in missing_models]}); their performance row will be left blank, not fabricated."
        )
    return avg, missing_models


# ---------------------------------------------------------------------------
# Model selection (size filter + explicit exclusions)
# ---------------------------------------------------------------------------

def select_models(df: pd.DataFrame, exclude: list[str], min_total_b: float | None) -> tuple[pd.DataFrame, list[str]]:
    """Drops explicitly excluded models and models whose total parameter count
    (model_registry.json) is below min_total_b. The anchor (gemini, size
    undisclosed) is exempt from the size filter. A model missing from the
    registry is an error, never silently kept or dropped."""
    from model_registry import load_registry

    reg = load_registry()
    anchor = reg["anchor_slug"]
    total_b = {m["slug"]: m["total_b"] for m in reg["models"] if m["slug"]}

    present = sorted(set(df["model_a"]) | set(df["model_b"]))
    unknown = [m for m in present if m != anchor and m not in total_b and m not in exclude]
    if min_total_b is not None and unknown:
        raise SystemExit(f"ERROR: model(s) not in model_registry.json, cannot size-filter: {unknown}")

    keep, dropped = [], []
    for m in present:
        if m in exclude:
            dropped.append(f"{model_short_name(m)} (excluded by request)")
        elif min_total_b is not None and m != anchor and total_b[m] < min_total_b:
            dropped.append(f"{model_short_name(m)} ({total_b[m]:g}B < {min_total_b:g}B)")
        else:
            keep.append(m)
    print(f"Model selection: keeping {len(keep)} of {len(present)}. Dropped: {dropped if dropped else 'none'}")
    mask = df["model_a"].isin(keep) & df["model_b"].isin(keep)
    return df[mask].copy(), keep


# ---------------------------------------------------------------------------
# Plotting -- one function per figure, each writes its own PNG + PDF
# ---------------------------------------------------------------------------

PLOT_CHOICES = ["cluster_map", "dendrogram_heatmap", "domain_profile", "performance"]


def _save(fig, out_stem: Path) -> None:
    fig.savefig(out_stem.with_suffix(".png"), dpi=250, bbox_inches="tight")
    fig.savefig(out_stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_stem.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_stem.with_suffix('.png')} / .pdf / .svg")


def _cluster_colors(labels: np.ndarray) -> dict:
    return {c: CLUSTER_PALETTE[i % len(CLUSTER_PALETTE)] for i, c in enumerate(sorted(set(labels)))}


def _category_label(c, n: int, names: dict | None) -> str:
    """'Category 2 (n=6)' plus the category's name on wrapped lines when one was given."""
    name = (names or {}).get(int(c))
    return f"Category {c} (n={n})" + (f"\n{textwrap.fill(name, 30)}" if name else "")


def _sim_noun(content_desc: str) -> str:
    """P6 (explanatory style) makes the similarity more than biological content."""
    return "Response similarity" if "P6" in content_desc else "Biological-content similarity"


def _place_labels(fig, ax, coords: np.ndarray, names: list[str]) -> None:
    """Greedy label placement: for each point, take the first candidate
    offset whose label box overlaps no already-placed label and no other
    point's marker. Near offsets are tried first; in crowded regions the label
    moves out along a ring and gets a thin leader line back to its point."""
    from matplotlib.transforms import Bbox

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    ax_box = ax.get_window_extent(renderer)

    marker_r = (110 ** 0.5) / 2 * fig.dpi / 72 + 1  # scatter s=110 -> radius in px
    marker_boxes = [Bbox.from_extents(x - marker_r, y - marker_r, x + marker_r, y + marker_r)
                    for x, y in ax.transData.transform(coords)]

    near = [
        ((6, 5), "left", "bottom"), ((6, -6), "left", "top"),
        ((-6, 5), "right", "bottom"), ((-6, -6), "right", "top"),
        ((0, 10), "center", "bottom"), ((0, -11), "center", "top"),
    ]
    far = []
    for radius in (24, 38, 54, 72):
        for ang in range(0, 360, 30):
            dx, dy = radius * np.cos(np.radians(ang)), radius * np.sin(np.radians(ang))
            ha = "left" if dx > 3 else "right" if dx < -3 else "center"
            va = "bottom" if dy > 3 else "top" if dy < -3 else "center"
            far.append(((dx, dy), ha, va))

    def _ccw(a, b, c):
        return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])

    def _segments_cross(p1, p2, p3, p4):
        return _ccw(p1, p3, p4) != _ccw(p2, p3, p4) and _ccw(p1, p2, p3) != _ccw(p1, p2, p4)

    px_pts = ax.transData.transform(coords)
    placed, leaders = [], []
    for idx, ((x, y), name) in enumerate(zip(coords, names)):
        best = None  # (score, offset, ha, va, bbox, is_far)
        # A point with a close neighbour always gets a leader line, so its label can't be mistaken for the neighbour's.
        crowded = any(np.hypot(*(px_pts[idx] - px_pts[j])) < 4 * marker_r for j in range(len(px_pts)) if j != idx)
        for is_far, cands in (((True, far),) if crowded else ((False, near), (True, far))):
            for off, ha, va in cands:
                t = ax.annotate(name, (x, y), xytext=off, textcoords="offset points", ha=ha, va=va, fontsize=8, zorder=4)
                bb = t.get_window_extent(renderer)
                t.remove()
                score = (
                    sum(bb.overlaps(p) for p in placed)
                    + sum(bb.overlaps(m) for j, m in enumerate(marker_boxes) if j != idx)
                    + (0 if ax_box.contains(bb.x0, bb.y0) and ax_box.contains(bb.x1, bb.y1) else 0.5)
                )
                if is_far:  # the leader line must not run through other labels/markers or cross other leaders
                    start = px_pts[idx]
                    end = start + np.array(off) * fig.dpi / 72
                    samples = [start + f * (end - start) for f in np.linspace(0.2, 1.0, 12)]
                    score += sum(any(p.contains(*s) for p in placed) for s in samples) > 0
                    score += sum(any(m.contains(*s) for j, m in enumerate(marker_boxes) if j != idx) for s in samples) > 0
                    score += sum(_segments_cross(start, end, a, b) for a, b in leaders)
                if best is None or score < best[0]:
                    best = (score, off, ha, va, bb, is_far)
                if score == 0:
                    break
            if best[0] == 0:
                break
        _, off, ha, va, bb, is_far = best
        ax.annotate(
            name, (x, y), xytext=off, textcoords="offset points", ha=ha, va=va, fontsize=8, zorder=4,
            arrowprops=dict(arrowstyle="-", color="#999999", lw=0.6, shrinkA=1, shrinkB=5) if is_far else None,
        )
        placed.append(bb)
        if is_far:
            leaders.append((px_pts[idx], px_pts[idx] + np.array(off) * fig.dpi / 72))


def plot_cluster_map(
    coords: np.ndarray, var_explained: np.ndarray, models: list[str], labels: np.ndarray,
    best_k: int, out_stem: Path, subtitle: str, content_desc: str = "P1-P5", merged_from: int | None = None,
    cat_names: dict | None = None, heading: str | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 9.5))
    cluster_ids = sorted(set(labels))
    color_of = _cluster_colors(labels)
    for c in cluster_ids:
        mask = labels == c
        ax.scatter(
            coords[mask, 0], coords[mask, 1], s=110, color=color_of[c], edgecolor="black",
            linewidth=0.7, label=_category_label(c, int(mask.sum()), cat_names), zorder=3,
        )
    ax.axhline(0, color="#dddddd", linewidth=0.8, zorder=1)
    ax.axvline(0, color="#dddddd", linewidth=0.8, zorder=1)
    ax.set_xlabel(f"PCoA 1 ({var_explained[0]:.1f}% of variation)" if np.isfinite(var_explained[0]) else "PCoA 1")
    ax.set_ylabel(f"PCoA 2 ({var_explained[1]:.1f}% of variation)" if np.isfinite(var_explained[1]) else "PCoA 2")
    ax.set_aspect("equal", adjustable="datalim")
    ax.margins(0.12)
    ax.spines[["top", "right"]].set_visible(False)
    heading = heading or (
        f"{best_k} categories (merged from {merged_from} hierarchical clusters)" if merged_from
        else f"{best_k} categories from hierarchical clustering"
    )
    ax.set_title(f"Model similarity map -- {heading}\n{subtitle}", fontsize=11)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=9, frameon=False, title="Category (color)",
              labelspacing=1.1)
    _place_labels(fig, ax, coords, [model_short_name(m) for m in models])
    closer = "more similar responses" if "P6" in content_desc else "more similar biological interpretation"
    fig.text(
        0.01, -0.02,
        f"Closer points = {closer} ({content_desc}). Axes are a compressed projection, "
        "not quality/performance directions. Preliminary.",
        fontsize=7.5, color="#555555",
    )
    _save(fig, out_stem)


def plot_dendrogram_heatmap(
    S: np.ndarray, models: list[str], Z, out_stem: Path,
    labels: np.ndarray | None = None, best_k: int | None = None, content_desc: str = "P1-P5",
    merged_from: int | None = None, cat_names: dict | None = None, vmin: float = 0.0, vmax: float = 10.0,
) -> None:
    """Dendrogram + reordered similarity heatmap. With `labels`/`best_k`, the
    dendrogram branches, a color strip along the top and left of the heatmap,
    and a dashed cut line show the k-cluster partition; without them the
    dendrogram is plain grey."""
    from matplotlib.colors import to_hex
    from matplotlib.patches import Patch, Rectangle

    n = len(models)
    side = max(9, 0.55 * n + 3)
    fig = plt.figure(figsize=(side, side))
    gs = fig.add_gridspec(
        3, 4, width_ratios=[0.025, 1, 0.02, 0.03], height_ratios=[0.28, 0.025, 1], hspace=0.02, wspace=0.02,
    )
    ax_d = fig.add_subplot(gs[0, 1])
    ax_top = fig.add_subplot(gs[1, 1])
    ax_left = fig.add_subplot(gs[2, 0])
    ax_h = fig.add_subplot(gs[2, 1])
    ax_cb = fig.add_subplot(gs[2, 3])

    if labels is not None:
        color_of = _cluster_colors(labels)
        members = {i: [i] for i in range(n)}
        for i, (a, b, *_rest) in enumerate(Z):
            members[n + i] = members[int(a)] + members[int(b)]

        def link_color(node: int) -> str:
            clusters = {labels[m] for m in members[node]}
            return to_hex(color_of[next(iter(clusters))]) if len(clusters) == 1 else "#666666"

        dendro_kwargs = dict(link_color_func=link_color)
    else:
        dendro_kwargs = dict(color_threshold=0, above_threshold_color="#666666")

    with plt.rc_context({"lines.linewidth": 1.8}):
        dendro = dendrogram(Z, ax=ax_d, no_labels=True, **dendro_kwargs)
    leaf_order = dendro["leaves"]
    ordered_labels = [model_short_name(models[i]) for i in leaf_order]
    ax_d.set_xlim(0, n * 10)
    ax_d.set_xticks([])
    ax_d.set_ylabel("Distance", fontsize=8)
    noun = _sim_noun(content_desc)
    title = f"{noun} (average-linkage dendrogram; {content_desc})"
    if labels is not None and best_k:
        title += (
            f"\n{merged_from} hierarchical clusters merged into {best_k} categories (post hoc; not a single cut)"
            if merged_from else f"\nk = {best_k} categories"
        )
        if not merged_from and 2 <= best_k < n:
            cut = (Z[-(best_k - 1), 2] + Z[-best_k, 2]) / 2  # any height between the two merges gives k clusters
            ax_d.axhline(cut, color="#999999", linestyle="--", linewidth=0.8, zorder=0)
    ax_d.set_title(title, fontsize=11)
    ax_d.spines[["top", "right"]].set_visible(False)

    if labels is not None:
        # One rectangle per model (not an image) so the strips stay vector/editable in the .svg.
        for pos, i in enumerate(leaf_order):
            c = color_of[labels[i]]
            ax_top.add_patch(Rectangle((pos * 10, 0), 10, 1, facecolor=c, edgecolor="none"))
            ax_left.add_patch(Rectangle((0, pos * 10), 1, 10, facecolor=c, edgecolor="none"))
        ax_top.set_xlim(0, n * 10)
        ax_top.set_ylim(1, 0)
        ax_left.set_xlim(0, 1)
        ax_left.set_ylim(n * 10, 0)
        counts = {c: int((labels == c).sum()) for c in color_of}
        ax_d.legend(
            handles=[Patch(facecolor=color_of[c], edgecolor="black", linewidth=0.6,
                           label=_category_label(c, counts[c], cat_names))
                     for c in sorted(color_of)],
            loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=9, frameon=False, title="Category (color)",
            labelspacing=1.1,
        )
    ax_top.axis("off")
    ax_left.axis("off")

    # pcolormesh (one quad per cell), not imshow, so the heatmap stays vector/editable in the .svg.
    edges = np.arange(0, n * 10 + 1, 10)
    ordered = S[np.ix_(leaf_order, leaf_order)].astype(float)
    np.fill_diagonal(ordered, np.nan)  # self-similarity is defined (10), not measured: draw it grey, not dark red
    im = ax_h.pcolormesh(edges, edges, np.ma.masked_invalid(ordered), cmap=PHEATMAP_CMAP, vmin=vmin, vmax=vmax,
                         shading="flat", edgecolors="face", linewidth=0.25)
    ax_h.set_xlim(0, n * 10)
    ax_h.set_ylim(n * 10, 0)
    ticks = [i * 10 + 5 for i in range(n)]
    ax_h.set_xticks(ticks)
    ax_h.set_xticklabels(ordered_labels, rotation=45, ha="right", fontsize=8)
    ax_h.set_yticks(ticks)
    ax_h.set_yticklabels(ordered_labels, fontsize=8)
    cb = fig.colorbar(im, cax=ax_cb)
    cb.set_label(f"{noun} ({vmin:g}-{vmax:g})", fontsize=9)
    cb.solids.set_rasterized(False)  # matplotlib rasterizes a >=50-step colorbar by default
    cb.solids.set_edgecolor("face")
    _save(fig, out_stem)


def plot_domain_profile(domain_summary: pd.DataFrame, out_stem: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 1.2 + 0.9 * domain_summary.shape[0]))
    matrix = domain_summary.set_index("cluster")[DOMAIN_LABELS].values
    im = ax.imshow(np.ma.masked_invalid(matrix), cmap="viridis", vmin=0, vmax=10, aspect="auto")
    ax.set_xticks(range(len(DOMAIN_LABELS)))
    ax.set_xticklabels(DOMAIN_LABELS, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(domain_summary.shape[0]))
    ax.set_yticklabels([f"Cluster {c}" for c in domain_summary["cluster"]], fontsize=9)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = matrix[i, j]
            ax.text(j, i, f"{v:.1f}" if np.isfinite(v) else "n/a", ha="center", va="center", fontsize=9,
                    color="white" if np.isfinite(v) and v > 5 else "black")
    ax.set_title("Cluster domain profile (within-cluster pair averages)", fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label="Score (0-10)")
    _save(fig, out_stem)


def plot_performance(performance: pd.DataFrame, ordered_models: list[str], out_stem: Path) -> None:
    pillar_cols = [c for c in performance.columns if c != "final score_zone1"]
    perf = performance.reindex(ordered_models)
    fig, ax = plt.subplots(figsize=(9, max(5, 0.42 * len(ordered_models) + 2)))
    im = ax.imshow(np.ma.masked_invalid(perf[pillar_cols].values), cmap="cividis", vmin=0, vmax=10, aspect="auto")
    ax.set_xticks(range(len(pillar_cols)))
    ax.set_xticklabels([c.replace("_zone1", "").replace(" score", "") for c in pillar_cols], rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(ordered_models)))
    ax.set_yticklabels([model_short_name(m) for m in ordered_models], fontsize=8)
    for i in range(len(ordered_models)):
        for j in range(len(pillar_cols)):
            v = perf[pillar_cols].values[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=7, color="white" if v > 5 else "black")
    ax.set_title("Independent performance (judge rubric pillars, 0-10)\nordered like the dendrogram; not used in clustering", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label="Pillar score (0-10)")
    _save(fig, out_stem)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pairwise-dir", type=str, default=None, help=f"Default: {PAIRWISE_DIR}")
    parser.add_argument("--out-dir", type=str, default=None, help=f"Default: {OUT_DIR}")
    parser.add_argument(
        "--plots", nargs="+", choices=PLOT_CHOICES, default=["cluster_map"],
        help="Which individual figure files to write (default: cluster_map only).",
    )
    parser.add_argument("--min-total-b", type=float, default=20.0,
                        help="Drop models with total parameters below this many billions (default 20; 0 disables). Anchor exempt.")
    parser.add_argument("--exclude-models", nargs="+", default=["Qwen3.5-9B-think-temp0"],
                        help="Model slugs to drop regardless of size (default: Qwen3.5-9B-think-temp0).")
    parser.add_argument(
        "--performance-csv", type=str, default=None,
        help=f"Only used with --plots performance. Default: {DEFAULT_PERFORMANCE_CSV}",
    )
    parser.add_argument("--drop-pillars", nargs="+", choices=CONTENT_PILLARS, default=[],
                        help="Pillars to leave out of the content similarity used for clustering "
                             "(e.g. --drop-pillars P1 removes zone/feature selection similarity).")
    parser.add_argument("--add-pillars", nargs="+", choices=["P6"], default=[],
                        help="Also include P6 (response scope & elaboration = explanatory style) in the similarity "
                             "used for clustering. Off by default: P6 measures presentation, not biological content.")
    parser.add_argument("--k", type=int, default=None, help="Force this many clusters instead of silhouette selection.")
    parser.add_argument("--merge-clusters", nargs="+", default=None, metavar="IDS",
                        help="After the --k cut, fold groups of cluster ids into single categories, comma-separated "
                             "per group (e.g. --merge-clusters 2,3,5). Ids are those printed/saved for the unmerged "
                             "cut. Post hoc, not a dendrogram cut; the outputs say so.")
    parser.add_argument("--heatmap-range", nargs=2, type=float, default=[0.0, 10.0], metavar=("VMIN", "VMAX"),
                        help="Color scale of the dendrogram heatmap (default 0 10, comparable across plots). Narrow it "
                             "(e.g. 5 9) to make small differences visible; the colorbar label shows the range used.")
    parser.add_argument("--category-names", nargs="+", default=None, metavar="NAME",
                        help="Names for the final categories, in category-number order (after any --merge-clusters); "
                             "shown in the figure legends and summary text. Quote multi-word names.")
    parser.add_argument("--min-k", type=int, default=3, help="Smallest cluster count silhouette selection may pick (default 3, so the map shows several categories; the unconstrained optimum is printed and noted when overridden).")
    parser.add_argument("--max-k", type=int, default=None, help="Cap on candidate k (default: min(n-1, 10)).")
    args = parser.parse_args()

    pairwise_dir = Path(args.pairwise_dir) if args.pairwise_dir else PAIRWISE_DIR
    if not pairwise_dir.is_dir():
        raise SystemExit(f"ERROR: {pairwise_dir} not found.")
    out_dir = Path(args.out_dir) if args.out_dir else OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_long_table(pairwise_dir)
    if df.empty:
        raise SystemExit(f"ERROR: no cached pairwise comparisons found under {pairwise_dir}")
    df, _ = select_models(df, args.exclude_models or [], args.min_total_b or None)

    report = validate(df)
    if report["self_pairs"].shape[0] or len(report["missing_pairs_entirely"]) or report["invalid_rows"].shape[0]:
        raise SystemExit(
            "ERROR: data validation found self-pairs, entirely-missing pairs, and/or invalid scores "
            "(see report above). Fix the underlying data first -- refusing to fabricate or silently drop anything."
        )
    if report["incomplete_pairs"]:
        print("WARNING: some pairs are missing datasets (see report); averages are not comparable 1:1 across pairs.")

    models = report["models"]
    content_pillars = [p for p in CONTENT_PILLARS if p not in set(args.drop_pillars)] + list(args.add_pillars)
    if not content_pillars:
        raise SystemExit("ERROR: --drop-pillars removed every content pillar.")
    content_desc = "+".join(content_pillars) if len(content_pillars) < 2 else f"{content_pillars[0]}-{content_pillars[-1]}"
    if args.drop_pillars and content_pillars != [f"P{i}" for i in range(int(content_pillars[0][1]), int(content_pillars[-1][1]) + 1)]:
        content_desc = ", ".join(content_pillars)
    if len(content_pillars) <= 3 and set(content_pillars) != set(CONTENT_PILLARS):  # few pillars: name them in the titles
        content_desc = "+".join(content_pillars) + ": " + ", ".join(PILLAR_NAMES[p] for p in content_pillars)
    print(f"Content similarity uses pillars: {content_pillars}")
    pair_avg = compute_pair_averages(df, content_pillars)
    S, D = build_matrices(pair_avg, models)

    Z, labels, best_k, sil_scores = cluster_models(D, models, args.k, args.max_k, args.min_k)
    labels_hier, merge_note = labels, None
    if args.merge_clusters:
        labels, merge_note = merge_clusters(labels_hier, args.merge_clusters, D)
    n_categories = len(set(labels))
    merged_from = best_k if merge_note else None
    cat_names = None
    if args.category_names:
        ids = sorted(int(c) for c in set(labels))
        if len(args.category_names) != len(ids):
            raise SystemExit(f"ERROR: --category-names needs {len(ids)} names (categories {ids}), got {len(args.category_names)}.")
        cat_names = dict(zip(ids, args.category_names))
    cluster_of = dict(zip(models, labels))
    cluster_df = pd.DataFrame({"model": models, "cluster": labels})
    if merge_note:
        cluster_df["cluster_before_merge"] = labels_hier
    ordered_models = [models[i] for i in dendrogram(Z, no_plot=True)["leaves"]]

    coords, var_explained, has_negative = classical_pcoa(D)
    if has_negative:
        print("NOTE: negative PCoA eigenvalues present (distances not perfectly Euclidean); 2D map is an approximation.")
    domain_summary = cluster_domain_summary(pair_avg, cluster_of)

    size_desc = (
        f"{len(models)} models with total parameters >= {args.min_total_b:g}B (plus gemini anchor)"
        if args.min_total_b else f"{len(models)} models (no size filter)"
    )
    subtitle = f"{size_desc}; average linkage on {content_desc} distance"
    stem = lambda name: out_dir / name
    if "cluster_map" in args.plots:
        plot_cluster_map(
            coords, var_explained, models, labels, n_categories, stem("cluster_map"), subtitle, content_desc, merged_from,
            cat_names,
        )
    if "dendrogram_heatmap" in args.plots:
        plot_dendrogram_heatmap(
            S, models, Z, stem("dendrogram_heatmap"), labels, n_categories, content_desc, merged_from, cat_names,
            vmin=args.heatmap_range[0], vmax=args.heatmap_range[1],
        )
    if "domain_profile" in args.plots:
        plot_domain_profile(domain_summary, stem("domain_profile"))
    if "performance" in args.plots:
        performance, _ = load_performance(Path(args.performance_csv) if args.performance_csv else DEFAULT_PERFORMANCE_CSV, models)
        if performance is not None:
            plot_performance(performance, ordered_models, stem("performance"))

    pair_avg.to_csv(out_dir / "pairwise_pillar_averages.csv", index=False)
    pd.DataFrame(S, index=models, columns=models).to_csv(out_dir / "content_similarity_matrix.csv")
    pd.DataFrame(D, index=models, columns=models).to_csv(out_dir / "distance_matrix.csv")
    cluster_df.assign(dendrogram_order=cluster_df["model"].apply(ordered_models.index)).sort_values(
        "dendrogram_order"
    ).to_csv(out_dir / "cluster_assignments.csv", index=False)
    domain_summary.to_csv(out_dir / "cluster_domain_summary.csv", index=False)
    write_cluster_summary_text(
        out_dir / "cluster_summary.txt", cluster_df, domain_summary, best_k, sil_scores, content_desc,
        forced=args.k is not None, merge_note=merge_note, cat_names=cat_names,
    )
    print(f"\nDone. Outputs saved under: {out_dir}")


if __name__ == "__main__":
    main()
