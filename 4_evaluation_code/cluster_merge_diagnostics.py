"""
Diagnostics for the post-hoc merge of clusters 2, 3 and 5 of the five-cluster cut (total-similarity analysis, Figure 7A).

The merge itself is `merge_clusters()` in analyze_pairwise_clustering.py (option --merge-clusters 2,3,5). This script does not choose
the clusters; it reports numbers that show why the merged groups are described the way they are in the paper:

  1. Silhouette of every cluster of the five-cluster cut and of the three merged categories.
  2. Mean model-pair similarity within and between the five clusters and within and between the three categories.
  3. Bootstrap over PROMPTS (the resampling unit is the prompt): the six prompts are resampled with replacement, the pair similarities are
     re-averaged, the distance matrix 1 - S/10 is rebuilt, average-linkage clustering is cut at five clusters again, and for every pair of
     models it is recorded whether they fall in the same cluster. Reported: the mean co-clustering frequency of the members of each
     five-cluster cluster and of each merged category (1.0 = always together). Default: 2,000 resamples, seed 0.
  4. The rubric scores of the merged category against the large category (from category_pillar_tests.csv; these scores were not used to
     form the groups).

Nothing here selects the number of clusters or the merge. The merged category is a post-hoc, residual grouping, not a discovered cluster.

Inputs (defaults, as in the repository layout):
    pairwise_similarity_results_v2/<prompt>/*__vs__*.json                                  judged pair records
    pairwise_similarity_results_v2/FINAL_3categories_similarity_and_performance/           cluster_assignments.csv, distance_matrix.csv,
                                                                                           category_pillar_tests.csv, category_pillar_summary.csv
Outputs (--out-dir, default <FINAL folder>/cluster_merge_diagnostics/):
    summary.txt, silhouette_by_group.csv, similarity_between_clusters_k5.csv, similarity_between_categories.csv,
    bootstrap_coclustering_by_group.csv, bootstrap_coclustering_matrix.csv, rubric_merged_vs_large_category.csv

Usage:
    python cluster_merge_diagnostics.py                  # 2000 resamples, seed 0
    python cluster_merge_diagnostics.py --n-boot 5000 --seed 1
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.metrics import silhouette_samples, silhouette_score

import analyze_pairwise_clustering as A

warnings.filterwarnings("ignore", message="Mean of empty slice")   # the all-NaN diagonal of the pair matrices

SIM_PILLARS = ["P1", "P2", "P3", "P4", "P5", "P6"]      # total similarity: all six criteria
FINAL_DIR = A.PAIRWISE_DIR / "FINAL_3categories_similarity_and_performance"


def block_mean(S: np.ndarray, lab: np.ndarray, gi, gj) -> float:
    ii, jj = np.where(lab == gi)[0], np.where(lab == gj)[0]
    vals = [S[i, j] for i in ii for j in jj if i != j]
    return float(np.mean(vals)) if vals else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairwise-dir", default=None)
    ap.add_argument("--final-dir", default=None, help=f"Default: {FINAL_DIR}")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--large-category", type=int, default=3, help="Category the merged one is contrasted with (default 3, the 13-model group).")
    ap.add_argument("--merged-category", type=int, default=2, help="The category formed by the merge (default 2).")
    args = ap.parse_args()

    final = Path(args.final_dir) if args.final_dir else FINAL_DIR
    out = Path(args.out_dir) if args.out_dir else final / "cluster_merge_diagnostics"
    out.mkdir(parents=True, exist_ok=True)
    pairwise_dir = Path(args.pairwise_dir) if args.pairwise_dir else A.PAIRWISE_DIR

    D = pd.read_csv(final / "distance_matrix.csv", index_col=0)
    asg = pd.read_csv(final / "cluster_assignments.csv").set_index("model").loc[D.index]
    models = list(D.index); n = len(models)
    Dm = D.values; S = 10 * (1 - Dm)
    k_lab, cat = asg["cluster_before_merge"].values, asg["cluster"].values
    short = [A.model_short_name(m) for m in models]
    lines = ["CLUSTER MERGE DIAGNOSTICS", "=" * 70,
             f"{n} models; similarity = mean of {', '.join(SIM_PILLARS)} over the prompts; distance = 1 - S/10; average linkage; k = {args.k}.", ""]

    # --- 1. silhouette --------------------------------------------------------------------------------------------------------
    s_k, s_c = silhouette_samples(Dm, k_lab, metric="precomputed"), silhouette_samples(Dm, cat, metric="precomputed")
    rows = []
    for c in sorted(set(k_lab)):
        rows.append(dict(partition=f"k={args.k} cluster", group=int(c), n_models=int((k_lab == c).sum()),
                         merged_into_category=int(cat[k_lab == c][0]), mean_silhouette=float(s_k[k_lab == c].mean()),
                         models="; ".join(s for s, x in zip(short, k_lab) if x == c)))
    for c in sorted(set(cat)):
        rows.append(dict(partition="merged category", group=int(c), n_models=int((cat == c).sum()), merged_into_category=int(c),
                         mean_silhouette=float(s_c[cat == c].mean()), models="; ".join(s for s, x in zip(short, cat) if x == c)))
    sil = pd.DataFrame(rows)
    sil.to_csv(out / "silhouette_by_group.csv", index=False)
    lines += [f"Silhouette of the k={args.k} partition: {silhouette_score(Dm, k_lab, metric='precomputed'):.3f}; "
              f"of the merged {len(set(cat))}-category partition: {silhouette_score(Dm, cat, metric='precomputed'):.3f}", sil.round(3).drop(columns="models").to_string(index=False), ""]

    # --- 2. similarity within / between groups ----------------------------------------------------------------------------------
    ck, cc = sorted(set(k_lab)), sorted(set(cat))
    Mk = pd.DataFrame([[block_mean(S, k_lab, i, j) for j in ck] for i in ck], index=ck, columns=ck)
    Mc = pd.DataFrame([[block_mean(S, cat, i, j) for j in cc] for i in cc], index=cc, columns=cc)
    Mk.round(3).to_csv(out / "similarity_between_clusters_k5.csv"); Mc.round(3).to_csv(out / "similarity_between_categories.csv")
    lines += ["Mean model-pair similarity (0-10); diagonal = within the group:", "  k=5 clusters:", Mk.round(2).to_string(),
              "  merged categories:", Mc.round(2).to_string(), ""]

    # --- 3. bootstrap over prompts --------------------------------------------------------------------------------------------------
    df = A.load_long_table(pairwise_dir)
    df = df[df["model_a"].isin(models) & df["model_b"].isin(models)].copy()
    df["sim"] = df[SIM_PILLARS].mean(axis=1)
    cases = sorted(df["dataset"].unique()); idx = {m: i for i, m in enumerate(models)}
    Sc = np.full((len(cases), n, n), np.nan)
    for r in df.itertuples():
        c = cases.index(r.dataset)
        Sc[c, idx[r.model_a], idx[r.model_b]] = Sc[c, idx[r.model_b], idx[r.model_a]] = r.sim
    mean_S = np.nanmean(Sc, axis=0)
    off = ~np.eye(n, dtype=bool)
    assert np.allclose(mean_S[off], S[off], atol=1e-9), "pair records do not reproduce the saved similarity matrix"
    rng = np.random.default_rng(args.seed)
    co = np.zeros((n, n))
    for _ in range(args.n_boot):
        pick = rng.integers(0, len(cases), len(cases))               # resample the PROMPTS with replacement
        Sb = np.nanmean(Sc[pick], axis=0); np.fill_diagonal(Sb, 10)
        Db = 1 - Sb / 10; np.fill_diagonal(Db, 0)
        lab = fcluster(linkage(squareform(Db, checks=False), method="average"), args.k, "maxclust")
        co += (lab[:, None] == lab[None, :])
    co /= args.n_boot
    pd.DataFrame(co, index=short, columns=short).round(3).to_csv(out / "bootstrap_coclustering_matrix.csv")

    def mean_co(lab, g):
        ii = np.where(lab == g)[0]
        return float(np.mean([co[i, j] for i in ii for j in ii if i < j])) if len(ii) > 1 else float("nan")

    rows = [dict(partition=f"k={args.k} cluster", group=int(c), n_models=int((k_lab == c).sum()), mean_pairwise_coclustering=mean_co(k_lab, c)) for c in ck]
    rows += [dict(partition="merged category", group=int(c), n_models=int((cat == c).sum()), mean_pairwise_coclustering=mean_co(cat, c)) for c in cc]
    bt = pd.DataFrame(rows); bt.to_csv(out / "bootstrap_coclustering_by_group.csv", index=False)
    mi, li = np.where(cat == args.merged_category)[0], np.where(cat == args.large_category)[0]
    cross = float(np.mean([co[i, j] for i in mi for j in li]))
    lines += [f"Bootstrap over prompts: {args.n_boot} resamples of the {len(cases)} prompts with replacement, seed {args.seed}; each resample re-averages the pair "
              f"similarities and re-cuts the average-linkage tree at k={args.k}.",
              "Mean co-clustering frequency of the members of each group (1 = always in the same cluster; single-model groups give NaN):",
              bt.round(3).to_string(index=False),
              f"Mean co-clustering between members of merged category {args.merged_category} and category {args.large_category}: {cross:.3f}", ""]

    # --- 4. rubric scores of the merged category against the large one (not used to form the groups) -----------------------------------------
    tests = pd.read_csv(final / "category_pillar_tests.csv"); summ = pd.read_csv(final / "category_pillar_summary.csv")
    con = tests[tests["contrast"] == f"cat {args.merged_category} vs cat {args.large_category}"][
        ["pillar", "mean_a", "mean_b", "mean_diff_a_minus_b", "cliffs_delta_a_vs_b", "p_value"]].rename(columns={"mean_a": "mean_merged", "mean_b": "mean_large", "p_value": "p_unadjusted_mann_whitney"})
    kw = tests[tests["contrast"].str.startswith("all categories")][["pillar", "p_value", "p_holm_core_pillars"]].rename(columns={"p_value": "p_kruskal_wallis"})
    con = con.merge(kw, on="pillar", how="left"); con.round(4).to_csv(out / "rubric_merged_vs_large_category.csv", index=False)
    lines += [f"Rubric scores (not used to form the groups): merged category {args.merged_category} vs category {args.large_category}. "
              "Pairwise p-values are unadjusted and descriptive; the Kruskal-Wallis column is across all three categories, Holm-corrected over the six core criteria.",
              con.round(3).to_string(index=False), "",
              "Reading: the merged category is a post-hoc residual grouping, not a discovered cluster. It is justified by the instability of its sub-clusters "
              "and by its rubric profile, not by a separate similarity cluster."]
    (out / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
