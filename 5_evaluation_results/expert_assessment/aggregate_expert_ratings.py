"""Aggregate the deidentified expert ratings. Figure 7E plots the factual accuracy (accuracy of specific molecular claims) and the pairwise similarity (mechanistic concordance) of the hypotheses; biological plausibility is aggregated the same way and written for completeness.

Inputs (this folder):  deidentified_ratings.csv, hypothesis_label_mapping.csv, figure_7E_plotting_input.csv, figure_7E_factual_accuracy_by_assessor.csv, figure_7E_pairwise_similarity_by_assessor.csv
Output:                aggregated_expert_ratings.csv  (all three metrics; column used_in_figure_7E marks the plotted ones)

Aggregation (the same arithmetic as the spreadsheet formulas of the evaluation workbook the figure was drawn from):
  1. a rating is one assessor's 1-5 score for one hypothesis and pillar, or for one pair of hypotheses;
  2. the blind labels A/B/C are translated to models with hypothesis_label_mapping.csv (a pair label such as B-C becomes a pair of models);
  3. dataset score = mean over the 5 assessors (pERK, GABA, Ki-67);
  4. overall mean = mean of the three dataset scores;
  Figure 7E = "Accuracy of specific molecular claims" (factual accuracy) and the "Mechanistic concordance" of model pairs (pairwise similarity); also aggregated: "Core biological plausibility".
Usage:  python aggregate_expert_ratings.py          (exit code 1 if a value differs from figure_7E_plotting_input.csv by more than 1e-9)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATASETS = ["pERK", "GABA", "Ki-67"]
MODELS = ["Gemini-3.1-Pro", "Qwen3.8-2.4T", "Qwen3.5-9B"]
METRICS = {"Biological plausibility": "Core biological plausibility",
           "Factual accuracy (accuracy of specific molecular claims)": "Accuracy of specific molecular claims",
           "Pairwise similarity (mechanistic concordance)": "Mechanistic concordance"}
FIG7E_METRICS = ["Factual accuracy (accuracy of specific molecular claims)", "Pairwise similarity (mechanistic concordance)"]


def main():
    r = pd.read_csv(HERE / "deidentified_ratings.csv")
    m = pd.read_csv(HERE / "hypothesis_label_mapping.csv")
    label_to_model = {(d, l): mod for d, l, mod in zip(m.dataset, m.blind_label, m.model)}
    h = r[r.item_type == "hypothesis"].copy()
    h["model"] = [label_to_model[(d, l)] for d, l in zip(h.dataset, h.blind_label_or_pair)]
    p = r[r.item_type == "pairwise"].copy()
    pair_names = []
    for d, bp in zip(p.dataset, p.blind_label_or_pair):
        x, y = bp.split("-")
        a, b = sorted([label_to_model[(d, x)], label_to_model[(d, y)]], key=MODELS.index)
        pair_names.append(f"{a} vs {b}")
    p["model"] = pair_names

    out = []
    for metric, criterion in METRICS.items():
        pair_metric = metric.startswith("Pairwise")
        src = p if pair_metric else h
        s = src[src.criterion == criterion]
        rows = ([f"{a} vs {b}" for i, a in enumerate(MODELS) for b in MODELS[i + 1:]] if pair_metric else MODELS)
        for row in rows:
            ds_means = [s[(s.model == row) & (s.dataset == d)].score.mean() for d in DATASETS]
            n = int(s[(s.model == row) & (s.dataset == DATASETS[0])].assessor_id.nunique())
            for col, v in zip(DATASETS + ["Overall mean"], ds_means + [float(np.mean(ds_means))]):
                out.append({"metric": metric, "model": row, "dataset": col, "mean_score": v, "n_assessors": n, "scale": "1-5", "used_in_figure_7E": metric in FIG7E_METRICS})
    agg = pd.DataFrame(out)
    agg.to_csv(HERE / "aggregated_expert_ratings.csv", index=False)

    ref = pd.read_csv(HERE / "figure_7E_plotting_input.csv")
    mrg = ref.merge(agg[agg.used_in_figure_7E], on=["metric", "model", "dataset"], suffixes=("_figure", "_recomputed"), how="outer", indicator=True)
    diff = (mrg.mean_score_figure - mrg.mean_score_recomputed).abs()
    ok = bool((mrg._merge == "both").all() and (diff < 1e-9).all() and (mrg.n_assessors_figure == mrg.n_assessors_recomputed).all())
    # the assessor-level files behind the plotted dots must equal the ratings
    lvl = pd.read_csv(HERE / "figure_7E_factual_accuracy_by_assessor.csv")
    hh = h[h.criterion == "Accuracy of specific molecular claims"][["assessor_id", "dataset", "model", "blind_label_or_pair", "score"]].rename(columns={"blind_label_or_pair": "blind_label", "score": "factual_accuracy_score"})
    same = lvl.merge(hh, on=["assessor_id", "dataset", "model", "blind_label", "factual_accuracy_score"], how="outer", indicator=True)
    lvp = pd.read_csv(HERE / "figure_7E_pairwise_similarity_by_assessor.csv")
    pp = p[["assessor_id", "dataset", "model", "blind_label_or_pair", "score"]].rename(columns={"model": "model_pair", "blind_label_or_pair": "blind_pair", "score": "pairwise_similarity_score"})
    samep = lvp.merge(pp, on=["assessor_id", "dataset", "model_pair", "blind_pair", "pairwise_similarity_score"], how="outer", indicator=True)
    ok2 = bool((same._merge == "both").all() and len(lvl) == 45 and (samep._merge == "both").all() and len(lvp) == 45)
    print(f"Figure 7E: {len(ref)} plotted values, max |difference| to the recomputed means = {diff.max():.2e}; all match = {ok}; assessor-level files (45 + 45 scores) consistent with the ratings = {ok2}")
    sys.exit(0 if (ok and ok2) else 1)


if __name__ == "__main__":
    main()
