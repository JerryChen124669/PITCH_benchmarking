"""
Total hypothesis-generation score of the 24 models: bar plot.

    total score = logic rigor + feature comprehensiveness + hypothesis specificity + biological plausibility
                  + mechanistic synthesis + literature grounding        (zone identification is NOT used)
    each criterion 0-10, so the maximum is 60; model score = mean over the six empirical prompts of that sum.

Derived from plot_evaluation_pillars.py (the 'overall rank' plot), with the criteria changed to the six above, all six prompts,
the short model names, and the colours of the three similarity groups:
    teal    = group 1 (rule-faithful descriptive annotators)
    orange  = group 2 (integrative, weakly grounded hypothesis generators)
    blue    = group 3 (mechanistic, literature-grounded integrators)
    grey    = the two biomedical models (not part of the similarity clustering)

Reads (this folder):  data/all_scores_24models_6prompts.csv, data/cluster_assignments.csv, data/model_labels.csv
Writes (this folder): total_score_barplot_6cases_excl_zone.{png,svg,pdf}, total_score_values.csv

    python plot_total_score_barplot.py
"""
from pathlib import Path

import matplotlib

matplotlib.rcParams["pdf.fonttype"] = 42       # editable text in Illustrator
matplotlib.rcParams["svg.fonttype"] = "none"
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"

CRITERIA = ["Logic Rigor_zone1", "Feature Comprehensiveness_zone1", "Specificity_zone1",
            "Biological Plausibility_zone1", "Grounded Mechanistic Synthesis_zone1", "Literature Grounding_zone1"]
COLORS = {1: "#66C2A5", 2: "#FC8D62", 3: "#8DA0CB", "bio": "#BDBDBD"}   # ColorBrewer Set2 + grey

scores = pd.read_csv(DATA / "all_scores_24models_6prompts.csv")
scores["slug"] = scores["model"].str.replace("/", "__", regex=False)
assert scores["slug"].nunique() == 24, scores["slug"].nunique()
assert scores.groupby("slug")["prompt_file"].nunique().eq(6).all(), "every model must have all six prompts"
assert not scores[CRITERIA].isna().any().any(), "missing criterion scores"
scores["total"] = scores[CRITERIA].sum(axis=1)
totals = scores.groupby("slug")["total"].mean()

labels = pd.read_csv(DATA / "model_labels.csv").set_index("slug")
groups = pd.read_csv(DATA / "cluster_assignments.csv").set_index("model")["cluster"]
rows = []
for slug, value in totals.items():
    label = labels.loc[slug, "label"]
    group = int(groups[slug]) if slug in groups.index else "bio"
    rows.append((slug, label, group, value))
df = pd.DataFrame(rows, columns=["model_slug", "label", "group", "total_score_0_60"]).sort_values("total_score_0_60").reset_index(drop=True)
df.to_csv(HERE / "total_score_values.csv", index=False)

fig, ax = plt.subplots(figsize=(14.0, 5.2))
ax.bar(range(len(df)), df["total_score_0_60"], width=0.8, color=[COLORS[g] for g in df["group"]], edgecolor="#222222", linewidth=0.8, zorder=3)
ax.set_xticks(range(len(df)))
ax.set_xticklabels(df["label"], rotation=30, ha="right", rotation_mode="anchor", fontsize=11)
ax.set_xlim(-0.9, len(df) - 0.1)
ax.set_ylim(0, 60)
ax.set_yticks(range(0, 61, 10))
ax.tick_params(axis="y", labelsize=12)
ax.set_ylabel("Total hypothesis-generation score", fontsize=12)
ax.grid(axis="y", color="#d0d0d0", linestyle="--", linewidth=0.9, zorder=0)
ax.set_axisbelow(True)
for s in ax.spines.values():
    s.set_linewidth(1.0)
fig.tight_layout()
for ext in ("png", "svg", "pdf"):
    fig.savefig(HERE / f"total_score_barplot_6cases_excl_zone.{ext}", dpi=300 if ext == "png" else None, bbox_inches="tight")
plt.close(fig)
print(df.to_string(index=False))
