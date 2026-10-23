"""Figure S16 - effects of classifier choice (A-C) and decision-tree settings (D-E) on PITCH hypothesis generation.

Reads data/*.csv only (no API calls, nothing is refitted) and writes one file per panel (svg, pdf, png) plus
data/FigS16*_plot_values.csv with the numbers that are drawn.

  A  accuracy of LR / DT / RF in predicting the IG-derived label: 5-fold CV accuracy per dataset (mean of the 5 folds), then mean and SD
     (ddof = 1) over the 3 datasets with "Plotted in Fig S16A" = True (n = 3; pERK EGF10 is in the table but not plotted)
  B  five criteria per classifier: mean over the datasets with "Plotted in Fig S16B" = True (CITE-seq CD45RA, CosMx Ki-67, Gouwens input resistance)
  C  Hypothesis Similarity between classifiers: mean over the datasets with "Plotted in Fig S16C" = True of the mean of the six pairwise criteria
  D  six criteria per decision-tree setting: mean over the 3 datasets
  E  Hypothesis Similarity between settings: mean over the 3 datasets of the mean of the three pairwise criteria (blank diagonal)
Usage: python plot_FigS16.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"

plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"], "svg.fonttype": "none", "pdf.fonttype": 42})

INK, SECONDARY_INK, MUTED, GRIDLINE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
CLASSIFIERS = ["LR", "DT", "RF"]
CLF_STYLE = {"LR": ("#2a78d6", "o", "-"), "DT": ("#eb6834", "s", "--"), "RF": ("#1baf7a", "^", "-.")}
BAR_COLOR = {"LR": "#1C7293", "DT": "#0B4F6C", "RF": "#2A9D8F"}
SETTINGS = ["30 genes, depth 4", "60 genes, depth 4", "30 genes, depth 6", "60 genes, depth 6"]
SETTING_STYLE = dict(zip(SETTINGS, [("#2a78d6", "o", "-"), ("#eb6834", "s", "--"), ("#1baf7a", "^", "-."), ("#4a3aa7", "D", ":")]))
CRITERIA5 = ["Feature Comprehensiveness", "Biological Plausibility", "Mechanistic Synthesis", "Hypothesis Specificity", "Literature Grounding"]   # clockwise from the top
CRITERIA6 = ["Feature Comprehensiveness", "Biological Plausibility", "Hypothesis Specificity", "Mechanistic Synthesis", "Logic Rigor", "Literature Grounding"]
SIM_RAMP = ["#4575B4", "#91BFDB", "#E0F3F8", "#FFFFBF", "#FEE090", "#FC8D59", "#D73027"]
SIM_CMAP = mcolors.LinearSegmentedColormap.from_list("hypothesis_similarity", SIM_RAMP)
SIM_VMIN, SIM_VMAX = 1, 10
DIAG = "#ecebe6"


def save(fig, stem, **kw):
    for ext in ("svg", "pdf", "png"):
        fig.savefig(HERE / f"{stem}.{ext}", bbox_inches="tight", facecolor="white", **({"dpi": 300} if ext == "png" else {}), **kw)
    plt.close(fig)
    print("saved", stem)


def text_on(rgba):
    lin = lambda c: c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
    r, g, b = rgba[:3]
    return "white" if 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b) < 0.179 else INK


def radar(values, labels, series, stem, legend_cols, size=(6.4, 6.4)):
    """values: DataFrame indexed by series, one column per axis in `labels`."""
    ang = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
    ac = np.concatenate([ang, ang[:1]])
    fig = plt.figure(figsize=size)
    ax = fig.add_subplot(111, polar=True)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_ylim(0, 10)
    ax.set_yticks([2, 4, 6, 8, 10])
    ax.set_yticklabels(["2", "4", "6", "8", "10"], fontsize=9, color=MUTED)
    ax.set_rlabel_position(360 / len(labels) / 2)
    ax.set_xticks(ang)
    ax.set_xticklabels([l.replace(" ", "\n", 1) if "\n" not in l and len(l) > 14 else l for l in labels], fontsize=11, color=INK)
    ax.tick_params(axis="x", pad=14)
    ax.grid(color=GRIDLINE, linewidth=0.9)
    ax.spines["polar"].set_color(GRIDLINE)
    for name, (col, mk, ls) in series.items():
        v = values.loc[name, labels].astype(float).values
        vc = np.concatenate([v, v[:1]])
        ax.plot(ac, vc, color=col, lw=2, ls=ls, marker=mk, ms=6, markeredgecolor="white", markeredgewidth=1, label=name, zorder=3)
        ax.fill(ac, vc, color=col, alpha=0.06, zorder=2)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=legend_cols, frameon=False, fontsize=10.5, handlelength=2.4, columnspacing=1.6)
    save(fig, stem)


def heatmap(mat, names, stem, row_labels=None, col_labels=None, figsize=(5.2, 4.4), rotate_rows=False):
    n = len(names)
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, n)
    ax.set_ylim(n, 0)
    ax.set_aspect("equal")
    ax.axis("off")
    for i in range(n):
        for j in range(n):
            v = mat[i, j]
            face = DIAG if np.isnan(v) else SIM_CMAP((v - SIM_VMIN) / (SIM_VMAX - SIM_VMIN))
            ax.add_patch(Rectangle((j + 0.02, i + 0.02), 0.96, 0.96, facecolor=face, edgecolor="none", clip_on=False))
            ax.text(j + 0.5, i + 0.5, "-" if np.isnan(v) else f"{v:.1f}", ha="center", va="center",
                    fontsize=14 if n == 3 else 13, fontweight="bold", color=MUTED if np.isnan(v) else text_on(face))
    row_labels = row_labels or names
    col_labels = col_labels or names
    for k in range(n):
        ax.text(k + 0.5, n + 0.07, col_labels[k], ha="center", va="top", fontsize=11, color=INK)
        if rotate_rows:
            ax.text(-0.14, k + 0.5, row_labels[k], ha="center", va="bottom", fontsize=10.5, color=INK, rotation=90, linespacing=1.1)
        else:
            ax.text(-0.07, k + 0.5, row_labels[k], ha="right", va="center", fontsize=11, color=INK)
    cb_x, cb_w, steps = n + 0.3, 0.16 if n == 3 else 0.2, 60
    for s in range(steps):
        y0 = n - (s + 1) * n / steps
        ax.add_patch(Rectangle((cb_x, y0), cb_w, n / steps + 0.004, facecolor=SIM_CMAP((s + 0.5) / steps), edgecolor="none", clip_on=False))
    for t in range(SIM_VMIN, SIM_VMAX + 1):
        ax.text(cb_x + cb_w + 0.07, n - (t - SIM_VMIN) * n / (SIM_VMAX - SIM_VMIN), str(t), ha="left", va="center", fontsize=9.5, color=SECONDARY_INK)
    ax.text(cb_x + cb_w + 0.62, n / 2, "Hypothesis Similarity", ha="center", va="center", rotation=90, fontsize=11, color=INK)
    save(fig, stem)


def panel_A():
    """Accuracy bars over the three datasets used in panels B-E (rows with "Plotted in Fig S16A" = True; pERK EGF10 is in the table but not plotted)."""
    f = pd.read_csv(DATA / "FigS16A_accuracy_by_fold.csv")
    f = f[f["Plotted in Fig S16A"]]
    per_ds = f.groupby(["Classifier", "Dataset"])["Accuracy"].mean().unstack("Dataset").loc[CLASSIFIERS]       # mean of the 5 folds
    out = pd.DataFrame({"Mean accuracy over datasets": per_ds.mean(axis=1), "SD over datasets (ddof=1)": per_ds.std(axis=1, ddof=1)})
    out.round(4).join(per_ds.round(4)).to_csv(DATA / "FigS16A_plot_values.csv", index_label="Classifier")
    fig, ax = plt.subplots(figsize=(4.6, 5))
    x = np.arange(3)
    ax.bar(x, out.iloc[:, 0], width=0.6, color=[BAR_COLOR[c] for c in CLASSIFIERS], zorder=3)
    ax.errorbar(x, out.iloc[:, 0], yerr=out.iloc[:, 1], fmt="none", ecolor=INK, elinewidth=1.3, capsize=5, capthick=1.3, zorder=4)
    ax.set_xticks(x)
    ax.set_xticklabels(CLASSIFIERS, fontsize=13, color=SECONDARY_INK)
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=11, color=SECONDARY_INK)
    ax.set_ylabel("Accuracy", fontsize=12, color=SECONDARY_INK)
    ax.yaxis.grid(True, color=GRIDLINE, linewidth=0.9, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRIDLINE)
    save(fig, "FigS16A_accuracy")


def panel_B():
    q = pd.read_csv(DATA / "FigS16B_scores_per_case.csv")
    q = q[q["Plotted in Fig S16B"]]
    m = q.groupby("Classifier")[CRITERIA5].mean().loc[CLASSIFIERS]
    m.round(3).to_csv(DATA / "FigS16B_plot_values.csv", index_label="Classifier")
    radar(m, CRITERIA5, {c: CLF_STYLE[c] for c in CLASSIFIERS}, "FigS16B_criteria_by_classifier", 3)


def panel_C():
    s = pd.read_csv(DATA / "FigS16C_similarity_per_case.csv")
    s = s[s["Plotted in Fig S16C"]]
    col = "Hypothesis Similarity (mean of 6 criteria)"
    g = s.groupby("Pair")[col].mean()
    mat = np.full((3, 3), np.nan)
    for pair, v in g.items():
        a, b = pair.split(" vs ")
        i, j = CLASSIFIERS.index(a), CLASSIFIERS.index(b)
        mat[i, j] = mat[j, i] = v
    pd.DataFrame(mat, index=CLASSIFIERS, columns=CLASSIFIERS).round(3).to_csv(DATA / "FigS16C_plot_values.csv", index_label="Classifier")
    heatmap(mat, CLASSIFIERS, "FigS16C_similarity_between_classifiers")


def panel_D():
    d = pd.read_csv(DATA / "FigS16D_scores_per_setting.csv")
    m = d.groupby("Setting")[CRITERIA6].mean().loc[SETTINGS]
    m.round(3).to_csv(DATA / "FigS16D_plot_values.csv", index_label="Setting")
    radar(m, CRITERIA6, SETTING_STYLE, "FigS16D_criteria_by_DT_setting", 2, size=(6.8, 6.8))


def panel_E():
    e = pd.read_csv(DATA / "FigS16E_similarity_between_settings.csv")
    col = "Hypothesis Similarity (mean of 3 criteria)"
    g = e.groupby(["Setting A", "Setting B"])[col].mean()
    mat = np.full((4, 4), np.nan)
    for (a, b), v in g.items():
        i, j = SETTINGS.index(a), SETTINGS.index(b)
        mat[i, j] = mat[j, i] = v
    pd.DataFrame(mat, index=SETTINGS, columns=SETTINGS).round(3).to_csv(DATA / "FigS16E_plot_values.csv", index_label="Setting")
    lab = [s.replace(", ", "\n") for s in SETTINGS]
    heatmap(mat, SETTINGS, "FigS16E_similarity_between_DT_settings", row_labels=lab, col_labels=lab, figsize=(6.4, 5.6), rotate_rows=True)


if __name__ == "__main__":
    panel_A(); panel_B(); panel_C(); panel_D(); panel_E()
