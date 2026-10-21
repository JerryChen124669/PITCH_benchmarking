# Total hypothesis-generation score, 24 models (six criteria, six prompts)

- **Score:** logic rigor + feature comprehensiveness + hypothesis specificity + biological plausibility + mechanistic synthesis + literature grounding,
  each 0-10 at full weight, maximum 60. Zone identification is not used. Per model: the mean over all six empirical prompts of that sum (one judged answer per model and prompt).
- **Colours:** teal = similarity group 1, orange = group 2, blue = group 3 (groups from the 22-model clustering), grey = the two biomedical models.
- **Names:** short model names with total parameter counts for the large models, as in the reference image (`data/model_labels.csv`).

| File | Content |
|---|---|
| `total_score_barplot_6cases_excl_zone.png / .svg / .pdf` | the plot (SVG and PDF keep editable text) |
| `plot_total_score_barplot.py` | the code; `python plot_total_score_barplot.py` redraws the plot from `data/` |
| `total_score_values.csv` | the plotted values, group and label of every bar |
| `data/all_scores_24models_6prompts.csv` | judge scores per model and prompt (output of the evaluator) |
| `data/cluster_assignments.csv` | similarity group of each of the 22 clustered models |
| `data/model_labels.csv` | model id -> label on the plot |

Note: the bar heights of the reference image do not match this score. They were not reproduced by the six-criteria sum used here, nor by the "excluding specificity" sum
(zone identification included) of the earlier overall-rank plot, so the reference image was probably made from a different version of the scores.
