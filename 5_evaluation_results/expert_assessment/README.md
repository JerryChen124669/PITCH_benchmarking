# Expert assessment (blinded evaluation of mechanistic hypotheses)

De-identified records of the blinded expert survey and the numbers plotted in Figure 7E.

| Path | Content |
|---|---|
| `questionnaire/` | `questionnaire_items.csv` (the 36 items) and `questionnaire_reconstructed.md`: a fixed text copy **reconstructed from the column headers of the survey export** (the original form was not found; see the checklist) |
| `background_materials/` | `dataset_titles_in_survey.csv` (section titles that appear in the export) and a README listing the background material that was not found |
| `anonymized_hypotheses/` | the hypotheses exactly as shown, with the anonymous labels A, B, C, one file per dataset |
| `hypothesis_label_mapping.csv` | blind label -> model for every dataset, canonical response it corresponds to, text hash |
| `deidentified_ratings.csv` | one row per assessor x dataset x item (180 ratings): 5 assessors x 3 datasets x (3 hypotheses x 3 pillars + 3 pairs) |
| `aggregate_expert_ratings.py` | aggregation of the ratings into the plotted values; checks them against `figure_7E_plotting_input.csv` |
| `figure_7E_plotting_input.csv` | **Figure 7E**: the plotted factual-accuracy values (3 models x 3 datasets + overall mean), frozen from the evaluation workbook the figure was drawn from |
| `figure_7E_factual_accuracy_by_assessor.csv` | the 45 individual factual-accuracy scores (5 assessors x 3 datasets x 3 models) behind those means |
| `aggregated_expert_ratings.csv` | output of the aggregation script: factual accuracy (the Figure 7E metric) plus plausibility and pairwise concordance, which are not plotted in Figure 7E |
| `validate_expert_archive.py`, `validation_report.txt` | the archive checks and their result |

## What the assessors did
Three datasets (pERK / EGF, GABAergic input resistance with Fxyd6, Ki-67 with PTMA), each with three single-answer hypotheses from three models (Gemini-3.1-Pro, Qwen3.8-2.4T-A95B and Qwen3.5-9B) under anonymous labels A, B, C; the label order differs between datasets (`hypothesis_label_mapping.csv`).
For every hypothesis the assessor gave three 1-5 scores: **Core biological plausibility**, **Mechanistic specificity and completeness**, **Accuracy of specific molecular claims** (called "Factual accuracy" in the figure data); then rated each of the three pairs for **mechanistic concordance** (1-5).
Five assessors completed the survey. The CD45RA case was not part of the survey.
The text shown is a condensed summary (five sentences, per the source document) of each model's primary-zone rationale, not the full response. The summaries correspond to the canonical selected responses: for all nine, the saved candidate most similar to the shown text (TF-IDF cosine, `lexical_check_*` columns of the mapping) is the canonical selected candidate (Gemini candidates 3, 5 and 2 for pERK, GABA and Ki-67). This is a similarity indication; no link between summary and response was recorded.

## Figure 7E and the aggregation
Figure 7E shows the blinded expert test: the **factual accuracy** (accuracy of specific molecular claims, 1-5) of the single-answer hypotheses of Qwen3.5-9B, Qwen3.8-2.4T and Gemini-3.1-Pro, as the mean over the 5 assessors for each dataset and the overall mean of the three datasets (Gemini-3.1-Pro 3.40; Qwen3.8-2.4T 3.87; Qwen3.5-9B 2.73).
`figure_7E_plotting_input.csv` holds these values as stored in the `Figure data` sheet of the evaluation workbook the figure was drawn from (the version with 5 assessors; an earlier workbook with 4 assessors was superseded and is not part of this archive). The plausibility and pairwise-concordance sheets of the same workbook are not part of Figure 7E; their per-assessor ratings are kept in `deidentified_ratings.csv` and aggregated in `aggregated_expert_ratings.csv`.
The workbook computed the means with spreadsheet formulas (`AVERAGE` over the assessor rows); no script was found. `aggregate_expert_ratings.py` repeats the same arithmetic on the de-identified ratings and reproduces all 12 plotted values exactly (and checks that the assessor-level file equals the ratings). It is a re-implementation checked against the workbook values, not the original code.
```
python aggregate_expert_ratings.py      # writes aggregated_expert_ratings.csv and exits with 1 if a plotted value differs
python validate_expert_archive.py       # all archive checks
```

## De-identification
Assessors are `Assessor_1` ... `Assessor_5`, numbered in order of submission. Names, e-mail addresses, job titles, timestamps and the raw survey export are not included (the export had no name or e-mail field; the timestamps were dropped). No qualification, recruitment or affiliation information is stored.

## Source and verification of the ratings
Assessors 1-4: the survey export was available locally and every one of their 144 ratings in the workbook was checked against it (0 differences). **Assessor 5: the ratings come from the evaluation workbook only; the survey export that contains them was not found locally** (column `record_source` in the ratings file).

## Missing-record checklist (not found locally; nothing was estimated or reconstructed beyond what is stated)
- [ ] the original questionnaire as presented (form text, introduction, instructions, definitions of the three pillars, labels of the 1-5 scale, page order; a PDF print of the form is needed);
- [ ] the background information given to the assessors (dataset description, decision-tree zones, what a "node" is, ...) - only the section titles in the export were found; the title of the first section (pERK) is not in the export;
- [ ] the survey export that contains Assessor 5;
- [ ] how the assessors were recruited, their expertise and the instructions they received (unrecorded);
- [ ] the file `mechanism_summaries_three_models(3).docx` named in the earlier workbook (the version found has the same labels and hypotheses but lists the model beside each label);
- [ ] the script that produced the Figure 7E plot (only the plotted values and an SVG exist).
