# Selection provenance of the 144 evaluated responses

The 144 files `3_answers/<model>/<case>_selected_answer.json` (24 models x 6 cases) are the canonical evaluated outputs. The responses (`answer` text) were **not modified, replaced or regenerated**; the only edit to these files is the label correction of the six Gemma 4 12B files described below;
this folder only adds the records needed to see how each one was chosen. Missing information is written as `unrecorded` or `not available`; nothing was reconstructed or inferred.
Model outputs from provider-served or sampled models are stochastic and cannot be assumed to be regenerable verbatim; the archived candidates below are the record, not a reproduction recipe.

| Path | Content |
|---|---|
| `selection_manifest.csv` | one row per model x case (144 rows), columns below |
| `manual_exceptions.csv` | the cases that need an explanation (random selections, manual selection, missing raw candidates, a label that does not describe the actual selection) |
| `selector_judgments/<model>__<case>.json` | for the 59 cases that went through the consistency check (57 `consistent_pair`, 2 `random_no_consensus`) plus the manual Gemini Fxyd6 case: the selector's judgment text verbatim and what is not stored |
| `raw_candidates/<model>/<case>_result.json` | all candidates of the model-case (`answers` list, as generated), unchanged copies of the files in the workstation `trial_results/` folder (137 cases) |
| `raw_candidates/gemini-3.1-pro-preview/<case>_results.jsonl` | the original 5-run generation archive of Gemini (runs 1-5) |
| `raw_candidates/gemini-3.1-pro-preview/superseded_fxyd6/` | the two earlier states of Gemini Fxyd6 (see below) |

`model` = the folder name in `3_answers/` (the 9B model is `Qwen3.5-9B-think-temp1`; in the raw candidate copies its `model` metadata label was corrected to this name, the answers are untouched). `case` = prompt file stem.

## Manifest columns
`candidate_count` candidates available to the selection; `selected_index` the candidate number, **1-based** (= `source_answer_index` of the canonical file + 1); `selection_method` the label of the canonical file;
`selector_judgment_file` path or `not applicable`; `random_seed` seed of the random step, `unrecorded` where one existed, `not applicable` otherwise; `formatting_retries`; `generation_date`;
`raw_candidates_available` (`yes`/`no`); `exception_notes`.

## Audit (what the records show)
- 144 canonical files; selection methods **78 `single_sample_no_consistency_check`, 57 `consistent_pair`, 6 `embedding_medoid_of_5` (Gemma 4 12B), 2 `random_no_consensus`, 1 `manual_selection_from_archive`**, as labelled in the files (the six Gemma labels were corrected, see below).
- All 144 canonical answers are byte-identical to the `selected_answer` files in the workstation `trial_results/`, and for the 143 cases with a raw candidate file the canonical answer equals candidate `source_answer_index` of that file.
- Candidate counts: 78 cases with 1 candidate, 46 with 3, 19 with 5, 1 `unrecorded` (MiniMax EGR1).
- Raw candidates found for 137 of 144 cases. Not found: Gemma 4 12B (6 cases; 5 candidates were generated, only the selected answer was kept) and MiniMax-M3 EGR1 (the file with the other candidates is missing).

## Special cases
- **Two `random_no_consensus`** (BioMistral IL2RA, 5 candidates, candidate 4; gpt-oss-120b EGR1, 3 candidates, candidate 1): the selector found no agreeing pair and `select_best_answer.py` drew a candidate with `random.choice`. The script does not seed Python's random module and the draw was not logged, so the seed is `unrecorded`.
- **Gemini Fxyd6** (canonical = candidate 5, `manual_selection_from_archive`): history from the records: (1) the first selection, a consistent pair over runs 1-3, chose run 3; (2) this was replaced by one ad hoc regeneration (`generate_gemini_answer.py`, single sample); (3) the ad hoc answer was replaced by run 5 of the original 5-run archive, "to avoid having two different provenances". All three states and the five archived candidates are in `raw_candidates/gemini-3.1-pro-preview/`.
  **Selection criterion unrecorded**: no saved note, log or script states why run 5 (rather than run 3 or another run) was chosen; no justification is inferred here.
- **Gemini, other five prompts**: five runs were generated; the self-consistency selection used runs 1-3 (`candidate_count` 3); runs 4-5 are in the archive jsonl only.
- **Gemma 4 12B** (6 cases): five candidates (seeds 1234-1238) were generated and the embedding medoid was kept (`1_generation_code/local_models_transformers/gemma12B/scripts/select_consistent.py`). The canonical files originally carried the label `single_sample_no_consistency_check` with the note "Only one generation was provided"; on the author's approval the **label, note and `source_answer_index` of these six files were corrected** (`embedding_medoid_of_5`, a note describing the medoid selection, `unrecorded`). The `answer` text was not changed (`canonical_answer_text.sha256`). The index of the medoid among the five is `unrecorded` and the five raw candidates were not found.

## What is not recorded
- The selector's full raw output (only its `reason` text is stored in `selection_notes`), the selector model per selection (`select_best_answer.py` names gpt-5.6-sol), and the random order in which candidates were shown to it.
- **Formatting retries**: counts were not stored in any result file (`unrecorded` for all rows). The generation scripts contain retry logic (`infer_vllm.py --max-format-retries`, default 3 at temperature > 0; `official_trial_batch2.py` retries on transient errors and token-limit truncation), so retries may have occurred.
- **Generation date** is the model-level date of the result files in the settings workbook (`1_generation_code/model_settings/llm_generation_settings.xlsx`, "Result files dated"), not a per-answer timestamp.

The expert-assessment records are in `5_evaluation_results/expert_assessment/`. A *SHA-256 hash* is a fixed-length fingerprint computed from a file's content: if a single character changes, the fingerprint changes, so a reviewer can verify that a file is the archived one. `canonical_answer_text.sha256` holds the fingerprint of the `answer` text of every canonical file (taken before the Gemma label correction), `canonical_selected_answers.sha256` the fingerprint of the whole files as archived. `validate_selection_provenance.py` (result: `validation_report.txt`) checks both.
