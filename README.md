# PITCH: biological hypothesizing from decision-tree prompts

Code, prompts, generated answers, similarity and quality evaluation, expert assessment and results for the 24-model benchmark of hypotheses generated from decision-tree interpretation prompts, with the supplementary ablation, classifier and decision-tree-setting analyses.

## Repository layout
| Folder | Content |
|---|---|
| `1_generation_code/` | scripts and settings used to generate the hypotheses of the 24 models: `api_models_together/` (11 Together AI models), `api_model_gemini/` (gemini-3.1-pro-preview), `local_models_vllm/` (Qwen3.8-27B, Qwen3.6-35B-A3B, Nemotron 3 Nano 30B-A3B, Gemma 4 31B-it, Mistral Small 3.1 24B, Qwen3-Next-80B-A3B-Instruct, Qwen3.5-9B), `local_models_transformers/` (Gemma 4 12B-it, Phi-4 14B, Ministral 3 14B, BioMistral-7B, PMC-LLaMA-13B), `answer_selection_and_conversion/` (self-consistency selection and converters into the common layout), `model_settings/` (generation settings workbook, version manifest, model registry) |
| `2_prompts/` | the six empirical prompt files (system prompt, experiment context, target zones, task); `ablation_conditions/` holds the ablation prompts |
| `3_answers/` | `<model>/<case>_selected_answer.json`: the 144 canonical evaluated answers (24 models x 6 cases); `selection_provenance/`: raw candidates and selection records; `ablation_conditions/`: ablation answers |
| `4_evaluation_code/` | rubric judging, pairwise similarity, clustering and selection scripts; `reproduce_figures.sh` regenerates the five evaluation figures (7A-7D, 7F); `ablation_conditions/` holds the ablation scripts |
| `5_evaluation_results/` | judged scores, pairwise similarity records, result tables and figures; `expert_assessment/` (blinded expert survey, Figure 7E); `ablation_conditions/` |
| `6_supplementary_figures/` | data tables, plotting scripts and records behind Supplementary Figures S15 (ablation) and S16 (classifier choice, decision-tree settings) |

## Reproducing the results
- **Evaluation figures (7A-7D, 7F):** `bash 4_evaluation_code/reproduce_figures.sh`, run directly from the repository (no staging); it reads the judged data in `5_evaluation_results/` and needs numpy, pandas, scipy, scikit-learn, matplotlib and Pillow, but no API key. See `4_evaluation_code/README.md`.
- **Expert assessment (Figure 7E):** `python 5_evaluation_results/expert_assessment/aggregate_expert_ratings.py`; checks: `validate_expert_archive.py` in the same folder.
- **Supplementary figures:** `python plot_FigS15.py` in `6_supplementary_figures/FigS15_ablation/` and `python plot_FigS16.py` in `6_supplementary_figures/FigS16_classifier_and_DT_settings/`; the records behind S15 are in the `ablation_conditions/` folders, those behind S16 in `6_supplementary_figures/FigS16_classifier_and_DT_settings/records/`.
- **Selection provenance:** `python 3_answers/selection_provenance/validate_selection_provenance.py`.
- Re-running model generation or the LLM judges needs the original datasets and API keys (read from environment variables; none is stored) and will not reproduce the answers verbatim (see below).

## Canonical outputs, provenance and expert assessment
- **The 144 selected responses are the canonical evaluated outputs.** `3_answers/<model>/<case>_selected_answer.json` is the single response per model and case that every score, similarity and figure in this repository was computed from. These files are not regenerated or edited.
- **Raw candidates are archived separately, where they could be recovered.** `3_answers/selection_provenance/` holds the raw candidate files of 137 of the 144 cases, `selection_manifest.csv` (one row per canonical response: candidate count, selected candidate, method, selector judgment, seed, retries, date), `manual_exceptions.csv` and the selector judgments. The seven cases without recovered raw candidates (six Gemma 4 12B, one MiniMax) are marked in the manifest.
  Selection methods: 78 single-sample, 57 consistent-pair, 6 embedding-medoid of five (Gemma 4 12B), 2 random no-consensus, 1 manual selection from the archive (Gemini, Fxyd6; its selection criterion is unrecorded). The responses themselves are unchanged; only the label of the six Gemma files was corrected.
- **Some settings or raw files are unavailable and are labelled `unrecorded`** (or `not available`) rather than inferred: for example the seed of the two random selections, the retry counts of the vLLM and second-batch Together runs (seeds and retries are recorded where a record exists, see `3_answers/selection_provenance/README.md`), checkpoint revisions, the five raw Gemma 4 12B candidates and the candidates of one MiniMax case. `1_generation_code/model_settings/benchmark_version_manifest.csv` lists, per model, the settings that were explicitly recorded, those left to the provider, and those unrecorded.
- **Provider-served and stochastic generations are not expected to be reproduced verbatim.** Sampling, provider-side model updates and unrecorded defaults mean that re-running a generation gives different text; the archived candidates and selected responses are the record.
- **The expert-assessment records are de-identified** (`5_evaluation_results/expert_assessment/`): five assessors (PhD-candidate level or above in biology-related fields, with research experience in molecular and cellular regulation) are anonymous (`Assessor_1` ...); no names, contact details or timestamps are kept. The original questionnaire (PDF), the background documents given to the assessors, the hypotheses as shown, the label-to-model mapping and the per-assessor ratings are archived; items that could not be recovered are listed in that folder's README.

## Citing this repository
Cite a fixed version, not the moving `main` branch: the release tag **`v1.0-submission`** and the exact commit it points to (given in `1_generation_code/model_settings/benchmark_repository_commit.txt`). A DOI has not been assigned yet; one can be minted by archiving the tagged release (for example with Zenodo's GitHub integration) and should then be added here.
