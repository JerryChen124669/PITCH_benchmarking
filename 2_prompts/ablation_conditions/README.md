# Ablation-condition prompts

The prompts of the evidence-condition ablation: what the LLM is given when asked to formulate a hypothesis for a prediction target. Three datasets (CITE-seq CD45RA, CosMx Ki-67, Gouwens/Patch-seq input resistance) x three conditions.

| File pattern | Condition (label in the results) | Content |
|---|---|---|
| `<dataset>/<dataset>_cond0_full_pipeline_prompt.txt` | **PITCH** | decision-tree zones and raw tree rules (MLP + integrated gradients + decision tree), focus feature and target |
| `<dataset>/<dataset>_cond1_ig_only_prompt.txt` | **MLP + IG** | the full ranking of every feature by mean \|IG\| attribution to the target; no decision tree, no focus feature |
| `<dataset>/<dataset>_cond2_raw_data_prompt.txt` | **Measurement-only** | a decile-stratified sample of cells x standardised features (CSV), data description and task; no IG, no tree |

* The measurement-only prompts contain the sampled cell-by-gene measurement tables (45 x 1,500 genes for CITE-seq, 152 x 500 for CosMx, 77 x 1,000 for Patch-seq) and are therefore large (0.4-1 MB each).
* The measurement-only prompt exists in three wordings that differ in one instruction sentence (none, "pick the important features", and the final "rank features by correlation with the target"); the files here are the final wording, and the wording each answer saw is recorded in the answer file (`prompt_variant`) and reproduced by `4_evaluation_code/ablation_conditions/raw_prompt_versions.py`.
* The PITCH prompt is the original decision-tree interpretation prompt and still asks for four contextual features (`top_4_contexts`); the MLP + IG and measurement-only prompts do not.
