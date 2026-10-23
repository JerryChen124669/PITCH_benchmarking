# benchmark_version_manifest.csv

One row per model (24). Columns: `model` (folder in `3_answers/`), `provider`, `model_identifier`, `checkpoint_revision`, `precision_or_quantization`, `temperature`, `top_p`, `reasoning_setting`, `max_output_tokens`, `seed`, `generation_date`, `setting_source`, `provider_controlled_defaults`, `notes`.
Values come from `llm_generation_settings.xlsx` and its sources; nothing was added.

Three kinds of entry:
- **Explicitly recorded**: a value that the run script or config set (for example `0.3`); the `notes` column lists them per model.
- **Provider- or library-controlled default**: the request did not set the parameter, so the provider's or library's default applied; the value is written `provider default (value unrecorded)` and the parameter is listed in `provider_controlled_defaults`.
- **Unrecorded**: not stored anywhere. This is the case for every `checkpoint_revision` (no immutable revision or commit hash of any checkpoint was recorded; provider-served models may have been updated by the provider under the same identifier) and for the precision of provider-served models.

`generation_date` is the date of the result files, not a per-answer timestamp. The repository commits are in `benchmark_repository_commit.txt`.
