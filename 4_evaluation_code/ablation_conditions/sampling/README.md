# Measurement-only sampling (code, seed and the sampled cells)

The measurement-only prompts (`2_prompts/ablation_conditions/<dataset>/*_cond2_raw_data_prompt.txt`) contain a sample of cells x all standardised features and nothing else.

| Item | Value |
|---|---|
| Code | `raw_data_sampler.py` (`stratified_indices`, `fit_to_token_budget`), called by `run_ablation.py` |
| Rule | decile-stratified by the **target response** (e.g. CD45RA): the target is cut into 10 quantile bins and cells are drawn without replacement, as equally as possible per bin; a bin that is too small is topped up from the remaining cells. No feature is removed. |
| Seed | `seed = 42` (`numpy.random.default_rng(42)`), in `stratified_indices` and `fit_to_token_budget` |
| Number of cells | the largest sample whose data block fits half of Gemini's 1,048,576-token input limit (`config.RAW_DATA_TOKEN_BUDGET` = 524,288), checked with the tokenizer: 45 of 8,617 cells (CITE-seq CD45RA, 1,500 features), 152 of 30,000 (CosMx Ki-67, 500 features), 77 of 3,395 (Gouwens input resistance, 1,000 features) (`measurement_only_sampling_summary.csv`) |
| Cells used | `measurement_only_sampled_cells.csv`: 0-based row index in the original dataset for every sampled cell (the datasets themselves are not in this repository) |
| Same cells in the other conditions | no sampling in the other conditions: PITCH uses the decision tree fitted on all cells and the 29 top-IG features; MLP + IG uses the IG ranking of all features computed on all cells. The sampled cells are used only in the measurement-only prompt. |

The data rows in the repository prompts are identical to the rows of the original ablation prompts that this code produced; only one instruction sentence differs between the three wordings of the prompt (see `../raw_prompt_versions.py` and `2_prompts/ablation_conditions/README.md`).
`pipeline.py`, `prompt_builder.py` and `config.py` are the original modules that `run_ablation.py` imports (MLP: hidden size 256, dropout 0.5, lr 2e-4, weight decay 1e-4, 2,000 epochs, seed 20; IG: 50 steps). They need the original datasets and `GEMINI_API_KEY` (token counting) in the environment and are provided as the exact record, not as a runnable demo.
