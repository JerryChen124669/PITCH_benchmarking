| Input condition | Prompt | Answers judged together | LLMs judged together | Score per pass (1 / 2 / 3) | Score (mean of 3 passes) | Records (relative to this folder) |
|---|---|---|---|---|---|---|
| PITCH | CITE-seq CD45RA | 4 | Gemini-3.1-pro-preview; GLM-5.3; DeepSeek V4.1 Flash; Qwen3.8-2.4T-A95B | 9 / 8 / 8 | 8.33 | data/cond0_full_pipeline/CITEseq_CD45RA_naive_pass1-3.json |
| PITCH | CosMx Ki-67 | 4 | Gemini-3.1-pro-preview; GLM-5.3; DeepSeek V4.1 Flash; Qwen3.8-2.4T-A95B | 8 / 8 / 8 | 8.00 | data/cond0_full_pipeline/CosMx_CancerKi67_pass1-3.json |
| PITCH | Gouwens input resistance | 4 | Gemini-3.1-pro-preview; GLM-5.3; DeepSeek V4.1 Flash; Qwen3.8-2.4T-A95B | 6 / 6 / 8 | 6.67 | data/cond0_full_pipeline/Gouwens_GABAergic_pass1-3.json |
| PITCH | mean of the 3 prompts |  |  |  | 7.67 |  |
| MLP + IG | CITE-seq CD45RA | 4 | Gemini-3.1-pro-preview; GLM-5.3; DeepSeek V4.1 Flash; Qwen3.8-2.4T-A95B | 6 / 4 / 3 | 4.33 | data/cond1_ig_only/CITEseq_CD45RA_naive_pass1-3.json |
| MLP + IG | CosMx Ki-67 | 4 | Gemini-3.1-pro-preview; GLM-5.3; DeepSeek V4.1 Flash; Qwen3.8-2.4T-A95B | 6 / 4 / 4 | 4.67 | data/cond1_ig_only/CosMx_CancerKi67_pass1-3.json |
| MLP + IG | Gouwens input resistance | 4 | Gemini-3.1-pro-preview; GLM-5.3; DeepSeek V4.1 Flash; Qwen3.8-2.4T-A95B | 4 / 4 / 4 | 4.00 | data/cond1_ig_only/Gouwens_GABAergic_pass1-3.json |
| MLP + IG | mean of the 3 prompts |  |  |  | 4.33 |  |
| Measurement-only | CITE-seq CD45RA | 2 | GLM-5.3; Qwen3.8-2.4T-A95B | 5 / 5 / 5 | 5.00 | data_available_models/cond2_raw_data/CITEseq_CD45RA_naive_pass1-3.json |
| Measurement-only | CosMx Ki-67 | 3 | GLM-5.3; DeepSeek V4.1 Flash; Qwen3.8-2.4T-A95B | 3 / 3 / 2 | 2.67 | data_available_models/cond2_raw_data/CosMx_CancerKi67_pass1-3.json |
| Measurement-only | Gouwens input resistance | 2 | Gemini-3.1-pro-preview; GLM-5.3 | 1 / 1 / 1 | 1.00 | data_available_models/cond2_raw_data/Gouwens_GABAergic_pass1-3.json |
| Measurement-only | mean of the 3 prompts |  |  |  | 2.89 |  |
