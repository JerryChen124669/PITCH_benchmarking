# Local vLLM runs: decision-tree interpretation prompts

Code and run configurations for the models served locally with vLLM. A prompt (system + user) goes in; the model's
reasoning block (if any) and final answer come out. Only the answer is passed on to evaluation.

## Settings used for the case-study answers
Prompts were sent unchanged (`--prompt-format plain`; they request no output format). vLLM 0.22.1, bfloat16, seed 0.
Each model used the sampling recommended on its own Hugging Face model card / `generation_config.json`, so the
settings differ between models; `configs/*.json` is the authoritative record of each run.

| Model | Mode | T | top_p | top_k | min_p | presence | rep. | max_tokens | Output tokens (mean / max) |
|---|---|---|---|---|---|---|---|---|---|
| Qwen/Qwen3.5-9B | thinking | 1.0 | 0.95 | 20 | 0.0 | 1.5 | 1.0 | 32,768 | 7,597 / 8,983 |
| Qwen/Qwen3.8-27B | thinking | 1.0 | 0.95 | 20 | 0.0 | 0.0 | 1.0 | 65,536 | 33,486 / 44,058 |
| Qwen/Qwen3.6-35B-A3B | thinking | 1.0 | 0.95 | 20 | 0.0 | 1.5 | 1.0 | 65,536 | 7,355 / 8,046 |
| nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 | thinking | 1.0 | 1.0 | off | 0.0 | 0.0 | 1.0 | 65,536 | 5,899 / 6,779 |
| google/gemma-4-31B-it | thinking | 1.0 | 0.95 | 64 | 0.0 | 0.0 | 1.0 | 65,536 | 2,731 / 2,999 |
| mistralai/Mistral-Small-3.1-24B-Instruct-2503 | no thinking mode | 0.15 | 1.0 | off | 0.0 | 0.0 | 1.0 | 65,536 | 1,807 / 2,008 |
| Qwen/Qwen3-Next-80B-A3B-Instruct | no thinking mode | 0.7 | 0.8 | 20 | 0.0 | 0.0 | 1.0 | 65,536 | 3,239 / 3,822 |

Sampling follows each model's own recommendation, so differences between models include sampling differences.

Format check: an answer is accepted if the reasoning block is at least 20 characters (thinking models) and the answer at
least 50 characters; otherwise it is regenerated with seed + attempt - 1, up to 3 attempts (`--max-format-retries`).
The shipped answer files do not record how many attempts each answer needed.
Thinking models: the text before the closing reasoning tag is stored as reasoning and only the text after it is the answer.

| | Qwen 9B | Six larger models |
|---|---|---|
| Hardware | 4 x RTX A6000 48 GB | same |
| Parallelism | TP=1, 4 data-parallel replicas | TP=4, one engine |
| GPU memory utilization | 0.90 | 0.92 (0.97 for Qwen3-Next-80B) |
| CUDA graphs | on | on (off for Qwen3-Next-80B) |

## Contents
```
src/infer_vllm.py        runner: prompt rendering through the model's chat template, sampling, format check and retries
src/interpret_prompt.py  prompt helpers and the answer validation rule
src/check_run.py         every row valid and nothing cut off at max_tokens
src/make_trimmed.py      writes the distributable [key, id, thinking, answer] file
scripts/run_9b_real.sh   the 9B run          scripts/run_all_big.sh   the six larger models, one at a time
configs/*.json           resolved settings of each run (start times 2026-09-17 for the 9B, 2026-09-23 for the others)
data/build_case_studies.py   how the prompt .txt files are split into system / user text (needs the original
                         reference-answer files, which are not part of this repository)
env/requirements.txt     vllm 0.22.1, torch 2.11.0+cu130, transformers 4.57.1, flashinfer 0.6.11
```

## Reproducing a run
Set `HF_HOME` to a model cache. `data/case_studies.jsonl` (one row per case with `key`, `system_prompt`,
`user_prompt`) is built from the prompt files by `data/build_case_studies.py`; the runner only needs the prompt fields.
```bash
python src/infer_vllm.py --model Qwen/Qwen3.6-35B-A3B --input data/case_studies.jsonl --prompt-format plain --thinking     --tp 4 --gpus 4,5,6,7 --gpu-mem 0.92 --temperature 1.0 --top-p 0.95 --top-k 20 --min-p 0.0     --presence-penalty 1.5 --repetition-penalty 1.0 --max-tokens 65536 --max-model-len 70000     --out-dir runs/real_qwen36_35b_a3b
python src/check_run.py runs/real_qwen36_35b_a3b
python src/make_trimmed.py runs/real_qwen36_35b_a3b real_qwen36_35b_a3b
```
Model-specific flags for the other models are in `scripts/run_all_big.sh` and `configs/`.
Special handling already in the code: Gemma 4 marks its reasoning with a `thought` channel (split at the first markdown
header); Mistral Small 3.1 has no Hugging Face chat template (`--tokenizer-mode mistral`); Qwen3-Next-80B needs
`--gpu-mem 0.97 --eager` to fit on 4 x 48 GB.
