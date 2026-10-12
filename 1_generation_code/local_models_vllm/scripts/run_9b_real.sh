#!/bin/bash
source "${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}" && conda activate "${VLLM_ENV:-vllm-nfs}"
export HF_HOME=${HF_HOME:?set HF_HOME to your model cache}
cd "$(dirname "$0")/.."
GPUS=${GPUS:-4,5,6,7}
# Qwen3.5-9B official thinking-mode sampling for general tasks (model card).
SAMPLING="--temperature 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 --presence-penalty 1.5 --repetition-penalty 1.0"
SYN=data/synthetic_prompt_dataset_v3_clean_refined_prompt_0828.jsonl

# 6 real case-study prompts, thinking mode, T=1.0. One engine per GPU (data parallel).
python src/infer_vllm.py --model Qwen/Qwen3.5-9B --input data/case_studies.jsonl \
    --thinking --prompt-format plain --gpus $GPUS --gpu-mem 0.90 $SAMPLING \
    --max-tokens 32768 --max-model-len 36000 \
    --out-dir runs/real_qwen35_9b || exit 1
python src/check_run.py runs/real_qwen35_9b
python src/make_trimmed.py runs/real_qwen35_9b real_qwen35_9b
