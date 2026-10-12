#!/bin/bash
# The six large models over the 6 real case-study prompts, TP=4 on GPUs 4-7, one at a time:
#   download -> run -> validate -> (retry with a bigger budget if truncated) -> delete the cache.
#
# Sampling follows each model's OWN official recommendation (HF model card / generation_config),
# not one shared setting -- they disagree with each other:
#   Qwen3.8-27B   thinking:  T1.0 top_p0.95 top_k20 min_p0 presence0.0 rep1.0
#   Qwen3.6-35B   thinking:  T1.0 top_p0.95 top_k20 min_p0 presence1.5 rep1.0   (general tasks)
#   Nemotron 3 Nano reasoning: T1.0 top_p1.0, no top_k
#   Gemma 4 31B-it:         T1.0 top_p0.95 top_k64        (card: same config for all use cases)
#   Mistral Small 3.1 24B:  T0.15                         (card: "relatively low temperature")
#   Qwen3-Next-80B Instruct: T0.7 top_p0.8 top_k20 min_p0
source "${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}" && conda activate "${VLLM_ENV:-vllm-nfs}"
export HF_HOME=${HF_HOME:?set HF_HOME to your model cache}
cd "$(dirname "$0")/.."
HUB=$HF_HOME/hub

do_model() {  # repo, run-name  (sampling/mode flags come from the EXTRA array)
  local repo=$1 name=$2
  local dir="$HUB/models--${repo//\//--}"
  echo "=== $repo -> runs/$name  ($(date +%H:%M))"
  python -c "
from huggingface_hub import snapshot_download
print(snapshot_download('$repo', max_workers=8,
      allow_patterns=['*.json','*.safetensors','*.txt','*.model','*.jinja','*.py'],
      ignore_patterns=['consolidated.safetensors','original/*']))" || return 1
  for budget in "65536 70000" "131072 140000"; do
    set -- $budget
    python src/infer_vllm.py --model "$repo" --input data/case_studies.jsonl \
        --prompt-format plain --tp 4 --gpus 4,5,6,7 --max-tokens $1 --max-model-len $2 \
        --out-dir "runs/$name" "${EXTRA[@]}" || return 1
    if python src/check_run.py "runs/$name"; then
      echo "=== $name OK; removing $dir"; rm -rf "$dir"; return 0
    fi
    echo "=== $name incomplete at max_tokens=$1; retrying with a bigger budget"
    mv "runs/$name" "runs/${name}_trunc$1"
  done
  echo "=== $name STILL INCOMPLETE - cache kept at $dir"; return 1
}

COMMON=(--gpu-mem 0.92 --temperature 1.0 --repetition-penalty 1.0 --min-p 0.0)

EXTRA=(--thinking "${COMMON[@]}" --top-p 0.95 --top-k 20 --presence-penalty 0.0)
do_model Qwen/Qwen3.8-27B real_qwen38_27b_think_t1

EXTRA=(--thinking "${COMMON[@]}" --top-p 0.95 --top-k 20 --presence-penalty 1.5)
do_model Qwen/Qwen3.6-35B-A3B real_qwen36_35b_think_t1

EXTRA=(--thinking "${COMMON[@]}" --top-p 1.0 --top-k 0 --presence-penalty 0.0)
do_model nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 real_nemotron3_nano_30b_think_t1

EXTRA=(--thinking "${COMMON[@]}" --top-p 0.95 --top-k 64 --presence-penalty 0.0)
do_model google/gemma-4-31B-it real_gemma4_31b_think_t1

EXTRA=(--gpu-mem 0.92 --temperature 0.15 --top-p 1.0 --top-k 0 --min-p 0.0 \
       --presence-penalty 0.0 --repetition-penalty 1.0 --tokenizer-mode mistral)
do_model mistralai/Mistral-Small-3.1-24B-Instruct-2503 real_mistral_small31_24b_t1

EXTRA=(--gpu-mem 0.97 --eager --temperature 0.7 --top-p 0.8 --top-k 20 --min-p 0.0 \
       --presence-penalty 0.0 --repetition-penalty 1.0)
do_model Qwen/Qwen3-Next-80B-A3B-Instruct real_qwen3next_80b_t1

echo ALL_BIG_DONE
