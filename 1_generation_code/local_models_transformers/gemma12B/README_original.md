# Local Gemma 4 12B Inference

Inference-only setup for running `google/gemma-4-12B-it` on the local RTX PRO
6000 Blackwell (96 GB VRAM) and processing prompts in `exp_prompts/`. No
training, fine-tuning, RAG, or agent framework. Same overall pattern as the
`bioLLM` project (model registry -> reusable inference module -> 5-response
batch runner over a prompts folder -> JSONL results), adapted to a single
model.

**Status:** model loads on GPU at bf16 and generates coherent text via
`inference.py`/`run_experiment.py` (verified with `exp_prompts/example.txt`).

## 1. Model

| | Gemma 4 12B |
|---|---|
| Official repo | [`google/gemma-4-12B-it`](https://huggingface.co/google/gemma-4-12B-it) |
| Architecture | `Gemma4UnifiedForConditionalGeneration` -- a "unified" text+vision+audio model; only the **text** path is used here (`AutoModelForCausalLM` covers it without pulling in image/audio processors) |
| Native/released dtype | `bfloat16` (config `torch_dtype`) |
| Format on disk | single `model.safetensors` |
| Disk size | ~24 GB |
| VRAM to load (bf16) | ~24 GB weights (measured: 24.0 GB allocated) |
| Native context length | 262144 tokens (this project caps generation headroom at 32768, see `model_config.py`) |
| Chat template | Yes (`chat_template.jinja`, system/user/assistant roles) |

Model files live on `/mnt/data` (11 TB volume, 8.9 TB free), **not** the root
filesystem (`/`, which was 98% full with only ~49 GB free when this project
was set up):

```
/mnt/data/models/gemma-4-12B-it/
```

This checkpoint was already downloaded there from a prior session -- this
project reuses it rather than re-downloading. If you ever need to
(re-)download it:
```bash
export HF_HUB_ENABLE_HF_TRANSFER=1
hf download google/gemma-4-12B-it --local-dir /mnt/data/models/gemma-4-12B-it
```

## 2. Environment: why this project has its own venv

`google/gemma-4-12B-it`'s `gemma4_unified` architecture requires
`transformers >= 5.10`. The shared miniconda base env (and `bioLLM`'s venv,
which inherits from it via `--system-site-packages`) is pinned to
`transformers 5.2.0` -- deliberately, per `bioLLM/README.md`, for its own
models. Upgrading the shared base env would risk breaking that already-
verified setup.

Instead, `~/gemma12B/.venv` is a `--system-site-packages` venv
(same approach `bioLLM` uses) that inherits `torch 2.10.0+cu128`,
`accelerate`, `safetensors`, `pillow`, `huggingface_hub`, etc. from the base
env, but has its **own local, newer copy** of `transformers` (5.17.0 at
setup time) installed directly into the venv -- this does not touch or
uninstall the base env's `transformers 5.2.0` (verified: `pip install`
reported "Not uninstalling ... outside environment" for the base copies).
`bioLLM` and this project can therefore run side by side without conflict.

Activate it:
```bash
source ~/gemma12B/.venv/bin/activate
```

If you ever need to recreate it:
```bash
python3 -m venv --system-site-packages ~/gemma12B/.venv
source ~/gemma12B/.venv/bin/activate
pip install --upgrade "transformers>=5.10"
```

## 3. Directory layout

```
gemma12B/
├── .venv/                  dedicated environment (see above)
├── exp_prompts/            prompt files (.txt)
├── scripts/
│   ├── model_config.py     model path, dtype, generation defaults
│   ├── inference.py         reusable load/format/generate module + single-prompt CLI test
│   └── run_experiment.py    5-response experiment runner over exp_prompts/
├── results/                 <prompt_stem>.json, one per prompt (never merged)
└── README.md
```

## 4. Commands

Smoke-test one prompt:
```bash
source ~/gemma12B/.venv/bin/activate
cd ~/gemma12B/scripts
python inference.py --prompt-file ../exp_prompts/example.txt
```

Run the full 5-response experiment over every prompt in `exp_prompts/`:
```bash
python run_experiment.py
```
Run one prompt file only with `--prompt-file ../exp_prompts/<name>.txt`.
Each run writes/overwrites `results/<prompt_stem>.json` -- one file per
prompt, never merged together.

## 5. Prompt handling

`inference.parse_prompt_file()` supports two formats:
- **Structured** (matching `bioLLM`'s convention): a file with >=2
  `--- SECTION ---` headers is split into `(system_prompt, user_prompt)` --
  the first section is the system prompt, everything from the second header
  onward (headers included) is the user prompt.
- **Plain**: a file with no section headers is treated as the user prompt
  alone, with no system prompt (`exp_prompts/example.txt` uses this form).

Nothing in `exp_prompts/*.txt` is reworded, summarized, or restructured --
formatting is applied only at the chat-template layer in `build_inputs()`.

## 6. Result schema

Contents of `results/<prompt_stem>.json`:
```jsonc
{
  "prompt_file": "PTMA-Ki-67_Interpretation_Prompt.txt",
  "model_repo": "google/gemma-4-12B-it",
  "system_prompt": "...",
  "user_prompt": "...",
  "prompt_token_length": 1234,
  "generation_settings": {"do_sample": true, "temperature": 1.0, "top_p": 0.95, "top_k": 64, "max_new_tokens": 1024},
  "num_responses": 5,
  "responses": [
    {"run_index": 0, "seed": 1234, "response": "...", "generation_time_sec": 12.3},
    ...
  ]
}
```
`temperature`/`top_p`/`top_k` defaults in `model_config.py` match this
checkpoint's own `generation_config.json` (not arbitrary choices).
