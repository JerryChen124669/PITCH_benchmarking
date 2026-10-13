# Local Biomedical LLM Inference: PMC-LLaMA 13B & BioMistral 7B

Inference-only setup for running the two models on the local RTX PRO 6000
Blackwell (96 GB VRAM) and processing the biological reasoning prompts in
`exp_prompts/`. No training, fine-tuning, RAG, or agent framework.

**Status:** both models download, load on GPU at full precision, and produce
coherent, on-topic biological reasoning that commits to one primary + one
secondary zone with mechanistic rationale (see section 8 for how that
required per-model prompt engineering, not just a single shared
instruction). The full 5-response experiment (`run_experiment.py`, default
settings, 1024-token budget) has been run for **both models over all four
`exp_prompts/` files** — results are in `results/biomistral/*.jsonl` and
`results/pmc_llama/*.jsonl`. Re-run the commands in section 6 any time to
regenerate (each run overwrites the matching output file).

Known residual limitations, not fixable by prompting alone (see section 8
for what was tried): (1) both models occasionally fabricate plausible-looking
but unverifiable gene names or citations -- expected for a 7-13B model
reasoning without retrieval (RAG is out of scope per the original ask); (2)
PMC-LLaMA sometimes continues past the requested primary+secondary zone
into a third or fourth zone -- its older, non-chat-tuned instruction
following is weaker than BioMistral's at strictly stopping. Both are worth
weighing when comparing the two models' output quality, not just their
biological content.

## 1. Models

| | PMC-LLaMA 13B | BioMistral 7B |
|---|---|---|
| Official repo | [`axiong/PMC_LLaMA_13B`](https://huggingface.co/axiong/PMC_LLaMA_13B) | [`BioMistral/BioMistral-7B`](https://huggingface.co/BioMistral/BioMistral-7B) |
| Architecture | LLaMA-1, 13B | Mistral-7B-Instruct-v0.1 base |
| Native/released dtype | `float32` (config `torch_dtype`) | `bfloat16` (config `torch_dtype`) |
| Format on disk | 6x `pytorch_model-*.bin` shards | single `pytorch_model.bin` |
| Disk size | ~52 GB | ~14.5 GB |
| VRAM to load (native dtype) | ~52 GB weights | ~15 GB weights |
| Native context length | **2048 tokens** | 32768 tokens (sliding window 4096) |
| Chat template | None (base/completion-style, instruction-tuned) | Yes (`[INST] ... [/INST]`, user/assistant only) |

Both are run at their **officially released full precision** (no int8/int4
quantization) — `float32` for PMC-LLaMA, `bfloat16` for BioMistral, since
that is the precision each was actually released and evaluated in. Loading
PMC-LLaMA's `float32` weights upcast to nothing-gained-fp64, or downcasting
BioMistral's native `bfloat16` to `float32`, would not add precision, only
memory pressure, so neither is done.

Models are stored locally under `/mnt/data/models/` (the workstation's large
data volume — the root filesystem only had ~29 GB free, not enough for both
models):

```
/mnt/data/models/PMC_LLaMA_13B/
/mnt/data/models/BioMistral-7B/
```

## 2. Known limitation: PMC-LLaMA's 2048-token context

PMC-LLaMA 13B is a LLaMA-1 era model configured with
`max_position_embeddings: 2048`. Tokenized with its own tokenizer, the
`exp_prompts/` files are already **1,931–2,768 tokens each** — i.e. every
prompt exceeds or nearly exhausts the native context on its own, before any
output tokens.

Per your decision, this is handled with **RoPE linear scaling**
(`rope_parameters={"rope_type": "linear", "factor": 2.0}`, `max_position_embeddings`
raised to 4096) applied at model-load time in `scripts/model_config.py`.
This is a config-only change — no fine-tuning, no rewriting of your prompt
text. It lets the full prompt survive intact with headroom for generation.

**Caveat:** PMC-LLaMA was never trained or evaluated at positions beyond
2048. Its behavior on the extended range is unverified — treat PMC-LLaMA
outputs on these long prompts as experimental, and weigh that when comparing
its reasoning quality against BioMistral (which handles the same prompts
natively, no extension needed).

## 3. Tokenizer gotcha (PMC-LLaMA)

`transformers 5.2.0`'s `AutoTokenizer` silently mis-tokenizes PMC-LLaMA:
it auto-converts the repo's raw sentencepiece `tokenizer.model` into a
broken fast tokenizer that mis-segments ordinary words (e.g. "Return" →
`Ret`/`ur`/`nt`), even though `tokenizer_config.json` explicitly declares
`"tokenizer_class": "LlamaTokenizer", "use_fast": false`. `inference.py`
works around this by loading `LlamaTokenizer` directly for PMC-LLaMA
(`architecture == "llama"`) instead of going through `AutoTokenizer`. If
you ever swap in a different LLaMA-family checkpoint that only ships a raw
`tokenizer.model` (no `tokenizer.json`), re-check this before trusting its
output — garbled, over-segmented text is the symptom. BioMistral is
unaffected since its repo ships a native `tokenizer.json`.

## 4. Environment

- Ubuntu 24.04, driver 595.84 (CUDA 13.2 runtime), Python via
  `~/miniconda3` (base env: `torch 2.10.0+cu128`, `transformers
  5.2.0`, `accelerate`, `safetensors`, `sentencepiece`, `huggingface_hub`
  already present and current).
- **Dedicated environment**: `~/bioLLM/.venv`, a venv created
  with `--system-site-packages` on top of the miniconda base. Rationale: the
  base env already has a fully compatible, up-to-date stack, and the root
  filesystem had only ~29 GB free — cloning a full separate torch/CUDA
  install (other project envs here run 5–8 GB each) risked filling the disk
  for no benefit. The venv still gives this project an isolated,
  independently-activatable environment for any bioLLM-specific packages,
  without duplicating what's already installed.

Activate it:
```bash
source ~/bioLLM/.venv/bin/activate
```

## 5. Directory layout

```
bioLLM/
├── .venv/                  dedicated environment (see above)
├── exp_prompts/            your existing prompt files (untouched)
├── scripts/
│   ├── model_config.py     model registry: paths, dtype, rope scaling, gen defaults
│   ├── inference.py         reusable load/format/generate module + single-prompt CLI test
│   └── run_experiment.py    5-response experiment runner over exp_prompts/
├── results/
│   ├── biomistral/<prompt_stem>.jsonl
│   └── pmc_llama/<prompt_stem>.jsonl
└── README.md
```

## 6. Commands

Download (already run once; re-running is a no-op / resumes):
```bash
source ~/bioLLM/.venv/bin/activate
export HF_HUB_ENABLE_HF_TRANSFER=1
hf download BioMistral/BioMistral-7B --local-dir /mnt/data/models/BioMistral-7B
hf download axiong/PMC_LLaMA_13B     --local-dir /mnt/data/models/PMC_LLaMA_13B
```

Smoke-test one model on one prompt:
```bash
cd ~/bioLLM/scripts
python inference.py --model biomistral --prompt-file ../exp_prompts/PTMA-Ki-67_Interpretation_Prompt.txt
python inference.py --model pmc_llama  --prompt-file ../exp_prompts/PTMA-Ki-67_Interpretation_Prompt.txt
```

Run the full 5-response experiment over every prompt in `exp_prompts/`:
```bash
cd ~/bioLLM/scripts
python run_experiment.py --model biomistral
python run_experiment.py --model pmc_llama
```
This writes one JSONL file per prompt to `results/<model>/<prompt_stem>.jsonl`.
Run one prompt file only with `--prompt-file ../exp_prompts/<name>.txt`.

Only one model is loaded into GPU memory at a time (each `python
run_experiment.py` invocation is single-model), so the two runs above should
be run one after another, not concurrently, in a given generation session —
though VRAM headroom (96 GB vs. ~52 GB for PMC-LLaMA, ~15 GB for BioMistral)
would technically allow both loaded simultaneously if that becomes useful
later.

## 7. Prompt handling

Each `exp_prompts/*.txt` file already contains a fixed structure:
```
--- SYSTEM PROMPT ---
<system prompt text>
--- <next section: EXPERIMENT CONTEXT & DATA PIPELINE, etc.> ---
...
--- TASK ---
...
```
`inference.parse_prompt_file()` splits this into `(system_prompt,
user_prompt)` by locating the section headers — `system_prompt` is the text
of the first section, `user_prompt` is everything from the second header
onward, verbatim (headers included, since they are part of the task
content). Nothing is reworded, summarized, or restructured.

Model-specific formatting is applied only at the chat/prompt-template layer,
never to the biological content:
- **BioMistral**: its chat template supports only `user`/`assistant` roles
  (no `system` role), so `system_prompt` is folded into the single user turn
  — the standard adaptation used for Mistral-Instruct-style models —
  followed by `tokenizer.apply_chat_template(..., add_generation_prompt=True)`.
- **PMC-LLaMA**: has no chat template (base/completion-style model). System
  and user text are concatenated into one continuous document and tokenized
  with `add_special_tokens=False`, matching the usage shown in the model's
  own README.

## 8. Prompt engineering: stopping the models from echoing the input

**Symptom observed** (in the first completed runs): both models spent most
of their output restating facts already present in `exp_prompts/*.txt` --
the zone definitions, Precision/TP/FP numbers, the raw decision tree, the
classifier's training setup -- rather than producing the biological
hypothesis the TASK section actually asks for. BioMistral additionally gave
only 1-3 sentence answers naming a zone with no reasoning; PMC-LLaMA listed
every zone (up to 7) instead of committing to the requested primary +
secondary pair.

**Fix, applied only at the model-input layer in `scripts/model_config.py` /
`inference.py` -- `exp_prompts/*.txt` is never modified:** a response-format
directive is appended to the user prompt (after your biological content,
so it has the highest-recency effect on generation), explicitly forbidding
restatement of the input and requiring commitment to exactly one primary +
one secondary zone with mechanistic rationale, literature-confirmed
findings, and new testable mechanisms. Two different variants are used, not
one -- this took several rounds of failed attempts to get right:

- **BioMistral** (chat/instruction-tuned) gets a closed directive: "output
  ONLY this structure" with an explicit skeleton
  (`RESPONSE_FORMAT_DIRECTIVE_CHAT`). It follows this well.
- **PMC-LLaMA** (completion-style, not chat-tuned) got the *same* closed
  directive first, and it broke badly: verified via direct ablation, the
  model responded to the closed "--- OUTPUT INSTRUCTIONS ---" block by
  writing `"---END INSTRUCTIONS ---"` and stopping almost immediately --
  it read the block as a document-closing marker, not a command to follow.
  Forcing it past that stop with `min_new_tokens` didn't help; with no
  relevant context left to condition on, it derailed into unrelated
  memorized pretraining text (a Reuters Health article, a Unisys hiring
  press release, a book review -- verbatim-looking web content with no
  connection to the prompt). The fix was to stop instructing it and start
  *priming* it: `RESPONSE_FORMAT_DIRECTIVE_COMPLETION` ends the prompt
  mid-sentence at `"PRIMARY ZONE:"`, so the model's job becomes "predict
  what comes next" -- the way completion models actually work -- instead of
  "obey a standalone command." That fixed the derailment completely. The
  literal prefix text is re-attached to the stored response afterward
  (`response_prefix` in `MODEL_REGISTRY`) purely for readability, since it
  was part of the priming prompt, not generated content.

Generation-parameter changes, also per-model (see `MODEL_REGISTRY` in
`model_config.py`):
- `repetition_penalty` / `no_repeat_ngram_size`: added after BioMistral fell
  into literal repetition loops on some seeds (e.g. re-emitting "the
  regulation of neuronal excitability by X, Y, Z" with one more gene
  appended each time, dozens of times). BioMistral tolerates a real value
  here (1.15 / 4). The *same* value caused PMC-LLaMA's derailment above --
  it is far more brittle to logit-level penalties, being smaller, older,
  and already running with a RoPE-extended context -- so it gets a much
  milder setting (1.05 / 3).
- `min_new_tokens`: BioMistral was stopping after one or two sentences
  identifying a zone with no reasoning; a floor of 400 tokens (combined with
  the "both sections are mandatory" line in the directive) fixes this.
  PMC-LLaMA needs a much smaller floor (50) -- the completion-style priming
  cue does the real work of keeping it on-topic and writing at length;
  forcing length directly is what caused its derailment in the first place.

**What this did and didn't fix**, based on the corrected full run in
`results/`: both models now produce on-topic, mechanistic, single-primary-
plus-single-secondary-zone reasoning with attempted literature citations,
and response lengths for BioMistral moved from ~100-650 characters to a
consistent ~1400-3000 characters. Two things prompting alone can't fully
solve, flagged rather than hidden: fabricated-sounding citations/gene names
(needs retrieval to fix, out of scope here) and PMC-LLaMA occasionally
continuing into a third zone despite the directive (a real instruction-
following ceiling for this model, not a settings bug).

## 9. Five-response experiment design

For each prompt, `run_experiment.py`:
1. Loads the model once.
2. Builds the model-formatted input from `(system_prompt, user_prompt)`.
3. Generates 5 times with **identical sampling settings** (`temperature
   0.7`, `top_p 0.9`, `do_sample=True`) and only the random seed varying
   (`1234, 1235, 1236, 1237, 1238`) — so the 5 responses are independent
   samples under the same decoding policy, not just one call repeated.
4. Writes one JSON line per prompt file to `results/<model>/<prompt_stem>.jsonl`
   containing:

```jsonc
{
  "prompt_file": "PTMA-Ki-67_Interpretation_Prompt.txt",
  "model_key": "biomistral",
  "model_repo": "BioMistral/BioMistral-7B",
  "system_prompt": "...",
  "user_prompt": "...",
  "prompt_token_length": 1931,
  "generation_settings": {"do_sample": true, "temperature": 0.7, "top_p": 0.9, "max_new_tokens": 1024},
  "num_responses": 5,
  "responses": [
    {"run_index": 0, "seed": 1234, "response": "...", "generation_time_sec": 12.3},
    {"run_index": 1, "seed": 1235, "response": "...", "generation_time_sec": 11.8},
    ...
  ]
}
```

This makes it straightforward to identify, per record: which prompt, which
model, the 5 individual responses, and the generation settings used —
matching the reference workflow (construct prompt → generate 5x → collect →
save structured JSON), adapted to local HF `transformers` inference instead
of an API-based generation loop.

No evaluation/scoring logic is included yet, by design — that's a separate
next step once both models are confirmed to run reliably end-to-end.

## 10. Model comparison

Because prompt parsing and the 5-response loop are model-agnostic
(`run_experiment.py --model {pmc_llama,biomistral}`), the exact same
`exp_prompts/*.txt` files feed both models unchanged, and results land in
parallel `results/pmc_llama/` and `results/biomistral/` trees for later
side-by-side comparison.
