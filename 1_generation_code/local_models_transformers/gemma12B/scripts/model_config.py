"""
Central configuration for local Gemma 4 12B inference.

The model is run at its officially released precision (bfloat16, per
config.json's torch_dtype). See README.md for the disk/VRAM sizing and the
transformers-version requirement that motivated this project's isolated venv.
"""

import torch

MODEL_PATH = "/mnt/data/models/gemma-4-12B-it"
HF_REPO = "google/gemma-4-12B-it"

MODEL_CONFIG = {
    "torch_dtype": torch.bfloat16,
    "effective_context": 32768,  # headroom pick; native max_position_embeddings is 262144
    "default_max_new_tokens": 1024,
}

# Held constant across the repeated runs per prompt. Matches this
# checkpoint's own generation_config.json defaults (temperature/top_p/top_k),
# not an arbitrary choice.
GENERATION_DEFAULTS = {
    "do_sample": True,
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": 64,
    "num_return_sequences": 1,
}

NUM_REPEATS = 5
BASE_SEED = 1234
