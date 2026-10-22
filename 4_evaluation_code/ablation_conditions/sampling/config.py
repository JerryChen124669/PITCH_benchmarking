"""Configuration for the hypothesis-generation ablation study.

Reviewer-requested ablation: does the full MLP -> IG -> Decision Tree ->
LLM pipeline actually need each of its computational stages?

  Condition 1 ("ig_only"):  MLP -> IG -> [skip DT] -> LLM.
      The LLM gets the top-30 IG-ranked features and their mean |IG|
      attribution magnitude only -- no fitted classifier, no per-node
      evidence, no tree/zone structure at all.

  Condition 2 ("raw_data"): [skip IG and DT entirely] -> LLM.
      The LLM gets a downsampled RAW cells x features table (the data as
      it looked before the MLP was ever trained -- last column is the
      target, everything before it is a raw feature), and nothing else.
      This tests whether the LLM's own prior knowledge + in-context
      numerical reasoning can replace the entire computational pipeline.

Both conditions answer the SAME biological question as the full pipeline
(why does the focused feature attribute to the target), for direct
comparability, and both require the LLM to self-report a confidence score
and explicitly flag insufficient information rather than confabulate.
"""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"

# --- Datasets (self-contained copies/derivations under PITCH_Ablation/data) -
DATASETS = [
    {
        "key": "CITEseq_CD45RA_naive",
        "mode": "ig_auto",
        "path": DATA_DIR / "CITEseq_1500HVG_CD45RA_naive.xlsx",
        "domain": "Immunology / CITE-seq (surface protein + RNA)",
        "cell_type": "Human PBMC-derived CD45RA+ naive T cells",
        "n_features": 30,
    },
    {
        "key": "CosMx_CancerKi67",
        "mode": "ig_auto",
        "path": DATA_DIR / "cancer_cells_Ki67_dualgaters_cleaned_500_0409.xlsx",
        "domain": "Oncology / CosMx spatial transcriptomics",
        "cell_type": "Breast cancer cells, Ki-67 dual-gated (proliferation status)",
        "n_features": 30,
    },
    {
        "key": "Gouwens_GABAergic",
        "mode": "ig_auto",
        "path": DATA_DIR / "Gouwens_SingleCell_Processed.xlsx",
        "domain": "Neuroscience / Patch-seq electrophysiology",
        "cell_type": "Mouse cortical GABAergic interneurons (Allen Institute)",
        "n_features": 30,
    },
    {
        "key": "pERK_EGF10",
        "mode": "manual",
        "raw_table_npz": DATA_DIR / "pERK_EGF10_raw_table.npz",
        "domain": "Signal Transduction, perturbation experiment with 10 ng/mL EGF",
        "cell_type": "Human Breast Epithelial",
        "feature_idx": 75,  # "Cytoplasm Intensity mean Actin" -- focused/target feature
        "use_list": [0, 1, 7, 13, 14, 20, 25, 26, 32, 37, 38, 44, 50, 51, 57, 63, 67, 75, 101, 102, 110, 134, 151],
        "target_name": "pERK",
        "n_features": 30,  # unused for manual mode's cond-1 (fixed 22-feature panel); kept for symmetry
    },
]

# --- Upstream MLP + IG (identical to ClassifierEvaluation / DTEvaluation) --
MLP_PARAMS = dict(hidden_size=256, dropout=0.5, lr=2e-4, weight_decay=1e-4, epochs=2000, seed=20)
IG_N_STEPS = 50
TEST_SIZE = 0.2
ATTR_DIRECTION = "Positive"
IG_PERCENTILE = 90

# --- Condition 2: raw-data token-budget policy -----------------------------
# Empirically measured via client.models.count_tokens on gemini-3.1-pro-preview
# (input_token_limit = 1,048,576). We reserve 50% of that hard ceiling for the
# data table itself -- leaves headroom for instructions, and stays clear of
# the well-documented "long-context reasoning degrades before the hard limit"
# risk for tasks that require synthesizing across ALL provided rows (this is
# not a needle-in-haystack retrieval task). The exact row count is then
# verified empirically per-dataset via count_tokens on the real assembled
# table (see raw_data_sampler.py) and trimmed if it overshoots -- the numbers
# below are starting estimates, not the final counts actually used.
GEMINI_INPUT_TOKEN_LIMIT = 1_048_576
RAW_DATA_TOKEN_BUDGET = int(GEMINI_INPUT_TOKEN_LIMIT * 0.5)  # 524,288 tokens
RAW_DATA_STRATIFY_BINS = 10  # decile-stratified sampling by target value

MODEL_ID = "gemini-3.1-pro-preview"
