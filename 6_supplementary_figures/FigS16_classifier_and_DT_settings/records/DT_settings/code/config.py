"""Configuration for the DT parameter-sweep evaluation.

Mirrors the default hyperparameters used interactively in app_v4.py
(hidden_size=256, dropout=0.5, lr=2e-4, weight_decay=1e-4, epochs=2000, seed=20),
but is meant to run headlessly (no Streamlit) across multiple datasets and
multiple Decision Tree settings.
"""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(r"C:\Users\twjer\nonlinear-multimodal")
RESULTS_DIR = BASE_DIR / "results"

# --- Datasets to evaluate -------------------------------------------------
# "domain" / "cell_type" fill the same free-text fields the Streamlit app
# asked the user for (Step 3 "Export LLM Interpretation Prompt" form).
DATASETS = [
    {
        "key": "CITEseq_CD45RA_naive",
        "path": DATA_DIR / "CITEseq_1500HVG_CD45RA_naive.xlsx",
        "domain": "Immunology / CITE-seq (surface protein + RNA)",
        "cell_type": "Human PBMC-derived CD45RA+ naive T cells",
    },
    {
        "key": "CosMx_CancerKi67",
        "path": DATA_DIR / "CosMx_breastcancer" / "withoutlier_0409" / "cancer_cells_Ki67_dualgaters_cleaned_500_0409.xlsx",
        "domain": "Oncology / CosMx spatial transcriptomics",
        "cell_type": "Breast cancer cells, Ki-67 dual-gated (proliferation status)",
    },
    {
        "key": "Gouwens_GABAergic",
        "path": DATA_DIR / "GABAergic_dataset" / "Gouwens_SingleCell_Processed.xlsx",
        "domain": "Neuroscience / Patch-seq electrophysiology",
        "cell_type": "Mouse cortical GABAergic interneurons (Allen Institute)",
    },
]

# --- Upstream MLP + IG (identical to app_v4.py defaults) ------------------
MLP_PARAMS = dict(hidden_size=256, dropout=0.5, lr=2e-4, weight_decay=1e-4, epochs=2000, seed=20)
IG_N_STEPS = 50
TEST_SIZE = 0.2

# --- Focused-feature / binary-label settings (app_v4.py Step 3 defaults) --
ATTR_DIRECTION = "Positive"   # first radio option in the app
IG_PERCENTILE = 90            # app default slider value
CLASS_1_WEIGHT = 4            # app default penalty weight

# --- The 4 Decision-Tree settings under test -------------------------------
DT_SETTINGS = [
    {"name": "top30_depth4", "n_features": 30, "max_depth": 4},
    {"name": "top60_depth4", "n_features": 60, "max_depth": 4},
    {"name": "top30_depth6", "n_features": 30, "max_depth": 6},
    {"name": "top60_depth6", "n_features": 60, "max_depth": 6},
]

# Everything except max_depth is held fixed to the app_v4.py values.
# max_leaf_nodes is intentionally NOT capped here: with it fixed at a single
# shared number (app_v4.py used 14), a depth-6 tree can't grow past what a
# depth-4 tree already reaches, defeating the point of testing depth. Leaving
# it unbounded lets max_depth do the regularization (depth4 -> <=16 leaves,
# depth6 -> <=64 leaves), further limited by min_samples_leaf.
DT_FIXED_PARAMS = dict(min_samples_leaf=5, random_state=42)

# --- Node support-threshold for flagging potential overfitting ------------
MIN_SUPPORT_FRACTION = 0.005  # 0.5% of the dataset
MIN_SUPPORT_ABS = 15          # floor, regardless of dataset size
