"""Configuration for the classifier-comparison evaluation.

Sweeps model TYPE (Logistic Regression / Decision Tree / Random Forest)
across 4 datasets = 12 runs, instead of DTEvaluation's sweep of DT
hyperparameters (top-K features x max_depth) on 3 datasets. Reuses
DTEvaluation's MLP+IG training, decision-tree fitting, and Gemini
hypothesis/evaluation machinery wherever the logic is shared.

3 of the datasets pick their attributing features automatically from IG
ranking (mode="ig_auto", identical to DTEvaluation). The 4th (pERK/EGF)
was NOT part of the DTEvaluation batch: it reuses precomputed IG
attributions from an upstream MultiMLP model (attributions/attr_3-10-...npy)
and a manually curated feature panel (mode="manual"), mirroring
classification.py's approach exactly (feature_idx=75 = "Cytoplasm
Intensity mean Actin" as the focused/target feature, self-excluded from
the 22 remaining attributing features; response_idx=3 = pERK;
concentrate_idx=3 = 10 ng/mL EGF).
"""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(r"C:\Users\twjer\nonlinear-multimodal")
RESULTS_DIR = BASE_DIR / "results"

# --- Datasets to evaluate -------------------------------------------------
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
        "path": DATA_DIR / "CosMx_breastcancer" / "withoutlier_0409" / "cancer_cells_Ki67_dualgaters_cleaned_500_0409.xlsx",
        "domain": "Oncology / CosMx spatial transcriptomics",
        "cell_type": "Breast cancer cells, Ki-67 dual-gated (proliferation status)",
        "n_features": 30,
    },
    {
        "key": "Gouwens_GABAergic",
        "mode": "ig_auto",
        "path": DATA_DIR / "GABAergic_dataset" / "Gouwens_SingleCell_Processed.xlsx",
        "domain": "Neuroscience / Patch-seq electrophysiology",
        "cell_type": "Mouse cortical GABAergic interneurons (Allen Institute)",
        "n_features": 30,
    },
    {
        "key": "pERK_EGF10",
        "mode": "manual",
        "domain": "Signal Transduction, perturbation experiment with 10 ng/mL EGF",
        "cell_type": "Human Breast Epithelial",
        # --- manual feature-selection parameters (mirrors classification.py) ---
        "mat_path": DATA_DIR / "Data_1.mat",
        "feature_data_path": DATA_DIR / "FeatureData-z-510.npy",
        "feature_name_path": DATA_DIR / "feature_name.txt",
        "attr_dir": DATA_DIR / "attributions",
        "response_idx": 3,          # pERK (verified against ResponseHeader / Data_1.mat)
        "concentrate_idx": 3,       # concentration[3] = 10 ng/mL EGF
        "feature_idx": 75,          # "Cytoplasm Intensity mean Actin" -- the focused/target feature
        "self_exclude": True,
        "linear_index_keep": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 14, 15, 16, 17, 18, 19, 20],
        "datanum_by_conc": [[13, 14, 15], [16, 17, 18], [10, 19, 20], [7, 8, 9], [4, 5, 6], [1, 2, 3]],
        "concentration_values": [0, 1, 6.25, 10, 25, 100],
        "use_list": [0, 1, 7, 13, 14, 20, 25, 26, 32, 37, 38, 44, 50, 51, 57, 63, 67, 75, 101, 102, 110, 134, 151],
        "target_name": "pERK",
    },
]

# --- Upstream MLP + IG (identical to app_v4.py / DTEvaluation defaults) ---
# Only used for mode="ig_auto" datasets. The mode="manual" dataset reuses a
# precomputed IG attribution matrix from a separate upstream MultiMLP model
# (see get_ig.py), so no MLP is retrained for it here.
MLP_PARAMS = dict(hidden_size=256, dropout=0.5, lr=2e-4, weight_decay=1e-4, epochs=2000, seed=20)
IG_N_STEPS = 50
TEST_SIZE = 0.2

# --- Focused-feature / binary-label settings (app_v4.py Step 3 defaults) --
ATTR_DIRECTION = "Positive"
IG_PERCENTILE = 90
CLASS_1_WEIGHT = 4

# --- The 3 classifier types under comparison -------------------------------
MODEL_SETTINGS = [
    {"key": "logreg", "label": "Logistic Regression"},
    {"key": "dt", "label": "Decision Tree"},
    {"key": "rf", "label": "Random Forest"},
]

# Decision Tree: single canonical setting (comparable in feature-set scale
# to the pERK dataset's fixed 22-feature manual panel), instead of
# DTEvaluation's 4-way top-K/depth sweep -- the sweep axis here is model
# type, not DT hyperparameters.
DT_MAX_DEPTH = 4
DT_FIXED_PARAMS = dict(min_samples_leaf=5, random_state=42)

# Random Forest: capped per-tree depth to keep model "complexity budget"
# comparable to the single Decision Tree above.
RF_N_ESTIMATORS = 200
RF_MAX_DEPTH = 4
RF_FIXED_PARAMS = dict(min_samples_leaf=5, random_state=42, n_jobs=-1)

# Logistic Regression: features standardized before fitting so coefficient
# magnitudes are comparable across features of different raw scale.
LOGREG_FIXED_PARAMS = dict(max_iter=2000, random_state=42)

# --- Node/feature support-threshold for flagging potential overfitting ----
MIN_SUPPORT_FRACTION = 0.005  # 0.5% of the dataset
MIN_SUPPORT_ABS = 15          # floor, regardless of dataset size
