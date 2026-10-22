"""MLP -> Integrated Gradients pipeline, shared by both ablation conditions.
Copied from ClassifierEvaluation/pipeline.py (kept independent per this
repo's convention of not cross-importing between study folders) minus the
classifier-fitting code, which this ablation deliberately never uses.
"""
import numpy as np
import openpyxl
import torch
import torch.nn as nn
import torch.optim as optim
from captum.attr import IntegratedGradients
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split

import config


class MLPRegression(nn.Module):
    def __init__(self, input_size, hidden_size, dropout):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        self.fc2 = nn.Linear(hidden_size, 1)
        self.initialize()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x.squeeze()

    def initialize(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.constant_(self.fc1.bias, 0.0)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.constant_(self.fc2.bias, 0.0)


def load_data(path):
    """ig_auto datasets: raw xlsx, columns[:-1] = features, last column =
    target -- this IS the "data before the MLP" for condition 2's raw table.
    Uses openpyxl's read_only streaming reader rather than pandas.read_excel
    -- measurably faster and more reliable on this machine for these
    large (47-175MB) files."""
    wb = openpyxl.load_workbook(str(path), read_only=True, data_only=True)
    ws = wb.active
    rows = list(ws.iter_rows(values_only=True))
    wb.close()
    header_row = np.array(rows[0])
    data = np.array(rows[1:], dtype=np.float32)
    headers = header_row[:-1]
    target_name = header_row[-1]
    X_raw = data[:, :-1]
    y_raw = data[:, -1]
    return X_raw, y_raw, headers, target_name


def load_pERK_raw_table(ds_meta):
    """manual dataset: precomputed derived .npz (see PITCH_Ablation/data and
    scripts/make_perk_npz.py) -- the exact 20,000 x 510 pre-MLP table, the
    pERK continuous target, AND the precomputed per-cell IG attribution
    matrix (from the upstream MultiMLP model -- this dataset never trains
    its own MLP in this project, see get_ig.py / classification.py). Returns
    attr too so condition 1 can rank by mean |IG| without needing to retrain
    anything, and so both conditions ask about the exact same 20,000 cells."""
    npz = np.load(str(ds_meta["raw_table_npz"]), allow_pickle=True)
    X_raw = npz["X"].astype(np.float32)
    y_raw = npz["y"].astype(np.float32)
    attr = npz["attr"].astype(np.float32)
    headers = npz["feature_names"]
    target_name = str(npz["target_name"])
    return X_raw, y_raw, headers, target_name, attr


def train_mlp(X_raw, y_raw, params=config.MLP_PARAMS, test_size=config.TEST_SIZE):
    seed = params["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)

    X_train, X_test, y_train, y_test = train_test_split(
        X_raw, y_raw, test_size=test_size, random_state=seed
    )

    model = MLPRegression(X_raw.shape[1], params["hidden_size"], params["dropout"])
    optimizer = optim.Adam(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])
    criterion = nn.MSELoss()

    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.float32)

    model.train()
    for _ in range(params["epochs"] + 1):
        optimizer.zero_grad()
        preds = model(X_t)
        loss = criterion(preds, y_t)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        test_preds = model(torch.tensor(X_test, dtype=torch.float32)).numpy()
    corr, _ = pearsonr(y_test, test_preds)
    return model, corr


def compute_ig(model, X_raw, n_steps=config.IG_N_STEPS):
    model.eval()
    ig = IntegratedGradients(model, multiply_by_inputs=False)
    W_tensor = ig.attribute(torch.from_numpy(X_raw), n_steps=n_steps)
    W = W_tensor.detach().numpy()
    attr = X_raw * W
    return attr


def pick_focused_feature(attr, headers, manual_idx=None):
    """Top-ranked feature by mean |IG attribution| -- same "focused feature"
    definition used throughout this project. manual_idx overrides it for the
    pERK dataset (feature_idx=75, matching classification.py / the earlier
    ClassifierEvaluation study), so all conditions ask about the SAME
    biological question."""
    avg_abs = np.mean(np.abs(attr), axis=0)
    ranking = np.argsort(avg_abs)[::-1]
    if manual_idx is not None:
        target_prot = headers[manual_idx]
    else:
        target_prot = headers[ranking[0]]
    return ranking, avg_abs, target_prot


def select_attributing_features(ranking, avg_abs, headers, target_prot, n_features):
    """Returns [(feature_name, mean_abs_ig), ...] for condition 1's evidence,
    top-N by mean |IG| attribution, target feature excluded."""
    out = []
    for idx in ranking:
        if len(out) >= n_features:
            break
        name = headers[idx]
        if name == target_prot:
            continue
        out.append((name, float(avg_abs[idx])))
    return out


def correlation_direction(X_raw, y_raw, headers, target_prot):
    idx = np.where(headers == target_prot)[0][0]
    r, _ = pearsonr(X_raw[:, idx], y_raw)
    return "Direct" if r >= 0 else "Inverse", r
