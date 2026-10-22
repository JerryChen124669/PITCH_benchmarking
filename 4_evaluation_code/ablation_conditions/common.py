"""Shared constants and helpers for the ablation-conditions scripts (generation rate, five-criteria quality radars, hypothesis composition agreement).

Folder layout (this repository):
  2_prompts/ablation_conditions/                      the prompts of the three evidence conditions (PITCH, MLP + IG, measurement-only), 3 datasets
  3_answers/ablation_conditions/all_attempts/         every saved model attempt: <model>/<dataset>_<condition>_run<k>.json
  3_answers/ablation_conditions/selected_answers/     the answer selected per model, condition and dataset (+ selection_manifest.json)
  4_evaluation_code/ablation_conditions/              these scripts (+ evaluation/: judge rubric module and rubrics)
  5_evaluation_results/ablation_conditions/           answer_screening.csv and all results (1_success_rate_table/, 2_quality_radar/, 4_hypothesis_composition_agreement/)
Datasets: CITE-seq CD45RA, CosMx Ki-67, Gouwens input resistance (pERK is not part of this evaluation).
"""
import json
import random
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent                     # 4_evaluation_code/ablation_conditions
REPO = HERE.parents[1]
ROOT = REPO / "5_evaluation_results" / "ablation_conditions"   # result folders are created / read here


class _Inputs:
    """INPUTS / "name" -> the folder (or file) of that input in the repository layout; the scripts address inputs through these names."""
    _MAP = {"prompts": REPO / "2_prompts" / "ablation_conditions",
            "answers": REPO / "3_answers" / "ablation_conditions" / "all_attempts",
            "single_answers": REPO / "3_answers" / "ablation_conditions" / "selected_answers",
            "answers_3per": REPO / "3_answers" / "ablation_conditions" / "three_per_prompt",      # derived by 02a_select_answers.py (not stored)
            "answer_screening.csv": ROOT / "answer_screening.csv"}

    def __truediv__(self, name):
        return self._MAP[name]


INPUTS = _Inputs()
sys.path.insert(0, str(HERE / "evaluation"))

GEMINI = "gemini-3.1-pro-preview"
GLM = "zai-org/GLM-5.3"
DEEPSEEK = "deepseek-ai/DeepSeek-V4.1-Flash"
QWEN = "Qwen/Qwen3.8-2.4T-A95B"
MODELS = [GEMINI, GLM, DEEPSEEK, QWEN]
MODEL_SHORT = {GEMINI: "Gemini", GLM: "GLM", DEEPSEEK: "DeepSeek", QWEN: "Qwen"}          # names used in answer_screening.csv
MODEL_LABEL = {GEMINI: "Gemini-3.1-pro-preview", GLM: "GLM-5.3", DEEPSEEK: "DeepSeek V4.1 Flash", QWEN: "Qwen3.8-2.4T-A95B"}
DATASETS = ["CITEseq_CD45RA_naive", "CosMx_CancerKi67", "Gouwens_GABAergic"]
DATASET_LABEL = {"CITEseq_CD45RA_naive": "CITE-seq CD45RA", "CosMx_CancerKi67": "CosMx Ki-67", "Gouwens_GABAergic": "Gouwens input resistance"}
DATASET_MARKER = {"CITEseq_CD45RA_naive": "o", "CosMx_CancerKi67": "s", "Gouwens_GABAergic": "^"}
CONDS = ["cond0_full_pipeline", "cond1_ig_only"]                       # conditions that are evaluated for quality / feature use / hypothesis composition agreement
ALL_CONDS = CONDS + ["cond2_raw_data"]                                  # the success-rate table also reports measurement-only
COND_LABEL = {"cond0_full_pipeline": "PITCH", "cond1_ig_only": "MLP + IG", "cond2_raw_data": "Measurement-only"}
COND_COLOR = {"cond0_full_pipeline": "#2b78d4", "cond1_ig_only": "#ec6a36", "cond2_raw_data": "#1db07a"}
LABELS = "ABCD"


def slug(model):
    return model.replace("/", "__")


def load_case(models, cond, ds):
    """One case = one prompt: the picked answer of every model, as [(model, picked_run, answer text without the JSON footer)]."""
    import judge_core as jc
    out = []
    for m in models:
        files = sorted((INPUTS / "single_answers" / cond / slug(m)).glob(f"{ds}_{cond}_picked_run*.json"))
        if len(files) != 1:
            raise SystemExit(f"expected exactly 1 picked answer for {m} {ds}/{cond}, found {len(files)} -- run 02a_select_answers.py first")
        rec = json.loads(files[0].read_text(encoding="utf-8"))
        out.append((m, int(rec["run"]), jc.je_strip(rec["final_answer"])))
    return out


def shuffled(case, seed, cond, ds, k):
    """Random order of the answers for one case and judging pass (seeded -> reproducible, different in every pass)."""
    order = list(case)
    random.Random(f"{seed}|agreement|{cond}|{ds}|pass{k}").shuffle(order)
    return order


def user_message(order):
    parts = [f"Below are {len(order)} anonymized answers from different language models addressing the same biological case.\n"]
    for i, (_m, _run, text) in enumerate(order):
        parts.append(f"--- ANSWER {LABELS[i]} ---\n{text}\n")
    return "\n".join(parts)


def parse_json(raw):
    m = re.search(r"```json\s*(.*?)\s*```", raw, re.DOTALL)
    s = m.group(1) if m else raw.strip()
    if not s.startswith("{"):
        s = s[s.index("{"): s.rindex("}") + 1]
    return json.loads(s)
