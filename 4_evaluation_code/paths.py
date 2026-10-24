"""Where the judged data are. The scripts run directly from this repository: the data are in ../5_evaluation_results.

If the data folders (evaluation_results_rubric_v12/, pairwise_similarity_results_v2/) were copied next to the scripts, those are used instead;
the environment variable PITCH_RESULTS_DIR overrides both.
"""
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _results_dir():
    if os.environ.get("PITCH_RESULTS_DIR"):
        return Path(os.environ["PITCH_RESULTS_DIR"]).resolve()
    if (HERE / "evaluation_results_rubric_v12").is_dir():
        return HERE
    return HERE.parent / "5_evaluation_results"


RESULTS_DIR = _results_dir()
