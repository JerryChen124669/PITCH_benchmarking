"""Self-contained copy of the evaluation pieces the single-answer judge needs (no imports from outside this folder).

  judge_evaluation.py  verbatim copy of ClassifierEvaluation/judge_evaluation.py (the 7-pillar rubric JUDGE_SYSTEM_PROMPT, JUDGE_MODEL, load_client).
                       Its sha256 is checked by judge_single_answers_v2.py against the hash of the rubric used in the earlier judging.
  judge_core.py        the small helpers that used to live in model_comparison/code/judge_ablation.py:
                       NO_TREE_PREAMBLE (five-criterion override), final5() (5-pillar total /45), je_strip() (removes the JSON footer).
"""
import importlib.util
import re
from pathlib import Path

_SRC = Path(__file__).resolve().parent / "judge_evaluation.py"
_spec = importlib.util.spec_from_file_location("judge_eval", _SRC)
je = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(je)

NO_TREE_PREAMBLE = """IMPORTANT OVERRIDE: this is a FIVE-criterion evaluation. Do NOT score Zone Identification or Logic Rigor: return null for "Zone Identification score_zone1" and "Logic Rigor_zone1", and do not penalize any tree-rule extraction issues anywhere. (The prompt you see may have given the model a decision tree, only an Integrated-Gradients feature ranking, or only a raw cells-x-features table; grade every case the same way.) Score the other five criteria as specified, treating the submission's single main hypothesis as its 'primary zone'. For Feature Comprehensiveness, the required items are the target, the focus feature, and the (up to) four contextual features the submission itself names as its main contextual features; judge role assignment for those. Judge factual claims against the evidence actually provided in the prompt and established biology. Keep all other output keys and formats exactly as specified.

"""


def final5(sc):
    g = lambda k: sc.get(k) or 0
    return (g("Feature Comprehensiveness_zone1") + g("Biological Plausibility_zone1") + g("Specificity_zone1")
            + g("Grounded Mechanistic Synthesis_zone1") + 0.5 * g("Literature Grounding_zone1"))


def je_strip(t):
    return re.sub(r"```json\s*\{.*?\}\s*```", "", t, flags=re.DOTALL).strip()
