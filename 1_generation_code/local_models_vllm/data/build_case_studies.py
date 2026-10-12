"""Pair the 6 case-study prompts (case_study_prompts/*.txt) with their 5 reference answers
(case_study_outputs/*_results.jsonl) into case_studies.jsonl, one row per case:
{key, dataset, system_prompt, user_prompt, reference_answers}.

Each .txt is "--- SYSTEM PROMPT ---" + system text, then the user text from
"--- EXPERIMENT CONTEXT & DATA PIPELINE ---" onward. CRLF is normalized to LF.
"""
import glob
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
USER_START = "--- EXPERIMENT CONTEXT & DATA PIPELINE ---"
DATASETS = {  # prompt stem -> source dataset in the manuscript
    "pERK_EGF10_DT_prompt": "EGF stimulation imaging, human breast epithelial (Kramer 2022)",
    "Fxyd6-input resistance_Interpretation_Prompt": "Patch-seq cortical interneurons (Gouwens 2020)",
    "PTMA-Ki-67_Interpretation_Prompt": "CosMx breast cancer spatial transcriptomics (He 2022)",
    "HUMAN_GNLY-CD45RA_Interpretation_Prompt": "CITE-seq cord blood mononuclear cells (Stoeckius 2017)",
    "EGR1-target_pRPS6_Interpretation_Prompt_80_4": "Phospho-seq brain organoids (Blair 2025)",
    "IL2RA-target_CCR5_Interpretation_Prompt_80_4": "CITE-seq HCC Tregs (Huang 2025)",
}

rows = []
for path in sorted(glob.glob(os.path.join(HERE, "case_study_prompts", "*.txt"))):
    stem = os.path.basename(path)[:-4]
    text = open(path, newline="").read().replace("\r\n", "\n").strip()
    head, _, rest = text.partition("--- SYSTEM PROMPT ---")
    assert not head.strip() and rest, path
    system, sep, user = rest.partition(USER_START)
    assert sep, path
    answers = [json.loads(l) for l in open(os.path.join(HERE, "case_study_outputs", f"{stem}_results.jsonl"))]
    assert all(a["id"] == stem for a in answers), path
    rows.append({
        "key": stem,
        "dataset": DATASETS[stem],
        "system_prompt": system.strip(),
        "user_prompt": (USER_START + user).strip(),
        "reference_answers": [a["final_answer"] for a in sorted(answers, key=lambda a: int(a["run"]))],
    })

out = os.path.join(HERE, "case_studies.jsonl")
with open(out, "w") as fh:
    for r in rows:
        fh.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"{len(rows)} cases -> {out}")
