"""Pairwise cross-setting similarity assessment, adapted from
ClassifierEvaluation/pairwise_similarity_v2.py (which compares different LLM
*models* answering the same prompt). Here the axis being compared is the 4 DT
settings (top30/top60 features x depth4/depth6) for the SAME dataset, using
each setting's primary-only hypothesis.

Only 3 of that script's 6 similarity criteria are kept, per the user's
request:
  - Feature Interpretation & Coverage Similarity  ("feature coverage explanation")
  - Mechanistic Hypothesis Similarity              ("primary zone mechanistic similarity")
  - Functional/Causal-Role Similarity               ("feature functional role comparison")
Dropped: Zone & Feature Selection Similarity (structural tree-path match --
not needed since these are already same-tree-family comparisons), Supporting
Reasoning & Experimental Implication Similarity, Response Scope & Elaboration
Similarity. Rubric text for the 3 kept criteria is otherwise unchanged from
the source (the "primary zone" scope note is now redundant but harmless,
since these hypotheses only ever discuss one zone).

This is a similarity metric, not a correctness metric -- judge_evaluation.py
already grades each setting's answer independently. Two settings' answers can
score high here while both being biologically wrong.

Judge model swapped from the source's OpenAI to Gemini 3.1 Pro, matching
every other script in DTEvaluation/.

Usage:
    python pairwise_similarity.py [--overwrite]
"""
import argparse
import json
import random
import re
import time

from google import genai
from google.genai import types

import config
from evaluate_reliability import API_KEY, MODEL_ID, SLEEP_BETWEEN_CALLS

OUT_DIR = config.RESULTS_DIR / "pairwise_similarity"

CRITERIA = [
    "Feature Interpretation & Coverage Similarity",
    "Mechanistic Hypothesis Similarity",
    "Functional/Causal-Role Similarity",
]

PAIRWISE_SYSTEM_PROMPT = """You are an expert computational biologist evaluating the pairwise similarity between two LLM-generated interpretations of the same decision-tree result, run under two different tree-building settings (different candidate-gene-pool size and/or max depth).

Response A and Response B are presented in random order. Grade blind and do not infer or mention which setting produced which response.

SCOPE

Each response discusses only its primary zone (the zone it identified as most important). Evaluate that.

GENERAL SCORING RULES

- Evaluate similarity, not correctness or quality. Two responses may be similar even if both are biologically wrong.
- Treat genes, thresholds, branch conditions, and other information supplied by the decision tree as shared input. Their presence alone is not interpretive similarity.
- Award high scores only for comparable interpretive choices: what the features mean, how they are integrated, which causal roles they occupy, and how the evidence supports the conclusion.
- Score each criterion independently. Agreement on one criterion must not automatically raise another.
- Compare each claim once and do not reward repetition, answer length, or shared formatting.
- Use integer scores from 0 to 10.

Use these general anchors unless a criterion specifies a stricter ceiling:

- 9-10: Nearly equivalent in the substance evaluated by that criterion.
- 7-8: Strongly similar, with at least one meaningful difference.
- 4-6: Partially similar, with important agreements and important differences.
- 1-3: Only limited, generic, or prompt-inherited overlap.
- 0: No meaningful similarity or directly opposing interpretations.

1. FEATURE INTERPRETATION & COVERAGE SIMILARITY

Compare the context-specific biological interpretations assigned to the focus feature, target, and important condition features, including their functions, directions of effect, and relative emphasis.

- 9-10: Nearly all important features receive the same context-specific interpretations and similar emphasis.
- 7-8: Most important interpretations agree, with one meaningful difference or omission.
- 4-6: Some important interpretations agree, while others differ in function, direction, context, or emphasis.
- 1-3: Overlap is mostly shared feature names, generic background, or one isolated interpretation.
- 0: No substantive agreement or predominantly conflicting interpretations.

Shared feature coverage without shared context-specific interpretation cannot score above 3.

2. MECHANISTIC HYPOTHESIS SIMILARITY

Compare the composed, integrated mechanism rather than individual feature descriptions. Evaluate the central mechanistic conclusion, the major components and intermediate processes, their directed relationships, and how the complete mechanism produces the target outcome.

- 9-10: Essentially the same integrated mechanism, causal architecture, and target explanation.
- 7-8: The same mechanistic backbone and conclusion, with one meaningful difference in an intermediate step, direction, or emphasis.
- 4-6: A related biological theme or conclusion, but substantially different integration or causal architecture.
- 1-3: Only individual feature roles, isolated links, or pathway names overlap; the composed mechanisms differ.
- 0: Fundamentally different, opposing, or absent central mechanisms.

Shared feature functions or pathway vocabulary without a similar integrated mechanism cannot score above 3. Reaching the same target outcome through substantially different mechanisms normally scores 4 or lower.

3. FUNCTIONAL/CAUSAL-ROLE SIMILARITY

Compare how the major components are positioned within the overall hypothesis -- for example, as causal drivers, upstream regulators, mediators, permissive contexts, downstream consequences, identity markers, proxies, or correlated readouts. Also compare the strength and certainty of the causal claims.

- 9-10: The major components occupy essentially the same functional and causal positions with similar causal strength.
- 7-8: The overall causal-role structure agrees, with one meaningful difference in role, directness, importance, or certainty.
- 4-6: Some major roles align, but one or more important components are organized differently.
- 1-3: Agreement is limited to generic labels or isolated roles; the overall causal-role structures differ.
- 0: The central role assignments are opposing, unrelated, or undeveloped.

Matching generic labels such as "regulator," "marker," or "driver" without the same role in the integrated hypothesis cannot score above 3.

FINAL CALIBRATION

For every score above 6, verify that the similarity comes from substantive interpretation rather than shared decision-tree input, terminology, or response structure. Apply all stated ceilings before producing the output.

OUTPUT FORMAT

Output only the three integer scores in a markdown JSON block using exactly these keys:

```json
{
  "Feature Interpretation & Coverage Similarity": 0,
  "Mechanistic Hypothesis Similarity": 0,
  "Functional/Causal-Role Similarity": 0
}
```
"""


def parse_judge_json(text):
    match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    raw = match.group(1) if match else text.strip().strip("`")
    return json.loads(raw)


def load_answer(dataset_key, setting_name):
    hyp_path = config.RESULTS_DIR / dataset_key / f"{dataset_key}_{setting_name}_primaryonly_hypothesis.jsonl"
    if not hyp_path.exists():
        return None
    with open(hyp_path, "r", encoding="utf-8") as f:
        return json.loads(f.readline())["final_answer"]


def compare_pair(client, answer_x, answer_y):
    pair = [answer_x, answer_y]
    order = [0, 1]
    random.shuffle(order)  # avoid position bias; scoring is symmetric so no unshuffling needed
    user_content = f"--- RESPONSE A ---\n{pair[order[0]]}\n\n--- RESPONSE B ---\n{pair[order[1]]}"
    pairwise_config = types.GenerateContentConfig(system_instruction=PAIRWISE_SYSTEM_PROMPT, temperature=0.1)
    response = client.models.generate_content(model=MODEL_ID, contents=user_content, config=pairwise_config)
    scores = parse_judge_json(response.text)
    overall = sum((scores.get(c) or 0) for c in CRITERIA) / len(CRITERIA)
    return {"scores": scores, "overall_similarity": overall}


def main(overwrite):
    client = genai.Client(api_key=API_KEY)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    setting_names = [s["name"] for s in config.DT_SETTINGS]
    pairs = [(setting_names[i], setting_names[j]) for i in range(len(setting_names)) for j in range(i + 1, len(setting_names))]

    done, skipped = 0, 0
    for ds in config.DATASETS:
        dataset_key = ds["key"]
        answers = {s: load_answer(dataset_key, s) for s in setting_names}
        missing = [s for s, a in answers.items() if a is None]
        if missing:
            print(f"SKIP {dataset_key}: missing primary-only hypotheses for {missing}")
            continue

        ds_out_dir = OUT_DIR / dataset_key
        ds_out_dir.mkdir(parents=True, exist_ok=True)

        for setting_a, setting_b in pairs:
            out_path = ds_out_dir / f"{setting_a}__vs__{setting_b}.json"
            if out_path.exists() and not overwrite:
                skipped += 1
                continue

            print(f"Comparing [{dataset_key}] {setting_a} vs {setting_b} ...", flush=True)
            try:
                result = compare_pair(client, answers[setting_a], answers[setting_b])
            except Exception as e:
                print(f"ERROR: {e}")
                time.sleep(SLEEP_BETWEEN_CALLS)
                continue

            record = {
                "dataset": dataset_key,
                "setting_a": setting_a,
                "setting_b": setting_b,
                "scores": result["scores"],
                "overall_similarity": result["overall_similarity"],
            }
            out_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
            print(f"  -> overall={result['overall_similarity']:.2f}: {out_path}")
            done += 1
            time.sleep(SLEEP_BETWEEN_CALLS)

    print(f"\nDone. {done} pair(s) compared, {skipped} skipped (cached).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    main(args.overwrite)
