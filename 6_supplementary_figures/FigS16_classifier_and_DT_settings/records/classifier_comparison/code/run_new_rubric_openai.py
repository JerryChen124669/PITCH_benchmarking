"""
Same adapted rubrics as run_new_rubric.py (quality + pairwise, classifier-type
axis: LogReg vs DT vs RF), run through OpenAI's gpt-5.6-sol instead of Gemini
3.1 Pro -- a genuinely independent model family, for cross-checking the
Gemini-graded results (which is Gemini judging Gemini's own hypotheses,
blind to classifier identity but not to model family).

Imports the rubric text and data-loading helpers directly from
run_new_rubric.py so the two judges are graded on the EXACT same prompts --
only the API client changes.

Requires OPENAI_API_KEY in the environment (not hardcoded here).

Usage:
    python run_new_rubric_openai.py
    python run_new_rubric_openai.py --skip-quality
    python run_new_rubric_openai.py --skip-pairwise
"""
import argparse
import itertools
import json
import os
import random
import time
from pathlib import Path

from openai import OpenAI

from run_new_rubric import (
    DATASETS, CLASSIFIERS, QUALITY_SYSTEM_PROMPT, PAIRWISE_SYSTEM_PROMPT,
    PAIRWISE_CRITERIA, load_hypothesis, load_prompt, extract_json,
    compute_universal_score, strip_json_footer,
)

JUDGE_MODEL = "gpt-5.6-sol"
OUT_DIR = Path(__file__).resolve().parent / "new_rubric_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_client():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("ERROR: OPENAI_API_KEY not set in environment.")
    return OpenAI(api_key=api_key, timeout=300.0)


def call_judge(client, system_prompt, user_content):
    response = client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
    )
    return response.choices[0].message.content or ""


def run_quality(client):
    rows = []
    for dataset in DATASETS:
        for clf in CLASSIFIERS:
            prompt_text = load_prompt(dataset, clf)
            answer_text = load_hypothesis(dataset, clf)
            user_content = f"--- INPUT 1: EVIDENCE/PROMPT GIVEN TO THE MODEL ---\n{prompt_text}\n\n--- INPUT 2: THE MODEL'S SUBMISSION ---\n{answer_text}"
            print(f"[quality/openai] {dataset} / {clf} ...", end=" ", flush=True)
            t0 = time.time()
            try:
                raw_text = call_judge(client, QUALITY_SYSTEM_PROMPT, user_content)
                scores = extract_json(raw_text)
            except Exception as e:
                print(f"ERROR {e}")
                continue
            if scores is None:
                print("PARSE ERROR")
                continue
            universal = compute_universal_score(scores)
            print(f"universal={universal:.1f} ({time.time()-t0:.1f}s)")
            rows.append({"dataset": dataset, "classifier": clf, **scores, "universal_score_of45": universal})
            time.sleep(1)
    out_path = OUT_DIR / "quality_scores_openai.json"
    out_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print("Saved ->", out_path)
    return rows


def run_pairwise(client):
    rows = []
    for dataset in DATASETS:
        answers = {clf: strip_json_footer(load_hypothesis(dataset, clf)) for clf in CLASSIFIERS}
        for clf_a, clf_b in itertools.combinations(CLASSIFIERS, 2):
            pair = [answers[clf_a], answers[clf_b]]
            order = [0, 1]
            random.shuffle(order)
            user_content = f"--- RESPONSE A ---\n{pair[order[0]]}\n\n--- RESPONSE B ---\n{pair[order[1]]}"
            print(f"[pairwise/openai] {dataset} / {clf_a} vs {clf_b} ...", end=" ", flush=True)
            t0 = time.time()
            try:
                raw_text = call_judge(client, PAIRWISE_SYSTEM_PROMPT, user_content)
                scores = extract_json(raw_text)
            except Exception as e:
                print(f"ERROR {e}")
                continue
            if scores is None:
                print("PARSE ERROR")
                continue
            overall = sum((scores.get(c) or 0) for c in PAIRWISE_CRITERIA) / len(PAIRWISE_CRITERIA)
            print(f"overall={overall:.1f} ({time.time()-t0:.1f}s)")
            rows.append({"dataset": dataset, "classifier_a": clf_a, "classifier_b": clf_b,
                         **scores, "overall_similarity": overall})
            time.sleep(1)
    out_path = OUT_DIR / "pairwise_scores_openai.json"
    out_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print("Saved ->", out_path)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-quality", action="store_true")
    parser.add_argument("--skip-pairwise", action="store_true")
    args = parser.parse_args()

    client = load_client()

    if not args.skip_quality:
        print("=== PART 1: Quality rubric (12 calls, OpenAI) ===")
        run_quality(client)
    if not args.skip_pairwise:
        print("\n=== PART 2: Pairwise similarity rubric (12 calls, OpenAI) ===")
        run_pairwise(client)


if __name__ == "__main__":
    main()
