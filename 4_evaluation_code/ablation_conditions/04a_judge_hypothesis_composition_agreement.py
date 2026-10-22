"""Result 4, judging step - HYPOTHESIS COMPOSITION AGREEMENT: how much of each model's molecular (marker-level) explanation is shared by the other models.

Score (0-10, integer) = agreement across the 4 models in the specific molecular markers used to construct the primary hypothesis of one prompt. The judge inventories every
marker that plays an explanatory role in each answer, compares the inventories, estimates what share of each answer's content is represented in the others and applies the
rubric's ceilings (e.g. maximum 4 if more than half of the contributions differ, maximum 6 if about a quarter or more differ). Different genes with similar functions do not
count as matches; matching names with different explanations earn less. Rubric: code/evaluation/hypothesis_composition_agreement_rubric.txt (the text as written; its empty OUTPUT
block is replaced by a short one: ONE-sentence rationale, then the score).

Case    = one prompt (condition x dataset): the 4 picked answers (3_answers/ablation_conditions/selected_answers), anonymized "ANSWER A..D", shown together in a random order (seeded per prompt+pass,
          different in every pass), JSON footer removed, no generation prompt, no model names. 2 conditions x 3 datasets = 6 prompts x 3 passes = 18 judge calls.
Judge   = gpt-5.6-sol (same client / retry schedule as the other judges, no temperature set). Case score = mean of the 3 passes (null scores are excluded).
Output  = 4_hypothesis_composition_agreement/data/<condition>/<dataset>_pass<k>.json (existing files are skipped), all_agreement.json, agreement_scores.csv, justifications.csv, agreement_meta.json.

Usage:  python 04a_judge_hypothesis_composition_agreement.py --dry-run       # no API call: orders, rubric hash
        python 04a_judge_hypothesis_composition_agreement.py [--workers 6] [--overwrite]      (needs OPENAI_API_KEY)
"""
import argparse
import csv
import hashlib
import json
import statistics as st
import sys
import time
from concurrent.futures import ThreadPoolExecutor

from common import CONDS, DATASETS, HERE, LABELS, MODELS, ROOT, load_case, parse_json, shuffled, user_message
import judge_core as jc  # noqa: E402  (code/evaluation/, put on the path by common.py)

RUBRIC_FILE = HERE / "evaluation" / "hypothesis_composition_agreement_rubric.txt"
OUT = ROOT / "4_hypothesis_composition_agreement" / "data"
KEY = "cross_model_agreement_score"
OUTPUT_BLOCK = ("Return valid JSON with exactly these keys, in this order, and nothing else:\n"
                "{\n"
                '"rationale": "<ONE sentence: why this score, i.e. roughly what share of the molecular content is shared and which ceiling, if any, applies>",\n'
                '"cross_model_agreement_score": <ONE integer from 0 to 10>\n'
                "}\n"
                'If the score is null, return instead: {"cross_model_agreement_score": null, "null_reason": "<one sentence>"}')
RUBRIC = RUBRIC_FILE.read_text(encoding="utf-8").strip().replace("{coverage_output}", OUTPUT_BLOCK)


def judge_group(client, order):
    messages = [{"role": "system", "content": RUBRIC}, {"role": "user", "content": user_message(order)}]
    last = None
    for delay in [0, 20, 60, 120]:
        if delay:
            time.sleep(delay)
        try:
            resp = client.chat.completions.create(model=jc.je.JUDGE_MODEL, messages=messages)
            res = parse_json(resp.choices[0].message.content or "")
            score = res[KEY]
            if isinstance(score, float) and score.is_integer():
                score = int(score)
            if not (score is None or (isinstance(score, int) and 0 <= score <= 10)):
                raise ValueError(f"score is neither null nor an integer 0-10: {score!r}")
            return {**res, KEY: score}
        except Exception as e:  # noqa: BLE001  (rate limits, 5xx, malformed JSON -> retry)
            last = e
    raise last


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--passes", type=int, default=3)
    ap.add_argument("--seed", type=int, default=20)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    cases = {(c, d): load_case(MODELS, c, d) for c in CONDS for d in DATASETS}
    tasks = [(c, d, k) for k in range(1, args.passes + 1) for c in CONDS for d in DATASETS]
    sha = hashlib.sha256(RUBRIC.encode()).hexdigest()

    if args.dry_run:
        print(f"rubric sha256 {sha}\n{len(tasks)} judge calls (NOT made); judge {jc.je.JUDGE_MODEL}")
        for c, d, k in tasks[:6]:
            print(f"  {c:20s} {d:22s} pass{k}: " + " > ".join(f"{m.split('/')[-1].split('-')[0]}(run{r})" for m, r, _ in shuffled(cases[(c, d)], args.seed, c, d, k)))
        return

    client = jc.je.load_client("OPENAI_API_KEY")

    def work(t):
        c, d, k = t
        out = OUT / c / f"{d}_pass{k}.json"
        if out.exists() and not args.overwrite:
            return "skip"
        order = shuffled(cases[(c, d)], args.seed, c, d, k)
        try:
            res = judge_group(client, order)
        except Exception as e:  # noqa: BLE001
            print(f"  ERROR {c}/{d}/pass{k}: {type(e).__name__}: {str(e)[:160]}", flush=True)
            return "error"
        rec = {"condition": c, "dataset": d, "pass": k, "seed": args.seed, "order": [{"label": LABELS[i], "model": m, "picked_run": r} for i, (m, r, _) in enumerate(order)],
               "judge": res, "score": res[KEY]}
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rec, indent=2), encoding="utf-8")
        print(f"  judged {c} {d} pass{k}: {res[KEY]}/10", flush=True)
        return "ok"

    with ThreadPoolExecutor(args.workers) as ex:
        results = list(ex.map(work, tasks))
    print({s: results.count(s) for s in set(results)})
    recs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(OUT.glob("*/*_pass*.json"))]
    (OUT / "all_agreement.json").write_text(json.dumps(recs, indent=2), encoding="utf-8")
    with open(OUT / "agreement_scores.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["condition", "dataset", "n_passes_scored", "mean_score", "sd_across_passes", "scores_per_pass"])
        for c in CONDS:
            for d in DATASETS:
                allp = [r["score"] for r in recs if r["condition"] == c and r["dataset"] == d]
                s = [x for x in allp if x is not None]
                w.writerow([c, d, len(s), round(st.mean(s), 3) if s else "", round(st.stdev(s), 3) if len(s) > 1 else "", allp])
    with open(OUT / "justifications.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["condition", "dataset", "pass", "score", "rationale"])
        for r in recs:
            w.writerow([r["condition"], r["dataset"], r["pass"], r["score"], r["judge"].get("rationale", r["judge"].get("null_reason", ""))])
    (OUT / "agreement_meta.json").write_text(json.dumps({"judge_model": jc.je.JUDGE_MODEL, "rubric": "hypothesis_composition_agreement", "rubric_file": RUBRIC_FILE.name,
                                                         "rubric_sha256": sha, "passes": args.passes, "seed": args.seed, "models": MODELS, "conditions": CONDS,
                                                         "datasets": DATASETS, "n_records": len(recs)}, indent=2), encoding="utf-8")
    print(f"{len(recs)} judge records -> {OUT}")


if __name__ == "__main__":
    main()
