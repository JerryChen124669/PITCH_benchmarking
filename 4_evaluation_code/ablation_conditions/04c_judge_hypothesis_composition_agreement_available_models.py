"""Hypothesis composition agreement of the MEASUREMENT-ONLY condition, computed on the models that have a successfully generated (committed) answer for a prompt.

Measurement-only has a committed answer for only some models per prompt (CITE-seq: GLM-5.3, Qwen3.8; CosMx: GLM-5.3, DeepSeek, Qwen3.8; Gouwens: Gemini, GLM-5.3), so the four-model score of
04a cannot be computed for it. The same rubric, anonymisation, random answer order (new order every pass) and judge as in 04a are applied to the answers of exactly those models
(groups of 2, 3 and 2 answers; 3 prompts x 3 passes = 9 judge calls). Scores of groups of different size are not directly comparable with the four-model scores of 04a (fewer models make agreement easier).
--conds also accepts the other two conditions (judged on the same models of each prompt) for a matched comparison; the default is measurement-only only.
Output: 4_hypothesis_composition_agreement/data_available_models/<condition>/<dataset>_pass<k>.json, all_agreement_available_models.json, agreement_scores_available_models.csv, meta.json
Usage:  python 04c_judge_hypothesis_composition_agreement_available_models.py --dry-run        (needs OPENAI_API_KEY otherwise)
"""
import argparse
import csv
import importlib.util
import json
import statistics as st
import sys
from concurrent.futures import ThreadPoolExecutor

from common import ALL_CONDS, DATASETS, HERE, INPUTS, LABELS, MODELS, ROOT, load_case, shuffled

spec = importlib.util.spec_from_file_location("agreement04a", HERE / "04a_judge_hypothesis_composition_agreement.py")
a04 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a04)          # reuse the rubric, the judge call and its parsing from 04a (nothing is re-implemented)
import judge_core as jc  # noqa: E402

OUT = ROOT / "4_hypothesis_composition_agreement" / "data_available_models"


def groups():
    """models with a selected (committed) measurement-only answer, per dataset, in the fixed model order"""
    manifest = json.loads((INPUTS / "single_answers" / "selection_manifest.json").read_text(encoding="utf-8"))
    have = {(c["model"], c["dataset"]) for c in manifest["cells"] if c["condition"] == "cond2_raw_data" and c["picked_run"] is not None}
    return {d: [m for m in MODELS if (m, d) in have] for d in DATASETS}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--conds", nargs="+", default=["cond2_raw_data"], choices=ALL_CONDS, help="conditions to judge (default: measurement-only)")
    ap.add_argument("--passes", type=int, default=3)
    ap.add_argument("--seed", type=int, default=20)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    grp = groups()
    cases = {(c, d): load_case(grp[d], c, d) for c in args.conds for d in DATASETS}
    tasks = [(c, d, k) for k in range(1, args.passes + 1) for c in args.conds for d in DATASETS]
    if args.dry_run:
        print(f"conditions {args.conds} | rubric sha256 {a04.hashlib.sha256(a04.RUBRIC.encode()).hexdigest()[:16]}... | {len(tasks)} judge calls (NOT made)")
        for d in DATASETS:
            print(f"  {d:22s} group = {[m.split('/')[-1] for m in grp[d]]}")
        for c, d, k in tasks[:9]:
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
            res = a04.judge_group(client, order)
        except Exception as e:  # noqa: BLE001
            print(f"  ERROR {c}/{d}/pass{k}: {type(e).__name__}: {str(e)[:160]}", flush=True)
            return "error"
        rec = {"condition": c, "dataset": d, "pass": k, "seed": args.seed, "models": grp[d],
               "order": [{"label": LABELS[i], "model": m, "picked_run": r} for i, (m, r, _) in enumerate(order)], "judge": res, "score": res[a04.KEY]}
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rec, indent=2), encoding="utf-8")
        print(f"  judged {c} {d} pass{k} ({len(order)} answers): {res[a04.KEY]}/10", flush=True)
        return "ok"

    with ThreadPoolExecutor(args.workers) as ex:
        results = list(ex.map(work, tasks))
    print({s: results.count(s) for s in set(results)})
    recs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(OUT.glob("*/*_pass*.json"))]
    (OUT / "all_agreement_available_models.json").write_text(json.dumps(recs, indent=2), encoding="utf-8")
    with open(OUT / "agreement_scores_available_models.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["condition", "dataset", "models_in_group", "n_models", "n_passes_scored", "mean_score", "sd_across_passes", "scores_per_pass"])
        for c in ALL_CONDS:
            for d in DATASETS:
                if not any(r["condition"] == c and r["dataset"] == d for r in recs):
                    continue
                allp = [r["score"] for r in recs if r["condition"] == c and r["dataset"] == d]
                s = [x for x in allp if x is not None]
                w.writerow([c, d, "; ".join(m.split("/")[-1] for m in grp[d]), len(grp[d]), len(s), round(st.mean(s), 3) if s else "", round(st.stdev(s), 3) if len(s) > 1 else "", allp])
    (OUT / "meta.json").write_text(json.dumps({"judge_model": jc.je.JUDGE_MODEL, "rubric_sha256": a04.hashlib.sha256(a04.RUBRIC.encode()).hexdigest(), "passes": args.passes,
                                               "seed": args.seed, "groups": grp, "n_records": len(recs)}, indent=2), encoding="utf-8")
    print(f"{len(recs)} judge records -> {OUT}")


if __name__ == "__main__":
    main()
