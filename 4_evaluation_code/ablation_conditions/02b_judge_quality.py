"""Result 2, judging step - 5-pillar blind quality judge on the single picked answers (3_answers/ablation_conditions/selected_answers/), PITCH and MLP + IG, 3 datasets.

Judge     : gpt-5.6-sol with the UNMODIFIED 7-pillar rubric (code/evaluation/judge_evaluation.py); its sha256 is asserted against the hash of the rubric used
            in the earlier judging. Pillars shown in the radar: Feature Comprehensiveness, Biological Plausibility, Specificity (Hypothesis Integration &
            Specificity), Grounded Mechanistic Synthesis, Literature Grounding.
Exam text : the judge is shown the prompt each model actually received (2_prompts/ablation_conditions/...) followed by the model's answer (JSON footer removed); the model
            name is never shown.
Preamble  : the "five-criterion, no Zone/Logic" override, with Feature Comprehensiveness defined as "does the answer explain every feature it actually used
            in its hypothesis" (number of features neither rewarded nor penalized).
Passes    : run this script once per pass (same answers, same rubric); the radar plots use the mean of the two passes (02c_plot_quality_radars.py).

Usage:   python 02b_judge_quality.py --out-dir ../../5_evaluation_results/ablation_conditions/2_quality_radar/scorecards/pass1 [--workers 6] [--overwrite]
Needs OPENAI_API_KEY in the environment. Existing scorecards are skipped (resumable).
"""
import argparse
import hashlib
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from common import ALL_CONDS, CONDS, DATASETS, INPUTS, MODELS, slug
import judge_core as jc  # noqa: E402  (code/evaluation/, put on the path by common.py)

EARLIER_RUBRIC_SHA256 = "522c88817cdc85a4a832bdbd11cfb0c4f58a1b1c0228eb9b018676e5d0cd2cff"
_OLD_FC = ("For Feature Comprehensiveness, the required items are the target, the focus feature, and the (up to) four "
           "contextual features the submission itself names as its main contextual features; judge role assignment for those.")
_NEW_FC = ("For Feature Comprehensiveness, the required items are every feature (gene or measurement, excluding the target itself) "
           "that the submission's hypothesis relies on or invokes as part of its mechanism; list them before scoring. Features that are "
           "only mentioned in passing, or appear in a list without being used in the hypothesis, are not required items. Judge whether "
           "each required item is explained, meaning it is given an item-specific biological or contextual role tied to the target "
           "association, using the 0-2 item credits and the formula as specified. Do not reward or penalize the NUMBER of features used.")
assert jc.NO_TREE_PREAMBLE.count(_OLD_FC) == 1
PREAMBLE = jc.NO_TREE_PREAMBLE.replace(_OLD_FC, _NEW_FC)


def build_exam(ds, cond, variant=None):
    text = (INPUTS / "prompts" / ds / f"{ds}_{cond}_prompt.txt").read_text(encoding="utf-8")
    if cond == "cond2_raw_data":
        import raw_prompt_versions as rpv      # rebuilds the exact measurement-only wording the answer was generated under
        text = rpv.reconstruct(text, variant)
        head, rest = text.split("--- RAW DATA", 1)
        table, task = rest.split("--- TASK ---", 1)
        tl = table.splitlines()                # the ~500K-token table is cut to its header + first row for judging
        text = (head + "--- RAW DATA" + "\n".join(tl[:3]) + f"\n[... {len(tl) - 3} more rows of the raw cells x features table omitted for judging ...]\n\n--- TASK ---" + task)
    m = re.match(r"\s*--- SYSTEM PROMPT ---\s*(.*)", text, re.DOTALL)
    return m.group(1) if m else text


def judge_one(client, ds, cond, answer, variant=None):
    user = f"--- INPUT 1: WHAT THE STUDENT WAS GIVEN ---\n{build_exam(ds, cond, variant)}\n\n--- INPUT 2: THE STUDENT'S SUBMISSION ---\n{answer}\n"
    last = None
    for delay in [0, 20, 60, 120]:
        if delay:
            time.sleep(delay)
        try:
            resp = client.chat.completions.create(model=jc.je.JUDGE_MODEL, messages=[
                {"role": "system", "content": PREAMBLE + jc.je.JUDGE_SYSTEM_PROMPT}, {"role": "user", "content": user}])
            raw = resp.choices[0].message.content or ""
            m = re.search(r"```json\s*(.*?)\s*```", raw, re.DOTALL)
            return json.loads(m.group(1) if m else raw.strip().strip("`"))
        except Exception as e:  # noqa: BLE001  (rate limits, 5xx, malformed JSON -> retry)
            last = e
    raise last


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--conds", nargs="+", default=CONDS, choices=ALL_CONDS, help="conditions to judge (default: PITCH and MLP + IG)")
    args = ap.parse_args()
    out_root = Path(args.out_dir)
    sha = hashlib.sha256(jc.je.JUDGE_SYSTEM_PROMPT.encode()).hexdigest()
    assert sha == EARLIER_RUBRIC_SHA256, "rubric text differs from the earlier judging"
    print(f"judge model: {jc.je.JUDGE_MODEL}; rubric sha256 {sha[:12]}... matches the earlier judging")
    manifest = json.loads((INPUTS / "single_answers" / "selection_manifest.json").read_text(encoding="utf-8"))
    cells = [c for c in manifest["cells"] if c["picked_run"] is not None and c["condition"] in args.conds]
    client = jc.je.load_client("OPENAI_API_KEY")

    def work(c):
        m, ds, cond, k = c["model"], c["dataset"], c["condition"], c["picked_run"]
        out = out_root / cond / slug(m) / f"{ds}_{cond}_scorecard.json"
        if out.exists() and not args.overwrite:
            return "skip"
        rec = json.loads((INPUTS / "single_answers" / cond / slug(m) / f"{ds}_{cond}_picked_run{k}.json").read_text(encoding="utf-8"))
        try:
            scores = judge_one(client, ds, cond, jc.je_strip(rec["final_answer"]), rec.get("prompt_variant", "standard"))   # files without the field were generated under the 'standard' raw wording
        except Exception as e:  # noqa: BLE001
            print(f"  ERROR {m} {ds}/{cond}: {type(e).__name__}: {str(e)[:160]}", flush=True)
            return "error"
        row = {"model": m, "model_slug": slug(m), "dataset": ds, "condition": cond, "picked_run": k, "preamble": "v2", "scores": scores,
               "final_score_5pillar": jc.final5(scores)}
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(row, indent=2), encoding="utf-8")
        print(f"  judged {m} {ds}/{cond} (run {k}) -> {row['final_score_5pillar']:.1f}/45", flush=True)
        return "ok"

    with ThreadPoolExecutor(args.workers) as ex:
        results = list(ex.map(work, cells))
    print({s: results.count(s) for s in set(results)})
    rows = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(out_root.glob("*/*/*_scorecard.json"))]
    (out_root / "all_scores.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    (out_root / "judge_meta.json").write_text(json.dumps({
        "judge_model": jc.je.JUDGE_MODEL, "rubric_sha256": sha, "matches_earlier_rubric": True, "preamble_mode": "v2", "preamble_text": PREAMBLE,
        "n_scorecards": len(rows), "selection_seed": manifest["seed"]}, indent=2), encoding="utf-8")
    print(f"{len(rows)} scorecards -> {out_root}")


if __name__ == "__main__":
    main()
