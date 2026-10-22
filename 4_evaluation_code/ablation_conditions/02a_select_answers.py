"""Selects the answers used by the evaluations (results 2, 3 and 4).

Step 1  3_answers/ablation_conditions/three_per_prompt/        exactly 3 answers per (model, condition, dataset): committed ("stated") answers first, then tentative, then declined,
                                    ties by lowest run number (so Gemini's many MLP + IG retries reduce to its best 3).
Step 2  3_answers/ablation_conditions/selected_answers/      ONE answer per (model, condition, dataset) for the PITCH, MLP + IG and measurement-only conditions, drawn at random (seeded per cell:
                                    seed|model|dataset|condition, so adding or removing other cells never changes a pick) from the committed answers of step 1.
                                    A cell without any committed answer is reported and left empty.
                                    selection_manifest.json lists, for every cell, the committed runs, the pick and the number of attempts.
Usage: python 02a_select_answers.py [--seed 20]
"""
import argparse
import csv
import json
import random
import shutil

from common import ALL_CONDS, CONDS, DATASETS, INPUTS, MODEL_SHORT, MODELS, slug

RANK = {"stated": 0, "tentative": 1, "declined": 2}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20)
    args = ap.parse_args()
    verdict = {(r["model"], r["condition"], r["dataset"], int(r["run"])): r["verdict"] for r in csv.DictReader(open(INPUTS / "answer_screening.csv", encoding="utf-8"))}

    # ---- step 1: three answers per prompt and model
    per3 = INPUTS / "answers_3per"
    if per3.exists():
        shutil.rmtree(per3)
    cells = {}
    for m in MODELS:
        for f in sorted((INPUTS / "answers" / slug(m)).glob("*.json")):
            r = json.loads(f.read_text(encoding="utf-8"))
            if r["dataset"] in DATASETS:
                cells.setdefault((m, r["condition"], r["dataset"]), []).append((int(r["run"]), f, bool((r.get("final_answer") or "").strip())))
    rows = []
    for (m, c, d), lst in sorted(cells.items()):
        ok = [(k, f) for k, f, has in lst if has]
        key = lambda kf: (RANK.get(verdict.get((MODEL_SHORT[m], c, d, kf[0])), 3), kf[0])
        for k, f in sorted(sorted(ok, key=key)[:3]):
            (per3 / slug(m)).mkdir(parents=True, exist_ok=True)
            shutil.copyfile(f, per3 / slug(m) / f.name)
            rows.append({"model": m, "condition": c, "dataset": d, "run": k, "verdict": verdict.get((MODEL_SHORT[m], c, d, k), "unscreened"),
                         "n_attempts_in_cell": len(lst)})
    with open(per3 / "manifest.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # ---- step 2: one random committed answer per cell (PITCH and MLP + IG)
    out_root = INPUTS / "single_answers"
    if out_root.exists():
        shutil.rmtree(out_root)
    manifest = {"seed": args.seed, "pool": "answers_3per, verdict == stated", "conditions": ALL_CONDS, "cells": []}
    for m in MODELS:
        for c in ALL_CONDS:        # PITCH, MLP + IG and (where a committed answer exists) measurement-only
            for d in DATASETS:
                sel = [r for r in rows if r["model"] == m and r["condition"] == c and r["dataset"] == d]
                good = sorted(r["run"] for r in sel if r["verdict"] == "stated")
                cell = {"model": m, "condition": c, "dataset": d, "committed_runs": good, "attempts_in_cell": sel[0]["n_attempts_in_cell"] if sel else 0, "picked_run": None}
                if good:
                    k = random.Random(f"{args.seed}|{m}|{d}|{c}").choice(good)
                    dst = out_root / c / slug(m) / f"{d}_{c}_picked_run{k}.json"
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(per3 / slug(m) / f"{d}_{c}_run{k}.json", dst)
                    cell["picked_run"] = k
                else:
                    print(f"NO committed answer for {m} {d}/{c}")
                manifest["cells"].append(cell)
    (out_root / "selection_manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(f"answers_3per: {len(rows)} files; single_answers: {sum(c['picked_run'] is not None for c in manifest['cells'])}/{len(manifest['cells'])} cells picked (seed {args.seed})")


if __name__ == "__main__":
    main()
