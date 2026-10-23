"""Validate 3_answers/selection_provenance against the 144 canonical selected answers.

Checks: (1) the responses are unchanged (SHA-256 of the answer text of every file, canonical_answer_text.sha256, taken before the label correction of the six Gemma files) and the files are the archived ones (canonical_selected_answers.sha256; line endings normalised to LF);
(2) one manifest row per canonical file; (3) selected_index agrees with source_answer_index + 1 (documented exceptions listed);
(4) selection methods and counts; (5) referenced judgment and raw-candidate files exist and, where raw candidates exist, the canonical answer equals the selected candidate;
(6) every row of manual_exceptions.csv refers to a manifest row.
Usage: python validate_selection_provenance.py [--write-hashes]      (run from anywhere; exit code 1 on any failure)
"""
import csv, glob, hashlib, json, os, sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ANS = HERE.parent
fails = []


def check(ok, msg):
    print(("PASS  " if ok else "FAIL  ") + msg)
    if not ok: fails.append(msg)


files = sorted(glob.glob(str(ANS / "*" / "*_selected_answer.json")))
sha = {os.path.relpath(f, ANS).replace("\\", "/"): hashlib.sha256(open(f, "rb").read().replace(b"\r\n", b"\n")).hexdigest() for f in files}
if "--write-hashes" in sys.argv:
    (HERE / "canonical_selected_answers.sha256").write_text("".join(f"{h}  {p}\n" for p, h in sha.items()), encoding="utf-8"); print("hashes written")
ref = dict(line.split("  ", 1)[::-1] for line in (HERE / "canonical_selected_answers.sha256").read_text(encoding="utf-8").splitlines())
ref = {k.strip(): v for k, v in ref.items()}
check(len(files) == 144, f"{len(files)} canonical selected files (expected 144)")
check(set(sha) == set(ref) and all(sha[k] == ref[k] for k in sha), "all canonical files are identical to the archived SHA-256 list")
atxt = dict(line.split("  ", 1)[::-1] for line in (HERE / "canonical_answer_text.sha256").read_text(encoding="utf-8").splitlines())
atxt = {k.strip(): v for k, v in atxt.items()}
cur = {k: hashlib.sha256(json.load(open(ANS / k, encoding="utf-8"))["answer"].encode("utf-8")).hexdigest() for k in sha}
check(set(cur) == set(atxt) and all(cur[k] == atxt[k] for k in cur), "the answer text of all 144 responses is unchanged (hash of the answer field)")

rows = list(csv.DictReader(open(HERE / "selection_manifest.csv", encoding="utf-8")))
key = lambda m, c: f"{m}/{c}_selected_answer.json"
mk = Counter(key(r["model"], r["case"]) for r in rows)
check(len(rows) == 144 and all(v == 1 for v in mk.values()) and set(mk) == set(sha), "one manifest row per canonical file (144 rows, no duplicates, none missing)")

idx_bad, ex_idx, meth_bad, raw_bad, judg_bad = [], [], [], [], []
for r in rows:
    can = json.load(open(ANS / key(r["model"], r["case"]), encoding="utf-8"))
    if r["selected_index"] == "unrecorded":
        ex_idx.append(r["model"] + "/" + r["case"])
        if can["source_answer_index"] != "unrecorded": idx_bad.append(key(r["model"], r["case"]))
    elif int(r["selected_index"]) != int(can["source_answer_index"]) + 1: idx_bad.append(key(r["model"], r["case"]))
    if r["selection_method"] != can["selection_method"]: meth_bad.append(key(r["model"], r["case"]))
    if r["selector_judgment_file"] != "not applicable" and not (HERE / r["selector_judgment_file"]).exists(): judg_bad.append(r["selector_judgment_file"])
    rp = HERE / "raw_candidates" / r["model"] / f"{r['case']}_result.json"
    if r["raw_candidates_available"] == "yes":
        if not rp.exists(): raw_bad.append(str(rp)); continue
        answers = json.load(open(rp, encoding="utf-8"))["answers"]
        if int(r["candidate_count"]) != len(answers) or answers[int(r["selected_index"]) - 1] != can["answer"]: raw_bad.append(str(rp))
    elif rp.exists(): raw_bad.append("flag 'no' but file exists: " + str(rp))
check(not idx_bad, f"selected_index = source_answer_index + 1 for all rows with a recorded index ({len(rows) - len(ex_idx)} rows); 'unrecorded' in both files otherwise")
check(len(ex_idx) == 6 and all("gemma-4-12B-it" in e for e in ex_idx), f"rows with selected_index 'unrecorded' are the 6 documented Gemma 4 12B rows ({len(ex_idx)})")
check(not meth_bad, "selection_method in the manifest equals the canonical file")
check(Counter(r["selection_method"] for r in rows) == Counter({"single_sample_no_consistency_check": 78, "consistent_pair": 57, "embedding_medoid_of_5": 6, "random_no_consensus": 2, "manual_selection_from_archive": 1}),
      "method counts 78 single / 57 consistent-pair / 6 embedding-medoid / 2 random / 1 manual")
check(not judg_bad, "every referenced selector judgment file exists")
check(not raw_bad, f"raw candidate files exist when flagged, candidate_count matches, canonical answer = selected candidate ({sum(r['raw_candidates_available'] == 'yes' for r in rows)} cases)")
exc = list(csv.DictReader(open(HERE / "manual_exceptions.csv", encoding="utf-8")))
check(all(key(e["model"], e["case"]) in mk for e in exc), f"all {len(exc)} manual_exceptions rows refer to a manifest row")
check(sum(e["exception_type"] == "random_no_consensus" for e in exc) == 2 and any(e["exception_type"] == "manual_selection_from_archive" for e in exc), "the 2 random selections and the Gemini Fxyd6 manual selection are listed")
print("\nRESULT:", "ALL CHECKS PASSED" if not fails else f"{len(fails)} CHECK(S) FAILED")
sys.exit(1 if fails else 0)
