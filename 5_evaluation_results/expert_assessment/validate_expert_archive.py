"""Validate the expert-assessment archive.

Checks: (1) every rating refers to a valid anonymous assessor, dataset, hypothesis label or pair, and criterion, with an integer 1-5 score, no duplicates, and the full set of 36 ratings per assessor;
(2) every hypothesis label maps to a model; the shown-text files exist and match the recorded hashes; (3) aggregate_expert_ratings.py reproduces figure_7E_plotting_input.csv;
(4) no identifying information: only the expected columns, no e-mail addresses, timestamps, URLs or personal paths in any file of this folder, no raw survey export.
Usage: python validate_expert_archive.py      (exit code 1 on any failure)
"""
import hashlib, re, subprocess, sys
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
fails = []


def check(ok, msg):
    print(("PASS  " if ok else "FAIL  ") + msg)
    if not ok: fails.append(msg)


r = pd.read_csv(HERE / "deidentified_ratings.csv"); m = pd.read_csv(HERE / "hypothesis_label_mapping.csv")
check(list(r.columns) == ["assessor_id", "dataset", "item_type", "blind_label_or_pair", "criterion", "workbook_criterion_name", "score", "record_source"], "ratings file has only the expected (de-identified) columns")
check(set(r.assessor_id) == {f"Assessor_{i}" for i in range(1, 6)}, "assessor ids are Assessor_1 ... Assessor_5")
check(set(r.dataset) == {"pERK", "GABA", "Ki-67"}, "datasets are pERK, GABA, Ki-67")
valid_h = set(zip(m.dataset, m.blind_label)); valid_p = {(d, f"{a}-{b}") for d in set(m.dataset) for a, b in (("A", "B"), ("A", "C"), ("B", "C"))}
bad = [row for row in r.itertuples() if (row.dataset, row.blind_label_or_pair) not in (valid_h if row.item_type == "hypothesis" else valid_p)]
check(not bad, "every rating refers to a valid hypothesis label or hypothesis pair of its dataset")
crit_h = {"Core biological plausibility", "Mechanistic specificity and completeness", "Accuracy of specific molecular claims"}
check(all(c in crit_h for c in r[r.item_type == "hypothesis"].criterion) and set(r[r.item_type == "pairwise"].criterion) == {"Mechanistic concordance"}, "criteria are the three hypothesis pillars and the pairwise concordance")
check(r.score.between(1, 5).all() and (r.score == r.score.round()).all(), "all scores are integers from 1 to 5")
check(not r.duplicated(["assessor_id", "dataset", "item_type", "blind_label_or_pair", "criterion"]).any() and (r.groupby("assessor_id").size() == 36).all() and len(r) == 180,
      "no duplicate ratings; 36 ratings per assessor (180 in total)")
check(set(m.model) == {"Gemini-3.1-Pro", "Qwen3.8-2.4T", "Qwen3.5-9B"} and len(m) == 9 and m.groupby("dataset").model.nunique().eq(3).all(), "each dataset's 3 blind labels map to 3 different models")
ok = True
for row in m.itertuples():
    f = HERE / row.shown_text_file; ok &= f.exists()
check(ok and set(m.shown_text_file) == {p for p in m.shown_text_file}, "the shown-hypothesis files exist")
text = {}
for f in (HERE / "anonymized_hypotheses").glob("*.md"):
    t = f.read_text(encoding="utf-8")
    for lab, body in re.findall(r"## Hypothesis ([ABC])\n\n(.*?)(?=\n\n## Hypothesis|\Z)", t, re.S):
        text[(f.name, lab)] = hashlib.sha256(body.strip().encode("utf-8")).hexdigest()[:16]
check(all(text[(Path(row.shown_text_file).name, row.blind_label)] == row.shown_text_sha256 for row in m.itertuples()), "shown texts match the recorded SHA-256 prefixes")
res = subprocess.run([sys.executable, str(HERE / "aggregate_expert_ratings.py")], capture_output=True, text=True)
print("      " + res.stdout.strip())
check(res.returncode == 0, "aggregate_expert_ratings.py reproduces figure_7E_plotting_input.csv")

pat = {"e-mail address": r"[\w.+-]+@[\w-]+\.[\w.-]+", "timestamp/date": r"\b20\d\d[/-]\d\d[/-]\d\d\b", "URL": r"https?://", "personal path": r"(?i)twjer|Jerry Chen|C:[\\/]Users|/Users/"}
hits = []
for f in HERE.rglob("*"):
    if f.is_file() and f.suffix in {".csv", ".md", ".py", ".txt"} and f.name != "validate_expert_archive.py":
        t = f.read_text(encoding="utf-8", errors="ignore")
        hits += [f"{f.name}: {k}" for k, p in pat.items() if re.search(p, t)]
check(not hits, "no e-mail addresses, timestamps, URLs or personal paths in any file" + (f" ({hits})" if hits else ""))
check(not list(HERE.rglob("*Responses*")) and not list(HERE.rglob("*.xlsx")), "no raw survey export or workbook is included")
print("\nRESULT:", "ALL CHECKS PASSED" if not fails else f"{len(fails)} CHECK(S) FAILED")
sys.exit(1 if fails else 0)
