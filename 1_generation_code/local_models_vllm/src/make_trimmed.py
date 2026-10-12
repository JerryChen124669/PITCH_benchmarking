#!/usr/bin/env python
"""Trim a run's raw_responses.jsonl to [key, id, reasoning, thinking, answer].

    python make_trimmed.py runs/<run> <output-name>      # -> runs/<run>/<output-name>.jsonl

Rows that failed validation are kept, with thinking/answer null, so the file covers every prompt.
"""
import json
import sys

FIELDS = ["key", "id", "reasoning", "thinking", "answer"]
# plain-format runs (the 6 real prompts) have no JSON thinking field, so `thinking` there is a
# copy of `reasoning` -- one of the two is dropped.
PLAIN_FIELDS = [f for f in FIELDS if f != "reasoning"]


def main(run, name):
    out = f"{run}/{name}.jsonl"
    rows = [json.loads(l) for l in open(f"{run}/raw_responses.jsonl")]
    # `reasoning` is dropped when it carries nothing of its own: plain-format thinking runs copy it
    # into `thinking`, and models without a thinking mode leave it null.
    dup = all(r["reasoning"] == r["thinking"] for r in rows)
    empty = all(not r["reasoning"] for r in rows)
    fields = PLAIN_FIELDS if (dup or empty) else FIELDS
    n = ok = 0
    with open(out, "w") as dst:
        for r in rows:
            n += 1
            ok += r["format_validation"] == "passed"
            dst.write(json.dumps({k: r[k] for k in fields}, ensure_ascii=False) + "\n")
    print(f"{out}: {n} rows ({ok} passed, {n - ok} with null thinking/answer)")


if __name__ == "__main__":
    main(*sys.argv[1:3])
