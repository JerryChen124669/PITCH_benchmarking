#!/usr/bin/env python
"""Check a run: every row valid and nothing cut off at max_tokens. Exit 0 if clean.

    python check_run.py runs/<run>
"""
import json
import sys


def main(run):
    rows = [json.loads(l) for l in open(f"{run}/raw_responses.jsonl")]
    bad = [r for r in rows if r["format_validation"] != "passed"]
    cut = [r for r in rows if r["attempts_log"][-1]["finish_reason"] == "length"]
    tok = [r["attempts_log"][-1]["output_tokens"] for r in rows]
    print(f"{run}: {len(rows)} rows, {len(rows) - len(bad)} valid, {len(cut)} hit max_tokens; "
          f"output tokens max {max(tok)} mean {sum(tok) // len(tok)}")
    for r in bad:
        print(f"  INVALID {r['key']}: {r['attempts_log'][-1]['reason']}")
    for r in cut:
        print(f"  TRUNCATED {r['key']}: {r['attempts_log'][-1]['output_tokens']} tokens")
    return 1 if (bad or cut) else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
